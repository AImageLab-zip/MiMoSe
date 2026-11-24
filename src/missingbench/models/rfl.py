from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from missingbench.models.abstract_model import AbstractModel

basic_dims = 8
transformer_basic_dims = 512
mlp_dim = 4096
num_heads = 8
depth = 1
num_modals = 4
patch_size = 8
input_patch_size = 128

# External pipeline uses [t1c, t1n, t2f, t2w]; RFL internals expect
# [flair, t1ce, t1, t2] (same convention as DCSeg/RFNet/M2FTrans/MMMViT).
DATASET_MODALITY_ORDER = (2, 0, 1, 3)


def normalization(planes: int, norm: str = "in") -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm3d(planes)
    if norm == "gn":
        return nn.GroupNorm(4, planes)
    if norm == "in":
        return nn.InstanceNorm3d(planes)
    raise ValueError(f"normalization type {norm} is not supported")


class GeneralConv3dPrenorm(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pad_type: str = "reflect",
        norm: str = "in",
        act_type: str = "lrelu",
        relufactor: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=k_size,
            stride=stride,
            padding=padding,
            padding_mode=pad_type,
            bias=True,
        )
        self.norm = normalization(in_ch, norm=norm)
        if act_type == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif act_type == "lrelu":
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)
        else:
            raise ValueError(f"activation type {act_type} is not supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        return x


class FusionPrenorm(nn.Module):
    """Zero out missing modalities, concatenate, and fuse with plain convs.

    Legacy's ``fusion_prenorm`` names its channel-multiplier constructor
    argument ``num_cls`` and passes the true segmentation-class count into
    it, but the module only ever concatenates the 4 stacked modality
    streams — it should always multiply by ``num_modals`` (4). This
    coincidentally worked for BraTS variants with 4 classes but would
    shape-mismatch for BraTS25's 5 classes, matching the same bug found and
    fixed in SRMNet/MMMViT/IMS2Trans. This port renames the parameter and
    always fixes it at 4.
    """

    def __init__(self, in_channel: int = 64, num_modals: int = 4) -> None:
        super().__init__()
        self.fusion_layer = nn.Sequential(
            GeneralConv3dPrenorm(in_channel * num_modals, in_channel, k_size=1, padding=0, stride=1),
            GeneralConv3dPrenorm(in_channel, in_channel, k_size=3, padding=1, stride=1),
            GeneralConv3dPrenorm(in_channel, in_channel, k_size=1, padding=0, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion_layer(x)


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.e1_c1 = nn.Conv3d(1, basic_dims, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=True)
        self.e1_c2 = GeneralConv3dPrenorm(basic_dims, basic_dims)
        self.e1_c3 = GeneralConv3dPrenorm(basic_dims, basic_dims)

        self.e2_c1 = GeneralConv3dPrenorm(basic_dims, basic_dims * 2, stride=2)
        self.e2_c2 = GeneralConv3dPrenorm(basic_dims * 2, basic_dims * 2)
        self.e2_c3 = GeneralConv3dPrenorm(basic_dims * 2, basic_dims * 2)

        self.e3_c1 = GeneralConv3dPrenorm(basic_dims * 2, basic_dims * 4, stride=2)
        self.e3_c2 = GeneralConv3dPrenorm(basic_dims * 4, basic_dims * 4)
        self.e3_c3 = GeneralConv3dPrenorm(basic_dims * 4, basic_dims * 4)

        self.e4_c1 = GeneralConv3dPrenorm(basic_dims * 4, basic_dims * 8, stride=2)
        self.e4_c2 = GeneralConv3dPrenorm(basic_dims * 8, basic_dims * 8)
        self.e4_c3 = GeneralConv3dPrenorm(basic_dims * 8, basic_dims * 8)

        self.e5_c1 = GeneralConv3dPrenorm(basic_dims * 8, basic_dims * 16, stride=2)
        self.e5_c2 = GeneralConv3dPrenorm(basic_dims * 16, basic_dims * 16)
        self.e5_c3 = GeneralConv3dPrenorm(basic_dims * 16, basic_dims * 16)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))
        return x1, x2, x3, x4, x5


class DecoderSep(nn.Module):
    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.d4 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d4_c1 = GeneralConv3dPrenorm(basic_dims * 16, basic_dims * 8)
        self.d4_c2 = GeneralConv3dPrenorm(basic_dims * 16, basic_dims * 8)
        self.d4_out = GeneralConv3dPrenorm(basic_dims * 8, basic_dims * 8, k_size=1, padding=0)

        self.d3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d3_c1 = GeneralConv3dPrenorm(basic_dims * 8, basic_dims * 4)
        self.d3_c2 = GeneralConv3dPrenorm(basic_dims * 8, basic_dims * 4)
        self.d3_out = GeneralConv3dPrenorm(basic_dims * 4, basic_dims * 4, k_size=1, padding=0)

        self.d2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d2_c1 = GeneralConv3dPrenorm(basic_dims * 4, basic_dims * 2)
        self.d2_c2 = GeneralConv3dPrenorm(basic_dims * 4, basic_dims * 2)
        self.d2_out = GeneralConv3dPrenorm(basic_dims * 2, basic_dims * 2, k_size=1, padding=0)

        self.d1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d1_c1 = GeneralConv3dPrenorm(basic_dims * 2, basic_dims)
        self.d1_c2 = GeneralConv3dPrenorm(basic_dims * 2, basic_dims)
        self.d1_out = GeneralConv3dPrenorm(basic_dims, basic_dims, k_size=1, padding=0)

        self.seg_layer = nn.Conv3d(basic_dims, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
    ) -> torch.Tensor:
        de_x5 = self.d4_c1(self.d4(x5))
        de_x4 = self.d4_out(self.d4_c2(torch.cat((de_x5, x4), dim=1)))

        de_x4 = self.d3_c1(self.d3(de_x4))
        de_x3 = self.d3_out(self.d3_c2(torch.cat((de_x4, x3), dim=1)))

        de_x3 = self.d2_c1(self.d2(de_x3))
        de_x2 = self.d2_out(self.d2_c2(torch.cat((de_x3, x2), dim=1)))

        de_x2 = self.d1_c1(self.d1(de_x2))
        de_x1 = self.d1_out(self.d1_c2(torch.cat((de_x2, x1), dim=1)))

        return self.softmax(self.seg_layer(de_x1))


def _channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    batch, channels, depth, height, width = x.size()
    channels_per_group = channels // groups
    x = x.view(batch, groups, channels_per_group, depth, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batch, -1, depth, height, width)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class DSAM(nn.Module):
    """Channel-refine + spatial-attention-gated upsample block used by the fusion decoder."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        inter_channels = in_channels // 4

        self.channel_refine = nn.Sequential(
            nn.Conv3d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, in_channels, kernel_size=1),
            nn.BatchNorm3d(in_channels),
        )
        self.spatial_attention = SpatialAttention()
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(True),
        )
        self.pwc = nn.Conv3d(in_channels, out_channels * 2, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_refine(x)
        x = self.spatial_attention(x) * x
        x = self.up_dwc(x)
        x = _channel_shuffle(x, self.in_channels)
        return self.pwc(x)


class PRMFusion(nn.Module):
    def __init__(self, in_channel: int, num_cls: int) -> None:
        super().__init__()
        self.seg_layer = nn.Conv3d(in_channel, num_cls, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.seg_layer(x))


class DecoderFuse(nn.Module):
    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.d4_c1 = GeneralConv3dPrenorm(basic_dims * 16, basic_dims * 8)
        self.d4_c2 = GeneralConv3dPrenorm(basic_dims * 16, basic_dims * 8)
        self.d4_out = GeneralConv3dPrenorm(basic_dims * 8, basic_dims * 8, k_size=1, padding=0)

        self.d3_c1 = GeneralConv3dPrenorm(basic_dims * 8, basic_dims * 4)
        self.d3_c2 = GeneralConv3dPrenorm(basic_dims * 8, basic_dims * 4)
        self.d3_out = GeneralConv3dPrenorm(basic_dims * 4, basic_dims * 4, k_size=1, padding=0)

        self.d2_c1 = GeneralConv3dPrenorm(basic_dims * 4, basic_dims * 2)
        self.d2_c2 = GeneralConv3dPrenorm(basic_dims * 4, basic_dims * 2)
        self.d2_out = GeneralConv3dPrenorm(basic_dims * 2, basic_dims * 2, k_size=1, padding=0)

        self.d1_c1 = GeneralConv3dPrenorm(basic_dims * 2, basic_dims)
        self.d1_c2 = GeneralConv3dPrenorm(basic_dims * 2, basic_dims)
        self.d1_out = GeneralConv3dPrenorm(basic_dims, basic_dims, k_size=1, padding=0)

        self.seg_layer = nn.Conv3d(basic_dims, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2_2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.up4_4 = nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)
        self.up8_8 = nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True)
        self.up16_16 = nn.Upsample(scale_factor=16, mode="trilinear", align_corners=True)

        self.up2 = DSAM(basic_dims * 16, basic_dims * 8)
        self.up4 = DSAM(basic_dims * 8, basic_dims * 4)
        self.up8 = DSAM(basic_dims * 4, basic_dims * 2)
        self.up16 = DSAM(basic_dims * 2, basic_dims)

        self.rfm5 = FusionPrenorm(in_channel=basic_dims * 16, num_modals=num_modals)
        self.rfm4 = FusionPrenorm(in_channel=basic_dims * 8, num_modals=num_modals)
        self.rfm3 = FusionPrenorm(in_channel=basic_dims * 4, num_modals=num_modals)
        self.rfm2 = FusionPrenorm(in_channel=basic_dims * 2, num_modals=num_modals)
        self.rfm1 = FusionPrenorm(in_channel=basic_dims * 1, num_modals=num_modals)

        self.prm_fusion4 = PRMFusion(basic_dims * 16, num_cls)
        self.prm_fusion3 = PRMFusion(basic_dims * 8, num_cls)
        self.prm_fusion2 = PRMFusion(basic_dims * 4, num_cls)
        self.prm_fusion1 = PRMFusion(basic_dims * 2, num_cls)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        de_x5 = self.rfm5(x5)
        pred4 = self.prm_fusion4(de_x5)
        de_x5 = self.d4_c1(self.up2(de_x5))

        de_x4 = self.rfm4(x4)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        pred3 = self.prm_fusion3(de_x4)
        de_x4 = self.d3_c1(self.up4(de_x4))

        de_x3 = self.rfm3(x3)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        pred2 = self.prm_fusion2(de_x3)
        de_x3 = self.d2_c1(self.up8(de_x3))

        de_x2 = self.rfm2(x2)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        pred1 = self.prm_fusion1(de_x2)
        de_x2 = self.d1_c1(self.up16(de_x2))

        de_x1 = self.rfm1(x1)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred, (self.up2_2(pred1), self.up4_4(pred2), self.up8_8(pred3), self.up16_16(pred4))


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim: int, dropout_rate: float, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, embedding_dim: int, depth: int, heads: int, mlp_dim: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.depth = depth
        self.attention_list = nn.ModuleList(
            [
                Residual(PreNormDrop(embedding_dim, dropout_rate, SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate)))
                for _ in range(depth)
            ]
        )
        self.ffn_list = nn.ModuleList(
            [Residual(PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate))) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        for j in range(self.depth):
            x = x + pos
            x = self.attention_list[j](x)
            x = self.ffn_list[j](x)
        return x


class MaskModal(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, num_modal, channels, height, width, depth = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        return y.view(batch, -1, height, width, depth)


def _create_feed_forward(dim: int, dropout_rate: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dim, dim * 4),
        nn.LayerNorm(dim * 4),
        nn.GELU(),
        nn.Dropout(dropout_rate),
        nn.Linear(dim * 4, dim),
        nn.LayerNorm(dim),
    )


class RFIM(nn.Module):
    """RFL Feature Interaction Module.

    A reversible coupling-style MLP over the token-sequence dimension: it
    imputes a missing modality's deep bottleneck tokens from the "global"
    (most reliably present) modality's own tokens (``forward_half``), and
    during training also computes an algebraic inverse of that same
    mapping (``forward``) so a self-consistency loss can supervise the
    imputation to be (approximately) invertible back to the source tokens.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.feed_forward_s1 = _create_feed_forward(dim)
        self.feed_forward_s2 = _create_feed_forward(dim)
        self.feed_forward_t1 = _create_feed_forward(dim)
        self.feed_forward_t2 = _create_feed_forward(dim)

    def forward_half(self, x: torch.Tensor) -> torch.Tensor:
        half = x.shape[1] // 2
        part1, part2 = x[:, :half, :], x[:, half:, :]

        s_out1 = self.feed_forward_s1(part1)
        t_out1 = self.feed_forward_t1(part2)

        return torch.cat(
            (
                self.feed_forward_t2(s_out1 + part2),
                self.feed_forward_s2(s_out1 + t_out1 + part2),
            ),
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        forward_out = self.forward_half(x)

        half = forward_out.shape[1] // 2
        inverse_part1, inverse_part2 = forward_out[:, :half, :], forward_out[:, half:, :]

        inverse_t_out = self.feed_forward_t2(inverse_part1)
        inverse_s_out = self.feed_forward_s2(inverse_part2)
        inverse_out2 = self.feed_forward_t1(inverse_s_out - inverse_t_out)
        inverse_out = torch.cat(
            (
                self.feed_forward_s1(inverse_t_out - inverse_out2),
                inverse_out2,
            ),
            dim=1,
        )
        return forward_out, inverse_out


class RFL(AbstractModel):
    """Implementation of RFL introduced in [1].

    4 per-modality encoders feed 4 per-modality self-attention transformers
    ("IntraFormer") at the bottleneck. The paper's namesake mechanism,
    RFIM ("RFL Feature Interaction Module"), imputes a missing
    modality's bottleneck tokens from the most reliably present modality
    (t1ce if present, else the first present modality — the "global
    modality") via a reversible coupling-style MLP; a self-consistency loss
    supervises the rfl/inverse direction to approximately reconstruct
    the true tokens. After substitution, a shared cross-modal transformer
    ("InterFormer") fuses all 4 modalities' tokens before a channel/spatial-
    attention-gated upsampling decoder (`DSAM`) with 4-scale deep
    supervision.

    Legacy has two real bugs this port fixes:
    - The 4 `RFIM` submodules are constructed in a different modality order
      (``flair, t1, t1ce, t2``) than the mask/encoder convention used
      everywhere else in the model (``flair, t1ce, t1, t2``), so legacy's
      substitution logic silently applies the wrong modality's RFIM module
      whenever t1ce or t1 (but not flair/t2) is missing. This port builds
      the 4 RFIM modules in the same order as the mask/encoder convention.
    - Legacy decides the "global modality" and which modalities to
      RFIM-impute using only ``mask[0]`` (the first sample's pattern) and
      hardcodes the bottleneck reshape to batch size 1, silently applying
      sample 0's missing-modality pattern to every sample in the batch.
      This port resolves the global modality and substitutes tokens
      per-sample, respecting MissingBench's true per-sample `[B, 4]` masks.

    Legacy also has the same `num_cls`/`num_modals` conflation bug found in
    SRMNet/MMMViT/IMS2Trans/MIFPN in its `fusion_prenorm` fusion blocks,
    fixed here the same way (`FusionPrenorm` always uses `num_modals=4`).

    [1] original RFL missing-modality brain tumor segmentation model.
    """

    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()

        self.flair_encode_conv = nn.Conv3d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t1ce_encode_conv = nn.Conv3d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t1_encode_conv = nn.Conv3d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t2_encode_conv = nn.Conv3d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)

        self.flair_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.t1ce_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.t1_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.t2_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))

        self.flair_transformer = Transformer(transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.t1ce_transformer = Transformer(transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.t1_transformer = Transformer(transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.t2_transformer = Transformer(transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)

        self.multimodal_transformer = Transformer(transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.multimodal_decode_conv = nn.Conv3d(
            transformer_basic_dims * num_modals, basic_dims * 16 * num_modals, kernel_size=1, padding=0
        )

        self.masker = MaskModal()
        self.decoder_fuse = DecoderFuse(num_cls=num_cls)
        self.decoder_sep = DecoderSep(num_cls=num_cls)

        # Ordered [flair, t1ce, t1, t2] to match the mask/encoder convention.
        self.rfim = nn.ModuleList([RFIM(dim=transformer_basic_dims) for _ in range(num_modals)])

        self.is_training = False

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                torch.nn.init.kaiming_normal_(module.weight)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | tuple[
        torch.Tensor,
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        x, mask = self._remap_input_order(x, mask)
        batch = x.size(0)

        flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.flair_encoder(x[:, 0:1])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t1ce_encoder(x[:, 1:2])
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t1_encoder(x[:, 2:3])
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 3:4])

        def _tokenize(conv: nn.Conv3d, feat: torch.Tensor) -> torch.Tensor:
            return conv(feat).permute(0, 2, 3, 4, 1).contiguous().view(batch, -1, transformer_basic_dims)

        flair_intra = self.flair_transformer(_tokenize(self.flair_encode_conv, flair_x5), self.flair_pos)
        t1ce_intra = self.t1ce_transformer(_tokenize(self.t1ce_encode_conv, t1ce_x5), self.t1ce_pos)
        t1_intra = self.t1_transformer(_tokenize(self.t1_encode_conv, t1_x5), self.t1_pos)
        t2_intra = self.t2_transformer(_tokenize(self.t2_encode_conv, t2_x5), self.t2_pos)
        intra_tokens = [flair_intra, t1ce_intra, t1_intra, t2_intra]

        # For each sample, pick the "global" (most reliably present) modality
        # and impute any missing modality's tokens from it. `forward_outputs`
        # target each modality's own real tokens (forward-imputation loss);
        # `global_tokens_used` targets the per-sample global tokens that fed
        # each RFIM call (rfl/inverse-consistency loss).
        substituted_tokens = [tokens.clone() for tokens in intra_tokens]
        inverse_outputs: list[torch.Tensor] = [tokens.clone() for tokens in intra_tokens]
        forward_outputs: list[torch.Tensor] = [tokens.clone() for tokens in intra_tokens]
        global_tokens_used: list[torch.Tensor] = [tokens.clone() for tokens in intra_tokens]

        for sample_index in range(batch):
            sample_mask = mask[sample_index]
            if sample_mask[1]:
                global_modality = 1
            else:
                present = torch.nonzero(sample_mask, as_tuple=False)
                global_modality = int(present[0].item()) if present.numel() > 0 else 0
            global_tokens = intra_tokens[global_modality][sample_index : sample_index + 1]

            for modal_index in range(num_modals):
                if self.is_training:
                    forward_out, inverse_out = self.rfim[modal_index](global_tokens)
                    forward_outputs[modal_index][sample_index : sample_index + 1] = forward_out
                    inverse_outputs[modal_index][sample_index : sample_index + 1] = inverse_out
                    global_tokens_used[modal_index][sample_index : sample_index + 1] = global_tokens
                    if not sample_mask[modal_index]:
                        substituted_tokens[modal_index][sample_index : sample_index + 1] = forward_out
                elif not sample_mask[modal_index]:
                    substituted_tokens[modal_index][sample_index : sample_index + 1] = self.rfim[modal_index].forward_half(
                        global_tokens
                    )

        def _untokenize(tokens: torch.Tensor) -> torch.Tensor:
            return (
                tokens.view(batch, patch_size, patch_size, patch_size, transformer_basic_dims)
                .permute(0, 4, 1, 2, 3)
                .contiguous()
            )

        flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5 = (_untokenize(tokens) for tokens in substituted_tokens)

        if self.is_training:
            flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4, flair_x5)
            t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5)
            t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4, t1_x5)
            t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5)

        x1 = self.masker(torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1), mask)
        x2 = self.masker(torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1), mask)
        x3 = self.masker(torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1), mask)
        x4 = self.masker(torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1), mask)

        multimodal_token_x5 = torch.cat(
            (
                flair_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(batch, -1, transformer_basic_dims),
                t1ce_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(batch, -1, transformer_basic_dims),
                t1_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(batch, -1, transformer_basic_dims),
                t2_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(batch, -1, transformer_basic_dims),
            ),
            dim=1,
        )
        multimodal_pos = torch.cat((self.flair_pos, self.t1ce_pos, self.t1_pos, self.t2_pos), dim=1)
        multimodal_inter_token_x5 = self.multimodal_transformer(multimodal_token_x5, multimodal_pos)
        multimodal_inter_x5 = self.multimodal_decode_conv(
            multimodal_inter_token_x5.view(batch, patch_size, patch_size, patch_size, transformer_basic_dims * num_modals)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )

        fuse_pred, preds = self.decoder_fuse(x1, x2, x3, x4, multimodal_inter_x5)

        if not self.is_training:
            return fuse_pred

        return (
            fuse_pred,
            (flair_pred, t1ce_pred, t1_pred, t2_pred),
            preds,
            forward_outputs,
            inverse_outputs,
            intra_tokens,
            global_tokens_used,
        )

    def predict(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, _, height, width, depth = images.shape
        if (height, width, depth) == (input_patch_size, input_patch_size, input_patch_size):
            return self(images, mask)

        h_starts = self._window_starts(height)
        w_starts = self._window_starts(width)
        d_starts = self._window_starts(depth)
        prediction = torch.zeros(
            images.size(0),
            self.decoder_fuse.seg_layer.out_channels,
            height,
            width,
            depth,
            device=images.device,
        )
        weight = torch.zeros(images.size(0), 1, height, width, depth, device=images.device)

        for h in h_starts:
            for w in w_starts:
                for d in d_starts:
                    patch = images[:, :, h : h + input_patch_size, w : w + input_patch_size, d : d + input_patch_size]
                    patch_pred = self(patch, mask)
                    prediction[
                        :, :, h : h + input_patch_size, w : w + input_patch_size, d : d + input_patch_size
                    ] += patch_pred
                    weight[
                        :, :, h : h + input_patch_size, w : w + input_patch_size, d : d + input_patch_size
                    ] += 1
        return prediction / weight

    @staticmethod
    def _window_starts(size: int) -> list[int]:
        if size < input_patch_size:
            raise RuntimeError(
                f"Model prediction expects spatial dimensions to be at least {input_patch_size}, got {size}"
            )
        if size == input_patch_size:
            return [0]

        stride = input_patch_size // 2
        starts = list(range(0, max(size - input_patch_size, 0), stride))
        last_start = size - input_patch_size
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        return starts

    @staticmethod
    def _remap_input_order(
        images: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images.size(1) != num_modals:
            raise RuntimeError(f"RFL expects {num_modals} input modalities, got {images.size(1)}")
        if mask.ndim != 2 or mask.size(1) != num_modals:
            raise RuntimeError(f"RFL expects mask shape [B, {num_modals}], got {tuple(mask.shape)}")

        return images[:, DATASET_MODALITY_ORDER, ...], mask[:, DATASET_MODALITY_ORDER]


Model = RFL


__all__ = ["RFL", "Model"]
