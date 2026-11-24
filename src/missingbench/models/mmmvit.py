from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from missingbench.models.abstract_model import AbstractModel

basic_dims = 8
transformer_basic_dims = 512
mlp_dim = 512
num_heads = 8
depth = 1
num_modals = 4
patch_size = 8
input_patch_size = 128

# External pipeline uses [t1c, t1n, t2f, t2w]; MMMViT internals expect
# [flair, t1ce, t1, t2] (same convention as DCSeg/RFNet/M2FTrans/mmformer).
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
        pad_type: str = "replicate",
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


class GeneralConv3d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pad_type: str = "replicate",
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
        self.norm = normalization(out_ch, norm=norm)
        if act_type == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif act_type == "lrelu":
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)
        else:
            raise ValueError(f"activation type {act_type} is not supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class FusionPrenorm(nn.Module):
    """Zero out missing modalities, concatenate, and fuse with plain convs.

    Legacy names this class's second constructor argument ``num_cls`` and
    passes the segmentation-class count into it, but it is only ever used as
    the input-channel multiplier for the 4 stacked modalities. Passing the
    real ``num_cls`` breaks whenever the dataset's class count differs from
    ``num_modals`` (e.g. BraTS25's 5 classes), so this port renames the
    parameter to ``num_modals`` and always fixes it at 4.
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
        self.e1_c1 = nn.Conv3d(1, basic_dims, kernel_size=3, stride=1, padding=1, padding_mode="replicate", bias=True)
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

        self.e5_c1 = GeneralConv3dPrenorm(basic_dims * 8, basic_dims * 8, stride=2)
        self.e5_c2 = GeneralConv3dPrenorm(basic_dims * 8, basic_dims * 8)
        self.e5_c3 = GeneralConv3dPrenorm(basic_dims * 8, basic_dims * 8)

        self.conv = nn.Conv3d(basic_dims * 23, basic_dims * 8, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        x1_ = F.interpolate(x1, (patch_size, patch_size, patch_size))
        x2_ = F.interpolate(x2, (patch_size, patch_size, patch_size))
        x3_ = F.interpolate(x3, (patch_size, patch_size, patch_size))
        x4_ = F.interpolate(x4, (patch_size, patch_size, patch_size))
        x6 = torch.cat([x1_, x2_, x3_, x4_, x5], dim=1)
        x6 = self.conv(x6)
        return x1, x2, x3, x4, x5, x6


class DecoderSep(nn.Module):
    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.d4 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d4_c1 = GeneralConv3dPrenorm(basic_dims * 8, basic_dims * 8)
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


class DecoderFuse(nn.Module):
    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.d4_c1 = GeneralConv3dPrenorm(basic_dims * 8, basic_dims * 8)
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

        self.seg_d4 = nn.Conv3d(basic_dims * 8, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d3 = nn.Conv3d(basic_dims * 8, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d2 = nn.Conv3d(basic_dims * 4, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d1 = nn.Conv3d(basic_dims * 2, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_layer = nn.Conv3d(basic_dims, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode="trilinear", align_corners=True)

        self.rfm5 = FusionPrenorm(in_channel=basic_dims * 8, num_modals=num_modals)
        self.rfm4 = FusionPrenorm(in_channel=basic_dims * 8, num_modals=num_modals)
        self.rfm3 = FusionPrenorm(in_channel=basic_dims * 4, num_modals=num_modals)
        self.rfm2 = FusionPrenorm(in_channel=basic_dims * 2, num_modals=num_modals)
        self.rfm1 = FusionPrenorm(in_channel=basic_dims * 1, num_modals=num_modals)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        de_x5 = self.rfm5(x5)
        pred4 = self.softmax(self.seg_d4(de_x5))
        de_x5 = self.d4_c1(self.up2(de_x5))

        de_x4 = self.rfm4(x4)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        pred3 = self.softmax(self.seg_d3(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))

        de_x3 = self.rfm3(x3)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        pred2 = self.softmax(self.seg_d2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        de_x2 = self.rfm2(x2)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        pred1 = self.softmax(self.seg_d1(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        de_x1 = self.rfm1(x1)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred, (self.up2(pred1), self.up4(pred2), self.up8(pred3), self.up16(pred4))


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, qkv_bias: bool = False, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
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
    """Single-modality self-attention transformer used by the IntraFormer stage."""

    def __init__(self, embedding_dim: int, depth: int, heads: int, mlp_dim: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.depth = depth
        self.cross_attention_list = nn.ModuleList(
            [
                Residual(
                    PreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
                for _ in range(depth)
            ]
        )
        self.cross_ffn_list = nn.ModuleList(
            [
                Residual(PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate)))
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        for j in range(self.depth):
            x = x + pos
            x = self.cross_attention_list[j](x)
            x = self.cross_ffn_list[j](x)
        return x


class MaskModal(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, num_modal, channels, height, width, depth = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        return y.view(batch, -1, height, width, depth)


class MMMViT(AbstractModel):
    """Implementation of MMMViT introduced in [1].

    IntraFormer per-modality self-attention transformers refine each
    modality's encoder bottleneck features, and an InterFormer models
    pairwise modality correlations via a learned softmax gate before fusing
    into a single bottleneck representation for the decoder.

    Legacy computes a masked version of the per-modality IntraFormer output
    (``x6_intra``) but never actually feeds it into the InterFormer
    correlation modeling, which uses the unmasked per-modality tokens
    instead — silently leaking "missing" modality information into the
    fused prediction and undermining the whole missing-modality contract.
    This port fixes that by feeding the masked per-modality tokens into the
    InterFormer stage, matching the intent of mmformer's equivalent (and
    correctly masked) bottleneck fusion.

    [1] original MMMViT missing-modality brain tumor segmentation model.
    """

    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()

        self.flair_encode_conv = nn.Conv3d(basic_dims * 8, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t1ce_encode_conv = nn.Conv3d(basic_dims * 8, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t1_encode_conv = nn.Conv3d(basic_dims * 8, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t2_encode_conv = nn.Conv3d(basic_dims * 8, transformer_basic_dims, kernel_size=1, stride=1, padding=0)

        self.flair_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.t1ce_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.t1_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.t2_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))

        self.flair_transformer = Transformer(transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.t1ce_transformer = Transformer(transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.t1_transformer = Transformer(transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.t2_transformer = Transformer(transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)

        # Multimodal correlation modeling (InterFormer).
        self.qkv_flair = nn.Conv3d(transformer_basic_dims, transformer_basic_dims * 3, kernel_size=1, stride=1, padding=0)
        self.qkv_t1ce = nn.Conv3d(transformer_basic_dims, transformer_basic_dims * 3, kernel_size=1, stride=1, padding=0)
        self.qkv_t1 = nn.Conv3d(transformer_basic_dims, transformer_basic_dims * 3, kernel_size=1, stride=1, padding=0)
        self.qkv_t2 = nn.Conv3d(transformer_basic_dims, transformer_basic_dims * 3, kernel_size=1, stride=1, padding=0)
        self.softmax_flair = nn.Softmax(dim=0)
        self.softmax_t1ce = nn.Softmax(dim=0)
        self.softmax_t1 = nn.Softmax(dim=0)
        self.softmax_t2 = nn.Softmax(dim=0)

        # Multimodal representation fusion.
        self.multimodal_transformer = Transformer(
            transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim
        )
        self.multimodal_decode_conv = nn.Conv3d(
            transformer_basic_dims * num_modals, basic_dims * 8 * num_modals, kernel_size=1, padding=0
        )

        self.masker = MaskModal()
        self.decoder_fuse = DecoderFuse(num_cls=num_cls)
        self.decoder_sep = DecoderSep(num_cls=num_cls)

        self.is_training = False

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                torch.nn.init.kaiming_normal_(module.weight)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        x, mask = self._remap_input_order(x, mask)

        flair_x1, flair_x2, flair_x3, flair_x4, flair_x5, flair_x6 = self.flair_encoder(x[:, 0:1])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5, t1ce_x6 = self.t1ce_encoder(x[:, 1:2])
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5, t1_x6 = self.t1_encoder(x[:, 2:3])
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5, t2_x6 = self.t2_encoder(x[:, 3:4])

        batch = x.size(0)

        def _tokenize(conv: nn.Conv3d, feat: torch.Tensor) -> torch.Tensor:
            return conv(feat).permute(0, 2, 3, 4, 1).contiguous().view(batch, -1, transformer_basic_dims)

        flair_token_x6 = _tokenize(self.flair_encode_conv, flair_x6)
        t1ce_token_x6 = _tokenize(self.t1ce_encode_conv, t1ce_x6)
        t1_token_x6 = _tokenize(self.t1_encode_conv, t1_x6)
        t2_token_x6 = _tokenize(self.t2_encode_conv, t2_x6)

        flair_intra_token_x6 = self.flair_transformer(flair_token_x6, self.flair_pos)
        t1ce_intra_token_x6 = self.t1ce_transformer(t1ce_token_x6, self.t1ce_pos)
        t1_intra_token_x6 = self.t1_transformer(t1_token_x6, self.t1_pos)
        t2_intra_token_x6 = self.t2_transformer(t2_token_x6, self.t2_pos)

        def _untokenize(tokens: torch.Tensor) -> torch.Tensor:
            return (
                tokens.view(batch, patch_size, patch_size, patch_size, transformer_basic_dims)
                .permute(0, 4, 1, 2, 3)
                .contiguous()
            )

        flair_intra_x6 = _untokenize(flair_intra_token_x6)
        t1ce_intra_x6 = _untokenize(t1ce_intra_token_x6)
        t1_intra_x6 = _untokenize(t1_intra_token_x6)
        t2_intra_x6 = _untokenize(t2_intra_token_x6)

        if self.is_training:
            flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4, flair_x5)
            t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5)
            t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4, t1_x5)
            t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5)

        x1 = self.masker(torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1), mask)
        x2 = self.masker(torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1), mask)
        x3 = self.masker(torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1), mask)
        x4 = self.masker(torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1), mask)
        x6_intra = self.masker(
            torch.stack((flair_intra_x6, t1ce_intra_x6, t1_intra_x6, t2_intra_x6), dim=1), mask
        )
        flair_intra_x6, t1ce_intra_x6, t1_intra_x6, t2_intra_x6 = torch.chunk(x6_intra, num_modals, dim=1)

        # Multimodal correlation modeling: cross-modality softmax-weighted value gathering.
        temp_flair = self.qkv_flair(flair_intra_x6)
        q_flair, k_flair, v_flair = torch.chunk(temp_flair, 3, dim=1)

        temp_t1ce = self.qkv_t1ce(t1ce_intra_x6)
        q_t1ce, k_t1ce, v_t1ce = torch.chunk(temp_t1ce, 3, dim=1)

        temp_t1 = self.qkv_t1(t1_intra_x6)
        q_t1, k_t1, v_t1 = torch.chunk(temp_t1, 3, dim=1)

        temp_t2 = self.qkv_t2(t2_intra_x6)
        q_t2, k_t2, v_t2 = torch.chunk(temp_t2, 3, dim=1)

        def _correlate(
            query: torch.Tensor,
            softmax: nn.Softmax,
            reference: torch.Tensor,
        ) -> torch.Tensor:
            para = torch.cat(
                [
                    (query * k_flair).view(1, -1),
                    (query * k_t1ce).view(1, -1),
                    (query * k_t1).view(1, -1),
                    (query * k_t2).view(1, -1),
                ],
                dim=0,
            )
            weights = softmax(para / np.sqrt(num_modals)).view(
                reference.size(0), reference.size(1) * 4, reference.size(2), reference.size(3), reference.size(4)
            )
            channels = reference.size(1)
            return (
                weights[:, 0:channels] * v_flair
                + weights[:, channels : channels * 2] * v_t1ce
                + weights[:, channels * 2 : channels * 3] * v_t1
                + weights[:, channels * 3 : channels * 4] * v_t2
            )

        x6_flair_ = _correlate(q_flair, self.softmax_flair, flair_intra_x6)
        x6_t1ce_ = _correlate(q_t1ce, self.softmax_t1ce, t1ce_intra_x6)
        x6_t1_ = _correlate(q_t1, self.softmax_t1, t1_intra_x6)
        x6_t2_ = _correlate(q_t2, self.softmax_t2, t2_intra_x6)

        x6_intra_ = torch.stack((x6_flair_, x6_t1ce_, x6_t1_, x6_t2_), dim=1).view(
            batch, -1, patch_size, patch_size, patch_size
        )

        # Multimodal representation fusion.
        flair_intra_x6, t1ce_intra_x6, t1_intra_x6, t2_intra_x6 = torch.chunk(x6_intra_, num_modals, dim=1)
        multimodal_token_x6 = torch.cat(
            (
                flair_intra_x6.permute(0, 2, 3, 4, 1).contiguous().view(batch, -1, transformer_basic_dims),
                t1ce_intra_x6.permute(0, 2, 3, 4, 1).contiguous().view(batch, -1, transformer_basic_dims),
                t1_intra_x6.permute(0, 2, 3, 4, 1).contiguous().view(batch, -1, transformer_basic_dims),
                t2_intra_x6.permute(0, 2, 3, 4, 1).contiguous().view(batch, -1, transformer_basic_dims),
            ),
            dim=1,
        )
        multimodal_pos = torch.cat((self.flair_pos, self.t1ce_pos, self.t1_pos, self.t2_pos), dim=1)
        multimodal_inter_token_x6 = self.multimodal_transformer(multimodal_token_x6, multimodal_pos)
        multimodal_inter_x6 = self.multimodal_decode_conv(
            multimodal_inter_token_x6.view(
                batch, patch_size, patch_size, patch_size, transformer_basic_dims * num_modals
            )
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )

        fuse_pred, preds = self.decoder_fuse(x1, x2, x3, x4, multimodal_inter_x6)

        if not self.is_training:
            return fuse_pred

        return fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), preds

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
            raise RuntimeError(f"MMMViT expects {num_modals} input modalities, got {images.size(1)}")
        if mask.ndim != 2 or mask.size(1) != num_modals:
            raise RuntimeError(f"MMMViT expects mask shape [B, {num_modals}], got {tuple(mask.shape)}")

        return images[:, DATASET_MODALITY_ORDER, ...], mask[:, DATASET_MODALITY_ORDER]


Model = MMMViT


__all__ = ["MMMViT", "Model"]
