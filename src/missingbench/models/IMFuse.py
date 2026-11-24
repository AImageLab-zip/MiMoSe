from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch.amp import autocast

from missingbench.models.abstract_model import AbstractModel


basic_dims = 8
transformer_basic_dims = 512
mlp_dim = 4096
num_heads = 8
depth = 1
num_modals = 4
patch_size = 8
input_patch_size = 128
DATASET_MODALITY_ORDER = (2, 0, 1, 3)


def normalization(planes: int, norm: str = "bn") -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm3d(planes)
    if norm == "gn":
        return nn.GroupNorm(4, planes)
    if norm == "in":
        return nn.InstanceNorm3d(planes)
    raise ValueError(f"normalization type {norm} is not supported")


class general_conv3d_prenorm(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pad_type: str = "zeros",
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
            self.activation = nn.LeakyReLU(
                negative_slope=relufactor,
                inplace=True,
            )
        else:
            raise ValueError(f"activation type {act_type} is not supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        return x


class fusion_prenorm(nn.Module):
    def __init__(self, in_channel: int = 64, num_cls: int = 4) -> None:
        super().__init__()
        self.fusion_layer = nn.Sequential(
            general_conv3d_prenorm(
                in_channel * num_cls,
                in_channel,
                k_size=1,
                padding=0,
                stride=1,
            ),
            general_conv3d_prenorm(
                in_channel,
                in_channel,
                k_size=3,
                padding=1,
                stride=1,
            ),
            general_conv3d_prenorm(
                in_channel,
                in_channel,
                k_size=1,
                padding=0,
                stride=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion_layer(x)


class InitWeights_He:
    def __init__(self, neg_slope: float = 1e-2) -> None:
        self.neg_slope = neg_slope

    def __call__(self, module: nn.Module) -> None:
        if isinstance(
            module,
            (nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d, nn.ConvTranspose3d),
        ):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class MambaTrans(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mamba = Mamba(
            d_model=channels,
            d_state=min(channels, 256),
            d_conv=4,
            expand=2,
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.head = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mamba(self.norm1(x)) + x
        x = self.head(self.norm2(x)) + x
        return x


class MambaLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mamba = MambaTrans(dim)

    @autocast(enabled=False,device_type='cuda')
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        return self.mamba(x)


class MambaFusionLayer(nn.Module):
    def __init__(self, dim: int, num_tokens_fused_representation: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_tokens_fused_representation = num_tokens_fused_representation
        self.fused_tokens = nn.Parameter(
            torch.randn(1, self.num_tokens_fused_representation, dim)
        )
        self.mamba_layer = MambaLayer(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        fused_tokens = self.fused_tokens.repeat(batch_size, 1, 1)
        x_fused = torch.cat([x, fused_tokens], dim=1)
        x_mamba = self.mamba_layer(x_fused)
        return x_mamba[:, -self.num_tokens_fused_representation :, :]


class MambaFusionCatLayer(nn.Module):
    def __init__(self, dim: int, num_tokens_fused_representation: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_tokens_fused_representation = num_tokens_fused_representation
        self.fused_tokens = nn.Parameter(
            torch.randn(1, self.num_tokens_fused_representation, dim)
        )
        self.mamba_layer = MambaLayer(dim)

    def forward(self, x: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
        batch_size = x[0].size(0)
        fused_tokens = self.fused_tokens.repeat(batch_size, 1, 1)
        stacked = torch.stack([*x, fused_tokens], dim=2)
        stacked = stacked.view(batch_size, -1, self.dim)
        stacked = self.mamba_layer(stacked)
        return stacked[:, 4::5, :]


class Tokenize(nn.Module):
    def __init__(self, dims: int, num_modals: int = 4) -> None:
        super().__init__()
        self.dims = dims
        self.num_modals = num_modals

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flair_intra_x, t1ce_intra_x, t1_intra_x, t2_intra_x = torch.chunk(
            x,
            self.num_modals,
            dim=1,
        )
        return torch.cat(
            (
                flair_intra_x.permute(0, 2, 3, 4, 1).contiguous().view(
                    x.size(0), -1, self.dims
                ),
                t1ce_intra_x.permute(0, 2, 3, 4, 1).contiguous().view(
                    x.size(0), -1, self.dims
                ),
                t1_intra_x.permute(0, 2, 3, 4, 1).contiguous().view(
                    x.size(0), -1, self.dims
                ),
                t2_intra_x.permute(0, 2, 3, 4, 1).contiguous().view(
                    x.size(0), -1, self.dims
                ),
            ),
            dim=1,
        )


class TokenizeSep(nn.Module):
    def __init__(self, dims: int, num_modals: int = 4) -> None:
        super().__init__()
        self.dims = dims
        self.num_modals = num_modals

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        flair_intra_x, t1ce_intra_x, t1_intra_x, t2_intra_x = torch.chunk(
            x,
            self.num_modals,
            dim=1,
        )
        return (
            flair_intra_x.permute(0, 2, 3, 4, 1).contiguous().view(
                x.size(0), -1, self.dims
            ),
            t1ce_intra_x.permute(0, 2, 3, 4, 1).contiguous().view(
                x.size(0), -1, self.dims
            ),
            t1_intra_x.permute(0, 2, 3, 4, 1).contiguous().view(
                x.size(0), -1, self.dims
            ),
            t2_intra_x.permute(0, 2, 3, 4, 1).contiguous().view(
                x.size(0), -1, self.dims
            ),
        )


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.e1_c1 = nn.Conv3d(
            in_channels=1,
            out_channels=basic_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
            bias=True,
        )
        self.e1_c2 = general_conv3d_prenorm(
            basic_dims,
            basic_dims,
            pad_type="reflect",
        )
        self.e1_c3 = general_conv3d_prenorm(
            basic_dims,
            basic_dims,
            pad_type="reflect",
        )
        self.e2_c1 = general_conv3d_prenorm(
            basic_dims,
            basic_dims * 2,
            stride=2,
            pad_type="reflect",
        )
        self.e2_c2 = general_conv3d_prenorm(
            basic_dims * 2,
            basic_dims * 2,
            pad_type="reflect",
        )
        self.e2_c3 = general_conv3d_prenorm(
            basic_dims * 2,
            basic_dims * 2,
            pad_type="reflect",
        )
        self.e3_c1 = general_conv3d_prenorm(
            basic_dims * 2,
            basic_dims * 4,
            stride=2,
            pad_type="reflect",
        )
        self.e3_c2 = general_conv3d_prenorm(
            basic_dims * 4,
            basic_dims * 4,
            pad_type="reflect",
        )
        self.e3_c3 = general_conv3d_prenorm(
            basic_dims * 4,
            basic_dims * 4,
            pad_type="reflect",
        )
        self.e4_c1 = general_conv3d_prenorm(
            basic_dims * 4,
            basic_dims * 8,
            stride=2,
            pad_type="reflect",
        )
        self.e4_c2 = general_conv3d_prenorm(
            basic_dims * 8,
            basic_dims * 8,
            pad_type="reflect",
        )
        self.e4_c3 = general_conv3d_prenorm(
            basic_dims * 8,
            basic_dims * 8,
            pad_type="reflect",
        )
        self.e5_c1 = general_conv3d_prenorm(
            basic_dims * 8,
            basic_dims * 16,
            stride=2,
            pad_type="reflect",
        )
        self.e5_c2 = general_conv3d_prenorm(
            basic_dims * 16,
            basic_dims * 16,
            pad_type="reflect",
        )
        self.e5_c3 = general_conv3d_prenorm(
            basic_dims * 16,
            basic_dims * 16,
            pad_type="reflect",
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
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


class Decoder_sep(nn.Module):
    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.d4 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d4_c1 = general_conv3d_prenorm(
            basic_dims * 16,
            basic_dims * 8,
            pad_type="reflect",
        )
        self.d4_c2 = general_conv3d_prenorm(
            basic_dims * 16,
            basic_dims * 8,
            pad_type="reflect",
        )
        self.d4_out = general_conv3d_prenorm(
            basic_dims * 8,
            basic_dims * 8,
            k_size=1,
            padding=0,
            pad_type="reflect",
        )
        self.d3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d3_c1 = general_conv3d_prenorm(
            basic_dims * 8,
            basic_dims * 4,
            pad_type="reflect",
        )
        self.d3_c2 = general_conv3d_prenorm(
            basic_dims * 8,
            basic_dims * 4,
            pad_type="reflect",
        )
        self.d3_out = general_conv3d_prenorm(
            basic_dims * 4,
            basic_dims * 4,
            k_size=1,
            padding=0,
            pad_type="reflect",
        )
        self.d2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d2_c1 = general_conv3d_prenorm(
            basic_dims * 4,
            basic_dims * 2,
            pad_type="reflect",
        )
        self.d2_c2 = general_conv3d_prenorm(
            basic_dims * 4,
            basic_dims * 2,
            pad_type="reflect",
        )
        self.d2_out = general_conv3d_prenorm(
            basic_dims * 2,
            basic_dims * 2,
            k_size=1,
            padding=0,
            pad_type="reflect",
        )
        self.d1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d1_c1 = general_conv3d_prenorm(
            basic_dims * 2,
            basic_dims,
            pad_type="reflect",
        )
        self.d1_c2 = general_conv3d_prenorm(
            basic_dims * 2,
            basic_dims,
            pad_type="reflect",
        )
        self.d1_out = general_conv3d_prenorm(
            basic_dims,
            basic_dims,
            k_size=1,
            padding=0,
            pad_type="reflect",
        )
        self.seg_layer = nn.Conv3d(
            in_channels=basic_dims,
            out_channels=num_cls,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
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
        cat_x4 = torch.cat((de_x5, x4), dim=1)
        de_x4 = self.d4_out(self.d4_c2(cat_x4))
        de_x4 = self.d3_c1(self.d3(de_x4))
        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))
        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))
        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))
        return self.softmax(self.seg_layer(de_x1))


class Decoder_fuse(nn.Module):
    def __init__(self, num_cls: int = 4, mamba_skip: bool = False) -> None:
        super().__init__()
        self.d4_c1 = general_conv3d_prenorm(
            basic_dims * 16,
            basic_dims * 8,
            pad_type="reflect",
        )
        self.d4_c2 = general_conv3d_prenorm(
            basic_dims * 16,
            basic_dims * 8,
            pad_type="reflect",
        )
        self.d4_out = general_conv3d_prenorm(
            basic_dims * 8,
            basic_dims * 8,
            k_size=1,
            padding=0,
            pad_type="reflect",
        )
        self.d3_c1 = general_conv3d_prenorm(
            basic_dims * 8,
            basic_dims * 4,
            pad_type="reflect",
        )
        self.d3_c2 = general_conv3d_prenorm(
            basic_dims * 8,
            basic_dims * 4,
            pad_type="reflect",
        )
        self.d3_out = general_conv3d_prenorm(
            basic_dims * 4,
            basic_dims * 4,
            k_size=1,
            padding=0,
            pad_type="reflect",
        )
        self.d2_c1 = general_conv3d_prenorm(
            basic_dims * 4,
            basic_dims * 2,
            pad_type="reflect",
        )
        self.d2_c2 = general_conv3d_prenorm(
            basic_dims * 4,
            basic_dims * 2,
            pad_type="reflect",
        )
        self.d2_out = general_conv3d_prenorm(
            basic_dims * 2,
            basic_dims * 2,
            k_size=1,
            padding=0,
            pad_type="reflect",
        )
        self.d1_c1 = general_conv3d_prenorm(
            basic_dims * 2,
            basic_dims,
            pad_type="reflect",
        )
        self.d1_c2 = general_conv3d_prenorm(
            basic_dims,
            basic_dims,
            pad_type="reflect",
        )
        self.d1_out = general_conv3d_prenorm(
            basic_dims,
            basic_dims,
            k_size=1,
            padding=0,
            pad_type="reflect",
        )
        self.seg_d4 = nn.Conv3d(basic_dims * 16, num_cls, 1, 1, 0, bias=True)
        self.seg_d3 = nn.Conv3d(basic_dims * 8, num_cls, 1, 1, 0, bias=True)
        self.seg_d2 = nn.Conv3d(basic_dims * 4, num_cls, 1, 1, 0, bias=True)
        self.seg_d1 = nn.Conv3d(basic_dims * 2, num_cls, 1, 1, 0, bias=True)
        self.seg_layer = nn.Conv3d(basic_dims, num_cls, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode="trilinear", align_corners=True)
        self.RFM5 = fusion_prenorm(in_channel=basic_dims * 16, num_cls=num_cls)
        self.RFM4 = fusion_prenorm(
            in_channel=basic_dims * 8,
            num_cls=1 if mamba_skip else num_cls,
        )
        self.RFM3 = fusion_prenorm(
            in_channel=basic_dims * 4,
            num_cls=1 if mamba_skip else num_cls,
        )
        self.RFM2 = fusion_prenorm(
            in_channel=basic_dims * 2,
            num_cls=1 if mamba_skip else num_cls,
        )
        self.mamba_skip = mamba_skip

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        de_x5 = self.RFM5(x5)
        pred4 = self.softmax(self.seg_d4(de_x5))
        de_x5 = self.d4_c1(self.up2(de_x5))
        de_x4 = self.RFM4(x4)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        pred3 = self.softmax(self.seg_d3(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))
        de_x3 = self.RFM3(x3)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        pred2 = self.softmax(self.seg_d2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))
        de_x2 = self.RFM2(x2)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        pred1 = self.softmax(self.seg_d1(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))
        de_x1 = self.d1_out(self.d1_c2(de_x2))
        pred = self.softmax(self.seg_layer(de_x1))
        return pred, (self.up2(pred1), self.up4(pred2), self.up8(pred3), self.up16(pred4))


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_size, num_tokens, 3, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, channels)
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


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Transformer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.cross_attention_list = nn.ModuleList()
        self.cross_ffn_list = nn.ModuleList()
        self.depth = depth
        for _ in range(self.depth):
            self.cross_attention_list.append(
                Residual(
                    PreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        SelfAttention(
                            embedding_dim,
                            heads=heads,
                            dropout_rate=dropout_rate,
                        ),
                    )
                )
            )
            self.cross_ffn_list.append(
                Residual(
                    PreNorm(
                        embedding_dim,
                        FeedForward(embedding_dim, mlp_dim, dropout_rate),
                    )
                )
            )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        for index in range(self.depth):
            x = x + pos
            x = self.cross_attention_list[index](x)
            x = self.cross_ffn_list[index](x)
        return x


class MaskModal(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, height, width, depth_size = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        return y.view(batch_size, -1, height, width, depth_size)


class IMFuse(AbstractModel):
    def __init__(
        self,
        num_cls: int = 4,
        interleaved_tokenization: bool = False,
        mamba_skip: bool = False,
    ) -> None:
        super().__init__()
        self.interleaved_tokenization = interleaved_tokenization
        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()

        if self.interleaved_tokenization:
            tokenizer_class = TokenizeSep
            mamba_fusion_layer_class = MambaFusionCatLayer
        else:
            tokenizer_class = Tokenize
            mamba_fusion_layer_class = MambaFusionLayer

        self.flair_encode_conv = nn.Conv3d(
            basic_dims * 16,
            transformer_basic_dims,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.t1ce_encode_conv = nn.Conv3d(
            basic_dims * 16,
            transformer_basic_dims,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.t1_encode_conv = nn.Conv3d(
            basic_dims * 16,
            transformer_basic_dims,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.t2_encode_conv = nn.Conv3d(
            basic_dims * 16,
            transformer_basic_dims,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.flair_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.t1ce_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.t1_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.t2_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.fused_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.flair_transformer = Transformer(transformer_basic_dims, depth, num_heads, mlp_dim)
        self.t1ce_transformer = Transformer(transformer_basic_dims, depth, num_heads, mlp_dim)
        self.t1_transformer = Transformer(transformer_basic_dims, depth, num_heads, mlp_dim)
        self.t2_transformer = Transformer(transformer_basic_dims, depth, num_heads, mlp_dim)
        self.mamba_fusion_layer = MambaFusionLayer(
            dim=transformer_basic_dims,
            num_tokens_fused_representation=patch_size**3,
        )
        self.multimodal_transformer = Transformer(
            transformer_basic_dims,
            depth,
            num_heads,
            mlp_dim,
        )
        self.multimodal_decode_conv = nn.Conv3d(
            transformer_basic_dims,
            basic_dims * 16 * num_modals,
            kernel_size=1,
            padding=0,
        )
        self.masker = MaskModal()

        if not mamba_skip:
            self.tokenize = nn.ModuleList(
                [
                    tokenizer_class(dims=8, num_modals=num_modals),
                    tokenizer_class(dims=16, num_modals=num_modals),
                    tokenizer_class(dims=32, num_modals=num_modals),
                    tokenizer_class(dims=64, num_modals=num_modals),
                    tokenizer_class(dims=512, num_modals=num_modals),
                ]
            )
            self.mamba_fusion_layers = nn.ModuleList(
                [
                    mamba_fusion_layer_class(dim=8, num_tokens_fused_representation=128**3),
                    mamba_fusion_layer_class(dim=16, num_tokens_fused_representation=64**3),
                    mamba_fusion_layer_class(dim=32, num_tokens_fused_representation=32**3),
                    mamba_fusion_layer_class(dim=64, num_tokens_fused_representation=16**3),
                    mamba_fusion_layer_class(dim=512, num_tokens_fused_representation=8**3),
                ]
            )
        else:
            self.tokenize = nn.ModuleList(
                [
                    tokenizer_class(dims=16, num_modals=num_modals),
                    tokenizer_class(dims=32, num_modals=num_modals),
                    tokenizer_class(dims=64, num_modals=num_modals),
                    tokenizer_class(dims=512, num_modals=num_modals),
                ]
            )
            self.mamba_fusion_layers = nn.ModuleList(
                [
                    mamba_fusion_layer_class(dim=16, num_tokens_fused_representation=64**3),
                    mamba_fusion_layer_class(dim=32, num_tokens_fused_representation=32**3),
                    mamba_fusion_layer_class(dim=64, num_tokens_fused_representation=16**3),
                    mamba_fusion_layer_class(dim=512, num_tokens_fused_representation=8**3),
                ]
            )

        self.decoder_fuse = Decoder_fuse(num_cls=num_cls, mamba_skip=mamba_skip)
        self.decoder_sep = Decoder_sep(num_cls=num_cls)
        self.is_training = False
        self.mamba_skip = mamba_skip
        self.apply(InitWeights_He(1e-2))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        x, mask = self._remap_input_order(x, mask)
        flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.flair_encoder(x[:, 0:1])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t1ce_encoder(x[:, 1:2])
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t1_encoder(x[:, 2:3])
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 3:4])

        flair_token_x5 = self.flair_encode_conv(flair_x5).permute(0, 2, 3, 4, 1).contiguous().view(
            x.size(0), -1, transformer_basic_dims
        )
        t1ce_token_x5 = self.t1ce_encode_conv(t1ce_x5).permute(0, 2, 3, 4, 1).contiguous().view(
            x.size(0), -1, transformer_basic_dims
        )
        t1_token_x5 = self.t1_encode_conv(t1_x5).permute(0, 2, 3, 4, 1).contiguous().view(
            x.size(0), -1, transformer_basic_dims
        )
        t2_token_x5 = self.t2_encode_conv(t2_x5).permute(0, 2, 3, 4, 1).contiguous().view(
            x.size(0), -1, transformer_basic_dims
        )

        flair_intra_token_x5 = self.flair_transformer(flair_token_x5, self.flair_pos)
        t1ce_intra_token_x5 = self.t1ce_transformer(t1ce_token_x5, self.t1ce_pos)
        t1_intra_token_x5 = self.t1_transformer(t1_token_x5, self.t1_pos)
        t2_intra_token_x5 = self.t2_transformer(t2_token_x5, self.t2_pos)

        flair_intra_x5 = flair_intra_token_x5.view(
            x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims
        ).permute(0, 4, 1, 2, 3).contiguous()
        t1ce_intra_x5 = t1ce_intra_token_x5.view(
            x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims
        ).permute(0, 4, 1, 2, 3).contiguous()
        t1_intra_x5 = t1_intra_token_x5.view(
            x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims
        ).permute(0, 4, 1, 2, 3).contiguous()
        t2_intra_x5 = t2_intra_token_x5.view(
            x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims
        ).permute(0, 4, 1, 2, 3).contiguous()

        if self.is_training:
            flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4, flair_x5)
            t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5)
            t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4, t1_x5)
            t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5)

        x1 = self.masker(torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1), mask)
        x2 = self.masker(torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1), mask)
        x3 = self.masker(torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1), mask)
        x4 = self.masker(torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1), mask)
        x5_intra = self.masker(
            torch.stack((flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5), dim=1),
            mask,
        )

        if self.mamba_skip:
            x2 = self.tokenize[-4](x2)
            x2 = self.mamba_fusion_layers[-4](x2)
            x2 = x2.view(
                x.size(0),
                input_patch_size // 2,
                input_patch_size // 2,
                input_patch_size // 2,
                basic_dims * 2,
            ).permute(0, 4, 1, 2, 3).contiguous()
            x3 = self.tokenize[-3](x3)
            x3 = self.mamba_fusion_layers[-3](x3)
            x3 = x3.view(
                x.size(0),
                input_patch_size // 4,
                input_patch_size // 4,
                input_patch_size // 4,
                basic_dims * 4,
            ).permute(0, 4, 1, 2, 3).contiguous()
            x4 = self.tokenize[-2](x4)
            x4 = self.mamba_fusion_layers[-2](x4)
            x4 = x4.view(
                x.size(0),
                input_patch_size // 8,
                input_patch_size // 8,
                input_patch_size // 8,
                basic_dims * 8,
            ).permute(0, 4, 1, 2, 3).contiguous()

        multimodal_token_x5 = self.tokenize[-1](x5_intra)
        fused_multimodal = self.mamba_fusion_layers[-1](multimodal_token_x5)
        multimodal_pos = self.fused_pos.repeat(x.size(0), 1, 1)
        multimodal_inter_token_x5 = self.multimodal_transformer(
            fused_multimodal,
            multimodal_pos,
        )
        multimodal_inter_x5 = self.multimodal_decode_conv(
            multimodal_inter_token_x5.view(
                multimodal_inter_token_x5.size(0),
                patch_size,
                patch_size,
                patch_size,
                transformer_basic_dims,
            ).permute(0, 4, 1, 2, 3).contiguous()
        )

        fuse_pred, preds = self.decoder_fuse(x1, x2, x3, x4, multimodal_inter_x5)
        if self.is_training:
            return fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), preds
        return fuse_pred

    def predict(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        images = self._normalize_predict_input(images)
        _, _, height, width, depth_size = images.shape
        if (height, width, depth_size) == (
            input_patch_size,
            input_patch_size,
            input_patch_size,
        ):
            return self(images, mask)

        h_starts = self._window_starts(height)
        w_starts = self._window_starts(width)
        d_starts = self._window_starts(depth_size)
        prediction = torch.zeros(
            images.size(0),
            self.decoder_fuse.seg_layer.out_channels,
            height,
            width,
            depth_size,
            device=images.device,
        )
        weight = torch.zeros(
            images.size(0),
            1,
            height,
            width,
            depth_size,
            device=images.device,
        )

        for h in h_starts:
            for w in w_starts:
                for d in d_starts:
                    patch = images[
                        :,
                        :,
                        h : h + input_patch_size,
                        w : w + input_patch_size,
                        d : d + input_patch_size,
                    ]
                    patch_pred = self(patch, mask)
                    prediction[
                        :,
                        :,
                        h : h + input_patch_size,
                        w : w + input_patch_size,
                        d : d + input_patch_size,
                    ] += patch_pred
                    weight[
                        :,
                        :,
                        h : h + input_patch_size,
                        w : w + input_patch_size,
                        d : d + input_patch_size,
                    ] += 1

        return prediction / weight

    @staticmethod
    def _window_starts(size: int) -> list[int]:
        if size < input_patch_size:
            raise RuntimeError(
                "Model prediction expects spatial dimensions to be at least "
                f"{input_patch_size}, got {size}"
            )
        if size == input_patch_size:
            return [0]

        stride = int(input_patch_size * (1 - 0.5))
        if stride <= 0:
            raise RuntimeError("Sliding-window stride must be positive")

        starts = list(range(0, max(size - input_patch_size, 0), stride))
        last_start = size - input_patch_size
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    @staticmethod
    def _normalize_predict_input(images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise RuntimeError(
                "Model prediction expects images with shape [B, C, H, W, D], "
                f"got {tuple(images.shape)}"
            )

        normalized = images.clone()
        for batch_index in range(normalized.size(0)):
            sample = normalized[batch_index]
            foreground_mask = sample.sum(dim=0) > 0
            if not torch.any(foreground_mask):
                continue

            for channel_index in range(sample.size(0)):
                modal = sample[channel_index]
                foreground = modal[foreground_mask]
                mean = foreground.mean()
                std = foreground.std(unbiased=False)

                if not torch.isfinite(std) or std == 0:
                    sample[channel_index] = torch.zeros_like(modal)
                    continue

                sample[channel_index] = (modal - mean) / std

        return normalized

    @staticmethod
    def _remap_input_order(
        images: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images.size(1) != num_modals:
            raise RuntimeError(
                f"IMFuse expects {num_modals} input modalities, got {images.size(1)}"
            )
        if mask.ndim != 2 or mask.size(1) != num_modals:
            raise RuntimeError(
                f"IMFuse expects mask shape [B, {num_modals}], got {tuple(mask.shape)}"
            )

        # External pipeline uses [t1c, t1n, t2f, t2w]; the model internals expect
        # [flair, t1ce, t1, t2], so remap both tensors at the boundary.
        return images[:, DATASET_MODALITY_ORDER, ...], mask[:, DATASET_MODALITY_ORDER]


Model = IMFuse


__all__ = ["IMFuse", "Model"]
