from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from missingbench.models.abstract_model import AbstractModel

basic_dims = 16
mlp_dim = 4096
num_heads = 8
depth = 3
num_modals = 4
patch_size = 5
input_patch_size = 80

# External pipeline uses [t1c, t1n, t2f, t2w]; M2FTrans internals expect
# [flair, t1ce, t1, t2] (same convention as DCSeg/RFNet/IMFuse).
DATASET_MODALITY_ORDER = (2, 0, 1, 3)


def normalization(planes: int, norm: str = "in") -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm3d(planes)
    if norm == "gn":
        return nn.GroupNorm(4, planes)
    if norm == "in":
        return nn.InstanceNorm3d(planes)
    raise ValueError(f"normalization type {norm} is not supported")


class GeneralConv3d(nn.Module):
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


class PRMFusion(nn.Module):
    """Segmentation head applied directly to an already-fused feature map."""

    def __init__(self, in_channel: int = 64, num_cls: int = 4) -> None:
        super().__init__()
        self.prm_layer = nn.Sequential(
            GeneralConv3d(in_channel, 16, k_size=1, stride=1, padding=0),
            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prm_layer(x)


class FusionPostnorm(nn.Module):
    """Zero out missing modalities, concatenate, and fuse with plain convs."""

    def __init__(self, in_channel: int = 64, num_cls: int = 4) -> None:
        super().__init__()
        self.fusion_layer = nn.Sequential(
            GeneralConv3d(in_channel * num_cls, in_channel, k_size=1, padding=0, stride=1),
            GeneralConv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
            GeneralConv3d(in_channel, in_channel, k_size=1, padding=0, stride=1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, height, width, depth = x.size()
        masked = torch.zeros_like(x)
        masked[mask, ...] = x[mask, ...]
        return self.fusion_layer(masked.view(batch_size, -1, height, width, depth))


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.e1_c1 = GeneralConv3d(1, basic_dims)
        self.e1_c2 = GeneralConv3d(basic_dims, basic_dims)
        self.e1_c3 = GeneralConv3d(basic_dims, basic_dims)

        self.e2_c1 = GeneralConv3d(basic_dims, basic_dims * 2, stride=2)
        self.e2_c2 = GeneralConv3d(basic_dims * 2, basic_dims * 2)
        self.e2_c3 = GeneralConv3d(basic_dims * 2, basic_dims * 2)

        self.e3_c1 = GeneralConv3d(basic_dims * 2, basic_dims * 4, stride=2)
        self.e3_c2 = GeneralConv3d(basic_dims * 4, basic_dims * 4)
        self.e3_c3 = GeneralConv3d(basic_dims * 4, basic_dims * 4)

        self.e4_c1 = GeneralConv3d(basic_dims * 4, basic_dims * 8, stride=2)
        self.e4_c2 = GeneralConv3d(basic_dims * 8, basic_dims * 8)
        self.e4_c3 = GeneralConv3d(basic_dims * 8, basic_dims * 8)

        self.e5_c1 = GeneralConv3d(basic_dims * 8, basic_dims * 16, stride=2)
        self.e5_c2 = GeneralConv3d(basic_dims * 16, basic_dims * 16)
        self.e5_c3 = GeneralConv3d(basic_dims * 16, basic_dims * 16)

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
        self.d4_c1 = GeneralConv3d(basic_dims * 16, basic_dims * 8)
        self.d4_c2 = GeneralConv3d(basic_dims * 16, basic_dims * 8)
        self.d4_out = GeneralConv3d(basic_dims * 8, basic_dims * 8, k_size=1, padding=0)

        self.d3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d3_c1 = GeneralConv3d(basic_dims * 8, basic_dims * 4)
        self.d3_c2 = GeneralConv3d(basic_dims * 8, basic_dims * 4)
        self.d3_out = GeneralConv3d(basic_dims * 4, basic_dims * 4, k_size=1, padding=0)

        self.d2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d2_c1 = GeneralConv3d(basic_dims * 4, basic_dims * 2)
        self.d2_c2 = GeneralConv3d(basic_dims * 4, basic_dims * 2)
        self.d2_out = GeneralConv3d(basic_dims * 2, basic_dims * 2, k_size=1, padding=0)

        self.d1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d1_c1 = GeneralConv3d(basic_dims * 2, basic_dims)
        self.d1_c2 = GeneralConv3d(basic_dims * 2, basic_dims)
        self.d1_out = GeneralConv3d(basic_dims, basic_dims, k_size=1, padding=0)

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


def _nchwd2nlc2nchwd(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width, depth = x.shape
    x = x.flatten(2).transpose(1, 2)
    x = module(x)
    return x.transpose(1, 2).reshape(batch, channels, height, width, depth).contiguous()


class DepthWiseConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        mid_channels = in_channels
        layer_norm = partial(nn.LayerNorm, eps=1e-6)
        self.conv1 = nn.Conv3d(in_channels, mid_channels, 1, 1)
        self.norm1 = layer_norm(mid_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv3d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels)
        self.norm2 = layer_norm(mid_channels)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv3d(mid_channels, out_channels, 1, 1)
        self.norm3 = layer_norm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = _nchwd2nlc2nchwd(self.norm1, x)
        x = self.act1(x)

        x = self.conv2(x)
        x = _nchwd2nlc2nchwd(self.norm2, x)
        x = self.act2(x)

        x = self.conv3(x)
        return _nchwd2nlc2nchwd(self.norm3, x)


class GroupConvBlock(nn.Module):
    def __init__(self, embed_dims: int, expand_ratio: int = 4, proj_drop: float = 0.0) -> None:
        super().__init__()
        hidden = embed_dims * expand_ratio
        layer_norm = partial(nn.LayerNorm, eps=1e-6)
        self.pwconv1 = nn.Conv3d(embed_dims, hidden, 1, 1)
        self.norm1 = layer_norm(hidden)
        self.act1 = nn.GELU()
        self.dwconv = nn.Conv3d(hidden, hidden, 3, 1, 1, groups=embed_dims)
        self.norm2 = layer_norm(hidden)
        self.act2 = nn.GELU()
        self.pwconv2 = nn.Conv3d(hidden, embed_dims, 1, 1)
        self.norm3 = layer_norm(embed_dims)
        self.final_act = nn.GELU()
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        out = self.pwconv1(x)
        out = _nchwd2nlc2nchwd(self.norm1, out)
        out = self.act1(out)

        out = self.dwconv(out)
        out = _nchwd2nlc2nchwd(self.norm2, out)
        out = self.act2(out)

        out = self.pwconv2(out)
        out = _nchwd2nlc2nchwd(self.norm3, out)

        out = identity + self.proj_drop(out)
        return self.final_act(out)


def _mask_gen_cross4(query_len: int, key_len: int, mask: torch.Tensor) -> torch.Tensor:
    """True where a query token may attend to a key token, given per-modality availability."""
    batch = mask.size(0)
    chunk = key_len // num_modals
    allowed = torch.ones(batch, query_len, key_len, dtype=torch.bool, device=mask.device)
    for modal_index in range(num_modals):
        missing = mask[:, modal_index] == 0
        allowed[missing, :, chunk * modal_index : chunk * (modal_index + 1)] = False
    return allowed


class MultiMaskAttentionLayer(nn.Module):
    def __init__(self, kv_dim: int = basic_dims, query_dim: int = num_modals, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        self.attn_drop = nn.Dropout(attn_drop)
        self.query_map = DepthWiseConvBlock(query_dim, query_dim)
        self.key_map_flair = DepthWiseConvBlock(kv_dim, kv_dim)
        self.value_map_flair = DepthWiseConvBlock(kv_dim, kv_dim)
        self.key_map_t1ce = DepthWiseConvBlock(kv_dim, kv_dim)
        self.value_map_t1ce = DepthWiseConvBlock(kv_dim, kv_dim)
        self.key_map_t1 = DepthWiseConvBlock(kv_dim, kv_dim)
        self.value_map_t1 = DepthWiseConvBlock(kv_dim, kv_dim)
        self.key_map_t2 = DepthWiseConvBlock(kv_dim, kv_dim)
        self.value_map_t2 = DepthWiseConvBlock(kv_dim, kv_dim)
        self.out_project = DepthWiseConvBlock(query_dim, query_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        key: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        identity = query
        flair, t1ce, t1, t2 = key
        qb, qc, qh, qw, qd = query.shape
        query = self.query_map(query).flatten(2)

        key_flair = self.key_map_flair(flair).flatten(2)
        value_flair = self.value_map_flair(flair).flatten(2)
        key_t1ce = self.key_map_t1ce(t1ce).flatten(2)
        value_t1ce = self.value_map_t1ce(t1ce).flatten(2)
        key_t1 = self.key_map_t1(t1).flatten(2)
        value_t1 = self.value_map_t1(t1).flatten(2)
        key_t2 = self.key_map_t2(t2).flatten(2)
        value_t2 = self.value_map_t2(t2).flatten(2)

        key_cat = torch.cat((key_flair, key_t1ce, key_t1, key_t2), dim=1)
        value_cat = torch.cat((value_flair, value_t1ce, value_t1, value_t2), dim=1)

        attn = (query @ key_cat.transpose(-2, -1)) * (query.shape[-1] ** -0.5)
        allowed = _mask_gen_cross4(query.shape[1], key_cat.shape[1], mask)
        attn = attn.masked_fill(~allowed, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ value_cat
        out = out.reshape(qb, qc, qh, qw, qd)
        out = self.out_project(out)
        return identity + self.proj_drop(out)


class MultiMaskCrossBlock(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        num_classes: int,
        expand_ratio: int = 4,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        ffn_feature_maps: bool = True,
    ) -> None:
        super().__init__()
        self.ffn_feature_maps = ffn_feature_maps
        self.cross_attn = MultiMaskAttentionLayer(
            kv_dim=feature_channels,
            query_dim=num_classes,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
        )
        if ffn_feature_maps:
            self.ffn2 = GroupConvBlock(embed_dims=feature_channels, expand_ratio=expand_ratio)
        self.ffn1 = GroupConvBlock(embed_dims=num_classes, expand_ratio=expand_ratio)

    def forward(
        self,
        kernels: torch.Tensor,
        feature_maps: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        kernels = self.cross_attn(query=kernels, key=feature_maps, mask=mask)
        kernels = self.ffn1(kernels, identity=kernels)

        if self.ffn_feature_maps:
            feature_maps = tuple(self.ffn2(feat, identity=feat) for feat in feature_maps)

        return kernels, feature_maps


class MultiCrossToken(nn.Module):
    def __init__(
        self,
        num_layers: int = 2,
        mlp_ratio: int = 4,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.0,
        channel: int = basic_dims * 16,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MultiMaskCrossBlock(
                    feature_channels=channel,
                    num_classes=channel,
                    expand_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    ffn_feature_maps=layer_index != num_layers - 1,
                )
                for layer_index in range(num_layers)
            ]
        )

    def forward(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        kernels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        feature_maps = inputs
        for layer in self.layers:
            kernels, feature_maps = layer(kernels, feature_maps, mask)
        return kernels


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


def _mask_gen_fusion(num_heads: int, patches: int, num_class: int, mask: torch.Tensor) -> torch.Tensor:
    """Attention mask over [modality_0 .. modality_{K-1}, fusion] token blocks.

    Each modality block only attends to itself; the fusion block attends to
    every block, except columns belonging to missing modalities.
    """
    batch = mask.size(0)
    size = patches * (num_class + 1)
    allowed = torch.zeros(batch, size, size, dtype=torch.bool, device=mask.device)
    for i in range(num_class):
        allowed[:, patches * i : patches * (i + 1), patches * i : patches * (i + 1)] = True
    allowed[:, patches * num_class :, :] = True
    for i in range(num_class):
        missing = mask[:, i] == 0
        allowed[missing, patches * num_class :, patches * i : patches * (i + 1)] = False
    return allowed.unsqueeze(1).repeat(1, num_heads, 1, 1)


class MaskedAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout_rate: float = 0.0, num_class: int = 4) -> None:
        super().__init__()
        self.num_heads = heads
        self.scale = (dim // heads) ** -0.5
        self.num_class = num_class
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        allowed = _mask_gen_fusion(self.num_heads, tokens // (self.num_class + 1), self.num_class, mask)
        attn = attn.masked_fill(~allowed, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(batch, tokens, channels)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn


class MaskedTransformer(nn.Module):
    def __init__(self, embedding_dim: int, num_layers: int, heads: int, mlp_dim_: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.attn_norms = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_layers)])
        self.attn_drops = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_layers)])
        self.attentions = nn.ModuleList(
            [MaskedAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate) for _ in range(num_layers)]
        )
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_layers)])
        self.ffns = nn.ModuleList(
            [FeedForward(embedding_dim, mlp_dim_, dropout_rate) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        attn_list = []
        for layer_index in range(self.num_layers):
            attn_out, attn = self.attentions[layer_index](self.attn_norms[layer_index](x), mask)
            x = x + self.attn_drops[layer_index](attn_out)
            attn_list.append(attn.detach())
            x = x + self.ffns[layer_index](self.ffn_norms[layer_index](x))
        return x, attn_list


class Bottleneck(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trans_bottle = MaskedTransformer(
            embedding_dim=basic_dims * 16,
            num_layers=depth,
            heads=num_heads,
            mlp_dim_=mlp_dim,
        )
        self.num_cls = num_modals

    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
        fusion: torch.Tensor,
        pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        flair, t1ce, t1, t2 = x
        embed_flair = flair.flatten(2).transpose(1, 2).contiguous()
        embed_t1ce = t1ce.flatten(2).transpose(1, 2).contiguous()
        embed_t1 = t1.flatten(2).transpose(1, 2).contiguous()
        embed_t2 = t2.flatten(2).transpose(1, 2).contiguous()

        embed_cat = torch.cat((embed_flair, embed_t1ce, embed_t1, embed_t2, fusion), dim=1)
        embed_cat = embed_cat + pos
        embed_cat_trans, attn = self.trans_bottle(embed_cat, mask)
        flair_trans, t1ce_trans, t1_trans, t2_trans, fusion_trans = torch.chunk(
            embed_cat_trans, self.num_cls + 1, dim=1
        )
        return flair_trans, t1ce_trans, t1_trans, t2_trans, fusion_trans, attn


class WeightAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(
        self,
        de_x1: tuple[torch.Tensor, ...],
        de_x2: tuple[torch.Tensor, ...],
        de_x3: tuple[torch.Tensor, ...],
        de_x4: tuple[torch.Tensor, ...],
        de_x5: tuple[torch.Tensor, ...],
        attn: list[torch.Tensor],
    ) -> tuple[tuple, tuple, tuple, tuple, tuple]:
        flair_tra, t1ce_tra, t1_tra, t2_tra = de_x5
        flair_x4, t1ce_x4, t1_x4, t2_x4 = de_x4
        flair_x3, t1ce_x3, t1_x3, t2_x3 = de_x3
        flair_x2, t1ce_x2, t1_x2, t2_x2 = de_x2
        flair_x1, t1ce_x1, t1_x1, t2_x1 = de_x1

        attn_0 = attn[0]
        attn_fusion = attn_0[:, :, (patch_size**3) * num_modals :, :]
        attn_flair, attn_t1ce, attn_t1, attn_t2, _attn_self = torch.chunk(attn_fusion, num_modals + 1, dim=-1)

        def _pool(attn_block: torch.Tensor) -> torch.Tensor:
            return (
                torch.sum(torch.sum(attn_block, dim=1), dim=-2)
                .reshape(flair_tra.size(0), patch_size, patch_size, patch_size)
                .unsqueeze(dim=1)
            )

        attn_flair, attn_t1ce, attn_t1, attn_t2 = _pool(attn_flair), _pool(attn_t1ce), _pool(attn_t1), _pool(attn_t2)
        dex5 = (flair_tra * attn_flair, t1ce_tra * attn_t1ce, t1_tra * attn_t1, t2_tra * attn_t2)

        attn_flair, attn_t1ce, attn_t1, attn_t2 = (
            self.upsample(attn_flair),
            self.upsample(attn_t1ce),
            self.upsample(attn_t1),
            self.upsample(attn_t2),
        )
        dex4 = (flair_x4 * attn_flair, t1ce_x4 * attn_t1ce, t1_x4 * attn_t1, t2_x4 * attn_t2)

        attn_flair, attn_t1ce, attn_t1, attn_t2 = (
            self.upsample(attn_flair),
            self.upsample(attn_t1ce),
            self.upsample(attn_t1),
            self.upsample(attn_t2),
        )
        dex3 = (flair_x3 * attn_flair, t1ce_x3 * attn_t1ce, t1_x3 * attn_t1, t2_x3 * attn_t2)

        attn_flair, attn_t1ce, attn_t1, attn_t2 = (
            self.upsample(attn_flair),
            self.upsample(attn_t1ce),
            self.upsample(attn_t1),
            self.upsample(attn_t2),
        )
        dex2 = (flair_x2 * attn_flair, t1ce_x2 * attn_t1ce, t1_x2 * attn_t1, t2_x2 * attn_t2)

        attn_flair, attn_t1ce, attn_t1, attn_t2 = (
            self.upsample(attn_flair),
            self.upsample(attn_t1ce),
            self.upsample(attn_t1),
            self.upsample(attn_t2),
        )
        dex1 = (flair_x1 * attn_flair, t1ce_x1 * attn_t1ce, t1_x1 * attn_t1, t2_x1 * attn_t2)

        return dex1, dex2, dex3, dex4, dex5


class DecoderFusion(nn.Module):
    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.d5_c2 = GeneralConv3d(basic_dims * 32, basic_dims * 16)
        self.d5_out = GeneralConv3d(basic_dims * 16, basic_dims * 16, k_size=1, padding=0)

        self.ct5 = MultiCrossToken(channel=basic_dims * 16)
        self.ct4 = MultiCrossToken(channel=basic_dims * 8)

        self.d4_c1 = GeneralConv3d(basic_dims * 16, basic_dims * 8)
        self.d4_c2 = GeneralConv3d(basic_dims * 16, basic_dims * 8)
        self.d4_out = GeneralConv3d(basic_dims * 8, basic_dims * 8, k_size=1, padding=0)

        self.d3_c1 = GeneralConv3d(basic_dims * 8, basic_dims * 4)
        self.d3_c2 = GeneralConv3d(basic_dims * 8, basic_dims * 4)
        self.d3_out = GeneralConv3d(basic_dims * 4, basic_dims * 4, k_size=1, padding=0)

        self.d2_c1 = GeneralConv3d(basic_dims * 4, basic_dims * 2)
        self.d2_c2 = GeneralConv3d(basic_dims * 4, basic_dims * 2)
        self.d2_out = GeneralConv3d(basic_dims * 2, basic_dims * 2, k_size=1, padding=0)

        self.d1_c1 = GeneralConv3d(basic_dims * 2, basic_dims)
        self.d1_c2 = GeneralConv3d(basic_dims * 2, basic_dims)
        self.d1_out = GeneralConv3d(basic_dims, basic_dims, k_size=1, padding=0)

        self.seg_layer = nn.Conv3d(basic_dims, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode="trilinear", align_corners=True)

        self.rfm3 = FusionPostnorm(in_channel=basic_dims * 4, num_cls=num_cls)
        self.rfm2 = FusionPostnorm(in_channel=basic_dims * 2, num_cls=num_cls)
        self.rfm1 = FusionPostnorm(in_channel=basic_dims * 1, num_cls=num_cls)

        self.prm_fusion5 = PRMFusion(in_channel=basic_dims * 16, num_cls=num_cls)
        self.prm_fusion4 = PRMFusion(in_channel=basic_dims * 8, num_cls=num_cls)
        self.prm_fusion3 = PRMFusion(in_channel=basic_dims * 4, num_cls=num_cls)
        self.prm_fusion2 = PRMFusion(in_channel=basic_dims * 2, num_cls=num_cls)
        self.prm_fusion1 = PRMFusion(in_channel=basic_dims * 1, num_cls=num_cls)

    def forward(
        self,
        dx1: torch.Tensor,
        dx2: torch.Tensor,
        dx3: torch.Tensor,
        dx4: tuple[torch.Tensor, ...],
        dx5: tuple[torch.Tensor, ...],
        fusion: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        prm_pred5 = self.prm_fusion5(fusion)
        de_x5 = self.ct5(dx5, fusion, mask)
        de_x5 = torch.cat((de_x5, fusion), dim=1)
        de_x5 = self.d5_out(self.d5_c2(de_x5))
        de_x5 = self.d4_c1(self.up2(de_x5))

        prm_pred4 = self.prm_fusion4(de_x5)
        de_x4 = self.ct4(dx4, de_x5, mask)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))

        prm_pred3 = self.prm_fusion3(de_x4)
        de_x3 = self.rfm3(dx3, mask)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        prm_pred2 = self.prm_fusion2(de_x3)
        de_x2 = self.rfm2(dx2, mask)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        prm_pred1 = self.prm_fusion1(de_x2)
        de_x1 = self.rfm1(dx1, mask)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred, (
            prm_pred1,
            self.up2(prm_pred2),
            self.up4(prm_pred3),
            self.up8(prm_pred4),
            self.up16(prm_pred5),
        )


class M2FTrans(AbstractModel):
    """Implementation of M2FTrans introduced in [1].

    [1] Shi, et al. "Multi-Modal Fusion Transformer for Missing Modalities
        Brain Tumor Segmentation". ISBI 2024.
    """

    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()
        self.bottleneck = Bottleneck()
        self.decoder_fusion = DecoderFusion(num_cls=num_cls)
        self.decoder_sep = DecoderSep(num_cls=num_cls)
        self.weight_attention = WeightAttention()

        self.pos = nn.Parameter(torch.zeros(1, (patch_size**3) * (num_modals + 1), basic_dims * 16))
        self.fusion = nn.Parameter(
            nn.init.normal_(torch.zeros(1, patch_size**3, basic_dims * 16), mean=0.0, std=1.0)
        )

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

        flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.flair_encoder(x[:, 0:1])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t1ce_encoder(x[:, 1:2])
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t1_encoder(x[:, 2:3])
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 3:4])

        x_bottle = (flair_x5, t1ce_x5, t1_x5, t2_x5)

        batch_size = x.size(0)
        fusion = self.fusion.tile([batch_size, 1, 1])

        flair_trans, t1ce_trans, t1_trans, t2_trans, fusion_trans, attn = self.bottleneck(
            x_bottle, mask, fusion, self.pos
        )

        def _unflatten(tokens: torch.Tensor) -> torch.Tensor:
            return (
                tokens.view(x.size(0), patch_size, patch_size, patch_size, basic_dims * 16)
                .permute(0, 4, 1, 2, 3)
                .contiguous()
            )

        de_x5 = (_unflatten(flair_trans), _unflatten(t1ce_trans), _unflatten(t1_trans), _unflatten(t2_trans))
        fusion_tra = _unflatten(fusion_trans)
        de_x4 = (flair_x4, t1ce_x4, t1_x4, t2_x4)
        de_x3 = (flair_x3, t1ce_x3, t1_x3, t2_x3)
        de_x2 = (flair_x2, t1ce_x2, t1_x2, t2_x2)
        de_x1 = (flair_x1, t1ce_x1, t1_x1, t2_x1)

        de_x1, de_x2, de_x3, de_x4, de_x5 = self.weight_attention(de_x1, de_x2, de_x3, de_x4, de_x5, attn)

        de_x3 = torch.stack(de_x3, dim=1)
        de_x2 = torch.stack(de_x2, dim=1)
        de_x1 = torch.stack(de_x1, dim=1)

        fuse_pred, prm_preds = self.decoder_fusion(de_x1, de_x2, de_x3, de_x4, de_x5, fusion_tra, mask)

        if not self.is_training:
            return fuse_pred

        flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4, flair_x5)
        t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5)
        t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4, t1_x5)
        t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5)
        return fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), prm_preds

    def predict(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, _, height, width, depth = images.shape
        if (height, width, depth) == (input_patch_size, input_patch_size, input_patch_size):
            return self(images, mask)

        h_starts = self._window_starts(height)
        w_starts = self._window_starts(width)
        d_starts = self._window_starts(depth)
        prediction = torch.zeros(
            images.size(0),
            self.decoder_fusion.seg_layer.out_channels,
            height,
            width,
            depth,
            device=images.device,
        )
        weight = torch.zeros(images.size(0), 1, height, width, depth, device=images.device)

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
            raise RuntimeError(f"M2FTrans expects {num_modals} input modalities, got {images.size(1)}")
        if mask.ndim != 2 or mask.size(1) != num_modals:
            raise RuntimeError(f"M2FTrans expects mask shape [B, {num_modals}], got {tuple(mask.shape)}")

        return images[:, DATASET_MODALITY_ORDER, ...], mask[:, DATASET_MODALITY_ORDER]


Model = M2FTrans


__all__ = ["M2FTrans", "Model"]
