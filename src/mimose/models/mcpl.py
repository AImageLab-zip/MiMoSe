from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimose.models.abstract_model import AbstractModel
from mimose.models.region_utils import regions_to_label_one_hot

num_modals = 4
init_channels = 16
input_patch_size = 128
bottleneck_channels = 128


def trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, std=std)


class DropPath(nn.Module):
    """Per-sample stochastic depth, matching ``timm.layers.DropPath``."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def get_activation(name: str | None) -> nn.Module:
    if name is None or name.lower() == "none":
        return nn.Identity()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "sig":
        return nn.Sigmoid()
    raise ValueError(f"Unknown activation {name!r}")


def get_norm(name: str | None, channels: int) -> nn.Module:
    if name is None or name.lower() == "none":
        return nn.Identity()
    if name.lower() == "bn":
        return nn.BatchNorm3d(channels, eps=1e-3, momentum=0.01)
    if name.lower() == "1b":
        return nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)
    raise ValueError(f"Unknown norm {name!r}")


class ConvBN(nn.Module):
    """Matches legacy MCPL's ``ConvBN`` (kmax_pixel_decoder.py), restricted to
    the ``conv_type`` in {"1d", "3d"} actually used by ``Unet_missing``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = True,
        norm: str | None = None,
        act: str | None = None,
        conv_type: str = "3d",
        norm_init: float = 1.0,
        padding: int = 0,
        groups: int = 1,
        conv_init: str = "he_normal",
    ) -> None:
        super().__init__()
        conv_cls = nn.Conv1d if conv_type == "1d" else nn.Conv3d
        self.conv = conv_cls(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias
        )
        self.norm = get_norm(norm, out_channels)
        self.act = get_activation(act)
        if conv_init == "he_normal":
            trunc_normal_(self.conv.weight, std=math.sqrt(2.0 / in_channels))
        elif conv_init == "xavier_uniform":
            nn.init.xavier_uniform_(self.conv.weight)
        elif conv_init == "trunc_normal":
            trunc_normal_(self.conv.weight, std=0.02)
        else:
            raise ValueError(f"Unknown conv_init {conv_init!r}")
        if bias:
            nn.init.zeros_(self.conv.bias)
        if norm is not None:
            nn.init.constant_(self.norm.weight, norm_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class AttentionOperation(nn.Module):
    """Matches legacy MCPL's ``AttentionOperation`` (KamxT.py)."""

    def __init__(self, channels_v: int, num_heads: int) -> None:
        super().__init__()
        self._batch_norm_similarity = nn.BatchNorm2d(num_heads, eps=1e-3, momentum=0.01)
        self._batch_norm_retrieved_value = nn.BatchNorm1d(channels_v, eps=1e-3, momentum=0.01)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        n, _, _, length = query.shape
        _, num_heads, channels, _ = value.shape
        similarity_logits = torch.einsum("bhdl,bhdm->bhlm", query, key)
        similarity_logits = self._batch_norm_similarity(similarity_logits)
        attention_weights = F.softmax(similarity_logits.float(), dim=-1)
        retrieved_value = torch.einsum("bhlm,bhdm->bhdl", attention_weights, value)
        retrieved_value = retrieved_value.reshape(n, num_heads * channels, length)
        retrieved_value = self._batch_norm_retrieved_value(retrieved_value)
        return F.gelu(retrieved_value)


def _add_bias_towards_void(query_class_logits: torch.Tensor, void_prior_prob: float = 0.9) -> torch.Tensor:
    import math

    num_classes = query_class_logits.shape[-1]
    init_bias = [0.0] * num_classes
    init_bias[-1] = math.log((num_classes - 1) * void_prior_prob / (1 - void_prior_prob))
    return query_class_logits + torch.tensor(init_bias, dtype=query_class_logits.dtype, device=query_class_logits.device)


class KMaXPredictor(nn.Module):
    """Matches legacy MCPL's ``kMaXPredictor`` (KamxT.py)."""

    def __init__(self, in_channel_pixel: int, nums_q: int) -> None:
        super().__init__()
        self._pixel_space_head_conv0bnact = ConvBN(
            in_channel_pixel,
            in_channel_pixel,
            kernel_size=5,
            padding=2,
            groups=in_channel_pixel,
            bias=False,
            norm="bn",
            act="gelu",
            conv_init="xavier_uniform",
        )
        self._pixel_space_head_conv1bnact = ConvBN(in_channel_pixel, 256, kernel_size=1, bias=False, norm="bn", act="gelu")
        self._pixel_space_head_last_convbn = ConvBN(256, 128, kernel_size=1, bias=True, norm="bn", act=None)
        trunc_normal_(self._pixel_space_head_last_convbn.conv.weight, std=0.01)

        self._transformer_mask_head = ConvBN(256, 128, kernel_size=1, bias=False, norm="1b", act=None, conv_type="1d")
        trunc_normal_(self._transformer_mask_head.conv.weight, std=0.01)

        self._pixel_space_mask_batch_norm = get_norm("bn", nums_q)
        nn.init.constant_(self._pixel_space_mask_batch_norm.weight, 0.1)

    def forward(self, query_space: torch.Tensor, pixel_featureK: torch.Tensor) -> dict[str, torch.Tensor]:
        pixel_space_feature = self._pixel_space_head_conv0bnact(pixel_featureK)
        pixel_space_feature = self._pixel_space_head_conv1bnact(pixel_space_feature)
        pixel_space_feature = self._pixel_space_head_last_convbn(pixel_space_feature)
        pixel_space_normalized_feature = F.normalize(pixel_space_feature, p=2, dim=1)

        cluster_mask_kernel = _add_bias_towards_void(query_space)
        cluster_mask_kernel = self._transformer_mask_head(cluster_mask_kernel)
        mask_logits = torch.einsum("bchwd,bcn->bnhwd", pixel_space_normalized_feature, cluster_mask_kernel)
        mask_logits = self._pixel_space_mask_batch_norm(mask_logits)

        return {"mask_logits": mask_logits, "pixel_feature": pixel_space_normalized_feature}


class KMaXTransformerLayer(nn.Module):
    """Matches legacy MCPL's ``kMaXTransformerLayer`` (KamxT.py), a k-means
    cross-attention block that updates a small set of (modality x class)
    prototype queries against the U-Net bottleneck pixel features."""

    def __init__(
        self,
        nums_q: int,
        base_filters: int = 128,
        num_heads: int = 8,
        bottleneck_expansion: float = 2.0,
        key_expansion: float = 1.0,
        value_expansion: float = 2.0,
        drop_path_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self._num_heads = num_heads
        self._bottleneck_channels = int(round(base_filters * bottleneck_expansion))
        self._total_key_depth = int(round(base_filters * key_expansion))
        self._total_value_depth = int(round(base_filters * value_expansion))

        self.drop_path_kmeans = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()
        self.drop_path_attn = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()
        self.drop_path_ffn = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()

        initialization_std = self._bottleneck_channels**-0.5
        self._query_conv1_bn_act = ConvBN(
            base_filters, self._bottleneck_channels, kernel_size=1, bias=False, norm="1b", act="gelu", conv_type="1d"
        )
        self._query_conv2_bn_act = ConvBN(
            base_filters, self._bottleneck_channels, kernel_size=1, bias=False, norm="1b", act="gelu", conv_type="1d"
        )
        self._pixel_conv1_bn_actV = ConvBN(bottleneck_channels, self._bottleneck_channels, kernel_size=1, bias=False, norm="bn", act="gelu")
        self._pixel_conv1_bn_actK = ConvBN(bottleneck_channels, self._bottleneck_channels, kernel_size=1, bias=False, norm="bn", act="gelu")

        self._query_qkv_conv_bn = ConvBN(
            self._bottleneck_channels,
            self._total_key_depth * 2 + self._total_value_depth,
            kernel_size=1,
            bias=False,
            norm="1b",
            act=None,
            conv_type="1d",
        )
        trunc_normal_(self._query_qkv_conv_bn.conv.weight, std=initialization_std)

        self._pixel_v_conv_bn = ConvBN(self._bottleneck_channels, self._total_value_depth, kernel_size=1, bias=False, norm="bn", act=None)
        self._pixel_k_conv_bn = ConvBN(self._bottleneck_channels, self._total_value_depth, kernel_size=1, bias=False, norm="bn", act=None)
        trunc_normal_(self._pixel_v_conv_bn.conv.weight, std=initialization_std)

        self._query_self_attention = AttentionOperation(channels_v=self._total_value_depth, num_heads=num_heads)

        self._query_conv3_bn = ConvBN(
            self._total_value_depth, base_filters, kernel_size=1, bias=False, norm="1b", act=None, conv_type="1d", norm_init=0.0
        )
        self._query_ffn_conv1_bn_act = ConvBN(base_filters, 2048, kernel_size=1, bias=False, norm="1b", act="gelu", conv_type="1d")
        self._query_ffn_conv2_bn = ConvBN(2048, base_filters, kernel_size=1, bias=False, norm="1b", act=None, conv_type="1d", norm_init=0.0)

        self._predictor = KMaXPredictor(in_channel_pixel=self._bottleneck_channels, nums_q=nums_q)
        self._kmeans_query_batch_norm_retrieved_value = get_norm("1b", self._total_value_depth)
        self._kmeans_query_conv3_bn = ConvBN(
            self._total_value_depth, base_filters, kernel_size=1, bias=False, norm="1b", act=None, conv_type="1d", norm_init=0.0
        )

    def forward(
        self, pixel_feature: torch.Tensor, query_feature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n, _, h, w, d = pixel_feature.shape
        _, _, length = query_feature.shape

        pixel_spaceK = self._pixel_conv1_bn_actK(F.gelu(pixel_feature))
        pixel_spaceV = self._pixel_conv1_bn_actV(F.gelu(pixel_feature))
        query_space = self._query_conv1_bn_act(query_feature)

        pixel_value = self._pixel_v_conv_bn(pixel_spaceV).reshape(n, self._total_value_depth, h * w * d)
        pixel_key = self._pixel_k_conv_bn(pixel_spaceK)

        prediction_result = self._predictor(query_space, pixel_key)

        with torch.no_grad():
            clustering_result = prediction_result["mask_logits"].flatten(2).detach()
            index = clustering_result.max(1, keepdim=True)[1]
            clustering_result = torch.zeros_like(clustering_result, memory_format=torch.legacy_contiguous_format).scatter_(
                1, index, 1.0
            )

        kmeans_update = torch.einsum("blm,bdm->bdl", clustering_result.float(), pixel_value.float())
        kmeans_update = self._kmeans_query_batch_norm_retrieved_value(kmeans_update)
        kmeans_update = self._kmeans_query_conv3_bn(kmeans_update)
        query_feature = query_feature + self.drop_path_kmeans(kmeans_update)

        query_space2 = self._query_conv2_bn_act(query_feature)
        query_qkv = self._query_qkv_conv_bn(query_space2)
        query_q, query_k, query_v = torch.split(
            query_qkv, [self._total_key_depth, self._total_key_depth, self._total_value_depth], dim=1
        )
        query_q = query_q.reshape(n, self._num_heads, self._total_key_depth // self._num_heads, length)
        query_k = query_k.reshape(n, self._num_heads, self._total_key_depth // self._num_heads, length)
        query_v = query_v.reshape(n, self._num_heads, self._total_value_depth // self._num_heads, length)
        self_attn_update = self._query_self_attention(query_q, query_k, query_v)
        self_attn_update = self._query_conv3_bn(self_attn_update)
        query_feature = query_feature + self.drop_path_attn(self_attn_update)
        query_feature = F.gelu(query_feature)

        ffn_update = self._query_ffn_conv1_bn_act(query_feature)
        ffn_update = self._query_ffn_conv2_bn(ffn_update)
        query_feature = query_feature + self.drop_path_ffn(ffn_update)
        query_feature = F.gelu(query_feature)

        return query_feature, prediction_result["pixel_feature"]


class Mlp(nn.Module):
    """Matches legacy MCPL's ``Mlp`` (Unet_kmaxT.py), used for the row
    (modality) / col (class) prototype projection heads."""

    def __init__(self, dim_input: int, dim_hidden: int, dim_output: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim_input, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_output)
        self.act_fn = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.act_fn(self.fc1(x)))
        return self.dropout(self.fc2(x))


class BasicBlock(nn.Module):
    """Matches legacy MCPL's ``BasicBlock`` (Unet_kmaxT.py, encoder side:
    pre-activation GroupNorm residual block with equal in/out channels)."""

    def __init__(self, channels: int, n_groups: int = 8) -> None:
        super().__init__()
        self.gn1 = nn.GroupNorm(n_groups, channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(n_groups, channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(self.relu1(self.gn1(x)))
        out = self.conv2(self.relu2(self.gn2(out)))
        return out + residual


class MCPL(AbstractModel):
    """Implementation of MCPL introduced in [1].

    [1] "Multimodal Contrastive Prototype Learning for Resilient Brain Tumor
        Segmentation with Missing Modalities". J-BHI.

    Legacy MCPL shares its entire training harness (the ``limage`` learnable
    placeholder for missing modalities, the two-stage pretrain/finetune
    schedule, the dual-view consistency loss) with legacy M3AE -- MCPL's own
    ``pretrain.py`` even imports M3AE's model file by mistake. This port
    reuses M3AE's already-ported ``limage`` mechanism (see
    ``mimose.models.m3ae.M3AE``) rather than legacy MCPL's own more complex
    patch/modality-dropout token-masking machinery (``MaskEmbeeding2`` /
    ``ShuffleIndex_with_MDP``), since MiMoSe's dataset already supplies a
    per-sample missing-modality mask.

    MCPL's actual novel contribution over M3AE is replacing the plain conv
    bottleneck with a stack of k-means cross-attention transformer layers
    (``KMaXTransformerLayer``) that refine a small set of learned
    (modality x class) prototype queries against the U-Net bottleneck. A
    segmentation volume is then produced via voxel-to-class-prototype cosine
    similarity, and a reconstruction volume via voxel-to-modality-prototype
    similarity (the "resilience"/pretraining objective).

    Like legacy MCPL (``CC_classes = 3``), the class prototypes are the three BraTS regions
    ``[ET, TC, WT]``: each region's logit is the raw voxel-to-prototype cosine, trained with
    ``EDiceLoss`` (sigmoid inside, so a region is predicted where its cosine is positive). Only at
    the output boundary (``mode="segment"`` / ``predict``) are the thresholded regions turned into a
    4-class one-hot label volume, so the framework's argmax-based evaluation applies unchanged.
    """

    def __init__(
        self,
        num_cls: int = 4,
        cc_modalities: int = 4,
        cc_classes: int = 3,
        layers: int = 6,
        learnable_filler: bool = False,
        logit_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_cls = num_cls
        # Upstream fills missing modalities with a frozen all-zero `limage` (Unet_kmaxT.py:425).
        self.learnable_filler = learnable_filler
        # Multiplier on the [-1, 1] voxel-prototype cosines before the sigmoid (1.0 = upstream).
        self.logit_scale = float(logit_scale)
        self.cc_modalities = cc_modalities
        self.cc_classes = cc_classes
        self.nums_q = cc_modalities * cc_classes

        filler_shape = (1, num_modals, input_patch_size, input_patch_size, input_patch_size)
        if learnable_filler:
            self.limage = nn.Parameter(torch.randn(*filler_shape) * 0.02)
        else:
            self.limage = nn.Parameter(torch.zeros(*filler_shape), requires_grad=False)

        # Encoder (legacy UNet3D_g.make_encoder): strided convs for
        # downsampling (not maxpool), unlike M3AE's UNet3D.
        channels = init_channels
        self.conv1a = nn.Conv3d(num_modals, channels, kernel_size=3, padding=1)
        self.conv1b = BasicBlock(channels)
        self.ds1 = nn.Conv3d(channels, channels * 2, kernel_size=3, stride=2, padding=1)

        self.conv2a = BasicBlock(channels * 2)
        self.conv2b = BasicBlock(channels * 2)
        self.ds2 = nn.Conv3d(channels * 2, channels * 4, kernel_size=3, stride=2, padding=1)

        self.conv3a = BasicBlock(channels * 4)
        self.conv3b = BasicBlock(channels * 4)
        self.ds3 = nn.Conv3d(channels * 4, channels * 8, kernel_size=3, stride=2, padding=1)

        self.conv4a = BasicBlock(channels * 8)
        self.conv4b = BasicBlock(channels * 8)
        self.conv4c = BasicBlock(channels * 8)
        self.conv4d = BasicBlock(channels * 8)

        # kMaX prototype module (legacy UNet3D_g.make_KmaxT).
        self._cluster_centers = nn.Embedding(bottleneck_channels, self.nums_q)
        self._kmax_transformer_layers = nn.ModuleList(
            [KMaXTransformerLayer(nums_q=self.nums_q, drop_path_prob=0.2) for _ in range(layers)]
        )
        self._128_16 = ConvBN(bottleneck_channels, 16, kernel_size=1, bias=False, norm="1b", act="gelu", conv_type="1d")
        self._128_32 = ConvBN(bottleneck_channels, 32, kernel_size=1, bias=False, norm="1b", act="gelu", conv_type="1d")
        self._128_64 = ConvBN(bottleneck_channels, 64, kernel_size=1, bias=False, norm="1b", act="gelu", conv_type="1d")

        # Decoder (legacy UNet3D_g.make_decoder): starts from the kMaX-refined
        # pixel feature, with additive skips from the encoder features (c3/c2/c1).
        self.up4conva = nn.Conv3d(channels * 8, channels * 4, kernel_size=1)
        self.up4 = nn.Upsample(scale_factor=2)
        self.up4convb = BasicBlock(channels * 4)

        self.up3conva = nn.Conv3d(channels * 4, channels * 2, kernel_size=1)
        self.up3 = nn.Upsample(scale_factor=2)
        self.up3convb = BasicBlock(channels * 2)

        self.up2conva = nn.Conv3d(channels * 2, channels, kernel_size=1)
        self.up2 = nn.Upsample(scale_factor=2)
        self.up2convb = BasicBlock(channels)

        # Prototype projection heads (legacy Unet_missing.__init__).
        self.row = Mlp(self.nums_q, 128, cc_modalities)
        self.col = Mlp(self.nums_q, 128, cc_classes)
        self.col32 = Mlp(self.nums_q, 128, cc_classes)
        self.col64 = Mlp(self.nums_q, 128, cc_classes)

        # Modality-reconstruction head (legacy UNet3D_g.make_decoder.head_reg).
        self.head_reg = nn.Conv3d(channels * cc_modalities, num_modals, kernel_size=1)

        # Deep-supervision heads on the coarser decoder levels (u4 / u3),
        # matching legacy ``ds_out``/``up_out``.
        self.ds_up4 = nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)
        self.ds_up3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = self.conv1b(self.conv1a(x))
        c1d = self.ds1(c1)

        c2 = self.conv2b(self.conv2a(c1d))
        c2d = self.ds2(c2)

        c3 = self.conv3b(self.conv3a(c2d))
        c3d = self.ds3(c3)

        c4 = self.conv4b(self.conv4a(c3d))
        c4d = self.conv4d(self.conv4c(c4))

        return c1, c2, c3, c4d

    def _kmax_bottleneck(self, content: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs the kMaX transformer stack over the bottleneck, returning the
        refined query (prototype) features, shape ``[B, 128, nums_q]``."""
        batch_size = content.size(0)
        cluster_centers = self._cluster_centers.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        prediction_result = content
        for layer in self._kmax_transformer_layers:
            cluster_centers, prediction_result = layer(prediction_result, cluster_centers)
        return cluster_centers, prediction_result

    def _forward_trunk(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.size(1) != num_modals:
            raise RuntimeError(f"MCPL expects {num_modals} input modalities, got {x.size(1)}")
        voxel_gate = mask.ndim == 5
        if voxel_gate:
            if mask.shape[1] != num_modals or mask.shape[-3:] != x.shape[-3:]:
                raise RuntimeError(f"MCPL expects a voxel gate shaped like x, got {tuple(mask.shape)}")
        elif mask.ndim != 2 or mask.size(1) != num_modals:
            raise RuntimeError(f"MCPL expects mask shape [B, {num_modals}], got {tuple(mask.shape)}")
        if x.shape[-3:] != self.limage.shape[-3:]:
            raise RuntimeError(
                f"MCPL requires spatial dimensions of exactly {tuple(self.limage.shape[-3:])}, got {tuple(x.shape[-3:])}"
            )

        # A [B, M, D, H, W] gate is the MAE pretraining patch mask; a [B, M] mask drops whole modalities.
        gate = mask.to(x.dtype) if voxel_gate else mask.view(-1, num_modals, 1, 1, 1).to(x.dtype)
        x = x * gate + self.limage * (1 - gate)

        c1, c2, c3, c4d = self._encode(x)
        cluster_centers, pixel_feature = self._kmax_bottleneck(c4d)

        # Upstream decodes from the last kMaX pixel feature (Unet_kmaxT.py:274, "c4d->prediction_result").
        u4 = self.up4(self.up4conva(pixel_feature))
        u4 = self.up4convb(u4 + c3)

        u3 = self.up3(self.up3conva(u4))
        u3 = self.up3convb(u3 + c2)

        u2 = self.up2(self.up2conva(u3))
        u2 = self.up2convb(u2 + c1)

        return u2, u3, u4, cluster_centers

    def _segmentation_from_prototypes(self, cluster_centers: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
        cluster_centers16 = self._128_16(cluster_centers)
        cc_class = self.col(cluster_centers16)
        logits = torch.einsum(
            "bnc,bcxyz->bnxyz",
            F.normalize(cc_class, p=2, dim=1).transpose(1, 2),
            F.normalize(u2, p=2, dim=1),
        )
        return logits * self.logit_scale

    def _deep_supervision_from_prototypes(
        self, cluster_centers: torch.Tensor, u4: torch.Tensor, u3: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cluster_centers64 = self._128_64(cluster_centers)
        cc64_class = self.col64(cluster_centers64)
        cluster_centers32 = self._128_32(cluster_centers)
        cc32_class = self.col32(cluster_centers32)

        out4_logits = torch.einsum(
            "bnc,bcxyz->bnxyz",
            F.normalize(cc64_class, p=2, dim=1).transpose(1, 2),
            F.normalize(self.ds_up4(u4), p=2, dim=1),
        )
        out3_logits = torch.einsum(
            "bnc,bcxyz->bnxyz",
            F.normalize(cc32_class, p=2, dim=1).transpose(1, 2),
            F.normalize(self.ds_up3(u3), p=2, dim=1),
        )
        return out4_logits * self.logit_scale, out3_logits * self.logit_scale

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, mode: str = "segment"
    ) -> torch.Tensor | tuple[list[torch.Tensor], torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        """``mode="segment"`` (default) returns the 4-class one-hot label volume,
        ``mode="segment_logits"`` the ``[ET, TC, WT]`` region logits, and
        ``mode="segment_train"`` the deep-supervision region logits in upstream order plus the
        refined ``cluster_centers``: ``([out4, out3, uout], cluster_centers)``.
        ``mode="reconstruct"`` returns ``(reconstruction, cluster_centers)``
        (the latter needed for the modality-contrastive loss), reconstructing
        the original *unmasked* input from the masked one -- the pretraining
        objective.

        All modes go through this single ``forward`` (rather than separate
        public methods) so trainer code can always call
        ``self.model(x, mask, mode=...)`` and pick up
        ``DistributedDataParallel``'s gradient-sync hooks.
        """
        u2, u3, u4, cluster_centers = self._forward_trunk(x, mask)

        if mode == "reconstruct":
            cluster_centers16 = self._128_16(cluster_centers)
            cc_modal = self.row(cluster_centers16)
            batch_size = u2.size(0)
            u2_reconstruction = torch.einsum("bcn,bcxyz->bcnxyz", cc_modal.float(), u2.float()).reshape(
                batch_size, -1, *u2.shape[2:]
            )
            reconstruction = self.head_reg(u2_reconstruction)
            return reconstruction, cluster_centers

        uout = self._segmentation_from_prototypes(cluster_centers, u2)

        if mode == "segment_train":
            out4_logits, out3_logits = self._deep_supervision_from_prototypes(cluster_centers, u4, u3)
            return [out4_logits, out3_logits, uout], cluster_centers
        if mode == "segment_logits":
            return uout
        return regions_to_label_one_hot(uout, self.num_cls)

    def reconstruct(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Non-distributed convenience wrapper around
        ``forward(x, mask, mode="reconstruct")``. Trainer code should call
        ``self.model(x, mask, mode="reconstruct")`` directly instead, to stay
        DDP-safe (see ``forward``'s docstring)."""
        return self(x, mask, mode="reconstruct")

    def predict(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Sliding window (50% overlap) averaging the region logits, as upstream's MONAI
        ``sliding_window_inference`` does, then thresholded into labels."""
        _, _, height, width, depth = images.shape
        h_starts = self._window_starts(height)
        w_starts = self._window_starts(width)
        d_starts = self._window_starts(depth)
        logits = torch.zeros(images.size(0), self.cc_classes, height, width, depth, device=images.device)
        weight = torch.zeros(images.size(0), 1, height, width, depth, device=images.device)

        for h in h_starts:
            for w in w_starts:
                for d in d_starts:
                    window = (
                        slice(None),
                        slice(None),
                        slice(h, h + input_patch_size),
                        slice(w, w + input_patch_size),
                        slice(d, d + input_patch_size),
                    )
                    logits[window] += self(images[window], mask, mode="segment_logits").float()
                    weight[window] += 1
        return regions_to_label_one_hot(logits / weight, self.num_cls)

    @staticmethod
    def _window_starts(size: int) -> list[int]:
        if size < input_patch_size:
            raise RuntimeError(
                f"Model prediction expects spatial dimensions to be at least {input_patch_size}, got {size}"
            )
        if size == input_patch_size:
            return [0]

        stride = int(input_patch_size * 0.5)
        starts = list(range(0, max(size - input_patch_size, 0), stride))
        last_start = size - input_patch_size
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts


Model = MCPL


__all__ = ["MCPL", "Model"]
