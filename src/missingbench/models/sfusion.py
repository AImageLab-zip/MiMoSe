from __future__ import annotations

import math

import torch
import torch.nn as nn

from missingbench.models.abstract_model import AbstractModel

MODALITIES = ["Flair", "T1c", "T1", "T2"]

n_base_ch_se = 32
levels = 4
num_modals = 4
patch_dim = 8
input_patch_size = 128

# External pipeline uses [t1c, t1n, t2f, t2w]; SFusion internals expect
# [flair, t1ce, t1, t2] (same convention as DCSeg/RFNet/RobustSeg).
DATASET_MODALITY_ORDER = (2, 0, 1, 3)


def normalization(planes: int, norm: str = "in") -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm3d(planes)
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
        drop_rate: float = 0.0,
        norm: bool = True,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k_size, stride=stride, padding=k_size // 2, bias=True)
        self.drop = nn.Dropout3d(p=drop_rate) if drop_rate > 0 else None
        self.norm = normalization(out_ch) if norm else None
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True) if act else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.drop is not None:
            x = self.drop(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Linear(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_ch, out_ch, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.view(x.shape[0], -1))


class StyleEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_0 = GeneralConv3d(1, n_base_ch_se, k_size=7, stride=1, norm=False)
        self.c_1 = GeneralConv3d(n_base_ch_se, n_base_ch_se * 2, k_size=4, stride=2, norm=False)
        self.c_2 = GeneralConv3d(n_base_ch_se * 2, n_base_ch_se * 4, k_size=4, stride=2, norm=False)
        self.c_3 = GeneralConv3d(n_base_ch_se * 4, n_base_ch_se * 4, k_size=4, stride=2, norm=False)
        self.c_4 = GeneralConv3d(n_base_ch_se * 4, n_base_ch_se * 4, k_size=4, stride=2, norm=False)
        self.se_logit = GeneralConv3d(n_base_ch_se * 4, 8, k_size=1, stride=1, norm=False, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_0(x)
        x = self.c_1(x)
        x = self.c_2(x)
        x = self.c_3(x)
        x = self.c_4(x)
        x = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        return self.se_logit(x)


class ContentEncoder(nn.Module):
    def __init__(self, n_base_filters: int) -> None:
        super().__init__()
        self.e1_c1 = GeneralConv3d(1, n_base_filters)
        self.e1_c2 = GeneralConv3d(n_base_filters, n_base_filters, drop_rate=0.3)
        self.e1_c3 = GeneralConv3d(n_base_filters, n_base_filters)

        self.e2_c1 = GeneralConv3d(n_base_filters, n_base_filters * 2, stride=2)
        self.e2_c2 = GeneralConv3d(n_base_filters * 2, n_base_filters * 2, drop_rate=0.3)
        self.e2_c3 = GeneralConv3d(n_base_filters * 2, n_base_filters * 2)

        self.e3_c1 = GeneralConv3d(n_base_filters * 2, n_base_filters * 4, stride=2)
        self.e3_c2 = GeneralConv3d(n_base_filters * 4, n_base_filters * 4, drop_rate=0.3)
        self.e3_c3 = GeneralConv3d(n_base_filters * 4, n_base_filters * 4)

        self.e4_c1 = GeneralConv3d(n_base_filters * 4, n_base_filters * 8, stride=2)
        self.e4_c2 = GeneralConv3d(n_base_filters * 8, n_base_filters * 8, drop_rate=0.3)
        self.e4_c3 = GeneralConv3d(n_base_filters * 8, n_base_filters * 8)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e1_c1 = self.e1_c1(x)
        e1_out = e1_c1 + self.e1_c3(self.e1_c2(e1_c1))

        e2_c1 = self.e2_c1(e1_out)
        e2_out = e2_c1 + self.e2_c3(self.e2_c2(e2_c1))

        e3_c1 = self.e3_c1(e2_out)
        e3_out = e3_c1 + self.e3_c3(self.e3_c2(e3_c1))

        e4_c1 = self.e4_c1(e3_out)
        e4_out = e4_c1 + self.e4_c3(self.e4_c2(e4_c1))

        return {"s1": e1_out, "s2": e2_out, "s3": e3_out, "s4": e4_out}


def _adaptive_instance_norm(content: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    c_mean = torch.mean(content, dim=(2, 3, 4), keepdim=True)
    c_std = torch.std(content, dim=(2, 3, 4), keepdim=True)
    return gamma * ((content - c_mean) / c_std) + beta


class AdaptiveResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = GeneralConv3d(channels, channels, norm=False, act=False)
        self.conv2 = GeneralConv3d(channels, channels, norm=False, act=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x_init: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x = _adaptive_instance_norm(self.conv1(x_init), mu, sigma)
        x = self.lrelu(x)
        x = _adaptive_instance_norm(self.conv2(x), mu, sigma)
        return x + x_init


class Mlp(nn.Module):
    def __init__(self, channel: int) -> None:
        super().__init__()
        self.channel = channel
        self.linear_0 = Linear(8, channel)
        self.linear_1 = Linear(channel, channel)
        self.get_mu = Linear(channel, channel)
        self.get_sigma = Linear(channel, channel)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, style: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.lrelu(self.linear_0(style))
        x = self.lrelu(self.linear_1(x))
        mu = self.get_mu(x).view(-1, self.channel, 1, 1, 1)
        sigma = self.get_sigma(x).view(-1, self.channel, 1, 1, 1)
        return mu, sigma


class ImageDecoder(nn.Module):
    """Style-conditioned reconstruction decoder. Uses InstanceNorm (not a
    fixed-shape LayerNorm), so unlike RobustSeg's decoder it works for any
    input resolution, not just the training patch size."""

    def __init__(self, input_channel: int, channel: int, img_ch: int = 1, scale: int = levels) -> None:
        super().__init__()
        if input_channel != channel:
            raise ValueError("ImageDecoder requires input_channel == channel (the bottleneck feeds AdaptiveResBlock residuals directly)")
        self.mlp = Mlp(channel)
        self.res_0 = AdaptiveResBlock(channel)
        self.res_1 = AdaptiveResBlock(channel)
        self.res_2 = AdaptiveResBlock(channel)
        self.res_3 = AdaptiveResBlock(channel)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)

        stages = []
        in_channel = channel
        out_channel = channel
        for _ in range(scale - 1):
            out_channel = in_channel // 2
            stages.append(
                nn.ModuleDict(
                    {
                        "conv": GeneralConv3d(in_channel, out_channel, k_size=5, norm=False, act=False),
                        "norm": nn.InstanceNorm3d(out_channel),
                    }
                )
            )
            in_channel = out_channel
        self.stages = nn.ModuleList(stages)
        self.conv_final = GeneralConv3d(out_channel, img_ch, k_size=7, norm=False, act=False)

    def forward(self, style: torch.Tensor, content: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, sigma = self.mlp(style)
        x = self.res_0(content, mu, sigma)
        x = self.res_1(x, mu, sigma)
        x = self.res_2(x, mu, sigma)
        x = self.res_3(x, mu, sigma)

        for stage in self.stages:
            x = self.upsample(x)
            x = stage["conv"](x)
            x = stage["norm"](x)
            x = self.lrelu(x)

        return self.conv_final(x), mu, sigma


class MaskDecoder(nn.Module):
    def __init__(self, input_channel: int, n_base_filters: int, num_cls: int = 4) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)

        in_channel = input_channel
        out_channel = n_base_filters * 4
        stages = []
        for _ in range(3):
            stages.append(
                nn.ModuleDict(
                    {
                        "conv1": GeneralConv3d(in_channel, out_channel),
                        "conv2": GeneralConv3d(out_channel * 2, out_channel),
                        "conv3": GeneralConv3d(out_channel, out_channel, k_size=1),
                    }
                )
            )
            in_channel = out_channel
            out_channel = out_channel // 2
        self.stages = nn.ModuleList(stages)

        self.conv_seg = GeneralConv3d(in_channel, num_cls, k_size=1, norm=False, act=False)
        self.seg_pred = nn.Softmax(dim=1)

    def forward(
        self,
        e1_out: torch.Tensor,
        e2_out: torch.Tensor,
        e3_out: torch.Tensor,
        e4_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        skips = (e3_out, e2_out, e1_out)
        out = e4_out
        for stage, skip in zip(self.stages, skips):
            out = self.upsample(out)
            out = stage["conv1"](out)
            out = torch.cat([out, skip], dim=1)
            out = stage["conv2"](out)
            out = stage["conv3"](out)

        seg_logit = self.conv_seg(out)
        seg_pred = self.seg_pred(seg_logit)
        return seg_pred, seg_logit


class DUpsampling3d(nn.Module):
    """Depth-to-space upsampling: expands channels then reshapes into space."""

    def __init__(self, channels: int, scale: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels * (scale**3), kernel_size=1, stride=1, bias=False)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        batch, channels, depth, height, width = x.size()
        scale = self.scale

        x = x.permute(0, 4, 3, 2, 1).contiguous().view(batch, width, height, depth * scale, channels // scale)
        x = x.permute(0, 3, 1, 2, 4).contiguous().view(batch, depth * scale, width, height * scale, channels // (scale**2))
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(batch, depth * scale, height * scale, width * scale, channels // (scale**3))
        return x.permute(0, 4, 1, 2, 3)


def _sinusoid_positional_encoding(num_positions: int, dim: int) -> torch.Tensor:
    position = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    table = torch.zeros(num_positions, dim)
    table[:, 0::2] = torch.sin(position * div_term)
    table[:, 1::2] = torch.cos(position * div_term[: table[:, 1::2].shape[1]])
    return table.unsqueeze(0)


class TransformerFusion(nn.Module):
    """Cross-modality fusion via a self-attention transformer over per-modality
    tokens, followed by a softmax-weighted combination across modalities.

    Adapted from legacy SFusion's ``TF_3D``: the legacy version builds a
    variable-length token sequence containing only the *present* modalities
    (assuming one shared missing-modality pattern per batch) and fuses their
    tokens with a plain ``nn.TransformerEncoder``. This version always builds
    a fixed-length 4-modality sequence and instead excludes missing
    modalities per-sample via ``src_key_padding_mask`` (so they don't
    participate in self-attention) and via a masked softmax in the final
    weighted combination (so they get zero weight) — this supports the
    per-sample ``[B, 4]`` masks used throughout MissingBench (legacy assumed a
    single mask for the whole batch) while being mathematically equivalent
    to the legacy behavior whenever a batch does share one mask.
    """

    def __init__(self, embedding_dim: int, volume_size: int, num_heads: int = 4, num_layers: int = 8) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.patch_dim = patch_dim
        self.tokens_per_modal = patch_dim**3
        self.scale = volume_size // patch_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=embedding_dim * 4,
        )
        self.fusion_block = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(p=0.1)
        self.avgpool = nn.AdaptiveAvgPool3d((patch_dim, patch_dim, patch_dim))
        self.upsample = DUpsampling3d(embedding_dim, self.scale)
        self.register_buffer(
            "pos_table",
            _sinusoid_positional_encoding(self.tokens_per_modal * num_modals, embedding_dim),
            persistent=False,
        )

    def _project(self, content: list[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        tokens = []
        for modal_index in range(num_modals):
            gate = mask[:, modal_index].view(-1, 1, 1, 1, 1)
            pooled = self.avgpool(content[modal_index] * gate)
            pooled = pooled.permute(0, 2, 3, 4, 1).contiguous().view(pooled.size(0), -1, self.embedding_dim)
            tokens.append(pooled)
        return torch.cat(tokens, dim=1)

    def _padding_mask(self, mask: torch.Tensor) -> torch.Tensor:
        return (~mask).repeat_interleave(self.tokens_per_modal, dim=1)

    def _reproject(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        chunks = torch.chunk(fused_tokens, num_modals, dim=1)
        maps = []
        for chunk in chunks:
            reshaped = chunk.view(
                chunk.size(0), self.patch_dim, self.patch_dim, self.patch_dim, self.embedding_dim
            ).permute(0, 4, 1, 2, 3).contiguous()
            maps.append(self.upsample(reshaped))
        return torch.stack(maps, dim=0)

    def forward(self, content: list[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        tokens = self._project(content, mask)
        tokens = self.dropout(tokens + self.pos_table[:, : tokens.size(1)].to(tokens.dtype))
        fused_tokens = self.fusion_block(tokens, src_key_padding_mask=self._padding_mask(mask))

        attn_map = self._reproject(fused_tokens)
        gate = mask.permute(1, 0).view(num_modals, mask.size(0), 1, 1, 1, 1)
        attn_map = attn_map.masked_fill(~gate, float("-inf"))
        attn_map = torch.softmax(attn_map, dim=0)

        output = content[0] * attn_map[0]
        for modal_index in range(1, num_modals):
            output = output + content[modal_index] * attn_map[modal_index]
        return output


class SFusion(AbstractModel):
    """Implementation of SFusion introduced in [1].

    [1] Style-content dual encoder with a self-attention transformer fusion
        module ("SF-FDGF"), replacing the gated-fusion (GFF) module used by
        related RobustSeg-family models with a per-scale
        ``nn.TransformerEncoder``-based fusion.
    """

    def __init__(self, num_cls: int = 4, n_base_filters: int = 16) -> None:
        super().__init__()
        self.num_cls = num_cls
        mlp_ch = n_base_filters * (2 ** (levels - 1))

        self.se_flair = StyleEncoder()
        self.se_t1c = StyleEncoder()
        self.se_t1 = StyleEncoder()
        self.se_t2 = StyleEncoder()

        self.ce_flair = ContentEncoder(n_base_filters=n_base_filters)
        self.ce_t1c = ContentEncoder(n_base_filters=n_base_filters)
        self.ce_t1 = ContentEncoder(n_base_filters=n_base_filters)
        self.ce_t2 = ContentEncoder(n_base_filters=n_base_filters)

        self.fusion1 = TransformerFusion(embedding_dim=n_base_filters, volume_size=input_patch_size)
        self.fusion2 = TransformerFusion(embedding_dim=n_base_filters * 2, volume_size=input_patch_size // 2)
        self.fusion3 = TransformerFusion(embedding_dim=n_base_filters * 4, volume_size=input_patch_size // 4)
        self.fusion4 = TransformerFusion(embedding_dim=n_base_filters * 8, volume_size=input_patch_size // 8)

        self.image_de_flair = ImageDecoder(input_channel=n_base_filters * 8, channel=mlp_ch)
        self.image_de_t1c = ImageDecoder(input_channel=n_base_filters * 8, channel=mlp_ch)
        self.image_de_t1 = ImageDecoder(input_channel=n_base_filters * 8, channel=mlp_ch)
        self.image_de_t2 = ImageDecoder(input_channel=n_base_filters * 8, channel=mlp_ch)

        self.mask_de = MaskDecoder(input_channel=n_base_filters * 8, n_base_filters=n_base_filters, num_cls=num_cls)

        self.is_training = False

        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                torch.nn.init.kaiming_normal_(module.weight)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        x, mask = self._remap_input_order(x, mask)

        style = {
            "Flair": self.se_flair(x[:, 0:1]),
            "T1c": self.se_t1c(x[:, 1:2]),
            "T1": self.se_t1(x[:, 2:3]),
            "T2": self.se_t2(x[:, 3:4]),
        }
        content = {
            "Flair": self.ce_flair(x[:, 0:1]),
            "T1c": self.ce_t1c(x[:, 1:2]),
            "T1": self.ce_t1(x[:, 2:3]),
            "T2": self.ce_t2(x[:, 3:4]),
        }

        fused = {}
        fusion_modules = (self.fusion1, self.fusion2, self.fusion3, self.fusion4)
        for level, fusion_module in enumerate(fusion_modules, start=1):
            scale_key = f"s{level}"
            per_modal_content = [content[modality][scale_key] for modality in MODALITIES]
            fused[scale_key] = fusion_module(per_modal_content, mask)

        if not self.is_training:
            seg_pred, _ = self.mask_de(fused["s1"], fused["s2"], fused["s3"], fused["s4"])
            return seg_pred

        outputs: dict[str, torch.Tensor] = {}
        for modality, decoder in (
            ("Flair", self.image_de_flair),
            ("T1c", self.image_de_t1c),
            ("T1", self.image_de_t1),
            ("T2", self.image_de_t2),
        ):
            reconstruction, mu, sigma = decoder(style[modality], fused["s4"])
            outputs[f"reconstruct_{modality}"] = reconstruction
            outputs[f"mu_{modality}"] = mu
            outputs[f"sigma_{modality}"] = sigma

        seg_pred, seg_logit = self.mask_de(fused["s1"], fused["s2"], fused["s3"], fused["s4"])
        outputs["seg_pred"] = seg_pred
        outputs["seg_logit"] = seg_logit
        outputs["images"] = {
            "Flair": x[:, 0:1],
            "T1c": x[:, 1:2],
            "T1": x[:, 2:3],
            "T2": x[:, 3:4],
        }
        return outputs

    def predict(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        was_training = self.is_training
        self.is_training = False
        try:
            _, _, height, width, depth = images.shape
            if (height, width, depth) == (input_patch_size, input_patch_size, input_patch_size):
                return self(images, mask)

            h_starts = self._window_starts(height)
            w_starts = self._window_starts(width)
            d_starts = self._window_starts(depth)
            prediction = torch.zeros(images.size(0), self.num_cls, height, width, depth, device=images.device)
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
        finally:
            self.is_training = was_training

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

    @staticmethod
    def _remap_input_order(
        images: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images.size(1) != num_modals:
            raise RuntimeError(f"SFusion expects {num_modals} input modalities, got {images.size(1)}")
        if mask.ndim != 2 or mask.size(1) != num_modals:
            raise RuntimeError(f"SFusion expects mask shape [B, {num_modals}], got {tuple(mask.shape)}")

        return images[:, DATASET_MODALITY_ORDER, ...], mask[:, DATASET_MODALITY_ORDER]


Model = SFusion


__all__ = ["SFusion", "Model"]
