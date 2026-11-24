from __future__ import annotations

import torch
import torch.nn as nn

from missingbench.models.abstract_model import AbstractModel

MODALITIES = ["Flair", "T1c", "T1", "T2"]

n_base_filters = 16
n_base_ch_se = 32
mlp_ch = 128
num_modals = 4
input_patch_size = 80

# External pipeline uses [t1c, t1n, t2f, t2w]; RobustSeg internals expect
# [flair, t1ce, t1, t2] (same convention as DCSeg/RFNet/IMFuse).
DATASET_MODALITY_ORDER = (2, 0, 1, 3)


def normalization(planes: int, norm: str = "in") -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm3d(planes)
    if norm == "gn":
        return nn.GroupNorm(32, planes)
    if norm == "in":
        return nn.InstanceNorm3d(planes, eps=1e-6)
    raise ValueError(f"normalization type {norm} is not supported")


class GeneralConv3d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pad_type: str = "zeros",
        norm: str | None = "in",
        drop_rate: float = 0.0,
        act_type: str | None = "lrelu",
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
        # Elementwise dropout as in legacy F.dropout (Dropout3d would drop whole channels).
        self.drop = nn.Dropout(p=drop_rate) if drop_rate > 0 else None
        self.norm = normalization(out_ch, norm) if norm is not None else None
        if act_type == "relu":
            self.activation: nn.Module | None = nn.ReLU(inplace=True)
        elif act_type == "lrelu":
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)
        elif act_type is None:
            self.activation = None
        else:
            raise ValueError(f"activation type {act_type} is not supported")

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


def adaptive_instance_norm(
    content: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    c_mean = torch.mean(content, dim=(2, 3, 4), keepdim=True)
    c_var = torch.var(content, dim=(2, 3, 4), keepdim=True)
    c_std = torch.sqrt(c_var + epsilon)
    return gamma * ((content - c_mean) / c_std) + beta


class AdaptiveResBlock(nn.Module):
    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.conv1 = GeneralConv3d(channels, channels, k_size=3, padding=1, pad_type="reflect", norm=None, act_type=None)
        self.conv2 = GeneralConv3d(channels, channels, k_size=3, padding=1, pad_type="reflect", norm=None, act_type=None)
        self.relu = nn.ReLU()

    def forward(self, x_init: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x_init)
        x = adaptive_instance_norm(x, mu, sigma)
        x = self.relu(x)
        x = self.conv2(x)
        x = adaptive_instance_norm(x, mu, sigma)
        return x + x_init


class StyleEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_0 = GeneralConv3d(1, n_base_ch_se, k_size=7, stride=1, padding=3, pad_type="reflect", norm=None, act_type="relu")
        self.c_1 = GeneralConv3d(n_base_ch_se, n_base_ch_se * 2, k_size=4, stride=2, padding=1, pad_type="reflect", norm=None, act_type="relu")
        self.c_2 = GeneralConv3d(n_base_ch_se * 2, n_base_ch_se * 4, k_size=4, stride=2, padding=1, pad_type="reflect", norm=None, act_type="relu")
        self.c_3 = GeneralConv3d(n_base_ch_se * 4, n_base_ch_se * 4, k_size=4, stride=2, padding=1, pad_type="reflect", norm=None, act_type="relu")
        self.c_4 = GeneralConv3d(n_base_ch_se * 4, n_base_ch_se * 4, k_size=4, stride=2, padding=1, pad_type="reflect", norm=None, act_type="relu")
        self.se_logit = GeneralConv3d(n_base_ch_se * 4, 8, k_size=1, stride=1, padding=0, pad_type="reflect", norm=None, act_type=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_0(x)
        x = self.c_1(x)
        x = self.c_2(x)
        x = self.c_3(x)
        x = self.c_4(x)
        x = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        return self.se_logit(x)


class ContentEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.e1_c1 = GeneralConv3d(1, n_base_filters, pad_type="reflect")
        self.e1_c2 = GeneralConv3d(n_base_filters, n_base_filters, pad_type="reflect", drop_rate=0.3)
        self.e1_c3 = GeneralConv3d(n_base_filters, n_base_filters, pad_type="reflect")

        self.e2_c1 = GeneralConv3d(n_base_filters, n_base_filters * 2, stride=2, pad_type="reflect")
        self.e2_c2 = GeneralConv3d(n_base_filters * 2, n_base_filters * 2, pad_type="reflect", drop_rate=0.3)
        self.e2_c3 = GeneralConv3d(n_base_filters * 2, n_base_filters * 2, pad_type="reflect")

        self.e3_c1 = GeneralConv3d(n_base_filters * 2, n_base_filters * 4, stride=2, pad_type="reflect")
        self.e3_c2 = GeneralConv3d(n_base_filters * 4, n_base_filters * 4, pad_type="reflect", drop_rate=0.3)
        self.e3_c3 = GeneralConv3d(n_base_filters * 4, n_base_filters * 4, pad_type="reflect")

        self.e4_c1 = GeneralConv3d(n_base_filters * 4, n_base_filters * 8, stride=2, pad_type="reflect")
        self.e4_c2 = GeneralConv3d(n_base_filters * 8, n_base_filters * 8, pad_type="reflect")
        self.e4_c3 = GeneralConv3d(n_base_filters * 8, n_base_filters * 8, pad_type="reflect")

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


class Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.channel = mlp_ch
        self.linear_0 = Linear(8, self.channel)
        self.linear_1 = Linear(self.channel, self.channel)
        self.mu = Linear(self.channel, self.channel)
        self.sigma = Linear(self.channel, self.channel)
        self.relu = nn.ReLU()

    def forward(self, style: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.relu(self.linear_0(style))
        x = self.relu(self.linear_1(x))
        mu = self.mu(x).view(x.shape[0], self.channel, 1, 1, 1)
        sigma = self.sigma(x).view(x.shape[0], self.channel, 1, 1, 1)
        return mu, sigma


class ImageDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.channel = mlp_ch
        self.mlp = Mlp()
        self.res_0 = AdaptiveResBlock(self.channel)
        self.res_1 = AdaptiveResBlock(self.channel)
        self.res_2 = AdaptiveResBlock(self.channel)
        self.res_3 = AdaptiveResBlock(self.channel)

        self.conv_0 = GeneralConv3d(128, 64, k_size=5, padding=2, norm=None, act_type=None)
        self.conv_1 = GeneralConv3d(64, 32, k_size=5, padding=2, norm=None, act_type=None)
        self.conv_2 = GeneralConv3d(32, 16, k_size=5, padding=2, norm=None, act_type=None)
        self.layer_norm_1 = nn.LayerNorm([20, 20, 20], eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm([40, 40, 40], eps=1e-6)
        self.layer_norm_3 = nn.LayerNorm([80, 80, 80], eps=1e-6)
        self.g_logit = GeneralConv3d(16, 1, k_size=7, padding=3, pad_type="reflect", norm=None, act_type=None)
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.relu = nn.ReLU()

    def forward(self, style: torch.Tensor, content: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, sigma = self.mlp(style)
        x = content
        x = self.res_0(x, mu, sigma)
        x = self.res_1(x, mu, sigma)
        x = self.res_2(x, mu, sigma)
        x = self.res_3(x, mu, sigma)

        x = self.upsample(x)
        x = self.relu(self.layer_norm_1(self.conv_0(x)))

        x = self.upsample(x)
        x = self.relu(self.layer_norm_2(self.conv_1(x)))

        x = self.upsample(x)
        x = self.relu(self.layer_norm_3(self.conv_2(x)))

        return self.g_logit(x), mu, sigma


class MaskDecoder(nn.Module):
    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d3_c1 = GeneralConv3d(128, n_base_filters * 4, pad_type="reflect")
        self.d3_c2 = GeneralConv3d(128, n_base_filters * 4, pad_type="reflect")
        self.d3_out = GeneralConv3d(n_base_filters * 4, n_base_filters * 4, k_size=1, padding=0, pad_type="reflect")

        self.d2_c1 = GeneralConv3d(64, n_base_filters * 2, pad_type="reflect")
        self.d2_c2 = GeneralConv3d(64, n_base_filters * 2, pad_type="reflect")
        self.d2_out = GeneralConv3d(n_base_filters * 2, n_base_filters * 2, k_size=1, padding=0, pad_type="reflect")

        self.d1_c1 = GeneralConv3d(32, n_base_filters, pad_type="reflect")
        self.d1_c2 = GeneralConv3d(32, n_base_filters, pad_type="reflect")
        self.d1_out = GeneralConv3d(n_base_filters, n_base_filters, k_size=1, padding=0, pad_type="reflect")

        self.seg_logit = GeneralConv3d(n_base_filters, num_cls, k_size=1, padding=0, norm=None, act_type=None)
        self.seg_pred = nn.Softmax(dim=1)

    def forward(
        self,
        e1_out: torch.Tensor,
        e2_out: torch.Tensor,
        e3_out: torch.Tensor,
        e4_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d3 = self.d3_c1(self.upsample(e4_out))
        d3_out = self.d3_out(self.d3_c2(torch.cat([d3, e3_out], dim=1)))

        d2 = self.d2_c1(self.upsample(d3_out))
        d2_out = self.d2_out(self.d2_c2(torch.cat([d2, e2_out], dim=1)))

        d1 = self.d1_c1(self.upsample(d2_out))
        d1_out = self.d1_out(self.d1_c2(torch.cat([d1, e1_out], dim=1)))

        seg_logit = self.seg_logit(d1_out)
        seg_pred = self.seg_pred(seg_logit)
        return seg_pred, seg_logit


class RobustSeg(AbstractModel):
    """Implementation of RobustSeg introduced in [1].

    [1] Ting, et al. "RobustSeg: Hierarchical Adaptive Fusion and Feature
        Consistency for Robust Multimodal Brain Tumor Segmentation with
        Missing Modalities". MICCAI 2021.
    """

    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.num_cls = num_cls

        self.se_flair = StyleEncoder()
        self.se_t1c = StyleEncoder()
        self.se_t1 = StyleEncoder()
        self.se_t2 = StyleEncoder()

        self.ce_flair = ContentEncoder()
        self.ce_t1c = ContentEncoder()
        self.ce_t1 = ContentEncoder()
        self.ce_t2 = ContentEncoder()

        self.att_c1 = GeneralConv3d(n_base_filters * 4, num_modals, pad_type="reflect")
        self.att_c2 = GeneralConv3d(n_base_filters * 8, num_modals, pad_type="reflect")
        self.att_c3 = GeneralConv3d(n_base_filters * 16, num_modals, pad_type="reflect")
        self.att_c4 = GeneralConv3d(n_base_filters * 32, num_modals, pad_type="reflect")

        self.fusion_c1 = GeneralConv3d(n_base_filters * 4, n_base_filters, k_size=1, padding=0, pad_type="reflect")
        self.fusion_c2 = GeneralConv3d(n_base_filters * 8, n_base_filters * 2, k_size=1, padding=0, pad_type="reflect")
        self.fusion_c3 = GeneralConv3d(n_base_filters * 16, n_base_filters * 4, k_size=1, padding=0, pad_type="reflect")
        self.fusion_c4 = GeneralConv3d(n_base_filters * 32, n_base_filters * 8, k_size=1, padding=0, pad_type="reflect")

        self.image_de_flair = ImageDecoder()
        self.image_de_t1c = ImageDecoder()
        self.image_de_t1 = ImageDecoder()
        self.image_de_t2 = ImageDecoder()

        self.mask_de = MaskDecoder(num_cls=num_cls)

        self.sigmoid = nn.Sigmoid()

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
        att_convs = (self.att_c1, self.att_c2, self.att_c3, self.att_c4)
        fusion_convs = (self.fusion_c1, self.fusion_c2, self.fusion_c3, self.fusion_c4)
        scale_channels = (n_base_filters, n_base_filters * 2, n_base_filters * 4, n_base_filters * 8)
        for level, (att_conv, fusion_conv, channels) in enumerate(
            zip(att_convs, fusion_convs, scale_channels), start=1
        ):
            scale_key = f"s{level}"
            masked_content = []
            for modal_index, modality in enumerate(MODALITIES):
                gate = mask[:, modal_index].view(-1, *([1] * (content[modality][scale_key].ndim - 1)))
                masked_content.append(content[modality][scale_key] * gate)

            concat = torch.cat(masked_content, dim=1)
            attmap = self.sigmoid(att_conv(concat))
            gated = torch.cat(
                [
                    masked_content[modal_index] * attmap[:, modal_index : modal_index + 1].repeat(1, channels, 1, 1, 1)
                    for modal_index in range(num_modals)
                ],
                dim=1,
            )
            fused[scale_key] = fusion_conv(gated)

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
            raise RuntimeError(f"RobustSeg expects {num_modals} input modalities, got {images.size(1)}")
        if mask.ndim != 2 or mask.size(1) != num_modals:
            raise RuntimeError(f"RobustSeg expects mask shape [B, {num_modals}], got {tuple(mask.shape)}")

        return images[:, DATASET_MODALITY_ORDER, ...], mask[:, DATASET_MODALITY_ORDER]


Model = RobustSeg


__all__ = ["RobustSeg", "Model"]
