from __future__ import annotations

import torch
import torch.nn as nn

from missingbench.models.abstract_model import AbstractModel

MODALITIES = ["T1", "T1c", "T2", "Flair"]

NB_CONV = 8
num_modals = 4
input_patch_size = 112

# External pipeline uses [t1c, t1n, t2f, t2w]; U-HVED internals expect
# [T1, T1c, T2, Flair] (the order ConvEncoder/GaussianSampler index by).
DATASET_MODALITY_ORDER = (1, 0, 3, 2)


def initialize_weights(model: nn.Module) -> None:
    """NiftyNet ``he_normal`` for convolutions (truncated normal, std sqrt(2 / fan_in) with
    fan_in = in_channels * prod(kernel)); norm layers keep gamma = 1 and beta = 0."""
    for module in model.modules():
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
            fan_in = _prod(module.weight.shape[1:])
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=(2.0 / fan_in) ** 0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def _prod(shape: torch.Size) -> float:
    result = 1.0
    for dim in shape:
        result *= dim
    return max(result, 1.0)


class MaskModal(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        return y


class GeneralConv3dPrenorm(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k_size: tuple[int, int, int] = (3, 3, 3),
        stride: int = 1,
        padding: str = "same",
        act_type: str = "leakyrelu",
        relufactor: float = 0.01,
    ) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm3d(in_ch, eps=1e-6, affine=True)
        if act_type == "relu":
            self.activation: nn.Module = nn.ReLU(inplace=True)
        elif act_type == "leakyrelu":
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)
        else:
            raise ValueError(f"activation type {act_type} is not supported")
        self.conv = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=k_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        return x


class ConvEncoder(nn.Module):
    """Encodes a single modality into 4 skip-level (mu, logvar) pairs."""

    def __init__(self) -> None:
        super().__init__()
        self.ini_f = NB_CONV
        self.hidden = [self.ini_f, 2 * self.ini_f, 4 * self.ini_f, 8 * self.ini_f]
        self.hidden = [dim // 2 for dim in self.hidden]

        self.conv1 = nn.Conv3d(1, self.ini_f, kernel_size=(1, 1, 1), bias=False, padding="same")
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.e1_c1 = GeneralConv3dPrenorm(self.ini_f, self.ini_f)
        self.e1_c2 = GeneralConv3dPrenorm(self.ini_f, self.ini_f)
        self.d1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.e2_c1 = GeneralConv3dPrenorm(self.ini_f, 2 * self.ini_f)
        self.e2_c2 = GeneralConv3dPrenorm(2 * self.ini_f, 2 * self.ini_f)
        self.d2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.e3_c1 = GeneralConv3dPrenorm(2 * self.ini_f, 4 * self.ini_f)
        self.e3_c2 = GeneralConv3dPrenorm(4 * self.ini_f, 4 * self.ini_f)
        self.d3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.e4_c1 = GeneralConv3dPrenorm(4 * self.ini_f, 8 * self.ini_f)
        self.e4_c2 = GeneralConv3dPrenorm(8 * self.ini_f, 8 * self.ini_f)

    @staticmethod
    def _clip(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=-50, max=50)

    def forward(self, x: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        skip_flows: list[dict[str, torch.Tensor]] = [dict() for _ in range(4)]

        x = self.act1(self.conv1(x))
        x = self.e1_c2(self.e1_c1(x))
        skip_flows[0]["mu"] = x[:, : self.hidden[0]]
        skip_flows[0]["logvar"] = self._clip(x[:, self.hidden[0] :])
        x = self.d1(x)

        x = self.e2_c2(self.e2_c1(x))
        skip_flows[1]["mu"] = x[:, : self.hidden[1]]
        skip_flows[1]["logvar"] = self._clip(x[:, self.hidden[1] :])
        x = self.d2(x)

        x = self.e3_c2(self.e3_c1(x))
        skip_flows[2]["mu"] = x[:, : self.hidden[2]]
        skip_flows[2]["logvar"] = self._clip(x[:, self.hidden[2] :])
        x = self.d3(x)

        x = self.e4_c2(self.e4_c1(x))
        skip_flows[3]["mu"] = x[:, : self.hidden[3]]
        skip_flows[3]["logvar"] = self._clip(x[:, self.hidden[3] :])

        return skip_flows


class GaussianSampler(nn.Module):
    """Product-of-experts fusion of per-modality posteriors, masked by availability."""

    def __init__(self) -> None:
        super().__init__()
        self.eps = 1e-7
        self.masker = MaskModal()

    def forward(
        self,
        means: dict[str, torch.Tensor],
        logvars: dict[str, torch.Tensor],
        list_mod: list[str],
        mask: torch.Tensor,
        is_inference: bool,
    ) -> torch.Tensor:
        mu_prior = torch.zeros_like(means[list_mod[0]])
        log_prior = torch.zeros_like(means[list_mod[0]])

        precision = self.masker(
            torch.stack([1 / (torch.exp(logvars[mod]) + self.eps) for mod in list_mod], dim=0),
            mask.permute(1, 0),
        )
        weighted_mu = self.masker(
            torch.stack([means[mod] / (torch.exp(logvars[mod]) + self.eps) for mod in list_mod], dim=0),
            mask.permute(1, 0),
        )

        precision = torch.cat([precision, (1 + log_prior).unsqueeze(0)], dim=0)
        weighted_mu = torch.cat([weighted_mu, mu_prior.unsqueeze(0)], dim=0)

        posterior_means = torch.sum(weighted_mu, dim=0) / torch.sum(precision, dim=0)
        var = 1 / torch.sum(precision, dim=0)
        posterior_logvars = torch.log(var + self.eps)

        if is_inference:
            return posterior_means

        noise_sample = torch.randn(posterior_means.shape, device=posterior_means.device)
        return posterior_means + torch.exp(0.5 * posterior_logvars) * noise_sample


class ConvDecoderImg(nn.Module):
    """Decodes the multi-scale hidden samples into a single-modality (or seg) volume."""

    def __init__(self, num_cls: int = 1) -> None:
        super().__init__()
        self.ini_f = NB_CONV
        self.num_cls = num_cls

        self.d1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d1_c1 = GeneralConv3dPrenorm(6 * self.ini_f, 4 * self.ini_f)
        self.d1_c2 = GeneralConv3dPrenorm(4 * self.ini_f, 2 * self.ini_f)

        self.d2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d2_c1 = GeneralConv3dPrenorm(3 * self.ini_f, 2 * self.ini_f)
        self.d2_c2 = GeneralConv3dPrenorm(2 * self.ini_f, self.ini_f)

        self.d3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d3_c1 = GeneralConv3dPrenorm(int(1.5 * self.ini_f), self.ini_f)
        self.d3_c2 = GeneralConv3dPrenorm(self.ini_f, self.ini_f // 2)

        self.conv_out = nn.Conv3d(self.ini_f // 2, num_cls, kernel_size=1, bias=False, padding="same")
        self.softmax = nn.Softmax(dim=1)

    def forward(self, list_skips: list[torch.Tensor], use_softmax: bool = False) -> torch.Tensor:
        list_skips = list_skips[::-1]

        de_x = self.d1(list_skips[0])
        de_x = self.d1_c2(self.d1_c1(torch.cat((de_x, list_skips[1]), dim=1)))

        de_x = self.d2(de_x)
        de_x = self.d2_c2(self.d2_c1(torch.cat((de_x, list_skips[2]), dim=1)))

        de_x = self.d3(de_x)
        de_x = self.d3_c2(self.d3_c1(torch.cat((de_x, list_skips[3]), dim=1)))

        logits = self.conv_out(de_x)
        if use_softmax:
            logits = self.softmax(logits)
        return logits


class UHVED(AbstractModel):
    """Implementation of U-HVED introduced in [1], mixing MVAE [2] and a U-Net [3].

    [1] Dorent, et al. "Hetero-Modal Variational Encoder-Decoder for Joint Modality
        Completion and Segmentation". MICCAI 2019.
    [2] Wu, et al. "Multimodal Generative Models for Scalable Weakly-Supervised
        Learning". NIPS 2018.
    [3] Ronneberger, et al. "U-Net: Convolutional Networks for Biomedical Image
        Segmentation". MICCAI 2015.
    """

    def __init__(self, num_cls: int = 4, test_time_samples: int = 0) -> None:
        super().__init__()
        self.num_cls = num_cls
        # 0 decodes the posterior mean; N > 0 averages N decoded posterior samples (the paper uses 10,
        # the released code decodes a single sample).
        self.test_time_samples = int(test_time_samples)

        self.t1_encoder = ConvEncoder()
        self.t1ce_encoder = ConvEncoder()
        self.t2_encoder = ConvEncoder()
        self.flair_encoder = ConvEncoder()

        self.approximate_sampler = GaussianSampler()

        self.t1_decoder = ConvDecoderImg(num_cls=1)
        self.t1ce_decoder = ConvDecoderImg(num_cls=1)
        self.t2_decoder = ConvDecoderImg(num_cls=1)
        self.flair_decoder = ConvDecoderImg(num_cls=1)
        self.seg_decoder = ConvDecoderImg(num_cls=num_cls)

        self.mod_img = MODALITIES

        self.is_training = False

        initialize_weights(self)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | tuple[dict[str, torch.Tensor], list[dict[str, dict[str, torch.Tensor]]]]:
        x, mask = self._remap_input_order(x, mask)

        images = {
            "T1": x[:, 0:1],
            "T1c": x[:, 1:2],
            "T2": x[:, 2:3],
            "Flair": x[:, 3:4],
        }

        t1_param = self.t1_encoder(images["T1"])
        t1c_param = self.t1ce_encoder(images["T1c"])
        t2_param = self.t2_encoder(images["T2"])
        flair_param = self.flair_encoder(images["Flair"])

        post_param = [
            {
                "mu": {
                    "T1": t1_param[i]["mu"],
                    "T1c": t1c_param[i]["mu"],
                    "T2": t2_param[i]["mu"],
                    "Flair": flair_param[i]["mu"],
                },
                "logvar": {
                    "T1": t1_param[i]["logvar"],
                    "T1c": t1c_param[i]["logvar"],
                    "T2": t2_param[i]["logvar"],
                    "Flair": flair_param[i]["logvar"],
                },
            }
            for i in range(len(t1_param))
        ]

        if not self.is_training:
            if self.test_time_samples <= 0:
                skip_flow = self._sample_skips(post_param, mask, use_mean=True)
                return self.seg_decoder(skip_flow, use_softmax=True)
            prediction = None
            for _ in range(self.test_time_samples):
                sample_pred = self.seg_decoder(self._sample_skips(post_param, mask, use_mean=False), use_softmax=True)
                prediction = sample_pred if prediction is None else prediction + sample_pred
            return prediction / self.test_time_samples

        skip_flow = self._sample_skips(post_param, mask, use_mean=False)

        outputs = {
            "T1": self.t1_decoder(skip_flow),
            "T1c": self.t1ce_decoder(skip_flow),
            "T2": self.t2_decoder(skip_flow),
            "Flair": self.flair_decoder(skip_flow),
            "seg": self.seg_decoder(skip_flow, use_softmax=True),
            "images": images,
        }
        return outputs, post_param

    def _sample_skips(
        self, post_param: list[dict[str, dict[str, torch.Tensor]]], mask: torch.Tensor, *, use_mean: bool
    ) -> list[torch.Tensor]:
        return [
            self.approximate_sampler(level["mu"], level["logvar"], self.mod_img, mask, is_inference=use_mean)
            for level in post_param
        ]

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
            prediction = torch.zeros(
                images.size(0),
                self.num_cls,
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
            raise RuntimeError(f"UHVED expects {num_modals} input modalities, got {images.size(1)}")
        if mask.ndim != 2 or mask.size(1) != num_modals:
            raise RuntimeError(f"UHVED expects mask shape [B, {num_modals}], got {tuple(mask.shape)}")

        return images[:, DATASET_MODALITY_ORDER, ...], mask[:, DATASET_MODALITY_ORDER]


Model = UHVED


__all__ = ["UHVED", "Model"]
