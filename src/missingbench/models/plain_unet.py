from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from missingbench.models.abstract_model import AbstractModel
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import (
    convert_dim_to_conv_op,
    get_matching_instancenorm,
)


NUM_MODALITIES = 4
DEFAULT_NUM_CLASSES = 4
DEFAULT_INPUT_SHAPE = (182, 218, 182)


class LatentMaskConditioning(nn.Module):
    def __init__(self, modality_count: int, latent_channels: int, bias_limit: float = 3.0) -> None:
        super().__init__()
        hidden = max(latent_channels // 2, modality_count * 2)
        self.mlp = nn.Sequential(
            nn.Linear(modality_count, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_channels * 2),
        )
        self.bias_limit = bias_limit

    def forward(self, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conditioning = self.mlp(mask.float())
        scale, bias = conditioning.chunk(2, dim=1)
        bias = self.bias_limit * torch.tanh(bias)
        return scale[..., None, None, None], bias[..., None, None, None]


class PlainUnetBackbone(AbstractModel):
    def __init__(
        self,
        *,
        num_cls: int = DEFAULT_NUM_CLASSES,
        num_modals: int = NUM_MODALITIES,
        input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
        features_per_stage: Sequence[int] = (8, 16, 32, 64),
        n_conv_per_stage: int = 2,
        n_conv_per_stage_decoder: int = 2,
        overlap: float = 0.5,
    ) -> None:
        super().__init__()
        resolved_input_shape = tuple(int(dim) for dim in input_shape)
        if len(resolved_input_shape) != 3 or any(dim <= 0 for dim in resolved_input_shape):
            raise ValueError(
                "PlainUnetBackbone input_shape must contain three positive spatial dimensions"
            )
        if num_modals <= 0:
            raise ValueError("PlainUnetBackbone num_modals must be positive")
        if not features_per_stage:
            raise ValueError("PlainUnetBackbone features_per_stage cannot be empty")
        if len(features_per_stage) < 4:
            raise ValueError("PlainUnetBackbone needs at least four feature stages for auxiliary heads")
        if not (0.0 <= overlap < 1.0):
            raise ValueError("PlainUnetBackbone overlap must be in the range [0, 1)")

        self.num_cls = int(num_cls)
        self.num_modals = int(num_modals)
        self.input_shape = resolved_input_shape
        self.features_per_stage = tuple(int(ch) for ch in features_per_stage)
        self.overlap = float(overlap)
        self.is_training = False
        self.spatial_multiple = 2 ** (len(self.features_per_stage) - 1)
        self.tile_shape = tuple(
            self._round_up_to_multiple(dim, self.spatial_multiple)
            for dim in self.input_shape
        )

        conv_op = convert_dim_to_conv_op(3)
        norm_op = get_matching_instancenorm(conv_op)
        self.backbone = PlainConvUNet(
            input_channels=self.num_modals,
            n_stages=len(self.features_per_stage),
            features_per_stage=list(self.features_per_stage),
            conv_op=conv_op,
            kernel_sizes=[3] * len(self.features_per_stage),
            strides=[1] + [2] * (len(self.features_per_stage) - 1),
            n_conv_per_stage=n_conv_per_stage,
            num_classes=self.num_cls,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=True,
            norm_op=norm_op,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
            deep_supervision=False,
        )

        self.mask_conditioning = LatentMaskConditioning(
            self.num_modals,
            self.features_per_stage[-1],
        )

        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, a=1e-2, nonlinearity="leaky_relu")

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[()], tuple[()]]:
        self._validate_boundary_inputs(images, mask)
        if tuple(int(dim) for dim in images.shape[2:]) != self.tile_shape:
            raise RuntimeError(
                "PlainUnetBackbone forward expects spatial shape "
                f"{self.tile_shape}, got {tuple(images.shape[2:])}"
            )

        # Zero out modalities the mask marks as missing *before* the joint
        # encoder sees them. Unlike the per-modality-encoder models in this
        # codebase (which can safely mask post-encoder features), PlainUnetBackbone
        # uses a single encoder over all channels, so leaving the raw pixel
        # data in place here would let it look straight through the "missing"
        # constraint that self.mask_conditioning below cannot undo.
        gate = mask.to(images.dtype).view(images.size(0), images.size(1), 1, 1, 1)
        images = images * gate

        skips = self.backbone.encoder(images)
        latent = skips[-1]
        scale, bias = self.mask_conditioning(mask)
        skips[-1] = latent * (1.0 + torch.tanh(scale)) + bias

        fuse_logits = self.backbone.decoder(skips)
        fuse_pred = torch.softmax(fuse_logits, dim=1)
        if self.is_training:
            return fuse_pred, (), ()
        return fuse_pred

    def predict(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._validate_boundary_inputs(images, mask)
        original_shape = tuple(int(dim) for dim in images.shape[2:])

        # Zero-pad symmetrically (matching the whole-volume padding transform
        # scheme, minus its multiple-of-spatial_multiple rounding: sliding-window tiles
        # below are always exactly tile_shape regardless of total volume size, so a
        # dimension already >= tile_shape needs no padding at all).
        target_shape = tuple(
            max(current, minimum) for current, minimum in zip(original_shape, self.tile_shape)
        )
        pad_before: list[int] = []
        pad_sizes: list[int] = []
        for current, target in zip(reversed(original_shape), reversed(target_shape)):
            total_pad = max(target - current, 0)
            before = total_pad // 2
            after = total_pad - before
            pad_before.append(before)
            pad_sizes.extend((before, after))
        pad_before.reverse()  # back to (H, W, D) order

        if any(pad_sizes):
            images = torch.nn.functional.pad(images, tuple(pad_sizes))

        padded_shape = tuple(int(dim) for dim in images.shape[2:])

        if padded_shape == self.tile_shape:
            prediction = self(images, mask)
            return self._crop_to_original_shape(prediction, pad_before, original_shape)

        prediction = torch.zeros(
            images.size(0),
            self.num_cls,
            *padded_shape,
            device=images.device,
            dtype=images.dtype,
        )
        weight = torch.zeros(
            images.size(0),
            1,
            *padded_shape,
            device=images.device,
            dtype=images.dtype,
        )

        h_starts = self._window_starts(
            padded_shape[0],
            self.tile_shape[0],
            self.overlap,
        )
        w_starts = self._window_starts(
            padded_shape[1],
            self.tile_shape[1],
            self.overlap,
        )
        d_starts = self._window_starts(
            padded_shape[2],
            self.tile_shape[2],
            self.overlap,
        )

        for h in h_starts:
            for w in w_starts:
                for d in d_starts:
                    patch = images[
                        :,
                        :,
                        h : h + self.tile_shape[0],
                        w : w + self.tile_shape[1],
                        d : d + self.tile_shape[2],
                    ]
                    patch_pred = self(patch, mask)
                    prediction[
                        :,
                        :,
                        h : h + self.tile_shape[0],
                        w : w + self.tile_shape[1],
                        d : d + self.tile_shape[2],
                    ] += patch_pred
                    weight[
                        :,
                        :,
                        h : h + self.tile_shape[0],
                        w : w + self.tile_shape[1],
                        d : d + self.tile_shape[2],
                    ] += 1

        prediction = prediction / weight.clamp_min(1)
        return self._crop_to_original_shape(prediction, pad_before, original_shape)

    @staticmethod
    def _crop_to_original_shape(
        prediction: torch.Tensor,
        pad_before: Sequence[int],
        original_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        if not any(pad_before) and tuple(prediction.shape[2:]) == original_shape:
            return prediction
        h0, w0, d0 = pad_before
        h1, w1, d1 = original_shape
        return prediction[:, :, h0 : h0 + h1, w0 : w0 + w1, d0 : d0 + d1]

    def _validate_boundary_inputs(self, images: torch.Tensor, mask: torch.Tensor) -> None:
        if images.ndim != 5 or images.size(1) != self.num_modals:
            raise RuntimeError(
                f"PlainUnetBackbone expects images with shape [B, {self.num_modals}, H, W, D], "
                f"got {tuple(images.shape)}"
            )
        if mask.ndim != 2 or mask.size(0) != images.size(0) or mask.size(1) != self.num_modals:
            raise RuntimeError(
                f"PlainUnetBackbone expects mask shape [B, {self.num_modals}], got {tuple(mask.shape)}"
            )

    @staticmethod
    def _round_up_to_multiple(value: int, multiple: int) -> int:
        if multiple <= 0:
            raise ValueError("multiple must be positive")
        return ((value + multiple - 1) // multiple) * multiple

    @staticmethod
    def _window_starts(size: int, window: int, overlap: float) -> list[int]:
        if size < window:
            raise RuntimeError(
                "PlainUnetBackbone sliding-window inference expects padded dimensions to be at least "
                f"{window}, got {size}"
            )
        if size == window:
            return [0]

        stride = max(int(window * (1.0 - overlap)), 1)
        starts = list(range(0, max(size - window, 0), stride))
        last_start = size - window
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        return starts

    @staticmethod
    def _remap_input_order(
        images: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images.size(1) != NUM_MODALITIES:
            raise RuntimeError(
                f"PlainUnetBackbone expects {NUM_MODALITIES} input modalities, got {images.size(1)}"
            )
        if mask.ndim != 2 or mask.size(1) != NUM_MODALITIES:
            raise RuntimeError(
                f"PlainUnetBackbone expects mask shape [B, {NUM_MODALITIES}], got {tuple(mask.shape)}"
            )

        return images, mask


# PlainUnet trains/infers on fixed 128x128x128 patches -- the same tile size the IM-Fuse-style
# transforms crop to (IMFuseTransformManager's default crop_size, custom_trainer_kwargs.patch_size=128).
PATCH_INPUT_SHAPE = (128, 128, 128)


class PlainUnet(PlainUnetBackbone):
    """Plain U-Net: the nnU-Net-style backbone above (single joint encoder over all modality
    channels, mask-conditioned bottleneck, sliding-window predict()) at a fixed 128^3 patch size.
    Pair with transform_kind=imfuse so training random-crops matching 128^3 patches."""

    def __init__(
        self,
        *,
        input_shape: Sequence[int] = PATCH_INPUT_SHAPE,
        **kwargs: object,
    ) -> None:
        super().__init__(input_shape=input_shape, **kwargs)


Model = PlainUnet


__all__ = ["PlainUnet", "Model"]
