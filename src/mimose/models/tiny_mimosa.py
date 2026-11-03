from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from mimose.models.abstract_model import AbstractModel
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


class TinyMimosa(AbstractModel):
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
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        resolved_input_shape = tuple(int(dim) for dim in input_shape)
        if len(resolved_input_shape) != 3 or any(dim <= 0 for dim in resolved_input_shape):
            raise ValueError(
                "TinyMimosa input_shape must contain three positive spatial dimensions"
            )
        if num_modals <= 0:
            raise ValueError("TinyMimosa num_modals must be positive")
        if not features_per_stage:
            raise ValueError("TinyMimosa features_per_stage cannot be empty")
        if len(features_per_stage) < 4:
            raise ValueError("TinyMimosa needs at least four feature stages for auxiliary heads")
        if not (0.0 <= overlap < 1.0):
            raise ValueError("TinyMimosa overlap must be in the range [0, 1)")

        self.num_cls = int(num_cls)
        self.num_modals = int(num_modals)
        self.input_shape = resolved_input_shape
        self.features_per_stage = tuple(int(ch) for ch in features_per_stage)
        self.overlap = float(overlap)
        self.gradient_checkpointing = bool(gradient_checkpointing)
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
                "TinyMimosa forward expects spatial shape "
                f"{self.tile_shape}, got {tuple(images.shape[2:])}"
            )

        # Zero out modalities the mask marks as missing *before* the joint
        # encoder sees them. Unlike the per-modality-encoder models in this
        # codebase (which can safely mask post-encoder features), TinyMimosa
        # uses a single encoder over all channels, so leaving the raw pixel
        # data in place here would let it look straight through the "missing"
        # constraint that self.mask_conditioning below cannot undo.
        gate = mask.to(images.dtype).view(images.size(0), images.size(1), 1, 1, 1)
        images = images * gate

        if self._checkpointing_active():
            skips = self._encoder_forward_checkpointed(images)
        else:
            skips = self.backbone.encoder(images)
        latent = skips[-1]
        scale, bias = self.mask_conditioning(mask)
        skips[-1] = latent * (1.0 + torch.tanh(scale)) + bias

        if self._checkpointing_active():
            fuse_logits = self._decoder_forward_checkpointed(skips)
        else:
            fuse_logits = self.backbone.decoder(skips)
        fuse_pred = torch.softmax(fuse_logits, dim=1)
        if self.is_training:
            return fuse_pred, (), ()
        return fuse_pred

    def _checkpointing_active(self) -> bool:
        """Checkpoint only when it can actually pay for itself.

        Recomputation is pure overhead unless a backward pass will follow, so
        inference and ``torch.no_grad`` sliding-window ``predict`` take the
        stock library path unchanged.
        """
        return self.gradient_checkpointing and self.training and torch.is_grad_enabled()

    def _encoder_forward_checkpointed(self, x: torch.Tensor) -> list[torch.Tensor]:
        """``PlainConvEncoder.forward`` with one checkpoint segment per stage.

        Reimplemented rather than wrapping each stage in a ``nn.Module``:
        a wrapper would insert itself into the parameter names
        (``encoder.stages.0.convs...`` -> ``encoder.stages.0.module.convs...``)
        and silently invalidate every existing checkpoint. Calling the same
        submodules from here keeps ``state_dict`` byte-identical, which is what
        lets ``gradient_checkpointing`` be toggled on a run mid-flight.

        The encoder returns all skips (``return_skips=True`` is what
        ``PlainConvUNet`` configures), so the ``return_skips`` branch of the
        library forward is not reproduced -- it is unreachable here.
        """
        skips: list[torch.Tensor] = []
        for stage in self.backbone.encoder.stages:
            x = checkpoint(stage, x, use_reentrant=False)
            skips.append(x)
        return skips

    def _decoder_forward_checkpointed(self, skips: list[torch.Tensor]) -> torch.Tensor:
        """``UNetDecoder.forward`` with one checkpoint segment per stage.

        Only the ``deep_supervision=False`` path is reproduced, which is the
        only one ``TinyMimosa`` builds; the assert below is what stops a future
        change to that constructor argument from silently losing the auxiliary
        heads.

        ``transpconvs`` are left outside the checkpoint on purpose: their output
        is immediately concatenated with a skip that is already stored, so
        recomputing them would buy nothing while forcing the (large, upsampled)
        concat input to be saved anyway.
        """
        decoder = self.backbone.decoder
        assert not decoder.deep_supervision, (
            "checkpointed decoder reproduces the deep_supervision=False path only"
        )

        lres_input = skips[-1]
        for stage in range(len(decoder.stages)):
            x = decoder.transpconvs[stage](lres_input)
            x = torch.cat((x, skips[-(stage + 2)]), 1)
            lres_input = checkpoint(decoder.stages[stage], x, use_reentrant=False)
        return decoder.seg_layers[-1](lres_input)

    def predict(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._validate_boundary_inputs(images, mask)
        original_shape = tuple(int(dim) for dim in images.shape[2:])

        # Zero-pad symmetrically (matching TinyMimosaTransform._pad_to_compatible_shape's
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
                f"TinyMimosa expects images with shape [B, {self.num_modals}, H, W, D], "
                f"got {tuple(images.shape)}"
            )
        if mask.ndim != 2 or mask.size(0) != images.size(0) or mask.size(1) != self.num_modals:
            raise RuntimeError(
                f"TinyMimosa expects mask shape [B, {self.num_modals}], got {tuple(mask.shape)}"
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
                "TinyMimosa sliding-window inference expects padded dimensions to be at least "
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
                f"TinyMimosa expects {NUM_MODALITIES} input modalities, got {images.size(1)}"
            )
        if mask.ndim != 2 or mask.size(1) != NUM_MODALITIES:
            raise RuntimeError(
                f"TinyMimosa expects mask shape [B, {NUM_MODALITIES}], got {tuple(mask.shape)}"
            )

        return images, mask


Model = TinyMimosa


__all__ = ["TinyMimosa", "Model"]
