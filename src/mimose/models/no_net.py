from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from dynamic_network_architectures.building_blocks.helper import (
    convert_dim_to_conv_op,
    get_matching_instancenorm,
)
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder

from mimose.models.abstract_model import AbstractModel
from mimose.models.tiny_mimosa import (
    DEFAULT_NUM_CLASSES,
    NUM_MODALITIES,
    LatentMaskConditioning,
    TinyMimosa,
)

DEFAULT_INPUT_SHAPE = (128, 128, 128)


class _SegHead(nn.Module):
    """Stand-in for UNetDecoder with zero decoder stages: PlainConvUNet with
    n_stages=1 builds an empty decoder whose forward() indexes an empty list,
    so the single-stage net maps the only feature map straight to logits."""

    def __init__(self, channels: int, num_classes: int, conv_op: type[nn.Module]) -> None:
        super().__init__()
        self.seg_layer = conv_op(channels, num_classes, 1, 1, 0, bias=True)

    def forward(self, skips: list[torch.Tensor]) -> torch.Tensor:
        return self.seg_layer(skips[-1])


class _Backbone(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


class NoNet(TinyMimosa):
    """no-net: a single-stage PlainConv "U-Net" (no downsampling, no decoder).
    One full-resolution stage of ``n_conv_per_stage`` 3x3x3 conv-IN-LeakyReLU
    blocks with ``features`` channels, a 1x1x1 segmentation head, and the same
    missing-modality handling as SimpleUnet (inputs zeroed by the mask + mask
    FiLM on the only feature map). Reuses TinyMimosa's forward / sliding-window
    predict on 128^3 tiles."""

    def __init__(
        self,
        *,
        num_cls: int = DEFAULT_NUM_CLASSES,
        num_modals: int = NUM_MODALITIES,
        input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
        features: int = 128,
        n_conv_per_stage: int = 2,
        overlap: float = 0.5,
    ) -> None:
        AbstractModel.__init__(self)
        shape = tuple(int(d) for d in input_shape)
        if len(shape) != 3 or any(d <= 0 for d in shape):
            raise ValueError("NoNet input_shape must contain three positive spatial dimensions")
        if num_modals <= 0 or features <= 0 or n_conv_per_stage <= 0:
            raise ValueError("NoNet num_modals, features and n_conv_per_stage must be positive")
        if not (0.0 <= overlap < 1.0):
            raise ValueError("NoNet overlap must be in the range [0, 1)")

        self.num_cls = int(num_cls)
        self.num_modals = int(num_modals)
        self.input_shape = shape
        self.features_per_stage = (int(features),)
        self.overlap = float(overlap)
        self.is_training = False
        self.spatial_multiple = 1
        self.tile_shape = shape

        conv_op = convert_dim_to_conv_op(3)
        encoder = PlainConvEncoder(
            input_channels=self.num_modals,
            n_stages=1,
            features_per_stage=[int(features)],
            conv_op=conv_op,
            kernel_sizes=[3],
            strides=[1],
            n_conv_per_stage=[int(n_conv_per_stage)],
            conv_bias=True,
            norm_op=get_matching_instancenorm(conv_op),
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
            return_skips=True,
        )
        self.backbone = _Backbone(encoder, _SegHead(int(features), self.num_cls, conv_op))
        self.mask_conditioning = LatentMaskConditioning(self.num_modals, int(features))

        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, a=1e-2, nonlinearity="leaky_relu")


Model = NoNet


__all__ = ["NoNet", "Model"]
