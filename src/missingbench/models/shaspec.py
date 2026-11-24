from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from missingbench.models.abstract_model import AbstractModel

MODALITIES = ["Flair", "T1", "T1c", "T2"]

num_modals = 4
input_patch_size = 80
bottleneck_channels = 256

# External pipeline uses [t1c, t1n, t2f, t2w]; ShaSpec internals expect
# [flair, t1, t1ce, t2] (legacy ShaSpec's own modality-index convention,
# different from the [flair, t1ce, t1, t2] convention used by DCSeg/RFNet).
DATASET_MODALITY_ORDER = (2, 1, 0, 3)


class Conv3dWD(nn.Conv3d):
    """Weight-standardized conv3d, matching legacy ShaSpec's default (--weight_std=True)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        weight = weight - weight.mean(dim=(1, 2, 3, 4), keepdim=True)
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(
    in_planes: int,
    out_planes: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    bias: bool = False,
) -> nn.Conv3d:
    return Conv3dWD(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
    )


def norm_layer(planes: int) -> nn.Module:
    return nn.InstanceNorm3d(planes, affine=True)


def activation_layer() -> nn.Module:
    return nn.LeakyReLU(negative_slope=1e-2, inplace=True)


class BasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int) -> None:
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=3, padding=1)
        self.norm1 = norm_layer(planes)
        self.conv2 = conv3x3x3(planes, planes, kernel_size=3, padding=1)
        self.norm2 = norm_layer(planes)
        self.nonlin = activation_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.nonlin(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + residual
        return self.nonlin(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=1)
        self.norm1 = norm_layer(planes)
        self.conv2 = conv3x3x3(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.norm2 = norm_layer(planes)
        self.conv3 = conv3x3x3(planes, planes * self.expansion, kernel_size=1)
        self.norm3 = norm_layer(planes * self.expansion)
        self.nonlin = activation_layer()
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.nonlin(self.norm1(self.conv1(x)))
        out = self.nonlin(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.nonlin(out + residual)


class ResNet50Backbone(nn.Module):
    """3D ResNet-50 backbone.

    Legacy ShaSpec's stem uses stride ``(1, 2, 2)`` (only downsampling H/W),
    which requires a non-cubic ``(80, 160, 160)`` input and an extra
    asymmetric upsample at the end of the decoder to compensate. This port
    uses an isotropic stride ``1`` stem instead, so the whole encoder/decoder
    downsamples/upsamples by 16x uniformly and works with the cubic patches
    used throughout the rest of MissingBench (matching legacy's actual bottleneck
    resolution when trained at ``patch_size: 80``, since legacy's D axis was
    always downsampled by exactly 16x too).
    """

    layer_blocks = (3, 4, 6, 3)
    layer_planes = (64, 128, 256, 320)

    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = conv3x3x3(1, 64, kernel_size=7, stride=1, padding=3)
        self.norm1 = norm_layer(64)
        self.nonlin = activation_layer()
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(self.layer_planes[0], self.layer_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.layer_planes[1], self.layer_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.layer_planes[2], self.layer_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.layer_planes[3], self.layer_blocks[3], stride=2)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                conv3x3x3(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride),
                norm_layer(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        layers = []
        x = self.nonlin(self.norm1(self.conv1(x)))
        layers.append(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        layers.append(x)
        x = self.layer2(x)
        layers.append(x)
        x = self.layer3(x)
        layers.append(x)
        x = self.layer4(x)
        layers.append(x)
        return layers


class Aspp(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(conv3x3x3(dim_in, dim_out, kernel_size=1), norm_layer(dim_out), activation_layer())
        self.branch2 = nn.Sequential(
            conv3x3x3(dim_in, dim_out, kernel_size=3, padding=2, dilation=2), norm_layer(dim_out), activation_layer()
        )
        self.branch3 = nn.Sequential(
            conv3x3x3(dim_in, dim_out, kernel_size=3, padding=4, dilation=4), norm_layer(dim_out), activation_layer()
        )
        self.branch4 = nn.Sequential(
            conv3x3x3(dim_in, dim_out, kernel_size=3, padding=8, dilation=8), norm_layer(dim_out), activation_layer()
        )
        # No norm on branch5: it operates on a globally-pooled (1, 1, 1) spatial
        # feature, where InstanceNorm3d's per-instance variance is degenerate.
        self.branch5_conv = conv3x3x3(dim_in, dim_out, kernel_size=1)
        self.branch5_nonlin = activation_layer()
        self.conv_cat = nn.Sequential(
            conv3x3x3(dim_out * 5, dim_out, kernel_size=1), norm_layer(dim_out), activation_layer()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, d, h, w = x.size()
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        global_feature = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        global_feature = self.branch5_nonlin(self.branch5_conv(global_feature))
        global_feature_map = F.interpolate(global_feature, (d, h, w), mode="trilinear", align_corners=True)

        result = self.conv_cat(torch.cat([branch1, branch2, branch3, branch4, global_feature_map], dim=1))
        return result, global_feature


backbone_out_channels = ResNet50Backbone.layer_planes[-1] * Bottleneck.expansion  # layer4 output: 320 * 4 = 1280


class Encoder(nn.Module):
    """U_Res3D_enc: a ResNet-50 backbone followed by ASPP."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ResNet50Backbone()
        self.aspp_reduce = nn.Sequential(
            conv3x3x3(backbone_out_channels, bottleneck_channels, kernel_size=1),
            norm_layer(bottleneck_channels),
            activation_layer(),
        )
        self.aspp = Aspp(bottleneck_channels, bottleneck_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        layers = self.backbone(x)
        reduced = self.aspp_reduce(layers[-1])
        content, global_feature = self.aspp(reduced)
        return content, global_feature, layers


class Decoder(nn.Module):
    """U_Res3D_dec: transposed-conv decoder with ResNet-backbone skip connections."""

    # Per-modality channel counts of backbone.layers[0..3] (stem, layer1, layer2, layer3).
    skip_channels = (64, 64 * Bottleneck.expansion, 128 * Bottleneck.expansion, 256 * Bottleneck.expansion)
    stage_channels = (32, 64, 128, 256)

    def __init__(self, num_cls: int) -> None:
        super().__init__()
        self.shortcuts = nn.ModuleList(
            [
                nn.Sequential(
                    conv3x3x3(self.skip_channels[i] * num_modals, self.stage_channels[i], kernel_size=1),
                    norm_layer(self.stage_channels[i]),
                    activation_layer(),
                )
                for i in range(4)
            ]
        )
        # Deepest stage first: the bottleneck is the 4-modality-concatenated fused content.
        upconv_in_channels = (bottleneck_channels * num_modals, *self.stage_channels[:0:-1])
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose3d(upconv_in_channels[i], self.stage_channels[3 - i], kernel_size=2, stride=2, bias=False)
                for i in range(4)
            ]
        )
        self.stages = nn.ModuleList(
            [BasicBlock(self.stage_channels[3 - i], self.stage_channels[3 - i]) for i in range(4)]
        )

        self.seg_layer = nn.Conv3d(self.stage_channels[0], num_cls, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, skip_layers: list[torch.Tensor]) -> torch.Tensor:
        # skip_layers[0..3] correspond to backbone stem/layer1/layer2/layer3 outputs,
        # already concatenated across the 4 modalities along the channel dim.
        for stage_index in range(4):
            x = self.upconvs[stage_index](x)
            skip = self.shortcuts[3 - stage_index](skip_layers[3 - stage_index])
            x = self.stages[stage_index](x + skip)

        logits = self.seg_layer(x)
        return self.softmax(logits)


class CompositionalLayer(nn.Module):
    """Fuses shared- and specific-modality features by residual composition."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = conv3x3x3(bottleneck_channels * 2, bottleneck_channels, kernel_size=3, padding=1)

    def forward(self, shared: torch.Tensor, specific: torch.Tensor) -> torch.Tensor:
        residual = self.conv(torch.cat((shared, specific), dim=1))
        return shared + residual


class ShaSpec(AbstractModel):
    """Implementation of ShaSpec introduced in [1].

    [1] Wang, et al. "Multi-Modal Learning With Missing Modality via
        Shared-Specific Feature Modelling". CVPR 2023.

    Each modality is encoded twice: once by a single weight-shared encoder
    (run separately per modality, since InstanceNorm makes this equivalent to
    legacy's modality-as-batch trick) producing a "shared" representation
    meant to be modality-invariant, and once by a modality-specific encoder.
    Present modalities compose their shared and specific features; missing
    modalities fall back to the shared feature of the first available
    modality (generalizing legacy's single-mask-per-batch fallback to MissingBench's
    per-sample masks).
    """

    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.num_cls = num_cls

        self.shared_enc = Encoder()
        self.flair_enc = Encoder()
        self.t1_enc = Encoder()
        self.t1ce_enc = Encoder()
        self.t2_enc = Encoder()

        self.compos_layer = CompositionalLayer()
        self.dom_classifier = nn.Linear(bottleneck_channels, num_modals, bias=True)
        self.decoder = Decoder(num_cls=num_cls)

        self.is_training = False

        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, nn.InstanceNorm3d):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        x, mask = self._remap_input_order(x, mask)
        batch_size = x.size(0)

        gate = mask.view(batch_size, num_modals, 1, 1, 1, 1).to(x.dtype)
        zeroed = x.unsqueeze(2) * gate  # [B, 4, 1, D, H, W]

        specific_encoders = (self.flair_enc, self.t1_enc, self.t1ce_enc, self.t2_enc)

        shared_content: list[torch.Tensor] = []
        shared_global: list[torch.Tensor] = []
        shared_skips: list[list[torch.Tensor]] = []
        specific_content: list[torch.Tensor] = []
        specific_global: list[torch.Tensor] = []

        for modal_index in range(num_modals):
            modal_input = zeroed[:, modal_index]
            content, global_feature, skips = self.shared_enc(modal_input)
            shared_content.append(content)
            shared_global.append(global_feature)
            shared_skips.append(skips)

            spec_content, spec_global, _ = specific_encoders[modal_index](modal_input)
            specific_content.append(spec_content)
            specific_global.append(spec_global)

        # Per-sample fallback: the shared content of the first available modality.
        first_available = mask.float().argmax(dim=1)
        shared_stack = torch.stack(shared_content, dim=1)  # [B, 4, C, d, h, w]
        gather_index = first_available.view(-1, 1, 1, 1, 1, 1).expand(-1, 1, *shared_stack.shape[2:])
        fallback_shared = torch.gather(shared_stack, dim=1, index=gather_index).squeeze(1)

        fused_content = []
        for modal_index in range(num_modals):
            composed = self.compos_layer(shared_content[modal_index], specific_content[modal_index])
            present = mask[:, modal_index].view(-1, 1, 1, 1, 1)
            fused_content.append(torch.where(present, composed, fallback_shared))

        bottleneck_in = torch.cat(fused_content, dim=1)

        # Concatenate the shared encoder's skip connections across modalities,
        # for each of the 4 backbone stages used by the decoder.
        skip_layers = [
            torch.cat([shared_skips[modal_index][stage] for modal_index in range(num_modals)], dim=1)
            for stage in range(4)
        ]

        seg_pred = self.decoder(bottleneck_in, skip_layers)

        if not self.is_training:
            return seg_pred

        return {
            "seg_pred": seg_pred,
            "shared_content": shared_content,
            "spec_global": torch.cat(specific_global, dim=0).squeeze(-1).squeeze(-1).squeeze(-1),
            "dom_logits": self.dom_classifier(
                torch.cat(specific_global, dim=0).squeeze(-1).squeeze(-1).squeeze(-1)
            ),
        }

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
            raise RuntimeError(f"ShaSpec expects {num_modals} input modalities, got {images.size(1)}")
        if mask.ndim != 2 or mask.size(1) != num_modals:
            raise RuntimeError(f"ShaSpec expects mask shape [B, {num_modals}], got {tuple(mask.shape)}")

        return images[:, DATASET_MODALITY_ORDER, ...], mask[:, DATASET_MODALITY_ORDER]


Model = ShaSpec


__all__ = ["ShaSpec", "Model"]
