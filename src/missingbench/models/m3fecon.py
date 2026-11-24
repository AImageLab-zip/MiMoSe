from __future__ import annotations

import torch
import torch.nn as nn

from missingbench.models.abstract_model import AbstractModel

num_modals = 4
input_patch_size = 128

# Per-stage channel counts, matching legacy M3FeCon's nnU-Net "3d_fullres" plan at
# base_num_features=32 capped at unet_max_num_features=512: [32,64,128,256,512,512]
# split evenly across the 4 per-modality encoders.
_channels_per_modality = (8, 16, 32, 64, 128, 128)
_combined_channels = tuple(c * num_modals for c in _channels_per_modality)
_strides = (1, 2, 2, 2, 2, 2)
num_stages = len(_channels_per_modality)


def normalization(planes: int) -> nn.Module:
    return nn.InstanceNorm3d(planes, affine=True)


class GeneralConv3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm = normalization(out_ch)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ModalityEncoder(nn.Module):
    """A single modality's nnU-Net-style plain conv encoder (2 convs/stage)."""

    def __init__(self) -> None:
        super().__init__()
        self.stages = nn.ModuleList()
        in_ch = 1
        for stage_index in range(num_stages):
            out_ch = _channels_per_modality[stage_index]
            self.stages.append(
                nn.Sequential(
                    GeneralConv3d(in_ch, out_ch, stride=_strides[stage_index]),
                    GeneralConv3d(out_ch, out_ch),
                )
            )
            in_ch = out_ch

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips


class BottleneckReconstruction(nn.Module):
    """Replaces missing modalities' bottleneck features with a learned token,
    then refines all 4 modalities' tokens jointly with a small transformer so
    the missing ones' features get reconstructed from the present ones."""

    def __init__(self, channels: int, spatial: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.spatial = spatial
        self.learned_tokens = nn.Parameter(torch.randn(num_modals, channels, spatial, spatial, spatial) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=4,
            dim_feedforward=channels * 2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(
        self,
        bottleneck: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        bottleneck: [B, num_modals, C, d, h, w] (each modality's own bottleneck feature)
        mask: [B, num_modals] bool, True = present
        Returns: (feature actually used downstream, ground-truth feature for the
                  reconstruction loss — both [B, num_modals, C, d, h, w]).
        """
        ground_truth = bottleneck.detach()

        gate = mask.view(-1, num_modals, 1, 1, 1, 1)
        tokens_in = torch.where(gate, bottleneck, self.learned_tokens.unsqueeze(0))

        batch_size = bottleneck.size(0)
        flattened = tokens_in.permute(0, 1, 3, 4, 5, 2).reshape(batch_size, -1, self.channels)
        refined = self.transformer(flattened)
        refined = refined.reshape(
            batch_size, num_modals, self.spatial, self.spatial, self.spatial, self.channels
        ).permute(0, 1, 5, 2, 3, 4)

        output = torch.where(gate, bottleneck, refined)
        return output, ground_truth


class Decoder(nn.Module):
    def __init__(self, num_cls: int) -> None:
        super().__init__()
        self.upsamples = nn.ModuleList(
            [nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True) for _ in range(num_stages - 1)]
        )
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    GeneralConv3d(_combined_channels[stage_index + 1] + _combined_channels[stage_index], _combined_channels[stage_index]),
                    GeneralConv3d(_combined_channels[stage_index], _combined_channels[stage_index]),
                )
                for stage_index in range(num_stages - 2, -1, -1)
            ]
        )
        self.seg_layer = nn.Conv3d(_combined_channels[0], num_cls, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, skips: list[torch.Tensor]) -> torch.Tensor:
        x = skips[-1]
        for i, (upsample, conv) in enumerate(zip(self.upsamples, self.convs)):
            stage_index = num_stages - 2 - i
            x = upsample(x)
            x = conv(torch.cat([x, skips[stage_index]], dim=1))
        return self.softmax(self.seg_layer(x))


class M3FeCon(AbstractModel):
    """Implementation of M3FeCon introduced in [1].

    [1] "Missing as Masking: Arbitrary Cross-modal Feature Reconstruction for
        Incomplete Multimodal Brain Tumor Segmentation". MICCAI 2024.

    Each modality is encoded independently (a plain nnU-Net-style conv
    encoder per modality). At the bottleneck, missing modalities' features
    are replaced with a learned token, and a small transformer jointly
    refines all 4 modalities' bottleneck tokens so missing ones are
    reconstructed from cross-modal attention with the present ones. At
    shallower scales, missing modalities are simply zeroed before the
    per-stage skip connections are concatenated across modalities and fed to
    a standard U-Net decoder.
    """

    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.num_cls = num_cls
        self.encoders = nn.ModuleList([ModalityEncoder() for _ in range(num_modals)])
        self.bottleneck = BottleneckReconstruction(channels=_channels_per_modality[-1])
        self.decoder = Decoder(num_cls=num_cls)

        self.is_training = False

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.size(1) != num_modals:
            raise RuntimeError(f"M3FeCon expects {num_modals} input modalities, got {x.size(1)}")
        if mask.ndim != 2 or mask.size(1) != num_modals:
            raise RuntimeError(f"M3FeCon expects mask shape [B, {num_modals}], got {tuple(mask.shape)}")

        per_modal_skips = [self.encoders[i](x[:, i : i + 1]) for i in range(num_modals)]

        bottleneck_stack = torch.stack([per_modal_skips[i][-1] for i in range(num_modals)], dim=1)
        bottleneck, ground_truth = self.bottleneck(bottleneck_stack, mask)

        gate = mask.view(-1, num_modals, 1, 1, 1, 1)
        combined_skips = []
        for stage_index in range(num_stages - 1):
            stacked = torch.stack([per_modal_skips[i][stage_index] for i in range(num_modals)], dim=1)
            stacked = torch.where(gate, stacked, torch.zeros_like(stacked))
            batch_size = stacked.size(0)
            combined_skips.append(stacked.reshape(batch_size, -1, *stacked.shape[3:]))
        combined_skips.append(bottleneck.reshape(bottleneck.size(0), -1, *bottleneck.shape[3:]))

        seg_pred = self.decoder(combined_skips)

        if not self.is_training:
            return seg_pred
        return seg_pred, bottleneck, ground_truth

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


Model = M3FeCon


__all__ = ["M3FeCon", "Model"]
