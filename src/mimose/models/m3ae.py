from __future__ import annotations

import torch
import torch.nn as nn

from mimose.models.abstract_model import AbstractModel
from mimose.models.region_utils import regions_to_label_one_hot

num_modals = 4
init_channels = 16
input_patch_size = 128


class BasicBlock(nn.Module):
    """Pre-activation GroupNorm residual block, matching legacy M3AE's ``BasicBlock``."""

    def __init__(self, in_channels: int, out_channels: int, n_groups: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(self.act1(self.norm1(x)))
        out = self.conv2(self.act2(self.norm2(out)))
        return out + residual


class UNet3D(nn.Module):
    """Standard 3D U-Net trunk with additive (not concatenated) skip
    connections, matching legacy M3AE's ``UNet3D_g``.

    Returns the three decoder feature volumes ``(u2, u3, u4)`` (with
    ``channels``, ``channels*2`` and ``channels*4`` output channels
    respectively) rather than a specific task head, since ``M3AE`` attaches
    multiple heads (segmentation, deep-supervision, reconstruction) to the same
    trunk for its two-stage training. ``u2`` is the full-resolution trunk
    output; ``u3``/``u4`` feed the deep-supervision heads.
    """

    def __init__(self, in_channels: int, channels: int = init_channels) -> None:
        super().__init__()
        self.conv1a = nn.Conv3d(in_channels, channels, kernel_size=3, padding=1)
        self.conv1b = BasicBlock(channels, channels)
        self.down1 = nn.Conv3d(channels, channels * 2, kernel_size=3, stride=2, padding=1)

        self.conv2a = BasicBlock(channels * 2, channels * 2)
        self.conv2b = BasicBlock(channels * 2, channels * 2)
        self.down2 = nn.Conv3d(channels * 2, channels * 4, kernel_size=3, stride=2, padding=1)

        self.conv3a = BasicBlock(channels * 4, channels * 4)
        self.conv3b = BasicBlock(channels * 4, channels * 4)
        self.down3 = nn.Conv3d(channels * 4, channels * 8, kernel_size=3, stride=2, padding=1)

        self.conv4a = BasicBlock(channels * 8, channels * 8)
        self.conv4b = BasicBlock(channels * 8, channels * 8)
        self.conv4c = BasicBlock(channels * 8, channels * 8)
        self.conv4d = BasicBlock(channels * 8, channels * 8)

        self.up4conv = nn.Conv3d(channels * 8, channels * 4, kernel_size=1)
        self.up4 = nn.Upsample(scale_factor=2)
        self.up4block = BasicBlock(channels * 4, channels * 4)

        self.up3conv = nn.Conv3d(channels * 4, channels * 2, kernel_size=1)
        self.up3 = nn.Upsample(scale_factor=2)
        self.up3block = BasicBlock(channels * 2, channels * 2)

        self.up2conv = nn.Conv3d(channels * 2, channels, kernel_size=1)
        self.up2 = nn.Upsample(scale_factor=2)
        self.up2block = BasicBlock(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.conv1b(self.conv1a(x))
        c1d = self.down1(c1)

        c2 = self.conv2b(self.conv2a(c1d))
        c2d = self.down2(c2)

        c3 = self.conv3b(self.conv3a(c2d))
        c3d = self.down3(c3)

        c4 = self.conv4b(self.conv4a(c3d))
        c4 = self.conv4d(self.conv4c(c4))

        u4 = self.up4block(self.up4(self.up4conv(c4)) + c3)
        u3 = self.up3block(self.up3(self.up3conv(u4)) + c2)
        u2 = self.up2block(self.up2(self.up2conv(u3)) + c1)

        return u2, u3, u4


class M3AE(AbstractModel):
    """Implementation of M3AE introduced in [1], following legacy M3AE (``legacy/m3ae``).

    [1] Liu, et al. "M3AE: Multimodal Representation Learning for Brain
        Tumor Segmentation with Missing Modalities". AAAI 2023.

    Missing modalities are substituted with a learned full-volume template (``limage``, the
    "model inversion" image, ``randn`` init as upstream ``Unet_missing``) sliced at the crop or
    sliding-window location, then all 4 channels are jointly encoded by one shared U-Net.

    Like upstream, the segmentation heads predict the three BraTS regions ``[ET, TC, WT]`` as
    independent sigmoid logits (``out_channels=3``), trained with ``EDiceLoss``. Only at the
    output boundary (``mode="segment"`` / ``predict``) are the thresholded regions turned into a
    4-class one-hot label volume, so the framework's argmax-based evaluation applies unchanged.

    Modes: ``"segment"`` (default) one-hot labels; ``"segment_logits"`` the full-resolution region
    logits; ``"segment_train"`` the deep-supervision region logits ``[out4, out3, uout]`` (upstream
    order; ``out4`` also feeds the cross-view consistency MSE); ``"reconstruct"`` the 4-channel
    pretraining reconstruction (upstream's pretraining ``up1conv``, discarded at fine-tuning).
    """

    def __init__(
        self,
        num_cls: int = 4,
        num_regions: int = 3,
        template_shape: tuple[int, int, int] | list[int] = (240, 240, 155),
    ) -> None:
        super().__init__()
        self.num_cls = num_cls
        self.num_regions = num_regions
        self.template_shape = tuple(int(dim) for dim in template_shape)
        # Upstream: nn.Parameter(torch.randn((1, 4, 155, 240, 240))), sliced at each crop location.
        self.limage = nn.Parameter(torch.randn(1, num_modals, *self.template_shape))
        self.unet = UNet3D(in_channels=num_modals)
        self.seg_layer = nn.Conv3d(init_channels, num_regions, kernel_size=1)
        self.recon_layer = nn.Conv3d(init_channels, num_modals, kernel_size=1)

        # Deep-supervision heads on the two coarser decoder levels, matching
        # legacy ``UNet3D_g`` (ds_out convs at init_channels*4 / *2 then
        # trilinear upsampling back to full resolution).
        self.ds_out4 = nn.Conv3d(init_channels * 4, num_regions, kernel_size=1)
        self.ds_up4 = nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)
        self.ds_out3 = nn.Conv3d(init_channels * 2, num_regions, kernel_size=1)
        self.ds_up3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        # No custom init: upstream UNet3D_g uses PyTorch defaults.

    def template_at(self, location: torch.Tensor | None, batch_size: int, size: tuple[int, int, int]) -> torch.Tensor:
        """``limage`` sliced at each sample's ``location`` (``[B, 3]`` start voxel; ``None`` = origin)."""
        if location is None:
            location = torch.zeros(batch_size, 3, dtype=torch.long)
        location = torch.as_tensor(location).reshape(-1, 3).tolist()
        if len(location) == 1 and batch_size > 1:
            location = location * batch_size
        slices = []
        for h, w, d in location:
            if h + size[0] > self.template_shape[0] or w + size[1] > self.template_shape[1] or d + size[2] > self.template_shape[2]:
                raise RuntimeError(
                    f"M3AE template {self.template_shape} cannot hold a {size} window at {(h, w, d)}"
                )
            slices.append(self.limage[:, :, h : h + size[0], w : w + size[1], d : d + size[2]])
        return torch.cat(slices, dim=0)

    def fill_missing(self, x: torch.Tensor, mask: torch.Tensor, location: torch.Tensor | None = None) -> torch.Tensor:
        """Upstream ``x * mask + limage[location] * (1 - mask)``; the template is detached at inference."""
        if x.size(1) != num_modals:
            raise RuntimeError(f"M3AE expects {num_modals} input modalities, got {x.size(1)}")
        voxel_gate = mask.ndim == 5
        if voxel_gate:
            if mask.shape[1] != num_modals or mask.shape[-3:] != x.shape[-3:]:
                raise RuntimeError(f"M3AE expects a voxel gate shaped like x, got {tuple(mask.shape)}")
        elif mask.ndim != 2 or mask.size(1) != num_modals:
            raise RuntimeError(f"M3AE expects mask shape [B, {num_modals}], got {tuple(mask.shape)}")
        # A [B, M, D, H, W] gate is the MAE pretraining patch mask; a [B, M] mask drops whole modalities.
        gate = mask.to(x.dtype) if voxel_gate else mask.view(-1, num_modals, 1, 1, 1).to(x.dtype)
        template = self.template_at(location, x.size(0), tuple(x.shape[-3:]))
        if not self.training:
            template = template.detach()
        return x * gate + template.to(x.dtype) * (1 - gate)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mode: str = "segment",
        location: torch.Tensor | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """All modes go through this single ``forward`` so trainer code can always call
        ``self.model(...)`` and keep ``DistributedDataParallel``'s gradient-sync hooks."""
        if tuple(x.shape[-3:]) != (input_patch_size,) * 3:
            raise RuntimeError(
                f"M3AE requires spatial dimensions of exactly {(input_patch_size,) * 3}, got {tuple(x.shape[-3:])}"
            )
        u2, u3, u4 = self.unet(self.fill_missing(x, mask, location))
        if mode == "reconstruct":
            return self.recon_layer(u2)
        uout = self.seg_layer(u2)
        if mode == "segment_train":
            out4 = self.ds_up4(self.ds_out4(u4))
            out3 = self.ds_up3(self.ds_out3(u3))
            return [out4, out3, uout]
        if mode == "segment_logits":
            return uout
        return regions_to_label_one_hot(uout, self.num_cls)

    def reconstruct(self, x: torch.Tensor, mask: torch.Tensor, location: torch.Tensor | None = None) -> torch.Tensor:
        """Non-distributed convenience wrapper around ``forward(..., mode="reconstruct")``."""
        return self(x, mask, mode="reconstruct", location=location)

    def predict(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Sliding window (50% overlap) averaging the region logits, as upstream's MONAI
        ``sliding_window_inference`` does, with ``limage`` sliced at each window's position in the
        (brain-cropped) volume, then thresholded into labels."""
        _, _, height, width, depth = images.shape
        h_starts = self._window_starts(height)
        w_starts = self._window_starts(width)
        d_starts = self._window_starts(depth)
        logits = torch.zeros(images.size(0), self.num_regions, height, width, depth, device=images.device)
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
                    location = torch.tensor([[h, w, d]], dtype=torch.long)
                    logits[window] += self(images[window], mask, mode="segment_logits", location=location).float()
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


Model = M3AE


__all__ = ["M3AE", "Model"]
