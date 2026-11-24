from __future__ import annotations

import torch


def legacy_intensity_flip(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    shift: float = 0.1,
    scale: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy M3AE/MCPL fine-tuning augmentation ``Compose([RandomIntensityChange((0.1, 0.1)),
    RandomFlip(0)])`` (``legacy/m3ae/dataset/transforms.py``), applied to a whole batch at once.

    Upstream runs it on a ``(B, X, Y, Z, C)`` array with factors shaped ``[1, X, 1, 1, C]``, so the
    scale/shift is drawn per channel *and* per index of the first spatial axis, shared across the
    batch; that layout is kept. Labels are only flipped. Each spatial axis flips with p=0.5, the
    same flips for every sample (both views of a sample stay aligned).
    """
    batch, channels, depth = images.shape[:3]
    factor_shape = (1, channels, depth, 1, 1)
    scale_factor = torch.empty(factor_shape, device=images.device, dtype=images.dtype).uniform_(1.0 - scale, 1.0 + scale)
    shift_factor = torch.empty(factor_shape, device=images.device, dtype=images.dtype).uniform_(-shift, shift)
    images = images * scale_factor + shift_factor

    flip_dims = [dim for dim in (2, 3, 4) if bool(torch.rand(()) < 0.5)]
    if flip_dims:
        images = torch.flip(images, dims=flip_dims)
        labels = torch.flip(labels, dims=flip_dims)
    return images.contiguous(), labels.contiguous()


__all__ = ["legacy_intensity_flip"]
