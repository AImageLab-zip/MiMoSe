from __future__ import annotations

import torch
import torch.nn.functional as F

from missingbench.training.transforms.base_transforms import TransformManager

# Mirror combinations and their cumulative probabilities from legacy ShaSpec BraTSDataSet.__getitem__:
# 30% no flip, then 10% for each of the 7 non-empty axis subsets.
_MIRROR_CHOICES = (
    (0.3, ()),
    (0.4, (3,)),
    (0.5, (2,)),
    (0.6, (1,)),
    (0.7, (2, 3)),
    (0.8, (1, 3)),
    (0.9, (1, 2)),
    (1.0, (1, 2, 3)),
)


class ShaSpecTransform:
    """Training augmentation of legacy ShaSpec (``BraTSDataSet`` with ``--random_scale --random_mirror``).

    - Crop: with probability 0.5 centred on a random voxel of a random tumour region (ET, WT or TC),
      otherwise uniform; regions without foreground fall back to a uniform crop.
    - Scale: with probability 0.5, crop a box scaled by U(0.9, 1.1) and resize it back to ``crop_size``
      (linear for images, nearest for labels).
    - Mirror: upstream's fixed table of axis-flip combinations. No rotation or intensity jitter.
    """

    def __init__(self, crop_size: tuple[int, int, int], scale: bool = True, mirror: bool = True) -> None:
        self.crop_size = tuple(int(size) for size in crop_size)
        self.scale = scale
        self.mirror = mirror

    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        images = images.to(dtype=torch.float32)
        labels_dtype = labels.dtype

        scaler = 1.0
        if self.scale and torch.rand(()) < 0.5:
            scaler = float(torch.empty(()).uniform_(0.9, 1.1))
        box = tuple(min(int(size * scaler), dim) for size, dim in zip(self.crop_size, images.shape[1:]))

        starts = self._crop_starts(labels, box)
        slices = tuple(slice(start, start + size) for start, size in zip(starts, box))
        images = images[(slice(None), *slices)]
        labels = labels[(slice(None), *slices)]

        if self.mirror:
            draw = float(torch.rand(()))
            dims = next(axes for threshold, axes in _MIRROR_CHOICES if draw <= threshold)
            if dims:
                images = torch.flip(images, dims=dims)
                labels = torch.flip(labels, dims=dims)

        if box != self.crop_size:
            images = F.interpolate(images.unsqueeze(0), size=self.crop_size, mode="trilinear", align_corners=False)[0]
            labels = F.interpolate(labels.unsqueeze(0).float(), size=self.crop_size, mode="nearest")[0]

        return images.contiguous(), labels.to(dtype=labels_dtype).contiguous()

    def _crop_starts(self, labels: torch.Tensor, box: tuple[int, int, int]) -> list[int]:
        spatial = labels.shape[1:]
        for dim, size in zip(spatial, box):
            if dim < size:
                raise ValueError(f"Crop {box} is larger than input spatial shape {tuple(spatial)}")

        if torch.rand(()) < 0.5:
            seg = labels[0]
            regions = (seg == 3, seg >= 1, (seg == 1) | (seg == 3))  # ET, WT, TC
            locations = regions[int(torch.randint(len(regions), ()))].nonzero()
            if len(locations) > 0:
                center = locations[int(torch.randint(len(locations), ()))].tolist()
                return [
                    min(max(c - size // 2, 0), dim - size)
                    for c, size, dim in zip(center, box, spatial)
                ]

        return [int(torch.randint(0, dim - size + 1, ())) for dim, size in zip(spatial, box)]


class ShaSpecTransformManager(TransformManager):
    def __init__(self, crop_size: tuple[int, int, int] = (80, 80, 80)) -> None:
        self.crop_size = tuple(int(dim) for dim in crop_size)
        super().__init__()

    def _setup_transforms(self) -> None:
        self.train_transforms = {"paired": ShaSpecTransform(crop_size=self.crop_size)}
        self.test_transforms = {"paired": _identity}


def _identity(images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return images.to(dtype=torch.float32).contiguous(), labels.contiguous()


__all__ = ["ShaSpecTransform", "ShaSpecTransformManager"]
