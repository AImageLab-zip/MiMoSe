from __future__ import annotations

from typing import Any

import numpy as np
import torch

from mimose.datasets.imfuse import IMFuseDataset


class LocatedCropDataset(IMFuseDataset):
    """``IMFuseDataset`` whose only training transform is a uniform random crop, returned together
    with its start coordinates (``location``, ``[3]`` long) so a location-dependent missing-modality
    template can be sliced at the same place (legacy M3AE ``crop_indexes``). With
    ``crop_size=None`` the full volume is returned with ``location = 0``. Augmentation, if any, is
    left to the trainer (legacy applies it after the missing modalities are filled in).
    """

    def __init__(self, *args: Any, crop_size: int | None = None, **kwargs: Any) -> None:
        kwargs.pop("sample_transform", None)
        super().__init__(*args, **kwargs)
        self.crop_size = crop_size

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        with np.load(sample["sub"]) as data:
            images = torch.from_numpy(data["images"]).clone().float()
            seg = self._normalize_seg_shape(torch.from_numpy(data["seg"]).clone().float())

        location = torch.zeros(3, dtype=torch.long)
        if self.crop_size is not None:
            spatial = images.shape[1:]
            if any(size < self.crop_size for size in spatial):
                raise ValueError(f"Crop size {self.crop_size} is larger than input spatial shape {tuple(spatial)}")
            starts = [int(torch.randint(0, size - self.crop_size + 1, ()).item()) for size in spatial]
            location = torch.tensor(starts, dtype=torch.long)
            h, w, d = starts
            c = self.crop_size
            images = images[:, h : h + c, w : w + c, d : d + c].contiguous()
            seg = seg[:, h : h + c, w : w + c, d : d + c].contiguous()

        return {
            "sub": sample["sub"].stem,
            "images": images,
            "seg": seg,
            "mask": self.resolve_mask(sample["mask"]),
            "location": location,
        }


__all__ = ["LocatedCropDataset"]
