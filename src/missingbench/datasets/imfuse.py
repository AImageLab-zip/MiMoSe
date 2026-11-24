from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from missingbench.datasets.base import BaseDataset
from missingbench.datasets.config import MaskingMode


MASK_PATTERNS = torch.tensor(
    [
        [True, False, False, False],
        [False, True, False, False],
        [False, False, True, False],
        [False, False, False, True],
        [True, True, False, False],
        [True, False, True, False],
        [True, False, False, True],
        [False, True, True, False],
        [False, True, False, True],
        [False, False, True, True],
        [True, True, True, False],
        [True, True, False, True],
        [True, False, True, True],
        [False, True, True, True],
        [True, True, True, True],
    ],
    dtype=torch.bool,
)


class IMFuseDataset(BaseDataset):
    def __init__(
        self,
        root: str | Path,
        masking_mode: MaskingMode,
        split: list[dict[str, Any]],
        sample_transform: Callable[[torch.Tensor, torch.Tensor], tuple[Any, Any]] | None = None,
        image_transforms: Callable[[torch.Tensor], Any] | None = None,
        target_transforms: Callable[[torch.Tensor], Any] | None = None,
        mask_sampling: str = "subset_uniform",
    ) -> None:
        super().__init__(
            root=root,
            split=split,
            masking_mode=masking_mode,
            mask_patterns=MASK_PATTERNS,
            sample_transform=sample_transform,
            image_transforms=image_transforms,
            target_transforms=target_transforms,
            mask_sampling=mask_sampling,
        )

    def build_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        subject_name = str(sample["sub"])
        subject_path = self.root / f"{subject_name}.npz"
        if not subject_path.is_file():
            raise FileNotFoundError(f"Sample file not found: {subject_path}")
        return {
            "sub": subject_path,
            "mask": sample.get("mask"),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        with np.load(sample["sub"]) as data:
            images = torch.from_numpy(data["images"]).clone().float()
            seg = self._normalize_seg_shape(torch.from_numpy(data["seg"]).clone().float())

        images, seg = self.apply_transforms(images, seg)
        seg = self._normalize_seg_shape(seg)

        item: dict[str, Any] = {
            "sub": sample["sub"].stem,
            "images": images,
            "seg": seg,
            "mask": self.resolve_mask(sample["mask"]),
        }
        return item

    @staticmethod
    def _normalize_seg_shape(seg: torch.Tensor) -> torch.Tensor:
        if seg.ndim == 3:
            return seg.unsqueeze(0)
        if seg.ndim == 4 and seg.shape[0] == 1:
            return seg
        raise ValueError(f"IMFuseDataset expects seg shape [H, W, D] or [1, H, W, D], got {tuple(seg.shape)}")
