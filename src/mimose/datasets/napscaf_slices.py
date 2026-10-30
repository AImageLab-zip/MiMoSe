from __future__ import annotations

import fcntl
import json
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

CACHE_VERSION = 1


def _select_slices(seg: np.ndarray, images: np.ndarray, min_nonzero_ratio: float) -> list[int]:
    """Legacy ``valid_slices``: tumor slices whose every modality has >= ``min_nonzero_ratio`` nonzero pixels."""
    has_tumor = (seg > 0).any(axis=(0, 1))
    nonzero_ratio = (images != 0).mean(axis=(1, 2))  # [C, D]
    ok = has_tumor & (nonzero_ratio >= min_nonzero_ratio).all(axis=0)
    return [int(z) for z in np.flatnonzero(ok)]


def _case_candidates(path: str, min_nonzero_ratio: float) -> tuple[str, list[int]]:
    with np.load(path) as data:
        seg = data["seg"].reshape(data["seg"].shape[-3:])
        return Path(path).stem, _select_slices(seg, data["images"], min_nonzero_ratio)


def _case_slices(path: str, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        images = data["images"][..., indices]  # [C, H, W, k], values k/255
        seg = data["seg"].reshape(data["seg"].shape[-3:])[..., indices]
    images = np.rint(np.moveaxis(images, -1, 0) * 255.0).astype(np.uint8)  # [k, C, H, W]
    return images, np.moveaxis(seg, -1, 0).astype(np.uint8)


def build_slice_cache(
    root: Path,
    cache_dir: Path,
    *,
    slices_per_case: int,
    seed: int,
    min_nonzero_ratio: float,
    num_workers: int = 8,
) -> None:
    """Legacy ``preprocess_brats_png.main``: per case (sorted), keep the valid slices and, when there
    are more than ``slices_per_case``, ``sorted(random.sample(...))`` with one ``random.seed(seed)``
    for the whole run. Slices are stored as uint8, like legacy's PNGs."""
    cases = sorted(str(p) for p in root.glob("*.npz"))
    if not cases:
        raise FileNotFoundError(f"No preprocessed cases in {root}")
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        candidates = list(pool.map(_case_candidates, cases, [min_nonzero_ratio] * len(cases), chunksize=8))

    rng = random.Random(seed)
    selection: list[tuple[str, str, list[int]]] = []
    for path, (name, indices) in zip(cases, candidates):
        if len(indices) > slices_per_case:
            indices = sorted(rng.sample(indices, slices_per_case))
        if indices:
            selection.append((path, name, indices))

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        extracted = list(pool.map(_case_slices, [s[0] for s in selection], [s[2] for s in selection], chunksize=8))

    tmp_dir = cache_dir.with_name(cache_dir.name + f".tmp{os.getpid()}")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    np.save(tmp_dir / "images.npy", np.concatenate([e[0] for e in extracted]))
    np.save(tmp_dir / "seg.npy", np.concatenate([e[1] for e in extracted]))
    index = {
        "version": CACHE_VERSION,
        "slices_per_case": slices_per_case,
        "seed": seed,
        "min_nonzero_ratio": min_nonzero_ratio,
        "cases": [s[1] for s in selection for _ in s[2]],
        "z": [z for s in selection for z in s[2]],
    }
    (tmp_dir / "index.json").write_text(json.dumps(index))
    os.replace(tmp_dir, cache_dir)


def ensure_slice_cache(root: Path, *, slices_per_case: int, seed: int, min_nonzero_ratio: float) -> Path:
    cache_dir = root / f"napscaf_slices_k{slices_per_case}_s{seed}_nz{min_nonzero_ratio:g}"
    lock_path = root / f".{cache_dir.name}.lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (cache_dir / "index.json").is_file():
            build_slice_cache(
                root, cache_dir, slices_per_case=slices_per_case, seed=seed, min_nonzero_ratio=min_nonzero_ratio
            )
    return cache_dir


class NAPSCAFSliceDataset(Dataset):
    """Legacy NAP-SCAF training data (``BraTSPNGDataset`` over ``preprocess_brats_png`` output).

    A fixed set of axial slices (at most ``slices_per_case`` tumor slices per case, chosen once
    with ``seed``), restricted to the cases of ``split``. Each item is one full slice resized to
    ``image_size`` (bilinear for images, nearest for labels, as legacy's ``cv2.resize``), with all
    four modalities present and no augmentation.
    """

    def __init__(
        self,
        root: str | Path,
        split: list[dict[str, Any]],
        *,
        slices_per_case: int = 5,
        seed: int = 7,
        min_nonzero_ratio: float = 0.05,
        image_size: int = 256,
    ) -> None:
        self.cache_dir = ensure_slice_cache(
            Path(root), slices_per_case=slices_per_case, seed=seed, min_nonzero_ratio=min_nonzero_ratio
        )
        index = json.loads((self.cache_dir / "index.json").read_text())
        wanted = {str(sample["sub"]) for sample in split}
        self.rows = [i for i, case in enumerate(index["cases"]) if case in wanted]
        if not self.rows:
            raise ValueError(f"No cached NAP-SCAF slices for the {len(wanted)} cases of this split")
        self.cases = index["cases"]
        self.image_size = int(image_size)
        self._images: np.ndarray | None = None
        self._seg: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def _arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self._images is None or self._seg is None:
            self._images = np.load(self.cache_dir / "images.npy", mmap_mode="r")
            self._seg = np.load(self.cache_dir / "seg.npy", mmap_mode="r")
        return self._images, self._seg

    def __getitem__(self, index: int) -> dict[str, Any]:
        images_all, seg_all = self._arrays()
        row = self.rows[index]
        images = torch.from_numpy(np.array(images_all[row])).float()[None]  # [1, C, H, W] in 0..255
        seg = torch.from_numpy(np.array(seg_all[row])).float()[None, None]
        size = (self.image_size, self.image_size)
        if tuple(images.shape[-2:]) != size:
            # cv2.INTER_LINEAR on the uint8 PNG returns rounded uint8; torch bilinear with
            # align_corners=False uses the same half-pixel sampling.
            images = F.interpolate(images, size=size, mode="bilinear", align_corners=False).round().clamp(0, 255)
            seg = F.interpolate(seg, size=size, mode="nearest")
        return {
            "sub": self.cases[row],
            "images": images[0] / 255.0,
            "seg": seg[0, 0].long(),
            "mask": torch.ones(images.shape[1], dtype=torch.bool),
        }


__all__ = ["NAPSCAFSliceDataset", "build_slice_cache", "ensure_slice_cache"]
