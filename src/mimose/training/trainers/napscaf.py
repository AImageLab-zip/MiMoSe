from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from mimose.datasets.napscaf_slices import NAPSCAFSliceDataset
from mimose.training.trainers.mstkdnet import MSTKDTrainer

DEFAULT_SLICES_PER_PATCH = 16
SLICE_SAMPLING_MODES = ("tumor", "random")


class NAPSCAFTrainer(MSTKDTrainer):
    """Trains the 2D NAP-SCAF network on axial slices.

    Validation (slice-wise ``predict`` over the whole volume) and checkpointing
    are MST-KDNet's. Two training-data modes:

    - ``legacy_slices: true`` (legacy ``preprocess_brats_png`` + ``BraTSPNGDataset``): a fixed
      set of at most ``slices_per_case`` tumor slices per case chosen once with ``slice_seed``,
      each a full slice resized to ``slice_size``, all modalities present, no augmentation;
      batches are ``batch_size`` shuffled slices (so from different cases).
    - otherwise: each step draws ``slices_per_patch`` axial slices from every 3D patch (tumor
      slices first by default) and trains on them with their patch's modality mask.

    ``final_from_best`` exports ``best.pth`` (legacy evaluates its best checkpoint).
    ``fp32_validation`` disables autocast during validation (legacy only autocasts training).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        trainer_kwargs = dict(kwargs.get("custom_trainer_kwargs") or {})
        self.slices_per_patch = int(trainer_kwargs.get("slices_per_patch", DEFAULT_SLICES_PER_PATCH))
        if self.slices_per_patch < 1:
            raise ValueError("slices_per_patch must be >= 1")
        self.slice_sampling = str(trainer_kwargs.get("slice_sampling", "tumor"))
        if self.slice_sampling not in SLICE_SAMPLING_MODES:
            raise ValueError(f"slice_sampling must be one of {SLICE_SAMPLING_MODES}, got {self.slice_sampling!r}")
        self.legacy_slices = bool(trainer_kwargs.get("legacy_slices", False))
        self.slices_per_case = int(trainer_kwargs.get("slices_per_case", 5))
        self.slice_seed = int(trainer_kwargs.get("slice_seed", 7))
        self.min_nonzero_ratio = float(trainer_kwargs.get("min_nonzero_ratio", 0.05))
        self.slice_size = int(trainer_kwargs.get("slice_size", 256))
        self.final_from_best = bool(trainer_kwargs.get("final_from_best", False))
        self.fp32_validation = bool(trainer_kwargs.get("fp32_validation", False))
        super().__init__(*args, **kwargs)

    def build_datasets(self) -> tuple[Dataset, Dataset]:
        train_set, val_set = super().build_datasets()
        if not self.legacy_slices:
            return train_set, val_set
        train_set = NAPSCAFSliceDataset(
            root=self.input_dir,
            split=self.train_split,
            slices_per_case=self.slices_per_case,
            seed=self.slice_seed,
            min_nonzero_ratio=self.min_nonzero_ratio,
            image_size=self.slice_size,
        )
        return train_set, val_set

    def save_final_checkpoint(self) -> Path:
        best_path = self.checkpoint_dir / "best.pth"
        if self.final_from_best and best_path.is_file():
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            self._model_for_state().load_state_dict(checkpoint["state_dict"])
        return super().save_final_checkpoint()

    def val_epoch(self, epoch: int) -> dict[str, float]:
        if not self.fp32_validation:
            return super().val_epoch(epoch)
        # Legacy autocasts only the training step; validation runs in fp32.
        amp_enabled, self.amp_enabled = self.amp_enabled, False
        try:
            return super().val_epoch(epoch)
        finally:
            self.amp_enabled = amp_enabled

    def train_epoch(self, epoch: int) -> dict[str, float]:
        if self.train_loader is None:
            raise RuntimeError("train_loader must be initialized before training")

        self.model.train()
        self._set_aux_training_flag(True)
        steps = self.iter_per_epoch if self.iter_per_epoch is not None else len(self.train_loader)
        totals: dict[str, float] = {}

        iterations = 0
        with self._progress(disable=not self.is_main_process) as progress:
            task_id = progress.add_task(
                f"Train {epoch + 1}/{self.num_epochs}",
                total=steps,
                metrics="",
                vram=self._vram_text(),
            )
            for _ in range(steps):
                batch = self._next_train_batch()
                metrics = self._train_step(batch, epoch)
                iterations += 1
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + value
                progress.update(
                    task_id,
                    advance=1,
                    vram=self._vram_text(),
                    metrics=(
                        f"loss {metrics['loss']:.4f}  "
                        f"avg {totals['loss'] / iterations:.4f}"
                    ),
                )

                if self.debug:
                    break

        return {
            key: value / max(iterations, 1)
            for key, value in totals.items()
        }

    def _sample_slice_indices(self, seg: torch.Tensor) -> torch.Tensor:
        """Axial (last-dim) slice indices for one [H, W, D] label patch."""
        depth = seg.size(-1)
        count = min(self.slices_per_patch, depth)
        order = torch.randperm(depth, device=seg.device)
        if self.slice_sampling == "tumor":
            has_tumor = (seg > 0).flatten(0, 1).any(dim=0)[order]
            # Tumor slices first (in random order), then the rest as filler.
            order = torch.cat([order[has_tumor], order[~has_tumor]])
        return order[:count]

    def _slice_batch(
        self, images: torch.Tensor, seg: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[B, 4, H, W, D] patches -> [N, 4, H, W] slices, [N, H, W] labels, [N, 4] masks."""
        image_slices, label_slices, mask_slices = [], [], []
        for sample_images, sample_seg, sample_mask in zip(images, seg, mask):
            labels = sample_seg[0]
            indices = self._sample_slice_indices(labels)
            image_slices.append(sample_images[..., indices].permute(3, 0, 1, 2))
            label_slices.append(labels[..., indices].permute(2, 0, 1))
            mask_slices.append(sample_mask.expand(indices.numel(), -1))
        return torch.cat(image_slices), torch.cat(label_slices), torch.cat(mask_slices)

    def _wandb_config_payload(self) -> dict[str, Any]:
        payload = super()._wandb_config_payload()
        payload.update(
            {
                "slices_per_patch": self.slices_per_patch,
                "slice_sampling": self.slice_sampling,
                "legacy_slices": self.legacy_slices,
                "slices_per_case": self.slices_per_case,
                "slice_seed": self.slice_seed,
                "slice_size": self.slice_size,
                "final_from_best": self.final_from_best,
                "fp32_validation": self.fp32_validation,
            }
        )
        return payload

    def _train_step(self, batch: dict[str, Any], epoch: int) -> dict[str, float]:
        images = batch["images"].to(self.device, non_blocking=True)
        seg = batch["seg"].to(self.device, non_blocking=True).long()
        mask = batch["mask"].to(self.device, non_blocking=True).bool()
        if self.legacy_slices:
            slices, labels, slice_mask = images, seg, mask
        else:
            slices, labels, slice_mask = self._slice_batch(images, seg, mask)
        with self._autocast_context():
            logits = self.model(slices, slice_mask)
            metrics = self._loss_impl().training_loss(logits, labels)
            loss = metrics["loss"]

        self._backward_step(loss)
        return {key: float(value.item()) for key, value in metrics.items()}
