from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mimose.training.trainers.mstkdnet import MSTKDTrainer

DEFAULT_CURRICULUM_EPOCHS = (10, 30, 50)
DEFAULT_MISSING_DROP_P = (0.2, 0.3, 0.4)


class ALoRAMoETrainer(MSTKDTrainer):
    """Legacy ALoRA-MoE training (``legacy/ALoRA-MoE/train.py`` + ``lora_moe_hgd/engine.py``).

    - Curriculum: ``curriculum_stage`` over ``curriculum_epochs`` picks stage 0/1/2; stage ``s``
      trains the first ``s + 1`` classes of the loss's ``curriculum_classes``.
    - Missing modalities (``simulate_missing_modalities``): each modality is dropped independently
      with the stage's ``missing_drop_p``, and one random modality is restored if all were dropped.
      Use with ``train_masking_mode: full`` so this is the only masking applied.
    - ``final_from_best`` exports ``best.pth`` (legacy infers from ``checkpoints/best.pt``).

    Data, validation and sliding-window inference are MST-KDNet's.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        trainer_kwargs = dict(kwargs.get("custom_trainer_kwargs") or {})
        self.curriculum_epochs = tuple(
            int(e) for e in trainer_kwargs.get("curriculum_epochs", DEFAULT_CURRICULUM_EPOCHS)
        )
        self.missing_drop_p = tuple(
            float(p) for p in trainer_kwargs.get("missing_drop_p", DEFAULT_MISSING_DROP_P)
        )
        if len(self.curriculum_epochs) < 2 or not self.missing_drop_p:
            raise ValueError("curriculum_epochs needs >= 2 milestones and missing_drop_p >= 1 value")
        self.final_from_best = bool(trainer_kwargs.get("final_from_best", False))
        super().__init__(*args, **kwargs)

    def curriculum_stage(self, epoch: int) -> int:
        # Legacy schedule.curriculum_stage: only the first two milestones matter.
        if epoch < self.curriculum_epochs[0]:
            return 0
        if epoch < self.curriculum_epochs[1]:
            return 1
        return 2

    def _drop_p(self, stage: int) -> float:
        return self.missing_drop_p[min(stage, len(self.missing_drop_p) - 1)]

    def _simulate_missing_modalities(self, batch_size: int, num_modalities: int, drop_p: float) -> torch.Tensor:
        masks = []
        for _ in range(batch_size):
            mask = torch.ones(num_modalities, dtype=torch.bool)
            for c in range(num_modalities):
                if torch.rand(1).item() < drop_p:
                    mask[c] = False
            if mask.sum() == 0:
                mask[torch.randint(0, num_modalities, (1,))] = True
            masks.append(mask)
        return torch.stack(masks)

    def save_final_checkpoint(self) -> Path:
        best_path = self.checkpoint_dir / "best.pth"
        if self.final_from_best and best_path.is_file():
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            self._model_for_state().load_state_dict(checkpoint["state_dict"])
        return super().save_final_checkpoint()

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

    def _wandb_config_payload(self) -> dict[str, Any]:
        payload = super()._wandb_config_payload()
        payload.update(
            {
                "curriculum_epochs": list(self.curriculum_epochs),
                "missing_drop_p": list(self.missing_drop_p),
                "final_from_best": self.final_from_best,
            }
        )
        return payload

    def _train_step(self, batch: dict[str, Any], epoch: int) -> dict[str, float]:
        stage = self.curriculum_stage(epoch)
        images = batch["images"].to(self.device, non_blocking=True)
        seg = batch["seg"].to(self.device, non_blocking=True).long()
        mask = batch["mask"].bool() & self._simulate_missing_modalities(
            images.size(0), images.size(1), self._drop_p(stage)
        )
        mask = mask.to(self.device, non_blocking=True)
        with self._autocast_context():
            logits = self.model(images, mask)
            metrics = self._loss_impl().training_loss(logits, seg, stage)
            loss = metrics["loss"]

        self._backward_step(loss)
        out = {key: float(value.item()) for key, value in metrics.items()}
        out["stage"] = float(stage)
        return out
