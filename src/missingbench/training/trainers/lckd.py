from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from missingbench.datasets import IMFuseDataset, MaskingMode
from missingbench.enums import Precision, TransformKind
from missingbench.losses.config import LossConfig
from missingbench.models.config import ModelConfig
from missingbench.training.config import OptimizerConfig, SchedulerConfig
from missingbench.training.trainers.base_trainer import CONSOLE, BaseTrainer
from missingbench.training.transforms import build_transform_manager
from missingbench.training.transforms.base_transforms import TransformManager

DEFAULT_PATCH_SIZE = 128


class LCKDTrainer(BaseTrainer):
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        custom_trainer_kwargs: dict[str, Any] | None = None,
        model_config: ModelConfig | None = None,
        loss_config: LossConfig | None = None,
        optimizer_config: OptimizerConfig | None = None,
        scheduler_config: SchedulerConfig | None = None,
        transform_manager: TransformManager | None = None,
        num_epochs: int = 1,
        validation_every: int = 1,
        batch_size: int | None = None,
        num_workers: int | None = None,
        precision: str | Precision = Precision.FP32,
        compile: bool = False,
        resume: bool = False,
        try_resume: bool = False,
        seed: int | None = None,
        pretrain: str | Path | None = None,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        wandb_mode: str | None = None,
        wandb_run_name: str | None = None,
        dataset_type: str | None = None,
        push_to_hf: bool = False,
        hf_repo: str | None = None,
    ) -> None:
        trainer_kwargs = dict(custom_trainer_kwargs or {})
        self.iter_per_epoch = (
            int(trainer_kwargs["iter_per_epoch"])
            if trainer_kwargs.get("iter_per_epoch") is not None
            else None
        )
        self.patch_size = int(trainer_kwargs.get("patch_size", DEFAULT_PATCH_SIZE))
        self.debug = bool(trainer_kwargs.get("debug", False))
        self.transform_kind = TransformKind(
            trainer_kwargs.get("transform_kind", TransformKind.IMFUSE)
        )
        self.train_masking_mode = MaskingMode(
            trainer_kwargs.get("train_masking_mode", MaskingMode.RANDOM)
        )
        self.val_masking_mode = MaskingMode(
            trainer_kwargs.get("val_masking_mode", MaskingMode.VALIDATION)
        )
        self.warmup_fraction = float(trainer_kwargs.get("warmup_fraction", 0.0))
        if not 0.0 <= self.warmup_fraction <= 1.0:
            raise ValueError(
                f"warmup_fraction must be between 0 and 1, got {self.warmup_fraction}"
            )
        self.warmup_epochs = round(self.warmup_fraction * num_epochs)
        self._in_warmup: bool | None = None
        self.best_val_dice = float("-inf")
        self._train_iterator: Any | None = None

        effective_batch_size = 1 if batch_size is None else batch_size
        effective_num_workers = 0 if num_workers is None else num_workers

        super().__init__(
            input_dir=input_dir,
            output_dir=output_dir,
            custom_trainer_kwargs=trainer_kwargs,
            model_config=model_config,
            loss_config=loss_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            transform_manager=transform_manager,
            num_epochs=num_epochs,
            validation_every=validation_every,
            batch_size=effective_batch_size,
            num_workers=effective_num_workers,
            precision=precision,
            compile=compile,
            resume=resume,
            try_resume=try_resume,
            seed=seed,
            pretrain=pretrain,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_mode=wandb_mode,
            wandb_run_name=wandb_run_name,
            dataset_type=dataset_type,
            push_to_hf=push_to_hf,
            hf_repo=hf_repo,
        )
        self.dataset_type = self._resolve_dataset_type(dataset_type)

        if self.pretrain is not None and self.resume is None:
            self._load_pretrain()

    def _apply_training_phase(self, epoch: int) -> None:
        in_warmup = epoch < self.warmup_epochs
        if in_warmup == self._in_warmup:
            return

        is_boundary_crossing = self._in_warmup is not None
        self._in_warmup = in_warmup
        phase_mode = MaskingMode.FULL if in_warmup else self.train_masking_mode
        self.train_set.masking_mode = phase_mode

        if is_boundary_crossing:
            # Legacy LCKD trains warmup and missing-modality as two separate
            # `train.py` runs (the second one launched with `--restart`): a
            # freshly-constructed optimizer with no carried-over momentum,
            # and an LR schedule that decays from scratch over just the new
            # phase's remaining length. Mirror that here instead of letting a
            # single optimizer/scheduler run continuously across the boundary.
            self._restart_optimizer_and_scheduler(epoch)

        if self.is_main_process:
            phase_name = "warmup (full modalities)" if in_warmup else "missing-modality"
            CONSOLE.print(
                f"[bold cyan]LCKD[/bold cyan] entering {phase_name} phase at epoch "
                f"{epoch + 1}/{self.num_epochs} (warmup_epochs={self.warmup_epochs})"
            )

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> Any | None:
        if self.scheduler_config is None or self.warmup_epochs <= 0:
            return super()._build_scheduler(optimizer)

        # The warmup phase is its own complete decay cycle (mirrors legacy's
        # separate warmup `train.py` run), so it must reach the scheduler's
        # terminal LR by the end of warmup_epochs, not by num_epochs -- else
        # the phase-boundary restart in _restart_optimizer_and_scheduler
        # kicks in mid-decay and the warmup LR never bottoms out.
        target_optimizer = optimizer or self.optimizer
        if target_optimizer is None:
            raise RuntimeError("optimizer must be initialized before building a scheduler")

        phase_kwargs = dict(self.scheduler_config.kwargs)
        for duration_key in ("total_iters", "T_max"):
            if duration_key in phase_kwargs:
                phase_kwargs[duration_key] = self.warmup_epochs

        self.scheduler = self.scheduler_config.scheduler_class(target_optimizer, **phase_kwargs)
        return self.scheduler

    def _restart_optimizer_and_scheduler(self, epoch: int) -> None:
        self._build_optimizer()

        if self.scheduler_config is None:
            self.scheduler = None
            return

        phase_kwargs = dict(self.scheduler_config.kwargs)
        remaining_epochs = self.num_epochs - epoch
        for duration_key in ("total_iters", "T_max"):
            if duration_key in phase_kwargs:
                phase_kwargs[duration_key] = remaining_epochs

        self.scheduler = self.scheduler_config.scheduler_class(self.optimizer, **phase_kwargs)

        if self.is_main_process:
            CONSOLE.print(
                f"[bold cyan]LCKD[/bold cyan] restarted optimizer/scheduler at epoch "
                f"{epoch + 1}/{self.num_epochs} (phase boundary)"
            )

    def train_epoch(self, epoch: int) -> dict[str, float]:
        if self.train_loader is None:
            raise RuntimeError("train_loader must be initialized before training")

        self._apply_training_phase(epoch)
        self.model.train()
        self._set_aux_training_flag(True)
        steps = self.iter_per_epoch if self.iter_per_epoch is not None else len(self.train_loader)
        totals = {
            "loss": 0.0,
            "cross": 0.0,
            "dice": 0.0,
            "kd": 0.0,
        }

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
                metrics = self._train_step(batch)
                iterations += 1
                for key, value in metrics.items():
                    totals[key] += value
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

    def val_epoch(self, epoch: int) -> dict[str, float]:
        if self.val_loader is None:
            raise RuntimeError("val_loader must be initialized before validation")

        self.model.eval()
        self._set_aux_training_flag(False)
        loss_sum = 0.0
        wt_sum = 0.0
        tc_sum = 0.0
        et_sum = 0.0
        etpp_sum = 0.0
        wt_hd95_sum = 0.0
        tc_hd95_sum = 0.0
        et_hd95_sum = 0.0
        etpp_hd95_sum = 0.0
        sample_count = 0

        with torch.no_grad():
            with self._progress(disable=not self.is_main_process) as progress:
                task_id = progress.add_task(
                    f"Val {epoch + 1}/{self.num_epochs}",
                    total=len(self.val_loader),
                    metrics="",
                    vram=self._vram_text(),
                )
                for batch in self.val_loader:
                    images = batch["images"].to(self.device, non_blocking=True)
                    seg = batch["seg"].to(self.device, non_blocking=True).long()
                    mask = batch["mask"].to(self.device, non_blocking=True).bool()

                    with self._autocast_context():
                        pred = self._predict_volume(images, mask)
                        target = self._seg_to_one_hot(seg)
                        seg_loss = self._segmentation_loss(pred, target)
                    prediction = pred.argmax(dim=1)
                    target_labels = seg.squeeze(1)
                    wt, tc, et, etpp = self._evaluate_scores(prediction, target_labels)
                    wt_hd95, tc_hd95, et_hd95, etpp_hd95 = self._evaluate_hd95(prediction, target_labels)

                    batch_size = images.shape[0]
                    sample_count += batch_size
                    loss_sum += float(seg_loss.item()) * batch_size
                    wt_sum += float(wt.sum().item())
                    tc_sum += float(tc.sum().item())
                    et_sum += float(et.sum().item())
                    etpp_sum += float(etpp.sum().item())
                    wt_hd95_sum += float(wt_hd95.sum().item())
                    tc_hd95_sum += float(tc_hd95.sum().item())
                    et_hd95_sum += float(et_hd95.sum().item())
                    etpp_hd95_sum += float(etpp_hd95.sum().item())
                    progress.update(
                        task_id,
                        advance=1,
                        vram=self._vram_text(),
                        metrics=(
                            f"loss {float(seg_loss.item()):.4f}  "
                            f"avg {loss_sum / max(sample_count, 1):.4f}"
                        ),
                    )

                    if self.debug:
                        break

        self.model.train()
        self._set_aux_training_flag(True)

        divisor = max(sample_count, 1)
        wt_score = wt_sum / divisor
        tc_score = tc_sum / divisor
        et_score = et_sum / divisor
        dice_score = (wt_score + tc_score + et_score) / 3.0
        return {
            "loss": loss_sum / divisor,
            "wt": wt_score,
            "tc": tc_score,
            "et": et_score,
            "etpp": etpp_sum / divisor,
            "dice": dice_score,
            "wt_hd95": wt_hd95_sum / divisor,
            "tc_hd95": tc_hd95_sum / divisor,
            "et_hd95": et_hd95_sum / divisor,
            "etpp_hd95": etpp_hd95_sum / divisor,
        }

    def build_datasets(self) -> tuple[Dataset, Dataset]:
        if self.train_split is None or self.val_split is None:
            raise RuntimeError("train_split and val_split must be loaded before building datasets")

        transform_manager = self.transform_manager
        if transform_manager is None:
            transform_manager = build_transform_manager(
                self.transform_kind,
                model_kwargs=(self.model_config.kwargs if self.model_config is not None else None),
                crop_size=(self.patch_size, self.patch_size, self.patch_size),
            )
        train_set = IMFuseDataset(
            root=self.input_dir,
            masking_mode=self.train_masking_mode,
            split=self.train_split,
            sample_transform=partial(transform_manager, mode="train"),
        )
        val_set = IMFuseDataset(
            root=self.input_dir,
            masking_mode=self.val_masking_mode,
            split=self.val_split,
            sample_transform=partial(transform_manager, mode="test"),
        )
        return train_set, val_set

    def _wandb_config_payload(self) -> dict[str, Any]:
        payload = super()._wandb_config_payload()
        payload.update(
            {
                "dataset_type": self.dataset_type,
                "iter_per_epoch": self.iter_per_epoch,
                "patch_size": self.patch_size,
                "debug": self.debug,
                "train_masking_mode": self.train_masking_mode,
                "val_masking_mode": self.val_masking_mode,
                "warmup_fraction": self.warmup_fraction,
                "warmup_epochs": self.warmup_epochs,
                "split_file": str(self.split_file),
            }
        )
        return payload

    def _train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        images = batch["images"].to(self.device, non_blocking=True)
        seg = batch["seg"].to(self.device, non_blocking=True).long()
        mask = batch["mask"].to(self.device, non_blocking=True).bool()
        with self._autocast_context():
            target = self._seg_to_one_hot(seg)
            outputs = self.model(images, mask)
            metrics = self._loss_impl().training_loss(outputs, target)
            loss = metrics["loss"]

        self._backward_step(loss)
        return {
            "loss": float(loss.item()),
            "cross": float(metrics["cross"].item()),
            "dice": float(metrics["dice"].item()),
            "kd": float(metrics["kd"].item()),
        }


__all__ = ["LCKDTrainer"]
