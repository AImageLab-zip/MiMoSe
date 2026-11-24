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
from missingbench.training.trainers.base_trainer import BaseTrainer
from missingbench.training.transforms import build_transform_manager
from missingbench.training.transforms.base_transforms import TransformManager

DEFAULT_PATCH_SIZE = 80


class ShaSpecTrainer(BaseTrainer):
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
        self.dataname = str(trainer_kwargs.get("dataname", "BRATS2023"))
        self.iter_per_epoch = (
            int(trainer_kwargs["iter_per_epoch"]) if trainer_kwargs.get("iter_per_epoch") is not None else None
        )
        self.patch_size = int(trainer_kwargs.get("patch_size", DEFAULT_PATCH_SIZE))
        self.debug = bool(trainer_kwargs.get("debug", False))
        self.transform_kind = TransformKind(trainer_kwargs.get("transform_kind", TransformKind.IMFUSE))
        self.train_masking_mode = MaskingMode(trainer_kwargs.get("train_masking_mode", MaskingMode.RANDOM))
        self.val_masking_mode = MaskingMode(trainer_kwargs.get("val_masking_mode", MaskingMode.VALIDATION))
        self.best_val_dice = float("-inf")
        self._train_iterator: Any | None = None

        # Two-stage schedule matching legacy ShaSpec's two train_SS.py runs: a
        # full-modality warmup (no missing-modality dropout), then a second run
        # with train_masking_mode for the rest of training. Upstream's second run
        # resumes the optimizer (momentum kept) and continues the same poly
        # schedule; restart_schedule_at_warmup_end=True instead rebuilds the
        # optimizer and restarts the schedule. Disabled by default (warmup_fraction=0.0).
        self.warmup_fraction = float(trainer_kwargs.get("warmup_fraction", 0.0))
        if not 0.0 <= self.warmup_fraction <= 1.0:
            raise ValueError("warmup_fraction must be within [0, 1]")
        self.warmup_epochs = round(self.warmup_fraction * num_epochs)
        self._in_warmup: bool | None = None
        self.restart_schedule_at_warmup_end = bool(trainer_kwargs.get("restart_schedule_at_warmup_end", False))
        self.mask_sampling = str(trainer_kwargs.get("mask_sampling", "subset_uniform"))

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

    def _apply_training_stage(self, epoch: int) -> None:
        if self.warmup_epochs == 0:
            return

        in_warmup = epoch < self.warmup_epochs
        if in_warmup == self._in_warmup:
            return

        entering_main = self._in_warmup is True and not in_warmup
        self._in_warmup = in_warmup

        if self.train_set is not None:
            self.train_set.masking_mode = MaskingMode.FULL if in_warmup else self.train_masking_mode

        if entering_main and self.scheduler_config is not None and self.restart_schedule_at_warmup_end:
            remaining_epochs = self.num_epochs - self.warmup_epochs
            self._build_optimizer()
            previous_total_iters = self.scheduler_config.kwargs.get("total_iters")
            self.scheduler_config.kwargs["total_iters"] = remaining_epochs
            self._build_scheduler(self.optimizer)
            if previous_total_iters is not None:
                self.scheduler_config.kwargs["total_iters"] = previous_total_iters

    def train_epoch(self, epoch: int) -> dict[str, float]:
        if self.train_loader is None:
            raise RuntimeError("train_loader must be initialized before training")

        self._apply_training_stage(epoch)

        self.model.train()
        self._set_aux_training_flag(True)
        steps = self.iter_per_epoch if self.iter_per_epoch is not None else len(self.train_loader)
        totals = {"loss": 0.0, "cross": 0.0, "dice": 0.0, "shared_similarity": 0.0, "domain_cls": 0.0}

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
                    metrics=f"loss {metrics['loss']:.4f}  avg {totals['loss'] / iterations:.4f}",
                )
                if self.debug:
                    break

        return {key: value / max(iterations, 1) for key, value in totals.items()}

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
                        metrics=f"loss {float(seg_loss.item()):.4f}  avg {loss_sum / max(sample_count, 1):.4f}",
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
            mask_sampling=self.mask_sampling,
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
            "shared_similarity": float(metrics["shared_similarity"].item()),
            "domain_cls": float(metrics["domain_cls"].item()),
        }
