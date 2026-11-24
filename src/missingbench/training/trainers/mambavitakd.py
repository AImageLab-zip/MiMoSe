from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from missingbench.enums import Precision
from missingbench.losses.config import LossConfig
from missingbench.models.config import ModelConfig
from missingbench.training.config import OptimizerConfig, SchedulerConfig
from missingbench.training.trainers.base_trainer import CONSOLE
from missingbench.training.trainers.imfuse import IMFuseTrainer
from missingbench.training.transforms.base_transforms import TransformManager


class MambaVitAKDTrainer(IMFuseTrainer):
    """Two-phase trainer matching legacy MambaVit-AKD's recipe of a wholly
    separate ``train_RFNet.py`` teacher-pretraining run followed by
    ``trainer.py``'s student/distillation run, folded into a single resumable
    trainer instead of requiring an externally-pretrained checkpoint.

    Root cause this addresses: ``MambaVitAKD``'s teacher (see its docstring)
    is a submodule trained jointly with the student rather than a frozen,
    pretrained one, but the KD loss (``kd_weight=10``) was active at full
    strength from epoch 0 regardless -- pulling the student toward a teacher
    that hadn't converged yet. Here, epochs ``[0, teacher_pretrain_epochs)``
    train *only* the teacher (direct segmentation loss, full modalities,
    student/channel_attention untouched); at ``teacher_pretrain_epochs`` the
    teacher is frozen (``requires_grad_(False)``) and the normal
    ``IMFuseTrainer``-style distillation training (student + KD/prototype/
    attention losses against the now-fixed teacher) begins, exactly matching
    legacy's frozen-teacher assumption.

    The phase split is configured as ``teacher_pretrain_fraction`` (a
    ``[0, 1]`` proportion of ``num_epochs``), the same "fraction of the total
    budget" knob ``LCKDTrainer`` exposes as ``warmup_fraction`` -- rather than
    an absolute epoch count -- so the split stays proportional when
    ``num_epochs`` is swept/overridden from the CLI instead of silently
    drifting to a different-looking schedule. It is converted once at init
    time via ``teacher_pretrain_epochs = round(teacher_pretrain_fraction *
    num_epochs)``; everything below operates on that resolved epoch count.

    ``region_fusion_start_epoch`` (``IMFuseTrainer``'s fuse-branch warmup) is
    interpreted relative to the start of the distillation phase, not epoch 0
    globally, since it exists to let the student's sep/prm branches warm up
    before fuse joins -- and the student doesn't start training at all until
    the distillation phase begins.

    Setting ``teacher_pretrain_fraction=0.0`` (the default) disables this
    phase entirely and falls back to the original single-phase joint-training
    behavior, for backward compatibility with existing configs.
    """

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
        self.teacher_pretrain_fraction = float(trainer_kwargs.get("teacher_pretrain_fraction", 0.0))
        if not 0.0 <= self.teacher_pretrain_fraction <= 1.0:
            raise ValueError(
                "teacher_pretrain_fraction must be within [0, 1], got "
                f"{self.teacher_pretrain_fraction}"
            )
        self.teacher_pretrain_epochs = round(self.teacher_pretrain_fraction * num_epochs)
        raw_teacher_lr = trainer_kwargs.get("teacher_lr")
        self.teacher_lr = float(raw_teacher_lr) if raw_teacher_lr is not None else None
        self._in_teacher_pretrain: bool | None = None

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
            batch_size=batch_size,
            num_workers=num_workers,
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

    def _build_optimizer(self, model: torch.nn.Module | None = None) -> torch.optim.Optimizer:
        if self.teacher_lr is None or self.teacher_pretrain_epochs == 0:
            return super()._build_optimizer(model)

        if self.optimizer_config is None:
            raise RuntimeError("optimizer_config must be set before building an optimizer")

        target_model = model or self._model_for_state()
        if target_model is None:
            raise RuntimeError("model must be initialized before building an optimizer")

        teacher_params = list(target_model.teacher.parameters())
        other_params = [
            param for name, param in target_model.named_parameters()
            if not name.startswith("teacher.")
        ]
        optimizer_kwargs: dict[str, Any] = {
            "lr": self.optimizer_config.lr,
            "weight_decay": self.optimizer_config.weight_decay,
        }
        if self.optimizer_config.betas is not None:
            optimizer_kwargs["betas"] = self.optimizer_config.betas
        if self.optimizer_config.amsgrad is not None:
            optimizer_kwargs["amsgrad"] = self.optimizer_config.amsgrad
        self.optimizer = self.optimizer_config.optim_class(
            [
                {"params": other_params},
                {"params": teacher_params, "lr": self.teacher_lr},
            ],
            **optimizer_kwargs,
        )
        return self.optimizer

    def _configured_lrs(self) -> list[float]:
        if self.optimizer is not None and len(self.optimizer.param_groups) == 2 and self.teacher_lr is not None:
            return [self.optimizer_config.lr, self.teacher_lr]
        return [self.optimizer_config.lr] * len(self.optimizer.param_groups)

    def _build_scheduler(self, optimizer: torch.optim.Optimizer | None = None) -> Any | None:
        if self.scheduler_config is None or self.teacher_pretrain_epochs == 0:
            return super()._build_scheduler(optimizer)

        # Only the very first build (before any epoch has run, so
        # `_in_teacher_pretrain` is still its `None` sentinel) needs scoping to
        # the teacher-pretrain phase's own duration. The phase-2 rebuild in
        # `_apply_training_stage` temporarily overwrites `scheduler_config.kwargs`
        # itself (mirroring `M3AETrainer`) before calling this method again, so
        # by then `self._in_teacher_pretrain` is already `False` and this just
        # falls through to the plain (now phase-2-scoped) default build.
        if self._in_teacher_pretrain is not None:
            return super()._build_scheduler(optimizer)

        target_optimizer = optimizer or self.optimizer
        if target_optimizer is None:
            raise RuntimeError("optimizer must be initialized before building a scheduler")

        phase_kwargs = dict(self.scheduler_config.kwargs)
        for duration_key in ("total_iters", "T_max"):
            if duration_key in phase_kwargs:
                phase_kwargs[duration_key] = self.teacher_pretrain_epochs
        self.scheduler = self.scheduler_config.scheduler_class(target_optimizer, **phase_kwargs)
        return self.scheduler

    def _apply_training_stage(self, epoch: int) -> None:
        """Resume-safe pretrain/distill stage switch: stage membership is a
        pure function of ``epoch``, so it's always correctly recomputed on
        resume (including resuming directly into the distillation phase)
        without needing to persist any stage state in the checkpoint."""
        if self.teacher_pretrain_epochs == 0:
            return

        in_teacher_pretrain = epoch < self.teacher_pretrain_epochs
        if in_teacher_pretrain == self._in_teacher_pretrain:
            return

        # True both on a normal in-process transition (was True, now False)
        # and when resuming directly into the distillation phase (was the
        # `None` sentinel, now False) -- both need the freeze + schedule
        # rebuild below.
        entering_distillation = (not in_teacher_pretrain) and (self._in_teacher_pretrain is not False)
        self._in_teacher_pretrain = in_teacher_pretrain

        if entering_distillation:
            model = self._model_for_state()
            if model is not None:
                model.teacher.requires_grad_(False)
                model.teacher.eval()
                model.teacher_frozen = True

            if self.scheduler_config is not None:
                # The phase-1 poly schedule ends at lr 0 and PolynomialLR is chainable, so a rebuilt
                # scheduler would keep the student at lr 0; restart from the configured LRs.
                self._reset_param_group_lrs(self._configured_lrs())
                remaining_epochs = self.num_epochs - self.teacher_pretrain_epochs
                previous = {}
                for duration_key in ("total_iters", "T_max"):
                    if duration_key in self.scheduler_config.kwargs:
                        previous[duration_key] = self.scheduler_config.kwargs[duration_key]
                        self.scheduler_config.kwargs[duration_key] = remaining_epochs
                self._build_scheduler(self.optimizer)
                self.scheduler_config.kwargs.update(previous)
                # Resuming mid-distillation: replay the phase-2 schedule up to the current epoch.
                for _ in range(epoch - self.teacher_pretrain_epochs):
                    self.scheduler.step()

        if self.is_main_process:
            phase_name = "student distillation" if entering_distillation else "teacher pretrain"
            CONSOLE.print(
                f"[bold cyan]MambaVit-AKD[/bold cyan] entering {phase_name} phase at epoch "
                f"{epoch + 1}/{self.num_epochs} "
                f"(teacher_pretrain_epochs={self.teacher_pretrain_epochs}, "
                f"teacher_pretrain_fraction={self.teacher_pretrain_fraction})"
            )

    def train_epoch(self, epoch: int) -> dict[str, float]:
        if self.train_loader is None:
            raise RuntimeError("train_loader must be initialized before training")

        self._apply_training_stage(epoch)
        self.model.train()
        model = self._model_for_state()
        if model is not None and getattr(model, "teacher_frozen", False):
            model.teacher.eval()
        in_teacher_pretrain = bool(self._in_teacher_pretrain)
        if not in_teacher_pretrain:
            self._set_aux_training_flag(True)

        steps = self.iter_per_epoch if self.iter_per_epoch is not None else len(self.train_loader)
        base_keys = ("loss", "fusecross", "fusedice", "sepcross", "sepdice", "prmcross", "prmdice")
        extra_keys = () if in_teacher_pretrain else ("kd", "proto", "attn", "teacher")
        totals = {key: 0.0 for key in base_keys + extra_keys}
        stage_label = "Teacher pretrain" if in_teacher_pretrain else "Distill"

        iterations = 0
        with self._progress(disable=not self.is_main_process) as progress:
            task_id = progress.add_task(
                f"{stage_label} {epoch + 1}/{self.num_epochs}",
                total=steps,
                metrics="",
                vram=self._vram_text(),
            )
            for _ in range(steps):
                batch = self._next_train_batch()
                metrics = (
                    self._teacher_pretrain_step(batch)
                    if in_teacher_pretrain
                    else self._train_step(batch, epoch)
                )
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

        return {key: value / max(iterations, 1) for key, value in totals.items()}

    def val_epoch(self, epoch: int) -> dict[str, float]:
        if self.val_loader is None:
            raise RuntimeError("val_loader must be initialized before validation")

        if self._in_teacher_pretrain:
            return self._teacher_pretrain_val_epoch(epoch)
        return super().val_epoch(epoch)

    def _teacher_pretrain_step(self, batch: dict[str, Any]) -> dict[str, float]:
        images = batch["images"].to(self.device, non_blocking=True)
        seg = batch["seg"].to(self.device, non_blocking=True).long()
        mask = batch["mask"].to(self.device, non_blocking=True).bool()
        with self._autocast_context():
            target = self._seg_to_one_hot(seg)
            # Goes through self.model(...) (not model.teacher(...) directly) so
            # DistributedDataParallel's gradient-sync hooks stay attached; see
            # MambaVitAKD.forward's docstring for the same DDP-safety concern.
            teacher_out = self.model(images, mask, mode="teacher_pretrain")
            metrics = self._loss_impl().teacher_pretrain_loss(teacher_out, target)
            loss = metrics["loss"]

        self._backward_step(loss)
        return {
            "loss": float(loss.item()),
            "fusecross": float(metrics["fusecross"].item()),
            "fusedice": float(metrics["fusedice"].item()),
            "sepcross": float(metrics["sepcross"].item()),
            "sepdice": float(metrics["sepdice"].item()),
            "prmcross": float(metrics["prmcross"].item()),
            "prmdice": float(metrics["prmdice"].item()),
        }

    def _train_step(self, batch: dict[str, Any], epoch: int) -> dict[str, float]:
        images = batch["images"].to(self.device, non_blocking=True)
        seg = batch["seg"].to(self.device, non_blocking=True).long()
        mask = batch["mask"].to(self.device, non_blocking=True).bool()
        with self._autocast_context():
            target = self._seg_to_one_hot(seg)
            outputs = self.model(images, mask)

            # region_fusion_start_epoch counts epochs since the distillation
            # phase started, not since epoch 0 globally -- see class docstring.
            distill_epoch = epoch - self.teacher_pretrain_epochs
            metrics = self._loss_impl().training_loss(
                outputs,
                target,
                include_fuse=distill_epoch >= self.region_fusion_start_epoch,
            )
            loss = metrics["loss"]

        self._backward_step(loss)
        return {
            "loss": float(loss.item()),
            "fusecross": float(metrics["fusecross"].item()),
            "fusedice": float(metrics["fusedice"].item()),
            "sepcross": float(metrics["sepcross"].item()),
            "sepdice": float(metrics["sepdice"].item()),
            "prmcross": float(metrics["prmcross"].item()),
            "prmdice": float(metrics["prmdice"].item()),
            "kd": float(metrics["kd"].item()),
            "proto": float(metrics["proto"].item()),
            "attn": float(metrics["attn"].item()),
            "teacher": float(metrics["teacher"].item()),
        }

    def _teacher_pretrain_val_epoch(self, epoch: int) -> dict[str, float]:
        assert self.val_loader is not None

        self.model.eval()
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
                    f"Teacher val {epoch + 1}/{self.num_epochs}",
                    total=len(self.val_loader),
                    metrics="",
                    vram=self._vram_text(),
                )
                for batch in self.val_loader:
                    images = batch["images"].to(self.device, non_blocking=True)
                    seg = batch["seg"].to(self.device, non_blocking=True).long()

                    model = self._model_for_state()
                    with self._autocast_context():
                        pred = model.predict_teacher(images)
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

    def _wandb_config_payload(self) -> dict[str, Any]:
        payload = super()._wandb_config_payload()
        payload.update(
            {
                "teacher_pretrain_fraction": self.teacher_pretrain_fraction,
                "teacher_pretrain_epochs": self.teacher_pretrain_epochs,
                "teacher_lr": self.teacher_lr,
            }
        )
        return payload


__all__ = ["MambaVitAKDTrainer"]
