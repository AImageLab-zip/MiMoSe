from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from mimose.datasets import MaskingMode
from mimose.datasets.legacy_crop import LocatedCropDataset
from mimose.datasets.base import sample_count_uniform_mask
from mimose.datasets.imfuse import MASK_PATTERNS
from mimose.enums import Precision, TransformKind
from mimose.losses.config import LossConfig
from mimose.losses.mcpl import class_contrastive_loss, mcpl_reconstruction_loss
from mimose.models.config import ModelConfig
from mimose.training.config import OptimizerConfig, SchedulerConfig
from mimose.training.legacy_augment import legacy_intensity_flip
from mimose.training.masking import mae_patch_gate
from mimose.training.trainers.base_trainer import BaseTrainer
from mimose.training.transforms.base_transforms import TransformManager

DEFAULT_PATCH_SIZE = 128


class MCPLTrainer(BaseTrainer):
    """Mirrors ``M3AETrainer``'s two-stage pretrain/finetune scaffolding
    (legacy MCPL shares that entire harness with legacy M3AE -- see
    ``mimose.models.mcpl.MCPL``'s docstring), adding the class/modality
    prototype contrastive losses that are MCPL's own contribution.
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
        self.dataname = str(trainer_kwargs.get("dataname", "BRATS2023"))
        self.iter_per_epoch = (
            int(trainer_kwargs["iter_per_epoch"]) if trainer_kwargs.get("iter_per_epoch") is not None else None
        )
        self.patch_size = int(trainer_kwargs.get("patch_size", DEFAULT_PATCH_SIZE))
        self.debug = bool(trainer_kwargs.get("debug", False))
        self.transform_kind = TransformKind(trainer_kwargs.get("transform_kind", TransformKind.IMFUSE))
        self.train_masking_mode = MaskingMode(trainer_kwargs.get("train_masking_mode", MaskingMode.RANDOM))
        self.val_masking_mode = MaskingMode(trainer_kwargs.get("val_masking_mode", MaskingMode.VALIDATION))
        self.mask_sampling = str(trainer_kwargs.get("mask_sampling", "subset_uniform"))
        self.best_val_dice = float("-inf")
        self._train_iterator: Any | None = None

        # Legacy MCPL train.py: each loader sample (batch 1) is duplicated into ``views`` (upstream
        # --batch-size 2) independently modality-masked copies, filled, augmented, and trained with
        # EDiceLoss over the duplicated batch. Upstream runs without deep supervision
        # (--deep_supervised defaults to False), so its ``style[feature_level + 1]`` consistency
        # term is ``segs_S1[0]`` -- the first view's output -- and the MSE compares that view's ET
        # and TC cosine maps; with deep supervision it would compare the two views' out4 logits.
        self.views = int(trainer_kwargs.get("views", 2))
        self.deep_supervised = bool(trainer_kwargs.get("deep_supervised", False))
        self.consistency_weight = float(trainer_kwargs.get("consistency_weight", 0.1))
        self.class_contrast_weight = float(trainer_kwargs.get("class_contrast_weight", 0.01))
        self.modal_contrast_weight = float(trainer_kwargs.get("modal_contrast_weight", 0.01))
        self.cc_modalities = int(trainer_kwargs.get("cc_modalities", 4))
        # Upstream fills missing modalities first, then augments (intensity + flips, no rotation).
        self.legacy_augment = bool(trainer_kwargs.get("legacy_augment", True))
        # Two-stage schedule matching legacy MCPL's separate pretrain.py (masked reconstruction
        # guided by the modality prototypes) and train.py (segmentation, fresh Adam).
        # Disabled by default (pretrain_fraction=0.0).
        self.pretrain_fraction = float(trainer_kwargs.get("pretrain_fraction", 0.0))
        if not 0.0 <= self.pretrain_fraction <= 1.0:
            raise ValueError("pretrain_fraction must be within [0, 1]")
        self.pretrain_epochs = round(self.pretrain_fraction * num_epochs)
        self.limage_lr = float(trainer_kwargs.get("limage_lr", 0.005))
        # Weight decay during pretraining only (legacy pretrain.py runs at wd=0, train.py at --wd);
        # None keeps the optimizer's weight_decay for both stages.
        pretrain_weight_decay = trainer_kwargs.get("pretrain_weight_decay")
        self.pretrain_weight_decay = None if pretrain_weight_decay is None else float(pretrain_weight_decay)
        # Upstream pretrain.py loads --batch_size 2 independent crops per step, while train.py loads 1.
        self.pretrain_samples_per_step = int(trainer_kwargs.get("pretrain_samples_per_step", 1))
        # Upstream pretrain.py masks 87.5% of 16^3 tokens plus up to 3 whole modalities (mdp=3).
        self.mae_patch_masking = bool(trainer_kwargs.get("mae_patch_masking", True))
        self.mae_mask_ratio = float(trainer_kwargs.get("mae_mask_ratio", 0.875))
        self.mae_mdp = int(trainer_kwargs.get("mae_mdp", 3))
        # Evaluate the best-validation checkpoint, as upstream test.py does with --checkpoint.
        self.final_from_best = bool(trainer_kwargs.get("final_from_best", False))
        self._in_pretrain: bool | None = None
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

    def _build_optimizer(self, model: torch.nn.Module | None = None) -> torch.optim.Optimizer:
        if self.pretrain_epochs == 0:
            return super()._build_optimizer(model)

        if self.optimizer_config is None:
            raise RuntimeError("optimizer_config must be set before building an optimizer")

        target_model = model or self._model_for_state()
        if target_model is None:
            raise RuntimeError("model must be initialized before building an optimizer")

        limage_params = [target_model.limage]
        other_params = [param for name, param in target_model.named_parameters() if name != "limage"]
        self.optimizer = self.optimizer_config.optim_class(
            [
                {"params": other_params},
                {"params": limage_params, "lr": self.limage_lr},
            ],
            lr=self.optimizer_config.lr,
            weight_decay=self.optimizer_config.weight_decay,
        )
        return self.optimizer

    def _apply_training_stage(self, epoch: int) -> None:
        """Resume-safe pretrain/finetune stage switch (stage is a pure function of ``epoch``).
        Entering fine-tuning reproduces upstream's fresh ``train.py`` Adam: moments cleared, the
        filler left out (lr 0), the fine-tuning weight decay, and a new cosine schedule over the
        remaining epochs starting from the base lr."""
        if self.pretrain_epochs == 0:
            return
        in_pretrain = epoch < self.pretrain_epochs
        if in_pretrain == self._in_pretrain:
            return
        entering_finetune = self._in_pretrain is True and not in_pretrain
        self._in_pretrain = in_pretrain
        model = self._model_for_state()
        if model is not None and getattr(model, "learnable_filler", True):
            model.limage.requires_grad_(in_pretrain)
        if self.pretrain_weight_decay is not None and self.optimizer is not None:
            weight_decay = self.pretrain_weight_decay if in_pretrain else self.optimizer_config.weight_decay
            for group in self.optimizer.param_groups:
                group["weight_decay"] = weight_decay
        if entering_finetune and self.optimizer is not None:
            self.optimizer.state.clear()
            self._reset_param_group_lrs([self.optimizer_config.lr, 0.0])
            if self.scheduler_config is not None:
                remaining_epochs = self.num_epochs - self.pretrain_epochs
                previous_t_max = self.scheduler_config.kwargs.get("T_max")
                self.scheduler_config.kwargs["T_max"] = remaining_epochs
                self._build_scheduler(self.optimizer)
                if previous_t_max is not None:
                    self.scheduler_config.kwargs["T_max"] = previous_t_max

    def train_epoch(self, epoch: int) -> dict[str, float]:
        if self.train_loader is None:
            raise RuntimeError("train_loader must be initialized before training")

        self._apply_training_stage(epoch)

        self.model.train()
        in_pretrain = bool(self._in_pretrain)
        samples_per_step = self.pretrain_samples_per_step if in_pretrain else 1
        steps = (
            self.iter_per_epoch
            if self.iter_per_epoch is not None
            else max(len(self.train_loader) // samples_per_step, 1)
        )
        totals = (
            {"loss": 0.0, "recon": 0.0, "modal_contrast": 0.0}
            if in_pretrain
            else {"loss": 0.0, "cross": 0.0, "dice": 0.0, "class_contrast": 0.0, "consistency": 0.0}
        )
        stage_label = "Pretrain" if in_pretrain else "Train"

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
                metrics = self._pretrain_step(batch) if in_pretrain else self._train_step(batch)
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

        if self._in_pretrain:
            return self._pretrain_val_epoch(epoch)

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

        # Upstream: a uniform random patch inside the brain bounding box; augmentation happens in
        # the trainer, after the missing modalities are filled.
        train_set = LocatedCropDataset(
            root=self.input_dir,
            masking_mode=self.train_masking_mode,
            split=self.train_split,
            mask_sampling=self.mask_sampling,
            crop_size=self.patch_size,
        )
        val_set = LocatedCropDataset(
            root=self.input_dir,
            masking_mode=self.val_masking_mode,
            split=self.val_split,
            crop_size=None,
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

    def _cluster_centers_weight(self) -> torch.Tensor:
        return self._model_for_state()._cluster_centers.weight

    def _center_crop_to_patch(self, images: torch.Tensor) -> torch.Tensor:
        """The pretraining reconstruction head (unlike ``predict``) requires
        inputs at exactly ``patch_size`` in every spatial dimension, but the
        val split's "test" transform leaves images at their native
        (uncropped) size. Deterministically center-crop to match, since
        pretrain validation only needs a representative patch, not full
        spatial coverage."""
        patch = self.patch_size
        spatial_shape = images.shape[-3:]
        if tuple(spatial_shape) == (patch, patch, patch):
            return images
        if any(dim < patch for dim in spatial_shape):
            raise RuntimeError(
                f"MCPL pretrain validation requires spatial dimensions of at least "
                f"{(patch, patch, patch)}, got {tuple(spatial_shape)}"
            )
        starts = [(dim - patch) // 2 for dim in spatial_shape]
        h0, w0, d0 = starts
        return images[:, :, h0 : h0 + patch, w0 : w0 + patch, d0 : d0 + patch]

    def _pretrain_val_epoch(self, epoch: int) -> dict[str, float]:
        assert self.val_loader is not None

        self.model.eval()
        loss_sum = 0.0
        sample_count = 0

        with torch.no_grad():
            with self._progress(disable=not self.is_main_process) as progress:
                task_id = progress.add_task(
                    f"Pretrain val {epoch + 1}/{self.num_epochs}",
                    total=len(self.val_loader),
                    metrics="",
                    vram=self._vram_text(),
                )
                for batch in self.val_loader:
                    images = batch["images"].to(self.device, non_blocking=True)
                    mask = batch["mask"].to(self.device, non_blocking=True).bool()
                    images = self._center_crop_to_patch(images)

                    with self._autocast_context():
                        recon, cluster_centers = self.model(images, mask, mode="reconstruct")
                        metrics = mcpl_reconstruction_loss(
                            recon,
                            images,
                            self._cluster_centers_weight(),
                            cc_modalities=self.cc_modalities,
                            cc_classes=self._cc_classes(),
                            modal_contrast_weight=self.modal_contrast_weight,
                        )

                    batch_size = images.shape[0]
                    sample_count += batch_size
                    loss_sum += float(metrics["loss"].item()) * batch_size
                    progress.update(
                        task_id,
                        advance=1,
                        vram=self._vram_text(),
                        metrics=f"loss {float(metrics['loss'].item()):.4f}  avg {loss_sum / max(sample_count, 1):.4f}",
                    )
                    if self.debug:
                        break

        self.model.train()
        return {"loss": loss_sum / max(sample_count, 1)}

    def _pretrain_step(self, batch: dict[str, Any]) -> dict[str, float]:
        batches = [batch] + [self._next_train_batch() for _ in range(self.pretrain_samples_per_step - 1)]
        images = torch.cat([item["images"] for item in batches]).to(self.device, non_blocking=True)
        mask = torch.cat([item["mask"] for item in batches]).to(self.device, non_blocking=True).bool()
        if self.mae_patch_masking:
            mask = mae_patch_gate(
                images.shape[0],
                num_modalities=images.shape[1],
                patch_shape=self.patch_size,
                mask_ratio=self.mae_mask_ratio,
                mdp=self.mae_mdp,
                device=self.device,
            )
        with self._autocast_context():
            recon, cluster_centers = self.model(images, mask, mode="reconstruct")
            metrics = mcpl_reconstruction_loss(
                recon,
                images,
                self._cluster_centers_weight(),
                cc_modalities=self.cc_modalities,
                cc_classes=self._cc_classes(),
                modal_contrast_weight=self.modal_contrast_weight,
            )
            loss = metrics["loss"]

        self._backward_step(loss)
        return {
            "loss": float(loss.item()),
            "recon": float(metrics["recon"].item()),
            "modal_contrast": float(metrics["modal_contrast"].item()),
        }

    def _random_view_mask(self, reference_mask: torch.Tensor) -> torch.Tensor:
        """Sample a fresh per-sample missing-modality mask for the second
        view, drawn from the same 15-pattern pool the RANDOM dataset masking
        uses (matches legacy MCPL train.py's duplicated-batch dual-view
        trick, adapted to MiMoSe's per-sample mask instead of a fixed
        per-batch modality-dropout count)."""
        if self.mask_sampling == "count_uniform":
            masks = torch.stack(
                [sample_count_uniform_mask(reference_mask.shape[1]) for _ in range(reference_mask.shape[0])]
            )
            return masks.to(device=reference_mask.device, dtype=reference_mask.dtype)
        indices = torch.randint(len(MASK_PATTERNS), size=(reference_mask.shape[0],))
        return MASK_PATTERNS[indices].to(device=reference_mask.device, dtype=reference_mask.dtype)

    def _cc_classes(self) -> int:
        return int(self._model_for_state().cc_classes)

    def save_final_checkpoint(self) -> Path:
        best_path = self.checkpoint_dir / "best.pth"
        if self.final_from_best and best_path.is_file() and self.is_main_process:
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            self._model_for_state().load_state_dict(checkpoint["state_dict"])
        return super().save_final_checkpoint()

    def _build_views(
        self, images: torch.Tensor, seg: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Upstream ``inputs_S1.repeat(batch_size)``: ``views`` copies of each sample, each with its
        own modality mask filled from ``limage`` (frozen zeros upstream), then augmented together.
        Returns view-major ``[views * B, ...]`` images and labels."""
        limage = self._model_for_state().limage.detach().to(images.dtype)
        view_images = []
        for view in range(self.views):
            view_mask = mask if view == 0 else self._random_view_mask(mask)
            gate = view_mask.view(-1, view_mask.shape[1], 1, 1, 1).to(images.dtype)
            view_images.append(images * gate + limage * (1 - gate))
        stacked = torch.stack(view_images, dim=1)  # [B, V, C, ...]
        labels = seg.unsqueeze(1).expand(-1, self.views, *seg.shape[1:])
        if self.legacy_augment:
            augmented = [legacy_intensity_flip(stacked[b], labels[b]) for b in range(stacked.shape[0])]
            stacked = torch.stack([item[0] for item in augmented])
            labels = torch.stack([item[1] for item in augmented])
        return stacked.transpose(0, 1).flatten(0, 1), labels.transpose(0, 1).flatten(0, 1)

    def _train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        images = batch["images"].to(self.device, non_blocking=True)
        seg = batch["seg"].to(self.device, non_blocking=True).float()
        mask = batch["mask"].to(self.device, non_blocking=True).bool()
        batch_size = images.shape[0]
        loss_impl = self._loss_impl()

        view_images, view_seg = self._build_views(images, seg, mask)
        filled = torch.ones(view_images.shape[0], view_images.shape[1], dtype=torch.bool, device=self.device)
        with self._autocast_context():
            target = self._seg_to_one_hot(view_seg.round().long())
            seg_outputs, _ = self.model(view_images, filled, mode="segment_train")
            if self.deep_supervised:
                metrics = loss_impl.deep_supervised_training_loss(seg_outputs, target)
            else:
                metrics = loss_impl.training_loss(seg_outputs[-1], target)
            loss, cross, dice = metrics["loss"], metrics["cross"], metrics["dice"]

            consistency = images.new_tensor(0.0)
            if self.consistency_weight > 0 and self.views > 1:
                if self.deep_supervised:
                    out4 = seg_outputs[0].float().unflatten(0, (self.views, batch_size))
                    consistency = F.mse_loss(out4[0], out4[1])
                else:
                    first_view = seg_outputs[-1].float().unflatten(0, (self.views, batch_size))[0]
                    consistency = F.mse_loss(first_view[:, 0], first_view[:, 1])
                loss = loss + self.consistency_weight * consistency

            class_contrast = class_contrastive_loss(
                self._cluster_centers_weight(), cc_modalities=self.cc_modalities, cc_classes=self._cc_classes()
            )
            loss = loss + (self.class_contrast_weight * class_contrast)

        self._backward_step(loss)
        return {
            "loss": float(loss.item()),
            "cross": float(cross.item()),
            "dice": float(dice.item()),
            "class_contrast": float(class_contrast.item()),
            "consistency": float(consistency.item()),
        }
