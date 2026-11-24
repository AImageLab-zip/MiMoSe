from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch.amp import GradScaler

from missingbench.enums import Precision
from missingbench.losses.config import LossConfig
from missingbench.models.config import ModelConfig
from missingbench.training.config import OptimizerConfig, SchedulerConfig
from missingbench.training.transforms.base_transforms import TransformManager


class AbstractTrainer(ABC):
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
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        self.model_config = model_config
        self.loss_config = loss_config
        self.custom_trainer_kwargs = custom_trainer_kwargs or {}
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.transform_manager = transform_manager
        self.num_epochs = int(num_epochs)
        if int(validation_every) < 1:
            raise ValueError("validation_every must be >= 1")
        self.validation_every = int(validation_every)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.precision = Precision(precision)
        self.compile = bool(compile)
        self.resume_requested = bool(resume)
        self.try_resume_requested = bool(try_resume)

        self.seed = seed
        self.pretrain = Path(pretrain) if pretrain is not None else None
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_mode = wandb_mode
        self.wandb_run_name = wandb_run_name or "training"
        self.dataset_type = dataset_type
        self.push_to_hf = bool(push_to_hf)
        self.hf_repo = hf_repo

        self.split_file: Path | None = None
        self.train_split: list[dict[str, Any]] | None = None
        self.val_split: list[dict[str, Any]] | None = None
        self.test_split: list[dict[str, Any]] | None = None

        self.distributed = False
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.device = torch.device("cpu")
        self.amp_enabled = False
        self.grad_scaler = GradScaler("cuda", enabled=False)

        self.model: torch.nn.Module | None = None
        self.loss_fn: Any | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any | None = None
        self.wandb_run: Any | None = None

        self.current_epoch = 0
        self.best_val_loss = float("inf")

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.resume: Path | None = None
        self._owns_process_group = False
        self.train_set: Any | None = None
        self.val_set: Any | None = None
        self.train_loader: Any | None = None
        self.val_loader: Any | None = None

    @abstractmethod
    def fit(self) -> None:
        pass
