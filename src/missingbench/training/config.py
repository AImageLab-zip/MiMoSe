from dataclasses import dataclass, field
from typing import Type, Any
import ast

from missingbench.enums import OptimizerKind, SchedulerKind, TrainerKind

import typer

from torch.optim import Adam,RAdam,AdamW,SGD, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    MultiStepLR,
    PolynomialLR,
    ReduceLROnPlateau,
    StepLR,
)


class WarmupPolyLR(LRScheduler):
    """Linear LR warmup for `warmup_iters` epochs, then polynomial decay for the rest.

    Matches legacy MIFPN/M2FTrans's ``LR_Scheduler(mode='warmuppoly')``: for
    ``epoch < warmup_iters``, lr ramps linearly from 0 to base_lr; afterwards it
    poly-decays to 0 by ``total_iters`` with the given ``power``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_iters: int,
        power: float = 0.9,
        warmup_iters: int = 100,
        last_epoch: int = -1,
    ) -> None:
        self.total_iters = total_iters
        self.power = power
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        epoch = self.last_epoch
        if epoch < self.warmup_iters:
            factor = epoch / self.warmup_iters
        else:
            factor = max(0.0, 1 - (epoch - self.warmup_iters) / (self.total_iters - self.warmup_iters)) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]

OPTIMIZER_DICT = {
    OptimizerKind.RADAM: RAdam,
    OptimizerKind.ADAMW: AdamW,
    OptimizerKind.ADAM:Adam,
    OptimizerKind.SGD:SGD
}

@dataclass(frozen=True)
class OptimizerConfig:
    optim_class: Type[Optimizer]
    lr: float
    weight_decay: float
    betas: tuple[float, float] | None = None
    momentum: float | None = None
    eps: float | None = None
    amsgrad: bool | None = None
    nesterov: bool | None = None

def build_optimizer_config(
    optimizer_kind: OptimizerKind,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] | None,
    momentum: float,
    nesterov: bool = False,
    amsgrad: bool | None = None,
) -> OptimizerConfig:
    if lr <= 0:
        raise ValueError("lr must be > 0")
    if weight_decay < 0:
        raise ValueError("weight_decay must be >= 0")
    if optimizer_kind is OptimizerKind.SGD and momentum < 0:
        raise ValueError("momentum must be >= 0")

    if optimizer_kind in {OptimizerKind.RADAM, OptimizerKind.ADAMW, OptimizerKind.ADAM}:
        if betas is None or len(betas) != 2:
            raise ValueError("betas must contain exactly two values")

        beta1, beta2 = betas
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("betas must be in the range [0, 1)")

        resolved_amsgrad = (
            amsgrad if amsgrad is not None else (True if optimizer_kind is OptimizerKind.ADAM else None)
        )
        return OptimizerConfig(
            optim_class=OPTIMIZER_DICT[optimizer_kind],
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=1e-8,
            amsgrad=resolved_amsgrad,
        )
    if optimizer_kind is OptimizerKind.SGD:
        return OptimizerConfig(
            optim_class=OPTIMIZER_DICT[optimizer_kind],
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov if nesterov else None,
        )
    raise typer.BadParameter(
        f"Unsupported optimizer: {optimizer_kind}",
        param_hint="--optimizer",
    )


@dataclass(frozen=True)
class SchedulerConfig:
    scheduler_class: type[LRScheduler] | type
    kwargs: dict[str, Any] = field(default_factory=dict)


SCHEDULER_DICT = {
    SchedulerKind.POLY: PolynomialLR,
    SchedulerKind.WARMUPPOLY: WarmupPolyLR,
    SchedulerKind.COSINE: CosineAnnealingLR,
    SchedulerKind.STEP: StepLR,
    SchedulerKind.MULTISTEP: MultiStepLR,
    SchedulerKind.PLATEAU: ReduceLROnPlateau,
}


def parse_kv_list(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise typer.BadParameter(
                f"Invalid value '{item}', expected key=value format.",
            )
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()

        try:
            out[key] = ast.literal_eval(value)
        except Exception:
            out[key] = value

    return out


def build_scheduler_config(
    scheduler_kind: SchedulerKind,
    *,
    poly_total_iters: int | None,
    poly_power: float,
    warmuppoly_warmup_iters: int = 100,
    cosine_t_max: int | None,
    cosine_eta_min: float,
    step_step_size: int | None,
    step_gamma: float,
    multistep_milestones: list[int] | None,
    multistep_gamma: float,
    plateau_mode: str,
    plateau_factor: float,
    plateau_patience: int,
) -> SchedulerConfig:
    if scheduler_kind is SchedulerKind.POLY:
        if poly_total_iters is None:
            raise typer.BadParameter(
                "--poly-total-iters is required for poly",
                param_hint="--poly-total-iters",
            )
        return SchedulerConfig(
            scheduler_class=SCHEDULER_DICT[SchedulerKind.POLY],
            kwargs={
                "total_iters": poly_total_iters,
                "power": poly_power,
            },
        )

    if scheduler_kind is SchedulerKind.WARMUPPOLY:
        if poly_total_iters is None:
            raise typer.BadParameter(
                "--poly-total-iters is required for warmuppoly",
                param_hint="--poly-total-iters",
            )
        return SchedulerConfig(
            scheduler_class=SCHEDULER_DICT[SchedulerKind.WARMUPPOLY],
            kwargs={
                "total_iters": poly_total_iters,
                "power": poly_power,
                "warmup_iters": warmuppoly_warmup_iters,
            },
        )

    if scheduler_kind is SchedulerKind.COSINE:
        if cosine_t_max is None:
            raise typer.BadParameter(
                "--cosine-t-max is required for cosine",
                param_hint="--cosine-t-max",
            )
        return SchedulerConfig(
            scheduler_class=SCHEDULER_DICT[SchedulerKind.COSINE],
            kwargs={
                "T_max": cosine_t_max,
                "eta_min": cosine_eta_min,
            },
        )

    if scheduler_kind is SchedulerKind.STEP:
        if step_step_size is None:
            raise typer.BadParameter(
                "--step-step-size is required for step",
                param_hint="--step-step-size",
            )
        return SchedulerConfig(
            scheduler_class=SCHEDULER_DICT[SchedulerKind.STEP],
            kwargs={
                "step_size": step_step_size,
                "gamma": step_gamma,
            },
        )

    if scheduler_kind is SchedulerKind.MULTISTEP:
        if not multistep_milestones:
            raise typer.BadParameter(
                "--multistep-milestones is required for multistep",
                param_hint="--multistep-milestones",
            )
        return SchedulerConfig(
            scheduler_class=SCHEDULER_DICT[SchedulerKind.MULTISTEP],
            kwargs={
                "milestones": multistep_milestones,
                "gamma": multistep_gamma,
            },
        )

    if scheduler_kind is SchedulerKind.PLATEAU:
        return SchedulerConfig(
            scheduler_class=SCHEDULER_DICT[SchedulerKind.PLATEAU],
            kwargs={
                "mode": plateau_mode,
                "factor": plateau_factor,
                "patience": plateau_patience,
            },
        )

    raise typer.BadParameter(
        f"Unsupported scheduler: {scheduler_kind}",
        param_hint="--scheduler",
    )


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = True
    project: str = "SegmentationMM"
    run_name: str | None = "training"
    mode: str = "online"
    resume: str = "allow"
