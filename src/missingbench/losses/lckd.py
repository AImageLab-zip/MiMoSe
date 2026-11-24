from __future__ import annotations

from typing import ClassVar

import torch

from missingbench.losses.config import KwargField, nonneg_float, positive_float, positive_int, unit_interval_float
from missingbench.losses.imfuse import dice_loss, softmax_weighted_loss


class LCKDLoss:
    """Loss wrapper matching legacy LCKD (``DualNet``)'s segmentation +
    feature-distillation training objective.

    Legacy trains with ``dice + bce + kd_wt * totKDLoss`` where the dice/bce
    terms are computed over a 3-channel BraTS-region (ET/WT/TC) sigmoid
    output. This port's model instead produces a standard softmax
    ``num_cls``-channel prediction, so the segmentation term reuses
    ``softmax_weighted_loss``/``dice_loss`` from ``losses/imfuse.py`` (same
    convention as every other model in this framework) in place of legacy's
    region-BCE + region-Dice. The feature-distillation term (``kd_loss``,
    computed inside ``LCKD.forward`` since it operates on internal encoder
    features) is combined at legacy's own weight, ``kd_wt = 0.1``.
    """

    KWARG_SPEC: ClassVar[dict[str, KwargField]] = {
        "num_classes": positive_int(),
        "kd_weight": nonneg_float(),
        "eps": positive_float(),
        "log_clamp_min": unit_interval_float(),
    }

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        num_cls: int | None = None,
        kd_weight: float = 0.1,
        eps: float = 1e-7,
        log_clamp_min: float = 0.005,
    ) -> None:
        resolved_num_classes = (
            int(num_classes)
            if num_classes is not None
            else int(num_cls) if num_cls is not None else 4
        )
        self.num_classes = resolved_num_classes
        self.kd_weight = float(kd_weight)
        self.eps = float(eps)
        self.log_clamp_min = float(log_clamp_min)

    def segmentation_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.softmax_weighted_loss(output, target) + self.dice_loss(output, target)

    def dice_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_loss(output, target, num_cls=self.num_classes, eps=self.eps)

    def softmax_weighted_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return softmax_weighted_loss(
            output, target, num_cls=self.num_classes, log_clamp_min=self.log_clamp_min, eps=self.eps
        )

    def training_loss(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        prediction, kd_loss = outputs

        cross = self.softmax_weighted_loss(prediction, target)
        dice = self.dice_loss(prediction, target)
        seg = cross + dice
        total = seg + self.kd_weight * kd_loss

        return {
            "loss": total,
            "cross": cross,
            "dice": dice,
            "kd": kd_loss,
        }


__all__ = ["LCKDLoss"]
