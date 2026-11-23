from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F

from mimose.losses.config import KwargField, nonneg_float, positive_int

# Legacy trains channel ci on label ci with the curriculum WT -> TC -> ET, where its labels are the
# nested BraTS regions overwritten inside-out (legacy README / scripts/make_dummy_data.py: 1 = WT
# shell, 2 = TC shell, 3 = ET), i.e. ED, NCR and ET in the pipeline's labels (1 NCR, 2 ED, 3 ET).
DEFAULT_CURRICULUM_CLASSES = (2, 1, 3)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Legacy ``dice_loss`` on one sigmoid channel: per-sample Dice, averaged over the batch."""
    pred = torch.sigmoid(logits)
    dims = tuple(range(2, pred.ndim))
    inter = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims) + eps
    return (1 - 2 * inter / denom).mean()


class ALoRAMoELoss:
    """Legacy ALoRA-MoE ``DiceBCELoss(0.5)`` applied to each foreground channel as an independent
    sigmoid (channel 0 is never trained), summed over the channels the curriculum has unlocked."""

    KWARG_SPEC: ClassVar[dict[str, KwargField]] = {
        "num_classes": positive_int(),
        "bce_weight": nonneg_float(),
    }

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        num_cls: int | None = None,
        bce_weight: float = 0.5,
        curriculum_classes: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        self.num_classes = (
            int(num_classes)
            if num_classes is not None
            else int(num_cls) if num_cls is not None else 4
        )
        self.bce_weight = float(bce_weight)
        self.curriculum_classes = tuple(
            int(c) for c in (curriculum_classes if curriculum_classes is not None else DEFAULT_CURRICULUM_CLASSES)
        )
        if sorted(self.curriculum_classes) != list(range(1, self.num_classes)):
            raise ValueError(
                f"curriculum_classes must order the foreground classes 1..{self.num_classes - 1}, "
                f"got {self.curriculum_classes}"
            )

    def _dice_bce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice_loss(logits, target)

    def active_classes(self, stage: int) -> tuple[int, ...]:
        return self.curriculum_classes[: stage + 1]

    def training_loss(self, logits: torch.Tensor, labels: torch.Tensor, stage: int) -> dict[str, torch.Tensor]:
        """``logits`` [B, C, H, W, D] and class ``labels`` [B, 1, H, W, D]."""
        logits = logits.float()
        metrics: dict[str, torch.Tensor] = {}
        loss = logits.new_zeros(())
        for c in self.curriculum_classes:
            term = self._dice_bce(logits[:, c : c + 1], (labels == c).float())
            if c in self.active_classes(stage):
                loss = loss + term
            metrics[f"class{c}"] = term.detach()
        metrics["loss"] = loss
        return metrics

    def segmentation_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Validation loss on ``predict`` logits and one-hot targets, over every foreground class."""
        output = output.float()
        target = target.float()
        return sum(
            self._dice_bce(output[:, c : c + 1], target[:, c : c + 1]) for c in self.curriculum_classes
        )


__all__ = ["ALoRAMoELoss", "dice_loss"]
