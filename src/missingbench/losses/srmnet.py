from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import torch

from missingbench.losses.config import KwargField, positive_int, unit_interval_float
from missingbench.losses.imfuse import dice_loss, softmax_weighted_loss

RECON_WEIGHT = 0.1  # legacy: loss = pred_loss + preds_loss + 0.1 * rec_loss


class SRMNetLoss:
    """Loss wrapper matching legacy SRMNet's training objective: segmentation
    cross-entropy + Dice on the main prediction, the same on each of the 4
    deep-supervision predictions, plus an L1 image-reconstruction term
    (summed, not averaged, over the 4 modalities) weighted by ``0.1``.
    """

    KWARG_SPEC: ClassVar[dict[str, KwargField]] = {
        "num_classes": positive_int(),
        "log_clamp_min": unit_interval_float(),
    }

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        num_cls: int | None = None,
        log_clamp_min: float = 0.005,
    ) -> None:
        resolved_num_classes = (
            int(num_classes) if num_classes is not None else int(num_cls) if num_cls is not None else 4
        )
        self.num_classes = resolved_num_classes
        self.log_clamp_min = float(log_clamp_min)

    def segmentation_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.softmax_weighted_loss(output, target) + self.dice_loss(output, target)

    def dice_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_loss(output, target, num_cls=self.num_classes, eps=1e-7)

    def softmax_weighted_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return softmax_weighted_loss(output, target, num_cls=self.num_classes, log_clamp_min=self.log_clamp_min)

    def training_loss(
        self,
        outputs: tuple[torch.Tensor, Sequence[torch.Tensor], Sequence[torch.Tensor], torch.Tensor],
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pred, preds, recs, images = outputs

        pred_cross = self.softmax_weighted_loss(pred, target)
        pred_dice = self.dice_loss(pred, target)

        preds_cross = pred.new_tensor(0.0)
        preds_dice = pred.new_tensor(0.0)
        for scale_pred in preds:
            preds_cross = preds_cross + self.softmax_weighted_loss(scale_pred, target)
            preds_dice = preds_dice + self.dice_loss(scale_pred, target)

        recon = pred.new_tensor(0.0)
        for modal_index, rec in enumerate(recs):
            recon = recon + torch.mean(torch.abs(images[:, modal_index : modal_index + 1] - rec))

        loss = (pred_cross + pred_dice) + (preds_cross + preds_dice) + (RECON_WEIGHT * recon)

        return {
            "loss": loss,
            "pred_cross": pred_cross,
            "pred_dice": pred_dice,
            "preds_cross": preds_cross,
            "preds_dice": preds_dice,
            "recon": recon,
        }


__all__ = ["SRMNetLoss"]
