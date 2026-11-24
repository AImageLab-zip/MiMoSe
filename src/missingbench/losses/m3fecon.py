from __future__ import annotations

from typing import ClassVar

import torch

from missingbench.losses.config import KwargField, positive_int, unit_interval_float
from missingbench.losses.imfuse import dice_loss, softmax_weighted_loss

RECON_WEIGHT = 1.0  # legacy adds the (unweighted) reconstruction MSE directly to the segmentation loss


class M3FeConLoss:
    """Loss wrapper matching legacy M3FeCon's training objective: a segmentation
    loss plus an MSE "feature reconstruction" term computed only over
    modalities that were missing in the input mask (reconstructed bottleneck
    feature vs. the modality's own true encoder output, detached).

    Legacy M3FeCon used nnU-Net's Dice+CE; this port reuses
    ``softmax_weighted_loss``/``dice_loss`` from ``losses/imfuse.py``, like
    every other model here.
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
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        seg_pred, bottleneck, ground_truth = outputs
        cross = self.softmax_weighted_loss(seg_pred, target)
        dice = self.dice_loss(seg_pred, target)

        missing = ~mask
        if missing.any():
            diff = (bottleneck - ground_truth) ** 2
            per_modal_mse = diff.mean(dim=tuple(range(2, diff.ndim)))  # [B, num_modals]
            recon = per_modal_mse[missing].mean()
        else:
            recon = seg_pred.new_tensor(0.0)

        loss = cross + dice + (RECON_WEIGHT * recon)

        return {"loss": loss, "cross": cross, "dice": dice, "recon": recon}


__all__ = ["M3FeConLoss"]
