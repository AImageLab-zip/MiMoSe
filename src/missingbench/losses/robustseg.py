from __future__ import annotations

from typing import Any, ClassVar

import torch

from missingbench.losses.config import KwargField, positive_int, unit_interval_float
from missingbench.losses.imfuse import dice_loss, softmax_weighted_loss
from missingbench.models.robustseg import MODALITIES

KLD_WEIGHT = 0.1
RECON_WEIGHT = 0.1


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Regularizes a per-modality style code towards N(0, 1)."""
    logvar = torch.clamp(logvar, min=-50, max=50)
    loss = 0.5 * torch.sum(torch.square(mu) + torch.exp(logvar) - 1 - logvar, dim=-1)
    return torch.mean(loss)


class RobustSegLoss:
    """Loss wrapper matching the legacy RobustSeg training objective.

    ``kld_weight``/``recon_weight`` are hardcoded to the legacy defaults (0.1)
    rather than exposed as kwargs, matching the original RobustSeg code.
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
        per_sample_dice: bool = False,
    ) -> None:
        resolved_num_classes = (
            int(num_classes) if num_classes is not None else int(num_cls) if num_cls is not None else 4
        )
        self.num_classes = resolved_num_classes
        self.log_clamp_min = float(log_clamp_min)
        # Originals train at batch 1; with larger batches a batch-pooled Dice lets easy samples
        # (T1c present) dominate the subsets without T1c.
        self.per_sample_dice = bool(per_sample_dice)

    def segmentation_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.softmax_weighted_loss(output, target) + self.dice_loss(output, target)

    def dice_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_loss(output, target, num_cls=self.num_classes, eps=1e-7, per_sample=self.per_sample_dice)

    def softmax_weighted_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return softmax_weighted_loss(output, target, num_cls=self.num_classes, log_clamp_min=self.log_clamp_min)

    def training_loss(self, outputs: dict[str, Any], target: torch.Tensor) -> dict[str, torch.Tensor]:
        cross = self.softmax_weighted_loss(outputs["seg_pred"], target)
        dice = self.dice_loss(outputs["seg_pred"], target)

        images = outputs["images"]
        recon = outputs["seg_pred"].new_tensor(0.0)
        kld = outputs["seg_pred"].new_tensor(0.0)
        for modality in MODALITIES:
            recon = recon + torch.mean(torch.abs(images[modality] - outputs[f"reconstruct_{modality}"]))
            mu = outputs[f"mu_{modality}"]
            sigma = outputs[f"sigma_{modality}"]
            kld = kld + kl_loss(mu, torch.log(torch.square(sigma)))

        loss = cross + dice + (RECON_WEIGHT * recon) + (KLD_WEIGHT * kld)

        return {
            "loss": loss,
            "cross": cross,
            "dice": dice,
            "recon": recon,
            "kld": kld,
        }


__all__ = ["RobustSegLoss", "kl_loss"]
