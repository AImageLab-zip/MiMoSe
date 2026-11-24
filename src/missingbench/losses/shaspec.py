from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn as nn

from missingbench.losses.config import KwargField, positive_int, unit_interval_float
from missingbench.losses.imfuse import dice_loss, softmax_weighted_loss

SHARED_SIMILARITY_WEIGHT = 0.1  # legacy "alpha"
DOMAIN_CLS_WEIGHT = 0.02  # legacy "beta"


class ShaSpecLoss:
    """Loss wrapper matching the legacy ShaSpec training objective.

    Legacy ShaSpec used a 3-channel sigmoid/BCE+Dice BraTS-region formulation;
    this port instead reuses MissingBench's standard softmax N-class segmentation
    loss (matching every other model here, and required for compatibility
    with ``BaseTrainer``'s ``_evaluate_scores``/``predict`` contract), while
    keeping the two ShaSpec-specific auxiliary terms with their original
    weights: an L1 "shared features should agree across modalities" term, and
    a cross-entropy "specific encoders should be modality-discriminative"
    term.

    ``shared_similarity_weight``/``domain_cls_weight`` are hardcoded to the
    legacy defaults (0.1/0.02) rather than exposed as kwargs, matching the
    original ShaSpec code.
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
        self.domain_cls_loss = nn.CrossEntropyLoss()

    def segmentation_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.softmax_weighted_loss(output, target) + self.dice_loss(output, target)

    def dice_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_loss(output, target, num_cls=self.num_classes, eps=1e-7)

    def softmax_weighted_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return softmax_weighted_loss(output, target, num_cls=self.num_classes, log_clamp_min=self.log_clamp_min)

    def training_loss(self, outputs: dict[str, Any], target: torch.Tensor) -> dict[str, torch.Tensor]:
        cross = self.softmax_weighted_loss(outputs["seg_pred"], target)
        dice = self.dice_loss(outputs["seg_pred"], target)

        shared = outputs["shared_content"]
        shared_similarity = (
            torch.mean(torch.abs(shared[0] - shared[1]))
            + torch.mean(torch.abs(shared[1] - shared[2]))
            + torch.mean(torch.abs(shared[2] - shared[3]))
            + torch.mean(torch.abs(shared[3] - shared[0]))
        )

        dom_logits = outputs["dom_logits"]
        batch_size = dom_logits.size(0) // 4
        dom_labels = torch.arange(4, device=dom_logits.device).repeat_interleave(batch_size)
        domain_cls = self.domain_cls_loss(dom_logits, dom_labels)

        loss = (
            cross
            + dice
            + (SHARED_SIMILARITY_WEIGHT * shared_similarity)
            + (DOMAIN_CLS_WEIGHT * domain_cls)
        )

        return {
            "loss": loss,
            "cross": cross,
            "dice": dice,
            "shared_similarity": shared_similarity,
            "domain_cls": domain_cls,
        }


__all__ = ["ShaSpecLoss"]
