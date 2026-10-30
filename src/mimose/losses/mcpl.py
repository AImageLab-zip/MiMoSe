from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F

from mimose.losses.config import KwargField, positive_int
from mimose.losses.edice import edice_loss, labels_to_regions
from mimose.losses.m3ae import legacy_region_loss


def _upstream_group_cos(h_i: torch.Tensor, h_j: torch.Tensor, temperature: float) -> torch.Tensor:
    """Literal port of legacy MCPL ``memory_Loss.forward_class_cos`` / ``forward_modal_cos``
    (legacy/MCPL/loss/dice.py:157-223). ``h_i``/``h_j``: ``[Q, G]`` prototype columns of two groups.

    Positives are pairs of columns *within* one group; negatives are all columns of the *other*
    group. The upstream accumulation sits after the inner ``j`` loop, so only the last ``j`` per
    ``i`` contributes (and the last ``i`` re-adds the previous pair); that quirk is kept on purpose.
    """
    num = h_i.shape[1]
    loss = h_i.new_tensor(0.0)
    count = 0
    for anchor, other in ((h_i, h_j), (h_j, h_i)):
        positive = negative = None
        for i in range(num):
            for j in range(i + 1, num):
                positive = torch.exp(torch.cosine_similarity(anchor[:, i], anchor[:, j], dim=0) / temperature)
                negative = h_i.new_tensor(0.0)
                for k in range(num):
                    negative = negative + torch.exp(
                        torch.cosine_similarity(anchor[:, i], other[:, k], dim=0) / temperature
                    )
                count += 1
            if positive is not None:
                loss = loss - torch.log(positive / negative)
    return loss / max(count, 1)


def class_contrastive_loss(
    cluster_centers_weight: torch.Tensor,
    *,
    cc_modalities: int,
    cc_classes: int,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Fine-tuning class contrast (legacy/MCPL/train.py:288-298): for each class pair (c1, c2),
    the modalities of c1 are positives of each other and the modalities of c2 are negatives."""
    prototypes = cluster_centers_weight.view(-1, cc_modalities, cc_classes)
    loss = prototypes.new_tensor(0.0)
    pairs = 0
    for c1 in range(cc_classes):
        for c2 in range(c1 + 1, cc_classes):
            loss = loss + _upstream_group_cos(prototypes[:, :, c1], prototypes[:, :, c2], temperature)
            pairs += 1
    return loss / max(pairs, 1)


def modal_contrastive_loss(
    cluster_centers_weight: torch.Tensor,
    *,
    cc_modalities: int,
    cc_classes: int,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Pretraining modality contrast (legacy/MCPL/pretrain.py:178-187): for each modality pair
    (m1, m2), the classes of m1 are positives of each other and the classes of m2 are negatives."""
    prototypes = cluster_centers_weight.view(-1, cc_modalities, cc_classes)
    loss = prototypes.new_tensor(0.0)
    pairs = 0
    for m1 in range(cc_modalities):
        for m2 in range(m1 + 1, cc_modalities):
            loss = loss + _upstream_group_cos(prototypes[:, m1, :], prototypes[:, m2, :], temperature)
            pairs += 1
    return loss / max(pairs, 1)


def mcpl_reconstruction_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    cluster_centers_weight: torch.Tensor,
    *,
    cc_modalities: int,
    cc_classes: int,
    modal_contrast_weight: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Legacy MCPL pretraining objective: MSE reconstruction of the original,
    unmasked input plus the modality-prototype contrastive loss (matches
    ``legacy/MCPL/pretrain.py``'s ``loss_ + modulContrast_loss * 0.01``)."""
    recon = F.mse_loss(output, target)
    modal_contrast = modal_contrastive_loss(cluster_centers_weight, cc_modalities=cc_modalities, cc_classes=cc_classes)
    return {
        "loss": recon + modal_contrast_weight * modal_contrast,
        "recon": recon,
        "modal_contrast": modal_contrast,
    }


class MCPLLoss:
    """Loss wrapper for legacy MCPL's fine-tuning segmentation objective, ``EDiceLoss`` (0.7 *
    region Dice + 0.3 * BCE over the sigmoid ``[ET, TC, WT]`` cosine logits, see ``edice_loss``).
    The consistency MSE and the class-prototype contrast are added by ``MCPLTrainer``; the
    pretraining stage uses ``mcpl_reconstruction_loss`` instead.

    ``segmentation_loss`` scores the 4-class one-hot volume returned by ``predict`` (validation
    only, no gradient) with the same region objective (``legacy_region_loss``).
    """

    KWARG_SPEC: ClassVar[dict[str, KwargField]] = {
        "num_classes": positive_int(),
    }

    def __init__(self, *, num_classes: int | None = None, num_cls: int | None = None) -> None:
        self.num_classes = int(num_classes) if num_classes is not None else int(num_cls) if num_cls is not None else 4

    def segmentation_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return legacy_region_loss(output, target)["loss"]

    def training_loss(self, region_logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        return edice_loss(region_logits, labels_to_regions(target))

    def deep_supervised_training_loss(
        self, seg_outputs: list[torch.Tensor], target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """``for l in segs_S1: loss_ += criterion(l, labels_S1)`` (``legacy/MCPL/train.py``)."""
        regions = labels_to_regions(target)
        cross = target.new_tensor(0.0)
        dice = target.new_tensor(0.0)
        for output in seg_outputs:
            metrics = edice_loss(output, regions)
            cross = cross + metrics["cross"]
            dice = dice + metrics["dice"]
        return {"loss": cross + dice, "cross": cross, "dice": dice}


__all__ = [
    "MCPLLoss",
    "class_contrastive_loss",
    "mcpl_reconstruction_loss",
    "modal_contrastive_loss",
]
