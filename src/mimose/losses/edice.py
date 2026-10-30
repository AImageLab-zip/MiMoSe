from __future__ import annotations

import torch
import torch.nn.functional as F


def labels_to_regions(target: torch.Tensor) -> torch.Tensor:
    """BraTS regions ``[ET, TC, WT]`` from a 4-class one-hot target (labels 1=NCR, 2=ED, 3=ET),
    in legacy M3AE/MCPL channel order (``legacy/m3ae/dataset/brats.py``: ``np.stack([et, tc, wt])``)."""
    et = target[:, 3]
    tc = target[:, 1] + target[:, 3]
    wt = target[:, 1] + target[:, 2] + target[:, 3]
    return torch.stack((et, tc, wt), dim=1).clamp(0.0, 1.0)


def edice_loss(logits: torch.Tensor, regions: torch.Tensor) -> dict[str, torch.Tensor]:
    """Literal port of legacy M3AE/MCPL ``EDiceLoss.forward`` (``legacy/m3ae/loss/dice.py``).

    Per region channel: a batch-pooled soft Dice on ``sigmoid(logits)`` with squared denominator
    and ``smooth=1``, plus ``BCELoss(sigmoid(logits), target)``; combined as
    ``(0.7 * dice + 0.3 * bce) / num_regions``. ``cross``/``dice`` are returned already weighted
    so that ``loss = cross + dice``.
    """
    logits = logits.float()
    regions = regions.float()
    dice = logits.new_tensor(0.0)
    bce = logits.new_tensor(0.0)
    for channel in range(regions.size(1)):
        probs = torch.sigmoid(logits[:, channel])
        target = regions[:, channel]
        intersection = torch.sum(probs * target)
        dice = dice + 1.0 - (2.0 * intersection + 1.0) / (probs.pow(2).sum() + target.pow(2).sum() + 1.0)
        # F.binary_cross_entropy clamps log at -100, like legacy's torch.nn.BCELoss.
        bce = bce + F.binary_cross_entropy(probs, target)
    num_regions = regions.size(1)
    cross = 0.3 * bce / num_regions
    dice = 0.7 * dice / num_regions
    return {"loss": cross + dice, "cross": cross, "dice": dice}


__all__ = ["edice_loss", "labels_to_regions"]
