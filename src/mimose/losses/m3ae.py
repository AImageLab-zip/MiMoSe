from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F

from mimose.losses.config import KwargField, positive_int
from mimose.losses.edice import edice_loss, labels_to_regions


def m3ae_reconstruction_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    limage: torch.Tensor,
    *,
    reg_weight: float = 0.005,
) -> dict[str, torch.Tensor]:
    """Legacy M3AE pretraining objective: MSE reconstruction of the original,
    unmasked input plus a smoothness regularizer on the learned placeholder
    (``limage``) that keeps it from degenerating into per-voxel noise.

    Matches ``legacy/m3ae/pretrain.py``'s ``loss_ + loss2 * .005``, where
    ``loss_`` is the MSE term and ``loss2`` is
    ``torch.norm(limage - limage.mean((2,3,4), keepdim=True), 2)``.
    """
    recon = F.mse_loss(output, target)
    smoothness = torch.norm(limage - limage.mean(dim=(2, 3, 4), keepdim=True), p=2)
    return {"loss": recon + reg_weight * smoothness, "recon": recon, "smoothness": smoothness}


def legacy_region_loss(output: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Legacy M3AE ``EDiceLoss`` (``legacy/m3ae/loss/dice.py``) on the BraTS
    regions derived from a 4-class softmax output.

    Legacy predicts 3 sigmoid region channels (ET/TC/WT) and, per channel,
    takes a squared-denominator soft Dice with smooth=1 over the whole batch
    plus a BCE, combined as ``(0.7 * dice + 0.3 * bce) / 3``. Here the region
    probabilities are sums of the softmax class probabilities (labels:
    1=NCR, 2=ED, 3=ET), so the model keeps the framework's 4-class output while
    being trained on legacy's region objective. Unlike the class-averaged
    softmax Dice, background is never a scored region.

    Returns ``cross``/``dice`` already weighted so that ``loss = cross + dice``.
    """
    output = output.float()
    target = target.float()
    regions_out = (output[:, 3], output[:, 1] + output[:, 3], output[:, 1] + output[:, 2] + output[:, 3])
    regions_tgt = (target[:, 3], target[:, 1] + target[:, 3], target[:, 1] + target[:, 2] + target[:, 3])
    dice = output.new_tensor(0.0)
    bce = output.new_tensor(0.0)
    for pred, tgt in zip(regions_out, regions_tgt):
        pred = pred.clamp(0.0, 1.0)
        intersection = torch.sum(pred * tgt)
        dice = dice + 1.0 - (2.0 * intersection + 1.0) / (pred.pow(2).sum() + tgt.pow(2).sum() + 1.0)
        # F.binary_cross_entropy clamps log at -100, like legacy's torch.nn.BCELoss.
        bce = bce + F.binary_cross_entropy(pred, tgt)
    num_regions = len(regions_out)
    cross = 0.3 * bce / num_regions
    dice = 0.7 * dice / num_regions
    return {"loss": cross + dice, "cross": cross, "dice": dice}


class M3AELoss:
    """Loss wrapper for legacy M3AE's fine-tuning objective, ``EDiceLoss`` (0.7 * region Dice +
    0.3 * BCE over the sigmoid ``[ET, TC, WT]`` logits, see ``edice_loss``), summed over the
    deep-supervision heads. The cross-view consistency MSE is added by ``M3AETrainer``.

    ``segmentation_loss`` scores the 4-class one-hot volume returned by ``predict`` (validation
    only, no gradient) with the same region objective (``legacy_region_loss``). The pretraining
    stage uses ``m3ae_reconstruction_loss`` instead.
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
        """``for l in segs_S1: loss_ += criterion(l, labels_S1)`` (``legacy/m3ae/train.py``)."""
        regions = labels_to_regions(target)
        cross = target.new_tensor(0.0)
        dice = target.new_tensor(0.0)
        for output in seg_outputs:
            metrics = edice_loss(output, regions)
            cross = cross + metrics["cross"]
            dice = dice + metrics["dice"]
        return {"loss": cross + dice, "cross": cross, "dice": dice}


__all__ = ["M3AELoss", "legacy_region_loss", "m3ae_reconstruction_loss"]
