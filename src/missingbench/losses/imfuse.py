from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import torch

from missingbench.losses.config import (
    KwargField,
    nonneg_float,
    positive_float,
    positive_int,
    unit_interval_float,
)


def dice_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    *,
    num_cls: int,
    eps: float = 1e-7,
    per_sample: bool = False,
) -> torch.Tensor:
    """Legacy IMFuse soft Dice loss over one-hot targets.

    The default pools the sums over the whole batch; ``per_sample=True`` computes Dice per
    sample and averages, which matches originals trained at batch size 1.
    """
    if output.ndim != 5 or target.ndim != 5:
        raise ValueError("dice_loss expects 5D tensors shaped [B, C, H, W, D]")

    target = target.float()
    if per_sample:
        dims = (2, 3, 4)
        numerator = torch.sum(output[:, :num_cls] * target[:, :num_cls], dim=dims)
        denominator = torch.sum(output[:, :num_cls], dim=dims) + torch.sum(target[:, :num_cls], dim=dims)
        per_class = 2.0 * numerator / (denominator + eps)
        return 1.0 - per_class.mean(dim=1).mean()

    dice = output.new_tensor(0.0)
    for class_index in range(num_cls):
        numerator = torch.sum(output[:, class_index] * target[:, class_index])
        left = torch.sum(output[:, class_index])
        right = torch.sum(target[:, class_index])
        dice = dice + (2.0 * numerator / (left + right + eps))
    return 1.0 - (dice / num_cls)


def softmax_weighted_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    *,
    num_cls: int,
    log_clamp_min: float = 0.005,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Legacy IMFuse class-balanced cross-entropy on softmax probabilities."""
    if output.ndim != 5 or target.ndim != 5:
        raise ValueError(
            "softmax_weighted_loss expects 5D tensors shaped [B, C, H, W, D]"
        )

    target = target.float()
    _, _, height, width, depth = output.size()
    total_target = torch.sum(target, dim=(1, 2, 3, 4)).clamp_min(eps)
    cross_loss = output.new_tensor(0.0)

    for class_index in range(num_cls):
        output_i = output[:, class_index]
        target_i = target[:, class_index]
        weights = 1.0 - (
            torch.sum(target_i, dim=(1, 2, 3)) / total_target
        )
        weights = weights.view(-1, 1, 1, 1).repeat(1, height, width, depth)
        cross_loss = cross_loss + (
            -weights
            * target_i
            * torch.log(torch.clamp(output_i, min=log_clamp_min, max=1.0)).float()
        )

    return torch.mean(cross_loss)


class IMFuseLoss:
    """Loss wrapper matching the branch-wise training logic from legacy IMFuse."""

    KWARG_SPEC: ClassVar[dict[str, KwargField]] = {
        "num_classes": positive_int(),
        "fuse_weight": nonneg_float(),
        "sep_weight": nonneg_float(),
        "prm_weight": nonneg_float(),
        "eps": positive_float(),
        "log_clamp_min": unit_interval_float(),
    }

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        num_cls: int | None = None,
        fuse_weight: float = 1.0,
        sep_weight: float = 1.0,
        prm_weight: float = 1.0,
        eps: float = 1e-7,
        log_clamp_min: float = 0.005,
    ) -> None:
        resolved_num_classes = (
            int(num_classes)
            if num_classes is not None
            else int(num_cls) if num_cls is not None else 4
        )
        self.num_classes = resolved_num_classes
        self.fuse_weight = float(fuse_weight)
        self.sep_weight = float(sep_weight)
        self.prm_weight = float(prm_weight)
        self.eps = float(eps)
        self.log_clamp_min = float(log_clamp_min)

    def segmentation_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return self.softmax_weighted_loss(output, target) + self.dice_loss(output, target)

    def training_loss(
        self,
        outputs: tuple[torch.Tensor, Sequence[torch.Tensor], Sequence[torch.Tensor]],
        target: torch.Tensor,
        include_fuse: bool,
        include_sep: bool = True,
    ) -> dict[str, torch.Tensor]:
        fuse_pred, sep_preds, prm_preds = outputs

        zero = target.new_tensor(0.0)

        # Only build the computation graph for branches that will actually be
        # backpropagated. Skipped branches return zeros so the caller always
        # gets the same dict shape, but their activation buffers are never
        # allocated — preventing the OOM that occurs when fuse/sep graphs are
        # built but then excluded from loss.backward().
        fuse = self._branch_loss(fuse_pred, target) if include_fuse else {"cross": zero, "dice": zero, "total": zero}
        sep = self._multi_branch_loss(sep_preds, target) if include_sep else {"cross": zero, "dice": zero, "total": zero}
        prm = self._multi_branch_loss(prm_preds, target)

        total = self.prm_weight * prm["total"]
        if include_sep:
            total = total + (self.sep_weight * sep["total"])
        if include_fuse:
            total = total + (self.fuse_weight * fuse["total"])

        return {
            "loss": total,
            "fusecross": fuse["cross"],
            "fusedice": fuse["dice"],
            "sepcross": sep["cross"],
            "sepdice": sep["dice"],
            "prmcross": prm["cross"],
            "prmdice": prm["dice"],
        }

    def dice_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return dice_loss(
            output,
            target,
            num_cls=self.num_classes,
            eps=self.eps,
        )

    def softmax_weighted_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return softmax_weighted_loss(
            output,
            target,
            num_cls=self.num_classes,
            log_clamp_min=self.log_clamp_min,
            eps=self.eps,
        )

    def _branch_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cross = self.softmax_weighted_loss(prediction, target)
        dice = self.dice_loss(prediction, target)
        return {
            "cross": cross,
            "dice": dice,
            "total": cross + dice,
        }

    def _multi_branch_loss(
        self,
        predictions: Sequence[torch.Tensor],
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not predictions:
            zero = target.new_tensor(0.0)
            return {"cross": zero, "dice": zero, "total": zero}

        cross = target.new_tensor(0.0)
        dice = target.new_tensor(0.0)
        for prediction in predictions:
            branch_loss = self._branch_loss(prediction, target)
            cross = cross + branch_loss["cross"]
            dice = dice + branch_loss["dice"]
        return {
            "cross": cross,
            "dice": dice,
            "total": cross + dice,
        }


__all__ = [
    "IMFuseLoss",
    "dice_loss",
    "softmax_weighted_loss",
]
