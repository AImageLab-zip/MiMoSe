from __future__ import annotations

from typing import ClassVar

import torch

from missingbench.losses.config import KwargField, positive_float, positive_int


def _default_class_weights(num_classes: int) -> list[float]:
    """Legacy `general_dice_loss` weights each of the 4 one-hot class
    channels by the fixed list `[0.1, 0.2, 0.3, 0.4]` (increasing importance
    from background to enhancing tumor). That list is hardcoded for exactly
    4 classes and silently breaks for BraTS25's 5 classes -- the same
    `num_cls`-vs.-class-count conflation bug found in several other ported
    models, just living in the loss instead of a module constructor here.

    This generalizes the same "monotonically increasing, triangular-number
    normalized" weighting scheme to any class count: weight_i = (i + 1) /
    sum(1..num_classes). For num_classes == 4 this reproduces legacy's
    exact weights [0.1, 0.2, 0.3, 0.4].
    """
    denom = num_classes * (num_classes + 1) / 2
    return [(i + 1) / denom for i in range(num_classes)]


def weighted_dice_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int,
    weights: list[float] | None = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Legacy `general_dice_loss` / `MulticlassDiceLoss`: a per-class soft
    Dice loss over one-hot targets, summed with fixed per-class weights (no
    cross-entropy term)."""
    if output.ndim != 5 or target.ndim != 5:
        raise ValueError("weighted_dice_loss expects 5D tensors shaped [B, C, H, W, D]")

    resolved_weights = weights if weights is not None else _default_class_weights(num_classes)
    if len(resolved_weights) != num_classes:
        raise ValueError("weights must have exactly num_classes entries")

    target = target.float()
    total = output.new_tensor(0.0)
    for class_index in range(num_classes):
        pred_i = output[:, class_index]
        target_i = target[:, class_index]
        intersection = (pred_i * target_i).sum()
        denominator = pred_i.sum() + target_i.sum()
        dice = 1.0 - (2.0 * intersection + eps) / (denominator + eps)
        total = total + resolved_weights[class_index] * dice
    return total


class InOutFusionLoss:
    """Loss wrapper matching legacy InOutFusion's `U_Hemis_loss`: a single
    weighted multi-class Dice term on the fused prediction, with no
    cross-entropy term, no auxiliary per-modality decoders, and no deep
    supervision (matching `InOutFusion.forward`, which only ever produces
    one segmentation tensor). Reuses `IMFuseTrainer`'s generic training
    loop -- the model returns the `(fuse_pred, (), ())` tuple shape that
    trainer expects, with the empty `sep`/`prm` slots simply unused here.
    """

    KWARG_SPEC: ClassVar[dict[str, KwargField]] = {
        "num_classes": positive_int(),
        "eps": positive_float(),
    }

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        num_cls: int | None = None,
        class_weights: list[float] | None = None,
        eps: float = 1e-7,
    ) -> None:
        resolved_num_classes = (
            int(num_classes)
            if num_classes is not None
            else int(num_cls) if num_cls is not None else 4
        )
        self.num_classes = resolved_num_classes
        self.class_weights = class_weights
        self.eps = float(eps)

    def segmentation_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return self.dice_loss(output, target)

    def training_loss(
        self,
        outputs: tuple[torch.Tensor, object, object],
        target: torch.Tensor,
        *,
        include_fuse: bool = True,
    ) -> dict[str, torch.Tensor]:
        fuse_pred, _, _ = outputs
        dice = self.dice_loss(fuse_pred, target)
        zero = dice.new_tensor(0.0)
        total = dice if include_fuse else zero
        return {
            "loss": total,
            "fusecross": zero,
            "fusedice": dice if include_fuse else zero,
            "sepcross": zero,
            "sepdice": zero,
            "prmcross": zero,
            "prmdice": zero,
        }

    def dice_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return weighted_dice_loss(
            output,
            target,
            num_classes=self.num_classes,
            weights=self.class_weights,
            eps=self.eps,
        )


__all__ = ["InOutFusionLoss", "weighted_dice_loss"]
