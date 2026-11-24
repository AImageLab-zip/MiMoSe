from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import torch
import torch.nn.functional as F

from missingbench.losses.config import (
    KwargField,
    nonneg_float,
    positive_float,
    positive_int,
    unit_interval_float,
)
from missingbench.losses.imfuse import dice_loss, softmax_weighted_loss


class RFLLoss:
    """Loss wrapper matching legacy RFL's fuse + sep + prm + RFIM consistency terms."""

    KWARG_SPEC: ClassVar[dict[str, KwargField]] = {
        "num_classes": positive_int(),
        "fuse_weight": nonneg_float(),
        "sep_weight": nonneg_float(),
        "prm_weight": nonneg_float(),
        "forward_weight": nonneg_float(),
        "rfl_weight": nonneg_float(),
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
        forward_weight: float = 1.0,
        rfl_weight: float = 1.0,
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
        self.forward_weight = float(forward_weight)
        self.rfl_weight = float(rfl_weight)
        self.eps = float(eps)
        self.log_clamp_min = float(log_clamp_min)

    def segmentation_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.softmax_weighted_loss(output, target) + self.dice_loss(output, target)

    def training_loss(
        self,
        outputs: tuple[
            torch.Tensor,
            Sequence[torch.Tensor],
            Sequence[torch.Tensor],
            Sequence[torch.Tensor],
            Sequence[torch.Tensor],
            Sequence[torch.Tensor],
            Sequence[torch.Tensor],
        ],
        target: torch.Tensor,
        include_fuse: bool,
    ) -> dict[str, torch.Tensor]:
        (
            fuse_pred,
            sep_preds,
            prm_preds,
            forward_outputs,
            inverse_outputs,
            intra_tokens,
            global_tokens_used,
        ) = outputs

        zero = target.new_tensor(0.0)

        fuse = self._branch_loss(fuse_pred, target) if include_fuse else {"cross": zero, "dice": zero, "total": zero}
        sep = self._multi_branch_loss(sep_preds, target)
        prm = self._multi_branch_loss(prm_preds, target)

        # "forward" imputation loss: each modality's imputed tokens should
        # match that modality's own real tokens (only meaningful/nonzero
        # for modalities the model didn't already have, but computed for
        # all 4 in training mode, matching legacy).
        forward_mse = zero
        for real_tokens, forward_out in zip(intra_tokens, forward_outputs):
            forward_mse = forward_mse + F.mse_loss(forward_out, real_tokens, reduction="mean")

        # "rfl" self-consistency loss: inverting the forward mapping
        # should approximately recover the per-sample global modality's
        # tokens that were fed into it.
        rfl_mse = zero
        for global_tokens, inverse_out in zip(global_tokens_used, inverse_outputs):
            rfl_mse = rfl_mse + F.mse_loss(inverse_out, global_tokens, reduction="mean")

        total = (
            (self.sep_weight * sep["total"])
            + (self.prm_weight * prm["total"])
            + (self.forward_weight * forward_mse)
            + (self.rfl_weight * rfl_mse)
        )
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
            "forwardmse": forward_mse,
            "rflmse": rfl_mse,
        }

    def dice_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_loss(output, target, num_cls=self.num_classes, eps=self.eps)

    def softmax_weighted_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return softmax_weighted_loss(
            output, target, num_cls=self.num_classes, log_clamp_min=self.log_clamp_min, eps=self.eps
        )

    def _branch_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        cross = self.softmax_weighted_loss(prediction, target)
        dice = self.dice_loss(prediction, target)
        return {"cross": cross, "dice": dice, "total": cross + dice}

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
        return {"cross": cross, "dice": dice, "total": cross + dice}


__all__ = ["RFLLoss"]
