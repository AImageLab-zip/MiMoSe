from __future__ import annotations

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


def softmax_kl_loss(
    input_logits: torch.Tensor, target_logits: torch.Tensor
) -> torch.Tensor:
    """Legacy ``loss.softmax_kl_loss``: elementwise KL divergence between the
    (temperature-scaled) student log-softmax and teacher softmax."""
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, reduction="none")


def prototype_loss(
    feature: torch.Tensor,
    feature_t: torch.Tensor,
    target: torch.Tensor,
    num_cls: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Legacy ``loss.prototype_loss``, adapted to a one-hot ``target`` (shape
    ``[N, num_cls, H, W, D]``) instead of legacy's integer label map (shape
    ``[N, 1, H, W, D]``) -- ``target[:, class_index]`` is exactly legacy's
    ``label == class_index`` boolean mask. Like legacy, a class is only
    included in the prototype-similarity comparison when every sample in
    the batch contains at least one voxel of that class (legacy's own
    ``(...).all()`` guard, written for a batch size of 1 where it reduces to
    "this class is present in the sample").
    """
    student_maps = []
    teacher_maps = []
    for class_index in range(num_cls):
        class_mask = target[:, class_index] > 0.5
        if not (class_mask.sum(dim=(-3, -2, -1)) > 0).all():
            continue

        weights = class_mask[:, None].float()
        proto_s = torch.sum(feature * weights, dim=(-3, -2, -1)) / (
            torch.sum(weights, dim=(-3, -2, -1)) + eps
        )
        proto_t = torch.sum(feature_t * weights, dim=(-3, -2, -1)) / (
            torch.sum(weights, dim=(-3, -2, -1)) + eps
        )

        proto_map_s = F.cosine_similarity(
            feature, proto_s[:, :, None, None, None], dim=1, eps=eps
        )
        proto_map_t = F.cosine_similarity(
            feature_t, proto_t[:, :, None, None, None], dim=1, eps=eps
        )
        student_maps.append(proto_map_s.unsqueeze(1))
        teacher_maps.append(proto_map_t.unsqueeze(1))

    if not student_maps:
        return feature.new_tensor(0.0)

    sim_map_s = torch.cat(student_maps, dim=1)
    sim_map_t = torch.cat(teacher_maps, dim=1)
    return torch.mean((sim_map_s - sim_map_t) ** 2)


class MambaVitAKDLoss:
    """Loss wrapper matching legacy MambaVit-AKD's ``trainer.py`` training loop.

    Legacy's ``apkd_loss`` (KD + prototype + attention) is computed from a
    frozen, separately-pretrained teacher. This port's teacher is instead a
    submodule trained jointly with the student (see
    ``missingbench.models.mambavitakd.MambaVitAKD`` docstring), so it also
    receives its own segmentation loss on full modalities here -- the KD/
    prototype terms still only pull the student toward a *detached* teacher
    output, matching legacy's ``torch.no_grad()`` teacher forward pass.

    The attention-distillation term is precomputed by the model (it needs a
    learnable, optimizer-tracked ``ChannelAttention`` submodule, which a
    plain loss object cannot own) and passed straight through as
    ``outputs[2]``.
    """

    KWARG_SPEC: ClassVar[dict[str, KwargField]] = {
        "num_classes": positive_int(),
        "fuse_weight": nonneg_float(),
        "sep_weight": nonneg_float(),
        "prm_weight": nonneg_float(),
        "teacher_weight": nonneg_float(),
        "kd_weight": nonneg_float(),
        "proto_weight": nonneg_float(),
        "attn_weight": nonneg_float(),
        "temperature": positive_float(),
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
        teacher_weight: float = 1.0,
        kd_weight: float = 10.0,
        proto_weight: float = 0.1,
        attn_weight: float = 0.1,
        temperature: float = 10.0,
        eps: float = 1e-7,
        log_clamp_min: float = 0.005,
    ) -> None:
        resolved_num_classes = (
            int(num_classes)
            if num_classes is not None
            else int(num_cls)
            if num_cls is not None
            else 4
        )
        self.num_classes = resolved_num_classes
        self.fuse_weight = float(fuse_weight)
        self.sep_weight = float(sep_weight)
        self.prm_weight = float(prm_weight)
        self.teacher_weight = float(teacher_weight)
        self.kd_weight = float(kd_weight)
        self.proto_weight = float(proto_weight)
        self.attn_weight = float(attn_weight)
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.log_clamp_min = float(log_clamp_min)

    def segmentation_loss(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self.softmax_weighted_loss(output, target) + self.dice_loss(
            output, target
        )

    def dice_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_loss(output, target, num_cls=self.num_classes, eps=self.eps)

    def softmax_weighted_loss(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return softmax_weighted_loss(
            output,
            target,
            num_cls=self.num_classes,
            log_clamp_min=self.log_clamp_min,
            eps=self.eps,
        )

    def _branch_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        cross = self.softmax_weighted_loss(prediction, target)
        dice = self.dice_loss(prediction, target)
        return {"cross": cross, "dice": dice, "total": cross + dice}

    def _multi_branch_loss(
        self, predictions: tuple[torch.Tensor, ...], target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if not predictions:
            zero = target.new_tensor(0.0)
            return {"cross": zero, "dice": zero, "total": zero}

        cross = target.new_tensor(0.0)
        dice = target.new_tensor(0.0)
        for prediction in predictions:
            branch = self._branch_loss(prediction, target)
            cross = cross + branch["cross"]
            dice = dice + branch["dice"]
        return {"cross": cross, "dice": dice, "total": cross + dice}

    def teacher_pretrain_loss(
        self,
        teacher_out: tuple,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Direct segmentation loss for ``MambaVitAKDTrainer``'s dedicated
        teacher-pretraining phase (mirrors legacy's separate ``train_RFNet.py``
        run): full fuse+sep+prm supervision on the always-full-modality
        teacher branch, at the same ``fuse_weight``/``sep_weight``/
        ``prm_weight`` used for ``teacher_total`` inside ``training_loss``."""
        fuse_pred_t, sep_preds_t, prm_preds_t, _feature_t, _logits_t = teacher_out

        fuse = self._branch_loss(fuse_pred_t, target)
        sep = self._multi_branch_loss(sep_preds_t, target)
        prm = self._multi_branch_loss(prm_preds_t, target)

        total = (
            self.fuse_weight * fuse["total"]
            + self.sep_weight * sep["total"]
            + self.prm_weight * prm["total"]
        )
        return {
            "loss": total,
            "fusecross": fuse["cross"],
            "fusedice": fuse["dice"],
            "sepcross": sep["cross"],
            "sepdice": sep["dice"],
            "prmcross": prm["cross"],
            "prmdice": prm["dice"],
        }

    def training_loss(
        self,
        outputs: tuple[tuple, tuple, torch.Tensor],
        target: torch.Tensor,
        include_fuse: bool,
        include_sep: bool = True,
    ) -> dict[str, torch.Tensor]:
        student_out, teacher_out, attn_loss = outputs
        fuse_pred, sep_preds, prm_preds, feature, logits = student_out
        fuse_pred_t, sep_preds_t, prm_preds_t, feature_t, logits_t = teacher_out

        zero = target.new_tensor(0.0)

        fuse = (
            self._branch_loss(fuse_pred, target)
            if include_fuse
            else {"cross": zero, "dice": zero, "total": zero}
        )
        sep = (
            self._multi_branch_loss(sep_preds, target)
            if include_sep
            else {"cross": zero, "dice": zero, "total": zero}
        )
        prm = self._multi_branch_loss(prm_preds, target)

        student_total = self.prm_weight * prm["total"]
        if include_sep:
            student_total = student_total + (self.sep_weight * sep["total"])
        if include_fuse:
            student_total = student_total + (self.fuse_weight * fuse["total"])

        if sep_preds_t:
            teacher_fuse = (
                self._branch_loss(fuse_pred_t, target) if include_fuse else {"total": zero}
            )
            teacher_sep = self._multi_branch_loss(sep_preds_t, target)
            teacher_prm = self._multi_branch_loss(prm_preds_t, target)
            teacher_total = (
                self.prm_weight * teacher_prm["total"]
                + self.sep_weight * teacher_sep["total"]
                + (self.fuse_weight * teacher_fuse["total"] if include_fuse else zero)
            )
        else:
            # Frozen teacher (run without its per-modality decoders): nothing left to train on it.
            teacher_total = zero

        kd_loss = softmax_kl_loss(
            logits / self.temperature, logits_t.detach() / self.temperature
        ).mean()
        proto_loss = prototype_loss(
            feature, feature_t.detach(), target, self.num_classes
        )
        apkd_loss = (
            self.kd_weight * kd_loss
            + self.proto_weight * proto_loss
            + self.attn_weight * attn_loss
        )

        total = student_total + (self.teacher_weight * teacher_total) + apkd_loss

        return {
            "loss": total,
            "fusecross": fuse["cross"],
            "fusedice": fuse["dice"],
            "sepcross": sep["cross"],
            "sepdice": sep["dice"],
            "prmcross": prm["cross"],
            "prmdice": prm["dice"],
            "kd": kd_loss,
            "proto": proto_loss,
            "attn": attn_loss,
            "teacher": teacher_total,
        }


__all__ = ["MambaVitAKDLoss", "prototype_loss", "softmax_kl_loss"]
