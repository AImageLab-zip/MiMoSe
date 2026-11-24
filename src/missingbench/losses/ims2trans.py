from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import torch
import torch.nn.functional as F

from missingbench.losses.config import KwargField, nonneg_float, positive_float, positive_int, unit_interval_float
from missingbench.losses.imfuse import dice_loss, softmax_weighted_loss


def contrastive_loss(emb_i: torch.Tensor, emb_j: torch.Tensor, *, temperature: float = 0.5) -> torch.Tensor:
    """Legacy IMS2Trans InfoNCE-style contrastive loss between two embedding batches."""
    batch_size = emb_i.size(0)
    z_i = F.normalize(emb_i, dim=1)
    z_j = F.normalize(emb_j, dim=1)

    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    sim_ij = torch.diag(similarity_matrix, batch_size)
    sim_ji = torch.diag(similarity_matrix, -batch_size)
    positives = torch.cat([sim_ij, sim_ji], dim=0)

    nominator = torch.exp(positives / temperature)
    negatives_mask = (
        ~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=emb_i.device)
    ).float()
    denominator = negatives_mask * torch.exp(similarity_matrix / temperature)

    loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
    return torch.sum(loss_partial) / (2 * batch_size)


class IMS2TransLoss:
    """Loss wrapper matching legacy IMS2Trans's fuse + pyramid + contrastive terms."""

    KWARG_SPEC: ClassVar[dict[str, KwargField]] = {
        "num_classes": positive_int(),
        "fuse_weight": nonneg_float(),
        "prm_weight": nonneg_float(),
        "dis_weight": nonneg_float(),
        "dis_temperature": positive_float(),
        "eps": positive_float(),
        "log_clamp_min": unit_interval_float(),
    }

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        num_cls: int | None = None,
        fuse_weight: float = 1.0,
        prm_weight: float = 1.0,
        dis_weight: float = 0.1,
        dis_temperature: float = 0.5,
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
        self.prm_weight = float(prm_weight)
        self.dis_weight = float(dis_weight)
        self.dis_temperature = float(dis_temperature)
        self.eps = float(eps)
        self.log_clamp_min = float(log_clamp_min)

    def training_loss(
        self,
        outputs: tuple[torch.Tensor, Sequence[torch.Tensor], Sequence[torch.Tensor]],
        target: torch.Tensor,
        include_fuse: bool,
    ) -> dict[str, torch.Tensor]:
        fuse_pred, embeddings, prm_preds = outputs
        *modal_embeddings, average_embedding = embeddings

        zero = target.new_tensor(0.0)

        fuse = self._branch_loss(fuse_pred, target) if include_fuse else {"cross": zero, "dice": zero, "total": zero}
        prm = self._multi_branch_loss(prm_preds, target)

        dis = zero
        for modal_embedding in modal_embeddings:
            dis = dis + contrastive_loss(modal_embedding, average_embedding, temperature=self.dis_temperature)

        total = self.prm_weight * prm["total"] + self.dis_weight * dis
        if include_fuse:
            total = total + (self.fuse_weight * fuse["total"])

        return {
            "loss": total,
            "fusecross": fuse["cross"],
            "fusedice": fuse["dice"],
            "prmcross": prm["cross"],
            "prmdice": prm["dice"],
            "dis": dis,
        }

    def segmentation_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.softmax_weighted_loss(output, target) + self.dice_loss(output, target)

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


__all__ = ["IMS2TransLoss", "contrastive_loss"]
