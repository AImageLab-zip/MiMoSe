from __future__ import annotations

from itertools import chain, combinations
from typing import Any, ClassVar

import torch

from missingbench.losses.config import KwargField, positive_float, positive_int, unit_interval_float
from missingbench.models.uhved import DATASET_MODALITY_ORDER, MODALITIES

KLD_WEIGHT = 0.1
RECON_WEIGHT = 0.1


def _all_nonempty_subsets(items: list[str]) -> list[tuple[str, ...]]:
    return list(chain(*(combinations(items, size) for size in range(1, len(items) + 1))))


SUBSETS_MODALITIES = _all_nonempty_subsets(MODALITIES)


def dice_loss(output: torch.Tensor, target: torch.Tensor, *, num_cls: int, eps: float = 1e-7) -> torch.Tensor:
    target = target.float()
    dice = output.new_tensor(0.0)
    for class_index in range(num_cls):
        numerator = torch.sum(output[:, class_index] * target[:, class_index])
        left = torch.sum(output[:, class_index])
        right = torch.sum(target[:, class_index])
        dice = dice + (2.0 * numerator / (left + right + eps))
    return 1.0 - (dice / num_cls)


def softmax_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    *,
    num_cls: int,
    log_clamp_min: float = 0.005,
) -> torch.Tensor:
    target = target.float()
    cross_loss = output.new_tensor(0.0)
    for class_index in range(num_cls):
        output_i = output[:, class_index]
        target_i = target[:, class_index]
        cross_loss = cross_loss + (
            -target_i * torch.log(torch.clamp(output_i, min=log_clamp_min, max=1.0)).float()
        )
    return torch.mean(cross_loss)


def _masker(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x)
    y[mask, ...] = x[mask, ...]
    return y


def _kl_divergence(
    mu_1: torch.Tensor,
    logvar_1: torch.Tensor,
    mu_2: torch.Tensor,
    logvar_2: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    var_1 = torch.exp(logvar_1)
    var_2 = torch.exp(logvar_2)
    return 0.5 * torch.mean(-1 + logvar_2 - logvar_1 + (var_1 + torch.square(mu_1 - mu_2)) / (var_2 + eps))


def _product_gaussian(
    means: dict[str, torch.Tensor],
    logvars: dict[str, torch.Tensor],
    modalities: tuple[str, ...],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mu_prior = torch.zeros_like(means[MODALITIES[0]])
    log_prior = torch.zeros_like(means[MODALITIES[0]])

    precision = torch.stack([1 / (torch.exp(logvars[mod]) + eps) for mod in modalities], dim=0)
    weighted_mu = torch.stack([means[mod] / (torch.exp(logvars[mod]) + eps) for mod in modalities], dim=0)
    precision = torch.cat([precision, (1 + log_prior).unsqueeze(0)], dim=0)
    weighted_mu = torch.cat([weighted_mu, mu_prior.unsqueeze(0)], dim=0)

    posterior_means = torch.sum(weighted_mu, dim=0) / torch.sum(precision, dim=0)
    var = 1 / torch.sum(precision, dim=0)
    posterior_logvars = torch.log(var + eps)
    return posterior_means, posterior_logvars


def _product_gaussian_masked(
    means: dict[str, torch.Tensor],
    logvars: dict[str, torch.Tensor],
    mask: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mu_prior = torch.zeros_like(means[MODALITIES[0]])
    log_prior = torch.zeros_like(means[MODALITIES[0]])

    precision = _masker(
        torch.stack([1 / (torch.exp(logvars[mod]) + eps) for mod in MODALITIES], dim=0),
        mask.permute(1, 0),
    )
    weighted_mu = _masker(
        torch.stack([means[mod] / (torch.exp(logvars[mod]) + eps) for mod in MODALITIES], dim=0),
        mask.permute(1, 0),
    )
    precision = torch.cat([precision, (1 + log_prior).unsqueeze(0)], dim=0)
    weighted_mu = torch.cat([weighted_mu, mu_prior.unsqueeze(0)], dim=0)

    posterior_means = torch.sum(weighted_mu, dim=0) / torch.sum(precision, dim=0)
    var = 1 / torch.sum(precision, dim=0)
    posterior_logvars = torch.log(var + eps)
    return posterior_means, posterior_logvars


def compute_kld(
    means: dict[str, torch.Tensor],
    logvars: dict[str, torch.Tensor],
    mask: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inter-modality and prior KLD, averaged over all 15 non-empty modality subsets."""
    mu_prior = torch.zeros_like(means[MODALITIES[0]])
    log_prior = torch.zeros_like(means[MODALITIES[0]])

    full_means, full_logvars = _product_gaussian_masked(means, logvars, mask, eps)

    sum_inter_kld = full_means.new_tensor(0.0)
    sum_prior_kld = full_means.new_tensor(0.0)
    for subset in SUBSETS_MODALITIES:
        sub_means, sub_logvars = _product_gaussian(means, logvars, subset, eps)
        sum_inter_kld = sum_inter_kld + _kl_divergence(full_means, full_logvars, sub_means, sub_logvars, eps)
        sum_prior_kld = sum_prior_kld + _kl_divergence(sub_means, sub_logvars, mu_prior, log_prior, eps)

    num_subsets = len(SUBSETS_MODALITIES)
    return sum_inter_kld / num_subsets, sum_prior_kld / num_subsets


class UHVEDLoss:
    """Loss wrapper matching the legacy U-HVED training objective.

    ``kld_weight``/``recon_weight`` are hardcoded to the legacy defaults (0.1)
    rather than exposed as kwargs, matching the original U-HVED paper/code.
    """

    KWARG_SPEC: ClassVar[dict[str, KwargField]] = {
        "num_classes": positive_int(),
        "eps": positive_float(),
        "log_clamp_min": unit_interval_float(),
    }

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        num_cls: int | None = None,
        eps: float = 1e-5,
        log_clamp_min: float = 0.005,
    ) -> None:
        resolved_num_classes = (
            int(num_classes) if num_classes is not None else int(num_cls) if num_cls is not None else 4
        )
        self.num_classes = resolved_num_classes
        self.eps = float(eps)
        self.log_clamp_min = float(log_clamp_min)

    def segmentation_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.softmax_loss(output, target) + self.dice_loss(output, target)

    def dice_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_loss(output, target, num_cls=self.num_classes, eps=1e-7)

    def softmax_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return softmax_loss(output, target, num_cls=self.num_classes, log_clamp_min=self.log_clamp_min)

    def training_loss(
        self,
        outputs: dict[str, Any],
        post_param: list[dict[str, dict[str, torch.Tensor]]],
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        remapped_mask = mask[:, DATASET_MODALITY_ORDER]

        cross = self.softmax_loss(outputs["seg"], target)
        dice = self.dice_loss(outputs["seg"], target)

        images = outputs["images"]
        recon = outputs["seg"].new_tensor(0.0)
        for key in MODALITIES:
            recon = recon + torch.mean(torch.square(outputs[key] - images[key]))
        # Legacy builds one 4-channel tensor and takes a single `torch.mean`
        # over it, which equals the average (not sum) of the 4 per-modality
        # means -- divide to match, otherwise this term is ~4x too heavy
        # relative to RECON_WEIGHT.
        recon = recon / len(MODALITIES)

        kld = outputs["seg"].new_tensor(0.0)
        for level in post_param:
            inter_kld, prior_kld = compute_kld(level["mu"], level["logvar"], remapped_mask, self.eps)
            kld = kld + inter_kld + prior_kld
        kld = kld / len(post_param)

        loss = cross + dice + (KLD_WEIGHT * kld) + (RECON_WEIGHT * recon)

        return {
            "loss": loss,
            "cross": cross,
            "dice": dice,
            "kld": kld,
            "recon": recon,
        }


__all__ = [
    "UHVEDLoss",
    "compute_kld",
    "dice_loss",
    "softmax_loss",
]
