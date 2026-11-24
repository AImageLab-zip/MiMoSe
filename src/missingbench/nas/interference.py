"""Does training different budgets pull the shared weights apart?

Multi-budget training rests on an assumption -- that subnetworks for nearby
budgets want the same thing from the weights they share, and distant ones do
not.  This module measures that directly instead of assuming it, by comparing
the gradients different budgets produce.

For a set of budgets ``b_1 .. b_n`` the quantity of interest is

    G_ij = cos( grad L_seg(A(b_i)), grad L_seg(A(b_j)) )

and the question is whether ``G_ij`` falls off as ``|log b_i - log b_j|`` grows.
If it does, local-budget training is empirically justified rather than merely
plausible.

Three things make the naive version of this measurement misleading, and all
three are handled here.

**Minibatch noise.**  Two gradients taken on *different* batches disagree
substantially even for identical architectures, simply because of sampling
noise.  Every budget is therefore evaluated on the *same* batch, so the only
difference between two gradients is the subnetwork; similarities are averaged
across batches afterwards.

**No noise floor.**  A cosine of 0.3 means nothing in isolation.
:func:`measure_gradient_interference` also computes, for each budget, the
similarity between gradients of *the same* budget on two different batches.
That is the ceiling any cross-budget pair could achieve given the noise, and it
is what cross-budget numbers must be read against.

**Support mismatch.**  A small subnetwork's gradient is exactly zero on every
channel it does not use, so a small/large pair scores low partly because their
gradients live on different coordinates -- not because they disagree about
direction.  Two cosines are therefore reported:

``full``
    Over all shared parameters.  Answers "how aligned are the updates as a
    whole", and necessarily decays with budget distance because supports
    diverge.
``overlap``
    Restricted to coordinates active in *both* subnetworks.  Answers the
    sharper question: on the weights they genuinely share, do they disagree?

Reporting only ``full`` would make budget distance look damning when much of the
effect is bookkeeping; reporting only ``overlap`` would hide that distant
budgets barely share weights at all.  ``overlap_fraction`` records how much
support the pair actually has in common.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

__all__ = [
    "InterferenceResult",
    "measure_gradient_interference",
]


@dataclass
class InterferenceResult:
    """Pairwise gradient statistics across budgets.

    All matrices are ``n x n`` with rows/columns ordered like ``budgets``.
    """

    budgets: list[float]
    cosine_full: list[list[float]]
    cosine_overlap: list[list[float]]
    overlap_fraction: list[list[float]]
    cosine_full_std: list[list[float]]
    #: Per-budget same-architecture, different-batch similarity: the ceiling a
    #: cross-budget pair could reach given minibatch noise.
    noise_floor: list[float]
    n_batches: int
    widths: list[tuple[int, ...]] = field(default_factory=list)
    n_shared_parameters: int = 0

    def log_distance(self, i: int, j: int) -> float:
        return abs(math.log(self.budgets[i]) - math.log(self.budgets[j]))

    def pairs(self) -> list[dict[str, float]]:
        """Flatten the upper triangle into one record per budget pair.

        ``relative_*`` divides a cosine by the pair's noise floor, which is the
        number to actually reason about.  A raw cosine of 0.73 sounds like
        substantial disagreement until you notice that the same architecture
        scores only 0.75 against itself on a different batch -- at which point
        the two budgets are agreeing as well as anything possibly could.  A
        relative value near 1 means no interference beyond minibatch noise;
        well below 1 means genuine conflict.
        """
        rows: list[dict[str, float]] = []
        for i in range(len(self.budgets)):
            for j in range(i + 1, len(self.budgets)):
                # Best a pair could do given noise, averaged over the two.
                floor = 0.5 * (self.noise_floor[i] + self.noise_floor[j])
                rows.append(
                    {
                        "budget_i": self.budgets[i],
                        "budget_j": self.budgets[j],
                        "log_distance": self.log_distance(i, j),
                        "cosine_full": self.cosine_full[i][j],
                        "cosine_full_std": self.cosine_full_std[i][j],
                        "cosine_overlap": self.cosine_overlap[i][j],
                        "overlap_fraction": self.overlap_fraction[i][j],
                        "noise_floor": floor,
                        "relative_full": (
                            self.cosine_full[i][j] / floor
                            if floor
                            else float("nan")
                        ),
                        "relative_overlap": (
                            self.cosine_overlap[i][j] / floor
                            if floor
                            else float("nan")
                        ),
                    }
                )
        return rows


def _shared_parameters(model: Any) -> list[torch.nn.Parameter]:
    """Weights shared across budgets, i.e. everything except the gates.

    Gate thresholds are excluded deliberately: they parameterize *which*
    subnetwork runs rather than what it computes, so their gradients would
    describe the budget conditioning rather than any conflict over the weights.
    """
    gate_ids = {id(p) for p in model.gates.parameters()}
    return [p for p in model.parameters() if id(p) not in gate_ids and p.requires_grad]


def _flat_gradient(parameters: Sequence[torch.nn.Parameter]) -> Tensor:
    """Concatenate gradients into one vector, treating missing grads as zero."""
    chunks = []
    for parameter in parameters:
        if parameter.grad is None:
            chunks.append(torch.zeros(parameter.numel(), device=parameter.device))
        else:
            chunks.append(parameter.grad.detach().reshape(-1))
    return torch.cat(chunks)


def _cosine(a: Tensor, b: Tensor) -> float:
    denominator = a.norm() * b.norm()
    if float(denominator) == 0.0:
        return float("nan")
    return float(torch.dot(a, b) / denominator)


def _masked_cosine(a: Tensor, b: Tensor) -> tuple[float, float]:
    """Cosine restricted to coordinates live in both gradients, plus the share.

    Dead channels receive exactly zero gradient (the gate multiplies their
    activations by a hard zero), so ``!= 0`` is an exact support test rather
    than a threshold heuristic.
    """
    support_a = a != 0
    support_b = b != 0
    both = support_a & support_b
    either = support_a | support_b

    either_count = int(either.sum())
    overlap_fraction = float(int(both.sum()) / either_count) if either_count else 0.0

    if not bool(both.any()):
        return float("nan"), overlap_fraction
    return _cosine(a[both], b[both]), overlap_fraction


@torch.enable_grad()
def _gradient_for_budget(
    trainer: Any,
    batch: dict[str, Tensor],
    budget: float,
    parameters: Sequence[torch.nn.Parameter],
) -> Tensor:
    """Gradient of the *segmentation* loss alone for one budget on one batch.

    The budget penalty is excluded on purpose: it is a function of the gates
    only, so including it would add a term that is identical in kind for every
    budget and touches none of the shared weights being compared.
    """
    model = trainer._model_for_state()
    model.set_budget(budget)
    model.zero_grad(set_to_none=True)

    images = batch["images"].to(trainer.device, non_blocking=True)
    seg = batch["seg"].to(trainer.device, non_blocking=True).long()
    mask = batch["mask"].to(trainer.device, non_blocking=True).bool()

    with trainer._autocast_context():
        target = trainer._seg_to_one_hot(seg)
        outputs = trainer.model(images, mask)
        loss = trainer._loss_impl().training_loss(
            outputs, target, include_fuse=True
        )["loss"]
    loss.backward()

    gradient = _flat_gradient(parameters).float().cpu()
    model.zero_grad(set_to_none=True)
    return gradient


def measure_gradient_interference(
    trainer: Any,
    budgets: Sequence[float],
    *,
    n_batches: int = 8,
    progress_callback: Any | None = None,
) -> InterferenceResult:
    """Measure pairwise gradient agreement between budgets.

    Args:
        trainer: a built ``MatUnetTrainer`` (supplies the model, loss, device and
            data loader).
        budgets: budgets to compare, ideally log-spaced across the range.
        n_batches: batches to average over. Each batch costs one
            forward/backward per budget, plus one extra for the noise floor.
    """
    budgets = [float(b) for b in budgets]
    n = len(budgets)
    if n < 2:
        raise ValueError("need at least two budgets to compare")
    if n_batches < 2:
        raise ValueError("need at least two batches to estimate the noise floor")

    model = trainer._model_for_state()
    parameters = _shared_parameters(model)
    n_shared = sum(p.numel() for p in parameters)

    was_training = getattr(model, "is_training", False)
    model.train()
    trainer._set_aux_training_flag(True)

    widths: list[tuple[int, ...]] = []
    for budget in budgets:
        model.set_budget(budget)
        widths.append(tuple(model.current_widths()))

    full_samples = [[[] for _ in range(n)] for _ in range(n)]
    overlap_samples = [[[] for _ in range(n)] for _ in range(n)]
    fraction_samples = [[[] for _ in range(n)] for _ in range(n)]
    # noise_samples[i]: same budget, consecutive (different) batches.
    noise_samples: list[list[float]] = [[] for _ in range(n)]

    previous: list[Tensor] | None = None
    try:
        for batch_index in range(n_batches):
            batch = trainer._next_train_batch()

            # All budgets see the identical batch, so any disagreement between
            # them is architectural rather than sampling noise.
            gradients = [
                _gradient_for_budget(trainer, batch, budget, parameters)
                for budget in budgets
            ]

            for i in range(n):
                for j in range(i + 1, n):
                    full_samples[i][j].append(_cosine(gradients[i], gradients[j]))
                    overlap, fraction = _masked_cosine(gradients[i], gradients[j])
                    overlap_samples[i][j].append(overlap)
                    fraction_samples[i][j].append(fraction)

            # Noise floor: same architecture, different batch.
            if previous is not None:
                for i in range(n):
                    noise_samples[i].append(_cosine(previous[i], gradients[i]))
            previous = gradients

            if progress_callback is not None:
                progress_callback(batch_index + 1, n_batches)
    finally:
        trainer._set_aux_training_flag(was_training)
        model.zero_grad(set_to_none=True)

    def _mean(values: list[float]) -> float:
        clean = [v for v in values if not math.isnan(v)]
        return sum(clean) / len(clean) if clean else float("nan")

    def _std(values: list[float]) -> float:
        clean = [v for v in values if not math.isnan(v)]
        if len(clean) < 2:
            return 0.0
        mean = sum(clean) / len(clean)
        return math.sqrt(sum((v - mean) ** 2 for v in clean) / (len(clean) - 1))

    full = [[1.0] * n for _ in range(n)]
    overlap = [[1.0] * n for _ in range(n)]
    fraction = [[1.0] * n for _ in range(n)]
    full_std = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            value = _mean(full_samples[i][j])
            full[i][j] = full[j][i] = value
            full_std[i][j] = full_std[j][i] = _std(full_samples[i][j])
            value = _mean(overlap_samples[i][j])
            overlap[i][j] = overlap[j][i] = value
            value = _mean(fraction_samples[i][j])
            fraction[i][j] = fraction[j][i] = value

    return InterferenceResult(
        budgets=budgets,
        cosine_full=full,
        cosine_overlap=overlap,
        overlap_fraction=fraction,
        cosine_full_std=full_std,
        noise_floor=[_mean(samples) for samples in noise_samples],
        n_batches=n_batches,
        widths=widths,
        n_shared_parameters=n_shared,
    )
