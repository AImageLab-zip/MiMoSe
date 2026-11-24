"""Budget <-> normalized-coordinate conversions for the Pareto supernet.

All budget conditioning happens in *log-parameter space*: the target range spans
roughly three orders of magnitude, so a linear parameterization would spend
almost all of its resolution on the largest architectures.  Define

    z(b) = (log(b) - log(B_min)) / (log(B_max) - log(B_min))

so that z in [0, 1] covers the whole budget range with uniform relative
resolution.  Every gate in :mod:`missingbench.nas.gates` thresholds on ``z``, never on
``b`` directly.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = [
    "budget_to_z",
    "z_to_budget",
    "sample_log_uniform_budget",
]


def _validate_range(budget_min: float, budget_max: float) -> None:
    if budget_min <= 0.0 or budget_max <= 0.0:
        raise ValueError("budget bounds must be positive (they are logged)")
    if not budget_max > budget_min:
        raise ValueError(
            f"budget_max must exceed budget_min, got {budget_min} .. {budget_max}"
        )


def budget_to_z(
    budget: Tensor | float,
    budget_min: float,
    budget_max: float,
) -> Tensor:
    """Map a parameter budget to its normalized log-space coordinate.

    The result is *not* clamped: budgets outside ``[budget_min, budget_max]``
    produce ``z`` outside ``[0, 1]``, which the gates handle gracefully (they
    simply saturate).  Clamping here would silently hide a misconfigured range.
    """
    _validate_range(budget_min, budget_max)
    if not isinstance(budget, Tensor):
        budget = torch.tensor(float(budget))
    if torch.any(budget <= 0):
        raise ValueError("budgets must be positive (they are logged)")

    log_min = math.log(budget_min)
    log_max = math.log(budget_max)
    return (torch.log(budget) - log_min) / (log_max - log_min)


def z_to_budget(
    z: Tensor | float,
    budget_min: float,
    budget_max: float,
) -> Tensor:
    """Inverse of :func:`budget_to_z`."""
    _validate_range(budget_min, budget_max)
    if not isinstance(z, Tensor):
        z = torch.tensor(float(z))

    log_min = math.log(budget_min)
    log_max = math.log(budget_max)
    return torch.exp(z * (log_max - log_min) + log_min)


def sample_log_uniform_budget(
    n: int,
    budget_min: float,
    budget_max: float,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Draw ``n`` budgets with ``log(b) ~ Uniform(log B_min, log B_max)``.

    Sampling uniformly in *linear* budget space would put ~99% of the mass above
    a tenth of ``budget_max``, so the small-architecture end of the Pareto front
    would essentially never be trained.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    _validate_range(budget_min, budget_max)

    z = torch.rand(n, generator=generator, device=device)
    return z_to_budget(z, budget_min, budget_max)
