"""Placing gate thresholds so that requested budget matches realized parameters.

At initialization the thresholds are arbitrary, so ``budget -> parameters`` is
monotonic but nowhere near the diagonal.  Fixing that does not need gradient
descent: for a uniform width scale ``alpha``, the parameter count is a known,
strictly increasing function of ``alpha``, so we can simply invert it.

``calibrate_thresholds_analytic`` places each group's threshold at the budget
where a uniformly-scaled nnU-Net would first be wide enough to contain it.  This
yields an exact diagonal, and -- unlike a gradient fit -- a *determinate*
per-stage allocation: nnU-Net's own channel-doubling convention, which is a
defensible prior for the segmentation training that follows.

A gradient-based fallback is provided for ablations, but note it is badly
under-determined (hundreds of thresholds fitted against a one-dimensional curve)
and must be fitted against the *hard* parameter count: at any budget one group
per stage sits near its threshold contributing about half its channels, which at
the bottleneck is worth several million parameters.  Fitting the soft count and
plotting the hard one produces a visibly wrong diagonal.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

from missingbench.nas.budget import budget_to_z, sample_log_uniform_budget, z_to_budget
from missingbench.nas.gates import WidthGates
from missingbench.nas.param_count import exact_parameters

__all__ = [
    "uniform_width_schedule",
    "alpha_for_budget",
    "calibrate_thresholds_analytic",
    "calibrate_thresholds_gradient",
    "budget_param_curve",
]


def uniform_width_schedule(
    max_channels: Sequence[int],
    alpha: float,
    *,
    group_size: int = 4,
) -> tuple[int, ...]:
    """Scale every stage by ``alpha``, rounded to the gate granularity.

    Each stage keeps at least one group, so the result is always a valid
    architecture even at ``alpha -> 0``.
    """
    widths = []
    for channels in max_channels:
        scaled = int(round(channels * alpha / group_size)) * group_size
        widths.append(max(scaled, group_size))
    return tuple(widths)


def alpha_for_budget(
    budget: float,
    max_channels: Sequence[int],
    *,
    group_size: int = 4,
    num_modals: int = 4,
    num_cls: int = 4,
    n_conv_per_stage: int = 2,
    n_conv_per_stage_decoder: int = 2,
    cost_fn: Callable[[Sequence[int]], float] | None = None,
    tolerance: float = 1e-6,
    max_iterations: int = 200,
) -> float:
    """Invert ``cost(alpha) = budget`` by bisection.

    ``cost_fn`` defaults to the parameter count; pass ``exact_macs`` (suitably
    bound) to calibrate a FLOPs budget instead.  Either cost is monotone in
    ``alpha``, so bisection is both robust and fast (milliseconds), and needs no
    autograd.
    """
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")

    if cost_fn is None:

        def cost_fn(widths: Sequence[int]) -> float:  # noqa: F811 - local default
            return float(
                exact_parameters(
                    widths,
                    num_modals=num_modals,
                    num_cls=num_cls,
                    n_conv_per_stage=n_conv_per_stage,
                    n_conv_per_stage_decoder=n_conv_per_stage_decoder,
                )
            )

    def params_at(alpha: float) -> float:
        return cost_fn(
            uniform_width_schedule(max_channels, alpha, group_size=group_size)
        )

    low, high = 0.0, 1.0
    # Grow the bracket if the caller asks for more than the supernet can hold.
    while params_at(high) < budget and high < 1e3:
        high *= 2.0

    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        if params_at(mid) < budget:
            low = mid
        else:
            high = mid
        if high - low < tolerance:
            break
    return 0.5 * (low + high)


@torch.no_grad()
def calibrate_thresholds_analytic(
    gates: WidthGates,
    *,
    num_modals: int = 4,
    num_cls: int = 4,
    n_conv_per_stage: int = 2,
    n_conv_per_stage_decoder: int = 2,
    cost_fn: Callable[[Sequence[int]], float] | None = None,
    samples: int = 512,
) -> None:
    """Place thresholds so the realized width matches a uniformly-scaled nnU-Net.

    Group ``i`` of stage ``s`` covers channels ``[i*G, (i+1)*G)``.  We switch it
    on at the budget whose uniform-scale width first reaches the group's
    midpoint, ``(i + 0.5) * G``; the midpoint (rather than the lower edge) keeps
    the rounding symmetric with ``uniform_width_schedule``.

    Thresholds are written through the cumulative parameterization, so the
    ordering invariant is preserved by construction.
    """
    # Tabulate z -> alpha once, then invert per group by searching the table.
    z_grid = torch.linspace(0.0, 1.0, samples, dtype=torch.float64)
    alpha_grid = torch.tensor(
        [
            alpha_for_budget(
                float(z_to_budget(z, gates.budget_min, gates.budget_max)),
                gates.max_channels,
                group_size=gates.group_size,
                num_modals=num_modals,
                num_cls=num_cls,
                n_conv_per_stage=n_conv_per_stage,
                n_conv_per_stage_decoder=n_conv_per_stage_decoder,
                cost_fn=cost_fn,
            )
            for z in z_grid
        ],
        dtype=torch.float64,
    )

    for stage, channels in enumerate(gates.max_channels):
        n_groups = gates.groups_per_stage[stage]
        width_grid = alpha_grid * channels

        # A threshold is a decision boundary, so it belongs *between* the last
        # sample where the group should be off and the first where it should be
        # on -- not exactly on the latter. Placing it exactly on that sample
        # makes the gate evaluate to sigmoid(0) = 0.5 there, which reads as off
        # and costs the group its first budget point.
        half_step = 0.5 / (samples - 1)

        stage_thresholds = []
        for group in range(n_groups):
            target_width = (group + 0.5) * gates.group_size
            # First z whose uniform width reaches this group's midpoint.
            reached = (width_grid >= target_width).nonzero()
            if reached.numel() == 0:
                # Never reached inside the range: park it just past the top so
                # the group stays off across the whole budget range.
                stage_thresholds.append(1.0 + 1e-3)
            else:
                stage_thresholds.append(float(z_grid[int(reached[0])]) - half_step)

        thresholds = torch.tensor(stage_thresholds, dtype=torch.float32)
        # Enforce non-decreasing before inverting the parameterization; the grid
        # search can tie adjacent groups, which is harmless but must not go
        # backwards.
        thresholds, _ = torch.cummax(thresholds, dim=0)

        gates.base_threshold[stage] = thresholds[0]
        if n_groups > 1:
            increments = (thresholds[1:] - thresholds[:-1]).clamp_min(0.0)
            # Invert softplus: delta = log(exp(y) - 1), stable via expm1.
            gates.delta[stage].copy_(
                torch.log(torch.expm1(increments.clamp_min(1e-8)))
            )


def calibrate_thresholds_gradient(
    gates: WidthGates,
    *,
    steps: int = 2000,
    lr: float = 1e-2,
    batch: int = 64,
    num_modals: int = 4,
    num_cls: int = 4,
    n_conv_per_stage: int = 2,
    n_conv_per_stage_decoder: int = 2,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Fit thresholds by minimizing squared error in log-parameter space.

    Kept for ablation against the analytic placement.  Optimizes the *soft*
    parameter count (the only differentiable one) but reports the final *hard*
    error, which is what the milestone figure plots -- the gap between the two is
    exactly the half-active boundary group discussed in the module docstring.
    """
    from missingbench.nas.param_count import expected_parameters

    optimizer = torch.optim.Adam(gates.parameters(), lr=lr)
    history: list[float] = []

    for _ in range(steps):
        budgets = sample_log_uniform_budget(
            batch, gates.budget_min, gates.budget_max, generator=generator
        )
        z = budget_to_z(budgets, gates.budget_min, gates.budget_max)

        predicted = expected_parameters(
            gates.soft_widths(z),
            num_modals=num_modals,
            num_cls=num_cls,
            n_conv_per_stage=n_conv_per_stage,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        )
        loss = F.mse_loss(torch.log(predicted.clamp_min(1.0)), torch.log(budgets))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss))

    curve = budget_param_curve(
        gates,
        n=128,
        num_modals=num_modals,
        num_cls=num_cls,
        n_conv_per_stage=n_conv_per_stage,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
    )
    hard_rel = [abs(hard - b) / b for b, hard, _ in curve]
    return {
        "final_soft_loss": history[-1],
        "mean_hard_relative_error": sum(hard_rel) / len(hard_rel),
        "max_hard_relative_error": max(hard_rel),
    }


@torch.no_grad()
def budget_param_curve(
    gates: WidthGates,
    *,
    n: int = 128,
    num_modals: int = 4,
    num_cls: int = 4,
    n_conv_per_stage: int = 2,
    n_conv_per_stage_decoder: int = 2,
) -> list[tuple[float, int, float]]:
    """Sample ``(requested_budget, extracted_params, soft_params)`` triples.

    This is the data behind the milestone figure.  ``extracted_params`` is the
    exact count of the architecture the gates actually select, so it is the
    honest number to plot against the diagonal.
    """
    from missingbench.nas.param_count import expected_parameters

    log_min = math.log(gates.budget_min)
    log_max = math.log(gates.budget_max)
    budgets = torch.exp(torch.linspace(log_min, log_max, n, dtype=torch.float64))

    rows: list[tuple[float, int, float]] = []
    for budget in budgets:
        b = float(budget)
        z = budget_to_z(b, gates.budget_min, gates.budget_max)
        widths = tuple(int(w) for w in gates.hard_widths(z).reshape(-1))
        hard = exact_parameters(
            widths,
            num_modals=num_modals,
            num_cls=num_cls,
            n_conv_per_stage=n_conv_per_stage,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        )
        soft = float(
            expected_parameters(
                gates.soft_widths(z).to(torch.float64),
                num_modals=num_modals,
                num_cls=num_cls,
                n_conv_per_stage=n_conv_per_stage,
                n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            ).reshape(-1)[0]
        )
        rows.append((b, hard, soft))
    return rows
