"""Budget-conditioned, monotonically nested width gates.

Each resolution stage splits its maximum channel count into groups of a fixed
size.  Group ``i`` of stage ``s`` owns a learnable activation threshold
``t[s, i]``, and is switched on smoothly as a function of the normalized budget
coordinate ``z``:

    g[s, i](z) = sigmoid((z - t[s, i]) / tau)

Two properties this buys, both structural rather than learned:

**Monotonicity.**  ``sigmoid`` is increasing in ``z``, so ``b1 < b2`` implies
``g[s,i](b1) <= g[s,i](b2)`` for every group.  A larger budget can therefore only
ever *add* channels -- the nesting requirement ``A(b1) subset A(b2)`` holds by
construction, not by penalty.

**Prefix structure.**  Thresholds use a cumulative parameterization,

    t[s, i] = t0[s] + sum_{j <= i} softplus(delta[s, j])

which forces ``t[s, 0] <= t[s, 1] <= ...``.  The active set of any stage is then
always an index *prefix* ``arange(c)``, so extracting a subnetwork is a slice
rather than a scatter.  (Monotonicity alone would not give this: unordered
thresholds still nest, but would activate channels in an arbitrary permutation.)

Group 0 is pinned on as a literal constant rather than a parameter -- a
parameter that never receives gradient is precisely the case that forces
``find_unused_parameters`` in DDP.

Note on conditioning: because ``t0[s]`` appears in every threshold of its stage,
its gradient is the *sum* over all groups and it therefore moves roughly ``G``
times faster than the last ``delta``.  When supernet training starts,
put these parameters in their own optimizer group with a scaled learning rate.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from missingbench.nas.budget import budget_to_z

__all__ = ["WidthGates", "straight_through"]


def straight_through(g_soft: Tensor) -> Tensor:
    """Binarize in the forward pass, keep the soft gradient in the backward one.

    A plain ``sigmoid`` never reaches exactly 1, so using it directly to mask
    activations would scale every live channel by ``g < 1`` -- compounded across
    ~18 layers, and with the boundary group multiplied by ~0.5.  The supernet
    forward would then *not* equal the standalone network extracted from it,
    which is the one property the whole design rests on.  Rounding forward while
    passing the soft derivative backward keeps the forward exact and still lets
    the thresholds learn.
    """
    g_hard = (g_soft > 0.5).to(g_soft.dtype)
    return g_hard + g_soft - g_soft.detach()


class WidthGates(nn.Module):
    """Per-stage, per-group budget gates for a width-elastic network.

    Args:
        max_channels: the supernet's per-stage channel counts (the widest
            architecture in the space).
        group_size: channels switched on or off together.  A *constant group
            size* is deliberate: a constant group *count* would make the
            bottleneck's quantum 64 channels wide while stage 0's is 4, and the
            small-budget end of the range would then quantize in ~4x parameter
            jumps.  Constant size keeps the granularity uniform in relative
            width, which is what a log-budget parameterization wants.
        tau: gate temperature.  Anneal toward 0 during training to sharpen the
            gates; the forward pass is already binary via the straight-through
            estimator, so ``tau`` only shapes gradients.
    """

    def __init__(
        self,
        max_channels: Sequence[int],
        *,
        budget_min: float,
        budget_max: float,
        group_size: int = 4,
        tau: float = 0.05,
        num_modals: int = 4,
    ) -> None:
        super().__init__()

        self.num_modals = int(num_modals)
        self.max_channels = tuple(int(c) for c in max_channels)
        if any(c <= 0 for c in self.max_channels):
            raise ValueError(f"max_channels must be positive, got {max_channels}")
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")
        for stage, channels in enumerate(self.max_channels):
            if channels % group_size:
                raise ValueError(
                    f"stage {stage} has {channels} channels, which is not a "
                    f"multiple of group_size={group_size}"
                )
        # LatentMaskConditioning's hidden width is bottleneck // 2, and we derive
        # its gate by halving the bottleneck group gates -- so a group must stay
        # an even number of channels for that halving to be exact.
        if group_size % 2:
            raise ValueError(f"group_size must be even, got {group_size}")

        if tau <= 0.0:
            raise ValueError(f"tau must be positive, got {tau}")

        self.group_size = int(group_size)
        self.budget_min = float(budget_min)
        self.budget_max = float(budget_max)
        self.n_stages = len(self.max_channels)
        self.groups_per_stage = tuple(c // group_size for c in self.max_channels)

        self.register_buffer("tau", torch.tensor(float(tau)), persistent=True)

        # One base threshold per stage, plus one increment per optional group.
        # Group 0 is always active, so it owns no parameter; stage s therefore
        # has (groups_per_stage[s] - 1) increments.
        self.base_threshold = nn.Parameter(torch.zeros(self.n_stages))
        self.delta = nn.ParameterList(
            [torch.zeros(max(n_groups - 1, 0)) for n_groups in self.groups_per_stage]
        )

    def extra_repr(self) -> str:
        return (
            f"max_channels={self.max_channels}, group_size={self.group_size}, "
            f"groups_per_stage={self.groups_per_stage}, tau={float(self.tau):.4g}, "
            f"budget_range=({self.budget_min:.3g}, {self.budget_max:.3g})"
        )

    def set_tau(self, tau: float) -> None:
        """Anneal the gate temperature in place."""
        if tau <= 0.0:
            raise ValueError(f"tau must be positive, got {tau}")
        self.tau.fill_(float(tau))

    def thresholds(self) -> list[Tensor]:
        """Ordered thresholds per stage, each shaped ``[groups_per_stage[s]]``.

        ``softplus`` keeps every increment positive, so the returned sequence is
        non-decreasing and the active set stays a prefix.
        """
        out: list[Tensor] = []
        for stage in range(self.n_stages):
            base = self.base_threshold[stage]
            increments = F.softplus(self.delta[stage])
            cumulative = torch.cumsum(increments, dim=0)
            out.append(torch.cat([base.unsqueeze(0), base + cumulative]))
        return out

    def group_gates(self, z: Tensor, *, hard: bool = True) -> list[Tensor]:
        """Gate value per group, each shaped ``[..., groups_per_stage[s]]``."""
        # ``z`` usually arrives as a plain float (a budget converted by
        # ``budget_to_z``), which lands on the CPU regardless of where the
        # gates live -- so pin it to the module's own device before it meets
        # ``thresholds()``.
        z = torch.as_tensor(z, device=self.tau.device)
        if z.ndim == 0:
            z = z.reshape(1)

        gates: list[Tensor] = []
        for thresholds in self.thresholds():
            logits = (z[..., None] - thresholds) / self.tau
            g = torch.sigmoid(logits)
            if hard:
                g = straight_through(g)
            # Pin group 0 on. Done by masking rather than by clamping a
            # parameter so no gradient path exists to it at all.
            always_on = torch.ones_like(g[..., :1])
            g = torch.cat([always_on, g[..., 1:]], dim=-1)
            gates.append(g)
        return gates

    def channel_gates(self, z: Tensor, *, hard: bool = True) -> list[Tensor]:
        """Expand group gates to per-channel gates, ``[..., max_channels[s]]``."""
        return [
            g.repeat_interleave(self.group_size, dim=-1)
            for g in self.group_gates(z, hard=hard)
        ]

    def hidden_gate(self, z: Tensor, *, hard: bool = True) -> Tensor:
        """Gate for ``LatentMaskConditioning``'s hidden layer.

        ``LatentMaskConditioning`` sizes its hidden layer as
        ``max(bottleneck // 2, num_modals * 2)``, so each bottleneck group of
        ``group_size`` channels maps onto ``group_size // 2`` hidden units --
        *except* that the floor pins a minimum number of units on regardless of
        how narrow the bottleneck gets.  Honouring that floor here is what keeps
        the gated forward equal to the extracted network at small budgets; the
        floor is inactive once the bottleneck exceeds ``num_modals * 4``
        channels, which covers the whole calibrated range of a realistic
        supernet but emphatically not the small end.

        Deriving this from the bottleneck gates rather than giving it thresholds
        of its own adds no parameters and keeps the two widths consistent.
        """
        bottleneck = self.group_gates(z, hard=hard)[-1]
        gate = bottleneck.repeat_interleave(self.group_size // 2, dim=-1)

        floor = min(self.num_modals * 2, gate.shape[-1])
        if floor:
            always_on = torch.ones_like(gate[..., :floor])
            gate = torch.cat([always_on, gate[..., floor:]], dim=-1)
        return gate

    def soft_widths(self, z: Tensor) -> Tensor:
        """Differentiable per-stage widths, ``[..., n_stages]``.

        Uses the *soft* gates on purpose: this feeds the parameter-budget
        penalty, which needs a real gradient with respect to the thresholds.
        Activations use the hard path instead.
        """
        gates = self.group_gates(z, hard=False)
        widths = [g.sum(dim=-1) * self.group_size for g in gates]
        return torch.stack(widths, dim=-1)

    def hard_widths(self, z: Tensor) -> Tensor:
        """Discrete per-stage widths, ``[..., n_stages]`` (integer-valued)."""
        with torch.no_grad():
            gates = self.group_gates(z, hard=True)
            widths = [(g > 0.5).sum(dim=-1) * self.group_size for g in gates]
        return torch.stack(widths, dim=-1)

    def widths_for_budget(self, budget: float) -> tuple[int, ...]:
        """Convenience: discrete widths selected for a scalar budget."""
        z = budget_to_z(budget, self.budget_min, self.budget_max)
        widths = self.hard_widths(z)
        return tuple(int(w) for w in widths.reshape(-1, self.n_stages)[0])
