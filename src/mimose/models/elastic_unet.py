"""Width-elastic SimpleUnet supernet conditioned on a parameter budget.

One network is instantiated at the widest architecture in the search space.  A
requested budget selects a prefix of each stage's channels via
:class:`mimose.nas.gates.WidthGates`, and the un-selected channels are zeroed
after every convolution block.  Because ``Conv3d`` and ``InstanceNorm3d`` act
per-channel with no cross-channel statistics, the surviving channels compute
*numerically identical* values to a genuinely narrower network -- so the subnet
selected here and the standalone ``SimpleUnet`` produced by
:func:`mimose.nas.extract.extract_subnet` agree to floating-point roundoff.

Where the gate is applied matters, and not for efficiency:

* ``Conv3d`` has ``bias=True``, so a "dead" output channel carries ``conv.bias``
  rather than zero.
* ``InstanceNorm3d(affine=True)`` maps an all-zero channel to ``norm.bias``
  (variance 0 gives ``0/sqrt(eps) = 0``, then affine adds the bias) -- it does
  *not* stay zero, and it does not produce NaN.

Gating only a stage's final output would therefore let those constants flow into
the *next* convolution and contaminate the channels that are supposed to be
live.  The gate must be applied after every ``conv -> norm -> nonlin`` block,
and after every transposed convolution, which is what ``_apply_gate`` below
enforces.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from mimose.models.abstract_model import AbstractModel
from mimose.models.simple_unet import SimpleUnet
from mimose.nas.budget import budget_to_z
from mimose.nas.gates import WidthGates
from mimose.nas.flop_count import exact_macs, expected_macs
from mimose.nas.param_count import exact_parameters, expected_parameters

__all__ = ["ElasticUnet"]

DEFAULT_MAX_FEATURES = (64, 128, 256, 512, 1024)
DEFAULT_INPUT_SHAPE = (128, 128, 128)


class ElasticUnet(AbstractModel):
    """A budget-conditioned family of nested SimpleUnet architectures.

    Args:
        features_per_stage: widths of the *widest* architecture (the supernet).
        group_size: channel granularity of the gates.
        tau: gate temperature; anneal toward 0 during training.
        budget_metric: what a budget *counts*.  ``"params"`` (default) budgets
            parameters; ``"flops"`` budgets multiply-accumulates of one forward
            pass at ``input_shape``.  The choice is consequential, not cosmetic:
            parameters live in the deep low-resolution stages and MACs in the
            shallow high-resolution ones, so the two budgets cut the network in
            opposite places.  See :mod:`mimose.nas.flop_count`.
        budget_min / budget_max: budget range, in the unit ``budget_metric``
            selects.  Defaults to the exact range this supernet can express --
            the uniform 1/8-width network up to the full-width one -- so ``z``
            spans ``[0, 1]`` meaningfully.
    """

    def __init__(
        self,
        *,
        num_cls: int = 4,
        num_modals: int = 4,
        input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
        features_per_stage: Sequence[int] = DEFAULT_MAX_FEATURES,
        n_conv_per_stage: int = 2,
        n_conv_per_stage_decoder: int = 2,
        group_size: int = 4,
        tau: float = 0.05,
        budget_metric: str = "params",
        budget_min: float | None = None,
        budget_max: float | None = None,
        calibrate: bool = True,
        overlap: float = 0.5,
    ) -> None:
        super().__init__()

        if budget_metric not in ("params", "flops"):
            raise ValueError(
                f"budget_metric must be 'params' or 'flops', got {budget_metric!r}"
            )
        self.budget_metric = budget_metric

        self.max_features = tuple(int(c) for c in features_per_stage)
        self.num_cls = int(num_cls)
        self.num_modals = int(num_modals)
        self.n_conv_per_stage = int(n_conv_per_stage)
        self.n_conv_per_stage_decoder = int(n_conv_per_stage_decoder)

        # The supernet is a plain SimpleUnet at maximum width; gating happens in
        # this class's forward, leaving the wrapped module a stock architecture
        # that extraction can copy weights out of by name.
        self.supernet = SimpleUnet(
            num_cls=num_cls,
            num_modals=num_modals,
            input_shape=input_shape,
            features_per_stage=self.max_features,
            n_conv_per_stage=n_conv_per_stage,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            overlap=overlap,
        )

        common = {
            "num_modals": num_modals,
            "num_cls": num_cls,
            "n_conv_per_stage": n_conv_per_stage,
            "n_conv_per_stage_decoder": n_conv_per_stage_decoder,
        }
        self._input_shape = tuple(int(d) for d in input_shape)
        # One callable used for all three jobs that must agree on the unit:
        # the default budget range, the threshold calibration, and the exact
        # cost reported back. If they ever disagree the budget axis is a lie.
        if budget_metric == "flops":
            self._exact_cost = lambda widths: float(
                exact_macs(widths, input_shape=self._input_shape, **common)
            )
        else:
            self._exact_cost = lambda widths: float(exact_parameters(widths, **common))

        if budget_max is None:
            budget_max = self._exact_cost(self.max_features)
        if budget_min is None:
            smallest = tuple(max(c // 8, group_size) for c in self.max_features)
            budget_min = self._exact_cost(smallest)

        self.gates = WidthGates(
            self.max_features,
            budget_min=budget_min,
            budget_max=budget_max,
            group_size=group_size,
            tau=tau,
            num_modals=num_modals,
        )

        if calibrate:
            # Untouched thresholds describe an arbitrary architecture family:
            # every increment starts at softplus(0) ~ 0.69, so even the maximum
            # budget would activate only the first couple of groups per stage.
            # Calibrating here (~100 ms, no data, no GPU) means a freshly built
            # supernet already spans its stated budget range, instead of
            # silently serving a degenerate one.
            from mimose.nas.calibration import calibrate_thresholds_analytic

            calibrate_thresholds_analytic(
                self.gates,
                num_modals=num_modals,
                num_cls=num_cls,
                n_conv_per_stage=n_conv_per_stage,
                n_conv_per_stage_decoder=n_conv_per_stage_decoder,
                cost_fn=self._exact_cost,
            )

        # Held as a buffer, and only ever updated in place. torch.compile is
        # configured with mode="reduce-overhead" (CUDA graphs) elsewhere in this
        # codebase, which captures tensor addresses; rebinding this attribute to
        # a fresh tensor would either force a recompile per budget or replay a
        # stale graph.
        self.register_buffer(
            "_budget", torch.tensor(float(budget_max)), persistent=False
        )

        self.is_training = False

        # Recorded for HF export / reconstruction (AbstractModel.get_hf_config).
        self._mimose_model_kwargs = {
            "num_cls": num_cls,
            "num_modals": num_modals,
            "input_shape": tuple(int(d) for d in input_shape),
            "features_per_stage": self.max_features,
            "n_conv_per_stage": n_conv_per_stage,
            "n_conv_per_stage_decoder": n_conv_per_stage_decoder,
            "group_size": group_size,
            "tau": tau,
            "budget_metric": budget_metric,
            "budget_min": budget_min,
            "budget_max": budget_max,
            "calibrate": calibrate,
            "overlap": overlap,
        }

    # ------------------------------------------------------------------
    # budget plumbing
    # ------------------------------------------------------------------

    @property
    def budget(self) -> float:
        return float(self._budget)

    @property
    def tile_shape(self) -> tuple[int, ...]:
        return self.supernet.tile_shape

    def set_budget(self, budget: float) -> None:
        """Select the subnetwork for ``budget``.

        The trainer/testing call sites pass only ``(images, mask)``, so the
        budget travels on the module -- mirroring how ``is_training`` is already
        threaded through this codebase.
        """
        if budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}")
        self._budget.fill_(float(budget))

    def set_budget_ceiling(self, budget: float, iters: int = 60) -> float:
        """Select the largest subnetwork whose exact parameter count is <= ``budget``.

        ``set_budget(budget)`` can realize a network slightly over budget when a
        gate group sits at the edge of the budget (the penalty is one-sided and
        soft). Widths are monotone non-decreasing in the budget, so bisect the
        selection budget in (0, budget] for the largest value whose realized
        network fits. Returns the selection budget used (== ``budget`` when the
        plain selection already fits).
        """
        self.set_budget(budget)
        if self.actual_parameters() <= budget:
            return float(budget)
        lo, hi = float(budget) * 1e-3, float(budget)  # lo: far below budget_min -> narrowest widths
        self.set_budget(lo)
        if self.actual_parameters() > budget:
            raise ValueError(f"no subnetwork fits within budget {budget:.0f}")
        for _ in range(iters):
            mid = (lo * hi) ** 0.5
            self.set_budget(mid)
            if self.actual_parameters() <= budget:
                lo = mid
            else:
                hi = mid
        self.set_budget(lo)
        return lo

    def _current_z(self) -> Tensor:
        return budget_to_z(self._budget, self.gates.budget_min, self.gates.budget_max)

    def current_widths(self) -> tuple[int, ...]:
        """Discrete per-stage widths selected by the current budget."""
        widths = self.gates.hard_widths(self._current_z())
        return tuple(int(w) for w in widths.reshape(-1))

    def expected_parameters(self) -> Tensor:
        """Differentiable parameter count of the current subnetwork.

        Uses the *soft* widths so the budget penalty has a gradient with respect
        to the gate thresholds.
        """
        return expected_parameters(
            self.gates.soft_widths(self._current_z()),
            num_modals=self.num_modals,
            num_cls=self.num_cls,
            n_conv_per_stage=self.n_conv_per_stage,
            n_conv_per_stage_decoder=self.n_conv_per_stage_decoder,
        ).reshape(())

    def actual_parameters(self) -> int:
        """Exact parameter count of the architecture currently selected."""
        return exact_parameters(
            self.current_widths(),
            num_modals=self.num_modals,
            num_cls=self.num_cls,
            n_conv_per_stage=self.n_conv_per_stage,
            n_conv_per_stage_decoder=self.n_conv_per_stage_decoder,
        )

    def expected_macs(self) -> Tensor:
        """Differentiable MAC count of the current subnetwork."""
        return expected_macs(
            self.gates.soft_widths(self._current_z()),
            input_shape=self._input_shape,
            num_modals=self.num_modals,
            num_cls=self.num_cls,
            n_conv_per_stage=self.n_conv_per_stage,
            n_conv_per_stage_decoder=self.n_conv_per_stage_decoder,
        ).reshape(())

    def expected_cost(self) -> Tensor:
        """Differentiable cost in whatever unit ``budget_metric`` selects.

        This is what the budget penalty must use -- comparing a parameter count
        against a FLOPs budget would not error, it would just silently train the
        wrong architecture family.
        """
        if self.budget_metric == "flops":
            return self.expected_macs()
        return self.expected_parameters()

    def actual_cost(self) -> float:
        """Exact cost of the selected architecture, in the budget's own unit."""
        return self._exact_cost(self.current_widths())

    # ------------------------------------------------------------------
    # gated forward
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_gate(x: Tensor, gate: Tensor) -> Tensor:
        """Zero the inactive channels of ``x``.

        Returns a new tensor rather than mutating: the blocks are built with
        ``nonlin_kwargs={"inplace": True}``, so an in-place multiply here would
        write through an activation the autograd graph still needs.  The dtype
        cast keeps autocast from silently promoting the whole activation to the
        gate's fp32.
        """
        return x * gate.to(x.dtype).reshape(1, -1, 1, 1, 1)

    def forward(
        self,
        images: Tensor,
        mask: Tensor,
    ) -> Tensor | tuple[Tensor, tuple[()], tuple[()]]:
        backbone = self.supernet.backbone
        z = self._current_z()
        channel_gates = self.gates.channel_gates(z, hard=True)
        channel_gates = [g.reshape(-1) for g in channel_gates]
        hidden_gate = self.gates.hidden_gate(z, hard=True).reshape(-1)
        n_stages = len(self.max_features)

        # Zero missing modalities before the joint encoder, exactly as
        # TinyMimosa does -- a single encoder over all channels would otherwise
        # see straight through the "missing" constraint.
        gate = mask.to(images.dtype).view(images.size(0), images.size(1), 1, 1, 1)
        x = images * gate

        # ---- encoder ---------------------------------------------------
        skips: list[Tensor] = []
        for stage_index, stage in enumerate(backbone.encoder.stages):
            stage_gate = channel_gates[stage_index]
            # stage is nn.Sequential(StackedConvBlocks); gate after every block.
            for block in stage[0].convs:
                x = self._apply_gate(block(x), stage_gate)
            skips.append(x)

        # ---- mask conditioning on the bottleneck ------------------------
        skips[-1] = self._conditioned_bottleneck(
            skips[-1], mask, channel_gates[-1], hidden_gate
        )

        # ---- decoder ---------------------------------------------------
        decoder = backbone.decoder
        lres_input = skips[-1]
        for s in range(len(decoder.stages)):
            # Decoder stage s upsamples to resolution level (n_stages - 2 - s),
            # since UNetDecoder is built bottleneck-first.
            level = n_stages - 2 - s
            level_gate = channel_gates[level]

            # transpconv carries a bias, so its dead channels are non-zero and
            # must be gated before entering the concat.
            up = self._apply_gate(decoder.transpconvs[s](lres_input), level_gate)
            x = torch.cat((up, skips[-(s + 2)]), 1)
            for block in decoder.stages[s].convs:
                x = self._apply_gate(block(x), level_gate)
            lres_input = x

        fuse_logits = decoder.seg_layers[-1](lres_input)
        fuse_pred = torch.softmax(fuse_logits, dim=1)
        if self.is_training:
            return fuse_pred, (), ()
        return fuse_pred

    def _conditioned_bottleneck(
        self,
        latent: Tensor,
        mask: Tensor,
        bottleneck_gate: Tensor,
        hidden_gate: Tensor,
    ) -> Tensor:
        """FiLM-condition the bottleneck on modality availability.

        Both the hidden activation and the output are gated.  The hidden units
        need it because ``GELU`` of a dead unit's bias is non-zero; the output
        needs it because the conditioning ends in ``+ bias``, which would
        otherwise resurrect bottleneck channels the budget switched off.
        """
        conditioning = self.supernet.mask_conditioning
        hidden = conditioning.mlp[0](mask.float())
        hidden = conditioning.mlp[1](hidden) * hidden_gate.to(hidden.dtype)
        out = conditioning.mlp[2](hidden)

        scale, bias = out.chunk(2, dim=1)
        gate = bottleneck_gate.to(out.dtype)
        scale = scale * gate
        bias = self.supernet.mask_conditioning.bias_limit * torch.tanh(bias) * gate

        scale = scale[..., None, None, None]
        bias = bias[..., None, None, None]
        return latent * (1.0 + torch.tanh(scale)) + bias

    # ------------------------------------------------------------------
    # inference / export
    # ------------------------------------------------------------------

    def predict(self, images: Tensor, mask: Tensor) -> Tensor:
        """Sliding-window inference for the currently selected subnetwork.

        Delegates to ``SimpleUnet.predict``'s tiling logic while forcing each
        tile through this class's gated forward.
        """
        was_training = self.is_training
        self.is_training = False
        try:
            return SimpleUnet.predict(self, images, mask)
        finally:
            self.is_training = was_training

    # SimpleUnet.predict reaches for these while tiling.
    @property
    def overlap(self) -> float:
        return self.supernet.overlap

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.supernet.input_shape

    def _validate_boundary_inputs(self, images: Tensor, mask: Tensor) -> None:
        self.supernet._validate_boundary_inputs(images, mask)

    @staticmethod
    def _window_starts(size: int, window: int, overlap: float) -> list[int]:
        return SimpleUnet._window_starts(size, window, overlap)

    @staticmethod
    def _crop_to_original_shape(prediction, pad_before, original_shape):
        return SimpleUnet._crop_to_original_shape(
            prediction, pad_before, original_shape
        )

    def model_for_export(self) -> AbstractModel:
        """Persist the *supernet*, not the subnet for whatever budget is current.

        The whole point of this model is the family it represents: exporting a
        single extracted subnet here would silently reduce
        ``final_weights_only.safetensors`` to one arbitrary operating point --
        whichever budget the last training step happened to sample -- and the
        dense Pareto sweep would have nothing left to extract from.

        Deployable single-budget artifacts are produced explicitly instead, via
        :meth:`extract_for_budget` / :func:`mimose.nas.extract.extract_subnet`.
        """
        return self

    def extract_for_budget(self, budget: float | None = None) -> AbstractModel:
        """Materialize the standalone ``SimpleUnet`` for ``budget``.

        Defaults to the currently selected budget.  This is the deployable
        artifact; :meth:`model_for_export` deliberately keeps the family.
        """
        from mimose.nas.extract import extract_subnet

        return extract_subnet(self, self.budget if budget is None else budget)


Model = ElasticUnet
