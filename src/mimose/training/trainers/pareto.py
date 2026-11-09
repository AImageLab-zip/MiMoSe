"""Phase 4: multi-budget supernet training.

One set of shared weights is trained to serve a whole range of parameter
budgets.  Each optimizer step samples several budgets, evaluates the
corresponding subnetworks on the *same* batch, and accumulates their gradients
before stepping once.  The objective for a budget ``b`` is

    L(b) = L_seg(A(b)) + lambda_param * ReLU((P_expected(A(b)) - b) / b)^2

The penalty is one-sided on purpose: a subnetwork that comes in *under* budget is
not a problem, so only overshoot is punished.  ``P_expected`` is computed from
the *soft* gate widths, so it is the term that actually trains the thresholds --
the segmentation loss reaches them only through the straight-through estimator.

Budget sampling is ``local`` by default (see :class:`BudgetSampler`): a centre
budget is drawn log-uniformly and its companions are drawn from a narrow window
around it, so a single update never mixes a 1M-parameter subnetwork with a 1.5B
one.  The hypothesis behind that -- nearby subnetworks share more useful
gradient than distant ones -- is exactly what Phase 5 measures, so ``global`` and
``sandwich`` strategies are available for the comparison.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from mimose.losses.tiny_mimosa import soft_target_loss
from mimose.models.elastic_unet import ElasticUnet
from mimose.training.masking import drop_one_available
from mimose.training.trainers.imfuse import IMFuseTrainer

__all__ = ["ParetoTrainer"]

DEFAULT_BUDGETS_PER_STEP = 3
DEFAULT_LOCAL_WINDOW_RATIO = 1.5
DEFAULT_LAMBDA_PARAM = 1.0
DEFAULT_TAU_START = 0.5
DEFAULT_TAU_END = 0.02
DEFAULT_DISTILL_ALPHA = 0.5


class ParetoTrainer(IMFuseTrainer):
    """Trains an :class:`~mimose.models.elastic_unet.ElasticUnet` across budgets.

    Recognized ``custom_trainer_kwargs`` (on top of ``IMFuseTrainer``'s):

    ``budgets_per_step``
        Subnetworks evaluated per optimizer step.  Cost scales linearly.
    ``budget_sampling``
        ``"local"`` (default), ``"global"`` or ``"sandwich"``.
    ``local_window_ratio``
        Half-width of the local window as a *ratio*.  1.5 reproduces the
        20M/30M/45M grouping around a 30M centre.
    ``lambda_param``
        Weight of the budget-overshoot penalty.
    ``tau_start`` / ``tau_end``
        Gate temperature, annealed geometrically across training.
    ``val_budgets``
        Budgets to validate at.  Defaults to the range's bottom, geometric
        middle and top, so the reported Dice covers the whole family rather
        than one operating point.
    ``gate_lr_scale``
        Learning-rate multiplier for the gate parameters (see
        :meth:`_build_optimizer`).
    ``inplace_distill``
        BigNAS-style inplace distillation.  The widest subnetwork in the step is
        evaluated first and its (detached) output becomes a soft target for the
        narrower ones, blended with the ground-truth loss by ``distill_alpha``.
        Off by default.
    ``distill_alpha``
        Blend between ground truth and teacher for the narrower subnetworks:
        ``(1 - alpha) * L_task + alpha * L_distill``.  ``1.0`` reproduces BigNAS
        (sub-networks never see the label); the 0.5 default keeps half the
        gradient on the true label, which matters more here than in
        classification because the tumor classes are severely imbalanced and a
        teacher that misses a small lesion would otherwise teach the student to
        miss it too.  Ignored when ``inplace_distill`` is false.
    ``student_modality_drop``
        Every subnetwork but the widest sees the batch mask with one more
        available modality switched off (never all of them); the widest keeps
        the original mask.  With ``inplace_distill`` this distils from more
        modalities and more width into fewer of both.  Off by default.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        trainer_kwargs = dict(kwargs.get("custom_trainer_kwargs") or {})

        self.budgets_per_step = int(
            trainer_kwargs.get("budgets_per_step", DEFAULT_BUDGETS_PER_STEP)
        )
        if self.budgets_per_step < 1:
            raise ValueError("budgets_per_step must be at least 1")

        self.budget_sampling = str(trainer_kwargs.get("budget_sampling", "local"))
        if self.budget_sampling not in {"local", "global", "sandwich"}:
            raise ValueError(
                "budget_sampling must be one of 'local', 'global', 'sandwich'; "
                f"got {self.budget_sampling!r}"
            )

        self.local_window_ratio = float(
            trainer_kwargs.get("local_window_ratio", DEFAULT_LOCAL_WINDOW_RATIO)
        )
        if self.local_window_ratio < 1.0:
            raise ValueError("local_window_ratio must be >= 1")

        self.lambda_param = float(
            trainer_kwargs.get("lambda_param", DEFAULT_LAMBDA_PARAM)
        )
        self.tau_start = float(trainer_kwargs.get("tau_start", DEFAULT_TAU_START))
        self.tau_end = float(trainer_kwargs.get("tau_end", DEFAULT_TAU_END))
        self.gate_lr_scale = float(trainer_kwargs.get("gate_lr_scale", 1.0))

        self.inplace_distill = bool(trainer_kwargs.get("inplace_distill", False))
        self.distill_alpha = float(
            trainer_kwargs.get("distill_alpha", DEFAULT_DISTILL_ALPHA)
        )
        if not 0.0 <= self.distill_alpha <= 1.0:
            raise ValueError(
                f"distill_alpha must be in [0, 1], got {self.distill_alpha}"
            )
        if self.inplace_distill and self.budgets_per_step < 2:
            raise ValueError(
                "inplace_distill needs budgets_per_step >= 2: with a single "
                "subnetwork per step there is no teacher to distil from"
            )
        self.student_modality_drop = bool(
            trainer_kwargs.get("student_modality_drop", False)
        )
        if self.student_modality_drop and self.budgets_per_step < 2:
            raise ValueError(
                "student_modality_drop needs budgets_per_step >= 2: the widest "
                "subnetwork keeps the original mask, so there is no student"
            )
        self._configured_val_budgets = trainer_kwargs.get("val_budgets")

        super().__init__(*args, **kwargs)

        supernet = self._supernet()
        if self.budgets_per_step < 2 and self.budget_sampling == "sandwich":
            raise ValueError("sandwich sampling needs budgets_per_step >= 2")

        self.budget_min = float(supernet.gates.budget_min)
        self.budget_max = float(supernet.gates.budget_max)
        self.val_budgets = self._resolve_val_budgets()

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def _supernet(self) -> ElasticUnet:
        model = self._model_for_state()
        if not isinstance(model, ElasticUnet):
            raise RuntimeError(
                "ParetoTrainer requires model=elasticunet, got "
                f"{type(model).__name__}"
            )
        return model

    def _resolve_val_budgets(self) -> list[float]:
        if self._configured_val_budgets:
            return [float(b) for b in self._configured_val_budgets]
        return [
            self.budget_min,
            math.sqrt(self.budget_min * self.budget_max),
            self.budget_max,
        ]

    def _build_optimizer(
        self,
        model: torch.nn.Module | None = None,
    ) -> torch.optim.Optimizer:
        """Give the gate thresholds their own parameter group.

        The cumulative threshold parameterization means a stage's base threshold
        appears in every one of its groups, so its gradient is the sum over all
        groups and it moves roughly ``G`` times faster than the last increment.
        Splitting the groups lets ``gate_lr_scale`` compensate, and keeps weight
        decay off the thresholds -- they are positions on a budget axis, not
        weights, and shrinking them toward zero is meaningless.
        """
        optimizer = super()._build_optimizer(model)
        if self.gate_lr_scale == 1.0:
            # Still split, so weight decay never touches the thresholds.
            pass

        target = model or self._model_for_state()
        gate_params = {id(p) for p in target.gates.parameters()}
        weights = [p for p in target.parameters() if id(p) not in gate_params]
        gates = list(target.gates.parameters())

        base_lr = optimizer.param_groups[0]["lr"]
        defaults = dict(optimizer.param_groups[0])
        defaults.pop("params", None)

        gate_defaults = dict(defaults)
        gate_defaults["lr"] = base_lr * self.gate_lr_scale
        gate_defaults["weight_decay"] = 0.0

        optimizer.param_groups = []
        optimizer.add_param_group({"params": weights, **defaults})
        optimizer.add_param_group({"params": gates, **gate_defaults})
        self.optimizer = optimizer
        return optimizer

    # ------------------------------------------------------------------
    # budget sampling
    # ------------------------------------------------------------------

    def _sample_budgets(self) -> list[float]:
        """Draw this step's budgets, identically on every rank.

        Under DDP the sampled budgets *must* agree across ranks: gradients are
        averaged parameter-wise, so if rank 0 trained a 5M subnet while rank 1
        trained a 50M one, each would be updated by the other's architecture.
        Rank 0 draws and broadcasts rather than relying on synchronized RNG,
        which silently diverges the moment any rank consumes a random number the
        others do not.
        """
        count = self.budgets_per_step
        if self.is_main_process or not self.distributed:
            budgets = self._draw_budgets(count)
        else:
            budgets = [0.0] * count

        if self.distributed and dist.is_initialized():
            payload = torch.tensor(budgets, dtype=torch.float64, device=self.device)
            dist.broadcast(payload, src=0)
            budgets = [float(v) for v in payload.tolist()]
        return budgets

    def _draw_budgets(self, count: int) -> list[float]:
        log_min = math.log(self.budget_min)
        log_max = math.log(self.budget_max)

        if self.budget_sampling == "global":
            draws = torch.rand(count)
            return [float(math.exp(log_min + (log_max - log_min) * u)) for u in draws]

        if self.budget_sampling == "sandwich":
            # BigNAS-style: always include both extremes so the ends of the
            # Pareto front keep receiving gradient.
            budgets = [self.budget_min, self.budget_max]
            for _ in range(count - 2):
                u = float(torch.rand(()))
                budgets.append(math.exp(log_min + (log_max - log_min) * u))
            return budgets[:count]

        if self.budget_sampling != "local":
            # Defence in depth: the constructor validates this, but falling
            # through to "local" for an unrecognized value would quietly train a
            # different curriculum than the config asked for.
            raise ValueError(f"unknown budget_sampling: {self.budget_sampling!r}")

        # local: a centre drawn log-uniformly, companions spread across a narrow
        # window around it, clamped to stay inside the supernet's range.
        centre_u = float(torch.rand(()))
        centre_log = log_min + (log_max - log_min) * centre_u
        half_width = math.log(self.local_window_ratio)

        if count == 1:
            offsets = [0.0]
        else:
            step = 2.0 * half_width / (count - 1)
            offsets = [-half_width + step * i for i in range(count)]

        return [
            float(math.exp(min(max(centre_log + offset, log_min), log_max)))
            for offset in offsets
        ]

    def _anneal_tau(self, epoch: int) -> float:
        """Geometric anneal from ``tau_start`` to ``tau_end``.

        Only gradients are affected: the forward pass is already binary through
        the straight-through estimator, so lowering tau sharpens how sharply a
        threshold responds rather than changing which subnetwork runs.
        """
        if self.num_epochs <= 1:
            tau = self.tau_end
        else:
            progress = min(max(epoch / (self.num_epochs - 1), 0.0), 1.0)
            tau = self.tau_start * (self.tau_end / self.tau_start) ** progress
        self._supernet().gates.set_tau(tau)
        return tau

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Own copy of ``IMFuseTrainer.train_epoch``'s loop.

        Cannot delegate to the base implementation: it pre-populates ``totals``
        with a fixed key set (``loss``, ``fusecross``, ...) and does
        ``totals[key] += value`` for every key ``_train_step`` returns, which
        ``KeyError``s the moment a step returns anything not in that set --
        exactly what ``budget_penalty``/``expected_params``/etc. do here.
        """
        if self.train_loader is None:
            raise RuntimeError("train_loader must be initialized before training")

        self._anneal_tau(epoch)

        self.model.train()
        self._set_aux_training_flag(True)
        steps = (
            self.iter_per_epoch
            if self.iter_per_epoch is not None
            else len(self.train_loader)
        )
        totals: dict[str, float] = {}

        iterations = 0
        with self._progress(disable=not self.is_main_process) as progress:
            task_id = progress.add_task(
                f"Train {epoch + 1}/{self.num_epochs}",
                total=steps,
                metrics="",
                vram=self._vram_text(),
            )
            for _ in range(steps):
                batch = self._next_train_batch()
                metrics = self._train_step(batch, epoch)
                iterations += 1
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + value
                progress.update(
                    task_id,
                    advance=1,
                    vram=self._vram_text(),
                    metrics=(
                        f"loss {metrics['loss']:.4f}  "
                        f"avg {totals['loss'] / iterations:.4f}  "
                        f"tau {metrics['tau']:.3g}"
                    ),
                )

                if self.debug:
                    break

        return {key: value / max(iterations, 1) for key, value in totals.items()}

    def _train_step(self, batch: dict[str, Any], epoch: int) -> dict[str, float]:
        """One optimizer step spanning several budgets on the same batch.

        ``BaseTrainer._backward_step`` cannot be reused here: it zeroes
        gradients *before* each backward, so calling it per budget would discard
        every subnetwork's contribution except the last.
        """
        images = batch["images"].to(self.device, non_blocking=True)
        seg = batch["seg"].to(self.device, non_blocking=True).long()
        mask = batch["mask"].to(self.device, non_blocking=True).bool()

        supernet = self._supernet()
        budgets = self._sample_budgets()
        include_fuse = epoch >= self.region_fusion_start_epoch

        self.optimizer.zero_grad(set_to_none=True)

        totals = {
            "loss": 0.0,
            "seg_loss": 0.0,
            "budget_penalty": 0.0,
            "fusecross": 0.0,
            "fusedice": 0.0,
            "sepcross": 0.0,
            "sepdice": 0.0,
            "prmcross": 0.0,
            "prmdice": 0.0,
        }
        totals["distill_loss"] = 0.0
        overshoot_count = 0.0
        params_sum = 0.0

        if self.inplace_distill or self.student_modality_drop:
            # Widest first: its output is the teacher for everything after it,
            # so it has to be computed before the others rather than wherever
            # the sampler happened to place it (sandwich returns [min, max, ...]).
            budgets = sorted(budgets, reverse=True)
        teacher: torch.Tensor | None = None

        for index, budget in enumerate(budgets):
            supernet.set_budget(budget)
            is_last = index == len(budgets) - 1
            # Only the widest subnetwork learns from the label alone; it *is*
            # the teacher, so distilling it into itself would be a no-op at best.
            is_teacher = self.inplace_distill and index == 0

            step_mask = mask
            if self.student_modality_drop and index > 0:
                step_mask = drop_one_available(mask)

            with self._autocast_context():
                target = self._seg_to_one_hot(seg)
                outputs = self.model(images, step_mask)
                metrics = self._loss_impl().training_loss(
                    outputs, target, include_fuse=include_fuse
                )
                seg_loss = metrics["loss"]

                distill_loss = seg_loss.new_tensor(0.0)
                if self.inplace_distill and include_fuse:
                    if is_teacher:
                        # Detached: the teacher is a target, not a path for the
                        # students' gradient to flow back into the wide weights.
                        teacher = outputs[0].detach()
                    elif teacher is not None:
                        loss_impl = self._loss_impl()
                        distill_loss = soft_target_loss(
                            outputs[0],
                            teacher,
                            dice_weight=loss_impl.dice_weight,
                            ce_weight=loss_impl.ce_weight,
                            eps=loss_impl.eps,
                        )
                        seg_loss = (
                            1.0 - self.distill_alpha
                        ) * seg_loss + self.distill_alpha * distill_loss

                expected = supernet.expected_cost()
                relative_overshoot = torch.relu(
                    (expected - budget) / budget
                )
                penalty = self.lambda_param * relative_overshoot.pow(2)

                # Average over budgets so the gradient magnitude -- and hence the
                # meaning of the configured learning rate -- does not depend on
                # budgets_per_step.
                total = (seg_loss + penalty) / len(budgets)

            self._accumulate_backward(total, sync=is_last)

            # "loss" is what was actually optimized (blended, if distilling);
            # "seg_loss" stays the pure ground-truth term so the two runs remain
            # comparable to a non-distilling one.
            totals["loss"] += float(seg_loss.item()) + float(penalty.item())
            totals["seg_loss"] += float(metrics["loss"].item())
            totals["budget_penalty"] += float(penalty.item())
            totals["distill_loss"] += float(distill_loss.item())
            for key in (
                "fusecross",
                "fusedice",
                "sepcross",
                "sepdice",
                "prmcross",
                "prmdice",
            ):
                totals[key] += float(metrics[key].item())
            overshoot_count += float((relative_overshoot > 0).float().item())
            params_sum += float(expected.item())

        self._finish_optimizer_step()

        count = len(budgets)
        result = {key: value / count for key, value in totals.items()}
        result["budget_min_sampled"] = min(budgets)
        result["budget_max_sampled"] = max(budgets)
        result["expected_params"] = params_sum / count
        result["overshoot_fraction"] = overshoot_count / count
        result["tau"] = float(self._supernet().gates.tau)
        return result

    def _accumulate_backward(self, loss: torch.Tensor, *, sync: bool) -> None:
        """Backward one budget's loss, deferring DDP's all-reduce until the last.

        Without ``no_sync`` every budget triggers its own gradient all-reduce,
        which is both wasteful and wrong: DDP averages at each backward, so the
        earlier budgets would be reduced repeatedly.
        """
        should_defer = (
            not sync
            and self.distributed
            and isinstance(self.model, DistributedDataParallel)
        )
        if should_defer:
            with self.model.no_sync():
                self._scaled_backward(loss)
            return
        self._scaled_backward(loss)

    def _scaled_backward(self, loss: torch.Tensor) -> None:
        if self.use_grad_scaler:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

    def _finish_optimizer_step(self) -> None:
        """Clip and step once, after every budget has contributed."""
        if self.use_grad_scaler:
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=12.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            return

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=12.0)
        self.optimizer.step()

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def val_epoch(self, epoch: int) -> dict[str, float]:
        """Validate at several budgets and report both per-budget and mean Dice.

        ``BaseTrainer._is_best_checkpoint`` selects on ``dice``, so that key
        carries the mean across budgets -- checkpointing on a single operating
        point would let the supernet drift toward favouring one end of the range.
        """
        supernet = self._supernet()
        original_budget = supernet.budget

        aggregated: dict[str, float] = {}
        per_budget: list[dict[str, float]] = []
        try:
            for budget in self.val_budgets:
                supernet.set_budget(budget)
                widths = supernet.current_widths()
                metrics = super().val_epoch(epoch)
                per_budget.append(metrics)

                tag = f"b{budget / 1e6:.1f}M"
                for key, value in metrics.items():
                    aggregated[f"{tag}/{key}"] = value
                aggregated[f"{tag}/params"] = float(supernet.actual_parameters())
                if self.is_main_process:
                    self._log(
                        f"  budget {budget:,.0f} -> widths {widths} "
                        f"dice {metrics['dice']:.4f}"
                    )
        finally:
            supernet.set_budget(original_budget)

        # Mean across budgets for the scalar metrics the base trainer expects.
        for key in per_budget[0]:
            aggregated[key] = sum(m[key] for m in per_budget) / len(per_budget)
        return aggregated

    def _log(self, message: str) -> None:
        console = getattr(self, "console", None)
        if console is not None:
            console.print(message)
        else:  # pragma: no cover - only when rich console is unavailable
            print(message)

    def _wandb_config_payload(self) -> dict[str, Any]:
        payload = super()._wandb_config_payload()
        payload.update(
            {
                "budgets_per_step": self.budgets_per_step,
                "budget_sampling": self.budget_sampling,
                "local_window_ratio": self.local_window_ratio,
                "lambda_param": self.lambda_param,
                "tau_start": self.tau_start,
                "tau_end": self.tau_end,
                "gate_lr_scale": self.gate_lr_scale,
                "inplace_distill": self.inplace_distill,
                "distill_alpha": self.distill_alpha,
                "student_modality_drop": self.student_modality_drop,
                "budget_min": self.budget_min,
                "budget_max": self.budget_max,
                "val_budgets": list(self.val_budgets),
            }
        )
        return payload
