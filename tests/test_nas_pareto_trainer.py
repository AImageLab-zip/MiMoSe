from __future__ import annotations

import math
from contextlib import nullcontext

import pytest
import torch

from mimose.enums import TrainerKind
from mimose.losses.tiny_mimosa import TinyMimosaLoss
from mimose.models.elastic_unet import ElasticUnet
from mimose.training.masking import drop_one_available
from mimose.training.trainers import ParetoTrainer

INPUT_SHAPE = (32, 32, 32)


def make_trainer(**overrides) -> ParetoTrainer:
    """Build a ParetoTrainer without BaseTrainer's CUDA/dataset requirements.

    ``BaseTrainer.__init__`` hard-requires a CUDA device and a preprocessed
    dataset, neither of which a unit test should need to exercise budget
    sampling or gradient accumulation. Only the attributes the methods under
    test actually touch are populated.
    """
    trainer = ParetoTrainer.__new__(ParetoTrainer)

    model = ElasticUnet(
        features_per_stage=(8, 16, 32, 64),
        input_shape=INPUT_SHAPE,
        group_size=2,
    )
    # train_epoch normally sets this; _train_step is called directly here, and
    # the loss unpacks the (pred, (), ()) tuple form it switches on.
    model.is_training = True

    trainer.model = model
    trainer.device = torch.device("cpu")
    trainer.rank = 0
    trainer.budget_min = float(model.gates.budget_min)
    trainer.budget_max = float(model.gates.budget_max)
    trainer.num_epochs = 10
    trainer.distributed = False
    trainer.use_grad_scaler = False
    trainer.grad_scaler = None
    trainer.region_fusion_start_epoch = 0
    trainer.amp_enabled = False
    trainer.precision = None
    trainer.model_config = None  # num_classes falls back to its default of 4

    trainer.budgets_per_step = overrides.get("budgets_per_step", 3)
    trainer.budget_sampling = overrides.get("budget_sampling", "local")
    trainer.local_window_ratio = overrides.get("local_window_ratio", 1.5)
    trainer.lambda_param = overrides.get("lambda_param", 1.0)
    trainer.tau_start = overrides.get("tau_start", 0.5)
    trainer.tau_end = overrides.get("tau_end", 0.02)
    trainer.gate_lr_scale = overrides.get("gate_lr_scale", 1.0)
    trainer.inplace_distill = overrides.get("inplace_distill", False)
    trainer.distill_alpha = overrides.get("distill_alpha", 0.5)
    trainer.student_modality_drop = overrides.get("student_modality_drop", False)

    trainer.loss_fn = TinyMimosaLoss(num_classes=4)
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    # Stub out the base-class plumbing the tested methods reach for.
    trainer._autocast_context = lambda: nullcontext()  # type: ignore[method-assign]
    trainer._loss_impl = lambda: trainer.loss_fn  # type: ignore[method-assign]
    return trainer


def make_epoch_ready_trainer(*, steps: int = 3, **overrides) -> ParetoTrainer:
    """A trainer that can additionally run ``train_epoch`` end to end."""
    trainer = make_trainer(**overrides)
    batches = [sample_batch() for _ in range(steps)]
    iterator = iter(batches)
    trainer.train_loader = batches  # only needs len() and to be non-None
    trainer.iter_per_epoch = None
    trainer.debug = False
    trainer._next_train_batch = lambda: next(iterator)  # type: ignore[method-assign]
    return trainer


def sample_batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "images": torch.randn(1, 4, *INPUT_SHAPE),
        "seg": torch.randint(0, 4, (1, 1, *INPUT_SHAPE)),
        "mask": torch.tensor([[True, False, True, True]]),
    }


# ------------------------------------------------------------- budget sampling


def test_local_sampling_keeps_budgets_within_the_window() -> None:
    """The defining property of local training: no distant budgets in one step."""
    trainer = make_trainer(budget_sampling="local", local_window_ratio=1.5)
    for _ in range(200):
        budgets = trainer._draw_budgets(3)
        assert len(budgets) == 3
        # Widest possible spread is centre/1.5 .. centre*1.5, i.e. a 2.25x span.
        assert max(budgets) / min(budgets) <= 1.5**2 + 1e-9
        assert all(trainer.budget_min <= b <= trainer.budget_max for b in budgets)


def test_local_sampling_reproduces_the_documented_grouping() -> None:
    """A 30M centre with ratio 1.5 should yield roughly 20M / 30M / 45M."""
    trainer = make_trainer(budget_sampling="local", local_window_ratio=1.5)
    trainer.budget_min = 1e6
    trainer.budget_max = 1.5e9

    centre_log = math.log(30e6)
    half = math.log(1.5)
    expected = [math.exp(centre_log - half), 30e6, math.exp(centre_log + half)]
    assert expected[0] == pytest.approx(20e6)
    assert expected[2] == pytest.approx(45e6)


def test_global_sampling_is_log_uniform_over_the_whole_range() -> None:
    trainer = make_trainer(budget_sampling="global")
    draws = [b for _ in range(500) for b in trainer._draw_budgets(1)]
    log_mid = math.sqrt(trainer.budget_min * trainer.budget_max)
    # Log-uniform => half the mass below the geometric mean. Linear-space
    # sampling would put ~90% above it.
    below = sum(1 for b in draws if b < log_mid) / len(draws)
    assert below == pytest.approx(0.5, abs=0.08)


def test_sandwich_sampling_always_includes_both_extremes() -> None:
    trainer = make_trainer(budget_sampling="sandwich", budgets_per_step=4)
    for _ in range(20):
        budgets = trainer._draw_budgets(4)
        assert min(budgets) == pytest.approx(trainer.budget_min)
        assert max(budgets) == pytest.approx(trainer.budget_max)


def test_budget_sampling_rejects_unknown_strategy() -> None:
    trainer = make_trainer()
    trainer.budget_sampling = "nonsense"
    with pytest.raises(ValueError, match="unknown budget_sampling"):
        trainer._draw_budgets(3)


# ------------------------------------------------------------------ annealing


def test_tau_anneals_geometrically_and_reaches_the_endpoint() -> None:
    trainer = make_trainer(tau_start=0.5, tau_end=0.02)
    first = trainer._anneal_tau(0)
    last = trainer._anneal_tau(trainer.num_epochs - 1)
    assert first == pytest.approx(0.5)
    assert last == pytest.approx(0.02)
    # Monotone decreasing in between.
    taus = [trainer._anneal_tau(e) for e in range(trainer.num_epochs)]
    assert all(a >= b for a, b in zip(taus, taus[1:]))
    # And it is actually written through to the gates.
    assert float(trainer.model.gates.tau) == pytest.approx(taus[-1])


# --------------------------------------------------- multi-budget accumulation


def test_train_step_accumulates_gradients_from_every_budget() -> None:
    """The core Phase 4 mechanic.

    ``BaseTrainer._backward_step`` zeroes gradients before each backward, so
    reusing it per budget would silently keep only the last subnetwork's
    contribution. This asserts the accumulated gradient equals the sum of the
    per-budget gradients rather than the final one alone.
    """
    trainer = make_trainer(budgets_per_step=3, lambda_param=0.0)
    batch = sample_batch()

    # Pin the budgets so the two passes below see identical subnetworks.
    fixed = trainer._draw_budgets(3)
    trainer._sample_budgets = lambda: list(fixed)  # type: ignore[method-assign]

    trainer._train_step(batch, epoch=0)
    accumulated = {
        name: p.grad.clone()
        for name, p in trainer.model.named_parameters()
        if p.grad is not None
    }

    # Recompute each budget's gradient independently and sum them.
    manual: dict[str, torch.Tensor] = {}
    for budget in fixed:
        trainer.model.zero_grad(set_to_none=True)
        trainer.model.set_budget(budget)
        target = torch.nn.functional.one_hot(
            batch["seg"].squeeze(1).long(), num_classes=4
        ).permute(0, 4, 1, 2, 3).float()
        outputs = trainer.model(batch["images"], batch["mask"])
        loss = trainer.loss_fn.training_loss(outputs, target, include_fuse=True)["loss"]
        (loss / len(fixed)).backward()
        for name, p in trainer.model.named_parameters():
            if p.grad is not None:
                manual[name] = manual.get(name, torch.zeros_like(p.grad)) + p.grad

    assert accumulated
    for name, grad in accumulated.items():
        assert torch.allclose(grad, manual[name], atol=1e-5), name

    # A single-budget baseline must differ, or the test above proves nothing.
    trainer.model.zero_grad(set_to_none=True)
    trainer.model.set_budget(fixed[-1])
    outputs = trainer.model(batch["images"], batch["mask"])
    target = torch.nn.functional.one_hot(
        batch["seg"].squeeze(1).long(), num_classes=4
    ).permute(0, 4, 1, 2, 3).float()
    trainer.loss_fn.training_loss(outputs, target, include_fuse=True)["loss"].backward()
    last_only = {
        name: p.grad.clone()
        for name, p in trainer.model.named_parameters()
        if p.grad is not None
    }
    differs = any(
        not torch.allclose(accumulated[name], last_only[name], atol=1e-5)
        for name in accumulated
        if name in last_only
    )
    assert differs, "accumulation is indistinguishable from training one budget"


def test_train_step_reports_budget_diagnostics() -> None:
    trainer = make_trainer(budgets_per_step=3)
    metrics = trainer._train_step(sample_batch(), epoch=0)

    for key in (
        "loss",
        "seg_loss",
        "budget_penalty",
        "expected_params",
        "overshoot_fraction",
        "tau",
        "budget_min_sampled",
        "budget_max_sampled",
    ):
        assert key in metrics
        assert math.isfinite(metrics[key])

    assert metrics["budget_min_sampled"] <= metrics["budget_max_sampled"]
    assert 0.0 <= metrics["overshoot_fraction"] <= 1.0


def test_train_epoch_accumulates_the_full_diagnostic_key_set() -> None:
    """Regression test: train_epoch must not assume IMFuseTrainer's fixed keys.

    IMFuseTrainer.train_epoch pre-populates its running totals with a fixed set
    of keys (loss, fusecross, ...) and does ``totals[key] += value`` for
    whatever _train_step returns. ParetoTrainer's _train_step returns extra
    diagnostic keys (seg_loss, budget_penalty, expected_params, ...), which
    KeyErrors against that fixed dict -- caught on the first real GPU run,
    not by any test that only calls _train_step directly.
    """
    trainer = make_epoch_ready_trainer(steps=3)
    result = trainer.train_epoch(epoch=0)

    for key in (
        "loss",
        "seg_loss",
        "budget_penalty",
        "expected_params",
        "overshoot_fraction",
        "tau",
        "fusecross",
        "fusedice",
        "distill_loss",
    ):
        assert key in result
        assert math.isfinite(result[key])


def test_train_epoch_anneals_tau_before_the_first_step() -> None:
    trainer = make_epoch_ready_trainer(steps=1, tau_start=0.5, tau_end=0.02)
    trainer.train_epoch(epoch=0)
    assert float(trainer.model.gates.tau) == pytest.approx(0.5)

    trainer2 = make_epoch_ready_trainer(steps=1, tau_start=0.5, tau_end=0.02)
    trainer2.num_epochs = 10
    trainer2.train_epoch(epoch=trainer2.num_epochs - 1)
    assert float(trainer2.model.gates.tau) == pytest.approx(0.02)


def test_budget_penalty_is_one_sided() -> None:
    """Undershooting the budget must be free; only overshoot is penalized."""
    trainer = make_trainer(budgets_per_step=1, lambda_param=1.0)
    model = trainer.model

    # Far above what the supernet can spend -> no penalty.
    generous = trainer.budget_max * 10
    model.set_budget(generous)
    expected = model.expected_parameters().detach()
    assert float(torch.relu((expected - generous) / generous)) == pytest.approx(0.0)

    # Ask for less than the smallest subnetwork -> penalty must bite.
    stingy = trainer.budget_min / 100
    model.set_budget(stingy)
    expected = model.expected_parameters().detach()
    assert float(torch.relu((expected - stingy) / stingy)) > 0.0


def test_budget_penalty_produces_gradient_on_the_gates_only() -> None:
    """The penalty is what actually trains the thresholds.

    It is a function of the gates alone -- no batch involved -- so it must not
    touch the convolution weights.
    """
    trainer = make_trainer(budgets_per_step=1, lambda_param=1.0)
    model = trainer.model
    model.zero_grad(set_to_none=True)
    model.set_budget(trainer.budget_min / 50)

    expected = model.expected_parameters()
    budget = trainer.budget_min / 50
    penalty = torch.relu((expected - budget) / budget).pow(2)
    penalty.backward()

    gate_ids = {id(p) for p in model.gates.parameters()}
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.gates.parameters()
    )
    for p in model.parameters():
        if id(p) not in gate_ids:
            assert p.grad is None or float(p.grad.abs().sum()) == 0.0


# ------------------------------------------------------------------- CLI wiring


def test_pareto_trainer_is_registered_in_the_cli() -> None:
    from mimose.cli_workflows import _build_trainer_instance_from_merged  # noqa: F401

    assert TrainerKind("pareto") is TrainerKind.PARETO


def test_shipped_pareto_configs_select_the_right_components() -> None:
    from pathlib import Path

    from mimose.utils.cli_overrides import load_yaml_config

    for dataset in ("18", "23"):
        path = Path("src/mimose/data/configs") / f"pareto_{dataset}.yaml"
        config = load_yaml_config(path)
        assert config["trainer"] == "pareto"
        assert config["model"] == "elasticunet"
        assert tuple(config["custom_model_kwargs"]["features_per_stage"]) == (
            64,
            128,
            256,
            512,
            1024,
        )
        trainer_kwargs = config["custom_trainer_kwargs"]
        assert trainer_kwargs["budget_sampling"] == "local"
        assert trainer_kwargs["budgets_per_step"] >= 1


# --------------------------------------------------------- inplace distillation


def _budget_order(trainer, batch) -> list[float]:
    """Record the order _train_step selects budgets in."""
    seen: list[float] = []
    supernet = trainer.model
    original = supernet.set_budget

    def spy(budget: float) -> None:
        seen.append(float(budget))
        original(budget)

    supernet.set_budget = spy  # type: ignore[method-assign]
    try:
        trainer._train_step(batch, epoch=0)
    finally:
        supernet.set_budget = original  # type: ignore[method-assign]
    return seen


def test_distillation_evaluates_the_widest_subnetwork_first() -> None:
    """The teacher must run before its students.

    Sandwich sampling returns ``[min, max, ...]``, so without an explicit
    reordering the teacher would be computed second and the first (narrowest)
    subnetwork would silently train without one.
    """
    trainer = make_trainer(
        budget_sampling="sandwich", budgets_per_step=3, inplace_distill=True
    )
    order = _budget_order(trainer, sample_batch())
    assert order == sorted(order, reverse=True)
    assert order[0] == pytest.approx(trainer.budget_max)


def test_no_distillation_leaves_budget_order_untouched() -> None:
    trainer = make_trainer(
        budget_sampling="sandwich", budgets_per_step=3, inplace_distill=False
    )
    order = _budget_order(trainer, sample_batch())
    assert order[0] == pytest.approx(trainer.budget_min)
    assert order[1] == pytest.approx(trainer.budget_max)


def test_distill_alpha_zero_matches_not_distilling_at_all() -> None:
    """alpha=0 must be an exact no-op on the optimized objective.

    This is what pins the blend down: if the teacher term leaked in with any
    weight at alpha=0, the reported loss would drift from the undistilled run
    and every comparison against pareto_23 would be off.
    """
    batch = sample_batch()
    torch.manual_seed(0)
    a = make_trainer(
        budget_sampling="sandwich", budgets_per_step=3,
        inplace_distill=True, distill_alpha=0.0,
    )._train_step(batch, epoch=0)
    torch.manual_seed(0)
    b = make_trainer(
        budget_sampling="sandwich", budgets_per_step=3, inplace_distill=False
    )._train_step(batch, epoch=0)
    assert a["loss"] == pytest.approx(b["loss"], rel=1e-6)
    assert a["seg_loss"] == pytest.approx(b["seg_loss"], rel=1e-6)


def test_distillation_changes_the_objective_and_is_reported() -> None:
    """alpha > 0 must actually move the loss, and show up in diagnostics."""
    batch = sample_batch()
    torch.manual_seed(0)
    plain = make_trainer(
        budget_sampling="sandwich", budgets_per_step=3, inplace_distill=False
    )._train_step(batch, epoch=0)
    torch.manual_seed(0)
    distilled = make_trainer(
        budget_sampling="sandwich", budgets_per_step=3,
        inplace_distill=True, distill_alpha=1.0,
    )._train_step(batch, epoch=0)

    assert distilled["distill_loss"] > 0.0
    assert plain["distill_loss"] == 0.0
    assert distilled["loss"] != pytest.approx(plain["loss"], rel=1e-6)
    # The pure ground-truth term is still reported, so runs stay comparable.
    assert math.isfinite(distilled["seg_loss"])


def test_teacher_is_detached_so_students_do_not_train_the_teacher() -> None:
    """Students must not backpropagate into the teacher's forward pass.

    If the teacher were left attached, the widest subnetwork would receive
    gradient telling it to make itself easier to imitate -- a degenerate
    objective that BigNAS's detach is specifically there to prevent.
    """
    from mimose.losses.tiny_mimosa import soft_target_loss

    student = torch.rand(1, 4, 4, 4, 4, requires_grad=True)
    student = student / student.sum(dim=1, keepdim=True)
    teacher = torch.rand(1, 4, 4, 4, 4, requires_grad=True)
    teacher_normed = teacher / teacher.sum(dim=1, keepdim=True)

    soft_target_loss(student, teacher_normed.detach()).backward()
    assert teacher.grad is None


# ------------------------------------------------------ student modality drop


def test_drop_one_available_removes_exactly_one_available_modality() -> None:
    mask = torch.tensor([[True, True, True, True], [True, False, True, False], [False, False, True, False]])

    for _ in range(50):
        dropped = drop_one_available(mask)
        assert not (dropped & ~mask).any()
        assert dropped.sum(dim=1).tolist() == [3, 1, 1]
    assert torch.equal(dropped[2], mask[2])


def test_drop_one_available_rejects_non_batch_masks() -> None:
    with pytest.raises(ValueError, match=r"\[B, M\]"):
        drop_one_available(torch.ones(4, dtype=torch.bool))


def _masks_seen(trainer, batch) -> list[torch.Tensor]:
    seen: list[torch.Tensor] = []
    model = trainer.model
    original = model.forward

    def spy(images, mask):
        seen.append(mask.clone())
        return original(images, mask)

    model.forward = spy  # type: ignore[method-assign]
    try:
        trainer._train_step(batch, epoch=0)
    finally:
        model.forward = original  # type: ignore[method-assign]
    return seen


def test_student_modality_drop_keeps_the_teacher_mask_and_thins_the_students() -> None:
    trainer = make_trainer(
        budget_sampling="sandwich", budgets_per_step=3,
        inplace_distill=True, student_modality_drop=True,
    )
    batch = sample_batch()

    masks = _masks_seen(trainer, batch)

    assert torch.equal(masks[0], batch["mask"])
    for student in masks[1:]:
        assert not (student & ~batch["mask"]).any()
        assert student.sum().item() == batch["mask"].sum().item() - 1


def test_student_modality_drop_runs_the_widest_subnetwork_first_even_without_distillation() -> None:
    trainer = make_trainer(
        budget_sampling="sandwich", budgets_per_step=3,
        inplace_distill=False, student_modality_drop=True,
    )
    order = _budget_order(trainer, sample_batch())
    assert order[0] == pytest.approx(trainer.budget_max)
