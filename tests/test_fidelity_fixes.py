from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, PolynomialLR

from mimose.checkpoints import assert_finite_weights
from mimose.datasets.base import sample_count_uniform_mask
from mimose.datasets.imfuse import MASK_PATTERNS
from mimose.losses.imfuse import dice_loss
from mimose.losses.mcpl import class_contrastive_loss, modal_contrastive_loss
from mimose.models.m3ae import M3AE, input_patch_size as m3ae_patch
from mimose.models.mambavitakd import MambaVitAKD
from mimose.models.mcpl import MCPL
from mimose.models.uhved import UHVED
from mimose.training.config import OptimizerConfig, SchedulerConfig
from mimose.training.masking import mae_patch_gate
from mimose.training.trainers.base_trainer import BaseTrainer
from mimose.training.trainers.m3ae import M3AETrainer
from mimose.training.trainers.mambavitakd import MambaVitAKDTrainer
from mimose.training.trainers.mcpl import MCPLTrainer
from mimose.training.transforms.shaspec import ShaSpecTransform


def _backward_stub(**kwargs: object) -> BaseTrainer:
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.model = nn.Linear(4, 1)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1.0)
    trainer.use_grad_scaler = False
    trainer.grad_clip_norm = kwargs.get("grad_clip_norm", 12.0)
    trainer.skip_nonfinite_loss = bool(kwargs.get("skip_nonfinite_loss", False))
    trainer.nonfinite_steps_skipped = 0
    trainer._consecutive_nonfinite_steps = 0
    return trainer


def test_default_grad_clip_is_unchanged_and_none_disables_it(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []
    original = torch.nn.utils.clip_grad_norm_

    def spy(params, max_norm, *args, **kwargs):
        calls.append(max_norm)
        return original(params, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    trainer = _backward_stub()
    trainer._backward_step(trainer.model(torch.ones(1, 4)).sum())
    assert calls == [12.0]

    trainer = _backward_stub(grad_clip_norm=None)
    trainer._backward_step(trainer.model(torch.ones(1, 4)).sum())
    assert calls == [12.0]


def test_nonfinite_loss_is_skipped_only_when_enabled() -> None:
    trainer = _backward_stub(skip_nonfinite_loss=True)
    before = trainer.model.weight.detach().clone()
    trainer._backward_step(trainer.model(torch.ones(1, 4)).sum() * float("nan"))
    assert torch.equal(trainer.model.weight, before)
    assert trainer.nonfinite_steps_skipped == 1

    trainer = _backward_stub(skip_nonfinite_loss=False)
    trainer._backward_step(trainer.model(torch.ones(1, 4)).sum() * float("nan"))
    assert not torch.isfinite(trainer.model.weight).all()


def test_reset_param_group_lrs_undoes_a_fully_decayed_schedule() -> None:
    trainer = _backward_stub()
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(trainer.optimizer, T_max=3, eta_min=0.0)
    for _ in range(3):
        scheduler.step()
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.0)

    trainer._reset_param_group_lrs([1e-3])
    restarted = CosineAnnealingLR(trainer.optimizer, T_max=5, eta_min=0.0)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    restarted.step()
    assert trainer.optimizer.param_groups[0]["lr"] > 0


def _stage_stub(cls, model: nn.Module, *, pretrain_epochs: int, num_epochs: int, scheduler_config: SchedulerConfig):
    trainer = cls.__new__(cls)
    trainer.model = model
    trainer.pretrain_epochs = pretrain_epochs
    trainer.limage_lr = 0.005
    trainer.limage_reg_weight = 0.005
    trainer.num_epochs = num_epochs
    trainer._in_pretrain = None
    trainer.pretrain_weight_decay = None
    trainer.optimizer_config = OptimizerConfig(optim_class=torch.optim.Adam, lr=1e-3, weight_decay=0.0)
    trainer.scheduler_config = scheduler_config
    trainer.optimizer = trainer._build_optimizer()
    trainer.scheduler = None
    return trainer


@pytest.mark.parametrize(
    ("cls", "model_factory"),
    [(MCPLTrainer, lambda: MCPL(num_cls=4, layers=1)), (M3AETrainer, lambda: M3AE(num_cls=4))],
)
def test_finetune_stage_restarts_at_base_lr_after_pretrain_cosine_hits_zero(cls, model_factory) -> None:
    pretrain_epochs, num_epochs = 3, 8
    trainer = _stage_stub(
        cls,
        model_factory(),
        pretrain_epochs=pretrain_epochs,
        num_epochs=num_epochs,
        scheduler_config=SchedulerConfig(
            scheduler_class=CosineAnnealingLR, kwargs={"T_max": pretrain_epochs, "eta_min": 0.0}
        ),
    )
    trainer._apply_training_stage(0)
    trainer.scheduler = trainer._build_scheduler(trainer.optimizer)
    for _ in range(pretrain_epochs):
        trainer.scheduler.step()
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.0)

    trainer._apply_training_stage(pretrain_epochs)

    # Upstream fine-tunes with a fresh Adam over everything but the missing-modality filler.
    lrs = [group["lr"] for group in trainer.optimizer.param_groups]
    assert lrs == pytest.approx([1e-3, 0.0])
    trainer.scheduler.step()
    assert trainer.optimizer.param_groups[0]["lr"] > 0


def test_mambavitakd_student_lr_restarts_and_teacher_freezes_at_distillation() -> None:
    trainer = MambaVitAKDTrainer.__new__(MambaVitAKDTrainer)
    trainer.model = MambaVitAKD(num_cls=4)
    trainer.teacher_pretrain_epochs = 3
    trainer.teacher_pretrain_fraction = 0.3
    trainer.teacher_lr = 4e-4
    trainer.num_epochs = 10
    trainer.region_fusion_start_epoch = 0
    trainer._in_teacher_pretrain = None
    trainer.optimizer_config = OptimizerConfig(optim_class=torch.optim.Adam, lr=4e-4, weight_decay=1e-5, amsgrad=True)
    trainer.scheduler_config = SchedulerConfig(scheduler_class=PolynomialLR, kwargs={"total_iters": 10, "power": 0.9})
    trainer.optimizer = trainer._build_optimizer()
    trainer.scheduler = None
    trainer.rank = 0
    trainer.scheduler = trainer._build_scheduler(trainer.optimizer)

    trainer._apply_training_stage(0)
    for _ in range(3):
        trainer.scheduler.step()
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.0)
    assert trainer.optimizer.param_groups[0]["amsgrad"] is True

    trainer._apply_training_stage(3)

    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(4e-4)
    assert trainer.model.teacher_frozen is True
    assert not any(p.requires_grad for p in trainer.model.teacher.parameters())
    assert not any(p.requires_grad for p in trainer.model.channel_attention.parameters())


def test_mambavitakd_resume_into_distillation_replays_the_schedule() -> None:
    trainer = MambaVitAKDTrainer.__new__(MambaVitAKDTrainer)
    trainer.model = MambaVitAKD(num_cls=4)
    trainer.teacher_pretrain_epochs = 3
    trainer.teacher_pretrain_fraction = 0.3
    trainer.teacher_lr = 4e-4
    trainer.num_epochs = 10
    trainer.region_fusion_start_epoch = 0
    trainer._in_teacher_pretrain = None
    trainer.optimizer_config = OptimizerConfig(optim_class=torch.optim.Adam, lr=4e-4, weight_decay=0.0)
    trainer.scheduler_config = SchedulerConfig(scheduler_class=PolynomialLR, kwargs={"total_iters": 10, "power": 1.0})
    trainer.optimizer = trainer._build_optimizer()
    trainer.scheduler = None
    trainer.rank = 0

    trainer._apply_training_stage(5)

    # Linear decay over the 7 distillation epochs, 2 of which already ran.
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(4e-4 * (1 - 2 / 7))


def test_dice_loss_default_is_unchanged_and_per_sample_differs_on_mixed_batches() -> None:
    torch.manual_seed(0)
    output = torch.softmax(torch.randn(2, 4, 8, 8, 8), dim=1)
    target = torch.zeros(2, 4, 8, 8, 8)
    target[0, 1] = 1
    target[1, 0] = 1

    pooled = torch.tensor(0.0)
    for c in range(4):
        num = torch.sum(output[:, c] * target[:, c])
        pooled = pooled + 2.0 * num / (output[:, c].sum() + target[:, c].sum() + 1e-7)
    assert torch.allclose(dice_loss(output, target, num_cls=4), 1.0 - pooled / 4)

    per_sample = dice_loss(output, target, num_cls=4, per_sample=True)
    singles = torch.stack([dice_loss(output[i : i + 1], target[i : i + 1], num_cls=4) for i in range(2)]).mean()
    assert torch.allclose(per_sample, singles)
    assert not torch.allclose(per_sample, dice_loss(output, target, num_cls=4))


def test_count_uniform_mask_sampling_gives_full_set_a_quarter_of_the_time() -> None:
    torch.manual_seed(0)
    draws = torch.stack([sample_count_uniform_mask(4) for _ in range(8000)])
    assert draws.any(dim=1).all()
    full = draws.all(dim=1).float().mean().item()
    assert full == pytest.approx(0.25, abs=0.02)
    counts = draws.sum(dim=1)
    for k in range(1, 5):
        assert (counts == k).float().mean().item() == pytest.approx(0.25, abs=0.02)
    assert len(MASK_PATTERNS) == 15


def test_mae_patch_gate_keeps_an_eighth_of_tokens_in_16_cubed_blocks() -> None:
    torch.manual_seed(0)
    gate = mae_patch_gate(3, patch_shape=64, token_size=16, mask_ratio=0.875, mdp=3)
    assert gate.shape == (3, 4, 64, 64, 64)
    tokens = gate.view(3, 4, 4, 16, 4, 16, 4, 16)
    per_block = tokens.permute(0, 1, 2, 4, 6, 3, 5, 7).reshape(3, 4 * 64, -1)
    assert torch.all((per_block.min(dim=-1).values == per_block.max(dim=-1).values))
    kept = per_block[..., 0].sum(dim=1)
    assert torch.all(kept == int(0.125 * 4 * 64))


def _upstream_forward_class_cos(h_i: torch.Tensor, h_j: torch.Tensor, modal_num: int, t: float) -> torch.Tensor:
    """Verbatim copy of legacy/MCPL/loss/dice.py memory_Loss.forward_class_cos."""
    loss_i_j = 0.0
    tt = 0
    for i in range(modal_num):
        for j in range(i + 1, modal_num):
            positive_samples = torch.exp(torch.cosine_similarity(h_i[:, i], h_i[:, j], dim=0) / t)
            negative_samples = 0
            for k in range(modal_num):
                negative_samples = negative_samples + torch.exp(torch.cosine_similarity(h_i[:, i], h_j[:, k], dim=0) / t)
            tt = tt + 1
        loss_i_j = loss_i_j - torch.log(positive_samples / negative_samples)
    for i in range(modal_num):
        for j in range(i + 1, modal_num):
            positive_samples = torch.exp(torch.cosine_similarity(h_j[:, i], h_j[:, j], dim=0) / t)
            negative_samples = 0
            for k in range(modal_num):
                negative_samples = negative_samples + torch.exp(torch.cosine_similarity(h_j[:, i], h_i[:, k], dim=0) / t)
            tt = tt + 1
        loss_i_j = loss_i_j - torch.log(positive_samples / negative_samples)
    return loss_i_j / tt


def test_mcpl_contrastive_losses_match_upstream_memory_loss() -> None:
    torch.manual_seed(0)
    modalities, classes = 4, 4
    weight = torch.randn(128, modalities * classes)
    table = weight.view(-1, modalities, classes)

    expected_class = sum(
        _upstream_forward_class_cos(table[:, :, c1], table[:, :, c2], modalities, 0.5)
        for c1 in range(classes)
        for c2 in range(c1 + 1, classes)
    ) / (classes * (classes - 1) / 2)
    expected_modal = sum(
        _upstream_forward_class_cos(table[:, m1, :], table[:, m2, :], classes, 0.5)
        for m1 in range(modalities)
        for m2 in range(m1 + 1, modalities)
    ) / (modalities * (modalities - 1) / 2)

    got_class = class_contrastive_loss(weight, cc_modalities=modalities, cc_classes=classes)
    got_modal = modal_contrastive_loss(weight, cc_modalities=modalities, cc_classes=classes)
    assert torch.allclose(got_class, expected_class)
    assert torch.allclose(got_modal, expected_modal)


def test_mcpl_missing_modality_filler_is_frozen_zeros_by_default() -> None:
    model = MCPL(num_cls=4, layers=1)
    assert not model.limage.requires_grad
    assert torch.count_nonzero(model.limage) == 0


def test_m3ae_default_init_does_not_saturate_the_region_sigmoids() -> None:
    torch.manual_seed(0)
    model = M3AE(num_cls=4).eval()
    images = torch.randn(1, 4, m3ae_patch, m3ae_patch, m3ae_patch)
    with torch.no_grad():
        logits = model(images, torch.ones(1, 4, dtype=torch.bool), mode="segment_logits")
    assert logits.abs().mean().item() < 2.0


def test_uhved_init_keeps_instance_norm_affine_identity() -> None:
    model = UHVED(num_cls=4)
    norms = [m for m in model.modules() if isinstance(m, nn.InstanceNorm3d)]
    assert norms
    for norm in norms:
        assert torch.all(norm.weight == 1)
        assert torch.all(norm.bias == 0)


def test_shaspec_transform_returns_fixed_size_crops() -> None:
    torch.manual_seed(0)
    images = torch.randn(4, 100, 110, 96)
    labels = torch.zeros(1, 100, 110, 96)
    labels[0, 40:60, 40:60, 40:60] = 3
    transform = ShaSpecTransform(crop_size=(80, 80, 80))
    for _ in range(10):
        out_images, out_labels = transform(images, labels)
        assert out_images.shape == (4, 80, 80, 80)
        assert out_labels.shape == (1, 80, 80, 80)
        assert set(out_labels.unique().tolist()) <= {0.0, 3.0}


def test_assert_finite_weights_rejects_nan_checkpoints(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    assert_finite_weights(model, tmp_path / "ok.safetensors")
    with torch.no_grad():
        model.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        assert_finite_weights(model, tmp_path / "bad.safetensors")
