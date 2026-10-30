from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from mimose.enums import TransformKind
from mimose.models.region_utils import regions_to_label_one_hot
from mimose.losses.config import build_loss_config
from mimose.losses.m3ae import M3AELoss, legacy_region_loss, m3ae_reconstruction_loss
from mimose.models.config import build_model_config
from mimose.models.m3ae import M3AE, input_patch_size
from mimose.training.config import OptimizerConfig, SchedulerConfig
from mimose.training.trainers.m3ae import M3AETrainer
from mimose.training.transforms import (
    IMFuseTransformManager,
    build_transform_manager,
)
from mimose.utils.cli_overrides import load_yaml_config


def _random_target(batch: int, num_classes: int, size: int) -> torch.Tensor:
    labels = torch.randint(0, num_classes, (batch, size, size, size))
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


def test_m3ae_forward_returns_segmentation_shape() -> None:
    torch.manual_seed(0)
    model = M3AE(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        output = model(images, mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)
    assert torch.allclose(output.sum(dim=1), torch.ones_like(output.sum(dim=1)), atol=1e-5)


def test_m3ae_train_and_eval_forward_have_the_same_output_contract() -> None:
    torch.manual_seed(0)
    model = M3AE(num_cls=4)
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    model.train()
    train_out = model(images, mask)
    model.eval()
    with torch.no_grad():
        eval_out = model(images, mask)

    assert train_out.shape == eval_out.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)


def test_m3ae_substitutes_missing_modalities_with_learned_embedding() -> None:
    torch.manual_seed(0)
    model = M3AE(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    all_present = torch.ones(1, 4, dtype=torch.bool)
    one_missing = torch.tensor([[True, True, True, False]])

    with torch.no_grad():
        out_full = model(images, all_present, mode="segment_logits")
        out_missing = model(images, one_missing, mode="segment_logits")

    assert torch.isfinite(out_missing).all()
    assert not torch.allclose(out_full, out_missing)


def test_m3ae_reconstruct_returns_input_shape() -> None:
    torch.manual_seed(0)
    model = M3AE(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.tensor([[True, True, False, True]])

    with torch.no_grad():
        recon = model.reconstruct(images, mask)

    assert recon.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)
    assert torch.isfinite(recon).all()


def test_m3ae_forward_mode_reconstruct_matches_reconstruct_method() -> None:
    torch.manual_seed(0)
    model = M3AE(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        via_forward = model(images, mask, mode="reconstruct")
        via_method = model.reconstruct(images, mask)

    assert torch.equal(via_forward, via_method)


def test_m3ae_forward_default_mode_is_unaffected_by_recon_head() -> None:
    torch.manual_seed(0)
    model = M3AE(num_cls=4).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        output = model(images, mask)

    assert output.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)
    assert torch.allclose(output.sum(dim=1), torch.ones_like(output.sum(dim=1)), atol=1e-5)


def test_m3ae_rejects_non_patch_sized_input() -> None:
    model = M3AE(num_cls=4).eval()
    images = torch.zeros(1, 4, 64, 64, 64)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with pytest.raises(RuntimeError, match="M3AE requires spatial dimensions"):
        model(images, mask)


@pytest.mark.parametrize(
    ("images", "mask"),
    [
        (torch.zeros(1, 3, input_patch_size, input_patch_size, input_patch_size), torch.ones(1, 4, dtype=torch.bool)),
        (torch.zeros(1, 4, input_patch_size, input_patch_size, input_patch_size), torch.ones(1, 3, dtype=torch.bool)),
    ],
)
def test_m3ae_rejects_invalid_modality_or_mask_shape(
    images: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    model = M3AE(num_cls=4).eval()

    with pytest.raises(RuntimeError, match="M3AE expects"):
        model(images, mask)


def test_m3ae_window_starts_use_50_percent_overlap() -> None:
    assert M3AE._window_starts(input_patch_size) == [0]
    assert M3AE._window_starts(input_patch_size + 64) == [0, 64]

    with pytest.raises(RuntimeError, match=f"at least {input_patch_size}"):
        M3AE._window_starts(input_patch_size - 1)


def test_m3ae_predict_slides_over_larger_volumes() -> None:
    torch.manual_seed(0)
    model = M3AE(num_cls=4).eval()
    images = torch.randn(1, 4, 160, 160, 144)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        pred = model.predict(images, mask)

    assert pred.shape == (1, 4, 160, 160, 144)
    assert torch.equal(pred.sum(dim=1), torch.ones(1, 160, 160, 144))


def test_m3ae_resolves_from_model_config_builder() -> None:
    model_config = build_model_config("m3ae", {"num_cls": 4})

    assert model_config.model_class is M3AE
    assert model_config.kwargs == {"num_cls": 4}


def test_m3ae_loss_resolves_from_loss_config_builder() -> None:
    loss_config = build_loss_config("m3ae", {"num_classes": 4})

    assert loss_config.loss_class is M3AELoss
    assert loss_config.kwargs == {"num_classes": 4}


def test_m3ae_transform_manager_resolves_from_kind() -> None:
    transform_manager = build_transform_manager(TransformKind.IMFUSE, crop_size=(128, 128, 128))

    assert isinstance(transform_manager, IMFuseTransformManager)
    assert transform_manager.crop_size == (128, 128, 128)


def test_m3ae_training_loss_end_to_end() -> None:
    torch.manual_seed(4)
    model = M3AE(num_cls=4)
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)
    target = _random_target(batch=1, num_classes=4, size=input_patch_size)

    output = model(images, mask, mode="segment_logits")
    assert output.shape == (1, 3, input_patch_size, input_patch_size, input_patch_size)
    loss_fn = M3AELoss(num_classes=4)
    metrics = loss_fn.training_loss(output, target)

    expected = metrics["cross"] + metrics["dice"]
    assert torch.allclose(metrics["loss"], expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("config_name", ["m3ae_18.yaml", "m3ae_23.yaml", "m3ae_25.yaml"])
def test_packaged_m3ae_configs_use_m3ae_training_contract(config_name: str) -> None:
    config = load_yaml_config(Path(config_name))

    assert config["trainer"] == "m3ae"
    assert config["model"] == "m3ae"
    assert config["loss"] == "m3ae"
    assert config["custom_trainer_kwargs"]["patch_size"] == 128
    assert config["norm_mode"] == "irm_min_max"
    assert config["batch_size"] == 1
    assert config["lr"] == 0.0003
    assert config["custom_trainer_kwargs"]["views"] == 2
    assert config["custom_trainer_kwargs"]["pretrain_samples_per_step"] == 2
    assert config["custom_trainer_kwargs"]["consistency_weight"] == 0.1
    assert config["custom_trainer_kwargs"]["final_from_best"] is True
    assert config["custom_model_kwargs"]["num_regions"] == 3
    assert config["scheduler"] == "cosine"
    assert config["optimizer"] == "adam"
    assert round(config["custom_trainer_kwargs"]["pretrain_fraction"] * config["num_epochs"]) == 600
    assert config["num_epochs"] - 600 == 300
    assert config["cosine_t_max"] == 600
    assert config["custom_trainer_kwargs"]["pretrain_weight_decay"] == 0.0
    assert config["custom_trainer_kwargs"]["limage_lr"] == 0.005
    assert config["custom_trainer_kwargs"]["limage_reg_weight"] == 0.005


def _legacy_edice(inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Verbatim legacy/m3ae/loss/dice.py EDiceLoss.forward.
    dice = 0
    ce = 0
    for i in range(target.size(1)):
        probs = torch.sigmoid(inputs[:, i, ...])
        t = target[:, i, ...]
        dice = dice + 1 - (2 * torch.sum(probs * t) + 1.0) / (probs.pow(2).sum() + t.pow(2).sum() + 1.0)
        ce = ce + torch.nn.BCELoss()(torch.sigmoid(inputs[:, i, ...]), t)
    return (0.7 * dice + 0.3 * ce) / target.size(1)


def test_m3ae_training_loss_is_legacy_edice_on_et_tc_wt() -> None:
    torch.manual_seed(2)
    logits = torch.randn(2, 3, 8, 8, 8) * 3
    target = _random_target(batch=2, num_classes=4, size=8)
    labels = target.argmax(dim=1)
    regions = torch.stack([labels == 3, (labels == 1) | (labels == 3), labels > 0], dim=1).float()

    metrics = M3AELoss(num_classes=4).training_loss(logits, target)

    assert torch.allclose(metrics["loss"], _legacy_edice(logits, regions), atol=1e-6)
    ds = M3AELoss(num_classes=4).deep_supervised_training_loss([logits, logits * 0.5], target)
    assert torch.allclose(ds["loss"], _legacy_edice(logits, regions) + _legacy_edice(logits * 0.5, regions), atol=1e-6)


def test_m3ae_legacy_region_loss_matches_edice_on_regions() -> None:
    torch.manual_seed(2)
    output = torch.softmax(torch.randn(2, 4, 8, 8, 8), dim=1)
    target = _random_target(batch=2, num_classes=4, size=8)

    metrics = legacy_region_loss(output, target)

    def region(t: torch.Tensor, classes: tuple[int, ...]) -> torch.Tensor:
        return sum(t[:, c] for c in classes)

    dice, bce = 0.0, 0.0
    for classes in ((3,), (1, 3), (1, 2, 3)):
        pred, tgt = region(output, classes), region(target, classes)
        dice += 1 - (2 * (pred * tgt).sum() + 1) / (pred.pow(2).sum() + tgt.pow(2).sum() + 1)
        bce += torch.nn.BCELoss()(pred, tgt)
    assert torch.allclose(metrics["dice"], 0.7 * dice / 3, atol=1e-6)
    assert torch.allclose(metrics["cross"], 0.3 * bce / 3, atol=1e-6)
    assert torch.allclose(metrics["loss"], metrics["dice"] + metrics["cross"])


def test_m3ae_legacy_region_loss_has_gradient_when_predicting_all_background() -> None:
    # The collapse seen with the softmax formulation: near one-hot background
    # predictions. The region loss must still push the tumour logits up.
    logits = torch.full((1, 4, 8, 8, 8), -12.0)
    logits[:, 0] = 12.0
    logits.requires_grad_(True)
    target = _random_target(batch=1, num_classes=4, size=8)

    legacy_region_loss(torch.softmax(logits, dim=1), target)["loss"].backward()

    assert logits.grad[:, 1:].abs().sum() > 0


def test_m3ae_reconstruction_loss_combines_mse_and_smoothness() -> None:
    torch.manual_seed(1)
    output = torch.randn(1, 4, 8, 8, 8)
    target = torch.randn(1, 4, 8, 8, 8)
    limage = torch.randn(1, 4, 8, 8, 8, requires_grad=True)

    metrics = m3ae_reconstruction_loss(output, target, limage, reg_weight=0.005)

    expected_recon = torch.nn.functional.mse_loss(output, target)
    expected_smoothness = torch.norm(limage - limage.mean(dim=(2, 3, 4), keepdim=True), p=2)
    assert torch.allclose(metrics["recon"], expected_recon)
    assert torch.allclose(metrics["smoothness"], expected_smoothness)
    assert torch.allclose(metrics["loss"], expected_recon + 0.005 * expected_smoothness)


def _new_m3ae_trainer_stub(*, pretrain_epochs: int, num_epochs: int = 10) -> M3AETrainer:
    """Constructs a bare, uninitialized ``M3AETrainer`` (bypassing
    ``BaseTrainer.__init__``, which needs real datasets/dataloaders) with
    just the attributes ``_apply_training_stage``/``_build_optimizer``
    touch, so the stage-switch logic can be unit tested in isolation."""
    trainer = M3AETrainer.__new__(M3AETrainer)
    trainer.model = M3AE(num_cls=4)
    trainer.pretrain_epochs = pretrain_epochs
    trainer.limage_lr = 0.005
    trainer.limage_reg_weight = 0.005
    trainer.num_epochs = num_epochs
    trainer._in_pretrain = None
    trainer.pretrain_weight_decay = None
    trainer.optimizer_config = OptimizerConfig(optim_class=torch.optim.Adam, lr=1e-3, weight_decay=1e-5)
    trainer.scheduler_config = SchedulerConfig(scheduler_class=CosineAnnealingLR, kwargs={"T_max": num_epochs, "eta_min": 0.0})
    trainer.optimizer = trainer._build_optimizer()
    trainer.scheduler = None
    return trainer


def test_m3ae_apply_training_stage_is_noop_when_disabled() -> None:
    trainer = _new_m3ae_trainer_stub(pretrain_epochs=0)

    trainer._apply_training_stage(0)

    assert trainer._in_pretrain is None
    assert trainer.model.limage.requires_grad is True


def test_m3ae_apply_training_stage_toggles_limage_requires_grad() -> None:
    trainer = _new_m3ae_trainer_stub(pretrain_epochs=3, num_epochs=10)

    trainer._apply_training_stage(0)
    assert trainer._in_pretrain is True
    assert trainer.model.limage.requires_grad is True

    trainer._apply_training_stage(2)
    assert trainer._in_pretrain is True

    trainer._apply_training_stage(3)
    assert trainer._in_pretrain is False
    assert trainer.model.limage.requires_grad is False


def test_m3ae_apply_training_stage_restarts_scheduler_entering_finetune() -> None:
    trainer = _new_m3ae_trainer_stub(pretrain_epochs=3, num_epochs=10)
    trainer._apply_training_stage(0)
    original_scheduler = trainer.scheduler = trainer._build_scheduler(trainer.optimizer)
    for _ in range(2):
        trainer.scheduler.step()

    trainer._apply_training_stage(3)

    assert trainer.scheduler is not original_scheduler
    assert trainer.scheduler.T_max == 7
    assert trainer.scheduler.last_epoch == 0
    # scheduler_config.kwargs is restored to its original (YAML-declared)
    # T_max after the rebuild, so logging/checkpointing still reflects the
    # configured value rather than the transient "remaining epochs" figure.
    assert trainer.scheduler_config.kwargs["T_max"] == 10


def test_m3ae_apply_training_stage_switches_pretrain_weight_decay() -> None:
    trainer = _new_m3ae_trainer_stub(pretrain_epochs=3, num_epochs=10)
    trainer.pretrain_weight_decay = 0.0

    trainer._apply_training_stage(0)
    assert all(group["weight_decay"] == 0.0 for group in trainer.optimizer.param_groups)

    trainer._apply_training_stage(3)
    assert all(group["weight_decay"] == 1e-5 for group in trainer.optimizer.param_groups)


def test_m3ae_build_optimizer_uses_single_group_when_pretrain_disabled() -> None:
    trainer = _new_m3ae_trainer_stub(pretrain_epochs=0)

    assert len(trainer.optimizer.param_groups) == 1


def test_m3ae_build_optimizer_splits_limage_into_own_group_when_enabled() -> None:
    trainer = _new_m3ae_trainer_stub(pretrain_epochs=3)

    assert len(trainer.optimizer.param_groups) == 2
    limage_group = next(g for g in trainer.optimizer.param_groups if g["lr"] == trainer.limage_lr)
    assert limage_group["params"] == [trainer.model.limage]
    other_group = next(g for g in trainer.optimizer.param_groups if g is not limage_group)
    assert not any(param is trainer.model.limage for param in other_group["params"])


def test_m3ae_entering_finetune_starts_a_fresh_adam_without_limage() -> None:
    trainer = _new_m3ae_trainer_stub(pretrain_epochs=3, num_epochs=10)
    trainer._apply_training_stage(0)
    trainer.scheduler = trainer._build_scheduler(trainer.optimizer)
    trainer.model.seg_layer.weight.grad = torch.ones_like(trainer.model.seg_layer.weight)
    trainer.optimizer.step()
    assert trainer.optimizer.state

    trainer._apply_training_stage(3)

    assert not trainer.optimizer.state
    assert trainer.optimizer.param_groups[0]["lr"] == 1e-3
    assert trainer.optimizer.param_groups[1]["lr"] == 0.0


def test_m3ae_template_is_sliced_at_the_crop_location() -> None:
    model = M3AE(num_cls=4, template_shape=(160, 160, 144)).train()
    images = torch.zeros(2, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.tensor([[True, False, True, True], [True, True, True, True]])
    location = torch.tensor([[5, 7, 9], [0, 0, 0]])

    filled = model.fill_missing(images, mask, location)

    expected = model.limage[0, 1, 5 : 5 + input_patch_size, 7 : 7 + input_patch_size, 9 : 9 + input_patch_size]
    assert torch.equal(filled[0, 1], expected)
    assert torch.equal(filled[1], images[1])


def test_regions_are_nested_into_brats_labels() -> None:
    # [ET, TC, WT] logits for 5 voxels: none, WT only, TC+WT, all, ET only.
    logits = torch.tensor([[-1.0, -1, -1, 1, 1], [-1, -1, 1, 1, -1], [-1, 1, 1, 1, -1]]).view(1, 3, 5, 1, 1)

    labels = regions_to_label_one_hot(logits).argmax(dim=1).flatten().tolist()

    assert labels == [0, 2, 1, 3, 3]
