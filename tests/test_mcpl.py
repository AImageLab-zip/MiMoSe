from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR

from mimose.losses.config import build_loss_config
from mimose.losses.mcpl import MCPLLoss, class_contrastive_loss, mcpl_reconstruction_loss, modal_contrastive_loss
from mimose.models.config import build_model_config
from mimose.models.mcpl import MCPL, input_patch_size
from mimose.training.config import OptimizerConfig, SchedulerConfig
from mimose.training.trainers.mcpl import MCPLTrainer

CONFIG_TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "src" / "mimose" / "data" / "config_templates"


def _random_target(batch: int, num_classes: int, size: int) -> torch.Tensor:
    labels = torch.randint(0, num_classes, (batch, size, size, size))
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


def test_mcpl_forward_returns_segmentation_shape() -> None:
    torch.manual_seed(0)
    model = MCPL(num_cls=4, layers=1).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        output = model(images, mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)
    assert torch.allclose(output.sum(dim=1), torch.ones_like(output.sum(dim=1)), atol=1e-5)


def test_mcpl_substitutes_missing_modalities_with_learned_embedding() -> None:
    torch.manual_seed(0)
    model = MCPL(num_cls=4, layers=1).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    all_present = torch.ones(1, 4, dtype=torch.bool)
    one_missing = torch.tensor([[True, True, True, False]])

    with torch.no_grad():
        out_full = model(images, all_present, mode="segment_logits")
        out_missing = model(images, one_missing, mode="segment_logits")

    assert torch.isfinite(out_missing).all()
    assert not torch.allclose(out_full, out_missing)


def test_mcpl_segment_train_returns_deep_supervision_heads_and_prototypes() -> None:
    torch.manual_seed(0)
    model = MCPL(num_cls=4, layers=1).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        seg_outputs, cluster_centers = model(images, mask, mode="segment_train")

    assert len(seg_outputs) == 3
    for output in seg_outputs:
        assert output.shape == (1, 3, input_patch_size, input_patch_size, input_patch_size)
        # Raw voxel-prototype cosines, as upstream feeds them to the sigmoid.
        assert output.abs().max() <= 1.0 + 1e-5
    assert model.nums_q == 12
    assert cluster_centers.shape == (1, 128, model.nums_q)


def test_mcpl_reconstruct_returns_input_shape() -> None:
    torch.manual_seed(0)
    model = MCPL(num_cls=4, layers=1).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.tensor([[True, True, False, True]])

    with torch.no_grad():
        recon, cluster_centers = model.reconstruct(images, mask)

    assert recon.shape == (1, 4, input_patch_size, input_patch_size, input_patch_size)
    assert torch.isfinite(recon).all()
    assert cluster_centers.shape == (1, 128, model.nums_q)


def test_mcpl_forward_mode_reconstruct_matches_reconstruct_method() -> None:
    torch.manual_seed(0)
    model = MCPL(num_cls=4, layers=1).eval()
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        via_forward = model(images, mask, mode="reconstruct")
        via_method = model.reconstruct(images, mask)

    assert torch.equal(via_forward[0], via_method[0])


def test_mcpl_rejects_non_patch_sized_input() -> None:
    model = MCPL(num_cls=4, layers=1).eval()
    images = torch.zeros(1, 4, 64, 64, 64)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with pytest.raises(RuntimeError, match="MCPL requires spatial dimensions"):
        model(images, mask)


@pytest.mark.parametrize(
    ("images", "mask"),
    [
        (torch.zeros(1, 3, input_patch_size, input_patch_size, input_patch_size), torch.ones(1, 4, dtype=torch.bool)),
        (torch.zeros(1, 4, input_patch_size, input_patch_size, input_patch_size), torch.ones(1, 3, dtype=torch.bool)),
    ],
)
def test_mcpl_rejects_invalid_modality_or_mask_shape(images: torch.Tensor, mask: torch.Tensor) -> None:
    model = MCPL(num_cls=4, layers=1).eval()

    with pytest.raises(RuntimeError, match="MCPL expects"):
        model(images, mask)


def test_mcpl_window_starts_use_50_percent_overlap() -> None:
    assert MCPL._window_starts(input_patch_size) == [0]
    assert MCPL._window_starts(input_patch_size + 64) == [0, 64]

    with pytest.raises(RuntimeError, match=f"at least {input_patch_size}"):
        MCPL._window_starts(input_patch_size - 1)


def test_mcpl_predict_slides_over_larger_volumes() -> None:
    torch.manual_seed(0)
    model = MCPL(num_cls=4, layers=1).eval()
    images = torch.randn(1, 4, 160, 160, 160)
    mask = torch.ones(1, 4, dtype=torch.bool)

    with torch.no_grad():
        pred = model.predict(images, mask)

    assert pred.shape == (1, 4, 160, 160, 160)
    assert torch.equal(pred.sum(dim=1), torch.ones(1, 160, 160, 160))


def test_mcpl_resolves_from_model_config_builder() -> None:
    model_config = build_model_config("mcpl", {"num_cls": 4})

    assert model_config.model_class is MCPL
    assert model_config.kwargs == {"num_cls": 4}


def test_mcpl_loss_resolves_from_loss_config_builder() -> None:
    loss_config = build_loss_config("mcpl", {"num_classes": 4})

    assert loss_config.loss_class is MCPLLoss
    assert loss_config.kwargs == {"num_classes": 4}


def test_mcpl_deep_supervised_training_loss_end_to_end() -> None:
    torch.manual_seed(4)
    model = MCPL(num_cls=4, layers=1)
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)
    target = _random_target(batch=1, num_classes=4, size=input_patch_size)

    seg_outputs, _ = model(images, mask, mode="segment_train")
    loss_fn = MCPLLoss(num_classes=4)
    metrics = loss_fn.deep_supervised_training_loss(seg_outputs, target)

    expected = metrics["cross"] + metrics["dice"]
    assert torch.allclose(metrics["loss"], expected, atol=1e-6, rtol=1e-6)


def test_mcpl_class_and_modal_contrastive_losses_are_finite() -> None:
    torch.manual_seed(0)
    model = MCPL(num_cls=4, layers=1)

    class_loss = class_contrastive_loss(model._cluster_centers.weight, cc_modalities=4, cc_classes=3)
    modal_loss = modal_contrastive_loss(model._cluster_centers.weight, cc_modalities=4, cc_classes=3)

    assert torch.isfinite(class_loss)
    assert torch.isfinite(modal_loss)


def test_mcpl_reconstruction_loss_combines_mse_and_modal_contrast() -> None:
    torch.manual_seed(0)
    model = MCPL(num_cls=4, layers=1)
    images = torch.randn(1, 4, input_patch_size, input_patch_size, input_patch_size)
    mask = torch.ones(1, 4, dtype=torch.bool)

    recon, cluster_centers = model(images, mask, mode="reconstruct")
    metrics = mcpl_reconstruction_loss(
        recon, images, model._cluster_centers.weight, cc_modalities=4, cc_classes=3, modal_contrast_weight=0.01
    )

    expected_recon = torch.nn.functional.mse_loss(recon, images)
    expected_modal_contrast = modal_contrastive_loss(model._cluster_centers.weight, cc_modalities=4, cc_classes=3)
    assert torch.allclose(metrics["recon"], expected_recon)
    assert torch.allclose(metrics["modal_contrast"], expected_modal_contrast)
    assert torch.allclose(metrics["loss"], expected_recon + 0.01 * expected_modal_contrast)


@pytest.mark.parametrize("config_name", ["mcpl_18.yaml", "mcpl_23.yaml", "mcpl_25.yaml"])
def test_packaged_mcpl_configs_use_mcpl_training_contract(config_name: str) -> None:
    config = yaml.safe_load((CONFIG_TEMPLATES_DIR / config_name).read_text(encoding="utf-8"))

    assert config["trainer"] == "mcpl"
    assert config["model"] == "mcpl"
    assert config["loss"] == "mcpl"
    assert config["custom_trainer_kwargs"]["patch_size"] == 128
    assert config["norm_mode"] == "subject_zscore_nonzero"
    assert config["batch_size"] == 1
    assert config["lr"] == 0.0003
    assert config["custom_model_kwargs"]["cc_classes"] == 3
    assert config["custom_model_kwargs"]["logit_scale"] == 1.0
    assert config["custom_trainer_kwargs"]["deep_supervised"] is False
    assert config["custom_trainer_kwargs"]["views"] == 2
    assert config["custom_trainer_kwargs"]["consistency_weight"] == 0.1
    assert round(config["custom_trainer_kwargs"]["pretrain_fraction"] * config["num_epochs"]) == 600
    assert config["num_epochs"] - 600 == 500
    assert config["scheduler"] == "cosine"
    assert config["optimizer"] == "adam"
    assert config["custom_trainer_kwargs"]["limage_lr"] == 0.005
    assert config["custom_trainer_kwargs"]["class_contrast_weight"] == 0.01
    assert config["custom_trainer_kwargs"]["modal_contrast_weight"] == 0.01


def _new_mcpl_trainer_stub(
    *, pretrain_epochs: int, num_epochs: int = 10, learnable_filler: bool = True
) -> MCPLTrainer:
    """Constructs a bare, uninitialized ``MCPLTrainer`` (bypassing
    ``BaseTrainer.__init__``, which needs real datasets/dataloaders) with
    just the attributes ``_apply_training_stage``/``_build_optimizer``
    touch, so the stage-switch logic can be unit tested in isolation."""
    trainer = MCPLTrainer.__new__(MCPLTrainer)
    trainer.model = MCPL(num_cls=4, layers=1, learnable_filler=learnable_filler)
    trainer.pretrain_epochs = pretrain_epochs
    trainer.limage_lr = 0.005
    trainer.num_epochs = num_epochs
    trainer._in_pretrain = None
    trainer.pretrain_weight_decay = None
    trainer.optimizer_config = OptimizerConfig(optim_class=torch.optim.Adam, lr=1e-3, weight_decay=1e-5)
    trainer.scheduler_config = SchedulerConfig(scheduler_class=CosineAnnealingLR, kwargs={"T_max": num_epochs, "eta_min": 0.0})
    trainer.optimizer = trainer._build_optimizer()
    trainer.scheduler = None
    return trainer


def test_mcpl_apply_training_stage_is_noop_when_disabled() -> None:
    trainer = _new_mcpl_trainer_stub(pretrain_epochs=0)

    trainer._apply_training_stage(0)

    assert trainer._in_pretrain is None
    assert trainer.model.limage.requires_grad is True


def test_mcpl_apply_training_stage_toggles_limage_requires_grad() -> None:
    trainer = _new_mcpl_trainer_stub(pretrain_epochs=3, num_epochs=10)

    trainer._apply_training_stage(0)
    assert trainer._in_pretrain is True
    assert trainer.model.limage.requires_grad is True

    trainer._apply_training_stage(2)
    assert trainer._in_pretrain is True

    trainer._apply_training_stage(3)
    assert trainer._in_pretrain is False
    assert trainer.model.limage.requires_grad is False


def test_mcpl_apply_training_stage_restarts_scheduler_entering_finetune() -> None:
    trainer = _new_mcpl_trainer_stub(pretrain_epochs=3, num_epochs=10)
    trainer._apply_training_stage(0)
    original_scheduler = trainer.scheduler = trainer._build_scheduler(trainer.optimizer)
    for _ in range(2):
        trainer.scheduler.step()

    trainer._apply_training_stage(3)

    assert trainer.scheduler is not original_scheduler
    assert trainer.scheduler.T_max == 7
    assert trainer.scheduler.last_epoch == 0
    assert trainer.scheduler_config.kwargs["T_max"] == 10


def test_mcpl_build_optimizer_uses_single_group_when_pretrain_disabled() -> None:
    trainer = _new_mcpl_trainer_stub(pretrain_epochs=0)

    assert len(trainer.optimizer.param_groups) == 1


def test_mcpl_build_optimizer_splits_limage_into_own_group_when_enabled() -> None:
    trainer = _new_mcpl_trainer_stub(pretrain_epochs=3)

    assert len(trainer.optimizer.param_groups) == 2
    limage_group = next(g for g in trainer.optimizer.param_groups if g["lr"] == trainer.limage_lr)
    assert limage_group["params"] == [trainer.model.limage]
    other_group = next(g for g in trainer.optimizer.param_groups if g is not limage_group)
    assert not any(param is trainer.model.limage for param in other_group["params"])


def test_mcpl_views_duplicate_each_sample_with_independent_masks_view_major() -> None:
    trainer = _new_mcpl_trainer_stub(pretrain_epochs=0, learnable_filler=False)
    trainer.views = 2
    trainer.legacy_augment = False
    trainer.mask_sampling = "count_uniform"
    images = torch.randn(2, 4, input_patch_size, input_patch_size, input_patch_size)
    seg = torch.randint(0, 4, (2, 1, input_patch_size, input_patch_size, input_patch_size)).float()
    mask = torch.tensor([[True, False, True, True], [True, True, True, True]])

    view_images, view_seg = trainer._build_views(images, seg, mask)

    assert view_images.shape == (4, 4, input_patch_size, input_patch_size, input_patch_size)
    assert torch.equal(view_seg[:2], seg) and torch.equal(view_seg[2:], seg)
    assert torch.equal(view_images[0, 1], torch.zeros(input_patch_size, input_patch_size, input_patch_size))
    assert torch.equal(view_images[1], images[1])
