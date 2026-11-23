from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mimose.enums import LossKind, ModelKind, TrainerKind, TransformKind
from mimose.losses.aloramoe import ALoRAMoELoss
from mimose.losses.config import build_loss_config
from mimose.models.aloramoe import DATASET_MODALITY_ORDER, ALoRAMoE
from mimose.models.config import build_model_config
from mimose.preprocessing.normalization import volume_zscore
from mimose.training.trainers.aloramoe import ALoRAMoETrainer
from mimose.training.transforms import ALoRAMoETransformManager, build_transform_manager
from mimose.utils.cli_overrides import load_yaml_config

LEGACY_ROOT = Path(__file__).resolve().parents[1] / "legacy" / "ALoRA-MoE"


@pytest.fixture
def legacy():
    if not LEGACY_ROOT.is_dir():
        pytest.skip("legacy/ALoRA-MoE is not available")
    pytest.importorskip("einops")
    sys.path.insert(0, str(LEGACY_ROOT))
    try:
        from lora_moe_hgd import datasets, losses
        from lora_moe_hgd.model.seg_model import SegModel

        yield {"SegModel": SegModel, "losses": losses, "datasets": datasets}
    finally:
        sys.path.remove(str(LEGACY_ROOT))


def _small_model(**kwargs) -> ALoRAMoE:
    return ALoRAMoE(embed_dim=8, **kwargs)


def _trainer(**trainer_kwargs) -> ALoRAMoETrainer:
    trainer = ALoRAMoETrainer.__new__(ALoRAMoETrainer)
    trainer.curriculum_epochs = tuple(trainer_kwargs.get("curriculum_epochs", (10, 30, 50)))
    trainer.missing_drop_p = tuple(trainer_kwargs.get("missing_drop_p", (0.2, 0.3, 0.4)))
    return trainer


def test_aloramoe_matches_legacy_parameter_count_and_state_dict(legacy) -> None:
    ours = ALoRAMoE()
    reference = legacy["SegModel"]()

    assert ours.state_dict().keys() == reference.state_dict().keys()
    assert sum(p.numel() for p in ours.parameters()) == sum(p.numel() for p in reference.parameters())


def test_aloramoe_forward_matches_legacy_before_the_final_upsampling(legacy) -> None:
    torch.manual_seed(0)
    reference = legacy["SegModel"](embed_dim=8).double().eval()
    ours = _small_model().double().eval()
    ours.load_state_dict(reference.state_dict())
    x_legacy = torch.randn(2, 4, 32, 32, 32, dtype=torch.float64)
    x_legacy[1, 2] = 0  # legacy zero-fills a dropped modality (T2) in the dataset
    inverse = torch.argsort(torch.tensor(DATASET_MODALITY_ORDER))
    mask = torch.ones(2, 4, dtype=torch.bool)
    mask[1, inverse.tolist().index(2)] = False
    captured: dict[str, torch.Tensor] = {}
    reference.dec.head.register_forward_hook(lambda m, i, o: captured.__setitem__("legacy", o))
    ours.dec.head.register_forward_hook(lambda m, i, o: captured.__setitem__("ours", o))

    with torch.no_grad():
        legacy_out = reference(x_legacy)
        ours_out = ours(x_legacy[:, inverse], mask)

    torch.testing.assert_close(captured["ours"], captured["legacy"])
    # Legacy's fixed x8 upsampling from the stride-4 map doubles the size; ours returns the input size.
    assert legacy_out.shape[-3:] == (64, 64, 64)
    assert ours_out.shape == (2, 4, 32, 32, 32)


def test_aloramoe_predict_covers_the_whole_volume() -> None:
    torch.manual_seed(0)
    model = _small_model().eval()
    volume = torch.randn(1, 4, 130, 128, 129)

    with torch.no_grad():
        logits = model.predict(volume, torch.ones(1, 4, dtype=torch.bool))

    assert logits.shape == (1, 4, 130, 128, 129)
    assert torch.isfinite(logits).all()


def test_aloramoe_resolves_from_config_builders_and_enums() -> None:
    assert TrainerKind("aloramoe") is TrainerKind.ALORAMOE
    assert ModelKind("aloramoe") is ModelKind.ALORAMOE
    assert LossKind("aloramoe") is LossKind.ALORAMOE
    assert build_model_config("aloramoe").model_class is ALoRAMoE
    assert build_loss_config("aloramoe").loss_class is ALoRAMoELoss
    assert isinstance(build_transform_manager(TransformKind.ALORAMOE), ALoRAMoETransformManager)


def test_aloramoe_loss_matches_legacy_dice_bce_with_curriculum(legacy) -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 8, 8, 8)
    labels = torch.randint(0, 4, (2, 1, 8, 8, 8))
    legacy_fn = legacy["losses"].DiceBCELoss(0.5)
    loss = ALoRAMoELoss()

    for stage, classes in ((0, (2,)), (1, (2, 1)), (2, (2, 1, 3))):
        expected = sum(legacy_fn(logits[:, c : c + 1], (labels == c).float()) for c in classes)
        torch.testing.assert_close(loss.training_loss(logits, labels, stage)["loss"], expected)


def test_aloramoe_loss_rejects_a_curriculum_that_skips_a_class() -> None:
    with pytest.raises(ValueError):
        ALoRAMoELoss(curriculum_classes=[1, 2])


def test_aloramoe_curriculum_stage_matches_legacy_schedule() -> None:
    trainer = _trainer()

    assert [trainer.curriculum_stage(e) for e in (0, 9, 10, 29, 30, 49)] == [0, 0, 1, 1, 2, 2]
    assert [trainer._drop_p(s) for s in (0, 1, 2)] == [0.2, 0.3, 0.4]


def test_aloramoe_missing_modality_simulation_matches_legacy(legacy) -> None:
    trainer = _trainer()
    torch.manual_seed(3)
    ours = trainer._simulate_missing_modalities(64, 4, 0.9)
    torch.manual_seed(3)
    expected = torch.stack(
        [legacy["datasets"].simulate_missing_modalities(torch.ones(4, 1), 0.9)[1].bool() for _ in range(64)]
    )

    assert torch.equal(ours, expected)
    assert ours.any(dim=1).all()


def test_aloramoe_transform_only_crops_and_flips() -> None:
    manager = ALoRAMoETransformManager(crop_size=(8, 8, 8))
    images = torch.arange(4 * 10 * 10 * 10, dtype=torch.float32).reshape(4, 10, 10, 10)
    labels = torch.zeros(1, 10, 10, 10)

    out_images, out_labels = manager(images, labels, mode="train")

    assert out_images.shape == (4, 8, 8, 8)
    assert out_labels.shape == (1, 8, 8, 8)
    # No intensity jitter: every output value is an input value.
    assert torch.isin(out_images, images).all()


def test_volume_zscore_matches_legacy_whole_volume_normalisation() -> None:
    rng = np.random.default_rng(0)
    images = rng.random((4, 6, 7, 5)).astype(np.float32)
    images[:, :2] = 0  # background counts towards the statistics

    out = volume_zscore(images, config=None)

    for c in range(4):
        x = torch.from_numpy(images[c])
        expected = (x - x.mean()) / (x.std() + 1e-6)
        np.testing.assert_allclose(out[c], expected.numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("config_name", ["aloramoe_18.yaml", "aloramoe_23.yaml", "aloramoe_25.yaml"])
def test_packaged_aloramoe_configs_use_legacy_recipe(config_name: str) -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "mimose" / "data" / "config_templates"
    config = load_yaml_config(root / config_name)

    assert (config["trainer"], config["model"], config["loss"]) == ("aloramoe", "aloramoe", "aloramoe")
    assert (config["crop_mode"], config["norm_mode"]) == ("none", "volume_zscore")
    assert (config["optimizer"], config["amsgrad"], config["lr"], config["weight_decay"]) == ("adam", False, 3e-4, 5e-5)
    assert (config["num_epochs"], config["batch_size"], config["precision"]) == (50, 1, "fp32")
    assert list(config["custom_loss_kwargs"]["curriculum_classes"]) == [2, 1, 3]
    trainer_kwargs = config["custom_trainer_kwargs"]
    assert list(trainer_kwargs["curriculum_epochs"]) == [10, 30, 50]
    assert list(trainer_kwargs["missing_drop_p"]) == [0.2, 0.3, 0.4]
    assert (trainer_kwargs["train_masking_mode"], trainer_kwargs["val_masking_mode"]) == ("full", "full")
    assert trainer_kwargs["final_from_best"] is True
    assert trainer_kwargs["grad_clip_norm"] is None
