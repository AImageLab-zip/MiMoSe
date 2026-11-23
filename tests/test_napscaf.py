from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path

import pytest
import torch

from mimose.enums import LossKind, ModelKind, TrainerKind
from mimose.losses.config import build_loss_config
from mimose.losses.napscaf import NAPSCAFLoss
from mimose.models.config import build_model_config
from mimose.models.napscaf import NAPSCAF
from mimose.utils.cli_overrides import load_yaml_config


def _model() -> NAPSCAF:
    # Skip the ImageNet download: the prior branch's weights do not matter for these checks.
    return NAPSCAF(num_cls=4, pretrained_prior=False)


def test_napscaf_matches_paper_parameter_count() -> None:
    params = sum(p.numel() for p in _model().parameters())

    # README: 58.3M parameters (DenseNet121 classifier excluded).
    assert params / 1e6 == pytest.approx(58.3, abs=0.05)


def test_napscaf_training_forward_returns_slice_logits() -> None:
    torch.manual_seed(0)
    model = _model().train()
    model.is_training = True
    slices = torch.randn(3, 4, 64, 64)
    mask = torch.tensor([[True, False, True, True]] * 3)

    logits = model(slices, mask)

    assert logits.shape == (3, 4, 64, 64)


def test_napscaf_volume_inference_handles_non_multiple_of_32_sizes() -> None:
    torch.manual_seed(0)
    model = _model().eval()
    volume = torch.randn(2, 4, 70, 90, 5)
    mask = torch.tensor([[True, True, True, True], [False, False, True, False]])

    with torch.no_grad():
        probabilities = model.predict(volume, mask)

    assert probabilities.shape == (2, 4, 70, 90, 5)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2, 70, 90, 5), atol=1e-5)


def test_napscaf_prior_branch_receives_no_gradient() -> None:
    model = _model().train()
    model.is_training = True

    model(torch.randn(2, 4, 64, 64), torch.ones(2, 4, dtype=torch.bool)).mean().backward()

    assert all(p.grad is None for p in model.net.nap_path.vae.parameters())
    assert model.net.convout.weight.grad is not None


def test_napscaf_zero_fills_missing_modalities() -> None:
    torch.manual_seed(0)
    model = _model().eval()
    model.is_training = True
    slices = torch.randn(1, 4, 64, 64)
    # External order [t1c, t1n, t2f, t2w]; dropping t1c must equal zeroing it by hand.
    mask = torch.tensor([[False, True, True, True]])
    zeroed = slices.clone()
    zeroed[:, 0] = 0

    torch.manual_seed(1)
    with torch.no_grad():
        masked = model(slices, mask)
    torch.manual_seed(1)
    with torch.no_grad():
        manual = model(zeroed, torch.ones(1, 4, dtype=torch.bool))

    assert torch.allclose(masked, manual, atol=1e-5)


@pytest.mark.parametrize(
    ("images", "mask"),
    [
        (torch.zeros(1, 3, 32, 32, 2), torch.ones(1, 4, dtype=torch.bool)),
        (torch.zeros(1, 4, 32, 32, 2), torch.ones(1, 3, dtype=torch.bool)),
    ],
)
def test_napscaf_rejects_invalid_modality_or_mask_shape(images: torch.Tensor, mask: torch.Tensor) -> None:
    with pytest.raises(RuntimeError, match="NAPSCAF expects"):
        _model().eval()(images, mask)


def test_napscaf_resolves_from_config_builders_and_enums() -> None:
    model_config = build_model_config("napscaf", {"num_cls": 4})
    loss_config = build_loss_config("napscaf", {"num_classes": 4, "ce_weight": 0.5})

    assert model_config.model_class is NAPSCAF
    assert loss_config.loss_class is NAPSCAFLoss
    assert loss_config.kwargs == {"num_classes": 4, "ce_weight": 0.5}
    assert TrainerKind("napscaf") and ModelKind("napscaf") and LossKind("napscaf")


def test_napscaf_loss_combines_ce_and_dice() -> None:
    torch.manual_seed(0)
    loss_fn = NAPSCAFLoss(num_classes=4)
    logits = torch.randn(2, 4, 16, 16)
    labels = torch.randint(0, 4, (2, 16, 16))

    metrics = loss_fn.training_loss(logits, labels)

    assert torch.allclose(metrics["loss"], 0.5 * metrics["ce"] + 0.5 * metrics["dice"])
    assert torch.allclose(metrics["ce"], torch.nn.functional.cross_entropy(logits, labels))
    one_hot = torch.nn.functional.one_hot(labels, 4).movedim(-1, 1).float()
    assert loss_fn.training_loss(one_hot * 50.0, labels)["dice"] < 1e-4


def _make_epoch_ready_trainer(steps: int = 2, slices_per_patch: int = 4):
    from mimose.training.trainers import NAPSCAFTrainer

    trainer = NAPSCAFTrainer.__new__(NAPSCAFTrainer)
    model = _model()
    trainer.model = model
    trainer.device = torch.device("cpu")
    trainer.rank = 0
    trainer.num_epochs = 10
    trainer.distributed = False
    trainer.use_grad_scaler = False
    trainer.grad_scaler = None
    trainer.amp_enabled = False
    trainer.precision = None
    trainer.model_config = None
    trainer.skip_nonfinite_loss = True
    trainer.grad_clip_norm = None
    trainer.slices_per_patch = slices_per_patch
    trainer.slice_sampling = "tumor"
    trainer.legacy_slices = False
    trainer.loss_fn = NAPSCAFLoss(num_classes=4)
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer._autocast_context = lambda: nullcontext()  # type: ignore[method-assign]
    trainer._loss_impl = lambda: trainer.loss_fn  # type: ignore[method-assign]

    torch.manual_seed(0)
    batches = [
        {
            "images": torch.randn(2, 4, 32, 32, 8),
            "seg": torch.randint(0, 4, (2, 1, 32, 32, 8)),
            "mask": torch.tensor([[True, False, True, True], [False, True, False, False]]),
        }
        for _ in range(steps)
    ]
    iterator = iter(batches)
    trainer.train_loader = batches
    trainer.iter_per_epoch = None
    trainer.debug = False
    trainer._next_train_batch = lambda: next(iterator)  # type: ignore[method-assign]
    return trainer


def test_napscaf_trainer_prefers_tumor_slices() -> None:
    trainer = _make_epoch_ready_trainer(slices_per_patch=3)
    seg = torch.zeros(8, 8, 10, dtype=torch.long)
    seg[2:4, 2:4, [1, 6]] = 2

    for _ in range(5):
        indices = trainer._sample_slice_indices(seg)
        assert len(indices) == 3
        assert {1, 6} <= set(indices.tolist())


def test_napscaf_trainer_slices_patches_with_their_masks() -> None:
    trainer = _make_epoch_ready_trainer(slices_per_patch=4)
    images = torch.randn(2, 4, 32, 32, 8)
    seg = torch.randint(0, 4, (2, 1, 32, 32, 8))
    mask = torch.tensor([[True, False, True, True], [False, True, False, False]])

    slices, labels, slice_mask = trainer._slice_batch(images, seg, mask)

    assert slices.shape == (8, 4, 32, 32)
    assert labels.shape == (8, 32, 32)
    assert torch.equal(slice_mask[:4], mask[0].expand(4, -1))
    assert torch.equal(slice_mask[4:], mask[1].expand(4, -1))


def test_napscaf_trainer_train_epoch_reports_loss_terms() -> None:
    trainer = _make_epoch_ready_trainer(steps=2)
    result = trainer.train_epoch(epoch=0)

    for key in ("loss", "ce", "dice"):
        assert key in result
        assert math.isfinite(result[key])


@pytest.mark.parametrize("config_name", ["napscaf_18.yaml", "napscaf_23.yaml", "napscaf_25.yaml"])
def test_packaged_napscaf_configs_use_napscaf_training_contract(config_name: str) -> None:
    config = load_yaml_config(Path(config_name))

    assert config["trainer"] == "napscaf"
    assert config["model"] == "napscaf"
    assert config["loss"] == "napscaf"
    trainer_kwargs = config["custom_trainer_kwargs"]
    assert trainer_kwargs["legacy_slices"] is True
    assert trainer_kwargs["slices_per_case"] == 5
    assert trainer_kwargs["slice_seed"] == 7
    assert trainer_kwargs["min_nonzero_ratio"] == pytest.approx(0.05)
    assert trainer_kwargs["slice_size"] == 256
    assert trainer_kwargs["train_masking_mode"] == "full"
    assert trainer_kwargs["final_from_best"] is True
    assert trainer_kwargs["fp32_validation"] is True
    assert config["validation_every"] == 1
    assert config["custom_model_kwargs"]["inference_size"] == 256
    assert config["crop_mode"] == "none"
    assert config["clamp_mode"] == "none"
    assert config["norm_mode"] == "slice_percentile_uint8"
    assert config["batch_size"] == 16
    assert config["num_epochs"] == 150
    assert config["precision"] == "fp16"
    assert config["optimizer"] == "adamw"
    assert tuple(config["betas"]) == (0.9, 0.99)
    assert config["scheduler"] == "step"
    assert config["step_gamma"] == 1.0
    assert config["lr"] == pytest.approx(1e-4)
    assert config["weight_decay"] == pytest.approx(1e-4)


def test_slice_percentile_uint8_matches_legacy_png_normalisation() -> None:
    import numpy as np

    from mimose.preprocessing.normalization import slice_percentile_uint8

    rng = np.random.default_rng(0)
    images = np.zeros((4, 24, 20, 3))
    images[:, 4:18, 3:15, :] = rng.gamma(2.0, 100.0, size=(4, 14, 12, 3))
    out = slice_percentile_uint8(images, None)

    def legacy(array):  # legacy/NAP-SCAF/scripts/preprocess_brats_png.py::normalize_to_uint8, then /255
        p1, p99 = np.percentile(array, (1, 99))
        array = np.clip(array, p1, p99)
        if p99 <= p1:
            return np.zeros_like(array, dtype=np.uint8)
        return ((array - p1) / (p99 - p1) * 255.0).astype(np.uint8)

    for c in range(4):
        for z in range(3):
            np.testing.assert_array_equal(out[c, ..., z], legacy(images[c, ..., z]) / np.float32(255.0))
    assert slice_percentile_uint8(np.zeros((4, 8, 8, 2)), None).max() == 0


def _write_case(root: Path, name: str, tumor_slices: list[int], depth: int = 10, empty_modality_slice: int | None = None):
    import numpy as np

    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    images = np.zeros((4, 16, 16, depth), dtype=np.float32)
    images[:, 2:14, 2:14, :] = rng.integers(1, 256, size=(4, 12, 12, depth)) / 255.0
    if empty_modality_slice is not None:
        images[1, ..., empty_modality_slice] = 0.0
    seg = np.zeros((1, 16, 16, depth), dtype=np.uint8)
    for z in tumor_slices:
        seg[0, 6:9, 6:9, z] = 3
    np.savez_compressed(root / f"{name}.npz", images=images, seg=seg)
    return images, seg


def test_napscaf_slice_dataset_reproduces_legacy_slice_selection(tmp_path: Path) -> None:
    import random

    import numpy as np

    from mimose.datasets.napscaf_slices import NAPSCAFSliceDataset

    _write_case(tmp_path, "case_a", tumor_slices=[1, 2, 3, 4, 5, 6, 7, 8])
    _write_case(tmp_path, "case_b", tumor_slices=[2, 4, 6], empty_modality_slice=4)
    case_c_images, case_c_seg = _write_case(tmp_path, "case_c", tumor_slices=[0, 1, 2, 3, 4, 5, 9])

    # Legacy: one random.seed(7) for the whole run, sorted cases, sample only when > 5 candidates.
    rng = random.Random(7)
    expected = {
        "case_a": sorted(rng.sample([1, 2, 3, 4, 5, 6, 7, 8], 5)),
        "case_b": [2, 6],
        "case_c": sorted(rng.sample([0, 1, 2, 3, 4, 5, 9], 5)),
    }

    split = [{"sub": name} for name in ("case_a", "case_b", "case_c")]
    dataset = NAPSCAFSliceDataset(tmp_path, split, slices_per_case=5, seed=7, image_size=16)
    index = __import__("json").loads((dataset.cache_dir / "index.json").read_text())
    got: dict[str, list[int]] = {}
    for case, z in zip(index["cases"], index["z"]):
        got.setdefault(case, []).append(z)
    assert got == expected

    item = dataset[len(dataset) - 1]  # last slice of case_c
    z = expected["case_c"][-1]
    assert item["sub"] == "case_c"
    np.testing.assert_allclose(item["images"].numpy(), case_c_images[..., z], atol=1e-6)
    np.testing.assert_array_equal(item["seg"].numpy(), case_c_seg[0, ..., z])
    assert item["mask"].all()

    only_b = NAPSCAFSliceDataset(tmp_path, [{"sub": "case_b"}], slices_per_case=5, seed=7, image_size=32)
    assert len(only_b) == 2
    assert tuple(only_b[0]["images"].shape) == (4, 32, 32)
    assert tuple(only_b[0]["seg"].shape) == (32, 32)


def test_napscaf_inference_size_resizes_slices_and_returns_native_grid() -> None:
    model = NAPSCAF(num_cls=4, pretrained_prior=False, inference_size=64).eval()
    images = torch.rand(1, 4, 40, 40, 2)
    mask = torch.ones(1, 4, dtype=torch.bool)
    with torch.no_grad():
        torch.manual_seed(0)
        out = model.predict(images, mask)
        slices = images.permute(0, 4, 1, 2, 3).reshape(2, 4, 40, 40)
        torch.manual_seed(0)
        logits = model.forward_slices(torch.nn.functional.interpolate(slices, size=(64, 64), mode="bilinear"), mask.expand(2, -1))
        expected = torch.nn.functional.interpolate(logits, size=(40, 40), mode="bilinear").softmax(dim=1)
    assert tuple(out.shape) == (1, 4, 40, 40, 2)
    torch.testing.assert_close(out, expected.reshape(1, 2, 4, 40, 40).permute(0, 2, 3, 4, 1))
    with pytest.raises(ValueError, match="multiple of 32"):
        NAPSCAF(num_cls=4, pretrained_prior=False, inference_size=250)


def test_napscaf_trainer_exports_best_checkpoint(tmp_path: Path) -> None:
    from mimose.training.trainers import NAPSCAFTrainer
    from mimose.training.trainers.mstkdnet import MSTKDTrainer

    trainer = NAPSCAFTrainer.__new__(NAPSCAFTrainer)
    trainer.model = torch.nn.Linear(2, 2)
    trainer.device = torch.device("cpu")
    trainer.checkpoint_dir = tmp_path
    trainer._model_for_state = lambda: trainer.model  # type: ignore[method-assign]
    best = {k: torch.full_like(v, 7.0) for k, v in trainer.model.state_dict().items()}
    torch.save({"state_dict": best}, tmp_path / "best.pth")
    seen = {}
    original = MSTKDTrainer.save_final_checkpoint
    def fake_save(self):
        seen["w"] = self.model.weight.clone()
        return tmp_path

    MSTKDTrainer.save_final_checkpoint = fake_save  # type: ignore[method-assign]
    try:
        trainer.final_from_best = False
        trainer.save_final_checkpoint()
        assert not torch.all(seen.pop("w") == 7.0)
        trainer.final_from_best = True
        trainer.save_final_checkpoint()
        assert torch.all(seen.pop("w") == 7.0)
    finally:
        MSTKDTrainer.save_final_checkpoint = original  # type: ignore[method-assign]


def test_napscaf_fp32_validation_disables_autocast_only_during_validation() -> None:
    from mimose.training.trainers import NAPSCAFTrainer
    from mimose.training.trainers.mstkdnet import MSTKDTrainer

    trainer = NAPSCAFTrainer.__new__(NAPSCAFTrainer)
    trainer.amp_enabled = True
    seen = []
    original = MSTKDTrainer.val_epoch

    def fake_val(self, epoch):
        seen.append(self.amp_enabled)
        return {}

    MSTKDTrainer.val_epoch = fake_val  # type: ignore[method-assign]
    try:
        trainer.fp32_validation = False
        trainer.val_epoch(0)
        trainer.fp32_validation = True
        trainer.val_epoch(0)
    finally:
        MSTKDTrainer.val_epoch = original  # type: ignore[method-assign]
    assert seen == [True, False]
    assert trainer.amp_enabled is True
