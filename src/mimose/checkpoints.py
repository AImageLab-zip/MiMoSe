from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_model, save_model


def save_weights_only_checkpoint(
    model: torch.nn.Module,
    destination: str | Path,
) -> Path:
    checkpoint_path = Path(destination)
    # Some architectures (e.g. TinyMimosa's nnU-Net-style backbone) alias the
    # same parameter under multiple state_dict keys (shared modules). Plain
    # save_file() rejects that; save_model() detects and dedupes shared
    # tensors, and load_model() reconstructs the aliasing on load.
    # safetensors refuses non-contiguous tensors, which channels_last_3d weights
    # are (see MIMOSE_CHANNELS_LAST in base_trainer): save a contiguous layout,
    # then restore the training layout.
    channels_last = any(
        p.dim() == 5 and not p.is_contiguous() and p.is_contiguous(memory_format=torch.channels_last_3d)
        for p in model.parameters()
    )
    if channels_last:
        model.to(memory_format=torch.contiguous_format)
    try:
        save_model(model, str(checkpoint_path))
    finally:
        if channels_last:
            model.to(memory_format=torch.channels_last_3d)
    return checkpoint_path


def load_weights_only_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> torch.nn.Module:
    path = Path(checkpoint_path)
    if path.suffix != ".safetensors":
        raise ValueError(
            f"Expected a .safetensors weights checkpoint, got: {path}"
        )
    load_model(model, str(path), strict=strict, device=str(device))
    return model


def assert_finite_weights(model: torch.nn.Module, checkpoint_path: str | Path) -> None:
    """Refuse checkpoints whose weights diverged: NaN logits silently argmax to background."""
    bad = [
        name
        for name, tensor in model.state_dict().items()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())
    ]
    if bad:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has non-finite weights in {len(bad)} tensors "
            f"(e.g. {bad[0]}); training most likely diverged."
        )
