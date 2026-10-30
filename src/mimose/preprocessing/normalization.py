import numpy as np
from argparse import Namespace
import click

from mimose.preprocessing.config import NormConfig
def check_size(images:np.ndarray):
    if len(images.shape) != 4:
        raise click.ClickException(
            f"`images` must have 4 dimensions, got shape {images.shape}."
        )
def template(images:np.ndarray,config:NormConfig)-> np.ndarray:
    """Template function for intensity normalization on multi-modal 3D images.

    Args:
        images: Input image array with shape `(C, X, Y, Z)`, where `C` is the
            modality/channel dimension.
        config: Normalization configuration controlling the normalization behavior.

    Returns:
        The normalized image array with the same shape as `images`.
    """
    # implement your normalization logic here
    raise NotImplementedError()
    # return out_image
def subject_zscore(images:np.ndarray,config:NormConfig)-> np.ndarray:
    check_size(images)
    out_image = np.zeros_like(images)
    for i in range(images.shape[0]):
        modal = images[i]
        foreground_mask = modal > 0
        foreground = modal[foreground_mask]

        if foreground.size == 0:
            out_image[i] = modal
            continue

        mean = np.mean(foreground)
        std = np.std(foreground)
        if not np.isfinite(std) or std == 0:
            out_image[i] = np.zeros_like(modal)
            continue

        out_image[i] = (modal - mean) / std
    return out_image



def min_max(images: np.ndarray, config: NormConfig) -> np.ndarray:
    check_size(images)

    target_min = config.min_max_range[0]
    target_max = config.min_max_range[1]

    out = images.copy()

    for i in range(out.shape[0]):
        single_modal = out[i]
        source_min = np.min(single_modal)
        source_max = np.max(single_modal)

        if source_max == source_min:
            raise click.ClickException(
                f"Cannot apply min-max normalization to modality {i}: "
                f"all values are equal to {source_min}."
            )

        out[i] = (
            (single_modal - source_min) / (source_max - source_min)
        ) * (target_max - target_min) + target_min

    return out

def dataset_zscore(images: np.ndarray, config: NormConfig) -> np.ndarray:
    check_size(images)
    out_image = np.zeros_like(images)
    for i in range(images.shape[0]):
        foreground = images[i][images[i] > 0]
        out_image[i] = (images[i] - config.mean[i]) / config.std[i]
    return out_image

def none(images: np.ndarray, config: NormConfig) -> np.ndarray:
    print("⚠️ Be careful: you are running preprocessing without any normalization.")
    return images


def irm_min_max(images: np.ndarray, config: NormConfig) -> np.ndarray:
    """Legacy M3AE ``irm_min_max_preprocess`` (``legacy/m3ae/dataset/image_utils.py``), per modality:
    clip to the 1st/99th percentile of the positive voxels, then min-max scale to [0, 1]
    (background lands on 0). Use with ``clamp_mode: none``."""
    check_size(images)
    out = images.astype(np.float32, copy=True)
    for i in range(out.shape[0]):
        modal = out[i]
        foreground = modal[modal > 0]
        if foreground.size == 0:
            continue
        low, high = np.percentile(foreground, [1, 99])
        modal = np.clip(modal, low, high)
        scale = modal.max() - modal.min()
        out[i] = (modal - modal.min()) / scale if scale > 0 else np.zeros_like(modal)
    return out


def slice_percentile_uint8(images: np.ndarray, config: NormConfig) -> np.ndarray:
    """Legacy NAP-SCAF ``normalize_to_uint8`` (``legacy/NAP-SCAF/scripts/preprocess_brats_png.py``),
    per modality and per axial (last-axis) slice: clip to the 1st/99th percentile of *all* the slice's
    pixels (background included), scale to [0, 255], truncate to uint8, then divide by 255 as the
    legacy PNG loader does. A slice with p99 <= p1 becomes all zeros. Use with ``crop_mode: none``
    and ``clamp_mode: none`` so the percentiles see the full 240x240 slice, as in legacy."""
    check_size(images)
    out = np.zeros(images.shape, dtype=np.float32)
    for i in range(images.shape[0]):
        for z in range(images.shape[-1]):
            plane = images[i, ..., z].astype(np.float64)
            p1, p99 = np.percentile(plane, (1, 99))
            if p99 <= p1:
                continue
            plane = np.clip(plane, p1, p99)
            out[i, ..., z] = ((plane - p1) / (p99 - p1) * 255.0).astype(np.uint8) / np.float32(255.0)
    return out


def subject_zscore_nonzero(images: np.ndarray, config: NormConfig) -> np.ndarray:
    """Legacy M3AE/MCPL ``zscore_normalise`` (``legacy/MCPL/dataset/image_utils.py``), per modality:
    z-score the nonzero voxels with their own mean/std and leave the background at 0."""
    check_size(images)
    out = images.astype(np.float32, copy=True)
    for i in range(out.shape[0]):
        modal = out[i]
        nonzero = modal != 0
        if not nonzero.any():
            continue
        values = modal[nonzero]
        std = values.std()
        modal[nonzero] = (values - values.mean()) / std if std > 0 else 0.0
    return out


def volume_zscore(images: np.ndarray, config: NormConfig) -> np.ndarray:
    """Legacy ALoRA-MoE per-channel z-score (``lora_moe_hgd/datasets.py``), per modality:
    ``(x - mean) / (std + 1e-6)`` over *all* voxels of the volume, background included.
    Use with ``crop_mode: none`` so the statistics see the whole volume, as in legacy."""
    check_size(images)
    out = images.astype(np.float32, copy=True)
    for i in range(out.shape[0]):
        modal = out[i]
        # torch.Tensor.std is unbiased.
        out[i] = (modal - modal.mean()) / (modal.std(ddof=1) + 1e-6)
    return out
