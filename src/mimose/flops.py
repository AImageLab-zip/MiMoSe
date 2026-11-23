from __future__ import annotations

import csv
import math
import statistics
import time
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from rich.console import Console, Group
from rich.markup import escape as _rich_escape
from rich.table import Table

from mimose.checkpoints import load_weights_only_checkpoint
from mimose.models.config import build_model_config
from mimose.utils.cli_overrides import CONFIGS_DIR, load_yaml_config

# The 15 non-empty modality-presence combinations exercised at test time (see
# mimose.testing.pipeline.MASKS -- duplicated here, rather than imported,
# so that `mimose flops` doesn't pull in that module's much heavier
# dependencies (medpy, pandas, click, dataset/checkpoint loading) just to
# profile a freshly constructed model). Keep in sync with MASKS there.
MODALITY_COMBINATIONS: tuple[tuple[bool, bool, bool, bool], ...] = (
    (False, False, False, True),
    (False, True, False, False),
    (False, False, True, False),
    (True, False, False, False),
    (False, True, False, True),
    (False, True, True, False),
    (True, False, True, False),
    (False, False, True, True),
    (True, False, False, True),
    (True, True, False, False),
    (True, True, True, False),
    (True, False, True, True),
    (True, True, False, True),
    (False, True, True, True),
    (True, True, True, True),
)
MODALITY_NAMES = ("t1c", "t1n", "t2f", "t2w")


@dataclass(frozen=True)
class PreparedInput:
    name: str
    images: torch.Tensor | None
    input_shape: tuple[int, ...] | None
    note: str = ""


@dataclass(frozen=True)
class ModalityMeasurement:
    modalities: str
    mask: tuple[bool, ...]
    macs: float
    flops: float
    peak_memory_bytes: float | None
    latency_seconds: float | None
    latency_std_seconds: float | None
    kernel_launches: int | None
    peak_reserved_memory_bytes: float | None


@dataclass(frozen=True)
class PathMeasurement:
    name: str
    input_shape: tuple[int, ...] | None
    params: int
    per_modality: tuple[ModalityMeasurement, ...]
    note: str = ""


@dataclass(frozen=True)
class FlopsReport:
    model_name: str
    model_class_name: str
    device: str
    sample_path: str
    sample_shape: tuple[int, ...]
    patch_size: int | None
    measurements: tuple[PathMeasurement, ...]
    #: Parameter budget this report was realized for, when profiling a
    #: budget-conditioned model. ``None`` for ordinary models.
    budget: float | None = None
    #: Per-stage widths of the standalone network the budget selected.
    realized_widths: tuple[int, ...] | None = None


class _FlopsWrapper(nn.Module):
    def __init__(self, model: nn.Module, *, use_predict: bool) -> None:
        super().__init__()
        self.model = model
        self.use_predict = use_predict

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.use_predict:
            return self.model.predict(images, mask)  # type: ignore[attr-defined]
        return self.model(images, mask)


def realize_for_budget(model: nn.Module, budget: float) -> nn.Module:
    """Materialize the standalone network a budget-conditioned model selects.

    This is the difference between measuring an architecture and measuring the
    supernet that contains it.  Setting a budget only *masks* channels: the
    forward pass still computes at full width and multiplies the unselected
    channels by zero, so parameters, FLOPs, peak memory and latency all come
    out as the widest model's numbers no matter which budget is set.  Only the
    extracted network has the resource profile the Pareto plots need.
    """
    if not hasattr(model, "extract_for_budget"):
        raise ValueError(
            f"{type(model).__name__} is not budget-conditioned; --budget only "
            "applies to models exposing extract_for_budget (e.g. ElasticUnet)"
        )
    model.set_budget(budget)
    return model.extract_for_budget(budget)


def budget_range_for_config(
    *,
    config_path: Path | None,
    model_name: str | None,
    custom_model_kwargs: dict[str, Any] | None,
) -> tuple[float, float]:
    """The budget range a config's model can express.

    The model is instantiated rather than derived from ``features_per_stage``
    so that an explicit ``budget_min``/``budget_max`` in ``custom_model_kwargs``
    is honoured instead of silently recomputed.
    """
    yaml_config = load_yaml_config(config_path)
    model_config = build_model_config(
        model_kind=resolve_model_name(yaml_config, model_name),
        model_kwargs=merge_model_kwargs(yaml_config, custom_model_kwargs),
    )
    model = model_config.model_class(**model_config.kwargs)
    gates = getattr(model, "gates", None)
    if gates is None:
        raise ValueError(
            f"{model_config.model_class.__name__} is not budget-conditioned; "
            "--budget-sweep only applies to models such as ElasticUnet"
        )
    return float(gates.budget_min), float(gates.budget_max)


def log_spaced_budgets(budget_min: float, budget_max: float, count: int) -> list[float]:
    """``count`` budgets spaced evenly in log space, endpoints included."""
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    if count == 1:
        return [budget_max]
    log_min, log_max = math.log(budget_min), math.log(budget_max)
    step = (log_max - log_min) / (count - 1)
    return [math.exp(log_min + step * i) for i in range(count)]


def run_flops_analysis(
    *,
    config_path: Path | None,
    model_name: str | None,
    custom_model_kwargs: dict[str, Any] | None,
    device: str = "auto",
    warmup: int = 0,
    budget: float | None = None,
    checkpoint: Path | None = None,
    precision: str = "fp32",
    latency_only: bool = False,
    full_modality_only: bool = False,
    half: bool = False,
) -> FlopsReport:
    """latency_only: time the whole-volume pass only -- no ptflops MAC
    counting (reported as NaN), no kernel/reserved-memory profiling, and no
    single-patch pass for patch-based models. Meant for slow CPU timing,
    where MACs are already known from a GPU run.
    full_modality_only: measure only the all-modalities-present combination
    instead of all 15.
    half: fp16 inference -- model weights and inputs converted to float16
    (model.half()) in channels_last_3d, run under fp16 autocast so models that
    create fp32 tensors internally still work. CUDA only."""
    if half and precision != "fp32":
        raise ValueError("half=True already runs pure fp16; leave precision at fp32 (no autocast)")
    yaml_config = load_yaml_config(config_path)
    resolved_model_name = resolve_model_name(yaml_config, model_name)
    model_kwargs = merge_model_kwargs(yaml_config, custom_model_kwargs)
    model_config = build_model_config(
        model_kind=resolved_model_name,
        model_kwargs=model_kwargs,
    )
    torch_device = resolve_device(device)
    model = model_config.model_class(**model_config.kwargs).to(torch_device)
    model.eval()
    if hasattr(model, "is_training"):
        model.is_training = False
    if hasattr(model, "allow_template_padding"):
        # M3AE: let its learned missing-modality template cover the shared benchmark volume
        # (cost-neutral; see M3AE.allow_template_padding).
        model.allow_template_padding = True

    if checkpoint is not None:
        # Gate thresholds move during training, so which architecture a budget
        # selects before and after training are different networks. Profiling
        # the freshly-built model would measure the analytic calibration rather
        # than what was actually learned.
        load_weights_only_checkpoint(
            model, checkpoint, device=torch_device, strict=False
        )

    realized_widths: tuple[int, ...] | None = None
    if budget is not None:
        if hasattr(model, "current_widths"):
            model.set_budget(budget)
            realized_widths = tuple(model.current_widths())
        # Replace the supernet with the standalone network *before* the input
        # shape is probed and parameters are counted, so everything downstream
        # describes the realized architecture.
        model = realize_for_budget(model, budget).to(torch_device)
        model.eval()
        if hasattr(model, "is_training"):
            model.is_training = False

    sample_path, dummy_images = load_dummy_brain()
    dummy_images = dummy_images.to(device=torch_device, dtype=torch.float32)
    patch_size = detect_input_patch_size(model)
    tile_shape = detect_input_tile_shape(model)
    prepared_inputs = prepare_measurement_inputs(
        dummy_images,
        patch_size=patch_size,
        tile_shape=tile_shape,
        device=torch_device,
    )
    if half:
        if torch_device.type != "cuda":
            raise ValueError("half=True (pure fp16) is only supported on CUDA")
        # channels_last_3d is required, not cosmetic: with the default NCDHW
        # layout, fp16/bf16 Conv3d bypasses cuDNN and falls back to the generic
        # vol2col+GEMM path (torch 2.9 + cuDNN 9.10), ~3x SLOWER than fp32
        # instead of ~3x faster. With the weights in channels_last_3d, cuDNN is
        # used, and sliding-window patches sliced from the input follow the
        # weights' layout.
        model = model.half().to(memory_format=torch.channels_last_3d)
        prepared_inputs = tuple(
            PreparedInput(
                name=p.name,
                images=(
                    p.images.half().contiguous(memory_format=torch.channels_last_3d)
                    if p.images is not None
                    else None
                ),
                input_shape=p.input_shape,
                note=p.note,
            )
            for p in prepared_inputs
        )
    params = count_parameters(model)
    combinations = ((True, True, True, True),) if full_modality_only else MODALITY_COMBINATIONS
    patch_based = hasattr(model, "predict") and any(
        p.name == "predict" and p.images is not None for p in prepared_inputs
    )
    measurements: list[PathMeasurement] = []
    for prepared in prepared_inputs:
        if latency_only and patch_based and prepared.name == "forward":
            measurements.append(
                PathMeasurement(
                    name=prepared.name,
                    input_shape=prepared.input_shape,
                    params=params,
                    per_modality=(),
                    note="skipped: --latency-only (whole volume only)",
                )
            )
            continue
        if prepared.images is None:
            measurements.append(
                PathMeasurement(
                    name=prepared.name,
                    input_shape=prepared.input_shape,
                    params=params,
                    per_modality=(),
                    note=prepared.note,
                )
            )
            continue

        use_predict = prepared.name == "predict"
        if use_predict and not hasattr(model, "predict"):
            measurements.append(
                PathMeasurement(
                    name=prepared.name,
                    input_shape=prepared.input_shape,
                    params=params,
                    per_modality=(),
                    note="skipped: model does not implement predict(images, mask)",
                )
            )
            continue

        per_modality = []
        for combo in combinations:
            mask = modality_mask(combo, batch_size=prepared.images.size(0), device=torch_device)
            # fp16/bf16 profile inference under torch.autocast (CUDA only), the
            # same way `mimose test --precision` runs it; fp32 leaves it off.
            # half=True also runs under fp16 autocast: some models (UNET-MFI,
            # ElasticUNet/RDoll-UNet) create fp32 tensors internally, which
            # crash against fp16 weights; autocast casts them per op. The
            # weights are already fp16, so no per-call weight casts are added.
            with _autocast_for("fp16" if half else precision, torch_device):
                if latency_only:
                    macs, peak_memory_bytes = float("nan"), None
                else:
                    macs, peak_memory_bytes = measure_pass(
                        model,
                        images=prepared.images,
                        mask=mask,
                        use_predict=use_predict,
                        device=torch_device,
                    )
                latency_seconds, latency_std_seconds = measure_latency(
                    model,
                    images=prepared.images,
                    mask=mask,
                    use_predict=use_predict,
                    device=torch_device,
                    warmup=warmup,
                )
                if latency_only:
                    kernel_launches, peak_reserved_memory_bytes = None, None
                else:
                    kernel_launches, peak_reserved_memory_bytes = measure_kernel_launches_and_memory(
                        model,
                        images=prepared.images,
                        mask=mask,
                        use_predict=use_predict,
                        device=torch_device,
                        warmup=warmup,
                    )
                per_modality.append(
                    ModalityMeasurement(
                        modalities=modality_label(combo),
                        mask=combo,
                        macs=macs,
                        flops=macs * 2,
                        peak_memory_bytes=peak_memory_bytes,
                        latency_seconds=latency_seconds,
                        latency_std_seconds=latency_std_seconds,
                        kernel_launches=kernel_launches,
                        peak_reserved_memory_bytes=peak_reserved_memory_bytes,
                    )
                )

        measurements.append(
            PathMeasurement(
                name=prepared.name,
                input_shape=prepared.input_shape,
                params=params,
                per_modality=tuple(per_modality),
            )
        )

    return FlopsReport(
        model_name=resolved_model_name,
        model_class_name=type(model).__name__,
        device=str(torch_device),
        sample_path=str(sample_path),
        sample_shape=tuple(dummy_images.shape),
        patch_size=patch_size,
        measurements=tuple(measurements),
        budget=budget,
        realized_widths=realized_widths,
    )


def resolve_model_name(
    yaml_config: dict[str, Any],
    cli_model_name: str | None,
) -> str:
    model_name = (
        cli_model_name
        if cli_model_name is not None
        else yaml_config.get("model")
    )
    if model_name is None or not str(model_name).strip():
        raise typer.BadParameter(
            "provide --model or set 'model' in --config",
            param_hint="--model",
        )
    return str(model_name).strip()


def resolve_2023_config_paths(configs_dir: Path = CONFIGS_DIR) -> tuple[Path, ...]:
    config_paths = tuple(sorted(configs_dir.glob("*_23.yaml")))
    if not config_paths:
        raise typer.BadParameter(
            f"no 2023 configs found under {configs_dir}",
            param_hint="--all",
        )
    return config_paths


def validate_flops_selection(
    *,
    all_configs: bool,
    config_path: Path | None,
    model_name: str | None,
) -> None:
    if not all_configs:
        return
    if config_path is not None or model_name is not None:
        raise typer.BadParameter(
            "--all cannot be combined with --config or --model",
            param_hint="--all",
        )


def merge_model_kwargs(
    yaml_config: dict[str, Any],
    cli_model_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    model_kwargs = dict(yaml_config.get("custom_model_kwargs") or {})
    model_kwargs.update(cli_model_kwargs or {})
    return model_kwargs


def resolve_device(device: str) -> torch.device:
    normalized = device.strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise typer.BadParameter(
            "device must be one of: auto, cpu, cuda",
            param_hint="--device",
        )
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise typer.BadParameter(
            "CUDA was requested but is not available",
            param_hint="--device",
        )
    return torch.device(normalized)


def load_dummy_brain() -> tuple[Path, torch.Tensor]:
    resource = files("mimose").joinpath("data/dummy_brain.npz")
    with as_file(resource) as sample_path:
        if not sample_path.is_file():
            raise RuntimeError(f"dummy brain sample not found at {sample_path}")
        with np.load(sample_path) as data:
            if "images" not in data:
                raise RuntimeError(f"{sample_path} must contain an 'images' array")
            images = np.asarray(data["images"], dtype=np.float32)

    if images.ndim == 4:
        images = images[None, ...]
    if images.ndim != 5:
        raise RuntimeError(
            "dummy brain images must have shape [M, H, W, D] or [B, M, H, W, D], "
            f"got {tuple(images.shape)}"
        )
    return sample_path, torch.from_numpy(np.ascontiguousarray(images))


def detect_input_patch_size(model: nn.Module) -> int | None:
    value = getattr(model, "input_patch_size", None)
    if value is None:
        module = __import__(model.__class__.__module__, fromlist=["input_patch_size"])
        value = getattr(module, "input_patch_size", None)
    if value is None:
        return None
    try:
        patch_size = int(value)
    except (TypeError, ValueError):
        return None
    return patch_size if patch_size > 0 else None


def detect_input_tile_shape(model: nn.Module) -> tuple[int, int, int] | None:
    value = getattr(model, "tile_shape", None)
    if value is None:
        return None
    try:
        dims = tuple(int(dim) for dim in value)
    except (TypeError, ValueError):
        return None
    if len(dims) != 3 or any(dim <= 0 for dim in dims):
        return None
    return dims


def prepare_measurement_inputs(
    images: torch.Tensor,
    *,
    patch_size: int | None,
    tile_shape: tuple[int, int, int] | None = None,
    device: torch.device,
) -> tuple[PreparedInput, ...]:
    images = images.to(device=device, dtype=torch.float32)
    if patch_size is None and tile_shape is None:
        return (
            PreparedInput(
                name="forward",
                images=images,
                input_shape=tuple(images.shape),
            ),
            PreparedInput(
                name="predict",
                images=None,
                input_shape=None,
                note="skipped: no input_patch_size detected",
            ),
        )

    forward_spatial_size = (
        (patch_size, patch_size, patch_size) if patch_size is not None else tile_shape
    )
    patch_images = center_crop_or_pad_volume(images, forward_spatial_size)
    return (
        PreparedInput(
            name="forward",
            images=patch_images,
            input_shape=tuple(patch_images.shape),
        ),
        PreparedInput(
            name="predict",
            images=images,
            input_shape=tuple(images.shape),
        ),
    )


def modality_mask(
    combination: tuple[bool, ...],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(combination, dtype=torch.bool, device=device).unsqueeze(0).expand(
        batch_size, -1
    )


def modality_label(combination: tuple[bool, ...]) -> str:
    present = [name for name, enabled in zip(MODALITY_NAMES, combination) if enabled]
    return "+".join(present)


def center_crop_or_pad_volume(
    images: torch.Tensor,
    spatial_size: tuple[int, int, int],
) -> torch.Tensor:
    if images.ndim != 5:
        raise ValueError(f"expected [B, M, H, W, D], got {tuple(images.shape)}")

    slices: list[slice] = [slice(None), slice(None)]
    for current, target in zip(images.shape[-3:], spatial_size, strict=True):
        if current <= target:
            slices.append(slice(None))
            continue
        start = (current - target) // 2
        slices.append(slice(start, start + target))

    cropped = images[tuple(slices)]
    pad: list[int] = []
    spatial_pairs = list(zip(cropped.shape[-3:], spatial_size, strict=True))
    for current, target in reversed(spatial_pairs):
        missing = max(target - current, 0)
        before = missing // 2
        after = missing - before
        pad.extend((before, after))
    if any(pad):
        cropped = F.pad(cropped, pad)
    return cropped


def measure_pass(
    model: nn.Module,
    *,
    images: torch.Tensor,
    mask: torch.Tensor,
    use_predict: bool,
    device: torch.device,
) -> tuple[float, float | None]:
    """Run one forward/predict pass and return (MACs, peak allocated CUDA bytes).

    Peak memory is only meaningful on CUDA (there is no cheap, reliable
    equivalent for CPU tensor allocations), so it's None on CPU.
    """
    try:
        from ptflops import get_model_complexity_info
    except ImportError as exc:
        raise RuntimeError(
            "ptflops>=0.7.5 is required for 'mimose flops'. "
            "Install project dependencies again before running this command."
        ) from exc

    wrapper = _FlopsWrapper(model, use_predict=use_predict)
    wrapper.eval()

    def input_constructor(_: tuple[int, ...]) -> dict[str, torch.Tensor]:
        return {"images": images, "mask": mask}

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        macs, _ = get_model_complexity_info(
            wrapper,
            tuple(images.shape[1:]),
            input_constructor=input_constructor,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
            backend="aten",
        )

    peak_memory_bytes = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_bytes = float(torch.cuda.max_memory_allocated(device))

    return float(macs), peak_memory_bytes


def _autocast_for(precision: str, device: torch.device):
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"precision must be fp32, fp16 or bf16, got {precision!r}")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=precision != "fp32" and device.type == "cuda",
    )


def measure_latency(
    model: nn.Module,
    *,
    images: torch.Tensor,
    mask: torch.Tensor,
    use_predict: bool,
    device: torch.device,
    warmup: int = 5,
    repeats: int = 3,
) -> tuple[float, float]:
    """(mean, std) wall-clock time (seconds) of one forward/predict call, over
    `repeats` timed calls after `warmup` untimed ones. Run separately from
    measure_pass, since ptflops's op-counting hooks add overhead that would
    make that call's own timing unrepresentative of real inference cost.
    Execution time varies run to run (more so on CPU, where there's no
    exclusive-device guarantee like on a dedicated CUDA device), so the std
    is reported alongside the mean rather than discarded."""
    wrapper = _FlopsWrapper(model, use_predict=use_predict)
    wrapper.eval()

    with torch.no_grad():
        for _ in range(warmup):
            wrapper(images, mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            wrapper(images, mask)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - start)

    mean = statistics.fmean(timings)
    std = statistics.stdev(timings) if len(timings) > 1 else 0.0
    return mean, std


def measure_kernel_launches_and_memory(
    model: nn.Module,
    *,
    images: torch.Tensor,
    mask: torch.Tensor,
    use_predict: bool,
    device: torch.device,
    warmup: int = 0,
) -> tuple[int | None, float | None]:
    """(kernel launch count, peak reserved CUDA memory bytes) for one pass, or
    (None, None) on CPU -- neither concept applies there: there's no discrete
    kernel-launch model for CPU ops, and torch.cuda.max_memory_reserved has no
    CPU equivalent.

    Peak *reserved* memory (the allocator's cached/reserved footprint) is a
    better proxy for device memory pressure than measure_pass's peak
    *allocated* memory: the gap between the two reflects allocator
    fragmentation/caching overhead that allocated-only figures hide.

    Profiled via torch.profiler on a clean call, run separately from
    measure_pass so ptflops's own op-counting hooks don't add extra kernel
    launches or memory traffic that would skew either number.
    """
    if device.type != "cuda":
        return None, None

    wrapper = _FlopsWrapper(model, use_predict=use_predict)
    wrapper.eval()

    with torch.no_grad():
        for _ in range(warmup):
            wrapper(images, mask)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            wrapper(images, mask)
        torch.cuda.synchronize(device)

    kernel_launches = sum(
        1
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    )
    peak_reserved_memory_bytes = float(torch.cuda.max_memory_reserved(device))
    return kernel_launches, peak_reserved_memory_bytes


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _mac_stats(per_modality: tuple[ModalityMeasurement, ...]) -> dict[str, tuple[float, float]] | None:
    """{"Mean": (macs, flops), "Max": (...), "Min": (...)} or None if empty."""
    if not per_modality:
        return None
    macs_values = [m.macs for m in per_modality]
    return {
        "Mean": (statistics.fmean(macs_values), statistics.fmean(macs_values) * 2),
        "Max": (max(macs_values), max(macs_values) * 2),
        "Min": (min(macs_values), min(macs_values) * 2),
    }


def _memory_stats(per_modality: tuple[ModalityMeasurement, ...]) -> dict[str, float] | None:
    values = [m.peak_memory_bytes for m in per_modality if m.peak_memory_bytes is not None]
    if not values:
        return None
    return {"Mean": statistics.fmean(values), "Max": max(values), "Min": min(values)}


def _reserved_memory_stats(
    per_modality: tuple[ModalityMeasurement, ...]
) -> dict[str, float] | None:
    values = [
        m.peak_reserved_memory_bytes
        for m in per_modality
        if m.peak_reserved_memory_bytes is not None
    ]
    if not values:
        return None
    return {"Mean": statistics.fmean(values), "Max": max(values), "Min": min(values)}


def _kernel_stats(per_modality: tuple[ModalityMeasurement, ...]) -> dict[str, float] | None:
    values = [m.kernel_launches for m in per_modality if m.kernel_launches is not None]
    if not values:
        return None
    return {"Mean": statistics.fmean(values), "Max": max(values), "Min": min(values)}


def _time_stats(per_modality: tuple[ModalityMeasurement, ...]) -> dict[str, float] | None:
    values = [m.latency_seconds for m in per_modality if m.latency_seconds is not None]
    if not values:
        return None
    return {
        "Mean": statistics.fmean(values),
        "Std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "Max": max(values),
        "Min": min(values),
    }


def _time_std_stat(per_modality: tuple[ModalityMeasurement, ...]) -> float | None:
    """Mean, across the 15 modality combinations, of each combination's own
    run-to-run latency std (see measure_latency) -- a representative figure
    for how much a single latency measurement here typically varies."""
    values = [m.latency_std_seconds for m in per_modality if m.latency_std_seconds is not None]
    if not values:
        return None
    return statistics.fmean(values)


def pass_scope(report: FlopsReport, path: PathMeasurement) -> str:
    """Classify a measurement pass as a single "patch" forward (one
    fixed-size patch/tile through the network) or a "whole_volume" one
    (the full input, whether processed directly or aggregated by a
    model's own sliding-window predict()).

    A model is patch-based iff it has a non-skipped "predict" pass: its
    "forward" pass is then the single-patch/tile cost and "predict" is the
    whole-volume cost. Models without a "predict" pass consume the whole
    volume directly in "forward", so that pass IS the whole-volume cost.
    """
    has_predict = any(
        p.name == "predict" and not p.note for p in report.measurements
    )
    if path.name == "predict":
        return "whole_volume"
    if path.name == "forward" and not has_predict:
        return "whole_volume"
    return "patch"


_SCOPE_LABELS = {"patch": "single patch/tile", "whole_volume": "whole volume"}


def build_flops_table(report: FlopsReport) -> Group:
    tables = []
    for path in report.measurements:
        scope_label = _SCOPE_LABELS[pass_scope(report, path)]
        title = _rich_escape(
            f"{report.model_name} ({report.model_class_name}) — {path.name} pass "
            f"[{scope_label}], device={report.device}, "
            f"input={_format_shape(path.input_shape)}, "
            f"params={_format_count(float(path.params))}"
        )
        table = Table(title=title)
        table.add_column("Modalities", style="bold cyan")
        table.add_column("MACs", justify="right")
        table.add_column("FLOPs", justify="right")
        table.add_column("Peak Memory", justify="right")
        table.add_column("Reserved Memory", justify="right")
        table.add_column("Kernel Launches", justify="right")
        table.add_column("Latency", justify="right")

        if path.note:
            table.add_column("Note")
            table.add_row("-", "-", "-", "-", "-", "-", "-", path.note)
            tables.append(table)
            continue

        mac_stats = _mac_stats(path.per_modality)
        memory_stats = _memory_stats(path.per_modality)
        reserved_memory_stats = _reserved_memory_stats(path.per_modality)
        kernel_stats = _kernel_stats(path.per_modality)
        time_stats = _time_stats(path.per_modality)
        time_std_stat = _time_std_stat(path.per_modality)

        for measurement in path.per_modality:
            table.add_row(
                measurement.modalities,
                _format_count(measurement.macs),
                _format_count(measurement.flops),
                _format_bytes(measurement.peak_memory_bytes),
                _format_bytes(measurement.peak_reserved_memory_bytes),
                _format_count(measurement.kernel_launches),
                _format_seconds(measurement.latency_seconds, measurement.latency_std_seconds),
            )

        if mac_stats is not None:
            table.add_section()
            for label in ("Mean", "Max", "Min"):
                macs, flops = mac_stats[label]
                memory = memory_stats[label] if memory_stats is not None else None
                reserved_memory = (
                    reserved_memory_stats[label] if reserved_memory_stats is not None else None
                )
                kernel_launches = kernel_stats[label] if kernel_stats is not None else None
                latency = time_stats[label] if time_stats is not None else None
                latency_std = time_std_stat if label == "Mean" else None
                table.add_row(
                    label,
                    _format_count(macs),
                    _format_count(flops),
                    _format_bytes(memory),
                    _format_bytes(reserved_memory),
                    _format_count(kernel_launches),
                    _format_seconds(latency, latency_std),
                    style="bold",
                )

        tables.append(table)

    return Group(*tables)


def write_flops_report(report: FlopsReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        Console(file=f, width=120, no_color=True).print(build_flops_table(report))


def write_flops_summary_csv(report: FlopsReport, output_path: Path) -> None:
    """Machine-readable sibling of write_flops_report, one row per non-skipped
    pass, so cross-model tables can be built without parsing rendered tables."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "model_class",
                "device",
                "pass",
                "scope",
                "input_shape",
                "params",
                "mean_macs",
                "mean_flops",
                "mean_peak_memory_bytes",
                "mean_peak_reserved_memory_bytes",
                "mean_kernel_launches",
                "mean_latency_seconds",
                "latency_std_across_modalities_seconds",
                "mean_latency_std_seconds",
                # Appended last so existing csv.DictReader consumers that
                # predate budget-conditioned models keep working unchanged.
                "budget",
                "realized_widths",
            ]
        )
        for path in report.measurements:
            if path.note:
                continue
            mac_stats = _mac_stats(path.per_modality)
            if mac_stats is None:
                continue
            memory_stats = _memory_stats(path.per_modality)
            reserved_memory_stats = _reserved_memory_stats(path.per_modality)
            kernel_stats = _kernel_stats(path.per_modality)
            time_stats = _time_stats(path.per_modality)
            time_std_stat = _time_std_stat(path.per_modality)
            mean_macs, mean_flops = mac_stats["Mean"]
            mean_memory = memory_stats["Mean"] if memory_stats is not None else ""
            mean_reserved_memory = (
                reserved_memory_stats["Mean"] if reserved_memory_stats is not None else ""
            )
            mean_kernel_launches = kernel_stats["Mean"] if kernel_stats is not None else ""
            mean_latency = time_stats["Mean"] if time_stats is not None else ""
            # Std, across the 15 modality-presence combinations, of each
            # combination's own (already repeat-averaged) latency -- how much
            # latency varies *between* combinations. Distinct from
            # mean_latency_std_seconds below, which is run-to-run noise
            # *within* a single combination's repeated timings.
            latency_std_across_modalities = time_stats["Std"] if time_stats is not None else ""
            mean_latency_std = time_std_stat if time_std_stat is not None else ""
            writer.writerow(
                [
                    report.model_name,
                    report.model_class_name,
                    report.device,
                    path.name,
                    pass_scope(report, path),
                    _format_shape(path.input_shape),
                    path.params,
                    mean_macs,
                    mean_flops,
                    mean_memory,
                    mean_reserved_memory,
                    mean_kernel_launches,
                    mean_latency,
                    latency_std_across_modalities,
                    mean_latency_std,
                    "" if report.budget is None else report.budget,
                    ""
                    if report.realized_widths is None
                    else " ".join(str(w) for w in report.realized_widths),
                ]
            )


def _format_shape(shape: tuple[int, ...] | None) -> str:
    if shape is None:
        return "-"
    return "[" + ", ".join(str(dim) for dim in shape) + "]"


def _format_count(value: float | None) -> str:
    if value is None:
        return "-"
    units = (
        (1e12, "T"),
        (1e9, "G"),
        (1e6, "M"),
        (1e3, "K"),
    )
    for factor, suffix in units:
        if abs(value) >= factor:
            return f"{value / factor:.3f}{suffix}"
    return f"{value:.0f}"


def _format_bytes(value: float | None) -> str:
    if value is None:
        return "n/a"
    units = (
        (2**40, "TiB"),
        (2**30, "GiB"),
        (2**20, "MiB"),
        (2**10, "KiB"),
    )
    for factor, suffix in units:
        if abs(value) >= factor:
            return f"{value / factor:.3f}{suffix}"
    return f"{value:.0f}B"


def _format_seconds(value: float | None, std: float | None = None) -> str:
    if value is None:
        return "n/a"
    if value >= 1:
        formatted = f"{value:.3f}s"
        std_formatted = f"{std:.3f}s" if std is not None else None
    else:
        formatted = f"{value * 1e3:.1f}ms"
        std_formatted = f"{std * 1e3:.1f}ms" if std is not None else None
    return formatted if std_formatted is None else f"{formatted} ± {std_formatted}"
