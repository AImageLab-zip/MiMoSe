from dataclasses import dataclass
from enum import StrEnum
import importlib
from typing import Callable

import typer

from missingbench.enums import CropMode, ClampMode, NormMode


def _mode_value(mode: StrEnum | str) -> str:
    return mode.value if isinstance(mode, StrEnum) else str(mode).strip()


def _load_mode_fn(module_path: str, mode: StrEnum | str, param_hint: str) -> Callable:
    module = importlib.import_module(module_path)
    mode_name = _mode_value(mode)
    fn = getattr(module, mode_name, None)

    if fn is None or not callable(fn):
        raise typer.BadParameter(
            f"Function {mode_name!r} was not found in module {module.__name__!r}.",
            param_hint=param_hint,
        )

    return fn


def _known_mode(enum_cls: type[StrEnum], mode: StrEnum | str) -> StrEnum | None:
    try:
        return enum_cls(_mode_value(mode))
    except ValueError:
        return None


def _require_tuple(
    value: object,
    length: int,
    item_type: type,
    param_hint: str,
) -> tuple | None:
    if value is None:
        return None
    if not isinstance(value, tuple):
        raise typer.BadParameter(
            f"must be a tuple of exactly {length} {item_type.__name__} values",
            param_hint=param_hint,
        )
    if len(value) != length:
        raise typer.BadParameter(
            f"must contain exactly {length} values, got {len(value)}",
            param_hint=param_hint,
        )
    if not all(isinstance(item, item_type) for item in value):
        raise typer.BadParameter(
            f"must contain only {item_type.__name__} values",
            param_hint=param_hint,
        )
    return value


@dataclass(frozen=True)
class CropConfig:
    fn: Callable
    size: tuple[int, int, int] | None = None
    min_size: tuple[int, int, int] | None = None


def build_crop_config(
    crop_mode: CropMode | str,
    crop_size: tuple[int, int, int] | None,
    crop_min_size: tuple[int, int, int] | None,
) -> CropConfig:
    crop_size = _require_tuple(crop_size, 3, int, "--crop-size")
    crop_min_size = _require_tuple(crop_min_size, 3, int, "--crop-min-size")
    fn = _load_mode_fn("missingbench.preprocessing.cropping", crop_mode, "--crop-mode")
    known_mode = _known_mode(CropMode, crop_mode)

    if known_mode is CropMode.NONE:
        if crop_size is not None:
            raise typer.BadParameter(
                "must be omitted when --crop-mode is 'none'",
                param_hint="--crop-size",
            )
        if crop_min_size is not None:
            raise typer.BadParameter(
                "must be omitted when --crop-mode is 'none'",
                param_hint="--crop-min-size",
            )
    elif known_mode is CropMode.CENTER:
        if crop_size is None:
            raise typer.BadParameter(
                "is required when --crop-mode is 'center'",
                param_hint="--crop-size",
            )
        if crop_min_size is not None:
            raise typer.BadParameter(
                "must be omitted when --crop-mode is 'center'",
                param_hint="--crop-min-size",
            )
    elif known_mode is CropMode.NON_EMPTY:
        if crop_min_size is None:
            raise typer.BadParameter(
                "is required when --crop-mode is 'non_empty'",
                param_hint="--crop-min-size",
            )
        if crop_size is not None:
            raise typer.BadParameter(
                "must be omitted when --crop-mode is 'non_empty'",
                param_hint="--crop-size",
            )

    return CropConfig(fn=fn, size=crop_size, min_size=crop_min_size)


@dataclass(frozen=True)
class ClampConfig:
    fn: Callable
    percentile: tuple[float, float] | None = None
    min: tuple[float, float, float, float] | None = None
    max: tuple[float, float, float, float] | None = None


def build_clamp_config(
    clamp_mode: ClampMode | str,
    clamp_percentile: tuple[float, float] | None,
    clamp_min: tuple[float, float, float, float] | None,
    clamp_max: tuple[float, float, float, float] | None,
) -> ClampConfig:
    clamp_percentile = _require_tuple(clamp_percentile, 2, float, "--clamp-percentile")
    clamp_min = _require_tuple(clamp_min, 4, float, "--clamp-min")
    clamp_max = _require_tuple(clamp_max, 4, float, "--clamp-max")
    fn = _load_mode_fn("missingbench.preprocessing.clamping", clamp_mode, "--clamp-mode")
    known_mode = _known_mode(ClampMode, clamp_mode)

    if known_mode is ClampMode.NONE:
        if clamp_percentile is not None:
            raise typer.BadParameter(
                "must be omitted when --clamp-mode is 'none'",
                param_hint="--clamp-percentile",
            )
        if clamp_min is not None:
            raise typer.BadParameter(
                "must be omitted when --clamp-mode is 'none'",
                param_hint="--clamp-min",
            )
        if clamp_max is not None:
            raise typer.BadParameter(
                "must be omitted when --clamp-mode is 'none'",
                param_hint="--clamp-max",
            )
    elif known_mode is ClampMode.SUBJECT:
        if clamp_percentile is None:
            raise typer.BadParameter(
                "is required when --clamp-mode is 'subject'",
                param_hint="--clamp-percentile",
            )
        if clamp_min is not None:
            raise typer.BadParameter(
                "must be omitted when --clamp-mode is 'subject'",
                param_hint="--clamp-min",
            )
        if clamp_max is not None:
            raise typer.BadParameter(
                "must be omitted when --clamp-mode is 'subject'",
                param_hint="--clamp-max",
            )
        low, high = clamp_percentile
        if low >= high:
            raise typer.BadParameter(
                "LOW must be strictly smaller than HIGH.",
                param_hint="--clamp-percentile",
            )
        if low < 0 or low > 100:
            raise typer.BadParameter(
                "LOW must be between 0 and 100.",
                param_hint="--clamp-percentile",
            )
        if high < 0 or high > 100:
            raise typer.BadParameter(
                "HIGH must be between 0 and 100.",
                param_hint="--clamp-percentile",
            )
    elif known_mode is ClampMode.DATASET:
        if clamp_percentile is not None:
            raise typer.BadParameter(
                "must be omitted when --clamp-mode is 'dataset'",
                param_hint="--clamp-percentile",
            )
        if clamp_min is None:
            raise typer.BadParameter(
                "is required when --clamp-mode is 'dataset'",
                param_hint="--clamp-min",
            )
        if clamp_max is None:
            raise typer.BadParameter(
                "is required when --clamp-mode is 'dataset'",
                param_hint="--clamp-max",
            )
        for i, (min_value, max_value) in enumerate(zip(clamp_min, clamp_max)):
            if min_value >= max_value:
                raise typer.BadParameter(
                    f"value at index {i} must be strictly smaller than the corresponding value in --clamp-max, got {min_value} >= {max_value}",
                    param_hint="--clamp-min",
                )

    return ClampConfig(fn=fn, percentile=clamp_percentile, min=clamp_min, max=clamp_max)


@dataclass(frozen=True)
class NormConfig:
    fn: Callable
    min_max_range: tuple[float, float] | None = None
    mean: tuple[float, float, float, float] | None = None
    std: tuple[float, float, float, float] | None = None


def build_norm_config(
    norm_mode: NormMode | str,
    norm_min_max_range: tuple[float, float] | None,
    norm_mean: tuple[float, float, float, float] | None,
    norm_std: tuple[float, float, float, float] | None,
) -> NormConfig:
    norm_min_max_range = _require_tuple(norm_min_max_range, 2, float, "--norm-min-max-range")
    norm_mean = _require_tuple(norm_mean, 4, float, "--norm-mean")
    norm_std = _require_tuple(norm_std, 4, float, "--norm-std")
    fn = _load_mode_fn("missingbench.preprocessing.normalization", norm_mode, "--norm-mode")
    known_mode = _known_mode(NormMode, norm_mode)

    if known_mode is NormMode.NONE:
        if norm_min_max_range is not None:
            raise typer.BadParameter(
                "must be omitted when --norm-mode is 'none'",
                param_hint="--norm-min-max-range",
            )
        if norm_mean is not None:
            raise typer.BadParameter(
                "must be omitted when --norm-mode is 'none'",
                param_hint="--norm-mean",
            )
        if norm_std is not None:
            raise typer.BadParameter(
                "must be omitted when --norm-mode is 'none'",
                param_hint="--norm-std",
            )
    elif known_mode is NormMode.MIN_MAX:
        if norm_min_max_range is None:
            raise typer.BadParameter(
                "is required when --norm-mode is 'min_max'",
                param_hint="--norm-min-max-range",
            )
        if norm_mean is not None:
            raise typer.BadParameter(
                "must be omitted when --norm-mode is 'min_max'",
                param_hint="--norm-mean",
            )
        if norm_std is not None:
            raise typer.BadParameter(
                "must be omitted when --norm-mode is 'min_max'",
                param_hint="--norm-std",
            )
        if norm_min_max_range[0] >= norm_min_max_range[1]:
            raise typer.BadParameter(
                "MIN must be strictly smaller than MAX.",
                param_hint="--norm-min-max-range",
            )
    elif known_mode is NormMode.SUBJECT_ZSCORE:
        if norm_min_max_range is not None:
            raise typer.BadParameter(
                "must be omitted when --norm-mode is 'subject_zscore'",
                param_hint="--norm-min-max-range",
            )
        if norm_mean is not None:
            raise typer.BadParameter(
                "must be omitted when --norm-mode is 'subject_zscore'",
                param_hint="--norm-mean",
            )
        if norm_std is not None:
            raise typer.BadParameter(
                "must be omitted when --norm-mode is 'subject_zscore'",
                param_hint="--norm-std",
            )
    elif known_mode is NormMode.DATASET_ZSCORE:
        if norm_min_max_range is not None:
            raise typer.BadParameter(
                "must be omitted when --norm-mode is 'dataset_zscore'",
                param_hint="--norm-min-max-range",
            )
        if norm_mean is None:
            raise typer.BadParameter(
                "is required when --norm-mode is 'dataset_zscore'",
                param_hint="--norm-mean",
            )
        if norm_std is None:
            raise typer.BadParameter(
                "is required when --norm-mode is 'dataset_zscore'",
                param_hint="--norm-std",
            )
        for i, std_value in enumerate(norm_std):
            if std_value <= 0:
                raise typer.BadParameter(
                    f"value at index {i} must be strictly greater than 0, got {std_value}",
                    param_hint="--norm-std",
                )

    return NormConfig(fn=fn, min_max_range=norm_min_max_range, mean=norm_mean, std=norm_std)
