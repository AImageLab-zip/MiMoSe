import ast
from typing import Any

from mimose.enums import TransformKind
from mimose.training.transforms.base_transforms import TransformManager
from mimose.training.transforms.imfuse import (
    ALoRAMoETransformManager,
    IMFuseTransformManager,
    TinyMimosaTransformManager,
)
from mimose.training.transforms.shaspec import ShaSpecTransformManager


def _resolve_int_tuple(
    value: object,
    default: tuple[int, ...],
) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"Expected list/tuple string, got {value!r}")
        value = parsed
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Expected list/tuple value, got {type(value).__name__}")
    return tuple(int(item) for item in value)


def build_transform_manager(
    kind: TransformKind,
    *,
    model_kwargs: dict[str, Any] | None = None,
    crop_size: tuple[int, int, int] | None = None,
) -> TransformManager:
    resolved_kind = kind if isinstance(kind, TransformKind) else TransformKind(str(kind).lower())

    if resolved_kind == TransformKind.IMFUSE:
        if crop_size is not None:
            return IMFuseTransformManager(crop_size=crop_size)
        return IMFuseTransformManager()
    if resolved_kind == TransformKind.TINYMIMOSA:
        resolved_model_kwargs = model_kwargs or {}
        return TinyMimosaTransformManager(
            input_shape=_resolve_int_tuple(
                resolved_model_kwargs.get("input_shape"),
                (182, 218, 182),
            ),
            features_per_stage=_resolve_int_tuple(
                resolved_model_kwargs.get("features_per_stage"),
                (8, 16, 32, 64),
            ),
        )

    if resolved_kind == TransformKind.SHASPEC:
        return ShaSpecTransformManager(crop_size=crop_size or (80, 80, 80))
    if resolved_kind == TransformKind.ALORAMOE:
        return ALoRAMoETransformManager(crop_size=crop_size or (128, 128, 128))

    raise ValueError(f"Unsupported transform kind: {kind}")


__all__ = [
    "TransformManager",
    "ALoRAMoETransformManager",
    "IMFuseTransformManager",
    "TinyMimosaTransformManager",
    "ShaSpecTransformManager",
    "TransformKind",
    "build_transform_manager",
]
