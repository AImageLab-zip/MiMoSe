from collections.abc import Callable
from dataclasses import dataclass, field
import importlib
from pathlib import Path
from typing import Any

import typer

from missingbench.enums import LossKind


@dataclass(frozen=True)
class LossConfig:
    loss_class: Any
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KwargField:
    """Describes how to cast/validate a single loss constructor kwarg."""

    caster: Callable[[Any], Any]
    validator: Callable[[Any], bool] | None = None
    error: str = "is invalid"


def positive_int(error: str = "must be > 0") -> KwargField:
    return KwargField(caster=int, validator=lambda value: value > 0, error=error)


def nonneg_float(error: str = "must be >= 0") -> KwargField:
    return KwargField(caster=float, validator=lambda value: value >= 0, error=error)


def positive_float(error: str = "must be > 0") -> KwargField:
    return KwargField(caster=float, validator=lambda value: value > 0, error=error)


def unit_interval_float(error: str = "must be in the range (0, 1]") -> KwargField:
    return KwargField(caster=float, validator=lambda value: 0 < value <= 1, error=error)


def build_loss_config(
    loss_kind: LossKind | str,
    loss_kwargs: dict[str, Any] | None = None,
) -> LossConfig:
    if isinstance(loss_kind, LossKind):
        resolved_loss_name = loss_kind.value
    else:
        resolved_loss_name = str(loss_kind).strip()

    if not resolved_loss_name:
        raise typer.BadParameter(
            "Loss name cannot be empty",
            param_hint="--loss",
        )

    loss_class = _resolve_loss(resolved_loss_name)
    resolved_kwargs = _normalize_loss_kwargs(loss_class, loss_kwargs or {})
    return LossConfig(
        loss_class=loss_class,
        kwargs=resolved_kwargs,
    )


def _resolve_loss(loss_name: str) -> Any:
    target_name = _normalize_name(loss_name)
    package_dir = Path(__file__).resolve().parent

    for module_path in sorted(package_dir.glob("*.py")):
        if module_path.stem in {"__init__", "config"}:
            continue

        module = importlib.import_module(f"missingbench.losses.{module_path.stem}")
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue

            attr = getattr(module, attr_name)
            if getattr(attr, "__module__", None) != module.__name__:
                continue
            if not callable(attr):
                continue
            if _matches_name(target_name, attr_name):
                return attr

    raise typer.BadParameter(
        f"Unsupported loss: {loss_name}",
        param_hint="--loss",
    )


def _normalize_loss_kwargs(loss_class: Any, loss_kwargs: dict[str, Any]) -> dict[str, Any]:
    kwarg_spec: dict[str, KwargField] = getattr(loss_class, "KWARG_SPEC", {})
    kwargs = dict(loss_kwargs)

    for key, kwarg_field in kwarg_spec.items():
        if kwargs.get(key) is None:
            continue

        value = kwarg_field.caster(kwargs[key])
        if kwarg_field.validator is not None and not kwarg_field.validator(value):
            raise typer.BadParameter(
                f"loss {key} {kwarg_field.error}",
                param_hint=f"--{key.replace('_', '-')}",
            )
        kwargs[key] = value

    return kwargs


def _matches_name(target_name: str, candidate_name: str) -> bool:
    normalized_candidate = _normalize_name(candidate_name)
    return (
        normalized_candidate == target_name
        or normalized_candidate.removesuffix("loss") == target_name
        or target_name.removesuffix("loss") == normalized_candidate
    )


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())
