from dataclasses import dataclass, field
import importlib
from pathlib import Path
from typing import Any

import typer

from mimose.enums import ModelKind


@dataclass(frozen=True)
class ModelConfig:
    model_class: Any
    kwargs: dict[str, Any] = field(default_factory=dict)


def build_model_config(
    model_kind: ModelKind | str,
    model_kwargs: dict[str, Any] | None = None,
) -> ModelConfig:
    if isinstance(model_kind, ModelKind):
        resolved_model_name = model_kind.value
    else:
        resolved_model_name = str(model_kind).strip()

    if not resolved_model_name:
        raise typer.BadParameter(
            "Model name cannot be empty",
            param_hint="--model",
        )

    model_class = _resolve_model(resolved_model_name)
    return ModelConfig(
        model_class=model_class,
        kwargs=model_kwargs or {},
    )


def _resolve_model(model_name: str) -> Any:
    target_name = _normalize_name(model_name)
    package_dir = Path(__file__).resolve().parent
    import_failures: dict[str, Exception] = {}

    for module_path in sorted(package_dir.glob("*.py")):
        if module_path.stem in {"__init__", "config"}:
            continue
        if not module_path.stem.isidentifier():
            continue

        try:
            module = importlib.import_module(f"mimose.models.{module_path.stem}")
        except ImportError as exc:
            # An unrelated model's optional dependency (e.g. mamba_ssm, whose
            # prebuilt kernels need a newer glibc than some clusters ship) must
            # not make every other model unresolvable. Reported below only if
            # the requested model is never found.
            import_failures[module_path.stem] = exc
            continue
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

    detail = "".join(
        f"\n  mimose.models.{stem} failed to import: {exc}" for stem, exc in import_failures.items()
    )
    raise typer.BadParameter(
        f"Unsupported model: {model_name}{detail}",
        param_hint="--model",
    )


def _matches_name(target_name: str, candidate_name: str) -> bool:
    normalized_candidate = _normalize_name(candidate_name)
    return (
        normalized_candidate == target_name
        or normalized_candidate.removesuffix("model") == target_name
        or target_name.removesuffix("model") == normalized_candidate
    )


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())
