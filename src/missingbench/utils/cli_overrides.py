import ast
import yaml
from pathlib import Path
from typing import Any
import typer
CONFIG_NONE = "none"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CONFIGS_DIR = DATA_DIR / "configs"
SPLITS_DIR = DATA_DIR / "splits"

# Combined cross-validation split file (fold index at the top level). Used as the
# default split source whenever a --fold is requested. See onetime_scripts/make_kfold_splits.py.
KFOLD_SPLIT_FILENAME = "splits_5fold.json"


def is_fold_nested(split_payload: dict) -> bool:
    """True when the payload's top-level keys are fold indices (e.g. splits_5fold.json)
    rather than dataset names (e.g. split.json)."""
    keys = list(split_payload.keys())
    return bool(keys) and all(str(k).isdigit() for k in keys)


def select_fold(split_payload: dict, fold: int | None, *, source: Any = "split file") -> dict:
    """Return the flat ``{dataset: {train,val,test}}`` mapping, indexing a fold first when
    the payload is fold-nested. Flat (single-split) payloads are returned unchanged."""
    if not is_fold_nested(split_payload):
        return split_payload
    if fold is None:
        raise typer.BadParameter(
            f"{source} is fold-nested (folds {sorted(split_payload)}); pass --fold",
            param_hint="--fold",
        )
    key = str(fold)
    if key not in split_payload:
        raise typer.BadParameter(
            f"fold {fold} not found in {source}; available folds: {sorted(split_payload)}",
            param_hint="--fold",
        )
    return split_payload[key]


def _resolve_config_path(config_path: Path) -> Path:
    if config_path.is_absolute() or config_path.exists():
        return config_path

    candidate = CONFIGS_DIR / config_path
    if candidate.exists():
        return candidate

    raise typer.BadParameter(
        "config file not found. Use an absolute path, a valid relative path, "
        f"or a filename under {CONFIGS_DIR}",
        param_hint="--config",
    )


def load_dataset_case_ids(
    dataset_type: Any, split_path: Path | str | None = None, fold: int | None = None
) -> set[str]:
    """Return the set of case ids (train+val+test) belonging to `dataset_type` in the split file.

    When the split file is fold-nested (splits_5fold.json), `fold` selects which fold to read."""
    import json

    resolved = resolve_split_path(split_path)
    split_payload = select_fold(json.loads(resolved.read_text()), fold, source=resolved)
    dataset_key = str(dataset_type).lower()
    if dataset_key not in split_payload:
        raise typer.BadParameter(
            f"Dataset split '{dataset_key}' not found in {resolved}",
            param_hint="--dataset-type",
        )
    dataset_splits = split_payload[dataset_key]
    return {
        entry["sub"]
        for subset in ("train", "val", "test")
        for entry in dataset_splits.get(subset, [])
    }


def resolve_split_path(split_path: Path | str | None) -> Path:
    if split_path is None:
        return SPLITS_DIR / "split.json"

    candidate = Path(split_path)
    if candidate.is_absolute() or candidate.exists():
        return candidate

    resolved = SPLITS_DIR / candidate
    if resolved.exists():
        return resolved

    raise typer.BadParameter(
        "split file not found. Use an absolute path, a valid relative path, "
        f"or a filename under {SPLITS_DIR}",
        param_hint="--split-file",
    )


def load_yaml_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}

    resolved_path = _resolve_config_path(config_path)
    with resolved_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data = _normalize_config_value(data)
    if not isinstance(data, dict):
        raise typer.BadParameter("YAML config must contain a mapping at the top level.", param_hint="--config")

    return data


def _normalize_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_config_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return tuple(_normalize_config_value(inner) for inner in value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("(", "[", "{")) and stripped.endswith((")", "]", "}")):
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                return value
            return _normalize_config_value(parsed)
    return value

def merge_cli_overrides(
    yaml_config: dict[str, Any],
    **cli_values: Any,
) -> dict[str, Any]:
    merged = dict(yaml_config)

    for key, value in cli_values.items():
        if value is not None and value != CONFIG_NONE:
            merged[key] = value

    return merged


def apply_run_suffix(merged: dict[str, Any]) -> None:
    """Append `run_suffix` to art_dir, wandb_run_name, hf_run_name, and output_path in place.

    Lets the same config/CLI invocation be reused for repeated runs (e.g. multiple
    seeds for a std computation) without each run clobbering the previous one's
    artifacts, W&B run, HF upload path, or test report.
    """
    run_suffix = merged.get("run_suffix")
    if not run_suffix:
        return

    art_dir = merged.get("art_dir")
    if art_dir is not None:
        art_dir_path = Path(art_dir)
        merged["art_dir"] = str(art_dir_path.with_name(f"{art_dir_path.name}_{run_suffix}"))

    wandb_run_name = merged.get("wandb_run_name")
    if wandb_run_name:
        merged["wandb_run_name"] = f"{wandb_run_name}_{run_suffix}"

    hf_run_name = merged.get("hf_run_name")
    if hf_run_name:
        merged["hf_run_name"] = f"{hf_run_name}_{run_suffix}"

    output_path = merged.get("output_path")
    if output_path is not None:
        output_path_obj = Path(output_path)
        merged["output_path"] = str(
            output_path_obj.with_name(f"{output_path_obj.stem}_{run_suffix}{output_path_obj.suffix}")
        )

    checkpoint_path = merged.get("checkpoint_path")
    if checkpoint_path is not None:
        checkpoint_path_obj = Path(checkpoint_path)
        run_dir = checkpoint_path_obj.parent.parent
        suffixed_run_dir = run_dir.with_name(f"{run_dir.name}_{run_suffix}")
        merged["checkpoint_path"] = str(
            suffixed_run_dir / checkpoint_path_obj.parent.name / checkpoint_path_obj.name
        )
