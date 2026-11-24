import re
import shutil
import zipfile
from pathlib import Path

import typer

from missingbench.utils.cli_overrides import CONFIGS_DIR


CONFIG_TEMPLATES_DIR = CONFIGS_DIR.parent / "config_templates"


def zip_member_count(zip_path: Path) -> int:
    with zipfile.ZipFile(zip_path, "r") as zf:
        return sum(1 for m in zf.infolist() if len(Path(m.filename).parts) > 1)


def unzip_strip_root(zip_path: Path, dest: Path, *, on_file=None) -> None:
    """Extract a ZIP, skipping its single root directory so its subfolders land directly in dest."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            parts = Path(member.filename).parts
            if len(parts) <= 1:
                continue  # the root dir entry itself
            target = dest / Path(*parts[1:])
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            if on_file is not None:
                on_file()
def unzip_worker(zip_path: Path, dest: Path, queue, task_id: int) -> None:
    try:
        unzip_strip_root(zip_path, dest, on_file=lambda: queue.put(task_id))
        queue.put((task_id, None))
    except Exception as exc:
        queue.put((task_id, exc))


HF_REPO_PLACEHOLDER_PREFIXES = ("/path/to/", "<", "__")


def config_run_tag(config_path: Path) -> str:
    return config_path.stem.replace("_", "")


def dataset_input_dir_for_config(
    config_name: str,
    *,
    brats_data_dir: Path | None,
) -> str | None:
    return str(brats_data_dir) if brats_data_dir is not None else None


def replace_yaml_line(
    content: str,
    *,
    key: str,
    value: str | None,
) -> str:
    replacement = f"{key}: {'null' if value is None else value}"
    pattern = re.compile(rf"^{re.escape(key)}:\s*.*$", re.MULTILINE)
    updated, count = pattern.subn(replacement, content, count=1)
    if count != 1:
        raise typer.BadParameter(
            f"Could not update '{key}' in config content",
            param_hint="missingbench setup",
        )
    return updated


def upsert_yaml_line(
    content: str,
    *,
    key: str,
    value: str | None,
) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:\s*.*$", re.MULTILINE)
    if pattern.search(content):
        return replace_yaml_line(content, key=key, value=value)

    if content and not content.endswith("\n"):
        content += "\n"
    return content + f"{key}: {'null' if value is None else value}\n"


def get_yaml_line_value(content: str, *, key: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(key)}:\s*(.*)$", re.MULTILINE)
    match = pattern.search(content)
    if match is None:
        return None
    return match.group(1).strip()


def is_placeholder_value(value: str | None) -> bool:
    if value is None:
        return False
    cleaned = value.strip().strip("\"'")
    if cleaned in {"", "null", "none", "~"}:
        return False
    return cleaned.startswith(HF_REPO_PLACEHOLDER_PREFIXES)


def update_setup_config(
    *,
    config_path: Path,
    brats_data_dir: Path | None,
    preprocessed_root_dir: Path,
    artifacts_root_dir: Path,
    results_root_dir: Path,
    hf_repo: str | None = None,
    wandb_mode: str | None = None,
    wandb_entity: str | None = None,
) -> None:
    content = config_path.read_text(encoding="utf-8")
    run_tag = config_run_tag(config_path)
    preprocessed_folder = Path(get_yaml_line_value(content, key="output_dir").rstrip("/")).name
    preprocessed_dir = preprocessed_root_dir / preprocessed_folder
    artifacts_dir = artifacts_root_dir / run_tag
    # Group results by config file (e.g. "imfuse_18") so every seed/run-suffix
    # variant of the same model+dataset lands in one shared folder, rather
    # than scattered across each run's own artifacts directory.
    results_dir = results_root_dir / config_path.stem
    content = replace_yaml_line(
        content,
        key="input_dir",
        value=dataset_input_dir_for_config(
            config_path.name,
            brats_data_dir=brats_data_dir,
        ),
    )
    content = replace_yaml_line(content, key="output_dir", value=str(preprocessed_dir))
    if config_path.name != "preprocessing.yaml":
        content = replace_yaml_line(content, key="data_dir", value=str(preprocessed_dir))
        content = replace_yaml_line(content, key="art_dir", value=str(artifacts_dir))
        content = replace_yaml_line(
            content,
            key="checkpoint_path",
            value=str(artifacts_dir / "checkpoints" / "final_weights_only.safetensors"),
        )
        content = replace_yaml_line(content, key="results_dir", value=str(results_dir))
        content = replace_yaml_line(
            content,
            key="output_path",
            value=str(results_dir / "results.txt"),
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        if wandb_mode is not None and get_yaml_line_value(content, key="wandb_mode") is not None:
            content = replace_yaml_line(content, key="wandb_mode", value=wandb_mode)
        if wandb_entity is not None and get_yaml_line_value(content, key="wandb_entity") is not None:
            content = replace_yaml_line(content, key="wandb_entity", value=wandb_entity)
        template_hf_repo = get_yaml_line_value(content, key="hf_repo")
        if is_placeholder_value(template_hf_repo):
            content = upsert_yaml_line(content, key="push_to_hf", value="true" if hf_repo else "false")
            content = upsert_yaml_line(content, key="hf_repo", value=hf_repo)
    config_path.write_text(content, encoding="utf-8")


def read_current_setup_values(config_path: Path) -> dict[str, str | None]:
    """Read the current local-path fields out of an already-configured YAML file.

    `update_setup_config` always writes `output_dir`/`art_dir`/`results_dir` as
    `<root>/<per-config suffix>`, so the root of each is just the parent of the
    current value. These are reused as editable defaults when re-running setup
    against an existing config.
    """
    content = config_path.read_text(encoding="utf-8")

    def _value(key: str) -> str | None:
        value = get_yaml_line_value(content, key=key)
        if not value or value.strip() in {"null", "none", "~", ""}:
            return None
        return value.strip()

    def _root(key: str) -> str | None:
        value = _value(key)
        if value is None:
            return None
        return str(Path(value.rstrip("/")).parent)

    return {
        "brats_data_dir": _value("input_dir"),
        "preprocessed_root_dir": _root("output_dir"),
        "artifacts_root_dir": _root("art_dir"),
        "results_root_dir": _root("results_dir"),
        "hf_repo": _value("hf_repo"),
        "wandb_mode": _value("wandb_mode"),
        "wandb_entity": _value("wandb_entity"),
    }


def templates_require_hf_repo_prompt() -> bool:
    for template_path in sorted(CONFIG_TEMPLATES_DIR.glob("*.y*ml")):
        content = template_path.read_text(encoding="utf-8")
        if is_placeholder_value(get_yaml_line_value(content, key="hf_repo")):
            return True
    return False


def copy_config_templates() -> list[Path]:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    template_paths = sorted(CONFIG_TEMPLATES_DIR.glob("*.y*ml"))
    if not template_paths:
        raise typer.BadParameter(
            f"No config templates found under {CONFIG_TEMPLATES_DIR}",
            param_hint="missingbench setup",
        )

    copied_paths: list[Path] = []
    for template_path in template_paths:
        destination = CONFIGS_DIR / template_path.name
        shutil.copyfile(template_path, destination)
        copied_paths.append(destination)
    return copied_paths
