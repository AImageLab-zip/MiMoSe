from pathlib import Path

from click.shell_completion import CompletionItem
import typer


# Compute paths without importing from missingbench (which is slow)
def _get_config_dir() -> Path:
    """Get configs dir without importing missingbench modules."""
    package_root = Path(__file__).resolve().parent
    return package_root / "data" / "configs"


def _get_splits_dir() -> Path:
    """Get splits dir without importing missingbench modules."""
    package_root = Path(__file__).resolve().parent
    return package_root / "data" / "splits"


# Pre-compute config and split files once at module load time
# This is cached per shell invocation (new process each time)
_CONFIG_FILES: tuple[str, ...] | None = None
_SPLIT_FILES: tuple[str, ...] | None = None


def _get_cached_config_files() -> tuple[str, ...]:
    """Get cached list of config files."""
    global _CONFIG_FILES
    if _CONFIG_FILES is None:
        configs_dir = _get_config_dir()
        _CONFIG_FILES = tuple(
            p.name for p in configs_dir.glob("*.y*ml")
        )
    return _CONFIG_FILES


def _get_cached_split_files() -> tuple[str, ...]:
    """Get cached list of split files (relative paths)."""
    global _SPLIT_FILES
    if _SPLIT_FILES is None:
        splits_dir = _get_splits_dir()
        _SPLIT_FILES = tuple(
            p.relative_to(splits_dir).as_posix()
            for p in splits_dir.rglob("*")
            if p.is_file()
        )
    return _SPLIT_FILES


def config_shell_complete(
    _ctx: typer.Context,
    _param: typer.CallbackParam,
    incomplete: str,
) -> list[CompletionItem]:
    suggestions: dict[str, CompletionItem] = {}
    configs_dir = _get_config_dir()

    # Only search packaged configs if no path separator in incomplete
    if "/" not in incomplete and not incomplete.startswith("."):
        # Fast: filter pre-computed cached list
        for config_name in _get_cached_config_files():
            if config_name.startswith(incomplete):
                suggestions[config_name] = CompletionItem(
                    config_name,
                    help=str(configs_dir),
                )
    else:
        # Handle absolute/relative paths
        raw_path = Path(incomplete).expanduser()
        parent = raw_path if incomplete.endswith("/") else raw_path.parent
        prefix = "" if incomplete.endswith("/") else raw_path.name
        if parent.exists() and parent.is_dir():
            for candidate in parent.iterdir():
                if candidate.suffix not in {".yaml", ".yml"}:
                    continue
                if not candidate.name.startswith(prefix):
                    continue
                suggestions[str(candidate)] = CompletionItem(str(candidate))

    return sorted(suggestions.values(), key=lambda x: x.value)


def split_shell_complete(
    _ctx: typer.Context,
    _param: typer.CallbackParam,
    incomplete: str,
) -> list[CompletionItem]:
    suggestions: dict[str, CompletionItem] = {}
    splits_dir = _get_splits_dir()

    # Only search packaged splits if no path separator in incomplete
    if "/" not in incomplete and not incomplete.startswith("."):
        # Fast: filter pre-computed cached list
        for split_name in _get_cached_split_files():
            if split_name.startswith(incomplete):
                suggestions[split_name] = CompletionItem(
                    split_name,
                    help=str(splits_dir),
                )
    else:
        # Handle absolute/relative paths - walk as needed
        raw_path = Path(incomplete).expanduser()
        parent = raw_path if incomplete.endswith("/") else raw_path.parent
        prefix = "" if incomplete.endswith("/") else raw_path.name
        if parent.exists() and parent.is_dir():
            for candidate in parent.iterdir():
                if not candidate.is_file():
                    continue
                if not candidate.name.startswith(prefix):
                    continue
                suggestions[str(candidate)] = CompletionItem(str(candidate))

    return sorted(suggestions.values(), key=lambda x: x.value)
