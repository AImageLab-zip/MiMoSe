import typer
from typing import Any

def require_preprocess_values(merged: dict[str, Any], *required_keys: str) -> None:
    for key in required_keys:
        value = merged.get(key)
        if value is None:
            raise typer.BadParameter(
                "missing value; provide it in the CLI or in --config",
                param_hint=f"--{key.replace('_', '-')}",
            )
