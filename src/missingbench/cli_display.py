from pathlib import Path

from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.completion import PathCompleter
from rich.console import Console
from rich.panel import Panel


CONSOLE = Console()
PATH_COMPLETER = PathCompleter(expanduser=True)


def expand_user_path(raw_value: str) -> Path:
    return Path(raw_value).expanduser().resolve(strict=False)


def prompt_path(prompt: str, *, default: str = "") -> str:
    return pt_prompt(
        f"{prompt}: ",
        completer=PATH_COMPLETER,
        complete_while_typing=True,
        default=default,
    ).strip()


def prompt_zip_file(
    *, label: str, prompt: str, default_dir: Path | None = None, optional: bool = False
) -> Path | None:
    import zipfile

    while True:
        raw_value = pt_prompt(
            f"{prompt}{' (leave empty to skip)' if optional else ''}: ",
            completer=PATH_COMPLETER,
            complete_while_typing=True,
            default=f"{default_dir}/" if default_dir is not None else "",
        ).strip()
        if not raw_value:
            if optional:
                CONSOLE.print(f"[dim]Skipping {label} -- not provided.[/dim]")
                return None
            CONSOLE.print(
                Panel(
                    f"[bold red]{label}[/bold red] is required.",
                    title="[bold red]Missing Value[/bold red]",
                    border_style="red",
                    expand=False,
                )
            )
            continue

        candidate = expand_user_path(raw_value)
        if not candidate.is_file():
            CONSOLE.print(
                Panel(
                    f"[bold red]{label}[/bold red]\n{candidate} does not exist or is not a file.",
                    title="[bold red]Invalid File[/bold red]",
                    border_style="red",
                    expand=False,
                )
            )
            continue

        if not zipfile.is_zipfile(candidate):
            CONSOLE.print(
                Panel(
                    f"[bold red]{label}[/bold red]\n{candidate} is not a valid ZIP file.",
                    title="[bold red]Invalid ZIP[/bold red]",
                    border_style="red",
                    expand=False,
                )
            )
            continue

        return candidate


def prompt_required_existing_directory(
    *,
    label: str,
    prompt: str,
    default_dir: Path | None = None,
) -> Path:
    while True:
        raw_value = prompt_path(
            prompt,
            default=f"{default_dir}/" if default_dir is not None else "",
        )
        if not raw_value:
            CONSOLE.print(
                Panel(
                    f"[bold red]{label}[/bold red] is required.",
                    title="[bold red]Missing Value[/bold red]",
                    border_style="red",
                    expand=False,
                )
            )
            continue

        candidate = expand_user_path(raw_value)
        if candidate.is_dir():
            return candidate

        CONSOLE.print(
            Panel(
                f"[bold red]{label}[/bold red]\n{candidate} is not an existing directory.",
                title="[bold red]Invalid Directory[/bold red]",
                border_style="red",
                expand=False,
            )
        )


def prompt_optional_existing_directory(
    *,
    label: str,
    prompt: str,
) -> Path | None:
    while True:
        raw_value = prompt_path(prompt)
        if not raw_value:
            return None

        candidate = expand_user_path(raw_value)
        if candidate.is_dir():
            return candidate

        CONSOLE.print(
            Panel(
                f"[bold red]{label}[/bold red]\n{candidate} is not an existing directory.",
                title="[bold red]Invalid Directory[/bold red]",
                border_style="red",
                expand=False,
            )
        )


def prompt_required_directory(
    *,
    label: str,
    prompt: str,
    default_dir: Path | None = None,
) -> Path:
    while True:
        raw_value = prompt_path(
            prompt,
            default=f"{default_dir}/" if default_dir is not None else "",
        )
        if not raw_value:
            CONSOLE.print(
                Panel(
                    f"[bold red]{label}[/bold red] is required.",
                    title="[bold red]Missing Value[/bold red]",
                    border_style="red",
                    expand=False,
                )
            )
            continue

        return expand_user_path(raw_value)
