# Standard library
from __future__ import annotations

from enum import Enum
import inspect
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, get_args, get_origin
import ttkbootstrap as tb

# Third-party
from typer.models import OptionInfo

# Internal modules
from missingbench.cli import app
from missingbench.preprocessing.config import ClampMode, CropMode, NormMode



APP_TITLE = "MissingBench GUI"


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is None:
        return annotation, False

    if origin in (tuple, list, dict):
        return annotation, False

    non_none = [arg for arg in args if arg is not type(None)]
    if len(non_none) == 1 and len(non_none) != len(args):
        return non_none[0], True

    return annotation, False


def _is_enum_type(annotation: Any) -> bool:
    return inspect.isclass(annotation) and issubclass(annotation, Enum)


def _is_path_type(annotation: Any) -> bool:
    return annotation is Path


def _is_tuple_of_ints(annotation: Any) -> bool:
    origin = get_origin(annotation)
    args = get_args(annotation)
    return origin is tuple and all(arg is int for arg in args)


def _is_tuple_of_floats(annotation: Any) -> bool:
    origin = get_origin(annotation)
    args = get_args(annotation)
    return origin is tuple and all(arg is float for arg in args)


def _is_list_of_ints(annotation: Any) -> bool:
    origin = get_origin(annotation)
    args = get_args(annotation)
    return origin is list and len(args) == 1 and args[0] is int


def _is_list_of_floats(annotation: Any) -> bool:
    origin = get_origin(annotation)
    args = get_args(annotation)
    return origin is list and len(args) == 1 and args[0] is float


def _extract_option_info(param: inspect.Parameter) -> OptionInfo | None:
    default = param.default
    if isinstance(default, OptionInfo):
        return default
    return None


def _option_default(param: inspect.Parameter) -> Any:
    option = _extract_option_info(param)
    if option is not None:
        return option.default
    if param.default is inspect._empty:
        return None
    return param.default


def _option_help(param: inspect.Parameter) -> str:
    option = _extract_option_info(param)
    if option is not None and option.help:
        return option.help
    return ""


def _option_panel(param: inspect.Parameter) -> str:
    option = _extract_option_info(param)
    if option is not None and getattr(option, "rich_help_panel", None):
        return option.rich_help_panel
    return "Options"


_COMMAND_ICONS: dict[str, str] = {
    "preprocess": "🧹",
    "train": "🏋️",
    "test": "🧪",
    "push": "📤",
    "export": "📦",
    "flops": "⚡",
    "setup": "🛠️",
    "version": "ℹ️",
    "destruct": "💥",
}


def _command_icon(name: str) -> str:
    key = name.replace("-", "_")
    for keyword, icon in _COMMAND_ICONS.items():
        if keyword in key:
            return icon
    return "🔧"


class Field:
    def get_value(self) -> Any:
        raise NotImplementedError

    def set_enabled(self, enabled: bool) -> None:
        raise NotImplementedError


class PathField(Field):
    def __init__(
        self,
        parent: ttk.Frame,
        label: str,
        directory: bool,
        help_text: str,
    ) -> None:
        self.var = tk.StringVar()
        self.label_widget = ttk.Label(parent, text=label)
        self.label_widget.pack(anchor="w", pady=(8, 0))

        self.row = ttk.Frame(parent)
        self.row.pack(fill="x")

        self.entry = ttk.Entry(self.row, textvariable=self.var)
        self.entry.pack(side="left", fill="x", expand=True)

        def browse() -> None:
            selected = filedialog.askdirectory() if directory else filedialog.askopenfilename()
            if selected:
                self.var.set(selected)

        self.button = ttk.Button(self.row, text="Browse", command=browse)
        self.button.pack(side="left", padx=(8, 0))

        self.help_widget: ttk.Label | None = None
        if help_text:
            self.help_widget = ttk.Label(parent, text=help_text, style="Help.TLabel")
            self.help_widget.pack(anchor="w")

    def get_value(self) -> str:
        return self.var.get().strip()

    def set_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.entry.configure(state=state)
        self.button.configure(state=state)


class EntryField(Field):
    def __init__(
        self,
        parent: ttk.Frame,
        label: str,
        help_text: str,
        default: Any = "",
    ) -> None:
        self.var = tk.StringVar(value="" if default in (None, ...) else str(default))

        self.label_widget = ttk.Label(parent, text=label)
        self.label_widget.pack(anchor="w", pady=(8, 0))

        self.entry = ttk.Entry(parent, textvariable=self.var)
        self.entry.pack(fill="x")

        self.help_widget: ttk.Label | None = None
        if help_text:
            self.help_widget = ttk.Label(parent, text=help_text, style="Help.TLabel")
            self.help_widget.pack(anchor="w")

    def get_value(self) -> str:
        return self.var.get().strip()

    def set_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.entry.configure(state=state)


class BoolField(Field):
    def __init__(
        self,
        parent: ttk.Frame,
        label: str,
        help_text: str,
        default: bool = False,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self.var = tk.BooleanVar(value=bool(default))
        self.checkbox = ttk.Checkbutton(parent, text=label, variable=self.var)
        self.checkbox.pack(anchor="w", pady=(8, 0))

        self.help_widget: ttk.Label | None = None
        if help_text:
            self.help_widget = ttk.Label(parent, text=help_text, style="Help.TLabel")
            self.help_widget.pack(anchor="w")

        if on_change is not None:
            self.var.trace_add("write", lambda *_: on_change())

    def get_value(self) -> bool:
        return self.var.get()

    def set_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.checkbox.configure(state=state)


class EnumButtonsField(Field):
    def __init__(
        self,
        parent: ttk.Frame,
        label: str,
        enum_cls: type[Enum],
        help_text: str,
        default: Any = None,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self.enum_cls = enum_cls

        self.label_widget = ttk.Label(parent, text=label)
        self.label_widget.pack(anchor="w", pady=(8, 0))

        default_value = None
        if default not in (None, ...):
            default_value = default.value if isinstance(default, Enum) else str(default)

        self.var = tk.StringVar(value=default_value or list(enum_cls)[0].value)

        self.row = ttk.Frame(parent)
        self.row.pack(fill="x")

        self.buttons: list[ttk.Radiobutton] = []
        for member in enum_cls:
            button = ttk.Radiobutton(
                self.row,
                text=member.value,
                value=member.value,
                variable=self.var,
            )
            button.pack(side="left", padx=(0, 8))
            self.buttons.append(button)

        self.help_widget: ttk.Label | None = None
        if help_text:
            self.help_widget = ttk.Label(parent, text=help_text, style="Help.TLabel")
            self.help_widget.pack(anchor="w")

        if on_change is not None:
            self.var.trace_add("write", lambda *_: on_change())

    def get_value(self) -> Enum:
        return self.enum_cls(self.var.get())

    def set_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in self.buttons:
            button.configure(state=state)


class Tuple3IntField(Field):
    def __init__(
        self,
        parent: ttk.Frame,
        label: str,
        help_text: str,
        default: Any = None,
    ) -> None:
        self.label_widget = ttk.Label(parent, text=label)
        self.label_widget.pack(anchor="w", pady=(8, 0))

        self.row = ttk.Frame(parent)
        self.row.pack(fill="x")

        values = ["", "", ""]
        if isinstance(default, tuple) and len(default) == 3:
            values = [str(v) for v in default]

        self.vars = [tk.StringVar(value=v) for v in values]
        self.entries: list[ttk.Entry] = []

        for idx, axis in enumerate(("X", "Y", "Z")):
            cell = ttk.Frame(self.row)
            cell.pack(side="left", fill="x", expand=True, padx=(0 if idx == 0 else 6, 0))
            ttk.Label(cell, text=axis).pack(anchor="w")
            entry = ttk.Entry(cell, textvariable=self.vars[idx], width=8)
            entry.pack(fill="x")
            self.entries.append(entry)

        self.help_widget: ttk.Label | None = None
        if help_text:
            self.help_widget = ttk.Label(parent, text=help_text, style="Help.TLabel")
            self.help_widget.pack(anchor="w")

    def get_value(self) -> tuple[int, int, int] | None:
        raw = [v.get().strip() for v in self.vars]
        if all(not x for x in raw):
            return None
        return tuple(int(x) for x in raw)

    def set_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for entry in self.entries:
            entry.configure(state=state)


class SequenceField(Field):
    def __init__(
        self,
        parent: ttk.Frame,
        label: str,
        help_text: str,
        default: Any = None,
    ) -> None:
        self.label_widget = ttk.Label(parent, text=label)
        self.label_widget.pack(anchor="w", pady=(8, 0))

        initial = ""
        if default not in (None, ...):
            if isinstance(default, (tuple, list)):
                initial = " ".join(str(v) for v in default)
            else:
                initial = str(default)

        self.var = tk.StringVar(value=initial)
        self.entry = ttk.Entry(parent, textvariable=self.var)
        self.entry.pack(fill="x")

        self.help_widget: ttk.Label | None = None
        if help_text:
            self.help_widget = ttk.Label(parent, text=help_text, style="Help.TLabel")
            self.help_widget.pack(anchor="w")

    def get_value(self) -> str:
        return self.var.get().strip()

    def set_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.entry.configure(state=state)


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent: ttk.Frame) -> None:
        super().__init__(parent)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas)

        self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.content.bind("<Configure>", self._on_content_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

    def _on_content_configure(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _bind_mousewheel(self, _event: tk.Event) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _unbind_mousewheel(self, _event: tk.Event) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event: tk.Event) -> None:
        delta = getattr(event, "delta", 0)
        if delta:
            self.canvas.yview_scroll(int(-delta / 120), "units")

    def _on_mousewheel_linux(self, event: tk.Event) -> None:
        num = getattr(event, "num", None)
        if num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif num == 5:
            self.canvas.yview_scroll(1, "units")


def _iter_top_level_commands(typer_app: Any) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []

    for command in typer_app.registered_commands:
        callback = getattr(command, "callback", None)
        if callback is None:
            continue

        name = command.name or callback.__name__.replace("_", "-")
        help_text = command.help or inspect.getdoc(callback) or ""

        commands.append(
            {
                "name": name,
                "callback": callback,
                "help": help_text,
            }
        )

    return commands


class AutoCommandForm(ttk.Frame):
    def __init__(
        self,
        parent: ttk.Frame,
        command_name: str,
        command_help: str,
        command_fn: Any,
        on_back: Callable[[], None],
        on_quit: Callable[[], None],
    ) -> None:
        super().__init__(parent, padding=(20, 16))
        self.command_name = command_name
        self.command_help = command_help
        self.command_fn = command_fn
        self.on_back = on_back
        self.on_quit = on_quit
        self.fields: dict[str, Field] = {}

        header = ttk.Frame(self)
        header.pack(fill="x", pady=(0, 4))

        icon = _command_icon(command_name)
        title = command_name.replace("-", " ").title()
        ttk.Label(header, text=f"{icon}  {title}", style="Title.TLabel").pack(anchor="w")

        if command_help:
            ttk.Label(self, text=command_help, style="Help.TLabel", wraplength=640, justify="left").pack(
                anchor="w", pady=(2, 4)
            )
        ttk.Label(
            self,
            text="Generated automatically from the Typer command signature.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 14))

        self._build_form()
        self._update_field_states()

        ttk.Separator(self).pack(fill="x", pady=(4, 12))

        actions = ttk.Frame(self)
        actions.pack(fill="x")

        ttk.Button(
            actions, text=f"▶  Run {title}", command=self._run, bootstyle="success"
        ).pack(side="left", ipadx=6)
        ttk.Button(
            actions, text="✕  Quit", command=self.on_quit, bootstyle="danger-outline"
        ).pack(side="right")
        ttk.Button(
            actions, text="←  Back", command=self.on_back, bootstyle="secondary-outline"
        ).pack(side="right", padx=(0, 8))

    def _build_form(self) -> None:
        sig = inspect.signature(self.command_fn)

        panels: dict[str, ttk.Labelframe] = {}

        for name, param in sig.parameters.items():
            annotation, _ = _unwrap_optional(param.annotation)
            help_text = _option_help(param)
            default = _option_default(param)
            label = name.replace("_", " ").title()
            panel_name = _option_panel(param)

            panel = panels.get(panel_name)
            if panel is None:
                panel = ttk.Labelframe(
                    self, text=f"  {panel_name}  ", padding=(14, 10), bootstyle="secondary"
                )
                panel.pack(fill="x", pady=(0, 12))
                panels[panel_name] = panel

            on_change = self._update_field_states if name in {"crop_mode", "clamp_mode", "norm_mode"} else None

            if _is_enum_type(annotation):
                field = EnumButtonsField(
                    panel,
                    label,
                    annotation,
                    help_text,
                    default,
                    on_change=on_change,
                )
            elif _is_path_type(annotation):
                directory = "dir" in name or name.endswith("_dir")
                field = PathField(panel, label, directory=directory, help_text=help_text)
            elif _is_tuple_of_ints(annotation):
                if len(get_args(annotation)) == 3:
                    field = Tuple3IntField(panel, label, help_text, default)
                else:
                    field = SequenceField(panel, label, help_text, default)
            elif _is_tuple_of_floats(annotation) or _is_list_of_ints(annotation) or _is_list_of_floats(annotation):
                field = SequenceField(panel, label, help_text, default)
            elif annotation is bool:
                field = BoolField(
                    panel,
                    label,
                    help_text,
                    default=bool(default),
                    on_change=on_change,
                )
            else:
                field = EntryField(panel, label, help_text, default=default)

            self.fields[name] = field

    def _current_values_for_logic(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name, field in self.fields.items():
            try:
                values[name] = field.get_value()
            except Exception:
                values[name] = None
        return values

    def _update_field_states(self) -> None:
        values = self._current_values_for_logic()
        crop_mode = values.get("crop_mode")
        clamp_mode = values.get("clamp_mode")
        norm_mode = values.get("norm_mode")

        if "crop_size" in self.fields:
            self.fields["crop_size"].set_enabled(crop_mode == CropMode.CENTER)

        if "crop_min_size" in self.fields:
            self.fields["crop_min_size"].set_enabled(crop_mode == CropMode.NON_EMPTY)

        if "clamp_percentile" in self.fields:
            self.fields["clamp_percentile"].set_enabled(clamp_mode == ClampMode.SUBJECT)

        if "clamp_min" in self.fields:
            self.fields["clamp_min"].set_enabled(clamp_mode == ClampMode.DATASET)

        if "clamp_max" in self.fields:
            self.fields["clamp_max"].set_enabled(clamp_mode == ClampMode.DATASET)

        if "norm_min_max_range" in self.fields:
            self.fields["norm_min_max_range"].set_enabled(norm_mode == NormMode.MIN_MAX)

        if "norm_mean" in self.fields:
            self.fields["norm_mean"].set_enabled(norm_mode == NormMode.DATASET_ZSCORE)

        if "norm_std" in self.fields:
            self.fields["norm_std"].set_enabled(norm_mode == NormMode.DATASET_ZSCORE)

    def _parse_space_separated(self, raw: str, caster: Callable[[str], Any]) -> list[Any]:
        if raw == "":
            return []
        return [caster(part) for part in raw.replace(",", " ").split()]

    def _convert_types(
        self,
        sig: inspect.Signature,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        converted: dict[str, Any] = {}

        for name, param in sig.parameters.items():
            raw = values[name]
            annotation, is_optional = _unwrap_optional(param.annotation)

            if raw in ("", None) and is_optional:
                converted[name] = None
                continue

            if _is_path_type(annotation):
                if raw == "":
                    raise ValueError(f"{name.replace('_', ' ').title()} is required.")
                converted[name] = Path(raw)
            elif name == "crop_size":
                converted[name] = parse_crop_size(self._parse_space_separated(raw, int))
            elif name == "crop_min_size":
                converted[name] = parse_crop_size(self._parse_space_separated(raw, int))
            elif name == "clamp_percentile":
                converted[name] = parse_percentile(self._parse_space_separated(raw, float))
            elif name == "clamp_min":
                converted[name] = parse_clamp(self._parse_space_separated(raw, int))
            elif name == "clamp_max":
                converted[name] = parse_clamp(self._parse_space_separated(raw, int))
            elif _is_tuple_of_ints(annotation) and raw != "":
                converted[name] = tuple(self._parse_space_separated(raw, int))
            elif _is_tuple_of_floats(annotation) and raw != "":
                converted[name] = tuple(self._parse_space_separated(raw, float))
            elif _is_list_of_ints(annotation):
                converted[name] = self._parse_space_separated(raw, int)
            elif _is_list_of_floats(annotation):
                converted[name] = self._parse_space_separated(raw, float)
            elif annotation is int and raw != "":
                converted[name] = int(raw)
            elif annotation is float and raw != "":
                converted[name] = float(raw)
            elif annotation is str:
                converted[name] = raw
            else:
                converted[name] = raw

        return converted

    def _run(self) -> None:
        try:
            sig = inspect.signature(self.command_fn)
            raw_values = {name: field.get_value() for name, field in self.fields.items()}
            kwargs = self._convert_types(sig, raw_values)
            self.command_fn(**kwargs)
            messagebox.showinfo("MissingBench", f"{self.command_name.replace('-', ' ').title()} completed.")
        except Exception as exc:
            messagebox.showerror("MissingBench", str(exc))


class CommandGrid(ttk.Frame):
    def __init__(
        self,
        parent: ttk.Frame,
        commands: list[dict[str, Any]],
        on_select: Callable[[dict[str, Any]], None],
    ) -> None:
        super().__init__(parent, padding=(20, 16))

        ttk.Label(
            self,
            text=f"Choose a command to configure and run — {len(commands)} available.",
            style="Help.TLabel",
        ).pack(anchor="w", pady=(0, 16))

        grid = ttk.Frame(self)
        grid.pack(fill="both", expand=True)

        columns = 2
        for col in range(columns):
            grid.columnconfigure(col, weight=1)

        for index, command in enumerate(commands):
            row = index // columns
            col = index % columns

            card = ttk.Frame(grid, padding=1, bootstyle="secondary")
            card.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
            inner = ttk.Frame(card, padding=16)
            inner.pack(fill="both", expand=True)

            title = command["name"].replace("-", " ").title()
            icon = _command_icon(command["name"])
            ttk.Label(inner, text=f"{icon}  {title}", style="Subtitle.TLabel").pack(anchor="w")
            ttk.Label(
                inner,
                text=command["help"] or "Open this command.",
                style="Help.TLabel",
                wraplength=260,
                justify="left",
            ).pack(anchor="w", pady=(6, 14))
            ttk.Button(
                inner,
                text="Open  →",
                command=lambda c=command: on_select(c),
                bootstyle="info-outline",
            ).pack(anchor="w", fill="x")

            clickable = [card, inner] + [
                w for w in inner.winfo_children() if not isinstance(w, ttk.Button)
            ]
            for widget in clickable:
                widget.configure(cursor="hand2")
                widget.bind("<Button-1>", lambda _e, c=command: on_select(c))


class BrainchMarkGUI:
    def __init__(self, root: tb.Window, typer_app: Any) -> None:
        self.root = root
        self.typer_app = typer_app
        self.commands = _iter_top_level_commands(typer_app)
        self.current_view: ttk.Frame | None = None

        header = ttk.Frame(root, padding=(20, 14, 20, 12))
        header.pack(fill="x")
        ttk.Label(header, text=f"🧠  {APP_TITLE}", style="Header.TLabel").pack(side="left")
        self.breadcrumb = ttk.Label(header, text="Home", style="Breadcrumb.TLabel")
        self.breadcrumb.pack(side="right")
        ttk.Separator(root).pack(fill="x")

        self.container = ttk.Frame(root)
        self.container.pack(fill="both", expand=True)

        self.show_home()

    def _set_view(self, view: ttk.Frame, *, breadcrumb: str) -> None:
        if self.current_view is not None:
            self.current_view.destroy()
        self.current_view = view
        self.current_view.pack(fill="both", expand=True)
        self.breadcrumb.configure(text=breadcrumb)

    def show_home(self) -> None:
        self._set_view(
            CommandGrid(self.container, self.commands, self.show_command),
            breadcrumb="Home",
        )

    def show_command(self, command: dict[str, Any]) -> None:
        scrollable = ScrollableFrame(self.container)
        form = AutoCommandForm(
            scrollable.content,
            command_name=command["name"],
            command_help=command["help"],
            command_fn=command["callback"],
            on_back=self.show_home,
            on_quit=self.root.destroy,
        )
        form.pack(fill="both", expand=True)
        title = command["name"].replace("-", " ").title()
        self._set_view(scrollable, breadcrumb=f"Home  ›  {title}")


def launch() -> None:
    root = tb.Window(themename="superhero")
    root.title(f"{APP_TITLE} — Model Toolkit")

    width, height = 920, 700
    root.minsize(720, 560)
    root.update_idletasks()
    x = (root.winfo_screenwidth() - width) // 2
    y = (root.winfo_screenheight() - height) // 3
    root.geometry(f"{width}x{height}+{max(x, 0)}+{max(y, 0)}")

    style = ttk.Style(root)
    muted = root.style.colors.get("secondary")
    style.configure("Header.TLabel", font=("TkDefaultFont", 17, "bold"))
    style.configure("Breadcrumb.TLabel", foreground=muted)
    style.configure("Title.TLabel", font=("TkDefaultFont", 15, "bold"))
    style.configure("Subtitle.TLabel", font=("TkDefaultFont", 12, "bold"))
    style.configure("Help.TLabel", foreground=muted)
    style.configure("Muted.TLabel", foreground=muted, font=("TkDefaultFont", 9))

    BrainchMarkGUI(root, app)
    root.mainloop()


if __name__ == "__main__":
    launch()
