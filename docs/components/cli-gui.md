# Extending CLI and GUI

This document explains how to extend the command-line and GUI surface.

Relevant files:

- `src/missingbench/cli.py`
- `src/missingbench/gui.py`
- `src/missingbench/utils/cli_overrides.py`

## Mental Model

The CLI is the source of truth.

The GUI is derived from the Typer command surface and does not define the workflow independently.

That means most extension work starts in `cli.py`.

## Add a New CLI Option

1. add the option to the relevant command function in `cli.py`
2. choose the correct Typer type and help text
3. merge it through `merge_cli_overrides(...)` if it participates in YAML/CLI merging
4. thread it into the correct builder or trainer/model construction path
5. update docs and examples

If the option is not merged correctly, it may appear in the CLI but never affect runtime behavior.

## Add a New Command

1. define a new `@app.command()` function
2. implement validation and runtime behavior
3. add docs
4. verify the GUI can surface it cleanly

## Add Shell Completion Support

If a new option points to files or split names:

1. add or reuse a completion helper
2. wire it into the Typer option definition
3. test partial-path and relative-path behavior

## GUI Considerations

Simple CLI option additions usually work automatically in the GUI.

You may need GUI follow-up when:

- a type is exotic
- a value needs custom interaction
- help text needs better UX phrasing
- a workflow requires grouping or extra validation

## Startup UX

The `train()` command currently shows an immediate Rich status animation before trainer construction begins.

If you change startup behavior:

1. keep early feedback visible
2. avoid moving all user feedback after heavy imports
3. preserve clean behavior under distributed relaunch

## Validation Checklist

Before calling a CLI/GUI extension done, verify:

- the option appears in `missingbench --help` or the relevant subcommand help
- YAML override behavior still works if applicable
- defaults are correct
- error messages are CLI-friendly
- the GUI still renders the command cleanly
