# Extending Runtime Configuration

This document explains how to extend optimizers, schedulers, and other runtime-configurable objects.

Relevant files:

- `src/missingbench/training/config.py`
- `src/missingbench/enums.py`
- `src/missingbench/cli.py`

## Mental Model

For runtime objects, MissingBench usually follows this pattern:

1. enum value in `enums.py`
2. config dataclass in `training/config.py`
3. resolver/builder function
4. CLI/YAML exposure

The important point is that adding a new object implementation is only one part of the work. The CLI and YAML layers also need to know how to select it.

## Optimizers

Current optimizer flow:

1. YAML/CLI chooses `optimizer`
2. `build_optimizer_config(...)` validates values
3. `OptimizerConfig` stores the resolved class and kwargs
4. the trainer builds the actual optimizer

### Add a New Optimizer

1. add a new value to `OptimizerKind`
2. import the optimizer implementation in `training/config.py`
3. add it to `OPTIMIZER_DICT`
4. extend `build_optimizer_config(...)`
5. document any required fields
6. update example configs

If the optimizer has required parameters that differ from the existing set, you must also:

1. add CLI options if needed
2. merge them through `merge_cli_overrides(...)`
3. include them in YAML examples

## Schedulers

Current scheduler flow:

1. YAML/CLI chooses `scheduler`
2. `build_scheduler_config(...)` validates values
3. `SchedulerConfig` stores the class and kwargs
4. the trainer builds and steps the scheduler

### Add a New Scheduler

1. add a new value to `SchedulerKind`
2. import the scheduler implementation
3. add it to `SCHEDULER_DICT`
4. extend `build_scheduler_config(...)`
5. make sure the trainer’s stepping logic is compatible
6. add CLI/YAML fields if needed
7. update docs and example configs

Important warning:

Some schedulers need special stepping behavior.

For example:

- epoch-stepped schedulers can usually use the default `self.scheduler.step()`
- metric-driven schedulers need validation metrics
- some schedulers need trainer-specific overrides to preserve legacy behavior

If a scheduler’s semantics do not match the generic stepping path, the builder alone is not enough.

## Extending Generic Runtime Fields

Examples:

- new precision mode
- new distributed flag
- new logging option

Typical steps:

1. add the CLI option in `cli.py`
2. merge it through `merge_cli_overrides(...)`
3. thread it into trainer construction
4. store it in the correct config or trainer field
5. use it inside the runtime
6. update YAML docs

## Validation Checklist

Before calling a runtime-config extension done, verify:

- the value can be provided from YAML
- the same value can be overridden from CLI
- invalid combinations fail cleanly
- the resolved config object stores the expected class and kwargs
- the trainer actually uses the new behavior
