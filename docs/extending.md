# Extending MissingBench

This document gives a high-level map of how to extend the active `missingbench` package under `src/missingbench`.

In practice, the easiest way to extend MissingBench is usually to start from an `imfuse` config and add a custom model module that follows the IMFuse-style trainer contract. This is the lowest-friction path because the repository already has a stable `IMFuseTrainer`, `IMFuseLoss`, dataset pipeline, masking logic, and testing flow.

Concretely, you can add a new file under [src/missingbench/models/](../src/missingbench/models), define a public class that inherits from [AbstractModel](../src/missingbench/models/abstract_model.py), implement `forward(images, mask)` so that training returns `(fuse_pred, sep_preds, prm_preds)` as expected by [IMFuseTrainer](../src/missingbench/training/trainers/imfuse.py), and implement `predict(images, mask)` for evaluation. You do not have to manually register the module in a hardcoded registry: [src/missingbench/models/config.py](../src/missingbench/models/config.py) scans `src/missingbench/models/*.py` dynamically and resolves public callables by normalized name, so once the file exists you can usually select it directly from YAML or CLI with `model: your_model_name`.

The maintained extension surface is `src/missingbench`. The `legacy/` directories are useful for reference and comparison, but they are not the package API you should extend.

If you need field-by-field or component-by-component instructions, use the detailed guides in [docs/components/README.md](components/README.md).

## Extension Flow

Most changes follow the same path:

1. add or modify the implementation in `src/missingbench/...`
2. wire the new behavior into config/build logic
3. expose it through CLI and YAML if users need to select it
4. update reference config templates under `src/missingbench/data/config_templates/`
5. update the relevant docs

If your change introduces or changes YAML fields, start with [docs/yaml-config.md](yaml-config.md).

## Repo Areas

These are the main places you will touch:

- [src/missingbench/cli.py](../src/missingbench/cli.py): CLI entrypoints, option definitions, command wiring
- [src/missingbench/gui.py](../src/missingbench/gui.py): GUI surface generated from the CLI
- [src/missingbench/enums.py](../src/missingbench/enums.py): shared enum choices exposed across config and CLI
- [src/missingbench/utils/cli_overrides.py](../src/missingbench/utils/cli_overrides.py): CLI-over-YAML merge behavior
- [src/missingbench/data/config_templates/](../src/missingbench/data/config_templates): shipped config templates copied by `missingbench setup`
- `src/missingbench/data/configs/`: local, path-patched configs written by `missingbench setup` (not shipped; created on first setup)
- [src/missingbench/data/splits/split.json](../src/missingbench/data/splits/split.json): packaged dataset split definition

## Preprocessing

The preprocessing stack lives in:

- [src/missingbench/preprocessing/config.py](../src/missingbench/preprocessing/config.py)
- [src/missingbench/preprocessing/cropping.py](../src/missingbench/preprocessing/cropping.py)
- [src/missingbench/preprocessing/clamping.py](../src/missingbench/preprocessing/clamping.py)
- [src/missingbench/preprocessing/normalization.py](../src/missingbench/preprocessing/normalization.py)
- [src/missingbench/preprocessing/pipeline.py](../src/missingbench/preprocessing/pipeline.py)

This is where you extend:

- crop/clamp/normalization modes
- dataset discovery for preprocessing
- per-case pipeline behavior

Typical file route:

1. add the implementation in one of `cropping.py`, `clamping.py`, or `normalization.py`
2. validate and resolve it in `config.py`
3. make sure `pipeline.py` can call it correctly
4. expose new options from `cli.py` if needed

Detailed guide: [docs/components/preprocessing.md](components/preprocessing.md)

## Models

The model integration layer lives in:

- [src/missingbench/models/abstract_model.py](../src/missingbench/models/abstract_model.py)
- [src/missingbench/models/config.py](../src/missingbench/models/config.py)
- [src/missingbench/models/IMFuse.py](../src/missingbench/models/IMFuse.py)
- [src/missingbench/models/mmformer.py](../src/missingbench/models/mmformer.py)
- [src/missingbench/models/dcseg.py](../src/missingbench/models/dcseg.py)
- [src/missingbench/models/rfnet.py](../src/missingbench/models/rfnet.py)

This is where you extend:

- new model families
- model selection logic
- inference contract through `predict(images, mask)`

Typical file route:

1. add a new module under `src/missingbench/models/`
2. inherit from `AbstractModel`
3. implement `forward(images, mask)` with the output structure expected by the chosen trainer
4. implement `predict(images, mask)` for evaluation and testing
5. wire selection through `models/config.py`
6. add enum support in `enums.py` if it should be a first-class built-in choice
7. make sure the chosen trainer understands the model output structure

The `predict(...)` method is the inference entrypoint used by [missingbench test](../src/missingbench/testing/pipeline.py). It should take a full input volume and a modality mask, run the model in test-time mode, and return segmentation logits or probabilities with shape `[B, C, H, W, D]`. In other words, `forward(...)` is the training contract, while `predict(...)` is the evaluation contract.

Detailed guide: [docs/components/models.md](components/models.md)

## Losses

The loss layer lives in:

- [src/missingbench/losses/config.py](../src/missingbench/losses/config.py)
- [src/missingbench/losses/imfuse.py](../src/missingbench/losses/imfuse.py)
- [src/missingbench/losses/dcseg.py](../src/missingbench/losses/dcseg.py)
- [src/missingbench/losses/__init__.py](../src/missingbench/losses/__init__.py)

This is where you extend:

- new loss classes
- loss selection logic
- loss kwargs exposed from YAML or CLI

Typical file route:

1. implement the loss module under `src/missingbench/losses/`
2. wire it through `losses/config.py`
3. add enum support in `enums.py` if needed
4. make sure the trainer passes the right tensors to it

The trainer/loss contract matters more than the loss file alone. If the model outputs or batch structure differ, you usually need trainer changes too.

## Datasets

The dataset layer lives in:

- [src/missingbench/datasets/base.py](../src/missingbench/datasets/base.py)
- [src/missingbench/datasets/imfuse.py](../src/missingbench/datasets/imfuse.py)
- [src/missingbench/datasets/config.py](../src/missingbench/datasets/config.py)
- [src/missingbench/data/splits/split.json](../src/missingbench/data/splits/split.json)

This is where you extend:

- sample loading
- masking behavior
- new dataset types
- split handling

Typical file route:

1. add enum support in `enums.py`
2. update dataset config or dataset classes
3. update preprocessing and testing pipeline assumptions if the sample format changes
4. add or update split files under `src/missingbench/data/splits/`

Detailed guide: [docs/components/datasets.md](components/datasets.md)

## Training Runtime

The shared runtime configuration lives in:

- [src/missingbench/training/config.py](../src/missingbench/training/config.py)
- [src/missingbench/enums.py](../src/missingbench/enums.py)

This is where you extend:

- optimizers
- schedulers
- runtime object builders
- common train-time config validation

Typical file route:

1. add the enum in `enums.py`
2. register the implementation in `training/config.py`
3. validate new kwargs there
4. ensure the trainer stepping behavior matches the new scheduler or optimizer semantics

Detailed guide: [docs/components/runtime-config.md](components/runtime-config.md)

## Trainers and Transforms

The trainer stack lives in:

- [src/missingbench/training/trainers/abstract_trainer.py](../src/missingbench/training/trainers/abstract_trainer.py)
- [src/missingbench/training/trainers/base_trainer.py](../src/missingbench/training/trainers/base_trainer.py)
- [src/missingbench/training/trainers/imfuse.py](../src/missingbench/training/trainers/imfuse.py)
- [src/missingbench/training/trainers/dcseg.py](../src/missingbench/training/trainers/dcseg.py)
- [src/missingbench/training/transforms/base_transforms.py](../src/missingbench/training/transforms/base_transforms.py)
- [src/missingbench/training/transforms/imfuse.py](../src/missingbench/training/transforms/imfuse.py)
- [src/missingbench/training/transforms/__init__.py](../src/missingbench/training/transforms/__init__.py)

This is where you extend:

- training loops
- validation behavior
- data augmentation and paired transforms
- family-specific runtime logic

Typical file route:

1. add or modify a concrete trainer under `training/trainers/`
2. wire trainer selection from `cli.py`
3. update transform managers if the trainer needs a different augmentation path
4. ensure the selected model and loss obey the trainer contract

Detailed guide: [docs/components/trainers.md](components/trainers.md)

## Testing

The evaluation path lives in:

- [src/missingbench/testing/pipeline.py](../src/missingbench/testing/pipeline.py)
- [src/missingbench/models/abstract_model.py](../src/missingbench/models/abstract_model.py)
- [src/missingbench/cli.py](../src/missingbench/cli.py)

This is where you extend:

- checkpoint evaluation behavior
- result formatting
- model-specific inference integration

The critical contract is that test-time models must support `predict(images, mask)`. If a model only works through `forward(...)`, `missingbench test` will not be enough on its own.

Checkpoint format is also part of the contract now:

- training resume uses `checkpoints/model_last.pth`
- inference and testing use `checkpoints/final_weights_only.safetensors`
- if you add or modify trainer checkpoint behavior, keep those two artifact roles separate
- if you add a new evaluation path, load the exported `.safetensors` weights artifact rather than a resumable pickle checkpoint

Detailed guide: [docs/components/testing.md](components/testing.md)

## CLI and GUI

The user-facing entrypoints live in:

- [src/missingbench/cli.py](../src/missingbench/cli.py)
- [src/missingbench/gui.py](../src/missingbench/gui.py)
- [src/missingbench/utils/cli_overrides.py](../src/missingbench/utils/cli_overrides.py)
- [src/missingbench/utils/cli_utils.py](../src/missingbench/utils/cli_utils.py)

This is where you extend:

- new command options
- command-level validation
- startup UX
- GUI exposure of CLI-backed features

Typical file route:

1. add the option or command in `cli.py`
2. merge CLI-over-YAML behavior in `utils/cli_overrides.py`
3. adjust helper formatting in `utils/cli_utils.py` if needed
4. verify the GUI still renders the new option correctly in `gui.py`

Detailed guide: [docs/components/cli-gui.md](components/cli-gui.md)

## Configs and Templates

The packaged reference configs live in:

- [src/missingbench/data/config_templates/](../src/missingbench/data/config_templates) (shipped templates)
- `src/missingbench/data/configs/` (local copies written by `missingbench setup`)

When you add a new built-in model, trainer, loss, or workflow option, update the shipped configs too. Otherwise the code may be technically wired but still hard to discover or use correctly.

## Practical Advice

- Keep generic runtime logic in shared config or `BaseTrainer`; keep family-specific assumptions in concrete trainers.
- Do not change only one layer. Most extension work crosses implementation, config, CLI, and docs.
- If behavior is selected by string name, check `enums.py`, config builders, and YAML examples together.
- If you are porting from `legacy/`, verify not only the model body but also transforms, scheduler semantics, losses, and trainer behavior.

## Detailed Guides

Use these when you need the exhaustive version:

- [docs/components/README.md](components/README.md)
- [docs/components/preprocessing.md](components/preprocessing.md)
- [docs/components/models.md](components/models.md)
- [docs/components/datasets.md](components/datasets.md)
- [docs/components/runtime-config.md](components/runtime-config.md)
- [docs/components/trainers.md](components/trainers.md)
- [docs/components/testing.md](components/testing.md)
- [docs/components/cli-gui.md](components/cli-gui.md)
