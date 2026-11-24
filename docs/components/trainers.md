# Extending Trainers

This document explains how to add or modify trainers in detail.

Relevant files:

- `src/missingbench/checkpoints.py`
- `src/missingbench/training/trainers/abstract_trainer.py`
- `src/missingbench/training/trainers/base_trainer.py`
- `src/missingbench/training/trainers/imfuse.py`
- `src/missingbench/training/trainers/dcseg.py`
- `src/missingbench/training/trainers/__init__.py`
- `src/missingbench/cli.py`

## Mental Model

The trainer stack is split into:

### `AbstractTrainer`

Stores common state and declares the abstract surface.

### `BaseTrainer`

Provides generic runtime behavior:

- distributed setup
- checkpoint save/load
- optimizer/scheduler/loss construction
- launch summary
- WandB integration
- OOM normalization

The shared checkpoint policy is split by purpose:

- `model_last.pth` is the resumable training checkpoint
- `final_weights_only.safetensors` is the exported inference/testing checkpoint

### Concrete trainers

These implement model-family-specific behavior:

- `IMFuseTrainer` for IMFuse-compatible models such as `imfuse`, `mmformer`,
  and `rfnet`
- `DCSegTrainer`
- `MatUnetTrainer` (`trainer: matunet`) for the budget-conditioned MatUNet supernet
- one trainer per baseline that needs its own training loop (`uhved`, `robustseg`,
  `shaspec`, `m3ae`, `m3fecon`, `srmnet`, `ims2trans`, `mstkdnet`, `rfl`, `lckd`,
  `mambavitakd`)

## Add a New Trainer

1. create `src/missingbench/training/trainers/my_trainer.py`
2. subclass `BaseTrainer`
3. implement:
   - `train_epoch(self, epoch)`
   - `val_epoch(self, epoch)`
   - `build_datasets(self)`
4. add it to `training/trainers/__init__.py`
5. expose it through `_resolve_trainer_class(...)` in `cli.py`
6. add a `TrainerKind` enum value if it should be first-class
7. add or update reference configs

## What Belongs in BaseTrainer vs Concrete Trainer

Put logic in `BaseTrainer` when it is generic across trainer families:

- checkpoint policy
- DDP behavior
- optimizer construction
- scheduler construction
- generic device/runtime setup

Put logic in the concrete trainer when it depends on:

- a specific model return structure
- a specific loss composition
- a specific dataset class or transform path
- model-family-specific validation behavior

If you put model-family-specific assumptions into `BaseTrainer`, you usually make the next trainer harder to add.

## Trainer Constructor Flow

A concrete trainer typically:

1. reads `custom_trainer_kwargs`
2. resolves trainer-specific defaults
3. optionally patches scheduler/config behavior
4. calls `super().__init__(...)`
5. creates any trainer-specific auxiliary modules

This order matters.

Anything that should influence model/loss/optimizer/scheduler construction must be resolved before `super().__init__(...)` if the base constructor uses it during setup.

## Training Step Design

Most trainer bugs come from mismatches between:

- batch structure
- model output structure
- loss expectations

When writing `_train_step(...)`, verify:

1. batch keys are correct
2. tensor dtypes are correct
3. masks are moved and cast correctly
4. segmentation tensors are converted to the expected representation
5. the chosen loss sees the expected shapes

## Validation Design

Validation should be explicit about:

- whether the model is in eval mode
- whether auxiliary training flags are disabled
- whether prediction uses direct forward or sliding-window `predict(...)`
- how metrics are reduced

If a trainer family uses a custom inference path, validation should go through that path instead of accidentally reusing training forward behavior.

## Trainer-Specific YAML Fields

Trainer-specific knobs should usually live in `custom_trainer_kwargs`.

Examples:

- `patch_size`
- `transform_kind`
- `train_masking_mode`
- `val_masking_mode`
- `region_fusion_start_epoch`

If a trainer needs a new knob:

1. read it from `custom_trainer_kwargs`
2. give it a clear default
3. document its type and behavior
4. update example configs

## Validation Checklist

Before calling a trainer extension done, verify:

- the trainer can be selected from CLI/YAML
- its model/output/loss contract is internally consistent
- checkpoint save/load works
- `save_final_checkpoint()` exports weights-only `.safetensors`
- validation runs without relying on training-only outputs
- testing still works if the model is expected to support `predict(...)`
