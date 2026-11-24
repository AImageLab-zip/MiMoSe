# YAML Config Guide

This document explains how to write a YAML config for MissingBench.

If you are extending the codebase itself rather than just writing configs, the detailed component extension guides live under [docs/components/README.md](components/README.md).

MissingBench uses a flat YAML style. One file can contain fields for:

- preprocessing
- training
- testing

Each command reads only the keys it needs and ignores the rest. One YAML file can hold preprocess, train, and test fields together, and that is the recommended setup for most runs.

## General Rules

- CLI values override YAML values when both are provided.
- Most fields are top-level keys.
- `custom_model_kwargs`, `custom_loss_kwargs`, and `custom_trainer_kwargs` are nested dictionaries.
- Shipped templates live in `src/missingbench/data/config_templates/`; `missingbench setup` writes path-patched copies to `src/missingbench/data/configs/`.

The easiest workflow is:

1. copy a shipped config
2. change paths and run-specific values
3. keep the rest until you need a deliberate override

## Common Field Groups

### Preprocessing

- `input_dir`: path
- `output_dir`: path
- `dataset_type`: string enum
  - `brats18`
  - `brats23`
  - `brats25`
- `crop_mode`: string enum
  - `none`
  - `center`
  - `non_empty`
- `crop_size`: list of 3 integers or `null`
- `crop_min_size`: list of 3 integers or `null`
- `clamp_mode`: string enum
  - `none`
  - `subject`
  - `dataset`
- `clamp_percentile`: list of 2 floats or `null`
- `clamp_min`: list of 4 floats or `null`
- `clamp_max`: list of 4 floats or `null`
- `norm_mode`: string enum
  - `none`
  - `min_max`
  - `subject_zscore`
  - `dataset_zscore`
- `norm_min_max_range`: list of 2 floats or `null`
- `norm_mean`: list of 4 floats or `null`
- `norm_std`: list of 4 floats or `null`
- `yes`: boolean

### Training

- `data_dir`: path
- `art_dir`: path
- `trainer`: string enum
  - `imfuse`
  - `dcseg`
  - `uhved`
  - `robustseg`
  - `shaspec`
  - `m3ae`
  - `m3fecon`
  - `srmnet`
  - `ims2trans`
  - `mstkdnet`
  - `rfl`
  - `lckd`
  - `mambavitakd`
  - `matunet`
- `model`: string enum
  - `imfuse`
  - `mmformer`
  - `dcseg`
  - `rfnet`
  - `a2fseg`
  - `plainunet`
  - `matunet`
  - `uhved`
  - `robustseg`
  - `m2ftrans`
  - `sfusion`
  - `shaspec`
  - `m3ae`
  - `m3fecon`
  - `srmnet`
  - `mmmvit`
  - `ims2trans`
  - `mstkdnet`
  - `rfl`
  - `unetmfi`
  - `lckd`
  - `inoutfusion`
  - `mambavitakd`
- `loss`: built-in string choice (resolved by loss class name in `src/missingbench/losses/`)
  - `imfuse`
  - `dicece`
  - `uhved`
  - `robustseg`
  - `shaspec`
  - `m3ae`
  - `m3fecon`
  - `srmnet`
  - `ims2trans`
  - `mstkdnet`
  - `rfl`
  - `lckd`
  - `inoutfusion`
  - `mambavitakd`
- `optimizer`: string enum
  - `radam`
  - `adamw`
  - `adam`
  - `sgd`
- `betas`: list of 2 floats or `null`
- `momentum`: float
- `scheduler`: string enum
  - `poly`
  - `warmuppoly`
  - `cosine`
  - `step`
  - `multistep`
  - `plateau`
- `poly_total_iters`: integer or `null`
- `poly_power`: float
- `cosine_t_max`: integer or `null`
- `cosine_eta_min`: float or `null`
- `step_step_size`: integer or `null`
- `step_gamma`: float or `null`
- `multistep_milestones`: list of integers or `null`
- `multistep_gamma`: float or `null`
- `plateau_mode`: string or `null`
  - `min`
  - `max`
- `plateau_factor`: float or `null`
- `plateau_patience`: integer or `null`
- `lr`: float
- `weight_decay`: float
- `batch_size`: integer
- `num_epochs`: integer
- `num_workers`: integer
- `precision`: string enum
  - `fp32`
  - `fp16`
  - `bf16`
- `compile`: boolean
- `validation_every`: integer
- `results_dir`: path
- `fold`: integer
- `resume`: boolean
- `pretrain`: path or `null`
- `seed`: integer or `null`
- `wandb_project`: string or `null`
- `wandb_mode`: string or `null`
  - `online`
  - `offline`
  - `disabled`
- `wandb_run_name`: string or `null`
- `wandb_entity`: string or `null`
- `push_to_hf`: boolean
- `hf_repo`: string or `null`
- `dataset_type`: string enum
  - `brats18`
  - `brats23`
  - `brats25`
- `split_file`: path or `null`
- `transform_kind`: string enum or `null`
  - `imfuse`
  - `shaspec`

### Testing

- `checkpoint_path`: path
- `output_path`: path
- `data_dir`: path
- `model`: string enum
  - `imfuse`
  - `mmformer`
  - `dcseg`
  - `rfnet`
  - `a2fseg`
  - `plainunet`
  - `matunet`
  - `uhved`
  - `robustseg`
  - `m2ftrans`
  - `sfusion`
  - `shaspec`
  - `m3ae`
  - `m3fecon`
  - `srmnet`
  - `mmmvit`
  - `ims2trans`
  - `mstkdnet`
  - `rfl`
  - `unetmfi`
  - `lckd`
  - `inoutfusion`
  - `mambavitakd`
- `dataset_type`: string enum
  - `brats18`
  - `brats23`
  - `brats25`
- `num_workers`: integer
- `seed`: integer or `null`
- `split_file`: path or `null`

### Nested Override Blocks

- `custom_model_kwargs`: dictionary
- `custom_loss_kwargs`: dictionary
- `custom_trainer_kwargs`: dictionary

## Field Type Notes

### Paths

Path-like values are written as strings:

```yaml
data_dir: /path/to/preprocessed
checkpoint_path: /path/to/run/checkpoints/final_weights_only.safetensors
```

### Enums

Enum-like values are written as lowercase strings:

```yaml
trainer: dcseg
optimizer: adam
dataset_type: brats23
transform_kind: imfuse
wandb_mode: offline
```

### Booleans

Use YAML booleans:

```yaml
compile: false
resume: true
yes: false
```

### Integers

Use plain integers:

```yaml
batch_size: 2
num_epochs: 500
num_workers: 8
```

### Floats

Use plain decimal values or scientific notation:

```yaml
lr: 0.0002
weight_decay: 0.0001
eps: 1.0e-7
```

## Minimal Examples

### Preprocess Example

```yaml
input_dir: /path/to/brats23
output_dir: /path/to/preprocessed
dataset_type: brats23

crop_mode: non_empty
crop_size: null
crop_min_size: [128, 128, 128]

clamp_mode: none
clamp_percentile: null
clamp_min: null
clamp_max: null

norm_mode: subject_zscore
norm_min_max_range: null
norm_mean: null
norm_std: null

yes: false
```

### Training Example

```yaml
data_dir: /path/to/preprocessed
art_dir: /path/to/run
trainer: dcseg
model: dcseg
loss: imfuse

custom_model_kwargs:
  num_cls: 4
  fusion_type: RFM

custom_trainer_kwargs:
  patch_size: 80
  transform_kind: imfuse
  train_masking_mode: random
  val_masking_mode: validation
  use_recon_loss: true
  use_reg_loss: true
  use_ana_contrastive: true
  use_mod_contrastive: true

optimizer: adam
betas: [0.9, 0.999]

scheduler: poly
poly_total_iters: 500
poly_power: 0.9

lr: 0.0002
weight_decay: 0.0001
batch_size: 1
num_epochs: 500
num_workers: 8
precision: fp32
resume: false
pretrain: null
seed: 999

wandb_project: MissingBench
wandb_mode: online
wandb_run_name: dcseg23_training
dataset_type: brats23
```

### Testing Example

```yaml
data_dir: /path/to/preprocessed
checkpoint_path: /path/to/final_weights_only.safetensors
output_path: /path/to/results.txt
model: dcseg

custom_model_kwargs:
  num_cls: 4
  fusion_type: RFM

dataset_type: brats23
num_workers: 8
seed: 42
```

## Nested Blocks

### `custom_model_kwargs`

Passed directly to the selected model class.

Type: dictionary with model-specific keys.

```yaml
custom_model_kwargs:
  num_cls: 4
  fusion_type: RFM
```

### `custom_loss_kwargs`

Passed directly to the selected loss class.

Type: dictionary with loss-specific keys.

```yaml
custom_loss_kwargs:
  num_classes: 4
  eps: 1.0e-7
```

### `custom_trainer_kwargs`

Passed directly to the selected trainer.

Type: dictionary with trainer-specific keys.

```yaml
custom_trainer_kwargs:
  patch_size: 80
  transform_kind: imfuse
  train_masking_mode: random
  val_masking_mode: validation
```

Common trainer-specific keys and types:

- `iter_per_epoch`: integer or `null`
- `region_fusion_start_epoch`: integer
- `patch_size`: integer
- `debug`: boolean
- `transform_kind`: string enum
- `transform_kind` values:
  - `imfuse`
  - `shaspec`
- `train_masking_mode`: string enum
  - `random`
  - `validation`
  - `test`
- `val_masking_mode`: string enum
  - `random`
  - `validation`
  - `test`
- `use_recon_loss`: boolean
- `use_reg_loss`: boolean
- `use_ana_contrastive`: boolean
- `use_mod_contrastive`: boolean
- `regularization_alpha`: float
- `anatomy_contrastive_method`: string
  - `cos_sim`
  - `ssim`
- `budgets_per_step`, `budget_sampling`, `local_window_ratio`, `lambda_param`, `tau_start`, `tau_end`, `gate_lr_scale`, `inplace_distill`, `distill_alpha`, `student_modality_drop`: MatUNet (`trainer: matunet`) multi-budget options; see `matunet_23.yaml` for the documented defaults

## Nulls and Lists

Use `null` for intentionally unset values:

- `crop_size: null`
- `pretrain: null`

Use YAML lists for sequences:

- `crop_min_size: [128, 128, 128]`
- `betas: [0.9, 0.999]`
- `multistep_milestones: [100, 200, 300]`

## One File for Multiple Commands

It is normal for one config to include preprocess, train, and test fields together.

- `missingbench preprocess --config ...` reads preprocess keys
- `missingbench train --config ...` reads train keys
- `missingbench test --config ...` reads test keys

You do not need separate files unless that is easier for your workflow.

## Starting Points

Recommended examples:

- `src/missingbench/data/configs/imfuse_18.yaml`
- `src/missingbench/data/configs/imfuse_23.yaml`
- `src/missingbench/data/configs/mmformer_18.yaml`
- `src/missingbench/data/configs/mmformer_23.yaml`
- `src/missingbench/data/configs/dcseg_18.yaml`
- `src/missingbench/data/configs/dcseg_23.yaml`
- `src/missingbench/data/configs/rfnet_18.yaml`
- `src/missingbench/data/configs/rfnet_23.yaml`
- `src/missingbench/data/configs/uhved_18.yaml`
- `src/missingbench/data/configs/uhved_23.yaml`
- `src/missingbench/data/configs/robustseg_18.yaml`
- `src/missingbench/data/configs/robustseg_23.yaml`
- `src/missingbench/data/configs/m2ftrans_18.yaml`
- `src/missingbench/data/configs/m2ftrans_23.yaml`
- `src/missingbench/data/configs/sfusion_18.yaml`
- `src/missingbench/data/configs/sfusion_23.yaml`
- `src/missingbench/data/configs/shaspec_18.yaml`
- `src/missingbench/data/configs/shaspec_23.yaml`
- `src/missingbench/data/configs/m3ae_18.yaml`
- `src/missingbench/data/configs/m3ae_23.yaml`
- `src/missingbench/data/configs/m3fecon_18.yaml`
- `src/missingbench/data/configs/m3fecon_23.yaml`
- `src/missingbench/data/configs/srmnet_18.yaml`
- `src/missingbench/data/configs/srmnet_23.yaml`
- `src/missingbench/data/configs/mmmvit_18.yaml`
- `src/missingbench/data/configs/mmmvit_23.yaml`
- `src/missingbench/data/configs/ims2trans_18.yaml`
- `src/missingbench/data/configs/ims2trans_23.yaml`
- `src/missingbench/data/configs/mstkdnet_18.yaml`
- `src/missingbench/data/configs/mstkdnet_23.yaml`
- `src/missingbench/data/configs/rfl_18.yaml`
- `src/missingbench/data/configs/rfl_23.yaml`
- `src/missingbench/data/configs/unetmfi_18.yaml`
- `src/missingbench/data/configs/unetmfi_23.yaml`
- `src/missingbench/data/configs/lckd_18.yaml`
- `src/missingbench/data/configs/inoutfusion_18.yaml`
- `src/missingbench/data/configs/inoutfusion_23.yaml`
- `src/missingbench/data/configs/a2fseg_23.yaml`
- `src/missingbench/data/configs/lckd_23.yaml`
- `src/missingbench/data/configs/mambavitakd_23.yaml`
- `src/missingbench/data/configs/plainunet_b01_23.yaml` ... `plainunet_b16_23.yaml`
- `src/missingbench/data/configs/matunet_23.yaml`

Every baseline also ships a `_25.yaml` (BraTS 2025) variant. Plain U-Net and MatUNet ship BraTS 2023 configs only.

## Related Docs

- [README.md](../README.md)
- [docs/preprocessing.md](preprocessing.md)
- [docs/training.md](training.md)
- [docs/testing.md](testing.md)
- [docs/extending.md](extending.md)
