# Extending Preprocessing

This document explains how to extend the preprocessing stack in detail.

Relevant files:

- `src/missingbench/preprocessing/config.py`
- `src/missingbench/preprocessing/cropping.py`
- `src/missingbench/preprocessing/clamping.py`
- `src/missingbench/preprocessing/normalization.py`
- `src/missingbench/preprocessing/pipeline.py`
- `src/missingbench/cli.py`

## Mental Model

The preprocessing path has three stages:

1. crop
2. clamp
3. normalize

Each stage is configured by:

1. CLI/YAML values
2. a config dataclass
3. a `build_*_config(...)` resolver
4. a concrete implementation function

The pipeline then applies those resolved functions in order inside `preprocess_case(...)`.

## What Each File Does

### `config.py`

This file defines:

- stage config dataclasses
- validation rules
- builders such as `build_crop_config(...)`

If you need a new user-facing parameter, this is usually where it must be added.

### `cropping.py`

This file contains crop implementations.

Expected function shape:

```python
def my_crop(
    images: np.ndarray,
    seg: np.ndarray,
    config: CropConfig,
) -> tuple[np.ndarray, np.ndarray]:
    ...
```

### `clamping.py`

This file contains clamp implementations.

Expected function shape:

```python
def my_clamp(
    images: np.ndarray,
    config: ClampConfig,
) -> np.ndarray:
    ...
```

### `normalization.py`

This file contains normalization implementations.

Expected function shape:

```python
def my_norm(
    images: np.ndarray,
    config: NormConfig,
) -> np.ndarray:
    ...
```

### `pipeline.py`

This file:

- discovers raw dataset files
- loads modalities and segmentation
- applies crop/clamp/normalize
- writes `.npz`

If you are adding a new dataset layout, this file usually needs changes.

## Add a New Preprocessing Function

### Add a custom crop mode

1. Implement the function in `cropping.py`.
2. Make sure it accepts `(images, seg, config)`.
3. Decide whether it should be:
   - a built-in enum-backed mode
   - a dynamic function name resolved by module lookup
4. If it is built-in:
   - add a new `CropMode` enum value
   - update `build_crop_config(...)`
5. If it needs extra parameters:
   - add fields to `CropConfig`
   - validate them in `build_crop_config(...)`
   - expose them in `cli.py`
6. Test it through `missingbench preprocess`.

### Add a custom clamp mode

1. Implement the function in `clamping.py`.
2. Make sure it accepts `(images, config)`.
3. Add a `ClampMode` enum value if this should be first-class.
4. Extend `build_clamp_config(...)`.
5. Add any new fields to `ClampConfig`.
6. Expose new CLI/YAML knobs if needed.

### Add a custom normalization mode

1. Implement the function in `normalization.py`.
2. Make sure it accepts `(images, config)`.
3. Add a `NormMode` enum value if this should be first-class.
4. Extend `build_norm_config(...)`.
5. Add any new fields to `NormConfig`.
6. Expose any new knobs in `cli.py`.

## Add a New Preprocessing Parameter

When a new implementation needs a new user-facing value, update all of these:

1. add the field to the correct config dataclass in `config.py`
2. validate it in the corresponding `build_*_config(...)`
3. expose a CLI option in `cli.py`
4. merge it through `merge_cli_overrides(...)`
5. use it inside the implementation
6. update YAML examples

If you skip one of these layers, the feature will usually look wired but not actually work.

## Add a New Dataset Layout

If raw data is organized differently from BraTS18/BraTS23:

1. add a new `DatasetType` value
2. extend raw-file discovery in `pipeline.py`
3. make sure modality names map into the expected output order
4. confirm segmentation remapping behavior if labels differ
5. update docs and example configs

Important constraint:

The training and testing stack expects the preprocessed `.npz` files to contain:

- `images`
- `seg`

and the channel order must remain consistent with the active model boundary expectations.

## Validation Checklist

Before calling a preprocessing extension done, verify:

- the new mode can be selected from CLI
- the same mode can be selected from YAML
- invalid combinations fail cleanly
- output `.npz` files still contain the expected keys
- the downstream dataset loader can read the files unchanged
