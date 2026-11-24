# Preprocessing

`missingbench preprocess` prepares a dataset before training or evaluation.

It currently supports three preprocessing stages:

- cropping
- intensity clamping
- intensity normalization

You can configure preprocessing in two places:

- CLI flags passed to `missingbench preprocess`
- a YAML config file passed with `--config`

If the same parameter is provided in both places, the CLI value should override the YAML value.

For the overall MissingBench YAML format, see [docs/yaml-config.md](yaml-config.md).
For a very detailed extension guide for this part of the stack, see [docs/components/preprocessing.md](components/preprocessing.md).

## Basic Usage

Run from YAML (recommended):

```bash
missingbench preprocess --config imfuse_23.yaml
```

Run from CLI:

```bash
missingbench preprocess \
  --input-dir /path/to/brats23 \
  --output-dir /path/to/preprocessed \
  --dataset-type brats23 \
  --crop-mode center \
  --crop-size 128 128 128 \
  --clamp-mode subject \
  --clamp-percentile 0.5 99.5 \
  --norm-mode min_max \
  --norm-min-max-range 0.0 1.0
```

## Where To Specify Options

All preprocessing parameters can be specified:

- in the CLI command
- in a YAML file

Recommended split:

- use YAML for stable experiment configuration
- use CLI for quick overrides such as `--output-dir` or `--yes`

## YAML Example

```yaml
input_dir: /path/to/brats23
output_dir: /path/to/preprocessed
dataset_type: brats23

crop_mode: center
crop_size: [128, 128, 128]
crop_min_size: null

clamp_mode: subject
clamp_percentile: [0.5, 99.5]
clamp_min: null
clamp_max: null

norm_mode: min_max
norm_min_max_range: [0.0, 1.0]
norm_mean: null
norm_std: null

yes: false
```

For the shared config structure used across commands, see [docs/yaml-config.md](yaml-config.md).

## Required Inputs

These values are required for preprocessing:

- `input_dir`
- `output_dir`
- `dataset_type`

`dataset_type` currently supports:

- `brats18`
- `brats23`
- `brats25`

## Output Format

Preprocessing writes one compressed `.npz` file per case.

Each file contains:

- `images`
- `seg`

The active training and testing paths both consume this format.

## Current UX Notes

The preprocessing command now prints Rich panels for:

- launch summary
- destructive output-directory confirmation
- output reset
- completion

Managed preprocessing failures are surfaced as CLI-style errors rather than raw runtime exceptions where applicable.

## Crop Options

### `crop_mode`

Controls the cropping strategy.

Allowed values:

- `none`
- `center`
- `non_empty`

### `crop_size`

Type:

- tuple of 3 integers

Meaning:

- spatial crop size `(X, Y, Z)` for center cropping

### `crop_min_size`

Type:

- tuple of 3 integers

Meaning:

- minimum crop size `(X, Y, Z)` for non-empty cropping

### Valid Crop Combinations

- `crop_mode: none`
  `crop_size` must be omitted
  `crop_min_size` must be omitted

- `crop_mode: center`
  `crop_size` is required
  `crop_min_size` must be omitted

- `crop_mode: non_empty`
  `crop_min_size` is required
  `crop_size` must be omitted

## Clamp Options

### `clamp_mode`

Controls intensity clamping.

Allowed values:

- `none`
- `subject`
- `dataset`

### `clamp_percentile`

Type:

- tuple of 2 floats

Meaning:

- percentile range `(LOW, HIGH)` used for subject-wise clamping

Constraints:

- `LOW < HIGH`
- both values must be between `0` and `100`

### `clamp_min`

Type:

- tuple of 4 floats

Meaning:

- lower clamp bound for each modality

### `clamp_max`

Type:

- tuple of 4 floats

Meaning:

- upper clamp bound for each modality

Constraints:

- for each modality, `clamp_min[i] < clamp_max[i]`

### Valid Clamp Combinations

- `clamp_mode: none`
  `clamp_percentile` must be omitted
  `clamp_min` must be omitted
  `clamp_max` must be omitted

- `clamp_mode: subject`
  `clamp_percentile` is required
  `clamp_min` must be omitted
  `clamp_max` must be omitted

- `clamp_mode: dataset`
  `clamp_min` is required
  `clamp_max` is required
  `clamp_percentile` must be omitted

## Normalization Options

### `norm_mode`

Controls intensity normalization.

Allowed values:

- `none`
- `min_max`
- `subject_zscore`
- `dataset_zscore`

### `norm_min_max_range`

Type:

- tuple of 2 floats

Meaning:

- target output range `(MIN, MAX)` for min-max normalization

Constraint:

- `MIN < MAX`

### `norm_mean`

Type:

- tuple of 4 floats

Meaning:

- dataset mean for each modality

### `norm_std`

Type:

- tuple of 4 floats

Meaning:

- dataset standard deviation for each modality

Constraints:

- every entry must be strictly greater than `0`

### Valid Normalization Combinations

- `norm_mode: none`
  `norm_min_max_range` must be omitted
  `norm_mean` must be omitted
  `norm_std` must be omitted

- `norm_mode: min_max`
  `norm_min_max_range` is required
  `norm_mean` must be omitted
  `norm_std` must be omitted

- `norm_mode: subject_zscore`
  `norm_min_max_range` must be omitted
  `norm_mean` must be omitted
  `norm_std` must be omitted

- `norm_mode: dataset_zscore`
  `norm_mean` is required
  `norm_std` is required
  `norm_min_max_range` must be omitted

## Output

The preprocessing pipeline writes one compressed `.npz` file per case into `output_dir`.

Each output archive contains:

- `images`
- `seg`

## Source Of Truth

The validation rules documented here are enforced in:

- [config.py](../src/missingbench/preprocessing/config.py)
- [cli.py](../src/missingbench/cli.py)

If the behavior changes, update this document together with those files.
