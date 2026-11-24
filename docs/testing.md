# Testing

`missingbench test` evaluates a checkpoint by sweeping the standard 15 missing-modality masks and writing a text report plus an Excel summary.

The active testing path currently uses:

- the preprocessed `.npz` dataset format used by the training pipeline
- `IMFuseDataset` for sample loading
- the selected model's `predict(images, mask)` method

## Basic Usage

Run from YAML:

```bash
missingbench test --config src/missingbench/data/configs/imfuse_23.yaml
```

Run from CLI:

```bash
missingbench test \
  --data-dir /path/to/preprocessed \
  --checkpoint-path /path/to/final_weights_only.safetensors \
  --output-path /path/to/results.txt \
  --dataset-type brats23 \
  --model dcseg
```

Run from CLI with an online checkpoint cache:

```bash
missingbench test \
  --data-dir /path/to/preprocessed \
  --art-dir /path/to/artifacts/run1 \
  --online \
  --hf-repo owner/repo \
  --hf-run-name run1 \
  --output-path /path/to/results.txt \
  --dataset-type brats23 \
  --model dcseg
```

## Testing a MatUNet Budget

MatUNet (`model: matunet`) is a budget-conditioned supernet, so a single checkpoint can be evaluated at any parameter budget inside its range (~1.4M to ~91.4M parameters for the shipped config). Select the budget with `--budget` (number of parameters) and add `--budget-ceiling` so that, if the subnetwork chosen for that budget slightly overshoots it, the largest subnetwork whose exact parameter count fits is evaluated instead:

```bash
missingbench test --config matunet_23.yaml --budget 30000000 --budget-ceiling
```

Without `--budget`, the widest subnetwork is evaluated.

## Faster Mask Sweep

`missingbench test-fast` produces the same mask-sweep report as `test`, but loads each subject once and reuses it across all 15 masks, batching the masked copies with `--batch-size`.

## Required Inputs

The test command requires:

- `data_dir`
- `output_path`
- `dataset_type`

Use one of these checkpoint inputs:

- `checkpoint_path` for a local `.safetensors` weights file
- `hf_repo` together with `hf_run_name`, `online=true`, and `art_dir` for a downloaded cached checkpoint
- `art_dir` by itself to auto-resolve `art_dir/checkpoints/final_weights_only.safetensors`

`model` is optional in the CLI because it defaults to `imfuse`, but for a real run you should set it explicitly unless the config already does.

For the overall MissingBench YAML format, see [docs/yaml-config.md](yaml-config.md).
For a very detailed extension guide for the testing path, see [docs/components/testing.md](components/testing.md).

Any value of `ModelKind` (see [docs/yaml-config.md](yaml-config.md)) can be tested: the 21 baselines, `plainunet`, and `matunet`.

Dataset type choices:

- `brats18`
- `brats23`
- `brats25`

## What It Produces

The command writes:

- a text report at `output_path`
- an Excel summary at `output_path` with the suffix changed to `.xlsx`

The text report contains one line per modality mask plus an average summary.

It evaluates these 15 masks:

- single-modality subsets
- two-modality subsets
- three-modality subsets
- all four modalities

The current metrics reported are:

- `WT`
- `TC`
- `ET`
- `ETpp`

## Current Constraints

- testing currently requires CUDA
- batch size is fixed to `1` for `test` (`test-fast` batches the masked copies of one subject)
- the evaluation path is built for the BraTS-style 4-class setup used by the active models

If a checkpoint is missing, the command raises a clean CLI error.

If `online` is enabled, MissingBench downloads the checkpoint from Hugging Face using `hf_repo` and `hf_run_name`, stores it at `art_dir/checkpoints/hf/<repo>/<run>/final_weights_only.safetensors`, and reuses that cached file on later runs if it is already present.

## YAML Fields

The test command reads these fields from the combined config files:

- `data_dir`
- `checkpoint_path`
- `art_dir`
- `online`
- `hf_repo`
- `hf_run_name`
- `output_path`
- `model`
- `custom_model_kwargs`
- `split_file`
- `fold`
- `num_workers`
- `seed`
- `dataset_type`

## Example

```yaml
data_dir: /path/to/missingbench/dcseg23-preprocessed
checkpoint_path: /path/to/missingbench/runs/dcseg23/checkpoints/final_weights_only.safetensors
output_path: /path/to/missingbench/runs/dcseg23/results.txt
model: dcseg
custom_model_kwargs:
  num_cls: 4
  fusion_type: RFM
dataset_type: brats23
num_workers: 8
seed: 42
```

Online-checkpoint example:

```yaml
data_dir: /path/to/missingbench/dcseg23-preprocessed
art_dir: /path/to/missingbench/runs/dcseg23
online: true
hf_repo: owner/repo
hf_run_name: dcseg23
output_path: /path/to/missingbench/runs/dcseg23/results.txt
model: dcseg
dataset_type: brats23
num_workers: 8
seed: 42
```

## Related Docs

- [README.md](../README.md)
- [docs/yaml-config.md](yaml-config.md)
- [docs/training.md](training.md)

## Statistical Tests

[scripts/wilcoxon_tests.py](../scripts/wilcoxon_tests.py) runs paired Wilcoxon signed-rank tests between models from the `*_per_subject.csv` files written by `missingbench test`. Each subject's scores are averaged over the 15 modality combinations. Subjects are pooled across cross-validation folds, or averaged over fold models when the same subjects are tested by every fold (an external cohort). Six tests are run per reference model (WT/TC/ET Dice and HD95), with Holm correction over all tests of one model:

```bash
python scripts/wilcoxon_tests.py \
    --results-root /path/to/results \
    --ours "MatUNet 11.4M=matunet_23_b11.4M" \
    --refs imfuse_23 mmformer_23 rfnet_23 \
    --folds 1,2,3,4,5 \
    --pattern "results_fold{fold}_per_subject.csv" \
    --out wilcoxon_brats23
```

It writes `<out>.csv` (per-test means, wins/losses, raw and Holm-adjusted p-values, verdict) and a `<out>.png`/`.pdf` summary figure.
