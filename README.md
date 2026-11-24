# MissingBench

MissingBench is a unified benchmark for brain tumour segmentation with missing MRI modalities. It reimplements state-of-the-art missing-modality methods inside a single package with a shared preprocessing pipeline, shared data splits, a common training runtime, and a common evaluation protocol (all 15 non-empty subsets of the four BraTS modalities), so that methods can be compared fairly on BraTS 2018, BraTS 2023, and BraTS 2025.

## What Is Included

- **21 baseline methods**, ported to the package in `src/missingbench`:
  A2FSeg (`a2fseg`), DC-Seg (`dcseg`), IM-Fuse (`imfuse`), IMS2Trans (`ims2trans`), InOutFusion (`inoutfusion`), LCKD (`lckd`), M2FTrans (`m2ftrans`), M3AE (`m3ae`), M3FeCon (`m3fecon`), MambaVit-AKD (`mambavitakd`), mmFormer (`mmformer`), MMMViT (`mmmvit`), MST-KDNet (`mstkdnet`), RFL (`rfl`), RFNet (`rfnet`), RobustSeg (`robustseg`), SFusion (`sfusion`), ShaSpec (`shaspec`), SRMNet (`srmnet`), U-HVED (`uhved`), UNET-MFI (`unetmfi`).
  Each has reference configs `<key>_18.yaml`, `<key>_23.yaml`, and `<key>_25.yaml`.
- **Original upstream code** of each baseline in `legacy/`, kept for reference and for checking the ports. It is not part of the package API.
- **Plain U-Net** (`model: plainunet`): a uniform-width nnU-Net-style 3D U-Net used as a non-specialised reference, in 16 sizes from 1.9M to 255.6M parameters (`plainunet_b01_23.yaml` ... `plainunet_b16_23.yaml`).
- **MatUNet** (Matryoshka U-Net, `model: matunet`, `trainer: matunet`): a budget-conditioned supernet trained with sandwich budget sampling, so that one set of weights serves a whole range of parameter budgets (~1.4M to ~91.4M parameters with `matunet_23.yaml`). Any budget in the range can be evaluated from the same checkpoint.

## Installation

Requirements: Python 3.12 and [`uv`](https://docs.astral.sh/uv/). Building the Mamba kernels (`mamba-ssm`, `causal-conv1d`, used by IM-Fuse and MambaVit-AKD) needs a CUDA toolkit with `nvcc` on the `PATH`.

From the repository root:

```bash
uv sync --no-install-package mamba-ssm --no-install-package causal-conv1d
uv pip install causal-conv1d mamba-ssm --no-build-isolation
missingbench setup
```

The second step compiles native code and can take a while the first time. Run `missingbench` from the project environment (e.g. after `source .venv/bin/activate`, or with `uv run missingbench ...`).

`missingbench setup` is interactive: it copies the packaged templates from `src/missingbench/data/config_templates/` into `src/missingbench/data/configs/` and patches only the local path fields (raw dataset location, preprocessing output, artifact and results roots) plus optional logging settings (Weights & Biases mode/entity, Hugging Face repo for checkpoints). Re-run it whenever you want to regenerate the local configs.

## Supported Hardware and Software

- Linux only.
- NVIDIA GPUs of the Turing architecture or newer (e.g. RTX 2000 series and later). Testing requires CUDA.

## Command-Line Interface

```bash
missingbench --help
missingbench --version
```

Commands:

| Command | Purpose |
| --- | --- |
| `setup` | Generate local configs from the packaged templates |
| `preprocess` | Preprocess a BraTS-style dataset into compressed `.npz` files |
| `preprocess-train` | Run preprocessing and then training from one config |
| `train` | Train a model from a YAML config and/or CLI overrides |
| `test` | Evaluate a checkpoint on all 15 missing-modality masks |
| `test-fast` | Same mask sweep as `test`, loading each subject once and batching the masks |
| `push` | Upload an existing local checkpoint to a Hugging Face repo |
| `export` | Package a trained checkpoint into a self-contained `.missingbenchpkg` archive |
| `flops` | Profile parameters, FLOPs, memory and latency of a model or config |
| `interference` | Measure gradient interference between budgets of a MatUNet supernet |

A Tk-based GUI generated from the same CLI is available as `missingbench-gui`.

Supported dataset types: `brats18`, `brats23`, `brats25`.

## Quick Start

Config names can be given as bare file names (resolved in `src/missingbench/data/configs/`) or as paths to your own YAML files.

### Preprocess

```bash
missingbench preprocess --config imfuse_23.yaml
```

Each case is written as one compressed `.npz` file with `images` and `seg`. To preprocess and immediately train:

```bash
missingbench preprocess-train --config imfuse_23.yaml
```

More detail: [docs/preprocessing.md](docs/preprocessing.md)

### Train

```bash
# a baseline
missingbench train --config imfuse_23.yaml

# Plain U-Net, smallest size (b01, 1.9M parameters)
missingbench train --config plainunet_b01_23.yaml

# MatUNet supernet
missingbench train --config matunet_23.yaml
```

Use `--fold <k>` to train on a fold of the 5-fold split and `--run-suffix <s>` (e.g. a seed) to keep repeated runs of the same config separate.

More detail: [docs/training.md](docs/training.md)

### Test

```bash
# a baseline
missingbench test --config imfuse_23.yaml

# Plain U-Net
missingbench test --config plainunet_b01_23.yaml

# MatUNet at a 30M-parameter budget
missingbench test --config matunet_23.yaml --budget 30000000 --budget-ceiling
```

`--budget` selects the subnetwork for a parameter budget; `--budget-ceiling` guarantees that the evaluated subnetwork does not exceed it. The test command writes a text report at `output_path` and an Excel summary next to it with the same stem and a `.xlsx` suffix.

More detail: [docs/testing.md](docs/testing.md)

## Configuration

- Stable experiment settings live in a flat YAML file; one file holds preprocessing, training, and testing fields together, and each command reads only the keys it needs.
- CLI values override YAML values when both are present.
- Model-, loss-, and trainer-specific options go in the nested `custom_model_kwargs`, `custom_loss_kwargs`, and `custom_trainer_kwargs` blocks.
- The shipped templates are in [src/missingbench/data/config_templates/](src/missingbench/data/config_templates); `missingbench setup` writes the path-patched copies used at runtime.
- Training writes a resumable checkpoint (`checkpoints/model_last.pth`) and an inference checkpoint (`checkpoints/final_weights_only.safetensors`) under `art_dir`; testing consumes the latter.

Field-by-field guide: [docs/yaml-config.md](docs/yaml-config.md)

## Project Layout

- [src/missingbench/cli.py](src/missingbench/cli.py): CLI entry points
- [src/missingbench/gui.py](src/missingbench/gui.py): GUI launcher
- [src/missingbench/preprocessing/](src/missingbench/preprocessing): preprocessing config and pipeline
- [src/missingbench/models/](src/missingbench/models): the 21 baselines, Plain U-Net (`plain_unet.py`), MatUNet (`matunet.py`)
- [src/missingbench/losses/](src/missingbench/losses): losses, including `dice_ce.py` (`dicece`)
- [src/missingbench/datasets/](src/missingbench/datasets): dataset loading and masking logic
- [src/missingbench/training/](src/missingbench/training): trainers, transforms, runtime config
- [src/missingbench/testing/](src/missingbench/testing): evaluation pipeline
- [src/missingbench/nas/](src/missingbench/nas): budget, gate, and parameter/FLOP counting utilities used by MatUNet
- [src/missingbench/data/](src/missingbench/data): config templates and dataset splits
- [legacy/](legacy): original upstream implementations of the baselines (reference only)

## Documentation

- [docs/yaml-config.md](docs/yaml-config.md): writing configs
- [docs/preprocessing.md](docs/preprocessing.md): preprocessing options
- [docs/training.md](docs/training.md): training, with per-method porting notes
- [docs/testing.md](docs/testing.md): evaluation
- [docs/extending.md](docs/extending.md): adding models, trainers, losses, datasets
- [docs/components/README.md](docs/components/README.md): detailed component-by-component extension guides
