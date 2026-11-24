# Extending Datasets

This document explains how to extend dataset loading and split handling.

Relevant files:

- `src/missingbench/datasets/base.py`
- `src/missingbench/datasets/imfuse.py`
- `src/missingbench/datasets/config.py`
- `src/missingbench/data/splits/`
- `src/missingbench/preprocessing/pipeline.py`

## Mental Model

The active training and testing path assumes:

1. raw data is preprocessed into `.npz`
2. dataset loaders read those `.npz` files
3. split files specify which subject belongs to which split
4. masks may be injected either by split metadata or by runtime masking mode

## Current Active Dataset Path

`IMFuseDataset` is the active `.npz` loader used by:

- IMFuse training/testing
- mmFormer training/testing
- DC-Seg training/testing

Even though the models differ, they currently share the same preprocessed sample format.

## Sample Format Contract

Each preprocessed sample file is expected to contain:

- `images`
- `seg`

`IMFuseDataset` then normalizes the segmentation shape and returns a dictionary with:

- `sub`
- `images`
- `seg`
- `mask`

If you change the stored file format, you must update the active dataset loader or create a new one.

## Add a New Dataset Loader

Create a new loader module when:

- sample files are not `.npz`
- stored keys differ
- labels need a different shape normalization path
- masks are stored differently

Typical steps:

1. create a new dataset module under `src/missingbench/datasets/`
2. subclass `BaseDataset`
3. implement `build_sample(...)`
4. implement `__getitem__(...)`
5. decide how transforms are applied
6. decide how masks are resolved
7. update the relevant trainer to use the new dataset class

## Add a New Dataset Type

1. add a value to `DatasetType`
2. update preprocessing discovery in `preprocessing/pipeline.py`
3. add or update split files
4. update any trainer logic that branches on dataset type
5. update docs and configs

## Split Files

The active split path uses JSON under `src/missingbench/data/splits/`.

Each dataset section typically contains:

- `train`
- `val`
- `test`

Each sample entry is a dictionary like:

```json
{
  "sub": "BraTS-GLI-00002-000",
  "mask": null
}
```

Validation entries can include explicit masks.

Training entries often leave `mask: null` and let the dataset resolve masks dynamically from the masking mode.

## Masking Behavior

Mask handling currently lives across:

- `datasets/base.py`
- `datasets/imfuse.py`
- trainer kwargs selecting masking mode

Important behaviors:

- `MaskingMode.RANDOM`: choose from predefined patterns
- explicit `mask` in split file: use that exact mask
- validation/testing paths may rely on deterministic masks

If you extend masking logic, change both:

1. dataset resolution behavior
2. trainer/config/documentation expectations

## Validation Checklist

Before calling a dataset extension done, verify:

- the dataset can build samples from the split payload
- sample transforms still receive tensors in the expected shapes
- masks behave correctly for train vs val/test
- the model sees the expected modality order
- testing can iterate the full split without shape errors
