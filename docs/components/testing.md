# Extending Testing

This document explains how to extend the evaluation path.

Relevant files:

- `src/missingbench/checkpoints.py`
- `src/missingbench/testing/pipeline.py`
- `src/missingbench/cli.py`
- model modules implementing `predict(images, mask)`

## Mental Model

The active testing path:

1. loads the test split
2. builds `IMFuseDataset`
3. loads a checkpoint
4. builds the selected model
5. calls `model.predict(images, mask)` across the standard missing-modality masks
6. writes a text report and Excel summary

The active checkpoint contract for testing is:

- testing consumes weights-only `.safetensors` files
- the default local artifact is `art_dir/checkpoints/final_weights_only.safetensors`
- resumable training checkpoints such as `model_last.pth` are not the testing artifact

## What Testing Assumes

The current path assumes:

- CUDA is available
- batch size is `1`
- the model inherits from `AbstractModel`
- the model implements `predict(images, mask)`
- the selected checkpoint is a weights-only `.safetensors` file
- predictions can be reduced to the BraTS-style metrics currently implemented

If your extension breaks one of these assumptions, testing likely needs explicit updates.

## Add a New Metric

1. implement the metric computation in `testing/pipeline.py`
2. decide whether it should:
   - replace the current report
   - be added alongside the current report
3. update report writing
4. update any Excel export logic if needed
5. update docs

## Add a New Mask Sweep Policy

Mask sweep logic is currently driven by the `MASKS` constant.

To change it:

1. update `MASKS`
2. confirm `_mask_name(...)` still makes sense
3. verify report formatting still matches the new mask set
4. check that averages are still computed as intended

## Add Support for a New Evaluation Regime

Examples:

- dataset-specific metrics
- no-mask evaluation
- different class definitions

Typical steps:

1. branch on `dataset_type`
2. implement the metric logic
3. confirm the model output semantics match the metric expectations
4. update docs and output examples

## Validation Checklist

Before calling a testing extension done, verify:

- the selected checkpoint loads cleanly
- the exported checkpoint is a `.safetensors` weights file
- `predict(...)` is actually used
- all masks complete without shape/device errors
- the report file is written
- the Excel summary is still generated if expected
