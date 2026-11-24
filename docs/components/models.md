# Extending Models

This document explains how to add or modify models in detail.

Relevant files:

- `src/missingbench/models/config.py`
- `src/missingbench/models/abstract_model.py`
- `src/missingbench/models/IMFuse.py`
- `src/missingbench/models/mmformer.py`
- `src/missingbench/models/dcseg.py`
- `src/missingbench/models/rfnet.py`
- `src/missingbench/enums.py`

## Mental Model

Model selection is driven by:

1. CLI/YAML value such as `model: dcseg`
2. `build_model_config(...)`
3. dynamic resolution inside `models/config.py`
4. model construction from `model_class(**model_kwargs)`

The runtime expects models to be normal PyTorch modules, but also to conform to the MissingBench inference contract.

## Required Base Contract

All active models should:

1. inherit from `AbstractModel`
2. implement `forward(...)`
3. implement `predict(images, mask)`

The `predict(...)` method is mandatory for `missingbench test`.

`predict(...)` is not just a duplicate of `forward(...)`. It is the test-time inference method used by [src/missingbench/testing/pipeline.py](../../src/missingbench/testing/pipeline.py), where the runtime calls:

```python
output = model.predict(images, mask)
```

That means `predict(...)` must:

1. accept `images` and `mask`
2. run the model in an inference-safe way
3. return a segmentation tensor shaped `[B, C, H, W, D]`

Depending on the model family, `predict(...)` may:

- call `forward(...)` and return only the fused segmentation output
- run sliding-window inference internally
- remap modalities before inference
- disable training-only branches or auxiliary outputs

The important point is that `missingbench test` expects a final segmentation prediction tensor, not the full training tuple returned by some trainer-specific `forward(...)` implementations.

## Add a New Model Module

1. create `src/missingbench/models/my_model.py`
2. define a public model class
3. inherit from `AbstractModel`
4. implement training forward behavior
5. implement `predict(images, mask)`

Minimal shape:

```python
class MyModel(AbstractModel):
    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        ...

    def forward(self, images: torch.Tensor, mask: torch.Tensor):
        ...

    def predict(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        ...
```

If you want the easiest integration path, make the model IMFuse-compatible:

1. start from an `imfuse` YAML config
2. make `forward(...)` return `(fuse_pred, sep_preds, prm_preds)`
3. keep the output shapes consistent with [src/missingbench/losses/imfuse.py](../../src/missingbench/losses/imfuse.py)
4. make `predict(...)` return the final fused segmentation tensor only

This lets you reuse the existing [IMFuseTrainer](../../src/missingbench/training/trainers/imfuse.py) and IMFuse-style training path without creating a new trainer family first.

## How Resolution Works

The model resolver scans modules under `src/missingbench/models/` and compares normalized names.

That means:

- `my_model`
- `MyModel`
- `model: mymodel`

can all resolve if the names normalize to the same alphanumeric string.

If you want the model to be a first-class enum value, also add it to `ModelKind`.

## Trainer Compatibility

Adding a model file is not enough. The model must match the trainer it will be used with.

### IMFuse-style trainer compatibility

For `IMFuseTrainer`, training forward is expected to return:

- `fuse_pred`
- `sep_preds`
- `prm_preds`

This is why `imfuse`, `mmformer`, and `rfnet` share the same trainer path.

### DC-Seg trainer compatibility

For `DCSegTrainer`, training forward is expected to return the richer DC-Seg tuple:

- fused prediction
- separate predictions
- PRM predictions
- reconstruction output
- `mu_list`
- `sigma_list`
- content features
- style features

If your model does not match the trainer’s expected return structure, training will fail even if the model itself is valid PyTorch code.

## Input Order and Mask Semantics

The active data pipeline uses modality order:

- `t1c`
- `t1n`
- `t2f`
- `t2w`

Some models remap internally to legacy orderings such as:

- `flair`
- `t1ce`
- `t1`
- `t2`

If your model expects a different modality order:

1. document it clearly
2. remap at the model boundary
3. remap `mask` consistently with `images`

Do not silently reinterpret channels in the middle of the forward path.

## Add Model-Specific YAML Fields

If your model takes extra kwargs:

1. accept them in `__init__`
2. pass them through `custom_model_kwargs`
3. add them to reference configs
4. document them

Example:

```yaml
custom_model_kwargs:
  num_cls: 4
  fusion_type: RFM
```

## Validation Checklist

Before calling a model extension done, verify:

- `build_model_config(...)` resolves it
- it can be selected from YAML
- it can be selected from CLI
- training forward matches the chosen trainer
- `predict(images, mask)` works under `missingbench test`
- checkpoint save/load works without key mismatches
