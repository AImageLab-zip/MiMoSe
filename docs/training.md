# Training

`missingbench train` launches config-driven training for the active MissingBench package.

The active training stack currently includes:

- `IMFuseTrainer` for:
  - `imfuse`
  - `mmformer`
  - `rfnet`
  - `a2fseg`
  - `plainunet` (Plain U-Net, with `loss: dicece`)
  - `m2ftrans`
  - `mmmvit`
  - `unetmfi`
  - `inoutfusion`
- `DCSegTrainer` for:
  - `dcseg`
- `UHVEDTrainer` for:
  - `uhved`
- `RobustSegTrainer` for:
  - `robustseg`
  - `sfusion`
- `ShaSpecTrainer` for:
  - `shaspec`
- `M3AETrainer` for:
  - `m3ae`
- `M3FeConTrainer` for:
  - `m3fecon`
- `SRMNetTrainer` for:
  - `srmnet`
- `IMS2TransTrainer` for:
  - `ims2trans`
- `MSTKDTrainer` for:
  - `mstkdnet`
- `RFLTrainer` for:
  - `rfl`
- `LCKDTrainer` for:
  - `lckd`
- `MambaVitAKDTrainer` for:
  - `mambavitakd`
- `MatUnetTrainer` for:
  - `matunet` (MatUNet, budget-conditioned supernet, with `loss: dicece`)

The runtime still shares common checkpointing, DDP, WandB, and Rich progress infrastructure across trainers, but the DC-Seg path now has its own trainer and transform wiring.

Training produces two checkpoint classes:

- resumable checkpoints such as `checkpoints/model_last.pth` for training resume only
- an inference/export checkpoint at `checkpoints/final_weights_only.safetensors` containing only model weights

## Basic Usage

Run from YAML:

```bash
missingbench train --config imfuse_23.yaml
```
The CLI automatically resolves and autocompletes config files found in `src/missingbench/data/configs/` (written by `missingbench setup`). Absolute or relative paths for custom yaml files are supported too.
Run from CLI:

```bash
missingbench train \
  --data-dir /path/to/preprocessed \
  --art-dir /path/to/artifacts_dir \
  --trainer dcseg \
  --model dcseg \
  --loss imfuse \
  --optimizer adam \
  --scheduler poly \
  --lr 2e-4 \
  --num-epochs 500 \
  --dataset-type brats23
```

CLI values override YAML values when both are provided.

For the overall MissingBench YAML format, see [docs/yaml-config.md](yaml-config.md).
For very detailed extension notes on trainers, models, and runtime wiring, see [docs/components/README.md](components/README.md).


## Reference Configs

The repo currently ships these reference training configs:

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
- `src/missingbench/data/configs/lckd_23.yaml`
- `src/missingbench/data/configs/inoutfusion_18.yaml`
- `src/missingbench/data/configs/inoutfusion_23.yaml`
- `src/missingbench/data/configs/a2fseg_18.yaml`
- `src/missingbench/data/configs/a2fseg_23.yaml`
- `src/missingbench/data/configs/mambavitakd_18.yaml`
- `src/missingbench/data/configs/mambavitakd_23.yaml`
- `src/missingbench/data/configs/plainunet_b01_23.yaml` ... `plainunet_b16_23.yaml` (Plain U-Net, 16 sizes from 1.9M to 255.6M parameters)
- `src/missingbench/data/configs/matunet_23.yaml` (MatUNet)

Every baseline also has a `_25.yaml` (BraTS 2025) variant. The templates live in `src/missingbench/data/config_templates/`; `missingbench setup` writes the path-patched copies listed above.

They are combined reference files that include preprocess, train, and test sections/fields. The train command reads the training-relevant keys and ignores the rest.

## Core Training Fields

Top-level training fields:

- `data_dir`
- `art_dir`
- `trainer`
- `model`
- `loss`
- `optimizer`
- `scheduler`
- `lr`
- `weight_decay`
- `batch_size`
- `num_epochs`
- `num_workers`
- `precision` (`fp32`, `fp16`, `bf16`)
- `compile`
- `validation_every`
- `resume`
- `pretrain`
- `seed`
- `wandb_project`
- `wandb_mode`
- `wandb_run_name`
- `dataset_type`

Structured extension points:

- `custom_model_kwargs`
- `custom_loss_kwargs`
- `custom_trainer_kwargs`

## Current Built-in Choices

Trainer values:

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

Model values:

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

Loss values:

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

Optimizer values:

- `radam`
- `adamw`
- `adam`
- `sgd`

Scheduler values:

- `poly`
- `warmuppoly`
- `cosine`
- `step`
- `multistep`
- `plateau`

Transform manager values:

- `imfuse`
- `shaspec`

Dataset type values:

- `brats18`
- `brats23`
- `brats25`

Masking mode values:

- `random`
- `validation`
- `test`
- `full`

`plateau_mode` values when `scheduler: plateau`:

- `min`
- `max`

`wandb_mode` values:

- `online`
- `offline`
- `disabled`

## Trainer-Specific Runtime Options

The active trainers consume overlapping but not identical trainer kwargs.

Common examples:

- `iter_per_epoch`
- `region_fusion_start_epoch`
- `patch_size`
- `debug`
- `transform_kind`
- `train_masking_mode`
- `val_masking_mode`
- `split_file`

Selectable values for the constrained trainer kwargs:

- `transform_kind`: `imfuse`, `shaspec`
- `train_masking_mode`: `random`, `validation`, `test`
- `val_masking_mode`: `random`, `validation`, `test`

Example:

```yaml
custom_trainer_kwargs:
  iter_per_epoch: null
  region_fusion_start_epoch: 0
  patch_size: 128
  debug: false
  transform_kind: imfuse
  train_masking_mode: random
  val_masking_mode: validation
```

Current masking defaults:

- training: `random`
- validation: `validation`

### DC-Seg-specific notes

The packaged DC-Seg configs are tuned to follow the maintained DC-Seg port rather than the earlier generic IMFuse defaults.

Current defaults in the shipped DC-Seg configs include:

- `trainer: dcseg`
- `model: dcseg`
- `loss: imfuse` (DC-Seg's auxiliary terms are added by `DCSegTrainer`)
- `transform_kind: imfuse`
- `patch_size: 80`
- `poly_total_iters: 500`
- `precision: fp16`
- `use_recon_loss: true`
- `use_reg_loss: true`
- `use_ana_contrastive: true`
- `use_mod_contrastive: true`

The DC-Seg inference path uses its own sliding-window `predict(...)` implementation inside the model, so training crop size and inference window size are not the same thing.

### RFNet-specific notes

The packaged RFNet configs reuse `IMFuseTrainer` and `IMFuseLoss`, because RFNet
returns the same fused, separate, and PRM prediction tuple during training.
RFNet uses `transform_kind: imfuse` and `patch_size: 80`, matching the legacy
RFNet crop and sliding-window size.

### U-HVED-specific notes

U-HVED (`trainer: uhved`, `model: uhved`, `loss: uhved`) is a hetero-modal
variational encoder-decoder: each modality is encoded independently into a
per-scale Gaussian posterior, fused via a masked product-of-experts, then
decoded back into per-modality reconstructions plus a shared segmentation
head. `UHVEDTrainer` reuses `IMFuseDataset`/`transform_kind: imfuse` like
DC-Seg and RFNet, but with `patch_size: 112` (matching the legacy U-HVED crop
size) and a `plateau` LR scheduler (matching legacy `ReduceLROnPlateau`).

The training loss combines segmentation cross-entropy and Dice with a KL
divergence term (inter-modality and prior, averaged over all 15 non-empty
modality subsets) and a modality reconstruction MSE term. The KL and
reconstruction loss weights are hardcoded to `0.1` each, matching the legacy
U-HVED defaults, and are not exposed as YAML kwargs. The reconstruction term
averages (not sums) the 4 per-modality MSE means before applying that
weight, matching legacy's own single `torch.mean` over one concatenated
4-channel reconstruction tensor — summing the 4 per-modality means without
dividing (an earlier bug in this port) made the term ~4x heavier than
intended relative to its `0.1` weight.

### RobustSeg-specific notes

RobustSeg (`trainer: robustseg`, `model: robustseg`, `loss: robustseg`) splits
each modality into a style code (a global appearance vector) and a multi-scale
content code, gates and fuses the per-modality content at each scale with a
mask-aware attention module, and decodes the fused content into a shared
segmentation head plus (during training only) per-modality image
reconstructions conditioned on that modality's style code.
`RobustSegTrainer` reuses `IMFuseDataset`/`transform_kind: imfuse` like
DC-Seg/RFNet/U-HVED, with `patch_size: 80` (matching the legacy RobustSeg crop
size — the image-reconstruction decoder uses fixed-shape `LayerNorm`s sized
for 80<sup>3</sup> patches, so training crops must stay at that size) and a
`poly` LR scheduler over `num_epochs`, matching legacy `LR_Scheduler_polinomial`.

The training loss combines segmentation cross-entropy and Dice with an L1
modality-reconstruction term and a per-modality style KL term (regularizing
each style code towards `N(0, 1)`, unlike U-HVED's cross-modality KL). Both
extra loss weights are hardcoded to `0.1`, matching the legacy RobustSeg
defaults, and are not exposed as YAML kwargs.

### M2FTrans-specific notes

The packaged M2FTrans configs reuse `IMFuseTrainer` and `IMFuseLoss`, same as
RFNet, because M2FTrans returns the same fused/separate/PRM prediction tuple
during training. Each modality is encoded independently, fused at the deepest
two scales via a mask-aware cross-attention transformer bottleneck (a set of
learned "fusion" tokens cross-attend to the per-modality tokens, with missing
modalities excluded from the attention mask), and fused at the shallower three
scales via simple mask-zero-and-concat convolutions — matching legacy M2FTrans.

Unlike RFNet/DC-Seg/U-HVED/RobustSeg, M2FTrans's bottleneck uses a fixed
positional embedding sized for exactly `patch_size: 80` (`80 / 16 = 5` tokens
per spatial dimension), so both training crops and `predict(...)` sliding-window
patches must be exactly 80<sup>3</sup> — there's no shortcut for other sizes,
only tiling with 50% overlap like the other sliding-window models. M2FTrans
uses `transform_kind: imfuse`, `patch_size: 80`, `region_fusion_start_epoch: 0`,
and `optimizer: adamw` (closest available match to legacy `torch.optim.AdamW`;
MissingBench's `adamw` optimizer config does not set `amsgrad=True` the way legacy
did, a pre-existing framework limitation not specific to M2FTrans).

### SFusion-specific notes

SFusion (`trainer: robustseg`, `model: sfusion`, `loss: robustseg`) reuses
`RobustSegTrainer`/`RobustSegLoss` as-is: it shares RobustSeg's style/content
dual-encoder architecture and produces the exact same training-output
contract (`seg_pred`, and per-modality `reconstruct_*`/`mu_*`/`sigma_*` keys),
just with a different fusion mechanism. Instead of RobustSeg's mask-gated
attention fusion, SFusion average-pools each modality's per-scale content
into an 8×8×8 token grid, refines all 4 modalities' tokens jointly through an
`nn.TransformerEncoder` (self-attention across tokens), reprojects back to
full spatial resolution via a depth-to-space upsample, and combines the
per-modality reprojections with a softmax-weighted sum.

Legacy SFusion assumed one shared missing-modality pattern per training batch
(it built a variable-length token sequence containing only the present
modalities). Since MissingBench's masks are per-sample (`[B, 4]`, potentially
different per row in a batch), this port instead always builds a fixed
4-modality token sequence and excludes missing modalities per-sample via
`src_key_padding_mask` in the transformer and via a masked softmax in the
final weighted combination — mathematically equivalent to the legacy
behavior whenever a batch happens to share one mask, but correct for
heterogeneous batches too.

SFusion trains at `patch_size: 128` (not 80, matching legacy `RandCrop3D`)
with `n_base_filters: 8` (not RobustSeg's 16) and `optimizer: adam` with
`amsgrad=True`, matching legacy `torch.optim.Adam(..., amsgrad=True)`
exactly (unlike M2FTrans, no optimizer mismatch here). Unlike RobustSeg's
image-reconstruction decoder, SFusion's uses `InstanceNorm3d` instead of a
fixed-shape `LayerNorm`, so it isn't tied to the training patch size the way
RobustSeg's `predict(...)` reconstruction path is — though `predict(...)`
still requires 128<sup>3</sup> sliding-window tiles because the transformer
fusion module's depth-to-space upsample uses a fixed integer scale factor
computed from the training-time content resolution.

### ShaSpec-specific notes

ShaSpec (`trainer: shaspec`, `model: shaspec`, `loss: shaspec`) encodes each
modality twice through a heavy 3D ResNet-50 + ASPP backbone: once through a
single weight-shared encoder (run once per modality; legacy ran it once on
all 4 modalities batch-folded together, which is numerically identical here
since every norm layer is `InstanceNorm3d`, so it's just called 4 times
instead) producing a modality-invariant "shared" feature, and once through a
modality-specific encoder. Present modalities compose their shared and
specific features via a residual "compositional layer"; missing modalities
fall back to the shared feature of the first available modality (generalizing
legacy's single-mask-per-batch fallback to MissingBench's per-sample masks, the
same adaptation used for SFusion's fusion module).

Legacy ShaSpec's ResNet stem used an anisotropic stride (only downsampling
H/W, not depth), requiring a non-cubic `(80, 160, 160)` input and an extra
asymmetric final upsample to compensate. This port uses an isotropic stride
instead, so the whole encoder/decoder downsamples/upsamples by 16x uniformly
and works with the cubic patches used everywhere else in MissingBench — at
`patch_size: 80` the bottleneck ends up at the same 5<sup>3</sup> resolution
legacy's ASPP dilation rates (2/4/8) were tuned for, so this isn't expected to
change model capacity, just the input aspect ratio.

Legacy ShaSpec also used a 3-channel sigmoid/BCE+Dice BraTS-region output
(WT/TC/ET, not mutually exclusive) rather than a softmax N-class prediction.
This port instead outputs a standard softmax `num_cls`-channel prediction
(reusing `softmax_weighted_loss`/`dice_loss` from `losses/imfuse.py`, like
every other model here) so it's compatible with `BaseTrainer`'s
`_evaluate_scores`/`predict(...)` contract used by the shared testing
pipeline. The two ShaSpec-specific auxiliary losses are kept with their
original weights: an L1 "shared features should agree across modalities"
term (`shared_similarity`, weight `0.1`, legacy "alpha") over all 4 cyclic
modality pairs, and a cross-entropy "specific encoders should be
modality-discriminative" term (`domain_cls`, weight `0.02`, legacy "beta").
Both weights are hardcoded, not exposed as YAML kwargs, matching legacy.
ShaSpec uses `optimizer: sgd` with `momentum: 0.99`, `nesterov: true`, matching
legacy's actual optimizer choice (unlike the commented-out Adam/AdamW
alternatives in the legacy script).

### M3AE-specific notes

M3AE (`trainer: m3ae`, `model: m3ae`, `loss: m3ae`) is architecturally the
simplest model here: all 4 modalities are concatenated as input channels to a
single shared 3D U-Net (GroupNorm residual blocks, additive — not
concatenated — skip connections), so there's no per-modality encoder/decoder
family to speak of. What distinguishes it is how missing modalities are
handled: instead of zero-filling (every other model here), missing modality
channels are replaced with a single **learned** per-voxel embedding
(`self.limage`), fixed at the training patch size.

Legacy M3AE is actually a two-stage pipeline (`pretrain.py` then `train.py`,
run as separate scripts producing separate checkpoints). `M3AETrainer`
reproduces this as a single, resume-safe, epoch-driven schedule within one
run rather than two scripts/checkpoints — see `custom_trainer_kwargs` below.
One piece is still not reproduced:

- Legacy fine-tuning re-samples each training crop 2-3 times with
  *independently random* missing-modality patterns and adds an MSE
  consistency loss between their encoder features — this assumes multiple
  forward passes per sample per step, which doesn't fit MissingBench's
  one-mask-per-sample contract (the dataset decides the mask once per
  sample, matching every other trainer here). This port drops that
  consistency term; fine-tuning trains with a plain softmax cross-entropy +
  Dice segmentation loss (`M3AELoss`, reusing `losses/imfuse.py`'s
  functions), like the simplest models in this framework.

What *is* reproduced, gated behind `custom_trainer_kwargs.pretrain_fraction`
(default `0.0`, i.e. disabled — single-stage, from-scratch training as
before):

- For the first `pretrain_fraction * num_epochs` epochs, the trainer runs a
  reconstruction stage: `M3AE.forward(x, mask, mode="reconstruct")`
  reconstructs the original, unmasked input from the (dataset-)masked one,
  trained with MSE plus a smoothness regularizer on `limage`
  (`m3ae_reconstruction_loss` in `losses/m3ae.py`, matching legacy's
  `loss_ + loss2 * .005`). `limage` gets its own optimizer param group at
  `limage_lr` (default `0.005`, matching legacy) while the rest of the
  network trains at the usual `lr`.
- For the remaining epochs, `limage` is frozen (`requires_grad_(False)`,
  reproducing legacy's fine-tuning optimizer excluding it entirely) and
  training switches to the normal segmentation loss/validation (dice, HD95).
  The LR scheduler restarts a fresh anneal at this boundary (matching
  legacy's two independent per-stage `CosineAnnealingLR` schedules, rather
  than one continuous decay across both stages).
- Legacy's `limage` is full-BraTS-volume sized with location-indexed spatial
  patch masking; MissingBench's `limage` is fixed at the training patch size (a
  pre-existing port decision, see below), so there is no spatial location to
  index — the pretraining stage reuses the same per-sample, per-modality
  masking (`train_masking_mode`/`val_masking_mode`) as fine-tuning, rather
  than a separate spatial-masking scheme.

Legacy M3AE also used a 3-channel sigmoid/Dice BraTS-region output; like
ShaSpec, this port instead uses a standard softmax `num_cls`-channel head for
compatibility with the shared testing pipeline. M3AE trains at
`patch_size: 128` (matching legacy), `optimizer: adam`, and `scheduler: cosine`
(matching legacy's `CosineAnnealingLR`), unlike every poly-scheduled model
elsewhere in this file.

### M3FeCon-specific notes

M3FeCon ("Missing as Masking", `trainer: m3fecon`, `model: m3fecon`, `loss: m3fecon`) is
built on top of a heavily customized nnU-Net fork rather than a standalone
model repo. Each modality gets its own plain nnU-Net-style conv encoder (6
stages, matching legacy's `3d_fullres` plan at `base_num_features: 32` capped
at 512, split evenly across the 4 per-modality encoders: channel schedule
`[8, 16, 32, 64, 128, 128]`). At the bottleneck, missing modalities' features
are replaced with a learned token, and a small transformer (`nn.TransformerEncoder`,
matching legacy's 1-layer/4-head reconstruction transformer) jointly refines
all 4 modalities' bottleneck tokens so the missing ones get reconstructed
from cross-modal attention with the present ones. At shallower scales,
missing modalities are simply zeroed before per-stage skip connections are
concatenated across modalities and decoded by a standard U-Net decoder — no
PRM/RFM machinery like the DCSeg/RFNet family.

The training loss adds an MSE "feature reconstruction" term, computed only
over modalities that were actually missing in that sample's mask (the
reconstructed bottleneck feature vs. that modality's own true — detached —
encoder output), unweighted (conceptually matching legacy, which adds
`mseloss` to the segmentation loss with weight 1 — see caveat below).
Legacy's own `train_step`/`validation_step` picked one shared
missing-modality pattern for the whole batch (`random.choice` over the same
15 combinations used by DCSeg/RFNet/RobustSeg/ShaSpec); this port instead
respects MissingBench's per-sample `[B, 4]` masks, so the reconstruction loss is
averaged only over each sample's own missing channels. M3FeCon trains at
`patch_size: 128` (matching legacy's `3d_fullres` plan) with
`optimizer: sgd`, `momentum: 0.99`, `nesterov: true`, and a `poly` LR
schedule, matching nnU-Net's (unmodified) default trainer hyperparameters.

**Caveat on "matching legacy"**: legacy's `nnUNetTrainerMissingRecon` only
backprops the combined seg+MSE loss on the non-AMP (`grad_scaler is None`)
code path; its normal CUDA/AMP training path (`grad_scaler` set) only calls
`.backward()` on the plain segmentation loss — the reconstruction MSE is
computed every step but never actually contributes gradient in legacy's own
standard training runs. Legacy's `calculate_mse_loss` also zips its
reconstruction tensor (indexed along the batch dimension) against the
4-element modality-presence list, so for batch sizes other than 4 the
"missing" flags line up against the wrong axis — a distinct bug in legacy
itself. This port implements the clean, paper-intended semantics (MSE over
missing modalities, gradients always flowing, per-sample masks) rather than
literally reproducing either of legacy's runtime quirks above — a
deliberate improvement, not a literal one-to-one port of what legacy's
script actually does at runtime.

### SRMNet-specific notes

SRMNet ("Style-Robust/Modality-Recalibration Net", `trainer: srmnet`,
`model: srmnet`, `loss: srmnet`) is built on a DCSeg/RFNet-style per-modality
encoder (`basic_dims: 8`, 5 levels) with zero-fill for missing modalities.
At the bottleneck, each present modality's features pass through a
Restormer-style `GlobalBlock` (channel attention + spatial attention +
gated-GELU feed-forward, reimplemented with plain `.reshape` calls instead of
`einops` to avoid an extra dependency). The decoder fuses modalities at every
scale with an `AdaptiveFusionBlock` (a renamed, corrected port of legacy's
`dap`): legacy's version took a `num_cls` constructor argument that was
actually always hardcoded to `num_modals` (4) at every call site, which
happened to work only by coincidence because BraTS-family training always
used 4 modalities — porting it literally would have silently broken BraTS25
(`num_cls=5`), so this port renames the parameter to `num_modals` and wires
it explicitly. `AdaptiveFusionBlock` also originally used a `DeformConvPack`
(deformable 3D convolution) branch that depends on a custom-compiled CUDA
extension unavailable in this environment; it is replaced here with a second
plain `ConvBlock` branch, keeping the same SE-style softmax gate that
dynamically combines the two branches per-channel.

The decoder produces deep-supervision predictions at 4 scales
(`pred4`/`pred3`/`pred2`/`pred1`, upsampled to full resolution) plus a
per-modality image-reconstruction decoder (`self.rec`, L1 loss, weight
`0.1`), matching legacy's multi-term loss. SRMNet trains at `patch_size: 128`
with `optimizer: adam` (matching legacy exactly) and a `poly` LR schedule.

### MMMViT-specific notes

MMMViT (`trainer: imfuse`, `model: mmmvit`, `loss: imfuse`) reuses the
IMFuse-style trainer/loss because its training loop and loss decomposition
(fuse + separated + pyramid region losses, with an optional
`region_fusion_start_epoch` warmup where the fuse-branch loss is zeroed) are
identical to RFNet/mmformer/M2FTrans. Architecturally it is closest to
mmformer: 4 per-modality encoders (`basic_dims: 8`) each produce a multiscale
bottleneck token (an "IntraFormer" `Transformer` self-attention block
refines each modality's own tokens independently), followed by an
"InterFormer" stage that models pairwise modality correlations through a
learned softmax gate over per-modality query/key/value projections, before a
final cross-modality `Transformer` fuses everything into one bottleneck
representation for the decoder.

Legacy computes a masked version of the per-modality IntraFormer tokens
(via its `MaskModal` zero-fill) but then never actually uses it — the
correlation-modeling stage reads the *unmasked* per-modality tokens instead,
silently leaking "missing" modality information into the fused prediction
and undermining the missing-modality contract the rest of the model (and
mmformer's own, correctly masked, equivalent stage) relies on. This port
fixes that by feeding the masked tokens into the correlation-modeling stage.
It also fixes the same `num_cls`/`num_modals` conflation bug seen in SRMNet's
`dap`: legacy's `fusion_prenorm` took a `num_cls` argument that was actually
only ever used as the "number of modalities to concatenate" channel
multiplier (always 4), which coincidentally worked only because BraTS
training always defaulted to `num_cls=4` — this port renames the parameter
to `num_modals` and fixes it at 4 regardless of the real segmentation class
count, so BraTS25 (`num_cls=5`) works correctly. MMMViT trains at
`patch_size: 128` with `optimizer: adam`, `lr: 0.0002`, `batch_size: 1`
(matching legacy's defaults) and a `poly` LR schedule.

### IMS2Trans-specific notes

IMS2Trans (`trainer: ims2trans`, `model: ims2trans`, `loss: ims2trans`) is
the one model in this batch that needed its own new trainer/loss, because
legacy's training contract genuinely differs from the IMFuse family: instead
of 4 per-modality segmentation decoders ("sep" predictions), legacy computes
an InfoNCE-style contrastive loss between each modality's bottleneck
embedding and the cross-modality average embedding, on top of the usual
fuse + pyramid deep-supervision segmentation losses (with the same
`region_fusion_start_epoch` fuse-loss warmup mechanism as the IMFuse family,
default `0` i.e. no warmup).

Legacy's actual backbone is a shared-weight 3D Swin Transformer encoder
(built from `timm`/`monai` blocks — neither is a MissingBench dependency) that is
specialized per modality via learnable additive "modality tokens" injected
into every window-attention block, plus a 3D CutMix augmentation during
training. Hand-rolling a correct 3D shifted-window-attention +
relative-position-bias backbone without those dependencies is
disproportionate to this port's scope, so this port replaces the Swin
backbone with the same per-modality CNN encoder + self-attention
IntraFormer/InterFormer bottleneck already used (and tested) by
mmformer/M2FTrans/MMMViT, and drops the CutMix augmentation entirely — a
training-time regularizer orthogonal to the architecture, in the same spirit
as M3AE dropping its MAE-pretraining stage. What's preserved because it's
IMS2Trans's actually-distinguishing contribution: the deep-supervision
fusion decoder and the contrastive missing-modality-robustness loss
(`dis_weight: 0.1`, temperature `0.5`, matching legacy's `dis_lambda`).

Legacy also has the same `num_cls`/`num_modals` conflation bug found in
SRMNet/MMMViT: `fusion_prenorm`'s channel multiplier is really "number of
modality streams" (always 4), not the true segmentation class count, which
would shape-mismatch for BraTS25 (`num_cls=5`) — `FusionPrenorm` here always
fixes it at 4. IMS2Trans trains at `patch_size: 128` with `optimizer: adam`
(`amsgrad` behavior matched via MissingBench's `adam`), `lr: 0.0002`,
`batch_size: 1` (matching legacy's defaults) and a `poly` LR schedule.

### MST-KDNet-specific notes

MST-KDNet (`trainer: mstkdnet`, `model: mstkdnet`, `loss: mstkdnet`) is a
teacher-student co-training scheme, not a single-forward-pass model: legacy
trains two full-weight instances of the same Myronenko 3D-UNet-with-UNETR-
bottleneck backbone — a "full" branch that always sees the true unmasked
4-channel volume, and a "missing" branch that sees the volume zero-filled
per the modality mask — with several distillation losses pulling the
missing branch toward the full one: bottleneck "content" feature MSE,
UNETR intermediate-deconv-feature MSE, attention "extreme value
distillation" (elementwise max/min/mean of the teacher's/student's 4
transformer-layer attention maps, MSE'd), a logit-standardized KL
divergence between pre-sigmoid logits, and a Gram-matrix-based "global
style match" over 3 bottleneck feature tensors. Confirmed via legacy's own
`eval.py`: **only the missing/student branch is ever evaluated at
inference** — the full/teacher branch is a training-time-only auxiliary,
so this port keeps both as submodules of one model class (mirroring the
teacher/student contract) but `predict()`/inference-mode `forward()` only
exercises the student.

Two things are intentionally not carried over from legacy:
- A small PatchGAN-style discriminator that adversarially aligns the
  missing branch's "style" feature with the full branch's. Legacy weights
  it at only `2e-4` and — seemingly by mistake — reconstructs its Adam
  optimizer from scratch on every single training iteration, discarding all
  of its momentum state each step. Given that bug and the added complexity
  of a second optimizer inside a single-optimizer trainer contract, this
  port drops the discriminator entirely; it is a minor secondary auxiliary,
  not the paper's central contribution.
- Legacy hardcodes `mask.view(1, 4, 1, 1, 1)` when zero-filling the missing
  branch's input, silently assuming `batch_size == 1` (true in all of
  legacy's own configs, but would break or silently misbehave for larger
  batches) — this port uses the actual batch size instead.

Legacy trains at a non-cubic `(160, 192, 128)` patch size (its UNETR
bottleneck, patch embedding, and decoder channel counts are all tuned to
that exact shape); for consistency with every other model in this project,
this port instead uses the standard cubic `patch_size: 128`, giving a
`(128, 16, 16, 16)` bottleneck and a UNETR configured with `patch_size: 4`
over a `(16, 16, 16)` grid (64 tokens) — the same architecture, just
reshaped to a cubic input. MST-KDNet trains with `optimizer: adam`,
`lr: 0.0001`, `weight_decay: 1e-5`, `batch_size: 1`, and a `poly` LR
schedule matching legacy's defaults. The consistency-loss weight ramps up
over the first 20 epochs via legacy's sigmoid rampup schedule
(`consistency_max_weight: 10.0`, `consistency_rampup_epochs: 20.0`); the
distillation-term weights (`unetr_weight: 0.2`, `evd_weight: 1e8`,
`weight_gsm: 1e11`) are copied as-is from legacy and are highly
scale-sensitive to the bottleneck's spatial/channel dimensions.

### RFL-specific notes

RFL (`trainer: rfl`, `model: rfl`, `loss: rfl`) needed its
own new trainer/loss because legacy adds two RFIM ("RFL Feature
Interaction Module") self-consistency MSE terms on top of the usual
fuse + 4×sep + 4×prm cross+dice terms (same `region_fusion_start_epoch`
fuse-loss warmup mechanism as the IMFuse family, default `0`, i.e. no
warmup). Architecturally it is close to mmformer/M2FTrans: 4 per-modality
encoders each feed a per-modality self-attention "IntraFormer" at the
bottleneck, followed by a shared cross-modal "InterFormer" transformer, and
a deep-supervision decoder — but the fusion decoder uses a channel-refine +
spatial-attention-gated upsampling block (`DSAM`) at every scale instead of
plain trilinear upsampling.

The paper's namesake mechanism, `RFIM`, imputes a missing modality's
bottleneck tokens from the most reliably present modality (t1ce, or the
first present modality otherwise — the "global modality") via a reversible,
coupling-style MLP; the same module's algebraic inverse is trained (via
`rfl_weight`) to approximately reconstruct the global modality's real
tokens from the imputed ones, while a separate `forward_weight` MSE term
supervises the forward imputation itself against each modality's own real
(always-available-at-train-time) tokens.

Legacy has two real bugs this port fixes:
- The 4 `RFIM` submodules are constructed in a different modality order
  (`flair, t1, t1ce, t2`) than the mask/encoder convention used everywhere
  else in the model (`flair, t1ce, t1, t2`), so legacy's substitution logic
  silently applies the wrong modality's RFIM module whenever t1ce or t1
  (but not flair/t2) is missing. This port builds the 4 RFIM modules in the
  same order as the mask/encoder convention.
- Legacy decides the "global modality" and which modalities to RFIM-impute
  using only `mask[0]` (the first sample's pattern), and hardcodes the
  bottleneck reshape to batch size 1 — silently applying sample 0's
  missing-modality pattern to every sample in the batch. This port resolves
  the global modality and substitutes tokens per-sample, respecting
  MissingBench's true per-sample `[B, 4]` masks (verified with heterogeneous
  per-sample masks at `batch_size > 1`).

Legacy also has the same `num_cls`/`num_modals` conflation bug found in
SRMNet/MMMViT/IMS2Trans in its `fusion_prenorm` fusion blocks, fixed
here the same way (`FusionPrenorm` always uses `num_modals=4`). A fair
amount of dead legacy code (unused `prm_generator`/`region_aware_modal_fusion`
layers, unused decode-conv modules, several unused loss variants) is simply
not ported. RFL trains at `patch_size: 128` with `optimizer: adam`
(`amsgrad=True` matched via MissingBench's `adam`), `lr: 0.0002`, `batch_size: 1`,
and a `poly` LR schedule.

### UNET-MFI-specific notes

UNET-MFI (`trainer: imfuse`, `model: unetmfi`, `loss: imfuse`) is
architecturally the odd one out in this family: instead of per-modality
encoders sharing a common bottleneck fusion stage, legacy builds 4 fully
**independent** 3D U-Net towers (no shared weights at all) and cross-informs
them at 5 resolutions (encoder stages at 64/128/256 channels, decoder stages
at 128/64 channels) via a "Modality-adaptive Feature Interaction" (`MFIBlock`)
module: each tower's feature map is updated with a per-channel softmax
combination of all 4 towers' globally-pooled features, conditioned on the
`[B, 4]` modality-availability code, added residually. There is no
cross-modal attention or shared bottleneck — each tower otherwise stays
private. Each tower produces 2 internal deep-supervision predictions plus
its own final prediction; the final fused prediction comes from a small
conv over the 4 towers' final logits. This maps directly onto MissingBench's
IMFuse-style contract (`fuse_pred`, 4 `sep_preds`, 8 `prm_preds` — 4
modalities x 2 deep-supervision stages), so this port reuses
`IMFuseTrainer`/`IMFuseLoss` verbatim.

Legacy's `fianl_diff_code_block` constructs 4 independent `Relation1..
Relation4` submodules (one per branch) but its `forward` calls
`self.Relation1` for all 4 branches — `Relation2/3/4` are constructed but
never receive gradient (dead weights). This port fixes that, giving each
branch its own independent relation network, matching what the 4 separate
constructors evidently intended. Legacy also zeroes each modality's raw
input once per sample inside the *dataset* rather than the model; this port
applies the same zero-fill inside the model instead (`MaskModal`), since
MissingBench's dataset hands the model the full raw tensor plus a separate mask.
Finally, legacy outputs 3 sigmoid BraTS-region channels (WT/TC/ET) trained
with a BCE+Dice loss and a `class_nums` hardcoded to 3 in `train.py`; like
every other model in this framework, this port instead uses a standard
softmax `num_cls`-channel head (reusing `softmax_weighted_loss`/`dice_loss`)
so it generalizes to BraTS25's 5 classes. UNET-MFI trains at
`patch_size: 128` with `optimizer: adam`, `lr: 0.00005`,
`weight_decay: 0.00001`, `batch_size: 1`, `num_epochs: 1200`, and a `poly`
LR schedule, matching legacy's defaults.

### LCKD-specific notes

LCKD (`trainer: lckd`, `model: lckd`, `loss: lckd` — the legacy repo folder
is named "LCKD", but its live model class, the one actually imported by its
own `train.py`, is called `DualNet`) needed its own new trainer/loss because
its training contract genuinely differs from the IMFuse family: instead of
fuse/sep/prm decoder branches, legacy returns a single segmentation
prediction plus a scalar feature-distillation loss term computed inside the
forward pass itself.

Despite the "KD" in the name, LCKD is **not** a teacher-student pair of full
networks like MST-KDNet. It is a single weight-shared, per-modality
ResNet50+ASPP encoder / U-Net decoder: all 4 modalities are batch-folded
through the *same* encoder (`[N, 4, D, H, W] -> [N*4, 1, D, H, W]`),
producing one 256-channel bottleneck feature map per modality (zero-filled
first for any missing modality). The namesake mechanism is a
**feature-level self-distillation regularizer**: legacy pulls each present
modality's bottleneck feature (detached, as a "teacher") toward every other
present modality's feature via an L1 loss, where the *set* of teacher
modalities is chosen dynamically — legacy re-runs validation-set Dice
scoring every `val_pred_every` training iterations and picks whichever
modality currently scores best per region as the next chunk's teacher(s).
That curriculum requires a mid-training, non-differentiable validation loop
wired directly into the optimization step, which has no natural place in
MissingBench's per-batch trainer contract, so this port replaces it with a fixed,
symmetric scheme: **every modality present in a sample distills into every
other present modality present in that same sample, equally, every training
step** (an all-pairs L1 over the currently-present subset). This keeps the
actual distinguishing mechanism — cross-modal feature-consistency
regularization that compensates for missing modalities — without depending
on an external, dice-score-driven teacher schedule. After distillation, any
still-missing modality's feature is filled with the mean of that sample's
present-modality features (legacy's "fts fill in process"), and all 4
modality feature maps are channel-concatenated (bottleneck and every
skip-connection scale alike) before a single shared U-Net decoder produces
the final prediction.

Two real issues were fixed rather than faithfully reproduced:
- Legacy applies one global `mode` string (which modalities are considered
  "available") to an entire training batch, the same batch-level masking
  bug found in ShaSpec/MMMViT. This port respects MissingBench's true
  per-sample `[B, 4]` mask for both the missing-modality fill-in and the
  distillation loss, looping over the batch dimension (cheap at the small
  batch sizes 3D segmentation uses).
- Legacy's segmentation head outputs 3 BraTS-region (ET/WT/TC) channels
  trained with sigmoid BCE + Dice — a hardcoded assumption that never
  generalizes to a disjoint `num_cls`-class one-hot target and would
  silently break for BraTS25 (5 classes). This port's head instead outputs
  `num_cls` channels with a softmax, reusing `softmax_weighted_loss`/
  `dice_loss` from `losses/imfuse.py`, matching every other model here.

Legacy's optional self-/cross-attention bottleneck refinement
(`nn.MultiheadAttention`) is constructed by the model class but never
enabled by legacy's own `train.py` entrypoint (always called with
`self_att=False, cross_att=False`) — dead code, not ported. Legacy's
`loss_Dual.py` also defines several loss classes (`KDLoss`, `DomainClsLoss`,
`DomainBCELoss`, `CELoss4MOTS`, `DiceLoss4MOTS`, `BCELossBoud`) that
`train.py` never imports; only `DiceLoss4BraTS`/`BCELoss4BraTS` (replaced
here by the softmax+Dice terms above) are on the live training path.

Legacy trains at a non-cubic `(80, 160, 160)` (D, H, W) patch — its ResNet
stem downsamples depth by 16x but height/width by 32x, with a final decoder
upsample that only doubles height/width to compensate. That asymmetry is
self-consistent for any patch shape divisible appropriately, including a
cubic one, so this port trains at the standard cubic `patch_size: 128`
(divisible by 32) without changing the architecture at all. LCKD trains
with `optimizer: sgd`, `momentum: 0.99`, `nesterov: true`, `lr: 0.01`,
`weight_decay: 0.0005` (all matching legacy's actual `train.py` defaults
exactly) and a `poly` LR schedule. The distillation term is weighted by
`kd_weight: 0.1`, matching legacy's `kd_wt`.

### InOutFusion-specific notes

InOutFusion (`trainer: imfuse`, `model: inoutfusion`, `loss: inoutfusion` —
the only model class legacy's `train.py` actually wires up is
`RsInOut_U_Hemis3D`; legacy's `Network_RMBTS.py`/`Network_LMCR.py` define
entirely separate model/loss families that are never selected by the
default training command and are not ported) reuses `IMFuseTrainer`'s
generic training loop with a bespoke `InOutFusionLoss` whose
`training_loss` returns a dict with the same keys `IMFuseTrainer` reads
(`fusecross`/`fusedice`/`sepcross`/`sepdice`/`prmcross`/`prmdice`) — the
unused `sep`/`prm` slots are always zero, since legacy's `U_Hemis_loss` is a
single weighted multi-class Dice term with no cross-entropy term, no
per-modality decoders, and no deep supervision (matching `forward()`, which
only ever returns one segmentation tensor plus two empty tuples).

Architecture ("Hierarchical In-Out Fusion"): 4 independent per-modality
encoders (`RSEncoder`, a 4-level stack of Restormer-style blocks — a
channel-wise self-attention branch, cheap regardless of spatial resolution,
fused with a local residual-conv branch via a small conv FFN). At the
bottleneck and at each of the 3 encoder skip levels, `OutFusion` tokenizes
every modality's feature map (average-pooled to a fixed grid), mixes the
per-modality tokens with a cross-modality mixer, and reprojects the mixed
tokens into a per-modality spatial attention map used to softmax-combine
the original per-modality features. The bottleneck uses a "deep" mixer
(self-attention + a frequency-domain spectral-gating branch combined via a
learned gate); the 3 skip levels use a cheaper "shallow" mixer (a single
self-attention layer). A single U-Net-style decoder consumes the fused
bottleneck and 3 fused skip connections.

Legacy resolves the batch's missing-modality pattern from a single scalar
(`m_d[0]`, since legacy's dataloader hard-asserts `batch_size == 1`) and
only ever encodes the *present* modalities (a variable-length list), which
cannot be batched with MissingBench's true per-sample `[B, 4]` masks. This port
always runs all 4 encoders and instead zero-fills token blocks plus masks
attention keys and the final modality-softmax logits for modalities missing
on a per-sample basis, which is equivalent to legacy's exclusion behavior
under `batch_size == 1` and generalizes correctly to batched, per-sample
masks. Legacy's `general_dice_loss` also hardcodes per-class Dice weights
`[0.1, 0.2, 0.3, 0.4]` tied to exactly 4 classes — the same
`num_cls`-vs.-class-count conflation bug found in several other ported
models, here living in the loss rather than a module constructor — this
port generalizes it to `weight_i = (i + 1) / sum(1..num_classes)`, which
reproduces legacy's exact weights for 4 classes and extends sensibly to
BraTS25's 5.

Two legacy dependencies are not portable to this environment and are
replaced with self-contained plain-torch equivalents preserving the same
idea: the bottleneck mixer's frequency-domain branch (`AF_3d`) vendors a
CVNets/AFFNet `AFNO1D_channelfirst` block requiring an unrelated `opts`
argparse configuration object, replaced with a self-contained real-FFT
spectral-gating unit; the skip-level mixer (`LTEncoderLayer`) wraps a pip
`local-attention` package configured as *causal* with `window_size=128`
(causal windowing has no meaning for volumetric tokens with no sequential
order — an artifact of reusing an NLP-oriented library without adapting
its settings), replaced with plain non-causal, full-sequence masked
self-attention. InOutFusion trains at `patch_size: 128` with
`optimizer: adamw`, `lr: 0.0003`, `weight_decay: 0.0001`, `batch_size: 1`,
`num_epochs: 200` (matching legacy's `train.py` defaults). Legacy's LR
schedule linearly decays from `lr` to `min_lr` over the last `decay_epoch`
epochs, but with legacy's own default (`decay_epoch: 0`) this condition
never triggers within a normal run — i.e. legacy trains at a constant LR in
practice, reproduced here via `scheduler: poly` with `poly_power: 0.0`
(a "poly" multiplier of `(1 - progress)**0 == 1` for the whole run).

### Plain U-Net notes

Plain U-Net (`trainer: imfuse`, `model: plainunet`, `loss: dicece`) is a
uniform-width nnU-Net-style 3D U-Net (`PlainUnet`, built on
`dynamic_network_architectures`' `PlainConvUNet`) that takes the 4 modalities
as input channels of a single joint encoder. Missing modalities are
zero-filled at the input, and the bottleneck is conditioned on the
modality-availability mask through a small FiLM-style MLP; beyond that it has
no missing-modality-specific machinery. Inference uses sliding-window
`predict()` over 128<sup>3</sup> tiles. Sixteen sizes are
shipped (`plainunet_b01_23.yaml` ... `plainunet_b16_23.yaml`, from 1.9M to
255.6M parameters); all share the same recipe (`patch_size: 128`,
`optimizer: adamw`, `lr: 0.0003`, `batch_size: 3`, `num_epochs: 1000`, `poly`
schedule, `precision: bf16`) and differ only in
`custom_model_kwargs.features_per_stage`. The `dicece` loss (`DiceCELoss` in
`losses/dice_ce.py`) is a weighted sum of soft Dice and cross-entropy
(`dice_weight`, `ce_weight`).

### MatUNet notes

MatUNet (Matryoshka U-Net; `trainer: matunet`, `model: matunet`,
`loss: dicece`) is a budget-conditioned supernet: one set of shared weights
(`MatUnet`) serves a whole range of parameter budgets through learnable
channel gates, and every budget selects a nested Plain U-Net subnetwork. The
shipped `matunet_23.yaml` spans uniform 1/8 width up to the full
`(64, 128, 256, 512, 1024)` width, i.e. ~1.43M to ~91.4M parameters.

`MatUnetTrainer` extends `IMFuseTrainer`. Each optimizer step samples
`budgets_per_step` budgets, evaluates the corresponding subnetworks on the
same batch, and steps once on the accumulated gradient. The objective for a
budget `b` is the segmentation loss plus a one-sided budget-overshoot penalty
weighted by `lambda_param`. The gate temperature is annealed geometrically
from `tau_start` to `tau_end` over `num_epochs`. The shipped config uses
`budget_sampling: sandwich` (the widest subnetwork is always included);
`local` and `global` sampling are also available.

Two optional regularisers exist as trainer options but are **off** in the
shipped config: `inplace_distill` (the widest subnetwork's detached output is
used as a soft target for the other sampled subnetworks, mixed with
`distill_alpha`) and `student_modality_drop` (non-widest subnetworks see one
fewer available modality than the widest one).

To evaluate a trained MatUNet at a given budget, pass the budget in
parameters to `missingbench test` (see [docs/testing.md](testing.md)):

```bash
missingbench test --config matunet_23.yaml --budget 30000000 --budget-ceiling
```

`missingbench flops --config matunet_23.yaml --budget <params>` profiles the
standalone network that budget selects.

## Model Kwargs

`custom_model_kwargs` are passed directly to the selected model class.

Current examples:

For `imfuse`:

```yaml
custom_model_kwargs:
  interleaved_tokenization: false
  mamba_skip: false
  num_cls: 4
```

For `mmformer`:

```yaml
custom_model_kwargs:
  num_cls: 4
```

For `dcseg`:

```yaml
custom_model_kwargs:
  num_cls: 4
  fusion_type: RFM
```

For `plainunet`:

```yaml
custom_model_kwargs:
  num_cls: 4
  input_shape: [128, 128, 128]
  features_per_stage: [9, 18, 37, 72, 152]  # b01; the 16 plainunet_bXX configs differ only here
```

For `matunet`:

```yaml
custom_model_kwargs:
  num_cls: 4
  input_shape: [128, 128, 128]
  features_per_stage: [64, 128, 256, 512, 1024]  # widest subnetwork
  group_size: 4
  tau: 0.05
```

Selectable model-specific values currently exposed in docs:

- `dcseg.fusion_type`: `RFM`, `gated`

For `rfnet`:

```yaml
custom_model_kwargs:
  num_cls: 4
```

For `uhved`:

```yaml
custom_model_kwargs:
  num_cls: 4
```

For `robustseg`:

```yaml
custom_model_kwargs:
  num_cls: 4
```

For `m2ftrans`:

```yaml
custom_model_kwargs:
  num_cls: 4
```

For `sfusion`:

```yaml
custom_model_kwargs:
  num_cls: 4
```

For `shaspec`:

```yaml
custom_model_kwargs:
  num_cls: 4
```

For `m3ae`:

```yaml
custom_model_kwargs:
  num_cls: 4
```

For `m3fecon`:

```yaml
custom_model_kwargs:
  num_cls: 4
```

For `srmnet`:

```yaml
custom_model_kwargs:
  num_cls: 4
```

For `mmmvit`:

```yaml
custom_model_kwargs:
  num_cls: 4
```

For `ims2trans`:

```yaml
custom_model_kwargs:
  num_cls: 4
```

For `mstkdnet`:

```yaml
custom_model_kwargs:
  num_cls: 4
```

For `rfl`:

```yaml
custom_model_kwargs:
  num_cls: 4
```

For `unetmfi`:

```yaml
custom_model_kwargs:
  num_cls: 4
```

For `lckd`:

```yaml
custom_model_kwargs:
  num_cls: 4
```

For `inoutfusion`:

```yaml
custom_model_kwargs:
  num_cls: 4
```

You can also override these from the CLI:

```bash
missingbench train \
  --model imfuse \
  --custom-model-kwargs interleaved_tokenization=True \
  --custom-model-kwargs mamba_skip=True
```

## WandB

Training supports:

- `wandb_project`
- `wandb_mode`
- `wandb_run_name`

Allowed `wandb_mode` values:

- `online`
- `offline`
- `disabled`

`wandb_run_name` is optional and currently defaults to `training`.

Example:

```bash
missingbench train \
  --config src/missingbench/data/configs/mmformer_23.yaml \
  --wandb-run-name mmformer-ablation-01
```

The training launch panel also prints the resolved run name.

If W&B logging is enabled but there is no active login in the terminal, MissingBench now fails with a clean CLI message telling you to run `wandb login` or disable W&B with `--wandb-mode disabled`.

## Distributed Training

Distributed relaunch can be configured either in YAML or from the CLI.

Runtime fields:

- `distributed`
- `nproc_per_node`

CLI flags:

- `--distributed`
- `--no-distributed`
- `--nproc-per-node`

When distributed mode is enabled outside an existing `torchrun` launch, MissingBench relaunches itself through `torchrun`.


## Example YAML

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
  anatomy_contrastive_method: ssim

optimizer: adam
betas: [0.9, 0.999]

scheduler: poly
poly_total_iters: 500
poly_power: 0.9

distributed: true
nproc_per_node: 4

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
wandb_run_name: training
dataset_type: brats23
```

`anatomy_contrastive_method` values for DC-Seg:

- `cos_sim`
- `ssim`

## Related Docs

- [README.md](../README.md)
- [docs/testing.md](testing.md)
- [docs/extending.md](extending.md)
