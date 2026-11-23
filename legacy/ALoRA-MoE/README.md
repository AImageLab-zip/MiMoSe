# Accurate brain tumor segmentation from incomplete MRI data

This repository provides a **minimal-yet-complete PyTorch implementation** of a **Mixture‑of‑Experts (MoE)** encoder with a **Holistically Guided Decoder (HGD)** for robust brain tumor segmentation under **missing MRI modalities**.

> Educational reference implementation with a tiny synthetic example. Replace the synthetic dataset with BraTS2018 (NIfTI) to train for real.

---

## Features
- 3D UNETR‑style encoder with **LoRA** adapters and **MoE** FFN blocks
- **Holistically Guided Decoder** (codebook‑guided upsampling at stride 8)
- Robustness via **random missing‑modality simulation**
- **Curriculum training**: WT → TC → ET
- Sliding‑window inference utilities
- Clean config file + single‑GPU training script

---

## Installation
```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick synthetic sanity‑check
This runs on CPU/GPU and overfits a toy dataset to validate the pipeline.
```bash
python scripts/make_dummy_data.py --root data/brats2018 --n 4
python train.py --config configs/brats2018.yaml --epochs 2 --num-workers 0
```

## Training on BraTS (outline)
1. Download BraTS2018 and preprocess (skull‑stripped, co‑registered). Organize as:
```
data/brats2018/
  case_000/
    T1.nii.gz   T1c.nii.gz   T2.nii.gz   FLAIR.nii.gz
    seg.nii.gz  # labels with {0:BG, 1:WT, 2:TC, 3:ET} or convert accordingly
  case_001/ ...
```
2. Update `configs/brats2018.yaml` paths and patch sizes (e.g., 128³).  
3. Start training:
```bash
python train.py --config configs/brats2018.yaml
```

## Inference
```bash
python infer.py --checkpoint checkpoints/best.pt --in-path data/brats2018/case_000 --out-path outputs/case_000
```

## Repository layout
```
lora_moe_hgd/
├─ configs/
│  └─ brats2018.yaml
├─ data/
│  └─ brats2018/
├─ lora_moe_hgd/
│  ├─ datasets.py
│  ├─ transforms.py
│  ├─ patches.py
│  ├─ losses.py
│  ├─ metrics.py
│  ├─ engine.py
│  ├─ schedule.py
│  ├─ utils.py
│  └─ model/
│     ├─ patch_embed.py
│     ├─ lora.py
│     ├─ moe.py
│     ├─ encoder.py
│     ├─ hgd_decoder.py
│     └─ seg_model.py
├─ scripts/
│  └─ make_dummy_data.py
├─ train.py
├─ infer.py
└─ requirements.txt
```

## Notes
- This is a compact implementation aimed at clarity. For large‑scale training, enable DDP, gradient checkpointing, and mixed precision.  
- The synthetic example writes small `.npy` volumes to keep this repository self‑contained.

## License
MIT
