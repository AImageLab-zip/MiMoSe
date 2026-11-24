# MambaVit-AKD:Mamba-Vision Transformer with Attention Knowledge Distillation for Missing-modality Brain Tumor Segmentation

This repository is the official PyTorch implementation of our work : Mamba-Vision Transformer with Attention Knowledge Distillation for Missing-modality Brain Tumor Segmentation .

# Setup

## Environment

All our experiments are implemented based on the PyTorch framework with 45GB NVIDIA A40 GPU , and we recommend installing the following package versions :

* Python 3.10.9
* PyTorch 2.1.2

# Data preparation

* Download the preprocessed datasets BraTS2020 and BraTS2021 from [kaggle](https://www.kaggle.com/datasets) .
* Process and store the data in NumPy array format ( .npy ) , see [RFNet](https://github.com/dyh127/RFNet/tree/main/data) for more details .
* The folder structure is assumed to be:

```plaintext
data/
├── BRATS2020_data/
│   ├── seg
│   │   ├── BraTS20_Training_001_seg.npy
│   │   ├── BraTS20_Training_002_seg.npy
│   │   ├── BraTS20_Training_003_seg.npy
│   │   ├──......
|   ├── vol
│   │   ├──BraTS20_Training_001_vol.npy
│   │   ├──BraTS20_Training_002_vol.npy
│   │   ├──BraTS20_Training_003_vol.npy
│   │   ├──......
|   ├── text.txt
|   ├── train.txt
│   └── vol.txt 
├── BRATS2021_data/
│   ├── seg/
│   │   ├──......
│   │   ├──......
│   ├── vol/
│   │   ├──......
│   │   ├──......
│   └── ......
└── ......
```

# Running Experiments

Experiments can be launched via the following commands.

* Train the RFNet model and generate the teacher_model(Use complete modalities) :

```python
python train.py
```

* Using the obtained teacher model and all modalities with designed random masking, we commence formal training. For the BRATS2020 dataset, it is split into 219:50:100 (train:val:test) and trained for 1,000 epochs. For the BRATS2021 dataset, it is split into 1000:125:126 and trained for 200 epochs. The learning rate is set to 1e-5 for both.

```python
python trainer.py
```

# Results

<img width="798" height="636" alt="image" src="https://github.com/user-attachments/assets/47a7f225-645b-4ae0-a737-1bac634abc87" />

Comparison of segmentation results on the BraTS2020 dataset (green indicates WT,blue indicates TC, and red indicates ET). From top to bottom, the
number of available modalities increases progressively.

<img width="1000" height="455" alt="image" src="https://github.com/user-attachments/assets/f20e83fc-845e-4f14-90dc-e91720729149" />

Comparison with state-of-the-art methods including RobustSeg, mmFormer, and M2FTrans on the BraTS2020 dataset. Complete, Core, and Enhancing represent the
Dice scores for whole tumor, tumor core, and enhancing tumor, respectively.
