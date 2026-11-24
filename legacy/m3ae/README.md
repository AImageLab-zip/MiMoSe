# M3AE: Multimodal Representation Learning for Brain Tumor Segmentation with Missing Modalities
[[Paper]](https://arxiv.org/abs/2303.05302) [[Code]](https://github.com/ccarliu/m3ae)  AAAI 2023
![m3ae overview](/m3ae/fig/M3AE.png)
## Requirements
Code was tested using:
```
python==3.10.12
torch==2.7.1
```

## How to run
Clone this repository, create a python env for the project and activate it. Then install all the dependencies with pip.
```
cd m3ae
python -m venv m3ae_venv
source m3ae_venv/bin/activate
pip install -r requirements.txt
```

## Pre-training
Perform a warm up with all modalities `pretrain.py` with the following arguments:
```
python pretrain.py \
--exp_name m3ae_pretrain \
--batch_size 2 \
--mdp 3 \
--dataset brats23 \
--mask_ratio 0.875 \
--lr 0.0003 
```

## Training
Run the training script `train.py` with the following arguments:
```
python train.py \
--batch_size 2 \
--lr 0.0003 \
--model_type cnnnet \
--seed 999 \
--weight_kl 0.1 \
--feature_level 2 \
--epochs 300 \
--mdp 3 \
--wd 0.0001 \
--deep_supervised \
--patch_shape 128 \
--exp_name m3ae_train 
```

## Test
Run the test script `test.py` with the following arguments:
```
python test.py \
----checkpoint runs/m3ae_train/best.pth.tar \
--exp_name m3ae_train
```
