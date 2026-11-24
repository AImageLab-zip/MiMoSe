# Learnable Cross-modal Knowledge Distillation for Multi-modal Learning with Missing Modality

[[Paper]](https://arxiv.org/abs/2310.01035) [[Code]](https://github.com/billh hh/LCKD) MICCAI 2023
![LCKD overview](/LCKD/fig/lckd.png)


## Requirements
This code was tested using:
```
python==3.11.5
torch==2.7.1+cu126
torchvision==0.22.1
```
Detailed versioning of every package can be found in the requirements.txt file

This code currently runs only on machines with CUDA support. GPU access is required for all processing steps.
## How to run
Before running any code, ensure you have correctly downloaded the BraTS 2023 Challenge dataset, specifically the subset for [Glioma Segmentation](https://www.synapse.org/Synapse:syn51156910/wiki/622351)\
\
Clone this repository, create a python env for the project and activate it. Then install all the dependencies with pip.
```
cd LCKD
python -m venv lckd_venv
source lckd_venv/bin/activate
pip install -r requirements.txt
```
If you want to track the training using wandb, setup the wandb library following [this guide](https://docs.wandb.ai/models/quickstart).

## Training
To train the model, first start a warmup run:
```
python train.py\
  --datapath <INPUT_PATH> \                    # Directory with the preprocessed dataset                    
  --checkpoint-path <CHECKPOINT_PATH> \        # Directory for saving the checkpoints
  --wandb-project-name LCKD \                  # Optional, allows for wandb tracking
  --num-workers <NUM_WORKERS> \                # Number of workers of the dataloaders
  --batch-size <BATCH SIZE> \                  # Batch size. Start with 4
  --num-steps 307018 \                         # Number of steps for BraTS 2023
```

To resume the warmup, run:
```
python train.py\
  --datapath <INPUT_PATH> \                    # Directory with the preprocessed dataset                    
  --checkpoint-path <CHECKPOINT_PATH> \        # Directory for saving the checkpoints
  --wandb-project-name LCKD \                  # Optional, allows for wandb tracking
  --num-workers <NUM_WORKERS> \                # Number of workers of the dataloaders
  --batch-size <BATCH SIZE> \                  # Batch size. Start with 4
  --num-steps 307018 \                         # Number of steps for BraTS 2023
  --reload-path <RELOAD_PATH> \                # File of the desired checkpoint
  --reload-from-checkpoint True \              # Remember to include this option too
```

After the first warmup, start training with missing modalities:
```
python train.py \
  --datapath <INPUT_PATH> \                    # Directory with the preprocessed dataset                    
  --checkpoint-path <CHECKPOINT_PATH> \        # Directory for saving the checkpoints
  --wandb-project-name LCKD \                  # Optional, allows for wandb tracking
  --num-workers <NUM_WORKERS> \                # Number of workers of the dataloaders
  --batch-size <BATCH SIZE> \                  # Batch size. Start with 4
  --num-steps 307018 \                         # Number of steps for BraTS 2023
  --reload-path <RELOAD_PATH> \                # The checkpoint from the previous training
  --reload-from-checkpoint True \
  --num-steps 441337 \                         # Number of steps for BraTS 2023
  --mode random  \                             # Start dropping modalities
  --restart                                    # Keep only model weights from checkpoint
```
To resume the training, run:
```
python train.py \
  --datapath <INPUT_PATH> \                    # Directory with the preprocessed dataset                    
  --checkpoint-path <CHECKPOINT_PATH> \        # Directory for saving the checkpoints
  --wandb-project-name LCKD \                  # Optional, allows for wandb tracking
  --num-workers <NUM_WORKERS> \                # Number of workers of the dataloaders
  --batch-size <BATCH SIZE> \                  # Batch size. Start with 4
  --num-steps 307018 \                         # Number of steps for BraTS 2023
  --reload-path <RELOAD_PATH> \                # File of the desired checkpoint
  --reload-from-checkpoint True \
  --num-steps 441337 \                         # Number of steps for BraTS 2023
  --mode random                                # Start dropping modalities
```
## Testing
After training, to test the model run:
```
python test.py \
  --datapath <INPUT_PATH> \                    # Directory with the preprocessed dataset
  --savepath <OUTPUT_PATH> \                   # File path for saving results
  --resume <RESUME_PATH> \                     # Path to the checkpoints 
```
