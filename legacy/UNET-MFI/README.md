# Modality-adaptive Feature Interaction for Brain Tumor Segmentation with Missing Modalities

[[Paper]](https://conferences.miccai.org/2022/papers/322-Paper0649.html) [[Code]](https://github.com/zzc1909/UNET-MFI) MICCAI 2022
![MODEL_NAME overview](/UNET-MFI/unet-mfi.png)


## Requirements
This code was tested using:
```
python==3.11.5
torch==2.9.1+cu128
```
Detailed versioning of every package can be found in the requirements.txt file
<>
This code currently runs only on machines with CUDA support. GPU access is required for all processing steps.
## How to run
Before running any code, ensure you have correctly downloaded the BraTS 2023 Challenge dataset, specifically the subset for [Glioma Segmentation](https://www.synapse.org/Synapse:syn51156910/wiki/622351)\
\
Clone this repository, create a python env for the project and activate it. Then install all the dependencies with pip.
```
cd UNET-MFI
python3.11 -m venv unetmfi_venv
source unetmfi_venv/bin/activate
pip install -r requirements.txt
```
If you want to track the training using wandb, setup the wandb library following [this guide](https://docs.wandb.ai/models/quickstart).
## Preprocessing
Before training, run the preprocessing script:
```
python preprocess.py \
  --input-path <INPUT_PATH>                    # Directory containing the unprocessed BraTS dataset
  --output-path <OUTPUT_Path>                  # Directory that will contain the preprocessed dataset
  --interactive                                # Optional, runs the process interactively
```
## Training
To train the model, first start a warmup run:
```
python train.py \
  --datapath <INPUT_PATH> \                    # Directory with the preprocessed dataset
  --num-epochs <NUM_EPOCHS> \                  # Number of epochS
  --checkpoint-path <CHECKPOINT_PATH> \        # Directory for saving the checkpoints
  --wandb-project-name PROJECT NAME \          # Optional, allows for wandb tracking
  --num-workers <NUM_WORKERS> \                # Number of workers of the dataloaders
  --batch-size <BATCH SIZE> \                  # Batch size. Start with 1
```
## Testing
After training, to test the model run:
```
python test.py \
  --datapath <INPUT_PATH> \                    # Directory with the preprocessed dataset
  --savepath <OUTPUT_PATH> \                   # File path for saving results
  --resume <RESUME_PATH> \                     # Path to the checkpoints 
```
