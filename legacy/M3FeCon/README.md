# Missing as Masking: Arbitrary Cross-modal Feature Reconstruction for Incomplete Multimodal Brain Tumor Segmentation
[[Paper]](https://papers.miccai.org/miccai-2024/paper/0067_paper.pdf)  MICCAI 2024
![Missing as Masking overview](/M3FeCon/fig/M3FeCon.png)
## Requirements
Code was tested using:
```
python==3.10.12
torch==2.7.1
```
## How to run
Clone this repository, create a python env for the project and activate it. 
```
cd M3FeCon
python -m venv m3fecon_venv
source m3fecon_venv/bin/activate
```
Install nnUNet into the project folder.
```
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
cd ..
```
Now, install the requirements
```
pip install -r requirements.txt
```
## nnUNet setup
nnU-Net requires a strict directory structure. Before proceeding to the next step, set up the dataset directory running this script:
```
python prepare_dataset.py \
  --input-dir <INPUT_DIR> \     # Input directory, containing the uncompressed and unprocessed BraTS2023 glioma dataset
  --output-dir <OUTPUT_DIR> \   # Output directory for the reworked dataset structure
```
This script will create a new directory at OUTPUT_DIR and copy-rename all the files.
Use the BRATS2023 glioma training directory, since it's the only one with tumor annotation. If you want to use your own dataset, skip this phase and reproduce the correct directory structure yourself.

```nnUNet_raw/
└── Dataset138_M3FeCon/
    ├── dataset.json
    ├── imagesTr/
    ├── labelsTr/
    ├── imagesTs/
    └── labelsTs/   
```    
After that, export these fundamental environment variables:
```
export nnUNet_raw=<RESTRUCTURED_DATASET_DIR>
export nnUNet_preprocessed=<PREPROCESSED_DATASET_DIR>
export nnUNet_results=<RESULTS_DIR>
```
You will have to do that every time you open a new shell. To do that automatically for every new shell, add the commands to your .bashrc file.
## Preprocess data
First, run the preprocessing pipeline with the following command:
```
nnUNetv2_plan_and_preprocess -d 138 --verify_dataset_integrity
```

## Training
To train the model, run this command:
```
python nnunetv2/run/run_training.py -tr nnUNetTrainerMissingReconBaseline 138 3d_fullres_multiencoder_recon 0 --c

```

## Test
To run the tests, simply run this bash file:
```
bash test_m3fecon.sh \
  --input-path                  # Input directory containing the preprocessed test images
  --output-path                 # Directory for saving the results file
  --checkpoint-path             # Path to the checkpoint

```