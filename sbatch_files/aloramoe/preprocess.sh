#!/bin/bash
#SBATCH --job-name=ALORAMOE_PREPROCESS
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=2:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/aloramoe_preprocess.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/aloramoe_preprocess.out
#SBATCH --gres=gpu:0
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate

# Legacy ALoRA-MoE preprocessing (no crop, whole-volume per-modality z-score, see
# legacy/ALoRA-MoE/lora_moe_hgd/datasets.py) into aloramoe-legacy-preprocessed.
# One BraTS23 pass covers both datasets: BraTS18 runs select their subjects through the split files.
mimose preprocess --config aloramoe_23.yaml --dataset-type brats23 --yes
