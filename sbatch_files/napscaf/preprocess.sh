#!/bin/bash
#SBATCH --job-name=NAPSCAF_PREPROCESS
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=2:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/napscaf_preprocess.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/napscaf_preprocess.out
#SBATCH --gres=gpu:0
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate

# Legacy NAP-SCAF preprocessing (no crop, per-slice 1/99 percentile uint8 normalisation, see
# legacy/NAP-SCAF/scripts/preprocess_brats_png.py) into napscaf-legacy-preprocessed.
# One BraTS23 pass covers both datasets: BraTS18 runs select their subjects through the split files.
mimose preprocess --config napscaf_23.yaml --dataset-type brats23 --yes
