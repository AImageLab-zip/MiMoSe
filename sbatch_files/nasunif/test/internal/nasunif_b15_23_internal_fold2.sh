#!/bin/bash
#SBATCH --job-name=NASUNIFB1523_TEST_INTERNAL_F2
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/nasunif_b15_23_test_internal_fold2.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/nasunif_b15_23_test_internal_fold2.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_L40S_45G
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

# Internal-dataset test of the nasunif b15 model trained on BraTS23 fold 2
# (weights from the AImageLab-Zip/mimose_runs dataset repo). bf16, which is what
# reproduces its stored BraTS23 per-subject scores most closely; internal data is
# the plain-U-Net copy (simpleunet_internal), same preprocessing as mimosa_preprocessed.
mimose test \
    --config nasunif_b15_23.yaml \
    --run-suffix "fold2" \
    --dataset-type internal \
    --split-file /homes/ocarpentiero/MiMoSe/src/mimose/data/splits/internal.json \
    --fold 1 \
    --data-dir /work/phd_mimose/internal_raw/simpleunet_internal \
    --precision bf16 \
    --num-workers 6 \
    --output-path "/work/phd_mimose/results/nasunif_b15_23/results_internal.txt"
