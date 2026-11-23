#!/bin/bash
#SBATCH --job-name=RFNET23_TEST_INTERNAL_F1
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/rfnet_23_test_internal_fold1.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/rfnet_23_test_internal_fold1.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_L40S_45G
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

# Rerun of the internal test for the brats23-trained rfnet fold1: the previous
# results predated the retrained checkpoint (moved to
# results/rfnet_23/stale_internal_pre_retrain/).
mimose test \
    --config rfnet_23.yaml \
    --run-suffix "fold1" \
    --dataset-type internal \
    --split-file /homes/ocarpentiero/MiMoSe/src/mimose/data/splits/internal.json \
    --fold 1 \
    --data-dir /work/phd_mimose/internal_raw/rfnet_internal \
    --num-workers 6 \
    --output-path "/work/phd_mimose/results/rfnet_23/results_internal.txt"
