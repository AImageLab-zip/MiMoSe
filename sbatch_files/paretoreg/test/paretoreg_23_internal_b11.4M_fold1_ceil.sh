#!/bin/bash
#SBATCH --job-name=PARETOREG23_TEST_INTERNAL_B11.4M_F1_CEIL
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/paretoreg_23_test_internal_b11.4M_fold1_ceil.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/paretoreg_23_test_internal_b11.4M_fold1_ceil.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_L40S_45G
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

# Internal-dataset test of the final paretoreg23 fold-1 supernet at the 11.4M
# budget (11435747), bf16 like its BraTS23 tests. The internal data is the mimosa
# family's simpleunet_internal (same preprocessing as mimosa_preprocessed:
# non_empty crop, min 128^3, no clamp, subject_zscore). Results land next to
# the BraTS23 results of the same budget as results_internal_fold1.*.
# Ceiling rule: the plain 11.4M selection is (64,128,176,148,168) = 11.59M, over budget;
# --budget-ceiling evaluates the largest subnetwork that fits, (64,128,176,144,164) = 11.36M.
# Overwrites the earlier results (copies kept in results/paretoreg_23_b11.4M/pre_ceiling_fold1/).
mimose test \
    --config paretoreg_23.yaml \
    --run-suffix "fold1" \
    --dataset-type internal \
    --split-file /homes/ocarpentiero/MiMoSe/src/mimose/data/splits/internal.json \
    --fold 1 \
    --data-dir /work/phd_mimose/internal_raw/simpleunet_internal \
    --budget 11435747 \
    --budget-ceiling \
    --precision bf16 \
    --num-workers 6 \
    --output-path "/work/phd_mimose/results/paretoreg_23_b11.4M/results_internal.txt"
