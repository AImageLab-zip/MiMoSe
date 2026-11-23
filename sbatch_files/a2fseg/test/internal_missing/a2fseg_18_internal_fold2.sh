#!/bin/bash
#SBATCH --job-name=A2FSEG18_TEST_INTERNAL_F2
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/a2fseg_18_test_internal_fold2.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/a2fseg_18_test_internal_fold2.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_L40S_45G
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

# Internal-dataset test of the brats18-trained a2fseg fold2 checkpoint
# (one of the folds missing from results/a2fseg_18/results_internal_fold*.txt).
mimose test \
    --config a2fseg_18.yaml \
    --run-suffix "fold2" \
    --dataset-type internal \
    --split-file /homes/ocarpentiero/MiMoSe/src/mimose/data/splits/internal.json \
    --fold 1 \
    --data-dir /work/phd_mimose/internal_raw/a2fseg_internal \
    --num-workers 6 \
    --output-path "/work/phd_mimose/results/a2fseg_18/results_internal.txt"
