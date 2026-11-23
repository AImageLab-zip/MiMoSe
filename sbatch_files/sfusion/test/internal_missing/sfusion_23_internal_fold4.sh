#!/bin/bash
#SBATCH --job-name=SFUSION23_TEST_INTERNAL_F4
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/sfusion_23_test_internal_fold4.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/sfusion_23_test_internal_fold4.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_L40S_45G
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

# Internal-dataset test of the brats23-trained sfusion fold4 checkpoint
# (one of the folds missing from results/sfusion_23/results_internal_fold*.txt).
mimose test \
    --config sfusion_23.yaml \
    --run-suffix "fold4" \
    --dataset-type internal \
    --split-file /homes/ocarpentiero/MiMoSe/src/mimose/data/splits/internal.json \
    --fold 1 \
    --data-dir /work/phd_mimose/internal_raw/sfusion_internal \
    --num-workers 6 \
    --output-path "/work/phd_mimose/results/sfusion_23/results_internal.txt"
