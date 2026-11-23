#!/bin/bash
#SBATCH --job-name=PARETOREG23_TEST_B11.4M_F1
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/paretoreg_23_test_b11.4M_fold1.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/paretoreg_23_test_b11.4M_fold1.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_L40S_45G
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

# BraTS23 fold-1 test of the final paretoreg23 supernet at the 11.4M budget
# (11435747 params, the same value the validation log reports), bf16 like the
# 91.4M test that runs at the end of the training job. Own results dir so
# it never collides with results/paretoreg_23/ (the default 91.4M budget).
mimose test \
    --config paretoreg_23.yaml \
    --fold 1 \
    --run-suffix fold1 \
    --budget 11435747 \
    --precision bf16 \
    --num-workers 6 \
    --output-path /work/phd_mimose/results/paretoreg_23_b11.4M/results.txt
