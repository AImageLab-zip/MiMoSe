#!/bin/bash
#SBATCH --job-name=PARETOREG23_TEST_B1.4M_F3_CEIL
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/paretoreg_23_test_b1.4M_fold3.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/paretoreg_23_test_b1.4M_fold3.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_L40S_45G
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

# BraTS23 fold-3 test of the final paretoreg23 supernet at the 1.4M budget
# (1431336 params, the same value the validation log reports), bf16 like the
# 91.4M test that runs at the end of the training job. Own results dir so
# it never collides with results/paretoreg_23/ (the default 91.4M budget).
# Ceiling rule: the plain 1.4M selection is (64,56,40,64,4) = 1.71M, over budget; --budget-ceiling
# evaluates the largest subnetwork that fits, (64,56,40,4,4) = 1.18M.
mimose test \
    --config paretoreg_23.yaml \
    --fold 3 \
    --run-suffix fold3 \
    --budget 1431336 \
    --budget-ceiling \
    --precision bf16 \
    --num-workers 6 \
    --output-path /work/phd_mimose/results/paretoreg_23_b1.4M/results.txt
