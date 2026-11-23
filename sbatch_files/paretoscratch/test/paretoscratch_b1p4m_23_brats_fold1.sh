#!/bin/bash
#SBATCH --job-name=PARETOSCRATCH_B1P4M23_TEST_BRATS_F1
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=02:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/paretoscratch_b1p4m_23_test_brats_fold1.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/paretoscratch_b1p4m_23_test_brats_fold1.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_RTXPro6000B_96G
#SBATCH --account=phd_mimose

# BraTS25-pre (fold 1) test of the from-scratch b1p4m retrain, split out of the
# training job (that job's own copy of the training was frozen and replaced by a
# channels-last run inside the same allocation, so its test line never runs).
# Same command as paretoscratch_b1p4m_23_1.sh. Refuses to run without final weights.
cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache
CKPT=/work/phd_mimose/runs/paretoscratch_b1p4m23_fold1/checkpoints/final_weights_only.safetensors
if [ ! -s "$CKPT" ]; then echo "final weights missing: $CKPT -- training did not finish"; exit 1; fi
mimose test --config paretoscratch_b1p4m_23.yaml --fold 1 --run-suffix fold1 --precision bf16 --num-workers 8 --data-dir /work/phd_mimose/mimosa_preprocessed
