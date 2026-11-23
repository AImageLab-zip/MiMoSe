#!/bin/bash
#SBATCH --job-name=PARETOSCRATCH_B11P4M23_FOLD_1
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/paretoscratch_b11p4m_23_1.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/paretoscratch_b11p4m_23_1.out
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu_RTXPro6000B_96G
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

# Read the training set from the node's local disk instead of /work.
DATA=/work/phd_mimose/mimosa_preprocessed
if cp -r $DATA /tmp/mimosa_preprocessed && [ "$(ls $DATA | wc -l)" = "$(ls /tmp/mimosa_preprocessed | wc -l)" ]; then DATA=/tmp/mimosa_preprocessed; fi
echo "data dir: $DATA"

mimose train --config paretoscratch_b11p4m_23.yaml --fold 1 --run-suffix fold1 --seed 0 --try-resume --distributed --num-workers 6 --data-dir $DATA || exit 1
mimose test --config paretoscratch_b11p4m_23.yaml --fold 1 --run-suffix fold1 --precision bf16 --num-workers 8 --data-dir $DATA

# Training completed (we reached the test), so cancel the remaining queued
# resume submissions of this same job's singleton chain.
scancel --state=PENDING --name="$SLURM_JOB_NAME"
