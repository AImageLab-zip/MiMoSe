#!/bin/bash
#SBATCH --job-name=PARETOREG23_FOLD_3
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/paretoreg_23_3.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/paretoreg_23_3.out
#SBATCH --gres=gpu:2
#SBATCH --account=grana_neuro
#SBATCH --constraint=gpu_RTXPro6000B_96G

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache
# channels_last_3d weights: bf16 Conv3d otherwise bypasses cuDNN (vol2col fallback).
# Same maths, ~1.75x faster training steps and ~40% less GPU memory (measured on
# paretoscratch b1p4m, 2x RTX Pro 6000: ~1.0 s -> ~0.58 s per step).
export MIMOSE_CHANNELS_LAST=1

mimose train --config paretoreg_23.yaml --fold 3 --run-suffix fold3 --seed 0 --try-resume --distributed || exit 1

# NOTE: this evaluates the supernet at its DEFAULT budget only, i.e. the widest
# subnetwork (~91.4M params). It is one operating point, not the Pareto family.
# The dense budget sweep over the whole range is Phase 9.
mimose test --config paretoreg_23.yaml --fold 3 --run-suffix fold3 --precision bf16

# Training completed (we reached the test), so cancel the remaining queued
# resume submissions of this same job from the paretoreg submit loop's singleton chain.
scancel --state=PENDING --name="$SLURM_JOB_NAME"
