#!/bin/bash
#SBATCH --job-name=PARETOREG23_FOLD_5
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/paretoreg_23_5.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/paretoreg_23_5.out
#SBATCH --gres=gpu:2
#SBATCH --account=grana_neuro
#SBATCH --constraint=gpu_RTXPro6000B_96G

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

mimose train --config paretoreg_23.yaml --fold 5 --run-suffix fold5 --seed 0 --try-resume --distributed || exit 1

# NOTE: this evaluates the supernet at its DEFAULT budget only, i.e. the widest
# subnetwork (~91.4M params). It is one operating point, not the Pareto family.
# The dense budget sweep over the whole range is Phase 9.
mimose test --config paretoreg_23.yaml --fold 5 --run-suffix fold5 --precision bf16

# Training completed (we reached the test), so cancel the remaining queued
# resume submissions of this same job from the paretoreg submit loop's singleton chain.
scancel --state=PENDING --name="$SLURM_JOB_NAME"
