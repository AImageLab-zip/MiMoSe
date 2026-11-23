#!/bin/bash
#SBATCH --job-name=ALORAMOE23_FOLD_3
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/aloramoe_23_3.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/aloramoe_23_3.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

mimose train --config aloramoe_23.yaml --fold 3 --run-suffix fold3 --seed 0 --try-resume
mimose test --config aloramoe_23.yaml --fold 3 --run-suffix fold3 --precision fp32

# Training completed (we reached the test), so cancel the remaining queued
# resume submissions of this same job from allsbatcher23.sh's singleton chain.
scancel --state=PENDING --name="$SLURM_JOB_NAME"
