#!/bin/bash
#SBATCH --job-name=NONET23_FOLD_1
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=56G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/nonet_23_1.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/nonet_23_1.out
#SBATCH --gres=gpu:1
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache
# bf16 Conv3d needs channels_last_3d to hit cuDNN (NCDHW falls back to vol2col).
export MIMOSE_CHANNELS_LAST=1

mimose train --config nonet_23.yaml --fold 1 --run-suffix fold1 --seed 0 --try-resume || exit 1
mimose test --config nonet_23.yaml --fold 1 --run-suffix fold1 --precision bf16

# Training completed (we reached the test): cancel the queued resumes of this singleton chain.
scancel --state=PENDING --name="$SLURM_JOB_NAME"
