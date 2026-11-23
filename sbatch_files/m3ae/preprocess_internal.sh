#!/bin/bash
#SBATCH --job-name=PREPROCESS_M3AE_INTERNAL
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=0:20:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/preprocess_m3ae_internal.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/preprocess_m3ae_internal.out
#SBATCH --gres=gpu:0
#SBATCH --account=phd_mimose
#_SBATCH --constraint=gpu_L40S_48G|gpu_A40_48G|gpu_RTXA5000_24G|gpu_RTX6000_24G

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
# M3AE is trained on irm_min_max data (m3ae-legacy-preprocessed, see m3ae_18.yaml),
# so its internal copy must use the same normalisation; crop/clamp match too.

mimose preprocess --input-dir /work/phd_mimose/internal_raw/internal_cropped --output-dir /work/phd_mimose/internal_raw/m3ae_internal --crop-mode non_empty --crop-min-size 128 128 128 --clamp-mode none --norm-mode irm_min_max --yes
