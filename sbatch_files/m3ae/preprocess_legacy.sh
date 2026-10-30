#!/bin/bash
#SBATCH --job-name=PREPROCESS_M3AE_LEGACY
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=1:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/preprocess_m3ae_legacy.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/preprocess_m3ae_legacy.out
#SBATCH --gres=gpu:0
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
# Legacy-normalised copy of the data (see the header of m3ae_18.yaml); one folder serves 18/23/25.
mimose preprocess --config m3ae_18.yaml --dataset-type brats23 --yes
