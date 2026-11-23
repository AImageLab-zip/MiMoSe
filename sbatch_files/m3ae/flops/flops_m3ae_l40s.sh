#!/bin/bash
#SBATCH --job-name=FLOPS_M3AE_L40S
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/flops_m3ae_l40s.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/flops_m3ae_l40s.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_L40S_45G
#SBATCH --account=phd_mimose

# Whole-volume cost (FLOPs, peak memory, GPU latency) of M3AE on the shared 182x218x182 benchmark
# volume, L40S like every other row of the cost table. It failed in flops_missing_models_l40s.sh
# because its learned missing-modality template (240x240x155) is shallower than the benchmark volume;
# mimose flops now sets M3AE.allow_template_padding (zero-pads the template, cost-neutral).
# UNET-MFI is re-timed first as a check that this node times like the earlier ones.
set -euo pipefail
cd /homes/ocarpentiero/MiMoSe
source .venv/bin/activate
R=/work/phd_mimose/results
nvidia-smi --query-gpu=name --format=csv,noheader
mimose flops --config unetmfi_23.yaml --device cuda --warmup 3 --output-path $R/flops_m3ae_check_unetmfi.txt
mimose flops --config m3ae_23.yaml --device cuda --warmup 3 --output-path $R/m3ae_23/flops.txt
