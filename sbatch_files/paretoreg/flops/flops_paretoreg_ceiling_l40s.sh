#!/bin/bash
#SBATCH --job-name=FLOPS_PARETOREG_CEIL_L40S
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/flops_paretoreg_ceiling_l40s.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/flops_paretoreg_ceiling_l40s.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_L40S_45G
#SBATCH --account=phd_mimose

# Cost of the fold-1 paretoreg23 subnetworks as TESTED with the ceiling rule (--budget-ceiling):
#   1.4M  budget -> plain selection fits: 1,337,xxx params (same as before)
#   11.4M budget -> ceiling picks selection budget 11172039: widths (64,128,176,144,164), 11,355,978 params
#                   (the old 11435747 selection was 11.59M, over budget)
#   91.4M budget -> 36.2M (unchanged)
# mimose flops has no --budget-ceiling; passing the selection budget the ceiling found realizes the same net.
# UNET-MFI re-timed in the same job as a node check.
set -euo pipefail
cd /homes/ocarpentiero/MiMoSe
source .venv/bin/activate
OUT=/work/phd_mimose/results/flops_paretoreg_ceiling_l40s
mkdir -p $OUT
nvidia-smi --query-gpu=name --format=csv,noheader
CKPT=/work/phd_mimose/runs/paretoreg23_fold1/checkpoints/final_weights_only.safetensors
mimose flops --config unetmfi_23.yaml --device cuda --warmup 3 --output-path $OUT/unetmfi.txt
for B in 1431336 11172039 91366608; do
  mimose flops --config paretoreg_23.yaml --device cuda --warmup 3 --checkpoint $CKPT --budget $B --output-path $OUT/paretoreg_b$B.txt
done
