#!/bin/bash
#SBATCH --job-name=FLOPS_PARETOREG_FINAL_L40S
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/flops_paretoreg_final_l40s.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/flops_paretoreg_final_l40s.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_L40S_45G
#SBATCH --account=phd_mimose

# Cost (GMACs, latency, peak memory) of the three TESTED paretoreg23 fold-1
# subnetworks, measured on the FINAL checkpoint (the one the tests used) and on
# an L40S like the baselines' flops csvs. Replaces the epoch-721 measurements of
# flops_mfi_vs_paretoreg_l40s.sh for the Pareto plots. UNET-MFI is re-timed in
# the same job as a check that this node times like the earlier ones.
set -euo pipefail
cd /homes/ocarpentiero/MiMoSe
source .venv/bin/activate
OUT=/work/phd_mimose/results/flops_paretoreg_final_l40s
mkdir -p $OUT
nvidia-smi --query-gpu=name --format=csv,noheader
CKPT=/work/phd_mimose/runs/paretoreg23_fold1/checkpoints/final_weights_only.safetensors
mimose flops --config unetmfi_23.yaml --device cuda --warmup 3 --output-path $OUT/unetmfi.txt
for B in 1431336 11435747 91366608; do
  mimose flops --config paretoreg_23.yaml --device cuda --warmup 3 --checkpoint $CKPT --budget $B --output-path $OUT/paretoreg_b$B.txt
done
