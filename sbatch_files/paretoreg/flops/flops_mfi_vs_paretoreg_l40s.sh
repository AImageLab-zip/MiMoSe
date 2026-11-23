#!/bin/bash
#SBATCH --job-name=FLOPS_MFI_VS_PARETOREG_L40S
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/flops_mfi_vs_paretoreg_l40s.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/flops_mfi_vs_paretoreg_l40s.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_L40S_45G
#SBATCH --account=phd_mimose
# Same as job 195898 (flops_mfi_vs_paretoreg_ep721), but pinned to L40S so the
# latencies are comparable with the baselines' flops csvs (all timed on L40S).
set -euo pipefail
cd /homes/ocarpentiero/MiMoSe
source .venv/bin/activate
OUT=/work/phd_mimose/results/flops_mfi_vs_paretoreg_ep721_l40s
mkdir -p $OUT
nvidia-smi --query-gpu=name --format=csv,noheader
CKPT=/work/phd_mimose/snapshots/paretoreg23_fold1_last.safetensors
mimose flops --config unetmfi_23.yaml --device cuda --warmup 3 --output-path $OUT/unetmfi.txt
for B in 1430000 11400000 91370000; do
  mimose flops --config paretoreg_23.yaml --device cuda --warmup 3 --checkpoint $CKPT --budget $B --output-path $OUT/paretoreg_b$B.txt
done
