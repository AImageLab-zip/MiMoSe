#!/bin/bash
#SBATCH --job-name=FLOPS_MISSING_MODELS_L40S
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/flops_missing_models_l40s.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/flops_missing_models_l40s.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_L40S_45G
#SBATCH --account=phd_mimose

# Cost benchmarks (GMACs, GPU latency, peak GPU memory) for every model of the
# fold-1 Pareto plots that has none, all on one L40S like the existing csvs:
#  1. the three TESTED paretoreg23 subnetworks on the FINAL checkpoint
#     (replaces the epoch-721 numbers of flops_mfi_vs_paretoreg_l40s.sh);
#  2. the four reproduced models without a flops report;
#  3. the 16 nasunif models (so far only their GMACs budget was known).
# UNET-MFI is re-timed first as a check that this node times like earlier ones.
# No `set -e`: one failing model must not stop the others.
cd /homes/ocarpentiero/MiMoSe
source .venv/bin/activate
R=/work/phd_mimose/results
nvidia-smi --query-gpu=name --format=csv,noheader

OUT=$R/flops_paretoreg_final_l40s; mkdir -p $OUT
CKPT=/work/phd_mimose/runs/paretoreg23_fold1/checkpoints/final_weights_only.safetensors
mimose flops --config unetmfi_23.yaml --device cuda --warmup 3 --output-path $OUT/unetmfi.txt
for B in 1431336 11435747 91366608; do
  mimose flops --config paretoreg_23.yaml --device cuda --warmup 3 --checkpoint $CKPT --budget $B --output-path $OUT/paretoreg_b$B.txt || echo "FAILED paretoreg $B"
done

for M in m3ae mambavitakd mstkdnet shaspec; do
  mimose flops --config ${M}_23.yaml --device cuda --warmup 3 --output-path $R/${M}_23/flops.txt || echo "FAILED $M"
done

for B in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16; do
  mimose flops --config nasunif_b${B}_23.yaml --device cuda --warmup 3 --output-path $R/nasunif_b${B}_23/flops.txt || echo "FAILED nasunif b$B"
done
echo ALL DONE
