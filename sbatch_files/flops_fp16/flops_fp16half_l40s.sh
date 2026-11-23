#!/bin/bash
#SBATCH --job-name=FLOPS_FP16HALF_L40S
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_L40S_45G
#SBATCH --account=phd_mimose
#SBATCH -o /homes/ocarpentiero/slurm/outerr/flops_fp16half_l40s.out
#SBATCH -e /homes/ocarpentiero/slurm/outerr/flops_fp16half_l40s.err

# fp16 GPU inference cost (--half: fp16 weights+inputs in channels_last_3d, under fp16 autocast)
# -- peak VRAM and latency -- for the fp16-trained models plus the three tested
# RDoll-UNet (paretoreg23) subnetworks. Supersedes flops_fp16_l40s.sh, whose
# autocast numbers measured cast overhead (slower and more memory than fp32).
# --half also puts weights/inputs in channels_last_3d: without it fp16 Conv3d
# bypasses cuDNN and is ~3x slower than fp32 (see src/mimose/flops.py).
# MODELS / RDOLL env vars select a subset, so the run can be split over jobs.
# UNET-MFI is first re-timed in fp32 as a node check (4.34 s whole volume).
# 150G host RAM: M2FTrans was OOM-killed at 60G in the autocast run.
# No `set -e`: one failure must not stop the rest.
# Results: /work/phd_mimose/results/flops_fp16half_l40s/<model>_gpu.{txt,csv}
cd /homes/ocarpentiero/MiMoSe
source .venv/bin/activate
nvidia-smi --query-gpu=name --format=csv,noheader
OUT=/work/phd_mimose/results/flops_fp16half_l40s; mkdir -p $OUT
CKPT=/work/phd_mimose/runs/paretoreg23_fold1/checkpoints/final_weights_only.safetensors

mimose flops --config unetmfi_23.yaml --device cuda --warmup 3 --output-path $OUT/unetmfi_fp32check_${SLURM_JOB_ID}.txt

for M in ${MODELS-a2fseg dcseg ims2trans m2ftrans m3fecon mmformer rfl srmnet unetmfi}; do
  mimose flops --config ${M}_23.yaml --device cuda --warmup 3 --half --output-path $OUT/$M.txt || echo "FAILED $M"
done

for B in ${RDOLL-1431336:1.34M 11435747:11.59M 91366608:36.2M}; do
  mimose flops --config paretoreg_23.yaml --device cuda --warmup 3 --half \
    --budget ${B%%:*} --checkpoint $CKPT --output-path $OUT/rdoll_${B##*:}.txt || echo "FAILED rdoll ${B##*:}"
done
echo ALL DONE
