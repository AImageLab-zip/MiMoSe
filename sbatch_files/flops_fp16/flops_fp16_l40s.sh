#!/bin/bash
#SBATCH --job-name=FLOPS_FP16_L40S
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_L40S_45G
#SBATCH --account=phd_mimose
#SBATCH -o /homes/ocarpentiero/slurm/outerr/flops_fp16_l40s.out
#SBATCH -e /homes/ocarpentiero/slurm/outerr/flops_fp16_l40s.err

# fp16 (torch.autocast) GPU inference cost -- peak VRAM and latency -- for every
# model trained/tested in fp16 (YAML `precision: fp16`), plus the three tested
# RDoll-UNet (paretoreg23) subnetworks from the final checkpoint.
# UNET-MFI is first re-timed in fp32 as a check that this node times like the
# fp32 csvs (4.34 s whole volume). No `set -e`: one failure must not stop the rest.
# Results: /work/phd_mimose/results/flops_fp16_l40s/<model>.{txt,csv}
cd /homes/ocarpentiero/MiMoSe
source .venv/bin/activate
nvidia-smi --query-gpu=name --format=csv,noheader
OUT=/work/phd_mimose/results/flops_fp16_l40s; mkdir -p $OUT
CKPT=/work/phd_mimose/runs/paretoreg23_fold1/checkpoints/final_weights_only.safetensors

mimose flops --config unetmfi_23.yaml --device cuda --warmup 3 --precision fp32 --output-path $OUT/unetmfi_fp32check.txt

for M in a2fseg dcseg ims2trans m2ftrans m3fecon mmformer rfl srmnet unetmfi; do
  mimose flops --config ${M}_23.yaml --device cuda --warmup 3 --precision fp16 --output-path $OUT/$M.txt || echo "FAILED $M"
done

for B in 1431336:1.34M 11435747:11.59M 91366608:36.2M; do
  mimose flops --config paretoreg_23.yaml --device cuda --warmup 3 --precision fp16 \
    --budget ${B%%:*} --checkpoint $CKPT --output-path $OUT/rdoll_${B##*:}.txt || echo "FAILED rdoll ${B##*:}"
done
echo ALL DONE
