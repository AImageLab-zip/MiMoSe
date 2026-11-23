#!/bin/bash
#SBATCH --job-name=LOOPCHECK_B11P4M23_NEWLOOP
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH -e /homes/ocarpentiero/slurm/outerr/paretoscratch_b11p4m_23_test_brats_fold1_newloop.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/paretoscratch_b11p4m_23_test_brats_fold1_newloop.out
#SBATCH --account=phd_mimose

# BraTS25-pre fold-1 test of the from-scratch b11p4m retrain (bf16, as the training
# job would have run it). new loop.
cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache
# repo code: new subject-major test loop (validation run)
CKPT=/work/phd_mimose/runs/paretoscratch_b11p4m23_fold1/checkpoints/final_weights_only.safetensors
if [ ! -s "$CKPT" ]; then echo "final weights missing: $CKPT"; exit 1; fi
python -c "import mimose.testing.pipeline as p; print('pipeline:', p.__file__)"
s=$(date +%s)
mimose test --config paretoscratch_b11p4m_23.yaml --fold 1 --run-suffix fold1 --precision bf16 --num-workers 8 --data-dir /work/phd_mimose/mimosa_preprocessed --output-path /work/phd_mimose/results/_loopcheck_b11_new/results.txt
echo "test took $(( $(date +%s) - s )) s"
