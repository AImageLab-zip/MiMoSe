#!/bin/bash
#SBATCH --job-name=PARETOSCRATCH_B1P4M23_TEST_INTERNAL_F1
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/paretoscratch_b1p4m_23_test_internal_fold1.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/paretoscratch_b1p4m_23_test_internal_fold1.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_L40S_45G
#SBATCH --account=phd_mimose

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache

# Internal-dataset test of the from-scratch retrain of the paretoreg b1p4m widths
# (fold 1). Submitted with --dependency=afterany on every job of its training
# chain, so it starts once the chain is over however it ended; it therefore
# refuses to run unless the training really produced its final weights.
# Job name deliberately differs from the chain's: the chain's last step cancels
# pending jobs that share ITS name. bf16 and simpleunet_internal, like the
# paretoreg and nasunif internal tests.
CKPT=/work/phd_mimose/runs/paretoscratch_b1p4m23_fold1/checkpoints/final_weights_only.safetensors
if [ ! -s "$CKPT" ]; then echo "final weights missing: $CKPT -- training did not finish"; exit 1; fi

mimose test \
    --config paretoscratch_b1p4m_23.yaml \
    --run-suffix "fold1" \
    --dataset-type internal \
    --split-file /homes/ocarpentiero/MiMoSe/src/mimose/data/splits/internal.json \
    --fold 1 \
    --data-dir /work/phd_mimose/internal_raw/simpleunet_internal \
    --precision bf16 \
    --num-workers 6 \
    --output-path "/work/phd_mimose/results/paretoscratch_b1p4m_23/results_internal.txt"
