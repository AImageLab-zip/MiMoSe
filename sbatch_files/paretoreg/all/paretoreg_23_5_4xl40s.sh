#!/bin/bash
#SBATCH --job-name=PARETOREG23_FOLD_5
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH -e /homes/ocarpentiero/slurm/outerr/paretoreg_23_5_4xl40s.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/paretoreg_23_5_4xl40s.out
#SBATCH --gres=gpu:4
#SBATCH --account=grana_neuro
#SBATCH --constraint=gpu_L40S_45G

cd /homes/ocarpentiero/MiMoSe
source /homes/ocarpentiero/MiMoSe/.venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache
# Alternative-hardware segments: each 24 h segment is queued twice (2x RTX Pro 6000
# and 4x L40S, same --comment tag, e.g. seg2); the twin that starts first cancels the
# other. The next segment's twins depend afterany on both, so a fold never runs twice.
SEG=$(squeue -h -j "$SLURM_JOB_ID" -o %k)
# Race guard: if both twins started in the same scheduling cycle, keep only the lower
# job id; this one exits before touching the run directory.
for j in $(squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -t RUNNING -o "%i %k" | awk -v s="$SEG" '$2 == s {print $1}'); do
    if [ "$j" -lt "$SLURM_JOB_ID" ]; then echo "both twins started; $j already runs segment $SEG -- exiting"; exit 0; fi
done
for j in $(squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -t PENDING -o "%i %k" | awk -v s="$SEG" '$2 == s {print $1}'); do
    scancel "$j" && echo "segment $SEG started here; cancelled twin $j"
done
# channels_last_3d weights: bf16 Conv3d otherwise bypasses cuDNN (vol2col fallback).
# Same maths, ~1.75x faster training steps and ~40% less GPU memory (measured on
# paretoscratch b1p4m, 2x RTX Pro 6000: ~1.0 s -> ~0.58 s per step).
# 4x L40S variant: batch_size 3 is per GPU, so the global batch is 12 instead of
# fold 1's 6 (2 GPUs). Learning rate scaled linearly: 0.0003 * 12/6 = 0.0006.
# Half as many optimizer steps per epoch as folds 1-3 -- a different recipe.
export MIMOSE_CHANNELS_LAST=1
# torch.compile, max-autotune without CUDA graphs: on the gated supernet step (2 GPUs,
# 3 budgets + distillation) 1504 -> 1096 ms/step (-27%) on top of channels_last, with
# zero recompiles across budgets; ~7-10 min of compilation at the start of each job.
export MIMOSE_COMPILE_MODE=max-autotune-no-cudagraphs

mimose train --config paretoreg_23.yaml --fold 5 --run-suffix fold5 --seed 0 --try-resume --distributed --validation-every 100 --compile --lr 0.0006 || exit 1

# NOTE: this evaluates the supernet at its DEFAULT budget only, i.e. the widest
# subnetwork (~91.4M params). It is one operating point, not the Pareto family.
# The dense budget sweep over the whole range is Phase 9.
mimose test --config paretoreg_23.yaml --fold 5 --run-suffix fold5 --precision bf16

# Training completed (we reached the test), so cancel the remaining queued
# resume submissions of this same job from the paretoreg submit loop's singleton chain.
scancel --state=PENDING --name="$SLURM_JOB_NAME"
