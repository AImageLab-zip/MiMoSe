#!/bin/bash
#SBATCH --job-name=MISSING_INTERNAL_MIMOSA18
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_RTXPro6000B_96G
#SBATCH --account=phd_mimose
#SBATCH -e /homes/ocarpentiero/slurm/outerr/missing_internal_mimosa18.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/missing_internal_mimosa18.out

# Missing MB-96 internal tests, run 4 at a time on ONE GPU (each needs well under
# a quarter of the 96 GB). Each test logs to its own file next to this job's log.
cd /homes/ocarpentiero/MiMoSe
source .venv/bin/activate
export WANDB_CACHE_DIR=/work/phd_mimose/.wandb-cache
SPLIT=/homes/ocarpentiero/MiMoSe/src/mimose/data/splits/internal.json
LOG=/homes/ocarpentiero/slurm/outerr
run() { # tag config fold data_dir results_dir
  mimose test --config $2 --run-suffix "fold$3" --dataset-type internal --split-file $SPLIT --fold 1 \
    --data-dir $4 --num-workers 6 --output-path "/work/phd_mimose/results/$5/results_internal.txt" \
    > $LOG/missing_internal_mimosa18_$1.log 2>&1 && echo "OK $1" || echo "FAILED $1"
}
run micro mimosa_micro_18.yaml 1 /work/phd_mimose/internal_raw/simpleunet_internal mimosa_micro_18 &
run tiny mimosa_tiny_18.yaml 1 /work/phd_mimose/internal_raw/simpleunet_internal mimosa_tiny_18 &
run small mimosa_small_18.yaml 1 /work/phd_mimose/internal_raw/simpleunet_internal mimosa_small_18 &
run medium mimosa_medium_18.yaml 1 /work/phd_mimose/internal_raw/simpleunet_internal mimosa_medium_18 &
wait
echo ALL DONE
