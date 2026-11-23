#!/bin/bash
#SBATCH --job-name=MISSING_INTERNAL_MIFPN
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_RTXPro6000B_96G
#SBATCH --account=phd_mimose
#SBATCH -e /homes/ocarpentiero/slurm/outerr/missing_internal_mifpn.err
#SBATCH -o /homes/ocarpentiero/slurm/outerr/missing_internal_mifpn.out

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
    > $LOG/missing_internal_mifpn_$1.log 2>&1 && echo "OK $1" || echo "FAILED $1"
}
run mifpn18_f2 mifpn_18.yaml 2 /work/phd_mimose/internal_raw/mifpn_internal mifpn_18 &
run mifpn18_f4 mifpn_18.yaml 4 /work/phd_mimose/internal_raw/mifpn_internal mifpn_18 &
run mifpn23_f2 mifpn_23.yaml 2 /work/phd_mimose/internal_raw/mifpn_internal mifpn_23 &
run mifpn23_f4 mifpn_23.yaml 4 /work/phd_mimose/internal_raw/mifpn_internal mifpn_23 &
wait
echo ALL DONE
