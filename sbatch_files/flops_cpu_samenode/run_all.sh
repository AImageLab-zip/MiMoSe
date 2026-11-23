#!/bin/bash
#SBATCH --job-name=CPUCOST_SAMENODE
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --constraint=gpu_RTXPro6000B_96G
#SBATCH --account=phd_mimose
#SBATCH -o /homes/ocarpentiero/slurm/outerr/cpucost_samenode.out
#SBATCH -e /homes/ocarpentiero/slurm/outerr/cpucost_samenode.err

# CPU cost benchmark for every model of the cost table, all in ONE job so every
# CPU latency comes from the same node (the cluster refuses --nodelist).
# No GPU is requested; the RTX Pro 6000 constraint only picks the node family
# (aldo/giovanni/giacomo/ringhio).
# Models run strictly one after another (no contention between them), threads
# pinned to the allocated cores. Latency only (--latency-only: no MAC counting,
# no single-patch pass; MACs come from the GPU runs), whole volume with all 4
# modalities present (--full-modality-only), 1 warmup + 3 timed calls.
# Fast models first, slowest (ShaSpec, IM-Fuse) last.
# U-HVED is timed first AND again near the end (uhved_repeat) to check that
# the first model is not penalised by one-off startup costs.
# Results: /work/phd_mimose/results/flops_cpu_samenode/<model>.{txt,csv}
cd /homes/ocarpentiero/MiMoSe
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=""
OUT=/work/phd_mimose/results/flops_cpu_samenode; mkdir -p $OUT
CKPT=/work/phd_mimose/runs/paretoreg23_fold1/checkpoints/final_weights_only.safetensors
echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK"; lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket"
# Same-socket pinning. Slurm cannot guarantee single-socket allocation, so we
# ask for 32 PHYSICAL cores (--hint=nomultithread) and run every model on 16 of
# them that sit on ONE socket: one logical CPU per physical core (no
# hyperthread siblings), threads pinned to those cores. Memory follows by
# first-touch onto that socket's NUMA node. The other allocated cores stay idle,
# shielding the timed process from other jobs. Aborts if no socket has 16.
NTHREADS=16
PIN=$(python - "$NTHREADS" <<'PY'
import os, sys
n = int(sys.argv[1])
topo = lambda c, f: open(f"/sys/devices/system/cpu/cpu{c}/topology/{f}").read().strip()
by_socket = {}
for c in sorted(os.sched_getaffinity(0)):
    by_socket.setdefault(topo(c, "physical_package_id"), {}).setdefault(topo(c, "core_id"), c)
sock, cores = max(by_socket.items(), key=lambda kv: len(kv[1]))
print(f"allocated physical cores per socket: { {s: len(v) for s, v in by_socket.items()} }", file=sys.stderr)
if len(cores) < n:
    sys.exit(f"ABORT: no socket has {n} allocated physical cores")
print(",".join(str(c) for c in sorted(cores.values())[:n]))
PY
) || { echo "ABORT: same-socket pinning not possible"; exit 1; }
export OMP_NUM_THREADS=$NTHREADS MKL_NUM_THREADS=$NTHREADS
export OMP_PROC_BIND=close OMP_PLACES=cores
echo "pinned to logical CPUs: $PIN"
taskset -c "$PIN" python - <<'PY'
import os
cpus = sorted(os.sched_getaffinity(0))  # read BEFORE importing torch: its OpenMP init pins the main thread to one core
import torch
topo = lambda c, f: open(f"/sys/devices/system/cpu/cpu{c}/topology/{f}").read().strip()
cores = {(topo(c, "physical_package_id"), topo(c, "core_id")) for c in cpus}
print(f"step affinity: {cpus} -> {len(cores)} physical cores on sockets {sorted({s for s, _ in cores})}; torch threads {torch.get_num_threads()}")
PY

# name|config|budget|mamba-patch
JOBS=(
  "uhved|uhved_23.yaml||" "srmnet|srmnet_23.yaml||" "ims2trans|ims2trans_23.yaml||"
  "m3fecon|m3fecon_23.yaml||" "mmformer|mmformer_23.yaml||" "mmmvit|mmmvit_23.yaml||"
  "lckd|lckd_23.yaml||" "rfl|rfl_23.yaml||" "mstkdnet|mstkdnet_23.yaml||"
  "rdoll_1.34M|paretoreg_23.yaml|1431336|" "rdoll_11.59M|paretoreg_23.yaml|11435747|"
  "rdoll_36.2M|paretoreg_23.yaml|91366608|"
  "sfusion|sfusion_23.yaml||" "mifpn|mifpn_23.yaml||" "m2ftrans|m2ftrans_23.yaml||"
  "robustseg|robustseg_23.yaml||" "a2fseg|a2fseg_23.yaml||" "mambavitakd|mambavitakd_23.yaml||"
  "inoutfusion|inoutfusion_23.yaml||" "unetmfi|unetmfi_23.yaml||" "rfnet|rfnet_23.yaml||"
  "dcseg|dcseg_23.yaml||" "shaspec|shaspec_23.yaml||"
  "uhved_repeat|uhved_23.yaml||" "imfuse|imfuse_23.yaml||1"
)
for J in "${JOBS[@]}"; do
  IFS='|' read -r M C B P <<< "$J"
  echo "=== $M start $(date +%FT%T)"
  ARGS=(flops --config "$C" --device cpu --warmup 1 --latency-only --full-modality-only --output-path "$OUT/$M.txt")
  [ -n "$B" ] && ARGS+=(--budget "$B" --checkpoint "$CKPT")
  if [ "$P" = 1 ]; then
    taskset -c "$PIN" python scripts/run_flops_cpu_mamba_patch.py "${ARGS[@]}" || echo "FAILED $M"
  else
    taskset -c "$PIN" mimose "${ARGS[@]}" || echo "FAILED $M"
  fi
  echo "=== $M end $(date +%FT%T)"
done
echo ALL DONE
