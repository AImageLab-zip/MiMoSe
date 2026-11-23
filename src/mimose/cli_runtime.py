import os
import subprocess
import sys

import typer


def distributed_launch_active() -> bool:
    return "LOCAL_RANK" in os.environ or int(os.environ.get("WORLD_SIZE", "1")) > 1


def slurm_topology() -> tuple[int, int, str] | None:
    """Return (nnodes, node_rank, master_addr) when running under srun, else None.

    Multi-node needs one launcher task per node, which only srun can place; a
    plain `sbatch --ntasks=1` body runs on a single node and cannot reach the
    others. SLURM_NODEID is set per task, so each launcher learns its own rank.
    """
    nnodes = os.environ.get("SLURM_NNODES") or os.environ.get("SLURM_JOB_NUM_NODES")
    node_rank = os.environ.get("SLURM_NODEID")
    if nnodes is None or node_rank is None:
        return None

    master_addr = os.environ.get("MASTER_ADDR")
    if not master_addr:
        nodelist = os.environ.get("SLURM_JOB_NODELIST")
        if not nodelist:
            return None
        try:
            hostnames = subprocess.run(
                ["scontrol", "show", "hostnames", nodelist],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.split()
        except (OSError, subprocess.CalledProcessError):
            return None
        if not hostnames:
            return None
        master_addr = hostnames[0]

    return int(nnodes), int(node_rank), master_addr


def infer_nproc_per_node() -> int:
    import torch

    num_devices = torch.cuda.device_count()
    if num_devices <= 0:
        raise typer.BadParameter(
            "unable to infer --nproc-per-node because no CUDA devices are visible",
            param_hint="--nproc-per-node",
        )
    return num_devices


def relaunch_with_torchrun(nproc_per_node: int, nnodes: int | None = None) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    topology = slurm_topology()
    if nnodes is None:
        # Under srun, the allocation itself says how many nodes to span; single
        # node stays the default everywhere else, so existing runs are unchanged.
        nnodes = topology[0] if topology is not None else 1

    # Single node: --standalone binds the rendezvous store to a free local port
    # instead of torchrun's fixed default 29500, so several DDP jobs can share
    # a node without EADDRINUSE.
    rendezvous: list[str] = ["--standalone"]
    if nnodes > 1:
        if topology is None:
            raise typer.BadParameter(
                "multi-node training must be launched with srun (one task per node) "
                "so every node gets a launcher; set MASTER_ADDR and SLURM_NODEID "
                "manually if you are not on Slurm",
                param_hint="--nnodes",
            )
        _, node_rank, master_addr = topology
        master_port = os.environ.get("MASTER_PORT", "29500")
        rendezvous = [
            "--nnodes",
            str(nnodes),
            "--node-rank",
            str(node_rank),
            "--rdzv-backend",
            "c10d",
            "--rdzv-endpoint",
            f"{master_addr}:{master_port}",
            "--rdzv-id",
            os.environ.get("SLURM_JOB_ID", "mimose"),
        ]

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        str(nproc_per_node),
        *rendezvous,
        sys.argv[0],
        *sys.argv[1:],
    ]
    raise typer.Exit(os.spawnvp(os.P_WAIT, sys.executable, command))
