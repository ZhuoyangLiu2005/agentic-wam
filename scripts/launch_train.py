#!/usr/bin/env python3
"""Launch `scripts/train.py` under accelerate + DeepSpeed, on one node or many.

    python scripts/launch_train.py --task <task> [--nproc 8] [--zero 1] [--dry-run] [hydra overrides ...]

The `train_*.sh` entry points call this after their preflight; it can also be used directly.

* Runs from the repository root (the accelerate configs name their DeepSpeed JSON relative to it) and
  puts `src/` on PYTHONPATH, so elastic workers import `agenticwam` however the launcher forwards env.
* Output goes to `./runs/<task>/<RUN_ID>`, or `./runs/<task>_<dataset>/<RUN_ID>` when a `data.dataset_root=<dir>` (the
  post-train configs) or `data=<name>` override picks the dataset; wandb is named the same way. An `output_dir=` / `wandb.name=` override wins.
* Multi-node: set NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT on every node. All ranks then form one
  process group, and node 0 publishes a single RUN_ID through a TCPStore on MASTER_PORT+11 so every
  node writes to the same output dir. Set RUN_ID yourself to skip the sync (e.g. to resume a run).
* Single node: MASTER_PORT, when set, is the rendezvous port, so two runs can share one machine.
"""
from __future__ import annotations

import argparse
import datetime
import os
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCELERATE_CONFIGS = {
    1: "scripts/accelerate_configs/accelerate_zero1_ds.yaml",
    2: "scripts/accelerate_configs/accelerate_zero2_ds.yaml",
}


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, str(default))
    if not v.isdigit():
        sys.exit(f"launch_train: {name}={v!r} must be a non-negative integer")
    return int(v)


def _sync_run_id(host: str, port: int, rank: int, world: int, task: str, timeout_s: int) -> str:
    """Node 0 picks the RUN_ID; every node reads it back, so all write to the same output dir."""
    from datetime import timedelta

    import torch.distributed as dist

    store = dist.TCPStore(host_name=host, port=port, world_size=world, is_master=(rank == 0),
                          timeout=timedelta(seconds=timeout_s))
    key = f"run_id::{task}"
    if rank == 0:
        store.set(key, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    return store.get(key).decode("utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="a task config in configs/{robotwin,real_pre_train,real_post_train}/task/")
    ap.add_argument("--nproc", type=int, default=8, help="GPUs per node")
    ap.add_argument("--zero", type=int, choices=sorted(ACCELERATE_CONFIGS), default=1, help="DeepSpeed ZeRO stage")
    ap.add_argument("--dry-run", action="store_true", help="print the command, do not launch")
    a, overrides = ap.parse_known_args()
    task = a.task[:-5] if a.task.endswith(".yaml") else a.task
    found = [p for p in (ROOT / "configs").glob(f"*/task/{task}.yaml")]
    if not found:
        known = sorted(p.stem for p in (ROOT / "configs").glob("*/task/*.yaml"))
        sys.exit(f"launch_train: no task config {task!r} (known: {', '.join(known)})")
    task_owns_output_dir = re.search(r"^output_dir:", found[0].read_text(), re.M) is not None
    root = next((o.split("=", 1)[1] for o in overrides if o.startswith("data.dataset_root=")), None)
    data = next((o.split("=", 1)[1] for o in overrides if o.startswith("data=")), None)
    tag = Path(root).name if root else data
    run_name = f"{task}_{tag}" if tag else task

    nnodes, node_rank = _env_int("NNODES", 1), _env_int("NODE_RANK", 0)
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = _env_int("MASTER_PORT", 29500)

    run_id = os.environ.get("RUN_ID")
    if not run_id:
        if nnodes <= 1 or a.dry_run:
            run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        else:
            port = _env_int("RUN_ID_SYNC_PORT", master_port + 11)
            run_id = _sync_run_id(master_addr, port, node_rank, nnodes, task, _env_int("RUN_ID_SYNC_TIMEOUT", 180))
            print(f"[run_id_sync] host={master_addr} port={port} run_id={run_id}", flush=True)

    cmd = [sys.executable, "-m", "accelerate.commands.launch", "--config_file", ACCELERATE_CONFIGS[a.zero]]
    if nnodes > 1:
        cmd += ["--num_processes", str(a.nproc * nnodes), "--num_machines", str(nnodes),
                "--machine_rank", str(node_rank), "--main_process_ip", master_addr,
                "--main_process_port", str(master_port), "--deepspeed_multinode_launcher", "standard"]
    else:
        cmd += ["--num_processes", str(a.nproc)]
        if "MASTER_PORT" in os.environ:          # two single-node runs on one machine need distinct ports
            cmd += ["--main_process_port", str(master_port)]
    cmd += ["scripts/train.py", f"task={task}"]
    if not task_owns_output_dir and not any(o.lstrip("+").startswith("output_dir=") for o in overrides):
        cmd.append(f"output_dir=./runs/{run_name}/{run_id}")
    if not any(o.lstrip("+").startswith("wandb.name=") for o in overrides):
        cmd.append(f"wandb.name={run_name}")
    cmd += overrides

    print(f"[launch] task={task} nproc_per_node={a.nproc} nnodes={nnodes} node_rank={node_rank} "
          f"zero={a.zero} run_id={run_id}", flush=True)
    print("[launch] " + shlex.join(cmd), flush=True)
    if a.dry_run:
        return
    os.chdir(ROOT)
    os.environ["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
