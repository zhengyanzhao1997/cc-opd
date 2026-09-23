#!/usr/bin/env python3
"""Local multi-node entry point for CC-OPD training.

This script is invoked once per node by ``run_cc_opd.sh``. On rank 0 it
starts a Ray head, waits for all worker nodes to join, then launches the
training driver ``scripts/train.sh``. On non-zero ranks it joins the Ray
cluster as a worker and blocks until shutdown.

For single-node usage, set ``WORLD_SIZE=1`` and ``RANK=0`` (defaults) and
the script will start a single Ray head and run training on the local GPUs.

Environment variables consulted (set by ``run_cc_opd.sh``):
    RANK         - 0-based node rank (default: 0)
    WORLD_SIZE   - number of nodes (default: 1)
    MASTER_ADDR  - hostname/IP of the rank-0 node (default: hostname)
    GPUS_PER_NODE - GPUs per node (default: 8)
    JOB_NAME     - identifier for this job (required)
    CODE_DIR     - absolute path to the verl fork (the ``code/`` directory)
    LAUNCH_DIR   - absolute path to the ``training/`` directory

Unlike the original cluster entry point, this script does NOT install
packages at runtime; install dependencies once via ``pip install -e CODE_DIR``
before launching.
"""

from __future__ import annotations

import argparse
import importlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


LAUNCH_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = str(LAUNCH_DIR / "scripts" / "train.sh")
RAY_PORT = int(os.environ.get("RAY_PORT", "6379"))


def start_ray_head(num_gpus: int) -> None:
    addr = os.environ.get("MASTER_ADDR", socket.gethostname())
    print(f"[ray] starting head on {addr}:{RAY_PORT} with {num_gpus} GPUs", flush=True)
    cmd = ["ray", "start", "--head", f"--port={RAY_PORT}", f"--num-gpus={num_gpus}"]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[FATAL] ray head start failed (code {result.returncode})", flush=True)
        sys.exit(1)
    time.sleep(5)


def start_ray_worker(rank: int, num_gpus: int) -> None:
    head = os.environ.get("MASTER_ADDR")
    if not head:
        print(f"[FATAL] rank {rank}: MASTER_ADDR not set", flush=True)
        sys.exit(1)
    addr = f"{head}:{RAY_PORT}"
    print(f"[ray] rank {rank}: joining head at {addr}", flush=True)
    deadline = time.time() + 600
    while time.time() < deadline:
        cmd = ["ray", "start", f"--address={addr}", f"--num-gpus={num_gpus}", "--block"]
        result = subprocess.run(cmd)
        if result.returncode == 0:
            return
        print(f"[ray] rank {rank}: worker join exited {result.returncode}, retry in 10s", flush=True)
        time.sleep(10)
    print(f"[FATAL] rank {rank}: timeout joining ray head", flush=True)
    sys.exit(1)


def wait_for_ray_cluster(world_size: int, gpus_per_node: int, timeout_s: int = 600) -> None:
    target = world_size * gpus_per_node
    print(f"[ray-wait] expecting {target} GPUs across {world_size} nodes", flush=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            ray = importlib.import_module("ray")
            if not ray.is_initialized():
                ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False)
            n_gpus = int(ray.cluster_resources().get("GPU", 0))
            n_nodes = len(ray.nodes())
            print(f"[ray-wait] currently {n_gpus} GPUs visible across {n_nodes} nodes", flush=True)
            if n_gpus >= target and n_nodes >= world_size:
                ray.shutdown()
                return
        except Exception as exc:
            print(f"[ray-wait] not ready yet: {exc}", flush=True)
        time.sleep(10)
    print(f"[FATAL] ray cluster did not reach {target} GPUs in {timeout_s}s", flush=True)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local multi-node entry for CC-OPD training")
    parser.add_argument("--world_size", type=int, required=True, help="Number of nodes")
    parser.add_argument("--job_name", type=str, required=True, help="Job identifier")
    parser.add_argument("--gpus_per_node", type=int, default=int(os.environ.get("GPUS_PER_NODE", "8")))
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    print(f"[RANK {rank}/{args.world_size}] entry start | job={args.job_name}", flush=True)
    print(
        f"[RANK {rank}] hostname={socket.gethostname()} "
        f"MASTER_ADDR={os.environ.get('MASTER_ADDR', 'UNSET')}",
        flush=True,
    )

    code_dir = os.environ.get("CODE_DIR")
    if not code_dir or not os.path.isfile(os.path.join(code_dir, "setup.py")):
        print(f"[FATAL] CODE_DIR={code_dir!r} does not contain setup.py", flush=True)
        sys.exit(1)
    os.environ["PYTHONPATH"] = f"{code_dir}:{os.environ.get('PYTHONPATH', '')}"

    if rank == 0:
        start_ray_head(args.gpus_per_node)
        wait_for_ray_cluster(args.world_size, args.gpus_per_node, timeout_s=600)
        env = os.environ.copy()
        env["JOB_NAME"] = args.job_name
        env["WORLD_SIZE"] = str(args.world_size)
        env["GPUS_PER_NODE"] = str(args.gpus_per_node)
        env["TOKENIZERS_PARALLELISM"] = "true"
        env["RAY_ADDRESS"] = "auto"
        env["CODE_DIR"] = code_dir
        env["LAUNCH_DIR"] = str(LAUNCH_DIR)
        result = subprocess.run(["bash", TRAIN_SCRIPT], env=env)
        sys.exit(result.returncode)
    else:
        time.sleep(15)
        start_ray_worker(rank, args.gpus_per_node)


if __name__ == "__main__":
    main()
