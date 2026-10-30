#!/usr/bin/env python3
"""Run the fixed shared model DAG on a multi-GPU host (GPU3 excluded on that host only).

This is a dependency scheduler for *training*, not a hyperparameter search. A completed
node is accepted only through prepare_models.py's hash-checked completion record.
It never reruns an attempted node or kills another process.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import prepare_models as pm

# One development host excludes its GPU3 after a 2026-09-24 hardware fault. The ban applies on that host only (by hostname,
# same test as the development queue scripts); every other machine, including the organizer's server, uses every GPU.
BANNED_GPUS = {3} if __import__("socket").gethostname() == "40902" else set()


def available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1048576
    raise RuntimeError("no MemAvailable")


def free_gpus(allowed: list[int]) -> list[int]:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True
    )
    states = [[int(v.strip()) for v in row.split(",")] for row in raw.splitlines()]
    return [g for g in allowed if g < len(states) and states[g][0] < 100 and states[g][1] < 20]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--gpus", default="1,4,5,6,7")
    ap.add_argument("--max-jobs", type=int, default=3)
    ap.add_argument("--min-available-gib", type=float, default=80)
    ap.add_argument("--settle-seconds", type=int, default=120)
    ap.add_argument("--poll-seconds", type=int, default=30)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    allowed = [int(x) for x in args.gpus.split(",")]
    assert len(allowed) == len(set(allowed)) and allowed and min(allowed) >= 0 and not set(allowed) & BANNED_GPUS
    assert 1 <= args.max_jobs <= len(allowed)
    assert args.min_available_gib >= 70 and args.settle_seconds >= 120 and args.poll_seconds >= 10
    work = (pm.ROOT / args.work).resolve()
    assert work.is_relative_to(Path(pm.__file__).parent / "Logger")
    spec = json.loads(pm.SPEC.read_text()); pm.validate_graph(spec)
    pm.prepare(work, spec)
    if args.plan_only:
        for node in spec["nodes"]:
            done = (work / "logs" / (node["id"] + ".complete.json")).exists()
            print(node["id"], "COMPLETE" if done else "WAIT", "parents=" + ",".join(node["parents"]))
        return
    pm.check_data()

    active: dict[str, tuple[subprocess.Popen, int, object]] = {}
    attempted: set[str] = set()
    nodes = spec["nodes"]
    controller_log = work / "logs/controller.jsonl"
    assert not controller_log.exists(), "controller already attempted; inspect before resuming"
    with controller_log.open("x") as events:
        def event(kind: str, **fields) -> None:
            item = {"ts": time.time(), "kind": kind, **fields}
            events.write(json.dumps(item, sort_keys=True) + "\n"); events.flush()
            print(json.dumps(item, sort_keys=True), flush=True)

        while True:
            for name, (proc, gpu, out) in list(active.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                out.close()
                del active[name]
                complete = (work / "logs" / (name + ".complete.json")).is_file()
                event("worker_exit", node=name, gpu=gpu, rc=rc, complete=complete)
                if rc != 0 or not complete:
                    event("halt_on_failure", node=name)
                    # Other already-running training is allowed to finish; no new work is launched.
                    for n, (p, g, f) in active.items():
                        event("still_running", node=n, gpu=g, pid=p.pid)
                    for n, (p, g, f) in list(active.items()):
                        event("remaining_exit", node=n, gpu=g, rc=p.wait())
                        f.close()
                    return
            complete = {n["id"] for n in nodes if (work / "logs" / (n["id"] + ".complete.json")).is_file()}
            if len(complete) == len(nodes):
                event("all_models_complete", count=len(complete))
                return
            if len(active) < args.max_jobs and available_gib() >= args.min_available_gib + 30 * len(active):
                occupied = {gpu for _, gpu, _ in active.values()}
                free = [gpu for gpu in free_gpus(allowed) if gpu not in occupied]
                ready = [n for n in nodes if n["id"] not in complete | attempted | set(active)
                         and set(n["parents"]) <= complete]
                if ready and free:
                    n, gpu = ready[0], free[0]
                    name = n["id"]
                    attempted.add(name)
                    cmd = [sys.executable, "-u", str(Path(pm.__file__)), "--work", str(work),
                           "--node", name, "--gpu", str(gpu)]
                    out = (work / "logs" / (name + ".controller.log")).open("x")
                    proc = subprocess.Popen(cmd, cwd=pm.ROOT, stdout=out, stderr=subprocess.STDOUT,
                                            env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"))
                    active[name] = (proc, gpu, out)
                    event("launch", node=name, gpu=gpu, pid=proc.pid, available_gib=round(available_gib(), 1))
                    time.sleep(args.settle_seconds)
                    continue
            if not active and attempted:
                ready = [n for n in nodes if n["id"] not in complete | attempted
                         and set(n["parents"]) <= complete]
                if not ready:
                    raise RuntimeError("incomplete DAG has no ready node; inspect attempted logs")
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
