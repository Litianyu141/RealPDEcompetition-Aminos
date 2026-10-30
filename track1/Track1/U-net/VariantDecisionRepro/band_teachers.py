#!/usr/bin/env python3
"""Isolated Re3 OOF teacher training for v185's SBAND2 head.

The original SBAND2 ship cache pooled out-of-fold predictions of WSW, MIMOWSW,
MIMOA and RLB on each of the three fixed Re folds. This script rebuilds those
12 weights with the same saved source configs, fixed terminal EMA, and explicit
parent hashes. It does not train the head or certify the final card by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import yaml

# One development host excludes its GPU3 after a 2026-09-24 hardware fault. The ban applies on that host only (by hostname,
# same test as the development queue scripts); every other machine, including the organizer's server, uses every GPU.
BANNED_GPUS = {3} if __import__("socket").gethostname() == "40902" else set()

ROOT = Path(__file__).resolve().parents[3]
SPEC = Path(__file__).parent / "config/band_teachers.json"
LOGROOT = Path(__file__).parent / "Logger"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def output(work: Path, name: str) -> Path:
    return work / "checkpoints" / (name + "_f0_terminal_ema.pth")


def jobs(spec: dict) -> list[dict]:
    out = []
    for fold in spec["folds"]:
        for role in spec["roles"]:
            name = fold + "_" + role.lower()
            src = spec["source_template"].format(fold=fold, role=role)
            out.append({"name": name, "fold": fold, "role": role, "source": src,
                        "parents": [] if role == "RLB" else [fold + "_rlb"]})
    assert len(out) == 12
    return out


def config(job: dict, work: Path) -> dict:
    c = yaml.safe_load((ROOT / job["source"]).read_text())
    assert c["fold"] == 0 and c["save_terminal_both"] is True and c["seed"] == 0
    assert c["stages"][0]["sources"][0]["tag"] == "comp_sim"
    real = c["stages"][1]["sources"][0]
    assert real["tag"] == "comp_real" and len(real["keep_files"]) in {53, 54, 55}
    assert "7575_0.h5" not in real["keep_files"]
    if job["role"] == "RLB":
        assert not c.get("distill_ckpt")
    else:
        assert Path(c["distill_ckpt"]).name == "RLB_f0_terminal_ema.pth"
        c["distill_ckpt"] = str(output(work, job["parents"][0]))
    c["checkpoint"] = str(work / "checkpoints" / (job["name"] + ".pth"))
    c["logger_root"] = str(work / "trainer_logs" / job["name"])
    c["variant"] = "decision_band_" + job["name"]
    c["desc"] = c["variant"]
    return c


def prepare(work: Path, spec: dict) -> None:
    plan = {"spec_sha256": sha(SPEC), "trainer_sha256": sha(ROOT / spec["trainer"]), "jobs": {}}
    for j in jobs(spec):
        c = config(j, work)
        body = yaml.safe_dump(c, sort_keys=False)
        plan["jobs"][j["name"]] = {"source": j["source"], "source_sha256": sha(ROOT / j["source"]),
                                   "config_sha256": hashlib.sha256(body.encode()).hexdigest(),
                                   "parents": j["parents"], "output": str(output(work, j["name"]))}
    plan_text = json.dumps(plan, sort_keys=True, indent=2) + "\n"
    p = work / "plan.json"
    if p.exists():
        assert p.read_text() == plan_text, "band teacher plan drift; use a new work directory"
    else:
        for d in ("configs", "checkpoints", "logs"):
            (work / d).mkdir(parents=True, exist_ok=True)
        p.write_text(plan_text)
    for j in jobs(spec):
        p = work / "configs" / (j["name"] + ".yaml")
        text = yaml.safe_dump(config(j, work), sort_keys=False)
        if p.exists():
            assert p.read_text() == text, "config drift " + j["name"]
        else:
            p.write_text(text)
    print("BAND_TEACHERS_PREPARED", work, len(plan["jobs"]), flush=True)


def check_data(spec: dict) -> None:
    data = ROOT / "data/RealPDE_competition_dataset/gdrive_raw/extracted/train_real/train_real"
    all_names = {p.name for p in data.glob("*.h5")}
    assert len(all_names) == 82 and "7575_0.h5" in all_names
    for j in jobs(spec):
        c = config(j, LOGROOT / "unused_data_check")
        files = c["stages"][1]["sources"][0]["keep_files"]
        assert set(files) <= all_names and len(set(files)) == len(files), j["name"]
        assert set(files) <= (all_names - {"7575_0.h5"}), j["name"]
    print("BAND_TEACHER_DATA_OK", len(all_names) - 1, flush=True)


def node(work: Path, spec: dict, name: str, gpu: int) -> None:
    assert gpu >= 0 and gpu not in BANNED_GPUS, (gpu, "GPU banned on this host")
    prepare(work, spec)
    by = {j["name"]: j for j in jobs(spec)}
    j = by[name]
    for parent in j["parents"]:
        p = work / "logs" / (parent + ".complete.json")
        assert p.is_file() and sha(output(work, parent)) == json.loads(p.read_text())["output_sha256"]
    complete = work / "logs" / (name + ".complete.json")
    ck = output(work, name)
    if complete.exists():
        assert ck.is_file() and sha(ck) == json.loads(complete.read_text())["output_sha256"]
        print("BAND_TEACHER_ALREADY_COMPLETE", name, flush=True); return
    assert not ck.exists(), (name, "unproved output exists")
    state = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                     "--format=csv,noheader,nounits"], text=True).splitlines()
    used, util = [int(x.strip()) for x in state[gpu].split(",")]
    assert __import__("os").environ.get("RPDE_PIPELINE_SHARES_GPU") == "1" or (used < 100 and util < 20), (gpu, used, util)
    log = work / "logs" / (name + ".train.log")
    assert not log.exists(), (name, "previous attempt exists; inspect, do not rerun")
    cmd = [sys.executable, "-u", str(ROOT / spec["trainer"]), "--config",
           str(work / "configs" / (name + ".yaml")), "--gpu", str(gpu)]
    print("BAND_TEACHER_START", name, gpu, datetime.now(timezone.utc).isoformat(), flush=True)
    with log.open("x") as f:
        f.write("COMMAND " + json.dumps(cmd) + "\n"); f.flush()
        rc = subprocess.call(cmd, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"),
                             stdout=f, stderr=subprocess.STDOUT)
    lines = log.read_text(errors="replace").splitlines()
    assert rc == 0 and any(x.startswith("TRAIN_V4_OK ") for x in lines), (name, rc, lines[-8:])
    assert ck.is_file() and ck.stat().st_size > 0, (name, ck)
    record = {"node": name, "status": "OK", "gpu": gpu, "utc": datetime.now(timezone.utc).isoformat(),
              "plan_sha256": sha(work / "plan.json"), "config_sha256": sha(work / "configs" / (name + ".yaml")),
              "source_sha256": sha(ROOT / j["source"]), "trainer_sha256": sha(ROOT / spec["trainer"]),
              "output": str(ck), "output_sha256": sha(ck), "log_sha256": sha(log)}
    complete.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print("BAND_TEACHER_COMPLETE", name, record["output_sha256"], flush=True)


def run(work: Path, spec: dict, gpus: list[int], max_jobs: int) -> None:
    assert gpus and len(set(gpus)) == len(gpus) and min(gpus) >= 0 and not set(gpus) & BANNED_GPUS
    assert 1 <= max_jobs <= len(gpus)
    prepare(work, spec); check_data(spec)
    active: dict[str, tuple[subprocess.Popen, int, object]] = {}
    attempted: set[str] = set()
    events = work / "logs" / "controller.jsonl"
    assert not events.exists(), "existing controller; inspect before resume"
    with events.open("x") as evt:
        def record(kind: str, **kw) -> None:
            d = {"time": time.time(), "kind": kind, **kw}
            evt.write(json.dumps(d, sort_keys=True) + "\n"); evt.flush()
            print(json.dumps(d, sort_keys=True), flush=True)
        while True:
            for n, (p, gpu, out) in list(active.items()):
                rc = p.poll()
                if rc is None: continue
                out.close(); del active[n]
                ok = rc == 0 and (work / "logs" / (n + ".complete.json")).is_file()
                record("exit", node=n, gpu=gpu, rc=rc, complete=ok)
                if not ok:
                    for q, (proc, g, f) in active.items():
                        record("remaining", node=q, gpu=g, pid=proc.pid)
                    for proc, g, f in active.values():
                        proc.wait(); f.close()
                    raise RuntimeError("band teacher failed: " + n)
            complete = {j["name"] for j in jobs(spec) if (work / "logs" / (j["name"] + ".complete.json")).is_file()}
            if len(complete) == 12:
                record("all_complete", count=12); return
            ready = [j for j in jobs(spec) if j["name"] not in complete | attempted | set(active)
                     and set(j["parents"]) <= complete]
            mem = next(int(x.split()[1]) for x in Path("/proc/meminfo").read_text().splitlines()
                       if x.startswith("MemAvailable:")) / 1048576
            if len(active) < max_jobs and ready and mem >= 90 + 30 * len(active):
                raw = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                               "--format=csv,noheader,nounits"], text=True).splitlines()
                free = [g for g in gpus if g not in {a[1] for a in active.values()}
                        and all(int(v.strip()) < lim for v, lim in zip(raw[g].split(","), (100, 20)))]
                if free:
                    j = ready[0]; gpu = free[0]; name = j["name"]
                    cmd = [sys.executable, "-u", str(Path(__file__)), "--work", str(work), "--node", name, "--gpu", str(gpu)]
                    out = (work / "logs" / (name + ".controller.log")).open("x")
                    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
                    active[name] = (proc, gpu, out); attempted.add(name)
                    record("launch", node=name, gpu=gpu, pid=proc.pid, available_gib=round(mem, 1))
                    time.sleep(120)
                    continue
            if not active and not ready:
                raise RuntimeError("incomplete band DAG with no ready job")
            time.sleep(30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--check-data", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--node")
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--gpus", default="6,7")
    ap.add_argument("--max-jobs", type=int, default=2)
    a = ap.parse_args()
    work = (ROOT / a.work).resolve()
    assert work.is_relative_to(LOGROOT)
    spec = json.loads(SPEC.read_text())
    assert spec["folds"] == ["re0", "re1", "re2"] and spec["roles"] == ["RLB", "WSW", "MIMOA", "MIMOWSW"]
    if a.check_data:check_data(spec)
    if a.prepare or a.plan_only or a.node:prepare(work, spec)
    if a.plan_only:
        for j in jobs(spec):
            print(j["name"], "parents", j["parents"], "source", j["source"])
    if a.node:
        assert a.gpu is not None
        node(work, spec, a.node, a.gpu)
    if a.run:
        run(work, spec, [int(x) for x in a.gpus.split(",")], a.max_jobs)


if __name__ == "__main__":
    main()
