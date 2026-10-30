#!/usr/bin/env python3
"""Rebuild the missing two AoA OOF fold graphs for the v156g FTL prior.

Re3 folds are rebuilt by band_teachers.py and re3_members.py.  This fixed
AoA2 companion produces the remaining 8 teacher and 16 current-member weights
from the saved source configs, with terminal EMA and explicit parent hashes.
No best-checkpoint or seed selection occurs here.
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
LOGROOT = Path(__file__).parent / "Logger"
FOLDS = ("aoa0", "aoa1")
TEACHERS = {"RLB": (), "WSW": ("RLB",), "MIMOA": ("RLB",), "MIMOWSW": ("RLB",)}
MEMBERS = {
    "TST5_WSW": ("RLB",), "TST5_MIMOA": ("RLB",), "TST5_MIMOWSW": ("RLB",),
    "T2RA3_H": ("WSW", "MIMOA", "MIMOWSW"),
    "T2RA3_SW": ("WSW", "MIMOA", "MIMOWSW"),
    "T1GEG_H": ("WSW", "MIMOA", "MIMOWSW"),
    "T2PHK_H": ("WSW", "MIMOA", "MIMOWSW", "T2RA3_H", "T2RA3_SW"),
    "T2PTK_SW": ("TST5_WSW", "TST5_MIMOA", "TST5_MIMOWSW", "T2RA3_H", "T2RA3_SW"),
}
ALL = {**TEACHERS, **MEMBERS}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nodes() -> list[dict]:
    out = []
    for fold in FOLDS:
        for role, parents in ALL.items():
            source = (f"Track1/U-net/VariantFiveFoldEvaluation/config/{fold}/outer/LIVE/{role}.yaml"
                      if role in TEACHERS else f"Track1/U-net/VariantRetrainPush/config/{fold}/{role}.yaml")
            out.append({"id": fold + "_" + role.lower(), "fold": fold, "role": role,
                        "parents": [fold + "_" + p.lower() for p in parents], "source": source,
                        "trainer": ("Track1/U-net/train_v4.py" if role in TEACHERS else
                                    "Track1/U-net/VariantRetrainPush/train_objective.py")})
    assert len(out) == 24 and len({n["id"] for n in out}) == 24
    return out


def ckpt(work: Path, node: dict | str) -> Path:
    name = node if isinstance(node, str) else node["id"]
    return work / "checkpoints" / (name + "_f0_terminal_ema.pth")


def config(node: dict, work: Path) -> dict:
    c = yaml.safe_load((ROOT / node["source"]).read_text())
    assert c["fold"] == 0 and c["save_terminal_both"] is True and c.get("seed", 0) == 0
    assert c["stages"][0]["sources"][0]["tag"] == "comp_sim"
    real = c["stages"][1]["sources"][0]
    assert real["tag"] == "comp_real" and len(real["keep_files"]) == (31 if node["fold"] == "aoa0" else 50)
    legal = set(json.loads((ROOT / "common/local_benchmark/splits_re3_aoa2_v1.json").read_text())["real_files"])
    assert set(real["keep_files"]) <= legal and len(set(real["keep_files"])) == len(real["keep_files"])
    orig = [Path(s).name for s in str(c.get("distill_ckpt", "")).split(",") if s]
    # Case is significant in the original recipe: WSW and T2RA3_H are names,
    # while every run-scoped output path has a lower-case node ID.
    expect = [p + "_f0_terminal_ema.pth" for p in ALL[node["role"]]]
    assert orig == expect, (node["id"], orig, expect)
    if node["parents"]:
        c["distill_ckpt"] = ",".join(str(ckpt(work, p)) for p in node["parents"])
    c["checkpoint"] = str(work / "checkpoints" / (node["id"] + ".pth"))
    c["logger_root"] = str(work / "trainer_logs" / node["id"])
    c["variant"] = "decision_aoa_" + node["id"]
    c["desc"] = c["variant"]
    return c


def prepare(work: Path) -> tuple[list[dict], dict]:
    ns = nodes()
    plan = {"split_sha256": sha(ROOT / "common/local_benchmark/splits_re3_aoa2_v1.json"),
            "node_recipes": {}}
    for n in ns:
        body = yaml.safe_dump(config(n, work), sort_keys=False)
        plan["node_recipes"][n["id"]] = {"source": n["source"], "source_sha256": sha(ROOT / n["source"]),
                                         "config_sha256": hashlib.sha256(body.encode()).hexdigest(),
                                         "trainer_sha256": sha(ROOT / n["trainer"]), "parents": n["parents"],
                                         "output": str(ckpt(work, n))}
    body = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    p = work / "plan.json"
    if p.exists(): assert p.read_text() == body, "AoA plan drift"
    else:
        for d in ("configs", "checkpoints", "logs"):(work / d).mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    for n in ns:
        p = work / "configs" / (n["id"] + ".yaml")
        body = yaml.safe_dump(config(n, work), sort_keys=False)
        if p.exists(): assert p.read_text() == body, n["id"]
        else: p.write_text(body)
    return ns, plan


def run_node(work: Path, name: str, gpu: int) -> None:
    assert gpu >= 0 and gpu not in BANNED_GPUS, (gpu, "GPU banned on this host")
    ns, plan = prepare(work); n = {x["id"]: x for x in ns}[name]
    for parent in n["parents"]:
        proof = json.loads((work / "logs" / (parent + ".complete.json")).read_text())
        assert proof["status"] == "OK" and proof["output_sha256"] == sha(ckpt(work, parent)), parent
    done = work / "logs" / (name + ".complete.json")
    out = ckpt(work, n)
    if done.exists():
        assert out.is_file() and json.loads(done.read_text())["output_sha256"] == sha(out)
        print("AOA_NODE_ALREADY_COMPLETE", name); return
    assert not out.exists() and not done.exists(), name
    log = work / "logs" / (name + ".train.log")
    assert not log.exists(), name
    gpu_state = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                         "--format=csv,noheader,nounits"], text=True).splitlines()
    used, util = [int(v.strip()) for v in gpu_state[gpu].split(",")]
    assert __import__("os").environ.get("RPDE_PIPELINE_SHARES_GPU") == "1" or (used < 100 and util < 20), (gpu, used, util)
    cmd = [sys.executable, "-u", str(ROOT / n["trainer"]), "--config",
           str(work / "configs" / (name + ".yaml")), "--gpu", str(gpu)]
    with log.open("x") as f:
        f.write("COMMAND " + json.dumps(cmd) + "\n"); f.flush()
        rc = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                             env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"))
    lines = log.read_text(errors="replace").splitlines()
    assert rc == 0 and any(x.startswith("TRAIN_V4_OK ") for x in lines), (name, rc, lines[-8:])
    assert out.is_file() and out.stat().st_size > 0, name
    rec = {"status": "OK", "node": name, "gpu": gpu, "utc": datetime.now(timezone.utc).isoformat(),
           "plan_sha256": sha(work / "plan.json"), "config_sha256": sha(work / "configs" / (name + ".yaml")),
           "trainer_sha256": sha(ROOT / n["trainer"]), "output_sha256": sha(out), "log_sha256": sha(log)}
    done.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print("AOA_NODE_COMPLETE", name, rec["output_sha256"], flush=True)


def run(work: Path, gpus: list[int], max_jobs: int) -> None:
    assert gpus and len(set(gpus)) == len(gpus) and min(gpus) >= 0 and not set(gpus) & BANNED_GPUS
    assert 1 <= max_jobs <= len(gpus)
    ns, _ = prepare(work)
    evt = work / "logs/controller.jsonl"
    assert not evt.exists(), "AoA controller already attempted"
    active = {}; attempted = set()
    with evt.open("x") as f:
        def event(kind, **kw):
            d = {"time": time.time(), "kind": kind, **kw}
            f.write(json.dumps(d, sort_keys=True) + "\n"); f.flush()
            print(json.dumps(d, sort_keys=True), flush=True)
        while True:
            for name, (proc, gpu, out) in list(active.items()):
                rc = proc.poll()
                if rc is None: continue
                out.close(); del active[name]
                good = rc == 0 and (work / "logs" / (name + ".complete.json")).is_file()
                event("exit", node=name, gpu=gpu, rc=rc, complete=good)
                if not good:
                    for p, g, stream in active.values():p.wait(); stream.close()
                    raise RuntimeError("AoA node failed: " + name)
            complete = {n["id"] for n in ns if (work / "logs" / (n["id"] + ".complete.json")).is_file()}
            if len(complete) == len(ns):event("all_complete", count=len(ns)); return
            ready = [n for n in ns if n["id"] not in complete | attempted | set(active)
                     and set(n["parents"]) <= complete]
            mem = next(int(x.split()[1]) for x in Path("/proc/meminfo").read_text().splitlines()
                       if x.startswith("MemAvailable:")) / 1048576
            if len(active) < max_jobs and ready and mem >= 90 + 30 * len(active):
                rows = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                                "--format=csv,noheader,nounits"], text=True).splitlines()
                free = [g for g in gpus if g not in {v[1] for v in active.values()}
                        and all(int(s.strip()) < lim for s, lim in zip(rows[g].split(","), (100, 20)))]
                if free:
                    n, gpu = ready[0], free[0]
                    name = n["id"]
                    cmd = [sys.executable, "-u", str(Path(__file__)), "--work", str(work), "--node", name, "--gpu", str(gpu)]
                    out = (work / "logs" / (name + ".controller.log")).open("x")
                    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
                    active[name] = (proc, gpu, out); attempted.add(name)
                    event("launch", node=name, gpu=gpu, pid=proc.pid, available_gib=round(mem, 1))
                    time.sleep(120); continue
            if not active and not ready:raise RuntimeError("AoA graph stalled")
            time.sleep(30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--node")
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--gpus", default="7")
    ap.add_argument("--max-jobs", type=int, default=1)
    a = ap.parse_args()
    work = (ROOT / a.work).resolve()
    assert work.is_relative_to(LOGROOT)
    if a.prepare:prepare(work)
    if a.node:
        assert a.gpu is not None
        run_node(work, a.node, a.gpu)
    if a.run:run(work, [int(v) for v in a.gpus.split(",")], a.max_jobs)


if __name__ == "__main__":
    main()
