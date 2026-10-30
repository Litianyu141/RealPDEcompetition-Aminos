#!/usr/bin/env python3
"""Fixed Re3 current-member training on a multi-GPU host, sharing fresh SBAND2 teachers.

This is training provenance for the 40 SPSR centre coefficients and the T2
fold-conditioned prior. It does not look at private-test data or tune seeds.
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

import band_teachers as bt

# One development host excludes its GPU3 after a 2026-09-24 hardware fault. The ban applies on that host only (by hostname,
# same test as the development queue scripts); every other machine, including the organizer's server, uses every GPU.
BANNED_GPUS = {3} if __import__("socket").gethostname() == "40902" else set()

ROOT = bt.ROOT
SPEC = Path(__file__).parent / "config/re3_members.json"
LOGROOT = Path(__file__).parent / "Logger"
EXTERNAL = {"RLB", "WSW", "MIMOA", "MIMOWSW"}


def nodes(spec: dict) -> list[dict]:
    out = []
    for fold in spec["folds"]:
        for item in spec["nodes"]:
            role = item["role"]
            out.append({"id": fold + "_" + role.lower(), "fold": fold, "role": role,
                        "parents": item["parents"],
                        "source": f"Track1/U-net/VariantRetrainPush/config/{fold}/{role}.yaml"})
    assert len(out) == 24 and len({n["id"] for n in out}) == 24
    return out


def output(work: Path, fold: str, role: str) -> Path:
    return work / "checkpoints" / (fold + "_" + role.lower() + "_f0_terminal_ema.pth")


def parent_output(work: Path, band: Path, fold: str, role: str) -> Path:
    if role in EXTERNAL:
        return bt.output(band, fold + "_" + role.lower())
    return output(work, fold, role)


def config(node: dict, work: Path, band: Path) -> dict:
    c = yaml.safe_load((ROOT / node["source"]).read_text())
    assert c["fold"] == 0 and c["save_terminal_both"] is True and c.get("seed", 0) == 0
    assert c["stages"][0]["sources"][0]["tag"] == "comp_sim"
    real = c["stages"][1]["sources"][0]
    assert real["tag"] == "comp_real" and len(real["keep_files"]) in {53, 54, 55}
    assert "7575_0.h5" not in real["keep_files"]
    orig = [Path(x).name for x in c["distill_ckpt"].split(",")]
    expect = [role + "_f0_terminal_ema.pth" for role in node["parents"]]
    assert orig == expect, (node["id"], orig, expect)
    c["distill_ckpt"] = ",".join(str(parent_output(work, band, node["fold"], role)) for role in node["parents"])
    c["checkpoint"] = str(work / "checkpoints" / (node["id"] + ".pth"))
    c["logger_root"] = str(work / "trainer_logs" / node["id"])
    c["variant"] = "decision_re3_" + node["id"]
    c["desc"] = c["variant"]
    return c


def external_proof(band: Path, fold: str, role: str) -> dict:
    name = fold + "_" + role.lower()
    done = json.loads((band / "logs" / (name + ".complete.json")).read_text())
    path = bt.output(band, name)
    assert done["status"] == "OK" and done["node"] == name and bt.sha(path) == done["output_sha256"]
    return {"path": str(path), "sha256": done["output_sha256"], "config_sha256": done["config_sha256"]}


def prepare(work: Path, band: Path, spec: dict) -> None:
    assert json.loads((band / "plan.json").read_text())["spec_sha256"] == bt.sha(bt.SPEC)
    externals = {fold + "/" + role: external_proof(band, fold, role)
                 for fold in spec["folds"] for role in sorted(EXTERNAL)}
    plan = {"spec_sha256": bt.sha(SPEC), "trainer_sha256": bt.sha(ROOT / spec["trainer"]),
            "band_plan_sha256": bt.sha(band / "plan.json"), "external_teachers": externals, "nodes": {}}
    for n in nodes(spec):
        cfg = config(n, work, band)
        body = yaml.safe_dump(cfg, sort_keys=False)
        plan["nodes"][n["id"]] = {"source": n["source"], "source_sha256": bt.sha(ROOT / n["source"]),
                                    "config_sha256": hashlib.sha256(body.encode()).hexdigest(),
                                    "parents": n["parents"], "output": str(output(work, n["fold"], n["role"]))}
    body = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    p = work / "plan.json"
    if p.exists():
        assert p.read_text() == body, "Re3 member plan drift; choose a new work dir"
    else:
        for d in ("configs", "checkpoints", "logs"):
            (work / d).mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    for n in nodes(spec):
        p = work / "configs" / (n["id"] + ".yaml")
        body = yaml.safe_dump(config(n, work, band), sort_keys=False)
        if p.exists(): assert p.read_text() == body, "config drift " + n["id"]
        else: p.write_text(body)
    print("RE3_MEMBERS_PREPARED", work, len(plan["nodes"]), flush=True)


def run_node(work: Path, band: Path, spec: dict, name: str, gpu: int) -> None:
    assert gpu >= 0 and gpu not in BANNED_GPUS, (gpu, "GPU banned on this host")
    prepare(work, band, spec)
    by = {n["id"]: n for n in nodes(spec)}
    assert name in by
    n = by[name]
    for role in n["parents"]:
        if role in EXTERNAL:
            external_proof(band, n["fold"], role)
        else:
            parent = n["fold"] + "_" + role.lower()
            p = work / "logs" / (parent + ".complete.json")
            done = json.loads(p.read_text())
            assert done["status"] == "OK" and bt.sha(output(work, n["fold"], role)) == done["output_sha256"]
    ck = output(work, n["fold"], n["role"])
    done = work / "logs" / (name + ".complete.json")
    if done.exists():
        assert ck.is_file() and bt.sha(ck) == json.loads(done.read_text())["output_sha256"]
        print("RE3_MEMBER_ALREADY_COMPLETE", name); return
    assert not ck.exists(), (name, "unproved output exists")
    log = work / "logs" / (name + ".train.log")
    assert not log.exists(), "prior attempt exists; inspect before retry"
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True).splitlines()
    used, util = [int(x.strip()) for x in raw[gpu].split(",")]
    assert __import__("os").environ.get("RPDE_PIPELINE_SHARES_GPU") == "1" or (used < 100 and util < 20), (gpu, used, util)
    cmd = [sys.executable, "-u", str(ROOT / spec["trainer"]), "--config",
           str(work / "configs" / (name + ".yaml")), "--gpu", str(gpu)]
    print("RE3_MEMBER_START", name, gpu, datetime.now(timezone.utc).isoformat(), flush=True)
    with log.open("x") as f:
        f.write("COMMAND " + json.dumps(cmd) + "\n"); f.flush()
        rc = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                             env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"))
    lines = log.read_text(errors="replace").splitlines()
    assert rc == 0 and any(x.startswith("TRAIN_V4_OK ") for x in lines), (name, rc, lines[-8:])
    assert ck.is_file() and ck.stat().st_size > 0
    rec = {"status": "OK", "node": name, "gpu": gpu, "utc": datetime.now(timezone.utc).isoformat(),
           "plan_sha256": bt.sha(work / "plan.json"), "config_sha256": bt.sha(work / "configs" / (name + ".yaml")),
           "trainer_sha256": bt.sha(ROOT / spec["trainer"]), "output_sha256": bt.sha(ck), "log_sha256": bt.sha(log)}
    done.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print("RE3_MEMBER_COMPLETE", name, rec["output_sha256"], flush=True)


def run(work: Path, band: Path, spec: dict, gpus: list[int], max_jobs: int, resume: bool = False) -> None:
    assert gpus and len(set(gpus)) == len(gpus) and min(gpus) >= 0 and not set(gpus) & BANNED_GPUS
    assert 1 <= max_jobs <= len(gpus)
    prepare(work, band, spec)
    active: dict[str, tuple[subprocess.Popen, int, object]] = {}
    attempted: set[str] = set()
    evt = work / "logs" / ("controller.resume1.jsonl" if resume else "controller.jsonl")
    if resume:
        assert (work / "logs/controller.jsonl").is_file(), "no original controller to resume"
        for n in nodes(spec):
            done = work / "logs" / (n["id"] + ".complete.json")
            train_log = work / "logs" / (n["id"] + ".train.log")
            if done.exists():
                rec = json.loads(done.read_text())
                assert rec["status"] == "OK" and bt.sha(output(work, n["fold"], n["role"])) == rec["output_sha256"]
            else:
                assert not train_log.exists() and not output(work, n["fold"], n["role"]).exists(), \
                    (n["id"], "incomplete attempt requires investigation, never reroll")
    assert not evt.exists(), "controller already attempted; inspect before resuming"
    with evt.open("x") as f:
        def event(kind: str, **kw):
            d = {"time": time.time(), "kind": kind, **kw}
            f.write(json.dumps(d, sort_keys=True) + "\n"); f.flush()
            print(json.dumps(d, sort_keys=True), flush=True)
        while True:
            for name, (p, gpu, out) in list(active.items()):
                rc = p.poll()
                if rc is None: continue
                out.close(); del active[name]
                good = rc == 0 and (work / "logs" / (name + ".complete.json")).is_file()
                event("exit", node=name, gpu=gpu, rc=rc, complete=good)
                if not good:
                    for q, (proc, g, stream) in active.items():event("remaining", node=q, gpu=g, pid=proc.pid)
                    for proc, g, stream in active.values():proc.wait(); stream.close()
                    raise RuntimeError("Re3 member failed: " + name)
            complete = {n["id"] for n in nodes(spec) if (work / "logs" / (n["id"] + ".complete.json")).is_file()}
            if len(complete) == 24:
                event("all_complete", count=24); return
            ready = [n for n in nodes(spec) if n["id"] not in complete | attempted | set(active)
                     and all(role in EXTERNAL or n["fold"] + "_" + role.lower() in complete for role in n["parents"])]
            mem = next(int(x.split()[1]) for x in Path("/proc/meminfo").read_text().splitlines()
                       if x.startswith("MemAvailable:")) / 1048576
            if len(active) < max_jobs and ready and mem >= 90 + 30 * len(active):
                raw = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                               "--format=csv,noheader,nounits"], text=True).splitlines()
                free = [g for g in gpus if g not in {a[1] for a in active.values()}
                        and all(int(v.strip()) < lim for v, lim in zip(raw[g].split(","), (100, 20)))]
                if free:
                    n = ready[0]; gpu = free[0]; name = n["id"]
                    cmd = [sys.executable, "-u", str(Path(__file__)), "--work", str(work), "--band-work", str(band),
                           "--node", name, "--gpu", str(gpu)]
                    out = (work / "logs" / (name + ".controller.log")).open("x")
                    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
                    active[name] = (proc, gpu, out); attempted.add(name)
                    event("launch", node=name, gpu=gpu, pid=proc.pid, available_gib=round(mem, 1))
                    time.sleep(120)
                    continue
            if not active and not ready:raise RuntimeError("incomplete Re3 DAG with no ready job")
            time.sleep(30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--band-work", required=True)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--node")
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue a fully audited interrupted controller without rerunning nodes")
    ap.add_argument("--gpus", default="0,1,4")
    ap.add_argument("--max-jobs", type=int, default=3)
    a = ap.parse_args()
    work = (ROOT / a.work).resolve(); band = (ROOT / a.band_work).resolve()
    assert work.is_relative_to(LOGROOT) and band.is_relative_to(LOGROOT)
    spec = json.loads(SPEC.read_text())
    assert spec["folds"] == ["re0", "re1", "re2"] and len(spec["nodes"]) == 8
    if a.prepare or a.node: prepare(work, band, spec)
    if a.node:
        assert a.gpu is not None
        run_node(work, band, spec, a.node, a.gpu)
    if a.run:
        run(work, band, spec, [int(x) for x in a.gpus.split(",")], a.max_jobs, a.resume)


if __name__ == "__main__":
    main()
