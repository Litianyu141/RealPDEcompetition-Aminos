#!/usr/bin/env python3
"""Port the fixed v185/v156g neural-member DAG to Final-kit /data and /output.

This is the shared model subgraph. Its 12 nodes produce exactly the three
distinct member roles in the submitted cards, but do not produce the learned
heads, bands, FTL tables or final ZIP. `train()` must call all producers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import yaml

import prepare_models as pm

ROOT = pm.ROOT
SPEC = pm.SPEC
RUNNER = Path(__file__).parent / "final_trainer.py"


def hash_file(p: Path) -> str:
    return pm.sha(p)


def config_for_container(node: dict, by: dict[str, dict], data: Path, output: Path) -> dict:
    work = output / "work/models"
    cfg, _ = pm.config_for(node, by, work)
    cfg["stats"] = str(output / "mean_std_real.pt")
    cfg["val_glob"] = str(output / "work/data/val9/test_real/*.h5")
    for stage in cfg["stages"]:
        assert len(stage["sources"]) == 1
        src = stage["sources"][0]
        if stage["name"] == "sim-pretrain":
            assert src["tag"] == "comp_sim"
            src["glob"] = str(data / "train_sim/*.h5")
            src["stats"] = str(output / "work/data/mean_std_comp_sim.pt")
        else:
            assert stage["name"] == "real-finetune" and src["tag"] == "comp_real"
            pool = "train72" if node["real_pool"] == "ship72" else "all81"
            src["glob"] = str(output / "work/data" / pool / "test_real/*.h5")
            src["stats"] = str(output / "mean_std_real.pt")
    return cfg


def prepare(data_dir: str, output_dir: str) -> tuple[Path, dict]:
    data = Path(data_dir).resolve()
    output = Path(output_dir).resolve()
    assert (output / "work/data_manifest.json").is_file(), "run stage_final_data first"
    staged = json.loads((output / "work/data_manifest.json").read_text())
    assert Path(staged["source"]) == data and len(staged["real_files"]) == 81 and len(staged["sim_files"]) == 100
    spec = json.loads(SPEC.read_text()); pm.validate_graph(spec)
    work = output / "work/models"
    by = {n["id"]: n for n in spec["nodes"]}
    plan = {"schema": 1, "spec_sha256": hash_file(SPEC), "data_manifest_sha256": hash_file(output / "work/data_manifest.json"),
            "source": {}, "trainer": {}, "final_runner_sha256": hash_file(RUNNER),
            "source_gate_sha256": hash_file(ROOT / "common/final_source_gate.py"), "config": {}}
    for node in spec["nodes"]:
        name = node["id"]
        plan["source"][node["source"]] = hash_file(ROOT / node["source"])
        plan["trainer"][node["trainer"]] = hash_file(ROOT / node["trainer"])
        cfg = config_for_container(node, by, data, output)
        body = yaml.safe_dump(cfg, sort_keys=False)
        plan["config"][name] = hashlib.sha256(body.encode()).hexdigest()
    body = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    plan_path = work / "plan.json"
    if plan_path.exists():
        assert plan_path.read_text() == body, "container model plan drift; use fresh output"
    else:
        for d in ("configs", "checkpoints", "logs"):
            (work / d).mkdir(parents=True, exist_ok=True)
        plan_path.write_text(body)
    for node in spec["nodes"]:
        path = work / "configs" / (node["id"] + ".yaml")
        text = yaml.safe_dump(config_for_container(node, by, data, output), sort_keys=False)
        if path.exists():
            assert path.read_text() == text, "container config drift " + node["id"]
        else:
            path.write_text(text)
    print("CONTAINER_MODELS_PREPARED", work, len(spec["nodes"]))
    return work, spec


def run_node(data_dir: str, output_dir: str, name: str, gpu: int) -> None:
    # Final train() of one track (final_pipeline.track_skips): nodes that are not ancestors of that track's members.
    if name in {n for n in os.environ.get("RPDE_TRACK_SKIP_MODELS", "").split(",") if n}:
        print("TRACK_SKIP model", name, flush=True)
        return
    work, spec = prepare(data_dir, output_dir)
    by = {n["id"]: n for n in spec["nodes"]}
    assert name in by and gpu >= 0
    node = by[name]
    for parent in node["parents"]:
        ck = pm.terminal_path(work, by[parent])
        done = work / "logs" / (parent + ".complete.json")
        assert ck.is_file() and done.is_file() and hash_file(ck) == json.loads(done.read_text())["output_sha256"], parent
    ck = pm.terminal_path(work, node)
    done = work / "logs" / (name + ".complete.json")
    if done.exists():
        assert ck.is_file() and hash_file(ck) == json.loads(done.read_text())["output_sha256"]
        print("CONTAINER_MODEL_ALREADY_COMPLETE", name); return
    assert not ck.exists(), "output exists without proof: " + str(ck)
    log = work / "logs" / (name + ".train.log")
    assert not log.exists(), "prior training attempt exists; inspect before resuming"
    cmd = [sys.executable, "-u", str(RUNNER), "--trainer", node["trainer"],
           "--config", str(work / "configs" / (name + ".yaml")), "--gpu", str(gpu)]
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               RPDE_FINAL_DATA_ROOT=str(Path(data_dir).resolve()))
    with log.open("x") as f:
        f.write("COMMAND " + json.dumps(cmd) + "\n"); f.flush()
        rc = subprocess.call(cmd, cwd=ROOT / "Track1/U-net", stdout=f, stderr=subprocess.STDOUT, env=env)
    lines = log.read_text(errors="replace").splitlines()
    assert rc == 0 and any(s.startswith("TRAIN_V4_OK ") for s in lines), (name, rc, lines[-8:])
    assert ck.is_file() and ck.stat().st_size > 0
    record = {"status": "OK", "node": name, "utc": datetime.now(timezone.utc).isoformat(),
              "plan_sha256": hash_file(work / "plan.json"), "config_sha256": hash_file(work / "configs" / (name + ".yaml")),
              "trainer_sha256": hash_file(ROOT / node["trainer"]), "output": str(ck),
              "final_runner_sha256": hash_file(RUNNER), "source_gate_sha256": hash_file(ROOT / "common/final_source_gate.py"),
              "output_sha256": hash_file(ck), "log_sha256": hash_file(log)}
    done.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print("CONTAINER_MODEL_COMPLETE", name, record["output_sha256"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--node")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    if a.prepare: prepare(a.data_dir, a.output_dir)
    if a.node: run_node(a.data_dir, a.output_dir, a.node, a.gpu)
    assert a.prepare or a.node


if __name__ == "__main__":
    main()
