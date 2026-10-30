#!/usr/bin/env python3
"""Re-scope SBAND2's 12 fixed Re-OOF teachers to Final-kit /data and /output.

The original source recipes, source allowlists, split groups and training knobs
remain unchanged. Manifest file paths are rewritten to the official mount.
This module prepares configs; acceptance still requires actual source-only runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

import band_teachers as bt

ROOT = bt.ROOT
SPEC = bt.SPEC
ORIG_REAL = "data/RealPDE_competition_dataset/gdrive_raw/extracted/train_real/train_real/"
ORIG_SIM = "data/RealPDE_competition_dataset/gdrive_raw/extracted/train_sim/train_sim/"


def manifest(fold: str, data: Path) -> dict:
    src = ROOT / "Track1/U-net/VariantFiveFoldEvaluation/config/manifests" / (fold + "_outer.json")
    m = json.loads(src.read_text())
    assert m["k"] == 1 and m["group_by"] == "re" and len(m["folds"]) == 1
    part = m["folds"][0]
    for key, old, new in (("train_files", ORIG_REAL, data / "train_real"),
                          ("val_files", ORIG_REAL, data / "train_real"),
                          ("sim_train_files", ORIG_SIM, data / "train_sim")):
        files = part[key]
        assert files and all(s.startswith(old) for s in files), (fold, key)
        part[key] = [str(new / Path(s).name) for s in files]
        assert all(Path(s).is_file() for s in part[key]), (fold, key, "missing released input")
    assert len(part["train_files"]) in {53, 54, 55}
    assert len(part["train_files"]) + len(part["val_files"]) == 81
    assert "7575_0.h5" not in {Path(s).name for s in part["train_files"] + part["val_files"]}
    return m


def config(job: dict, work: Path, data: Path, output: Path) -> dict:
    c = bt.config(job, work)
    c["stats"] = str(output / "mean_std_real.pt")
    c["kfold_manifest"] = str(work / "manifests" / (job["fold"] + "_outer.json"))
    for stage in c["stages"]:
        assert len(stage["sources"]) == 1
        src = stage["sources"][0]
        if stage["name"] == "sim-pretrain":
            assert src["tag"] == "comp_sim"
            src["glob"] = str(data / "train_sim/*.h5")
            src["stats"] = str(output / "work/data/mean_std_comp_sim.pt")
        else:
            assert stage["name"] == "real-finetune" and src["tag"] == "comp_real"
            src["glob"] = str(data / "train_real/*.h5")
            src["stats"] = str(output / "mean_std_real.pt")
    return c


def prepare(data_dir: str, output_dir: str) -> None:
    data = Path(data_dir).resolve()
    output = Path(output_dir).resolve()
    stage = json.loads((output / "work/data_manifest.json").read_text())
    assert Path(stage["source"]) == data and len(stage["real_files"]) == 81 and len(stage["sim_files"]) == 100
    spec = json.loads(SPEC.read_text())
    work = output / "work/band_teachers"
    cfgs = {}
    mans = {}
    for fold in spec["folds"]:
        body = json.dumps(manifest(fold, data), indent=2, sort_keys=True) + "\n"
        mans[fold] = body
    for job in bt.jobs(spec):
        cfgs[job["name"]] = yaml.safe_dump(config(job, work, data, output), sort_keys=False)
    record = {"spec_sha256": bt.sha(SPEC), "data_manifest_sha256": bt.sha(output / "work/data_manifest.json"),
              "trainer_sha256": bt.sha(ROOT / spec["trainer"]),
              "final_runner_sha256": bt.sha(Path(__file__).parent / "final_trainer.py"),
              "source_gate_sha256": bt.sha(ROOT / "common/final_source_gate.py"),
              "configs_sha256": {n: hashlib.sha256(t.encode()).hexdigest() for n, t in cfgs.items()},
              "manifests_sha256": {n: hashlib.sha256(t.encode()).hexdigest() for n, t in mans.items()}}
    plan = json.dumps(record, indent=2, sort_keys=True) + "\n"
    p = work / "plan.json"
    if p.exists():
        assert p.read_text() == plan, "Final band-teacher plan drift"
    else:
        for d in ("configs", "manifests", "checkpoints", "logs"):
            (work / d).mkdir(parents=True, exist_ok=True)
        p.write_text(plan)
    for fold, body in mans.items():
        p = work / "manifests" / (fold + "_outer.json")
        if p.exists(): assert p.read_text() == body
        else: p.write_text(body)
    for name, body in cfgs.items():
        p = work / "configs" / (name + ".yaml")
        if p.exists(): assert p.read_text() == body
        else: p.write_text(body)
    print("CONTAINER_BAND_TEACHERS_PREPARED", len(cfgs), len(mans), work)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()
    prepare(a.data_dir, a.output_dir)
