#!/usr/bin/env python3
"""Verify the completed shared model DAG and three shipped architecture roles."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from zipfile import ZipFile

import torch

import prepare_models as pm

ROOT = pm.ROOT


def state(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(d, dict) and "model_state_dict" in d
    s = d["model_state_dict"]
    assert s and all(isinstance(v, torch.Tensor) and torch.isfinite(v).all().item() for v in s.values())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--t1-reference", required=True)
    ap.add_argument("--t2-reference", required=True)
    a = ap.parse_args()
    t1 = (ROOT / a.t1_reference).resolve()
    t2 = (ROOT / a.t2_reference).resolve()
    assert t1.is_file() and t2.is_file()
    references = {"t1_geg_h": (t1, "model.pth"),
                  "t2_phk_h": (t2, "model.pth"),
                  "t2_ptk_sw": (t2, "model2.pth")}
    work = (ROOT / a.work).resolve()
    assert work.is_relative_to(Path(__file__).parent / "Logger")
    spec = json.loads(pm.SPEC.read_text()); pm.validate_graph(spec)
    plan = json.loads((work / "plan.json").read_text())
    assert plan["spec_sha256"] == pm.sha(pm.SPEC)
    complete = {}
    for n in spec["nodes"]:
        name = n["id"]
        cp = work / "logs" / (name + ".complete.json")
        info = json.loads(cp.read_text())
        out = pm.terminal_path(work, n)
        assert info["node"] == name and pm.sha(out) == info["output_sha256"]
        assert pm.sha(work / "configs" / (name + ".yaml")) == info["config_sha256"]
        assert pm.sha(ROOT / n["trainer"]) == info["trainer_sha256"]
        log = work / "logs" / (name + ".train.log")
        assert pm.sha(log) == info["log_sha256"]
        assert "TRAIN_V4_OK " in log.read_text(errors="replace")
        complete[name] = {"checkpoint": str(out.relative_to(ROOT)), "sha256": info["output_sha256"],
                          "utc": info["utc"], "parents": n["parents"]}
    for name, info in complete.items():
        for parent in info["parents"]:
            assert complete[parent]["utc"] < info["utc"], (name, parent)
    roles = {}
    for name, (zip_path, member) in references.items():
        s = state(ROOT / complete[name]["checkpoint"])
        with ZipFile(zip_path) as z:
            ref = torch.load(io.BytesIO(z.read(member)), map_location="cpu", weights_only=False)["model_state_dict"]
        same_keys = set(s) == set(ref)
        same_shapes = same_keys and all(s[k].shape == ref[k].shape for k in s)
        assert same_shapes, (name, same_keys)
        roles[name] = {"reference": str(zip_path.relative_to(ROOT)) + ":" + member,
                       "tensor_count": len(s), "parameter_count": sum(v.numel() for v in s.values()),
                       "architecture_shapes_match": True, "finite": True}
    doc = {"status": "SHARED_MODEL_DAG_COMPLETE", "nodes": complete, "shipping_roles": roles,
           "scope": "source training and architecture only; calibration/prior/card acceptance pending"}
    out = (ROOT / a.out).resolve()
    assert out.is_relative_to(work) and not out.exists()
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(doc["status"], len(complete), roles)


if __name__ == "__main__":
    main()
