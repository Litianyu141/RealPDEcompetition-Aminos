#!/usr/bin/env python3
"""Audit every Final OOF checkpoint against its fixed split and completion proof."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import yaml

import aoa_folds as AF
import band_teachers as BT
import re3_members as RM

ROOT = Path(__file__).resolve().parents[3]
SPLIT = ROOT / "common/local_benchmark/splits_re3_aoa2_v1.json"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def data_guard(config: dict, fold: dict, strict_sim: bool) -> list[int]:
    real = config["stages"][1]["sources"][0]["keep_files"]
    sim = config["stages"][0]["sources"][0]["keep_files"]
    assert len(real) == len(set(real)) and set(real) == set(fold["train_files"])
    assert not (set(real) & set(fold["val_files"]))
    axis = 0 if fold["axis"] == "re" else 1
    val_groups = set(fold["val_groups"])
    overlap = sorted({int(Path(name).stem.split("_")[axis]) for name in sim} & val_groups)
    if strict_sim:
        assert not overlap, (fold["id"], "teacher sim producer saw held-out group", overlap)
    else:
        # The inherited TST5/RA3/current-member recipes deliberately pretrain
        # on all 100 public sim trajectories, including held-out real regimes.
        # This is legal Final training data, but it is NOT strict regime-OOF
        # evidence for a calibrated local private-score forecast.
        assert len(sim) == 100 and overlap, (fold["id"], "unexpected sim recipe", overlap)
    assert "7575_0.h5" not in real
    return overlap


def verify_node(work: Path, name: str, output: Path, config: Path,
                source: Path, trainer: Path, fold: dict, parents: list[Path]) -> dict:
    proof = work / "logs" / (name + ".complete.json")
    log = work / "logs" / (name + ".train.log")
    rec = json.loads(proof.read_text())
    assert rec["status"] == "OK" and rec["node"] == name
    assert output.is_file() and sha(output) == rec["output_sha256"]
    assert sha(config) == rec["config_sha256"] and sha(trainer) == rec["trainer_sha256"]
    assert log.is_file() and sha(log) == rec["log_sha256"]
    c = yaml.safe_load(config.read_text())
    strict_sim = "VariantFiveFoldEvaluation" in source.parts
    overlap = data_guard(c, fold, strict_sim)
    got_parents = [Path(s).resolve(strict=True) for s in str(c.get("distill_ckpt", "") or "").split(",") if s]
    want_parents = [p.resolve(strict=True) for p in parents]
    assert got_parents == want_parents, (name, "teacher path/order mismatch")
    parent_hashes = [sha(p) for p in want_parents]
    ck = torch.load(output, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    endpoint = sum(int(s["steps"]) for s in c["stages"])
    assert meta["selection"] == "fixed_terminal_ema" and meta["stage"] == "final"
    assert int(meta["gstep"]) == endpoint and int(meta["fold"]) == 0
    assert len(ck["model_state_dict"]) > 0
    return {"node": name, "output_sha256": rec["output_sha256"],
            "config_sha256": rec["config_sha256"], "source_sha256": sha(source),
            "terminal_gstep": endpoint, "train_files": len(fold["train_files"]),
            "val_files": len(fold["val_files"]), "sim_overlap_val_groups": overlap,
            "strict_sim_isolation": strict_sim, "parent_sha256": parent_hashes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("re3", "aoa"), required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--band-work")
    ap.add_argument("--skip-roles", default="", help="comma list of roles this track's train() does not train")
    a = ap.parse_args()
    skip = {r for r in a.skip_roles.split(",") if r}
    raw = Path(a.work).absolute()
    work = raw.resolve(strict=True)
    # train() links ROOT's Logger/ into OUT/work (stage_legacy_layout), so the resolved path lies under OUT: /output in the
    # organizer container, the run's output directory elsewhere. Accept either the given path under ROOT or a resolved
    # path under ROOT or /output.
    assert raw.is_relative_to(ROOT) or work.is_relative_to(ROOT) or work.is_relative_to(Path(os.environ.get("RPDE_FINAL_OUTPUT", "/output"))), (raw, work)
    split = json.loads(SPLIT.read_text())
    fixed = {f["id"]: f for f in split["folds"]}
    rows = []
    if a.kind == "re3":
        assert a.band_work
        band = Path(a.band_work).resolve(strict=True)
        bs = json.loads(BT.SPEC.read_text())
        for job in BT.jobs(bs):
            fold = fixed[job["fold"]]
            rows.append(verify_node(band, job["name"], BT.output(band, job["name"]),
                                    band / "configs" / (job["name"] + ".yaml"),
                                    ROOT / job["source"], ROOT / bs["trainer"], fold,
                                    [BT.output(band, p) for p in job["parents"]]))
        rs = json.loads(RM.SPEC.read_text())
        for node in RM.nodes(rs):
            if node["role"] in skip:
                rows.append({"node": node["id"], "skipped": "not consumed by this track's Final card"})
                continue
            fold = fixed[node["fold"]]
            rows.append(verify_node(work, node["id"], RM.output(work, node["fold"], node["role"]),
                                    work / "configs" / (node["id"] + ".yaml"),
                                    ROOT / node["source"], ROOT / rs["trainer"], fold,
                                    [RM.parent_output(work, band, node["fold"], p) for p in node["parents"]]))
        assert len(rows) == 36
    else:
        assert not a.band_work
        for node in AF.nodes():
            if node["role"] in skip:
                rows.append({"node": node["id"], "skipped": "not consumed by this track's Final card"})
                continue
            fold = fixed[node["fold"]]
            rows.append(verify_node(work, node["id"], AF.ckpt(work, node),
                                    work / "configs" / (node["id"] + ".yaml"),
                                    ROOT / node["source"], ROOT / node["trainer"], fold,
                                    [AF.ckpt(work, p) for p in node["parents"]]))
        assert len(rows) == 24
    out = work / "fold_graph_audit.json"
    doc = {"status": "PUBLIC_SOURCE_VALID_REAL_OOF_SIM_REGIME_EXPOSURE", "kind": a.kind,
           "private_score_calibration": "NOT_STRICT", "split_sha256": sha(SPLIT),
           "n_nodes": len(rows), "nodes": rows}
    body = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    if out.exists():assert out.read_text() == body, "OOF audit drift"
    else:out.write_text(body)
    print("FOLD_SOURCE_AUDIT_OK", a.kind, len(rows), out, sha(out), "SIM_EXPOSURE_DISCLOSED")


if __name__ == "__main__":
    main()
