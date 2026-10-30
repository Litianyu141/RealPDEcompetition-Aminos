#!/usr/bin/env python3
"""Seal the source-generated T1 regime, LK and SPSR tables for one card stage."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import make_runtime_template as RT

ROOT = Path(__file__).resolve().parents[3]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def filled(shape, value):
    return value if not shape else [filled(shape[1:], value) for _ in range(shape[0])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--lk")
    ap.add_argument("--u")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cluster = Path(a.cluster).resolve(strict=True)
    lk = Path(a.lk).resolve(strict=True) if a.lk else None
    u = Path(a.u).resolve(strict=True) if a.u else None
    out = Path(a.out).resolve()
    assert (out.is_relative_to(ROOT) or out.is_relative_to(Path(os.environ.get("RPDE_FINAL_OUTPUT", "/output")))) and not out.exists()
    c = json.loads(cluster.read_text())
    assert c["K"] == 3 and c["seed"] == 0 and c["block"] == [8, 8]
    tables = {"_KC_FM": c["feat_mean"], "_KC_FS": c["feat_std"],
              "_KC_CEN": c["centroids"], "_KC_MU": c["mu_maps"], "_KC_LAM": c["lam_maps"],
              "_SY_GRAW": filled(RT.TABLES["T1"]["_SY_GRAW"], 1.0),
              "_BCD_SR_TABLE": filled(RT.TABLES["T1"]["_BCD_SR_TABLE"], 0.0)}
    if lk is not None:
        z = json.loads(lk.read_text())
        assert z["status"] == "SOURCE_FIT" and z["cluster_sha256"] == sha(cluster)
        tables["_SY_GRAW"] = z["_SY_GRAW"]
    if u is not None:
        assert lk is not None, "SPSR fit must use a generated LK table"
        z = json.loads(u.read_text())
        assert z["status"] == "SOURCE_FIT"
        t = z["tables"]["all81"]
        tables["_BCD_SR_TABLE"] = [[float(t["u_u"][i]), float(t["u_v"][i])] for i in range(20)]
    assert set(tables) == set(RT.TABLES["T1"])
    for name, value in tables.items():
        assert RT.shape(value) == RT.TABLES["T1"][name], (name, RT.shape(value))
    doc = {"status": "SOURCE_FIT", "stage": ("cluster_lk_spsr" if u else "cluster_lk" if lk else "cluster"),
           "runtime_tables": tables, "source_proof": {"cluster_sha256": sha(cluster),
                                                      "lk_sha256": sha(lk) if lk else None,
                                                      "spsr_sha256": sha(u) if u else None},
           "note": "The fixed F3 and BK numerical hyperparameters remain method settings; all stateful regime/LK/SPSR arrays were generated in train()."}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print("T1_RUNTIME_TABLES_OK", doc["stage"], out)


if __name__ == "__main__":
    main()
