#!/usr/bin/env python3
"""Regenerate v156g's KE/time-mean FTL warm start from public-data OOF captures.

The numerical utility identities are the frozen ftl_prior.py method.  This
wrapper supplies explicit new capture paths, checks all Re3+AoA2 fold sources,
then writes the two arrays that train() inserts into the inference runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RECIPE = HERE / "config/ftl_prior_recipe.json"
SPLIT = ROOT / "common/local_benchmark/splits_re3_aoa2_v1.json"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_pair(root: Path, fold: str, wanted: set[str]) -> tuple[dict, dict, dict]:
    out = []
    proofs = {}
    card_hash = None
    for kind in ("ke", "mean"):
        p = root / kind / (fold + "_T2.npz")
        side = Path(str(p) + ".json")
        rec = json.loads(side.read_text())
        assert rec["status"] == "OK" and rec["fold"] == fold and rec["track"] == 2
        assert rec["n_windows"] > 0 and rec.get("card")
        actual_card_hash = sha(Path(rec["card"]).resolve(strict=True))
        if card_hash is None:card_hash = actual_card_hash
        else:assert actual_card_hash == card_hash, (fold, "capture cards differ")
        with np.load(p) as stored:
            z = {k: stored[k] for k in stored.files}
        assert len(z["files"]) == rec["n_windows"] and {str(v) for v in z["files"]} == wanted
        assert len(z["files"]) == len(z["t0"]), kind
        out.append(z)
        proofs[kind] = {"path": str(p), "sha256": sha(p), "sidecar_sha256": sha(side),
                        "n_windows": rec["n_windows"]}
    assert np.array_equal(out[0]["files"], out[1]["files"])
    assert np.array_equal(out[0]["t0"], out[1]["t0"])
    proofs["card_sha256"] = card_hash
    return out[0], out[1], proofs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-root", required=True, help="contains ke/ and mean/ OOF capture subdirectories")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.stats_root).resolve(strict=True)
    out = Path(a.out).resolve()
    assert (out.is_relative_to(ROOT) or out.is_relative_to(Path(os.environ.get("RPDE_FINAL_OUTPUT", "/output")))) and not out.exists()
    recipe = json.loads(RECIPE.read_text())
    raw = json.loads(SPLIT.read_text())
    fixed = {f["id"]: set(f["val_files"]) for f in raw["folds"]}
    assert set(recipe["folds"]) == set(fixed) and len(recipe["folds"]) == 5
    assert set.union(*(fixed[f] for f in recipe["folds"][:3])) == set(raw["real_files"])
    assert set.union(*(fixed[f] for f in recipe["folds"][3:])) == set(raw["real_files"])
    sys.path.insert(0, str(ROOT / "Track2/U-net/VariantAdaptiveKE"))
    import ftl_prior as FP

    assert list(FP.LAMS) == recipe["lams"] and list(FP.MU) == recipe["mu_grid"]
    FP.MU = list(recipe["mu_grid"])
    FP.CG = list(recipe["gains"])
    assert len(FP.MU) * len(FP.CG) == 70
    lo, hi = recipe["clamp"]
    ke = np.zeros((70, 32), np.float64)
    mean = np.zeros((10, 32), np.float64)
    count = 0
    proof = {}
    for fold in recipe["folds"]:
        zk, zm, proof[fold] = load_pair(root, fold, fixed[fold])
        n = len(zk["files"])
        ke += FP.ke_utils(zk, lo, hi, FP.S.WT * recipe["wt_scale"]) * n
        mean += FP.mean_utils(zm) * n
        count += n
    assert count > 0
    ke /= count; mean /= count
    i0 = FP.MU.index(0.1) * len(FP.CG) + FP.CG.index(1.0)
    assert i0 == 11
    prior_ke = ke - ke[i0][None]
    prior_mean = mean - mean[0][None]
    tables = {"_FP_KE": (recipe["kappa_ke"] * prior_ke).tolist(),
              "_FP_MEAN": (recipe["kappa_mean"] * prior_mean).tolist()}
    assert np.isfinite(np.asarray(tables["_FP_KE"])).all() and np.isfinite(np.asarray(tables["_FP_MEAN"])).all()
    assert np.allclose(tables["_FP_KE"][i0], 0, atol=1e-12)
    assert np.allclose(tables["_FP_MEAN"][0], 0, atol=1e-12)
    doc = {"status": "SOURCE_FIT", "recipe_sha256": sha(RECIPE), "split_sha256": sha(SPLIT),
           "utility_source_sha256": sha(ROOT / "Track2/U-net/VariantAdaptiveKE/ftl_prior.py"),
           "n_windows": count, "capture_proof": proof, "reference_arm": i0,
           "runtime_tables": tables,
           "note": "Full member OOF is checked here; separate lineage checks are required for fold card head, band and P1."}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print("FTL_PRIOR_OOF_FIT_OK", out, count, i0)


if __name__ == "__main__":
    main()
