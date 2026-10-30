#!/usr/bin/env python3
"""Fit the v185 interval-centre U table from newly captured Re3 member outputs.

For each lead/channel and each fixed grid point, score the actual served
interval after the centre shift, keeping the point prediction and width fixed.
Only out-of-fold captured windows enter the fit.  The capture sidecars and the
fixed split manifest are mandatory so a mislabelled all81 card cannot enter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SPLIT = ROOT / "common/local_benchmark/splits_re3_aoa2_v1.json"
FOLDS = ("re0", "re1", "re2")
GRID = np.round(np.arange(-0.8, 0.8 + 1e-9, 0.05), 3).astype(np.float32)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_files() -> dict[str, set[str]]:
    raw = json.loads(SPLIT.read_text())
    folds = {f["id"]: set(f["val_files"]) for f in raw["folds"] if f["id"] in FOLDS}
    assert set(folds) == set(FOLDS) and set.union(*folds.values()) == set(raw["real_files"])
    assert not any(folds[a] & folds[b] for a in FOLDS for b in FOLDS if a != b)
    return folds


def capture_score(path: Path, fold: str, want: set[str], device: str) -> tuple[np.ndarray, dict]:
    import torch

    side = Path(str(path) + ".json")
    rec = json.loads(side.read_text())
    assert rec["status"] == "OK" and rec["fold"] == fold and rec["track"] == 1 and rec.get("members") is True
    assert rec["card_sha256"] and rec["n_windows"] > 0
    with np.load(path) as stored:
        z = {k: stored[k] for k in ("files", "pred", "tgt", "lo", "hi", "y1", "y2",
                                         "sigma", "rel", "tke", "mvpe")}
    files = [str(s) for s in z["files"]]
    assert len(files) == rec["n_windows"] and set(files) == want, fold
    assert all(z[k].shape == (len(files), 20, 32, 64, 2)
               for k in ("pred", "tgt", "lo", "hi", "y1", "y2")), fold
    sigma = float(z["sigma"])
    assert 0 < sigma < 1 and np.isfinite(sigma)
    rel, tke, mv = (np.asarray(z[k], np.float64) for k in ("rel", "tke", "mvpe"))
    acc = 1.0 - (0.5 * rel / (0.5 + rel) + 0.3 * tke / (0.5 + tke) + 0.2 * mv / (0.5 + mv))
    assert np.isfinite(acc).all() and (acc > 0).all()
    N = len(files)
    out = np.zeros((20, 2, len(GRID)), np.float64)
    for t in range(20):
        for c in range(2):
            tg = torch.as_tensor(np.asarray(z["tgt"][:, t, :, :, c]), dtype=torch.float32, device=device)
            lo = torch.as_tensor(np.asarray(z["lo"][:, t, :, :, c]), dtype=torch.float32, device=device)
            hi = torch.as_tensor(np.asarray(z["hi"][:, t, :, :, c]), dtype=torch.float32, device=device)
            dy = torch.as_tensor(np.asarray(z["y1"][:, t, :, :, c] - z["y2"][:, t, :, :, c]),
                                 dtype=torch.float32, device=device)
            # The original instrument computes this one width-dependent factor
            # in float64; it does not depend on U and need only be computed once.
            e = np.exp(-(np.asarray(z["hi"][:, t, :, :, c], np.float64)
                         - np.asarray(z["lo"][:, t, :, :, c], np.float64)) / sigma)
            w = torch.as_tensor(e * acc[:, None, None], dtype=torch.float64, device=device)
            mask = tg != 0
            for j, raw in enumerate(GRID):
                delta = dy * torch.tensor(float(raw), dtype=torch.float32, device=device)
                inside = (tg >= lo + delta) & (tg <= hi + delta) & mask
                out[t, c, j] = float(w[inside].sum().item())
            del tg, lo, hi, dy, w
    meta = {"capture": str(path), "capture_sha256": sha(path), "sidecar_sha256": sha(side),
            "card_sha256": rec["card_sha256"], "n_windows": N,
            "files": len(want), "sigma": sigma}
    return out, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", required=True, help="directory with re0.npz,re1.npz,re2.npz and sidecars")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    a = ap.parse_args()
    captures = Path(a.captures).resolve(strict=True)
    out = Path(a.out).resolve()
    assert (out.is_relative_to(ROOT) or out.is_relative_to(Path(os.environ.get("RPDE_FINAL_OUTPUT", "/output")))) and not out.exists()
    expect = expected_files()
    score, proof = {}, {}
    sigma = None
    for fold in FOLDS:
        p = captures / (fold + ".npz")
        score[fold], proof[fold] = capture_score(p, fold, expect[fold], a.device)
        if sigma is None: sigma = proof[fold]["sigma"]
        else: assert sigma == proof[fold]["sigma"], "sigma changed between folds"
    tables, heldout = {}, {}
    for name in FOLDS + ("all81",):
        train = [f for f in FOLDS if f != name]
        fit = sum(score[f] for f in train)
        ids = fit.argmax(axis=-1)
        u = GRID[ids]
        tables[name] = {"u_u": u[:, 0].astype(float).tolist(), "u_v": u[:, 1].astype(float).tolist()}
        if name in FOLDS:
            fold_score = score[name]
            heldout[name] = {"selected_weighted_S": float(np.take_along_axis(fold_score, ids[..., None], -1).sum()),
                             "zero_weighted_S": float(fold_score[:, :, list(GRID).index(0.0)].sum())}
    doc = {"status": "SOURCE_FIT", "split_manifest_sha256": sha(SPLIT),
           "grid": GRID.astype(float).tolist(), "capture_proof": proof,
           "tables": tables, "heldout_diagnostic": heldout,
           "note": "The head/band used by capture cards must be checked separately for nested-fit lineage; this script certifies member OOF and the table computation only."}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print("SPSR_OOF_FIT_OK", out, "all81_first", tables["all81"]["u_u"][0], tables["all81"]["u_v"][0])


if __name__ == "__main__":
    main()
