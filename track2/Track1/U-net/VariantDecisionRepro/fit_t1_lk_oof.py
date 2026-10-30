#!/usr/bin/env python3
"""Fit T1's 3×20×4×8×2 KE-preserving temporal gain from Re3 OOF captures.

The sufficient-statistic identity and per-cluster energy normalization match
the historical synth_lib.lk_stats/lk_table formulas.  Regime labels come from
this run's generated cluster map, and input windows are reread from the legal
public real files; no old card or cached fitted array enters the computation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SPLIT = ROOT / "common/local_benchmark/splits_re3_aoa2_v1.json"
FOLDS = ("re0", "re1", "re2")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def window_features(real_dir: Path, files: np.ndarray, t0: np.ndarray) -> np.ndarray:
    out = np.empty((len(files), 64), np.float32)
    names = [str(v) for v in files]
    for name in sorted(set(names)):
        idx = np.array([i for i, f in enumerate(names) if f == name], dtype=np.int64)
        with h5py.File(real_dir / name, "r") as f:
            x = np.stack([np.asarray(f[k][:, ::2, ::2], np.float32) for k in ("u", "v")], -1)
        assert x.shape[1:] == (32, 64, 2)
        for i in idx:
            s = int(t0[i]); assert 0 <= s <= len(x) - 40
            xm = x[s:s + 20].mean(axis=0, dtype=np.float32)
            out[i] = xm.reshape(4, 8, 8, 8, 2).mean(axis=(1, 3), dtype=np.float32).reshape(-1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", required=True)
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    captures = Path(a.captures).resolve(strict=True)
    cluster = Path(a.cluster).resolve(strict=True)
    real_dir = Path(a.real_dir).resolve(strict=True)
    out = Path(a.out).resolve()
    assert (out.is_relative_to(ROOT) or out.is_relative_to(Path(os.environ.get("RPDE_FINAL_OUTPUT", "/output")))) and not out.exists()
    split = json.loads(SPLIT.read_text())
    folds = {f["id"]: set(f["val_files"]) for f in split["folds"] if f["id"] in FOLDS}
    cm = json.loads(cluster.read_text())
    assert cm["K"] == 3 and cm["seed"] == 0 and cm["block"] == [8, 8]
    fm = np.asarray(cm["feat_mean"], np.float32)
    fs = np.asarray(cm["feat_std"], np.float32)
    cen = np.asarray(cm["centroids"], np.float32)
    assert fm.shape == fs.shape == (64,) and cen.shape == (3, 64) and np.all(fs > 0)
    C = np.zeros((3, 20, 4, 8, 2), np.float64)
    V = np.zeros_like(C)
    label_counts = np.zeros(3, np.int64)
    proof = {}
    for fold in FOLDS:
        p = captures / (fold + ".npz")
        side = Path(str(p) + ".json")
        rec = json.loads(side.read_text())
        assert rec["status"] == "OK" and rec["fold"] == fold and rec["track"] == 1 and rec.get("members") is True
        with np.load(p) as stored:
            z = {k: stored[k] for k in ("files", "t0", "pred", "tgt")}
        assert len(z["files"]) == rec["n_windows"] and set(str(v) for v in z["files"]) == folds[fold]
        assert z["pred"].shape == z["tgt"].shape == (len(z["files"]), 20, 32, 64, 2)
        feat = window_features(real_dir, z["files"], z["t0"])
        zz = (feat - fm) / fs
        lab = ((zz[:, None, :] - cen[None]) ** 2).sum(-1).argmin(axis=1)
        label_counts += np.bincount(lab, minlength=3)
        for i in range(0, len(lab), 64):
            sl = slice(i, min(len(lab), i + 64))
            pred = z["pred"][sl].astype(np.float64)
            tgt = z["tgt"][sl].astype(np.float64)
            m = pred.mean(axis=1, keepdims=True)
            f = pred - m
            f2 = (f * (tgt - m)).reshape(len(pred), 20, 4, 8, 8, 8, 2).sum(axis=(3, 5))
            v2 = (f * f).reshape(len(pred), 20, 4, 8, 8, 8, 2).sum(axis=(3, 5))
            for k in range(3):
                sel = lab[sl] == k
                if sel.any():
                    C[k] += f2[sel].sum(axis=0)
                    V[k] += v2[sel].sum(axis=0)
        proof[fold] = {"capture_sha256": sha(p), "sidecar_sha256": sha(side),
                       "card_sha256": rec["card_sha256"], "n_windows": rec["n_windows"]}
    assert (label_counts >= 20).all(), ("empty regime", label_counts.tolist())
    g = C / np.clip(V, 1e-30, None)
    lam = np.sqrt(V.sum(axis=1) / np.clip((g * g * V).sum(axis=1), 1e-30, None))
    raw = (g * lam[:, None]).astype(np.float32)
    assert raw.shape == (3, 20, 4, 8, 2) and np.isfinite(raw).all()
    doc = {"status": "SOURCE_FIT", "shape": list(raw.shape),
           "_SY_GRAW": np.round(raw.astype(np.float64), 7).tolist(),
           "split_sha256": sha(SPLIT), "cluster_sha256": sha(cluster),
           "capture_proof": proof, "label_counts": label_counts.tolist(),
           "formula": "g=C/V; lam=sqrt(sum_lead V / sum_lead(g^2 V)); raw=g*lam, float32 then 7-digit table"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print("T1_LK_OOF_FIT_OK", out, label_counts.tolist(), flush=True)


if __name__ == "__main__":
    main()
