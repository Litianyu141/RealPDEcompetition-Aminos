#!/usr/bin/env python3
"""Held-out band-width response of a fold card, on the fixed Re3+AoA2 protocol.

Runs one fold's own card through its real interface (T1 predict / T2 streaming TTT, reusing
common/local_benchmark/splitsel/score_card.py verbatim) on that fold's HELD-OUT files only, and
records per-window SPS numerators for a grid of global band-width multipliers m.

Two one-parameter families are recorded:
  mid : band scaled about its own midpoint c=(lo+hi)/2  (what common/scale_band.py ships)
  pred: band offsets about the prediction scaled by m    (score_card.scale_bounds)
For the mid family the numerator is also split by channel and lead frame, so per-channel /
per-lead multipliers can be read off the same pass: S_mid_cf[w, m, ch, frame].

The prediction is never modified, so rel/tke/mvpe and each window's accuracy factor are fixed and
sps(m) = sum_w acc_w * S_w(m) / sum_w n_w exactly (score_card identity; self-checked at m=1 against
the kit's aggregate_sps on the first files).

    python common/band_width_fivefold.py --card <fold card.zip> --fold re2 --track 1 --gpu 0 \
        --work /tmp/bwf_work --out <repo Logger path>/re2_T1.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "common/local_benchmark/splitsel"))
import score_card as SC  # noqa: E402

MGRID = np.array([0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5,
                  1.6, 1.8, 2.0, 2.3, 2.6, 3.0], np.float64)
I1 = int(np.where(MGRID == 1.0)[0][0])
SPLITS = REPO / "common/local_benchmark/splits_re3_aoa2_v1.json"

STASH: list = []


def heldout_files(fold_id: str):
    d = json.loads(SPLITS.read_text())
    fold = [f for f in d["folds"] if f["id"] == fold_id]
    assert len(fold) == 1, fold_id
    train = set(fold[0]["train_files"])
    allf = SC.real_files()
    held = [f for f in allf if Path(f).name not in train]
    assert held and len(held) + len(train) == len(allf), (len(held), len(train), len(allf))
    return held


def ext_stats(sc, pred, tgt, lower, upper, jgrid=False):
    base = SC.__dict__["_orig_pws"](sc, pred, tgt, lower, upper, jgrid=False)
    c = base["c"]
    p = pred[..., :c].astype(np.float32)
    t = tgt[..., :c].astype(np.float32)
    lo = np.asarray(lower, np.float32)[..., :c]
    hi = np.asarray(upper, np.float32)[..., :c]
    scored = t != 0.0
    width = hi - lo
    nil0 = (hi - lo) / sc.SIGMA_GLOBAL          # identical expression/dtype to per_window_stats
    hw = np.float32(0.5) * width
    wlo = p - lo            # may be negative when p lies outside the band
    whi = hi - p
    N, T = p.shape[0], p.shape[1]
    M = len(MGRID)
    S_mid_cf = np.zeros((N, M, c, T), np.float64)
    nin_mid = np.zeros((N, M), np.int64)
    S_pred = np.zeros((N, M), np.float64)
    nin_pred = np.zeros((N, M), np.int64)
    sp_ax = (2, 3)            # H, W inside (N,T,H,W,c)
    for im, m in enumerate(MGRID):
        d = np.float32(m - 1.0)           # exactly 0 at m=1, so the bounds are bit-identical there
        e = np.exp(-nil0) if m == 1.0 else np.exp(-np.float32(m) * nil0)
        ins = (t >= lo - d * hw) & (t <= hi + d * hw) & scored        # about the midpoint
        val = np.where(ins, e, 0.0)                         # (N,T,H,W,c)
        S_mid_cf[:, im] = val.sum(axis=sp_ax, dtype=np.float64).transpose(0, 2, 1)
        nin_mid[:, im] = ins.sum(axis=(1, 2, 3, 4))
        insp = (t >= lo - d * wlo) & (t <= hi + d * whi) & scored     # about the prediction
        S_pred[:, im] = np.where(insp, e, 0.0).sum(axis=(1, 2, 3, 4), dtype=np.float64)
        nin_pred[:, im] = insp.sum(axis=(1, 2, 3, 4))
    # m=1 must reproduce the unscaled statistics exactly up to float32 summation order
    S1 = S_mid_cf[:, I1].sum(axis=(1, 2))
    assert np.array_equal(nin_mid[:, I1], base["nin"]) and np.array_equal(nin_pred[:, I1], base["nin"]), \
        (np.abs(nin_mid[:, I1] - base["nin"]).max(), np.abs(nin_pred[:, I1] - base["nin"]).max())
    assert np.allclose(S1, base["S"], rtol=1e-9, atol=1e-6), np.abs(S1 - base["S"]).max()
    assert np.allclose(S_pred[:, I1], base["S"], rtol=1e-9, atol=1e-6), np.abs(S_pred[:, I1] - base["S"]).max()
    STASH.append(dict(S_mid_cf=S_mid_cf, nin_mid=nin_mid, S_pred=S_pred, nin_pred=nin_pred,
                      rel=base["rel"], tke=base["tke"], mvpe=base["mvpe"], n=base["n"],
                      n_c=scored.sum(axis=(1, 2, 3)).astype(np.int64)))
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--track", type=int, required=True, choices=[1, 2])
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--online", action="append", help="T2 ONLINE override KEY=JSON (score_card semantics)")
    ap.add_argument("--set", action="append", help="module constant override NAME=JSON (score_card semantics)")
    a = ap.parse_args()
    for k in ("card", "work", "out"):
        setattr(a, k, os.path.abspath(getattr(a, k)))
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(card=a.card, fold=a.fold, track=a.track, status="FAIL", mgrid=MGRID.tolist(),
                started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    t0 = time.time()
    try:
        meta["card_sha256"] = SC.archive_sha256(a.card)
        sc = SC.load_scoring()
        files = heldout_files(a.fold)
        SC.__dict__["_orig_pws"] = SC.per_window_stats
        SC.per_window_stats = ext_stats
        SC.real_files = lambda: files
        sub = SC.extract(Path(a.card), Path(a.work))
        opts = argparse.Namespace(set=a.set, online=a.online, fluct_mult=1.0, band_mult=1.0, jgrid=False,
                                  _JS=[], _JC=[])
        meta["overrides"] = dict(set=a.set or [], online=a.online or [])
        if a.track == 1:
            rows, checks, extra = SC.run_t1(sub, sc, files, a.chunk, opts)
        else:
            rows, checks, extra = SC.run_t2(sub, sc, files, "cuda", opts)
        assert all(ch["abs_diff"] < 1e-6 for ch in checks), checks
        assert SC.archive_sha256(a.card) == meta["card_sha256"]
        cat = {k: np.concatenate([s[k] for s in STASH], 0) for k in STASH[0]}
        np.savez_compressed(str(out), files=np.array([r[0] for r in rows]),
                            t0=np.array([r[1] for r in rows]), mgrid=MGRID, **cat)
        meta.update(status="OK", n_windows=len(rows), n_files=len(files), selfcheck=checks, **extra)
    except Exception as e:  # noqa: BLE001
        meta.update(error="%s: %s" % (type(e).__name__, e), traceback=traceback.format_exc())
    meta["elapsed_s"] = time.time() - t0
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=1))
    print(a.fold, a.track, meta["status"], meta.get("n_windows"), "%.0fs" % meta["elapsed_s"], meta.get("error", ""))


if __name__ == "__main__":
    main()
