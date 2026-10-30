#!/usr/bin/env python3
"""Held-out response of a fold card to TIME-MEAN ANCHORING on the input window (Track 1).

    p'(t) = p(t) + lam * (mean_t x_in - mean_t p)          (u,v only; band shifted by the same field)

The fluctuation p - mean_t p is untouched, so tke is exactly invariant; rel, mvpe and the SPS interval
term move.  lam=0 is the card.  Optional spatial smoothing of the anchor field (box k x k) is recorded
as a second family, since the input time-mean carries its own 20-frame sampling noise.

Per window and per lam it stores rel, mvpe and S (interval numerator with the band shifted along with
the prediction), so any lam or per-axis aggregation is exact afterwards.  Card run through
score_card.run_t1 logic (same windows, same chunking), fold held-out files only.
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
sys.path.insert(0, str(REPO / "common"))
import score_card as SC  # noqa: E402
from band_width_fivefold import heldout_files  # noqa: E402

LAMS = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
SMOOTH = [1, 3, 5]


def box(a, k):
    """k x k box mean over (H,W) of (N,H,W,C) with edge replication."""
    if k == 1:
        return a
    p = k // 2
    ap = np.pad(a, ((0, 0), (p, p), (p, p), (0, 0)), mode="edge")
    c = ap.cumsum(1).cumsum(2)
    c = np.pad(c, ((0, 0), (1, 0), (1, 0), (0, 0)))
    H, W = a.shape[1], a.shape[2]
    s = c[:, k:k + H, k:k + W] - c[:, :H, k:k + W] - c[:, k:k + H, :W] + c[:, :H, :W]
    return s / (k * k)


def stats(sc, xin, pred, tgt, lower, upper):
    c = sc.measured_channels(tgt)
    out = {}
    base_rel = sc.rel_l2_per_sample(pred, tgt, c)
    base_tke = sc.tke_rel_l2_per_sample(pred, tgt, c)
    pm = pred[..., :2].mean(axis=1)                        # (N,H,W,2)
    im = xin[..., :2].mean(axis=1)
    t = tgt[..., :c]
    scored = t != 0.0
    for k in SMOOTH:
        d = box(im - pm, k)                                # anchor displacement field
        R, M, S, NIN = [], [], [], []
        for lam in LAMS:
            sh = np.zeros_like(pred)
            sh[..., :2] = (lam * d)[:, None]
            p2 = pred + sh
            lo2 = lower + sh
            hi2 = upper + sh
            R.append(sc.rel_l2_per_sample(p2, tgt, c))
            M.append(sc.mvpe_rel_l2_per_sample(p2, tgt))
            lo_c, hi_c = lo2[..., :c], hi2[..., :c]
            inside = (t >= lo_c) & (t <= hi_c) & scored
            nil = (hi_c - lo_c) / sc.SIGMA_GLOBAL
            S.append(np.where(inside, np.exp(-nil), 0.0).sum(axis=(1, 2, 3, 4), dtype=np.float64))
            NIN.append(inside.sum(axis=(1, 2, 3, 4)))
        out[k] = dict(rel=np.stack(R, 1), mvpe=np.stack(M, 1), S=np.stack(S, 1), nin=np.stack(NIN, 1))
    return dict(tke=base_tke, rel0=base_rel, n=scored.sum(axis=(1, 2, 3, 4)), fam=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--chunk", type=int, default=64)
    a = ap.parse_args()
    for k in ("card", "work", "out"):
        setattr(a, k, os.path.abspath(getattr(a, k)))
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(card=a.card, fold=a.fold, status="FAIL", lams=LAMS.tolist(), smooth=SMOOTH)
    t0 = time.time()
    try:
        sc = SC.load_scoring()
        files = heldout_files(a.fold)
        sub = SC.extract(Path(a.card), Path(a.work))
        mod = SC.import_submission(sub)
        os.chdir(sub)
        acc = []
        names, t0s = [], []
        for fn in files:
            x, ts = SC.windows_of(fn)
            xin = np.stack([x[s:s + SC.T_IN] for s in ts], 0)
            tgt = np.stack([x[s + SC.T_IN:s + SC.HOR] for s in ts], 0)
            P, L, H = [], [], []
            for i in range(0, len(ts), a.chunk):
                o = mod.predict(xin[i:i + a.chunk], None)
                P.append(np.asarray(o["prediction"], np.float32))
                L.append(np.asarray(o["lower"], np.float32)); H.append(np.asarray(o["upper"], np.float32))
            pred, lower, upper = np.concatenate(P), np.concatenate(L), np.concatenate(H)
            acc.append(stats(sc, xin, pred, tgt, lower, upper))
            names += [Path(fn).name] * len(ts); t0s += list(ts)
        save = dict(files=np.array(names), t0=np.array(t0s), lams=LAMS,
                    tke=np.concatenate([s["tke"] for s in acc]), rel0=np.concatenate([s["rel0"] for s in acc]),
                    n=np.concatenate([s["n"] for s in acc]))
        for k in SMOOTH:
            for key in ("rel", "mvpe", "S", "nin"):
                save[f"k{k}_{key}"] = np.concatenate([s["fam"][k][key] for s in acc])
        assert np.allclose(save["k1_rel"][:, 0], save["rel0"])
        np.savez_compressed(str(out), **save)
        meta.update(status="OK", n_windows=len(names))
    except Exception as e:  # noqa: BLE001
        meta.update(error="%s: %s" % (type(e).__name__, e), traceback=traceback.format_exc())
    meta["elapsed_s"] = time.time() - t0
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=1))
    print(a.fold, meta["status"], meta.get("n_windows"), "%.0fs" % meta["elapsed_s"], meta.get("error", ""))


if __name__ == "__main__":
    main()
