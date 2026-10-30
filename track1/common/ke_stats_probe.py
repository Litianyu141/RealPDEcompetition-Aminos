#!/usr/bin/env python3
"""Per-window second-order sufficient statistics of a card's held-out predictions (T1 or T2 streaming), so that ANY
per-pixel fluctuation-gain policy g(x) can be scored offline EXACTLY (vgain_head.subscores identity):

    ||p'-t||^2 = M + sum_{x,c} (g^2 A - 2 g B + C),  KE_p' = g^2 (A_u+A_v)/2T,  KE_t = (C_u+C_v)/2T
    A = sum_t pf^2, B = sum_t pf*tf, C = sum_t tf^2 (physical, per pixel & channel), M = T*sum (pbar-tbar)^2, tn2 = ||t||^2

plus the input window's own per-pixel KE (KE_in), the baseline per-window rel/tke/mvpe, the SPS interval numerator S and
count n (band untouched by gain policies), and file/t0 order so causal (T2) policies can walk each trajectory in order.
MVPE is invariant under gain policies (time-mean untouched).
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "common/local_benchmark/splitsel")); sys.path.insert(0, str(REPO / "common"))
import score_card as SC  # noqa: E402
from band_width_fivefold import heldout_files  # noqa: E402

STASH = []
XIN = {}


def stats(sc, pred, tgt, lower, upper, jgrid=False):
    base = SC.__dict__["_orig_pws"](sc, pred, tgt, lower, upper, jgrid=False)
    p2, t2 = pred[..., :2].astype(np.float64), tgt[..., :2].astype(np.float64)
    pb, tb = p2.mean(1), t2.mean(1)
    pf, tf = p2 - pb[:, None], t2 - tb[:, None]
    T = p2.shape[1]
    rec = dict(A=(pf * pf).sum(1).astype(np.float32), B=(pf * tf).sum(1).astype(np.float32), C=(tf * tf).sum(1).astype(np.float32),
               M=(T * ((pb - tb) ** 2).reshape(len(p2), -1).sum(1)), tn2=(t2 * t2).reshape(len(p2), -1).sum(1),
               rel=base["rel"], tke=base["tke"], mvpe=base["mvpe"], S=base["S"], n=base["n"])
    STASH.append(rec)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True); ap.add_argument("--fold", required=True); ap.add_argument("--track", type=int, required=True)
    ap.add_argument("--work", required=True); ap.add_argument("--out", required=True); ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    for k in ("card", "work", "out"):
        setattr(a, k, os.path.abspath(getattr(a, k)))
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(card=a.card, fold=a.fold, track=a.track, status="FAIL"); t0 = time.time()
    try:
        sc = SC.load_scoring(); files = heldout_files(a.fold)
        SC.__dict__["_orig_pws"] = SC.per_window_stats; SC.per_window_stats = stats
        sub = SC.extract(Path(a.card), Path(a.work))
        opts = argparse.Namespace(set=None, online=None, fluct_mult=1.0, band_mult=1.0, jgrid=False, _JS=[], _JC=[])
        if a.track == 1:
            rows, checks, extra = SC.run_t1(sub, sc, files, 64, opts)
        else:
            rows, checks, extra = SC.run_t2(sub, sc, files, "cuda", opts)
        kin = []
        for fn in files:                               # input KE per window, same order as rows
            x, ts = SC.windows_of(fn)
            xin = np.stack([x[s:s + SC.T_IN] for s in ts], 0)[..., :2].astype(np.float64)
            kin.append((0.5 * ((xin - xin.mean(1, keepdims=True)) ** 2).mean(1).sum(-1)).astype(np.float32))
        cat = {k: np.concatenate([s[k] for s in STASH]) for k in STASH[0]}
        cat["KEin"] = np.concatenate(kin)
        np.savez_compressed(str(out), files=np.array([r[0] for r in rows]), t0=np.array([r[1] for r in rows]), T=20, **cat)
        meta.update(status="OK", n_windows=len(rows), selfcheck=checks)
    except Exception as e:  # noqa: BLE001
        meta.update(error="%s: %s" % (type(e).__name__, e), traceback=traceback.format_exc())
    meta["elapsed_s"] = time.time() - t0
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=1))
    print(a.fold, a.track, meta["status"], meta.get("n_windows"), "%.0fs" % meta["elapsed_s"], meta.get("error", ""))


if __name__ == "__main__":
    main()
