#!/usr/bin/env python3
"""Per-window, per-pixel time-mean statistics of a card's held-out predictions, so that any per-pixel time-mean anchor
    pbar' = pbar + lam(x) * (xbar_in - pbar)          (u, v; fluctuations and band untouched)
can be scored offline EXACTLY for rel (via M = T * sum_x |pbar' - tbar|^2 and the fluctuation part F = ||p - pbar - (t -
tbar)||^2, unchanged) and MVPE (official probe points are time-means), with TKE unchanged.
Stores D = pbar - tbar, E = xbar_in - pbar and TB = tbar (N,H,W,2, physical), F, ||t||^2, baseline rel/tke/mvpe, the interval
numerator S and count n (band untouched), in trajectory order.

    python common/mean_stats_probe.py --card <fold card.zip> --fold re2 --track 2 --gpu 0 --work /tmp/msp --out <npz>
"""
from __future__ import annotations

import argparse, json, os, sys, time, traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "common/local_benchmark/splitsel")); sys.path.insert(0, str(REPO / "common"))
import score_card as SC  # noqa: E402
from band_width_fivefold import heldout_files  # noqa: E402

STASH: list = []
XBAR: list = []


def stats(sc, pred, tgt, lower, upper, jgrid=False):
    base = SC.__dict__["_orig_pws"](sc, pred, tgt, lower, upper, jgrid=False)
    xb = XBAR.pop(0)
    assert xb.shape[0] == pred.shape[0]
    p2, t2 = pred[..., :2].astype(np.float64), tgt[..., :2].astype(np.float64)
    pb, tb = p2.mean(1), t2.mean(1)
    F = (((p2 - pb[:, None]) - (t2 - tb[:, None])) ** 2).reshape(len(p2), -1).sum(1)
    STASH.append(dict(D=(pb - tb).astype(np.float32), E=(xb - pb).astype(np.float32), TB=tb.astype(np.float32), F=F, tn2=(t2 * t2).reshape(len(p2), -1).sum(1),
                      rel=base["rel"], tke=base["tke"], mvpe=base["mvpe"], S=base["S"], n=base["n"]))
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True); ap.add_argument("--fold", required=True)
    ap.add_argument("--track", type=int, required=True, choices=[1, 2])
    ap.add_argument("--work", required=True); ap.add_argument("--out", required=True); ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    for k in ("card", "work", "out"):
        setattr(a, k, os.path.abspath(getattr(a, k)))
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(card=a.card, fold=a.fold, track=a.track, status="FAIL"); t0 = time.time()
    try:
        sc = SC.load_scoring(); files = heldout_files(a.fold)
        for fn in files:
            x, ts = SC.windows_of(fn)
            XBAR.append(np.stack([x[s:s + SC.T_IN, ..., :2].astype(np.float64).mean(0) for s in ts], 0))
        SC.__dict__["_orig_pws"] = SC.per_window_stats; SC.per_window_stats = stats
        sub = SC.extract(Path(a.card), Path(a.work))
        opts = argparse.Namespace(set=None, online=None, fluct_mult=1.0, band_mult=1.0, jgrid=False, _JS=[], _JC=[])
        if a.track == 1:
            rows, checks, extra = SC.run_t1(sub, sc, files, 64, opts)
        else:
            rows, checks, extra = SC.run_t2(sub, sc, files, "cuda", opts)
        assert not XBAR
        cat = {k: np.concatenate([s[k] for s in STASH]) for k in STASH[0]}
        np.savez_compressed(str(out), files=np.array([r[0] for r in rows]), t0=np.array([r[1] for r in rows]), T=20, **cat)
        meta.update(status="OK", n_windows=len(rows))
    except Exception as e:  # noqa: BLE001
        meta.update(error="%s: %s" % (type(e).__name__, e), traceback=traceback.format_exc())
    meta["elapsed_s"] = time.time() - t0
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=1))
    print(a.fold, a.track, meta["status"], meta.get("n_windows"), "%.0fs" % meta["elapsed_s"], meta.get("error", ""))


if __name__ == "__main__":
    main()
