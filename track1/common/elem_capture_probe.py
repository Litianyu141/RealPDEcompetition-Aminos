#!/usr/bin/env python3
"""Held-out capture: per-element prediction / target / interval of a served fold card (real interface: T1 predict /
T2 streaming TTT), for offline analyses of per-element interval rules that the window-sum statistics of
band_width_fivefold.py cannot score (e.g. per-pixel scales or centres driven by the revealed residuals of earlier T2
windows).  Writes one npz per fold: files, t0, pred/tgt/lo/hi (u, v only, float32, physical), rel/tke/mvpe per window,
and verifies that the stored arrays reproduce the official per-window SPS numerator exactly.

    python common/elem_capture_probe.py --card <fold card.zip> --fold re2 --track 2 --gpu 0 --work /tmp/ec --out /tmp/ec/re2.npz

--members (T1 only, 2026-09-26 V200 T1 interval riders): after every served predict() call, the card's two blend members are
run EAGERLY on the same chunk (mod._eager_model.members[i] on ((x - _MEAN_IN) / _STD_IN).to(_MDTYPE), exactly the card's own
member input), and their u,v outputs are stored in physical units (y.float() * _STD_T + _MEAN_T) as y1 / y2 (float32).  The
served arrays and every other key are unchanged; without the flag the file is written exactly as before.  Eager outputs may
differ from the graphed replays by rounding (different batch shapes), so they are an offline-screen input only; the real
fold cards re-prove any rider.
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
MSTASH: list = []


def _wrap_members(mod):
    """--members: stash the two blend members' eager outputs (physical u,v) after each served predict() call."""
    import torch
    em = mod._eager_model
    assert len(em.members) == 2, "--members needs the 2-member blend"
    orig = mod.predict

    def predict(x, metadata=None):
        out = orig(x, metadata)
        with torch.no_grad():
            xt = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=mod._MEAN_IN.device)
            xn = ((xt - mod._MEAN_IN) / mod._STD_IN).to(mod._MDTYPE)
            ys = [(em.members[i](xn).float() * mod._STD_T + mod._MEAN_T)[..., :2].cpu().numpy().astype(np.float32) for i in (0, 1)]
        MSTASH.append(dict(y1=ys[0], y2=ys[1]))
        return out
    mod.predict = predict
    return mod


def stats(sc, pred, tgt, lower, upper, jgrid=False):
    base = SC.__dict__["_orig_pws"](sc, pred, tgt, lower, upper, jgrid=False)
    STASH.append(dict(pred=pred[..., :2].astype(np.float32), tgt=tgt[..., :2].astype(np.float32),
                      lo=np.asarray(lower, np.float32)[..., :2], hi=np.asarray(upper, np.float32)[..., :2], base=base))
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True); ap.add_argument("--fold", required=True)
    ap.add_argument("--track", type=int, required=True, choices=[1, 2])
    ap.add_argument("--work", required=True); ap.add_argument("--out", required=True); ap.add_argument("--gpu", default="0")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--members", action="store_true", help="T1: also store the two blend members' eager outputs y1/y2 (physical u,v)")
    a = ap.parse_args()
    assert not (a.members and a.track != 1), "--members is T1 only"
    for k in ("card", "work", "out"):
        setattr(a, k, os.path.abspath(getattr(a, k)))
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(card=a.card, fold=a.fold, track=a.track, status="FAIL"); t_start = time.time()
    try:
        meta["card_sha256"] = SC.archive_sha256(a.card)
        sc = SC.load_scoring(); files = heldout_files(a.fold)
        SC.__dict__["_orig_pws"] = SC.per_window_stats; SC.per_window_stats = stats
        if a.members:
            _imp = SC.import_submission
            SC.import_submission = lambda sub_dir: _wrap_members(_imp(sub_dir))
        sub = SC.extract(Path(a.card), Path(a.work))
        opts = argparse.Namespace(set=None, online=None, fluct_mult=1.0, band_mult=1.0, jgrid=False, _JS=[], _JC=[])
        if a.track == 1:
            rows, checks, extra = SC.run_t1(sub, sc, files, a.chunk, opts)
        else:
            rows, checks, extra = SC.run_t2(sub, sc, files, "cuda", opts)
        assert len(STASH) == len(files)
        # exactness: the stored arrays reproduce the official per-window sps numerator (inside & scored) * exp(-nil)
        for st in STASH:
            t, lo, hi = st["tgt"], st["lo"], st["hi"]
            e = np.where((t >= lo) & (t <= hi) & (t != 0.0), np.exp(-(hi - lo) / sc.SIGMA_GLOBAL), 0.0).sum(axis=(1, 2, 3, 4))
            assert np.allclose(e, st["base"]["S"], rtol=1e-6, atol=1e-6), "stored arrays must reproduce the served S"
        cat = lambda k: np.concatenate([st[k] for st in STASH])
        extra = {}
        if a.members:
            extra = dict(y1=np.concatenate([m["y1"] for m in MSTASH]), y2=np.concatenate([m["y2"] for m in MSTASH]))
            assert extra["y1"].shape == cat("pred").shape, (extra["y1"].shape, cat("pred").shape)
        np.savez(str(out), files=np.array([r[0] for r in rows]), t0=np.array([r[1] for r in rows]),
                 pred=cat("pred"), tgt=cat("tgt"), lo=cat("lo"), hi=cat("hi"),
                 rel=np.concatenate([st["base"]["rel"] for st in STASH]), tke=np.concatenate([st["base"]["tke"] for st in STASH]),
                 mvpe=np.concatenate([st["base"]["mvpe"] for st in STASH]), S=np.concatenate([st["base"]["S"] for st in STASH]),
                 n=np.concatenate([st["base"]["n"] for st in STASH]), sigma=np.float64(sc.SIGMA_GLOBAL), **extra)
        meta.update(status="OK", n_windows=len(rows), **({"members": True} if a.members else {}))
    except Exception as e:  # noqa: BLE001
        meta.update(error="%s: %s" % (type(e).__name__, e), traceback=traceback.format_exc())
    meta["elapsed_s"] = time.time() - t_start
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=1, default=str))
    print(a.fold, meta["status"], meta.get("n_windows"), "%.0fs" % meta["elapsed_s"], meta.get("error", ""))


if __name__ == "__main__":
    main()
