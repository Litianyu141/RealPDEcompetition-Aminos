#!/usr/bin/env python3
"""SPLITSEL worker: score ONE historical card zip, through its own shipped interface, on every
window of the 81 legal real files, and write PER-WINDOW sufficient statistics.

Why per-window: every local split (val9, rex_f2, the k-folds, all81, per-Re groups ...) is a subset
of these windows, and all five official accuracy legs can be re-aggregated EXACTLY from the
per-window numbers below, so one pass per card prices every split at once:

  rel_w, tke_w, mvpe_w      raw per-sample errors (scoring.py rel_l2_per_sample / tke_rel_l2_per_sample /
                            mvpe_rel_l2_per_sample); a split's leg = score_error(mean over its windows)
  S_w                       sum over scored elements of exp(-nil) * inside   (interval term of SPS)
  n_w, nin_w                scored element count, and scored & inside count (coverage)
  sps(split) = sum_w [0.5(1-f(rel_w)) + 0.3(1-f(tke_w)) + 0.2(1-f(mvpe_w))] * S_w / sum_w n_w,  f(e)=e/(0.5+e)

That identity follows from scoring.aggregate_sps line by line (branch() sums over all scored
elements then divides by the global n_scored; the per-window factor (1-pm) is constant inside a
window), and is self-checked here against the kit's own aggregate_sps on a random subset.

Track 1: the card's predict(input_physical (N,20,32,64,3), metadata=None) -> {prediction, lower, upper}
         (or a bare array), one call per trajectory, exactly the platform's teacher-forced regime.
Track 2: the kit's streaming protocol (local_eval.py of kit v6): sorted (file, time_id), reset at
         trajectory boundaries, prev_target passing, kit Normalizer with SEPARATE input/target affines,
         batch 1, bounds denormalised with the target affine.

Nothing here trains, uploads, or touches LIVE_CARD; the card is extracted into a scratch dir.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

def default_repo(env=os.environ):
    """This repository's root (the file lives at common/local_benchmark/splitsel/), unless REALPDE_REPO is set."""
    return Path(env["REALPDE_REPO"]) if env.get("REALPDE_REPO") else Path(__file__).resolve().parents[3]


def resolve_real_dir(repo):
    """The 82 released real files, in this repo's data layout."""
    cands = [repo / "data/RealPDE_competition_dataset/gdrive_raw/extracted/train_real/train_real"]
    for c in cands:
        if c.is_dir():
            return c
    raise FileNotFoundError("no released real-data directory under %s; looked for %s" % (repo, [str(c) for c in cands]))


REPO = default_repo()
REAL_DIR = None        # resolved lazily in real_files(), so importing this module needs no data
T1_KIT = REPO / "Track1/realpde_t1_starting_kit_v9"
T2_KIT = REPO / "Track2/realpde_t2_starting_kit_v6"
STATS = REPO / "Track2/realpde_t2_starting_kit_v4/example_data/mean_std_real.pt"
EXCLUDED = {"7575_0.h5"}          # organizer notice 2026-08-13
T_IN, T_OUT, HOR, STRIDE, SUB = 20, 20, 40, 20, 2


def load_scoring():
    sys.path.insert(0, str(T1_KIT))
    import scoring  # the kit's own reference implementation
    return scoring


def real_files():
    fs = sorted(glob.glob(str((REAL_DIR or resolve_real_dir(REPO)) / "*.h5")))
    fs = [f for f in fs if Path(f).name not in EXCLUDED]
    assert len(fs) == 81, "expected 81 legal files, found %d" % len(fs)
    return fs


def windows_of(fn):
    """Physical (T,32,64,3) with p=0, and the list of t0 for 20->20 windows at stride 20."""
    import h5py
    with h5py.File(fn, "r") as f:
        u = np.asarray(f["u"][:, ::SUB, ::SUB], np.float32)
        v = np.asarray(f["v"][:, ::SUB, ::SUB], np.float32)
    x = np.stack([u, v, np.zeros_like(u)], -1)
    T = x.shape[0]
    t0s = list(range(0, T - HOR + 1, STRIDE))
    return x, t0s


JGRID = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0])


def scale_bounds(pred, lower, upper, m):
    """Global band multiplier: offsets around the prediction scaled by m (asymmetric offsets preserved)."""
    if lower is None or m == 1.0:
        return lower, upper
    return pred + m * (lower - pred), pred + m * (upper - pred)


def scale_fluct(pred, m):
    """Global fluctuation-gain multiplier: p -> mean_t(p) + m*(p - mean_t(p)) on the u,v channels (the card's own
    gain-head operation with a scalar), bounds left where the card put them (centred on the pre-head field)."""
    if m == 1.0:
        return pred
    pm = pred.mean(axis=1, keepdims=True)
    out = pm + m * (pred - pm)
    out[..., 2] = pred[..., 2]
    return out.astype(np.float32)


def per_window_stats(sc, pred, tgt, lower, upper, jgrid=False):
    """pred/tgt/lower/upper: (N,20,32,64,3) physical.  Returns dict of per-window arrays.

    jgrid=True additionally returns J[w, ia, ib] = sum over scored elements of exp(-a*nil) * 1{inside at
    width factor b}, for a, b in JGRID: the width-response table from which sps at ANY global band
    multiplier m is J[:, m, m], and the response under an online residual scale kappa is J[:, m, m/kappa].
    """
    c = sc.measured_channels(tgt)
    rel = sc.rel_l2_per_sample(pred, tgt, c)
    tke = sc.tke_rel_l2_per_sample(pred, tgt, c)
    mv = sc.mvpe_rel_l2_per_sample(pred, tgt)
    p = pred[..., :c]
    t = tgt[..., :c]
    if lower is None:
        interval = 0.1 * np.abs(p)
        lo, hi = p - interval / 2.0, p + interval / 2.0
    else:
        lo = np.asarray(lower, np.float32)[..., :c]
        hi = np.asarray(upper, np.float32)[..., :c]
        assert lo.shape == p.shape and hi.shape == p.shape
        assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and not np.any(lo > hi)
    inside = (t >= lo) & (t <= hi)
    nil = (hi - lo) / sc.SIGMA_GLOBAL
    scored = t != 0.0
    elem = np.where(inside & scored, np.exp(-nil), 0.0)
    ax = tuple(range(1, p.ndim))
    S = elem.sum(axis=ax, dtype=np.float64)
    n = scored.sum(axis=ax)
    nin = (inside & scored).sum(axis=ax)
    out = dict(rel=rel.astype(np.float64), tke=tke.astype(np.float64), mvpe=mv.astype(np.float64),
               S=S, n=n.astype(np.int64), nin=nin.astype(np.int64), c=int(c))
    if jgrid:
        J = np.zeros((p.shape[0], len(JGRID), len(JGRID)), np.float64)
        Jcov = np.zeros((p.shape[0], len(JGRID)), np.float64)
        wlo, whi = (p - lo).astype(np.float32), (hi - p).astype(np.float32)
        for ib, b in enumerate(JGRID):
            ins_b = (t >= p - b * wlo) & (t <= p + b * whi) & scored
            Jcov[:, ib] = ins_b.sum(axis=ax)
            for ia, a in enumerate(JGRID):
                J[:, ia, ib] = np.where(ins_b, np.exp(-a * nil), 0.0).sum(axis=ax, dtype=np.float64)
        out["J"], out["Jcov"] = J, Jcov
    return out


def selfcheck(sc, pred, tgt, lower, upper, st):
    """Recompute SPS from the per-window stats and compare with the kit's aggregate_sps."""
    c = st["c"]
    f = lambda e: e / (0.5 + e)
    w = 0.5 * (1 - f(st["rel"])) + 0.3 * (1 - f(st["tke"])) + 0.2 * (1 - f(st["mvpe"]))
    mine = float((w * st["S"]).sum() / st["n"].sum())
    ref, cov = sc.aggregate_sps(pred, tgt, c, lower=lower, upper=upper)
    covm = float(st["nin"].sum() / st["n"].sum())
    return dict(sps_mine=mine, sps_kit=float(ref), abs_diff=abs(mine - ref),
                cov_mine=covm, cov_kit=float(cov))


def archive_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def extract(zip_path: Path, work: Path) -> Path:
    """Bind extraction to actual archive bytes; retain old scratch directories."""
    digest = archive_sha256(zip_path)
    d = work / (zip_path.stem + "__" + digest[:16])
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or "submission.py" not in names:
            raise ValueError("archive requires unique members and root submission.py")
        if any(PurePosixPath(n).is_absolute() or ".." in PurePosixPath(n).parts or "\\" in n for n in names):
            raise ValueError("unsafe archive member path")
        if not d.exists():
            d.mkdir(parents=True)
            z.extractall(d)
        for entry in z.infolist():
            if entry.is_dir():
                continue
            extracted = d / entry.filename
            if (not extracted.is_file() or not extracted.resolve().is_relative_to(d.resolve())
                    or archive_sha256(extracted) != hashlib.sha256(z.read(entry.filename)).hexdigest()):
                raise ValueError("extracted archive member missing or changed: " + entry.filename)
    return d


def import_submission(sub_dir: Path):
    sys.path.insert(0, str(sub_dir))
    spec = importlib.util.spec_from_file_location("participant_submission", sub_dir / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def apply_overrides(mod, opts):
    """--set NAME=JSON sets a module-level constant (e.g. _SGATE=1.0) BEFORE any model/graph is built;
    --online KEY=JSON updates the module's ONLINE dict (T2).  Both are recorded in the result."""
    for kv in opts.set or []:
        k, v = kv.split("=", 1)
        assert hasattr(mod, k), "module has no attribute %s" % k
        setattr(mod, k, json.loads(v))
    for kv in opts.online or []:
        k, v = kv.split("=", 1)
        d = getattr(mod, "ONLINE")
        assert k in d, "ONLINE has no key %s (has %s)" % (k, sorted(d))
        d[k] = json.loads(v)


# ----------------------------------------------------------------------------- Track 1
def run_t1(sub_dir, sc, files, chunk, opts):
    mod = import_submission(sub_dir)
    apply_overrides(mod, opts)
    os.chdir(sub_dir)
    rows, checks, times = [], [], []
    JS, JC = opts._JS, opts._JC
    for fn in files:
        x, t0s = windows_of(fn)
        xin = np.stack([x[t0:t0 + T_IN] for t0 in t0s], 0)
        tgt = np.stack([x[t0 + T_IN:t0 + HOR] for t0 in t0s], 0)
        preds, los, his = [], [], []
        for i in range(0, len(t0s), chunk):
            t = time.perf_counter()
            out = mod.predict(xin[i:i + chunk], None)
            times.append(time.perf_counter() - t)
            if isinstance(out, dict):
                preds.append(np.asarray(out["prediction"], np.float32))
                if "lower" in out and out["lower"] is not None:
                    los.append(np.asarray(out["lower"], np.float32)); his.append(np.asarray(out["upper"], np.float32))
            else:
                preds.append(np.asarray(out, np.float32))
        pred = np.concatenate(preds, 0)
        lower = np.concatenate(los, 0) if los else None
        upper = np.concatenate(his, 0) if his else None
        assert pred.shape == tgt.shape, (pred.shape, tgt.shape)
        pred = scale_fluct(pred, opts.fluct_mult)
        lower, upper = scale_bounds(pred, lower, upper, opts.band_mult)
        st = per_window_stats(sc, pred, tgt, lower, upper, jgrid=opts.jgrid)
        if len(checks) < 3:
            checks.append(dict(file=Path(fn).name, **selfcheck(sc, pred, tgt, lower, upper, st)))
        for k, t0 in enumerate(t0s):
            rows.append([Path(fn).name, int(t0), st["rel"][k], st["tke"][k], st["mvpe"][k],
                         st["S"][k], int(st["n"][k]), int(st["nin"][k])])
        if opts.jgrid:
            JS.append(st["J"]); JC.append(st["Jcov"])
    return rows, checks, dict(has_bounds=lower is not None, mean_call_s=float(np.mean(times)))


# ----------------------------------------------------------------------------- Track 2
def run_t2(sub_dir, sc, files, device, opts):
    import torch
    sys.path.insert(0, str(T2_KIT))
    from local_eval import Normalizer
    mod = import_submission(sub_dir)
    apply_overrides(mod, opts)
    os.chdir(sub_dir)
    JS, JC = opts._JS, opts._JC
    model = mod.get_ttt_model(str(sub_dir), device)
    norm = Normalizer(STATS)
    rows, checks = [], []
    times = []
    any_bounds = None
    for fn in files:                      # sorted filenames = the platform's stream order
        x, t0s = windows_of(fn)
        model.reset_ttt_state()
        prev = None
        preds, tgts, los, his = [], [], [], []
        for t0 in t0s:
            inp = torch.from_numpy(x[t0:t0 + T_IN]).unsqueeze(0)
            tgt = torch.from_numpy(x[t0 + T_IN:t0 + HOR]).unsqueeze(0)
            inp_n, tgt_n = norm.preprocess(inp, tgt)
            t = time.perf_counter()
            pred_n, info = model.ttt_step(inp_n, prev)
            times.append(time.perf_counter() - t)
            prev = tgt_n.detach()
            pred_n = torch.as_tensor(pred_n).detach().cpu().float()
            preds.append(norm.postprocess_pred(pred_n).squeeze(0).numpy().astype(np.float32))
            tgts.append(tgt.squeeze(0).numpy().astype(np.float32))
            hb = isinstance(info, dict) and info.get("lower") is not None
            if any_bounds is None:
                any_bounds = hb
            assert hb == any_bounds, "bounds must be supplied on every step or none"
            if hb:
                lo = torch.as_tensor(info["lower"]).detach().cpu().float()
                hi = torch.as_tensor(info["upper"]).detach().cpu().float()
                los.append(norm.postprocess_pred(lo).squeeze(0).numpy().astype(np.float32))
                his.append(norm.postprocess_pred(hi).squeeze(0).numpy().astype(np.float32))
        pred = np.stack(preds, 0); tgt = np.stack(tgts, 0)
        lower = np.stack(los, 0) if los else None
        upper = np.stack(his, 0) if his else None
        pred = scale_fluct(pred, opts.fluct_mult)
        lower, upper = scale_bounds(pred, lower, upper, opts.band_mult)
        st = per_window_stats(sc, pred, tgt, lower, upper, jgrid=opts.jgrid)
        if len(checks) < 3:
            checks.append(dict(file=Path(fn).name, **selfcheck(sc, pred, tgt, lower, upper, st)))
        for k, t0 in enumerate(t0s):
            rows.append([Path(fn).name, int(t0), st["rel"][k], st["tke"][k], st["mvpe"][k],
                         st["S"][k], int(st["n"][k]), int(st["nin"][k])])
        if opts.jgrid:
            JS.append(st["J"]); JC.append(st["Jcov"])
    return rows, checks, dict(has_bounds=bool(any_bounds), mean_step_s=float(np.mean(times)))


def absolutize_cli_paths(a):
    """The card is imported after os.chdir(<extracted card>), so relative --zip/--work/--out must be fixed first."""
    for k in ("zip", "work", "out"):
        setattr(a, k, os.path.abspath(getattr(a, k)))
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--track", type=int, required=True, choices=[1, 2])
    ap.add_argument("--work", required=True, help="scratch dir for extraction")
    ap.add_argument("--out", required=True, help="output json path")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--band-mult", type=float, default=1.0, help="post-hoc global band multiplier (probe)")
    ap.add_argument("--fluct-mult", type=float, default=1.0, help="post-hoc global fluctuation-gain multiplier (probe)")
    ap.add_argument("--jgrid", action="store_true", help="also write the width-response table J to <out>.npz")
    ap.add_argument("--set", action="append", help="module constant override NAME=JSON, e.g. _SGATE=1.0")
    ap.add_argument("--online", action="append", help="T2 ONLINE dict override KEY=JSON, e.g. band_beta=0.2")
    a = absolutize_cli_paths(ap.parse_args())
    a._JS, a._JC = [], []
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    zp = Path(a.zip)
    res = dict(card=zp.stem, track=a.track, zip=str(zp), status="FAIL",
               started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    t_all = time.time()
    try:
        res["zip_sha256"] = archive_sha256(zp)
        sc = load_scoring()
        files = real_files()
        sub = extract(zp, Path(a.work))
        if a.track == 1:
            rows, checks, extra = run_t1(sub, sc, files, a.chunk, a)
        else:
            rows, checks, extra = run_t2(sub, sc, files, "cuda", a)
        if archive_sha256(zp) != res["zip_sha256"]:
            raise RuntimeError("submission ZIP changed during scoring")
        res.update(band_mult=a.band_mult, fluct_mult=a.fluct_mult, overrides=dict(set=a.set or [], online=a.online or []))
        if a.jgrid:
            np.savez_compressed(str(out) + ".npz", J=np.concatenate(a._JS, 0), Jcov=np.concatenate(a._JC, 0),
                                grid=JGRID, files=np.array([r[0] for r in rows]), t0=np.array([r[1] for r in rows]))
        # summary over all 81 files (all81 windows), as a sanity anchor
        R = np.array([[r[2], r[3], r[4], r[5], r[6], r[7]] for r in rows], dtype=np.float64)
        f = lambda e: e / (0.5 + e)
        w = 0.5 * (1 - f(R[:, 0])) + 0.3 * (1 - f(R[:, 1])) + 0.2 * (1 - f(R[:, 2]))
        res.update(status="OK", n_windows=len(rows), n_files=len(files), selfcheck=checks, **extra,
                   all81=dict(rel=sc.score_error(R[:, 0].mean()), tke=sc.score_error(R[:, 1].mean()),
                              mvpe=sc.score_error(R[:, 2].mean()),
                              sps=100.0 * float((w * R[:, 3]).sum() / R[:, 4].sum()),
                              coverage=float(R[:, 5].sum() / R[:, 4].sum())),
                   columns=["file", "t0", "rel", "tke", "mvpe", "S", "n", "nin"], rows=rows)
        # float32 elementwise sums vs the kit's float64 where-sum: agreement is ~1e-9, the test is 1e-6
        assert all(ch["abs_diff"] < 1e-6 for ch in checks), "SPS identity self-check failed: %r" % checks
    except Exception as e:  # record the failure, never hide it
        res.update(status="FAIL", error="%s: %s" % (type(e).__name__, e), traceback=traceback.format_exc())
    res["elapsed_s"] = time.time() - t_all
    out.write_text(json.dumps(res))
    print(res["card"], res["status"], res.get("all81"), "%.0fs" % res["elapsed_s"], res.get("error", ""))


if __name__ == "__main__":
    main()
