#!/usr/bin/env python3
"""STAGE B -- train the residual-scale head and score it out of fold.

  python sbB.py --fold k          nested-OOF arm: train on the 5 other Reynolds groups, fit the
                                  width table on them too, evaluate on group k.
  python sbB.py --fold ship       the shipping head: train on all 72 non-benchmark files.

TARGET = the GEOMETRIC MEAN of |r| over two independent OOF forecaster fleets (RSa_kf and
RSa_kfs1: same recipe, same split, different seed).  Averaging log|r| over M models cancels the
independent realisation term while leaving the shared, input-predictable term intact, so the
pooled target's MINIMISER is the transferable component.  That is not a regulariser, it is a change
of what the loss is trying to fit -- measured basis: a PERFECT oracle for one model retains only
28.5% on another, and the same-recipe/different-seed pair retains 29.0%, so ~71% of the residual
field is one weight vector's realisation.  Pooling two models lifts the transferable ceiling from
57.64 to 58.14 sps; a third adds nothing (58.12).
"""
import os, sys, json, time, math, argparse
import os as _os
from pathlib import Path
import numpy as np, torch

# Derived from this file's own location (2026-08-12): the absolute hardcode silently
# escaped a fresh clone back to the production tree.  NB: run these tools from their
# in-repo location (the chains do); a /tmp copy would derive the wrong root.
REPO = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO / "common"))
from fivefold_heads import HeadPlan, verify_cache_metadata
KIT = REPO / "Track1/realpde_t1_starting_kit_v9"
T1U = REPO / "Track1/U-net"
import os as _os_c
# Cache directory.  Default is the value hardcoded here until 2026-08-09, so the shipped
# path is unchanged; override with SBAND_CACHE to fit a head for a DIFFERENT fleet without
# overwriting the cache the shipped head was fitted from.
CA = Path(_os_c.environ.get("SBAND_CACHE", str(Path(__file__).resolve().parent / "sband_cache")))
sys.path.insert(0, str(KIT)); import scoring
sys.path.insert(0, str(T1U))
from sband_head import SBandHead, sband_features, N_FEAT, N_FEAT_V2

SIG = float(scoring.SIGMA_GLOBAL)
DEV = "cuda"
H, W, LT, C2 = 32, 64, 20, 2
NOUT = LT * C2

ap = argparse.ArgumentParser()
ap.add_argument("--fold", required=True)
ap.add_argument("--feat", default="v1",
                help="v1 = the eleven shipped planes; v2 = + nine unpredictability/shear planes")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--width", type=int, default=64)
ap.add_argument("--layers", type=int, default=5)
ap.add_argument("--q", type=int, default=32)
ap.add_argument("--tag", default="")
ap.add_argument("--cache", default=str(CA))
ap.add_argument("--out-dir", default="", help="head/report output directory, defaults to cache")
ap.add_argument("--strict-plan", default="")
ap.add_argument("--quantile", type=float, default=0.5,
                help="target quantile of log|r|. 0.5 keeps plain L1 (bit-exact control). The scorer trades exp(-nil) against a HARD coverage indicator, so the median is not the optimal ranking target -- see the 64-bin-beats-exact-|r| result.")
ap.add_argument("--coord", type=int, default=0,
                help="2 = append normalised absolute-position planes.  The head is Conv2d-only\n                      and therefore translation-EQUIVARIANT, so without this it cannot express\n                      'cell (h,w) is always harder' -- and in this rig the airfoil, wake and FOV\n                      boundary sit at FIXED pixels.")
a = ap.parse_args()
CA = Path(a.cache)
OUT = Path(a.out_dir) if a.out_dir else CA
OUT.mkdir(parents=True, exist_ok=True)
TAG = a.tag or (f"f{a.fold}" if a.fold != "ship" else f"ship_s{a.seed}")

import os as _os


def _sband_fleet(_var):
    """Return $_var or REFUSE.  No default: the old RSa default mis-stamped every head."""
    _v = _os.environ.get(_var, "").strip()
    if not _v:
        raise SystemExit(
            "REFUSING: " + _var + " is unset.\n"
            "  Export BOTH SBAND_FLEET_A and SBAND_FLEET_B with the fold-checkpoint stems whose\n"
            "  residuals this cache/head is built from (comma-separated for a blend), e.g.\n"
            "    export SBAND_FLEET_A=unet_v4_CW128_kf_base128,unet_v4_WSW_kf_base128\n"
            "    export SBAND_FLEET_B=unet_v4_CW128s1_kf_base128,unet_v4_WSWs1_kf_base128\n"
            "  There is no safe default.  Until 2026-08-19 these defaulted to the RSa strings, so\n"
            "  the shipped t1_v71_xcb head saved an RSa fit_kind over a verified CW128+WSW cache.")
    return _v
_SBAND_FLEET_A = _sband_fleet("SBAND_FLEET_A")
_SBAND_FLEET_B = _sband_fleet("SBAND_FLEET_B")
plan = HeadPlan(a.strict_plan) if a.strict_plan else None
if plan:
    if a.fold != "ship":
        raise ValueError("strict outer-fold band requires --fold ship")
    if (OUT / f"sband_{TAG}.pth").exists():
        raise ValueError("strict band head output already exists; use a fresh tag")
    strict_meta = verify_cache_metadata(CA, plan, _SBAND_FLEET_A.split(',') + _SBAND_FLEET_B.split(','),
                                       _SBAND_FLEET_A.split(','), _SBAND_FLEET_B.split(','))
torch.manual_seed(a.seed); np.random.seed(a.seed)

Xp = np.load(CA / "Xp.npy", mmap_mode="r")             # (N,20,H,W,2) PHYSICAL input
Rkf = np.load(CA / "Rkf.npy", mmap_mode="r")           # (N,20,H,W,2) PHYSICAL signed residual
Rks = np.load(CA / "Rks.npy", mmap_mode="r")
SC = np.load(CA / "SC.npy", mmap_mode="r")             # (N,20,H,W,2) bool scored mask
AF = np.load(CA / "afac.npy")                          # (N,2) per-sample accuracy factor
grp = np.load(CA / "grp.npy")
N = Xp.shape[0]
# --- UNIT FINGERPRINT (physical, not normalised).  A normalised cache would land nowhere near.
assert 0.05 < float(np.mean(~SC[:64])) < 0.30, "scored-mask fingerprint failed"
if plan:
    assert np.isfinite(Rkf[:64]).all() and float(np.abs(Rkf[:64]).mean()) > 0, "invalid residual cache"
else:
    assert 0.0015 < float(np.abs(Rkf[:64]).mean()) < 0.010, "|r| fingerprint failed"

if a.fold == "ship":
    tr = np.arange(N); ev = np.array([], np.int64)
else:
    k = int(a.fold); tr = np.flatnonzero(grp != k); ev = np.flatnonzero(grp == k)
print(f"# [{TAG}] train {tr.size} eval {ev.size} windows", flush=True)

EPSC = np.array([np.quantile(np.abs(Rkf[::7, ..., c]), 0.01) for c in range(C2)], np.float32)
print(f"# eps per channel {EPSC}", flush=True)
EPS_T = torch.as_tensor(EPSC).to(DEV).reshape(1, 1, 1, 1, C2)


def _b(idx, arr):
    return torch.from_numpy(np.asarray(arr[idx])).to(DEV)


NF = (N_FEAT_V2 if a.feat == "v2" else N_FEAT) + int(a.coord)
net = SBandHead(NF, a.width, a.layers, NOUT).to(DEV)
with torch.no_grad():
    fs = sband_features(_b(np.sort(np.random.default_rng(0).choice(tr, min(96, tr.size), False)), Xp), a.coord, a.feat)
    net.set_feature_norm(fs.mean((0, 2, 3)), fs.std((0, 2, 3)))
    del fs

opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
bs = 16
steps = a.epochs * ((tr.size + bs - 1) // bs)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.2)
t0 = time.time()
for ep in range(a.epochs):
    perm = np.random.permutation(tr); tot = 0.0; nb = 0
    for i in range(0, tr.size, bs):
        idx = np.sort(perm[i:i + bs])
        xb = _b(idx, Xp)
        rk = _b(idx, Rkf).abs(); rs = _b(idx, Rks).abs()
        msk = _b(idx, SC)
        aw = torch.from_numpy(AF[idx].mean(1)).to(DEV).reshape(-1, 1, 1, 1, 1)
        # pooled log-target = log of the GEOMETRIC MEAN of the two fleets' |r|
        tgt = 0.5 * (torch.log(rk + EPS_T) + torch.log(rs + EPS_T))
        f = net(sband_features(xb, a.coord, a.feat)).reshape(-1, LT, C2, H, W).permute(0, 1, 3, 4, 2)  # (B,LT,H,W,C2)
        _e = f - tgt
        if abs(a.quantile - 0.5) < 1e-12:
            _pl = _e.abs()                                     # bit-exact: plain L1 -> conditional MEDIAN
        else:
            # pinball / quantile loss: minimiser is the a.quantile-th conditional quantile of log|r|
            _pl = torch.maximum(a.quantile * (-_e), (a.quantile - 1.0) * (-_e)) * 2.0
        loss = (_pl * aw)[msk].mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sch.step()
        tot += float(loss); nb += 1
    print(f"    [{TAG}] ep{ep} L1 {tot/nb:.4f} ({time.time()-t0:.0f}s)", flush=True)
net.eval()


def infer(idx):
    o = np.empty((idx.size, LT, H, W, C2), np.float32)
    with torch.no_grad():
        for i in range(0, idx.size, 32):
            j = min(i + 32, idx.size)
            f = net(sband_features(_b(idx[i:j], Xp), a.coord, a.feat))
            o[i:j] = f.reshape(-1, LT, C2, H, W).permute(0, 1, 3, 4, 2).cpu().numpy()
    return o


# ------------------------------------------------------------------ the empirical width table --
CENTERS = np.array([0.0, .35, .40, .45, .50, .55, .60, .65])   # 0.0 = symmetric; rest = quantiles


def fit_table(S, R, MS, AW, Q, nsub=60000, rng=0):
    """S,R,MS: (n,LT,H,W,C2).  AW: (n,).  -> edges (LT,C2,Q-1), tab (LT,C2,Q,2) PHYSICAL offsets."""
    g = np.random.default_rng(rng)
    edges = np.zeros((LT, C2, Q - 1), np.float64)
    tab = np.zeros((LT, C2, Q, 2), np.float64)
    awf = np.broadcast_to(AW.reshape(-1, 1, 1, 1), (AW.size, LT, H, W))
    for t in range(LT):
        for c in range(C2):
            m = MS[:, t, :, :, c]
            s = S[:, t, :, :, c][m].astype(np.float64)
            r = R[:, t, :, :, c][m].astype(np.float64)
            w = awf[:, t][m].astype(np.float64)
            e = np.quantile(s, np.linspace(0, 1, Q + 1)[1:-1])
            e = np.maximum.accumulate(e)
            edges[t, c] = e
            b = np.searchsorted(e, s, side="right")
            for k in range(Q):
                sel = np.flatnonzero(b == k)
                if sel.size < 8:
                    tab[t, c, k] = (-0.01, 0.01); continue
                if sel.size > nsub:
                    sel = g.choice(sel, nsub, replace=False)
                rr = r[sel]; ww = w[sel]
                qs = np.quantile(rr, CENTERS[1:])
                best = (-1.0, -0.01, 0.01)
                for ci, cc in enumerate([0.0] + list(qs)):
                    d = np.abs(rr - cc); o = np.argsort(d)
                    ds = d[o]; cw = np.cumsum(ww[o])
                    val = np.exp(-2.0 * ds / SIG) * cw
                    q = int(np.argmax(val))
                    if val[q] > best[0]:
                        best = (val[q], cc - ds[q], cc + ds[q])
                tab[t, c, k] = (best[1], best[2])
    return edges, tab


def apply_and_score(S, R, MS, AW, edges, tab):
    """Fast element-wise mirror of the official aggregate; validated to <1e-6 in stage D."""
    num = 0.0; nsc = 0
    for t in range(LT):
        for c in range(C2):
            m = MS[:, t, :, :, c]
            s = S[:, t, :, :, c][m].astype(np.float64)
            r = R[:, t, :, :, c][m].astype(np.float64)
            w = np.broadcast_to(AW.reshape(-1, 1, 1), (AW.size, H, W))[m].astype(np.float64)
            b = np.searchsorted(edges[t, c], s, side="right")
            lo = tab[t, c, b, 0]; hi = tab[t, c, b, 1]
            ins = (r >= lo) & (r <= hi)
            num += float(np.sum(w * np.exp(-(hi - lo) / SIG) * ins))
            nsc += int(m.sum())
    return 100.0 * num / nsc, nsc


RES = {"tag": TAG, "coord": int(a.coord), "fold": a.fold, "seed": a.seed, "n_train": int(tr.size), "n_eval": int(ev.size),
       "width": a.width, "layers": a.layers, "q": a.q, "epochs": a.epochs,
       "params": int(sum(p.numel() for p in net.parameters()))}

if ev.size:
    Str, Sev = infer(tr), infer(ev)
    Rtr = np.asarray(Rkf[tr]); Rev = np.asarray(Rkf[ev])
    Mtr = np.asarray(SC[tr]);  Mev = np.asarray(SC[ev])
    Atr = AF[tr].mean(1);      Aev = AF[ev].mean(1)
    Itr = np.asarray(Xp[tr]).std(1, ddof=0)[:, None].repeat(LT, 1)   # incumbent observable
    Iev = np.asarray(Xp[ev]).std(1, ddof=0)[:, None].repeat(LT, 1)
    for nm, St, Se in (("learned", Str, Sev), ("incumbent_instd", Itr, Iev)):
        e_, t_ = fit_table(St, Rtr, Mtr, Atr, a.q)
        sp, nsc = apply_and_score(Se, Rev, Mev, Aev, e_, t_)
        RES[nm] = round(sp, 4)
        print(f"# [{TAG}] {nm:16s} OOS sps {sp:8.3f}  (nsc {nsc})", flush=True)
    # a single global width, fitted on train -- the model-free floor
    e1, t1 = fit_table(np.zeros_like(Str), Rtr, Mtr, Atr, 1)
    sp, _ = apply_and_score(np.zeros_like(Sev), Rev, Mev, Aev, e1, t1)
    RES["const"] = round(sp, 4)
    # ORACLE ceiling on this group, same table machinery (fit on the EVAL group's own |r|)
    e2, t2 = fit_table(np.abs(Rev), Rev, Mev, Aev, a.q)
    sp, _ = apply_and_score(np.abs(Rev), Rev, Mev, Aev, e2, t2)
    RES["oracle"] = round(sp, 4)
    sub = np.random.default_rng(0).choice(int(Mev.sum()), min(400000, int(Mev.sum())), replace=False)
    sv = Sev[Mev][sub]; rv = np.abs(Rev)[Mev][sub]
    rk = lambda x: np.argsort(np.argsort(x)).astype(np.float64)
    RES["spearman"] = round(float(np.corrcoef(rk(sv), rk(rv))[0, 1]), 4)
    print(f"# [{TAG}] const {RES['const']:.3f} | incumbent {RES['incumbent_instd']:.3f} | "
          f"learned {RES['learned']:.3f} | oracle {RES['oracle']:.3f} | rho {RES['spearman']:.3f}",
          flush=True)

torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
            "n_feat": NF, "n_coord": int(a.coord), "feat": a.feat, "width": a.width, "n_layers": a.layers, "n_out": NOUT,
            # 2026-08-16: this string was HARD-CODED, so it named the RSa fleets no matter which
            # fleets were actually cached -- it carried NO provenance, and the claim that a
            # given head "was fitted on RSa" could never be supported by reading it.  An audit
            # made exactly that claim tonight and I repeated it before checking.  Stamp real.
            "fit_kind": ("pooled-geomean|r| "
                         + _SBAND_FLEET_A + "+" + _SBAND_FLEET_B
                         + f" OOF | L1-on-log | {TAG}"),
            "n_train_windows": int(tr.size),
            **({"strict_plan_stamp": strict_meta['strict_plan_stamp'],
                "fit_files": plan.fitting_files, "strict_plan": str(plan.path)} if plan else {})},
           OUT / f"sband_{TAG}.pth")
json.dump(RES, open(OUT / f"oof_{TAG}.json", "w"), indent=1)
print("SBB_OK " + json.dumps(RES), flush=True)
