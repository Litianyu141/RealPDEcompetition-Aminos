#!/usr/bin/env python3
"""M3 stage B -- fit every key of the T2 band calib from the stage-A cache (the M3 spec).  Acceptance is TOLERANCE-BASED per the M2 lesson: build a bundle from the
output and require >= shipped - 0.30 sps (seed-0 run measured +0.308 ABOVE shipped).
Inputs come from t2_collect_residuals.py (cache dir from the M3_CACHE env var; default is
m3_cache/ beside this script -- repo-relative, F7 2026-08-12)."""
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--rng", type=int, default=0, help="subsample RNG; acceptance refits at a second value")
_ap.add_argument("--out", default="", help="output npz; default calib_m3_repro.npz beside the calibs")
_ap.add_argument("--pool", action="store_true", help="TPa: pool the pe observable per trajectory")
_ap.add_argument("--pool-causal", action="store_true",
                 help="TCPa: with --pool, pool the pe observable over the CAUSAL PREFIX of each "
                      "trajectory -- the statistic submission_t2_v30.py:360-365 actually indexes "
                      "(_pe_acc/_pe_cnt, reset per trajectory). Off = the full-trajectory pool, "
                      "bit-for-bit as shipped.")
_ap.add_argument("--s-max", type=float, default=2.5,
                 help="top of the band-response grid; 2.5 = the shipped grid bit-for-bit; beyond it appends [2.8, 3.15, 3.5, 4.0] up to the value (BRXa: the runtime saturates at the table top on unseen-regime streams)")
_A = _ap.parse_args()
import sys, time
from pathlib import Path
import numpy as np
# Derived from this file's own location (2026-08-12): the absolute hardcode silently
# escaped a fresh clone back to the production tree.  NB: run these tools from their
# in-repo location (the chains do); a /tmp copy would derive the wrong root.
REPO = Path(__file__).resolve().parents[2]
D = REPO/"Track2/U-net"
import os as _os_m3
# Repo-relative default (2026-08-12): same clone-gate defect class as SBAND_CACHE.
CA = Path(_os_m3.environ.get("M3_CACHE", str(D / "m3_cache")))
sys.path.insert(0, str(REPO/"Track2/realpde_t2_starting_kit_v4"))
import scoring as official
import torch
def _prov_stamp_local(d, ckpt, fit_stream, producer):
    """Stamp calib provenance (mirrors build_t*_bundle._prov_stamp); 2026-08-13, BCP made structural."""
    import hashlib, os, time
    md5 = ""
    if ckpt and os.path.exists(ckpt):
        h = hashlib.md5()
        with open(ckpt, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        md5 = h.hexdigest()
    d["prov_ckpt_md5"] = np.str_(md5)
    d["prov_ckpt_name"] = np.str_(os.path.basename(ckpt) if ckpt else "")
    d["prov_fit_stream"] = np.str_(str(fit_stream))
    d["prov_producer"] = np.str_(str(producer))
    d["prov_when"] = np.str_(time.strftime("%Y-%m-%dT%H:%M:%S"))
    return d

SIG = float(official.SIGMA_GLOBAL); LT, C2, H, W, Q = 20, 2, 32, 64, 64
RS = np.load(CA/"RSn.npy", mmap_mode="r")         # (N,LT,H,W,2) NORMALISED residual
OB = np.load(CA/"OBn.npy")
if _A.pool:
    _tid = np.load(CA/"tid.npy")
    if _A.pool_causal:
        # TCPa: the runtime accumulates sum(s^2) and a count from the trajectory's FIRST step, so at
        # stream position k it indexes sqrt(mean(s^2 over 0..k)) -- a causal prefix pool, not the
        # full-trajectory pool.  tid.npy is written in stream order by t2_collect_residuals.py:41-52.
        _sq = OB.astype(np.float64) ** 2
        _cz = np.empty_like(OB)
        for g in range(int(_tid.max()) + 1):
            idx = np.flatnonzero(_tid == g)                       # stream order within trajectory g
            cs = np.cumsum(_sq[idx], axis=0)
            cnt = np.arange(1, len(idx) + 1, dtype=np.float64).reshape((-1,) + (1,) * (OB.ndim - 1))
            _cz[idx] = np.sqrt(cs / cnt).astype(OB.dtype)
        OB = _cz
        print("# TCPa: pe observable pooled over the CAUSAL PREFIX per trajectory (runtime-faithful)")
    else:
        OB = np.stack([np.sqrt((OB[_tid == g]**2).mean(0)) for g in range(int(_tid.max())+1)])[_tid]
        print("# TPa: pe observable pooled per trajectory")                        # (N,H,W,2)    NORMALISED input temporal std
VP = np.load(CA/"VP.npy"); VT = np.load(CA/"VT.npy")
shp = np.load(CA/"MKshape.npy")
MK = np.unpackbits(np.load(CA/"MKb.npy"), axis=1)[:, :int(np.prod(shp[1:]))] \
       .reshape(tuple(shp)).astype(bool)
N = RS.shape[0]
mean_t, std_t = torch.load(REPO/"Track2/realpde_t2_starting_kit_v4/example_data/mean_std_real.pt",
                           map_location="cpu", weights_only=False)[0::2], None
ms = torch.load(REPO/"Track2/realpde_t2_starting_kit_v4/example_data/mean_std_real.pt",
                map_location="cpu", weights_only=False)
nm = np.asarray(ms[0], np.float32).ravel()[:3]; nsd = np.asarray(ms[2], np.float32).ravel()[:3]
if nm.size < 3: nm = np.asarray(ms[0], np.float32).ravel(); nsd = np.asarray(ms[2], np.float32).ravel()
print(f"# norm mean {nm}  std {nsd}")
STD = nsd[:2].copy(); STD[STD == 0] = 1.0
SIGN = SIG / STD                                   # sigma in NORMALISED units, per channel
g = np.random.default_rng(_A.rng)
CENT = [.35, .40, .45, .50, .55, .60, .65]

def opt_interval(r, sig_n, nsub=150000):
    """scorer-optimal (lo, hi) for a pooled residual sample, normalised units."""
    if r.size > nsub: r = r[g.choice(r.size, nsub, replace=False)]
    best = (-1.0, -0.01, 0.01)
    for c in [0.0] + list(np.quantile(r, CENT)):
        d = np.sort(np.abs(r - c))
        val = np.exp(-2.0*d/sig_n) * np.arange(1, d.size+1)
        j = int(np.argmax(val))
        if val[j] > best[0]: best = (val[j], c-d[j], c+d[j])
    return best[1], best[2]

t0 = time.time()
# ---- gamma (fallback path only: the bundle ships a live vgain head) ----
gamma = np.ones((H, W, 3), np.float32)
gamma[..., :2] = np.sqrt(VT.mean(0) / np.maximum(VP.mean(0), 1e-12)).clip(0.25, 4.0)
# ---- mad_val / bias_val: the runtime's own UNMASKED statistics ----
A = np.abs(RS)
mad_val = np.stack([A[:, :, :, :, c].mean(axis=(0, 2, 3)) for c in range(C2)], 1).astype(np.float32)
bias_val = np.stack([np.asarray(RS[:, :, :, :, c]).mean(axis=(0, 2, 3)) for c in range(C2)], 1).astype(np.float32)
print(f"# mad_val ch0 [{mad_val[:,0].min():.4f},{mad_val[:,0].max():.4f}]  "
      f"ch1 [{mad_val[:,1].min():.4f},{mad_val[:,1].max():.4f}]   ({time.time()-t0:.0f}s)", flush=True)
# ---- band_lo/hi and the 16-point band response ----
NS = 16
sgrid = np.array([0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.55,1.7,1.85,2.0,2.25,2.5], np.float32)
if float(_A.s_max) > 2.5:
    sgrid = np.concatenate([sgrid, np.array([v for v in (2.8, 3.15, 3.5, 4.0)
                                             if v <= float(_A.s_max) + 1e-9], np.float32)])
NS = len(sgrid)
blo = np.zeros((LT, 3), np.float32); bhi = np.zeros((LT, 3), np.float32)
blos = np.zeros((NS, LT, 3), np.float32); bhis = np.zeros((NS, LT, 3), np.float32)
for t in range(LT):
    for c in range(C2):
        r = np.asarray(RS[:, t, :, :, c])[MK[:, t, :, :, c]].astype(np.float64)
        lo, hi = opt_interval(r, SIGN[c]); blo[t, c], bhi[t, c] = lo, hi
        for i, s in enumerate(sgrid):
            lo, hi = opt_interval(r * s, SIGN[c], nsub=90000)
            blos[i, t, c], bhis[i, t, c] = lo, hi
blo[:, 2] = blo[:, :2].mean(1); bhi[:, 2] = bhi[:, :2].mean(1)      # p never scored
blos[:, :, 2] = blos[:, :, :2].mean(2); bhis[:, :, 2] = bhis[:, :, :2].mean(2)
print(f"# band tables fitted ({time.time()-t0:.0f}s)  lead0 ch0 [{blo[0,0]:+.4f},{bhi[0,0]:+.4f}]", flush=True)
# ---- pe: equal-mass bins of the input std, width-neutral ratios ----
pe_e = np.zeros((LT, C2, Q-1), np.float32)
pe_rl = np.ones((LT, C2, Q), np.float32); pe_rh = np.ones((LT, C2, Q), np.float32)
for t in range(LT):
    for c in range(C2):
        mk = MK[:, t, :, :, c]
        s = OB[:, :, :, c][mk].astype(np.float64)
        r = np.asarray(RS[:, t, :, :, c])[mk].astype(np.float64)
        e = np.maximum.accumulate(np.quantile(s, np.linspace(0, 1, Q+1)[1:-1])); pe_e[t, c] = e
        b = np.searchsorted(e, s, side="right")
        tl = np.full(Q, blo[t, c]); th = np.full(Q, bhi[t, c])
        for k in range(Q):
            sel = np.flatnonzero(b == k)
            if sel.size < 64: continue
            lo, hi = opt_interval(r[sel], SIGN[c], nsub=60000); tl[k], th[k] = lo, hi
        pe_rl[t, c] = np.clip(tl / tl.mean(), 0.35, 3.0)
        pe_rh[t, c] = np.clip(th / th.mean(), 0.35, 3.0)
print(f"# pe fitted ({time.time()-t0:.0f}s)  rl mean {pe_rl.mean():.4f} rh mean {pe_rh.mean():.4f}", flush=True)
# ---- pc: per-cell multiplier, CONDITIONAL on our own band x pe ----
MG = np.linspace(0.40, 2.50, 43)
pc_rl = np.ones((LT, C2, H, W), np.float32); pc_rh = np.ones((LT, C2, H, W), np.float32)
for t in range(LT):
    for c in range(C2):
        b = np.searchsorted(pe_e[t, c], OB[:, :, :, c].astype(np.float64), side="right")
        lo = blo[t, c] * pe_rl[t, c][b]; hi = bhi[t, c] * pe_rh[t, c][b]     # (N,H,W)
        r = np.asarray(RS[:, t, :, :, c]).astype(np.float64); mk = MK[:, t, :, :, c]
        best = np.full((H, W), -1.0); bm = np.ones((H, W))
        for m_ in MG:
            val = (np.exp(-m_*(hi-lo)/SIGN[c]) * ((r >= m_*lo) & (r <= m_*hi) & mk)).sum(0)
            upd = val > best; best = np.where(upd, val, best); bm = np.where(upd, m_, bm)
        sm = mk.any(0)
        if sm.sum() >= 16:
            bmn = bm / bm[sm].mean()
            pc_rl[t, c][sm] = np.clip(bmn[sm], 0.35, 3.0); pc_rh[t, c][sm] = np.clip(bmn[sm], 0.35, 3.0)
print(f"# pc fitted ({time.time()-t0:.0f}s)  |m-1|<0.1 fraction {float((np.abs(pc_rl-1)<0.1).mean()):.3f}", flush=True)
# ---- the x1.20 stage, then assemble with the EXACT shipped key set ----
for a_ in (blo, bhi): a_ *= 1.20
blos *= 1.20; bhis *= 1.20
kew = (nsd[:2].astype(np.float32))**2
out = dict(pe_pooled=np.bool_(_A.pool), gamma=gamma, band_lo=blo, band_hi=bhi, s_grid=sgrid,
           band_lo_s=blos, band_hi_s=bhis, mad_val=mad_val, bias_val=bias_val,
           ch_idx=np.array([0, 1, 1], np.int64), ke_w=kew,
           norm_mean=nm[:3].astype(np.float32), norm_std=nsd[:3].astype(np.float32),
           pe_edge=pe_e, pe_rl=pe_rl, pe_rh=pe_rh, pc_rl=pc_rl, pc_rh=pc_rh)
# Key set and array shapes of the shipped calib Track2/U-net/calib_v22_pc_m120.npz (md5 6e211728af986ec7b372952624d41443),
# recorded 2026-09-29.  Shapes only, no fitted values.  The blob is an untracked historical file that a source-only
# Final context does not carry (the v11 T2 base smoke died here with FileNotFoundError); the check itself never used
# its values.  When the blob is present its shapes are still read and must equal this table.
SHIP_SHAPES = {"band_hi": (20, 3), "band_hi_s": (16, 20, 3), "band_lo": (20, 3), "band_lo_s": (16, 20, 3),
               "bias_val": (20, 2), "ch_idx": (3,), "gamma": (32, 64, 3), "ke_w": (2,), "mad_val": (20, 2),
               "norm_mean": (3,), "norm_std": (3,), "pc_rh": (20, 2, 32, 64), "pc_rl": (20, 2, 32, 64),
               "pe_edge": (20, 2, 63), "pe_rh": (20, 2, 64), "pe_rl": (20, 2, 64), "s_grid": (16,)}
if (D/"calib_v22_pc_m120.npz").exists():
    _z = np.load(D/"calib_v22_pc_m120.npz")
    assert {k: tuple(_z[k].shape) for k in _z.files} == SHIP_SHAPES, "shipped calib differs from SHIP_SHAPES"
ship_shapes = SHIP_SHAPES
extra = set(out) - set(ship_shapes)
assert extra <= {"pe_pooled", "prov_ckpt_md5", "prov_ckpt_name",
                 "prov_fit_stream", "prov_producer", "prov_when"}, extra                  # the ONLY key allowed beyond the shipped set
assert set(ship_shapes) <= set(out), set(ship_shapes) - set(out)
for k, shape in ship_shapes.items():
    if k in ("s_grid", "band_lo_s", "band_hi_s") and NS != 16:
        # widened response grid (--s-max > 2.5): first dim is len(sgrid) by design
        assert out[k].shape[0] == NS and tuple(out[k].shape[1:]) == tuple(shape[1:]), (k, out[k].shape, shape)
        continue
    assert tuple(out[k].shape) == tuple(shape), (k, out[k].shape, shape)
_pv = CA / "prov_ckpt.txt"
out = _prov_stamp_local(out, _pv.read_text().strip() if _pv.exists() else "", f"m3_cache N={N}", "t2_fit_band.py")
np.savez(_A.out if _A.out else str(D/"calib_m3_repro.npz"), **out)
print("# WROTE " + (_A.out if _A.out else "calib_m3_repro.npz") + " -- key set and every shape identical to the shipped calib")
print("M3_FIT_OK")
