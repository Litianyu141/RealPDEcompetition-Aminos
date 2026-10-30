#!/usr/bin/env python3
"""Track1 sps, round 2: lock down the heteroscedastic band and make it shippable.

Round 1 established the effect and its control:

    homoscedastic ceiling on HOLD   61.006  == the shipped band exactly (every rescale was worse)
    best heteroscedastic on HOLD    61.598  chosen on FIT, dFINAL +0.1183

The control is the important half. Rescaling the shipped band by any global factor made HOLD worse,
which independently reproduces the earlier finding that homoscedastic band tuning is exhausted --
so the entire +0.118 is attributable to letting the width follow an input-derived activity field,
not to having tuned a scale.

Three things round 1 left open, all handled here.

1. A NORMALISATION LEAK. Round 1 divided the activity field by its mean over ALL 378 samples,
   HOLD included. It is one scalar per channel and the effect is surely tiny, but a holdout number
   computed with a constant that saw the holdout is not a holdout number. Here the constant is
   fitted on FIT alone.

2. WHAT THE BUNDLE CAN ACTUALLY COMPUTE. A frozen per-channel constant is legitimate -- it is a
   calibration parameter, not test information -- but it assumes the platform's test set has the
   same overall activity level as our local val set, which is exactly the assumption that has
   already burned us once on this competition (the local/platform offset is -2.66). So the
   self-normalising variant is tested head to head: divide each sample's activity by its own
   spatial mean, which needs no constant at all and adapts to whatever the platform feeds it. If
   it is close, it ships, because it cannot be wrong-footed by a distribution shift.

3. PER-LEAD-TIME REFIT. The width is currently the shipped band times one global scale times the
   activity field. But sps is EXACTLY separable over (lead time, channel):

       x = (1/(LT*c)) * sum_{t,ch} mean_n( W[n] * mean_{y,x}( E * 1{inside} )[n,t,ch] )

   so each (t,ch) cell can be optimised independently and the result is the exact global optimum
   over per-cell scales -- no coordinate descent, no local minima. That is also precisely the
   family that overfitted last time when it was fitted freely, so it is shrunk back toward the
   global scale and the shrinkage strength is chosen on FIT.

Everything is chosen on FIT (even trajectories, 210 samples) and reported on HOLD (odd, 168).
The winner is re-scored end to end with the unmodified official scorer before anything is claimed.
"""
from __future__ import annotations
import os, sys, json, argparse, zipfile, tempfile
from pathlib import Path
import numpy as np

# Derived from this file's own location (2026-08-12, F2): the absolute hardcode silently
# escaped a fresh clone back to the production tree.  Run it from its in-repo location.
REPO = Path(__file__).resolve().parents[2]
T1U, T2U = REPO / "Track1/U-net", REPO / "Track2/U-net"
KIT = REPO / "Track2/realpde_t2_starting_kit_v4"      # local_eval lives only here; scorer is shared
DATA = REPO / "data/RealPDE_competition_dataset/gdrive_raw/local_bench/val"

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default=str(T1U / "checkpoints/unet_v3_rel_tke20_s1.pth"))
ap.add_argument("--bundle", default=str(T1U / "submissions/t1_unet_v7.zip"))
ap.add_argument("--gpu", default="7")
ap.add_argument("--json", default="/tmp/t1_sps_v2.json")
ap.add_argument("--out", default=str(T1U / "calib_sps_hetero.npz"))
A = ap.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(A.gpu)
import torch

# T1U LAST-inserted == FIRST-resolved: unet_model MUST come from Track1 (Track2 holds the
# tombstone; it caught this pattern twice in the 2026-08-02 clean-room reproduce run).
sys.path.insert(0, str(KIT)); sys.path.insert(0, str(T2U)); sys.path.insert(0, str(T1U))
from local_eval import build_stream, Normalizer
from unet_model import build_unet, build_from_meta
import scoring as official

with zipfile.ZipFile(A.bundle) as z:
    nm = next(n for n in z.namelist() if n.endswith("calib.npz"))
    with tempfile.TemporaryDirectory() as td:
        z.extract(nm, td)
        cal = dict(np.load(Path(td) / nm))
def _band_from_calib(cal, key):
    """Read a band from a calib dict under either name.

    `t1_band_v5.py` (the from-scratch fitter) writes `band_lo`/`band_hi`; the historical
    `calib_v9.npz` chain used `band_lo_phys`/`band_hi_phys`.  Both are the PHYSICAL band -- v5 fits
    against SIGMA_GLOBAL and scores on real targets -- so this is a name, not a space.  Preferring
    the explicit `_phys` name keeps historical calibs bit-identical in behaviour.
    """
    import numpy as _np
    for k in (key + "_phys", key):
        if k in cal:
            a = _np.asarray(cal[k], _np.float32)
            if a.ndim != 2:
                raise SystemExit(f"calib['{k}'] has shape {a.shape}; expected (n_t, C)")
            return a
    raise SystemExit(f"calib has neither '{key}_phys' nor '{key}'; keys are {sorted(cal)}")

GAMMA = cal["gamma"]
BL0 = _band_from_calib(cal, "band_lo")
BH0 = _band_from_calib(cal, "band_hi")

dev = "cuda" if torch.cuda.is_available() else "cpu"
st = torch.load(A.checkpoint, map_location="cpu", weights_only=False)
meta = st.get("meta", {}) if isinstance(st, dict) else {}
# 2026-08-04 XNXa clean-room incident: same base-only rebuild as t1_band_v5 - see comment there.
model = build_from_meta(meta).to(dev).eval()
model.load_state_dict(st["model_state_dict"] if isinstance(st, dict) and "model_state_dict" in st else st)
norm = Normalizer(DATA / "mean_std_real.pt")
g_t = torch.as_tensor(GAMMA.reshape(1, 1, *GAMMA.shape)).to(dev)

pn, tg, il, traj, tid = [], [], [], [], -1
for step in build_stream(DATA):
    inp, tgt = step["input"].unsqueeze(0), step["target"].unsqueeze(0)
    if step["is_first"]:
        tid += 1
    inp_norm, _ = norm.preprocess(inp, tgt)
    with torch.no_grad():
        out = model(torch.as_tensor(inp_norm).to(dev))
        m = out.mean(dim=1, keepdim=True)
        out = m + (out - m) * g_t
    pn.append(out.detach().cpu())
    tg.append(tgt.squeeze(0).cpu().numpy().astype(np.float32))
    il.append(inp.squeeze(0).cpu().numpy().astype(np.float32))
    traj.append(tid)

P = norm.postprocess_pred(torch.cat(pn, 0)).numpy().astype(np.float32)
T = np.stack(tg, 0); IN = np.stack(il, 0); traj = np.asarray(traj)
c = int(official.measured_channels(T))
N, LT, H, Wd = P.shape[:4]
fit_i = np.where(traj % 2 == 0)[0]; hold_i = np.where(traj % 2 == 1)[0]
Pc = np.ascontiguousarray(P[..., :c]); Tc = np.ascontiguousarray(T[..., :c])
del pn, tg

dm = np.asarray(official.rel_l2_per_sample(P, T, c))
tk = np.asarray(official.tke_rel_l2_per_sample(P, T, c))
mv = np.asarray(official.mvpe_rel_l2_per_sample(P, T))
PM = {k: np.asarray(v, np.float64) / (0.5 + np.asarray(v, np.float64))
      for k, v in (("dm", dm), ("tke", tk), ("mvpe", mv))}
Wn = 0.5 * (1 - PM["dm"]) + 0.3 * (1 - PM["tke"]) + 0.2 * (1 - PM["mvpe"])
print(f"# N={N} fit {fit_i.size} / hold {hold_i.size}  c={c}  W mean {Wn.mean():.4f}  "
      f"ceiling at this accuracy score_sps(W) = {official.score_sps(float(Wn.mean())):.2f}",
      flush=True)


def masked_E(LO, HI):
    """The scorer's per-element E*1{inside}, float32, computed exactly as scoring.py does it."""
    lower = (Pc + LO).astype(np.float32)
    upper = (Pc + HI).astype(np.float32)
    inside = (Tc >= lower) & (Tc <= upper)
    nil = (upper - lower) / official.SIGMA_GLOBAL
    return np.where(inside, np.exp(-nil), 0.0), inside


def sps_of(LO, HI):
    e, inside = masked_E(LO, HI)
    per_n = e.reshape(N, -1).mean(axis=1, dtype=np.float64)
    cov_n = inside.reshape(N, -1).mean(axis=1, dtype=np.float64)
    w = Wn * per_n
    return (official.score_sps(w[fit_i].mean()), official.score_sps(w[hold_i].mean()),
            float(cov_n[hold_i].mean()))


# ------------------------------------------------------------------ activity fields, FIT-normalised
raw = {"std_t_in": IN[..., :c].std(axis=1).astype(np.float32)}
ke_in = 0.5 * (IN[..., 0].var(axis=1) + IN[..., 1].var(axis=1))
raw["sqrt_ke_in"] = np.repeat(np.sqrt(np.maximum(ke_in, 0))[..., None], c, -1).astype(np.float32)
del IN, il

ACT, CONST = {}, {}
for base, a in raw.items():
    cst = a[fit_i].mean(axis=(0, 1, 2))                       # FIT ONLY -- the round-1 leak, fixed
    ACT[base + "|glob"] = a / np.maximum(cst, 1e-12)
    CONST[base + "|glob"] = cst.astype(np.float32)
    ACT[base + "|self"] = a / np.maximum(a.mean(axis=(1, 2), keepdims=True), 1e-12)
    CONST[base + "|self"] = None
    print(f"# {base:<11} FIT const per channel {np.round(cst, 6).tolist()}", flush=True)
del raw

# stage 5 pads the band to 3 channels for the submission's reader; the fit runs on the c
# MEASURED channels, so slice rather than reshape a padded array.
LO0 = BL0[:, :c].reshape(1, LT, 1, 1, c); HI0 = BH0[:, :c].reshape(1, LT, 1, 1, c)
BF, BH_, BC = sps_of(LO0, HI0)
print(f"# baseline shipped band: FIT {BF:.3f}  HOLD {BH_:.3f}  cov {BC:.4f}\n", flush=True)


def s_of(tag, p, lam):
    a = np.clip(ACT[tag] ** p, 0.2, 5.0).astype(np.float32)
    return ((1.0 - lam) + lam * a).astype(np.float32)[:, None]     # (N,1,H,W,c)


rows = []
hdr = f"{'activity':<20}{'p':>6}{'lam':>6}{'k':>6}{'FIT':>9}{'HOLD':>9}{'cov':>8}{'dFINAL':>9}"
print(hdr + "\n" + "-" * len(hdr), flush=True)
for tag in ACT:
    for p in (0.35, 0.5, 0.65):
        for lam in (0.7, 0.85, 1.0):
            for k in (1.1, 1.2, 1.3, 1.4):
                s = s_of(tag, p, lam)
                sf, sh, cv = sps_of(LO0 * k * s, HI0 * k * s)
                rows.append(dict(activity=tag, p=p, lam=lam, k=k, fit=sf, hold=sh, cov=cv,
                                 dFINAL=(sh - BH_) / 5.0))
                print(f"{tag:<20}{p:>6.2f}{lam:>6.2f}{k:>6.2f}{sf:>9.3f}{sh:>9.3f}{cv:>8.4f}"
                      f"{(sh-BH_)/5.0:>+9.4f}", flush=True)

best = max(rows, key=lambda r: r["fit"])
print(f"\n# FIT-best: {best['activity']} p={best['p']} lam={best['lam']} k={best['k']}"
      f"  -> HOLD {best['hold']:.3f}  dFINAL {best['dFINAL']:+.4f}", flush=True)
for fam in ("glob", "self"):
    sub = [r for r in rows if r["activity"].endswith(fam)]
    b = max(sub, key=lambda r: r["fit"])
    print(f"#   best within '{fam}': {b['activity']} p={b['p']} lam={b['lam']} k={b['k']} "
          f"-> HOLD {b['hold']:.3f} ({b['dFINAL']:+.4f})", flush=True)

# --------------------------------------------- exact per-(lead time, channel) scale, then shrunk
S = s_of(best["activity"], best["p"], best["lam"])
KGRID = np.round(best["k"] * np.linspace(0.55, 1.75, 25), 5)
Cfit = np.zeros((KGRID.size, LT, c))
for j, k in enumerate(KGRID):
    e, _ = masked_E(LO0 * k * S, HI0 * k * S)
    g = e.mean(axis=(2, 3), dtype=np.float64)                       # (N, LT, c)
    Cfit[j] = (Wn[fit_i, None, None] * g[fit_i]).mean(axis=0)
    del e, g
kcell = KGRID[Cfit.argmax(axis=0)]                                  # (LT, c) exact per-cell optimum
print(f"\n# exact per-cell scale on FIT: min {kcell.min():.3f} max {kcell.max():.3f} "
      f"mean {kcell.mean():.3f}  (global {best['k']:.3f}); grid endpoints hit: "
      f"{int((kcell==KGRID[0]).sum())} low / {int((kcell==KGRID[-1]).sum())} high", flush=True)
print(f"{'shrinkage':<20}{'':>12}{'':>6}{'FIT':>9}{'HOLD':>9}{'cov':>8}{'dFINAL':>9}", flush=True)
shr = []
for lam_s in (0.0, 0.25, 0.5, 0.75, 1.0):
    kk = (best["k"] ** (1 - lam_s)) * (kcell ** lam_s)
    kk3 = kk.reshape(1, LT, 1, 1, c).astype(np.float32)
    sf, sh, cv = sps_of(LO0 * kk3 * S, HI0 * kk3 * S)
    shr.append(dict(lam_s=lam_s, fit=sf, hold=sh, cov=cv, dFINAL=(sh - BH_) / 5.0,
                    kcell=kk.tolist()))
    print(f"{'lam_s=%.2f' % lam_s:<20}{'':>12}{'':>6}{sf:>9.3f}{sh:>9.3f}{cv:>8.4f}"
          f"{(sh-BH_)/5.0:>+9.4f}", flush=True)

win = max(shr, key=lambda r: r["fit"])
print(f"\n# shrinkage chosen on FIT: lam_s={win['lam_s']}  -> HOLD {win['hold']:.3f} "
      f"dFINAL {win['dFINAL']:+.4f}", flush=True)

# ------------------------------------------------------- final end-to-end check, official scorer
kk = (best["k"] ** (1 - win["lam_s"])) * (kcell ** win["lam_s"])
kk3 = kk.reshape(1, LT, 1, 1, c).astype(np.float32)
LOf, HIf = LO0 * kk3 * S, HI0 * kk3 * S
lo_full = (Pc + LOf).astype(np.float32); hi_full = (Pc + HIf).astype(np.float32)
assert np.all(lo_full <= hi_full), "final band is not ordered"
x_ref, cov_ref = official.aggregate_sps(P, T, c, lower=lo_full, upper=hi_full)
xb, covb = official.aggregate_sps(P, T, c, lower=(Pc + LO0).astype(np.float32),
                                  upper=(Pc + HI0).astype(np.float32))
print(f"\n# OFFICIAL scorer, all 378 samples (this is the number the bundle must reproduce)",
      flush=True)
print(f"#   shipped band   sps {official.score_sps(xb):.4f}  cov {covb:.4f}", flush=True)
print(f"#   hetero band    sps {official.score_sps(x_ref):.4f}  cov {cov_ref:.4f}   "
      f"dSPS {official.score_sps(x_ref)-official.score_sps(xb):+.4f}  "
      f"dFINAL {(official.score_sps(x_ref)-official.score_sps(xb))/5:+.4f}", flush=True)

np.savez_compressed(A.out, gamma=GAMMA, band_lo_phys=BL0, band_hi_phys=BH0,
                    act_kind=best["activity"], p=best["p"], lam=best["lam"],
                    k_cell=kk.astype(np.float32),
                    act_const=(CONST[best["activity"]] if CONST[best["activity"]] is not None
                               else np.zeros(c, np.float32)),
                    self_norm=np.array(best["activity"].endswith("self")))
json.dump({"baseline_hold": BH_, "rows": rows, "best": best, "shrink": shr,
           "official_shipped_sps": official.score_sps(xb),
           "official_hetero_sps": official.score_sps(x_ref)}, open(A.json, "w"), indent=2)
print(f"\n# wrote {A.out}", flush=True)
print("\nT1_SPS_V2_OK", flush=True)
