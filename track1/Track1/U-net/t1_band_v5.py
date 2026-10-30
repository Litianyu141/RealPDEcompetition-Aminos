#!/usr/bin/env python3
"""Track1 SPS band refit at the PLATFORM residual scale.

WHY. Our local val residuals are systematically smaller than the platform's
(Track2 measured: platform rel_l2 err 0.121307 vs local 0.07339 -> 1.65x).  A band
fitted at s=1 is too narrow when it meets the real test set: `inside` goes false and
aggregate_sps collapses.  But naively widening by the SAME 1.65x is also wrong -- the
band optimum grows strictly SUB-linearly in s, because
    J(h) = exp(-2h/SIGMA) * F(h/s)
trades an exponential width penalty against a coverage gain, so h* solves
    f(z)/F(z) = 2s/SIGMA,  z = h/s.
We therefore do it empirically: inflate residuals synthetically, refit, and score the
resulting band against a RANGE of true inflations -- a robustness matrix, not a point
estimate.

Synthetic inflation is exact for the thing that matters: with T_s = P + s*(T-P) the
residual is s*(T-P) identically, so `inside` sees precisely the geometry the platform
would show us, and the sample weights W re-derive from the inflated errors on their own.
"""
from __future__ import annotations
import os, sys, time, json, argparse, importlib.util
from pathlib import Path
import numpy as np

# Derived from this file's own location (2026-08-12): the absolute hardcode silently
# escaped a fresh clone back to the production tree.  NB: run these tools from their
# in-repo location (the chains do); a /tmp copy would derive the wrong root.
REPO = Path(__file__).resolve().parents[2]
T2U, KIT, T1U = REPO/"Track2/U-net", REPO/"Track2/realpde_t2_starting_kit_v4", REPO/"Track1/U-net"
DATA = REPO/"data/RealPDE_competition_dataset/gdrive_raw/local_bench/val"

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoints", default=str(REPO/"checkpoints/t1_u-net/unet_v3_rel_tke30.pth"))
ap.add_argument("--gpu", default="6")
ap.add_argument("--out", default=str(T1U/"calib_v5.npz"))
ap.add_argument("--json", default="/tmp/t1_band_v5.json")
ap.add_argument("--ref-hw", type=float, default=0.15)
ap.add_argument("--head", default="", help="vgain.pth -- apply the shipped head before fitting")
A = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(A.gpu)

import torch
# T1U LAST-inserted == FIRST-resolved: unet_model MUST come from Track1 (Track2 holds the
# tombstone that caught exactly this on 2026-08-02 during the clean-room reproduce run).
sys.path.insert(0, str(KIT)); sys.path.insert(0, str(T2U)); sys.path.insert(0, str(T1U))
spec = importlib.util.spec_from_file_location("cal", T2U/"calibrate_t2.py")
cal = importlib.util.module_from_spec(spec); spec.loader.exec_module(cal)
from local_eval import build_stream, Normalizer
from unet_model import build_unet, build_from_meta
import scoring as official

dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- 1. stream P, T
models = []
for cp in [p.strip() for p in A.checkpoints.split(",") if p.strip()]:
    s = torch.load(cp, map_location="cpu", weights_only=False)
    meta = s.get("meta", {}) if isinstance(s, dict) else {}
    st = s["model_state_dict"] if isinstance(s, dict) and "model_state_dict" in s else s
    # 2026-08-04 XNXa clean-room incident: rebuilding with base ONLY dropped the other
    # arch keys. Loud for block=convnext (state_dict mismatch); SILENT for parameter-free
    # keys like in_norm=self in earlier from-scratch runs (loaded fine, forwarded wrong).
    m = build_from_meta(meta).to(dev).eval()
    m.load_state_dict(st); models.append(m)
    print(f"# member {Path(cp).name} base={meta.get('base')} "
          f"loss={meta.get('loss_type')} tke_w={meta.get('tke_weight')} "
          f"val_rel_l2={meta.get('val_rel_l2')}", flush=True)

norm = Normalizer(DATA/"mean_std_real.pt")
pn, tg, xi, tms, traj, tid = [], [], [], [], [], -1
for step in build_stream(DATA):
    inp, tgt = step["input"].unsqueeze(0), step["target"].unsqueeze(0)
    if step["is_first"]: tid += 1
    inp_norm, _ = norm.preprocess(inp, tgt)
    t0 = time.perf_counter()
    x = torch.as_tensor(inp_norm).to(dev)
    with torch.no_grad():
        out = torch.stack([m(x) for m in models], 0).mean(0)
    if dev == "cuda": torch.cuda.synchronize()
    tms.append(time.perf_counter()-t0)
    pn.append(out.detach().cpu()); tg.append(tgt.squeeze(0).cpu().numpy().astype(np.float32))
    xi.append(inp.squeeze(0).cpu().numpy().astype(np.float32))
    traj.append(tid)

P0 = norm.postprocess_pred(torch.cat(pn, 0)).numpy().astype(np.float32)
T = np.stack(tg, 0); traj = np.asarray(traj)
X_IN = np.stack(xi, 0)

# ---- the shipped head, applied to the forecast the band will be fitted to -------------------
HEAD_ON = False
if A.head:
    from vgain_head import VGainHead, N_FEAT as _NFEAT, stats_from_window
    _hd = torch.load(A.head, map_location="cpu", weights_only=False)
    _h = VGainHead(int(_hd.get("n_feat", _NFEAT)), int(_hd["width"]),
                   float(_hd["gmin"]), float(_hd["gmax"]))
    _h.load_state_dict(_hd["state_dict"]); _h.to(dev).eval()
    _g = []
    with torch.no_grad():
        for i in range(0, P0.shape[0], 64):
            pb = torch.from_numpy(P0[i:i+64]).to(dev)
            xb = torch.from_numpy(X_IN[i:i+64]).to(dev)
            # feat is a function of (p, x_in) only -- the target slot is never read, so passing
            # pb there cannot leak. Equivalence with the submission's inlined copy: 4.8e-07.
            # 2026-08-06: the head may be a VG2 rich head (15 features).  Route through
            # the CANONICAL stats_from_window rather than assuming 7 -- three copies of
            # this computation exist and a drift between them mis-scales silently.
            f = stats_from_window(pb, pb, xb,
                                  rich=(_h.c1.in_channels != _NFEAT))["feat"]
            _g.append(_h(f).cpu().numpy().astype(np.float32))
    G = np.concatenate(_g, 0)[:, None, :, :, None]
    _pm = P0.mean(axis=1, keepdims=True)
    P0 = _pm + (P0 - _pm) * G
    HEAD_ON = True
    print(f"# HEAD {Path(A.head).name} applied: g in [{G.min():.3f},{G.max():.3f}] "
          f"mean {G.mean():.4f} -- the band is now fitted to the SHIPPED forecast", flush=True)
mean_step = float(np.mean(tms))
c, n_t, sigma = official.measured_channels(T), P0.shape[1], official.SIGMA_GLOBAL
print(f"# stream N={P0.shape[0]} traj={traj.max()+1} shape={P0.shape} c={c} "
      f"mean_step={mean_step*1e3:.2f} ms sigma={sigma}", flush=True)

_, _, _si, _st = torch.load(KIT/"example_data/mean_std_real.pt", map_location="cpu", weights_only=False)
std_ch = np.asarray(_st, np.float32).reshape(-1).astype(np.float64).copy(); std_ch[std_ch == 0] = 1.0
ref_band = (A.ref_hw*std_ch).astype(np.float32).reshape(1,1,1,1,-1)

# ---------------------------------------------------------------- 2. gamma re-scan
Pm = P0.mean(axis=1, keepdims=True); Pf = P0 - Pm
P_of = lambda g, idx=None: (Pm + np.float32(g)*Pf) if idx is None else (Pm+np.float32(g)*Pf)[idx]
print("\n# gamma scan (constant normalized half-width reference)", flush=True)
grows = []
GSCAN = [1.0] if HEAD_ON else [1.0,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.5,1.6]
for g in GSCAN:
    Q = P_of(g)
    sub, fin, cov = cal.final_score(Q, T, official, mean_step, Q-ref_band, Q+ref_band)
    grows.append((fin, g, sub))
    print(f"  gamma={g:<5} relL2={sub['rel_l2_score']:.2f} tke={sub['tke_score']:.2f} "
          f"mvpe={sub['mvpe_score']:.2f} sps={sub['sps_score']:.2f} FINAL={fin:.3f}", flush=True)
grows.sort(key=lambda r: -r[0]); gstar = grows[0][1]
print(f"# gamma* = {gstar}   (v2 shipped a different one; the tke-weighted model already "
      f"predicts more fluctuation)", flush=True)

# ---------------------------------------------------------------- 3. s_fit x s_true sweep
S_FIT  = [1.0, 1.15, 1.30, 1.45, 1.60, 1.75, 1.90, 2.10]
S_TRUE = [1.00, 1.30, 1.50, 1.65, 1.80, 2.00]
PLAUS  = [1.30, 1.50, 1.65, 1.80, 2.00]        # the platform is certainly not at s=1

fit_i = np.where(traj % 2 == 0)[0]; hold_i = np.where(traj % 2 == 1)[0]
Pfit, Phold = P_of(gstar, fit_i), P_of(gstar, hold_i)
Tfit, Thold = T[fit_i], T[hold_i]
print(f"\n# holdout: fit {len(fit_i)} / hold {len(hold_i)} samples (split by trajectory)", flush=True)

infl = lambda P, Tt, s: (P + np.float32(s)*(Tt - P)).astype(np.float32)

bands, mat = {}, {}
for sf in S_FIT:
    Ts = infl(Pfit, Tfit, sf)
    W = cal.sample_weights(Pfit, Ts, c, official)
    lo, hi = cal.fit_band(Pfit, Ts, c, W, sigma, n_t)
    bands[sf] = (lo, hi)
    L, U = cal.expand_band(Phold, lo, hi, c)
    row = {}
    for st_ in S_TRUE:
        Th = infl(Phold, Thold, st_)
        sub, fin, cov = cal.final_score(Phold, Th, official, mean_step, L, U)
        row[st_] = (sub["sps_score"], cov)
    mat[sf] = row
    hw = float(np.mean(hi-lo))
    print(f"  s_fit={sf:<5} mean_width={hw:.4f}  " +
          "  ".join(f"s={st_}:{row[st_][0]:6.2f}(cov {row[st_][1]:.2f})" for st_ in S_TRUE), flush=True)

score_of = lambda sf: float(np.mean([mat[sf][s][0] for s in PLAUS]))
worst_of = lambda sf: float(min(mat[sf][s][0] for s in PLAUS))
rank = sorted(S_FIT, key=lambda sf: -(0.5*score_of(sf) + 0.5*worst_of(sf)))
print("\n# selection over the plausible platform range " + str(PLAUS), flush=True)
for sf in rank:
    print(f"  s_fit={sf:<5} mean_sps={score_of(sf):6.3f}  worst_sps={worst_of(sf):6.3f}  "
          f"sps@s=1={mat[sf][1.0][0]:6.3f}", flush=True)
s_star = rank[0]
print(f"# s_fit* = {s_star}  (chosen on 0.5*mean + 0.5*worst-case, holdout only)", flush=True)

# ---------------------------------------------------------------- 4. refit on all val, ship
Pall = P_of(gstar)
Tall_s = infl(Pall, T, s_star)
Wall = cal.sample_weights(Pall, Tall_s, c, official)
lo_f, hi_f = cal.fit_band(Pall, Tall_s, c, Wall, sigma, n_t)
Lf, Uf = cal.expand_band(Pall, lo_f, hi_f, c)
sub_f, fin_f, cov_f = cal.final_score(Pall, T, official, mean_step, Lf, Uf)   # scored on TRUE targets
sub_r, fin_r, cov_r = cal.final_score(Pall, T, official, mean_step, Pall-ref_band, Pall+ref_band)
print(f"\n# FINAL band, refit on all val at s={s_star}, scored against the REAL local targets:")
print(f"  constant-hw : sps={sub_r['sps_score']:.3f} cov={cov_r:.4f} FINAL={fin_r:.3f}")
print(f"  fitted band : sps={sub_f['sps_score']:.3f} cov={cov_f:.4f} FINAL={fin_f:.3f}")
for k in ("rel_l2_score","tke_score","mvpe_score","time_score","sps_score"):
    print(f"    {k:15s} {sub_f[k]:.3f}")

gam = np.full((P0.shape[2], P0.shape[3], P0.shape[4]), np.float32(gstar), np.float32)
# UNITS.  lo_f/hi_f are PHYSICAL offsets: the fit runs against P0 (de-normalised) and a
# physical reference band (ref_hw * std_ch).  `submission_t1_v6.py` applies calib["band_lo"] in
# NORMALISED space (`lo_n = pn + lo_b`) and de-normalises afterwards, so the shipped key must be the
# normalised offset.  Writing the physical one there costs 8.64 sps / 1.73 FINAL -- measured, same
# checkpoint and head, 2026-07-29.  Both representations are stored, related by the identity
# `band_lo * std_t == band_lo_phys` that t1_v8_verify.py asserts and that t1_sps_v2.py relies on
# when it fits k_cell/act_const in physical units.
_sd = std_ch.astype(np.float32).reshape(1, -1)[:, :lo_f.shape[1]]
lo_n_, hi_n_ = (lo_f / _sd).astype(np.float32), (hi_f / _sd).astype(np.float32)
assert np.abs(lo_n_ * _sd - lo_f).max() <= 1e-6 * max(np.abs(lo_f).max(), 1e-12), "band unit identity"
assert np.abs(hi_n_ * _sd - hi_f).max() <= 1e-6 * max(np.abs(hi_f).max(), 1e-12), "band unit identity"
print(f"# band units: phys mean|lo| {np.abs(lo_f).mean():.6f} -> normalised {np.abs(lo_n_).mean():.6f} "
      f"(std_t {std_ch.ravel()})", flush=True)
np.savez(A.out, gamma=gam, band_lo=lo_n_, band_hi=hi_n_,
         fit_kind=np.asarray("t1_band_v5 per-lead-time fit (band_lo normalised, band_lo_phys physical)"),
         band_lo_phys=lo_f.astype(np.float32), band_hi_phys=hi_f.astype(np.float32),
         s_fit=np.float32(s_star), gamma_star=np.float32(gstar),
         s_fit_grid=np.asarray(S_FIT, np.float32), s_true_grid=np.asarray(S_TRUE, np.float32),
         sps_matrix=np.asarray([[mat[sf][st_][0] for st_ in S_TRUE] for sf in S_FIT], np.float32))
json.dump({"gamma_star": gstar, "s_fit": s_star, "mean_step_s": mean_step,
           "coverage": cov_f, "FINAL_LOCAL": fin_f, "sub": sub_f,
           "constant_hw": {"FINAL": fin_r, "sps": sub_r["sps_score"], "cov": cov_r},
           "matrix": {str(sf): {str(st_): mat[sf][st_][0] for st_ in S_TRUE} for sf in S_FIT}},
          open(A.json, "w"), indent=2)
print(f"\n# wrote {A.out}  and  {A.json}")
print("T1_BAND_V5_OK")
