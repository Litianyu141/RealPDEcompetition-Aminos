#!/usr/bin/env python3
"""Merge the v8 calibration, then prove v8 == v7 + a better interval and nothing else.

WHAT v8 IS
----------
v7's forecast, unchanged, with a heteroscedastic uncertainty interval: the shipped per-(lead time,
channel) band multiplied by an exact per-cell scale k_cell and by an activity field computed at
test time from the 20 input frames. Holdout-validated at +0.1185 FINAL; +0.1459 when re-scored on
the full 378-window local benchmark through the official scorer.

WHY A SEPARATE VERIFY SCRIPT INSTEAD OF JUST RUNNING THE BUILDER
----------------------------------------------------------------
build_t1_bundle.py scores one bundle in isolation. The claim being made here is a DIFFERENCE claim
-- "only the interval moved" -- and the only way to test that is to run both bundles over the same
inputs in the same process and compare the returned arrays elementwise. If the prediction arrays
are not bit-identical then something other than the band changed, every subscore comparison below
is confounded, and the build must not proceed. np.array_equal, not allclose: there is no reason for
a single ulp of difference to exist, so any difference at all is a bug worth finding.

The two things this script is really guarding against:

  1. A UNIT MISMATCH. The study fitted k_cell and act_const against band_lo_phys, in physical
     units. The bundle applies the band in NORMALISED space and de-normalises afterwards. Those
     agree only if band_lo * std_t == band_lo_phys, which is an assumption about how the v7
     calibration was built, not a fact about this file. It is asserted, not trusted.
  2. A SILENT DEMOTION. The v4 loader is deliberately forgiving -- every failure path leaves it
     behaving exactly like v3. That is the right behaviour on the platform and the wrong behaviour
     here, where a demoted bundle would quietly score v7's numbers and look like "no gain" rather
     than like a bug. So the sps delta is required to be positive and non-trivial before the zip is
     allowed to be built.

TIMING
------
The time subscore is the one thing the extra work can cost, so it is measured rather than projected:
both bundles are driven R times in the same process and the MEDIAN per-window cost is compared. The
builder's own report uses a single cold call, which is a different (larger) number; both conventions
are printed so the v7 row can be matched against the existing verify_report.json.
"""
from __future__ import annotations
import os, sys, json, time, argparse, importlib.util, shutil
from pathlib import Path
import numpy as np

# Derived from this file's own location (2026-08-12): the absolute hardcode silently
# escaped a fresh clone back to the production tree.  NB: run these tools from their
# in-repo location (the chains do); a /tmp copy would derive the wrong root.
REPO = Path(__file__).resolve().parents[2]
# 2026-08-05 PHASE RESET: kit v9 is mandatory (linear sps, FOV/body cells excluded).
T1_KIT = REPO / "Track1/realpde_t1_starting_kit_v9"
T2_KIT = REPO / "Track2/realpde_t2_starting_kit_v4"
T1U = REPO / "Track1/U-net"

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="7")
ap.add_argument("--v7-calib", default=str(T1U / "submissions/t1_unet_v7/calib.npz"))
ap.add_argument("--hetero", default=str(T1U / "calib_sps_hetero.npz"))
ap.add_argument("--out-calib", default=str(T1U / "calib_v8.npz"))
ap.add_argument("--checkpoint", default=str(T1U / "checkpoints/unet_v3_rel_tke20_s1.pth"))
ap.add_argument("--tpl-v3", default=str(T1U / "submission_t1_v3.py"))
ap.add_argument("--tpl-v4", default=str(T1U / "submission_t1_v4.py"))
ap.add_argument("--chunk", type=int, default=128)      # v7's verify_report.json
ap.add_argument("--halfwidth", type=float, default=0.15)
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--json", default="/tmp/t1_v8_verify.json")
A = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(A.gpu)

sys.path.insert(0, str(T1U))
builder = importlib.util.spec_from_file_location("t1_builder", T1U / "build_t1_bundle.py")
BLD = importlib.util.module_from_spec(builder); builder.loader.exec_module(BLD)
official = BLD._load_module(T1_KIT / "scoring.py", "t1_scoring")
import torch


def _band_from_calib(cal, key):
    """Read a band under either name; `_phys` is the historical calib_v9 spelling, the bare name is
    what t1_band_v5.py writes. Both are the PHYSICAL band, so this is a name and not a space."""
    import numpy as _np
    for k in (key + "_phys", key):
        if k in cal:
            a = _np.asarray(cal[k], _np.float32)
            if a.ndim != 2:
                raise SystemExit(f"calib['{k}'] has shape {a.shape}; expected (n_t, C)")
            return a
    raise SystemExit(f"calib has neither '{key}_phys' nor '{key}'; keys are {sorted(cal)}")


# ============================================================================ 1. merge the calib
v7 = dict(np.load(A.v7_calib, allow_pickle=True))
hx = dict(np.load(A.hetero, allow_pickle=True))
print(f"# v7 calib keys     {sorted(v7)}", flush=True)
print(f"# hetero calib keys {sorted(hx)}", flush=True)

assert np.array_equal(np.asarray(v7["gamma"], np.float32), np.asarray(hx["gamma"], np.float32)), \
    "the study used a different gamma than v7 ships; the whole comparison would be confounded"
for k in ("band_lo_phys", "band_hi_phys"):
    assert np.allclose(v7[k], hx[k], atol=0, rtol=1e-6), f"{k} disagrees between v7 and the study"

_, _, _, std_t = torch.load(T2_KIT / "example_data/mean_std_real.pt", map_location="cpu",
                            weights_only=False)
std_t = np.asarray(std_t, np.float32).ravel()
BL, BH = np.asarray(v7["band_lo"], np.float32), np.asarray(v7["band_hi"], np.float32)
BLP, BHP = _band_from_calib(v7, "band_lo"), _band_from_calib(v7, "band_hi")
nc_phys = BLP.shape[1]
# THE UNIT ASSERTION. Everything the study fitted lives in physical units; the bundle applies the
# band in normalised units. These are the same band only if this holds.
err_lo = np.abs(BL[:, :nc_phys] * std_t[:nc_phys] - BLP).max()
err_hi = np.abs(BH[:, :nc_phys] * std_t[:nc_phys] - BHP).max()
tol = 1e-6 * max(np.abs(BLP).max(), np.abs(BHP).max())
print(f"# std_t {std_t}   band_lo*std_t vs band_lo_phys: max|d| lo {err_lo:.3e} hi {err_hi:.3e}  "
      f"(tol {tol:.3e})", flush=True)
assert err_lo <= tol and err_hi <= tol, \
    "band_lo * std_t != band_lo_phys -- k_cell and act_const were fitted against a different scale"

nc_full = BL.shape[1]
kc = np.ones((BL.shape[0], nc_full), np.float32)
ac = np.ones((nc_full,), np.float32)
kc[:, :nc_phys] = np.asarray(hx["k_cell"], np.float32)
ac[:nc_phys] = np.asarray(hx["act_const"], np.float32)
merged = dict(gamma=np.asarray(v7["gamma"], np.float32), band_lo=BL, band_hi=BH,
              band_lo_phys=BLP, band_hi_phys=BHP, k_cell=kc, act_const=ac,
              p=np.float64(np.asarray(hx["p"]).ravel()[0]),
              lam=np.float64(np.asarray(hx["lam"]).ravel()[0]),
              self_norm=np.bool_(np.asarray(hx["self_norm"]).ravel()[0]),
              act_kind=np.asarray(hx["act_kind"]), fit_kind=np.asarray(v7.get("fit_kind", "unstamped (pre-t1_band_v5 calib)")))
np.savez(A.out_calib, **merged)
print(f"# wrote {A.out_calib}", flush=True)
print(f"#   k_cell padded to {kc.shape}: velocity cols [{kc[:, :nc_phys].min():.3f},"
      f"{kc[:, :nc_phys].max():.3f}], pressure col fixed at {kc[:, -1].min():.3f}", flush=True)
print(f"#   act_const {ac}   p={float(merged['p'])}  lam={float(merged['lam'])}  "
      f"self_norm={bool(merged['self_norm'])}  act_kind={merged['act_kind']}", flush=True)

# ============================================================================ 2. run both bundles
X, Y = BLD.load_val()
c = int(official.measured_channels(Y))
print(f"\n# local benchmark X{X.shape} Y{Y.shape} measured_channels={c}", flush=True)




def drive(tag, template, calib, repeats):
    """Materialise, import the PACKAGED code, run it `repeats` times, keep the last output."""
    d = T1U / "submissions" / f"_v8chk_{tag}"
    BLD.materialise(d, Path(A.checkpoint), calib, Path(template), A.halfwidth, A.chunk)
    for stale in ("participant_t1", "unet_model"):
        sys.modules.pop(stale, None)
    sys.path.insert(0, str(d))
    try:
        mod = BLD._load_module(d / "submission.py", "participant_t1")
        ts, out = [], None
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = mod.predict(X)
            ts.append(time.perf_counter() - t0)
        fast = bool(getattr(mod, "_FAST", False))
        has_k = getattr(mod, "_KCELL", None) is not None
    finally:
        sys.path.remove(str(d))
        shutil.rmtree(d, ignore_errors=True)
    P = np.asarray(out["prediction"], np.float32)
    L = np.asarray(out["lower"], np.float32)
    U = np.asarray(out["upper"], np.float32)
    return P, L, U, np.array(ts) / X.shape[0], fast, has_k


P7, L7, U7, t7, fast7, k7 = drive("v7", A.tpl_v3, A.v7_calib, A.repeats)
P8, L8, U8, t8, fast8, k8 = drive("v8", A.tpl_v4, A.out_calib, A.repeats)
print(f"# v7 packaged: fast path {fast7}, hetero band loaded {k7}  (must be False)", flush=True)
print(f"# v8 packaged: fast path {fast8}, hetero band loaded {k8}  (must be True)", flush=True)
assert not k7, "the v3 template somehow loaded a hetero band"
assert k8, ("v4 silently demoted to the v3 band -- the calib keys or the shape guards are wrong. "
            "This is exactly the failure this script exists to catch.")

# ---------------------------------------------------------- the difference claim, tested exactly
same_pred = np.array_equal(P7, P8)
print(f"\n# prediction arrays bit-identical: {same_pred}", flush=True)
if not same_pred:
    d = np.abs(P7.astype(np.float64) - P8.astype(np.float64))
    print(f"#   max|d| {d.max():.3e} at {np.unravel_index(d.argmax(), d.shape)}, "
          f"{int((d > 0).sum())} of {d.size} elements differ", flush=True)
assert same_pred, "v8 changed the forecast; it was only supposed to change the interval"
band7 = float((U7 - L7).mean())
band8 = float((U8 - L8).mean())
print(f"# mean interval width: v7 {band7:.6f}  v8 {band8:.6f}  ratio {band8/band7:.4f}", flush=True)
print(f"# v8 width spread across elements: p10 {np.percentile(U8-L8, 10):.6f}  "
      f"p50 {np.percentile(U8-L8, 50):.6f}  p90 {np.percentile(U8-L8, 90):.6f}   "
      f"(v7 is constant per lead time, so its spread is only the (t,ch) structure)", flush=True)
assert np.isfinite(L8).all() and np.isfinite(U8).all(), "non-finite bounds -> hard error on platform"
assert (L8 <= U8).all(), "lower > upper -> hard error on platform"


def score(P, L, U, ms):
    s = {"rel_l2_score": official.score_error(float(np.mean(official.rel_l2_per_sample(P, Y, c)))),
         "tke_score": official.score_error(float(np.mean(official.tke_rel_l2_per_sample(P, Y, c)))),
         "mvpe_score": official.score_error(official.mvpe_rel_l2(P, Y)),
         "time_score": official.score_time(ms)}
    sps, cov = official.aggregate_sps(P, Y, c, lower=L, upper=U)
    s["sps_score"] = float(official.score_sps(sps))
    s["FINAL"] = float(np.mean([s[k] for k in ("rel_l2_score", "tke_score", "mvpe_score",
                                               "time_score", "sps_score")]))
    s["coverage"] = float(cov)
    return s


m7, m8 = float(np.median(t7)), float(np.median(t8))
s7, s8 = score(P7, L7, U7, m7), score(P8, L8, U8, m8)
# The same accuracy at a common time, to separate "the band earned it" from "the clock moved".
s8_fixed = score(P8, L8, U8, m7)

print(f"\n# per-window seconds over {A.repeats} repeats", flush=True)
print(f"#   v7 median {m7*1e3:.4f} ms   all {np.round(t7*1e3, 4)}", flush=True)
print(f"#   v8 median {m8*1e3:.4f} ms   all {np.round(t8*1e3, 4)}", flush=True)
print(f"#   marginal cost of the heteroscedastic band: {(m8-m7)*1e3:+.4f} ms/window "
      f"({100*(m8-m7)/m7:+.2f}%)", flush=True)

hdr = f"{'bundle':<26}{'rel_l2':>9}{'tke':>9}{'mvpe':>9}{'time':>9}{'sps':>9}{'FINAL':>10}{'cov':>8}"
print("\n" + hdr + "\n" + "-" * len(hdr), flush=True)
for nm, s in (("v7 (shipped)", s7), ("v8 (hetero band)", s8),
              ("v8 @ v7's clock", s8_fixed)):
    print(f"{nm:<26}{s['rel_l2_score']:>9.4f}{s['tke_score']:>9.4f}{s['mvpe_score']:>9.4f}"
          f"{s['time_score']:>9.4f}{s['sps_score']:>9.4f}{s['FINAL']:>10.4f}{s['coverage']:>8.4f}",
          flush=True)
print(f"\n# dFINAL v8 - v7            {s8['FINAL'] - s7['FINAL']:+.4f}", flush=True)
print(f"#   of which sps            {(s8['sps_score'] - s7['sps_score'])/5:+.4f}", flush=True)
print(f"#   of which time           {(s8['time_score'] - s7['time_score'])/5:+.4f}", flush=True)

assert s8["sps_score"] > s7["sps_score"] + 0.3, \
    f"sps moved only {s8['sps_score'] - s7['sps_score']:+.4f}; expected about +0.73"

json.dump(dict(v7=s7, v8=s8, v8_at_v7_clock=s8_fixed, ms_v7=m7 * 1e3, ms_v8=m8 * 1e3,
               t7=list(t7 * 1e3), t8=list(t8 * 1e3), same_prediction=bool(same_pred),
               width_v7=band7, width_v8=band8, calib=A.out_calib),
          open(A.json, "w"), indent=1, default=float)
print("\nT1_V8_VERIFY_OK", flush=True)
