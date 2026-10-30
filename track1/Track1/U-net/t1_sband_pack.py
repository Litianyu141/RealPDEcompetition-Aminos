#!/usr/bin/env python3
"""STAGE D -- pack the learned band into calib.npz, sweep the global width multiplier, and
score every arm through build_t1_bundle.py's OWN materialise/run_and_score (the authoritative
mirror), including a same-session rebuild of the incumbent on the same 378 windows.

UNITS.  The table is fitted in PHYSICAL offsets (SIGMA_GLOBAL is physical), and divided by std_t
ONCE here at pack time to become the NORMALISED offsets submission.py adds to `pn`.  The identity
`sb_lo * std_t == sb_lo_phys` is asserted after the division -- unlike the incumbent's band, whose
multiplier stage (an out-of-repo heredoc) broke exactly that identity.
"""
import os, sys, json, time, shutil, argparse
from pathlib import Path
import numpy as np, torch

# Derived from this file's own location (2026-08-12): the absolute hardcode silently
# escaped a fresh clone back to the production tree.  NB: run these tools from their
# in-repo location (the chains do); a /tmp copy would derive the wrong root.
REPO = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO / "common"))
from fivefold_heads import (HeadPlan, verify_cache_metadata, strict_selftest_inputs,
                            neutral_t1_base)
T1U = REPO / "Track1/U-net"
KIT = REPO / "Track1/realpde_t1_starting_kit_v9"
import os as _os_ca
# 2026-08-10: honour SBAND_CACHE so a head fitted on a DIFFERENT fleet is packed against
# ITS OWN cache. Default is the value that was hardcoded here, so the shipped path is
# bit-identical.
CA = Path(_os_ca.environ.get("SBAND_CACHE", str(Path(__file__).resolve().parent / "sband_cache")))
OUT = T1U / "sband"
sys.path.insert(0, str(KIT)); import scoring
sys.path.insert(0, str(T1U))
from sband_head import build_from_blob, sband_features
import importlib.util
_sp = importlib.util.spec_from_file_location("btb", T1U / "build_t1_bundle.py")
btb = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(btb)

SIG = float(scoring.SIGMA_GLOBAL)
H, W, LT, C2 = 32, 64, 20, 2
DEV = "cuda"
ap = argparse.ArgumentParser()
ap.add_argument("--head", default="ship_s0")
ap.add_argument("--mults", default="0.55,0.70,0.85,1.00,1.15,1.30,1.50,1.75,2.00")
ap.add_argument("--base-calib", default=str(T1U / "rsa/calib_rsa_m1185.npz"))
ap.add_argument("--q", type=int, default=32)
ap.add_argument("--out-tag", default="")
ap.add_argument("--cache", default=str(CA))
ap.add_argument("--out-dir", default=str(OUT))
ap.add_argument("--head-file", default="", help="explicit fitted head path (otherwise cache/sband_<head>.pth)")
ap.add_argument("--strict-plan", default="", help="strict fitting cache; neutral fallback base; train-only selftest")
# 2026-08-10: these were hard-coded to the RSa family, which is the ONLY reason a faithful
# Decision-Phase rebuild needs the 18 extra RSa training runs.  Defaults preserve the shipped path.
ap.add_argument("--inc-ckpt", default=str(REPO / "checkpoints/t1_u-net/unet_v4_ship_RSa.pth"))
ap.add_argument("--prov-ckpt", default="",
                help="checkpoint to STAMP the calibs to (default: --inc-ckpt). Set this "
                     "when the bundle ships a different primary member than the one "
                     "scored here, e.g. a blend whose model.pth is member 1.")
ap.add_argument("--inc-head", default=str(REPO / "checkpoints/t1_u-net/vgain_head_ship_RSa.pth"))
# AWB (2026-08-12): optional replacement of the per-fleet afac weights, plus a fit-only
# mode.  Default "" keeps the shipped code path byte-for-byte.
ap.add_argument("--a-weight", default="",
                help="path to a (N,) .npy of per-window weights used for BOTH fleets in "
                     "place of the cache afac columns ('' = shipped behaviour)")
ap.add_argument("--fit-only", action="store_true",
                help="emit table+calibs only, skip the internal score sweep (its scorer "
                     "mirror is 0.156 sps off build_t1_bundle.py -- score candidates by "
                     "building real bundles)")
a = ap.parse_args()
CA, OUT = Path(a.cache), Path(a.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
plan = HeadPlan(a.strict_plan) if a.strict_plan else None
if plan:
    if not a.fit_only or a.a_weight or a.mults != "1.00":
        raise ValueError("strict packing requires --fit-only --mults 1.00 and no external --a-weight")
    arms = os.environ.get('SBAND_FLEET_A', '').split(',') + os.environ.get('SBAND_FLEET_B', '').split(',')
    if any(not arm for arm in arms):
        raise ValueError("strict packing requires SBAND_FLEET_A and SBAND_FLEET_B")
    strict_meta = verify_cache_metadata(CA, plan, arms,
                                       os.environ['SBAND_FLEET_A'].split(','),
                                       os.environ['SBAND_FLEET_B'].split(','))
Q = int(a.q)
OT = a.out_tag or f"{a.head}_q{Q}"

STD = np.array([0.09681041, 0.01596364], np.float64)      # mean_std_real.pt, u/v
INC_CK = str(a.inc_ckpt)
INC_HEAD = str(a.inc_head)
for _q in (INC_CK, INC_HEAD):
    import os as _os_p
    if not _os_p.path.exists(_q):
        raise SystemExit(f"missing incumbent artifact: {_q}")
print(f"# incumbent: {INC_CK}\n#            {INC_HEAD}", flush=True)
V33, V36 = T1U / "submission_t1_v33.py", T1U / "submission_t1_v36.py"

# ----------------------------------------------------------------- 1. head -> scale on the cache
HEAD_PATH = Path(a.head_file) if a.head_file else CA / f"sband_{a.head}.pth"
blob = torch.load(HEAD_PATH, map_location="cpu", weights_only=False)
if plan and (blob.get('strict_plan_stamp') != strict_meta['strict_plan_stamp'] or
             set(blob.get('fit_files', [])) != set(plan.fitting_files)):
    raise ValueError("strict sband head/cache plan mismatch")
net = build_from_blob(blob).to(DEV).eval()
# what the packaged bundle will ship as model.pth -- the thing the band must match
PROV_CK = str(a.prov_ckpt or INC_CK)
_HEAD_FIT_KIND = str(blob.get("fit_kind", "?"))
print(f"# [prov] stamping calibs to {_os_ca.path.basename(PROV_CK)}"
      f"\n#        head fit_kind: {_HEAD_FIT_KIND}", flush=True)
Xp = np.load(CA / "Xp.npy", mmap_mode="r")
N = Xp.shape[0]
S = np.empty((N, LT, H, W, C2), np.float32)
with torch.no_grad():
    for i in range(0, N, 32):
        j = min(i + 32, N)
        f = net(sband_features(torch.from_numpy(np.asarray(Xp[i:j])).to(DEV)))
        S[i:j] = f.reshape(-1, LT, C2, H, W).permute(0, 1, 3, 4, 2).cpu().numpy()
print(f"# head {a.head}: score range [{S.min():.3f},{S.max():.3f}] mean {S.mean():.3f}", flush=True)

Rkf = np.load(CA / "Rkf.npy", mmap_mode="r"); Rks = np.load(CA / "Rks.npy", mmap_mode="r")
SC = np.load(CA / "SC.npy", mmap_mode="r"); AF = np.load(CA / "afac.npy")
CENTERS = [.35, .40, .45, .50, .55, .60, .65]


def fit_table(Slist, Rlist, Mlist, Alist, nsub=80000, rng=0):
    g = np.random.default_rng(rng)
    edges = np.zeros((LT, C2, Q - 1), np.float64)
    tab = np.zeros((LT, C2, Q, 2), np.float64)
    for t in range(LT):
        for c in range(C2):
            s = np.concatenate([Sx[:, t, :, :, c][Mx[:, t, :, :, c]] for Sx, Mx in zip(Slist, Mlist)]).astype(np.float64)
            r = np.concatenate([Rx[:, t, :, :, c][Mx[:, t, :, :, c]] for Rx, Mx in zip(Rlist, Mlist)]).astype(np.float64)
            w = np.concatenate([np.broadcast_to(Ax.reshape(-1, 1, 1), (Ax.size, H, W))[Mx[:, t, :, :, c]]
                                for Ax, Mx in zip(Alist, Mlist)]).astype(np.float64)
            e = np.maximum.accumulate(np.quantile(s, np.linspace(0, 1, Q + 1)[1:-1]))
            edges[t, c] = e
            b = np.searchsorted(e, s, side="right")
            for k in range(Q):
                sel = np.flatnonzero(b == k)
                if sel.size < 8:
                    tab[t, c, k] = (-0.01, 0.01); continue
                if sel.size > nsub:
                    sel = g.choice(sel, nsub, replace=False)
                rr = r[sel]; ww = w[sel]
                best = (-1.0, -0.01, 0.01)
                for cc in [0.0] + list(np.quantile(rr, CENTERS)):
                    d = np.abs(rr - cc); o = np.argsort(d)
                    ds = d[o]; cw = np.cumsum(ww[o])
                    val = np.exp(-2.0 * ds / SIG) * cw
                    q = int(np.argmax(val))
                    if val[q] > best[0]:
                        best = (val[q], cc - ds[q], cc + ds[q])
                tab[t, c, k] = (best[1], best[2])
    return edges, tab


t0 = time.time()
# POOLED over both fleets: the same head score, both models' residuals in the same bin.
if a.a_weight:
    _AW = np.load(a.a_weight).astype(np.float32)
    assert _AW.shape == (N,), f"--a-weight shape {_AW.shape} != ({N},)"
    assert np.isfinite(_AW).all() and (_AW > 0).all(), "--a-weight must be finite, > 0"
    print(f"# AWB weights {a.a_weight}: mean {_AW.mean():.4f} "
          f"range [{_AW.min():.4f},{_AW.max():.4f}]", flush=True)
    _W0 = _W1 = _AW
else:
    _W0, _W1 = AF[:, 0], AF[:, 1]
edges, tab = fit_table([S, S], [Rkf, Rks], [SC, SC], [_W0, _W1])
print(f"# table fitted in {time.time()-t0:.0f}s  width range "
      f"[{(tab[...,1]-tab[...,0]).min():.5f},{(tab[...,1]-tab[...,0]).max():.5f}] phys", flush=True)
np.savez(OUT / f"table_{OT}.npz", edges=edges, tab=tab)

if plan:
    X = strict_selftest_inputs(Xp)
    Y = official = None
else:
    X, Y = btb.load_val()
    official = btb._load_module(KIT / "scoring.py", "t1_scoring")
    print(f"# builder benchmark X{X.shape}", flush=True)
    assert 0.05 < float((Y[..., :2] == 0).mean()) < 0.30

# numpy mirror of the template's lookup, used ONLY to predict selftest_mean_nil in advance
with torch.no_grad():
    f8 = net(sband_features(torch.from_numpy(X[:8]).to(DEV)))
S8 = f8.reshape(-1, LT, C2, H, W).permute(0, 1, 3, 4, 2).cpu().numpy()


def nil8(tb, mult):
    tot = 0.0
    for t in range(LT):
        for c in range(C2):
            b = np.searchsorted(edges[t, c], S8[:, t, :, :, c].astype(np.float64), side="right")
            tot += float(np.sum((tb[t, c, b, 1] - tb[t, c, b, 0]) * mult)) / SIG
    return tot / (len(X[:8]) * LT * H * W * C2)


def write_calib(path, tb, mult):
    if plan:
        if Path(path).exists():
            raise ValueError("strict calibration output already exists; use a fresh tag")
        # Never import learned fallback fields from an all81/ship72 calibration.
        d = neutral_t1_base()
    else:
        with np.load(a.base_calib, allow_pickle=True) as z:
            d = {k: z[k] for k in z.files}
    lo_p = (tb[..., 0] * mult).astype(np.float64)
    hi_p = (tb[..., 1] * mult).astype(np.float64)
    lo_n = (lo_p / STD.reshape(1, C2, 1)).astype(np.float32)
    hi_n = (hi_p / STD.reshape(1, C2, 1)).astype(np.float32)
    # the ONE unit conversion in the chain, asserted immediately after it is made
    assert np.allclose(lo_n.astype(np.float64) * STD.reshape(1, C2, 1), lo_p, rtol=1e-5, atol=1e-12)
    assert np.allclose(hi_n.astype(np.float64) * STD.reshape(1, C2, 1), hi_p, rtol=1e-5, atol=1e-12)
    assert (hi_n >= lo_n).all()
    d.update(sb_edge=edges.astype(np.float32), sb_lo=lo_n, sb_hi=hi_n,
             sb_lo_phys=lo_p.astype(np.float32), sb_hi_phys=hi_p.astype(np.float32),
             sb_mult=np.float64(mult), sband_required=np.bool_(True),
             sb_head=str(a.head), sb_q=np.int64(Q),
             selftest_mean_nil=np.float64(nil8(tb, mult)))
    if plan:
        d.update(strict_plan_stamp=np.str_(strict_meta['strict_plan_stamp']),
                 fit_files=np.asarray(plan.fitting_files), fallback_policy=np.str_('neutral'))
    # PROVENANCE (2026-08-19).  T1 shipped calibs carried ZERO prov_* keys, so
    # build_t1_bundle._prov_check always took its no-stamp warn-and-return branch and
    # T1's gate was inert while T2's passed by matching.  Stamp the checkpoint that
    # materialise() will package as model.pth, and record the head's OWN fit_kind (which
    # now names the real fleets) as the fit stream.  This also OVERWRITES any stamp
    # inherited from --base-calib: a copied stale stamp is worse than no stamp.
    btb._prov_stamp(d, ckpt=PROV_CK,
                    fit_stream=f"sband head {a.head} | {_HEAD_FIT_KIND} | cache {CA}",
                    producer=f"t1_sband_pack.py --head {a.head} --out-tag {OT} m={mult}")
    np.savez(path, **d)
    return d["selftest_mean_nil"]


def score_arm(name, calib, template, sband):
    tmp = T1U / "submissions" / f"_sb_{name}_{os.getpid()}"
    btb.materialise(tmp, Path(INC_CK), str(calib) if calib else None, Path(template),
                    0.15, 64, INC_HEAD, sband=str(sband) if sband else "")
    sub, ex = btb.run_and_score(tmp, X, Y, official, use_bounds=True)
    shutil.rmtree(tmp, ignore_errors=True)
    fin = float(np.mean(list(sub.values())))
    print(f"{name:26s} rel {sub['rel_l2_score']:7.3f} tke {sub['tke_score']:7.3f} "
          f"mvpe {sub['mvpe_score']:7.3f} time {sub['time_score']:7.3f} sps {sub['sps_score']:7.3f} "
          f"| FINAL {fin:7.3f} cov {ex['coverage']:.4f} | sband {ex['sband_status']} "
          f"kcell {ex['kcell_live']} t {ex['mean_t_neural_s']*1e3:.3f}ms", flush=True)
    return {"name": name, **{k: round(v, 4) for k, v in sub.items()}, "FINAL": round(fin, 4),
            "coverage": round(ex["coverage"], 4), "mean_t_ms": round(ex["mean_t_neural_s"] * 1e3, 4),
            "sband_status": ex["sband_status"], "kcell_live": ex["kcell_live"]}


ROWS = []
if os.environ.get("SB_SKIP_INC") != "1" and not a.fit_only:
    print("\n=== INCUMBENT, REBUILT THIS SESSION ON THESE 378 WINDOWS ===", flush=True)
    ROWS.append(score_arm("incumbent_v35_m1185", a.base_calib, V33, ""))
print("\n=== LEARNED BAND: GLOBAL WIDTH MULTIPLIER SWEEP ===", flush=True)
SB = HEAD_PATH
for m in [float(x) for x in a.mults.split(",")]:
    cp = OUT / f"calib_sband_{OT}_m{int(round(m*100)):04d}.npz"
    nl = write_calib(cp, tab, m)
    if a.fit_only:
        print(f"# fit-only: wrote {cp.name}  selftest_mean_nil {float(nl):.9f}", flush=True)
        continue
    r = score_arm(f"sband_m{m:.2f}", cp, V36, SB)
    r["mult"] = m; r["selftest_mean_nil"] = float(nl)
    ROWS.append(r)

json.dump(ROWS, open(OUT / f"sweep_{OT}.json", "w"), indent=1)
print("\nSBD_OK", flush=True)
