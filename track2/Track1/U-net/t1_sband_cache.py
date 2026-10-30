#!/usr/bin/env python3
"""STAGE A -- cache the two-fleet OOF residual field for the learned residual-scale band head.

UNIT CONVENTION, ASSERTED AT EVERY BOUNDARY (this is the defence against the normalisation trap
that has been hit twice on this project):
  * tv4.build_windows returns NORMALISED windows.  The MODEL is fed the NORMALISED array.
  * Everything cached to disk is PHYSICAL (x*std + mean).
  * Fingerprint asserted below: the PHYSICAL target's exact-zero fraction is ~0.10 (mask/FOV
    cells); a normalised target would have ~0.  And mean|r| lands at 0.0033-0.0041.
"""
import os, sys, json, time, importlib.util, argparse
from pathlib import Path
import numpy as np, torch

# The two same-recipe / different-seed OOF fleets whose |r| geometric mean is the target.
# Defaults are the values that were hardcoded here until 2026-08-09, so the shipped path is
# unchanged; override with --fleet-a / --fleet-b to fit a head for a DIFFERENT forecaster.
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
FLEET_A = _sband_fleet("SBAND_FLEET_A")
FLEET_B = _sband_fleet("SBAND_FLEET_B")

# Derived from this file's own location (2026-08-12): the absolute hardcode silently
# escaped a fresh clone back to the production tree.  NB: run these tools from their
# in-repo location (the chains do); a /tmp copy would derive the wrong root.
REPO = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO / "common"))
from repo_paths import resolve_paths   # configs and split manifests carry repo-relative paths
from fivefold_heads import HeadPlan, sha256_file, sband_stamp
KIT = REPO / "Track1/realpde_t1_starting_kit_v9"
T1U = REPO / "Track1/U-net"
import os as _os_c
# Cache directory.  Default is the value hardcoded here until 2026-08-09, so the shipped
# path is unchanged; override with SBAND_CACHE to fit a head for a DIFFERENT fleet without
# overwriting the cache the shipped head was fitted from.
# Repo-relative default (2026-08-12, RPXa): the old host-scratch default meant a clean clone
# silently pointed at a cache that does not exist off this box.
OUT = Path(_os_c.environ.get("SBAND_CACHE", str(Path(__file__).resolve().parent / "sband_cache")))
ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=str(REPO / "common/local_benchmark/splits_kfold_dedup.json"))
ap.add_argument("--ckpt-dir", default=str(REPO / "checkpoints/t1_u-net"))
ap.add_argument("--cache", default=str(OUT))
ap.add_argument("--strict-plan", default="")
a = ap.parse_args()
OUT = Path(a.cache)
plan = HeadPlan(a.strict_plan) if a.strict_plan else None
arms = FLEET_A.split(',') + FLEET_B.split(',')
strict_stamp = sband_stamp(plan, FLEET_A.split(','), FLEET_B.split(',')) if plan else ""
if plan and any(OUT.glob('*.npy')):
    raise ValueError("strict cache output already exists; use a fresh output directory")
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(KIT)); import scoring
sys.path.insert(0, str(T1U)); from unet_model import build_from_meta
spec = importlib.util.spec_from_file_location("tv4", T1U / "train_v4.py")
tv4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(tv4)


class _N:
    def info(self, *a, **k): pass
    warning = debug = error = info


DEV = "cuda"
BENCH = ("11400_", "17775_")          # the 378-window builder benchmark -- HELD OUT of the head
man = resolve_paths(dict(plan.manifest, folds=plan.folds) if plan else json.load(open(a.manifest)))
if plan:
    man['stats_real'] = str(REPO / 'Track2/realpde_t2_starting_kit_v4/example_data/mean_std_real.pt')
mean, std = tv4.load_stats(man["stats_real"])
m5 = np.asarray(mean, np.float32).reshape(1, 1, 1, 1, -1)
s5 = np.asarray(std, np.float32).reshape(1, 1, 1, 1, -1)
print("# std", s5.ravel(), flush=True)


def fwd(ck, Xn):
    d = torch.load(ck, map_location="cpu", weights_only=False)
    m = build_from_meta(d.get("meta", {}), 20, 20, 3).to(DEV).eval()
    m.load_state_dict(d["model_state_dict"], strict=True)
    out = []
    with torch.no_grad():
        for i in range(0, Xn.shape[0], 32):
            out.append(m(torch.from_numpy(Xn[i:i + 32]).to(DEV)).float().cpu().numpy())
    del m; torch.cuda.empty_cache()
    return (np.concatenate(out, 0) * s5 + m5).astype(np.float32)[..., :2]


sq = lambda e: e / (0.5 + e)
def afac(P, T):
    P = np.asarray(P, np.float64); T = np.asarray(T, np.float64)
    return (0.5 * (1 - sq(scoring.rel_l2_per_sample(P, T, 2)))
            + 0.3 * (1 - sq(scoring.tke_rel_l2_per_sample(P, T, 2)))
            + 0.2 * (1 - sq(scoring.mvpe_rel_l2_per_sample(P, T)))).astype(np.float32)


XPS, RKF, RKS, GRP, AKF, AKS, FIDS, SCS = [], [], [], [], [], [], [], []
t0 = time.time()
for fd in man["folds"]:
    k = int(fd["fold"])
    vf = sorted(str(f) for f in fd["val_files"]
                if plan or not any(b in str(f) for b in BENCH))
    if not vf:
        print(f"# fold {k}: 0 non-benchmark files", flush=True); continue
    Xn, Yn = tv4.build_windows(vf, 2, 20, 20, mean, std, 20, _N(), f"f{k}", 1)
    Xp = (Xn * s5 + m5).astype(np.float32)[..., :2]
    Tp = (Yn * s5 + m5).astype(np.float32)[..., :2]
    zf = float((Tp == 0).mean())
    assert 0.05 < zf < 0.30, f"PHYSICAL-target fingerprint failed: zero-frac {zf}"
    def _fleet(stems, kk):
        # comma-separated stems are AVERAGED -- an ensemble member set, not two seeds.
        # one stem is bit-identical to the historical path (sum of one, divided by one).
        _ps = [fwd(plan.checkpoint(_st, kk) if plan else Path(a.ckpt_dir) / f"{_st}_f{kk}.pth", Xn)
               for _st in stems.split(",")]
        return _ps[0] if len(_ps) == 1 else sum(_ps) / float(len(_ps))
    Pkf = _fleet(FLEET_A, k)
    Pks = _fleet(FLEET_B, k)
    rkf, rks = Tp - Pkf, Tp - Pks
    mk = np.abs(rkf).mean(); ms = np.abs(rks).mean()
    if plan:
        assert np.isfinite(rkf).all() and np.isfinite(rks).all() and mk > 0 and ms > 0, f"invalid residuals {mk} {ms}"
    else:
        assert 0.0015 < mk < 0.010 and 0.0015 < ms < 0.010, f"|r| fingerprint {mk} {ms}"
    XPS.append(Xp); RKF.append(rkf); RKS.append(rks); SCS.append(Tp != 0.0)
    AKF.append(afac(Pkf, Tp)); AKS.append(afac(Pks, Tp))
    GRP.append(np.full(Xp.shape[0], k, np.int16))
    FIDS.append([Path(f).name for f in vf])
    print(f"# fold {k}: {len(vf)} files {Xp.shape[0]} win  zf {zf:.4f} "
          f"|r|kf {mk:.6f} |r|kfs1 {ms:.6f}  ({time.time()-t0:.0f}s)", flush=True)

Xp = np.concatenate(XPS, 0); del XPS
Rkf = np.concatenate(RKF, 0); del RKF
Rks = np.concatenate(RKS, 0); del RKS
grp = np.concatenate(GRP, 0)
akf = np.concatenate(AKF, 0); aks = np.concatenate(AKS, 0)
SC = np.concatenate(SCS, 0); del SCS
print(f"# TOTAL {Xp.shape[0]} windows over {sum(len(f) for f in FIDS)} files "
      f"scored-frac {float(SC.mean()):.4f}", flush=True)
np.save(OUT / "Xp.npy", Xp)
np.save(OUT / "Rkf.npy", Rkf)
np.save(OUT / "Rks.npy", Rks)
np.save(OUT / "grp.npy", grp)
np.save(OUT / "SC.npy", SC)
np.save(OUT / "afac.npy", np.stack([akf, aks], 1))
json.dump({"files": FIDS, "n": int(Xp.shape[0]), "std": s5.ravel().tolist(),
           "mean": m5.ravel().tolist(),
           **({"strict_plan_stamp": strict_stamp, "strict_plan": str(plan.path),
               "fleet_a": FLEET_A.split(','), "fleet_b": FLEET_B.split(','),
               "array_sha256": {p.name: sha256_file(p) for p in OUT.glob('*.npy')}} if plan else {})},
          open(OUT / "meta.json", "w"), indent=1)
print("SBA_OK", flush=True)
