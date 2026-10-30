#!/usr/bin/env python3
"""L2 T1-SBAND2 step 2: the INNER-OOF residual cache of one outer Re fold, rebuilt from the fixed five-fold campaign's own
inner producers (checkpoints/fivefold_v1/<fold>/inner*/LIVE/*_f0_terminal_ema.pth), mirroring t1_sband_cache.py's strict
path (which needs a head_plan.json that is no longer available).  The isolation proof that the HeadPlan stamp carried is
re-derived here from Track1/U-net/VariantFiveFoldEvaluation/config/campaign.json and written to meta.json:

  for every producer job and every job in its transitive `depends` closure (teachers):
    real train files  are disjoint from (outer held-out files  U  this inner context's held-out files)
    sim  train files  carry no Reynolds group of the outer held-out or of the inner held-out
    the checkpoint's own saved config names the job ('FF_<jobid>'), and its sha256 is recorded
  the inner held-out files of the contexts partition the outer TRAINING pool exactly (no outer held-out file anywhere).

PRODUCERS.  The fold cards' own head was fitted on fleets A = H,SW and B = CW128s1,WSWs1, but on the two usable GPU hosts
(the hosts in use at the time) the re0 / re1 inner contexts hold only the four TEACHER producers WSW, MIMOWSW, MIMOA,
RLB (every re0 / re1 / re2 inner context has all four; re2 alone also has H, SW, CW128(s1), WSWs1).  The cache therefore
stores one signed residual per SINGLE producer (--roles, default the four teachers); a fleet residual is the mean of its
members' residuals (= the residual of the averaged prediction), formed at head-training time.
Arrays (PHYSICAL, float32): Xp (N,20,32,64,2) input; R_<role> signed residuals; SC scored mask; T (target); grp (N,) context.

    CUDA_VISIBLE_DEVICES=G nice -n 10 python Track1/U-net/VariantSPSBand2/sb2_cache.py --fold re0 --gpu G [--stride 20]
"""
import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
T1U = REPO / "Track1/U-net"
KIT = REPO / "Track1/realpde_t1_starting_kit_v9"
CAMP = REPO / "Track1/U-net/VariantFiveFoldEvaluation/config/campaign.json"
PROTO = REPO / "common/local_benchmark/splits_re3_aoa2_v1.json"
STATS = REPO / "Track2/realpde_t2_starting_kit_v4/example_data/mean_std_real.pt"
REAL = REPO / "data/RealPDE_competition_dataset/gdrive_raw/extracted/train_real/train_real"
ROLES = ("WSW", "MIMOWSW", "MIMOA", "RLB")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for b in iter(lambda: s.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def re_of(n):
    return int(Path(n).name.split("_")[0])


def isolation(camp, fold, roles):
    outer = [f for f in camp["folds"] if f["id"] == fold][0]
    ov = set(outer["val_files"]); ot = set(outer["train_files"]); ovg = set(int(g) for g in outer["val_groups"])
    ax = {"re": 0, "aoa": 1}[outer["axis"]]
    jobs = {j["id"]: j for j in camp["jobs"]}
    inner = [j for j in camp["jobs"] if j["outer_fold"] == fold and j["scope"] == "inner" and j["method"] == "LIVE"]
    ctx = sorted({j["context"] for j in inner})
    proof = dict(outer_fold=fold, outer_val_groups=sorted(ovg), n_outer_val=len(ov), n_outer_train=len(ot), contexts={})
    cover = []
    for cx in ctx:
        js = {j["role"]: j for j in inner if j["context"] == cx}
        iv = set(js["H"]["val_files"]); ivg = {re_of(n) for n in iv}
        assert iv <= ot and not (iv & ov), cx
        cover += sorted(iv)
        prods = {}
        for r in roles:
            if True:
                seen, stack, chk = set(), [js[r]["id"]], []
                while stack:
                    jid = stack.pop()
                    if jid in seen:
                        continue
                    seen.add(jid); j = jobs[jid]
                    tr = set(j["train_files"]); sim = set(j["sim_train_files"])
                    assert not (tr & ov) and not (tr & iv), ("real contamination", jid)
                    assert not ({int(Path(n).name[:-3].split("_")[ax]) for n in sim} & ovg), ("sim outer-group contamination", jid)
                    assert not ({re_of(n) for n in sim} & ivg), ("sim inner-group contamination", jid)
                    assert j["val_files"] == js["H"]["val_files"] and j["outer_fold"] == fold and j["context"] == cx, jid
                    for st in j["config"]["stages"]:
                        for s in st["sources"]:
                            kf = set(s["keep_files"])
                            assert kf == (tr if s["tag"] == "comp_real" else sim), ("keep_files", jid)
                    chk.append(jid); stack += j["depends"]
                prods[r] = dict(job=js[r]["id"], output=js[r]["output"], closure=sorted(chk))
        proof["contexts"][cx] = dict(val_files=sorted(iv), val_groups=sorted(ivg), producers=prods)
    assert sorted(cover) == sorted(ot) and len(cover) == len(set(cover)), "inner held-out files do not partition the outer training pool"
    proof["partition_of_outer_train"] = True
    return proof


def ship_proof(camp, roles):
    """SHIP cache: every one of the 81 files gets the residual of the Re-outer-fold teacher that held it out (OOF)."""
    jobs = {j["id"]: j for j in camp["jobs"]}
    proof = dict(outer_fold="ship", contexts={})
    cover = []
    for fo in [f for f in camp["folds"] if f["id"].startswith("re")]:
        ov = set(fo["val_files"]); cover += sorted(ov)
        prods = {}
        for r in roles:
            jid = "%s__outer__LIVE__%s" % (fo["id"], r); j = jobs[jid]
            seen, stack, chk = set(), [jid], []
            while stack:
                q = stack.pop()
                if q in seen:
                    continue
                seen.add(q); jj = jobs[q]
                assert not (set(jj["train_files"]) & ov), ("ship OOF contamination", q)
                chk.append(q); stack += jj["depends"]
            prods[r] = dict(job=jid, output=j["output"], closure=sorted(chk))
        proof["contexts"][fo["id"]] = dict(val_files=sorted(ov), val_groups=sorted(int(g) for g in fo["val_groups"]), producers=prods)
    assert len(cover) == len(set(cover)) == 81, len(cover)
    proof["partition_of_all81"] = True
    return proof


class _N:
    def info(self, *a, **k): pass
    warning = debug = error = info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True); ap.add_argument("--gpu", default="0")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--out", default="")
    ap.add_argument("--roles", default=",".join(ROLES))
    ap.add_argument("--ship", action="store_true", help="all81 OOF cache from the Re-outer-fold teachers (ship head)")
    ap.add_argument("--campaign", default=str(CAMP), help="same fixed fold/job plan with run-scoped output paths")
    ap.add_argument("--real-dir", default=str(REAL), help="released train_real directory")
    ap.add_argument("--stats", default=str(STATS), help="official real normalization file")
    ap.add_argument("--expected-variants", default="", help="JSON job-id -> saved variant for run-scoped final retraining")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    import torch
    sys.path.insert(0, str(KIT)); import scoring
    sys.path.insert(0, str(T1U)); from unet_model import build_from_meta
    spec = importlib.util.spec_from_file_location("tv4", T1U / "train_v4.py")
    tv4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(tv4)
    t00 = time.time()
    camp = json.loads(Path(a.campaign).read_text())
    real_dir = Path(a.real_dir)
    stats_path = Path(a.stats)
    expected_variants = json.loads(Path(a.expected_variants).read_text()) if a.expected_variants else {}
    roles = tuple(a.roles.split(","))
    proof = ship_proof(camp, roles) if a.ship else isolation(camp, a.fold, roles)
    out = Path(a.out or (REPO / ("_tmp/sband2/cache/%s_s%d" % (a.fold, a.stride))))
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "meta.json").exists(), "cache exists"
    mean, std = tv4.load_stats(str(stats_path))
    m5 = np.asarray(mean, np.float32).reshape(1, 1, 1, 1, -1); s5 = np.asarray(std, np.float32).reshape(1, 1, 1, 1, -1)
    ckmeta = {}

    def fwd(ck, Xn):
        d = torch.load(REPO / ck, map_location="cpu", weights_only=False)
        mdl = build_from_meta(d.get("meta", {}), 20, 20, 3).cuda().eval()
        mdl.load_state_dict(d["model_state_dict"], strict=True)
        o = []
        with torch.no_grad():
            for i in range(0, Xn.shape[0], 32):
                o.append(mdl(torch.from_numpy(Xn[i:i + 32]).cuda()).float().cpu().numpy())
        cfgv = None
        for k in ("config", "cfg", "args"):
            if isinstance(d.get(k), dict):
                cfgv = d[k].get("variant"); break
        if cfgv is None and isinstance(d.get("meta"), dict):
            cfgv = d["meta"].get("variant")
        del mdl; torch.cuda.empty_cache()
        return (np.concatenate(o, 0) * s5 + m5).astype(np.float32)[..., :2], cfgv

    sq = lambda e: e / (0.5 + e)

    def afac(P, T):
        P = np.asarray(P, np.float64); T = np.asarray(T, np.float64)
        return (0.5 * (1 - sq(scoring.rel_l2_per_sample(P, T, 2))) + 0.3 * (1 - sq(scoring.tke_rel_l2_per_sample(P, T, 2)))
                + 0.2 * (1 - sq(scoring.mvpe_rel_l2_per_sample(P, T)))).astype(np.float32)

    parts = []
    for gi, (cx, info) in enumerate(sorted(proof["contexts"].items())):
        vf = [str(real_dir / n) for n in info["val_files"]]
        Xn, Yn = tv4.build_windows(vf, 2, 20, 20, mean, std, a.stride, _N(), cx, 1)
        Xp = (Xn * s5 + m5).astype(np.float32)[..., :2]; Tp = (Yn * s5 + m5).astype(np.float32)[..., :2]
        zf = float((Tp == 0).mean()); assert 0.05 < zf < 0.30, zf
        R = {}
        for r in roles:
            pr = info["producers"][r]
            p, v = fwd(pr["output"], Xn)
            expected_variant = expected_variants.get(pr["job"], "FF_" + pr["job"])
            ckmeta["%s/%s" % (cx, r)] = dict(path=pr["output"], sha256=sha(REPO / pr["output"]), saved_variant=v,
                                             expected=expected_variant, match=(v == expected_variant) if v else None)
            R[r] = (Tp - p).astype(np.float32)
            mk = float(np.abs(R[r]).mean()); assert np.isfinite(R[r]).all() and 0.0015 < mk < 0.010, (cx, r, mk)
        parts.append(dict(Xp=Xp, R=R, SC=Tp != 0.0, T=Tp, grp=np.full(len(Xp), gi, np.int16),
                          files=[Path(f).name for f in vf], n=len(Xp)))
        print("# %s %s: %d files %d win zf %.4f |r| %s (%.0fs)" % (a.fold, cx, len(vf), len(Xp), zf,
              {k: round(float(np.abs(v).mean()), 6) for k, v in R.items()}, time.time() - t00), flush=True)
        del Xn, Yn
    np.save(out / "Xp.npy", np.concatenate([p["Xp"] for p in parts], 0))
    for r in roles:
        np.save(out / ("R_%s.npy" % r), np.concatenate([p["R"][r] for p in parts], 0))
    np.save(out / "SC.npy", np.concatenate([p["SC"] for p in parts], 0))
    np.save(out / "T.npy", np.concatenate([p["T"] for p in parts], 0))
    np.save(out / "grp.npy", np.concatenate([p["grp"] for p in parts], 0))
    bad = [k for k, v in ckmeta.items() if v["match"] is False]
    stats_label = str(stats_path.relative_to(REPO)) if stats_path.is_relative_to(REPO) else str(stats_path)
    meta = dict(tool="Track1/U-net/VariantSPSBand2/sb2_cache.py", fold=a.fold, stride=a.stride, stats=stats_label,
                n=int(sum(p["n"] for p in parts)), files=[p["files"] for p in parts], roles=list(roles),
                isolation=proof, checkpoints=ckmeta, variant_mismatch=bad,
                array_sha256={p.name: sha(p) for p in sorted(out.glob("*.npy"))}, elapsed_s=time.time() - t00)
    assert not bad, bad
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("CACHE_OK", a.fold, meta["n"], "%.0fs" % meta["elapsed_s"], flush=True)


if __name__ == "__main__":
    main()
