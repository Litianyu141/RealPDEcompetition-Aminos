#!/usr/bin/env python3
"""Train and out-of-fold evaluate the LEARNED variance-deficit field (see `vgain_head.py`).

THE ONE DESIGN DECISION THAT MAKES OR BREAKS THIS.  The head's target is `sqrt(KE_t / KE_p)` -- the
forecaster's variance deficit.  On data the FORECASTER was trained on, that deficit is small: the
net fits its own training windows, so `KE_p ~ KE_t`, `g* ~ 1`, and a head trained there learns
"output 1" and is useless at test time.  So the head must be trained on **genuinely out-of-fold
forecasts**, which means a NESTED design:

    * the 5 AoA fold checkpoints already give an out-of-fold forecast for every one of the 82 files
      (fold j's checkpoint runs only on fold j's held-out files) -- that is one full pass, cached;
    * the head evaluated on fold k is trained on the cached entries of folds j != k.

So no head ever sees fold k's files, and every forecast it learns from was produced by a model that
had not seen that file either.  Both leaks are closed, and the deficit the head learns is the
deficit it will face.

FOUR ARMS, each the exact optimum at its level, so each bounds the ones below:

    g = 1        no correction
    static       ONE g[y,x] fitted on the training folds by the same loss -- the incumbent family
                 (this is what the shipped `gamma(32,64,3)` is, and it is what the head must beat;
                 beating `g=1` proves nothing, the competition already beats `g=1`)
    head         the learned SAMPLE-DEPENDENT field
    oracle       g = sqrt(KE_t/KE_p) pointwise -- the L5 ceiling, needs the target, not submittable

PRE-REGISTERED BAR, written before the first number: the head ships only if it beats **static** by
dFINAL >= +0.06 with paired |t| >= 2 over the 5 folds and >= 4/5 folds improving.  Beating `g=1` is
explicitly NOT the bar.

    python train_vgain.py --gpu 6 --epochs 60
"""
from __future__ import annotations
import os, sys, json, time, argparse, datetime, csv, importlib.util, random
from pathlib import Path
import numpy as np
import torch

# Derived from this file's own location (2026-08-12): the absolute hardcode silently
# escaped a fresh clone back to the production tree.  NB: run these tools from their
# in-repo location (the chains do); a /tmp copy would derive the wrong root.
REPO = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO / "common"))
from repo_paths import resolve_paths   # configs and split manifests carry repo-relative paths
from fivefold_heads import HeadPlan, sha256_file
T1U = REPO / "Track1/U-net"
T2_KIT = REPO / "Track2/realpde_t2_starting_kit_v4"
sys.path.insert(0, str(T1U))
from unet_model import build_unet, build_from_meta                          # noqa: E402
from vgain_head import (VGainHead, stats_from_window, subscores, loss_from_stats,
                        N_FEAT, N_FEAT_RICH,   # noqa: E402
                        score_error, oracle_gain, selfcheck, N_FEAT)

spec = importlib.util.spec_from_file_location("tv4", T1U / "train_v4.py")
tv4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(tv4)


class _Null:
    def info(self, *a, **k):
        pass
    warning = debug = info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="unet_v4_Aa_aoa_mse")
    ap.add_argument("--arm2", default="", help="second arm; per-fold forecasts are averaged (equal-weight two-architecture blend) BEFORE stats, so the head is fitted to the blend")
    ap.add_argument("--ckpt-dir", default=str(REPO / "checkpoints/t1_u-net"))
    ap.add_argument("--manifest", default=str(REPO / "common/local_benchmark/splits_aoa_dedup.json"))
    ap.add_argument("--stats", default=str(T2_KIT / "example_data/mean_std_real.pt"))
    ap.add_argument("--gpu", default="6")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--rich", action="store_true",
                    help="VG2: 15 scale- and oscillator-aware features instead of 7")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--gmin", type=float, default=0.5)
    ap.add_argument("--gmax", type=float, default=3.0)
    ap.add_argument("--static-steps", type=int, default=400)
    ap.add_argument("--cache", default="", help="npz to reuse the out-of-fold forecast pass")
    ap.add_argument("--tag", default="")
    ap.add_argument("--ship", default="", help="train ONE head on ALL folds' out-of-fold data and "
                                               "save it here; no holdout, the 5-fold study is the "
                                               "evidence, this is the artefact")
    ap.add_argument("--logger-root", default=str(T1U))
    ap.add_argument("--seed", type=int, default=0, help="one fixed initialization/shuffle RNG")
    ap.add_argument("--strict-plan", default="", help="hash-bound outer/inner producer plan; no legacy cache reuse")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    selfcheck(dev)
    plan = HeadPlan(a.strict_plan) if a.strict_plan else None
    arms = [a.arm] + ([a.arm2] if a.arm2 else [])
    strict_stamp = plan.stamp(arms) if plan else ""
    man = (dict(plan.manifest, folds=plan.folds) if plan else
           json.loads(Path(a.manifest).read_text()))
    man = resolve_paths(man)
    if plan and not a.ship:
        raise ValueError("strict outer-fold heads require --ship; inner folds only supply OOF fitting data")
    if plan and Path(a.ship).exists():
        raise ValueError("strict head output already exists; use a fresh path")
    if plan and sha256_file(a.stats) != sha256_file(T2_KIT / 'example_data/mean_std_real.pt'):
        raise ValueError("strict vgain normalization must use the official released statistics")
    mean, std = tv4.load_stats(a.stats)
    mt = torch.from_numpy(mean).to(dev)
    st = torch.from_numpy(std).to(dev)
    T_IN = T_OUT = 20

    # ---------------- 1. one out-of-fold forecast pass over all 82 files -----------------------
    cache = {}
    t0 = time.time()
    if a.cache and Path(a.cache).exists():
        if plan:
            bound = json.loads(Path(a.cache + '.strict.json').read_text())
            if bound.get('sha256') != sha256_file(a.cache) or bound.get('strict_plan_stamp') != strict_stamp:
                raise ValueError("strict vgain cache content hash mismatch")
        z = np.load(a.cache)
        if plan and ("__strict_plan_stamp__" not in z.files or
                     str(z["__strict_plan_stamp__"].item()) != strict_stamp):
            raise ValueError("strict vgain cache plan/producer stamp mismatch")
        # CACHESTAMP (2026-08-27).  An EXISTING --cache skipped the whole out-of-fold pass below
        # REGARDLESS of --arm, and the npz carried no arm stamp: point a new arm at another
        # fleet's cache and the head is fitted on the WRONG fleet's residuals, TRAIN_VGAIN_OK and
        # all.  reproduce_t1.sh stage 4 passes a fixed --cache path, so a resumed or re-tagged
        # chain run hits exactly this.  REFUSE only on a stamp that EXISTS and disagrees; a cache
        # written before stamping cannot be verified, so WARN LOUDLY and proceed -- a guard whose
        # likeliest failure is a false refusal on the happy path is worse than the hole it closes.
        _want = (a.arm, a.arm2 or "")
        if "__arm__" in z.files:
            _have = (str(z["__arm__"].item()), str(z["__arm2__"].item())
                     if "__arm2__" in z.files else "")
            if _have != _want:
                raise SystemExit(
                    f"REFUSING: forecast cache {a.cache} was built by a DIFFERENT fleet.\n"
                    f"  cache was built with --arm {_have[0]!r} --arm2 {_have[1]!r}\n"
                    f"  this run requested   --arm {_want[0]!r} --arm2 {_want[1]!r}\n"
                    f"  Fitting the head on another fleet's residuals completes cleanly and is\n"
                    f"  wrong everywhere.  Remedy: pass --cache <a path for THIS arm> (the pass\n"
                    f"  is ~90 s), or delete {a.cache} to force a rebuild, or correct --arm.")
            print(f"# cache arm stamp OK: --arm {_have[0]} --arm2 {_have[1]!r}", flush=True)
        else:
            print(f"# WARNING: forecast cache {a.cache} carries NO arm stamp (written before\n"
                  f"#   2026-08-27).  It CANNOT be verified to belong to --arm {a.arm!r}\n"
                  f"#   --arm2 {a.arm2 or ''!r}.  If it does not, the head is being fitted on the\n"
                  f"#   WRONG fleet's residuals.  Delete it to rebuild a stamped one.", flush=True)
        # `if "/" in n` is NOT cosmetic: the stamp keys above are top-level names, and the old
        # unfiltered int(n.split("/")[0]) raises ValueError on them -- adding the stamp without
        # this filter would break every cache read, a worse bug than the one being closed.
        for k in sorted({int(n.split("/")[0]) for n in z.files if "/" in n}):
            cache[k] = {kk: torch.from_numpy(z[f"{k}/{kk}"]) for kk in
                        ("A", "B", "C", "M", "tn2", "KEp", "KEt", "feat")}
        if plan and set(cache) != {int(fd['fold']) for fd in plan.folds}:
            raise ValueError("strict vgain cache inner folds mismatch")
        print(f"# reused forecast cache {a.cache} "
              f"({sum(v['A'].shape[0] for v in cache.values())} windows)", flush=True)
    for fd in ([] if cache else man["folds"]):
        k = int(fd["fold"])
        Xn, Yn = tv4.build_windows(fd["val_files"], 2, T_IN, T_OUT, mean, std, a.stride,
                                   _Null(), f"vg_f{k}")
        cp = plan.checkpoint(a.arm, k) if plan else Path(a.ckpt_dir) / f"{a.arm}_f{k}.pth"
        d = torch.load(cp, map_location="cpu", weights_only=False)
        _meta = d.get("meta", {})
        # EVERY architecture key, from the checkpoint's own meta -- `base`, `fuse`, `in_aug`,
        # `in_norm` and the residual wrapper.  Rebuilding is mandatory, not cosmetic: without it
        # the head is fitted to a forecast the checkpoint does not produce.  And `in_norm` adds NO
        # PARAMETERS, so the strict load below cannot catch a missing one -- measured cost of
        # exactly that bug, 2026-07-29: 0.7 FINAL, on a bundle that looked like a real result.
        model = build_from_meta(_meta, T_IN, T_OUT, 3).to(dev).eval()
        model.load_state_dict(d["model_state_dict"], strict=True)
        model2 = None
        if a.arm2:
            cp2 = plan.checkpoint(a.arm2, k) if plan else Path(a.ckpt_dir) / f"{a.arm2}_f{k}.pth"
            d2 = torch.load(cp2, map_location="cpu",
                            weights_only=False)
            model2 = build_from_meta(d2.get("meta", {}), T_IN, T_OUT, 3).to(dev).eval()
            model2.load_state_dict(d2["model_state_dict"], strict=True)
        acc = {kk: [] for kk in ("A", "B", "C", "M", "tn2", "KEp", "KEt", "feat")}
        with torch.no_grad():
            for i in range(0, Xn.shape[0], 32):
                xb = torch.from_numpy(Xn[i:i + 32]).to(dev)
                yb = torch.from_numpy(Yn[i:i + 32]).to(dev)
                p = model(xb)
                if model2 is not None:
                    p = 0.5 * (p + model2(xb))   # blend in normalised space
                p = p * st + mt                              # physical forecast
                s = stats_from_window(p, yb * st + mt, xb * st + mt, T_OUT, rich=a.rich)
                for kk in acc:
                    acc[kk].append(s[kk].float().cpu())
        cache[k] = {kk: torch.cat(v, 0) for kk, v in acc.items()}
        print(f"# fold {k}: {len(fd['val_files'])} files -> {cache[k]['A'].shape[0]} windows "
              f"out-of-fold  ({time.time()-t0:.0f}s)", flush=True)
        del model, model2
        torch.cuda.empty_cache()
    if a.cache and not Path(a.cache).exists():
        # CACHESTAMP: bind the cache to the fleet that produced it (0-d unicode arrays; the npz
        # already carries arbitrary string keys like "0/A").  The reader above refuses a mismatch.
        np.savez(a.cache,
                 **{f"{k}/{kk}": v.numpy() for k, c in cache.items() for kk, v in c.items()},
                 __arm__=np.array(a.arm), __arm2__=np.array(a.arm2 or ""),
                 **({"__strict_plan_stamp__": np.array(strict_stamp)} if plan else {}))
        if plan:
            Path(a.cache + '.strict.json').write_text(json.dumps({
                'sha256': sha256_file(a.cache), 'strict_plan_stamp': strict_stamp,
                'fit_files': plan.fitting_files, 'strict_plan': str(plan.path)}, indent=2))
        print(f"# wrote forecast cache {a.cache}  [arm stamp: --arm {a.arm} "
              f"--arm2 {a.arm2 or ''!r}]", flush=True)
    folds = sorted(cache)

    def cat(ks, key):
        return torch.cat([cache[k][key] for k in ks], 0)

    def fit_static(tr, steps):
        """The exact optimum over a single static field, by the same loss the head minimises."""
        A, B, C = cat(tr, "A").to(dev), cat(tr, "B").to(dev), cat(tr, "C").to(dev)
        M, N2 = cat(tr, "M").to(dev), cat(tr, "tn2").to(dev)
        lg = torch.zeros(1, A.shape[1], A.shape[2], device=dev, requires_grad=True)
        opt = torch.optim.Adam([lg], lr=2e-2)
        import math
        for _ in range(steps):
            opt.zero_grad()
            g = lg.clamp(math.log(a.gmin), math.log(a.gmax)).exp().expand(A.shape[0], -1, -1)
            l, _, _ = loss_from_stats(g, A, B, C, M, N2, T_OUT)
            l.backward(); opt.step()
        return lg.detach().clamp(math.log(a.gmin), math.log(a.gmax)).exp()

    def evaluate(g, k):
        c = cache[k]
        rel, tke = subscores(g, c["A"].to(dev), c["B"].to(dev), c["C"].to(dev),
                             c["M"].to(dev), c["tn2"].to(dev), T_OUT)
        return (float(score_error(rel).mean()), float(score_error(tke).mean()),
                float(rel.mean()), float(tke.mean()))

    # ---------------- 2. per evaluation fold: static baseline, then the head ------------------
    rows = []
    for k in ([None] if a.ship else folds):
        tr = folds if k is None else [j for j in folds if j != k]
        ntr = sum(cache[j]["A"].shape[0] for j in tr)
        nk = cache[tr[0] if k is None else k]["A"].shape[0]
        if k is None:
            k = tr[0]          # the "eval" slice is only used for the printed sanity numbers
        one = torch.ones(nk, cache[k]["A"].shape[1], cache[k]["A"].shape[2], device=dev)
        r_one = evaluate(one, k)
        gs = fit_static(tr, a.static_steps)
        r_static = evaluate(gs.expand(nk, -1, -1), k)
        r_orc = evaluate(oracle_gain(cache[k]["KEp"].to(dev), cache[k]["KEt"].to(dev),
                                     a.gmin, a.gmax), k)

        # ---- the head ----
        Ftr = cat(tr, "feat")
        head = VGainHead((N_FEAT_RICH if a.rich else N_FEAT), a.width, a.gmin, a.gmax).to(dev)
        head.set_feature_norm(Ftr.mean(dim=(0, 2, 3)).to(dev), Ftr.std(dim=(0, 2, 3)).to(dev))
        opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.wd)
        Atr, Btr, Ctr = cat(tr, "A"), cat(tr, "B"), cat(tr, "C")
        Mtr, Ntr = cat(tr, "M"), cat(tr, "tn2")
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=a.lr, total_steps=max(1, a.epochs * ((ntr + a.batch - 1) // a.batch)))
        best, best_state = -1e30, None
        for ep in range(a.epochs):
            head.train()
            perm = torch.randperm(ntr)
            tot = 0.0
            for i in range(0, ntr, a.batch):
                idx = perm[i:i + a.batch]
                opt.zero_grad()
                g = head(Ftr[idx].to(dev))
                l, _, _ = loss_from_stats(g, Atr[idx].to(dev), Btr[idx].to(dev), Ctr[idx].to(dev),
                                          Mtr[idx].to(dev), Ntr[idx].to(dev), T_OUT)
                l.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                opt.step(); sched.step()
                tot += float(l) * len(idx)
            # model selection on the TRAINING folds only -- fold k is never looked at
            head.eval()
            with torch.no_grad():
                sc = 0.0
                for i in range(0, ntr, 512):
                    g = head(Ftr[i:i + 512].to(dev))
                    r, t_ = subscores(g, Atr[i:i + 512].to(dev), Btr[i:i + 512].to(dev),
                                      Ctr[i:i + 512].to(dev), Mtr[i:i + 512].to(dev),
                                      Ntr[i:i + 512].to(dev), T_OUT)
                    from vgain_head import W_TKE as _wt
                    sc += float((score_error(r) + _wt * score_error(t_)).sum())
                sc /= ntr
            if sc > best:
                best = sc
                best_state = {kk: v.detach().clone() for kk, v in head.state_dict().items()}
            if ep % 10 == 0 or ep == a.epochs - 1:
                print(f"  f{k} ep{ep:3d} train_loss {tot/ntr:+.4f}  fit_score {sc:.4f}", flush=True)
        head.load_state_dict(best_state)
        head.eval()
        with torch.no_grad():
            gh = torch.cat([head(cache[k]["feat"][i:i + 512].to(dev)).cpu()
                            for i in range(0, nk, 512)], 0).to(dev)
        r_head = evaluate(gh, k)
        torch.save({"state_dict": best_state, "width": a.width,
                    "gmin": a.gmin, "gmax": a.gmax, "n_feat": (N_FEAT_RICH if a.rich else N_FEAT),
                    "train_folds": tr, "eval_fold": k, "seed": a.seed,
                    **({"strict_plan_stamp": strict_stamp, "fit_files": plan.fitting_files,
                        "strict_plan": str(plan.path)} if plan else {})},
                   Path(a.ship) if a.ship else
                   Path(a.ckpt_dir) / f"vgain_head_{a.arm}{('_'+a.tag) if a.tag else ''}_f{k}.pth")
        if a.ship:
            print(f"# SHIP head -> {a.ship}  (trained on all {len(tr)} folds, {ntr} windows; the "
                  f"numbers printed above are IN-FIT and are not evidence)", flush=True)

        for nm, r in (("g=1", r_one), ("static", r_static), ("head", r_head), ("ORACLE", r_orc)):
            rows.append({"arm": nm, "fold": k, "n": nk, "s_rel": r[0], "s_tke": r[1],
                         "rel": r[2], "tke": r[3], "sum": r[0] + r[1]})
        print(f"# fold {k} ({nk} win, head trained on {ntr}):  "
              + "  ".join(f"{nm} {r[0]+r[1]:.3f}" for nm, r in
                          (("g=1", r_one), ("static", r_static), ("head", r_head), ("ORC", r_orc))),
              flush=True)
        gmn, gmx = float(gh.min()), float(gh.max())
        print(f"#   head g in [{gmn:.3f}, {gmx:.3f}] mean {float(gh.mean()):.3f}; "
              f"static g in [{float(gs.min()):.3f}, {float(gs.max()):.3f}]", flush=True)

    # ---------------- 3. report, paired against STATIC ----------------------------------------
    def per_fold(nm, key="sum"):
        return {r["fold"]: r[key] for r in rows if r["arm"] == nm}

    # in --ship mode only one row per arm exists, so report over the folds actually present
    folds = sorted({r["fold"] for r in rows})
    base = per_fold("static")
    hdr = (f"{'arm':10s} | {'s_rel':>8s} {'s_tke':>8s} {'sum':>9s} | {'vs g=1':>8s} | "
           f"{'vs static':>10s} {'SEM':>6s} {'t':>6s} {'won':>5s} | {'dFINAL':>8s}")
    print("\n" + hdr); print("-" * len(hdr))
    one = per_fold("g=1")
    for nm in ("g=1", "static", "head", "ORACLE"):
        rs = [r for r in rows if r["arm"] == nm]
        m = {kk: float(np.mean([r[kk] for r in rs])) for kk in ("s_rel", "s_tke", "sum")}
        cur = per_fold(nm)
        d = np.asarray([cur[f] - base[f] for f in folds])
        d1 = np.asarray([cur[f] - one[f] for f in folds])
        sd = float(np.std(d, ddof=1)); sem = sd / np.sqrt(len(d))
        t = d.mean() / sem if sem > 0 else float("nan")
        print(f"{nm:10s} | {m['s_rel']:8.3f} {m['s_tke']:8.3f} {m['sum']:9.3f} | {d1.mean():+8.3f} | "
              f"{d.mean():+10.3f} {sem:6.3f} {t:6.2f} {int((d>0).sum()):>2d}/{len(d):<2d} | "
              f"{d.mean()/5:+8.3f}")
    print()
    print("# `sum` = score_rel + score_tke, the only two subscores this correction can move")
    print("#   (mvpe is invariant by construction, and the band/timing are untouched).")
    print("# dFINAL = (sum difference)/5.  PRE-REGISTERED BAR: the head ships only if it beats")
    print("#   STATIC by dFINAL >= +0.06, paired |t| >= 2, >= 4/5 folds.  Beating g=1 is NOT the bar.")

    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    rd = Path(a.logger_root) / "Logger" / "vgain" / f"{ts}-{a.arm.replace('unet_v4_','')}{('-'+a.tag) if a.tag else ''}"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "results.json").write_text(json.dumps({"args": vars(a), "rows": rows}, indent=2))
    with open(rd / "errors.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "fold", "n", "rel_l2", "tke", "s_rel", "s_tke", "sum"])
        for r in rows:
            w.writerow([r["arm"], r["fold"], r["n"], f"{r['rel']:.6f}", f"{r['tke']:.6f}",
                        f"{r['s_rel']:.4f}", f"{r['s_tke']:.4f}", f"{r['sum']:.4f}"])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        for nm in ("g=1", "static", "head", "ORACLE"):
            pf = per_fold(nm)
            ax.plot(folds, [pf[f] for f in folds], "o-", label=nm)
        ax.set_xlabel("held-out AoA fold"); ax.set_ylabel("score_rel + score_tke")
        ax.set_title("learned variance-deficit field, out-of-fold")
        ax.grid(alpha=.3); ax.legend(fontsize=9)
        fig.tight_layout(); fig.savefig(rd / "error_curve.png", dpi=130)
    except Exception as e:                                        # noqa: BLE001
        print(f"# plot skipped: {e}", file=sys.stderr)
    print(f"# run record -> {rd}")
    print("TRAIN_VGAIN_OK")


if __name__ == "__main__":
    main()
