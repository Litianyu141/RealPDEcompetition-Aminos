#!/usr/bin/env python3
"""LEAVE-ONE-REYNOLDS-OUT stress test of the Track-2 post-hoc calibration.

Why this exists. calibrate_t2.py / calibrate_t2_v2.py both held out trajectories by parity, but the
local val split turns out to contain only TWO Reynolds numbers (11400 and 17775, ~5 angles of attack
each), so a parity split puts BOTH Re in both halves. It measures generalisation across angle of
attack and says nothing about generalisation across Re -- which is exactly the shift between our val
split and the platform's hidden test set. Any gain measured that way is therefore not yet evidence
that the calibration is safe to ship.

This script fits every calibration lever on ONE Reynolds number and scores it on the OTHER, in both
directions, so each reported number is genuinely out-of-condition. Arms, all sharing one forward
pass (so time_score is identical and cancels out of every comparison):

    A  raw                      gamma=1, constant normalized half-width      <- incumbent recipe
    B  band only                gamma=1, per-(t,ch) fitted band
    C  scalar gamma + band      gamma fitted on the fit Re
    D  spatial field + band     per-(y,x,ch) field fitted on the fit Re
    E  D with the band width inflated by kappa                               <- OOD hedge

Arm E answers the question a fitted interval always raises: out-of-condition errors are larger than
in-condition ones, so a band fitted in-condition under-covers. If widening helps out of condition,
that is measured here rather than guessed.

    python calibrate_t2_v3.py --checkpoints a.pth --gpu 6 --out calib3.json
"""
import os, sys, json, time, argparse, re
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
T2_KIT = REPO / "Track2/realpde_t2_starting_kit_v4"
sys.path.insert(0, str(REPO / "Track2/U-net"))
from calibrate_t2 import sample_weights, fit_band, expand_band, final_score  # noqa: E402
from calibrate_t2_v2 import temporal_var, blur3  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", required=True)
    p.add_argument("--data-dir", default=str(REPO / "data/RealPDE_competition_dataset/gdrive_raw/local_bench/val"))
    p.add_argument("--gpu", default=None)
    p.add_argument("--tag", default="cal3")
    p.add_argument("--out", default="")
    p.add_argument("--ref-hw", type=float, default=0.15)
    p.add_argument("--lam", type=float, default=0.25)
    p.add_argument("--blur", type=int, default=1)
    return p.parse_args()


def main():
    a = parse_args()
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sys.path.insert(0, str(T2_KIT))
    from local_eval import build_stream, Normalizer
    from unet_model import build_unet
    import scoring as official

    paths = [p.strip() for p in a.checkpoints.split(",") if p.strip()]
    models = []
    for cp in paths:
        s = torch.load(cp, map_location="cpu", weights_only=False)
        state = s["model_state_dict"] if isinstance(s, dict) and "model_state_dict" in s else s
        meta = s.get("meta", {}) if isinstance(s, dict) else {}
        m = build_unet(t_in=20, t_out=20, c=3, base=int(meta.get("base", 32))).to(dev).eval()
        m.load_state_dict(state)
        models.append(m)
        print(f"# member {Path(cp).name} base={meta.get('base')} val_rel_l2={meta.get('val_rel_l2')}", flush=True)

    data_dir = Path(a.data_dir)
    normalizer = Normalizer(data_dir / "mean_std_real.pt")
    preds_n, tgts, times, sim = [], [], [], []
    for step in build_stream(data_dir):
        inp = step["input"].unsqueeze(0)
        tgt = step["target"].unsqueeze(0)
        inp_norm, _ = normalizer.preprocess(inp, tgt)
        t0 = time.perf_counter()
        x = torch.as_tensor(inp_norm).to(dev)
        with torch.no_grad():
            out = torch.stack([m(x) for m in models], 0).mean(0)
        if dev == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        preds_n.append(out.detach().cpu())
        tgts.append(tgt.squeeze(0).cpu().numpy().astype(np.float32))
        sim.append(step["sim_id"])

    P0 = normalizer.postprocess_pred(torch.cat(preds_n, 0)).numpy().astype(np.float32)
    T = np.stack(tgts, 0)
    sim = np.asarray(sim)
    re_of = np.asarray([int(re.match(r"(\d+)_", s).group(1)) for s in sim])
    mean_step = float(np.mean(times))
    c = official.measured_channels(T)
    n_t, H, Wd, C = P0.shape[1], P0.shape[2], P0.shape[3], P0.shape[4]
    sigma = official.SIGMA_GLOBAL
    groups = sorted(set(re_of.tolist()))
    print(f"# stream N={P0.shape[0]} shape={P0.shape} c={c} mean_step={mean_step*1000:.2f} ms", flush=True)
    for g in groups:
        print(f"#   Re={g}: {int((re_of==g).sum())} samples, "
              f"{len(set(sim[re_of==g].tolist()))} trajectories -> {sorted(set(sim[re_of==g].tolist()))}", flush=True)

    _, _, _, st_ = torch.load(T2_KIT / "example_data/mean_std_real.pt", map_location="cpu", weights_only=False)
    std_ch = np.asarray(st_, np.float32).reshape(-1).astype(np.float64).copy()
    std_ch[std_ch == 0] = 1.0
    ref_band = (a.ref_hw * std_ch).astype(np.float32).reshape(1, 1, 1, 1, -1)

    Pm = P0.mean(axis=1, keepdims=True)
    Pf = P0 - Pm

    def apply_g(idx, g):
        """g is a scalar or an (H,W,C) field."""
        gf = np.asarray(g, np.float32)
        if gf.ndim == 3:
            gf = gf.reshape(1, 1, H, Wd, C)
        return (Pm[idx] + Pf[idx] * gf).astype(np.float32)

    def gamma_field(idx):
        vp = temporal_var(P0[idx]).mean(axis=0)
        vt = temporal_var(T[idx]).mean(axis=0)
        g = np.sqrt(np.divide(vt, np.maximum(vp, 1e-12), out=np.ones_like(vt), where=vp > 1e-12))
        gb = np.clip(g, 0.3, 4.0).astype(np.float64)
        for _ in range(a.blur):
            gb = blur3(gb)
        return 1.0 + a.lam * (gb - 1.0)

    def scalar_gamma(idx):
        best = (-1e9, 1.0)
        for g in [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]:
            Q = apply_g(idx, g)
            _, fin, _ = final_score(Q, T[idx], official, mean_step, Q - ref_band, Q + ref_band)
            if fin > best[0]:
                best = (fin, g)
        return best[1]

    def inflate(lo_t, hi_t, k):
        mid, half = 0.5 * (lo_t + hi_t), 0.5 * (hi_t - lo_t)
        return mid - k * half, mid + k * half

    KAPPAS = [1.0, 1.1, 1.25, 1.5, 1.75]
    fold_rows, kap_acc = [], {k: [] for k in KAPPAS}
    for held in groups:
        hi_ = np.where(re_of == held)[0]
        fi_ = np.where(re_of != held)[0]
        Th, Tf = T[hi_], T[fi_]
        gs = scalar_gamma(fi_)
        gfld = gamma_field(fi_)
        print(f"\n### held-out Re={held}  (fit on {sorted(set(re_of[fi_].tolist()))}, "
              f"{len(fi_)} fit / {len(hi_)} held samples)", flush=True)
        print(f"#   scalar gamma*={gs}   field: u mean={gfld[...,0].mean():.3f} "
              f"[{gfld[...,0].min():.2f},{gfld[...,0].max():.2f}]  "
              f"v mean={gfld[...,1].mean():.3f} [{gfld[...,1].min():.2f},{gfld[...,1].max():.2f}]", flush=True)

        row = {"held_re": held, "scalar_gamma": float(gs)}
        Qa = apply_g(hi_, 1.0)
        sub, fin, cov = final_score(Qa, Th, official, mean_step, Qa - ref_band, Qa + ref_band)
        row["A_raw"] = {"FINAL": fin, "sub": sub, "cov": cov}
        print(f"  A raw            relL2={sub['rel_l2_score']:.2f} tke={sub['tke_score']:.2f} "
              f"sps={sub['sps_score']:.2f} cov={cov:.3f}  FINAL={fin:.3f}", flush=True)

        for key, gfit in [("B_band", 1.0), ("C_scalar", gs), ("D_field", gfld)]:
            Pfit = apply_g(fi_, gfit)
            lo_t, hi_t = fit_band(Pfit, Tf, c, sample_weights(Pfit, Tf, c, official), sigma, n_t)
            Q = apply_g(hi_, gfit)
            L, U = expand_band(Q, lo_t, hi_t, c)
            sub, fin, cov = final_score(Q, Th, official, mean_step, L, U)
            row[key] = {"FINAL": fin, "sub": sub, "cov": cov}
            print(f"  {key:<16} relL2={sub['rel_l2_score']:.2f} tke={sub['tke_score']:.2f} "
                  f"sps={sub['sps_score']:.2f} cov={cov:.3f}  FINAL={fin:.3f}"
                  f"  delta_vs_A={fin-row['A_raw']['FINAL']:+.3f}", flush=True)
            if key == "D_field":
                row["E_kappa"] = {}
                for k in KAPPAS:
                    lo_k, hi_k = inflate(lo_t, hi_t, k)
                    Lk, Uk = expand_band(Q, lo_k, hi_k, c)
                    s2, f2, c2 = final_score(Q, Th, official, mean_step, Lk, Uk)
                    row["E_kappa"][str(k)] = {"FINAL": f2, "sps": s2["sps_score"], "cov": c2}
                    kap_acc[k].append(f2)
                    print(f"      kappa={k:<5} sps={s2['sps_score']:.2f} cov={c2:.3f} FINAL={f2:.3f}", flush=True)
        fold_rows.append(row)

    print("\n############ leave-one-Re-out means ############", flush=True)
    means = {}
    for key in ["A_raw", "B_band", "C_scalar", "D_field"]:
        m = float(np.mean([r[key]["FINAL"] for r in fold_rows]))
        means[key] = m
        print(f"  {key:<10} LORO_FINAL={m:.3f}   delta_vs_A={m-np.mean([r['A_raw']['FINAL'] for r in fold_rows]):+.3f}", flush=True)
    for k in KAPPAS:
        means[f"E_kappa_{k}"] = float(np.mean(kap_acc[k]))
        print(f"  E kappa={k:<5} LORO_FINAL={means[f'E_kappa_{k}']:.3f}   "
              f"delta_vs_A={means[f'E_kappa_{k}']-means['A_raw']:+.3f}", flush=True)

    best_key = max(means, key=lambda k: means[k])
    print(f"\n# LORO winner: {best_key} at {means[best_key]:.3f} "
          f"({means[best_key]-means['A_raw']:+.3f} vs the incumbent recipe)", flush=True)

    out = {"tag": a.tag, "checkpoints": paths, "lam": a.lam, "blur": a.blur, "ref_hw": a.ref_hw,
           "mean_step_s": mean_step, "groups": groups, "folds": fold_rows, "loro_means": means,
           "loro_winner": best_key}
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"CALIB3_OK tag={a.tag} winner={best_key} loro={means[best_key]:.3f} raw={means['A_raw']:.3f}", flush=True)


if __name__ == "__main__":
    main()
