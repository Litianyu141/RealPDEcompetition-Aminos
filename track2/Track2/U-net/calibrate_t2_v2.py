#!/usr/bin/env python3
"""Stage-2 calibration: a SPATIAL fluctuation-inflation field, on top of the scalar gamma + band.

Motivation. scoring.kinetic_energy() is the temporal variance of u and v about their time-mean at
each grid point, so `tke_score` is a purely LOCAL statistic -- and an MSE-trained forecaster's
over-smoothing is strongly inhomogeneous: it is severe in the airfoil wake, where the fluctuations
live, and near-absent in the freestream. A single scalar gamma therefore has to compromise between
regions that need very different corrections. A per-(channel, y, x) field does not:

    gamma[ch,y,x]^2 = mean_n var_t(target[...,ch]) / mean_n var_t(pred[...,ch])

matches each channel's temporal variance pointwise, which is exactly the quantity TKE scores. The
geometry (airfoil location) is fixed across the whole dataset, so this field is a physical property
of the flow rather than a fit to particular trajectories -- but that is an argument, not evidence,
so lambda (shrinkage toward 1) and an optional 3x3 blur are BOTH selected on held-out trajectories,
and the field is only kept if it beats the scalar gamma there.

    python calibrate_t2_v2.py --checkpoints a.pth --gpu 6 --out calib_v2.json
"""
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
T2_KIT = REPO / "Track2/realpde_t2_starting_kit_v4"
sys.path.insert(0, str(REPO / "Track2/U-net"))
from calibrate_t2 import sample_weights, fit_band, expand_band, final_score  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", required=True)
    p.add_argument("--data-dir", default=str(REPO / "data/RealPDE_competition_dataset/gdrive_raw/local_bench/val"))
    p.add_argument("--gpu", default=None)
    p.add_argument("--tag", default="cal2")
    p.add_argument("--out", default="")
    p.add_argument("--ref-hw", type=float, default=0.15)
    return p.parse_args()


def temporal_var(x):
    """var over the lead-time axis, shape (N,T,H,W,C) -> (N,H,W,C). Matches scoring.kinetic_energy."""
    return np.mean((x - x.mean(axis=1, keepdims=True)) ** 2, axis=1)


def blur3(f):
    """3x3 box blur with edge replication, applied per channel; f is (H,W,C)."""
    g = np.pad(f, ((1, 1), (1, 1), (0, 0)), mode="edge")
    out = np.zeros_like(f)
    for dy in range(3):
        for dx in range(3):
            out += g[dy:dy + f.shape[0], dx:dx + f.shape[1]]
    return out / 9.0


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
    preds_n, tgts, times, traj = [], [], [], []
    tid = -1
    for step in build_stream(data_dir):
        inp = step["input"].unsqueeze(0)
        tgt = step["target"].unsqueeze(0)
        if step["is_first"]:
            tid += 1
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
        traj.append(tid)

    P0 = normalizer.postprocess_pred(torch.cat(preds_n, 0)).numpy().astype(np.float32)
    T = np.stack(tgts, 0)
    traj = np.asarray(traj)
    mean_step = float(np.mean(times))
    c = official.measured_channels(T)
    n_t, H, Wd, C = P0.shape[1], P0.shape[2], P0.shape[3], P0.shape[4]
    sigma = official.SIGMA_GLOBAL
    print(f"# stream N={P0.shape[0]} traj={traj.max()+1} shape={P0.shape} c={c} "
          f"mean_step={mean_step*1000:.2f} ms", flush=True)

    _, _, _, st_ = torch.load(T2_KIT / "example_data/mean_std_real.pt", map_location="cpu", weights_only=False)
    std_ch = np.asarray(st_, np.float32).reshape(-1).astype(np.float64).copy()
    std_ch[std_ch == 0] = 1.0

    Pm = P0.mean(axis=1, keepdims=True)
    Pf = P0 - Pm
    ref_band = (a.ref_hw * std_ch).astype(np.float32).reshape(1, 1, 1, 1, -1)

    fit_idx = np.where(traj % 2 == 0)[0]
    hold_idx = np.where(traj % 2 == 1)[0]
    print(f"# holdout: fit {len(fit_idx)} / hold {len(hold_idx)} samples "
          f"({len(np.unique(traj[fit_idx]))} vs {len(np.unique(traj[hold_idx]))} trajectories)", flush=True)

    def gamma_field(idx):
        """Per-(y,x,ch) variance-matching inflation, estimated on the given sample subset."""
        vp = temporal_var(P0[idx]).mean(axis=0)     # (H,W,C)
        vt = temporal_var(T[idx]).mean(axis=0)
        g = np.sqrt(np.divide(vt, np.maximum(vp, 1e-12), out=np.ones_like(vt), where=vp > 1e-12))
        return np.clip(g, 0.3, 4.0).astype(np.float64)

    def apply_field(idx, g_field):
        return (Pm[idx] + Pf[idx] * g_field.reshape(1, 1, H, Wd, C).astype(np.float32)).astype(np.float32)

    # ---------- select lambda / blur on held-out trajectories ----------
    g_fit = gamma_field(fit_idx)
    print(f"# gamma field on fit half: u mean={g_fit[...,0].mean():.3f} "
          f"[{g_fit[...,0].min():.2f},{g_fit[...,0].max():.2f}]  "
          f"v mean={g_fit[...,1].mean():.3f} [{g_fit[...,1].min():.2f},{g_fit[...,1].max():.2f}]", flush=True)

    Thold = T[hold_idx]
    rows = []
    for blur in [0, 1, 2]:
        gb = g_fit.copy()
        for _ in range(blur):
            gb = blur3(gb)
        for lam in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]:
            ge = 1.0 + lam * (gb - 1.0)
            Qh = apply_field(hold_idx, ge)
            sub, fin, _ = final_score(Qh, Thold, official, mean_step, Qh - ref_band, Qh + ref_band)
            rows.append((fin, blur, lam, sub))
            print(f"  blur={blur} lam={lam:<5} relL2={sub['rel_l2_score']:.2f} tke={sub['tke_score']:.2f} "
                  f"mvpe={sub['mvpe_score']:.2f} sps={sub['sps_score']:.2f} HOLDOUT_FINAL={fin:.3f}", flush=True)
    rows.sort(key=lambda r: -r[0])
    best_fin, best_blur, best_lam, _ = rows[0]

    # scalar-gamma reference on the SAME holdout, so the comparison is like-for-like
    srows = []
    for g in [1.0, 1.05, 1.1, 1.15, 1.2, 1.25]:
        Qh = (Pm[hold_idx] + np.float32(g) * Pf[hold_idx]).astype(np.float32)
        sub, fin, _ = final_score(Qh, Thold, official, mean_step, Qh - ref_band, Qh + ref_band)
        srows.append((fin, g))
    srows.sort(key=lambda r: -r[0])
    print(f"\n# HOLDOUT best scalar gamma={srows[0][1]} -> {srows[0][0]:.3f}", flush=True)
    print(f"# HOLDOUT best field  blur={best_blur} lam={best_lam} -> {best_fin:.3f}"
          f"   delta={best_fin - srows[0][0]:+.3f}", flush=True)
    use_field = best_fin > srows[0][0]
    print(f"# -> {'USING SPATIAL FIELD' if use_field else 'FALLING BACK TO SCALAR GAMMA'}", flush=True)

    # ---------- refit the chosen correction + band on ALL of val ----------
    if use_field:
        gb = gamma_field(np.arange(P0.shape[0]))
        for _ in range(best_blur):
            gb = blur3(gb)
        g_final = 1.0 + best_lam * (gb - 1.0)
    else:
        g_final = np.full((H, Wd, C), float(srows[0][1]), np.float64)
    Pall = (Pm + Pf * g_final.reshape(1, 1, H, Wd, C).astype(np.float32)).astype(np.float32)

    Wall = sample_weights(Pall, T, c, official)
    lo_t, hi_t = fit_band(Pall, T, c, Wall, sigma, n_t)
    Lf, Uf = expand_band(Pall, lo_t, hi_t, c)
    sub_f, fin_f, cov_f = final_score(Pall, T, official, mean_step, Lf, Uf)
    sub_b, fin_b, cov_b = final_score(P0, T, official, mean_step, P0 - ref_band, P0 + ref_band)

    # honest end-to-end holdout: field + band BOTH fitted on the fit half only
    gh = gamma_field(fit_idx)
    for _ in range(best_blur):
        gh = blur3(gh)
    gh = 1.0 + best_lam * (gh - 1.0) if use_field else np.full((H, Wd, C), float(srows[0][1]), np.float64)
    Pfit_h = apply_field(fit_idx, gh)
    lo_h, hi_h = fit_band(Pfit_h, T[fit_idx], c, sample_weights(Pfit_h, T[fit_idx], c, official), sigma, n_t)
    Qh = apply_field(hold_idx, gh)
    Lh, Uh = expand_band(Qh, lo_h, hi_h, c)
    sub_h, fin_h, cov_h = final_score(Qh, Thold, official, mean_step, Lh, Uh)
    sub_hb, fin_hb, _ = final_score(P0[hold_idx], Thold, official, mean_step,
                                    P0[hold_idx] - ref_band, P0[hold_idx] + ref_band)
    print(f"\n# HOLDOUT end-to-end  raw={fin_hb:.3f}  calibrated={fin_h:.3f}  delta={fin_h-fin_hb:+.3f}", flush=True)
    print(f"# FULL-VAL raw        FINAL={fin_b:.3f} {sub_b}", flush=True)
    print(f"# FULL-VAL calibrated FINAL={fin_f:.3f} {sub_f}", flush=True)

    out = {"tag": a.tag, "checkpoints": paths, "use_field": bool(use_field),
           "blur": int(best_blur), "lam": float(best_lam), "scalar_gamma": float(srows[0][1]),
           "gamma_field": g_final.tolist(), "band_lo_norm": (lo_t / std_ch[:c].reshape(1, -1)).tolist(),
           "band_hi_norm": (hi_t / std_ch[:c].reshape(1, -1)).tolist(), "c": int(c), "n_t": int(n_t),
           "mean_step_s": mean_step,
           "holdout_end_to_end": {"raw_FINAL": fin_hb, "calibrated_FINAL": fin_h, "sub": sub_h},
           "full_val_raw": {"sub": sub_b, "FINAL": fin_b, "cov": cov_b},
           "full_val_calibrated": {"sub": sub_f, "FINAL": fin_f, "cov": cov_f}}
    if a.out:
        Path(a.out).write_text(json.dumps(out))
    print(f"CALIB2_OK tag={a.tag} use_field={use_field} blur={best_blur} lam={best_lam} "
          f"full_val={fin_f:.3f} holdout_delta={fin_h-fin_hb:+.3f}", flush=True)


if __name__ == "__main__":
    main()
