#!/usr/bin/env python3
"""Out-of-fold empirical-Bayes priors for the T2 per-block FTL accumulators (warm start at trajectory reset).

For the KE-anchor arms (mu grid x gain c, climatology anchor, gain clamp LO..HI) and the time-mean arms (lam grid,
climatology mean anchor), the per-window block utilities are computed exactly as the card's `_kf_update` computes them
(rel/tke linearized at the unanchored window; common/ke_ftl_card_t2.py), averaged over all windows of the given folds,
and expressed relative to the prior arm:  P[a, b] = mean_n U[n, a, b] - mean_n U[n, i0, b].  The card starts each
trajectory with acc = kappa * P (kappa in windows of evidence); online evidence then overrides it.
Stats: Logger/20260923/{fold}_T2.npz (A, B, C, M, tn2, KEin; v100-equivalent fold cards = the FTL card's pre-anchor
prediction) and Logger/20260923/meanstats/{fold}_T2.npz (D, E, TB).

    python Track2/U-net/VariantAdaptiveKE/ftl_prior.py --folds re1,re2 --clamp 0.25,4 --out prior.json
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.argv, _argv = [sys.argv[0]], sys.argv[1:]
sys.path.insert(0, str(HERE))
import simulate as S  # noqa: E402
import clim_sim as CS  # noqa: E402
import mean_anchor_sim as M  # noqa: E402

T = 20.0; BH = BW = 8
MU = [0, .05, .1, .15, .2, .25, .3, .35, .4, .5, .6, .7, .8, 1.0]; CG = [0.95, 1.0, 1.05]
LAMS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]


def ke_utils(z, lo, hi, wt):
    A = z["A"].astype(np.float64); B = z["B"].astype(np.float64); C = z["C"].astype(np.float64)
    N, H, W, _ = A.shape; nb = (H // BH) * (W // BW)
    kp, kt = A.sum(-1) / (2 * T), C.sum(-1) / (2 * T)
    ka = CS.causal_mean(kt, z["files"]); first = np.isnan(ka[:, 0, 0]); ka[first] = z["KEin"][first]
    rel0, tke0 = S.score_windows(z, np.ones_like(kp)); ktn = (kt.reshape(N, -1) ** 2).sum(1)
    blk = lambda x: x.reshape(N, H // BH, BH, W // BW, BW).sum(axis=(2, 4)).reshape(N, nb)
    U = []
    for m in MU:
        for c in CG:
            g = c * np.clip(np.sqrt(np.clip(kp + m * (ka - kp), 1e-12, None) / np.clip(kp, 1e-12, None)), lo, hi)
            e_ = ((g * g)[..., None] * A - 2 * g[..., None] * B + C).sum(-1)
            U.append(-S.WR * blk(e_) / (2 * rel0[:, None] * z["tn2"][:, None])
                     - wt * blk((g * g * kp - kt) ** 2) / (2 * np.clip(tke0, 1e-8, None)[:, None] * np.clip(ktn, 1e-16, None)[:, None]))
    return np.array(U).mean(1)                                       # (A, nb)


def mean_utils(z):
    z = dict(z)
    c = CS.causal_mean(z["TB"].astype(np.float64), z["files"]); pb = z["D"].astype(np.float64) + z["TB"].astype(np.float64)
    Ec = c - pb; first = np.isnan(Ec[:, 0, 0, 0]); Ec[first] = z["E"][first]; z["E"] = Ec.astype(np.float32)
    return np.array([M.block_util(z, l).mean(0) for l in LAMS])      # (L, nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--clamp", default="0.5,2"); ap.add_argument("--wt-scale", type=float, default=1.0)
    ap.add_argument("--gains", default="", help="comma list replacing the gain arms c (default 0.95,1.0,1.05)")
    ap.add_argument("--mu-grid", default="", help="comma list replacing the KE-anchor strength grid")
    a = ap.parse_args(_argv)
    lo, hi = (float(v) for v in a.clamp.split(","))
    global MU, CG
    if a.gains:
        CG = [float(v) for v in a.gains.split(",")]
    if a.mu_grid:
        MU = [float(v) for v in a.mu_grid.split(",")]
    folds = a.folds.split(",")
    ke, mn, nw = 0.0, 0.0, 0
    for f in folds:
        z = S.load([f]); zm = M.load(2, [f]); n = len(z["files"])
        ke = ke + ke_utils(z, lo, hi, S.WT * a.wt_scale) * n; mn = mn + mean_utils(zm) * n; nw += n
    ke, mn = ke / nw, mn / nw
    i0 = MU.index(0.1) * len(CG) + CG.index(1.0)
    P = dict(ke=(ke - ke[i0][None]).tolist(), mean=(mn - mn[0][None]).tolist(), folds=folds, clamp=[lo, hi], wt_scale=a.wt_scale,
             mu_grid=MU, gains=CG, lams=LAMS, arm_order="mu-major, then c", n_windows=nw)
    Path(a.out).write_text(json.dumps(P))
    print("PRIOR_OK", a.out, "folds", folds, "windows", nw, "ke leader share of non-prior arms %.2f" % float((np.argmax(ke, 0) != i0).mean()))


if __name__ == "__main__":
    main()
