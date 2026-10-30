#!/usr/bin/env python3
"""L2 T1-SBAND2 (2026-09-27): shared helpers.

The v160k-instrument fold cards (eval_KSM_MXGEGPTK, T1 Re folds) build their interval as

    lo0 = pn_pre + sb_lo[bin(s)] + ka_sh ;  hi0 = pn_pre + sb_hi[bin(s)] + ka_sh      (s = sband head score, bin per (lead, ch))
    mid = (lo0 + hi0)/2 ; hw = (hi0 - lo0)/2
    hw' = hw * BM + BK * |xl - mid| ;  mid' = mid + BB * (xl - mid)                    (synth BK persistence line)
    lo = mid' - hw' ; hi = mid' + hw'

(card source: `_chunk` / `_chunk_par`; BM/BK/BB identical in the three Re fold cards and the v160k ship card).
Everything here is PHYSICAL (the card works in normalised units and multiplies by _STD_T at the end; every step
above is affine per channel, so the physical form is the same map).  A new sband head changes ONLY (sb_lo, sb_hi)
per element, so the base centre B = pn_pre + ka_sh of the served card is recovered by inverting the BK line on
the SERVED lo / hi and subtracting the served table's centre offset; pred is untouched (rel / tke / mvpe exactly 0).
"""
import numpy as np

LT, C2, H, W = 20, 2, 32, 64
SIG = 0.0563870259
STD = np.array([0.09681041, 0.01596364], np.float64)       # mean_std_real.pt u, v (card _STD_T[..., :2])
# synth BK constants (fold cards re0/re1/re2 and t1_v160k_ksm.zip submission.py lines 175-177, identical)
BB = np.zeros((LT, C2)); BB[0] = (0.1, 0.15)
BM = np.ones((LT, C2)); BM[0] = (0.9, 0.9); BM[1] = BM[2] = (0.95, 0.95); BM[3] = (0.95, 1.0); BM[19] = (0.95, 0.95)
BK = np.zeros((LT, C2)); BK[0] = (0.1, 0.1); BK[1] = BK[2] = (0.05, 0.05); BK[3] = (0.05, 0.0); BK[19] = (0.05, 0.05)
CENTERS = [.35, .40, .45, .50, .55, .60, .65]                # t1_sband_pack.py fit_table
ff = lambda e: e / (0.5 + e)
RATIOS = {"registered": dict(rel=1.03, tke=1.86, mvpe=1.0, sps=0.69), "0926": dict(rel=1.42, tke=0.52, mvpe=1.0, sps=1.18)}
WDER = dict(rel=0.478, tke=0.101, mvpe=0.1, sps=0.243)
HI_RE = 22875


def acc_of(rel, tke, mv):
    return 1.0 - (0.5 * ff(rel) + 0.3 * ff(tke) + 0.2 * ff(mv))


def proj(dsps):
    return {k: round(WDER["sps"] * R["sps"] * dsps, 5) for k, R in RATIOS.items()}


def bk_forward(B, lo_off, hi_off, xl, t, c):
    """B, lo_off, hi_off: (N,H,W) float64 for one (lead t, ch c); xl (N,H,W).  -> lo, hi (float64)."""
    lo0 = B + lo_off; hi0 = B + hi_off
    mid = 0.5 * (lo0 + hi0); hw = 0.5 * (hi0 - lo0)
    hw2 = hw * BM[t, c] + BK[t, c] * np.abs(xl - mid)
    mid2 = mid + BB[t, c] * (xl - mid)
    lo, hi = mid2 - hw2, mid2 + hw2
    return np.minimum(lo, hi), np.maximum(lo, hi)


def bk_invert(lo_s, hi_s, xl, t, c):
    """Served lo / hi -> (mid, hw) BEFORE the BK line."""
    m2 = 0.5 * (lo_s.astype(np.float64) + hi_s.astype(np.float64)); h2 = 0.5 * (hi_s.astype(np.float64) - lo_s.astype(np.float64))
    mid = (m2 - BB[t, c] * xl) / (1.0 - BB[t, c])
    hw = (h2 - BK[t, c] * np.abs(xl - mid)) / BM[t, c]
    return mid, hw


def window_numerator(tgt, lo, hi):
    """(N,H,W) slices -> per-window sum of exp(-(hi-lo)/sig) * inside over scored elements (tgt != 0)."""
    sc = tgt != 0.0
    ins = (tgt >= lo) & (tgt <= hi) & sc
    e = np.where(ins, np.exp(-(hi - lo) / SIG), 0.0)
    return e.reshape(e.shape[0], -1).sum(1)


def fit_table(s_list, r_list, m_list, a_list, Q, nsub=80000, rng=0, side="left"):
    """t1_sband_pack.py::fit_table for ONE (lead, ch) slice, pooled over the given (score, residual) lists.
    s,r,m: (n,H,W); a: (n,) window weights.  -> edges (Q-1,), tab (Q,2) PHYSICAL offsets."""
    g = np.random.default_rng(rng)
    s = np.concatenate([sx[mx] for sx, mx in zip(s_list, m_list)]).astype(np.float64)
    r = np.concatenate([rx[mx] for rx, mx in zip(r_list, m_list)]).astype(np.float64)
    w = np.concatenate([np.broadcast_to(ax.reshape(-1, 1, 1), mx.shape)[mx] for ax, mx in zip(a_list, m_list)]).astype(np.float64)
    e = np.maximum.accumulate(np.quantile(s, np.linspace(0, 1, Q + 1)[1:-1]))
    b = np.searchsorted(e, s, side=side)
    tab = np.zeros((Q, 2))
    for k in range(Q):
        sel = np.flatnonzero(b == k)
        if sel.size < 8:
            tab[k] = (-0.01, 0.01); continue
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
        tab[k] = (best[1], best[2])
    return e, tab


def lookup(s, edges, tab, side="left"):
    b = np.searchsorted(edges, s.astype(np.float64), side=side)
    return tab[b, 0], tab[b, 1]


def sps_pool(acc_list, S_list, n_list, sel_list=None):
    acc = np.concatenate(acc_list); S = np.concatenate(S_list); n = np.concatenate(n_list)
    sel = np.ones(len(S), bool) if sel_list is None else np.concatenate(sel_list)
    return 100.0 * float((acc[sel] * S[sel]).sum() / n[sel].sum())


def read_vs(base, new, folds=("re0", "re1", "re2")):
    """base / new: fold -> dict(acc, n, S, re).  Re-axis dSPS reads (pooled, per fold, hi-Re, per group)."""
    res = {}
    for f in folds:
        res[f] = round(sps_pool([base[f]["acc"]], [new[f]["S"]], [base[f]["n"]]) - sps_pool([base[f]["acc"]], [base[f]["S"]], [base[f]["n"]]), 5)
    A = [base[f]["acc"] for f in folds]; N = [base[f]["n"] for f in folds]
    res["pooled"] = round(sps_pool(A, [new[f]["S"] for f in folds], N) - sps_pool(A, [base[f]["S"] for f in folds], N), 5)
    re = np.concatenate([base[f]["re"] for f in folds])
    hi = re >= HI_RE
    res["hi22875"] = round(sps_pool(A, [new[f]["S"] for f in folds], N, [hi]) - sps_pool(A, [base[f]["S"] for f in folds], N, [hi]), 5)
    grp = {int(v): round(sps_pool(A, [new[f]["S"] for f in folds], N, [re == v]) - sps_pool(A, [base[f]["S"] for f in folds], N, [re == v]), 5)
           for v in np.unique(re)}
    res["groups_ge0"] = int(sum(v >= 0 for v in grp.values())); res["n_groups"] = len(grp); res["groups"] = grp
    res["proj"] = proj(res["pooled"]); res["proj_folds"] = {f: proj(res[f]) for f in folds}
    res["pass_bar"] = bool(res["pooled"] >= 0.030 and all(v >= 0.005 for v in res["proj"].values())
                           and all(min(res["proj_folds"][f].values()) > -0.005 for f in folds))
    return res
