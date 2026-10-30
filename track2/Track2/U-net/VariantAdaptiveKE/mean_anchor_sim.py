#!/usr/bin/env python3
"""Offline exact scoring of time-mean anchors pbar' = pbar + lam * (xbar_in - pbar) (u,v) from common/mean_stats_probe.py.

rel^2 = (T * sum |D + lam E|^2 + F) / ||t||^2 exactly (fluctuations untouched); MVPE from the official probe points
(time-means); TKE unchanged; interval unchanged (SPS moves only through the accuracy factor).
Policies: T2 per-trajectory FTL (global lam, exact window utility incl. MVPE) and per-8x8-block FTL (rel part of the
utility linearized per block); T1 static per-block lam map chosen cross-fold.

    python Track2/U-net/VariantAdaptiveKE/mean_anchor_sim.py [DIR]
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D0 = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "Logger/20260923/meanstats"
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(HERE)); import simulate as S  # noqa: E402

T = 20.0
PY = [8 + 2 * j for j in range(9)]; PX = [13, 21, 29, 37]
LAMS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
WM = 0.0957 * 50 / (1 + 0.5 * ((100 / 94.1207421473 - 1) / 0.5)) ** 2
ONL = {2: (dict(rel=94.9440053617, tke=78.6723530238, mvpe=94.1207421473, sps=43.6321996726), 92.4802961403),
       1: (dict(rel=94.8509965935, tke=78.6560024934, mvpe=94.1401099639, sps=43.5291262123), 92.8762136464)}


def load(tr, folds):
    Z = [dict(np.load(D0 / f'{f}_T{tr}.npz')) for f in folds]
    return {k: np.concatenate([z[k] for z in Z]) for k in Z[0] if k != 'T'}


def metrics(z, lam):
    """lam: (N,H,W) map or scalar -> per-window rel, mvpe."""
    lam = np.broadcast_to(np.asarray(lam, np.float64), z['D'].shape[:3])[..., None]
    Dp = z['D'].astype(np.float64) + lam * z['E'].astype(np.float64)
    rel = np.sqrt((T * (Dp ** 2).reshape(len(Dp), -1).sum(1) + z['F']) / z['tn2'])
    errs = []
    for x in PX:
        pp = Dp[:, PY, x, :].reshape(len(Dp), -1); tt = z['TB'][:, PY, x, :].astype(np.float64).reshape(len(Dp), -1)
        errs.append(np.linalg.norm(pp, axis=1) / np.clip(np.linalg.norm(tt, axis=1), 1e-8, None))
    return rel, np.mean(errs, 0)


def legs(z, rel, mv, sel):
    acc = 0.5 * (1 - S.ff(rel)) + 0.3 * (1 - S.ff(z['tke'])) + 0.2 * (1 - S.ff(mv))
    return dict(rel=S.se(rel[sel].mean()), tke=S.se(z['tke'][sel].mean()), mvpe=S.se(mv[sel].mean()),
                sps=100 * (acc[sel] * z['S'][sel]).sum() / z['n'][sel].sum())


def eq(tr, res):
    on, tm = ONL[tr]; S.TM = tm
    m = {k: 0.5 * (res['re'][k] + res['aoa'][k]) for k in on}
    return S.fin({k: on[k] + m[k] for k in on}) - S.fin(on), m


def check(tr):
    z = load(tr, ['re0']); r, m = metrics(z, 0.0)
    print(f'T{tr} self-check re0: rel max|d| {np.abs(r - z["rel"]).max():.2e}  mvpe max|d| {np.abs(m - z["mvpe"]).max():.2e}')


def fixed(tr):
    for lam in LAMS[1:]:
        res = {}
        for ax, folds in S.AX.items():
            z = load(tr, folds); r0, m0 = metrics(z, 0.0); r, m = metrics(z, lam); sel = np.ones(len(r), bool)
            b, c = legs(z, r0, m0, sel), legs(z, r, m, sel); res[ax] = {k: c[k] - b[k] for k in b}
        dF, mm = eq(tr, res)
        print(f'T{tr} fixed lam={lam}: EQ dF {dF:+.4f} rel{mm["rel"]:+.3f} mvpe{mm["mvpe"]:+.3f} sps{mm["sps"]:+.3f}')


def ftl_global(tr=2):
    res = {}
    for ax, folds in S.AX.items():
        z = load(tr, folds); R, M = zip(*[metrics(z, l) for l in LAMS]); R, M = np.array(R), np.array(M)
        U = -S.WR * R - WM * M; files = z['files']; N = len(files); ch = np.zeros(N, int); s = 0; i0 = 0
        while s < N:
            e = s
            while e < N and files[e] == files[s]:
                e += 1
            a = np.zeros(len(LAMS))
            for k in range(s, e):
                d = a - a[i0]; d[i0] += 1e-9; ch[k] = int(np.argmax(d)); a = a + U[:, k]
            s = e
        idx = np.arange(N); sel = np.ones(N, bool)
        b, c = legs(z, R[0], M[0], sel), legs(z, R[ch, idx], M[ch, idx], sel); res[ax] = {k: c[k] - b[k] for k in b}
    dF, mm = eq(tr, res)
    print(f'T{tr} FTL global lam: EQ dF {dF:+.4f} rel{mm["rel"]:+.3f} mvpe{mm["mvpe"]:+.3f} sps{mm["sps"]:+.3f}')


def block_util(z, lam, bh=8, bw=8):
    """rel part of the utility per block, linearized at lam=0: -WR * T*sum_b |D+lam E|^2 / (2 rel0 ||t||^2)."""
    N, H, W, _ = z['D'].shape
    Dp = z['D'].astype(np.float64) + lam * z['E'].astype(np.float64)
    e = (Dp ** 2).sum(-1).reshape(N, H // bh, bh, W // bw, bw).sum(axis=(2, 4)).reshape(N, -1)
    rel0 = metrics(z, 0.0)[0]
    return -S.WR * T * e / (2 * rel0[:, None] * z['tn2'][:, None])


def expand(ch, N, H, W, bh=8, bw=8):
    return np.repeat(np.repeat(np.array(LAMS)[ch].reshape(N, H // bh, W // bw), bh, 1), bw, 2)


def ftl_block(tr=2):
    res = {}
    for ax, folds in S.AX.items():
        z = load(tr, folds); N, H, W, _ = z['D'].shape
        U = np.transpose(np.array([block_util(z, l) for l in LAMS]), (1, 0, 2))     # (N, A, nb)
        files = z['files']; ch = np.zeros((N, U.shape[2]), int); s = 0
        while s < N:
            e = s
            while e < N and files[e] == files[s]:
                e += 1
            a = np.zeros(U.shape[1:])
            for k in range(s, e):
                d = a - a[0][None]; d[0] += 1e-12; ch[k] = np.argmax(d, axis=0); a = a + U[k]
            s = e
        r0, m0 = metrics(z, 0.0); r, m = metrics(z, expand(ch, N, H, W)); sel = np.ones(N, bool)
        b, c = legs(z, r0, m0, sel), legs(z, r, m, sel); res[ax] = {k: c[k] - b[k] for k in b}
    dF, mm = eq(tr, res)
    print(f'T{tr} FTL 8x8 block lam (rel-linearized): EQ dF {dF:+.4f} rel{mm["rel"]:+.3f} mvpe{mm["mvpe"]:+.3f} sps{mm["sps"]:+.3f}')


def static_map(tr):
    res = {}
    for ax, folds in S.AX.items():
        acc, nt = {}, 0
        for fo in folds:
            tot = sum(np.array([block_util(load(tr, [o]), l).sum(0) for l in LAMS]) for o in folds if o != fo)
            best = np.argmax(tot, axis=0)
            z = load(tr, [fo]); N, H, W, _ = z['D'].shape
            r0, m0 = metrics(z, 0.0); r, m = metrics(z, expand(np.broadcast_to(best[None], (N, len(best))), N, H, W))
            sel = np.ones(N, bool); b, c = legs(z, r0, m0, sel), legs(z, r, m, sel)
            for k in b: acc[k] = acc.get(k, 0.0) + (c[k] - b[k]) * N
            nt += N
        res[ax] = {k: v / nt for k, v in acc.items()}
    dF, mm = eq(tr, res)
    print(f'T{tr} static cross-fold 8x8 lam map: EQ dF {dF:+.4f} rel{mm["rel"]:+.3f} mvpe{mm["mvpe"]:+.3f} sps{mm["sps"]:+.3f}')


if __name__ == '__main__':
    for tr in (2, 1):
        check(tr); fixed(tr)
    ftl_global(2); ftl_block(2); static_map(2); static_map(1)
