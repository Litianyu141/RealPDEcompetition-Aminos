#!/usr/bin/env python3
"""Offline exact scoring of revealed-target CLIMATOLOGY anchors for the T2 per-8x8-block FTL (v111, --clim).

KE part (v108 arms (mu, c), block FTL, first-order block utility): anchor KE = input-window KE (v108/v110b) or the causal
running mean / EMA of the revealed previous targets' per-pixel fluctuation KE of the same trajectory (first window: input).
Mean part (block FTL over lam, rel-linearized, optional switching margin): anchor = input time-mean (v110b) or the running
mean of the revealed previous targets' time-means.  Also: per-channel (u, v) KE climatology.
Stats: common/ke_stats_probe.py / common/mean_stats_probe.py on the v100-equivalent fold cards (Logger/20260923).
Real fold cards (common/ke_ftl_card_t2.py --clim): KE +0.0801 / mean +0.0360 vs v110b, matching this simulation.

    python Track2/U-net/VariantAdaptiveKE/clim_sim.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(HERE))
import simulate as S  # noqa: E402
import mean_anchor_sim as M  # noqa: E402

T = 20.0; BH = BW = 8
MU = [0, .05, .1, .15, .2, .25, .3, .35, .4]; CS = [0.95, 1.0, 1.05]


def traj_runs(files):
    s, N = 0, len(files)
    while s < N:
        e = s
        while e < N and files[e] == files[s]:
            e += 1
        yield s, e
        s = e


def causal_mean(x, files, beta=None):
    """out[k] = running mean (beta None) or bias-corrected EMA of x[j], j<k, same trajectory; NaN where none revealed."""
    out = np.full(x.shape, np.nan)
    for s, e in traj_runs(files):
        acc, w = None, 0.0
        for k in range(s, e):
            if acc is not None:
                out[k] = acc / w
            if beta is None:
                acc = x[k].copy() if acc is None else acc + x[k]; w += 1.0
            else:
                acc = x[k] * beta if acc is None else acc * (1 - beta) + x[k] * beta; w = 1 - (1 - beta) ** (k - s + 1)
    return out


def ftl_blocks(U, files, i0, margin=0.0):
    """U: (N, A, nb) block utilities -> causal per-block leader index (N, nb); ties/first -> i0."""
    ch = np.zeros((U.shape[0], U.shape[2]), int)
    for s, e in traj_runs(files):
        a = np.zeros(U.shape[1:])
        for k in range(s, e):
            d = a - a[i0][None]; d[i0] += 1e-12; j = np.argmax(d, axis=0)
            if margin > 0:
                j = np.where(d[j, np.arange(d.shape[1])] > margin * np.abs(a[i0]).clip(1e-12), j, i0)
            ch[k] = j; a = a + U[k]
        # windows after the last are never used
    return ch


def ke_part(anchor, mu=MU, perch=False):
    arms = [(m, c) for m in mu for c in CS]; i0 = arms.index((0.1, 1.0)); res = {}
    for ax, folds in S.AX.items():
        parts = []
        for fo in folds:
            z = S.load([fo]); A = z['A'].astype(np.float64); B = z['B'].astype(np.float64); C = z['C'].astype(np.float64)
            N, H, W, _ = A.shape; nb = (H // BH) * (W // BW)
            kpc, ktc = A / (2 * T), C / (2 * T); kp, kt = kpc.sum(-1), ktc.sum(-1); ki = z['KEin'].astype(np.float64)
            if anchor == 'input':
                ka = ki
            else:
                cl = causal_mean(ktc if perch else kt, z['files'], None if anchor == 'clim' else float(anchor[3:]))
                if perch:
                    first = np.isnan(cl[:, 0, 0, 0]); cl[first] = ki[first][..., None] * kpc[first] / np.clip(kp[first][..., None], 1e-12, None)
                else:
                    first = np.isnan(cl[:, 0, 0]); cl[first] = ki[first]
                ka = cl
            rel0, tke0 = S.score_windows(z, np.ones_like(kp)); ktn = (kt.reshape(N, -1) ** 2).sum(1)
            blk = lambda x: x.reshape(N, H // BH, BH, W // BW, BW).sum(axis=(2, 4)).reshape(N, nb)
            U, G = [], []
            for (m, c) in arms:
                if perch:
                    g = c * np.clip(np.sqrt(np.clip(kpc + m * (ka - kpc), 1e-12, None) / np.clip(kpc, 1e-12, None)), .5, 2.)
                    e_ = (g * g * A - 2 * g * B + C).sum(-1); kp2 = (g * g * kpc).sum(-1)
                else:
                    g = c * np.clip(np.sqrt(np.clip(kp + m * (ka - kp), 1e-12, None) / np.clip(kp, 1e-12, None)), .5, 2.)
                    e_ = ((g * g)[..., None] * A - 2 * g[..., None] * B + C).sum(-1); kp2 = g * g * kp
                    g = np.repeat(g[..., None], 2, -1)
                G.append(g.astype(np.float32))
                U.append(-S.WR * blk(e_) / (2 * rel0[:, None] * z['tn2'][:, None])
                         - S.WT * blk((kp2 - kt) ** 2) / (2 * np.clip(tke0, 1e-8, None)[:, None] * np.clip(ktn, 1e-16, None)[:, None]))
            ch = ftl_blocks(np.transpose(np.array(U), (1, 0, 2)), z['files'], i0)
            chp = np.repeat(np.repeat(ch.reshape(N, H // BH, W // BW), BH, 1), BW, 2)
            g = np.take_along_axis(np.array(G), chp[None, ..., None].repeat(2, -1), axis=0)[0].astype(np.float64)
            sse = z['M'] + (g * g * A - 2 * g * B + C).reshape(N, -1).sum(1)
            rel = np.sqrt(np.clip(sse, 0, None)) / np.sqrt(np.clip(z['tn2'], 1e-16, None))
            tke = np.linalg.norm(((g * g * A).sum(-1) / (2 * T) - kt).reshape(N, -1), axis=1) / np.clip(np.linalg.norm(kt.reshape(N, -1), axis=1), 1e-8, None)
            parts.append((z, rel0, tke0, rel, tke))
        zz = {k: np.concatenate([p[0][k] for p in parts]) for k in ['mvpe', 'S', 'n']}
        cat = lambda i: np.concatenate([p[i] for p in parts]); sel = np.ones(len(zz['S']), bool)
        b, c = S.legs(zz, cat(1), cat(2), sel), S.legs(zz, cat(3), cat(4), sel); res[ax] = {k: c[k] - b[k] for k in b}
    m = {k: 0.5 * (res['re'][k] + res['aoa'][k]) for k in S.ON}
    return S.fin({k: S.ON[k] + m[k] for k in S.ON}) - S.F0, m


def mean_part(anchor, lams=M.LAMS, margin=0.0):
    M.LAMS = list(lams); res = {}
    for ax, folds in S.AX.items():
        z = dict(M.load(2, folds)); N, H, W, _ = z['D'].shape
        if anchor == 'clim':
            c = causal_mean(z['TB'].astype(np.float64), z['files']); pb = z['D'].astype(np.float64) + z['TB'].astype(np.float64)
            Ec = c - pb; first = np.isnan(Ec[:, 0, 0, 0]); Ec[first] = z['E'][first]; z['E'] = Ec.astype(np.float32)
        U = np.transpose(np.array([M.block_util(z, l) for l in M.LAMS]), (1, 0, 2))
        ch = ftl_blocks(U, z['files'], 0, margin)
        r0, m0 = M.metrics(z, 0.0); r, mv = M.metrics(z, M.expand(ch, N, H, W)); sel = np.ones(N, bool)
        b, c = M.legs(z, r0, m0, sel), M.legs(z, r, mv, sel); res[ax] = {k: c[k] - b[k] for k in b}
    return M.eq(2, res)


if __name__ == '__main__':
    fmt = lambda m: ' '.join('%s%+.3f' % (k, m[k]) for k in ('rel', 'tke', 'mvpe', 'sps'))
    for anc in ('input', 'clim', 'ema0.2'):
        dF, m = ke_part(anc); print('KE   anchor=%-7s EQ dF %+.4f  %s' % (anc, dF, fmt(m)), flush=True)
    dF, m = ke_part('clim', perch=True); print('KE   anchor=clim per-channel EQ dF %+.4f  %s' % (dF, fmt(m)), flush=True)
    dF, m = ke_part('clim', mu=MU + [.5, .6, .7, .8, 1.0]); print('KE   anchor=clim mu<=1  EQ dF %+.4f  %s' % (dF, fmt(m)), flush=True)
    for anc in ('input', 'clim'):
        for mg in (0.0, 0.1):
            dF, m = mean_part(anc, margin=mg); print('MEAN anchor=%-5s margin=%.1f EQ dF %+.4f  %s' % (anc, mg, dF, fmt(m)), flush=True)
