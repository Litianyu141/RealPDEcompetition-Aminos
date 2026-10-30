#!/usr/bin/env python3
"""Offline exact scoring of T2 KE-anchor policies from per-window sufficient statistics (common/ke_stats_probe.py).

Policies (causal: window k only uses revealed targets of windows j<k of the same trajectory):
  fixed(mu)                       g from mu everywhere
  traj(beta, mumax, warm, mu0)    scalar mu_k = Ep/(Ep+Ein) with Ep/Ein = bias-corrected EMAs over past windows of the
                                  pixel-summed squared KE errors of the model output and of the input KE; blended toward
                                  mu0 with weight min(1, k/warm); clipped to [0, mumax]
  pix(beta, mumax, warm, mu0)     same per pixel (3x3 box-smoothed EMAs)
g = clip(sqrt(((1-mu)KE_p + mu KE_in)/KE_p), 0.5, 2).  rel/tke recomputed exactly; MVPE and the band are unchanged;
SPS recomputed through the per-window accuracy factor.  The in-card anchor also feeds the band MAD of later steps; that
second-order effect is ignored here (fixed-mu results are compared with the real-card evaluation as a check).
"""
import sys, itertools, numpy as np
from pathlib import Path

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Track2/U-net/VariantAdaptiveKE/Logger/20260923")
se = lambda e: 100 / (1 + 0.5 * e); ff = lambda e: e / (0.5 + e); P_ = 0.6854
ON = dict(rel=94.9440053617, tke=78.6723530238, mvpe=94.1207421473, sps=43.6321996726); TM = 92.4802961403
def fin(L):
    w = {'rel': .5, 'tke': .1, 'mvpe': .1, 'time': .1, 'sps': .2}; L = dict(L, time=TM)
    return sum(w[k] * L[k] ** P_ for k in w) ** (1 / P_)
F0 = fin(ON)
AX = {'re': ['re0', 're1', 're2'], 'aoa': ['aoa0', 'aoa1']}


def box3(a):
    p = np.pad(a, ((1, 1), (1, 1)), mode='edge')
    return sum(p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)) / 9.0


def load(folds):
    Z = [dict(np.load(D / f'{fo}_T2.npz')) for fo in folds]
    return {k: np.concatenate([z[k] for z in Z]) for k in Z[0] if k != 'T'}


def score_windows(z, G):
    """G: (N,H,W) gain per window -> per-window rel, tke."""
    T = 20.0
    A, B, C = z['A'].astype(np.float64), z['B'].astype(np.float64), z['C'].astype(np.float64)
    g = G[..., None]
    sse = z['M'] + (g * g * A - 2 * g * B + C).reshape(len(A), -1).sum(1)
    rel = np.sqrt(np.clip(sse, 0, None)) / np.sqrt(np.clip(z['tn2'], 1e-16, None))
    kp = (G * G) * (A[..., 0] + A[..., 1]) / (2 * T); kt = (C[..., 0] + C[..., 1]) / (2 * T)
    tke = np.linalg.norm((kp - kt).reshape(len(A), -1), axis=1) / np.clip(np.linalg.norm(kt.reshape(len(A), -1), axis=1), 1e-8, None)
    return rel, tke


def gains(z, policy, **kw):
    T = 20.0
    kp = (z['A'][..., 0] + z['A'][..., 1]).astype(np.float64) / (2 * T)
    kt = (z['C'][..., 0] + z['C'][..., 1]).astype(np.float64) / (2 * T)
    ki = z['KEin'].astype(np.float64)
    N = len(kp); G = np.ones_like(kp); MU = np.zeros(N)
    files = z['files']
    if policy == 'fixed':
        mu = np.full_like(kp, kw['mu'])
    else:
        mu = np.zeros_like(kp)
        beta, mumax, warm, mu0 = kw['beta'], kw['mumax'], kw['warm'], kw['mu0']
        start = 0
        while start < N:
            end = start
            while end < N and files[end] == files[start]:
                end += 1
            ep = ei = None; pw = 1.0
            for k in range(start, end):
                kk = k - start
                if kk == 0 or ep is None:
                    m = np.full(kp.shape[1:], mu0)
                else:
                    Ep, Ei = ep / (1 - pw), ei / (1 - pw)
                    if policy == 'traj':
                        est = Ep.sum() / max(Ep.sum() + Ei.sum(), 1e-30); est = np.full(kp.shape[1:], est)
                    else:
                        est = box3(Ep) / np.clip(box3(Ep) + box3(Ei), 1e-30, None)
                    w = min(1.0, kk / warm)
                    m = np.clip(w * est + (1 - w) * mu0, 0, mumax)
                mu[k] = m
                dp, di = (kp[k] - kt[k]) ** 2, (ki[k] - kt[k]) ** 2      # revealed after window k is scored
                ep = dp * beta if ep is None else ep * (1 - beta) + dp * beta
                ei = di * beta if ei is None else ei * (1 - beta) + di * beta
                pw *= (1 - beta)
            start = end
    tg = (1 - mu) * kp + mu * ki
    G = np.clip(np.sqrt(np.clip(tg, 1e-12, None) / np.clip(kp, 1e-12, None)), 0.5, 2.0)
    return G, mu


def legs(z, rel, tke, sel):
    acc = 0.5 * (1 - ff(rel)) + 0.3 * (1 - ff(tke)) + 0.2 * (1 - ff(z['mvpe']))
    return dict(rel=se(rel[sel].mean()), tke=se(tke[sel].mean()), mvpe=se(z['mvpe'][sel].mean()), sps=100 * (acc[sel] * z['S'][sel]).sum() / z['n'][sel].sum())


def evaluate(policy, **kw):
    res = {}
    for ax, folds in AX.items():
        z = load(folds)
        re = np.array([int(str(x).split('_')[0]) for x in z['files']])
        G, mu = gains(z, policy, **kw); rel, tke = score_windows(z, G)
        G0 = np.ones_like(G); rel0, tke0 = score_windows(z, G0)
        for nm, sel in [('all', re > 0), ('hi', re >= 22875)]:
            b, c = legs(z, rel0, tke0, sel), legs(z, rel, tke, sel)
            res[(ax, nm)] = ({k: c[k] - b[k] for k in b}, float(mu[sel].mean()))
    m = {k: 0.5 * (res[('re', 'all')][0][k] + res[('aoa', 'all')][0][k]) for k in ON}
    dF = fin({k: ON[k] + m[k] for k in ON}) - F0
    dFn = fin({**{k: ON[k] + m[k] for k in ON}, 'sps': ON['sps']}) - F0
    return dF, dFn, res


if __name__ == '__main__':
    rows = []
    for mu in (0.1, 0.15, 0.2, 0.3):
        rows.append((f'fixed mu={mu}',) + evaluate('fixed', mu=mu))
    for pol, beta, mumax, warm in itertools.product(('traj', 'pix'), (0.1, 0.3), (0.3, 0.5), (4, 8)):
        rows.append((f'{pol} b={beta} max={mumax} warm={warm}',) + evaluate(pol, beta=beta, mumax=mumax, warm=warm, mu0=0.1))
    for name, dF, dFn, res in rows:
        cells = ' | '.join(f"{a}{n}: rel{r[0]['rel']:+.3f} tke{r[0]['tke']:+.3f} sps{r[0]['sps']:+.3f} mu{r[1]:.2f}" for (a, n), r in res.items())
        print(f'{name:32s} EQ dF {dF:+.4f} (noSPS {dFn:+.4f}) || {cells}')


# ---- follow-the-leader over a mu grid (per trajectory, causal) --------------------------------------------------
# Per-window first-order utility of the online Final at the v100 online level (exact linear decomposition of the
# window-averaged legs): U = -WR*rel_w - WT*tke_w + WS*acc_w*S_w/n_w.
MUS = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
_rbar = (100 / ON['rel'] - 1) / 0.5; _tbar = (100 / ON['tke'] - 1) / 0.5
WR = 0.477 * 50 / (1 + 0.5 * _rbar) ** 2; WT = 0.101 * 50 / (1 + 0.5 * _tbar) ** 2; WS = 0.244 * 100


def grid_scores(z):
    T = 20.0
    kp = (z['A'][..., 0] + z['A'][..., 1]).astype(np.float64) / (2 * T); ki = z['KEin'].astype(np.float64)
    R, K, U = [], [], []
    for mu in MUS:
        G = np.clip(np.sqrt(np.clip((1 - mu) * kp + mu * ki, 1e-12, None) / np.clip(kp, 1e-12, None)), 0.5, 2.0)
        rel, tke = score_windows(z, G)
        acc = 0.5 * (1 - ff(rel)) + 0.3 * (1 - ff(tke)) + 0.2 * (1 - ff(z['mvpe']))
        R.append(rel); K.append(tke); U.append(-WR * rel - WT * tke + WS * acc * z['S'] / z['n'])
    return np.array(R), np.array(K), np.array(U)


def ftl(z, U, lam=1.0, margin=0.0, prior=0.1, oracle=False):
    """choice index per window; oracle = best fixed mu per trajectory in hindsight (upper bound, not causal)."""
    files = z['files']; N = len(files); ch = np.zeros(N, int); i0 = int(np.argmin(np.abs(MUS - prior)))
    start = 0
    while start < N:
        end = start
        while end < N and files[end] == files[start]:
            end += 1
        if oracle:
            ch[start:end] = int(np.argmax(U[:, start:end].sum(1)))
        else:
            acc = np.zeros(len(MUS))
            for k in range(start, end):
                j = int(np.argmax(acc)); ch[k] = j if acc[j] - acc[i0] > margin else i0
                acc = lam * acc + U[:, k]
        start = end
    return ch


def eval_choice(policy_fn):
    res = {}
    for ax, folds in AX.items():
        z = load(folds); re = np.array([int(str(x).split('_')[0]) for x in z['files']])
        R, K, U = grid_scores(z); ch = policy_fn(z, U); idx = np.arange(len(ch))
        rel, tke = R[ch, idx], K[ch, idx]; i0 = int(np.argmin(np.abs(MUS - 0.0)))
        for nm, sel in [('all', re > 0), ('hi', re >= 22875)]:
            b, c = legs(z, R[i0], K[i0], sel), legs(z, rel, tke, sel)
            res[(ax, nm)] = ({k: c[k] - b[k] for k in b}, float(MUS[ch[sel]].mean()))
    m = {k: 0.5 * (res[('re', 'all')][0][k] + res[('aoa', 'all')][0][k]) for k in ON}
    return fin({k: ON[k] + m[k] for k in ON}) - F0, fin({**{k: ON[k] + m[k] for k in ON}, 'sps': ON['sps']}) - F0, res


def main_ftl():
    rows = [('oracle per-trajectory',) + eval_choice(lambda z, U: ftl(z, U, oracle=True))]
    for lam, margin in itertools.product((1.0, 0.95, 0.9), (0.0, 0.05, 0.2)):
        rows.append((f'ftl lam={lam} margin={margin}',) + eval_choice(lambda z, U, l=lam, m=margin: ftl(z, U, l, m)))
    for name, dF, dFn, res in rows:
        cells = ' | '.join(f"{a}{n}: rel{r[0]['rel']:+.3f} tke{r[0]['tke']:+.3f} sps{r[0]['sps']:+.3f} mu{r[1]:.2f}" for (a, n), r in res.items())
        print(f'{name:32s} EQ dF {dF:+.4f} (noSPS {dFn:+.4f}) || {cells}')


# ---- the card configuration (common/ke_ftl_card_t2.py): grid 0..0.4 step .05, U = -WR rel - WT tke, prior .1 ---
CARD_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]


def card_config_per_fold():
    global MUS, WS
    MUS, WS = np.array(CARD_GRID), 0.0
    out = {}
    for fo in ['re0', 're1', 're2', 'aoa0', 'aoa1']:
        z = load([fo]); R, K, U = grid_scores(z); ch = ftl(z, U, 1.0, 0.0, 0.1); idx = np.arange(len(ch))
        i0 = int(np.argmin(np.abs(MUS))); sel = np.ones(len(ch), bool)
        b, c = legs(z, R[i0], K[i0], sel), legs(z, R[ch, idx], K[ch, idx], sel)
        out[fo] = dict(delta={k: c[k] - b[k] for k in b}, mean_mu=float(MUS[ch].mean()), mu_seq_first_file=MUS[ch[z['files'] == z['files'][0]]].tolist())
    return out


# ---- v107 configuration: (mu, c) FTL + per-(lead, u/v) band-width FTL (common/ke_ftl_card_t2.py --gains --band-grid) ----
V107_GAINS = [0.95, 1.0, 1.05]
V107_BAND = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4]
BAND_BASE = Path("Track2/U-net/VariantMeanAnchorT2/Logger/20260923/dgxbase__%s_T2.npz")   # same-host m-grid baseline


def _traj_ftl(files, U, i0):
    """U: (N, G, K) utilities; per trajectory and per column k, FTL with ties to i0.  Returns (N, K) choices."""
    N = len(files); ch = np.zeros((N, U.shape[2]), int); s = 0
    while s < N:
        e = s
        while e < N and files[e] == files[s]:
            e += 1
        a = np.zeros(U.shape[1:])
        for k in range(s, e):
            d = a - a[i0][None]; d[i0] += 1e-9; ch[k] = np.argmax(d, axis=0); a = a + U[k]
        s = e
    return ch


def v107_fold(fo, gains=V107_GAINS, band=V107_BAND):
    T = 20.0
    z = load([fo]); b = dict(np.load(str(BAND_BASE) % fo))
    assert (b['files'] == z['files']).all() and (b['t0'] == z['t0']).all()
    kp = (z['A'][..., 0] + z['A'][..., 1]).astype(np.float64) / (2 * T); ki = z['KEin'].astype(np.float64)
    R, K, U, lab = [], [], [], []
    for mu in CARD_GRID:
        G0 = np.clip(np.sqrt(np.clip(kp + mu * (ki - kp), 1e-12, None) / np.clip(kp, 1e-12, None)), 0.5, 2.0)
        for c in gains:
            r, k = score_windows(z, c * G0); R.append(r); K.append(k); U.append(-WR * r - WT * k); lab.append((mu, c))
    R, K, U = map(np.array, (R, K, U)); i0 = lab.index((0.1, 1.0)); ib = lab.index((0.0, 1.0))
    ch = _traj_ftl(z['files'], U.T[:, :, None], i0)[:, 0]; idx = np.arange(len(ch))
    mg = list(b['mgrid']); mi = [mg.index(m) for m in band]
    Su = b['S_mid_cf'][:, mi].reshape(len(ch), len(mi), -1)            # (N, G, 40)
    bch = _traj_ftl(z['files'], Su, band.index(1.0))
    S1 = np.take_along_axis(Su, bch[:, None, :], axis=1)[:, 0].sum(1); S0 = Su[:, band.index(1.0)].sum(1)
    rel, tke = R[ch, idx], K[ch, idx]
    acc = lambda r, t: 0.5 * (1 - ff(r)) + 0.3 * (1 - ff(t)) + 0.2 * (1 - ff(z['mvpe']))
    base = dict(rel=se(R[ib].mean()), tke=se(K[ib].mean()), sps=100 * (acc(R[ib], K[ib]) * S0).sum() / b['n'].sum())
    new = dict(rel=se(rel.mean()), tke=se(tke.mean()), sps=100 * (acc(rel, tke) * S1).sum() / b['n'].sum())
    return {k: new[k] - base[k] for k in base}, float(np.mean([lab[j][0] for j in ch])), float(np.mean([lab[j][1] for j in ch]))


# ---- v108 configuration: v107 with one (mu, c) leader per 8x8 spatial block (common/ke_ftl_card_t2.py --blocks 8x8) ----
def v108_fold(fo, bh=8, bw=8, gains=V107_GAINS, band=V107_BAND):
    T = 20.0
    z = load([fo]); b = dict(np.load(str(BAND_BASE) % fo))
    assert (b['files'] == z['files']).all() and (b['t0'] == z['t0']).all()
    A = (z['A'][..., 0] + z['A'][..., 1]).astype(np.float64); B = (z['B'][..., 0] + z['B'][..., 1]).astype(np.float64)
    C = (z['C'][..., 0] + z['C'][..., 1]).astype(np.float64)
    kp, kt, ki = A / (2 * T), C / (2 * T), z['KEin'].astype(np.float64); N, H, W = kp.shape; nb = (H // bh) * (W // bw)
    rel0, tke0 = score_windows(z, np.ones_like(kp)); ktn = (kt.reshape(N, -1) ** 2).sum(1)
    blk = lambda x: x.reshape(N, H // bh, bh, W // bw, bw).sum(axis=(2, 4)).reshape(N, nb)
    arms = [(m, c) for m in CARD_GRID for c in gains]; i0 = arms.index((0.1, 1.0))
    Ub, G = [], []
    for (m, c) in arms:
        g = c * np.clip(np.sqrt(np.clip(kp + m * (ki - kp), 1e-12, None) / np.clip(kp, 1e-12, None)), 0.5, 2.0); G.append(g.astype(np.float32))
        Ub.append(-WR * blk(g * g * A - 2 * g * B + C) / (2 * rel0[:, None] * z['tn2'][:, None])
                  - WT * blk((g * g * kp - kt) ** 2) / (2 * np.clip(tke0, 1e-8, None)[:, None] * np.clip(ktn, 1e-16, None)[:, None]))
    ch = _traj_ftl(z['files'], np.transpose(np.array(Ub), (1, 0, 2)), i0)            # (N, nb)
    chp = np.repeat(np.repeat(ch.reshape(N, H // bh, W // bw), bh, 1), bw, 2)
    rel, tke = score_windows(z, np.take_along_axis(np.array(G), chp[None], axis=0)[0])
    mg = list(b['mgrid']); mi = [mg.index(m) for m in band]
    Su = b['S_mid_cf'][:, mi].reshape(N, len(mi), -1); bch = _traj_ftl(z['files'], Su, band.index(1.0))
    S1 = np.take_along_axis(Su, bch[:, None, :], axis=1)[:, 0].sum(1); S0 = Su[:, band.index(1.0)].sum(1)
    acc = lambda r, t: 0.5 * (1 - ff(r)) + 0.3 * (1 - ff(t)) + 0.2 * (1 - ff(z['mvpe']))
    base = dict(rel=se(rel0.mean()), tke=se(tke0.mean()), sps=100 * (acc(rel0, tke0) * S0).sum() / b['n'].sum())
    new = dict(rel=se(rel.mean()), tke=se(tke.mean()), sps=100 * (acc(rel, tke) * S1).sum() / b['n'].sum())
    return {k: new[k] - base[k] for k in base}
