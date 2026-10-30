#!/usr/bin/env python3
"""T1 regime-conditioned static anchor maps: K input clusters, each with its own 8x8 KE-anchor mu map and time-mean
anchor lam map.  The cluster of a window is the nearest k-means centroid of its standardized INPUT time-mean block
features (8x8 block means of u, v; inductive, window-local).  Centroids, standardization and maps are fitted on the
out-of-fold sufficient statistics of the fixed-fivefold v90 fold cards (common/ke_stats_probe.py, common/mean_stats_probe.py);
evaluation is cross-fold within each axis (model fitted on the axis' other folds, scored exactly on the held-out fold).
Writes cluster_maps_K{K}.json (all-fold model for the all81 card) and cluster_maps_K{K}_crossfold.json (per-fold models
and expected deltas for real fold-card checks).

    python Track1/U-net/VariantKEAnchor/cluster_maps_sim.py [--K 3] [--seed 0]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.argv, _argv = [sys.argv[0]], sys.argv[1:]
sys.path.insert(0, str(REPO / "Track2/U-net/VariantAdaptiveKE"))
import simulate as S  # noqa: E402
import mean_anchor_sim as MA  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--K", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--stats", default=str(REPO / "Track2/U-net/VariantAdaptiveKE/Logger/20260923/T1stats"))
ap.add_argument("--means", default=str(REPO / "Track2/U-net/VariantAdaptiveKE/Logger/20260923/meanstats"))
ap.add_argument("--out", default=str(REPO / "Track1/U-net/VariantKEAnchor/config"))
a = ap.parse_args(_argv)
MA.D0 = Path(a.means)
S.ON = dict(rel=94.8509965935, tke=78.6560024934, mvpe=94.1401099639, sps=43.5291262123); S.TM = 92.8762136464
S.F0 = S.fin(S.ON)
_rb, _tb = (100 / S.ON['rel'] - 1) / 0.5, (100 / S.ON['tke'] - 1) / 0.5
WR, WT = 0.477 * 50 / (1 + 0.5 * _rb) ** 2, 0.101 * 50 / (1 + 0.5 * _tb) ** 2
LAMS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]; MU = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
MA.LAMS = LAMS
T, BH, BW = 20.0, 8, 8
FOLDS = ['re0', 're1', 're2', 'aoa0', 'aoa1']


def prep(fo):
    z = dict(np.load(Path(a.stats) / f'{fo}_T1.npz')); zm = MA.load(1, [fo])
    assert (z['files'] == zm['files']).all() and (z['t0'] == zm['t0']).all()
    A = (z['A'][..., 0] + z['A'][..., 1]).astype(np.float64); B = (z['B'][..., 0] + z['B'][..., 1]).astype(np.float64)
    C = (z['C'][..., 0] + z['C'][..., 1]).astype(np.float64)
    kp, kt, ki = A / (2 * T), C / (2 * T), z['KEin'].astype(np.float64); N, H, W = kp.shape; nh, nw = H // BH, W // BW
    rel0, tke0 = S.score_windows(z, np.ones_like(kp)); ktn = (kt.reshape(N, -1) ** 2).sum(1)
    blk = lambda x: x.reshape(N, nh, BH, nw, BW).sum(axis=(2, 4)).reshape(N, nh * nw)
    Um, G = [], []
    for m in MU:
        g = np.clip(np.sqrt(np.clip(kp + m * (ki - kp), 1e-12, None) / np.clip(kp, 1e-12, None)), 0.5, 2.0); G.append(g.astype(np.float32))
        Um.append(-WR * blk(g * g * A - 2 * g * B + C) / (2 * rel0[:, None] * z['tn2'][:, None])
                  - WT * blk((g * g * kp - kt) ** 2) / (2 * np.clip(tke0, 1e-8, None)[:, None] * np.clip(ktn, 1e-16, None)[:, None]))
    Ul = [MA.block_util(zm, l, BH, BW) for l in LAMS]
    xbar = (zm['E'] + zm['D'] + zm['TB']).astype(np.float64)                       # input time-mean, physical (N,H,W,2)
    feat = xbar.reshape(N, nh, BH, nw, BW, 2).mean(axis=(2, 4)).reshape(N, -1)     # (N, nh*nw*2), order (nh, nw, c)
    return dict(z=z, zm=zm, Um=np.array(Um), Ul=np.array(Ul), G=np.array(G), A=A, B=B, C=C, kp=kp, kt=kt, feat=feat,
                rel0=rel0, tke0=tke0, N=N, nh=nh, nw=nw)


P = {fo: prep(fo) for fo in FOLDS}
nh, nw = P['re0']['nh'], P['re0']['nw']


def kmeans(X, K, it=50, seed=0):
    rng = np.random.default_rng(seed); c = X[rng.choice(len(X), K, replace=False)].copy()
    for _ in range(it):
        lab = np.argmin(((X[:, None] - c[None]) ** 2).sum(-1), 1)
        for k in range(K):
            if (lab == k).any():
                c[k] = X[lab == k].mean(0)
    return c


def fit(folds):
    X = np.concatenate([P[o]['feat'] for o in folds]); mu_, sd_ = X.mean(0), X.std(0) + 1e-9
    cen = kmeans((X - mu_) / sd_, a.K, seed=a.seed)
    lab = lambda F: np.argmin(((((F - mu_) / sd_)[:, None] - cen[None]) ** 2).sum(-1), 1)
    mu_maps, lam_maps = [], []
    for k in range(a.K):
        sm = sum(P[o]['Um'][:, lab(P[o]['feat']) == k].sum(1) for o in folds)
        sl = sum(P[o]['Ul'][:, lab(P[o]['feat']) == k].sum(1) for o in folds)
        mu_maps.append(np.array(MU)[np.argmax(sm, 0)].reshape(nh, nw)); lam_maps.append(np.array(LAMS)[np.argmax(sl, 0)].reshape(nh, nw))
    return dict(feat_mean=mu_.tolist(), feat_std=sd_.tolist(), centroids=cen.tolist(),
                mu_maps=[m.tolist() for m in mu_maps], lam_maps=[m.tolist() for m in lam_maps]), lab


def score(fo, model, lab):
    te = P[fo]; N = te['N']; lt = lab(te['feat'])
    mi = np.array([[MU.index(float(v)) for v in np.array(model['mu_maps'][k]).ravel()] for k in lt])
    lv = np.array([np.array(model['lam_maps'][k]).ravel() for k in lt])
    ex = lambda q: np.repeat(np.repeat(q.reshape(N, nh, nw), BH, 1), BW, 2)
    g = np.take_along_axis(te['G'], ex(mi)[None], axis=0)[0].astype(np.float64)
    fl = (g ** 2 * te['A'] - 2 * g * te['B'] + te['C']).reshape(N, -1).sum(1)
    tke = np.linalg.norm((g * g * te['kp'] - te['kt']).reshape(N, -1), axis=1) / np.clip(np.linalg.norm(te['kt'].reshape(N, -1), axis=1), 1e-8, None)
    lam = ex(lv); zm = te['zm']
    Dp = zm['D'].astype(np.float64) + lam[..., None] * zm['E'].astype(np.float64)
    rel = np.sqrt((T * (Dp ** 2).reshape(N, -1).sum(1) + fl) / te['z']['tn2']); _, mv = MA.metrics(zm, lam)
    acc = 0.5 * (1 - S.ff(rel)) + 0.3 * (1 - S.ff(tke)) + 0.2 * (1 - S.ff(mv))
    new = dict(rel=S.se(rel.mean()), tke=S.se(tke.mean()), mvpe=S.se(mv.mean()), sps=100 * (acc * te['z']['S']).sum() / te['z']['n'].sum())
    base = S.legs(te['z'], te['rel0'], te['tke0'], np.ones(N, bool))
    return {k: float(new[k] - base[k]) for k in base}, np.bincount(lt, minlength=a.K).tolist()


res, per_fold, models = {}, {}, {}
for ax, folds in S.AX.items():
    acc, nt = {}, 0
    for fo in folds:
        model, lab = fit([o for o in folds if o != fo]); models[fo] = model
        d, counts = score(fo, model, lab); per_fold[fo] = dict(delta=d, cluster_counts=counts); n = P[fo]['N']
        for k in d: acc[k] = acc.get(k, 0.0) + d[k] * n
        nt += n
    res[ax] = {k: v / nt for k, v in acc.items()}
m = {k: 0.5 * (res['re'][k] + res['aoa'][k]) for k in S.ON}
dF = S.fin({k: S.ON[k] + m[k] for k in S.ON}) - S.F0
final, _ = fit(FOLDS)
print(f'K={a.K} seed={a.seed} cross-fold EQ dF {dF:+.4f}', {k: round(float(v), 4) for k, v in m.items()})
for fo in FOLDS:
    print(' ', fo, {k: round(v, 4) for k, v in per_fold[fo]['delta'].items()}, 'clusters', per_fold[fo]['cluster_counts'])
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
spec = dict(K=a.K, seed=a.seed, block=[BH, BW], feature="8x8 block means of the input window's time-mean u,v (physical), order (row, col, channel)",
            mu_grid=MU, lam_grid=LAMS)
json.dump(dict(spec, **final, crossfold_equal_axis_dF=dF, crossfold_legs=m), open(out / f'cluster_maps_K{a.K}.json', 'w'), indent=1)
json.dump(dict(spec, models=models, expected=per_fold), open(out / f'cluster_maps_K{a.K}_crossfold.json', 'w'), indent=1)
