#!/usr/bin/env python3
"""A LEARNED, sample-dependent variance-deficit field -- the last large opportunity measured.

WHERE THE +5 IS.  `t1_kegain.py` walked an adaptivity ladder over the fluctuation gain in
`p -> mean_t(p) + g*(p - mean_t(p))`.  Because `scoring.kinetic_energy` subtracts the SAME temporal
mean, `KE(p') = g**2 * KE(p)` holds EXACTLY, so the ladder is arithmetic:

    L1  g scalar          oracle  +0.021 FINAL
    L2  g[y,x] static     oracle  -1.269   (overfits when fitted freely)
    L3  g[n]              oracle  +0.196
    L4  g[n,y]            oracle  +0.425
    L5  g[n,y,x]          oracle  +4.579   <- tke -> exactly 100

Scaled to the online v9 numbers that is **~+5.5 FINAL**, against a 0.074 gap to first place.  Every
rung below L5 is nearly worthless, and every hand-written realizable estimator failed (+0.034 at
best).  Track 2's revealed-target version was then measured and CLOSED: on the real bundle it is
redundant with the frozen static field (`Track2/U-net/README.md`).

So the field is not unpredictable -- nobody has LEARNED it.  `g*[n,y,x] = sqrt(KE_t / KE_p)` is
fully supervised, computable on every training window, and reaches the metric in closed form.

WHY THIS IS CHEAP.  Both subscores that move are closed forms in per-pixel SECOND-ORDER statistics.
With `p_f = p - mean_t(p)` and `t_f = t - mean_t(t)`, the cross term `sum_t (g*p_f - t_f)` vanishes
because both factors have zero temporal mean, so

    ||p' - t||^2 = T * sum_{y,x,c} (pbar - tbar)^2  +  sum_{y,x,c} [ g^2 A - 2 g B + C ]
        A = sum_t p_f^2      B = sum_t p_f t_f      C = sum_t t_f^2          (all (H,W,C))
    KE_p = (A_u + A_v) / 2T                  KE_t = (C_u + C_v) / 2T

EXACTLY.  So training never touches a 20-frame tensor: cache A, B, C and two scalars per window and
the whole study runs on ~100 KB/window.  `mvpe` needs no term at all -- the correction preserves
`mean_t(p)` by construction, so it is invariant, which is also this module's self-test.

THE LOSS IS THE SCORER.  `-(score_error(rel) + score_error(tke))` with the official
`score_error(e) = 100/(1+0.5e)`.  No hand-tuned trade-off weight: the scorer's own curvature does
the weighting (at our operating point d(score)/de is -45.8 for rel and -28.5 for tke, so a unit of
rel error costs 1.6x a unit of tke error -- exactly what the competition says it costs).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-12
N_FEAT = 7
N_FEAT_RICH = 15     # VG2 (2026-08-06): + scale-resolved and oscillator descriptors


def score_error(e):
    """scoring.py::score_error, differentiable."""
    return 100.0 / (1.0 + 0.5 * e)


def _hp_var(a):
    """Variance of the SMALL-SCALE part: field minus its 3x3 spatial mean.
    tke lives in the low-energy tail of the spectrum (measured: 64 POD modes hold
    99.14% of the energy but cap tke at 75.41), and the damping is band-dependent, so a
    total-variance feature cannot express it."""
    import torch.nn.functional as _F
    n, T_, H, W, C = a.shape
    z = a.permute(0, 1, 4, 2, 3).reshape(n * T_ * C, 1, H, W)
    sm = _F.avg_pool2d(_F.pad(z, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    hp = (z - sm).reshape(n, T_, C, H, W).permute(0, 1, 3, 4, 2)
    return (hp - hp.mean(1, keepdim=True)).pow(2).mean(1).sum(-1)      # (n,H,W)


def _lag1(a):
    """Per-pixel lag-1 temporal autocorrelation of the fluctuation, summed over u,v.
    An over-smoothed forecast is MORE autocorrelated than the truth, so this is a
    direct signature of the deficit that total variance cannot see."""
    f = a - a.mean(1, keepdim=True)
    num = (f[:, 1:] * f[:, :-1]).mean(1).sum(-1)
    den = (f * f).mean(1).sum(-1).clamp_min(EPS)
    return num / den                                                   # (n,H,W)


def _osc(x_in):
    """Per-window oscillator descriptors from the input window's own POD quadrature
    pair: shedding periods per window, phase-fit R^2, and sigma1/sigma2.  Measured
    2026-08-06: the pair is detected in 80-100% of windows and the phase fit has
    R^2 ~ 0.99, so this is real information the head has never had."""
    n, T_, H, W, C = x_in.shape
    f = (x_in[..., :2] - x_in[..., :2].mean(1, keepdim=True)).reshape(n, T_, -1)
    U, S, _ = torch.linalg.svd(f.float(), full_matrices=False)
    a1, a2 = U[:, :, 0] * S[:, :1], U[:, :, 1] * S[:, 1:2]
    ph = torch.atan2(a2, a1)
    d = ph[:, 1:] - ph[:, :-1]
    d = d - 2 * torch.pi * torch.round(d / (2 * torch.pi))            # unwrap
    ph = torch.cat([ph[:, :1], ph[:, :1] + torch.cumsum(d, 1)], 1)
    tt = torch.arange(T_, device=ph.device, dtype=ph.dtype)
    tb = tt - tt.mean()
    sl = ((ph - ph.mean(1, keepdim=True)) * tb).sum(1) / (tb * tb).sum()
    fit = ph.mean(1, keepdim=True) + sl[:, None] * tb[None]
    ss = 1.0 - ((ph - fit) ** 2).sum(1) / ((ph - ph.mean(1, keepdim=True)) ** 2).sum(1).clamp_min(EPS)
    per = sl.abs() * T_ / (2 * torch.pi)
    rat = S[:, 0] / S[:, 1].clamp_min(EPS)
    return per, ss, rat


def stats_from_window(p, t, x_in, T=None, rich=False):
    """Per-window second-order statistics + head features, from PHYSICAL (N,T,H,W,>=2) tensors.

    Returns a dict of small tensors; nothing here keeps a T-length axis, which is the whole point.
    `x_in` is the 20-frame INPUT window -- the only extra information `predict()` actually has.
    """
    T = T or p.shape[1]
    p2, t2, i2 = p[..., :2], t[..., :2], x_in[..., :2]
    pb, tb, ib = p2.mean(1), t2.mean(1), i2.mean(1)                 # (N,H,W,2)
    pf, tf = p2 - pb[:, None], t2 - tb[:, None]
    A = (pf * pf).sum(1)                                            # (N,H,W,2)
    B = (pf * tf).sum(1)
    C = (tf * tf).sum(1)
    M = T * ((pb - tb) ** 2).flatten(1).sum(1)                      # (N,)
    tn2 = (t2 * t2).flatten(1).sum(1)                               # (N,)
    KEp = 0.5 * (A[..., 0] + A[..., 1]) / T                         # (N,H,W)
    KEt = 0.5 * (C[..., 0] + C[..., 1]) / T
    KEi = 0.5 * ((i2 - ib[:, None]) ** 2).mean(1).sum(-1)           # input window's own variance
    feat = torch.stack([
        torch.log(KEp + EPS), torch.log(KEi + EPS),
        torch.log(KEi + EPS) - torch.log(KEp + EPS),   # the R-in estimator, as a FEATURE not a rule
        pb[..., 0], pb[..., 1], ib[..., 0], ib[..., 1],
    ], dim=1)                                                       # (N, N_FEAT, H, W)
    if rich:
        hp_p, hp_i = _hp_var(p2), _hp_var(i2)
        l1p, l1i = _lag1(p2), _lag1(i2)
        per, ss, rat = _osc(x_in)
        bc = lambda v: v.reshape(-1, 1, 1).expand(-1, feat.shape[-2], feat.shape[-1])
        feat = torch.cat([feat, torch.stack([
            torch.log(hp_p + EPS), torch.log(hp_i + EPS),
            torch.log(hp_i + EPS) - torch.log(hp_p + EPS),
            l1p, l1i,
            bc(per), bc(ss), bc(torch.log(rat + EPS)),
        ], dim=1)], dim=1)                                          # (N, 15, H, W)
    return {"A": A, "B": B, "C": C, "M": M, "tn2": tn2, "KEp": KEp, "KEt": KEt,
            "feat": feat, "T": T}


def subscores(g, A, B, C, M, tn2, T):
    """(rel_l2, tke_rel_l2) per sample after applying the gain field g (N,H,W).  Exact, see header."""
    g2 = g * g
    KEp = 0.5 * (A[..., 0] + A[..., 1]) / T
    KEt = 0.5 * (C[..., 0] + C[..., 1]) / T
    num = torch.linalg.vector_norm((g2 * KEp - KEt).flatten(1), dim=1)
    den = torch.linalg.vector_norm(KEt.flatten(1), dim=1).clamp_min(1e-8)
    tke = num / den
    gu = g.unsqueeze(-1)
    sse = M + (g2.unsqueeze(-1) * A - 2.0 * gu * B + C).flatten(1).sum(1)
    rel = sse.clamp_min(0.0).sqrt() / tn2.sqrt().clamp_min(1e-8)
    return rel, tke


import os as _os_w
# Weight on the tke term relative to rel.  1.0 reproduces the historical equal-weight loss
# (correct under the pre-2026-08-08 shared-weight assumption).  The solved leaderboard weights
# give an effective ratio tke/rel = 0.1463/0.5424 = 0.2697 at the live operating point.
# 0.2697 = w_tke/w_rel at the live operating point, and the value the SHIPPED head was
# trained at.  The old default of 1.0 was correct only while the three accuracy subscores
# were believed to share one weight -- refuted 2026-08-08 at 102x the free-fit residual --
# and it silently trained a DIFFERENT head for any caller that did not set the env var.
W_TKE = float(_os_w.environ.get("VGAIN_W_TKE", "0.2697"))


def loss_from_stats(g, A, B, C, M, tn2, T):
    """The negative of the (weighted) subscores the correction can move.  At W_TKE = 1.0 this
    is bit-identical to the historical loss; the docstring claim "minimising this IS the
    metric" was true under equal Final weights and is now conditional on W_TKE."""
    rel, tke = subscores(g, A, B, C, M, tn2, T)
    return -(score_error(rel) + W_TKE * score_error(tke)).mean(), rel, tke


def oracle_gain(KEp, KEt, gmin, gmax):
    """L5: the pointwise tke optimum.  Zeroes the tke error by identity (up to the clamp)."""
    return (KEt / KEp.clamp_min(EPS)).clamp_min(0.0).sqrt().clamp(gmin, gmax)


class VGainHead(nn.Module):
    """(7 summary fields at H x W) -> log g[y,x].  Dilated so the receptive field spans the grid.

    Deliberately a statistics-to-statistics map, NOT a map from the 20-frame stack: the target is
    itself a second-order statistic, and the submission pays this net's cost inside the timed
    region.  Four 3x3 convs at 32 channels on 32x64 is ~75 MFLOP -- microseconds beside the U-net.

    The final conv is ZERO-initialised, so an untrained head outputs g == 1 exactly and is a no-op.
    A run that fails to learn therefore degrades to the current behaviour rather than to noise.
    """

    def __init__(self, n_feat=N_FEAT, width=32, gmin=0.5, gmax=3.0):
        super().__init__()
        self.gmin, self.gmax = float(gmin), float(gmax)
        self.c1 = nn.Conv2d(n_feat, width, 3, padding=1)
        self.c2 = nn.Conv2d(width, width, 3, padding=2, dilation=2)
        self.c3 = nn.Conv2d(width, width, 3, padding=4, dilation=4)
        self.c4 = nn.Conv2d(width, width, 3, padding=8, dilation=8)
        self.out = nn.Conv2d(width, 1, 1)
        # a global branch, so the scalar level (the rung v5 ships) is available without the
        # convolutions having to synthesise it from a finite receptive field
        self.gpool = nn.Linear(width, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.gpool.weight); nn.init.zeros_(self.gpool.bias)
        self.register_buffer("f_mu", torch.zeros(n_feat))
        self.register_buffer("f_sd", torch.ones(n_feat))

    def set_feature_norm(self, mu, sd):
        self.f_mu.copy_(mu.reshape(-1))
        self.f_sd.copy_(sd.reshape(-1).clamp_min(1e-6))

    def forward(self, feat):
        z = (feat - self.f_mu.reshape(1, -1, 1, 1)) / self.f_sd.reshape(1, -1, 1, 1)
        h = F.gelu(self.c1(z))
        h = F.gelu(self.c2(h))
        h = F.gelu(self.c3(h))
        h = F.gelu(self.c4(h))
        lg = self.out(h) + self.gpool(h.mean(dim=(2, 3))).reshape(-1, 1, 1, 1)
        import math
        return lg.squeeze(1).clamp(math.log(self.gmin), math.log(self.gmax)).exp()


def selfcheck(device="cpu"):
    """mvpe invariance + the closed form against a brute-force evaluation.  Cheap, run at startup."""
    torch.manual_seed(0)
    N, T, H, W = 3, 20, 32, 64
    p = torch.randn(N, T, H, W, 3, device=device) * 0.1 + 0.15
    t = torch.randn(N, T, H, W, 3, device=device) * 0.1 + 0.15
    xi = torch.randn(N, T, H, W, 3, device=device) * 0.1 + 0.15
    g = 0.7 + 0.9 * torch.rand(N, H, W, device=device)
    s = stats_from_window(p, t, xi)
    rel, tke = subscores(g, s["A"], s["B"], s["C"], s["M"], s["tn2"], T)

    pm = p.mean(1, keepdim=True)
    pc = pm + (p - pm) * g[:, None, :, :, None]
    d = (pc[..., :2] - t[..., :2]).flatten(1)
    rel_b = torch.linalg.vector_norm(d, dim=1) / torch.linalg.vector_norm(t[..., :2].flatten(1), dim=1)

    def ke(x):
        u, v = x[..., 0], x[..., 1]
        return 0.5 * (((u - u.mean(1, keepdim=True)) ** 2).mean(1)
                      + ((v - v.mean(1, keepdim=True)) ** 2).mean(1))
    kb = ke(pc) - ke(t)
    tke_b = (torch.linalg.vector_norm(kb.flatten(1), dim=1)
             / torch.linalg.vector_norm(ke(t).flatten(1), dim=1))

    e_rel = float((rel - rel_b).abs().max())
    e_tke = float((tke - tke_b).abs().max())
    e_mv = float((pc.mean(1) - p.mean(1)).abs().max())          # mvpe invariance
    ok = e_rel < 1e-5 and e_tke < 1e-5 and e_mv < 1e-6
    print(f"# vgain selfcheck  rel {e_rel:.2e}  tke {e_tke:.2e}  mean_t-invariance {e_mv:.2e}  "
          f"-> {'OK' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("vgain_head closed form disagrees with brute force")
    return True


if __name__ == "__main__":
    selfcheck()
    m = VGainHead()
    n = sum(p.numel() for p in m.parameters())
    f = torch.randn(2, N_FEAT, 32, 64)
    g = m(f)
    print(f"# VGainHead params {n/1e3:.1f}k   out {tuple(g.shape)}  "
          f"g in [{float(g.min()):.6f}, {float(g.max()):.6f}] (zero-init => exactly 1)")
    assert abs(float(g.min()) - 1.0) < 1e-6 and abs(float(g.max()) - 1.0) < 1e-6
    print("VGAIN_HEAD_OK")
