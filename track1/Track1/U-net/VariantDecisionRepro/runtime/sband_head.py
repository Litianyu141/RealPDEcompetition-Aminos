"""sband_head.py -- the LEARNED per-element residual-scale head for the Track-1 SPS band.

WHAT IT REPLACES.  submission_t1_v33.py::_activity builds the band's per-element scale from a
HAND-DESIGNED observable: the temporal standard deviation of each input pixel.  Measured against
the oracle |r| it ranks at Spearman 0.57 (top-decile lift 3.6) where the oracle is 1.00 (4.8).
This module learns the observable instead.

TWO DESIGN CHOICES THAT ARE NOT FREE, both measured on this project:

1. INPUT-ONLY FEATURES.  The head never sees a prediction.  A PERFECT oracle |r| fitted for one
   forecaster retains only 28.5% of its sps gain when applied to another -- and the same-recipe,
   different-seed pair retains 29.0%, so ~71% of the element-level residual field is the
   realisation of one weight vector and carries no information about any other model.  Any
   prediction-derived feature is model-specific by construction, so it is excluded.  (This is the
   mechanism behind v28's 89% and RSa's 63% fold-model -> ship-model evaporation.)

2. ONLY THE RANK OF THE OUTPUT IS EVER USED.  submission.py quantile-bins this net's output per
   (lead, channel) and looks up an empirically-fitted interval.  Any strictly monotone rescaling of
   the output -- including a whole-array unit error -- is therefore a BIT-EXACT no-op on the shipped
   bundle.  The normalisation trap that has been hit twice on this project is designed out here
   rather than guarded against.

UNITS.  `sband_features` consumes the PHYSICAL input window (m/s), matching `_activity`'s stated
convention and the vgain head's feature convention.  Every temporal std uses unbiased=False (ddof=0)
for the same reason `_activity` does.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

N_FEAT = 11
# v2 adds nine planes that measure UNPREDICTABILITY and SHEAR STRUCTURE
# rather than amplitude.  See the `feat == "v2"` branch for why each one is
# not a function of the eleven.
N_FEAT_V2 = 20
EPS = 1e-6


def _coord_planes(b, h, w, dev, dt):
    """Two normalised absolute-position planes in [-1, 1].  This is the ONLY thing in the feature
    stack that is not translation-equivariant, and it is the whole point of it."""
    yy = torch.linspace(-1.0, 1.0, h, device=dev, dtype=dt).reshape(1, 1, h, 1).expand(b, 1, h, w)
    xx = torch.linspace(-1.0, 1.0, w, device=dev, dtype=dt).reshape(1, 1, 1, w).expand(b, 1, h, w)
    return torch.cat([yy, xx], dim=1)


def sband_features(x_ph, n_coord=0, feat="v1"):
    """PHYSICAL (B, T, H, W, >=2) -> (B, N_FEAT, H, W) float32 summary maps.

    Kept deliberately cheap: this runs inside the timed region.  All eleven planes are functions
    of the input window alone.
    """
    u = x_ph[..., :2].float()
    sd = u.std(1, unbiased=False).clamp_min(0.0)                       # (B,H,W,2)
    mn = u.mean(1)
    if u.shape[1] > 1:
        d = (u[:, 1:] - u[:, :-1]).pow(2).mean(1).clamp_min(0.0).sqrt()
    else:
        d = torch.zeros_like(sd)
    sp = mn.pow(2).sum(-1).clamp_min(0.0).sqrt()                       # (B,H,W)
    gx = torch.zeros_like(sp)
    gy = torch.zeros_like(sp)
    gx[:, :, 1:-1] = sp[:, :, 2:] - sp[:, :, :-2]
    gy[:, 1:-1] = sp[:, 2:] - sp[:, :-2]
    g = (gx * gx + gy * gy).clamp_min(0.0).sqrt()
    stat = (sd.sum(-1) < 1e-9).to(sd.dtype)                            # the static-cell indicator
    lastd = (u[:, -1] - mn).abs()
    win = sd.mean(dim=(1, 2), keepdim=True).expand_as(sd)              # per-sample activity level

    def L(a):
        return torch.log(a + EPS)

    f = torch.stack([L(sd[..., 0]), L(sd[..., 1]), mn[..., 0], mn[..., 1],
                     L(d[..., 0]), L(d[..., 1]), L(g), stat,
                     L(lastd[..., 0]), L(lastd[..., 1]), L(win[..., 0])], dim=1)

    if feat == "v2":
        # ---- nine ADDITIONAL planes.  Every one is a pointwise reduction over the 20-frame
        # window or a 3x3 central difference, so the cost class is the same as the eleven above.
        #
        # WHY THESE AND NOT MORE OF THE SAME.  rms(du/dt)^2 ~= 2*sd^2*(1 - rho_1), so features
        # 1,2,5,6 already CONTAIN the lag-1 autocorrelation; anything reconstructible from the
        # eleven is excluded here by construction.  What is genuinely absent is (i) temporal
        # correlation at lags > 1, i.e. a decorrelation TIME rather than a one-step roughness,
        # (ii) any cross-channel quantity at all, and (iii) the antisymmetric and symmetric parts
        # of the mean velocity gradient -- feature 7 is |grad speed|, a scalar-field gradient,
        # which is neither.
        T = u.shape[1]
        a = u - mn.unsqueeze(1)                                    # (B,T,H,W,2) fluctuation
        v2 = sd[..., 0] * sd[..., 0] + EPS
        v2v = sd[..., 1] * sd[..., 1] + EPS

        # (1,2) lag-4 autocorrelation -- a DECORRELATION TIME, the per-pixel analogue of the
        # divergence measurement that put the forecaster 3.6x above the predictability limit.
        if T > 4:
            r4u = (a[:, :-4, ..., 0] * a[:, 4:, ..., 0]).mean(1) / v2
            r4v = (a[:, :-4, ..., 1] * a[:, 4:, ..., 1]).mean(1) / v2v
        else:
            r4u = torch.zeros_like(sd[..., 0]); r4v = torch.zeros_like(sd[..., 1])
        r4u = r4u.clamp(-1.0, 1.0); r4v = r4v.clamp(-1.0, 1.0)

        # (3) normalised Reynolds shear stress -- the momentum-transporting shear layer.  The
        # eleven planes hold no cross-channel quantity whatsoever.
        ruv = ((a[..., 0] * a[..., 1]).mean(1) / (sd[..., 0] * sd[..., 1] + EPS)).clamp(-1.0, 1.0)

        def _ddx(q):                                               # (B,H,W) central difference
            o = torch.zeros_like(q); o[:, :, 1:-1] = q[:, :, 2:] - q[:, :, :-2]; return o

        def _ddy(q):
            o = torch.zeros_like(q); o[:, 1:-1] = q[:, 2:] - q[:, :-2]; return o

        # (4,5) the two invariants of the MEAN velocity gradient: vorticity (antisymmetric) and
        # strain rate (symmetric).
        ub, vb = mn[..., 0], mn[..., 1]
        wz = _ddx(vb) - _ddy(ub)
        st = ((_ddx(ub) - _ddy(vb)).pow(2) + (_ddy(ub) + _ddx(vb)).pow(2)).clamp_min(0.0).sqrt()

        # (6) FLUCTUATING vorticity amplitude -- computed per frame, then reduced over time.
        uf, vf = u[..., 0], u[..., 1]
        wzt = (vf[:, :, :, 2:] - vf[:, :, :, :-2])
        wq = torch.zeros_like(uf)
        wq[:, :, :, 1:-1] = wzt
        wy = torch.zeros_like(uf)
        wy[:, :, 1:-1] = uf[:, :, 2:] - uf[:, :, :-2]
        wsd = (wq - wy).std(1, unbiased=False).clamp_min(0.0)

        # (7,8) roughness ratio rms(second difference)/rms(first difference): the fraction of the
        # window's temporal energy sitting at the highest resolved frequency.  Scale-free, so it
        # separates "large and smooth" from "small and jagged" where sd alone cannot.
        if T > 2:
            d2 = (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]).pow(2).mean(1).clamp_min(0.0).sqrt()
            rr = d2 / (d + EPS)
        else:
            rr = torch.zeros_like(sd)

        # (9) NON-STATIONARITY: the linear trend across the window, in units of its own sd.  A
        # window that is drifting is a window whose future is not its own past.
        tt = torch.linspace(-1.0, 1.0, T, device=u.device, dtype=u.dtype).reshape(1, T, 1, 1)
        spd = (u[..., 0].pow(2) + u[..., 1].pow(2)).clamp_min(0.0).sqrt()
        trend = (spd * tt).mean(1).abs() / (spd.std(1, unbiased=False) + EPS)

        f = torch.cat([f, torch.stack([r4u, r4v, ruv, L(wz.abs()), L(st), L(wsd),
                                       L(rr[..., 0]), L(rr[..., 1]), L(trend)], dim=1)], dim=1)
    elif feat != "v1":
        raise ValueError(f"unknown feature set {feat!r}")

    if int(n_coord) > 0:
        f = torch.cat([f, _coord_planes(f.shape[0], f.shape[2], f.shape[3], f.device, f.dtype)], 1)
    return f


class SBandHead(nn.Module):
    """(N_FEAT, H, W) -> (n_out, H, W) log residual-scale, one plane per (lead, channel).

    Per-lead output is not decoration.  Measured with the official aggregate_sps: an oracle |r|
    AVERAGED over the 20 leads -- the ceiling of any lead-shared head -- scores 56.70, while the
    same oracle kept per-lead but blurred 3x3 in space scores 58.17.  Lead resolution is worth
    +2.3 sps where fine spatial detail beyond 3x3 is worth +0.9.

    Dilations 1,2,4,2,1 span the 32x64 grid.  The final 1x1 is zero-initialised with a constant
    bias, so an UNTRAINED head emits a constant field: every element lands in one bin and the band
    degenerates to a single global interval (~50.8 sps) rather than to noise.
    """

    def __init__(self, n_feat=N_FEAT, width=64, n_layers=5, n_out=40, b0=-5.5):
        super().__init__()
        self.n_feat, self.width, self.n_layers, self.n_out = int(n_feat), int(width), int(n_layers), int(n_out)
        dl = [1, 2, 4, 2, 1]
        layers = []
        c = int(n_feat)
        for i in range(int(n_layers)):
            k = dl[i % len(dl)]
            layers += [nn.Conv2d(c, width, 3, padding=k, dilation=k, padding_mode="replicate"),
                       nn.GroupNorm(8, width), nn.GELU()]
            c = width
        self.body = nn.Sequential(*layers)
        self.out = nn.Conv2d(width, int(n_out), 1)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, float(b0))
        self.register_buffer("f_mu", torch.zeros(int(n_feat)))
        self.register_buffer("f_sd", torch.ones(int(n_feat)))

    def set_feature_norm(self, mu, sd):
        self.f_mu.copy_(mu.reshape(-1))
        self.f_sd.copy_(sd.reshape(-1).clamp_min(1e-6))

    def forward(self, feat):
        z = (feat - self.f_mu.reshape(1, -1, 1, 1)) / self.f_sd.reshape(1, -1, 1, 1)
        return self.out(self.body(z)).clamp(-12.0, 2.0)


def build_from_blob(blob):
    """Reconstruct a head from the dict written by t1_sband_fit.py, weights and all.

    Never rebuild from defaults: dropping width/n_layers/n_out would either raise (loud) or, for a
    matching-shape mistake, mis-forward silently -- the exact bug class `build_from_meta` was
    introduced to kill on the forecaster side.
    """
    h = SBandHead(int(blob["n_feat"]), int(blob["width"]), int(blob["n_layers"]),
                  int(blob["n_out"]))
    h.load_state_dict(blob["state_dict"])
    # n_coord travels WITH the head so the caller cannot mismatch the feature stack against the
    # first conv.  Absent (every pre-2026-08-08 blob, including sband_ship.pth) it is 0 and this
    # reconstructs the old head exactly.
    h.n_coord = int(blob.get("n_coord", 0))
    return h
