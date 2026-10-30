"""RealPDE Track 2 (LTTTA) submission -- v4's calibration, captured as a single CUDA graph.

WHAT CHANGED FROM v4, AND WHY.

v4's numerics are unchanged. This file changes only *how many times the host talks to the driver*.

The measurement first, because the fix follows from it. Profiling the shipped v4 bundle on a 4090
under the harness's own accounting (drain the queue between steps, never inside the timed region --
`local_eval` times `ttt_step` and only calls `.cpu()` afterwards) gives:

    full ttt_step (scored)              1.978 ms          time_score 95.05
      h2d, pageable (as shipped)        0.053 ms          2.7% -- NOT the bottleneck
      base forward, eager               0.910 ms          46%
      base forward, GPU-complete        0.895 ms          <- enqueue ~= completion
      residual (online/gamma/band)      1.015 ms          51%

`enqueue ~= completion` is the whole story. At batch 1 on a 32x64 grid, none of the ~90 kernels in
this step has enough arithmetic to cover its own launch, so the host spends ~1.9 ms *asking* for work
the device finishes almost immediately. That is not a compute problem and no amount of kernel tuning
touches it. The same profile priced the fix:

      base forward, cudnn.benchmark     0.887 ms          bit-identical, and worth nothing
      base forward, CUDA graph          0.014 ms          bit-identical, 65x

A graph replaces N launches with one. It is exactly the right tool here and exactly the wrong tool
for Track1 (whose bottleneck was a pageable D->H copy, where a graph would have done nothing) -- the
two tracks needed opposite fixes, which is why each got measured before either got edited.

Capturing only the forward would bank 0.9 ms of the 1.9. Capturing the *whole step* needs the
estimator tail too, and that tail is the interesting part, because v4 deliberately computed its EMA
debiasing on the host:

    k = self._k                                   # a python int
    wf = min(1.0, k / warm)                       # a python float
    cb = 1.0 / (1.0 - (1.0 - beta) ** k)          # a python float

That was the right call for v4 -- it kept `.item()` out of the timed region. But a graph replays
fixed kernels on fixed memory: a python scalar that changes per step is baked in at capture and is
silently wrong on every later replay. So every one of them moves onto the device:

    k      -> `_k_dev`, a 0-d tensor incremented in-graph by `_have` (0 on a trajectory's first step)
    wf     -> `(_k_dev / warm).clamp(max=1.0)`
    cb     -> a running power `_pw_*`, multiplied by `(1 - have*beta)` in-graph, so
              `cb = usable / (1 - _pw_*)` reproduces `1/(1-(1-beta)^k)` exactly at every k

and the two python branches go with them. `if prev_target is not None` becomes multiplication by
`_have`, which makes the EMA update a provable no-op on a first step:

    m.mul_(1 - have*beta).add_(r * (have*beta))       have=0 -> m *= 1, m += 0

and `if k > 0` becomes multiplication by `usable = _k_dev.clamp(max=1.0)`. Three of the four levers
already vanish at wf=0 by construction (`b * (scale*wf)`, `lam = gam_lam*wf`), so only the band's
lookup-vs-frozen choice needed an explicit blend:

    off = band_frozen + usable * (off_lookup - band_frozen)

which is bit-exact to v4's `elif` at usable=0.

One algebraic reformulation was necessary. v4 applies a gain as `pm + (p-pm)*c`; at c=1 that is `p`
only to within a rounding step. Written as `p + (p-pm)*(c-1)` it is *exactly* `p` when c=1, so a
first step where the online gamma is disabled is bit-identical to v4 rather than 1 ulp away. For
c != 1 the two forms differ by <= 1 ulp, which is why the gate below is a score comparison and not a
bit comparison.

WHAT IS GATED, AND ON WHAT. Two independent checks, each testing one thing, both run at load time
and both untimed:

  1. `graph replay == the same body run eagerly`, BIT-EXACT, over a multi-step synthetic stream that
     includes a reset. This is the only check that can catch an unsound capture (a buffer that is
     not actually static, a scalar left on the host). A graph is deterministic; anything short of
     bit-equality means the capture is wrong, and the graph is discarded rather than shipped.
  2. `this file's numerics vs v4's`, to a tolerance, reported by the offline verifier -- the 1-ulp
     question above, which is a numerical claim and not a correctness one.

If capture raises, if the replay check fails, if the device is not CUDA, or if `ttt_lr > 0` (the
gradient path mutates weights and cannot be captured), `self._graph` stays None and every call falls
through to the eager body, which is v4's. The degradation is to v4's score, not to a crash.

OUTPUT ALIASING -- the one real hazard. A graph writes its outputs into fixed memory, so returning
those tensors directly would hand the evaluator 378 aliases of one buffer if it defers denormalizing
to the end of the stream. `local_eval.py` denormalizes inside the loop, so this would happen to work;
"happens to work against the copy of the harness I can read" is not a property worth betting a
submission on. The three outputs are therefore packed into one contiguous (3, ...) buffer inside the
graph and cloned once outside it -- a single extra launch, and the returned views are independent.

Contract is unchanged:
    get_ttt_model(submission_dir, device) -> object with
        reset_ttt_state()                       called at the start of every trajectory, untimed
        ttt_step(input_norm, prev_target_norm)  -> (pred_norm, {"adapt_loss":..., "lower":..., "upper":...})
    Everything is in NORMALIZED space; the evaluator denormalizes the prediction AND the bounds.

v6 adds ONE option to v5: `gamf_*`, the online fluctuation gain as a per-pixel FIELD instead of the
single scalar v5 already ships.  It is off by default and every line it adds is guarded on
`gamf_beta > 0`, so a v6 bundle built with `gamf_beta = 0` is v5 bit-for-bit.

WHY A FIELD.  `Track1/U-net/t1_kegain.py` walked an adaptivity ladder over the gain in
`p -> mean_t(p) + g*(p - mean_t(p))`.  Because `scoring.kinetic_energy` subtracts the same temporal
mean, `KE(p') = g**2 * KE(p)` EXACTLY, so the ladder is arithmetic, not approximation.  It found the
scalar rung worth +0.021 FINAL and the POINTWISE rung worth +4.579 -- the whole `tke` deficit is a
per-pixel amplitude problem, and a global scale cannot reach it.  Reproduced here on Track 2's own
streaming protocol: the pointwise oracle is +3.452 S3 over the shipped scalar, the scalar oracle is
worth nothing.  v5 was therefore adapting the one degree of freedom with no headroom in it.

    ESTIMATOR.  Exactly the v5 scalar's construction, one level finer: instead of collapsing the
    variance ratio to a number it is kept per pixel, weighted into the scorer's units by ke_w,
    floored at a fraction of the window's own mean energy (so a freestream pixel where both
    variances are ~0 cannot manufacture a ratio out of noise), 3x3 smoothed, and clamped BEFORE it
    enters the EMA so no single bad window can poison the running field.

MEASURED, out-of-fold over the 5 AoA folds, paired per fold against the shipped scalar
(`local_benchmark/eval_kfold_t2_vgain.py`, round 2):

    smooth lam=1.0 beta=0.10   dS3 +0.112  SEM 0.034  t 3.35  5/5 folds   = +0.067 FINAL
    smooth lam=1.0 beta=0.05   dS3 +0.104                     4/5
    point  lam=1.0 beta=0.10   dS3 +0.098                     5/5
    scalar lam=0.75 (retune)   dS3 +0.051                     4/5

Cost is two variance reductions and one 3x3 pool per step.  `mvpe` is EXACTLY invariant under this
correction -- it is a time-averaged profile and the gain preserves `mean_t(p)` by construction --
and `rel_l2` pays only a second-order price, which is why a net gain has to exist at all.

v7 replaces the frozen `gamma(32,64,3)` prior with the LEARNED, sample-dependent variance-deficit
field of `vgain_head.py`.  Same operation, `p -> mean_t(p) + g*(p - mean_t(p))`; `g` is now
predicted per window by a 30k-parameter CNN from summary statistics of the input and the forecast.

This is a Track-1 method that Track 2 inherits for free: it needs no revealed target, and both
tracks share the forecaster (base 64, 1.90M params) and the scorer.  Measured out-of-fold on the
base-64 arm over 5 held-out AoA angles it beats the best static field by +1.8 to +2.2 on
(score_rel + score_tke), 5/5 folds; on Track 1's packaged bundle the same change was
**+0.617 FINAL, tke 77.26 -> 81.00**.  `mvpe` is EXACTLY invariant under it.

Absent `vgain.pth`, or absent `norm_mean`/`norm_std` in the calib, or on any exception while
loading, `self.head` stays None and this file behaves exactly as v6 -- the head can never take the
submission down.
"""
from __future__ import annotations
import os, sys, copy
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from unet_model import build_unet, build_from_meta

try:
    from ttt_model import TTTModel
except Exception:  # base class is optional; the evaluator duck-types this object
    TTTModel = nn.Module  # type: ignore

TTT_LR = __TTT_LR__
TTT_STEPS = __TTT_STEPS__      # inner SGD steps per revealed pair; 0 lr disables the branch entirely
HALFWIDTH = __HALFWIDTH__      # fallback constant normalized half-width, used iff calib.npz is unusable
ONLINE = __ONLINE__            # online-calibration rates; all-zero == the frozen v3 bundle

# The graph is on unless explicitly disabled. An env var rather than a build-time placeholder so the
# SAME archive can be A/B'd against its own eager path offline; the platform never sets it.
USE_GRAPH = os.environ.get("T2_NO_GRAPH", "") == ""

_DEF = {
    "bias_beta": 0.0, "bias_scale": 1.0, "bias_clip": 0.75,
    "sp_beta": 0.0, "sp_scale": 1.0, "sp_clip": 1.50,
    "gam_beta": 0.0, "gam_lam": 0.25, "gam_min": 0.85, "gam_max": 1.35,
    # v6: the same fluctuation gain, but as a FIELD g[y,x] instead of one scalar.  Off by default,
    # and at gamf_beta == 0 every line it adds is skipped, so v6 == v5 bit-for-bit.  gamf and gam
    # are mutually exclusive (the smoothed field's spatial mean already IS the scalar) -- asserted
    # in __init__ rather than left to produce a silently double-counted inflation.
    "gamf_beta": 0.0, "gamf_lam": 1.0, "gamf_warm": 1, "gamf_floor": 0.02, "gamf_smooth": 1,
    "band_beta": 0.0, "s_min": 0.60, "s_max": 2.40,
    "warm": 3,
}
import torch.nn.functional as _F


def _vg_features(p_ph, x_ph, eps=1e-12):
    """The 7 summary fields VGainHead consumes, from PHYSICAL (1,T,H,W,>=2) tensors.

    Bit-compatible with `vgain_head.stats_from_window` (asserted by `t2_vgain_verify.py`).  Static
    shapes and no data-dependent control flow, so it captures into the CUDA graph unchanged.
    """
    p2, i2 = p_ph[..., :2], x_ph[..., :2]
    pb, ib = p2.mean(1), i2.mean(1)
    pf = p2 - pb.unsqueeze(1)
    KEp = 0.5 * (pf * pf).mean(1).sum(-1)
    KEi = 0.5 * ((i2 - ib.unsqueeze(1)) ** 2).mean(1).sum(-1)
    lp, li = torch.log(KEp + eps), torch.log(KEi + eps)
    return torch.stack([lp, li, li - lp, pb[..., 0], pb[..., 1], ib[..., 0], ib[..., 1]], dim=1)


def _box3(g):
    """3x3 box smoothing of a (1,H,W) field, replicate-padded.  Static shapes only, so this is
    safe inside the CUDA graph.  Smoothing is what makes the pointwise estimator usable: measured
    out-of-fold, `smooth` beat `point` at every (lam, beta) tried, because a per-pixel ratio from
    one revealed window is noisy and averaging 9 neighbours costs almost no spatial resolution."""
    return _F.avg_pool2d(_F.pad(g.unsqueeze(1), (1, 1, 1, 1), mode="replicate"), 3, stride=1).squeeze(1)


_NF, _S0, _S1 = 101, 0.50, 2.50      # fine band-response lookup grid
# SGATE (2026-08-13): one-sided gate on the band adaptation. band_beta helps only where the static
# band UNDER-covers (unseen-regime streams: s_mean 2.478, 47.1% of entries at the s_max clamp,
# coverage 0.6925) and hurts where it does not (interpolation: s_mean 1.172, -0.391 sps). s is the
# discriminator, so engage adaptation only above _SGATE with a short ramp. _SGATE = 0 -> gate == 1
# everywhere -> bit-exact to the ungated template.
_SGATE, _SRAMP = 2.3, 0.3
_SSC = (_NF - 1) / (_S1 - _S0)

# BCD-T2 (2026-08-18): the interval is centred on the field its width table was fitted around.
# `t1_sband_cache` builds the band's training residual from a BARE model forward, but the runtime
# applies the vgain head first and then centres the band on the INFLATED field -- T1 measured
# +0.272 sps for removing that mismatch.  T2 has the identical topology (`p = self.base(...)`,
# head overwrites `p`, band centres on the overwritten `p`).  Measured on T2 before this patch:
# moving the centre to the pre-head field cuts mean|t-ctr|/sigma by 3.50% (T1's probe: -3.6%).
# 0.0 == ship (post-head centre, `_ctr is p`, a structural no-op).  1.0 == full pre-head centre.
_BCDT2 = 1.0
_BCDT2_LIVE = [False]
_T2FP16 = True              # T1 has done this since v33; T2 never did
_T2FP16_LIVE = [False]
_BLEND_LIVE = [False]        # XCB execution witness        # execution witness; a bare `except` cannot fake a list mutation


class _Half(torch.nn.Module):
    """fp16 the forecaster, fp32 everything else.  Casting at the boundary keeps the band, the
    vgain features and every online estimator bit-for-bit on the path they were fitted on."""

    def __init__(self, m):
        super().__init__()
        self.m = m.half()

    def forward(self, x):
        return self.m(x.half()).float()


class CalibratedTTTModel(TTTModel):
    def __init__(self, base: nn.Module, device: str, calib: Optional[dict] = None,
                 ttt_lr: float = TTT_LR, ttt_steps: int = TTT_STEPS,
                 halfwidth: float = HALFWIDTH, online: Optional[dict] = None,
                 submission_dir: str = ""):
        super().__init__()
        self.base = base.to(device)
        if _T2FP16 and str(device).startswith('cuda'):
            try:
                self.base = _Half(self.base).to(device).eval()
                _T2FP16_LIVE[0] = True
            except Exception:
                pass
        self.device = device
        self.ttt_lr = float(ttt_lr)
        self.ttt_steps = int(ttt_steps)
        self.hw = float(halfwidth)
        self._init_state = copy.deepcopy(self.base.state_dict()) if self.ttt_lr > 0 else None
        self._opt = torch.optim.SGD(self.base.parameters(), lr=self.ttt_lr) if self.ttt_lr > 0 else None
        self._prev_input: Optional[torch.Tensor] = None

        o = dict(_DEF)
        o.update(online or {})
        self.o = o

        # ---- frozen priors: gamma (H,W,C) and band offsets (n_t,C), all normalized space ----
        self.gamma = self.band_lo = self.band_hi = None
        self.LOf = self.HIf = self.mad_ref = self.ch_idx = None
        self.cm = 0
        # ---- per-element band shape (v21).  Absent -> every path below is bit-identical to v7 ----
        self._pe_edge = self._pe_rl = self._pe_rh = None
        self._pe_nt = self._pe_c2 = 0
        self._PE_STATUS = "absent"
        self._pc_rl = self._pc_rh = None
        self._PC_STATUS = "absent"
        self._pe_pooled = False          # v30: causal pooled index, set by calib key pe_pooled
        self._pe_acc = None
        self._pe_cnt = None
        if calib is not None:
            g = calib.get("gamma")
            if g is not None:
                g = np.asarray(g, np.float32)
                self.gamma = torch.as_tensor(g).to(device).reshape(1, 1, *g.shape)
            lo, hi = calib.get("band_lo"), calib.get("band_hi")
            if lo is not None and hi is not None:
                lo = np.asarray(lo, np.float32); hi = np.asarray(hi, np.float32)
                lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
                self.band_lo = torch.as_tensor(lo).to(device).reshape(1, lo.shape[0], 1, 1, lo.shape[1])
                self.band_hi = torch.as_tensor(hi).to(device).reshape(1, hi.shape[0], 1, 1, hi.shape[1])
            kw = calib.get("ke_w")
            if kw is not None:
                self._ke_w_np = np.asarray(kw, np.float32).ravel()
            nm, ns_ = calib.get("norm_mean"), calib.get("norm_std")
            if nm is not None and ns_ is not None:
                self._nmean_np = np.asarray(nm, np.float32).ravel()
                self._nstd_np = np.asarray(ns_, np.float32).ravel()
            self._build_online_tables(calib, device)
            try:
                _e, _rl, _rh = calib.get("pe_edge"), calib.get("pe_rl"), calib.get("pe_rh")
                if _e is not None and _rl is not None and _rh is not None:
                    _e = np.asarray(_e, np.float32)
                    _rl = np.asarray(_rl, np.float32)
                    _rh = np.asarray(_rh, np.float32)
                    _nt, _c2, _qm1 = _e.shape
                    assert _rl.shape == _rh.shape == (_nt, _c2, _qm1 + 1)
                    assert np.all(np.diff(_e, axis=-1) >= 0.0), "pe_edge not sorted"
                    assert float(_rl.min()) > 0.0 and float(_rh.min()) > 0.0, "ratios must be > 0"
                    self._pe_edge = torch.as_tensor(_e).to(device).reshape(_nt * _c2, _qm1).contiguous()
                    self._pe_rl = torch.as_tensor(_rl).to(device).reshape(_nt * _c2, _qm1 + 1).contiguous()
                    self._pe_rh = torch.as_tensor(_rh).to(device).reshape(_nt * _c2, _qm1 + 1).contiguous()
                    self._pe_nt, self._pe_c2 = int(_nt), int(_c2)
                    self._PE_STATUS = "ok"
                    if bool(np.asarray(calib.get("pe_pooled", False)).ravel()[0]):
                        self._pe_pooled = True
                        self._pe_acc = torch.zeros(32, 64, _c2, device=device)
                        self._pe_cnt = torch.zeros((), device=device)
                        self._PE_STATUS = "ok-pooled"
                _cl, _ch = calib.get("pc_rl"), calib.get("pc_rh")
                if _cl is not None and _ch is not None:
                    _cl = np.asarray(_cl, np.float32); _ch = np.asarray(_ch, np.float32)
                    assert _cl.shape == _ch.shape and _cl.ndim == 4
                    assert float(_cl.min()) > 0.0 and float(_ch.min()) > 0.0
                    # (LT,C2,H,W) -> (1,LT,H,W,C2), the layout the band offsets use
                    self._pc_rl = torch.as_tensor(_cl).permute(0,2,3,1).unsqueeze(0).contiguous().to(device)
                    self._pc_rh = torch.as_tensor(_ch).permute(0,2,3,1).unsqueeze(0).contiguous().to(device)
                    self._PC_STATUS = "ok"
            except Exception as _pee:
                self._pe_edge = self._pe_rl = self._pe_rh = None
                self._pe_nt = self._pe_c2 = 0
                self._PE_STATUS = "fail:" + type(_pee).__name__

        self._n_t = int(self.band_lo.shape[1]) if self.band_lo is not None else 20
        self._C = int(self.band_lo.shape[-1]) if self.band_lo is not None else 3
        # `ke_w[c] = std_t[c]**2`.  scoring.kinetic_energy is computed on PHYSICAL fields, but
        # everything in here is normalised, and var_phys[c] = std_t[c]**2 * var_norm[c].  For real
        # data std = [0.0968, 0.0160], so u carries 36x the physical energy of v: summing normalised
        # channel variances (what the v5 SCALAR does) is a different quantity from the one the
        # scorer measures.  Absent ke_w this falls back to equal weights, i.e. v5's convention.
        self._ke_w = None
        assert not (self.o["gamf_beta"] > 0.0 and self.o["gam_beta"] > 0.0), \
            "gam_beta and gamf_beta are mutually exclusive: the field subsumes the scalar"

        # v7: the LEARNED variance-deficit field.  Same operation as the frozen `gamma` -- it is
        # the sample-dependent version of it and REPLACES it, exactly as it does in Track 1's
        # submission_t1_v6.py, where it measured +0.617 FINAL on the packaged bundle (tke
        # 77.26 -> 81.00).  Track 1 and Track 2 share the forecaster architecture and the scorer,
        # so the head transfers with no Track-2-specific retraining; it needs no revealed target,
        # which is why it is a Track-1 method Track 2 simply inherits.
        # Requires calib to carry norm_mean/norm_std: the head consumes PHYSICAL summary fields and
        # everything in this file is normalised.  Absent either, the head stays off and the frozen
        # gamma is used, so a missing key degrades to v6 rather than to garbage.
        self.head = None
        self._nmean = self._nstd = None
        try:
            cands = [os.path.join(d, "vgain.pth") for d in (submission_dir, _HERE) if d]
            hp = next((c for c in cands if os.path.exists(c)), "")
            if hp and hasattr(self, "_nmean_np"):
                from vgain_head import VGainHead, N_FEAT as _NFEAT
                hd = torch.load(hp, map_location="cpu", weights_only=False)
                h = VGainHead(int(hd.get("n_feat", _NFEAT)), int(hd["width"]),
                              float(hd["gmin"]), float(hd["gmax"]))
                h.load_state_dict(hd["state_dict"])
                self.head = h.to(device).eval()
                for q in self.head.parameters():
                    q.requires_grad_(False)
                r = lambda a_: torch.as_tensor(a_.reshape(1, 1, 1, 1, -1).copy()).to(device)
                self._nmean, self._nstd = r(self._nmean_np), r(self._nstd_np)
        except Exception:
            self.head = None
        self._H, self._W = 32, 64
        if self.gamma is not None and self.gamma.dim() == 5:
            self._H, self._W = int(self.gamma.shape[2]), int(self.gamma.shape[3])
        km = self.cm if self.cm else 2
        w = getattr(self, "_ke_w_np", None)
        w = np.ones(km, np.float32) if w is None or w.size < km else w[:km].astype(np.float32)
        self._ke_w = torch.as_tensor(w).to(device).reshape(1, 1, 1, km)
        self._alloc()
        self._graph = None
        self._have_flag = None
        self.reset_ttt_state()
        if USE_GRAPH and self.ttt_lr == 0.0 and str(device).startswith("cuda") \
                and torch.cuda.is_available():
            try:
                self._capture()
            except Exception:
                self._graph = None
            finally:
                self.reset_ttt_state()

    def _pe_shape(self, xn, C):
        """NORMALISED (1,T,H,W,>=c2) -> (rl, rh), each (1, n_t, H, W, C) and 1.0 on any channel the
        table does not cover.  Width-neutral by construction: the ratios average 1 over the
        equal-mass bins, so this redistributes width without changing how much there is.

        Deterministic (searchsorted + gather), which is required -- `_capture` refuses to ship a
        graph whose replay is not bit-identical to the eager body."""
        if self._pe_edge is None:
            return None, None
        nt, c2 = self._pe_nt, self._pe_c2
        if xn.shape[-1] < c2 or C < c2:
            return None, None
        s = xn[0, :, :, :, :c2].std(dim=0, unbiased=False)          # (H, W, c2)
        if self._pe_pooled and self._pe_acc is not None:
            # causal pool over the trajectory so far: acc/cnt live on-device, updated every step,
            # reset at trajectory boundaries.  One mul/add per pixel -- capturable, like the EMAs.
            self._pe_acc.add_(s * s)
            self._pe_cnt.add_(1.0)
            s = torch.sqrt(self._pe_acc / self._pe_cnt.clamp_min(1.0))
        H_, W_ = int(s.shape[0]), int(s.shape[1])
        q = s.permute(2, 0, 1).reshape(1, c2, H_ * W_).expand(nt, c2, H_ * W_) \
             .reshape(nt * c2, H_ * W_).contiguous()                # row t*c2+c -> channel c
        b = torch.searchsorted(self._pe_edge, q, right=True)
        rl = torch.gather(self._pe_rl, 1, b).reshape(nt, c2, H_, W_).permute(0, 2, 3, 1).unsqueeze(0)
        rh = torch.gather(self._pe_rh, 1, b).reshape(nt, c2, H_, W_).permute(0, 2, 3, 1).unsqueeze(0)
        if self._pc_rl is not None and self._pc_rl.shape[1] == nt \
                and tuple(self._pc_rl.shape[2:4]) == (H_, W_) and self._pc_rl.shape[-1] == c2:
            rl = rl * self._pc_rl
            rh = rh * self._pc_rh
        if C > c2:
            rl = torch.nn.functional.pad(rl, (0, C - c2), value=1.0)
            rh = torch.nn.functional.pad(rh, (0, C - c2), value=1.0)
        return rl, rh

    # ------------------------------------------------------------------ setup, all outside timing
    def _build_online_tables(self, calib: dict, device: str) -> None:
        """Densify the (n_s, n_t, C) band-response curve onto a uniform grid so the per-step lookup
        is a single gather instead of a searchsorted + interpolate. Any missing / mismatched key
        disables ONLY the band lever."""
        try:
            sg = np.asarray(calib["s_grid"], np.float64).reshape(-1)
            los = np.asarray(calib["band_lo_s"], np.float32)
            his = np.asarray(calib["band_hi_s"], np.float32)
            mad = np.asarray(calib["mad_val"], np.float32)
            cidx = np.asarray(calib["ch_idx"], np.int64).reshape(-1)
            n_t, C = los.shape[1], los.shape[2]
            assert his.shape == los.shape and los.shape[0] == sg.size
            assert mad.ndim == 2 and mad.shape[0] == n_t and cidx.size == C
            fine = np.linspace(_S0, _S1, _NF)
            LOf = np.empty((_NF, n_t, C), np.float32)
            HIf = np.empty((_NF, n_t, C), np.float32)
            for t in range(n_t):
                for ch in range(C):
                    LOf[:, t, ch] = np.interp(fine, sg, los[:, t, ch])
                    HIf[:, t, ch] = np.interp(fine, sg, his[:, t, ch])
            LOf, HIf = np.minimum(LOf, HIf), np.maximum(LOf, HIf)
            assert np.isfinite(LOf).all() and np.isfinite(HIf).all() and np.all(mad > 0)
            self.LOf = torch.as_tensor(LOf).to(device)
            self.HIf = torch.as_tensor(HIf).to(device)
            self.mad_ref = torch.as_tensor(mad).to(device)          # (n_t, cm)
            self.ch_idx = torch.as_tensor(cidx).to(device)          # (C,)
            self.cm = int(mad.shape[1])
        except Exception:
            self.LOf = self.HIf = self.mad_ref = self.ch_idx = None
            self.cm = 0

    def _alloc(self) -> None:
        """Pre-allocate every online buffer once so no step ever allocates, and hoist every constant
        that v4 recomputed per step (the clip limits are `mad_ref` times a fixed scalar, gathered by
        a fixed index -- six launches a step for a value that never changes)."""
        d, n_t, C, cm = self.device, self._n_t, self._C, self.cm
        H, W = self._H, self._W
        z = lambda *s: torch.zeros(*s, dtype=torch.float32, device=d)
        self._bias_m = z(n_t, C)                    # channels >= cm stay identically zero
        self._sp_m = z(1, n_t, H, W, C) if self.o["sp_beta"] > 0 else None
        self._gam_m = z(())
        self._gamf_m = z(1, 1, H, W, 1) if self.o["gamf_beta"] > 0 else None
        self._mad_m = z(n_t, max(cm, 1))
        self._chmask = None
        if cm and cm < C:
            m = np.zeros((1, 1, 1, 1, C), np.float32); m[..., :cm] = 1.0
            self._chmask = torch.as_tensor(m).to(d)
        self._lim_bias = self._lim_sp = None
        if self.mad_ref is not None and self.ch_idx is not None:
            self._lim_bias = (self.o["bias_clip"] * self.mad_ref)[:, self.ch_idx].contiguous()
            self._lim_sp = (self.o["sp_clip"] * self.mad_ref)[:, self.ch_idx] \
                .reshape(1, n_t, 1, 1, C).contiguous()

        # ---- graph-only state: every host scalar of v4, moved onto the device ----
        self._k_dev = z(())          # step counter, incremented in-graph by _have
        self._have = z(())           # 1.0 iff prev_target_norm was supplied this step
        self._pw_bias = z(())        # running (1-beta)^k, one per estimator
        self._pw_sp = z(())
        self._pw_gam = z(())
        self._pw_gamf = z(())
        self._pw_band = z(())
        self._sx = z(1, n_t, H, W, C)          # static graph input
        self._sy = z(1, n_t, H, W, C)          # static graph prev-target
        self._sg_pg = z(1, n_t, H, W, C)       # prediction after the FROZEN gamma, previous step
        self._sg_pfin = z(1, n_t, H, W, C)     # what we actually returned, previous step
        self._out3 = z(3, 1, n_t, H, W, C)     # pred / lower / upper, packed for a single clone
        self._pin_x = torch.empty(1, n_t, H, W, C, dtype=torch.float32,
                                  pin_memory=str(self.device).startswith("cuda"))
        self._pin_y = torch.empty_like(self._pin_x, pin_memory=str(self.device).startswith("cuda"))

    def reset_ttt_state(self) -> None:
        if getattr(self, '_pe_acc', None) is not None:
            self._pe_acc.zero_(); self._pe_cnt.zero_()
        if self._init_state is not None:
            self.base.load_state_dict(copy.deepcopy(self._init_state))
            self._opt = torch.optim.SGD(self.base.parameters(), lr=self.ttt_lr)
        self._prev_input = None
        self._prev_pg = None            # prediction after the FROZEN gamma, before any online term
        self._prev_pfin = None          # what we actually returned
        self._k = 0
        for b in (self._bias_m, self._sp_m, self._gam_m, self._gamf_m, self._mad_m):
            if b is not None:
                b.zero_()
        # graph state: the running powers start at (1-beta)^0 = 1
        for b in (self._k_dev, self._have, self._sy, self._sg_pg, self._sg_pfin):
            b.zero_()
        for b in (self._pw_bias, self._pw_sp, self._pw_gam, self._pw_gamf, self._pw_band):
            b.fill_(1.0)
        self._have_flag = False

    # ------------------------------------------------------------------ the capturable body
    def _body(self) -> None:
        """Exactly what the timed region computes, using ONLY device-resident scalars.

        Reads  self._sx (input), self._sy (prev target), self._have (0/1) and the EMA state.
        Writes self._out3 and the EMA state. No python branch depends on anything that changes
        between steps, which is what makes it capturable."""
        o = self.o
        have = self._have
        with torch.no_grad():
            # ---------- 1. gradient-free online update from the revealed previous pair ----------
            #  `have` is 0 on a trajectory's first step, which turns every update below into a
            #  provable no-op: m *= (1-0), m += r*0.
            r = self._sy - self._sg_pg
            if o["bias_beta"] > 0.0:
                a = have * o["bias_beta"]
                rm = r.mean(dim=(0, 2, 3))            # (n_t, C)
                if self._chmask is not None:
                    rm = rm * self._chmask.reshape(1, -1)
                self._bias_m.mul_(1.0 - a).add_(rm * a)
                self._pw_bias.mul_(1.0 - a)
            if self._sp_m is not None:
                a = have * o["sp_beta"]
                rs = r - r.mean(dim=(0, 2, 3), keepdim=True)   # orthogonal to the bias term
                if self._chmask is not None:
                    rs = rs * self._chmask
                self._sp_m.mul_(1.0 - a).add_(rs * a)
                self._pw_sp.mul_(1.0 - a)
            if o["gam_beta"] > 0.0:
                a = have * o["gam_beta"]
                k = self.cm if self.cm else 2
                vt = self._sy[..., :k].var(dim=1, unbiased=False).sum()
                vp = self._sg_pg[..., :k].var(dim=1, unbiased=False).sum()
                cn = torch.sqrt(vt / vp.clamp_min(1e-12))
                # on a first step _sg_pg is all zeros, so vp==0 and cn is huge-but-finite; it is
                # multiplied by a==0 below. nan_to_num makes that argument unnecessary.
                cn = torch.nan_to_num(cn, nan=1.0, posinf=1.0, neginf=1.0)
                self._gam_m.mul_(1.0 - a).add_(cn * a)
                self._pw_gam.mul_(1.0 - a)
            if self._gamf_m is not None:
                # v6 FIELD estimator. Measured out-of-fold on the 5 AoA folds against the v5 scalar
                # it replaces: dS3 +0.112, SEM 0.034, t 3.35, 5/5 folds (= +0.067 FINAL) at
                # gamf_lam 1.0 / gamf_beta 0.1. See Track2/U-net/README.md.
                a = have * o["gamf_beta"]
                k = self.cm if self.cm else 2
                vt = (self._sy[..., :k].var(dim=1, unbiased=False) * self._ke_w).sum(-1)
                vp = (self._sg_pg[..., :k].var(dim=1, unbiased=False) * self._ke_w).sum(-1)
                fl = o["gamf_floor"] * vp.mean(dim=(1, 2), keepdim=True)
                rf = (vt.clamp_min(fl) / vp.clamp_min(fl).clamp_min(1e-20)).sqrt()
                if o["gamf_smooth"]:
                    rf = _box3(rf)
                # on a first step _sg_pg is zero, so vp==0 and rf is huge; it is clamped here and
                # multiplied by a==0 below, so neither the clamp nor nan_to_num is load-bearing --
                # they are there so a degenerate window later in a trajectory also cannot escape.
                rf = torch.nan_to_num(rf, nan=1.0, posinf=1.0, neginf=1.0) \
                    .clamp(o["gam_min"], o["gam_max"])
                self._gamf_m.mul_(1.0 - a).add_(rf.reshape(1, 1, *rf.shape[1:], 1) * a)
                self._pw_gamf.mul_(1.0 - a)
            if o["band_beta"] > 0.0 and self.mad_ref is not None:
                a = have * o["band_beta"]
                k = self.cm
                md = (self._sy[..., :k] - self._sg_pfin[..., :k]).abs().mean(dim=(0, 2, 3))
                self._mad_m.mul_(1.0 - a).add_(md * a)
                self._pw_band.mul_(1.0 - a)
            self._k_dev.add_(have)

            # ---------- 2. forward ----------
            p = self.base(self._sx)
            _pre = p

            # ---------- 3. the gain field: LEARNED if present, else the frozen prior ----------
            # `g` multiplies the fluctuation, which scales linearly under de-normalisation, so
            # applying it here in normalised space is identical to applying it to the physical
            # field -- but the FEATURES must be physical.
            if self.head is not None:
                gh = self.head(_vg_features(p * self._nstd + self._nmean,
                                            self._sx * self._nstd + self._nmean))
                pm = p.mean(dim=1, keepdim=True)
                p = pm + (p - pm) * gh[:, None, :, :, None]
            elif self.gamma is not None and tuple(p.shape[2:]) == tuple(self.gamma.shape[2:]):
                pm = p.mean(dim=1, keepdim=True)
                p = pm + (p - pm) * self.gamma
            self._sg_pg.copy_(p)         # estimator reference: frozen pipeline, no online terms

            # ---------- 4. online corrections ----------
            #  v4 computed `wf` and `cb` on the host from a python int. Both are device tensors here;
            #  `cb = usable/(1-(1-beta)^k)` is reproduced exactly by the running power `_pw_*`.
            usable = self._k_dev.clamp(max=1.0)
            wf = (self._k_dev / float(max(1, int(o["warm"])))).clamp(max=1.0)
            n_t, C = p.shape[1], p.shape[-1]

            if o["gam_beta"] > 0.0:
                cb = usable / (1.0 - self._pw_gam).clamp_min(1e-12)
                lam = o["gam_lam"] * wf
                c_used = (1.0 + lam * (self._gam_m * cb - 1.0)).clamp(o["gam_min"], o["gam_max"])
                pm = p.mean(dim=1, keepdim=True)
                # `p + (p-pm)*(c-1)` rather than v4's `pm + (p-pm)*c`: identical algebraically, but
                # this form is EXACTLY p at c==1, so a first step is bit-identical to v4's skip.
                p = p + (p - pm) * (c_used - 1.0)

            if self._gamf_m is not None:
                cb = usable / (1.0 - self._pw_gamf).clamp_min(1e-12)
                wff = (self._k_dev / float(max(1, int(o["gamf_warm"])))).clamp(max=1.0)
                cf = (1.0 + o["gamf_lam"] * wff * (self._gamf_m * cb - 1.0)) \
                    .clamp(o["gam_min"], o["gam_max"])
                pm = p.mean(dim=1, keepdim=True)
                p = p + (p - pm) * (cf - 1.0)

            if o["bias_beta"] > 0.0 and self._lim_bias is not None:
                cb = usable / (1.0 - self._pw_bias).clamp_min(1e-12)
                b = (self._bias_m * cb).clamp(-self._lim_bias, self._lim_bias) \
                    * (o["bias_scale"] * wf)
                p = p + b.reshape(1, n_t, 1, 1, C)

            if self._sp_m is not None and self._lim_sp is not None:
                cb = usable / (1.0 - self._pw_sp).clamp_min(1e-12)
                p = p + (self._sp_m * cb).clamp(-self._lim_sp, self._lim_sp) \
                    * (o["sp_scale"] * wf)

            # ---------- 5. interval ----------
            _ctr = p if _BCDT2 == 0.0 else (p + _BCDT2 * (_pre - p))
            if _BCDT2 != 0.0:
                _BCDT2_LIVE[0] = True
            if (o["band_beta"] > 0.0 and self.LOf is not None and self.band_lo is not None
                    and p.shape[1] == self.LOf.shape[1] and p.shape[-1] == self.LOf.shape[-1]):
                cb = usable / (1.0 - self._pw_band).clamp_min(1e-12)
                s = ((self._mad_m * cb) / self.mad_ref)[:, self.ch_idx]      # (n_t, C)
                s = (1.0 + wf * (s - 1.0)).clamp(o["s_min"], o["s_max"])
                j = ((s - _S0) * _SSC).round().long().clamp_(0, _NF - 1)
                gl = torch.gather(self.LOf, 0, j.unsqueeze(0)).reshape(1, n_t, 1, 1, C)
                gh = torch.gather(self.HIf, 0, j.unsqueeze(0)).reshape(1, n_t, 1, 1, C)
                # v4 took the FROZEN band whenever k==0; blend instead of branch. At usable==0 this
                # is bit-exact to `p + self.band_lo`.
                _w = usable
                if _SGATE > 0.0:
                    _w = _w * ((s - _SGATE) / _SRAMP).clamp(0.0, 1.0) \
                                .reshape(1, n_t, 1, 1, C)
                _ol = self.band_lo + _w * (gl - self.band_lo)
                _oh = self.band_hi + _w * (gh - self.band_hi)
                _rl, _rh = self._pe_shape(self._sx, C)
                if _rl is not None and _rl.shape[1] == n_t:
                    _ol, _oh = _ol * _rl, _oh * _rh
                lo, hi = _ctr + _ol, _ctr + _oh
            elif (self.band_lo is not None and p.shape[1] == self.band_lo.shape[1]
                    and p.shape[-1] == self.band_lo.shape[-1]):
                _ol, _oh = self.band_lo, self.band_hi
                _rl, _rh = self._pe_shape(self._sx, C)
                if _rl is not None and _rl.shape[1] == n_t:
                    _ol, _oh = _ol * _rl, _oh * _rh
                lo, hi = _ctr + _ol, _ctr + _oh
            else:
                lo, hi = _ctr - self.hw, _ctr + self.hw

            # ---------- 6. hard safety: local_eval checks NONE of this, scoring.py zeroes on it ----------
            p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
            lo = torch.nan_to_num(lo, nan=-1e3, posinf=-1e3, neginf=-1e3)
            hi = torch.nan_to_num(hi, nan=1e3, posinf=1e3, neginf=1e3)
            self._sg_pfin.copy_(p)
            self._out3[0].copy_(p)
            self._out3[1].copy_(torch.minimum(lo, hi))
            self._out3[2].copy_(torch.maximum(lo, hi))

    # ------------------------------------------------------------------ capture + its gate
    def _capture(self) -> None:
        """Warm up on a side stream, capture, then prove the replay is bit-identical to the same
        body run eagerly across a synthetic stream that includes a first step (have=0), several
        adapting steps, and a reset. A graph is deterministic, so anything short of bit-equality
        means a buffer that should be static is not -- discard rather than ship."""
        g0 = torch.Generator(device="cpu").manual_seed(20260726)
        shape = tuple(self._sx.shape)
        xs = [torch.randn(shape, generator=g0, dtype=torch.float32) for _ in range(6)]
        ys = [torch.randn(shape, generator=g0, dtype=torch.float32) * 0.9 for _ in range(6)]

        def drive(run_one):
            """One synthetic trajectory: first step has no target, the rest do."""
            self.reset_ttt_state()
            out = []
            for i in range(6):
                self._sx.copy_(xs[i])
                if i == 0:
                    self._have.zero_()
                else:
                    self._sy.copy_(ys[i])
                    self._have.fill_(1.0)
                run_one()
                out.append(self._out3.clone())
            return out

        self.base.eval()
        eager = drive(lambda: self._body())

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            self._body()
        torch.cuda.synchronize()

        replay = drive(gr.replay)
        torch.cuda.synchronize()
        ok = len(eager) == len(replay) and all(
            torch.equal(a, b) for a, b in zip(eager, replay))
        self._graph = gr if ok else None
        self._graph_checked = bool(ok)

    # ------------------------------------------------------------------ the timed region
    def _graph_step(self, input_norm: torch.Tensor, prev_target_norm: Optional[torch.Tensor]
                    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        self._pin_x.copy_(torch.as_tensor(input_norm))
        self._sx.copy_(self._pin_x, non_blocking=True)
        hp = prev_target_norm is not None
        if hp:
            self._pin_y.copy_(torch.as_tensor(prev_target_norm))
            self._sy.copy_(self._pin_y, non_blocking=True)
        if hp != self._have_flag:
            self._have.fill_(1.0 if hp else 0.0)
            self._have_flag = hp
        self._graph.replay()
        # One clone, not three: the graph packs its outputs contiguously so the aliasing hazard
        # costs a single launch. The views below are independent of the graph's static memory.
        c = self._out3.clone()
        return c[0], {"adapt_loss": None, "lower": c[1], "upper": c[2]}

    def ttt_step(self, input_norm: torch.Tensor, prev_target_norm: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self._graph is not None:
            return self._graph_step(input_norm, prev_target_norm)

        # ---- eager fallback: v4's body verbatim, reached iff capture was impossible or unsound ----
        o = self.o
        x = torch.as_tensor(input_norm).to(self.device, non_blocking=True)

        if self._opt is not None and prev_target_norm is not None and self._prev_input is not None:
            y0 = torch.as_tensor(prev_target_norm).to(self.device, non_blocking=True)
            self.base.train()
            for _ in range(self.ttt_steps):
                self._opt.zero_grad()
                loss = torch.mean((self.base(self._prev_input) - y0) ** 2)
                loss.backward()
                self._opt.step()
        self.base.eval()

        with torch.no_grad():
            if prev_target_norm is not None and self._prev_pg is not None:
                y = torch.as_tensor(prev_target_norm).to(self.device, non_blocking=True)
                if y.shape == self._prev_pg.shape:
                    self._k += 1
                    r = y - self._prev_pg
                    if o["bias_beta"] > 0.0:
                        rm = r.mean(dim=(0, 2, 3))
                        if self._chmask is not None:
                            rm = rm * self._chmask.reshape(1, -1)
                        self._bias_m.mul_(1.0 - o["bias_beta"]).add_(rm, alpha=o["bias_beta"])
                    if self._sp_m is not None:
                        rs = r - r.mean(dim=(0, 2, 3), keepdim=True)
                        if self._chmask is not None:
                            rs = rs * self._chmask
                        self._sp_m.mul_(1.0 - o["sp_beta"]).add_(rs, alpha=o["sp_beta"])
                    if o["gam_beta"] > 0.0:
                        k = self.cm if self.cm else 2
                        vt = y[..., :k].var(dim=1, unbiased=False).sum()
                        vp = self._prev_pg[..., :k].var(dim=1, unbiased=False).sum()
                        cn = torch.sqrt(vt / vp.clamp_min(1e-12))
                        self._gam_m.mul_(1.0 - o["gam_beta"]).add_(cn, alpha=o["gam_beta"])
                    if self._gamf_m is not None:
                        k = self.cm if self.cm else 2
                        vt = (y[..., :k].var(dim=1, unbiased=False) * self._ke_w).sum(-1)
                        vp = (self._prev_pg[..., :k].var(dim=1, unbiased=False)
                              * self._ke_w).sum(-1)
                        fl = o["gamf_floor"] * vp.mean(dim=(1, 2), keepdim=True)
                        rf = (vt.clamp_min(fl) / vp.clamp_min(fl).clamp_min(1e-20)).sqrt()
                        if o["gamf_smooth"]:
                            rf = _box3(rf)
                        rf = torch.nan_to_num(rf, nan=1.0, posinf=1.0, neginf=1.0) \
                            .clamp(o["gam_min"], o["gam_max"])
                        self._gamf_m.mul_(1.0 - o["gamf_beta"]) \
                            .add_(rf.reshape(1, 1, *rf.shape[1:], 1), alpha=o["gamf_beta"])
                    if o["band_beta"] > 0.0 and self.mad_ref is not None and self._prev_pfin is not None:
                        k = self.cm
                        md = (y[..., :k] - self._prev_pfin[..., :k]).abs().mean(dim=(0, 2, 3))
                        self._mad_m.mul_(1.0 - o["band_beta"]).add_(md, alpha=o["band_beta"])

            p = self.base(x)
            _pre = p

            if self.head is not None:
                gh = self.head(_vg_features(p * self._nstd + self._nmean,
                                            x * self._nstd + self._nmean))
                pm = p.mean(dim=1, keepdim=True)
                p = pm + (p - pm) * gh[:, None, :, :, None]
            elif self.gamma is not None and tuple(p.shape[2:]) == tuple(self.gamma.shape[2:]):
                pm = p.mean(dim=1, keepdim=True)
                p = pm + (p - pm) * self.gamma
            pg = p

            k = self._k
            usable = k > 0
            wf = min(1.0, k / float(max(1, int(o["warm"])))) if usable else 0.0
            n_t, C = p.shape[1], p.shape[-1]

            if usable and o["gam_beta"] > 0.0:
                cb = 1.0 / (1.0 - (1.0 - o["gam_beta"]) ** k)
                lam = o["gam_lam"] * wf
                c_used = (1.0 + lam * (self._gam_m * cb - 1.0)).clamp(o["gam_min"], o["gam_max"])
                pm = p.mean(dim=1, keepdim=True)
                p = pm + (p - pm) * c_used

            if usable and self._gamf_m is not None:
                cb = 1.0 / (1.0 - (1.0 - o["gamf_beta"]) ** k)
                wff = min(1.0, k / float(max(1, int(o["gamf_warm"]))))
                cf = (1.0 + o["gamf_lam"] * wff * (self._gamf_m * cb - 1.0)) \
                    .clamp(o["gam_min"], o["gam_max"])
                pm = p.mean(dim=1, keepdim=True)
                p = pm + (p - pm) * cf

            if usable and o["bias_beta"] > 0.0 and self._lim_bias is not None:
                cb = 1.0 / (1.0 - (1.0 - o["bias_beta"]) ** k)
                b = (self._bias_m * cb).clamp(-self._lim_bias, self._lim_bias) \
                    * (o["bias_scale"] * wf)
                p = p + b.reshape(1, n_t, 1, 1, C)

            if usable and self._sp_m is not None and self._lim_sp is not None:
                cb = 1.0 / (1.0 - (1.0 - o["sp_beta"]) ** k)
                p = p + (self._sp_m * cb).clamp(-self._lim_sp, self._lim_sp) \
                    * (o["sp_scale"] * wf)

            _ctr = p if _BCDT2 == 0.0 else (p + _BCDT2 * (_pre - p))
            if _BCDT2 != 0.0:
                _BCDT2_LIVE[0] = True
            if (usable and o["band_beta"] > 0.0 and self.LOf is not None
                    and p.shape[1] == self.LOf.shape[1] and p.shape[-1] == self.LOf.shape[-1]):
                cb = 1.0 / (1.0 - (1.0 - o["band_beta"]) ** k)
                s = ((self._mad_m * cb) / self.mad_ref)[:, self.ch_idx]
                s = (1.0 + wf * (s - 1.0)).clamp(o["s_min"], o["s_max"])
                j = ((s - _S0) * _SSC).round().long().clamp_(0, _NF - 1)
                _ol = torch.gather(self.LOf, 0, j.unsqueeze(0)).reshape(1, n_t, 1, 1, C)
                _oh = torch.gather(self.HIf, 0, j.unsqueeze(0)).reshape(1, n_t, 1, 1, C)
                if _SGATE > 0.0 and self.band_lo is not None \
                        and self.band_lo.shape[-1] == _ol.shape[-1]:
                    _g = ((s - _SGATE) / _SRAMP).clamp(0.0, 1.0).reshape(1, n_t, 1, 1, C)
                    _ol = self.band_lo + _g * (_ol - self.band_lo)
                    _oh = self.band_hi + _g * (_oh - self.band_hi)
                _rl, _rh = self._pe_shape(x, C)
                if _rl is not None and _rl.shape[1] == n_t:
                    _ol, _oh = _ol * _rl, _oh * _rh
                lo, hi = _ctr + _ol, _ctr + _oh
            elif (self.band_lo is not None and p.shape[1] == self.band_lo.shape[1]
                    and p.shape[-1] == self.band_lo.shape[-1]):
                _ol, _oh = self.band_lo, self.band_hi
                _rl, _rh = self._pe_shape(x, C)
                if _rl is not None and _rl.shape[1] == n_t:
                    _ol, _oh = _ol * _rl, _oh * _rh
                lo, hi = _ctr + _ol, _ctr + _oh
            else:
                lo, hi = _ctr - self.hw, _ctr + self.hw

            p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
            lo = torch.nan_to_num(lo, nan=-1e3, posinf=-1e3, neginf=-1e3)
            hi = torch.nan_to_num(hi, nan=1e3, posinf=1e3, neginf=1e3)
            lower, upper = torch.minimum(lo, hi), torch.maximum(lo, hi)

            self._prev_pg = pg
            self._prev_pfin = p
        self._prev_input = x

        # ttt_model.py documents info["adapt_loss"] as "a python float, or None when no adaptation
        # happened this step", and local_eval.py hard-fails only if the KEY is absent. We take no
        # gradient step, and every quantity we do update lives on the device -- turning one into a
        # python float would force a synchronise inside the timed region and cost time_score for a
        # value the scorer never reads. None is both contractually correct and provably handled,
        # since the first step of every trajectory must report None for any submission whatsoever.
        return p, {"adapt_loss": None, "lower": lower, "upper": upper}


class _Blend(torch.nn.Module):
    """Equal-weight mean of two forecasters in the shared normalised space.  XCB (2026-08-18):
    CW128 (convnext, 7.96M) + WSW (selfwarp, 4.24M) share 93.805% of their residual VARIANCE
    against 96.220% for two seeds of one recipe, so there is more to average; and the second
    member is 0.533x the first, so the time bill is a third of a same-recipe 4-member ensemble.
    On T1 this measured +0.101 accuracy+sps for a MEASURED -0.046 time bill.  The vgain head and
    the band shipped beside this file must be the BLEND's own refits (BCP), never a member's."""

    def __init__(self, m1, m2):
        super().__init__()
        self.m1 = m1
        self.m2 = m2

    def forward(self, x):
        return 0.5 * (self.m1(x) + self.m2(x))


def get_ttt_model(submission_dir: str, device: str):
    ckpt = os.path.join(submission_dir, "model.pth")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(_HERE, "model.pth")
    s = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = s["model_state_dict"] if isinstance(s, dict) and "model_state_dict" in s else s
    meta = s.get("meta", {}) if isinstance(s, dict) else {}
    # EVERY architecture key from the checkpoint's own meta, not just `base`.  `in_norm` adds no
    # parameters, so the strict load below CANNOT catch a missing one -- measured cost of exactly
    # that omission on Track 1, 2026-07-29: 0.7 FINAL on a bundle that looked like a real result.
    base = build_from_meta(meta, int(meta.get("t_in", 20)), int(meta.get("t_out", 20)),
                           int(meta.get("c", 3)))
    base.load_state_dict(state)
    # XCB: a second forecaster turns the base into an equal-weight cross-recipe blend.  Absent in
    # single-member bundles, in which case this file is bit-identical to its source template.
    _c2 = ""
    for _cand in ("model2.pth", "model_1.pth"):
        for _root in (submission_dir, _HERE):
            _p = os.path.join(_root, _cand)
            if os.path.exists(_p):
                _c2 = _p
                break
        if _c2:
            break
    if _c2:
        _s2 = torch.load(_c2, map_location="cpu", weights_only=False)
        _st2 = _s2["model_state_dict"] if isinstance(_s2, dict) and "model_state_dict" in _s2 else _s2
        _mt2 = _s2.get("meta", {}) if isinstance(_s2, dict) else {}
        _m2 = build_from_meta(_mt2, int(_mt2.get("t_in", 20)), int(_mt2.get("t_out", 20)),
                              int(_mt2.get("c", 3)))
        _m2.load_state_dict(_st2)
        base = _Blend(base, _m2)
        globals()["_BLEND_LIVE"] = [True]

    calib = None
    for cand in (os.path.join(submission_dir, "calib.npz"), os.path.join(_HERE, "calib.npz")):
        if os.path.exists(cand):
            try:
                with np.load(cand) as z:
                    calib = {k: z[k] for k in z.files}
            except Exception:
                calib = None            # unreadable calibration must not take the submission down
            break
    return CalibratedTTTModel(base, device, calib, online=ONLINE,
                              submission_dir=submission_dir)
