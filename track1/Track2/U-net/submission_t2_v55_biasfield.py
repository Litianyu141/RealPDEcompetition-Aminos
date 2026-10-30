"""RealPDE Track 2 (LTTTA) submission -- v4's calibration, captured as a single CUDA graph.

v55 BIASFIELD (2026-09-02) = v54 + a LOW-VARIANCE estimator of the SAME residual the bias/sp EMAs
average: the revealed residual r = _sy - _sg_pg is block-pooled to fld_res (h x w) and, if fld_lead ==
"pooled", averaged over the 20 leads, EMA'd (fld_beta, its own second moment _fld_m2 and power _pw_fld,
JS-gated with c = gate_c or fld_gate_c, clipped at fld_clip * mad), and SUBTRACTED (up-sampled by
reshape/expand, static shapes, no F.interpolate) from r BEFORE the bias and sp EMAs see it, so those
estimate the remainder.  Ungated and unclipped, field + b + s == v53's b + s exactly; the field
changes only WHERE the gate acts (a pooled block mean has ~20 x (H/h)(W/w) lower sampling variance than
a (lead, pixel) cell, so the gate keeps it).  The estimator runs FIRST in the prediction-independent
prefix at all four sites; the application is ONE more term inside `_gate_corr` (side2 in par4), so the
main stream still pays exactly one add (`_corr`).  fld_beta 0 (default) allocates and executes nothing:
v55 is then statement-for-statement v54 (proven by md5 on the output arrays).  fld_pyr 1 stacks the
levels 4x8 -> 8x16 -> 32x64, each on the remainder of the previous.  No new capture, no empty_cache,
`_strpar_eligible` unchanged, `_side`/`_side2` untouched.

v54 BIASGATE (2026-09-02) = v53 + a positive-part James-Stein / soft-threshold SHRINKAGE GATE on
the two online corrections.  `b` (per-(lead,ch) residual-mean EMA) and `s` (per-element spatial
residual EMA) are each multiplied, BEFORE the clamp, by
    js:   f = clamp(1 - gate_c * v / (m^2 + gate_eps), 0, 1)
    soft: f = clamp(1 - gate_c * sqrt(v) / (|m| + sqrt(gate_eps)), 0, 1)
where m is the debiased EMA mean of the SAME statistic and v its own sampling variance,
    v = (m2*cb - (m*cb)^2)_+ * phi_k,   phi_k = beta/(2-beta) * (1+pw)/(1-pw),  pw = (1-beta)^k,
built from a SECOND-moment EMA of the same quantity (`_bias_m2` of rm^2 per (lead,ch); `_sp_m2`
of rs^2 per element) with the same beta and the same `cb` (they share `_pw_bias`/`_pw_sp`).
phi_k is the exact sum of squared normalised EMA weights at step k (1 at k=1, -> beta/(2-beta)).
NEVER `_mad_m`: that is a per-element residual scale, ~N_eff larger than the variance of a
spatial mean, and would shrink `b` to zero everywhere.  `gate_c = 0` (default) allocates nothing
and executes nothing -- v54 is then statement-for-statement v53 (proven by md5 on the output
arrays).  The update lives next to the bias EMAs in the prediction-independent prefix (all four
sites: `_body`, `_body_strpar` par3 + par4, eager); the factor is multiplication only, so the
body stays capturable and the capture law is untouched (par4 is still the last and only
capture; no `empty_cache()`; `_side`/`_side2` only inside `_body_strpar`).
FUSED PLACEMENT (same day, after the v53 online row showed time as the live risk): with the gate ON
the whole gated/clamped/warm-scaled `b + s` term is computed by `_gate_corr` in the PREFIX (main
before the fork in par3; side2 before e4 in par4; step 4 of `_body`) into the static `_corr`, and
the main stream pays ONE add -- fewer main-stream launches than v53's two applications.  Note the
float order is p + (b + s) instead of (p + b) + s, so a gate-ON output is not expected to be
bit-identical to the un-fused text (4-dp subscores agree); gate-OFF is untouched and bit-exact v53.
The eager fallback keeps the un-fused inline form.

v53 T2BIASPAR (2026-09-02) = v51 PAR4 with the `bias_beta` / `sp_beta` residual-EMA terms
PORTED INTO `_body_strpar`, and the two clauses that refused them dropped from
`_strpar_eligible`.  Nothing else changes; with both knobs at zero every added statement is
skipped by the same `if` that skips it in `_body`, so a v53 bundle built with v51's ONLINE dict
is v51 statement-for-statement (proven by md5 on the output arrays, `_t2bp_v53null` vs
`_t2bp_null`).

  WHY.  T2REXBIAS (README, 2026-09-01) measured `bias_beta 0.05, sp_beta 0.05` at +0.03667
  Final on the population that reproduces Track 2's online row (rex-f2, held out) and could
  not ship it: `_strpar_eligible` refused whenever either knob was non-zero, so turning one
  on dropped the card to the SEQUENTIAL body and forfeited STRPAR's banked +0.0808.  The
  refusal was a declaration that `_body_strpar` did not implement the terms -- it now does.

  WHERE THE TWO TERMS GO, and why the fork stays safe.  The UPDATE reads only `_sy` and LAST
  step's frozen `_sg_pg` (through `r`, the residual v51 computed and left dead), so it is
  prediction-independent and lives in the prefix: on the main stream before the fork in
  par3, on side2 before `e4` in par4 -- identical statements in identical order to `_body`'s
  step 1, ahead of the band EMA.  The APPLICATION is a serial add on the main stream after
  the gain head and after `self._sg_pg.copy_(p)`, before `_ctr` -- `_body`'s step 4
  verbatim.  Four static buffers cross the stream boundary (`_bias_m`, `_sp_m`, `_pw_bias`,
  `_pw_sp`); main reads them only after `cur.wait_event(e4)`, the edge that already orders
  side2's reads of `_sg_pg`/`_sg_pfin` ahead of main's writes, and next step's `ef` orders
  main's reads ahead of side2's next writes (WAR).  `usable`/`wf` are consumed on main the
  way `band_ol`/`band_oh` already are: a side2 temporary read after an event.  `_band_branch`
  never reads `p`, so the band fork is untouched and `_pre` stays the pre-head field.

  THE FREEZE TRAP, named so the next editor does not fall into it.  `if o["bias_beta"] > 0.0
  and self._lim_bias is not None` is a PYTHON branch, evaluated ONCE while the graph is
  captured.  `_lim_bias`/`_lim_sp` are built in `_alloc()` from `mad_ref`, and `__init__` runs
  `_alloc()` before `_capture()`; were either still None at capture, the correction would bake
  out of the graph while every gate in `_capture` printed OK (the gates compare the par body to
  `_body`, which would have baked out identically).  A config is therefore not a witness: the
  actuator witness is `analysis/t2biaspar_witness.py`, which compares the packaged card's
  output tensors against the same-session null and reads `_bias_m`/`_sp_m` off the model.
  `_BIASPAR_LIVE` below is a branch-taken flag only, `_SBPORT_LIVE`-shaped, not proof.

v50 STRPAR (2026-08-26) = v49 SBPORT + a multi-stream fork/join INSIDE the same capture.
Blend member m2 runs on a second stream and the prediction-independent band branch on a
third; only `lo = _ctr + _ol` needs the forecast, so nothing races.  BIT-EXACT -- the
accuracy trio and sps are identically the v49 numbers, and two gates in `_capture` prove
it rather than asserting it.  The general `_body` below is byte-identical to v49 and is
what still runs for every other config and as the fallback.

v51 PAR4 + T2IMG (2026-08-28) adds two things to v50 and changes no numerics at all.

  PAR4 -- one more rung of the same fork.  v50 left a prediction-INDEPENDENT prefix on the
  main stream: the band MAD-EMA and its debias powers, the `_k_dev` step counter with the
  `usable`/`wf` scalars derived from it, and the dead `r` residual `_body` computes.  All of
  it reads only `_sx`, `_sy`, `_sg_pg`, `_sg_pfin` and the EMA state -- never the forecast --
  so it belongs on the side stream that already consumes it.  Moved with it: the INPUT-ONLY
  half of the vgain features (`li`, `ib`, i.e. everything derived from `x_ph` rather than from
  the prediction), which the main stream previously recomputed while the band branch was
  computing its own copy of the very same `x_ph` denormalisation.  In par4 `x_ph` is
  denormalised ONCE, on the side stream, and serves both consumers.  Itemised against the T2
  step anatomy the moved work is s01 0.006 + s02 0.035 + s10 0.007 + the input half of s06
  ~0.035 ms -- so this is a SMALL rung, and it is priced as one.  What it is NOT is a deletion:
  every kernel v50 ran, v51 still runs.  ULV's arithmetic (which priced DELETING band work)
  therefore does not bound it, and the accuracy legs are exactly zero by construction.

  THE DAG, because a fork is only as safe as its edges.  Side stream 2 runs, in order:
  `r` -> band EMA -> `_k_dev`/`usable`/`wf` -> `x_ph` -> (`li`, `ib`) -> EVENT e4 ->
  `_band_branch` -> EVENT e3.  The main stream waits on e4 before it assembles the vgain
  feature stack and on e3 before it adds the band offsets.  That ordering is what makes the
  two write-after-read hazards safe: main writes `_sg_pg` only after the e4 wait (side2 read
  it in `r` before e4), and writes `_sg_pfin` only after the e3 wait (side2 read it in the
  EMA before e4, hence before e3).  Both cross-stream VALUES (`li`, `ib`) go through
  pre-allocated static buffers, never a temporary, mirroring `_out3`.

  T2IMG -- a MEASURED torch allowlist, and it is a correctness guard rather than a score arm.
  v50 has no version guard of any kind: `_strpar_eligible` is config-only, so on an unmeasured
  torch the card would construct side streams, run the multi-stream body eagerly, and then
  CAPTURE a multi-stream graph -- and only afterwards discover the result was wrong.  Every
  gate is a post-hoc detector of a computation that has already run, and on Track 1 that exact
  capture was measured to poison the CUDA context in a way `try/except` cannot catch (an async
  illegal access is a sticky process-level fault).  So `_strpar_eligible` gains a clause that
  is evaluated BEFORE the streams are constructed and before any gate runs, and the streams
  themselves are not constructed at all on a torch that is not on the list.

  THE LIST IS PER-CARD-BODY, NOT PER-TRACK, AND THE TRACK-1 RULE MUST NOT BE COPIED HERE.
  On Track 1, torch 2.2.2 is measured MEMORY-CORRUPTING with that card's par3 body and is
  excluded by a `>= 2.8` rule.  On Track 2, torch 2.2.2 is measured GOOD -- twice: online, by
  submission 902563 on an A800, and locally, by running the shipped card inside the
  digest-pinned evaluation image on a V100 -- and it is THE EVALUATION IMAGE'S OWN TORCH.  A
  `>= 2.8` rule here would refuse par-mode on the proven-good production image, ship a
  guaranteed sequential no-op, and forfeit the +0.0808 Final v50 banked.  ALLOW = {2.2.2
  (measured good online AND locally), 2.8.0 (measured good at build time on V100)};
  everything else REFUSES to sequential, which is v49's behaviour and is what H2 always
  claimed the worst case was.  The comparison is against `torch.__version__` with the local
  build tag stripped -- the build box reports `2.8.0+cu128`, so an exact-string match would
  be a false refusal on the happy path.

  THE FALLBACK IS A LADDER NOW, par4 -> par3 -> sequential.  v50 fell straight from par3 to
  sequential.  If par4 fails a gate in some environment we have not measured, dropping all the
  way to sequential would forfeit par3's already-banked online gain for no reason, so the gate
  loop tries par3 before it gives up on forking.  par3 is byte-for-byte the body that is on the
  board today.

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
    # BIASGATE (v54): shrinkage gate on b and s.  gate_c 0 == off (bit-exact v53).
    "gate_c": 0.0, "gate_mode": "js", "gate_eps": 1e-10,
    # BIASFIELD (v55): pooled / coarse residual field.  fld_beta 0 == off (bit-exact v54).
    "fld_beta": 0.0, "fld_res": "8x16", "fld_lead": "pooled", "fld_gate_c": -1.0,
    "fld_scale": 1.0, "fld_clip": 1.50, "fld_pyr": 0,
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
# SBPORT (2026-08-23): Track 1's learned per-element band (sband head + quantile table) deployed
# in place of the pe/pc static band.  XBAND: on the identical 378-window population with the
# identical forecaster the T1 band scores A 0.71163 vs T2's 0.70249 (+0.710 sps).  1.0 == on;
# 0.0 (or sband.pth absent) == the pe/pc path, bit-identical to v46.
_SBPORT = 1.0
_SBPORT_LIVE = [False]
_SBPORT_STATUS = ['absent']
_BLEND_LIVE = [False]        # XCB execution witness        # execution witness; a bare `except` cannot fake a list mutation

# ---- STRPAR (2026-08-26): multi-stream fork/join INSIDE the capture -------------------------
# Measured on V100 against this very card: SEQ 5.033 -> PAR2 3.977 -> PAR3 3.443 ms/step
# device-complete, 3 passes, spread 0.007 ms.  Dropping member 2 entirely costs 3.343 ms, so the
# fork makes the second forecaster's marginal cost 0.100 ms instead of 1.692 ms.
# T2_NO_STRPAR=1 forces the sequential body (the null arm; must reproduce the shipped row exactly).
_STRPAR = os.environ.get("T2_NO_STRPAR", "") == ""
_STRPAR_LIVE = [False]       # execution witness: the multi-stream body is what got captured
_STRPAR_REFUSED = [False]    # the eligibility guard or the bit-exactness gate said no
_STRPAR_MODE = [""]          # "par4" | "par3" | "par2" | "" -- what actually shipped
# T2BIASPAR (v53): [bias applied, sp applied] inside `_body_strpar`.  A branch-taken flag, set
# while the body runs eagerly for the gates and again at capture; the actuator witness is the
# output-tensor comparison in analysis/t2biaspar_witness.py, never this list.
_BIASPAR_LIVE = [False, False]
_BIASFIELD_LIVE = [False]      # BIASFIELD (v55): branch-taken flag, set inside `_gate_corr`


def _fld_levels(o):
    """BIASFIELD (v55): the (h, w) ladder -- the pyramid or the single fld_res."""
    if int(o["fld_pyr"]):
        return [(4, 8), (8, 16), (32, 64)]
    h, w = str(o["fld_res"]).lower().split("x")
    return [(int(h), int(w))]


def _fld_down(r, h, w):
    """(1, T, H, W, C) -> (1, T, h, w, C) block mean.  Static shapes; h == H, w == W is a plain copy."""
    _, T, H, W, C = r.shape
    return r.reshape(1, T, h, H // h, w, W // w, C).mean(dim=(3, 5))


def _fld_up(f, H, W):
    """(1, T, h, w, C) -> (1, T, H, W, C) by reshape/expand/reshape (the last reshape copies)."""
    _, T, h, w, C = f.shape
    return f.reshape(1, T, h, 1, w, 1, C).expand(1, T, h, H // h, w, W // w, C).reshape(1, T, H, W, C)
# BIASGATE (v54, fused): [gated correction applied via `_corr` inside `_body_strpar`].  A
# branch-taken flag only; the actuator witness is analysis/t2biasgate_witness.py.
_BIASGATE_LIVE = [False]

# ---- T2IMG (2026-08-28): the MEASURED torch allowlist ---------------------------------------
# Read the header for why this is per-card-body and why Track 1's ">= 2.8" rule is the exact
# wrong rule here.  Short form: 2.2.2 is the T2 evaluation image's own torch and is measured
# good for THIS body both online (submission 902563, A800) and locally (V100 sm_70, inside the
# digest-pinned image); refusing it would ship a guaranteed no-op.
#
# The strip matters.  `torch.__version__` carries a local build tag -- the box this card is
# built on reports "2.8.0+cu128" -- so an exact-string match against "2.8.0" would refuse
# par-mode on the build box itself.  That is the false-refusal-on-the-happy-path failure this
# repo has already paid for once; the version is compared with the tag removed.
_STRPAR_TORCH_OK = ("2.2.2", "2.8.0")


def _torch_base_version() -> str:
    """`torch.__version__` with any local/build tag stripped ("2.8.0+cu128" -> "2.8.0")."""
    try:
        return str(torch.__version__).split("+")[0].strip()
    except Exception:
        return ""


_STRPAR_TV = _torch_base_version()
# [allowed?, the version actually seen] -- an execution witness the builder records.  A bare
# `except` cannot fake a module-level list whose second element is the real version string.
_STRPAR_TVOK = [_STRPAR_TV in _STRPAR_TORCH_OK, _STRPAR_TV]

# Every capture this module performs, in the order it performed them: ("multistream", mode) or
# ("sequential", "").  Appended inside `_capture_body`, which is the ONLY capture site in this
# file, so the log is complete by construction rather than by discipline.  `build_t2_bundle.py`
# reads it and REFUSES a bundle whose live multi-stream graph is not the last capture -- see the
# CAPTURE-ORDER block above `_capture_body` for why that ordering is a correctness requirement
# and not a preference.
_CAP_LOG = []


class CalibratedTTTModel(TTTModel):
    def __init__(self, base: nn.Module, device: str, calib: Optional[dict] = None,
                 ttt_lr: float = TTT_LR, ttt_steps: int = TTT_STEPS,
                 halfwidth: float = HALFWIDTH, online: Optional[dict] = None,
                 submission_dir: str = ""):
        super().__init__()
        self.base = base.to(device)
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
        # SBPORT: Track 1's sband head + per-(lead,ch) quantile table (sb_edge/sb_lo/sb_hi).
        self._sb = None
        self._sb_edge = self._sb_lo = self._sb_hi = None
        self._sb_lt = self._sb_c2 = self._sb_q = 0
        if _SBPORT != 0.0:
            try:
                cands = [os.path.join(d, 'sband.pth') for d in (submission_dir, _HERE) if d]
                sp = next((c for c in cands if os.path.exists(c)), '')
                if (sp and self._nmean is not None and calib is not None
                        and all(k in calib for k in ('sb_edge', 'sb_lo', 'sb_hi'))):
                    from sband_head import build_from_blob as _sb_build, sband_features as _sb_feat
                    _e = np.asarray(calib['sb_edge'], np.float32)      # (LT, C2, Q-1)
                    _l = np.asarray(calib['sb_lo'], np.float32)        # (LT, C2, Q) NORMALISED
                    _u = np.asarray(calib['sb_hi'], np.float32)        # (LT, C2, Q) NORMALISED
                    assert _l.ndim == 3 and _u.shape == _l.shape and _e.ndim == 3
                    _lt, _c2, _q = int(_l.shape[0]), int(_l.shape[1]), int(_l.shape[2])
                    assert _e.shape == (_lt, _c2, _q - 1) and _lt == 20 and _c2 == 2 and 2 <= _q <= 4096
                    assert np.isfinite(_e).all() and np.isfinite(_l).all() and np.isfinite(_u).all()
                    assert (_u >= _l).all() and (np.diff(_e, axis=-1) >= 0).all()
                    _blob = torch.load(sp, map_location='cpu', weights_only=False)
                    assert int(_blob['n_out']) == _lt * _c2
                    _hd = _sb_build(_blob).to(device).eval()
                    for q_ in _hd.parameters():
                        q_.requires_grad_(False)
                    _g = _lt * _c2
                    self._sb_feat = _sb_feat
                    self._sb_edge = torch.as_tensor(_e.reshape(_g, _q - 1)).contiguous().to(device)
                    self._sb_lo = torch.as_tensor(_l.reshape(_g, _q)).contiguous().to(device)
                    self._sb_hi = torch.as_tensor(_u.reshape(_g, _q)).contiguous().to(device)
                    self._sb_lt, self._sb_c2, self._sb_q = _lt, _c2, _q
                    self._sb = _hd
                    # live probe on STRUCTURED input (a zeros probe sails through a dead head)
                    _pg = torch.Generator(device='cpu').manual_seed(20260823)
                    _pa = (torch.randn((1, 20, 32, 64, 3), generator=_pg) * 0.5).to(device)
                    _pa[..., 2] = 0.0
                    with torch.inference_mode():
                        _po = self._sb_offsets(_pa, 3)
                    assert _po[0] is not None and tuple(_po[0].shape) == (1, 20, 32, 64, 3)
                    assert bool(torch.isfinite(_po[0]).all()) and bool((_po[1] >= _po[0]).all())
                    assert float((_po[1] - _po[0])[..., :2].std()) > 1e-7, 'head is dead'
                    _SBPORT_STATUS[0] = 'ok'
                else:
                    _SBPORT_STATUS[0] = 'absent'
            except Exception as _sbe:
                self._sb = None
                _SBPORT_STATUS[0] = 'fail:' + type(_sbe).__name__
        self._H, self._W = 32, 64
        if self.gamma is not None and self.gamma.dim() == 5:
            self._H, self._W = int(self.gamma.shape[2]), int(self.gamma.shape[3])
        km = self.cm if self.cm else 2
        w = getattr(self, "_ke_w_np", None)
        w = np.ones(km, np.float32) if w is None or w.size < km else w[:km].astype(np.float32)
        self._ke_w = torch.as_tensor(w).to(device).reshape(1, 1, 1, km)
        self._alloc()
        # T2BIASPAR freeze-trap guard (2026-09-02, capture review).  The application inside BOTH
        # bodies is behind Python-level branches -- `if o["bias_beta"] > 0.0 and self._lim_bias
        # is not None` and `if self._sp_m is not None` -- which are evaluated ONCE and frozen at
        # capture.  `_lim_bias` / `_lim_sp` are filled in `_alloc()` only if `mad_ref` / `ch_idx`
        # survived `_build_online_tables`'s bare except; with a calib lacking s_grid / band_lo_s /
        # mad_val / ch_idx they stay None, the correction bakes OUT of `_body` and `_body_strpar`
        # alike, every torch.equal gate still passes (both bodies bake out identically) and the
        # build prints BUILD_OK while the EMAs keep updating a term nothing applies.  Refuse at
        # construction instead.  Cannot false-refuse on the happy path: the card's calib carries
        # mad_ref (v52 shipped it), and with both knobs at zero neither line executes.
        if self.o["bias_beta"] > 0.0:
            assert self._lim_bias is not None, (
                "T2BIASPAR: bias_beta > 0 but _lim_bias is None (calib lacks mad_val/ch_idx) -- "
                "the bias term would bake out at capture")
        if self.o["sp_beta"] > 0.0:
            assert self._sp_m is not None and self._lim_sp is not None, (
                "T2BIASPAR: sp_beta > 0 but _sp_m/_lim_sp is None (calib lacks mad_val/ch_idx) -- "
                "the sp term would bake out at capture")
        # BIASGATE (v54) freeze trap: the gate statements are behind `if self._bias_m2 is not None`
        # (frozen at capture); with gate_c > 0 the buffers must exist exactly where the betas are on.
        if self.o["gate_c"] > 0.0:
            assert self.o["gate_mode"] in ("js", "soft"), self.o["gate_mode"]
            assert self.o["bias_beta"] > 0.0 or self.o["sp_beta"] > 0.0, "BIASGATE: nothing to gate"
            assert (self._bias_m2 is not None) == (self.o["bias_beta"] > 0.0), "BIASGATE: _bias_m2"
            assert (self._sp_m2 is not None) == (self.o["sp_beta"] > 0.0), "BIASGATE: _sp_m2"
        # BIASFIELD (v55) freeze trap: the field statements are behind `if self._fld_m is not None`
        # (frozen at capture) and the application lives ONLY inside `_gate_corr` (needs the gate on).
        if self.o["fld_beta"] > 0.0:
            assert self._fld_m is not None and self._fld_m2 is not None, "BIASFIELD: _fld_m"
            assert self._lim_fld is not None, "BIASFIELD: fld_beta > 0 but _lim_fld is None (calib lacks mad_val)"
            assert self._gate_on, "BIASFIELD: fld_beta > 0 needs gate_c > 0 (applied inside _gate_corr)"
            assert str(self.o["fld_lead"]) in ("pooled", "perlead"), self.o["fld_lead"]
            assert float(self.o["fld_gate_c"]) >= -1.0, self.o["fld_gate_c"]
            for (_h, _w) in self._fld_levels:
                assert self._H % _h == 0 and self._W % _w == 0, (self._H, self._W, _h, _w)
        else:
            assert self._fld_m is None and self._fld_levels == [], "BIASFIELD: allocated while off"
        self._graph = None
        self._have_flag = None
        # STRPAR: the side streams are created HERE, before _capture(), because a CUDA-graph body
        # must not construct anything it reads (project rule, the v65 fallback bake).  They are plain
        # attributes so a non-CUDA device leaves them None and the eligibility guard refuses.
        #
        # T2IMG: the allowlist is consulted HERE, before the streams exist.  On an unmeasured
        # torch nothing multi-stream is ever constructed, so `_strpar_eligible` refuses, no
        # multi-stream body runs and no multi-stream graph is captured -- which is the only
        # placement that can prevent those operations rather than report on them afterwards.
        self._side = self._side2 = None
        if _STRPAR and _STRPAR_TVOK[0] and str(device).startswith("cuda") \
                and torch.cuda.is_available():
            try:
                self._side = torch.cuda.Stream()
                self._side2 = torch.cuda.Stream()
            except Exception:
                self._side = self._side2 = None
        self.reset_ttt_state()
        if USE_GRAPH and self.ttt_lr == 0.0 and str(device).startswith("cuda") \
                and torch.cuda.is_available():
            try:
                self._capture()
            except Exception:
                self._graph = None
            finally:
                self.reset_ttt_state()

    def _sb_offsets(self, xn, C, x_ph=None):
        """NORMALISED (1,T,H,W,>=2) input window -> (lo, hi) NORMALISED per-element offsets,
        each (1, LT, H, W, C), zero on channels the table does not cover.  Mirrors Track 1's
        _sband_offsets exactly: physical features -> head -> per-(lead,ch) quantile bin -> table.
        Deterministic (searchsorted + gather): graph replay is bit-identical to the eager body.

        `x_ph` (PAR4) lets a caller hand in the denormalised window it has already built, so the
        side stream computes `xn * std + mean` once for both its consumers instead of twice.  It
        DEFAULTS to None and is recomputed here, so `_body`, the eager path and the `__init__`
        liveness probe are untouched and keep calling this with two arguments."""
        if self._sb is None or self._sb_lo is None:
            return None, None
        c2 = self._sb_c2
        if xn.shape[-1] < c2 or C < c2:
            return None, None
        if x_ph is None:
            x_ph = xn * self._nstd + self._nmean                    # the head was trained PHYSICAL
        m = self._sb(self._sb_feat(x_ph))                           # (1, LT*C2, H, W)
        b_, h_, w_ = int(m.shape[0]), int(m.shape[2]), int(m.shape[3])
        g = self._sb_lt * c2
        if int(m.shape[1]) != g:
            return None, None
        mv = m.reshape(b_, g, h_ * w_).permute(1, 0, 2).reshape(g, b_ * h_ * w_).contiguous()
        bi = torch.searchsorted(self._sb_edge, mv)                  # (g, B*H*W) in [0, Q-1]
        lo = torch.gather(self._sb_lo, 1, bi).reshape(self._sb_lt, c2, b_, h_, w_).permute(2, 0, 3, 4, 1)
        hi = torch.gather(self._sb_hi, 1, bi).reshape(self._sb_lt, c2, b_, h_, w_).permute(2, 0, 3, 4, 1)
        if C > c2:
            lo = torch.nn.functional.pad(lo, (0, C - c2), value=0.0)
            hi = torch.nn.functional.pad(hi, (0, C - c2), value=0.0)
        return lo, hi

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
        # BIASGATE (v54): second-moment EMAs of the SAME statistics the bias EMAs average.  None
        # when the gate is off, so every gate statement is skipped and v54 == v53.
        _gc = self.o["gate_c"] > 0.0
        self._bias_m2 = z(n_t, C) if (_gc and self.o["bias_beta"] > 0) else None
        self._sp_m2 = z(1, n_t, H, W, C) if (_gc and self.o["sp_beta"] > 0) else None
        self._gate_on = bool(self._bias_m2 is not None or self._sp_m2 is not None)
        self._corr = z(1, n_t, H, W, C) if self._gate_on else None     # static: crosses the stream boundary
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
        # BIASFIELD (v55): one (m, m2) pair per level, shared power; None when off so every field
        # statement is skipped (frozen at capture) and v55 == v54.
        self._fld_levels = []
        self._fld_m = self._fld_m2 = self._lim_fld = None
        self._fld_pooled = (str(self.o["fld_lead"]) == "pooled")
        self._pw_fld = z(())
        if self.o["fld_beta"] > 0.0:
            self._fld_levels = _fld_levels(self.o)
            T_f = 1 if self._fld_pooled else n_t
            self._fld_m = [z(1, T_f, h, w, C) for (h, w) in self._fld_levels]
            self._fld_m2 = [z(1, T_f, h, w, C) for (h, w) in self._fld_levels]
            if self.mad_ref is not None and self.ch_idx is not None:
                lim = (self.o["fld_clip"] * self.mad_ref)[:, self.ch_idx]          # (n_t, C)
                if self._fld_pooled:
                    lim = lim.mean(dim=0, keepdim=True)                             # lead-mean MAD
                self._lim_fld = lim.reshape(1, T_f, 1, 1, C).contiguous()

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
        # PAR4 cross-stream buffers.  The input-only half of the vgain features is computed on
        # the side stream and read on the main one, so it goes through STATIC memory exactly
        # like `_out3` -- a graph replays fixed kernels on fixed addresses, and a temporary
        # handed across streams is the one thing a capture must never rely on.  Shapes follow
        # `_vg_features`: KEi/li collapse the channel axis, ib keeps it.
        self._vg_li = z(1, H, W)               # log(KEi + eps), input-only
        self._vg_ib = z(1, H, W, 2)            # time-mean of the input's u,v
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
        for b in (self._bias_m, self._sp_m, self._gam_m, self._gamf_m, self._mad_m,
                  self._bias_m2, self._sp_m2):
            if b is not None:
                b.zero_()
        for b in (self._fld_m or []) + (self._fld_m2 or []):      # BIASFIELD (v55)
            b.zero_()
        # graph state: the running powers start at (1-beta)^0 = 1
        for b in (self._k_dev, self._have, self._sy, self._sg_pg, self._sg_pfin):
            b.zero_()
        for b in (self._pw_bias, self._pw_sp, self._pw_gam, self._pw_gamf, self._pw_band, self._pw_fld):
            b.fill_(1.0)
        self._have_flag = False

    # ------------------------------------------------------------------ BIASGATE (v54)
    def _gate_factor(self, m, m2, pw, beta, c=None):
        """Positive-part James-Stein (js) / soft-threshold (soft) factor in [0, 1] for a debiased
        EMA mean `m` whose debiased second moment is `m2` (same statistic, same beta, same cb).
        v = (m2 - m^2)_+ * phi_k is the estimator's OWN sampling variance at step k, phi_k the exact
        sum of squared normalised EMA weights.  Tensor ops only (capturable); `pw` may be the in-graph
        0-d `_pw_*` tensor or, on the eager path, a python float.  At usable == 0: m == m2 == 0 and
        pw == 1, so v == 0 * finite == 0 and f == 1 -- finite, never NaN.  At k == 1 v == 0 exactly (one
        sample): the gate cannot act before a trajectory's second adapting step, by construction."""
        o = self.o
        var_t = (m2 - m * m).clamp_min(0.0)
        den = 1.0 - pw
        den = den.clamp_min(1e-12) if torch.is_tensor(den) else max(den, 1e-12)
        phi = (beta / (2.0 - beta)) * (1.0 + pw) / den
        v = var_t * phi
        c = float(o["gate_c"]) if c is None else float(c)
        if o["gate_mode"] == "soft":
            return (1.0 - c * v.sqrt() / (m.abs() + float(o["gate_eps"]) ** 0.5)).clamp(0.0, 1.0)
        return (1.0 - c * v / (m * m + float(o["gate_eps"]))).clamp(0.0, 1.0)

    def _fld_apply(self, cb, pw, wf):
        """BIASFIELD (v55): sum over levels of up(clamp(gate(fm))) * fld_scale * wf, from prefix state
        only (`_fld_m`/`_fld_m2`/`_pw_fld`).  `pw` is the in-graph 0-d tensor or an eager float."""
        o = self.o
        c = float(o["gate_c"]) if float(o["fld_gate_c"]) < 0.0 else float(o["fld_gate_c"])
        tot = None
        for _i in range(len(self._fld_levels)):
            fm = self._fld_m[_i] * cb
            if c > 0.0:
                fm = fm * self._gate_factor(fm, self._fld_m2[_i] * cb, pw, o["fld_beta"], c)
            tm = _fld_up(fm.clamp(-self._lim_fld, self._lim_fld), self._H, self._W) \
                * (o["fld_scale"] * wf)
            tot = tm if tot is None else tot + tm
        return tot

    def _gate_corr(self, usable, wf) -> None:
        """BIASGATE (v54, fused): the gated, clamped, warm-scaled `b + s` correction, built from prefix
        state only (`_bias_m`/`_bias_m2`/`_pw_bias`, `_sp_m`/`_sp_m2`/`_pw_sp`, `usable`, `wf`) and
        written into the static `_corr`.  In par4 this runs on side2 before e4, so main pays ONE add.
        Gate-off never calls this (the v53 statements run instead, bit-exact)."""
        o = self.o
        n_t, C = self._n_t, self._C
        corr = None
        if self._bias_m2 is not None:
            cb = usable / (1.0 - self._pw_bias).clamp_min(1e-12)
            bm = self._bias_m * cb
            bm = bm * self._gate_factor(bm, self._bias_m2 * cb, self._pw_bias, o["bias_beta"])
            corr = (bm.clamp(-self._lim_bias, self._lim_bias) * (o["bias_scale"] * wf)) \
                .reshape(1, n_t, 1, 1, C)
        if self._sp_m2 is not None:
            cb = usable / (1.0 - self._pw_sp).clamp_min(1e-12)
            sm = self._sp_m * cb
            sm = sm * self._gate_factor(sm, self._sp_m2 * cb, self._pw_sp, o["sp_beta"])
            s = sm.clamp(-self._lim_sp, self._lim_sp) * (o["sp_scale"] * wf)
            corr = s if corr is None else corr + s
        if self._fld_m is not None:                  # BIASFIELD (v55): one more prefix term
            cb = usable / (1.0 - self._pw_fld).clamp_min(1e-12)
            fs = self._fld_apply(cb, self._pw_fld, wf)
            corr = fs if corr is None else corr + fs
            _BIASFIELD_LIVE[0] = True
        self._corr.copy_(corr)

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
            if self._fld_m is not None:                 # BIASFIELD (v55): first in the prefix
                a = have * o["fld_beta"]
                for _i, (_h, _w) in enumerate(self._fld_levels):
                    rf = _fld_down(r, _h, _w)
                    if self._fld_pooled:
                        rf = rf.mean(dim=1, keepdim=True)
                    if self._chmask is not None:
                        rf = rf * self._chmask
                    self._fld_m[_i].mul_(1.0 - a).add_(rf * a)
                    self._fld_m2[_i].mul_(1.0 - a).add_(rf * rf * a)
                    r = r - _fld_up(rf, self._H, self._W)
                self._pw_fld.mul_(1.0 - a)
            if o["bias_beta"] > 0.0:
                a = have * o["bias_beta"]
                rm = r.mean(dim=(0, 2, 3))            # (n_t, C)
                if self._chmask is not None:
                    rm = rm * self._chmask.reshape(1, -1)
                self._bias_m.mul_(1.0 - a).add_(rm * a)
                if self._bias_m2 is not None:
                    self._bias_m2.mul_(1.0 - a).add_(rm * rm * a)
                self._pw_bias.mul_(1.0 - a)
            if self._sp_m is not None:
                a = have * o["sp_beta"]
                rs = r - r.mean(dim=(0, 2, 3), keepdim=True)   # orthogonal to the bias term
                if self._chmask is not None:
                    rs = rs * self._chmask
                self._sp_m.mul_(1.0 - a).add_(rs * a)
                if self._sp_m2 is not None:
                    self._sp_m2.mul_(1.0 - a).add_(rs * rs * a)
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
            if self._gate_on:
                self._gate_corr(usable, wf)          # BIASGATE (v54, fused)
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

            if self._gate_on:
                # BIASGATE (v54, fused): the gated b + s term was built in the prefix from prefix
                # state only (side2 in par4, published by e4 -- main has already waited on it); ONE
                # add on the main stream replaces the two applications below.
                p = p + self._corr
            if (not self._gate_on) and o["bias_beta"] > 0.0 and self._lim_bias is not None:
                cb = usable / (1.0 - self._pw_bias).clamp_min(1e-12)
                b = (self._bias_m * cb).clamp(-self._lim_bias, self._lim_bias) \
                    * (o["bias_scale"] * wf)
                p = p + b.reshape(1, n_t, 1, 1, C)

            if (not self._gate_on) and self._sp_m is not None and self._lim_sp is not None:
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
                _sl, _sh = self._sb_offsets(self._sx, C) if self._sb is not None else (None, None)
                if _sl is not None and _sl.shape[1] == n_t:
                    # SBPORT: T1's absolute per-element offsets x the RELATIVE widening the
                    # gate applied to the frozen table (== 1 exactly while the gate is shut)
                    _dl = torch.where(self.band_lo != 0, self.band_lo, torch.ones_like(self.band_lo))
                    _dh = torch.where(self.band_hi != 0, self.band_hi, torch.ones_like(self.band_hi))
                    _ol, _oh = _sl * (_ol / _dl), _sh * (_oh / _dh)
                    _SBPORT_LIVE[0] = True
                else:
                    _rl, _rh = self._pe_shape(self._sx, C)
                    if _rl is not None and _rl.shape[1] == n_t:
                        _ol, _oh = _ol * _rl, _oh * _rh
                lo, hi = _ctr + _ol, _ctr + _oh
            elif (self.band_lo is not None and p.shape[1] == self.band_lo.shape[1]
                    and p.shape[-1] == self.band_lo.shape[-1]):
                _ol, _oh = self.band_lo, self.band_hi
                _sl, _sh = self._sb_offsets(self._sx, C) if self._sb is not None else (None, None)
                if _sl is not None and _sl.shape[1] == n_t:
                    _ol, _oh = _sl, _sh                       # SBPORT: T1's band, static
                    _SBPORT_LIVE[0] = True
                else:
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

    # ------------------------------------------------------------------ STRPAR fast path
    def _strpar_eligible(self) -> bool:
        """True iff the live config is EXACTLY the one `_body_strpar` reproduces.

        This is a refusal gate, not an optimisation gate: every clause names a term of `_body`
        that `_body_strpar` does NOT implement.  Adding an online term to a future card without
        adding it here makes this return False and the card ships sequential -- slower, never
        wrong.  The bit-exactness gate in `_capture` is the second line of defence.

        T2BIASPAR (v53): the `bias_beta` / `sp_beta` clauses are GONE because `_body_strpar` now
        implements both terms (prefix update + main-stream application, `_body` verbatim).  The
        `gam_beta` / `gamf_beta` clauses stay: those terms are still not ported.

        T2IMG (2026-08-28) adds the one clause that is NOT about the config: the measured torch
        allowlist.  It is FIRST here and it is also checked in `__init__` before the streams are
        constructed, because every other guard in this file is a post-hoc detector -- it reports
        on a multi-stream body that has already executed and a multi-stream graph that has
        already been captured.  Only a clause consulted before those operations can refuse them."""
        o = self.o
        return bool(
            _STRPAR_TVOK[0]                                             # T2IMG: measured torch
            and self._side is not None and self._side2 is not None
            and hasattr(self.base, "m1") and hasattr(self.base, "m2")   # a 2-member blend
            and self.head is not None                                   # vgain head on
            and o.get("gam_beta", 0.0) == 0.0 and self._gamf_m is None  # no gamma / no field gamma
            and o.get("band_beta", 0.0) > 0.0 and self.mad_ref is not None
            and self.LOf is not None and self.HIf is not None
            and self.band_lo is not None and self.band_hi is not None
            and self._sb is not None                                    # SBPORT band
        )

    def _band_branch(self, n_t: int, C: int, usable, wf, x_ph=None):
        """The prediction-INDEPENDENT half of the interval.

        Reads only `self._sx` and the EMA state updated in step 1 of the body -- never `p`.  That
        is the whole reason it can be forked: only `lo = _ctr + _ol` needs the forecast.  Lifted
        verbatim from `_body`'s step 5 so the two cannot drift.

        `x_ph` (PAR4) is passed straight through to `_sb_offsets`; None reproduces par3 exactly."""
        o = self.o
        cb = usable / (1.0 - self._pw_band).clamp_min(1e-12)
        s = ((self._mad_m * cb) / self.mad_ref)[:, self.ch_idx]      # (n_t, C)
        s = (1.0 + wf * (s - 1.0)).clamp(o["s_min"], o["s_max"])
        j = ((s - _S0) * _SSC).round().long().clamp_(0, _NF - 1)
        gl = torch.gather(self.LOf, 0, j.unsqueeze(0)).reshape(1, n_t, 1, 1, C)
        gh = torch.gather(self.HIf, 0, j.unsqueeze(0)).reshape(1, n_t, 1, 1, C)
        _w = usable
        if _SGATE > 0.0:
            _w = _w * ((s - _SGATE) / _SRAMP).clamp(0.0, 1.0).reshape(1, n_t, 1, 1, C)
        _ol = self.band_lo + _w * (gl - self.band_lo)
        _oh = self.band_hi + _w * (gh - self.band_hi)
        _sl, _sh = self._sb_offsets(self._sx, C, x_ph)
        _dl = torch.where(self.band_lo != 0, self.band_lo, torch.ones_like(self.band_lo))
        _dh = torch.where(self.band_hi != 0, self.band_hi, torch.ones_like(self.band_hi))
        return _sl * (_ol / _dl), _sh * (_oh / _dh)

    def _body_strpar(self, mode: str = "par3") -> None:
        """`_body` for the live config, with m2 (and, from par3, the band branch; from par4, the
        whole prediction-independent prefix) forked onto side streams inside the capture.
        Bit-exact: the fork changes only WHEN kernels run, never what they compute.  Proven, not
        asserted -- see `_capture`.

        par4's DAG and the two hazards it has to respect are written out in the file header; the
        short form is that main waits on e4 before it touches `_sg_pg` and on e3 before it touches
        `_sg_pfin`, which is what keeps side2's reads of both ordered ahead of main's writes."""
        o = self.o
        have = self._have
        _par4 = (mode == "par4")
        with torch.no_grad():
            cur = torch.cuda.current_stream()
            n_t, C = self.band_lo.shape[1], self.band_lo.shape[-1]

            if not _par4:
                # ---------- 1. gradient-free online update ----------
                # T2BIASPAR (v53): `r` is no longer dead.  The bias / sp EMA updates are `_body`'s
                # step-1 statements verbatim, in `_body`'s order, ahead of the band EMA.  They read
                # only `_sy` and LAST step's frozen `_sg_pg`, so they are prediction-independent and
                # belong in this prefix.  With both knobs at zero the two `if`s are skipped and this
                # block is statement-for-statement v51's.
                r = self._sy - self._sg_pg
                if self._fld_m is not None:                 # BIASFIELD (v55): first in the prefix
                    a = have * o["fld_beta"]
                    for _i, (_h, _w) in enumerate(self._fld_levels):
                        rf = _fld_down(r, _h, _w)
                        if self._fld_pooled:
                            rf = rf.mean(dim=1, keepdim=True)
                        if self._chmask is not None:
                            rf = rf * self._chmask
                        self._fld_m[_i].mul_(1.0 - a).add_(rf * a)
                        self._fld_m2[_i].mul_(1.0 - a).add_(rf * rf * a)
                        r = r - _fld_up(rf, self._H, self._W)
                    self._pw_fld.mul_(1.0 - a)
                if o["bias_beta"] > 0.0:
                    a = have * o["bias_beta"]
                    rm = r.mean(dim=(0, 2, 3))            # (n_t, C)
                    if self._chmask is not None:
                        rm = rm * self._chmask.reshape(1, -1)
                    self._bias_m.mul_(1.0 - a).add_(rm * a)
                    if self._bias_m2 is not None:
                        self._bias_m2.mul_(1.0 - a).add_(rm * rm * a)
                    self._pw_bias.mul_(1.0 - a)
                if self._sp_m is not None:
                    a = have * o["sp_beta"]
                    rs = r - r.mean(dim=(0, 2, 3), keepdim=True)   # orthogonal to the bias term
                    if self._chmask is not None:
                        rs = rs * self._chmask
                    self._sp_m.mul_(1.0 - a).add_(rs * a)
                    if self._sp_m2 is not None:
                        self._sp_m2.mul_(1.0 - a).add_(rs * rs * a)
                    self._pw_sp.mul_(1.0 - a)
                a = have * o["band_beta"]
                k = self.cm
                md = (self._sy[..., :k] - self._sg_pfin[..., :k]).abs().mean(dim=(0, 2, 3))
                self._mad_m.mul_(1.0 - a).add_(md * a)
                self._pw_band.mul_(1.0 - a)
                self._k_dev.add_(have)

                # `usable`/`wf` read only `_k_dev`, updated above -> hoisted ABOVE the forward so
                # the band branch has everything it needs and can start immediately.
                usable = self._k_dev.clamp(max=1.0)
                wf = (self._k_dev / float(max(1, int(o["warm"])))).clamp(max=1.0)
                if self._gate_on:
                    self._gate_corr(usable, wf)          # BIASGATE (v54, fused)

            # ---------- 2. fork ----------
            # In par4 the fork is the FIRST thing in the step: nothing prediction-independent is
            # left ahead of it, so m1 owns the main stream from its very first kernel and both
            # side streams start at the same instant.
            ef = torch.cuda.Event(); ef.record(cur)
            self._side.wait_event(ef)
            with torch.cuda.stream(self._side):
                y2 = self.base.m2(self._sx)
                e2 = torch.cuda.Event(); e2.record(self._side)
            e3 = e4 = None
            if _par4:
                self._side2.wait_event(ef)
                with torch.cuda.stream(self._side2):
                    # ---- 1'. the prediction-independent prefix, moved off the critical path ----
                    # Identical statements in identical order to the par3 block above; only the
                    # stream they are enqueued on differs, which is why this is bit-exact.
                    r = self._sy - self._sg_pg
                    if self._fld_m is not None:                 # BIASFIELD (v55): first in the prefix
                        a = have * o["fld_beta"]
                        for _i, (_h, _w) in enumerate(self._fld_levels):
                            rf = _fld_down(r, _h, _w)
                            if self._fld_pooled:
                                rf = rf.mean(dim=1, keepdim=True)
                            if self._chmask is not None:
                                rf = rf * self._chmask
                            self._fld_m[_i].mul_(1.0 - a).add_(rf * a)
                            self._fld_m2[_i].mul_(1.0 - a).add_(rf * rf * a)
                            r = r - _fld_up(rf, self._H, self._W)
                        self._pw_fld.mul_(1.0 - a)
                    if o["bias_beta"] > 0.0:
                        a = have * o["bias_beta"]
                        rm = r.mean(dim=(0, 2, 3))            # (n_t, C)
                        if self._chmask is not None:
                            rm = rm * self._chmask.reshape(1, -1)
                        self._bias_m.mul_(1.0 - a).add_(rm * a)
                        if self._bias_m2 is not None:
                            self._bias_m2.mul_(1.0 - a).add_(rm * rm * a)
                        self._pw_bias.mul_(1.0 - a)
                    if self._sp_m is not None:
                        a = have * o["sp_beta"]
                        rs = r - r.mean(dim=(0, 2, 3), keepdim=True)   # orthogonal to the bias term
                        if self._chmask is not None:
                            rs = rs * self._chmask
                        self._sp_m.mul_(1.0 - a).add_(rs * a)
                        if self._sp_m2 is not None:
                            self._sp_m2.mul_(1.0 - a).add_(rs * rs * a)
                        self._pw_sp.mul_(1.0 - a)
                    a = have * o["band_beta"]
                    k = self.cm
                    md = (self._sy[..., :k] - self._sg_pfin[..., :k]).abs().mean(dim=(0, 2, 3))
                    self._mad_m.mul_(1.0 - a).add_(md * a)
                    self._pw_band.mul_(1.0 - a)
                    self._k_dev.add_(have)
                    usable = self._k_dev.clamp(max=1.0)
                    wf = (self._k_dev / float(max(1, int(o["warm"])))).clamp(max=1.0)
                    if self._gate_on:
                        self._gate_corr(usable, wf)          # BIASGATE (v54, fused)

                    # ---- the INPUT-ONLY half of the vgain features ----
                    # `x_ph` is denormalised once here and serves both this block and the sband
                    # head inside `_band_branch`; par3 computed it twice, once on each stream.
                    # The expressions below are lifted verbatim from `_vg_features` so the two
                    # cannot drift, and the results land in STATIC buffers because they cross a
                    # stream boundary.
                    x_ph = self._sx * self._nstd + self._nmean
                    i2 = x_ph[..., :2]
                    ib = i2.mean(1)
                    KEi = 0.5 * ((i2 - ib.unsqueeze(1)) ** 2).mean(1).sum(-1)
                    self._vg_li.copy_(torch.log(KEi + 1e-12))
                    self._vg_ib.copy_(ib)
                    # e4 publishes (`li`, `ib`) AND -- because it is recorded after them -- the
                    # reads of `_sg_pg` (in `r`) and `_sg_pfin` (in the EMA) that precede them.
                    e4 = torch.cuda.Event(); e4.record(self._side2)

                    # ---- the band branch ----
                    band_ol, band_oh = self._band_branch(n_t, C, usable, wf, x_ph)
                    e3 = torch.cuda.Event(); e3.record(self._side2)
            elif mode == "par3":
                self._side2.wait_event(ef)
                with torch.cuda.stream(self._side2):
                    band_ol, band_oh = self._band_branch(n_t, C, usable, wf)
                    e3 = torch.cuda.Event(); e3.record(self._side2)
            y1 = self.base.m1(self._sx)
            cur.wait_event(e2)
            p = 0.5 * (y1 + y2)
            _pre = p

            # ---------- 3. the gain field ----------
            if _par4:
                # Same seven fields `_vg_features` builds, with the four input-only ones taken
                # from the side stream instead of recomputed.  `li` and `ib` are bitwise the
                # values `_vg_features` would have produced -- same kernels, same inputs -- and
                # gate 1 in `_capture` proves that rather than trusting it.
                cur.wait_event(e4)
                _p_ph = p * self._nstd + self._nmean
                _p2 = _p_ph[..., :2]
                _pb = _p2.mean(1)
                _pf = _p2 - _pb.unsqueeze(1)
                _KEp = 0.5 * (_pf * _pf).mean(1).sum(-1)
                _lp = torch.log(_KEp + 1e-12)
                _li, _ib = self._vg_li, self._vg_ib
                gh_ = self.head(torch.stack(
                    [_lp, _li, _li - _lp, _pb[..., 0], _pb[..., 1], _ib[..., 0], _ib[..., 1]],
                    dim=1))
            else:
                gh_ = self.head(_vg_features(p * self._nstd + self._nmean,
                                             self._sx * self._nstd + self._nmean))
            pm = p.mean(dim=1, keepdim=True)
            p = pm + (p - pm) * gh_[:, None, :, :, None]
            # Safe in par4: the e4 wait above orders this write after side2's read of `_sg_pg`.
            self._sg_pg.copy_(p)

            # ---------- 4. online corrections (T2BIASPAR, v53) ----------
            # `_body`'s step-4 bias / sp application verbatim.  `usable` / `wf` are the tensors the
            # prefix built (main in par3; side2 in par4, published by e4, which main has already
            # waited on above -- the same edge that orders every side2 write to `_bias_m`, `_sp_m`,
            # `_pw_bias`, `_pw_sp` ahead of these reads).  `_pre` is untouched, so the interval
            # stays centred on the pre-head field exactly as in `_body`.  Both `if`s keep `_body`'s
            # shape, `_lim_*` included -- see THE FREEZE TRAP in the header.
            if self._gate_on:
                # BIASGATE (v54, fused): the gated b + s term was built in the prefix from prefix
                # state only (side2 in par4, published by e4 -- main has already waited on it); ONE
                # add on the main stream replaces the two applications below.
                p = p + self._corr
                _BIASPAR_LIVE[0] = self._bias_m2 is not None    # bias term rides in _corr
                _BIASPAR_LIVE[1] = self._sp_m2 is not None      # sp term rides in _corr
                _BIASGATE_LIVE[0] = True
            if (not self._gate_on) and o["bias_beta"] > 0.0 and self._lim_bias is not None:
                cb = usable / (1.0 - self._pw_bias).clamp_min(1e-12)
                b = (self._bias_m * cb).clamp(-self._lim_bias, self._lim_bias) \
                    * (o["bias_scale"] * wf)
                p = p + b.reshape(1, n_t, 1, 1, C)
                _BIASPAR_LIVE[0] = True

            if (not self._gate_on) and self._sp_m is not None and self._lim_sp is not None:
                cb = usable / (1.0 - self._pw_sp).clamp_min(1e-12)
                p = p + (self._sp_m * cb).clamp(-self._lim_sp, self._lim_sp) \
                    * (o["sp_scale"] * wf)
                _BIASPAR_LIVE[1] = True

            # ---------- 5. interval ----------
            _ctr = p if _BCDT2 == 0.0 else (p + _BCDT2 * (_pre - p))
            if _BCDT2 != 0.0:
                _BCDT2_LIVE[0] = True
            if e3 is not None:
                cur.wait_event(e3)
                _ol, _oh = band_ol, band_oh
            else:
                _ol, _oh = self._band_branch(n_t, C, usable, wf)
            _SBPORT_LIVE[0] = True
            lo, hi = _ctr + _ol, _ctr + _oh

            # ---------- 6. hard safety ----------
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
        eager = drive(lambda: self._body())          # the SEQUENTIAL reference, unchanged

        # ==========================================================================================
        # CAPTURE ORDER IS A CORRECTNESS REQUIREMENT ON torch 2.2.2.  READ THIS BEFORE EDITING.
        # ==========================================================================================
        # This card survives multi-stream capture on the production torch for THREE STRUCTURAL
        # REASONS.  They are not stylistic and they are not luck.  Break any one and the card
        # corrupts SILENTLY -- which is strictly worse than the Track-1 failure it comes from,
        # because that one at least aborted loudly at import.
        #
        #   (1) THE MULTI-STREAM CAPTURE IS THE LAST CAPTURE.  Nothing captures after it.
        #   (2) THIS FILE CONTAINS NO `torch.cuda.empty_cache()`.  Not one call.
        #   (3) `_side` / `_side2` ARE TOUCHED ONLY INSIDE `_body_strpar`.
        #
        # THE MECHANISM, measured rather than guessed (2026-08-28), and it is none of the things
        # that were suspected -- not dtype, not shape variety, not a copy stream or executor, not
        # stream lifetime, and not the number of private pools (a SHARED pool was tried and does
        # NOT fix it):
        #
        #   torch 2.2.2 binds a graph's private memory pool to the CAPTURE STREAM ONLY -- its one
        #   binding API is `_cuda_beginAllocateCurrentStreamToPool` (2.8 adds the stream-agnostic
        #   `_cuda_beginAllocateToPool`).  So allocations made on a SIDE stream during capture
        #   ESCAPE the private pool into the ordinary caching allocator.  `torch.cuda.graph
        #   .__enter__` itself calls `empty_cache()`.  A LATER capture therefore `cudaFree`s
        #   blocks the earlier graph still holds addresses into; the next allocation claims that
        #   virtual address; and the earlier graph now replays onto memory that belongs to someone
        #   else.  The causal proof: suppressing `empty_cache()` across the capture loop PASSES,
        #   and re-introducing a single later `empty_cache()` FAILS.
        #
        #   Track 1 dies of exactly this -- it captures 11 bucket graphs in a loop, so ten of them
        #   land after the first multi-stream capture.
        #
        # WHAT THAT MEANS FOR ANYONE EDITING THIS FILE.  Adding a second graph (a bucket loop, a
        # warm-up graph, a per-shape graph), or calling `empty_cache()` anywhere after the capture,
        # or using `_side`/`_side2` outside `_body_strpar`, re-creates the fatal ordering.  IF A
        # DESIGN NEEDS A CAPTURE AFTER THE MULTI-STREAM ONE, THAT DESIGN IS KNOWN-FATAL ON THE
        # PRODUCTION TORCH -- stop, do not work around it here.
        #
        # PAR4 MAKES (1) MORE DELICATE, NOT LESS: it adds a fourth stream, so more allocation
        # escapes the private pool and there is more to corrupt.  The ladder below can legitimately
        # capture more than once (par4 -> par3 -> sequential), but only ever KEEPS the last one it
        # captured, and every earlier attempt is discarded unreplayed.  `_CAP_LOG` records the
        # order so `build_t2_bundle.py` can REFUSE a bundle in which the graph that actually ships
        # is not the last capture -- a comment protects the next human, an assertion protects the
        # next agent.
        # ==========================================================================================
        def _capture_body(fn, tag):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            # The single capture site in this file, which is what makes `_CAP_LOG` complete.
            _CAP_LOG.append(tag)
            with torch.cuda.graph(g):
                fn()
            torch.cuda.synchronize()
            return g

        # ---- STRPAR gates 1 and 2, walked down a LADDER: par4 -> par3 -> sequential.
        # Gate 1 is `the multi-stream body equals the sequential one bit-for-bit, run eagerly`.
        # `drive` includes a first step (have=0), several adapting steps and a reset, so a fork
        # that raced on the EMA state or on _sg_pg would show up here as a single unequal element.
        # Gate 2 is `the CAPTURED graph replays bit-equal to that same sequential reference`.
        # A rung must clear BOTH to ship; otherwise the next rung is tried.
        #
        # Why a ladder and not v50's single par3-or-sequential choice: par3 is the body already
        # scored online, so if par4 fails a gate in an environment we have not measured there is
        # no reason to give up par3's banked gain as well.  Each rung is attempted lazily, so on
        # the normal path par3 is never even run.
        gr = None
        if _STRPAR and self._strpar_eligible():
            for _m in ("par4", "par3"):
                try:
                    par = drive(lambda m=_m: self._body_strpar(m))
                    torch.cuda.synchronize()
                    if not (len(par) == len(eager) and all(
                            torch.equal(a, b) for a, b in zip(eager, par))):
                        self.reset_ttt_state()
                        continue
                    g_ = _capture_body(lambda m=_m: self._body_strpar(m), ("multistream", _m))
                    replay = drive(g_.replay)
                    torch.cuda.synchronize()
                    if len(eager) == len(replay) and all(
                            torch.equal(a, b) for a, b in zip(eager, replay)):
                        gr = g_
                        _STRPAR_MODE[0] = _m
                        break
                except Exception:
                    pass
                self.reset_ttt_state()
            if gr is None:
                _STRPAR_MODE[0] = ""
                _STRPAR_REFUSED[0] = True
        elif _STRPAR:
            _STRPAR_REFUSED[0] = True

        # The fallback that makes this arm downside-free: with no multi-stream rung shipping, we
        # capture `_body`, so the worst case is exactly the card that is on the board today --
        # never the (much slower) eager path.
        if gr is None:
            gr = _capture_body(self._body, ("sequential", ""))
            replay = drive(gr.replay)
            torch.cuda.synchronize()
            if not (len(eager) == len(replay) and all(
                    torch.equal(a, b) for a, b in zip(eager, replay))):
                gr = None

        self._graph = gr
        self._graph_checked = gr is not None
        if gr is not None and _STRPAR_MODE[0]:
            _STRPAR_LIVE[0] = True

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
                    if self._fld_m is not None:                 # BIASFIELD (v55) eager
                        for _i, (_h, _w) in enumerate(self._fld_levels):
                            rf = _fld_down(r, _h, _w)
                            if self._fld_pooled:
                                rf = rf.mean(dim=1, keepdim=True)
                            if self._chmask is not None:
                                rf = rf * self._chmask
                            self._fld_m[_i].mul_(1.0 - o["fld_beta"]).add_(rf, alpha=o["fld_beta"])
                            self._fld_m2[_i].mul_(1.0 - o["fld_beta"]).add_(rf * rf, alpha=o["fld_beta"])
                            r = r - _fld_up(rf, self._H, self._W)
                        self._pw_fld.mul_(1.0 - o["fld_beta"])
                    if o["bias_beta"] > 0.0:
                        rm = r.mean(dim=(0, 2, 3))
                        if self._chmask is not None:
                            rm = rm * self._chmask.reshape(1, -1)
                        self._bias_m.mul_(1.0 - o["bias_beta"]).add_(rm, alpha=o["bias_beta"])
                        if self._bias_m2 is not None:
                            self._bias_m2.mul_(1.0 - o["bias_beta"]).add_(rm * rm, alpha=o["bias_beta"])
                    if self._sp_m is not None:
                        rs = r - r.mean(dim=(0, 2, 3), keepdim=True)
                        if self._chmask is not None:
                            rs = rs * self._chmask
                        self._sp_m.mul_(1.0 - o["sp_beta"]).add_(rs, alpha=o["sp_beta"])
                        if self._sp_m2 is not None:
                            self._sp_m2.mul_(1.0 - o["sp_beta"]).add_(rs * rs, alpha=o["sp_beta"])
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
                bm = self._bias_m * cb
                if self._bias_m2 is not None:      # BIASGATE (v54)
                    bm = bm * self._gate_factor(bm, self._bias_m2 * cb,
                                                (1.0 - o["bias_beta"]) ** k, o["bias_beta"])
                b = bm.clamp(-self._lim_bias, self._lim_bias) \
                    * (o["bias_scale"] * wf)
                p = p + b.reshape(1, n_t, 1, 1, C)

            if usable and self._sp_m is not None and self._lim_sp is not None:
                cb = 1.0 / (1.0 - (1.0 - o["sp_beta"]) ** k)
                sm = self._sp_m * cb
                if self._sp_m2 is not None:        # BIASGATE (v54)
                    sm = sm * self._gate_factor(sm, self._sp_m2 * cb,
                                                (1.0 - o["sp_beta"]) ** k, o["sp_beta"])
                p = p + sm.clamp(-self._lim_sp, self._lim_sp) \
                    * (o["sp_scale"] * wf)

            if usable and self._fld_m is not None and self._lim_fld is not None:   # BIASFIELD (v55)
                cb = 1.0 / (1.0 - (1.0 - o["fld_beta"]) ** k)
                p = p + self._fld_apply(cb, (1.0 - o["fld_beta"]) ** k, float(wf))

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
                _sl, _sh = self._sb_offsets(x, C) if self._sb is not None else (None, None)
                if _sl is not None and _sl.shape[1] == n_t:
                    # SBPORT: T1's absolute per-element offsets x the RELATIVE widening the
                    # gate applied to the frozen table (== 1 exactly while the gate is shut)
                    _dl = torch.where(self.band_lo != 0, self.band_lo, torch.ones_like(self.band_lo))
                    _dh = torch.where(self.band_hi != 0, self.band_hi, torch.ones_like(self.band_hi))
                    _ol, _oh = _sl * (_ol / _dl), _sh * (_oh / _dh)
                    _SBPORT_LIVE[0] = True
                else:
                    _rl, _rh = self._pe_shape(x, C)
                    if _rl is not None and _rl.shape[1] == n_t:
                        _ol, _oh = _ol * _rl, _oh * _rh
                lo, hi = _ctr + _ol, _ctr + _oh
            elif (self.band_lo is not None and p.shape[1] == self.band_lo.shape[1]
                    and p.shape[-1] == self.band_lo.shape[-1]):
                _ol, _oh = self.band_lo, self.band_hi
                _sl, _sh = self._sb_offsets(x, C) if self._sb is not None else (None, None)
                if _sl is not None and _sl.shape[1] == n_t:
                    _ol, _oh = _sl, _sh                       # SBPORT: T1's band, static
                    _SBPORT_LIVE[0] = True
                else:
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
