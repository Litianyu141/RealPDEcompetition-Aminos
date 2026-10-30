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
  (measured good online AND locally), 2.8.0 (measured good at build time on dgx-V100-2)};
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


# ------------------------------------------------------------------------------------------------------------------
# fused-TTT card patch (common/fused_ttt_card.py, arm T2FUSE-J): jiterator (NVRTC) kernels for the KE-FTL / band-FTL
# elementwise chains.  Pure functions with verbatim aten references; per-piece load-time guard in `_fj_run`.
_FJ_TOL = 1e-4
_FJ_PARTS = ('kf_terms', 'bf_terms', 'tail_anchor', 'tail_band')
_FJ_STATE = {"on": [], "off": [], "built": None}
_FJ = {}
_FJ_SRC = {
    "kf_r5": ("template <typename T> T kf_r5(T pp, T pm, T yy, T ym, T s2, T sd, T mn, T sel) {"
              " T pf = pp - pm; T tf = yy - ym;"
              " if (sel < T(0.5)) return pf * pf * s2;"
              " if (sel < T(1.5)) return pf * tf * s2;"
              " if (sel < T(2.5)) return tf * tf * s2;"
              " if (sel < T(3.5)) { T d = pm - ym; return d * d * s2; }"
              " T t = yy * sd + mn; return t * t; }", 1),
    "kf_q5": ("template <typename T> T kf_q5(T A, T B, T Cc, T Mm, T Tm, T sel, T inv2t) {"
              " T kt = Cc * inv2t; T kp = A * inv2t;"
              " if (sel < T(0.5)) return A - T(2) * B + Cc;"
              " if (sel < T(1.5)) return kt * kt;"
              " if (sel < T(2.5)) { T d = kp - kt; return d * d; }"
              " if (sel < T(3.5)) return Mm;"
              " return Tm; }", 1),
    "kf_scal": ("template <typename T> void kf_scal(T s1, T s2, T s3, T M, T tn2, T wr, T wt, T& a1, T& a2) {"
                " T sse = M + s1; sse = sse < T(1e-30) ? T(1e-30) : sse;"
                " T tn = tn2 < T(1e-30) ? T(1e-30) : tn2;"
                " T rel0 = sqrt(sse) / sqrt(tn); rel0 = rel0 < T(1e-12) ? T(1e-12) : rel0;"
                " T ktn = s2 < T(1e-16) ? T(1e-16) : s2;"
                " T tke0 = sqrt(s3) / sqrt(ktn); tke0 = tke0 < T(1e-8) ? T(1e-8) : tke0;"
                " a1 = wr / (T(2) * rel0 * tn); a2 = wt / (T(2) * tke0 * ktn); }", 2),
    "kf_arm": ("template <typename T> T kf_arm(T A, T B, T Cc, T kin, T mu, T cg, T sel, T inv2t) {"
               " T kp = A * inv2t; T kt = Cc * inv2t;"
               " T rr = kp + mu * (kin - kp); rr = rr < T(1e-12) ? T(1e-12) : rr;"
               " T kc = kp < T(1e-12) ? T(1e-12) : kp;"
               " T q = sqrt(rr / kc); q = q < T(0.25) ? T(0.25) : (q > T(4) ? T(4) : q);"
               " T g = cg * q;"
               " if (sel < T(0.5)) return g * g * A - T(2) * g * B + Cc;"
               " T t = g * g * kp - kt; return t * t; }", 1),
    "kf_u": ("template <typename T> T kf_u(T eb, T tb, T a1, T a2) {"
             " T v = -(a1 * eb) - a2 * tb; return (v - v == T(0)) ? v : T(0); }", 1),
    "kf_em": ("template <typename T> T kf_em(T pmu, T pmv, T ymu, T ymv, T xbu, T xbv, T lam, T sdu, T sdv) {"
              " T du = (pmu - ymu) * sdu + lam * ((xbu - pmu) * sdu);"
              " T dv = (pmv - ymv) * sdv + lam * ((xbv - pmv) * sdv);"
              " return du * du + dv * dv; }", 1),
    "kf_um": ("template <typename T> T kf_um(T emb, T a1, T nt) {"
              " T v = -(nt * a1) * emb; return (v - v == T(0)) ? v : T(0); }", 1),
    "bf_s": ("template <typename T> T bf_s(T yy, T mid, T hw, T tsd, T tmn, T sd, T m, T isig) {"
             " T hc = hw < T(1e-12) ? T(1e-12) : hw;"
             " T r = fabs(yy - mid) / hc;"
             " r = (fabs(yy * tsd + tmn) > T(1e-6)) ? r : T(1e9);"
             " T w = isig * (hw * sd);"
             " return (r <= m) ? exp(-m * w) : T(0); }", 1),
    "ka_g": ("template <typename T> T ka_g(T vu, T vv, T mu, T kin, T cc, T s2hu, T s2hv) {"
             " T kp = vu * s2hu + vv * s2hv;"
             " T rr = kp + mu * (kin - kp); rr = rr < T(1e-12) ? T(1e-12) : rr;"
             " T kc = kp < T(1e-12) ? T(1e-12) : kp;"
             " T q = sqrt(rr / kc); q = q < T(0.25) ? T(0.25) : (q > T(4) ? T(4) : q);"
             " return cc * q; }", 1),
    "ka_p": ("template <typename T> T ka_p(T p, T m, T g, T lam, T xb) {"
             " T m2 = m + lam * (xb - m); return m2 + g * (p - m); }", 1),
    "ka_sh": ("template <typename T> T ka_sh(T m, T lam, T xb) { return (m + lam * (xb - m)) - m; }", 1),
    "bf_ch": ("template <typename T> void bf_ch(T lo, T hi, T sh, T& c, T& h) {"
              " c = T(0.5) * (lo + hi) + sh; h = T(0.5) * fabs(hi - lo); }", 2),
    "bf_bt": ("template <typename T> T bf_bt(T c, T bt, T pm, T cm) { return c + bt * (pm - cm); }", 1),
    "bf_lh": ("template <typename T> void bf_lh(T c, T m, T h, T& lo, T& hi) {"
              " T d = m * h; lo = c - d; hi = c + d; }", 2),
}
_FJ_KW = {"kf_q5": dict(inv2t=0.025), "kf_scal": dict(wr=1.0, wt=1.0), "kf_arm": dict(inv2t=0.025),
          "kf_em": dict(sdu=1.0, sdv=1.0), "kf_um": dict(nt=20.0), "bf_s": dict(isig=1.0),
          "ka_g": dict(s2hu=1.0, s2hv=1.0)}


def _fj_kernels():
    """Create the jiterator functions once (compiled lazily on their first, eager, call); {} if unavailable."""
    if _FJ_STATE["built"] is None:
        try:
            J = torch.cuda.jiterator
            for k, (src, nout) in _FJ_SRC.items():
                kw = _FJ_KW.get(k, {})
                _FJ[k] = (J._create_jit_fn(src, **kw) if nout == 1
                          else J._create_multi_output_jit_fn(src, num_outputs=nout, **kw))
            _FJ_STATE["built"] = True
        except Exception as e:
            _FJ.clear()
            _FJ_STATE["built"] = False
            _FJ_STATE["off"].append(("build", "%s: %s" % (type(e).__name__, str(e)[:200])))
    return _FJ if _FJ_STATE["built"] else {}


def _fj_synth(a, g):
    """A synthetic stand-in with the same shape AND strides (so the same kernel variants compile), values in [0.1, 1)."""
    if not isinstance(a, torch.Tensor):
        return a
    t = torch.empty_strided(a.size(), a.stride(), dtype=a.dtype, device=a.device)
    t.copy_(torch.rand(a.size(), generator=g, device=a.device, dtype=a.dtype) * 0.9 + 0.1)
    return t


def _fj_cmp(fa, fb):
    worst = 0.0
    for a, b in zip(fa, fb):
        if a is None and b is None:
            continue
        if a.shape != b.shape or not torch.equal(torch.isfinite(a), torch.isfinite(b)):
            return float("inf")
        m = torch.isfinite(b)
        d = float(torch.where(m, (a - b).abs(), torch.zeros_like(b)).max())
        s = float(torch.where(m, b.abs(), torch.zeros_like(b)).max())
        worst = max(worst, d / max(s, 1e-30))
    return worst

# ------------------------------------------------------------------------------------------------------------------
# kernel-cut card patch (common/fused_kcut_card.py, arm T2KCUT): the serial TAIL after the par4 join, fused.
# Pure pieces with verbatim references; per-piece guard in `CalibratedTTTModel._kc_run`.
_KC_PARTS = ('blend', 'vgain', 'gctr', 'fin')
_KC_TOL = {"blend": (1e-5, 1e-5), "vgain": (1e-5, 2e-3), "gctr": (1e-5, 1e-5), "fin": (1e-5, 1e-5)}   # (TF32 off, ambient)
_KC_STATE = {"on": [], "off": [], "built": None}
_KC = {}
_KC_SRC = {
    "kc_blend": ("template <typename T> T kc_blend(T a, T b) { return T(0.5) * (a + b); }", 1),
    "kc_lp": ("template <typename T> T kc_lp(T vu, T vv, T s2hu, T s2hv) {"
              " return log(s2hu * vu + s2hv * vv + T(1e-12)); }", 1),
    "kc_vz": ("template <typename T> T kc_vz(T lp, T li, T mu, T mv, T ib, T sel, T fm, T fs, T sdu, T sdv, T mnu, T mnv) {"
              " T v;"
              " if (sel < T(0.5)) v = lp;"
              " else if (sel < T(1.5)) v = li;"
              " else if (sel < T(2.5)) v = li - lp;"
              " else if (sel < T(3.5)) v = mu * sdu + mnu;"
              " else if (sel < T(4.5)) v = mv * sdv + mnv;"
              " else v = ib;"
              " return (v - fm) / fs; }", 1),
    "kc_bgelu": ("template <typename T> T kc_bgelu(T x, T b) {"
                 " T z = x + b; return z * T(0.5) * (T(1) + erf(z * T(0.70710678118654752440))); }", 1),
    "kc_vout": ("template <typename T> T kc_vout(T o, T bo, T g, T bg, T lmin, T lmax) {"
                " T lg = (o + bo) + (g + bg);"
                " if (!(lg != lg)) { lg = lg < lmin ? lmin : lg; lg = lg > lmax ? lmax : lg; }"
                " return exp(lg); }", 1),
    "kc_gctr": ("template <typename T> void kc_gctr(T x, T pm, T g, T cr, T ol, T oh, T bcd, T& pg, T& pc, T& lo, T& hi) {"
                " pg = pm + __fmul_rn(x - pm, g);"
                " pc = pg + cr;"
                " T ctr = pc + __fmul_rn(bcd, x - pc);"
                " ctr = ctr + __fmul_rn(T(0.5), pc - pg);"
                " lo = ctr + ol; hi = ctr + oh; }", 4),
    "kc_kag": ("template <typename T> T kc_kag(T vu, T vv, T mu, T kin, T cc, T s2hu, T s2hv) {"
               " T kp = vu * s2hu + vv * s2hv;"
               " T rr = kp + mu * (kin - kp); rr = rr < T(1e-12) ? T(1e-12) : rr;"
               " T kc = kp < T(1e-12) ? T(1e-12) : kp;"
               " T q = sqrt(rr / kc); q = q < T(0.25) ? T(0.25) : (q > T(4) ? T(4) : q);"
               " return cc * q; }", 1),
    "kc_kap3": ("template <typename T> T kc_kap3(T p, T m, T g, T lam, T xb, T pas) {"
                " if (pas > T(0.5)) return p;"
                " T m2 = m + lam * (xb - m); return m2 + g * (p - m); }", 1),
    "kc_bfch3": ("template <typename T> void kc_bfch3(T lo, T hi, T m, T lam, T xb, T& c, T& h) {"
                 " T sh = (m + lam * (xb - m)) - m;"
                 " c = T(0.5) * (lo + hi) + sh; h = T(0.5) * fabs(hi - lo); }", 2),
    "kc_bfbt3": ("template <typename T> T kc_bfbt3(T c, T bt, T pm, T cm) { return c + bt * (pm - cm); }", 1),
    "kc_fin3": ("template <typename T> T kc_fin3(T sk, T sc, T p, T lo, T hi, T c, T h, T m) {"
                " if (sk < T(0.5)) return (p - p == T(0)) ? p : T(0);"
                " T l, u;"
                " if (sc < T(1.5)) { T d = m * h; l = c - d; u = c + d; } else { l = lo; u = hi; }"
                " l = (l - l == T(0)) ? l : T(-1000); u = (u - u == T(0)) ? u : T(1000);"
                " if (sk < T(1.5)) return (u < l) ? u : l;"
                " return (u < l) ? l : u; }", 1),
}
_KC_VIEW = {"fin": lambda r: (r[0], r[1][..., :2], r[2][..., :2])}   # what the guard compares
_KC_KW = {"kc_lp": dict(s2hu=1.0, s2hv=1.0), "kc_vz": dict(sdu=1.0, sdv=1.0, mnu=0.0, mnv=0.0),
          "kc_vout": dict(lmin=0.0, lmax=0.0), "kc_gctr": dict(bcd=1.0), "kc_kag": dict(s2hu=1.0, s2hv=1.0)}


def _kc_kernels():
    """Create the jiterator functions once (compiled lazily on their first, eager, call); {} if unavailable."""
    if _KC_STATE["built"] is None:
        try:
            J = torch.cuda.jiterator
            for k, (src, nout) in _KC_SRC.items():
                kw = _KC_KW.get(k, {})
                _KC[k] = (J._create_jit_fn(src, **kw) if nout == 1
                          else J._create_multi_output_jit_fn(src, num_outputs=nout, **kw))
            _KC_STATE["built"] = True
        except Exception as e:
            _KC.clear()
            _KC_STATE["built"] = False
            _KC_STATE["off"].append(("build", "%s: %s" % (type(e).__name__, str(e)[:200])))
    return _KC if _KC_STATE["built"] else {}


def _kc_tup(r):
    return tuple(r) if isinstance(r, (tuple, list)) else (r,)


# ------------------------------------------------------------------------------------------------------------------
# kernel-cut card patch, round 2 (common/fused_kcut2_card.py, arm T2KCUT2): tail remainder + side2.
_KC2_PARTS = ('wr', 'gate', 'ema', 'clim', 'band', 'sbf')
# (TF32 off, ambient); default (1e-5, 1e-5).  The three pieces whose reference itself runs TF32 cuDNN convs get an ambient
# band of 5e-3: across the five folds the vgain head's ambient fused-vs-reference difference measured 4e-4 .. 2.0e-3 (TF32
# rounding of the same convs on inputs that differ in the last fp32 bit) while the TF32-off check stayed <= 3.7e-6, and at
# 2e-3 the round-1 guard switched a fold's tail back to the reference path.  The TF32-off (fp32 math) check stays at 1e-5.
_KC2_TOL = {"sbh": (1e-5, 5e-3), "vgcl": (1e-5, 5e-3)}
_KC_TOL["vgain"] = (1e-5, 5e-3)
_KC2_STATE = {"on": [], "off": [], "built": None}
_KC2 = {}
_KC2_SRC = {
    "kc2_gctr2": ("template <typename T> void kc2_gctr2(T ol, T oh, T cr, T sel, T x, T pm, T g, T bcd, T& j, T& lo, T& hi) {"
                  " T pg = pm + __fmul_rn(x - pm, g);"
                  " T pc = pg + cr;"
                  " T ctr = pc + __fmul_rn(bcd, x - pc);"
                  " ctr = ctr + __fmul_rn(T(0.5), pc - pg);"
                  " j = (sel < T(0.5)) ? pg : pc; lo = ctr + ol; hi = ctr + oh; }", 3),
    "kc2_fin5": ("template <typename T> T kc2_fin5(T sk, T sc, T p, T lo, T hi, T c, T h, T m) {"
                 " if (sk < T(0.5)) return (p - p == T(0)) ? p : T(0);"
                 " if (sk > T(3.5)) return h;"
                 " if (sk > T(2.5)) return c;"
                 " T l, u;"
                 " if (sc < T(1.5)) { T d = m * h; l = c - d; u = c + d; } else { l = lo; u = hi; }"
                 " l = (l - l == T(0)) ? l : T(-1000); u = (u - u == T(0)) ? u : T(1000);"
                 " if (sk < T(1.5)) return (u < l) ? u : l;"
                 " return (u < l) ? l : u; }", 1),
    "kc2_gate": ("template <typename T> T kc2_gate(T m, T m2, T pw, T us, T wf, T lim, T k, T c, T eps, T sc) {"
                 " T den = T(1) - pw; den = (den < T(1e-12)) ? T(1e-12) : den;"
                 " T cb = us / den;"
                 " T bm = __fmul_rn(m, cb); T bm2 = __fmul_rn(m2, cb);"
                 " T vt = bm2 - __fmul_rn(bm, bm); vt = (vt < T(0)) ? T(0) : vt;"
                 " T phi = __fmul_rn(k, T(1) + pw) / den;"
                 " T v = __fmul_rn(vt, phi);"
                 " T f = T(1) - __fmul_rn(c, v) / (__fmul_rn(bm, bm) + eps);"
                 " f = (f < T(0)) ? T(0) : ((f > T(1)) ? T(1) : f);"
                 " T y = __fmul_rn(bm, f);"
                 " y = (y < -lim) ? -lim : ((y > lim) ? lim : y);"
                 " return __fmul_rn(y, __fmul_rn(sc, wf)); }", 1),
    "kc2_gsum": ("template <typename T> T kc2_gsum(T b, T t0, T t1, T t2) { return b + ((t0 + t1) + t2); }", 1),
    "kc2_ema": ("template <typename T> T kc2_ema(T sel, T m, T q, T rf, T mk, T hv, T beta) {"
                " T rfm = __fmul_rn(rf, mk);"
                " T a = __fmul_rn(hv, beta); T om = T(1) - a;"
                " if (sel < T(0.5)) return __fmul_rn(m, om) + __fmul_rn(rfm, a);"
                " if (sel < T(1.5)) return __fmul_rn(q, om) + __fmul_rn(__fmul_rn(rfm, rfm), a);"
                " return rfm; }", 1),
    "kc2_absd": ("template <typename T> T kc2_absd(T a, T b) { return fabs(a - b); }", 1),
    "kc2_sc": ("template <typename T> T kc2_sc(T sel, T s, T kd, T hv, T fb, T bb, T nb, T warm) {"
               " if (sel < T(0.5)) return __fmul_rn(s, T(1) - __fmul_rn(hv, fb));"
               " if (sel < T(1.5)) return __fmul_rn(s, T(1) - __fmul_rn(hv, bb));"
               " if (sel < T(2.5)) return __fmul_rn(s, T(1) - __fmul_rn(hv, nb));"
               " T k = kd + hv;"
               " if (sel < T(3.5)) return k;"
               " if (sel < T(4.5)) return (k > T(1)) ? T(1) : k;"
               " if (sel < T(5.5)) { T q = k / warm; return (q > T(1)) ? T(1) : q; }"
               " return s; }", 1),
    "kc2_clf": ("template <typename T> T kc2_clf(T n, T c0) { return ((n + c0) < T(0.5)) ? T(1) : T(0); }", 1),
    "kc2_clp": ("template <typename T> T kc2_clp(T n, T c0, T acc, T cur) {"
                " T s = n + c0; T d = (s < T(1)) ? T(1) : s; return (s > T(0.5)) ? acc / d : cur; }", 1),
    "kc2_bs": ("template <typename T> void kc2_bs(T mi, T us, T pw, T wf, T mr, T smin, T smax, T s0, T ssc, T nfm1, T sg, T sr,"
               " T& jf, T& w) {"
               " T den = T(1) - pw; den = (den < T(1e-12)) ? T(1e-12) : den;"
               " T cb = us / den;"
               " T s = __fmul_rn(mi, cb) / mr;"
               " s = T(1) + __fmul_rn(wf, s - T(1));"
               " s = (s < smin) ? smin : ((s > smax) ? smax : s);"
               " T q = nearbyint(__fmul_rn(s - s0, ssc));"
               " q = (q >= T(0)) ? q : T(0); q = (q > nfm1) ? nfm1 : q;"
               " jf = q;"
               " T gt = (s - sg) / sr; gt = (gt < T(0)) ? T(0) : ((gt > T(1)) ? T(1) : gt);"
               " w = __fmul_rn(us, gt); }", 2),
    "kc2_br": ("template <typename T> void kc2_br(T blo, T bhi, T gl, T gh, T w, T dl, T dh, T& rl, T& rh) {"
               " rl = (blo + __fmul_rn(w, gl - blo)) / dl; rh = (bhi + __fmul_rn(w, gh - bhi)) / dh; }", 2),
    "kc2_bb": ("template <typename T> void kc2_bb(T lu, T lv, T hu, T hv, T rl, T rh, T sc, T& ol, T& oh) {"
               " T l = (sc < T(0.5)) ? lu : ((sc < T(1.5)) ? lv : T(0));"
               " T h = (sc < T(0.5)) ? hu : ((sc < T(1.5)) ? hv : T(0));"
               " ol = l * rl; oh = h * rh; }", 2),
    "kc2_sp": ("template <typename T> T kc2_sp(T mu, T mv) {"
               " T s = __fmul_rn(mu, mu) + __fmul_rn(mv, mv); s = (s < T(0)) ? T(0) : s; return sqrt(s); }", 1),
    "kc2_g": ("template <typename T> T kc2_g(T r, T l, T dn, T up, T mx, T my) {"
              " T gx = (mx > T(0.5)) ? (r - l) : T(0); T gy = (my > T(0.5)) ? (dn - up) : T(0);"
              " T s = __fmul_rn(gx, gx) + __fmul_rn(gy, gy); s = (s < T(0)) ? T(0) : s; return sqrt(s); }", 1),
    "kc2_za": ("template <typename T> T kc2_za(T a, T b, T c, T d, T sel, T fm, T fs) {"
               " T v;"
               " if (sel < T(0.5)) v = log(a + T(1e-6));"
               " else if (sel < T(1.5)) v = log(b + T(1e-6));"
               " else if (sel < T(2.5)) v = c;"
               " else v = d;"
               " return (v - fm) / fs; }", 1),
    "kc2_zb": ("template <typename T> T kc2_zb(T du, T dv, T g, T su, T sv, T sel, T fm, T fs) {"
               " T v;"
               " if (sel < T(0.5)) { T q = (du < T(0)) ? T(0) : du; v = log(sqrt(q) + T(1e-6)); }"
               " else if (sel < T(1.5)) { T q = (dv < T(0)) ? T(0) : dv; v = log(sqrt(q) + T(1e-6)); }"
               " else if (sel < T(2.5)) v = log(g + T(1e-6));"
               " else v = ((su + sv) < T(1e-9)) ? T(1) : T(0);"
               " return (v - fm) / fs; }", 1),
    "kc2_zc": ("template <typename T> T kc2_zc(T lu, T lv, T mu, T mv, T wn, T sel, T fm, T fs) {"
               " T v;"
               " if (sel < T(0.5)) v = log(fabs(lu - mu) + T(1e-6));"
               " else if (sel < T(1.5)) v = log(fabs(lv - mv) + T(1e-6));"
               " else v = log(wn + T(1e-6));"
               " return (v - fm) / fs; }", 1),
    "kc2_gng": ("template <typename T> T kc2_gng(T x, T mean, T var, T w, T b, T eps) {"
                " T sc = w * rsqrt(var + eps); T z = x * sc + (b - mean * sc);"
                " return T(0.5) * z * (T(1) + erf(z * T(0.7071067811865476))); }", 1),
    "kc2_bcl": ("template <typename T> T kc2_bcl(T x, T b, T lo, T hi) {"
                " T y = x + b; if (!(y != y)) { y = (y < lo) ? lo : y; y = (y > hi) ? hi : y; } return y; }", 1),
}
_KC2_KW = {"kc2_gctr2": dict(bcd=1.0), "kc2_gate": dict(k=1.0, c=1.0, eps=1e-10, sc=1.0), "kc2_ema": dict(beta=0.0),
           "kc2_sc": dict(fb=0.0, bb=0.0, nb=0.0, warm=1.0),
           "kc2_bs": dict(smin=0.0, smax=1.0, s0=0.0, ssc=1.0, nfm1=1.0, sg=0.0, sr=1.0), "kc2_gng": dict(eps=1e-5),
           "kc2_bcl": dict(lo=-12.0, hi=2.0)}


def _kc2_kernels():
    if _KC2_STATE["built"] is None:
        try:
            J = torch.cuda.jiterator
            for k, (src, nout) in _KC2_SRC.items():
                kw = _KC2_KW.get(k, {})
                _KC2[k] = (J._create_jit_fn(src, **kw) if nout == 1
                           else J._create_multi_output_jit_fn(src, num_outputs=nout, **kw))
            _KC2_STATE["built"] = True
        except Exception as e:
            _KC2.clear()
            _KC2_STATE["built"] = False
            _KC2_STATE["off"].append(("build", "%s: %s" % (type(e).__name__, str(e)[:200])))
    return _KC2 if _KC2_STATE["built"] else {}


try:
    from ttt_model import TTTModel
except Exception:  # base class is optional; the evaluator duck-types this object
    TTTModel = nn.Module  # type: ignore

TTT_LR = 0.0
TTT_STEPS = 1      # inner SGD steps per revealed pair; 0 lr disables the branch entirely
HALFWIDTH = 0.15      # fallback constant normalized half-width, used iff calib.npz is unusable
_KF_GRID = (0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.2, 0.2, 0.2, 0.2, 0.2, 0.25, 0.25, 0.25, 0.25, 0.25, 0.3, 0.3, 0.3, 0.3, 0.3, 0.35, 0.35, 0.35, 0.35, 0.35, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0)   # common/ke_ftl_card_t2.py: KE-anchor strength of each (mu, c) arm
_KF_CG = (0.95, 1.0, 1.05, 1.1, 1.15, 0.95, 1.0, 1.05, 1.1, 1.15, 0.95, 1.0, 1.05, 1.1, 1.15, 0.95, 1.0, 1.05, 1.1, 1.15, 0.95, 1.0, 1.05, 1.1, 1.15, 0.95, 1.0, 1.05, 1.1, 1.15, 0.95, 1.0, 1.05, 1.1, 1.15, 0.95, 1.0, 1.05, 1.1, 1.15, 0.95, 1.0, 1.05, 1.1, 1.15, 0.95, 1.0, 1.05, 1.1, 1.15, 0.95, 1.0, 1.05, 1.1, 1.15, 0.95, 1.0, 1.05, 1.1, 1.15, 0.95, 1.0, 1.05, 1.1, 1.15, 0.95, 1.0, 1.05, 1.1, 1.15)   # global fluctuation gain of each arm
_KF_I0 = 11   # index of the prior (first step, ties)
_KF_WR = 21.5   # utility weights: first-order online dFinal per unit rel / tke
_KF_WT = 6.26

_MF_LAMS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8)   # time-mean anchor FTL grid (prior: index 0)
_FP_KE = __TRAINED__FP_KE__  # FINAL_TRAIN_GENERATES__FP_KE
_FP_MEAN = __TRAINED__FP_MEAN__  # FINAL_TRAIN_GENERATES__FP_MEAN
_KF_BH = 8   # FTL spatial block (rows)
_KF_BW = 8   # FTL spatial block (cols)

_BF_GRID = (0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4)   # common/ke_ftl_card_t2.py: interval-width multipliers (per lead, u/v)
_BF_I0 = 4
_BF_BH = 8   # band FTL spatial block (rows)
_BF_BW = 8   # band FTL spatial block (cols)
_BT_A = (0.0, 0.25, 0.5, 0.5)   # band time-mean transport per lead group
_BF_TMN = (0.15496256947517395, -0.0005177936982363462)   # official target normalization (u, v) for the scored mask
_BF_TSD = (0.09681040793657303, 0.015963643789291382)
_BF_SIGMA = 0.0563870259

ONLINE = {'band_beta': 0.1, 'warm': 16, 'bias_beta': 0.05, 'sp_beta': 0.0, 'gate_c': 1, 'gate_mode': 'js', 'fld_beta': 0.05, 'fld_pyr': 1}            # online-calibration rates; all-zero == the frozen v3 bundle

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
# Measured on dgx-V100-2 against this very card: SEQ 5.033 -> PAR2 3.977 -> PAR3 3.443 ms/step
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


# ---- LL-ridge card patch (common/ll_ridge_v155_card.py; V200 T2-LLR1) -------------------------------------------------
# Online last-layer ridge of member m1: fixed preconditioner P1 (ll_p1.npy), running mean R/k, energy-preserving rescale,
# causal SSE gate.  Refusal of any kind leaves the card exactly the base card (m1's outc hook is never installed).
_LL_ON = 1
_LL_GSRC = "blend"           # gate utility source: "blend" (raw blend, the probe's P1-EP arm) or "m1" (m1 alone)
_LL_GATE = 0.0                  # the correction is served while U > _LL_GATE and at least one window has been revealed
_LL_LIVE = [False]              # witness: the prefix ran (set on every prefix call, i.e. inside the shipped body too)
_LL_STATE = {"on": False, "why": "", "built": None}
_LL_K = {}
_LL_SRC = {
    # prefix, elementwise over (H, W, T, C): residual r, gate numerator e, gate denominator tt (u/v only)
    "ll_r": ("template <typename T> void ll_r(T o0, T dd, T y, T u2, T s, T m, T cm, T h, T bl, T& r, T& e, T& tt) {"
             " T sc = (s > T(1e-30)) ? s : T(1e-30);"
             " r = cm * h * ((y - m) / sc - o0);"
             " T ou = o0 * s + m;"
             " T pu = (bl > T(0.5)) ? T(0.5) * (ou + u2) : ou;"
             " T dl = (bl > T(0.5)) ? T(0.5) * s * dd : s * dd;"
             " T a = y - pu;"
             " e = cm * (T(2) * a * dl - dl * dl);"
             " tt = cm * y * y; }", 3),
    # prefix, scalar: U + have * num / den (0 on no evidence, NaN-guarded)
    "ll_u": ("template <typename T> T ll_u(T u, T num, T den, T h) {"
             " T d = (h > T(0.5) && den > T(0)) ? num / den : T(0);"
             " d = (d - d == T(0)) ? d : T(0);"
             " return u + d; }", 1),
    # prefix, (Co, D): dW = (R P1) / max(k, 1), NaN-guarded
    "ll_ws": ("template <typename T> T ll_ws(T t, T k) {"
              " T w = t / ((k > T(1)) ? k : T(1));"
              " return (w - w == T(0)) ? w : T(0); }", 1),
    # m1's outc site, elementwise over (H, W, T, C): served output and the would-be correction dd (outc units)
    "ll_comb": ("template <typename T> void ll_comb(T o0, T o1, T v0, T mn1, T v1, T u, T k, T cm, T gate, T& out, T& dd) {"
                " T v1c = (v1 > T(1e-30)) ? v1 : T(1e-30);"
                " T kf = sqrt(v0 / v1c);"
                " T pres = o1 + (o1 - mn1) * (kf - T(1));"
                " bool ok = (cm > T(0.5)) && (k > T(0.5)) && (pres - pres == T(0));"
                " dd = ok ? pres - o0 : T(0);"
                " out = (ok && (u > gate)) ? pres : o0; }", 2),
}
_LL_KW = {"ll_r": dict(bl=1.0), "ll_comb": dict(gate=0.0)}
_LLF_ON = 1                     # LLR-fast (common/ll_ridge_fast_card.py): par4 forks the LL work onto its own lane
_LLF_LIVE = [False]             # witness: the forked outc path ran (set on every forked call, i.e. in the shipped body)
_LLX_ON = 1                     # v156g (common/ll_ridge_faster_card.py): m1's ll_comb fused with m1's in_norm undo
_LLX_LIVE = [False]             # witness: ll_comb_ep served m1's output (set on every fused call)
_LLX_STATE = {"built": None, "why": ""}
_LLX_K = {}
# jiterator takes at most 8 tensor inputs: ll_comb's (cm, k, u > gate) clauses are pre-folded into a per-channel flag
# f = cm && k ? (u > gate ? 2 : 1) : 0 by `llx_flag` on the LL lane right after the prefix (off m1's path); the served
# select and dd are then ll_comb's own: ok = f > 0.5 && finite(pres), out = ok && f > 1.5 ? pres : o0.
_LLX_SRC = {
    "llx_flag": ("template <typename T> T llx_flag(T cm, T k, T u, T gate) {"
                 " return ((cm > T(0.5)) && (k > T(0.5))) ? ((u > gate) ? T(2) : T(1)) : T(0); }", 1),
    "ll_comb_ep": ("template <typename T> void ll_comb_ep(T o0, T o1, T v0, T mn1, T v1, T f, T s, T m, T& y, T& dd) {"
                   " T v1c = (v1 > T(1e-30)) ? v1 : T(1e-30);"
                   " T kf = sqrt(v0 / v1c);"
                   " T pres = o1 + (o1 - mn1) * (kf - T(1));"
                   " bool ok = (f > T(0.5)) && (pres - pres == T(0));"
                   " dd = ok ? pres - o0 : T(0);"
                   " T out = (ok && (f > T(1.5))) ? pres : o0;"
                   " y = __fadd_rn(__fmul_rn(out, s), m); }", 2),
}
_RS_ON = 1                      # v156g RSET: batched reset (one foreach copy), proven equal to the original reset
_RS_LIVE = [False]              # witness: the batched reset ran
_RS_STATE = {"on": False, "why": ""}
_TPAR_ON = 1                    # v156g TPAR: par4 tail -- bfch3 + mean(bc0) and the kc_j1 state write on the idle m2 stream
_TPAR_LIVE = [False]            # witness: the forked tail ran (set on every forked call, i.e. in the shipped body)
_SHIN_ON = 1                    # v156g SHIN: par4 -- the members share one input preparation (m2 makes it, m1 reuses it)
_SHIN_LIVE = [False]            # witness: m1 reused m2's preparation (set on every shared call)
_LLDLY_ON = 0                   # v156g LLDLY: par4 -- the LL prefix waits for m2's opening (SHIN's event) as well as the fork
_BFOLD_ON = 1                   # v156g BFOLD: par4 -- m2's in_norm undo and the tail's blend as ONE kernel on m2's stream
_BFOLD_LIVE = [False]           # witness: the folded blend served the tail (set on every folded call)
_BF_K = {}
_BF_SRC = ("template <typename T> void ep_blend(T o, T s, T m, T y1, T& y2, T& p) {"
           " T v = __fadd_rn(__fmul_rn(o, s), m); y2 = v; p = T(0.5) * (y1 + v); }")


def _bf_kernels():
    if not _BF_K:
        try:
            _BF_K["ep_blend"] = torch.cuda.jiterator._create_multi_output_jit_fn(_BF_SRC, num_outputs=2)
        except Exception:
            _BF_K.clear()
    return _BF_K


def _llx_kernels():
    if _LLX_STATE["built"] is None:
        try:
            J = torch.cuda.jiterator
            _LLX_K["llx_flag"] = J._create_jit_fn(_LLX_SRC["llx_flag"][0], gate=0.0)
            _LLX_K["ll_comb_ep"] = J._create_multi_output_jit_fn(_LLX_SRC["ll_comb_ep"][0], num_outputs=2)
            _LLX_STATE["built"] = True
        except Exception as e:
            _LLX_K.clear()
            _LLX_STATE["built"] = False
            _LLX_STATE["why"] = "jiterator: %s: %s" % (type(e).__name__, str(e)[:200])
    return _LLX_K if _LLX_STATE["built"] else {}


def _bitseq(a, b):
    """Bitwise equality of float32 tensors (bit patterns), same shape, same strides on non-singleton dims."""
    if a.shape != b.shape or a.dtype != torch.float32 or b.dtype != torch.float32:
        return False
    if any(sa != sb for sa, sb, n in zip(a.stride(), b.stride(), a.shape) if n > 1):
        return False
    return bool(torch.equal(a.view(torch.int32), b.view(torch.int32)))


def _ll_kernels():
    """Create the jiterator functions once (NVRTC compiles them at their first, eager, call); {} if unavailable."""
    if _LL_STATE["built"] is None:
        try:
            J = torch.cuda.jiterator
            for k, (src, nout) in _LL_SRC.items():
                kw = _LL_KW.get(k, {})
                _LL_K[k] = (J._create_jit_fn(src, **kw) if nout == 1
                            else J._create_multi_output_jit_fn(src, num_outputs=nout, **kw))
            _LL_STATE["built"] = True
        except Exception as e:
            _LL_K.clear()
            _LL_STATE["built"] = False
            _LL_STATE["why"] = "jiterator: %s: %s" % (type(e).__name__, str(e)[:200])
    return _LL_K if _LL_STATE["built"] else {}


def _ll_fp32(fn, *a):
    """Run a GEMM with cuBLAS TF32 OFF (host-side flag read at kernel selection, so a capture records the FP32 kernel)."""
    mm = torch.backends.cuda.matmul
    prev = mm.allow_tf32
    if prev:
        mm.allow_tf32 = False
        try:
            return fn(*a)
        finally:
            mm.allow_tf32 = prev
    return fn(*a)


def _ll_lin(U_, X, Wm, b):
    """F.linear on (W_eff, b_eff) with FUSE-A's _lin TF32 mirror (cuBLAS TF32 follows cuDNN's flag for this call only)."""
    if getattr(U_, "_FUSEA_TF32", 0):
        mm = torch.backends.cuda.matmul
        prev, want = mm.allow_tf32, torch.backends.cudnn.allow_tf32
        if prev != want:
            mm.allow_tf32 = want
            try:
                return torch.nn.functional.linear(X, Wm, b)
            finally:
                mm.allow_tf32 = prev
    return torch.nn.functional.linear(X, Wm, b)


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
        self._ll_setup(submission_dir)          # LL-ridge card patch (before reset / capture)
        self._graph = None
        self._have_flag = None
        # STRPAR: the side streams are created HERE, before _capture(), because a CUDA-graph body
        # must not construct anything it reads (CLAUDE.md, the v65 fallback bake).  They are plain
        # attributes so a non-CUDA device leaves them None and the eligibility guard refuses.
        #
        # T2IMG: the allowlist is consulted HERE, before the streams exist.  On an unmeasured
        # torch nothing multi-stream is ever constructed, so `_strpar_eligible` refuses, no
        # multi-stream body runs and no multi-stream graph is captured -- which is the only
        # placement that can prevent those operations rather than report on them afterwards.
        self._side = self._side2 = None
        self._ll_side = None                    # LLR-fast: the LL lane (par4 only), guarded like the side streams
        if _STRPAR and _STRPAR_TVOK[0] and str(device).startswith("cuda") \
                and torch.cuda.is_available():
            try:
                self._side = torch.cuda.Stream()
                self._side2 = torch.cuda.Stream()
                if _LLF_ON and self.__dict__.get("_ll_on", False):
                    self._ll_side = torch.cuda.Stream()
            except Exception:
                self._side = self._side2 = None
                self._ll_side = None
        if self.__dict__.get("_ll_on", False):
            print("LLRFAST lane %s" % ("ready" if self._ll_side is not None else "absent (v156 serial LL path)"), flush=True)
        self.reset_ttt_state()
        self._rs_build()                        # v156g RSET: batched reset, proven before the capture uses it
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
        self._kc2_bj = z(2, n_t, C)                 # kernel-cut-2: [_bias_m, _bias_m2] carrier
        self._bias_m = self._kc2_bj[0]              # channels >= cm stay identically zero
        self._sp_m = z(1, n_t, H, W, C) if self.o["sp_beta"] > 0 else None
        # BIASGATE (v54): second-moment EMAs of the SAME statistics the bias EMAs average.  None
        # when the gate is off, so every gate statement is skipped and v54 == v53.
        _gc = self.o["gate_c"] > 0.0
        self._bias_m2 = self._kc2_bj[1] if (_gc and self.o["bias_beta"] > 0) else None
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
        self._kc2_sc = z(8)                         # kernel-cut-2: [pw_fld, pw_bias, pw_band, k_dev, usable, wf, -, -]
        self._pw_fld = self._kc2_sc[0]
        if self.o["fld_beta"] > 0.0:
            self._fld_levels = _fld_levels(self.o)
            T_f = 1 if self._fld_pooled else n_t
            self._kc2_fj = [z(2, 1, T_f, h, w, C) for (h, w) in self._fld_levels]   # kernel-cut-2: [m, m2] per level
            self._fld_m = [j_[0] for j_ in self._kc2_fj]
            self._fld_m2 = [j_[1] for j_ in self._kc2_fj]
            if self.mad_ref is not None and self.ch_idx is not None:
                lim = (self.o["fld_clip"] * self.mad_ref)[:, self.ch_idx]          # (n_t, C)
                if self._fld_pooled:
                    lim = lim.mean(dim=0, keepdim=True)                             # lead-mean MAD
                self._lim_fld = lim.reshape(1, T_f, 1, 1, C).contiguous()

        # ---- graph-only state: every host scalar of v4, moved onto the device ----
        self._k_dev = self._kc2_sc[3]  # step counter, incremented in-graph by _have (kernel-cut-2 view)
        self._have = z(())           # 1.0 iff prev_target_norm was supplied this step
        self._pw_bias = self._kc2_sc[1]  # running (1-beta)^k, one per estimator (kernel-cut-2 view)
        self._pw_sp = z(())
        self._pw_gam = z(())
        self._pw_gamf = z(())
        self._pw_band = self._kc2_sc[2]
        self._sx = z(1, n_t, H, W, C)          # static graph input
        self._sy = z(1, n_t, H, W, C)          # static graph prev-target
        self._kc_j1 = z(2, 1, n_t, H, W, C)    # kernel-cut card patch: [_sg_pg, _kf_pp (u,v of plane 1)]
        self._sg_pg = self._kc_j1[0]           # prediction after the FROZEN gamma, previous step
        self._kc_j2 = z(5, 1, n_t, H, W, C)    # kernel-cut card patch: [_out3 (3 planes), _bf_mid, _bf_hw]
        self._sg_pfin = self._kc_j2[0]         # what we actually returned, previous step (== _out3[0], same write point)
        self._out3 = self._kc_j2[0:3]          # pred / lower / upper, packed for a single clone
        # PAR4 cross-stream buffers.  The input-only half of the vgain features is computed on
        # the side stream and read on the main one, so it goes through STATIC memory exactly
        # like `_out3` -- a graph replays fixed kernels on fixed addresses, and a temporary
        # handed across streams is the one thing a capture must never rely on.  Shapes follow
        # `_vg_features`: KEi/li collapse the channel axis, ib keeps it.
        self._vg_li = z(1, H, W)               # log(KEi + eps), input-only
        self._vg_ib7 = z(1, 7, H, W)           # kernel-cut card patch: ib lives at planes 5:7 of the NCHW feature layout
        self._vg_ib = self._vg_ib7[:, 5:7].permute(0, 2, 3, 1)   # time-mean of the input's u,v (view)
        self._pin_x = torch.empty(1, n_t, H, W, C, dtype=torch.float32,
                                  pin_memory=str(self.device).startswith("cuda"))
        self._pin_y = torch.empty_like(self._pin_x, pin_memory=str(self.device).startswith("cuda"))
        # KE-FTL (common/ke_ftl_card_t2.py): static state of the per-trajectory mu leader
        self._kf_on = self._nstd is not None and self._nmean is not None
        if self._kf_on:
            self._kf_grid = torch.tensor(_KF_GRID, dtype=torch.float32, device=d)
            _tie = torch.zeros(len(_KF_GRID), 1, dtype=torch.float32); _tie[_KF_I0] = 1e-12
            self._kf_tie = _tie.to(d)
            _sd = self._nstd[..., :2].reshape(-1)[-2:].float()
            self._kf_sd = _sd.clone().contiguous()
            self._kf_mn = self._nmean[..., :2].reshape(-1)[-2:].float().clone().contiguous()
            self._kf_s2 = (_sd * _sd).contiguous()           # physical scale of squared u,v
            self._kf_s2h = (0.5 * _sd * _sd).contiguous()    # ... with the KE one-half folded in
            assert H % _KF_BH == 0 and W % _KF_BW == 0
            self._kf_acc = z(len(_KF_GRID), (H // _KF_BH) * (W // _KF_BW))   # running utility per arm and block
            self._kf_mu = z(1, H, W)               # this step's mu map (per block)
            self._kf_cg = torch.tensor(_KF_CG, dtype=torch.float32, device=d)
            self._kf_c = z(1, H, W)                # this step's gain map (per block)
            self._kf_pp = self._kc_j1[1][..., :2]  # previous step's pre-anchor u,v prediction (normalized; kc view)
            self._kf_kin = z(1, H, W)              # input KE (physical): previous step's until the prefix rewrites it
            self._bf_grid = torch.tensor(_BF_GRID, dtype=torch.float32, device=d)
            self._bf_tmn = torch.tensor(_BF_TMN, dtype=torch.float32, device=d)   # --band-mask-target
            self._bf_tsd = torch.tensor(_BF_TSD, dtype=torch.float32, device=d)
            _bta = torch.tensor(_BT_A, dtype=torch.float32).repeat_interleave((n_t + 3) // 4)[:n_t]
            self._bt_a = _bta.reshape(1, n_t, 1, 1, 1).to(d).contiguous()   # --band-transport
            _btie = torch.zeros(len(_BF_GRID), 1, 1, dtype=torch.float32); _btie[_BF_I0] = 1e-6
            self._bf_tie = _btie.to(d)
            assert H % _BF_BH == 0 and W % _BF_BW == 0
            self._bf_acc = z(len(_BF_GRID), (H // _BF_BH) * (W // _BF_BW), 2)   # per multiplier, block, u/v
            self._bf_m3 = z(1, 1, H, W, 3)            # kernel-cut card patch: 3-channel carrier (channel 2 unused)
            self._bf_m = self._bf_m3[..., :2]         # this step's multiplier map (per block and u/v)
            self._bf_mid = self._kc_j2[3][..., :2]    # previous step's unscaled interval midpoint (normalized; kc view)
            self._bf_hw = self._kc_j2[4][..., :2]     # ... and half-width (kc view)
            self._cl_ks = z(1, H, W)               # --clim: running sum of revealed target KE (physical)
            self._cl_ms = z(1, 1, H, W, 2)         # --clim: running sum of revealed target time-means (normalized)
            self._cl_n = z(())                     # --clim: revealed targets so far this trajectory
            self._cl_k0 = z(())                    # --clim-in0: first input window counted in the KE / mean climatology
            self._cl_m0 = z(())
            self._mf_grid = torch.tensor(_MF_LAMS, dtype=torch.float32, device=d)
            _mtie = torch.zeros(len(_MF_LAMS), 1, dtype=torch.float32); _mtie[0] = 1e-12
            self._mf_tie = _mtie.to(d)
            self._mf_acc = z(len(_MF_LAMS), (H // _KF_BH) * (W // _KF_BW))   # running rel utility per lam and block
            self._mf_lam = z(1, 1, H, W, 1)         # this step's time-mean anchor map
            self._mf_xbar3 = z(1, 1, H, W, 3)       # kernel-cut card patch: 3-channel carrier (channel 2 stays 0)
            self._mf_xbar = self._mf_xbar3[..., :2]  # input time-mean (normalized): previous step's until the prefix rewrites it
            self._kf_prior = torch.tensor(_FP_KE, dtype=torch.float32, device=d)    # --ftl-prior
            self._mf_prior = torch.tensor(_FP_MEAN, dtype=torch.float32, device=d)
            assert tuple(self._kf_prior.shape) == tuple(self._kf_acc.shape) and tuple(self._mf_prior.shape) == tuple(self._mf_acc.shape)

    # ------------------------------------------------------------------ v156g RSET (common/ll_ridge_faster_card.py)
    def reset_ttt_state(self) -> None:
        pl = self.__dict__.get("_rs_plan")
        if pl is not None:                    # the batched reset, proven equal to `_reset_orig` in `_rs_build`
            self._rs_fast(pl)
            return
        self._reset_orig()

    def _rs_targets(self):
        """Every tensor `_reset_orig` writes (its statements, in its order; views as it names them)."""
        L = []
        if getattr(self, '_pe_acc', None) is not None:
            L += [self._pe_acc, self._pe_cnt]
        L += [b for b in (self._bias_m, self._sp_m, self._gam_m, self._gamf_m, self._mad_m, self._bias_m2, self._sp_m2)
              if b is not None]
        L += list(self._fld_m or []) + list(self._fld_m2 or [])
        L += [self._k_dev, self._have, self._sy, self._sg_pg, self._sg_pfin]
        L += [self._pw_bias, self._pw_sp, self._pw_gam, self._pw_gamf, self._pw_band, self._pw_fld]
        if getattr(self, '_kf_on', False):
            L += [self._kf_acc, self._kf_pp, self._kf_kin, self._bf_acc, self._bf_mid, self._bf_hw, self._bf_m,
                  self._cl_ks, self._cl_ms, self._cl_n, self._cl_k0, self._cl_m0, self._mf_acc, self._mf_lam,
                  self._mf_xbar, self._kf_mu, self._kf_c]
        if self.__dict__.get("_ll_on", False):
            L += [self._ll_R, self._ll_k, self._ll_U, self._ll_o0, self._ll_dd, self._ll_u2,
                  self._ll_phiT[:, :self._ll_D0], self._ll_sm, self._ll_Weff, self._ll_beff]
        return L

    def _rs_fast(self, pl) -> None:
        self._prev_input = None
        self._prev_pg = None
        self._prev_pfin = None
        self._k = 0
        torch._foreach_copy_(pl["d"], pl["s"])
        if pl["xd"]:
            torch._foreach_copy_(pl["xd"], pl["xs"])    # strided targets: the per-tensor copy_ loop, in C++
        self._have_flag = False
        _RS_LIVE[0] = True

    def _rs_build(self) -> None:
        """Build the batched reset from the post-reset values of `_rs_targets` and PROVE it: from one random state of
        every float32 CUDA tensor this object holds, `_reset_orig` and `_rs_fast` must leave all of them bitwise equal
        (a missed or an extra target shows up as a random value on one side); else the original reset stays."""
        self.__dict__.pop("_rs_plan", None)
        ok, why = False, ""
        try:
            if not (_RS_ON and str(self.device).startswith("cuda") and torch.cuda.is_available()
                    and getattr(self, "_init_state", None) is None and hasattr(torch, "_foreach_copy_")):
                why = "not covered"
            elif torch.cuda.is_current_stream_capturing():
                why = "capturing"
            else:
                self._reset_orig()
                seen, d, s, x = set(), [], [], []
                for t in self._rs_targets():
                    key = (t.data_ptr(), tuple(t.shape), tuple(t.stride()))
                    if key in seen:
                        continue
                    seen.add(key)
                    if not (t.is_cuda and t.dtype == torch.float32):
                        raise RuntimeError("reset target is not a float32 CUDA tensor")
                    if t.is_contiguous():
                        d.append(t); s.append(t.clone())
                    else:
                        x.append((t, t.clone()))
                pl = dict(d=d, s=s, xd=[t_ for t_, _ in x], xs=[c_ for _, c_ in x])
                ok, why = self._rs_verify(pl)
                why = "%d dense + %d strided targets; %s" % (len(d), len(x), why)
                if ok:
                    self._rs_plan = pl
        except Exception as e:
            ok, why = False, "%s: %s" % (type(e).__name__, str(e)[:200])
        _RS_STATE.update(on=bool(ok), why=why)
        print("RSET %s (%s)" % ("on" if ok else "OFF", why), flush=True)

    def _rs_verify(self, pl):
        prm = set()
        mods = [v for v in list(self.__dict__.values()) + list(getattr(self, "_modules", {}).values())
                if isinstance(v, nn.Module)]
        for mod in mods:
            for p in list(mod.parameters()) + list(mod.buffers()):
                prm.add(p.untyped_storage().data_ptr())
        srcs = set()
        for n in ("_kf_prior", "_mf_prior", "_ll_W", "_ll_b"):    # what `_reset_orig` READS keeps its value
            t = self.__dict__.get(n)
            if torch.is_tensor(t):
                srcs.add(t.untyped_storage().data_ptr())
        ts = []
        for v in list(self.__dict__.values()):
            for t in (v if isinstance(v, (list, tuple)) else [v]):
                if (torch.is_tensor(t) and t.is_cuda and t.dtype == torch.float32 and t.numel() > 0 and not t.requires_grad
                        and t.untyped_storage().data_ptr() not in prm):
                    ts.append(t)
        keep = [t.clone() for t in ts]
        try:
            g = torch.Generator(device=ts[0].device).manual_seed(20260926)
            for t in ts:
                if t.untyped_storage().data_ptr() not in srcs:
                    t.copy_(torch.rand(t.shape, generator=g, device=t.device) * 4.0 - 2.0)
            R = [t.clone() for t in ts]
            self._reset_orig()
            A = [t.clone() for t in ts]
            pa = (self._prev_input, self._prev_pg, self._prev_pfin, self._k, self._have_flag)
            for t, r_ in zip(ts, R):
                t.copy_(r_)
            self._rs_fast(pl)
            B = [t.clone() for t in ts]
            pb = (self._prev_input, self._prev_pg, self._prev_pfin, self._k, self._have_flag)
            torch.cuda.synchronize()
            ok = all(_bitseq(a, b) for a, b in zip(A, B)) and pa == pb
            touched = sum(1 for r_, a in zip(R, A) if not _bitseq(r_, a))
            return bool(ok), "bitwise %s over %d tensors (%d changed by the reset)" % (
                "equal" if ok else "DIFFERENT", len(ts), touched)
        finally:
            for t, k_ in zip(ts, keep):
                t.copy_(k_)
            torch.cuda.synchronize()

    def _reset_orig(self) -> None:
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
        if getattr(self, '_kf_on', False):      # KE-FTL
            self._kf_acc.zero_(); self._kf_pp.zero_(); self._kf_kin.zero_()
            self._bf_acc.zero_(); self._bf_mid.zero_(); self._bf_hw.zero_(); self._bf_m.fill_(float(_BF_GRID[_BF_I0]))
            self._cl_ks.zero_(); self._cl_ms.zero_(); self._cl_n.zero_(); self._cl_k0.zero_(); self._cl_m0.zero_()
            self._mf_acc.zero_(); self._mf_lam.zero_(); self._mf_xbar.zero_()
            self._kf_acc.copy_(self._kf_prior); self._mf_acc.copy_(self._mf_prior)   # --ftl-prior warm start
            self._kf_mu.fill_(float(_KF_GRID[_KF_I0])); self._kf_c.fill_(float(_KF_CG[_KF_I0]))
        if self.__dict__.get("_ll_on", False):      # LL-ridge card patch
            self._ll_reset()

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

    def _gate_corr_orig(self, usable, wf) -> None:
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
    # ---- fused-TTT card patch (common/fused_ttt_card.py): guard + pure pieces -------------------------------------
    def _fj_run(self, key, fused, ref, *args):
        dec = self.__dict__.setdefault("_fj_dec", {})
        st = dec.get(key)
        if st is True:
            return fused(*args)
        if st is False:
            return ref(*args)
        ok, why = False, ""
        if key not in _FJ_PARTS:
            why = "part not enabled"
        elif torch.cuda.is_current_stream_capturing():
            why = "undecided at capture"
        elif not _fj_kernels():
            why = "jiterator unavailable"
        else:
            try:
                g = torch.Generator(device=args[0].device).manual_seed(20260925)
                syn = [_fj_synth(a, g) for a in args]
                worst = _fj_cmp(fused(*syn), ref(*syn))
                ok = worst <= _FJ_TOL
                why = "synthetic rel %.2e" % worst
            except Exception as e:
                why = "%s: %s" % (type(e).__name__, str(e)[:200])
        dec[key] = bool(ok)
        (_FJ_STATE["on"] if ok else _FJ_STATE["off"]).append((key, why))
        if ok or key in _FJ_PARTS:
            print("FUSEJ %s %s (%s)" % ("on" if ok else "OFF", key, why), flush=True)
        return fused(*args) if ok else ref(*args)

    def _fj_const(self):
        c = self.__dict__.get("_fj_c")
        if c is None:
            d = self._kf_grid.device
            f = lambda v: torch.tensor(v, dtype=torch.float32, device=d)
            sd = [float(v) for v in self._kf_sd.tolist()]
            s2h = [float(v) for v in self._kf_s2h.tolist()]
            c = dict(sel5=f([0.0, 1.0, 2.0, 3.0, 4.0]).reshape(5, 1, 1, 1, 1, 1),
                     sel5h=f([0.0, 1.0, 2.0, 3.0, 4.0]).reshape(5, 1, 1, 1),
                     sel2=f([0.0, 1.0]).reshape(2, 1, 1, 1),
                     mu=self._kf_grid.reshape(-1, 1, 1), cg=self._kf_cg.reshape(-1, 1, 1),
                     lam=self._mf_grid.reshape(-1, 1, 1), bm=self._bf_grid.reshape(-1, 1, 1, 1, 1),
                     zero=f(0.0), sdu=sd[0], sdv=sd[1], s2hu=s2h[0], s2hv=s2h[1])
            self.__dict__["_fj_c"] = c
        return c

    def _kf_terms_ref(self, y, pp, kin, xbar):
        """`_kf_update`'s prediction-independent terms, the original statements verbatim."""
        yy = y[..., :2]
        n_t = pp.shape[1]
        pm = pp.mean(dim=1, keepdim=True)
        ym = yy.mean(dim=1, keepdim=True)
        pf = pp - pm
        tf = yy - ym
        A = ((pf * pf).sum(dim=1) * self._kf_s2).sum(-1)          # (1,H,W), physical, u+v
        B = ((pf * tf).sum(dim=1) * self._kf_s2).sum(-1)
        Cc = ((tf * tf).sum(dim=1) * self._kf_s2).sum(-1)
        M = float(n_t) * (((pm - ym) ** 2)[:, 0] * self._kf_s2).sum()
        tn2 = ((yy * self._kf_sd + self._kf_mn) ** 2).sum()
        kp = A / (2.0 * n_t)
        kt = Cc / (2.0 * n_t)
        mu = self._kf_grid.reshape(-1, 1, 1)
        g = self._kf_cg.reshape(-1, 1, 1) * ((kp + mu * (kin - kp)).clamp_min(1e-12)
                                             / kp.clamp_min(1e-12)).sqrt().clamp(0.25, 4.0)
        g2 = g * g
        H, W = kp.shape[-2], kp.shape[-1]
        nh, nw = H // _KF_BH, W // _KF_BW
        sse0 = M + (A - 2.0 * B + Cc).sum()
        rel0 = sse0.clamp_min(1e-30).sqrt() / tn2.clamp_min(1e-30).sqrt()
        ktn = kt.pow(2).sum().clamp_min(1e-16)
        tke0 = (kp - kt).pow(2).sum().sqrt() / ktn.sqrt()
        e = g2 * A - 2.0 * g * B + Cc                                   # (G,H,W) squared-error share
        tk = (g2 * kp - kt).pow(2)                                       # (G,H,W) KE squared-error share
        Gn = e.shape[0]
        eb = e.reshape(Gn, nh, _KF_BH, nw, _KF_BW).sum(dim=(2, 4)).reshape(Gn, nh * nw)
        tb = tk.reshape(Gn, nh, _KF_BH, nw, _KF_BW).sum(dim=(2, 4)).reshape(Gn, nh * nw)
        u = -_KF_WR * eb / (2.0 * rel0.clamp_min(1e-12) * tn2.clamp_min(1e-30)) \
            - _KF_WT * tb / (2.0 * tke0.clamp_min(1e-8) * ktn)
        u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        # time-mean anchor FTL (common/ke_ftl_card_t2.py --mean-ftl)
        Dm = (pm - ym)[:, 0] * self._kf_sd                                # (1,H,W,2) physical
        Em = (xbar[:, 0] - pm[:, 0]) * self._kf_sd
        Ln = self._mf_grid.shape[0]
        em = ((Dm + self._mf_grid.reshape(-1, 1, 1, 1) * Em) ** 2).sum(-1)   # (L,H,W)
        emb = em.reshape(Ln, nh, _KF_BH, nw, _KF_BW).sum(dim=(2, 4)).reshape(Ln, nh * nw)
        um = -_KF_WR * float(n_t) * emb / (2.0 * rel0.clamp_min(1e-12) * tn2.clamp_min(1e-30))
        return u, torch.nan_to_num(um, nan=0.0, posinf=0.0, neginf=0.0), kt, ym

    def _kf_terms_fj(self, y, pp, kin, xbar):
        """The same terms with 7 jiterator kernels + 5 reductions (selector-stacked so one reduction serves several)."""
        K = _FJ; c = self._fj_const()
        yy = y[..., :2]
        n_t = pp.shape[1]
        H, W = pp.shape[2], pp.shape[3]
        nh, nw = H // _KF_BH, W // _KF_BW
        inv2t = 1.0 / (2.0 * n_t)
        pm = pp.mean(dim=1, keepdim=True)
        ym = yy.mean(dim=1, keepdim=True)
        R = K["kf_r5"](pp, pm, yy, ym, self._kf_s2, self._kf_sd, self._kf_mn, c["sel5"]).sum(dim=(2, 5))   # (5,1,H,W)
        Q = K["kf_q5"](R[0], R[1], R[2], R[3], R[4], c["sel5h"], inv2t=inv2t).sum(dim=(1, 2, 3))            # (5,)
        a1, a2 = K["kf_scal"](Q[0], Q[1], Q[2], Q[3], Q[4], wr=float(_KF_WR), wt=float(_KF_WT))
        E = K["kf_arm"](R[0], R[1], R[2], kin, c["mu"], c["cg"], c["sel2"], inv2t=inv2t)                   # (2,G,H,W)
        Gn = E.shape[1]
        EB = E.reshape(2, Gn, nh, _KF_BH, nw, _KF_BW).sum(dim=(3, 5)).reshape(2, Gn, nh * nw)
        u = K["kf_u"](EB[0], EB[1], a1, a2)
        em = K["kf_em"](pm[:, 0, ..., 0], pm[:, 0, ..., 1], ym[:, 0, ..., 0], ym[:, 0, ..., 1],
                        xbar[:, 0, ..., 0], xbar[:, 0, ..., 1], c["lam"], sdu=c["sdu"], sdv=c["sdv"])       # (L,H,W)
        Ln = em.shape[0]
        emb = em.reshape(Ln, nh, _KF_BH, nw, _KF_BW).sum(dim=(2, 4)).reshape(Ln, nh * nw)
        um = K["kf_um"](emb, a1, nt=float(n_t))
        kt = R[2] / (2.0 * n_t)
        return u, um, kt, ym

    def _bf_terms_ref(self, y, mid, hw):
        """`_bf_update`'s per-multiplier SPS numerators, the original statements verbatim."""
        yy = y[..., :2]
        sc = (yy * self._bf_tsd + self._bf_tmn).abs() > 1e-6
        r = (yy - mid).abs() / hw.clamp_min(1e-12)
        r = torch.where(sc, r, torch.full_like(r, 1e9)).unsqueeze(0)
        w = (2.0 / _BF_SIGMA) * (hw * self._kf_sd).unsqueeze(0)     # physical width / sigma at m = 1
        m = self._bf_grid.reshape(-1, 1, 1, 1, 1, 1)
        H, W = yy.shape[2], yy.shape[3]
        nh, nw = H // _BF_BH, W // _BF_BW
        S = torch.where(r <= m, torch.exp(-m * w), torch.zeros((), device=w.device)).sum(dim=(1, 2))   # (G,H,W,2)
        S = S.reshape(S.shape[0], nh, _BF_BH, nw, _BF_BW, 2).sum(dim=(2, 4)).reshape(S.shape[0], nh * nw, 2)
        return torch.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)          # (G,nb,2)

    def _bf_terms_fj(self, y, mid, hw):
        c = self._fj_const()
        yy = y[..., :2]
        n_t, H, W = yy.shape[1], yy.shape[2], yy.shape[3]
        nh, nw = H // _BF_BH, W // _BF_BW
        S = _FJ["bf_s"](yy, mid, hw, self._bf_tsd, self._bf_tmn, self._kf_sd, c["bm"], isig=2.0 / _BF_SIGMA)   # (G,T,H,W,2)
        Gn = S.shape[0]
        S = S.reshape(Gn, n_t, nh, _BF_BH, nw, _BF_BW, 2).sum(dim=(1, 3, 5)).reshape(Gn, nh * nw, 2)
        return torch.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

    def _fj_anchor_ref(self, p, mu, kin, cc, lam, xbar):
        """The tail's KE / time-mean anchor, the original statements verbatim (minus the `_kf_pp` write)."""
        _ka_p2 = p[..., :2]
        _ka_m = _ka_p2.mean(dim=1, keepdim=True)
        _ka_pf = _ka_p2 - _ka_m
        _ka_kp = ((_ka_pf * _ka_pf).mean(dim=1) * self._kf_s2h).sum(-1)
        _ka_g = cc * ((_ka_kp + mu * (kin - _ka_kp)).clamp_min(1e-12)
                      / _ka_kp.clamp_min(1e-12)).sqrt().clamp(0.25, 4.0)
        _ka_m2 = _ka_m + lam * (xbar - _ka_m)          # time-mean anchor (FTL lam map)
        _ka_sh = _ka_m2 - _ka_m                          # --band-follow: the same shift for the band
        return torch.cat([_ka_m2 + _ka_g[:, None, :, :, None] * _ka_pf, p[..., 2:]], dim=-1), _ka_sh

    def _fj_anchor_fj(self, p, mu, kin, cc, lam, xbar):
        c = self._fj_const()
        p2 = p[..., :2]
        var, m = torch.var_mean(p2, dim=1, unbiased=False, keepdim=True)                  # (1,1,H,W,2)
        g = _FJ["ka_g"](var[:, 0, ..., 0], var[:, 0, ..., 1], mu, kin, cc, s2hu=c["s2hu"], s2hv=c["s2hv"])   # (1,H,W)
        pn2 = _FJ["ka_p"](p2, m, g[:, None, :, :, None], lam, xbar)
        sh = _FJ["ka_sh"](m, lam, xbar)
        return torch.cat([pn2, p[..., 2:]], dim=-1), sh

    def _fj_anchor(self, p):
        pn, sh = self._fj_run("tail_anchor", self._fj_anchor_fj, self._fj_anchor_ref,
                              p, self._kf_mu, self._kf_kin, self._kf_c, self._mf_lam, self._mf_xbar)
        self._kf_pp.copy_(p[..., :2])
        return pn, sh

    def _fj_band_ref(self, p, lo, hi, sh, bfm):
        """The tail's band FTL, the original statements verbatim (minus the `_bf_mid` / `_bf_hw` writes)."""
        _bf_c = 0.5 * (lo[..., :2] + hi[..., :2])                    # band FTL (common/ke_ftl_card_t2.py)
        if sh is not None:
            _bf_c = _bf_c + sh                                        # --band-follow
        _bf_h = 0.5 * (hi[..., :2] - lo[..., :2]).abs()
        if self._bt_a.shape[1] == _bf_c.shape[1]:                     # --band-transport
            _bf_c = _bf_c + self._bt_a * (p[..., :2].mean(dim=1, keepdim=True) - _bf_c.mean(dim=1, keepdim=True))
        _bf_d = bfm * _bf_h
        lo = torch.cat([_bf_c - _bf_d, lo[..., 2:]], dim=-1)
        hi = torch.cat([_bf_c + _bf_d, hi[..., 2:]], dim=-1)
        return lo, hi, _bf_c, _bf_h

    def _fj_band_fj(self, p, lo, hi, sh, bfm):
        c = self._fj_const()
        bc, bh = _FJ["bf_ch"](lo[..., :2], hi[..., :2], sh if sh is not None else c["zero"])
        if self._bt_a.shape[1] == bc.shape[1]:
            bc = _FJ["bf_bt"](bc, self._bt_a, p[..., :2].mean(dim=1, keepdim=True), bc.mean(dim=1, keepdim=True))
        l2, h2 = _FJ["bf_lh"](bc, bfm, bh)
        return torch.cat([l2, lo[..., 2:]], dim=-1), torch.cat([h2, hi[..., 2:]], dim=-1), bc, bh

    def _fj_band(self, p, lo, hi, sh):
        lo, hi, bc, bh = self._fj_run("tail_band", self._fj_band_fj, self._fj_band_ref, p, lo, hi, sh, self._bf_m)
        self._bf_mid.copy_(bc)
        self._bf_hw.copy_(bh)
        return lo, hi

    # ---- kernel-cut card patch (common/fused_kcut_card.py, T2KCUT): the serial tail, fused ------------------------
    def _kc_ok(self) -> bool:
        """True iff the live config is the one `_kc_tail` implements (else the original tail runs, verbatim)."""
        v = self.__dict__.get("_kc_cfg")
        if v is None:
            o = self.o
            try:
                v = bool(hasattr(self.base, "m1") and hasattr(self.base, "m2") and self.head is not None
                         and self._nstd is not None and self._nmean is not None
                         and self._gate_on and self._corr is not None and self._kf_on and _BCDT2 != 0.0
                         and o.get("gam_beta", 0.0) == 0.0 and self._gamf_m is None
                         and o.get("band_beta", 0.0) > 0.0 and self.mad_ref is not None
                         and self.LOf is not None and self.HIf is not None
                         and self.band_lo is not None and self.band_hi is not None and self._sb is not None
                         and self._sx.shape[1] > 1 and self._sx.shape[-1] == 3
                         and tuple(self._sx.shape[:4]) == tuple(self._kf_pp.shape[:4])
                         and tuple(self._sx.shape[:4]) == tuple(self._bf_mid.shape[:4])
                         and int(self._bt_a.shape[1]) == int(self._sx.shape[1])
                         and tuple(self.band_lo.shape[1:2]) == tuple(self._sx.shape[1:2])
                         and self.__dict__.get("_vg_ib7") is not None and self.__dict__.get("_mf_xbar3") is not None
                         and self.__dict__.get("_bf_m3") is not None and self.__dict__.get("_kc_j1") is not None
                         and self.__dict__.get("_kc_j2") is not None
                         and self._ll_strpar_ok())                   # LL-ridge card patch
            except Exception:
                v = False
            self.__dict__["_kc_cfg"] = v
            print("KCUT %s" % ("config ok" if v else "config NOT covered -> original tail"), flush=True)
        return v

    def _kc_run(self, key, fused, ref, *args):
        dec = self.__dict__.setdefault("_kc_dec", {})
        st = dec.get(key)
        if st is True:
            return fused(*args)
        if st is False:
            return ref(*args)
        ok, why = False, ""
        if key not in _KC_PARTS:
            why = "part not enabled"
        elif torch.cuda.is_current_stream_capturing():
            why = "undecided at capture"
        elif not _kc_kernels():
            why = "jiterator unavailable"
        else:
            cu, mm = torch.backends.cudnn, torch.backends.cuda.matmul
            t_cu, t_mm = cu.allow_tf32, mm.allow_tf32
            try:
                tol32, tolamb = _KC_TOL.get(key, (1e-5, 1e-5))
                g = torch.Generator(device=args[0].device).manual_seed(20260926)
                syn = [_fj_synth(a, g) for a in args]
                vw = _KC_VIEW.get(key, _kc_tup)
                w_amb = max(_fj_cmp(vw(fused(*syn)), vw(ref(*syn))), _fj_cmp(vw(fused(*args)), vw(ref(*args))))
                cu.allow_tf32 = False; mm.allow_tf32 = False
                w_32 = max(_fj_cmp(vw(fused(*syn)), vw(ref(*syn))), _fj_cmp(vw(fused(*args)), vw(ref(*args))))
                ok = (w_32 <= tol32) and (w_amb <= tolamb)
                why = "rel fp32 %.2e ambient %.2e" % (w_32, w_amb)
            except Exception as e:
                why = "%s: %s" % (type(e).__name__, str(e)[:200])
            finally:
                cu.allow_tf32 = t_cu; mm.allow_tf32 = t_mm
            torch.cuda.synchronize()
        dec[key] = bool(ok)
        (_KC_STATE["on"] if ok else _KC_STATE["off"]).append((key, why))
        print("KCUT %s %s (%s)" % ("on" if ok else "OFF", key, why), flush=True)
        return fused(*args) if ok else ref(*args)

    def _kc_const(self):
        c = self.__dict__.get("_kc_c")
        if c is None:
            d = self._sx.device
            f = lambda v: torch.tensor(v, dtype=torch.float32, device=d)
            sd = [float(v) for v in self._nstd.reshape(-1)[:2].tolist()]
            mn = [float(v) for v in self._nmean.reshape(-1)[:2].tolist()]
            hd = self.head
            import math as _m
            C = int(self._sx.shape[-1])
            c = dict(vsel=f([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(1, 7, 1, 1),
                     vfm=hd.f_mu.detach().reshape(1, -1, 1, 1).clone(), vfs=hd.f_sd.detach().reshape(1, -1, 1, 1).clone(),
                     sdu=sd[0], sdv=sd[1], mnu=mn[0], mnv=mn[1],
                     s2hu=float(self._kf_s2h.reshape(-1)[0]), s2hv=float(self._kf_s2h.reshape(-1)[1]),
                     lmin=_m.log(hd.gmin), lmax=_m.log(hd.gmax),
                     pass3=f([0.0, 0.0] + [1.0] * (C - 2)),
                     selk=f([0.0, 1.0, 2.0]).reshape(3, 1, 1, 1, 1, 1),
                     selc=f([0.0, 1.0] + [2.0] * (C - 2)))
            self.__dict__["_kc_c"] = c
        return c

    # blend
    def _kc_blend_ref(self, y1, y2):
        return 0.5 * (y1 + y2)

    def _kc_blend_fj(self, y1, y2):
        return _KC["kc_blend"](y1, y2)

    # vgain: features + head -> (gain (1,H,W), pm (1,1,H,W,C))
    def _kc_vgain_ref(self, p, li, ib7):
        """par4's gain-field statements verbatim (`_li`/`_ib` from the static buffers) + the gain's pm."""
        _p_ph = p * self._nstd + self._nmean
        _p2 = _p_ph[..., :2]
        _pb = _p2.mean(1)
        _pf = _p2 - _pb.unsqueeze(1)
        _KEp = 0.5 * (_pf * _pf).mean(1).sum(-1)
        _lp = torch.log(_KEp + 1e-12)
        _li, _ib = li, ib7[:, 5:7].permute(0, 2, 3, 1)
        gh_ = self.head(torch.stack(
            [_lp, _li, _li - _lp, _pb[..., 0], _pb[..., 1], _ib[..., 0], _ib[..., 1]],
            dim=1))
        return gh_, p.mean(dim=1, keepdim=True)

    def _kc_vgain_fj(self, p, li, ib7):
        K = _KC; c = self._kc_const(); hd = self.head
        F_ = torch.nn.functional
        var, mean = torch.var_mean(p, dim=1, unbiased=False, keepdim=True)                     # (1,1,H,W,C)
        lp = K["kc_lp"](var[:, 0, ..., 0], var[:, 0, ..., 1], s2hu=c["s2hu"], s2hv=c["s2hv"])    # (1,H,W)
        z = K["kc_vz"](lp.unsqueeze(1), li.unsqueeze(1), mean[:, 0, ..., 0].unsqueeze(1), mean[:, 0, ..., 1].unsqueeze(1),
                       ib7, c["vsel"], c["vfm"], c["vfs"],
                       sdu=c["sdu"], sdv=c["sdv"], mnu=c["mnu"], mnv=c["mnv"])                   # (1,7,H,W) normalised
        h = z
        for cv in (hd.c1, hd.c2, hd.c3, hd.c4):
            h = K["kc_bgelu"](F_.conv2d(h, cv.weight, None, cv.stride, cv.padding, cv.dilation, cv.groups),
                              cv.bias.view(1, -1, 1, 1))
        o_ = F_.conv2d(h, hd.out.weight, None, hd.out.stride, hd.out.padding, hd.out.dilation, hd.out.groups)   # (1,1,H,W)
        gp = F_.linear(h.mean(dim=(2, 3)), hd.gpool.weight)                                     # (1,1)
        gh = K["kc_vout"](o_, hd.out.bias.view(1, 1, 1, 1), gp.view(1, 1, 1, 1), hd.gpool.bias.view(1, 1, 1, 1),
                          lmin=c["lmin"], lmax=c["lmax"])
        return gh.squeeze(1), mean

    # gctr: gain application, the online correction, the centre transport and the interval
    def _kc_gctr_ref(self, x, pm, g, corr, ol, oh):
        p = pm + (x - pm) * g
        pg = p
        p = p + corr
        _ctr = p + _BCDT2 * (x - p)
        _ctr = _ctr + 0.5 * (p - pg)
        return pg, p, _ctr + ol, _ctr + oh

    def _kc_gctr_fj(self, x, pm, g, corr, ol, oh):
        return tuple(_KC["kc_gctr"](x, pm, g, corr, ol, oh, bcd=float(_BCDT2)))

    # fin: KE / time-mean anchor + band FTL + hard safety + min/max, as the (3, 1, T, H, W, C) `_out3` value
    def _kc_fin_ref(self, pc, lo, hi, mu, kin, cc, lam, xb3, bfm3):
        pn, sh = self._fj_anchor_ref(pc, mu, kin, cc, lam, xb3[..., :2])
        lo2, hi2, bc, bh = self._fj_band_ref(pn, lo, hi, sh, bfm3[..., :2])
        pn = torch.nan_to_num(pn, nan=0.0, posinf=0.0, neginf=0.0)
        lo2 = torch.nan_to_num(lo2, nan=-1e3, posinf=-1e3, neginf=-1e3)
        hi2 = torch.nan_to_num(hi2, nan=1e3, posinf=1e3, neginf=1e3)
        return torch.stack([pn, torch.minimum(lo2, hi2), torch.maximum(lo2, hi2)]), bc, bh

    def _kc_fin_fj(self, pc, lo, hi, mu, kin, cc, lam, xb3, bfm3):
        K = _KC; c = self._kc_const()
        var, m = torch.var_mean(pc, dim=1, unbiased=False, keepdim=True)                         # (1,1,H,W,C)
        g = K["kc_kag"](var[:, 0, ..., 0], var[:, 0, ..., 1], mu, kin, cc, s2hu=c["s2hu"], s2hv=c["s2hv"])   # (1,H,W)
        pn = K["kc_kap3"](pc, m, g[:, None, :, :, None], lam, xb3, c["pass3"])                  # (1,T,H,W,C)
        bc0, bh = K["kc_bfch3"](lo, hi, m, lam, xb3)
        bc = K["kc_bfbt3"](bc0, self._bt_a, pn.mean(dim=1, keepdim=True), bc0.mean(dim=1, keepdim=True))
        res = K["kc_fin3"](c["selk"], c["selc"], pn, lo, hi, bc, bh, bfm3)                      # (3,1,T,H,W,C)
        return res, bc, bh                                                                    # bc/bh: 3 channels (2 = unused)

    def _kc_vg_input(self):
        """The INPUT-ONLY half of the vgain features into the static buffers: par4's side2 statements verbatim."""
        x_ph = self._sx * self._nstd + self._nmean
        i2 = x_ph[..., :2]
        ib = i2.mean(1)
        KEi = 0.5 * ((i2 - ib.unsqueeze(1)) ** 2).mean(1).sum(-1)
        self._vg_li.copy_(torch.log(KEi + 1e-12))
        self._vg_ib.copy_(ib)

    def _kc_tail(self, y1, y2, band_fn, cur=None, e3=None, e4=None, e5=None, llu2=True, tp=None, pb=None) -> None:
        """Everything after the members, for `_body` (no events) and `_body_strpar` (par3 / par4 events)."""
        p = pb if pb is not None else self._kc_run("blend", self._kc_blend_fj, self._kc_blend_ref, y1, y2)   # v156g BFOLD
        if self.__dict__.get("_ll_on", False) and _LL_GSRC == "blend" and llu2:
            self._ll_m2_stash(y2)                        # LL-ridge blend gate: m2's u/v after the join
        if e4 is not None:
            cur.wait_event(e4)
        gh_, pm = self._kc_run("vgain", self._kc2_vgain_fj, self._kc_vgain_ref, p, self._vg_li, self._vg_ib7)
        if e3 is not None:
            cur.wait_event(e3)
        _ol, _oh = band_fn()
        _SBPORT_LIVE[0] = True
        pg, pc, lo, hi = self._kc_run("gctr", self._kc2_gctr_fj, self._kc_gctr_ref,
                                      p, pm, gh_[:, None, :, :, None], self._corr, _ol, _oh)
        if tp is not None:                              # v156g TPAR: the state write on the side stream, joined at the
            _ew = torch.cuda.Event(); _ew.record(cur)   # end (nothing later in this step reads _kc_j1)
            tp.wait_event(_ew)
            with torch.cuda.stream(tp):
                self._kc2_w1(pg, pc)
            self.__dict__.setdefault("_tp_keep", []).extend([pg, pc])
        else:
            self._kc2_w1(pg, pc)                        # _sg_pg <- pg ; _kf_pp <- pc[..., :2] (kernel-cut-2: one copy)
        _BIASPAR_LIVE[0] = self._bias_m2 is not None
        _BIASPAR_LIVE[1] = self._sp_m2 is not None
        _BIASGATE_LIVE[0] = True
        _BCDT2_LIVE[0] = True
        if e5 is not None:
            cur.wait_event(e5)
        if tp is not None and self._tp_fin_ok():       # v156g TPAR: bfch3 + mean(bc0) beside kag / kap3 / mean(pn)
            res, bc, bh = self._kc2_fin_fj_par(pc, lo, hi, self._kf_mu, self._kf_kin, self._kf_c, self._mf_lam,
                                               self._mf_xbar3, self._bf_m3, cur, tp)
        else:
            res, bc, bh = self._kc_run("fin", self._kc2_fin_fj, self._kc_fin_ref, pc, lo, hi, self._kf_mu, self._kf_kin,
                                       self._kf_c, self._mf_lam, self._mf_xbar3, self._bf_m3)
        self._kc2_w2(res, bc, bh)                       # _out3 (= _sg_pfin), _bf_mid, _bf_hw (kernel-cut-2: one copy)
        if tp is not None:                              # v156g TPAR: join the side stream before the step ends
            _et = torch.cuda.Event(); _et.record(tp)
            cur.wait_event(_et)
            _TPAR_LIVE[0] = True

    # ---- kernel-cut card patch, round 2 (common/fused_kcut2_card.py, T2KCUT2) ----------------------------------
    def _kc2_cfg(self) -> bool:
        v = self.__dict__.get("_kc2_cfgv")
        if v is None:
            o = self.o
            try:
                v = bool(self._kc_ok() and o.get("gate_mode", "js") == "js" and self._sp_m is None and self._sp_m2 is None
                         and self._bias_m2 is not None and self._fld_m is not None and self._fld_pooled
                         and [tuple(x) for x in self._fld_levels] == [(4, 8), (8, 16), (32, 64)]
                         and (self._H, self._W) == (32, 64) and self._chmask is not None and self.cm == 2
                         and int(self._sx.shape[-1]) == 3 and _SGATE > 0.0 and self._sb_lt == int(self._sx.shape[1])
                         and self._sb_c2 == 2 and self._lim_fld is not None and self._lim_bias is not None
                         and self.__dict__.get("_kc2_sc") is not None and self.__dict__.get("_kc2_bj") is not None
                         and self.__dict__.get("_kc2_fj") is not None)
            except Exception:
                v = False
            self.__dict__["_kc2_cfgv"] = v
            print("KCUT2 %s" % ("config ok" if v else "config NOT covered -> round-1 / original code"), flush=True)
        return v

    def _kc2_decide(self, key, fused, ref, args, view=None):
        dec = self.__dict__.setdefault("_kc2_dec", {})
        st = dec.get(key)
        if st is not None:
            return st
        ok, why = False, ""
        if key not in _KC2_PARTS:
            why = "part not enabled"
        elif not self._kc2_cfg():
            why = "config not covered"
        elif torch.cuda.is_current_stream_capturing():
            why = "undecided at capture"
        elif not _kc2_kernels():
            why = "jiterator unavailable"
        else:
            cu, mm = torch.backends.cudnn, torch.backends.cuda.matmul
            t_cu, t_mm = cu.allow_tf32, mm.allow_tf32
            vw = view or _kc_tup
            try:
                tol32, tolamb = _KC2_TOL.get(key, (1e-5, 1e-5))
                g = torch.Generator(device=args[0].device).manual_seed(20260927)
                syn = [_fj_synth(a, g) for a in args]
                w_amb = max(_fj_cmp(vw(fused(*syn)), vw(ref(*syn))), _fj_cmp(vw(fused(*args)), vw(ref(*args))))
                cu.allow_tf32 = False; mm.allow_tf32 = False
                w_32 = max(_fj_cmp(vw(fused(*syn)), vw(ref(*syn))), _fj_cmp(vw(fused(*args)), vw(ref(*args))))
                ok = (w_32 <= tol32) and (w_amb <= tolamb)
                why = "rel fp32 %.2e ambient %.2e" % (w_32, w_amb)
            except Exception as e:
                why = "%s: %s" % (type(e).__name__, str(e)[:200])
            finally:
                cu.allow_tf32 = t_cu; mm.allow_tf32 = t_mm
            torch.cuda.synchronize()
        dec[key] = bool(ok)
        (_KC2_STATE["on"] if ok else _KC2_STATE["off"]).append((key, why))
        print("KCUT2 %s %s (%s)" % ("on" if ok else "OFF", key, why), flush=True)
        return bool(ok)

    def _kc2_run(self, key, fused, ref, *args):
        return fused(*args) if self._kc2_decide(key, fused, ref, args) else ref(*args)

    def _kc2_const(self):
        c = self.__dict__.get("_kc2_c")
        if c is None:
            d = self._sx.device
            f = lambda v: torch.tensor(v, dtype=torch.float32, device=d)
            n_t, C = int(self._sx.shape[1]), int(self._sx.shape[-1])
            H, W = self._H, self._W
            mx = torch.zeros(1, 1, W, device=d); mx[..., 1:-1] = 1.0
            my = torch.zeros(1, H, 1, device=d); my[:, 1:-1] = 1.0
            sb = self._sb
            c = dict(sel2j=f([0.0, 1.0]).reshape(2, 1, 1, 1, 1, 1), selk5=f([0.0, 1.0, 2.0, 3.0, 4.0]).reshape(5, 1, 1, 1, 1, 1),
                     sel3=f([0.0, 1.0, 2.0]).reshape(3, 1, 1, 1, 1, 1), sel2b=f([0.0, 1.0]).reshape(2, 1, 1),
                     sel1=f([0.0]).reshape(1, 1, 1), one=f(1.0), sel8=f([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
                     mref_i=self.mad_ref.index_select(1, self.ch_idx).contiguous(),
                     blo=self.band_lo.reshape(n_t, C).contiguous(), bhi=self.band_hi.reshape(n_t, C).contiguous(),
                     dl=torch.where(self.band_lo != 0, self.band_lo, torch.ones_like(self.band_lo)).reshape(n_t, C).contiguous(),
                     dh=torch.where(self.band_hi != 0, self.band_hi, torch.ones_like(self.band_hi)).reshape(n_t, C).contiguous(),
                     selc=f([0.0, 1.0] + [2.0] * (C - 2)), mx=mx, my=my,
                     s4=f([0.0, 1.0, 2.0, 3.0]).reshape(1, 4, 1, 1), s3=f([0.0, 1.0, 2.0]).reshape(1, 3, 1, 1),
                     sfm=sb.f_mu.detach().reshape(1, -1, 1, 1).clone(), sfs=sb.f_sd.detach().reshape(1, -1, 1, 1).clone())
            self.__dict__["_kc2_c"] = c
        return c

    # -- wr: tail state writes -------------------------------------------------------------------------------------
    def _kc2_gctr_fj(self, x, pm, g, corr, ol, oh):
        if "wr" not in _KC2_PARTS or not self._kc2_cfg() or not _kc2_kernels():
            return self._kc_gctr_fj(x, pm, g, corr, ol, oh)
        # a contiguous full-shape operand FIRST: TensorIterator then lays the outputs out contiguously (the members'
        # output p is a permuted view, and the old operand order made every downstream tail tensor inherit it)
        J, L, Hh = _KC2["kc2_gctr2"](ol, oh, corr, self._kc2_const()["sel2j"], x, pm, g, bcd=float(_BCDT2))
        return J[0], J[1], L[0], Hh[0]

    def _kc2_w1(self, pg, pc):
        b = pg._base
        if b is not None and pc._base is b and tuple(b.shape) == tuple(self._kc_j1.shape):
            self._kc_j1.copy_(b)
        else:
            torch.stack([pg, pc], out=self._kc_j1)

    def _bf_ok(self) -> bool:
        """v156g BFOLD: decided once, eagerly (never inside a capture): m2's EPI decided on, the KCUT blend decided fused,
        and ep_blend == (ep_ms, then kc_blend) bitwise on synthetic tensors laid out like the live ones."""
        st = self.__dict__.get("_bf_dec")
        if st is not None:
            return st
        ok, why = False, ""
        try:
            if torch.cuda.is_current_stream_capturing():
                why = "undecided at capture"
            elif not (_BFOLD_ON and self.base.m2.__dict__.get("_epi_ok") is True
                      and self.__dict__.get("_kc_dec", {}).get("blend") is True and _bf_kernels() and _KC):
                why = "not covered"
            else:
                UM = sys.modules[type(self.base.m2).__module__]
                H, W, T, C = int(self._H), int(self._W), int(self._n_t), int(self._C)
                g = torch.Generator(device=self._sx.device).manual_seed(20260928)
                r = lambda *s_: torch.randn(*s_, generator=g, device=self._sx.device)
                lay = lambda t_: t_.t().view(1, T * C, H, W).reshape(1, T, C, H, W).permute(0, 1, 3, 4, 2)
                o = lay(r(H * W, T * C) * 30.0)
                y1 = lay(r(H * W, T * C) * 3.0)
                s = r(1, 1, 1, 1, C).abs() + 1e-3
                m = r(1, 1, 1, 1, C)
                y2r = UM._EPI_K["ep_ms"](o, s, m)
                pr = _KC["kc_blend"](y1, y2r)
                y2f, pf = _BF_K["ep_blend"](o, s, m, y1)
                ok = bool(_bitseq(y2f, y2r) and _bitseq(pf, pr))
                torch.cuda.synchronize()
                why = "synthetic bitwise %s" % ("equal" if ok else "DIFFERENT")
        except Exception as e:
            ok, why = False, "%s: %s" % (type(e).__name__, str(e)[:200])
        self._bf_dec = bool(ok)
        print("BFOLD %s (%s)" % ("on" if ok else "OFF", why), flush=True)
        return bool(ok)

    def _shin_ok(self) -> bool:
        """v156g SHIN: both members prepare their input identically (config) and both decided the PRO pass on."""
        st = self.__dict__.get("_sh_dec")
        if st is not None:
            return st
        if torch.cuda.is_current_stream_capturing():
            return False
        ok = False
        try:
            m1, m2 = self.base.m1, self.base.m2
            keys = ("in_norm", "cond", "in_aug", "tattn", "condx", "cond_film", "mimo", "_tooth", "t_in")
            same = all(repr(getattr(m1, k_, None)) == repr(getattr(m2, k_, None)) for k_ in keys)
            ok = bool(_SHIN_ON and same and m1.in_norm == "self" and m1.cond == "stats" and m1.tattn is None
                      and not m1.in_aug and not m1.condx and not getattr(m1, "cond_film", 0)
                      and m1.__dict__.get("_pro_ok") is True and m2.__dict__.get("_pro_ok") is True
                      and type(m1).__module__ == type(m2).__module__
                      and hasattr(sys.modules[type(m1).__module__], "_shin_z"))
        except Exception:
            ok = False
        self._sh_dec = ok
        print("SHIN %s" % ("on" if ok else "OFF"), flush=True)
        return ok

    def _tp_fin_ok(self) -> bool:
        """v156g TPAR: the forked fin mirrors `_kc2_fin_fj`'s KCUT2 branch -- only where that branch is the decided one."""
        return bool(self.__dict__.get("_kc_dec", {}).get("fin") is True and "wr" in _KC2_PARTS and self._kc2_cfg()
                    and _kc2_kernels())

    def _kc2_fin_fj_par(self, pc, lo, hi, mu, kin, cc, lam, xb3, bfm3, cur, tp):
        """`_kc2_fin_fj` (KCUT2 branch) with bfch3 and mean(bc0) on the side stream `tp`: the same kernels on the same
        inputs; main waits for them only before bfbt3.  Every tensor that crosses the stream boundary is held in
        `_tp_keep` until the step's final join."""
        K = _KC; c = self._kc_const()
        var, m = torch.var_mean(pc, dim=1, unbiased=False, keepdim=True)
        em = torch.cuda.Event(); em.record(cur)
        tp.wait_event(em)
        with torch.cuda.stream(tp):
            bc0, bh = K["kc_bfch3"](lo, hi, m, lam, xb3)
            bcm = bc0.mean(dim=1, keepdim=True)
            eb = torch.cuda.Event(); eb.record(tp)
        g = K["kc_kag"](var[:, 0, ..., 0], var[:, 0, ..., 1], mu, kin, cc, s2hu=c["s2hu"], s2hv=c["s2hv"])
        pn = K["kc_kap3"](pc, m, g[:, None, :, :, None], lam, xb3, c["pass3"])
        pnm = pn.mean(dim=1, keepdim=True)
        cur.wait_event(eb)
        bc = K["kc_bfbt3"](bc0, self._bt_a, pnm, bcm)
        r5 = _KC2["kc2_fin5"](self._kc2_const()["selk5"], c["selc"], pn, lo, hi, bc, bh, bfm3)   # (5,1,T,H,W,C)
        self.__dict__.setdefault("_tp_keep", []).extend([pc, lo, hi, var, m, bc0, bh, bcm, g, pn, pnm, bc, r5])
        return r5[0:3], r5[3], r5[4]

    def _kc2_fin_fj(self, pc, lo, hi, mu, kin, cc, lam, xb3, bfm3):
        if "wr" not in _KC2_PARTS or not self._kc2_cfg() or not _kc2_kernels():
            return self._kc_fin_fj(pc, lo, hi, mu, kin, cc, lam, xb3, bfm3)
        K = _KC; c = self._kc_const()
        var, m = torch.var_mean(pc, dim=1, unbiased=False, keepdim=True)
        g = K["kc_kag"](var[:, 0, ..., 0], var[:, 0, ..., 1], mu, kin, cc, s2hu=c["s2hu"], s2hv=c["s2hv"])
        pn = K["kc_kap3"](pc, m, g[:, None, :, :, None], lam, xb3, c["pass3"])
        bc0, bh = K["kc_bfch3"](lo, hi, m, lam, xb3)
        bc = K["kc_bfbt3"](bc0, self._bt_a, pn.mean(dim=1, keepdim=True), bc0.mean(dim=1, keepdim=True))
        r5 = _KC2["kc2_fin5"](self._kc2_const()["selk5"], c["selc"], pn, lo, hi, bc, bh, bfm3)   # (5,1,T,H,W,C)
        return r5[0:3], r5[3], r5[4]

    def _kc2_w2(self, res, bc, bh):
        b = res._base
        if b is not None and bc._base is b and bh._base is b and tuple(b.shape) == tuple(self._kc_j2.shape):
            self._kc_j2.copy_(b)
        elif bc.shape[-1] == self._kc_j2.shape[-1]:
            torch.cat([res, bc.unsqueeze(0), bh.unsqueeze(0)], dim=0, out=self._kc_j2)
        else:
            self._bf_mid.copy_(bc)
            self._bf_hw.copy_(bh)
            self._out3.copy_(res)

    # -- vgcl: vgain head channels_last --------------------------------------------------------------------------
    def _kc2_vgain_fj(self, p, li, ib7):
        if not (self._kc2_cfg() and self._kc2_decide("vgcl", self._kc2_vgcl_fj, self._kc_vgain_ref, (p, li, ib7))):
            return self._kc_vgain_fj(p, li, ib7)
        return self._kc2_vgcl_fj(p, li, ib7)

    def _kc2_vgcl_fj(self, p, li, ib7):
        K = _KC; c = self._kc_const(); hd = self.head
        F_ = torch.nn.functional
        wcl = self.__dict__.get("_kc2_wcl")
        if wcl is None:
            wcl = [cv.weight.detach().contiguous(memory_format=torch.channels_last) for cv in (hd.c1, hd.c2, hd.c3, hd.c4)]
            self.__dict__["_kc2_wcl"] = wcl
        var, mean = torch.var_mean(p, dim=1, unbiased=False, keepdim=True)
        lp = K["kc_lp"](var[:, 0, ..., 0], var[:, 0, ..., 1], s2hu=c["s2hu"], s2hv=c["s2hv"])
        z = K["kc_vz"](lp.unsqueeze(1), li.unsqueeze(1), mean[:, 0, ..., 0].unsqueeze(1), mean[:, 0, ..., 1].unsqueeze(1),
                       ib7, c["vsel"], c["vfm"], c["vfs"], sdu=c["sdu"], sdv=c["sdv"], mnu=c["mnu"], mnv=c["mnv"])
        h = z if z.is_contiguous(memory_format=torch.channels_last) else z.contiguous(memory_format=torch.channels_last)
        for cv, w_ in zip((hd.c1, hd.c2, hd.c3, hd.c4), wcl):
            h = K["kc_bgelu"](F_.conv2d(h, w_, None, cv.stride, cv.padding, cv.dilation, cv.groups), cv.bias.view(1, -1, 1, 1))
        o_ = F_.conv2d(h, hd.out.weight, None, hd.out.stride, hd.out.padding, hd.out.dilation, hd.out.groups)
        gp = F_.linear(h.mean(dim=(2, 3)), hd.gpool.weight)
        gh = K["kc_vout"](o_, hd.out.bias.view(1, 1, 1, 1), gp.view(1, 1, 1, 1), hd.gpool.bias.view(1, 1, 1, 1),
                          lmin=c["lmin"], lmax=c["lmax"])
        return gh.reshape(1, gh.shape[-2], gh.shape[-1]), mean

    # -- gate ----------------------------------------------------------------------------------------------------
    def _kc2_gate_ref(self, usable, wf, bm_, bq_, pwb, f0, f1, f2, q0, q1, q2, pwf):
        """`_gate_corr` (bias + BIASFIELD, js gate, no sp term) verbatim, state as arguments, returning `corr`."""
        o = self.o
        n_t, C = self._n_t, self._C
        cb = usable / (1.0 - pwb).clamp_min(1e-12)
        bm = bm_ * cb
        bm = bm * self._gate_factor(bm, bq_ * cb, pwb, o["bias_beta"])
        corr = (bm.clamp(-self._lim_bias, self._lim_bias) * (o["bias_scale"] * wf)).reshape(1, n_t, 1, 1, C)
        cb = usable / (1.0 - pwf).clamp_min(1e-12)
        c = float(o["gate_c"]) if float(o["fld_gate_c"]) < 0.0 else float(o["fld_gate_c"])
        tot = None
        for fm_, fq_ in ((f0, q0), (f1, q1), (f2, q2)):
            fm = fm_ * cb
            if c > 0.0:
                fm = fm * self._gate_factor(fm, fq_ * cb, pwf, o["fld_beta"], c)
            tm = _fld_up(fm.clamp(-self._lim_fld, self._lim_fld), self._H, self._W) * (o["fld_scale"] * wf)
            tot = tm if tot is None else tot + tm
        return corr + tot

    def _kc2_gate_fj(self, usable, wf, bm_, bq_, pwb, f0, f1, f2, q0, q1, q2, pwf):
        K = _KC2; o = self.o
        n_t, C = self._n_t, self._C
        eps = float(o["gate_eps"])
        bb, fb = float(o["bias_beta"]), float(o["fld_beta"])
        cf = float(o["gate_c"]) if float(o["fld_gate_c"]) < 0.0 else float(o["fld_gate_c"])
        tb = K["kc2_gate"](bm_, bq_, pwb, usable, wf, self._lim_bias, k=bb / (2.0 - bb), c=float(o["gate_c"]), eps=eps,
                           sc=float(o["bias_scale"]))                                         # (n_t, C)
        lv = []
        for fm_, fq_ in ((f0, q0), (f1, q1), (f2, q2)):
            lv.append(K["kc2_gate"](fm_, fq_, pwf, usable, wf, self._lim_fld, k=fb / (2.0 - fb), c=cf, eps=eps,
                                    sc=float(o["fld_scale"])))                                # (1,1,h,w,C)
        y = K["kc2_gsum"](tb.reshape(1, n_t, 1, 1, 1, 1, 1, 1, C), lv[0].reshape(1, 1, 4, 1, 1, 8, 1, 1, C),
                          lv[1].reshape(1, 1, 4, 2, 1, 8, 2, 1, C), lv[2].reshape(1, 1, 4, 2, 4, 8, 2, 4, C))
        return y.reshape(1, n_t, self._H, self._W, C)

    def _gate_corr(self, usable, wf) -> None:
        """kernel-cut-2 dispatcher; the original body is `_gate_corr_orig`."""
        if self._kc2_cfg() and torch.is_tensor(usable) and torch.is_tensor(wf):
            args = (usable, wf, self._bias_m, self._bias_m2, self._pw_bias, self._fld_m[0], self._fld_m[1], self._fld_m[2],
                    self._fld_m2[0], self._fld_m2[1], self._fld_m2[2], self._pw_fld)
            if self._kc2_decide("gate", self._kc2_gate_fj, self._kc2_gate_ref, args):
                self._corr.copy_(self._kc2_gate_fj(*args))
                _BIASFIELD_LIVE[0] = True
                return
        self._gate_corr_orig(usable, wf)

    # -- ema -----------------------------------------------------------------------------------------------------
    def _kc2_ema_args(self):
        return (self._have, self._sy, self._sg_pg, self._sg_pfin, self._fld_m[0], self._fld_m2[0], self._fld_m[1],
                self._fld_m2[1], self._fld_m[2], self._fld_m2[2], self._bias_m, self._bias_m2, self._mad_m, self._kc2_sc)

    def _kc2_ema_ref(self, have, sy, pg, pfin, a0, b0, a1, b1, a2, b2, bm, bq, mad, sc):
        """The prefix EMA statements verbatim (fld pyramid, bias + m2, band MAD, powers, step, usable, wf) on clones."""
        o = self.o
        fm = [a0.clone(), a1.clone(), a2.clone()]; fq = [b0.clone(), b1.clone(), b2.clone()]
        bm = bm.clone(); bq = bq.clone(); mad = mad.clone(); sc = sc.clone()
        pwf, pwb, pwn, kd = sc[0], sc[1], sc[2], sc[3]
        r = sy - pg
        a = have * o["fld_beta"]
        for _i, (_h, _w) in enumerate(self._fld_levels):
            rf = _fld_down(r, _h, _w)
            if self._fld_pooled:
                rf = rf.mean(dim=1, keepdim=True)
            if self._chmask is not None:
                rf = rf * self._chmask
            fm[_i].mul_(1.0 - a).add_(rf * a)
            fq[_i].mul_(1.0 - a).add_(rf * rf * a)
            r = r - _fld_up(rf, self._H, self._W)
        pwf.mul_(1.0 - a)
        a = have * o["bias_beta"]
        rm = r.mean(dim=(0, 2, 3))
        if self._chmask is not None:
            rm = rm * self._chmask.reshape(1, -1)
        bm.mul_(1.0 - a).add_(rm * a)
        bq.mul_(1.0 - a).add_(rm * rm * a)
        pwb.mul_(1.0 - a)
        a = have * o["band_beta"]
        k = self.cm
        md = (sy[..., :k] - pfin[..., :k]).abs().mean(dim=(0, 2, 3))
        mad.mul_(1.0 - a).add_(md * a)
        pwn.mul_(1.0 - a)
        kd.add_(have)
        sc[4].copy_(kd.clamp(max=1.0))
        sc[5].copy_((kd / float(max(1, int(o["warm"])))).clamp(max=1.0))
        return fm[0], fq[0], fm[1], fq[1], fm[2], fq[2], bm, bq, mad, sc[:6]

    def _kc2_ema_fj(self, have, sy, pg, pfin, a0, b0, a1, b1, a2, b2, bm, bq, mad, sc):
        K = _KC2; o = self.o; c = self._kc2_const()
        T_, H, W, C = int(sy.shape[1]), self._H, self._W, int(sy.shape[-1])
        r = sy - pg
        outs = []
        for (m_, q_), (_h, _w) in zip(((a0, b0), (a1, b1), (a2, b2)), self._fld_levels):
            rf = _fld_down(r, _h, _w).mean(dim=1, keepdim=True)
            X = K["kc2_ema"](c["sel3"], m_, q_, rf, self._chmask, have, beta=float(o["fld_beta"]))      # (3,1,1,h,w,C)
            outs += [X[0], X[1]]
            r = (r.reshape(1, T_, _h, H // _h, _w, W // _w, C)
                 - X[2].reshape(1, 1, _h, 1, _w, 1, C)).reshape(1, T_, H, W, C)
        rm = r.mean(dim=(0, 2, 3))
        B = K["kc2_ema"](c["sel2b"], bm, bq, rm, self._chmask.reshape(1, -1), have, beta=float(o["bias_beta"]))   # (2,n_t,C)
        k = self.cm
        md = K["kc2_absd"](sy[..., :k], pfin[..., :k]).mean(dim=(0, 2, 3))
        MD = K["kc2_ema"](c["sel1"], mad, mad, md, c["one"], have, beta=float(o["band_beta"]))           # (1,n_t,k)
        SC = K["kc2_sc"](c["sel8"], sc, sc[3], have, fb=float(o["fld_beta"]), bb=float(o["bias_beta"]),
                         nb=float(o["band_beta"]), warm=float(max(1, int(o["warm"]))))                   # (8,)
        return outs[0], outs[1], outs[2], outs[3], outs[4], outs[5], B[0], B[1], MD[0], SC[:6]

    def _kc2_ema_ok(self) -> bool:
        if not (self._kc2_cfg() and self._kf_on):
            return False
        return self._kc2_decide("ema", self._kc2_ema_fj, self._kc2_ema_ref, self._kc2_ema_args())

    def _kc2_ema(self, have) -> None:
        t = self._kc2_ema_fj(*self._kc2_ema_args())
        for i in range(3):
            self._kc2_fj[i].copy_(t[2 * i]._base[0:2])
        self._kc2_bj.copy_(t[6]._base)
        self._mad_m.copy_(t[8])
        self._kc2_sc[0:6].copy_(t[9])

    def _kc2_uw(self):
        if self.__dict__.get("_kc2_dec", {}).get("ema") is True:
            return self._kc2_sc[4], self._kc2_sc[5]
        o = self.o
        return self._k_dev.clamp(max=1.0), (self._k_dev / float(max(1, int(o["warm"])))).clamp(max=1.0)

    # -- clim ----------------------------------------------------------------------------------------------------
    def _kc2_clim_ref(self, cur, acc, c0, n):
        acc = acc.clone(); c0 = c0.clone()
        f = ((n + c0) < 0.5).float()
        acc.add_(cur * f)
        c0.add_(f)
        nn_ = n + c0
        return torch.where(nn_ > 0.5, acc / nn_.clamp_min(1.0), cur), acc, c0

    def _kc2_clim_fj(self, cur, acc, c0, n):
        acc = acc.clone(); c0 = c0.clone()
        f = _KC2["kc2_clf"](n, c0)
        acc.addcmul_(cur, f)
        c0.add_(f)
        return _KC2["kc2_clp"](n, c0, acc, cur), acc, c0

    def _cl_pick0(self, cur, acc, c0):
        """kernel-cut-2 dispatcher (in place, like the original); the original body is `_cl_pick0_orig`."""
        if self._kc2_cfg() and self._kc2_decide("clim", self._kc2_clim_fj, self._kc2_clim_ref, (cur, acc, c0, self._cl_n)):
            f = _KC2["kc2_clf"](self._cl_n, c0)
            acc.addcmul_(cur, f)
            c0.add_(f)
            return _KC2["kc2_clp"](self._cl_n, c0, acc, cur)
        return self._cl_pick0_orig(cur, acc, c0)

    # -- band / sband ---------------------------------------------------------------------------------------------
    def _kc2_sbf_ref(self, x_ph):
        sb = self._sb
        return (self._sb_feat(x_ph) - sb.f_mu.reshape(1, -1, 1, 1)) / sb.f_sd.reshape(1, -1, 1, 1)

    def _kc2_sbf_fj(self, x_ph):
        K = _KC2; c = self._kc2_const()
        u = x_ph[..., :2]
        sd = u.std(1, unbiased=False)                                                    # (1,H,W,2)
        mn = u.mean(1)
        d2 = (u[:, 1:] - u[:, :-1]).pow(2).mean(1)
        sp = K["kc2_sp"](mn[..., 0], mn[..., 1])                                          # (1,H,W)
        P = torch.nn.functional.pad(sp, (1, 1, 1, 1))
        g = K["kc2_g"](P[:, 1:-1, 2:], P[:, 1:-1, :-2], P[:, 2:, 1:-1], P[:, :-2, 1:-1], c["mx"], c["my"])
        win = sd.mean(dim=(1, 2), keepdim=True)                                          # (1,1,1,2)
        q = lambda t: t.unsqueeze(1)
        za = K["kc2_za"](q(sd[..., 0]), q(sd[..., 1]), q(mn[..., 0]), q(mn[..., 1]), c["s4"], c["sfm"][:, 0:4], c["sfs"][:, 0:4])
        zb = K["kc2_zb"](q(d2[..., 0]), q(d2[..., 1]), q(g), q(sd[..., 0]), q(sd[..., 1]), c["s4"], c["sfm"][:, 4:8],
                         c["sfs"][:, 4:8])
        zc = K["kc2_zc"](q(u[:, -1, ..., 0]), q(u[:, -1, ..., 1]), q(mn[..., 0]), q(mn[..., 1]), win[..., 0].unsqueeze(1),
                         c["s3"], c["sfm"][:, 8:11], c["sfs"][:, 8:11])
        return torch.cat([za, zb, zc], dim=1)

    def _kc2_sbh_ref(self, z):
        sb = self._sb
        return sb.out(sb.body(z)).clamp(-12.0, 2.0)

    def _kc2_sbh_fj(self, z):
        K = _KC2; sb = self._sb; F_ = torch.nn.functional
        h = z
        mods = list(sb.body)
        for i in range(0, len(mods), 3):
            cv, gn = mods[i], mods[i + 1]
            y = cv(h)                                                                     # replicate pad + conv + bias
            G = gn.num_groups; Cc = y.shape[1]
            v, m = torch.var_mean(y.reshape(G, -1), dim=1, unbiased=False, keepdim=True)
            h = K["kc2_gng"](y.reshape(G, Cc // G, -1), m.view(G, 1, 1), v.view(G, 1, 1), gn.weight.view(G, Cc // G, 1),
                             gn.bias.view(G, Cc // G, 1), eps=float(gn.eps)).view(y.shape)
        o_ = F_.conv2d(h, sb.out.weight, None, sb.out.stride, sb.out.padding, sb.out.dilation, sb.out.groups)
        return K["kc2_bcl"](o_, sb.out.bias.view(1, -1, 1, 1))

    def _kc2_band_ref(self, m, usable, wf, mad, pwn):
        """`_band_branch` + `_sb_offsets`' post-head part verbatim, from the head output `m`."""
        o = self.o
        n_t, C = self._n_t, self._C
        cb = usable / (1.0 - pwn).clamp_min(1e-12)
        s = ((mad * cb) / self.mad_ref)[:, self.ch_idx]
        s = (1.0 + wf * (s - 1.0)).clamp(o["s_min"], o["s_max"])
        j = ((s - _S0) * _SSC).round().long().clamp_(0, _NF - 1)
        gl = torch.gather(self.LOf, 0, j.unsqueeze(0)).reshape(1, n_t, 1, 1, C)
        gh = torch.gather(self.HIf, 0, j.unsqueeze(0)).reshape(1, n_t, 1, 1, C)
        _w = usable
        if _SGATE > 0.0:
            _w = _w * ((s - _SGATE) / _SRAMP).clamp(0.0, 1.0).reshape(1, n_t, 1, 1, C)
        _ol = self.band_lo + _w * (gl - self.band_lo)
        _oh = self.band_hi + _w * (gh - self.band_hi)
        c2 = self._sb_c2
        b_, h_, w_ = int(m.shape[0]), int(m.shape[2]), int(m.shape[3])
        g = self._sb_lt * c2
        mv = m.reshape(b_, g, h_ * w_).permute(1, 0, 2).reshape(g, b_ * h_ * w_).contiguous()
        bi = torch.searchsorted(self._sb_edge, mv)
        lo = torch.gather(self._sb_lo, 1, bi).reshape(self._sb_lt, c2, b_, h_, w_).permute(2, 0, 3, 4, 1)
        hi = torch.gather(self._sb_hi, 1, bi).reshape(self._sb_lt, c2, b_, h_, w_).permute(2, 0, 3, 4, 1)
        if C > c2:
            lo = torch.nn.functional.pad(lo, (0, C - c2), value=0.0)
            hi = torch.nn.functional.pad(hi, (0, C - c2), value=0.0)
        _dl = torch.where(self.band_lo != 0, self.band_lo, torch.ones_like(self.band_lo))
        _dh = torch.where(self.band_hi != 0, self.band_hi, torch.ones_like(self.band_hi))
        return lo * (_ol / _dl), hi * (_oh / _dh)

    def _kc2_band_fj(self, m, usable, wf, mad, pwn):
        K = _KC2; o = self.o; c = self._kc2_const()
        n_t, C = self._n_t, self._C
        mi = mad.index_select(1, self.ch_idx)                                             # (n_t, C)
        jf, w = K["kc2_bs"](mi, usable, pwn, wf, c["mref_i"], smin=float(o["s_min"]), smax=float(o["s_max"]),
                            s0=float(_S0), ssc=float(_SSC), nfm1=float(_NF - 1), sg=float(_SGATE), sr=float(_SRAMP))
        j = jf.long()
        gl = torch.gather(self.LOf, 0, j.unsqueeze(0))[0]
        gh = torch.gather(self.HIf, 0, j.unsqueeze(0))[0]
        rl, rh = K["kc2_br"](c["blo"], c["bhi"], gl, gh, w, c["dl"], c["dh"])             # (n_t, C)
        c2 = self._sb_c2
        h_, w_ = int(m.shape[2]), int(m.shape[3])
        g = self._sb_lt * c2
        bi = torch.searchsorted(self._sb_edge, m.reshape(g, h_ * w_))
        lo = torch.gather(self._sb_lo, 1, bi).reshape(self._sb_lt, c2, 1, h_, w_).permute(2, 0, 3, 4, 1)
        hi = torch.gather(self._sb_hi, 1, bi).reshape(self._sb_lt, c2, 1, h_, w_).permute(2, 0, 3, 4, 1)
        ol, oh = K["kc2_bb"](lo[..., 0:1], lo[..., 1:2], hi[..., 0:1], hi[..., 1:2], rl.reshape(1, n_t, 1, 1, C),
                             rh.reshape(1, n_t, 1, 1, C), c["selc"])
        return ol, oh

    def _band_branch(self, n_t: int, C: int, usable, wf, x_ph=None):
        """kernel-cut-2 dispatcher; the original body is `_band_branch_orig`."""
        if (self._kc2_cfg() and torch.is_tensor(usable) and torch.is_tensor(wf) and self._sb is not None
                and "band" in _KC2_PARTS and n_t == self._n_t and C == self._C):
            xp = x_ph if x_ph is not None else self._sx * self._nstd + self._nmean
            z = self._kc2_run("sbf", self._kc2_sbf_fj, self._kc2_sbf_ref, xp)
            m = self._kc2_run("sbh", self._kc2_sbh_fj, self._kc2_sbh_ref, z)
            return self._kc2_run("band", self._kc2_band_fj, self._kc2_band_ref, m, usable, wf, self._mad_m, self._pw_band)
        return self._band_branch_orig(n_t, C, usable, wf, x_ph)

    def _kf_kin_of(self, x):
        """Input window KE per pixel, physical units (input-only)."""
        xf = x[..., :2] - x[..., :2].mean(dim=1, keepdim=True)
        return ((xf * xf).mean(dim=1) * self._kf_s2h).sum(-1)

    def _cl_pick0_orig(self, cur, acc, c0):
        """--clim-in0: with no history yet, count `cur` (the first input window) into `acc`; then the climatology."""
        f = ((self._cl_n + c0) < 0.5).float()
        acc.add_(cur * f)
        c0.add_(f)
        n = self._cl_n + c0
        return torch.where(n > 0.5, acc / n.clamp_min(1.0), cur)

    def _cl_pick(self, cur, acc):
        """--clim: the revealed-target climatology once a target of this trajectory is revealed, else `cur`."""
        return torch.where(self._cl_n > 0.5, acc / self._cl_n.clamp_min(1.0), cur)

    def _bf_update(self, y, have) -> None:
        """Band FTL: official SPS numerator of every width multiplier on the revealed previous window, per lead and u/v;
        accumulate and set `_bf_m` to the per-(lead, channel) leader.  Prediction-independent.  An element is inside the
        m-scaled interval iff m >= |t - mid| / hw; unscored elements (physical target 0) are never inside.
        (fused-TTT card patch: the numerators come from `_bf_terms_fj` or its verbatim reference.)"""
        S = self._fj_run("bf_terms", self._bf_terms_fj, self._bf_terms_ref, y, self._bf_mid, self._bf_hw)
        H, W = y.shape[2], y.shape[3]
        nh, nw = H // _BF_BH, W // _BF_BW
        self._bf_acc.add_(S * have)
        d = self._bf_acc - self._bf_acc[_BF_I0:_BF_I0 + 1] + self._bf_tie
        _j = torch.argmax(d, dim=0)                                         # (nb,2)
        _mv = torch.gather(self._bf_grid, 0, _j.reshape(-1)).reshape(1, 1, nh, 1, nw, 1, 2)
        _mb = torch.gather(self._bf_grid, 0, _j.reshape(-1)).reshape(nh, nw, 2).permute(2, 0, 1).unsqueeze(0)   # --smooth-band
        self._bf_m.copy_(torch.nn.functional.interpolate(_mb, size=(H, W), mode="bilinear", align_corners=False)
                         .permute(0, 2, 3, 1).reshape(1, 1, H, W, 2))

    def _kf_update(self, y, have) -> None:
        """KE-FTL: score every grid mu on the revealed previous pair, accumulate, and set `_kf_mu` to the leader.
        Reads `y` (previous target, normalized) and `_kf_pp`/`_kf_kin` as the previous step left them.
        `have` is the in-graph 0/1 tensor or an eager float; with have == 0 the accumulator is unchanged.
        (fused-TTT card patch: the terms come from `_kf_terms_fj` or its verbatim reference; the commit is verbatim.)"""
        u, um, kt, ym = self._fj_run("kf_terms", self._kf_terms_fj, self._kf_terms_ref,
                                     y, self._kf_pp, self._kf_kin, self._mf_xbar)
        H, W = kt.shape[-2], kt.shape[-1]
        nh, nw = H // _KF_BH, W // _KF_BW
        _up = lambda v: torch.nn.functional.interpolate(v.reshape(1, 1, nh, nw), size=(H, W), mode="bilinear",
                                                        align_corners=False).reshape(1, H, W)   # --smooth-leaders
        self._mf_acc.add_(um * have)
        _dm = self._mf_acc - self._mf_acc[0:1] + self._mf_tie
        _jm = torch.argmax(_dm, dim=0)
        self._mf_lam.copy_(_up(torch.gather(self._mf_grid, 0, _jm)).reshape(1, 1, H, W, 1))
        self._kf_acc.add_(u * have)
        self._cl_ks.add_(kt * have)                  # --clim: after the utilities read the previous anchors
        self._cl_ms.add_(ym * have)
        self._cl_n.add_(have)
        d = self._kf_acc - self._kf_acc[_KF_I0:_KF_I0 + 1] + self._kf_tie
        _j = torch.argmax(d, dim=0)                                        # (nb,)
        _ex = lambda v: v.reshape(1, nh, 1, nw, 1).expand(1, nh, _KF_BH, nw, _KF_BW).reshape(1, H, W)
        self._kf_mu.copy_(_up(torch.gather(self._kf_grid, 0, _j)))
        self._kf_c.copy_(_up(torch.gather(self._kf_cg, 0, _j)))

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
            if self._kf_on:                             # KE-FTL (prediction-independent: previous pair + input)
                self._kf_update(self._sy, have)
                self._bf_update(self._sy, have)
                self._kf_kin.copy_(self._cl_pick0(self._kf_kin_of(self._sx), self._cl_ks, self._cl_k0))
                self._mf_xbar.copy_(self._cl_pick0(self._sx[..., :2].mean(dim=1, keepdim=True), self._cl_ms, self._cl_m0))
            if self._kc2_ema_ok():                  # kernel-cut-2: the prefix EMAs fused (bit-exact)
                self._kc2_ema(have)
            else:
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

            if self._kc_ok():                       # kernel-cut card patch: steps 2-6 through the fused tail
                usable, wf = self._kc2_uw()              # kernel-cut-2 (== the clamps)
                self._gate_corr(usable, wf)
                self._kc_vg_input()
                n_t, C = self.band_lo.shape[1], self.band_lo.shape[-1]
                if self.__dict__.get("_ll_on", False):     # LL-ridge prefix (sequential body)
                    self._ll_prefix(self._sy, have)
                y1 = self.base.m1(self._sx)
                y2 = self.base.m2(self._sx)
                self._kc_tail(y1, y2, lambda: self._band_branch(n_t, C, usable, wf))
                return
            # ---------- 2. forward ----------
            if self.__dict__.get("_ll_on", False):     # LL-ridge prefix (sequential body, original tail)
                self._ll_prefix(self._sy, have)
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
            _ctr = _ctr + 0.5 * (p - self._sg_pg)  # BAND_CENTER_TRANSPORT
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
            _ka_sh = None
            # --- KE-FTL anchor (common/ke_ftl_card_t2.py): v105's KE anchor, mu = the trajectory's FTL leader
            if self._kf_on and self._sx.shape[1] > 1 and tuple(p.shape[:4]) == tuple(self._kf_pp.shape[:4]):
                p, _ka_sh = self._fj_anchor(p)                                    # fused-TTT card patch (KE / time-mean anchor)
            if self._kf_on and tuple(lo.shape[:4]) == tuple(self._bf_mid.shape[:4]) and lo.shape[-1] >= 2:
                lo, hi = self._fj_band(p, lo, hi, _ka_sh)                      # fused-TTT card patch (band FTL)
            p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
            lo = torch.nan_to_num(lo, nan=-1e3, posinf=-1e3, neginf=-1e3)
            hi = torch.nan_to_num(hi, nan=1e3, posinf=1e3, neginf=1e3)
            self._sg_pfin.copy_(p)
            self._out3[0].copy_(p)
            self._out3[1].copy_(torch.minimum(lo, hi))
            self._out3[2].copy_(torch.maximum(lo, hi))

    # ------------------------------------------------------------------ STRPAR fast path
    # ------------------------------------------------------------------ LL-ridge card patch (common/ll_ridge_v155_card.py)
    def _ll_setup(self, submission_dir) -> None:
        """Allocate the LL-ridge state and install m1's outc hook.  Any refusal leaves the base card's path untouched."""
        self._ll_on = False
        self._ll_wait = None
        why = ""
        m1 = getattr(self.base, "m1", None)
        try:
            m2 = getattr(self.base, "m2", None)
            oc = getattr(m1, "outc", None)
            ok = bool(_LL_ON and m1 is not None and m2 is not None and str(self.device).startswith("cuda")
                      and torch.cuda.is_available() and isinstance(oc, nn.Conv2d) and tuple(oc.kernel_size) == (1, 1)
                      and tuple(oc.stride) == (1, 1) and tuple(oc.padding) == (0, 0) and oc.groups == 1
                      and oc.bias is not None and getattr(m1, "in_norm", "") == "self" and getattr(m1, "head", None) == ""
                      and getattr(m1, "out_mode", None) == "direct" and getattr(m1, "freq_hi", 0) is None
                      and not getattr(m1, "mimo", 0) and getattr(m1, "aux_head", None) is None
                      and not getattr(m1, "hetero_head", 0) and getattr(m1, "_tooth", None) is None
                      and int(oc.out_channels) == int(self._n_t) * int(self._C) and int(self._C) == 3
                      and _LL_GSRC in ("blend", "m1"))
            if not ok:
                why = "config not covered"
            else:
                p1 = None
                for d_ in (submission_dir, _HERE):
                    f_ = os.path.join(d_, "ll_p1.npy") if d_ else ""
                    if f_ and os.path.exists(f_):
                        p1 = np.load(f_, allow_pickle=False)
                        break
                D = int(oc.in_channels) + 1
                if p1 is None or tuple(p1.shape) != (D, D) or not np.isfinite(p1).all():
                    ok = False
                    why = "ll_p1.npy missing / shape %s != %s / non-finite" % (None if p1 is None else tuple(p1.shape), (D, D))
            if ok and not _ll_kernels():
                ok = False
                why = _LL_STATE["why"] or "jiterator unavailable"
            if ok:
                d = self.device
                H, W, T, C = int(self._H), int(self._W), int(self._n_t), int(self._C)
                z = lambda *s: torch.zeros(*s, dtype=torch.float32, device=d)
                Co, D0 = T * C, D - 1
                self._ll_D0 = D0
                self._ll_P1 = torch.as_tensor(np.ascontiguousarray(p1, dtype=np.float32)).to(d)
                self._ll_W = oc.weight.detach().reshape(Co, D0).clone().contiguous()    # copies: never a Parameter
                self._ll_b = oc.bias.detach().clone().contiguous()
                self._ll_Weff = self._ll_W.clone()
                self._ll_beff = self._ll_b.clone()
                self._ll_phiT = z(H * W, D)
                self._ll_phiT[:, D0] = 1.0
                self._ll_o0 = z(H, W, T, C)          # m1's uncorrected outc output of the previous window
                self._ll_dd = z(H, W, T, C)          # its would-be (energy-preserving) correction
                self._ll_u2 = z(H, W, T, C)          # m2's output of the previous window (blend gate only)
                self._ll_sm = z(2, C)                # m1's in_norm [s; m] of the previous window
                self._ll_sm[0].fill_(1.0)
                self._ll_R = z(Co, D)
                self._ll_k = z(())
                self._ll_U = z(())
                self._ll_one = torch.ones((), dtype=torch.float32, device=d)
                self._ll_zero = z(())
                self._ll_zy = z(1, T, H, W, C)
                _cm = torch.zeros(C, dtype=torch.float32)
                _cm[:2] = 1.0
                self._ll_cm = _cm.to(d)
                self._ll_on = True
                why = "D %d, gate %s > %g, P1 |max| %.3e" % (D, _LL_GSRC, _LL_GATE, float(np.abs(p1).max()))
        except Exception as e:
            self._ll_on = False
            why = "%s: %s" % (type(e).__name__, str(e)[:200])
        if self._ll_on:
            m1.__dict__["_ll_fn"] = self._ll_outc
            if _LLX_ON:
                m1.__dict__["_ll_epx"] = self._ll_epx      # v156g: fused ll_comb + in_norm undo (decided at first eager call)
            if _LL_GSRC == "blend":
                self.base.__dict__["_ll_m2fn"] = self._ll_m2_stash
        _LL_STATE["on"] = bool(self._ll_on)
        _LL_STATE["why"] = why
        print("LLR1 %s (%s)" % ("on" if self._ll_on else "OFF", why), flush=True)

    def _ll_reset(self) -> None:
        for b in (self._ll_R, self._ll_k, self._ll_U, self._ll_o0, self._ll_dd, self._ll_u2):
            b.zero_()
        self._ll_phiT[:, :self._ll_D0].zero_()
        self._ll_sm.zero_()
        self._ll_sm[0].fill_(1.0)
        self._ll_Weff.copy_(self._ll_W)
        self._ll_beff.copy_(self._ll_b)

    def _ll_strpar_ok(self) -> bool:
        """LL clause of `_strpar_eligible` / `_kc_ok`: off, or fully set up (both bodies implement it)."""
        return (not self.__dict__.get("_ll_on", False)) or (bool(_LL_K) and self.__dict__.get("_ll_P1") is not None)

    def _ll_prefix(self, y, have) -> None:
        """Prediction-independent LL-ridge update from the previous revealed window (par4: head of side2)."""
        K = _LL_K
        H, W, D0 = int(self._H), int(self._W), self._ll_D0
        sm = self._ll_sm
        yv = y[0].permute(1, 2, 0, 3)                                            # (H, W, T, C) view
        r, e, tt = K["ll_r"](self._ll_o0, self._ll_dd, yv, self._ll_u2, sm[0], sm[1], self._ll_cm, have,
                             bl=(1.0 if _LL_GSRC == "blend" else 0.0))
        num = e.sum()
        den = tt.sum()
        _ll_fp32(self._ll_R.addmm_, r.reshape(H * W, -1).t(), self._ll_phiT)
        self._ll_k.add_(have)
        self._ll_U.copy_(K["ll_u"](self._ll_U, num, den, have))
        ws = K["ll_ws"](_ll_fp32(torch.mm, self._ll_R, self._ll_P1), self._ll_k)
        torch.add(self._ll_W, ws[:, :D0], out=self._ll_Weff)
        torch.add(self._ll_b, ws[:, D0], out=self._ll_beff)
        _LL_LIVE[0] = True

    def _ll_outc(self, mem, y, m_, s_):
        """m1's outc site: o0 (the original outc), o1 (same GEMM path on W_eff/b_eff), energy-preserving rescale, exact
        gated select, stash.  Returns a (1, Co, H, W) tensor with o0's strides."""
        if self.__dict__.get("_llf_par", False):          # LLR-fast: par4 only (set around m1's call)
            return self._llf_outc(mem, y, m_, s_)
        oc = mem.outc
        ev = self.__dict__.get("_ll_wait")
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)
        o0 = oc(y)
        K = _LL_K
        U_ = sys.modules[type(oc).__module__]
        H, W, T, C = int(self._H), int(self._W), int(self._n_t), int(self._C)
        Co, D0 = T * C, self._ll_D0
        X = U_._nhwc(y) if hasattr(U_, "_nhwc") else y[0].permute(1, 2, 0).reshape(H * W, D0)
        if getattr(oc, "_fusea", None) is True:
            o1v = _ll_lin(U_, X, self._ll_Weff, self._ll_beff).view(H, W, T, C)
        else:
            o1 = torch.nn.functional.conv2d(y, self._ll_Weff.view(Co, D0, 1, 1), self._ll_beff)
            o1v = o1[0].permute(1, 2, 0).reshape(H, W, T, C)
        o0v = o0[0].permute(1, 2, 0).reshape(H, W, T, C)
        v0, _mn0 = torch.var_mean(o0v, dim=2, unbiased=False, keepdim=True)
        v1, mn1 = torch.var_mean(o1v, dim=2, unbiased=False, keepdim=True)
        out, dd = K["ll_comb"](o0v, o1v, v0, mn1, v1, self._ll_U, self._ll_k, self._ll_cm, gate=float(_LL_GATE))
        self._ll_phiT[:, :D0].copy_(X)
        self._ll_o0.copy_(o0v)
        self._ll_dd.copy_(dd)
        torch.stack([s_.reshape(C), m_.reshape(C)], out=self._ll_sm)
        return out.permute(2, 3, 0, 1).reshape(1, Co, H, W)

    def _llf_outc(self, mem, y, m_, s_):
        """LLR-fast (par4): `_ll_outc`'s statements -- the same kernels on the same inputs -- split over two lanes.
        LL lane (after `ey`): o1 GEMM + var_mean(o1) -> `e1`; phiT / sm stash.  main: o0 = outc(y) + var_mean(o0), wait `e1`,
        ll_comb -> `ec`.  LL lane (after `ec`): o0 / dd stash.  Cross-lane tensors are held until the step's final join."""
        oc = mem.outc
        K = _LL_K
        U_ = sys.modules[type(oc).__module__]
        H, W, T, C = int(self._H), int(self._W), int(self._n_t), int(self._C)
        Co, D0 = T * C, self._ll_D0
        cur = torch.cuda.current_stream()
        ls = self._ll_side
        ey = torch.cuda.Event(); ey.record(cur)
        ls.wait_event(ey)
        with torch.cuda.stream(ls):
            X = U_._nhwc(y) if hasattr(U_, "_nhwc") else y[0].permute(1, 2, 0).reshape(H * W, D0)
            if getattr(oc, "_fusea", None) is True:
                o1v = _ll_lin(U_, X, self._ll_Weff, self._ll_beff).view(H, W, T, C)
            else:
                o1 = torch.nn.functional.conv2d(y, self._ll_Weff.view(Co, D0, 1, 1), self._ll_beff)
                o1v = o1[0].permute(1, 2, 0).reshape(H, W, T, C)
            v1, mn1 = torch.var_mean(o1v, dim=2, unbiased=False, keepdim=True)
            e1 = torch.cuda.Event(); e1.record(ls)
            self._ll_phiT[:, :D0].copy_(X)
            torch.stack([s_.reshape(C), m_.reshape(C)], out=self._ll_sm)
        o0 = oc(y)
        o0v = o0[0].permute(1, 2, 0).reshape(H, W, T, C)
        v0, _mn0 = torch.var_mean(o0v, dim=2, unbiased=False, keepdim=True)
        cur.wait_event(e1)
        out, dd = K["ll_comb"](o0v, o1v, v0, mn1, v1, self._ll_U, self._ll_k, self._ll_cm, gate=float(_LL_GATE))
        ec = torch.cuda.Event(); ec.record(cur)
        ls.wait_event(ec)
        with torch.cuda.stream(ls):
            self._ll_o0.copy_(o0v)
            self._ll_dd.copy_(dd)
        self.__dict__["_llf_keep"] = (y, X, o0, o0v, v0, o1v, v1, mn1, out, dd, m_, s_)
        _LLF_LIVE[0] = True
        return out.permute(2, 3, 0, 1).reshape(1, Co, H, W)

    # ------------------------------------------------------------------ v156g LLX (common/ll_ridge_faster_card.py)
    def _ll_epx(self, mem, y, m_, s_):
        """m1's whole output (outc + LL correction + in_norm undo) with ll_comb and the undo in ONE kernel (ll_comb_ep),
        or None (the caller then runs v156f's `_ll_outc` / `_llf_outc` and the undo ops)."""
        st = self.__dict__.get("_llx_ok")
        if st is False:
            return None
        if st is None:
            return self._llx_decide(mem, y, m_, s_)
        _LLX_LIVE[0] = True
        if self.__dict__.get("_llf_par", False):
            return self._llf_outc_ep(mem, y, m_, s_)
        return self._ll_outc_ep(mem, y, m_, s_)

    def _llx_synth(self) -> bool:
        """ll_comb_ep vs ll_comb + (out * s) + m on synthetic tensors, gate open / shut / no evidence, bitwise."""
        d = self._ll_U.device
        H, W, T, C = int(self._H), int(self._W), int(self._n_t), int(self._C)
        g = torch.Generator(device=d).manual_seed(20260927)
        r = lambda *s_: torch.randn(*s_, generator=g, device=d)
        o0 = r(H, W, T, C) * 3.0
        o1 = o0 + 0.3 * r(H, W, T, C)
        v0 = r(H, W, 1, C).abs()
        v1 = r(H, W, 1, C).abs()
        v1[0, 0] = 0.0
        mn1 = r(H, W, 1, C)
        s = r(1, 1, 1, C).abs() + 1e-3
        m = r(1, 1, 1, C)
        ok = True
        for u, k in ((1.0, 2.0), (-1.0, 2.0), (1.0, 0.0), (0.0, 1.0)):
            U = torch.full((), u, device=d)
            Kk = torch.full((), k, device=d)
            out, dd = _LL_K["ll_comb"](o0, o1, v0, mn1, v1, U, Kk, self._ll_cm, gate=float(_LL_GATE))
            yr = out * s
            yr = yr + m
            f = _LLX_K["llx_flag"](self._ll_cm, Kk, U, gate=float(_LL_GATE))
            yf, ddf = _LLX_K["ll_comb_ep"](o0, o1, v0, mn1, v1, f, s, m)
            ok = ok and _bitseq(yf, yr) and _bitseq(ddf, dd)
        torch.cuda.synchronize()
        return bool(ok)

    def _llx_decide(self, mem, y, m_, s_):
        ok, why, yf = False, "", None
        try:
            H, W, T, C = int(self._H), int(self._W), int(self._n_t), int(self._C)
            if not (tuple(s_.shape) == (1, 1, 1, 1, C) and tuple(m_.shape) == (1, 1, 1, 1, C)
                    and int(getattr(mem, "t_out", -1)) == T and getattr(mem, "in_norm", "") == "self"):
                why = "config not covered"
            elif torch.cuda.is_current_stream_capturing():
                why = "undecided at capture"
            elif self.__dict__.get("_llf_par", False):
                why = "first call inside the par4 body"
            elif not _llx_kernels():
                why = _LLX_STATE["why"] or "jiterator unavailable"
            elif not self._llx_synth():
                why = "synthetic mismatch"
            else:
                out = self._ll_outc(mem, y, m_, s_)                        # v156f's path: stash written, out returned
                yr = out.reshape(1, T, C, H, W).permute(0, 1, 3, 4, 2)
                yr = yr * s_
                yr = yr + m_
                snap = [t.clone() for t in (self._ll_phiT, self._ll_o0, self._ll_dd, self._ll_sm)]
                yf = self._ll_outc_ep(mem, y, m_, s_)                      # the fused path: the same stash rewritten
                ok = _bitseq(yf, yr) and all(_bitseq(a, b) for a, b in zip(
                    snap, (self._ll_phiT, self._ll_o0, self._ll_dd, self._ll_sm)))
                why = "synthetic + live bitwise %s" % ("equal" if ok else "DIFFERENT")
        except Exception as e:
            ok, why = False, "%s: %s" % (type(e).__name__, str(e)[:200])
        self._llx_ok = bool(ok)
        print("LLX %s (%s)" % ("on" if ok else "OFF", why), flush=True)
        if not ok:
            return None                                                   # v156f's path recomputes (idempotent stash)
        _LLX_LIVE[0] = True
        return yf

    def _ll_outc_ep(self, mem, y, m_, s_):
        """`_ll_outc`'s statements with ll_comb -> ll_comb_ep; returns m1's final (1, T, H, W, C) output (the strides the
        original undo ops allocate)."""
        oc = mem.outc
        ev = self.__dict__.get("_ll_wait")
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)
        o0 = oc(y)
        U_ = sys.modules[type(oc).__module__]
        H, W, T, C = int(self._H), int(self._W), int(self._n_t), int(self._C)
        Co, D0 = T * C, self._ll_D0
        X = U_._nhwc(y) if hasattr(U_, "_nhwc") else y[0].permute(1, 2, 0).reshape(H * W, D0)
        if getattr(oc, "_fusea", None) is True:
            o1v = _ll_lin(U_, X, self._ll_Weff, self._ll_beff).view(H, W, T, C)
        else:
            o1 = torch.nn.functional.conv2d(y, self._ll_Weff.view(Co, D0, 1, 1), self._ll_beff)
            o1v = o1[0].permute(1, 2, 0).reshape(H, W, T, C)
        o0v = o0[0].permute(1, 2, 0).reshape(H, W, T, C)
        v0, _mn0 = torch.var_mean(o0v, dim=2, unbiased=False, keepdim=True)
        v1, mn1 = torch.var_mean(o1v, dim=2, unbiased=False, keepdim=True)
        fl = _LLX_K["llx_flag"](self._ll_cm, self._ll_k, self._ll_U, gate=float(_LL_GATE))
        yo, dd = _LLX_K["ll_comb_ep"](o0v, o1v, v0, mn1, v1, fl, s_.reshape(1, 1, 1, C), m_.reshape(1, 1, 1, C))
        self._ll_phiT[:, :D0].copy_(X)
        self._ll_o0.copy_(o0v)
        self._ll_dd.copy_(dd)
        torch.stack([s_.reshape(C), m_.reshape(C)], out=self._ll_sm)
        return yo.view(1, H, W, T, C).permute(0, 3, 1, 2, 4)

    def _llf_outc_ep(self, mem, y, m_, s_):
        """`_llf_outc` (par4, LL lane) with ll_comb -> ll_comb_ep; the same lanes, events and stash placement."""
        oc = mem.outc
        U_ = sys.modules[type(oc).__module__]
        H, W, T, C = int(self._H), int(self._W), int(self._n_t), int(self._C)
        Co, D0 = T * C, self._ll_D0
        cur = torch.cuda.current_stream()
        ls = self._ll_side
        with torch.cuda.stream(ls):                   # the flag right after the prefix: before the lane waits for m1
            fl = _LLX_K["llx_flag"](self._ll_cm, self._ll_k, self._ll_U, gate=float(_LL_GATE))
        ey = torch.cuda.Event(); ey.record(cur)
        ls.wait_event(ey)
        with torch.cuda.stream(ls):
            X = U_._nhwc(y) if hasattr(U_, "_nhwc") else y[0].permute(1, 2, 0).reshape(H * W, D0)
            if getattr(oc, "_fusea", None) is True:
                o1v = _ll_lin(U_, X, self._ll_Weff, self._ll_beff).view(H, W, T, C)
            else:
                o1 = torch.nn.functional.conv2d(y, self._ll_Weff.view(Co, D0, 1, 1), self._ll_beff)
                o1v = o1[0].permute(1, 2, 0).reshape(H, W, T, C)
            v1, mn1 = torch.var_mean(o1v, dim=2, unbiased=False, keepdim=True)
            e1 = torch.cuda.Event(); e1.record(ls)
            self._ll_phiT[:, :D0].copy_(X)
        o0 = oc(y)
        o0v = o0[0].permute(1, 2, 0).reshape(H, W, T, C)
        v0, _mn0 = torch.var_mean(o0v, dim=2, unbiased=False, keepdim=True)
        cur.wait_event(e1)
        yo, dd = _LLX_K["ll_comb_ep"](o0v, o1v, v0, mn1, v1, fl, s_.reshape(1, 1, 1, C), m_.reshape(1, 1, 1, C))
        ec = torch.cuda.Event(); ec.record(cur)
        ls.wait_event(ec)
        with torch.cuda.stream(ls):                   # the o0 / dd / [s; m] stash as ONE multi-tensor copy (same bytes)
            torch._foreach_copy_([self._ll_o0, self._ll_dd, self._ll_sm[0], self._ll_sm[1]],
                                 [o0v, dd, s_.reshape(C), m_.reshape(C)])
        self.__dict__["_llf_keep"] = (y, X, o0, o0v, v0, o1v, v1, mn1, yo, dd, m_, s_, fl)
        _LLF_LIVE[0] = True
        return yo.view(1, H, W, T, C).permute(0, 3, 1, 2, 4)

    def _ll_m2_stash(self, y2) -> None:
        """Blend gate: m2's output of this window (for the next window's utility)."""
        self._ll_u2.copy_(y2[0].permute(1, 2, 0, 3))

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
            and self._ll_strpar_ok()                                    # LL-ridge card patch
        )

    def _band_branch_orig(self, n_t: int, C: int, usable, wf, x_ph=None):
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
        # LLR-fast: fork the LL work onto `_ll_side` only in par4 with the KCUT tail live and m1.outc's FUSE-A decision
        # made (both are settled by the sequential eager reference that `_capture` runs first); else v156's code.
        _llf = bool(_par4 and _LLF_ON and self.__dict__.get("_ll_on", False)
                    and self.__dict__.get("_ll_side") is not None and self._kc_ok()
                    and getattr(getattr(self.base.m1, "outc", None), "_fusea", None) is not None)
        _lle_end = None
        self.__dict__.pop("_llf_keep", None)
        self.__dict__.pop("_tp_keep", None)
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
                if self._kf_on:                             # KE-FTL (prediction-independent: previous pair + input)
                    self._kf_update(self._sy, have)
                    self._bf_update(self._sy, have)
                    self._kf_kin.copy_(self._cl_pick0(self._kf_kin_of(self._sx), self._cl_ks, self._cl_k0))
                    self._mf_xbar.copy_(self._cl_pick0(self._sx[..., :2].mean(dim=1, keepdim=True), self._cl_ms, self._cl_m0))
                if self._kc2_ema_ok():                  # kernel-cut-2: the prefix EMAs fused (bit-exact)
                    self._kc2_ema(have)
                else:
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
                usable, wf = self._kc2_uw()              # kernel-cut-2 (== the clamps)
                if self._gate_on:
                    self._gate_corr(usable, wf)          # BIASGATE (v54, fused)
                if self.__dict__.get("_ll_on", False):     # LL-ridge prefix (par3: main, before the fork)
                    self._ll_prefix(self._sy, have)

            # ---------- 2. fork ----------
            # In par4 the fork is the FIRST thing in the step: nothing prediction-independent is
            # left ahead of it, so m1 owns the main stream from its very first kernel and both
            # side streams start at the same instant.
            ef = torch.cuda.Event(); ef.record(cur)
            self._side.wait_event(ef)
            _bfd = bool(_par4 and _BFOLD_ON and self._kc_ok() and self._bf_ok())   # v156g BFOLD (decided eagerly)
            _dv = None
            self.__dict__.pop("_bf_keep", None)
            _sh = None
            self.__dict__.pop("_sh_keep", None)
            if _par4 and _SHIN_ON and self._shin_ok():      # v156g SHIN: the input mean on main, right at the fork
                _sh = dict(m=self._sx.mean(dim=(1, 2, 3), keepdim=True), ez=None)
                _sh["em"] = torch.cuda.Event(); _sh["em"].record(cur)
                self.base.m2.__dict__["_shin"] = ("prod", _sh)
            with torch.cuda.stream(self._side):
                if _bfd:
                    self.base.m2.__dict__["_epi_defer"] = True
                try:
                    y2 = self.base.m2(self._sx)
                finally:
                    self.base.m2.__dict__["_epi_defer"] = False
                    self.base.m2.__dict__.pop("_shin", None)
                _dv = self.base.m2.__dict__.pop("_epi_dv", None)
                if _dv is None:
                    e2 = torch.cuda.Event(); e2.record(self._side)
            e3 = e4 = e5 = None
            if _par4:
                if _llf:                                   # LLR-fast: the LL prefix on its own lane, forked at ef
                    self._ll_side.wait_event(ef)
                    with torch.cuda.stream(self._ll_side):
                        self._ll_prefix(self._sy, have)
                self._side2.wait_event(ef)
                with torch.cuda.stream(self._side2):
                    # ---- 1'. the prediction-independent prefix, moved off the critical path ----
                    if self.__dict__.get("_ll_on", False) and not _llf:     # LL-ridge prefix at side2's head; m1 waits on it
                        self._ll_prefix(self._sy, have)          # just before its outc (every stash read is
                        _lle = torch.cuda.Event(); _lle.record(self._side2)   # ahead of the event)
                        self.__dict__["_ll_wait"] = _lle
                    # Identical statements in identical order to the par3 block above; only the
                    # stream they are enqueued on differs, which is why this is bit-exact.
                    if self._kf_on:                             # KE-FTL (prediction-independent: previous pair + input)
                        self._kf_update(self._sy, have)
                        self._kf_kin.copy_(self._cl_pick0(self._kf_kin_of(self._sx), self._cl_ks, self._cl_k0))
                        self._mf_xbar.copy_(self._cl_pick0(self._sx[..., :2].mean(dim=1, keepdim=True), self._cl_ms, self._cl_m0))
                    if self._kc2_ema_ok():                  # kernel-cut-2: the prefix EMAs fused (bit-exact)
                        self._kc2_ema(have)
                    else:
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
                    usable, wf = self._kc2_uw()              # kernel-cut-2 (== the clamps)
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
                    if self._kf_on:                             # band FTL (off the critical path)
                        self._bf_update(self._sy, have)
                    e5 = torch.cuda.Event(); e5.record(self._side2)
            elif mode == "par3":
                self._side2.wait_event(ef)
                with torch.cuda.stream(self._side2):
                    band_ol, band_oh = self._band_branch(n_t, C, usable, wf)
                    e3 = torch.cuda.Event(); e3.record(self._side2)
            self.__dict__["_llf_par"] = _llf                # LLR-fast: m1's outc site forks (see `_llf_outc`)
            if _sh is not None and _sh.get("ez") is not None:  # v156g SHIN: m1 reuses m2's (z60, s, m)
                self.base.m1.__dict__["_shin"] = ("cons", _sh)
                self.__dict__["_sh_keep"] = _sh
                _SHIN_LIVE[0] = True
            try:
                y1 = self.base.m1(self._sx)
            finally:
                self.base.m1.__dict__.pop("_shin", None)
                self.__dict__["_ll_wait"] = None             # LL-ridge: the side2 event is m1's only
                self.__dict__["_llf_par"] = False
            _pb = None
            if _dv is not None:                              # v156g BFOLD: m2's undo + the blend, on m2's stream once y1
                _ey1 = torch.cuda.Event(); _ey1.record(cur)  # exists (m1 finishes well before m2)
                self._side.wait_event(_ey1)
                with torch.cuda.stream(self._side):
                    y2, _pb = _BF_K["ep_blend"](_dv[0], _dv[1], _dv[2], y1)
                    e2 = torch.cuda.Event(); e2.record(self._side)
                self.__dict__["_bf_keep"] = (_dv, y1, y2, _pb)
                _BFOLD_LIVE[0] = True
            if _llf:                                         # LLR-fast: m2's u/v stash on the LL lane, off the tail
                self._ll_side.wait_event(e2)
                with torch.cuda.stream(self._ll_side):
                    if _LL_GSRC == "blend":
                        self._ll_m2_stash(y2)
                    _lle_end = torch.cuda.Event(); _lle_end.record(self._ll_side)
            cur.wait_event(e2)
            if self._kc_ok():                       # kernel-cut card patch: the fused tail
                if not _par4:
                    self._kc_vg_input()             # par3: the input-only vgain half on main (par4: side2)
                if e3 is not None:
                    _bfn = lambda: (band_ol, band_oh)
                else:
                    _bfn = lambda: self._band_branch(n_t, C, usable, wf)
                self._kc_tail(y1, y2, _bfn, cur, e3, e4, e5, llu2=not _llf,
                              tp=(self._side if (_par4 and _TPAR_ON and self._side is not None) else None), pb=_pb)
                self.__dict__.pop("_tp_keep", None)
                self.__dict__.pop("_bf_keep", None)
                self.__dict__.pop("_sh_keep", None)
                if _lle_end is not None:
                    cur.wait_event(_lle_end)                 # LLR-fast: join the LL lane at the very end of the step
                self.__dict__.pop("_llf_keep", None)
                return
            p = 0.5 * (y1 + y2)
            if self.__dict__.get("_ll_on", False) and _LL_GSRC == "blend":
                self._ll_m2_stash(y2)                    # LL-ridge blend gate (original tail)
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
            _ctr = _ctr + 0.5 * (p - self._sg_pg)  # BAND_CENTER_TRANSPORT
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
            _ka_sh = None
            # --- KE-FTL anchor (common/ke_ftl_card_t2.py): v105's KE anchor, mu = the trajectory's FTL leader
            if self._kf_on and self._sx.shape[1] > 1 and tuple(p.shape[:4]) == tuple(self._kf_pp.shape[:4]):
                p, _ka_sh = self._fj_anchor(p)                                    # fused-TTT card patch (KE / time-mean anchor)
            if e5 is not None:
                cur.wait_event(e5)
            if self._kf_on and tuple(lo.shape[:4]) == tuple(self._bf_mid.shape[:4]) and lo.shape[-1] >= 2:
                lo, hi = self._fj_band(p, lo, hi, _ka_sh)                      # fused-TTT card patch (band FTL)
            p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
            lo = torch.nan_to_num(lo, nan=-1e3, posinf=-1e3, neginf=-1e3)
            hi = torch.nan_to_num(hi, nan=1e3, posinf=1e3, neginf=1e3)
            self._sg_pfin.copy_(p)
            self._out3[0].copy_(p)
            self._out3[1].copy_(torch.minimum(lo, hi))
            self._out3[2].copy_(torch.maximum(lo, hi))
            if _lle_end is not None:
                cur.wait_event(_lle_end)                     # LLR-fast: join (unreachable: _llf implies the KCUT tail)
            self.__dict__.pop("_llf_keep", None)

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
        # T2TIME: the pinned staging exists for a CPU-resident input.  When the harness already
        # hands us a tensor ON THIS DEVICE the round trip is pure waste -- a blocking D->H copy
        # (which also drains the queue) plus an H->D copy, twice a step.  Copy straight into the
        # static buffer instead; the bytes landing in _sx/_sy are identical either way.
        _xin = torch.as_tensor(input_norm)
        if _xin.is_cuda and _xin.device == self._sx.device:
            self._sx.copy_(_xin, non_blocking=True)
        else:
            self._pin_x.copy_(_xin)
            self._sx.copy_(self._pin_x, non_blocking=True)
        hp = prev_target_norm is not None
        if hp:
            _yin = torch.as_tensor(prev_target_norm)
            if _yin.is_cuda and _yin.device == self._sy.device:
                self._sy.copy_(_yin, non_blocking=True)
            else:
                self._pin_y.copy_(_yin)
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
                    if self._kf_on:                     # KE-FTL
                        self._kf_update(y, 1.0)
                        self._bf_update(y, 1.0)
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

            if self.__dict__.get("_ll_on", False):     # LL-ridge prefix (eager fallback)
                _llh = prev_target_norm is not None and self._prev_pg is not None
                self._ll_prefix(torch.as_tensor(prev_target_norm).to(self.device) if _llh else self._ll_zy,
                                self._ll_one if _llh else self._ll_zero)
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
            _ctr = _ctr + 0.5 * (p - pg)  # BAND_CENTER_TRANSPORT
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

            _ka_sh = None
            # --- KE-FTL anchor (common/ke_ftl_card_t2.py): v105's KE anchor, mu = the trajectory's FTL leader
            if self._kf_on and x.shape[1] > 1 and tuple(p.shape[:4]) == tuple(self._kf_pp.shape[:4]):
                self._kf_kin.copy_(self._cl_pick0(self._kf_kin_of(x), self._cl_ks, self._cl_k0))
                self._mf_xbar.copy_(self._cl_pick0(x[..., :2].mean(dim=1, keepdim=True), self._cl_ms, self._cl_m0))
                p, _ka_sh = self._fj_anchor(p)                                    # fused-TTT card patch (KE / time-mean anchor)
            if self._kf_on and tuple(lo.shape[:4]) == tuple(self._bf_mid.shape[:4]) and lo.shape[-1] >= 2:
                lo, hi = self._fj_band(p, lo, hi, _ka_sh)                      # fused-TTT card patch (band FTL)
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
        _llf = self.__dict__.get("_ll_m2fn")      # LL-ridge blend gate (paths that call the blend as a whole)
        if _llf is None:
            return 0.5 * (self.m1(x) + self.m2(x))
        a = self.m1(x)
        b = self.m2(x)
        _llf(b)
        return 0.5 * (a + b)


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
