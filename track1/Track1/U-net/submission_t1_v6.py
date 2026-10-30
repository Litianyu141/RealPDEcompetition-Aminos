"""RealPDE Track 1 (Sim2Real) submission -- frame-stacked U-net + post-hoc calibration + an
overlapped device->host transport, with the forward pass in half precision.

v5 = v4 with EXACTLY ONE change: the model runs in fp16 (see the `_MDTYPE` block below). The
mathematics of `_chunk` is otherwise v4's, which is itself byte-for-byte v2's prediction path, so
gamma, the heteroscedastic band, the de-normalisation and the returned dtypes are all untouched and
all still fp32. On CPU, or if `.half()` ever refuses, `_MDTYPE` stays float32 and this file is
numerically identical to v4.

v3 changes NOTHING about the mathematics. `_chunk` below is byte-for-byte the v2 predictor,
including every shape guard, so the prediction, lower and upper arrays this file returns are
bit-identical to v2's (verified with np.array_equal, not allclose). The only thing that changed is
how those arrays get from the GPU into numpy -- and that turned out to be where nearly all of the
wall clock was.

Why that matters here: `final_score` averages five subscores, one of which is
`score_time(t) = 100/(1+sqrt(t/0.72896))`. Profiling the v2 bundle on the 378-window local benchmark
gave this per-window phase split:

    h2d 4.3% | forward 5.5% | GPU post-processing 1.8% | DEVICE->HOST 80.3% | torch.cat 8.0%

The model is tiny at 32x64; the bundle was never compute-bound, it was bound on moving three
(N,20,32,64,3) float32 arrays -- 557 MB out for 186 MB in -- across PCIe into pageable host memory,
serially, with the GPU idle the whole time. Three fixes, each measured:

  1. PINNED + OVERLAPPED. A dedicated copy stream carries chunk i's D->H while the default stream
     already runs chunk i+1's forward, so the transfer hides behind compute instead of following it.
     Both endpoints are page-locked (allocated at import, outside the timed region), which is what
     makes the copy async and DMA-able at all.
  2. PREALLOCATED DESTINATION. v2 accumulated a list of chunks and paid a final torch.cat, which
     allocates a second 557 MB and copies everything twice. Writing each chunk straight into its
     slice of the final array removes that pass entirely (the 8.0% above).
  3. PARALLEL DRAIN. Once (1) and (2) land, the residual is the pinned->numpy host memcpy, which
     np.copyto does single-threaded. Draining the three arrays concurrently through torch's copy_
     (a TensorIterator: releases the GIL and is internally threaded) collapses it.

Measured on the 378-window benchmark against the PACKAGED v2 bundle, every variant checked with
np.array_equal on all three returned arrays:

    v2 shipped                    1.2225 ms/win   time_score 96.066
    + pinned/overlapped           1.0498 ms/win   time_score 96.344
    + parallel torch drain        0.2994 ms/win   time_score 98.014   <- this file
    (ceiling, full-size pool)     0.1385 ms/win   time_score 98.640

The last row is what a pinned output pool large enough for the whole batch would give -- another
+0.056 FINAL. It is deliberately NOT taken: it needs a guess at N before N is known, and a ~600 MB
page-locked allocation on hardware we cannot inspect. A failed pinned allocation is a zero, and
0.056 points is not worth any probability of that.

SAFETY. The fast path is one of three. Every failure mode -- no CUDA, pinned allocation refused, no
worker threads, an unexpected grid size, anything raising mid-flight -- degrades to a slower path
that produces the same numbers, rather than propagating. The chain is:

    fast (pinned + copy stream + parallel drain)          0.30 ms/win
      -> legacy (v2 verbatim: chunk list + torch.cat)     1.31 ms/win
        -> prealloc (D->H straight into the destination)  5.17 ms/win

That order is measured, not intuited. `prealloc` reads like the obvious middle rung -- it skips the
torch.cat pass, which was 8% of v2 -- but a D->H copy into PAGEABLE memory cannot DMA: the driver
stages it through an internal bounce buffer, synchronously, per call. Measured, that costs 4x more
than the cat pass ever saved. So legacy goes second (it is also the exact code already scored on the
platform, i.e. the best-understood floor), and prealloc goes last, kept only because it is the one
path that never holds two full copies of the output at once -- if legacy dies, memory is the likely
reason, and prealloc is then the rung that can still finish.

Contract (Track 1 starting kit v6):
    predict(input_array, metadata=None)
        input_array : (N, T_in=20, H=32, W=64, C=3) PHYSICAL, channel order [u, v, p]
        returns     : {"prediction": (N, 20, 32, 64, 3), "lower": ..., "upper": ...}, all PHYSICAL

v6 = v5 (fp16 forward) + a LEARNED variance-deficit field in place of the calibration's static
`gamma`.  Both are the same operation, `p -> mean_t(p) + g*(p - mean_t(p))`; the only difference is
that `g` is now predicted per sample from the input window by a 30k-parameter CNN instead of being
one field fitted offline for the whole dataset.  Measured out-of-fold: +0.439 FINAL over the best
static field on the base-96 arm, t = 4.52, 5/5 held-out AoA folds.  `mvpe` is EXACTLY invariant
under it (the temporal mean is preserved by construction), so only `rel_l2`, `tke` and `sps` move.

Absent `vgain.pth`, or if anything in the head raises, this file falls back to v5's static gamma and
is v5 exactly -- the head is never allowed to take the submission down.
"""
from __future__ import annotations
import os
import numpy as np
import torch

from unet_model import build_unet, build_from_meta

_HERE = os.path.dirname(os.path.abspath(__file__))
_CKPT = os.path.join(_HERE, "model.pth")
_STATS = os.path.join(_HERE, "mean_std_real.pt")
_CALIB = os.path.join(_HERE, "calib.npz")
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HALFWIDTH = __HALFWIDTH__   # fallback constant NORMALIZED half-width, used iff calib.npz is unusable
CHUNK = __CHUNK__           # max samples per forward; a short final chunk is fine (benchmark=False)


def _load_stats(path):
    # mean_std_real.pt = (mean_in, mean_tgt, std_in, std_tgt); for real data mean_in == mean_tgt.
    # Channel order [u, v, p]. Pressure std is 0 for real data -> avoid divide-by-zero.
    mi, mt, si, st = torch.load(path, map_location="cpu", weights_only=False)
    to = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32).reshape(1, 1, 1, 1, -1).copy())
    mean_in, std_in, mean_t, std_t = to(mi), to(si), to(mt), to(st)
    std_in[std_in == 0] = 1.0
    std_t[std_t == 0] = 1.0
    return mean_in, std_in, mean_t, std_t


# --------------------------------------------------------------------------- #
# Everything below runs at IMPORT time, i.e. outside the timed region.
# --------------------------------------------------------------------------- #
_MEAN_IN, _STD_IN, _MEAN_T, _STD_T = [t.to(_DEVICE) for t in _load_stats(_STATS)]

_ckpt = torch.load(_CKPT, map_location="cpu", weights_only=False)
_state = _ckpt["model_state_dict"] if isinstance(_ckpt, dict) and "model_state_dict" in _ckpt else _ckpt
_meta = _ckpt.get("meta", {}) if isinstance(_ckpt, dict) else {}
_MODEL = build_from_meta(_meta)
_MODEL.load_state_dict(_state, strict=True)
_MODEL.to(_DEVICE).eval()

# --- v5: run the FORWARD PASS in half precision, and nothing else -------------------------- #
# Measured out-of-fold in both precisions on all 6 Reynolds folds (`local_benchmark/eval_kfold.py
# --dtype fp16`): the accuracy subscores agree with fp32 to the 5th decimal (A 89.375 -> 89.376,
# E 89.497 -> 89.497), while the forward pass costs 0.63x the time. On a scorer that averages in
# `score_time`, that is +0.063 FINAL for zero accuracy -- and, because at 32x64 this net is
# LATENCY-bound rather than FLOP-bound, it is also what makes a 2.25x wider model (base 96) cost
# no more wall clock than the shipped fp32 base 64. See the U-net README, "fp16 + more width".
#
# The cast is deliberately confined to the model call. Everything downstream -- the gamma
# inflation, the band arithmetic, the de-normalisation, the returned arrays -- stays float32, so
# the calibration is applied in exactly the precision it was fitted in and the returned dtype is
# unchanged. CPU stays fp32: half on CPU is slower, not faster.
# --- v6: a LEARNED, sample-dependent variance-deficit field, replacing the static gamma -------- #
# `scoring.kinetic_energy` subtracts the same temporal mean the correction preserves, so
# `KE(pbar + g*(p-pbar)) = g**2 * KE(p)` EXACTLY.  The shipped calibration corrects that with ONE
# STATIC field; measured out-of-fold on the 5 AoA folds of the base-96 arm, a 30k-parameter head
# that predicts the field PER SAMPLE from the input window beats the best static field by +2.196 on
# (score_rel + score_tke) = +0.439 FINAL, t = 4.52, 5/5 folds.  See Track1/U-net/README.md and
# vgain_head.py.  The head REPLACES the static gamma rather than stacking on it -- the static field
# is exactly the baseline it was measured against.
_HEAD = None
try:
    _HP = os.path.join(_HERE, "vgain.pth")
    if os.path.exists(_HP):
        from vgain_head import VGainHead, N_FEAT as _NFEAT
        _hd = torch.load(_HP, map_location="cpu", weights_only=False)
        _HEAD = VGainHead(int(_hd.get("n_feat", _NFEAT)), int(_hd["width"]),
                          float(_hd["gmin"]), float(_hd["gmax"]))
        _HEAD.load_state_dict(_hd["state_dict"])
        _HEAD.to(_DEVICE).eval()
except Exception:
    _HEAD = None

_MDTYPE = torch.float32
if _DEVICE.type == "cuda":
    try:
        _MODEL.half()
        _MDTYPE = torch.float16
    except Exception:
        _MODEL.float()
        _MDTYPE = torch.float32

# Calibration is optional and must never be able to take the submission down.
_GAMMA = _BAND_LO = _BAND_HI = None
try:
    if os.path.exists(_CALIB):
        with np.load(_CALIB) as z:
            g = np.asarray(z["gamma"], np.float32)
            _lo = np.asarray(z["band_lo"], np.float32)
            _hi = np.asarray(z["band_hi"], np.float32)
            _lo, _hi = np.minimum(_lo, _hi), np.maximum(_lo, _hi)   # enforce lower <= upper at load
            _GAMMA = torch.as_tensor(g.reshape(1, 1, *g.shape)).to(_DEVICE)
            _BAND_LO = torch.as_tensor(_lo.reshape(1, _lo.shape[0], 1, 1, _lo.shape[1])).to(_DEVICE)
            _BAND_HI = torch.as_tensor(_hi.reshape(1, _hi.shape[0], 1, 1, _hi.shape[1])).to(_DEVICE)
except Exception:
    _GAMMA = _BAND_LO = _BAND_HI = None

# ---- v4: the heteroscedastic band. A SECOND, INDEPENDENT try/except on purpose ------------------
# These five keys did not exist before v4, so a v7-era calib.npz has none of them. Folding this into
# the block above would mean one KeyError takes the gamma and the band down with it and silently
# demotes the whole submission to the HALFWIDTH fallback. Loaded separately, the worst case here is
# that _ACT_C stays None and the file behaves EXACTLY like v3.
_ACT_C = _KCELL = None
_ACT_P = _ACT_LAM = 1.0
_ACT_SELF = False
try:
    if os.path.exists(_CALIB) and _BAND_LO is not None:
        with np.load(_CALIB, allow_pickle=True) as z:
            if "k_cell" in z.files and "act_const" in z.files:
                _kc = np.asarray(z["k_cell"], np.float32)                    # (LT, c)
                _ac = np.asarray(z["act_const"], np.float32).ravel()         # (c,)
                _ACT_P = float(np.asarray(z["p"]).ravel()[0])
                _ACT_LAM = float(np.asarray(z["lam"]).ravel()[0])
                _ACT_SELF = bool(np.asarray(z["self_norm"]).ravel()[0])
                nc = int(_lo.shape[1])
                assert _kc.shape == (_lo.shape[0], nc) and _ac.shape == (nc,)
                assert np.isfinite(_kc).all() and np.isfinite(_ac).all() and (_ac > 0).all()
                assert (_kc > 0).all()
                # The band arrays carry nc channels; the model emits 3. Pad the trailing channels
                # with the identity so pressure -- which is zeroed and re-bounded downstream anyway
                # -- cannot be reached by a scale fitted on velocities.
                _KCELL = torch.as_tensor(_kc.reshape(1, _kc.shape[0], 1, 1, nc)).to(_DEVICE)
                _ACT_C = torch.as_tensor(_ac.reshape(1, 1, 1, nc)).to(_DEVICE)
except Exception:
    _ACT_C = _KCELL = None
    _ACT_P = _ACT_LAM = 1.0
    _ACT_SELF = False

_IN_TAIL = (20, 32, 64, 3)      # expected input tail; corrected from the warm-up below
_OUT_TAIL = (20, 32, 64, 3)     # expected output tail; corrected from the warm-up below
try:
    # benchmark=False on purpose: with autotuning ON, every distinct batch size pays a fresh
    # algorithm search inside the timed region, which is exactly what a ragged final chunk causes.
    torch.backends.cudnn.benchmark = False
    with torch.inference_mode():
        _w = torch.zeros((CHUNK,) + _IN_TAIL, dtype=_MDTYPE, device=_DEVICE)
        _o = _MODEL(_w)     # CUDA context init + cuDNN handles + allocator blocks, all paid here
        _OUT_TAIL = tuple(int(v) for v in _o.shape[1:])
        if _DEVICE.type == "cuda":
            torch.cuda.synchronize()
    del _w, _o
except Exception:
    pass


def _activity(xb, nc):
    """Per-element band scale s from the INPUT window alone: (B, 1, H, W, nc), or None.

    This is the only thing v4 adds to the mathematics, and it is computable at test time because
    predict() is handed the 20 input frames. s is the temporal standard deviation of each input
    pixel -- near zero in the freestream and inside the body, large in the shear layer -- rescaled
    to a mean of one and then softened toward one by lam.

    Two details that are not free choices:

      * unbiased=False. The calibration measured the activity with numpy's .std(), which is ddof=0.
        torch's default is ddof=1, a uniform factor sqrt(20/19) = 1.0260 on every element. It would
        pass straight through act_const and land as a ~1.3% miscalibration of every band width. It
        would also never raise.
      * the reduction is over the PHYSICAL xb, not the normalised input, because act_const was
        fitted in physical units.
    """
    if _ACT_C is None:
        return None
    a = xb[..., :nc].std(dim=1, unbiased=False).clamp_min(0.0)        # (B, H, W, nc)
    if _ACT_SELF:
        a = a / a.mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    else:
        a = a / _ACT_C
    s = a.pow(_ACT_P).clamp(0.2, 5.0)
    s = (1.0 - _ACT_LAM) + _ACT_LAM * s
    return s.unsqueeze(1)                                             # (B, 1, H, W, nc)


def _vg_features(p_ph, x_ph, eps=1e-12):
    """The 7 summary fields VGainHead consumes, from PHYSICAL (B,T,H,W,>=2) tensors.

    Must stay bit-compatible with `vgain_head.stats_from_window`: same channel order, same eps,
    same definition of KE (temporal variance averaged over t, u and v halved and summed).  A drift
    here would not raise -- it would just feed the head garbage and silently mis-scale every
    forecast -- so it is duplicated deliberately rather than imported, and `t1_vgain_verify.py`
    asserts the two agree before any bundle is built.
    """
    p2, i2 = p_ph[..., :2], x_ph[..., :2]
    pb, ib = p2.mean(1), i2.mean(1)
    pf = p2 - pb.unsqueeze(1)
    KEp = 0.5 * (pf * pf).mean(1).sum(-1)
    KEi = 0.5 * ((i2 - ib.unsqueeze(1)) ** 2).mean(1).sum(-1)
    lp, li = torch.log(KEp + eps), torch.log(KEi + eps)
    return torch.stack([lp, li, li - lp, pb[..., 0], pb[..., 1], ib[..., 0], ib[..., 1]], dim=1)


def _chunk(xb):
    """One chunk, physical in -> (pred, lower, upper) physical out, never leaving the device.

    The PREDICTION path is still byte-for-byte v2/v3 -- the model call and the gamma inflation below
    are untouched, so rel_l2, tke and mvpe are bit-identical to v7 and only sps and time can move.
    v4's single change is that the interval is no longer constant over space and samples.
    """
    # v5: normalise in fp32 (the stats are fp32), hand the model its own dtype, come straight back
    # to fp32. `.to()` is a no-op when _MDTYPE is float32, so the fp32 path is byte-identical to v4.
    pn = _MODEL(((xb - _MEAN_IN) / _STD_IN).to(_MDTYPE)).float()

    # v6: the learned field, else v5's static one. `g` multiplies the fluctuation, which scales
    # linearly under de-normalisation, so applying it here in normalised space is identical to
    # applying it to the physical field -- but the FEATURES must be physical, because that is where
    # the head was trained and where scoring.py computes KE.
    _done = False
    if _HEAD is not None and pn.shape[2:4] == xb.shape[2:4]:
        try:
            g = _HEAD(_vg_features(pn * _STD_T + _MEAN_T, xb))
            pm = pn.mean(dim=1, keepdim=True)
            pn = pm + (pn - pm) * g[:, None, :, :, None]
            _done = True
        except Exception:
            _done = False
    if not _done and _GAMMA is not None and pn.shape[2:] == _GAMMA.shape[2:]:
        pm = pn.mean(dim=1, keepdim=True)
        pn = pm + (pn - pm) * _GAMMA

    # per-(lead time, channel) asymmetric interval, else the constant fallback
    if (_BAND_LO is not None and pn.shape[1] == _BAND_LO.shape[1]
            and pn.shape[-1] == _BAND_LO.shape[-1]):
        lo_b, hi_b = _BAND_LO, _BAND_HI
        # v4: widen where the input says the flow is active, narrow where it is quiet. Guarded on
        # the input grid matching the output grid, because the activity field is built from xb and
        # broadcast against pn; if a platform ever feeds a different input resolution this silently
        # degrades to the v3 homoscedastic band instead of raising inside the timed region.
        if _KCELL is not None and xb.shape[2:4] == pn.shape[2:4] and xb.shape[1] > 1:
            s = _activity(xb, int(_BAND_LO.shape[-1]))
            if s is not None:
                lo_b, hi_b = lo_b * _KCELL * s, hi_b * _KCELL * s
        lo_n, hi_n = pn + lo_b, pn + hi_b
    else:
        lo_n, hi_n = pn - HALFWIDTH, pn + HALFWIDTH

    pred = pn * _STD_T + _MEAN_T
    lo = lo_n * _STD_T + _MEAN_T
    hi = hi_n * _STD_T + _MEAN_T

    # pressure is unmeasured for real data: measured_channels(target) == 2, so channel 2 is sliced
    # off by every subscore. Zero it exactly as the scored v1 baseline did, and keep its bounds
    # well-ordered around that zero so the submission stays valid even if the scorer ever sees c==3.
    if pred.shape[-1] > 2:
        half = 0.5 * torch.abs(hi[..., 1] - lo[..., 1])
        pred = torch.cat([pred[..., :2], torch.zeros_like(pred[..., 2:3])], dim=-1)
        lo = torch.cat([lo[..., :2], -half.unsqueeze(-1)], dim=-1)
        hi = torch.cat([hi[..., :2], +half.unsqueeze(-1)], dim=-1)

    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    lo = torch.nan_to_num(lo, nan=-1e3, posinf=-1e3, neginf=-1e3)
    hi = torch.nan_to_num(hi, nan=1e3, posinf=1e3, neginf=1e3)
    return pred, torch.minimum(lo, hi), torch.maximum(lo, hi)


# --------------------------------------------------------------------------- #
# Transport scaffolding -- also import time, also outside the timed region.
# _FAST stays False unless every single piece came up, so a partial failure here can only cost
# speed, never correctness.
# --------------------------------------------------------------------------- #
_FAST = False
_STG = _STG_T = _XBUF = _CPY = _EXE = None
try:
    if _DEVICE.type == "cuda":
        from concurrent.futures import ThreadPoolExecutor
        # 3 output staging buffers (one per returned array) + 2 input buffers (double-buffered so
        # chunk i+1's host->pinned fill cannot race chunk i's still-in-flight H2D).
        _STG_T = [torch.empty((CHUNK,) + _OUT_TAIL, dtype=torch.float32, pin_memory=True)
                  for _ in range(3)]
        _XBUF = [torch.empty((CHUNK,) + _IN_TAIL, dtype=torch.float32, pin_memory=True)
                 for _ in range(2)]
        _CPY = torch.cuda.Stream()
        _EXE = ThreadPoolExecutor(max_workers=3)
        list(_EXE.map(lambda j: j, range(3)))       # pay thread creation at import, not in predict
        # Exercise the copy stream once so its first real use pays no lazy init.
        with torch.inference_mode():
            _d = torch.zeros((1,) + _OUT_TAIL, dtype=torch.float32, device=_DEVICE)
            _CPY.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(_CPY):
                _STG_T[0][:1].copy_(_d, non_blocking=True)
            _CPY.synchronize()
        del _d
        _STG = [t.numpy() for t in _STG_T]
        _FAST = True
except Exception:
    _FAST = False
    _STG = _STG_T = _XBUF = _CPY = _EXE = None


def _blank(n):
    return [np.empty((n,) + _OUT_TAIL, dtype=np.float32) for _ in range(3)]


def _pack(out):
    return {"prediction": out[0], "lower": out[1], "upper": out[2]}


def _predict_fast(x_np, n):
    """Pinned, overlapped, parallel-drained. Chunk i's D->H rides a dedicated copy stream while the
    default stream is already computing chunk i+1.

    Three hazards are handled explicitly, because each one is a silent-corruption bug rather than a
    crash:
      1. the pinned INPUT buffer is double-buffered and its H2D retired by an event before refill;
      2. the pinned OUTPUT staging is drained into numpy BEFORE the next chunk's copies are queued;
      3. the on-device result tensors are record_stream'd onto the copy stream, so the caching
         allocator cannot hand their memory to the next forward while the DMA is still reading it.
    """
    out = _blank(n)
    xt = torch.from_numpy(x_np)
    cur = torch.cuda.current_stream()
    xev = [torch.cuda.Event(), torch.cuda.Event()]
    armed = [False, False]
    pend = None

    def drain(i, k):
        list(_EXE.map(lambda j: torch.from_numpy(out[j][i:i + k]).copy_(_STG_T[j][:k]), range(3)))

    with torch.inference_mode():
        for b, i in enumerate(range(0, n, CHUNK)):
            k, sl = min(CHUNK, n - i), b % 2
            if armed[sl]:
                xev[sl].synchronize()
            _XBUF[sl][:k].copy_(xt[i:i + k])
            xg = _XBUF[sl][:k].to(_DEVICE, non_blocking=True)
            cur.record_event(xev[sl])
            armed[sl] = True

            res = _chunk(xg)
            if tuple(res[0].shape[1:]) != _OUT_TAIL:
                raise RuntimeError("output tail changed; staging buffers do not fit")
            done = torch.cuda.Event()
            cur.record_event(done)

            if pend is not None:                       # overlaps with the work just enqueued
                pi, pk, pev = pend
                pev.synchronize()
                drain(pi, pk)

            _CPY.wait_event(done)
            with torch.cuda.stream(_CPY):
                for j in range(3):
                    res[j].record_stream(_CPY)
                    _STG_T[j][:k].copy_(res[j], non_blocking=True)
                cev = torch.cuda.Event()
                _CPY.record_event(cev)
            pend = (i, k, cev)

    pi, pk, pev = pend
    pev.synchronize()
    drain(pi, pk)
    return _pack(out)


def _predict_prealloc(x_np, n):
    """No pinned memory, no threads, no extra stream: still skips the torch.cat pass by writing each
    chunk straight into its slice. This is the path a CPU-only or memory-refusing host lands on."""
    xt = torch.from_numpy(x_np)
    out = None
    with torch.inference_mode():
        for i in range(0, n, CHUNK):
            k = min(CHUNK, n - i)
            res = _chunk(xt[i:i + k].to(_DEVICE, non_blocking=True))
            if out is None:
                tail = tuple(int(v) for v in res[0].shape[1:])
                out = [np.empty((n,) + tail, dtype=np.float32) for _ in range(3)]
            for j in range(3):
                torch.from_numpy(out[j][i:i + k]).copy_(res[j])
    return _pack(out)


def _predict_legacy(x_np, n):
    """v2 verbatim. Slowest, fewest assumptions -- the floor of the chain."""
    x = torch.from_numpy(x_np)
    P, L, U = [], [], []
    with torch.inference_mode():
        for i in range(0, n, CHUNK):
            p, lo, hi = _chunk(x[i:i + CHUNK].to(_DEVICE, non_blocking=True))
            P.append(p.to("cpu", non_blocking=False))
            L.append(lo.to("cpu", non_blocking=False))
            U.append(hi.to("cpu", non_blocking=False))
    cat = lambda parts: torch.cat(parts, dim=0).numpy().astype(np.float32, copy=False)
    return {"prediction": cat(P), "lower": cat(L), "upper": cat(U)}


def _quiesce():
    """After a fast-path abort the copy stream may still hold queued work referencing tensors we are
    about to drop. Block until the device is idle before retrying on a simpler path."""
    try:
        if _DEVICE.type == "cuda":
            torch.cuda.synchronize()
    except Exception:
        pass


def predict(input_array, metadata=None):
    """(N, 20, 32, 64, 3) physical -> dict of three (N, 20, 32, 64, 3) physical arrays."""
    x_np = np.asarray(input_array, dtype=np.float32)
    if not x_np.flags["C_CONTIGUOUS"]:
        x_np = np.ascontiguousarray(x_np)
    n = int(x_np.shape[0])
    if n == 0:
        z = np.zeros((0,) + _OUT_TAIL, dtype=np.float32)
        return {"prediction": z, "lower": z.copy(), "upper": z.copy()}

    if _FAST and tuple(x_np.shape[1:]) == _IN_TAIL:
        try:
            return _predict_fast(x_np, n)
        except Exception:
            _quiesce()
    try:
        return _predict_legacy(x_np, n)
    except Exception:
        _quiesce()
    return _predict_prealloc(x_np, n)


# --------------------------------------------------------------------------- #
# Final import-time act: drive the REAL predict() once, on a throwaway batch.
#
# The earlier warm-up only ran _MODEL, so it paid for cuDNN and the convolution workspaces but not
# for anything downstream of them -- the gamma broadcast, the band add, the de-normalise, the two
# torch.cat calls, nan_to_num, minimum/maximum. Each of those wants its own caching-allocator block,
# and on a first call every one of them is a fresh cudaMalloc INSIDE the timed region. Two chunks
# are used rather than one so the double-buffer swap, the deferred drain of the previous chunk, and
# the copy stream's wait_event all execute at least once here too.
#
# Measured on the local benchmark this closes most of the cold-call penalty: the first predict() was
# 2x the steady-state cost, and essentially all of that gap was allocator traffic, not compute.
# --------------------------------------------------------------------------- #
try:
    _nw = 2 * CHUNK
    predict(np.zeros((_nw,) + _IN_TAIL, dtype=np.float32))
    if _DEVICE.type == "cuda":
        torch.cuda.synchronize()
    del _nw
except Exception:
    pass
