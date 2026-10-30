#!/usr/bin/env python3
"""Stage-wise U-net trainer v4 = v3 + an MVPE loss term + GROUPED K-FOLD training.

v3 is left untouched so `unet_v3_rel_tke20_s1` (and the v5..v8 bundles built on it) stay
reproducible.  With `mvpe_weight: 0.0` and no `fold`, this file reproduces v3 step for step.

WHY THESE TWO ADDITIONS (both come straight out of the 2026-07-28 leaderboard analysis, see
`Track1/U-net/README.md`):

(4) `mvpe_weight`  --  mvpe is 1/5 of the final score and NOTHING in the objective has ever
    targeted it.  `tke_weight` exists for the tke subscore; this is the exact analogue for
    mvpe.  mvpe is an unusually easy target to attack directly because it is a very
    low-dimensional functional of the prediction: the TIME-AVERAGED u,v profile at 4 probe
    columns behind the airfoil x 9 y-positions x 2 channels.  Online we sit at mvpe 93.288
    (raw err 0.144); 96 would be worth +0.54 FINAL.

    UNITS MATTER HERE.  scoring.py computes mvpe in PHYSICAL space and normalises by the
    target's own norm.  In normalised space the u-channel mean is ~0, so that denominator
    nearly vanishes and the term explodes.  So this loss DENORMALISES with the stage's own
    (mean, std) before computing it -- which is why `mvpe_weight > 0` requires a stage with
    exactly ONE source (with two sources a batch mixes normalisations and there is no single
    correct denormalisation).  Stages that violate this get the term switched off, loudly.
    A startup self-check compares the torch port against the kit's own scoring.py.

(5) `kfold_manifest` + `fold`  --  the 2-Re/9-file/378-window val split was PROVEN to
    mis-rank models (it preferred the v3 forecast; the hidden test preferred v2 by 0.97, and
    the v2-based bundle then scored +0.284 on the leaderboard).  Passing `--fold k` takes the
    held-out GROUP VALUES and the val file list from a manifest under `common/local_benchmark/`, so K
    runs cover every one of the 82 real files exactly once out-of-fold.  The checkpoint path,
    variant and desc all get an `_f{k}` suffix so the K runs never overwrite each other.

    The manifest's `group_by` picks the holdout AXIS, and the axis is the whole point:
      re   `splits_kfold.json`, K=6 x 3 Reynolds  -- tests INTERPOLATION in Re.  MEASURED
           2026-07-28: this does NOT fix the mis-ranking (it prefers the v3 objective by
           +0.53 S3 at paired t=9.77, 6/6 folds, while the hidden test prefers v2).
      aoa  `splits_aoa.json`, K=5 x 1 angle of attack -- tests REGIME SHIFT.  Motivated by the
           measured public->hidden error inflation of the SAME model: tke x1.07-1.16, rel x1.55,
           but mvpe x2.2-2.6, and the model-to-model mvpe GAP inflates 27x.  mvpe is a
           wake-profile metric at 4 fixed streamwise stations, so AoA -- which moves the wake --
           is the axis that should stress it.

--------------------------------------------------------------------------------------
(1) LOSS   `loss_type: mse | l2 | rel_l2`   [+ LWa `lead_weight: "" | linear:k | late:k`]
--------------------------------------------------------------------------------------
    mse     mean(err^2) over every element -- the v2 objective.
    l2      TRUE L2 norm: sum the squared errors of a SAMPLE, then root it, then average
            over the batch.  This is `LpLoss(p=2).abs` in neuraloperator / the FNO codes.
    rel_l2  the same, divided by the sample's own target norm.  This is literally the
            competition's `rel_l2` subscore.

    Why this is not just cosmetics.  Write E = sqrt(sum_i e_i^2) for one sample.

        d(MSE)/de_i = 2 e_i / N          -> shrinks as the sample gets accurate
        d(L2)/de_i  = e_i / E            -> scale-INVARIANT in e

    MSE's gradient decays quadratically as a sample is fitted, so late in training the
    optimiser stops caring about the samples it has almost solved -- which is exactly
    where the remaining error lives in the small scales.  The L2 gradient is homogeneous
    of degree 0: an already-good sample still contributes a full-magnitude gradient
    direction.  That is the mechanism by which L2 resists the conditional-mean collapse
    that shows up in our scoreboard as tke=70.8 while rel_l2=94.3.

    `l2_normalize: true` (default) divides by sqrt(numel_per_sample), i.e. reports and
    back-props sqrt(mean(err^2)) = per-sample RMSE.  This is a CONSTANT rescale: it does
    not change the gradient direction, the per-sample weighting, or anything else that
    matters -- it only keeps the loss on the same order as the MSE run so `lr` and
    `grad_clip` stay comparable.  With it off, gradients are ~sqrt(N) = 286x larger and
    grad_clip=1.0 would silently clip on every single step.

--------------------------------------------------------------------------------------
(2) UNROLLING   `n_roll: K`, `unroll_mode`, `unroll_warmup`
--------------------------------------------------------------------------------------
    The net is a block map: 20 frames in -> the next 20 frames out.  Unrolling feeds its
    own output back in to predict block k+1, for K blocks total.

    NOTE ON WHAT THIS IS FOR HERE.  Neither track rolls out autoregressively at inference
    -- Track1 `predict()` is one shot per sample and Track2 `ttt_step` is teacher-forced
    on the revealed target.  So this is NOT the usual "fix error accumulation" argument;
    that failure mode does not exist in our evaluation.  The reason to do it anyway is
    that a blurred prediction is an INVALID INPUT: it lies off the data manifold the
    encoder was fitted on.  Demanding that the model's own output still work as its own
    input penalises destroying small-scale content, which single-step MSE never does
    because the conditional mean is the optimum for one step.  Read it as a spectral
    self-consistency regulariser, not as a stability fix.

    unroll_mode:
      detach       loss on EVERY block, state detached between blocks.  No BPTT cost,
                   every block supervised.  Default.
      pushforward  blocks 0..K-2 run under no_grad with no loss, loss on the final block
                   only (Brandstetter et al. / PDEArena).  Cheapest, supervises least.
      bptt         loss on every block, gradients flow through the whole chain
                   (tum-pbs/unrolling).  Most expensive, can be unstable.

    unroll_warmup: fraction of a stage trained single-step before unrolling engages.
    Unrolling from step 0 feeds garbage back and diverges.

--------------------------------------------------------------------------------------
(3) `tke_weight` (default 0.0 -- OFF, does not perturb 1 or 2)
--------------------------------------------------------------------------------------
    Optional auxiliary term matching scoring.py's `tke_rel_l2_per_sample` exactly:
    the relative L2 of the per-pixel TEMPORAL-VARIANCE field.  Off by default; available
    because tke is our weakest subscore and it is the one metric no reweighting of a
    pointwise loss can reach directly.

`val_rel_l2` is computed on K=1 windows regardless of n_roll, so the primary number stays
directly comparable to v2.  Without `--fold` the val split is the legacy 2-Re one, unchanged.

Usage:  python train_v4.py --config configs/config_v4_kfold.yaml --fold 0 [--gpu 6]
"""
from __future__ import annotations
import os, sys, argparse, json, glob, math, datetime, shutil, importlib.util, re as _re
import zlib
from pathlib import Path
import numpy as np
import yaml

CASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(CASE_DIR.parents[1] / "common"))
from repo_paths import resolve_paths   # configs and split manifests carry repo-relative paths

# OFFICIAL EXCLUSION (RealPDE Committee email, 2026-08-13): train_real/7575_0.h5 "incorrectly
# reused data from 6300_0"; participants must not use it.  This is an UNCONDITIONAL rules gate,
# not a config knob.  It is scoped to REAL sources by parent-directory name ("train_real",
# "test_real"): data/RealPDE_competition_dataset/gdrive_raw/extracted/train_sim/train_sim/7575_0.h5 is a DIFFERENT,
# legitimate CFD trajectory of the same name and the notice bans only the real one.
# Same ban as common/local_benchmark/make_splits.py (commit c097d32); this is the training-time
# backstop for any glob or stale manifest that still reaches the banned file.
BANNED_REAL = {"7575_0.h5"}


def _drop_banned_real(files, logger, label=""):
    """Filter the official 2026-08-13 exclusion out of a REAL file list, loudly."""
    drop = [f for f in files
            if os.path.basename(f) in BANNED_REAL
            and "real" in os.path.basename(os.path.dirname(f)).lower()]
    for f in drop:
        logger.info(f"  {label}EXCLUDED (official notice 2026-08-13): {os.path.basename(f)}")
    return [f for f in files if f not in drop]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--gpu", default=None)
    p.add_argument("--fold", type=int, default=None,
                   help="grouped fold index from the config's kfold_manifest (splits_kfold.json "
                        "= Reynolds folds, splits_aoa.json = AoA folds); overrides exclude_re "
                        "and the val file list")
    return p.parse_args()


def setup_logger(run_dir):
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")
    logger.add(str(run_dir / "run.log"), level="DEBUG", enqueue=True,
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}")
    return logger


def apply_fluc_scale(arr, alpha):
    """SIMDAMP: per-case temporal fluctuation damping, arr <- m + alpha*(arr - m) per pixel.

    m is the WHOLE-TRAJECTORY per-pixel mean of the array as loaded, so the transform preserves
    every pixel's time mean exactly (and therefore the per-source normalisation mean), scales
    the fluctuation std by alpha, and keeps overlapping windows bit-consistent because it is
    applied once per file before any windowing.  alpha == 1 is a bitwise no-op, which is what
    makes the C0 null cell an exact path control.  Applied to u and v only; p is never touched.
    """
    alpha = float(alpha)
    if not (np.isfinite(alpha) and alpha > 0.0):
        raise SystemExit(f"SIMDAMP: non-finite or non-positive alpha {alpha!r}")
    if alpha == 1.0:
        return arr
    m = arr.mean(axis=0, keepdims=True)
    return (m + alpha * (arr - m)).astype(arr.dtype)


def load_stats(stats_path):
    import torch
    mi, mt, si, st = torch.load(stats_path, weights_only=False)
    mean = np.asarray(mt, np.float32).copy()
    std = np.asarray(st, np.float32).copy()
    std[std == 0] = 1.0
    return mean, std


def file_re(fn):
    """Reynolds number encoded as the leading integer of the filename (e.g. 11400_0.h5)."""
    m = _re.match(r"(\d+)", Path(fn).name)
    return int(m.group(1)) if m else None


def file_aoa(fn):
    """Angle of attack: the integer after the underscore (e.g. 11400_15.h5 -> 15)."""
    m = _re.match(r"\d+_(\d+)", Path(fn).name)
    return int(m.group(1)) if m else None


# a fold manifest declares which axis it groups on; the trainer excludes the fold's held-out
# values of THAT axis from every source.  Reynolds folds test interpolation in Re; AoA folds
# test regime shift, which is what the mvpe probes are sensitive to.
GROUP_FN = {"re": file_re, "aoa": file_aoa}


LOADED_CH_STD = {}     # label -> per-channel std of the LOADED array (defect-4 guard)
LOADED_WIN_RE = {}     # label -> per-window trajectory Re (int64, filename-derived; -1 if
                       # unparseable).  WRa metadata side-channel: written unconditionally by
                       # build_windows (pure recording, no RNG, no effect on X/Y), read only
                       # when `regime_weight > 0` arms the real-finetune sampling tilt.
LOADED_WIN_AOA = {}
LOADED_WIN_KEY = {}    # CLUSTERW knob (2026-09-10): label -> per-window key '<basename>|dy|dx|t0' in X order (pure recording)
LOADED_FRAME_STEP = {}   # SIMDT: label -> {spec, realised k histogram}    # label -> per-window trajectory AoA (int64, filename-derived; -1 if
                       # unparseable).  AOSIM side-channel, same contract as LOADED_WIN_RE
                       # above (pure recording, no RNG, no effect on X/Y), read only when
                       # `sim_aoa_weight != 1.0` arms the sim-pretrain sampling tilt.


def physguard_build(model, Xtr, Ytr, lch, n_samples, tau, max_dim, min_numel, alpha, logger):
    """PhysGuard phase 1 (arXiv 2606.16602, Alg. 1): protected per-layer subspaces from the
    Gram matrix of N per-sample SIM-loss gradients at the pretrained weights.  Loss-level
    Fisher on plain per-window MSE over `lch` channels (deliberate simplification, see
    Ref_papers/physguard_fisher_2606.16602/NOTES.md).  Returns {param_name: U (d x k)}."""
    import torch          # this file imports torch per-function, not at module scope
    dev = next(model.parameters()).device
    idx = np.random.default_rng(0).choice(Xtr.shape[0],
                                          size=min(int(n_samples), Xtr.shape[0]), replace=False)
    names = [n for n, p in model.named_parameters()
             if p.requires_grad and p.numel() >= int(min_numel)]
    G = {n: [] for n in names}
    for i in idx:
        xb = Xtr[int(i):int(i) + 1]
        yb = Ytr[int(i):int(i) + 1]
        xb = (torch.from_numpy(xb) if isinstance(xb, np.ndarray) else xb).to(dev)
        yb = (torch.from_numpy(yb) if isinstance(yb, np.ndarray) else yb).to(dev)
        model.zero_grad(set_to_none=True)
        ((model(xb) - yb)[..., :lch] ** 2).mean().backward()
        for n, p in model.named_parameters():
            if n in G and p.grad is not None:
                G[n].append(p.grad.detach().reshape(-1).to(torch.float16).cpu())
    model.zero_grad(set_to_none=True)
    U, ks = {}, []
    for n in names:
        if not G[n]:
            continue
        Gm = torch.stack(G[n], 0).to(dev, torch.float32)          # (N, d)
        K = Gm @ Gm.T                                             # (N, N) Gram trick
        lam, V = torch.linalg.eigh(K)                             # ascending
        lam = lam.flip(0).clamp_min(0.0)
        V = V.flip(1)
        cs = torch.cumsum(lam, 0) / lam.sum().clamp_min(1e-30)
        k = min(int((cs < float(tau)).sum().item()) + 1, int(max_dim), int(Gm.shape[0]))
        Uk = Gm.T @ V[:, :k] / lam[:k].sqrt().clamp_min(1e-20)    # right singular vectors of G
        Uk, _ = torch.linalg.qr(Uk)                               # numerical safety
        U[n] = Uk.contiguous()
        ks.append(k)
        del Gm, K
    logger.info(f"  [physguard] protected {len(U)}/{len(names)} layers | "
                f"k median {int(np.median(ks)) if ks else 0}, max {max(ks) if ks else 0} | "
                f"N={len(idx)} tau={tau} alpha={alpha} (arXiv 2606.16602)")
    return U


def fisher_diag_build(model, loss_call, n_windows, bs, n_batches, q, min_numel, logger):
    """FGP (2026-08-12): DIAGONAL-Fisher gradient masking -- a deliberate coarsening of
    PhysGuard's subspace projection (`physguard_build` above, arXiv 2606.16602).  Empirical
    Fisher diagonal = the MEAN OF SQUARED MINIBATCH GRADIENTS of the stage's own training loss
    over `n_batches` minibatches of the sim stage's windows, taken at the end-of-sim-pretrain
    weights.  Per parameter TENSOR ("layer"), the top-q fraction of coordinates by Fisher value
    is PROTECTED: their gradients are zeroed for every later stage.  Where physguard damps
    updates along shared low-rank directions, this freezes individual coordinates -- the
    importance-masking reading of the same idea.  Returns {param_name: float mask, 1 = free,
    0 = protected}.  Batch selection uses a DEDICATED numpy RNG so the global torch RNG stream
    -- and therefore the following stage's minibatch sequence -- is untouched, which is what
    keeps an arm/control pair at the same seed exactly paired."""
    import torch          # this file imports torch per-function, not at module scope
    rng = np.random.default_rng(20260812)
    names = [n for n, p in model.named_parameters()
             if p.requires_grad and p.numel() >= int(min_numel)]
    fish = {n: None for n in names}
    for _ in range(int(n_batches)):
        idx = torch.from_numpy(rng.integers(0, n_windows, size=min(int(bs), int(n_windows))))
        model.zero_grad(set_to_none=True)
        loss_call(idx).backward()
        for n, p in model.named_parameters():
            if n in fish and p.grad is not None:
                g2 = (p.grad.detach().float() ** 2)
                fish[n] = g2 if fish[n] is None else fish[n] + g2
    model.zero_grad(set_to_none=True)
    masks, tot_p, tot_n = {}, 0, 0
    for n in names:
        F = fish[n]
        if F is None:
            continue
        k = int(round(float(q) * F.numel()))
        if k <= 0:
            continue
        flat = (F / float(n_batches)).reshape(-1)
        top = flat.topk(k).indices
        m = torch.ones_like(flat)
        m[top] = 0.0
        mass = float(flat[top].sum() / flat.sum().clamp_min(1e-30))
        masks[n] = m.view_as(F).contiguous()
        tot_p += k
        tot_n += F.numel()
        logger.info(f"  [fgp] {n}: numel={F.numel()} protected={k} "
                    f"({k / F.numel():.3f}) fisher-mass={mass:.3f}")
    logger.info(f"  [fgp] TOTAL protected {tot_p}/{tot_n} = {tot_p / max(1, tot_n):.4f} over "
                f"{len(masks)} tensors | q={q} n_batches={n_batches} bs={bs} "
                f"(diagonal-Fisher simplification of arXiv 2606.16602)")
    return masks


def build_windows(files, sub_s, t_in, t_out, mean, std, stride, logger, label,
                  n_roll=1, roll_stride=None, phases=None, t0_start=0, load_p=False,
                  aux_phases=None, p_cache=None, p_target_only=False,
                  meas_noise=0.0, meas_floor=0.0149, frame_step=1,
                  fluc_scale=None, fluc_scale_missing="refuse", dr_chan_gain=None):
    """X = t_in frames, Y = the frames every unrolled block needs.  n_roll=1 reproduces v2.

    `roll_stride` s is how far the INPUT WINDOW ADVANCES per unrolled block (default t_out).
    Block k targets frames [t_in + k*s, t_in + k*s + t_out), so the span is
        H = t_in + (n_roll-1)*s + t_out
    and s = t_out recovers the old non-overlapping behaviour exactly.

    `t0_start` skips the first frames of every trajectory.  It exists so that two arms with
    DIFFERENT t_in can be scored on the SAME target windows: the target of a window is
    [t0+t_in, t0+H), so without an offset a longer input window silently shifts the target
    forward and the two arms stop solving the same problem.  Setting
    t0_start = t_in_max - t_in aligns the targets exactly, and makes the per-trajectory
    window counts match as well.  Default 0 = the historical behaviour."""
    import h5py
    s_roll = int(roll_stride or t_out)
    H = t_in + (n_roll - 1) * s_roll + t_out
    X, Y = [], []
    T_seen = set()
    # (0,0) is the scored phase and must always be present; extra phases are augmentation.
    ph = [tuple(p) for p in (phases or [(0, 0)])]
    assert (0, 0) in ph, f"phases must include the scored phase (0,0), got {ph}"
    # AXP: `aux_phases` asks for AUXILIARY TARGETS -- the genuinely unseen sibling-phase pixels
    # of the SAME target frames -- returned as a third array (N, P, t_out, H, W, C).  This is a
    # different object from `phases` (which builds extra WINDOWS, arm P4); combining them is
    # undefined and refused.  X and Y are bit-identical to the aux_phases=None call.
    aux_ph = [tuple(p) for p in (aux_phases or [])]
    Yx = []
    if aux_ph:
        assert n_roll == 1, f"aux targets are defined for n_roll=1 only, got {n_roll}"
        assert ph == [(0, 0)], f"aux_phases cannot combine with `phases` augmentation, got {ph}"
        assert (0, 0) not in aux_ph, f"(0,0) is the MAIN target, not an aux phase: {aux_ph}"
    # SIMDT: `frame_step` builds each window from frame indices t0, t0+k, ... t0+(H-1)k --
    # ONE arithmetic sequence spanning INPUT AND TARGET.  k == 1 keeps the historical slice path
    # bit-for-bit, so every previous arm is untouched.  Spec forms:
    #   int / "1"          fixed k
    #   "match:<a>:<c>:<K>"  per-FILE k = clip(round(exp(c) * Re**a / D_SIM), 1, K)   (SIMDT_match)
    #   "rand:<K>"         per-WINDOW k ~ Uniform{1..K}, seeded per file               (SIMDT_rand)
    #   "randmatch:<a>:<c>:<K>:<s>"  per-WINDOW k ~ Uniform{lo..hi} around THAT FILE's matched k,
    #                      lo = max(1, round(k_match / s)), hi = min(K, round(k_match * s))     (SIMDR_rm)
    #                      The cell neither SIMDT (matched but FIXED) nor DRT (random but UNMATCHED)
    #                      covers: centre the sim windows on the physically right rate AND keep the
    #                      spread that stops the net from locking onto one wrong rate.  Measured
    #                      2026-09-20: DRT (rand:6) removes the whole harm of the sim stage
    #                      (+0.0137 both tracks vs the plain-sim control) while SIMDT's fixed match
    #                      lost (-0.010) -- so the spread, not the centre, was doing the work; this
    #                      spec asks whether the centre adds anything on top.
    # WHY IT IS LEGAL: every frame used is a released frame verbatim -- this is a subset
    # selection, not an interpolation, so every sample traces to a released one.
    _fs_spec = str(frame_step)
    _fs_hist = {}
    _fs_checked = 0
    # SIMDR (2026-09-20): PER-WINDOW, PER-CHANNEL fluctuation gain -- domain randomisation of the
    # anisotropy E_v/E_u, which the measured sim-real gap puts at real 14.7 vs sim 5.6 in the model's
    # own normalised space (`common/sim_real_distribution.py`) and which `fluc_scale` CANNOT express
    # (it is one alpha per file, applied to u and v alike).  For each window, u and v are scaled about
    # the trajectory's own per-pixel temporal mean by independent gains drawn log-uniformly from
    # [lo, hi]; the SAME pair scales that window's input and target, so a window stays one coherent
    # sample.  None == OFF == the historical bit path.
    _dr = None
    if dr_chan_gain:
        _dr_lo, _dr_hi = (float(x) for x in dr_chan_gain)
        assert 0.0 < _dr_lo <= _dr_hi, f"dr_chan_gain must be 0 < lo <= hi, got {dr_chan_gain!r}"
        _dr = (_dr_lo, _dr_hi)
    _dr_mus, _dr_w, _dr_g = [], [], []
    if _fs_spec.startswith("match:"):
        _p = _fs_spec.split(":"); _fs_a, _fs_c, _fs_K = float(_p[1]), float(_p[2]), int(_p[3])
    elif _fs_spec.startswith("randmatch:"):
        _p = _fs_spec.split(":")
        _fs_a, _fs_c, _fs_K, _fs_S = float(_p[1]), float(_p[2]), int(_p[3]), float(_p[4])
        assert _fs_S > 1.0, f"randmatch spread must be > 1, got {_fs_S}"
    elif _fs_spec.startswith("rand:"):
        _fs_K = int(_fs_spec.split(":")[1])
    else:
        _fs_fixed = int(_fs_spec)
        assert _fs_fixed >= 1, f"frame_step must be >= 1, got {_fs_fixed}"
    n_with_p = 0
    n_from_cache = 0
    _fs_applied = []
    w_re = []              # WRa: per-window trajectory Re, appended in EXACT X.append order
    w_aoa = []             # AOSIM: per-window trajectory AoA, same order and contract as w_re
    w_key = []             # CLUSTERW knob (2026-09-10): per-window '<basename>|dy|dx|t0', same order as X
    for fn in files:
        # files are named <Re>_<AoA>.h5; -1 marks an unparseable name
        # so the WRa arming code can refuse loudly instead of tilting on garbage.
        try:
            _re_val = int(Path(fn).stem.split("_")[0])
        except (ValueError, IndexError):
            _re_val = -1
        try:
            _aoa_val = int(Path(fn).stem.split("_")[1])
        except (ValueError, IndexError):
            _aoa_val = -1
        with h5py.File(fn, "r") as f:            # one native read, reused by every phase
            un = np.asarray(f["u"][:], np.float32)
            vn = np.asarray(f["v"][:], np.float32)
            # defect 4: `p` used to be np.zeros_like(u) unconditionally, so loss_channels: 3
            # supervised a constant.  Read it when the file actually has it and the source
            # asks for it.
            pn = np.asarray(f["p"][:], np.float32) if (load_p and "p" in f) else None
        if fluc_scale:
            # SIMDAMP: damp this trajectory's temporal fluctuation by its per-case alpha BEFORE
            # any windowing, so input and target of every window carry the same transform of the
            # same frames.  A stem missing from the table is a hard refusal by default: silently
            # falling back to alpha=1 would train an undocumented mixture.
            _fs_stem = Path(fn).stem
            if _fs_stem in fluc_scale:
                _fs_ent = fluc_scale[_fs_stem]
                _fs_a = float(_fs_ent["alpha"] if isinstance(_fs_ent, dict) else _fs_ent)
                un = apply_fluc_scale(un, _fs_a)
                vn = apply_fluc_scale(vn, _fs_a)
                _fs_applied.append((_fs_stem, _fs_a))
            elif fluc_scale_missing == "refuse":
                raise SystemExit(f"SIMDAMP: no alpha for {_fs_stem!r} in the table and the "
                                 f"missing-policy is 'refuse'")
        if meas_noise > 0.0:
            # SIMULATE THE MEASUREMENT.  Isotropic additive noise in PHYSICAL units at
            # meas_noise x meas_floor x rms(field), applied to the NATIVE field ONCE per file with
            # a seed derived from the filename, so (a) input and target of every window carry the
            # SAME corruption of the SAME frame, (b) overlapping windows agree, and (c) the run is
            # reproducible.  meas_floor 0.0149 is t1_noise_floor.py's measured real-PIV level.
            #
            # THE SEED IS crc32, NOT hash() -- fixed 2026-09-01 (ROUND7-FIXBACKLOG item 2).  This
            # read `abs(hash(Path(fn).stem)) % (2**31)`, and Python's STRING hash is randomised per
            # process unless PYTHONHASHSEED is set, which it is nowhere in this repo.  So (a) and
            # (b) held -- they only need one process -- while (c), the claim the comment actually
            # makes, was FALSE: every re-run drew a different noise field, and a Decision-Phase
            # rebuild by the organizers could not have reproduced a meas_noise arm at all.
            # Measured before the fix, one key across three fresh processes: 1830722960 /
            # 1437168430 / 1778206435.  crc32 is a pure function of the bytes and is stable across
            # processes, machines and interpreter versions.
            _rs = np.random.RandomState(zlib.crc32(Path(fn).stem.encode("utf-8")) % (2**31))
            for _a in (un, vn):
                _sg = meas_noise * meas_floor * float(np.sqrt((_a.astype(np.float64) ** 2).mean()))
                _a += _rs.normal(0.0, _sg, _a.shape).astype(np.float32)
        if p_cache:
            # PIN: channel 2 := pressure RECONSTRUCTED from this trajectory's own velocity
            # (PPE variant A, per-connected-component Neumann gauge).  Overrides the file's own
            # p on purpose -- the sim pool has true CFD pressure, but the slot must carry one
            # physical quantity across both stages or the network sees two different things.
            _pp = Path(p_cache) / (Path(fn).stem + ".npy")
            assert _pp.exists(), f"p_cache miss: {_pp}"
            pn = np.load(_pp).astype(np.float32)
            assert pn.shape == un.shape, (pn.shape, un.shape, str(_pp))
            n_from_cache += 1
        if pn is not None:
            n_with_p += 1
        for (dy, dx) in ph:
            u = un[:, dy::sub_s, dx::sub_s]
            v = vn[:, dy::sub_s, dx::sub_s]
            p = pn[:, dy::sub_s, dx::sub_s] if pn is not None else None
            T = u.shape[0]
            T_seen.add(T)
            if T < H:
                logger.warning(f"  {label}: skip {Path(fn).name} (T={T} < H={H})")
                continue
            data = np.stack([u, v, (p if p is not None else np.zeros_like(u))],
                            -1).astype(np.float32)
            data = ((data - mean) / std).astype(np.float32)
            aux_data = None
            if aux_ph:
                # the same trajectory arrays, the same normalisation, the OTHER subsample
                # phases -- (0,1)/(1,0)/(1,1) of a 64x128 native grid are each exactly 32x64.
                aux_data = []
                for (ay, ax) in aux_ph:
                    au = un[:, ay::sub_s, ax::sub_s]
                    av = vn[:, ay::sub_s, ax::sub_s]
                    ap = pn[:, ay::sub_s, ax::sub_s] if pn is not None else None
                    ad = np.stack([au, av, (ap if ap is not None else np.zeros_like(au))],
                                  -1).astype(np.float32)
                    assert ad.shape == data.shape, (ad.shape, data.shape, (ay, ax))
                    aux_data.append(((ad - mean) / std).astype(np.float32))
            # SIMDT: per-file frame step.  D_SIM = 0.3253 px/frame is the MEASURED median
            # displacement of the sim pool (flat in Re).
            if _fs_spec.startswith("match:"):
                _k_file = int(np.clip(round(float(np.exp(_fs_c) * (max(_re_val, 1) ** _fs_a))
                                            / 0.3253), 1, _fs_K))
                _k_max = _k_file
            elif _fs_spec.startswith("randmatch:"):
                # this file's matched rate, then a multiplicative window around it
                _k_m = int(np.clip(round(float(np.exp(_fs_c) * (max(_re_val, 1) ** _fs_a))
                                         / 0.3253), 1, _fs_K))
                _rm_lo = max(1, int(round(_k_m / _fs_S)))
                _rm_hi = min(_fs_K, max(_rm_lo, int(round(_k_m * _fs_S))))
                _k_file = None
                _k_max = _rm_hi
            elif _fs_spec.startswith("rand:"):
                _k_file = None
                _k_max = _fs_K
            else:
                _k_file = _fs_fixed
                _k_max = _fs_fixed
            # SEEDED WITH crc32, NOT hash() -- fixed 2026-09-01 (ROUND7-FIXBACKLOG item 2, the
            # SECOND instance of the same defect, found 51 lines below the first one during the
            # verification pass for it).  This read
            #     abs(hash(("simdt", Path(fn).stem, dy, dx))) % (2 ** 31)
            # and a tuple hash MIXES THE HASHES OF ITS MEMBERS, so the STRING member carries
            # Python's per-process hash randomisation straight into the seed.  MEASURED at the
            # consumer before the fix -- build_windows(frame_step="rand:4") over one real .h5 in
            # three fresh interpreters returned THREE DIFFERENT window tensors (md5 78ad726f /
            # b6921362 / dd3c98d9), so the SIMDTr and SIMDTrD arms were not reproducible run to
            # run.  Only the "rand:" spec ever DRAWS from this RandomState (`_k_file is None`);
            # for frame_step=1 and "match:" it is constructed and never used, which is why the
            # historical bit path is provably untouched -- witnessed, not asserted, in T5.
            # The key is ONE explicit string, so the seed is a pure function of bytes: stable
            # across processes, machines and interpreter versions.
            _fs_key = "simdt:%s:%d:%d" % (Path(fn).stem, dy, dx)
            _rs_fs = np.random.RandomState(zlib.crc32(_fs_key.encode("utf-8")) % (2 ** 31))
            if _dr is not None:
                _dr_fi = 0
                _rs_dr = np.random.RandomState(
                    zlib.crc32(("drcg:%s:%d:%d" % (Path(fn).stem, dy, dx)).encode("utf-8")) % (2 ** 31))
            # t0 upper bound uses _k_max so that EVERY admissible t0 admits EVERY k -- otherwise
            # the realised k distribution would be biased by trajectory length.
            for t0 in range(t0_start, T - (H - 1) * _k_max, stride):
                if _k_file is not None:
                    k = _k_file
                elif _fs_spec.startswith("randmatch:"):
                    k = int(_rs_fs.randint(_rm_lo, _rm_hi + 1))
                else:
                    k = int(_rs_fs.randint(1, _fs_K + 1))
                if k == 1:
                    # historical path, untouched -- SIMDT_null is bit-identical to every past arm
                    X.append(data[t0:t0 + t_in])
                    Y.append(data[t0 + t_in:t0 + H])
                else:
                    # ONE index array feeds BOTH X and Y, so an implementation that strides the
                    # input but not the target is structurally impossible, not merely asserted.
                    _idx = np.arange(t0, t0 + H * k, k)
                    if _fs_checked < 3:
                        assert _idx.size == H and np.all(np.diff(_idx) == k) and _idx[-1] < T, \
                            ("SIMDT index witness", _idx.size, H, k, T)
                        _fs_checked += 1
                    X.append(data[_idx[:t_in]])
                    Y.append(data[_idx[t_in:]])
                _fs_hist[k] = _fs_hist.get(k, 0) + 1
                if _dr is not None:
                    _g = np.exp(_rs_dr.uniform(np.log(_dr[0]), np.log(_dr[1]), 2))
                    _dr_w.append((_dr_fi, float(_g[0]), float(_g[1])))
                    _dr_g.append(_g)
                w_re.append(_re_val)             # WRa: keep 1:1 with X
                w_aoa.append(_aoa_val)           # AOSIM: keep 1:1 with X
                w_key.append("%s|%d|%d|%d" % (Path(fn).name, dy, dx, t0))   # CLUSTERW knob (2026-09-10)
                if aux_data is not None:
                    assert k == 1, "SIMDT frame_step > 1 does not compose with aux_phases"
                    Yx.append(np.stack([ad[t0 + t_in:t0 + H] for ad in aux_data], 0))
    X = np.asarray(X, np.float32)
    Y = np.asarray(Y, np.float32)
    if _dr is not None and X.size:
        # applied AFTER stacking so the windowing path above stays byte-identical to every past arm.
        # THE FIXED POINT IS THE WINDOW'S OWN per-pixel mean over its FULL span (input + target), so
        # the window's time-mean field -- what mvpe scores -- is preserved exactly and the knob moves
        # ONLY the fluctuation (what tke scores). One gain pair per window, shared by input and target.
        assert len(_dr_w) == X.shape[0] == Y.shape[0], (len(_dr_w), X.shape[0], Y.shape[0])
        _nsp = X.shape[1] + Y.shape[1]
        for _i, (_fi, _gu, _gv) in enumerate(_dr_w):
            for _c, _g in ((0, _gu), (1, _gv)):
                _mu = (X[_i, ..., _c].sum(0) + Y[_i, ..., _c].sum(0)) / _nsp
                X[_i, ..., _c] = _mu + _g * (X[_i, ..., _c] - _mu)
                Y[_i, ..., _c] = _mu + _g * (Y[_i, ..., _c] - _mu)
        _gg = np.asarray(_dr_g)
        logger.info(f"  {label}: SIMDR dr_chan_gain={list(_dr)} windows={len(_dr_w)} "
                    f"gu geo-mean={float(np.exp(np.log(_gg[:, 0]).mean())):.4f} "
                    f"gv geo-mean={float(np.exp(np.log(_gg[:, 1]).mean())):.4f} "
                    f"realised gv/gu geo-mean={float(np.exp(np.log(_gg[:, 1] / _gg[:, 0]).mean())):.4f}")
    if p_target_only and X.size:
        # PAUX: the pressure is a TARGET, not an input.  Zeroing here (rather than not loading it)
        # keeps Y identical to the PIN arm, so PAUX and PIN differ in exactly one thing.
        X[..., 2] = 0.0
    # defect 4: record what the DATA actually looks like, per channel, so the caller can
    # refuse to supervise a channel that does not vary.  dtype=np.float64 is REQUIRED: with a
    # float32 accumulator a genuinely constant channel reads std = 2.6e-4 rather than 0
    # (E[x^2]-E[x]^2 cancels catastrophically), which is enough to slip past any sane threshold.
    # The constant-channel guard downstream reads this to refuse supervising a constant.  It must
    # describe the SUPERVISED array.  X and Y come from the same array for every arm except
    # p_target_only, which zeroes X[...,2] on purpose while Y keeps the pressure -- so read Y there
    # and X everywhere else, which leaves every historical arm bit-identical.
    _gsrc = Y if (p_target_only and Y.size) else X
    ch_std = (_gsrc.reshape(-1, _gsrc.shape[-1]).std(0, dtype=np.float64) if _gsrc.size
              else np.zeros(3, np.float64))
    LOADED_CH_STD[label] = ch_std
    LOADED_WIN_RE[label] = np.asarray(w_re, np.int64)
    assert LOADED_WIN_RE[label].shape[0] == X.shape[0], \
        (LOADED_WIN_RE[label].shape, X.shape)   # WRa alignment invariant
    LOADED_WIN_AOA[label] = np.asarray(w_aoa, np.int64)
    assert LOADED_WIN_AOA[label].shape[0] == X.shape[0], \
        (LOADED_WIN_AOA[label].shape, X.shape)  # AOSIM alignment invariant
    LOADED_WIN_KEY[label] = list(w_key)   # CLUSTERW knob (2026-09-10)
    assert len(LOADED_WIN_KEY[label]) == X.shape[0], (len(LOADED_WIN_KEY[label]), X.shape)   # CLUSTERW knob (2026-09-10)
    logger.info(f"  {label}: loaded p from {n_with_p}/{len(files)} file(s) (load_p={load_p}, p_cache={n_from_cache} from {p_cache}); "
                f"normalised per-channel data std = {np.asarray(ch_std).tolist()}")
    logger.info(f"  {label}: {len(files)} files x {len(ph)} phase(s) {ph} (T={sorted(T_seen)}) stride={stride} n_roll={n_roll} "
                f"roll_stride={s_roll} (H={H}) -> "
                f"{X.shape[0]} windows, {(X.nbytes+Y.nbytes)/1e9:.2f} GB")
    # SIMDT execution witness: the realised k histogram of the windows THIS CALL BUILT.
    # Logged unconditionally so a run whose knob silently did nothing is visible in run.log.
    LOADED_FRAME_STEP[label] = dict(spec=_fs_spec, hist={int(a): int(b) for a, b in
                                                         sorted(_fs_hist.items())})
    logger.info(f"  {label}: SIMDT frame_step spec={_fs_spec!r} realised k histogram="
                f"{ {int(a): int(b) for a, b in sorted(_fs_hist.items())} } "
                f"(sum={sum(_fs_hist.values())} == {X.shape[0]} windows)")
    assert sum(_fs_hist.values()) == X.shape[0], (sum(_fs_hist.values()), X.shape)
    if aux_ph:
        Yx = np.asarray(Yx, np.float32)
        assert Yx.shape == (Y.shape[0], len(aux_ph)) + Y.shape[1:], (Yx.shape, Y.shape)
        logger.info(f"  {label}: aux phase targets {aux_ph} -> {Yx.shape}, "
                    f"{Yx.nbytes/1e9:.2f} GB (AXP)")
    if _fs_applied:
        _al = np.array([a for _s, a in _fs_applied])
        logger.info(f"  {label}: SIMDAMP applied to {len(_fs_applied)}/{len(files)} file(s) | "
                    f"alpha min/median/max {_al.min():.4f}/{np.median(_al):.4f}/{_al.max():.4f} | "
                    f"list md5 {zlib.crc32(repr(sorted(_fs_applied)).encode()) & 0xffffffff:08x}")
    elif fluc_scale:
        logger.info(f"  {label}: SIMDAMP table present but matched 0 of {len(files)} file(s)")
    if aux_ph:
        return X, Y, Yx
    return X, Y


_LR_WARMUP = 0.03      # cosine_lr's own default; overridden from config by LRSCH
_LR_MINR = 0.02        # ditto


def cosine_lr(step, total, base_lr, warmup=0.03, min_ratio=0.02):
    w = max(1, int(total * warmup))
    if step < w:
        return base_lr * (step + 1) / w
    p = (step - w) / max(1, total - w)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def _zeropower_ns5(G, steps=5):
    """Newton-Schulz quintic: return an approximately ORTHOGONAL matrix with G's column space.

    The quintic coefficients are Jordan's; they are tuned so the iteration converges on the
    singular values from below without needing them normalised to (0,1] exactly.  Runs in bf16 on
    purpose -- the iteration is contractive, so the precision does not propagate into the weights.
    """
    import torch
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + 1e-7)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(int(steps)):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon:
    """Orthogonalised-momentum optimiser for >=2D parameters (MUa, 2026-08-15).

    Deliberately NOT a torch.optim.Optimizer subclass: it holds only the params AdamW does not,
    so the two never touch the same tensor, and `lr` is written per step by the same schedule
    that drives AdamW -- see the step site.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        import torch
        self.params = [p for p in params if p.ndim >= 2]
        self.lr, self.momentum, self.nesterov, self.ns_steps = lr, momentum, nesterov, ns_steps
        self.buf = {id(p): torch.zeros_like(p) for p in self.params}

    def step(self):
        import torch
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                g = p.grad
                b = self.buf[id(p)]
                b.mul_(self.momentum).add_(g)
                u = g.add(b, alpha=self.momentum) if self.nesterov else b
                flat = u.reshape(u.shape[0], -1)
                o = _zeropower_ns5(flat, self.ns_steps).reshape(u.shape)
                # scale so the update's RMS matches Adam's, independent of shape
                o = o * (max(1.0, flat.shape[0] / flat.shape[1]) ** 0.5)
                p.add_(o, alpha=-self.lr)


# ------------------------------------------------------------------ losses
_LOWK_CACHE = {}


def _lowk_mask(h, w, cut, device):
    """(1,1,h,w//2+1,1) real mask: 1 on radial wavenumbers <= cut*Nyquist, 0 above.  Cached."""
    import torch          # train_v4 imports torch lazily; this helper is module-level
    key = (int(h), int(w), round(float(cut), 6), str(device))
    m = _LOWK_CACHE.get(key)
    if m is None:
        fy = torch.fft.fftfreq(int(h), device=device).reshape(-1, 1)
        fx = torch.fft.rfftfreq(int(w), device=device).reshape(1, -1)
        m = ((fy.pow(2) + fx.pow(2)).sqrt() <= float(cut) * 0.5).to(torch.float32)
        # rfft2 keeps only half the x-spectrum, so every INTERIOR column stands for two conjugate
        # modes.  Without this multiplicity the term silently under-weights them and stops being an
        # L2 on the low-passed field -- caught by the Parseval self-test at 1.2e-04, not by eyeball.
        mult = torch.full((1, int(w) // 2 + 1), 2.0, device=device)
        mult[0, 0] = 1.0
        if int(w) % 2 == 0:
            mult[0, -1] = 1.0
        m = (m * mult.sqrt()).reshape(1, 1, int(h), int(w) // 2 + 1, 1)
        _LOWK_CACHE[key] = m
    return m


def parse_lead_weight(spec, t_out):
    """LWa (2026-08-13): parse `lead_weight` into a length-t_out vector whose mean is EXACTLY 1.0.

    Schemes (i = 0-based index, lead = i+1, T = t_out):
      ""          -> None.  The knob is OFF and make_loss returns its ORIGINAL closures, bit-exact.
      "linear:k"  -> raw[i] = 1 + (k-1)*i/(T-1)          rises linearly from 1 (lead 1) to k (lead T)
      "late:k"    -> raw[i] = 1 for i < T//2 else k      leads 1..T/2 = 1, leads T/2+1..T = k
      "a,b,c,..." -> an explicit length-T vector

    NORMALISATION IS LOAD-BEARING, not cosmetic: w = raw * T / sum(raw) forces mean(w) == 1.0, so
    the arm redistributes gradient ACROSS leads without changing the objective's overall scale.
    Without it, "weight the late leads up" would also multiply the whole loss (and hence the
    effective learning rate), confounding the reweighting with an LR sweep.  Asserted numerically.
    """
    s = str(spec or "").strip()
    if not s:
        return None
    T = int(t_out)
    if T < 2:
        raise SystemExit(f"lead_weight needs t_out >= 2, got {T}")
    if ":" in s:
        kind, _, val = s.partition(":")
        kind = kind.strip().lower()
        try:
            k = float(val)
        except ValueError:
            raise SystemExit(f"lead_weight {s!r}: k is not a number")
        if k <= 0.0:
            raise SystemExit(f"lead_weight {s!r}: k must be > 0, got {k}")
        if kind == "linear":
            raw = [1.0 + (k - 1.0) * (i / float(T - 1)) for i in range(T)]
        elif kind == "late":
            raw = [1.0 if i < T // 2 else k for i in range(T)]
        else:
            raise SystemExit(f"lead_weight scheme must be linear|late, got {kind!r}")
    else:
        raw = [float(x) for x in s.split(",")]
        if len(raw) != T:
            raise SystemExit(f"lead_weight vector has {len(raw)} entries, need t_out={T}")
        if min(raw) <= 0.0:
            raise SystemExit(f"lead_weight entries must all be > 0: {raw}")
    tot = float(sum(raw))
    w = [x * T / tot for x in raw]
    mean_w = sum(w) / float(T)
    assert abs(mean_w - 1.0) < 1e-12, f"lead_weight mean must be 1.0, got {mean_w!r}"
    return w


def make_loss(kind, normalize=True, eps=1e-12, lead_w=None, elem_w=None, chan_w=None,
              phys_den=None):
    """Returns fn(pred, tgt) -> scalar.  pred/tgt are (B, T, H, W, C_loss).

    lead_w (LWa, 2026-08-13): None -> the ORIGINAL closures, returned untouched, so the default
    path is bit-exact.  A length-T list with mean 1.0 -> the per-lead weight multiplies the
    SQUARED-ERROR contribution of each output frame INSIDE the existing reduction.  Consequences,
    and they are why this parameterisation was chosen over a per-lead-normalised loss:
      * the gradient reaching output frame t is scaled by exactly w[t], so the knob's meaning is
        "relative gradient share per lead" and nothing else;
      * w == all-ones reproduces the unweighted loss identically, so the OFF arm is the baseline;
      * for `rel_l2` only the NUMERATOR is weighted -- the denominator is the target window's own
        norm (the thing that makes this loss the competition's rel_l2 subscore), and weighting it
        too would partially cancel the reweighting.
    The weight applies through `loss_fn`, hence to the distillation term as well as the hard-target
    term -- i.e. to the objective, not to one summand of it.
    """
    import torch

    def _mse(p, y):
        return torch.mean((p - y) ** 2)

    def _l2(p, y):
        n = p.shape[0]
        se = ((p - y) ** 2).reshape(n, -1).sum(1)
        d = float(p[0].numel()) if normalize else 1.0
        return torch.sqrt(se / d + eps).mean()

    def _rel(p, y):
        n = p.shape[0]
        se = ((p - y) ** 2).reshape(n, -1).sum(1)
        yn = ((y ** 2).reshape(n, -1).sum(1)).clamp(min=1e-12)
        return (torch.sqrt(se + eps) / torch.sqrt(yn)).mean()

    if lead_w is None and elem_w is None and chan_w is None and phys_den is None:
        try:
            return {"mse": _mse, "l2": _l2, "rel_l2": _rel}[kind]
        except KeyError:
            raise SystemExit(f"loss_type must be one of mse|l2|rel_l2, got {kind!r}")

    # ---- LWa armed: the same three reductions with a per-lead factor on the squared error ----
    _wcache = {}

    def _w(p):
        base = None
        if lead_w is not None:
            key = (str(p.device), p.dtype, int(p.shape[1]))
            base = _wcache.get(key)
            if base is None:
                if int(p.shape[1]) != len(lead_w):
                    raise SystemExit(f"lead_weight has {len(lead_w)} entries but the loss saw "
                                     f"T={int(p.shape[1])} -- refusing to broadcast silently")
                base = torch.tensor(lead_w, device=p.device, dtype=p.dtype).reshape(1, -1, 1, 1, 1)
                _wcache[key] = base
        # SPSW: a per-element weight for the CURRENT batch, rebound by the training loop each
        # step.  A one-key dict rather than a closure argument so the reduction never changes.
        ew = elem_w.get("w") if elem_w is not None else None
        if chan_w is not None:
            # CW: a per-CHANNEL factor on the squared-error NUMERATOR, so the objective reduces
            # error in the same units scoring.py measures it in.  Broadcast over (B,T,H,W,C).
            key = ("chan", str(p.device), p.dtype, int(p.shape[-1]))
            cw = _wcache.get(key)
            if cw is None:
                if int(p.shape[-1]) != len(chan_w):
                    raise SystemExit(f"chan_weight has {len(chan_w)} entries but the loss saw "
                                     f"C={int(p.shape[-1])} -- refusing to broadcast silently")
                cw = torch.tensor(chan_w, device=p.device, dtype=p.dtype).reshape(1, 1, 1, 1, -1)
                _wcache[key] = cw
            base = cw if base is None else base * cw
        if ew is None:
            return base if base is not None else 1.0
        return ew if base is None else base * ew

    def _mse_w(p, y):
        return torch.mean(_w(p) * (p - y) ** 2)

    def _l2_w(p, y):
        n = p.shape[0]
        se = (_w(p) * (p - y) ** 2).reshape(n, -1).sum(1)
        d = float(p[0].numel()) if normalize else 1.0
        return torch.sqrt(se / d + eps).mean()

    def _rel_w(p, y):
        n = p.shape[0]
        se = (_w(p) * (p - y) ** 2).reshape(n, -1).sum(1)
        # CWD: the scorer's denominator is the PHYSICAL target norm (scoring.py:89), which is
        # dominated by the mean flow and near-constant across samples; the normalised one is the
        # fluctuation magnitude and varies ~7x over the Reynolds sweep, so using it silently
        # over-weights low-Re samples relative to what is scored.
        yd = y
        if phys_den is not None and phys_den.get("ms") is not None:
            _m, _s = phys_den["ms"]
            yd = y * _s + _m
        yn = ((yd ** 2).reshape(n, -1).sum(1)).clamp(min=1e-12)
        return (torch.sqrt(se + eps) / torch.sqrt(yn)).mean()

    try:
        return {"mse": _mse_w, "l2": _l2_w, "rel_l2": _rel_w}[kind]
    except KeyError:
        raise SystemExit(f"loss_type must be one of mse|l2|rel_l2, got {kind!r}")


def tke_field(x, per_channel=False):
    """scoring.py::kinetic_energy -- per-pixel temporal variance.  x (B,T,H,W,C>=2) -> (B,H,W).

    per_channel=True returns the UNSUMMED variances (B,H,W,2) so a caller can restore physical
    units per channel before summing -- see tke_rel_l2's chan_std.  Default False is unchanged.
    """
    import torch          # train_v4 imports torch lazily; this helper is module-level
    u, v = x[..., 0], x[..., 1]
    up = ((u - u.mean(1, keepdim=True)) ** 2).mean(1)
    vp = ((v - v.mean(1, keepdim=True)) ** 2).mean(1)
    if per_channel:
        return torch.stack([up, vp], dim=-1)
    return 0.5 * (up + vp)


_BSP_CACHE = {}


def _bsp_bins(h, w, nbins, device):
    """Radial wavenumber shell index for an rfft2 half-plane, cached per (h, w, nbins, device)."""
    import torch          # this file imports torch per-function, not at module scope
    key = (h, w, nbins, str(device))
    if key not in _BSP_CACHE:
        ky = torch.fft.fftfreq(h, device=device).view(-1, 1) * h
        kx = torch.fft.rfftfreq(w, device=device).view(1, -1) * w
        kmag = torch.sqrt(ky ** 2 + kx ** 2)
        idx = torch.clamp((kmag / (kmag.max() + 1e-12) * nbins).long(), max=nbins - 1)
        onehot = torch.zeros(nbins, h, kx.shape[1], device=device)
        onehot.scatter_(0, idx.unsqueeze(0), 1.0)
        _BSP_CACHE[key] = onehot
    return _BSP_CACHE[key]


def bsp_loss(pred, tgt, nbins=8, eps=1e-6):
    """Binned Spectral Power loss, arXiv:2502.00472.

    Mean squared difference of LOG binned spatial power. An L2 objective is minimised by the
    conditional mean, whose spectrum is damped at high wavenumber; this constrains where the energy
    sits rather than how much of it there is in total, which is what makes it different in kind from
    a TKE penalty. `eps` is relative to the mean power so the term is invariant to normalisation.
    """
    import torch          # this file imports torch per-function, not at module scope
    x = pred.permute(0, 1, 4, 2, 3)                      # (B,T,C,H,W)
    y = tgt.permute(0, 1, 4, 2, 3)
    h, w = x.shape[-2], x.shape[-1]
    oh = _bsp_bins(h, w, nbins, x.device)                # (nbins,H,W//2+1)
    px = torch.fft.rfft2(x, norm="ortho").abs() ** 2
    py = torch.fft.rfft2(y, norm="ortho").abs() ** 2
    bx = torch.einsum("btchw,nhw->btcn", px, oh)
    by = torch.einsum("btchw,nhw->btcn", py, oh)
    e = eps * by.mean().clamp(min=1e-12)
    return (torch.log(bx + e) - torch.log(by + e)).pow(2).mean()


def dt_rel_l2(p, t, order=1):
    """Relative L2 of the TIME DERIVATIVE of the block, per sample.

    order=1 : central difference, O(dt^2)          d[k] = (x[k+1] - x[k-1]) / 2
    order=2 : second difference,  O(dt^2)          a[k] = x[k+1] - 2 x[k] + x[k-1]
    dt is a constant here (measured min 0.01999855 / max 0.02000046 over the release), so it
    cancels in a RELATIVE norm and is omitted.  Axis 1 is TIME, as in tke_rel_l2.
    """
    import torch          # this file imports torch per-function, not at module scope
    if order == 1:
        dp = 0.5 * (p[:, 2:] - p[:, :-2])
        dt_ = 0.5 * (t[:, 2:] - t[:, :-2])
    else:
        dp = p[:, 2:] - 2.0 * p[:, 1:-1] + p[:, :-2]
        dt_ = t[:, 2:] - 2.0 * t[:, 1:-1] + t[:, :-2]
    num = torch.sqrt(((dp - dt_) ** 2).sum(dim=tuple(range(1, dp.ndim))))
    den = torch.sqrt((dt_ ** 2).sum(dim=tuple(range(1, dt_.ndim)))).clamp_min(1e-12)
    return num / den


def tke_rel_l2(pred, tgt, eps=1e-8, chan_std=None):
    """scoring.py::tke_rel_l2_per_sample -- relative L2 of the TKE field.  -> (B,)

    TKP (2026-08-15): `chan_std` is the per-channel denormalisation std.  Supplied, the variances
    are put back in PHYSICAL units before the two channels are summed -- which is what the scorer
    does.  Omitted (the default, and every existing caller), behaviour is bit-identical to before.
    Why it matters: var_t(u_norm) = var_t(u_phys)/s_u^2, so summing the two channels in normalised
    space weights them by 1/s_u^2 and 1/s_v^2, a ratio of 36.79.  The MEAN cancels in a relative
    norm (as the note at the tke_field docstring says); the per-channel SCALE does not.
    """
    import torch
    n = pred.shape[0]
    if chan_std is None:
        p = tke_field(pred).reshape(n, -1)
        t = tke_field(tgt).reshape(n, -1)
    else:
        _s2 = (chan_std ** 2).reshape(1, 1, 1, -1)
        p = (tke_field(pred, per_channel=True) * _s2).sum(-1).mul(0.5).reshape(n, -1)
        t = (tke_field(tgt, per_channel=True) * _s2).sum(-1).mul(0.5).reshape(n, -1)
    return torch.norm(p - t, dim=1) / torch.norm(t, dim=1).clamp(min=eps)


def mvpe_rel_l2(pred, tgt, sub_s_real=2, eps=1e-8):
    """Differentiable port of scoring.py::mvpe_rel_l2_per_sample.

    pred/tgt are (N, T, h, w, >=2) in PHYSICAL units.  Returns (N,).  Same probe geometry as
    the official scorer: 4 columns behind the airfoil, 9 y-positions, compared on the
    TIME-AVERAGED profile.  Identical to `eval_v3.py::mvpe_rel_l2_t`, which is checked against
    scoring.py under --selfcheck; `mvpe_selfcheck()` below re-runs that check at startup.
    """
    import torch
    d, center_x, center_y, n_probe = 16, 10, 32, 9
    N, _, h, w, _ = pred.shape
    pcy = int(center_y / sub_s_real)
    iy = min(2, int(h / (n_probe + 1)))
    probe_y = [pcy + iy * j for j in range(-(n_probe - 1) // 2, n_probe - (n_probe - 1) // 2)]
    probe_y = [y for y in probe_y if 0 <= y < h]
    if not probe_y:
        return torch.zeros(N, device=pred.device, dtype=pred.dtype)
    errs = []
    for i in range(4):
        if int((2 * d + center_x) / sub_s_real) < w:
            px = int(((i + 1) * d + center_x) / sub_s_real)
        else:
            px = int((0.5 * (i + 2) * d + center_x) / sub_s_real)
        if not 0 <= px < w:
            continue
        pp = pred[:, :, probe_y, px, :2].mean(1).reshape(N, -1)
        tt = tgt[:, :, probe_y, px, :2].mean(1).reshape(N, -1)
        errs.append(torch.norm(pp - tt, dim=1) / torch.norm(tt, dim=1).clamp(min=eps))
    if not errs:
        return torch.zeros(N, device=pred.device, dtype=pred.dtype)
    return torch.stack(errs, 0).mean(0)


def mvpe_selfcheck(kit_dir, t_out, h, w, sub_s, logger, tol=1e-5):
    """Assert the torch mvpe port matches the kit's scoring.py on random data.  A silently
    wrong loss term is worse than no loss term, so this runs before any training step."""
    import torch
    spec = importlib.util.spec_from_file_location("t1_scoring_chk", Path(kit_dir) / "scoring.py")
    off = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(off)
    p = np.random.default_rng(0).normal(size=(4, t_out, h, w, 3)).astype(np.float32)
    t = np.random.default_rng(1).normal(size=(4, t_out, h, w, 3)).astype(np.float32)
    want = np.asarray(off.mvpe_rel_l2_per_sample(p, t, sub_s), np.float64)
    got = mvpe_rel_l2(torch.from_numpy(p), torch.from_numpy(t), sub_s).numpy().astype(np.float64)
    dmax = float(np.max(np.abs(got - want)))
    logger.info(f"MVPE SELFCHECK vs scoring.py: max|d|={dmax:.3g} (tol {tol:g})")
    if dmax > tol:
        raise SystemExit(f"MVPE self-check FAILED (max|d|={dmax:.3g}) -- refusing to train with "
                         f"a loss term that does not match the scorer")


def feedback(pred, xb, lch, cur=None, s_roll=None):
    """Model output -> a VALID next input.

    Channels >= lch carry no loss, so the net's values there are unconstrained garbage.
    The data pipeline sets those channels to physical zero, which after per-source
    normalisation is a CONSTANT field whose value differs between the sim and real stages.
    Reading that constant back out of xb (channel is uniform, so any single element is
    exact) keeps this source-agnostic and avoids poisoning the next block's input.
    """
    import torch
    if pred.shape[-1] <= lch:
        nxt = pred
    else:
        const = xb[:, :1, :1, :1, lch:]
        fill = const.expand(pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3],
                            pred.shape[4] - lch)
        nxt = torch.cat([pred[..., :lch], fill], dim=-1)
    # SHORT-HORIZON unrolling: advance the input window by s < t_out, so the next input is
    # (t_in - s) TRUE frames followed by only s self-generated ones.  s = t_out (the default)
    # hands the whole window back and is the old behaviour.
    if cur is None or s_roll is None or s_roll >= nxt.shape[1]:
        return nxt
    return torch.cat([cur[:, s_roll:], nxt[:, :s_roll]], dim=1)


# ===================== MIMOSCALE (2026-08-27): the MIMO slot helpers =====================
# Factored out of main() so they are TESTABLE: a `null` check asserts, against
# the pre-generalization file recovered from git HEAD, that at M=2 / rho_rep=0.0 each one is
# bit-identical to the inline expression it replaces -- same tensor, same generator state,
# zero extra RNG consumed.  Both are pure; neither touches global state.
def _mimo_pair_idx(idx, n, M, rep, gen):
    """The M-1 partner-window draws for one MIMO batch.

    Each partner rides the PRIVATE generator (the CORALA precedent, seed 20260827) so the
    PRIMARY batch stream -- and hence the control-matched minibatch sequence -- is untouched.
    `rep` is Havasi's INPUT REPETITION (arXiv:2010.06610 Sec. 3.2): with probability `rep`
    a slot repeats slot 0's window instead of an independent draw, the paper's remedy when
    the shared width is too tight for M independent subnetworks.  rep == 0.0 takes the
    `if` branch never and therefore consumes EXACTLY the draws the pre-MIMOSCALE inline
    code consumed.
    """
    import torch          # this file imports torch per-function, not at module scope
    out = []
    for _ in range(M - 1):
        j = torch.randint(0, n, (int(idx.numel()),), generator=gen)
        if rep > 0.0:
            keep = torch.rand((int(idx.numel()),), generator=gen) < float(rep)
            j = torch.where(keep, idx, j)
        out.append(j)
    return out


def _mimo_fold(t, M, c=3):
    """Fold the M-slot channel axis into the BATCH: (B, ..., M*c) -> (M*B, ..., c).

    At M=2 this is exactly `torch.cat([t[..., :c], t[..., c:]], 0)`, the expression it
    replaces.  Slot order is preserved (slot k occupies batch rows [k*B, (k+1)*B)), which
    is what makes the per-sample loss reduction equal the mean of the M heads' losses
    against their own targets.
    """
    import torch          # this file imports torch per-function, not at module scope
    return torch.cat([t[..., k * c:(k + 1) * c] for k in range(M)], 0)


def main():
    import torch
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cfg = resolve_paths(yaml.safe_load(open(args.config)))

    variant = cfg.get("variant", "UNet_v4")
    sub_s = int(cfg.get("sub_s", 2))
    t_in = int(cfg.get("t_in", 20))
    t_out = int(cfg.get("t_out", 20))
    base = int(cfg.get("unet_base", 64))
    bs = int(cfg.get("batch_size", 16))
    eval_every = int(cfg.get("eval_every", 250))
    stride = int(cfg.get("train_stride", 10))
    ema_decay = float(cfg.get("ema_decay", 0.999))
    switch_ema_every = int(cfg.get("switch_ema_every", 0))
    # Checkpoint selection criterion.  "rel" (default) is the historical rule -- minimum val
    # rel-L2 -- and reduces the key below to d["rel"] exactly, so every existing arm is unchanged.
    # "s3" selects on what the benchmark actually scores: the mean of the three error subscores.
    sel_on = str(cfg.get("select_on", "rel")).lower()
    if sel_on not in ("rel", "s3"):
        raise SystemExit(f"select_on must be 'rel' or 's3', got {sel_on!r}")

    def _sel_key(d):
        """Lower is better, for both rules, so the comparison below does not branch."""
        if sel_on == "rel":
            return float(d["rel"])
        sc = lambda e: 100.0 / (1.0 + 0.5 * float(e))     # noqa: E731  -- scoring.py's score_error
        return -(sc(d["rel"]) + sc(d["tke"]) + sc(d["mvpe"])) / 3.0

    lch = int(cfg.get("loss_channels", 2))
    # Kept separate because the stage loop REBINDS `lch`; without this, stage 2 would default to
    # stage 1's override instead of to the config's global value.
    _LCH_GLOBAL = lch
    global _LR_WARMUP, _LR_MINR
    _LR_WARMUP = float(cfg.get("lr_warmup", 0.03))
    _LR_MINR = float(cfg.get("lr_min_ratio", 0.02))
    clip = float(cfg.get("grad_clip", 1.0))
    wd = float(cfg.get("weight_decay", 1e-5))
    excl = set(int(x) for x in (cfg.get("exclude_re") or []))
    ckpt_out = cfg["checkpoint"]

    # --- v3 knobs ---
    loss_type = str(cfg.get("loss_type", "l2"))
    l2_norm = bool(cfg.get("l2_normalize", True))
    # LWa (2026-08-13): per-lead LOSS WEIGHT.  "" (default) is BIT-EXACT OFF -- make_loss returns
    # its original closures, no new line runs and no RNG is consumed.  Motivation: the RELF
    # re-measurement of the analogue predictability floor (2026-08-13) is
    # lead-resolved on 9/9 val trajectories and finds the intrinsic floor grows only 1.22x from
    # lead 1 to lead 20 while OUR error grows 1.90x (ratio 1.55x), with 49% of the recoverable
    # headroom in leads 16-20 against 5% in leads 1-5 -- yet every loss in this file weights all
    # 20 output frames EQUALLY, so the axis has never been swept.  Distinct from the CLOSED
    # per-lead persistence blend (dPriced -0.0644) and per-lead BAND shape (+0.048), neither of
    # which touched training.  See parse_lead_weight for the schemes and the mean-1.0 assert.
    lead_weight = str(cfg.get("lead_weight", "") or "")
    _lead_w = parse_lead_weight(lead_weight, t_out)
    n_roll = int(cfg.get("n_roll", 1))
    unroll_mode = str(cfg.get("unroll_mode", "detach"))
    unroll_warmup = float(cfg.get("unroll_warmup", 0.3))
    roll_stride = int(cfg.get("unroll_stride", t_out))   # feedback step; t_out = old behaviour
    tke_w = float(cfg.get("tke_weight", 0.0))
    pgrad_w = float(cfg.get("pgrad_weight", 0.0))   # SPGa: sim-stage grad-p supervision
    # WRa (2026-08-13): regime-weighted sampling in the REAL-FINETUNE stage only.  When
    # regime_weight w > 0, per-window sampling probability becomes proportional to
    # (1 + w * rank_norm(Re)) where rank_norm ranks the window's trajectory Re among the
    # distinct Re values present in the stage's train pool, normalised to [0, 1] (linear
    # tilt: at w=1.0 the top-Re windows are sampled 2x the bottom-Re ones).  Default 0.0 is
    # BIT-EXACT OFF: the sampling line stays the original torch.randint and no new code
    # consumes RNG.  Motivation: RX1/BRXa measured the online sps deficit as high-Re
    # regime-shift residual mass (2026-08-12); prior art RWa (AoA-axis file-stride
    # upweighting, 0/5 folds, 2026-08-02) closed the lever ON THAT AXIS only.
    regime_w = float(cfg.get("regime_weight", 0.0))
    if regime_w < 0.0:
        raise SystemExit(f"regime_weight must be >= 0.0, got {regime_w}")
    # >>> CLUSTERW knob (2026-09-10): flow-STATE-cluster inverse-frequency draw of the REAL-FINETUNE windows
    # (CLUSTERW arm; the weight file is a fitted npz).  state_weight: {file: <npz>, alpha: a}.
    # ABSENT (default) == BIT-PATH-IDENTICAL off: _sw_cfg is None, every armed branch is `if _sw_cfg is not None` /
    # `elif _sw_probs is not None`, no RNG is consumed, no tensor is touched.  One sampling tilt per stage.
    _sw_cfg = cfg.get("state_weight", None)
    if _sw_cfg is not None:
        if not isinstance(_sw_cfg, dict) or set(_sw_cfg) != {"file", "alpha"}:
            raise SystemExit(f"state_weight must be {{file: <npz>, alpha: <float>}}, got {_sw_cfg!r}")
        _sw_cfg = {"file": str(_sw_cfg["file"]), "alpha": float(_sw_cfg["alpha"])}
        if not _sw_cfg["alpha"] > 0.0:
            raise SystemExit(f"state_weight.alpha must be > 0 (alpha 0 IS the control), got {_sw_cfg['alpha']}")
        if not Path(_sw_cfg["file"]).is_file():
            raise SystemExit(f"state_weight.file not found: {_sw_cfg['file']}")
        if regime_w > 0.0:
            raise SystemExit("state_weight and regime_weight are mutually exclusive (one sampling tilt per stage)")
    # <<< CLUSTERW knob (2026-09-10)
    # AOSIM (2026-08-26): sim-pretrain-stage FILE-sampling weight for the separated regime
    # (AoA in {15, 20}).  Default 1.0 is BIT-PATH-IDENTICAL off: the sampling line stays the
    # original torch.randint and no new code consumes RNG (same additive contract as
    # WRa/SAMa/AXP above).  Gate evidence: the sim-only end-of-
    # pretrain checkpoint's high/low-AoA skill ratio-of-ratios = 1.1861 on held-out sim).
    sim_aoa_w = float(cfg.get("sim_aoa_weight", 1.0))
    if not sim_aoa_w > 0.0:
        raise SystemExit(f"sim_aoa_weight must be > 0.0, got {sim_aoa_w}")
    distill_w = float(cfg.get("distill_weight", 0.0))
    distill_ckpt = str(cfg.get("distill_ckpt", ""))
    # DFLUCT: optional temporal-fluctuation KD relaxation; alpha1 preserves the literal old call.
    _dfl_alpha = cfg.get("distill_fluct_alpha", 1.0)
    if type(_dfl_alpha) not in (int, float) or not 0.0 <= _dfl_alpha <= 1.0:
        raise SystemExit("distill_fluct_alpha must be a finite number in [0,1]")
    _dfl_seen = {"calls": 0}
    if _dfl_alpha != 1.0:
        if not (distill_w > 0 and distill_ckpt and cfg.get("loss_type") == "rel_l2"
                and bool(cfg.get("l2_normalize", True)) and int(cfg.get("loss_channels", 2)) == 2
                and int(cfg.get("n_roll", 1)) == 1):
            raise SystemExit("active distill_fluct_alpha requires weighted normalized rel_l2, u/v, one rollout and a teacher")
        if (any(cfg.get(k) for k in ("strideaug", "tooth", "distill_sched", "distill_arm", "coblend"))
                or cfg.get("fuse") not in (None, "", "none")):
            raise SystemExit("active distill_fluct_alpha currently requires the plain unaugmented single-model loss")
        if float(cfg.get("sam_rho", 0.0)) > 0.0:
            raise SystemExit("active distill_fluct_alpha requires one loss application per step; SAM is unsupported")
        if (str(cfg.get("pretext", "none")) != "none"
                or any(str(q.get("pretext", "none")) != "none" for q in cfg.get("stages", []))):
            raise SystemExit("active distill_fluct_alpha requires pretext none globally and in every stage")
        _dfl_real_stages = [q for q in cfg.get("stages", []) if q.get("name") == "real-finetune"]
        if len(_dfl_real_stages) != 1:
            raise SystemExit("active distill_fluct_alpha requires exactly one real-finetune stage")
        _dfl_real = _dfl_real_stages[0]
        if (str(_dfl_real.get("objective", "block")) != "block"
                or int(_dfl_real.get("loss_channels", cfg.get("loss_channels", 2))) != 2
                or int(_dfl_real.get("n_roll", cfg.get("n_roll", 1))) != 1):
            raise SystemExit("active distill_fluct_alpha requires real-finetune objective block, u/v and one rollout")
        sys.path.insert(0, str(CASE_DIR.parent.parent / "common"))
        from temporal_distillation import temporal_distillation_loss as _dfl_loss
    # DSPAT: the SPATIAL mirror of distill_fluct_alpha -- shrink only the small-scale part of the KD
    # residual, so the teacher keeps pulling at the large scales (where it earns rel) and stops
    # flattening the small scales the tke subscore measures.  beta=1 is the literal old call.
    _dsp_beta = cfg.get("distill_spatial_beta", 1.0)
    _dsp_kern = int(cfg.get("distill_spatial_kernel", 3))
    if type(_dsp_beta) not in (int, float) or not 0.0 <= _dsp_beta <= 1.0:
        raise SystemExit("distill_spatial_beta must be a finite number in [0,1]")
    if _dsp_kern < 3 or _dsp_kern % 2 == 0:
        raise SystemExit(f"distill_spatial_kernel must be an odd int >= 3, got {_dsp_kern}")
    _dsp_seen = {"calls": 0}
    if _dsp_beta != 1.0:
        if _dfl_alpha != 1.0:
            raise SystemExit("distill_spatial_beta and distill_fluct_alpha together would confound two "
                             "residual splits in one arm; run them separately")
        if not (distill_w > 0 and distill_ckpt and cfg.get("loss_type") == "rel_l2"
                and bool(cfg.get("l2_normalize", True)) and int(cfg.get("loss_channels", 2)) == 2
                and int(cfg.get("n_roll", 1)) == 1):
            raise SystemExit("active distill_spatial_beta requires weighted normalized rel_l2, u/v, one rollout and a teacher")
        if (any(cfg.get(k) for k in ("strideaug", "tooth", "distill_sched", "distill_arm", "coblend"))
                or cfg.get("fuse") not in (None, "", "none")):
            raise SystemExit("active distill_spatial_beta currently requires the plain unaugmented single-model loss")
        if float(cfg.get("sam_rho", 0.0)) > 0.0:
            raise SystemExit("active distill_spatial_beta requires one loss application per step; SAM is unsupported")
        _dsp_real = [q for q in cfg.get("stages", []) if q.get("name") == "real-finetune"]
        if len(_dsp_real) != 1:
            raise SystemExit("active distill_spatial_beta requires exactly one real-finetune stage")
        sys.path.insert(0, str(CASE_DIR.parent.parent / "common"))
        from spatial_distillation import spatial_distillation_loss as _dsp_loss
    # >>> DWSCHED knob (2026-09-10): a linear distillation-weight SCHEDULE across the real-finetune stage at a FIXED dose
    # (DWSCHED arm).  Config line:
    #   distill_sched: {start: a, end: b, shape: linear, stage: real-finetune}
    # PRECEDENCE: `distill_weight` stays the ARMING GATE (teachers load iff > 0 with a distill_ckpt) AND must equal the schedule's
    # time-mean (a+b)/2; inside that stage w(s) = a + (b-a)*s/(S-1) is the weight at EVERY step (the schedule wins per step) by
    # rebinding `distill_w` -- the closure cell step_loss / COBLEND / SAM read -- and the constant is restored at stage end.
    # ABSENT (default) == the historical path: _dws is None, every armed statement sits under `if _dws is not None`, no RNG is
    # consumed, no tensor is touched, the checkpoint meta bytes are unchanged.
    _dws = cfg.get("distill_sched", None)   # DWSCHED knob (2026-09-10)
    if _dws is not None:   # DWSCHED knob (2026-09-10): refusals at config parse, before any data load
        if not isinstance(_dws, dict) or set(_dws) != {"start", "end", "shape", "stage"}:
            raise SystemExit(f"distill_sched must be exactly {{start: a, end: b, shape: linear, stage: real-finetune}}, got {_dws!r}")
        _dws = {"start": float(_dws["start"]), "end": float(_dws["end"]), "shape": str(_dws["shape"]),
                "stage": str(_dws["stage"]), "const": float(distill_w)}
        if not (distill_w > 0.0 and distill_ckpt.strip()):
            raise SystemExit(f"distill_sched needs teachers: distill_weight > 0 AND a non-empty distill_ckpt "
                             f"(got distill_weight={distill_w}, distill_ckpt={distill_ckpt!r})")
        if _dws["shape"] != "linear":
            raise SystemExit(f"distill_sched.shape must be 'linear', got {_dws['shape']!r}")
        if _dws["stage"] != "real-finetune":
            raise SystemExit(f"distill_sched.stage must be 'real-finetune' (distillation is real-finetune only), got {_dws['stage']!r}")
        if not (_dws["start"] > 0.0 and _dws["end"] > 0.0):
            raise SystemExit(f"distill_sched needs start > 0 and end > 0 (a zero weight would switch the teacher leg off "
                             f"mid-stage), got start={_dws['start']} end={_dws['end']}")
        if _dws["start"] == _dws["end"]:
            raise SystemExit(f"distill_sched: start == end == {_dws['start']} is a flat schedule -- that IS the constant control")
        if abs(distill_w - 0.5 * (_dws["start"] + _dws["end"])) >= 1e-9:
            raise SystemExit(f"distill_sched time-mean {0.5 * (_dws['start'] + _dws['end'])} != distill_weight {distill_w}: the "
                             f"constant key is the schedule's mean (a dose change is a distill_weight edit, not this knob)")
        if cfg.get("distill_arm"):
            raise SystemExit("distill_sched cannot combine with distill_arm (target blending is a different mechanism)")
        _dws_st = [q for q in cfg["stages"] if q.get("name") == _dws["stage"]]
        if len(_dws_st) != 1:
            raise SystemExit(f"distill_sched needs exactly one stage named {_dws['stage']!r}, the config has {len(_dws_st)}")
        if int(_dws_st[0]["steps"]) < 2:
            raise SystemExit(f"distill_sched needs >= 2 steps in {_dws['stage']!r}, got {_dws_st[0]['steps']}")
        _dws_seen = {"n": 0, "sum": 0.0, "first": None, "last": None}
    # <<< DWSCHED knob (2026-09-10)
    tspec_w = float(cfg.get("tspec_weight", 0.0))
    # HPL1 (2026-09-20): relative L1 on the HIGH-PASS part of the field.  A full-field L2 is dominated by
    # the mean flow and the large eddies, so the conditional-mean solution can shrink the small scales almost
    # for free -- and the small scales are exactly what `tke` scores (the per-pixel temporal variance).  L1,
    # not L2, on that component: an L1 optimum is the conditional MEDIAN, which stays sharp when the phase is
    # uncertain, where the mean blurs.  Normalised per sample by the target's own high-pass magnitude, so the
    # weight is scale-free and comparable with rel_l2.  0.0 == OFF == the historical bit path.
    hpl1_w = float(cfg.get("hpl1_weight", 0.0))
    hpl1_mode = str(cfg.get("hpl1_mode", "temporal"))
    if hpl1_w > 0.0 and hpl1_mode not in ("temporal", "spatial", "both"):
        raise SystemExit(f"hpl1_mode must be temporal|spatial|both, got {hpl1_mode!r}")
    # HPL1S: which spatial kernels the high-pass uses. Default [3] reproduces every arm measured so
    # far BIT-FOR-BIT, so this option is off unless a config names it.
    hpl1_scales = [int(x) for x in str(cfg.get("hpl1_scales", "3")).replace(",", " ").split()]
    if hpl1_w > 0.0 and hpl1_mode in ("spatial", "both"):
        if not hpl1_scales or any(k < 3 or k % 2 == 0 for k in hpl1_scales):
            raise SystemExit(f"hpl1_scales must be odd integers >= 3, got {hpl1_scales!r}")
    lowk_w = float(cfg.get("lowk_weight", 0.0))     # LKa: large-scale relative-L2 term
    lowk_cut = float(cfg.get("lowk_cut", 0.25))     # radial cutoff as a fraction of Nyquist
    tspec_drop_top = bool(cfg.get("tspec_drop_top", True))
    dt1_w = float(cfg.get("dt1_weight", 0.0))
    dt2_w = float(cfg.get("dt2_weight", 0.0))
    # SAMa (2026-08-12): Sharpness-Aware Minimization (Foret et al. 2020, arXiv 2010.01412).
    # 0.0 (default) is an exact no-op: a single `if sam_rho > 0.0` gates every new line, no
    # extra forward runs, no RNG is consumed, every existing config is bit-identical.  When
    # armed it applies in the REAL-FINETUNE stage ONLY (the sim pretrain stays untouched so
    # the arm is one-variable against SDa), doubling that stage's per-step compute: ascend
    # rho * g/||g||_2 from the clean weights, recompute the SAME objective on the SAME batch
    # at the perturbed point, restore the weights bit-exactly from clones, then let the
    # ordinary opt.step() below consume the perturbed-point gradients.
    sam_rho = float(cfg.get("sam_rho", 0.0))
    _sam_state = {"checked": False}
    if sam_rho > 0.0:
        if float(cfg.get("mixup_alpha", 0.0) or 0.0) > 0.0:
            # mixup itself is fine (it acts on the batch BEFORE both passes), but nobody has
            # armed the pair; refuse rather than silently compose two regularisers in one arm.
            raise SystemExit("sam_rho > 0 with mixup_alpha > 0: one regulariser per arm")
    # AXP (2026-08-12): auxiliary sub-phase supervision.  0.0 (default) is an exact no-op --
    # no module is created, no RNG is consumed, every existing config is bit-identical.
    aux_w = float(cfg.get("aux_phase_weight", 0.0))
    AUX_PHASES = [(0, 1), (1, 0), (1, 1)]     # the 3 unseen stride-2 subsamplings
    if aux_w > 0.0:
        if str(cfg.get("arch", "unet")) != "unet" or cfg.get("refine"):
            raise SystemExit("aux_phase_weight > 0 needs the plain arch: unet (no refine): the "
                             "aux head lives in UNet2D and the wrapper would hide _last_aux")
        if (cfg.get("out_mode") and str(cfg.get("out_mode")) != "direct") or cfg.get("head"):
            raise SystemExit("aux_phase_weight > 0 composes with the direct head only")
        if int(cfg.get("n_roll", 1)) != 1 or any(int(s.get("n_roll", 1)) != 1
                                                 for s in cfg.get("stages", [])):
            raise SystemExit("aux_phase_weight > 0 requires n_roll == 1 everywhere: the aux "
                             "target is cut for the single-block window only")
        if float(cfg.get("mixup_alpha", 0.0) or 0.0) > 0.0 or \
           float(cfg.get("rot_aug_deg", 0.0) or 0.0) > 0.0:
            raise SystemExit("aux_phase_weight > 0 is incompatible with mixup/rot_aug: both "
                             "transform yb but not the aux targets")
        if int(cfg.get("sub_s", 2)) != 2:
            raise SystemExit("aux_phase_weight > 0 assumes sub_s == 2 (the 3 sibling phases)")
        for _st in cfg.get("stages", []):
            for _sr in _st.get("sources", []):
                if _sr.get("phases"):
                    raise SystemExit("aux_phase_weight > 0 cannot combine with a source-level "
                                     "`phases` augmentation (P4): undefined together")

    # --- v4 knobs ---
    mvpe_w = float(cfg.get("mvpe_weight", 0.0))
    # INPUT-NOISE AUGMENTATION (2026-07-31).  Multiplier on the MEASURED PIV noise level;
    # 0.0 is an exact no-op and every pre-existing config reproduces bit-for-bit.  See
    # `add_input_noise.py` for why this axis and why this scale.
    in_noise = float(cfg.get("input_noise", 0.0))
    # "add"  = isotropic in physical units, the PIV-floor model this repo assumed (arms NZ1/NZ3).
    # "mult" = proportional to the local physical value, which is what the benchmark's own
    #          fluid_dataset.py:379 does and what PIV displacement error actually looks like.
    noise_mode = str(cfg.get("noise_mode", "add"))
    noise_stage = str(cfg.get("noise_stage", "all"))
    if noise_stage not in ("all", "sim-pretrain", "real-finetune"):
        raise SystemExit(f"noise_stage must be all|sim-pretrain|real-finetune, got {noise_stage!r}")
    if noise_mode not in ("add", "mult"):
        raise SystemExit(f"noise_mode must be add|mult, got {noise_mode!r}")
    bsp_w = float(cfg.get("bsp_weight", 0.0))
    bsp_nb = int(cfg.get("bsp_bins", 8))
    kit_dir = cfg.get("kit_dir") or str(CASE_DIR.parent / "realpde_t1_starting_kit_v6")
    fold = args.fold if args.fold is not None else cfg.get("fold")
    fold = None if fold is None else int(fold)
    fold_val_files, fold_val_re, group_by = None, None, "re"
    key_fn = file_re
    if fold is not None:
        man = resolve_paths(json.loads(Path(cfg["kfold_manifest"]).read_text()))
        fd = next(f for f in man["folds"] if int(f["fold"]) == fold)
        group_by = str(man.get("group_by", "re"))
        key_fn = GROUP_FN[group_by]
        fold_val_re = [int(r) for r in (fd.get("val_groups") or fd["val_re"])]
        fold_val_files = list(fd["val_files"])
        # the fold's held-out group values must be excluded from EVERY source (sim shares both
        # the Reynolds and the AoA axis with real, so leaving them in leaks via pretraining)
        excl = set(fold_val_re)
        variant = f"{variant}_f{fold}"
        cfg["desc"] = f"{cfg.get('desc', 'unet_v4')}_f{fold}"
        p = Path(ckpt_out)
        ckpt_out = str(p.with_name(f"{p.stem}_f{fold}{p.suffix}"))
    elif cfg.get("kfold_manifest"):
        raise SystemExit("this config declares kfold_manifest but no --fold was given -- it has no "
                         "exclude_re/val_glob of its own, so running it without a fold would "
                         "validate on training data.  Pass --fold 0..K-1.")
    if n_roll < 1:
        raise SystemExit("n_roll must be >= 1")
    if not 1 <= roll_stride <= t_out:
        raise SystemExit(f"unroll_stride must be in [1, t_out={t_out}], got {roll_stride}")
    if unroll_mode not in ("detach", "pushforward", "bptt"):
        raise SystemExit(f"unroll_mode must be detach|pushforward|bptt, got {unroll_mode!r}")
    if n_roll > 1 and t_in != t_out:
        raise SystemExit(f"unrolling needs t_in == t_out so the output is a valid input "
                         f"(got t_in={t_in}, t_out={t_out})")

    # Seeded so the arms of the loss ablation see the SAME minibatch sequence and any
    # difference between them is the objective, not the sampler.  (Arms with different
    # n_roll have different window counts, so only same-n_roll arms are exactly paired.)
    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = Path(cfg.get("logger_root") or CASE_DIR) / "Logger" / variant / f"{ts}-{cfg.get('desc','unet_v4')}"
    (run_dir / "states").mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir)
    shutil.copy(args.config, run_dir / "config.yaml")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    tke_phys = bool(cfg.get("tke_phys", False))          # TKP: False = OFF, bit-exact
    _hpl1_seen = {"live": False}                         # HPL1 execution witness
    _pd_on = bool(cfg.get("phys_denom", False))          # CWD: False = OFF, bit-exact
    _pd_box = {"ms": None} if _pd_on else None
    if _pd_on:
        logger.info("CWD:          phys_denom -- rel_l2 denominator in PHYSICAL units, matching "
                    "scoring.py:89; without it training optimises ||e_phys||/||y_norm|| while the "
                    "scorer measures ||e_phys||/||t_phys||")
    _cw_spec = str(cfg.get("chan_weight", "")).strip()   # CW: "" = OFF, bit-exact
    _chan_w = None
    if _cw_spec:
        _chan_w = [float(x) for x in _cw_spec.split(",")]
        if len(_chan_w) < 2 or min(_chan_w) <= 0.0:
            raise SystemExit(f"chan_weight must be >=2 positive numbers, got {_cw_spec!r}")
        logger.info(f"CW:           chan_weight={_chan_w} -- per-channel NUMERATOR factor; the "
                    f"scorer flattens u,v jointly in PHYSICAL units while _rel reduces the "
                    f"NORMALISED tensor, a std_u^2/std_v^2 = 36.79 mismatch")
    sps_lam = float(cfg.get("sps_lam", 0.0))       # SPSW: 0.0 = OFF, bit-exact
    sps_s = float(cfg.get("sps_s", 0.25))          # bump width in units of |r|/h
    _sps_box = {"w": None} if sps_lam > 0.0 else None

    # ============== WAVE 2 (2026-08-27): four additive default-off loss/transfer knobs ==========
    # CORALA (`feat_align: coral` + `feat_align_w`) / JAWSW (`loss_weight_map: prec|antiprec` +
    # `lwm_strength`) / SEMIG (`semigroup_w`) / GDROA (`group_dro_eta`).  Each defaults to
    # none/0 == BIT-PATH-IDENTICAL off: no module, no hook, no elem box, no RNG consumed (the
    # WRa/SAMa/AXP/AOSIM/SOBG additive contract).  One knob per arm: any two armed together are
    # refused (WAVE2).
    fa_mode = str(cfg.get("feat_align", "none") or "none").lower()
    fa_w = float(cfg.get("feat_align_w", 0.0))
    lwm_mode = str(cfg.get("loss_weight_map", "none") or "none").lower()
    lwm_s = float(cfg.get("lwm_strength", 1.0))
    semi_w = float(cfg.get("semigroup_w", 0.0))
    gdro_eta = float(cfg.get("group_dro_eta", 0.0))
    if fa_mode not in ("none", "coral"):
        raise SystemExit(f"feat_align must be none|coral, got {fa_mode!r}")
    if lwm_mode not in ("none", "prec", "antiprec"):
        raise SystemExit(f"loss_weight_map must be none|prec|antiprec, got {lwm_mode!r}")
    if semi_w < 0.0 or gdro_eta < 0.0 or fa_w < 0.0:
        raise SystemExit("semigroup_w / group_dro_eta / feat_align_w must be >= 0.0, got "
                         f"({semi_w}, {gdro_eta}, {fa_w})")
    if lwm_s <= 0.0:
        raise SystemExit(f"lwm_strength must be > 0.0, got {lwm_s}")
    _w2_armed = [k for k, on in (("feat_align", fa_mode != "none"),
                                 ("loss_weight_map", lwm_mode != "none"),
                                 ("semigroup_w", semi_w > 0.0),
                                 ("group_dro_eta", gdro_eta > 0.0)) if on]
    if len(_w2_armed) > 1:
        raise SystemExit(f"wave-2 knobs are one-mechanism-per-arm: {_w2_armed} armed together")
    if _w2_armed:
        _k2 = _w2_armed[0]
        if fa_mode != "none" and fa_w <= 0.0:
            raise SystemExit(f"feat_align={fa_mode} needs feat_align_w > 0, got {fa_w}")
        if sps_lam > 0.0:
            raise SystemExit(f"{_k2} cannot combine with sps_lam > 0 (one elem-weight box)")
        if sam_rho > 0.0:
            raise SystemExit(f"{_k2} cannot combine with sam_rho > 0 (SAM re-evaluates step_loss; "
                             f"EMA/capture state would double-update)")
        if float(cfg.get("mixup_alpha", 0.0) or 0.0) > 0.0:
            raise SystemExit(f"{_k2} cannot combine with mixup (one regulariser per arm)")
        if aux_w > 0.0:
            raise SystemExit(f"{_k2} cannot combine with aux_phase_weight")
        if n_roll != 1 or any(int(s.get("n_roll", 1)) != 1 for s in cfg.get("stages", [])):
            raise SystemExit(f"{_k2} requires n_roll == 1 everywhere")
        _w2_bad = [f"{s.get('name')}:{s.get('objective', 'block')}/{s.get('pretext', 'none')}"
                   for s in cfg.get("stages", [])
                   if str(s.get("objective", "block")) != "block"
                   or str(s.get("pretext", "none")) != "none"]
        if _w2_bad:
            raise SystemExit(f"{_k2} needs objective: block / pretext: none in every stage, "
                             f"got {_w2_bad}")
    if fa_mode != "none":
        if str(cfg.get("arch", "unet")) != "unet" or cfg.get("refine"):
            raise SystemExit("feat_align needs the plain arch: unet (no refine): the bottleneck "
                             "hook lives on model.d2")
        _w2_names = [str(s.get("name")) for s in cfg.get("stages", [])]
        if "sim-pretrain" not in _w2_names or "real-finetune" not in _w2_names or \
                _w2_names.index("sim-pretrain") > _w2_names.index("real-finetune"):
            raise SystemExit(f"feat_align needs a sim-pretrain stage BEFORE real-finetune (its "
                             f"sim batches are the sim stage's own windows), got {_w2_names}")
    if (lwm_mode != "none" or gdro_eta > 0.0) and (
            lch != 2 or any(int(s.get("loss_channels", lch)) != 2
                            for s in cfg.get("stages", []))):
        raise SystemExit(f"{_w2_armed[0]} requires loss_channels == 2 in every stage")
    if gdro_eta > 0.0:
        if loss_type not in ("l2", "rel_l2"):
            raise SystemExit(f"group_dro_eta > 0 needs loss_type l2|rel_l2 (the per-sample w^2 "
                             f"elem weight equals w on the per-sample loss ONLY inside a sqrt "
                             f"reduction), got {loss_type!r}")
        if bool(cfg.get("phys_denom", False)):
            raise SystemExit("group_dro_eta > 0 with phys_denom is not implemented (the group-"
                             "loss proxy would diverge from the trained objective)")
    if semi_w > 0.0:
        if t_in != t_out or t_out % 2 != 0:
            raise SystemExit(f"semigroup_w > 0 needs t_in == t_out and even t_out, got "
                             f"({t_in}, {t_out})")
        if cfg.get("refine"):
            raise SystemExit("semigroup_w > 0 with refine is untested -- refusing")
    _lwm_box = {"w": None} if lwm_mode != "none" else None
    _lwm_state = {"ema": None, "live": False}
    _gd_box = {"w": None, "gid": None} if gdro_eta > 0.0 else None
    _gd_state = {"ema": {}, "groups": None, "live": False, "cw": None}
    _fa_box = {"capture": False, "feat": None}
    _fa_sim = {"X": None}
    _fa_gen = torch.Generator().manual_seed(20260827) if fa_mode != "none" else None
    _fa_seen = {"live": False}
    _semi_seen = {"live": False}
    if lwm_mode != "none":
        logger.info(f"JAWSW ARMED: loss_weight_map={lwm_mode} lwm_strength={lwm_s} -- per-"
                    f"(lead,pixel,channel) weight (EMA+eps)^({'-' if lwm_mode == 'prec' else '+'}"
                    f"{lwm_s}) on the squared error, EMA beta=0.99 of the batch-mean squared "
                    f"residual, relative eps = 1% of the map mean, map normalised to mean 1.0; "
                    f"BOTH stages, EMA reset per stage; weight always from PREVIOUS steps' EMA")
    if gdro_eta > 0.0:
        logger.info(f"GDROA ARMED: group_dro_eta={gdro_eta} -- per-AoA-group loss EMAs "
                    f"(beta=0.98), per-sample weights exp(eta*EMA_g) normalised to batch-mean 1, "
                    f"real-finetune stage only; uniform until every group has an EMA")
    if semi_w > 0.0:
        logger.info(f"SEMIG ARMED: semigroup_w={semi_w} -- teacher-forced semigroup self-"
                    f"consistency, real-finetune only: compose [input {t_out // 2 + 1}..{t_out}, "
                    f"detached pred 1..{t_out // 2}] -> model -> penalise vs direct pred "
                    f"{t_out // 2 + 1}..{t_out} (MSE, u,v, normalized space); composed p-channel "
                    f"ZERO (the real-input convention)")

    _elem_box = _sps_box if _sps_box is not None else (
        _lwm_box if _lwm_box is not None else _gd_box)
    loss_fn = make_loss(loss_type, normalize=l2_norm, lead_w=_lead_w, elem_w=_elem_box,
                        chan_w=_chan_w, phys_den=_pd_box)
    _SPS = None
    if sps_lam > 0.0:
        import numpy as _np
        from sband_head import build_from_blob as _sbb, sband_features as _sbf
        _z = _np.load(CASE_DIR / "sband/calib_v38_plead.npz", allow_pickle=True)
        _sh = _sbb(torch.load(CASE_DIR / "sband/sband_ship.pth", map_location="cpu",
                              weights_only=False)).to(dev).eval()
        for _q in _sh.parameters():
            _q.requires_grad_(False)
        _SPS = dict(head=_sh, feat=_sbf,
                    ed=torch.as_tensor(_np.asarray(_z["sb_edge"], _np.float32)).to(dev),
                    lo=torch.as_tensor(_np.asarray(_z["sb_lo_phys"], _np.float32)).to(dev),
                    hi=torch.as_tensor(_np.asarray(_z["sb_hi_phys"], _np.float32)).to(dev))
        logger.info(f"SPSW:         sps_lam={sps_lam} sps_s={sps_s} -- band-edge reweighting, "
                    f"real stage only, mean weight asserted 1.0")

    # SOBG (2026-08-26): spatial-gradient (Sobolev) matching on the ERROR field.  0.0 (default,
    # or key absent) is BIT-IDENTICAL off: a single `if sobolev_w > 0.0` gates every new line in
    # step_loss, no module is created and no RNG is consumed (the WRa/SAMa/AXP/AOSIM additive
    # contract).  When armed: l += sobolev_w * ( mean|dx(pred-tgt)|^2 + mean|dy(pred-tgt)|^2 )
    # over the u,v channels only (loss_channels == 2 enforced, global and per-stage), dx/dy =
    # forward finite differences on the 32x64 grid, in the SAME normalized space as the main
    # loss, both stages.  History: the Sobolev/gradient-matching family was set aside on a
    # reading-level pricing estimate ("targets tke, which transfers at zero" -- the S2 sweeps)
    # without ever being run on this rig; this knob exists to replace that estimate with a
    # measured number under tke@0 pricing, the case carried by rel + mvpe alone.
    sobolev_w = float(cfg.get("sobolev_w", 0.0))
    if sobolev_w < 0.0:
        raise SystemExit(f"sobolev_w must be >= 0.0, got {sobolev_w}")
    _sobg_seen = {"live": False}
    if sobolev_w > 0.0:
        if lch != 2 or any(int(s.get("loss_channels", lch)) != 2 for s in cfg.get("stages", [])):
            raise SystemExit(f"sobolev_w > 0 requires loss_channels == 2 (u,v only) in every "
                             f"stage, got global {lch}")
        _sobg_bad = [str(s.get("objective", "block")) for s in cfg.get("stages", [])
                     if str(s.get("objective", "block")) != "block"]
        if _sobg_bad:
            raise SystemExit(f"sobolev_w > 0 lives in the block objective only; stages declare "
                             f"{_sobg_bad} -- refusing rather than silently not applying")
        logger.info(f"SOBG ARMED:   sobolev_w={sobolev_w} -- spatial-gradient (Sobolev) matching "
                    f"of the error field; u,v; normalized space; forward differences; both stages")

    # ============ WAVE 3a (2026-08-27): two additive default-off loss knobs ==============
    # PINOS (`ns_resid_w`) / HEROL (`hero_w`).  Each defaults to 0.0 == BIT-PATH-IDENTICAL off:
    # no module, no state, no RNG consumed (the WRa/SAMa/AXP/AOSIM/SOBG/Wave-2 additive
    # contract).  One mechanism per arm: any two of the wave-2 + wave-3a knobs armed together
    # are refused (WAVE3A).
    ns_w = float(cfg.get("ns_resid_w", 0.0))
    hero_w = float(cfg.get("hero_w", 0.0))
    if ns_w < 0.0 or hero_w < 0.0:
        raise SystemExit(f"ns_resid_w / hero_w must be >= 0.0, got ({ns_w}, {hero_w})")
    _w3_armed = [k for k, on in (("ns_resid_w", ns_w > 0.0), ("hero_w", hero_w > 0.0)) if on]
    if len(_w3_armed) > 1:
        raise SystemExit(f"wave-3a knobs are one-mechanism-per-arm: {_w3_armed} armed together")
    if _w3_armed:
        _k3 = _w3_armed[0]
        if _w2_armed:
            raise SystemExit(f"{_k3} cannot combine with wave-2 knob(s) {_w2_armed}")
        if sobolev_w > 0.0:
            raise SystemExit(f"{_k3} cannot combine with sobolev_w > 0 (one mechanism per arm)")
        if sps_lam > 0.0 or sam_rho > 0.0 or aux_w > 0.0 or \
                float(cfg.get("mixup_alpha", 0.0) or 0.0) > 0.0:
            raise SystemExit(f"{_k3} cannot combine with sps_lam/sam_rho/aux_phase_weight/mixup")
        if n_roll != 1 or any(int(s.get("n_roll", 1)) != 1 for s in cfg.get("stages", [])):
            raise SystemExit(f"{_k3} requires n_roll == 1 everywhere")
        if str(cfg.get("arch", "unet")) != "unet" or cfg.get("refine"):
            raise SystemExit(f"{_k3} needs the plain arch: unet (no refine)")
        _w3_bad = [f"{s.get('name')}:{s.get('objective', 'block')}/{s.get('pretext', 'none')}"
                   for s in cfg.get("stages", [])
                   if str(s.get("objective", "block")) != "block"
                   or str(s.get("pretext", "none")) != "none"]
        if _w3_bad:
            raise SystemExit(f"{_k3} needs objective: block / pretext: none in every stage, "
                             f"got {_w3_bad}")
        _w3_names = [str(s.get("name")) for s in cfg.get("stages", [])]
        if _w3_names != ["sim-pretrain", "real-finetune"]:
            # PINOS arms the FORMER only (the task guard: real has no p and must be untouched);
            # HEROL arms the LATTER only.  A differently-named stage could not be proven
            # real-free, so the whole stage list is pinned when either knob is armed.
            raise SystemExit(f"{_k3} needs stages == ['sim-pretrain', 'real-finetune'], "
                             f"got {_w3_names}")
        if ns_w > 0.0 and t_out < 2:
            raise SystemExit(f"ns_resid_w > 0 needs t_out >= 2 for the forward time difference")
    if hero_w > 0.0 and (lch != 2 or any(int(s.get("loss_channels", lch)) != 2
                                         for s in cfg.get("stages", []))):
        raise SystemExit("hero_w > 0 requires loss_channels == 2 in every stage")
    # PINOS physical constants -- derivations + smoke checks in the WAVE3A LOCK.  dx == dy on
    # the scored ::2 grid (verified against the h5 x/y arrays in wave3a_smoke.py); the sim
    # frame step is t = linspace(0, 20, 1000), NOT the real pool's 0.020.
    _NS_DX = 3.4217323e-3            # m, 2x native 1.71086615e-3
    _NS_DT_SIM = 0.020020020020020   # s
    _NS_L = 0.0654                   # m, chord from the real-pool closure Re = U*L/nu_water
    # PIVGE-G constants.  dx / dy are the SCORED (::2) spacings MEASURED from the real pool's own
    # h5 x,y arrays (the PIVGE step-1 instrument asserted both to a relative 1e-5 on
    # every file); dy is NEGATIVE because the h5 y axis DESCENDS with row index.  dt is the REAL
    # pool's 0.02 s -- NOT `_NS_DT_SIM`, which is the sim pool's frame spacing.
    _WV_DX = 3.4216640e-3            # m
    _WV_DY = -3.4216640e-3           # m, SIGNED and negative
    _WV_DT = 2.0e-2                  # s
    _WV_L = (10, 6, 6)               # (Lx, Ly, Lt) half-widths, scored cells / frames
    _WV_LCHORD = 0.0654              # m, PINOS's nu = U95 * L / Re convention
    _ns_box = {"nu": None, "denorm": None, "live": False} if ns_w > 0.0 else None
    _hero_state = {"ema": None, "live": False} if hero_w > 0.0 else None
    if ns_w > 0.0:
        logger.info(f"PINOS ARMED: ns_resid_w={ns_w} -- discrete 2D NS momentum residual of the "
                    f"PREDICTED (denormalised) fields, sim-pretrain stage ONLY: dt fwd-diff over "
                    f"the {t_out} leads (dt={_NS_DT_SIM:.6f} s), central spatial diffs + 5-pt "
                    f"Laplacian (dx=dy={_NS_DX:.4e} m), p from the PREDICTED channel 2 (trained "
                    f"by this term alone -- the recipe supervises u,v only), nu=U95*L/Re per file "
                    f"(L={_NS_L} m), interior crop 1px, live-mask from the target, term = "
                    f"mean(r_u^2+r_v^2) / mean(target dt-accel^2) (detached normaliser); "
                    f"real-finetune UNTOUCHED")
    if hero_w > 0.0:
        logger.info(f"HEROL ARMED: hero_w={hero_w} -- self-error-history supervision, "
                    f"real-finetune stage ONLY: EMA (beta=0.99) E of the batch-mean SIGNED "
                    f"per-(lead,pixel,chan) residual; term = w * mean(resid * sign(E) * "
                    f"min(|E|/(1e-12+rms(E)), 3)), sign/magnitude DETACHED, weight from PREVIOUS "
                    f"steps' EMA only; normalized space, u,v; sim-pretrain UNTOUCHED")

    # ===================== PIVGE-G (2026-08-31): `wvort_w`, one additive default-off knob =======
    # WVORT -- the WEAK-FORM pressure-free vorticity residual of the PREDICTED rollout.
    #
    #   w = d_x v - d_y u  (2-D curl, PHYSICAL y, which DESCENDS with row index)
    #   F = d_t w + div_2(u w) - nu lap_2 w              [pressure-free: curl grad p == 0]
    #   b = int F phi  =  T + Adv + Vis   against phi = psi(s_x)psi(s_y)psi(s_t), psi=(1-s^2)^5,
    #   every derivative moved onto phi, so the loss NEVER differentiates the network twice.
    #
    # WHY THIS IS NOT PINOS.  PINOS needed `p`, so it was hard-gated to the sim-pretrain stage.
    # The vorticity form needs none, so `wvort_w` arms the REAL-FINETUNE stage -- which no
    # physics prior in this repo has ever touched, and which is the DIMENSIONALLY SELF-CONSISTENT
    # pool (PIVGE-C.10.1: sim stores u,t non-dimensionally while storing x,y in metres, so a
    # sim-stage physics residual carries a ~13-16x unit-system factor).  SIM ARMING IS REFUSED
    # BY THE STAGE GATE BELOW AND THAT REFUSAL IS DELIBERATE.
    #
    # Kernels, gates and the width come from the PIVGE `wvort_core` module and the STEP-1
    # instrument that validated it -- ONE shared
    # implementation, so the instrument measures the code path the trainer runs.
    # ===================== DIVCONS (2026-09-09): `divstat_w`, one additive default-off knob =======
    # PIVGE-I closed the physics-prior lane for the MOMENTUM balance in any formulation and reopened it
    # only for a NEW OBSERVABLE.  DIVCONS-0 measured the
    # continuity SOURCE S = div2(u) on the v79 forecast: rms(div) pred/target 0.83 (interp) / 0.75
    # (rex_f2) at leads 11-20 and steady fraction 0.81 vs 0.70 / 0.72 vs 0.58 -- the learned forecaster
    # DELETES the fluctuating part of the out-of-plane source, more so off-support.  This term drives the
    # per-lead rms(div) and the window steady fraction of the PREDICTION toward the TARGET's own values
    # (both DETACHED), in physical units, on the target-live cells eroded by one pixel for the stencil.
    # It is exactly 0 at pred == tgt; u,v only; real-finetune only; no RNG; nothing at inference.
    ds_w = float(cfg.get("divstat_w", 0.0))
    if ds_w < 0.0:
        raise SystemExit(f"divstat_w must be >= 0.0, got {ds_w}")
    _ds_box = {"denorm": None, "live": False} if ds_w > 0.0 else None
    if ds_w > 0.0:
        _ds_stages = [str(st.get("name", "")) for st in cfg.get("stages", [])]
        if "real-finetune" not in _ds_stages:
            raise SystemExit("divstat_w needs a real-finetune stage (it arms there only)")
        logger.info(f"DIVSTAT ARMED: divstat_w={ds_w} -- continuity-source statistics consistency "
                    f"(per-lead rms(div) + window steady fraction of the PREDICTED rollout vs the TARGET's, "
                    f"both detached, physical units, target-live cells eroded 1px), real-finetune stage ONLY")
    wv_w = float(cfg.get("wvort_w", 0.0))
    if wv_w < 0.0:
        raise SystemExit(f"wvort_w must be >= 0.0, got {wv_w}")
    # PIVGE-G2: the residual TARGET.  `zero` (default) is PIVGE-G unchanged -- drive the weak-form
    # residual of the prediction to 0.  `data` matches the TARGET's own weak-form residual instead,
    # so the loss is EXACTLY zero at pred == tgt and its manifold is the data itself.  STEP-1's
    # stage-0 probe measured the champion at 0.198 against a ground-truth floor of 0.386, i.e. the
    # forecaster is ALREADY past the zero-target's direction; `data` is the arm that separates
    # "the target was wrong" from "the prior is empty".
    wv_tgt = str(cfg.get("wvort_target", "zero"))
    if wv_tgt not in ("zero", "data"):
        raise SystemExit(f"wvort_target must be 'zero' or 'data', got {wv_tgt!r}")
    if wv_tgt != "zero" and wv_w <= 0.0:
        raise SystemExit("wvort_target is set but wvort_w == 0 -- the mode would be a silent no-op")
    _wvc = None
    _wv_box = None
    if wv_w > 0.0:
        if _w3_armed:
            raise SystemExit(f"wvort_w cannot combine with wave-3a knob(s) {_w3_armed}")
        if _w2_armed:
            raise SystemExit(f"wvort_w cannot combine with wave-2 knob(s) {_w2_armed}")
        if sobolev_w > 0.0:
            raise SystemExit("wvort_w cannot combine with sobolev_w > 0 (one mechanism per arm)")
        if sps_lam > 0.0 or sam_rho > 0.0 or aux_w > 0.0 or \
                float(cfg.get("mixup_alpha", 0.0) or 0.0) > 0.0:
            raise SystemExit("wvort_w cannot combine with sps_lam/sam_rho/aux_phase_weight/mixup")
        if n_roll != 1 or any(int(s.get("n_roll", 1)) != 1 for s in cfg.get("stages", [])):
            raise SystemExit("wvort_w requires n_roll == 1 everywhere")
        if str(cfg.get("arch", "unet")) != "unet" or cfg.get("refine"):
            raise SystemExit("wvort_w needs the plain arch: unet (no refine)")
        _wv_bad = [f"{s.get('name')}:{s.get('objective', 'block')}/{s.get('pretext', 'none')}"
                   for s in cfg.get("stages", [])
                   if str(s.get("objective", "block")) != "block"
                   or str(s.get("pretext", "none")) != "none"]
        if _wv_bad:
            raise SystemExit(f"wvort_w needs objective: block / pretext: none in every stage, "
                             f"got {_wv_bad}")
        _wv_names = [str(s.get("name")) for s in cfg.get("stages", [])]
        if _wv_names != ["sim-pretrain", "real-finetune"]:
            raise SystemExit(f"wvort_w needs stages == ['sim-pretrain', 'real-finetune'], "
                             f"got {_wv_names} (the stage list is pinned so the arming stage is "
                             f"provably the REAL one -- sim is refused by construction)")
        if lch != 2 or any(int(s.get("loss_channels", lch)) != 2 for s in cfg.get("stages", [])):
            raise SystemExit("wvort_w requires loss_channels == 2 (it reads u, v only)")
        if t_out < 2 * _WV_L[2] + 1:
            raise SystemExit(f"wvort_w needs t_out >= {2 * _WV_L[2] + 1} for the temporal support")
        import wvort_core as _wvc                                          # noqa: F811
        _wv_box = {"nu": None, "denorm": None, "K": None, "live": False}
        logger.info(f"WVORT ARMED: wvort_w={wv_w} wvort_target={wv_tgt} -- WEAK-FORM "
                    f"pressure-free vorticity residual "
                    f"of the PREDICTED (denormalised) rollout, real-finetune stage ONLY "
                    f"(sim REFUSED: PIVGE-C.10.1 unit-system factor). phi = (1-s^2)^5 product, "
                    f"half-widths (Lx,Ly,Lt)={_WV_L} SCORED cells/frames (pre-registered width "
                    f"rule argmin|k-1| over the STEP-1 ladder, acc >= 0.10), dx={_WV_DX:.7e} m, "
                    f"dy={_WV_DY:.7e} m (DESCENDING), dt={_WV_DT:.6f} s, nu=U95*L/Re per file "
                    f"(L={_WV_LCHORD} m), accepted subdomain = zero masked cells in the whole "
                    f"support (live-mask from the TARGET at 1e-4 m/s, eroded 1px for the curl "
                    f"stencil), term = sum_acc b_pred^2 / sum_acc T_tgt^2 (denominator from the "
                    f"TARGET and DETACHED), fp32; u,v only -- the p channel receives exactly "
                    f"zero gradient; sim-pretrain UNTOUCHED")

    # ---- validation: UNCHANGED held-out competition val split, real stats ----
    # Built at K=1 ALWAYS so val_rel_l2 is over the same windows as v2, whatever n_roll is.
    v_mean, v_std = load_stats(cfg["stats"])
    if fold_val_files is not None:
        val_files = sorted(fold_val_files)          # this fold's held-out Reynolds, all of them
        missing = [f for f in val_files if not Path(f).exists()]
        if missing:
            raise SystemExit(f"fold {fold}: {len(missing)} val file(s) missing, e.g. {missing[0]}")
    elif "val_glob" in cfg:
        val_files = sorted(glob.glob(cfg["val_glob"]))[:int(cfg.get("n_val_files", 999))]
    else:
        raise SystemExit("no --fold and no val_glob -- nothing to validate on")
    # Official 2026-08-13 exclusion applies to VALIDATION ingestion too: manifests that predate
    # the ban (splits_aoa.json, splits_kfold.json) still list the file in their val_files, and
    # "must not be used" covers reading it at all, not only training on it.
    val_files = _drop_banned_real(val_files, logger)
    # PIN: validation MUST see the same channel-2 content as training, or the model is
    # trained with pressure and evaluated with zeros.  `p_cache_val` defaults to the real-stage
    # cache; absent, this is the historical call exactly.
    _pcv = cfg.get("p_cache_val")
    Xva_np, Yva_np = build_windows(val_files, sub_s, t_in, t_out, v_mean, v_std, t_out,
                                   logger, "val(held-out Re)", n_roll=1, p_cache=_pcv,
                                   p_target_only=bool(cfg.get("p_target_only", False)))
    Xva = torch.from_numpy(Xva_np).to(dev)
    Yva = torch.from_numpy(Yva_np).to(dev)
    std_t = torch.from_numpy(v_std).to(dev)
    mean_t = torch.from_numpy(v_mean).to(dev)

    # ---- optional rollout-val (diagnostic only, never used for model selection) ----
    Xvr = Yvr = None
    if n_roll > 1:
        Xvr_np, Yvr_np = build_windows(val_files, sub_s, t_in, t_out, v_mean, v_std,
                                       t_in + (n_roll - 1) * roll_stride + t_out, logger,
                                       "val-rollout", n_roll=n_roll, roll_stride=roll_stride,
                                       p_cache=_pcv)
        if Xvr_np.shape[0]:
            Xvr = torch.from_numpy(Xvr_np).to(dev)
            Yvr = torch.from_numpy(Yvr_np).to(dev)

    sys.path.insert(0, str(CASE_DIR))
    # `arch` selects the backbone.  Default `unet` is the shipped frame-stacked U-net and is
    # byte-identical to every previous v4 run; `seqformer` is the swappable spatio-temporal
    # backbone of the seqformer arm (time attention + Transolver slice tokens, or any registered
    # mixer pair).  Everything downstream -- stages, folds, losses, unrolling, the scorer -- is
    # shared, so an arch A/B is same-data, same-protocol and directly comparable to the whole
    # existing v4 table.
    arch = str(cfg.get("arch", "unet"))
    if "cno2d82" in cfg and arch != "cno2d82":
        raise SystemExit("cno2d82 configuration requires explicit arch: cno2d82")
    if "galerkin82" in cfg and arch != "galerkin82":
        raise SystemExit("galerkin82 configuration requires explicit arch: galerkin82")
    if "dcno82" in cfg and arch != "dcno82":
        raise SystemExit("dcno82 configuration requires explicit arch: dcno82")
    if "transolver2d82" in cfg and arch != "transolver2d82":
        raise SystemExit("transolver2d82 configuration requires explicit arch: transolver2d82")
    if "dpot82" in cfg and arch != "dpot82":
        raise SystemExit("dpot82 configuration requires explicit arch: dpot82")
    if "uno82" in cfg and arch != "uno82":
        raise SystemExit("uno82 configuration requires explicit arch: uno82")
    if arch == "unet":
        from unet_model import build_unet
        # `fuse` adds a bottleneck-attention and/or a full-resolution Fourier branch. Both are
        # exact no-ops at init and cost <=0.005 FINAL in time, so this is a near-free A/B.
        _arch = dict(cfg.get("fuse") or {}) if isinstance(cfg.get("fuse"), dict) \
            else ({"fuse": cfg["fuse"]} if cfg.get("fuse") else {})
        for _k in ("in_aug", "in_norm", "act", "out_mode", "anchor", "dt",
                   "substeps", "proj_iters", "mask_rel", "block", "levels", "drop_path",
                   "layer_scale", "nxt_k", "nxt_grn", "cond", "head", "refine",
                   "refine_base", "t_attn", "t_attn_dim", "t_attn_heads",
                   "freq_decomp", "freq_width", "freq_layers", "freq_smooth_n", "freq_w"):
            if cfg.get(_k):
                _arch[_k] = cfg[_k]
        if aux_w > 0.0:
            # AXP: gate MODULE CREATION on the weight, so aux_phase_weight: 0.0 (or absent)
            # builds a model state_dict-identical to the shipped one.
            _arch["aux_phases"] = len(AUX_PHASES)
        model = build_unet(t_in=t_in, t_out=t_out, c=3, base=base, **_arch).to(dev)
        # The object must BE what the config asked for.  out_mode/anchor/dt change the computation
        # while leaving the parameter set identical, so neither a strict state_dict load nor a
        # rebuild-from-meta round-trip can catch them being dropped here -- both would pass while
        # the trained model quietly stayed `direct`.  Assert on the built object instead.
        for _k, _default in (("out_mode", "direct"), ("anchor", "mean")):
            _want, _got = str(cfg.get(_k) or _default), str(getattr(model, _k, _default))
            assert _want == _got, f"config asks {_k}={_want} but the built model has {_got}"
        if cfg.get("dt") is not None:
            assert abs(float(getattr(model, "dt", 0.02)) - float(cfg["dt"])) < 1e-12, \
                f"config asks dt={cfg['dt']} but the built model has {getattr(model, 'dt', None)}"
        arch_desc = f"UNet2D base={base}" + (f" {_arch}" if _arch else "")
        if fa_mode != "none":
            # CORALA: capture-gated forward hook on the bottleneck.  Registered ONLY when armed,
            # so the off path does not even carry an inert hook.
            assert hasattr(model, "d2"), "feat_align: built model has no d2 bottleneck stage"

            def _fa_hook(_hm, _hi, _ho):
                if _fa_box["capture"]:
                    _fa_box["feat"] = _ho
            model.d2.register_forward_hook(_fa_hook)
            logger.info(f"CORALA ARMED: feat_align={fa_mode} feat_align_w={fa_w} -- d2 bottleneck "
                        f"forward hook registered; CORAL = ||cov_s-cov_r||_F^2/d^2 + "
                        f"||mu_s-mu_r||^2/d on spatially pooled features, real-finetune only, "
                        f"grad through BOTH passes, no label loss on sim; sim batches from a "
                        f"PRIVATE generator (seed 20260827) so the real-batch RNG stream matches "
                        f"the control exactly")
    elif arch == "seqformer":
        from seqformer import build_seqformer, DEFAULTS as _SFD
        scfg = dict(_SFD)
        scfg.update({k: v for k, v in (cfg.get("seqformer") or {}).items()})
        scfg.update(t_in=t_in, t_out=t_out, c=3, h=64 // sub_s, w=128 // sub_s)
        model = build_seqformer(scfg).to(dev)
        arch_desc = ("SeqFormer " + " ".join(f"{k}={scfg[k]}" for k in
                     ("spatial", "temporal", "d_model", "depth", "n_heads", "n_slices", "patch")))
        logger.info(f"SEQFORMER cfg: {scfg}")
    elif arch == "convltc":
        # LTCa family probe: liquid-time-constant conv-recurrent latent (Hasani AAAI-21
        # fused semi-implicit cell on conv states) — the backbone ledger's one unmeasured
        # family.  20 sequential encode steps + 20 fed-back generate steps; window
        # self-norm inside the model.
        from convltc import build_convltc
        model = build_convltc(t_in=t_in, t_out=t_out, c=3,
                              ch=int(cfg.get("ltc_ch", 128) or 128)).to(dev)
        arch_desc = f"ConvLTC ch={int(cfg.get('ltc_ch', 128) or 128)}"
    elif arch == "fno":
        # The OFFICIAL Track-1 FNO, built by the kit's own loader so the hyper-parameters are the
        # ones read off the released checkpoint shapes rather than ones we guessed, but with RANDOM
        # weights -- this trains the architecture from scratch under our recipe, it does not
        # fine-tune the organizers' checkpoint.  This arm
        # exists because their fno scores above our U-net on data it was trained on, and only a
        # same-protocol, out-of-fold arm can say whether the architecture is actually better.
        _KIT = CASE_DIR.parent / "realpde_t1_starting_kit_v6"
        sys.path.insert(0, str(_KIT))
        from load_baseline import build_model as _build_official, MODEL_KWARGS as _OFFK
        _h, _w = 64 // sub_s, 128 // sub_s
        model = _build_official("fno", device=str(dev),
                                in_shape=(t_in, _h, _w, 3), out_shape=(t_out, _h, _w, 3)).to(dev)
        arch_desc = "FNO(official) " + " ".join(f"{k}={v}" for k, v in _OFFK["fno"].items()
                                                if k != "model_name")
        logger.info(f"FNO from the T1 kit, RANDOM init: {_OFFK['fno']}")
    elif arch in ("transolver", "cno"):
        # The other two official backbones, same treatment as `fno`: the kit's own builder, the
        # kit's MODEL_KWARGS (read off the released checkpoint shapes, NOT the stale repo yamls --
        # configs/foil/trainsolver.yaml still says n_layers=1 while the checkpoints have 3), and
        # RANDOM weights.  This trains the ARCHITECTURE under our recipe; it does not fine-tune
        # their checkpoint.
        _KIT = CASE_DIR.parent / "realpde_t1_starting_kit_v6"
        sys.path.insert(0, str(_KIT))
        from load_baseline import build_model as _build_official, MODEL_KWARGS as _OFFK
        _h, _w = 64 // sub_s, 128 // sub_s
        model = _build_official(arch, device=str(dev),
                                in_shape=(t_in, _h, _w, 3), out_shape=(t_out, _h, _w, 3)).to(dev)
        arch_desc = f"{arch.upper()}(official) " + " ".join(
            f"{k}={v}" for k, v in _OFFK[arch].items() if k != "model_name")
        logger.info(f"{arch.upper()} from the T1 kit, RANDOM init: {_OFFK[arch]}")
    elif arch == "sol":
        # Solver-in-the-loop: 20 unrolled steps of (linear transport by the frozen window-mean
        # flow -> zero-init local corrector -> exact global implicit solve of a diagonally-
        # corrected momentum operator).  Substrate chosen BY MEASUREMENT (t2_solver_floor.py:
        # frozen transport 0.0977 beat persistence 0.1313; nonlinear self-advection 0.2226 and
        # hard 2-D projection 0.2593 rejected), and the model at init IS that 0.0977 floor.
        from sol_model import build_sol
        model = build_sol(t_in=t_in, t_out=t_out, c=3, base=base).to(dev)
        arch_desc = f"SOL(frozen-transport + corr{base} + implicit-solve)"
        logger.info(f"SOL (sol_model), init == frozen-transport floor")
    elif arch == "cfo":
        # CFO (arXiv:2512.05297, ICLR 2026): a flow-matching velocity operator integrated at
        # inference.  The velocity net IS build_unet, so an A/B against the champion measures the
        # OBJECTIVE, not a new architecture.  `CFONet.forward` runs the ODE internally and returns
        # the standard (B,t_out,H,W,3), so validation, selection and the scorer are untouched.
        # cfo_model.py documents every departure from the paper / official code.
        from cfo_model import build_cfo
        _cfo_yaml = dict(cfg.get("cfo") or {})
        if cfg.get("in_norm"):
            raise SystemExit(
                "arch: cfo refuses a top-level in_norm.  `in_norm: self` normalises by the "
                "INPUT's own statistics, which on an evolving single-frame ODE state is "
                "time-dependent and corrupts the velocity target.  Use cfo.state_norm "
                "(''|none = off, 'ctx' = frozen input-window statistics) instead.")
        # architecture knobs may be written at the top level (as every other arm does) or inside
        # the cfo: block; the cfo: block wins.
        for _k in ("block", "levels", "act", "nxt_k", "nxt_grn", "drop_path", "layer_scale"):
            if cfg.get(_k) is not None and _k not in _cfo_yaml:
                _cfo_yaml[_k] = cfg[_k]
        model = build_cfo(t_in=t_in, t_out=t_out, c=3, base=base, **_cfo_yaml).to(dev)
        # The object must BE what the config asked for.  solver / spline / gamma /
        # steps_per_frame / train_segments all change the computation while leaving the parameter
        # set identical, so neither a strict state_dict load nor a rebuild round-trip can catch
        # one being dropped.  Assert on the built object, exactly as the unet branch does.
        for _k, _v in _cfo_yaml.items():
            if _k in ("variant", "solver", "spline", "state_norm", "train_segments"):
                _got = str(getattr(model, _k))
                _want = "" if str(_v) == "none" else str(_v)
                assert _want == _got, f"config asks cfo.{_k}={_want} but the model has {_got}"
            elif _k in ("gamma", "steps_per_frame", "faithful_stencil", "time_dim"):
                assert getattr(model, _k) == type(getattr(model, _k))(_v), \
                    f"config asks cfo.{_k}={_v} but the model has {getattr(model, _k)}"
        _cfo_cfg = model.config()      # RESOLVED options -- this is what goes into ckpt meta
        arch_desc = (f"CFO-{model.variant} velocity=UNet2D base={base} "
                     f"{model.spline} gamma={model.gamma} state_norm={model.state_norm or 'none'} "
                     f"| inference {model.solver} x{model.steps_per_frame}/frame "
                     f"= {model.nfe_per_call} NFE")
        logger.info(f"CFO cfg (resolved): {_cfo_cfg}")
        logger.info(f"CFO inference cost: {model.nfe_per_call} velocity calls per predict() "
                    f"vs 1 forward for the block U-net")
    elif arch == "no82":
        # NO82 task adaptations share the data/loss recipe, with normalization inside the operator.
        _no82_incompatible = ("block", "fuse", "levels", "unet_base", "in_aug", "in_adv", "out_mode",
                             "anchor", "head", "t_attn", "refine", "residual", "tooth", "mimo",
                             "hetero_head", "coblend", "cond_film", "freq_decomp", "aux_phase_weight",
                             "act", "drop_path", "layer_scale", "nxt_k", "nxt_grn", "feat_align")
        _no82_bad = [k for k in _no82_incompatible if cfg.get(k) not in (None, "", "none", 0, False)]
        if _no82_bad:
            raise SystemExit(f"arch: no82 has incompatible backbone options {_no82_bad}")
        from no82_model import build_no82, DEFAULTS as _NO82D
        _no82_user = dict(cfg.get("no82") or {})
        _no82_runtime_keys = {"t_in", "t_out", "c", "h", "w", "in_norm", "cond"}
        if set(_no82_user) - (set(_NO82D) - _no82_runtime_keys):
            raise SystemExit("no82 block contains unknown keys or duplicates top-level geometry/normalization")
        _no82_cfg = dict(_NO82D, **_no82_user)
        _no82_cfg.update(t_in=t_in, t_out=t_out, c=3, h=64 // sub_s, w=128 // sub_s,
                         in_norm=str(cfg.get("in_norm") or ""), cond=str(cfg.get("cond") or ""))
        model = build_no82(_no82_cfg).to(dev)
        _no82_cfg = model.config()
        arch_desc = f"NO82 {_no82_cfg['operator']} width={_no82_cfg['width']} depth={_no82_cfg['depth']}"
        logger.info(f"NO82 ARMED: {arch_desc} resolved={_no82_cfg}")
    elif arch == "lsm82":
        # LSM82: queued compact latent-value spectral model, with normalization inside the adapter.
        _lsm_incompatible = ("block", "fuse", "levels", "unet_base", "in_aug", "in_adv", "out_mode",
                             "anchor", "head", "t_attn", "refine", "residual", "tooth", "mimo",
                             "hetero_head", "coblend", "cond_film", "freq_decomp", "aux_phase_weight",
                             "act", "drop_path", "layer_scale", "nxt_k", "nxt_grn", "feat_align",
                             "nxt_gate", "nxt_depth", "nxt_act", "nxt_pad", "nxt_se", "skip", "stage_res",
                             "route", "head_split", "flat_depth", "norm", "mimo_rep", "seqformer", "cfo",
                             "vit", "no82", "dt", "substeps", "proj_iters", "mask_rel", "t_attn_dim",
                             "t_attn_heads", "freq_width", "freq_layers", "freq_smooth_n", "freq_w",
                             "refine_base", "refine_zinit", "refine_ds", "aux_phases", "spec_inner",
                             "spec_modes", "heads")
        _lsm_bad = [k for k in _lsm_incompatible if cfg.get(k) not in (None, "", 0, False)]
        if _lsm_bad:
            raise SystemExit(f"arch: lsm82 has incompatible backbone options {_lsm_bad}")
        from lsm82_model import build_lsm82, DEFAULTS as _LSM82D
        _lsm_user = cfg.get("lsm82") or {}
        if not isinstance(_lsm_user, dict):
            raise SystemExit("lsm82 configuration must be a dictionary")
        _lsm_runtime_keys = {"t_in", "t_out", "c", "h", "w", "in_norm", "cond"}
        if set(_lsm_user) - (set(_LSM82D) - _lsm_runtime_keys):
            raise SystemExit("lsm82 block contains unknown/derived keys or duplicates top-level geometry/normalization")
        _lsm_requested = dict(_LSM82D, **_lsm_user)
        _lsm_requested.update(t_in=t_in, t_out=t_out, c=3, h=64 // sub_s, w=128 // sub_s,
                              in_norm=str(cfg.get("in_norm") or ""), cond=str(cfg.get("cond") or ""))
        model = build_lsm82(_lsm_requested).to(dev)
        _lsm_cfg = model.config()
        for _key, _value in _lsm_requested.items():
            assert _lsm_cfg[_key] == _value, f"lsm82 requested {_key}={_value!r} but built {_lsm_cfg[_key]!r}"
        assert len(model.process) == 3 and _lsm_cfg["num_parameters"] == sum(p.numel() for p in model.parameters())
        arch_desc = f"LSM82 width={_lsm_cfg['width']} heads={_lsm_cfg['heads']} tokens={_lsm_cfg['num_token']} bases={_lsm_cfg['num_basis']}"
        logger.info(f"LSM82 ARMED: {type(model).__name__} {arch_desc} resolved={_lsm_cfg}")
    elif arch == "sinenet82":
        # SINENET82: genuine multi-wave latent U shapes with the adapter's internal normalization.
        _sine_incompatible = ("block", "fuse", "levels", "unet_base", "in_aug", "in_adv", "out_mode",
                              "anchor", "head", "t_attn", "refine", "residual", "tooth", "mimo",
                              "hetero_head", "coblend", "cond_film", "freq_decomp", "aux_phase_weight",
                              "act", "drop_path", "layer_scale", "nxt_k", "nxt_grn", "feat_align",
                              "nxt_gate", "nxt_depth", "nxt_act", "nxt_pad", "nxt_se", "skip", "stage_res",
                              "route", "head_split", "flat_depth", "norm", "mimo_rep", "seqformer", "cfo",
                              "vit", "no82", "lsm82", "dt", "substeps", "proj_iters", "mask_rel",
                              "t_attn_dim", "t_attn_heads", "freq_width", "freq_layers", "freq_smooth_n",
                              "freq_w", "refine_base", "refine_zinit", "refine_ds", "aux_phases", "heads",
                              "width", "num_waves", "num_layers", "mult", "num_blocks", "block_residual",
                              "wave_residual", "disentangle", "activation", "padding_mode", "interpolation_mode")
        _sine_bad = [k for k in _sine_incompatible if cfg.get(k) not in (None, "", 0, False)]
        if _sine_bad:
            raise SystemExit(f"arch: sinenet82 has incompatible top-level backbone options {_sine_bad}")
        from sinenet82_model import build_sinenet82, DEFAULTS as _SINE82D
        _sine_user = cfg.get("sinenet82") or {}
        if not isinstance(_sine_user, dict):
            raise SystemExit("sinenet82 configuration must be a dictionary")
        _sine_runtime_keys = {"t_in", "t_out", "c", "h", "w", "in_norm", "cond"}
        if set(_sine_user) - (set(_SINE82D) - _sine_runtime_keys):
            raise SystemExit("sinenet82 block contains unknown/derived keys or duplicates top-level geometry/normalization")
        _sine_requested = dict(_SINE82D, **_sine_user)
        _sine_requested.update(t_in=t_in, t_out=t_out, c=3, h=64 // sub_s, w=128 // sub_s,
                               in_norm=str(cfg.get("in_norm") or ""), cond=str(cfg.get("cond") or ""))
        model = build_sinenet82(_sine_requested).to(dev)
        _sine_cfg = model.config()
        for _key, _value in _sine_requested.items():
            assert _sine_cfg[_key] == _value, f"sinenet82 requested {_key}={_value!r} but built {_sine_cfg[_key]!r}"
        assert len(model.waves) == _sine_cfg["num_waves"]
        assert all(len(w.down) == len(w.up) == _sine_cfg["num_layers"] for w in model.waves)
        assert _sine_cfg["num_parameters"] == sum(p.numel() for p in model.parameters())
        arch_desc = f"SINENET82 width={_sine_cfg['width']} waves={_sine_cfg['num_waves']} levels={_sine_cfg['num_layers']}"
        logger.info(f"SINENET82 ARMED: {type(model).__name__} {arch_desc} resolved={_sine_cfg}")
    elif arch == "uno82":
        # UNO82: seven spectral grid-change stages with internal window normalization.
        _uno_incompatible = ("block", "fuse", "levels", "unet_base", "in_aug", "in_adv", "out_mode",
                              "anchor", "head", "t_attn", "refine", "residual", "tooth", "mimo",
                              "hetero_head", "coblend", "cond_film", "freq_decomp", "aux_phase_weight",
                              "act", "drop_path", "layer_scale", "nxt_k", "nxt_grn", "feat_align",
                              "nxt_gate", "nxt_depth", "nxt_act", "nxt_pad", "nxt_se", "skip", "stage_res",
                              "route", "head_split", "flat_depth", "norm", "mimo_rep", "seqformer", "cfo",
                              "vit", "no82", "lsm82", "sinenet82", "dt", "substeps", "proj_iters", "mask_rel",
                              "t_attn_dim", "t_attn_heads", "freq_width", "freq_layers", "freq_smooth_n",
                              "freq_w", "refine_base", "refine_zinit", "refine_ds", "aux_phases", "heads",
                              "width", "num_waves", "num_layers", "mult", "num_blocks", "block_residual",
                              "wave_residual", "disentangle", "activation", "padding_mode", "interpolation_mode")
        _uno_bad = [k for k in _uno_incompatible if cfg.get(k) not in (None, "", 0, False)]
        if _uno_bad:
            raise SystemExit(f"arch: uno82 has incompatible top-level backbone options {_uno_bad}")
        from uno82_model import build_uno82, DEFAULTS as _UNO82D
        _uno_user = cfg.get("uno82")
        if _uno_user is None:
            _uno_user = {}
        if not isinstance(_uno_user, dict):
            raise SystemExit("uno82 configuration must be a dictionary")
        _uno_runtime_keys = {"t_in", "t_out", "c", "h", "w", "in_norm", "cond"}
        if set(_uno_user) - (set(_UNO82D) - _uno_runtime_keys):
            raise SystemExit("uno82 block contains unknown/derived keys or duplicates top-level geometry/normalization")
        _uno_requested = dict(_UNO82D, **_uno_user)
        _uno_requested.update(t_in=t_in, t_out=t_out, c=3, h=64 // sub_s, w=128 // sub_s,
                               in_norm=str(cfg.get("in_norm") or ""), cond=str(cfg.get("cond") or ""))
        model = build_uno82(_uno_requested).to(dev)
        _uno_cfg = model.config()
        for _key, _value in _uno_requested.items():
            assert _uno_cfg[_key] == _value, f"uno82 requested {_key}={_value!r} but built {_uno_cfg[_key]!r}"
        assert len(model.blocks) == 7 and len(_uno_cfg["stage_shapes"]) == 7
        assert _uno_cfg["num_parameters"] == sum(p.numel() for p in model.parameters())
        arch_desc = f"UNO82 width={_uno_cfg['width']} stages=7 modes={_uno_cfg['modes_h']}/{_uno_cfg['modes_w']}"
        logger.info(f"UNO82 ARMED: {type(model).__name__} {arch_desc} resolved={_uno_cfg}")
    elif arch == "cno2d82":
        # CNO2D82: classical filtered multiresolution 2D operator with window normalization.
        _cno_incompatible = ("block", "fuse", "levels", "unet_base", "in_aug", "in_adv", "out_mode",
                              "anchor", "head", "t_attn", "refine", "residual", "tooth", "mimo",
                              "hetero_head", "coblend", "cond_film", "freq_decomp", "aux_phase_weight",
                              "act", "drop_path", "layer_scale", "nxt_k", "nxt_grn", "feat_align",
                              "nxt_gate", "nxt_depth", "nxt_act", "nxt_pad", "nxt_se", "skip", "stage_res",
                              "route", "head_split", "flat_depth", "norm", "mimo_rep", "seqformer", "cfo",
                              "vit", "no82", "lsm82", "sinenet82", "uno82", "dt", "substeps", "proj_iters", "mask_rel",
                              "t_attn_dim", "t_attn_heads", "freq_width", "freq_layers", "freq_smooth_n",
                              "freq_w", "refine_base", "refine_zinit", "refine_ds", "aux_phases", "heads",
                              "width", "num_waves", "num_layers", "mult", "num_blocks", "block_residual",
                              "wave_residual", "disentangle", "activation", "padding_mode", "interpolation_mode")
        _cno_bad = [k for k in _cno_incompatible if cfg.get(k) not in (None, "", 0, False)]
        if _cno_bad:
            raise SystemExit(f"arch: cno2d82 has incompatible top-level backbone options {_cno_bad}")
        from cno2d82_model import build_cno2d82, DEFAULTS as _CNO2D82D
        _cno_user = cfg.get("cno2d82")
        if _cno_user is None:
            _cno_user = {}
        if not isinstance(_cno_user, dict):
            raise SystemExit("cno2d82 configuration must be a dictionary")
        _cno_runtime_keys = {"t_in", "t_out", "c", "h", "w", "in_norm", "cond"}
        if set(_cno_user) - (set(_CNO2D82D) - _cno_runtime_keys):
            raise SystemExit("cno2d82 block contains unknown/derived keys or duplicates top-level geometry/normalization")
        _cno_requested = dict(_CNO2D82D, **_cno_user)
        _cno_requested.update(t_in=t_in, t_out=t_out, c=3, h=64 // sub_s, w=128 // sub_s,
                               in_norm=str(cfg.get("in_norm") or ""), cond=str(cfg.get("cond") or ""))
        model = build_cno2d82(_cno_requested).to(dev)
        _cno_cfg = model.config()
        for _key, _value in _cno_requested.items():
            assert _cno_cfg[_key] == _value, f"cno2d82 requested {_key}={_value!r} but built {_cno_cfg[_key]!r}"
        assert len(model.encoder) == len(model.decoder) == len(model.decoder_inv) == 3
        assert len(model.res_nets) == 9 and len(_cno_cfg["stage_shapes"]) == 4
        assert sum(type(m).__name__ == "FilteredLeakyReLU" for m in model.modules()) == 24
        assert _cno_cfg["num_parameters"] == sum(p.numel() for p in model.parameters())
        arch_desc = f"CNO2D82 width={_cno_cfg['width']} levels=3 filtered=24"
        logger.info(f"CNO2D82 ARMED: {type(model).__name__} {arch_desc} resolved={_cno_cfg}")
    elif arch == "dpot82":
        # DPOT82: per-frame time aggregation and grouped nonlinear Fourier mixing.
        _dpot_incompatible = ("block", "fuse", "levels", "unet_base", "in_aug", "in_adv", "out_mode",
                              "anchor", "head", "t_attn", "refine", "residual", "tooth", "mimo",
                              "hetero_head", "coblend", "cond_film", "freq_decomp", "aux_phase_weight",
                              "act", "drop_path", "layer_scale", "nxt_k", "nxt_grn", "feat_align",
                              "nxt_gate", "nxt_depth", "nxt_act", "nxt_pad", "nxt_se", "skip", "stage_res",
                              "route", "head_split", "flat_depth", "norm", "mimo_rep", "seqformer", "cfo",
                              "vit", "no82", "lsm82", "sinenet82", "uno82", "cno2d82", "dt", "substeps", "proj_iters", "mask_rel",
                              "t_attn_dim", "t_attn_heads", "freq_width", "freq_layers", "freq_smooth_n",
                              "freq_w", "refine_base", "refine_zinit", "refine_ds", "aux_phases", "heads",
                              "width", "num_waves", "num_layers", "mult", "num_blocks", "block_residual",
                              "wave_residual", "disentangle", "activation", "padding_mode", "interpolation_mode", "depth", "groups", "patch_size", "head_width")
        _dpot_bad = [k for k in _dpot_incompatible if cfg.get(k) not in (None, "", 0, False)]
        if _dpot_bad:
            raise SystemExit(f"arch: dpot82 has incompatible top-level backbone options {_dpot_bad}")
        from dpot82_model import build_dpot82, DEFAULTS as _DPOT82D
        _dpot_user = cfg.get("dpot82")
        if _dpot_user is None:
            _dpot_user = {}
        if not isinstance(_dpot_user, dict):
            raise SystemExit("dpot82 configuration must be a dictionary")
        _dpot_runtime_keys = {"t_in", "t_out", "c", "h", "w", "in_norm", "cond"}
        if set(_dpot_user) - (set(_DPOT82D) - _dpot_runtime_keys):
            raise SystemExit("dpot82 block contains unknown/derived keys or duplicates top-level geometry/normalization")
        _dpot_requested = dict(_DPOT82D, **_dpot_user)
        _dpot_requested.update(t_in=t_in, t_out=t_out, c=3, h=64 // sub_s, w=128 // sub_s,
                               in_norm=str(cfg.get("in_norm") or ""), cond=str(cfg.get("cond") or ""))
        model = build_dpot82(_dpot_requested).to(dev)
        _dpot_cfg = model.config()
        for _key, _value in _dpot_requested.items():
            assert _dpot_cfg[_key] == _value, f"dpot82 requested {_key}={_value!r} but built {_dpot_cfg[_key]!r}"
        assert len(model.blocks) == _dpot_cfg["depth"] and model.temporal.timesteps == t_in
        assert _dpot_cfg["latent_shape"] == [model.h // model.patch_size, model.w // model.patch_size]
        assert _dpot_cfg["num_parameters"] == sum(p.numel() for p in model.parameters())
        arch_desc = f"DPOT82 width={_dpot_cfg['width']} depth={_dpot_cfg['depth']} groups={_dpot_cfg['groups']} patch={_dpot_cfg['patch_size']}"
        logger.info(f"DPOT82 ARMED: {type(model).__name__} {arch_desc} resolved={_dpot_cfg}")
    elif arch == "galerkin82":
        # GALERKIN82: native-grid signed Galerkin attention with internal window normalization.
        _gt_incompatible = ("block", "fuse", "levels", "unet_base", "in_aug", "in_adv", "out_mode",
                              "anchor", "head", "t_attn", "refine", "residual", "tooth", "mimo",
                              "hetero_head", "coblend", "cond_film", "freq_decomp", "aux_phase_weight",
                              "act", "drop_path", "layer_scale", "nxt_k", "nxt_grn", "feat_align",
                              "nxt_gate", "nxt_depth", "nxt_act", "nxt_pad", "nxt_se", "skip", "stage_res",
                              "route", "head_split", "flat_depth", "norm", "mimo_rep", "seqformer", "cfo",
                              "vit", "no82", "lsm82", "sinenet82", "uno82", "cno2d82", "dpot82", "dt", "substeps", "proj_iters", "mask_rel",
                              "t_attn_dim", "t_attn_heads", "freq_width", "freq_layers", "freq_smooth_n",
                              "freq_w", "refine_base", "refine_zinit", "refine_ds", "aux_phases", "heads",
                              "width", "num_waves", "num_layers", "mult", "num_blocks", "block_residual",
                              "wave_residual", "disentangle", "activation", "padding_mode", "interpolation_mode", "depth", "groups", "patch_size", "head_width")
        _gt_bad = [k for k in _gt_incompatible if cfg.get(k) not in (None, "", 0, False)]
        if _gt_bad:
            raise SystemExit(f"arch: galerkin82 has incompatible top-level backbone options {_gt_bad}")
        from galerkin82_model import build_galerkin82, DEFAULTS as _GALERKIN82D
        _gt_user = cfg.get("galerkin82")
        if _gt_user is None:
            _gt_user = {}
        if not isinstance(_gt_user, dict):
            raise SystemExit("galerkin82 configuration must be a dictionary")
        _gt_runtime_keys = {"t_in", "t_out", "c", "h", "w", "in_norm", "cond"}
        _gt_misplaced = sorted((set(_GALERKIN82D) - _gt_runtime_keys) & set(cfg))
        if _gt_misplaced:
            raise SystemExit(f"galerkin82 computational settings must be inside the galerkin82 block: {_gt_misplaced}")
        if set(_gt_user) - (set(_GALERKIN82D) - _gt_runtime_keys):
            raise SystemExit("galerkin82 block contains unknown/derived keys or duplicates top-level geometry/normalization")
        _gt_requested = dict(_GALERKIN82D, **_gt_user)
        _gt_requested.update(t_in=t_in, t_out=t_out, c=3, h=64 // sub_s, w=128 // sub_s,
                               in_norm=str(cfg.get("in_norm") or ""), cond=str(cfg.get("cond") or ""))
        model = build_galerkin82(_gt_requested).to(dev)
        _gt_cfg = model.config()
        for _key, _value in _gt_requested.items():
            assert _gt_cfg[_key] == _value, f"galerkin82 requested {_key}={_value!r} but built {_gt_cfg[_key]!r}"
        assert len(model.blocks) == _gt_cfg["depth"] and _gt_cfg["n_tokens"] == model.h * model.w
        assert _gt_cfg["head_dim"] == model.width // model.heads
        assert _gt_cfg["num_parameters"] == sum(p.numel() for p in model.parameters())
        arch_desc = f"GALERKIN82 width={_gt_cfg['width']} depth={_gt_cfg['depth']} heads={_gt_cfg['heads']} tokens={_gt_cfg['n_tokens']}"
        logger.info(f"GALERKIN82 ARMED: {type(model).__name__} {arch_desc} resolved={_gt_cfg}")
    elif arch == "dcno82":
        # DCNO82: native-grid Fourier and sequential dilated residual convolution.
        _dc_incompatible = ("block", "fuse", "levels", "unet_base", "in_aug", "in_adv", "out_mode",
                              "anchor", "head", "t_attn", "refine", "residual", "tooth", "mimo",
                              "hetero_head", "coblend", "cond_film", "freq_decomp", "aux_phase_weight",
                              "act", "drop_path", "layer_scale", "nxt_k", "nxt_grn", "feat_align",
                              "nxt_gate", "nxt_depth", "nxt_act", "nxt_pad", "nxt_se", "skip", "stage_res",
                              "route", "head_split", "flat_depth", "norm", "mimo_rep", "seqformer", "cfo",
                              "vit", "no82", "lsm82", "sinenet82", "uno82", "cno2d82", "dpot82", "galerkin82", "dt", "substeps", "proj_iters", "mask_rel",
                              "t_attn_dim", "t_attn_heads", "freq_width", "freq_layers", "freq_smooth_n",
                              "freq_w", "refine_base", "refine_zinit", "refine_ds", "aux_phases", "heads",
                              "width", "num_waves", "num_layers", "mult", "num_blocks", "block_residual",
                              "wave_residual", "disentangle", "activation", "padding_mode", "interpolation_mode", "depth", "groups", "patch_size", "head_width")
        _dc_bad = [k for k in _dc_incompatible if cfg.get(k) not in (None, "", 0, False)]
        if _dc_bad:
            raise SystemExit(f"arch: dcno82 has incompatible top-level backbone options {_dc_bad}")
        from dcno82_model import build_dcno82, DEFAULTS as _DCNO82D
        _dc_user = cfg.get("dcno82")
        if _dc_user is None:
            _dc_user = {}
        if not isinstance(_dc_user, dict):
            raise SystemExit("dcno82 configuration must be a dictionary")
        _dc_runtime_keys = {"t_in", "t_out", "c", "h", "w", "in_norm", "cond"}
        _dc_misplaced = sorted((set(_DCNO82D) - _dc_runtime_keys) & set(cfg))
        if _dc_misplaced:
            raise SystemExit(f"dcno82 computational settings must be inside the dcno82 block: {_dc_misplaced}")
        if set(_dc_user) - (set(_DCNO82D) - _dc_runtime_keys):
            raise SystemExit("dcno82 block contains unknown/derived keys or duplicates top-level geometry/normalization")
        _dc_requested = dict(_DCNO82D, **_dc_user)
        _dc_requested.update(t_in=t_in, t_out=t_out, c=3, h=64 // sub_s, w=128 // sub_s,
                               in_norm=str(cfg.get("in_norm") or ""), cond=str(cfg.get("cond") or ""))
        model = build_dcno82(_dc_requested).to(dev)
        _dc_cfg = model.config()
        for _key, _value in _dc_requested.items():
            assert (tuple(_dc_cfg[_key]) == tuple(_value) if _key == "dilation" else _dc_cfg[_key] == _value), f"dcno82 requested {_key}={_value!r} but built {_dc_cfg[_key]!r}"
        assert len(model.global_layers) == 5 and len(model.local_layers) == 3
        assert _dc_cfg["num_parameters"] == sum(p.numel() for p in model.parameters())
        arch_desc = f"DCNO82 width={_dc_cfg['width']} modes={_dc_cfg['modes_h']}x{_dc_cfg['modes_w']} cycles={_dc_cfg['cycles']}"
        logger.info(f"DCNO82 ARMED: {type(model).__name__} {arch_desc} resolved={_dc_cfg}")
    elif arch == "transolver2d82":
        # TRANSOLVER2D82: direct temporal feature stack and 2D physics-slice attention.
        _ts_incompatible = ("block", "fuse", "levels", "unet_base", "in_aug", "in_adv", "out_mode",
                              "anchor", "head", "t_attn", "refine", "residual", "tooth", "mimo",
                              "hetero_head", "coblend", "cond_film", "freq_decomp", "aux_phase_weight",
                              "act", "drop_path", "layer_scale", "nxt_k", "nxt_grn", "feat_align",
                              "nxt_gate", "nxt_depth", "nxt_act", "nxt_pad", "nxt_se", "skip", "stage_res",
                              "route", "head_split", "flat_depth", "norm", "mimo_rep", "seqformer", "cfo",
                              "vit", "no82", "lsm82", "sinenet82", "uno82", "cno2d82", "dpot82", "galerkin82", "dcno82", "dt", "substeps", "proj_iters", "mask_rel",
                              "t_attn_dim", "t_attn_heads", "freq_width", "freq_layers", "freq_smooth_n",
                              "freq_w", "refine_base", "refine_zinit", "refine_ds", "aux_phases", "heads",
                              "width", "num_waves", "num_layers", "mult", "num_blocks", "block_residual",
                              "wave_residual", "disentangle", "activation", "padding_mode", "interpolation_mode", "depth", "groups", "patch_size", "head_width")
        _ts_bad = [k for k in _ts_incompatible if cfg.get(k) not in (None, "", 0, False)]
        if _ts_bad:
            raise SystemExit(f"arch: transolver2d82 has incompatible top-level backbone options {_ts_bad}")
        from transolver2d82_model import build_transolver2d82, DEFAULTS as _TRANSOLVER2D82D
        _ts_user = cfg.get("transolver2d82")
        if _ts_user is None:
            _ts_user = {}
        if not isinstance(_ts_user, dict):
            raise SystemExit("transolver2d82 configuration must be a dictionary")
        _ts_runtime_keys = {"t_in", "t_out", "c", "h", "w", "in_norm", "cond"}
        _ts_misplaced = sorted((set(_TRANSOLVER2D82D) - _ts_runtime_keys) & set(cfg))
        if _ts_misplaced:
            raise SystemExit(f"transolver2d82 computational settings must be inside the transolver2d82 block: {_ts_misplaced}")
        if set(_ts_user) - (set(_TRANSOLVER2D82D) - _ts_runtime_keys):
            raise SystemExit("transolver2d82 block contains unknown/derived keys or duplicates top-level geometry/normalization")
        _ts_requested = dict(_TRANSOLVER2D82D, **_ts_user)
        _ts_requested.update(t_in=t_in, t_out=t_out, c=3, h=64 // sub_s, w=128 // sub_s,
                               in_norm=str(cfg.get("in_norm") or ""), cond=str(cfg.get("cond") or ""))
        model = build_transolver2d82(_ts_requested).to(dev)
        _ts_cfg = model.config()
        for _key, _value in _ts_requested.items():
            assert _ts_cfg[_key] == _value, f"transolver2d82 requested {_key}={_value!r} but built {_ts_cfg[_key]!r}"
        assert len(model.blocks) == _ts_cfg["depth"] and _ts_cfg["spatial_tokens"] == model.h * model.w
        assert _ts_cfg["num_parameters"] == sum(p.numel() for p in model.parameters())
        arch_desc = f"TRANSOLVER2D82 width={_ts_cfg['width']} depth={_ts_cfg['depth']} slices={_ts_cfg['slices']} tokens={_ts_cfg['spatial_tokens']}"
        logger.info(f"TRANSOLVER2D82 ARMED: {type(model).__name__} {arch_desc} resolved={_ts_cfg}")
    elif arch == "vit":
        # VIT knob (2026-09-10): a PURE Vision Transformer forecaster (vit_model.py).
        # Frame-stacked window ->
        # patch embedding -> learned 2-D position -> pre-norm global MHSA encoder -> linear un-patchify.
        # `in_norm` and `cond` are applied INSIDE the model with UNet2D's arithmetic (the generic
        # InNorm wrapper below is skipped for vit, E2).  Every U-net-only knob REFUSES rather than
        # silently no-opping, and the built object is asserted against the config (the cfo pattern).
        _vit_refuse = ("block", "fuse", "levels", "in_aug", "in_adv", "out_mode", "anchor", "head",
                       "t_attn", "refine", "residual", "tooth", "mimo", "hetero_head", "coblend",
                       "cond_film", "seqformer", "cfo", "freq_decomp", "aux_phase_weight", "unet_base",
                       "act", "drop_path", "layer_scale", "nxt_k", "nxt_grn", "skip", "nxt_se", "route",
                       "head_split", "flat_depth", "norm", "stage_res", "dt", "substeps", "proj_iters",
                       "mask_rel", "feat_align", "mimo_rep")
        _vit_bad = [k for k in _vit_refuse if k in cfg and cfg.get(k) not in (None, "", "none", 0, 0.0, False)]
        if _vit_bad:
            raise SystemExit(f"arch: vit refuses U-net-only / incompatible keys {_vit_bad} -- they would be "
                             f"silent no-ops on a ViT; remove them from the config")
        if str(cfg.get("in_norm") or "") not in ("", "self") or str(cfg.get("cond") or "") not in ("", "stats"):
            raise SystemExit(f"arch: vit implements in_norm in {{'', 'self'}} and cond in {{'', 'stats'}} only, "
                             f"got in_norm={cfg.get('in_norm')!r} cond={cfg.get('cond')!r}")
        from vit_model import build_vit, DEFAULTS as _VITD
        _vit_user = dict(cfg.get("vit") or {})
        _vit_unk = set(_vit_user) - {"patch", "d_model", "depth", "n_heads", "mlp_ratio", "qkv_bias"}
        if _vit_unk:
            raise SystemExit(f"vit: unknown/forbidden keys in the vit: block {sorted(_vit_unk)}")
        _vit_cfg = dict(_VITD)
        _vit_cfg.update(_vit_user)
        _vit_cfg.update(t_in=t_in, t_out=t_out, c=3, h=64 // sub_s, w=128 // sub_s,
                        in_norm=str(cfg.get("in_norm") or ""), cond=str(cfg.get("cond") or ""))
        model = build_vit(_vit_cfg).to(dev)
        _vit_got = model.config()
        for _k, _v in _vit_cfg.items():
            assert _vit_got[_k] == type(_vit_got[_k])(_v), f"config asks vit.{_k}={_v} but the built model has {_vit_got[_k]}"
        _vit_cfg = dict(_vit_got)
        _vit_np = int(sum(q.numel() for q in model.parameters()))
        arch_desc = (f"ViTForecaster patch={_vit_cfg['patch']} d={_vit_cfg['d_model']} depth={_vit_cfg['depth']} "
                     f"heads={_vit_cfg['n_heads']} mlp={_vit_cfg['mlp_ratio']} tokens={model.n_tok} ci={model.ci}")
        logger.info(f"VIT ARMED: {arch_desc} params={_vit_np} in_norm={_vit_cfg['in_norm'] or 'none'} "
                    f"cond={_vit_cfg['cond'] or 'none'} -- pure ViT, in_norm/cond inside the model (UNet2D "
                    f"arithmetic), zero-init un-patchify (step-0 prediction == window mean), SDPA attention")
    else:
        raise SystemExit(f"unknown arch {arch!r} (expected 'unet', 'seqformer', "
                         f"'fno', 'transolver', 'cno', 'sol' or 'cfo')")

    # `in_norm` for the backbones that do not implement it themselves.  The U-net handles it inside
    # its own forward (byte-identical to every previous arm); everything else gets the wrapper,
    # which unet_model._selfcheck proves is the same computation.
    if arch not in ("unet", "vit", "no82", "lsm82", "sinenet82", "uno82", "cno2d82", "dpot82", "galerkin82", "dcno82", "transolver2d82") and cfg.get("in_norm"):  # these models normalize internally
        from unet_model import InNorm
        model = InNorm(model, str(cfg["in_norm"])).to(dev)
        arch_desc += f" +InNorm({cfg['in_norm']})"
        logger.info(f"IN_NORM WRAPPER: {cfg['in_norm']} around {arch}")
    if cfg.get("in_adv"):
        from unet_model import AdvectCanon
        model = AdvectCanon(model, str(cfg["in_adv"])).to(dev)
        arch_desc += f" +AdvectCanon({cfg['in_adv']})"
        logger.info(f"IN_ADV WRAPPER: {cfg['in_adv']} transport canonicalisation")

    # ---- optional persistence-anchored residual parameterisation ----------------------
    class _Residual(torch.nn.Module):
        """out = last input frame + net(x). No parameters; the anchor never carries gradient because
        `x` is the input. Broadcasting over the output time axis makes it one add."""

        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x) + x[:, -1:]

    residual = bool(cfg.get("residual", False))
    if residual:
        model = _Residual(model).to(dev)
        arch_desc += " +residual(persistence-anchored)"
        logger.info("RESIDUAL: out = x[:, -1] + net(x); state_dict keys are prefixed 'net.'")

    # ============ MWAVE (2026-08-27): MIMOA (fuse mimo) / CRPSH (fuse hetero_head) ============
    # Both are ARCH knobs and pass ONLY via the fuse dict (the D1/gridfix copy-list lesson: a
    # flat key would be silently dropped).  train_v4 reads the INTENT from cfg's fuse dict and
    # REFUSES unless the BUILT OBJECT carries the same values -- a wrapper or a non-unet arch
    # would otherwise hide the knob and silently no-op the mechanism (the model would take
    # _forward_mimo's eval path on single windows and never see a second draw).  Absent or 0
    # == BIT-PATH-IDENTICAL off: no state, no RNG, no new line on the live path.
    _fuse_d = cfg.get("fuse") if isinstance(cfg.get("fuse"), dict) else {}
    _mimo = int(_fuse_d.get("mimo", 0) or 0)
    _hh = int(_fuse_d.get("hetero_head", 0) or 0)
    # MIMOSCALE (2026-08-27): Havasi INPUT REPETITION.  A TRAIN knob, not an ARCH knob -- it
    # changes how training windows are paired, not the module -- so it is a FLAT top-level
    # cfg key read directly here and NEVER enters the fuse dict: the built object and its
    # `meta` are identical to rho_rep=0, and a checkpoint rebuilds correctly through
    # build_from_meta without it.  (The D1/gridfix "a flat key is silently dropped" trap is
    # the _arch copy list, which mimo_rep never travels through.)  Absent or 0.0 is
    # BIT-PATH-IDENTICAL off: no extra RNG draw -- proven by the MIMOSCALE `null` check.
    # Its EXECUTION WITNESS is the MIMOA LIVE line's realized same_idx_frac.
    _mimo_rep = float(cfg.get("mimo_rep", 0.0) or 0.0)
    if not (0.0 <= _mimo_rep <= 1.0):
        raise SystemExit(f"mimo_rep must lie in [0, 1], got {_mimo_rep}")
    if _mimo_rep > 0.0 and not _mimo:
        raise SystemExit(f"mimo_rep={_mimo_rep} without fuse {{mimo: M>=2}} -- refusing a "
                         f"silent no-op (input repetition needs MIMO slots to repeat into)")
    nll_w = float(cfg.get("nll_w", 1.0))
    _mimo_seen = {"live": False}
    _hh_seen = {"live": False}
    _mimo_gen = None
    if _mimo or _hh:
        if not (_mimo == 0 or 2 <= _mimo <= 8) or _hh not in (0, 1):
            raise SystemExit(f"mimo must be 0 or 2..8 and hetero_head 0|1, got "
                             f"({_mimo}, {_hh})")
        if _mimo and _hh:
            raise SystemExit("mimo and hetero_head are one-mechanism-per-arm: both armed")
        _got_mh = (int(getattr(model, "mimo", 0) or 0), int(getattr(model, "hetero_head", 0) or 0))
        if _got_mh != (_mimo, _hh):
            raise SystemExit(f"MWAVE: config fuse asks (mimo={_mimo}, hetero_head={_hh}) but "
                             f"the built object carries {_got_mh} -- a wrapper or arch mismatch "
                             f"is hiding the knob; refusing a silent no-op")
        _mk = "mimo" if _mimo else "hetero_head"
        if arch != "unet" or residual or cfg.get("in_adv") or cfg.get("refine"):
            raise SystemExit(f"{_mk} needs the plain arch: unet (no residual/in_adv/refine)")
        if n_roll != 1 or any(int(s.get("n_roll", 1)) != 1 for s in cfg.get("stages", [])):
            raise SystemExit(f"{_mk} requires n_roll == 1 everywhere")
        if _w2_armed or _w3_armed or sobolev_w > 0.0:
            raise SystemExit(f"{_mk} cannot combine with wave-2/3a knobs or sobolev_w")
        if sps_lam > 0.0 or sam_rho > 0.0 or aux_w > 0.0:
            raise SystemExit(f"{_mk} cannot combine with sps_lam/sam_rho/aux_phase_weight")
        for _q in ("mixup_alpha", "rot_aug_deg", "input_noise", "regime_weight",
                   "fisher_protect", "muon_lr", "refine_ds"):
            if float(cfg.get(_q, 0.0) or 0.0) > 0.0:
                raise SystemExit(f"{_mk} cannot combine with {_q}")
        if float(cfg.get("sim_aoa_weight", 1.0)) != 1.0 or bool(cfg.get("physguard", False)) \
                or cfg.get("distill_arm"):
            raise SystemExit(f"{_mk} cannot combine with sim_aoa_weight/physguard/distill_arm")
        if tke_w > 0.0 or tspec_w > 0.0 or lowk_w > 0.0 or dt1_w > 0.0 or dt2_w > 0.0 or \
                bsp_w > 0.0 or mvpe_w > 0.0 or pgrad_w > 0.0 or hpl1_w > 0.0:
            raise SystemExit(f"{_mk} composes with the bare {loss_type}+distill recipe only")
        if lch != 2 or any(int(s.get("loss_channels", lch)) != 2 for s in cfg.get("stages", [])):
            raise SystemExit(f"{_mk} requires loss_channels == 2 in every stage")
        _mw_names = [str(s.get("name")) for s in cfg.get("stages", [])]
        if _mw_names != ["sim-pretrain", "real-finetune"]:
            raise SystemExit(f"{_mk} needs stages == ['sim-pretrain', 'real-finetune'], "
                             f"got {_mw_names}")
        _mw_bad = [f"{s.get('name')}:{s.get('objective', 'block')}/{s.get('pretext', 'none')}"
                   for s in cfg.get("stages", [])
                   if str(s.get("objective", "block")) != "block"
                   or str(s.get("pretext", "none")) != "none"]
        if _mw_bad:
            raise SystemExit(f"{_mk} needs objective: block / pretext: none in every stage, "
                             f"got {_mw_bad}")
        if any(int(s.get("lp_steps", 0)) for s in cfg.get("stages", [])):
            raise SystemExit(f"{_mk} cannot combine with lp_steps (head-group semantics differ)")
    if _mimo:
        # Second-window draws ride a PRIVATE generator (the CORALA precedent, seed 20260827)
        # so the PRIMARY batch stream -- and hence the control-matched minibatch sequence --
        # is untouched by the arm's extra draw.
        _mimo_gen = torch.Generator().manual_seed(20260827)
        _mw_inc = next(_q for _q in model.inc.modules() if isinstance(_q, torch.nn.Conv2d))
        logger.info(f"MIMOA ARMED: mimo={_mimo} -- MIMO implicit {_mimo}-head ensemble "
                    f"(Havasi arXiv:2010.06610): stem consumes {_mimo} stacked windows (inc "
                    f"in_channels={_mw_inc.in_channels}), outc emits {_mimo} stacked "
                    f"predictions (out_channels={model.outc.out_channels}); training pairs "
                    f"each batch element with {_mimo - 1} further uniformly-drawn window(s) "
                    f"(PRIVATE generator seed 20260827, input repetition rho_rep="
                    f"{_mimo_rep}); loss = mean of the {_mimo} heads' losses (slot axis "
                    f"folded into the batch, distill teacher fed the folded windows); "
                    f"eval = the same window in every slot, heads averaged")
    if _hh:
        if nll_w <= 0.0:
            raise SystemExit(f"hetero_head=1 needs nll_w > 0, got {nll_w}")
        _lv_np = sum(int(p.numel()) for p in model.outc_lv.parameters())
        logger.info(f"CRPSH ARMED: hetero_head=1 nll_w={nll_w} -- heteroscedastic (mean, "
                    f"logvar) head: the MAIN supervised term is REPLACED by the Gaussian NLL "
                    f"0.5*(logvar + resid^2/exp(logvar)) (logvar clamped [-10, 4]), u,v, "
                    f"normalized space, BOTH stages; the distill leg keeps loss_fn on the "
                    f"MEAN (chan_weight therefore reaches the distill leg only); second head "
                    f"unit outc_lv={_lv_np} params (an ADDITION -- the mean path is capacity-"
                    f"unchanged); the submission-relevant output is the MEAN")

    # ============ COBLEND (2026-09-02): joint training of the blend members ============
    # ARCH knob `coblend: M` (+ `coblend_blocks`) rides the fuse dict; TRAIN knobs
    # `coblend_w_blend` / `coblend_lam_ind` are FLAT keys read here (the mimo_rep precedent).
    # Absent == BIT-PATH-IDENTICAL off: `_cob = 0`, every armed branch below is `if _cob:`.
    # Refuses (MWAVE-style) unless the BUILT object is a CoBlend whose members carry the asked
    # block classes -- a module-class walk, because a state_dict witness is provably blind to a
    # block whose parameters share names/shapes (the 2026-09-01 LayerNorm2d/GroupNorm lesson).
    _cob = int(_fuse_d.get("coblend", 0) or 0)
    _cob_wb = float(cfg["coblend_w_blend"]) if cfg.get("coblend_w_blend") is not None else 1.0
    _cob_li = float(cfg.get("coblend_lam_ind", 0.0) or 0.0)
    _cob_seen = {"live": False}
    if not _cob and (cfg.get("coblend_w_blend") is not None or cfg.get("coblend_lam_ind")):
        raise SystemExit("coblend_w_blend/coblend_lam_ind set without fuse {coblend: M} -- "
                         "refusing a silent no-op")
    if _cob:
        from unet_model import CoBlend as _CoBlend
        _COB_CLS = {"convnext": "NeXtConv", "selfwarp": "SelfWarpConv", "double": "DoubleConv",
                    "ssm": "SSMConv", "swin": "SwinConv"}
        if not isinstance(model, _CoBlend) or type(model).__name__ != "CoBlend":
            raise SystemExit(f"COBLEND: fuse asks coblend={_cob} but the built object is "
                             f"{type(model).__name__} -- refusing a silent no-op")
        _cob_blocks = [str(q) for q in (_fuse_d.get("coblend_blocks") or [])]
        if len(model.members) != _cob or tuple(_cob_blocks) != tuple(model.coblend_blocks):
            raise SystemExit(f"COBLEND: asked coblend={_cob} blocks={_cob_blocks}, built "
                             f"{len(model.members)} members blocks={model.coblend_blocks}")
        _cob_np = []
        for _ck, _cm in enumerate(model.members):
            _cen = {type(_q).__name__ for _q in _cm.modules()}
            _hit = {v for v in _COB_CLS.values() if v in _cen}
            if _hit != {_COB_CLS[_cob_blocks[_ck]]}:
                raise SystemExit(f"COBLEND: member {_ck} asks block={_cob_blocks[_ck]} but its "
                                 f"module-class walk finds block classes {sorted(_hit)}")
            _cob_np.append(sum(int(_q.numel()) for _q in _cm.parameters()))
        if not (_cob_wb >= 0.0 and _cob_li >= 0.0) or (_cob_wb == 0.0 and _cob_li == 0.0):
            raise SystemExit(f"COBLEND: need w_blend, lam_ind >= 0 and not both 0, got "
                             f"({_cob_wb}, {_cob_li})")
        if arch != "unet" or residual or cfg.get("in_adv") or cfg.get("refine"):
            raise SystemExit("coblend needs the plain arch: unet (no residual/in_adv/refine)")
        if _mimo or _hh:
            raise SystemExit("coblend does not compose with mimo / hetero_head")
        if n_roll != 1 or any(int(s.get("n_roll", 1)) != 1 for s in cfg.get("stages", [])):
            raise SystemExit("coblend requires n_roll == 1 everywhere")
        if _w2_armed or _w3_armed or sobolev_w > 0.0 or sps_lam > 0.0 or sam_rho > 0.0 \
                or aux_w > 0.0 or fa_mode != "none":
            raise SystemExit("coblend cannot combine with wave-2/3a / sobolev / sps_lam / "
                             "sam_rho / aux_phase_weight / feat_align")
        for _q in ("mixup_alpha", "rot_aug_deg", "input_noise", "regime_weight",
                   "fisher_protect", "muon_lr", "refine_ds", "mimo_rep"):
            if float(cfg.get(_q, 0.0) or 0.0) > 0.0:
                raise SystemExit(f"coblend cannot combine with {_q}")
        if float(cfg.get("sim_aoa_weight", 1.0)) != 1.0 or bool(cfg.get("physguard", False)) \
                or cfg.get("distill_arm"):
            raise SystemExit("coblend cannot combine with sim_aoa_weight/physguard/distill_arm")
        if tke_w > 0.0 or tspec_w > 0.0 or lowk_w > 0.0 or dt1_w > 0.0 or dt2_w > 0.0 or \
                bsp_w > 0.0 or mvpe_w > 0.0 or pgrad_w > 0.0 or hpl1_w > 0.0:
            raise SystemExit(f"coblend composes with the bare {loss_type}+distill recipe only")
        if lch != 2 or any(int(s.get("loss_channels", lch)) != 2 for s in cfg.get("stages", [])):
            raise SystemExit("coblend requires loss_channels == 2 in every stage")
        if any(int(s.get("lp_steps", 0)) for s in cfg.get("stages", [])):
            raise SystemExit("coblend cannot combine with lp_steps (the outc head group is "
                             "empty inside the wrapper)")
        if any(str(s.get("objective", "block")) != "block" or str(s.get("pretext", "none")) != "none"
               for s in cfg.get("stages", [])):
            raise SystemExit("coblend needs objective: block / pretext: none in every stage")
        logger.info(f"COBLEND ARMED: coblend={_cob} blocks={list(model.coblend_blocks)} "
                    f"member_params={_cob_np} total={sum(_cob_np)} w_blend={_cob_wb} "
                    f"lam_ind={_cob_li} -- forward = _BlendNet uniform mean (same ops, same "
                    f"order); loss = w_blend*[loss_fn(blend)+distill(blend)] + "
                    f"lam_ind*SUM_k[loss_fn(member_k)+distill(member_k)] (one teacher forward "
                    f"per step, shared); EMA/selection on the wrapper; grad_clip PER MEMBER "
                    f"(each member sees the standalone recipe's clip); module-class walk "
                    f"witnessed {[_COB_CLS[b] for b in _cob_blocks]}")
    # ============ CONDFILM (2026-09-02): feature-wise conditioning on the cond:stats scalars ============
    # ARCH knob `cond_film: 1` rides the fuse dict ONLY (spread verbatim into build_unet, recorded
    # in meta["fuse"], spread back by build_from_meta).  Absent == BIT-PATH-IDENTICAL off: `_cf = 0`,
    # every armed branch is `if _cf:`.  Refuses (MWAVE-style) unless the BUILT object carries the
    # knob AND a module-class walk (unet_model.film_census) finds exactly five CondFiLM -- the
    # state_dict witness is provably blind to a zero-initialised FiLM.  The last layers must read
    # exactly 0 here (identity at step 0); the LIVE witness below is the first eval whose stashed
    # gamma has left zero.
    _cf = int(_fuse_d.get("cond_film", 0) or 0)
    _cf_seen = {"live": False, "eval0": False}
    if _cf:
        from unet_model import film_census as _film_census
        if _cf != 1:
            raise SystemExit(f"cond_film must be 0 or 1, got {_cf}")
        if arch != "unet" or residual or cfg.get("in_adv") or cfg.get("refine"):
            raise SystemExit("cond_film needs the plain arch: unet (no residual/in_adv/refine)")
        if _cob or _mimo or _hh:
            raise SystemExit("cond_film does not compose with coblend / mimo / hetero_head")
        if int(getattr(model, "cond_film", 0) or 0) != 1:
            raise SystemExit("CONDFILM: config fuse asks cond_film=1 but the built object does not "
                             "carry it -- a wrapper or arch mismatch is hiding the knob; refusing")
        if str(getattr(model, "cond", "")) != "stats":
            raise SystemExit(f"CONDFILM needs cond: stats on the built object, got {getattr(model, 'cond', None)!r}")
        _cfc = _film_census(model)
        if _cfc["CondFiLM"] != 5 or sorted(_cfc["sites"]) != sorted(["inc", "d1", "d2", "c2", "c1"]):
            raise SystemExit(f"CONDFILM: module-class walk found {_cfc}, expected five CondFiLM at inc/d1/d2/c2/c1")
        _cf_np = int(sum(p.numel() for p in model.film.parameters()))
        _cf_wmax = max(float(q.out.weight.abs().max()) for q in model.film.values())
        _cf_bmax = max(float(q.out.bias.abs().max()) for q in model.film.values())
        if _cf_wmax != 0.0 or _cf_bmax != 0.0:
            raise SystemExit(f"CONDFILM: last layer is not zero-initialised (max|w|={_cf_wmax}, max|b|={_cf_bmax})")
        logger.info(f"CONDFILM ARMED: cond_film=1 sites={list(model.film.keys())} "
                    f"widths={[int(q.ch) for q in model.film.values()]} film_params={_cf_np} "
                    f"max|w_last|={_cf_wmax} max|b_last|={_cf_bmax} census={_cfc} -- "
                    f"t*(1+gamma(c))+beta(c) after inc/d1/d2/c2/c1, c=(cm_u,cm_v,cs_u,cs_v) via "
                    f"cond_tap; exact identity at step 0; hidden init from a private generator "
                    f"(global RNG untouched, minibatch stream == control)")
    elif arch == "unet":
        from unet_model import film_census as _film_census
        logger.info(f"CONDFILM census: CondFiLM={_film_census(model)['CondFiLM']} (knob absent; module-class walk, read-only)")
    # ============ ATTNUNET (2026-09-03): attention-class knobs, witnessed by module-class walk ============
    # `skip: gate` (AttnGate, the GRIDFIX AGa object -- untouched), `skip: scse` (SCSEGate) and
    # `nxt_se: r` (SEGate on every NeXtBlock) ride the fuse dict ONLY; `fuse: attn` is the string
    # form (BottleneckAttn).  Absent == BIT-PATH-IDENTICAL off (`_au_armed` False, every branch
    # below is gated on it).  Refuses (CONDFILM-style) unless the BUILT object's census matches the
    # config's intent EXACTLY -- the state_dict witness is blind to a zero-initialised gate, and
    # AttnGate/SCSEGate/SEGate all start at a constant.  LIVE witness at the evals below: the
    # stashed gate maps have left their step-0 constant.
    _au_se = int(_fuse_d.get("nxt_se", 0) or 0)
    _au_skip = str(_fuse_d.get("skip", "") or "")
    _au_attn = str(cfg.get("fuse") or "") in ("attn", "both")
    _au_armed = bool(_au_se or _au_skip in ("gate", "scse") or _au_attn)
    _au_seen = {"live": False, "eval0": False}
    _au_ag_hooks = []
    if arch == "unet" and not _cob:
        from unet_model import attn_census as _attn_census
        _auc = _attn_census(model)
        _n_nxt = sum(1 for _m in model.modules() if type(_m).__name__ == "NeXtBlock")
        _want = {"SEGate": _n_nxt if _au_se else 0, "SCSEGate": 2 if _au_skip == "scse" else 0,
                 "AttnGate": 2 if _au_skip == "gate" else 0, "BottleneckAttn": 1 if _au_attn else 0}
        _got = {k: _auc[k] for k in _want}
        if _got != _want:
            raise SystemExit(f"ATTNUNET: module-class walk found {_got}, config intent {_want} "
                             f"(fuse={cfg.get('fuse')!r}) -- the built object is not what was asked; refusing")
        if _au_se and int(getattr(model, "nxt_se", 0) or 0) != _au_se:
            raise SystemExit("ATTNUNET: config fuse asks nxt_se but the built object does not carry it")
        if _au_armed:
            if _au_se:
                _ses = [_m for _m in model.modules() if type(_m).__name__ == "SEGate"]
                if any(float(_m.fc2.weight.abs().max()) != 0.0 or float(_m.fc2.bias.abs().max()) != 0.0 for _m in _ses):
                    raise SystemExit("ATTNUNET-SE: fc2 is not zero-initialised (gate != 1 at step 0)")
            if _au_skip == "scse":
                _scs = [_m for _m in model.modules() if type(_m).__name__ == "SCSEGate"]
                if any(float(_m.fc2.weight.abs().max()) != 0.0 or float(_m.sp.weight.abs().max()) != 0.0 for _m in _scs):
                    raise SystemExit("ATTNUNET-SCSE: gates are not zero-initialised (not identity at step 0)")
            if _au_skip == "gate":
                # read-only hook on AttnGate.psi: stash sigmoid(psi) as the gate map (numerics untouched)
                def _mk_hook(_ag):
                    def _h(_mod, _inp, _out):
                        _ag._last_gate = torch.sigmoid(_out.detach())
                    return _h
                for _m in model.modules():
                    if type(_m).__name__ == "AttnGate":
                        _au_ag_hooks.append(_m.psi.register_forward_hook(_mk_hook(_m)))
            _au_np = sum(p.numel() for _n, p in model.named_parameters()
                         if any(s in _n for s in (".se.", "sc1.", "sc2.", "ag1.", "ag2.", "attn.")))
            logger.info(f"ATTNUNET ARMED: nxt_se={_au_se} skip={_au_skip!r} fuse_attn={_au_attn} census={_got} "
                        f"sites={_auc['sites'][:6]}{'...' if len(_auc['sites']) > 6 else ''} attn_params={_au_np} "
                        f"-- SE/SCSE exact identity at step 0 (private-generator init, minibatch stream == control); "
                        f"AttnGate = the GRIDFIX object (0.5 at step 0, global-RNG init, same-seed not same-minibatch)")
        else:
            logger.info(f"ATTNUNET census: {_got} (knobs absent; module-class walk, read-only)")

    def _au_gate_stats():
        """max deviation of every stashed gate from its step-0 constant + a spread summary."""
        out = {}
        for _n, _m in model.named_modules():
            _cn = type(_m).__name__
            _g = getattr(_m, "_last_gate", None)
            if _cn == "SEGate" and _g is not None:
                out[_n] = {"dev": float((_g - 1.0).abs().max()), "std_ch": float(_g.std(dim=1).mean()), "mean": float(_g.mean())}
            elif _cn == "SCSEGate" and _g is not None:
                _gc, _gs = _g
                out[_n] = {"dev": float(max((_gc - 0.5).abs().max(), (_gs - 0.5).abs().max())),
                           "std_c": float(_gc.std(dim=1).mean()), "std_s": float(_gs.std(dim=(2, 3)).mean()),
                           "mean_c": float(_gc.mean()), "mean_s": float(_gs.mean())}
            elif _cn == "AttnGate" and _g is not None:
                out[_n] = {"dev": float((_g - 0.5).abs().max()), "std_s": float(_g.std(dim=(2, 3)).mean()), "mean": float(_g.mean())}
            elif _cn == "BottleneckAttn":
                out[_n] = {"dev": float(_m.proj.weight.abs().max())}
        return out
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"{arch_desc} params={n_params/1e6:.3f}M -> ckpt ~{n_params*4/1e6:.1f} MB dev={dev}")
    logger.info(f"stages={[s['name'] for s in cfg['stages']]} bs={bs} stride={stride} "
                f"ema={ema_decay} loss_channels={lch} exclude_{group_by}={sorted(excl)}")
    if bsp_w > 0.0:
        logger.info(f"BSP LOSS:     weight={bsp_w} bins={bsp_nb} (binned log spatial power, arXiv:2502.00472)")
    logger.info(f"SELECTION:    select_on={sel_on}"
                f"{'  (val rel-L2, the historical rule)' if sel_on == 'rel' else '  (mean of the three error subscores, i.e. what is scored)'}")
    logger.info(f"V3 OBJECTIVE: loss_type={loss_type} l2_normalize={l2_norm} tke_weight={tke_w} tspec_weight={tspec_w} lowk_weight={lowk_w} lowk_cut={lowk_cut} dt1_weight={dt1_w} dt2_weight={dt2_w}")
    if _lead_w is not None:
        logger.info(f"LWa LEAD WEIGHT: lead_weight={lead_weight!r} mean={sum(_lead_w)/len(_lead_w):.12f} "
                    f"w=[{', '.join(f'{x:.4f}' for x in _lead_w)}]")
    logger.info(f"V3 UNROLL:    n_roll={n_roll} mode={unroll_mode} warmup={unroll_warmup} "
                f"stride={roll_stride}/{t_out} "
                f"(self-generated share of the fed-back input = {roll_stride/t_in:.0%})")
    logger.info(f"V4 OBJECTIVE: mvpe_weight={mvpe_w} input_noise={in_noise} mode={noise_mode} noise_stage={noise_stage}"
                f"{' (OFF -- bit-identical to every previous arm)' if in_noise == 0 else ''}")
    if aux_w > 0.0:
        logger.info(f"AXP AUX:      aux_phase_weight={aux_w} phases={AUX_PHASES} -- auxiliary "
                    f"head supervises the 3 unseen stride-2 subsamplings of the same target "
                    f"window (training only; unused at inference)")
    if fold is None:
        logger.info(f"V4 SPLIT:     legacy 2-Re val_glob ({len(val_files)} files) -- "
                    f"NOTE this split is known to mis-rank models, prefer --fold")
    else:
        logger.info(f"V4 SPLIT:     fold {fold}/{man['k']} group_by={group_by} val={fold_val_re} "
                    f"({len(val_files)} files, {Xva.shape[0]} windows) | manifest {cfg['kfold_manifest']}")
    if mvpe_w > 0.0:
        mvpe_selfcheck(kit_dir, t_out, int(Xva.shape[2]), int(Xva.shape[3]), sub_s, logger)

    # COMPLEX-AWARE.  `.float()` on a complex tensor DISCARDS the imaginary part, and
    # `torch.complex64.is_floating_point` is False, so the original real-only version silently
    # zeroed the phase of every spectral weight.  For the kit's FNO that is 50.33 M of 50.36 M
    # parameters and 70.7% of the weight magnitude -- the EMA shadow was not a worse model, it was
    # a broken one (raw val 0.070 against a flat EMA val 0.257).  Real and integer tensors take
    # exactly the branches they took before, so every pre-2026-08-01 arm is bit-identical.
    # -- self/cross distillation teacher (default OFF).  Loaded once; the {fold}
    # placeholder lets one config serve every fold with the matching OOF-safe teacher.
    _distill_model = None
    _distill_stage = {"on": False}
    _distill_models = []
    if distill_w > 0.0 and distill_ckpt:
        from unet_model import build_from_meta as _bfm
        # comma-separated => ENSEMBLE teacher; the soft target is the MEAN of their predictions, which
        # is the same object `eval_kfold --ensemble` scores.  A single path reproduces the old behaviour
        # exactly (one model, mean of one).
        for _spec in [q for q in distill_ckpt.split(",") if q.strip()]:
            _tp = _spec.strip().format(fold=fold)
            _td = torch.load(_tp, map_location="cpu", weights_only=False)
            # The teacher is built at ITS OWN t_in, not the student's.  LHIST (2026-08-24) is the
            # first arm whose t_in differs from the teacher's: building a t_in=40 teacher and loading
            # t_in=20 weights raised `size mismatch for inc.net.0.weight: [128,64] vs [128,124]`.
            # When they differ the teacher is fed the student's MOST RECENT frames (`cur[:, -tt:]`),
            # which is exactly the window the teacher was trained on.  Identical t_in => unchanged.
            _tt = int(_td.get("meta", {}).get("t_in", t_in) or t_in)
            assert _td.get("meta", {}).get("t_out", t_out) in (None, t_out), \
                f"teacher t_out {_td['meta'].get('t_out')} != student {t_out} -- not a soft-target match"
            assert 0 < _tt <= t_in, f"teacher t_in {_tt} exceeds the student's {t_in}: no frames to feed it"
            _tm = _bfm(_td.get("meta", {}), _tt, t_out, 3).to(dev).eval()
            _tm.load_state_dict(_td["model_state_dict"], strict=True)
            for _q in _tm.parameters(): _q.requires_grad_(False)
            _tm._t_in_own = _tt
            _distill_models.append(_tm)
            logger.info(f"DISTILL:      teacher={_tp} weight={distill_w} t_in={_tt}"
                        f"{'' if _tt == t_in else f' (student t_in={t_in}; fed the last {_tt} frames)'}"
                        f" (real-finetune stage only)")
        _distill_model = _distill_models[0]
        logger.info(f"DISTILL:      {len(_distill_models)} teacher(s), soft target = their MEAN")
    if _dws is not None:   # DWSCHED knob (2026-09-10): the armed line, once, after the teachers are loaded
        if not _distill_models:
            raise SystemExit("DWSCHED: distill_sched armed but no teacher was loaded")
        logger.info(f"DWSCHED ARMED: start={_dws['start']!r} end={_dws['end']!r} shape={_dws['shape']} stage={_dws['stage']} "
                    f"mean={0.5 * (_dws['start'] + _dws['end'])!r} == distill_weight={_dws['const']!r} (w(s) = start + "
                    f"(end-start)*s/(S-1) overrides the constant at every {_dws['stage']} step; the constant is the arming gate "
                    f"and the time-mean, restored at stage end; {len(_distill_models)} teacher(s))")
    # ---- EQFREE TOOTH (2026-09-09): gap-tooth random spatial crops as a TRAINING-DISTRIBUTION ----
    # construction (Fabiani/Kevrekidis 2505.02308 'patch' NOs).  Training-only; the parameter set is
    # untouched; absent => this file is byte-for-byte the historical path.
    _tooth_cfg = cfg.get("tooth")
    _tooth = None
    _tooth_seen = {"live": False, "n": 0}
    if _tooth_cfg:
        assert isinstance(_tooth_cfg, dict), _tooth_cfg
        _tk = set(_tooth_cfg) - {"h", "w", "halo", "p", "stage", "input"}
        assert not _tk, f"tooth: unknown keys {sorted(_tk)}"
        _th, _tw, _thl = int(_tooth_cfg["h"]), int(_tooth_cfg["w"]), int(_tooth_cfg.get("halo", 0))
        _tp = float(_tooth_cfg.get("p", 1.0))
        _tstage = str(_tooth_cfg.get("stage", "real-finetune"))
        _hin, _win = _th + 2 * _thl, _tw + 2 * _thl
        _Hf, _Wf = int(Xva.shape[2]), int(Xva.shape[3])
        _div = 8 if int(cfg.get("levels", 2) or 2) == 3 else 4
        assert 0 < _hin <= _Hf and 0 < _win <= _Wf and _hin % _div == 0 and _win % _div == 0, \
            (_hin, _win, _Hf, _Wf, _div)
        assert 0.0 < _tp <= 1.0 and _tstage in ("real-finetune", "sim-pretrain", "all"), (_tp, _tstage)
        assert arch == "unet" and not _mimo and not _cob and n_roll == 1 and sam_rho == 0.0 \
            and _SPS is None and not cfg.get("refine"), "tooth composes with the bare unet recipe only"
        assert tke_w == 0.0 and mvpe_w == 0.0 and not cfg.get("loss_weight_map") \
            and not cfg.get("physguard") and not cfg.get("feat_align") and not cfg.get("bsp_weight"), \
            "tooth: field-level / probe losses are undefined on a crop"
        assert not cfg.get("strideaug") and not cfg.get("aux_phase_weight") \
            and not cfg.get("distill_arm") and not cfg.get("mixup_alpha"), "tooth: refused combination"
        assert hasattr(model, "_tooth"), type(model)
        for _tm in _distill_models:
            assert hasattr(_tm, "_tooth"), f"tooth: teacher {type(_tm)} has no crop path"
        _tinput = str(_tooth_cfg.get("input", "crop"))
        assert _tinput in ("crop", "full"), _tinput      # full = ELW control: full-field input, loss on the tooth only
        _tooth = {"h": _th, "w": _tw, "halo": _thl, "hin": _hin, "win": _win, "p": _tp, "stage": _tstage, "input": _tinput}
        _tooth_gen = torch.Generator().manual_seed(int(cfg.get("seed", 0) or 0) + 20260909)
        logger.info(f"TOOTH:        inner {_th}x{_tw} halo {_thl} -> input {_hin}x{_win} of {_Hf}x{_Wf}; input={_tinput}; "
                    f"p={_tp} stage={_tstage}; boxes on a PRIVATE generator; "
                    f"{len(_distill_models)} teacher(s) cropped identically")
    _tooth_box = {"boxes": None}     # ELW control: the per-sample inner boxes when the model is NOT cropped
    def _tooth_inner(full_pred):
        """ELW control: cut the INNER tooth out of a full-field prediction (B,t,H,W,c) -> (B,t,h,w,c)."""
        bx = _tooth_box["boxes"]; hl = _tooth["halo"]
        return torch.stack([full_pred[i, :, int(bx[i, 0]) + hl:int(bx[i, 0]) + hl + _tooth["h"],
                                      int(bx[i, 1]) + hl:int(bx[i, 1]) + hl + _tooth["w"]] for i in range(int(full_pred.shape[0]))], 0)
    def _tooth_embed(crop, full):
        """EQFREE TOOTH: write the inner-tooth prediction (B,t,h,w,c) into a copy of the FULL target.
        Every loss leg then keeps its full-window denominator (rel_l2 = ||e_tooth|| / ||y_full|| per
        sample); the gradient's support is the tooth, the objective is unchanged."""
        ts = model._tooth if getattr(model, "_tooth", None) is not None else {"boxes": _tooth_box["boxes"], "halo": _tooth["halo"]}
        bx = ts["boxes"]; hl = int(ts["halo"]); hh, ww = int(crop.shape[2]), int(crop.shape[3])
        out = full.detach().clone()
        for i in range(int(crop.shape[0])):
            y0, x0 = int(bx[i, 0]) + hl, int(bx[i, 1]) + hl
            out[i, :, y0:y0 + hh, x0:x0 + ww] = crop[i]
        return out
    ema = {k: (v.detach().clone() if v.is_complex() else v.detach().clone().float())
           for k, v in model.state_dict().items()}

    def ema_update(d):
        msd = model.state_dict()
        for k, v in ema.items():
            src = msd[k].detach()
            if v.is_complex() or v.dtype.is_floating_point:
                v.mul_(d).add_(src.to(v.dtype), alpha=1 - d)
            else:
                v.copy_(src)

    def _val_pass():
        """{mse,rel,tke,mvpe} on the K=1 held-out windows, PHYSICAL space, u/v only.

        v4 adds `mvpe` here so the third accuracy subscore is visible during training, not
        only in `eval_v3.py` afterwards -- it is the term the new loss targets."""
        acc = {"mse": 0.0, "rel": 0.0, "tke": 0.0, "mvpe": 0.0}
        with torch.no_grad():
            for i in range(0, Xva.shape[0], 32):
                xb, yb = Xva[i:i + 32], Yva[i:i + 32]
                pb = model(xb)
                acc["mse"] += float(torch.sum((pb - yb) ** 2).cpu()) / (yb[0].numel())
                pph = (pb * std_t + mean_t)[..., :2]
                tth = (yb * std_t + mean_t)[..., :2]
                pp = pph.reshape(xb.shape[0], -1)
                tt = tth.reshape(xb.shape[0], -1)
                acc["rel"] += float((torch.norm(pp - tt, dim=1) / torch.norm(tt, dim=1).clamp(min=1e-8)).sum().cpu())
                acc["tke"] += float(tke_rel_l2(pph, tth).sum().cpu())
                acc["mvpe"] += float(mvpe_rel_l2(pph, tth, sub_s).sum().cpu())
        n = Xva.shape[0]
        return {k: v / n for k, v in acc.items()}

    def _roll_val_pass():
        """Diagnostic: mean rel-L2 of the LAST block after n_roll self-fed blocks."""
        if Xvr is None:
            return float("nan")
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, Xvr.shape[0], 16):
                xb, yb = Xvr[i:i + 16], Yvr[i:i + 16]
                cur = xb
                for k in range(n_roll):
                    pb = model(cur)
                    cur = feedback(pb, xb, lch, cur, roll_stride)
                tgt = yb[:, (n_roll - 1) * roll_stride:(n_roll - 1) * roll_stride + t_out]
                pp = ((pb * std_t + mean_t)[..., :2]).reshape(xb.shape[0], -1)
                tt = ((tgt * std_t + mean_t)[..., :2]).reshape(xb.shape[0], -1)
                tot += float((torch.norm(pp - tt, dim=1) / torch.norm(tt, dim=1).clamp(min=1e-8)).sum().cpu())
                cnt += xb.shape[0]
        return tot / max(1, cnt)

    def val_eval():
        """({raw}, {ema}, raw_rollout) -- each dict is _val_pass()'s mse/rel/tke/mvpe."""
        model.eval()
        raw = _val_pass()
        rroll = _roll_val_pass()
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v.to(backup[k].dtype) for k, v in ema.items()})
        eva = _val_pass()
        model.load_state_dict(backup)
        model.train()
        return raw, eva, rroll

    best = {"key": float("inf"), "rel": float("inf"), "which": None, "stage": None, "step": -1,
            "tke": None, "mvpe": None}
    evals = []

    def _ckpt_meta(which, stage, gstep, m):
        _d = {"arch": ({"unet": "unet2d_framestacked"}.get(arch, arch)),
                "residual": residual,
                "fuse": (cfg.get("fuse") or "none"),
                "in_aug": (cfg.get("in_aug") or ""),
                "in_norm": (cfg.get("in_norm") or ""),
                "in_adv": (cfg.get("in_adv") or ""),
                "block": (cfg.get("block") or "double"),
                "levels": int(cfg.get("levels", 2) or 2),
                "drop_path": float(cfg.get("drop_path", 0.0) or 0.0),
                "layer_scale": float(cfg.get("layer_scale", 0.0) or 0.0),
                "nxt_k": int(cfg.get("nxt_k", 7) or 7),
                "nxt_grn": float(cfg.get("nxt_grn", 0.0) or 0.0),
                "cond": (cfg.get("cond") or ""),
                "ltc_ch": int(cfg.get("ltc_ch", 128) or 128),
                "head": (cfg.get("head") or ""),
                "freq_decomp": int(cfg.get("freq_decomp", 0) or 0),
                "freq_width": int(cfg.get("freq_width", 64) or 64),
                "freq_layers": int(cfg.get("freq_layers", 2) or 2),
                "freq_smooth_n": int(cfg.get("freq_smooth_n", 2) or 2),
                "freq_w": float(cfg.get("freq_w", 0.0) or 0.0),
                "t_attn": int(cfg.get("t_attn", 0) or 0),
                "t_attn_dim": int(cfg.get("t_attn_dim", 32) or 32),
                "t_attn_heads": int(cfg.get("t_attn_heads", 4) or 4),
                "refine": int(cfg.get("refine", 0) or 0),
                "refine_base": int(cfg.get("refine_base", 32) or 32),
                "refine_ds": float(cfg.get("refine_ds", 0.0) or 0.0),
                "mixup_alpha": float(cfg.get("mixup_alpha", 0.0) or 0.0),
                "seqformer": (dict(scfg) if arch == "seqformer" else None),
                "act": (cfg.get("act") or "gelu"),
        "out_mode": (cfg.get("out_mode") or "direct"),
        "anchor": (cfg.get("anchor") or "mean"),
        "dt": float(cfg.get("dt", 0.02)),
        "substeps": int(cfg.get("substeps", 2)),
        "proj_iters": int(cfg.get("proj_iters", 0)),
        "mask_rel": float(cfg.get("mask_rel", 0.05)),
                "t_in": t_in, "t_out": t_out, "c": 3, "base": base,
                "val_rel_l2": float(m["rel"]), "val_tke_rel": float(m["tke"]),
                "val_mvpe_rel": float(m["mvpe"]), "weights": which,
                "stage": stage, "gstep": int(gstep),
                "loss_type": loss_type, "l2_normalize": l2_norm, "tke_weight": tke_w, "dt1_weight": dt1_w, "dt2_weight": dt2_w,
         "lead_weight": lead_weight,
                "mvpe_weight": mvpe_w, "fold": fold, "group_by": group_by,
                "val_re": fold_val_re, "val_groups": fold_val_re,
                "n_roll": n_roll, "unroll_mode": unroll_mode}
        if arch == "cfo":
            # CONDITIONAL on purpose: adding a key unconditionally would change the serialised
            # `meta` dict -- and therefore the .pth bytes -- of every future checkpoint of every
            # other arm, even though the model is identical.
            _d["cfo"] = dict(_cfo_cfg)
        if arch == "vit":   # VIT knob (2026-09-10): conditional -- absent for every other arch, so their meta bytes are unchanged
            _d["vit"] = dict(_vit_cfg)
        if arch == "no82":
            _d["no82"] = dict(_no82_cfg)
        if arch == "lsm82":
            _d["lsm82"] = dict(_lsm_cfg)  # includes validated derived shapes and parameter count
        if arch == "sinenet82":
            _d["sinenet82"] = dict(_sine_cfg)
        if arch == "uno82":
            _d["uno82"] = dict(_uno_cfg)
        if arch == "dpot82":
            _d["dpot82"] = dict(_dpot_cfg)
        if arch == "galerkin82":
            _d["galerkin82"] = dict(_gt_cfg)
        if arch == "dcno82":
            _d["dcno82"] = dict(_dc_cfg)
        if arch == "transolver2d82":
            _d["transolver2d82"] = dict(_ts_cfg)
        if arch == "cno2d82":
            _d["cno2d82"] = dict(_cno_cfg)
        if aux_w > 0.0:
            # AXP: conditional for the same reason as `cfo` above.  `aux_phases` is in
            # ARCH_KEYS, so build_from_meta rebuilds the aux head and a strict load works.
            _d["aux_phases"] = len(AUX_PHASES)
            _d["aux_phase_weight"] = aux_w
        if fisher_q > 0.0:
            # FGP: conditional for the same byte-identity reason as `cfo`/`aux` above.  The
            # mask changes TRAINING only; the parameter set is untouched, so build_from_meta
            # needs nothing -- this is pure provenance.
            _d["fisher_protect"] = fisher_q
            _d["fisher_batches"] = fisher_nb
        if _sag_on and _sag_meta is not None:
            # STRIDEAUG: conditional for the same byte-identity reason as `cfo`/`aux`/`fisher`.
            # Training-only knob; the parameter set is untouched -- pure provenance + witness.
            _d["strideaug"] = dict(_sag_meta)
        if _tooth is not None:
            # EQFREE TOOTH: conditional for the same byte-identity reason.  Training-only knob;
            # the parameter set is untouched -- provenance + the cropped-batch count (witness).
            _d["tooth"] = dict(_tooth, n_cropped_batches=int(_tooth_seen["n"]))
        if _dws is not None:   # DWSCHED knob (2026-09-10): conditional meta key (absent => every other arm's meta bytes unchanged)
            _d["distill_sched"] = {"start": _dws["start"], "end": _dws["end"], "shape": _dws["shape"], "stage": _dws["stage"],
                                   "const": _dws["const"], "n_applied": int(_dws_seen["n"]),
                                   "w_first": _dws_seen["first"], "w_last": _dws_seen["last"],
                                   "w_mean": (float(_dws_seen["sum"]) / int(_dws_seen["n"])) if _dws_seen["n"] else None}
        if _dfl_alpha != 1.0:
            _d["distill_fluct"] = {"alpha": float(_dfl_alpha), "stage": "real-finetune", "n_applied": int(_dfl_seen["calls"])}
        return _d

    def maybe_save(stage, gstep, raw, eva):
        """Model selection is UNCHANGED from v3 (best val rel-L2, raw vs EMA) so the arms stay
        comparable.  Caveat for the K-fold protocol: selecting on the same windows that are then
        scored out-of-fold is mildly optimistic -- it biases every arm the same way, so A/B
        comparisons hold, and `save_last: true` additionally dumps the unbiased final weights."""
        nonlocal best
        which, m = (("ema", eva) if _sel_key(eva) <= _sel_key(raw)
                    else ("raw", raw))
        if _sel_key(m) < best["key"]:
            sd = ({k: v.to(model.state_dict()[k].dtype) for k, v in ema.items()}
                  if which == "ema" else {k: v.detach().clone() for k, v in model.state_dict().items()})
            best = {"key": float(_sel_key(m)),
                    "rel": float(m["rel"]), "which": which, "stage": stage, "step": int(gstep),
                    "tke": float(m["tke"]), "mvpe": float(m["mvpe"])}
            Path(ckpt_out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": {k: v.cpu() for k, v in sd.items()},
                        "meta": _ckpt_meta(which, stage, gstep, m)}, ckpt_out)
            logger.info(f"    * new BEST val_relL2={best['rel']:.5f} tke={best['tke']:.5f} "
                        f"mvpe={best['mvpe']:.5f} ({which}) -> saved {ckpt_out}")

    # (mean, std) of the CURRENT stage, as torch tensors -- None means "do not denormalise",
    # which switches the mvpe term off.  Set at the top of every stage; step_loss reads it
    # through the closure so it always sees the stage it is actually training on.
    stage_denorm = None

    def step_loss(xb, yb, rolling, aux=None):
        """One optimisation objective.  rolling=False -> pure single-step on block 0."""
        K = n_roll if rolling else 1
        parts = []
        cur = xb
        pb = None
        for k in range(K):
            tgt = yb[:, k * roll_stride:k * roll_stride + t_out]
            last = (k == K - 1)
            if unroll_mode == "pushforward" and not last:
                with torch.no_grad():
                    pb = model(cur)
                cur = feedback(pb, xb, lch, cur, roll_stride)
                continue
            pb = model(cur)
            if _tooth is not None and (getattr(model, "_tooth", None) is not None or _tooth_box["boxes"] is not None):
                # EQFREE TOOTH: the model returned the INNER tooth (B,t_out,h,w,c); embed it into a copy of
                # the FULL target so every leg below keeps rel_l2 = ||e_tooth|| / ||y_full|| per sample.
                # ELW control (input: full): the model returned the FULL field; cut the same inner tooth out first.
                if getattr(model, "_tooth", None) is None:
                    pb = _tooth_inner(pb)
                if not _tooth_seen.get("embed_logged"):
                    _tooth_seen["embed_logged"] = True
                    logger.info(f"TOOTH EMBED: student tooth {tuple(pb.shape)} -> full {tuple(tgt.shape)} (input={_tooth['input']})")
                pb = _tooth_embed(pb, tgt)
            if _mimo:
                # MIMOA: fold the M-window slot axis into the BATCH so every loss leg below
                # sees standard (M*B, t_out, H, W, 3) tensors.  loss_fn's per-sample reduction
                # then yields exactly the mean of the M heads' losses against their own
                # targets (equal M-ths; asserted numerically in the mwave smoke), and the
                # folded `cur` feeds the distill teacher one standard window per folded
                # sample.  Safe to rebind cur: n_roll == 1 is enforced, so feedback never
                # runs, and every knob that reads cur upstream of here is refused with mimo.
                pb = _mimo_fold(pb, _mimo)
                tgt = _mimo_fold(tgt, _mimo)
                cur = _mimo_fold(cur, _mimo)
            if _SPS is not None:
                # SPSW.  h is a function of the INPUT only (the sband head is input-only by
                # design), so nothing here leaks the target and nothing is needed at inference.
                # Gated to the real stage: the band was fitted on real residuals.
                if ("real" in name) and stage_denorm is not None:
                    with torch.no_grad():
                        _sm, _ss = stage_denorm
                        _xph = cur[..., :lch] * _ss + _sm
                        _mm = _SPS["head"](_SPS["feat"](_xph, getattr(_SPS["head"], "n_coord", 0)))
                        _b, _t = pb.shape[0], pb.shape[1]
                        _mm = _mm.reshape(_b, _t, lch, pb.shape[2], pb.shape[3]).permute(0, 1, 3, 4, 2)
                        _bi = torch.searchsorted(
                            _SPS["ed"].reshape(_t * lch, -1).contiguous(),
                            _mm.permute(1, 4, 0, 2, 3).reshape(_t * lch, -1).contiguous())
                        _lo = torch.gather(_SPS["lo"].reshape(_t * lch, -1), 1, _bi)
                        _hi = torch.gather(_SPS["hi"].reshape(_t * lch, -1), 1, _bi)
                        _sh_ = (_t, lch, _b, pb.shape[2], pb.shape[3])
                        _lo = _lo.reshape(_sh_).permute(2, 0, 3, 4, 1)
                        _hi = _hi.reshape(_sh_).permute(2, 0, 3, 4, 1)
                        _r = (tgt[..., :lch] - pb[..., :lch]).detach() * _ss   # PHYSICAL residual
                        _h = torch.where(_r >= 0, _hi, -_lo).clamp_min(1e-8)
                        _ratio = _r.abs() / _h
                        _w = 1.0 + sps_lam * torch.exp(-((_ratio - 1.0) / sps_s) ** 2)
                        _w = _w / _w.mean()
                        assert abs(float(_w.mean()) - 1.0) < 1e-5, "SPSW mean weight != 1"
                        _sps_box["w"] = _w
                else:
                    _sps_box["w"] = None
            if _lwm_box is not None:
                # JAWSW: per-(lead,pixel,channel) loss weight from the EMA of the SQUARED
                # residual over PREVIOUS training batches (this step's residual enters the EMA
                # only AFTER the weight is read, so w never depends on the current prediction).
                # Both stages; the EMA is reset at each stage start.  eps is RELATIVE (1% of the
                # map mean) so the raw weight ratio is capped ~100x before mean-normalisation.
                # The weight rides make_loss's elem_w box, i.e. it multiplies the squared error
                # inside the existing reduction -- and therefore reaches the distillation term
                # too, exactly as LWa's lead weights do.  No RNG consumed.
                with torch.no_grad():
                    _le = _lwm_state["ema"]
                    if _le is not None:
                        _leps = 0.01 * float(_le.mean())
                        _lwr = (_le + _leps) ** (-lwm_s if lwm_mode == "prec" else lwm_s)
                        _lwm_ = _lwr / _lwr.mean()
                        _lwm_box["w"] = _lwm_.reshape(1, *_lwm_.shape)
                        if not _lwm_state["live"]:
                            _lwm_state["live"] = True
                            logger.info(f"JAWSW LIVE: first weight application -- mode={lwm_mode} "
                                        f"strength={lwm_s} w_min={float(_lwm_.min()):.4f} "
                                        f"w_max={float(_lwm_.max()):.4f} "
                                        f"w_mean={float(_lwm_.mean()):.6f} "
                                        f"ema_mean={float(_le.mean()):.4e}")
                    else:
                        _lwm_box["w"] = None           # first step of a stage: uniform
                    _lsq = ((pb[..., :lch] - tgt[..., :lch]).detach() ** 2).mean(0)
                    _lwm_state["ema"] = _lsq if _le is None else 0.99 * _le + 0.01 * _lsq
            if _gd_box is not None and _gd_box["gid"] is not None:
                # GDROA: per-sample weights exp(eta*EMA_g) from the per-AoA-group loss EMAs of
                # PREVIOUS steps, normalised to batch-mean 1; uniform until every group has an
                # EMA.  The elem box carries w^2 because elem_w multiplies the SQUARED error
                # inside the l2/rel_l2 sqrt: sqrt(sum cw*w^2*e^2) = w*sqrt(sum cw*e^2), so w^2
                # inside == exactly w on the per-sample loss (the loss_type guard enforces the
                # sqrt reduction).  The group loss is the run's own loss_type + chan_weight,
                # WITHOUT the elem tilt itself (self-referential otherwise).  No RNG consumed.
                with torch.no_grad():
                    _gid = _gd_box["gid"]
                    _ge = _gd_state["ema"]
                    if _gd_state["groups"] is not None and \
                            all(g in _ge for g in _gd_state["groups"]):
                        _gmx = max(_ge.values())
                        _gwv = np.asarray([math.exp(gdro_eta * (_ge[int(a)] - _gmx))
                                           for a in _gid], np.float64)
                        _gwv = _gwv / _gwv.mean()
                        _gwt = torch.as_tensor(_gwv, dtype=pb.dtype, device=pb.device)
                        _gd_box["w"] = (_gwt ** 2).reshape(-1, 1, 1, 1, 1)
                        if not _gd_state["live"]:
                            _gd_state["live"] = True
                            _gestr = ", ".join(f"{k}:{v:.4f}" for k, v in sorted(_ge.items()))
                            logger.info(f"GDROA LIVE: first weight application -- eta={gdro_eta} "
                                        f"ema={{{_gestr}}} w_min={_gwv.min():.4f} "
                                        f"w_max={_gwv.max():.4f} w_mean={_gwv.mean():.6f}")
                    else:
                        _gd_box["w"] = None            # warm-up: not every group seen yet
                    _ge2 = (pb[..., :lch] - tgt[..., :lch]).detach() ** 2
                    if _gd_state["cw"] is None and _chan_w is not None:
                        _gd_state["cw"] = torch.tensor(_chan_w, device=pb.device,
                                                       dtype=pb.dtype).reshape(1, 1, 1, 1, -1)
                    if _gd_state["cw"] is not None:
                        _ge2 = _ge2 * _gd_state["cw"]
                    _gnb = _ge2.shape[0]
                    _gse = _ge2.reshape(_gnb, -1).sum(1)
                    if loss_type == "rel_l2":
                        _gli = (_gse.sqrt() / (tgt[..., :lch] ** 2).reshape(_gnb, -1).sum(1)
                                .clamp(min=1e-12).sqrt())
                    else:
                        _gli = (_gse / (float(_ge2[0].numel()) if l2_norm else 1.0)).sqrt()
                    _gli = _gli.cpu().numpy()
                    for _gg in set(int(a) for a in _gid):
                        _glg = float(_gli[np.asarray(_gid) == _gg].mean())
                        _ge[_gg] = _glg if _gg not in _ge else 0.98 * _ge[_gg] + 0.02 * _glg
            if _hh:
                # CRPSH: the main supervised term IS the Gaussian NLL (REPLACE, not add --
                # the MWAVE LOCK's fixed decision (a)): l = nll_w * mean[0.5*(logvar~ +
                # resid^2/exp(logvar~))], logvar~ clamped to [-10, 4], u,v, normalized space,
                # BOTH stages.  The distill leg below keeps loss_fn on the MEAN unchanged.
                _lv5 = getattr(model, "_last_logvar", None)
                if _lv5 is None:
                    raise SystemExit("CRPSH: hetero_head armed but the forward stashed no "
                                     "_last_logvar -- the logvar head is not live")
                _lvc = _lv5[..., :lch].clamp(-10.0, 4.0)
                l = nll_w * 0.5 * (_lvc + (pb[..., :lch] - tgt[..., :lch]).pow(2)
                                   / _lvc.exp()).mean()
                if not _hh_seen["live"]:
                    _hh_seen["live"] = True
                    with torch.no_grad():
                        _lvr = _lv5[..., :lch]
                        logger.info(f"CRPSH LIVE: first NLL eval -- nll={float(l):.4e} "
                                    f"mean_logvar={float(_lvc.mean()):.4f} "
                                    f"[{float(_lvc.min()):.3f}, {float(_lvc.max()):.3f}] "
                                    f"frac_clamp_lo={float((_lvr <= -10.0).float().mean()):.4f} "
                                    f"frac_clamp_hi={float((_lvr >= 4.0).float().mean()):.4f}")
            else:
                l = loss_fn(pb[..., :lch], tgt[..., :lch])
            if _ref_ds > 0.0:
                _y1 = getattr(model, "_last_y1", None)
                if _y1 is not None:
                    l = (1.0 - _ref_ds) * l + _ref_ds * loss_fn(_y1[..., :lch], tgt[..., :lch])
            if bsp_w > 0.0:
                l = l + bsp_w * bsp_loss(pb[..., :lch], tgt[..., :lch], bsp_nb)
            if distill_w > 0.0 and _distill_model is not None and _distill_stage["on"]:
                # Born-Again: soft targets from the (no-grad) teacher on the SAME input.  Under the
                # 2026-08-10 pricing (rel at face, tke at zero) the smoothing objection is void.
                with torch.no_grad():
                    # `_tcur` slices the student's window down to whatever each teacher expects; the
                    # slice is from the END so the teacher sees the frames closest to the target.
                    _tcur = lambda _m: (cur if getattr(_m, "_t_in_own", cur.shape[1]) == cur.shape[1]
                                        else cur[:, -_m._t_in_own:])
                    _tpb = _distill_models[0](_tcur(_distill_models[0]))
                    if len(_distill_models) > 1:
                        for _tm in _distill_models[1:]:
                            _tpb = _tpb + _tm(_tcur(_tm))
                        _tpb = _tpb / float(len(_distill_models))
                    if _tooth is not None and (getattr(model, "_tooth", None) is not None or _tooth_box["boxes"] is not None):
                        if getattr(model, "_tooth", None) is None:
                            _tpb = _tooth_inner(_tpb)       # ELW: full-field teacher, same inner tooth
                        _tpb = _tooth_embed(_tpb, tgt)      # the teacher saw the same tooth; same embedding
                if _sag_box["m"] is not None:
                    # STRIDEAUG: the teacher (trained at k=1) predicts a stride-1 continuation of
                    # `cur`; on a stride-k window that contradicts the label, so the distill leg is
                    # computed on the k=1 windows of this batch only (loss_fn is a per-sample mean).
                    _sag_m = _sag_box["m"]
                    _sag_keep = ~_sag_m if not _sag_dist else torch.ones_like(_sag_m)
                    if bool(_sag_keep.any()):
                        l = l + distill_w * loss_fn(pb[_sag_keep][..., :lch], _tpb[_sag_keep][..., :lch])
                    if not _sag_seen["live"] and bool(_sag_m.any()):
                        _sag_seen["live"] = True
                        logger.info(f"STRIDEAUG LIVE: first batch carrying stride-k windows -- "
                                    f"batch {int(_sag_m.numel())} aug {int(_sag_m.sum())} "
                                    f"distill_on {int(_sag_keep.sum())} distill_on_aug={_sag_dist}")
                else:
                    if _dfl_alpha == 1.0 and _dsp_beta != 1.0:
                        if name != "real-finetune" or lch != 2:
                            raise RuntimeError("DSPAT escaped its real-finetune u/v scope")
                        l = l + distill_w * _dsp_loss(pb[..., :lch], _tpb[..., :lch], loss_fn,
                                                      beta=_dsp_beta, kernel=_dsp_kern)
                        if _dsp_seen["calls"] == 0:
                            logger.info(f"DSPAT LIVE: stage={name} beta={_dsp_beta} kernel={_dsp_kern} "
                                        f"KD_weight={distill_w} large-scale pull retained")
                        _dsp_seen["calls"] += 1
                    elif _dfl_alpha == 1.0:
                        l = l + distill_w * loss_fn(pb[..., :lch], _tpb[..., :lch])
                    else:
                        if name != "real-finetune" or lch != 2:
                            raise RuntimeError("DFLUCT escaped its real-finetune u/v scope")
                        l = l + distill_w * _dfl_loss(pb[..., :lch], _tpb[..., :lch], loss_fn, alpha=_dfl_alpha)
                        if _dfl_seen["calls"] == 0:
                            logger.info(f"DFLUCT LIVE: stage={name} alpha={_dfl_alpha} KD_weight={distill_w} mean residual retained")
                        _dfl_seen["calls"] += 1
            if _cob:
                # COBLEND: `pb` is the wrapper's uniform mean; the members' own outputs were
                # stashed NON-detached by CoBlend.forward on this very call.  Each member gets
                # the SAME recipe loss (loss_fn = rel_l2 + chan_weight, + distill against the
                # SAME teacher tensor `_tpb`, one teacher forward per step).  SUM over members,
                # so at (w_blend=0, lam_ind=1) member k's gradient is exactly its standalone
                # recipe's gradient.
                _cmo = getattr(model, "_last_member_outs", None)
                if _cmo is None or len(_cmo) != _cob:
                    raise SystemExit("COBLEND: forward stashed no member outputs -- the "
                                     "wrapper is not the live forward")
                _cli = None
                _cl_solo = []
                for _cyk in _cmo:
                    _clk = loss_fn(_cyk[..., :lch], tgt[..., :lch])
                    _cl_solo.append(float(_clk.detach()))
                    if distill_w > 0.0 and _distill_model is not None and _distill_stage["on"]:
                        _clk = _clk + distill_w * loss_fn(_cyk[..., :lch], _tpb[..., :lch])
                    _cli = _clk if _cli is None else _cli + _clk
                _cl_blend = l
                l = ((_cob_wb * _cl_blend) if _cob_wb > 0.0 else 0.0) \
                    + ((_cob_li * _cli) if _cob_li > 0.0 else 0.0)
                if not _cob_seen["live"]:
                    _cob_seen["live"] = True
                    logger.info(f"COBLEND LIVE: first armed loss eval [{name}] -- "
                                f"l_blend={float(_cl_blend):.5f} l_ind_sum={float(_cli):.5f} "
                                f"member_solo_loss={[round(q, 5) for q in _cl_solo]} "
                                f"distill_on={bool(distill_w > 0.0 and _distill_model is not None and _distill_stage['on'])} "
                                f"w_blend={_cob_wb} lam_ind={_cob_li} l_after={float(l):.5f}")
            if tspec_w > 0.0:
                # binned TEMPORAL power of the fluctuation, per pixel; the top band ([8-10] of 10
                # harmonics) is dropped by default -- that is where PIV noise dominates (the DT2a
                # lesson).  Bins on harmonics 1..10 after DC removal: [1-2] [3-4] [5-7] (+[8-10]).
                _pf = pb[..., :lch] - pb[..., :lch].mean(1, keepdim=True)
                _tf = tgt[..., :lch] - tgt[..., :lch].mean(1, keepdim=True)
                _Pp = torch.fft.rfft(_pf, dim=1).abs().pow(2)[:, 1:]
                _Qp = torch.fft.rfft(_tf, dim=1).abs().pow(2)[:, 1:]
                _bins = ((0, 2), (2, 4), (4, 7), (7, 10))
                _kept = _bins[:-1] if tspec_drop_top else _bins
                _terr = 0.0
                for _a, _b in _kept:
                    _pbnd = _Pp[:, _a:_b].sum(1)
                    _qbnd = _Qp[:, _a:_b].sum(1)
                    _num = ((_pbnd - _qbnd) ** 2).reshape(_pbnd.shape[0], -1).sum(1)
                    _den = (_qbnd ** 2).reshape(_qbnd.shape[0], -1).sum(1).clamp(min=1e-12)
                    _terr = _terr + (_num / _den).sqrt().mean()
                l = l + tspec_w * (_terr / len(_kept))
            if lowk_w > 0.0:
                # relative L2 of the LOW-PASSED field.  By Parseval the masked spectral norm IS an
                # L2 on the low-passed field, so this is the same objective the model is already
                # trained on, restricted to the scales RELF measured as still having headroom.
                # High-k content is masked EXACTLY, so the term is provably blind to it -- that is
                # the self-test, and a broken mask cannot pass it.
                _lk = _lowk_mask(pb.shape[2], pb.shape[3], lowk_cut, pb.device)
                _Pk = torch.fft.rfft2(pb[..., :lch], dim=(2, 3)) * _lk
                _Qk = torch.fft.rfft2(tgt[..., :lch], dim=(2, 3)) * _lk
                _num = (_Pk - _Qk).abs().pow(2).sum(dim=(1, 2, 3, 4))
                _den = _Qk.abs().pow(2).sum(dim=(1, 2, 3, 4)).clamp(min=1e-12)
                l = l + lowk_w * (_num / _den).sqrt().mean()
            if tke_w > 0.0:
                # TKP: with tke_phys the variances are restored to PHYSICAL units per channel
                # before summing, which is what scoring.py:94-99 does.  Without it the two
                # channels are divided by s_u^2 and s_v^2 (ratio 36.79) before being summed, and
                # the resulting objective drifts 0.743-0.800 against the scored one as the
                # fluctuation level changes.  Verified: with it the ratio is 1.000000 +/- 1.7e-07.
                _cs = stage_denorm[1] if (tke_phys and stage_denorm is not None) else None
                l = l + tke_w * tke_rel_l2(pb[..., :lch], tgt[..., :lch], chan_std=_cs).mean()
            if hpl1_w > 0.0:
                _hp_p, _hp_t = pb[..., :lch], tgt[..., :lch]
                _terms = []
                if hpl1_mode in ("temporal", "both"):
                    # the fluctuation about each window's own time mean -- the field whose per-pixel
                    # variance IS the tke subscore
                    _terms.append((_hp_p - _hp_p.mean(dim=1, keepdim=True),
                                   _hp_t - _hp_t.mean(dim=1, keepdim=True)))
                if hpl1_mode in ("spatial", "both"):
                    # the field minus its k x k spatial mean -- the small-scale structure an L2 loss
                    # is free to erase (k=3 is the same operator as vgain_head._hp_var).  `hpl1_scales`
                    # adds further kernels: one term per k, so a list targets a BAND of scales rather
                    # than the single octave k=3 reaches.  tke is a variance over the whole spectrum,
                    # and the spatial high-pass is the only knob measured to move it, so which scales
                    # it protects is the natural next variable.
                    def _sp(a, k):
                        n, T_, H_, W_, C_ = a.shape
                        pad = k // 2
                        z = a.permute(0, 1, 4, 2, 3).reshape(n * T_ * C_, 1, H_, W_)
                        sm = torch.nn.functional.avg_pool2d(
                            torch.nn.functional.pad(z, (pad, pad, pad, pad), mode="replicate"), k, stride=1)
                        return (z - sm).reshape(n, T_, C_, H_, W_).permute(0, 1, 3, 4, 2)
                    for _k in hpl1_scales:
                        _terms.append((_sp(_hp_p, _k), _sp(_hp_t, _k)))
                for _ep, _et in _terms:
                    _num = (_ep - _et).abs().mean(dim=(1, 2, 3, 4))
                    _den = _et.abs().mean(dim=(1, 2, 3, 4)).clamp(min=1e-12)
                    l = l + hpl1_w * (_num / _den).mean()
                if not _hpl1_seen["live"]:
                    _hpl1_seen["live"] = True
                    logger.info(f"HPL1 LIVE: mode={hpl1_mode} weight={hpl1_w} scales={hpl1_scales} "
                                f"terms={len(_terms)} "
                                f"first weighted value={float(hpl1_w * sum(((_ep - _et).abs().mean(dim=(1, 2, 3, 4)) / _et.abs().mean(dim=(1, 2, 3, 4)).clamp(min=1e-12)).mean() for _ep, _et in _terms)):.4e} "
                                f"l_after={float(l):.4e}")
            if pgrad_w > 0.0 and pb.shape[-1] > 2 and bool((tgt[..., 2] != 0).any()):
                # SPGa: supervise the PRESSURE GRADIENT (gauge-invariant) wherever the target
                # actually carries p -- i.e. the sim-pretrain stage with load_p: true.  A real
                # stage's p targets are all-zero and the term self-disables, so real-finetune is
                # byte-identical with the knob on.
                _gy_p = pb[..., 1:, :, 2] - pb[..., :-1, :, 2]
                _gy_t = tgt[..., 1:, :, 2] - tgt[..., :-1, :, 2]
                _gx_p = pb[..., :, 1:, 2] - pb[..., :, :-1, 2]
                _gx_t = tgt[..., :, 1:, 2] - tgt[..., :, :-1, 2]
                _num = ((_gy_p - _gy_t) ** 2).sum(dim=(1, 2, 3)) + ((_gx_p - _gx_t) ** 2).sum(dim=(1, 2, 3))
                _den = (_gy_t ** 2).sum(dim=(1, 2, 3)) + (_gx_t ** 2).sum(dim=(1, 2, 3))
                l = l + pgrad_w * (_num.clamp_min(0.0) / _den.clamp_min(1e-12)).sqrt().mean()
            if dt1_w > 0.0:
                l = l + dt1_w * dt_rel_l2(pb[..., :lch], tgt[..., :lch], 1).mean()
            if dt2_w > 0.0:
                l = l + dt2_w * dt_rel_l2(pb[..., :lch], tgt[..., :lch], 2).mean()
            if sobolev_w > 0.0:
                # SOBG: forward finite differences of the error field along H (y) and W (x),
                # normalized space, u,v only.  The two difference tensors have different shapes
                # (H-1 vs W-1), so "mean(|dx e|^2 + |dy e|^2)" is realised as mean-each-then-sum
                # -- recorded as such in the sobg_screen.py lock.  No RNG; nothing at inference.
                _se = pb[..., :lch] - tgt[..., :lch]
                _sgy = _se[:, :, 1:, :, :] - _se[:, :, :-1, :, :]
                _sgx = _se[:, :, :, 1:, :] - _se[:, :, :, :-1, :]
                _sg = (_sgx ** 2).mean() + (_sgy ** 2).mean()
                l = l + sobolev_w * _sg
                if not _sobg_seen["live"]:
                    # execution witness: proves the branch ran with a finite value in the real
                    # training path, and records the term's magnitude against the main loss.
                    _sobg_seen["live"] = True
                    logger.info(f"SOBG LIVE: first armed loss eval -- term(dx)="
                                f"{float((_sgx ** 2).mean()):.4e} term(dy)="
                                f"{float((_sgy ** 2).mean()):.4e} weighted={float(sobolev_w * _sg):.4e} "
                                f"l_after={float(l):.4e}")
            if semi_w > 0.0 and ("real" in name):
                # SEMIG: teacher-forced semigroup self-consistency, real-finetune only.  Compose
                # the 20-frame window [input frames half+1..T, DETACHED predicted leads 1..half]
                # and demand the model's output on it agree with the direct prediction's leads
                # half+1..T.  The inner prediction is detached (no double backprop through the
                # composition); gradient flows through BOTH remaining sides (the outer forward
                # and the direct late leads).  The composed frames' p-channel is ZERO -- the
                # real-input convention (real inputs carry p == 0 in normalized space; the sim
                # stage is NOT armed, where channel 2 is a nonzero normalised constant and a
                # zero fill would be off-distribution).  MSE, u,v, normalized space.  One extra
                # forward per step; no RNG consumed; nothing at inference.
                _shh = t_out // 2
                _spf = torch.zeros_like(cur[:, :_shh])
                _spf[..., :lch] = pb[:, :_shh, ..., :lch].detach()
                _sx2 = torch.cat([cur[:, _shh:], _spf], dim=1)
                _sp2 = model(_sx2)
                _sgm = ((_sp2[:, :t_out - _shh, ..., :lch] - pb[:, _shh:, ..., :lch]) ** 2).mean()
                l = l + semi_w * _sgm
                if not _semi_seen["live"]:
                    _semi_seen["live"] = True
                    logger.info(f"SEMIG LIVE: first armed eval -- term={float(_sgm):.4e} "
                                f"weighted={float(semi_w * _sgm):.4e} l_after={float(l):.4e} "
                                f"(inner pred DETACHED, composed p-channel ZERO, grad through "
                                f"both outer sides)")
            if ns_w > 0.0 and name == "sim-pretrain":
                # PINOS: discrete 2D NS momentum residual of the PREDICTED fields, sim stage
                # ONLY (the stage-name gate IS the real-untouched guard; the stage list is
                # pinned to [sim-pretrain, real-finetune] at parse time).  All quantities in
                # PHYSICAL units (stage denorm); p from the predicted channel 2, trained by
                # this term alone through grad-p (the gauge constant cancels).  Forward dt over
                # the leads, central spatial differences + 5-pt Laplacian on the interior
                # (1px crop), live-mask from the TARGET (sim body px are exactly 0 for all t),
                # normalised by the target's own mean squared dt-acceleration (detached), so
                # the term is dimensionless and O(1)-comparable to the main loss.
                assert pb.shape[-1] == 3, pb.shape
                _pm3, _ps3 = _ns_box["denorm"]
                _nu4 = _ns_box["nu"]
                assert _pm3 is not None and _nu4 is not None, "PINOS: stage state not armed"
                _U = pb[..., 0] * _ps3[0] + _pm3[0]              # (B,T,H,W) physical
                _V = pb[..., 1] * _ps3[1] + _pm3[1]
                _P = pb[..., 2] * _ps3[2] + _pm3[2]

                def _ddx(z):     # central, along W -> interior (B,T-1,H-2,W-2)
                    return (z[..., 1:-1, 2:] - z[..., 1:-1, :-2]) / (2.0 * _NS_DX)

                def _ddy(z):     # central, along PHYSICAL y: the h5 y axis DESCENDS with row
                    #  index (y[0,:] = 0.1899 top), so d/dy = (z[i-1] - z[i+1]) / (2 dy);
                    #  |dy| = dx to 0.014% (wave3a_smoke.py), treated as square.
                    return (z[..., :-2, 1:-1] - z[..., 2:, 1:-1]) / (2.0 * _NS_DX)

                def _lap(z):     # 5-point
                    return (z[..., 1:-1, 2:] + z[..., 1:-1, :-2] + z[..., 2:, 1:-1]
                            + z[..., :-2, 1:-1] - 4.0 * z[..., 1:-1, 1:-1]) / (_NS_DX ** 2)

                with torch.no_grad():
                    _Ut = tgt[..., 0] * _ps3[0] + _pm3[0]
                    _Vt = tgt[..., 1] * _ps3[1] + _pm3[1]
                    # body/FOV pixels are exact zeros PRE-normalization; the fp32
                    # normalize->denormalize round trip leaves ~1e-7 m/s, so the live test is
                    # a 1e-4 m/s threshold -- 3 orders above round-trip noise, 3 below U.
                    _lvm = ((_Ut.abs().amax(1) > 1e-4) | (_Vt.abs().amax(1) > 1e-4))
                    _lvm = _lvm[:, None, 1:-1, 1:-1].float()     # (B,1,H-2,W-2) live interior
                    _ncnt = (_lvm.sum() * float(_U.shape[1] - 1)).clamp(min=1.0)
                    _at = (_Ut[:, 1:] - _Ut[:, :-1]) / _NS_DT_SIM
                    _av = (_Vt[:, 1:] - _Vt[:, :-1]) / _NS_DT_SIM
                    _A2 = ((_at[..., 1:-1, 1:-1] ** 2 + _av[..., 1:-1, 1:-1] ** 2)
                           * _lvm).sum() / _ncnt
                _U0, _V0, _P0 = _U[:, :-1], _V[:, :-1], _P[:, :-1]
                _Ui, _Vi = _U0[..., 1:-1, 1:-1], _V0[..., 1:-1, 1:-1]
                _ru = (_U[:, 1:, 1:-1, 1:-1] - _Ui) / _NS_DT_SIM \
                    + _Ui * _ddx(_U0) + _Vi * _ddy(_U0) + _ddx(_P0) - _nu4 * _lap(_U0)
                _rv = (_V[:, 1:, 1:-1, 1:-1] - _Vi) / _NS_DT_SIM \
                    + _Ui * _ddx(_V0) + _Vi * _ddy(_V0) + _ddy(_P0) - _nu4 * _lap(_V0)
                _nst = ((_ru ** 2 + _rv ** 2) * _lvm).sum() / _ncnt / _A2.clamp(min=1e-12)
                l = l + ns_w * _nst
                if not _ns_box["live"]:
                    _ns_box["live"] = True
                    with torch.no_grad():
                        _c_dt = ((((_U[:, 1:, 1:-1, 1:-1] - _Ui) / _NS_DT_SIM) ** 2)
                                 * _lvm).sum() / _ncnt
                        _c_adv = (((_Ui * _ddx(_U0) + _Vi * _ddy(_U0)) ** 2) * _lvm).sum() / _ncnt
                        _c_gp = ((_ddx(_P0) ** 2) * _lvm).sum() / _ncnt
                        _c_vis = (((_nu4 * _lap(_U0)) ** 2) * _lvm).sum() / _ncnt
                    logger.info(f"PINOS LIVE: first armed loss eval -- term={float(_nst):.4e} "
                                f"weighted={float(ns_w * _nst):.4e} A2={float(_A2):.4e} "
                                f"rms(u-leg): dt={float(_c_dt.sqrt()):.3e} "
                                f"adv={float(_c_adv.sqrt()):.3e} gradp={float(_c_gp.sqrt()):.3e} "
                                f"visc={float(_c_vis.sqrt()):.3e} "
                                f"live_frac={float(_lvm.mean()):.4f} l_after={float(l):.4e}")
            if wv_w > 0.0 and name == "real-finetune":
                # WVORT (PIVGE-G): the WEAK-FORM pressure-free vorticity residual of the
                # PREDICTED rollout, REAL-FINETUNE STAGE ONLY -- the stage-name gate IS the
                # sim-untouched guard, and the stage list is pinned at parse time.  Physical
                # units (stage denorm).  The denominator is built from the TARGET velocities on
                # the SAME accepted subdomains and is DETACHED, so the term cannot be gamed by
                # inflating the prediction's own advective magnitude; it makes the term
                # dimensionless, O(1) and self-scaling across regimes -- the direct analogue of
                # PINOS's `A2`, and the quantity whose ground-truth value is PINOS's 73.0.
                # No RNG consumed; nothing at inference; u,v only.
                assert pb.shape[-1] >= 2, pb.shape
                _wvm, _wvs = _wv_box["denorm"]
                _wK, _wnu = _wv_box["K"], _wv_box["nu"]
                assert _wK is not None and _wnu is not None, "WVORT: stage state not armed"
                _wUp = pb[..., 0] * _wvs[0] + _wvm[0]
                _wVp = pb[..., 1] * _wvs[1] + _wvm[1]
                with torch.no_grad():
                    _wUt = tgt[..., 0] * _wvs[0] + _wvm[0]
                    _wVt = tgt[..., 1] * _wvs[1] + _wvm[1]
                    # body / out-of-FOV pixels are EXACT zeros pre-normalisation; the fp32
                    # normalise->denormalise round trip leaves ~1e-7 m/s, so this is a
                    # THRESHOLD test at 1e-4 m/s, never an equality (the PINOS smoke rule).
                    _wacc = _wK.accept_mask(_wvc.live_mask(_wUt, _wVt, 1e-4))
                    _wbt, _wTt, _wAt, _wVt2 = _wK.residual(_wUt, _wVt, nu=_wnu)
                    _wden = (_wacc * _wTt * _wTt).sum().clamp(min=1e-30)
                    _wfloor = ((_wacc * _wbt * _wbt).sum() / _wden)
                _wb, _wT, _wA, _wV = _wK.residual(_wUp, _wVp, nu=_wnu)
                # `zero`: drive int F phi of the PREDICTION to 0.  `data`: match the TARGET's own
                # int F phi (DETACHED, computed above), so the term is exactly 0 at pred == tgt.
                _wr = _wb if wv_tgt == "zero" else (_wb - _wbt)
                _wterm = (_wacc * _wr * _wr).sum() / _wden
                l = l + wv_w * _wterm
                if not _wv_box["live"]:
                    _wv_box["live"] = True
                    with torch.no_grad():
                        _wn = (_wacc.sum() * float(_wb.shape[1])).clamp(min=1.0)
                        _r = lambda z: float((((_wacc * z) ** 2).sum() / _wn).sqrt())
                    logger.info(f"WVORT LIVE: first armed loss eval -- target={wv_tgt} "
                                f"term={float(_wterm):.4e} "
                                f"weighted={float(wv_w * _wterm):.4e} "
                                f"GROUND-TRUTH FLOOR sum(b_tgt^2)/sum(T_tgt^2)="
                                f"{float(_wfloor):.4e} (PINOS's pointwise analogue read 73.0) | "
                                f"pred rms: T={_r(_wT):.3e} Adv={_r(_wA):.3e} Vis={_r(_wV):.3e} "
                                f"b={_r(_wb):.3e} | tgt rms: T={_r(_wTt):.3e} b={_r(_wbt):.3e} | "
                                f"accepted_frac={float(_wacc.mean()):.4f} "
                                f"nu=[{float(_wnu.min()):.3e},{float(_wnu.max()):.3e}] "
                                f"l_after={float(l):.4e}")
            if ds_w > 0.0 and name == "real-finetune":
                # DIVCONS: continuity-source statistics of the PREDICTED rollout vs the TARGET's (detached).
                assert pb.shape[-1] >= 2, pb.shape
                _dsm, _dss = _ds_box["denorm"]
                assert _dsm is not None, "DIVSTAT: stage state not armed"
                def _ds_div(U, V):        # index-unit central differences, replicate edges (the HYBIN convention: div = du/di_x - dv/di_y)
                    Fp = torch.nn.functional
                    ux = 0.5 * (Fp.pad(U, (1, 1, 0, 0), mode="replicate")[..., 2:] - Fp.pad(U, (1, 1, 0, 0), mode="replicate")[..., :-2])
                    vy = 0.5 * (Fp.pad(V, (0, 0, 1, 1), mode="replicate")[..., 2:, :] - Fp.pad(V, (0, 0, 1, 1), mode="replicate")[..., :-2, :])
                    return ux - vy
                _dUp = pb[..., 0] * _dss[0] + _dsm[0]; _dVp = pb[..., 1] * _dss[1] + _dsm[1]
                with torch.no_grad():
                    _dUt = tgt[..., 0] * _dss[0] + _dsm[0]; _dVt = tgt[..., 1] * _dss[1] + _dsm[1]
                    _dlive = ((_dUt.abs() + _dVt.abs()) > 1e-4).float()                       # (B,T,H,W)
                    _dacc = -torch.nn.functional.max_pool2d(-_dlive.reshape(-1, 1, *_dlive.shape[-2:]), 3, 1, 1).reshape_as(_dlive)  # 1px erosion
                    _dDt = _ds_div(_dUt, _dVt) * _dacc
                    _dn = _dacc.sum(dim=(2, 3)).clamp(min=1.0)                                   # (B,T)
                    _drt = ((_dDt ** 2).sum(dim=(2, 3)) / _dn).sqrt()                            # per-lead rms, target
                    _dmt = _dDt.sum(1) / _dacc.sum(1).clamp(min=1.0)                            # steady (time-mean) div field, target
                    _dst = (_dmt ** 2 * _dacc.prod(1)).sum(dim=(1, 2)); _dft = (((_dDt - _dmt.unsqueeze(1)) * _dacc) ** 2).sum(dim=(1, 2, 3))
                    _dfrt = _dst / (_dst + _dft + 1e-30)
                _dDp = _ds_div(_dUp, _dVp) * _dacc
                _drp = ((_dDp ** 2).sum(dim=(2, 3)) / _dn).sqrt()
                _dmp = _dDp.sum(1) / _dacc.sum(1).clamp(min=1.0)
                _dsp = (_dmp ** 2 * _dacc.prod(1)).sum(dim=(1, 2)); _dfp = (((_dDp - _dmp.unsqueeze(1)) * _dacc) ** 2).sum(dim=(1, 2, 3))
                _dfrp = _dsp / (_dsp + _dfp + 1e-30)
                _dterm = (((_drp - _drt) / _drt.clamp(min=1e-12)) ** 2).mean() + ((_dfrp - _dfrt) ** 2).mean()
                l = l + ds_w * _dterm
                if not _ds_box["live"]:
                    _ds_box["live"] = True
                    logger.info(f"DIVSTAT LIVE: first armed loss eval -- term={float(_dterm):.4e} weighted={float(ds_w * _dterm):.4e} "
                                f"| rms(div) pred/target per-lead ratio mean={float((_drp / _drt.clamp(min=1e-12)).mean()):.4f} "
                                f"| steady frac pred={float(_dfrp.mean()):.4f} tgt={float(_dfrt.mean()):.4f} "
                                f"| accepted_frac={float(_dacc.mean()):.4f} l_after={float(l):.4e}")
            if hero_w > 0.0 and name == "real-finetune":
                # HEROL: self-error-history supervision, real-finetune ONLY.  E = EMA
                # (beta=0.99) of the batch-mean SIGNED residual over PREVIOUS steps (the
                # JAWSW read-before-update convention: the weight at step k never sees step
                # k's residual).  Penalty is LINEAR in the current residual and positive when
                # it aligns with the historical pattern; sign(E) and the capped magnitude are
                # DETACHED, so gradient flows through the current prediction only.  Normalized
                # space, u,v; step 0 of the stage seeds E and applies no term.  tgt here is
                # the TRUE target (the recipe's distill term is a separate loss leg), so E
                # tracks the data residual.  No RNG consumed.
                _hres = pb[..., :lch] - tgt[..., :lch]
                _hE = _hero_state["ema"]
                if _hE is not None:
                    with torch.no_grad():
                        _hrms = _hE.pow(2).mean().sqrt()
                        _hmag = (_hE.abs() / (1e-12 + _hrms)).clamp(max=3.0)
                        _hdir = _hE.sign() * _hmag               # (T,H,W,lch), detached
                    _hterm = (_hres * _hdir).mean()
                    l = l + hero_w * _hterm
                    if not _hero_state["live"]:
                        _hero_state["live"] = True
                        logger.info(f"HEROL LIVE: first penalty application -- "
                                    f"term={float(_hterm):.4e} "
                                    f"weighted={float(hero_w * _hterm):.4e} "
                                    f"rms(E)={float(_hrms):.4e} "
                                    f"mag_mean={float(_hmag.mean()):.4f} mag_at_cap="
                                    f"{float((_hmag >= 3.0).float().mean()):.4f} "
                                    f"l_after={float(l):.4e}")
                with torch.no_grad():
                    _hbm = _hres.detach().mean(0)                # (T,H,W,lch) signed
                    _hero_state["ema"] = _hbm if _hE is None else 0.99 * _hE + 0.01 * _hbm
            if mvpe_w > 0.0 and stage_denorm is not None:
                # PHYSICAL space -- see the units note in the module docstring
                sm, ss = stage_denorm
                l = l + mvpe_w * mvpe_rel_l2(pb[..., :lch] * ss + sm,
                                             tgt[..., :lch] * ss + sm, sub_s).mean()
            if aux_w > 0.0 and aux is not None:
                # AXP: same loss, the 3 sibling-phase targets, phases folded into the sample
                # axis so loss_fn's per-sample reduction treats each (window, phase) alike.
                # n_roll == 1 is asserted at parse time, so K == 1 and `pb` is the aux forward.
                _ax = getattr(model, "_last_aux", None)
                if _ax is None:
                    raise SystemExit("aux_phase_weight > 0 but the forward stashed no _last_aux "
                                     "-- the aux head is not live")
                l = l + aux_w * loss_fn(_ax.flatten(0, 1)[..., :lch],
                                        aux.flatten(0, 1)[..., :lch])
            parts.append(l)
            if not last:
                nxt = feedback(pb, xb, lch, cur, roll_stride)
                cur = nxt.detach() if unroll_mode == "detach" else nxt
        return sum(parts) / len(parts)

    def causal_loss(xb, yb):
        """Self-supervised next-frame objective -- the GPT half of the seqformer recipe.

        Slot t predicts frame t+1.  The first t_in-1 targets come from inside the input window and
        the last from the first frame of the target window, so ALL t_in slots are supervised while
        the sequence stays within pos_t's length.  That is t_in next-frame targets per sample
        against the block objective's ONE block -- the density difference is the mechanism under
        test.  The temporal mask (asserted leak-free in seqformer.py) is what makes it honest.
        """
        tgt = torch.cat([xb[:, 1:], yb[:, :1]], dim=1)
        pb = model(xb, mode="causal")
        return loss_fn(pb[..., :lch], tgt[..., :lch])

    def mae_loss(xb):
        """MAEa: mask 50% of 8x8 spatial blocks (consistent across T and C — video-MAE
        tubes) and reconstruct the ORIGINAL t_in frames; t_out == t_in makes the same
        trunk fit.  Loss on the MASKED cells only, supervised channels only."""
        b_, t_, h_, w_, _ = xb.shape
        mb = (torch.rand(b_, 1, h_ // 8, w_ // 8, 1, device=xb.device) < 0.5).float()
        m = mb.repeat_interleave(8, dim=2).repeat_interleave(8, dim=3)
        xm = xb * (1.0 - m)
        pb = model(xm)
        d = (pb[..., :lch] - xb[..., :lch]) ** 2
        return (d * m).sum() / (m.expand_as(d).sum() + 1e-8)

    _mkp_stat = {"cov": None}

    def mkp_loss(xb):
        """MKPa (2026-08-12): masked SPACE-TIME patch reconstruction PRETEXT, gated by the
        per-stage knob `pretext: mask` (key absent == "none" == the bit-exact historical
        path: no branch taken, no RNG consumed).  Mask ~50% of 8x8-spatial x 5-frame-temporal
        blocks (zeroed in NORMALISED space, i.e. set to the pool mean) and reconstruct the
        FULL unmasked window with the run's own loss_fn on the supervised channels.
        NOT a rerun of the recorded MAEa null (README 2026-08-05, +0.009): that arm used
        full-time 8x8 tubes, masked-cells-only MSE, and PREPENDED an extra stage on the
        now-banned pools under the pre-reset mse recipe.  MKPa REPLACES the sim-pretrain
        objective on the legal pool under the modern rel_l2 / cond:stats recipe."""
        b_, t_, h_, w_, _ = xb.shape
        assert t_ % 5 == 0 and h_ % 8 == 0 and w_ % 8 == 0, (t_, h_, w_)
        mb = (torch.rand(b_, t_ // 5, h_ // 8, w_ // 8, 1, device=xb.device) < 0.5).float()
        m = (mb.repeat_interleave(5, dim=1)
               .repeat_interleave(8, dim=2)
               .repeat_interleave(8, dim=3))
        _mkp_stat["cov"] = float(m.mean())
        pb = model(xb * (1.0 - m))
        return loss_fn(pb[..., :lch], xb[..., :lch])

    def cfo_loss(xb, yb):
        """CFO flow matching: L = E_t,z || N(t, I(t;u)) - dI/dt ||^2 with I the stochastic
        interpolant around a quintic-Hermite temporal spline through the window's own frames.
        No ODE solver in this graph -- one velocity call per step, so training costs the same
        wall clock as the block objective.  `cat([xb, yb], 1)` is the contiguous 40-frame chunk
        build_windows already produces at n_roll=1, which is exactly the knot set."""
        from cfo_model import cfo_flow_loss
        return cfo_flow_loss(model, xb, yb, lch=lch)

    gstep = 0
    # Two param groups so a stage can run an LP-FT schedule (Kumar et al., arXiv:2202.10054):
    # `outc` is the output head, everything else is the backbone whose pretrained features full
    # fine-tuning is known to distort.  With equal lr the split is a no-op -- asserted in the
    # patch's self-check -- so this cannot disturb any existing arm.
    _HEAD_MODULES = ("outc",)
    _head_p = [q for n, q in model.named_parameters() if n.split(".")[0] in _HEAD_MODULES]
    _back_p = [q for n, q in model.named_parameters() if n.split(".")[0] not in _HEAD_MODULES]
    if not _head_p:
        _head_p, _back_p = [], list(model.parameters())     # e.g. seqformer: no `outc`
    muon_lr = float(cfg.get("muon_lr", 0.0))     # MUa: 0.0 = OFF, bit-exact
    muon_stage = str(cfg.get("muon_stage", ""))  # MUb: "" = all stages; "real" = finetune only
    _muon = None
    if muon_lr > 0.0:
        _muon_p = [q for q in _back_p if q.ndim >= 2]
        _back_p = [q for q in _back_p if q.ndim < 2]
        _muon = Muon(_muon_p, lr=muon_lr,
                     momentum=float(cfg.get("muon_momentum", 0.95)),
                     ns_steps=int(cfg.get("muon_ns_steps", 5)))
        logger.info(f"MUa:          muon_lr={muon_lr} over {len(_muon_p)} >=2D backbone tensors "
                    f"({sum(q.numel() for q in _muon_p)} params); AdamW keeps {len(_back_p)} 1D "
                    f"backbone tensors + {len(_head_p)} head tensors")
    opt = torch.optim.AdamW([{"params": _back_p, "is_head": False},
                             {"params": _head_p, "is_head": True}],
                            lr=1e-3, weight_decay=wd)
    model.train()

    _pg_u = {}     # PhysGuard protected bases {param_name: U}; filled after sim-pretrain
    # FGP (2026-08-12): diagonal-Fisher gradient masking.  fisher_protect: 0.0 (default,
    # absent from every existing config) is an EXACT no-op: no Fisher pass, no mask, no RNG
    # consumption, no meta key -- the only live-path cost is the falsy `if _fp_mask:` below,
    # the same default-off pattern as `_pg_u`.
    _fp_mask = {}  # {param_name: float mask, 1 = free / 0 = protected}; filled after sim-pretrain
    fisher_q = float(cfg.get("fisher_protect", 0.0))
    fisher_nb = int(cfg.get("fisher_batches", 200))
    if fisher_q > 0.0:
        if not 0.0 < fisher_q < 1.0:
            raise SystemExit(f"fisher_protect must be in (0, 1), got {fisher_q}")
        if bool(cfg.get("physguard", False)):
            raise SystemExit("fisher_protect and physguard both act on the post-backward "
                             "gradient of the same stage -- run one mechanism per arm")
        logger.info(f"FGP:          fisher_protect={fisher_q} fisher_batches={fisher_nb} -- "
                    f"diagonal-Fisher mask fitted on the sim-pretrain windows, gradients of the "
                    f"top-{fisher_q:.0%} per-tensor coordinates ZEROED in later stages "
                    f"(coarsened PhysGuard, arXiv 2606.16602)")
    _mix_a = float(cfg.get("mixup_alpha", 0.0) or 0.0)   # MIXa window-mixup Beta parameter
    _ref_ds = float(cfg.get("refine_ds", 0.0) or 0.0)   # REFb deep-supervision weight on y1
    _rot_deg = float(cfg.get("rot_aug_deg", 0.0) or 0.0)
    if _rot_deg > 0.0:
        logger.info(f"[rot_aug] +-{_rot_deg} deg frame+vector co-rotation, p=0.5, border padding")
    # ---- STRIDEAUG (2026-09-03): temporal stride-k windows (frame_step=k through build_windows)
    # from a REYNOLDS BAND of the real-finetune pool, APPENDED to that stage's window pool, with the
    # distillation leg MASKED on the stride-k windows (the teacher was trained at k=1 and predicts a
    # stride-1 continuation, which contradicts a stride-k label).  Absent == BIT-PATH-IDENTICAL off:
    # `_sag_on = False`, every armed branch is `if _sag_on:` / `if _sag_box["m"] is not None:`, no RNG
    # consumed, no meta key (same additive contract as WRa/AOSIM/DISTCON keep_files).
    _sag_cfg = cfg.get("strideaug")
    _sag_on = bool(_sag_cfg)
    _sag_box = {"m": None}        # per-batch bool aug mask, rebound by the loop before step_loss
    _sag_isaug_t = None           # (n,) bool over the real-finetune pool (torch), set at build
    _sag_meta = None
    _sag_dist = False
    _sag_seen = {"live": False}
    _sag_real_files = None
    if _sag_on:
        if not isinstance(_sag_cfg, dict):
            raise SystemExit("strideaug must be a dict {step, re_lo, re_hi, share, share_tol, "
                             "distill_on_aug}")
        _sag_step = int(_sag_cfg.get("step", 2))
        _sag_lo = int(_sag_cfg.get("re_lo", 0))
        _sag_hi = int(_sag_cfg.get("re_hi", 10 ** 9))
        _sag_share = float(_sag_cfg.get("share", 0.2))
        _sag_tol = float(_sag_cfg.get("share_tol", 0.05))
        _sag_dist = bool(_sag_cfg.get("distill_on_aug", False))
        _sag_amp = bool(_sag_cfg.get("amp_scale", False))      # SA1C: Reynolds-similar image
        _sag_unk = set(_sag_cfg) - {"step", "re_lo", "re_hi", "share", "share_tol", "distill_on_aug",
                                    "amp_scale"}
        if _sag_unk:
            raise SystemExit(f"strideaug: unknown key(s) {sorted(_sag_unk)} -- refusing a knob that "
                             f"would silently do nothing")
        if _sag_step < 2:
            raise SystemExit(f"strideaug.step must be >= 2 (step 1 IS the control), got {_sag_step}")
        if not (0.0 < _sag_share <= 0.5):
            raise SystemExit(f"strideaug.share must be in (0, 0.5], got {_sag_share}")
        if _sag_lo >= _sag_hi:
            raise SystemExit(f"strideaug: empty Re band [{_sag_lo}, {_sag_hi}]")
        for _k, _v in (("fuse.mimo", _mimo), ("fuse.coblend", _cob), ("aux_phase_weight", aux_w),
                       ("regime_weight", regime_w), ("mixup_alpha", _mix_a),
                       ("n_roll>1", int(n_roll > 1)), ("distill_arm", int(bool(cfg.get("distill_arm"))))):
            if _v:
                raise SystemExit(f"strideaug cannot combine with {_k} (={_v}); the per-window "
                                 f"aug mask is defined for the bare block objective only")
        _sag_meta = {"step": _sag_step, "re_lo": _sag_lo, "re_hi": _sag_hi, "share_target": _sag_share,
                    "distill_on_aug": _sag_dist, "amp_scale": _sag_amp, "armed": False}
        logger.info(f"STRIDEAUG:    step={_sag_step} re_band=[{_sag_lo}, {_sag_hi}] share={_sag_share} "
                    f"(tol {_sag_tol}) distill_on_aug={_sag_dist} -- real-finetune only, armed at build")
    for st in cfg["stages"]:
        name = st["name"]
        _distill_stage["on"] = (name == "real-finetune")
        if _tooth is not None:   # TOOTHPRE knob (2026-09-10): per-stage cropped-batch count (log only; no RNG, no tensor)
            _tooth_seen["stage_n0"] = int(_tooth_seen["n"])
        steps = int(st["steps"])
        lr0 = float(st["lr"])
        srcs = st["sources"]
        # a stage may opt out of unrolling (e.g. keep the sim pretrain single-step)
        st_roll = int(st.get("n_roll", n_roll))
        # `objective` selects this stage's training signal.  Default "block" is the historical
        # path; "causal" is the self-supervised next-frame pretrain and needs a sequence model,
        # because a frame-stacked U-net mixes all input times into every output time and therefore
        # cannot be made causal at all.
        # A stage may supervise more channels than the run's default.  The motivating case is the
        # sim pretrain, whose data carries pressure that the global `loss_channels: 2` discards.
        # Every nested loss/feedback closure reads `lch` from this scope, so rebinding it here is
        # what actually switches the objective for this stage.
        lch = int(st.get("loss_channels", _LCH_GLOBAL))
        if lch != _LCH_GLOBAL:
            if mvpe_w > 0.0:
                raise SystemExit(f"[{name}] stage loss_channels={lch} with mvpe_weight={mvpe_w}: "
                                 f"the MVPE term denormalises with m[:lch], so this would redefine "
                                 f"the metric rather than just widen the supervision")
            logger.info(f"  [{name}] loss_channels {_LCH_GLOBAL} -> {lch} (stage override)")
        lp_steps = int(st.get("lp_steps", 0))
        if lp_steps and not _head_p:
            raise SystemExit(f"[{name}] lp_steps needs an identifiable head module "
                             f"{_HEAD_MODULES}; this architecture has none")
        if lp_steps >= steps:
            raise SystemExit(f"[{name}] lp_steps={lp_steps} >= steps={steps}: that is linear "
                             f"probing, not LP-FT -- the FT phase would never run")
        if lp_steps:
            logger.info(f"  [{name}] LP-FT: first {lp_steps}/{steps} steps train the HEAD ONLY "
                        f"({sum(q.numel() for q in _head_p)/1e3:.1f} k of "
                        f"{sum(q.numel() for q in model.parameters())/1e6:.3f} M), then all")
        objective = str(st.get("objective", "block"))
        if objective not in ("block", "causal", "mae", "cfo"):
            raise SystemExit(f"[{name}] unknown objective {objective!r} (block|causal|mae|cfo)")
        # MKP (2026-08-12): a stage may swap the block objective's (input, target) pairing
        # for masked-window self-reconstruction, keeping architecture / loss_fn / plumbing.
        pretext = str(st.get("pretext", "none"))
        if pretext not in ("none", "mask"):
            raise SystemExit(f"[{name}] unknown pretext {pretext!r} (none|mask)")
        if pretext == "mask":
            if objective != "block":
                raise SystemExit(f"[{name}] pretext: mask replaces the block objective's "
                                 f"(input,target) pairing and needs objective: block "
                                 f"(got {objective!r})")
            if st_roll > 1:
                raise SystemExit(f"[{name}] pretext: mask is a same-window reconstruction "
                                 f"and does not unroll (got n_roll={st_roll})")
            if aux_w > 0.0:
                raise SystemExit(f"[{name}] pretext: mask is incompatible with "
                                 f"aux_phase_weight (the aux target is a forecast block)")
            logger.info(f"  [{name}] MKP pretext: mask ~50% of 8x8x5 space-time blocks, "
                        f"reconstruct the FULL window ({loss_type} on channels [:{lch}])")
        if aux_w > 0.0 and objective != "block":
            raise SystemExit(f"[{name}] aux_phase_weight > 0 needs objective: block "
                             f"(got {objective!r}) -- the aux target is a 20-frame block")
        if sam_rho > 0.0 and name == "real-finetune":
            # SAM's second pass re-evaluates the stage objective on the same batch; mae/mkp
            # draw fresh random masks (different objective AND a shifted RNG stream), so only
            # the deterministic block objective is admissible.
            if objective != "block" or pretext != "none":
                raise SystemExit(f"[{name}] sam_rho > 0 needs objective: block / pretext: none "
                                 f"(got objective={objective!r}, pretext={pretext!r})")
            logger.info(f"  [{name}] SAM armed: rho={sam_rho} (Foret et al. 2020; two-step, "
                        f"global-L2 ascent, bit-exact clone restore; this stage only -- "
                        f"~2x step compute, step COUNT unchanged)")
        if objective == "cfo":
            if arch != "cfo":
                raise SystemExit(f"[{name}] objective: cfo needs arch: cfo -- the flow-matching "
                                 f"target is a per-frame VELOCITY and only CFONet exposes a "
                                 f"velocity operator (got arch={arch!r})")
            if st_roll > 1:
                raise SystemExit(f"[{name}] objective: cfo does not unroll: the ODE integration "
                                 f"already spans the horizon (got n_roll={st_roll})")
            if tke_w > 0.0 or mvpe_w > 0.0 or bsp_w > 0.0 or dt1_w > 0.0 or dt2_w > 0.0:
                raise SystemExit(f"[{name}] objective: cfo is incompatible with tke_weight/"
                                 f"mvpe_weight/bsp_weight/dt1_weight/dt2_weight: all five are "
                                 f"functionals of a 20-frame BLOCK and are undefined against a "
                                 f"single-time velocity target (same reasoning as `causal`)")
            if lch > 2:
                raise SystemExit(f"[{name}] objective: cfo has a 2-channel ODE state (u,v); "
                                 f"loss_channels={lch} would supervise a channel the state does "
                                 f"not carry")
        if objective == "causal":
            if arch != "seqformer":
                raise SystemExit(f"[{name}] objective: causal needs arch: seqformer -- a "
                                 f"frame-stacked U-net cannot be causal (got arch={arch!r})")
            if tke_w > 0.0 or mvpe_w > 0.0 or dt1_w > 0.0 or dt2_w > 0.0:
                raise SystemExit(f"[{name}] objective: causal is incompatible with tke_weight/"
                                 f"mvpe_weight: both are functionals of a 20-frame BLOCK (a "
                                 f"temporal variance and a temporal mean) and are undefined for a "
                                 f"next-frame target")
            if st_roll > 1:
                raise SystemExit(f"[{name}] objective: causal does not unroll (got n_roll={st_roll})")
        Xs, Ys, Yxs, src_stats, src_labels = [], [], [], [], []
        win_re_srcs = []   # WRa: per-source window-Re arrays, same concat order as Xs
        win_aoa_srcs = []  # AOSIM: per-source window-AoA arrays, same concat order as Xs
        win_key_srcs = []  # CLUSTERW knob (2026-09-10): per-source window-key lists, same concat order as Xs
        _ns_files_for_nu = []  # PINOS: this stage's post-exclusion source files (nu lookup)
        for src in srcs:
            # ---- PROVENANCE GATE (2026-08-10).  The 2026-08-05 rules require every training sample to
            # trace to the competition's own release; a different SOURCE is banned (transforms OF the
            # release are fine).  A missing glob used only to warn-and-continue, so deleting a banned
            # directory silently trained a different model under the same config name.
            _g = str(src["glob"])
            if not os.environ.get("RPDE_ALLOW_ANY_SOURCE"):
                if "/data/RealPDE_competition_dataset/gdrive_raw/" not in _g:
                    raise SystemExit(
                        f"[{name}] ILLEGAL SOURCE: {_g}\n"
                        f"  Training data must come from the competition release (data/RealPDE_competition_dataset/gdrive_raw/**).\n"
                        f"  Set RPDE_ALLOW_ANY_SOURCE=1 only for local diagnostics -- never for a run "
                        f"whose output ships.")
            files = sorted(glob.glob(src["glob"]))
            if not files:
                raise SystemExit(f"[{name}] no files matched {_g!r} -- refusing to train on a "
                                 f"silently-empty source")
            # Official 2026-08-13 exclusion (see BANNED_REAL at the top of this file).  This is
            # the single choke point every stage's training list passes through -- ship and fold
            # configs alike -- so the banned real trajectory can never be ingested, whatever the
            # glob (extracted/train_real is the raw 82-file pool; local_bench is already
            # materialized without it, so the gate simply won't match there unless stale).
            files = _drop_banned_real(files, logger, label=f"[{name}] ")
            if excl:
                keep = [f for f in files if key_fn(f) not in excl]
                if len(keep) != len(files):
                    logger.info(f"  [{name}] excluded {len(files)-len(keep)} file(s) at val {group_by} from {src['glob']}")
                files = keep
            files = files[:int(src.get("n_files", 9999))]
            # ---- DISTCON (2026-09-01): explicit per-source trajectory support.
            # `keep_files: [<basename>, ...]` restricts THIS source to exactly the named
            # trajectories.  Absent  ==  bit-path-identical to before (no branch taken, no
            # RNG consumed, no state created) -- the WRa/SAMa/AXP/AOSIM/SOBG additive
            # contract.  Applied AFTER the banned-file gate and the fold exclusion, so a
            # keep list can never re-admit a banned or held-out trajectory.
            if "keep_files" in src:
                import hashlib as _kf_hl
                _kf_want = [str(x) for x in src["keep_files"]]
                if len(set(_kf_want)) != len(_kf_want):
                    raise SystemExit(f"[{name}] DISTCON keep_files has duplicate entries")
                _kf_have = {os.path.basename(f): f for f in files}
                _kf_miss = [b for b in _kf_want if b not in _kf_have]
                if _kf_miss:
                    raise SystemExit(
                        f"[{name}] DISTCON keep_files names {len(_kf_miss)} file(s) that are "
                        f"NOT in this source after the banned-file gate and the fold exclusion: "
                        f"{_kf_miss[:8]}{' ...' if len(_kf_miss) > 8 else ''}\n"
                        f"  A keep list that silently shrinks is a different experiment. Refusing.")
                _kf_before = len(files)
                files = [_kf_have[b] for b in sorted(_kf_want)]
                if len(files) < 2:
                    raise SystemExit(f"[{name}] DISTCON keep_files leaves {len(files)} file(s); "
                                     f"refusing to train on a degenerate pool")
                _kf_bn = sorted(os.path.basename(f) for f in files)
                _kf_re = sorted({file_re(f) for f in files})
                _kf_md5 = _kf_hl.md5("\n".join(_kf_bn).encode()).hexdigest()
                logger.info(f"  [{name}] DISTCON KEEP_FILES ACTIVE: kept {len(files)} of "
                            f"{_kf_before} file(s) | {len(_kf_re)} Re group(s) "
                            f"[{_kf_re[0]}..{_kf_re[-1]}] | list_md5 {_kf_md5} | "
                            f"Re {_kf_re}")
            if not files:
                logger.warning(f"  [{name}] no files matched {src['glob']!r}")
                continue
            if (ns_w > 0.0 and name == "sim-pretrain") or \
                    (wv_w > 0.0 and name == "real-finetune"):
                # PINOS / WVORT: collect the POST-EXCLUSION file list, so the nu map's keys are
                # exactly the (Re, AoA) pairs the windows can carry.  Both arms use the same
                # per-file nu = U95 * L / Re; they differ in WHICH stage collects.
                _ns_files_for_nu.extend(files)
            if _sag_on and name == "real-finetune":
                _sag_real_files = list(files)     # post-banned / post-fold-exclusion / post-keep
                _sag_real_stats = src["stats"]
            m, s = load_stats(src["stats"])
            _wlabel = f"{name}:{src.get('tag','src')}"
            _fluc_scale = None
            if src.get("fluc_scale_table"):
                # SIMDAMP: a per-case temporal damping table for the SIM pool only.  Damping the
                # released PIV would destroy the very signal the real stage must fit, so a table
                # on any non-sim source is a configuration error, not an experiment.
                if src.get("tag") != "comp_sim":
                    raise SystemExit(f"[{name}] fluc_scale_table is only legal on comp_sim "
                                     f"sources, got tag {src.get('tag')!r}")
                with open(src["fluc_scale_table"]) as _fh:
                    _fst = json.load(_fh)
                if _fst.get("status") != "trainable":
                    raise SystemExit(f"[{name}] SIMDAMP table {_fst.get('status')!r} is not "
                                     f"marked trainable; refusing to train from it")
                _fluc_scale = _fst["cases"]
                _al = np.array([float(v["alpha"] if isinstance(v, dict) else v)
                                for v in _fluc_scale.values()])
                logger.info(f"  [{name}] SIMDAMP ACTIVE: {len(_fluc_scale)} cases | alpha "
                            f"min/median/max {_al.min():.4f}/{np.median(_al):.4f}/{_al.max():.4f} "
                            f"| fold {_fst.get('fold')} | table md5 "
                            f"{_fst.get('table_md5', '?')} | producer {_fst.get('producer_sha256', '?')[:12]}")
            if aux_w > 0.0:
                x, y, ya = build_windows(files, sub_s, t_in, t_out, m, s,
                                         int(src.get("stride", stride)),
                                         logger, _wlabel, n_roll=st_roll,
                                         roll_stride=roll_stride, phases=src.get("phases"),
                                         load_p=bool(src.get("load_p", False)),
                                         p_cache=src.get("p_cache"),
                                         p_target_only=bool(cfg.get("p_target_only", False)),
                                         meas_noise=float(cfg.get("meas_noise", 0.0)),
                                         aux_phases=AUX_PHASES,
                                         frame_step=src.get("frame_step", 1),
                                         fluc_scale=_fluc_scale,
                                         fluc_scale_missing=src.get("fluc_scale_missing",
                                                                    "refuse"),
                                         dr_chan_gain=src.get("dr_chan_gain"))
                Yxs.append(ya)
            else:
                x, y = build_windows(files, sub_s, t_in, t_out, m, s,
                                     int(src.get("stride", stride)),
                                     logger, _wlabel, n_roll=st_roll,
                                     roll_stride=roll_stride, phases=src.get("phases"),
                                     load_p=bool(src.get("load_p", False)),
                                     p_cache=src.get("p_cache"),
                                     p_target_only=bool(cfg.get("p_target_only", False)),
                                     meas_noise=float(cfg.get("meas_noise", 0.0)),
                                     frame_step=src.get("frame_step", 1),
                                     fluc_scale=_fluc_scale,
                                     fluc_scale_missing=src.get("fluc_scale_missing", "refuse"),
                                     dr_chan_gain=src.get("dr_chan_gain"))
            Xs.append(x)
            Ys.append(y)
            src_stats.append((m, s))
            src_labels.append(_wlabel)
            # WRa: read the side-channel IMMEDIATELY after this build call, so a duplicated
            # source tag in one stage cannot alias another source's array.
            win_re_srcs.append(np.asarray(LOADED_WIN_RE[_wlabel], np.int64))
            win_aoa_srcs.append(np.asarray(LOADED_WIN_AOA[_wlabel], np.int64))
            win_key_srcs.append(list(LOADED_WIN_KEY[_wlabel]))   # CLUSTERW knob (2026-09-10)
        if _sag_on and name == "real-finetune":
            # STRIDEAUG build.  Refuse every silently-inert configuration loudly.
            if len(srcs) != 1:
                raise SystemExit(f"[{name}] strideaug needs exactly ONE real source (got {len(srcs)})")
            if str(st.get("objective", "block")) != "block" or str(st.get("pretext", "none")) != "none" \
                    or st_roll > 1:
                raise SystemExit(f"[{name}] strideaug needs objective block / pretext none / n_roll 1")
            if _sag_real_files is None or not _sag_real_files:
                raise SystemExit(f"[{name}] strideaug: real source file list was not captured")
            _sag_bad = [repr(f) for f in _sag_real_files if file_re(f) is None]
            if _sag_bad:
                raise SystemExit(f"[{name}] strideaug: {len(_sag_bad)} real file(s) with no parseable Re: "
                                 f"{_sag_bad[:5]} -- refusing to band on garbage")
            _sag_files = [f for f in _sag_real_files if _sag_lo <= file_re(f) <= _sag_hi]
            if not _sag_files:
                raise SystemExit(f"[{name}] strideaug: Re band [{_sag_lo}, {_sag_hi}] selects 0 of "
                                 f"{len(_sag_real_files)} files -- refusing an inert knob")
            import h5py as _sag_h5
            _sag_H = t_in + t_out
            _sag_avail = 0
            for _f in _sag_files:
                with _sag_h5.File(_f, "r") as _fh:
                    _T = int(_fh["u"].shape[0])
                _sag_avail += max(0, _T - (_sag_H - 1) * _sag_step)   # stride-1 window starts at step k
            _sag_nreal = int(sum(int(x.shape[0]) for x in Xs))
            _sag_ntarget = _sag_share / (1.0 - _sag_share) * _sag_nreal
            _sag_stride = max(1, int(round(_sag_avail / _sag_ntarget)))
            _sag_m, _sag_s = load_stats(_sag_real_stats)
            _sag_label = f"{name}:strideaug"
            _sag_x, _sag_y = build_windows(_sag_files, sub_s, t_in, t_out, _sag_m, _sag_s, _sag_stride,
                                         logger, _sag_label, n_roll=st_roll, roll_stride=roll_stride,
                                         meas_noise=float(cfg.get("meas_noise", 0.0)),
                                         frame_step=_sag_step)
            _sag_naug = int(_sag_x.shape[0])
            if _sag_amp and _sag_naug:
                # SA1C AMPSCALE: u,v (physical) -> k*u, k*v on X and Y of the stride-k windows, i.e. in
                # normalised space x_n -> k*x_n + (k-1)*mu/sigma per channel; p (channel 2) untouched.
                _sag_mu = np.asarray(_sag_m, np.float32).reshape(-1)[:2]
                _sag_sg = np.asarray(_sag_s, np.float32).reshape(-1)[:2]
                _sag_off = ((_sag_step - 1.0) * _sag_mu / _sag_sg).astype(np.float32)
                _sag_b4 = (float(_sag_x[..., 0].mean()), float(_sag_x[..., 0].std()),
                           float(_sag_x[..., 1].mean()), float(_sag_x[..., 1].std()))
                for _arr in (_sag_x, _sag_y):
                    _arr[..., :2] = (float(_sag_step) * _arr[..., :2] + _sag_off).astype(np.float32)
                _sag_af = (float(_sag_x[..., 0].mean()), float(_sag_x[..., 0].std()),
                           float(_sag_x[..., 1].mean()), float(_sag_x[..., 1].std()))
                assert abs(_sag_af[1] - _sag_step * _sag_b4[1]) < 1e-3 * max(1.0, _sag_af[1]), (_sag_b4, _sag_af)
                assert abs(_sag_af[0] - (_sag_step * _sag_b4[0] + float(_sag_off[0]))) < 1e-3, (_sag_b4, _sag_af)
                logger.info(f"  [{name}] STRIDEAUG AMPSCALE: k={_sag_step} normalised map x -> {_sag_step}*x + "
                            f"{_sag_off.tolist()} (u,v) | aug-pool mean/std u,v BEFORE {_sag_b4} AFTER {_sag_af} "
                            f"(cond planes of the stride-k windows now read a {_sag_step}x-amplitude flow)")
                _sag_meta.update({"amp_before": list(_sag_b4), "amp_after": list(_sag_af),
                                  "amp_offset": _sag_off.tolist()})
            if _sag_naug == 0:
                raise SystemExit(f"[{name}] strideaug built 0 windows -- refusing an inert knob")
            _sag_share_r = _sag_naug / float(_sag_nreal + _sag_naug)
            if abs(_sag_share_r - _sag_share) > _sag_tol:
                raise SystemExit(f"[{name}] strideaug realised share {_sag_share_r:.4f} is outside "
                                 f"{_sag_share} +- {_sag_tol} (stride {_sag_stride}, {_sag_naug} aug / "
                                 f"{_sag_nreal} real) -- refusing")
            _sag_hist = LOADED_FRAME_STEP[_sag_label]["hist"]
            if set(_sag_hist) != {_sag_step}:
                raise SystemExit(f"[{name}] strideaug: realised frame-step histogram {_sag_hist} is "
                                 f"not {{{_sag_step}}}")
            Xs.append(_sag_x)
            Ys.append(_sag_y)
            win_re_srcs.append(np.asarray(LOADED_WIN_RE[_sag_label], np.int64))
            win_aoa_srcs.append(np.asarray(LOADED_WIN_AOA[_sag_label], np.int64))
            _sag_isaug_t = torch.from_numpy(np.concatenate([np.zeros(_sag_nreal, bool),
                                                           np.ones(_sag_naug, bool)]))
            _sag_groups = sorted({int(file_re(f)) for f in _sag_files})
            _sag_meta.update({"armed": True, "n_files": len(_sag_files), "re_groups": _sag_groups,
                             "n_windows_stride2": _sag_naug, "n_real": _sag_nreal,
                             "share_realised": float(_sag_share_r), "stride": int(_sag_stride),
                             "hist": {int(a): int(b) for a, b in _sag_hist.items()}})
            logger.info(f"  [{name}] STRIDEAUG ARMED: n_files={len(_sag_files)} re_groups={_sag_groups} "
                        f"n_windows_stride2={_sag_naug} n_real={_sag_nreal} share={_sag_share_r:.4f} "
                        f"(target {_sag_share}) stride={_sag_stride} step={_sag_step} hist={_sag_hist} "
                        f"distill_on_aug={_sag_dist}")
        if not Xs:
            raise SystemExit(f"stage {name}: no data")

        # --- v4: arm the mvpe term for this stage, or refuse to, loudly ---
        stage_denorm = None
        if mvpe_w > 0.0:
            if not bool(st.get("mvpe", True)):
                logger.info(f"  [{name}] mvpe term OFF (stage sets mvpe: false)")
            elif len(src_stats) != 1:
                logger.warning(f"  [{name}] MVPE TERM OFF: the stage mixes {len(src_stats)} sources, "
                               f"so one batch carries two different normalisations and there is no "
                               f"single correct denormalisation to physical units")
            else:
                m, s = src_stats[0]
                stage_denorm = (torch.from_numpy(np.ascontiguousarray(m[:lch])).to(dev),
                                torch.from_numpy(np.ascontiguousarray(s[:lch])).to(dev))
                if _pd_box is not None:
                    _pd_box["ms"] = (stage_denorm[0], stage_denorm[1])
                logger.info(f"  [{name}] mvpe term ON weight={mvpe_w} "
                            f"denorm mean={m[:lch].tolist()} std={s[:lch].tolist()}")

        if lch != _LCH_GLOBAL:
            for (m_, s_), src in zip(src_stats, srcs):
                bad = [c for c in range(_LCH_GLOBAL, lch) if abs(float(np.asarray(s_).ravel()[c])) < 1e-8]
                if bad:
                    raise SystemExit(f"[{name}] source {src.get('tag','?')} has std~0 in channel(s) "
                                     f"{bad}; supervising them normalises by zero. This is what the "
                                     f"real pool's pressure looks like -- it is padding, not data")

        # --- DEFECT 4 (2026-07-30): the guard above inspects the STATS FILE, so it passed
        # while build_windows was substituting np.zeros_like(u) for `p` and loss_channels: 3
        # was supervising a constant.  This one inspects THE DATA that was actually loaded.
        for lbl, src in zip(src_labels, srcs):
            cs = np.asarray(LOADED_CH_STD.get(lbl, np.zeros(3)), np.float64).ravel()
            dead = [c for c in range(min(lch, cs.size)) if cs[c] < 1e-6]
            if dead:
                raise SystemExit(
                    f"[{name}] source {src.get('tag','?')} ({lbl}): loss_channels={lch} but the "
                    f"LOADED DATA is constant in channel(s) {dead} (per-channel std "
                    f"{cs.tolist()}). Supervising a constant is a no-op that silently rescales "
                    f"the loss. If this is `p`, set `load_p: true` on the source (and use files "
                    f"that HAVE p); otherwise lower loss_channels. The stats-file guard above "
                    f"cannot see this: comp_sim's stats say p std = 12.497 while the array held "
                    f"zeros.")
        Xtr = torch.from_numpy(np.concatenate(Xs, 0))
        Ytr = torch.from_numpy(np.concatenate(Ys, 0))
        Yaux_tr = torch.from_numpy(np.concatenate(Yxs, 0)) if aux_w > 0.0 else None
        if aux_w > 0.0:
            assert Yaux_tr.shape[0] == Ytr.shape[0], (Yaux_tr.shape, Ytr.shape)
            logger.info(f"  [{name}] AXP aux targets: {tuple(Yaux_tr.shape)} "
                        f"({Yaux_tr.numel()*4/1e9:.2f} GB, weight={aux_w})")
        del Xs, Ys, Yxs
        if cfg.get("distill_arm") and name == "real-finetune":
            # Teacher-target blending == mixture-loss distillation (MSE identity, see patch doc).
            _dw = float(cfg.get("distill_w", 0.5))
            _sfx = f"_f{fold}" if fold is not None else ""
            _tp = CASE_DIR.parents[1] / "checkpoints" / "t1_u-net"
            _tp = _tp / f"{cfg['distill_arm']}{_sfx}.pth"
            _td = torch.load(_tp, map_location="cpu", weights_only=False)
            from unet_model import build_from_meta as _bfm
            _tea = _bfm(_td.get("meta", {}), t_in, t_out, 3).to(dev).eval()
            _tea.load_state_dict(_td["model_state_dict"], strict=True)
            with torch.no_grad():
                for _i in range(0, Xtr.shape[0], 64):
                    _pb = _tea(Xtr[_i:_i + 64].to(dev)).float().cpu()
                    Ytr[_i:_i + 64] = (1.0 - _dw) * Ytr[_i:_i + 64] + _dw * _pb
            logger.info(f"  [distill] targets blended: w={_dw} teacher={_tp.name} "
                        f"({Xtr.shape[0]} windows)")
            del _tea, _td
            torch.cuda.empty_cache()
        n = Xtr.shape[0]
        # ---- WRa: regime-weighted sampling (real-finetune only; None == bit-exact off) ----
        if _sag_isaug_t is not None:
            assert name == "real-finetune" and int(_sag_isaug_t.shape[0]) == n, \
                ("STRIDEAUG alignment", name, tuple(_sag_isaug_t.shape), n)
        _rw_probs, _rw_rn = None, None
        _sw_probs = None   # CLUSTERW knob (2026-09-10)
        if regime_w > 0.0 and name == "real-finetune":
            _wre = np.concatenate(win_re_srcs) if win_re_srcs else np.zeros(0, np.int64)
            assert _wre.shape[0] == n, (_wre.shape, n)          # alignment with Xtr
            assert (_wre >= 0).all(), \
                f"[{name}] WRa: unparseable Re in {int((_wre < 0).sum())} window(s) -- " \
                f"refusing to tilt on garbage"
            _u_re = np.unique(_wre)                              # sorted ascending
            if _u_re.size < 2:
                raise SystemExit(f"[{name}] WRa: only {_u_re.size} distinct Re in the train "
                                 f"pool -- a rank tilt is undefined")
            _rank = {int(r): i / (_u_re.size - 1.0) for i, r in enumerate(_u_re)}
            _rw_rn = np.asarray([_rank[int(r)] for r in _wre], np.float64)
            _w = 1.0 + regime_w * _rw_rn
            _rw_probs = torch.from_numpy(_w)                     # multinomial normalises
            # self-test: realized tilt at the REAL batch size, on a PRIVATE generator so the
            # check itself never touches the training RNG stream.
            _g = torch.Generator().manual_seed(0)
            _nb = 256
            _smp = torch.multinomial(_rw_probs, min(bs, n) * _nb, replacement=True,
                                     generator=_g).numpy()
            _top = _rw_rn >= 0.75
            _unif = float(_top.mean())
            _expc = float(_w[_top].sum() / _w.sum())
            _real = float((_rw_rn[_smp] >= 0.75).mean())
            logger.info(f"  [{name}] WRa ARMED: regime_weight={regime_w} | {n} windows, "
                        f"{_u_re.size} Re groups [{int(_u_re.min())}..{int(_u_re.max())}] | "
                        f"top/bottom prob ratio {1.0 + regime_w:.2f} | top-quartile-Re "
                        f"(rank_norm>=0.75) fraction: uniform {_unif:.4f} expected {_expc:.4f} "
                        f"realized({_nb}x batch={min(bs, n)}) {_real:.4f}")
            assert abs(_real - _expc) < 0.02, (_real, _expc)     # tilt sanity gate
        # >>> CLUSTERW knob (2026-09-10): state-cluster inverse-frequency draw (real-finetune only; None == off)
        if _sw_cfg is not None and name == "real-finetune":
            import hashlib as _sw_hl
            if _sag_on or _mimo or _cob or gdro_eta > 0.0 or _rw_probs is not None:
                raise SystemExit(f"[{name}] state_weight cannot combine with strideaug / fuse.mimo / fuse.coblend / "
                                 f"group_dro_eta / regime_weight")
            if fold is None or fold_val_files is None:
                raise SystemExit(f"[{name}] STATEW: the weight file is fold-fitted; this run has no --fold")
            _sw_keys = [_k for _ks in win_key_srcs for _k in _ks]
            if len(_sw_keys) != n:
                raise SystemExit(f"[{name}] STATEW: {len(_sw_keys)} window keys vs {n} windows -- alignment broken")
            _sw_md5 = _sw_hl.md5(Path(_sw_cfg["file"]).read_bytes()).hexdigest()
            _swz = np.load(_sw_cfg["file"], allow_pickle=False)
            _sw_fk = [str(_x) for _x in _swz["keys"]]
            _sw_lab = np.asarray(_swz["labels"], np.int64)
            _sw_k = int(_swz["k"])
            _sw_valmd5 = _sw_hl.md5("\n".join(sorted(Path(_f).name for _f in fold_val_files)).encode()).hexdigest()
            _sw_got = {"fold": int(_swz["fold"]), "manifest": str(_swz["manifest"]), "val_md5": str(_swz["val_md5"])}
            _sw_want = {"fold": int(fold), "manifest": Path(cfg["kfold_manifest"]).name, "val_md5": _sw_valmd5}
            if _sw_got != _sw_want:
                raise SystemExit(f"[{name}] STATEW: weight file fitted for {_sw_got}, this run is {_sw_want}")
            _sw_leak = set(str(_x) for _x in _swz["fit_files"]) & set(Path(_f).name for _f in fold_val_files)
            if _sw_leak:
                raise SystemExit(f"[{name}] STATEW: held-out file(s) inside the cluster fit: {sorted(_sw_leak)}")
            if len(set(_sw_fk)) != len(_sw_fk) or len(set(_sw_keys)) != len(_sw_keys) or len(_sw_lab) != len(_sw_fk):
                raise SystemExit(f"[{name}] STATEW: duplicate window keys or key/label length mismatch")
            if set(_sw_fk) != set(_sw_keys):
                raise SystemExit(f"[{name}] STATEW: pool keys != weight-file keys (pool-only "
                                 f"{len(set(_sw_keys) - set(_sw_fk))}, file-only {len(set(_sw_fk) - set(_sw_keys))}) "
                                 f"-- stale stride / phase / fold / pool")
            _sw_map = dict(zip(_sw_fk, _sw_lab.tolist()))
            _sw_c = np.asarray([_sw_map[_k] for _k in _sw_keys], np.int64)
            _sw_cnt = np.bincount(_sw_c, minlength=_sw_k)
            if _sw_c.min() < 0 or _sw_c.max() >= _sw_k or (_sw_cnt == 0).any():
                raise SystemExit(f"[{name}] STATEW: bad cluster labels or an empty cluster {_sw_cnt.tolist()}")
            _sw_share = _sw_cnt / float(n)
            _sw_w = _sw_share[_sw_c] ** (-float(_sw_cfg["alpha"]))
            _sw_w = _sw_w / _sw_w.mean()
            _sw_probs = torch.from_numpy(_sw_w)                   # multinomial normalises
            _sw_exp = np.asarray([_sw_w[_sw_c == _j].sum() for _j in range(_sw_k)]) / _sw_w.sum()
            # self-test at the REAL batch size on a PRIVATE generator (never the training RNG stream), as WRa
            _sw_g = torch.Generator().manual_seed(0)
            _sw_nb = 256
            _sw_smp = torch.multinomial(_sw_probs, min(bs, n) * _sw_nb, replacement=True, generator=_sw_g).numpy()
            _sw_real = np.bincount(_sw_c[_sw_smp], minlength=_sw_k) / float(_sw_smp.size)
            _sw_ess = float(_sw_w.sum() ** 2 / (n * (_sw_w ** 2).sum()))
            _sw_cmd5 = _sw_hl.md5(("\n".join(_sw_fk) + "|" + ",".join(map(str, _sw_lab.tolist()))).encode()).hexdigest()
            logger.info(f"  [{name}] STATEW ARMED: state_weight file={Path(_sw_cfg['file']).name} md5={_sw_md5} "
                        f"content_md5={_sw_cmd5} alpha={_sw_cfg['alpha']} | {n} pool windows SET-EQUAL to {len(_sw_fk)} "
                        f"file keys | fold {fold} {_sw_want['manifest']} val_md5 {_sw_valmd5[:12]} | k={_sw_k} "
                        f"pool shares {np.round(_sw_share, 4).tolist()} | expected draw shares "
                        f"{np.round(_sw_exp, 4).tolist()} | realized({_sw_nb}x batch={min(bs, n)}) "
                        f"{np.round(_sw_real, 4).tolist()} | ESS {_sw_ess:.4f} w_min {float(_sw_w.min()):.4f} "
                        f"w_max {float(_sw_w.max()):.4f}")
            assert float(np.abs(_sw_real - _sw_exp).max()) < 0.02, (_sw_real, _sw_exp)   # draw sanity gate
        # <<< CLUSTERW knob (2026-09-10)
        # ---- AOSIM: separated-regime sim sampling tilt (sim-pretrain only; None == off) ----
        _sa_probs, _sa_hi = None, None
        if sim_aoa_w != 1.0 and name == "sim-pretrain":
            assert _rw_probs is None, "one sampling tilt per stage (WRa is finetune-only)"
            _waoa = np.concatenate(win_aoa_srcs) if win_aoa_srcs else np.zeros(0, np.int64)
            assert _waoa.shape[0] == n, (_waoa.shape, n)         # alignment with Xtr
            assert (_waoa >= 0).all(), \
                f"[{name}] AOSIM: unparseable AoA in {int((_waoa < 0).sum())} window(s) -- " \
                f"refusing to tilt on garbage"
            _sa_hi = np.isin(_waoa, (15, 20))
            _n_hi = int(_sa_hi.sum())
            if _n_hi == 0 or _n_hi == n:
                raise SystemExit(f"[{name}] AOSIM: upweighted class has {_n_hi}/{n} windows -- "
                                 f"a tilt is undefined (fold exclusion may have emptied it)")
            _w = np.where(_sa_hi, float(sim_aoa_w), 1.0)
            _sa_probs = torch.from_numpy(_w)                     # multinomial normalises
            # self-test at the REAL batch size on a PRIVATE generator (never the training RNG)
            _g = torch.Generator().manual_seed(0)
            _nb = 256
            _smp = torch.multinomial(_sa_probs, min(bs, n) * _nb, replacement=True,
                                     generator=_g).numpy()
            _unif = float(_sa_hi.mean())
            _expc = float(_w[_sa_hi].sum() / _w.sum())
            _real = float(_sa_hi[_smp].mean())
            logger.info(f"  [{name}] AOSIM ARMED: sim_aoa_weight={sim_aoa_w} | {n} windows, "
                        f"{_n_hi} upweighted (AoA {sorted(set(int(v) for v in _waoa[_sa_hi]))} "
                        f"present of {sorted(set(int(v) for v in _waoa))}) | hi-AoA fraction: "
                        f"uniform {_unif:.4f} expected {_expc:.4f} "
                        f"realized({_nb}x batch={min(bs, n)}) {_real:.4f}")
            assert abs(_real - _expc) < 0.02, (_real, _expc)     # tilt sanity gate
        # ---- WAVE 2 per-stage arming (all four knobs no-op here when off) ----
        if _lwm_box is not None:
            _lwm_state["ema"] = None               # JAWSW: per-stage residual statistics
            _lwm_box["w"] = None
            logger.info(f"  [{name}] JAWSW: EMA reset (per-stage map; beta=0.99)")
        _gd_stage_gid = None
        if _gd_box is not None:
            _gd_box["gid"] = None
            _gd_box["w"] = None
        if gdro_eta > 0.0 and name == "real-finetune":
            _gaoa = np.concatenate(win_aoa_srcs) if win_aoa_srcs else np.zeros(0, np.int64)
            assert _gaoa.shape[0] == n, (_gaoa.shape, n)         # alignment with Xtr
            assert (_gaoa >= 0).all(), \
                f"[{name}] GDROA: unparseable AoA in {int((_gaoa < 0).sum())} window(s)"
            _gd_state["groups"] = sorted(set(int(a) for a in _gaoa))
            if len(_gd_state["groups"]) != 5:
                raise SystemExit(f"[{name}] GDROA: expected 5 AoA groups, pool has "
                                 f"{_gd_state['groups']}")
            _gd_stage_gid = _gaoa
            _gcnt = {g: int((_gaoa == g).sum()) for g in _gd_state["groups"]}
            logger.info(f"  [{name}] GDROA ARMED (stage): eta={gdro_eta} "
                        f"groups(window counts)={_gcnt} beta=0.98 | per-sample weights "
                        f"exp(eta*EMA_g)/batch-mean, applied as w^2 on the squared error inside "
                        f"{loss_type} (== w on the per-sample loss)")
        _fa_live_stage = (fa_mode != "none" and name == "real-finetune")
        if _fa_live_stage:
            if _fa_sim["X"] is None:
                raise SystemExit(f"[{name}] CORALA: no sim-pretrain window stash -- the config "
                                 f"must run sim-pretrain before real-finetune")
            logger.info(f"  [{name}] CORALA ARMED (stage): feat_align_w={fa_w} | sim pool "
                        f"{int(_fa_sim['X'].shape[0])} windows (the sim-pretrain stage's own "
                        f"fold-excluded Xtr)")
        # ---- WAVE 3a per-stage arming (both knobs no-op here when off) ----
        _ns_stage_nu = None
        if _ns_box is not None:
            _ns_box["nu"] = None       # never carry sim-stage state into another stage
            _ns_box["denorm"] = None
        if ns_w > 0.0 and name == "sim-pretrain":
            import h5py
            if len(src_stats) != 1:
                raise SystemExit(f"[{name}] PINOS: needs exactly 1 source (a single denorm), "
                                 f"got {len(src_stats)}")
            _nm_, _nss_ = src_stats[0]
            _nsm3 = np.asarray(_nm_, np.float64).ravel()[:3]
            _nss3 = np.asarray(_nss_, np.float64).ravel()[:3]
            if not (_nss3 > 0).all():
                raise SystemExit(f"[{name}] PINOS: stage stats have a non-positive std "
                                 f"{_nss3.tolist()} -- denormalisation undefined")
            _ns_box["denorm"] = (torch.tensor(_nsm3, dtype=torch.float32, device=dev),
                                 torch.tensor(_nss3, dtype=torch.float32, device=dev))
            # per-file nu = U95 * L / Re; U95 = 95th percentile of the time-mean u over live
            # pixels on the scored ::2 grid, frames ::5 (trom_advection_fov_budget convention)
            _nu_map = {}
            for _nf in _ns_files_for_nu:
                with h5py.File(_nf, "r") as _fh:
                    _uu = np.asarray(_fh["u"][::5, ::sub_s, ::sub_s], np.float64)
                    _vv = np.asarray(_fh["v"][::5, ::sub_s, ::sub_s], np.float64)
                    _re3, _aoa3 = int(_fh["re"][()]), int(_fh["aoa"][()])
                _lv3 = ~((np.abs(_uu).max(0) == 0) & (np.abs(_vv).max(0) == 0))
                _u95 = float(np.percentile(_uu.mean(0)[_lv3], 95))
                _nu_map[(_re3, _aoa3)] = _u95 * _NS_L / float(_re3)
            _wre3 = np.concatenate(win_re_srcs) if win_re_srcs else np.zeros(0, np.int64)
            _waoa3 = np.concatenate(win_aoa_srcs) if win_aoa_srcs else np.zeros(0, np.int64)
            assert _wre3.shape[0] == n and _waoa3.shape[0] == n, (_wre3.shape, _waoa3.shape, n)
            _miss3 = sorted(set(zip(_wre3.tolist(), _waoa3.tolist())) - set(_nu_map))
            if _miss3:
                raise SystemExit(f"[{name}] PINOS: no nu for window key(s) {_miss3[:3]} -- "
                                 f"the nu map must cover every (Re, AoA) the windows carry")
            _ns_stage_nu = np.asarray([_nu_map[(int(_r3), int(_a3))]
                                       for _r3, _a3 in zip(_wre3, _waoa3)], np.float64)
            _nuv3 = np.asarray(sorted(_nu_map.values()))
            logger.info(f"  [{name}] PINOS ARMED (stage): {len(_nu_map)} files, nu range "
                        f"[{_nuv3[0]:.3e}, {_nuv3[-1]:.3e}] m^2/s (L={_NS_L} m, U95 "
                        f"convention), dt={_NS_DT_SIM:.6f} dx={_NS_DX:.4e} "
                        f"sigma_p={_nss3[2]:.4f} | max diffusion no. nu*dt/dx^2 = "
                        f"{_nuv3[-1] * _NS_DT_SIM / _NS_DX ** 2:.3e}")
        _wv_stage_nu = None
        if _wv_box is not None:
            _wv_box["nu"] = None          # never carry another stage's state
            _wv_box["denorm"] = None
            _wv_box["K"] = None
        if _ds_box is not None:
            _ds_box["denorm"] = None
            if name == "real-finetune":
                if len(src_stats) != 1:
                    raise SystemExit(f"[{name}] DIVSTAT: needs exactly 1 source (a single denorm), got {len(src_stats)}")
                _dm_, _dss_ = src_stats[0]
                _dsm2 = np.asarray(_dm_, np.float64).ravel()[:2]; _dss2 = np.asarray(_dss_, np.float64).ravel()[:2]
                if not (_dss2 > 0).all():
                    raise SystemExit(f"[{name}] DIVSTAT: non-positive std {_dss2.tolist()}")
                _ds_box["denorm"] = (torch.tensor(_dsm2, dtype=torch.float32, device=dev),
                                     torch.tensor(_dss2, dtype=torch.float32, device=dev))
                logger.info(f"  [{name}] DIVSTAT ARMED (stage): denorm mean={_dsm2.tolist()} std={_dss2.tolist()}")
        if wv_w > 0.0 and name == "real-finetune":
            import h5py
            if len(src_stats) != 1:
                raise SystemExit(f"[{name}] WVORT: needs exactly 1 source (a single denorm), "
                                 f"got {len(src_stats)}")
            _wm_, _wss_ = src_stats[0]
            _wvm2 = np.asarray(_wm_, np.float64).ravel()[:2]
            _wvs2 = np.asarray(_wss_, np.float64).ravel()[:2]
            if not (_wvs2 > 0).all():
                raise SystemExit(f"[{name}] WVORT: stage stats have a non-positive std "
                                 f"{_wvs2.tolist()} -- denormalisation undefined")
            _wv_box["denorm"] = (torch.tensor(_wvm2, dtype=torch.float32, device=dev),
                                 torch.tensor(_wvs2, dtype=torch.float32, device=dev))
            _wnu_map = {}
            for _wf in _ns_files_for_nu:
                with h5py.File(_wf, "r") as _fh:
                    _wu = np.asarray(_fh["u"][::5, ::sub_s, ::sub_s], np.float64)
                    _wv2 = np.asarray(_fh["v"][::5, ::sub_s, ::sub_s], np.float64)
                    _wre_attr = float(np.asarray(_fh["re"]).ravel()[0])
                # KEY off the FILENAME, VALUE off the h5 attribute.  On the REAL pool the two
                # DISAGREE -- `train_real/11400_0.h5` carries `re = 11420` (a standing
                # data rule) -- while `LOADED_WIN_RE` / `LOADED_WIN_AOA` are FILENAME-derived
                # (train_v4.py:187).  PINOS's nu map keys off the ATTRIBUTE and is correct only
                # because sim's attribute equals its filename; keying that way here misses EVERY
                # window and the arming guard refuses (measured 2026-08-31).  The true Reynolds
                # number is the attribute, so that is what divides; the label is what indexes.
                _wkey = tuple(int(_z) for _z in Path(_wf).stem.split("_")[:2])
                _wlv = ~((np.abs(_wu).max(0) == 0) & (np.abs(_wv2).max(0) == 0))
                _wu95 = float(np.percentile(_wu.mean(0)[_wlv], 95))
                _wnu_map[_wkey] = _wu95 * _WV_LCHORD / _wre_attr
            _wwre = np.concatenate(win_re_srcs) if win_re_srcs else np.zeros(0, np.int64)
            _wwaoa = np.concatenate(win_aoa_srcs) if win_aoa_srcs else np.zeros(0, np.int64)
            assert _wwre.shape[0] == n and _wwaoa.shape[0] == n, (_wwre.shape, _wwaoa.shape, n)
            _wmiss = sorted(set(zip(_wwre.tolist(), _wwaoa.tolist())) - set(_wnu_map))
            if _wmiss:
                raise SystemExit(f"[{name}] WVORT: no nu for window key(s) {_wmiss[:3]} -- the nu "
                                 f"map must cover every (Re, AoA) the windows carry")
            _wv_stage_nu = np.asarray([_wnu_map[(int(_r), int(_a))]
                                       for _r, _a in zip(_wwre, _wwaoa)], np.float64)
            _wv_box["K"] = _wvc.WeakVort(_WV_L[0], _WV_L[1], _WV_L[2], _WV_DX, _WV_DY, _WV_DT,
                                         dev, torch.float32)
            _wnuv = np.asarray(sorted(_wnu_map.values()))
            logger.info(f"  [{name}] WVORT ARMED (stage): {len(_wnu_map)} files, nu range "
                        f"[{_wnuv[0]:.3e}, {_wnuv[-1]:.3e}] m^2/s (L={_WV_LCHORD} m, U95 "
                        f"convention), phi half-widths {_WV_L}, support "
                        f"{2 * _WV_L[0] + 1}x{2 * _WV_L[1] + 1}x{2 * _WV_L[2] + 1} cells, "
                        f"centres per window "
                        f"{(t_out - 2 * _WV_L[2])}x{(32 - 2 * _WV_L[1])}x{(64 - 2 * _WV_L[0])}, "
                        f"denorm mean={_wvm2.tolist()} std={_wvs2.tolist()}")
        elif wv_w > 0.0:
            logger.info(f"  [{name}] WVORT OFF for this stage (real-finetune only)")
        if _hero_state is not None and name == "real-finetune":
            _hero_state["ema"] = None
            logger.info(f"  [{name}] HEROL ARMED (stage): hero_w={hero_w} E reset (beta=0.99); "
                        f"weight always from PREVIOUS steps' EMA (step 0 seeds, applies no term)")
        roll_from = int(steps * unroll_warmup) if st_roll > 1 else steps + 1
        logger.info(f"[{name}] {n} train windows | steps={steps} lr={lr0} "
                    f"objective={objective} "
                    f"n_roll={st_roll} unroll_from_step={roll_from if st_roll>1 else 'never'}")
        if objective == "causal":
            logger.info(f"  [{name}] CAUSAL next-frame pretrain: slot t predicts frame t+1, "
                        f"{t_in} supervised slots/sample vs 1 block for the block objective")
        for g in opt.param_groups:
            g["lr"] = lr0

        _gstat = [0, 0.0, 0]          # GNORM: [n_steps, max_norm, n_clipped]
        for s_i in range(steps):
            if _dws is not None and name == _dws["stage"]:   # DWSCHED knob (2026-09-10): this step's distillation weight (no RNG, no tensor)
                distill_w = _dws["start"] + (_dws["end"] - _dws["start"]) * s_i / (steps - 1)
                _dws_seen["n"] += 1
                _dws_seen["sum"] += distill_w
                _dws_seen["last"] = distill_w
                if _dws_seen["first"] is None:
                    _dws_seen["first"] = distill_w
                if s_i in (0, steps // 4, steps // 2, (3 * steps) // 4, steps - 1):
                    logger.info(f"DWSCHED STEP: [{name}] s_i={s_i} of {steps} gstep={gstep} w={distill_w!r}")
            # LRSCH (2026-08-26): the schedule's warmup fraction and its floor were
            # HARDCODED defaults, so neither was reachable from a config and neither
            # had ever been measured.  Defaults below reproduce cosine_lr's own
            # signature exactly, so an unset config is BIT-IDENTICAL.
            lr = cosine_lr(s_i, steps, lr0, warmup=_LR_WARMUP, min_ratio=_LR_MINR)
            # LP phase: head only.  lr=0 on the backbone group freezes it completely, since
            # AdamW's decoupled weight decay is itself scaled by lr.
            _lp = s_i < lp_steps
            for g in opt.param_groups:
                g["lr"] = lr if (g.get("is_head") or not _lp) else 0.0
            if _rw_probs is not None:
                # WRa: tilted draw.  Replaces (not augments) the uniform draw, so the arm's
                # RNG stream necessarily differs from the control's -- inherent to the arm.
                idx = torch.multinomial(_rw_probs, min(bs, n), replacement=True)
                if s_i == 0:
                    logger.info(f"  [{name}] WRa step-0 TRAINING batch realized "
                                f"top-quartile-Re fraction: "
                                f"{float((_rw_rn[idx.numpy()] >= 0.75).mean()):.4f} "
                                f"(batch={int(idx.numel())})")
            elif _sw_probs is not None:   # CLUSTERW knob (2026-09-10): state-cluster draw REPLACES the uniform draw
                idx = torch.multinomial(_sw_probs, min(bs, n), replacement=True)   # CLUSTERW knob (2026-09-10)
                if s_i == 0:   # CLUSTERW knob (2026-09-10)
                    logger.info(f"  [{name}] STATEW step-0 TRAINING batch cluster counts "   # CLUSTERW knob (2026-09-10)
                                f"{np.bincount(_sw_c[idx.numpy()], minlength=_sw_k).tolist()} (batch={int(idx.numel())})")   # CLUSTERW knob (2026-09-10)
            elif _sa_probs is not None:
                # AOSIM: tilted draw in the sim stage.  Replaces (not augments) the uniform
                # draw, so the arm's RNG stream differs from the control's -- inherent, as WRa.
                idx = torch.multinomial(_sa_probs, min(bs, n), replacement=True)
                if s_i == 0:
                    logger.info(f"  [{name}] AOSIM step-0 TRAINING batch realized hi-AoA "
                                f"fraction: {float(_sa_hi[idx.numpy()].mean()):.4f} "
                                f"(batch={int(idx.numel())})")
            else:
                idx = torch.randint(0, n, (min(bs, n),))
            xb = Xtr[idx].to(dev, non_blocking=True)
            yb = Ytr[idx].to(dev, non_blocking=True)
            ab = (Yaux_tr[idx].to(dev, non_blocking=True) if Yaux_tr is not None else None)
            if _sag_isaug_t is not None:
                _sag_box["m"] = _sag_isaug_t[idx].to(dev)      # STRIDEAUG: this batch's aug mask
            if _mimo:
                # MIMOA: pair each batch element with M-1 FURTHER uniformly-drawn windows
                # from the same stage pool (input repetition, if armed, makes a slot repeat
                # slot 0 instead).  The draws ride the PRIVATE generator, so the primary RNG
                # stream (and hence the control-matched minibatch sequence) is untouched.
                # Channel-stacked on the last axis; the model returns the M heads'
                # predictions stacked the same way, folded into the batch in step_loss.
                _mids = _mimo_pair_idx(idx, n, _mimo, _mimo_rep, _mimo_gen)
                xb = torch.cat([xb] + [Xtr[j].to(dev, non_blocking=True) for j in _mids], -1)
                yb = torch.cat([yb] + [Ytr[j].to(dev, non_blocking=True) for j in _mids], -1)
                if not _mimo_seen["live"]:
                    _mimo_seen["live"] = True
                    # same_idx_frac is the EXECUTION WITNESS for rho_rep: ~1/n when the knob
                    # is off, ~rho_rep when it is on.  An inert mimo_rep cannot fake it.
                    _sif = float(sum(float((idx == j).float().mean()) for j in _mids)
                                 / len(_mids))
                    logger.info(f"MIMOA LIVE: first paired batch -- xb={tuple(xb.shape)} "
                                f"yb={tuple(yb.shape)} slots={_mimo} rho_rep={_mimo_rep} "
                                f"idx[:4]={idx[:4].tolist()} "
                                f"idx2[:4]={_mids[0][:4].tolist()} "
                                f"same_idx_frac={_sif:.4f}")
            if _mix_a > 0.0:
                # MIXa: convex window mixing = regime-manifold interpolation (the
                # leaderboard axis IS Re-interpolation at seen angles).  lam pulled to
                # the dominant endpoint keeps the pair near-real; a REGULARIZER, not a
                # physics claim (NS is nonlinear).
                _lam = float(np.random.beta(_mix_a, _mix_a))
                _lam = max(_lam, 1.0 - _lam)
                _p = torch.randperm(xb.shape[0])
                xb = _lam * xb + (1.0 - _lam) * xb[_p]
                yb = _lam * yb + (1.0 - _lam) * yb[_p]
            if in_noise > 0.0 and (noise_stage == "all" or name == noise_stage):
                # PIV noise is isotropic in PHYSICAL units but the channels are normalised by very
                # different stds (u 0.0968, v 0.0160), so the normalised sigma is per-channel.
                # FLOOR is t1_noise_floor.py's measured noise level relative to the field rms;
                # RMS_PHYS is the field rms t1_input_noise.py measured per fold (0.1181-0.1427).
                _FLOOR, _RMS_PHYS = 0.0149, 0.13
                # `v_std` is mean_std_real.pt, the stats the model's OUTPUT is measured in and
                # the stats the real-finetune source itself uses.  A stage whose source has
                # different stats (sim-pretrain) gets a slightly mis-scaled sigma; that is
                # acceptable for an augmentation and exact on the stage that decides the score.
                _sig = (in_noise * _FLOOR * _RMS_PHYS) / torch.as_tensor(
                    v_std[:2], dtype=xb.dtype, device=xb.device)
                xb = xb.clone()
                if noise_mode == "add":
                    xb[..., :2] = xb[..., :2] + _sig * torch.randn_like(xb[..., :2])
                else:
                    # Proportional to the PHYSICAL value, so denormalise, scale, renormalise.  The
                    # relative sigma is set so the field-averaged noise ENERGY matches `add` at the
                    # same `input_noise`: sigma_rel = sigma_phys / RMS_PHYS = in_noise * FLOOR.
                    _m2 = torch.as_tensor(v_mean[:2], dtype=xb.dtype, device=xb.device)
                    _s2 = torch.as_tensor(v_std[:2], dtype=xb.dtype, device=xb.device)
                    _phys = xb[..., :2] * _s2 + _m2
                    _phys = _phys * (1.0 + (in_noise * _FLOOR) * torch.randn_like(_phys))
                    xb[..., :2] = (_phys - _m2) / _s2
            if _rot_deg > 0.0:
                import torch.nn.functional as _F
                _B = xb.shape[0]
                _th = (torch.rand(_B, device=xb.device) - 0.5) * (2.0 * _rot_deg * 3.141592653589793 / 180.0)
                _th = _th * (torch.rand(_B, device=xb.device) < 0.5)
                if bool((_th != 0).any()):
                    _c = torch.cos(_th).to(xb.dtype)
                    _s = torch.sin(_th).to(xb.dtype)
                    _mat = torch.zeros(_B, 2, 3, device=xb.device, dtype=xb.dtype)
                    # ASPECT CORRECTION (2026-08-31).  `affine_grid` works in NORMALISED [-1,1]^2
                    # coords, where x spans W pixels and y spans H pixels.  Writing a plain
                    # rotation matrix here realises the PHYSICAL map [[c, -a s],[s/a, c]] with
                    # a = W/H = 2 on this 32x64 grid -- area-preserving (|det| = 1) but NOT
                    # orthogonal, i.e. an anisotropic shear, while the velocity VECTORS below are
                    # co-rotated by the full nominal theta.  Measured as-shipped: nominal 5 deg
                    # realised as 2.505 / 9.925 deg; nominal 20 deg as 10.314 / 36.052 deg.
                    # A true physical rotation needs x'_n = c x_n - s (H/W) y_n,
                    # y'_n = s (W/H) x_n + c y_n.  det is still 1.
                    # NOTE FOR THE RECORD: `ROa`'s recorded null (rot_aug_deg 2, dS3 -0.003,
                    # t = -0.22, 2/5) was therefore measured on a 1.00-3.97 deg anisotropic SHEAR
                    # with an inconsistent 2.00 deg vector rotation -- the verdict likely survives
                    # at that amplitude but the LABEL was wrong, and the 2026-08-06 closure's
                    # "rotation (already null in-repo as ROa)" over-claims: ROa was +/-2 deg
                    # JITTER, never canonicalisation.
                    _aH, _aW = float(xb.shape[2]), float(xb.shape[3])
                    _mat[:, 0, 0] = _c
                    _mat[:, 0, 1] = -_s * (_aH / _aW)
                    _mat[:, 1, 0] = _s * (_aW / _aH)
                    _mat[:, 1, 1] = _c
                    _cv = _c.view(-1, 1, 1, 1)
                    _sv = _s.view(-1, 1, 1, 1)

                    def _rot_block(zb):
                        _Z = zb.permute(0, 1, 4, 2, 3).reshape(_B, -1, zb.shape[2], zb.shape[3])
                        _g = _F.affine_grid(_mat, _Z.shape, align_corners=False)
                        _R = _F.grid_sample(_Z, _g, mode="bilinear", padding_mode="border",
                                            align_corners=False)
                        _R = _R.reshape(_B, zb.shape[1], zb.shape[4], zb.shape[2],
                                        zb.shape[3]).permute(0, 1, 3, 4, 2)
                        _o = zb.clone()
                        _o[..., 0] = _R[..., 0] * _cv + _R[..., 1] * _sv
                        _o[..., 1] = -_R[..., 0] * _sv + _R[..., 1] * _cv
                        if zb.shape[4] > 2:
                            _o[..., 2] = _R[..., 2]
                        return _o

                    xb = _rot_block(xb)
                    yb = _rot_block(yb)
            if _gd_stage_gid is not None:
                # GDROA: hand step_loss this batch's per-sample AoA group ids (pure recording).
                _gd_box["gid"] = _gd_stage_gid[idx.numpy()]
            if _ns_stage_nu is not None:
                # PINOS: hand step_loss this batch's per-sample nu (pure recording, no RNG).
                _ns_box["nu"] = torch.as_tensor(_ns_stage_nu[idx.numpy()],
                                                dtype=torch.float32,
                                                device=dev).reshape(-1, 1, 1, 1)
            if _wv_stage_nu is not None:
                # WVORT: hand step_loss this batch's per-sample nu (pure recording, no RNG).
                _wv_box["nu"] = torch.as_tensor(_wv_stage_nu[idx.numpy()],
                                                dtype=torch.float32, device=dev)
            if _fa_live_stage:
                _fa_box["capture"] = True     # capture the REAL forward's bottleneck in step_loss
            rolling = (st_roll > 1) and (s_i >= roll_from)
            if _tooth is not None and (_tooth["stage"] == "all" or name == _tooth["stage"]) \
                    and (_tooth["p"] >= 1.0 or float(torch.rand(1, generator=_tooth_gen)) < _tooth["p"]):
                # EQFREE TOOTH: one random tooth+halo box PER SAMPLE.  The model takes in_norm/cond
                # statistics on the FULL window and crops after them (unet_model.forward); the
                # teacher(s) get the same boxes; the target is cropped to the INNER tooth (halo
                # excluded from the loss).  Boxes ride a PRIVATE generator, so the primary
                # minibatch RNG stream matches the control exactly.
                assert objective == "block" and pretext != "mask", (objective, pretext)
                _tB = int(xb.shape[0])
                _ty = torch.randint(0, int(xb.shape[2]) - _tooth["hin"] + 1, (_tB,), generator=_tooth_gen)
                _tx = torch.randint(0, int(xb.shape[3]) - _tooth["win"] + 1, (_tB,), generator=_tooth_gen)
                _tbox = torch.stack([_ty, _tx], 1)
                _tset = {"boxes": _tbox, "hin": _tooth["hin"], "win": _tooth["win"], "halo": _tooth["halo"]}
                if _tooth["input"] == "full":
                    _tooth_box["boxes"] = _tbox          # ELW: the model is NOT cropped; step_loss cuts the tooth out
                    model._tooth = None
                    for _tm in _distill_models:
                        _tm._tooth = None
                else:
                    model._tooth = _tset
                    for _tm in _distill_models:
                        _tm._tooth = _tset
                _tooth_seen["n"] += 1     # the target stays FULL: step_loss embeds the tooth prediction into it
                if not _tooth_seen["live"]:
                    _tooth_seen["live"] = True
                    logger.info(f"TOOTH LIVE: [{name}] gstep={gstep} first cropped batch -- xb={tuple(xb.shape)} "
                                f"(full; cropped inside the model) yb={tuple(yb.shape)} (full; tooth embedded, full-window denominator) "
                                f"boxes[:4]={_tbox[:4].tolist()}")
            elif _tooth is not None:
                _tooth_box["boxes"] = None
                model._tooth = None
                for _tm in _distill_models:
                    _tm._tooth = None
            loss = (causal_loss(xb, yb) if objective == "causal"
                    else mae_loss(xb) if objective == "mae"
                    else cfo_loss(xb, yb) if objective == "cfo"
                    else mkp_loss(xb) if pretext == "mask"
                    else step_loss(xb, yb, rolling, aux=ab))
            if _tooth is not None:
                # cleared BEFORE anything else can run a forward (SAM is refused; val is full-field)
                _tooth_box["boxes"] = None
                model._tooth = None
                for _tm in _distill_models:
                    _tm._tooth = None
            if _fa_live_stage:
                # CORALA: CORAL feature alignment between this step's REAL bottleneck (captured
                # by the d2 hook during step_loss's own forward) and a fresh SIM batch drawn from
                # the sim-pretrain stage's stashed windows with a PRIVATE generator (the real-
                # batch RNG stream is untouched, so the minibatch sequence matches the control
                # exactly).  Features are spatially pooled to (B, C); CORAL = ||cov_s-cov_r||_F^2
                # / d^2 + ||mu_s-mu_r||^2 / d (normalised by feature count); gradient flows
                # through BOTH the real and the sim encoder passes; no label loss on sim.
                _fa_box["capture"] = False
                _far = _fa_box["feat"]
                _fa_box["feat"] = None
                assert _far is not None, "CORALA: no bottleneck captured on the real forward"
                _fsN = int(_fa_sim["X"].shape[0])
                _fsi = torch.randint(0, _fsN, (min(bs, _fsN),), generator=_fa_gen)
                _fsx = _fa_sim["X"][_fsi].to(dev, non_blocking=True)
                _fa_box["capture"] = True
                model(_fsx)
                _fa_box["capture"] = False
                _fas = _fa_box["feat"]
                _fa_box["feat"] = None
                assert _fas is not None, "CORALA: no bottleneck captured on the sim forward"
                _frp = _far.mean(dim=(2, 3))
                _fsp = _fas.mean(dim=(2, 3))
                _fdc = int(_frp.shape[1])
                _fmr = _frp.mean(0)
                _fms = _fsp.mean(0)
                _fcr = (_frp - _fmr).T @ (_frp - _fmr) / max(_frp.shape[0] - 1, 1)
                _fcs = (_fsp - _fms).T @ (_fsp - _fms) / max(_fsp.shape[0] - 1, 1)
                _fcoral = ((_fcs - _fcr) ** 2).sum() / float(_fdc * _fdc) \
                    + ((_fms - _fmr) ** 2).sum() / float(_fdc)
                loss = loss + fa_w * _fcoral
                if not _fa_seen["live"]:
                    _fa_seen["live"] = True
                    logger.info(f"CORALA LIVE: first armed step -- d={_fdc} "
                                f"coral={float(_fcoral):.4e} "
                                f"weighted={float(fa_w * _fcoral):.4e} "
                                f"sim_batch={int(_fsx.shape[0])} l_after={float(loss):.4e}")
            if pretext == "mask" and s_i == 0:
                logger.info(f"  [{name}] MKP mask coverage (step 0): "
                            f"{_mkp_stat['cov']:.4f} (target ~0.50)")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if sam_rho > 0.0 and name == "real-finetune":
                # SAMa: two-step Sharpness-Aware Minimization.  Sits BEFORE the FGP/PhysGuard
                # gradient hooks and grad_clip so all of those act on the gradients the step
                # actually consumes (the perturbed-point ones); ema_update below runs AFTER
                # restore + opt.step(), i.e. always on the real weights.  Restore goes through
                # clones + copy_ because fp32 add-then-subtract is NOT bit-exact.  The forward
                # is deterministic in this recipe (no dropout / droppath), so the extra pass
                # draws no RNG and the batch sequence stays identical to the base run.
                with torch.no_grad():
                    _sam_sq = None
                    for _sp in model.parameters():
                        if _sp.grad is not None:
                            _q = (_sp.grad.detach().float() ** 2).sum()
                            _sam_sq = _q if _sam_sq is None else _sam_sq + _q
                    _sam_gn = _sam_sq.sqrt().clamp_min(1e-12)
                    _sam_orig = {}
                    for _sn, _sp in model.named_parameters():
                        if _sp.grad is None:
                            continue
                        _sam_orig[_sn] = _sp.detach().clone()
                        _sp.add_(_sp.grad.detach() * (sam_rho / _sam_gn).to(_sp.dtype))
                _sam_l2 = step_loss(xb, yb, rolling, aux=ab)
                opt.zero_grad(set_to_none=True)
                _sam_l2.backward()
                with torch.no_grad():
                    _sam_maxd = None
                    for _sn, _sp in model.named_parameters():
                        _o = _sam_orig.get(_sn)
                        if _o is None:
                            continue
                        _sp.copy_(_o)
                        if not _sam_state["checked"]:
                            _d = (_sp.detach() - _o).abs().max()
                            _sam_maxd = _d if _sam_maxd is None else torch.maximum(_sam_maxd, _d)
                if not _sam_state["checked"]:
                    # one-time self-test on the FIRST real SAM step: both losses finite, and
                    # the post-restore weights bit-identical to the pre-perturbation clones.
                    _l1v, _l2v = float(loss.detach()), float(_sam_l2.detach())
                    assert math.isfinite(_l1v) and math.isfinite(_l2v), \
                        f"SAM selftest: non-finite loss (clean={_l1v}, perturbed={_l2v})"
                    assert float(_sam_maxd) == 0.0, \
                        f"SAM selftest: restore not bit-exact (max|d|={float(_sam_maxd):.3e})"
                    logger.info(f"  [{name}] SAM selftest OK @ step {s_i}: loss_clean={_l1v:.5f} "
                                f"loss_perturbed={_l2v:.5f} ||g||={float(_sam_gn):.4e} "
                                f"restore max|d|=0.0 (batch={xb.shape[0]})")
                    _sam_state["checked"] = True
                del _sam_orig
            if _fp_mask:
                # FGP phase 2: hard-freeze the protected coordinates -- active only once the
                # mask exists, i.e. every stage after sim-pretrain.
                with torch.no_grad():
                    for _fn, _fp in model.named_parameters():
                        _fm = _fp_mask.get(_fn)
                        if _fm is not None and _fp.grad is not None:
                            _fp.grad.mul_(_fm)
            if _pg_u:
                # PhysGuard phase 2: g <- g - alpha * U (U^T g) on every protected layer;
                # active only once the subspace exists, i.e. the post-sim stages.
                _pga = float(cfg.get("pg_alpha", 1.0))
                with torch.no_grad():
                    for _pn, _pp in model.named_parameters():
                        _pu = _pg_u.get(_pn)
                        if _pu is not None and _pp.grad is not None:
                            _pg = _pp.grad.reshape(-1)
                            _pg.sub_(_pu @ (_pu.T @ _pg), alpha=_pga)
            # GNORM (2026-08-26): clip_grad_norm_ ALREADY computes the pre-clip total norm and
            # this call was discarding it.  Recording it costs nothing, changes no numerics, and
            # answers for free the question a grad_clip ARM would have spent 2 GPU-hours on:
            # does the clip ever bind?  Logged at the eval cadence as a running max + fire rate.
            if _cob:
                # COBLEND: clip each member's gradient norm separately, so a member sees exactly
                # the clip the standalone recipe applies to it (a joint norm would couple the two
                # members' step sizes).  Logged value = the larger of the two pre-clip norms.
                _gn = max(float(torch.nn.utils.clip_grad_norm_(_cm.parameters(), clip))
                          for _cm in model.members)
            else:
                _gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
            _gstat[0] += 1
            _gstat[1] = max(_gstat[1], _gn)
            _gstat[2] += 1 if _gn > clip else 0
            if _muon is not None and (not muon_stage or muon_stage in name):
                # ride the live schedule: AdamW's backbone group carries the scheduled lr, so
                # Muon tracks it proportionally and inherits warmup/cosine for free.
                # `lr` is the SCHEDULED value and `lr0` the stage base (see cosine_lr at the
                # top of the loop), so lr/lr0 is the schedule fraction -- comparing the scheduled
                # lr against itself would have pinned Muon at a constant rate with no warmup.
                _muon.lr = muon_lr * (lr / max(lr0, 1e-12))
                _muon.step()
            opt.step()
            ema_update(ema_decay if gstep > 200 else 0.9)
            if switch_ema_every > 0 and gstep > 200 and gstep % switch_ema_every == 0:
                # Switch EMA (arXiv 2402.09240): restart training from the flatter averaged point.
                # The EMA buffer itself is untouched; only the live weights jump.  Optimiser moments
                # are deliberately kept (that is the method as published).
                model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in ema.items()})
            tl = float(loss.detach().cpu())
            gstep += 1
            if s_i % eval_every == 0 or s_i == steps - 1:
                if _gstat[0]:
                    logger.info(f"  [{name}] GNORM n={_gstat[0]} max={_gstat[1]:.3f} "
                                f"clip={clip} fired={_gstat[2]}/{_gstat[0]} "
                                f"({100.0*_gstat[2]/_gstat[0]:.1f}%)")
                raw, eva, rroll = val_eval()
                if _au_armed and not _au_seen["live"]:
                    _ast = _au_gate_stats()
                    _amax = max((v["dev"] for v in _ast.values()), default=-1.0)
                    if not _au_seen["eval0"]:
                        _au_seen["eval0"] = True
                        logger.info(f"ATTNUNET eval0: [{name}] gstep={gstep} n_gates={len(_ast)} max_dev={_amax:.3e}")
                    if _amax > 0.0:
                        _au_seen["live"] = True
                        _k0 = sorted(_ast, key=lambda k: -_ast[k]["dev"])[:3]
                        logger.info(f"ATTNUNET LIVE: first eval with a gate off its step-0 constant -- [{name}] "
                                    f"gstep={gstep} max_dev={_amax:.3e} top={ {k: _ast[k] for k in _k0} }")
                if _cf:
                    # CONDFILM LIVE witness: the val pass's last forward (EMA weights) stashed
                    # gamma/beta per site; report the first eval, and the first eval with
                    # max|gamma| > 0 (warmup may keep step-0 gamma at exactly zero).
                    _lf = getattr(model, "_last_film", None) or {}
                    _gm = {k: float(v[0].abs().max()) for k, v in _lf.items()}
                    _bm = {k: float(v[1].abs().max()) for k, v in _lf.items()}
                    if not _cf_seen["eval0"]:
                        _cf_seen["eval0"] = True
                        logger.info(f"CONDFILM EVAL0: [{name}] gstep={gstep} max|gamma|={_gm} max|beta|={_bm}")
                    if not _cf_seen["live"] and _gm and max(_gm.values()) > 0.0:
                        _cf_seen["live"] = True
                        logger.info(f"CONDFILM LIVE: first eval with max|gamma|>0 -- [{name}] gstep={gstep} "
                                    f"max|gamma|={_gm} max|beta|={_bm}")
                evals.append({"stage": name, "gstep": gstep, "step": s_i, "lr": lr, "train_loss": tl,
                              "rolling": bool(rolling),
                              "val_mse": raw["mse"], "val_rel_l2": raw["rel"],
                              "val_tke_rel": raw["tke"], "val_mvpe_rel": raw["mvpe"],
                              "val_mse_ema": eva["mse"], "val_rel_l2_ema": eva["rel"],
                              "val_tke_rel_ema": eva["tke"], "val_mvpe_rel_ema": eva["mvpe"],
                              "val_rel_l2_rollout": rroll})
                logger.info(f"  [{name}] {s_i:05d}/{steps} lr={lr:.2e} train={tl:.5f}"
                            f"{' [ROLL]' if rolling else ''} | val_relL2 raw={raw['rel']:.5f} "
                            f"ema={eva['rel']:.5f} | tke raw={raw['tke']:.5f} ema={eva['tke']:.5f}"
                            f" | mvpe raw={raw['mvpe']:.5f} ema={eva['mvpe']:.5f}"
                            + (f" | roll{n_roll}={rroll:.5f}" if Xvr is not None else ""))
                maybe_save(name, gstep, raw, eva)
        if bool(cfg.get("physguard", False)) and name == "sim-pretrain" and not _pg_u:
            _pg_u.update(physguard_build(
                model, Xtr, Ytr, lch, cfg.get("pg_n", 256), cfg.get("pg_tau", 0.90),
                cfg.get("pg_max_dim", 500), cfg.get("pg_min_numel", 1000),
                cfg.get("pg_alpha", 1.0), logger))
        if fisher_q > 0.0 and name == "sim-pretrain" and not _fp_mask:
            # FGP phase 1: the Fisher pass uses the stage's OWN step loss (single-step block
            # objective with whatever auxiliary terms this config arms), at the end-of-stage
            # live weights, on the stage's own window pool -- same hook point as physguard.
            _fp_mask.update(fisher_diag_build(
                model,
                lambda _ix: step_loss(Xtr[_ix].to(dev, non_blocking=True),
                                      Ytr[_ix].to(dev, non_blocking=True), False),
                Xtr.shape[0], bs, fisher_nb, fisher_q,
                cfg.get("pg_min_numel", 1000), logger))
            # Freeze harder: wipe the AdamW first moment on the protected coordinates ONCE, so
            # the first fine-tune steps cannot move them on decayed sim-stage momentum either.
            # (AdamW's decoupled weight decay still shrinks them by lr*wd per step -- at
            # 5e-4 * 1e-5 that is a drift of ~5e-9/step, recorded, not fought.)
            with torch.no_grad():
                for _fn, _fp in model.named_parameters():
                    _fm = _fp_mask.get(_fn)
                    _os = opt.state.get(_fp)
                    if _fm is not None and _os and "exp_avg" in _os:
                        _os["exp_avg"].mul_(_fm)
        if fa_mode != "none" and name == "sim-pretrain":
            # CORALA: keep the INPUT windows (X only, Y freed) for the finetune stage's sim draws
            _fa_sim["X"] = Xtr
            logger.info(f"  [{name}] CORALA: stashed {int(Xtr.shape[0])} sim input windows "
                        f"({Xtr.numel() * 4 / 1e9:.2f} GB CPU) for real-finetune")
        if _tooth is not None:   # TOOTHPRE knob (2026-09-10): per-stage cropped-batch count (log only; no RNG, no tensor)
            logger.info(f"TOOTH STAGE: [{name}] cropped_batches={int(_tooth_seen['n']) - int(_tooth_seen.get('stage_n0', 0))} "
                        f"of {steps} steps (tooth stage={_tooth['stage']} p={_tooth['p']} input={_tooth['input']})")
        if _dws is not None and name == _dws["stage"]:   # DWSCHED knob (2026-09-10): stage summary, then the constant is restored
            logger.info(f"DWSCHED STAGE: [{name}] applied={_dws_seen['n']} of {steps} steps w_first={_dws_seen['first']!r} "
                        f"w_last={_dws_seen['last']!r} w_mean={(_dws_seen['sum'] / max(_dws_seen['n'], 1))!r} "
                        f"(constant distill_weight={_dws['const']!r} restored)")
            distill_w = _dws["const"]
        del Xtr, Ytr, Yaux_tr

    if _au_armed and not _au_seen["live"]:
        raise SystemExit("ATTNUNET: armed but no gate ever left its step-0 constant at any eval -- the "
                         "pathway is not live; refusing to save a checkpoint that is the control in disguise")
    if _au_armed:
        _ast = _au_gate_stats()
        logger.info(f"ATTNUNET FINAL gate stats (last val pass): { {k: {q: round(w, 5) for q, w in v.items()} for k, v in _ast.items()} }")
    if _cf and not _cf_seen["live"]:
        raise SystemExit("CONDFILM: armed but max|gamma| never left zero at any eval -- the pathway "
                         "is not live; refusing to save a checkpoint that is the control in disguise")
    if _dfl_alpha != 1.0:
        _dfl_steps = sum(int(q["steps"]) for q in cfg["stages"] if q.get("name") == "real-finetune")
        if _dfl_seen["calls"] != _dfl_steps:
            raise RuntimeError(f"DFLUCT expected {_dfl_steps} real-stage applications, got {_dfl_seen['calls']}")
        logger.info(f"DFLUCT FINAL: applied={_dfl_seen['calls']} alpha={_dfl_alpha} real_only=True")
    if bool(cfg.get("save_terminal_both", False)):
        # NO82: retain both fixed terminal policies. Validation is measured, never used to select one.
        sys.path.insert(0, str(CASE_DIR.parent.parent / "common"))
        from terminal_checkpoint import save_terminal_pair
        _traw, _tema, _ = val_eval()
        _terminal_paths = save_terminal_pair(
            model, ema, _ckpt_meta("raw", "final", gstep, _traw),
            _ckpt_meta("ema", "final", gstep, _tema), ckpt_out)
        logger.info(f"FIXED_TERMINAL_BOTH gstep={gstep} paths={_terminal_paths}")
    if bool(cfg.get("save_last", False)):
        # unbiased companion to the best-on-val checkpoint: the FINAL weights, chosen by nothing
        raw, eva, _ = val_eval()
        which, m = ("ema", eva) if eva["rel"] <= raw["rel"] else ("raw", raw)
        sd = ({k: v.to(model.state_dict()[k].dtype) for k, v in ema.items()}
              if which == "ema" else {k: v.detach().clone() for k, v in model.state_dict().items()})
        last_out = str(Path(ckpt_out).with_name(Path(ckpt_out).stem + "_last" + Path(ckpt_out).suffix))
        meta = _ckpt_meta(which, "final", gstep, m)
        meta["selection"] = "last step (no model selection)"
        torch.save({"model_state_dict": {k: v.cpu() for k, v in sd.items()}, "meta": meta}, last_out)
        logger.info(f"save_last: rel={m['rel']:.5f} tke={m['tke']:.5f} mvpe={m['mvpe']:.5f} "
                    f"({which}) -> {last_out}")

    (run_dir / "metrics.json").write_text(json.dumps(
        {"variant": variant, "base": base, "params": int(n_params), "stages": cfg["stages"],
         "loss_type": loss_type, "l2_normalize": l2_norm, "tke_weight": tke_w, "dt1_weight": dt1_w, "dt2_weight": dt2_w,
         "lead_weight": lead_weight,
         "mvpe_weight": mvpe_w,
         "fold": fold, "group_by": group_by, "val_groups": fold_val_re,
         "kfold_manifest": cfg.get("kfold_manifest") if fold is not None else None,
         "n_roll": n_roll, "unroll_mode": unroll_mode, "unroll_warmup": unroll_warmup,
         "unroll_stride": roll_stride,
         "n_val_windows": int(Xva.shape[0]), "val_files": [Path(f).name for f in val_files],
         "best": best, "evals": evals, "checkpoint": ckpt_out}, indent=2))

    if evals:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        gs = [e["gstep"] for e in evals]
        fig, ax = plt.subplots(1, 3, figsize=(19, 4.5))
        ax[0].plot(gs, [e["train_loss"] for e in evals], ".", color="navy", alpha=.5,
                   label=f"train {loss_type}")
        ax[0].plot(gs, [e["val_mse"] for e in evals], "-", color="crimson", label="val MSE (raw)")
        ax[0].plot(gs, [e["val_mse_ema"] for e in evals], "-", color="darkorange", label="val MSE (EMA)")
        ax[0].set_yscale("log"); ax[0].set_xlabel("global step"); ax[0].grid(alpha=.3, which="both")
        ax[0].legend(); ax[0].set_title(f"{variant} - overfitting monitor")
        ax[1].plot(gs, [e["val_rel_l2"] for e in evals], "-", color="teal", label="val rel-L2 (raw)")
        ax[1].plot(gs, [e["val_rel_l2_ema"] for e in evals], "-", color="purple", label="val rel-L2 (EMA)")
        if Xvr is not None:
            ax[1].plot(gs, [e["val_rel_l2_rollout"] for e in evals], "--", color="grey",
                       label=f"rollout x{n_roll} (diag)")
        ax[1].axhline(best["rel"], ls="--", c="k", lw=.8, label=f"best {best['rel']:.4f} ({best['which']})")
        ax[1].set_xlabel("global step"); ax[1].grid(alpha=.3); ax[1].legend()
        ax[1].set_title("Held-out val rel-L2 (competition metric)")
        ax[2].plot(gs, [e["val_tke_rel"] for e in evals], "-", color="firebrick", label="val TKE rel-L2 (raw)")
        ax[2].plot(gs, [e["val_tke_rel_ema"] for e in evals], "-", color="seagreen", label="val TKE rel-L2 (EMA)")
        ax[2].plot(gs, [e["val_mvpe_rel"] for e in evals], "--", color="darkviolet", label="val MVPE rel-L2 (raw)")
        ax[2].plot(gs, [e["val_mvpe_rel_ema"] for e in evals], "--", color="olive", label="val MVPE rel-L2 (EMA)")
        ax[2].set_xlabel("global step"); ax[2].grid(alpha=.3); ax[2].legend()
        ax[2].set_title(f"Held-out val TKE / MVPE rel-L2 (mvpe_weight={mvpe_w})")
        fig.tight_layout(); fig.savefig(run_dir / "train_curve.png", dpi=130)
        logger.info(f"wrote {run_dir/'train_curve.png'}")

    logger.info(f"BEST val_rel_l2={best['rel']:.5f} tke={best['tke']} mvpe={best['mvpe']} "
                f"from {best['which']} weights @ {best['stage']} gstep {best['step']} -> {ckpt_out}")
    print("TRAIN_V4_OK", run_dir, json.dumps(best))


if __name__ == "__main__":
    main()
