"""Attenuate only the SPATIALLY small-scale residual in knowledge distillation.

Sibling of `temporal_distillation.py`, and deliberately its spatial mirror, because on this task the
two axes have measured opposite value:

    the spatial high-pass L1 term is the ONLY knob that raises tke (+0.046..+0.063 local points,
    two seeds, four held-out Re lines positive) -- while the temporal high-pass term reads zero
    (+0.0014/+0.0019 on the calibrated population), and `distill_fluct_alpha`, which attenuates the
    TEMPORAL residual fluctuation, was measured negative at every dose and its family closed.

Meanwhile the distillation teachers are what suppresses tke in the first place: turning distillation
off entirely moves tke by +0.254 local points, five times anything else we have, but costs rel
-0.085, and at the measured marginal values that trade needs a tke/rel ratio above 7.3 where it
delivers 3.0.  So switching the teacher off is wrong; what we want is to keep the teacher where it
earns rel -- the large scales -- and stop it flattening the small scales the tke subscore measures.

For D = P - Q, split D by a k x k spatial box mean and shrink only the high-pass part:

    D_low  = boxmean_k(D)
    D_high = D - D_low
    D_active = D_low + sqrt(beta) * D_high

then score `loss_fn(Q + D_active, Q)`, which leaves the teacher denominator untouched exactly as the
temporal helper does.  beta = 1 is an exact, immediate passthrough to the caller's own loss call, so
a run that does not set it is bit-identical to the shipped recipe.  The sqrt matches the temporal
helper's convention: it is the squared residual that carries beta.

This is NOT a rescaling of the distillation coefficient -- that would shrink the large-scale pull
too, which is the part measured to be worth keeping.
"""
from __future__ import annotations

import math
import numbers


def spatial_distillation_loss(prediction_uv, teacher_uv, loss_fn, *, beta=1.0, kernel=3):
    """Apply a fixed beta in [0,1] to the spatially high-pass part of the residual.

    Active beta expects (B,T,H,W,2) floating tensors on one device/dtype, with an already-detached
    teacher.  Channel weighting stays entirely in loss_fn; nothing here converts units.
    """
    if isinstance(beta, bool) or not isinstance(beta, numbers.Real) or not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be a finite real number in [0,1]")
    if not isinstance(kernel, int) or isinstance(kernel, bool) or kernel < 3 or kernel % 2 == 0:
        raise ValueError("kernel must be an odd int >= 3")
    if beta == 1.0:
        return loss_fn(prediction_uv, teacher_uv)
    import torch
    if not isinstance(prediction_uv, torch.Tensor) or not isinstance(teacher_uv, torch.Tensor):
        raise ValueError("active spatial distillation requires tensors")
    if (prediction_uv.ndim != 5 or prediction_uv.shape != teacher_uv.shape
            or prediction_uv.shape[-1] != 2 or any(n == 0 for n in prediction_uv.shape)):
        raise ValueError("pass matching nonempty (B,T,H,W,2) u/v tensors")
    if (not prediction_uv.is_floating_point() or not teacher_uv.is_floating_point()
            or prediction_uv.dtype != teacher_uv.dtype or prediction_uv.device != teacher_uv.device):
        raise ValueError("prediction and teacher need the same floating dtype/device")
    if teacher_uv.requires_grad:
        raise ValueError("teacher must already be detached by the no_grad teacher forward")
    residual = prediction_uv - teacher_uv
    n, t, h, w, c = residual.shape
    pad = kernel // 2
    z = residual.permute(0, 1, 4, 2, 3).reshape(n * t * c, 1, h, w)
    low = torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(z, (pad, pad, pad, pad), mode="replicate"), kernel, stride=1)
    low = low.reshape(n, t, c, h, w).permute(0, 1, 3, 4, 2)
    adjusted_residual = low + math.sqrt(beta) * (residual - low)
    return loss_fn(teacher_uv + adjusted_residual, teacher_uv)
