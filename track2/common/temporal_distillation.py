"""Prospective attenuation of only the temporal residual fluctuation in KD.

The caller supplies the existing loss callable and already-sliced normalized
u/v tensors. The callable keeps its channel numerator weights, target norm,
epsilon and reduction. This helper never computes or alters the hard loss,
the outer distillation coefficient, or a teacher prediction.

For D=P-Q, the active residual is mean_t(D)+sqrt(alpha)*(D-mean_t(D)).
Returning loss_fn(Q+D_active,Q) preserves the unchanged teacher denominator.
The residual mean is preserved; the square-root norm still couples mean and
fluctuation gradient magnitudes. This is not scalar KD-weight rescaling.

Integration must remain INSIDE the trainer's existing real-finetune-only KD
guard. Keep its literal legacy loss call for alpha1 where possible. The
helper also makes alpha1 an immediate exact callable passthrough, with no
tensor inspection, temporal operations, cast, clone, RNG or loss arithmetic.
"""
from __future__ import annotations

import math
import numbers


def temporal_distillation_loss(prediction_uv, teacher_uv, loss_fn, *, alpha=1.0):
    """Apply a fixed alpha in [0,1] to the squared fluctuation residual.

    Active alpha expects (B,T,H,W,2) floating tensors on one device/dtype,
    with an already-detached teacher. Physical channel weighting remains
    solely in loss_fn; no statistics or unit conversion occurs here.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, numbers.Real) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be a finite real number in [0,1]")
    if alpha == 1.0:
        return loss_fn(prediction_uv, teacher_uv)
    import torch
    if not isinstance(prediction_uv, torch.Tensor) or not isinstance(teacher_uv, torch.Tensor):
        raise ValueError("active temporal distillation requires tensors")
    if (prediction_uv.ndim != 5 or prediction_uv.shape != teacher_uv.shape
            or prediction_uv.shape[-1] != 2 or any(n == 0 for n in prediction_uv.shape)):
        raise ValueError("pass matching nonempty (B,T,H,W,2) u/v tensors")
    if (not prediction_uv.is_floating_point() or not teacher_uv.is_floating_point()
            or prediction_uv.dtype != teacher_uv.dtype or prediction_uv.device != teacher_uv.device):
        raise ValueError("prediction and teacher need the same floating dtype/device")
    if teacher_uv.requires_grad:
        raise ValueError("teacher must already be detached by the no_grad teacher forward")
    residual = prediction_uv - teacher_uv
    residual_mean = residual.mean(dim=1, keepdim=True)
    adjusted_residual = residual_mean + math.sqrt(alpha) * (residual - residual_mean)
    return loss_fn(teacher_uv + adjusted_residual, teacher_uv)
