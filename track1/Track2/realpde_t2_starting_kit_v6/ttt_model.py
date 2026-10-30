"""Convenience base class for RealPDE Track 2 LTTTA submissions.

The evaluator is **duck-typed**: it only checks that the object returned by
``get_ttt_model(submission_dir, device)`` exposes ``reset_ttt_state()`` and
``ttt_step(...)``. Subclassing ``TTTModel`` below is therefore optional; it is
provided so your model has a clear place for the two methods and so the shape
and timing contract is documented next to the code.

Streaming contract (see docs/interface.md for the full description):

- Every tensor is in the evaluator's normalized space and has shape
  ``(1, 20, 32, 64, 3)`` == ``(batch, in_step, H, W, [u, v, p])``.
- ``reset_ttt_state()`` is called once at the start of every trajectory and is
  **timed**, charged to the step that follows the trajectory boundary. Restore
  your checkpoint weights and clear adaptation state.
- ``ttt_step(input_norm, prev_target_norm)`` is called once per step and is
  **fully timed** (adaptation, controller logic, optional LLM round-trips, and
  the forward pass all count). ``prev_target_norm`` is the ground truth of the
  PREVIOUS step, or ``None`` on the first step of a trajectory. The current
  step's target is never passed in.
- ``ttt_step`` returns ``(pred_norm, info)``. The evaluator reads
  ``info["adapt_loss"]`` and, optionally, ``info["lower"]`` / ``info["upper"]``
  for the Safe Prediction Score; return both bounds or neither.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

# Concrete tensor shape seen at evaluation time (batch is always 1).
IN_STEP = 20
HEIGHT = 32
WIDTH = 64
CHANNELS = 3  # [u, v, p]; real data scores u, v only (p is zero-filled).
EXPECTED_SHAPE = (1, IN_STEP, HEIGHT, WIDTH, CHANNELS)


class TTTModel(nn.Module, ABC):
    """Base class for models that adapt at test time.

    Typical per-step structure implemented by ``ttt_step``:

    1. If ``prev_target_norm`` is not ``None``, adapt using your **cached
       previous input** and this revealed previous target (a genuine
       ``input -> target`` pair). Do not pair the previous target with the
       current input.
    2. Predict the current ``input_norm`` without using the current target.
    3. Cache the current input for the next step's adaptation.
    """

    @abstractmethod
    def reset_ttt_state(self) -> None:
        """Reset adaptation state at the start of a new trajectory.

        Restore the checkpoint weights and clear any cached tensors / optimizer
        state so each trajectory starts from the same base model. Timed, and
        charged to the step that follows the boundary.
        """

    @abstractmethod
    def ttt_step(
        self,
        input_norm: torch.Tensor,
        prev_target_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Adapt on the previous pair, then predict ``input_norm``.

        Parameters
        ----------
        input_norm : torch.Tensor
            Current input window, shape ``(1, 20, 32, 64, 3)``, normalized.
        prev_target_norm : torch.Tensor or None
            Ground truth of the PREVIOUS step (same shape), or ``None`` on the
            first step of a trajectory.

        Returns
        -------
        (pred_norm, info) : Tuple[torch.Tensor, dict]
            ``pred_norm`` has shape ``(1, 20, 32, 64, 3)``. The evaluator reads
            ``info["adapt_loss"]`` (a python ``float``, or ``None`` when no
            adaptation happened this step) and, optionally, ``info["lower"]`` /
            ``info["upper"]``: per-element Safe Prediction Score bounds in the
            same normalized space and shape as ``pred_norm``. Bounds are
            optional, but set both or neither, a lone ``lower`` or ``upper``
            key is invalid. Any other keys you add to ``info`` are ignored.
        """
