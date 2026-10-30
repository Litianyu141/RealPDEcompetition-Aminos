"""RealPDE Track 2 LTTTA submission template.

Rename this file to ``submission.py``, put your checkpoint (``model.pth``) next
to it, and zip the archive with ``submission.py`` at the root.

What this template shows:
  * the ``get_ttt_model(submission_dir, device)`` entry point,
  * a **correct** prev-target adaptation loop in ``ttt_step`` (this is the
    reference implementation the Submission page refers to),
  * checkpoint restore on ``reset_ttt_state``.

The forecaster here is a tiny illustrative conv net so the template runs on CPU
with no checkpoint. Replace ``TinyForecaster`` with your architecture; keep the
``reset_ttt_state`` / ``ttt_step`` contract.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

# The base class is a convenience only; evaluation is duck-typed, so this import
# is optional. Fall back to nn.Module if ttt_model.py is not shipped in the zip.
try:
    from ttt_model import TTTModel
except Exception:  # pragma: no cover - base class is optional
    TTTModel = nn.Module  # type: ignore

IN_STEP = 20
CHANNELS = 3  # [u, v, p]


class TinyForecaster(nn.Module):
    """Minimal per-frame conv forecaster: (B, T, H, W, C) -> (B, T, H, W, C).

    A residual around a 2-layer conv, so an untrained net starts near identity
    (a persistence-like prior). Placeholder only -- swap in your own model.
    """

    def __init__(self, channels: int = CHANNELS, hidden: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h, w, c = x.shape
        z = x.reshape(b * t, h, w, c).permute(0, 3, 1, 2)  # (B*T, C, H, W)
        z = z + self.conv2(torch.relu(self.conv1(z)))       # residual
        return z.permute(0, 2, 3, 1).reshape(b, t, h, w, c)


class ReferenceTTTModel(TTTModel):
    """One-gradient-step test-time adaptation reference.

    Each ``ttt_step``:
      1. If ``prev_target_norm`` is given, take ONE SGD step supervising the
         **cached previous input** against the revealed **previous target**
         (a genuine input->target pair -- never the current target).
      2. Predict the current input with no gradients.
      3. Cache the current input for the next step.
    ``reset_ttt_state`` restores the checkpoint weights and clears the cache.
    """

    def __init__(self, base: nn.Module, device: str, ttt_lr: float = 1e-3):
        super().__init__()
        self.base = base.to(device)
        self.device = device
        self.ttt_lr = ttt_lr
        self._init_state = copy.deepcopy(self.base.state_dict())
        self._opt = torch.optim.SGD(self.base.parameters(), lr=ttt_lr)
        self._prev_input: Optional[torch.Tensor] = None

    def reset_ttt_state(self) -> None:
        self.base.load_state_dict(copy.deepcopy(self._init_state))
        self._opt = torch.optim.SGD(self.base.parameters(), lr=self.ttt_lr)
        self._prev_input = None

    def ttt_step(
        self,
        input_norm: torch.Tensor,
        prev_target_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        input_norm = torch.as_tensor(input_norm).to(self.device)

        # (1) Adapt on the PREVIOUS (input, target) pair, both fully revealed.
        adapt_loss: Optional[float] = None
        if prev_target_norm is not None and self._prev_input is not None:
            prev_target_norm = torch.as_tensor(prev_target_norm).to(self.device)
            self.base.train()
            self._opt.zero_grad()
            pred_prev = self.base(self._prev_input)
            loss = torch.mean((pred_prev - prev_target_norm) ** 2)
            loss.backward()
            self._opt.step()
            adapt_loss = float(loss.detach().cpu())

        # (2) Predict the current window -- no gradients, no current target.
        self.base.eval()
        with torch.no_grad():
            pred_norm = self.base(input_norm)

        # (3) Cache the current input for the next step's adaptation.
        self._prev_input = input_norm.detach()

        # Optional: also return SPS interval bounds. Uncomment to try it.
        # `lower_norm` / `upper_norm` must be the same shape as `pred_norm` and
        # live in the same normalized space -- the evaluator denormalizes them
        # with the same transform it applies to the prediction. Set both or
        # neither; a lone `lower` or `upper` key fails the run. This toy
        # example just widens a fixed fraction of the prediction magnitude,
        # close to the scorer's own default band; a real submission should
        # size the interval from an actual uncertainty estimate.
        #
        # half_width = 0.05 * pred_norm.detach().abs()
        # lower_norm = pred_norm.detach() - half_width
        # upper_norm = pred_norm.detach() + half_width
        # return pred_norm, {
        #     "adapt_loss": adapt_loss,
        #     "lower": lower_norm,
        #     "upper": upper_norm,
        # }

        return pred_norm, {"adapt_loss": adapt_loss}


def get_ttt_model(submission_dir: str, device: str):
    """Entry point called once by the evaluator (construction is not timed)."""
    base = TinyForecaster()

    # Load your checkpoint if present. If you packed it with pack_ckpt_fp16.py,
    # unpack here first (see that tool's docstring for the ``unpack_fp16`` helper).
    ckpt_path = os.path.join(submission_dir, "model.pth")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        base.load_state_dict(state)

    return ReferenceTTTModel(base, device)
