#!/usr/bin/env python3
"""Frame-stacked 2D U-net baseline for RealPDE (20->20 forecasting).

Maps (B, T_in, H, W, C) -> (B, T_out, H, W, C) by stacking the T*C input frames
into a 2D image, running a small U-net, and unstacking. Same I/O contract as the
official CNO/FNO/Transolver baselines, so it drops into both the Track-1
teacher-forced rollout and the Track-2 ReferenceTTTModel wrapper unchanged.
This is a from-scratch baseline (no official checkpoint) — train it first.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


_FASTGN_MIN = 1 << 17      # fast-GN (common/fast_gn_card.py): one-group rows at least this long use the parallel path


class GroupNorm(nn.GroupNorm):
    """nn.GroupNorm with a parallel-reduction forward for one-group rows >= _FASTGN_MIN elements (fast-GN card patch).

    torch's group_norm reduces each (sample, group) row in ONE block, so GroupNorm(1, c) at batch 1-4 runs on 1-4 SMs.
    Here the row statistics come from torch.var_mean (grid-wide reduction) and the same folded affine
    y = x * (w * rstd) + (b - mean * w * rstd) is applied in float32.  Same parameters / state_dict / class name."""

    def forward(self, x):
        n = x.shape[0]
        if (self.num_groups != 1 or x.dim() != 4 or not x.is_cuda or self.weight is None
                or x.numel() < _FASTGN_MIN * max(n, 1)):
            return super().forward(x)
        c = x.shape[1]
        x32 = x.reshape(n, 1, -1)
        if x32.dtype != torch.float32:
            x32 = x32.float()
        var, mean = torch.var_mean(x32, dim=2, unbiased=False, keepdim=True)        # (n, 1, 1)
        scale = self.weight.float().view(1, c, 1) * torch.rsqrt(var + self.eps)     # (n, c, 1)
        shift = torch.addcmul(self.bias.float().view(1, c, 1), mean, scale, value=-1.0)
        y = torch.addcmul(shift, x32.view(n, c, -1), scale).view(x.shape)
        return y if y.dtype == x.dtype else y.to(x.dtype)


class DoubleConv(nn.Module):
    def __init__(self, ci, co, groups=8):
        super().__init__()
        g = groups if co % groups == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), _norm2d(g, co), nn.GELU(),
            nn.Conv2d(co, co, 3, padding=1), _norm2d(g, co), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class GatedDoubleConv(nn.Module):
    """DoubleConv with GEGLU in place of GELU (Shazeer 2020, arXiv:2002.05202).

    Each 3x3 conv emits 2*co channels; after GroupNorm the halves are split into a value v and a
    gate g, and the block returns v * GELU(g).  Keeping the norm BEFORE the split makes this exactly
    the historical `Conv -> GroupNorm -> nonlinearity` with one nonlinearity swapped, so the arm has
    a single variable.

    Why it might matter for this problem specifically: the gate is multiplicative, and the advective
    term of Navier-Stokes, (u . grad)u, is a product of fields.  A gate expresses a product in one
    layer; a pointwise activation has to approximate one.

    Costs 2x the conv parameters, so any arm using it must reduce `base` to stay parameter-matched
    against the shipped model -- otherwise the result is confounded with capacity, which is already
    known to be saturated here.
    """

    def __init__(self, ci, co, groups=8):
        super().__init__()
        g2 = groups if (2 * co) % groups == 0 else 1
        self.c1 = nn.Conv2d(ci, 2 * co, 3, padding=1)
        self.n1 = _norm2d(g2, 2 * co)
        self.c2 = nn.Conv2d(co, 2 * co, 3, padding=1)
        self.n2 = _norm2d(g2, 2 * co)
        self.act = nn.GELU()

    def _glu(self, y):
        v, g = y.chunk(2, dim=1)
        return v * self.act(g)

    def forward(self, x):
        x = self._glu(self.n1(self.c1(x)))
        return self._glu(self.n2(self.c2(x)))


def _block(kind):
    """DoubleConv factory. `gelu` is the historical block, byte-identical."""
    return {"gelu": DoubleConv, "geglu": GatedDoubleConv}[kind]


class BottleneckAttn(nn.Module):
    """Multi-head self-attention over the H/4 x W/4 bottleneck tokens, added residually.

    128 tokens at 32x64 input, so the quadratic term is 128^2 -- irrelevant beside the convolutions.
    Zero-initialised output projection, so an untrained block is exactly the identity and a run that
    fails to use it degrades to the plain U-net rather than to noise.
    """

    def __init__(self, ch, heads=4):
        super().__init__()
        self.h = heads if ch % heads == 0 else 1
        self.norm = GroupNorm(8 if ch % 8 == 0 else 1, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, hh, ww = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.h, c // self.h, hh * ww).unbind(1)
        a = torch.softmax(q.transpose(-2, -1) @ k / (c // self.h) ** 0.5, dim=-1)
        o = (v @ a.transpose(-2, -1)).reshape(b, c, hh, ww)
        return x + self.proj(o)


class SpectralBranch(nn.Module):
    """A Fourier layer in PARALLEL with the U-net, at full resolution, added residually.

    An independent complex weight per retained (kx, ky) mode -- no k^-2 preference, which is the
    point: the deficit being attacked is 43% of the fluctuation energy, and that energy is at high k.
    Width is reduced to `inner` before the transform so the complex weight stays small
    (modes_h*modes_w*inner^2 complex), and the output projection is zero-initialised so the branch
    starts as an exact no-op.
    """

    def __init__(self, ch, inner=32, modes_h=12, modes_w=17):
        super().__init__()
        self.mh, self.mw = modes_h, modes_w
        self.down = nn.Conv2d(ch, inner, 1)
        self.up = nn.Conv2d(inner, ch, 1)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
        s = 1.0 / (inner * inner)
        self.w = nn.Parameter(s * torch.randn(modes_h, modes_w, inner, inner, 2))

    def forward(self, x):
        z = self.down(x).float()
        b, c, hh, ww = z.shape
        f = torch.fft.rfft2(z.float(), norm="ortho")
        mh, mw = min(self.mh, f.shape[-2]), min(self.mw, f.shape[-1])
        # fp32 throughout: ComplexHalf has no baddbmm kernel, and the submission ships fp16,
        # so this branch would crash inside the timed region. Half-precision FFT is unreliable
        # anyway; the cast back to x.dtype at the end keeps the rest of the net in fp16.
        w = torch.view_as_complex(self.w[:mh, :mw].float().contiguous())
        out = torch.zeros_like(f)
        # (mh, mw, ci, co) x (b, ci, mh, mw) -> (b, co, mh, mw); the low block and its conjugate
        out[..., :mh, :mw] = torch.einsum("bihw,hwio->bohw", f[..., :mh, :mw], w)
        if mh < f.shape[-2]:
            out[..., -mh:, :mw] = torch.einsum("bihw,hwio->bohw", f[..., -mh:, :mw], w)
        y = torch.fft.irfft2(out, s=(hh, ww), norm="ortho").to(x.dtype)
        return self.up(y)


# ---- discrete differential operators, kernels verbatim from Wandel et al. ICLR'21 -------------
# `derivatives.py` of github.com/vc-bonn/Unsupervised_Deep_Learning_of_Incompressible_Fluid_Dynamics
# The one-sided stencils are what make div(rot_mac(a)) vanish EXACTLY on a MAC grid, which is the
# whole reason to parametrise velocity as a curl rather than penalise its divergence.
_DX_RIGHT = torch.tensor([0.0, -1.0, 1.0]).view(1, 1, 1, 3)
_DY_BOTTOM = torch.tensor([0.0, -1.0, 1.0]).view(1, 1, 3, 1)


def rot_mac(a):
    """(B,1,H,W) stream function -> (B,2,H,W) velocity, divergence-free by construction."""
    kx = _DX_RIGHT.to(a.device, a.dtype)
    ky = _DY_BOTTOM.to(a.device, a.dtype)
    # CHANNEL CONVENTION.  Wandel writes rot_mac(a) = cat([-dx_right(a), dy_bottom(a)]) because his
    # MAC channel order pairs component 0 with the y-derivative.  Our data is [u, v] = [x-vel,
    # y-vel], so the components are swapped here.  With this ordering
    #     div = dx_right(u) + dy_bottom(v) = dx_right(dy_bottom(a)) - dy_bottom(dx_right(a)) = 0
    # exactly, because the two 1-D stencils act on different axes and therefore commute.
    return torch.cat([F.conv2d(a, ky, padding=(1, 0)),
                      -F.conv2d(a, kx, padding=(0, 1))], dim=1)


_NEXT_DP = 0.0     # set/reset by UNet2D.__init__ around block construction (single-threaded)
_NEXT_LS = 0.0
_NEXT_K = 7
_NEXT_GRN = 0.0
_NEXT_DEPTH = 2       # NeXtBlocks per U-net stage (arm D*, 2026-08-25)
_NEXT_GATE = ""       # "" | "swiglu" | "geglu"  (arm GLUa, 2026-08-25)
_NEXT_PAD = "zeros"   # spatial padding_mode of the ConvNeXt depthwise conv (arm IFACE-PAD,
                      # 2026-09-01).  "zeros" is PyTorch's Conv2d default and is the historical
                      # path VERBATIM.  The champion has exactly 10 padded convolutions and they
                      # are ALL `NeXtBlock.dw` (7x7, pad 3) -- every other Conv2d in the net is
                      # 1x1 -- so this one line is the model's ENTIRE spatial boundary treatment.
                      # The window is an OPEN-FLOW CROP (see the note at SelfWarpConv, which for
                      # exactly this reason uses padding_mode="border" for its grid_sample).

# ==========================================================================================
# NORMACT-N (2026-09-01) -- THE NORMALIZATION LAYER AS A KNOB.
#
# WHY.  `nn.GroupNorm` was HARDCODED at every norm site in this file and there was no
# config key for it: the axis was absent from ARCH_KEYS, absent from the 27-cell
# architecture x training grid, and `prior_art.sh` returns ZERO hits for
# LayerNorm / RMSNorm / norm_type / norm-layer.  The champion (`block: convnext`) has
# exactly 10 norm sites, all of them `nn.GroupNorm(1, c)`.
#
# WHAT IS ACTUALLY DIFFERENT, because these two are routinely conflated:
#   GroupNorm(1, c) reduces over (C, H, W) JOINTLY  -> the block output is invariant to
#     the per-sample global amplitude of the whole feature volume.
#   LayerNorm (ConvNeXt's canonical choice) reduces over C ONLY, per spatial location ->
#     the spatial amplitude FIELD survives; only the cross-channel profile is normalised.
# For a wake-forecasting problem the spatial amplitude field is signal (H5: rel error
# tracks angle of attack at r = +0.876), so this is a real modelling choice, not cosmetic.
#
# NUMERICS.  Every reduction is taken in float32 and cast back to the input dtype -- which
# is what torch's own GroupNorm does internally for half inputs.  Training here is pure
# fp32 (train_v4 has no autocast and no GradScaler), so this matters only for the SHIPPED
# fp16 bundle; arm NORMACT-N measures max|fp32 - fp16| per module before any shippability
# claim (the Swin lesson: attention in fp16 is not conv in fp16).
#
# RMSNorm is HAND-ROLLED on purpose: `nn.RMSNorm` does not exist in torch 2.2.2 and BOTH
# eval images are 2.2.2, so using the torch class would measure something that can never
# ship.
# ==========================================================================================

_NEXT_NORM = "group"   # "group" | "layer" | "rms" | "batch" | "none"  (arm NORMACT-N)
NORM_KINDS = ("group", "layer", "rms", "batch", "none")


class LayerNorm2d(nn.Module):
    """LayerNorm over the CHANNEL axis only, on an NCHW tensor -- ConvNeXt's convention.

    Computed directly in NCHW rather than via two permutes: mathematically identical to
    channels-last `nn.LayerNorm(c)` and it avoids two extra kernels per site.
    Parameters: weight (c,), bias (c,) -- the same count and shapes as GroupNorm(g, c),
    which is exactly why the state_dict witness cannot tell them apart (see norm_census).
    """

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = float(eps)

    def forward(self, x):
        dt = x.dtype
        xf = x.float()
        d = xf - xf.mean(1, keepdim=True)
        y = d * torch.rsqrt(d.pow(2).mean(1, keepdim=True) + self.eps)
        y = y * self.weight.float().view(1, -1, 1, 1) + self.bias.float().view(1, -1, 1, 1)
        return y.to(dt)


class RMSNorm2d(nn.Module):
    """RMSNorm over the channel axis (Zhang & Sennrich 2019): no mean subtraction.

    `layer` minus the centring step, so the pair (layer, rms) isolates exactly what the
    mean subtraction is worth.  Parameters: weight (c,) only -- NO bias, which makes this
    cell state_dict-DISTINGUISHABLE from group/layer (they are not distinguishable from
    each other).
    """

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.eps = float(eps)

    def forward(self, x):
        dt = x.dtype
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(1, keepdim=True) + self.eps)
        return (y * self.weight.float().view(1, -1, 1, 1)).to(dt)


class AffineOnly(nn.Module):
    """`norm: none` -- normalisation REMOVED, a learned per-channel affine kept.

    Holds the parameter count and the affine degrees of freedom fixed against GroupNorm,
    so the cell measures the NORMALISATION and not the affine.  Data-independent, hence
    exactly foldable into an adjacent 1x1 convolution at inference.
    """

    def __init__(self, c):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def _norm2d(groups, ch):
    """The norm factory every threaded site calls.

    `_NEXT_NORM == "group"` returns `nn.GroupNorm(groups, ch)` -- the EXACT incumbent call
    with the EXACT incumbent group count of that site, so the default path is
    byte-identical and each site keeps its own historical `groups` (GN(1,c) in NeXtBlock,
    GN(g,co) in DoubleConv).  The group counts are deliberately NOT unified.
    """
    k = str(_NEXT_NORM or "group")
    if k == "group":
        return GroupNorm(groups, ch)
    if k == "layer":
        return LayerNorm2d(ch)
    if k == "rms":
        return RMSNorm2d(ch)
    if k == "batch":
        # DEPLOYMENT RISK, carried here so it cannot be lost: the T1 harness calls predict
        # in fresh subprocesses at chunk size {5,6} and Track 2 is batch-1 streaming.  In
        # TRAIN mode this makes a prediction depend on the other windows in its chunk --
        # cross-window information, a disqualifying condition post-reset -- and at batch 1
        # it collapses the block toward its own bias.  In EVAL mode it is deterministic and
        # safe, but the running statistics are fitted on the TRAIN distribution, and
        # ONLINEGAP measures the online set as a high-Reynolds EXTRAPOLATION population.
        # Measured as an architecture datum; NOT authorised for a card by arm NORMACT-N.
        return nn.BatchNorm2d(ch)
    if k == "none":
        return AffineOnly(ch)
    raise ValueError(f"unknown norm kind {k!r}; expected one of {NORM_KINDS}")


def norm_census(model):
    """Module-class witness for arm NORMACT-N.

    THE STATE_DICT WITNESS THIS REPO NORMALLY RELIES ON IS BLIND TO THIS KNOB:
    nn.GroupNorm(1, c) and LayerNorm2d both carry weight (c,) + bias (c,) under the same
    attribute name, so a checkpoint trained as `layer` STRICT-LOADS into a `group` model
    in silence, with identical parameter counts.  That is the GRIDFIX shape exactly.
    Witness the CONSTRUCTED MODULE CLASSES instead -- this function, not `meta`, is what
    an arm may cite as evidence that the knob was read.
    """
    out = {k: 0 for k in ("GroupNorm", "LayerNorm2d", "RMSNorm2d", "BatchNorm2d",
                          "AffineOnly", "LayerNorm")}
    for m in model.modules():
        n = type(m).__name__
        if n in out:
            out[n] += 1
    return out


class CondFiLM(nn.Module):
    """CONDFILM (2026-09-02, Track1/CONDFILM): feature-wise affine on the cond:stats scalars.

    (gamma, beta) = Linear(32, 2c)(GELU(Linear(4, 32)(c))), c = (cm_u, cm_v, cs_u, cs_v) -- the SAME
    four numbers `cond: stats` already concatenates as constant stem planes, which reach the trunk
    only as ONE additive bias through the stem's 1x1 conv (LEDGER:8519 "cond: stats IS NOT FiLM").
    The LAST layer is zero-initialised (weight AND bias) so gamma = beta = 0 at step 0 and the armed
    network is `torch.equal` to the control from the same seed; the hidden layer is initialised
    under fork_rng(devices=[]) from a PRIVATE generator (seed 20260902) so the GLOBAL RNG stream
    -- and therefore the arm's minibatch sequence -- is the control's.  Applied as
    t * (1 + gamma) + beta at five stage outputs (inc/d1/d2/c2/c1) by UNet2D._film_apply.
    Witness = film_census (module-class walk); a state_dict witness is blind at zero-init."""

    SEED = 20260902

    def __init__(self, ch, hidden=32, n_in=4):
        super().__init__()
        ch, hidden, n_in = int(ch), int(hidden), int(n_in)
        with torch.random.fork_rng(devices=[]):
            self.hid = nn.Linear(n_in, hidden)
            self.out = nn.Linear(hidden, 2 * ch)
            g = torch.Generator().manual_seed(self.SEED + ch)
            with torch.no_grad():
                b = 1.0 / float(n_in) ** 0.5            # nn.Linear's own default bound (1/sqrt(fan_in))
                self.hid.weight.uniform_(-b, b, generator=g)
                self.hid.bias.uniform_(-b, b, generator=g)
                self.out.weight.zero_()
                self.out.bias.zero_()
        self.ch = ch

    def forward(self, c):                          # c (B, n_in) -> gamma (B, ch), beta (B, ch)
        h = F.gelu(self.hid(c))
        gb = self.out(h)
        return gb[:, :self.ch], gb[:, self.ch:]


_FILM_SITES = ("inc", "d1", "d2", "c2", "c1")


def film_census(model):
    """CONDFILM constructed-module witness: count CondFiLM modules by a module-class walk and
    name the sites.  The state_dict witness is provably blind to a zero-initialised FiLM (its
    tensors exist but the function is the identity), so THIS is what train_v4 asserts (5 armed /
    0 control) and what the scorer records per checkpoint."""
    out = {"CondFiLM": 0, "sites": []}
    for name, m in model.named_modules():
        if type(m).__name__ == "CondFiLM":
            out["CondFiLM"] += 1
            out["sites"].append(name.split(".")[-1])
    return out

_NEXT_ACT = ""        # "" | gelu | silu | mish | tanh | beta   (arm NORMACT-A, 2026-09-01)
_NEXT_SE = 0          # 0 | reduction r: SE channel attention on every NeXtBlock residual branch
                      # (arm ATTNUNET-SE, 2026-09-03).  0 (default/absent) builds NOTHING.


class SEGate(nn.Module):
    """ATTNUNET-SE (2026-09-03, Track1/ATTNUNET): squeeze-and-excitation channel attention (Hu,
    Shen & Sun, CVPR 2018, arXiv:1709.01507) on the residual branch of one NeXtBlock, the SE-ResNet
    placement.  gate = 2*sigmoid(W2 relu(W1 mean_hw(h))), h <- h * gate.  The DOUBLED sigmoid is the
    one departure from the canonical block: with W2 zero-initialised (weight AND bias) it makes the
    gate EXACTLY 1 at step 0, so the armed network is torch.equal to the control from the same seed
    (the AttnGate precedent starts at a constant 0.5 rescale and is honest about it; ten residual
    branches at 0.5 would confound the mechanism with an init-scale change, so this arm pays the
    non-canonical range (0, 2) to keep the comparison same-function-at-step-0).  W1 is initialised
    under fork_rng(devices=[]) from a PRIVATE generator (seed 20260903 + c) so the GLOBAL RNG stream
    -- and therefore the arm's minibatch sequence -- is the control's.  Launches per block (n=6,
    fp16, captured): mean, fc1, relu, fc2, sigmoid, mul ~ 6.  Witness = attn_census (module-class
    walk) + the trained gate's per-channel spread; a state_dict witness is blind at zero-init."""

    SEED = 20260903

    def __init__(self, c, r=16):
        super().__init__()
        c, r = int(c), int(r)
        hid = max(4, c // r)
        with torch.random.fork_rng(devices=[]):
            self.fc1 = nn.Linear(c, hid)
            self.fc2 = nn.Linear(hid, c)
            g = torch.Generator().manual_seed(self.SEED + c)
            with torch.no_grad():
                b = 1.0 / float(c) ** 0.5
                self.fc1.weight.uniform_(-b, b, generator=g)
                self.fc1.bias.uniform_(-b, b, generator=g)
                self.fc2.weight.zero_(); self.fc2.bias.zero_()
        self.c, self.r = c, r

    def gate(self, h):
        s = h.mean(dim=(2, 3))
        return 2.0 * torch.sigmoid(self.fc2(F.relu(self.fc1(s))))

    def forward(self, h):
        g = self.gate(h)
        self._last_gate = g.detach()            # LIVE witness (never a parameter, never saved)
        return h * g.reshape(h.shape[0], self.c, 1, 1)


class SCSEGate(nn.Module):
    """ATTNUNET-SCSE (2026-09-03): concurrent spatial and channel squeeze-and-excitation (Roy,
    Navab & Wachinger, MICCAI 2018, arXiv:1803.02579) on ONE encoder skip, x <- x*sig_c(x) +
    x*sig_s(x): the channel gate is SE (mean_hw -> W1 -> relu -> W2 -> sigmoid), the spatial gate a
    1x1 conv -> sigmoid.  Both last layers zero-initialised => 0.5 + 0.5 = 1, an EXACT identity at
    step 0 with the CANONICAL additive combination (no doubled sigmoid needed here).  This is the
    skip-gating dual of AttnGate: the skip decides what it admits from ITSELF (no decoder gating
    signal), which is the feature-recalibration reading of attention rather than Oktay's
    coarse-to-fine one.  W1 / the 1x1 conv init under fork_rng from a private generator (seed
    20260903 + 7*c).  Launches ~ 10 per skip (2 skips)."""

    SEED = 20260903

    def __init__(self, c, r=16):
        super().__init__()
        c, r = int(c), int(r)
        hid = max(4, c // r)
        with torch.random.fork_rng(devices=[]):
            self.fc1 = nn.Linear(c, hid)
            self.fc2 = nn.Linear(hid, c)
            self.sp = nn.Conv2d(c, 1, 1)
            g = torch.Generator().manual_seed(self.SEED + 7 * c)
            with torch.no_grad():
                b = 1.0 / float(c) ** 0.5
                self.fc1.weight.uniform_(-b, b, generator=g)
                self.fc1.bias.uniform_(-b, b, generator=g)
                self.fc2.weight.zero_(); self.fc2.bias.zero_()
                self.sp.weight.zero_(); self.sp.bias.zero_()
        self.c = c

    def gates(self, x):
        gc = torch.sigmoid(self.fc2(F.relu(self.fc1(x.mean(dim=(2, 3)))))).reshape(x.shape[0], self.c, 1, 1)
        gs = torch.sigmoid(self.sp(x))
        return gc, gs

    def forward(self, x):
        gc, gs = self.gates(x)
        self._last_gate = (gc.detach(), gs.detach())   # LIVE witness (never a parameter, never saved)
        return x * gc + x * gs


def attn_census(model):
    """ATTNUNET constructed-module witness: module-class walk counting every attention-class
    module (AttnGate = `skip: gate`, SCSEGate = `skip: scse`, SEGate = `nxt_se`, BottleneckAttn =
    `fuse: attn`, TemporalAttn = `t_attn`, SwinConv = `block: swin`) and naming the sites.  The
    state_dict witness is blind to a zero-initialised gate; THIS is what train_v4 asserts and the
    scorer records."""
    out = {k: 0 for k in ("AttnGate", "SCSEGate", "SEGate", "BottleneckAttn", "TemporalAttn", "SwinConv")}
    sites = []
    for name, m in model.named_modules():
        n = type(m).__name__
        if n in out:
            out[n] += 1
            sites.append(f"{n}:{name}")
    out["sites"] = sites
    return out


class BetaSwish(nn.Module):
    """Swish with a LEARNABLE per-channel beta:  y = x * sigmoid(beta * x).

    A one-parameter family that CONTAINS the incumbent and its neighbours:
        beta -> 0      linear
        beta  = 1.0    SiLU
        beta  = 1.702  ~ GELU  (Hendrycks & Gimpel 2016 eq. 2's own approximation)
        beta -> inf    ReLU
    Initialised at 1.702, so an untrained ABETA block is the incumbent to within the GELU/Swish
    approximation error (NOT bit-identical -- stated in the pre-registration).  The point of the
    arm is the TRAINED beta: it reads out where the network wants to sit inside that family,
    which no fixed-curve arm can report.
    """

    def __init__(self, width, init=1.702):
        super().__init__()
        self.beta = nn.Parameter(torch.full((1, int(width), 1, 1), float(init)))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


def _next_act(gate, width):
    """The pointwise nonlinearity inside NeXtBlock's MLP.

    `_NEXT_ACT == ""` returns the historical expression VERBATIM, so the shipped path keeps its
    exact class, construction order and RNG consumption.  Every other value is an arm.
    """
    a = str(_NEXT_ACT or "")
    if not a:
        return nn.SiLU() if gate == "swiglu" else nn.GELU()
    if a == "gelu":
        return nn.GELU()
    if a == "silu":
        return nn.SiLU()
    if a == "mish":
        return nn.Mish()
    if a == "tanh":
        return nn.Tanh()
    if a == "beta":
        return BetaSwish(width)
    raise ValueError(f"nxt_act={a!r} is not one of '', gelu, silu, mish, tanh, beta")


class NeXtBlock(nn.Module):
    """One ConvNeXt-style residual block: dw 7x7 -> GN(1) -> pw 4x -> GELU -> pw, residual.
    Optional LayerScale (per-channel gamma) and DropPath (sample-wise stochastic depth) — the
    training half of the canonical ConvNeXt recipe, off by default (NXa-compatible)."""

    def __init__(self, c):
        super().__init__()
        _k = int(_NEXT_K)
        # IFACE-PAD: `_NEXT_PAD == "zeros"` is nn.Conv2d's own default, so the historical
        # construction is BYTE-IDENTICAL (same module class, same parameters, same RNG).
        self.dw = nn.Conv2d(c, c, _k, padding=_k // 2, groups=c,
                            padding_mode=str(_NEXT_PAD or "zeros"))
        self.n = _norm2d(1, c)
        # GLUa (2026-08-25).  gate="" is the historical block, BYTE-IDENTICAL.  With a gate the
        # MLP becomes  p1: c -> 2h,  out = v * act(g),  p2: h -> c.  The ungated block costs
        # 4c^2 + 4c^2 = 8c^2; the gated one costs 2hc + hc = 3hc, so h = 8c/3 holds the parameter
        # count fixed and the arm measures the GATE, not capacity.  Shazeer 2020 arXiv:2002.05202.
        self.gate = str(_NEXT_GATE or "")
        assert self.gate in ("", "swiglu", "geglu"), self.gate
        if self.gate:
            h = int(round(8 * c / 3))
            self.p1 = nn.Conv2d(c, 2 * h, 1)
            self.p2 = nn.Conv2d(h, c, 1)
        else:
            self.p1 = nn.Conv2d(c, 4 * c, 1)
            self.p2 = nn.Conv2d(4 * c, c, 1)
        # NORMACT-A: `_NEXT_ACT == ""` reproduces the line this replaces, VERBATIM.
        self.act = _next_act(self.gate, (int(round(8 * c / 3)) if self.gate else 4 * c))
        self.grn = bool(_NEXT_GRN)
        if self.grn:
            _w = int(round(8 * c / 3)) if self.gate else 4 * c
            self.g_g = nn.Parameter(torch.zeros(1, _w, 1, 1))
            self.g_b = nn.Parameter(torch.zeros(1, _w, 1, 1))
        self.dp = float(_NEXT_DP)
        self.gamma = nn.Parameter(torch.full((c, 1, 1), float(_NEXT_LS))) if _NEXT_LS > 0 else None
        # ATTNUNET-SE: `_NEXT_SE == 0` (default/absent) builds NOTHING -- no module, no RNG draw,
        # the historical block BYTE-IDENTICAL.  Constructed LAST so the block's own RNG draws are
        # unchanged; SEGate itself draws only from a private generator.
        self.se = SEGate(c, int(_NEXT_SE)) if int(_NEXT_SE) > 0 else None

    def forward(self, x):
        if self.gate:
            v, g = self.p1(self.n(self.dw(x))).chunk(2, 1)
            h = v * self.act(g)
        else:
            h = self.act(self.p1(self.n(self.dw(x))))
        if self.grn:
            # ConvNeXt-V2 GRN: global response normalisation on the expanded features;
            # zero-init gamma/beta -> exact no-op at start, like every fuse before it.
            gx = torch.norm(h, p=2, dim=(2, 3), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            h = h + self.g_g * (h * nx) + self.g_b * h
        h = self.p2(h)
        if self.se is not None:
            h = self.se(h)                   # ATTNUNET-SE: gate == 1 exactly at step 0
        if self.gamma is not None:
            h = h * self.gamma
        if self.training and self.dp > 0.0:
            keep = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) >= self.dp).to(x.dtype)
            h = h * keep / (1.0 - self.dp)
        return x + h


class NeXtConv(nn.Module):
    """ConvNeXt-style stage in DoubleConv's slot: 1x1 channel change + two residual blocks.
    Same parameter class as DoubleConv at equal width (arm NXa measures the block design,
    not capacity)."""

    def __init__(self, ci, co, groups=8):
        super().__init__()
        # _NEXT_DEPTH == 2 is the historical stage and is BYTE-IDENTICAL (same modules, same
        # construction order, same RNG draws).  SelfWarpConv has had _SW_DEPTH since WARPa;
        # ConvNeXt never got the equivalent knob, so "how many conv layers" was unmeasurable
        # on the champion's own block until now.
        d = int(_NEXT_DEPTH)
        assert d >= 1, d
        self.net = nn.Sequential(nn.Conv2d(ci, co, 1), *[NeXtBlock(co) for _ in range(d)])

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------------------------------
# SSM (arm SSMa, 2026-08-25) -- a DIAGONAL STATE-SPACE block of the S4D / Mamba class, sitting
# in DoubleConv's slot so it is a like-for-like swap against NeXtConv (same 1x1 + two residual
# blocks, same parameter class at equal width).
#
# WHY IT EXISTS.  It is not proposed as a better forecaster.  XCORR/XDEC measured rho = 0.91-0.96
# and lambda ~ 1.00 between every pair of models this repo can train, and that measurement covers
# exactly TWO architecture families (CNO and U-net-with-conv-blocks).  The standing conclusion
# "any architecture we can train lands on the same conditional mean" is therefore an INDUCTION
# from two points.  SSMa is the third point, chosen to be the most different MIXING family we can
# build: a learned-decay GLOBAL scan gives a full-extent receptive field in ONE block, which no
# 7x7 depthwise convolution has at any depth reachable here.
#
# WHAT IT IS, precisely.  h_t = a*h_{t-1} + u_t with a diagonal (per-channel) a in (0,1) is a
# convolution with the kernel a^k, so the scan needs no custom CUDA kernel and -- per the repo's
# standing rule about the fp16 half path -- NO FFT in the forward.  Bidirectional along W, then
# bidirectional along H.  `a` is initialised log-spaced across channels, which is S4D's own
# multi-timescale initialisation.  The SiLU gate on a second projected branch is Mamba's other
# half (input-dependent gating); the selective/input-dependent `a` is NOT included, because that
# is what needs the fused scan kernel.  Stated so the arm is not over-claimed: this is S4D + a
# Mamba gate, not full selective Mamba.
#
# gamma is zero-initialised, so at step 0 the block is EXACTLY the identity plus the 1x1 -- the
# same no-op-at-start discipline every fuse in this file uses.

_SSM_E = 2.0


class _Scan(nn.Module):
    """Bidirectional diagonal scan along the LAST axis of an (N, C, L) tensor."""

    def __init__(self, c):
        super().__init__()
        # log-spaced decays: a from ~0.05 (very local) to ~0.995 (near-global) across channels
        lin = torch.linspace(0.0, 1.0, c)
        a0 = 0.05 ** (1.0 - lin) * 0.995 ** lin
        self.logit_f = nn.Parameter(torch.log(a0 / (1.0 - a0)))
        self.logit_b = nn.Parameter(torch.log(a0 / (1.0 - a0)))
        self._cache = {}

    def _kern(self, logit, L):
        a = torch.sigmoid(logit).unsqueeze(1)                     # (C,1)
        k = torch.arange(L, device=logit.device, dtype=logit.dtype).unsqueeze(0)
        return (a ** k).unsqueeze(1)                              # (C,1,L)

    def forward(self, x):                                         # (N,C,L)
        L = x.shape[-1]
        C = x.shape[1]
        kf = self._kern(self.logit_f, L)
        kb = self._kern(self.logit_b, L)
        # causal: pad L-1 on the left, drop the tail.  anti-causal: flip, same, flip back.
        yf = F.conv1d(F.pad(x, (L - 1, 0)), kf, groups=C)
        yb = F.conv1d(F.pad(x.flip(-1), (L - 1, 0)), kb, groups=C).flip(-1)
        return yf + yb


class SSMBlock(nn.Module):
    """One S4D/Mamba-class residual block: GN -> 1x1 to (u, gate) -> scan_W -> scan_H
    -> u * SiLU(gate) -> 1x1 -> LayerScale -> residual."""

    def __init__(self, c, expand=_SSM_E):
        super().__init__()
        e = int(round(expand * c))
        self.n = GroupNorm(1, c)
        self.inp = nn.Conv2d(c, 2 * e, 1)
        self.dw = nn.Conv2d(e, e, 7, padding=3, groups=e)   # Mamba-2 / vision-Mamba local mixer
        self.sw = _Scan(e)
        self.sh = _Scan(e)
        self.out = nn.Conv2d(e, c, 1)
        self.gamma = nn.Parameter(torch.zeros(c, 1, 1))

    def forward(self, x):
        b, _, h, w = x.shape
        u, g = self.inp(self.n(x)).chunk(2, 1)
        u = self.dw(u)
        c = u.shape[1]
        # scan along W: fold H into the batch axis
        u = self.sw(u.permute(0, 2, 1, 3).reshape(b * h, c, w)).reshape(b, h, c, w).permute(0, 2, 1, 3)
        # scan along H: fold W into the batch axis
        u = self.sh(u.permute(0, 3, 1, 2).reshape(b * w, c, h)).reshape(b, w, c, h).permute(0, 2, 3, 1)
        return x + self.out(u * F.silu(g)) * self.gamma


class SSMConv(nn.Module):
    """SSM stage in DoubleConv's slot: 1x1 channel change + two residual SSM blocks --
    structurally identical to NeXtConv, so the arm measures the MIXING family, not capacity."""

    def __init__(self, ci, co, groups=8):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ci, co, 1), SSMBlock(co), SSMBlock(co))

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------------------------------
# SWIN (arm SWINa, 2026-08-25) -- shifted-window multi-head self-attention in DoubleConv's slot.
#
# WHY.  The user asked for the U-net-vs-Swin comparison the literature keeps running, done in OUR
# rig instead of quoted from a paper.  Structurally it is the fourth mixing family measured here
# (conv-stack, CNO, diagonal-SSM, now windowed attention) and the first with CONTENT-DEPENDENT
# mixing -- SSMa's scan is input-independent by construction, so a high rho there did not settle
# whether attention would behave the same way.
#
# WHAT.  A faithful W-MSA / SW-MSA pair: LayerNorm -> window partition -> multi-head attention with
# a learned relative-position bias -> merge -> residual, then LayerNorm -> MLP(4x) -> residual, with
# the second block cyclically shifted by half a window and masked so wrapped pairs cannot attend.
# Window is min(8, H) x min(8, W) so the deeper U-net stages (16x32, 8x16) stay legal.  No external
# dependency, no einops, no FFT.

def _win_part(x, wh, ww):                       # (B,H,W,C) -> (B*nW, wh*ww, C)
    B, H, W, C = x.shape
    x = x.view(B, H // wh, wh, W // ww, ww, C).permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(-1, wh * ww, C)


def _win_merge(w, B, H, W, wh, ww):             # inverse of _win_part
    C = w.shape[-1]
    x = w.view(B, H // wh, W // ww, wh, ww, C).permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, H, W, C)


class SwinBlock(nn.Module):
    def __init__(self, c, heads=4, win=8, shift=False, mlp=4):
        super().__init__()
        self.c, self.heads, self.win, self.shift = c, heads, win, bool(shift)
        self.n1, self.n2 = nn.LayerNorm(c), nn.LayerNorm(c)
        self.qkv = nn.Linear(c, 3 * c)
        self.proj = nn.Linear(c, c)
        self.mlp = nn.Sequential(nn.Linear(c, mlp * c), nn.GELU(), nn.Linear(mlp * c, c))
        self.rpb = nn.Parameter(torch.zeros((2 * win - 1) ** 2, heads))
        nn.init.trunc_normal_(self.rpb, std=0.02)
        self.gamma = nn.Parameter(torch.zeros(c))      # zero-init: identity at step 0
        self._idx, self._mask = {}, {}

    def _bias_idx(self, wh, ww, dev):
        k = (wh, ww, str(dev))
        if k not in self._idx:
            cy = torch.arange(wh, device=dev); cx = torch.arange(ww, device=dev)
            cc = torch.stack(torch.meshgrid(cy, cx, indexing="ij")).flatten(1)   # (2, wh*ww)
            rel = (cc[:, :, None] - cc[:, None, :]).permute(1, 2, 0)             # (N,N,2)
            rel[..., 0] += wh - 1; rel[..., 1] += ww - 1
            self._idx[k] = (rel[..., 0] * (2 * ww - 1) + rel[..., 1]).reshape(-1)
        return self._idx[k]

    def _attn_mask(self, B, H, W, wh, ww, dev):
        k = (H, W, wh, ww, str(dev))
        if k not in self._mask:
            img = torch.zeros(1, H, W, 1, device=dev)
            sh, sw = wh // 2, ww // 2
            cnt = 0
            for hs in (slice(0, -wh), slice(-wh, -sh), slice(-sh, None)):
                for ws in (slice(0, -ww), slice(-ww, -sw), slice(-sw, None)):
                    img[:, hs, ws, :] = cnt; cnt += 1
            mw = _win_part(img, wh, ww).squeeze(-1)                              # (nW, N)
            m = mw.unsqueeze(1) - mw.unsqueeze(2)
            self._mask[k] = m.masked_fill(m != 0, float(-100.0)).masked_fill(m == 0, 0.0)
        return self._mask[k]

    def forward(self, x):                        # (B,C,H,W)
        B, C, H, W = x.shape
        wh, ww = min(self.win, H), min(self.win, W)
        sh, sw = (wh // 2, ww // 2) if self.shift else (0, 0)
        z = x.permute(0, 2, 3, 1)                                                # (B,H,W,C)
        r = z
        z = self.n1(z)
        if sh or sw:
            z = torch.roll(z, shifts=(-sh, -sw), dims=(1, 2))
        w = _win_part(z, wh, ww)                                                 # (B*nW, N, C)
        BN, N, _ = w.shape
        qkv = self.qkv(w).reshape(BN, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = (q @ k.transpose(-2, -1)) * ((C // self.heads) ** -0.5)
        a = a + self.rpb[self._bias_idx(wh, ww, x.device)].view(N, N, self.heads).permute(2, 0, 1)
        if sh or sw:
            msk = self._attn_mask(B, H, W, wh, ww, x.device)                     # (nW,N,N)
            # .to(a.dtype): the mask is cached in a plain python dict, which `.half()` cannot
            # reach, so without this cast `a + msk` promotes the logits back to fp32 and the
            # following `a @ v` mixes dtypes -- a hard crash on the fp16 path EVERY cuda
            # bundle takes.  Found by the 2026-08-26 3-blend audit, reproduced on dgx.
            a = (a.view(B, -1, self.heads, N, N)
                 + msk.to(a.dtype).unsqueeze(0).unsqueeze(2)).view(BN, self.heads, N, N)
        a = a.softmax(-1)
        o = (a @ v).transpose(1, 2).reshape(BN, N, C)
        o = _win_merge(self.proj(o), B, H, W, wh, ww)
        if sh or sw:
            o = torch.roll(o, shifts=(sh, sw), dims=(1, 2))
        z = r + o * self.gamma
        z = z + self.mlp(self.n2(z)) * self.gamma
        return z.permute(0, 3, 1, 2).contiguous()


class SwinConv(nn.Module):
    """Swin stage in DoubleConv's slot: 1x1 channel change + W-MSA block + SW-MSA block --
    the same 1x1-plus-two-blocks shape as NeXtConv, so the arm measures the mixing family."""

    def __init__(self, ci, co, groups=8):
        super().__init__()
        h = max(1, min(4, co // 32))
        self.pre = nn.Conv2d(ci, co, 1)
        self.b1 = SwinBlock(co, heads=h, shift=False)
        self.b2 = SwinBlock(co, heads=h, shift=True)

    def forward(self, x):
        return self.b2(self.b1(self.pre(x)))


# ------------------------------------------------------------------------------------------
# SKIP-PATH VARIANTS (arms AGa / U3Pa / SRa, 2026-08-25).  All three change WHAT REACHES THE
# DECODER, which is the one part of the U-net every block-level arm so far has left alone.
# `skip=""` / `stage_res=0` are the historical paths and are BYTE-IDENTICAL.

class AttnGate(nn.Module):
    """Attention U-Net gate (Oktay et al. 2018, arXiv:1804.03999) on ONE skip connection.

    psi = sigmoid(W_psi(relu(W_g(g) + W_x(x))));  the skip is returned as x * psi.  `g` is the
    decoder feature arriving from below, `x` the encoder feature at this level -- so the encoder
    detail is admitted only where the coarse decoder state says it is wanted.  W_psi is
    ZERO-INITIALISED, which makes psi = 0.5 everywhere at step 0: a constant rescale, not an
    identity, so the arm is honest about starting slightly off the control (a sigmoid gate cannot
    be an exact identity at init without a bias hack that would defeat the mechanism)."""

    def __init__(self, cx, cg, ci=None):
        super().__init__()
        ci = ci or max(8, cx // 2)
        self.wg = nn.Conv2d(cg, ci, 1)
        self.wx = nn.Conv2d(cx, ci, 1)
        self.psi = nn.Conv2d(ci, 1, 1)
        nn.init.zeros_(self.psi.weight); nn.init.zeros_(self.psi.bias)

    def forward(self, x, g):
        return x * torch.sigmoid(self.psi(F.relu(self.wg(g) + self.wx(x))))


class StageRes(nn.Module):
    """Residual U-Net stage: a 1x1 shortcut ACROSS a whole stage (the channel change plus both
    blocks).  NeXtBlock is already residual internally, so this adds only the one skip the
    ConvNeXt stage does not have -- stated plainly because it means the arm is a small delta,
    not the textbook ResUNet-vs-UNet gap."""

    def __init__(self, inner, ci, co):
        super().__init__()
        self.inner = inner
        self.sc = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros(co, 1, 1))   # exact no-op at step 0

    def forward(self, x):
        return self.inner(x) + self.sc(x) * self.gamma


# ------------------------------------------------------------------------------------------
# SELFWARP (arm WARPa) -- "Flowers: A Warp Drive for Neural PDE Solvers", Muser, Spitzer,
# Lassas, de Hoop, Dokmanic, ICML 2026, arXiv:2603.04430v2.  Reference implementation:
# github.com/t-muser/flowers (flowers/models/flower_layers.py).
#
# The block mixes information NON-LOCALLY with no convolution, no FFT and no attention matrix:
# at every pixel a POINTWISE MLP predicts H per-head displacements rho^(h)(x); a pointwise
# affine map gives a value field v = Vu split into H heads of c/H channels; each head is
# resampled at x + rho^(h)(x) -- one bilinear `grid_sample` -- and the heads are concatenated.
# Cost is linear in grid points; nonlocality enters ONLY through one sparse sample per head.
# Wrapped as the paper's App C.3 block:  u -> GELU(GroupNorm(SELFWARP[u] + IdProj[u])).
# Note this is NOT a conventional residual block: the "skip" is a learned 1x1 (IdProj), so
# there is no identity path, exactly as in the reference.
#
# SEVEN DELIBERATE DEPARTURES from the paper / reference code.  Everything NOT listed here
# reproduces the reference exactly (heads-major channel partition, one shared value map,
# bilinear + align_corners=True, GroupNorm groups tied to the head count, GELU block
# activation, no per-head mixing after the warp).
#  1. FUSED INPUT PROJECTION.  The reference runs three separate 1x1 convs on u (value,
#     IdProj, flow-MLP hidden); we emit all three from ONE conv c -> (2c + c/2).  Same math,
#     one GEMM instead of three.  Measured: it cut the batch-8 latency penalty from 2.04x to
#     1.21x of the ConvNeXt champion -- at 32x64 this net is kernel-LAUNCH-bound, not
#     FLOP-bound, so kernel COUNT is the cost that matters at small batch.
#  2. FLOW-MLP HIDDEN WIDTH c/2 instead of the reference's c.  Pure cost: the projection term
#     2.5c^2 dominates the block and the flow head is ~6% of it either way.
#  3. GELU in the flow MLP where the reference uses ReLU -- consistency with every other block
#     in this file; measured cost difference nil.
#  4. BOUNDED DISPLACEMENTS.  rho = tanh(raw) * (_SW_MAX_FRAC * H) PIXELS; the reference adds
#     the raw conv output straight onto the [-1,1] normalised grid, unbounded.  Two reasons.
#     (a) Our grid is 32x64 where theirs is 128x128+, so a displacement that is a modest
#     fraction of their domain is most of ours.  (b) Measured on THIS rig, honest advection
#     over the 20-frame horizon is 0.27 px (rigid) to 0.76 px (per-pixel), so the cap here
#     (8 px at the finest stage) is already ~10x the physical scale -- there is nothing to buy
#     above it.  The cap is isotropic IN PIXELS, not in normalised units, because this rig's
#     pixels are physically square (dx 3.45e-3 m, dy 3.48e-3 m) on a 2:1 grid -- a "fraction of
#     the domain" would allow 2x more displacement in x than in y for no physical reason.
#     Being a fraction of H it is 8 / 4 / 2 px at the U-net's 32x64 / 16x32 / 8x16 stages: the
#     same physical reach at every scale, never more than a quarter of the field.
#  5. ZERO-INIT of the flow head's last conv, so an untrained block is the IDENTITY WARP and a
#     failed arm degrades to a pointwise projection rather than to noise.  The reference uses
#     PyTorch defaults, which ON OUR GRID start at |rho| ~ 6.6 px mean / 30 px max -- i.e. the
#     paper's own "non-learned random warps" ablation (1-step 0.2190 vs 0.0624).  This file's
#     convention (BottleneckAttn, GRN) is that a new block starts as a no-op.
#  6. padding_mode="border", not the reference's "zeros".  Our PIV window is an open-flow CROP:
#     neither periodic nor zero at its edges, and zero-padding injects a hard artificial edge
#     exactly where the wake leaves the field of view.  Worse, the real targets are EXACTLY
#     zero inside the airfoil mask, so "zeros" is a second source of zeros the model can
#     confuse with the mask.
#  7. _SW_DEPTH blocks per U-net STAGE, where the paper's U-Net has one FlowerBlock per level.
#     We match OUR stage structure (NeXtConv = 1x1 + 2 residual blocks) rather than theirs, so
#     the arm measures the BLOCK and not the depth, and we size to the measured compute budget:
#     base=128, depth=3 gives 0.94x the champion's parameters and 1.01x its MACs.
#
# ONE THING THE FRAME-STACKING DOES TO THIS LAYER, worth knowing: `UNet2D.forward` folds time
# into channels, so a head warps all 20 frames of its channel group with ONE displacement
# field.  That is the same convention the paper uses (it stacks its history frames too), just
# at T=20 instead of T=4.  And with `in_norm: self` the flow MLP reads NORMALISED values, so it
# cannot observe absolute speed -- it sees the flow's SHAPE.  The compensating benefit is that
# the learned displacement field is automatically transferable across Reynolds number, which is
# the axis the leaderboard lives on.  (`cond: stats` would restore absolute scale to the stem;
# that is a separate arm.)
#
# THE CONSTANTS BELOW ARE DELIBERATELY NOT CONFIG KEYS.  `block` is already carried through
# checkpoint meta, so `block: selfwarp` rebuilds exactly with ZERO new plumbing.  A new
# hyper-parameter would need THREE simultaneous edits -- ARCH_KEYS in this file, the `_arch`
# copy list in train_v4.py and `_ckpt_meta` there -- or it silently reverts at rebuild time and
# produces a wrong forward that still strict-loads.  SWEEPING ANY OF THESE REQUIRES ADDING
# THOSE KEYS FIRST.  Each block freezes them into itself at construction, so at least one
# process cannot build two models that disagree with their own state_dicts.
_SW_HEADS = 8          # warp heads per block; channels per head = c // _SW_HEADS (16/32/64)
_SW_DEPTH = 3          # SelfWarpBlocks per U-net stage (NeXtConv's slot holds 2 NeXtBlocks)
_SW_HID = 0.5          # flow-MLP hidden width as a fraction of c
_SW_MAX_FRAC = 0.25    # max |displacement| in PIXELS as a fraction of H -> 8 / 4 / 2 px
_SW_PAD = "border"     # grid_sample padding; the reference uses "zeros" (departure 6)

_SW_GRID_CACHE: dict = {}


def _sw_base_grid(h, w, device, dtype):
    """Identity sampling grid, (1, h, w, 2): [...,0] = x across W, [...,1] = y across H, both in
    grid_sample's normalised [-1,1] with align_corners=True.

    Cached per (h, w, device, dtype).  It must NOT be a persistent buffer -- it is a pure
    function of the shape, and baking one resolution into the state_dict would break loading at
    another.  Rebuilding it every call is both wasteful (the reference's own source warns about
    exactly this) and extra work inside a CUDA-graph capture; built once during the evaluator's
    untimed warm-up call, capture sees a plain tensor read.
    """
    key = (int(h), int(w), str(device), dtype)
    g = _SW_GRID_CACHE.get(key)
    if g is None:
        ys = torch.linspace(-1.0, 1.0, int(h), device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, int(w), device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        g = torch.stack([gx, gy], dim=-1).unsqueeze(0)              # (1, h, w, 2)
        _SW_GRID_CACHE[key] = g
    return g


class SelfWarpBlock(nn.Module):
    """One FLOWER block at fixed width: GELU(GroupNorm(SELFWARP[u] + IdProj[u]))."""

    def __init__(self, c, heads=None):
        super().__init__()
        c = int(c)
        h = int(heads or _SW_HEADS)
        self.c = c
        self.hh = h if c % h == 0 else 1
        self.hid = max(8, int(round(float(_SW_HID) * c)))
        self.max_frac = float(_SW_MAX_FRAC)      # frozen at construction, see the note above
        self.pad = str(_SW_PAD)
        # ONE 1x1 emits [ value v (c) | IdProj (c) | flow-MLP hidden (hid) ].
        self.pin = nn.Conv2d(c, 2 * c + self.hid, 1)
        self.flow = nn.Conv2d(self.hid, 2 * self.hh, 1)
        self.act = nn.GELU()
        # channels are HEADS-MAJOR, so groups == heads normalises each head as its own group --
        # the alignment the reference uses (num_heads == groups == 40).
        self.norm = GroupNorm(self.hh, c)
        nn.init.zeros_(self.flow.weight)         # identity warp at init (departure 5)
        nn.init.zeros_(self.flow.bias)

    def forward(self, x):
        b, _, h, w = x.shape
        y = self.pin(x)
        v = y[:, :self.c]
        idp = y[:, self.c:2 * self.c]
        raw = self.flow(self.act(y[:, 2 * self.c:]))                 # (b, 2*hh, h, w)
        # bounded displacement in pixels -> normalised grid units.  Python floats only, so no
        # host tensor is created here and the whole forward stays CUDA-graph capturable.
        max_px = self.max_frac * h
        sx = 2.0 * max_px / max(w - 1, 1)
        sy = 2.0 * max_px / max(h - 1, 1)
        d = torch.tanh(raw).view(b, self.hh, 2, h, w)
        d = torch.stack([d[:, :, 0] * sx, d[:, :, 1] * sy], dim=-1)  # (b, hh, h, w, 2)
        grid = _sw_base_grid(h, w, x.device, x.dtype) + d.reshape(b * self.hh, h, w, 2)
        vh = v.reshape(b * self.hh, self.c // self.hh, h, w)
        wv = F.grid_sample(vh, grid, mode="bilinear", padding_mode=self.pad,
                           align_corners=True)
        return self.act(self.norm(wv.reshape(b, self.c, h, w) + idp))

    def disp_px(self, x):
        """Diagnostic ONLY (never called in forward): the induced displacement field in PIXELS,
        (b, hh, h, w, 2).  The paper leaves the units open and the reference settles them as a
        fraction of the domain; our Stage-0 measured honest advection at 0.27-0.76 px over the
        horizon, so if a trained model's displacements sit two orders above that the layer is
        doing reconstruction rather than transport -- and we should learn that from a log rather
        than from a score."""
        b, _, h, w = x.shape
        raw = self.flow(self.act(self.pin(x)[:, 2 * self.c:]))
        return torch.tanh(raw).view(b, self.hh, 2, h, w).permute(0, 1, 3, 4, 2) * \
            (self.max_frac * h)


class SelfWarpConv(nn.Module):
    """SELFWARP stage in DoubleConv's slot: 1x1 channel change + _SW_DEPTH warp blocks.
    Mirrors NeXtConv's structure so that `block` is the only variable in the A/B."""

    def __init__(self, ci, co, groups=8):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ci, co, 1),
                                 *[SelfWarpBlock(co) for _ in range(int(_SW_DEPTH))])

    def forward(self, x):
        return self.net(x)


class TemporalAttn(nn.Module):
    """Input-dependent self-attention along the TIME axis of the input window (arm TAa).

    The trunk stacks the t_in frames into channels, so it can only apply a FIXED, per-lag-distinct
    linear map across time.  This applies one shared, input-dependent operator instead: for every
    pixel independently, the length-`t` sequence of `c` channels attends to itself.

    Bit-exact identity at initialisation (`out` is zero-init), which is what lets it be A/B'd
    against the champion as a strict superset.
    """

    def __init__(self, c_in, t=20, dim=32, heads=4):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} not divisible by heads {heads}"
        self.dim, self.nh, self.hd = int(dim), int(heads), int(dim) // int(heads)
        self.proj = nn.Linear(c_in, dim)
        # Learned absolute positional code over the 20 lags.  Absolute (not relative) on purpose:
        # the lags are a FIXED, short, fully-observed grid, so there is nothing to generalise to.
        self.pos = nn.Parameter(torch.zeros(t, dim))
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, c_in)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):                       # x (B,T,H,W,C) -> same
        b, t, hh, ww, c = x.shape
        s = x.permute(0, 2, 3, 1, 4).reshape(b * hh * ww, t, c)
        y = self.proj(s) + self.pos[:t]
        q, k, v = self.qkv(self.norm(y)).chunk(3, dim=-1)
        q = q.reshape(-1, t, self.nh, self.hd).transpose(1, 2)
        k = k.reshape(-1, t, self.nh, self.hd).transpose(1, 2)
        v = v.reshape(-1, t, self.nh, self.hd).transpose(1, 2)
        # Explicit, NOT F.scaled_dot_product_attention: one sequence per pixel makes the kernel
        # batch B*H*W*heads (131072 at batch 16), which is past a CUDA grid limit the fused kernels
        # launch against.  t = 20, so the 20x20 attention matrix is cheap written out.
        a = (q @ k.transpose(-2, -1)) * (self.hd ** -0.5)
        o = (a.softmax(dim=-1) @ v)
        o = o.transpose(1, 2).reshape(-1, t, self.dim)
        d = self.out(o).reshape(b, hh, ww, t, c).permute(0, 3, 1, 2, 4)
        return x + d


def _lap5(x):
    """5-point Laplacian on (B,C,H,W), replicate-padded.  Matches the probe's numpy operator."""
    p = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="replicate")
    return (p[:, :, 1:-1, 2:] + p[:, :, 1:-1, :-2] + p[:, :, 2:, 1:-1] + p[:, :, :-2, 1:-1]
            - 4.0 * x)


def freq_smooth(x, n, k=0.25):
    """`n` passes of x <- x + k*lap(x): explicit heat diffusion, k=0.25 is the 2-D stability limit.
    Shape-preserving and parameter-free, so it can be applied to targets inside the loss."""
    for _ in range(int(n)):
        x = x + k * _lap5(x)
    return x


class PointwiseMLP(nn.Module):
    """The high-frequency branch of Freq-decomp (arm FDa): a stack of 1x1 convolutions, i.e. an exactly
    POINTWISE MLP.  It has NO spatial support, so unlike the convolutional head it cannot smooth --
    which is the whole point of giving the rough component its own decoder.

    Zero-initialised last layer, so at step 0 the network's output is bit-identical to the shipped one
    and the arm is a strict superset rather than a different model.  Kept as a SIBLING of `self.outc`
    rather than a wrapper around it: wrapping renames outc.weight -> outc.lo.weight, which stops
    existing checkpoints from loading, and `outc` is referenced elsewhere in this class.
    """

    def __init__(self, ch, out_ch, width=64, layers=2, act="gelu"):
        super().__init__()
        A = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}.get(str(act), nn.GELU)
        seq, c = [], int(ch)
        for _ in range(max(1, int(layers) - 1)):
            seq += [nn.Conv2d(c, int(width), 1), A()]
            c = int(width)
        last = nn.Conv2d(c, int(out_ch), 1)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        seq.append(last)
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


class RouteStage(nn.Module):
    """RROUTE (2026-08-26; queue entry from arXiv:2608.13490's regime-routing idea): the deepest
    U-net stage becomes TWO parallel reduced-width branches of the SAME block class, mixed per
    location by a learned soft spatial mask computed from the stage INPUT alone:

        out = m * B1(x) + (1 - m) * B2(x),      m = sigmoid(1x1(x))  in (0,1), shape (B,1,h,w)

    The binding physical fact: per-file OOF rel runs 0.044 (AoA 0) -> 0.104 (AoA 20), so the
    separated wake carries the error; a per-pixel router lets capacity specialise between
    attached-flow and wake regions WITHOUT partitioning the data (the mixture-of-Re-experts
    closure, LEDGER 2026-08-06, killed per-regime FINETUNING on data-diversity loss -- both
    branches here train on every sample, jointly).

    PARAMETER MATCHING.  Branch params are ~quadratic in width, so literal co/2 branches would
    carry only ~58% of the reference stage's parameters (the D1 lesson in reverse).  The branch
    width w is instead SOLVED at construction -- smallest |delta| against the unmodified stage
    `block_cls(ci, co)` -- and asserted within 2%, so the arm measures ROUTING, not capacity.
    Trial builds run inside `fork_rng(devices=[])`: the search consumes NO RNG, so the real
    builds below draw exactly the same stream regardless of how many trials the search took.

    The gate bias starts at 0 (m ~= 0.5 everywhere: a reseed, not a disruption); the gate weight
    keeps its default init so the mask is spatially informative from step 0.  `_last_mask` is a
    DETACHED plain attribute (the `_freq_lo` / `_last_aux` convention: never a parameter, never
    saved) for the screen's mask-statistics reading.
    """

    def __init__(self, block_cls, ci, co):
        super().__init__()
        ci, co = int(ci), int(co)

        def _n(m):
            return sum(p.numel() for p in m.parameters())

        def _cost(w):
            # two branches x (block stack at width w + 1x1 lift w->co) + the 1x1 gate conv
            return 2 * (_n(block_cls(ci, w)) + (w * co + co)) + (ci + 1)

        with torch.random.fork_rng(devices=[]):
            target = _n(block_cls(ci, co))
            lo, hi = 4, 4 * co
            while lo < hi:                       # smallest w with _cost(w) >= target
                mid = (lo + hi) // 2
                if _cost(mid) >= target:
                    hi = mid
                else:
                    lo = mid + 1
            w = lo if abs(_cost(lo) - target) <= abs(_cost(lo - 1) - target) else lo - 1
            got = _cost(w)
        assert abs(got - target) / target <= 0.02, \
            f"route: no width within 2% of the reference stage ({got} vs {target}, w={w})"
        self.width = int(w)
        self.ref_params, self.own_params = int(target), int(got)
        self.b1 = nn.Sequential(block_cls(ci, w), nn.Conv2d(w, co, 1))
        self.b2 = nn.Sequential(block_cls(ci, w), nn.Conv2d(w, co, 1))
        self.gate = nn.Conv2d(ci, 1, 1)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x):
        m = torch.sigmoid(self.gate(x))
        self._last_mask = m.detach()
        return m * self.b1(x) + (1.0 - m) * self.b2(x)


class LeadSplitHead(nn.Module):
    """MSMOE (2026-08-27; lead-split output heads at the measured lead-4/5 break, Ms-MoE
    arXiv:2604.12794-inspired).  LEADCORR measured the residual's lead-axis correlation breaking
    at leads 4/5 (corr cliff 0.720 -> 0.149): leads 1..4 are a "riding the observed input"
    regime, 5..20 the divergent one.  The HEAD UNIT (the last decoder stage `c1` plus the
    projection `outc`) is replaced by TWO parallel branches of the SAME block class, each a
    reduced-width copy of that unit: branch A decodes leads 1..L, branch B leads L+1..t_out,
    concatenated in lead-major channel order (bit-compatible with the reshape in forward).
    The trunk (inc..up1) stays shared, so this is head-only specialisation, not a data split.

    PARAMETER MATCHING (the RouteStage w=344 precedent).  Two full-width copies of the head
    unit would add ~3.9% model params, so the branch width w is SOLVED at construction --
    smallest |delta| against the reference unit  block_cls(ci, ref_base) +
    Conv2d(ref_base, t_out*c, 1)  -- and asserted within 2%, so the arm measures LEAD
    SPECIALISATION, not capacity.  Trial builds run inside fork_rng(devices=[]): the search
    consumes NO RNG, so the real builds below draw the same stream regardless of the search.
    """

    def __init__(self, block_cls, ci, ref_base, t_out, c, L):
        super().__init__()
        ci, ref_base, t_out, c, L = int(ci), int(ref_base), int(t_out), int(c), int(L)
        assert 1 <= L < t_out, L

        def _n(m):
            return sum(p.numel() for p in m.parameters())

        def _cost(w):
            # two branch stacks at width w + the two lead-range 1x1 projections:
            # (w*L*c + L*c) + (w*(t_out-L)*c + (t_out-L)*c) == w*t_out*c + t_out*c
            return 2 * _n(block_cls(ci, w)) + (w * t_out * c + t_out * c)

        with torch.random.fork_rng(devices=[]):
            target = _n(block_cls(ci, ref_base)) + (ref_base * t_out * c + t_out * c)
            lo, hi = 4, 4 * ref_base
            while lo < hi:                       # smallest w with _cost(w) >= target
                mid = (lo + hi) // 2
                if _cost(mid) >= target:
                    hi = mid
                else:
                    lo = mid + 1
            w = lo if abs(_cost(lo) - target) <= abs(_cost(lo - 1) - target) else lo - 1
            got = _cost(w)
        assert abs(got - target) / target <= 0.02, \
            f"head_split: no width within 2% of the reference head unit ({got} vs {target}, w={w})"
        self.width, self.L, self.t_out, self.c = int(w), L, t_out, c
        self.ref_params, self.own_params = int(target), int(got)
        self.ha = nn.Sequential(block_cls(ci, w), nn.Conv2d(w, L * c, 1))
        self.hb = nn.Sequential(block_cls(ci, w), nn.Conv2d(w, (t_out - L) * c, 1))

    def forward(self, x):
        return torch.cat([self.ha(x), self.hb(x)], dim=1)


# HYBIN (2026-08-30, Track1/LBM/hybrid) -- per-WINDOW derived planes appended at the `cond` cat
# site, name -> plane count.  Selected by `cond: "stats+<name>"`; absent for every other value, so
# the champion build is unchanged in ci, in construction order and in the state_dict.
#   divms   mean_t(div) and std_t(div) per pixel      -- div = the PIV-LBM mass source S
#   vortms  the same two moments of the PHYSICAL curl -- the mandatory generic-feature control
#   divg    two CONSTANT planes: div steady fraction and rms(div) -- a global reduction the
#           levels-2 trunk has no pooling to compute
# SIGNS.  Measured on this release: dx = +0.00342166 m, dy = -0.00342166 m (the h5 y axis DESCENDS
# and build_windows does not flip it), so in ARRAY-INDEX units
#     div = d u/d i_x - d v/d i_y      vort = d v/d i_x + d u/d i_y
# NOTE the pre-existing `_derive(x, "vort")` computes d v/d i_x - d u/d i_y, which on this grid is
# dv/dx + du/dy = 2*S_xy, the off-diagonal STRAIN rate -- not vorticity.  Harmless for a linear
# plane fed to a conv (arm IAv's verdict stands) but the label is wrong; left untouched so no
# closed arm changes meaning, and the physically correct signs are used below.
_HYBIN_CONDX = {"divms": 2, "vortms": 2, "divg": 2,
                "vortg": 2, "gradg": 2, "shuf": 2, "divk": 5}   # HYBREX 2026-09-09: controls + the stack


class UNet2D(nn.Module):
    """2-level U-net on the frame-stacked tensor. H,W must be divisible by 4."""

    def __init__(self, t_in=20, t_out=20, c=3, base=32, fuse="none", heads=4, act="gelu", out_mode="direct", anchor="mean", dt=0.02, substeps=2, proj_iters=0,
                 mask_rel=0.05,
                 spec_inner=32, spec_modes=(12, 17), in_aug="", in_norm="", block="double",
                 levels=2, drop_path=0.0, layer_scale=0.0, nxt_k=7, nxt_grn=0.0, nxt_gate="", nxt_depth=2, nxt_act="", nxt_pad="", nxt_se=0, norm="group", skip="", stage_res=0, cond="",
                 head="", t_attn=0, t_attn_dim=32, t_attn_heads=4,
                 freq_decomp=0, freq_width=64, freq_layers=2, freq_smooth_n=2, freq_w=0.0,
                 aux_phases=0, route=0, head_split=0, mimo=0, hetero_head=0, cond_film=0,
                 flat_depth=0):
        super().__init__()
        self.t_in, self.t_out, self.c = t_in, t_out, c
        self.out_mode = str(out_mode)
        self.anchor = str(anchor)
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.proj_iters = int(proj_iters)
        self.mask_rel = float(mask_rel)
        if self.anchor == "global":
            # Sample-independent on purpose: this is the ablation that removes regime information
            # from the anchor while keeping the integrated-tendency parametrisation intact.
            self.anchor_c = nn.Parameter(torch.zeros(c))
        if self.out_mode == "solver":
            # Initialised from the measured physics of this grid: the mean flow advects 0.82 cells
            # per frame (CFL), self-advection is a fraction of that, and molecular viscosity is
            # negligible (diffusion number 1.7e-3) so `nu` starts small and may grow into an eddy
            # viscosity -- the closure a 32x64 grid actually needs.
            self.g_self = nn.Parameter(torch.tensor(0.10))
            self.g_mean = nn.Parameter(torch.tensor([0.82, 0.0]))
            self.nu = nn.Parameter(torch.tensor(0.01))
            self.eps_s = nn.Parameter(torch.tensor(1.0))
        assert self.out_mode in ("direct", "inc", "curl", "solver"), self.out_mode
        assert self.anchor in ("mean", "last", "global"), self.anchor
        self.fuse = str(fuse)
        # derived input fields, one extra channel per input frame each; "" = shipped behaviour
        self.in_aug = tuple(s for s in str(in_aug).replace(" ", "").split(",") if s)
        assert all(s in ("vort", "speed", "div") for s in self.in_aug), self.in_aug
        # per-sample normalisation by the INPUT window's own statistics; "" = shipped behaviour
        self.in_norm = str(in_norm)
        assert self.in_norm in ("", "self", "scale", "mean", "vec"), self.in_norm
        self.cond = str(cond)
        # HYBIN: `stats+<k>` for k in _HYBIN_CONDX selects a per-window derived plane block.
        self.condx = (self.cond[6:] if (self.cond.startswith("stats+")
                                        and self.cond[6:] in _HYBIN_CONDX) else "")
        assert self.cond in ("", "stats", "stats+phase") or self.condx, self.cond
        self.head = str(head)
        assert self.head in ("", "split", "fourier"), self.head
        if self.head == "split":
            assert str(out_mode) == "direct", "head=split composes with direct output only"
        ci, co = t_in * (c + len(self.in_aug)), t_out * c
        if self.cond.startswith("stats"):
            ci += 4     # 4 conditioning planes: window mean_u, mean_v, std_u, std_v
        if self.cond == "stats+phase":
            ci += 2 * int(t_out)   # FPa: a*cos(theta_k), a*sin(theta_k) per output lead
        if self.condx:
            ci += _HYBIN_CONDX[self.condx]      # HYBIN: 2 per-window planes, appended LAST
        assert str(block) in ("double", "convnext", "selfwarp", "ssm", "swin"), block
        self.levels = int(levels)
        assert self.levels in (1, 2, 3), levels
        # EQFREE TOOTH (2026-09-09): training-only crop state, set per batch by train_v4 and
        # cleared after every step.  None (the default, and always at eval) is the shipped
        # forward, untouched.  A plain attribute: never a parameter, never saved.
        self._tooth = None
        global _NEXT_DP, _NEXT_LS, _NEXT_K, _NEXT_GRN, _NEXT_GATE, _NEXT_DEPTH, _NEXT_ACT, _NEXT_NORM, _NEXT_PAD, _NEXT_SE
        # NORMACT-N.  `norm` is threaded into the DoubleConv/NeXtBlock slot ONLY.  A
        # non-default norm with a block whose norm sites are NOT threaded RAISES here:
        # a knob that silently covered only some of its apparent sites would itself be a
        # new instance of this repo's most expensive defect class.
        self.norm = str(norm or "group")
        assert self.norm in NORM_KINDS, f"norm={self.norm!r} not in {NORM_KINDS}"
        if self.norm != "group" and str(block) not in ("double", "convnext"):
            raise ValueError(
                f"norm={self.norm!r} is threaded through the DoubleConv/NeXtBlock slot "
                f"only; block={block!r} keeps hardcoded GroupNorm at its own sites, so "
                f"this build would be a PARTIAL swap masquerading as a full one.  Refusing.")
        _NEXT_NORM = self.norm
        _NEXT_DP, _NEXT_LS = float(drop_path or 0.0), float(layer_scale or 0.0)
        _NEXT_K, _NEXT_GRN = int(nxt_k or 7), float(nxt_grn or 0.0)
        _NEXT_GATE = str(nxt_gate or "")
        _NEXT_DEPTH = int(nxt_depth or 2)
        _NEXT_ACT = str(nxt_act or "")
        # ATTNUNET-SE.  Unconditional, exactly like _NEXT_ACT above: a second build in the same
        # process with no nxt_se resets to 0 rather than inheriting.
        _NEXT_SE = int(nxt_se or 0)
        assert _NEXT_SE >= 0, nxt_se
        if _NEXT_SE and str(block) != "convnext":
            raise ValueError(f"nxt_se={_NEXT_SE} is threaded into NeXtBlock ONLY; block={block!r} "
                             f"would be a PARTIAL swap masquerading as a full one.  Refusing.")
        self.nxt_se = _NEXT_SE
        # IFACE-PAD.  Unconditional, exactly like _NEXT_K / _NEXT_ACT above: a second build in
        # the same process with no nxt_pad resets to "zeros" rather than inheriting.
        _NEXT_PAD = str(nxt_pad or "zeros")
        assert _NEXT_PAD in ("zeros", "reflect", "replicate", "circular"), _NEXT_PAD
        if _NEXT_PAD != "zeros" and str(block) != "convnext":
            raise ValueError(f"nxt_pad={_NEXT_PAD!r} is threaded into NeXtBlock.dw ONLY; "
                             f"block={block!r} keeps its own hardcoded padding, so this build "
                             f"would be a PARTIAL swap masquerading as a full one.  Refusing.")
        _B = _block(act)
        if str(block) == "convnext":
            _B = NeXtConv
        elif str(block) == "selfwarp":
            _B = SelfWarpConv
        elif str(block) == "ssm":
            _B = SSMConv
        elif str(block) == "swin":
            _B = SwinConv
        self.skip = str(skip or "")
        assert self.skip in ("", "gate", "unet3p", "scse"), self.skip
        self.stage_res = bool(int(stage_res or 0))
        _W = (lambda m, a, b: StageRes(m, a, b)) if self.stage_res else (lambda m, a, b: m)
        # MWAVE MIMOA (2026-08-27): MIMO implicit M-head ensemble (Havasi arXiv:2010.06610).
        # mimo=0 (default/absent) is BIT-IDENTICAL off: ci/co and every construction line are
        # unchanged, no new module, no RNG draw.  mimo=M>=2: the stem consumes M channel-stacked
        # windows (each slot gets its own in_norm stats + cond planes, so ci scales by M) and
        # outc emits M stacked predictions (co scales by M at the outc line below); forward
        # dispatches to _forward_mimo, which mirrors exactly the composition pinned here.
        # Cross-knob asserts use the RAW args (route/head_split/hetero_head are assigned later).
        # MIMOSCALE (2026-08-27) generalized M from the hardcoded 2; the M=2 and M=0 paths are
        # BIT-IDENTICAL to the pre-generalization file (proven: analysis/mimoscale.py `null`).
        # Pre-registration: analysis/mwave_screen.py (MWAVE LOCK), analysis/mimoscale.py.
        self.mimo = int(mimo or 0)
        assert self.mimo == 0 or 2 <= self.mimo <= 8, mimo
        if self.mimo:
            assert self.out_mode == "direct" and self.head == "" and self.anchor == "mean", \
                "mimo composes with the plain direct head only"
            assert self.fuse == "none" and not self.in_aug and self.in_norm in ("", "self") \
                and self.cond in ("", "stats"), (self.fuse, self.in_aug, self.in_norm, self.cond)
            assert self.levels == 2, "mimo: _forward_mimo mirrors the levels=2 trunk only"
            assert self.skip == "" and not self.stage_res, \
                "mimo does not compose with skip variants / stage_res"
            assert not int(route or 0) and not int(head_split or 0) and not int(hetero_head or 0), \
                "mimo does not compose with route / head_split / hetero_head"
            assert not int(t_attn or 0) and not int(freq_decomp or 0) and not int(aux_phases or 0), \
                "mimo does not compose with t_attn / freq_decomp / aux_phases"
            ci *= self.mimo
        self.inc = _W(_B(ci, base), ci, base)
        # EQFREE LOCALNO (2026-09-09): levels=1 is a FLAT stack with NO pyramid -- the paper's
        # 'local neural operator' as a 20->20 block map: inc + flat_depth NeXtConv stages at full
        # resolution, receptive field 1 + 6*(2*(1+flat_depth)) px at nxt_k=7 / _NEXT_DEPTH=2, so
        # it is bounded (never global in the streamwise 64 px for flat_depth <= 4).  levels=2/3
        # builds are BIT-IDENTICAL: flat_depth=0 creates no module and draws no RNG.
        self.flat_depth = int(flat_depth or 0)
        if self.levels == 1:
            assert self.flat_depth >= 1, "levels=1 needs fuse: {flat_depth: K>=1}"
            assert self.fuse == "none" and self.skip == "" and not self.stage_res \
                and not int(route or 0) and not int(head_split or 0) and not int(hetero_head or 0) \
                and not int(aux_phases or 0) and not int(mimo or 0) and not int(freq_decomp or 0) \
                and not int(cond_film or 0) and self.out_mode == "direct" and self.head == "" \
                and not int(t_attn or 0), "levels=1 composes with the plain direct trunk only"
            self.flat = nn.Sequential(*[_W(_B(base, base), base, base) for _ in range(self.flat_depth)])
        else:
            assert not self.flat_depth, "flat_depth is a levels=1 knob"
        self.d1 = _W(_B(base, base * 2), base, base * 2)
        # RROUTE: default 0 is the historical path -- the exact same construction line, no module
        # created, no extra RNG draw, so a route=0 build is BIT-identical to a build without the
        # knob.  route=1 replaces only the deepest stage (levels=2's d2) with the routed pair.
        self.route = int(route or 0)
        assert self.route in (0, 1), route
        if self.route:
            assert self.levels == 2, "route=1 replaces the deepest stage; implemented for levels=2 only"
            assert not self.stage_res, "route=1 does not compose with stage_res"
            self.d2 = RouteStage(_B, base * 2, base * 4)
        else:
            self.d2 = _W(_B(base * 2, base * 4), base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        # U3Pa: every decoder level also receives EVERY other encoder scale, rescaled -- the
        # "full-scale skip connection" of UNet 3+ (Huang et al. 2020, arXiv:2004.08790).
        _c2_in = base * 6 if self.skip == "unet3p" else base * 4
        _c1_in = base * 3 if self.skip == "unet3p" else base * 2
        self.c2 = _W(_B(_c2_in, base * 2), _c2_in, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        # MSMOE: default 0 is the historical path -- the exact same construction line at its
        # original position, no module created, no extra RNG draw, so a head_split=0 build is
        # BIT-identical to a build without the knob.  head_split=L>0 replaces the HEAD UNIT
        # (c1 + outc) with LeadSplitHead's two parameter-matched reduced-width branches
        # (leads 1..L / L+1..t_out); c1 becomes a parameter-free Identity so the forward path
        # is untouched, and `outc` (the name train_v4's LP-FT head group reads) carries every
        # head parameter.  Built HERE, not at the outc line, so the branch blocks see the same
        # _NEXT_* globals the c1 stage would have seen.
        self.head_split = int(head_split or 0)
        assert 0 <= self.head_split < int(t_out), head_split
        if self.head_split:
            assert self.levels == 2, "head_split replaces the last decoder stage; levels=2 only"
            assert not self.stage_res, "head_split does not compose with stage_res"
            assert self.skip == "", "head_split does not compose with skip variants"
            assert self.out_mode == "direct" and self.head == "", \
                "head_split composes with the plain direct head only"
            assert not int(freq_decomp or 0) and not int(aux_phases or 0) and not self.route, \
                "head_split does not compose with freq_decomp / aux_phases / route"
            self.c1 = nn.Identity()
            self.outc = LeadSplitHead(_B, _c1_in, base, t_out, c, self.head_split)
        else:
            self.c1 = _W(_B(_c1_in, base), _c1_in, base)
        # MWAVE CRPSH (2026-08-27): heteroscedastic (mean, logvar) head.  hetero_head=0
        # (default/absent) is BIT-IDENTICAL off: no module, no RNG draw.  hetero_head=1 adds a
        # SECOND HEAD UNIT (the MSMOE head-unit precedent: a full-width block stage + 1x1
        # projection, reading the same [u1, s0] input as c1) that emits the per-(lead,pixel,
        # channel) LOG-VARIANCE.  It is an ADDITION (the REFz precedent): the MEAN path
        # (trunk + c1 + outc) is parameter-identical to the control, so a rel change cannot
        # be a capacity artifact.  Constructed HERE, before the _NEXT_* reset below, so the
        # block sees the same globals c1 saw.  Pre-registration: mwave_screen.py (MWAVE LOCK).
        self.hetero_head = int(hetero_head or 0)
        assert self.hetero_head in (0, 1), hetero_head
        if self.hetero_head:
            assert self.out_mode == "direct" and self.head == "", \
                "hetero_head composes with the plain direct head only"
            assert self.skip == "" and not self.stage_res and not self.head_split, \
                "hetero_head does not compose with skip variants / stage_res / head_split"
            assert not int(route or 0) and not int(freq_decomp or 0) \
                and not int(aux_phases or 0) and not int(mimo or 0), \
                "hetero_head does not compose with route / freq_decomp / aux_phases / mimo"
            self.outc_lv = nn.Sequential(_B(_c1_in, base), nn.Conv2d(base, t_out * c, 1))
        if self.skip == "unet3p":
            self.fs_d2 = nn.Conv2d(base, base * 2, 1)        # x0 pooled  -> decoder level 1
            self.fs_u1 = nn.Conv2d(base * 2, base, 1)        # x1 upsampled -> decoder level 0
        if self.skip == "gate":
            self.ag2 = AttnGate(base * 2, base * 2)
            self.ag1 = AttnGate(base, base)
        if self.skip == "scse":
            # ATTNUNET-SCSE: private-generator init, no global RNG draw (see SCSEGate).
            self.sc2 = SCSEGate(base * 2)
            self.sc1 = SCSEGate(base)
        if self.levels == 3:
            self.d3 = _B(base * 4, base * 8)
            self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
            self.c3 = _B(base * 8, base * 4)
        _NEXT_DP, _NEXT_LS, _NEXT_K, _NEXT_GRN = 0.0, 0.0, 7, 0.0   # reset for later builds
        _NEXT_PAD = "zeros"                                         # IFACE-PAD: same discipline
        _NEXT_NORM = "group"                                        # NORMACT-N: same reset discipline
        _NEXT_SE = 0                                                # ATTNUNET-SE: same reset discipline
        if not self.head_split:      # MSMOE: the split head above already IS the projection
            # MWAVE MIMOA: co scales by M when mimo=M (M stacked head predictions); the
            # mimo=0 expression evaluates to exactly `co`, same shapes, same RNG draws.
            self.outc = nn.Conv2d(base, co * (self.mimo if self.mimo else 1), 1)
        if self.levels == 1:
            # EQFREE LOCALNO: the pyramid was constructed above (same code path as every build);
            # it is never used by the levels=1 forward, so drop it here -- state_dict, parameter
            # count and the optimizer then see the flat trunk only, and build_from_meta(levels=1)
            # rebuilds the identical module set for a strict load.
            for _n in ("d1", "d2", "pool", "up2", "c2", "up1", "c1"):
                delattr(self, _n)
        if self.head in ("split", "fourier"):
            self.outc_m = nn.Conv2d(base, c, 1)
        if self.head == "fourier":
            # temporal Fourier basis, k=1..K, no DC -> zero-mean along T by construction.
            # sin_K at K=t_out//2 is identically zero on the integer grid — one dead
            # column kept for shape simplicity (documented, harmless).
            import math
            K = t_out // 2
            tt = torch.arange(t_out, dtype=torch.float32)
            cols = []
            for k in range(1, K + 1):
                cols.append(torch.cos(2 * math.pi * k * tt / t_out))
                cols.append(torch.sin(2 * math.pi * k * tt / t_out))
            B_ = torch.stack(cols, 1)          # (t_out, 2K) == (t_out, t_out)
            self.register_buffer("four_basis", B_)
        if self.out_mode != "direct":
            # Residual start: with a zero head the model returns the anchor exactly, which for the
            # input-window time-mean is already a reasonable forecast.  Learning then only has to
            # find the correction, which is the point of an integrated tendency.
            nn.init.zeros_(self.outc.weight)
            nn.init.zeros_(self.outc.bias)
        # both default OFF; with fuse="none" every buffer below is absent and the module is
        # state_dict-identical to the shipped model.
        # FDa: frequency-decomposed head.  Wrapping `self.outc` keeps the conv path byte-identical
        # and adds a pointwise-MLP branch beside it; absent unless asked for.
        self.freq_smooth_n = int(freq_smooth_n or 0)
        self.freq_w = float(freq_w or 0.0)
        self.freq_hi = None
        if int(freq_decomp or 0):
            _in = getattr(self.outc, "in_channels", base)
            _ot = getattr(self.outc, "out_channels", t_out * c)
            self.freq_hi = PointwiseMLP(_in, _ot, width=freq_width, layers=freq_layers, act=act)
        # TAa: the only operator in this model that sees the input window as a SEQUENCE.
        # Absent unless asked for, so the shipped state_dict is unchanged.
        self.tattn = (TemporalAttn(c, t=t_in, dim=int(t_attn_dim), heads=int(t_attn_heads))
                      if int(t_attn or 0) else None)
        self.attn = BottleneckAttn(base * 4, heads) if self.fuse in ("attn", "both") else None
        self.spec = (SpectralBranch(base, spec_inner, *spec_modes)
                     if self.fuse in ("spec", "both") else None)
        # AXP (2026-08-12): auxiliary sub-phase head.  The 64x128 release is eval-subsampled to
        # 32x64 at phase (0,0), so 3/4 of the measured pixels never enter training; this head
        # predicts the other `aux_phases` phase-shifted subsamplings of the SAME target window,
        # supervised at a small weight during training only (train_v4 `aux_phase_weight`) and
        # unused at inference.  MODULE CREATION IS GATED on aux_phases > 0 so the default build
        # is state_dict-identical (and RNG-identical) to the shipped model, every existing
        # checkpoint loads unchanged, and the head is a SIBLING of `self.outc` (the FreqHead
        # lesson: never wrap/rename existing parameters).
        self.aux_phases = int(aux_phases or 0)
        self.aux_head = None
        if self.aux_phases:
            assert self.out_mode == "direct" and self.head == "" and self.freq_hi is None, \
                ("aux_phases composes with the plain direct head only "
                 f"(out_mode={self.out_mode!r}, head={self.head!r}, freq_decomp={freq_decomp!r})")
            self.aux_head = nn.Conv2d(base, self.aux_phases * t_out * c, 1)

        if self.cond == "stats+phase":
            # STRICT SUPERSET: the FPa planes are appended LAST, so zeroing exactly those stem
            # input channels makes the network bit-identical to the cond:stats champion at step 0.
            # The TMa protocol -- the arm can only differ by what training does with the planes.
            _first = None
            for _mod in self.inc.modules():
                if isinstance(_mod, nn.Conv2d):
                    _first = _mod
                    break
            assert _first is not None and _first.in_channels == ci, \
                f"FPa zero-init: stem conv in_channels {getattr(_first, 'in_channels', None)} != ci {ci}"
            with torch.no_grad():
                _first.weight[:, -2 * int(t_out):].zero_()

        if self.condx:
            # HYBIN, the FPa/TMa protocol verbatim: the added planes are appended LAST, so zeroing
            # exactly those stem input columns makes this arm BIT-IDENTICAL to the cond:stats
            # champion at step 0.  The arm can then only differ by what TRAINING does with them.
            _first = None
            for _mod in self.inc.modules():
                if isinstance(_mod, nn.Conv2d):
                    _first = _mod
                    break
            assert _first is not None and _first.in_channels == ci, \
                f"HYBIN zero-init: stem conv in_channels {getattr(_first, 'in_channels', None)} != ci {ci}"
            with torch.no_grad():
                _first.weight[:, -_HYBIN_CONDX[self.condx]:].zero_()

        # CONDFILM (2026-09-02, Track1/CONDFILM): five CondFiLM built as the LAST lines of __init__
        # so every construction line above draws the same RNG as the control; cond_film=0 builds
        # NOTHING (no module, no Identity, no RNG draw -- proven by
        # Track1/CONDFILM/analysis/condfilm_null.py).  `cond_tap` is an Identity the sensitivity
        # witness hooks (an `inc` pre-hook rewrites the PLANES and never reaches this vector).
        self.cond_film = int(cond_film or 0)
        assert self.cond_film in (0, 1), cond_film
        if self.cond_film:
            assert self.cond == "stats", f"cond_film needs cond: stats (got {self.cond!r})"
            assert self.fuse == "none" and not self.in_aug and self.in_norm in ("", "self"), \
                (self.fuse, self.in_aug, self.in_norm)
            assert self.levels == 2 and self.skip == "" and not self.stage_res and not self.route \
                and not self.head_split and not self.mimo and not self.hetero_head, \
                "cond_film composes with the plain levels=2 trunk only"
            assert self.out_mode == "direct" and self.head == "" and not int(t_attn or 0) \
                and not int(freq_decomp or 0) and not int(aux_phases or 0), \
                "cond_film composes with the plain direct head only"
            self.film = nn.ModuleDict({"inc": CondFiLM(base), "d1": CondFiLM(base * 2),
                                       "d2": CondFiLM(base * 4), "c2": CondFiLM(base * 2),
                                       "c1": CondFiLM(base)})
            self.cond_tap = nn.Identity()
            self._last_film = {}

    def _film_apply(self, site, t, cv):
        """CONDFILM: t * (1 + gamma(c)) + beta(c); gamma/beta stashed DETACHED in _last_film (the
        _last_logvar convention: never a parameter, never saved) for the train_v4 LIVE witness."""
        g, bt = self.film[site](cv)
        g = g.reshape(t.shape[0], -1, 1, 1)
        bt = bt.reshape(t.shape[0], -1, 1, 1)
        self._last_film[site] = (g.detach(), bt.detach())
        return t * (1.0 + g) + bt

    def _phase_planes(self, x, b, h, w):
        """FPa: (B, 2*t_out, H, W) amplitude-gated shedding-phase clock, from the window alone.

        RELF localised the whole remaining rel pool as large-scale PHASE at late leads, and nothing
        in this network is told the shedding frequency: the 20 frames are flattened into channels,
        and `cond: stats` carries only mean/std.  So the stack must infer the phase AND extrapolate
        it 20 frames.  Here it is handed cos/sin of the phase already extrapolated to each output
        lead, which turns extrapolation into interpolation.

        Probe = the mid-wake station the mvpe metric itself reads, on the transverse velocity v,
        whose oscillation IS the vortex shedding.  Phase is invariant to `in_norm: self`, so it
        does not matter that x has already been normalised.  The gate is the dominant harmonic's
        share of the signal's amplitude, so a window with no detectable shedding contributes ~0
        instead of injecting a meaningless phase.
        """
        t = int(x.shape[1])
        py, px = int(h // 2), min(int(round(29 * w / 64.0)), int(w) - 1)
        s = x[:, :, py, px, 1]
        s = s - s.mean(dim=1, keepdim=True)
        Fc = torch.fft.rfft(s, dim=1)[:, 1:]
        mag = Fc.abs()
        idx = mag.argmax(dim=1).reshape(-1, 1)
        cm = torch.gather(Fc, 1, idx).reshape(-1)
        phi = torch.angle(cm)
        freq = (idx.reshape(-1).to(x.dtype) + 1.0) / float(t)
        gate = (mag.gather(1, idx).reshape(-1)
                / mag.pow(2).sum(1).sqrt().clamp_min(1e-8)).clamp(0.0, 1.0)
        k = torch.arange(1, int(self.t_out) + 1, device=x.device, dtype=x.dtype).reshape(1, -1)
        th = phi.reshape(-1, 1) + 2.0 * math.pi * freq.reshape(-1, 1) * (float(t) - 1.0 + k)
        g = gate.reshape(-1, 1)
        pl = torch.cat([g * torch.cos(th), g * torch.sin(th)], 1)
        return pl.reshape(b, -1, 1, 1).expand(b, pl.shape[1], h, w)

    def _solve(self, x, src):
        """Roll the coarse solver forward 20 frames, driven by the network's source fields.

        `x`   (B,T_in,H,W,C) the normalised input window -- supplies the initial state and the mask
        `src` (B,T_out,H,W,C) the network's output, read as (s_u, s_v) per frame
        """
        from solver_ops import body_mask, laplace, upwind, project
        mask = body_mask(x, self.mask_rel)                       # (B,1,H,W)
        keep = 1.0 - mask
        u = (x[:, -1, :, :, 0] * keep[:, 0]).unsqueeze(1)        # last MEASURED state, not the mean
        v = (x[:, -1, :, :, 1] * keep[:, 0]).unsqueeze(1)
        gm = self.g_mean.reshape(2)
        hsub = 1.0 / float(self.substeps)
        outs = []
        for j in range(self.t_out):
            su = src[:, j, :, :, 0].unsqueeze(1) * self.eps_s
            sv = src[:, j, :, :, 1].unsqueeze(1) * self.eps_s
            for _ in range(self.substeps):
                cu = (self.g_self * u + gm[0]) * hsub
                cv = (self.g_self * v + gm[1]) * hsub
                au, av = upwind(u, cu, cv), upwind(v, cu, cv)
                u = u + au + hsub * (self.nu * laplace(u) + su)
                v = v + av + hsub * (self.nu * laplace(v) + sv)
                u, v = u * keep, v * keep
            if self.proj_iters > 0:
                u, v = project(u, v, self.proj_iters, mask)
                u, v = u * keep, v * keep
            outs.append(torch.cat([u, v], dim=1))
        o = torch.stack(outs, dim=1).permute(0, 1, 3, 4, 2)      # (B,T_out,H,W,2)
        if src.shape[-1] > 2:
            o = torch.cat([o, torch.zeros_like(src[..., 2:])], dim=-1)
        return o

    def forward(self, x):
        if getattr(self, "mimo", 0):
            # MWAVE MIMOA: M-slot forward (train: M stacked windows; eval: the same window
            # in every slot, heads averaged).  The single-window path below is untouched.
            return self._forward_mimo(x)
        b, t, h, w, c = x.shape
        _m = _s = None
        if self.in_norm:
            # statistics of the INPUT WINDOW ONLY -- available at predict time.  Undone on the
            # output below, so the network is exactly equivariant to the flow's own scale and a
            # Reynolds number outside the training sweep reaches the convolutions looking like one
            # inside it.
            if self.in_norm == "mean":
                # Galilean only: exactly equivariant to x -> x + b, and to nothing else.
                _m = x.mean(dim=(1, 2, 3), keepdim=True)
                _s = None
                x = x - _m
            elif self.in_norm == "vec":
                # The physical group: ONE velocity scale shared by u and v (a change of units),
                # plus a per-channel offset (a change of frame).  Dividing u and v by separate
                # stds -- what "self" does -- would force sd_v/sd_u to 1 for every window, and
                # that ratio runs 0.111 -> 0.254 across AoA 0 -> 20 in this dataset, i.e. it is
                # the leading amplitude signature of the very regime variable we hold out.
                _m = x.mean(dim=(1, 2, 3), keepdim=True)
                xc = x - _m
                _s = xc.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)   # per-channel, for p
                if c >= 2:
                    # p is a different physical quantity and is identically zero on real data, so
                    # it keeps its own scale rather than diluting the velocity one.
                    sv = xc[..., :2].std(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)
                    _s = torch.cat([sv.expand(-1, -1, -1, -1, 2), _s[..., 2:]], dim=-1)
                x = xc / _s
            else:
                _s = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
                if self.in_norm == "self":
                    _m = x.mean(dim=(1, 2, 3), keepdim=True)
                    x = (x - _m) / _s
                else:
                    x = x / _s
        if self.tattn is not None:
            # Applied to the NORMALISED window, before the channel stacking that destroys the
            # time axis.  Residual and zero-init, so this line is an exact no-op at step 0.
            x = self.tattn(x)
        z = x.permute(0, 1, 4, 2, 3).reshape(b, t * c, h, w)   # (B, t_in*c, H, W)
        if self.in_aug:
            z = torch.cat([z] + [self._derive(x, s) for s in self.in_aug], 1)
        _cf, _cv = int(getattr(self, "cond_film", 0) or 0), None   # CONDFILM
        if self.cond.startswith("stats"):
            # NCa: in_norm strips the window's scale coordinates (u-mean tracks Re,
            # sd_v/sd_u tracks AoA); feed them back as explicit constant planes so the
            # stack stays scale-equivariant AND sees the regime.  With in_norm unset the
            # fallback stats are post-norm (well-defined, near-constant) — cond is meant
            # to ride on in_norm=self.
            cm = _m if (self.in_norm and _m is not None) else x.mean(dim=(1, 2, 3), keepdim=True)
            cs = _s if (self.in_norm and _s is not None) else x.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
            planes = torch.cat([cm[..., :2].reshape(b, 2, 1, 1),
                                cs[..., :2].reshape(b, 2, 1, 1)], 1).expand(b, 4, h, w)
            z = torch.cat([z, planes], 1)
            if _cf:   # CONDFILM: the same four scalars, as a (B,4) vector through cond_tap
                _cv = self.cond_tap(torch.cat([cm[..., :2].reshape(b, 2), cs[..., :2].reshape(b, 2)], 1))
        if self.cond == "stats+phase":
            z = torch.cat([z, self._phase_planes(x, b, h, w)], 1)
        if self.condx:
            z = torch.cat([z, self._condx_planes(x, b, h, w)], 1)
        _tooth = getattr(self, "_tooth", None)
        if _tooth is not None:
            # EQFREE TOOTH: crop the stacked input (frames + derived channels + cond planes) to
            # one random tooth+halo box PER SAMPLE, AFTER the window statistics above were taken
            # on the FULL window (in_norm undo below therefore uses full-window stats too).
            assert self.out_mode == "direct" and self.head == "" and not getattr(self, "mimo", 0) \
                and self.freq_hi is None and self.aux_head is None \
                and not getattr(self, "hetero_head", 0), "tooth composes with the plain direct head only"
            _bx = _tooth["boxes"]
            _hin, _win = int(_tooth["hin"]), int(_tooth["win"])
            assert tuple(_bx.shape) == (b, 2), (tuple(_bx.shape), b)
            z = torch.stack([z[i, :, int(_bx[i, 0]):int(_bx[i, 0]) + _hin,
                               int(_bx[i, 1]):int(_bx[i, 1]) + _win] for i in range(b)], 0)
            h, w = _hin, _win
        x0 = self.inc(z)                     # base,   H,   W
        if _cf:
            x0 = self._film_apply("inc", x0, _cv)   # CONDFILM
        if self.spec is not None:
            x0 = x0 + self.spec(x0)          # full-resolution Fourier branch, no k^-2 bias
        if self.levels == 1:
            # EQFREE LOCALNO: no pyramid -- the flat receptive-field-bounded stack (see __init__).
            y = self.flat(x0)
            u1 = s0 = None
        else:
            x1 = self.d1(self.pool(x0))          # 2base,  H/2, W/2
            if _cf:
                x1 = self._film_apply("d1", x1, _cv)    # CONDFILM
            x2 = self.d2(self.pool(x1))          # 4base,  H/4, W/4
            if _cf:
                x2 = self._film_apply("d2", x2, _cv)    # CONDFILM
            if self.levels == 3:
                x3 = self.d3(self.pool(x2))      # 8base,  H/8, W/8
                x2 = self.c3(torch.cat([self.up3(x3), x2], 1))
            if self.attn is not None:
                x2 = self.attn(x2)               # global context the 2-level RF cannot reach
            u2 = self.up2(x2)
            s1 = self.ag2(x1, u2) if self.skip == "gate" else (self.sc2(x1) if self.skip == "scse" else x1)
            if self.skip == "unet3p":
                y = self.c2(torch.cat([u2, s1, self.fs_d2(self.pool(x0))], 1))
            else:
                y = self.c2(torch.cat([u2, s1], 1))
            if _cf:
                y = self._film_apply("c2", y, _cv)      # CONDFILM
            u1 = self.up1(y)
            s0 = self.ag1(x0, u1) if self.skip == "gate" else (self.sc1(x0) if self.skip == "scse" else x0)
            if self.skip == "unet3p":
                _up = F.interpolate(self.fs_u1(x1), size=x0.shape[-2:], mode="nearest")
                y = self.c1(torch.cat([u1, s0, _up], 1))
            else:
                y = self.c1(torch.cat([u1, s0], 1))
            if _cf:
                y = self._film_apply("c1", y, _cv)      # CONDFILM
        out = self.outc(y)                   # t_out*c, H, W
        if self.freq_hi is not None:
            # FDa: conv head carries the smooth field, a pointwise MLP carries the rough part.
            # `_freq_lo` is a plain attribute (not a buffer/parameter) so the split-supervision loss
            # can reach the low branch alone; it is overwritten every forward and never saved.
            self._freq_lo = out
            out = out + self.freq_hi(y)
        out = out.reshape(b, self.t_out, c, h, w).permute(0, 1, 3, 4, 2)   # (B,t_out,H,W,c)
        if self.head == "split":
            # DCa: outc carries the ZERO-MEAN fluctuation (mean removed along T), outc_m
            # the window time-mean — the scorer's own decomposition (tke reads only the
            # former, mvpe only the latter), so the two targets stop trading inside one
            # parameter set.
            out = out - out.mean(dim=1, keepdim=True)
            om = self.outc_m(y).reshape(b, 1, c, h, w).permute(0, 1, 3, 4, 2)
            out = out + om
        elif self.head == "fourier":
            # DCb: `out` slots are COEFFICIENTS of the temporal Fourier basis (zero-mean
            # reconstruction), plus the DCa mean head.
            fl = torch.einsum("tk,bkhwc->bthwc", self.four_basis, out)
            om = self.outc_m(y).reshape(b, 1, c, h, w).permute(0, 1, 3, 4, 2)
            out = fl + om
        if self.out_mode != "direct":
            # INC (arXiv:2511.12764): the net emits a TENDENCY; the output is its running integral.
            # `dt` is a CONDITIONING constant, not the physical timestep -- it scales both the head's
            # output and the gradient reaching it.  Measured on fold 0, it barely matters (val rel
            # 0.05311 / 0.05314 / 0.05354 for dt 1.0 / 0.05 / 0.02), so the shipped arm uses 1.0,
            # which matches the direct head's gradient scale.  An earlier revision claimed dt was
            # "absorbed into the net's output scale"; that was wrong and is why the first attempt
            # diverged.
            if self.out_mode == "curl":
                # Wandel ICLR'21: channel 0 is a stream function; velocity is its curl, so the
                # increment this model adds is divergence-free by construction.  Remaining channels
                # (pressure, unscored for real data) pass through untouched.
                psi = out[..., 0].reshape(b * self.t_out, 1, h, w)
                uv = rot_mac(psi).reshape(b, self.t_out, 2, h, w).permute(0, 1, 3, 4, 2)
                out = torch.cat([uv, out[..., 2:]], dim=-1) if c > 2 else uv
            if self.out_mode == "solver":
                out = self._solve(x, out)
            else:
                step = self.dt * torch.cumsum(out, dim=1)
                if self.anchor == "mean":
                    base_f = x.mean(dim=1, keepdim=True)
                elif self.anchor == "last":
                    base_f = x[:, -1:, ...]
                else:
                    base_f = self.anchor_c.reshape(1, 1, 1, 1, -1).expand(b, 1, h, w, c)
                out = base_f[..., :out.shape[-1]] + step
        if self.in_norm:
            if _s is not None:
                out = out * _s
            if _m is not None:
                out = out + _m
        if getattr(self, "hetero_head", 0):
            # MWAVE CRPSH: per-element log-variance from the second head unit (same [u1, s0]
            # input as c1; hetero_head asserts skip == "" so s0 is x0).  The in_norm undo is
            # applied in LOG space (sigma scales with _s => logvar += 2 log _s), making the
            # head scale-equivariant exactly like the mean path.  Stashed as a plain
            # attribute (the _last_aux convention: never a parameter, never saved); the
            # RETURNED prediction stays the MEAN -- the submission-relevant output.
            lv = (self.outc_lv(torch.cat([u1, s0], 1))
                  .reshape(b, self.t_out, c, h, w).permute(0, 1, 3, 4, 2))
            if self.in_norm and _s is not None:
                lv = lv + 2.0 * torch.log(_s)
            self._last_logvar = lv
        if self.aux_head is not None:
            # AXP: aux predictions for the sibling sub-phases, from the same final decoder
            # feature `y` the main head reads.  Same output convention as the main path
            # (direct head + in_norm undo), stashed as a plain attribute exactly like
            # RefineNet._last_y1 / self._freq_lo -- never a parameter, never saved.
            ax = (self.aux_head(y)
                  .reshape(b, self.aux_phases, self.t_out, c, h, w)
                  .permute(0, 1, 2, 4, 5, 3))          # (B, P, t_out, H, W, c)
            if self.in_norm:
                if _s is not None:
                    ax = ax * _s.unsqueeze(1)
                if _m is not None:
                    ax = ax + _m.unsqueeze(1)
            self._last_aux = ax
        if _tooth is not None:
            # EQFREE TOOTH: return the INNER tooth (halo excluded from the loss).
            _hl = int(_tooth["halo"])
            out = out[:, :, _hl:h - _hl, _hl:w - _hl]
        return out

    def _forward_mimo(self, x):
        """MWAVE MIMOA (Havasi arXiv:2010.06610): the MIMO M-slot forward.

        Train: x is (B, t_in, H, W, M*c) -- M INDEPENDENT windows channel-stacked on the last
        axis; returns (B, t_out, H, W, M*c) with head k predicting slot k's future.  Eval: x
        is the standard (B, t_in, H, W, c); the SAME window fills every slot and the heads'
        predictions are AVERAGED to the standard (B, t_out, H, W, c) -- the MIMO test-time
        rule -- so every existing eval/scoring path works unchanged.  Each slot gets its own
        in_norm stats + cond planes (ci scaled by M at construction); each head's output is
        un-normalised with its OWN slot's stats.  The composition surface is pinned in
        __init__ (direct head, levels=2, skip "", fuse none, cond in ("", "stats"), in_norm
        in ("", "self")) and this method mirrors exactly that trunk path.

        MIMOSCALE (2026-08-27): M was hardcoded to 2 until this edit.  At M=2 every line
        below is bit-identical to the pre-generalization file -- the slot slices
        x[..., 0:c] / x[..., c:2c] are the same tensors as x[..., :c] / x[..., c:], and the
        head average is the same product (see the fold at the end).
        """
        b, t, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        c = self.c
        M = self.mimo
        if x.shape[-1] == c:
            slots, paired = tuple(x for _ in range(M)), False
        else:
            assert x.shape[-1] == M * c, (x.shape, M)
            slots, paired = tuple(x[..., k * c:(k + 1) * c] for k in range(M)), True
        zs, ms, ss = [], [], []
        for xi in slots:
            _m = _s = None
            if self.in_norm == "self":
                _s = xi.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
                _m = xi.mean(dim=(1, 2, 3), keepdim=True)
                xi = (xi - _m) / _s
            z = xi.permute(0, 1, 4, 2, 3).reshape(b, t * c, h, w)
            if self.cond == "stats":
                cm = _m if (self.in_norm and _m is not None) \
                    else xi.mean(dim=(1, 2, 3), keepdim=True)
                cs = _s if (self.in_norm and _s is not None) \
                    else xi.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
                planes = torch.cat([cm[..., :2].reshape(b, 2, 1, 1),
                                    cs[..., :2].reshape(b, 2, 1, 1)], 1).expand(b, 4, h, w)
                z = torch.cat([z, planes], 1)
            zs.append(z)
            ms.append(_m)
            ss.append(_s)
        z = torch.cat(zs, 1)
        _tooth = getattr(self, "_tooth", None)
        if _tooth is not None:
            # EQFREE TOOTHM (2026-09-09): the same per-sample tooth+halo crop as forward(), applied to the
            # M-slot stack AFTER every slot's FULL-window stats were taken (the un-normalisation below
            # therefore uses full-window stats too); the halo is stripped from every head below.
            _bx = _tooth["boxes"]
            _hin, _win = int(_tooth["hin"]), int(_tooth["win"])
            assert tuple(_bx.shape) == (b, 2), (tuple(_bx.shape), b)
            z = torch.stack([z[i, :, int(_bx[i, 0]):int(_bx[i, 0]) + _hin,
                               int(_bx[i, 1]):int(_bx[i, 1]) + _win] for i in range(b)], 0)
            h, w = _hin, _win
        x0 = self.inc(z)
        x1 = self.d1(self.pool(x0))
        x2 = self.d2(self.pool(x1))
        u2 = self.up2(x2)
        y = self.c2(torch.cat([u2, x1], 1))
        u1 = self.up1(y)
        y = self.c1(torch.cat([u1, x0], 1))
        out = self.outc(y)                                    # (B, M*t_out*c, H, W)
        heads = []
        n1 = self.t_out * c
        for hi in range(M):
            o = (out[:, hi * n1:(hi + 1) * n1]
                 .reshape(b, self.t_out, c, h, w).permute(0, 1, 3, 4, 2))
            if ss[hi] is not None:
                o = o * ss[hi]
            if ms[hi] is not None:
                o = o + ms[hi]
            heads.append(o)
        if _tooth is not None:
            # EQFREE TOOTHM: return the INNER tooth per head (halo excluded), as forward() does.
            _hl = int(_tooth["halo"])
            heads = [q[:, :, _hl:h - _hl, _hl:w - _hl] for q in heads]
        if paired:
            return torch.cat(heads, -1)                       # (B, t_out, H, W, M*c)
        # eval: the same window in every slot -> implicit-ensemble average (the MIMO
        # test-time rule).  Written as a left fold times 1/M so that at M=2 it is
        # BIT-IDENTICAL to the pre-MIMOSCALE `0.5 * (heads[0] + heads[1])`: scalar
        # multiplication is commutative in IEEE754 and 1/2 is exact.  Proven, not
        # reasoned -- analysis/mimoscale.py `null` check N2.
        # Heads stashed for the smoke witness (plain attribute, never a parameter/saved).
        self._last_mimo_heads = tuple(q.detach() for q in heads)
        acc = heads[0]
        for q in heads[1:]:
            acc = acc + q
        return acc * (1.0 / M)


    def _condx_planes(self, x, b, h, w):
        """HYBIN: (B, _HYBIN_CONDX[k], H, W) per-window planes from the kinetic continuity source.

        `x` is the POST-in_norm window (B, t_in, H, W, C), i.e. these live in normalised units; the
        steady fraction is a ratio and is scale-invariant, the two moment planes are not.  Computed
        from the t_in OBSERVED frames ONLY -- no target frame is reachable from here, so the
        construction cannot leak.
        """
        u, v = x[..., 0], x[..., 1]                       # (B, t, H, W)
        if self.condx in ("vortg", "gradg", "shuf", "divk"):
            return self._condx_planes_hybrex(u, v, b, h, w)
        if self.condx == "vortms":
            pa = torch.nn.functional.pad(v, (1, 1, 0, 0), mode="replicate")
            pb = torch.nn.functional.pad(u, (0, 0, 1, 1), mode="replicate")
            # PHYSICAL curl dv/dx - du/dy; dy < 0 makes it a PLUS in index units.
            f = 0.5 * (pa[..., 2:] - pa[..., :-2]) + 0.5 * (pb[..., 2:, :] - pb[..., :-2, :])
        else:
            pa = torch.nn.functional.pad(u, (1, 1, 0, 0), mode="replicate")
            pb = torch.nn.functional.pad(v, (0, 0, 1, 1), mode="replicate")
            f = 0.5 * (pa[..., 2:] - pa[..., :-2]) - 0.5 * (pb[..., 2:, :] - pb[..., :-2, :])
        if self.condx == "divg":
            fm = f.mean(dim=1)                            # (B, H, W) steady part
            fl = f - fm.unsqueeze(1)
            s_st = (fm * fm).mean(dim=(1, 2))
            s_fl = (fl * fl).mean(dim=(1, 2, 3))
            frac = s_st / (s_st + s_fl + 1e-12)           # in [0, 1]; 0.644 real vs 0.090 sim
            rms = torch.sqrt((f * f).mean(dim=(1, 2, 3)) + 1e-12)
            return torch.stack([frac, rms], 1).reshape(b, 2, 1, 1).expand(b, 2, h, w)
        return torch.cat([f.mean(dim=1, keepdim=True), f.std(dim=1, keepdim=True)], 1)

    @staticmethod
    def _hybrex_fields(u, v):
        """div and PHYSICAL curl in array-index units (dy < 0 on this grid, see the HYBIN note above)."""
        F = torch.nn.functional
        ux = 0.5 * (F.pad(u, (1, 1, 0, 0), mode="replicate")[..., 2:] - F.pad(u, (1, 1, 0, 0), mode="replicate")[..., :-2])
        uy = 0.5 * (F.pad(u, (0, 0, 1, 1), mode="replicate")[..., 2:, :] - F.pad(u, (0, 0, 1, 1), mode="replicate")[..., :-2, :])
        vx = 0.5 * (F.pad(v, (1, 1, 0, 0), mode="replicate")[..., 2:] - F.pad(v, (1, 1, 0, 0), mode="replicate")[..., :-2])
        vy = 0.5 * (F.pad(v, (0, 0, 1, 1), mode="replicate")[..., 2:, :] - F.pad(v, (0, 0, 1, 1), mode="replicate")[..., :-2, :])
        div = ux - vy
        vort = vx + uy
        grad = torch.sqrt(ux * ux + uy * uy + vx * vx + vy * vy + 1e-12)
        return div, vort, grad

    @staticmethod
    def _steady_frac_rms(f):
        fm = f.mean(dim=1)
        fl = f - fm.unsqueeze(1)
        s_st = (fm * fm).mean(dim=(1, 2))
        s_fl = (fl * fl).mean(dim=(1, 2, 3))
        frac = s_st / (s_st + s_fl + 1e-12)
        rms = torch.sqrt((f * f).mean(dim=(1, 2, 3)) + 1e-12)
        return frac, rms, torch.sqrt(s_fl + 1e-12)

    def _condx_planes_hybrex(self, u, v, b, h, w):
        """HYBREX (2026-09-09) controls and the stacked-scalar arm; all CONSTANT planes, observed frames only."""
        div, vort, grad = self._hybrex_fields(u, v)
        if self.condx == "vortg":
            frac, rms, _ = self._steady_frac_rms(vort)
            cols = [frac, rms]
        elif self.condx == "gradg":
            frac, rms, _ = self._steady_frac_rms(grad)
            cols = [frac, rms]
        elif self.condx == "shuf":
            # the SAME two divg scalars, but taken from the NEXT window in the batch (roll by 1):
            # a plane that is live and in-distribution yet carries no information about THIS window.
            frac, rms, _ = self._steady_frac_rms(div)
            cols = [torch.roll(frac, 1, 0), torch.roll(rms, 1, 0)]
        else:  # divk
            dfrac, drms, dfl = self._steady_frac_rms(div)
            _, vrms, _ = self._steady_frac_rms(vort)
            cols = [dfrac, drms, dfl, vrms, drms / (vrms + 1e-12)]
        k = len(cols)
        return torch.stack(cols, 1).reshape(b, k, 1, 1).expand(b, k, h, w)

    def _derive(self, x, kind):
        """One derived field per input frame -> (B, t_in, H, W), appended as extra channels.

        Grid spacing is dropped: it is a constant factor per axis and the very next operation is a
        learned convolution that absorbs it.  Edges use replicate (one-sided) differences, so no
        periodicity is assumed in a domain that is not periodic.
        """
        u, v = x[..., 0], x[..., 1]                       # (B, t, H, W)
        if kind == "speed":
            return torch.sqrt(u * u + v * v + 1e-12)
        if kind == "div":
            # HYBIN: the in-plane divergence = the PIV-LBM mass source S = -rho d_z w, the only
            # field in this problem carrying information about the third dimension.  dy < 0 on this
            # grid (see _HYBIN_CONDX), hence the MINUS between the two index differences.
            du = torch.nn.functional.pad(u, (1, 1, 0, 0), mode="replicate")
            dv = torch.nn.functional.pad(v, (0, 0, 1, 1), mode="replicate")
            return 0.5 * (du[..., 2:] - du[..., :-2]) - 0.5 * (dv[..., 2:, :] - dv[..., :-2, :])
        pv = torch.nn.functional.pad(v, (1, 1, 0, 0), mode="replicate")
        pu = torch.nn.functional.pad(u, (0, 0, 1, 1), mode="replicate")
        return 0.5 * (pv[..., 2:] - pv[..., :-2]) - 0.5 * (pu[..., 2:, :] - pu[..., :-2, :])


def build_unet(t_in=20, t_out=20, c=3, base=32, **kw):
    """`fuse="none"` (default) is the shipped architecture, unchanged.
    `refine`/`refine_base` wrap the net in a learned one-step corrector (arm REFa).
    """
    refine = int(kw.pop("refine", 0) or 0)
    rbase = int(kw.pop("refine_base", 32) or 32)
    rzinit = int(kw.pop("refine_zinit", 0) or 0)
    # COBLEND (2026-09-02): `coblend: M` (+ `coblend_blocks: [..]`) builds M UNet2D members that
    # share every other kw and wraps them in CoBlend, whose forward is the shipped _BlendNet mean.
    # Travels inside the `fuse:` dict (train_v4 spreads it verbatim; _ckpt_meta records it; the
    # ARCH_KEYS copy list cannot drop it).  Absent/0 == BIT-IDENTICAL off: two pops of absent keys,
    # no module, no RNG draw -- proven by Track1/COBLEND/analysis/coblend_null.py.
    _cob = int(kw.pop("coblend", 0) or 0)
    _cob_blocks = kw.pop("coblend_blocks", None)
    if _cob:
        assert 2 <= _cob <= 4, _cob
        assert not refine, "coblend does not compose with refine"
        assert not int(kw.get("cond_film", 0) or 0), "coblend does not compose with cond_film"
        _blocks = [str(q) for q in (_cob_blocks or [kw.get("block") or "double"] * _cob)]
        assert len(_blocks) == _cob, (_blocks, _cob)
        _members = []
        for _bk in _blocks:
            _mk = dict(kw)
            _mk["block"] = _bk
            _members.append(UNet2D(t_in=t_in, t_out=t_out, c=c, base=base, **_mk))
        return CoBlend(_members, _blocks)
    m = UNet2D(t_in=t_in, t_out=t_out, c=c, base=base, **kw)
    if refine:
        g = UNet2D(t_in=t_in + t_out, t_out=t_out, c=c, base=rbase,
                   block=str(kw.get("block") or "double"), in_norm="self")
        return RefineNet(m, g, zinit=rzinit)
    return m



class CoBlend(nn.Module):
    """COBLEND (2026-09-02): M sub-networks trained JOINTLY under a loss on their uniform mean.

    forward is `submission_t1_v78_edbl2.py:_BlendNet.forward` (lines 173-177) op-for-op and in the
    same order -- `o = m0(x); o = o + m1(x); ...; o / float(M)` -- so a member pair exported from
    this wrapper reproduces the trained function BIT-FOR-BIT through the shipped card body (proven
    per checkpoint by Track1/COBLEND/analysis/coblend_export.py).  Each member applies its own
    `in_norm` inside UNet2D.forward; the wrapper performs no normalisation of its own, exactly as
    the card composes two independently trained members.  The members' outputs are stashed
    NON-detached in `_last_member_outs` so train_v4 can add the per-member (independent) loss
    term; at inference the stash is a plain attribute, never a parameter, never saved.
    Attribute proxies (t_in, out_mode, anchor, dt, in_norm, cond) mirror member 0 so
    train_v4's post-build object asserts read the same values a bare UNet2D exposes."""

    def __init__(self, members, blocks):
        super().__init__()
        assert len(members) == len(blocks) >= 2, (len(members), blocks)
        self.members = nn.ModuleList(members)
        self.coblend = len(members)
        self.coblend_blocks = tuple(str(b) for b in blocks)
        _m0 = members[0]
        for _a in ("t_in", "t_out", "c", "out_mode", "anchor", "dt", "in_norm", "cond"):
            setattr(self, _a, getattr(_m0, _a))

    def forward(self, x):
        outs = [m(x) for m in self.members]
        o = outs[0]
        for q in outs[1:]:
            o = o + q
        self._last_member_outs = tuple(outs)
        return o / float(len(self.members))


class RefineNet(nn.Module):
    """Learned one-step iterative refinement (arm REFa, 2026-08-05; 'iterative
    refinement neural operators are learned fixed-point solvers', S2 sweep 3).
    y1 = f(x); y2 = y1 + g([x, y1]) — the corrector reads the first pass's own
    output.  In-place refinement: NOT time-rollout (unrolling, closed) and NOT
    input anchoring (arm R, closed).  Loss lands on the final output only (v1).

    `refine_zinit: 1` (arm REFz, 2026-08-24) puts a ZERO-INITIALISED SCALAR GATE on the corrector
    branch so the wrapper is an EXACT identity at init and any deviation is learned improvement.
    REFa v1 had no gate: at step 0 the randomly initialised corrector dumped a large field on top of
    the base output and training had to unlearn it -- measured fold 0 stuck at val rel_l2 0.1007
    against CW128's 0.0687 *with a higher train loss* (underfitting, not overfitting), fold 3 0.0746.
    NOTE the trap this cost: zeroing `g.outc` does NOT zero the branch, because `g` runs with
    `in_norm="self"` and de-normalises by the input window's own statistics -- a zero head returns
    that window's MEAN field, not zero (measured: max|y2-y1| = 2.0e-02).  The gate is independent of
    whatever `g` does internally, which is exactly why it is the right mechanism."""

    def __init__(self, f, g, zinit=0):
        super().__init__()
        self.f = f
        self.g = g
        # a Parameter, not a buffer: alpha must train.  d(loss)/d(alpha) is non-zero at alpha=0, so
        # alpha leaves zero on step 1 and `g` starts receiving gradient from step 2 -- the intended
        # slow start, identical in spirit to ControlNet's zero-conv and LayerScale-0.
        self.alpha = nn.Parameter(torch.zeros(())) if int(zinit or 0) else None

    def forward(self, x):
        y1 = self.f(x)
        self._last_y1 = y1     # deep-supervision hook (REFb); plain attr, not a param
        dy = self.g(torch.cat([x, y1], dim=1))
        return y1 + dy if self.alpha is None else y1 + self.alpha * dy


class AdvectCanon(nn.Module):
    """Transport canonicalisation (`in_adv: mean`): warp every input frame to the LAST-frame
    time along the window's own mean flow, run the net on the slow canonical field, warp the
    output frames back.  Pure shift per frame (Galilean), border padding, no parameters.

    Constants are the verified data facts: competition normalisation stats, 32x64 grid spacing,
    dt = 0.02.  dy < 0 in the physical grid (row index runs against y), so positive v moves
    content toward SMALLER row index -- hence the minus on the row shift.
    """

    _MEAN_UV = (0.1550, -0.0005)
    _STD_UV = (0.0968, 0.0160)
    _DXY = (3.449e-3, 3.484e-3)
    _DT = 0.02

    def __init__(self, net, mode="mean"):
        super().__init__()
        assert str(mode) in ("mean", "meanb6"), mode
        self.net = net
        self.mode = str(mode)

    @staticmethod
    def _shift(z, sx, sy):
        """z: (B, T, H, W, C); move CONTENT by (+sx, +sy) pixels (per-sample scalars)."""
        import torch.nn.functional as F
        B, T, H, W, C = z.shape
        zc = z.permute(0, 1, 4, 2, 3).reshape(B, T * C, H, W)
        th = torch.zeros(B, 2, 3, device=z.device, dtype=z.dtype)
        th[:, 0, 0] = 1.0
        th[:, 1, 1] = 1.0
        th[:, 0, 2] = -2.0 * sx / max(W - 1, 1)
        th[:, 1, 2] = -2.0 * sy / max(H - 1, 1)
        g = F.affine_grid(th, zc.shape, align_corners=True)
        out = F.grid_sample(zc, g, mode="bilinear", padding_mode="border", align_corners=True)
        return out.reshape(B, T, C, H, W).permute(0, 1, 3, 4, 2)

    def _flow_px(self, x):
        u = x[..., 0] * self._STD_UV[0] + self._MEAN_UV[0]
        v = x[..., 1] * self._STD_UV[1] + self._MEAN_UV[1]
        pxu = u.mean(dim=(1, 2, 3)) * self._DT / self._DXY[0]
        pxv = -v.mean(dim=(1, 2, 3)) * self._DT / self._DXY[1]
        return pxu, pxv

    def _off(self, p, fac):
        s = p * fac
        return s.clamp(-6.0, 6.0) if self.mode == "meanb6" else s

    def canon(self, x):
        T = x.shape[1]
        pxu, pxv = self._flow_px(x)
        return torch.cat([self._shift(x[:, j:j + 1], self._off(pxu, T - 1 - j),
                                      self._off(pxv, T - 1 - j))
                          for j in range(T)], dim=1), pxu, pxv

    def forward(self, x):
        xc, pxu, pxv = self.canon(x)
        y = self.net(xc)
        T = y.shape[1]
        return torch.cat([self._shift(y[:, k:k + 1], self._off(pxu, k + 1),
                                      self._off(pxv, k + 1))
                          for k in range(T)], dim=1)


class InNorm(nn.Module):
    """`in_norm` as a wrapper, so any backbone can have it -- not just the U-net.

    Canonicalise the input window by its OWN statistics, run the backbone on the canonical field,
    and undo the transform on the output.  Identical arithmetic to the block inside `UNet2D.forward`
    (`_selfcheck` proves it bit-for-bit); the U-net keeps its internal path so that every existing
    checkpoint and recorded result stays valid, and this is what everything else uses.

    Modes, and what each one declares to be nuisance:
        "self"  per-channel mean AND per-channel scale  -- the full canonicalisation, the winner
        "scale" per-channel scale only
        "mean"  per-channel mean only  (a Galilean frame change)
        "vec"   per-channel mean, ONE velocity scale shared by u and v, p keeps its own
    """

    def __init__(self, net, mode="self"):
        super().__init__()
        assert mode in ("self", "scale", "mean", "vec"), mode
        self.net = net
        self.mode = mode

    def forward(self, x):
        c = x.shape[-1]
        m = s = None
        if self.mode == "mean":
            m = x.mean(dim=(1, 2, 3), keepdim=True)
            z = x - m
        elif self.mode == "vec":
            m = x.mean(dim=(1, 2, 3), keepdim=True)
            xc = x - m
            s = xc.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
            if c >= 2:
                sv = xc[..., :2].std(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)
                s = torch.cat([sv.expand(-1, -1, -1, -1, 2), s[..., 2:]], dim=-1)
            z = xc / s
        else:
            s = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
            if self.mode == "self":
                m = x.mean(dim=(1, 2, 3), keepdim=True)
                z = (x - m) / s
            else:
                z = x / s
        out = self.net(z)
        if s is not None:
            out = out * s
        if m is not None:
            out = out + m
        return out


class Residual(nn.Module):
    """out = last input frame + net(x).  The attribute name `net` is load-bearing: it is what puts
    the `net.` prefix on every key of a residual checkpoint's state_dict."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x) + x[:, -1:]


ARCH_KEYS = ("fuse", "in_aug", "in_norm", "in_adv", "act", "out_mode", "anchor", "dt",
              "substeps", "proj_iters", "mask_rel", "block", "levels", "drop_path", "layer_scale",
              "nxt_k", "nxt_grn", "nxt_gate", "nxt_depth", "nxt_act", "nxt_pad", "nxt_se", "skip", "stage_res", "cond", "head", "refine", "refine_base", "refine_zinit",
              "t_attn", "t_attn_dim", "t_attn_heads",
              "freq_decomp", "freq_width", "freq_layers", "freq_smooth_n", "freq_w",
              "aux_phases", "route", "head_split", "mimo", "hetero_head", "flat_depth")


def _build_from_meta_core(meta, t_in=None, t_out=None, c=None):
    """Rebuild the architecture a checkpoint was trained with, from its own `meta`.

    `base` alone is not enough.  `fuse` ADDS parameters, `residual` RENAMES them -- both of which a
    strict load would catch -- but `in_aug` and `in_norm` change the computation while leaving the
    parameter set untouched for `in_norm`, so a strict load CANNOT catch a missing one.  Read them
    back here or not at all.

    Keys absent from meta mean the shipped architecture, so every older checkpoint is unaffected.
    """
    meta = meta or {}
    # `arch` selects the backbone.  Absent (or "unet") means the shipped frame-stacked U-net, so
    # every existing checkpoint is unaffected.  The seqformer import is LAZY on purpose: this file
    # ships inside every submission zip, and a U-net bundle must not acquire that dependency.
    if str(meta.get("arch", "unet")) == "convltc":
        # LTCa checkpoints rebuild through their own module — lazy import, same dir.
        import sys as _sys
        from pathlib import Path as _P
        _d = str(_P(__file__).resolve().parent)
        if _d not in _sys.path:
            _sys.path.insert(0, _d)
        from convltc import build_convltc
        return build_convltc(t_in=int(t_in or meta.get("t_in", 20)),
                             t_out=int(t_out or meta.get("t_out", 20)),
                             c=int(c or meta.get("c", 3)),
                             ch=int(meta.get("ltc_ch", 128)))
    if str(meta.get("arch", "unet")) in ("seqformer", "seqformer_v1"):
        import sys as _sys
        from pathlib import Path as _P
        _sf = _P(__file__).resolve().parent.parent / "seqformer"
        if str(_sf) not in _sys.path:
            _sys.path.insert(0, str(_sf))
        from seqformer import build_seqformer, DEFAULTS as _SFD
        scfg = dict(_SFD)
        scfg.update(meta.get("seqformer") or {})
        if t_in is not None:
            scfg["t_in"] = t_in
        if t_out is not None:
            scfg["t_out"] = t_out
        if c is not None:
            scfg["c"] = c
        return build_seqformer(scfg)
    if str(meta.get("arch", "unet")) == "cfo":
        # CFO velocity operator + internal RK4; lives in cfo_model.py next to this file.  Lazy
        # for the same reason as every non-unet arch: shipped bundles carry U-nets, and a U-net
        # bundle must not acquire this import.  meta["cfo"] is the RESOLVED option dict written
        # by train_v4._ckpt_meta, so nothing about the forward can be silently dropped here.
        import sys as _sys
        from pathlib import Path as _P
        _d = str(_P(__file__).resolve().parent)
        if _d not in _sys.path:
            _sys.path.insert(0, _d)
        from cfo_model import build_cfo
        return build_cfo(t_in=int(meta.get("t_in", 20)) if t_in is None else t_in,
                         t_out=int(meta.get("t_out", 20)) if t_out is None else t_out,
                         c=int(meta.get("c", 3)) if c is None else c,
                         base=int(meta.get("base", 96)), **dict(meta.get("cfo") or {}))
    if str(meta.get("arch", "unet")) == "sol":
        # Solver-in-the-loop model; lives with its measurement in Track2/solver-in-the-loop.
        # Lazy for the same reason as every non-unet arch: bundles carry U-nets.
        import sys as _sys
        from pathlib import Path as _P
        _sol = None
        for _q in (_P(__file__).resolve().parent.parent.parent / "Track2" / "solver-in-the-loop",
                   _P(__file__).resolve().parent.parent / "solver-in-the-loop"):
            if (_q / "sol_model.py").exists():
                _sol = _q
                break
        if _sol is None:
            raise RuntimeError("meta says arch='sol' but sol_model.py was not found")
        if str(_sol) not in _sys.path:
            _sys.path.insert(0, str(_sol))
        from sol_model import build_sol
        m = build_sol(t_in=int(meta.get("t_in", 20)) if t_in is None else t_in,
                      t_out=int(meta.get("t_out", 20)) if t_out is None else t_out,
                      c=int(meta.get("c", 3)) if c is None else c,
                      base=int(meta.get("base", 48)))
        if meta.get("in_norm"):
            m = InNorm(m, str(meta["in_norm"]))
        return Residual(m) if meta.get("residual") else m
    if str(meta.get("arch", "unet")) in ("fno", "transolver", "cno"):
        # One of the OFFICIAL baselines, trained from random init by train_v4.py's `arch:` switch.
        # Built through the starting kit's own `build_model`, so the hyper-parameters come from the
        # kit (read off the released checkpoint shapes) instead of being duplicated here.  Lazy for
        # the same reason as seqformer above.
        import sys as _sys
        from pathlib import Path as _P
        _a = str(meta["arch"])
        _here = _P(__file__).resolve().parent
        # THE BUNDLE DIR COMES FIRST: inside a shipped zip the kit does not exist and
        # load_baseline.py + rpde_baselines/ travel next to this file.
        _cands = [_here,
                  _here.parent / "realpde_t1_starting_kit_v6",
                  _here.parent.parent / "Track1" / "realpde_t1_starting_kit_v6"]
        _kit = next((q for q in _cands if (q / "load_baseline.py").exists()), None)
        if _kit is None:
            raise RuntimeError(
                f"meta says arch={_a!r}, an official baseline, which needs the Track-1 starting "
                f"kit's load_baseline.py to rebuild; looked in {[str(q) for q in _cands]}.  A "
                f"SHIPPED bundle should never reach this path -- bundles carry U-nets.")
        if str(_kit) not in _sys.path:
            _sys.path.insert(0, str(_kit))
        from load_baseline import build_model as _bm
        _ti = int(meta.get("t_in", 20)) if t_in is None else t_in
        _to = int(meta.get("t_out", 20)) if t_out is None else t_out
        _c = int(meta.get("c", 3)) if c is None else c
        m = _bm(_a, device="cpu", in_shape=(_ti, 32, 64, _c), out_shape=(_to, 32, 64, _c))
        if meta.get("in_norm"):
            m = InNorm(m, str(meta["in_norm"]))
        return Residual(m) if meta.get("residual") else m
    fuse = meta.get("fuse") or "none"
    kw = dict(fuse) if isinstance(fuse, dict) else ({} if fuse == "none" else {"fuse": fuse})
    for k in ARCH_KEYS:
        if k != "fuse" and meta.get(k):
            kw[k] = meta[k]
    m = build_unet(t_in=int(meta.get("t_in", 20)) if t_in is None else t_in,
                   t_out=int(meta.get("t_out", 20)) if t_out is None else t_out,
                   c=int(meta.get("c", 3)) if c is None else c,
                   base=int(meta.get("base", 32)), **kw)
    return Residual(m) if meta.get("residual") else m


def _selfcheck():
    """Shapes for every fuse mode, and -- the one that matters -- that both branches start as
    EXACT no-ops, so `fuse` changes nothing until the weights move."""
    torch.manual_seed(0)
    x = torch.randn(2, 20, 32, 64, 3)
    base_out = None
    for f in ("none", "attn", "spec", "both"):
        torch.manual_seed(0)
        m = build_unet(base=32, fuse=f).eval()
        with torch.no_grad():
            y = m(x)
        n = sum(p.numel() for p in m.parameters())
        assert y.shape == (2, 20, 32, 64, 3), (f, y.shape)
        if f == "none":
            base_out = y
        else:
            d = (y - base_out).abs().max().item()
            assert d < 1e-5, f"fuse={f} is not a no-op at init: max|d| = {d:.2e}"
        print(f"  fuse={f:5s} params {n/1e6:.3f}M  out {tuple(y.shape)}  "
              f"{'(reference)' if f == 'none' else 'no-op at init OK'}")
    for aug in ("vort", "speed", "vort,speed"):
        m = build_unet(base=32, in_aug=aug).eval()
        with torch.no_grad():
            y = m(x)
        n_extra = len(aug.split(","))
        assert y.shape == (2, 20, 32, 64, 3), (aug, y.shape)
        assert m.inc.net[0].in_channels == 20 * (3 + n_extra), m.inc.net[0].in_channels
        print(f"  in_aug={aug:11s} ci={m.inc.net[0].in_channels}  OK")

    # vorticity is EXACT on a field whose curl is known: u = -y, v = x has dv/dx - du/dy = 2
    # everywhere, and central differences are exact on linear data.
    mm = build_unet(base=32, in_aug="vort")
    gy, gx = torch.meshgrid(torch.arange(32.0), torch.arange(64.0), indexing="ij")
    tt = torch.zeros(1, 20, 32, 64, 3)
    tt[..., 0], tt[..., 1] = -gy, gx
    interior = mm._derive(tt, "vort")[:, :, 1:-1, 1:-1]
    assert (interior - 2.0).abs().max() < 1e-4, interior.unique()
    print(f"  vorticity exact on solid-body rotation: max|w-2| = "
          f"{(interior - 2.0).abs().max().item():.2e}  OK")

    # the whole point of in_norm is the EQUIVARIANCE, so test that and not the shapes
    for nm, (al, be) in (("self", (2.7, 0.4)), ("scale", (2.7, 0.0)), ("mean", (1.0, 0.4)),
                        ("vec", (2.7, 0.4))):
        torch.manual_seed(0)
        mm = build_unet(base=32, in_norm=nm).eval()
        with torch.no_grad():
            y0 = mm(x)
            y1 = mm(al * x + be)
        d = (y1 - (al * y0 + be)).abs().max().item()
        assert d < 2e-3, f"in_norm={nm} is not equivariant: max|d| = {d:.2e}"
        print(f"  in_norm={nm:5s} equivariant under x -> {al}x + {be}: max|d| = {d:.2e}  OK")
    # `vec` claims a SMALLER group than `self`, and the whole point is what it does NOT quotient
    # out.  Shared scale + per-channel shift must be exact; a PER-CHANNEL scale must visibly
    # change the answer, because that is the u/v anisotropy that encodes the angle of attack.
    torch.manual_seed(0)
    mm = build_unet(base=32, in_norm="vec").eval()
    bc = torch.tensor([0.4, -0.2, 0.0]).view(1, 1, 1, 1, 3)
    ac = torch.tensor([2.7, 1.0, 1.0]).view(1, 1, 1, 1, 3)
    with torch.no_grad():
        y0 = mm(x)
        y1 = mm(2.7 * x + bc)          # shared scale, per-channel shift -> must be exact
        y2 = mm(ac * x)                # per-channel scale -> must NOT be absorbed
    d1 = (y1 - (2.7 * y0 + bc)).abs().max().item()
    d2 = (y2 - (ac * y0)).abs().max().item()
    assert d1 < 2e-3, f"vec is not equivariant to its own group: {d1:.2e}"
    assert d2 > 1e-2, f"vec absorbed a per-channel scale -- it threw the u/v ratio away: {d2:.2e}"
    print(f"  in_norm=vec   exact under (shared a, per-channel b): {d1:.2e};  and a per-channel a "
          f"is NOT absorbed ({d2:.2e}) -- the u/v ratio survives  OK")

    # build_from_meta must reproduce a PARAMETER-FREE key.  This is the check that would have
    # caught the train_vgain.py bug: the two models below have identical state_dicts, so nothing
    # about loading can tell them apart -- only the meta can.
    torch.manual_seed(0)
    ref = build_unet(base=32, in_norm="self").eval()
    got = build_from_meta({"base": 32, "in_norm": "self"}).eval()
    got.load_state_dict(ref.state_dict(), strict=True)
    plain = build_unet(base=32).eval()
    plain.load_state_dict(ref.state_dict(), strict=True)          # succeeds -- that is the trap
    with torch.no_grad():
        a, b, cq = ref(x), got(x), plain(x)
    assert (a - b).abs().max() < 1e-6, "build_from_meta did not reproduce in_norm"
    assert (a - cq).abs().max() > 1e-3, "in_norm made no difference -- the check is vacuous"
    print(f"  build_from_meta reproduces in_norm (d={float((a-b).abs().max()):.1e}) while a plain "
          f"build loads the SAME weights and differs by {float((a-cq).abs().max()):.3f}  OK")
    # The WRAPPER and the U-net's INTERNAL path must be the same computation.  Two implementations
    # of one idea is exactly how the train_vgain.py divergence started, so tie them together here
    # rather than trusting that they match.
    for nm in ("self", "scale", "mean", "vec"):
        torch.manual_seed(0)
        inner = build_unet(base=32, in_norm=nm).eval()
        torch.manual_seed(0)
        plain = build_unet(base=32).eval()
        plain.load_state_dict({k: v for k, v in inner.state_dict().items()}, strict=True)
        wrapped = InNorm(plain, nm).eval()
        with torch.no_grad():
            d = (inner(x) - wrapped(x)).abs().max().item()
        assert d < 1e-6, f"InNorm wrapper != internal in_norm for mode={nm}: {d:.2e}"
    print(f"  InNorm wrapper reproduces the U-net's internal in_norm for all 4 modes (max|d|<1e-6)"
          f"  OK")
    # every arch a checkpoint can record must round-trip through build_from_meta -- this is the
    # check that would have caught tonight's "FNO trains fine, then cannot be scored" failure.
    for _a in ("fno", "transolver", "cno"):
        mm = build_from_meta({"arch": _a, "t_in": 20, "t_out": 20, "c": 3}).eval()
        wn = build_from_meta({"arch": _a, "t_in": 20, "t_out": 20, "c": 3,
                              "in_norm": "self"}).eval()
        with torch.no_grad():
            o1, o2 = mm(x), wn(x)
        assert o1.shape == o2.shape == (2, 20, 32, 64, 3), (o1.shape, o2.shape)
        assert type(wn).__name__ == "InNorm", f"in_norm dropped for arch={_a}"
        print(f"  build_from_meta(arch={_a:10s}) rebuilds, and in_norm wraps it  OK")

    # ---- block=selfwarp (arm WARPa) -------------------------------------------------------
    # Per the repo's SELF-TEST RULE, two independent probes minimum: one that asserts the
    # feature is LIVE, and one that shows a deliberately wrong setting MOVES the score.  A warp
    # block that silently degenerates (all-zero rho forever, or a mis-shaped grid swallowed by
    # an `except`) reproduces the paper's own "non-learned warps" ablation and would look like a
    # plausible-but-dead arm, so it must be impossible to pass this check while dead.
    torch.manual_seed(0)
    msw = build_unet(base=32, block="selfwarp", in_norm="self").eval()
    with torch.no_grad():
        ysw = msw(x)
    nsw = sum(p.numel() for p in msw.parameters())
    torch.manual_seed(0)
    ncx = sum(p.numel() for p in build_unet(base=32, block="convnext",
                                            in_norm="self").parameters())
    assert ysw.shape == (2, 20, 32, 64, 3), ysw.shape
    assert torch.isfinite(ysw).all(), "selfwarp produced a non-finite output"
    print(f"  block=selfwarp params {nsw/1e6:.3f}M (convnext at base=32: {ncx/1e6:.3f}M)  "
          f"out {tuple(ysw.shape)}  finite  OK")

    # (i) at init the warp is the IDENTITY (departure 5) -- displacement is exactly zero px
    blk = msw.inc.net[1]
    with torch.no_grad():
        d0 = blk.disp_px(torch.randn(2, 32, 32, 64)).abs().max().item()
    assert d0 == 0.0, f"zero-init flow head did not give an identity warp: {d0:.2e} px"

    # (ii) FEATURE IS LIVE: perturbing ONLY the displacement head must move the output.  This is
    # the probe a dead warp cannot pass -- it touches no other weight.
    with torch.no_grad():
        base_y = msw(x)
        for m_ in msw.modules():
            if isinstance(m_, SelfWarpBlock):
                m_.flow.bias.normal_(0.0, 1.0)
        moved_y = msw(x)
        dmv = (moved_y - base_y).abs().max().item()
        dpx = blk.disp_px(torch.randn(2, 32, 32, 64)).abs().max().item()
    assert dmv > 1e-3, f"the displacement head does not affect the output -- warp is dead: {dmv:.2e}"
    assert dpx > 0.0, "displacement still zero after perturbing the flow head"
    print(f"  warp is LIVE: identity at init (0.00 px), and perturbing ONLY the flow head moves "
          f"the output by {dmv:.3f} at |rho|max {dpx:.2f} px  OK")

    # (iii) gradients REACH the displacement MLP.  A warp whose rho gets no gradient degrades to
    # a fixed resample -- the paper's non-learned-warp ablation (0.2190 vs 0.0624).
    torch.manual_seed(0)
    mg = build_unet(base=32, block="selfwarp", in_norm="self")
    mg(x).pow(2).mean().backward()
    gn = [(n_, p.grad) for n_, p in mg.named_parameters() if n_.endswith("flow.weight")]
    assert gn and all(g is not None for _, g in gn), "no grad on the flow head at all"
    gtot = sum(float(g.norm()) for _, g in gn)
    assert gtot > 0.0, f"displacement MLP received ZERO gradient ({len(gn)} flow convs)"
    print(f"  grad reaches the displacement MLP: {len(gn)} flow convs, summed |grad| = "
          f"{gtot:.4e} > 0  OK")

    # (iv) build_from_meta round-trips the new block value, and a convnext build of the same
    # meta MUST fail to strict-load -- the loud counterpart of the parameter-free trap above.
    got = build_from_meta({"base": 32, "block": "selfwarp", "in_norm": "self"}).eval()
    got.load_state_dict(msw.state_dict(), strict=True)
    with torch.no_grad():
        dmeta = (got(x) - msw(x)).abs().max().item()
    assert dmeta < 1e-6, f"build_from_meta did not reproduce block=selfwarp: {dmeta:.2e}"
    try:
        build_from_meta({"base": 32, "in_norm": "self"}).load_state_dict(msw.state_dict(),
                                                                        strict=True)
        raise AssertionError("a non-selfwarp build strict-loaded selfwarp weights")
    except RuntimeError:
        pass
    print(f"  build_from_meta(block=selfwarp) reproduces it (d={dmeta:.1e}) and a build without "
          f"the key refuses the weights  OK")

    # (v) NO LEAKAGE: constructing a selfwarp model must not perturb a later convnext build.
    # The `_NEXT_*` globals are exactly this hazard; SelfWarp uses none, and this proves it.
    torch.manual_seed(7)
    ref_cx = build_unet(base=32, block="convnext", in_norm="self").state_dict()
    torch.manual_seed(7)
    _ = build_unet(base=32, block="selfwarp", in_norm="self")
    torch.manual_seed(7)
    aft_cx = build_unet(base=32, block="convnext", in_norm="self").state_dict()
    assert set(ref_cx) == set(aft_cx)
    dleak = max(float((ref_cx[k] - aft_cx[k]).abs().max()) for k in ref_cx)
    assert dleak == 0.0, f"a selfwarp build changed a later convnext build: {dleak:.2e}"
    print(f"  no global-state leakage: a convnext build is bit-identical (max|d| = {dleak:.1f}) "
          f"before and after a selfwarp build  OK")
    print("UNET_FUSE_OK")


def build_from_meta(meta, t_in=None, t_out=None, c=None):
    """Public entry: the core dispatch plus wrappers that apply to EVERY arch (in_adv)."""
    m = _build_from_meta_core(meta, t_in, t_out, c)
    if (meta or {}).get("in_adv"):
        m = AdvectCanon(m, str(meta["in_adv"]))
    return m


# NOTE 2026-08-07: this guard used to sit ABOVE `build_from_meta`, so `python unet_model.py`
# died with `NameError: build_from_meta` partway through `_selfcheck` and the checks after that
# point had never actually run.  Moved to the end of the file; nothing else changed by it (the
# guard creates no parameters and consumes no RNG, and importing the module was never affected).
if __name__ == "__main__":
    _selfcheck()
