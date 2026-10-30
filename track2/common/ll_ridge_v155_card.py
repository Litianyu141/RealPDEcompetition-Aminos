#!/usr/bin/env python3
"""T2-LLR1 card patch (V200 slate, 2026-09-26; Track1/U-net/VariantRetrainPush/Logger/20260926/v200_plan.json, item T2-LLR1):
online LAST-LAYER ridge of forecaster member m1 (the H member) on the v155 body, fixed preconditioner P1, energy-preserving,
causal raw-blend (or m1-only) SSE gate.  Anchor-based patch of v155's submission.py + unet_model.py; adds ll_p1.npy.

What the card computes per step (probe twin: common/ll_precond_probe.py --est mean --alpha 1 --mode fullre --only 0 --gate 0):
  prefix (prediction-independent; par4: head of side2, then an event m1 waits on just before its outc):
    r   = cm * have * ((y - m_prev) / s_prev - o0_prev)          (previous window, m1 outc units, u/v rows only)
    R  += r^T [phi_prev; 1]        k += have                      (FP32 GEMM, matmul TF32 off for the call)
    U  += have * sum(e) / sum(cm y^2)   e = raw-blend (or m1) SSE gain of the previous window's would-be correction
    dW  = nan0((R @ P1) / max(k, 1));  W_eff = W + dW[:, :128];  b_eff = b + dW[:, 128]      (static buffers, never Parameters)
  m1's outc site (UNet2D.forward calls m1.__dict__['_ll_fn'] instead of self.outc(y)):
    o0 = outc(y) (the original FUSE-A addmm);  o1 = the same GEMM path on (W_eff, b_eff)  (FUSE-A _lin TF32 mirror / conv2d)
    per pixel and channel over the 20 leads: pres = o1 + (o1 - mean_T o1) (sqrt(var_T o0 / var_T o1) - 1)  (== the probe's
    fullre rescale; exactly o1 when the variances are equal)
    served = pres where (u, v) and U > _LL_GATE and k >= 1, else o0 (exact select);  dd = pres - o0 where (u, v) and k >= 1
    stash (static): phi = [y; 1], o0, dd, [s; m] (m1's in_norm stats) for the next prefix; blend gate: m2's u/v after the join.
  P1 = (G_hat + alpha tr(G_hat)/D I)^-1, G_hat = mean per-window Gram of m1's [y; 1] over the training windows of the card's
  data (fold card: the fold's training pool; all81 card: all 81 files), inverted in float64 here, stored float32 (ll_p1.npy).

Deviations from the plan text (documented, deliberate):
  * W_eff carries the UNGATED dW: the gate utility needs the would-be correction on every window (the probe computes it
    whether or not it is applied); folding g into W_eff would freeze U at 0.  g is applied as an exact select at serve time.
  * single stash buffers instead of a double buffer: par4's side2 prefix records its event after every stash read and m1
    waits on that event before its outc (where the stash is written), so the prefix never races the stash.
  * P1 ships as ll_p1.npy (numpy float32), not ll_p1.pt: the generator is numpy-only and md5-reproducible without torch.
  * --rctr is refused: the RCTR probe failed its registered gate (EQ dSPS +0.0133 < +0.02).
Without --ll the output is a byte-identical copy of --base.

    python3 common/ll_ridge_v155_card.py --base BASE.zip --out OUT.zip --ll (--ghat G.npz [--alpha 1.0] | --p1-zero)
                                          [--gate-src blend|m1]
"""
import argparse, hashlib, json, shutil, zipfile
from pathlib import Path

import numpy as np

MARK = "_LL_ON = "

# ---------------------------------------------------------------------------------------------------- unet_model.py
UNET_OLD = "        out = self.outc(y)                   # t_out*c, H, W\n        if self.freq_hi is not None:\n"
UNET_NEW = ("        _llf = self.__dict__.get(\"_ll_fn\")      # LL-ridge card patch (common/ll_ridge_v155_card.py): m1 only\n"
            "        out = self.outc(y) if _llf is None else _llf(self, y, _m, _s)   # t_out*c, H, W\n"
            "        if self.freq_hi is not None:\n")

# ---------------------------------------------------------------------------------------------------- submission.py
MOD_ANCHOR = "\n\nclass CalibratedTTTModel(TTTModel):\n"
MOD_BLOCK = r'''

# ---- LL-ridge card patch (common/ll_ridge_v155_card.py; V200 T2-LLR1) -------------------------------------------------
# Online last-layer ridge of member m1: fixed preconditioner P1 (ll_p1.npy), running mean R/k, energy-preserving rescale,
# causal SSE gate.  Refusal of any kind leaves the card exactly the base card (m1's outc hook is never installed).
_LL_ON = 1
_LL_GSRC = "__GSRC__"           # gate utility source: "blend" (raw blend, the probe's P1-EP arm) or "m1" (m1 alone)
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
'''

INIT_OLD = "        self._graph = None\n        self._have_flag = None\n        # STRPAR: the side streams are created HERE"
INIT_NEW = ("        self._ll_setup(submission_dir)          # LL-ridge card patch (before reset / capture)\n"
            "        self._graph = None\n        self._have_flag = None\n        # STRPAR: the side streams are created HERE")

RESET_OLD = ("            self._kf_mu.fill_(float(_KF_GRID[_KF_I0])); self._kf_c.fill_(float(_KF_CG[_KF_I0]))\n"
             "\n    # ------------------------------------------------------------------ BIASGATE (v54)\n")
RESET_NEW = ("            self._kf_mu.fill_(float(_KF_GRID[_KF_I0])); self._kf_c.fill_(float(_KF_CG[_KF_I0]))\n"
             "        if self.__dict__.get(\"_ll_on\", False):      # LL-ridge card patch\n"
             "            self._ll_reset()\n"
             "\n    # ------------------------------------------------------------------ BIASGATE (v54)\n")

METH_ANCHOR = "    def _strpar_eligible(self) -> bool:\n"
METHODS = r'''    # ------------------------------------------------------------------ LL-ridge card patch (common/ll_ridge_v155_card.py)
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

    def _ll_m2_stash(self, y2) -> None:
        """Blend gate: m2's output of this window (for the next window's utility)."""
        self._ll_u2.copy_(y2[0].permute(1, 2, 0, 3))

'''

ELIG_OLD = ("            and self._sb is not None                                    # SBPORT band\n"
            "        )\n")
ELIG_NEW = ("            and self._sb is not None                                    # SBPORT band\n"
            "            and self._ll_strpar_ok()                                    # LL-ridge card patch\n"
            "        )\n")

KCOK_OLD = "                         and self.__dict__.get(\"_kc_j2\") is not None)\n"
KCOK_NEW = ("                         and self.__dict__.get(\"_kc_j2\") is not None\n"
            "                         and self._ll_strpar_ok())                   # LL-ridge card patch\n")

BODY_KC_OLD = ("                y1 = self.base.m1(self._sx)\n"
               "                y2 = self.base.m2(self._sx)\n"
               "                self._kc_tail(y1, y2, lambda: self._band_branch(n_t, C, usable, wf))\n")
BODY_KC_NEW = ("                if self.__dict__.get(\"_ll_on\", False):     # LL-ridge prefix (sequential body)\n"
               "                    self._ll_prefix(self._sy, have)\n" + BODY_KC_OLD)

BODY_NK_OLD = "            # ---------- 2. forward ----------\n            p = self.base(self._sx)\n"
BODY_NK_NEW = ("            # ---------- 2. forward ----------\n"
               "            if self.__dict__.get(\"_ll_on\", False):     # LL-ridge prefix (sequential body, original tail)\n"
               "                self._ll_prefix(self._sy, have)\n"
               "            p = self.base(self._sx)\n")

SP3_OLD = ("                if self._gate_on:\n                    self._gate_corr(usable, wf)          # BIASGATE (v54, fused)\n"
           "\n            # ---------- 2. fork ----------\n")
SP3_NEW = ("                if self._gate_on:\n                    self._gate_corr(usable, wf)          # BIASGATE (v54, fused)\n"
           "                if self.__dict__.get(\"_ll_on\", False):     # LL-ridge prefix (par3: main, before the fork)\n"
           "                    self._ll_prefix(self._sy, have)\n"
           "\n            # ---------- 2. fork ----------\n")

SP4_OLD = ("                with torch.cuda.stream(self._side2):\n"
           "                    # ---- 1'. the prediction-independent prefix, moved off the critical path ----\n")
SP4_NEW = (SP4_OLD +
           "                    if self.__dict__.get(\"_ll_on\", False):     # LL-ridge prefix at side2's head; m1 waits on it\n"
           "                        self._ll_prefix(self._sy, have)          # just before its outc (every stash read is\n"
           "                        _lle = torch.cuda.Event(); _lle.record(self._side2)   # ahead of the event)\n"
           "                        self.__dict__[\"_ll_wait\"] = _lle\n")

SPM1_OLD = "            y1 = self.base.m1(self._sx)\n            cur.wait_event(e2)\n"
SPM1_NEW = ("            try:\n"
            "                y1 = self.base.m1(self._sx)\n"
            "            finally:\n"
            "                self.__dict__[\"_ll_wait\"] = None             # LL-ridge: the side2 event is m1's only\n"
            "            cur.wait_event(e2)\n")

SPNK_OLD = "            p = 0.5 * (y1 + y2)\n            _pre = p\n"
SPNK_NEW = ("            p = 0.5 * (y1 + y2)\n"
            "            if self.__dict__.get(\"_ll_on\", False) and _LL_GSRC == \"blend\":\n"
            "                self._ll_m2_stash(y2)                    # LL-ridge blend gate (original tail)\n"
            "            _pre = p\n")

TAIL_OLD = ("        p = self._kc_run(\"blend\", self._kc_blend_fj, self._kc_blend_ref, y1, y2)\n")
TAIL_NEW = (TAIL_OLD +
            "        if self.__dict__.get(\"_ll_on\", False) and _LL_GSRC == \"blend\":\n"
            "            self._ll_m2_stash(y2)                        # LL-ridge blend gate: m2's u/v after the join\n")

EAGER_OLD = "            p = self.base(x)\n            _pre = p\n"
EAGER_NEW = ("            if self.__dict__.get(\"_ll_on\", False):     # LL-ridge prefix (eager fallback)\n"
             "                _llh = prev_target_norm is not None and self._prev_pg is not None\n"
             "                self._ll_prefix(torch.as_tensor(prev_target_norm).to(self.device) if _llh else self._ll_zy,\n"
             "                                self._ll_one if _llh else self._ll_zero)\n"
             "            p = self.base(x)\n            _pre = p\n")

BLEND_OLD = "    def forward(self, x):\n        return 0.5 * (self.m1(x) + self.m2(x))\n"
BLEND_NEW = ("    def forward(self, x):\n"
             "        _llf = self.__dict__.get(\"_ll_m2fn\")      # LL-ridge blend gate (paths that call the blend as a whole)\n"
             "        if _llf is None:\n"
             "            return 0.5 * (self.m1(x) + self.m2(x))\n"
             "        a = self.m1(x)\n"
             "        b = self.m2(x)\n"
             "        _llf(b)\n"
             "        return 0.5 * (a + b)\n")

SUB_PATCHES = [("module", MOD_ANCHOR, None), ("init", INIT_OLD, INIT_NEW), ("reset", RESET_OLD, RESET_NEW),
               ("methods", METH_ANCHOR, METHODS + METH_ANCHOR), ("strpar_eligible", ELIG_OLD, ELIG_NEW),
               ("kc_ok", KCOK_OLD, KCOK_NEW), ("body_kc", BODY_KC_OLD, BODY_KC_NEW), ("body_nk", BODY_NK_OLD, BODY_NK_NEW),
               ("strpar_par3", SP3_OLD, SP3_NEW), ("strpar_par4", SP4_OLD, SP4_NEW), ("strpar_m1", SPM1_OLD, SPM1_NEW),
               ("strpar_nk", SPNK_OLD, SPNK_NEW), ("kc_tail", TAIL_OLD, TAIL_NEW), ("eager", EAGER_OLD, EAGER_NEW),
               ("blend", BLEND_OLD, BLEND_NEW)]


# ==================================================================================================== T2-DTTT-A (--feat h|hy)
# V200 plan priority 8 (Track1/U-net/VariantRetrainPush/Logger/20260926/v200_plan.json, item T2-DTTT-A); probe twin:
# common/deep_ttt_probe.py --feat hy --stats <fold stats> --alpha 0.1 --lag 1 --solve32 --mode fullre --only 0 --gate 0.
# The LLR1 machinery (outc site, energy-preserving rescale, raw-blend gate, stash, m2 stash, every body's prefix call) with
#   * features phi = [(h - mu_h)/sd_h; (y - mu_y)/sd_y; 1] (D = 641; --feat h: [(h - mu_h)/sd_h; 1], D = 513), where h is the
#     output of m1.c1.net[-1].act (the GELU that feeds p2), taken by a forward hook on that module (fires in the FUSE-A
#     fused path and in the conv path; no copy: the hook keeps a reference, the outc site consumes it), mu / sd = the card's
#     data pool statistics (ll_dt.npy, [mu; sd], float32; fold card: the fold's training pool, all81 card: all 81 files);
#   * the exact ridge instead of the fixed preconditioner: G += have phi phi^T, R += r phi^T (FP32 GEMMs, TF32 off);
#     dW^T = cholesky_solve(R^T, cholesky_ex(G + _DT_ALPHA tr(G)/D I)) in float32, zeroed unless info == 0 and finite;
#   * LAG ONE: the solve made at step k (targets up to k-1) serves window k+1.  In par4 the prefix (promote the pending
#     solve, residual / U, R, G) and then the solve run on a DEDICATED stream `_dt_side`, forked from the step's fork event
#     like m2 / side2: m1 waits on the prefix event just before its outc, main joins the solve at the very end of the step
#     (event _dt_e6), side2 is exactly the base card's.  (A first layout ran the prefix at side2's head and the solve at
#     side2's tail: the image kernel attribution showed side2 +0.92 ms busy and the step end moved +0.35 ms -- the solve
#     sat on the critical path.)  Sequential bodies / par3 / eager: the solve follows the prefix directly;
#   * the correction is applied at m1's outc as o1 = o0 + phi @ dW^T (one FP32 addmm onto o0's (H*W, Co) matrix);
#     served while U > _LL_GATE and the served solve saw >= 1 revealed window (the probe's lagged dW is not None).
# _DT_ZERO = 1 forces dW = 0 (the correction-off identity card: P/L/H bit-identical to the base card).
DT_MOD_EXTRA = r'''
# ---- DTTT-A extension (common/ll_ridge_v155_card.py --feat __FEAT__; V200 T2-DTTT-A) --------------------------------------
_DT_FEAT = "__FEAT__"           # "h": phi = [h~; 1]; "hy": phi = [h~; y~; 1]  (~: standardized with ll_dt.npy)
_DT_ALPHA = __ALPHA__             # ridge strength (x tr(G)/D), the probe's registered primary
_DT_ZERO = __ZERO__                 # 1: dW = 0 (the correction-off identity card)
_LL_SRC["dt_g"] = ("template <typename T> T dt_g(T x, T ok) { return (ok > T(0.5) && x - x == T(0)) ? x : T(0); }", 1)
'''

DT_METHODS = r'''    # ------------------------------------------------------------------ DTTT-A card patch (common/ll_ridge_v155_card.py --feat)
    def _ll_setup(self, submission_dir) -> None:
        """Allocate the DTTT-A state, install m1's outc site and the h tap.  Any refusal leaves the base card's path untouched."""
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
                      and not getattr(m1, "stage_res", False) and not int(getattr(m1, "cond_film", 0) or 0)
                      and int(oc.out_channels) == int(self._n_t) * int(self._C) and int(self._C) == 3
                      and _LL_GSRC in ("blend", "m1") and _DT_FEAT in ("h", "hy"))
            B = None
            if ok:
                B = m1.c1.net[-1]
                ok = bool(type(B).__name__ == "NeXtBlock" and B.gate == "" and B.gamma is None and B.se is None
                          and not B.grn and isinstance(B.act, nn.Module) and isinstance(B.p2, nn.Conv2d))
            if not ok:
                why = "config not covered"
            else:
                D0y, D0h = int(oc.in_channels), int(B.p2.in_channels)
                D0 = D0h + (D0y if _DT_FEAT == "hy" else 0)
                ms = None
                for d_ in (submission_dir, _HERE):
                    f_ = os.path.join(d_, "ll_dt.npy") if d_ else ""
                    if f_ and os.path.exists(f_):
                        ms = np.load(f_, allow_pickle=False)
                        break
                if ms is None or tuple(ms.shape) != (2, D0) or not np.isfinite(ms).all() or not (ms[1] > 0).all():
                    ok = False
                    why = "ll_dt.npy missing / shape %s != %s / non-finite / sd <= 0" % (None if ms is None else tuple(ms.shape), (2, D0))
            if ok and not _ll_kernels():
                ok = False
                why = _LL_STATE["why"] or "jiterator unavailable"
            if ok:
                d = self.device
                H, W, T, C = int(self._H), int(self._W), int(self._n_t), int(self._C)
                z = lambda *s: torch.zeros(*s, dtype=torch.float32, device=d)
                Co, D = T * C, D0 + 1
                self._ll_D0, self._dt_D0h, self._dt_D0y = D0, D0h, D0y
                mu, sd = ms[0].astype(np.float64), ms[1].astype(np.float64)
                a_, b_ = 1.0 / sd, -mu / sd                                          # (x - mu) / sd = x * a + b
                self._dt_a = torch.as_tensor(a_.astype(np.float32)).to(d).reshape(1, D0)
                self._dt_b = torch.as_tensor(b_.astype(np.float32)).to(d).reshape(1, D0)
                self._ll_phiT = z(H * W, D)
                self._ll_phiT[:, D0] = 1.0
                self._dt_phiH = z(H * W, D)
                self._dt_G = z(D, D)
                self._dt_eye = torch.eye(D, dtype=torch.float32, device=d)
                try:
                    self._dt_side = torch.cuda.Stream()      # par4: the DTTT prefix + solve lane
                except Exception:
                    self._dt_side = None                     # par4 then has no DTTT lane: `_ll_strpar_ok` refuses it
                self._dt_Wt = z(D, Co)               # the solve serving this window (previous step's)
                self._dt_Wtn = z(D, Co)              # this step's solve (serves the next window)
                self._dt_ks = z(())                  # revealed windows behind the serving solve
                self._dt_kn = z(())                  # ... behind the pending one
                self._dt_ok = z(())
                self._ll_o0 = z(H, W, T, C)
                self._ll_dd = z(H, W, T, C)
                self._ll_u2 = z(H, W, T, C)
                self._ll_sm = z(2, C)
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
                self.__dict__["_dt_B"] = B            # plain attribute (never registered as a submodule)
                B.act.register_forward_hook(self._dt_htap)
                self._ll_on = True
                why = "DTTT-A feat %s D %d alpha %g%s, gate %s > %g, |a| max %.3e" % (
                    _DT_FEAT, D, _DT_ALPHA, " ZERO" if _DT_ZERO else "", _LL_GSRC, _LL_GATE, float(np.abs(a_).max()))
        except Exception as e:
            self._ll_on = False
            why = "%s: %s" % (type(e).__name__, str(e)[:200])
        if self._ll_on:
            m1.__dict__["_ll_fn"] = self._ll_outc
            if _LL_GSRC == "blend":
                self.base.__dict__["_ll_m2fn"] = self._ll_m2_stash
        _LL_STATE["on"] = bool(self._ll_on)
        _LL_STATE["why"] = why
        print("DTTT %s (%s)" % ("on" if self._ll_on else "OFF", why), flush=True)

    def _dt_htap(self, _mod, _inp, out):
        """Forward hook on m1.c1.net[-1].act: keep a reference to every h of this m1 forward (no copy).  One call per
        forward once FUSE-A has decided; on the block's FIRST eligible call the FUSE-A load-time guard runs it four times
        (fused / ref at ambient precision, then both in FP32) and serves the first (fused, ambient) when it turns the
        fused path on, or a fifth, ref-path call when it turns it off -- `_dt_hprod` picks the served one."""
        self.__dict__.setdefault("_dt_hs", []).append(out)

    def _dt_hprod(self):
        hs = self.__dict__.pop("_dt_hs")
        return hs[0] if getattr(self.__dict__["_dt_B"], "_fusea", None) is True else hs[-1]

    def _ll_reset(self) -> None:
        for b in (self._ll_R, self._ll_k, self._ll_U, self._ll_o0, self._ll_dd, self._ll_u2, self._dt_G, self._dt_Wt,
                  self._dt_Wtn, self._dt_ks, self._dt_kn):
            b.zero_()
        self._ll_phiT[:, :self._ll_D0].zero_()
        self._ll_sm.zero_()
        self._ll_sm[0].fill_(1.0)

    def _ll_strpar_ok(self) -> bool:
        """DTTT clause of `_strpar_eligible` / `_kc_ok`: off, or fully set up (every body implements it)."""
        return (not self.__dict__.get("_ll_on", False)) or (bool(_LL_K) and self.__dict__.get("_dt_G") is not None
                                                             and self.__dict__.get("_dt_side") is not None)

    def _ll_prefix(self, y, have, solve=True) -> None:
        """Prediction-independent DTTT update from the previous revealed window: promote the pending (lag-one) solve,
        residual / gate utility, G += have phi phi^T, R += r phi^T.  par4 passes solve=False and runs `_dt_solve` at
        side2's tail; every other body solves here."""
        K = _LL_K
        H, W = int(self._H), int(self._W)
        self._dt_Wt.copy_(self._dt_Wtn)
        self._dt_ks.copy_(self._dt_kn)
        sm = self._ll_sm
        yv = y[0].permute(1, 2, 0, 3)                                            # (H, W, T, C) view
        r, e, tt = K["ll_r"](self._ll_o0, self._ll_dd, yv, self._ll_u2, sm[0], sm[1], self._ll_cm, have,
                             bl=(1.0 if _LL_GSRC == "blend" else 0.0))
        num = e.sum()
        den = tt.sum()
        _ll_fp32(self._ll_R.addmm_, r.reshape(H * W, -1).t(), self._ll_phiT)
        torch.mul(self._ll_phiT, have, out=self._dt_phiH)
        _ll_fp32(self._dt_G.addmm_, self._dt_phiH.t(), self._ll_phiT)
        self._ll_k.add_(have)
        self._ll_U.copy_(K["ll_u"](self._ll_U, num, den, have))
        _LL_LIVE[0] = True
        if solve:
            self._dt_solve()

    def _dt_solve(self) -> None:
        """dW^T = (G + a tr(G)/D I)^-1 R^T in float32 (Cholesky), NaN / failure -> 0; it serves the NEXT window."""
        G = self._dt_G
        lam = torch.diagonal(G).sum() * (float(_DT_ALPHA) / G.shape[0])
        L_, info = torch.linalg.cholesky_ex(torch.addcmul(G, self._dt_eye, lam), check_errors=False)
        X = torch.cholesky_solve(self._ll_R.t(), L_)
        torch.eq(info, 0, out=self._dt_ok)
        if _DT_ZERO:
            self._dt_ok.zero_()
        self._dt_Wtn.copy_(_LL_K["dt_g"](X, self._dt_ok))
        self._dt_kn.copy_(self._ll_k)

    def _dt_join(self, cur) -> None:
        """par4: main joins the DTTT solve lane (_dt_side) at the very end of the step."""
        e6 = self.__dict__.pop("_dt_e6", None)
        if e6 is not None:
            cur.wait_event(e6)

    def _ll_outc(self, mem, y, m_, s_):
        """m1's outc site: o0 (the original outc), phi of this window (standardized h, y; stashed for the next prefix),
        o1 = o0 + phi @ dW^T (FP32), energy-preserving rescale, exact gated select, stash."""
        oc = mem.outc
        ev = self.__dict__.get("_ll_wait")
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)
        o0 = oc(y)
        K = _LL_K
        U_ = sys.modules[type(oc).__module__]
        H, W, T, C = int(self._H), int(self._W), int(self._n_t), int(self._C)
        Co, D0, D0h = T * C, self._ll_D0, self._dt_D0h
        h = self._dt_hprod()
        hm = h if h.dim() == 2 else h[0].reshape(h.shape[1], -1).t()               # (H*W, 512): fused / conv layout
        torch.addcmul(self._dt_b[:, :D0h], hm, self._dt_a[:, :D0h], out=self._ll_phiT[:, :D0h])
        if _DT_FEAT == "hy":
            X = U_._nhwc(y) if hasattr(U_, "_nhwc") else y[0].permute(1, 2, 0).reshape(H * W, -1)
            torch.addcmul(self._dt_b[:, D0h:], X, self._dt_a[:, D0h:], out=self._ll_phiT[:, D0h:D0])
        o0m = o0[0].permute(1, 2, 0).reshape(H * W, Co)
        o1v = _ll_fp32(torch.addmm, o0m, self._ll_phiT, self._dt_Wt).view(H, W, T, C)
        o0v = o0m.view(H, W, T, C) if o0m.is_contiguous() else o0[0].permute(1, 2, 0).reshape(H, W, T, C)
        v0, _mn0 = torch.var_mean(o0v, dim=2, unbiased=False, keepdim=True)
        v1, mn1 = torch.var_mean(o1v, dim=2, unbiased=False, keepdim=True)
        out, dd = K["ll_comb"](o0v, o1v, v0, mn1, v1, self._ll_U, self._dt_ks, self._ll_cm, gate=float(_LL_GATE))
        self._ll_o0.copy_(o0v)
        self._ll_dd.copy_(dd)
        torch.stack([s_.reshape(C), m_.reshape(C)], out=self._ll_sm)
        return out.permute(2, 3, 0, 1).reshape(1, Co, H, W)

    def _ll_m2_stash(self, y2) -> None:
        """Blend gate: m2's output of this window (for the next window's utility)."""
        self._ll_u2.copy_(y2[0].permute(1, 2, 0, 3))

'''

SP4_NEW_DT = ("                if self.__dict__.get(\"_ll_on\", False) and self.__dict__.get(\"_dt_side\") is not None:\n"
              "                    # DTTT: prefix + lag-one solve on their OWN stream, forked with the others; side2 is untouched\n"
              "                    self._dt_side.wait_event(ef)\n"
              "                    with torch.cuda.stream(self._dt_side):\n"
              "                        self._ll_prefix(self._sy, have, solve=False)   # promote, residual / U, R, G\n"
              "                        _lle = torch.cuda.Event(); _lle.record(self._dt_side)   # m1 waits on it before its outc\n"
              "                        self.__dict__[\"_ll_wait\"] = _lle\n"
              "                        self._dt_solve()                               # the next window's dW (off every path)\n"
              "                        _e6 = torch.cuda.Event(); _e6.record(self._dt_side)   # main joins it at the step's end\n"
              "                        self.__dict__[\"_dt_e6\"] = _e6\n"
              + SP4_OLD)
DTJOINKC_OLD = "                self._kc_tail(y1, y2, _bfn, cur, e3, e4, e5)\n                return\n"
DTJOINKC_NEW = ("                self._kc_tail(y1, y2, _bfn, cur, e3, e4, e5)\n"
                "                if self.__dict__.get(\"_ll_on\", False):\n"
                "                    self._dt_join(cur)                   # DTTT: join the solve lane (_dt_side)\n"
                "                return\n")
DTJOINEND_OLD = ("            self._out3[2].copy_(torch.maximum(lo, hi))\n\n"
                 "    # ------------------------------------------------------------------ capture + its gate\n")
DTJOINEND_NEW = ("            self._out3[2].copy_(torch.maximum(lo, hi))\n"
                 "            if self.__dict__.get(\"_ll_on\", False):\n"
                 "                self._dt_join(cur)                       # DTTT: join the solve lane (_dt_side)\n\n"
                 "    # ------------------------------------------------------------------ capture + its gate\n")


# ---- prereg addendum P1-hy (--dt-est p1): the SAME card with LLR1's estimator class on the hy features ------------------
# dW^T = P1 R^T / max(k, 1) (ll_ws), P1 = (G_hat + alpha tr(G_hat)/D I)^-1 from the card data pool's standardized [h~; y~; 1]
# Gram (deep_ttt_probe.py --gram-out), inverted in float64 by this generator, shipped float32 (ll_p1.npy); lag 0 (served from
# the first revealed window, like v156); no running Gram, no Cholesky, no extra stream: the prefix sits at side2's head exactly
# like LLR1's.  The p1 method text is DERIVED from DT_METHODS at build time, so the exact-estimator output (--dt-est exact,
# t2_v158_dttt) stays byte-identical to the build of commit a49b936.
_P1_REPS = [
    ("""                if ms is None or tuple(ms.shape) != (2, D0) or not np.isfinite(ms).all() or not (ms[1] > 0).all():
                    ok = False
                    why = "ll_dt.npy missing / shape %s != %s / non-finite / sd <= 0" % (None if ms is None else tuple(ms.shape), (2, D0))
""", """                if ms is None or tuple(ms.shape) != (2, D0) or not np.isfinite(ms).all() or not (ms[1] > 0).all():
                    ok = False
                    why = "ll_dt.npy missing / shape %s != %s / non-finite / sd <= 0" % (None if ms is None else tuple(ms.shape), (2, D0))
                p1 = None
                if ok:
                    for d_ in (submission_dir, _HERE):
                        f_ = os.path.join(d_, "ll_p1.npy") if d_ else ""
                        if f_ and os.path.exists(f_):
                            p1 = np.load(f_, allow_pickle=False)
                            break
                    if p1 is None or tuple(p1.shape) != (D0 + 1, D0 + 1) or not np.isfinite(p1).all():
                        ok = False
                        why = "ll_p1.npy missing / shape %s != %s / non-finite" % (None if p1 is None else tuple(p1.shape), (D0 + 1, D0 + 1))
"""),
    ("""                self._dt_phiH = z(H * W, D)
                self._dt_G = z(D, D)
                self._dt_eye = torch.eye(D, dtype=torch.float32, device=d)
                try:
                    self._dt_side = torch.cuda.Stream()      # par4: the DTTT prefix + solve lane
                except Exception:
                    self._dt_side = None                     # par4 then has no DTTT lane: `_ll_strpar_ok` refuses it
                self._dt_Wt = z(D, Co)               # the solve serving this window (previous step's)
                self._dt_Wtn = z(D, Co)              # this step's solve (serves the next window)
                self._dt_ks = z(())                  # revealed windows behind the serving solve
                self._dt_kn = z(())                  # ... behind the pending one
                self._dt_ok = z(())
""", """                self._dt_P1 = torch.as_tensor(np.ascontiguousarray(p1, dtype=np.float32)).to(d)
                self._dt_Wt = z(D, Co)               # dW^T = P1 R^T / max(k, 1), refreshed by every prefix (lag 0)
                self._ll_k = z(())
                self._dt_ks = self._ll_k             # the serve count IS the running count (same tensor)
"""),
    ("""                self._ll_R = z(Co, D)
                self._ll_k = z(())
                self._ll_U = z(())
""", """                self._ll_R = z(Co, D)
                self._ll_U = z(())
"""),
    ("""                why = "DTTT-A feat %s D %d alpha %g%s, gate %s > %g, |a| max %.3e" % (""",
     """                why = "DTTT-A P1 feat %s D %d alpha %g%s, gate %s > %g, |a| max %.3e" % ("""),
    ("""        for b in (self._ll_R, self._ll_k, self._ll_U, self._ll_o0, self._ll_dd, self._ll_u2, self._dt_G, self._dt_Wt,
                  self._dt_Wtn, self._dt_ks, self._dt_kn):
            b.zero_()
""", """        for b in (self._ll_R, self._ll_k, self._ll_U, self._ll_o0, self._ll_dd, self._ll_u2, self._dt_Wt):
            b.zero_()
"""),
    ("""        return (not self.__dict__.get("_ll_on", False)) or (bool(_LL_K) and self.__dict__.get("_dt_G") is not None
                                                             and self.__dict__.get("_dt_side") is not None)
""", """        return (not self.__dict__.get("_ll_on", False)) or (bool(_LL_K) and self.__dict__.get("_dt_P1") is not None)
"""),
    ("""        self._dt_Wt.copy_(self._dt_Wtn)
        self._dt_ks.copy_(self._dt_kn)
        sm = self._ll_sm
""", """        sm = self._ll_sm
"""),
    ("""        torch.mul(self._ll_phiT, have, out=self._dt_phiH)
        _ll_fp32(self._dt_G.addmm_, self._dt_phiH.t(), self._ll_phiT)
        self._ll_k.add_(have)
        self._ll_U.copy_(K["ll_u"](self._ll_U, num, den, have))
        _LL_LIVE[0] = True
        if solve:
            self._dt_solve()
""", """        self._ll_k.add_(have)
        self._ll_U.copy_(K["ll_u"](self._ll_U, num, den, have))
        self._dt_Wt.copy_(K["ll_ws"](_ll_fp32(torch.mm, self._dt_P1, self._ll_R.t()), self._ll_k))
        if _DT_ZERO:
            self._dt_Wt.zero_()
        _LL_LIVE[0] = True
"""),
]


def dt_methods_p1():
    t = DT_METHODS
    for old, new in _P1_REPS:
        assert t.count(old) == 1, ("P1 method anchor", old[:70], t.count(old))
        t = t.replace(old, new)
    assert "_dt_solve()" not in t.split("def _dt_solve")[0], "p1 text still solves in the prefix"
    return t


def patch_submission_dt(s, gsrc, feat, alpha, zero, est="exact"):
    """--feat h|hy: the LLR1 patch set with the DTTT-A module extension / methods / par4 DTTT lane (prefix + solve on _dt_side) + the end-of-step joins.
    est "p1": the P1 method text (derived from DT_METHODS) and LLR1's side2-head prefix instead of the DTTT lane."""
    assert MARK not in s, "already LL-patched"
    assert est in ("exact", "p1"), est
    ext = DT_MOD_EXTRA.replace("__FEAT__", feat).replace("__ALPHA__", repr(float(alpha))).replace("__ZERO__", str(int(zero)))
    for name, old, new in SUB_PATCHES:
        if name == "module":
            s = _rep(s, old, MOD_BLOCK.replace("__GSRC__", gsrc) + ext + old, name)
        elif name == "methods":
            s = _rep(s, old, (DT_METHODS if est == "exact" else dt_methods_p1()) + METH_ANCHOR, name)
        elif name == "strpar_par4":
            s = _rep(s, old, SP4_NEW_DT if est == "exact" else new, name)
        else:
            s = _rep(s, old, new, name)
    s = _rep(s, DTJOINKC_OLD, DTJOINKC_NEW, "dt_join_kc")
    s = _rep(s, DTJOINEND_OLD, DTJOINEND_NEW, "dt_join_end")
    compile(s, "submission.py", "exec")
    return s


def dt_ms_from_stats(path, feat):
    """ll_dt.npy = [mu; sd] (float32) of the feature block from deep_ttt_probe.py --stats-out; sd clamped below at
    1e-3 x median(sd) of its block, exactly as the probe (deep_ttt_probe.SD_CLAMP_REL)."""
    z = np.load(path)
    out, meta = [], dict(stats=str(path), stats_md5=hashlib.md5(Path(path).read_bytes()).hexdigest(), fold=str(z["fold"]),
                         n_windows=int(z["n_windows"]), feat=feat)
    for b in (("h", "y") if feat == "hy" else ("h",)):
        mu, sd = np.asarray(z["mu_" + b], np.float64), np.asarray(z["sd_" + b], np.float64)
        floor = 1e-3 * float(np.median(sd))
        meta["n_clamped_" + b] = int((sd < floor).sum())
        out.append(np.stack([mu, np.maximum(sd, floor)], 0))
    ms = np.concatenate(out, 1)
    assert np.isfinite(ms).all() and (ms[1] > 0).all()
    meta["D0"] = int(ms.shape[1])
    return ms.astype(np.float32), meta


def _rep(s, old, new, name):
    n = s.count(old)
    assert n == 1, "anchor %s found %d times" % (name, n)
    return s.replace(old, new)


def patch_submission(s, gsrc):
    assert MARK not in s, "already LL-patched"
    for name, old, new in SUB_PATCHES:
        if name == "module":
            s = _rep(s, old, MOD_BLOCK.replace("__GSRC__", gsrc) + old, name)
        else:
            s = _rep(s, old, new, name)
    compile(s, "submission.py", "exec")
    return s


def patch_unet(s):
    assert "_ll_fn" not in s, "unet_model.py already LL-patched"
    s = _rep(s, UNET_OLD, UNET_NEW, "unet_outc")
    compile(s, "unet_model.py", "exec")
    return s


def p1_from_ghat(path, alpha):
    z = np.load(path)
    G = np.asarray(z["G_hat"], np.float64); D = G.shape[0]
    assert G.shape == (D, D) and np.isfinite(G).all()
    A = G + alpha * np.trace(G) / D * np.eye(D)
    P = np.linalg.inv(A)
    P = 0.5 * (P + P.T)
    # LAPACK self-check: one host's system python3 (numpy 1.21.5) returned a wrong inverse (|A P - I| ~ 3e2) on
    # 2026-09-26; refuse any P1 that is not an inverse to near machine precision
    resid = float(np.abs(A @ P - np.eye(D)).max())
    assert np.isfinite(P).all() and resid <= 1e-9, "P1 inverse self-check failed: max|A P - I| = %.3e (numpy %s)" % (resid, np.__version__)
    meta = dict(ghat=str(path), ghat_md5=hashlib.md5(Path(path).read_bytes()).hexdigest(), alpha=alpha, D=int(D),
                n_windows=int(z["n_windows"]) if "n_windows" in z.files else None,
                n_files=int(len(z["files"])) if "files" in z.files else None,
                cond=float(np.linalg.cond(A)), p1_absmax=float(np.abs(P).max()), inv_resid=resid, numpy=np.__version__)
    return P.astype(np.float32), meta


def npy_bytes(a):
    import io
    b = io.BytesIO(); np.save(b, np.ascontiguousarray(a, dtype="<f4"), allow_pickle=False)
    return b.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ll", action="store_true", help="apply the LL-ridge patch (without it: byte-identical copy of --base)")
    ap.add_argument("--ghat", default=None, help="G_hat npz (ll_precond_probe / ll_center_probe --ghat-out)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--p1-zero", action="store_true", help="ll_p1.npy of zeros (the bit-identity test card)")
    ap.add_argument("--gate-src", default="blend", choices=["blend", "m1"])
    ap.add_argument("--rctr", action="store_true", help="refused: RCTR failed its registered probe gate")
    ap.add_argument("--feat", default="y", choices=["y", "h", "hy"], help="y: the LLR1 card (default, unchanged); h / hy: "
                    "the DTTT-A card (exact lag-one float32 ridge on m1's last-block hidden activation, needs --stats)")
    ap.add_argument("--stats", default=None, help="--feat h|hy: deep_ttt_probe.py --stats-out npz of the card's data pool")
    ap.add_argument("--dt-alpha", type=float, default=0.1)
    ap.add_argument("--dt-zero", action="store_true", help="--feat h|hy: dW = 0 (the correction-off identity card)")
    ap.add_argument("--dt-est", default="exact", choices=["exact", "p1"], help="--feat h|hy: exact lag-one Cholesky (default) or "
                    "the fixed pool preconditioner P1 (prereg addendum P1-hy; --ghat = deep_ttt_probe --gram-out npz)")
    ap.add_argument("--manifest", default=None)
    a = ap.parse_args()
    assert not Path(a.out).exists(), "refusing to overwrite %s" % a.out
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    md5 = lambda b: hashlib.md5(b).hexdigest()
    if a.rctr:
        raise SystemExit("--rctr refused: the RCTR probe failed its registered gate (EQ dSPS +0.0133 < +0.02)")
    if not a.ll:
        shutil.copyfile(a.base, a.out)
        assert md5(Path(a.out).read_bytes()) == md5(Path(a.base).read_bytes())
        print("LLR1_BUILD_OK (no --ll: byte-identical copy)", a.out, md5(Path(a.out).read_bytes()))
        return
    if a.feat != "y":
        return main_dt(a, md5)
    assert (a.ghat is not None) != bool(a.p1_zero), "exactly one of --ghat / --p1-zero"
    with zipfile.ZipFile(a.base) as z:
        names = z.namelist(); old = {n: z.read(n) for n in names}
    assert "ll_p1.npy" not in names
    new = dict(old)
    new["submission.py"] = patch_submission(old["submission.py"].decode(), a.gate_src).encode()
    new["unet_model.py"] = patch_unet(old["unet_model.py"].decode()).encode()
    if a.p1_zero:
        P1 = np.zeros((129, 129), np.float32); pmeta = dict(p1="zeros", D=129)
    else:
        P1, pmeta = p1_from_ghat(a.ghat, a.alpha)
    new["ll_p1.npy"] = npy_bytes(P1)
    order = names + ["ll_p1.npy"]
    with zipfile.ZipFile(a.out, "x", zipfile.ZIP_DEFLATED) as z:
        for n in order:
            zi = zipfile.ZipInfo(n, date_time=(1980, 1, 1, 0, 0, 0)); zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, new[n])
    with zipfile.ZipFile(a.out) as z:
        back = {n: z.read(n) for n in z.namelist()}
    assert list(back) == order and all(back[n] == new[n] for n in order)
    changed = [n for n in names if new[n] != old[n]]
    assert changed == ["submission.py", "unet_model.py"], changed
    man = dict(out=a.out, out_md5=md5(Path(a.out).read_bytes()), base=a.base, base_md5=md5(Path(a.base).read_bytes()),
               gate_src=a.gate_src, changed=changed, added=["ll_p1.npy"], p1=pmeta,
               submission_md5_old=md5(old["submission.py"]), submission_md5_new=md5(new["submission.py"]),
               unet_md5_old=md5(old["unet_model.py"]), unet_md5_new=md5(new["unet_model.py"]),
               ll_p1_md5=md5(new["ll_p1.npy"]), members_unchanged={n: md5(old[n]) for n in names if n.endswith(".pth")},
               generator_md5=md5(Path(__file__).read_bytes()))
    if a.manifest:
        Path(a.manifest).parent.mkdir(parents=True, exist_ok=True); Path(a.manifest).write_text(json.dumps(man, indent=1))
    print("LLR1_BUILD_OK", a.out, man["out_md5"], json.dumps({k: man[k] for k in ("gate_src", "p1", "submission_md5_new", "unet_md5_new")}))


def main_dt(a, md5):
    """--feat h|hy: the DTTT-A card (adds ll_dt.npy; no ll_p1.npy)."""
    assert a.stats and not a.p1_zero, "--feat h|hy takes --stats (and optionally --dt-zero), not --p1-zero"
    assert (a.dt_est == "p1") == bool(a.ghat), "--dt-est p1 needs --ghat (deep_ttt_probe --gram-out npz); exact takes none"
    with zipfile.ZipFile(a.base) as z:
        names = z.namelist(); old = {n: z.read(n) for n in names}
    assert "ll_p1.npy" not in names and "ll_dt.npy" not in names
    new = dict(old)
    new["submission.py"] = patch_submission_dt(old["submission.py"].decode(), a.gate_src, a.feat, a.dt_alpha, a.dt_zero,
                                               a.dt_est).encode()
    new["unet_model.py"] = patch_unet(old["unet_model.py"].decode()).encode()
    ms, smeta = dt_ms_from_stats(a.stats, a.feat)
    new["ll_dt.npy"] = npy_bytes(ms)
    order = names + ["ll_dt.npy"]
    pmeta = None
    if a.dt_est == "p1":
        zg = np.load(a.ghat)
        assert str(zg["fold"]) == smeta["fold"] and str(zg["feat"]) == a.feat, ("gram fold/feat", str(zg["fold"]), str(zg["feat"]))
        P1, pmeta = p1_from_ghat(a.ghat, a.dt_alpha)
        assert P1.shape == (ms.shape[1] + 1, ms.shape[1] + 1)
        new["ll_p1.npy"] = npy_bytes(P1)
        order = order + ["ll_p1.npy"]
    with zipfile.ZipFile(a.out, "x", zipfile.ZIP_DEFLATED) as z:
        for n in order:
            zi = zipfile.ZipInfo(n, date_time=(1980, 1, 1, 0, 0, 0)); zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, new[n])
    with zipfile.ZipFile(a.out) as z:
        back = {n: z.read(n) for n in z.namelist()}
    assert list(back) == order and all(back[n] == new[n] for n in order)
    changed = [n for n in names if new[n] != old[n]]
    assert changed == ["submission.py", "unet_model.py"], changed
    man = dict(out=a.out, out_md5=md5(Path(a.out).read_bytes()), base=a.base, base_md5=md5(Path(a.base).read_bytes()),
               item="T2-DTTT-A", feat=a.feat, dt_alpha=a.dt_alpha, dt_zero=bool(a.dt_zero), gate_src=a.gate_src,
               changed=changed, added=[n for n in order if n not in names], stats=smeta, dt_est=a.dt_est, p1=pmeta,
               submission_md5_old=md5(old["submission.py"]), submission_md5_new=md5(new["submission.py"]),
               unet_md5_old=md5(old["unet_model.py"]), unet_md5_new=md5(new["unet_model.py"]),
               ll_dt_md5=md5(new["ll_dt.npy"]), members_unchanged={n: md5(old[n]) for n in names if n.endswith(".pth")},
               generator_md5=md5(Path(__file__).read_bytes()))
    if a.manifest:
        Path(a.manifest).parent.mkdir(parents=True, exist_ok=True); Path(a.manifest).write_text(json.dumps(man, indent=1))
    print("DTTT_BUILD_OK", a.out, man["out_md5"], json.dumps({k: man[k] for k in ("feat", "dt_est", "dt_alpha", "dt_zero", "gate_src", "stats",
                                                                                 "submission_md5_new", "unet_md5_new")}))


if __name__ == "__main__":
    main()
