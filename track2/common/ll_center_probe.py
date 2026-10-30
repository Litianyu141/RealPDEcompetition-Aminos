#!/usr/bin/env python3
"""T2-LLR1 / T2-RCTR probe (V200 slate, 2026-09-26): common/ll_precond_probe.py (= the committed probe_cheap.py, md5 e5c4b796)
plus two DEFAULT-OFF options.  With neither option given this file runs the identical code path as ll_precond_probe.py
(checked per window, bit for bit, on every fold before any arm read counts).

  --gate-src m1 : the causal FTL gate scores m1 alone -- U += (|pu - t|^2 - |pu + dl - t|^2) / |t|^2 with pu = m1's UNcorrected
                  output (member units) and dl = m1's served correction (member units) on the previous revealed window.
                  The default (blend) is the probe's raw-blend gate: pu = 0.5 (m1u + m2), dl = 0.5 dl_m1 (needs m2's previous u/v).
  --rctr        : stacked interval-centre sub-arm (requires --only 0 --mode fullre).  The point path is unchanged (energy-
                  preserving correction); the interval centre additionally gets 0.5 * s1 * (o_full - o_pres)[u, v], i.e.
                  0.5 * s1 * (Delta_full - Delta_pres), where o_full = o0 + dW phi (no rescale), o_pres = the served fullre output,
                  s1 = m1's in_norm scale recovered per window (least squares of the member output on its served outc output),
                  0.5 = the blend weight.  Applied only while the gate is open (zero otherwise).  The term is added at the
                  body's centre site: the extracted card's `_ctr = _ctr + 0.5 * (p - ...)  # BAND_CENTER_TRANSPORT` lines
                  get `+ getattr(self, "_rctr_add", 0.0)` (patched in a private copy of the extracted card; the band FTL then
                  scores the served centre).  The vGain gain is NOT applied (_BCDT2 = 1.0 centre = pre-head blend).
                  With --alpha none the patch is live but the term is identically zero (self-check vs REF).
  --ghat-out F --fold all81 : G_hat over all 81 legal real files (the shipped all81 card's P1; no holdout).  An arm run with
                  --fold all81 scores all 81 files (in-sample; the eager all81 twin of a card, integrity only).
"""
from __future__ import annotations

import argparse, json, os, sys, time, traceback
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("REALPDE_REPO_PROBE", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(REPO / "common/local_benchmark/splitsel")); sys.path.insert(0, str(REPO / "common"))
import score_card as SC  # noqa: E402
from band_width_fivefold import heldout_files  # noqa: E402

SPLITS = REPO / "common/local_benchmark/splits_re3_aoa2_v1.json"


def install(model, alpha, mode="full", gate=None, only=None, pool=0, est="exact", lag=0, beta=None, ema_init="first", P1=None,
            gate_src="blend", rctr=False):
    import torch
    if rctr:
        assert only == 0 and mode == "fullre", "--rctr needs --only 0 --mode fullre"
        model._rctr_add = torch.zeros_like(model._out3[0])
    members = [model.base.m1, model.base.m2]
    st = dict(dW=[None, None], G=[None, None], R=[None, None], prev=[None, None], cur=[{}, {}], T=int(model._sx.shape[1]), open=True, gate=gate, U=0.0, pend=None,
              k=[0, 0], lagbuf=[None, None], napplied=0, nsteps=0)
    model._ll_st = st

    def mk_hooks(j, m):
        assert isinstance(m.outc, torch.nn.Conv2d) and m.freq_hi is None and m.head == "" and m.out_mode == "direct"

        def h_outc(_m, inp, o):
            c = st["cur"][j]; c["y"] = inp[0].detach(); c["o"] = o.detach()
            dW = st["dW"][j]
            if dW is None or (only is not None and j != only):
                return None
            y = inp[0].double()
            if pool:
                y = y.reshape(1, pool, y.shape[1] // pool, *y.shape[2:]).mean(2)
            _, D0, H, W = y.shape
            phi = torch.cat([y.reshape(D0, -1), torch.ones(1, H * W, device=y.device, dtype=torch.float64)], 0)
            T = st["T"]; C = o.shape[1] // T
            if mode == "mean":                      # one (u, v) correction shared by every lead: time-mean only
                do = (dW @ phi).reshape(1, 2, H, W).expand(T, 2, H, W)
            else:
                do = (dW @ phi).reshape(T, 2, H, W)
            o2 = o.clone().reshape(1, T, C, H, W)
            o2[:, :, :2] = o2[:, :, :2] + do.to(o.dtype).unsqueeze(0)
            if rctr and j == 0:
                c["ofull"] = o2[:, :, :2].detach().clone()      # o0 + dW phi, before the energy-preserving rescale
            if mode == "fullre":                    # keep each pixel's fluctuation energy (u, v) of the uncorrected o
                o0 = o.reshape(1, T, C, H, W)[:, :, :2]; o1 = o2[:, :, :2]
                f0 = o0 - o0.mean(1, keepdim=True); m1 = o1.mean(1, keepdim=True); f1 = o1 - m1
                k = ((f0 * f0).sum(1, keepdim=True) / (f1 * f1).sum(1, keepdim=True).clamp_min(1e-30)).sqrt()
                o2[:, :, :2] = m1 + f1 * k
            c["oc"] = o2.reshape(o.shape).detach(); c["applied"] = bool(st["open"])
            if not st["open"]:                      # gate closed: correction computed (for the utility) but not applied
                return None
            st["napplied"] += 1
            return o2.reshape(o.shape)

        def h_mem(_m, inp, out):
            st["cur"][j]["out"] = out.detach()
            if rctr and j == 0:
                c = st["cur"][0]; add = model._rctr_add; ofull = c.pop("ofull", None)
                if ofull is None or "oc" not in c or not c.get("applied", False):
                    add.zero_()
                    return
                fo = out.detach().double(); _, T, H, W, C = fo.shape
                oc5 = c["oc"].double().reshape(1, T, C, H, W)
                oc_r = oc5.permute(0, 1, 3, 4, 2)
                s_ = []
                for ch in range(2):
                    ov, fv = oc_r[..., ch].reshape(-1), fo[..., ch].reshape(-1)
                    vo = ((ov - ov.mean()) ** 2).mean()
                    s_.append(((ov - ov.mean()) * (fv - fv.mean())).mean() / vo.clamp_min(1e-30))
                s1 = torch.stack(s_).reshape(1, 1, 1, 1, 2)
                dff = (ofull.double() - oc5[:, :, :2]).permute(0, 1, 3, 4, 2)
                add.zero_(); add[..., :2] = (0.5 * s1 * dff).to(add.dtype)
                st["rctr_applied"] = st.get("rctr_applied", 0) + 1
        m.outc.register_forward_hook(h_outc); m.register_forward_hook(h_mem)

    for j, m in enumerate(members):
        mk_hooks(j, m)

    def reset():
        for j in range(2):
            st["dW"][j] = st["G"][j] = st["R"][j] = st["prev"][j] = st["lagbuf"][j] = None
            st["k"][j] = 0
        st["U"] = 0.0; st["pend"] = None; st["open"] = gate is None

    orig_reset = model.reset_ttt_state

    def reset_ttt_state():
        orig_reset(); reset()

    def ttt_step(input_norm, prev_target_norm=None):
        m = model
        st["nsteps"] += 1
        _xin = torch.as_tensor(input_norm)
        m._sx.copy_(_xin.to(m._sx.device))
        hp = prev_target_norm is not None
        if hp:
            m._sy.copy_(torch.as_tensor(prev_target_norm).to(m._sy.device))
        m._have.fill_(1.0 if hp else 0.0)
        if alpha is not None and gate is not None:
            if hp and st["pend"] is not None:
                pu, dl = st["pend"]; t_ = m._sy.double()[..., :2]
                e0 = ((pu - t_) ** 2).sum(); e1 = ((pu + dl - t_) ** 2).sum()
                st["U"] += float((e0 - e1) / (t_ * t_).sum().clamp_min(1e-30))
            st["open"] = st["U"] > gate
        if alpha is not None:
            for j in range(2):
                if only is not None and j != only:
                    continue                        # never applied: no estimator work (reference output unchanged)
                pv = st["prev"][j]
                if hp and pv is not None:
                    y, o, s_t, m_t = pv
                    _, D0, H, W = y.shape; T = m._sy.shape[1]; C = o.shape[1] // T
                    phi = torch.cat([y.reshape(D0, -1), torch.ones(1, H * W, device=y.device, dtype=torch.float64)], 0)
                    o_r = o.reshape(1, T, C, H, W).permute(0, 1, 3, 4, 2)
                    r = (m._sy.double()[..., :2] - m_t) / s_t - o_r[..., :2]
                    rr = r[0].mean(0).permute(2, 0, 1).reshape(2, H * W) if mode == "mean" else r[0].permute(0, 3, 1, 2).reshape(T * 2, H * W)
                    rp = rr @ phi.T
                    st["k"][j] += 1
                    if est == "exact":
                        g = phi @ phi.T
                        st["G"][j] = g if st["G"][j] is None else st["G"][j] + g
                        st["R"][j] = rp if st["R"][j] is None else st["R"][j] + rp
                    elif est == "mean":
                        st["R"][j] = rp if st["R"][j] is None else st["R"][j] + rp
                    elif est == "ema":
                        kk = st["k"][j]
                        if ema_init == "zero":
                            b = beta
                            st["R"][j] = b * rp if st["R"][j] is None else (1 - b) * st["R"][j] + b * rp
                        elif ema_init == "first":
                            st["R"][j] = rp if st["R"][j] is None else (1 - beta) * st["R"][j] + beta * rp
                        elif ema_init == "warm":
                            b = max(beta, 1.0 / kk)
                            st["R"][j] = rp if st["R"][j] is None else (1 - b) * st["R"][j] + b * rp
                        else:
                            raise ValueError(ema_init)
                    else:
                        raise ValueError(est)
                new = None
                if est == "exact":
                    if st["G"][j] is not None:
                        G = st["G"][j]; D = G.shape[0]
                        lam = alpha * torch.trace(G) / D
                        new = torch.linalg.solve(G + lam * torch.eye(D, device=G.device, dtype=torch.float64), st["R"][j].T).T
                elif est == "mean":
                    if st["R"][j] is not None:
                        new = (st["R"][j] / st["k"][j]) @ P1
                elif est == "ema":
                    if st["R"][j] is not None:
                        new = st["R"][j] @ P1
                if lag:
                    st["dW"][j] = st["lagbuf"][j]; st["lagbuf"][j] = new
                elif new is not None:
                    st["dW"][j] = new
        m._body()
        if alpha is not None:
            for j in range(2):
                c = st["cur"][j]
                y, o, out = c["y"].double(), c["o"].double(), c["out"].double()
                if pool:
                    y = y.reshape(1, pool, y.shape[1] // pool, *y.shape[2:]).mean(2)
                c["oc_last"] = c.get("oc")
                oc = (c["oc"] if ("oc" in c and c.get("applied", True)) else c["o"]).double(); c.pop("oc", None)   # the o actually served
                _, T, H, W, C = out.shape
                oc_r = oc.reshape(1, T, C, H, W).permute(0, 1, 3, 4, 2)
                sc_, mc_ = [], []
                for ch in range(2):
                    ov, fv = oc_r[..., ch].reshape(-1), out[..., ch].reshape(-1)
                    vo = ((ov - ov.mean()) ** 2).mean()
                    s_ = ((ov - ov.mean()) * (fv - fv.mean())).mean() / vo.clamp_min(1e-30)
                    sc_.append(s_); mc_.append(fv.mean() - s_ * ov.mean())
                st["prev"][j] = (y, o, torch.stack(sc_).reshape(1, 1, 1, 1, 2), torch.stack(mc_).reshape(1, 1, 1, 1, 2))
            if gate is not None:
                pu, dl = 0.0, 0.0
                for j in range(2):
                    c = st["cur"][j]; y_, o_, s_t, m_t = st["prev"][j]
                    _, T, H, W, C = c["out"].shape
                    ou = o_.reshape(1, T, C, H, W).permute(0, 1, 3, 4, 2)[..., :2] * s_t + m_t          # uncorrected member output
                    oc = c.get("oc_last")
                    d_ = 0.0 if oc is None else (oc.double().reshape(1, T, C, H, W).permute(0, 1, 3, 4, 2)[..., :2] - o_.reshape(1, T, C, H, W).permute(0, 1, 3, 4, 2)[..., :2]) * s_t
                    if gate_src == "m1":               # m1-only gate: m1's own uncorrected output and served correction
                        if j == 0:
                            pu, dl = ou, d_
                    else:
                        pu = pu + 0.5 * ou; dl = dl + 0.5 * d_
                st["pend"] = (pu, dl)
        c3 = m._out3.clone()
        return c3[0], {"adapt_loss": None, "lower": c3[1], "upper": c3[2]}

    model.ttt_step = ttt_step
    model.reset_ttt_state = reset_ttt_state
    return model


def precompute_ghat(a):
    """Mean per-window Gram of m1's [y; 1] (outc input) over the fold's TRAINING windows, card loaded as in run_t2."""
    import torch
    sys.path.insert(0, str(SC.T2_KIT))
    from local_eval import Normalizer
    if a.fold == "all81":                  # all81 card (ll_p1.pt of the shipped card): every legal real file, no holdout
        files = SC.real_files()
        assert len(files) == 81 and "7575_0.h5" not in set(Path(f).name for f in files), "all81 file set"
    else:
        d = json.loads(SPLITS.read_text()); fold = [f for f in d["folds"] if f["id"] == a.fold]; assert len(fold) == 1
        train = set(fold[0]["train_files"]); held = set(Path(f).name for f in heldout_files(a.fold))
        files = [f for f in SC.real_files() if Path(f).name not in held]
        assert set(Path(f).name for f in files) == train and not (train & held), "train/held mismatch"
    sub = SC.extract(Path(a.card), Path(a.work))
    mod = SC.import_submission(sub); os.chdir(sub)
    model = mod.get_ttt_model(str(sub), "cuda")
    m1 = model.base.m1
    training_flags = dict(base=bool(model.base.training), m1=bool(m1.training))
    buf = {}
    hk = m1.outc.register_forward_hook(lambda _m, inp, o: buf.__setitem__("y", inp[0].detach()))
    norm = Normalizer(SC.STATS)
    G = None; n = 0
    with torch.no_grad():
        for fn in files:
            x, t0s = SC.windows_of(fn)
            for t0 in t0s:
                inp = torch.from_numpy(x[t0:t0 + SC.T_IN]).unsqueeze(0)
                tgt = torch.from_numpy(x[t0 + SC.T_IN:t0 + SC.HOR]).unsqueeze(0)
                inp_n, _ = norm.preprocess(inp, tgt)
                model._sx.copy_(inp_n.to(model._sx.device))
                m1(model._sx)
                y = buf["y"].double(); _, D0, H, W = y.shape
                phi = torch.cat([y.reshape(D0, -1), torch.ones(1, H * W, device=y.device, dtype=torch.float64)], 0)
                g = phi @ phi.T
                G = g if G is None else G + g
                n += 1
    hk.remove()
    Gh = (G / n).cpu().numpy()
    ev = np.linalg.eigvalsh(Gh)
    np.savez(a.ghat_out, G_hat=Gh, n_windows=n, files=np.array([Path(f).name for f in files]))
    meta = dict(status="OK", fold=a.fold, card=a.card, card_sha256=SC.archive_sha256(a.card), n_files=len(files), n_windows=n,
                trace=float(np.trace(Gh)), D=int(Gh.shape[0]), eig_min=float(ev[0]), eig_max=float(ev[-1]), training_flags=training_flags)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True); ap.add_argument("--fold", required=True)
    ap.add_argument("--alpha", default="1.0", help="ridge strength (x tr(G)/D) or 'none'")
    ap.add_argument("--work", required=True); ap.add_argument("--out", default=None); ap.add_argument("--gpu", default="0")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--mode", default="full", choices=["full", "mean", "fullre"])
    ap.add_argument("--only", type=int, default=None, choices=[0, 1])
    ap.add_argument("--pool", type=int, default=0)
    ap.add_argument("--gate", type=float, default=None)
    ap.add_argument("--est", default="exact", choices=["exact", "mean", "ema"])
    ap.add_argument("--lag", type=int, default=0, choices=[0, 1])
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--ema-init", default="first", choices=["zero", "first", "warm"])
    ap.add_argument("--ghat", default=None, help="G_hat npz from --ghat-out (needed for --est mean/ema)")
    ap.add_argument("--ghat-out", default=None, help="precompute mode")
    ap.add_argument("--gate-src", default="blend", choices=["blend", "m1"], help="gate utility source (default: raw blend)")
    ap.add_argument("--rctr", action="store_true", help="stacked interval-centre term 0.5 s1 (o_full - o_pres)[u,v]")
    a = ap.parse_args()
    for k in ("card", "work", "out", "ghat", "ghat_out"):
        if getattr(a, k):
            setattr(a, k, os.path.abspath(getattr(a, k)))
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    t_start = time.time()
    if a.ghat_out:
        try:
            meta = precompute_ghat(a)
        except Exception as e:  # noqa: BLE001
            meta = dict(status="FAIL", error="%s: %s" % (type(e).__name__, e), traceback=traceback.format_exc())
        meta["elapsed_s"] = time.time() - t_start
        Path(a.ghat_out + ".json").write_text(json.dumps(meta, indent=1, default=str))
        print("ghat", a.fold, meta["status"], meta.get("n_windows"), "%.0fs" % meta["elapsed_s"], meta.get("error", ""))
        return
    alpha = None if a.alpha == "none" else float(a.alpha)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(card=a.card, fold=a.fold, alpha=a.alpha, mode=a.mode, gate=a.gate, only=a.only, pool=a.pool, est=a.est, lag=a.lag,
                beta=a.beta, ema_init=a.ema_init, ghat=a.ghat, gate_src=a.gate_src, rctr=bool(a.rctr), status="FAIL")
    try:
        import torch
        meta["card_sha256"] = SC.archive_sha256(a.card)
        P1 = None
        if a.est in ("mean", "ema"):
            assert a.ghat and alpha is not None
            if a.est == "ema":
                assert a.beta is not None
            z = np.load(a.ghat); Gh = torch.from_numpy(z["G_hat"]).double().cuda(); D = Gh.shape[0]
            P1 = torch.linalg.inv(Gh + alpha * torch.trace(Gh) / D * torch.eye(D, device=Gh.device, dtype=torch.float64))
            P1 = 0.5 * (P1 + P1.T)
            meta["ghat_n_windows"] = int(z["n_windows"])
        sc = SC.load_scoring(); files = SC.real_files() if a.fold == "all81" else heldout_files(a.fold)   # all81: in-sample twin
        if a.max_files:
            files = files[:a.max_files]
        sub = SC.extract(Path(a.card), Path(a.work))
        if a.rctr:                                  # private patched copy: SC.extract verifies the pristine extraction
            import re, shutil
            psub = Path(a.work) / ("%s__rctr_%d_%d" % (sub.name, os.getpid(), int(time.time())))
            shutil.copytree(sub, psub)
            f = psub / "submission.py"; t = f.read_text(); n = t.count("# BAND_CENTER_TRANSPORT")
            t2 = re.sub(r"(\n[ ]*_ctr = _ctr \+ 0\.5 \* \(p - (?:self\._sg_pg|pg)\))(  # BAND_CENTER_TRANSPORT)",
                        r'\1 + getattr(self, "_rctr_add", 0.0)\2', t)
            assert n >= 1 and t2.count('+ getattr(self, "_rctr_add", 0.0)') == n, "RCTR patch sites"
            f.write_text(t2); sub = psub; meta["rctr_patch_sites"] = n; meta["rctr_sub"] = str(psub)
        _imp = SC.import_submission
        holder = {}

        def imp(sub_dir):
            mod = _imp(sub_dir); g0 = mod.get_ttt_model

            def g(d, dev):
                mdl = install(g0(d, dev), alpha, a.mode, a.gate, a.only, a.pool, a.est, a.lag, a.beta, a.ema_init, P1,
                              a.gate_src, a.rctr)
                holder["m"] = mdl
                return mdl
            mod.get_ttt_model = g
            return mod
        SC.import_submission = imp
        opts = argparse.Namespace(set=None, online=None, fluct_mult=1.0, band_mult=1.0, jgrid=False, _JS=[], _JC=[])
        rows, checks, extra = SC.run_t2(sub, sc, files, "cuda", opts)
        R = np.array([r[2:] for r in rows], np.float64)
        np.savez_compressed(str(out), files=np.array([r[0] for r in rows]), t0=np.array([r[1] for r in rows]),
                            rel=R[:, 0], tke=R[:, 1], mvpe=R[:, 2], n=R[:, 4].astype(np.int64),
                            S_mid_cf=R[:, 3].reshape(-1, 1, 1, 1), mgrid=np.array([1.0]))
        stt = getattr(holder.get("m"), "_ll_st", None)
        if stt is not None:
            meta["member0_applied_windows"] = int(stt["napplied"]); meta["steps"] = int(stt["nsteps"])
            meta["base_training"] = bool(holder["m"].base.training)
            meta["rctr_applied_windows"] = int(stt.get("rctr_applied", 0))
        meta.update(status="OK", n_windows=len(rows), selfcheck=checks, **extra)
    except Exception as e:  # noqa: BLE001
        meta.update(error="%s: %s" % (type(e).__name__, e), traceback=traceback.format_exc())
    meta["elapsed_s"] = time.time() - t_start
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=1, default=str))
    print(a.fold, a.alpha, a.est, a.lag, a.beta, a.ema_init, meta["status"], meta.get("n_windows"), "%.0fs" % meta["elapsed_s"], meta.get("error", ""))


if __name__ == "__main__":
    main()
