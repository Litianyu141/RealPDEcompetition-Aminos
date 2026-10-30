#!/usr/bin/env python3
"""Build AND offline-verify a Track-2 submission bundle in one shot.

Writes Track2/U-net/submissions/<name>/{submission.py,unet_model.py,ttt_model.py,model.pth}, zips it,
then re-imports the PACKAGED code (not the source tree) and drives it through the official streaming
protocol + scoring.py on the held-out val split. Refuses to leave a zip behind if verification fails,
so a broken bundle can never reach the platform.

    python build_t2_bundle.py --checkpoint ... --ttt-lr 1e-3 --ttt-steps 1 --halfwidth 0.2 --name t2_unet_v2
"""
import json
import os, sys, json, time, shutil, zipfile, argparse, importlib.util
from pathlib import Path
import numpy as np

# Derived from this file's own location (2026-08-12): the absolute hardcode silently
# escaped a fresh clone back to the production tree.  NB: run these tools from their
# in-repo location (the chains do); a /tmp copy would derive the wrong root.
REPO = Path(__file__).resolve().parents[2]
# 2026-08-10: kit v4 score_sps is the PRE-RESET logistic 100/(1+exp(-x)); kit v6 is the
# live linear 100*clamp(x,0,1) AND excludes mask/FOV cells.  Measured: the band argmax is
# preserved but the ordering of non-optimal points is NOT, and local sps drops ~63.1 -> ~52.1.
T2_KIT = REPO / "Track2/realpde_t2_starting_kit_v6"
T2_UNET = REPO / "Track2/U-net"
STATS = T2_KIT / "example_data/mean_std_real.pt"
MAX_MB = 64.0  # ceiling with its calibration points kept explicit: our OWN 201 MB bundle
# failed platform download twice, and the forum's transient-storage failures involved
# ~186 MB bundles.  64 keeps a 3x margin under both; 40 predated any 2-model zip and the
# 45.7 MB U-net+CNO blend is a legitimate payload (raised 2026-08-01).


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--checkpoint2", default="", help="second forecaster -> model2.pth; the blend template averages the two")
    p.add_argument("--template", default=str(T2_UNET / "submission_t2_v3.py"))  # repo-relative (F7, 2026-08-12)
    p.add_argument("--ttt-lr", type=float, required=True)
    p.add_argument("--ttt-steps", type=int, default=1)
    p.add_argument("--halfwidth", type=float, required=True)
    p.add_argument("--calib", default="", help="calib.npz to ship alongside the checkpoint (optional)")
    p.add_argument("--allow-calib-mismatch", action="store_true",
        help="package a calib whose provenance stamp names a different forecaster (default: refuse)")
    p.add_argument("--online", default="",
                   help="JSON object of online-calibration rates; substituted into the "
                        "template's __ONLINE__ placeholder. Empty dict == v3 behaviour.")
    p.add_argument("--head", default="", help="vgain head .pth to ship (turns the head on)")
    p.add_argument("--sband", default="", help="Track-1 sband head .pth to ship as sband.pth (+ sband_head.py); SBPORT templates turn the learned per-element band on iff present")
    p.add_argument("--name", required=True)
    p.add_argument("--data-dir", default=str(REPO / "data/RealPDE_competition_dataset/gdrive_raw/local_bench/val"))
    p.add_argument("--gpu", default=None)
    return p.parse_args()




# ---------------------------------------------------------------------------------------------
# CALIB PROVENANCE (2026-08-13).  A band fitted against one forecaster's residuals is the wrong
# band for another (t1_v54_tsb shipped exactly that mismatch).  Producers stamp; builders refuse.
# Stamped values are plain strings so np.savez keeps them without pickle; missing -> "".
# ---------------------------------------------------------------------------------------------
def _prov_md5(path):
    import hashlib
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()
    except Exception:
        return ""


def _prov_stamp(d, ckpt=None, fit_stream="", producer=""):
    """Add provenance keys to a calib dict in place and return it."""
    import time as _t
    import os as _o
    import numpy as _np
    d["prov_ckpt_md5"] = _np.str_(_prov_md5(ckpt) if ckpt else "")
    d["prov_ckpt_name"] = _np.str_(_o.path.basename(str(ckpt)) if ckpt else "")
    d["prov_fit_stream"] = _np.str_(str(fit_stream))
    d["prov_producer"] = _np.str_(str(producer))
    d["prov_when"] = _np.str_(_t.strftime("%Y-%m-%dT%H:%M:%S"))
    return d


def _prov_check(calib_path, ckpt_path, allow=False):
    """Refuse a calib/checkpoint mismatch. Unstamped calibs warn only (every historical artifact)."""
    import numpy as _np
    if not calib_path:
        return
    try:
        with _np.load(calib_path, allow_pickle=True) as z:
            stamped = str(z["prov_ckpt_md5"]) if "prov_ckpt_md5" in z.files else ""
            name = str(z["prov_ckpt_name"]) if "prov_ckpt_name" in z.files else ""
    except Exception as e:
        print(f"[prov] WARNING: could not read provenance from {calib_path}: {e}")
        return
    if not stamped:
        print(f"[prov] WARNING: {calib_path} carries no provenance stamp (pre-2026-08-13 artifact); "
              f"cannot verify it was fitted on this forecaster's residuals")
        return
    actual = _prov_md5(ckpt_path)
    if actual and stamped != actual:
        msg = (f"[prov] CALIB/CHECKPOINT MISMATCH\n"
               f"        calib      : {calib_path}\n"
               f"        fitted on  : {name or '?'} md5 {stamped}\n"
               f"        packaging  : {ckpt_path} md5 {actual}\n"
               f"        A band fitted on other residuals is the wrong band. Refit it (see BCP), or "
               f"pass --allow-calib-mismatch to override deliberately.")
        if allow:
            print(msg.replace("MISMATCH", "MISMATCH (OVERRIDDEN by --allow-calib-mismatch)"))
        else:
            raise SystemExit(msg)
    else:
        print(f"[prov] ok: calib provenance matches the packaged checkpoint ({actual[:8]}...)")


def vendor_model_dependencies(out_dir: Path, checkpoints):
    """Copy model code required by any member; return its payload whitelist entries."""
    import torch
    arches = []
    for checkpoint in checkpoints:
        try:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            arch = str((saved.get("meta", {}) or {}).get("arch", "")) if isinstance(saved, dict) else ""
            del saved
        except Exception:
            # Keep legacy checkpoint handling; packaged strict loading remains authoritative.
            continue
        if arch not in arches:
            arches.append(arch)
    standalone = {"no82": Path(__file__).resolve().parent / "no82_model.py",
                  "lsm82": Path(__file__).resolve().parent / "lsm82_model.py",
                  "sinenet82": Path(__file__).resolve().parent / "sinenet82_model.py",
                  "uno82": Path(__file__).resolve().parent / "uno82_model.py",
                  "cno2d82": Path(__file__).resolve().parent / "cno2d82_model.py",
                  "dpot82": Path(__file__).resolve().parent / "dpot82_model.py",
                  "galerkin82": Path(__file__).resolve().parent / "galerkin82_model.py",
                  "dcno82": Path(__file__).resolve().parent / "dcno82_model.py",
                  "transolver2d82": Path(__file__).resolve().parent / "transolver2d82_model.py"}
    required = [source for arch, source in standalone.items() if arch in arches]
    for source in required:
        if not source.is_file():
            raise FileNotFoundError(f"checkpoint requires standalone model code: {source}")
    payload = []
    if any(arch in ("cno", "fno", "transolver") for arch in arches):
        kit = REPO / "Track1/realpde_t1_starting_kit_v9"
        shutil.copy(kit / "load_baseline.py", out_dir / "load_baseline.py")
        shutil.copytree(kit / "rpde_baselines", out_dir / "rpde_baselines",
                        ignore=shutil.ignore_patterns("__pycache__"))
        payload = ["load_baseline.py"] + [str(q.relative_to(out_dir))
                   for q in (out_dir / "rpde_baselines").rglob("*") if q.is_file()]
    for source in required:
        shutil.copy(source, out_dir / source.name)
        if not (out_dir / source.name).is_file():
            raise RuntimeError(f"required model module absent from bundle: {source.name}")
        payload.append(source.name)
    if payload:
        print(f"# vendored model code for member architectures {arches}: {len(payload)} files")
    return payload


def main():
    a = parse_args()
    _prov_check(getattr(a, "calib", None), a.checkpoint, getattr(a, "allow_calib_mismatch", False))
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    import torch

    out_dir = T2_UNET / "submissions" / a.name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # ---- 1. materialise the bundle ----
    tpl = Path(a.template).read_text()
    for k, v in [("__TTT_LR__", repr(a.ttt_lr)), ("__TTT_STEPS__", repr(a.ttt_steps)),
                 ("__HALFWIDTH__", repr(a.halfwidth))]:
        assert k in tpl, f"template placeholder {k} missing"
        tpl = tpl.replace(k, v)
    # __ONLINE__ is OPTIONAL so submission_t2_v3.py still builds unchanged.
    online = {}
    if "__ONLINE__" in tpl:
        online = json.loads(a.online) if a.online else {}
        assert isinstance(online, dict), "--online must be a JSON object"
        tpl = tpl.replace("__ONLINE__", repr(online))
        print("# online knobs: " + json.dumps(online, sort_keys=True))
    else:
        assert not a.online, "--online given but template has no __ONLINE__ placeholder"
    (out_dir / "submission.py").write_text(tpl)
    # THE MODEL DEFINITION THAT SHIPS MUST BE THE ONE THE CHECKPOINT WAS TRAINED WITH.  Track 2
    # forecasters are Track-1 checkpoints, but this line used to copy Track2/U-net/unet_model.py --
    # a 2.4 kB file last touched 2026-07-23, i.e. the original baseline, against Track 1's 25 kB
    # developed module.  It happened to strict-load because the base architecture never changed, so
    # the bug was invisible: every Track-2 bundle shipped a stale definition and NO architecture key
    # (in_norm, fuse, in_aug) could ever reach the evaluator.  in_norm adds no parameters, so such a
    # load SUCCEEDS and the submission runs a model that does not exist.
    shutil.copy(REPO / "Track1/U-net/unet_model.py", out_dir / "unet_model.py")
    shutil.copy(T2_KIT / "ttt_model.py", out_dir / "ttt_model.py")
    shutil.copy(a.checkpoint, out_dir / "model.pth")
    PAYLOAD = ["submission.py", "unet_model.py", "ttt_model.py", "model.pth"]
    if a.checkpoint2:
        shutil.copy(a.checkpoint2, out_dir / "model2.pth")
        PAYLOAD.append("model2.pth")
    PAYLOAD += vendor_model_dependencies(out_dir, [a.checkpoint] + ([a.checkpoint2] if a.checkpoint2 else []))
    if a.head:
        # the learned variance-deficit field: weights + the module defining the class.  submission.py
        # falls back to the frozen gamma if either is missing, so shipping them IS the switch.
        shutil.copy(a.head, out_dir / "vgain.pth")
        shutil.copy(REPO / "Track1/U-net/vgain_head.py", out_dir / "vgain_head.py")
        PAYLOAD += ["vgain.pth", "vgain_head.py"]
    if a.sband:
        # SBPORT (2026-08-23): Track 1 learned per-element band head; the template reads its
        # sb_edge/sb_lo/sb_hi table from calib.npz and stays on the pe/pc path if either is absent.
        shutil.copy(a.sband, out_dir / "sband.pth")
        shutil.copy(REPO / "Track1/U-net/sband_head.py", out_dir / "sband_head.py")
        PAYLOAD += ["sband.pth", "sband_head.py"]
    if a.calib:
        shutil.copy(a.calib, out_dir / "calib.npz")
        PAYLOAD.append("calib.npz")
        with np.load(out_dir / "calib.npz") as z:
            print("# shipping calib.npz: " + ", ".join(f"{k}{z[k].shape}" for k in z.files))

    # ---- 2. verify the PACKAGED code end-to-end against the official scorer ----
    # A previously imported repository loader/model must not replace a vendored sibling.
    for cached in ("unet_model", "no82_model", "lsm82_model", "sinenet82_model", "uno82_model", "dpot82_model", "cno2d82_model", "galerkin82_model", "dcno82_model", "transolver2d82_model"):
        sys.modules.pop(cached, None)
    sys.path.insert(0, str(T2_KIT))
    sys.path.insert(0, str(out_dir))
    from local_eval import build_stream, Normalizer
    import scoring as official

    spec = importlib.util.spec_from_file_location("packaged_submission", out_dir / "submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = mod.get_ttt_model(str(out_dir), dev)
    print(f"# packaged get_ttt_model OK: ttt_lr={mod.TTT_LR} steps={mod.TTT_STEPS} hw={mod.HALFWIDTH}")

    data_dir = Path(a.data_dir)
    normalizer = Normalizer(data_dir / "mean_std_real.pt")
    preds, lows, ups, tgts, times = [], [], [], [], []
    prev_pair = None
    for step in build_stream(data_dir):
        inp = step["input"].unsqueeze(0)
        tgt = step["target"].unsqueeze(0)
        if step["is_first"]:
            model.reset_ttt_state()
            prev_pair = None
        inp_norm, tgt_norm = normalizer.preprocess(inp, tgt)
        prev_target = prev_pair[1] if prev_pair is not None else None
        t0 = time.perf_counter()
        pred_norm, info = model.ttt_step(inp_norm, prev_target)
        times.append(time.perf_counter() - t0)
        prev_pair = (inp_norm.detach(), tgt_norm.detach())
        # denormalize prediction AND bounds exactly like the evaluator does
        preds.append(normalizer.postprocess_pred(torch.as_tensor(pred_norm).detach().cpu()).squeeze(0).numpy().astype(np.float32))
        lows.append(normalizer.postprocess_pred(torch.as_tensor(info["lower"]).detach().cpu()).squeeze(0).numpy().astype(np.float32))
        ups.append(normalizer.postprocess_pred(torch.as_tensor(info["upper"]).detach().cpu()).squeeze(0).numpy().astype(np.float32))
        tgts.append(tgt.squeeze(0).cpu().numpy().astype(np.float32))

    # ---- 2b. CAPTURE-ORDER GUARD (2026-08-28).  A correctness requirement, not a preference. ----
    # torch 2.2.2 binds a CUDA graph's private pool to the CAPTURE STREAM ONLY, so allocations made
    # on a SIDE stream during capture escape into the ordinary allocator; `torch.cuda.graph
    # .__enter__` then calls `empty_cache()`, so ANY capture performed after a multi-stream one
    # frees memory the earlier graph still addresses.  The failure is SILENT corruption, not an
    # abort.  Track 1's card dies of exactly this (11 bucket graphs in a loop).
    #
    # So: if a multi-stream graph is what actually ships, it must be the LAST capture the module
    # performed, and the module must contain no `empty_cache()` of its own.  Both are checked
    # against the PACKAGED module -- `_CAP_LOG` is appended at the single capture site, so it is a
    # real execution record and not a declaration.
    #
    # Written to be incapable of a false refusal on the happy path: templates older than v51 have
    # no `_CAP_LOG` (guard skipped), and a card that ships the SEQUENTIAL body has no multi-stream
    # graph to protect (guard skipped).  Only the case it exists for can trip it.
    _cap_log = list(getattr(mod, "_CAP_LOG", []) or [])
    _sp_mode = (getattr(mod, "_STRPAR_MODE", [""]) or [""])[0]
    if _cap_log and _sp_mode:
        _last = _cap_log[-1]
        if not (isinstance(_last, (tuple, list)) and _last[0] == "multistream" and _last[1] == _sp_mode):
            raise SystemExit(
                "[capture-order] REFUSED: the shipping graph is multi-stream "
                f"(mode {_sp_mode!r}) but the LAST capture was {_last!r}.\n"
                f"        capture log: {_cap_log}\n"
                "        On torch 2.2.2 a capture after a multi-stream capture frees memory the\n"
                "        earlier graph still addresses (side-stream capture allocations escape the\n"
                "        private pool; torch.cuda.graph.__enter__ calls empty_cache()).  The result\n"
                "        is SILENT corruption.  Do not work around this -- remove the later capture.")
        print(f"[capture-order] ok: {len(_cap_log)} capture(s), the shipping "
              f"multi-stream graph ({_sp_mode}) is the last  {_cap_log}")
        # Scan the AST, NOT the text.  The template's own CAPTURE-ORDER comment block explains
        # the empty_cache mechanism and therefore contains the string five times -- a substring
        # check refuses the happy path, which is the failure mode a fail-loud guard is most
        # likely to have (and this one did, on its first run, before it ever shipped).  An AST
        # walk sees calls and ignores comments and docstrings.
        import ast as _ast
        _src = (out_dir / "submission.py").read_text()
        _bad = []
        try:
            for _n in _ast.walk(_ast.parse(_src)):
                if isinstance(_n, _ast.Call):
                    _f = _n.func
                    _nm = _f.attr if isinstance(_f, _ast.Attribute) else (
                          _f.id if isinstance(_f, _ast.Name) else "")
                    if _nm in ("empty_cache", "reset_peak_memory_stats"):
                        _bad.append(f"{_nm}() at line {_n.lineno}")
        except SyntaxError as _se:
            raise SystemExit(f"[capture-order] REFUSED: packaged submission.py does not parse: {_se}")
        if _bad:
            raise SystemExit(
                "[capture-order] REFUSED: the packaged submission.py CALLS " + ", ".join(_bad) + ".\n"
                "        With a multi-stream graph live on torch 2.2.2 that frees memory the graph\n"
                "        still addresses.  See the CAPTURE-ORDER block in the template.")
        print("[capture-order] ok: packaged submission.py makes no empty_cache() call (AST-checked)")

    P = np.stack(preds, 0); L = np.stack(lows, 0); U = np.stack(ups, 0); T = np.stack(tgts, 0)
    assert np.isfinite(P).all() and np.isfinite(L).all() and np.isfinite(U).all(), "non-finite output -> score 0"
    assert (L <= U).all(), "lower > upper -> score 0"
    assert P.shape == T.shape, f"shape {P.shape} != {T.shape} -> score 0"
    c = official.measured_channels(T)
    sub = {
        "rel_l2_score": official.score_error(float(np.mean(official.rel_l2_per_sample(P, T, c)))),
        "tke_score": official.score_error(float(np.mean(official.tke_rel_l2_per_sample(P, T, c)))),
        "mvpe_score": official.score_error(official.mvpe_rel_l2(P, T)),
        "time_score": official.score_time(float(np.mean(times))),
    }
    sps, cov = official.aggregate_sps(P, T, c, lower=L, upper=U)
    sub["sps_score"] = float(official.score_sps(sps))
    final = float(np.mean(list(sub.values())))

    # ---- 3. zip only after verification passed ----
    zip_path = T2_UNET / "submissions" / f"{a.name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    # Whitelist, never iterdir(): verification above imports the packaged code, which leaves a
    # __pycache__/ directory behind. Zipping the bundle dir wholesale would put a stray directory
    # entry (and potentially stale .pyc files) into the archive the platform ingests.
    # RELAYOUT 2026-09-07: DETERMINISTIC zip -- sorted, ZIP_STORED, fixed timestamps -- so the
    # same card directory gives the same bytes/md5 on every box; the upload zip is made on the Mac by
    # tools/make_submission_zip.py from the synced directory, and a server regenerates it instead of keeping copies.
    sys.path.insert(0, str(REPO / "common"))
    from make_submission_zip import make_zip
    make_zip(out_dir, zip_path, names=list(PAYLOAD))
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        assert sorted(names) == sorted(PAYLOAD), f"unexpected archive contents: {names}"
        assert z.testzip() is None, "archive failed CRC self-test"
    mb = zip_path.stat().st_size / 1e6
    assert mb <= MAX_MB, f"bundle {mb:.1f} MB exceeds the {MAX_MB} MB safety ceiling"

    rep = {"name": a.name, "checkpoint": a.checkpoint, "head": a.head,
           # 2026-08-19: T2's report recorded neither the TEMPLATE nor the second
           # forecaster, so nothing on disk said the live card is a blend built from
           # submission_t2_v47_fp16.py.  Both are required to re-anchor reproduce_t2.sh.
           "template": a.template, "checkpoint2": a.checkpoint2,
           "ttt_lr": a.ttt_lr, "ttt_steps": a.ttt_steps,
           "halfwidth": a.halfwidth, "calib": a.calib,
           # 2026-08-23 (SBPORT): record the sband head too, or live_card.py cannot md5-anchor
           # sband.pth and reproduce_t2.sh has no producer target for it (D6-class, found on v49).
           "sband": a.sband, "online": online, "zip": str(zip_path), "zip_mb": round(mb, 2),
           "files": sorted(p.name for p in out_dir.iterdir()),
           # 2026-08-26: the report recorded INTENT (which files were passed) but no
           # EXECUTION witness -- the same hole H7 closed on the T1 builder.  These are read
           # off the PACKAGED module after the 378-step verification ran, so a bare `except`
           # inside the template cannot fake them.  Recorded, never asserted: a fresh guard's
           # likeliest failure is a false refusal on the happy path.
           "witness": {
               "strpar_live": bool((getattr(mod, "_STRPAR_LIVE", [False]) or [False])[0]),
               "strpar_mode": (getattr(mod, "_STRPAR_MODE", [""]) or [""])[0],
               "strpar_refused": bool((getattr(mod, "_STRPAR_REFUSED", [False]) or [False])[0]),
               "sbport_live": bool((getattr(mod, "_SBPORT_LIVE", [False]) or [False])[0]),
               "blend_live": bool((getattr(mod, "_BLEND_LIVE", [False]) or [False])[0]),
               "bcdt2_live": bool((getattr(mod, "_BCDT2_LIVE", [False]) or [False])[0]),
               "graph_checked": bool(getattr(model, "_graph_checked", False)),
               "graph_on": getattr(model, "_graph", None) is not None,
               # T2IMG (2026-08-28): did the packaged module's MEASURED torch allowlist pass,
               # and on what version?  Templates older than v51 have no such global, so both
               # entries default -- the same builder still has to build the v50 null.
               "strpar_torch_ok": bool((getattr(mod, "_STRPAR_TVOK", [None, ""]) or [None, ""])[0])
                                  if getattr(mod, "_STRPAR_TVOK", None) is not None else None,
               "strpar_torch_version": (getattr(mod, "_STRPAR_TVOK", [None, ""]) or [None, ""])[1],
               # The real capture order, appended at the module's single capture site.  On the
               # happy path this is exactly one entry and it is the multi-stream one.
               "capture_log": [list(c) for c in _cap_log],
               "capture_multistream_is_last": (bool(_cap_log) and _cap_log[-1][0] == "multistream")
                                              if _cap_log else None,
               # T2BIASPAR (2026-09-02): the freeze-trap reads and the branch-taken flag, off the
               # CONSTRUCTED model / module.  Recorded, never asserted; templates without the
               # attributes default to None so every older card still builds.  `biaspar_live`
               # is set during `_capture`'s gate DRIVE, so read it TOGETHER with `strpar_mode`.
               "lim_bias_set": (getattr(model, "_lim_bias", None) is not None) if hasattr(model, "_lim_bias") else None,
               "lim_sp_set": (getattr(model, "_lim_sp", None) is not None) if hasattr(model, "_lim_sp") else None,
               "sp_m_allocated": (getattr(model, "_sp_m", None) is not None) if hasattr(model, "_sp_m") else None,
               "biaspar_live": (list(getattr(mod, "_BIASPAR_LIVE")) if getattr(mod, "_BIASPAR_LIVE", None) is not None else None),
               # BIASGATE fused (v54): the fused-branch flag; None on every template without it.
               "biasgate_live": (list(getattr(mod, "_BIASGATE_LIVE")) if getattr(mod, "_BIASGATE_LIVE", None) is not None else None),
               # BIASGATE (2026-09-02): gate knob + the two second-moment buffers, off the CONSTRUCTED model.
               "gate_c": (float(model.o.get("gate_c", 0.0)) if hasattr(model, "o") else None),
               "gate_mode": (str(model.o.get("gate_mode", "")) if hasattr(model, "o") else None),
               "bias_m2_allocated": (getattr(model, "_bias_m2", None) is not None) if hasattr(model, "_bias_m2") else None,
               "sp_m2_allocated": (getattr(model, "_sp_m2", None) is not None) if hasattr(model, "_sp_m2") else None,
           },
           "coverage": round(float(cov), 4), "mean_step_s": round(float(np.mean(times)), 4),
           "subscores": {k: round(v, 3) for k, v in sub.items()}, "FINAL_LOCAL": round(final, 3)}
    (out_dir / "verify_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    print(f"BUILD_OK {zip_path} {mb:.2f} MB  FINAL_LOCAL={final:.3f}")


if __name__ == "__main__":
    main()
