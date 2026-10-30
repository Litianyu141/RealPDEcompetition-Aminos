#!/usr/bin/env python3
"""Build AND offline-verify a Track-1 (Sim2Real) submission bundle in one shot.

Two modes, sharing the same materialise-then-import path so that a swept number and a shipped
number can never diverge:

  --sweep A.pth,B.pth ...   score each checkpoint, print a table, write no zip
  --checkpoint X --name Y   materialise, verify, zip

Verification always re-imports the PACKAGED code (not the source tree) and drives it through the
official Track-1 scoring.py on the held-out val split. Refuses to leave a zip behind if verification
fails, so a broken bundle can never reach the platform.

The val windows come from the Track-2 kit's build_stream, deliberately: it is the same 378-window
local benchmark both tracks have been scored on all along, so Track-1 numbers stay comparable with
Track-2 numbers and with the local->platform bias we have already calibrated.
"""
import os, sys, json, time, shutil, zipfile, argparse, importlib.util
from pathlib import Path
import numpy as np

# Derived from this file's own location (2026-08-12): the absolute hardcode silently
# escaped a fresh clone back to the production tree.  NB: run these tools from their
# in-repo location (the chains do); a /tmp copy would derive the wrong root.
REPO = Path(__file__).resolve().parents[2]
# 2026-08-05 PHASE RESET: kit v9 is mandatory (linear sps, FOV/body cells excluded).
T1_KIT = REPO / "Track1/realpde_t1_starting_kit_v9"
T2_KIT = REPO / "Track2/realpde_t2_starting_kit_v4"
T1_UNET = REPO / "Track1/U-net"
STATS = T2_KIT / "example_data/mean_std_real.pt"
DATA = REPO / "data/RealPDE_competition_dataset/gdrive_raw/local_bench/val"
MAX_MB = 200.0   # platform's declared ceiling is 256 MB (per the ingestion log); keep headroom


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
            # Preserve legacy raw/unrecognized checkpoint handling. Packaged strict loading
            # remains the final check; a known new architecture never skips its source guard.
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
        shutil.copy(T1_KIT / "load_baseline.py", out_dir / "load_baseline.py")
        shutil.copytree(T1_KIT / "rpde_baselines", out_dir / "rpde_baselines",
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


def materialise(out_dir: Path, ckpt: Path, calib, template: Path, halfwidth: float, chunk: int,
                head: str = "", *, extra=None, sband: str = ""):
    """Write a complete bundle directory. Returns the payload whitelist."""
    extra = list(extra or [])
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    tpl = template.read_text()
    for k, v in [("__HALFWIDTH__", repr(halfwidth)), ("__CHUNK__", repr(int(chunk)))]:
        assert k in tpl, f"template placeholder {k} missing"
        tpl = tpl.replace(k, v)
    (out_dir / "submission.py").write_text(tpl)
    shutil.copy(T1_UNET / "unet_model.py", out_dir / "unet_model.py")
    shutil.copy(ckpt, out_dir / "model.pth")
    shutil.copy(STATS, out_dir / "mean_std_real.pt")
    # Resolve every member, including families present only in model_1.pth or later.
    vendor_files = vendor_model_dependencies(out_dir, [ckpt] + extra)
    payload = ["submission.py", "unet_model.py", "model.pth", "mean_std_real.pt"] + vendor_files
    if calib:
        shutil.copy(calib, out_dir / "calib.npz")
        payload.append("calib.npz")
    for _j, _extra in enumerate(extra or [], start=1):
        shutil.copy(_extra, out_dir / f"model_{_j}.pth")
        payload.append(f"model_{_j}.pth")
    if head:
        # the learned variance-deficit field: its weights plus the module that defines the class.
        # submission.py falls back to the static gamma if either is missing, so shipping them is
        # what turns the head on -- there is no flag to get out of sync with.
        shutil.copy(head, out_dir / "vgain.pth")
        shutil.copy(T1_UNET / "vgain_head.py", out_dir / "vgain_head.py")
        payload += ["vgain.pth", "vgain_head.py"]
    if sband:
        # the learned residual-scale band head: weights + the module that defines the class.
        # submission_t1_v36 falls back to the v33 hand-designed band if either is missing, so
        # shipping them is what turns it on -- there is no flag that can get out of sync.
        shutil.copy(sband, out_dir / "sband.pth")
        shutil.copy(T1_UNET / "sband_head.py", out_dir / "sband_head.py")
        payload += ["sband.pth", "sband_head.py"]
    return payload


def load_val(data_dir=None):
    """(X, Y) physical float32, (N, 20, 32, 64, 3) each -- the exact 378-window local benchmark."""
    sys.path.insert(0, str(T2_KIT))
    from local_eval import build_stream
    X, Y = [], []
    for step in build_stream(Path(data_dir) if data_dir else DATA):
        X.append(np.asarray(step["input"], dtype=np.float32))
        Y.append(np.asarray(step["target"], dtype=np.float32))
    return np.stack(X, 0), np.stack(Y, 0)


def run_and_score(sub_dir: Path, X, Y, official, use_bounds: bool):
    """Import the PACKAGED submission, run predict once, score it. Returns (subscores, extras)."""
    for stale in list(sys.modules):
        if stale in ("participant_t1", "unet_model", "no82_model", "lsm82_model", "sinenet82_model", "uno82_model", "dpot82_model", "cno2d82_model", "galerkin82_model", "dcno82_model", "transolver2d82_model", "sband_head", "vgain_head"):
            del sys.modules[stale]
    sys.path.insert(0, str(sub_dir))
    try:
        mod = _load_module(sub_dir / "submission.py", "participant_t1")
        # ---- BUILD-TIME LOUDNESS -------------------------------------------------------------
        # Every band tier in submission.py degrades through a bare `except`.  That is correct on
        # the platform and disastrous as a measurement protocol: the v26 arm shipped a mis-shaped
        # table, its `except` wiped BOTH the table and _KCELL, and the degraded bundle scored
        # 47.47 against a remembered 47.82 -- close enough to pass a naive self-test.  These two
        # asserts make that outcome unbuildable.  `sband_required` is an INTENT MARKER written
        # into calib.npz by the packer; the runtime only records it.
        _req = bool(getattr(mod, "_SBAND_REQ", False))
        _st = str(getattr(mod, "_SBAND_STATUS", "absent"))
        if _req:
            assert _st == "ok" and getattr(mod, "_SHEAD_LIVE", False) is True, \
                f"calib declares sband_required but the packaged module reports {_st!r}"
        if getattr(mod, "_BAND_LO", None) is not None:
            assert getattr(mod, "_KCELL", None) is not None, \
                "the v33 heteroscedastic fallback was wiped at load -- v26 mode"
        t0 = time.perf_counter()
        out = mod.predict(X)                      # cold call; the bundle warms up at import
        elapsed = time.perf_counter() - t0
    finally:
        sys.path.remove(str(sub_dir))

    if isinstance(out, dict):
        P = np.asarray(out["prediction"], np.float32)
        L = np.asarray(out["lower"], np.float32) if out.get("lower") is not None else None
        U = np.asarray(out["upper"], np.float32) if out.get("upper") is not None else None
    else:
        P, L, U = np.asarray(out, np.float32), None, None
    if not use_bounds:
        L = U = None                              # reproduce the scorer's 0.05*|pred| fallback

    assert np.isfinite(P).all(), "non-finite prediction -> score 0"
    assert P.shape == Y.shape, f"shape {P.shape} != {Y.shape} -> score 0"
    if L is not None:
        assert np.isfinite(L).all() and np.isfinite(U).all(), "non-finite bounds -> hard error"
        assert (L <= U).all(), "lower > upper -> hard error"

    c = official.measured_channels(Y)
    mean_t = elapsed / X.shape[0]
    sub = {
        "rel_l2_score": official.score_error(float(np.mean(official.rel_l2_per_sample(P, Y, c)))),
        "tke_score": official.score_error(float(np.mean(official.tke_rel_l2_per_sample(P, Y, c)))),
        "mvpe_score": official.score_error(official.mvpe_rel_l2(P, Y)),
        "time_score": official.score_time(mean_t),
    }
    sps, cov = official.aggregate_sps(P, Y, c, lower=L, upper=U)
    sub["sps_score"] = float(official.score_sps(sps))
    return sub, {"c": c, "coverage": float(cov), "mean_t_neural_s": mean_t,
                 "total_s": elapsed, "n": int(X.shape[0]),
                 "sband_status": str(getattr(mod, "_SBAND_STATUS", "absent")),
                 "sband_required": bool(getattr(mod, "_SBAND_REQ", False)),
                 "shead_live": bool(getattr(mod, "_SHEAD_LIVE", False)),
                 "kcell_live": getattr(mod, "_KCELL", None) is not None,
                 "vgain_live": getattr(mod, "_HEAD", None) is not None,
                 # EXECUTION WITNESS (2026-08-26 audit): the report recorded only INTENT
                 # (extra_checkpoints from argv) and PRESENCE (the zip namelist).  Neither
                 # proves the runtime LOADED them.  _NMEM is what the module actually built.
                 "n_members": int(getattr(mod, "_NMEM", 1)),
                 # v73 STRPAR witness, read off the module the scorer just ran.  A bare `except`
                 # in the template cannot fake these: they default to refused and are set only
                 # after a torch.equal gate has passed on real tensors.
                 "strpar": {
                     "strpar_live": bool(getattr(mod, "_STRPAR_LIVE", False)),
                     "strpar_mode": str(getattr(mod, "_STRPAR_MODE", "")),
                     "refused": bool(getattr(mod, "_STRPAR_REFUSED", False)),
                     "why": str(getattr(mod, "_STRPAR_WHY", "")),
                     "graph_checked": bool(getattr(mod, "_GRAPHED_FULL", False)),
                     "gates": {str(k): v for k, v in
                               dict(getattr(mod, "_STRPAR_GATES", {})).items()},
                 }}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="", help="comma-separated checkpoints to score (no zip)")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--calib", default="")
    ap.add_argument("--allow-calib-mismatch", action="store_true",
        help="package a calib whose provenance stamp names a different forecaster (default: refuse)")
    ap.add_argument("--name", default="")
    ap.add_argument("--template", default=str(T1_UNET / "submission_t1_v2.py"))
    ap.add_argument("--halfwidth", type=float, default=0.15)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--head", default="", help="vgain head .pth to ship (turns the head on)")
    ap.add_argument("--sband", default="", help="learned residual-scale band head .pth to ship")
    ap.add_argument("--extra-checkpoints", default="",
                    help="comma-separated additional ensemble members "
                         "-> packed as model_1.pth, model_2.pth, ...")
    ap.add_argument("--val-dir", default="", help="score on a DIFFERENT stream (regime-shift tests). Default = the 378-window interpolation benchmark, bit-identical to before.")
    ap.add_argument("--gpu", default=None)
    a = ap.parse_args()
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    official = _load_module(T1_KIT / "scoring.py", "t1_scoring")   # explicit path: never T2's copy
    X, Y = load_val(a.val_dir or None)
    print(f"# local benchmark: X{X.shape} Y{Y.shape}  measured_channels={official.measured_channels(Y)}")

    if a.sweep:
        rows = []
        for ck in [Path(s.strip()) for s in a.sweep.split(",") if s.strip()]:
            tmp = T1_UNET / "submissions" / f"_sweep_{ck.stem}"
            # config A: no calibration, bounds ignored -> exactly what the v1 baseline scored
            materialise(tmp, ck, None, Path(a.template), a.halfwidth, a.chunk, a.head,
                        sband=a.sband)
            sA, xA = run_and_score(tmp, X, Y, official, use_bounds=False)
            rows.append((ck.stem, "A raw + scorer-default band", sA, xA))
            if a.calib:
                materialise(tmp, ck, a.calib, Path(a.template), a.halfwidth, a.chunk, a.head,
                            sband=a.sband)
                sB, xB = run_and_score(tmp, X, Y, official, use_bounds=True)
                rows.append((ck.stem, "B gamma + calibrated band", sB, xB))
            shutil.rmtree(tmp, ignore_errors=True)

        hdr = f"{'checkpoint':<16}{'config':<30}{'rel_l2':>8}{'tke':>8}{'mvpe':>8}{'time':>8}{'sps':>8}{'FINAL':>9}{'cov':>7}{'s/sample':>10}"
        print("\n" + hdr + "\n" + "-" * len(hdr))
        for nm, cfg, s, x in rows:
            f = float(np.mean(list(s.values())))
            print(f"{nm:<16}{cfg:<30}{s['rel_l2_score']:>8.2f}{s['tke_score']:>8.2f}"
                  f"{s['mvpe_score']:>8.2f}{s['time_score']:>8.2f}{s['sps_score']:>8.2f}"
                  f"{f:>9.3f}{x['coverage']:>7.3f}{x['mean_t_neural_s']:>10.5f}")
        print("\nSWEEP_DONE")
        return

    assert a.checkpoint and a.name, "need --checkpoint and --name (or --sweep)"
    out_dir = T1_UNET / "submissions" / a.name
    _extra = [Path(s.strip()) for s in a.extra_checkpoints.split(',') if s.strip()]
    if _extra:
        # SILENT-DROP GUARD (2026-08-26 audit).  Only 4 of 31 templates contain the model_%d.pth
        # loop and --template DEFAULTS to submission_t1_v2.py, which does not.  Without this check
        # a blend build copies the members into the zip, passes the payload/namelist asserts, adds
        # ~30 MB of dead weight, and ships a ONE-member card whose report claims the extras.
        _tpl_src = Path(a.template).read_text()
        assert "model_%d.pth" in _tpl_src, (
            f"--extra-checkpoints given ({len(_extra)}) but template {Path(a.template).name} has "
            f"no model_%d.pth loader -- it would ship a ONE-member card. Use a blend template.")
        for _e in _extra:
            assert _e.exists(), f"extra checkpoint missing: {_e}"
            assert _e.resolve() != Path(a.checkpoint).resolve(), \
                f"extra checkpoint duplicates --checkpoint: {_e}"
        assert len({_e.resolve() for _e in _extra}) == len(_extra), "duplicate --extra-checkpoints"
    # provenance is checked HERE, not at parse time, so the blend members are in scope
    _prov_check(getattr(a, "calib", None), a.checkpoint,
                getattr(a, "allow_calib_mismatch", False), extra=_extra)
    payload = materialise(out_dir, Path(a.checkpoint), a.calib, Path(a.template), a.halfwidth,
                          a.chunk, a.head, extra=_extra, sband=a.sband)
    if a.calib:
        with np.load(out_dir / "calib.npz") as z:
            print("# shipping calib.npz: " + ", ".join(f"{k}{z[k].shape}" for k in z.files))
    sub, extra = run_and_score(out_dir, X, Y, official, use_bounds=True)
    final = float(np.mean(list(sub.values())))

    # ---- zip only after verification passed ----
    zip_path = T1_UNET / "submissions" / f"{a.name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    # Whitelist, never iterdir(): verification above imports the packaged code, which leaves a
    # __pycache__/ behind. Zipping the dir wholesale would ship a stray directory entry.
    # RELAYOUT 2026-09-07: DETERMINISTIC zip -- sorted, ZIP_STORED, fixed timestamps -- so the
    # same card directory gives the same bytes/md5 on every box; the upload zip is made on the Mac by
    # tools/make_submission_zip.py from the synced directory, and a server regenerates it instead of keeping copies.
    sys.path.insert(0, str(REPO / "common"))
    from make_submission_zip import make_zip
    make_zip(out_dir, zip_path, names=list(payload))
    with zipfile.ZipFile(zip_path) as z:
        assert sorted(z.namelist()) == sorted(payload), f"unexpected archive contents: {z.namelist()}"
        assert z.testzip() is None, "archive failed CRC self-test"
    mb = zip_path.stat().st_size / 1e6
    assert mb <= MAX_MB, f"bundle {mb:.1f} MB exceeds the {MAX_MB} MB safety ceiling"

    rep = {"track": 1, "name": a.name, "checkpoint": a.checkpoint, "calib": a.calib,
           "head": a.head, "sband": a.sband, "template": a.template,
           # 2026-08-19: the live card is a 2-member cross-recipe blend and the report did
           # not record the second member, so a chain re-anchored from this file would have
           # rebuilt a ONE-member bundle.  Record it.
           "extra_checkpoints": [str(x) for x in _extra],
           "sband_status": extra["sband_status"], "shead_live": extra["shead_live"],
           "kcell_live": extra["kcell_live"], "vgain_live": extra["vgain_live"],
           "halfwidth": a.halfwidth, "chunk": a.chunk, "zip": str(zip_path), "zip_mb": round(mb, 2),
           "files": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
           "measured_channels": extra["c"], "coverage": round(extra["coverage"], 4),
           "n_windows": extra["n"], "mean_t_neural_s": round(extra["mean_t_neural_s"], 6),
           # 2026-08-27: `run_and_score` has computed this execution witness since the 08-26 audit
           # but `main()` never copied it into the report, so the fix was only half-landed and the
           # witness was discarded at the last step.  `n_members` is the module's OWN _NMEM, i.e.
           # what the runtime actually built -- not what argv asked for and not what the zip holds.
           "n_members": extra.get("n_members"),
           # STRPAR (v73) execution witness, read off the PACKAGED module.  Absent/False on every
           # template that does not implement the arm, which is the correct reading for those.
           "strpar": extra.get("strpar"),
           "subscores": {k: round(v, 3) for k, v in sub.items()}, "FINAL_LOCAL": round(final, 3)}
    (out_dir / "verify_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    print(f"BUILD_OK {zip_path} {mb:.2f} MB  FINAL_LOCAL={final:.3f}")




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


def _prov_stamp(d, ckpt=None, fit_stream="", producer="", extra=None):
    """Add provenance keys to a calib dict in place and return it.

    `extra` (2026-08-26 audit): the BLEND members.  Before this the stamp bound model.pth ONLY, so
    on a blend card -- which is what T1 ships -- model_1.pth could be swapped, reordered or replaced
    with anything and the gate still printed '[prov] ok'.  Half the shipped prediction was unguarded.
    """
    import time as _t
    import os as _o
    import numpy as _np
    d["prov_ckpt_md5"] = _np.str_(_prov_md5(ckpt) if ckpt else "")
    d["prov_ckpt_name"] = _np.str_(_o.path.basename(str(ckpt)) if ckpt else "")
    _ex = [str(x) for x in (extra or [])]
    d["prov_extra_md5"] = _np.str_(",".join(_prov_md5(x) or "" for x in _ex))
    d["prov_extra_name"] = _np.str_(",".join(_o.path.basename(x) for x in _ex))
    d["prov_fit_stream"] = _np.str_(str(fit_stream))
    d["prov_producer"] = _np.str_(str(producer))
    d["prov_when"] = _np.str_(_t.strftime("%Y-%m-%dT%H:%M:%S"))
    return d


def _prov_check(calib_path, ckpt_path, allow=False, extra=None):
    """Refuse a calib/checkpoint mismatch. Unstamped calibs warn only (every historical artifact)."""
    import numpy as _np
    if not calib_path:
        return
    try:
        with _np.load(calib_path, allow_pickle=True) as z:
            stamped = str(z["prov_ckpt_md5"]) if "prov_ckpt_md5" in z.files else ""
            name = str(z["prov_ckpt_name"]) if "prov_ckpt_name" in z.files else ""
            ex_st = str(z["prov_extra_md5"]) if "prov_extra_md5" in z.files else None
            ex_nm = str(z["prov_extra_name"]) if "prov_extra_name" in z.files else ""
    except Exception as e:
        print(f"[prov] WARNING: could not read provenance from {calib_path}: {e}")
        return
    if not stamped:
        print(f"[prov] WARNING: {calib_path} carries no provenance stamp (pre-2026-08-13 artifact); "
              f"cannot verify it was fitted on this forecaster's residuals")
        return
    # BLEND MEMBERS (2026-08-26 audit).  Absent key == a pre-2026-08-26 stamp: warn, never refuse,
    # so every historical artifact still builds.  Present key == the band knows its member set and a
    # mismatch is the same class of error as a mismatched primary.
    _ex = [str(x) for x in (extra or [])]
    # An EMPTY stamp is "this band was packed by a producer that does not yet pass its members"
    # (t1_sband_pack.py today), NOT "this band was fitted on zero members".  Treating the two the
    # same would REFUSE every normal 2-member build against a freshly packed band -- caught before
    # it shipped, but it is exactly the regression shape a fail-loud guard invites.
    if ex_st is None or ex_st == "":
        if _ex:
            print(f"[prov] WARNING: {calib_path} predates member stamping -- the {len(_ex)} blend "
                  f"member(s) are NOT provenance-checked (the primary still is)")
    else:
        want = [t for t in ex_st.split(",") if t]
        got = [_prov_md5(x) or "" for x in _ex]
        if want != got:
            msg = (f"[prov] BLEND MEMBER MISMATCH\n"
                   f"        calib   : {calib_path}\n"
                   f"        fitted on members : {ex_nm or '?'}\n"
                   f"        md5 expected      : {want}\n"
                   f"        md5 packaging     : {got}\n"
                   f"        The band was fitted on a DIFFERENT blend. Refit it (BCP), or pass "
                   f"--allow-calib-mismatch to override deliberately.")
            if allow:
                print(msg + "\n[prov] --allow-calib-mismatch: continuing anyway")
            else:
                raise SystemExit(msg)
        elif _ex:
            print(f"[prov] ok: {len(_ex)} blend member(s) match the band's stamp")
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


if __name__ == "__main__":
    main()
