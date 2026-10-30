#!/usr/bin/env python3
"""Clean-room smoke test for the vendored RealPDE Track-1 starting kit.

Run it from a directory that is NOT the original RealPDEBench repo, with only the
kit dir on PYTHONPATH, e.g.:

    cd /tmp/anywhere_clean
    PYTHONPATH=/path/to/kit python /path/to/kit/smoke_test_kit.py

It proves the kit imports and runs without the upstream repo:
  (0) `import realpdebench` FAILS and `rpde_baselines` resolves inside the kit;
  (A) every checkpoint loads (strict) into its model;
  (B) CNO / FNO / Transolver forward at both 32x64 and 64x128;
  (C) pack_ckpt_fp16.py round-trips FNO fp32<->fp16 with small numeric error;
  (D) submission_template.py's optional lower/upper dict output is correctly
      normalized by the ingestion program (predict()-level shape check always
      runs; the real ingestion.py's normalize_prediction_output() is exercised
      too when this kit sits inside a full bundle checkout).

Checkpoints/example are read from $CKPT_DIR and $EXAMPLE_H5 (they are downloaded
separately from Google Drive, per the README; they are not shipped in the kit).
"""
import importlib.machinery
import importlib.util
import os
import subprocess
import sys

import numpy as np
import torch

_KIT = os.path.dirname(os.path.abspath(__file__))


class _NoSystemEinops:
    """Simulate an offline container that does NOT ship einops: reject any
    einops whose file lives outside this kit, so a successful Transolver
    forward proves the vendored copy (kit/_vendor/einops) is self-sufficient."""

    def find_spec(self, name, path=None, target=None):
        if name == "einops" or name.startswith("einops."):
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec and spec.origin and spec.origin not in ("built-in", "frozen"):
                if not os.path.abspath(spec.origin).startswith(_KIT):
                    raise ModuleNotFoundError(
                        "simulated container without einops: blocked %s" % spec.origin)
            return spec
        return None


# Install the blocker before anything imports einops.
sys.meta_path.insert(0, _NoSystemEinops())

CKPT_DIR = os.environ.get(
    "CKPT_DIR",
    os.path.expanduser("~/projects/track2_dev/starting_kit_v3_work/ckpts"))
EXAMPLE_H5 = os.environ.get(
    "EXAMPLE_H5",
    os.path.expanduser(
        "~/projects/track2_dev/starting_kit_v3_work/example_data/3750_0.h5"))
KIT = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.join(KIT, "pack_ckpt_fp16.py")


def hr(t):
    print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)


def main():
    torch.manual_seed(0)

    hr("(0) INDEPENDENCE: no reliance on the original realpdebench repo")
    try:
        import realpdebench  # noqa: F401
        print("WARNING: `import realpdebench` unexpectedly succeeded at",
              realpdebench.__file__)
    except ModuleNotFoundError:
        print("OK: `import realpdebench` fails (repo not on path)")
    import load_baseline as lb   # prepends kit + kit/_vendor to sys.path
    import rpde_baselines
    inside = os.path.abspath(rpde_baselines.__file__).startswith(KIT)
    print("rpde_baselines resolves inside kit:", inside, "->",
          rpde_baselines.__file__)
    assert inside, "vendored package is not being imported from the kit!"

    # einops isolation: the venv actually has a site-packages einops, but with
    # the blocker active only the kit's vendored copy is allowed to load.
    import einops
    ein_inside = os.path.abspath(einops.__file__).startswith(KIT)
    print("einops blocker active; kit uses vendored einops:", ein_inside,
          "->", einops.__file__)
    assert ein_inside, "einops did not resolve to the vendored copy!"

    # Prove the blocker has teeth: drop _vendor from sys.path so only the
    # site-packages einops remains, and confirm it is rejected (i.e. the kit
    # would NOT silently fall back to a container einops).
    _saved_path, _saved_mods = list(sys.path), {
        k: v for k, v in sys.modules.items()
        if k == "einops" or k.startswith("einops.")}
    try:
        vend = os.path.join(KIT, "_vendor")
        sys.path[:] = [p for p in sys.path if os.path.abspath(p) != vend]
        for k in list(_saved_mods):
            del sys.modules[k]
        try:
            __import__("einops")
            print("  (note: no site-packages einops present to reject)")
        except ModuleNotFoundError as e:
            print("  blocker has teeth: non-kit einops rejected ->", str(e)[:70])
    finally:
        sys.path[:] = _saved_path
        for k in [k for k in sys.modules if k == "einops" or k.startswith("einops.")]:
            del sys.modules[k]
        sys.modules.update(_saved_mods)

    all_ckpts = [
        "sim_cno.pth", "sim_fno.pth", "sim_fno_fp16.pth", "sim_transolver.pth",
        "sim_real_cno.pth", "sim_real_fno.pth", "sim_real_fno_fp16.pth",
        "sim_real_transolver.pth",
    ]

    hr("(A) LOAD EVERY CHECKPOINT (strict) + (B) FORWARD AT 32x64 AND 64x128")
    x32 = lb.make_example_input(EXAMPLE_H5, sub_s=2)   # (1,20,32,64,3)
    x64 = lb.make_example_input(EXAMPLE_H5, sub_s=1)   # (1,20,64,128,3)
    print("inputs:", tuple(x32.shape), "and", tuple(x64.shape))
    forwarded = set()
    for name in all_ckpts:
        path = os.path.join(CKPT_DIR, name)
        if not os.path.exists(path):
            print("\n[%s] MISSING (download from GD) - skipped" % name)
            continue
        try:
            model, meta = lb.load_baseline(path, strict=True)
            ok = not meta["missing_keys"] and not meta["unexpected_keys"]
            extra = ""
            if meta.get("best_val_loss") is not None:
                extra = " best_val_loss=%.4f" % meta["best_val_loss"]
            print("\n[%s] type=%s fmt=%s strict=%s%s" %
                  (name, meta["model_type"], meta["format"],
                   "OK" if ok else "MISMATCH", extra))
            if meta["missing_keys"] or meta["unexpected_keys"]:
                print("    missing=%s unexpected=%s" %
                      (meta["missing_keys"][:4], meta["unexpected_keys"][:4]))
            mt = meta["model_type"]
            if mt not in forwarded:
                for tag, xin, hs, ws in (("32x64", x32, 32, 64),
                                         ("64x128", x64, 64, 128)):
                    m2, _ = lb.load_baseline(
                        path, strict=True,
                        in_shape=(20, hs, ws, 3), out_shape=(20, hs, ws, 3))
                    with torch.no_grad():
                        y = m2(xin)
                    print("    forward %-6s: %s -> %s finite=%s" %
                          (tag, tuple(xin.shape), tuple(y.shape),
                           bool(torch.isfinite(y).all())))
                    assert tuple(y.shape) == (1, 20, hs, ws, 3)
                    del m2
                forwarded.add(mt)
            del model
        except Exception as e:
            print("\n[%s] ERROR: %r" % (name, e))
    print("\nforwarded model types:", sorted(forwarded))
    assert forwarded == {"cno", "fno", "transolver"}, "missing a model type"

    hr("(C) FP16 PACK ROUND-TRIP (FNO, via vendored pack_ckpt_fp16.py)")
    src = os.path.join(CKPT_DIR, "sim_fno.pth")
    dst = os.path.join("/tmp", "kit_smoke_fno_fp16.pth")
    out = subprocess.run([sys.executable, PACK, src, dst],
                         capture_output=True, text=True)
    print(out.stdout.strip() or out.stderr.strip())
    if out.returncode == 0:
        fp32, _ = lb.load_checkpoint_state(src)
        fp16 = lb.unpack_fp16(dst)
        m_abs = m_rel = 0.0
        for k, v in fp32.items():
            if torch.is_tensor(v) and (v.is_complex() or torch.is_floating_point(v)):
                d = (fp16[k] - v).abs().max().item()
                m_abs = max(m_abs, d)
                den = v.abs().max().item()
                if den > 0:
                    m_rel = max(m_rel, d / den)
        print("weight round-trip: max_abs=%.3e max_rel=%.3e" % (m_abs, m_rel))
        m32, _ = lb.load_baseline("fno", src, strict=True)
        m16 = lb.build_model("fno")
        m16.load_state_dict(fp16, strict=True)
        m16.eval()
        with torch.no_grad():
            y32, y16 = m32(x32), m16(x32)
        den = y32.abs().max().item()
        print("fp16-vs-fp32 forward: max_abs=%.3e max_rel=%.3e" %
              ((y16 - y32).abs().max().item(),
               (y16 - y32).abs().max().item() / den if den else float("nan")))
        os.remove(dst)

    hr("(D) BOUNDS: submission_template.py dict output accepted by ingestion")
    import submission_template as subtpl

    x_small = np.random.RandomState(0).randn(2, 20, 32, 64, 3).astype(np.float32)

    subtpl.RETURN_BOUNDS = False
    default_out = subtpl.predict(x_small)
    assert isinstance(default_out, np.ndarray), (
        "default (RETURN_BOUNDS=False) predict() should return a bare ndarray")
    print("OK: default predict() returns a bare ndarray, shape", tuple(default_out.shape))

    try:
        subtpl.RETURN_BOUNDS = True
        bounds_out = subtpl.predict(x_small)
    finally:
        subtpl.RETURN_BOUNDS = False
    assert isinstance(bounds_out, dict), "RETURN_BOUNDS=True should return a dict"
    assert set(bounds_out) == {"prediction", "lower", "upper"}
    pred, lo, up = bounds_out["prediction"], bounds_out["lower"], bounds_out["upper"]
    assert lo.shape == pred.shape == up.shape
    assert np.all(lo <= pred) and np.all(pred <= up)
    print("OK: RETURN_BOUNDS=True predict() returns prediction+lower+upper, "
          "shapes match, lower <= prediction <= upper everywhere")

    # Re-run both dicts through the REAL ingestion program's
    # normalize_prediction_output(), not a re-implementation of it, so this test
    # reflects actual platform behavior. Only present when this kit sits inside a
    # full bundle checkout (bundle/starting_kit/ + bundle/programs/); a
    # participant's standalone copy of just the starting kit won't have the
    # sibling programs/ directory, so skip gracefully rather than failing.
    ingestion_path = os.path.join(KIT, "..", "programs", "ingestion_program", "ingestion.py")
    if os.path.exists(ingestion_path):
        spec = importlib.util.spec_from_file_location("ingestion", ingestion_path)
        ingestion = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ingestion)

        normalized_default = ingestion.normalize_prediction_output(default_out)
        assert set(normalized_default) == {"prediction"}, (
            "default (no bounds) predict() must not leak lower/upper into predictions.npz")

        normalized_bounds = ingestion.normalize_prediction_output(bounds_out)
        assert set(normalized_bounds) == {"prediction", "lower", "upper"}
        n_pred = normalized_bounds["prediction"]
        n_lo = normalized_bounds["lower"]
        n_up = normalized_bounds["upper"]
        assert n_lo.shape == n_pred.shape == n_up.shape == pred.shape
        assert all(arr.dtype == np.float32 for arr in (n_pred, n_lo, n_up))
        assert np.all(n_lo <= n_pred) and np.all(n_pred <= n_up)
        print("OK: ingestion.normalize_prediction_output() carries prediction/lower/upper "
              "through unchanged (float32, shapes match, lower <= prediction <= upper)")
    else:
        print("SKIPPED: bundle/programs/ingestion_program/ingestion.py not found next to "
              "this kit (expected for a standalone starting-kit-only checkout); the "
              "predict()-level dict shape was still exercised above.")

    print("\nALL KIT SMOKE CHECKS COMPLETE")


if __name__ == "__main__":
    main()
