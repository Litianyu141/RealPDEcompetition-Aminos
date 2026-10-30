#!/usr/bin/env python3
"""Final-kit interface smoke on reference assets; never a training acceptance test."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def load(path: Path):
    spec = importlib.util.spec_from_file_location("decision_solution_smoke", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def unpack(result):
    def array(x):
        if x is None:
            return None
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.asarray(x)
    if isinstance(result, tuple):
        pred, info = result
        info = info if isinstance(info, dict) else {}
        low, high = info.get("lower"), info.get("upper")
    elif isinstance(result, dict):
        pred, low, high = result["prediction"], result.get("lower"), result.get("upper")
    else:
        pred, low, high = result, None, None
    return array(pred), array(low), array(high)


def check(arr, shape):
    assert arr.shape == shape and arr.dtype == np.float32 and np.isfinite(arr).all(), (arr.shape, arr.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=("T1", "T2"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    folder = (ROOT / a.output_dir).resolve()
    out = (ROOT / a.out).resolve()
    assert out.is_relative_to(folder.parent) and not out.exists()
    variant = ROOT / ("Track1" if a.track == "T1" else "Track2") / "U-net/VariantDecisionRepro"
    sol = load(variant / "solution.py")
    rng = np.random.default_rng(20260928)
    if a.track == "T1":
        shape = (4, 20, 32, 64, 3)
        x = rng.normal(0, 0.01, shape).astype(np.float32)
        pred, lo, hi = unpack(sol.load_predictor(str(folder), "cuda").predict(x))
        check(pred, shape)
        assert lo is not None and hi is not None
        check(lo, shape); check(hi, shape)
        assert (lo <= hi).all()
        doc = {"status": "T1_FINAL_INTERFACE_OK", "shape": shape,
               "pred_mean": float(pred.mean()), "bounds_ordered": True}
    else:
        shape = (1, 20, 32, 64, 3)
        x = rng.normal(0, 0.1, shape).astype(np.float32)
        y = rng.normal(0, 0.1, shape).astype(np.float32)
        model = sol.get_ttt_model(str(folder), "cuda")
        model.reset_ttt_state()
        p0, l0, h0 = unpack(model.ttt_step(x, None))
        check(p0, shape)
        assert l0 is not None and h0 is not None
        check(l0, shape); check(h0, shape)
        assert (l0 <= h0).all()
        p1, _, _ = unpack(model.ttt_step(x, y))
        check(p1, shape)
        model.reset_ttt_state()
        p2, _, _ = unpack(model.ttt_step(x, None))
        check(p2, shape)
        reset_equal = bool(np.array_equal(p0, p2))
        assert reset_equal, "trajectory reset did not restore first-step output"
        doc = {"status": "T2_FINAL_INTERFACE_OK", "shape": shape,
               "pred_mean": float(p0.mean()), "reset_bitwise": reset_equal,
               "causal_prev_target_call": True, "bounds_ordered": True}
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(doc["status"], json.dumps(doc))


if __name__ == "__main__":
    main()
