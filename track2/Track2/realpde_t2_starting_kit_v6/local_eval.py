#!/usr/bin/env python3
"""Local CPU smoke test for a RealPDE Track 2 LTTTA submission.

Self-contained: it does NOT need the downloaded ingestion/scoring programs. It
mirrors the streaming loop of the official ingestion program on the tiny
``example_data`` bundled here -- reset at trajectory boundaries, prev-target
passing, and per-step timing -- then prints the mean per-step time and a mock
relative-L2 over the measured (u, v) channels.

This is only a shape/plumbing check. The mock rel-L2 is computed on the two
synthetic example trajectories and is NOT comparable to the leaderboard; the
official ingestion.py / scoring.py are authoritative.

If your submission's ``ttt_step`` also returns ``info["lower"]`` and
``info["upper"]`` (the optional SPS interval bounds), this script forwards
them through the same denormalisation used for the prediction and passes them
into the bundled ``scoring.py``, with the same all-or-nothing and matching-shape
checks, so you can see ``sps_score`` change locally.

Usage:
    python local_eval.py --submission <dir with submission.py>
    python local_eval.py --submission agentic_demo
    python local_eval.py            # defaults --submission to this kit dir

Data layout:
    <data>/test_real/*.h5      (flat u, v[, p] datasets, native 64x128)
    <data>/mean_std_real.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent

# Fixed streaming protocol constants (must match the official pipeline).
IN_STEP = 20
OUT_STEP = 20
INTERVAL = 20
SUB_S = 2  # spatial subsample: native 64x128 -> 32x64
HORIZON = IN_STEP + OUT_STEP


# --------------------------------------------------------------------------- #
# Data pipeline.
# --------------------------------------------------------------------------- #
def build_stream(data_dir: Path):
    """Return an ordered list of steps: dicts with input/target tensors + meta.

    Sorted by (filename, time_id); ``is_first`` marks trajectory boundaries.
    """
    test_dir = data_dir / "test_real"
    files = sorted(f for f in os.listdir(test_dir) if f.endswith(".h5"))
    if not files:
        raise SystemExit(f"No .h5 files under {test_dir}")

    entries = []  # (filename, time_id)
    for name in files:
        with h5py.File(test_dir / name, "r") as f:
            n_frames = f["u"].shape[0]
        max_time_id = n_frames - HORIZON
        if max_time_id < 0:
            continue
        for time_id in range(0, max_time_id + 1, INTERVAL):
            entries.append((name, time_id))
    entries.sort(key=lambda e: (e[0], e[1]))

    stream = []
    prev_name = None
    for name, time_id in entries:
        with h5py.File(test_dir / name, "r") as f:
            n_frames = f["u"].shape[0]
            end = min(time_id + HORIZON, n_frames)
            u = f["u"][time_id:end, ::SUB_S, ::SUB_S]
            v = f["v"][time_id:end, ::SUB_S, ::SUB_S]
            p = np.zeros_like(u)  # real data: p is zero-filled
        data = np.stack([u, v, p], axis=-1)
        if data.shape[0] < HORIZON:  # pad short tails by repeating last frame
            pad = np.repeat(data[-1:], HORIZON - data.shape[0], axis=0)
            data = np.concatenate([data, pad], axis=0)
        inp = torch.tensor(data[:IN_STEP], dtype=torch.float32)
        tgt = torch.tensor(data[IN_STEP:HORIZON], dtype=torch.float32)
        stream.append({
            "input": inp, "target": tgt,
            "sim_id": name, "time_id": time_id,
            "is_first": name != prev_name,
        })
        prev_name = name
    return stream


class Normalizer:
    """Minimal GaussianNormalizer: affine per-channel, std==0 -> 1."""

    def __init__(self, stats_path: Path):
        mi, mt, si, st = torch.load(stats_path, weights_only=False)
        one = torch.ones_like
        self.mean_in, self.mean_tgt = mi.float(), mt.float()
        self.std_in = torch.where(si == 0, one(si), si).float()
        self.std_tgt = torch.where(st == 0, one(st), st).float()

    def preprocess(self, x, y):
        c1, c2 = x.shape[-1], y.shape[-1]
        xn = (x - self.mean_in[..., :c1]) / self.std_in[..., :c1]
        yn = (y - self.mean_tgt[..., :c2]) / self.std_tgt[..., :c2]
        return xn, yn

    def postprocess_pred(self, pred_norm):
        c = pred_norm.shape[-1]
        return pred_norm * self.std_tgt[..., :c] + self.mean_tgt[..., :c]


# --------------------------------------------------------------------------- #
# Submission loading + streaming loop.
# --------------------------------------------------------------------------- #
def import_submission(submission_dir: Path):
    sub_file = submission_dir / "submission.py"
    if not sub_file.exists():
        raise SystemExit(f"Expected submission.py at {sub_file}")
    sys.path.insert(0, str(submission_dir))  # so relative imports resolve
    spec = importlib.util.spec_from_file_location("participant_submission", sub_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_ttt_model"):
        raise SystemExit("submission.py must define get_ttt_model(submission_dir, device).")
    return module


def measured_channels(target: np.ndarray) -> int:
    active = sum(not np.allclose(target[..., i], 0.0) for i in range(target.shape[-1]))
    return max(1, int(active))


def rel_l2(pred: np.ndarray, target: np.ndarray, c: int) -> float:
    p = pred[..., :c].reshape(pred.shape[0], -1)
    t = target[..., :c].reshape(target.shape[0], -1)
    denom = np.linalg.norm(t, axis=1).clip(min=1e-8)
    return float(np.mean(np.linalg.norm(p - t, axis=1) / denom))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", default=str(HERE),
                    help="directory containing submission.py (default: this kit)")
    ap.add_argument("--data", default=str(HERE / "example_data"),
                    help="example_data directory (default: ./example_data)")
    args = ap.parse_args()

    submission_dir = Path(args.submission).resolve()
    data_dir = Path(args.data).resolve()
    device = "cpu"

    stats_path = data_dir / "mean_std_real.pt"
    if not stats_path.exists():
        raise SystemExit(f"Missing {stats_path}. Run example_data/make_example.py first.")

    stream = build_stream(data_dir)
    normalizer = Normalizer(stats_path)
    n_traj = len({s["sim_id"] for s in stream})
    print(f"[local_eval] {len(stream)} steps over {n_traj} trajectories, batch size 1")

    module = import_submission(submission_dir)
    model = module.get_ttt_model(str(submission_dir), device)
    for attr in ("reset_ttt_state", "ttt_step"):
        if not hasattr(model, attr):
            raise SystemExit(f"get_ttt_model() returned an object lacking {attr}().")

    per_step_times, preds, tgts = [], [], []
    lowers, uppers = [], []
    # SPS bounds are optional; None = undecided, else fixed by step 0.
    bounds_active = None
    prev_pair = None  # (inp_norm, tgt_norm) of the previous step
    expected_shape = None

    for step in stream:
        inp = step["input"].unsqueeze(0)     # (1, 20, 32, 64, 3)
        tgt = step["target"].unsqueeze(0)

        # reset_ttt_state is timed and charged to the step that follows it, the
        # same way the platform does it, so a local time_score does not read
        # better than the real one.
        reset_elapsed = 0.0
        if step["is_first"]:
            t_reset = perf_counter()
            model.reset_ttt_state()
            reset_elapsed = perf_counter() - t_reset
            prev_pair = None

        inp_norm, tgt_norm = normalizer.preprocess(inp, tgt)
        prev_target = prev_pair[1] if prev_pair is not None else None

        t0 = perf_counter()
        pred_norm, info = model.ttt_step(inp_norm, prev_target)
        per_step_times.append(perf_counter() - t0 + reset_elapsed)

        prev_pair = (inp_norm.detach(), tgt_norm.detach())  # cache AFTER timing

        pred_norm = torch.as_tensor(pred_norm)
        if expected_shape is None:
            expected_shape = tuple(tgt_norm.shape)
        if tuple(pred_norm.shape) != expected_shape:
            raise SystemExit(
                f"ttt_step returned shape {tuple(pred_norm.shape)}, expected {expected_shape}."
            )
        if not isinstance(info, dict) or "adapt_loss" not in info:
            raise SystemExit('ttt_step must return (pred, info) with info["adapt_loss"].')

        # Optional SPS bounds, same rules as the platform: a lone
        # `lower`/`upper` key is an error, a key set to None is an error,
        # bounds must match pred_norm's shape, and a run supplies them on
        # every step or on none. NaN/Inf and `lower > upper` are left to
        # scoring.py, which zeroes the score.
        has_lower_key = "lower" in info
        has_upper_key = "upper" in info
        if has_lower_key != has_upper_key:
            raise SystemExit(
                "ttt_step() returned only one of info['lower'] / info['upper']. "
                "SPS bounds must be supplied together (both keys set to arrays "
                "shaped like the prediction) or not at all."
            )
        lower_val = info.get("lower")
        upper_val = info.get("upper")
        if has_lower_key and (lower_val is None or upper_val is None):
            raise SystemExit(
                "ttt_step() set info['lower'] / info['upper'] to None. SPS "
                "bounds must be arrays shaped like the prediction, or the "
                "keys must be left out of info entirely -- None is not a "
                "valid value for either key."
            )
        has_bounds = has_lower_key
        if bounds_active is None:
            bounds_active = has_bounds
        elif has_bounds != bounds_active:
            raise SystemExit(
                f"ttt_step {'started' if has_bounds else 'stopped'} returning SPS "
                "lower/upper bounds in info; a submission must supply them on "
                f"every step or on none (step 0 {'did' if bounds_active else 'did not'})."
            )

        pred_raw = normalizer.postprocess_pred(pred_norm.detach())
        preds.append(pred_raw.squeeze(0).cpu().numpy().astype(np.float32))
        tgts.append(tgt.squeeze(0).cpu().numpy().astype(np.float32))

        if has_bounds:
            lower_norm = torch.as_tensor(lower_val)
            upper_norm = torch.as_tensor(upper_val)
            if tuple(lower_norm.shape) != expected_shape or tuple(upper_norm.shape) != expected_shape:
                raise SystemExit(
                    f"ttt_step returned SPS bounds shape {tuple(lower_norm.shape)}/"
                    f"{tuple(upper_norm.shape)}, expected {expected_shape}."
                )
            lower_raw = normalizer.postprocess_pred(lower_norm.detach())
            upper_raw = normalizer.postprocess_pred(upper_norm.detach())
            lowers.append(lower_raw.squeeze(0).cpu().numpy().astype(np.float32))
            uppers.append(upper_raw.squeeze(0).cpu().numpy().astype(np.float32))

    pred_all = np.stack(preds, axis=0)
    tgt_all = np.stack(tgts, axis=0)
    lower_all = np.stack(lowers, axis=0) if bounds_active else None
    upper_all = np.stack(uppers, axis=0) if bounds_active else None
    c = measured_channels(tgt_all)
    mean_t = float(np.mean(per_step_times))

    print(f"[local_eval] prediction stack shape {pred_all.shape}")
    print(f"[local_eval] mean per-step time: {mean_t * 1e3:.2f} ms "
          f"(total {sum(per_step_times):.3f} s over {len(stream)} steps)")
    bounds_note = ("participant-supplied (info.lower/info.upper)" if bounds_active
                   else "none -> scorer default +/-5% band")
    print(f"[local_eval] SPS bounds: {bounds_note}")

    # Real subscores from the bundled scoring.py. On the tiny example_data these
    # are illustrative, NOT leaderboard-comparable; point --data at real data for
    # the true metric, and time_score reflects LOCAL wall time.
    try:
        sys.path.insert(0, str(HERE))
        import scoring as official
    except Exception as exc:  # noqa: BLE001
        print(f"[local_eval] bundled scoring.py not available ({type(exc).__name__}: {exc}); "
              "shape/plumbing check still passed, but subscores were not computed.")
        return

    try:
        rl = float(np.mean(official.rel_l2_per_sample(pred_all, tgt_all, c)))
        tk = float(np.mean(official.tke_rel_l2_per_sample(pred_all, tgt_all, c)))
        mv = official.mvpe_rel_l2(pred_all, tgt_all)
        sps, _cov = official.aggregate_sps(pred_all, tgt_all, c, lower=lower_all, upper=upper_all)
        subscores = {
            "rel_l2_score": official.score_error(rl),
            "tke_score": official.score_error(tk),
            "mvpe_score": official.score_error(mv),
            "time_score": official.score_time(mean_t),
            "sps_score": official.score_sps(sps),
        }
    except Exception as exc:  # noqa: BLE001 - the platform zeroes this class of
        # failure rather than crashing, so report the same verdict here.
        print(f"[local_eval] scoring.py FAILED on this submission's output "
              f"({type(exc).__name__}: {exc})")
        print("[local_eval] this means the REAL scorer would zero every subscore for "
              "this run (rel_l2_score/tke_score/mvpe_score/time_score/sps_score all 0) "
              "-- this is NOT a passing result.")
        # Non-zero exit, so a caller that only checks $? does not read this
        # as success.
        raise SystemExit(1)

    print("[local_eval] real subscores from bundled scoring.py "
          "(example data, NOT leaderboard):")
    for name, val in subscores.items():
        print(f"[local_eval]     {name:12s} {val:7.3f}")
    print("[local_eval] the leaderboard combines these five into final_score by a "
          "rule that is not published, so no total is shown here.")
    print("[local_eval] OK: submission ran end-to-end with correct shapes.")


if __name__ == "__main__":
    main()
