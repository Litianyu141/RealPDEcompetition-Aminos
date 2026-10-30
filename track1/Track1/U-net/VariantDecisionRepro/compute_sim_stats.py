#!/usr/bin/env python3
"""Derive the missing sim mean/std from the released train_sim files.

The Final kit mounts only train_sim/, train_real/ and mean_std_real.pt. Historical
configs also need mean_std_comp_sim.pt; this producer derives it inside train()
without carrying a fitted statistics file into the image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare-existing")
    ap.add_argument("--spatial-stride", type=int, default=2,
                    help="the historical trainer uses 32x64 fields, subsampled by two from the released 64x128 sim grid")
    a = ap.parse_args()
    assert a.spatial_stride in (1, 2)
    folder = Path(a.data_dir).resolve() / "train_sim"
    files = sorted(folder.glob("*.h5"))
    assert len(files) == 100 and len({p.name for p in files}) == 100, len(files)
    sums = np.zeros(3, np.float64)
    sums2 = np.zeros(3, np.float64)
    counts = np.zeros(3, np.int64)
    shapes = set()
    for file in files:
        with h5py.File(file, "r") as f:
            for j, key in enumerate(("u", "v", "p")):
                ds = f[key]
                assert len(ds.shape) == 3, (file, key, ds.shape)
                shapes.add(tuple(ds.shape[1:]))
                for start in range(0, ds.shape[0], 32):
                    x = np.asarray(ds[start:start + 32, ::a.spatial_stride, ::a.spatial_stride], np.float64)
                    assert np.isfinite(x).all(), (file, key, start)
                    sums[j] += x.sum(dtype=np.float64)
                    sums2[j] += np.square(x).sum(dtype=np.float64)
                    counts[j] += x.size
    assert len(shapes) == 1 and np.all(counts == counts[0]), (shapes, counts)
    mean = (sums / counts).astype(np.float32)
    var = np.maximum(sums2 / counts - (sums / counts) ** 2, 0.0)
    std = np.sqrt(var).astype(np.float32)
    assert np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()
    out = Path(a.out).resolve()
    assert not out.exists(), out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tuple(torch.from_numpy(x.copy()) for x in (mean, mean, std, std)), out)
    record = {"source": "released train_sim", "spatial_stride": a.spatial_stride,
              "files": [p.name for p in files], "shapes": [list(s) for s in shapes],
              "counts": counts.tolist(), "mean": mean.tolist(), "std": std.tolist(), "output": str(out),
              "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    if a.compare_existing:
        old = torch.load(a.compare_existing, map_location="cpu", weights_only=False)
        want = [mean, mean, std, std]
        record["historical_max_abs"] = [float(np.max(np.abs(np.asarray(x) - y))) for x, y in zip(old, want)]
    Path(str(out) + ".json").write_text(json.dumps(record, indent=2) + "\n")
    print("SIM_STATS_OK", len(files), "mean", mean.tolist(), "std", std.tolist(),
          "historical_max_abs", record.get("historical_max_abs"))


if __name__ == "__main__":
    main()
