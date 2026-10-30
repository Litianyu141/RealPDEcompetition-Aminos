#!/usr/bin/env python3
"""Add `ke_w` to the Track-2 calibration so v6's gain field can be measured in the SCORER's units.

`submission_t2_v*.py` works entirely in normalised space -- the harness normalises the input and
de-normalises the output, and the bundle never ships `mean_std_real.pt`. But
`scoring.kinetic_energy` is computed on PHYSICAL fields, and

    var_phys[c] = std_t[c]**2 * var_norm[c]

with std_t = [0.0968, 0.0160] for real data, so u carries **36x** the physical fluctuation energy of
v. Summing NORMALISED channel variances -- which is what v5's scalar `cn` does -- therefore
estimates a different quantity from the one that is scored. v6's field estimator weights by
`ke_w[c] = std_t[c]**2` instead; this script computes that vector once and copies it into the calib
alongside everything v5 already ships, byte-for-byte.

    python t2_v6_prep.py        # calib_v4.npz -> calib_v7.npz  (+ ke_w, norm_mean, norm_std)
"""
from pathlib import Path
import numpy as np
import torch

T2U = Path(__file__).resolve().parent
# Kit derived from this file's own location (2026-08-12, F2; kit v6 = the 2026-08-05 reset):
# the absolute hardcode silently escaped a fresh clone back to the production tree.
KIT = T2U.parent / "realpde_t2_starting_kit_v6"
SRC, OUT = T2U / "calib_v4.npz", T2U / "calib_v7.npz"

src = dict(np.load(SRC, allow_pickle=True))
print(f"# source {SRC}\n#   keys {sorted(src)}")

_, _, _, std_t = torch.load(KIT / "example_data/mean_std_real.pt", map_location="cpu",
                            weights_only=False)
std_t = np.asarray(std_t, np.float32).ravel()
nc = int(np.count_nonzero(std_t))                     # measured channels: u, v (p has std 0)
ke_w = (std_t[:nc] ** 2).astype(np.float32)
print(f"#   std_t {std_t} -> nc={nc}, ke_w={ke_w} (ratio u:v = {ke_w[0]/ke_w[1]:.1f}:1)")
assert nc >= 2 and np.all(ke_w > 0) and np.isfinite(ke_w).all()

# v7 also carries the normalisation itself.  The Track-2 bundle never ships mean_std_real.pt and
# works entirely in normalised space, but the learned variance head consumes PHYSICAL summary fields
# (that is where it was trained and where the scorer computes KE), so the submission needs mean/std
# to build its features.  They are constants of the benchmark, not test information.
mi, mt, si, st = torch.load(KIT / "example_data/mean_std_real.pt", map_location="cpu",
                            weights_only=False)
norm_mean = np.asarray(mt, np.float32).ravel().copy()
norm_std = np.asarray(st, np.float32).ravel().copy()
norm_std[norm_std == 0] = 1.0
print(f"#   norm_mean {norm_mean}  norm_std {norm_std}")

merged = dict(src); merged["ke_w"] = ke_w
merged["norm_mean"] = norm_mean
merged["norm_std"] = norm_std
np.savez(OUT, **merged)

chk = np.load(OUT, allow_pickle=True)
for k in src:                                          # nothing v5 relies on may have moved
    assert np.array_equal(np.asarray(chk[k]), np.asarray(src[k])), f"{k} was modified"
assert np.array_equal(np.asarray(chk["ke_w"]), ke_w)
assert np.array_equal(np.asarray(chk["norm_std"]), norm_std)
print(f"# wrote {OUT}  keys {sorted(chk.files)}")
print("T2_V6_PREP_OK")
