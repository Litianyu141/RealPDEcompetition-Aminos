#!/usr/bin/env python3
"""Produce the Track-2 BOOTSTRAP calibration from the competition release alone.

WHY THIS EXISTS (docs/decision/RECIPE.md open item 2, 2026-09-12).
`reproduce_t2.sh` stage 3b builds a throwaway bundle (`t2_boot`) purely to drive
`t2_collect_residuals.py`, and it passes `--calib $D/calib_v22_pc_m120.npz` -- a learned blob that
exists only as an UNTRACKED file on two servers and, per the chain's own OPEN GAP note, descends from
`calib_v4.npz`, which "has no reconstruction script".  A clean clone therefore cannot run stage 3b at
all, and the official Decision-Phase requirement is training code only, no weights.

The blob is not actually needed.  The boot bundle is driven with `--online '{}'`, so every rate that
would read the band/gamma/bias arrays is zero; the only calib keys that can reach the PREDICTION are the
three the variance head consumes, and all three are statistics of the benchmark, not fitted parameters:

    norm_mean = mean over the released real split, per channel
    norm_std  = std  over the released real split, per channel (a zero channel is set to 1.0)
    ke_w[c]   = norm_std[c]**2        # scoring.kinetic_energy is computed on PHYSICAL fields, so
                                      # var_phys[c] = std_t[c]**2 * var_norm[c]  (t2_v6_prep.py)

They are read from the kit's own TRACKED `example_data/mean_std_real.pt`, exactly as `t2_v6_prep.py`
does, and they reproduce the historical blob's three arrays **bit-for-bit** (verified on V100 against
`calib_v22_pc_m120.npz`: max|diff| = 0.000e+00 on all three).  So the deepest declared input in the T2
chain is replaced by ~40 lines that read a tracked file.

This script emits ONLY those keys plus `ch_idx`.  It deliberately does not invent band/gamma arrays:
with `--online '{}'` they are not applied, and fabricating them would let a wrong band masquerade as a
reproduced one.  Whether the omission is inert was not assumed: an A/B check (2026-09-12) ran stage 3b's residual collection
with the historical blob and with this file and compared the caches (seven residual arrays bit-identical).

usage:  python3 Track2/U-net/make_boot_calib.py --out Track2/U-net/calib_boot.npz [--kit v6|v4]

AFFINE SELECTION (2026-09-21): the legacy default remains the TARGET affine
(tuple positions 1/3). The current v68 runtime calibration and t2_fit_band.py
carry the INPUT affine (positions 0/2), which differs slightly in both u/v
channels. Fresh re0/final-fit prefit collection must pass --affine input so
its variance-head predictions use the same fields as the final calibration.
The selector changes only which published constants are copied; no values are
fitted, and zero standard deviations retain the historical replacement by 1.
"""
from __future__ import annotations

import argparse
import datetime
import pathlib

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
R = HERE.parents[1]
KITS = {"v6": R / "Track2/realpde_t2_starting_kit_v6",
        "v4": R / "Track2/realpde_t2_starting_kit_v4"}


def stats_from_kit(kit: pathlib.Path, affine: str = "target"):
    if affine not in ("target", "input"):
        raise ValueError("affine must be target or input")
    p = kit / "example_data/mean_std_real.pt"
    if not p.is_file():
        raise SystemExit("MISSING %s -- the kit's own normalisation stats are tracked; a clone without "
                         "them is not a clone of this repo" % p)
    mi, mt, si, st = torch.load(p, map_location="cpu", weights_only=False)
    if affine == "input":
        mt, st = mi, si
    mt = np.asarray(mt, np.float32).ravel().copy()
    st = np.asarray(st, np.float32).ravel().copy()
    nc = int(np.count_nonzero(st))            # measured channels: u, v (real p is zero-filled)
    if nc < 2:
        raise SystemExit("mean_std_real.pt reports %d nonzero channels; expected u and v" % nc)
    ke_w = (st[:nc] ** 2).astype(np.float32)
    norm_mean = mt
    norm_std = st.copy()
    norm_std[norm_std == 0] = 1.0             # t2_v6_prep.py's convention, and the card's own
    if not (np.all(ke_w > 0) and np.isfinite(ke_w).all()
            and np.isfinite(norm_mean).all() and np.isfinite(norm_std).all()):
        raise SystemExit("non-finite or non-positive statistics; refusing to emit a boot calib")
    return norm_mean, norm_std, ke_w, p


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(HERE / "calib_boot.npz"))
    ap.add_argument("--kit", default="v6", choices=sorted(KITS), help="which starting kit's stats to read")
    ap.add_argument("--affine", default="target", choices=("target", "input"),
                    help="published affine; input matches v68/t2_fit_band, target preserves legacy default")
    ap.add_argument("--assert-equals", default="",
                    help="optional historical calib to compare against; refuse unless the three "
                         "reconstructed arrays are bit-identical to it")
    a = ap.parse_args()

    norm_mean, norm_std, ke_w, src = stats_from_kit(KITS[a.kit], affine=a.affine)
    # ch_idx marks the measured channels for the scorer's u,v-only convention; t2_fit_band.py writes
    # the same constant, so this is not a new invention.
    out = {"norm_mean": norm_mean, "norm_std": norm_std, "ke_w": ke_w,
           "ch_idx": np.array([0, 1, 1], np.int64)}

    if a.assert_equals:
        ref = dict(np.load(a.assert_equals, allow_pickle=True))
        bad = []
        for k, v in (("norm_mean", norm_mean), ("norm_std", norm_std), ("ke_w", ke_w)):
            if k not in ref:
                bad.append("%s: absent from the reference calib" % k)
            elif not np.array_equal(v.astype(np.float32), np.asarray(ref[k], np.float32)):
                bad.append("%s: max|diff| %.3e" % (k, float(np.abs(
                    v.astype(np.float64) - np.asarray(ref[k], np.float64)).max())))
        if bad:
            raise SystemExit("BOOT_CALIB_MISMATCH vs %s:\n  %s" % (a.assert_equals, "\n  ".join(bad)))
        print("# bit-identical to %s on norm_mean / norm_std / ke_w" % a.assert_equals)

    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    # np.savez writes fixed 1980 zip timestamps, so this output is byte-reproducible run to run --
    # the same property make_calib_sbport.py relies on for its content guard.
    np.savez(p, **out)
    print("# boot calib %s  (%d bytes)" % (p, p.stat().st_size))
    print("#   source     %s" % src)
    print("#   affine     %s" % a.affine)
    print("#   generated  %s" % datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    for k in sorted(out):
        print("#   %-10s %s %s" % (k, np.asarray(out[k]).shape, np.round(np.asarray(out[k]).ravel()[:4], 8)))
    print("MAKE_BOOT_CALIB_OK")


if __name__ == "__main__":
    main()
