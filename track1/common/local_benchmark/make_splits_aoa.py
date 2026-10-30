#!/usr/bin/env python3
"""ANGLE-OF-ATTACK-grouped K-fold split: a local benchmark that tests REGIME SHIFT, not
interpolation.

WHY THIS EXISTS, AND WHY THE REYNOLDS K-FOLD WAS NOT ENOUGH.  `make_splits_kfold.py` replaced the
2-Reynolds/9-file/378-window split with 6 Reynolds-grouped folds covering all 82 files out-of-fold
(3383 windows, zero leakage).  Measured 2026-07-28, it STILL mis-ranks: it prefers the v3 objective
(`rel_l2` + `tke_weight`) over the v2 objective by +0.529 S3 at paired t=9.77 on 6/6 folds, while
the hidden test prefers v2 by ~0.32 S3-equivalent.  So the public/hidden discrepancy is NOT sampling
noise in the split -- more public data, split better, does not help.

WHAT THE FAILURE POINTED AT.  Compare each shipped model's OWN public-data error with its measured
online error (same model, both numbers real):

    subscore   v2_b64 local -> online      v3_relt20 local -> online     inflation
    rel_l2     0.0728 -> 0.1130            0.0738 -> 0.1159              x1.55 / x1.57
    tke        0.6272 -> 0.6721            0.5721 -> 0.6631              x1.07 / x1.16
    mvpe       0.0650 -> 0.1439            0.0659 -> 0.1692              x2.21 / x2.57

`tke` transfers almost perfectly.  `mvpe` roughly DOUBLES, and the model-to-model mvpe GAP inflates
from 0.0009 locally to 0.0253 online -- a factor of 27.  That is the whole v8-vs-v9 story: any change
that trades mvpe for tke looks good on public data and bad on the leaderboard.

mvpe is the time-averaged u,v profile at 4 FIXED streamwise stations behind the airfoil.  What moves
that profile is not Reynolds -- it is the ANGLE OF ATTACK, which moves the wake itself.  Reynolds
folds interpolate inside a dense 18-value sweep and never stress it.  So: hold out an AoA.

WHAT THIS DOES.  5 folds, one per AoA in {0,5,10,15,20}; the held-out angle is dropped from BOTH the
real and the sim pool (sim shares the AoA axis, so leaving it in leaks the holdout through
pretraining).  Every real file is scored exactly once out-of-fold, same as the Reynolds manifest.
This is a HARDER benchmark by construction -- each fold extrapolates to an unseen angle -- and that
is the point.

THE TEST IT EXISTS TO RUN.  Train the v2 objective (arm A) and the v3 objective (arm B) on these
folds.  The hidden test ranks A above B.  If AoA-holdout ALSO ranks A above B, we finally have a
local benchmark whose verdicts can be trusted; if it too prefers B, then no split of the public real
data can arbitrate this and the leaderboard is the only arbiter at 1 submission/day.

    python make_splits_aoa.py [--out splits_aoa.json]
"""
from __future__ import annotations
import argparse, glob, json, re
from collections import defaultdict
from pathlib import Path
import sys as _sys
_sys.path.append(str(Path(__file__).resolve().parents[1]))
from repo_paths import relativize_paths   # manifests store repo-relative paths

# Derived from this file's own location (2026-08-12, F2): the absolute hardcode silently
# escaped a fresh clone back to the production tree.
REPO = Path(__file__).resolve().parents[2]   # common/local_benchmark/<this> -> the repo root (moved under common/ 2026-09-08; parents[1] pointed at common/ and broke the clean-clone chain 2026-09-09)
REAL = REPO / "data/RealPDE_competition_dataset/gdrive_raw/extracted/train_real/train_real"
SIM = REPO / "data/RealPDE_competition_dataset/gdrive_raw/extracted/train_sim/train_sim"
STATS_REAL = REPO / "Track2/realpde_t2_starting_kit_v4/example_data/mean_std_real.pt"
STATS_SIM = REPO / "data/RealPDE_competition_dataset/mean_std_comp_sim.pt"

# OFFICIAL EXCLUSION (RealPDE Committee email, 2026-08-13) -- applied to the REAL pool in main().
BANNED = {"7575_0.h5"}


def file_aoa(p):
    m = re.match(r"\d+_(\d+)", Path(p).name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "splits_aoa.json"))
    a = ap.parse_args()

    real = sorted(glob.glob(str(REAL / "*.h5")))
    sim = sorted(glob.glob(str(SIM / "*.h5")))
    assert real, f"no real files under {REAL}"
    # OFFICIAL EXCLUSION (RealPDE Committee email, 2026-08-13): train_real/7575_0.h5
    # "incorrectly reused data from 6300_0"; participants must not use it.  Unconditional --
    # same gate as make_splits.py (commit c097d32) and train_v4.py.  The sim file of the same
    # name is a different, legitimate CFD trajectory and stays.
    for f in [f for f in real if Path(f).name in BANNED]:
        print(f"  EXCLUDED (official notice 2026-08-13): {Path(f).name}")
    real = [f for f in real if Path(f).name not in BANNED]

    by = defaultdict(list)
    for f in real:
        by[file_aoa(f)].append(f)
    angles = sorted(by)

    folds = []
    for k, ang in enumerate(angles):
        val_files = sorted(by[ang])
        train_files = sorted(f for f in real if file_aoa(f) != ang)
        sim_train = sorted(f for f in sim if file_aoa(f) != ang)
        folds.append({"fold": k, "val_groups": [ang], "val_re": [ang],
                      "n_val_files": len(val_files), "n_train_files": len(train_files),
                      "n_sim_train_files": len(sim_train),
                      "val_files": val_files, "train_files": train_files,
                      "sim_train_files": sim_train})

    assert sorted(g for f in folds for g in f["val_groups"]) == angles, "folds do not partition AoA"
    n_val = sum(f["n_val_files"] for f in folds)
    assert n_val == len(real), f"out-of-fold coverage {n_val} != {len(real)}"

    man = {
        "scheme": ("AoA-GROUPED leave-one-angle-out, one fold per angle of attack. Held out from "
                   "BOTH real and sim. Official public release only; the hidden competition test "
                   "set is NOT used. Every real file is scored exactly once out-of-fold. Tests "
                   "REGIME SHIFT (the wake moves with AoA) rather than interpolation in Reynolds, "
                   "because mvpe -- the subscore whose public->hidden gap inflates 27x -- is a "
                   "wake profile at 4 fixed streamwise stations."),
        "group_by": "aoa",
        "k": len(angles),
        "n_real_files": len(real), "n_sim_files": len(sim),
        "all_groups": angles, "all_aoa": angles,
        "stats_real": str(STATS_REAL), "stats_sim": str(STATS_SIM),
        "real_glob": str(REAL / "*.h5"), "sim_glob": str(SIM / "*.h5"),
        "folds": folds,
    }
    Path(a.out).write_text(json.dumps(relativize_paths(man), indent=2))

    print(f"# {len(real)} real files, AoA {angles} -> {len(angles)} folds")
    for f in folds:
        print(f"  fold {f['fold']}: val AoA {f['val_groups']}  "
              f"val {f['n_val_files']} files / train {f['n_train_files']} / sim {f['n_sim_train_files']}")
    print(f"# out-of-fold coverage {n_val}/{len(real)} files (every file exactly once)")
    print(f"# wrote {a.out}")
    print("MAKE_SPLITS_AOA_OK")


if __name__ == "__main__":
    main()
