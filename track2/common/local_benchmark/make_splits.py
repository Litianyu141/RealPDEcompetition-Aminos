#!/usr/bin/env python3
"""Build a condition-disjoint (Reynolds-holdout) local TRAIN/VAL split of the OFFICIAL public
train_real / train_sim release, and materialize kit-style eval dirs (test_real/*.h5 symlinks +
mean_std_real.pt) so the official scoring.py can be run locally on held-out data.

2-WAY split (train / val only): we are splitting the competition's OWN public training release, so
the Codabench leaderboard itself is the held-out test set. WHOLE Reynolds numbers are held out for
val, so a local score estimates generalization to UNSEEN flow conditions (interpolation across Re)
rather than memorization. Uses ONLY the official public training release — it does NOT touch, probe,
or reconstruct the hidden competition test set.
"""
import os, json, glob, shutil
from pathlib import Path
import sys as _sys
_sys.path.append(str(Path(__file__).resolve().parents[1]))
from repo_paths import relativize_paths   # manifests store repo-relative paths

# Derived from this file's own location (2026-08-12, F2): the absolute hardcode silently
# escaped a fresh clone back to the production tree.
REPO = Path(__file__).resolve().parents[2]   # common/local_benchmark/<this> -> the repo root (moved under common/ 2026-09-08; parents[1] pointed at common/ and broke the clean-clone chain 2026-09-09)
REAL_DIR = REPO / "data/RealPDE_competition_dataset/gdrive_raw/extracted/train_real/train_real"
SIM_DIR = REPO / "data/RealPDE_competition_dataset/gdrive_raw/extracted/train_sim/train_sim"
STATS = REPO / "Track2/realpde_t2_starting_kit_v4/example_data/mean_std_real.pt"
OUT = REPO / "data/RealPDE_competition_dataset/gdrive_raw/local_bench"                 # symlink eval dirs (under ignored data/)
MANIFEST = Path(__file__).resolve().parent / "splits.json"

# Held-out Reynolds numbers (whole conditions) for validation, chosen to span the range and stay
# disjoint from train. Everything else is train. (The Codabench score is the "test" set.)
VAL_RE = {11400, 17775}


def parse(fn):
    re_, aoa = os.path.basename(fn)[:-3].split("_")
    return int(re_), int(aoa)


# OFFICIAL EXCLUSION (RealPDE Committee email, 2026-08-13): train_real/7575_0.h5 "incorrectly
# reused data from 6300_0"; participants must not adopt it during development. This confirms the
# repo's 2026-08-06 bit-identical-fingerprint finding. Excluded from every materialized pool.
BANNED = {"7575_0.h5"}

def assign(files):
    d = {"train": [], "val": []}
    for f in sorted(files):
        re_, _ = parse(f)
        split = "val" if re_ in VAL_RE else "train"
        d[split].append(os.path.basename(f))
    return d


def link_dir(dst, names, src_dir, with_stats=True):
    tr = dst / "test_real"
    tr.mkdir(parents=True, exist_ok=True)
    for old in list(tr.glob("*.h5")):
        old.unlink()
    for name in names:
        link = tr / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(src_dir / name)
    if with_stats:
        s = dst / "mean_std_real.pt"
        if s.exists() or s.is_symlink():
            s.unlink()
        s.symlink_to(STATS)


def main():
    _all_real = glob.glob(str(REAL_DIR / "*.h5"))
    _drop = [f for f in _all_real if os.path.basename(f) in BANNED]
    for f in _drop:
        print(f"  EXCLUDED (official notice 2026-08-13): {os.path.basename(f)}")
    real = assign([f for f in _all_real if os.path.basename(f) not in BANNED])
    sim = assign(glob.glob(str(SIM_DIR / "*.h5")))
    for split in ("train", "val"):
        link_dir(OUT / split, real[split], REAL_DIR)     # kit-style dir for scoring on real
    # remove any stale 3-way `test` dir from an earlier run
    old_test = OUT / "test"
    if old_test.exists():
        shutil.rmtree(old_test)
    manifest = {
        "scheme": "Reynolds-holdout (condition-disjoint), 2-way TRAIN/VAL; OFFICIAL public train "
                  "release only; Codabench leaderboard is the held-out test; hidden competition "
                  "test set NOT used",
        "held_out_Re": {"val": sorted(VAL_RE)},
        "counts": {"real": {k: len(v) for k, v in real.items()},
                   "sim": {k: len(v) for k, v in sim.items()}},
        "eval_dirs": {s: str(OUT / s) for s in ("train", "val")},
        "real": real,
        "sim": sim,
    }
    MANIFEST.write_text(json.dumps(relativize_paths(manifest), indent=2))
    print("REAL counts:", manifest["counts"]["real"], "(held-out val Re", sorted(VAL_RE), ")")
    print("SIM  counts:", manifest["counts"]["sim"])
    print("eval dirs   :", str(OUT))
    print("manifest    :", str(MANIFEST))
    print("SPLITS_OK")


if __name__ == "__main__":
    main()
