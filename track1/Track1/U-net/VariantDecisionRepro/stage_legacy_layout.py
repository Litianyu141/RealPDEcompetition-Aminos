#!/usr/bin/env python3
"""Expose the Final kit's released /data to the unchanged historical recipe.

Only symlinks and a run directory are made. No HDF5 file is copied, and a
pre-existing different path is never overwritten. Intended for the clean
/workspace image; the production checkout with real data will refuse.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def directory_link(dst: Path, src: Path) -> None:
    assert src.is_dir(), src
    if dst.is_symlink():
        assert dst.resolve() == src.resolve(), (dst, src)
    else:
        assert not dst.exists(), ("legacy path already exists", dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)


def file_link(dst: Path, src: Path) -> None:
    assert src.is_file(), src
    if dst.is_symlink():
        assert dst.resolve() == src.resolve(), (dst, src)
    elif dst.is_file():
        assert dst.read_bytes() == src.read_bytes(), ("legacy stats differ", dst)
    else:
        assert not dst.exists(), dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)


def stage(workspace: str, data_dir: str, output_dir: str, tag: str) -> None:
    root = Path(workspace).resolve(strict=True)
    data = Path(data_dir).resolve(strict=True)
    output = Path(output_dir).resolve(strict=True)
    assert tag.isalnum() and tag.startswith("dp"), tag
    assert (root / "Track1/U-net/reproduce_t1.sh").is_file()
    assert (output / "work/data_manifest.json").is_file()
    legacy = root / "data/RealPDE_competition_dataset"
    raw = legacy / "gdrive_raw"
    directory_link(raw / "extracted/train_real/train_real", data / "train_real")
    directory_link(raw / "extracted/train_sim/train_sim", data / "train_sim")
    for old, new in (("train", "train72"), ("val", "val9"), ("all81", "all81")):
        directory_link(raw / "local_bench" / old / "test_real", output / "work/data" / new / "test_real")
        file_link(raw / "local_bench" / old / "mean_std_real.pt", data / "mean_std_real.pt")
    file_link(legacy / "mean_std_comp_sim.pt", output / "work/data/mean_std_comp_sim.pt")
    file_link(root / "Track2/realpde_t2_starting_kit_v4/example_data/mean_std_real.pt", data / "mean_std_real.pt")
    file_link(root / "Track2/realpde_t2_starting_kit_v6/example_data/mean_std_real.pt", data / "mean_std_real.pt")
    base = output / "work/base_t1"
    base.mkdir(parents=True, exist_ok=True)
    directory_link(root / "Track1/U-net" / tag, base)
    t2base = output / "work/base_t2"
    t2base.mkdir(parents=True, exist_ok=True)
    directory_link(root / "Track2/U-net/dp156base", t2base)
    for relative, target in (
        ("Track1/U-net/submissions", "work/submissions_t1"),
        ("Track2/U-net/submissions", "work/submissions_t2"),
        ("Track1/U-net/sband", "work/t1_sband"),
        ("Track1/U-net/Logger", "work/t1_trainer_logs"),
        ("Track2/U-net/Logger", "work/t2_trainer_logs"),
        ("checkpoints/t1_u-net", "work/t1_fold_resolution"),
        ("_tmp", "work/_tmp"),
        ("Track1/U-net/VariantDecisionRepro/Logger", "work/decision_t1"),
        ("Track2/U-net/VariantDecisionRepro/Logger", "work/decision_t2"),
    ):
        dst = output / target
        dst.mkdir(parents=True, exist_ok=True)
        directory_link(root / relative, dst)
    print("LEGACY_LAYOUT_READY", root, data, output, tag)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tag", default="dpv185base")
    a = ap.parse_args()
    stage(a.workspace, a.data_dir, a.output_dir, a.tag)
