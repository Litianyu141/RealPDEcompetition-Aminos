"""Strict Final-kit training-source gate for the official read-only /data mount.

Historical trainers accepted only the old local gdrive_raw path. The Final kit
mounts exactly 100 sim and 81 legal real files at a different path. This gate
accepts a source glob only if every resolved file comes from one of those two
mount folders. It never enables the broad RPDE_ALLOW_ANY_SOURCE bypass.
"""
from __future__ import annotations

import glob
from pathlib import Path


def verify_final_source(pattern: str, data_root: str | None) -> bool:
    if not data_root:
        return False
    root = Path(data_root).resolve(strict=True)
    sim = (root / "train_sim").resolve(strict=True)
    real = (root / "train_real").resolve(strict=True)
    assert sim.is_dir() and real.is_dir() and (root / "mean_std_real.pt").is_file()
    sim_names = {p.name for p in sim.glob("*.h5")}
    real_names = {p.name for p in real.glob("*.h5")}
    assert len(sim_names) == 100 and len(real_names) == 81 and "7575_0.h5" not in real_names, \
        ("unexpected Final mount", len(sim_names), len(real_names))
    files = [Path(p).resolve(strict=True) for p in sorted(glob.glob(pattern))]
    assert files, ("empty Final source", pattern)
    parents = {p.parent for p in files}
    assert len(parents) == 1 and parents <= {sim, real}, ("outside Final mount", pattern, parents)
    assert all(p.suffix == ".h5" for p in files)
    assert all(p.name != "7575_0.h5" for p in files if p.parent == real)
    return True
