#!/usr/bin/env python3
"""Map the Final kit's read-only /data into the historical recipe's legal pools.

All links point to files in the released training mount. No data are copied or
simulated. The historical 72/9 split is an internal split of the 81 legal real
files; the final students consume all81.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
THIS = Path(__file__).resolve().parent
VAL_RE = {11400, 17775}


def link(src: Path, dst: Path) -> None:
    assert src.is_file(), src
    if dst.is_symlink():
        assert dst.resolve() == src.resolve(), (dst, "symlink points to wrong file")
    else:
        assert not dst.exists(), ("refuse to replace existing file", dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)


def stage(data_dir: str, output_dir: str) -> dict:
    data = Path(data_dir).resolve()
    output = Path(output_dir).resolve()
    assert data.is_dir() and output.is_dir(), (data, output)
    real = sorted((data / "train_real").glob("*.h5"))
    sim = sorted((data / "train_sim").glob("*.h5"))
    assert len(real) == 81 and len(sim) == 100, (len(real), len(sim))
    assert "7575_0.h5" not in {p.name for p in real}
    assert (data / "mean_std_real.pt").is_file()
    # The inherited sband fitter uses these public-release std values in its
    # physical-to-normalized interval conversion.  Refuse a mismatched mount
    # before spending GPU hours or silently changing its interval units.
    import numpy as np
    import torch
    stats = torch.load(data / "mean_std_real.pt", map_location="cpu", weights_only=False)
    assert isinstance(stats, (tuple, list)) and len(stats) == 4
    std_uv = np.asarray(stats[3], np.float64).ravel()[:2]
    assert np.allclose(std_uv, [0.09681041, 0.01596364], rtol=1e-5, atol=1e-7), std_uv
    selected = {"train72": [], "val9": [], "all81": []}
    for p in real:
        group = int(p.stem.split("_", 1)[0])
        target = "val9" if group in VAL_RE else "train72"
        selected[target].append(p.name)
        selected["all81"].append(p.name)
        link(p, output / "work/data" / target / "test_real" / p.name)
        link(p, output / "work/data/all81/test_real" / p.name)
    assert (len(selected["train72"]), len(selected["val9"]), len(selected["all81"])) == (72, 9, 81)
    stats_copy = output / "mean_std_real.pt"
    if stats_copy.exists():
        assert stats_copy.read_bytes() == (data / "mean_std_real.pt").read_bytes()
    else:
        shutil.copyfile(data / "mean_std_real.pt", stats_copy)
    sim_stats = output / "work/data/mean_std_comp_sim.pt"
    if not sim_stats.exists():
        subprocess.run([sys.executable, str(THIS / "compute_sim_stats.py"), "--data-dir", str(data),
                        "--out", str(sim_stats)], check=True, cwd=REPO)
    stats_proof = json.loads(Path(str(sim_stats) + ".json").read_text())
    assert stats_proof["source"] == "released train_sim" and stats_proof["spatial_stride"] == 2
    assert stats_proof["files"] == [p.name for p in sim]
    assert stats_proof["output_sha256"] == hashlib.sha256(sim_stats.read_bytes()).hexdigest()
    summary = {"source": str(data), "real_files": [p.name for p in real], "sim_files": [p.name for p in sim],
               "train72": selected["train72"], "val9": selected["val9"], "all81": selected["all81"],
               "stats_real": str(stats_copy), "stats_sim": str(sim_stats)}
    plan = output / "work/data_manifest.json"
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if plan.exists():
        assert plan.read_text() == text, "data staging changed"
    else:
        plan.write_text(text)
    print("FINAL_DATA_STAGE_OK", len(real), len(sim), len(selected["train72"]), len(selected["val9"]))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()
    stage(a.data_dir, a.output_dir)
