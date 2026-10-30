#!/usr/bin/env python3
"""Run the frozen historical trainer with one strict Final-mount path adaptation.

The original source has a guard for the old gdrive_raw path. Final-kit training
mounts the same released files at /data. We replace exactly that guard's one
condition in memory and require common.final_source_gate to verify every
resolved file. No optimizer, model, objective, RNG, schedule or checkpoint
selection statement is changed. The original and adapted source hashes are
logged so a reviewer can inspect the sole difference.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
ALLOWED = {
    "Track1/U-net/train_v4.py",
    "Track1/U-net/VariantRetrainPush/train_objective.py",
}
OLD = 'if "/data/RealPDE_competition_dataset/gdrive_raw/" not in _g:'
NEW = ('if "/data/RealPDE_competition_dataset/gdrive_raw/" not in _g '
       'and not __import__("final_source_gate").verify_final_source(_g, os.environ.get("RPDE_FINAL_DATA_ROOT")):')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", required=True, choices=sorted(ALLOWED))
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpu", required=True)
    args = ap.parse_args()
    assert not os.environ.get("RPDE_ALLOW_ANY_SOURCE"), "broad source bypass is forbidden in Final training"
    assert os.environ.get("RPDE_FINAL_DATA_ROOT"), "Final data root must be explicit"
    src = (ROOT / args.trainer).resolve()
    assert src.is_file() and src.is_relative_to(ROOT)
    code = src.read_text()
    assert code.count(OLD) == 1 and NEW not in code, "trainer guard changed; review before adapting"
    adapted = code.replace(OLD, NEW)
    print("FINAL_TRAINER_SOURCE", args.trainer, "original_sha256", hashlib.sha256(code.encode()).hexdigest(),
          "adapted_sha256", hashlib.sha256(adapted.encode()).hexdigest(), flush=True)
    sys.path.insert(0, str(src.parent))
    sys.path.insert(0, str(ROOT / "common"))
    sys.argv = [str(src), "--config", args.config, "--gpu", args.gpu]
    context = {"__name__": "__main__", "__file__": str(src), "__package__": None}
    exec(compile(adapted, str(src), "exec"), context)


if __name__ == "__main__":
    main()
