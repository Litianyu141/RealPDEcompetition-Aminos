#!/usr/bin/env python3
"""Fit v156g's LLR1 P1 within one fixed outer fold's training files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SPLIT = ROOT / "common/local_benchmark/splits_re3_aoa2_v1.json"
sys.path.insert(0, str(ROOT / "common"))
from ll_ridge_v155_card import npy_bytes, p1_from_ghat  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ghat", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", choices=("re0", "re1", "re2", "aoa0", "aoa1"), required=True)
    a = ap.parse_args()
    ghat = Path(a.ghat).resolve(strict=True)
    out = Path(a.out).resolve()
    assert (out.is_relative_to(ROOT) or out.is_relative_to(Path(os.environ.get("RPDE_FINAL_OUTPUT", "/output")))) and not out.exists()
    side = Path(str(ghat) + ".json")
    rec = json.loads(side.read_text())
    split = json.loads(SPLIT.read_text())
    f = next(x for x in split["folds"] if x["id"] == a.fold)
    assert rec["status"] == "OK" and rec["fold"] == a.fold and rec["D"] == 129
    assert rec["n_files"] == len(f["train_files"]) and rec["n_windows"] > 0
    with np.load(ghat) as z:
        assert set(str(v) for v in z["files"]) == set(f["train_files"])
        assert int(z["n_windows"]) == rec["n_windows"] and z["G_hat"].shape == (129, 129)
    p1, meta = p1_from_ghat(ghat, 1.0)
    assert p1.shape == (129, 129) and p1.dtype == np.float32
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(npy_bytes(p1))
    proof = {"status": "OK", "fold": a.fold, "source_card_sha256": rec["card_sha256"],
             "ghat_sha256": sha(ghat), "ghat_sidecar_sha256": sha(side),
             "split_sha256": sha(SPLIT), "out_sha256": sha(out), "n_files": rec["n_files"],
             "n_windows": rec["n_windows"], "fit": meta}
    Path(str(out) + ".json").write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    print("FOLD_P1_OK", a.fold, out, meta["inv_resid"])


if __name__ == "__main__":
    main()
