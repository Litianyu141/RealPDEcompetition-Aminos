#!/usr/bin/env python3
"""Fit the T2 LLR1 preconditioner from an all81 Gram produced by ll_center_probe.

The supplied Gram must be regenerated from a card containing the model trained in
this reproduction run. `--expect-md5` is only for the historical-reference probe;
new stochastic training is checked numerically, not forced to old weight bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "common"))
from ll_ridge_v155_card import npy_bytes, p1_from_ghat  # noqa: E402


def hashfile(path: Path, kind: str) -> str:
    return getattr(hashlib, kind)(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ghat", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-md5")
    a = ap.parse_args()
    ghat = (ROOT / a.ghat).resolve()
    out = (ROOT / a.out).resolve()
    variant = Path(__file__).resolve().parent
    assert (ghat.is_relative_to(variant / "Logger") or ghat.is_relative_to(Path(os.environ.get("RPDE_FINAL_OUTPUT", "/output"))))
    assert (out.is_relative_to(variant / "Logger") or out.is_relative_to(Path(os.environ.get("RPDE_FINAL_OUTPUT", "/output"))))
    assert ghat.is_file() and not out.exists(), (ghat, out)
    side = Path(str(ghat) + ".json")
    source = json.loads(side.read_text())
    assert source["status"] == "OK" and source["fold"] == "all81"
    assert source["n_files"] == 81 and source["n_windows"] == 3341 and source["D"] == 129, source
    p1, meta = p1_from_ghat(ghat, 1.0)
    assert p1.shape == (129, 129) and str(p1.dtype) == "float32"
    raw = npy_bytes(p1)
    md5 = hashlib.md5(raw).hexdigest()
    if a.expect_md5:
        assert md5 == a.expect_md5, (md5, a.expect_md5)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    record = {"status": "OK", "ghat": str(ghat.relative_to(ROOT)) if ghat.is_relative_to(ROOT) else str(ghat), "ghat_sha256": hashfile(ghat, "sha256"),
              "ghat_sidecar_sha256": hashfile(side, "sha256"), "source_card_sha256": source["card_sha256"],
              "out": str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out), "out_md5": md5, "out_sha256": hashfile(out, "sha256"),
              "fit": meta, "historical_reference_match": bool(a.expect_md5 and md5 == a.expect_md5)}
    Path(str(out) + ".json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print("P1_OK", md5, "inverse_resid", meta["inv_resid"], "windows", source["n_windows"])


if __name__ == "__main__":
    main()
