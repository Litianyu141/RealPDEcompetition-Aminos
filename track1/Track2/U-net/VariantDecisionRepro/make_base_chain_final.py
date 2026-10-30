#!/usr/bin/env python3
"""Emit a run-scoped T2 v58 calibration-chain copy for Final source runs.

The original historical recipe is retained unchanged.  This copy only makes its
root and output explicit, and lets a parallel T1 chain finish the two sband
artifacts before the dependent merge.  It is a producer of the inherited T2
head/band, not a reproduction claim for v156g by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "Track2/U-net/reproduce_t2.sh"
OLD_ROOT = 'R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)'
NEW_ROOT = 'R=${RPDE_ROOT:?set RPDE_ROOT to the source-tree root}'
OLD_OUT = 'OUT=$D/repro'
NEW_OUT = 'OUT=${RPDE_T2_BASE_OUT:?set RPDE_T2_BASE_OUT to a fresh output directory}'
OLD_WAIT = '''  [ -e "$SHIP_SBAND" ] || { echo "MISSING $SHIP_SBAND (the live sband head)"; exit 1; }
  [ -e "$T1_SBAND_CALIB" ] || { echo "MISSING $T1_SBAND_CALIB"; exit 1; }'''
NEW_WAIT = '''  if [ "${RPDE_WAIT_T1_BAND:-0}" = "1" ]; then
    _wait_start=$SECONDS
    until [ -e "$SHIP_SBAND" ] && [ -e "$T1_SBAND_CALIB" ]; do
      [ $((SECONDS-_wait_start)) -lt 86400 ] || { echo "T1 BAND WAIT TIMEOUT"; exit 1; }
      echo "  waiting for this run's T1 band: $SHIP_SBAND / $T1_SBAND_CALIB"
      sleep 60
    done
  fi
  [ -e "$SHIP_SBAND" ] || { echo "MISSING $SHIP_SBAND (the live sband head)"; exit 1; }
  [ -e "$T1_SBAND_CALIB" ] || { echo "MISSING $T1_SBAND_CALIB"; exit 1; }'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    assert out.is_relative_to(ROOT / "Track2/U-net/VariantDecisionRepro/Logger") and not out.exists()
    before = SOURCE.read_text()
    for old in (OLD_ROOT, OLD_OUT, OLD_WAIT):
        assert before.count(old) == 1, old[:60]
    after = before.replace(OLD_ROOT, NEW_ROOT).replace(OLD_OUT, NEW_OUT).replace(OLD_WAIT, NEW_WAIT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(after)
    doc = {"source": str(SOURCE.relative_to(ROOT)), "source_sha256": hashlib.sha256(before.encode()).hexdigest(),
           "out": str(out.relative_to(ROOT)), "out_sha256": hashlib.sha256(after.encode()).hexdigest(),
           "changes": ["explicit RPDE_ROOT", "explicit run-scoped OUT", "optional wait for T1 band dependency"]}
    Path(str(out) + ".json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print("FINAL_T2_BASE_CHAIN_EMITTED", doc["out_sha256"])


if __name__ == "__main__":
    main()
