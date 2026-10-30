#!/usr/bin/env python3
"""Exercise the Final data gate against both released and extraneous file paths."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "common"))
from final_source_gate import verify_final_source  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--good-glob", required=True)
    ap.add_argument("--bad-glob", required=True)
    args = ap.parse_args()
    assert verify_final_source(args.good_glob, args.data_root)
    assert not verify_final_source(args.good_glob, None)
    rejected = False
    try:
        verify_final_source(args.bad_glob, args.data_root)
    except AssertionError:
        rejected = True
    assert rejected, "a source outside Final mount was accepted"
    rejected_empty = False
    try:
        verify_final_source(args.data_root + "/train_real/NO_MATCH_*.h5", args.data_root)
    except AssertionError:
        rejected_empty = True
    assert rejected_empty
    print("FINAL_SOURCE_GATE_SAFETY_OK")


if __name__ == "__main__":
    main()
