#!/usr/bin/env python3
"""Produce a Final-container twin of reproduce_t1.sh with two path-only guards.

The original active server chain stays untouched. Changes are: the repository
root is supplied explicitly, and the tracked-source preflight checks a shipped
SHA manifest instead of Git metadata. Every training/eval/build statement is
copied verbatim. The emitted script itself belongs in the source package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "Track1/U-net/reproduce_t1.sh"
OLD_ROOT = 'R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)'
NEW_ROOT = 'R=${RPDE_ROOT:?set RPDE_ROOT to /workspace in the Final container}'
OLD_GIT = '''  if ! ( cd "$R" && git ls-files --error-unmatch "${_p#$R/}" >/dev/null 2>&1 ); then
    echo "REPRO_GUARD: NOT TRACKED IN GIT  ${_p#$R/}"; _guard_fail=1
  fi'''
NEW_GIT = '''  if [ -n "${RPDE_FINAL_SOURCE_MANIFEST:-}" ]; then
    "$PY" "$R/Track1/U-net/VariantDecisionRepro/verify_source_manifest.py" \\
      --root "$R" --manifest "$RPDE_FINAL_SOURCE_MANIFEST" --path "$_p" \\
      || { echo "REPRO_GUARD: SOURCE MANIFEST MISMATCH  ${_p#$R/}"; _guard_fail=1; }
  elif ! ( cd "$R" && git ls-files --error-unmatch "${_p#$R/}" >/dev/null 2>&1 ); then
    echo "REPRO_GUARD: NOT TRACKED IN GIT  ${_p#$R/}"; _guard_fail=1
  fi'''
OLD_BAND = 'bash "$D/t1_band_pipeline.sh"'
NEW_BAND = 'bash "${RPDE_FINAL_BAND_SCRIPT:-$D/t1_band_pipeline.sh}"'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    assert out.is_relative_to(ROOT) and not out.exists()
    before = SOURCE.read_text()
    assert before.count(OLD_ROOT) == before.count(OLD_GIT) == before.count(OLD_BAND) == 1
    after = before.replace(OLD_ROOT, NEW_ROOT).replace(OLD_GIT, NEW_GIT).replace(OLD_BAND, NEW_BAND)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(after)
    info = {"source": str(SOURCE.relative_to(ROOT)), "source_sha256": hashlib.sha256(before.encode()).hexdigest(),
            "out": str(out.relative_to(ROOT)), "out_sha256": hashlib.sha256(after.encode()).hexdigest(),
            "changes": ["explicit RPDE_ROOT", "SHA-manifest alternative to git-ls-files",
                        "optional source-only band script path"]}
    Path(str(out) + ".json").write_text(json.dumps(info, indent=2) + "\n")
    print("FINAL_BASE_CHAIN_EMITTED", info["out_sha256"])


if __name__ == "__main__":
    main()
