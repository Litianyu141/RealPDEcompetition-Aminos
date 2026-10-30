#!/usr/bin/env python3
"""Create a deterministic, source-only Final ZIP matching one Docker context."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_STORED

ROOT = Path(__file__).resolve().parents[3]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=("T1", "T2"), required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    context = Path(a.context).resolve(strict=True)
    variant = ROOT / ("Track1" if a.track == "T1" else "Track2") / "U-net/VariantDecisionRepro"
    assert context.is_relative_to(variant / "Logger")
    out = Path(a.out).resolve()
    assert out.is_relative_to(variant / "Logger") and not out.exists()
    manifest = context / "source_manifest.json"
    doc = json.loads(manifest.read_text())
    assert doc["source_only"] is True
    expected = set(doc["source_sha256"]) | {"source_manifest.json"}
    actual = {p.relative_to(context).as_posix() for p in context.rglob("*") if p.is_file()}
    assert actual == expected, {"unexpected": sorted(actual - expected), "missing": sorted(expected - actual)}
    for name, digest in doc["source_sha256"].items():
        assert sha(context / name) == digest, (name, "source changed after manifest")
    assert {"Dockerfile", "solution.py", "README.md", "METHOD.md"} <= expected
    out.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(out, "x", ZIP_STORED, allowZip64=True) as z:
        for name in sorted(expected):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_STORED
            info.external_attr = 0o644 << 16
            info.create_system = 3
            z.writestr(info, (context / name).read_bytes())
    with ZipFile(out) as z:
        assert set(z.namelist()) == expected and z.testzip() is None
    rec = {"track": a.track, "status": "SOURCE_ONLY_UNVALIDATED_FULL_TRAIN",
           "source_zip": str(out), "zip_md5": hashlib.md5(out.read_bytes()).hexdigest(),
           "zip_sha256": sha(out), "manifest_sha256": sha(manifest),
           "n_source_files": len(expected), "size_bytes": out.stat().st_size}
    Path(str(out) + ".json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print("SOURCE_ZIP_BUILT_UNVALIDATED", a.track, rec["zip_md5"], rec["zip_sha256"], len(expected))


if __name__ == "__main__":
    main()
