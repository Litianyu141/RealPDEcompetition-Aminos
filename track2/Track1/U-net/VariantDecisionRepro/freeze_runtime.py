#!/usr/bin/env python3
"""Freeze only executable Python sources of the two actual submitted cards.

No model/head/band/normalization arrays are copied. This yields a stable runtime
template whose code bytes can be checked without any historical submission ZIP.
Learned literals embedded in these sources still need explicit producer audits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
JOBS = {
    "T1": ("Track1/submissions/t1_v185_s2stack.zip", "b44fda74b13a17ed4b94a8f587904e67",
           "Track1/U-net/VariantDecisionRepro", ("submission.py", "unet_model.py", "vgain_head.py", "sband_head.py")),
    "T2": ("Track2/submissions/t2_v156g_llrfuse.zip", "a3aa416379b4348cc9b1e309dcbc10cb",
           "Track2/U-net/VariantDecisionRepro", ("submission.py", "unet_model.py", "ttt_model.py", "vgain_head.py", "sband_head.py")),
}


def digest(data: bytes, kind: str = "sha256") -> str:
    return getattr(hashlib, kind)(data).hexdigest()


def run(which: str, check: bool) -> None:
    zip_name, expected, dst, names = JOBS[which]
    raw = (ROOT / zip_name).read_bytes()
    assert digest(raw, "md5") == expected, (which, "wrong local ZIP identity")
    dest = ROOT / dst
    with ZipFile(ROOT / zip_name) as z:
        assert all(n in z.namelist() for n in names), (which, z.namelist())
        files = {n: z.read(n) for n in names}
    manifest = {"track": which, "source_zip": zip_name, "source_zip_md5": expected,
                "source_zip_sha256": digest(raw),
                "note": "Executable source only. Embedded fitted constants and all learned binary components require separate producers.",
                "runtime": {n: {"sha256": digest(data), "bytes": len(data)} for n, data in files.items()}}
    for name, data in files.items():
        target = dest / "runtime" / name
        if check:
            assert target.is_file() and target.read_bytes() == data, (which, name, "runtime drift")
        elif target.exists():
            assert target.read_bytes() == data, (which, name, "refuse overwrite")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    p = dest / "config/runtime_manifest.json"
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if check:
        assert p.is_file() and p.read_text() == payload, (which, "manifest drift")
    elif p.exists():
        assert p.read_text() == payload, (which, "refuse manifest overwrite")
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(payload)
    print("RUNTIME_" + ("CHECK" if check else "FREEZE") + "_OK", which, len(files), digest(payload.encode()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=("T1", "T2", "both"), default="both")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    for track in ("T1", "T2") if args.track == "both" else (args.track,):
        run(track, args.check)


if __name__ == "__main__":
    main()
