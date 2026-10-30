#!/usr/bin/env python3
"""Replace a clean clone's git-ls-files preflight in a source-only Final image."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--path", required=True)
    a = ap.parse_args()
    root = Path(a.root).resolve(strict=True)
    path = Path(a.path).resolve(strict=True)
    rel = path.relative_to(root).as_posix()
    expected = json.loads(Path(a.manifest).read_text())["source_sha256"][rel]
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    assert got == expected, (rel, got, expected)
    print("FINAL_SOURCE_OK", rel, got)


if __name__ == "__main__":
    main()
