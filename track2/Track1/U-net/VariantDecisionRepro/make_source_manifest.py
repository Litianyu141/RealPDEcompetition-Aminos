#!/usr/bin/env python3
"""Seal a curated Final source tree without shipping learned weights or data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FORBIDDEN_PARTS = {".git", "Logger", "_tmp", "data", "submissions", "checkpoints", "Ref_code", "Ref_papers"}
FORBIDDEN_SUFFIXES = {".zip", ".pth", ".pt", ".npz", ".npy", ".h5", ".hdf5", ".ckpt"}
SOURCE_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".json", ".toml", ".md", ".txt", ".c", ".cpp", ".h", ".cu", ".cuh"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root).resolve(strict=True)
    out = Path(a.out).resolve()
    assert out.is_relative_to(root) and not out.exists()
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_dir(): continue
        rel = p.relative_to(root)
        assert not (set(rel.parts) & FORBIDDEN_PARTS), ("forbidden source-tree directory", rel)
        assert p.suffix not in FORBIDDEN_SUFFIXES, ("learned/data file in source package", rel)
        assert p.suffix in SOURCE_SUFFIXES or p.name in {"Dockerfile", "LICENSE"}, ("not a source file", rel)
        assert not p.is_symlink(), ("source package contains symlink", rel)
        files[rel.as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    assert "solution.py" in files and "Dockerfile" in files and "README.md" in files and "METHOD.md" in files
    doc = {"schema": 1, "source_sha256": files, "source_only": True}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print("SOURCE_MANIFEST_OK", len(files), out)


if __name__ == "__main__":
    main()
