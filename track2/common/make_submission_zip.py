#!/usr/bin/env python3
"""Deterministic card zipper (2026-09-07).

    python common/make_submission_zip.py Track1/submissions/<name> [--out Track1/submissions/<name>.zip]

Zips the built card directory so that the SAME directory content gives the SAME bytes on every box (Mac and
GPU hosts): entries sorted, ZIP_STORED (no zlib -> no compressor-version drift), fixed 1980-01-01 timestamps, fixed
0644 attributes, no extra fields.  Prints `ZIP_OK <path> <md5> <MB>`.  Skips __pycache__ and dotfiles.
The upload zip is made on the Mac from the synced directory; a server only needs a zip when a chain anchors on
its md5, and it regenerates the identical bytes with this tool instead of keeping copies around.
"""
import argparse, hashlib, sys, zipfile
from pathlib import Path


def payload_of(src_dir):
    """The whitelist the build scripts zipped: verify_report.json's `payload` (T1) / `PAYLOAD` (T2) list if present,
    else every regular file except the report itself, __pycache__ and dotfiles."""
    rep = src_dir / "verify_report.json"
    if rep.exists():
        import json
        d = json.loads(rep.read_text())
        for k in ("payload", "PAYLOAD", "files", "archive"):
            if isinstance(d.get(k), list) and d[k]:
                # T2's builder lists the card DIRECTORY after importing submission.py in place, so its
                # `files` can carry `__pycache__` / `*.pyc` (t2_v57_hybin, 2026-09-09).  Caches are never
                # payload -- the builder's own zip omits them -- so drop them here, deterministically.
                return [str(x) for x in d[k]
                        if str(x) != "__pycache__" and "__pycache__" not in str(x).split("/")
                        and not str(x).endswith(".pyc") and str(x) != "verify_report.json"]
    return sorted(p.relative_to(src_dir).as_posix() for p in src_dir.rglob("*") if p.is_file() and p.name != "verify_report.json"
                  and "__pycache__" not in p.parts and not any(part.startswith(".") for part in p.relative_to(src_dir).parts))


def make_zip(src_dir, zip_path, names=None):
    src_dir, zip_path = Path(src_dir), Path(zip_path)
    assert (src_dir / "submission.py").exists(), f"{src_dir} has no submission.py -- not a card directory"
    names = sorted(names) if names else payload_of(src_dir)
    files = [src_dir / n for n in names]
    for f in files:
        assert f.is_file(), f"missing payload file {f}"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for p in files:
            zi = zipfile.ZipInfo(p.relative_to(src_dir).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = 0o644 << 16
            zi.create_system = 3
            z.writestr(zi, p.read_bytes())
    md5 = hashlib.md5(zip_path.read_bytes()).hexdigest()
    return md5, zip_path.stat().st_size / 1e6, len(files)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("card_dir"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(str(a.card_dir).rstrip("/"))
    out = Path(a.out) if a.out else d.parent / (d.name + ".zip")
    md5, mb, n = make_zip(d, out)
    print(f"ZIP_OK {out} {md5} {mb:.2f}MB {n} files")
