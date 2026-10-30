"""Check track1/ and track2/ against the SHA-256 lists in their source_manifest.json."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check(track_dir: Path) -> bool:
    manifest = json.loads((track_dir / "source_manifest.json").read_text())
    expected = manifest["source_sha256"]
    ok = True
    for rel, digest in sorted(expected.items()):
        path = track_dir / rel
        if not path.is_file():
            print(f"MISSING  {track_dir.name}/{rel}")
            ok = False
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            print(f"CHANGED  {track_dir.name}/{rel}")
            ok = False
    listed = set(expected) | {"source_manifest.json"}
    for path in sorted(p for p in track_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(track_dir).as_posix()
        if rel not in listed and "__pycache__" not in rel:
            print(f"EXTRA    {track_dir.name}/{rel}")
            ok = False
    print(f"{track_dir.name}: {len(expected)} files, {'OK' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    results = [check(ROOT / name) for name in ("track1", "track2")]
    sys.exit(0 if all(results) else 1)
