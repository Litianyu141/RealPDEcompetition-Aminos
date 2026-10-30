"""Repository-relative paths for configs and split manifests.

Tracked YAML configs and the split manifests under common/local_benchmark/ store every path relative to
the repository root (``data/RealPDE_competition_dataset/...``, ``checkpoints/t1_u-net/...``,
``Track2/realpde_t2_starting_kit_v4/...``), so one file works unchanged on every machine that holds a
checkout.  ``resolve_paths`` turns them into absolute paths under the checkout the calling script lives in;
absolute paths and non-path strings pass through untouched.  A comma-joined list of paths (a
multi-teacher ``distill_ckpt``) is resolved element by element.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# top-level entries a repo-relative path may start with
REPO_PREFIXES = ("data/", "checkpoints/", "Track1/", "Track2/", "common/", "docs/", "_tmp/")


def is_repo_relative(s: str) -> bool:
    return isinstance(s, str) and s.startswith(REPO_PREFIXES)


def resolve_path(s, root=REPO_ROOT):
    """One string: repo-relative path(s) -> absolute under ``root``; anything else unchanged."""
    if not isinstance(s, str):
        return s
    parts = s.split(",")
    if len(parts) > 1 and all(is_repo_relative(p) for p in parts):
        return ",".join(str(Path(root) / p) for p in parts)
    return str(Path(root) / s) if is_repo_relative(s) else s


def resolve_paths(obj, root=REPO_ROOT):
    """Walk a loaded config/manifest (dict/list/str) and resolve every repo-relative path string."""
    if isinstance(obj, dict):
        return {k: resolve_paths(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_paths(v, root) for v in obj]
    return resolve_path(obj, root)


def relativize_paths(obj, root=REPO_ROOT):
    """Inverse of ``resolve_paths`` for writers: absolute paths under ``root`` -> repo-relative strings."""
    if isinstance(obj, dict):
        return {k: relativize_paths(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [relativize_paths(v, root) for v in obj]
    if isinstance(obj, str):
        pre = str(Path(root)).rstrip("/") + "/"
        return ",".join(p[len(pre):] if p.startswith(pre) else p for p in obj.split(","))
    return obj
