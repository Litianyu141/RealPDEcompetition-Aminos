#!/usr/bin/env python3
"""Assemble current T1/T2 inference cards from frozen source and explicit assets.

The normal path takes a JSON mapping asset name -> path. It never reads an old ZIP.
`--reference-selfcheck` is a diagnostic only: it copies data members from the
submitted ZIP to prove the frozen runtime source and member list are complete.
It cannot certify training reproducibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[3]
VARIANTS = {
    "T1": ROOT / "Track1/U-net/VariantDecisionRepro",
    "T2": ROOT / "Track2/U-net/VariantDecisionRepro",
}
ASSETS = {
    "T1": ("calib.npz", "mean_std_real.pt", "model.pth", "model_1.pth", "sband.pth", "vgain.pth"),
    "T2": ("calib.npz", "ll_p1.npy", "model.pth", "model2.pth", "sband.pth", "vgain.pth"),
}
REFERENCE = {
    "T1": ROOT / "Track1/submissions/t1_v185_s2stack.zip",
    "T2": ROOT / "Track2/submissions/t2_v156g_llrfuse.zip",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=("T1", "T2"))
    ap.add_argument("--assets", help="JSON mapping each learned/data asset member to a generated path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference-selfcheck", action="store_true")
    ap.add_argument("--diagnostic-reference-calibration", action="store_true",
                    help="label a package that still uses old reference head/band/calib as diagnostic only")
    args = ap.parse_args()
    assert args.reference_selfcheck != bool(args.assets), "choose exactly one asset source"
    assert not (args.reference_selfcheck and args.diagnostic_reference_calibration)
    variant = VARIANTS[args.track]
    out = (ROOT / args.out).resolve()
    assert out.is_relative_to(variant / "Logger"), "output must be this Variant's Logger child"
    manifest = json.loads((variant / "config/runtime_manifest.json").read_text())
    assert manifest["track"] == args.track
    code: dict[str, bytes] = {}
    for n, record in manifest["runtime"].items():
        data = (variant / "runtime" / n).read_bytes()
        assert sha(data) == record["sha256"], (n, "frozen runtime drift")
        code[n] = data
    learned: dict[str, bytes] = {}
    source: dict[str, str] = {}
    if args.reference_selfcheck:
        with ZipFile(REFERENCE[args.track]) as z:
            assert sorted(z.namelist()) == sorted(set(code) | set(ASSETS[args.track]))
            learned = {n: z.read(n) for n in ASSETS[args.track]}
        source = {n: "REFERENCE_DIAGNOSTIC_ONLY" for n in learned}
    else:
        assets = json.loads((ROOT / args.assets).read_text())
        assert set(assets) == set(ASSETS[args.track]), (set(assets), set(ASSETS[args.track]))
        for n, raw in assets.items():
            p = (ROOT / raw).resolve()
            assert p.is_file() and p.is_relative_to(ROOT), (n, p)
            if n != "mean_std_real.pt":
                assert p.is_relative_to(VARIANTS["T1"] / "Logger") or p.is_relative_to(VARIANTS["T2"] / "Logger"), \
                    (n, "learned input must come from the new reproduction run", p)
            learned[n] = p.read_bytes()
            source[n] = str(p.relative_to(ROOT))
        borrowed = [n for n, path in source.items() if "adapter_smoke/output" in path]
        assert bool(borrowed) == args.diagnostic_reference_calibration, \
            ("old calibration/reference assets require an explicit diagnostic label", borrowed)
    members = {**code, **learned}
    assert set(members) == set(code) | set(ASSETS[args.track])
    assert not out.exists(), ("refuse overwrite", out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(out, "x", ZIP_DEFLATED) as z:
        for n in sorted(members):
            z.writestr(n, members[n])
    with ZipFile(out) as z:
        assert {n: z.read(n) for n in z.namelist()} == members
    raw_size = sum(len(v) for v in members.values())
    assert raw_size <= 256 * 1024 * 1024, raw_size
    status = "REFERENCE_SELF_CHECK" if args.reference_selfcheck else \
        ("DIAGNOSTIC_REFERENCE_CALIBRATION" if args.diagnostic_reference_calibration else "BUILT_UNVALIDATED")
    result = {"track": args.track, "status": status,
              "zip": str(out.relative_to(ROOT)), "zip_sha256": sha(out.read_bytes()),
              "raw_bytes": raw_size,
              "members": {n: {"sha256": sha(v), "source": source.get(n, "FROZEN_RUNTIME_SOURCE")}
                          for n, v in sorted(members.items())}}
    if args.reference_selfcheck:
        with ZipFile(REFERENCE[args.track]) as z:
            result["all_member_bytes_equal_reference"] = all(z.read(n) == v for n, v in members.items())
        assert result["all_member_bytes_equal_reference"]
    (out.with_suffix(out.suffix + ".json")).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("BUILD_OK", args.track, result["status"], out, result["zip_sha256"], raw_size)


if __name__ == "__main__":
    main()
