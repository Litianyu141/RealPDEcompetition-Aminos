#!/usr/bin/env python3
"""Materialize current-method inference cards from explicit newly produced assets.

Fold cards may use zero SPSR/FTL priors for capture.  Final cards require the
fitted-table JSONs.  The frozen submitted ZIP is never opened or used as a
source.  A materialized card remains UNVALIDATED until its real-interface and
full score checks pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import make_runtime_template as RT

ROOT = Path(__file__).resolve().parents[3]
VARIANT = {"T1": ROOT / "Track1/U-net/VariantDecisionRepro",
           "T2": ROOT / "Track2/U-net/VariantDecisionRepro"}
ASSETS = {"T1": ("calib.npz", "mean_std_real.pt", "model.pth", "model_1.pth", "sband.pth", "vgain.pth"),
          "T2": ("calib.npz", "ll_p1.npy", "model.pth", "model2.pth", "sband.pth", "vgain.pth")}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def runtime_template(track: str) -> str:
    source = (VARIANT[track] / "runtime/submission.py").read_text()
    if "__TRAINED_" in source:
        for name in RT.TABLES[track]:
            assert source.count(f"__TRAINED_{name}__") == RT.COUNTS[track][name]
        return source
    template, _ = RT.strip(track, source)
    return template


def load_tables(track: str, path: Path | None) -> dict[str, list]:
    if path is None:
        def filled(shape, value):
            return value if not shape else [filled(shape[1:], value) for _ in range(shape[0])]
        tables = {name: filled(shape, 0.0) for name, shape in RT.TABLES[track].items()}
        if track == "T1":
            tables["_KC_FS"] = filled(RT.TABLES[track]["_KC_FS"], 1.0)
            tables["_SY_GRAW"] = filled(RT.TABLES[track]["_SY_GRAW"], 1.0)
        return tables
    doc = json.loads(path.read_text())
    assert doc["status"] == "SOURCE_FIT", path
    return doc["runtime_tables"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=("T1", "T2"), required=True)
    ap.add_argument("--assets", required=True, help="JSON member-name to newly generated asset path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--table", help="source-fit SPSR or FTL JSON; absent only for fold capture")
    ap.add_argument("--capture-card", action="store_true")
    ap.add_argument("--capture-pre-f3", action="store_true",
                    help="T1 OOF fitting card only: identity temporal filter before fitting the LK table")
    a = ap.parse_args()
    assert a.capture_card or a.table is not None, "final cards require generated tables"
    assert not a.capture_pre_f3 or (a.capture_card and a.track == "T1")
    variant = VARIANT[a.track]
    out = Path(a.out).resolve()
    assert (out.is_relative_to(variant / "Logger") or out.is_relative_to(Path(os.environ.get("RPDE_FINAL_OUTPUT", "/output")))) and not out.exists(), out
    table_path = Path(a.table).resolve(strict=True) if a.table else None
    if table_path is not None:
        assert (table_path.is_relative_to(ROOT) or table_path.is_relative_to(Path(os.environ.get("RPDE_FINAL_OUTPUT", "/output")))) \
            and "adapter_smoke" not in str(table_path)
    assets = json.loads(Path(a.assets).read_text())
    assert set(assets) == set(ASSETS[a.track]), (set(assets), ASSETS[a.track])
    members = {}
    proof = {}
    for name, raw in assets.items():
        p = Path(raw).resolve(strict=True)
        assert p.is_file() and (p.is_relative_to(ROOT) or p.is_relative_to(Path(os.environ.get("RPDE_FINAL_OUTPUT", "/output")))), p
        assert not any(s in p.parts for s in ("submissions", "adapter_smoke", "fresh_card")), p
        data = p.read_bytes()
        assert data, p
        members[name] = data
        proof[name] = {"path": str(p), "sha256": hashlib.sha256(data).hexdigest()}
    template = runtime_template(a.track)
    text = RT.fill(a.track, template, load_tables(a.track, table_path))
    if a.capture_pre_f3:
        rows = []
        for t in range(20):
            row = [0.0] * 21
            row[t + 1] = 1.0
            rows.append(row)
        lines = text.splitlines(keepends=True)
        found = [i for i, line in enumerate(lines) if line.startswith("_SY_MF = torch.tensor(")]
        assert len(found) == 1
        lines[found[0]] = f"_SY_MF = torch.tensor({rows!r}, dtype=torch.float32, device=_DEVICE)  # OOF pre-F3 capture\n"
        text = "".join(lines)
        compile(text, "<pre-F3-fold-card>", "exec")
    if a.track == "T1" and a.capture_card:
        old = "_BCD_SR_ENABLE = True"
        assert text.count(old) == 1
        text = text.replace(old, "_BCD_SR_ENABLE = False")
    members["submission.py"] = text.encode()
    for p in sorted((variant / "runtime").glob("*.py")):
        if p.name != "submission.py":
            members[p.name] = p.read_bytes()
    assert set(members) == set(ASSETS[a.track]) | {p.name for p in (variant / "runtime").glob("*.py")}
    assert sum(len(v) for v in members.values()) <= 256 * 1024 * 1024
    out.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(out, "x", ZIP_DEFLATED) as z:
        for name in sorted(members):
            z.writestr(name, members[name])
    with ZipFile(out) as z:
        assert set(z.namelist()) == set(members) and all(z.read(n) == v for n, v in members.items())
    doc = {"track": a.track, "status": "CAPTURE_CARD_UNVALIDATED" if a.capture_card else "SOURCE_CARD_UNVALIDATED",
           "capture_pre_f3": bool(a.capture_pre_f3),
           "table": str(table_path) if table_path else None,
           "table_sha256": sha(table_path) if table_path else None,
           "zip": str(out), "zip_sha256": sha(out), "assets": proof,
           "member_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in members.items()}}
    Path(str(out) + ".json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print("SOURCE_CARD_BUILT_UNVALIDATED", a.track, out, doc["zip_sha256"])


if __name__ == "__main__":
    main()
