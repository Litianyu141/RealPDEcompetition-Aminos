#!/usr/bin/env python3
"""Strip fitted tables from the current-card inference sources for Final training.

The submitted runtime is kept frozen in this repository for audit, but the
source-only container must generate its fitted SPSR/FTL arrays in train().
Only named assignment lines are replaced; all other runtime source stays exact.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCES = {
    "T1": ROOT / "Track1/U-net/VariantDecisionRepro/runtime/submission.py",
    "T2": ROOT / "Track2/U-net/VariantDecisionRepro/runtime/submission.py",
}
TABLES = {"T1": {"_KC_FM": (64,), "_KC_FS": (64,), "_KC_CEN": (3, 64),
                 "_KC_MU": (3, 4, 8), "_KC_LAM": (3, 4, 8),
                 "_SY_GRAW": (3, 20, 4, 8, 2), "_BCD_SR_TABLE": (20, 2)},
          "T2": {"_FP_KE": (70, 32), "_FP_MEAN": (10, 32)}}
COUNTS = {"T1": {"_KC_FM": 1, "_KC_FS": 1, "_KC_CEN": 1,
                 "_KC_MU": 2, "_KC_LAM": 2, "_SY_GRAW": 1, "_BCD_SR_TABLE": 1},
          "T2": {"_FP_KE": 1, "_FP_MEAN": 1}}


def shape(value):
    if not isinstance(value, (list, tuple)):
        assert isinstance(value, (int, float)) and math.isfinite(float(value)), type(value)
        return ()
    assert value, "empty fitted table"
    first = shape(value[0])
    assert all(shape(v) == first for v in value[1:]), "ragged fitted table"
    return (len(value),) + first


def strip(track: str, source: str) -> tuple[str, dict]:
    names = TABLES[track]
    found = {}
    lines = []
    for line in source.splitlines(keepends=True):
        if track == "T1" and line.startswith(("_KC_DBE =", "_KC_DBM =")):
            lines.append("# Dead pre-P1 distance-band constant removed from source-only runtime.\n")
            continue
        name = line.split("=", 1)[0].strip() if "=" in line else ""
        if name not in names:
            lines.append(line)
            continue
        if "torch.tensor(" in line:
            begin = line.index("torch.tensor(") + len("torch.tensor(")
            end = line.index(", dtype=", begin)
            literal = line[begin:end]
            value = ast.literal_eval(literal)
            replacement = line[:begin] + f"__TRAINED_{name}__" + line[end:]
        else:
            value = ast.literal_eval(line.split("=", 1)[1].split("#", 1)[0].strip())
            replacement = f"{name} = __TRAINED_{name}__  # FINAL_TRAIN_GENERATES_{name}\n"
        assert shape(value) == names[name], (track, name, shape(value), names[name])
        digest = hashlib.sha256(repr(value).encode()).hexdigest()
        if name in found:
            assert found[name]["literal_sha256"] == digest, (track, name, "duplicate literals differ")
            found[name]["occurrences"] += 1
        else:
            found[name] = {"shape": list(names[name]), "literal_sha256": digest, "occurrences": 1}
        lines.append(replacement)
    assert set(found) == set(names), (track, found)
    assert {name: rec["occurrences"] for name, rec in found.items()} == COUNTS[track]
    template = "".join(lines)
    for name in names:
        assert template.count(f"__TRAINED_{name}__") == COUNTS[track][name]
    compile(template, "<final-runtime-template>", "exec")
    return template, found


def fill(track: str, template: str, tables: dict[str, list]) -> str:
    assert set(tables) == set(TABLES[track])
    result = template
    for name, expected_shape in TABLES[track].items():
        value = tables[name]
        assert shape(value) == expected_shape, (name, shape(value), expected_shape)
        marker = f"__TRAINED_{name}__"
        assert result.count(marker) == COUNTS[track][name]
        result = result.replace(marker, repr(value))
    assert "__TRAINED_" not in result
    compile(result, "<final-runtime-filled>", "exec")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=("T1", "T2"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    assert out.is_relative_to(ROOT) and not out.exists()
    source = SOURCES[a.track].read_text()
    template, tables = strip(a.track, source)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(template)
    metadata = {"track": a.track, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "template_sha256": hashlib.sha256(template.encode()).hexdigest(), "stripped": tables}
    Path(str(out) + ".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print("RUNTIME_TEMPLATE_OK", a.track, tables, metadata["template_sha256"])


if __name__ == "__main__":
    main()
