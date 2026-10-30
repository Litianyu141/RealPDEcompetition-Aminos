#!/usr/bin/env python3
"""Final mode only: give the Track-2 base chain the Track-1 base chain's draw of an IDENTICAL recipe.

Inside one T2 `solution.train()` the T1 base chain runs first and then the T2 base chain trains, among others, the
RLB128 ship teacher (T2 stage 1) and the RLB_kf teacher fold fleet (T2 stage 2).  Their effective configs equal T1
stages 3a / 2a except for the output path and the variant name (checked 2026-09-29 on the v13 smoke configs), so a
second training is just a second random draw of the same recipe.  This helper refuses unless the two scoped YAMLs are
identical after dropping `checkpoint` and `variant`, then copies the T1 checkpoint(s) to the paths the T2 script
expects, so its own `skip` finds them.  Every reuse is recorded next to the T2 checkpoints (reuse_draws.jsonl).

    python final_reuse_draw.py --t1-cfg T1.yaml --t2-cfg T2.yaml [--folds "0 1 2 3 4 5"]
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import yaml

IGNORED = ("checkpoint", "variant")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1-cfg", required=True)
    ap.add_argument("--t2-cfg", required=True)
    ap.add_argument("--folds", default="", help="space-separated fold ids for a fold fleet (<stem>_f<F>.pth files)")
    a = ap.parse_args()
    c1 = yaml.safe_load(Path(a.t1_cfg).read_text())
    c2 = yaml.safe_load(Path(a.t2_cfg).read_text())
    k1 = {k: v for k, v in c1.items() if k not in IGNORED}
    k2 = {k: v for k, v in c2.items() if k not in IGNORED}
    if k1 != k2:
        diff = sorted(k for k in set(k1) | set(k2) if k1.get(k) != k2.get(k))
        sys.exit("REUSE_REFUSED: configs differ beyond %s: %s" % (IGNORED, diff))
    p1, p2 = Path(c1["checkpoint"]), Path(c2["checkpoint"])
    folds = a.folds.split()
    pairs = ([(p1.with_name(p1.stem + "_f" + f + ".pth"), p2.with_name(p2.stem + "_f" + f + ".pth")) for f in folds]
             if folds else [(p1, p2)])
    log = p2.parent / "reuse_draws.jsonl"
    p2.parent.mkdir(parents=True, exist_ok=True)
    for src, dst in pairs:
        assert src.is_file(), ("T1 draw missing", src)
        if dst.exists():
            assert sha(dst) == sha(src), ("T2 checkpoint exists and differs from the T1 draw", dst)
            continue
        tmp = dst.with_name(dst.name + ".reuse_tmp")
        shutil.copy2(src, tmp)
        tmp.replace(dst)
        rec = {"reused_into": str(dst), "from": str(src), "sha256": sha(dst),
               "t1_cfg": str(a.t1_cfg), "t2_cfg": str(a.t2_cfg), "ignored_keys": list(IGNORED)}
        with log.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        print("REUSE_DRAW", dst.name, "<-", src.name, flush=True)
    print("REUSE_DRAW_OK", len(pairs), flush=True)


if __name__ == "__main__":
    main()
