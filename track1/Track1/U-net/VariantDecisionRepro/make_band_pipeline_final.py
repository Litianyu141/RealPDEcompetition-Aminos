#!/usr/bin/env python3
"""Emit a source-only twin of the T1 base-band pipeline.

The historical pipeline compares its newly trained band to a submitted band's
calibration.  The Final kit does not supply that historical artifact.  This
copy preserves stages 1-4 (width, OOF cache, head fit, pack) and replaces only
the incumbent comparison with intrinsic shape/finite/provenance checks.  It
also removes shell dependencies on bc and Git from those checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "Track1/U-net/t1_band_pipeline.sh"
START = "# ---------------------------------------------------------------- 0. resolve the live card\n"
NEXT = "# STAGE5_ONLY=1 runs ONLY the acceptance gate"
STAGE5 = "# ---------------------------------------------------------------- 5. PCDEL + content acceptance\n"
OLD_BC = 'BASE090="$OUT/calib_bwid_$(printf \'%03d\' "$(echo "$WIDTH*100/1" | bc)").npz"'
NEW_BC = '''BASE090="$OUT/calib_bwid_$(${PY} -c 'import sys; print(f"{round(float(sys.argv[1])*100):03d}")' "$WIDTH").npz"'''
OLD_GIT = '''if ( cd "$R" && git ls-files --error-unmatch "${_WOULD#$R/}" >/dev/null 2>&1 ); then
  echo "REFUSING: TAG=$TAG would overwrite the TRACKED artifact ${_WOULD#$R/}"
  echo "          pick another TAG (the pipeline never rewrites a shipped band in place)."
  exit 1
fi
'''
NEW_GIT = '''# Final source tree has no Git index or incumbent band; the following path-existence
# guard is the only clobber condition needed for this run's own output.
'''
NEW_STAGE0 = '''# ---------------------------------------------------------------- 0. source-only run
say "0/5  source-only calibration; no incumbent artifact is available or used"
LIVE_CALIB=""
LIVE_SBAND=""
echo "  mode       : $MODE   TAG=$TAG  Q=$Q  WIDTH=$WIDTH"

'''
NEW_STAGE5 = '''# ---------------------------------------------------------------- 5. PCDEL + source-only intrinsic acceptance
say "5/5  drop pc_rl / pc_rh and verify the generated calibration"
"$PY" - "$OUT/packed_m0100.npz" "$OUT/calib_rebuilt.npz" "$Q" "$PROV_CKPT" <<'PYEOF'
import hashlib, sys
import numpy as np
src, dst, qraw, ckpt = sys.argv[1:]
q = int(qraw)
d = {k: v for k, v in np.load(src, allow_pickle=True).items()}
for k in ("pc_rl", "pc_rh"):
    d.pop(k, None)
assert int(np.asarray(d["sb_q"])) == q
for k, shape in {"sb_lo": (20, 2, q), "sb_hi": (20, 2, q),
                 "sb_lo_phys": (20, 2, q), "sb_hi_phys": (20, 2, q),
                 "sb_edge": (20, 2, q-1)}.items():
    a = np.asarray(d[k]); assert a.shape == shape and np.isfinite(a).all(), (k, a.shape)
assert np.all(np.diff(d["sb_edge"], axis=-1) >= 0), "sband edges not monotone"
assert np.all(d["sb_lo"] <= d["sb_hi"]), "sband lower exceeds upper"
for k in ("band_lo", "band_hi", "band_lo_phys", "band_hi_phys"):
    a = np.asarray(d[k]); assert a.shape[:2] == (20, 3) or a.shape[:2] == (20, 2), (k, a.shape)
    assert np.isfinite(a).all(), k
assert np.isfinite(float(np.asarray(d["selftest_mean_nil"]))), "nonfinite head selftest"
md5 = hashlib.md5(open(ckpt, "rb").read()).hexdigest()
assert str(np.asarray(d["prov_ckpt_md5"])) == md5, "band/checkpoint lineage mismatch"
np.savez(dst, **d)
with np.load(dst, allow_pickle=True) as z:
    assert set(z.files) == set(d) and "pc_rl" not in z.files and "pc_rh" not in z.files
print("T1_BAND_PIPELINE_OK  source-only structural, finite, and checkpoint-lineage gates")
PYEOF
echo "  rebuilt -> $OUT/calib_rebuilt.npz"
echo "T1_BAND_PIPELINE_DONE"
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    assert out.is_relative_to(ROOT) and not out.exists()
    before = SOURCE.read_text()
    assert before.count(START) == before.count(NEXT) == before.count(STAGE5) == 1
    assert before.count(OLD_BC) == before.count(OLD_GIT) == 1
    i, j = before.index(START), before.index(NEXT)
    after = before[:i] + NEW_STAGE0 + before[j:]
    after = after.replace(OLD_BC, NEW_BC).replace(OLD_GIT, NEW_GIT)
    after = after[:after.index(STAGE5)] + NEW_STAGE5
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(after)
    meta = {"source": str(SOURCE.relative_to(ROOT)), "source_sha256": hashlib.sha256(before.encode()).hexdigest(),
            "out": str(out.relative_to(ROOT)), "out_sha256": hashlib.sha256(after.encode()).hexdigest(),
            "changes": ["remove historical calibration dependency", "self-contained generated-calib gate",
                        "replace bc with Python", "replace Git clobber test with path-existence test"]}
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print("FINAL_BAND_PIPELINE_EMITTED", meta["out_sha256"])


if __name__ == "__main__":
    main()
