#!/usr/bin/env python
"""SBPORT calib producer (2026-08-23).  Merges the Track-2 prefit calib (everything the v46 card shipped) with
the Track-1 learned-sband quantile table (sb_edge / sb_lo / sb_hi / sb_q / sb_mult) into the calib.npz the
t2_v49_sbport card ships.  Pure CPU; exists so that reproduce_t2.sh has a producer for calib_sbport.npz instead of
the hand-merge that built the live card.  Key order is the T2 prefit order followed by the five sb_* keys,
which makes np.savez byte-reproducible (np.savez writes fixed 1980 zip timestamps) -- the --expect-content
check below proves the producer reproduces the shipped artefact's DATA bit-for-bit.

DO NOT go back to an md5 of the whole .npz.  Measured 2026-08-24 on V100: all 28 arrays
reproduced bit-for-bit while the file md5 differed, because the shipped calib is a ZIP64 archive
(create/extract_version 45) and numpy 2.1.2 / py3.11 writes a plain ZIP (version 20).  The .npz
container format is a function of the numpy/zipfile VERSION, not of the data, so a container md5
is a guard on the environment and fails on any machine that does not match the one that built the
card -- which is exactly the Decision-Phase situation this chain exists to model.

    python Track2/U-net/make_calib_sbport.py \
        --t2 Track2/U-net/calib_xcbpre_prefit.npz \
        --t1 Track1/U-net/sband/calib_sband_xcb_m0100.npz \
        --out Track2/U-net/calib_sbport.npz [--expect-md5 6359401dec75bceef8aa63c43bfe458f]
"""
import argparse, hashlib, pathlib, sys
import numpy as np

SB_KEYS = ("sb_edge", "sb_lo", "sb_hi", "sb_q", "sb_mult")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t2", required=True, help="Track-2 prefit calib (the calib.npz of the v46 card)")
    ap.add_argument("--t1", required=True, help="Track-1 sband calib carrying sb_edge/sb_lo/sb_hi/sb_q/sb_mult")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-md5", default="", help="informational only -- see --expect-content")
    ap.add_argument("--expect-content", default="",
                    help="refuse unless the written npz has the same keys/shapes/dtypes/bytes as this npz")
    a = ap.parse_args()
    t2 = dict(np.load(a.t2, allow_pickle=True))
    t1 = np.load(a.t1, allow_pickle=True)
    missing = [k for k in SB_KEYS if k not in t1.files]
    assert not missing, f"{a.t1} lacks {missing}"
    assert not any(k in t2 for k in SB_KEYS), "T2 prefit calib already carries sb_* keys -- wrong input"
    e, lo, hi = t1["sb_edge"], t1["sb_lo"], t1["sb_hi"]
    assert e.ndim == 3 and lo.shape == hi.shape and lo.shape[:2] == e.shape[:2] and lo.shape[2] == e.shape[2] + 1, \
        f"sb table shapes inconsistent: edge {e.shape} lo {lo.shape} hi {hi.shape}"
    assert int(t1["sb_q"]) == lo.shape[2], "sb_q disagrees with the table width"
    assert np.all(np.diff(e, axis=-1) >= 0), "sb_edge must be non-decreasing per (lead, ch)"
    for k in SB_KEYS:
        t2[k] = t1[k]
    out = pathlib.Path(a.out)
    np.savez(out, **t2)
    z = np.load(out, allow_pickle=True)
    assert all(k in z.files for k in SB_KEYS)
    got = md5(out)
    print(f"wrote {out} ({out.stat().st_size} B) md5 {got}  keys={len(z.files)}  sb_lo{z['sb_lo'].shape} sb_q={int(z['sb_q'])} sb_mult={float(z['sb_mult'])}")
    if a.expect_content:
        ref = np.load(a.expect_content, allow_pickle=True)
        rk, gk = set(ref.files), set(z.files)
        bad = []
        if rk != gk:
            bad.append(f"key set differs: only-in-live {sorted(rk - gk)} only-in-new {sorted(gk - rk)}")
        for k in sorted(rk & gk):
            x, y = ref[k], z[k]
            if x.shape != y.shape or x.dtype != y.dtype:
                bad.append(f"{k}: {x.shape}/{x.dtype} vs {y.shape}/{y.dtype}")
            elif x.tobytes() != y.tobytes():
                d = np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))
                bad.append(f"{k}: {int((d > 0).sum())}/{d.size} elements differ, maxabs {d.max():.3e}")
        ref_md5 = md5(a.expect_content)
        if bad:
            print(f"CONTENT MISMATCH vs {a.expect_content} (md5 {ref_md5}):", file=sys.stderr)
            for b in bad[:20]:
                print(f"   {b}", file=sys.stderr)
            sys.exit(2)
        print(f"# content identical to {a.expect_content} in all {len(rk)} arrays "
              f"(keys, shapes, dtypes, raw bytes)")
        if got != ref_md5:
            print(f"#   NOTE: file md5 differs ({got} vs {ref_md5}) with IDENTICAL data -- this is the")
            print( "#   .npz ZIP container (zip64 vs plain zip), which numpy picks by version.  Expected")
            print( "#   across environments; it is why this guard checks content, not the container.")
    if a.expect_md5 and got != a.expect_md5:
        print(f"# NOTE: --expect-md5 {a.expect_md5} != {got} (container md5, not checked -- see --expect-content)")
    print("CALIB_SBPORT_OK")


if __name__ == "__main__":
    main()
