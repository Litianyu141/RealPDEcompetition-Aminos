#!/usr/bin/env python3
"""L2 T1-SBAND2: restamp the provenance of a card whose sband.pth + calib.npz:sb_edge were replaced by sb2_foldcard.py.
The replaced calib still carries the donor head's prov_fit_stream / prov_producer / prov_when / selftest_mean_nil ("a copied
stale stamp is worse than no stamp", t1_sband_pack.py).  Rewrites exactly those four keys (+ sb2_* source keys); prov_ckpt_* /
prov_extra_* (the members, unchanged) are kept.  selftest_mean_nil is recomputed the t1_sband_pack.py way (mean nil of the
served table over the first 8 windows of the head's own training cache, strict_selftest_inputs).
    python Track1/U-net/VariantSPSBand2/sb2_restamp.py --card <zip> --run q75s5_15/ship_s0 --out <zip>
"""
import argparse, hashlib, io, json, sys, time, zipfile
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "Track1/U-net"))
LT, C2, H, W = 20, 2, 32, 64
SIG = 0.0563870259


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--card", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from sband_head import build_from_blob, sband_features
    card = Path(a.card); card = card if card.is_absolute() else REPO / card
    with zipfile.ZipFile(card) as z:
        cz = np.load(io.BytesIO(z.read("calib.npz")), allow_pickle=True); cal = {k: cz[k] for k in cz.files}
        blob = torch.load(io.BytesIO(z.read("sband.pth")), map_location="cpu", weights_only=False)
        mstd = torch.load(io.BytesIO(z.read("mean_std_real.pt")), weights_only=False)
    std = np.asarray(mstd[3], np.float64).ravel()[:2]
    hb = torch.load(REPO / "_tmp/sband2/heads" / a.run / "head.pth", map_location="cpu", weights_only=False)
    cache = REPO / hb["cache"]
    net = build_from_blob(blob).eval()
    X = np.array(np.load(cache / "Xp.npy", mmap_mode="r")[:8])
    with torch.no_grad():
        S8 = net(sband_features(torch.from_numpy(X))).reshape(-1, LT, C2, H, W).permute(0, 1, 3, 4, 2).numpy()
    E = cal["sb_edge"].astype(np.float32); LO = cal["sb_lo"].astype(np.float64); HI = cal["sb_hi"].astype(np.float64)
    tot = 0.0
    for t in range(LT):
        for c in range(C2):
            b = np.searchsorted(E[t, c], S8[:, t, :, :, c], side="left")
            tot += float(np.sum((HI[t, c, b] - LO[t, c, b]) * std[c])) / SIG
    nil = tot / (8 * LT * H * W * C2)
    old = {k: str(cal.get(k, "")) for k in ("prov_fit_stream", "prov_producer", "prov_when", "selftest_mean_nil")}
    cal["prov_fit_stream"] = np.str_("sband head sb2:%s | %s | pinball q%.2f on mean log|r| | cache %s (all81 OOF of the Re-outer-fold "
                                     "teachers, stride 5) | sb_edge = the head's quantiles on that cache | sb_lo / sb_hi = the base "
                                     "card's table (rank-swap)" % (a.run, hb["fleets"], hb["quantile"], hb["cache"]))
    cal["prov_producer"] = np.str_("Track1/U-net/VariantSPSBand2/sb2_foldcard.py --run %s --base <v160k> ; sb2_restamp.py" % a.run)
    cal["prov_when"] = np.str_(time.strftime("%Y-%m-%dT%H:%M:%S"))
    cal["selftest_mean_nil"] = np.float64(nil)
    cal["sb2_table_source"] = np.str_("base card sb_lo / sb_hi unchanged (rank-swap)")
    buf = io.BytesIO(); np.savez(buf, **cal)
    out = Path(a.out); out = out if out.is_absolute() else REPO / out
    tmp = out.with_suffix(".tmp.zip")
    with zipfile.ZipFile(card) as zi, zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zo:
        for info in zi.infolist():
            zo.writestr(info, buf.getvalue() if info.filename == "calib.npz" else zi.read(info.filename))
    tmp.replace(out)
    meta = dict(tool="Track1/U-net/VariantSPSBand2/sb2_restamp.py", card_in=str(card), card_out=str(out),
                sha256_out=hashlib.sha256(out.read_bytes()).hexdigest(), md5_out=hashlib.md5(out.read_bytes()).hexdigest(),
                old_stamp=old, new_selftest_mean_nil=nil)
    Path(str(out) + ".restamp.json").write_text(json.dumps(meta, indent=1))
    print("RESTAMP_OK", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
