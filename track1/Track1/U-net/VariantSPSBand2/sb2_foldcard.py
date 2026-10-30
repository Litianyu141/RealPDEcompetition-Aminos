#!/usr/bin/env python3
"""L2 T1-SBAND2: build a REAL fold card for a trained head under the rank-swap table -- the v160k-instrument fold card
(_tmp/tkeriders/ksm/<fold>/t1.zip) with exactly two members replaced:
  sband.pth   the new head (card sband_head.build_from_blob format; same class, same dilations 1,2,4,2,1 -> strict load)
  calib.npz   sb_edge <- the head's per-(lead, ch) quantile edges on its own inner-OOF training cache (what sb2_cardtab.py
              used); sb_lo / sb_hi / sb_*_phys and every other key untouched
Every other file is copied byte-for-byte.  The card is then scored by common/band_width_fivefold.py (the instrument that
produced eval_KSM_MXGEGPTK) so the offline read can be checked against the real interface.
    CUDA_VISIBLE_DEVICES=G python Track1/U-net/VariantSPSBand2/sb2_foldcard.py --run q75s5_15/re0_s2 --gpu G --out <zip>
"""
import argparse, hashlib, io, json, os, sys, zipfile
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(REPO / "Track1/U-net"))
LT, C2, H, W = 20, 2, 32, 64


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--gpu", default="0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="", help="base card zip (default: the v160k-instrument fold card of the run's fold)")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    import torch, torch.nn as nn
    from sband_head import sband_features, SBandHead
    import sb2_head as SH
    d = REPO / "_tmp/sband2/heads" / a.run
    f = d.name.split("_s")[0]
    blob = torch.load(d / "head.pth", map_location="cpu", weights_only=False)
    assert blob["dil"] == [1, 2, 4, 2, 1] and blob["feat"] == "v1" and blob["n_coord"] == 0, "card-compatible v1 heads only"
    # card-format blob, verified by a strict load into the card's own class
    cblob = {"state_dict": blob["state_dict"], "n_feat": blob["n_feat"], "width": blob["width"], "n_layers": blob["n_layers"],
             "n_out": blob["n_out"], "n_coord": 0, "feat": "v1", "fit_kind": "sb2 %s | %s | q%.2f" % (a.run, blob["fleets"], blob["quantile"])}
    ref = SBandHead(cblob["n_feat"], cblob["width"], cblob["n_layers"], cblob["n_out"]); ref.load_state_dict(cblob["state_dict"], strict=True)
    ref = ref.cuda().eval()
    mine = SH.build_head(nn, torch, blob["n_feat"], blob["width"], blob["n_layers"], blob["dil"]).cuda().eval()
    mine.load_state_dict(blob["state_dict"])
    cache = REPO / blob["cache"]
    X = np.load(cache / "Xp.npy", mmap_mode="r"); SCm = np.load(cache / "SC.npy", mmap_mode="r")
    S = np.empty((X.shape[0], LT, H, W, C2), np.float32); dmax = 0.0
    with torch.no_grad():
        for i in range(0, X.shape[0], 32):
            xb = torch.from_numpy(np.array(X[i:i + 32])).cuda()
            y = ref(sband_features(xb)); y2 = mine(sband_features(xb))
            dmax = max(dmax, float((y - y2).abs().max()))
            S[i:i + 32] = y.reshape(-1, LT, C2, H, W).permute(0, 1, 3, 4, 2).cpu().numpy()
    src = Path(a.base) if a.base else REPO / ("_tmp/tkeriders/ksm/%s/t1.zip" % f)
    src = src if src.is_absolute() else REPO / src
    with zipfile.ZipFile(src) as z:
        cz = np.load(io.BytesIO(z.read("calib.npz")), allow_pickle=True)
        cal = {k: cz[k] for k in cz.files}
    Q = int(cal["sb_lo"].shape[-1])
    E = np.zeros((LT, C2, Q - 1), np.float32)
    for t in range(LT):
        for c in range(C2):
            s = S[:, t, :, :, c][np.asarray(SCm[:, t, :, :, c])].astype(np.float64)
            E[t, c] = np.maximum.accumulate(np.quantile(s, np.linspace(0, 1, Q + 1)[1:-1])).astype(np.float32)
    cal["sb_edge"] = E
    cal["sb_head"] = np.str_("sb2:" + a.run)
    buf = io.BytesIO(); np.savez(buf, **cal); calib_bytes = buf.getvalue()
    buf = io.BytesIO(); torch.save(cblob, buf); sband_bytes = buf.getvalue()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zi, zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zo:
        for info in zi.infolist():
            data = {"calib.npz": calib_bytes, "sband.pth": sband_bytes}.get(info.filename, None)
            zo.writestr(info, zi.read(info.filename) if data is None else data)
    base_label = str(src.relative_to(REPO)) if src.is_relative_to(REPO) else str(src)
    meta = dict(tool="Track1/U-net/VariantSPSBand2/sb2_foldcard.py", run=a.run, fold=f, base_card=base_label,
                base_sha256=hashlib.sha256(src.read_bytes()).hexdigest(), out=str(out), out_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                card_class_vs_sb2_class_maxabs=dmax, replaced=["sband.pth", "calib.npz:sb_edge", "calib.npz:sb_head"])
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=1))
    print("FOLDCARD_OK", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
