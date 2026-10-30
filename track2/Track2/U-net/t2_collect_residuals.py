#!/usr/bin/env python3
"""M3 stage A -- one pass of the FROZEN shipped pipeline over local_bench/train, caching what the
band fitter (t2_fit_band.py) needs: NORMALISED residuals, input temporal std, variance maps, and
the packed scored-mask.  The bundle driven MUST run online rates {} (the frozen pipeline)."""
import argparse, sys, time, zipfile, tempfile, importlib.util
import hashlib
from pathlib import Path
import numpy as np, torch

# Derived from this file's own location (2026-08-12): the absolute hardcode silently
# escaped a fresh clone back to the production tree.  NB: run these tools from their
# in-repo location (the chains do); a /tmp copy would derive the wrong root.
REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--bundle", default=str(REPO/"Track2/U-net/submissions/t2_v27_nocal"))
ap.add_argument("--out", default=str(REPO/"Track2/U-net/m3_cache"))
ap.add_argument("--data-dir", default=str(REPO/"data/RealPDE_competition_dataset/gdrive_raw/local_bench/train"))
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO/"Track2/realpde_t2_starting_kit_v4"))
from local_eval import build_stream, Normalizer
BD = Path(a.bundle)
if not BD.is_dir():
    w = Path(tempfile.mkdtemp(prefix="m3a_")); zipfile.ZipFile(str(BD)+".zip").extractall(w); BD = w
sys.path.insert(0, str(BD))
sp = importlib.util.spec_from_file_location("pk_m3", BD/"submission.py")
mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)
m = mod.get_ttt_model(str(BD), "cuda")
assert getattr(mod, "ONLINE", None) == {}, f"need the FROZEN pipeline, got ONLINE={getattr(mod,'ONLINE',None)}"
dd = Path(a.data_dir); nz = Normalizer(dd/"mean_std_real.pt")
C2 = 2
RS, OB, VP, VT, TID, MS = [], [], [], [], [], []
prev = None; tr = -1; t0 = time.time()
for st in build_stream(dd):
    inp = st["input"].unsqueeze(0); tgt = st["target"].unsqueeze(0)
    if st["is_first"]: m.reset_ttt_state(); prev = None; tr += 1
    xn, tn = nz.preprocess(inp, tgt)
    p, info = m.ttt_step(xn, prev[1] if prev is not None else None)
    prev = (xn.detach(), tn.detach())
    pn = torch.as_tensor(p).detach().cpu()[0]; tnn = tn.detach().cpu()[0]
    RS.append((tnn - pn)[..., :C2].numpy().astype(np.float32))
    OB.append(xn[0, :, :, :, :C2].std(dim=0, unbiased=False).cpu().numpy().astype(np.float32))
    VP.append(pn[..., :C2].var(dim=0, unbiased=False).numpy().astype(np.float32))
    VT.append(tnn[..., :C2].var(dim=0, unbiased=False).numpy().astype(np.float32))
    MS.append((tgt.squeeze(0).numpy()[..., :C2] != 0.0))
    TID.append(tr)
RS = np.stack(RS); OB = np.stack(OB); MK = np.stack(MS)
print(f"# {RS.shape[0]} steps / {tr+1} trajectories in {time.time()-t0:.0f}s")
zf = 1.0 - MK.mean()
assert 0.05 < zf < 0.30 and 0.05 < float(OB.mean()) < 2.0 and float(OB.max()) > 1.0
np.save(OUT/"RSn.npy", RS); np.save(OUT/"OBn.npy", OB)
np.save(OUT/"VP.npy", np.stack(VP)); np.save(OUT/"VT.npy", np.stack(VT))
np.save(OUT/"tid.npy", np.asarray(TID))
np.save(OUT/"MKb.npy", np.packbits(MK.reshape(MK.shape[0], -1), axis=1))
np.save(OUT/"MKshape.npy", np.array(MK.shape))
# BCP: stamp the cache with the forecaster it was collected through.  t2_fit_band.py reads this
# file to fill prov_ckpt_md5/prov_ckpt_name, and both builders refuse a calib whose stamp names a
# different checkpoint.  Without it the T2 chain can only ever emit UNSTAMPED calibs, which the
# gate downgrades to a warning -- i.e. the structural half of BCP was inert on T2.  Found
# 2026-08-16 when a refit for the base128 forecaster came back with an empty stamp.
_ck = BD / "model.pth"
if _ck.exists():
    _md5 = hashlib.md5(_ck.read_bytes()).hexdigest()
    # PATH ONLY: t2_fit_band.py:35-45 treats the whole stripped file as a path and hashes it
    # itself, so any extra line makes os.path.exists() False and the stamp comes out empty.
    (OUT / "prov_ckpt.txt").write_text(str(_ck) + "\n")
    print("M3_PROV_STAMPED", _ck.name, _md5)
else:
    print("M3_PROV_MISSING: no model.pth in", BD)
print("M3_COLLECT_OK", OUT)
