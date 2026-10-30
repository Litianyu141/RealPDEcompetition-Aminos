"""Forecast every 20-frame window of one real trajectory with a trained package and save the arrays.

Run inside the track's image, with the trained output and the data mounted:

    docker run --rm --gpus '"device=0"' -v $OUT:/output:ro -v $DATA:/data:ro \
        -v $PWD/tools:/tools:ro -v $PWD/results:/results \
        aminos-t1 python /tools/predict_trajectory.py T1 /data/train_real/10125_10.h5 /results/T1_10125_10.npz

Track 1 calls predict() once on all windows (inductive; each window is forecast from itself only).
Track 2 runs the causal protocol: reset, then ttt_step per window, revealing each target only after its forecast.
Saved arrays are in physical units, shape (windows, 20, 32, 64, 3): X inputs, Y targets, P prediction, L/U bounds.
"""
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, "/workspace")
import solution  # noqa: E402

track, h5_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
output_dir = sys.argv[4] if len(sys.argv) > 4 else "/output"

with h5py.File(h5_path, "r") as f:   # native 64x128 grid -> the 32x64 evaluation grid
    u = np.asarray(f["u"][:, ::2, ::2], np.float32)
    v = np.asarray(f["v"][:, ::2, ::2], np.float32)
x = np.stack([u, v, np.zeros_like(u)], -1)
t0s = list(range(0, x.shape[0] - 40 + 1, 20))
X = np.stack([x[t:t + 20] for t in t0s])
Y = np.stack([x[t + 20:t + 40] for t in t0s])

if track == "T1":
    out = solution.load_predictor(output_dir, "cuda").predict(X.copy())   # all windows in one batch, ~12 GB GPU
    P, L, U = (np.asarray(out[k]) for k in ("prediction", "lower", "upper"))
else:
    model = solution.get_ttt_model(output_dir, torch.device("cuda"))
    # The official affine: (mean_in, mean_tgt, std_in, std_tgt), with std == 0 replaced by 1.
    mi, mt, si, st = (t.float() for t in torch.load(f"{output_dir}/mean_std_real.pt", map_location="cpu",
                                                    weights_only=False))
    si = torch.where(si == 0, torch.ones_like(si), si)
    st = torch.where(st == 0, torch.ones_like(st), st)
    norm_in = lambda a: (torch.from_numpy(a) - mi[..., :3]) / si[..., :3]
    norm_tgt = lambda a: (torch.from_numpy(a) - mt[..., :3]) / st[..., :3]
    denorm = lambda a: (a.detach().float().cpu() * st[..., :3] + mt[..., :3]).numpy()[0]
    model.reset_ttt_state()
    P, L, U, prev = [], [], [], None
    for i in range(len(X)):
        pred, info = model.ttt_step(norm_in(X[i:i + 1]).cuda(), prev)
        P.append(denorm(pred)); L.append(denorm(info["lower"])); U.append(denorm(info["upper"]))
        prev = norm_tgt(Y[i:i + 1]).cuda()
    P, L, U = map(np.stack, (P, L, U))

np.savez_compressed(out_path, X=X, Y=Y, P=P, L=L, U=U, t0=np.array(t0s))
print(f"saved {out_path}: {len(X)} windows")
