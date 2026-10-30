#!/usr/bin/env python3
"""L2 T1-SBAND2 step 3/4: train ONE sband-head variant for ONE outer Re fold on that fold's inner-OOF cache (sb2_cache.py),
fit its bin -> interval table on the same cache (t1_sband_pack.py fit_table, pooled over the table fleets, acc-weighted), then
rebuild the fold card's served band with it on the fold's held-out windows (sb2_prep.py: B, xl, tgt; synth BK line) and write
per-window SPS numerators.  Nothing here ever reads a held-out residual before the held-out apply step.

Training = t1_sband_fit.py's loop (AdamW 3e-4, OneCycle 0.2, bs 16, grad clip 1, acc-weighted loss on the scored mask, feature
normalisation from 96 random training windows), generalised to:
  --fleets     target fleets 'a+b,c+d' (a fleet's residual = mean of its members' residuals); target = mean_f log(|r_f| + eps)
  --tfleets    fleets pooled in the table fit (default = --fleets)
  --feat v1|v2, --coord 0|2, --width, --layers, --dil (dilation cycle), --quantile (pinball on log|r|; 0.5 = plain L1),
  --q (table bins), --epochs, --cache (stride-20 or denser cache)

    CUDA_VISIBLE_DEVICES=G nice -n 10 python Track1/U-net/VariantSPSBand2/sb2_head.py --fold re0 --gpu G --tag base --seed 0
"""
import argparse, json, os, sys, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
T1U = REPO / "Track1/U-net"
KIT = REPO / "Track1/realpde_t1_starting_kit_v9"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sb2_lib as L  # noqa: E402

LT, C2, H, W = 20, 2, 32, 64
NOUT = LT * C2


def build_head(nn, torch, n_feat, width, n_layers, dil, n_out=NOUT, b0=-5.5):
    """sband_head.SBandHead with a configurable dilation cycle (the shipped class hard-codes 1,2,4,2,1)."""
    class SBandHead2(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_feat, self.width, self.n_layers, self.n_out, self.dil = int(n_feat), int(width), int(n_layers), int(n_out), list(dil)
            layers, c = [], int(n_feat)
            for i in range(int(n_layers)):
                k = self.dil[i % len(self.dil)]
                layers += [nn.Conv2d(c, width, 3, padding=k, dilation=k, padding_mode="replicate"), nn.GroupNorm(8, width), nn.GELU()]
                c = width
            self.body = nn.Sequential(*layers)
            self.out = nn.Conv2d(width, int(n_out), 1)
            nn.init.zeros_(self.out.weight); nn.init.constant_(self.out.bias, float(b0))
            self.register_buffer("f_mu", torch.zeros(int(n_feat))); self.register_buffer("f_sd", torch.ones(int(n_feat)))

        def set_feature_norm(self, mu, sd):
            self.f_mu.copy_(mu.reshape(-1)); self.f_sd.copy_(sd.reshape(-1).clamp_min(1e-6))

        def forward(self, feat):
            z = (feat - self.f_mu.reshape(1, -1, 1, 1)) / self.f_sd.reshape(1, -1, 1, 1)
            return self.out(self.body(z)).clamp(-12.0, 2.0)
    return SBandHead2()


# ------------------------------------------------------------------ table fit workers (one (lead, ch) slice each)
_TF = {}


def _tf_task(args):
    t, c, S_path, cache, tfleets, afac_path, Q = args
    S = np.load(S_path, mmap_mode="r"); SC = np.load(Path(cache) / "SC.npy", mmap_mode="r")
    AF = np.load(afac_path)
    s = np.asarray(S[:, t, :, :, c]); m = np.asarray(SC[:, t, :, :, c])
    rl = []
    for fl in tfleets:
        rr = sum(np.asarray(np.load(Path(cache) / ("R_%s.npy" % r), mmap_mode="r")[:, t, :, :, c], np.float64) for r in fl) / len(fl)
        rl.append(rr)
    e, tab = L.fit_table([s] * len(tfleets), rl, [m] * len(tfleets), [AF[:, i] for i in range(len(tfleets))], Q)
    return t, c, e, tab


def _ho_task(args):
    t, c, s_path, fold, e, tab = args
    ho = REPO / "_tmp/sband2/ho" / fold
    s = np.asarray(np.load(s_path, mmap_mode="r")[:, t, :, :, c])
    B = np.asarray(np.load(ho / "B.npy", mmap_mode="r")[:, t, :, :, c]); tg = np.asarray(np.load(ho / "tgt.npy", mmap_mode="r")[:, t, :, :, c], np.float64)
    xl = np.asarray(np.load(ho / "xl.npy", mmap_mode="r")[..., c], np.float64)
    lo_off, hi_off = L.lookup(s, e, tab)
    lo, hi = L.bk_forward(B, lo_off, hi_off, xl, t, c)
    absr = np.abs(tg - B); m = tg != 0
    sub = np.random.default_rng(t * 2 + c).choice(int(m.sum()), min(40000, int(m.sum())), replace=False)
    sv = s[m][sub]; rv = absr[m][sub]
    rk = lambda x: np.argsort(np.argsort(x)).astype(np.float64)
    rho = float(np.corrcoef(rk(sv), rk(rv))[0, 1])
    return t, c, L.window_numerator(tg, lo, hi), rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True); ap.add_argument("--gpu", default="0")
    ap.add_argument("--tag", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="")
    ap.add_argument("--fleets", default="WSW+MIMOWSW,MIMOA+RLB"); ap.add_argument("--tfleets", default="")
    ap.add_argument("--feat", default="v1"); ap.add_argument("--coord", type=int, default=0)
    ap.add_argument("--width", type=int, default=64); ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--dil", default="1,2,4,2,1"); ap.add_argument("--quantile", type=float, default=0.5)
    ap.add_argument("--q", type=int, default=64); ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--no-ho", action="store_true", help="train + table only (AoA folds / the ship head: no sb2_prep held-out arrays)")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    import torch, torch.nn as nn
    sys.path.insert(0, str(KIT)); import scoring
    sys.path.insert(0, str(T1U)); from sband_head import sband_features, N_FEAT, N_FEAT_V2
    t00 = time.time()
    cache = Path(a.cache or (REPO / ("_tmp/sband2/cache/%s_s20" % a.fold)))
    cache = cache if cache.is_absolute() else (REPO / cache)
    cache_label = str(cache.relative_to(REPO)) if cache.is_relative_to(REPO) else str(cache)
    cmeta = json.loads((cache / "meta.json").read_text())
    assert cmeta["fold"] == a.fold
    od = REPO / "_tmp/sband2/heads" / a.tag / ("%s_s%d" % (a.fold, a.seed)); od.mkdir(parents=True, exist_ok=True)
    fleets = [f.split("+") for f in a.fleets.split(",")]
    tfleets = [f.split("+") for f in (a.tfleets or a.fleets).split(",")]
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    Xp = np.load(cache / "Xp.npy", mmap_mode="r"); SC = np.load(cache / "SC.npy", mmap_mode="r"); T = np.load(cache / "T.npy", mmap_mode="r")
    R = {r: np.load(cache / ("R_%s.npy" % r), mmap_mode="r") for fl in fleets + tfleets for r in fl}
    N = Xp.shape[0]
    DEV = "cuda"

    # per-fleet accuracy factor of the fleet's averaged prediction (as t1_sband_cache.py afac)
    sq = lambda e: e / (0.5 + e)

    def afac_of(fl):
        out = np.empty(N, np.float32)
        for i in range(0, N, 128):
            Tt = np.asarray(T[i:i + 128], np.float64)
            P = Tt - sum(np.asarray(R[r][i:i + 128], np.float64) for r in fl) / len(fl)
            out[i:i + 128] = (0.5 * (1 - sq(scoring.rel_l2_per_sample(P, Tt, 2))) + 0.3 * (1 - sq(scoring.tke_rel_l2_per_sample(P, Tt, 2)))
                              + 0.2 * (1 - sq(scoring.mvpe_rel_l2_per_sample(P, Tt))))
        return out
    AFt = np.stack([afac_of(fl) for fl in fleets], 1)
    AFb = np.stack([afac_of(fl) for fl in tfleets], 1)
    np.save(od / "afac_table.npy", AFb)
    r0 = sum(np.asarray(R[r][::7], np.float64) for r in fleets[0]) / len(fleets[0])
    EPSC = np.array([np.quantile(np.abs(r0[..., c]), 0.01) for c in range(C2)], np.float32); del r0
    EPS_T = torch.as_tensor(EPSC).to(DEV).reshape(1, 1, 1, 1, C2)
    NF = (N_FEAT_V2 if a.feat == "v2" else N_FEAT) + int(a.coord)
    dil = [int(x) for x in a.dil.split(",")]
    net = build_head(nn, torch, NF, a.width, a.layers, dil).to(DEV)
    b = lambda idx, arr: torch.from_numpy(np.asarray(arr[idx])).to(DEV)
    tr = np.arange(N)
    with torch.no_grad():
        fs = sband_features(b(np.sort(np.random.default_rng(0).choice(tr, min(96, N), False)), Xp), a.coord, a.feat)
        net.set_feature_norm(fs.mean((0, 2, 3)), fs.std((0, 2, 3))); del fs
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    bs = 16; steps = a.epochs * ((N + bs - 1) // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.2)
    hist = []
    for ep in range(a.epochs):
        perm = np.random.permutation(tr); tot = 0.0; nb = 0
        for i in range(0, N, bs):
            idx = np.sort(perm[i:i + bs])
            xb = b(idx, Xp); msk = b(idx, SC)
            tgt = 0.0
            for fl in fleets:
                rf = sum(b(idx, R[r]) for r in fl) / float(len(fl))
                tgt = tgt + torch.log(rf.abs() + EPS_T)
            tgt = tgt / float(len(fleets))
            aw = torch.from_numpy(AFt[idx].mean(1)).to(DEV).reshape(-1, 1, 1, 1, 1)
            f = net(sband_features(xb, a.coord, a.feat)).reshape(-1, LT, C2, H, W).permute(0, 1, 3, 4, 2)
            e = f - tgt
            pl = e.abs() if abs(a.quantile - 0.5) < 1e-12 else torch.maximum(a.quantile * (-e), (a.quantile - 1.0) * (-e)) * 2.0
            loss = (pl * aw)[msk].mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sch.step()
            tot += float(loss); nb += 1
        hist.append(round(tot / nb, 5))
        print("  [%s %s s%d] ep%d loss %.4f (%.0fs)" % (a.tag, a.fold, a.seed, ep, tot / nb, time.time() - t00), flush=True)
    net.eval()

    def infer(X, path):
        o = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(X.shape[0], LT, H, W, C2))
        with torch.no_grad():
            for i in range(0, X.shape[0], 32):
                f = net(sband_features(torch.from_numpy(np.asarray(X[i:i + 32])).to(DEV), a.coord, a.feat))
                o[i:i + 32] = f.reshape(-1, LT, C2, H, W).permute(0, 1, 3, 4, 2).cpu().numpy()
        o.flush(); del o
    S_path = str(od / "S_cache.npy"); infer(Xp, S_path)
    ho = REPO / "_tmp/sband2/ho" / a.fold
    s_ho_path = str(od / "S_ho.npy")
    if not a.no_ho:
        infer(np.load(ho / "Xp.npy", mmap_mode="r"), s_ho_path)
    torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "n_feat": NF, "n_coord": int(a.coord), "feat": a.feat,
                "width": a.width, "n_layers": a.layers, "dil": dil, "n_out": NOUT, "fleets": a.fleets, "quantile": a.quantile,
                "fit_kind": "sb2 inner-OOF %s | %s" % (a.fold, a.fleets), "cache": cache_label}, od / "head.pth")
    t1 = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(a.nproc) as pool:
        res = pool.map(_tf_task, [(t, c, S_path, str(cache), tfleets, str(od / "afac_table.npy"), a.q) for t in range(LT) for c in range(C2)])
    E = np.zeros((LT, C2, a.q - 1)); TAB = np.zeros((LT, C2, a.q, 2))
    for t, c, e, tab in res:
        E[t, c] = e; TAB[t, c] = tab
    np.savez(od / "table.npz", edges=E, tab=TAB)
    t2 = time.time()
    if a.no_ho:
        out = dict(tool="Track1/U-net/VariantSPSBand2/sb2_head.py", fold=a.fold, tag=a.tag, seed=a.seed, args=vars(a), no_ho=True,
                   cache=cache_label, cache_n=int(N), params=int(sum(p.numel() for p in net.parameters())),
                   loss_hist=hist, t_train_s=round(t1 - t00), t_table_s=round(t2 - t1))
        (od / "res_noho.json").write_text(json.dumps(out, indent=1)); os.remove(S_path)
        print("HEAD_OK %s %s s%d no-ho (%.0fs)" % (a.tag, a.fold, a.seed, time.time() - t00), flush=True)
        return
    with ctx.Pool(a.nproc) as pool:
        res = pool.map(_ho_task, [(t, c, s_ho_path, a.fold, E[t, c], TAB[t, c]) for t in range(LT) for c in range(C2)])
    Snum = sum(r[2] for r in res); rho = float(np.mean([r[3] for r in res]))
    wz = np.load(ho / "win.npz")
    sps_new = L.sps_pool([wz["acc"]], [Snum], [wz["n"]]); sps_srv = L.sps_pool([wz["acc"]], [wz["S_served"]], [wz["n"]])
    np.savez(od / "ho_S.npz", S=Snum)
    out = dict(tool="Track1/U-net/VariantSPSBand2/sb2_head.py", fold=a.fold, tag=a.tag, seed=a.seed, args=vars(a),
               cache=cache_label, cache_n=int(N), params=int(sum(p.numel() for p in net.parameters())),
               loss_hist=hist, spearman_ho_vs_card_absr=round(rho, 4), sps_new=sps_new, sps_served=sps_srv,
               dsps=round(sps_new - sps_srv, 5), t_train_s=round(t1 - t00), t_table_s=round(t2 - t1), t_apply_s=round(time.time() - t2))
    (od / "res.json").write_text(json.dumps(out, indent=1))
    os.remove(S_path)
    print("HEAD_OK %s %s s%d dsps %+.4f rho %.4f (%.0fs)" % (a.tag, a.fold, a.seed, out["dsps"], rho, time.time() - t00), flush=True)


if __name__ == "__main__":
    main()
