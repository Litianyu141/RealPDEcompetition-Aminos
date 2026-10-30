#!/usr/bin/env python3
"""C (shared fivefold OOF, design `shared_oof/README.md`): seed-1 replicas of a track's Final members on the Re3 folds,
and the fleet adapter that lets the unchanged vgain / sband scripts read the members' out-of-fold forecasts.

Fleet A = the track's two Final members' Re3 fold versions (already trained by the re3 graph); fleet B = their seed-1
replicas (the original node's scoped config with `seed: 1` and new output names; same data, parents and steps).
The adapter links `<stem>_f<k>.pth` (k = 0, 1, 2 for re0, re1, re2) into one directory and writes a three-fold manifest
in the legacy `splits_kfold_dedup.json` format, so `train_vgain.py --arm/--arm2/--ckpt-dir/--manifest` and
`t1_sband_cache.py` (SBAND_FLEET_A/B, --manifest) run unchanged.  `ready.json` binds every linked checkpoint by sha256;
`ready.failed` tells the waiting base chain that the graph side failed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

import re3_members as RM

ROOT = Path(__file__).resolve().parents[3]
DEC = Path(__file__).resolve().parent
SPLIT = ROOT / "common/local_benchmark/splits_re3_aoa2_v1.json"
REAL_DIR = "data/RealPDE_competition_dataset/gdrive_raw/extracted/train_real/train_real"
# Dataset-level keys of the legacy manifest format (common/local_benchmark/splits_kfold_dedup.json), same repo-relative
# values: t1_sband_cache.py reads `stats_real` (C smoke 2026-09-30: KeyError at band step 2/5 without it).
LEGACY_KEYS = {"stats_real": "Track2/realpde_t2_starting_kit_v4/example_data/mean_std_real.pt",
               "stats_sim": "data/RealPDE_competition_dataset/mean_std_comp_sim.pt",
               "real_glob": REAL_DIR + "/*.h5",
               "sim_glob": "data/RealPDE_competition_dataset/gdrive_raw/extracted/train_sim/train_sim/*.h5"}
TRACK_FLEETS = {"T1": {"A": ("T1GEG_H", "T2PTK_SW"), "B": ("T1GEGs1_H", "T2PTK_SWs1")},
                "T2": {"A": ("T2PHK_H", "T2PTK_SW"), "B": ("T2PHK_Hs1", "T2PTK_SWs1")}}
REPLICA_OF = {"T1GEGs1_H": "T1GEG_H", "T2PTK_SWs1": "T2PTK_SW", "T2PHK_Hs1": "T2PHK_H"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def folds() -> tuple:
    """The Re folds of the re3 graph spec (all three in the Final context; nested contexts rewrite the spec)."""
    return tuple(json.loads(RM.SPEC.read_text())["folds"])


def work_dir() -> Path:
    return DEC / "Logger/final_shared_oof"


def stem(role: str) -> str:
    return "c_" + role


def replica_id(fold: str, role: str) -> str:
    return fold + "_" + role.lower()


def replica_output(work: Path, rid: str) -> Path:
    return work / "checkpoints" / (rid + "_f0_terminal_ema.pth")


def replica_config(re3: Path, work: Path, fold: str, role: str) -> dict:
    orig = fold + "_" + REPLICA_OF[role].lower()
    c = yaml.safe_load((re3 / "configs" / (orig + ".yaml")).read_text())
    assert c.get("seed", 0) == 0 and c["save_terminal_both"] is True, orig
    rid = replica_id(fold, role)
    c["seed"] = 1
    c["checkpoint"] = str(work / "checkpoints" / (rid + ".pth"))
    c["logger_root"] = str(work / "trainer_logs" / rid)
    c["variant"] = "decision_re3_" + rid
    c["desc"] = c["variant"]
    return c


def run_replica(re3: Path, work: Path, fold: str, role: str, gpu: int) -> Path:
    """Train one seed-1 replica with the re3 trainer; same guards and proof format as re3_members.run_node."""
    assert gpu >= 0 and gpu not in RM.BANNED_GPUS, (gpu, "GPU banned on this host")
    orig = fold + "_" + REPLICA_OF[role].lower()
    done_orig = json.loads((re3 / "logs" / (orig + ".complete.json")).read_text())
    assert done_orig["status"] == "OK" and sha(RM.output(re3, fold, REPLICA_OF[role])) == done_orig["output_sha256"], orig
    rid = replica_id(fold, role)
    for d in ("configs", "checkpoints", "logs"):
        (work / d).mkdir(parents=True, exist_ok=True)
    cfg = work / "configs" / (rid + ".yaml")
    body = yaml.safe_dump(replica_config(re3, work, fold, role), sort_keys=False)
    if cfg.exists():
        assert cfg.read_text() == body, ("replica config drift", rid)
    else:
        cfg.write_text(body)
    ck, done = replica_output(work, rid), work / "logs" / (rid + ".complete.json")
    if done.exists():
        assert ck.is_file() and sha(ck) == json.loads(done.read_text())["output_sha256"]
        print("C_REPLICA_ALREADY_COMPLETE", rid, flush=True)
        return ck
    assert not ck.exists(), (rid, "unproved output exists")
    log = work / "logs" / (rid + ".train.log")
    assert not log.exists(), "prior replica attempt exists; inspect before retry"
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True).splitlines()
    used, util = [int(x.strip()) for x in raw[gpu].split(",")]
    assert os.environ.get("RPDE_PIPELINE_SHARES_GPU") == "1" or (used < 100 and util < 20), (gpu, used, util)
    trainer = ROOT / json.loads(RM.SPEC.read_text())["trainer"]
    cmd = [sys.executable, "-u", str(trainer), "--config", str(cfg), "--gpu", str(gpu)]
    print("C_REPLICA_START", rid, gpu, datetime.now(timezone.utc).isoformat(), flush=True)
    with log.open("x") as f:
        f.write("COMMAND " + json.dumps(cmd) + "\n"); f.flush()
        rc = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                             env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"))
    lines = log.read_text(errors="replace").splitlines()
    assert rc == 0 and any(x.startswith("TRAIN_V4_OK ") for x in lines), (rid, rc, lines[-8:])
    assert ck.is_file() and ck.stat().st_size > 0
    rec = {"status": "OK", "node": rid, "replica_of": orig, "seed": 1, "gpu": gpu,
           "utc": datetime.now(timezone.utc).isoformat(), "config_sha256": sha(cfg),
           "trainer_sha256": sha(trainer), "output_sha256": sha(ck), "log_sha256": sha(log)}
    done.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print("C_REPLICA_COMPLETE", rid, rec["output_sha256"], flush=True)
    return ck


def manifest_doc() -> dict:
    split = json.loads(SPLIT.read_text())
    by = {f["id"]: f for f in split["folds"]}
    folds_ = []
    for k, fid in enumerate(folds()):
        f = by[fid]
        folds_.append({"fold": k, "id": fid, "val_re": f["val_groups"],
                      "val_files": [REAL_DIR + "/" + n for n in f["val_files"]]})
    names = [n for fd in folds_ for n in fd["val_files"]]
    assert names and len(names) == len(set(names)), "duplicate or empty Re-fold validation files"
    return {"scheme": "re3_aoa2_v1 Re folds (C shared fivefold OOF)", "k": len(folds_), **LEGACY_KEYS, "folds": folds_}


def build_fleet(track: str, re3: Path, work: Path) -> dict:
    """Link fleet A (re3 members) and fleet B (replicas) as <stem>_f<k>.pth, write the manifest and ready.json."""
    fleet = work / "fleet" / track
    fleet.mkdir(parents=True, exist_ok=True)
    links = {}
    for group in ("A", "B"):
        for role in TRACK_FLEETS[track][group]:
            for k, fold in enumerate(folds()):
                src = (RM.output(re3, fold, role) if group == "A"
                       else replica_output(work, replica_id(fold, role)))
                assert src.is_file(), src
                dst = fleet / (stem(role) + "_f" + str(k) + ".pth")
                if dst.is_symlink() or dst.exists():
                    assert dst.resolve() == src.resolve(), (dst, "fleet link drift")
                else:
                    dst.symlink_to(src)
                links[dst.name] = sha(src)
    man = fleet / "re3_manifest.json"
    body = json.dumps(manifest_doc(), indent=2) + "\n"
    if man.exists():
        assert man.read_text() == body
    else:
        man.write_text(body)
    a, b = TRACK_FLEETS[track]["A"], TRACK_FLEETS[track]["B"]
    ready = {"status": "READY", "track": track, "fleet_dir": str(fleet), "manifest": str(man),
             "arm": stem(a[0]), "arm2": stem(a[1]), "fleet_a": [stem(r) for r in a], "fleet_b": [stem(r) for r in b],
             "folds": list(folds()), "checkpoint_sha256": links, "utc": datetime.now(timezone.utc).isoformat()}
    (fleet / "ready.json").write_text(json.dumps(ready, indent=2, sort_keys=True) + "\n")
    print("C_FLEET_READY", track, fleet, flush=True)
    return ready


def env_for(track: str) -> dict:
    """Environment the base-chain scripts read in C mode (paths are fixed in advance; ready.json gates them)."""
    fleet = work_dir() / "fleet" / track
    a, b = TRACK_FLEETS[track]["A"], TRACK_FLEETS[track]["B"]
    return {"RPDE_C_SHARED_OOF": "1", "RPDE_C_READY": str(fleet / "ready.json"),
            "RPDE_C_FAILED": str(fleet / "ready.failed"), "RPDE_C_FLEET_DIR": str(fleet),
            "RPDE_C_MANIFEST": str(fleet / "re3_manifest.json"), "RPDE_C_ARM": stem(a[0]), "RPDE_C_ARM2": stem(a[1]),
            "RPDE_C_FLEET_B1": stem(b[0]), "RPDE_C_FLEET_B2": stem(b[1]),
            "RPDE_C_FOLD_IDS": " ".join(str(k) for k in range(len(folds())))}
