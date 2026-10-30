#!/usr/bin/env python3
"""Isolated, fixed-recipe model subgraph for the submitted T1/T2 cards.

Example, from the repository root on a multi-GPU host:
  python Track1/U-net/VariantDecisionRepro/prepare_models.py --work Track1/U-net/VariantDecisionRepro/Logger/20260928_models --prepare --check-data
  python Track1/U-net/VariantDecisionRepro/prepare_models.py --work Track1/U-net/VariantDecisionRepro/Logger/20260928_models --node rlb72 --gpu 1

This deliberately stops at member weights. Calibration and packaging have separate
producers and must not be claimed complete merely because the three members train.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import yaml

# One development host excludes its GPU3 after a 2026-09-24 hardware fault. The ban applies on that host only (by hostname,
# same test as the development queue scripts); every other machine, including the organizer's server, uses every GPU.
BANNED_GPUS = {3} if __import__("socket").gethostname() == "40902" else set()

ROOT = Path(__file__).resolve().parents[3]
SPEC = Path(__file__).parent / "config/models.json"
SHIP72 = "data/RealPDE_competition_dataset/gdrive_raw/local_bench/train/test_real/*.h5"
ALL81 = "data/RealPDE_competition_dataset/gdrive_raw/local_bench/all81/test_real/*.h5"
SIM = "data/RealPDE_competition_dataset/gdrive_raw/extracted/train_sim/train_sim/*.h5"
ALLOWED_STATS = {
    "Track2/realpde_t2_starting_kit_v4/example_data/mean_std_real.pt",
    "data/RealPDE_competition_dataset/mean_std_comp_sim.pt",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def terminal_path(work: Path, node: dict) -> Path:
    base = work / "checkpoints" / (node["id"] + ".pth")
    return base.with_name(base.stem + "_terminal_ema.pth") if node["terminal"] else base


def source_parent_names(cfg: dict) -> list[str]:
    return [Path(x).name for x in str(cfg.get("distill_ckpt", "")).split(",") if x]


def validate_graph(spec: dict) -> None:
    nodes = spec["nodes"]
    ids = [n["id"] for n in nodes]
    assert len(ids) == len(set(ids)) and len(ids) == 12, ids
    seen: set[str] = set()
    for node in nodes:
        assert node["id"] not in seen and set(node["parents"]) <= seen, node
        assert node["real_pool"] in {"ship72", "all81"}
        assert (ROOT / node["source"]).is_file() and (ROOT / node["trainer"]).is_file(), node
        seen.add(node["id"])
    roles = sorted(r for n in nodes for r in n.get("ships_as", []))
    assert roles == sorted(["T1:model.pth", "T1:model_1.pth", "T2:model.pth", "T2:model2.pth"]), roles


def assert_sources(cfg: dict, node: dict) -> None:
    stages = cfg["stages"]
    assert [s["name"] for s in stages] == ["sim-pretrain", "real-finetune"], node["id"]
    sim = stages[0]["sources"]
    real = stages[1]["sources"]
    assert len(sim) == len(real) == 1
    assert sim[0]["tag"] == "comp_sim" and sim[0]["glob"] == SIM
    assert real[0]["tag"] == "comp_real"
    want = SHIP72 if node["real_pool"] == "ship72" else ALL81
    if node["id"].startswith("ad_"):
        assert real[0]["glob"] == SHIP72 and sorted(cfg.get("exclude_re", [])) == [11400, 17775]
    else:
        assert real[0]["glob"] == want, (node["id"], real[0]["glob"], want)
    assert cfg["stats"] in ALLOWED_STATS
    assert sim[0]["stats"] in ALLOWED_STATS and real[0]["stats"] in ALLOWED_STATS
    assert cfg["val_glob"].startswith("data/RealPDE_competition_dataset/gdrive_raw/local_bench/")
    assert "init_ckpt" not in cfg or not cfg["init_ckpt"], node["id"]


def config_for(node: dict, by_id: dict[str, dict], work: Path) -> tuple[dict, list[str]]:
    cfg = copy.deepcopy(yaml.safe_load((ROOT / node["source"]).read_text()))
    assert_sources(cfg, node)
    prior = source_parent_names(cfg)
    assert len(prior) == len(node["parents"]), (node["id"], prior, node["parents"])
    # Explicit original parent order is a recipe property: an ensemble's teacher order matters.
    for got, parent in zip(prior, node["parents"]):
        expected = Path(yaml.safe_load((ROOT / by_id[parent]["source"]).read_text())["checkpoint"]).name
        if parent == "rlb72":
            assert got == expected, (node["id"], got, expected)
        elif parent.startswith("ad_"):
            assert got == "unet_v4_ship_AD_" + parent.removeprefix("ad_").upper() + ".pth", (node["id"], got)
        else:
            assert got == expected.replace(".pth", "_terminal_ema.pth"), (node["id"], got, expected)
    if node["id"].startswith("ad_"):
        cfg["stages"][1]["sources"][0]["glob"] = ALL81
        cfg["exclude_re"] = []
    cfg["checkpoint"] = str(work / "checkpoints" / (node["id"] + ".pth"))
    cfg["logger_root"] = str(work / "trainer_logs" / node["id"])
    cfg["variant"] = "decision_repro_" + node["id"]
    cfg["desc"] = "decision_repro_" + node["id"]
    if node["parents"]:
        cfg["distill_ckpt"] = ",".join(str(terminal_path(work, by_id[p])) for p in node["parents"])
    assert bool(cfg.get("save_terminal_both", False)) == bool(node["terminal"]), node["id"]
    return cfg, prior


def check_data() -> None:
    from glob import glob
    real81 = [Path(p).name for p in glob(str(ROOT / ALL81))]
    real72 = [Path(p).name for p in glob(str(ROOT / SHIP72))]
    sim = [Path(p).name for p in glob(str(ROOT / SIM))]
    assert len(real81) == 81 and len(set(real81)) == 81 and "7575_0.h5" not in real81, (len(real81), real81[:5])
    assert len(real72) == 72 and set(real72) < set(real81), (len(real72), len(set(real72) - set(real81)))
    assert len(sim) == 100 and len(set(sim)) == 100, len(sim)
    for p in ALLOWED_STATS:
        assert (ROOT / p).is_file(), p
    print("DATA_OK", len(real72), len(real81), len(sim), "7575_0 excluded")


def prepare(work: Path, spec: dict) -> None:
    by_id = {n["id"]: n for n in spec["nodes"]}
    payload = {"schema": 1, "spec_sha256": sha(SPEC), "sources": {}, "trainers": {}, "configs": {}, "work": str(work)}
    for node in spec["nodes"]:
        payload["sources"][node["source"]] = sha(ROOT / node["source"])
        payload["trainers"][node["trainer"]] = sha(ROOT / node["trainer"])
        cfg, prior = config_for(node, by_id, work)
        text = yaml.safe_dump(cfg, sort_keys=False)
        payload["configs"][node["id"]] = {"sha256": hashlib.sha256(text.encode()).hexdigest(),
                                           "parents_original": prior, "checkpoint": cfg["checkpoint"]}
    plan = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    plan_path = work / "plan.json"
    if plan_path.exists():
        assert plan_path.read_text() == plan, "PLAN DRIFT: keep old run immutable; choose a fresh work name"
    else:
        (work / "configs").mkdir(parents=True, exist_ok=True)
        (work / "checkpoints").mkdir(exist_ok=True)
        (work / "logs").mkdir(exist_ok=True)
        plan_path.write_text(plan)
    for node in spec["nodes"]:
        cfg, _ = config_for(node, by_id, work)
        path = work / "configs" / (node["id"] + ".yaml")
        text = yaml.safe_dump(cfg, sort_keys=False)
        if path.exists():
            assert path.read_text() == text, "CONFIG DRIFT " + node["id"]
        else:
            path.write_text(text)
    print("PREPARED", work, "nodes", len(spec["nodes"]), "spec", payload["spec_sha256"])


def run_node(work: Path, spec: dict, name: str, gpu: int) -> None:
    assert gpu >= 0 and gpu not in BANNED_GPUS, (gpu, "GPU3 is excluded on this host")
    by_id = {n["id"]: n for n in spec["nodes"]}
    assert name in by_id, name
    prepare(work, spec)
    node = by_id[name]
    for p in node["parents"]:
        done = work / "logs" / (p + ".complete.json")
        assert done.is_file(), (name, "missing completed parent", p)
        pd = json.loads(done.read_text())
        pp = terminal_path(work, by_id[p])
        assert pp.is_file() and sha(pp) == pd["output_sha256"], (name, p, "parent changed")
    out = terminal_path(work, node)
    done = work / "logs" / (name + ".complete.json")
    if done.exists():
        assert out.is_file() and sha(out) == json.loads(done.read_text())["output_sha256"], name
        print("ALREADY_COMPLETE", name, sha(out)); return
    assert not out.exists(), (name, "output exists without completion proof; inspect before continuing")
    from subprocess import check_output
    info = check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).splitlines()
    used, util = [int(x.strip()) for x in info[gpu].split(",")]
    assert __import__("os").environ.get("RPDE_PIPELINE_SHARES_GPU") == "1" or (used < 100 and util < 20), (gpu, "not idle", used, util)
    cmd = [sys.executable, "-u", str(ROOT / node["trainer"]), "--config", str(work / "configs" / (name + ".yaml")), "--gpu", str(gpu)]
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    log = work / "logs" / (name + ".train.log")
    assert not log.exists(), (name, "prior attempt exists; inspect it, do not silently rerun")
    print("START", name, "gpu", gpu, "log", log, "utc", datetime.now(timezone.utc).isoformat(), flush=True)
    with log.open("x") as f:
        f.write("COMMAND " + json.dumps(cmd) + "\n")
        f.flush()
        rc = subprocess.call(cmd, cwd=ROOT / "Track1/U-net", env=env, stdout=f, stderr=subprocess.STDOUT)
    lines = log.read_text(errors="replace").splitlines()
    assert rc == 0 and any(line.startswith("TRAIN_V4_OK ") for line in lines), (name, rc, lines[-8:])
    assert out.is_file() and out.stat().st_size > 0, (name, "missing terminal output", out)
    record = {"node": name, "gpu": gpu, "utc": datetime.now(timezone.utc).isoformat(),
              "plan_sha256": sha(work / "plan.json"), "config_sha256": sha(work / "configs" / (name + ".yaml")),
              "trainer_sha256": sha(ROOT / node["trainer"]), "output": str(out), "output_sha256": sha(out),
              "log": str(log), "log_sha256": sha(log)}
    done.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print("COMPLETE", name, record["output_sha256"], flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--check-data", action="store_true")
    ap.add_argument("--node", choices=[n["id"] for n in json.loads(SPEC.read_text())["nodes"]])
    ap.add_argument("--gpu", type=int)
    args = ap.parse_args()
    work = (ROOT / args.work).resolve()
    assert work.is_relative_to(Path(__file__).parent / "Logger"), "work must be this Variant's Logger child"
    spec = json.loads(SPEC.read_text()); validate_graph(spec)
    if args.check_data: check_data()
    if args.prepare or args.node: prepare(work, spec)
    if args.node:
        assert args.gpu is not None, "--node needs --gpu"
        run_node(work, spec, args.node, args.gpu)


if __name__ == "__main__":
    main()
