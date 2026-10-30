#!/usr/bin/env python3
"""Final-kit train() orchestration for source-only T1 v185 / T2 v156g descendants.

Every neural checkpoint, band, head, prior and preconditioner is produced from
the mounted public release in this run.  The historic submitted ZIPs are never
mounted or read.  All intermediate products live below output_dir/work; the
flat inference files are written to output_dir only after all producers pass.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from zipfile import ZipFile

if (Path(__file__).resolve().parent / "Track1").is_dir():
    ROOT = Path(__file__).resolve().parent
else:
    ROOT = Path(__file__).resolve().parents[3]
DEC = ROOT / "Track1/U-net/VariantDecisionRepro"
T2DEC = ROOT / "Track2/U-net/VariantDecisionRepro"
sys.path.insert(0, str(DEC))
import stage_final_data as FD  # noqa: E402
import stage_legacy_layout as LL  # noqa: E402
import container_models as CM  # noqa: E402
import band_teachers as BT  # noqa: E402
import re3_members as RM  # noqa: E402
import aoa_folds as AF  # noqa: E402
import make_runtime_template as RT  # noqa: E402
import assemble_source as AS  # noqa: E402
import shared_oof_fleet as SO  # noqa: E402

# One development host excludes its GPU3 after a 2026-09-24 hardware fault. The ban applies on that host only (by hostname,
# same test as the development queue scripts); every other machine, including the organizer's server, uses every GPU.
BANNED_GPUS = {3} if __import__("socket").gethostname() == "40902" else set()


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def source_guard() -> None:
    manifest = ROOT / "source_manifest.json"
    doc = json.loads(manifest.read_text())
    assert doc["source_only"] is True and len(doc["source_sha256"]) > 150
    for name, want in doc["source_sha256"].items():
        p = ROOT / name
        assert p.is_file() and sha(p) == want, (name, "source drift")
    for which in ("T1", "T2"):
        s = (ROOT / ("Track1" if which == "T1" else "Track2") /
             "U-net/VariantDecisionRepro/runtime/submission.py").read_text()
        for name, count in RT.COUNTS[which].items():
            assert s.count(f"__TRAINED_{name}__") == count, \
                (which, name, "fitted runtime literal or template drift")
    AS.validate_static_refs(ROOT)
    AS.validate_import_closure(ROOT)
    print("FINAL_SOURCE_GUARD_OK", len(doc["source_sha256"]), flush=True)


class ResourceMonitor:
    """Low-rate in-container GPU-memory evidence for the hardware README."""

    def __init__(self, output: Path):
        self.path = output / "work/logs/resource_samples.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        assert not self.path.exists(), "prior resource monitor; inspect before another run"
        self.stop_event = threading.Event()
        self.peak: dict[int, int] = {}
        self.failures = 0
        self.ever_ok = False
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self, stream) -> None:
        try:
            raw = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"], text=True, timeout=8)
            rows = []
            for line in raw.splitlines():
                idx, used, util, temp = [int(v.strip()) for v in line.split(",")]
                self.peak[idx] = max(self.peak.get(idx, 0), used)
                rows.append({"gpu": idx, "used_mib": used, "util_pct": util, "temp_c": temp})
            record = {"utc": datetime.now(timezone.utc).isoformat(), "gpus": rows}
            self.failures = 0
            self.ever_ok = True
        except Exception as exc:
            record = {"utc": datetime.now(timezone.utc).isoformat(), "error": str(exc)}
            self.failures += 1
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        # GPU LOST (2026-09-30, development host): a running container lost its GPUs ("Failed to initialize NVML: Unknown Error")
        # and the base chain silently continued on the CPU for 8.6 h.  Twenty failed samples in a row (5 min) end the
        # whole train() at once with a clear message instead of a day of CPU training.  Only after nvidia-smi has
        # worked once in this run, so a host without nvidia-smi (or with N/A fields) never trips it.
        if self.ever_ok and self.failures >= 20:
            print("GPU_LOST_ABORT nvidia-smi failed %d times in a row; last error: %s"
                  % (self.failures, record.get("error")), flush=True)
            os.killpg(os.getpgrp(), signal.SIGKILL)
            os._exit(97)  # reached only if this process ignores the group kill (PID 1 of a container)

    def _loop(self) -> None:
        with self.path.open("x") as stream:
            self._sample(stream)
            while not self.stop_event.wait(15):
                self._sample(stream)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> dict[int, int]:
        self.stop_event.set()
        self.thread.join(timeout=12)
        assert not self.thread.is_alive(), "resource monitor did not stop"
        return dict(sorted(self.peak.items()))


MPS_ENV = ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY")
MPS_PROBE = ("import torch; x = torch.ones(1 << 20, device='cuda'); torch.cuda.synchronize(); "
             "print('MPS_PROBE_OK', int(x.sum().item()))")


class SharedGpuMps:
    """Shared single GPU only (the organizer's default): NVIDIA MPS lets the three concurrent subgraphs' kernels run
    together instead of time-slicing between CUDA contexts.  A100 2026-09-30, synthetic re3-style step in the Final
    image: 3 time-sliced processes 28.1 steps/s in total (below one process alone, 31.6), 3 under MPS 39.0
    (acceptance/mps_bench_20260930).  Same code, data and seeds; only the GPU scheduling changes.

    Probe-gated: the daemon must start, a CUDA probe subprocess must run through it, and the server log must show the
    server it spawned.  Any failure quits the daemon, restores the environment and falls back to time-slicing
    (MPS_FALLBACK).  RPDE_SHARED_GPU_MPS=0 disables it.  A server failure later in the run is not retried: the
    affected node fails and train() stops like any other node failure (never observed; 131 trainings ran clean)."""

    def __init__(self, out: Path):
        self.out = out
        self.active = False
        self.reason = ""
        self.dirs: dict[str, str] = {}
        self.quit_sent = False

    def _control(self, args: list[str], stdin: str | None = None, timeout: int = 60) -> subprocess.CompletedProcess:
        return subprocess.run(["nvidia-cuda-mps-control"] + args, input=stdin, text=True, capture_output=True,
                              env=dict(os.environ, **self.dirs), timeout=timeout)

    def _log_text(self) -> str:
        p = Path(self.dirs.get("CUDA_MPS_LOG_DIRECTORY", "/nonexistent")) / "control.log"
        return p.read_text(errors="replace") if p.is_file() else ""

    def start(self) -> bool:
        if os.environ.get("RPDE_SHARED_GPU_MPS", "1") == "0":
            self.reason = "disabled by RPDE_SHARED_GPU_MPS=0"
        elif shutil.which("nvidia-cuda-mps-control") is None:
            self.reason = "nvidia-cuda-mps-control not found"
        elif any(os.environ.get(k) for k in MPS_ENV):
            self.reason = "an MPS environment is already set by the caller"
        else:
            try:
                base = Path(tempfile.mkdtemp(prefix="rpde_mps_"))
                self.dirs = {"CUDA_MPS_PIPE_DIRECTORY": str(base / "pipe"), "CUDA_MPS_LOG_DIRECTORY": str(base / "log")}
                for d in self.dirs.values():
                    Path(d).mkdir(parents=True)
                # The daemon forks: give it no pipes (an inherited pipe would keep this call waiting for EOF).
                start_log = Path(self.dirs["CUDA_MPS_LOG_DIRECTORY"]) / "daemon_start.out"
                with start_log.open("w") as f:
                    rc = subprocess.call(["nvidia-cuda-mps-control", "-d"], stdin=subprocess.DEVNULL, stdout=f,
                                         stderr=subprocess.STDOUT, env=dict(os.environ, **self.dirs), timeout=60)
                assert rc == 0, ("daemon start", rc, start_log.read_text(errors="replace")[-300:])
                probe = subprocess.run([sys.executable, "-c", MPS_PROBE], text=True, capture_output=True, timeout=180,
                                       env=dict(os.environ, **self.dirs))
                assert probe.returncode == 0 and "MPS_PROBE_OK 1048576" in probe.stdout, \
                    ("probe", probe.returncode, probe.stdout[-200:], probe.stderr[-300:])
                log = self._log_text()
                assert "starting new server" in log.lower() or "new client" in log.lower(), ("server log", log[-300:])
                os.environ.update(self.dirs)   # inherited by every training subprocess started after this point
                self.active = True
                self.reason = "probe served by the MPS server"
            except Exception as exc:
                self.reason = "start failed: %s" % (exc,)
                self._quit()
        print(("MPS_ACTIVE " if self.active else "MPS_FALLBACK ") + self.reason, flush=True)
        return self.active

    def _quit(self) -> None:
        if not self.dirs or self.quit_sent:
            return
        try:
            self._control([], stdin="quit\n", timeout=60)
        except Exception as exc:
            print("MPS_QUIT_ERROR", exc, flush=True)
        for k in MPS_ENV:
            if os.environ.get(k) == self.dirs.get(k):
                os.environ.pop(k)
        self.quit_sent = True

    def stop(self) -> dict:
        log = self._log_text()
        if log:
            dst = self.out / "work/logs/mps_control.log"
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open("a") as f:
                f.write(log)
        self._quit()
        return {"active": self.active, "reason": self.reason}


def run_logged(label: str, cmd: list[str], out: Path, *, env: dict[str, str] | None = None,
               cwd: Path = ROOT, marker: str | None = None) -> None:
    logs = out / "work/logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / (label + ".log")
    proof = logs / (label + ".complete.json")
    if proof.exists():
        rec = json.loads(proof.read_text())
        assert rec["status"] == "OK" and log.is_file() and sha(log) == rec["log_sha256"]
        assert rec["command"] == cmd
        return
    assert not log.exists(), (label, "prior attempt without acceptance; inspect, never reroll")
    e = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
    if env:e.update(env)
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    with log.open("x") as f:
        f.write("COMMAND " + json.dumps(cmd) + "\n"); f.flush()
        rc = subprocess.call(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=e)
    lines = log.read_text(errors="replace").splitlines()
    assert rc == 0 and (marker is None or any(marker in line for line in lines)), \
        (label, rc, lines[-12:])
    rec = {"status": "OK", "command": cmd, "log_sha256": sha(log), "marker": marker,
           "started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(),
           "elapsed_s": time.monotonic() - t0}
    proof.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print("FINAL_STAGE_OK", label, flush=True)


def run_t1_base(out: Path, gpu: int) -> Path:
    work = out / "work/base_t1"
    env = {"RPDE_ROOT": str(ROOT), "RPDE_FINAL_SOURCE_MANIFEST": str(ROOT / "source_manifest.json"),
           "TAG": "dpv185base", "BOOTSTRAP": "1", "REPRO_SCOPED": "1",
           "BAND_PIPELINE": "1", "ALL81": "1", "GEN2": "1", "GPU": str(gpu), "PY": sys.executable,
           "RPDE_FINAL_MODE": "1"}
    run_logged("t1_base", ["bash", str(ROOT / "Track1/U-net/reproduce_t1.sh")], out,
               env=env, marker="REPRODUCE_T1_OK")
    for p in (work / "checkpoints/vgain_head_ship_dpv185base.pth",
              work / "band_pipeline/calib_rebuilt.npz",
              work / "band_pipeline/cache/sband_dpv185base_band.pth"):
        assert p.is_file(), p
    return work


def run_t2_base(out: Path, t1base: Path, gpu: int) -> Path:
    work = out / "work/base_t2"
    env = {"RPDE_ROOT": str(ROOT), "RPDE_T2_BASE_OUT": str(ROOT / "Track2/U-net/dp156base"), "RPDE_WAIT_T1_BAND": "0",
           "SHIP_SBAND": str(t1base / "band_pipeline/cache/sband_dpv185base_band.pth"),
           "T1_SBAND_CALIB": str(t1base / "band_pipeline/calib_rebuilt.npz"),
           "T2_PREFIT_CALIB": str(work / "deferred_prefit.npz"),
           "RPDE_T1_BASE_DIR": str(t1base),
           "BOOT_CALIB": str(work / "calib_boot.npz"),
           "BOOTSTRAP": "1", "REPRO_SCOPED": "1", "ALL81": "1", "GEN2": "1",
           "GPU": str(gpu), "PY": sys.executable, "RPDE_FINAL_MODE": "1"}
    assert not Path(env["T2_PREFIT_CALIB"]).exists(), "prefit must be generated by this chain"
    run_logged("t2_base", ["bash", str(ROOT / "Track2/U-net/reproduce_t2.sh")], out,
               env=env, marker="REPRODUCE_T2_OK")
    for p in (work / "checkpoints/vgain_head_ship_t2repro.pth", work / "calib_boot.npz",
              work / "calib_sbport_repro.npz"):
        assert p.is_file(), p
    return work


# Final members per track (t1_assets / t2_assets and the fold tables that capture them).  A track's train() trains only
# the ancestors of its own members in the models, Re3 and AoA graphs (2026-09-30): T1 never consumes the T2 H member
# and T2 never consumes the T1 H member.  The skip sets are derived from the graph specs and checked against the
# expected sets at startup, so a spec change fails in the first minute instead of at the card stage.
TRACK_MEMBERS = {"T1": ({"t1_geg_h", "t2_ptk_sw"}, {"T1GEG_H", "T2PTK_SW"}),
                 "T2": ({"t2_phk_h", "t2_ptk_sw"}, {"T2PHK_H", "T2PTK_SW"})}
EXPECTED_SKIP = {"T1": ({"t2_phk_h"}, {"T2PHK_H"}, {"T2PHK_H"}),
                 "T2": ({"t1_geg_h"}, {"T1GEG_H"}, {"T1GEG_H"})}


def _closure(targets: set, parents: dict) -> set:
    need, stack = set(), list(targets)
    while stack:
        n = stack.pop()
        if n not in need:
            need.add(n)
            stack += [q for q in parents.get(n, ()) if q in parents]
    return need


def track_skips(track: str) -> tuple[set, set, set]:
    models_t, roles_t = TRACK_MEMBERS[track]
    ms = json.loads(CM.SPEC.read_text())
    mp = {n["id"]: list(n.get("parents") or []) for n in ms["nodes"]}
    rs = json.loads(RM.SPEC.read_text())
    rp = {n["role"]: list(n["parents"]) for n in rs["nodes"]}
    ap = {r: list(p) for r, p in AF.ALL.items()}
    skips = (set(mp) - _closure(models_t, mp), set(rp) - _closure(roles_t, rp), set(ap) - _closure(roles_t, ap))
    assert skips == EXPECTED_SKIP[track], (track, "track pruning drifted from the graph specs", skips)
    return skips


def run_models(data: Path, out: Path, gpu: int) -> Path:
    work, spec = CM.prepare(str(data), str(out))
    for node in spec["nodes"]:
        CM.run_node(str(data), str(out), node["id"], gpu)
    return work


def run_re_folds(out: Path, gpu: int, skip_roles: frozenset = frozenset()) -> tuple[Path, Path]:
    band = DEC / "Logger/final_band_teachers"
    bs = json.loads(BT.SPEC.read_text())
    BT.prepare(band, bs)
    for j in BT.jobs(bs):
        BT.node(band, bs, j["name"], gpu)
    re3 = DEC / "Logger/final_re3_members"
    rs = json.loads(RM.SPEC.read_text())
    RM.prepare(re3, band, rs)
    for n in RM.nodes(rs):
        if n["role"] in skip_roles:
            print("TRACK_SKIP re3", n["id"], flush=True)
            continue
        RM.run_node(re3, band, rs, n["id"], gpu)
    run_logged("audit_re3_folds", [sys.executable, str(DEC / "verify_fold_graph.py"),
                                   "--kind", "re3", "--work", str(re3),
                                   "--band-work", str(band)]
               + (["--skip-roles", ",".join(sorted(skip_roles))] if skip_roles else []), out,
               marker="FOLD_SOURCE_AUDIT_OK re3")
    return band, re3


def run_re_folds_c(out: Path, gpu: int, skip_roles: frozenset, c_track: str | None) -> tuple[Path, Path]:
    """Re3 graph; in C mode (shared_oof/README.md) then the track's seed-1 replicas and the shared-OOF fleet
    (ready.json), which the base chains wait for before vgain / the band pipeline.  A failure here writes
    ready.failed so a waiting base chain stops instead of waiting out its timeout."""
    try:
        band, re3 = run_re_folds(out, gpu, skip_roles)
        if c_track:
            for fold in SO.folds():
                for role in SO.TRACK_FLEETS[c_track]["B"]:
                    SO.run_replica(re3, SO.work_dir(), fold, role, gpu)
            SO.build_fleet(c_track, re3, SO.work_dir())
        return band, re3
    except BaseException:
        if c_track:
            failed = Path(SO.env_for(c_track)["RPDE_C_FAILED"])
            failed.parent.mkdir(parents=True, exist_ok=True)
            failed.write_text("failed " + datetime.now(timezone.utc).isoformat() + "\n")
        raise


def run_aoa_folds(out: Path, gpu: int, skip_roles: frozenset = frozenset()) -> Path:
    work = DEC / "Logger/final_aoa_folds"
    AF.prepare(work)
    for n in AF.nodes():
        if n["role"] in skip_roles:
            print("TRACK_SKIP aoa", n["id"], flush=True)
            continue
        AF.run_node(work, n["id"], gpu)
    run_logged("audit_aoa_folds", [sys.executable, str(DEC / "verify_fold_graph.py"),
                                   "--kind", "aoa", "--work", str(work)]
               + (["--skip-roles", ",".join(sorted(skip_roles))] if skip_roles else []), out,
               marker="FOLD_SOURCE_AUDIT_OK aoa")
    return work


def model(work: Path, name: str) -> Path:
    p = work / "checkpoints" / (name + "_terminal_ema.pth")
    assert p.is_file(), p
    return p


def write_assets(out: Path, name: str, values: dict[str, Path]) -> Path:
    p = out / "work/assets" / (name + ".json")
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {k: str(v) for k, v in values.items()}
    text = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    if p.exists():assert p.read_text() == text, (name, "asset plan drift")
    else:p.write_text(text)
    return p


def card(track: str, name: str, out: Path, assets: dict[str, Path], *, gpu: int = 0,
         table: Path | None = None, capture: bool | None = None,
         pre_f3: bool = False) -> Path:
    folder = out / "work/cards"
    folder.mkdir(parents=True, exist_ok=True)
    z = folder / (name + ".zip")
    args = [sys.executable, str(DEC / "assemble_card.py"), "--track", track,
            "--assets", str(write_assets(out, name, assets)), "--out", str(z)]
    if capture is None:capture = table is None
    if table is not None:args += ["--table", str(table)]
    if capture:args += ["--capture-card"]
    if pre_f3:args += ["--capture-pre-f3"]
    run_logged("card_" + name, args, out, marker="SOURCE_CARD_BUILT_UNVALIDATED")
    assert z.is_file()
    return z


def require_status(path: Path, status: str = "OK") -> None:
    p = Path(str(path) + ".json")
    assert json.loads(p.read_text())["status"] == status, p


def run_sband2(data: Path, out: Path, band: Path, gpu: int) -> tuple[Path, str]:
    run = "decision_q75s5_15/ship_s0"
    run_logged("ship_campaign", [sys.executable, str(DEC / "write_ship_campaign.py"),
                                 "--work", str(band)], out, marker="SHIP_CAMPAIGN_BOUND")
    cache = out / "work/sband2/cache_ship_s5"
    run_logged("sband2_cache", [sys.executable, str(ROOT / "Track1/U-net/VariantSPSBand2/sb2_cache.py"),
                                "--fold", "ship", "--ship", "--stride", "5", "--gpu", str(gpu),
                                "--campaign", str(band / "ship_campaign.json"),
                                "--expected-variants", str(band / "expected_variants.json"),
                                "--real-dir", str(data / "train_real"), "--stats", str(data / "mean_std_real.pt"),
                                "--out", str(cache)], out, marker="CACHE_OK ship")
    meta = json.loads((cache / "meta.json").read_text())
    assert meta["n"] > 10000 and not meta["variant_mismatch"] and meta["isolation"]["partition_of_all81"]
    cmd = [sys.executable, str(ROOT / "Track1/U-net/VariantSPSBand2/sb2_head.py"),
           "--fold", "ship", "--gpu", str(gpu), "--tag", "decision_q75s5_15", "--seed", "0",
           "--cache", str(cache), "--fleets", "WSW+MIMOWSW,MIMOA+RLB", "--feat", "v1",
           "--coord", "0", "--width", "64", "--layers", "5", "--dil", "1,2,4,2,1",
           "--quantile", "0.75", "--q", "64", "--epochs", "15", "--nproc", "2", "--no-ho"]
    run_logged("sband2_head", cmd, out, marker="HEAD_OK decision_q75s5_15 ship s0")
    head = ROOT / "_tmp/sband2/heads" / run / "head.pth"
    assert head.is_file(), head
    return cache, run


def t1_assets(out: Path, models: Path, t1base: Path, sband: Path, calib: Path,
              h: Path | None = None, sw: Path | None = None) -> dict[str, Path]:
    return {"model.pth": h or model(models, "t1_geg_h"),
            "model_1.pth": sw or model(models, "t2_ptk_sw"),
            "vgain.pth": t1base / "checkpoints/vgain_head_ship_dpv185base.pth",
            "sband.pth": sband, "calib.npz": calib,
            "mean_std_real.pt": out / "mean_std_real.pt"}


def t1_fold_models(re3: Path, aoa: Path, fold: str) -> tuple[Path, Path]:
    if fold.startswith("re"):
        return RM.output(re3, fold, "T1GEG_H"), RM.output(re3, fold, "T2PTK_SW")
    return AF.ckpt(aoa, fold + "_t1geg_h"), AF.ckpt(aoa, fold + "_t2ptk_sw")


def t1_table_stage(out: Path, cluster: Path, lk: Path | None = None,
                   u: Path | None = None) -> Path:
    stage = "cluster_lk_spsr" if u else "cluster_lk" if lk else "cluster"
    target = out / "work/t1_tables" / (stage + ".json")
    cmd = [sys.executable, str(DEC / "fit_t1_runtime_tables.py"),
           "--cluster", str(cluster), "--out", str(target)]
    if lk:cmd += ["--lk", str(lk)]
    if u:cmd += ["--u", str(u)]
    run_logged("t1_tables_" + stage, cmd, out, marker="T1_RUNTIME_TABLES_OK")
    assert json.loads(target.read_text())["stage"] == stage
    return target


def fit_t1_regimes(data: Path, out: Path, models: Path, t1base: Path, re3: Path,
                   aoa: Path, sband: Path, calib: Path, gpu: int) -> Path:
    stats = out / "work/t1_regime_stats"
    (stats / "ke").mkdir(parents=True, exist_ok=True)
    (stats / "mean").mkdir(parents=True, exist_ok=True)
    for fold in ("re0", "re1", "re2", "aoa0", "aoa1"):
        h, sw = t1_fold_models(re3, aoa, fold)
        base = card("T1", "t1_" + fold + "_neutral_regime", out,
                    t1_assets(out, models, t1base, sband, calib, h=h, sw=sw), pre_f3=True)
        for kind, script in (("ke", "ke_stats_probe.py"), ("mean", "mean_stats_probe.py")):
            capture = stats / kind / (fold + "_T1.npz")
            run_logged("t1_regime_" + kind + "_" + fold,
                       [sys.executable, str(ROOT / "common" / script), "--card", str(base),
                        "--fold", fold, "--track", "1", "--gpu", str(gpu),
                        "--work", str(stats / kind / (fold + "_extract")), "--out", str(capture)],
                       out, marker=fold + " 1 OK")
            require_status(capture)
    maps = out / "work/t1_clustermaps"
    maps.mkdir(parents=True, exist_ok=True)
    run_logged("fit_t1_clustermaps",
               [sys.executable, str(ROOT / "Track1/U-net/VariantKEAnchor/cluster_maps_sim.py"),
                "--K", "3", "--seed", "0", "--stats", str(stats / "ke"),
                "--means", str(stats / "mean"), "--out", str(maps)],
               out, marker="K=3 seed=0")
    cluster = maps / "cluster_maps_K3.json"
    doc = json.loads(cluster.read_text())
    assert doc["K"] == 3 and doc["seed"] == 0 and len(doc["centroids"]) == 3
    return cluster


def fit_t1_lk(data: Path, out: Path, models: Path, t1base: Path, re3: Path, aoa: Path,
              sband: Path, calib: Path, cluster: Path, gpu: int) -> Path:
    interim = t1_table_stage(out, cluster)
    captures = out / "work/t1_lk_captures"
    captures.mkdir(parents=True, exist_ok=True)
    for fold in ("re0", "re1", "re2"):
        h, sw = t1_fold_models(re3, aoa, fold)
        z = card("T1", "t1_" + fold + "_cluster_zero_lk", out,
                 t1_assets(out, models, t1base, sband, calib, h=h, sw=sw),
                 table=interim, capture=True, pre_f3=True)
        cap = captures / (fold + ".npz")
        run_logged("t1_lk_capture_" + fold,
                   [sys.executable, str(ROOT / "common/elem_capture_probe.py"),
                    "--card", str(z), "--fold", fold, "--track", "1", "--gpu", str(gpu),
                    "--members", "--work", str(captures / (fold + "_extract")), "--out", str(cap)],
                   out, marker=fold + " OK")
        require_status(cap)
    lk = out / "work/t1_tables/lk_oof.json"
    run_logged("fit_t1_lk_oof", [sys.executable, str(DEC / "fit_t1_lk_oof.py"),
                                 "--captures", str(captures), "--cluster", str(cluster),
                                 "--real-dir", str(data / "train_real"), "--out", str(lk)],
               out, marker="T1_LK_OOF_FIT_OK")
    assert json.loads(lk.read_text())["status"] == "SOURCE_FIT"
    return lk


def t2_assets(out: Path, models: Path, t2base: Path, sband: Path, p1: Path,
              h: Path | None = None, sw: Path | None = None) -> dict[str, Path]:
    return {"model.pth": h or model(models, "t2_phk_h"),
            "model2.pth": sw or model(models, "t2_ptk_sw"),
            "vgain.pth": t2base / "checkpoints/vgain_head_ship_t2repro.pth",
            "sband.pth": sband, "calib.npz": t2base / "calib_sbport_repro.npz",
            "ll_p1.npy": p1}


def t1_sband_assets(out: Path, models: Path, t1base: Path, run: str, gpu: int) -> tuple[Path, Path]:
    base = card("T1", "t1_base_zero_u", out,
                t1_assets(out, models, t1base,
                          t1base / "band_pipeline/cache/sband_dpv185base_band.pth",
                          t1base / "band_pipeline/calib_rebuilt.npz"))
    folder = out / "work/cards"
    rank = folder / "t1_sband2_unstamped.zip"
    run_logged("sband2_rank_swap", [sys.executable, str(ROOT / "Track1/U-net/VariantSPSBand2/sb2_foldcard.py"),
                                  "--run", run, "--gpu", str(gpu), "--base", str(base), "--out", str(rank)],
               out, marker="FOLDCARD_OK")
    stamped = folder / "t1_sband2_stamped.zip"
    run_logged("sband2_restamp", [sys.executable, str(ROOT / "Track1/U-net/VariantSPSBand2/sb2_restamp.py"),
                                "--card", str(rank), "--run", run, "--out", str(stamped)],
               out, marker="RESTAMP_OK")
    assets = out / "work/t1_sband2_assets"
    assets.mkdir(parents=True, exist_ok=True)
    with ZipFile(stamped) as z:
        for name in ("sband.pth", "calib.npz"):
            p = assets / name
            if p.exists():assert p.read_bytes() == z.read(name), (name, "rank-swap drift")
            else:p.write_bytes(z.read(name))
    return assets / "sband.pth", assets / "calib.npz"


def fit_t1_u(out: Path, models: Path, t1base: Path, re3: Path, sband: Path,
             calib: Path, cluster: Path, lk: Path, gpu: int) -> Path:
    interim = t1_table_stage(out, cluster, lk)
    captures = out / "work/spsr_captures"
    captures.mkdir(parents=True, exist_ok=True)
    for fold in ("re0", "re1", "re2"):
        h = RM.output(re3, fold, "T1GEG_H")
        sw = RM.output(re3, fold, "T2PTK_SW")
        z = card("T1", "t1_" + fold + "_zero_u", out,
                 t1_assets(out, models, t1base, sband, calib, h=h, sw=sw),
                 table=interim, capture=True)
        cap = captures / (fold + ".npz")
        run_logged("t1_capture_" + fold,
                   [sys.executable, str(ROOT / "common/elem_capture_probe.py"),
                    "--card", str(z), "--fold", fold, "--track", "1", "--gpu", str(gpu),
                    "--members", "--work", str(captures / (fold + "_extract")), "--out", str(cap)],
                   out, marker=fold + " OK")
        require_status(cap)
    table = out / "work/spsr_tables.json"
    run_logged("fit_spsr_oof", [sys.executable, str(DEC / "fit_spsr_oof.py"),
                                "--captures", str(captures), "--out", str(table), "--device", "cuda"],
               out, marker="SPSR_OOF_FIT_OK")
    assert json.loads(table.read_text())["status"] == "SOURCE_FIT"
    return table


def p1_from_gram(out: Path, z: Path, card_path: Path, fold: str, gpu: int) -> Path:
    gram = out / "work/p1" / (fold + "_ghat.npz")
    gram.parent.mkdir(parents=True, exist_ok=True)
    run_logged("ghat_" + fold,
               [sys.executable, str(ROOT / "common/ll_center_probe.py"),
                "--card", str(card_path), "--fold", fold, "--ghat-out", str(gram),
                "--work", str(out / "work/p1" / (fold + "_extract")), "--gpu", str(gpu)],
               out, marker="ghat " + fold + " OK")
    require_status(gram)
    p1 = out / "work/p1" / (fold + "_p1.npy")
    if fold == "all81":
        cmd = [sys.executable, str(T2DEC / "build_p1.py"), "--ghat", str(gram), "--out", str(p1)]
        label = "p1_all81"
        marker = "P1_OK"
    else:
        cmd = [sys.executable, str(T2DEC / "build_fold_p1.py"), "--ghat", str(gram),
               "--out", str(p1), "--fold", fold]
        label = "p1_" + fold
        marker = "FOLD_P1_OK"
    run_logged(label, cmd, out, marker=marker)
    require_status(p1)
    return p1


def t2_fold_models(re3: Path, aoa: Path, fold: str) -> tuple[Path, Path]:
    if fold.startswith("re"):
        return RM.output(re3, fold, "T2PHK_H"), RM.output(re3, fold, "T2PTK_SW")
    return AF.ckpt(aoa, fold + "_t2phk_h"), AF.ckpt(aoa, fold + "_t2ptk_sw")


def fit_t2_prior(out: Path, models: Path, t2base: Path, sband: Path,
                 re3: Path, aoa: Path, gpu: int) -> tuple[Path, Path]:
    import numpy as np
    boot = out / "work/p1/identity_p1.npy"
    boot.parent.mkdir(parents=True, exist_ok=True)
    if boot.exists():assert np.array_equal(np.load(boot), np.eye(129, dtype=np.float32))
    else:np.save(boot, np.eye(129, dtype=np.float32))
    ship_boot = card("T2", "t2_ship_identity_p1_zero_prior", out,
                     t2_assets(out, models, t2base, sband, boot))
    ship_p1 = p1_from_gram(out, out / "work/p1/ship_ghat.npz", ship_boot, "all81", gpu)
    stats = out / "work/ftl_stats"
    (stats / "ke").mkdir(parents=True, exist_ok=True)
    (stats / "mean").mkdir(parents=True, exist_ok=True)
    for fold in ("re0", "re1", "re2", "aoa0", "aoa1"):
        h, sw = t2_fold_models(re3, aoa, fold)
        idcard = card("T2", "t2_" + fold + "_identity_p1_zero_prior", out,
                      t2_assets(out, models, t2base, sband, boot, h=h, sw=sw))
        fp1 = p1_from_gram(out, out / "work/p1" / (fold + "_ghat.npz"), idcard, fold, gpu)
        fcard = card("T2", "t2_" + fold + "_fold_p1_zero_prior", out,
                     t2_assets(out, models, t2base, sband, fp1, h=h, sw=sw))
        for kind, script in (("ke", "ke_stats_probe.py"), ("mean", "mean_stats_probe.py")):
            capture = stats / kind / (fold + "_T2.npz")
            run_logged("t2_" + kind + "_" + fold,
                       [sys.executable, str(ROOT / "common" / script), "--card", str(fcard),
                        "--fold", fold, "--track", "2", "--gpu", str(gpu),
                        "--work", str(stats / kind / (fold + "_extract")), "--out", str(capture)],
                       out, marker=fold + " 2 OK")
            require_status(capture)
    prior = out / "work/ftl_prior.json"
    run_logged("fit_ftl_prior_oof", [sys.executable, str(T2DEC / "fit_ftl_prior_oof.py"),
                                     "--stats-root", str(stats), "--out", str(prior)],
               out, marker="FTL_PRIOR_OOF_FIT_OK")
    assert json.loads(prior.read_text())["status"] == "SOURCE_FIT"
    return ship_p1, prior


def install_card(z: Path, out: Path, track: str) -> None:
    names = ({"submission.py", "unet_model.py", "vgain_head.py", "sband_head.py",
              "model.pth", "model_1.pth", "vgain.pth", "sband.pth", "calib.npz", "mean_std_real.pt"}
             if track == "T1" else
             {"submission.py", "unet_model.py", "ttt_model.py", "vgain_head.py", "sband_head.py",
              "model.pth", "model2.pth", "vgain.pth", "sband.pth", "calib.npz", "ll_p1.npy"})
    with ZipFile(z) as pack:
        assert set(pack.namelist()) == names
        for name in sorted(names):
            p = out / name
            raw = pack.read(name)
            if p.exists():assert p.read_bytes() == raw, (name, "output collision")
            else:p.write_bytes(raw)
    print("FINAL_SOURCE_OUTPUT_INSTALLED_FOR_INTERFACE_TEST", track, out, flush=True)


def smoke_final_output(z: Path, out: Path, track: str) -> None:
    import numpy as np
    import torch
    import solution

    shape = (4, 20, 32, 64, 3) if track == "T1" else (1, 20, 32, 64, 3)
    x = np.zeros(shape, np.float32)

    def array(value):
        if hasattr(value, "detach"):value = value.detach().cpu().numpy()
        result = np.asarray(value)
        assert result.shape == shape and result.dtype == np.float32 and np.isfinite(result).all()
        return result

    if track == "T1":
        result = solution.load_predictor(str(out), "cuda").predict(x)
        pred = array(result["prediction"])
        low = array(result["lower"])
        high = array(result["upper"])
        assert (low <= high).all()
        check = {"prediction_shape": list(pred.shape), "bounds_ordered": True}
    else:
        model = solution.get_ttt_model(str(out), "cuda")
        model.reset_ttt_state()
        p0, info0 = model.ttt_step(x, None)
        first = array(p0)
        assert (array(info0["lower"]) <= array(info0["upper"])).all()
        model.ttt_step(x, x)
        model.reset_ttt_state()
        p2, _ = model.ttt_step(x, None)
        assert np.array_equal(first, array(p2)), "reset did not restore first-step output"
        check = {"prediction_shape": list(first.shape), "bounds_ordered": True,
                 "causal_previous_target": True, "reset_bitwise": True}
    with ZipFile(z) as pack:
        names = pack.namelist()
    receipt = {"status": "SOURCE_INTERFACE_VALIDATED", "track": track, "card_sha256": sha(z),
               "files": {name: sha(out / name) for name in names}, "interface": check,
               "private_score": None}
    p = out / "final_source_receipt.json"
    assert not p.exists(), "final interface receipt already exists"
    p.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("FINAL_SOURCE_INTERFACE_OK", track, out, flush=True)


def train(track: str, data_dir: str, output_dir: str, device: str) -> None:
    assert track in ("T1", "T2") and str(device) == "cuda"
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    source_guard()
    data = Path(data_dir).resolve(strict=True)
    out = Path(output_dir).resolve(strict=True)
    # Scripts may write only under ROOT or under this train() output dir (/output in the container).
    os.environ["RPDE_FINAL_OUTPUT"] = str(out)
    assert data != out and not out.is_relative_to(data)
    FD.stage(str(data), str(out))
    LL.stage(str(ROOT), str(data), str(out), "dpv185base")
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 1
    allowed = [g for g in range(torch.cuda.device_count()) if g not in BANNED_GPUS]
    assert allowed and allowed[0] == 0
    skip_models, skip_re3, skip_aoa = (frozenset(x) for x in track_skips(track))
    os.environ["RPDE_TRACK_SKIP_MODELS"] = ",".join(sorted(skip_models))   # read by container_models.run_node
    # C (shared_oof/README.md) is OFF unless RPDE_C_SHARED_OOF=1; it was cancelled for the delivered package (2026-10-01).
    c_track = track if os.environ.get("RPDE_C_SHARED_OOF", "0") == "1" else None
    if c_track:
        os.environ.update(SO.env_for(c_track))   # read by both base chains
    print("C_SHARED_OOF", "on" if c_track else "off", flush=True)
    print("TRACK_PRUNING", track, "models", sorted(skip_models), "re3", sorted(skip_re3), "aoa", sorted(skip_aoa),
          flush=True)
    monitor = ResourceMonitor(out)
    monitor.start()
    t2base = None
    # The three independent subgraphs always run concurrently. With >=3 visible GPUs each gets its own GPU; with fewer
    # (the organizer's default single A800) they share GPU allowed[0]: one small U-Net training uses ~40% of a GPU,
    # so three concurrent trainings fit the 24 h budget where a strictly sequential run would not. Recipes, seeds and
    # data are identical either way; the node runners' "GPU must be idle" guard is waived only for this deliberate
    # sharing (RPDE_PIPELINE_SHARES_GPU=1), never for any other caller.
    slots = allowed[:3] if len(allowed) >= 3 else [allowed[0]] * 3
    if len(allowed) < 3:
        os.environ["RPDE_PIPELINE_SHARES_GPU"] = "1"
    mps = SharedGpuMps(out)
    if len(allowed) < 3:
        mps.start()
    else:
        mps.reason = "one GPU per subgraph; not needed"
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            fbase = pool.submit(run_t1_base, out, slots[0])
            fmodel = pool.submit(run_models, data, out, slots[1])
            fre = pool.submit(run_re_folds_c, out, slots[2], skip_re3, c_track)
            models = fmodel.result()
            faoa = pool.submit(run_aoa_folds, out, slots[1], skip_aoa)
            t1base = fbase.result()
            ft2base = (pool.submit(run_t2_base, out, t1base, slots[0]) if track == "T2" else None)
            (band, re3), aoa = fre.result(), faoa.result()
            if ft2base is not None:t2base = ft2base.result()
    finally:
        mps_record = mps.stop()   # the fits below run one at a time
    g = allowed[0]
    if track == "T1":
        assert aoa is not None
        _, run = run_sband2(data, out, band, g)
        sband, calib = t1_sband_assets(out, models, t1base, run, g)
        cluster = fit_t1_regimes(data, out, models, t1base, re3, aoa, sband, calib, g)
        lk = fit_t1_lk(data, out, models, t1base, re3, aoa, sband, calib, cluster, g)
        u = fit_t1_u(out, models, t1base, re3, sband, calib, cluster, lk, g)
        table = t1_table_stage(out, cluster, lk, u)
        final = card("T1", "t1_final_source", out,
                     t1_assets(out, models, t1base, sband, calib), table=table)
    else:
        assert aoa is not None
        if t2base is None:t2base = run_t2_base(out, t1base, g)
        sband = t1base / "band_pipeline/cache/sband_dpv185base_band.pth"
        p1, prior = fit_t2_prior(out, models, t2base, sband, re3, aoa, g)
        final = card("T2", "t2_final_source", out,
                     t2_assets(out, models, t2base, sband, p1), table=prior)
    install_card(final, out, track)
    smoke_final_output(final, out, track)
    peak = monitor.stop()
    summary = {"status": "SOURCE_INTERFACE_VALIDATED", "track": track,
               "started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_s": time.monotonic() - t0,
               "gpu_names": [torch.cuda.get_device_name(i) for i in sorted(set(slots))],
               "gpu_count_used": len(set(slots)), "concurrent_subgraphs": 3, "torch": torch.__version__,
               "track_pruning": {"models": sorted(skip_models), "re3": sorted(skip_re3), "aoa": sorted(skip_aoa)},
               "shared_oof_c": bool(c_track),
               "shared_gpu_mps": mps_record,
               "cuda_runtime": torch.version.cuda,
               "peak_gpu_memory_mib_by_visible_index": peak,
               "resource_samples_sha256": sha(monitor.path),
               "note": "End-to-end source training and interface only; no private-set score is inferred."}
    p = out / "training_summary.json"
    assert not p.exists()
    p.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("FINAL_TRAIN_COMPLETE", track, round(summary["elapsed_s"], 1), flush=True)
