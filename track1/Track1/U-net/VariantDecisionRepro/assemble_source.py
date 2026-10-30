#!/usr/bin/env python3
"""Build a small, source-only Final-container work tree for either track.

This is a curator, not a model builder.  It intentionally excludes historical
submission ZIPs, model checkpoints, caches, logs, and the fitted numerical
tables in the frozen current-card submission.py files.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import make_runtime_template
import yaml

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
T1 = ROOT / "Track1/U-net"
T2 = ROOT / "Track2/U-net"

T1_ROOT = (
    "build_t1_bundle.py", "reproduce_t1.sh", "sband_head.py", "submission_t1_v6.py",
    "submission_t1_v79_riders.py", "t1_band_pipeline.sh", "t1_band_v5.py",
    "t1_boot_calib.py", "t1_sband_cache.py", "t1_sband_fit.py", "t1_sband_pack.py",
    "t1_sps_v2.py", "t1_v8_verify.py", "train_v4.py", "train_vgain.py",
    "unet_model.py", "vgain_head.py",
)
T2_ROOT = (
    "build_t2_bundle.py", "calibrate_t2.py", "calibrate_t2_v2.py",
    "calibrate_t2_v3.py", "make_boot_calib.py",
    "make_calib_sbport.py", "reproduce_t2.sh", "submission_t2_v47_fp16.py",
    "submission_t2_v55_biasfield.py", "t2_collect_residuals.py", "t2_fit_band.py",
    "t2_v6_prep.py", "unet_model.py",
)
COMMON = (
    "final_source_gate.py", "fivefold.py", "fivefold_heads.py", "live_card.py", "repo_paths.py",
    "spatial_distillation.py", "temporal_distillation.py", "terminal_checkpoint.py",
    "ll_center_probe.py",
    "mean_anchor_probe.py", "mean_stats_probe.py", "elem_capture_probe.py",
    "ke_stats_probe.py", "band_width_fivefold.py", "ll_ridge_v155_card.py",
    "make_submission_zip.py",
)
LB = (
    "make_splits.py", "make_splits_aoa.py", "splits_kfold_dedup.json",
    "splits_re3_aoa2_v1.json", "re3_aoa2_data_snapshot_v1.json",
)
SB2 = ("sb2_cache.py", "sb2_head.py", "sb2_lib.py", "sb2_foldcard.py", "sb2_restamp.py")
KE = ("ftl_prior.py", "simulate.py", "clim_sim.py", "mean_anchor_sim.py")


def source_paths() -> set[Path]:
    paths = {T1 / n for n in T1_ROOT} | {T2 / n for n in T2_ROOT}
    paths |= {ROOT / "common" / n for n in COMMON}
    paths |= {ROOT / "common/local_benchmark" / n for n in LB}
    paths |= {T1 / "VariantSPSBand2" / n for n in SB2}
    paths |= {T2 / "VariantAdaptiveKE" / n for n in KE}
    paths |= {ROOT / "docs/decision/live_card_anchors.json"}
    paths |= {T1 / "VariantFiveFoldEvaluation/config/campaign.json"}
    paths |= {ROOT / "common/local_benchmark/splitsel/score_card.py"}
    paths |= set((T1 / "configs").glob("config_v4_*.yaml"))
    for kit in (ROOT / "Track1/realpde_t1_starting_kit_v6",
                ROOT / "Track1/realpde_t1_starting_kit_v9",
                ROOT / "Track2/realpde_t2_starting_kit_v4",
                ROOT / "Track2/realpde_t2_starting_kit_v6"):
        paths |= set(kit.glob("*.py"))
    for variant in (HERE, ROOT / "Track2/U-net/VariantDecisionRepro"):
        paths |= set(variant.glob("*.py"))
    paths |= {HERE / "config/models.json", HERE / "config/band_teachers.json",
              HERE / "config/re3_members.json",
              ROOT / "Track2/U-net/VariantDecisionRepro/config/ftl_prior_recipe.json"}
    paths |= {T1 / "VariantRetrainPush/train_objective.py",
              T1 / "VariantKEAnchor/cluster_maps_sim.py"}
    for spec in (HERE / "config/models.json", HERE / "config/band_teachers.json", HERE / "config/re3_members.json"):
        data = json.loads(spec.read_text())
        if spec.name == "re3_members.json":
            for fold in data["folds"]:
                for item in data["nodes"]:
                    paths.add(T1 / "VariantRetrainPush/config" / fold / (item["role"] + ".yaml"))
        elif spec.name == "band_teachers.json":
            for fold in data["folds"]:
                for role in data["roles"]:
                    paths.add(ROOT / data["source_template"].format(fold=fold, role=role))
        else:
            paths |= {ROOT / n["source"] for n in data["nodes"]}
    for fold in ("aoa0", "aoa1"):
        for role in ("RLB", "WSW", "MIMOA", "MIMOWSW"):
            paths.add(T1 / "VariantFiveFoldEvaluation/config" / fold / "outer/LIVE" / (role + ".yaml"))
        for role in ("TST5_WSW", "TST5_MIMOA", "TST5_MIMOWSW", "T2RA3_H", "T2RA3_SW",
                     "T1GEG_H", "T2PHK_H", "T2PTK_SW"):
            paths.add(T1 / "VariantRetrainPush/config" / fold / (role + ".yaml"))
    for fold in ("re0", "re1", "re2", "aoa0", "aoa1"):
        paths.add(T1 / "VariantFiveFoldEvaluation/config/manifests" / (fold + "_outer.json"))
    return paths


def validate_static_refs(context: Path) -> None:
    missing = []
    for config in context.rglob("*.yaml"):
        doc = yaml.safe_load(config.read_text())
        def walk(value):
            if isinstance(value, dict):
                for item in value.values():walk(item)
            elif isinstance(value, list):
                for item in value:walk(item)
            elif isinstance(value, str) and value.endswith((".json", ".yaml")) \
                    and value.startswith(("Track1/", "Track2/", "common/")):
                if not (context / value).is_file():
                    missing.append((str(config.relative_to(context)), value))
        walk(doc)
    assert not missing, ("source YAML references absent static files", missing)


# Imported somewhere in the context but deliberately absent; each is off every Final train/inference path.
IMPORT_ALLOW = {
    "finalfit_heads": "fivefold_heads imports it only for a head plan with purpose == 'final_fit'; "
                      "the Final chain passes no --strict-plan to train_vgain.py",
    "solver_ops": "lazy import inside the solver-in-the-loop block of unet_model.py; no Final member uses it",
    "einops": "only the official kits' smoke_test_kit.py (vendored einops, not packaged); not run by train()",
}


def imported_names(path: Path) -> set[str]:
    """Top-level module names a file imports anywhere (module level, functions, try blocks)."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text(errors="replace"))):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:                      # relative: `from .x import y` / `from . import y`
                names |= {node.module.split(".")[-1]} if node.module else {a.name for a in node.names}
            elif node.module:
                names.add(node.module.split(".")[0])
    return names


def _import_node_names(node) -> set[str]:
    if isinstance(node, ast.Import):
        return {a.name.split(".")[0] for a in node.names}
    if isinstance(node, ast.ImportFrom):
        if node.level:
            return {node.module.split(".")[-1]} if node.module else {a.name for a in node.names}
        return {node.module.split(".")[0]} if node.module else set()
    return set()


def required_module_imports(path: Path) -> set[str]:
    """Names a file needs at import time: module-level imports (also inside module-level if/with/for), outside
    function and class bodies.  In a try block a name counts only if an except handler imports it too (the
    relative-then-absolute fallback of fivefold_heads); a try-only import is optional."""
    def walk(stmts, out):
        for s in stmts:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(s, (ast.Import, ast.ImportFrom)):
                out |= _import_node_names(s)
            elif isinstance(s, ast.Try):
                body, handlers = set(), set()
                walk(s.body, body)
                for h in s.handlers:
                    walk(h.body, handlers)
                out |= body & handlers
                out |= handlers - body
                walk(s.orelse, out); walk(s.finalbody, out)
            else:
                for field in ("body", "orelse"):
                    if isinstance(getattr(s, field, None), list):
                        walk(getattr(s, field), out)
        return out
    return walk(ast.parse(path.read_text(errors="replace")).body, set())


def validate_import_closure(context: Path, repo: Path | None = None) -> None:
    """Fail when a packaged file imports a module that the context does not carry.

    With `repo` (assembly on a development host): a name is missing when the repository has a .py of that name
    (outside Logger/_archive/_tmp/_vendor) and the context has none (every import, also lazy ones).  Without it
    (inside the Final container): a name is missing when a module-level required import (required_module_imports;
    lazy imports of other architectures are skipped) has no file or package in the context and the environment
    cannot import it.  The official starting kits are not run by train() and are skipped in container mode.
    2026-09-29: the v8/v9 contexts lacked common/fivefold.py and failed at base-chain stage 4/9 after 9.6 h.
    """
    local = {p.stem for p in context.rglob("*.py")} | {p.name for p in context.rglob("*") if p.is_dir()}
    names = {}
    for f in context.rglob("*.py"):
        rel = f.relative_to(context)
        if repo is None and any(part.startswith("realpde_t") and "starting_kit" in part for part in rel.parts):
            continue
        for n in (imported_names(f) if repo is not None else required_module_imports(f)):
            names.setdefault(n, []).append(str(rel))
    if repo is not None:
        skip = ("Logger", "_archive", "_tmp", "_vendor", "__pycache__", "Ref_code")
        repo_stems = {p.stem for base in ("common", "Track1", "Track2") for p in (repo / base).rglob("*.py")
                      if not any(s in p.parts for s in skip)}
        missing = {n: u for n, u in names.items() if n in repo_stems and n not in local and n not in IMPORT_ALLOW}
    else:
        missing = {}
        for n, u in names.items():
            if n in local or n in IMPORT_ALLOW or n in sys.builtin_module_names:
                continue
            try:
                found = importlib.util.find_spec(n) is not None
            except (ImportError, ValueError):
                found = False
            if not found:
                missing[n] = u
    assert not missing, ("packaged files import modules absent from the source context",
                         {n: sorted(u)[:3] for n, u in sorted(missing.items())})


def assemble(track: str, out: Path) -> None:
    variant = ROOT / ("Track1" if track == "T1" else "Track2") / "U-net/VariantDecisionRepro"
    assert out.is_relative_to(variant / "Logger") and not out.exists()
    out.mkdir(parents=True)
    paths = source_paths()
    for p in sorted(paths):
        assert p.is_file() and p.is_relative_to(ROOT), p
        dst = out / p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    # The two legacy scripts derive their old root from their original location;
    # the Final copies accept an explicit /workspace root instead.
    t1_final = HERE / "config/final_scripts/reproduce_t1_final.sh"
    band_final = HERE / "config/final_scripts/t1_band_pipeline_final.sh"
    t2_final = ROOT / "Track2/U-net/VariantDecisionRepro/config/final_scripts/reproduce_t2_final.sh"
    assert t1_final.is_file() and band_final.is_file() and t2_final.is_file()
    shutil.copy2(t1_final, out / "Track1/U-net/reproduce_t1.sh")
    shutil.copy2(band_final, out / "Track1/U-net/t1_band_pipeline.sh")
    shutil.copy2(t2_final, out / "Track2/U-net/reproduce_t2.sh")
    # Replace the embedded fitted arrays with explicit train-time placeholders.
    for which in ("T1", "T2"):
        src = make_runtime_template.SOURCES[which]
        template, _ = make_runtime_template.strip(which, src.read_text())
        folder = out / src.parent.relative_to(ROOT)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "submission.py").write_text(template)
        for p in src.parent.glob("*.py"):
            if p.name != "submission.py":
                shutil.copy2(p, folder / p.name)
    sol = variant / "solution.py"
    shutil.copy2(sol, out / "solution.py")
    shutil.copy2(HERE / "final_pipeline.py", out / "final_pipeline.py")
    docker = variant / "config" / ("Dockerfile.t1" if track == "T1" else "Dockerfile.t2")
    shutil.copy2(docker, out / "Dockerfile")
    # These two files are required by the organizer and describe this exact
    # source package.  They are not handoff notes or a generic kit template.
    for name in ("README", "METHOD"):
        shutil.copy2(variant / "config" / (name + ".final.md"), out / (name + ".md"))
    validate_static_refs(out)
    validate_import_closure(out, ROOT)
    subprocess.run([sys.executable, str(HERE / "make_source_manifest.py"), "--root", str(out),
                    "--out", str(out / "source_manifest.json")], check=True)
    print("SOURCE_CONTEXT_WORKING_ONLY", track, len(paths), out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=("T1", "T2"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assemble(a.track, Path(a.out).resolve())


if __name__ == "__main__":
    main()
