#!/usr/bin/env bash
# =====================================================================================
#  Track 2: raw data -> shipping zip, from a CLEAN CHECKOUT. No historical artifact.
# =====================================================================================
#
# WHY THIS EXISTS.  Same reason as `Track1/U-net/reproduce_t1.sh`: the Decision Phase re-trains each
# shortlisted method from scratch, so anything that cannot be rebuilt from a clean tree is a
# liability regardless of its leaderboard score.
#
# WHAT THE AUDIT FOUND (2026-07-29).  T2's build command took its forecaster from
#
#     CK="$D/submissions/t2_unet_v4_online/model.pth"
#
# -- a weights file lifted out of a PREVIOUSLY BUILT SUBMISSION FOLDER, with no recorded link to any
# training run. That is worse than T1's derived calibration, because the thing without provenance is
# the model. Traced by md5: it is bit-identical to `checkpoints/t1_u-net/unet_v2_b64.pth`, the v2
# base-64 baseline (`arch unet2d_framestacked`, base 64, real-finetune gstep 12501,
# val_rel_l2 0.07275051856167102). So the lineage IS recoverable, and stage 1 below rebuilds it from
# its config instead of copying it out of a zip.
#
# WORTH KNOWING WHILE READING THIS: that means T2 ships a GENERATION-OLD forecaster (v2 base-64 with
# the base-64 head) while T1 ships v4 base-96. T2's timing has enormous slack -- 0.89 ms/step against
# a 10-minute budget -- so the smaller model is not buying anything. `remote_runs/t2up_work.sh`
# probes the upgrade; if it wins, change SHIP_CFG/HEAD here and refit the calibration, in that order.
#
# ACCEPTANCE TEST.  Stage 5 asserts FINAL_LOCAL is within 0.10 of the shipped v7's 86.746, and
# separately asserts the CUDA graph survived capture -- a bundle that silently falls back to eager is
# still correct but no longer the thing that was measured.
#
#   bash reproduce_t2.sh [GPU]
# =====================================================================================
#
# XNXa (2026-08-04). Defaults flipped to the champion recipe t2_unet_v16 ships (ship_XNXa
# forecaster + XNXa head, online 83.85439 on 2026-08-04); TARGET_FINAL 87.044 is the v16
# build's own FINAL_LOCAL. Historical note below kept for the audit trail.
#
# BASE-96 (2026-07-29). This script now rebuilds the base-96 bundle (t2_unet_v8), because the
# upgrade the header below anticipated was measured and won: 86.860 vs 86.745, four subscores up,
# only `time` down 0.032. The forecaster is bit-identical to the one t1_unet_v11 ships, i.e. already
# validated online at 84.510 on Track 1.
#
# OPEN GAP, stated rather than hidden: calib_v7.npz is derived from calib_v4.npz, which was fitted
# against BASE-64 residuals and has no reconstruction script (t2_v6_prep.py only derives v7 FROM
# v4). The rebuilt bundle therefore carries a calibration matched to a forecaster it no longer uses.
# That is headroom left on the table, not an error -- the +0.115 was measured with this exact
# mismatch in place -- but refitting it is the next Track-2 task.
set -euo pipefail

GPU="${GPU:-${1:-0}}"
# Derived (2026-08-12, same clone-escape defect as reproduce_t1.sh).  Clone runs MUST set
# REPRO_SCOPED=1: tracked configs carry absolute production checkpoint paths.
R=${RPDE_ROOT:?set RPDE_ROOT to the source-tree root}
D=$R/Track2/U-net
T1=$R/Track1/U-net
# DPBOOTSTRAP (2026-09-12, docs/decision/RECIPE.md open item 5): the default used to be a
# hard-coded host path that exists on neither host this chain is run on, so a caller who forgot PY=
# got "No such file or directory" from the very first live_card.py call and then "LIVE_NAME: unbound
# variable" from `set -u` -- a diagnostic that points at the wrong file.  Detect the two real hosts and
# REFUSE with a usable message if neither is present.  PY= still overrides everything.
if [ -z "${PY:-}" ]; then
  for _c in "$HOME/miniconda3/envs/FVGN-pt2.8/bin/python" \
            "$HOME/miniconda3/envs/FVGN-pt2.7/bin/python" \
            "$HOME/miniconda3/bin/python"; do
    [ -x "$_c" ] && { PY="$_c"; break; }
  done
fi
[ -n "${PY:-}" ] && [ -x "$PY" ] || {
  echo "REPRO_GUARD_FAIL -- no usable python.  Set PY= to this host's FVGN env "
  echo "                  (for example \$HOME/miniconda3/envs/FVGN-pt2.8/bin/python "
  echo "                   or \$HOME/miniconda3/envs/FVGN-pt2.7/bin/python)"; exit 1; }
OUT=${RPDE_T2_BASE_OUT:?set RPDE_T2_BASE_OUT to a fresh output directory}
mkdir -p "$OUT"
cd "$D"

# 2026-08-10: every line here was HARD-ASSIGNED to XNXa, whose stage sources include
# data/extra_paper_v5 -- a DIFFERENT SOURCE, banned by the 2026-08-05 rules.  Running this script
# would have rebuilt an ILLEGAL model in the Decision Phase.  All overridable now.
# ---------------------------------------------------------------- THE LIVE CARD, DERIVED
# 2026-08-19, defect class D6.  Everything below used to be a hand-maintained snapshot, and it
# was SIX generations stale: this chain named SDa_lre/LRe_kf/vgain_head_lre while the card ships
# the CW128+WSW cross-recipe blend, and ALL FOUR anchors missed the live row by
# 0.133/0.510/0.168/1.353.  Unlike T1 this chain had NO md5 guard at all, so nothing would have
# announced it.  Now derived from LIVE_CARD and md5-checked against the shipped zip.
# ---------------------------------------------------------------- DPBOOTSTRAP (2026-09-12)
# See reproduce_t1.sh: Track*/submissions/ is gitignored, so a clean clone -- and the Decision-Phase
# environment, which accepts training code only -- has no live zip, and this line used to die here.
# BOOTSTRAP=1 takes the card's identity from the committed docs/decision/live_card_anchors.json.
LC_BOOT=""
if [ "${BOOTSTRAP:-0}" = "1" ]; then
  LC_BOOT="--bootstrap"
  [ "${REPRO_SCOPED:-0}" = "1" ] || { echo "REPRO_GUARD_FAIL -- BOOTSTRAP=1 requires REPRO_SCOPED=1: with no live zip the members and head must be OUTPUTS of this run"; exit 1; }
  # THE SILENT-SKIP TRAP.  SHIP_SBAND defaults to $LIVE_SBAND, which bootstrap emits EMPTY, and the
  # SBPORT stage below is guarded by `[ -n "$SHIP_SBAND" ]` -- so an unwired bootstrap run would print
  # "(live card ships no sband head -- SBPORT stage skipped)" and package a card WITHOUT the SBPORT
  # calib the live card actually ships.  The live T2 card does ship one, so refuse instead of skipping.
  [ -n "${SHIP_SBAND:-}" ] || { echo "REPRO_GUARD_FAIL -- BOOTSTRAP=1 needs SHIP_SBAND wired to the T1 chain's sband output; the live t2_v58_alldata DOES ship an sband head, and an empty SHIP_SBAND would silently skip the SBPORT stage and package a different card"; exit 1; }
  echo "  [bootstrap] live-card identity from docs/decision/live_card_anchors.json -- no zip required"
  echo "  [bootstrap] the md5-against-zip guard cannot run; acceptance is the four accuracy anchors"
fi
eval "$($PY "$R/common/live_card.py" --track 2 --emit shell $LC_BOOT)" || {
  echo "REPRO_GUARD_FAIL -- common/live_card.py could not resolve the live card"; exit 1; }
echo "LIVE CARD: $LIVE_NAME  zip md5 $LIVE_ZIP_MD5"
[ -n "${LIVE_DRIFT:-}" ] && echo "  !! NEWER card-shaped bundles exist ($LIVE_DRIFT). If one of
     them is now on the board, update Track2/LIVE_CARD before trusting this run."

# T2 ships the T1 forecaster, so the lineage is T1's: CW128 (convnext, ships as model.pth)
# distilled from the RLB128 teacher, blended with WSW (selfwarp, ships as model2.pth).
# ---------------------------------------------------------------- DP-CHAIN-2GEN (2026-09-09)
# The live T2 card (v56; t2_v58_alldata since 2026-09-10 -- ALL81=1 rebuilds that generation in stage 3d on top of this one)
# ships EDISTMSH + EDISTMSSW (distilled from ship WSW + MIMOA + MIMOWSW, themselves
# from RLB128) with the CW128_kf-fitted head and the sbport band unchanged.  Until this edit the chain rebuilt
# the CW128+WSW class and hard-coded a band-only ONLINE literal (the JS-gated pyramid bias was absent).
GEN2=${GEN2:-1}
if [ "$GEN2" = "1" ]; then
  SHIP_CFG=${SHIP_CFG:-$T1/configs/config_v4_ship_EDISTMSH.yaml}
  SECOND_CFG=${SECOND_CFG:-$T1/configs/config_v4_ship_EDISTMSSW.yaml}
  WSW_SHIP_CFG=${WSW_SHIP_CFG:-$T1/configs/config_v4_ship_WSW.yaml}
  MIMOA_SHIP_CFG=${MIMOA_SHIP_CFG:-$T1/configs/config_v4_ship_MIMOA.yaml}
  MIMOWSW_SHIP_CFG=${MIMOWSW_SHIP_CFG:-$T1/configs/config_v4_ship_MIMOWSW.yaml}
fi
# ---------------------------------------------------------------- ALL81 (2026-09-10)
# LIVE_CARD moved to t2_v58_alldata (2026-09-10): the v57 recipe (v55 template, v56 ONLINE dict, XCB head,
# sband_xcb, calib_sbport_q64) with the three ship teachers (WSW, MIMOA, MIMOWSW -- each still distilled from the
# 72-file RLB128) AND the two students (EDISTMSH, EDISTMSSW) retrained on the 81-file pool = local_bench/train (72) +
# local_bench/val (9, Re 11400/17775); 7575_0 stays banned.  The config delta is exactly {real-finetune glob -> the
# pool, exclude_re -> []} plus names/paths.  The card was built with
# --allow-calib-mismatch.
# DESIGN RULE: stage 2 (head), stage 3 (calib_v7), stage 3b (M3 band: residuals collected through t2_boot) and stage
# 3c (SBPORT merge) stay on the 72-file members / the declared inputs -- 3b collects residuals over the 9 interp files,
# which are IN-SAMPLE for an 81-file member.  Stage 3d trains the 81-file members; stage 4 swaps them in at PACKAGING
# ONLY.  ALL81=0 restores the v56-generation packaging, whose anchors no longer match a v58 LIVE_CARD.
ALL81=${ALL81:-1}
if [ "$ALL81" = "1" ] && [ "$GEN2" != "1" ]; then
  echo "REFUSING: ALL81=1 needs GEN2=1 -- stage 3d retrains the GEN2 ship teachers and students on 81 files"; exit 1
fi
SHIP_CFG=${SHIP_CFG:-$T1/configs/config_v4_ship_CW128.yaml}
SECOND_CFG=${SECOND_CFG:-$T1/configs/config_v4_ship_WSW.yaml}   # the blend's 2nd member
TEACHER_CFG=${TEACHER_CFG:-$T1/configs/config_v4_ship_RLB128.yaml}   # BOTH members distil from it
FOLD_CFG=${FOLD_CFG:-$T1/configs/config_v4_CW128_kf.yaml}        # 6 dedup-Reynolds folds for the head OOF
TEACHER_FOLD_CFG=${TEACHER_FOLD_CFG:-$T1/configs/config_v4_RLB_kf.yaml}
ARM=${ARM:-unet_v4_CW128_kf_base128}
FOLDS=${FOLDS:-"0 1 2 3 4 5"}
MANIFEST=${MANIFEST:-$R/common/local_benchmark/splits_kfold_dedup.json}
T1CKPT=${T1CKPT:-$R/checkpoints/t1_u-net}
# DERIVED.  Was vgain_head_lre_w0100.pth -- a base-96 LRe-era head; the card ships the blend's
# OWN refit head (md5 1abc682b), which is also what Track 1 ships.
HEAD=${HEAD:-$LIVE_HEAD}
HEAD_W=${HEAD_W:-64}; HEAD_GMIN=${HEAD_GMIN:-0.5}; HEAD_GMAX=${HEAD_GMAX:-3.0}
HEAD_W_TKE=${HEAD_W_TKE:-0.10}
# DERIVED.  Was v31_sgate23; the card ships submission_t2_v47_fp16.py (= v44_xcb + the _Half
# boundary-casting wrapper).  The blend template is also what consumes model2.pth, so the
# stale value silently disabled the second member.
TEMPLATE=${TEMPLATE:-$LIVE_TEMPLATE}
# 2026-08-12: the live card is the FROZEN pipeline (rates {}). The v22-era rates measured
# -0.060 locally (bias_beta/sp_beta are the harmful pair) and were dropped online 2026-08-11
# as t2_v27_nocal (+0.225); the M3-reproduced calib on the same build is the live #1 (v29).
# DERIVED (2026-09-09): the ONLINE dict the LIVE card was built with (common/live_card.py reads it from the card's
# verify_report.json).  The 2026-08-12 literal below is kept ONLY as the GEN2=0 fallback; under GEN2 an empty
# LIVE_ONLINE is a REFUSAL, never a silent fallback to a band-only card.
if [ "$GEN2" = "1" ]; then
  [ -n "${LIVE_ONLINE:-}" ] || { echo "REPRO_GUARD_FAIL -- live_card.py emitted no LIVE_ONLINE for $LIVE_NAME"; exit 1; }
  ONLINE=${ONLINE:-$LIVE_ONLINE}
  echo "  ONLINE (derived from the live card): $ONLINE"
fi
ONLINE=${ONLINE:-'{"band_beta":0.1,"warm":16}'}
HALFWIDTH=0.15
# 2026-08-12: acceptance is ACCURACY-ANCHORED. FINAL_LOCAL carries the local `time` subscore,
# whose contention noise was measured at 15x any usable tolerance (two FINAL_LOCAL outliers at 81.9
# in the 8-seed rebuild run were pure local time noise), so it is printed but never gated on.
# Anchors = the v29 build (the live #1 card): trio bit-identical across the calib swap, sps 52.572.
# DERIVED from the card's own verify_report.json.  All four of the old anchors were wrong.
TARGET_REL=${TARGET_REL:-$LIVE_REL};  TOL_REL=${TOL_REL:-0.10}
TARGET_TKE=${TARGET_TKE:-$LIVE_TKE};  TOL_TKE=${TOL_TKE:-0.30}
TARGET_MVPE=${TARGET_MVPE:-$LIVE_MVPE}; TOL_MVPE=${TOL_MVPE:-0.10}
TARGET_SPS=${TARGET_SPS:-$LIVE_SPS};  TOL_SPS=${TOL_SPS:-0.30}

if [ "${REPRO_SCOPED:-0}" = "1" ]; then
  # Clean-room / clone mode, mirroring reproduce_t1.sh: nothing below may read or write a
  # production checkpoint.  REQUIRED for any clone run -- the tracked configs carry absolute
  # production checkpoint paths and this block is what rewrites them.
  mkdir -p "$OUT/checkpoints"
  _gen2_argv=""; [ "$GEN2" = "1" ] && _gen2_argv="$WSW_SHIP_CFG $MIMOA_SHIP_CFG $MIMOWSW_SHIP_CFG"
  PYY=$PY $PY - "$SHIP_CFG" "$FOLD_CFG" "$TEACHER_CFG" "$TEACHER_FOLD_CFG" "$OUT" "$R" "$SECOND_CFG" $_gen2_argv <<'SCOPEEOF'
import sys, yaml, pathlib
ship, fold, tship, tfold, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], pathlib.Path(sys.argv[5])
root = sys.argv[6].rstrip("/")
second = sys.argv[7]   # the blend's 2nd member must be scoped too, or the clean room trains
                       # it straight into the PRODUCTION checkpoints dir.
gen2 = sys.argv[8:11] if len(sys.argv) >= 11 else None   # GEN2: wsw_ship, mimoa_ship, mimowsw_ship (teachers of the students)
ck = out / "checkpoints"
# F1 (portability): the tracked configs store every path (stats, val_glob, stages[*].sources[*].{glob,stats},
# kfold_manifest, kit_dir, ...) relative to the repo root.  Scoping resolves EVERY one of them onto THIS
# checkout's root ($R), comma-joined multi-teacher distill_ckpt included; checkpoint/distill_ckpt then get their
# dedicated $OUT handling below.  The F1 assert refuses a scoped config that names any absolute path outside $R.
sys.path.insert(0, root + "/common")
from repo_paths import resolve_paths
def reroot(x):
    return resolve_paths(x, root)
def outside_root(x):
    if isinstance(x, dict):
        return [p for v in x.values() for p in outside_root(v)]
    if isinstance(x, list):
        return [p for v in x for p in outside_root(v)]
    if isinstance(x, str):
        return [p for p in x.split(",") if p.startswith("/") and not p.startswith(root + "/")]
    return []
def dump_checked(c, path):
    yaml.safe_dump(c, open(path, "w"), sort_keys=False)
    bad = outside_root(yaml.safe_load(open(path)))
    assert not bad, f"F1 assert: {path} names path(s) outside {root}: {bad[:3]}"
def scope(src, nm):
    c = reroot(yaml.safe_load(open(src)))
    stem = pathlib.Path(c["checkpoint"]).stem + "_t2" + out.name
    c["checkpoint"] = str(ck / (stem + ".pth"))
    c["variant"] = str(c.get("variant", "v")) + "_t2" + out.name
    dump_checked(c, out / nm)
    print("  scoped:", nm, "->", c["checkpoint"])
    return c
def student(src, nm, teachers):
    c = scope(src, nm)
    if c.get("distill_weight", 0.0) and c.get("distill_ckpt"):
        old = str(c["distill_ckpt"]); n_old = len([x for x in old.split(",") if x])
        assert n_old == len(teachers), f"{nm}: {n_old} teachers in the tracked config, {len(teachers)} scoped -- refusing"
        suf = "_f{fold}" if "{fold}" in old else ""
        c["distill_ckpt"] = ",".join(str(pathlib.Path(t["checkpoint"]).parent / (pathlib.Path(t["checkpoint"]).stem + suf + ".pth")) for t in teachers)
        dump_checked(c, out / nm); print("    distill_ckpt ->", c["distill_ckpt"])
    return c
t_ship = scope(tship, "cfg_teacher.yaml")
t_fold = scope(tfold, "cfg_teacher_fold.yaml")
student(fold, "cfg_fold.yaml", [t_fold])
if gen2 is None:
    student(ship, "cfg_ship.yaml", [t_ship]); student(second, "cfg_second.yaml", [t_ship])
else:
    t_wsw = student(gen2[0], "cfg_t_wsw.yaml", [t_ship]); t_mimoa = student(gen2[1], "cfg_t_mimoa.yaml", [t_ship]); t_mimowsw = student(gen2[2], "cfg_t_mimowsw.yaml", [t_ship])
    student(ship, "cfg_ship.yaml", [t_wsw, t_mimoa, t_mimowsw]); student(second, "cfg_second.yaml", [t_wsw, t_mimoa, t_mimowsw])
SCOPEEOF
  SHIP_CFG=$OUT/cfg_ship.yaml
  SECOND_CFG=$OUT/cfg_second.yaml
  FOLD_CFG=$OUT/cfg_fold.yaml
  TEACHER_CFG=$OUT/cfg_teacher.yaml
  TEACHER_FOLD_CFG=$OUT/cfg_teacher_fold.yaml
  if [ "$GEN2" = "1" ]; then WSW_SHIP_CFG=$OUT/cfg_t_wsw.yaml; MIMOA_SHIP_CFG=$OUT/cfg_t_mimoa.yaml; MIMOWSW_SHIP_CFG=$OUT/cfg_t_mimowsw.yaml; fi
  ARM=$($PY -c "import yaml,pathlib;print(pathlib.Path(yaml.safe_load(open('$OUT/cfg_fold.yaml'))['checkpoint']).stem)")
  T1CKPT=$OUT/checkpoints
  HEAD=$OUT/checkpoints/vgain_head_ship_t2repro.pth
  M3_CALIB=$OUT/calib_m3_repro.npz
  M3_CACHE=$OUT/m3_cache
  echo "REPRO_SCOPED: ARM=$ARM T1CKPT=$T1CKPT"
fi

say () { echo; echo "=============== $* ==============="; date; }
skip () { [ -e "$1" ] && { echo "  [have] $1 -- skipping"; return 0; } || return 1; }
# cfg_path <config> <key>: one config value with its repo-relative path(s) resolved onto THIS checkout
# (tracked configs store paths relative to the repo root; common/repo_paths.py).
cfg_path () { "$PY" -c "import sys,yaml; sys.path.insert(0, sys.argv[3]); from repo_paths import resolve_path; print(resolve_path(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]]))" "$1" "$2" "$R/common"; }

# ------------------------------------------------------------------ 0b. local benchmark dirs (F3)
# Every ship config's val_glob and both bundle builders read data/RealPDE_competition_dataset/gdrive_raw/local_bench/{train,val}.
# Those are MATERIALIZED (kit-style symlink eval dirs) by common/local_benchmark/make_splits.py from the
# extracted release -- they do not ship with the repo, so a clean clone must build them here.
if [ ! -e "$R/data/RealPDE_competition_dataset/gdrive_raw/local_bench/train/test_real" ] || [ ! -e "$R/data/RealPDE_competition_dataset/gdrive_raw/local_bench/val/test_real" ]; then
  if [ ! -d "$R/data/RealPDE_competition_dataset/gdrive_raw/extracted/train_real/train_real" ] || [ ! -d "$R/data/RealPDE_competition_dataset/gdrive_raw/extracted/train_sim/train_sim" ]; then
    echo "MISSING DATA: expected the competition release extracted at exactly"
    echo "  $R/data/RealPDE_competition_dataset/gdrive_raw/extracted/train_real/train_real/*.h5   (82 real trajectories)"
    echo "  $R/data/RealPDE_competition_dataset/gdrive_raw/extracted/train_sim/train_sim/*.h5     (100 sim trajectories)"
    echo "Place train_real.tar.gz / train_sim.tar.gz contents there (see ENV.md section 3), then rerun."
    exit 1
  fi
  say "0b  materializing data/RealPDE_competition_dataset/gdrive_raw/local_bench/{train,val} (common/local_benchmark/make_splits.py)"
  $PY "$R/common/local_benchmark/make_splits.py"
fi
# 0b-stats (2026-09-10): the kit-style eval dirs also need <data_dir>/mean_std_real.pt.  Track2/U-net/build_t2_bundle.py
# (Normalizer(data_dir / "mean_std_real.pt")), Track2/U-net/t2_collect_residuals.py and the T2 kit's local_eval.py read it
# there, and make_splits.py does not create it.  The test above checks only test_real, so a pre-built tree without the
# stats file skipped this stage and stage 3b died with FileNotFoundError (clean clone, 2026-09-10).  Link it to the
# T2 kit stats, the same file build_t2_bundle.py's STATS names.  A dangling link is refused, never replaced.
_KSTATS_0B="$R/Track2/realpde_t2_starting_kit_v4/example_data/mean_std_real.pt"
for _lb0b in train val; do
  _st0b="$R/data/RealPDE_competition_dataset/gdrive_raw/local_bench/$_lb0b/mean_std_real.pt"
  if [ -L "$_st0b" ] && [ ! -e "$_st0b" ]; then echo "DANGLING LINK $_st0b -- remove it and rerun"; exit 1; fi
  if [ ! -e "$_st0b" ]; then
    [ -e "$_KSTATS_0B" ] || { echo "MISSING $_KSTATS_0B (the T2 starting kit stats)"; exit 1; }
    [ -d "$R/data/RealPDE_competition_dataset/gdrive_raw/local_bench/$_lb0b" ] || { echo "MISSING $R/data/RealPDE_competition_dataset/gdrive_raw/local_bench/$_lb0b"; exit 1; }
    ln -s "$_KSTATS_0B" "$_st0b"
    say "0b  linked $_st0b -> $_KSTATS_0B"
  fi
done

# ------------------------------------------------------------------ 1. the forecaster
say "1/5  forecaster, rebuilt from its config (NOT copied out of a submission folder)"
# SDa self-distils from the RLa TEACHER ship model, so a clean tree must train the teacher
# FIRST (stage 2 already does this for the fold fleets; stage 1 was missing it -- found by the
# 2026-08-12 RPXa scoped-clone rerun, which died FileNotFoundError on the scoped teacher ckpt.
# The production box never exposed it because unet_v4_ship_RLa.pth pre-exists there).
TCKPT=$(cfg_path "$TEACHER_CFG" checkpoint)
# FINAL MODE (RPDE_FINAL_MODE=1, set only by final_pipeline.run_t2_base; 2026-09-30):
#  B -- the T1 base chain of the same train() already trained this exact recipe (T1 stage 3a; scoped configs differ
#       only in checkpoint/variant); final_reuse_draw.py verifies that and copies its checkpoint so `skip` finds it.
#  A -- the 72-file ship teachers, ship student and second member below feed only the M3 band (3b) and the stage-4
#       packaging bundle, neither of which reaches the Final card (it takes vgain from stage 2 and calib_sbport from
#       3d/3e); they are not trained.  RLB128 itself stays: stage 3d distils the ALL81 teachers from it.
if [ "${RPDE_FINAL_MODE:-0}" = "1" ] && [ -n "${RPDE_T1_BASE_DIR:-}" ]; then
  $PY -u "$R/Track1/U-net/VariantDecisionRepro/final_reuse_draw.py" \
      --t1-cfg "$RPDE_T1_BASE_DIR/cfg_teacher.yaml" --t2-cfg "$TEACHER_CFG"
fi
if ! skip "$TCKPT"; then
  ( cd "$T1" && $PY -u train_v4.py --config "$TEACHER_CFG" --gpu $GPU 2>&1 | tail -3 )
fi
CKPT=$(cfg_path "$SHIP_CFG" checkpoint)
SECOND_CKPT=$(cfg_path "$SECOND_CFG" checkpoint)
if [ "${RPDE_FINAL_MODE:-0}" = "1" ]; then
  echo "  [final mode] stage 1 ship teachers / ship student / second member skipped (no consumer in the Final card)"
else
if [ "$GEN2" = "1" ]; then
  # GEN2 ship teachers WSW + MIMOA + MIMOWSW (each distilled from RLB128 above); the students below distil from all three
  for TC in "$WSW_SHIP_CFG" "$MIMOA_SHIP_CFG" "$MIMOWSW_SHIP_CFG"; do
    TCK=$(cfg_path "$TC" checkpoint)
    if ! skip "$TCK"; then ( cd "$T1" && $PY -u train_v4.py --config "$TC" --gpu $GPU 2>&1 | tail -3 ); fi
    [ -e "$TCK" ] || { echo "FAIL: $TCK was not produced"; exit 1; }
  done
fi
if ! skip "$CKPT"; then
  ( cd "$T1" && $PY -u train_v4.py --config "$SHIP_CFG" --gpu $GPU 2>&1 | tail -3 )
fi
$PY - "$CKPT" <<'EOF'
import sys, torch
m = torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("meta", {})
print(f"  base={m.get('base')} arch={m.get('arch')} gstep={m.get('gstep')} "
      f"val_rel_l2={m.get('val_rel_l2')}")
assert int(m.get("base", 0)) == 128, "T2 ships the base-128 forecaster (CW128); got base=%r" % m.get("base")
EOF

# The blend's SECOND member (WSW selfwarp, ships as model2.pth).  Distils from the same teacher
# trained above, so no extra teacher stage is needed.  Without this the chain rebuilt a
# ONE-member bundle and could not reach the card's row.
if ! skip "$SECOND_CKPT"; then
  ( cd "$T1" && $PY -u train_v4.py --config "$SECOND_CFG" --gpu $GPU 2>&1 | tail -3 )
fi
fi   # end of the non-Final stage-1 members

# ------------------------------------------------------------------ 2. head
# T2 reuses T1's head unchanged -- it consumes prediction/input summary statistics, not weights, so
# it transfers with no Track-2-specific retraining. Built by Track1/U-net/reproduce_t1.sh stage 4
# with --gmin/--gmax matching this forecaster's base.
say "2/5  variance-deficit head (shared with Track 1, base-96 variant)"
# The head is fitted on the RESIDUALS of the forecaster it corrects, so its nested-OOF source
# must be the base-96 AoA folds -- not the base-64 arm the previous version named. These five
# exist only to give the head an honestly out-of-fold forecast for every real file; skipping
# them silently would fit the head on its own forecaster's training data, where the variance
# deficit is ~0 and it learns "output 1".
if [ "${RPDE_C_SHARED_OOF:-0}" = "1" ] && ! skip "$HEAD"; then
  # C (shared_oof/README.md): the T2 head is fitted on the blend of the T2 members' Re3 fold versions (the fleet the
  # T1 base chain of this train() already waited for); the RLB_kf / CW128_kf fold fleets are not trained.
  [ -e "$RPDE_C_READY" ] || { echo "FAIL: shared-OOF fleet not ready ($RPDE_C_READY)"; exit 1; }
  echo "  [C] T2 vgain on the shared fivefold OOF: arm=$RPDE_C_ARM arm2=$RPDE_C_ARM2"
  ( cd "$T1" && VGAIN_W_TKE=$HEAD_W_TKE $PY -u train_vgain.py --arm "$RPDE_C_ARM" --arm2 "$RPDE_C_ARM2" \
      --ckpt-dir "$RPDE_C_FLEET_DIR" --gpu $GPU --width $HEAD_W --gmin $HEAD_GMIN --gmax $HEAD_GMAX \
      --manifest "$RPDE_C_MANIFEST" --cache $OUT/vgain_oof_c.npz --ship "$HEAD" 2>&1 | tail -6 )
fi
if ! skip "$HEAD"; then
  # SDa is self-distilled, so each student fold needs the SAME fold's teacher first.
  TEACHER_ARM=$($PY -c "import yaml,os;print(os.path.basename(yaml.safe_load(open('$TEACHER_FOLD_CFG'))['checkpoint']).replace('.pth',''))")
  if [ "${RPDE_FINAL_MODE:-0}" = "1" ] && [ -n "${RPDE_T1_BASE_DIR:-}" ]; then   # B: T1 stage 2a trained this fleet
    $PY -u "$R/Track1/U-net/VariantDecisionRepro/final_reuse_draw.py" \
        --t1-cfg "$RPDE_T1_BASE_DIR/cfg_teacher_fold.yaml" --t2-cfg "$TEACHER_FOLD_CFG" --folds "$FOLDS"
  fi
  for F in $FOLDS; do
    if ! skip "$T1CKPT/${TEACHER_ARM}_f${F}.pth"; then
      ( cd "$T1" && $PY -u train_v4.py --config "$TEACHER_FOLD_CFG" --fold $F --gpu $GPU 2>&1 | tail -3 )
    fi
  done
  for F in $FOLDS; do
    if ! skip "$T1CKPT/${ARM}_f${F}.pth"; then
      ( cd "$T1" && $PY -u train_v4.py --config "$FOLD_CFG" --fold $F --gpu $GPU 2>&1 | tail -3 )
    fi
  done
  ( cd "$T1" && VGAIN_W_TKE=$HEAD_W_TKE $PY -u train_vgain.py --arm $ARM --ckpt-dir "$T1CKPT" --gpu $GPU \
      --width $HEAD_W --gmin $HEAD_GMIN --gmax $HEAD_GMAX --manifest "$MANIFEST" \
      --cache $OUT/vgain_oof_b96.npz --ship "$HEAD" 2>&1 | tail -6 )
fi

# ------------------------------------------------------------------ 3. calibration
# t2_v6_prep.py takes no arguments: it streams the forecaster over the local benchmark and writes
# calib_v7.npz with the band, the TKE channel weights (ke_w = std_t**2, u:v = 36.8:1) and the
# norm_mean/norm_std the head needs to see PHYSICAL fields from inside a normalised submission.
say "3/5  calibration (band + ke_w + norm_mean/norm_std)"
# BOOTCALIB (2026-09-12).  TWO facts, both checked rather than assumed:
#   (a) `calib_v7.npz` is consumed by NOTHING downstream.  grep for calib_v7 in this file returns only
#       the OPEN GAP comment, the DESIGN RULE comment, this stage's own `say`, its `skip` test and the
#       key assertion below.  Stage 4 packages $SHIP_CALIB (stage 3c's SBPORT output) and stage 3b boots
#       from calib_v22_pc_m120.npz / the generated boot calib -- never from v7.
#   (b) its producer `t2_v6_prep.py` reads `calib_v4.npz`, an UNTRACKED learned blob with, per this
#       file's own header, "no reconstruction script".
# So under `set -euo pipefail` a clean clone died HERE, at a stage whose output nothing reads, because a
# dead-end input was missing.  The stage is kept (it is a cheap smoke test that the release statistics
# still produce the head's physical units) but it is now SKIPPED WITH A NOTICE when its base is absent,
# instead of taking the chain down.  Stage 3b's generated boot calib supplies the same three keys from
# the release, which is what actually matters.
if [ ! -e "$D/calib_v4.npz" ]; then
  echo "  [stage 3 SKIPPED] $D/calib_v4.npz is absent and has no reconstruction script.  Its only"
  echo "                    consumer is calib_v7.npz, which NO later stage reads, so nothing downstream"
  echo "                    is affected.  norm_mean/norm_std/ke_w are produced from the release by"
  echo "                    make_boot_calib.py in stage 3b instead."
elif ! skip "$D/calib_v7.npz"; then
  $PY t2_v6_prep.py 2>&1 | tail -8
fi
if [ -e "$D/calib_v7.npz" ]; then
$PY - "$D/calib_v7.npz" <<'EOF'
import sys, numpy as np
z = np.load(sys.argv[1])
need = {"norm_mean", "norm_std"}
missing = need - set(z.files)
assert not missing, f"calib is missing {missing}; the head cannot run without them and will be " \
                    f"silently switched off (submission_t2_v7 falls back to the frozen field)"
print(f"  keys OK: {sorted(z.files)}")
EOF
fi

# ------------------------------------------------------------------ 3b. the band (item M3 -- CLOSED)
# The T2 band chain HAS producers since 2026-08-12 (commit 5333c94): collect residuals through the
# PACKAGED frozen pipeline, then refit all 17 calib keys to the consumer's own conventions.
# Validated: non-inferiority +0.308 sps vs the shipped calib (bar -0.30), trio bit-identical,
# determinism +-0.002 across a different subsample RNG, and the reproduced calib is the LIVE #1 card
# (t2_v29_m3band, online +0.107 sps). The bootstrap bundle below IS the v27 configuration.
M3_CALIB=${M3_CALIB:-$D/calib_m3_repro.npz}
if [ "${RPDE_FINAL_MODE:-0}" = "1" ]; then
  say "3b/5  [final mode] M3 band skipped: the Final card's calib is calib_sbport (3d/3e); 3e builds calib_boot itself"
else
say "3b/5  M3 band: residual collection + fit (producers of the live card's calib)"
M3_CALIB=${M3_CALIB:-$D/calib_m3_repro.npz}
M3_CACHE=${M3_CACHE:-$D/m3_cache}
if ! skip "$M3_CALIB"; then
  # BOOTCALIB (2026-09-12, docs/decision/RECIPE.md open item 2).  calib_v22_pc_m120.npz is
  # an UNTRACKED learned blob present only on two servers, and it descends from calib_v4.npz, which this
  # file's own OPEN GAP note says "has no reconstruction script" -- so a clean clone, and the Decision
  # Phase environment that accepts training code only, could not run this stage at all.
  # Only THREE of its keys can reach a prediction: norm_mean / norm_std / ke_w, the physical units the
  # variance head consumes (ke_w[c] = std_t[c]**2 because scoring.kinetic_energy is computed on physical
  # fields).  All three are statistics of the release, and make_boot_calib.py rebuilds them from the
  # kit's TRACKED example_data/mean_std_real.pt BIT-IDENTICALLY (max|diff| 0.000e+00 on all three).
  # An A/B check (2026-09-12) proved the substitution inert rather than assuming it: two bundles identical
  # except --calib, driven through this stage's own collector over 378 steps, give seven residual arrays
  # that are bit-identical, worst max|diff| 0.0.
  # The historical blob is still PREFERRED when present, so a server run is byte-for-byte what it always
  # was; the generated file is the clean-clone path.  Set BOOT_CALIB= to choose explicitly.
  BOOT_CALIB=${BOOT_CALIB:-$D/calib_v22_pc_m120.npz}
  if [ ! -e "$BOOT_CALIB" ]; then
    BOOT_CALIB=$OUT/calib_boot.npz
    echo "  [bootcalib] $D/calib_v22_pc_m120.npz is absent -- generating a boot calib from the release"
    $PY -u "$D/make_boot_calib.py" --out "$BOOT_CALIB" 2>&1 | tail -4
  else
    echo "  [bootcalib] using $BOOT_CALIB (override with BOOT_CALIB=)"
  fi
  [ -e "$BOOT_CALIB" ] || { echo "MISSING $BOOT_CALIB and make_boot_calib.py did not emit it"; exit 1; }
  $PY -u build_t2_bundle.py --gpu $GPU --checkpoint "$CKPT" --head "$HEAD" \
      --template "$TEMPLATE" --ttt-lr 0.0 --ttt-steps 1 --halfwidth $HALFWIDTH \
      --calib "$BOOT_CALIB" --online '{}' --name t2_boot 2>&1 | tail -3
  $PY -u t2_collect_residuals.py --bundle "$D/submissions/t2_boot" --out "$M3_CACHE" 2>&1 | tail -2
  # --pool (TPa) and --pool-causal (TCPa) are BOTH adopted and BOTH present in every shipped zip
  # (pe_pooled reads True).  Omitting them here made the chain regenerate a calib the runtime
  # does not index -- the chain's own calib_m3_repro.npz has pe_pooled ABSENT.  Found 2026-08-16.
  M3_CACHE="$M3_CACHE" $PY -u t2_fit_band.py --pool --pool-causal --out "$M3_CALIB" 2>&1 | tail -4
  $PY -c "import numpy as np,sys; d=np.load(sys.argv[1],allow_pickle=True); assert 'pe_pooled' in d.files, 'TPa/TCPa pooling did not land in the refit calib'; print('M3_POOLED_OK')" "$M3_CALIB"
fi
[ -e "$M3_CALIB" ] || { echo "MISSING $M3_CALIB -- the M3 producers did not emit it"; exit 1; }
fi

# ------------------------------------------------------------------ 3c. SBPORT calib (2026-08-23)
say "3c/5  SBPORT calib: T2 prefit band + Track-1 learned sband table (producer of the live calib)"
# The live card (t2_v49_sbport, 2026-08-23, +0.1008 online over v46) ships Track 1's learned sband head
# (sband.pth = $LIVE_SBAND, produced by the T1 chain's t1_sband_{cache,fit,pack}.py) and a calib.npz
# that is the T2 prefit calib PLUS the five sb_* keys of the T1 sband calib.  That merge used to be a
# hand step; make_calib_sbport.py is its producer and must reproduce the live calib's DATA
# BIT-FOR-BIT, which is asserted below by comparing every array (keys/shapes/dtypes/raw bytes).
# NOT by an md5 of the .npz file: measured 2026-08-24, all 28 arrays reproduced exactly while the
# file md5 differed, because the .npz ZIP container is zip64 or plain depending on the numpy
# version.  A container md5 here is a guard on the environment, and would fail for the organizers.  The two inputs remain DECLARED inputs
# (their own build order -- BCD -> prefit on T2, sband fit on T1 -- is printed in stage 4, not hidden).
SHIP_SBAND=${SHIP_SBAND:-$LIVE_SBAND}
T2_PREFIT_CALIB=${T2_PREFIT_CALIB:-$D/calib_xcbpre_prefit.npz}
# 3c default (2026-09-10): the Q64 table behind calib_sbport_q64, the band the live card ships.  The old Q32 xcb
# table fails the stage-3c guard (sb_q 32 vs 64) against that calib; found on a clean-clone run.
T1_SBAND_CALIB=${T1_SBAND_CALIB:-$T1/sband/calib_sband_sbq64_m0100.npz}
SBPORT_CALIB=$OUT/calib_sbport_repro.npz
# PREFITBLEND (2026-09-13, docs/decision/RECIPE.md open item 2 (a)).  Set to 1 when this
# stage performed the merge, 0 when it deferred to stage 3e because the historical prefit blob is absent.
SBPORT_DONE=0
if [ -n "$SHIP_SBAND" ]; then
  if [ "${RPDE_WAIT_T1_BAND:-0}" = "1" ]; then
    _wait_start=$SECONDS
    until [ -e "$SHIP_SBAND" ] && [ -e "$T1_SBAND_CALIB" ]; do
      [ $((SECONDS-_wait_start)) -lt 86400 ] || { echo "T1 BAND WAIT TIMEOUT"; exit 1; }
      echo "  waiting for this run's T1 band: $SHIP_SBAND / $T1_SBAND_CALIB"
      sleep 60
    done
  fi
  [ -e "$SHIP_SBAND" ] || { echo "MISSING $SHIP_SBAND (the live sband head)"; exit 1; }
  [ -e "$T1_SBAND_CALIB" ] || { echo "MISSING $T1_SBAND_CALIB"; exit 1; }
  if [ -e "$T2_PREFIT_CALIB" ]; then
    echo "  !! DECLARED INPUTS of the SBPORT merge (producers: T1 chain sband stage / T2 BCD+prefit order):"
    echo "     sband head   $SHIP_SBAND"
    echo "     T2 prefit    $T2_PREFIT_CALIB"
    echo "     T1 sb table  $T1_SBAND_CALIB"
    $PY -u "$D/make_calib_sbport.py" --t2 "$T2_PREFIT_CALIB" --t1 "$T1_SBAND_CALIB" --out "$SBPORT_CALIB" \
        --expect-content "$LIVE_CALIB" \
      || { echo "REPRO_GUARD_FAIL -- make_calib_sbport.py did not reproduce the live calib $LIVE_CALIB bit-for-bit"; exit 1; }
    SBPORT_DONE=1
  else
    # Not an error any more.  The prefit is the output of THIS chain's own collector+fitter run on the
    # XCB BLEND, and the blend that ships is built by stage 3d -- which has not run yet here.  So the
    # merge is DEFERRED to stage 3e, which fits the prefit on the all81 members and then performs it.
    # That is the reconstruction ORDER RECIPE item 2 asks for; the historical blob
    # (calib_xcbpre_prefit.npz) was fitted on the 72-file XCB blend and is untracked besides.
    echo "  [prefit] DEFERRED -- $T2_PREFIT_CALIB is absent and no producer runs before stage 3d."
    echo "           Stage 3e will fit it on the ALL81 BLEND (the members the card actually ships) and"
    echo "           perform this merge there.  The bit-for-bit --expect-content guard cannot apply on"
    echo "           that path: with no live calib present there is nothing to compare against, so 3e"
    echo "           gates on the fitter's own pe_pooled assertion instead."
  fi
else
  echo "  (live card ships no sband head -- SBPORT stage skipped)"
fi

# ------------------------------------------------------------------ 3d. ALL81: the 81-file members (v58)
if [ "$ALL81" = "1" ]; then
say "3d/5  ALL81: ship teachers WSW/MIMOA/MIMOWSW + students EDISTMSH/EDISTMSSW on the 81-file pool (packaging-only swap)"
A81=$OUT/all81
mkdir -p "$A81"
# (a) The pool is a SYMLINK FARM over the released files.  Under REPRO_SCOPED=1 it lives under $OUT like every other
# output of the clean room: a clone's data/RealPDE_competition_dataset/gdrive_raw may itself be a symlink into another tree (on one host's scratch
# clone it points at the production checkout), and materializing through it would write outside the clone.  The path
# keeps the /data/RealPDE_competition_dataset/gdrive_raw/ component train_v4.py's provenance gate requires.  The generator refuses if the first
# EXISTING ancestor of the pool resolves outside the containment root -- checked before mkdir.
if [ "${REPRO_SCOPED:-0}" = "1" ]; then
  A81_POOL=${A81_POOL:-$OUT/data/RealPDE_competition_dataset/gdrive_raw/local_bench/all81/test_real}; A81_CONTAIN=$OUT
else
  A81_POOL=${A81_POOL:-$R/data/RealPDE_competition_dataset/gdrive_raw/local_bench/all81/test_real}; A81_CONTAIN=$R
fi
# (b) Configs, from the chain's OWN ship configs (the scoped ones under REPRO_SCOPED=1): YAML readback, diff-set
# assertion, checkpoint-collision assertion, stale-recipe refusal; under REPRO_SCOPED=1 also every absolute path inside
# this checkout (the PROD-string leak check cannot see a production tree that lives somewhere else, as on one development host).
$PY - "$R/data/RealPDE_competition_dataset/gdrive_raw/local_bench" "$A81_POOL" "$A81_CONTAIN" "$A81" "${REPRO_SCOPED:-0}" "$TCKPT" "$R" \
    "$WSW_SHIP_CFG" "$MIMOA_SHIP_CFG" "$MIMOWSW_SHIP_CFG" "$SHIP_CFG" "$SECOND_CFG" \
    "$T1CKPT" "$R/checkpoints/t1_u-net" <<'A81GEN'
import copy, os, pathlib, sys, yaml
lb, pool, contain, a81 = (pathlib.Path(x) for x in sys.argv[1:5])
scoped, t_ckpt, root = sys.argv[5] == "1", sys.argv[6], sys.argv[7].rstrip("/")
TEACH, STUD = ["WSW", "MIMOA", "MIMOWSW"], ["EDISTMSH", "EDISTMSSW"]
src = dict(zip(TEACH + STUD, sys.argv[8:13]))
ckdirs = [pathlib.Path(x) for x in sys.argv[13:15]]
sys.path.insert(0, root + "/common")
from repo_paths import resolve_paths          # tracked (unscoped) ship configs carry repo-relative paths
SENT = object()

# ---- (a) the 81-file pool = local_bench/train/test_real (72) U local_bench/val/test_real (9)
tr = {p.name: p for p in sorted((lb / "train/test_real").glob("*.h5"))}
va = {p.name: p for p in sorted((lb / "val/test_real").glob("*.h5"))}
assert len(tr) == 72 and len(va) == 9, f"local_bench sizes {len(tr)}/{len(va)}, expected 72/9"
assert not set(tr) & set(va), f"train/val overlap {sorted(set(tr) & set(va))}"
union = {**tr, **va}
assert len(union) == 81 and "7575_0.h5" not in union, (len(union), "7575_0.h5" in union)
croot = os.path.realpath(contain)
anc = pool
while not os.path.lexists(anc):
    anc = anc.parent
ra = os.path.realpath(anc)
assert ra == croot or ra.startswith(croot + "/"), \
    f"ALL81 pool {pool} would be created through {anc} -> {ra}, OUTSIDE the containment root {croot} -- refusing"
pool.mkdir(parents=True, exist_ok=True)
assert os.path.realpath(pool).startswith(croot + "/"), os.path.realpath(pool)
for nm, f in sorted(union.items()):
    tgt = os.path.realpath(f)
    assert os.path.isfile(tgt), f"{f} -> {tgt} is not a file"
    ln = pool / nm
    if os.path.lexists(ln):
        assert ln.is_symlink() and os.readlink(ln) == tgt, f"stale pool entry {ln}"
    else:
        os.symlink(tgt, ln)
entries = {p.name for p in pool.iterdir()}
assert entries == set(union), f"pool entries != train U val: {sorted(entries ^ set(union))}"
h5 = sorted(pool.glob("*.h5"))
assert len(h5) == 81 and not os.path.lexists(pool / "7575_0.h5")
assert all(os.path.realpath(pool / n) == os.path.realpath(f) for n, f in union.items())
print(f"  pool {pool}: {len(h5)} .h5 = train {len(tr)} + val {len(va)} (disjoint); 7575_0 absent; each entry == its local_bench file")
print(f"    added (val) files: {' '.join(sorted(va))}")

# ---- (b) the five configs
old_glob, new_glob = str(lb / "train/test_real/*.h5"), str(pool / "*.h5")
sfx = f"_t2{a81.parent.name}" if scoped else ""        # the scoping block's `stem + "_t2" + out.name`
plan = {}
for mem in TEACH + STUD:
    pd = resolve_paths(yaml.safe_load(open(src[mem])), root)
    assert pd["variant"] == f"UNet_v4_ship_{mem}{sfx}", (mem, pd["variant"])
    assert pd["desc"] == f"v4_ship_{mem}", (mem, pd["desc"])
    ck0 = pathlib.Path(pd["checkpoint"])
    assert ck0.name == f"unet_v4_ship_{mem}{sfx}.pth", (mem, ck0)
    plan[mem] = (pd, ck0.parent / f"unet_v4_ship_AD_{mem}{sfx}.pth")
mem_of = {}
for m, (pd, ck) in plan.items():
    mem_of[ck.name] = m
    mem_of[ck.stem + "_last.pth"] = m
assert len(mem_of) == 10, "all81 checkpoint names are not distinct"
for d in sorted({ck.parent for _, ck in plan.values()} | set(ckdirs)):
    if not d.is_dir():
        continue
    for p in sorted(d.glob("*.pth")):
        if p.name in mem_of:
            m = mem_of[p.name]
            assert (a81 / f"train_{m}.out").exists(), \
                f"checkpoint collision: {p} exists but stage 3d never ran {m} (no {a81}/train_{m}.out) -- refusing to overwrite it"
            print(f"  (resume) {p.name} exists and was produced by this stage ({m}'s run log exists)")
def abs_paths(x):
    # every absolute-path fragment a config names (comma-joined multi-teacher strings split)
    if isinstance(x, dict):
        for v in x.values():
            yield from abs_paths(v)
    elif isinstance(x, list):
        for v in x:
            yield from abs_paths(v)
    elif isinstance(x, str):
        for part in x.split(","):
            if part.startswith("/"):
                yield part
for mem in TEACH + STUD:
    pd, ck = plan[mem]
    b = copy.deepcopy(pd)
    assert [s["name"] for s in b["stages"]] == ["sim-pretrain", "real-finetune"], (mem, [s["name"] for s in b["stages"]])
    srcs = b["stages"][1]["sources"]
    assert len(srcs) == 1 and srcs[0]["tag"] == "comp_real" and srcs[0]["glob"] == old_glob, (mem, srcs, old_glob)
    srcs[0]["glob"] = new_glob
    assert sorted(int(x) for x in (pd.get("exclude_re") or [])) == [11400, 17775], (mem, pd.get("exclude_re"))
    b["exclude_re"] = []
    b["variant"], b["desc"] = f"UNet_v4_ship_AD_{mem}{sfx}", f"v4_ship_AD_{mem}"
    b["checkpoint"], b["logger_root"] = str(ck), str(a81)
    if mem in TEACH:
        # a teacher distils from the chain's OWN 72-file RLB128 (stage 1) -- the v58 lineage, never a production file
        assert str(pd["distill_ckpt"]) == t_ckpt and os.path.isfile(t_ckpt), (mem, pd["distill_ckpt"], t_ckpt)
    else:
        old = [x for x in str(pd["distill_ckpt"]).split(",") if x]
        assert old == [str(plan[t][0]["checkpoint"]) for t in TEACH], (mem, old)
        b["distill_ckpt"] = ",".join(str(plan[t][1]) for t in TEACH)
    dst, log = a81 / f"cfg_a81_{mem}.yaml", a81 / f"train_{mem}.out"
    if dst.exists() and yaml.safe_load(open(dst)) != b and log.exists() and \
            any(l.startswith("TRAIN_V4_OK ") for l in open(log, errors="replace")):
        raise SystemExit(f"ALL81: {dst} would CHANGE after {mem} finished training ({log}) -- its checkpoint "
                         f"belongs to the old recipe; move the log and checkpoint aside to retrain")
    txt = yaml.safe_dump(b, sort_keys=False)
    if scoped:
        outside = sorted({p for p in abs_paths(b) if not p.startswith(root + "/")})
        assert not outside, f"{dst}: REPRO_SCOPED=1 but the config names path(s) outside {root}: {outside}"
    dst.write_text(txt)
    rb = yaml.safe_load(open(dst))                                  # YAML READBACK
    assert rb == b, f"{dst}: readback differs from the intended dict"
    diff = {k for k in set(pd) | set(rb) if pd.get(k, SENT) != rb.get(k, SENT)}
    want = {"variant", "desc", "checkpoint", "logger_root", "stages", "exclude_re"} | ({"distill_ckpt"} if mem in STUD else set())
    assert diff == want, (mem, "diff set", sorted(diff), "expected", sorted(want))
    assert rb["stages"][0] == pd["stages"][0], (mem, "sim-pretrain stage changed")
    s0, s1 = pd["stages"][1], rb["stages"][1]
    assert {k: v for k, v in s0.items() if k != "sources"} == {k: v for k, v in s1.items() if k != "sources"}, (mem, "real-finetune knobs changed")
    assert {k: v for k, v in s0["sources"][0].items() if k != "glob"} == {k: v for k, v in s1["sources"][0].items() if k != "glob"}, (mem, "source knobs changed")
    teachers = [pathlib.Path(x).name for x in str(rb.get("distill_ckpt", "")).split(",") if x]
    print(f"  {dst.name}: diff {sorted(diff)} | ckpt {ck.name} | steps {[s['steps'] for s in rb['stages']]} "
          f"| seed {rb.get('seed')} fuse {rb.get('fuse')} block {rb['block']} dw {rb.get('distill_weight')}")
    print(f"    teachers ({len(teachers)}): {teachers}")
print("ALL81_GEN_OK")
A81GEN
# (c) Train.  Completion is keyed on the run log's TRAIN_V4_OK line naming THIS variant's run dir -- never on the
# checkpoint, which train_v4.py writes incrementally; a resume skips a member only on that line.  T2 differences from
# the T1 stage: training runs from $T1 (this chain sits in Track2/U-net); a train_v4 that died by a SIGNAL propagates
# its rc (>= 128) so a signal-only supervisor can retry; a member whose config is still being trained by a live
# train_v4 process (an orphan of a killed chain) is refused instead of started twice onto the same checkpoint.
a81_train () {
  local mem=$1 cfg log var ck rc t
  cfg=$A81/cfg_a81_$1.yaml; log=$A81/train_$1.out
  var=$($PY -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['variant'])" "$cfg")
  ck=$(cfg_path "$cfg" checkpoint)
  if [ -e "$log" ] && grep -q "^TRAIN_V4_OK $A81/Logger/$var/" "$log"; then
    echo "  [have] $mem -- TRAIN_V4_OK for $var in $log -- skipping"
  else
    if pgrep -f -- "train_v4[.]py --config $cfg" > /dev/null; then
      echo "REFUSING: a train_v4 process is already training $cfg -- not starting $mem twice"; exit 1
    fi
    if [ -e "$log" ]; then
      mv "$log" "$log.incomplete.$(date -u +%Y%m%dT%H%M%SZ)"; echo "  (an incomplete run log of $mem was moved aside)"
    fi
    echo "  training $mem ($var) on GPU $GPU -> $log   [$(date -u +%FT%TZ)]"
    rc=0; ( cd "$T1" && exec $PY -u train_v4.py --config "$cfg" --gpu $GPU ) > "$log" 2>&1 || rc=$?
    tail -3 "$log"
    if ! grep -q "^TRAIN_V4_OK $A81/Logger/$var/" "$log"; then
      echo "FAIL: $mem (rc=$rc) -- no TRAIN_V4_OK for $var in $log"
      if [ "$rc" -ge 128 ]; then exit "$rc"; fi
      exit 1
    fi
  fi
  grep -q "real-finetune:comp_real: 81 files " "$log" || { echo "FAIL: $log does not show the 81-file real-finetune pool"; exit 1; }
  [ -s "$ck" ] || { echo "FAIL: $mem logged TRAIN_V4_OK but $ck is missing"; exit 1; }
  if [ "$mem" = "EDISTMSH" ] || [ "$mem" = "EDISTMSSW" ]; then
    for t in $A81_TEACHER_CKPTS; do
      [ "$log" -nt "$t" ] || { echo "FAIL: $mem's run log is older than teacher $t -- the student predates its teacher; retrain it"; exit 1; }
    done
  fi
}
A81_TEACHER_CKPTS=""
for M in WSW MIMOA MIMOWSW; do
  a81_train $M
  A81_TEACHER_CKPTS="$A81_TEACHER_CKPTS $(cfg_path "$A81/cfg_a81_$M.yaml" checkpoint)"
done
echo "  ALL81 teachers OK:$A81_TEACHER_CKPTS"
for M in EDISTMSH EDISTMSSW; do a81_train $M; done
A81_SHIP_CKPT=$(cfg_path "$A81/cfg_a81_EDISTMSH.yaml" checkpoint)
A81_SECOND_CKPT=$(cfg_path "$A81/cfg_a81_EDISTMSSW.yaml" checkpoint)
$PY - "$A81_SHIP_CKPT" <<'EOF'
import sys, torch
m = torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("meta", {})
print(f"  ALL81 model.pth: base={m.get('base')} arch={m.get('arch')} gstep={m.get('gstep')} "
      f"val_rel_l2={m.get('val_rel_l2')}  (val = the 9 interp files, IN-SAMPLE for this member)")
assert int(m.get("base", 0)) == 128, "T2 ships the base-128 forecaster; got base=%r" % m.get("base")
EOF
echo "  ALL81 students OK: model.pth <- $A81_SHIP_CKPT ; model2.pth <- $A81_SECOND_CKPT"
fi

# ------------------------------------------------------------------ 3e. blend prefit + deferred SBPORT merge
# PREFITBLEND (2026-09-13, docs/decision/RECIPE.md open item 2 (a)).  Runs only when 3c
# deferred.  Produces the prefit with THIS CHAIN'S OWN producers -- the same collector and the same
# fitter stage 3b uses, with the same --pool/--pool-causal flags every shipped zip carries -- but driven
# through a boot bundle built from the two ALL81 members, i.e. the blend the card actually ships, instead
# of the single-member $CKPT.  Then performs the merge 3c deferred.  No new dependency and no new code
# path: build_t2_bundle.py, t2_collect_residuals.py, t2_fit_band.py and make_calib_sbport.py are all
# already in the chain; what changes is the ORDER and WHICH FORECASTER the residuals are collected through.
PREFIT_PRODUCED=0
if [ "${SBPORT_DONE:-0}" = "0" ] && [ -n "$SHIP_SBAND" ]; then
  say "3e/5  blend prefit on the ALL81 members, then the SBPORT merge deferred from 3c"
  { [ -n "${A81_SHIP_CKPT:-}" ] && [ -n "${A81_SECOND_CKPT:-}" ]; } || {
    echo "REPRO_GUARD_FAIL -- stage 3e needs stage 3d's ALL81 members (A81_SHIP_CKPT / A81_SECOND_CKPT).";
    echo "                    Run with ALL81=1, or supply T2_PREFIT_CALIB= so stage 3c does not defer."; exit 1; }
  [ -e "$T1_SBAND_CALIB" ] || { echo "MISSING $T1_SBAND_CALIB (the T1 sband table; the T1 chain's stage 8b produces it under BAND_PIPELINE=1)"; exit 1; }
  PREFIT_BLEND=$OUT/calib_prefit_blend.npz
  PREFIT_CACHE=$OUT/prefit_blend_cache
  PREFIT_BOOT=$OUT/calib_boot.npz
  if ! skip "$PREFIT_BLEND"; then
    [ -e "$PREFIT_BOOT" ] || $PY -u "$D/make_boot_calib.py" --out "$PREFIT_BOOT" 2>&1 | tail -3
    [ -e "$PREFIT_BOOT" ] || { echo "MISSING $PREFIT_BOOT and make_boot_calib.py did not emit it"; exit 1; }
    $PY -u build_t2_bundle.py --gpu $GPU --checkpoint "$A81_SHIP_CKPT" --checkpoint2 "$A81_SECOND_CKPT" \
        --head "$HEAD" --template "$TEMPLATE" --ttt-lr 0.0 --ttt-steps 1 --halfwidth $HALFWIDTH \
        --calib "$PREFIT_BOOT" --online '{}' --name t2_prefit_boot 2>&1 | tail -3
    $PY -u t2_collect_residuals.py --bundle "$D/submissions/t2_prefit_boot" --out "$PREFIT_CACHE" \
        --data-dir "$R/data/RealPDE_competition_dataset/gdrive_raw/local_bench/train" 2>&1 | tail -2
    M3_CACHE="$PREFIT_CACHE" $PY -u t2_fit_band.py --pool --pool-causal --out "$PREFIT_BLEND" 2>&1 | tail -4
    $PY -c "import numpy as np,sys; d=np.load(sys.argv[1],allow_pickle=True); assert 'pe_pooled' in d.files, 'TPa/TCPa pooling did not land in the blend prefit'; print('PREFIT_POOLED_OK', len(d.files), 'keys')" "$PREFIT_BLEND"
  fi
  [ -e "$PREFIT_BLEND" ] || { echo "MISSING $PREFIT_BLEND -- the 3e producers did not emit it"; exit 1; }
  T2_PREFIT_CALIB=$PREFIT_BLEND
  $PY -u "$D/make_calib_sbport.py" --t2 "$T2_PREFIT_CALIB" --t1 "$T1_SBAND_CALIB" --out "$SBPORT_CALIB" \
    || { echo "REPRO_GUARD_FAIL -- make_calib_sbport.py failed on the produced blend prefit"; exit 1; }
  [ -e "$SBPORT_CALIB" ] || { echo "MISSING $SBPORT_CALIB"; exit 1; }
  SBPORT_DONE=1; PREFIT_PRODUCED=1
  echo "  [prefit] SBPORT merge done from the PRODUCED blend prefit"
  echo "     prefit  $T2_PREFIT_CALIB  (fitted on the all81 blend, this chain's own collector+fitter)"
  echo "     sbport  $SBPORT_CALIB"
fi

if [ "${RPDE_FINAL_MODE:-0}" = "1" ]; then
  echo "  [final mode] stages 4/5 skipped: the v58-generation packaging bundle and its in-sample anchors have no consumer in the Final card"
  echo "REPRODUCE_T2_OK"
  exit 0
fi
# ------------------------------------------------------------------ 4. bundle
say "4/5  shipping bundle"
# DERIVED.  The M3 stage above is still the T2 band PRODUCER and its output is a validated
# non-inferior alternative (+0.308 sps vs the calib it replaced, 2026-08-11), but the LIVE card
# ships the BCD-recentred, pre-head-fitted band (2026-08-18, calib_xcbpre_prefit.npz) -- whose
# build ORDER is not yet a script -- merged with the T1 sband table (stage 3c, a real producer).
# So the prefit band is a declared input, exactly as T1's band is, and the gap is printed rather
# than hidden.  Set SHIP_CALIB=$M3_CALIB to rebuild the M3 band instead.
if [ -n "$SHIP_SBAND" ]; then SHIP_CALIB=${SHIP_CALIB:-$SBPORT_CALIB}; else SHIP_CALIB=${SHIP_CALIB:-$LIVE_CALIB}; fi
if [ "$SHIP_CALIB" != "$M3_CALIB" ]; then
  if [ "${PREFIT_PRODUCED:-0}" = "1" ]; then
    echo "  [prefit] the band inside $SHIP_CALIB was PRODUCED by stage 3e on the all81 blend -- no"
    echo "           declared input.  ($M3_CALIB is the single-member M3 band, a validated non-inferior"
    echo "           alternative; the card ships the blend prefit + sband table, as the live card does.)"
  else
    echo "  !! NOT-YET-REPRODUCIBLE INPUT: the BCD+prefit band inside $SHIP_CALIB"
    echo "     (the M3 producers DID run above and wrote $M3_CALIB, which is a validated"
    echo "      non-inferior alternative; the live card carries the BCD+prefit band + sband table instead)"
  fi
fi
[ -e "$SHIP_CALIB" ] || { echo "MISSING $SHIP_CALIB"; exit 1; }

# ALL81 (2026-09-10): what gets PACKAGED.  ALL81=1 swaps in stage 3d's 81-file students here and only here; the head,
# the SBPORT calib and the sband stay what the 72-file generation / the declared inputs produced (design rule), so the
# build overrides the calib provenance gate with --allow-calib-mismatch exactly as t2_v58_alldata was built,
# and a distinct bundle name + build log keep the 72-file stage-4 output
# (t2_repro) from being reused or re-parsed by stage 5.
PKG_CKPT=$CKPT; PKG_CKPT2=$SECOND_CKPT; PKG_NAME=t2_repro; PKG_LOG=$OUT/build.log
if [ "$ALL81" = "1" ]; then
  PKG_CKPT=$A81_SHIP_CKPT; PKG_CKPT2=$A81_SECOND_CKPT
  PKG_NAME=${BUNDLE_ALL81:-t2_repro_all81}; PKG_LOG=$OUT/build_all81.log
  case "$PKG_NAME" in t2_repro|t2_boot|t2_prefit_boot) echo "REFUSING: BUNDLE_ALL81=$PKG_NAME would overwrite a stage output"; exit 1;; esac
  echo "  ALL81=1 -- packaging the 81-file students with the head/calib/sband of the 72-file generation:"
  echo "     model.pth   $PKG_CKPT"
  echo "     model2.pth  $PKG_CKPT2"
  echo "     vgain.pth   $HEAD"
  echo "     calib.npz   $SHIP_CALIB"
  echo "     bundle $PKG_NAME   build log $PKG_LOG"
fi

# GUARD -- T2 had NONE.  All packaged artifacts (five since SBPORT) are compared against the members
# inside the live card's own zip, which cannot drift the way a hand-copied snapshot does.
_CHECK="${SHIP_SBAND:+sband.pth=$SHIP_SBAND}"
if [ "${REPRO_SCOPED:-0}" = "1" ]; then
  echo "  [guard] REPRO_SCOPED=1 -- model.pth / model2.pth / vgain.pth are OUTPUTS of this run"
  echo "          (retrained from scratch); they are gated by the accuracy anchors in stage 5, not"
  echo "          by md5.  calib.npz is an OUTPUT of stage 3c, which already proved it array-by-array"
  echo "          against the live calib (keys/shapes/dtypes/raw bytes) -- a strictly stronger check"
  echo "          than a container md5, which on an .npz tracks the numpy version, not the data."
  echo "          sband.pth is a declared INPUT copied from the T1 chain and IS md5-checked."
else
  _CHECK="model.pth=$PKG_CKPT,model2.pth=$PKG_CKPT2,vgain.pth=$HEAD,calib.npz=$SHIP_CALIB${SHIP_SBAND:+,sband.pth=$SHIP_SBAND}"
fi
if [ -n "$_CHECK" ]; then
  $PY "$R/common/live_card.py" --track 2 --check "$_CHECK" $LC_BOOT \
    || { echo "REPRO_GUARD_FAIL -- see above; re-anchor LIVE_CARD or fix the override"; exit 1; }
fi
# The live band is a DECLARED INPUT (its build order is unscripted, stated above), so in scoped
# mode it is fitted on the LIVE forecaster's residuals while this run packages a RETRAINED one.
# build_t2_bundle.py's BCP gate is right to refuse that by default; here it is overridden on
# purpose, loudly.  Drop this override the moment the band's build order becomes a script.
_PROV=""
if [ "${REPRO_SCOPED:-0}" = "1" ] && [ "$SHIP_CALIB" != "$M3_CALIB" ]; then
  echo "  [prov] DELIBERATE OVERRIDE: $SHIP_CALIB is fitted on the LIVE forecaster's residuals and"
  echo "         this run packages a RETRAINED forecaster.  That is the declared-input gap above,"
  echo "         not a silent drift.  Provenance-clean alternative: SHIP_CALIB=$M3_CALIB"
  echo "         (the M3 producers ran in stage 3b and DID refit on this run's own model)."
  _PROV="--allow-calib-mismatch"
fi
if [ "$ALL81" = "1" ]; then
  echo "  [prov] ALL81=1 DELIBERATE OVERRIDE: $SHIP_CALIB was not fitted on the 81-file students (design rule);"
  echo "         t2_v58_alldata was built with the same --allow-calib-mismatch."
  _PROV="--allow-calib-mismatch"
fi
$PY -u build_t2_bundle.py --gpu $GPU --checkpoint "$PKG_CKPT" --checkpoint2 "$PKG_CKPT2" \
    --head "$HEAD" ${SHIP_SBAND:+--sband "$SHIP_SBAND"} $_PROV \
    --template "$TEMPLATE" --ttt-lr 0.0 --ttt-steps 1 --halfwidth $HALFWIDTH \
    --calib "$SHIP_CALIB" --online "$ONLINE" --name "$PKG_NAME" 2>&1 \
  | tee "$PKG_LOG" | grep -E '"(rel_l2|tke|mvpe|sps|time)_score"|FINAL_LOCAL=|coverage|\[prov\]'

# ------------------------------------------------------------------ 5. ACCEPTANCE TESTS
say "5/5  acceptance tests"
REL=$(grep -oE '"rel_l2_score": [0-9.]+' "$PKG_LOG" | tail -1 | grep -oE '[0-9.]+$')
TKE=$(grep -oE '"tke_score": [0-9.]+' "$PKG_LOG" | tail -1 | grep -oE '[0-9.]+$')
MVPE=$(grep -oE '"mvpe_score": [0-9.]+' "$PKG_LOG" | tail -1 | grep -oE '[0-9.]+$')
SPS=$(grep -oE '"sps_score": [0-9.]+' "$PKG_LOG" | tail -1 | grep -oE '[0-9.]+$')
GOT=$(grep -oE "FINAL_LOCAL=[0-9.]+" "$PKG_LOG" | tail -1 | cut -d= -f2)
for v in "$REL" "$TKE" "$MVPE" "$SPS"; do
  [ -n "$v" ] || { echo "FAIL: a subscore is missing from $PKG_LOG"; exit 1; }
done
if [ "$ALL81" = "1" ]; then
  echo "  anchors = $LIVE_NAME verify_report.json (interp378) -- IN-SAMPLE for 81-file members, on the card's side and on this rebuild's"
fi
# the acceptance load pins the chain's GPU (it used to land on cuda:0 whatever GPU was)
# RPDE_FINAL_MODE=1 (set only by the Final package's final_pipeline.run_t2_base, 2026-09-29): the organizer runs
# train() ONCE with no retry; the four anchors are in-sample values of a retrained (stochastic) chain -- the v12 smoke
# measured tke -0.313 against +-0.3.  In that mode anchor misses and a CUDA-graph fallback (speed only, other GPU
# stack) are recorded in $OUT/anchor_check.json and never abort; a head that does not load stays fatal.  Internal
# repro chains (no RPDE_FINAL_MODE) keep every gate.
CUDA_VISIBLE_DEVICES=$GPU RPDE_ANCHOR_JSON="$OUT/anchor_check.json" $PY - "$D/submissions/$PKG_NAME.zip" "$REL" "$TKE" "$MVPE" "$SPS" "${GOT:-nan}" \
      "$TARGET_REL" "$TOL_REL" "$TARGET_TKE" "$TOL_TKE" \
      "$TARGET_MVPE" "$TOL_MVPE" "$TARGET_SPS" "$TOL_SPS" <<'EOF'
import sys, os, zipfile, tempfile, importlib.util, torch, json
zp = sys.argv[1]
rel, tke, mvpe, sps, got = (float(x) for x in sys.argv[2:7])
t_rel, tol_rel, t_tke, tol_tke, t_mvpe, tol_mvpe, t_sps, tol_sps = (float(x) for x in sys.argv[7:15])

d = tempfile.mkdtemp(); zipfile.ZipFile(zp).extractall(d); sys.path.insert(0, d)
spec = importlib.util.spec_from_file_location("sub", os.path.join(d, "submission.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
mdl = m.get_ttt_model(d, torch.device("cuda"))

ok_head = getattr(mdl, "head", None) is not None
ok_graph = getattr(mdl, "_graph", None) is not None and getattr(mdl, "_graph_checked", False)
print(f"  head loaded        : {ok_head}")
print(f"  CUDA graph captured: {ok_graph}   (bit-exact vs eager, else discarded by the submission)")

bad = []
if not ok_head:  bad.append("head did not load -- the calib is missing norm_mean/norm_std")
if not ok_graph: bad.append("graph fell back to eager -- slower than what was measured")
for name, val, want, tol in (("rel_l2", rel, t_rel, tol_rel), ("tke", tke, t_tke, tol_tke),
                             ("mvpe", mvpe, t_mvpe, tol_mvpe), ("sps", sps, t_sps, tol_sps)):
    print(f"  {name:7s} {val:8.3f}   anchor {want:8.3f}   delta {val-want:+7.3f}   tol +-{tol}")
    if abs(val - want) > tol:
        bad.append(f"{name} outside +-{tol}")
print(f"  FINAL_LOCAL {got:.3f}  (INFORMATIONAL ONLY: its local `time` component carries contention"
      f" noise measured at 15x any usable tolerance)")
if os.environ.get("RPDE_FINAL_MODE") == "1":
    rows = {n: {"value": v, "anchor": w, "delta": v - w, "tol": t} for n, v, w, t in
            (("rel_l2", rel, t_rel, tol_rel), ("tke", tke, t_tke, tol_tke), ("mvpe", mvpe, t_mvpe, tol_mvpe),
             ("sps", sps, t_sps, tol_sps))}
    json.dump({"status": "RECORDED_NOT_GATED", "stage": "t2_base 5/5", "head_loaded": ok_head,
               "cuda_graph_captured": ok_graph, "misses": bad, "legs": rows, "final_local_informational": got},
              open(os.environ["RPDE_ANCHOR_JSON"], "w"), indent=1)
    if not ok_head:
        print("  FAIL: head did not load -- the calib is missing norm_mean/norm_std"); sys.exit(1)
    print("  RECORDED (Final mode, not gated): " + ("; ".join(bad) if bad else "all checks inside tolerance"))
    sys.exit(0)
if bad:
    print("  FAIL: " + "; ".join(bad)); sys.exit(1)
print("  ACCEPT: the chain reproduces the shipped ACCURACY+BAND result from a clean tree.")
EOF
if [ "$ALL81" = "1" ]; then echo "  ALL81=1: the accepted bundle is $PKG_NAME (the $LIVE_NAME generation)"; fi
echo "REPRODUCE_T2_OK"
