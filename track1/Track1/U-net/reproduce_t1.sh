#!/usr/bin/env bash
# =====================================================================================
#  Track 1: raw data -> shipping zip, from a CLEAN CHECKOUT. No historical artifact.
# =====================================================================================
#
# WHY THIS EXISTS.  The Rules page states that the development leaderboard only shortlists the top
# 10, and that final placings come from the Decision Phase, where "organizers re-train each
# shortlisted team's method from scratch on the competition's GPU cluster to ensure
# reproducibility, then evaluate on a private test set with unseen parameter regimes".
#
# v11 could not have survived that. Audited 2026-07-29, the calibration it ships was produced by a
# chain of DERIVED artifacts reaching back three bundle generations:
#
#     t1_sps_v2.py    --bundle     defaulted to submissions/t1_unet_v7.zip
#     t1_v8_verify.py --v7-calib   defaulted to submissions/t1_unet_v7/calib.npz
#     t1_v8_verify.py --checkpoint defaulted to unet_v3_rel_tke20_s1.pth   (a v3-era model)
#     t1_v9_prep.py                derived band_*_phys from the v2_calib bundle
#
# and t1_unet_v7.zip is itself built from t1_unet_v2_calib. On a fresh clone none of those exist, so
# the chain simply does not run. Worse, three of the scripts in it lived only in the server's /tmp
# until they were rescued into the repo on 2026-07-29 -- the lineage was one cleanup away from being
# unreproducible even here.
#
# THE FIX IS A BOOTSTRAP, NOT A DOCUMENT.  `t1_sps_v2.py --bundle` only reads a BASELINE BAND out of
# that zip; nothing requires the zip to be historical. So stage 4 fits a band from scratch with
# `t1_band_v5.py` (which needs no prior calibration at all), stage 5 packs it into a throwaway
# bundle, and stage 6 fits the activity scaling on top of THAT. The loop closes with no dependency
# on anything outside a clean checkout plus `data/`.
#
# ACCEPTANCE TEST.  Stage 9 rebuilds the bundle and gates on per-subscore ACCURACY anchors
# (rel/tke/mvpe/sps); FINAL_LOCAL is printed as informational only, because its local `time`
# component carries contention noise (F5, 2026-08-12).  Reproducibility that is not checked is a
# claim, not a property.
#
#   bash reproduce_t1.sh [GPU]        # ~2.5 h on one idle RTX 4090; stages are resumable
#
# NOT reproduced here: the daily Codabench upload, which is a human action.
# =====================================================================================
set -euo pipefail

# env-first, then positional: the fresh-clone launcher passed GPU=0 as env and was ignored
GPU="${GPU:-${1:-0}}"
# Derived from the script's own location: the absolute hardcode sent a CLONE's run back into
# the production tree.  A clone run must ALSO set REPRO_SCOPED=1 -- the tracked configs still
# carry absolute production checkpoint paths, which only the scoped mode rewrites.
R=${RPDE_ROOT:?set RPDE_ROOT to /workspace in the Final container}
D=$R/Track1/U-net
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
# CACHESTAMP (2026-08-27).  THE fold-checkpoint-stem derivation, in ONE place.  This expression
# was open-coded in SIX spots in THREE spellings -- REPRO_SCOPED's ARM, TEACHER_ARM (as
# os.path.basename(...).replace('.pth','')), BSTEM in stage 2c, and the four BP_STEM_* band fleet
# stems -- while ARM's own default was a HARDCODED STRING that no config produced.  That asymmetry
# is the defect: override FOLD_CFG alone and the band pipeline follows it while the vgain head
# does not.  A seventh copy would be a seventh place to drift, so every caller now routes here.
# The three expressions that return the FULL checkpoint PATH (TEACHER_CKPT / SHIP_CKPT /
# SECOND_CKPT) are a different quantity and deliberately do NOT use this.
cfg_stem () {   # cfg_stem <path/to/config.yaml> -> the `checkpoint:` key's basename, no extension
  local _s
  _s=$("$PY" -c "import yaml,pathlib,sys;print(pathlib.Path(yaml.safe_load(open(sys.argv[1]))['checkpoint']).stem)" "$1") \
    || { echo "cfg_stem: could not read a 'checkpoint:' key from $1" >&2; return 1; }
  [ -n "$_s" ] || { echo "cfg_stem: empty stem from $1" >&2; return 1; }
  printf '%s' "$_s"
}
# TAG keys every scratch path and bundle name, so a second arm cannot overwrite the first's
# head, band or calibration.  Unset -> `repro` -> every path below is the string it always was.
TAG=${TAG:-repro}
OUT=${OUT:-$D/$TAG}
# OUT/TAG COUPLING GUARD (2026-09-13).  The scoped configs are WRITTEN to $OUT (the
# scope() helper below is handed "$OUT") but every reader resolves them as $D/$TAG/cfg_*.yaml -- see
# SHIP_CFG/SECOND_CFG/FOLD_CFG/TEACHER_CFG/TEACHER_FOLD_CFG/BAND_CFG_* and their uses at the
# `$PY - "$D/$SHIP_CFG" ...` line and `cfg_stem "$D/$TEACHER_FOLD_CFG"`.  The two agree only when OUT is
# left at its default.  Setting OUT anywhere else therefore fails TWO STAGES LATER with
#   FileNotFoundError: .../Track1/U-net/<TAG>/cfg_teacher_fold.yaml
#   cfg_stem: could not read a 'checkpoint:' key from ...
# which names a file that was never meant to exist and points at the wrong cause.  Measured: a
# from-scratch Decision-Phase run launched with OUT=<fresh dir> TAG=dpboot wrote
# all fourteen cfg_*.yaml into $OUT, passed stage 1, and died at 2a.  Refuse immediately instead, with
# the two ways out stated.  Rewiring the ~15 reader sites to honour $OUT is the real fix and is a
# separate, testable change; this guard only stops the misdiagnosis.
if [ -n "${OUT_EXPLICIT:-}" ] || [ "${OUT:-}" != "$D/$TAG" ]; then
  if [ "${OUT:-}" != "$D/$TAG" ]; then
    echo "REPRO_GUARD_FAIL -- OUT ($OUT) must equal \$D/\$TAG ($D/$TAG): the scoped configs are written"
    echo "                    to OUT but read from \$D/\$TAG, so any other combination dies at stage 2a"
    echo "                    with a FileNotFoundError naming a cfg that was never meant to exist."
    echo "                    Either unset OUT (it defaults to \$D/\$TAG), or set TAG so that they agree."
    exit 1
  fi
fi
mkdir -p "$OUT"
cd "$D"

# The shipped configuration, in one place. Everything below reads these; nothing hard-codes a path.
# 2026-08-10: was XNXa, which trains on data/extra_paper_v5 -- a DIFFERENT SOURCE, banned by the
# 2026-08-05 rules ("training data must come from the competition's own release").  Running that
# recipe would have rebuilt an ILLEGAL model in the Decision Phase.  What ships is SDa =
# self-distilled RLa; note it is a TWO-STAGE chain, so the teacher must be trained first (stage 1a).
# ---------------------------------------------------------------- THE LIVE CARD, DERIVED
# 2026-08-19, defect class D6 (FOURTH recurrence).  Everything about the card used to be a
# hand-maintained snapshot here: template, head, calib, sband, four anchors, and a hard-coded
# LIVE= path for the md5 guard.  Nine values, drifting independently.  On 2026-08-19 the
# template/head/calib/sband all still named v70 while the card was v71_xcb, and because the
# guard compared the stale pair against the stale LIVE= it PASSED -- so a rebuild reproducing
# the stale sps 54.950 would print ACCEPT while the live 55.314 sat OUTSIDE the same window.
# Now all nine are derived from LIVE_CARD and verified BY MD5 against the shipped
# zip.  common/live_card.py refuses if the pointer and the zip disagree.
# ---------------------------------------------------------------- DPBOOTSTRAP (2026-09-12)
# Track*/submissions/ is gitignored, so a CLEAN CLONE HAS NO LIVE ZIP and the line below used to die
# here, before stage 0 -- which is also exactly the Decision-Phase situation, since the official
# requirement is training code only, no model weights.  BOOTSTRAP=1 resolves the card's IDENTITY from
# the committed docs/decision/live_card_anchors.json (numbers and hashes, not weights) instead.
# Default off: with BOOTSTRAP unset this script behaves byte-for-byte as before.
LC_BOOT=""
if [ "${BOOTSTRAP:-0}" = "1" ]; then
  LC_BOOT="--bootstrap"
  # With no zip there is nothing to reuse, so every artifact must be an OUTPUT of this run.  Refusing
  # here is the point: the alternative is a chain that silently packages whatever it finds lying around.
  [ "${REPRO_SCOPED:-0}" = "1" ] || { echo "REPRO_GUARD_FAIL -- BOOTSTRAP=1 requires REPRO_SCOPED=1: with no live zip the members and head must be OUTPUTS of this run"; exit 1; }
  [ "${BAND_PIPELINE:-0}" = "1" ] || { echo "REPRO_GUARD_FAIL -- BOOTSTRAP=1 requires BAND_PIPELINE=1: with no live zip, calib.npz and sband.pth must be OUTPUTS of stage 8b, not declared inputs"; exit 1; }
  echo "  [bootstrap] live-card identity from docs/decision/live_card_anchors.json -- no zip required"
  echo "  [bootstrap] the md5-against-zip guard cannot run; acceptance is the four accuracy anchors,"
  echo "              which are committed numbers and are still gated below"
fi
eval "$($PY "$R/common/live_card.py" --track 1 --emit shell $LC_BOOT)" || {
  echo "REPRO_GUARD_FAIL -- common/live_card.py could not resolve the live card"; exit 1; }
echo "LIVE CARD: $LIVE_NAME  zip md5 $LIVE_ZIP_MD5"
[ -n "${LIVE_DRIFT:-}" ] && echo "  !! NEWER card-shaped bundles exist ($LIVE_DRIFT). If one of
     them is now on the board, update Track1/LIVE_CARD before trusting this run."

# ---------------------------------------------------------------- DP-CHAIN-2GEN (2026-09-09)
# The live card (t1_v82_alldata since 2026-09-10; ALL81=1 rebuilds that generation in stage 8c on top of this one) ships EDISTMSH (convnext) + EDISTMSSW (selfwarp), each distilled at 0.5
# from THREE ship teachers (WSW + MIMOA + MIMOWSW, themselves distilled from RLB128); its head is fitted
# on the BLEND of the two students' 6-fold fleets (distilled from WSW_kf + MIMOA_kf + MIMOWSW_kf) and its
# learned band is Q=64 on fleet A = the two student fleets.  Until this edit the chain rebuilt the v74/v77
# CW128+WSW class and the clean-clone dry run failed its LIVE_CARD tke anchor by -0.341
# (2026-09-09).  GEN2=0 restores the old chain.
GEN2=${GEN2:-1}
if [ "$GEN2" = "1" ]; then
  SHIP_CFG=${SHIP_CFG:-configs/config_v4_ship_EDISTMSH.yaml}
  SECOND_CFG=${SECOND_CFG:-configs/config_v4_ship_EDISTMSSW.yaml}
  FOLD_CFG=${FOLD_CFG:-configs/config_v4_EDISTMSH_kf.yaml}
  STUDENT2_FOLD_CFG=${STUDENT2_FOLD_CFG:-configs/config_v4_EDISTMSSW_kf.yaml}   # the blend's 2nd student fleet (head --arm2, band fleet A2)
  WSW_SHIP_CFG=${WSW_SHIP_CFG:-configs/config_v4_ship_WSW.yaml}                 # now a TEACHER, not the 2nd member
  MIMOA_SHIP_CFG=${MIMOA_SHIP_CFG:-configs/config_v4_ship_MIMOA.yaml}
  MIMOWSW_SHIP_CFG=${MIMOWSW_SHIP_CFG:-configs/config_v4_ship_MIMOWSW.yaml}
  MIMOA_KF_CFG=${MIMOA_KF_CFG:-configs/config_v4_MIMOA_kf.yaml}
  MIMOWSW_KF_CFG=${MIMOWSW_KF_CFG:-configs/config_v4_MIMOWSW_kf.yaml}
  BAND_Q=${BAND_Q:-64}
else
  BAND_Q=${BAND_Q:-32}
fi
# ---------------------------------------------------------------- ALL81 (2026-09-10)
# LIVE_CARD moved to t1_v82_alldata (2026-09-10): the v79 recipe with the three ship teachers
# (WSW, MIMOA, MIMOWSW -- each still distilled from the 72-file RLB128) AND the two students (EDISTMSH,
# EDISTMSSW) retrained on the 81-file pool = local_bench/train (72) + local_bench/val (9, Re 11400/17775);
# 7575_0 stays banned.  The config delta is exactly {real-finetune glob -> the pool, exclude_re -> []} plus
# names/paths.  The card ships v79's head/calib/sband, fitted on the
# 72-file members, and was built with --allow-calib-mismatch.
# DESIGN RULE: stage 4 (head), stages 5-8 (base band) and 8b (learned band) stay on the 72-file members and
# the kf fleets -- stages 5 and 7 fit on ship-member residuals over the 9 interp files, which are IN-SAMPLE for
# an 81-file member.  Stage 8c trains the 81-file members; stage 9 swaps them in at PACKAGING ONLY.
# ALL81=0 restores the v79-generation packaging, whose anchors no longer match a v82 LIVE_CARD.
ALL81=${ALL81:-1}
if [ "$ALL81" = "1" ] && [ "$GEN2" != "1" ]; then
  echo "REFUSING: ALL81=1 needs GEN2=1 -- stage 8c retrains the GEN2 ship teachers and students on 81 files"; exit 1
fi
SHIP_CFG=${SHIP_CFG:-configs/config_v4_ship_CW128.yaml}      # the forecaster that ships as model.pth
# THE SECOND BLEND MEMBER.  The live card is a 2-member CROSS-RECIPE blend (CW128 convnext +
# WSW selfwarp, corr 0.938, XCB 2026-08-18) and this chain contained NO reference to a second
# forecaster at all -- grep for extra-checkpoint returned 0 hits -- so a from-scratch rebuild
# would have produced a ONE-member bundle and could not reach the card's row.  It distils from
# the SAME teacher as CW128 (unet_v4_ship_RLB128.pth), so this adds one student stage, not two.
SECOND_CFG=${SECOND_CFG:-configs/config_v4_ship_WSW.yaml}
TEACHER_CFG=${TEACHER_CFG:-configs/config_v4_ship_RLB128.yaml} # SDa distils from this; must exist first
FOLD_CFG=${FOLD_CFG:-configs/config_v4_CW128_kf.yaml}        # the 6 dedup-Reynolds folds the head's OOF needs
TEACHER_FOLD_CFG=${TEACHER_FOLD_CFG:-configs/config_v4_RLB_kf.yaml}
# BAND PIPELINE (2026-08-24).  The learned sps band needs FOUR 6-fold fleets; FOLD_CFG above is one
# of them.  These are the other three -- all distil from the SAME RLB teacher folds stage 2a trains,
# so they add 18 fold trainings and no new dependency.  Opt-in until it has accepted end-to-end.
# DEFAULT ON since 2026-08-25: accepted end-to-end (REPRODUCE_T1_OK; sps 55.225 vs anchor
# 55.314, tol +-0.3) on a scoped clean-room run whose forecasters AND band were both built
# from scratch.  BAND_PIPELINE=0 restores the old behaviour (band as a declared input).
BAND_PIPELINE=${BAND_PIPELINE:-1}
BAND_CFG_1=${BAND_CFG_1:-configs/config_v4_WSW_kf.yaml}
BAND_CFG_2=${BAND_CFG_2:-configs/config_v4_CW128s1_kf.yaml}
BAND_CFG_3=${BAND_CFG_3:-configs/config_v4_WSWs1_kf.yaml}
# ARM is --arm for train_vgain: the fold checkpoint stem.  It is NOT independent of FOLD_CFG --
# stage 2b (below) TRAINS FOLD_CFG's folds and then gates them with `skip $CKPTDIR/${ARM}_f$F.pth`,
# i.e. the same object under two names.  Until 2026-08-27 this line hardcoded the stem of the
# DEFAULT FOLD_CFG, so overriding FOLD_CFG alone did two silent wrong things: (1) the vgain head
# was fitted on the PLAIN fleet while the card shipped FOLD_CFG's forecaster, and (2) stage 2b's
# skip-test named the wrong file, so FOLD_CFG's folds could be skipped on a stale plain-fleet
# checkpoint.  There is no legitimate ARM != FOLD_CFG's stem: fitting the head on a fleet other
# than the shipped forecaster's is exactly what BCP forbids.  So DERIVE it, and refuse a conflict.
#
# Why the arm stamp added to train_vgain.py the same day does NOT close this on its own: that
# stamp catches a wrong ARM only when a stamped cache from a CORRECT run already exists.  A first
# clean-room run with a mismatched ARM builds its OWN cache, stamps it with the WRONG arm, and is
# self-consistently wrong -- which is precisely the Decision-Phase scenario (the organizers run
# from a clean tree).  The stamp is a backstop; this derivation is the fix.
_ARM_FROM_CFG=$(cfg_stem "$D/$FOLD_CFG")
if [ -n "${ARM:-}" ] && [ "$ARM" != "$_ARM_FROM_CFG" ] && [ "${REPRO_SCOPED:-0}" != "1" ]; then
  echo "REFUSING: ARM does not match FOLD_CFG."
  echo "  ARM      = $ARM   (set explicitly in the environment)"
  echo "  FOLD_CFG = $FOLD_CFG"
  echo "             -> checkpoint stem $_ARM_FROM_CFG"
  echo "  Stage 2b trains FOLD_CFG's folds but gates them on \${ARM}_f<k>.pth, and stage 4 fits the"
  echo "  vgain head on \${ARM}_f<k>.pth -- so a mismatch fits the head on a fleet the card does not"
  echo "  ship (BCP violation) and can skip training the fleet it does.  Remedy: unset ARM and let"
  echo "  it follow FOLD_CFG, or set FOLD_CFG to the config that produces $ARM."
  exit 1
fi
# Not gated under REPRO_SCOPED=1: that mode re-derives ARM from its own scoped cfg_fold.yaml a few
# lines below, so an explicit ARM was already discarded there.  Refusing would be a NEW false
# refusal on a mode that was already correct.
ARM=${ARM:-$_ARM_FROM_CFG}
FOLDS=${FOLDS:-"0 1 2 3 4 5"}                   # the kf manifest has SIX folds; the old AoA one had five
MANIFEST=${MANIFEST:-$R/common/local_benchmark/splits_kfold_dedup.json}
CKPTDIR=${CKPTDIR:-$R/checkpoints/t1_u-net}              # REPRO_SCOPED retargets this
# 2026-08-10: the SHIPPED head is width 64 / gmin 0.5 / gmax 3.0, fitted on the 6-fold dedup
# Reynolds manifest.  The old 32 / 0.2 / 15.0 on splits_aoa.json reproduced a DIFFERENT head.
HEAD_W=${HEAD_W:-64}
HEAD_GMIN=${HEAD_GMIN:-0.5}
HEAD_GMAX=${HEAD_GMAX:-3.0}
HEAD_W_TKE=${HEAD_W_TKE:-0.10}                # w_tke/w_rel at the live operating point; repo default is 1.0
HALFWIDTH=0.15
CHUNK=64
# WHICH SUBMISSION TEMPLATE.  The `MATCH=1` branch that used to live here ran mk_t1_match.py /
# mk_t1_v15.py to GENERATE the transductive (TRD) templates, which locate a window's successor inside
# the input array.  Since 2026-08-05 that is a DISQUALIFYING condition, not a scoring penalty
# ("producing or selecting a prediction using information from evaluation windows other than the one
# it belongs to is strictly prohibited"), so the branch and both generators are gone.  The default is
# the shipped inductive template.
# v51 = v43 + the single-chunk path (_ONE_MAX=16): BIT-EXACT to v43 at every N, ~11% faster at the
# small N the online harness appears to use.  v43 remains a valid fallback.
# DERIVED from the card (was hard-coded to v63_arm while the card ships v69_blend -- and the
# blend template is what consumes model_1.pth, so the stale value silently disabled the blend).
TEMPLATE=${TEMPLATE:-$LIVE_TEMPLATE}

# ---------------------------------------------------------------- tracked-and-present PRE-FLIGHT
# 2026-08-16.  The audit found every artifact of the live card untracked in git, so a clone died at
# stage 0; and the anchors still matched a bundle three recipe generations old, so a rebuild would
# have printed ACCEPT for something we never submitted.  Item D6 (2026-08-10) was the same defect,
# fixed for that day's config and never re-applied as the recipe moved.  This is the structural
# version.  TWO placement lessons are baked in, both learned by getting it wrong tonight:
#   (a) it must run BEFORE any training -- my first version sat after the head stage and only
#       fired 2 GPU-hours in;
#   (b) it must check INPUTS ONLY.  Under REPRO_SCOPED=1 the HEAD path is an OUTPUT the chain
#       produces, and guarding it produced a false REPRO_GUARD_FAIL on a healthy chain.
_guard_fail=0
# the three band fleets are chain INPUTS too since 2026-08-24 (stage 2c/8b) -- guard them
_gen2_cfgs=""
[ "$GEN2" = "1" ] && _gen2_cfgs="$D/$STUDENT2_FOLD_CFG $D/$WSW_SHIP_CFG $D/$MIMOA_SHIP_CFG $D/$MIMOWSW_SHIP_CFG $D/$MIMOA_KF_CFG $D/$MIMOWSW_KF_CFG"
for _p in "$D/$SHIP_CFG" "$D/$SECOND_CFG" "$D/$TEACHER_CFG" "$D/$FOLD_CFG" "$TEMPLATE" \
         "$D/$BAND_CFG_1" "$D/$BAND_CFG_2" "$D/$BAND_CFG_3" $_gen2_cfgs; do
  [ -e "$_p" ] || { echo "REPRO_GUARD: MISSING ON DISK  $_p"; _guard_fail=1; continue; }
  if [ -n "${RPDE_FINAL_SOURCE_MANIFEST:-}" ]; then
    "$PY" "$R/Track1/U-net/VariantDecisionRepro/verify_source_manifest.py" \
      --root "$R" --manifest "$RPDE_FINAL_SOURCE_MANIFEST" --path "$_p" \
      || { echo "REPRO_GUARD: SOURCE MANIFEST MISMATCH  ${_p#$R/}"; _guard_fail=1; }
  elif ! ( cd "$R" && git ls-files --error-unmatch "${_p#$R/}" >/dev/null 2>&1 ); then
    echo "REPRO_GUARD: NOT TRACKED IN GIT  ${_p#$R/}"; _guard_fail=1
  fi
done
[ "$_guard_fail" -eq 0 ] || { echo "REPRO_GUARD_FAIL -- a clone cannot reproduce this chain"; exit 1; }
echo "REPRO_GUARD_OK  ship=$SHIP_CFG second=$SECOND_CFG teacher=$TEACHER_CFG fold=$FOLD_CFG template=$(basename $TEMPLATE)"
case "$TEMPLATE" in
  *v14_match*|*v15_fast*|*v20*|*v21*)
    echo "REFUSING: $TEMPLATE is a TRD template -- cross-window information is disqualifying"; exit 1;;
esac
# 2026-08-12 (F5): acceptance is ACCURACY-ANCHORED, mirroring reproduce_t2.sh.  FINAL_LOCAL
# contains the local `time` subscore, whose contention noise was measured at 15x any usable
# tolerance (the old FINAL+-0.10 gate passed today by only 0.023 -- i.e. on the dice), so it is
# printed as INFORMATIONAL ONLY and never gated on.  Anchors = the t1_v48/v51 build (the live
# card) on the 378-window held-out stream; tolerances sit several rebuild-sd wide (8-seed
# from-scratch rebuild: priced sd 0.0147, per-subscore spread tke 0.125 / rel 0.020 / sps 0.09).
# DERIVED from the card's own verify_report.json.  The old snapshot was 96.624/81.123/97.103/
# 54.950 = the v70 row; against the live 55.314 the sps window [54.650,55.250] REJECTS the card
# and ACCEPTS a stale rebuild.  Tolerances stay where they were (several rebuild-sd wide).
TARGET_REL=${TARGET_REL:-$LIVE_REL};   TOL_REL=${TOL_REL:-0.10}
TARGET_TKE=${TARGET_TKE:-$LIVE_TKE};   TOL_TKE=${TOL_TKE:-0.30}
TARGET_MVPE=${TARGET_MVPE:-$LIVE_MVPE}; TOL_MVPE=${TOL_MVPE:-0.10}
TARGET_SPS=${TARGET_SPS:-$LIVE_SPS};   TOL_SPS=${TOL_SPS:-0.30}


say () { echo; echo "=============== $* ==============="; date; }
skip () { [ -e "$1" ] && { echo "  [have] $1 -- skipping"; return 0; } || return 1; }
# cfg_path <config> <key>: one config value with its repo-relative path(s) resolved onto THIS checkout
# (tracked configs store paths relative to the repo root; common/repo_paths.py).
cfg_path () { "$PY" -c "import sys,yaml; sys.path.insert(0, sys.argv[3]); from repo_paths import resolve_path; print(resolve_path(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]]))" "$1" "$2" "$R/common"; }

# ------------------------------------------------------------------ 0. (removed)
# Stage 0 used to run a converter that turned the ICLR-paper dataset (data/RealPDE_ICLR_paper_dataset) into training
# files -- manufacturing the banned extra_paper source from a different dataset.  Deleted 2026-08-10.
# The legal recipe reads only data/RealPDE_competition_dataset/gdrive_raw/**; train_v4.py now refuses anything else.
say "0/9  (no external inputs: the legal recipe reads only the competition release)"

if [ "${REPRO_SCOPED:-0}" = "1" ]; then
  # Clean-room mode: nothing below may read or write a production checkpoint.
  mkdir -p "$OUT/checkpoints"
  # ALL FOUR configs must be scoped.  Scoping only ship+fold (as this block did before the
  # self-distillation chain was added) would train the TEACHERS into the production checkpoints dir.
  _gen2_argv=""
  [ "$GEN2" = "1" ] && _gen2_argv="$D/$STUDENT2_FOLD_CFG $D/$WSW_SHIP_CFG $D/$MIMOA_SHIP_CFG $D/$MIMOWSW_SHIP_CFG $D/$MIMOA_KF_CFG $D/$MIMOWSW_KF_CFG"
  $PY - "$D/$SHIP_CFG" "$D/$FOLD_CFG" "$D/$TEACHER_CFG" "$D/$TEACHER_FOLD_CFG" "$OUT" "$R" "$D/$SECOND_CFG" \
      "$D/$BAND_CFG_1" "$D/$BAND_CFG_2" "$D/$BAND_CFG_3" $_gen2_argv <<'EOF'
import sys, yaml, pathlib
ship, fold, tship, tfold, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], pathlib.Path(sys.argv[5])
root = sys.argv[6].rstrip("/")
second = sys.argv[7]   # the blend's 2nd member: it must be scoped too, or the clean room
                       # trains it straight into the PRODUCTION checkpoints dir.
band_cfgs = sys.argv[8:11]  # the three extra band fleets -- same reasoning, same treatment.
gen2 = sys.argv[11:17] if len(sys.argv) >= 17 else None   # GEN2: student2_fold, wsw_ship, mimoa_ship, mimowsw_ship, mimoa_kf, mimowsw_kf
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
    stem = pathlib.Path(c["checkpoint"]).stem + "_" + out.name
    c["checkpoint"] = str(ck / (stem + ".pth"))
    c["variant"] = str(c.get("variant", "v")) + "_" + out.name
    dump_checked(c, out / nm)
    print("  scoped:", nm, "->", c["checkpoint"])
    return c
def student(src, nm, teachers):
    """scope a student and repoint its distill_ckpt at the SCOPED teachers (one or several, comma-joined);
    a scoped student MUST distil from scoped teachers, or the clean room reads a production artifact --
    the exact provenance violation this mode exists to prevent.  {fold} is preserved."""
    c = scope(src, nm)
    if c.get("distill_weight", 0.0) and c.get("distill_ckpt"):
        old = str(c["distill_ckpt"])
        n_old = len([x for x in old.split(",") if x])
        assert n_old == len(teachers), f"{nm}: {n_old} teachers in the tracked config, {len(teachers)} scoped -- refusing"
        suf = "_f{fold}" if "{fold}" in old else ""
        c["distill_ckpt"] = ",".join(str(pathlib.Path(t["checkpoint"]).parent / (pathlib.Path(t["checkpoint"]).stem + suf + ".pth")) for t in teachers)
        dump_checked(c, out / nm)
        print("    distill_ckpt ->", c["distill_ckpt"])
    return c
# teachers first, so the students can be repointed at the SCOPED teacher paths
t_ship = scope(tship, "cfg_teacher.yaml")
t_fold = scope(tfold, "cfg_teacher_fold.yaml")
if gen2 is None:
    for src, nm, tgt in ((ship, "cfg_ship.yaml", t_ship), (fold, "cfg_fold.yaml", t_fold),
                         (second, "cfg_second.yaml", t_ship)):
        student(src, nm, [tgt])
else:
    s2fold, wsw_ship, mimoa_ship, mimowsw_ship, mimoa_kf, mimowsw_kf = gen2
    # GEN2 lineage.  Ship: RLB128 -> {WSW, MIMOA, MIMOWSW} -> {EDISTMSH, EDISTMSSW}.
    #                Fold: RLB_kf -> {WSW_kf (= band fleet 1), MIMOA_kf, MIMOWSW_kf} -> {EDISTMSH_kf, EDISTMSSW_kf}.
    t_wsw = student(wsw_ship, "cfg_t_wsw.yaml", [t_ship])
    t_mimoa = student(mimoa_ship, "cfg_t_mimoa.yaml", [t_ship])
    t_mimowsw = student(mimowsw_ship, "cfg_t_mimowsw.yaml", [t_ship])
    student(ship, "cfg_ship.yaml", [t_wsw, t_mimoa, t_mimowsw])
    student(second, "cfg_second.yaml", [t_wsw, t_mimoa, t_mimowsw])
    # band fleet 1 (WSW_kf) is scoped below by the band loop; scope it here first so the students can name it
    b1 = student(band_cfgs[0], "cfg_band1.yaml", [t_fold])
    k_mimoa = student(mimoa_kf, "cfg_t_mimoa_kf.yaml", [t_fold])
    k_mimowsw = student(mimowsw_kf, "cfg_t_mimowsw_kf.yaml", [t_fold])
    student(fold, "cfg_fold.yaml", [b1, k_mimoa, k_mimowsw])
    student(s2fold, "cfg_fold2.yaml", [b1, k_mimoa, k_mimowsw])
# the three band fleets: scoped exactly like the students above, distilling from the SCOPED teacher
for _i, _src in enumerate(band_cfgs, start=1):
    if gen2 is not None and _i == 1:
        continue      # already scoped above (the students had to name it)
    _c = scope(_src, f"cfg_band{_i}.yaml")
    if _c.get("distill_weight", 0.0) and _c.get("distill_ckpt"):
        _old = str(_c["distill_ckpt"]); _b = pathlib.Path(t_fold["checkpoint"])
        _c["distill_ckpt"] = str(_b.parent / (_b.stem + ("_f{fold}" if "{fold}" in _old else "") + ".pth"))
        dump_checked(_c, out / f"cfg_band{_i}.yaml")
        print("    distill_ckpt ->", _c["distill_ckpt"])
EOF
  SHIP_CFG=$TAG/cfg_ship.yaml
  SECOND_CFG=$TAG/cfg_second.yaml
  FOLD_CFG=$TAG/cfg_fold.yaml
  TEACHER_CFG=$TAG/cfg_teacher.yaml
  TEACHER_FOLD_CFG=$TAG/cfg_teacher_fold.yaml
  BAND_CFG_1=$TAG/cfg_band1.yaml
  BAND_CFG_2=$TAG/cfg_band2.yaml
  BAND_CFG_3=$TAG/cfg_band3.yaml
  if [ "$GEN2" = "1" ]; then
    STUDENT2_FOLD_CFG=$TAG/cfg_fold2.yaml; WSW_SHIP_CFG=$TAG/cfg_t_wsw.yaml
    MIMOA_SHIP_CFG=$TAG/cfg_t_mimoa.yaml; MIMOWSW_SHIP_CFG=$TAG/cfg_t_mimowsw.yaml
    MIMOA_KF_CFG=$TAG/cfg_t_mimoa_kf.yaml; MIMOWSW_KF_CFG=$TAG/cfg_t_mimowsw_kf.yaml
  fi
  ARM=$(cfg_stem "$OUT/cfg_fold.yaml")
  CKPTDIR=$OUT/checkpoints
  HEAD=${HEAD:-$OUT/checkpoints/vgain_head_ship_$TAG.pth}
  echo "REPRO_SCOPED: ARM=$ARM CKPTDIR=$CKPTDIR"
fi

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

# ------------------------------------------------------------------ 1. the AoA fold manifest
say "1/9  AoA split manifest"
if ! skip $R/common/local_benchmark/splits_aoa.json; then
  $PY $R/common/local_benchmark/make_splits_aoa.py
fi
# The live configs read the DE-DUPLICATED Reynolds 6-fold manifest (it drops the 7575_0.h5 twin, whose
# fields are bit-identical to 6300_0.h5 under a contradictory label).  It has no producer in the repo
# yet -- see M4 -- so refuse rather than train on the wrong split.
if [ ! -e "$MANIFEST" ]; then
  echo "MISSING: $MANIFEST"
  echo "  The 6-fold dedup Reynolds manifest has no producer script in the repo yet (item M4)."
  echo "  It must drop 7575_0.h5 (bit-identical fields to 6300_0.h5, contradictory Re label)."
  exit 1
fi

# ------------------------------------------------------------------ 2. the 5 fold forecasters
# These exist ONLY to give the head an honestly out-of-fold forecast for every real file. They are
# not shipped. Skipping this stage silently would train the head on its own forecaster's training
# data, where the variance deficit is ~0 and the head learns "output 1".
# TEACHER FOLDS FIRST.  SDa is self-distilled, so each student fold needs the SAME fold's teacher
# (config_v4_SDa_kf.yaml's distill_ckpt is unet_v4_RLa_kf_base96_f{fold}.pth).  Sequential training is
# explicitly legal -- the organizers' rule is that every model in the chain retrains from the release --
# but it doubles this stage.
# C (RPDE_C_SHARED_OOF=1, shared_oof/README.md): the legacy 6-fold fleets of 2a-2e are not trained; stage 4 and 8b
# read the track members' Re3 fold versions and their seed-1 replicas (shared_oof_fleet.py) instead.
if [ "${RPDE_C_SHARED_OOF:-0}" = "1" ]; then
  echo "  [C] stages 2a-2e skipped: vgain and the band pipeline use the shared fivefold OOF fleets"
else
say "2a/9  teacher fold forecasters (RLa), the self-distillation source"
TEACHER_ARM=$(cfg_stem "$D/$TEACHER_FOLD_CFG")
for F in $FOLDS; do
  if ! skip $CKPTDIR/${TEACHER_ARM}_f${F}.pth; then
    $PY -u train_v4.py --config $D/$TEACHER_FOLD_CFG --fold $F --gpu $GPU 2>&1 | tail -3
  fi
done

if [ "$GEN2" != "1" ]; then
say "2b/9  student fold forecasters (SDa), the head's nested-OOF source"
for F in $FOLDS; do
  if ! skip $CKPTDIR/${ARM}_f${F}.pth; then
    $PY -u train_v4.py --config $D/$FOLD_CFG --fold $F --gpu $GPU 2>&1 | tail -3
  fi
done
fi

if [ "$BAND_PIPELINE" = "1" ]; then
say "2c/9  the three EXTRA fold fleets the learned band needs (WSW, CW128s1, WSWs1)"
# The band's head is fitted on the pooled geomean |r| of TWO 2-member fleets; FOLD_CFG supplies one
# member of fleet A and these supply the other three.  Without them the band stage cannot run, which
# is the whole reason it was a declared input rather than a producer (measured 2026-08-24).
BAND_STEMS=""
for BC in "$BAND_CFG_1" "$BAND_CFG_2" "$BAND_CFG_3"; do
  BSTEM=$(cfg_stem "$D/$BC")
  BAND_STEMS="$BAND_STEMS $BSTEM"
  for F in $FOLDS; do
    if ! skip $CKPTDIR/${BSTEM}_f${F}.pth; then
      $PY -u train_v4.py --config $D/$BC --fold $F --gpu $GPU 2>&1 | tail -3
    fi
    [ -e "$CKPTDIR/${BSTEM}_f${F}.pth" ] || { echo "FAIL: $CKPTDIR/${BSTEM}_f${F}.pth was not produced"; exit 1; }
  done
done
echo "  band fleets:$BAND_STEMS"
fi

if [ "$GEN2" = "1" ]; then
say "2d/9  GEN2 teacher fold fleets MIMOA_kf + MIMOWSW_kf (distilled from the RLB_kf folds of 2a)"
for BC in "$MIMOA_KF_CFG" "$MIMOWSW_KF_CFG"; do
  BSTEM=$(cfg_stem "$D/$BC")
  for F in $FOLDS; do
    if ! skip $CKPTDIR/${BSTEM}_f${F}.pth; then
      $PY -u train_v4.py --config $D/$BC --fold $F --gpu $GPU 2>&1 | tail -3
    fi
    [ -e "$CKPTDIR/${BSTEM}_f${F}.pth" ] || { echo "FAIL: $CKPTDIR/${BSTEM}_f${F}.pth was not produced"; exit 1; }
  done
done
say "2e/9  GEN2 student fold fleets EDISTMSH_kf (= ARM) + EDISTMSSW_kf (= ARM2), distilled from WSW_kf + MIMOA_kf + MIMOWSW_kf"
ARM2=$(cfg_stem "$D/$STUDENT2_FOLD_CFG")
for BC in "$FOLD_CFG" "$STUDENT2_FOLD_CFG"; do
  BSTEM=$(cfg_stem "$D/$BC")
  for F in $FOLDS; do
    if ! skip $CKPTDIR/${BSTEM}_f${F}.pth; then
      $PY -u train_v4.py --config $D/$BC --fold $F --gpu $GPU 2>&1 | tail -3
    fi
    [ -e "$CKPTDIR/${BSTEM}_f${F}.pth" ] || { echo "FAIL: $CKPTDIR/${BSTEM}_f${F}.pth was not produced"; exit 1; }
  done
done
echo "  GEN2 student fleets: ARM=$ARM ARM2=$ARM2"
fi

# ------------------------------------------------------------------ 3. the shipping forecaster
fi   # end of the legacy 2a-2e fleets (not in C mode)
say "3a/9  teacher shipping forecaster (RLa)"
TEACHER_CKPT=$(cfg_path "$D/$TEACHER_CFG" checkpoint)
if ! skip "$TEACHER_CKPT"; then
  $PY -u train_v4.py --config $D/$TEACHER_CFG --gpu $GPU 2>&1 | tail -3
fi

if [ "$GEN2" = "1" ]; then
say "3b/9  GEN2 ship teachers WSW + MIMOA + MIMOWSW (each distilled from RLB128 of 3a)"
for TC in "$WSW_SHIP_CFG" "$MIMOA_SHIP_CFG" "$MIMOWSW_SHIP_CFG"; do
  TCK=$(cfg_path "$D/$TC" checkpoint)
  if ! skip "$TCK"; then
    $PY -u train_v4.py --config $D/$TC --gpu $GPU 2>&1 | tail -3
  fi
  [ -e "$TCK" ] || { echo "FAIL: $TCK was not produced"; exit 1; }
done
fi
say "3b/9  shipping forecaster (member 0; GEN2: EDISTMSH distilled from the three teachers above)"
SHIP_CKPT=$(cfg_path "$D/$SHIP_CFG" checkpoint)
if ! skip "$SHIP_CKPT"; then
  $PY -u train_v4.py --config $D/$SHIP_CFG --gpu $GPU 2>&1 | tail -3
fi

say "3c/9  the blend's SECOND forecaster (selfwarp, ships as model_1.pth; GEN2: EDISTMSSW)"
# Distils from the same teacher trained in 3a, so no extra teacher stage is needed.  WSW is
# 4.24M params against CW128's 7.96M; the members share 93.8% of their residual variance
# (against 96.2% for two seeds of one recipe), which is why the blend beats the closed
# same-recipe ensemble cells.
SECOND_CKPT=$(cfg_path "$D/$SECOND_CFG" checkpoint)
# FINAL MODE (RPDE_FINAL_MODE=1, 2026-09-30): the second member feeds only the stage-9 packaging bundle; the Final
# card's members come from the models graph, so it is not trained.
if [ "${RPDE_FINAL_MODE:-0}" = "1" ]; then
  echo "  [final mode] 3c skipped (no consumer in the Final card)"
elif ! skip "$SECOND_CKPT"; then
  $PY -u train_v4.py --config $D/$SECOND_CFG --gpu $GPU 2>&1 | tail -3
fi

# ------------------------------------------------------------------ 4. head
say "4/9  variance-deficit head (nested OOF over all 5 folds)"
# Overridable, and it MUST be overridden for a new arm: the head is fitted to the forecaster's
# residuals, and `skip` below would otherwise silently reuse another arm's head.
# DERIVED (was vgain_head_cw128_w0100.pth md5 a916e0e6; the card ships 1abc682b, the blend's
# OWN refit head -- per BCP a card carrying a new forecaster ships its own head AND band).
HEAD=${HEAD:-$LIVE_HEAD}
VG_CKPTDIR=$CKPTDIR; VG_MANIFEST=$MANIFEST
if [ "${RPDE_C_SHARED_OOF:-0}" = "1" ]; then
  # C: wait until the graph thread has trained the Re3 members + seed-1 replicas and linked the fleet (ready.json).
  _w=$SECONDS
  until [ -e "$RPDE_C_READY" ]; do
    [ ! -e "$RPDE_C_FAILED" ] || { echo "FAIL: the shared-OOF fleet side failed ($RPDE_C_FAILED)"; exit 1; }
    [ $((SECONDS-_w)) -lt 172800 ] || { echo "FAIL: shared-OOF fleet wait timeout"; exit 1; }
    sleep 60
  done
  ARM=$RPDE_C_ARM; ARM2=$RPDE_C_ARM2; VG_CKPTDIR=$RPDE_C_FLEET_DIR; VG_MANIFEST=$RPDE_C_MANIFEST
  echo "  [C] vgain on the shared fivefold OOF: arm=$ARM arm2=$ARM2 fleet=$VG_CKPTDIR"
fi

if ! skip "$HEAD"; then
  # VGAIN_W_TKE is w_tke/w_rel at the live operating point.  The repo default is 1.0 (equal weight),
  # which was correct only while the three accuracy subscores were believed to share one weight -- an
  # assumption refuted 2026-08-08 at 102x the residual of the free fit.
  _arm2=""; [ "$GEN2" = "1" ] && _arm2="--arm2 $ARM2"     # the live head is fitted on the BLEND (edistmf.py stage 4)
  VGAIN_W_TKE=$HEAD_W_TKE $PY -u train_vgain.py --arm $ARM $_arm2 --ckpt-dir "$VG_CKPTDIR" --gpu $GPU \
      --width $HEAD_W --gmin $HEAD_GMIN --gmax $HEAD_GMAX --manifest "$VG_MANIFEST" \
      --cache $OUT/vgain_oof_cache.npz --ship "$HEAD" 2>&1 | tail -6
fi

# ------------------------------------------------------------------ 5. band, FROM SCRATCH
# t1_band_v5 needs no prior calibration -- it streams the forecaster over val and fits the
# per-lead-time band directly. `--head` (added 2026-07-29) puts the shipped head in the fitting
# path, so the band is fitted to the residuals the submission actually produces.
say "5/9  per-lead-time band, fitted from scratch (no historical bundle)"
BAND0=$OUT/calib_band0.npz
if ! skip "$BAND0"; then
  $PY t1_band_v5.py --checkpoints "$SHIP_CKPT" --head "$HEAD" --gpu $GPU \
      --out "$BAND0" --json $OUT/band0.json 2>&1 | tail -8
fi

# t1_band_v5 writes band_lo/band_hi over the 2 MEASURED channels; the submission indexes 3. A
# missing third column makes the reader fall back to the constant band silently -- measured on
# 2026-07-28 as a 2-point sps loss that looked like a modelling result and was not.
$PY - "$BAND0" <<'EOF'
import sys, numpy as np
p = sys.argv[1]; z = dict(np.load(p))
for k in ("band_lo", "band_hi"):
    a = np.asarray(z[k], np.float32)
    if a.shape[-1] == 2:
        pad = np.zeros((a.shape[0], 1), np.float32)
        z[k] = np.concatenate([a, pad], axis=-1)
        print(f"  padded {k} {a.shape} -> {z[k].shape} (channel 2 is pressure, unscored)")
np.savez(p, **z)
EOF

# ------------------------------------------------------------------ 6. bootstrap bundle
# Exists for exactly one reason: t1_sps_v2 reads its baseline band out of a ZIP. Not shipped.
say "6/9  throwaway bootstrap bundle (carries the band into the activity fit)"
BOOT=$D/submissions/_${TAG}_boot.zip
if ! skip "$BOOT"; then
  # band0 has no activity keys yet (stage 7 fits them) but the builder's anti-v26 assert
  # refuses a dead _KCELL -- bridge with NEUTRAL keys (bit-identical band).  Found by the
  # first real end-to-end run, 2026-08-11.  And tee, never bare-grep: a bare grep EATS the
  # builder's traceback and turns a real failure into a silent set-e exit (same incident).
  $PY t1_boot_calib.py "$BAND0" "$OUT/calib_band0_boot.npz"
  $PY build_t1_bundle.py --checkpoint "$SHIP_CKPT" --head "$HEAD" --calib "$OUT/calib_band0_boot.npz" \
      --template "$TEMPLATE" --halfwidth $HALFWIDTH --chunk $CHUNK --gpu $GPU \
      --name _${TAG}_boot 2>&1 | tee $OUT/boot_build.log | tail -4
  grep -q "BUILD_OK" $OUT/boot_build.log
fi

# ------------------------------------------------------------------ 7. activity scaling
say "7/9  heteroscedastic activity scaling (k_cell / act_const / p / lam / self_norm)"
HET=$OUT/calib_hetero.npz
if ! skip "$HET"; then
  $PY t1_sps_v2.py --checkpoint "$SHIP_CKPT" --bundle "$BOOT" --gpu $GPU \
      --out "$HET" --json $OUT/sps_v2.json 2>&1 | tail -10
fi

# ------------------------------------------------------------------ 8. merge
say "8/9  merge band + activity scaling into the shipping calibration"
CALIB=$OUT/calib_repro.npz
if ! skip "$CALIB"; then
  # 2026-08-04 XNXa incident: v8_verify defaults to the FROZEN v3/v4-era templates, which
  # cannot import a convnext checkpoint. The negative control lives in the CALIB (band-only
  # calib has no hetero keys -> k7 False), not in the template era. Both drives use the
  # MODERN IND template (v6): under a TRD template 369/378 windows take the zero-width
  # substitution band, diluting the hetero sps gain 40x below the +0.3 assert (measured
  # +0.0163 vs +0.81 at full coverage) — the assert needs full band exposure.
  $PY t1_v8_verify.py --v7-calib "$BAND0" --hetero "$HET" --out-calib "$CALIB" \
      --checkpoint "$SHIP_CKPT" --chunk $CHUNK --halfwidth $HALFWIDTH \
      --tpl-v3 "$D/submission_t1_v6.py" --tpl-v4 "$D/submission_t1_v6.py" \
      --gpu $GPU --json $OUT/v8_verify.json 2>&1 | tail -10
fi

# ------------------------------------------------------------------ 8b. the LEARNED band
if [ "$BAND_PIPELINE" = "1" ]; then
say "8b/9  learned sps band from the chain's OWN fleets (t1_band_pipeline.sh)"
# $CALIB from stage 8 IS the base band (band_lo/hi + gamma + k_cell + act_* + p + lam + self_norm).
# The pipeline applies width 0.90, caches the OOF residuals over the four fleets trained above, fits
# the sband head, packs it onto that base and drops pc_rl/pc_rh.  It also PROVENANCE-STAMPS the
# result against $SHIP_CKPT, which is what makes T1's provenance gate active instead of inert.
BP_OUT=$OUT/band_pipeline
BP_STEM_A1=$(cfg_stem "$D/$FOLD_CFG")
BP_STEM_A2=$(cfg_stem "$D/$BAND_CFG_1")
[ "$GEN2" = "1" ] && BP_STEM_A2=$ARM2     # GEN2: fleet A = the two STUDENT fleets (edistmf.py stage 8b), B unchanged
BP_STEM_B1=$(cfg_stem "$D/$BAND_CFG_2")
BP_STEM_B2=$(cfg_stem "$D/$BAND_CFG_3")
# t1_sband_cache.py resolves a stem as checkpoints/t1_u-net/<stem>_f<k>.pth, so a scoped run must expose its
# scoped folds under that name.  Symlink rather than copy: 24 x 90 MB is not worth duplicating.
BP_CKPTDIR=$CKPTDIR; BP_FOLDS=$FOLDS
if [ "${RPDE_C_SHARED_OOF:-0}" = "1" ]; then   # C: fleet A = the members' Re3 folds, fleet B = their seed-1 replicas
  BP_STEM_A1=$RPDE_C_ARM; BP_STEM_A2=$RPDE_C_ARM2; BP_STEM_B1=$RPDE_C_FLEET_B1; BP_STEM_B2=$RPDE_C_FLEET_B2
  BP_CKPTDIR=$RPDE_C_FLEET_DIR; BP_FOLDS=$RPDE_C_FOLD_IDS; export SBAND_MANIFEST=$RPDE_C_MANIFEST
  echo "  [C] band fleets A=$BP_STEM_A1,$BP_STEM_A2 B=$BP_STEM_B1,$BP_STEM_B2 (Re3 folds $BP_FOLDS)"
fi
mkdir -p "$R/checkpoints/t1_u-net"
for S in "$BP_STEM_A1" "$BP_STEM_A2" "$BP_STEM_B1" "$BP_STEM_B2"; do
  for F in $BP_FOLDS; do
    [ -e "$R/checkpoints/t1_u-net/${S}_f${F}.pth" ] || ln -s "$BP_CKPTDIR/${S}_f${F}.pth" "$R/checkpoints/t1_u-net/${S}_f${F}.pth"
  done
done
if ! skip "$BP_OUT/calib_rebuilt.npz"; then
MODE=scratch TAG=${TAG}_band GPU=$GPU SCORE=0 PY="$PY" OUT="$BP_OUT" Q=$BAND_Q \
  BASE100="$CALIB" PROV_CKPT="$SHIP_CKPT" \
  SBAND_CACHE="$BP_OUT/cache" \
  SBAND_FLEET_A="$BP_STEM_A1,$BP_STEM_A2" SBAND_FLEET_B="$BP_STEM_B1,$BP_STEM_B2" \
  bash "${RPDE_FINAL_BAND_SCRIPT:-$D/t1_band_pipeline.sh}" 2>&1 | tee $OUT/band_pipeline.log | tail -25
fi
# OUTSIDE the skip: a reused artifact must still be gated.  `skip` is a pure existence test, so
# without this a band that FAILED acceptance on an earlier run would be adopted silently on the
# next one.  (t1_band_pipeline.sh no longer writes the artifact on a failure path either -- belt
# and braces, because either fix alone leaves a hole if the other is edited later.)
grep -q "T1_BAND_PIPELINE_OK" $OUT/band_pipeline.log \
  || { echo "FAIL: no T1_BAND_PIPELINE_OK in $OUT/band_pipeline.log (stale or failed band)"; exit 1; }
SHIP_CALIB="$BP_OUT/calib_rebuilt.npz"
SHIP_SBAND="$BP_OUT/cache/sband_${TAG}_band.pth"
[ -e "$SHIP_CALIB" ] || { echo "FAIL: no $SHIP_CALIB"; exit 1; }
[ -e "$SHIP_SBAND" ] || { echo "FAIL: no $SHIP_SBAND"; exit 1; }
echo "  band  $SHIP_CALIB   (REBUILT -- no longer a declared input)"
echo "  sband $SHIP_SBAND"
# Clean up the resolution symlinks: t1_sband_cache.py hard-codes checkpoints/t1_u-net/<stem>_f<k>.pth, so
# a scoped run has to expose its folds there, but leaving them behind turns into DANGLING links
# the moment $OUT is deleted.  Remove exactly the ones this stage created, and only if they are
# still symlinks (never touch a real file).
for S in "$BP_STEM_A1" "$BP_STEM_A2" "$BP_STEM_B1" "$BP_STEM_B2"; do
  for F in $BP_FOLDS; do
    [ -L "$R/checkpoints/t1_u-net/${S}_f${F}.pth" ] && rm -f "$R/checkpoints/t1_u-net/${S}_f${F}.pth"
  done
done
echo "  (removed this stage's fold-resolution symlinks from $R/checkpoints/t1_u-net)"
fi

# FINAL MODE (2026-09-30): 8c (ALL81 packaging members) and 9 (v82-generation bundle + in-sample anchors) have no
# consumer in the Final card, which takes vgain (stage 4) and the band pipeline (8b) from this chain.
if [ "${RPDE_FINAL_MODE:-0}" = "1" ]; then
  echo "  [final mode] stages 8c/9 skipped (no consumer in the Final card)"
  echo "REPRODUCE_T1_OK"
  exit 0
fi
# ------------------------------------------------------------------ 8c. ALL81: the 81-file members (v82)
if [ "$ALL81" = "1" ]; then
say "8c/9  ALL81: ship teachers WSW/MIMOA/MIMOWSW + students EDISTMSH/EDISTMSSW on the 81-file pool (packaging-only swap)"
A81=$OUT/all81
mkdir -p "$A81"
# (a) The pool is a SYMLINK FARM over the released files.  Under REPRO_SCOPED=1 it lives under $OUT like every
# other output of the clean room: a clone's data/RealPDE_competition_dataset/gdrive_raw may itself be a symlink into another tree (on one host's
# scratch clone it points at the production checkout), and materializing through it would write outside the
# clone.  The path keeps the /data/RealPDE_competition_dataset/gdrive_raw/ component train_v4.py's provenance gate requires.  The generator
# refuses if the first EXISTING ancestor of the pool resolves outside the containment root -- checked before mkdir.
if [ "${REPRO_SCOPED:-0}" = "1" ]; then
  A81_POOL=${A81_POOL:-$OUT/data/RealPDE_competition_dataset/gdrive_raw/local_bench/all81/test_real}; A81_CONTAIN=$OUT
else
  A81_POOL=${A81_POOL:-$R/data/RealPDE_competition_dataset/gdrive_raw/local_bench/all81/test_real}; A81_CONTAIN=$R
fi
# (b) Configs, from the chain's OWN ship configs (the scoped ones under REPRO_SCOPED=1): YAML readback, diff-set
# assertion, checkpoint-collision assertion, stale-recipe refusal.
$PY - "$R/data/RealPDE_competition_dataset/gdrive_raw/local_bench" "$A81_POOL" "$A81_CONTAIN" "$A81" "${REPRO_SCOPED:-0}" "$TEACHER_CKPT" "$R" \
    "$D/$WSW_SHIP_CFG" "$D/$MIMOA_SHIP_CFG" "$D/$MIMOWSW_SHIP_CFG" "$D/$SHIP_CFG" "$D/$SECOND_CFG" \
    "$CKPTDIR" "$R/checkpoints/t1_u-net" <<'A81GEN'
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
sfx = f"_{a81.parent.name}" if scoped else ""          # the scoping block's `stem + "_" + out.name`
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
                f"checkpoint collision: {p} exists but stage 8c never ran {m} (no {a81}/train_{m}.out) -- refusing to overwrite it"
            print(f"  (resume) {p.name} exists and was produced by this stage ({m}'s run log exists)")
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
        # a teacher distils from the chain's OWN 72-file RLB128 (stage 3a) -- the v82 lineage, never a production file
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
# checkpoint, which train_v4.py writes incrementally; a resume skips a member only on that line.
a81_train () {
  local mem=$1 cfg log var ck rc t
  cfg=$A81/cfg_a81_$1.yaml; log=$A81/train_$1.out
  var=$($PY -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['variant'])" "$cfg")
  ck=$(cfg_path "$cfg" checkpoint)
  if [ -e "$log" ] && grep -q "^TRAIN_V4_OK $A81/Logger/$var/" "$log"; then
    echo "  [have] $mem -- TRAIN_V4_OK for $var in $log -- skipping"
  else
    if [ -e "$log" ]; then
      mv "$log" "$log.incomplete.$(date -u +%Y%m%dT%H%M%SZ)"; echo "  (an incomplete run log of $mem was moved aside)"
    fi
    echo "  training $mem ($var) on GPU $GPU -> $log   [$(date -u +%FT%TZ)]"
    rc=0; $PY -u train_v4.py --config "$cfg" --gpu $GPU > "$log" 2>&1 || rc=$?
    tail -3 "$log"
    grep -q "^TRAIN_V4_OK $A81/Logger/$var/" "$log" || { echo "FAIL: $mem (rc=$rc) -- no TRAIN_V4_OK for $var in $log"; exit 1; }
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
echo "  ALL81 students OK: model.pth <- $A81_SHIP_CKPT ; model_1.pth <- $A81_SECOND_CKPT"
fi

# ------------------------------------------------------------------ 9. build + ACCEPTANCE TEST
say "9/9  shipping bundle + acceptance test"
# --------------------------------------------------------------------------------------------
# NOT YET REPRODUCIBLE, AND SAID OUT LOUD RATHER THAN HIDDEN (items M1 and M2).
# Stages 5-8 fit a band FROM SCRATCH, which is NOT the shipped band: the shipped calib descends from a
# per-cell then per-lead fit, each done CONDITIONALLY on an already-built bundle, and the shipped sband
# head comes from a 30-epoch fit over two 6-fold GM fleets whose cache lands outside the repo.  Neither
# has a producer in the repo.  So they are inputs here, defaulting to the real artifacts, and the gap is
# printed.  Do not delete this notice until t1_band_pipeline.sh and the sband stage exist.
# 2026-08-19: THE SNAPSHOT IS GONE.  Re-anchoring by hand is what failed four times (D6, on
# 2026-08-10, -08-16, -08-18 and again -08-19: the -08-18 re-anchor named v70 and the card moved to
# v71_xcb the same day).  Both artifacts are now DERIVED from LIVE_CARD and checked by
# md5 against the shipped zip, so the next card cannot drift silently -- see common/live_card.py.
# DERIVED.  Was calib_MYpcnull.npz (7acaedd0) + sband_cw128.pth (16d4eb39); the card ships
# 8a4cc064 + 2dfdbb70.  The sband head in particular had NO copy in the repo at all -- its only
# copy was on a host's WIPEABLE SCRATCH, rescued to
# sband/sband_xcb.pth on 2026-08-19.  live_card.py locates it by md5, so it follows the file.
SHIP_CALIB=${SHIP_CALIB:-$LIVE_CALIB}
SHIP_SBAND=${SHIP_SBAND:-$LIVE_SBAND}
# GUARD -- the failure mode being closed is a SILENT PASS, so this has to be loud.
# The old version compared 2 artifacts (calib, sband) against a HARD-CODED LIVE= directory.  Both
# halves went stale together, so it PASSED on the stale pair: a snapshot cannot police a snapshot.
# Now all FIVE packaged artifacts are compared against the members inside the live card's own ZIP,
# which cannot drift.  Verified 2026-08-19 in both directions: the correct quintuple passes, and
# substituting any single v70 artifact refuses.
# 2026-08-24, first clean-clone run: THE SAME MISTAKE AS COMMENT (b) ABOVE, ONE STAGE LATER.
# This guard md5s all five packaged artifacts against the live zip.  Three of them --
# model.pth, model_1.pth, vgain.pth -- are INPUTS only when the chain reuses the live artifacts.
# Under REPRO_SCOPED=1 (the mode that models the Decision Phase: retrain everything in a clean
# room) they are OUTPUTS this run just produced, and no honest retrain can md5-match a previous
# one.  So the guard fired AFTER all nine stages of training, every time, and the acceptance test
# below had NEVER RUN in the only mode that matters.  Measured 2026-08-24: 9/9 stages
# completed, then REPRO_GUARD_FAIL on exactly those three hashes.
# The fix mirrors (b): md5-guard the artifacts that are INPUTS in THIS mode.  calib.npz and
# sband.pth are declared inputs in every mode (no in-repo producer for the live band's build
# ORDER), so they are always checked; the retrained trio is gated by the ACCURACY ANCHORS below,
# whose tolerances are already set several from-scratch rebuild-sd wide for exactly this purpose.
# Under BAND_PIPELINE=1 the band is an OUTPUT of stage 8b, exactly as the trio is an output under
# REPRO_SCOPED=1 -- md5-guarding an output is the (b) defect this block already documents.  It is
# gated by the sps ANCHOR below instead.
# ALL81 (2026-09-10): what gets PACKAGED.  ALL81=1 swaps in stage 8c's 81-file students here and only here; the
# head, band and sband stay the 72-file fits (design rule), so the build overrides the band's provenance stamp
# with --allow-calib-mismatch exactly as t1_v82_alldata was built, and a
# distinct bundle name + build log keep the 72-file stage-9 output from being reused or re-parsed.
PKG_CKPT=$SHIP_CKPT; PKG_CKPT2=$SECOND_CKPT; PKG_NAME=${BUNDLE:-t1_$TAG}; PKG_LOG=$OUT/build.log; PKG_ALLOW=""
if [ "$ALL81" = "1" ]; then
  PKG_CKPT=$A81_SHIP_CKPT; PKG_CKPT2=$A81_SECOND_CKPT
  PKG_NAME=${BUNDLE_ALL81:-t1_${TAG}_all81}; PKG_LOG=$OUT/build_all81.log; PKG_ALLOW="--allow-calib-mismatch"
  echo "  ALL81=1 -- packaging the 81-file students with the head/band fitted on the 72-file members:"
  echo "     model.pth   $PKG_CKPT"
  echo "     model_1.pth $PKG_CKPT2"
  echo "     vgain.pth   $HEAD"
  echo "     bundle $PKG_NAME   build log $PKG_LOG   (the [prov] check below reports a GAP by design)"
fi
if [ "$BAND_PIPELINE" = "1" ]; then _CHECK=""; else _CHECK="calib.npz=$SHIP_CALIB,sband.pth=$SHIP_SBAND"; fi
if [ "${REPRO_SCOPED:-0}" = "1" ]; then
  echo "  [guard] REPRO_SCOPED=1 -- model.pth / model_1.pth / vgain.pth are OUTPUTS of this run"
  echo "          (retrained from scratch); they are gated by the accuracy anchors below, not by"
  echo "          md5.  calib.npz + sband.pth remain declared INPUTS and ARE md5-checked."
else
  _CHECK="model.pth=$PKG_CKPT,model_1.pth=$PKG_CKPT2,vgain.pth=$HEAD,$_CHECK"
fi
if [ -n "$_CHECK" ]; then
  $PY "$R/common/live_card.py" --track 1 --check "$_CHECK" $LC_BOOT \
    || { echo "REPRO_GUARD_FAIL -- see above; re-anchor LIVE_CARD or fix the override"; exit 1; }
else
  echo "  [guard] BAND_PIPELINE=1 -- calib.npz + sband.pth are OUTPUTS of stage 8b (rebuilt from"
  echo "          this run's own fleets); they are gated by the sps anchor below, not by md5."
fi
echo
if [ "$BAND_PIPELINE" = "1" ]; then
echo "  BAND + SBAND ARE OUTPUTS OF THIS RUN (stage 8b), not declared inputs:"
echo "     band  $SHIP_CALIB"
echo "     sband $SHIP_SBAND"
echo "     Built by t1_band_pipeline.sh from THIS run's own four 6-fold fleets (stage 2b + 2c),"
echo "     on the from-scratch base band of stage 8.  The ORDER it applies -- width 0.90 (a uniform"
echo "     x0.90 on the 8 band arrays), sband cache -> fit -> pack, then PCDEL (drop pc_rl/pc_rh) --"
echo "     was recovered from the tracked artifacts on 2026-08-24 because bwid/pcnull/PCDEL had NO"
echo "     producer anywhere in the repo.  BCD is not a calib step at all: it is the template"
echo "     constant _BCD=1.  The band is PROVENANCE-STAMPED against the packaged checkpoint, so the"
echo "     [prov] line below is a real check rather than the warn-and-continue the unstamped live"
echo "     calib produced.  Gated by the sps ANCHOR below, not by md5 -- an md5 on a rebuilt npz"
echo "     guards the library version, not the data (measured 2026-08-24)."
echo "     KNOWN LIMIT: the shipped 2026-08-18 card's own band predates this pipeline and its head"
echo "     names the RSa base96 fleet, so THAT artifact is still not bit-reproducible; what is"
echo "     reproducible is the RECIPE, which is what a from-scratch retrain needs.  Measured three"
echo "     ways, the sband head is nearly fleet-insensitive (sps 55.315 / 55.314 / 55.314)."
else
echo "  !! NOT-YET-REPRODUCIBLE INPUTS (items M1/M2 above):"
echo "     band  $SHIP_CALIB"
echo "     sband $SHIP_SBAND"
echo "     The sband cache->fit->pack chain exists in-repo (t1_sband_{cache,fit,pack}.py), and since"
echo "     2026-08-24 so does the BUILD ORDER: Track1/U-net/t1_band_pipeline.sh."
echo "       width 0.90 = a uniform x0.90 on band_{lo,hi}{,_phys} and sb_{lo,hi}{,_phys} (8 arrays);"
echo "       PCDEL     = delete pc_rl and pc_rh;  BCD is NOT a calib step (template const _BCD=1)."
echo "     MEASURED: rebuilding the band with EVERY input replaced (24 retrained fold checkpoints,"
echo "     a refitted head, sb_lo_phys 34% different in max-norm) and A/B-ing it against the live"
echo "     band ON THE SAME FORECASTER gives sps 55.314 vs 55.314, replicated on a second refit."
echo "     CORRECTION (2026-08-24): this notice used to argue that the 08-19 refit reproducing sps"
echo "     55.315 proved the shipped band was fitted on the blend own residuals (BCP).  IT DOES NOT"
echo "     FOLLOW -- a matching sps shows the head is INSENSITIVE to the fleet, not where it came"
echo "     from.  sband_xcb.pth own fit_kind names unet_v4_RSa_kf_base96, so the LIVE band was NOT"
echo "     refit on the blend.  The insensitivity is now measured three ways (55.315/55.314/55.314),"
echo "     which is why a band refit on a from-scratch retrain is expected to land in the same place."
echo "     These two remain DECLARED INPUTS here only because this chain does not yet invoke the"
echo "     pipeline: the band needs FOUR 6-fold fleets (CW128, WSW and both s1 seeds) and this chain"
echo "     trains TWO (FOLD_CFG + the RLB teacher).  All three missing fleets distil from the RLB"
echo "     folds this chain already builds, so wiring them in is 18 extra fold trainings, no new"
echo "     dependency.  Until then stages 5-8 fit a DIFFERENT band than the submitted one."
fi
[ -e "$SHIP_CALIB" ] || { echo "MISSING $SHIP_CALIB"; exit 1; }
[ -e "$SHIP_SBAND" ] || { echo "MISSING $SHIP_SBAND"; exit 1; }
# BAND PROVENANCE (2026-08-24).  build_t1_bundle._prov_check silently warn-and-returns when the
# calib carries no prov_ckpt_md5, which is the case for every calib packed before 2026-08-19 --
# including the live card's.  T2's equivalent gate FIRED on the identical situation the same night.
# Say out loud which of the three cases this run is in; never let it pass as "checked".
$PY - "$SHIP_CALIB" "$PKG_CKPT" <<'PROVEOF'
import sys, hashlib, numpy as np
calib, ckpt = sys.argv[1], sys.argv[2]
try:
    with np.load(calib, allow_pickle=True) as z:
        stamp = str(z["prov_ckpt_md5"]) if "prov_ckpt_md5" in z.files else ""
        name = str(z["prov_ckpt_name"]) if "prov_ckpt_name" in z.files else ""
except Exception as e:
    print(f"  [prov] PROV GAP: cannot read {calib}: {e}"); raise SystemExit(0)
h = hashlib.md5(open(ckpt, "rb").read()).hexdigest()
if not stamp:
    print( "  [prov] PROV GAP -- the packaged band carries NO provenance stamp, so it CANNOT be")
    print(f"         verified to have been fitted on this forecaster.  calib {calib}")
    print(f"         packaged checkpoint md5 {h}")
    print( "         Cause: packed before the 2026-08-19 stamping fix in t1_sband_pack.py; the")
    print( "         producer stamps now, so the NEXT card closes this.  Track 2's gate fired on")
    print( "         the identical situation because its calib IS stamped.")
elif stamp != h:
    print( "  [prov] PROV GAP -- band fitted on a DIFFERENT forecaster (declared input, see above):")
    print(f"         fitted on {name or '?'} md5 {stamp}   packaging md5 {h}")
else:
    print(f"  [prov] ok: band provenance matches the packaged checkpoint ({h[:8]}...)")
PROVEOF
$PY build_t1_bundle.py --checkpoint "$PKG_CKPT" --extra-checkpoints "$PKG_CKPT2" \
    --head "$HEAD" --calib "$SHIP_CALIB" \
    --sband "$SHIP_SBAND" \
    --template "$TEMPLATE" --halfwidth $HALFWIDTH --chunk $CHUNK --gpu $GPU \
    --name $PKG_NAME $PKG_ALLOW 2>&1 | tee $PKG_LOG | grep -E "BUILD_OK|FINAL|score|\[prov\]"

REL=$(grep -oE '"rel_l2_score": [0-9.]+' $PKG_LOG | tail -1 | grep -oE '[0-9.]+$')
TKE=$(grep -oE '"tke_score": [0-9.]+' $PKG_LOG | tail -1 | grep -oE '[0-9.]+$')
MVPE=$(grep -oE '"mvpe_score": [0-9.]+' $PKG_LOG | tail -1 | grep -oE '[0-9.]+$')
SPS=$(grep -oE '"sps_score": [0-9.]+' $PKG_LOG | tail -1 | grep -oE '[0-9.]+$')
GOT=$(grep -oE "FINAL_LOCAL=[0-9.]+" $PKG_LOG | tail -1 | cut -d= -f2)
# S11: an empty value used to reach python as float('') and raise, indistinguishable from a real FAIL.
for v in "$REL" "$TKE" "$MVPE" "$SPS"; do
  [ -n "$v" ] || { echo "FAIL: a subscore is missing from $PKG_LOG -- the build did not complete"; exit 1; }
done
if [ "$ALL81" = "1" ]; then
  echo "  anchors = $LIVE_NAME verify_report.json (interp378) -- IN-SAMPLE for 81-file members, on the card's side and on this rebuild's"
fi
# S1 history note: an earlier version of this block duplicated its own command line into the heredoc,
# so the acceptance test had never actually run.  F5 (2026-08-12): the gate is now the four accuracy
# anchors; FINAL_LOCAL is informational only (local `time` contention noise >> any usable tolerance).
# RPDE_FINAL_MODE=1 (set only by the Final package's final_pipeline.run_t1_base, 2026-09-29): the organizer runs train()
# ONCE with no retry, and these four anchors are in-sample values of a retrained (stochastic) chain -- measured rebuild
# deltas tke +0.201 (T1) and -0.313 (T2) against +-0.3.  In that mode the deltas are printed and recorded in
# $OUT/anchor_check.json but never abort the run; internal repro chains (no RPDE_FINAL_MODE) keep the fatal gate.
RPDE_ANCHOR_JSON="$OUT/anchor_check.json" $PY - "$REL" "$TKE" "$MVPE" "$SPS" "${GOT:-nan}" \
      "$TARGET_REL" "$TOL_REL" "$TARGET_TKE" "$TOL_TKE" \
      "$TARGET_MVPE" "$TOL_MVPE" "$TARGET_SPS" "$TOL_SPS" <<'PYEOF'
import sys, os, json
rel, tke, mvpe, sps, got = (float(x) for x in sys.argv[1:6])
t_rel, tol_rel, t_tke, tol_tke, t_mvpe, tol_mvpe, t_sps, tol_sps = (float(x) for x in sys.argv[6:14])
bad = []
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
    json.dump({"status": "RECORDED_NOT_GATED", "stage": "t1_base 9/9", "misses": bad, "legs": rows,
               "final_local_informational": got}, open(os.environ["RPDE_ANCHOR_JSON"], "w"), indent=1)
    print("  RECORDED (Final mode, not gated): " + ("; ".join(bad) if bad else "all four anchors inside tolerance"))
    sys.exit(0)
if bad:
    print("  FAIL: " + "; ".join(bad)); sys.exit(1)
print("  ACCEPT: the chain reproduces the shipped ACCURACY+BAND result from a clean tree.")
PYEOF
if [ "$ALL81" = "1" ]; then echo "  ALL81=1: the accepted bundle is $PKG_NAME (the $LIVE_NAME generation)"; fi
echo "REPRODUCE_T1_OK"
