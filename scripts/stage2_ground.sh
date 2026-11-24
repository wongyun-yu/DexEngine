#!/bin/bash
# Stage 2 — floating hand → robot placement (IK grounding): f0 → judge → bake (placement + arm IK) → retime → gate
#   → status ledger. Judge and bake read the same trajectory (the f0 shadow) and the same constraint functions
#   (dexengine/placement/collision/core.py). Judge search conditions: configs/placement/judge.yaml.
# In : source layer (paths.default_src_layer) + f0 shadow / wrist cache + reach map (paths.pool_root) + clip list.
#      Every path is resolved by `python3 -m dexengine.core.paths --what …`.
# Out: <precompute layer>/<combo><variant>/{placement,arm_ik_poses_*_cont25,wrist_poses_*} · <trees layer>/<clip>_pl_rt/
#      · gate table (paths.gate_table on <gate_meta layer><variant>) · <OUT>/status_ledger.tsv (one terminal state
#      per clip, sum == input) · <OUT>/summary.txt.
# Done: count match — judged n / bake OK n / gate PASS n on the summary. A per-clip f0 data defect (DATA_DEFECT) is
#      only recorded ($OUT/f0_failed.txt); an f0 run failure fails the chunk (rc 3).
#
# Usage: bash scripts/stage2_ground.sh --arm A --hand H --dataset D --emb E --clips <list.txt> [options]
#   --arm/--hand/--dataset/--emb  required (scripts/_lib.sh common options, no default); also --gpu N --data-root DIR
#                       (the data root; --main-root DIR is the same option)
#   --clips FILE        base clip names, one per line
#   --tag TAG           chunk tag (default VAL_<time>); must read G1<letters>_<ledger axis>_<digits> to be folded
#   --variant SFX       combo suffix (e.g. _val50): writes to the <combo><variant> roots, the canonical root untouched
#   --out DIR           chunk folder (default paths.run_path run_chunk <tag>); must be the tag folder or one below it
#   --f0-root DIR       f0 shadow root (default the f0 axis node <f0_stage><variant>; name the shared node explicitly to
#                       reuse the canonical cache from a variant run)
#   --sandbox DIR       isolated run: only the layers the chain writes (layout.SANDBOX_LAYERS) resolve under DIR; the
#                       f0 cache of the listed clips is copied in first. Marker [SANDBOX]. A test tag (regress · canary)
#                       must run sandboxed (dexengine.placement.sandbox)
#   --judge-extra "FLAGS"      flags appended to the judge verbatim (default none · run.env judge_extra=)
#   --retry-budget REC --exhaust-resume TABLE  a D37 retry chunk — the D37 round (always on, ground_d37_round) passes them
#   --place-reselect FILE      re-selection record (place_check.RESELECT_FILE form): only its clips are judged
#                              with the robot spot disc + their margin; the post-bake check and the audit keep margin 0.
#                              run.env place_reselect= (relative to OUT inside it); a --skip-judge resume passes the
#                              same file. Campaign lanes derive it per chunk (place_check --chunk-record). A clip still
#                              owed it is not judged or baked without it (place_check --carry-guard)
#   --retime-safety F          retiming safety factor (retime --safety, default 1.15; the judge's dp_safety stays)
#   --retime-max-stretch F     retiming total stretch cap (retime --max-stretch, default 3.0)
#   --cross-repair 0|1         cross-repair after IK (default canon_env.sh CANON_CROSS_REPAIR = 1); a residual (rc 3)
#                              is recorded in the status ledger, not filtered
#   --active-sides both|auto   one-hand rule (default auto): auto judges only the active arm of a one-side clip
#                              (task_info.case.active) and bakes the idle arm at rest; exported to the bake
#   --scene table|aerial       scene mode (default datasets.f0_axis); exported as DEXENGINE_SCENE (coords.scene_mode)
#   --bake-npar N              bake lanes (default 16) · --bake-cpu  bake without a GPU
#   --allow-noncandidate       judge clips outside the candidate set of the ledger axis
#   --allow-tag-mismatch       run a tag whose ledger axis differs (its results are not folded)
#   --skip-f0                  reuse the f0 shadow (f0_root.txt of the chunk)
#   --skip-judge               resume an existing --out: reuse f0 + judge outputs, run bake → gate
#   --resume-no-run-env        allow --skip-judge without the chunk's run.env (judge identity rebuilt by the resume)
#   --f0-only / --judge-only   stop after f0 + preflight / after the judge
#   --dry-run                  resolve paths and layers, print them, stop (no writes)
#   --fold                     then fold this hand's chunk ledgers of the dataset into its final ledgers (status_fold,
#                              the RL bundle's input · [FOLD] · rc 8 = fold failed, results stand) · ground_fold_check
set -uo pipefail
_S2_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case ":${PYTHONPATH:-}:" in *":$_S2_REPO:"*) ;; *) export PYTHONPATH="$_S2_REPO${PYTHONPATH:+:$PYTHONPATH}" ;; esac
# Clip names are joined only after normalisation: `canon "$DATASET"` (scripts/_clip_canon.sh).
. "$_S2_REPO/scripts/_clip_canon.sh" || { echo "[stage2] cannot read scripts/_clip_canon.sh" >&2; exit 2; }
. "$_S2_REPO/scripts/stage2/_ground_lib.sh" || { echo "[stage2] cannot read scripts/stage2/_ground_lib.sh" >&2; exit 2; }
paths() { python3 -m dexengine.core.paths "$@"; }
layer() { paths --what layer --dataset "$DATASET" --hand "$HAND" --emb "$EMB" --layer "$@"; }   # layer <layer> [args…]
source "$_S2_REPO/scripts/_lib.sh"
_S2_ARGV=("$@"); pipeline_init "$@" || exit 2
set -- "${PIPELINE_ARGS[@]}"
# These inputs are options now; an inherited setting would otherwise be ignored without a word, so it stops the run.
for _v in "${STAGE2_REFUSED_ENV[@]}"; do
  [ -n "${!_v+x}" ] && { echo "[stage2] ✗ environment $_v is no longer read — pass the option instead (--help)" >&2; exit 2; }
done
CLIPS=""; TAG="VAL_$(date +%m%d_%H%M)"; VARIANT=""; OUT=""; F0ROOT=""; DRY_RUN=0; SANDBOX=""; FOLD=0
SKIP_F0=0; SKIP_JUDGE=0; JUDGE_ONLY=0; F0_ONLY=0; RESUME_NO_RUN_ENV=0; ALLOW_NONCAND=0; ALLOW_TAG=0
BAKE_NPAR=16; BAKE_CPU=0; XREPAIR=""; RETIME_SAFETY=""; RETIME_MAX_STRETCH=""; JUDGE_EXTRA=""; ACTIVE_SIDES=""; SCENE=""; PLACE_RESELECT=""; RETRY_REC=""; EXH_RESUME=""
while [ $# -gt 0 ]; do
  case "$1" in
    --clips) CLIPS=$2; shift ;;
    --tag) TAG=$2; shift ;;
    --variant) VARIANT=$2; shift ;;
    --dry-run) DRY_RUN=1 ;;
    --fold) FOLD=1 ;;
    --out) OUT=$2; shift ;;
    --f0-root) F0ROOT=$2; shift ;;
    --sandbox) SANDBOX=$2; shift ;;
    --retime-safety) RETIME_SAFETY=$2; shift ;;
    --retime-max-stretch) RETIME_MAX_STRETCH=$2; shift ;;
    --judge-extra) JUDGE_EXTRA=$2; shift ;;
    --retry-budget) RETRY_REC=$2; shift ;;
    --exhaust-resume) EXH_RESUME=$2; shift ;;
    --place-reselect) PLACE_RESELECT=$2; shift ;;
    --cross-repair) XREPAIR=$2; shift ;;
    --active-sides) ACTIVE_SIDES=$2; shift ;;
    --scene) SCENE=$2; shift ;;
    --bake-npar) BAKE_NPAR=$2; shift ;;
    --bake-cpu) BAKE_CPU=1 ;;
    --allow-noncandidate) ALLOW_NONCAND=1 ;;
    --allow-tag-mismatch) ALLOW_TAG=1 ;;
    --skip-f0) SKIP_F0=1 ;;
    --skip-judge) SKIP_JUDGE=1; SKIP_F0=1 ;;
    --resume-no-run-env) RESUME_NO_RUN_ENV=1 ;;
    --judge-only) JUDGE_ONLY=1 ;;   # stop after judging (GPU only); bake later with --skip-judge --out <OUT>
    --f0-only) F0_ONLY=1 ;;         # stop after f0 + preflight; judge later with --skip-f0
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) echo "[stage2] unknown option: $1" >&2; exit 2 ;;
  esac; shift
done
# Retiming defaults are applied after option parsing (the environment is not read for them).
RETIME_SAFETY=${RETIME_SAFETY:-1.15}; RETIME_MAX_STRETCH=${RETIME_MAX_STRETCH:-3.0}
# Marker on every path (also the early --dry-run exit): the lane count actually applied.
echo "[NPAR] bake=$BAKE_NPAR (stage2_ground.sh --bake-npar · f0 runs on the GPU engine — [F0_ENGINE] in f0.log names the device)"
[ -n "$CLIPS" ] || [ "$DRY_RUN" = 1 ] || { echo "[stage2] --clips <list> is required" >&2; exit 2; }
case "$ACTIVE_SIDES" in ""|both|auto) ;; *) echo "[stage2] --active-sides takes both|auto: $ACTIVE_SIDES" >&2; exit 2 ;; esac
AS_FLAG=(); [ -n "$ACTIVE_SIDES" ] && AS_FLAG=(--active-sides "$ACTIVE_SIDES")
# The scene mode owner is coords.scene_mode (env DEXENGINE_SCENE): exported here so judge, bake and gate read one value.
case "$SCENE" in ""|table|aerial) ;; *) echo "[stage2] --scene takes table|aerial: $SCENE" >&2; exit 2 ;; esac
[ -n "$SCENE" ] && export DEXENGINE_SCENE=$SCENE
# Isolated run: the root goes to the children as DEXENGINE_SANDBOX_ROOT (the layer owner reads it); test tags must use it.
SANDBOX=$(python3 -m dexengine.placement.sandbox init "$SANDBOX" "$TAG" "$DRY_RUN") || exit 2
[ -n "$SANDBOX" ] && export DEXENGINE_SANDBOX_ROOT=$SANDBOX
ground_fold_check || exit 2
pipeline_registry_check || exit 2
[ "$DRY_RUN" = 1 ] || pipeline_require "clip list" "$CLIPS" || exit 2
pipeline_require "Isaac venv python (PYI)" "$PYI" || exit 2
# The retarget python is used by f0 and the bake only; `--judge-only --skip-f0` judges only and does not need it.
if [ "$JUDGE_ONLY" = 1 ] && [ "$SKIP_F0" = 1 ]; then
  [ -x "${RETARGET_PY:-}" ] || echo "[stage2] --judge-only --skip-f0: judging without the retarget venv (f0 and bake run elsewhere)"
else
  pipeline_require "retarget python (RETARGET_PY)" "$RETARGET_PY" || exit 2
fi
cd "$REPO"
source configs/placement/canon_env.sh
XREPAIR=${XREPAIR:-$CANON_CROSS_REPAIR}
COMBO_V="${COMBO}${VARIANT}"
SRC_LAYER=$(paths --what src-layer --dataset "$DATASET") || exit 2   # owner paths.default_src_layer
# Ledger axis of this run (datasets.ledger_key) — the chunk tag must carry it to be folded.
_LEDGER_KEY=$(python3 -c "from dexengine.data import rules as R; print(R.ledger_key('$DATASET', '$EMB', '$VARIANT'))") || exit 2
# Scene and f0 root defaults = the f0 axis owner datasets.f0_axis(dataset, variant) (the variant `_aerial` is the
# deskless axis). The caller's --scene / --f0-root win.
_F0AX=$(python3 -m dexengine.data.rules --f0-axis "$DATASET" --variant "$VARIANT") || exit 2
IFS=$'\t' read -r _AX_SCENE _AX_NODE _AX_TAIL <<< "$_F0AX"
[ -n "$_AX_SCENE" ] && [ -n "$_AX_NODE" ] && [ -n "$_AX_TAIL" ] || { echo "[stage2] cannot resolve the f0 axis: '$_F0AX'" >&2; exit 2; }
if [ -z "$SCENE" ]; then
  SCENE=$_AX_SCENE; export DEXENGINE_SCENE=$SCENE
  echo "[SCENE_CANON] $DATASET variant=${VARIANT:-(canonical)} → scene=$SCENE (datasets.f0_axis · not given by the caller)"
fi
# dz layers after the scene is fixed (an aerial run without --scene must not get the table layers — the judge
# compares the layers with the policy and would die).
DZ_CELLS=$(python3 -m dexengine.placement.search_policy --cells "$DATASET" "${SCENE:-table}") || { echo "[stage2] search_policy failed: dataset=$DATASET scene=${SCENE:-table}" >&2; exit 2; }
echo "[stage2] dz_cells=$DZ_CELLS (search_policy, dataset=$DATASET scene=${SCENE:-table})"
# Active sides default: f0 stages the wrist cache of the manipulating side only, so `both` would require both sides
# and reject everything — the canonical default is auto.
if [ -z "$ACTIVE_SIDES" ]; then
  ACTIVE_SIDES=auto; AS_FLAG=(--active-sides "$ACTIVE_SIDES")
  echo "[ACTIVE_SIDES_CANON] src_layer=$SRC_LAYER → --active-sides $ACTIVE_SIDES (canonical default · not given by the caller)"
fi
# Reference clip root (read by bake, gate T_orig, contact derivation T_base) — paths.ref_base_dir.
BASE_ROOT=$(paths --what ref-base --dataset "$DATASET" --hand "$HAND" --emb "$EMB" --src-layer "$SRC_LAYER") || exit 2
HAND_DIR=$(paths --what hand-spec-dir --hand "$HAND") || exit 2   # f0 shadow subfolder = the hand spec directory
# A one-hand-emb bake reads the hand's one-hand scene template (uni_layout) — die before baking when it is missing.
if [ "$EMB" != bimanual ] && [ "$JUDGE_ONLY" != 1 ] && [ "$DRY_RUN" != 1 ]; then
  python3 -m dexengine.placement.uni_layout --check "$HAND" \
    || { echo "[stage2] ✗ no one-hand scene template — not starting the bake" >&2; exit 2; }
fi
POOL=$(paths --what pool_root) || exit 2   # canonical reach map root (env DEXENGINE_POOL_ROOT wins)
# Default output = this hand's chunk folder (paths.run_path run_chunk), not /tmp (the judge ledger must survive).
[ -n "$OUT" ] || OUT=$(paths --what run-path --layer run_chunk --hand "$HAND" --tag "$TAG") || exit 2
# An input list that is the output file itself would be overwritten below (0 clips, silently) → die.
if [ "$(readlink -f "$CLIPS")" = "$(readlink -f "$OUT")/clips_judge.txt" ]; then
  echo "[stage2] --clips is \$OUT/clips_judge.txt itself — pass the source list (e.g. \$OUT/clips.txt)" >&2; exit 2; fi
# A resume (--skip-f0) uses the f0 root the previous run wrote ($OUT/f0_root.txt); absent that, the default rule.
[ -z "$F0ROOT" ] && [ "$SKIP_F0" = 1 ] && [ -f "$OUT/f0_root.txt" ] && F0ROOT=$(cat "$OUT/f0_root.txt")
# f0 shadow root = f0 axis node (f0_stage + variant)/tail, both from datasets.f0_axis. A relative node hangs off the
# write root (paths.output_root), not the code tree.
F0ROOT_GIVEN=0; [ -n "$F0ROOT" ] && F0ROOT_GIVEN=1
F0ROOT=${F0ROOT:-$_AX_NODE}
DATA_ROOT=$(paths --what main_root) || exit 2
WRITE_ROOT=$(paths --what output_root) || exit 2     # the sandbox root in an isolated run, else the data root
case "$F0ROOT" in /*) ;; *) F0ROOT=$WRITE_ROOT/$F0ROOT ;; esac
F0=$F0ROOT/$_AX_TAIL
F0_CANON=$DATA_ROOT/$_AX_NODE/$_AX_TAIL               # the canonical f0 cache (seed source of an isolated run)
# Roots of the later stages (layer owner), resolved once so f0, bake, gate and summary see the same values.
W2=$(layer precompute --combo "$COMBO_V") || exit 2       # tree · placement · arm IK · wrist output root
TROOT=$(layer trees --combo "$COMBO_V") || exit 2         # training tree root
GROOT=$(layer gate_meta)${VARIANT} || exit 2              # gate meta (<gate_meta><variant> — variant isolation)
VIEWS=$(layer views) || exit 2                            # training list root (train_set.select --partition-prefix)
echo "[stage2] dataset=$DATASET emb=$EMB src_layer=$SRC_LAYER base_ref=$BASE_ROOT f0=$F0 trees=$TROOT robots=$W2 gate_meta=$GROOT"
# Isolated run: every write path must lie under the sandbox root (e.g. an explicit canonical --f0-root dies here, before
# anything is written). One marker line on every path.
python3 -m dexengine.placement.sandbox prove "$DATA_ROOT" "$F0" "$W2" "$TROOT" "$GROOT" \
  "$(layer contact_derive --combo "$COMBO_V")" || exit 2
if [ "$DRY_RUN" = 1 ]; then
  echo "[stage2] --dry-run: layers resolved, stopping (no f0, judge or bake)"
  echo "  wrist_base_right=$(layer wrist_base)/wrist_poses_right"
  echo "  contact_derive=$(layer contact_derive --combo "$COMBO_V")  contact_base=$(layer contact_base)  views=$VIEWS  pool=$POOL"
  exit 0
fi
mkdir -p "$OUT"
# Fold rule: the chunk TAG = G1<letters>_<ledger axis>_<digits>, and OUT is the tag folder or an attempt folder right
# below it (the fold reads <chunk root>/<tag>). Otherwise the run judges but its results are never folded.
_OUTB=$(basename "$OUT"); _OUTP=$(basename "$(dirname "$OUT")")
if [ "$_OUTB" != "$TAG" ] && [ "$_OUTP" != "$TAG" ]; then
  echo "[stage2] output $OUT is neither the tag ($TAG) folder nor an attempt folder below it — the fold cannot find it" >&2
  exit 2
fi
case "$TAG" in
  G1*_"$_LEDGER_KEY"_[0-9]*) echo "[LEDGER_KEY] $TAG ← $_LEDGER_KEY (fold rule matches · output $_OUTP/$_OUTB)" ;;
  *) echo "[stage2] chunk tag differs from the ledger axis: '$TAG' — the rule is G1<letters>_${_LEDGER_KEY}_<digits>." >&2
     echo "         Run as is, it judges but its results are not folded into the ledger. Fix the tag." >&2
     [ "$ALLOW_TAG" = 1 ] || exit 2 ;;
esac
NCLIP=$(grep -c . "$CLIPS")
ground_check_clips
# The judge loads the reach maps of both sides.
pipeline_require "reach map (${COMBO})" "$POOL/${COMBO}_right/meta.npz" "$POOL/${COMBO}_left/meta.npz" || exit 2
if [ "$SKIP_JUDGE" = 1 ]; then
  # Resume: the judge outputs must be in that OUT (no silent bake of 0 clips).
  pipeline_require "judge plans (--skip-judge resume)" "$OUT/plans.jsonl" || exit 2
  pipeline_require "judge pass list (--skip-judge resume)" "$OUT/passed.txt" || exit 2
  # A chunk with 0 passes (a legitimate reject-all) never gets an IK cache folder — only then it is not required.
  [ "$(grep -c . "$OUT/passed.txt" || true)" = 0 ] || pipeline_require "judge IK cache (--skip-judge resume)" "$OUT/ikc" || exit 2
fi
XR_LEDGER=$OUT/cross_repair_ledger.tsv
ground_retry_inputs || exit 2; ground_run_env; ground_retry_run_env
ground_place_reselect || exit 2
echo "[stage2] tag=$TAG combo=$COMBO_V clips=$NCLIP out=$OUT pool=$POOL bake_npar=$BAKE_NPAR gpu=$GPU skip_f0=$SKIP_F0 skip_judge=$SKIP_JUDGE f0_root=$F0ROOT retime_safety=$RETIME_SAFETY retime_max_stretch=$RETIME_MAX_STRETCH judge_extra=[$JUDGE_EXTRA] cross_repair=$XREPAIR active_sides=$ACTIVE_SIDES scene=$SCENE place_reselect=${PLACE_RESELECT:-none}"
S0=$(date +%s)

# ── 1) f0 shadow (one pass before judging, cached per clip) ──────────────────────────────────────────────────────
# A per-clip data defect (DATA_DEFECT) is only recorded and the batch goes on (the final gate counts). REPAIR_FAIL and
# REFUSED_BIG go on to the judge. f0 rc 1 only means "completed ≠ all", so the run stops only on a crash (no
# completion record). A run failure (missing file, import, CUDA, worker death …) is not a clip verdict: the chunk
# dies after f0 ([F0_RUN_FAIL], rc 3).
[ "$SKIP_F0" = 1 ] && [ -f "$OUT/f0_failed.txt" ] || : > "$OUT/f0_failed.txt"   # a resume keeps the previous record
[ "$SKIP_F0" = 1 ] || : > "$OUT/f0_run_failed.txt"      # a resume keeps it too — a leftover run failure kills it below
if [ "$SKIP_F0" != 1 ]; then
  # f0 device = this chain's GPU (--gpu), unless the caller set DEXENGINE_F0_GPUS (read by f0.prejudge itself).
  F0_GPU_ARGS=(); [ -z "${DEXENGINE_F0_GPUS:-}" ] && [ -n "${GPU:-}" ] && F0_GPU_ARGS=(--gpus "$GPU")
  # An isolated run seeds its f0 shadows by copy from the canonical cache, so it reaches the same cache verdicts
  # (FRESH) as a canonical run; an explicit --f0-root is the caller's decision and is not seeded.
  if [ -n "$SANDBOX" ] && [ "$F0ROOT_GIVEN" = 0 ]; then
    "$RETARGET_PY" -m dexengine.retarget.f0.prejudge --hand "$HAND" --clips "$CLIPS" --dataset "$DATASET" --emb "$EMB" \
      --src-layer "$SRC_LAYER" --f0-root "$F0" --seed-from "$F0_CANON" > "$OUT/f0_seed.log" 2>&1 \
      || { echo "[stage2] ✗ sandbox f0 seed failed — $OUT/f0_seed.log" >&2; exit 1; }
    grep -h '^\[F0_SEED\]' "$OUT/f0_seed.log" | tail -n 1
  fi
  # The shadow root is passed as is ($F0 already carries the layer tail).
  "$RETARGET_PY" -m dexengine.retarget.f0.prejudge --hand "$HAND" --clips "$CLIPS" "${F0_GPU_ARGS[@]}" \
    --dataset "$DATASET" --emb "$EMB" --src-layer "$SRC_LAYER" \
    --f0-root "$F0" --python "$RETARGET_PY" > "$OUT/f0.log" 2>&1
  F0RC=$?
  # The failure lists come from the ledger the producer wrote (LEDGER.tsv of this hand), not from a regex over f0.log
  # (a clip name with spaces would be cut). The status meaning is owned by f0.prejudge (is_judgeable ·
  # is_data_defect): f0_failed.txt = declared data defects → SOURCE_DEFECT · f0_run_failed.txt = run failures.
  python3 -m dexengine.retarget.f0.ledger split-failures "$F0" "$CLIPS" "$HAND_DIR" --run-failed "$OUT/f0_run_failed.txt" \
    > "$OUT/f0_failed.txt" || exit 2
  # Completion counts come from the producer's machine-readable record (f0.ledger); no record = "no completion line".
  F0OK=""; F0TOTAL=""
  if F0TSV=$(python3 -m dexengine.retarget.f0.ledger --log "$OUT/f0.log" 2>"$OUT/f0_summary.err"); then
    IFS=$'\t' read -r F0OK F0TOTAL F0BSKIP F0JRC F0JSON <<< "$F0TSV"
    echo "[stage2] f0 record ok=$F0OK total=$F0TOTAL budget_skip=$F0BSKIP json=$F0JSON"
    # Request ↔ record count: a warning, not a stop (total counts names after normalisation and de-duplication).
    _NU=$(sort -u "$CLIPS" | grep -c . || true)
    [ "$F0TOTAL" = "$_NU" ] || echo "[stage2] ★ f0 record total=$F0TOTAL ≠ requested (unique) $_NU — list and record are different sets" >&2
  fi
  # rc 0 without a record = the producer did not prove completion → invalid run.
  if [ "$F0RC" = 0 ] && [ -z "$F0OK" ]; then
    echo "[stage2] f0 rc=0 but no completion record — this run is invalid: $(tail -n 2 "$OUT/f0_summary.err" 2>/dev/null | tr '\n' ' ')" >&2
    exit 1
  fi
  NF0=$(grep -c . "$OUT/f0_failed.txt" || true)
  if [ "$F0RC" != 0 ]; then
    # Only a crash (no completion record) dies here; "completed 0" can come from a batch of REPAIR_FAIL only, and
    # those clips still go to the judge (whether the judge set is empty is decided by clips_judge.txt below).
    if [ -z "$F0OK" ]; then
      echo "[stage2] f0.prejudge crashed (no completion record) — $OUT/f0.log" >&2; exit 1
    fi
    echo "[stage2] warning: f0 data-defect clips $NF0 (recorded only, batch goes on · $OUT/f0_failed.txt): $(paste -sd' ' "$OUT/f0_failed.txt" | cut -c1-300)" >&2
  fi
fi
# One run failure fails the chunk without judging (recording it as SOURCE_DEFECT would make an environment defect a
# terminal state). A resume with a leftover run-failure record dies too — re-run f0. Last line of run.log.
NRF=$(grep -c . "$OUT/f0_run_failed.txt" 2>/dev/null || true)
if [ "${NRF:-0}" -gt 0 ]; then
  echo "[F0_RUN_FAIL] n=$NRF/$NCLIP not a data verdict (environment or code exception) · rc 3 without judging · $OUT/f0_run_failed.txt · e.g. $(awk -F'\t' 'NR==1{n=$3; print $1" "$2" … "substr(n, length(n)>220 ? length(n)-219 : 1)}' "$OUT/f0_run_failed.txt")" >&2
  exit 3
fi
NF0=$(grep -c . "$OUT/f0_failed.txt" || true)
[ "$SKIP_F0" = 1 ] || printf '%s\n' "$F0ROOT" > "$OUT/f0_root.txt"   # the root f0 actually ran in (a resume reads it)
# A run that skipped f0 must have the shadow in that root (else every bake dies silently). Shadow subfolder = the hand
# spec directory. The first clip outside the f0 failure list is checked.
if [ "$SKIP_F0" = 1 ]; then
  _c=$(grep -vxF -f "$OUT/f0_failed.txt" "$CLIPS" | grep -m1 .)
  if [ -n "$_c" ] && [ ! -d "$F0/$HAND_DIR/$EMB/$_c" ]; then
    echo "[stage2] no f0 shadow: $F0/$HAND_DIR/$EMB/$_c — name the previous run's f0 root with --f0-root (this axis: $DATA_ROOT/$_AX_NODE)" >&2; exit 2; fi
fi
# Only clips with an f0 shadow go to the judge.
comm -23 <(canon "$DATASET" < "$CLIPS" | sort -u) <(canon "$DATASET" < "$OUT/f0_failed.txt" | sort -u) > "$OUT/clips_judge.txt"
if [ ! -s "$OUT/clips_judge.txt" ]; then
  # Every clip an f0 data defect = a legitimate end state (SOURCE_DEFECT): write the status ledger only, rc 0.
  # 0 clips to judge with 0 f0 failures is a list problem → die.
  [ "${NF0:-0}" -gt 0 ] || { echo "[stage2] 0 clips to judge and 0 f0 failures — check the list: $CLIPS" >&2; exit 1; }
  echo "[stage2] 0 clips to judge (f0 failures $NF0/$NCLIP) — writing the status ledger only (SOURCE_DEFECT)" >&2
  mkdir -p "$OUT/p0"; : > "$OUT/passed.txt"; : > "$OUT/plans.jsonl"
  python3 -m dexengine.placement.ledger --input "$CLIPS" --judge-root "$OUT" --dataset "$DATASET" --out "$OUT/status_ledger.tsv" 2>&1 | sed 's/^/  [status] /' \
    && [ -s "$OUT/status_ledger.tsv" ] || { echo "[stage2] status ledger invariant violated (all-f0-failed path) — $OUT" >&2; exit 1; }
  NROW=$(python3 -m dexengine.placement.ledger --count "$OUT/status_ledger.tsv") \
    || { echo "[stage2] cannot read the status ledger (all-f0-failed path — see the [status] lines) — $OUT/status_ledger.tsv" >&2; exit 1; }
  echo "[stage2] $TAG combo=$COMBO_V clips=$NCLIP · f0 failed $NF0 · ledger ${NROW} rows" | tee "$OUT/summary.txt"
  ground_fold
  exit 0
fi
# Judgeability per clip, up front: the judge would stamp a clip without its inputs "rejected". It reads the f0 shadow
# staging (<F0>/<hand spec dir>/, --wrist-root auto). Required sides: both = every side of the emb · auto = at least
# one (f0 stages the manipulating side). Missing clips are dropped from the judge list (not written to the ledger).
WRIST_BASE=$F0/$HAND_DIR
case "$EMB" in bimanual) WRIST_SIDES="right left" ;; right) WRIST_SIDES="right" ;; left) WRIST_SIDES="left" ;; *) echo "[stage2] unknown emb=$EMB" >&2; exit 2 ;; esac
WRIST_NEED=all; [ "$ACTIVE_SIDES" = auto ] && WRIST_NEED=any
if [ "$SKIP_JUDGE" != 1 ]; then
  : > "$OUT/preflight_missing.txt"
  while IFS= read -r c; do
    [ -z "$c" ] && continue
    [ -f "$F0/$HAND_DIR/$EMB/$c/F0_PREJUDGE.json" ] || { echo -e "$c\tf0_shadow"; continue; }
    [ -f "$F0/$HAND_DIR/$EMB/$c/task_info.json" ] || { echo -e "$c\tf0_task_info"; continue; }
    _have=""; _miss=""
    for _s in $WRIST_SIDES; do
      if [ -f "$WRIST_BASE/wrist_poses_$_s/$c.npz" ]; then _have="$_have $_s"; else _miss="$_miss$_s,"; fi
    done
    if [ "$WRIST_NEED" = all ]; then
      [ -n "$_miss" ] && echo -e "$c\twrist_cache:${_miss%,}"
    else
      [ -z "$_have" ] && echo -e "$c\twrist_cache:none (both sides)"
    fi
  done < "$OUT/clips_judge.txt" >> "$OUT/preflight_missing.txt"
  NMISS=$(grep -c . "$OUT/preflight_missing.txt" || true)
  if [ "$NMISS" != 0 ]; then
    NIN=$(grep -c . "$OUT/clips_judge.txt")
    echo "[stage2] $NMISS/$NIN clips cannot be judged — input files missing (list $OUT/preflight_missing.txt):" >&2
    head -n 5 "$OUT/preflight_missing.txt" >&2
    cut -f1 "$OUT/preflight_missing.txt" | sort -u > "$OUT/.pf_skip.txt"
    comm -23 <(sort -u "$OUT/clips_judge.txt") "$OUT/.pf_skip.txt" > "$OUT/.cj.txt" && mv "$OUT/.cj.txt" "$OUT/clips_judge.txt"
    rm -f "$OUT/.pf_skip.txt"
    echo "[stage2] those $NMISS clips are left out of judging (no ledger row = previous state kept). Left to judge $(grep -c . "$OUT/clips_judge.txt")/$NIN" >&2
    if [ ! -s "$OUT/clips_judge.txt" ]; then
      echo "[stage2] nothing left to judge — build the f0 shadow / wrist cache first. rc 5" >&2; exit 5
    fi
  fi
  echo "[PREFLIGHT_WRIST] $WRIST_BASE · sides {$WRIST_SIDES} · need=$WRIST_NEED (active sides $ACTIVE_SIDES)"
  echo "[stage2] preflight: all $(grep -c . "$OUT/clips_judge.txt") clips to judge have an f0 shadow and wrist cache (per side)"
fi
if [ "$F0_ONLY" = 1 ]; then
  echo "[stage2] --f0-only: f0 + preflight done — to judge $(grep -c . "$OUT/clips_judge.txt")/$NCLIP · f0 root $F0ROOT · list $OUT/clips_judge.txt (judge with --skip-f0; f0_root.txt written)"
  exit 0
fi
S1=$(date +%s)

# ── 2) Judge (GPU) — reads the f0 shadow (--wrist-root / --traj-root default auto) ──────────────────────────────
if [ "$SKIP_JUDGE" != 1 ]; then
# The judge appends to p0/ (open "a"), so judging over an old p0 would mix two attempts. The caller clears it.
if [ -e "$OUT/p0" ]; then
  echo "[stage2] ✗ $OUT/p0 already exists — the judge folder must be new (clear it first)" >&2; exit 2
fi
# End marks (one line per clip after its judging ENDS) live in the new p0, so no old marks mix in.
JUDGE_MARKS=$OUT/p0/judged.tsv
[ "$JUDGE_ONLY" = 1 ] || ground_carry_guard "$CLIPS" || exit 2   # a judge-only run has no ledger: the bake host checks it
CUDA_VISIBLE_DEVICES=$GPU DEXENGINE_F0_STAGE=$F0 "$PYI" -m dexengine.placement.judge.core \
  --arm "$ARM" --hand "$HAND" --clips "$OUT/clips_judge.txt" \
  --dataset "$DATASET" --emb "$EMB" --src-layer "$SRC_LAYER" \
  "--dz-cells=$DZ_CELLS" --pool-root "$POOL" $JUDGE_EXTRA "${AS_FLAG[@]}" "${PRS[@]}" \
  --end-marks "$JUDGE_MARKS" --save-ik "$OUT/ikc" --out "$OUT/p0" > "$OUT/judge.log" 2>&1
_jrc=$?
# Judging done = judge rc 0 ∧ an end mark for every target clip. Plan/reject rows are no proof: an anchor-failure row is
# written before the exhaustive search, so a clip killed during it (OOM) has a reject row too. Incomplete → write no
# judge output and die (no JUDGE_ONLY, no DONE, no bake). This is the last line of run.log (SIGKILL leaves no traceback).
# End-mark names are normalised before the join; canon runs outside the comparison so its rc is checked.
_jdone=""   # no mark file = not even the first clip ended (death during boot included) → every target is incomplete
if [ -s "$JUDGE_MARKS" ]; then
  _jdone=$(cut -f1 "$JUDGE_MARKS" | canon "$DATASET") \
    || { echo "[JUDGE_INCOMPLETE] rc=$_jrc · end-mark name normalisation (canon $DATASET) failed — no judge output written · $JUDGE_MARKS" >&2; exit 4; }
fi
_jmiss=$(comm -23 <(grep -v '^$' "$OUT/clips_judge.txt" | sort -u) <(printf '%s\n' "$_jdone" | grep -v '^$' | sort -u))
NTARGET=$(grep -c . "$OUT/clips_judge.txt" || true); NJUDGED=$(( NTARGET - $(printf '%s\n' "$_jmiss" | grep -c . || true) ))
# Two empty sets matching is no evidence — the target must be ≥ 1.
if [ "$_jrc" != 0 ] || [ -n "$_jmiss" ] || [ "${NTARGET:-0}" -lt 1 ]; then
  echo "[JUDGE_INCOMPLETE] rc=$_jrc · end marks $NJUDGED/$NTARGET · incomplete e.g. $(printf '%s\n' "$_jmiss" | head -n 3 | tr '\n' ' ')— no judge output written (no JUDGE_ONLY, DONE or bake) · end of $OUT/judge.log: $(grep -av '^\s*$' "$OUT/judge.log" | tail -n 1 | cut -c1-160)" >&2
  exit 4
fi
echo "[JUDGE_DONE] end marks $NJUDGED/$NTARGET · rc 0 · $JUDGE_MARKS"
# judge = bake contract (same criteria, same functions): no retry at the judge end with constraints off.
cat "$OUT"/p0/plans2_*.jsonl > "$OUT/plans.jsonl" 2>/dev/null || true
# A broken line is not skipped silently (every clip after it would vanish from passed) — stop here.
python3 - "$OUT/plans.jsonl" <<'PY' > "$OUT/passed.txt" || { echo "[stage2] plans.jsonl is broken — $OUT/plans.jsonl" >&2; exit 1; }
import json, sys
s = set()
for ln, l in enumerate(open(sys.argv[1]), 1):
    if not l.strip():
        continue
    try:
        s.add(json.loads(l)["clip"])
    except (ValueError, KeyError) as e:
        print(f"[stage2] plans.jsonl line {ln} is broken: {e}", file=sys.stderr)
        sys.exit(3)
print("\n".join(sorted(s)))
PY
else
  echo "[stage2] judge reused: $OUT/plans.jsonl · $OUT/passed.txt · $OUT/ikc (--skip-judge)"
fi
NJ=$(grep -c . "$OUT/passed.txt" || true)
S2=$(date +%s)
if [ "$JUDGE_ONLY" = 1 ]; then
  # The completion count is the end-mark check above ([JUDGE_DONE]) against the judge target (clips_judge.txt).
  echo "[stage2] --judge-only: judging done (passed=$NJ, judged $NJUDGED/$NTARGET · input $NCLIP) — bake and gate continue with --skip-judge --out $OUT"; echo "judge_only" > "$OUT/JUDGE_ONLY"
  # End mark DONE, published by rename: no pid + DONE = normal end, neither = death.
  printf '%s %s/%s\n' "$(date +%s)" "$NJUDGED" "$NTARGET" > "$OUT/.DONE.tmp" && mv "$OUT/.DONE.tmp" "$OUT/DONE" \
    || { echo "[stage2] cannot publish the end mark DONE" >&2; exit 4; }
  exit 0
fi

# ── 3) Bake (placement + f0 reuse + judge IK reuse + retime + final guard), CPU ─────────────────────────────────
ground_carry_guard "$OUT/passed.txt" || exit 2
LEDGER=$OUT/bake_ledger.tsv LOG=$OUT/bake_logs CUDA_VISIBLE_DEVICES=$GPU RETIME_SAFETY=$RETIME_SAFETY RETIME_MAX_STRETCH=$RETIME_MAX_STRETCH \
  DEXENGINE_REUSE_IK=$OUT/ikc DEXENGINE_F0_STAGE=$F0 DEXENGINE_MAIN_ROOT=$DEXENGINE_MAIN_ROOT PYS=$RETARGET_PY \
  DEXENGINE_CROSS_REPAIR=$XREPAIR DEXENGINE_XR_LEDGER=$XR_LEDGER \
  DEXENGINE_DATASET=$DATASET DEXENGINE_EMBODIMENT=$EMB DEXENGINE_SRC_LAYER=$SRC_LAYER \
  DEXENGINE_ACTIVE_SIDES=$ACTIVE_SIDES DEXENGINE_BAKE_CPU=$BAKE_CPU \
  bash scripts/stage2/bake_drive.sh "$COMBO_V" "$OUT/plans.jsonl" "$OUT/passed.txt" "$BAKE_NPAR" "$PLACE_RESELECT" \
  > "$OUT/bake.log" 2>&1; _bd_rc=$?
# The driver's rc is not swallowed. rc 4 = some clips not fully written (their FAIL rows are in the ledger); any other
# non-zero rc is an infrastructure failure → die. Then count match: every judge-passed clip has a row in the ledger.
case "$_bd_rc" in 0|4) ;; *) echo "[stage2] bake driver ended abnormally rc=$_bd_rc — $OUT/bake.log" >&2; exit 1 ;; esac
_bd_miss=$(comm -23 <(grep -v '^$' "$OUT/passed.txt" | sort -u) <(awk -F'\t' 'NF>1{print $1}' "$OUT/bake_ledger.tsv" 2>/dev/null | sort -u) | wc -l)
[ "$_bd_miss" -eq 0 ] || { echo "[stage2] $_bd_miss judge-passed clips have no bake ledger row — the bake did not run · $OUT/bake.log" >&2; exit 1; }
echo "[BAKE_RC] rc=$_bd_rc · ledger missing 0 (judge passed $(grep -c . "$OUT/passed.txt" || true))"
# Bake-OK clips = last ledger row per clip is OK (RETRY → FAIL excluded). Derivation and gate read this list only.
awk -F'\t' 'NF>1{s[$1]=$2} END{for(c in s) if(s[c]=="OK") print c}' "$OUT/bake_ledger.tsv" 2>/dev/null | sort > "$OUT/baked_ok.txt"
NB=$(grep -c . "$OUT/baked_ok.txt" || true)   # grep -c prints 0 and exits 1 on no hit (`|| echo 0` would give two lines)

ground_contact_derive
S3=$(date +%s)

# ── 4) Gate — this run's bake-OK clips read in place, per-side PASS/FAIL ────────────────────────────────────────
# Inputs = the bake-OK list + the output owners (paths.bake_outputs: trees $TROOT · placement/arm IK $W2); the list
# holds this run's clips only. Output = one chunk table (paths.gate_table). Object shape root (gate frame-0
# penetration and the palm oracle below):
ASSETS_ROOT=$(dirname "$(layer assets_objects)") || exit 2
"$PYI" -m dexengine.placement.gate.core --arm "$ARM" --hand "$HAND" --tag "$TAG" --root "$GROOT" --dataset "$DATASET" \
  --clips "$OUT/baked_ok.txt" --tree-root "$TROOT" --robots-dir "$W2" --base-root "$BASE_ROOT" --judge-root "$OUT" \
  --assets-root "$ASSETS_ROOT" --workers "$BAKE_NPAR" > "$OUT/gate.log" 2>&1
_grc=$?
GTSV=$(paths --what gate-table --root "$GROOT" --combo "$COMBO" --tag "$TAG") || exit 2
# The gate is a safety step and must prove it ran. It ends rc 0 whatever the verdicts, so rc != 0 is an accident or a
# systematic non-verdict failure ([GATE_ERROR_FAIL]: status not written). Every bake-OK clip needs both side rows.
if [ "${NB:-0}" -gt 0 ]; then
  [ "$_grc" = 0 ] || { echo "[stage2] ✗ gate rc=$_grc — nothing is adopted without the gate $(grep -ah '^\[GATE_ERROR_FAIL\]' "$OUT/gate.log" | tail -n 1) · $OUT/gate.log" >&2; exit 7; }
  _gmiss=$(comm -23 <(canon "$DATASET" < "$OUT/baked_ok.txt" | sort -u) \
    <(awk -F'\t' 'NR>1{print $1"\t"$3}' "$GTSV" 2>/dev/null | sort -u | cut -f1 | uniq -c | awk '$1>=2{print $2}' | canon "$DATASET" | sort -u) | head -n 3 | tr '\n' ' ')
  [ -z "$_gmiss" ] || { echo "[stage2] ✗ bake-OK clips without both side rows in the gate table (e.g. $_gmiss) — $GTSV" >&2; exit 7; }
  echo "[GATE_COVER] all $NB bake-OK clips have both side rows in the gate"
fi
# ── 4b) Palm frame oracle — the gate measures tracking with the judge's target formula, so a missing translation
#    between the palm reference body and the URDF palm frame (palm_frame_offset_pos) reads as 0 there. The oracle
#    compares rig FK fingertips with merged-URDF FK fingertips on a sample of this run's trees. FAIL or an input
#    defect kills the chunk (a systematic hand shift is a pipeline defect). No [PALM_ORACLE] marker = invalid.
if [ "${NB:-0}" -gt 0 ]; then
  CUDA_VISIBLE_DEVICES= "$PYI" -m dexengine.placement.gate.palm_check --clips "$OUT/baked_ok.txt" --tree-root "$TROOT" \
    --robots-dir "$W2" --dataset "$DATASET" --assets-root "$ASSETS_ROOT" --sample 3 \
    > "$OUT/palm_oracle.log" 2>&1
  _po=$?
  grep -h '\[PALM_ORACLE\]' "$OUT/palm_oracle.log" | tail -n 1
  [ "$_po" = 0 ] || { echo "[stage2] ✗ palm frame oracle rc=$_po — the training robot hand is off the reference or an input is broken: $OUT/palm_oracle.log" >&2; exit 7; }
fi
GP=$(awk -F'\t' 'NR>1 { for(i=1;i<=NF;i++) if($i=="PASS"){c++; break} } END{print c+0}' "$GTSV" 2>/dev/null)
GT=$(awk 'NR>1' "$GTSV" 2>/dev/null | wc -l)
S4=$(date +%s)

# ── 5) Summary (done = count match; summary.txt lines are read by other tools — keep their form) ──────────────
comm -23 <(canon "$DATASET" < "$CLIPS" | sort -u) <(cat "$OUT/passed.txt" "$OUT/f0_failed.txt" | canon "$DATASET" | sort -u) > "$OUT/fails.txt"
{
  echo "[stage2] $TAG combo=$COMBO_V clips=$NCLIP"
  echo "  f0 $((S1-S0))s · judge $NJ/$NCLIP ($((S2-S1))s) · bake OK $NB ($((S3-S2))s) · gate PASS $GP/$GT sides ($((S4-S3))s)"
  echo "  retiming: safety $RETIME_SAFETY · max_stretch $RETIME_MAX_STRETCH (judge --dp-safety is the canonical one)"
  echo "  f0 failed clips $NF0 (recorded only, not judged): $(paste -sd' ' "$OUT/f0_failed.txt" 2>/dev/null | cut -c1-300)"
  echo "  failed (judge, f0 failures excluded): $(paste -sd' ' "$OUT/fails.txt" 2>/dev/null | cut -c1-300)"
  echo "  gate table: $GTSV · ledger: $OUT/bake_ledger.tsv"
} | tee "$OUT/summary.txt"
# ── 6) Status ledger — exactly one terminal state per clip (vocabulary dexengine/placement/ledger_states.py), sum ==
#    input. The only input of the training-set selector. Cross-repair cells: the sets that are on = the training
#    preset COLL_PRESET (set by pipeline_init from scripts/_rl_recipe.sh).
python3 -m dexengine.placement.ledger --input "$CLIPS" --judge-root "$OUT" --bake-ledger "$OUT/bake_ledger.tsv" \
  --dataset "$DATASET" --xr "$([ "$XREPAIR" = 1 ] && echo on || echo off)" --tree-root "$TROOT" \
  --coll-preset "${COLL_PRESET:?the canonical table (pipeline_init) did not set the training preset COLL_PRESET}" \
  --gate "$GTSV" --out "$OUT/status_ledger.tsv" 2>&1 | sed 's/^/  [status] /' || { echo "[stage2] status ledger invariant violated — $OUT/status_ledger.tsv" >&2; exit 1; }
# Cross-repair residual summary and judge budget summary = cells of the ledger just written (appended to summary.txt).
python3 -m dexengine.placement.ledger --xr-summary "$OUT/status_ledger.tsv" | sed 's/^/  /' | tee -a "$OUT/summary.txt" \
  || { echo "[stage2] cannot read the cross-repair cells of the status ledger — $OUT/status_ledger.tsv" >&2; exit 1; }
python3 -m dexengine.placement.judge.budget --summary "$OUT/status_ledger.tsv" | sed 's/^/  /' | tee -a "$OUT/summary.txt" \
  || { echo "[stage2] cannot read the budget cells of the status ledger — $OUT/status_ledger.tsv" >&2; exit 1; }
echo "  status ledger: $OUT/status_ledger.tsv → training set: python3 -m dexengine.placement.train_set.select --status-ledger $OUT/status_ledger.tsv --n 64 --k 4 --partition-prefix $VIEWS/<name> --combo $COMBO_V --dataset $DATASET --emb $EMB → training SEQ_DIR=<tree root> DEXENGINE_SEQ_LIST=$VIEWS/<name>_NN.txt"
# Exit code: reaching here means judge set ≥ 1 and the status ledger invariant (sum == input) hold. 0 passes or 0 bakes
# is a legitimate reject-all → rc 0.
[ -s "$OUT/status_ledger.tsv" ] || exit 1
ground_d37_round || exit $?; ground_fold
