#!/bin/bash
# Bake chain of one clip and one placement candidate, run as ONE stage list in a resident worker.
#
# Order: apply the placement (no retiming) on the f0 shadow → wrists → arm IK once (coarse-path DP) [→ cross-repair]
#   → retime (stretch the time axis only) → body positions → wrists. Stretching by k makes the speed exactly 1/k
#   (linear interpolation), and the IK is never solved again, so the IK valley the judge accepted is kept. Solving IK
#   twice could land in two valleys; inserting frames between plan and IK would desynchronise the retiming plan.
#
# f0 first: the f0 shadow was built once per (hand, clip) before judging (dexengine.retarget.f0.prejudge); this chain
#   does not run f0 — it reads that staging (DEXENGINE_F0_STAGE, required), so judge and bake see the same trajectory.
#
# Cross-repair (DEXENGINE_CROSS_REPAIR=1): after IK, dexengine.placement.repair.cross_repair pushes apart the
#   penetrations of the canonical collision set (hand-hand, hand-opposite object, hand-torso) with the smallest
#   displacement, correcting the 7 arm joints locally (damped least squares) and writing them into the IK npz.
#   IK is not solved again (a full DP re-solve leaves the accepted valley). Body/wrist caches are rebuilt after the
#   retiming by POST (retime does not read them).
#
# Only caller: bake_drive.sh, which starts the resident workers (DEXENGINE_BAKE_POOL) and passes the values it
# resolved once (arm, hand, layer roots, suffixes, IK pool, source layer, the three placement-record values).
# Usage: bash chain_fast.sh <combo> <base> <safety> <max_stretch> --arm A --hand H --troot DIR --w2 DIR --ik-sfx S
#          --tree-sfx S --ik-pool DIR --assets DIR --src-layer L --zoff Z --anchor N --idle "<idle sides>" --cand K/N
#   (safety / max_stretch → retime; only --ik-sfx and --idle may be empty; unknown or missing options die)
# Env read (set by the parent):
#   DEXENGINE_F0_STAGE   f0 shadow root (required)
#   DEXENGINE_BAKE_POOL  resident worker pool (required)
#   DEXENGINE_DATASET / DEXENGINE_EMBODIMENT  dataset axis (required; passed to the bake stage)
#   DEXENGINE_REUSE_IK   judge IK cache (reuse the joint path the judge accepted)
#   DEXENGINE_CROSS_REPAIR=1  insert the cross-repair stage
#   DEXENGINE_XR_LEDGER  cross-repair ledger (one row per clip) · DEXENGINE_XR_URDF  URDF of the cross distances
#   LOG  log folder · PYS / PYDP  interpreters (retarget venv / Isaac venv)
# Verdict: last line CLEAN_OK / CLEAN_FAIL <stage>
set -uo pipefail

WT=$(cd "$(dirname "$0")/../.." && pwd)
# The repo is importable as a package (no-op under bake_drive.sh, which already exported it — the resident worker
# refuses a request whose environment differs from its start environment).
case ":${PYTHONPATH:-}:" in *":$WT:"*) ;; *) export PYTHONPATH="$WT${PYTHONPATH:+:$PYTHONPATH}" ;; esac
PYS=${PYS:-$(python3 -m dexengine.core.paths --what retarget-py)} || exit 2   # owner paths.retarget_py (RETARGET_PY first)
. "$WT/scripts/_isaac_py.sh" || { echo "[chain_fast] cannot read scripts/_isaac_py.sh" >&2; exit 2; }
PYDP=${PYDP:-$(isaac_py)} || exit 2   # Isaac python location (path only); PYDP / ISAACLAB_DIR win
[ $# -ge 4 ] || { echo "CLEAN_FAIL args (<combo> <base> <safety> <max_stretch> are required — caller bake_drive.sh)"; exit 2; }
COMBO=$1; BASE=$2; SAFETY=$3; MAX_STRETCH=$4; shift 4
# Named options — the values come from bake_drive.sh (not resolved again here). An unknown option dies.
declare -A _A=()
while [ $# -gt 0 ]; do
  case $1 in
    --arm|--hand|--troot|--w2|--ik-sfx|--tree-sfx|--ik-pool|--assets|--src-layer|--zoff|--anchor|--idle|--cand)
      [ $# -ge 2 ] || { echo "CLEAN_FAIL args ($1 has no value)"; exit 2; }
      _A[$1]=$2; shift 2 ;;
    *) echo "CLEAN_FAIL args (unknown argument '$1' — the only caller is bake_drive.sh)"; exit 2 ;;
  esac
done
for _k in --arm --hand --troot --w2 --ik-sfx --tree-sfx --ik-pool --assets --src-layer --zoff --anchor --idle --cand; do
  [ -n "${_A[$_k]+x}" ] || { echo "CLEAN_FAIL args ($_k missing — bake_drive.sh passes it)"; exit 2; }
  case $_k in --ik-sfx|--idle) ;; *) [ -n "${_A[$_k]}" ] || { echo "CLEAN_FAIL args ($_k is empty)"; exit 2; } ;; esac
done
ARM=${_A[--arm]}; HAND=${_A[--hand]}                 # combo → (arm, hand) = paths.split_combo
TROOT=${_A[--troot]}; WROOT=${_A[--w2]}              # layer roots paths.trees_dir / robots_dir
IK_SFX=${_A[--ik-sfx]}; TREE_SFX=${_A[--tree-sfx]}   # suffixes paths.ik_suffix / paths.TREE_SFX
POOL=${_A[--ik-pool]}                                # IK pool (paths.pool_root) — the same pool as the judge
# Cross-repair mesh root = parent of the assets_objects layer. Base-layer trees carry an absolute scene.xml meshdir,
# which wins; this value is used only for a relative meshdir.
ASSETS_ROOT=${_A[--assets]}; SRC_LAYER=${_A[--src-layer]}
# The three placement-record values (bake_drive writes adopted.json and passes them): robot z offset, anchor (the
# judge file <clip>_a<anchor>_<side>.npz), idle sides (one-hand rule; empty = both hands move).
ZOFF=${_A[--zoff]}; ANCHOR=${_A[--anchor]}; IDLE=${_A[--idle]}
CAND=${_A[--cand]}                                    # which candidate this attempt is, <k>/<n> (post-bake check marker)
TREE="${BASE}${TREE_SFX}"
DS_ARGS=(--dataset "${DEXENGINE_DATASET:?DEXENGINE_DATASET is required (no default)}"
         --emb "${DEXENGINE_EMBODIMENT:?DEXENGINE_EMBODIMENT is required (no default)}")
LOG=${LOG:-/tmp/v5_clean}; mkdir -p "$LOG"
SLUG="${COMBO}__${BASE// /_}"
echo "[chain_fast] dataset=$DEXENGINE_DATASET emb=$DEXENGINE_EMBODIMENT src_layer=$SRC_LAYER trees=$TROOT robots=$WROOT assets=$ASSETS_ROOT"
cd "$WT"
# Half-write check. The chain writes the tree and the right arm before the left: a failure in between leaves tree and
# right arm in the new generation and the left arm in the old one (the per-side gate cannot see it). Every failure
# path (die) therefore measures the three lengths; bake_drive.sh re-checks every clip at the end ($LEDGER.halfwrite).
# The lengths (R, L, qpos; -1 = missing or unreadable) come from bake.chain.halfwrite_lengths (same function).
lengths() {   # → "R L Q" on one line
  $PYS - "$WROOT/arm_ik_poses_right${IK_SFX}/$TREE.npz" \
         "$WROOT/arm_ik_poses_left${IK_SFX}/$TREE.npz" \
         "$TROOT/$TREE/0/trajectory_kinematic.npz" <<'PYHW' 2>/dev/null
import sys
from dexengine.placement.bake import chain as SR
print(*SR.halfwrite_lengths(*sys.argv[1:4]))
PYHW
}
halfwrite_check() {
  local r l q
  read -r r l q < <(lengths)
  # Lengths not measured (python failure, empty output): no verdict here — the end-of-bake check sees every clip.
  [ -z "$q" ] || { [ "$r" = "$l" ] && [ "$l" = "$q" ]; } || echo "ikR=$r,ikL=$l,tree=$q"
}
die() {
  local why=$1 hw
  hw=$(halfwrite_check)
  if [ -n "$hw" ]; then
    echo "CLEAN_FAIL $why ($SLUG) ★half-write — sides/tree generation mismatch: $hw"
    echo "  This clip's outputs cannot be used for training (the two arms point at different times)."
    echo "  Fix: re-bake both sides together with the same warp (DEXENGINE_REUSE_IK reuses the judge joints)."
    echo "CLEAN_FAIL $why ($SLUG — log $LOG)"; exit 2
  fi
  echo "CLEAN_FAIL $why ($SLUG — log $LOG)"; exit 1
}

# Reuse the joint path the judge accepted for this anchor when it is cached (the same IK is not solved twice).
# One-hand rule: the idle side (--idle) gets the resting pose from the DP stage, which changes that hand block of the
# tree, so body/wrist of that side are recomputed after ik and cross-repair sees one generation. Empty = no recompute.
REUSE_DIR=${DEXENGINE_REUSE_IK:-}

# One process: bake (f0 staging) → body → wrist (both) → IK (both) [→ cross-repair] → retime → body → wrist
RE_R=(); RE_L=()
if [ -n "$REUSE_DIR" ] && [ -n "$ANCHOR" ]; then
  [ -f "$REUSE_DIR/${BASE}_a${ANCHOR}_right.npz" ] && RE_R=(",--reuse-npz,$REUSE_DIR/${BASE}_a${ANCHOR}_right.npz")
  [ -f "$REUSE_DIR/${BASE}_a${ANCHOR}_left.npz" ]  && RE_L=(",--reuse-npz,$REUSE_DIR/${BASE}_a${ANCHOR}_left.npz")
fi
# The bake uses the judge's IK pool (--ik-pool); a different pool would seed IK from another robot base (the DP stage
# checks the pool base and dies on a mismatch).
F0STAGE=${DEXENGINE_F0_STAGE:-}
[ -n "$F0STAGE" ] || { echo "CLEAN_FAIL no_f0_root ($SLUG) — DEXENGINE_F0_STAGE is required (order f0 → judge → bake)"; exit 1; }
# No --out_root for the bake stage: the tree root owner is paths.trees_dir, the same function as TROOT above.
BAKE_ARGS="--combo,$COMBO,--base,$BASE,--processed_root,$F0STAGE,--dataset,$DEXENGINE_DATASET,--emb,$DEXENGINE_EMBODIMENT,--src-layer,$SRC_LAYER"
# ik (DP stage) gets --tree: desk z offset of the IK targets = coords.scene_z_offset(tree).
PRE=("bake:$BAKE_ARGS"
     "body:$TROOT/$TREE,-q"
     "wrist:--data_dir,$TROOT,--output_dir,$WROOT/wrist_poses_left,--side,left,--sequence,$TREE"
     "wrist:--data_dir,$TROOT,--output_dir,$WROOT/wrist_poses_right,--side,right,--sequence,$TREE"
     "ik:--sequence,$TREE,--tree,$TROOT/$TREE,--side,right,--arm,$ARM,--hand,$HAND,--base_z_off,$ZOFF,--pool-root,$POOL,--wrist_dir,$WROOT/wrist_poses_right,--output_dir,$WROOT/arm_ik_poses_right${IK_SFX}${RE_R[0]:-}"
     "ik:--sequence,$TREE,--tree,$TROOT/$TREE,--side,left,--arm,$ARM,--hand,$HAND,--base_z_off,$ZOFF,--pool-root,$POOL,--wrist_dir,$WROOT/wrist_poses_left,--output_dir,$WROOT/arm_ik_poses_left${IK_SFX}${RE_L[0]:-}")
if [ -n "$IDLE" ]; then
  PRE+=("body:$TROOT/$TREE,-q")
  for _sd in $IDLE; do
    PRE+=("wrist:--data_dir,$TROOT,--output_dir,$WROOT/wrist_poses_$_sd,--side,$_sd,--sequence,$TREE")
  done
  echo "[IDLE_SIDE] chain_fast $SLUG: idle side [$IDLE] — body/wrist($IDLE) recomputed after ik"
fi
POST=("retime:--tree,$TROOT/$TREE,--arm,$ARM,--hand,$HAND,--ik-right,$WROOT/arm_ik_poses_right${IK_SFX}/$TREE.npz,--ik-left,$WROOT/arm_ik_poses_left${IK_SFX}/$TREE.npz,--safety,$SAFETY,--max-stretch,$MAX_STRETCH"
      "body:$TROOT/$TREE,-q"
      "wrist:--data_dir,$TROOT,--output_dir,$WROOT/wrist_poses_left,--side,left,--sequence,$TREE"
      "wrist:--data_dir,$TROOT,--output_dir,$WROOT/wrist_poses_right,--side,right,--sequence,$TREE")
# The stage list runs in a resident worker started by bake_drive.sh (DEXENGINE_BAKE_POOL). Without the pool this chain
# was called outside bake_drive — die instead of starting a fresh interpreter.
[ -n "${DEXENGINE_BAKE_POOL:-}" ] && [ -d "$DEXENGINE_BAKE_POOL" ] || { echo "CLEAN_FAIL no_bake_pool ($SLUG) — DEXENGINE_BAKE_POOL is missing (bake_drive.sh starts it)"; exit 2; }
if [ "${DEXENGINE_CROSS_REPAIR:-0}" = "1" ]; then
  # Cross-repair rc 3 = "penetration left" (the tree is already improved and updated): recorded, the bake goes on
  # (the runner's XR_PASS_RC). Any other rc fails the chain (1 = unhandled exception, 2 = input error). The runner
  # also stops when the report (0/xr_report.json) is missing. Hard constraints are checked by the final guard below.
  XR_ARGS="--tree,$TROOT/$TREE,--arm,$ARM,--hand,$HAND,--ik-right,$WROOT/arm_ik_poses_right${IK_SFX}/$TREE.npz,--ik-left,$WROOT/arm_ik_poses_left${IK_SFX}/$TREE.npz,--assets-root,$ASSETS_ROOT${DEXENGINE_XR_URDF:+,--urdf,$DEXENGINE_XR_URDF}${DEXENGINE_XR_LEDGER:+,--ledger,$DEXENGINE_XR_LEDGER}"
  ALL=("${PRE[@]}" "xr:$XR_ARGS" "${POST[@]}")
else
  ALL=("${PRE[@]}" "${POST[@]}")
fi
echo "[CHAIN_ONEPROC] $SLUG: ${#ALL[@]} stages in one interpreter (cross_repair=${DEXENGINE_CROSS_REPAIR:-0})" > "$LOG/all_${SLUG}.log"   # a fresh log per attempt
echo "[BAKE_RESIDENT] $SLUG → pool $DEXENGINE_BAKE_POOL" >> "$LOG/all_${SLUG}.log"
# No free worker, or a worker environment different from this request → the client dies (rc 95-97 → die below).
$PYDP -m dexengine.placement.bake.server send --pool "$DEXENGINE_BAKE_POOL" --log "$LOG/all_${SLUG}.log" -- "${ALL[@]}"
_orc=$?
if [ "$_orc" -ne 0 ]; then
  _stop=$(grep -a '^\[runner\] stopped — ' "$LOG/all_${SLUG}.log" | tail -1)
  case "$_stop" in
    *"xr report missing"*) die "cross_repair report missing ($TROOT/$TREE/0/xr_report.json)" ;;
    *"xr failed rc="*)   die "cross_repair (rc=${_stop##*rc=} — 1=exception · 2=input error)" ;;
    *)                  die chain ;;
  esac
fi
# Contact labels are carried by the bake stage (missing ones are caught by the gate).

# Length gate (silent corruption; the only retry gate) — same lengths() as die: a missing/unreadable file (-1) or
# three different lengths fail.
read -r _NR _NL _NQ < <(lengths)
if [ -z "$_NQ" ] || [ "$_NR" -lt 0 ] || [ "$_NL" -lt 0 ] || [ "$_NQ" -lt 0 ] || [ "$_NR" != "$_NL" ] || [ "$_NL" != "$_NQ" ]; then
  echo "length mismatch R=${_NR:-?} L=${_NL:-?} qpos=${_NQ:-?} (-1 = missing or unreadable)" >&2
  die length
fi
echo "OK $_NQ"

# Final hard-constraint gate on the exact bytes the gate reads. The DP stage's guard sees the joints before retiming,
# so a violation the retiming creates on a chord between frames is caught only here.
$PYS -m dexengine.placement.final_guard --arm "$ARM" --hand "$HAND" --sequence "$TREE" --tree "$TROOT/$TREE" \
  --ik-right "$WROOT/arm_ik_poses_right${IK_SFX}/$TREE.npz" \
  --ik-left  "$WROOT/arm_ik_poses_left${IK_SFX}/$TREE.npz" \
  --base-z-off "$ZOFF" >> "$LOG/all_${SLUG}.log" 2>&1 || die final_guard

# Post-bake placement check (scene.place_rule.baked_reject) on the baked tree. The judge sees the trajectory before the
# bake; cross-repair can move a touched object and the retiming stretches the clock, so a placement the judge accepted
# can fail here — then this candidate is dropped (rc 3 → chain failure → bake_drive tries the next one). A check that
# dies by itself (other rc, or no marker) is a chain failure, not a rejection. [BAKE_PLACE_CHECK] prints on pass too.
_pl=$($PYS -m dexengine.placement.bake.place_check --bake-check --tree "$TROOT/$TREE" --arm "$ARM" --clip "$BASE" --cand "$CAND" 2>> "$LOG/all_${SLUG}.log"); _prc=$?
[ -z "$_pl" ] || { echo "$_pl"; echo "$_pl" >> "$LOG/all_${SLUG}.log"; }
case "$_prc" in
  0) [ -n "$_pl" ] || die "placement_check (no marker)" ;;
  3) die placement ;;
  *) die "placement_check (rc=$_prc)" ;;
esac
echo "CLEAN_OK $SLUG"
