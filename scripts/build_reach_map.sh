#!/bin/bash
# Build the reach map (arm IK pool) of an arm+hand combo — the IK solutions the judge and the bake use as seeds.
# The judge and the per-hand pipeline call this one entry, so a new hand gets its map before it reaches the judge.
#
# Usage: bash scripts/build_reach_map.sh <arm> <hand> [GPU=0] [WORKERS=6]
#        bash scripts/build_reach_map.sh <arm> <hand> --verify-only   (no build: measure the completion condition below ·
#        no writes · marker [IKPOOL_GATE] OK|✗ · rc 0 = both sides loadable by the judge)
# Out:   <pool_root>/<arm>_<hand>_{right,left}/  (pool_root owner = dexengine.core.paths --what pool_root)
# Done:  per side, the map identity (arm/hand/side) equals the request and the clearance sidecar is usable — every
#        layer present and its header (ruler, arm shape) current (reach_map.clearance.check_sidecar, the function the
#        judge calls when it loads the map; marker [CLRCAP_ID]). File counts alone are not checked.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO=$(pwd)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"   # the repo is importable as a package

ARM=${1:?arm (e.g. rby1)}
HAND=${2:?hand id (e.g. dg5f_s)}
VERIFY_ONLY=0; [ "${3:-}" = --verify-only ] && { VERIFY_ONLY=1; set -- "$1" "$2"; }
GPU=${3:-0}
NW=${4:-6}

# Isaac python = $ISAAC_PY of scripts/_isaac_env.sh (it also activates that venv, so `python3` below is that venv's).
# The training recipe is not sourced: it requires a dataset axis that the reach map (arm + hand only) does not have.
source scripts/_isaac_env.sh || exit 2
PYI=${PYI:-$ISAAC_PY}
POOL=$(python3 -m dexengine.core.paths --what pool_root) || exit 2
# A new pool root (e.g. DEXENGINE_POOL_ROOT=/tmp/pool_<TAG>) needs the folder for the logs below.
mkdir -p "$POOL" || { echo "[ikpool] ✗ cannot create the pool root: $POOL" >&2; exit 2; }
COMBO=$(python3 -c "
from dexengine.core import paths as P; print(P.combo_id('$ARM', '$HAND'))") || exit 2

# Grid, atom count, seeds and valley count are canonical here (they differ from the build.py argparse defaults
# 0.03 / 128); the existing maps' meta.npz record them (pos_step 0.035 · atoms 96).
POS_STEP=0.035
N_ATOMS=96
SEEDS=16
N_VALLEYS=4

# Done = every layer + current header (check_sidecar). Counting files would pass a map missing one shard or with an
# old header. No [CLRCAP_ID] marker = failure. The end of a build and --verify-only call the same function.
verify_sides() {
  local rc=0 SIDE OUT ID
  for SIDE in right left; do
    OUT="$POOL/${COMBO}_$SIDE"
    [ -f "$OUT/meta.npz" ] || { echo "[ikpool] ✗ $SIDE incomplete: no meta ($OUT)" >&2; rc=2; continue; }
    "$PYI" -m dexengine.placement.reach_map.store --pool-dir "$OUT" --arm "$ARM" --hand "$HAND" --side "$SIDE" \
      || { echo "[ikpool] ✗ $SIDE: the map identity differs from the request ($OUT)" >&2; rc=2; continue; }
    ID=$("$PYI" -c "import sys; from pathlib import Path; from dexengine.placement.reach_map import clearance as K; K.check_sidecar(Path(sys.argv[1]), sys.argv[2])" "$OUT" "$ARM" 2>&1)
    printf '%s\n' "$ID" | grep -q "^\[CLRCAP_ID\] ${COMBO}_$SIDE " \
      || { echo "[ikpool] ✗ $SIDE sidecar — $(printf '%s\n' "$ID" | tail -n 1 | cut -c1-200)" >&2; rc=2; continue; }
    echo "[ikpool] OK $SIDE: $(printf '%s\n' "$ID" | grep '^\[CLRCAP_ID\]') · $(du -sh "$OUT" | cut -f1)"
  done
  return $rc
}
if [ "$VERIFY_ONLY" = 1 ]; then
  verify_sides; vrc=$?
  if [ "$vrc" = 0 ]; then echo "[IKPOOL_GATE] OK combo=$COMBO pool=$POOL"
  else echo "[IKPOOL_GATE] ✗ combo=$COMBO — build: bash scripts/build_reach_map.sh $ARM $HAND" >&2; fi
  exit $vrc
fi
echo "[ikpool] combo=$COMBO pool=$POOL gpu=$GPU workers=$NW · pos-step=$POS_STEP atoms=$N_ATOMS seeds=$SEEDS valleys=$N_VALLEYS"

# Worker rc per pid — a bare `wait` drops it, and a map missing a dead worker's share (OOM, kill) would reach the
# completion check. Any rc != 0 dies here (rc 2, with the log path). Marker [IKPOOL_WAIT] = workers collected.
wait_workers() {  # <what> <pid…>
  local what=$1 p bad=0 n; shift; n=$#
  for p in "$@"; do wait "$p" || bad=$((bad + 1)); done
  [ "$bad" = 0 ] || { echo "[ikpool] ✗ $what: $bad/$n workers rc != 0 — the map is not complete" >&2; exit 2; }
  echo "[IKPOOL_WAIT] $what workers $n/$n rc 0"
}

BUILT=0
for SIDE in right left; do
  OUT="$POOL/${COMBO}_$SIDE"
  if [ -f "$OUT/meta.npz" ]; then
    # An existing map is skipped only when its identity matches (a map built with an old palm frame must not stay).
    "$PYI" -m dexengine.placement.reach_map.store --pool-dir "$OUT" --arm "$ARM" --hand "$HAND" --side "$SIDE" \
      || { echo "[ikpool] ✗ $SIDE: the existing map's identity differs from the current spec — move it aside (do not delete) and build again: $OUT" >&2; exit 3; }
    echo "[ikpool] $SIDE: exists, identity matches ($OUT) — skipped"
    continue
  fi
  mkdir -p "$OUT"
  echo "[ikpool] $SIDE: building with $NW workers → $OUT"
  BUILT=1
  pids=()
  for ((w = 0; w < NW; w++)); do
    CUDA_VISIBLE_DEVICES=$GPU "$PYI" -m dexengine.placement.reach_map.build \
      --arm "$ARM" --hand "$HAND" --side "$SIDE" \
      --pos-step "$POS_STEP" --n-atoms "$N_ATOMS" --seeds "$SEEDS" --n-valleys "$N_VALLEYS" \
      --worker "$w" --n-workers "$NW" --out "$OUT" \
      > "$OUT/build_w$w.log" 2>&1 &
    pids+=($!)
  done
  wait_workers "map build $SIDE($OUT/build_w*.log)" "${pids[@]}"
done

# Desk-clearance sidecar (clrcap_z*.npz) — the judge reads this output. Built only when a side was built here: this
# entry builds new maps; re-baking the sidecar of an existing map is a separate step (the judge may be reading it).
if [ "$BUILT" = 0 ]; then
  echo "[ikpool] no side built here — clearance rebake skipped (other combos are not overwritten)"
  exit 0
fi
echo "[ikpool] clearance rebake ($NW workers)"
pids=()
for ((w = 0; w < NW; w++)); do
  # --arm/--hand are required (no default) — this combo is the target.
  "$PYI" -m dexengine.placement.reach_map.clearance --pool-root "$POOL" \
    --arm "$ARM" --hand "$HAND" \
    --worker "$w/$NW" > "$POOL/clr_${COMBO}_w$w.log" 2>&1 &
  pids+=($!)
done
wait_workers "clearance rebake($POOL/clr_${COMBO}_w*.log)" "${pids[@]}"

verify_sides
exit $?
