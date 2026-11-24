#!/bin/bash
# DexMachina inputs for Tab. 2 clips (CPU + a short Genesis FK check): inertia dump (SPIDER venv, read only) -> adapter.
# usage: prep_clips.sh <gpu> <idx...>   (idx = E123_CLIPS_v1 row) · per clip: data/dexmachina/<key>/{inertia.json,
#   <key>_adapter.json, adapter.log, prep.rc}; generated DexMachina assets under code/dexmachina/dexmachina/assets
# Variant (env DM_VARIANT — variant.py; unset = the original prep): CPU only (gpu '-'). Per clip,
#   data/dexmachina/<variant key>/{<inertia>, <key>_adapter.json, adapter.log, prep.rc}: the dump with the variant's mass
#   (--mass-variant: owner value · inertia x M/m_scene) -> the adapter (assets under the variant's own object root). The
#   variant changes only mass and inertia, so the original clip's Genesis FK check holds for it. Clips in parallel are safe
#   (shared object files are written atomically and must agree; the shared hand config is only compared).
set -u
GPU=${1:?gpu}; shift
B="${DEXENGINE_E123_ROOT:?set DEXENGINE_E123_ROOT to the experiment workspace}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=$(cd "$HERE/../../.." && pwd)
CLIPS=$R/experiments/scoring/clips/E123_CLIPS_v1.tsv
PY=$B/code/dexmachina/env/bin/python; SPY=$B/code/spider/.venv/bin/python
DM=$B/code/dexmachina/dexmachina; SPD=$B/data/spider/processed/taco/sharpa_wave/bimanual
ADAPT=$HERE/taco_to_dexmachina.py
if [ -n "${DM_VARIANT:-}" ]; then
  eval "$(python3 "$HERE/variant.py" --sh "$DM_VARIANT")" && [ "${DMV_NAME:-}" = "$DM_VARIANT" ] \
    || { echo "PREP FAIL variant '$DM_VARIANT' (variant.py)"; exit 2; }
  for idx in "$@"; do
    okey=$(printf "e123c%03d" "$idx"); key=$(python3 "$HERE/variant.py" --key "$DM_VARIANT" "$idx") || exit 2
    clip=$(awk -F'\t' -v i="$idx" '!/^#/ && $1==i {print $2}' "$CLIPS")
    [ -n "$clip" ] || { echo "PREP FAIL $key no clip row"; continue; }
    O=$B/data/dexmachina/$key; mkdir -p "$O"
    for s in right left; do [ -f "$O/actuated_$s.json" ] || cp "$B/data/dexmachina/$okey/actuated_$s.json" "$O/"; done
    if [ -f "$O/prep.rc" ] && grep -q "^OK" "$O/prep.rc"; then echo "PREP SKIP $key (done)"; continue; fi
    { timeout 600 env -u LD_LIBRARY_PATH PYTHONPATH="$R" $SPY -s -B "$ADAPT" --dump-inertia "$SPD/$clip/scene.xml" \
          "$O/$DMV_INERTIA" --mass-variant "$DM_VARIANT" &&
      ( cd $DM && timeout 900 env PYTHONPATH="$R" $PY "$ADAPT" --inertia-json "$O/$DMV_INERTIA" \
          --clip-dir "$SPD/$clip" --key "$key" --dm "$DM" --out "$O" ); } > "$O/adapter.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && grep -q "^\[ADAPTER\] OK" "$O/adapter.log"; then
      echo "OK $(grep '^\[ADAPTER\] OK' "$O/adapter.log" | head -1 | cut -c13-400)" > "$O/prep.rc"; echo "PREP OK $key $clip"
    else
      echo "FAIL rc=$rc" > "$O/prep.rc"; echo "PREP FAIL $key $clip rc=$rc: $(tail -n 2 "$O/adapter.log" | tr '\n' ' ' | cut -c1-200)"
    fi
  done
  exit 0
fi
for idx in "$@"; do
  key=$(printf "e123c%03d" "$idx")
  clip=$(awk -F'\t' -v i="$idx" '!/^#/ && $1==i {print $2}' "$CLIPS")
  [ -n "$clip" ] || { echo "PREP FAIL $key no clip row"; continue; }
  O=$B/data/dexmachina/$key; mkdir -p "$O"
  # the hand's Genesis actuated dof order (hand-level, written once for e123c000; --fk-check re-checks it per clip)
  for s in right left; do [ -f "$O/actuated_$s.json" ] || cp $B/data/dexmachina/e123c000/actuated_$s.json "$O/"; done
  if [ -f "$O/prep.rc" ] && grep -q "^OK" "$O/prep.rc"; then echo "PREP SKIP $key (done)"; continue; fi
  { timeout 600 env -u LD_LIBRARY_PATH PYTHONPATH="$R" $SPY -s -B "$ADAPT" --dump-inertia "$SPD/$clip/scene.xml" "$O/inertia.json" &&
    ( cd $DM && CUDA_VISIBLE_DEVICES=$GPU timeout 900 env PYTHONPATH="$R" $PY "$ADAPT" --inertia-json "$O/inertia.json" \
        --clip-dir "$SPD/$clip" --key "$key" --dm "$DM" --out "$O" --fk-check ); } > "$O/adapter.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && grep -q "^\[ADAPTER\] OK" "$O/adapter.log"; then
    echo "OK $(grep '^\[ADAPTER\] OK' "$O/adapter.log" | head -1 | cut -c13-400)" > "$O/prep.rc"; echo "PREP OK $key $clip"
  else
    echo "FAIL rc=$rc" > "$O/prep.rc"; echo "PREP FAIL $key $clip rc=$rc: $(tail -n 2 "$O/adapter.log" | tr '\n' ' ' | cut -c1-200)"
  fi
done
