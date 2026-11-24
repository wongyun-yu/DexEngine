#!/bin/bash
# Per-side USD bake of the floating hand (DEXENGINE_ARM=floating) — the same steps and scripts as the merged combo bake,
# with three differences:
#   1) input = the per-side free-root URDFs of make_floating_urdf.py (the hand subtree of the canonical combo URDF)
#   2) convert with --fix-base false (repository bake rule: floating hand = fix_base False)
#   3) output = where the split wiring's path rule (split_artic.split_usd_paths) reads them:
#      <usd_root>/<arm>_<hand>_split/<arm>_<hand>_split_{R,L}.usd (contact_report_bake --combo-dir bakes into one
#      folder; there is no merged <arm>_<hand>.usd — the floating hand is two articulations)
# Needs Isaac Kit (GPU). Done = 2/2 per-side canonical outputs + contact-report bake rc 0 (count match only).
# Virtual-joint body (FH_WRIST=joint, the asset of DEXENGINE_FH_WRIST=joint): the generator's --wrist-joints URDFs
# (stem <arm>_<hand>_vj), baked with --fix-base true (root = the world-fixed anchor, the hand under 6 virtual joints).
# Unset = the default body.
# Parsed lines: "[NA_HAND] <hand>" (the job proofs) and the final "FLOATING_USD_DONE …" line (the E1-E3 job check
# matches its "stage3 2/2 · stage4 rc=0 · canonical usd 2/2" text).
#
# usage: HAND=sharpa_wave bash experiments/ablations/floating_hand/build_floating_usd.sh   [FH_WRIST=joint]
#        env: DEXENGINE_MAIN_ROOT (data root, default the repo) · ISAACLAB_DIR · LOG (default /tmp/floating_usd) · TO (s)
set -u
WT=$(cd "$(dirname "$0")/../../.." && pwd)
MAIN=${DEXENGINE_MAIN_ROOT:-$WT}
HAND=${HAND:?HAND is required (no default)}
echo "[NA_HAND] $HAND"   # hand identity marker — the floating job compares it with the bundle hand
LOG=${LOG:-/tmp/floating_usd}
case "${FH_WRIST:-}" in "") VJ=(); FIXB=false;; joint) VJ=(--wrist-joints); FIXB=true;; *) echo "FLOATING_USD FH_WRIST='${FH_WRIST}' — only joint (default unset)"; exit 2;; esac
cd "$WT" || exit 2
GEN=(env PYTHONPATH="$WT" python3 -m experiments.ablations.floating_hand.make_floating_urdf --hand "$HAND")
STEM=$("${GEN[@]}" --print-stem "${VJ[@]}") || exit 1
[ "${#VJ[@]}" = 0 ] || echo "[FLOATING_USD] virtual-joint body stem $STEM · fix-base $FIXB"
URDF_DIR=$(PYTHONPATH="$WT" python3 -c "from dexengine.core.repo import FULL_ROBOT_URDF_DIR as D; print(D)")/$STEM
D=$MAIN/data/usd_cache/robots/full_robot_gen/${STEM}_split
IL=${ISAACLAB_DIR:-$WT/../IsaacLab}/isaaclab.sh
TO=${TO:-1200}   # timeout of every Isaac step (s)
mkdir -p "$LOG" "$D"
# the generator's checks (structure, identical elements, meshes; for the virtual-joint body: root = anchor and the 6
# added joints) gate the bake
"${GEN[@]}" --check "${VJ[@]}" || { echo "FLOATING_USD source URDF check failed"; exit 1; }
ok=0; combos=""
for S in R L; do
  c=${STEM}_split_$S
  combos="${combos:+$combos,}$c"
  rm -f "$D/${c}_base_test.usd" "$D/${c}_flat_test.usd" "$D/${c}_stage3_test.usd" "$D/$c.usd"
  timeout "$TO" "$IL" -p dexengine/rl/assets/urdf_to_usd.py --fix-base "$FIXB" \
      --input-urdf "$URDF_DIR/$c.urdf" --output-usd "$D/${c}_base_test.usd" --force \
      > "$LOG/${c}_1convert.log" 2>&1
  timeout "$TO" "$IL" -p dexengine/robots/usd/flatten.py \
      --input "$D/${c}_base_test.usd" --output "$D/${c}_flat_test.usd" > "$LOG/${c}_2flatten.log" 2>&1
  timeout "$TO" "$IL" -p dexengine/robots/usd/mimic.py \
      --input-usd "$D/${c}_flat_test.usd" --output-usd "$D/${c}_stage3_test.usd" > "$LOG/${c}_3mimic.log" 2>&1
  [ -f "$D/${c}_stage3_test.usd" ] && ok=$((ok+1)) || echo "$c: no stage-3 output — $(grep -m1 -iE 'error|traceback' "$LOG/${c}_3mimic.log" | cut -c1-100)"
done
timeout "$TO" "$IL" -p dexengine/rl/assets/contact_report_bake.py --combos "$combos" --combo-dir "$D" > "$LOG/stage4.log" 2>&1
s4=$?
grep -E "^\[stage4\]" "$LOG/stage4.log"
grep -q "^\[stage4\] DONE" "$LOG/stage4.log" || s4=${s4/#0/9}   # Isaac can exit 0 on failure — the DONE marker decides
final=0
for S in R L; do [ -f "$D/${STEM}_split_$S.usd" ] && final=$((final+1)); done
echo "FLOATING_USD_DONE stage3 $ok/2 · stage4 rc=$s4 · canonical usd $final/2 · $D"
[ "$ok" = 2 ] && [ "$final" = 2 ] && [ "$s4" = 0 ]
