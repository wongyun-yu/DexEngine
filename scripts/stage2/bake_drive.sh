#!/bin/bash
# Bake driver — per-clip anchor retry, parallel lanes over resident workers.
#
# Why retry: the judge sees the wrists before the f0 repair and the bake solves the wrists after it. When the repair
# creates a valley jump at that placement only the bake fails. The judge already holds several candidates per clip
# (ordered by anchor), so a failed candidate moves on to the next one instead of dropping the clip.
#
# Usage: bake_drive.sh <combo> <plans.jsonl> <clips.txt> [lanes] [re-selection record]
#   re-selection record = the record this chunk's judge used (stage2_ground.sh --place-reselect; '' = none): a clip in
#   it whose every candidate fails again is dropped, not re-selected again (place_check --all-rejected --reselect-in).
# Env read (set by stage2_ground.sh):
#   DEXENGINE_DATASET / DEXENGINE_EMBODIMENT (required) · DEXENGINE_SRC_LAYER (empty = paths.default_src_layer)
#   LEDGER (bake ledger) · LOG (log folder) · PYS / PYDP (interpreters; resolved by their owners when unset)
#   RETIME_SAFETY / RETIME_MAX_STRETCH (retiming, passed to chain_fast.sh) · DEXENGINE_ACTIVE_SIDES (auto = derive
#   the idle side of an old two-arm plan) · DEXENGINE_BAKE_CPU=1 (bake without a GPU) · CUDA_VISIBLE_DEVICES (lanes)
#   and, passed through to chain_fast.sh: DEXENGINE_F0_STAGE · DEXENGINE_REUSE_IK · DEXENGINE_CROSS_REPAIR ·
#   DEXENGINE_XR_LEDGER · DEXENGINE_MAIN_ROOT.
# The values the chain needs (arm, hand, layer roots, suffixes, IK pool, source layer, the three placement-record
# values) are resolved here once and passed as options — the chain does not resolve them per attempt.
# Out: <ledger> rows "<clip>\t(OK|FAIL|RETRY)\t<anchor>\t<reason>"
set -uo pipefail
WT=$(cd "$(dirname "$0")/../.." && pwd)
# The repo is importable as a package. Exported before the resident workers start (a worker refuses a request whose
# environment differs from its start environment), and idempotent (stage2_ground.sh already exported it).
case ":${PYTHONPATH:-}:" in *":$WT:"*) ;; *) export PYTHONPATH="$WT${PYTHONPATH:+:$PYTHONPATH}" ;; esac
# The dataset axis is required (no silent default).
DS_ARGS=(--dataset "${DEXENGINE_DATASET:?DEXENGINE_DATASET is required (no default)}"
         --emb "${DEXENGINE_EMBODIMENT:?DEXENGINE_EMBODIMENT is required (no default)}")
TREE_SFX=$(python3 -m dexengine.core.paths --what sfx)   # tree suffix owner (not assembled in the shell)
# Default lanes = 2/3 of the usable CPUs (capacity: cpuset and cpu.max). The GPU is used briefly in the IK stage only,
# so it is not the bottleneck.
COMBO=${1:?combo}; PLANS=${2:?plans.jsonl}; CLIPS=${3:?clips.txt}
NPAR=${4:-}
if [ -z "$NPAR" ]; then CPUS=$(python3 -m dexengine.core.capacity cpus) || exit 2; NPAR=$(( CPUS * 2 / 3 )); [ "$NPAR" -ge 1 ] || NPAR=1; fi
RESELECT_IN=${5:-}
[ -z "$RESELECT_IN" ] || [ -f "$RESELECT_IN" ] || { echo "[bake_drive] ✗ re-selection record $RESELECT_IN is missing" >&2; exit 2; }
LEDGER=${LEDGER:-/tmp/v5_bake.tsv}
LOG=${LOG:-/tmp/v5_clean}
# Interpreters from their owners (contract env wins): retarget = paths.retarget_py (RETARGET_PY first) · Isaac =
# scripts/_isaac_py.sh (ISAACLAB_DIR first, path only). Exported before the workers start.
PYS=${PYS:-$(python3 -m dexengine.core.paths --what retarget-py)} || exit 2
export PYS
. "$WT/scripts/_isaac_py.sh" || { echo "[bake_drive] cannot read scripts/_isaac_py.sh" >&2; exit 2; }
_PYDP=${PYDP:-$(isaac_py)} || exit 2
# Failure reasons go to the combo meta folder (paths.meta_dir — /tmp logs vanish on reboot).
_META_DIR=$(python3 -m dexengine.core.paths --what meta-dir --combo "$COMBO" "${DS_ARGS[@]}") || exit 2
FAILTSV=$_META_DIR/bake_fail_reasons.tsv
mkdir -p "$LOG" "$(dirname "$FAILTSV")"; : > "$LEDGER"
[ -s "$FAILTSV" ] || printf 'clip\tanchor\tattempt\treason (stage log)\tsummary\ttime\n' > "$FAILTSV"

cd "$WT"

IK_SFX=$(python3 -m dexengine.core.paths --what ik_dir_sfx) || exit 2   # suffix owner = paths.ik_suffix (env first)
# Tree and combo output roots — layer owner (paths.trees_dir / robots_dir), passed to the chain.
TROOT=$(python3 -m dexengine.core.paths --what layer --layer trees --combo "$COMBO" "${DS_ARGS[@]}") || exit 2
W2=$(python3 -m dexengine.core.paths --what layer --layer precompute --combo "$COMBO" "${DS_ARGS[@]}") || exit 2
# combo → (arm, hand) = paths.split_combo (dies when it cannot split) · source layer (empty = paths.default_src_layer) ·
# IK pool (paths.pool_root, env DEXENGINE_POOL_ROOT first) · cross-repair mesh root = parent of the assets_objects layer.
_SC=$(python3 -m dexengine.core.paths --what split-combo --combo "$COMBO") || exit 2
IFS=$'\t' read -r ARM HAND _VAR <<< "$_SC"
SRC_LAYER=${DEXENGINE_SRC_LAYER:-$(python3 -m dexengine.core.paths --what src-layer "${DS_ARGS[@]}")} || exit 2
POOL=$(python3 -m dexengine.core.paths --what pool_root) || exit 2
ASSETS_ROOT=$(python3 -m dexengine.core.paths --what layer --layer assets_objects "${DS_ARGS[@]}") || exit 2
ASSETS_ROOT=$(dirname "$ASSETS_ROOT")
echo "[bake_drive] dataset=$DEXENGINE_DATASET emb=$DEXENGINE_EMBODIMENT src_layer=$SRC_LAYER arm=$ARM hand=$HAND trees=$TROOT robots=$W2 pool=$POOL"
# The bake overwrites the tree and both IK files IN PLACE. A chain dying between its right and left IK leaves a
# half-written clip; the end-of-run check below finds every such clip (rc 4) but cannot restore it.
echo "[bake_drive] no re-bake protection: a failed re-bake can leave a clip half-written (see [HALFWRITE_CHECK])"

# Per clip, the anchor candidates in ascending order (anchor = placement preference).
PLANDIR=$(mktemp -d /tmp/v5plan.XXXXXX)
# If the split dies there is no plan file and the lanes would bake 0 clips and end "OK=0 FAIL=0" — check its rc.
$PYS - "$PLANS" "$CLIPS" "$PLANDIR" "$COMBO" <<'PYP' || { echo "[bake_drive][fatal] plan split failed (rc=$?) — not baking" >&2; exit 2; }
import json, sys
from pathlib import Path
plans, clips_f, outdir, combo = sys.argv[1:5]
by = {}
for l in open(plans):
    d = json.loads(l)
    by.setdefault(d["clip"], []).append(d)
want = [l.strip() for l in open(clips_f) if l.strip()]
n = 0
for c in want:
    cand = sorted(by.get(c, []), key=lambda d: d["anchor"])
    if not cand:
        continue
    Path(outdir, f"{n:04d}.json").write_text(json.dumps(
        dict(clip=c, cands=cand), ensure_ascii=False))
    Path(outdir, f"{n:04d}.meta").write_text(f"{len(cand)}\t{c}\n")   # (candidates, clip) read by one() in the shell
    n += 1
print(f"[bake_drive] clips to bake {n} (anchor candidates on average "
      f"{sum(len(by.get(c, [])) for c in want) / max(n, 1):.1f})")
PYP

# Post-bake placement check (the chain's last step) rejected this candidate: its marker line is the reason. Read only
# when the chain's last line is that death (`CLEAN_FAIL placement (`), so an older attempt's log is not read as a rejection.
# The line form belongs to place_check (bake_check_line) and the status ledger reads it in that form — copied whole.
# Updates the caller's (one) DETAIL · PLALL (candidate lines) · NPL (rejections). $1 = chain last line, $2 = all_ log
place_reject_note() {
  local ln=""
  case "$1" in "CLEAN_FAIL placement ("*) ln=$(grep -ha '^\[BAKE_PLACE_CHECK\] ' "$2" 2>/dev/null | tail -1) ;; esac
  [ -z "$ln" ] || { DETAIL=$ln; PLALL+="$ln"$'\n'; NPL=$((NPL + 1)); }
}

# Every candidate failed the placement check (NPL = N): place_check --all-rejected decides the next way (re-selection
# or drop) and records it in this chunk's place_reselect.tsv (marker [PLACE_RESELECT] → bake.log). If recording fails,
# a tail is printed for the FAIL row so its last line is not a marker → BAKE_ERROR (re-bake), never silently lost.
# Uses the caller's C · N · NPL · PLALL.
place_fail_tail() {
  [ "$NPL" -eq "$N" ] || return 0
  printf '%s' "$PLALL" | "$PYS" -m dexengine.placement.bake.place_check --all-rejected --clip "$C" \
    --reselect-out "$(dirname "$LEDGER")/place_reselect.tsv" ${RESELECT_IN:+--reselect-in "$RESELECT_IN"} >&2 \
    || printf '; reselect_record_failed'
}

# A clip out of candidates with attempts lost to the bake-end arm-flip gate (arm_flip.bake_gate): one count line (the
# clip stays GUARD_FAIL). Uses the caller's C · N · LEDGER.
arm_flip_exhausted_note() {
  local naf; naf=$(awk -F'\t' -v c="$C" '$1 == c && $2 == "RETRY" && /arm flipped — failed/' "$LEDGER" | wc -l)
  [ "$naf" -gt 0 ] && echo "[ARM_FLIP_BAKE] $C exhausted — all $N candidates failed · $naf of them arm flip (bake-end gate)"
  return 0
}

one() {   # $1 = plan json, $2 = GPU
  local PJ=$1 G=$2 C ANC RC WHY
  # DEXENGINE_BAKE_CPU=1: bake without a GPU (empty CUDA_VISIBLE_DEVICES → torch CPU). Default = the lane's device.
  [ "${DEXENGINE_BAKE_CPU:-0}" = 1 ] && G=""
  local N AOUT ZOFF IDLE PLALL="" NPL=0 i=0
  IFS=$'\t' read -r N C < "${PJ%.json}.meta"
  while [ "$i" -lt "$N" ]; do
    AOUT=$($PYS - "$PJ" "$i" "$W2/placement" "$HAND" "$SRC_LAYER" <<'PYA'
import json, os, sys
from pathlib import Path
from dexengine.core.paths import tree_name, PLACEMENT_SFX
d = json.load(open(sys.argv[1])); p = d["cands"][int(sys.argv[2])]
out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
rec = dict(
    sequence=tree_name(d["clip"]), dx=p["dx"], dy=p["dy"], yaw=p["yaw"],
    pivot=p["pivot"], robot_z_off=p["robot_z_off"], anchor=p["anchor"],
    valid=True, source="v5 bake_drive")
# The reach map ids the judge verified ({side: store.pool_id}) — the bake writes them into task_info provenance and
# the coarse path compares them with the map it verifies. An old plan line without them writes none.
if "pool" in p:
    rec["pool"] = dict(p["pool"])
# One-hand rule: a plan judged with --active-sides auto carries active_sides/idle_sides — copied as is (the DP stage
# idle rest, cross-repair freeze, contact derivation and gate read them from the tree task_info).
if "active_sides" in p:
    rec["active_sides"] = list(p["active_sides"]); rec["idle_sides"] = list(p["idle_sides"])   # the judge writes both
    rec["active_sides_source"] = "plan"
elif os.environ.get("DEXENGINE_ACTIVE_SIDES", "") == "auto":
    # Re-bake of an old two-arm plan with auto: the idle side comes from the base tree task_info.case (the judge's
    # owner function). No case = die (no silent both).
    from dexengine.core.paths import ref_base_dir
    from dexengine.placement.collision import idle_arm as IA
    from dexengine.core import qpos_layout as QL
    hand, sl = sys.argv[4], sys.argv[5]
    ds = os.environ["DEXENGINE_DATASET"]; emb = os.environ["DEXENGINE_EMBODIMENT"]
    cti = ref_base_dir(ds, hand, emb, sl) / d["clip"] / "task_info.json"
    if not cti.is_file():
        raise SystemExit(f"[bake_drive][ACTIVE_SIDES] {d['clip']}: no base tree task_info — {cti}")
    try:
        # One hand in the tree (qpos_layout) → the missing hand rests ("layout"); two hands → the case rule.
        act, idle, rule = IA.active_idle_for_tree(json.loads(cti.read_text()).get("case"), QL.read(str(cti.parent), nq=None).hands)
    except ValueError as e:
        raise SystemExit(f"[bake_drive][ACTIVE_SIDES] {d['clip']}: {e}")
    rec["active_sides"], rec["idle_sides"], rec["active_sides_source"] = list(act), list(idle), f"{rule}@bake"
    print(f"[ACTIVE_SIDES] bake {d['clip']} active={'+'.join(act)} idle={'+'.join(idle) + '(rest)' if idle else '-'} source={rule}@bake", file=sys.stderr)
(out / f"{tree_name(d['clip'])}{PLACEMENT_SFX}").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
# The three record values for the chain (anchor · robot z offset · idle sides) — it does not reopen adopted.json.
print(f"{rec['anchor']}\t{rec['robot_z_off']}\t{' '.join(rec.get('idle_sides', []))}")
PYA
)
    RC=$?
    IFS=$'\t' read -r ANC ZOFF IDLE <<< "$AOUT"
    if [ "$RC" -ne 0 ] || [ -z "$ANC" ]; then
      # No placement record (adopted.json) → no chain with an empty anchor. Recorded as a failure.
      printf '%s\tFAIL\t-\tplacement record failed (adopted.json rc=%s) — the [bake_drive][ACTIVE_SIDES] line of bake.log\n' "$C" "$RC" >> "$LEDGER"
      return 1
    fi
    # Retiming safety factor / total stretch cap (from stage2_ground.sh; same defaults as retime --safety / --max-stretch)
    CUDA_VISIBLE_DEVICES=$G LOG=$LOG timeout 1200 bash scripts/stage2/chain_fast.sh \
      "$COMBO" "$C" "${RETIME_SAFETY:-1.15}" "${RETIME_MAX_STRETCH:-3.0}" \
      --arm "$ARM" --hand "$HAND" --troot "$TROOT" --w2 "$W2" --ik-sfx "$IK_SFX" --tree-sfx "$TREE_SFX" \
      --ik-pool "$POOL" --assets "$ASSETS_ROOT" --src-layer "$SRC_LAYER" --zoff "$ZOFF" --anchor "$ANC" --idle "$IDLE" \
      --cand "$((i + 1))/$N" >> "$LOG/chain_${C// /_}.log" 2>&1
    RC=$?
    if [ "$RC" -eq 0 ]; then
      printf '%s\tOK\t%s\tattempt%d\n' "$C" "$ANC" "$((i + 1))" >> "$LEDGER"
      return 0
    fi
    # Cut by characters (`cut -c` cuts bytes and would split a multi-byte character; the status ledger reads UTF-8).
    WHY=$(tail -1 "$LOG/chain_${C// /_}.log" | tr -d '\n' | LC_ALL=C.UTF-8 sed -E 's/^(.{70}).*/\1/')
    # The real reason is in the stage log `all_<slug>.log` (the chain log's last line is a summary such as
    # "CLEAN_FAIL chain"); copy it into the ledgers now. The word list matches the failure / stop / violation
    # lines (bake.chain `[runner] stopped — …`, coarse_path `— failed`, final_guard `violation after retiming`).
    DETAIL=$(grep -hoE '\[(stage_dp|final_guard|runner)\][^|]*' \
               "$LOG/all_${COMBO}__${C// /_}.log" 2>/dev/null \
             | grep -E '[Ff]ail|[Aa]bort|[Vv]iolat|stopped' | tail -2 | tr '\n' ';' | LC_ALL=C.UTF-8 sed -E 's/^(.{300}).*/\1/')
    place_reject_note "$WHY" "$LOG/all_${COMBO}__${C// /_}.log"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$C" "$ANC" "attempt$((i + 1))" "${DETAIL:-$WHY}" "$WHY" "$(date +%FT%T)" >> "$FAILTSV"
    # One ledger row per anchor attempt (OK/FAIL counts match `\tOK\t` · `\tFAIL\t`, so RETRY rows do not count).
    printf '%s\tRETRY\t%s\tattempt%d failed: %s\n' "$C" "$ANC" "$((i + 1))" "${DETAIL:-$WHY}" >> "$LEDGER"
    i=$((i + 1))
    [ "$i" -lt "$N" ] && echo "[bake_drive] $C anchor $ANC failed → retrying the next candidate ($WHY)"
  done
  printf '%s\tFAIL\t-\tall %d candidates failed: %s | last reason: %s%s\n' \
    "$C" "$N" "$WHY" "${DETAIL:-none}" "$(place_fail_tail)" >> "$LEDGER"
  arm_flip_exhausted_note; return 1
}

# Device list, resolved once. A child gets an element of the visible list — re-setting CUDA_VISIBLE_DEVICES to an
# index would replace, not compose with, the parent's mask (a `--gpu 1` bake would run on physical GPU 0).
# CUDA_VISIBLE_DEVICES set (even empty) = that list · unset = every nvidia-smi device.
gpu_devices() {
  if [ -n "${CUDA_VISIBLE_DEVICES+x}" ]; then printf '%s' "$CUDA_VISIBLE_DEVICES"; return 0; fi
  nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -
}
gpu_count() { local d=$1; [ -z "$d" ] && { echo 0; return; }; awk -F, '{print NF}' <<<"$d"; }
gpu_nth() {   # $1 = list, $2 = i → list[i % n]; empty list → empty string (CPU)
  local d=$1 i=$2 n; n=$(gpu_count "$d"); [ "$n" -eq 0 ] && { printf ''; return; }
  awk -F, -v k=$(( i % n + 1 )) '{printf "%s", $k}' <<<"$d"
}
GPU_DEVS=$(gpu_devices); NGPU_EFF=$(gpu_count "$GPU_DEVS")
echo "[bake_drive] lanes $NPAR · GPU $NGPU_EFF"
echo "[bake_drive][GPU] visible='${CUDA_VISIBLE_DEVICES-<unset>}' devs='$GPU_DEVS' n=$NGPU_EFF"

# ── Resident worker pool ───────────────────────────────────────────────────────────────────────────────
# One bake/server.py per lane (fewer when there are fewer clips); each clip's chain (chain_fast.sh) sends its stage
# list to a free worker (`server.py send`) instead of starting an interpreter. Worker k's device = the k-th of the
# device list; a worker only takes requests from its own device. The pool path is exported before the start so
# workers and clients see the same environment. At the end only the recorded PIDs are stopped (no pkill).
POOL_PIDS=()
stop_pool() {
  local p
  for p in "${POOL_PIDS[@]}"; do
    if kill -0 "$p" 2>/dev/null && grep -qa dexengine.placement.bake.server "/proc/$p/cmdline" 2>/dev/null; then kill "$p"; fi
  done
  for p in "${POOL_PIDS[@]}"; do while kill -0 "$p" 2>/dev/null; do sleep 0.2; done; done
  [ -n "${DEXENGINE_BAKE_POOL:-}" ] && [ -d "$DEXENGINE_BAKE_POOL" ] && { rm -f "$DEXENGINE_BAKE_POOL"/*.sock "$DEXENGINE_BAKE_POOL"/*.lock "$DEXENGINE_BAKE_POOL"/*.ready "$DEXENGINE_BAKE_POOL"/*.pid "$DEXENGINE_BAKE_POOL"/*.log; rmdir "$DEXENGINE_BAKE_POOL"; }
}
export DEXENGINE_BAKE_POOL; DEXENGINE_BAKE_POOL=$(mktemp -d /tmp/bakepool.XXXXXX)
trap stop_pool EXIT                                    # workers started before a death are stopped too
# Workers = min(lanes, clips to bake) — extra workers only pay the start cost (1 when there are 0 clips).
NPLAN=$(find "$PLANDIR" -maxdepth 1 -name '*.json' | wc -l); NSRV=$(( NPLAN < NPAR ? NPLAN : NPAR )); [ "$NSRV" -ge 1 ] || NSRV=1
for ((k = 0; k < NSRV; k++)); do
  _dev=$(gpu_nth "$GPU_DEVS" "$k"); [ "${DEXENGINE_BAKE_CPU:-0}" = 1 ] && _dev=""
  _sk="$DEXENGINE_BAKE_POOL/s$k.${_dev:-cpu}.sock"
  # Started from a subshell so it is not in this shell's job table — the lane limit below would count the workers
  # as lanes and wait forever after the first clip. The PID comes back through a file.
  ( CUDA_VISIBLE_DEVICES=$_dev exec "$_PYDP" -m dexengine.placement.bake.server --sock "$_sk" --ready "$_sk.ready" \
      > "$DEXENGINE_BAKE_POOL/s$k.log" 2>&1 & echo $! > "$_sk.pid" )
  POOL_PIDS+=("$(cat "$_sk.pid")")
done
# Readiness — success: NSRV ready files · failure: any worker dead · deadline 600 s (first kernel compile included)
_t_end=$(( $(date +%s) + 600 ))
while :; do
  _nr=$(ls "$DEXENGINE_BAKE_POOL"/*.ready 2>/dev/null | wc -l)
  [ "$_nr" -ge "$NSRV" ] && break
  for p in "${POOL_PIDS[@]}"; do kill -0 "$p" 2>/dev/null || { echo "[bake_drive][fatal] resident worker pid $p died before ready — $DEXENGINE_BAKE_POOL/*.log" >&2; exit 2; }; done
  [ "$(date +%s)" -lt "$_t_end" ] || { echo "[bake_drive][fatal] resident worker readiness deadline exceeded ($_nr/$NSRV)" >&2; exit 2; }
  sleep 0.5
done
echo "[BAKE_RESIDENT] pool=$DEXENGINE_BAKE_POOL servers=$NSRV(lanes $NPAR · clips $NPLAN) pids=${POOL_PIDS[*]} (bake/server.py)"
# Lane = worker. A clip's device is the device of the lane that came free (lane k device = worker k device), so a
# request always finds a free worker of its own device. Finished lanes are collected by reap_one (`wait <pid>`).
declare -A LANE_OF=()
FREE_LANES=(); for ((k = 0; k < NSRV; k++)); do FREE_LANES+=("$k"); done
# Lane rc one by one — one() returns only 0 (OK row) or 1 (FAIL row). Anything else (137 = kill/OOM, a bash error) is
# a death in the middle of a clip with no final row for it; after collecting all lanes the driver dies (rc 2).
LANE_BAD=()
lane_rc() { case "$1" in 0|1) ;; *) LANE_BAD+=("lane$2:rc=$1") ;; esac; }
reap_lanes() {  # collect the remaining lanes per pid · marker [LANE_RC]
  local p
  for p in "${!LANE_OF[@]}"; do wait "$p"; lane_rc $? "${LANE_OF[$p]}"; done
  echo "[LANE_RC] abnormal lanes ${#LANE_BAD[@]}${LANE_BAD[*]:+ (${LANE_BAD[*]})}"
}
reap_one() {  # collect one finished lane → _done (pid) · _lrc. Polls each lane pid (kill -0) and collects the finished
  # one with `wait <pid>` (a finished job keeps its rc for `wait <pid>`). Not `wait -n`: in bash 5.2 it skips jobs that
  # ended before the call (rc 127 and _done unset under set -u when every lane ended first).
  local p
  [ "${#LANE_OF[@]}" -gt 0 ] || { echo "[bake_drive][fatal] no lane to collect and no free lane (NSRV=$NSRV)" >&2; exit 2; }
  while :; do
    for p in "${!LANE_OF[@]}"; do kill -0 "$p" 2> /dev/null || { _done=$p; wait "$p"; _lrc=$?; return 0; }; done
    sleep 0.2                                            # one lane = one clip (seconds or more); 0.2 s polling is negligible
  done
}
for PJ in "$PLANDIR"/*.json; do
  [ -e "$PJ" ] || continue
  if [ "${#FREE_LANES[@]}" -eq 0 ]; then
    reap_one
    lane_rc "$_lrc" "${LANE_OF[$_done]}"; FREE_LANES+=("${LANE_OF[$_done]}"); unset "LANE_OF[$_done]"
  fi
  _lane=${FREE_LANES[0]}; FREE_LANES=("${FREE_LANES[@]:1}")
  one "$PJ" "$(gpu_nth "$GPU_DEVS" "$_lane")" &
  LANE_OF[$!]=$_lane
done
reap_lanes
rm -rf "$PLANDIR"
# Half-write check over every clip at the end. A clip that died mid-chain has tree and right arm in the new generation
# and the left arm in the old one; the per-side gate cannot see it. Lengths = bake.chain.halfwrite_lengths (the
# chain's function) on the layer roots the chain wrote ($TROOT · $W2). Half-written = the three differ or R < 0.
# The checker's rc is checked — a dead checker would leave HW empty and end "no half-write" with rc 0.
HW=$($PYS - "$CLIPS" "$W2" "$TROOT" <<'PYHW'
import sys
from pathlib import Path
from dexengine.placement.bake import chain as SR
from dexengine.core.paths import tree_name, ik_suffix
clips_f = sys.argv[1]
sfx = ik_suffix()
W2, TR = Path(sys.argv[2]), Path(sys.argv[3])   # roots resolved by the layer owner in the shell
bad = []
for c in [l.rstrip("\n") for l in open(clips_f) if l.strip()]:
    t = tree_name(c)
    r, l_, q = SR.halfwrite_lengths(W2 / f"arm_ik_poses_right{sfx}" / f"{t}.npz",
                                    W2 / f"arm_ik_poses_left{sfx}" / f"{t}.npz", TR / t / "0/trajectory_kinematic.npz")
    if not (r == l_ == q) or r < 0:
        bad.append(f"{c}\tikR={r},ikL={l_},tree={q}")
print("\n".join(bad))
PYHW
) || { echo "[bake_drive][fatal] half-write checker failed (rc=$?) — not ending as a success without the check" >&2; exit 2; }
echo "[HALFWRITE_CHECK] clips $(grep -c . "$CLIPS" || true) · half-written $(printf '%s\n' "$HW" | grep -c . || true)"
if [ -n "$HW" ]; then
  echo "[bake_drive] ★★ half-write detected — clips whose sides and tree are from different generations:"
  echo "$HW" | sed 's/^/  /'
  echo "[bake_drive] these clips cannot be used for training. Re-bake both sides together with the same warp"
  echo "[bake_drive] (DEXENGINE_REUSE_IK=<judge IK cache> reuses the solution the judge accepted)."
  printf '%s\n' "$HW" >> "$LEDGER.halfwrite"
fi
echo "[bake_drive] OK=$(grep -ca $'\tOK\t' "$LEDGER") FAIL=$(grep -ca $'\tFAIL\t' "$LEDGER")"
echo "[bake_drive] clips saved by a retry: $(awk -F'\t' '$2=="OK" && $4!="attempt1"' "$LEDGER" | wc -l)"

# A lane that died mid-clip is an infrastructure failure (rc 2 — stage 2 dies as an abnormal driver exit). After the
# half-write report so the diagnosis is kept.
[ ${#LANE_BAD[@]} = 0 ] || { echo "[bake_drive][fatal] ${#LANE_BAD[@]} bake lanes died mid-clip (rc ∉ {0,1}): ${LANE_BAD[*]} — $LOG" >&2; exit 2; }
# Any half-write → not a success (done = count match).
[ -n "$HW" ] && exit 4
exit 0
