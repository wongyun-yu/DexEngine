#!/bin/bash
# Judge sidecar: the fresh-process PARK judge of one training run (paper Sec. 3.4 evaluation protocol).
#
# Judge value = the mean AGGREGATE completion of the final judges: the checkpoints at MI x 7/8 and MI
#   (dexengine.rl.judge.metrics final_iters), each measured by a new trainer process (python3 scripts/stage3_rl.py
#   with FROZEN_REPLAY_CKPT; PARK between judge batches is that context's default, dexengine.rl.judge.launch) with one
#   warm-up batch and the judge shape (judge NE, evaluations per clip >= the minimum; dexengine.rl.judge.metrics).
#   Every judge also records its 10 % pass rate (column pass10); before the DONE rc line the sidecar writes the judge
#   record <RUN>_judge_extras.json (run value, best and last).
#   During training, RLJUDGE_QUICK_EVERY (default 200; campaigns pass 0) iterations apart, a quick judge of the same
#   shape reads the trend; only the newest is measured, older pending ones get SKIP rows, and none start once the
#   last checkpoint exists (final judges first). Iterations: dexengine.rl.judge.metrics mids.
# Scope: SIDE=both (bimanual) and SIDE=right|left (one hand); any other SIDE dies.
# Usage: start it with the SAME environment as the training (judge conditions = training conditions; the one
#   exception is the termination rule: the judge carries no contact termination, dexengine.rl.judge.terms). The RL
#   switches come from the training run's switches record ($CKR/<RUN>/switches.json, written by the launcher when the
#   training starts): every judge gets `--switches-from` that record (no environment carries them); a run without
#   the record fails.
#   ENV="DEXENGINE_ARM=rby1 DEXENGINE_HAND=<hand> DEXENGINE_DATASET=<ds> DEXENGINE_EMBODIMENT=<emb> SIDE=<side> SEQ_DIR=<dir> DEXENGINE_SEQ_LIST=<list>"
#   env $ENV GPU=0 setsid nohup python3 scripts/stage3_rl.py <RUN> > <RUN>.launch.out 2>&1 < /dev/null &
#   env $ENV RLJUDGE_GPUS="1" setsid nohup bash scripts/rl/judge_sidecar.sh <RUN> > <RUN>.judge.out 2>&1 < /dev/null &
#   A finished run is judged by the same command (all checkpoints exist, so it goes straight to the final judges).
# Gate (scripts/rl/host_gate.sh, shared with the training runner): a candidate GPU from RLJUDGE_GPUS with
#   memory.used < 500 MiB that no judge or training holds; host MemAvailable minus the start-up reservations
#   (RLJUDGE_RSS_GB per judge for RLJUDGE_BOOT_S seconds, the runner's value per training) >= RLJUDGE_MIN_AVAIL_GB;
#   at most RLJUDGE_MAX_CONC judges per machine.
# Outputs (in the output folder, below): <RUN>_judge.tsv (one row per judge) and <RUN>_judge.rc =
#   "DONE <value> pass10=<k>/<n>" | "FAIL <reason>" | "INCOMPLETE <reason>". The JUDGE row label park_last3 is a
#   machine id kept for readers. A training failure is a TRAIN_FAIL row of this run in the launcher's summary table
#   (dexengine.rl.launch.records summary-tsv). Judge logs: overfit/logs/refbase/<RUN>_j{q,f}<iter>_frozenreplay_train.log.
#   Output folder = overfit/logs/refbase, or RLJUDGE_OUT_DIR=<folder> (the campaign runner's cell). That folder must
#   exist, be an RL cell (dexengine.core.paths --what rl-cell-of) and belong to DEXENGINE_HAND; it then needs
#   RLJUDGE_LAUNCH_ID: a table/rc/id already in the cell is continued only under the same id (a restart of the same
#   launch; the old rc is dropped), a different id dies, a new cell records the id in <RUN>_judge.id. The rc file's
#   second line is "launch=<id>". RLJUDGE_LAUNCH_ID without RLJUDGE_OUT_DIR dies.
# Overflow: the final-judge candidates are RLJUDGE_GPUS plus the GPUs the host gate holds for this run (the runner
#   holds a run's training GPU when the judge wait passes its threshold). Quick judges never use held GPUs. Holds are
#   released as soon as every final judge has started.
# A judge row is valid (else INVALID, not counted) when: launcher rc 0; [EVAL_LAUNCH] ctx=fresh with PARK on; the
#   [EVAL_PARK] and [EVAL_PARK_CUT] batch lines both number warm-up + real batches; warm-up = 1; no PARK-invalid batch
#   (dexengine.rl.judge.park_recount); the judge shape holds (per-clip evaluations >= minimum, sums = AGGREGATE,
#   clips = roster; metrics row); COLL_PRESET, RESET_SCHEME, SIDE and the hand-recipe env channels
#   (dexengine.rl.launch.hand_recipe env-channels) of the judge manifest equal the training manifest's; the judge log
#   has the recipe ledger and create_env [HAND_RECIPE] lines; the finger-penalty marker (path=frozen_replay) appears
#   once per side and the [bimanual] line matches SIDE; the contact termination is the judge standard, both as
#   declared ([JUDGE_TERM] contact_term=<std>) and as measured ([FROZEN EVAL iter ...] contact_term=<std>); the
#   articulation theta term count matches its declaration; the roster hash ([ROSTER] sha256), the extra-slot and
#   extra-collision markers equal the training's; the code tree id (manifest code=) is equal and both codedirty=0
#   (HEAD is not compared). Judging an old checkpoint with different code needs RLJUDGE_ALLOW_TREE_CHANGE=1, which the
#   row records. EXTRA and TRAIN_EXTRA lines must equal the training's.
# Checkpoint <-> iteration: agent_<iter x rollouts>.pt (rollouts = the manifest's DEXENGINE_ROLLOUTS, else 32). Every
#   [CKPT_SAVED] marker of the training logs must follow this rule (else FAIL). With a current trainer (its log has
#   the [INTRAIN_EVAL] start-up line) a checkpoint is judged only after its [CKPT_SAVED] line (printed once
#   torch.save returned); older logs are finished runs, judged by file presence. Auto-resume pieces <RUN>_R<iter> count.
# Deadlines: whole sidecar RLJUDGE_DEADLINE_S (86400); one judge RLJUDGE_TIMEOUT_QUICK_S (1500) / _FINAL_S (2400).
#   Past a deadline the sidecar stops with INCOMPLETE (no retry).
# Trajectory record (RLJUDGE_TRAJ=1, default 0): every final judge also writes <OUT>/<RUN>_jf<iter>_traj.npz through
#   the trainer's recorder (DEXENGINE_TRAJ_RECORD); the row is OK only with the recorder's [TRAJ_REC] DONE line and file.
set -uo pipefail

say() { echo "[JUDGE_SIDECAR $(date +%T)] $*"; }
die() { echo "[JUDGE_SIDECAR] $*" >&2; exit 2; }

release_gpu() { host_gate_release gpu "$1"; }

finish() { # $1 = DONE|FAIL|INCOMPLETE  $2 = value or reason
  local k
  for k in "${!RUNNING[@]}"; do
    kill -TERM -- "-${RUNNING[$k]}" 2>/dev/null; release_gpu "${KEYGPU[$k]:-x}"
    printf "%s\t%s\t%s\t-\t%s\t-\t-\t-\tABORT\tsidecar stopped (%s)%s\n" "$(date -Is)" "${k%%:*}" "${k#*:}" "${KEYGPU[$k]:-?}" "$1" \
      "$JMPAD" >> "$TSV"
  done
  { echo "$1 $2"; [ -z "$LID" ] || echo "launch=$LID"; } > "$RCF.tmp.$$" && mv -f "$RCF.tmp.$$" "$RCF"; rm -f "$PIDF"
  say "$1 $RUN — $2"
  case $1 in DONE) exit 0 ;; INCOMPLETE) exit 7 ;; *) exit 1 ;; esac
}

man_tok() { # $1 = manifest, $2 = key -> the value of the first "key=value" token (up to a space)
  grep -aoE "(^| )$2=[^ ]*" "$1" 2>/dev/null | head -1 | sed -E "s/^ ?$2=//"
}

man_rest() { # $1 = manifest, $2 = line head -> the rest of that line (values with spaces such as EXTRA)
  grep -a "^$2=" "$1" 2>/dev/null | head -1 | sed -E "s/^$2=//; s/ LGATE_RAMP=.*$//"
}

park_bad() { # $1 = -q|-c|-h, rest = logs (or stdin) -> the [EVAL_PARK] batches invalid under today's rule
  # (dexengine.rl.judge.park_recount still-invalid). Byte semantics (LC_ALL=C) so any grep build matches the UTF-8 text.
  local o=$1; shift
  LC_ALL=C grep -ah "^\[EVAL_PARK\] .*INVALID(" "$@" | python3 -m dexengine.rl.judge.park_recount still-invalid | case "$o" in
    -q) LC_ALL=C grep -q . ;; -c) LC_ALL=C grep -c . ;; -h) cat ;; *) echo "park_bad: unknown option $o" >&2; return 2 ;; esac
}

park_gates() { # $1 = log -> invalid batches per gate, "gate A 3 · gate D 1" (reason form 'INVALID(gate X — …)', dexengine.rl.judge.park)
  local g; g=$(park_bad -h "$1" 2>/dev/null | LC_ALL=C grep -ao "INVALID(gate [A-Z]" | LC_ALL=C sort | uniq -c \
    | awk '{printf "%sgate %s %d", (NR > 1 ? " · " : ""), $3, $1}')
  echo "${g:-gate not marked}"
}

coll_payload() { # $@ = logs -> statement part of the first [EXTRA_COLL] colliders line (counts dropped)
  grep -ahE "^\[EXTRA_COLL\] colliders\(" "$@" 2>/dev/null | head -1 | sed -E 's/^[^)]*\) //; s# [0-9]+/[0-9]+##g'
}

park_excl_a_note() { # $1 = log -> row note " · PARK A check excluded clips k (still scored)" when gate A excluded clips (judge record park_excl_a)
  LC_ALL=C grep -a "^\[EVAL_PARK_EXCL_A\] tally: " "$1" 2>/dev/null | tail -1 \
    | sed -nE 's/^\[EVAL_PARK_EXCL_A\] tally: clips left out of the gate checks ([0-9]+).*/\1/p' \
    | awk '$1 > 0 {printf " · PARK A check excluded clips %d (still scored)", $1}'
}

ckpt_of() { # $1 = iteration -> path of a judgeable checkpoint ("" when none): the run folder, then auto-resume pieces
  local n="agent_$(( $1 * ROLL )).pt" d
  if [ "$NEWTR" = 1 ] && ! grep -qx -- "$n $1" <<< "$SAVED"; then echo ""; return 0; fi   # not yet marked saved
  for d in "$CKR/$RUN" "$CKR/${RUN}"_R[0-9]*; do
    [ -f "$d/checkpoints/$n" ] && { echo "$d/checkpoints/$n"; return 0; }
  done
  echo ""
}

refresh_saved() { # once per poll, in the main shell (finish must end the sidecar): [CKPT_SAVED] list and trainer kind
  local logs=("$REF/${RUN}"_train*.log "$REF/${RUN}"_R[0-9]*_train.log) bad
  SAVED=$(python3 -m dexengine.rl.launch.records ckpt-saved "${logs[@]}")
  bad=$(awk -v r="$ROLL" 'NF == 2 && $1 != "agent_" ($2 * r) ".pt" {print; exit}' <<< "$SAVED")
  [ -z "$bad" ] || finish FAIL "checkpoint mapping mismatch: training log [CKPT_SAVED] '$bad' ≠ rule agent_<iter × rollouts $ROLL>.pt"
  if grep -aqs '\[INTRAIN_EVAL\] ' -- "${logs[@]}"; then NEWTR=1; else NEWTR=0; fi   # file arguments (piped grep -q can SIGPIPE)
}

alias_live() { # $1 = judge name -> rc 0 when a live claim (also of another sidecar instance) is measuring it
  local f p t a _r _x
  for f in "$LOCKDIR"/gpu*.claim; do
    [ -e "$f" ] && read -r p t _r a _x < "$f" 2>/dev/null && [ "$a" = "$1" ] && host_gate_pid_ok "$p" "$t" && return 0
  done
  return 1
}

claim_gpu() { # $1 = judge name $2 = candidate GPUs -> stdout = claimed GPU (rc 0), or rc 1 and the reason on stderr
  host_gate_claim gpu "$2" "$RUN" "$1" "$RSSGB" "$BOOT_S" "$MAXC" "$$"
}

final_cands() { # final-judge candidates into FCANDS = RLJUDGE_GPUS + this run's held GPUs (main shell: the note must print)
  local h; h=$(host_gate_held "$RUN"); FCANDS="$GPUS${h:+ $h}"
  [ -z "$h" ] || [ "$h" = "$HELD_SEEN" ] || say "overflow GPU $h — the host gate holds this run's training GPU; added to the final-judge candidates"
  HELD_SEEN=$h
}

traj_file() { # $1 = kind $2 = iteration -> absolute record path (RLJUDGE_TRAJ=1 and final judges only; else empty)
  [ "$JT" = 1 ] && [ "$1" = f ] || return 0
  local n; n=$(python3 -m dexengine.rl.judge.metrics traj-name --run "$RUN" --kind f --it "$2") && [ -n "$n" ] || return 1
  realpath -sm "$OUT/$n"   # the recorder prints os.path.abspath: same form (symlinks not resolved)
}

traj_verify() { # $1 = kind $2 = iteration $3 = judge log -> mismatch reason (empty when fine)
  [ "$JT" = 1 ] && [ "$1" = f ] || return 0
  local f; f=$(traj_file "$1" "$2") || { echo "no trajectory name (dexengine/rl/judge/metrics.py traj-name)"; return 0; }
  grep -aF "[TRAJ_REC] DONE " "$3" 2>/dev/null | grep -qF " · file $f sha256=" \
    || { echo "[TRAJ_REC] DONE marker missing ($(basename "$f"))"; return 0; }
  [ -s "$f" ] || echo "trajectory file missing ($(basename "$f"))"
}

launch_judge() { # $1 = q|f  $2 = iteration  $3 = GPU -> start one judge in the background and record it in RUNNING
  local kind=$1 it=$2 g=$3 alias ck to trj
  alias=$(printf "%s_j%s%04d" "$RUN" "$kind" "$it"); ck=$(ckpt_of "$it")
  trj=$(traj_file "$kind" "$it") || finish FAIL "no trajectory record name right before the judge start (dexengine/rl/judge/metrics.py traj-name)"
  [ -n "$ck" ] || finish FAIL "no checkpoint of iteration $it right before the judge start (internal error)"
  if [ "$kind" = f ]; then to=${RLJUDGE_TIMEOUT_FINAL_S:-2400}; else to=${RLJUDGE_TIMEOUT_QUICK_S:-1500}; fi
  mkdir -p "$CKR/$alias"; cp "$CKR/$RUN/coll_preset.txt" "$CKR/$alias/coll_preset.txt" 2>/dev/null
  say "judge start $alias <- $(basename "$ck") GPU$g (EPC $JEPC · timeout ${to}s)"
  # Training/render inputs the judge mode has no consumer for are dropped (TRAIN_EXTRA stays: the manifests are
  # compared); trajectory-record inputs never leak in from the outer shell — a final judge gets its record file only
  # with RLJUDGE_TRAJ=1.
  setsid env -u PROBE -u DO_SMOKE -u RENDER_ONLY -u EVAL_CKPT -u SMOKE -u RENDER -u FORCE_TRAIN \
    -u DEXENGINE_TRAJ_RECORD -u DEXENGINE_TRAJ_RENDER ${trj:+"DEXENGINE_TRAJ_RECORD=$trj"} \
    GPU="$g" FROZEN_REPLAY_CKPT="$ck" FROZEN_REPLAY_NE="$JNE" FROZEN_REPLAY_EVALS_PER_CLIP="$JEPC" FROZEN_REPLAY_ITER="$it" \
    timeout "$to" python3 scripts/stage3_rl.py "$alias" --switches-from "$SWR" > "$OUT/${alias}_judge_launch.out" 2>&1 < /dev/null &
  RUNNING["$kind:$it"]=$!; KEYGPU["$kind:$it"]=$g
  host_gate_bind gpu "$g" "$!" "$RUN" "$alias" "$RSSGB" "$BOOT_S"
}

judge_row() { # $1 = kind $2 = judge log -> stdout "k/K" (rc 0), or the shape-check reason (rc 1)
  local out
  out=$(python3 -m dexengine.rl.judge.metrics row --log "$2" --n-clips "$K" --save-dir "$OUT" --standard 2>&1) \
    && { echo "$out"; return 0; }
  echo "${out//$'\n'/ }"; return 1
}

verify_judge() { # $1 = q|f  $2 = iteration  $3 = launcher rc -> one TSV row (status OK|INVALID), prints the status
  local kind=$1 it=$2 rc=$3 alias log man bad=() nreal w np ns agg per k tv jv
  alias=$(printf "%s_j%s%04d" "$RUN" "$kind" "$it")
  log="$REF/${alias}_frozenreplay_train.log"; man="$REF/${alias}_frozenreplay_cmd.txt"
  [ "$rc" = 0 ] || bad+=("launcher rc=$rc")
  grep -aqE "^\[EVAL_LAUNCH\] ctx=fresh · DEXENGINE_EVAL_PARK=1\(" "$log" 2>/dev/null || bad+=("PARK-on marker missing")
  read -r nreal w < <(LC_ALL=C grep -aoE "\[C41\]\[(static|balanced)\] [^:]*: real batches [0-9]+ \(\+warmup [0-9]+\)" "$log" 2>/dev/null \
                     | tail -1 | LC_ALL=C sed -E 's/.*real batches ([0-9]+) \(\+warmup ([0-9]+)\)/\1 \2/')
  np=$(grep -acE "^\[EVAL_PARK\] batch [0-9]+: parked" "$log" 2>/dev/null)
  [[ "${nreal:-x}" =~ ^[1-9][0-9]*$ ]] || bad+=("real-batch marker ${nreal:-missing}")   # per-clip evaluations are checked by judge_row
  [ "${w:-x}" = 1 ] || bad+=("warm-up ${w:-?} ≠ 1 (judge standard)")
  [ -n "${w:-}" ] && [ "$np" = $(( w + ${nreal:-0} )) ] || bad+=("[EVAL_PARK] lines $np ≠ warm-up ${w:-?} + real batches ${nreal:-?}")
  ns=$(grep -acE "^\[EVAL_PARK_CUT\] batch [0-9]+:" "$log" 2>/dev/null)
  [ -n "${w:-}" ] && [ "$ns" = $(( w + ${nreal:-0} )) ] || bad+=("[EVAL_PARK_CUT] lines $ns ≠ warm-up ${w:-?} + real batches ${nreal:-?}")
  ! park_bad -q "$log" 2>/dev/null || bad+=("[EVAL_PARK] invalid batch ($(park_gates "$log"))")
  grep -aqE "^\[HAND_RECIPE\] train: checkpoint ledger (matches|missing)" "$log" 2>/dev/null || bad+=("[HAND_RECIPE] ledger check marker missing")
  grep -aqE "^\[HAND_RECIPE\] create_env: " "$log" 2>/dev/null || bad+=("[HAND_RECIPE] create_env marker missing")
  grep -aqE "^\[JUDGE_TERM\] contact_term=$JCT_STD\(judge standard" "$log" 2>/dev/null || bad+=("[JUDGE_TERM] judge-standard contact termination ($JCT_STD) marker missing")
  local ctm ats atm; ctm=$(LC_ALL=C grep -aoE "\[FROZEN EVAL iter [0-9]+\] .* contact_term=[0-9]+" "$log" 2>/dev/null | tail -1 | grep -oE "[0-9]+$")
  [ "${ctm:-x}" = "$JCT_STD" ] || bad+=("judge log contact termination contact_term=${ctm:-missing} ≠ judge standard $JCT_STD")
  ats=$(grep -aoE "^\[JUDGE_TERM\] artic_theta=[^(]*" "$log" 2>/dev/null | tail -1 | cut -d= -f2 | tr ',' '\n' | grep -c .)
  atm=$(LC_ALL=C grep -aoE "\[FROZEN EVAL iter [0-9]+\] .* artic_theta=[0-9]+" "$log" 2>/dev/null | tail -1 | grep -oE "[0-9]+$")
  [ "${atm:-x}" = "${ats:-0}" ] || bad+=("θ termination term: judge log measured artic_theta=${atm:-missing} ≠ registered marker ${ats:-0}")
  local rst rsj
  rst=$(grep -ah "^\[ROSTER\] " "$REF/${RUN}"_train*.log 2>/dev/null | head -1 | grep -oE "sha256=[0-9a-f]+$")
  rsj=$(grep -a "^\[ROSTER\] " "$log" 2>/dev/null | head -1 | grep -oE "sha256=[0-9a-f]+$")
  [ -n "$rst" ] && [ "$rst" = "$rsj" ] || bad+=("clip roster training ${rst:-unrecorded} ≠ judge ${rsj:-unrecorded} ([ROSTER] sha256)")
  local nc nsd=1 lb=0
  nc=$(grep -acE "^\[FINGER_PENALTY\] path=frozen_replay " "$log" 2>/dev/null)
  [ "$SIDE" = both ] && nsd=2
  grep -aqE "^\[bimanual\] --bimanual 1" "$log" 2>/dev/null && lb=1
  [ "${nc:-0}" = "$nsd" ] || bad+=("[FINGER_PENALTY] path=frozen_replay lines ${nc:-0} ≠ sides $nsd (SIDE=$SIDE)")
  local xst xsj
  xst=$(grep -ah "^\[EXTRA_SLOTS\] n=" "$REF/${RUN}"_train*.log 2>/dev/null | head -1 | grep -oE "n=[0-9]+ K=[0-9]+ clips=[0-9]+/[0-9]+")
  xsj=$(grep -a "^\[EXTRA_SLOTS\] n=" "$log" 2>/dev/null | head -1 | grep -oE "n=[0-9]+ K=[0-9]+ clips=[0-9]+/[0-9]+")
  [ "$xst" = "$xsj" ] || bad+=("extra slots training ${xst:-off} ≠ judge ${xsj:-off} ([EXTRA_SLOTS] marker)")
  local xct xcj
  xct=$(coll_payload "$REF/${RUN}"_train*.log)
  xcj=$(coll_payload "$log")
  [ -z "$xct" ] || [ -z "$xcj" ] || [ "$xct" = "$xcj" ] || bad+=("extra-slot colliders training [$xct] ≠ judge [$xcj] ([EXTRA_COLL] colliders marker)")
  [ "$lb" = $(( nsd - 1 )) ] || bad+=("judge log bimanual marker ([bimanual] --bimanual 1) $([ "$lb" = 1 ] && echo present || echo missing) ≠ SIDE=$SIDE")
  for k in COLL_PRESET RESET_SCHEME SIDE $RECIPE_ENVS; do   # recipe env channels too (the EXTRA comparison cannot see them)
    tv=$(man_tok "$TMAN" "$k"); jv=$(man_tok "$man" "$k"); [ "$tv" = "$jv" ] || bad+=("$k training $tv ≠ judge ${jv:-missing}")
  done
  local note="" ct cj dt dj tree=""
  note=$(park_excl_a_note "$log")
  ct=$(man_tok "$TMAN" code); cj=$(man_tok "$man" code); dt=$(man_tok "$TMAN" codedirty); dj=$(man_tok "$man" codedirty)
  if [ -z "$ct" ] || [ -z "$cj" ]; then tree="code tree unrecorded (training ${ct:-missing} · judge ${cj:-missing})"
  elif [ "$ct" != "$cj" ]; then tree="code tree training $ct ≠ judge $cj"
  elif [ "$dt" != 0 ] || [ "$dj" != 0 ]; then tree="uncommitted code changes (training ${dt:-?} · judge ${dj:-?})"
  fi
  if [ -n "$tree" ]; then
    if [ "${RLJUDGE_ALLOW_TREE_CHANGE:-0}" = 1 ]; then note+=" ($tree — explicitly allowed)"
    else bad+=("$tree — set RLJUDGE_ALLOW_TREE_CHANGE=1 to judge with other code"); fi
  fi
  [ "$(man_rest "$TMAN" EXTRA)" = "$(man_rest "$man" EXTRA)" ] || bad+=("EXTRA differs from training")
  local tj; tj=$(traj_verify "$kind" "$it" "$log"); [ -z "$tj" ] || bad+=("$tj")
  [ "$(man_rest "$TMAN" TRAIN_EXTRA)" = "$(man_rest "$man" TRAIN_EXTRA)" ] || bad+=("TRAIN_EXTRA differs from training (the judge branch does not take it)")
  agg=$(LC_ALL=C grep -aoE "^\[FROZEN_REPLAY\] DONE .*aggregate_completion_rate=[0-9.]+" "$log" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$")
  [ -n "$agg" ] || bad+=("AGGREGATE missing")
  per=$(awk '/AGGREGATE EVALUATION REPORT/{exit} /^  Completion rate:/{print $3}' "$log" 2>/dev/null | paste -sd,)
  local st=OK p10
  p10=$(judge_row "$kind" "$log") || { bad+=("$p10"); p10="-"; }
  [ ${#bad[@]} = 0 ] || st=INVALID
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Is)" "$kind" "$it" "$(basename "$(ckpt_of "$it")")" \
    "${KEYGPU[$kind:$it]:-?}" "${agg:+$(awk -v a="$agg" 'BEGIN{printf "%.2f", 100 * a}')}" "${per:--}" "${nreal:-?}" "$st" \
    "$( [ ${#bad[@]} = 0 ] && echo "$log$note" || (IFS='; '; echo "${bad[*]}") )" "$p10" >> "$TSV"
  echo "$st"
}

reap() { # collect finished judges: write their rows and release their GPUs
  local k pid rc st
  for k in "${!RUNNING[@]}"; do
    pid=${RUNNING[$k]}
    kill -0 "$pid" 2>/dev/null && continue
    wait "$pid"; rc=$?
    st=$(verify_judge "${k%%:*}" "${k#*:}" "$rc")
    if [ -z "$st" ]; then   # the verifier itself died: record it, never relaunch silently
      st=INVALID
      printf "%s\t%s\t%s\t-\t%s\t-\t-\t-\t%s\tverifier error (see the sidecar output)%s\n" "$(date -Is)" "${k%%:*}" "${k#*:}" \
        "${KEYGPU[$k]}" "$st" "$JMPAD" >> "$TSV"
    fi
    unset "RUNNING[$k]"; STATE[$k]=$st
    say "judge done $k rc=$rc -> $st"
    release_gpu "${KEYGPU[$k]}"
  done
}

wake_sig() { # what the poll loop reacts to: checkpoint files · their [CKPT_SAVED] lines (written after the file; a late
  # line must wake the loop too) · live judges · GPU claims · summary size (builtins + grep + wc)
  local c=("$CKR/$RUN"/checkpoints/agent_*.pt "$CKR/${RUN}"_R[0-9]*/checkpoints/agent_*.pt) l=("$LOCKDIR"/*) p up=""
  local logs=("$REF/${RUN}"_train*.log "$REF/${RUN}"_R[0-9]*_train.log)
  for p in "${RUNNING[@]}"; do kill -0 "$p" 2>/dev/null && up+=" $p"; done
  echo "${c[*]}|$(grep -ahcs '\[CKPT_SAVED\] ' -- "${logs[@]}" | tr '\n' ' ')|$up|${l[*]}|$(wc -c < "$SUMF" 2>/dev/null)"
}

nap() { # the poll loop's wait: up to POLL s, back NAP s after wake_sig changes (a checkpoint saved, a judge ended, a
  # GPU claim released, a summary row) — event wake, so a judge or a free GPU does not wait out the rest of POLL
  local s0 i n
  n=$(awk -v p="$POLL" -v s="$NAP" 'BEGIN { n = int(p / s); print (n < 1 ? 1 : n) }')   # NAP ticks in POLL
  [ "$n" -gt 1 ] || { sleep "$POLL"; return 0; }
  s0=${SIG0:-$(wake_sig)}   # taken at the top of the pass: an event during the pass wakes the first NAP
  for (( i = 0; i < n; i++ )); do
    sleep "$NAP"
    [ "$(wake_sig)" = "$s0" ] || { sleep "$NAP"; return 0; }   # one more NAP: the [CKPT_SAVED] line follows the file
  done
}

skip_row() { # $1 = iteration $2 = reason
  printf "%s\tq\t%s\t-\t-\t-\t-\t-\tSKIP\t%s%s\n" "$(date -Is)" "$1" "$2" "$JMPAD" >> "$TSV"; STATE["q:$1"]=SKIP
}

write_extras() { # in final_value (main shell): judge record -> $OUT/<RUN>_judge_extras.json, JM_P10, best/last rows
  local line f=$OUT/${RUN}_judge_extras.json rows kind it ck agg why p10 n=0
  line=$(python3 -m dexengine.rl.judge.metrics extras --tsv "$TSV" --dir "$OUT" --run "$RUN" --finals "$FINALS" \
         --judge-ne "$JNE" --judge-epc "$JEPC" --out "$f" 2>&1) \
    || finish FAIL "could not write the judge record (10 % completion · special tags): ${line//$'\n'/ }"
  JM_P10=$(grep -oE "pass10=[0-9]+/[0-9]+" <<< "$line" | cut -d= -f2)
  [ -n "$JM_P10" ] || finish FAIL "the judge record summary has no pass10: $line"
  rows=$(python3 -m dexengine.rl.judge.metrics picks --extras "$f" 2>&1) || finish FAIL "could not read best/last of the judge record: ${rows//$'\n'/ }"
  while IFS=$'\t' read -r kind it ck agg why p10; do
    printf "%s\t%s\t%s\t%s\t-\t%s\t-\t-\t-\t%s\t%s\n" "$(date -Is)" "$kind" "$it" "$ck" "$agg" "$why" "$p10" >> "$TSV"
    n=$((n + 1))
  done <<< "$rows"
  [ "$n" = 2 ] || finish FAIL "judge table best/last rows $n ≠ 2"
  say "[JUDGE_METRICS] $RUN $line -> $f"
}

final_value() { # every final judge OK -> their mean (else FAIL); every quick-judge iteration must have one row
  local it v vals=() sum nq want_q
  want_q=$(wc -w <<< "$JM_MIDS")
  nq=$(awk -F'\t' '$2=="q"{print $3}' "$TSV" | sort -u | wc -l)
  [ "$nq" = "$want_q" ] || finish FAIL "unique iterations of quick-judge rows $nq ≠ $want_q (TSV $TSV)"
  for it in $FINALS; do
    [ "${STATE[f:$it]}" = OK ] || finish FAIL "final judge f:$it = ${STATE[f:$it]} (TSV $TSV)"
    v=$(awk -F'\t' -v i="$it" '$2=="f" && $3==i && $9=="OK"{v=$6} END{print v}' "$TSV"); vals+=("$v")
  done
  sum=$(printf "%s\n" "${vals[@]}" | awk '{s+=$1} END{printf "%.2f", s/NR}')
  write_extras
  printf "%s\tJUDGE\t%s\t-\t-\t%s\t%s\t-\t%s\t%s\t%s\n" "$(date -Is)" "$(echo $FINALS | tr ' ' ,)" "$sum" \
    "$(IFS=,; echo "${vals[*]}")" "$LABEL" "$JNOTE" "$JM_P10" >> "$TSV"
  say "JUDGE $RUN $LABEL = $sum % (iterations $FINALS : ${vals[*]}) · pass10 $JM_P10"
  finish DONE "$sum pass10=$JM_P10"
}

# ---------------------------------------------------------------- start-up checks
case "${1:-}" in -h|--help) sed -n '2,/^set -uo pipefail$/p' "$0" | sed '$d' | cut -c3-; exit 0 ;; esac
RUN=${1:?usage: judge_sidecar.sh <RUN> (environment = the training environment; RLJUDGE_GPUS=<candidate GPUs>)}
cd "$(dirname "$0")/../.." || exit 2
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/rl/host_gate.sh || exit 2   # host gate (locks, reservations, lock folder)
for v in RLJUDGE_ITERS RLJUDGE_RECOUNT RLJUDGE_RECOUNT_FROM RLJUDGE_SPECIAL_WAIT_S RLJUDGE_TIMEOUT_SHADOW_S RLJUDGE_SHADOW_WAIT_S; do
  [ -z "${!v+x}" ] || die "$v is a removed input (explicit iterations, recount, shadow and special judges are gone)"
done
for v in RLJUDGE_SHADOW RLJUDGE_SPECIAL; do
  case "${!v:-0}" in 0) ;; *) die "$v='${!v}': the shadow judge and the special-checkpoint tags are removed (only 0 is accepted)" ;; esac
done
printf '[EVAL_PARK] batch 1: free fall 0 — INVALID(probe)\n' | park_bad -q \
  || die "PARK invalid-line probe failed — this host's grep or the PARK rule module cannot match the invalid sample line"
RECIPE_ENVS=$(python3 -m dexengine.rl.launch.hand_recipe env-channels) && [ -n "$RECIPE_ENVS" ] \
  || die "cannot read the recipe env channels (dexengine.rl.launch.hand_recipe env-channels)"
[ -z "${DEXENGINE_JUDGE_CONTACT_TERM+x}" ] || die "DEXENGINE_JUDGE_CONTACT_TERM is a measurement-arm input outside the judge standard — the sidecar does not take it"
[ -z "${DEXENGINE_JUDGE_OBJTRACK_TERM+x}" ] || die "DEXENGINE_JUDGE_OBJTRACK_TERM is a measurement-arm input outside the judge standard — the sidecar does not take it"
JCT_STD=$(python3 -m dexengine.rl.judge.terms contact-term --standard) && [[ "$JCT_STD" =~ ^[01]$ ]] \
  || die "cannot read the judge-standard contact termination (dexengine.rl.judge.terms)"
: "${DEXENGINE_DATASET:?DEXENGINE_DATASET is required — start with the training environment}" "${DEXENGINE_EMBODIMENT:?DEXENGINE_EMBODIMENT is required}"
: "${ARM:-${DEXENGINE_ARM:?ARM or DEXENGINE_ARM is required}}" "${HAND:-${DEXENGINE_HAND:?HAND or DEXENGINE_HAND is required}}"
case "${SIDE:-}" in both|right|left) ;; *) die "SIDE must be both|right|left (got '${SIDE:-}')" ;; esac
SUMF=$(python3 -m dexengine.rl.launch.records summary-tsv "$SIDE") || exit 2   # the launcher's summary table
[ -n "${SEQ_DIR:-}" ] || die "SEQ_DIR is required — start with the training environment"
# the work areas of the training's run tree (paths.work_area: the data root for a release or copied data root, this
# code tree for the canonical checkout) — where the launcher and the trainer wrote them, not the cwd
REF=$(python3 -m dexengine.core.paths --what work-area --area work_log) || die "cannot read the trainer log area (dexengine.core.paths --what work-area --area work_log)"
LID=""
CKR=$(python3 -m dexengine.core.paths --what work-area --area work_ckpt) || die "cannot read the trainer checkpoint area (dexengine.core.paths --what work-area --area work_ckpt)"
if [ -n "${RLJUDGE_OUT_DIR+x}" ]; then
  [ -n "$RLJUDGE_OUT_DIR" ] || die "RLJUDGE_OUT_DIR is empty — give a folder or unset it"
  OUT=$RLJUDGE_OUT_DIR
  [ -d "$OUT" ] || die "RLJUDGE_OUT_DIR $OUT does not exist (the runner's launch record creates the cell)"
  cell=$(python3 -m dexengine.core.paths --what rl-cell-of --path "$OUT" --tree "$PWD") \
    || die "RLJUDGE_OUT_DIR $OUT is not an RL cell (dexengine.core.paths --what rl-cell-of)"
  [ "${cell%%$'\t'*}" = "${DEXENGINE_HAND:-}" ] \
    || die "RLJUDGE_OUT_DIR belongs to hand ${cell%%$'\t'*}, not DEXENGINE_HAND '${DEXENGINE_HAND:-}'"
  LID=${RLJUDGE_LAUNCH_ID:-}
  [[ "$LID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "RLJUDGE_OUT_DIR needs RLJUDGE_LAUNCH_ID (the launch record id); got '$LID'"
else
  OUT=$REF
  [ -z "${RLJUDGE_LAUNCH_ID+x}" ] || die "RLJUDGE_LAUNCH_ID goes only with RLJUDGE_OUT_DIR"
fi
GPUS=${RLJUDGE_GPUS:?RLJUDGE_GPUS=\"<candidate GPU indices>\" is required (no default; leave out the training GPU)}
QEVERY=${RLJUDGE_QUICK_EVERY:-200}; MAXC=${RLJUDGE_MAX_CONC:-4}
[[ "$QEVERY" =~ ^[0-9]+$ ]] || die "RLJUDGE_QUICK_EVERY '$QEVERY' must be a non-negative integer (0 = no quick judges)"
BOOT_S=${RLJUDGE_BOOT_S:-600}; POLL=${RLJUDGE_POLL_S:-60}; NAP=${RLJUDGE_NAP_S:-2}   # NAP = event check (nap)
[[ "$POLL$NAP" =~ ^[0-9.]+$ ]] && awk -v p="$POLL" -v s="$NAP" 'BEGIN { exit !(p + 0 >= 0 && s + 0 > 0) }' \
  || die "RLJUDGE_POLL_S '$POLL' / RLJUDGE_NAP_S '$NAP': seconds (NAP > 0)"
case "${RLJUDGE_TRAJ:-0}" in
  0) JT=0 ;;
  1) JT=1; echo "[JUDGE_TRAJ] final-judge trajectory record on — $OUT/<RUN>_jf<iter>_traj.npz; a row without the marker and file is INVALID" ;;
  *) die "RLJUDGE_TRAJ='${RLJUDGE_TRAJ}' — only 0|1" ;;
esac
host_gate_env   # LOCKDIR · MINGB · MEMINFO · JUDGE_RSS_GB (defaults in host_gate.sh)
RSSGB=$JUDGE_RSS_GB   # host RAM reserved per judge
T_END=$(( $(date +%s) + ${RLJUDGE_DEADLINE_S:-86400} ))
TSV=$OUT/${RUN}_judge.tsv; RCF=$OUT/${RUN}_judge.rc; PIDF=$OUT/${RUN}_judge.pid; IDF=$OUT/${RUN}_judge.id; TMAN=$REF/${RUN}_cmd.txt
SUMM_N0=$(wc -l < "$SUMF" 2>/dev/null || echo 0)   # only summary rows appended after the sidecar started count
declare -A RUNNING=() KEYGPU=() STATE=()
SAVED=""; NEWTR=0; FCANDS=""; HELD_SEEN=""   # refresh_saved / final_cands fill them every poll
JMPAD=$'\t-'; JM_P10=""   # pass10 column filler for rows without a value
HDR='time\tkind\titer\tckpt\tgpu\tagg_pct\tbatch_pct\tn_real\tstatus\tlog_or_reason\tpass10'
mkdir -p "$LOCKDIR" "$REF" || exit 2
[ "$OUT" = "$REF" ] || say "output folder $OUT (RLJUDGE_OUT_DIR) — judge.tsv · judge.rc · judge.pid · judge launch outputs"
if [ -f "$PIDF" ] && host_gate_pid_ok "$(cat "$PIDF")" ""; then   # alive = the host gate's rule (zombies are dead)
  echo "[JUDGE_SIDECAR] a sidecar for $RUN already runs (pid $(cat "$PIDF")) — refusing a second" >&2; exit 3
fi
r=$(python3 -m dexengine.rl.judge.metrics rule) || die "cannot read the judge rule (dexengine.rl.judge.metrics rule)"
echo "[JUDGE_METRICS] $r"
if [ -f "$TSV" ] && [ "$(head -1 "$TSV")" != "$(printf "$HDR")" ]; then   # never continue a table with other columns
  die "judge table $TSV has another header (an older sidecar's columns) — judge it with the code that made it"
fi
if [ -n "$LID" ]; then   # the cell's table must belong to this launch (never write DONE from another attempt's table)
  if [ -f "$TSV" ] || [ -f "$RCF" ] || [ -f "$IDF" ]; then
    [ "$(cat "$IDF" 2>/dev/null)" = "$LID" ] \
      || die "cell $OUT has the judge table/rc of another launch ($(cat "$IDF" 2>/dev/null || echo 'no id')) — not continuing it as $LID"
    rm -f "$RCF"; say "launch id $LID — restart of the same launch (old rc dropped, recorded judges continue)"
  else
    echo "$LID" > "$IDF"; say "launch id $LID"
  fi
fi
echo $$ > "$PIDF"
rm -f "$OUT/${RUN}_judge_shadow.rc"   # a shadow verdict left by an older sidecar does not describe this judge
trap 'finish INCOMPLETE "stopped by a signal"' INT TERM

# ---------------------------------------------------------------- training manifest, finals, roster size
until [ -f "$TMAN" ]; do   # the training manifest states the judge conditions (MI, NE, tree, preset)
  [ "$(date +%s)" -lt "$T_END" ] || finish INCOMPLETE "training manifest $TMAN missing (deadline)"; sleep "$POLL"
done
SWR=$CKR/$RUN/switches.json   # the training's RL switches: every judge runs under them
until [ -s "$SWR" ]; do   # written with the training's first process (before its GPU wait); checkpoints without it = an old run
  compgen -G "$CKR/$RUN/checkpoints/agent_*.pt" > /dev/null \
    && finish FAIL "training run has checkpoints but no switches record $SWR — judge it with the code that trained it"
  [ "$(date +%s)" -lt "$T_END" ] || finish INCOMPLETE "no switches record $SWR (deadline)"; sleep "$POLL"
done
MI=$(man_tok "$TMAN" MI); NE=$(man_tok "$TMAN" NE); ROLL=$(man_tok "$TMAN" DEXENGINE_ROLLOUTS); ROLL=${ROLL:-32}
[[ "$MI" =~ ^[0-9]+$ && "$NE" =~ ^[0-9]+$ ]] || finish FAIL "could not read MI·NE from the training manifest ($TMAN)"
FINALS=$(python3 -m dexengine.rl.judge.metrics finals --mi "$MI" 2>&1) || finish FAIL "could not read the final-judge iterations from their owner — $FINALS"
LABEL=park_last3; JNOTE="mean = final judges ($(echo $FINALS | tr ' ' ,)) AGGREGATE"
NFIN=$(wc -w <<< "$FINALS")
# the [ROSTER] marker of placement.train_ready_util says "→ train N"
until K=$(cat "$REF/${RUN}"_train*.log 2>/dev/null | LC_ALL=C grep -aoE "^\[ROSTER\] .*→ train [0-9]+" | head -1 | grep -oE "[0-9]+$"); [ -n "$K" ]; do
  [ "$(date +%s)" -lt "$T_END" ] || finish INCOMPLETE "no [ROSTER] clip count in the training log (deadline)"; sleep "$POLL"
done
[ "$K" -gt 0 ] && [ "$K" -le "$NE" ] || finish FAIL "clip count $K is outside (0, NE $NE]"
p=$(python3 -m dexengine.rl.judge.metrics shape --train-ne "$NE" --n-clips "$K" 2>&1) || finish FAIL "no judge shape: ${p//$'\n'/ }"
read -r JNE JEPC <<< "$p"
JM_MIDS=$(python3 -m dexengine.rl.judge.metrics mids --mi "$MI" --quick-every "$QEVERY" 2>&1) || finish FAIL "no mid-judge iterations: ${JM_MIDS//$'\n'/ }"
[ -f "$TSV" ] || printf "$HDR\n" > "$TSV"
while IFS=$'\t' read -r _t kind it _c _g _a _b _n st _r; do   # restart: judges already recorded are not measured again
  case "$kind:$st" in *:ABORT) ;; q:*|f:*) STATE["$kind:$it"]=$st ;; esac   # aborted judges are measured again
done < <(tail -n +2 "$TSV")
say "$RUN MI=$MI NE=$NE rollouts=$ROLL clips=$K · final judges $FINALS · $([ "$QEVERY" = 0 ] && echo "no quick judges" || echo "quick judges every $QEVERY") · GPU candidates $GPUS · gate ${MINGB} GB / max $MAXC"

# ---------------------------------------------------------------- poll loop
while :; do
  SIG0=$(wake_sig)          # before this pass reads anything (nap compares against it)
  [ "$(date +%s)" -lt "$T_END" ] || finish INCOMPLETE "overall deadline — finished judges: $(cut -f2,3,9 "$TSV" | tail -n +2 | paste -sd' ')"
  reap
  refresh_saved
  if grep -aqP "^\Q$RUN\E\t.*\tTRAIN_FAIL\t" <(tail -n +"$((SUMM_N0 + 1))" "$SUMF" 2>/dev/null) \
     && [ -z "$(ckpt_of "$MI")" ]; then   # read through a substitution (piped grep -q can turn false under pipefail)
    finish FAIL "training failed ($(basename "$SUMF") TRAIN_FAIL) — no final checkpoint"
  fi
  if [ -n "$(ckpt_of "$MI")" ]; then            # training finished: pending quick judges SKIP, final judges start
    if [ "$QEVERY" -gt 0 ]; then
      for it in $JM_MIDS; do [ -n "${STATE[q:$it]:-}" ] || skip_row "$it" "training ended — final judges first"; done
    fi
    final_cands
    for it in $FINALS; do
      [ -n "${STATE[f:$it]:-}" ] || [ -n "${RUNNING[f:$it]:-}" ] && continue
      [ -n "$(ckpt_of "$it")" ] || finish FAIL "no final checkpoint at iteration $it (does DEXENGINE_CKPT_INTERVAL not divide it?)"
      a=$(printf "%s_jf%04d" "$RUN" "$it"); alias_live "$a" && { say "waiting f:$it — another instance measures $a"; continue; }
      g=$(claim_gpu "$a" "$FCANDS" 2>"$LOCKDIR/.why.$$") || { say "waiting f:$it — $(cat "$LOCKDIR/.why.$$")"; break; }
      launch_judge f "$it" "$g"
    done
    up=0; for it in $FINALS; do [ -n "${STATE[f:$it]:-}${RUNNING[f:$it]:-}" ] && up=$((up + 1)); done
    if [ "$up" = "$NFIN" ] && [ -n "$HELD_SEEN" ]; then   # every final judge started: the held GPUs have no further use
      for g in $HELD_SEEN; do host_gate_release hold "$g"; done; say "overflow hold released GPU $HELD_SEEN — every final judge started"; HELD_SEEN=""
    fi
    done_n=0; for it in $FINALS; do [ -n "${STATE[f:$it]:-}" ] && done_n=$((done_n + 1)); done
    [ "$done_n" = "$NFIN" ] && [ "${#RUNNING[@]}" = 0 ] && final_value
  elif [ "$QEVERY" -gt 0 ]; then                  # training: the newest quick judge (one at a time per sidecar)
    latest=""
    for it in $JM_MIDS; do
      [ -z "${STATE[q:$it]:-}" ] && [ -z "${RUNNING[q:$it]:-}" ] && [ -n "$(ckpt_of "$it")" ] && latest=$it
    done
    if [ -n "$latest" ] && [ "${#RUNNING[@]}" = 0 ] && ! alias_live "$(printf "%s_jq%04d" "$RUN" "$latest")"; then
      if g=$(claim_gpu "$(printf "%s_jq%04d" "$RUN" "$latest")" "$GPUS" 2>"$LOCKDIR/.why.$$"); then
        for it in $JM_MIDS; do
          [ "$it" -lt "$latest" ] || break
          [ -z "${STATE[q:$it]:-}" ] && [ -n "$(ckpt_of "$it")" ] && skip_row "$it" "superseded — measuring the newest $latest"
        done
        launch_judge q "$latest" "$g"
      else
        say "waiting q:$latest — $(cat "$LOCKDIR/.why.$$")"
      fi
    fi
  fi
  nap
done
