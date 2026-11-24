#!/bin/bash
# The launch gate of one host. Judges (scripts/rl/judge_sidecar.sh) and trainings (dexengine/rl/campaign.py) use the
#   same lock and the same reservation table, and this file is its one owner: two launchers that do not see each
#   other's reservations read the same MemAvailable and over-allocate (host OOM, judges killed with rc 137).
# Claim file: $LOCKDIR/<kind><GPU>.claim, one line "pid t0 run alias reserve_gb ramp_s".
#   kind gpu = judge (sidecar) · train = training (runner) · data = a judge lane of the data pipeline
#   (a judge launcher that runs several lanes on one GPU, so the file is data<GPU>_<lane>.claim).
#   A claim lives while its pid is alive and did not start after t0 (a later start is a reused pid; an ended zombie
#   is dead — host_gate_pid_ok); dead claims are removed inside the gate.
#   reserve_gb and ramp_s are written by the claimer; each kind has its own value and owner (judge = RLJUDGE_RSS_GB
#   below, host_gate_env · training = dexengine/rl/cell_state.py TRAIN_RESERVE_GB · data = the data-lane
#   launcher's DATA_LANE_RESERVE_GB); the gate only reads them. A claim without both fields closes the gate (its
#   reservation is unknown; it is never counted as 0).
# Gate (checked under flock $LOCKDIR/claim.lock; on success the claim is written before leaving, so two launches never
#   pass together):
#   count — live claims of the same kind < maxc (0 = no limit)
#   RAM   — MemAvailable − Σ(reserve_gb of live claims still starting, i.e. now − t0 < ramp_s, any kind) ≥ MINGB
#   GPU   — the first candidate with no judge, training or data claim and memory.used < 500 MiB. A GPU with an
#           overflow hold is not taken by trainings, and by judges only for the held run. A GPU with a judge-only
#           reservation (rsv) is taken only by judges (gpu)
#   judge share of RAM — the RAM gate of non-judge claims (train, data) keeps, for every judge-only reserved GPU that
#           has no judge seated yet, one judge reservation (JUDGE_RSS_GB) free: MemAvailable − starting reservations −
#           judge share ≥ MINGB. This keeps judges from queueing behind RAM (no reservation = 0)
#   data  — the count field (maxc) is the lane count of one GPU (lanes split one GPU; not a machine-wide limit).
#           The first candidate with no judge or training claim, no hold and fewer than maxc live data claims, at its
#           smallest free lane. memory.used is not read (other lanes already use the GPU; the dispatcher checks
#           memory.free). stdout = <GPU>_<lane> (the GPU field of bind and release)
# Overflow hold — kind hold: binds a GPU whose training ended to that run's final judges only (the runner sets it when
#   the judge GPU wait passes its threshold). One line "pid t0 run hold 0 0", pid = that run's judge sidecar; the hold
#   dies with the sidecar. The sidecar adds the GPUs held for its run to the final-judge candidates (host_gate_held).
#   No reservation, not counted (a marker, not a claim).
# Judge-only reservation — kind rsv: keeps the campaign's judge GPUs (runner --judge-gpus) for judges only (set when
#   the runner loop starts, rechecked every step, removed at the end). One line "pid t0 campaign rsv JUDGE_RSS_GB 0",
#   pid = that loop; the reservation dies with the loop. Trainings, data claims, holds and room skip that GPU. The
#   reservation is neither counted nor a starting reservation (only the judge share of RAM).
# Settings (owned here; the RLJUDGE_* names are kept and the training runner reads the same env):
#   RLJUDGE_LOCK_DIR (default /tmp/dexengine_judge_sidecar_<uid>) · RLJUDGE_MIN_AVAIL_GB (default 70) · RLJUDGE_MEMINFO
#   (default /proc/meminfo) · RLJUDGE_RSS_GB (default 20: the host RAM reserved for one judge, read by the sidecar's
#   claim and the judge share of RAM; a judge with 1024 envs peaks near 16–18 GB per cgroup, 4096 envs near 45–51 GB. It
#   moves with the judge shape in dexengine/rl/judge/metrics.py)
# Use (sourced): source scripts/rl/host_gate.sh; host_gate_env
#   host_gate_claim <kind> "<GPU…>" <run> <alias> <reserve_gb> <ramp_s> <maxc> <pid> → stdout GPU (rc 0) · closed:
#                                        rc 1 + reason (stderr)
#   host_gate_bind <kind> <GPU> <pid> <run> <alias> <reserve_gb> <ramp_s>   # rewrite with the started child pid (t0 = now)
#   host_gate_release <kind> <GPU>
#   host_gate_hold <GPU> <pid> <run>   → overflow hold (rc 0) · bad arguments rc 1 · no sidecar (pid) rc 3 · a live
#                                        training, judge or data claim, another hold or a judge-only reservation on
#                                        that GPU rc 4 (transient: the runner retries next step) + reason (stderr)
#   host_gate_reserve "<GPU…>" <pid> <campaign> → judge-only reservation (rc 0; the same pid's reservation stays) · a
#                                        reservation of another pid, a live training or data claim or a hold: rc 1 +
#                                        reason (nothing written)
#   host_gate_unreserve "<GPU…>" <pid> → removes only that pid's judge-only reservations
#   host_gate_held <run>               → the live hold GPUs of this run (stdout, space separated; empty line if none)
#   host_gate_list                     → live claims and holds, one per line "kind GPU pid t0 run alias reserve_gb
#                                        ramp_s" (read only: called outside the lock, so dead claims are skipped, not
#                                        removed; the runner's judge-wait estimate reads it; a data lane's GPU field is
#                                        the GPU number only)
#   host_gate_room_data "<GPU…>" <lanes> → the GPUs a data claim would fit now, one "GPU free_lane" line each (read
#                                        only: the dispatcher looks before preparing). Host RAM gate closed: rc 1 +
#                                        reason. claim checks again under the lock
#   (the claim's pid is a live process that holds the reservation until the child starts: the sidecar's $$, the
#   runner's own pid)
# Use (CLI): bash scripts/rl/host_gate.sh claim|bind|release <same arguments> · bash scripts/rl/host_gate.sh env

host_gate_env() { # the settings as globals (LOCKDIR · MINGB · MEMINFO)
  LOCKDIR=${RLJUDGE_LOCK_DIR:-/tmp/dexengine_judge_sidecar_$(id -u)}
  MINGB=${RLJUDGE_MIN_AVAIL_GB:-70}
  MEMINFO=${RLJUDGE_MEMINFO:-/proc/meminfo}
  JUDGE_RSS_GB=${RLJUDGE_RSS_GB:-20}
  local v                      # integer GB only: a value like '20.0' makes [ -lt ] below fail as false and the RAM gate would silently open
  for v in "RLJUDGE_MIN_AVAIL_GB=$MINGB" "RLJUDGE_RSS_GB=$JUDGE_RSS_GB"; do
    [[ "${v#*=}" =~ ^[0-9]+$ ]] || { echo "[HOST_GATE] ${v%%=*} '${v#*=}' — must be a non-negative integer GB" >&2; exit 2; }
  done
  mkdir -p "$LOCKDIR"
}

HOST_GATE_PROC=${HOST_GATE_PROC:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." 2>/dev/null && pwd)/dexengine/cluster/proc_ended.py}
#   process-ended rule (standard library only; the shell calls its CLI). Resolved from the real file, so a tree that
#   symlinks only this script still finds it; a test that uses this file alone sets HOST_GATE_PROC.
[ -f "$HOST_GATE_PROC" ] || { echo "[HOST_GATE] FAIL the process-ended rule owner is missing: $HOST_GATE_PROC" >&2; return 2 2>/dev/null || exit 2; }

host_gate_ended() { # $1 pid -> rc 0 = ended. The rule (zombie, X or missing = ended; the leading group and session
  #   are not read) is owned by dexengine/cluster/proc_ended.py `ended`; the shell calls that CLI. An ended process
  #   (Z, X) has an empty /proc/<pid>/cmdline (its address space is gone): a non-empty one means not ended and no
  #   interpreter is started (a shortcut for every live claim, not a verdict — only empty ones go to the owner). If the
  #   owner cannot answer (rc >= 2) it warns and treats the process as not ended (a doubtful claim is alive)
  local a="" rc=0
  { IFS= read -r -d '' a < "/proc/$1/cmdline"; } 2> /dev/null || true     # up to the first argument (NUL)
  [ -n "$a" ] && return 1
  python3 -I -S "$HOST_GATE_PROC" ended "$1" || rc=$?
  [ "$rc" -le 1 ] && return "$rc"
  echo "[HOST_GATE] ✗ liveness check failed (proc_ended.py ended rc $rc · pid $1) — treated as not ended" >&2
  return 1
}

host_gate_pid_ok() { # $1 pid $2 t0 -> rc 0 = alive and started before t0 (+1 s) (a later start = a reused pid); an ended
  #   process is dead (host_gate_ended — kill -0 succeeds on zombies too). The only liveness test of shell claims and locks
  local st bt
  kill -0 "$1" 2>/dev/null || return 1
  host_gate_ended "$1" && return 1
  [[ "$2" =~ ^[0-9]+$ ]] || return 0     # a claim with a broken t0 stays alive, so the gate closes on it as an unknown reservation
  st=$(sed 's/.*) //' "/proc/$1/stat" 2>/dev/null | awk '{print $20}')
  bt=$(awk '/^btime/{print $2}' /proc/stat)
  [[ "$st" =~ ^[0-9]+$ && "$bt" =~ ^[0-9]+$ ]] || return 1
  [ $(( bt + st / $(getconf CLK_TCK) )) -le $(( $2 + 1 )) ]
}

host_gate_live() { # $1 = kind (gpu|train|*) -> live claim files, one per line (dead ones are removed: call inside the gate only)
  local f pid t0
  for f in "$LOCKDIR"/$1[0-9]*.claim; do
    [ -e "$f" ] || continue
    read -r pid t0 _rest < "$f" 2>/dev/null || continue
    if [[ "$pid" =~ ^[0-9]+$ ]] && host_gate_pid_ok "$pid" "$t0"; then echo "$f"; else rm -f "$f"; fi
  done
}

host_gate_data_lane() { # $1 GPU $2 lane count -> the smallest free lane (rc 0) · a hold or all lanes taken: rc 1
  local k
  [ -e "$LOCKDIR/hold$1.claim" ] && return 1
  for ((k = 0; k < $2; k++)); do [ -e "$LOCKDIR/data${1}_$k.claim" ] || { echo "$k"; return 0; }; done
  return 1
}

host_gate_room_live() { # $1 GPU field $2… kinds -> rc 0 when that GPU has no live claim of those kinds
  local k pid t0
  for k in "${@:2}"; do
    [ -e "$LOCKDIR/$k$1.claim" ] || continue
    read -r pid t0 _rest < "$LOCKDIR/$k$1.claim" 2>/dev/null || continue
    [[ "$pid" =~ ^[0-9]+$ ]] && host_gate_pid_ok "$pid" "$t0" && return 1
  done
  return 0
}

host_gate_judge_room() { # judge share of RAM (GB): the reservations of live judge-only GPUs with no judge seated yet (read only)
  local f g pid t0 r s=0
  for f in "$LOCKDIR"/rsv[0-9]*.claim; do
    [ -e "$f" ] || continue
    read -r pid t0 _r _a r _d < "$f" 2>/dev/null || continue
    [[ "$pid" =~ ^[0-9]+$ ]] && host_gate_pid_ok "$pid" "$t0" || continue
    g=${f##*/rsv}; g=${g%.claim}
    host_gate_room_live "$g" gpu || continue    # a judge is seated: its RAM is already in the starting reservations / MemAvailable
    [[ "$r" =~ ^[0-9]+$ ]] && s=$((s + r))
  done
  echo "$s"
}

host_gate_claim() { # see the header
  local kind=$1 gpus=$2 run=$3 alias=$4 res=$5 ramp=$6 maxc=$7 hold=$8 fd f g used n=0 young=0 now avail t r d lbl prio=0
  case $kind in gpu) lbl=judges ;; train) lbl=trainings ;; data) lbl=data ;; *) echo "unknown claim kind '$kind' (gpu|train|data)" >&2; return 1 ;; esac
  [ "$kind" != data ] || [ "$maxc" -ge 1 ] 2>/dev/null || { echo "data claim lanes '$maxc' — >= 1" >&2; return 1; }
  [[ "$res" =~ ^[0-9]+$ && "$ramp" =~ ^[0-9]+$ && "$maxc" =~ ^[0-9]+$ && "$hold" =~ ^[0-9]+$ ]] \
    || { echo "non-integer claim args (reserve '$res' · ramp '$ramp' · maxc '$maxc' · pid '$hold')" >&2; return 1; }
  exec {fd}>"$LOCKDIR/claim.lock"; flock -x "$fd"
  now=$(date +%s)
  while IFS= read -r f; do
    read -r _p t _r _a r d < "$f" 2>/dev/null || continue    # a claim just released outside the lock
    if ! [[ "${t:-}" =~ ^[0-9]+$ && "${r:-}" =~ ^[0-9]+$ && "${d:-}" =~ ^[0-9]+$ ]]; then
      echo "claim without a reservation $(basename "$f") (no reserve_gb · ramp_s fields — an old launcher?) — closing the gate" >&2; exec {fd}>&-; return 1
    fi
    [ $((now - t)) -lt "$d" ] && young=$((young + r))
    case $(basename "$f") in "$kind"[0-9]*) n=$((n + 1)) ;; esac
  done < <(host_gate_live '*')
  if [ "$kind" != data ] && [ "$maxc" -gt 0 ] && [ "$n" -ge "$maxc" ]; then echo "concurrent $lbl $n ≥ $maxc" >&2; exec {fd}>&-; return 1; fi
  avail=$(awk '/^MemAvailable:/{print int($2 / 1048576)}' "$MEMINFO")
  [ "$kind" = gpu ] || prio=$(host_gate_judge_room)
  if [ $(( ${avail:-0} - young - prio )) -lt "$MINGB" ]; then
    echo "host free ${avail:-?} GB − reserved while starting $young GB$([ "$prio" = 0 ] || echo " − judge share $prio GB") < ${MINGB} GB" >&2; exec {fd}>&-; return 1
  fi
  for g in $gpus; do
    [ -n "$(ls "$LOCKDIR"/gpu"$g".claim "$LOCKDIR"/train"$g".claim 2>/dev/null)" ] && continue
    [ "$kind" != gpu ] && [ -e "$LOCKDIR/rsv$g.claim" ] && continue      # judge-only reservation: judges only
    if [ "$kind" = data ]; then
      d=$(host_gate_data_lane "$g" "$maxc") || continue
      echo "$hold $now $run $alias $res $ramp" > "$LOCKDIR/data${g}_$d.claim"; exec {fd}>&-; echo "${g}_$d"; return 0
    fi
    [ -n "$(ls "$LOCKDIR"/data"$g"_*.claim 2>/dev/null)" ] && continue      # a GPU used by data lanes
    if [ -e "$LOCKDIR/hold$g.claim" ]; then   # overflow hold: only that run's judges
      [ "$kind" = gpu ] && [ "$(awk '{print $3}' "$LOCKDIR/hold$g.claim" 2>/dev/null)" = "$run" ] || continue
    fi
    used=$(timeout 30 nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')   # inside the lock: a timeout keeps a hung nvidia-smi from blocking the gate forever
    [[ "$used" =~ ^[0-9]+$ ]] && [ "$used" -lt 500 ] || continue
    echo "$hold $now $run $alias $res $ramp" > "$LOCKDIR/$kind$g.claim"; exec {fd}>&-; echo "$g"; return 0
  done
  echo "no free GPU (candidates $gpus)" >&2; exec {fd}>&-; return 1
}

host_gate_bind() { # $1 kind $2 GPU $3 pid $4 run $5 alias $6 reserve_gb $7 ramp_s
  echo "$3 $(date +%s) $4 $5 $6 $7" > "$LOCKDIR/$1$2.claim"
}

host_gate_release() { # $1 kind $2 GPU
  [ "$2" = x ] || rm -f "$LOCKDIR/$1$2.claim"
}

host_gate_hold() { # $1 GPU $2 pid (the run's sidecar) $3 run — header "overflow hold"
  local g=$1 pid=$2 run=$3 fd f
  [[ "$g" =~ ^[0-9]+$ && "$pid" =~ ^[0-9]+$ && -n "$run" ]] || { echo "hold arguments: GPU '$g' · pid '$pid' · run '$run'" >&2; return 1; }
  host_gate_pid_ok "$pid" "" || { echo "hold: pid $pid is not alive (no sidecar)" >&2; return 3; }
  exec {fd}>"$LOCKDIR/claim.lock"; flock -x "$fd"
  host_gate_live '*' > /dev/null                              # clear dead claims first
  for f in "$LOCKDIR"/{train,gpu,hold,rsv}"$g".claim "$LOCKDIR"/data"$g"_*.claim; do
    [ -e "$f" ] && { echo "hold: GPU $g has a live claim $(basename "$f") ($(awk '{print $3}' "$f")) — not holding it" >&2; exec {fd}>&-; return 4; }
  done
  echo "$pid $(date +%s) $run hold 0 0" > "$LOCKDIR/hold$g.claim"; exec {fd}>&-
}

host_gate_reserve() { # $1 "<GPU…>" $2 pid (the runner loop) $3 campaign — header "judge-only reservation"
  local g f fd op pid=$2 camp=$3
  [[ "$pid" =~ ^[0-9]+$ && -n "$camp" && -n "${1// /}" ]] || { echo "reserve arguments: GPU '$1' · pid '$pid' · campaign '$camp'" >&2; return 1; }
  host_gate_pid_ok "$pid" "" || { echo "reserve: pid $pid is not alive" >&2; return 1; }
  exec {fd}>"$LOCKDIR/claim.lock"; flock -x "$fd"
  host_gate_live '*' > /dev/null                              # clear dead claims and reservations first
  for g in $1; do
    [[ "$g" =~ ^[0-9]+$ ]] || { echo "reserve: GPU '$g'" >&2; exec {fd}>&-; return 1; }
    if [ -e "$LOCKDIR/rsv$g.claim" ]; then
      read -r op _rest < "$LOCKDIR/rsv$g.claim"
      [ "$op" = "$pid" ] && continue
      echo "reserve: GPU $g is reserved by another loop (pid $op · $(awk '{print $3}' "$LOCKDIR/rsv$g.claim"))" >&2; exec {fd}>&-; return 1
    fi
    for f in "$LOCKDIR"/{train,hold}"$g".claim "$LOCKDIR"/data"$g"_*.claim; do
      [ -e "$f" ] && { echo "reserve: judge GPU $g has a live claim $(basename "$f") ($(awk '{print $3}' "$f")) — reserving after it is free" >&2; exec {fd}>&-; return 1; }
    done
  done
  for g in $1; do [ -e "$LOCKDIR/rsv$g.claim" ] || echo "$pid $(date +%s) $camp rsv $JUDGE_RSS_GB 0" > "$LOCKDIR/rsv$g.claim"; done
  exec {fd}>&-
}

host_gate_unreserve() { # $1 "<GPU…>" $2 pid — that pid's reservations only
  local g op
  for g in $1; do
    [ -e "$LOCKDIR/rsv$g.claim" ] || continue
    read -r op _rest < "$LOCKDIR/rsv$g.claim" 2>/dev/null && [ "$op" = "$2" ] && rm -f "$LOCKDIR/rsv$g.claim"
  done
  return 0
}

host_gate_held() { # $1 run -> the live hold GPUs of this run
  local f pid t0 run out=""
  for f in "$LOCKDIR"/hold[0-9]*.claim; do
    [ -e "$f" ] || continue
    read -r pid t0 run _rest < "$f" 2>/dev/null || continue
    [ "$run" = "$1" ] && host_gate_pid_ok "$pid" "$t0" && { f=${f##*/hold}; out="$out ${f%.claim}"; }
  done
  echo "${out# }"
}

host_gate_list() { # see the header; the one reader of the claim file format (the runner reads only this output)
  local f b k pid t0 rest
  for f in "$LOCKDIR"/*[0-9].claim; do
    [ -e "$f" ] || continue
    read -r pid t0 rest < "$f" 2>/dev/null || continue
    [[ "$pid" =~ ^[0-9]+$ ]] && host_gate_pid_ok "$pid" "$t0" || continue
    b=${f##*/}; b=${b%.claim}; k=${b%%[0-9]*}; b=${b#"$k"}
    echo "$k ${b%%_*} $pid $t0 $rest"
  done
}

host_gate_room_data() { # $1 "<GPU…>" $2 lane count — see the header (read only: dead claims are skipped, not removed)
  local g k f pid t0 r d young=0 now avail free prio
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || { echo "lane count '$2' — must be >= 1" >&2; return 1; }
  now=$(date +%s)
  for f in "$LOCKDIR"/*[0-9].claim; do
    [ -e "$f" ] || continue
    read -r pid t0 _r _a r d < "$f" 2>/dev/null || continue
    [[ "$pid" =~ ^[0-9]+$ ]] && host_gate_pid_ok "$pid" "$t0" || continue
    [[ "${t0:-}" =~ ^[0-9]+$ && "${r:-}" =~ ^[0-9]+$ && "${d:-}" =~ ^[0-9]+$ ]] \
      || { echo "claim without a reservation $(basename "$f") — the gate is closed" >&2; return 1; }
    [ $((now - t0)) -lt "$d" ] && young=$((young + r))
  done
  avail=$(awk '/^MemAvailable:/{print int($2 / 1048576)}' "$MEMINFO")
  prio=$(host_gate_judge_room)
  [ $(( ${avail:-0} - young - prio )) -ge "$MINGB" ] \
    || { echo "host free ${avail:-?} GB − reserved while starting $young GB$([ "$prio" = 0 ] || echo " − judge share $prio GB") < ${MINGB} GB" >&2; return 1; }
  for g in $1; do
    host_gate_room_live "$g" gpu train hold rsv || continue
    free=0
    for ((k = 0; k < $2; k++)); do host_gate_room_live "${g}_$k" data && free=$((free + 1)); done
    [ "$free" -gt 0 ] && echo "$g $free"
  done
  return 0
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then   # CLI (called by the runner)
  host_gate_env
  cmd=${1:-}; shift || true
  case $cmd in
    claim) [ $# = 8 ] || { echo "usage: claim <kind> \"<GPU…>\" <run> <alias> <reserve_gb> <ramp_s> <maxc> <pid>" >&2; exit 2; }
           host_gate_claim "$@"; exit $? ;;
    bind) [ $# = 7 ] || { echo "usage: bind <kind> <GPU> <pid> <run> <alias> <reserve_gb> <ramp_s>" >&2; exit 2; }
          host_gate_bind "$@" ;;
    release) [ $# = 2 ] || { echo "usage: release <kind> <GPU>" >&2; exit 2; }
             host_gate_release "$@" ;;
    hold) [ $# = 3 ] || { echo "usage: hold <GPU> <pid> <run>" >&2; exit 2; }
          host_gate_hold "$@"; exit $? ;;
    reserve) [ $# = 3 ] || { echo "usage: reserve \"<GPU…>\" <pid> <campaign>" >&2; exit 2; }
             host_gate_reserve "$@"; exit $? ;;
    unreserve) [ $# = 2 ] || { echo "usage: unreserve \"<GPU…>\" <pid>" >&2; exit 2; }
               host_gate_unreserve "$@"; exit $? ;;
    held) [ $# = 1 ] || { echo "usage: held <run>" >&2; exit 2; }
          host_gate_held "$@" ;;
    list) [ $# = 0 ] || { echo "usage: list" >&2; exit 2; }
          host_gate_list ;;
    room) [ $# = 3 ] && [ "$1" = data ] || { echo "usage: room data \"<GPU…>\" <lanes>" >&2; exit 2; }
          host_gate_room_data "$2" "$3"; exit $? ;;
    env) echo "LOCKDIR=$LOCKDIR MINGB=$MINGB MEMINFO=$MEMINFO JUDGE_RSS_GB=$JUDGE_RSS_GB" ;;
    *) sed -n '2,/^host_gate_env()/p' "$0" | sed '$d' | cut -c3-; exit 2 ;;
  esac
fi
