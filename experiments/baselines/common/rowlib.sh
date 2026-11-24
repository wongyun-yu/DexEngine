# Row library — the one owner of how one queue row runs and is judged (run_row · trim_reason · verdict · scoreline).
# Source it in the shell that runs a queue's rows. Needs: LOG RCD DEADLINE CAPBIN mark() and the counters NPASS NFAIL NSKIP NPLAN; a
# planned-skip file $SKIPF (one row name per line; may be empty).
# Verdict rule: a row cut by a time cap (the row cap, or any inner `timeout`, which goes through capbin/timeout) reads TRIM
# and FAIL even if it left a checkpoint or a result; a step that reached its declared budget (capbin/budget) is not a cut:
# the row is judged by what it left and the marker carries 'BUDGET budget_hit=1'.
kill_session() {  # $1 = session id, $2 = signal — every live process of that session (inner `timeout`s make own groups)
  local p s
  for p in /proc/[0-9]*; do
    s=$(sed 's/^.*) //' "$p/stat" 2>/dev/null | awk '{print $4}')
    [ "$s" = "$1" ] && kill -"$2" "${p#/proc/}" 2>/dev/null
  done
}
run_row() {  # name cap "ENV=… ENV=…" "command"
  local name=$1 cap=$2 envp=$3 cmd=$4 rem t0 rc sid wd dt
  # planned skip ($SKIPF — already measured / out of this job's scope): recorded as SKIP, counted apart from time skips,
  # never runs
  if grep -qxF -- "$name" "${SKIPF:?SKIPF unset}"; then
    echo "SKIP 0" > "$RCD/$name.rc"; NPLAN=$((NPLAN+1)); ROW_RC=SKIP; ROW_DT=0
    mark "SKIP $name — planned ($SKIPF)"; return 0
  fi
  rem=$(( DEADLINE - $(date +%s) )); [ "$cap" -gt "$rem" ] && cap=$rem
  if [ "$cap" -lt 300 ]; then echo "SKIP 0" > "$RCD/$name.rc"; NSKIP=$((NSKIP+1)); ROW_RC=SKIP; ROW_DT=0; mark "ROW $name SKIP no time (rem ${rem}s)"; return 0; fi
  printf '%s\n' "$cmd" > "$LOG/rows/$name.cmd"
  rm -f "$LOG/rows/$name.caps" "$LOG/rows/$name.caps.used" "$LOG/rows/$name.caps.budget" "$LOG/rows/$name.timeout"
  mark "ROW $name START cap=${cap}s"
  t0=$(date +%s)
  # every `timeout` inside the row goes through the cap shim (capbin/timeout → $name.caps on a cap hit · verdict TRIM)
  # shellcheck disable=SC2086
  setsid env $envp bash -c "export PATH=$CAPBIN:\$PATH E123_CAPLOG=$LOG/rows/$name.caps; $cmd" > "$LOG/rows/$name.out" 2>&1 < /dev/null &
  sid=$!
  ( end=$(( $(date +%s) + cap )); while [ "$(date +%s)" -lt "$end" ]; do sleep 10; done
    touch "$LOG/rows/$name.timeout"; kill_session "$sid" TERM; sleep 30; kill_session "$sid" KILL ) &
  wd=$!
  wait "$sid"; rc=$?
  kill "$wd" 2>/dev/null; wait "$wd" 2>/dev/null
  kill_session "$sid" KILL   # stragglers of a finished row (e.g. a background child) must not hold the GPU
  dt=$(( $(date +%s) - t0 ))
  [ -f "$LOG/rows/$name.timeout" ] && rc=124
  echo "$rc $dt" > "$RCD/$name.rc"
  ROW_RC=$rc; ROW_DT=$dt
}
trim_reason() {  # name → why the row was cut by a time cap (empty = not cut) — one rule for every row of every method
  local r=""
  [ "$ROW_RC" = 124 ] && r="row cap reached (run_row watchdog/deadline)"
  [ -s "$LOG/rows/$1.caps" ] && r="${r:+$r · }inner cap $(wc -l < "$LOG/rows/$1.caps")× — $(tail -1 "$LOG/rows/$1.caps" | cut -c1-160)"
  grep -Eq '(^|[ ;&|(])timeout [0-9-]' "$LOG/rows/$1.cmd" 2>/dev/null && [ ! -s "$LOG/rows/$1.caps.used" ] \
    && r="${r:+$r · }cap shim not used (a timeout ran without capbin on PATH — a cap hit would be invisible)"
  printf '%s' "$r"
}
budget_note() {  # name → the step(s) that reached their declared budget (capbin/budget — not a cap, not TRIM)
  [ -s "$LOG/rows/$1.caps.budget" ] && printf 'BUDGET budget_hit=1 %s× — %s' "$(wc -l < "$LOG/rows/$1.caps.budget")" \
    "$(tail -1 "$LOG/rows/$1.caps.budget" | cut -c1-140)"
}
verdict() {  # name ok(0/1) "detail" — see the verdict rule above
  [ "$ROW_RC" = SKIP ] && return 0
  local ok=$2 tr bn; tr=$(trim_reason "$1"); bn=$(budget_note "$1"); [ -n "$tr" ] && ok=1
  if [ "$ok" = 0 ]; then NPASS=$((NPASS+1)); mark "PASS $1 rc=$ROW_RC dt=${ROW_DT}s · ${bn:+$bn · }$3"
  else NFAIL=$((NFAIL+1)); mark "FAIL $1 rc=$ROW_RC dt=${ROW_DT}s · ${tr:+TRIM $tr · }${bn:+$bn · }$3"; fi
}
scoreline() { grep -a '^\[SCORE\]' "$LOG/rows/$1.out" | tail -1 | cut -c1-400; }
