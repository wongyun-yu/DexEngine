#!/bin/bash
# Self-contained steps of scripts/stage2_ground.sh (sourced by it; they read and set its variables).

# Environment names stage 2 refuses (rc 2, even when empty): inputs that became options (--help). The one list — stage 2
# checks it before parsing, and a caller's guard checks the same list
# before they launch anything.
STAGE2_REFUSED_ENV=(DEXENGINE_BAKE_NPAR DEXENGINE_BAKE_CPU DEXENGINE_CROSS_REPAIR RETIME_SAFETY RETIME_MAX_STRETCH
                    JUDGE_EXTRA DEXENGINE_ACTIVE_SIDES CLIPS_ALLOW_NONCANDIDATE LEDGER_KEY_ALLOW_MISMATCH
                    STAGE2_RESUME_NO_RUN_ENV DEXENGINE_PLACE_RESELECT)

# Input list check: the list must lie inside the candidate set of the ledger axis. Uses CLIPS DATASET EMB HAND_DIR
# _LEDGER_KEY ALLOW_NONCAND REPO.
ground_check_clips() {
# ── Input list check ──────────────────────────────────────────────────────────────────────────────────────────
# The list must lie inside the candidate set of the ledger axis (data.candidates.axis_sets — the owner of "what is
# judged"); a list mixing clips that a rule excludes or clips of another scene path gives structurally wrong output.
# Deliberate exceptions: --allow-noncandidate. A verified list leaves a stamp <list>.canon (sha256 · dataset · emb);
# any stamp in the list's folder with the same content proves the same bytes.
_CANON_STAMP="$CLIPS.canon"
_CLIPS_SHA=$(sha256sum "$CLIPS" | cut -d" " -f1)
_canon_ok() { [ -f "$1" ] && grep -qx "sha256=$_CLIPS_SHA" "$1" && grep -qx "dataset=$DATASET" "$1" && grep -qx "emb=$EMB" "$1"; }
_CANON_HIT=""
for _s in "$_CANON_STAMP" "$(dirname "$CLIPS")"/*.canon; do _canon_ok "$_s" && { _CANON_HIT=$_s; break; }; done
if [ "$ALLOW_NONCAND" != 1 ] && [ -n "$_CANON_HIT" ]; then
  echo "[CLIPS_CANON] stamp $(basename "$_CANON_HIT") $(grep -m1 '^verified=' "$_CANON_HIT") — the candidate check was done when the list was made"
elif [ "$ALLOW_NONCAND" != 1 ]; then
  python3 - "$CLIPS" "$DATASET" "$EMB" "$HAND_DIR" "$_LEDGER_KEY" <<'PYCHK' || exit 2
import collections, pathlib, sys
from dexengine.data import candidates as TC
lst, ds, emb, hand, key = sys.argv[1:6]
want = [l.strip() for l in pathlib.Path(lst).read_text().splitlines() if l.strip()]
axes = TC.axis_sets(ds, hand, emb)   # each ledger axis has its own legitimate set
if key not in axes:
    print(f"[stage2] ledger axis {key} has no set to judge (axes: {sorted(axes)})", file=sys.stderr); raise SystemExit(2)
keep = axes[key]
bad = [c for c in want if c not in keep]
if bad:
    R = {}; TC.candidates(ds, hand, emb, reasons=R)   # drop reasons, measured only when something is dropped
    c = collections.Counter(R.get(b, "(outside the candidate list — no reason)") for b in bad)
    print(f"[stage2] {len(bad)} of {len(want)} listed clips are not candidates — {ds}/{emb}", file=sys.stderr)
    for k, v in c.most_common(8):
        print(f"           {v:>5}  {k}", file=sys.stderr)
    print("[stage2] the list owner is dexengine/data/candidates.py; for a deliberate exception pass --allow-noncandidate",
          file=sys.stderr)
    raise SystemExit(2)
print(f"[CLIPS_CANON] {ds}/{emb} axis {key}: all {len(want)} clips lie inside this axis's legitimate set")
PYCHK
  { echo "sha256=$_CLIPS_SHA"; echo "dataset=$DATASET"; echo "emb=$EMB"; echo "n=$(grep -c . "$CLIPS")";
    echo "commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)";
    echo "verified=$(hostname):$(date +%FT%T)"; } > "$_CANON_STAMP"
  echo "[CLIPS_CANON] stamp written $_CANON_STAMP"
fi
}

# Interpreter versions and $OUT/run.env. Uses OUT PYI SKIP_JUDGE RESUME_NO_RUN_ENV TAG DATASET EMB COMBO_V ARM HAND
# REPO JUDGE_EXTRA; sets _VERS.
ground_run_env() {
# The run records itself in $OUT/run.env (identity, tree, commit, interpreter versions). The judge block is written by
# the judge run only; a --skip-judge resume appends its harvest_* lines and keeps the judge block.
# Versions come from the judge interpreter itself ($PYI); a missing module is written as none (diagnostic field).
_VERS=$("$PYI" - <<'PYVERS'
import sys
out = ["python=" + sys.version.split()[0]]
for m in ("numpy", "mujoco", "torch"):
    try:
        out.append(f"{m}={__import__(m).__version__}")
    except Exception:
        out.append(m + "=none")
print(" ".join(out))
PYVERS
) || _VERS="python=? numpy=? mujoco=? torch=?"
echo "[ENV] $_VERS (judge interpreter $PYI · run.env versions=)"

_KEEP_RUN_ENV=0
if [ "$SKIP_JUDGE" = 1 ]; then
  if [ -f "$OUT/run.env" ]; then
    _KEEP_RUN_ENV=1
  elif [ "$RESUME_NO_RUN_ENV" != 1 ]; then
    echo "[stage2] ✗ --skip-judge but $OUT/run.env is missing — the judge identity is not made up at resume time." >&2
    echo "         To resume an old run knowingly, pass --resume-no-run-env (recorded in run.env)." >&2
    exit 2
  fi
fi
if [ "$_KEEP_RUN_ENV" = 1 ]; then
  echo "[RUN_ENV] judge block kept — appending the harvest lines only"
  {
    echo "harvest_started=$(date -Is)"
    echo "harvest_tree=$REPO"
    echo "harvest_branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
    echo "harvest_commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo '?')"
    echo "harvest_versions=$_VERS"
  } >> "$OUT/run.env"
else
  {
    echo "tag=$TAG"
    echo "dataset=$DATASET emb=$EMB"
    echo "combo=$COMBO_V"
    echo "robot=$ARM hand=$HAND"
    echo "tree=$REPO"
    echo "branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
    echo "commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo '?')"
    echo "started=$(date -Is)"
    echo "versions=$_VERS"
    [ "$SKIP_JUDGE" = 1 ] && echo "judge_provenance=reconstructed_by_resume(STAGE2_RESUME_NO_RUN_ENV=1)"
  } > "$OUT/run.env"
fi
_RS=$(python3 -c "from dexengine.data import rules as R; print(R.RULESET_VERSION)") || exit 2
echo "[RULESET] $_RS (datasets.RULESET_VERSION — the rule set this judging follows)"
[ "$_KEEP_RUN_ENV" = 1 ] || echo "ruleset=$_RS" >> "$OUT/run.env"
read -ra _JXA <<< "$JUDGE_EXTRA"
[ "$_KEEP_RUN_ENV" = 1 ] || echo "judge_extra=${_JXA[*]}" >> "$OUT/run.env"
}

# Re-selection (the re-selection rule, owner dexengine.placement.bake.place_check — form · checks · values): only the
# record's clips are judged with the robot spot disc + their margin. run.env place_reselect= is part of the judge block,
# so a --skip-judge resume must pass the same record (else it dies: a placement found with the margin read as one found
# without, or the reverse). A record inside the chunk is written relative to OUT (the judge and bake hosts may differ, and so
# does OUT). A resume checks the record form only (the bake list can be smaller than the judge list); the
# judge checks the record against its list. Sets PRS (judge flags) and PLACE_RESELECT (absolute). Not given = no
# change. Uses OUT PYI CLIPS SKIP_JUDGE _KEEP_RUN_ENV _LEDGER_KEY (call after ground_run_env).
ground_place_reselect() {
  PRS=()
  local rec="" env_rec out_abs cl=()
  if [ -n "$PLACE_RESELECT" ]; then
    PLACE_RESELECT=$(readlink -f "$PLACE_RESELECT") || { echo "[stage2] ✗ --place-reselect: path cannot be resolved" >&2; return 2; }
    out_abs=$(readlink -f "$OUT")
    case "$PLACE_RESELECT" in "$out_abs"/*) rec=${PLACE_RESELECT#"$out_abs"/} ;; *) rec=$PLACE_RESELECT ;; esac
    [ "$SKIP_JUDGE" = 1 ] || cl=(--clips "$CLIPS" --axis "$_LEDGER_KEY")
    "$PYI" -m dexengine.placement.bake.place_check --reselect-check "$PLACE_RESELECT" "${cl[@]}" || return 2
    [ "$_KEEP_RUN_ENV" = 1 ] || echo "place_reselect=$rec" >> "$OUT/run.env"
    PRS=(--place-reselect "$PLACE_RESELECT")
  fi
  env_rec=$(sed -n 's/^place_reselect=//p' "$OUT/run.env" 2>/dev/null | head -n 1)
  [ "$env_rec" = "$rec" ] || {
    echo "[stage2] ✗ re-selection record '${rec:-none}' != the one the judge used '${env_rec:-none}' (run.env place_reselect=)" >&2
    return 2; }
}

# Re-selection carry gate (place_check --carry-guard): a clip still owed the re-selection is not judged or baked
# without its record. Off in an isolated run (it reads the canonical ledger). $1 = clip list. Uses HAND TAG PYI
# _LEDGER_KEY PLACE_RESELECT SANDBOX.
ground_carry_guard() {
  [ -z "${SANDBOX:-}" ] || return 0
  "$PYI" -m dexengine.placement.bake.place_check --carry-guard --hand "$HAND" --axis "$_LEDGER_KEY" --clips "$1" \
    --tag "$TAG" ${PLACE_RESELECT:+--carried "$PLACE_RESELECT"}
}

# Contact label derivation (step 3b). Uses OUT COMBO_V DATASET EMB SRC_LAYER; updates NB and baked_ok.txt.
ground_contact_derive() {
# ── 3b) Contact label derivation — the canonical contact layer moved onto this tree's time axis and placement ────
# The gate's grasp criterion and the training guard read it. Count match: bake-OK trees × both hands == derived OK,
# read from the producer's machine-readable summary (--summary-json, removed before every run).
_CD_SUM=$OUT/contact_derive.summary.json
_cd_run(){ rm -f "$_CD_SUM"; python3 -m dexengine.data.contact.derive --combo "$COMBO_V" --clips-file "$OUT/baked_ok.txt" \
    --apply --in-place --dataset "$DATASET" --emb "$EMB" --src-layer "$SRC_LAYER" --summary-json "$_CD_SUM" > "$OUT/contact_derive.log" 2>&1; }
if [ "${NB:-0}" -gt 0 ]; then
  _cd_ND(){ python3 -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["run_ok"]))' "$_CD_SUM" 2>/dev/null; }
  _cd_run; _cd_rc=$?; ND=$(_cd_ND)
  # A structurally underivable minority must not stop everything: drop just those clips and go on. The producer owns
  # the kind: fails.txt = declared structural failure (→ cd_excluded → GUARD_FAIL) · errors.txt = input deficit or
  # undeclared exception (→ cd_error → BAKE_ERROR). A clip on both is a fail. Systematic errors (producer's
  # error_systematic) are not clip verdicts: stop without writing the status. Ids match whole lines (-x). A re-run
  # can fail new clips, so it loops until it converges — at most 3 rounds.
  _cd_try=0
  while { [ "$_cd_rc" -ne 0 ] || [ "${ND:-0}" -ne $((NB * 2)) ]; } && [ "$_cd_try" -lt 3 ]; do
    _cd_try=$((_cd_try + 1))
    _sysc=$(python3 -c 'import json,sys; print(int(bool(json.load(open(sys.argv[1]))["error_systematic"])))' "$_CD_SUM")
    [ "$_sysc" != 1 ] || { echo "[stage2] ✗ contact derivation errors are systematic — status not written $(grep -ah '^\[CD_ERROR_FAIL\]' "$OUT/contact_derive.log" | tail -n 1) · $OUT/contact_derive.log" >&2; exit 7; }
    _CDD=$(paths --what layer --layer contact_derive --dataset "$DATASET" --combo "$COMBO_V" --emb "$EMB") || exit 2
    _nx=0
    for _k in fails:cd_excluded errors:cd_error; do
      [ -s "$_CDD/${_k%%:*}.txt" ] || continue
      cut -f1 "$_CDD/${_k%%:*}.txt" | sort -u > "$OUT/.cd_ids"
      _n=$(grep -cxFf "$OUT/.cd_ids" "$OUT/baked_ok.txt" || true)
      [ "${_n:-0}" -gt 0 ] || continue
      cp -n "$OUT/baked_ok.txt" "$OUT/baked_ok_before_cdfail.txt"
      grep -xFf "$OUT/.cd_ids" "$OUT/baked_ok.txt" >> "$OUT/${_k#*:}.txt"   # the ledger reads it (kept out of adoption)
      { grep -vxFf "$OUT/.cd_ids" "$OUT/baked_ok.txt" || true; } > "$OUT/.bo" && mv "$OUT/.bo" "$OUT/baked_ok.txt"
      _nx=$((_nx + _n)); echo "[stage2] contact derivation ${_k%%:*} $_n clips → ${_k#*:}.txt" >&2
    done
    rm -f "$OUT/.cd_ids"
    [ "$_nx" -gt 0 ] || break   # nothing left to drop — another round changes nothing
    NB=$(grep -c . "$OUT/baked_ok.txt" || true)
    echo "[stage2] contact derivation: $_nx structural failures / errors dropped, going on (round ${_cd_try}) — bake OK left $NB · lists $_CDD" >&2
    [ "${NB:-0}" -gt 0 ] || break
    _cd_run; _cd_rc=$?; ND=$(_cd_ND)
  done
  [ "$_cd_rc" -eq 0 ] || { echo "[stage2] contact derivation failed (after ${_cd_try} drop rounds) — $OUT/contact_derive.log" >&2; exit 1; }
  [ "${ND:-0}" -eq $((NB * 2)) ] || { echo "[stage2] contact derivation count mismatch: derived OK ${ND:-none} != bake OK $NB × 2 — $OUT/contact_derive.log" >&2; exit 1; }
  echo "[stage2] contact derivation $ND/$((NB * 2)) (bake OK $NB trees × both hands)"
fi
}

# --fold: this hand's chunk ledgers of the dataset -> its final ledgers (status_fold). Off = nothing printed or done.
ground_fold() {
  [ "$FOLD" = 1 ] || return 0
  python3 dexengine/placement/status_fold.py --datasets "$DATASET" > "$OUT/fold.log" 2>&1
  local _frc=$?
  if [ "$_frc" != 0 ]; then
    echo "[stage2] ✗ fold rc=$_frc ($(grep -av '^\s*$' "$OUT/fold.log" | tail -n 1 | cut -c1-160)) — the chunk's results stand; fold again: DEXENGINE_HAND=$HAND python3 dexengine/placement/status_fold.py --datasets $DATASET" >&2
    exit 8
  fi
  echo "[FOLD] $DATASET folded for hand $HAND (status_fold rc 0 · $OUT/fold.log)"
}
# --fold takes canonical chunks that end with a status ledger only; rc 1 (with the reason) for an option it cannot take.
ground_fold_check() {
  [ "$FOLD" = 1 ] || return 0
  local _no=""
  [ -n "$SANDBOX" ] && _no+=" --sandbox"
  [ -n "$VARIANT" ] && _no+=" --variant"
  [ "$ALLOW_TAG" = 1 ] && _no+=" --allow-tag-mismatch"
  [ "$F0_ONLY" = 1 ] && _no+=" --f0-only"
  [ "$JUDGE_ONLY" = 1 ] && _no+=" --judge-only"
  [ -z "$_no" ] || { echo "[stage2] ✗ --fold cannot go with$_no (the fold takes canonical chunks that end with a status ledger)" >&2; return 1; }
}

# D37 retry inputs (rule D37 · owner dexengine.placement.judge.retry): --retry-budget <record> and --exhaust-resume
# <table> come as a pair — the stage's own D37 round (ground_d37_round) passes them to its retry chunk. The record sets
# the judge budget (budget x factor), the table the start cell per clip. Marker [RETRY_BUDGET] + run.env retry_budget= ·
# exh_resume= (ground_retry_run_env). Sets _RB · adds the budget flags to JUDGE_EXTRA. Call before ground_run_env.
ground_retry_inputs() {
  _RB=""
  [ -n "$RETRY_REC$EXH_RESUME" ] || return 0
  [ -n "$RETRY_REC" ] && [ -n "$EXH_RESUME" ] || {
    echo "[RETRY_BUDGET] ✗ --retry-budget and --exhaust-resume go together (a retry starts at the earlier stop cell)" >&2
    return 2
  }
  [ -s "$EXH_RESUME" ] || { echo "[EXH_RESUME] ✗ no table: $EXH_RESUME" >&2; return 2; }
  RETRY_REC=$(readlink -f "$RETRY_REC") && EXH_RESUME=$(readlink -f "$EXH_RESUME") || return 2
  local l ja
  l=$(python3 -m dexengine.placement.judge.retry --stage2 "$RETRY_REC") || return 2
  _RB=$(cut -f1 <<< "$l"); ja=$(cut -f2 <<< "$l")
  [ -n "$_RB" ] && [ -n "$ja" ] || { echo "[RETRY_BUDGET] ✗ empty reading of the retry record $RETRY_REC" >&2; return 2; }
  cut -f3 <<< "$l"
  JUDGE_EXTRA="${JUDGE_EXTRA:+$JUDGE_EXTRA }$ja"
}

# After ground_run_env: the retry identity joins the judge block of run.env (a --skip-judge resume keeps the block), and
# the resume table goes to the judge (kept out of judge_extra= — the path differs between machines).
ground_retry_run_env() {
  [ -n "$_RB" ] || return 0
  [ "$_KEEP_RUN_ENV" = 1 ] || { echo "retry_budget=$_RB"; echo "exh_resume=$(($(grep -c . "$EXH_RESUME") - 1))"; } >> "$OUT/run.env"
  JUDGE_EXTRA="$JUDGE_EXTRA --exhaust-resume $EXH_RESUME"
}

# D37 round — on by default, no switch (rule D37 · owner dexengine.placement.judge.retry: targets · budget · resume ·
# retry tag). After the status ledger, the UNPROVEN clips the budget stopped (kinds le2x · assembly) are judged once more
# with budget x factor from their stop cell, in the sibling chunk <retry tag> (this stage, same inputs, with
# --retry-budget --exhaust-resume); the fold takes the newer chunk per clip. A retry chunk's rows are not retried again
# (max_rounds 1): still UNPROVEN at the larger budget stays UNPROVEN (terminal, ledger_states). Marker on every run that
# reaches the status ledger: [RETRY_BUDGET] r1 exhaust_max=<budget> n=<targets> (n=0 included). Uses _S2_ARGV OUT TAG.
ground_d37_round() {
  local d=$OUT/d37 n rtag x skip=0 rc a=()
  python3 -m dexengine.placement.judge.retry plan --chunk "$OUT" --tag "$TAG" --out-dir "$d" | tee -a "$OUT/summary.txt"
  [ "${PIPESTATUS[0]}" = 0 ] || { echo "[RETRY_BUDGET] ✗ D37 targets of $OUT could not be read" >&2; return 2; }
  n=$(grep -c . "$d/targets.txt"); rtag=$(cat "$d/tag.txt")
  [ "$n" -gt 0 ] || return 0
  for x in "${_S2_ARGV[@]}"; do   # the same inputs minus the chunk identity (list · tag · folder) and one-shot stops
    if [ "$skip" = 1 ]; then skip=0; continue; fi
    case "$x" in
      --clips|--tag|--out|--retry-budget|--exhaust-resume) skip=1 ;;
      --fold|--skip-f0|--skip-judge|--resume-no-run-env|--judge-only|--f0-only|--dry-run) ;;
      *) a+=("$x") ;;
    esac
  done
  echo "[RETRY_BUDGET] round: $n clips → chunk $rtag (log $OUT/d37_round.log)"
  bash "$_S2_REPO/scripts/stage2_ground.sh" "${a[@]}" --clips "$d/targets.txt" --tag "$rtag" \
    --retry-budget "$d/retry.json" --exhaust-resume "$d/resume.tsv" > "$OUT/d37_round.log" 2>&1
  rc=$?
  grep -a '^\[RETRY_BUDGET\]\|^  \[status\] \[JUDGE_BUDGET\]' "$OUT/d37_round.log" | tail -n 3 | sed 's/^/  [d37] /'
  [ "$rc" = 0 ] || { echo "[RETRY_BUDGET] ✗ retry chunk $rtag rc=$rc — $OUT/d37_round.log" >&2; return "$rc"; }
}
