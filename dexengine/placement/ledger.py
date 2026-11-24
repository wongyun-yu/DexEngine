#!/usr/bin/env python3
"""Per-chunk status ledger — exactly one terminal state per input clip.

States: the vocabulary and the meaning of each state belong to `ledger_states.STATES`. Bake-ledger FAIL rows are
  classified by `bake_fail_status` alone — only structural evidence of a guard-stage rejection is GUARD_FAIL
  (terminal); everything else is BAKE_ERROR (re-baked).
SOURCE_DEFECT (a defect of the input data itself, e.g. a degenerate object pose; list =
  datasets.retarget_excluded_path) wins over every other evidence: judge, bake and gate cannot catch a well-formed
  defect. `--from <old status.tsv>` restarts from an existing ledger without judge outputs.
Usage: ledger.py --input <clip list> --judge-root <judge out>… [--bake-ledger <tsv>] [--gate <gate.tsv>] --out <tsv>
Invariant: every input clip has exactly one element of STATES (sum per state == input count). A clip without a
  terminal state (MISSING) or a value outside the vocabulary leaves no ledger: rc 2 and the clip names.
Reading: every reader uses `read_status_ledger(path)` (header based) — never column positions. Shell readers call
  its CLI: ledger.py --list-clips <tsv> (one clip per row) · --count <tsv> (rows) · --xr-summary <tsv> (one summary
  line of the cross-repair residue cells, read by the stage-2 summary).
Carried cells on adopted rows (after the judge input fingerprint): cross-repair residue (two cells, --xr on|off) and
  the per-side gate post-hoc comparison (four cells; a gate table without those columns dies) — CARRY_COLUMNS.
"""

import argparse
import collections
import csv
import glob
import json
import os
import sys

from dexengine.core.errors import InputMissingError
from dexengine.core.paths import canonical_clip, clip_from_ikc, tree_name
from dexengine.placement.judge import budget as _JB
from dexengine.placement.judge.input_record import COLUMN as _JFP_COL
from dexengine.placement.judge.input_record import read_files as _JFP_READ
from dexengine.placement.ledger_cells import (
    CARRY_COLUMNS,
    XR_COLUMNS,
    carry_cells,
    gate_depth_cells,
    xr_adopted_cells,
    xr_summary,
)

# STATES and MISSING are re-exported: `from dexengine.placement.ledger import STATES` is the ledger_states object.
from dexengine.placement.ledger_states import (
    GATE_DEPTH_COLUMNS,
    GATE_DEPTH_LEDGER,
    GATE_VERDICT_STATE,
    MISSING,
    STATES,
    noconfirm_fold,
)

#: Header — main() writes this order. Old ledgers may lack the fingerprint and the carried cells.
COLUMNS = ("clip", "status", _JFP_COL) + CARRY_COLUMNS + (_JB.COLUMN,)

#: Shape of the reason of a bake-ledger FAIL row (written by bake_drive.sh): `all N candidates failed: <chain summary> |
#: last reason: <line>;<line>;`. The last lines are the last two failure/abort/violation lines of the last attempt's
#: `[stage_dp|final_guard|runner] …` log lines (that grep never carries `|`, so the last separator in a row is real).
BAKE_DETAIL_SEP = " | last reason: "
#: Head of the coarse-path DP verdict line. The final-guard verdict line form belongs to final_guard.
GUARD_TAG_DP = "[stage_dp]"
#: The bake chain's coarse-path DP stage module (its key in bake.chain.STAGES names it in the abort line).
DP_STAGE_MODULE = "dexengine.placement.coarse_path.dp"


def _dp_stage(stages: dict) -> str:
    """The one key of bake.chain.STAGES (stage name → module) whose module is the coarse-path DP."""
    dp = [k for k, v in stages.items() if v == DP_STAGE_MODULE]
    if len(dp) != 1:
        raise SystemExit(f"[status] ✗ no single coarse-path DP stage in bake.chain.STAGES: {dp}")
    return dp[0]


def bake_fail_status(reason: str) -> str:
    """Terminal state of one bake-ledger FAIL row (reason = cells from the third on) — the one owner of the
    bake-failure classification.

    GUARD_FAIL (physics rejection, terminal) only with structural evidence that a guard stage rejected it:
      1. last line = the runner's abort line of the IK stage (`chain.abort_line` — the stage returned an rc) and the
         line before it is a `[stage_dp]` line — the DP printed its verdict and returned its reject rc. A stage that
         ended by SystemExit prints a different fixed line (`fault_line`), so it is outside 1. Exceptions are
         swallowed by the resident server as rc 1 with no abort line.
      2. last line = a final-guard verdict line with the reject head (`final_guard.verdict_head` == HEAD_REJECT) —
         missing input (no IK npz) and the exception head are outside 2. The final guard runs last, outside the
         runner.
      3. last line = a rejection line of the post-bake placement check → place_check.fail_state.
    Everything else is BAKE_ERROR (re-baked, not terminal): exceptions, timeout (rc 124 — the dead chain leaves no
    abort line), dead server, OOM, failed placement record, no last reason, an abort of another stage (retime …),
    SystemExit, final-guard missing input or exception, unsupported input contract, bake contract violation.
    """
    from dexengine.placement import final_guard as FG  # noqa: PLC0415
    from dexengine.placement.bake import chain as SR  # noqa: PLC0415

    dp = _dp_stage(SR.STAGES)
    _, sep, detail = reason.rpartition(BAKE_DETAIL_SEP)
    lines = [x.strip() for x in detail.split(";") if x.strip()] if sep else []
    if len(lines) >= 2 and lines[-1] == SR.abort_line(dp) and lines[-2].startswith(GUARD_TAG_DP):
        return "GUARD_FAIL"
    if lines and FG.verdict_head(lines[-1]) == FG.HEAD_REJECT:
        return "GUARD_FAIL"
    from dexengine.placement.bake import place_check as PB  # noqa: PLC0415

    if lines and (st := PB.fail_state(lines[-1])):
        return st
    return "BAKE_ERROR"


def gate_clip_state(verdicts) -> str | None:
    """Gate verdicts of the two sides → the clip's terminal state (ledger_states.GATE_VERDICT_STATE — the earlier
    value wins) · None = the gate does not block. A value outside the vocabulary dies."""
    vs = set(verdicts)
    if vs - set(GATE_VERDICT_STATE):
        raise SystemExit(
            f"[status] ✗ gate verdict {sorted(vs - set(GATE_VERDICT_STATE))} — "
            f"outside the vocabulary {list(GATE_VERDICT_STATE)}"
        )
    return next(GATE_VERDICT_STATE[k] for k in GATE_VERDICT_STATE if k in vs)


#: Lists of clips the contact derivation dropped from the chain (written by stage 2) → terminal state. The producer
#: classified them: a declared structural defect = GUARD_FAIL · missing input or undeclared exception = BAKE_ERROR.
CONTACT_DROPS = (("cd_excluded.txt", "GUARD_FAIL"), ("cd_error.txt", "BAKE_ERROR"))


def _contact_drops(root, st: dict, key) -> None:
    """CONTACT_DROPS lists of one judge root → st[clip] = that state."""
    for name, state in CONTACT_DROPS:
        if os.path.exists(os.path.join(root, name)):
            with open(os.path.join(root, name)) as fh:
                st.update({key(ln.strip()): state for ln in fh if ln.strip()})


def read_status_ledger(path) -> list[dict]:
    """The one reader of a status ledger (`status_ledger.tsv`), by header — one {column: value} per row.

    Tab separated, no quoting (the writer's format). Empty lines are skipped. A short row has None in its missing
    cells (the caller decides whether to drop it). A header without clip and status (an empty file included) raises
    ValueError.
    """
    with open(path, encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        rows = list(rd)
        head = list(rd.fieldnames or [])
    if "clip" not in head or "status" not in head:
        raise ValueError(f"not a status ledger (clip and status columns required, header {head}): {path}")
    return rows


#: Read-CLI flags — a first argument among these reads (read_cli), anything else writes (main). Two parsers, so the
#: writer's required arguments stay required.
READ_FLAGS = ("--list-clips", "--count", "--xr-summary")


def read_cli(argv) -> int:
    """Shell CLI of `read_status_ledger`: --list-clips = one clip per row (file order) · --count = row count ·
    --xr-summary = xr_summary. A missing or malformed ledger → rc 2 (no output)."""
    ap = argparse.ArgumentParser(prog="ledger.py", description="read a status ledger")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list-clips", metavar="TSV", help="the ledger's clips — one per row, file order")
    g.add_argument("--count", metavar="TSV", help="the ledger's row count")
    g.add_argument("--xr-summary", metavar="TSV", help="one summary line of the cross-repair cells (XR_COLUMNS)")
    a = ap.parse_args(argv)
    try:
        rows = read_status_ledger(a.list_clips or a.count or a.xr_summary)
    except (OSError, ValueError) as e:
        print(f"[status] ✗ {e}", file=sys.stderr)
        return 2
    sys.stdout.write(
        f"{len(rows)}\n"
        if a.count
        else xr_summary(rows) + "\n"
        if a.xr_summary
        else "".join(f"{r['clip']}\n" for r in rows)
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    """The CLI of main (one chunk's status ledger)."""
    ap = argparse.ArgumentParser(
        epilog="read (shell CLI): ledger.py --list-clips <tsv> | --count <tsv> | --xr-summary <tsv>"
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--judge-root", nargs="*", default=[])
    ap.add_argument("--bake-ledger", default="")
    ap.add_argument("--gate", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--from", dest="from_tsv", default="", help="start from an existing status.tsv (no judge outputs)")
    ap.add_argument(
        "--dataset",
        required=True,
        help="dataset id (no default) — picks the source-defect list precompute/<ds>/diag/excluded_clips.txt",
    )
    ap.add_argument(
        "--excluded",
        default="",
        help="source-defect clip list (<clip><TAB><reason>) → SOURCE_DEFECT. Empty = the dataset's ledger "
        "(precompute/<dataset>/diag/excluded_clips.txt ∪ configs/datasets/<dataset>/excluded_clips.txt)",
    )
    ap.add_argument(
        "--xr",
        choices=("on", "off"),
        default="",
        help="cross-repair (bake chain xr stage) on/off in this run — required when this run has "
        "adopted rows (no default)",
    )
    ap.add_argument("--tree-root", default="", help="--xr on: the baked tree folder (stage 2's TROOT)")
    ap.add_argument(
        "--coll-preset",
        default="",
        help="--xr on: training preset (COLL_PRESET of scripts/_rl_recipe.sh, passed by stage 2) → the "
        "cross-repair sets on = collision_presets.xr_sets_on",
    )
    return ap


def _fold_judge_roots(a, st: dict, fresh: set, _K) -> None:
    """Adoption statements of this run's judge roots (plans2 · passed.txt · ikc) and the contact-derivation drops."""
    for R in a.judge_root:
        for f in glob.glob(R + "/**/plans2_*.jsonl", recursive=True):
            with open(f) as fh:  # adoption statement — only empty lines are skipped; a broken line dies
                adopted = {_K(json.loads(ln)["clip"]) for ln in fh if ln.strip()}
            st.update(dict.fromkeys(adopted, "ADOPTED"))
            fresh |= adopted
        # The adoption statement is `passed.txt` (clips with a placement); `ikc` is only the judge's cache. They are
        # equal on the normal path; they differ only for placements inherited without a judge (segment split), where
        # ikc is empty. Contact-derivation drops are set FIRST: the assignments below are setdefault, so these win
        # (otherwise an excluded clip would become adopted from the judge statement alone).
        _contact_drops(R, st, _K)
        _pf = os.path.join(R, "passed.txt")
        if os.path.exists(_pf):
            with open(_pf) as fin:
                for line in fin:
                    if line.strip():
                        st.setdefault(_K(line.strip()), "ADOPTED")
                        fresh.add(_K(line.strip()))
        _ikc = {_K(clip_from_ikc(f)) for f in glob.glob(R + "/ikc/*_right.npz")}
        fresh |= _ikc
        for c in _ikc:
            st.setdefault(c, "ADOPTED")
        if _ikc and os.path.exists(_pf):  # statement ↔ cache: report a difference
            with open(_pf) as fin:
                _ps = {_K(line.strip()) for line in fin if line.strip()}
            if _ikc != _ps:
                print(
                    f"[warning] passed.txt and ikc differ: passed-ikc {len(_ps - _ikc)} · ikc-passed "
                    f"{len(_ikc - _ps)} ({R})",
                    file=sys.stderr,
                )


def _fold_noconfirm(a, st: dict, _K) -> None:
    """The judge's no-confirm rows (noconfirm_fold) of every judge root."""
    for R in a.judge_root:
        for f in glob.glob(R + "/**/noconfirm_*.tsv", recursive=True):
            with open(f) as fin:
                for n, line in enumerate(fin, 1):
                    p = line.rstrip("\n").split("\t")
                    if len(p) < 2:
                        continue
                    k = _K(p[0])
                    try:
                        v = noconfirm_fold(st.get(k), p[1], p[2] if len(p) > 2 else "")
                    except ValueError as e:
                        sys.exit(f"[status] ✗ {f}:{n}: {e}")
                    if v:
                        st[k] = v


def _fold_bake_ledger(a, st: dict, _K) -> None:
    """Bake-ledger FAIL rows → GUARD_FAIL · BAKE_ERROR (bake_fail_status)."""
    if a.bake_ledger and os.path.exists(a.bake_ledger):
        _kinds = collections.Counter()
        with open(a.bake_ledger) as fin:
            for line in fin:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2 and p[1] == "FAIL":
                    st[_K(p[0])] = bake_fail_status("\t".join(p[2:]))
                    _kinds[st[_K(p[0])]] += 1
        # marker: the classification rule was applied to this ledger — terminal (GUARD_FAIL) · re-bake (BAKE_ERROR)
        print(
            f"[BAKE_FAILKIND] FAIL {sum(_kinds.values())} → GUARD_FAIL {_kinds['GUARD_FAIL']} · "
            f"BAKE_ERROR {_kinds['BAKE_ERROR']} (rule ledger.bake_fail_status)"
        )


def _fold_gate(a, st: dict, gdep, _K):
    """The gate table → per-clip gate state; returns the side depth cells (unchanged when no table)."""
    if a.gate and os.path.exists(a.gate):
        side, gdep = collections.defaultdict(dict), collections.defaultdict(dict)
        with open(a.gate) as fin:
            _gl = fin.read().splitlines()
        _gh = _gl[0].split("\t") if _gl else []
        # columns are found by header name (ledger_states.GATE_COLUMNS)
        if not {"clip", "side", "verdict", *GATE_DEPTH_COLUMNS} <= set(_gh):
            raise InputMissingError(
                f"[status] ✗ the gate table header lacks the clip·side·verdict·{'·'.join(GATE_DEPTH_COLUMNS)}"
                f" columns — {a.gate}: {_gh}"
            )
        _ic, _is, _iv = _gh.index("clip"), _gh.index("side"), _gh.index("verdict")
        for line in _gl[1:]:
            c = line.split("\t")
            side[_K(c[_ic])][c[_is]] = c[_iv]
            gdep[_K(c[_ic])][c[_is]] = {k: c[_gh.index(k)] for k in GATE_DEPTH_COLUMNS}
        _gerr = 0
        for c, v in side.items():
            if len(v) == 2 and (_gs := gate_clip_state(v.values())):
                st[c] = _gs
                _gerr += _gs == "BAKE_ERROR"
        print(
            f"[GATE_VERDICT] gate ERROR (undeclared measurement exceptions only) → BAKE_ERROR {_gerr} "
            "(vocabulary ledger_states.GATE_VERDICT_STATE)"
        )
        # an adopted candidate without both side rows in the gate table did not go through the gate — die
        _ng = sorted(c for c, v in st.items() if v == "ADOPTED" and len(side.get(c, {})) < 2)
        if _ng:
            sys.exit(
                f"[status] ✗ {len(_ng)} adopted candidates without both side rows in the gate table "
                f"(e.g. {_ng[:3]}) — {a.gate}"
            )
    elif a.gate:
        # --gate given but the table is missing: adopted candidates would become ADOPTED without a gate
        _na = sum(1 for v in st.values() if v == "ADOPTED")
        if _na:
            sys.exit(
                f"[status] ✗ gate table {a.gate} is missing but there are {_na} adopted candidates — not "
                "writing ADOPTED without a gate"
            )
    return gdep


def _fold_f0_held(a, st: dict, _K) -> None:
    """f0 held clips (stage2 f0_held.txt: the object repair refused a side) → HELD. They were not judged; this run's f0
    verdict replaces a state carried over by --from. Source defects, folded after, still win."""
    n = 0
    for R in a.judge_root:
        p = os.path.join(R, "f0_held.txt")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fin:
                for line in fin:
                    k = _K(line.split("\t")[0].strip())
                    if k:
                        st[k] = "HELD"
                        n += 1
    if n:
        print(f"f0 held → HELD {n}")


def _fold_source_defects(a, st: dict, clips: list, _K) -> None:
    """f0 data defects and the excluded-clip list → SOURCE_DEFECT."""
    # Clips dropped at the f0 stage for a DECLARED data defect (dexengine.retarget.f0.prejudge DATA_DEFECT —
    # SourceDefectError) never reach the judge and carry no evidence; no reference can be built from such an input, so
    # they end as SOURCE_DEFECT (reason = the DATA_DEFECT row of <OUT>/f0.log; otherwise MISSING → rc 2). f0_failed.txt
    # belongs to stage 2 and holds data defects only; run failures go to <OUT>/f0_run_failed.txt and kill the chunk.
    n_f0 = 0
    for R in a.judge_root:
        p = os.path.join(R, "f0_failed.txt")
        if os.path.exists(p):
            with open(p) as fin:
                for line in fin:
                    k = _K(line.split("\t")[0].strip())
                    if k and k not in st:
                        st[k] = "SOURCE_DEFECT"
                        n_f0 += 1
    if n_f0:
        print(f"f0 failure → SOURCE_DEFECT {n_f0}")
    # source defects win over every other evidence (well-formed, so judge, bake and gate pass them)
    if a.excluded:  # an explicit list (missing = empty list)
        from dexengine.data.rules import excluded_from_file  # noqa: PLC0415

        listed = excluded_from_file(a.excluded) if os.path.exists(a.excluded) else {}
    else:  # the dataset's ledger: data root ∪ repository (rules.retarget_excluded)
        from dexengine.data.rules import retarget_excluded  # noqa: PLC0415

        listed = retarget_excluded(a.dataset)
    excluded = {_K(c): why for c, why in listed.items()}
    for c in clips:
        if c in excluded:
            st[c] = "SOURCE_DEFECT"


def _check_terminal(a, st: dict, clips: list) -> None:
    """Every clip has exactly one state from the vocabulary — else exit 2 before writing."""
    unknown = [c for c in clips if c not in st]
    cnt = collections.Counter(st.get(c, MISSING) for c in clips)
    print(" · ".join(f"{k} {v}" for k, v in sorted(cnt.items())) + f" · sum {sum(cnt.values())}/{len(clips)}")
    if unknown:
        # The file is written only after this check: a lane that exits 2 must not leave a ledger with MISSING rows
        # for the fold to aggregate. An invariant violation leaves no ledger — the old one stands until repaired.
        print(
            f"[invariant violation] {len(unknown)} clips without a terminal state: "
            + ", ".join(unknown[:5])
            + f" — not writing {a.out}",
            file=sys.stderr,
        )
        sys.exit(2)
    # a value outside the vocabulary (a typo, an unregistered new state) must not reach the ledger
    alien = [c for c in clips if st[c] not in STATES]
    if alien:
        print(
            f"[invariant violation] {len(alien)} states outside the vocabulary (ledger_states.STATES): "
            + ", ".join(f"{c}={st[c]}" for c in alien[:5])
            + f" — not writing {a.out}",
            file=sys.stderr,
        )
        sys.exit(2)


def _write_ledger(a, clips: list, st: dict, jfp: dict, xr: dict, dep: dict, bud: dict) -> None:
    """Write the ledger rows (temp file, then replace)."""
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for c in clips:
            _x = xr.get(c, ("", "")) if st[c] == "ADOPTED" else ("", "")
            _d = dep.get(c, ("",) * len(GATE_DEPTH_LEDGER))  # adopted rows only (gate_depth_cells)
            f.write("\t".join((c, st[c], jfp.get(c, ""), *_x, *_d, bud.get(c, ""))) + "\n")
    os.replace(tmp, a.out)


def main() -> int:
    """Write the status ledger of one chunk (module doc)."""
    ap = _parser()
    a = ap.parse_args()

    def _K(n):  # names are normalized with this ledger's dataset table (canonical_clip has no dataset default here)
        return canonical_clip(n, a.dataset)

    if not a.judge_root and not a.from_tsv:
        ap.error("--judge-root or --from is required")
    with open(a.input) as fin:
        clips = [_K(line.rstrip("\n")) for line in fin if line.strip()]
    st = {}
    # Judge input fingerprint — the judge writes `<FILE_PREFIX><combo>.tsv` (clip · fingerprint) per clip; it moves
    # to the third column and on to the folded ledger. Rows that never went through the judge (f0 failure, source
    # defect, inherited segment placement) have an empty cell (= "unknown" in comparisons).
    jfp, bud = {}, {}  # bud = judge budget cells — --from's old cells, then this run's records win
    xr = {}  # cross-repair cells moved from --from (this run's adopted rows are measured below)
    dep, gdep = {}, None  # gate comparison cells — --from's old cells · the gate table's side cells
    fresh = set()  # clips this run's judge roots state as adopted (their cross-repair cells are new)
    if a.from_tsv:
        for r in read_status_ledger(a.from_tsv):
            if r["clip"] and r["status"] is not None:
                st[_K(r["clip"])] = r["status"]
                if r.get(_JFP_COL):
                    jfp[_K(r["clip"])] = r[_JFP_COL]
                bud[_K(r["clip"])] = _JB.cell_of(r)
                _c = carry_cells(r)
                xr[_K(r["clip"])], dep[_K(r["clip"])] = _c[: len(XR_COLUMNS)], _c[len(XR_COLUMNS) :]
    _fold_judge_roots(a, st, fresh, _K)
    jfp.update(_JFP_READ(a.judge_root, key=_K))  # the new judge's fingerprint wins
    bud.update(_JB.read_files(a.judge_root, key=_K))
    _fold_noconfirm(a, st, _K)
    _fold_bake_ledger(a, st, _K)
    gdep = _fold_gate(a, st, gdep, _K)
    _fold_f0_held(a, st, _K)
    _fold_source_defects(a, st, clips, _K)
    _check_terminal(a, st, clips)
    _nj = sum(1 for c in clips if jfp.get(c))
    print(
        f"[JUDGE_FP] judge input fingerprint on {_nj}/{len(clips)} rows (none = not judged or judged before "
        "fingerprints)"
    )
    # Cross-repair residue cells: this run's adopted rows are measured from --xr · adopted rows only from --from keep
    # their old cells · non-adopted rows are empty.
    _new = [c for c in clips if st[c] == "ADOPTED" and c in fresh]

    def _sets_on(preset):
        from dexengine.rl.tasks.scene import collision_presets  # noqa: PLC0415

        return collision_presets.xr_sets_on(preset)

    xr.update(xr_adopted_cells(_new, a.xr, a.tree_root, a.coll_preset, _sets_on, tree_name))
    dep = gate_depth_cells([c for c in clips if st[c] == "ADOPTED"], fresh, gdep, dep)
    _write_ledger(a, clips, st, jfp, xr, dep, bud)
    return 0


if __name__ == "__main__":
    sys.exit(read_cli(sys.argv[1:]) if sys.argv[1:2] and sys.argv[1] in READ_FLAGS else main())
