#!/usr/bin/env python3
"""Post-bake placement check, bake side — the marker line, the way after every candidate failed (re-selection ·
drop) and the re-selection record.

The verdict is place_rule.baked_check (the same body as the audit's stale reason); this module only puts it into
the bake chain and decides what follows. The judge sees the trajectory **before** the bake, the audit the baked
one — in between, cross repair can move a held object — so the bake checks once more with the audit's function.
  check       chain_fast step 7 `place_check.py --bake-check` → marker `[BAKE_PLACE_CHECK] clip=… cand=k/n verdict=…
              short_mm=…` · rc 0 pass · BAKE_REJECT_RC rejected (bake_drive tries the next candidate) · else = the
              check itself failed
  classify    a FAIL row whose every candidate was rejected → ledger.bake_fail_status reads all_rejected_state()
  re-select   when every candidate overlapped the robot spot, only that clip is judged once more with the judge's
              robot-spot disc widened by 'deepest overlap + RESELECT_PAD_M' (record RESELECT_FILE → stage2 · judge
              --place-reselect); still rejected with the margin → drop. The margin enters only the judge's search —
              the post-bake check and the audit keep margin 0.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

from dexengine.placement.scene import place_rule as PR

#: Marker tag of the chain-end check (--bake-check) · pass value · reject rc (any other non-zero rc = the check
#: itself failed).
BAKE_TAG = "[BAKE_PLACE_CHECK]"
BAKE_PASS = "-"
BAKE_REJECT_RC = 3
_CAND = re.compile(r"\d+/\d+")
_BAKE_LINE = re.compile(re.escape(BAKE_TAG) + r" clip=.* cand=\d+/\d+ verdict=(\S+) short_mm=(\d+\.\d\d)")
#: Re-selection pad: only for a clip whose every candidate failed the post-bake check on the robot spot, the judge's
#: robot-spot disc radius is widened by 'the shortfall (deepest candidate) + this'. The margin enters only the
#: judge's search — the post-bake check and the audit keep the margin-0 rule, so no tree's verdict changes.
RESELECT_PAD_M = 0.002
RESELECT_TAG = "[PLACE_RESELECT]"
#: Re-selection record — one file in the chunk output folder: written by the bake (bake_drive), read by the
#: re-selection chunk's judge (`judge --place-reselect`) and its bake (bake_drive.sh 5th argument). Header =
#: RESELECT_COLS (mm with two decimals).
RESELECT_FILE = "place_reselect.tsv"
RESELECT_COLS = ("clip", "short_mm", "margin_mm", "verdict")
#: The way after every candidate failed (all_rejected) — values printed in the RESELECT_TAG line.
RESELECT, DROP = "reselect", "drop"
#: The re-selection record a chunk carries into its judge and bake (input) — named apart from RESELECT_FILE (the
#: bake's output) because both live in the same chunk folder. Campaign lanes (local lane, axis preparation,
#: remote launch, harvest) derive it from the ledger (write_record, --chunk-record) and pass it to stage 2 as
#: `--place-reselect`.
RESELECT_IN = "place_reselect_in.tsv"


def bake_check_line(clip_name: str, cand: str, why: str, short_m: float) -> str:
    """One marker line of the chain-end check — the single owner of its form: `[BAKE_PLACE_CHECK] clip=<clip>
    cand=<k>/<n> verdict=<reject reason|-> short_mm=<depth into the robot spot, mm, two decimals>`. Clip names often
    contain spaces, so readers parse from the **end** of the line (bake_check_verdict · bake_check_short_mm)."""
    if not _CAND.fullmatch(cand or ""):
        raise PR.PlaceRuleError(f"{clip_name}: candidate label {cand!r} is not <k>/<n>")
    if not (np.isfinite(short_m) and short_m >= 0):
        raise PR.PlaceRuleError(f"{clip_name}: depth {short_m!r} is not a finite value >= 0")
    v = why or BAKE_PASS
    if v != v.strip() or any(c.isspace() for c in v) or (why and v == BAKE_PASS):
        raise PR.PlaceRuleError(
            f"{clip_name}: reject reason {why!r} cannot go on the marker line (whitespace · equals the pass value)"
        )
    return f"{BAKE_TAG} clip={clip_name} cand={cand} verdict={v} short_mm={short_m * 1000:.2f}"


def bake_check_verdict(line: str) -> str | None:
    """Marker line → verdict (BAKE_PASS = pass · else the reject reason) · None for any other line. Parses the
    positions of bake_check_line's form (starts with the tag · ends with ` cand=<k>/<n> verdict=<no spaces>`), never
    searches for words: a cut or foreign line is None (no evidence of rejection — the safe side)."""
    m = _BAKE_LINE.fullmatch(line or "")
    return m.group(1) if m else None


def bake_check_short_mm(line: str) -> float | None:
    """Marker line → depth into the robot spot (mm) · None for any other line."""
    m = _BAKE_LINE.fullmatch(line or "")
    return float(m.group(2)) if m else None


def all_rejected(clip_name: str, lines: list, had_margin_mm: float | None) -> tuple:
    """The next way of a clip whose every candidate failed the post-bake check → (way, margin mm | None, reason).
    lines = the marker line of each candidate (any non-rejection dies). had_margin_mm = the margin this chunk
    already judged the clip with (re-selection record · read_reselect), else None.
      · no margin yet and every reason is the robot-spot overlap → RESELECT · margin = deepest overlap + RESELECT_PAD_M
      · failed with a margin, or a reason is off the desk (a disc margin cannot fix it) → DROP
    Either way the chunk's end state is all_rejected_state() (the re-selection chunk writes the clip's new row)."""
    vs = [bake_check_verdict(x) for x in lines]
    if not lines or any(v in (None, BAKE_PASS) for v in vs):
        raise PR.PlaceRuleError(f"{clip_name}: these are not the lines of a clip whose every candidate failed — {vs}")
    if had_margin_mm is not None:
        return DROP, None, f"failed again in the re-selection with margin {had_margin_mm:.2f} mm"
    if not all(v.endswith(":" + PR.KEEPOUT) for v in vs):
        return DROP, None, f"a reason other than {PR.KEEPOUT} (a disc margin cannot fix it)"
    short = max(bake_check_short_mm(x) for x in lines)
    pad = RESELECT_PAD_M * 1000
    return RESELECT, round(short + pad, 2), f"shortfall {short:.2f} mm + {pad:g} mm"


def read_reselect(path) -> dict:
    """Re-selection record (RESELECT_FILE form) → {clip: margin m}. Header · column count · margin > 0 · one row per
    clip — anything else dies."""
    rows = Path(path).read_text(encoding="utf-8").splitlines()
    if not rows or tuple(rows[0].split("\t")) != RESELECT_COLS:
        raise PR.PlaceRuleError(f"{path}: re-selection record header is not {RESELECT_COLS}")
    out = {}
    for n, ln in enumerate(rows[1:], 2):
        f = ln.split("\t")
        try:
            ok = len(f) == len(RESELECT_COLS) and float(f[2]) > 0 and f[0] not in out
        except ValueError:
            ok = False
        if not ok:
            raise PR.PlaceRuleError(
                f"{path}:{n}: bad re-selection record row (column count · margin > 0 · one row per clip) — {ln!r}"
            )
        out[f[0]] = float(f[2]) / 1000.0
    return out


def add_reselect(path, clip_name: str, lines: list, margin_mm: float) -> None:
    """Append one row to the re-selection record (header first when new) — a clip already in it dies."""
    p = Path(path)
    if p.is_file() and clip_name in read_reselect(p):
        raise PR.PlaceRuleError(f"{p}: {clip_name} is already in the re-selection record")
    short = max(bake_check_short_mm(x) for x in lines)
    row = "\t".join((clip_name, f"{short:.2f}", f"{margin_mm:.2f}", bake_check_verdict(lines[-1])))
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(("" if p.stat().st_size else "\t".join(RESELECT_COLS) + "\n") + row + "\n")


def _record_lines(path) -> dict:
    """Re-selection record → {clip: its row} (the form is checked by read_reselect first)."""
    read_reselect(path)
    return {ln.split("\t")[0]: ln for ln in Path(path).read_text(encoding="utf-8").splitlines()[1:]}


def pending(hand: str, axis: str, clips, this_tag: str = "") -> dict:
    """Clips still owed the re-selection (judging once more with the margin) → their record row. Definition: the
    clip's row in the folded ledger (hand · axis) is all_rejected_state() and the record (RESELECT_FILE) of that
    row's source chunk (source_tag) holds the clip. A re-selection chunk either adopts or drops (no add_reselect),
    so a clip is owed this only once. this_tag = the chunk being baked now — re-baking the very chunk that wrote the
    record is a retry, not a re-selection, and is not counted."""
    from dexengine.core import paths as P  # noqa: PLC0415 — chunk location owner
    from dexengine.data import rules as RR  # noqa: PLC0415 — ledger file name owner
    from dexengine.placement import final_ledger as FL  # noqa: PLC0415 — folded ledger reader

    f = RR.ledger_file(axis, hand)
    rows = {P.clip_name(r["clip"]): r for r in FL.read_final_rows(f)} if f.is_file() else {}
    out = {}
    for c in clips:
        r = rows.get(c)
        if not r or r["status"] != all_rejected_state() or r["source_tag"] == this_tag:
            continue
        rec = P.run_path("run_chunk", hand, tag=r["source_tag"]) / RESELECT_FILE
        if rec.is_file() and (ln := _record_lines(rec).get(c)):
            out[c] = ln
    return out


def write_record(path, lines: dict) -> int:
    """Write record rows in the RESELECT_IN form (no rows = no file; an old file is removed). Rows written."""
    p = Path(path)
    p.unlink(missing_ok=True)
    if lines:
        p.write_text("\t".join(RESELECT_COLS) + "\n" + "".join(v + "\n" for v in lines.values()), encoding="utf-8")
        read_reselect(p)
    return len(lines)


def uncarried(pend: dict, carried: dict) -> list:
    """Clips owed the re-selection (pend — pending) that the carried record (carried — read_reselect) lacks or
    carries with another margin."""
    return sorted(
        c for c, ln in pend.items() if c not in carried or abs(carried[c] * 1000 - float(ln.split("\t")[2])) > 1e-6
    )


def _names(path, axis: str) -> list:
    """Clip list file → contract names (one reader, paths.read_clip_list · dataset = the ledger axis's)."""
    from dexengine.core import paths as P  # noqa: PLC0415
    from dexengine.data import rules as RR  # noqa: PLC0415

    return P.read_clip_list(path, dataset=RR.ledger_axis_parts(axis)[0], label="place_bake", quiet=True)


def fail_state(line: str) -> str | None:
    """If the last line of a bake FAIL row is a **reject** marker of the post-bake check, the clip's state
    (all_rejected_state — outside the vocabulary ledger_states.STATES dies), else None (pass, cut or foreign line).
    Read by ledger.bake_fail_status."""
    if bake_check_verdict(line) in (None, BAKE_PASS):
        return None
    from dexengine.placement import ledger_states as LS  # noqa: PLC0415 — the state vocabulary (no dependencies)

    if (st := all_rejected_state()) not in LS.STATES:
        raise PR.PlaceRuleError(f"all_rejected_state() = {st!r} — outside the vocabulary {list(LS.STATES)}")
    return st


def all_rejected_state() -> str:
    """The chunk end state of a clip whose **last candidate** failed the post-bake check (vocabulary =
    ledger_states.STATES), read by ledger.bake_fail_status. GUARD_FAIL: the same judge result is not baked again
    (it is not the re-bake state BAKE_ERROR). The next way (re-selection · drop) is decided by all_rejected and
    kept in the re-selection record — the re-selection chunk writes the clip's new row."""
    return "GUARD_FAIL"


def _parser():
    """The command line (one mode flag and its inputs)."""
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(
        description="Post-bake placement check (bake side) — marker line · the way after "
        "every candidate failed · re-selection record"
    )
    ap.add_argument(
        "--bake-check",
        action="store_true",
        help="chain-end check (chain_fast step 7): look at the baked tree (--tree) with the audit's "
        f"function (place_rule.baked_check) and print one {BAKE_TAG} line — rc 0 pass · "
        f"{BAKE_REJECT_RC} rejected (next candidate) · else = the check failed",
    )
    ap.add_argument("--tree", default="")
    ap.add_argument("--arm", default=None, help="arm that defines the robot spot (default: place_rule.clip's order)")
    ap.add_argument("--clip", default="", help="clip name of the marker line · record")
    ap.add_argument("--cand", default="", help="--bake-check: candidate label <k>/<n>")
    ap.add_argument(
        "--all-rejected",
        action="store_true",
        help="the next way (all_rejected) of a clip (--clip) whose every candidate failed: reads each "
        f"candidate's marker line on stdin, prints one {RESELECT_TAG} line and, on re-selection, "
        "appends a row to --reselect-out",
    )
    ap.add_argument(
        "--reselect-in",
        default="",
        help="--all-rejected: the re-selection record this chunk read (bake_drive.sh 5th argument) — "
        "a clip in it is dropped",
    )
    ap.add_argument(
        "--reselect-out",
        default="",
        help="--all-rejected: this chunk's re-selection record (<chunk>/" + RESELECT_FILE + ")",
    )
    ap.add_argument(
        "--reselect-check",
        default="",
        help="check one re-selection record (read_reselect) and that it is a subset of --clips — one marker line",
    )
    ap.add_argument("--clips", default="", help="--reselect-check · --chunk-record · --carry-guard: clip list file")
    ap.add_argument(
        "--chunk-record",
        default="",
        metavar="OUT",
        help="chunk preparation: write the record of the --clips still owed the re-selection (pending) to OUT ("
        + RESELECT_IN
        + " form) — none = no OUT file · one marker line",
    )
    ap.add_argument(
        "--carry-guard",
        action="store_true",
        help="judge · bake gate: die naming the clips of --clips that are owed the re-selection but are "
        "missing from the --carried record",
    )
    ap.add_argument(
        "--carried", default="", help="--carry-guard: the re-selection record this run carries (none = empty)"
    )
    ap.add_argument(
        "--tag", default="", help="--carry-guard: tag of the chunk baked now (its own record is not counted)"
    )
    ap.add_argument("--hand", default="", help="--chunk-record · --carry-guard: hand id")
    ap.add_argument(
        "--axis",
        default="",
        help="--chunk-record · --carry-guard · --reselect-check --clips: ledger axis (datasets.ledger_key)",
    )
    return ap


def _reselect_check(a, ap):
    """--reselect-check: the reselect table covers exactly the named clips; prints its margin summary."""
    rs = read_reselect(a.reselect_check)
    if a.clips and not a.axis:
        ap.error("--reselect-check --clips needs --axis (the dataset of the list reader)")
    names = set(_names(a.clips, a.axis)) if a.clips else None
    if not rs or (names is not None and set(rs) - names):
        raise PR.PlaceRuleError(
            f"{a.reselect_check}: the re-selection record is empty or has clips outside --clips — "
            f"{sorted(set(rs) - (names or set()))[:5]}"
        )
    mm = sorted(v * 1000 for v in rs.values())
    print(
        f"{RESELECT_TAG} record {a.reselect_check} clips {len(rs)} · margin {mm[0]:.2f}–{mm[-1]:.2f} mm — "
        f"only the judge's robot-spot disc is widened (the post-bake check and the audit keep margin 0)",
        flush=True,
    )
    return 0


def main(argv=None) -> int:
    """CLI: one of --bake-check · --all-rejected · --reselect-check · --chunk-record · --carry-guard."""
    ap = _parser()
    a = ap.parse_args(argv)
    if a.bake_check:
        if not (a.clip and a.tree):
            ap.error("--bake-check needs --tree · --clip")
        why, short = PR.baked_check(a.tree, arm=a.arm)
        print(bake_check_line(a.clip, a.cand, why, short), flush=True)
        return BAKE_REJECT_RC if why else 0
    if a.all_rejected:
        if not (a.clip and a.reselect_out):
            ap.error("--all-rejected needs --clip · --reselect-out")
        lines = [x for x in sys.stdin.read().splitlines() if x.strip()]
        had = read_reselect(a.reselect_in).get(a.clip) if a.reselect_in else None
        way, margin, why = all_rejected(a.clip, lines, None if had is None else had * 1000)
        if way == RESELECT:
            add_reselect(a.reselect_out, a.clip, lines, margin)
        print(
            f"{RESELECT_TAG} clip={a.clip} all {len(lines)} candidates failed → {way}"
            + (f" margin {margin:.2f} mm ({why}) · record {a.reselect_out}" if margin else f"({why})"),
            flush=True,
        )
        return 0
    if (a.chunk_record or a.carry_guard) and not (a.hand and a.axis and a.clips):
        ap.error("--chunk-record · --carry-guard need --hand · --axis · --clips")
    if a.chunk_record:
        n = write_record(a.chunk_record, pending(a.hand, a.axis, _names(a.clips, a.axis)))
        print(f"{RESELECT_TAG} chunk record {n} clips — {a.chunk_record if n else 'none (no file)'}", flush=True)
        return 0
    if a.carry_guard:
        pend = pending(a.hand, a.axis, _names(a.clips, a.axis), a.tag)
        bad = uncarried(pend, read_reselect(a.carried) if a.carried else {})
        if bad:
            raise PR.PlaceRuleError(
                f"{RESELECT_TAG} ✗ clips owed the re-selection (judging once more with the margin) "
                f"are not judged or baked without their record — {bad[:5]} "
                f"(pass the chunk record {RESELECT_IN} with --place-reselect)"
            )
        print(
            f"{RESELECT_TAG} judge · bake gate — {len(pend)} clips owed the re-selection · covered by the carried "
            f"record",
            flush=True,
        )
        return 0
    if a.reselect_check:
        return _reselect_check(a, ap)
    ap.error("one of --bake-check · --all-rejected · --reselect-check · --chunk-record · --carry-guard")
    return 2


if __name__ == "__main__":
    sys.exit(main())
