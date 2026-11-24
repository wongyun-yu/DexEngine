#!/usr/bin/env python3
"""Fold the chunk status ledgers of one hand into the final per-clip ledgers `ledger/status_final_<axis>.tsv` of the
hand root (columns LEDGER_COLS: clip · status · source_tag · note · judge input fingerprint (third column of the chunk
ledger) · the cells adopted rows carry (ledger_cells.CARRY_COLUMNS — cross-repair residue, per-side gate
comparison) · judge budget). Place: `datasets.ledger_file(<axis>, <hand>)` under `paths.run_path("run_ledger",
<hand>)`. The RL training entry reads these files as its roster (train_ready_util.ledger_roster).

Rules:
  1. Chunks = `G1*_<axis>_NN/` with a `status_ledger.tsv` under this hand's chunk root
     (`paths.run_path("run_chunks", <hand>)`). Test tags (`datasets.TEST_TAG_PREFIXES`) are excluded and counted.
  2. Generation guard: chunk ledgers finished before a base regeneration that changed the dataset's qpos
     (fold_rules.REGEN_EVENTS) were judged on the old base and are excluded.
  3. Final row per clip = the chunk with the latest `status_ledger.tsv` modification time (the latest attempt).
  4. Rows that differ from the existing file are reported (`--check` writes nothing and returns rc 1 on a
     difference).
  5. Scope = the campaign scope (datasets.datasets_scope — empty `--datasets` = all minus the campaign exclusions).
     Datasets outside it are not computed — `[FOLD_SCOPE] excluded <ds>(<reason>)` per exclusion.
  Rows the current rules retire (fold_rules) go to the per-hand retired ledger, a cumulative history
  (final_ledger.merge_retired).
rc: 0 = folded (warnings are ⚠ lines) · 2 = refused · anything else (a Python death 1 included) = failure. Callers
stop on non-zero.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import re
import sys
import time
from pathlib import Path

from dexengine.core import paths as P
from dexengine.core.errors import InputMissingError, input_missing
from dexengine.data import candidates as TC
from dexengine.data import rules as RR
from dexengine.placement import final_ledger as FL
from dexengine.placement import fold_rules as _FR
from dexengine.placement.fold_rules import (  # noqa: F401
    JUDGE_INPUT_EVENTS,
    REGEN_EVENTS,
    REGEN_EVENTS_UNITS,
    RETIRED,
    RS_SCOPE,
    WINDOW_RETIRE,
    _dz_policy_for,
    _judge_markers,
    _live_rows,
    _old_window_reason,
    _ruleset_scope,
    _ruleset_stamp,
    _stale_judge_reason,
    ruleset_scope_notes,
    split_verdicts,
    window_note,
)
from dexengine.placement.judge import budget as _JB
from dexengine.placement.judge.input_record import COLUMN as _JFP_COLUMN
from dexengine.placement.judge.input_record import err_fields as _jfp_err_fields
from dexengine.placement.ledger import read_status_ledger
from dexengine.placement.ledger_cells import CARRY_COLUMNS, carry_cells
from dexengine.placement.status_fold_scope import axis_keys


def _K(name: str, dataset: str = "taco") -> str:
    """The joinable form of a clip name (paths.canonical_clip — idempotent)."""
    return P.canonical_clip(name, dataset)


_TEST_PFX = tuple(RR.TEST_TAG_PREFIXES.values())  # test tags (regression, canary) are not folded
_EXCLUDED_TEST: set[str] = set()

ACTIVE_COMBO = RR.combo()
RR_HAND = RR.active_hand()
#: The ledger axes of the scope — set by main() (status_fold_scope.axis_keys(scope)); not computed at import.
DATASETS: tuple[str, ...] = ()

#: Final ledger columns — fifth = judge input fingerprint (judge.input_record.COLUMN) · then the cells adopted rows
#: carry (ledger_cells.CARRY_COLUMNS — empty in old chunks) · last = judge budget (judge.budget.COLUMN — every row;
#: empty in old chunks = no record).
LEDGER_COLS = ("clip", "status", "source_tag", "note", _JFP_COLUMN) + CARRY_COLUMNS + (_JB.COLUMN,)
#: Stamp-note distribution of the folded rows (current / ruleset:equivalent(...)) — main() prints it.
STAMP_CENSUS: dict[str, int] = {}


def _chunks_root() -> Path:
    """This hand's chunk root — another hand's chunks live in another folder. Another arm combo of the same hand is
    told apart by chunk_combo (consolidate)."""
    return P.run_path("run_chunks", RR_HAND)


def _fold_chunk_rows(ds, _rs, d, t, led, seen, rows, notes):
    """Fold one chunk ledger's live rows into rows/seen (stale rows by the unit cut, superseded wins counted into
    notes)."""
    _note = RR.ruleset_note(ds.split("-")[0], _rs)
    ucut = REGEN_EVENTS_UNITS.get(ds.split("-")[0], 0.0)
    # Which unit generation a chunk judged is told by its list stamp (clips.txt.canon `verified=`) — the ledger
    # mtime is when judging ended and would take an old-list chunk finished after a re-split as new. Without a
    # stamp the mtime is used.
    _stamp = d / "clips.txt.canon"
    t_list = t
    if _stamp.is_file():  # a stamp that does not parse dies (never the ledger mtime)
        for _ln in _stamp.read_text().splitlines():
            if _ln.startswith("verified=") and ":" in _ln:
                t_list = _dt.datetime.fromisoformat(_ln.split(":", 1)[1].strip()).timestamp()
    # Stale judge input (JUDGE_INPUT_EVENTS): chunk markers vs the current owner values.
    _jds, _jemb, _jv = RR.ledger_axis_parts(ds)  # every axis comes from ledger_key — an unknown one dies
    _scene = "aerial" if _jv == "_aerial" else getattr(RR.DATASET_PROFILE.get(_jds), "scene", "table")
    _dz_pol = _dz_policy_for(_jds, _scene)
    _dzj, _act = _judge_markers(d)
    _stale_here = _old_win = 0
    for _row in _live_rows(led, ds, d.name, _ruleset_scope(ds, _rs)):
        r = (_row["clip"], _row["status"], _row.get(_JFP_COLUMN) or "", *carry_cells(_row), _JB.cell_of(_row))
        # an old-window verdict is not folded but kept in the retired ledger (_old_window_reason)
        if _why_w := _old_window_reason(r[0], t_list, ucut, (_jds, _jemb, _scene, r[2], r[1]), ds):
            RETIRED.append([ds, r[0], r[1], d.name, _why_w])
            _old_win += 1
            continue
        if r[1] != "ADOPTED":
            _why = _stale_judge_reason(_jds, _jemb, _scene, r[0], r[1], t, _dzj, _act, _dz_pol)
            if _why:
                RETIRED.append([ds, r[0], r[1], d.name, _why])
                _stale_here += 1
                continue
        _ck = _K(r[0], _jds)  # the dataset part, not the ledger axis key
        seen.setdefault(_ck, []).append((t, d.name, r[1], r[2], r[-1]))  # split_verdicts
        if _ck not in rows or t > rows[_ck][0]:
            rows[_ck] = (t, r[1], d.name, _note, *r[2:])
    if _stale_here:
        notes.append(
            f"[status_final] {d.name}: {_stale_here} rejection rows of an old judge input retired (JUDGE_INPUT_EVENTS)"
        )
    if _old_win:
        notes.append(
            f"[status_final] {d.name}: {_old_win} old-window verdict rows retired (window replacement · "
            "re-split — judged again)"
        )


def _drop_rule_gone(ds, out, notes):
    """The axis's legitimate clip set (candidates.axis_sets), the hand carry fold, then rows of clips the candidate rule
    no longer selects are retired (reasons in notes)."""
    legit = None
    _ds, _emb, _v = RR.ledger_axis_parts(ds)
    _hand = RR.hand_dir()  # axis_sets takes the hand spec directory (RR_HAND is the hand id)
    try:
        legit = TC.axis_sets(_ds, _hand, _emb).get(ds)
    except TC.BaseLayerMissing as e:
        if out:
            raise InputMissingError(
                f"[status_final] ✗ {ds}: {len(out)} folded rows but no base layer ({e})"
                " — cannot apply the current rules"
            ) from None
        notes.append(f"[status_final] {ds}: no base layer · 0 rows — rule cleanup not applicable")
    if legit is None and _ds in RR.DATASET_PROFILE and not (_v and ds not in TC.axis_sets(_ds, _hand, _emb)):
        notes.append(f"[status_final] {ds}: this axis has no set — rule cleanup skipped")
    _FR.fold_hand_carry(RR_HAND, ds, out, legit, notes, LEDGER_COLS[1:], _K)  # only a hand with a declaration
    if legit is not None:
        R: dict = {}
        TC.candidates(_ds, _hand, _emb, reasons=R)
        gone = sorted(c for c in out if c not in legit)
        if gone:
            for c in gone:
                RETIRED.append([ds, c, out[c][0], out[c][1], R.get(c, "outside the axis set (no reason)")])
            _cnt: dict = {}
            for c in gone:
                _cnt[R.get(c, "outside the axis set")] = _cnt.get(R.get(c, "outside the axis set"), 0) + 1
            notes.append(
                f"[status_final] {ds}: {len(gone)} rows outside the current rules retired — "
                + " · ".join(f"{k} {v}" for k, v in sorted(_cnt.items(), key=lambda kv: -kv[1])[:6])
            )
            for c in gone:
                out.pop(c)


def consolidate(ds: str) -> tuple[dict[str, tuple[str, ...]], list[str]]:
    """Fold every chunk ledger of axis `ds` → ({clip: LEDGER_COLS[1:] row}, notes)."""
    cut = REGEN_EVENTS.get(ds.split("-")[0], 0.0)
    notes = []
    rows, seen = {}, {}  # clip → winning row · clip → all rows (split_verdicts)
    for d in sorted(_chunks_root().glob(f"G1*_{ds}_[0-9]*")):
        if not re.fullmatch(rf"G1[a-z]*_{re.escape(ds)}_\d+", d.name):
            continue
        if d.name.startswith(_TEST_PFX):
            _EXCLUDED_TEST.add(d.name)
            continue
        led = d / "status_ledger.tsv"
        if not led.is_file():
            continue
        # The chunk ledger has no hand axis: check the chunk's combo identity (datasets.chunk_combo — measured from
        # the run's own outputs). Chunks of unknown identity were refused by main() before folding.
        _combo = RR.chunk_combo(d)
        if _combo != ACTIVE_COMBO:
            notes.append(f"[status_final] {d.name}: another hand's chunk excluded (combo {_combo} ≠ {ACTIVE_COMBO})")
            continue
        t = led.stat().st_mtime
        if t < cut:
            notes.append(
                f"[status_final] {d.name}: old-base chunk excluded (ledger {_fmt(t)} < regeneration {_fmt(cut)})"
            )
            continue
        # Ruleset retirement (datasets.ruleset_retire_reason): in a dataset whose whole judge input changed, a chunk
        # judged under a stamp other than RULESET_VERSION (old or none, from run.env) is not folded; its rows go to
        # the retired ledger.
        _rs = _ruleset_stamp(d)
        if _why_rs := RR.ruleset_retire_reason(ds.split("-")[0], _rs):
            RETIRED.extend(
                [ds, _r["clip"], _r["status"], d.name, _why_rs]
                for _r in read_status_ledger(led)
                if _r["status"] is not None
            )
            notes.append(
                f"[status_final] {d.name}: ruleset stamp {_rs or '(none)'} ≠ {RR.RULESET_VERSION} — chunk "
                f"retired ({_why_rs})"
            )
            continue
        # A non-current stamp that is not retired keeps its state and the row says why (datasets.RULESET_EQUIVALENT
        # owns the reasons); a stamp missing from that table dies there.
        _fold_chunk_rows(ds, _rs, d, t, led, seen, rows, notes)
    split_verdicts(ds, seen)
    out = {c: v[1:] for c, v in rows.items()}  # (state, tag, note, fingerprint, *carried, budget)
    # Rows outside this axis's legitimate set under the current rules (candidates.axis_sets) are retired; the
    # candidate reasons (candidates.candidates) say why, and the rows stay in the retired ledger.
    _drop_rule_gone(ds, out, notes)
    return out, notes


def adopted_input_errors(rows: dict) -> list[tuple[str, list[str]]]:
    """Adopted rows whose judge input fingerprint has an `error:` component [(clip, [cell=value …])] — the judge did
    not prove its own input. rows = consolidate's output."""
    return [(c, bad) for c, v in sorted(rows.items()) if v[0] == "ADOPTED" and (bad := _jfp_err_fields(v[3]))]


def _fmt(t: float) -> str:
    import datetime as _dt  # noqa: PLC0415

    return _dt.datetime.fromtimestamp(t).strftime("%m-%d %H:%M")


def _key_dataset(key: str) -> str | None:
    """The dataset a tag key names — the inverse of ledger_key (ledger_axis_parts) when it applies, else the key's
    first part when that is a dataset name, else None."""
    try:
        return RR.ledger_axis_parts(key)[0]
    except KeyError:
        head = key.split("-")[0]
        return head if head in RR.DATASET_PROFILE else None


def _in_scope(key: str, scope: set) -> bool:
    """Whether a tag key's chunk is in scope — a key of unknown dataset counts as inside (not skipped)."""
    ds = _key_dataset(key)
    return ds is None or ds in scope


def orphan_chunks(scope: set) -> list[str]:
    """Chunks no ledger axis folds — the tag's <key> is not in DATASETS (a variant axis tagged with the dataset name
    does not match `G1*_<key>_NN` and its results would be dropped silently). Only chunks of datasets in `scope`."""
    out = []
    for d in sorted(_chunks_root().glob("G1*_*_[0-9]*")):
        m = re.fullmatch(r"G1[a-z]*_(.+)_\d+", d.name)
        if d.name.startswith(_TEST_PFX):
            continue
        if not m or m.group(1) in DATASETS:
            continue
        if not _in_scope(m.group(1), scope):
            continue
        # another hand's chunk is not "outside this hand's axes"; unknown identities were refused by main()
        if (d / "status_ledger.tsv").is_file() and RR.chunk_combo(d) == ACTIVE_COMBO:
            out.append(d.name)
    return out


def unidentified_chunks(scope: set) -> list[str]:
    """Judge chunks whose identity (combo) cannot be determined — main() refuses to fold on them."""
    out = []
    for d in sorted(_chunks_root().glob("G1*_*_[0-9]*")):
        m = re.fullmatch(r"G1[a-z]*_(.+)_\d+", d.name)
        if not m or d.name.startswith(_TEST_PFX) or not (d / "status_ledger.tsv").is_file():
            continue
        if not _in_scope(m.group(1), scope):
            continue
        if not RR.chunk_combo(d):
            out.append(d.name)
    return out


#: Lock file of one hand's fold (in its ledger folder). Without it two overlapping folds could interleave their
#: writes, and the fold that read the chunks first could overwrite a newer ledger. The lock is taken before reading.
FOLD_LOCK = ".status_final.lock"
_FOLD_LOCK_FD: list[int] = []


def _fold_lock(ledger_dir: Path) -> None:
    """Exclusive, blocking lock of this hand's fold, held until the process ends. Marker `[FOLD_LOCK]`."""
    import fcntl  # noqa: PLC0415

    ledger_dir.mkdir(parents=True, exist_ok=True)
    fd, t0 = os.open(ledger_dir / FOLD_LOCK, os.O_CREAT | os.O_RDWR, 0o644), time.monotonic()
    fcntl.flock(fd, fcntl.LOCK_EX)
    _FOLD_LOCK_FD.append(fd)
    print(f"[FOLD_LOCK] {RR_HAND} held · waited {time.monotonic() - t0:.0f} s ({ledger_dir / FOLD_LOCK})", flush=True)


def _report_excluded_test_tags() -> None:
    print(
        f"[fold] test tags excluded {len(_EXCLUDED_TEST)}: {' '.join(sorted(_EXCLUDED_TEST)) or '-'}  "
        "(datasets.TEST_TAG_PREFIXES)",
        flush=True,
    )


def _parser():
    """Command line: --check, --datasets, --combo-switch."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument(
        "--datasets",
        default="",
        help="dataset names (comma separated) — fold every ledger axis of those datasets. Empty = the "
        "campaign scope (datasets.datasets_scope · exclusions printed as [FOLD_SCOPE]). An unknown "
        "name dies",
    )
    ap.add_argument(
        "--combo-switch",
        action="store_true",
        help="make this ledger belong to ANOTHER combo (the existing content is lost) — explicit choice",
    )
    a = ap.parse_args()
    return a


def _fold_axis(ds, live):
    """Fold one ledger axis: consolidate, refuse adoptions with an error input, print the notes and the change summary.
    Returns the new rows, the live row identities (grown by this axis), the ledger file and the (state, tag) diff."""
    new, notes = consolidate(ds)
    if bad := adopted_input_errors(new):  # die before writing
        raise SystemExit(
            f"[status_final] ✗ {ds}: {len(bad)} adopted rows with an error component in their judge "
            f"input fingerprint — not folding adoptions whose judge did not prove its input · e.g. "
            f"{bad[:3]} · fix = judge those clips again, then rebuild the chunk ledger from that "
            "judge's fingerprint"
        )
    for n in notes + [window_note(ds)] + ruleset_scope_notes(ds):
        print(n)
    live |= {(ds, c, v[1]) for c, v in new.items()}
    out = RR.ledger_file(ds, RR_HAND)
    old = (
        {r["clip"]: (r["status"], r["source_tag"]) for r in FL.read_final_rows(out) if r["source_tag"] is not None}
        if out.is_file()
        else {}
    )
    # compared on (state, tag) only — note is a later fourth column absent from old files
    new2 = {c: (v[0], v[1]) for c, v in new.items()}
    diff = [(c, old.get(c), new2.get(c)) for c in sorted(set(old) | set(new2)) if old.get(c) != new2.get(c)]
    cnt = {}
    for s, *_rest in new.values():
        cnt[s] = cnt.get(s, 0) + 1
    for _s, _t, _nt, *_ in new.values():
        _k = _nt or "(current)"
        STAMP_CENSUS[_k] = STAMP_CENSUS.get(_k, 0) + 1
    print(
        f"[status_final] {ds}: {len(new)} clips {cnt} · changed vs existing {len(diff)}"
        + (f" e.g. {diff[:3]}" if diff else "")
    )
    print(_JB.summary([dict(zip(LEDGER_COLS, (c, *v), strict=True)) for c, v in new.items()], label=ds))
    return new, live, out, diff


def _write_retired(live, axes, a):
    """Stamp census line, then the cumulative retired ledger (written only when its content changes)."""
    if STAMP_CENSUS:
        _ret_rs = sum(1 for r in RETIRED if str(r[4]).startswith("ruleset:"))
        print(
            "[RULESET_CENSUS] stamp distribution of folded rows — "
            + " · ".join(f"{k} {v:,}" for k, v in sorted(STAMP_CENSUS.items()))
            + f" · retired (ruleset:) {_ret_rs:,}  (equivalence reasons = datasets.RULESET_EQUIVALENT)"
        )
    # Retired ledger = cumulative history (final_ledger.merge_retired). Runs even with nothing retired now —
    # reinstated rows must leave. Written only when the content changes.
    rec = RR.retired_ledger_file(RR_HAND)
    old_ret = FL.read_final_rows(rec) if rec.is_file() else []
    merged, st = FL.merge_retired(old_ret, RETIRED, live, set(axes))
    back = st["reinstated"]
    print(
        f"[RETIRED_LEDGER] existing {st['old']} · retired now {len(RETIRED)} (new {st['added']} · updated "
        f"{st['updated']}) · reinstated {len(back)}"
        + (f" e.g. {back[:3]}" if back else "")
        + f" → {len(merged)} rows"
        + (" (--check: not written)" if a.check else "")
    )
    if not a.check and merged != [[r[c] for c in FL.RETIRED_COLS] for r in old_ret]:
        rec = RR.table_out("ledger_retired.tsv", RR_HAND)
        tmp = rec.with_name(rec.name + ".tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f, **FL.WRITER)
            w.writerow(FL.RETIRED_COLS)
            w.writerows(merged)
        tmp.replace(rec)


def _fold_axes(scope, a, rc):
    """Fold every axis of the scope (carry declaration first, orphan chunks reported), then the retired ledger. Returns
    rc."""
    # the carry declaration is read first, whatever the axes: a broken one dies even in a fold of 0 axes, and the
    # applied-state marker is printed by every fold
    _FR.hand_carry_axis(RR_HAND, "", LEDGER_COLS[1:], _K)
    if orph := orphan_chunks(scope):
        print(
            f"[status_final] ⚠ {len(orph)} chunks outside the ledger axes — their results are not folded (tag rule "
            f"= datasets.ledger_key): {', '.join(orph)}"
        )
    axes = list(DATASETS)
    print(f"[status_final] scope datasets={','.join(sorted(scope))} → {len(axes)} axes: {','.join(axes) or '-'}")
    live: set = set()  # live row identities of the axes folded now (final_ledger.merge_retired)
    for ds in axes:
        new, live, out, diff = _fold_axis(ds, live)
        if not new:
            # 0 rows now: leaving the old file would keep "already judged" — write the header only (= empty)
            if out.is_file() and not a.check:
                with open(out, "w", newline="") as f:
                    _w0 = csv.writer(f, **FL.WRITER)
                    _w0.writerow(LEDGER_COLS)
                print(f"[status_final] {ds}: 0 rows → header only (every old row retired)")
            continue
        if a.check:
            rc |= int(bool(diff))
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        # readers do not lock — the file is replaced whole
        tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f, **FL.WRITER)
            w.writerow(LEDGER_COLS)
            for c in sorted(new):
                w.writerow([c, *new[c]])
        tmp.replace(out)
    _write_retired(live, axes, a)
    return rc


def main() -> int:
    """Fold this hand's chunk ledgers (module doc)."""
    import atexit  # noqa: PLC0415

    atexit.register(_report_excluded_test_tags)
    a = _parser()
    rc = 0
    # One ledger folder per hand; what the folder cannot separate is another arm of the same hand, hence the combo
    # stamp per hand. A different stamp dies.
    ledger_dir = P.run_path("run_ledger", RR_HAND)
    if not a.check:
        _fold_lock(ledger_dir)  # wait for another fold of this hand (before reading)
    stamp = ledger_dir / "status_final.combo"
    prev = stamp.read_text().strip() if stamp.is_file() else ""
    if prev and prev != ACTIVE_COMBO and not a.combo_switch:
        print(
            input_missing(
                f"[status_final] ✗ this ledger belongs to combo {prev}; the current axis is {ACTIVE_COMBO} "
                f"—\n  writing it would lose {prev}'s results. Pass --combo-switch to change it."
            ),
            file=sys.stderr,
        )
        return 2
    if not a.check:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(ACTIVE_COMBO + "\n")
    global DATASETS
    try:
        scope = set(RR.datasets_scope(a.datasets))  # empty = the campaign scope
    except KeyError as e:
        print(f"[status_final] ✗ {e}", file=sys.stderr)
        return 2
    for ds, why in RR.campaign_excluded().items():
        if ds not in scope:
            print(f"[FOLD_SCOPE] excluded {ds}({why})")
    DATASETS = axis_keys(scope)
    # Any chunk of unknown identity → write nothing and die (another hand's verdicts could mix into this ledger).
    # Evidence = datasets.CHUNK_IDENTITY_SOURCES.
    try:
        unk = unidentified_chunks(scope)
    except ValueError as e:  # two identities in one chunk (chunk_combo)
        print(f"[status_final] ✗ {e} — not folding", file=sys.stderr)
        return 2
    if unk:
        print(
            f"[status_final] ✗ {len(unk)} chunks of unknown identity (combo): {', '.join(unk[:10])} — not folding. "
            "Evidence = p0/plans2_<combo>.jsonl · datasets.CHUNK_IDENTITY_SOURCES",
            file=sys.stderr,
        )
        return 2
    print(f"[CHUNK_IDENTITY] chunk identities in scope settled — this hand's combo {ACTIVE_COMBO}")
    return _fold_axes(scope, a, rc)


if __name__ == "__main__":
    raise SystemExit(main())
