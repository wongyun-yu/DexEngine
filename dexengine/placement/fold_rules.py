"""Row rules of the ledger fold (`status_fold.py`): which chunk ledger rows are retired instead of folded, and the
check that one clip's verdicts do not split across campaigns.

A retired row goes to the retired ledger with a reason (written as data — keep the reason heads): `stale_judge:` (the
judge input changed since), `unit_regen:` / `window_replaced:` (the clip's window changed since), the ruleset retire
reasons (datasets.ruleset_retire_reason / ruleset_row_reason). The accumulators below (RETIRED, WINDOW_RETIRE,
RS_SCOPE) are per-process state that status_fold reads after folding.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import re
from pathlib import Path

from dexengine.core import paths as P
from dexengine.core.errors import InputMissingError
from dexengine.data import rules as RR
from dexengine.data.cases import assign_version as AR
from dexengine.placement.ledger import read_status_ledger

# Unit (_tN) regeneration events — units whose window (segment) changed under the same name (MANO re-split). Rows of
# `_tN` clips from chunks listed before this time are not folded; they stay in the retired ledger (`unit_regen:` —
# _old_window_reason).
REGEN_EVENTS_UNITS = {
    ds: _dt.datetime.fromisoformat("2026-09-21T00:12:00+09:00").timestamp() for ds in ("hocap", "hot3d", "oakink2")
}
REGEN_EVENTS = {  # dataset → end of the base regeneration that changed qpos
    "h2o": _dt.datetime.fromisoformat("2026-09-15T18:55:00+09:00").timestamp(),  # per-take tilt rotation
}
# These two tables are not derived from task_info `provenance.time`: that is when the base adapter built the tree,
# while REGEN_EVENTS_UNITS is the later MANO re-split — a different event.

#: Events that changed the judge INPUT. A rejection row (not ADOPTED) judged before such an event proves nothing about
#: the current input, so it is not folded — the retired row (`stale_judge:`) counts as unjudged and is queued again.
#: Events with a log marker (active sides [ACTIVE_SIDES] · dz cells [DZ_CELLS]) are compared by value, the others by
#: time.
JUDGE_INPUT_EVENTS = {
    "D22": (
        _dt.datetime.fromisoformat("2026-09-17T22:57:00+09:00").timestamp(),
        "table",
        "before the canonical desk box (judged with the tree footprint box)",
    ),
    "D23_f0": (
        _dt.datetime.fromisoformat("2026-09-18T00:10:00+09:00").timestamp(),
        "SOURCE_DEFECT",
        "f0 layout failure before roles (D23) — the tree has a main object now",
    ),
}


def _dz_policy_for(ds: str, scene: str) -> str:
    from dexengine.placement import search_policy as _DZ  # noqa: PLC0415

    return _DZ.dz_cells_for(ds, scene)


def _judge_markers(d: Path) -> tuple[str | None, dict]:
    """Markers of a chunk's judge.log: ([DZ_CELLS] cells=… (None when absent), {clip: active=…}). A chunk without
    markers was judged before the active-side (D16) and dz-cell (D17) rules."""
    dz = None
    act: dict = {}
    f = d / "judge.log"
    if not f.is_file():
        return dz, act
    with open(f, errors="replace") as fh:
        for ln in fh:
            if ln.startswith("[DZ_CELLS]") and dz is None:
                m = re.search(r"cells=(\S+)", ln)
                dz = m.group(1) if m else "?"
            elif ln.startswith("[ACTIVE_SIDES] ") and " active=" in ln:
                parts = ln.split()
                m = re.search(r"active=(\S+)", ln)
                if len(parts) > 1 and m:
                    act[parts[1]] = {"both": "left+right"}.get(m.group(1), m.group(1))
    return dz, act


def _current_active(ds: str, emb: str, clip: str) -> str | None:
    """Active sides of the current tree = case.active (D16) ∩ manipulating sides (the sides the assignment gave a
    main object). None when there is no tree."""
    f = P.layer_path("base", ds, hand=RR.hand_dir(), emb=emb) / clip / "task_info.json"
    if not f.is_file():
        return None
    d = json.loads(f.read_text())
    c = d.get("case") or {}
    per = ((d.get("rules") or {}).get(AR.ASSIGN_KEY) or {}).get("per_side") or {}
    a16 = set(c.get("active") or [])
    a23 = {s for s, p in per.items() if p.get("main")}
    return "+".join(sorted((a16 & a23) if (a16 and a23) else (a16 or a23)))


def _stale_judge_reason(
    ds: str,
    emb: str,
    scene: str,
    clip: str,
    status: str,
    t_led: float,
    dz_judged: str | None,
    act: dict,
    dz_policy: str,
) -> str | None:
    """The reason (string) when a rejection row was judged on an old judge input, else None. Not called for
    ADOPTED."""
    dzj = dz_judged if dz_judged is not None else "-5.5"  # no marker = the single default layer before D17
    if dzj != dz_policy:
        return f"stale_judge:dz({dzj}≠{dz_policy})"
    if not act:
        return "stale_judge:pre_D16(no active-side record · both arms forced)"
    ja = act.get(clip)
    if ja is not None:
        now = _current_active(ds, emb, clip)
        if now and ja != now:
            return f"stale_judge:active({ja}≠{now})"
    t22, sc22, _ = JUDGE_INPUT_EVENTS["D22"]
    if scene == sc22 and t_led < t22:
        return "stale_judge:D22(before the desk box)"
    t23, st23, _ = JUDGE_INPUT_EVENTS["D23_f0"]
    if status == st23 and t_led < t23:
        return "stale_judge:D23(before the f0 layout)"
    return None


_REPLACED: dict | None = None


def _replaced_units() -> dict:
    """Segment name → window replacement time (epoch), from the file segment split writes (--reslice-stale ·
    --purge-unregistered; name = datasets.units_replaced_file)."""
    global _REPLACED
    if _REPLACED is None:
        _REPLACED = {}
        f = RR.units_replaced_file()  # keyed by hand spec directory, not hand id
        if f.is_file():
            with open(f) as fh:
                for r in csv.reader(fh, delimiter="\t"):
                    if len(r) >= 6 and r[0] != "dataset":  # a time that does not parse dies (skipped, the old window's
                        _REPLACED[r[2]] = max(
                            _REPLACED.get(r[2], 0.0),  # verdict would pass as current)
                            _dt.datetime.fromisoformat(r[5]).timestamp(),
                        )
    return _REPLACED


#: Verdict rows of lists older than a window replacement — kept · retired in this fold ([WINDOW_RETIRE] per axis).
WINDOW_RETIRE: dict[str, list[int]] = {}
_CANON_COND: dict[tuple, dict] = {}


def _window_input_same(ds: str, emb: str, scene: str, clip: str, stored: str, status: str) -> tuple[bool, str]:
    """Whether the input this verdict row saw (judge input fingerprint) equals the current input → (same,
    verdict:differing cells). Uses judge.input_record.fingerprint and diff_ruleset. No fingerprint (unknown), not
    measurable now (unevaluable) and different (stale) are not the same."""
    from dexengine.placement.judge import input_record as JFP  # noqa: PLC0415 — heavy, only with replaced windows

    if JFP.no_fingerprint(stored):  # judged before fingerprints — no input identity
        return False, "unknown:-"
    if (ds, scene) not in _CANON_COND:
        _CANON_COND[(ds, scene)] = JFP.canon_conditions(ds, scene, RR.active_arm())
    cur = JFP.fingerprint(ds, emb, RR.active_hand(), clip, scene, _CANON_COND[(ds, scene)])
    verdict, comps = JFP.diff_ruleset(stored or None, cur, status, (ds, emb, RR.hand_dir(), clip))
    return verdict == "same", f"{verdict}:{','.join(comps) or '-'}"


def window_note(axis: str) -> str:
    """One marker line per axis — verdict rows of pre-replacement lists kept · retired in this fold."""
    keep, retire = WINDOW_RETIRE.get(axis, [0, 0])
    return (
        f"[WINDOW_RETIRE] {axis} kept {keep} · retired {retire} (verdict rows of lists before a window "
        "replacement — same input fingerprint = kept)"
    )


def _old_window_reason(
    clip: str, t_list: float, ucut: float, judged: tuple | None = None, axis: str = ""
) -> str | None:
    """The retire reason when this clip's verdict from a chunk whose list is stamped `t_list` belongs to an OLD
    window, else None.

    Two events: 1. `_tN` rows of lists older than the re-split (REGEN_EVENTS_UNITS · dataset time `ucut`);
    2. rows of lists older than a window replacement record (units_replaced) — kept when the input the verdict saw
    (judge input fingerprint) equals the current input (_window_input_same · judged = (dataset, emb, scene,
    fingerprint, state)): a segment re-cut to the same range keeps its verdict. Kept · retired counts go to
    WINDOW_RETIRE per axis. The reason heads (`unit_regen:` · `window_replaced:`) do not overlap `stale_judge:` and
    `ruleset:` (readers group retired rows by the text before `:`)."""
    if t_list < ucut and re.fullmatch(r".+_t\d+", clip):
        return f"unit_regen:{_iso(ucut)}(_tN of a list from before the re-split — old window · REGEN_EVENTS_UNITS)"
    rep = _replaced_units().get(clip)
    if rep is None or t_list >= rep:
        return None
    if judged is None:
        raise InputMissingError(
            f"[status_final] ✗ {clip}: a row older than a window replacement but no input "
            "comparison (judged) — not retiring by time alone"
        )
    same, why = _window_input_same(judged[0], judged[1], judged[2], clip, judged[3], judged[4])
    cnt = WINDOW_RETIRE.setdefault(axis, [0, 0])
    if same:
        cnt[0] += 1
        return None
    cnt[1] += 1
    return (
        f"window_replaced:{_iso(rep)}(verdict of a list from before the window replacement · "
        f"input {why} — units_replaced)"
    )


def _iso(t: float) -> str:
    return _dt.datetime.fromtimestamp(t).isoformat(timespec="seconds")


def _ruleset_stamp(d: Path) -> str | None:
    """The chunk's `ruleset=` stamp in run.env (stage 2 writes it). None when the file or line is missing."""
    env = d / "run.env"
    if not env.is_file():
        return None
    out = None
    for ln in env.read_text().splitlines():
        if ln.startswith("ruleset="):
            out = ln.split("=", 1)[1].strip()
    return out


#: Rows this fold retired from the current rules — main() merges them into the per-hand retired ledger.
RETIRED: list[list[str]] = []
#: (axis, old stamp) → [hit (retired), carried] — rows of old-ruleset chunks bridged by a predicate ([RULESET_RETIRE]).
RS_SCOPE: dict[tuple, list] = {}


def _ruleset_scope(ds: str, stamp: str | None) -> tuple | None:
    """(dataset, emb, stamp) when the chunk's old ruleset is bridged by predicates (datasets.RULESET_STEPS), else
    None."""
    if not (RR.ruleset_chain(stamp) and ds.split("-")[0] in RR.RULESET_RETIRE):
        return None
    _d, _e, _v = RR.ledger_axis_parts(ds)
    RS_SCOPE.setdefault((ds, stamp), [0, 0])
    return _d, _e, stamp


def _live_rows(led: Path, ds: str, chunk: str, scope: tuple | None):
    """Rows with a state of one chunk ledger. Under a predicate ruleset (scope) each row is checked against that
    change (datasets.ruleset_row_reason) — a row it touches or cannot measure goes to the retired ledger and is
    skipped (only that row is judged again) · an untouched row is carried."""
    for row in read_status_ledger(led):
        if row["status"] is None:
            continue
        if scope:
            cnt = RS_SCOPE[(ds, scope[2])]
            why = RR.ruleset_row_reason(scope[0], scope[1], RR.hand_dir(), row["clip"], scope[2])
            if why:
                RETIRED.append([ds, row["clip"], row["status"], chunk, why])
                cnt[0] += 1
                continue
            cnt[1] += 1
        yield row


def ruleset_scope_notes(ds: str) -> list[str]:
    """[RULESET_RETIRE] marker lines of one axis — per old stamp the retired (hit) and carried rows and the
    predicates."""
    cur = RR.RULESET_VERSION
    return [
        f"[RULESET_RETIRE] {st}→{cur} hit {hit} · carried {carry} ({ds} · predicates "
        f"{'+'.join(x[2] for x in RR.ruleset_chain(st))})"
        for (ax, st), (hit, carry) in sorted(RS_SCOPE.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))
        if ax == ds
    ]


#: Columns of the per-hand fold owner declaration (datasets.fold_owner_file).
FOLD_OWNER_COLS = ("axis", "clip", "owner", "evidence")


def fold_owners(ds: str) -> dict[str, str]:
    """Clip → owner campaign of this hand's owner declarations for axis `ds`. An empty evidence cell or a wrong
    header dies."""
    f = RR.fold_owner_file(RR.active_hand())
    if not f.is_file():
        return {}
    out = {}
    for i, r in enumerate(csv.DictReader(f.open(encoding="utf-8"), delimiter="\t"), 2):
        if tuple(r) != FOLD_OWNER_COLS or not all((r[k] or "").strip() for k in FOLD_OWNER_COLS):
            raise SystemExit(
                f"[FOLD_SPLIT] ✗ owner declaration {f}:{i} — columns {FOLD_OWNER_COLS} must all be "
                "filled (a declaration without evidence is refused)"
            )
        if r["axis"] == ds:
            out[r["clip"]] = r["owner"]
    return out


def split_verdicts(ds: str, seen: dict[str, list]) -> None:
    """Two verdicts for one clip: per campaign take that campaign's LAST verdict per (judge input fingerprint,
    budget); die when campaigns differ in state. The fold picks the latest ledger, so a clip with two judge owners
    would otherwise be decided silently by collection order. Re-judging inside one campaign (retry, deadline round):
    the later one wins. Empty fingerprints (old chunks) are not compared. A split with an owner declaration
    (fold_owners) passes — only when the declared owner is the campaign of the clip's LATEST row (the fold picks the
    latest row, so owner = folded result). seen = clip → [(ledger time, chunk, state, fingerprint, budget)]."""
    bad, owners = [], fold_owners(ds)
    for c, rs in seen.items():
        last: dict[tuple, tuple] = {}  # (fingerprint, budget, campaign) → (time, state)
        for t, chunk, st, fp, bud in rs:
            k = (fp, bud, re.match(r"G1[a-z]*", chunk).group(0))
            if fp and (k not in last or t > last[k][0]):
                last[k] = (t, st)
        by: dict[tuple, dict[str, tuple]] = {}
        for (fp, bud, camp), (t, st) in last.items():
            by.setdefault((fp, bud), {})[camp] = (t, st)
        late = re.match(r"G1[a-z]*", max(rs)[1]).group(0)  # campaign of the clip's latest row (the folded one)
        for camps in by.values():
            if len({v[1] for v in camps.values()}) > 1:
                if owners.get(c) == late:
                    continue
                bad.append(
                    f"{c} "
                    + " · ".join(f"{k}:{v[1]}" for k, v in sorted(camps.items()))
                    + (f" (declared owner {owners[c]} ≠ latest {late})" if c in owners else "")
                )
    if bad:
        raise SystemExit(
            f"[FOLD_SPLIT] ✗ {ds}: {len(bad)} clips whose verdicts split across campaigns for the same "
            f"judge input and budget — e.g. {bad[:3]} · pick one judge owner, or declare it with evidence "
            "in the owner declaration (datasets.fold_owner_file), then fold"
        )


# ── Hand carry: a receiving hand's declaration (`paths.run_path("run_hand_carry", <hand>)`, written by the hand-carry
# tool) carries the source hand's final rows for clips this hand did not judge itself. No declaration = off, and the
# fold is the same text as without the rule.
CARRY_TAG = "[HAND_CARRY]"
CARRY_RULE = "hc1"  # rule version in the declaration header
CARRY_NOTE_HEAD = "hand_carry("  # note of a carried row: hand_carry(<source hand>:<category>) — parsed downstream
_CARRY_DECL: list = []  # read once per process — [declaration | None]


def read_hand_carry(hand: str, root: Path | None = None) -> dict | None:
    """The receiving hand's carry declaration → {"from", "mark", "expiry", "canon", "rows": {(axis, clip): row},
    "file"}; None when there is none. A header whose hand / rule version / row count does not match dies."""
    from dexengine.core import layout as DL  # noqa: PLC0415 — owner of the provisional mark

    p = P.run_path("run_hand_carry", hand, root=root)
    if not p.is_file():
        return None
    mark = DL.hand_provisional(hand, str(root if root is not None else P.run_root()))
    head: dict[str, str] = {}
    body: list[str] = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.startswith("#"):
            k, _, v = ln[1:].strip().partition("=")
            head[k.strip()] = v.strip()
        else:
            body.append(ln)
    if head.get("to") != hand or head.get("rule") != CARRY_RULE:
        raise SystemExit(
            f"{CARRY_TAG} ✗ {p}: receiving hand {head.get('to')!r} ≠ {hand!r} or rule "
            f"{head.get('rule')!r} ≠ {CARRY_RULE} — rewrite the declaration"
        )
    rows = {(r["axis"], r["clip"]): r for r in csv.DictReader(body, delimiter="\t")}
    if str(len(rows)) != head.get("rows"):
        raise SystemExit(f"{CARRY_TAG} ✗ {p}: {len(rows)} rows ≠ header rows={head.get('rows')} — broken declaration")
    return {
        "from": head["from"],
        "mark": mark["mark"] if mark else None,
        "expiry": mark["expiry"] if mark else None,
        "canon": head.get(DL.CANON_KEY) or None,
        "rows": rows,
        "file": str(p),
    }


def hand_carry_axis(hand: str, ax: str, cols: tuple, key) -> dict[str, tuple[str, ...]]:
    """One axis of the hand's carry declaration → {clip key: row in `cols` order} (note = hand_carry(<source>:
    <category>)); {} without a declaration. `key(clip, dataset)` builds the fold's clip key."""
    if not _CARRY_DECL:
        d = read_hand_carry(hand)
        _CARRY_DECL.append(d)
        if d is not None:  # applied-state marker, once
            print(
                f"{CARRY_TAG} on — source {d['from']} · declaration {d['file']} · rows {len(d['rows'])} · "
                f"{d['mark']} · expiry {d['expiry']}"
            )
    d = _CARRY_DECL[0]
    if d is None:
        return {}
    out = {}
    for (a, c), r in d["rows"].items():
        if a != ax:
            continue
        cells = {k: r.get(k) or "" for k in cols}
        cells["note"] = f"{CARRY_NOTE_HEAD}{d['from']}:{r['category']})"
        out[key(c, ax.split("-")[0])] = tuple(cells[k] for k in cols)
    return out


def fold_hand_carry(hand: str, ds: str, out: dict, legit, notes: list, cols: tuple, key) -> None:
    """Fold the carried rows into clips that have **no** row of this hand. Nothing is folded without a base layer
    (no rule cleanup is possible on that axis), nor for clips outside this hand's axis set."""
    hc = hand_carry_axis(hand, ds, cols, key)
    if not hc:
        return
    if legit is None:
        notes.append(
            f"{CARRY_TAG} {ds}: no base layer / axis set — {len(hc)} carried rows not folded (fold again after stage 1)"
        )
        return
    own = {c for c in hc if c in out}
    add = {c: v for c, v in hc.items() if c not in out and c in legit}
    out.update(add)
    notes.append(
        f"{CARRY_TAG} {ds}: {len(add)} carried rows folded · {len(own)} won by this hand's own row"
        f" · {len(hc) - len(add) - len(own)} without a base of this hand"
    )
