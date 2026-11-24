"""Owner moves inside one rule set: changes that moved only the **owner symbol** of a threshold in the rules body,
keeping the rule-set version (datasets.RULESET_VERSION).

The rules field of the judge input record hashes, per stage, each threshold's value **and its owner symbol**
(`path:name`). Moving a value to another owner without changing it therefore changes the hash, and every earlier row
would read as "rules differ within the same rule set = stale". Raising the rule-set version instead would also expire
the generation events tied to it, so a value-preserving move keeps the version and its equivalence is measured here.

Rule (the table is the input of the rule; the rule decides sameness):
  a stored stage hash is the same when it equals the hash of the current canonical body with the MOVES rows applied
  **backwards, newest first**. Backwards = moved symbols go back to their old symbol, and keys the move added are
  dropped only when their value equals the value the old code used (otherwise the value changed — not equivalent). A row
  whose scope predicate is false for a dataset cannot be crossed there (no equivalence with anything older). When the
  current record's stage hash differs from the hash of today's canonical body (a non-canonical run), the body cannot be
  rebuilt, so no equivalence is claimed.
A row is added only for a value-preserving owner move; a changed value raises the rule-set version.

Hash pairs (PAIRS_REL): judge records written before the release layout were hashed with symbols the release code does
not hold. Their equivalence ships as generated rows (dataset · scene · stage · stored stage hash · current stage
hash) — produced by a generator from owner-move rows (the one owner of that symbol table, outside this package; a
row exists only where the move's scope holds for that dataset) and checked against a regeneration where it is
present. A stored stage reads as the same only when its dataset, scene and both hashes match one row: a changed
threshold changes the current hash, so the row no longer matches (stale). The rows name no symbol.
"""

from __future__ import annotations

import collections
import copy
import functools
import math
import re

#: row = (commit · date · moved symbols {new: old} · added keys {new symbol: (old owner, old value)} ·
#: scope predicate (name, argument) · reason).
Move = collections.namedtuple("Move", "commit date renames added scope why")
#: Rows of this code. The rows that only keep the stored records of one deployment valid (their symbols spell the
#: paths of its earlier layouts) live in that deployment's plugin, which registers them through the record plugin hook
#: (dexengine.record.hooks, kind "owner_moves"; loaded from DEXENGINE_RECORD_PLUGINS).
MOVES: tuple = (
    Move(
        "4ab00d3fc",
        "2026-11-20 21:45",
        {},
        {
            "dexengine/placement/scene/object_desk.py:OBJ_BELOW_M": (
                "dexengine/data/candidate_rules.py:OBJ_BELOW_M",
                0.02,
            )
        },
        ("all", None),
        "rule D24 moved from the first pass into the judge (object_desk), value unchanged. The judge check "
        "(footprint + top) is a subset of the first-pass one (height only) on the same hull reader and desk plane: "
        "every tree an earlier judge record holds passed the first-pass D24 with this value, so the check cuts none "
        "of them (census kept->drop 0/400)",
    ),
)
#: Hash pair file (repository-relative) and its header — generated, never edited by hand (module docstring).
PAIRS_REL = "configs/placement/judge_record_pairs.tsv"
PAIRS_HEAD = ("dataset", "scene", "stage", "stored", "current")
_HASH = re.compile(r"[0-9a-f]{12}")
#: The plugin hook kind the rows of a plugin are registered under (one tuple of Move rows per name).
HOOK_KIND = "owner_moves"
_SAID: list = []


def _jfp():
    from dexengine.placement.judge import input_record as JFP  # noqa: PLC0415 — owner of the body and the hash

    return JFP


def _scope_desk_top(ds: str, box: tuple) -> bool:
    """The desk top the judge uses for this dataset (coords.desk_top_for) equals the old box."""
    now = tuple(float(v) for v in _jfp()._coords().desk_top_for(ds))
    return len(now) == len(box) and all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(now, box, strict=True)
    )


#: scope predicate name → function(ds, argument) → True = that row is a value-preserving move for this dataset.
SCOPES = {"desk_top": _scope_desk_top, "all": lambda ds, arg: True}


def moves() -> tuple:
    """This code's rows and every plugin's (record plugin hook, kind HOOK_KIND), oldest first by date — the order the
    rule walks backwards. A row naming a scope predicate missing from SCOPES raises KeyError. Marker (stderr, once per
    process): `[OWNER_MOVES] rows <n> (plugin rows <k>: <names>)` — a deployment whose stored records need its rows
    sees 0 when its plugin was not loaded."""
    from dexengine.record import hooks  # noqa: PLC0415 — stdlib-only plugin registry

    hooks.load_plugins()
    names = hooks.names(HOOK_KIND)
    extra = [m for n in names for m in hooks.get(HOOK_KIND, n)]
    rows = tuple(sorted((*MOVES, *extra), key=lambda m: m.date))
    if miss := [m.commit for m in rows if m.scope[0] not in SCOPES]:
        raise KeyError(f"owner_moves rows {miss} name a scope predicate missing from SCOPES")
    if not _SAID:
        import sys  # noqa: PLC0415

        _SAID.append(1)
        print(
            f"[OWNER_MOVES] rows {len(rows)} (plugin rows {len(extra)}: {', '.join(names) or 'none'})",
            file=sys.stderr,
            flush=True,
        )
    return rows


@functools.cache
def pairs() -> frozenset:
    """{(dataset, scene, stage, stored hash, current hash)} of the hash pair file; a missing file, another header, a
    dataset · scene · stage outside the rules table or a value that is not a stage hash raises (the file is part of the
    release)."""
    from dexengine.core.repo import REPO_ROOT  # noqa: PLC0415

    path, J = REPO_ROOT / PAIRS_REL, _jfp()
    rr = J._rr()
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or tuple(lines[0].split("\t")) != PAIRS_HEAD:
        raise ValueError(f"{path}: header is not {PAIRS_HEAD}")
    out = set()
    for n, line in enumerate(lines[1:], 2):
        row = tuple(line.split("\t"))
        ok = len(row) == len(PAIRS_HEAD) and row[0] in rr.DATASET_PROFILE and row[1] in rr.SCENES
        if not (ok and row[2] in J.STAGES and all(_HASH.fullmatch(h) for h in row[3:])):
            raise ValueError(f"{path}:{n}: not a {PAIRS_HEAD} row of the rules table: {line!r}")
        out.add(row)
    return frozenset(out)


def reverted_hashes(ds: str, body: dict, rows=None) -> list[dict]:
    """{stage: hash} of the canonical body after each row (`rows`, default moves()) applied backwards, newest first —
    stops at a row out of scope for `ds` or one that does not revert (a changed value)."""
    J, out, b = _jfp(), [], body
    for mv in reversed(moves() if rows is None else rows):
        if not in_scope(mv, ds):
            break
        b = revert(b, mv)
        if b is None:
            break
        out.append({s: J._h(b[s]) for s in J.STAGES})
    return out


def in_scope(mv: Move, ds: str) -> bool:
    return bool(SCOPES[mv.scope[0]](ds, mv.scope[1]))


def revert(body: dict, mv: Move) -> dict | None:
    """A copy of the body {stage: {thresholds, …}} with one row applied backwards — None (not equivalent) when an added
    key's value differs from the old value, or a reverted symbol is already present (overlap)."""
    out = copy.deepcopy(body)
    for part in out.values():
        th, new = part["thresholds"], {}
        for k, v in th.items():
            if k in mv.added:
                old = mv.added[k][1]
                if not (isinstance(v, (int, float)) and math.isclose(float(v), float(old), rel_tol=0.0, abs_tol=1e-12)):
                    return None
                continue
            nk = mv.renames.get(k, k)
            if nk in new:
                return None
            new[nk] = v
        part["thresholds"] = new
    return out


def equivalent_stages(ds: str, scene: str, stored_rules: str, current_rules: str, stages) -> set:
    """Of `stages`, those whose two rules fields (`j:<h>,b:<h>,g:<h>`) are equivalent within the rule set — the stored
    hash equals the hash of the current canonical body reverted through MOVES (newest first, stopping at a row out of
    scope). Empty when the current field differs from today's canonical body or the scene / dataset is outside the
    rules table (the body cannot be rebuilt). A stage also reads the same when (dataset, scene, stage, stored, current)
    is a row of the hash pair file (pairs)."""
    J = _jfp()
    want = list(stages)
    try:
        st, cur = J.rules_stages(stored_rules), J.rules_stages(current_rules)
    except ValueError:
        return set()
    if scene not in J._rr().SCENES or ds not in J._rr().DATASET_PROFILE:
        return set()
    body = J._rules_body(ds, scene, J.canon_run_flags(ds, scene))
    if any(J._h(body[s]) != cur[s] for s in want):
        return set()
    out = {s for s in want if (ds, scene, s, st[s], cur[s]) in pairs()}
    for h in reverted_hashes(ds, body):
        out |= {s for s in want if h[s] == st[s]}
    return out
