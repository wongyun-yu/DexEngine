"""Top-up: add only new placement variants to a clip that is already in the augmented final (mostly 4,
if the cost is not large). The merge (store.merge.merge_piece) calls it for the overlapping (aug, source clip) keys
left after the same-input skip and the row-0 replacement.

Allow table = <aug root>/ALLOW_DIR/<combo>.tsv — outside the final data folder; written by the top-up list (built
from the finals on the judging host).
Header ALLOW_HEAD; a row = dataset · source clip (tree name, the same form as the final line's
selection.source_clip) · need (target variant count). Without a table, or for a clip not in it, the overlap dies as
before — the safety net against assigning a clip twice stays.
Rule (AUG_TOPUP_RULE): for a listed clip, only the new piece's variants (episode clip names) that are not among the
final's live variants are added, in variant-number (k) order, as many as final names (the original fallback row 0
included) + added ≤ need. All look rows of one variant go together. The rest (original fallback rows · already in
the final · over the target) are skipped and recorded in the pieces line topup_dropped (AUG_TOPUP_DROP_RULE).
A name is the cell list's order (k), so in a clip whose cell list changed the same name points at another placement —
the placement values (selection dx · dy · yaw) are compared: the same name with another placement, or a new name with
the placement of a final variant, dies (the name alone cannot catch it).
Added = the pieces line topup {source clip: [variants]}. The checks (merge_check.meta_problems · post_problems) leave
that clip's (aug, source clip) key out of that piece (key_clips) and check that an earlier piece has the clip and that
the variants are disjoint from the earlier ones (problems).
"""

from __future__ import annotations

import re
from pathlib import Path

from dexengine.core.paths import PV_SFX_PAT
from dexengine.record.store import merge_rows as MR  # no cycle: merge_rows does not import this module

AUG_TOPUP_RULE = "aug_topup_new_variants"
AUG_TOPUP_DROP_RULE = "aug_topup_dropped_variant"
ALLOW_DIR = "topup_allow"
ALLOW_HEAD = ["dataset", "clip", "need"]
_K = re.compile(r"_pv(\d{2,})")
MARK = "[AUG_TOPUP]"
#: stored values (pieces.jsonl topup_dropped why — the generation line writes the same text): original fallback row,
#: not a top-up target · already in the final · over the target (final n + added ≥ need)
WHY_FALLBACK = "original fallback row — not a top-up target"
WHY_IN_FINAL = "already in the final"
WHY_OVER = "over the target (final {have} + added ≥ {need})"


def _die(msg: str):
    raise SystemExit(f"{MARK} ✗ {msg}")


def allow_path(final) -> Path | None:
    """Final <root>/data/<ds>/collect/<combo> → <root>/ALLOW_DIR/<combo>.tsv · another form → None (no top-up)."""
    f = Path(final)
    if f.parent.name != "collect" or f.parents[2].name != "data":
        return None
    return f.parents[3] / ALLOW_DIR / f"{f.name}.tsv"


def read_allow(final) -> dict[str, int]:
    """The allow table of this final's dataset {source clip: need} — no table: {} · a wrong header or value dies."""
    p = allow_path(final)
    if p is None or not p.is_file():
        return {}
    rows = p.read_text(encoding="utf-8").splitlines()
    if not rows or rows[0].split("\t")[:3] != ALLOW_HEAD:
        _die(f"allow table {p} header != {ALLOW_HEAD}")
    ds, out = Path(final).parents[1].name, {}
    for n, ln in enumerate(rows[1:], 2):
        f = ln.split("\t")
        if len(f) < 3 or not f[2].isdigit() or f[1] in out and f[0] == ds:
            _die(f"allow table {p}:{n} bad row (columns · need · same clip twice) — {ln!r}")
        if f[0] == ds:
            out[f[1]] = int(f[2])
    return out


def is_variant(name: str) -> bool:
    return bool(re.search(PV_SFX_PAT, name))


def k_of(name: str) -> int:
    """Variant number of a variant episode clip name (form owner paths.PV_SFX_PAT) — not a variant dies."""
    m = _K.search(name)
    if not m or not is_variant(name):
        _die(f"not a variant name {name!r}")
    return int(m.group(1))


def place_of(e) -> tuple | None:
    """The line's placement (selection dx · dy · yaw) — an original fallback row (not a variant name) gives None; a
    variant without values dies."""
    if not is_variant(e["clip"]):
        return None
    s = e.get("selection") or {}
    try:
        return tuple(round(float(s[k]), 6) for k in ("dx", "dy", "yaw"))
    except (KeyError, TypeError, ValueError):
        _die(f"variant line {e['clip']} has no placement values (selection dx · dy · yaw)")


def live_variants(cur_eps, gone: set, clips: set) -> dict[str, dict]:
    """The final's live aug lines → {source clip: {name: placement}} (clips only; an original fallback row = None)."""

    out: dict[str, dict] = {}
    for e in cur_eps:
        if e.get("round") == "aug" and int(e["episode_index"]) not in gone and (c := MR.src_clip(e)) in clips:
            out.setdefault(c, {})[e["clip"]] = place_of(e)
    return out


def same_place(c: str, have: dict, got: dict) -> None:
    """Piece variants got {name: {placement}} ↔ final have {name: placement} — the same name = the same placement ·
    a new name ≠ any final placement."""
    old = {p: n for n, p in have.items() if p is not None}
    for v, ps in got.items():
        if len(ps) != 1:
            _die(f"{c}: the look rows of piece variant {v} have different placements {sorted(ps)[:2]}")
        p = next(iter(ps))
        if v in have and have[v] != p:
            _die(
                f"{c}: variant {v} has the final's name but another placement ({have[v]} != {p}) — a clip whose "
                "cell list changed"
            )
        if v not in have and p in old:
            _die(
                f"{c}: new variant {v} has the placement of final variant {old[p]} — same placement, another name "
                "(a clip whose cell list changed)"
            )


def plan(cur_eps, gone: set, piece_eps, keys: set, allow: dict) -> tuple[list[int], dict, list]:
    """Overlapping keys of allow-listed aug clips → (piece episode indices to keep, added {source clip: [variants]},
    skipped rows). Clips outside the table and episodes of other rounds are kept (the later overlap gate checks
    them)."""

    want = {c for r, c in keys if r == "aug" and c in allow}
    live = live_variants(cur_eps, gone, want)
    by: dict[str, dict[str, list[int]]] = {}
    keep: list[int] = []
    for i, e in enumerate(piece_eps):
        c = MR.src_clip(e)
        if c in want:
            by.setdefault(c, {}).setdefault(e["clip"], []).append(i)
        else:
            keep.append(i)
    added, dropped = {}, []
    for c in sorted(by):
        have = live.get(c, {})
        vs = sorted((v for v in by[c] if is_variant(v)), key=k_of)
        same_place(c, have, {v: {place_of(piece_eps[i]) for i in by[c][v]} for v in vs})
        take = [v for v in vs if v not in have][: max(0, allow[c] - len(have))]
        for v in [v for v in by[c] if not is_variant(v)] + vs:
            if v in take:
                keep += by[c][v]
            else:
                dropped.append(
                    {
                        "clip": v,
                        "src_clip": c,
                        "rule": AUG_TOPUP_DROP_RULE,
                        "piece_eps": by[c][v],
                        "why": WHY_FALLBACK
                        if not is_variant(v)
                        else WHY_IN_FINAL
                        if v in have
                        else WHY_OVER.format(have=len(have), need=allow[c]),
                    }
                )
        if take:
            added[c] = take
    return sorted(keep), added, dropped


def key_clips(r: dict, clips) -> list:
    """The (round, source clip) key clips of pieces line r — clips added by a top-up are left out (an earlier piece
    owns them)."""
    tops = set(r.get("topup") or {})
    return [c for c in clips if c not in tops]


def problems(pieces, eps, gone: set) -> list[str]:
    """Top-up oracle (whole final) — an added clip must be in an earlier aug piece, the added variants inside that
    piece's episode_clips and disjoint from the earlier pieces' variants of the clip (the live variant count is the
    allow table's business and is not measured here)."""

    bad, seen = [], {}
    for r in pieces:
        a, b = r["episodes"]
        tops = r.get("topup") or {}
        for c, vs in tops.items():
            if r["round"] != "aug" or c not in seen or not set(vs) <= set(r["episode_clips"]) or set(vs) & seen[c]:
                bad.append(
                    f"top-up piece {r['piece_id']} clip {c}: no earlier piece · variant outside · overlaps an "
                    f"earlier variant ({vs[:3]})"
                )
        if r["round"] == "aug":
            for e in eps[a:b]:
                if int(e["episode_index"]) not in gone:
                    seen.setdefault(MR.src_clip(e), set()).add(e["clip"])
    return bad
