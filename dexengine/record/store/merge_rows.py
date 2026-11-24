"""Row-0 and supersede planning of the merge check (record.store.merge_check): which episodes drew row 0, which old
original or fallback rows an aug piece supersedes, and which same-input re-renders are skipped."""

from __future__ import annotations

from dexengine.core import paths as P
from dexengine.record import episodes as EPI


def src_clip(e) -> str:
    """Source clip (tree name) of an episode: selection.source_clip for a variant, its own clip for an original."""
    return (e.get("selection") or {}).get("source_clip") or e["clip"]


def row0_clips(eps) -> set:
    """Source clips that drew row 0 in an aug piece (traj_choice.row0, off-screen original rule)."""
    return {src_clip(e) for e in eps if e.get("round") == "aug" and (e.get("traj_choice") or {}).get("row0")}


def old_tree_aug(eps) -> list:
    """Aug episodes without the off-screen original rule record (traj_choice.orig_fov.rule = ORIG_FOV_RULE): pieces
    from trees that made no row-0 decision; the merge refuses them."""
    return [
        e["episode_index"]
        for e in eps
        if e.get("round") == "aug"
        and ((e.get("traj_choice") or {}).get("orig_fov") or {}).get("rule") != EPI.ORIG_FOV_RULE
    ]


def supersede_plan(cur_eps, gone: set, piece_eps, piece_id: str) -> list[dict]:
    """Old orig episodes of the final that this aug piece supersedes (not yet superseded) -> superseded.jsonl lines."""
    clips = row0_clips(piece_eps)
    return [
        {
            "episode_index": e["episode_index"],
            "clip": e["clip"],
            "src_clip": src_clip(e),
            "by_piece": piece_id,
            "rule": EPI.ORIG_FOV_RULE,
        }
        for e in cur_eps
        if e.get("round") == "orig" and src_clip(e) in clips and e["episode_index"] not in gone
    ]


#: aug -> aug row-0 replacement: when the final's aug row 0 is the original-trajectory fallback (orig_fov.row0_source
#: orig) and a new aug piece drew row 0 of the same source clip with a variant, the old fallback rows are marked
#: superseded (nothing deleted or overwritten). If any old row is not a fallback row 0, nothing is replaced.
AUG_ROW0_RULE = "aug_row0_orig_fallback_replaced"


SAME_FALLBACK_RULE = "aug_same_input_fallback_clip_skip"  # same-input fallback row-0 re-render: skip the clip


#: Original re-render of a clip already merged from the same input: keep the final copy, skip the clip.
ORIG_DUP_RULE = "orig_same_input_dup_clip_skip"


def fallback_row0(e) -> bool:
    """Is this aug episode the original-trajectory fallback row 0 (row 0 = the original itself, not a variant)?"""
    tc = e.get("traj_choice") or {}
    of = tc.get("orig_fov") or {}
    return (
        e.get("round") == "aug"
        and of.get("rule") == EPI.ORIG_FOV_RULE
        and of.get("row0_source") == "orig"
        and tc.get("row0") == e["clip"] == src_clip(e)
    )


def variant_row0_clips(eps) -> set:
    """Source clips whose row 0 was drawn with a variant (orig_fov.row0_source variant, that episode = the variant)."""
    out = set()
    for e in eps:
        tc = e.get("traj_choice") or {}
        of = tc.get("orig_fov") or {}
        if (
            e.get("round") == "aug"
            and of.get("rule") == EPI.ORIG_FOV_RULE
            and of.get("row0_source") == "variant"
            and tc.get("row0") == e["clip"] != src_clip(e)
        ):
            out.add(src_clip(e))
    return out


def aug_row0_plan(cur_eps, gone: set, piece_eps, piece_id: str, keys: set) -> list[dict]:
    """Overlapping (aug, source clip) keys that can be replaced -> superseded.jsonl lines (AUG_ROW0_RULE).
    Condition: the new piece drew that clip's row 0 with a variant and the final's live aug rows of the clip are one
    or more and all original-fallback row 0."""
    want = variant_row0_clips(piece_eps) & {c for r, c in keys if r == "aug"}
    live = {}
    for e in cur_eps:
        if e.get("round") == "aug" and src_clip(e) in want and int(e["episode_index"]) not in gone:
            live.setdefault(src_clip(e), []).append(e)
    return [
        {
            "episode_index": int(e["episode_index"]),
            "clip": e["clip"],
            "src_clip": c,
            "by_piece": piece_id,
            "rule": AUG_ROW0_RULE,
        }
        for c in sorted(want)
        if live.get(c) and all(fallback_row0(e) for e in live[c])
        for e in live[c]
    ]


def same_fallback_clips(cur_eps, gone: set, piece_eps, keys: set) -> set:
    """Overlapping (aug, source clip) keys that are a same-input re-render: the new piece's rows and the final's live
    aug rows of the clip are all original-fallback row 0 with the same cards -> clips to skip."""
    out = set()
    for c in sorted(x for r, x in keys if r == "aug"):
        new = [e for e in piece_eps if src_clip(e) == c]
        live = [
            e for e in cur_eps if e.get("round") == "aug" and src_clip(e) == c and int(e["episode_index"]) not in gone
        ]
        if new and live and all(map(fallback_row0, new + live)) and {card(e) for e in new} == {card(e) for e in live}:
            out.add(c)
    return out


def look_rev(e) -> str:
    """The episode's look version (look.measured.rev; 'None' when absent)."""
    return str(((e.get("look") or {}).get("measured") or {}).get("rev"))


def cell_of(src: dict, clip: str) -> dict:
    """The cell of a piece record (collect.json, or a final pieces line's collect) that carried the clip; {} when
    none or several."""
    hit = [c for c in src.get("cells") or [] if clip in {P.tree_name(x) for x in c.get("clips") or []}]
    return hit[0] if len(hit) == 1 else {}


def cell_key(c: dict) -> tuple:
    """Same spec base and same policy = (campaign, bundle, ckpt_sha256)."""
    return c.get("campaign"), c.get("bundle"), c.get("ckpt_sha256")


def orig_same_input_clips(cur_eps, rows, gone: set, piece_eps, src: dict, keys: set) -> tuple[set, list]:
    """Overlapping (orig, clip) keys that are a same-input re-render -> (clips to skip, [(clip, reason)] not skippable).
    Same input = the final has live orig rows of the clip, the card sets are equal, the look versions are equal, and
    the final pieces that carried them have the same (campaign, bundle, ckpt_sha256) cell as the new piece."""
    out, why = set(), []
    for c in sorted(x for r, x in keys if r == "orig"):
        new = [e for e in piece_eps if src_clip(e) == c]
        live = [
            e for e in cur_eps if e.get("round") == "orig" and src_clip(e) == c and int(e["episode_index"]) not in gone
        ]
        idx = {int(e["episode_index"]) for e in live}
        own = [r for r in rows if r["round"] == "orig" and any(r["episodes"][0] <= i < r["episodes"][1] for i in idx)]
        a, b = cell_of(src, c), [cell_of(r.get("collect") or {}, c) for r in own]
        bad = (
            "no live orig rows in the final (only superseded)"
            if not live
            else "cards differ"
            if {card(e) for e in new} != {card(e) for e in live}
            else "look versions differ"
            if {look_rev(e) for e in new} != {look_rev(e) for e in live}
            else "cell unknown (no policy fingerprint)"
            if not own or None in cell_key(a) or not all(b)
            else "spec base / policy differ"
            if any(cell_key(x) != cell_key(a) for x in b)
            else ""
        )
        if bad:
            why.append((c, bad))
        else:
            out.add(c)
    return out, why


def live_clips(r: dict, eps, gone: set) -> list:
    """Key clips of pieces line r; in an aug piece a clip whose episodes are all superseded is dropped (it has a new
    owner after AUG_ROW0_RULE)."""
    if r["round"] != "aug":
        return list(r["clips"])
    a, b = r["episodes"]
    by = {}
    for e in eps[a:b]:
        by.setdefault(src_clip(e), []).append(int(e["episode_index"]) in gone)
    return [c for c in r["clips"] if not (by.get(c) and all(by[c]))]


def row0_source(e, gone: set) -> str | None:
    """The source clip when the episode is its clip's row-0 (original stage) trajectory, else None: a live original,
    or a variant that an aug piece drew at row 0 under the off-screen original rule (traj_choice.row0)."""
    if int(e["episode_index"]) in gone:
        return None
    src = (e.get("selection") or {}).get("source_clip")
    return e["clip"] if not src else (src if (e.get("traj_choice") or {}).get("row0") == e["clip"] else None)


def card(e) -> tuple:
    """Episode card = (episode clip incl. variant suffix, look name); disjoint within the final."""
    return e["clip"], e["look"]["name"]
