"""Same-key overlap rules of a merge: which episodes of a piece are kept out when the final already holds their
(round, clip) keys — aug row-0 replacement, same-input clip skips, top-up (aug) and same-input skips of an original
re-render. Owner of keep_eps (a renumbered subset of a read piece). Callers: store.merge.merge_piece."""

from __future__ import annotations

import json

from dexengine.core.errors import input_missing
from dexengine.record.store import merge_check as MC
from dexengine.record.store import merge_rows as MR
from dexengine.record.store import merge_topup as TU
from dexengine.record.store.merge_check import EPISODES, PIECES, die, jsonl


def keep_eps(pc: dict, keep: list[int]) -> dict:
    """A subset of a read piece's episodes (renumbered 0..m-1; files, stats and source clips in the same order), for
    per-clip skips. info.collect *_counts are recounted from the kept episodes (only loophole_counts is known; any
    other field, or a recount that disagrees with the stored total, is an error)."""
    info = json.loads(json.dumps(pc["info"]))

    def cnt(eps, names):
        return {x: sum(e.get("loophole") == x for e in eps) for x in names}

    for k in [k for k in info["collect"] if k.endswith(MC.COUNTS)]:
        if k != "loophole_counts" or cnt(pc["eps"], info["collect"][k]) != info["collect"][k]:
            die(input_missing(f"clip skip: cannot recount info.collect.{k} from the episodes"))
        info["collect"][k] = cnt([pc["eps"][i] for i in keep], info["collect"][k])
    return dict(
        pc,
        info=info,
        eps=[dict(pc["eps"][i], episode_index=n) for n, i in enumerate(keep)],
        files=[pc["files"][i] for i in keep],
        stats=[pc["stats"][i] for i in keep],
        srcs=[pc["srcs"][i] for i in keep],
    )


def aug_overlap(dupk, rnd, out_, piece_id, final, pc, skipped, sup2, tadd, tdrop, topped):
    """Overlapping aug keys: an old fallback-only row 0 gives way to a variant row 0; same-input clips skip; top-up."""
    if dupk and rnd == "aug":  # overlapping aug keys: an old fallback-only row 0 is replaced by a variant row 0
        allx = jsonl(final / EPISODES)  # full read on this rare path only
        sup2 = MR.aug_row0_plan(allx, out_, pc["eps"], piece_id, dupk)
        out_ |= {x["episode_index"] for x in sup2}
        dupk -= {("aug", x["src_clip"]) for x in sup2}
        same = MR.same_fallback_clips(allx, out_, pc["eps"], dupk)  # same input (both fallback, same card)
        if same and (keep := [k for k, c in enumerate(pc["srcs"]) if c not in same]):
            skipped = [
                {"clip": c, "rule": MR.SAME_FALLBACK_RULE, "piece_eps": [k for k, x in enumerate(pc["srcs"]) if x == c]}
                for c in sorted(same)
            ]
            pc = keep_eps(pc, keep)
            dupk -= {("aug", c) for c in same}
        tkeep, tadd, tdrop = TU.plan(allx, out_, pc["eps"], dupk, TU.read_allow(final)) if dupk else ([], {}, [])
        if topped := bool((tadd or tdrop) and tkeep):  # top-up: an allow-listed clip takes new variants only
            pc = keep_eps(pc, tkeep)
            dupk -= {("aug", x["src_clip"]) for x in tdrop} | {("aug", c) for c in tadd}
    return sup2, out_, dupk, skipped, pc, tadd, tdrop, topped


def orig_overlap(dupk, rnd, out_, final, pc):
    """Original re-render: same input skips the overlapping clips (final copy kept)."""
    odup, owhy = [], []
    if dupk and rnd == "orig":  # original re-render: same input skips the overlapping clips, final copy kept
        rows = jsonl(final / PIECES)
        same, owhy = MR.orig_same_input_clips(jsonl(final / EPISODES), rows, out_, pc["eps"], pc["src"], dupk)
        if same and not owhy and (keep := [k for k, c in enumerate(pc["srcs"]) if c not in same]):
            odup = [
                {
                    "clip": c,
                    "rule": MR.ORIG_DUP_RULE,
                    "piece_eps": [k for k, x in enumerate(pc["srcs"]) if x == c],
                    "final_pieces": sorted(r["piece_id"] for r in rows if r["round"] == "orig" and c in r["clips"]),
                }
                for c in sorted(same)
            ]
            pc = keep_eps(pc, keep)
            dupk -= {("orig", c) for c in same}
    return odup, owhy, pc, dupk
