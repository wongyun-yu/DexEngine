"""The record entry's render-only object index (the record's render reads it; shape and reading: scene.render_only,
rule and writer: scene.render_only_build). The index is not a shipped input. The record entries (record.bundle before
stage 1 of a run that renders, record.driver before its workers) call ensure_index: when the canonical layer has no
index for the dataset, it is built for the whole dataset with the hands derived from the base layers (root_hands: the
standard rule when the axis hand has trees, the one hand when the root holds exactly one hand; any other root dies —
the generator's --root-hands is the way for it), under a lock on the layer folder (the driver's bundles run side by
side; the index found after the lock wins). A clip the generator cannot measure writes nothing and dies with its error
lines (the generator's CLI writes the other clips and ends rc 1). Marker: `[RENDER_ONLY] <ds>: index missing … the
record builds it …`, then the generator's end line.
"""

from __future__ import annotations

import os

from dexengine.record.scene import render_only_build as ROB
from dexengine.record.scene.render_only import INDEX, MARK, RenderOnlyError, list_dir


def root_hands(ds: str) -> tuple:
    """The hands statement of the record entry (module doc), derived from the hands whose base layer holds trees here:
    () = the standard rule when the axis hand has trees · (hand,) when the root holds exactly that one hand. Any other
    root (no hand, or two or more hands without the axis hand) dies: the record does not guess its hands; the
    generator's --root-hands states them."""
    from dexengine.core import paths as P

    have = tuple(h for h in P.HAND_IDS if ROB.hand_trees(ds, h))
    ax = ROB.axis_hand()
    if ax in have:
        return ()
    if len(have) == 1:
        return have
    raise RenderOnlyError(
        f"{ds}: the hands with a base tree here are {list(have)}, without the axis hand {ax} — the record builds the "
        "render-only index only for a root that holds the axis hand or one hand; state this root's hands to the "
        "generator (python -m dexengine.record.scene.render_only_build --dataset <ds> --root-hands <hands>)"
    )


def ensure_index(ds: str) -> bool:
    """The render-only index of ds (module doc): nothing when the canonical layer has one; else, under a lock on the
    layer folder, build it for the whole dataset with the derived hands (root_hands) and write it. Every clip must be
    measured (an error writes nothing and dies). True = built here."""
    import fcntl

    d = list_dir(ds)
    ip = d / INDEX
    if ip.is_file():
        return False
    made = not d.is_dir()
    d.mkdir(parents=True, exist_ok=True)
    fd = os.open(d, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)  # another record entry building the same index: wait for it
        if ip.is_file():
            print(f"{MARK} {ds}: index written by another record entry while this one waited · {ip}", flush=True)
            return False
        hands = root_hands(ds)
        said = f"root hands {','.join(hands)}" if hands else f"axis hand {ROB.axis_hand()} (standard rule)"
        print(
            f"{MARK} {ds}: index missing {ip} — the record builds it for the whole dataset · {said}, derived from "
            "the base layers",
            flush=True,
        )
        recs = ROB.generate(ds, hands)
        bad = [r["clip"] for r in recs if "error" in r]
        if bad:
            raise RenderOnlyError(
                f"{ds}: {len(bad)} of {len(recs)} clips could not be measured (error lines above, first {bad[:3]}) — "
                "nothing written; fix them or run the generator by hand"
            )
        ROB.write(ds, recs, None)
        print(ROB.summary_line(ds, recs, ROB.run_rc(recs)), flush=True)
    except BaseException:
        if made and not any(d.iterdir()):  # nothing written: leave no empty layer folder behind
            d.rmdir()
        raise
    finally:
        os.close(fd)
    return True
