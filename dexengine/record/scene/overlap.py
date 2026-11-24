"""The distance kernel of the record render's rollout overlap gate (`scene.extras_apply.rollout_overlap`), and rule
S-prox.

Robot gate (`nearest_update`, every user): per render step, the nearest distance from robot surface samples (world)
to each visual-only extra's surface samples (a KD-tree per extra; only samples inside the extra's box grown by the
measure margin are queried, a sparse stride of all samples when none is inside) — the running minimum and its step;
the threshold is the robot clearance `extras.ROLL_CLEAR_M` (unchanged by S-prox).
Rule S-prox (scene rule, rules registry `rule_on`): the gate also measures the **main objects and the extra slot
bodies** of the rollout against the visual-only extras (proposal B5 — a main object or a slot passing through a
visual-only static object went unseen). Trigger = **penetration** deeper than `PEN_TRIGGER_M` (`pen_update`), read on
the static's cooked convex parts (`data.objects.convex_depth`: a mover sample behind every face plane of a part is
inside, depth = the distance to that part's nearest face plane) — the shapes the simulation collides, so a concave
static (a bowl) reads its cavity as outside and no surface normal is read. Same reason (`rollout_overlap`: dropped in
that episode only), row fields `OBJ_FIELDS`; an episode gate, no new collider. Datasets that opt out measure the
robot alone, rows and meta byte-identical.
numpy inside functions only (the render session imports this module before AppLauncher).
"""

from __future__ import annotations

PROX_RULE = "S-prox"
SLOT_PREFIX = "extra_obj_"  # extra slot entity prefix (owner rl.tasks.scene.extra_slots.PREFIX — checked in tests)
#: S-prox trigger (m): a moving object deeper than this inside a visual-only static object is a pass-through. Reason:
#: the E6 evaluation counts a frame as penetrating above 5 mm (its pen_m 0.005), and that penetration trigger is
#: used for objects instead of the robot's 1 cm clearance.
PEN_TRIGGER_M = 0.005
OBJ_FIELDS = ("pen_obj_max_m", "pen_obj_step")


def on(ds: str) -> bool:
    """Is S-prox on for this dataset (scene rule)? A dataset outside the registry is off."""
    from dexengine.data import rules as RR  # noqa: PLC0415

    return str(ds) in RR.ALL and RR.rule_on(PROX_RULE, str(ds))


def is_main(b) -> bool:
    """A main-object bind of the rollout (`object*`)."""
    return b.name.startswith("object")


def is_slot(b) -> bool:
    """An extra slot body bind of the rollout (`extra_obj_*`)."""
    return b.name.startswith(SLOT_PREFIX)


def trees_of(cands: list, pts: dict, origin, margin: float) -> dict:
    """{name: (KD-tree of the extra's world samples, box low − margin, box high + margin)}."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    return {
        r["name"]: (
            cKDTree(pts[r["name"]] + origin),
            pts[r["name"]].min(0) + origin - margin,
            pts[r["name"]].max(0) + origin + margin,
        )
        for r in cands
    }


def nearest_update(W, trees: dict, best: dict, t: int, stride: int) -> None:
    """Running minimum {name: (distance, step)} of world samples W (N,3) against each extra (`trees_of`)."""
    import numpy as np  # noqa: PLC0415

    if not len(W):
        return
    for nm, (tr, lo, hi) in trees.items():
        sel = W[np.all((lo <= W) & (hi >= W), axis=1)]
        d = float(tr.query(sel if len(sel) else W[::stride])[0].min())
        if d < best[nm][0]:
            best[nm] = (d, int(t))


def solids_of(names, parts: dict, frames: dict, origin) -> dict:
    """{name: (convex shape (`convex_depth.shape`), R, c)} — parts = {name: cooked convex parts in the object frame},
    frames = {name: (R (3,3), c (3,))} with world = R p + c + origin (the render pose)."""
    import numpy as np  # noqa: PLC0415

    from dexengine.data.objects import convex_depth as CD  # noqa: PLC0415

    out = {}
    for nm in names:
        R, c = (np.asarray(x, float) for x in frames[nm])
        out[nm] = (CD.shape(parts[nm]), R, c + np.asarray(origin, float))
    return out


def penetration(W, solid) -> float:
    """Depth (m) of the deepest world point of W inside one static's convex parts (`solids_of` entry), 0 if none."""
    import numpy as np  # noqa: PLC0415

    from dexengine.data.objects import convex_depth as CD  # noqa: PLC0415

    sh, R, c = solid
    return CD.depth((np.asarray(W, float) - c) @ R, sh)


def pen_update(W, solids: dict, best: dict, t: int) -> None:
    """Running maximum {name: (depth, step)} of the penetration of world samples W (N,3) into each static."""
    if not len(W):
        return
    for nm, solid in solids.items():
        p = penetration(W, solid)
        if p > best[nm][0]:
            best[nm] = (p, int(t))


def gate_rows(cands: list, best: dict, pen_m: float = PEN_TRIGGER_M) -> None:
    """S-prox result on the rows: fields OBJ_FIELDS · reason rollout_overlap when the deepest penetration exceeds the
    trigger (first reason kept). best = {name: (depth, step)} from `pen_update` (step -1 = never inside)."""
    for r in cands:
        p, t = best[r["name"]]
        r.update(pen_obj_max_m=round(p, 4), pen_obj_step=t)
        if p > pen_m and not r.get("reason"):
            r["reason"] = "rollout_overlap"
