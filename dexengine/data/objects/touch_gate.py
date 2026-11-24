"""Rule S-touch-gate: a still extra a hand touched collides with the main objects; a recorded motion that passes
through it deeper than the true-shape overlap drops the clip.

Why: a touched static spawn (extra_slots kind touched) stood with the main objects OFF, so a main object the policy
moves went through it (a cup through the bowl it was touched against). The rule: a real overlap above
scene_requirement.TRUE_SHAPE_OVERLAP_M is dropped, never kept with the collision off; below it the pair collides (the
frame-0 object separation moves the main object out at the start, physics handles the rest).

Measurement, per (main object, touched static) pair, on the shapes physics collides (collider_shape: the static as its
pinned shape — triangles where S-true-shape is on — and the main object as its cooked convex parts, read both ways,
`collider_shape.pair_depth`):
  pair     the largest depth over all tree frames, the static at its frame-0 pose (it is pinned there)
Gate: every pair at or below TRUE_SHAPE_OVERLAP_M → kind touched_solid (objects ON). A pair above it → the row keeps
kind touched and states `overlap_drop` {partner, depth_m}: the clip is dropped (`collider_shape.overlap_drop`, read by
the stage-1 step 2f). A row turns ON only when every measured pair passes, because the slot authoring filters a slot
against every main object at once.
Bodies (`body_of`, also read by S-envelope): a moving body as the simulation collides it — a rigid body's cooked parts,
an articulated main object's every link posed at each frame's joint values (the f0 articulated model,
`retarget.f0.repair_obj.cooked` — one owner of that shape). A body that cannot be measured stays OFF with the reason
(`why=unmeasured`, pair field `unmeasured`).
"""

from __future__ import annotations

import numpy as np

from dexengine.data.objects import collider_shape as CS
from dexengine.data.objects import extra_slots_geom as XG
from dexengine.data.objects import scene_requirement as SR

RULE_ID = "S-touch-gate"
FRAME_STRIDE = 1  # every tree frame is measured
MEMO: dict = {}


class Body:
    """A moving body as the simulation collides it: link rulers (collider_shape.Parts, link frame) and `links(qrow)` =
    [(ruler, 4x4 link pose in the body frame)] at a qpos row (one link at the identity for a rigid body)."""

    def __init__(self, shapes: list, art=None):
        self.shapes, self.art = shapes, art

    def links(self, qrow) -> list:
        if self.art is None:
            return [(self.shapes[0], np.eye(4))]
        out = []
        for sh, (_c, R, p) in zip(self.shapes, self.art.links_at(qrow), strict=True):
            L = np.eye(4)
            L[:3, :3], L[:3, 3] = R, p
            out.append((sh, L))
        return out


def body_of(ds: str, o: dict, tree_dir) -> tuple[Body | None, str]:
    """(Body, '') of an object as the training scene collides it (module doc), or (None, why) when it has no cooked
    parts."""
    if o.get("articulation") and o.get("role") == "main":
        from dexengine.retarget.f0 import repair_obj as RO  # noqa: PLC0415 — owner of the articulated shape

        art, why = RO.cooked(o, ds, tree_dir)
        if art is None:
            return None, why
        return Body([CS.Parts([pc[0] for pc in c.pieces]) for c, _R, _p in art.links_at(_rest(art))], art), ""
    parts = parts_of(ds, o)
    return (Body([CS.Parts(parts)]), "") if parts else (None, "no cooked parts")


def body_inputs(ds: str, o: dict, tree_dir) -> list:
    """The data-root files `body_of` (and `parts_of`) open for an object besides its USD: the cooked-hull dump of the
    training USD (hull.cooked_dump; only when present — a USD without one is read as having no collision mesh) and, for
    an articulated main object, the tree's scene.xml its links are read from (repair_obj.scene_file); the link meshes
    are the tree's object files (coords.object_files)."""
    from dexengine.data.objects import hull as _oh  # noqa: PLC0415

    art = bool(o.get("articulation")) and o.get("role") == "main"
    dump = _oh.cooked_dump(_oh.usd_file(ds, _oh.asset_id(o) if art else str(o["obj_id"]), art))
    out = [dump] if dump.is_file() else []
    if art:
        from dexengine.retarget.f0 import repair_obj as RO  # noqa: PLC0415

        out.append(RO.scene_file(tree_dir))
    return out


def _rest(art) -> np.ndarray:
    """A qpos row long enough for every part column (the link shapes do not depend on it)."""
    return np.zeros(max(col for _s, col, _c in art.parts) + 1)


def pair_depth(
    main: Body, q: np.ndarray, T_main: np.ndarray, static, T_static: np.ndarray, stop_above: float | None = None
) -> float:
    """Largest recorded overlap between a moving body (`main`, poses (T,4,4), joint values from the qpos rows `q`) and
    a pinned one (ruler `static`, pose (4,4)), both ways (`collider_shape.pair_depth`), over every FRAME_STRIDE-th
    frame. stop_above: return at the first frame that takes the largest value above it (the value is then a lower
    bound of the largest, already above stop_above; the frames before it are read as without it)."""
    inv_s = np.linalg.inv(T_static)
    best = 0.0
    for f in range(0, len(T_main), FRAME_STRIDE):
        for sh, L in main.links(q[f]):
            best = max(best, CS.pair_depth(static, sh, inv_s @ T_main[f] @ L))
        if stop_above is not None and best > stop_above:
            return best
    return best


def static_body(ds: str, o: dict, row: dict):
    """(ruler, '') of a still row's body in the shape physics uses (`collider_shape.row_body`), or (None, why)."""
    from dexengine.data.objects import hull as _oh  # noqa: PLC0415

    try:
        return CS.row_body(ds, o, row), ""
    except _oh.HullMissing as e:
        return None, f"no collision shape: {str(e)[:120]}"


def parts_of(ds: str, o: dict) -> list | None:
    """Cooked convex parts of an object's rigid USD (body frame; `collider_shape.parts`, every link) or None when the
    object has none (HullMissing: no collision mesh in its USD)."""
    from dexengine.data.objects import hull as _oh  # noqa: PLC0415 — owner of the sim collision shapes

    try:
        return CS.parts(ds, str(o["obj_id"])).pieces or None
    except _oh.HullMissing:
        return None


def gate_row(tree_dir, ti: dict, q: np.ndarray, row: dict, mains: list[dict], kinds_only: bool = False) -> dict:
    """The gate statement of one touched static row: {on, pairs: {main: {depth_m | None, pass[, unmeasured]
    [, depth_at_least]}}, why, shape, drop}. on = every pair passes (and there is at least one); drop = the deepest pair
    above the true-shape overlap ({partner, depth_m}) or None. kinds_only (a reader of the kind only, f0): a pair stops
    at its first frame over the overlap (pair_depth stop_above) — the same pass, on and kind; a failing pair's depth_m
    is then a lower bound (`depth_at_least`) and drop names the deepest of those bounds."""
    ds = str(ti["dataset_name"])
    objs = {str(o["name"]): o for o in ti.get("objects") or []}
    o = objs[row["name"]]
    st, why_s = static_body(ds, o, row)  # a static is a slot: the rigid USD it spawns, in its pinned shape
    T0 = XG.pose_mats(q[:1], int(o["qpos_start"]))[0]
    pairs = {}
    for m in mains:
        sh, why_m = body_of(ds, m, tree_dir) if st is not None else (None, why_s)
        if sh is None:
            pairs[str(m["name"])] = {"depth_m": None, "pass": False, "unmeasured": why_m}
            continue
        cols = q if sh.art is not None else q[:, int(m["qpos_start"]) : int(m["qpos_start"]) + 7]
        key = (
            ds,
            str(o["obj_id"]),
            st.kind,
            str(m["obj_id"]),
            hash(np.ascontiguousarray(cols).tobytes()),
            hash(T0.tobytes()),
            kinds_only,
        )
        if key not in MEMO:
            if len(MEMO) >= 256:
                MEMO.clear()
            stop = SR.TRUE_SHAPE_OVERLAP_M if kinds_only else None
            MEMO[key] = pair_depth(sh, q, XG.pose_mats(q, int(m["qpos_start"])), st, T0, stop_above=stop)
        d = MEMO[key]
        pairs[str(m["name"])] = {"depth_m": round(d, 4), "pass": d <= SR.TRUE_SHAPE_OVERLAP_M}
        if kinds_only and not pairs[str(m["name"])]["pass"]:
            pairs[str(m["name"])]["depth_at_least"] = True
    on = bool(pairs) and all(p["pass"] for p in pairs.values())
    why = (
        "no main object"
        if not pairs
        else "unmeasured"
        if any(p["depth_m"] is None for p in pairs.values())
        else "true-shape overlap"
        if not on
        else "clear"
    )
    over = [(p["depth_m"], n) for n, p in pairs.items() if p["depth_m"] is not None and not p["pass"]]
    drop = None if not over else {"partner": max(over)[1], "depth_m": max(over)[0]}
    return {"on": on, "pairs": pairs, "why": why, "shape": None if st is None else st.kind, "drop": drop}


def main_objects(ti: dict) -> list[dict]:
    """The main objects a touched static is measured against: role main with a mesh and a pose column."""
    return [
        o
        for o in ti.get("objects") or []
        if o.get("role") == "main" and o.get("mesh_dir") and o.get("qpos_start") is not None
    ]
