"""Objects the env spawns fixed at frame 0 (support objects · still extras STATIC_SPAWN): their rows, outlines,
robot-spot overlap, pedestals and arm-guard obstacles.

Rule     which placement is rejected and which object is dropped from the spawn is the **put-down rule** (place_rule):
  must place     one that receives a main object (coords.receives_main · a support object by statement) or a still
                 extra touched for place_rule.GRAZE_MAX_S or longer. It cannot leave the scene → a placement whose
                 outline overlaps the robot spot (coords.robot_keepout) or whose off-desk area exceeds the threshold
                 (off_desk) is rejected by the judge (place_rule.verdict).
  grazed extra   not spawned in a placement where it violates (spawned_at · extra_slots uses place_rule.drop_why) — it
                 does not narrow the placement.
This module holds the rows that rule reads (columns), the outline, the robot-spot overlap measure, and — in the
pedestal rule's scope — the pedestals and the arm-guard obstacles (out of scope or aerial: fixed_objects has no rows).
Arm-guard obstacles: the fixed objects' bodies and their pedestals (from the desk top, or from the floor off the desk
box), measured by the arm guard (support_surface.scene_supports). On a pre-placement tree the judge re-decides per
candidate placement (at); a baked tree reads the same column_base · spawned_at at its own place.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _coords():
    from dexengine.placement.coords import core as C  # noqa: PLC0415 — robot spot · receiving · fixed spawn list

    return C


FIXED_TAG = "[DESK_FIXED]"
#: Outline sample spacing (m) — convex hull edges split at this spacing measure the robot-spot overlap (small next to
#: the spot radius).
FIXED_EDGE_M = 0.01


def _outline(xy: np.ndarray) -> np.ndarray:
    """The convex hull outline of xy points split at FIXED_EDGE_M spacing (M,2) — overlap with a disc only needs the
    outline (an inner point never touches before the outline · the hull centre is added for a hull that contains the
    whole disc)."""
    from scipy.spatial import ConvexHull  # noqa: PLC0415

    xy = np.asarray(xy, float)
    try:
        h = xy[ConvexHull(xy).vertices]
    except Exception:  # noqa: BLE001 — degenerate (a line · a point): the points themselves
        return xy
    out = [h.mean(0)[None, :]]
    for a_, b_ in zip(h, np.roll(h, -1, axis=0), strict=True):
        n = max(1, int(np.ceil(np.linalg.norm(b_ - a_) / FIXED_EDGE_M)))
        out.append(a_[None, :] + (b_ - a_)[None, :] * (np.arange(n) / n)[:, None])
    return np.concatenate(out, 0)


def columns(tree_dir, ti: dict, q, dz: float, mains_v: list, receives: bool = False) -> list:
    """One frame-0 row per fixed object (coords.fixed_spawned) — pedestals, obstacles and the put-down rule all read
    it: {name, role, pose (frame 0 · capture x·y·z · wxyz), verts_xy (N,2 capture), xy_cap (outline), row (pedestal disc
    row — coords._pedestal_row), high (floor − desk ≥ coords.HIGH_REST_M), stands (stands on another spawned object →
    no pedestal), receives (receives a main object — measured only when receives=True · a support object by
    statement), touch_s (place_rule.touch_s), must (must place = not grazed — place_rule.grazed: receives, is a receiver
    extra (extra_slots RECEIVER), or was touched long)}. The scope is the caller's."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415 — receiver extras (RECEIVER)
    from dexengine.placement.scene import place_rule as PR  # noqa: PLC0415 — graze threshold · touch time

    C = _coords()
    names = C.fixed_spawned(tree_dir, ti, q)
    if not names:
        return []
    slot_recv = {
        str(r["name"])
        for r in XS.clip_rows(str(tree_dir), ti, q, strict=False)
        if r["cls"] == XS.STATIC_SPAWN and r.get("why") == XS.RECEIVER
    }
    placed, _ = C._placers(tree_dir, ti, q)
    by = {str(o.get("name")): o for o in ti.get("objects") or []}
    Vx = {n: placed(by[n], 0) for n in names}
    Tb = C._base_T(ti)
    out = []
    for n in names:
        V, s0 = Vx[n], int(by[n]["qpos_start"])
        lower = list(mains_v) + [v for k, v in Vx.items() if k != n]
        recv = by[n].get("role") == "support" or (receives and C.receives_main(tree_dir, ti, q, V, placed))
        sec = PR.touch_s(ti, n)
        out.append(
            {
                "name": n,
                "role": by[n].get("role"),
                "pose": [round(float(v), 6) for v in q[0, s0 : s0 + 7]],
                "verts_xy": V[:, :2].copy(),
                "xy_cap": _outline(V[:, :2]),
                "row": C._pedestal_row(V, n, 0, Tb, "desk"),
                "high": float(V[:, 2].min()) - float(dz) >= C.HIGH_REST_M,
                "stands": any(rests_on(V, w) for w in lower),
                "receives": recv,
                "touch_s": sec,
                "must": not PR.grazed(sec, recv or n in slot_recv),
            }
        )
    return out


def rests_on(V: np.ndarray, W: np.ndarray) -> bool:
    """Does a fixed object (vertices V · capture) rest on a **lower object** W — the one meaning of "stands on another
    spawned object → no pedestal" (columns' `stands`). All three must hold:
      ① W's floor is at least coords.HIGH_REST_M below V's — floors within one band are the same surface: a neighbour
         beside it, or a thin tray holding V (floor difference < band), is not under it and V gets its own pedestal
      ② among W's vertices inside V's xy box (± HIGH_REST_M) and no higher than V's floor + coords.START_EMBED_M, the
         highest is within HIGH_REST_M of V's floor — W's surface touches right under V (also catches a sparse box)
      ③ W's mean height is below V's."""
    C = _coords()
    V, W = np.asarray(V, float), np.asarray(W, float)
    fl = float(V[:, 2].min())
    if float(W[:, 2].min()) > fl - C.HIGH_REST_M or float(W[:, 2].mean()) >= float(V[:, 2].mean()):
        return False
    lo, hi = V[:, :2].min(0) - C.HIGH_REST_M, V[:, :2].max(0) + C.HIGH_REST_M
    under = W[np.all((W[:, :2] >= lo) & (W[:, :2] <= hi), axis=1)]
    under = under[under[:, 2] <= fl + C.START_EMBED_M]
    return bool(len(under)) and fl - float(under[:, 2].max()) <= C.HIGH_REST_M


def _scene_xy(o: dict, place) -> np.ndarray:
    """Scene xy of a fixed object's vertices at placement `place` (capture dx · dy · yaw · pivot — the bake's formula
    coords.place_objects). place None = the tree's own place (a baked tree)."""
    C = _coords()
    return C.capture_to_scene(o["verts_xy"] if place is None else C.place_objects(o["verts_xy"], *place))


def spawned_at(o: dict, keepout, place, placed: bool, box=None) -> bool:
    """Does the env spawn this fixed object at that placement — a grazed still extra (o["must"] false) whose outline
    overlaps the robot spot or whose off-desk area exceeds the threshold is not spawned (place_rule.hit — the same
    formula as place_rule.drop_why on a baked tree). A must-place object is always spawned (the judge rejects such
    placements). placed false (pre-placement tree · no placement) = no scene place, so spawned."""
    if bool(o["must"]) or not placed:
        return True
    from dexengine.placement.scene import place_rule as PR  # noqa: PLC0415

    return not PR.hit(o["xy_cap"], place, keepout, box)


def column_base(o: dict, box, place, placed: bool) -> str | None:
    """The foot of the pedestal under this fixed object at that placement — None (none) · "desk" (from the desk top) ·
    "floor" (from the floor). **One formula for the judge (candidate placements) and the bake stamp (baked tree)**:
    stands on another spawned object → none · more than static_desk.OVERHANG_MAX_FRAC of its top-view area outside the
    desk box → floor · floor at least a band above the desk → desk. placed false = no place to measure off-desk."""
    if o["stands"]:
        return None
    if placed and off_desk(_scene_xy(o, place), box):
        return "floor"
    return "desk" if o["high"] else None


def off_desk(xy_scene: np.ndarray, box) -> bool:
    """Is more than static_desk.OVERHANG_MAX_FRAC of the top-view outline **area** of scene xy vertices outside the
    desk box — the put-down rule's measure and threshold (static_desk.overhang_frac)."""
    from dexengine.placement.scene import static_desk as SD  # noqa: PLC0415 — overhang measure · threshold

    return SD.overhang_frac(xy_scene, box=box) > SD.OVERHANG_MAX_FRAC


def pedestal_rows(tree_dir, ti: dict, q, dz: float, mains_v: list) -> list:
    """The fixed-object pedestal rows the bake stamp reads (the body of coords.static_pedestals) — column_base at the
    tree's own place."""
    C = _coords()
    placed = C.is_baked_tree(ti)
    box = C.desk_bounds_scene_for(tree_dir, margin=0.0) if placed else None
    return [
        o["row"] | {"base": b}
        for o in columns(tree_dir, ti, q, dz, mains_v)
        if (b := column_base(o, box, None, placed))
    ]


def _in_scope(tree_dir, ti: dict) -> bool:
    C = _coords()
    if not any(o.get("role") != "main" for o in ti.get("objects") or []):  # mains only — do not open the trajectory
        return False
    return not (C.scene_mode(tree_dir) == "aerial" or ti.get("scene") == "aerial" or not C.pedestal_scene(ti))


def fixed_objects(tree_dir, ti: dict | None = None, q=None) -> dict:
    """The statement of the objects the env spawns fixed at frame 0 (coords.fixed_spawned) → {keepout: (cx, cy, r)
    scene, box: desk box, dz, placed: baked tree, objs: [columns rows — with receives]}. Desk scenes in the pedestal
    rule's scope (coords.pedestal_scene) only — out of scope or aerial: objs [] (readers skip it). Read by the judge
    mask (receives · xy_cap), the arm-guard obstacles (support_surface — bodies · pedestals · per-placement `at`) and
    the put-down rule. must = must place (receives · touched long): the put-down rule rejects placements where it
    violates; a grazed still extra is not spawned there (spawned_at)."""
    C = _coords()
    tree_dir = Path(tree_dir)
    if ti is None:
        ti = json.loads((tree_dir / "task_info.json").read_text(encoding="utf-8"))
    out = {"keepout": None, "box": None, "dz": None, "placed": C.is_baked_tree(ti), "objs": []}
    if not _in_scope(tree_dir, ti):
        return out
    if q is None:
        q = np.load(tree_dir / "0/trajectory_kinematic.npz")["qpos"]
    dz = float(C._qpos_layout_mod().desk_z_of(ti, str(tree_dir))[0])
    placed, _ = C._placers(tree_dir, ti, q)
    mains_v = [
        placed(o, 0)
        for o in ti.get("objects") or []
        if o.get("role") == "main" and o.get("mesh_dir") and o.get("qpos_start") is not None
    ]
    objs = columns(tree_dir, ti, q, dz, mains_v, receives=True)
    if objs:
        out.update(
            keepout=C.robot_keepout(tree_dir), box=C.desk_bounds_scene_for(tree_dir, margin=0.0), dz=dz, objs=objs
        )
    return out


def at(fx: dict, place) -> list:
    """Per fixed object its state at that placement [(row, spawned, pedestal foot | None)] — fx = fixed_objects'
    return · place = a candidate placement (capture dx · dy · yaw · pivot) or None (the baked tree itself). An object
    not spawned has no pedestal."""
    placed = place is not None or bool(fx["placed"])
    out = []
    for o in fx["objs"]:
        up = spawned_at(o, fx["keepout"], place, placed, fx["box"])
        out.append((o, up, column_base(o, fx["box"], place, placed) if up else None))
    return out


def keepout_hit(xy_scene: np.ndarray, keepout) -> bool:
    """Does any scene xy point lie inside the robot's standing spot (disc) — the final check."""
    return _coords().in_robot_keepout(xy_scene, keepout)


def grid_keepout_ok(ww: np.ndarray, gx: np.ndarray, gy: np.ndarray, keepout) -> np.ndarray:
    """Judge mask — ww (N,2) scene xy (already rotated) · cell offsets gx (X,) · gy (Y,) → (X,Y) bool: per cell no point
    of ww + (gx, gy) inside the robot disc. The same value as the final check (keepout_hit)."""
    cx, cy, r = keepout
    dx2 = (ww[:, 0:1] + gx[None, :] - cx) ** 2  # (N,X)
    dy2 = (ww[:, 1:2] + gy[None, :] - cy) ** 2  # (N,Y)
    return (dx2[:, :, None] + dy2[:, None, :]).min(0) >= r * r
