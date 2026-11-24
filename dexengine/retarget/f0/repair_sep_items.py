"""The bodies of rule 1d (repair_sep): the collision-on Items of a tree at frame 0 (main objects, supports and the
spawned extra rows whose objects cell is ON), the scene rows they are read from (once per process) and their true-shape
depth. repair_sep re-exports these names (the same objects); the rule itself stays there."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from dexengine.core import stdio
from dexengine.data.objects import collider_shape as CS
from dexengine.data.objects import touch_gate as TG
from dexengine.retarget.f0 import repair_obj as RO


class Item:
    """One collision-on body of the tree: name, class (main · still · slot · support), kind, qpos column, ruler
    (touch_gate.Body for a main object, a collider_shape ruler otherwise), pinned at frame 0, digest, side."""

    def __init__(self, name, cls, kind, qs, ruler, pinned, digest, side=""):
        self.name, self.cls, self.kind, self.qs = name, cls, kind, qs
        self.ruler, self.pinned, self.digest, self.side = ruler, pinned, digest, side
        self.art = getattr(ruler, "art", None)

    def world(self, q, T, f, move=None) -> list:
        """[(ruler, 4x4 world pose)] at frame f; `move` (3,) translates the body."""
        M = (T[0] if self.pinned else T[f]).copy()
        if move is not None:
            M[:3, 3] += move
        if self.cls != "main":
            return [(self.ruler, M)]
        return [(sh, M @ L) for sh, L in self.ruler.links(q[f])]


def _first(a: list, b: list) -> tuple:
    """The pair_depth order: a triangle ruler reads first (its surface samples near the other body; touch_gate's
    order)."""
    return (b, a) if any(r.kind == CS.TRIANGLES for r, _M in b) else (a, b)


def depth(a: Item, b: Item, q, poses: dict, f: int, ma=None, mb=None) -> float:
    """True-shape depth (m) of two bodies at frame f, each moved by its move (or not): collider_shape.pair_depth over
    their link pairs, a bounding-sphere test first."""
    la, lb = _first(a.world(q, poses[a.name], f, ma), b.world(q, poses[b.name], f, mb))
    best = 0.0
    for ra, Ta in la:
        ca = Ta[:3, :3] @ ra.center + Ta[:3, 3]
        for rb, Tb in lb:
            if np.linalg.norm(Tb[:3, :3] @ rb.center + Tb[:3, 3] - ca) > ra.radius + rb.radius:
                continue
            best = max(best, CS.pair_depth(ra, rb, np.linalg.inv(Ta) @ Tb))
    return best


def _main_item(ds: str, o: dict, tree: Path) -> tuple:
    """(Item | None, why) of a main object (touch_gate.body_of; the digest of its cooked parts or articulated links)."""
    body, why = TG.body_of(ds, o, tree)
    if body is None:
        return None, why
    dig = RO.cooked(o, ds, tree)[0].digest if body.art is not None else CS.parts(ds, str(o["obj_id"])).digest
    return Item(str(o["name"]), "main", "main", int(o["qpos_start"]), body, False, dig, str(o.get("side") or "")), ""


def _support_item(ds: str, o: dict) -> tuple:
    from dexengine.data.objects import hull as OH  # noqa: PLC0415 — owner of the collision shapes

    try:
        r = CS.body(ds, str(o["obj_id"]), CS.shape_of(ds, CS.SUPPORT, True))
    except OH.HullMissing as e:
        return None, f"no collision shape: {str(e)[:80]}"
    return Item(str(o["name"]), "support", CS.SUPPORT, int(o["qpos_start"]), r, True, r.digest), ""


def _row_item(ds: str, o: dict, row: dict) -> tuple:
    """(Item | None, why) of a spawned extra row whose objects cell is ON (None, '' for a row that does not collide
    with the main objects)."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415
    from dexengine.rl.tasks.scene import collision_presets as CP  # noqa: PLC0415

    if row.get("cls") not in XS.SLOTTED or not row.get("kind"):
        return None, ""
    if CP.extra_collide(row["kind"], full=False, ds=ds)["objects"] != "ON":
        return None, ""
    r, why = TG.static_body(ds, o, row)  # the row's body in its shape (collider_shape.row_body)
    if r is None:
        return None, why
    cls = "still" if row.get("static") else "slot"
    return Item(str(o["name"]), cls, str(row["kind"]), int(o["qpos_start"]), r, cls == "still", r.digest), ""


#: extra_slots.clip_rows of the trees this process read last, with what the scene rules printed while computing them.
#: `bodies` here and repair_bodies.with_bodies read the rows of the same tree, task_info and qpos rows.
_ROWS: dict = {}
_ROWS_MAX = 8


def scene_rows(tree, ti: dict, q: np.ndarray, echo: bool) -> list:
    """extra_slots.clip_rows(tree, ti, q, strict=False, kinds_only=True), computed once per process for the same tree
    path, task_info and qpos rows; each call gets its own deep copy. f0 reads the rows' classes and kinds only: the
    envelope overlap (a recorded measure and the clip drop of step 2f, no class or kind) is not measured and the touch
    gate stops a failing pair at its first frame over the true-shape overlap (the same kind). The scene
    rules print their own markers: `echo` writes them to this thread's stdout as a direct call would, else they are
    dropped as under stdio.capture."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415

    qa = np.ascontiguousarray(q)
    key = (
        str(tree),
        hashlib.sha1(json.dumps(ti, sort_keys=True, default=str).encode()).hexdigest(),
        hashlib.sha1(qa.tobytes()).hexdigest(),
        qa.shape,
        qa.dtype.str,
    )
    if key not in _ROWS:
        if len(_ROWS) >= _ROWS_MAX:
            _ROWS.clear()
        with stdio.capture() as buf:
            rows = XS.clip_rows(str(tree), ti, q, strict=False, kinds_only=True)
        _ROWS[key] = (rows, buf.getvalue())
    rows, text = _ROWS[key]
    if echo and text:
        sys.stdout.write(text)
        sys.stdout.flush()
    return copy.deepcopy(rows)


def bodies(tree: Path, ti: dict, q: np.ndarray, ds: str) -> tuple[list, list]:
    """([Item], [unmeasured 'name(class): why']) of the tree's collision-on bodies (repair_sep module doc)."""
    objs = [o for o in ti["objects"] if o.get("qpos_start") is not None]
    rows = {str(r["name"]): r for r in scene_rows(tree, ti, q, echo=False)}  # the markers dropped, as before
    items, lost = [], []
    for o in objs:
        role, name = o.get("role"), str(o["name"])
        if role == "main":
            it, why = _main_item(ds, o, tree)
        elif role == "support":
            it, why = _support_item(ds, o)
        elif name in rows:
            it, why = _row_item(ds, o, rows[name])
        else:
            continue
        if it is not None:
            items.append(it)
        elif why:
            lost.append(f"{name}({role}): {why}")
    return items, lost


def _depth_within(a: Item, b: Item, q, poses: dict, f: int, tol: float, ma=None) -> float | None:
    """`depth` when it is at most tol, else None: the same link pairs and sphere test, each read by
    collider_shape.pair_depth_within (a link pair above tol ends the reading)."""
    la, lb = _first(a.world(q, poses[a.name], f, ma), b.world(q, poses[b.name], f))
    best = 0.0
    for ra, Ta in la:
        ca = Ta[:3, :3] @ ra.center + Ta[:3, 3]
        for rb, Tb in lb:
            if np.linalg.norm(Tb[:3, :3] @ rb.center + Tb[:3, 3] - ca) > ra.radius + rb.radius:
                continue
            d = CS.pair_depth_within(ra, rb, np.linalg.inv(Ta) @ Tb, tol)
            if d is None:
                return None
            best = max(best, d)
    return best
