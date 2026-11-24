"""The bodies the f0 object repair measures on one side (rule repair_obj, THE BODIES): the side's own main object, the
other side's main object, every hand-on kinematic slot of the clip and every hand-on pinned still.

A hand-on kinematic slot is an extra row of class SLOT (extra_slots: it moves, so it follows its reference — kinematic)
whose training preset keeps the hand colliding (collision_presets.extra_collide(kind, full=False, ds)["hand"] == ON).
A hand-on pinned still is a STATIC_SPAWN row (pinned at its frame-0 pose) with the same hand cell ON; it is measured in
the shape physics collides it with (collider_shape.row_shape): on triangles by repair_still (exact measure, grid
solve), on the convex ruler as cooked convex pieces of class still. Frames after the f0 window stay S-hand-clear's
(measured only).
Each convex body is measured on its cooked convex pieces (repair_obj.cooked — the shape PhysX collides), posed at each
qpos row by its free joint, in the main object's frame. A body without a collision shape is not measured, and the
side's marker line says so. A side without other bodies gets its main object back unchanged (same pieces, digest and
key).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from dexengine.data.objects import collider_shape as CS
from dexengine.data.objects import extra_slots_geom as XG
from dexengine.retarget.f0 import repair_obj as RO
from dexengine.retarget.f0 import repair_sep as RSEP
from dexengine.retarget.f0 import repair_still as RST

MARK = "[F0_BODIES]"


class BodySet:
    """The main object plus other bodies [(name, class, cooked, qpos_start)] and the pinned stills on triangles
    [repair_still.StillBody]; `at(row)` = one CookedObject of every convex body's pieces in the main object's frame at
    that row, with `bodies` and `body_of` naming each piece's body, and `stills` (pinned: the same at every row)."""

    def __init__(self, main, main_start: int, others: list, digest: str, stills: list = (), row0=None):
        self.main, self.ms, self.others, self.digest = main, main_start, others, digest
        self.stills = list(stills)
        self.row0 = None if row0 is None else np.asarray(row0, float)[None]  # a convex still's pose row: frame 0
        self.articulation = main.articulation

    def at(self, qrow) -> RO.CookedObject:
        base = self.main.at(qrow)
        out = RO.CookedObject.__new__(RO.CookedObject)
        out.digest, out.pieces = self.digest, list(base.pieces)
        out.bodies = [("main", "main")] + [(n, c) for n, c, _o, _s in self.others]
        out.body_of = [0] * len(out.pieces)
        out.stills = self.stills
        row = np.asarray(qrow, float)[None]
        M = XG.pose_mats(row, self.ms)[0]  # the main object's pose (the scene's 7-column pose reader)
        for b, (_n, c, cobj, st) in enumerate(self.others, start=1):
            src = self.row0 if c == "still" else row  # a still stands at its frame-0 pose (pinned)
            rel = np.linalg.inv(M) @ XG.pose_mats(src, st)[0]  # the body in the main object's frame
            moved = [RO._moved(pc, rel[:3, :3], rel[:3, 3]) for pc in cobj.at(src[0]).pieces]
            out.pieces += moved
            out.body_of += [b] * len(moved)
        return out


def hand_on_slots(rows: list, ti: dict, dataset: str) -> list:
    """The clip's hand-on kinematic slots: [(objects[] entry, kind)] of the SLOT rows (`rows` = the tree's
    extra_slots.clip_rows) whose training preset keeps the hand colliding (module doc)."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415
    from dexengine.rl.tasks.scene import collision_presets as CP  # noqa: PLC0415

    objs = {str(o["name"]): o for o in ti["objects"] if o.get("qpos_start") is not None}
    out = []
    for r in rows:
        if r.get("cls") != XS.SLOT or not r.get("kind") or str(r["name"]) not in objs:
            continue
        if CP.extra_collide(r["kind"], full=False, ds=dataset)["hand"] == "ON":
            out.append((objs[str(r["name"])], r["kind"]))
    return out


def hand_on_stills(rows: list, ti: dict, dataset: str) -> list:
    """The clip's hand-on pinned stills: [(objects[] entry, extra_slots row)] of the STATIC_SPAWN rows (`rows` = the
    tree's extra_slots.clip_rows) whose training preset keeps the hand colliding (module doc)."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415
    from dexengine.rl.tasks.scene import collision_presets as CP  # noqa: PLC0415

    objs = {str(o["name"]): o for o in ti["objects"] if o.get("qpos_start") is not None}
    out = []
    for r in rows:
        if r.get("cls") != XS.STATIC_SPAWN or not r.get("kind") or str(r["name"]) not in objs:
            continue
        if CP.extra_collide(r["kind"], full=False, ds=dataset)["hand"] == "ON":
            out.append((objs[str(r["name"])], r))
    return out


def _still_bodies(tree: Path, q: np.ndarray, dataset: str, rows: list) -> tuple[list, list, list]:
    """(StillBody on triangles, [(name, "still", cooked, qpos_start)] on the convex ruler, lost) of the hand-on stills,
    each at its frame-0 pose (the env pins it there)."""
    from dexengine.data.objects import extra_slots_geom as XGM  # noqa: PLC0415
    from dexengine.data.objects import touch_gate as TG  # noqa: PLC0415 — the still's body in its shape

    tri, conv, lost = [], [], []
    for o, r in rows:
        ruler, why = TG.static_body(dataset, o, r)
        if ruler is None:
            lost.append(f"{o['name']}(still {r['kind']}): {why}")
        elif ruler.kind == CS.TRIANGLES:
            tri.append(
                RST.StillBody(str(o["name"]), str(r["kind"]), ruler, XGM.pose_mats(q[:1], int(o["qpos_start"]))[0])
            )
        else:
            c, why = RO.cooked(o, dataset, tree)
            if c is None:
                lost.append(f"{o['name']}(still {r['kind']}): {why}")
            else:
                conv.append((str(o["name"]), "still", c, int(o["qpos_start"])))
    return tri, conv, lost


_SLOTS: dict = {}


def with_bodies(main, o: dict, tree: Path, side: str, dataset: str):
    """The side's measured bodies: `main` itself when the clip has no other body, else a BodySet with the other side's
    main object (class main) and the hand-on kinematic slots (class slot). Prints one marker line per side."""
    tree = Path(tree)
    ti = json.loads((tree / "task_info.json").read_text())
    if str(tree) not in _SLOTS:
        q = RSEP.applied(tree, np.load(tree / "0/trajectory_kinematic.npz")["qpos"])  # the rows f0 reads
        rows = RSEP.scene_rows(tree, ti, q, echo=True)  # the slots and the stills read it (repair_sep's rows if same q)
        stills = _still_bodies(tree, q, dataset, hand_on_stills(rows, ti, dataset))
        _SLOTS[str(tree)] = (hand_on_slots(rows, ti, dataset), stills, q[0])
    slots, (stills, conv_stills, lost), row0 = _SLOTS[str(tree)]
    mains = [x for x in ti["objects"] if x.get("role") == "main" and x.get("qpos_start") is not None]
    cands = [(x, "main", "other_main") for x in mains if int(x["qpos_start"]) != int(o["qpos_start"])]
    cands += [(x, "slot", kind) for x, kind in slots]
    others, names, lost = [], [], list(lost)
    for x, cls, kind in cands:
        c, why = RO.cooked(x, dataset, tree)
        if c is None:
            lost.append(f"{x['name']}({kind}): {why}")
            continue
        others.append((str(x["name"]), cls, c, int(x["qpos_start"])))
        names.append(f"{x['name']}:{cls}({kind})")
    others += conv_stills
    names += [f"{n}:still(convex)" for n, _c, _o, _s in conv_stills] + [f"{b.name}:still({b.kind})" for b in stills]
    miss = f" · not measured {lost}" if lost else ""
    print(f"{MARK} {tree.name} [{side}] main {o['name']} + {names or '-'}{miss}", flush=True)
    if not others and not stills:
        return main
    key = "|".join(
        [main.digest]
        + [f"{n}/{c}/{cobj.digest}" for n, c, cobj, _s in others]
        + [f"{b.name}/still/{b.digest}" for b in stills]
    )
    return BodySet(main, int(o["qpos_start"]), others, hashlib.sha256(key.encode()).hexdigest()[:16], stills, row0)
