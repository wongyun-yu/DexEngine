"""Support discs: the stamp vocabulary (disc · pedestal, collide · visibility · owner) and the rest-interval discs."""

from __future__ import annotations

import numpy as np

from dexengine.placement.coords.contacts import _held_in_interval, frame_toucher
from dexengine.placement.coords.frames import capture_to_scene
from dexengine.placement.coords.objects import _base_T, _base_to_tree, _placers, _rest_rule
from dexengine.placement.coords.thresholds import (
    DISC_COLLIDE,
    DISC_VISIBLE,
    PEDESTAL_COLLIDE,
    PEDESTAL_RULE,
    PEDESTAL_VISIBLE,
)


def pedestal_scene(ti: dict, aerial: bool | None = None) -> bool:
    """This tree's desk-scene supports are pedestals — a desk scene (not aerial) of a dataset in the pedestal rule's
    scope (rules registry). The dataset is the tree's `dataset_name`; none raises."""
    if aerial if aerial is not None else ti.get("scene") == "aerial":
        return False
    ds = ti.get("dataset_name")
    if not ds:
        raise ValueError("task_info has no dataset_name — cannot decide the pedestal rule's scope")
    from dexengine.data import rules as RR  # noqa: PLC0415

    scope = RR.rule_datasets(PEDESTAL_RULE)
    return not scope or str(ds) in scope


def disc_collide_of(ti: dict, aerial: bool | None = None, kind: str | None = None) -> str:
    """The disc stamp `collide` a support of `kind` carries in this tree — the pedestal's (PEDESTAL_COLLIDE[kind]) in a
    pedestal scene, else the objects-only disc. A pedestal scene needs the kind — none or an unknown one raises."""
    if not pedestal_scene(ti, aerial):
        return DISC_COLLIDE
    if kind not in PEDESTAL_COLLIDE:
        raise ValueError(f"pedestal kind {kind!r} — only {sorted(PEDESTAL_COLLIDE)}")
    return PEDESTAL_COLLIDE[kind]


def disc_visible(collide) -> bool:
    """Render/camera visibility of a stamped disc — the aerial disc (objects only) is invisible, the desk pedestal is
    visible. An unknown stamp raises."""
    vis = {DISC_COLLIDE: DISC_VISIBLE} | dict.fromkeys(PEDESTAL_COLLIDE.values(), PEDESTAL_VISIBLE)
    if collide not in vis:
        raise ValueError(f"disc stamp collide={collide!r} — only {sorted(vis)}")
    return vis[collide]


def disc_owner_side(ti: dict, d: dict) -> str:
    """The side whose **main** object support disc `d` carries: "right" · "left" · "both" (one shared object) · "static"
    (a fixed object's pedestal). The disc's `obj` is a main object's layout name; the side is that object's
    `objects[].side`, never the disc's own `side` (a rest interval's hand, which a re-role can change). No main of that
    name → raises."""
    name = str(d.get("obj"))
    got = [o for o in (ti.get("objects") or []) if o.get("role") == "main" and str(o.get("name")) == name]
    if d.get("kind") == "static":  # a fixed object's pedestal carries no main object — no side owns it
        return "static"
    sides = {str(o.get("side")) for o in got}
    if len(sides) != 1 or not sides <= {"right", "left", "both"}:
        raise ValueError(
            f"cannot decide the main-object side of support disc obj={name!r} — {len(got)} main objects "
            f"in objects[] · sides {sorted(sides)}"
        )
    return sides.pop()


def disc_stamp(kind: str, collide: str = DISC_COLLIDE) -> dict:
    """The stamp fields every support disc carries — one constructor so both scenes write the same keys: kind "start"
    (a start support) | "rest" (a rest interval) | "static" (a fixed object's pedestal) | "end" (where the clip leaves a
    main object — desk pedestal scenes only) — the keys of PEDESTAL_COLLIDE · collide = the disc (DISC_COLLIDE) or a
    pedestal's (PEDESTAL_COLLIDE's values). Every disc stands from reset to the episode end — no lifetime field."""
    if kind not in PEDESTAL_COLLIDE:
        raise ValueError(f"support kind {kind!r} — only {'|'.join(PEDESTAL_COLLIDE)}")
    if collide not in (DISC_COLLIDE, *PEDESTAL_COLLIDE.values()):
        raise ValueError(
            f"disc stamp collide {collide!r} — only {DISC_COLLIDE}|{sorted(set(PEDESTAL_COLLIDE.values()))}"
        )
    return {"kind": kind, "collide": collide}


def _declared_floor_frame(e, s, decl, _touch, RRu, _floor):
    """The base frame whose object floor is nearest the declared bottom (untouched frames first): coarse pass of ≤ 64
    frames, then ±stride. Returns (frame, floor − declared)."""
    # coarse pass (≤ ~64 base frames) then refine ±stride — bounded cost, exact near the choice
    stride = max(1, (e - s) // 64)

    def _key(b, _decl=decl, _toucher=_touch, _floor=_floor):
        d = abs(_floor(b) - _decl)
        return (d > RRu.Z_STAB, _toucher(b), d)

    best = min(range(s, e, stride), key=_key)
    best = min(range(max(s, best - stride), min(e - 1, best + stride) + 1), key=_key)
    b = int(best)
    dev = float(_floor(b) - decl)
    return b, dev


def _append_disc(b2t, b, _placed, o, floor_at, discs, decl, dev, s, e, iv, touched, name):
    """The disc under the resting object at frame b (centre, radius, floor heights) appended to discs."""
    f = b2t(b)
    V = _placed(o, f)
    cxy = (V[:, :2].min(0) + V[:, :2].max(0)) / 2.0
    r = float(np.linalg.norm(V[:, :2] - cxy, axis=1).max())
    sxy = capture_to_scene(cxy[None, :])[0]
    fl = np.array(list(floor_at.values())) if floor_at else np.array([float(V[:, 2].min())])
    discs.append(
        dict(
            x=float(sxy[0]),
            y=float(sxy[1]),
            r=r,
            bottom_z_capture=float(V[:, 2].min()),
            bottom_z_decl=decl,
            dev_m=dev,
            start=s,
            end=e,
            side=str(iv.get("side")),
            frame=int(f),
            base_frame=int(b),
            contact_at_frame=touched,
            obj=str(name),
            floor_range_sampled_m=float(fl.max() - fl.min()),
            **disc_stamp("rest"),
        )
    )


def _rest_inputs(tree_dir, ti, q, src_idx):
    """task_info and qpos of the tree (read when not given), the base→tree frame map, the base length, the placer and
    the objects by name."""
    import json as _json
    import pathlib as _pl

    RRu = _rest_rule()
    tree_dir = _pl.Path(tree_dir)
    if ti is None:
        ti = _json.loads((tree_dir / "task_info.json").read_text())
    if q is None:
        q = np.load(tree_dir / "0/trajectory_kinematic.npz")["qpos"]
    T = len(q)
    b2t = _base_to_tree(tree_dir, ti, T, src_idx=src_idx)
    Tb = _base_T(ti)
    _placed, _ = _placers(tree_dir, ti, q)
    by_name = {
        o.get("name"): o for o in (ti.get("objects") or []) if o.get("qpos_start") is not None and o.get("mesh_dir")
    }
    return RRu, tree_dir, ti, q, b2t, Tb, _placed, by_name


def rest_discs(
    tree_dir,
    ti: dict | None = None,
    q=None,
    stats: dict | None = None,
    src_idx="tree",
    require_unheld: bool = False,
    require_free_frame: bool = False,
) -> list:
    """The support discs of the **rest intervals** (support.rest_intervals) — for the aerial and the desk scene. A disc
    goes only under an object that is really at rest and not held:
      object   = the object the interval was measured on (rest.rest_object_name — the slot name), not "that hand's main
                 object". Unresolvable → skip (`rest_object_unresolved`); not a main (not spawned) → skip
                 (`rest_object_not_spawned`).
      held     = the rest interval itself already means "hand-free ≥ REST_MIN_S". `require_unheld=True` (desk) also
                 skips an interval whose object's own sidecar shows contact on ≥ HELD_CONTACT_FRAC of it (`held`); the
                 aerial scene does not (an articulated object rests while the hand works its lid).
                 `require_free_frame=True` (aerial): the chosen frame must be untouched by `frame_toucher`
                 (articulated objects: by the base part only) — else skip (`held_at_frame`).
      frame    = the base frame inside the interval (mapped to tree frames with the f0 pad and retime map) whose mesh
                 floor is within rest Z_STAB of the **declared floor** `bottom_z`, preferring a frame with no hand
                 contact, then the closest floor. None within Z_STAB → skip (`declared_floor_absent`).
      struct   = an interval outside the base window [0, segment.T] raises (an unsliced piece must be refreshed, never
                 clamped).
    Every disc stands from reset to the episode end; whether the object is on it is on_own_disc. start/end record the
    interval (base frames). Returns [{x,y(scene),r,bottom_z_capture,bottom_z_decl,dev_m,start,end,side,frame,
    base_frame,contact_at_frame, obj, floor_range_sampled_m, kind, collide}] — world z is the caller's. A `ti` override
    lets audits evaluate a statement in memory."""
    RRu, tree_dir, ti, q, b2t, Tb, _placed, by_name = _rest_inputs(tree_dir, ti, q, src_idx)
    st = stats if stats is not None else {}
    discs = []
    for iv in sorted((ti.get("support") or {}).get("rest_intervals") or [], key=lambda d: int(d.get("start", 0))):
        s, e = int(iv["start"]), int(iv["end"])
        if s < 0 or e <= s or (Tb is not None and e > Tb):
            raise ValueError(
                f"{tree_dir.name}: rest interval [{s},{e}) lies outside the base window [0,{Tb}] — re-cut "
                f"the window (segment/split.py --apply --refresh-support); it is never clamped"
            )
        name = RRu.rest_object_name(iv, ti)
        if name is None or name not in by_name:
            st["rest_object_unresolved"] = st.get("rest_object_unresolved", 0) + 1
            continue
        o = by_name[name]
        if o.get("role") != "main":
            st["rest_object_not_spawned"] = st.get("rest_object_not_spawned", 0) + 1
            continue
        if require_unheld and _held_in_interval(tree_dir, ti, name, s, e):
            st["held"] = st.get("held", 0) + 1
            continue
        decl = float(iv["bottom_z"]) if iv.get("bottom_z") is not None else None
        floor_at: dict = {}

        def _floor(b, _o=o, _cache=floor_at):
            f = b2t(b)
            if f not in _cache:
                _cache[f] = float(_placed(_o, f)[:, 2].min())
            return _cache[f]

        _touch = frame_toucher(tree_dir, ti, q, name, b2t)
        if decl is not None:
            b, dev = _declared_floor_frame(e, s, decl, _touch, RRu, _floor)
            if abs(dev) > RRu.Z_STAB:
                st["declared_floor_absent"] = st.get("declared_floor_absent", 0) + 1
                continue
        else:
            b, dev = (s + e - 1) // 2, None
        touched = bool(_touch(b))
        if require_free_frame and touched:
            st["held_at_frame"] = st.get("held_at_frame", 0) + 1
            continue
        _append_disc(b2t, b, _placed, o, floor_at, discs, decl, dev, s, e, iv, touched, name)
    return discs


def on_own_disc(V, mine: list) -> bool:
    """**"The object stands on one of its own support discs"**: the floor of vertices V (capture coordinates) is within
    rest Z_STAB of a disc top (`bottom_z_capture`) and the floor's xy centre (scene coordinates) is inside that disc's
    radius. Discs stand from reset to the episode end, so a disc being present does not mean the object is on it."""
    if not mine:
        return False
    z_stab = float(_rest_rule().Z_STAB)
    floor = float(V[:, 2].min())
    cxy = capture_to_scene(((V[:, :2].min(0) + V[:, :2].max(0)) / 2.0)[None, :])[0]
    return any(
        abs(floor - float(d["bottom_z_capture"])) <= z_stab
        and float(np.hypot(cxy[0] - d["x"], cxy[1] - d["y"])) <= float(d["r"])
        for d in mine
    )
