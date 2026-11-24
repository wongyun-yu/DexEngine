"""Desk-scene pedestals beyond the main objects' start and rest discs: under fixed spawned objects, under the rests
of moved extras, and under where the clip leaves a main object (end pedestals)."""

from __future__ import annotations

import pathlib as _pathlib

import numpy as np

from dexengine.placement.coords.contacts import _held_in_interval
from dexengine.placement.coords.discs import disc_stamp
from dexengine.placement.coords.frames import capture_to_scene, desk_bounds_scene_for
from dexengine.placement.coords.objects import _base_T, _base_to_tree, _placers, _rest_rule, is_baked_tree
from dexengine.placement.coords.support_kind import support_contact
from dexengine.placement.coords.thresholds import (
    END_PATH_PEN_M,
    END_SKIP_REASONS,
    HIGH_REST_M,
    PEDESTAL_COLLIDE,
    SUPPORT_POLY_PAD_M,
)

#: Trees whose `[END_PEDESTAL]` marker was printed.
_END_PEDESTAL_SAID: set = set()


def main_rest_frames(tree_dir, ti: dict, q, src_idx="tree") -> list:
    """[(main object, tree frame)] where a main object is set down: frame 0 (where it starts), the middle base frame
    of each of its rest intervals (support.rest_intervals · rest.rest_object_name) and the last frame (where the clip
    leaves it — a put-down shorter than a rest interval still ends there)."""
    RRu = _rest_rule()
    mains = [
        o
        for o in ti.get("objects") or []
        if o.get("role") == "main" and o.get("mesh_dir") and o.get("qpos_start") is not None
    ]
    by = {str(o["name"]): o for o in mains}
    out = [(o, f) for o in mains for f in (0, len(q) - 1)]
    ivs = (ti.get("support") or {}).get("rest_intervals") or []
    if ivs:
        b2t = _base_to_tree(tree_dir, ti, len(q), src_idx=src_idx)
        for iv in ivs:
            name = RRu.rest_object_name(iv, ti)
            if name in by:
                out.append((by[name], int(b2t((int(iv["start"]) + int(iv["end"]) - 1) // 2))))
    return out


def receives_main(tree_dir, ti: dict, q, V_fixed, placed=None) -> bool:
    """A fixed object (its frame-0 vertices `V_fixed`) **receives a main object**: at frame 0, a rest frame or the
    last frame (main_rest_frames) a main stands on it by the one geometric test (support_contact — the main above, its
    centre inside the contact polygon + SUPPORT_POLY_PAD_M). Such an object cannot leave the scene."""
    if placed is None:
        placed, _ = _placers(tree_dir, ti, q)
    for o, f in main_rest_frames(tree_dir, ti, q):
        c = support_contact(placed(o, f), V_fixed)  # the fixed object under that main (not above it)
        if c is not None and c["poly_out_m"] <= SUPPORT_POLY_PAD_M and not c["above"]:
            return True
    return False


def fixed_spawned(tree_dir, ti: dict, q) -> list:
    """The objects the env spawns **fixed at frame 0** that are not main objects — the stated support objects (role
    "support") and the still extras (extra_slots STATIC_SPAWN) → [name]. One list for the pedestals and the judge's
    fixed-object checks (static_fixed.fixed_objects)."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415 — the spawn rule

    names = [
        str(o["name"])
        for o in ti.get("objects") or []
        if o.get("role") == "support" and o.get("mesh_dir") and o.get("qpos_start") is not None
    ]
    still = [r["name"] for r in XS.clip_rows(str(tree_dir), ti, q, strict=False) if r["cls"] == XS.STATIC_SPAWN]
    return names + still


def _stands_on(V, lower: list) -> bool:
    """`V` stands on one of the vertex sets `lower` (support_contact with that one below it)."""
    return any(
        c is not None and c["poly_out_m"] <= SUPPORT_POLY_PAD_M and not c["above"]
        for c in (support_contact(V, w) for w in lower)
    )


def enclosing_circle(xy) -> tuple:
    """The smallest circle around xy points (n, 2) → (cx, cy, r). Unique, so it turns with the points: a column laid
    under an object before the placement and moved with it is the column laid under the placed object (an axis-aligned
    box centre is not — it shifts with yaw). Welzl's incremental construction over the convex hull vertices."""
    from scipy.spatial import ConvexHull

    P = np.asarray(xy, float)
    if len(P) > 3 and np.linalg.matrix_rank(P - P.mean(0), tol=1e-12) == 2:  # else degenerate: the points themselves
        P = P[ConvexHull(P).vertices]

    def two(a, b):
        return (a + b) / 2.0, float(np.linalg.norm(a - b)) / 2.0

    def three(a, b, c):
        d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        if abs(d) < 1e-18:  # collinear — the farthest pair
            return max((two(a, b), two(a, c), two(b, c)), key=lambda t: t[1])
        u = (
            np.array(
                [
                    (a @ a) * (b[1] - c[1]) + (b @ b) * (c[1] - a[1]) + (c @ c) * (a[1] - b[1]),
                    (a @ a) * (c[0] - b[0]) + (b @ b) * (a[0] - c[0]) + (c @ c) * (b[0] - a[0]),
                ]
            )
            / d
        )
        return u, float(np.linalg.norm(a - u))

    def out(c, r, q):
        return float(np.linalg.norm(q - c)) > r * (1.0 + 1e-12) + 1e-12

    c, r = P[0], 0.0
    for i in range(1, len(P)):
        if out(c, r, P[i]):
            c, r = P[i], 0.0
            for j in range(i):
                if out(c, r, P[j]):
                    c, r = two(P[i], P[j])
                    for k in range(j):
                        if out(c, r, P[k]):
                            c, r = three(P[i], P[j], P[k])
    return float(c[0]), float(c[1]), float(r)


def _pedestal_row(V, name: str, start: int, end: int, base: str) -> dict:
    """One pedestal row under vertices V (capture) — footprint = the smallest circle around the object's projection
    (enclosing_circle: it turns with the object, so judge and bake lay the same column)."""
    cx, cy, r = enclosing_circle(V[:, :2])
    sxy = capture_to_scene(np.array([[cx, cy]]))[0]
    return dict(
        x=float(sxy[0]),
        y=float(sxy[1]),
        r=r,
        bottom_z_capture=float(V[:, 2].min()),
        obj=name,
        side="static",
        start=int(start),
        end=int(end),
        base=base,
        **disc_stamp("static", PEDESTAL_COLLIDE["static"]),
    )


def static_pedestals(tree_dir, ti: dict, q, dz: float, mains_v: list) -> list:
    """Pedestals under the objects the env spawns **fixed at frame 0** that are not main objects (fixed_spawned). The
    rule lives in static_fixed (columns · column_base — the one formula the judge's candidate placement and this stamp
    both read):
      high      the frame-0 floor is at least HIGH_REST_M above the desk → a column from the desk top (`base` "desk").
      off desk  it stands off the desk box (baked trees — more than static_desk.OVERHANG_MAX_FRAC of its outline area
                outside desk_bounds_scene_for) → a column from the floor (`base` "floor").
    One resting on a lower spawned object at frame 0 (static_fixed.rests_on) gets none: that object carries it.
    Returns disc rows like rest_discs' (x·y scene · r · bottom_z_capture · obj · kind "static" · base); world z is the
    caller's."""
    from dexengine.placement.scene import static_fixed  # noqa: PLC0415

    return static_fixed.pedestal_rows(tree_dir, ti, q, dz, mains_v)


def _spawned_bodies(tree_dir, ti: dict, q) -> tuple:
    """What the env spawns in this clip, for "does it stand on another spawned object" → (rows: extra_slots.clip_rows
    by name, slot: the moved extras' names, by: every object with a body {name: objects[] row}, spawned: the spawned
    names (mains · supports · slotted extras), moving: the spawned ones that follow the reference (mains · moved
    extras) — the rest stand at their frame-0 pose)."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415

    rows = {r["name"]: r for r in XS.clip_rows(str(tree_dir), ti, q, strict=False)}
    slot = {n for n, r in rows.items() if r["cls"] == XS.SLOT}
    by = {
        str(o.get("name")): o for o in ti.get("objects") or [] if o.get("qpos_start") is not None and o.get("mesh_dir")
    }
    spawned = [
        n for n, o in by.items() if o.get("role") in XS.SPAWNED_ROLES or (n in rows and rows[n]["cls"] in XS.SLOTTED)
    ]
    moving = {n for n in spawned if by[n].get("role") == "main" or n in slot}
    return rows, slot, by, spawned, moving


def extra_rest_pedestals(tree_dir, ti: dict, q, dz: float, stats: dict | None = None, src_idx="tree") -> list:
    """Pedestals under the **rests of the moved extras** the env spawns (extra_slots SLOT — bodies that follow the
    reference): a rest interval measured on such an extra, its floor at the interval's frame at least HIGH_REST_M above
    the desk and not standing on another spawned object there → a column from the desk top under that pose. The frame
    = the base frame of the interval whose mesh floor is closest to the declared floor (within rest Z_STAB, else
    `declared_floor_absent`). An interval of an object the scene does not spawn gives none; `stats` counts each
    (rest_extra_pedestal · rest_extra_on_object · rest_extra_low · rest_object_absent)."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415

    RRu = _rest_rule()
    st = stats if stats is not None else {}
    ivs = (ti.get("support") or {}).get("rest_intervals") or []
    if not ivs:
        return []
    rows, slot, by, spawned, moving = _spawned_bodies(tree_dir, ti, q)
    placed, _ = _placers(tree_dir, ti, q) if by else (None, None)
    b2t = _base_to_tree(tree_dir, ti, len(q), src_idx=src_idx)
    out, seen = [], set()

    def _count(k):
        st[k] = st.get(k, 0) + 1

    for iv in sorted(ivs, key=lambda d: int(d.get("start", 0))):
        name = RRu.rest_object_name(iv, ti)
        if name is None or name not in by or (by[name].get("role") not in XS.SPAWNED_ROLES and name not in rows):
            _count("rest_object_absent")
            continue
        if name not in slot:
            continue  # mains → rest_discs · fixed objects → static_pedestals · unslotted → no body
        s, e = int(iv["start"]), int(iv["end"])
        o = by[name]
        decl = float(iv["bottom_z"]) if iv.get("bottom_z") is not None else None
        stride = max(1, (e - s) // 32)
        cand = [(b, float(placed(o, b2t(b))[:, 2].min())) for b in range(s, e, stride)]
        b, fl = min(cand, key=lambda t: abs(t[1] - decl)) if decl is not None else cand[len(cand) // 2]
        if decl is not None and abs(fl - decl) > RRu.Z_STAB:
            _count("declared_floor_absent")
            continue
        if fl - float(dz) < HIGH_REST_M:
            _count("rest_extra_low")
            continue
        f = b2t(b)
        V = placed(o, f)
        if _stands_on(V, [placed(by[n], f if n in moving else 0) for n in spawned if n != name]):
            _count("rest_extra_on_object")
            continue
        key = (name, round(float(V[:, 0].mean()), 2), round(float(V[:, 1].mean()), 2), round(fl, 2))
        if key in seen:  # one pedestal per resting place (an interval listed twice · a return to it)
            continue
        seen.add(key)
        _count("rest_extra_pedestal")
        out.append(_pedestal_row(V, name, s, e, "desk"))
    return out


def end_window(tree_dir, ti: dict, q, src_idx="tree") -> tuple:
    """The clip's **end window** → (f0: its first tree frame, b0: its first base frame, Tb: the base length).
    The window = the last place_rule.PUT_STILL_S before the end on the pre-retime clock (place_rule.still_start, read
    back through the tree's retime map); its base frames [b0, Tb) are the ones that play inside it (f0 pad and retime
    map applied), where the contact sidecar is read. A tree without segment.T raises."""
    from dexengine.placement.scene import place_rule  # noqa: PLC0415

    f0 = int(place_rule.still_start(tree_dir, ti, len(q)))
    Tb = _base_T(ti)
    if Tb is None:
        raise ValueError(f"{tree_dir}: task_info.segment.T is missing — cannot place the end window in base frames")
    b2t = _base_to_tree(tree_dir, ti, len(q), src_idx=src_idx)
    return f0, next((b for b in range(Tb) if b2t(b) >= f0), Tb - 1), Tb


def end_path_depth(placed, o: dict, last: int, circle: tuple, top: float) -> float:
    """How deep the object's own reference passes through a column (circle (cx, cy, r) capture · top z) before the
    last frame — the largest (top − z) over its vertices inside the circle, over tree frames [0, last) (m, ≤ 0 = it
    never enters)."""
    cx, cy, r = circle
    deep = 0.0
    for f in range(last):
        V = placed(o, f)
        ins = np.hypot(V[:, 0] - cx, V[:, 1] - cy) < r
        if ins.any():
            deep = max(deep, float(top) - float(V[ins, 2].min()))
    return deep


def _end_mains(ti):
    """The main objects of the clip that have a mesh and a qpos start, sorted by name."""
    mains = sorted(
        (
            o
            for o in ti.get("objects") or []
            if o.get("role") == "main" and o.get("mesh_dir") and o.get("qpos_start") is not None
        ),
        key=lambda o: str(o["name"]),
    )
    return mains


def _append_end_pedestal(out, sxy, r, fl, name, o, b0, Tb):
    """The end pedestal row of this object (scene centre, radius, floor heights) appended to out."""
    out.append(
        dict(
            x=float(sxy[0]),
            y=float(sxy[1]),
            r=r,
            bottom_z_capture=fl,
            obj=name,
            side=str(o.get("side")),
            start=int(b0),
            end=int(Tb),
            base="desk",
            **disc_stamp("end", PEDESTAL_COLLIDE["end"]),
        )
    )


def _report_end_skips(tree_dir, said, win):
    """Once per tree: why end pedestals were skipped."""
    key = str(tree_dir)
    if said and key not in _END_PEDESTAL_SAID:
        _END_PEDESTAL_SAID.add(key)
        skips = " · ".join(f"{k} {said[k]}" for k in END_SKIP_REASONS if k in said) or "none"
        print(
            f"[END_PEDESTAL] {_pathlib.Path(key).name}: end pedestals {said.get('end_pedestal', 0)} · "
            f"skipped {skips} (end window from frame {win[0]} · path allowance {END_PATH_PEN_M * 1000:.0f} mm)",
            flush=True,
        )


def end_pedestals(
    tree_dir, ti: dict, q, dz: float, own: list, placed, stats: dict | None = None, src_idx="tree"
) -> list:
    """Pedestals under where the clip **leaves** its main objects (kind "end"): the clip ends before the put-down is
    long enough to be a rest interval, so no rest pedestal stands there and the released object would fall. A main
    object whose floor at the last frame is at least HIGH_REST_M above the desk (below that the desk holds it — not
    counted) gets a column from the desk top under that pose, unless (the first that holds — END_SKIP_REASONS, counted
    in `stats`):
      end_moving        it is not still at the end (place_rule.end_still over the end window)
      end_covered       one of its own pedestals (`own` — the start · rest rows) already carries that pose: its top
                        within HIGH_REST_M of the end floor and the end footprint's centre inside its circle
      end_held          a hand holds it over the end window (`_held_in_interval` on the window's base frames)
      end_off_desk      the end footprint stands off the desk box (static_fixed.off_desk). Baked trees only: an
                        unplaced tree has no scene place to measure
      end_on_object     it stands on another spawned object there (_stands_on over _spawned_bodies, the moved ones at
                        the last frame): that object carries it
      end_path_blocked  its own reference passes through where the column would stand (end_path_depth >
                        END_PATH_PEN_M)
    Footprint = the circle around the end pose's projection (enclosing_circle). Rows like start_support's discs
    (x·y scene · r · bottom_z_capture · obj · side · start/end = the end window's base frames · kind "end" · base
    "desk"); world z is the caller's. Marker `[END_PEDESTAL]` once per tree that has a candidate."""
    from dexengine.placement.scene import place_rule as PR  # noqa: PLC0415
    from dexengine.placement.scene import static_fixed  # noqa: PLC0415

    st = stats if stats is not None else {}
    mains = _end_mains(ti)
    last = len(q) - 1
    out, said, win, bodies, box = [], {}, None, None, None

    def _count(k):
        st[k] = st.get(k, 0) + 1
        said[k] = said.get(k, 0) + 1

    for o in mains:
        name = str(o["name"])
        VT = placed(o, last)
        fl = float(VT[:, 2].min())
        if fl - float(dz) < HIGH_REST_M:
            continue  # the desk holds it
        if win is None:
            win = end_window(tree_dir, ti, q, src_idx=src_idx)
        f0, b0, Tb = win
        if not PR.end_still(q, int(o["qpos_start"]), f0):
            _count("end_moving")
            continue
        cx, cy, r = enclosing_circle(VT[:, :2])
        sxy = capture_to_scene(np.array([[cx, cy]]))[0]
        if any(
            str(d["obj"]) == name
            and abs(float(d["bottom_z_capture"]) - fl) < HIGH_REST_M
            and float(np.hypot(float(d["x"]) - sxy[0], float(d["y"]) - sxy[1])) <= float(d["r"])
            for d in own
        ):
            _count("end_covered")
            continue
        if _held_in_interval(tree_dir, ti, name, b0, Tb - 1):
            _count("end_held")
            continue
        if is_baked_tree(ti):
            if box is None:
                box = desk_bounds_scene_for(tree_dir, margin=0.0)
            if static_fixed.off_desk(capture_to_scene(VT[:, :2]), box):
                _count("end_off_desk")
                continue
        if bodies is None:
            bodies = _spawned_bodies(tree_dir, ti, q)
        _, _, by, spawned, moving = bodies
        if _stands_on(VT, [placed(by[n], last if n in moving else 0) for n in spawned if n != name]):
            _count("end_on_object")
            continue
        if end_path_depth(placed, o, last, (cx, cy, r), fl) > END_PATH_PEN_M:
            _count("end_path_blocked")
            continue
        _count("end_pedestal")
        _append_end_pedestal(out, sxy, r, fl, name, o, b0, Tb)
    _report_end_skips(tree_dir, said, win)
    return out
