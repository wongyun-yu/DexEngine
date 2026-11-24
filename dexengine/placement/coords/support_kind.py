"""Support kind: what holds an unheld main object up at the start — the desk, a **support object** (a real tracked
object under it: a test-tube rack · tripod · plate · a still sibling part), a disc (nothing tracked under it), or
nothing (excluded). Geometry at tree frame 0 on the visual meshes; thresholds in `thresholds`."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as _Rot

from dexengine.placement.coords.contacts import start_gaps
from dexengine.placement.coords.discs import disc_stamp, on_own_disc
from dexengine.placement.coords.frames import RZ_CAP2SCENE, capture_to_scene
from dexengine.placement.coords.objects import _base_T, _placers
from dexengine.placement.coords.thresholds import (
    START_HOLD_FRAMES,
    SUPPORT_BAND_M,
    SUPPORT_GAP_MAX_M,
    SUPPORT_MIN_VERTS,
    SUPPORT_NEST_MAX_M,
    SUPPORT_POLY_PAD_M,
    SUPPORT_REACH_M,
    SUPPORT_STILL_DEG,
    SUPPORT_STILL_M,
)


def _contact_poly_out(pts_xy, c_xy) -> float:
    """Distance (m) by which point c lies outside the convex hull of pts_xy (≤ 0 inside); a degenerate hull (≤ 2
    distinct points or collinear) is measured as the distance to the segment/point set."""
    from scipy.spatial import ConvexHull, QhullError

    P2 = np.unique(np.round(np.asarray(pts_xy, float), 6), axis=0)
    if len(P2) >= 3:
        try:
            eq = ConvexHull(P2).equations
            return float((eq[:, :2] @ c_xy + eq[:, 2]).max())
        except QhullError:
            pass
    a, b = P2[np.argmin(P2 @ (P2[-1] - P2[0]))], P2[np.argmax(P2 @ (P2[-1] - P2[0]))]
    ab = b - a
    t = float(np.clip((c_xy - a) @ ab / max(float(ab @ ab), 1e-18), 0.0, 1.0))
    return float(np.linalg.norm(c_xy - (a + t * ab)))


def support_contact(Vm, Vc) -> dict | None:
    """**Is candidate Vc under main Vm** (both (n,3) capture vertices at the same frame) — the one geometric test.
    None = not under. Else {n, gap_m, nested, poly_out_m, above}: n candidate vertices within SUPPORT_REACH_M (xy) of
    the main's bottom band and no higher than floor + SUPPORT_NEST_MAX_M · gap = floor − their highest z · nested = a
    candidate vertex within reach rises above that · poly_out = main centre outside their polygon · above = candidate
    centroid z ≥ main centroid z."""
    from scipy.spatial import cKDTree

    Vm, Vc = np.asarray(Vm, float), np.asarray(Vc, float)
    fl = float(Vm[:, 2].min())
    band = Vm[Vm[:, 2] <= fl + SUPPORT_BAND_M]
    d, _ = cKDTree(band[:, :2]).query(Vc[:, :2], distance_upper_bound=SUPPORT_REACH_M)
    reach = Vc[np.isfinite(d)]
    low = reach[reach[:, 2] <= fl + SUPPORT_NEST_MAX_M]
    if len(low) < SUPPORT_MIN_VERTS:
        return None
    gap = fl - float(low[:, 2].max())
    if gap > SUPPORT_GAP_MAX_M:
        return None
    return {
        "n": int(len(low)),
        "gap_m": round(gap, 4),
        "nested": bool(len(reach) > len(low)),
        "poly_out_m": round(_contact_poly_out(low[:, :2], Vm[:, :2].mean(0)), 4),
        "above": bool(float(Vc[:, 2].mean()) >= float(Vm[:, 2].mean())),
    }


def pose_still(pos, quat_wxyz) -> bool:
    """A support stays put over the window: origin moves ≤ SUPPORT_STILL_M and rotates ≤ SUPPORT_STILL_DEG from the
    first row (pos (T,3) · quat (T,4) wxyz)."""
    from scipy.spatial.transform import Rotation as _R

    pos, qw = np.asarray(pos, float), np.asarray(quat_wxyz, float)
    if float(np.linalg.norm(pos - pos[0], axis=1).max()) > SUPPORT_STILL_M:
        return False
    R = _R.from_quat(qw[:, [1, 2, 3, 0]])
    return float(np.degrees((R * R[0].inv()).magnitude()).max()) <= SUPPORT_STILL_DEG


def support_of(tree_dir, ti: dict, q, main: dict) -> dict:
    """The support object of one main object — **independent of holding** (a scene property: object_roles reads it to
    give the role `support`). Candidates = the tree's non-main objects with a pose and a mesh. Returns {support: name |
    None, why: None | "multiple" | "moving" | "upside_down" | "nested", gap_m, n, candidates: [names under it]} — a
    `why` means undecided (two or more candidates · the support moves · it sits above the main · the main is nested
    deeper than SUPPORT_NEST_MAX_M)."""
    placed, _ = _placers(tree_dir, ti, q)
    Vm = placed(main, 0)
    hits = []
    for o in ti.get("objects") or []:
        if o.get("role") == "main" or o.get("qpos_start") is None or not o.get("mesh_dir"):
            continue
        c = support_contact(Vm, placed(o, 0))
        if c is None or c["poly_out_m"] > SUPPORT_POLY_PAD_M:
            continue
        s0 = int(o["qpos_start"])
        hits.append((o, c, pose_still(q[:, s0 : s0 + 3], q[:, s0 + 3 : s0 + 7])))
    out = {"support": None, "why": None, "gap_m": None, "n": None, "candidates": [str(o["name"]) for o, _c, _s in hits]}
    if not hits:
        return out
    if len(hits) > 1:
        return {**out, "why": "multiple"}
    o, c, still = hits[0]
    why = "moving" if not still else "upside_down" if c["above"] else "nested" if c["nested"] else None
    return {**out, "why": why, "support": None if why else str(o["name"]), "gap_m": c["gap_m"], "n": c["n"]}


def support_objects(ti: dict, side: str) -> list:
    """The support objects stated for the main object of `side` — objects[] with role `support` whose side is that side
    or "both" (written by object_roles from `support_of`; read, never recomputed here)."""
    return [o for o in ti.get("objects") or [] if o.get("role") == "support" and o.get("side") in (side, "both")]


def scene_support_poses(ti: dict, q) -> dict:
    """{name: pose} — every stated support object of the tree (role "support", any side: the env spawns each fixed at
    this pose) with its tree qpos at frame 0 (capture x·y·z · wxyz). Read by the bake stamp and the arm guard."""
    out = {}
    for o in ti.get("objects") or []:
        if o.get("role") == "support":
            s0 = int(o["qpos_start"])
            out[str(o["name"])] = [round(float(v), 6) for v in q[0, s0 : s0 + 7]]
    return out


def support_scene_pose(pose, z_off: float) -> tuple:
    """A support object's scene pose from its stamp pose (tree qpos frame 0: capture x·y·z · wxyz) — position
    Rz(−90°)·p + (0, 0, z_off) · orientation Rz(−90°)·R. Read by the env's fixed slot and the judge's support geometry.
    A pose that is not 7 finite numbers with a unit quaternion raises. Returns (xyz (3,), wxyz (4,))."""
    p = np.asarray(pose, float)
    if p.shape != (7,) or not np.isfinite(p).all() or abs(float(np.linalg.norm(p[3:])) - 1.0) > 1e-3:
        raise ValueError(f"support pose {list(p.ravel())[:7]} — must be 7 values x·y·z · wxyz (unit quaternion)")
    xyz = capture_to_scene(p[:3]) + np.array([0.0, 0.0, float(z_off)])
    q = (RZ_CAP2SCENE * _Rot.from_quat([p[4], p[5], p[6], p[3]])).as_quat()  # xyzw
    return xyz, np.array([q[3], q[0], q[1], q[2]])


def clip_supports(ti: dict, supports, clip: str) -> tuple[list, bool]:
    """A clip's support objects from the stamp `supports` → (one row per support object, is it an old stamp without
    the field). No field (None) and no support-role object = 0 supports (an old stamp) · no field but a support-role
    object = raises (re-bake). With the field, rows are **grouped per object**: the stamp has one row per main
    object, so a support under two mains is written twice — one object = one slot; the same object written with two
    poses raises."""
    if supports is None:
        roles = [o.get("name") for o in ti.get("objects") or [] if o.get("role") == "support"]
        if roles:
            raise RuntimeError(
                f"[SUPPORT] {clip}: support-role objects {roles} but the stamp has no supports field — "
                "a different bake generation (re-bake)"
            )
        return [], True
    by: dict = {}
    for s in supports:
        first = by.setdefault(str(s["obj"]), s)
        if [float(v) for v in first["pose"]] != [float(v) for v in s["pose"]]:
            raise RuntimeError(
                f"[SUPPORT] {clip}: support {s['obj']!r} is stamped with two poses ({first['pose']} · "
                f"{s['pose']}) — one object stands in one place"
            )
    return list(by.values()), False


def support_object_verts(tree_dir, ti: dict, q) -> list:
    """Capture vertices of every stated support object (role "support", any side) at tree frame 0 — the pose the env
    fixes it at. [] when the tree states none."""
    sups = [o for o in ti.get("objects") or [] if o.get("role") == "support"]
    if not sups:
        return []
    placed, _ = _placers(tree_dir, ti, q)
    return [placed(o, 0) for o in sups]


def on_support_object(V, sup_v: list) -> bool:
    """**"The object stands on a support object"** — support_contact with the centre inside the contact polygon +
    SUPPORT_POLY_PAD_M, against each support at its fixed pose (support_object_verts)."""
    return any(c is not None and c["poly_out_m"] <= SUPPORT_POLY_PAD_M for c in (support_contact(V, v) for v in sup_v))


def stands_supported(V, mine: list, sup_v: list) -> bool:
    """Something the sim spawns carries the object at V (capture vertices): its own disc (on_own_disc) or a support
    object (on_support_object) — the one "supported" reading of the aerial gap and tail."""
    return (bool(mine) and on_own_disc(V, mine)) or (bool(sup_v) and on_support_object(V, sup_v))


def support_kinds(tree_dir, ti: dict, q, rest: list, *, aerial: bool, src_idx="tree") -> list:
    """**The support kind of every main object at the start.** Per start_gaps record: held (in a hand over the start
    window) · desk (a table scene and the floor below HIGH_REST_M) · support_object (a stated support object under it —
    then no start disc on that side) · disc (a rest disc of that object starting inside the start window covers it —
    `covered` = that disc — or, still over the window, a start disc) · excluded (moves unheld with nothing to stand
    on). `rest` = the rest discs left after rest_on_support, so a support object never shares the start with a disc.
    `support` = the stated support object of that side whatever the kind (the env spawns every one).
    Returns [{obj, side, kind, support | None, covered | None, gap: start_gaps record}]."""
    out = []
    for g in start_gaps(tree_dir, ti, q, aerial=aerial, src_idx=src_idx):
        sup = support_objects(ti, g["side"])
        if len(sup) > 1:
            raise ValueError(
                f"{tree_dir}: the {g['side']} main object {g['obj']} has {len(sup)} support objects — the "
                f"role statement allows one (object_roles · support_of)"
            )
        rec = {
            "obj": g["obj"],
            "side": g["side"],
            "support": str(sup[0]["name"]) if sup else None,
            "covered": None,
            "gap": g,
        }
        if not g["needs_support"]:
            out.append({**rec, "kind": "held" if g["held"] else "desk"})
            continue
        cover = [d for d in rest if d["obj"] == g["obj"] and int(d["start"]) < START_HOLD_FRAMES]
        if sup:
            out.append({**rec, "kind": "support_object"})
        elif cover or g["still"]:
            out.append({**rec, "kind": "disc", "covered": min(cover, key=lambda x: int(x["start"])) if cover else None})
        else:
            out.append({**rec, "kind": "excluded"})
    return out


def rest_on_support(tree_dir, ti: dict, q, rest: list) -> tuple[list, list]:
    """**A support object carries what rests on it**: a rest disc whose main object, at the disc's frame, stands on a
    stated support object (the env fixes every one at its frame-0 pose) by the one geometric test is not a disc — two
    things would carry one object. Rest discs elsewhere stay. Returns (kept, dropped), order kept."""
    sv = support_object_verts(tree_dir, ti, q) if rest else []
    if not sv:
        return list(rest), []
    placed, _ = _placers(tree_dir, ti, q)
    by = {str(o.get("name")): o for o in ti.get("objects") or []}
    kept, dropped = [], []
    for d in rest:
        (dropped if on_support_object(placed(by[str(d["obj"])], int(d["frame"])), sv) else kept).append(d)
    return kept, dropped


def start_support(tree_dir, ti: dict, q, rest: list, *, aerial: bool, src_idx="tree") -> dict:
    """**The start supports**, built from `support_kinds`, for both scenes. The rest discs a support object carries go
    first (rest_on_support), then per kind:
      disc (covered)  that rest disc (start < START_HOLD_FRAMES) carries the object from reset — no new disc.
      disc (start)    a disc under it, kind "start", standing to the episode end.
      support_object  no disc: the stated support object holds it.
      excluded        'start_moving' — needs support but moves unheld and nothing carries it.
    `supports` lists every stated support object of the tree's mains (any kind · `holds` = it is the start support of
    an unheld main · pose = its tree qpos at frame 0, the pose the env fixes it at).
    Returns {rest: [the rest discs kept], on_support: [the rest discs a support carries], discs: [start discs:
    x,y(scene),r,bottom_z_capture,obj,side,start,end(base frames), kind · collide], excluded: [{obj, side, why, t_s}],
    supports: [{obj, main, side, holds, pose}]}; world z is the caller's."""
    poses = scene_support_poses(ti, q)
    Tb = _base_T(ti)
    discs, excluded, supports = [], [], []
    rest, on_support = rest_on_support(tree_dir, ti, q, rest)
    for k in support_kinds(tree_dir, ti, q, rest, aerial=aerial, src_idx=src_idx):
        g = k["gap"]
        if k["support"]:
            supports.append(
                {
                    "obj": k["support"],
                    "main": k["obj"],
                    "side": k["side"],
                    "holds": k["kind"] == "support_object",
                    "pose": poses[k["support"]],
                }
            )
        if k["kind"] == "disc" and not k["covered"]:
            discs.append(
                dict(
                    x=g["x"],
                    y=g["y"],
                    r=g["r"],
                    bottom_z_capture=g["floor_capture"],
                    obj=k["obj"],
                    side=k["side"],
                    start=0,
                    end=Tb,
                    **disc_stamp("start"),
                )
            )
        elif k["kind"] == "excluded":
            excluded.append({"obj": k["obj"], "side": k["side"], "why": "start_moving", "t_s": 0.0})
    return {"rest": rest, "on_support": on_support, "discs": discs, "excluded": excluded, "supports": supports}
