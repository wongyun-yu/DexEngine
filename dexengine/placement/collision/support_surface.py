#!/usr/bin/env python3
"""Arm ↔ support object — the support term of the arm guard (judge = bake = gate = final guard,
collision.core.side_guard).

A support object (objects[] role "support" — object_roles from coords.support_of) is spawned fixed at its frame-0 pose
and collides with the arms and hands (collision_presets.SUPPORT_COLLIDE — this module measures the "arm" cell only when
it is "ON", and reports the "hand" cell). Every stated support object of the tree counts for both arms (a rack beside
the left hand blocks the right arm too), whatever its `holds`.

Ruler = the arm surface ruler, read from arm_surface (never re-derived): the arm bodies are ctx["surf"] (vendor
capsules + revision hulls — link_<side>_arm_6 included, the desk ruler's link set) and the distance is the two-arm
crossing ruler (arm_surface.hull_cap_sd · hull_hull_sd, signed surface distance). The support's shape is what the sim
loads: objects.hull.body_parts (the cooked hulls of the USD collision parts — the bake's dump — matched to the object's
mesh frame), each part as its convex hull (a non-convex part is measured as its hull — conservative). Tolerance =
collision.core.DESK_SURF_PEN_MM: the support stands in for the desk under its object, so it gets the desk's value,
never a third one. Poses = the guard's: the arm at ctx base + z_off over the frames and the retiming chords
(collision.core._chords, the same subdivision as the desk term); the support at coords.support_scene_pose(its tree pose,
coords.scene_z_offset(tree)) — Rz(−90°)·pose + z, the pose the env fixes the support slot at (one owner).

Not in the search scalar (collision.core.desk_penetration · desk_penetration_each · their torch twins): the guard
decides adoption, the search does not see supports — a support in the way costs yield (a guard rejection), never a
wrong adoption.

Hand ↔ support: HAND_SUPPORT_MODE "report" — the hand bodies (palm + finger groups of the hand tree the torso guard
reads) are measured at frame 0 and reported (gate marker), never a guard reason.
The support list is a SceneSupports — only scene_supports (the statement reader) builds one, and the guard takes
nothing else: an empty or hand-made list would switch the term off silently.
Placement: the judge reads the **base** tree (capture pose) and adopts a placement the bake then applies; its guard
takes placed_supports(scene_supports(base), dx, dy, yaw, pivot) — the bake's own transform (coords.place_pose) — so
judge and gate measure the same support. What depends on where the scene lands — a fixed object off the desk box
stands on a column from the floor, a still extra on the robot's standing spot is not spawned — is re-decided there
by the formula the bake stamps (static_fixed.at · column_base · spawned_at). The bake · gate · final guard read
the baked tree (scene_supports alone).
Markers: [SUPPORT_RULER] once per tree with supports · the guard reason
`<side><support-penetration word><mm>mm(<object>)` (the word is owned by collision.core; the judge and the gate parse
it).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dexengine.placement.collision import arm_surface as AS
from dexengine.placement.collision.geom import arm_link_transforms, capsule_endpoints
from dexengine.placement.coords import core as VC

#: Hand ↔ support overlap at frame 0 is measured and reported, never a reason (the clip stays and trains — like
#: coords.F0_HAND_OWNOBJ_MODE). The only mode; hand_support_mm dies on any other value.
HAND_SUPPORT_MODE = "report"
_SAID: set = set()


class SceneSupports(list):
    """The support list of one tree, as scene_supports read it from the statement (role "support"). A plain list —
    `[]` included — is refused by the guard (checked), so a caller cannot switch the support term off by passing
    nothing. Copies (list(...), slices) are plain lists and are refused too."""

    def __init__(self, items, *, tree: str, fixed: dict | None = None):
        super().__init__(items)
        self.tree = str(tree)
        self.fixed = fixed  # a base tree's fixed-object statement (static_fixed) — placed_supports re-decides it


def checked(supports) -> SceneSupports:
    """The guard's entry check — `supports` must be scene_supports' own return (SceneSupports); else dies."""
    if type(supports) is not SceneSupports:
        raise TypeError(
            f"the support list did not come from the statement reader (support_surface.scene_supports)"
            f" ({type(supports).__name__}) — an empty or hand-made list would switch the support term off "
            "silently"
        )
    return supports


def _presets():
    from dexengine.rl.tasks.scene import collision_presets

    return collision_presets


def arm_measured() -> bool:
    """The arm term runs iff the sim collides the arm with support objects (collision_presets.SUPPORT_COLLIDE["arm"]
    — the statement the env authoring and the collision audit read); an unknown value dies there."""
    return _presets().expected_support(1, "arm") == "ON"


def extra_measured() -> bool:
    """The arm term takes the fixed extras (the still slots — touched · receiver) iff the judge's statement says so
    (collision_presets.EXTRA_FIXED_JUDGE_ARM). It is a statement of its own, not the sim's collision table: the sim
    does not collide the robot with a still slot (EXTRA_KIND_COLLIDE — exploration), the judge still drops a placement
    whose arm passes through one. An unknown value dies."""
    v = _presets().EXTRA_FIXED_JUDGE_ARM
    if v not in ("ON", "OFF"):
        raise ValueError(f"EXTRA_FIXED_JUDGE_ARM={v!r} — ON | OFF only")
    return v == "ON"


def scene_supports(tree_dir) -> SceneSupports:
    """SceneSupports [{obj, parts: [hull record in the guard frame]}] — what stands fixed in the scene and meets the
    arms: every stated support object of the tree (coords.scene_support_poses: role "support", its frame-0 tree pose),
    the pedestals (pedestals) and, in the pedestal rule's scope, the still extras the env spawns fixed at frame 0
    (static_fixed.fixed_objects — extra_measured); empty when the tree states none. An articulated support (its parts'
    pose needs a joint value) or an object without its USD dies; a still extra is measured as the rigid USD
    the env's extra slot spawns (objects.hull.usd_file(…, False) — articulated or not). A base tree carries its
    fixed-object statement (`fixed`) so placed_supports can re-decide it at a candidate placement."""
    from dexengine.data.objects import hull as OH
    from dexengine.placement.scene import static_fixed as SF

    tree_dir = Path(tree_dir)
    ti = json.loads((tree_dir / "task_info.json").read_text())
    q = np.load(tree_dir / "0/trajectory_kinematic.npz")["qpos"]
    poses = VC.scene_support_poses(ti, q)
    peds = pedestals(tree_dir, ti)
    fx = SF.fixed_objects(tree_dir, ti, q)
    still = {o["name"]: o["pose"] for o in fx["objs"] if o["role"] != "support"} if extra_measured() else {}
    if not poses and not peds and not fx["objs"]:
        return SceneSupports([], tree=tree_dir.name)
    by = {str(o.get("name")): o for o in ti.get("objects") or []}
    zoff = float(VC.scene_z_offset(tree_dir))
    out = [_posed(PEDESTAL_PREFIX + str(d["obj"]), *_pedestal_pose_local(d, zoff), zoff) for d in peds]
    for name, pose in sorted({**still, **poses}.items()):
        o = by[name]
        if o.get("articulation") and name in poses:  # (a still extra = its rigid USD, as the env's slot spawns it)
            raise ValueError(
                f"{tree_dir.name}: support object {name} is articulated — its parts' pose needs a "
                "joint value, so it cannot be measured"
            )
        mesh = VC._object_mesh_dir(tree_dir, ti, o) / "visual.obj"
        parts = [
            np.asarray(v, float)
            for v in OH.body_parts(str(ti["dataset_name"]), str(o["obj_id"]), False, [(name, [mesh])])[name]
        ]
        out.append(_posed(name, pose, parts, zoff))
    if str(tree_dir) not in _SAID:
        _SAID.add(str(tree_dir))
        print(
            f"[SUPPORT_RULER] {tree_dir.name}: support objects {len(poses)} · pedestals {len(peds)} · still extras "
            f"{len(still)} · parts {sum(len(s['parts']) for s in out)} · "
            f"arm {'measured' if arm_measured() else 'not measured (SUPPORT_COLLIDE arm OFF)'} · hand "
            f"{HAND_SUPPORT_MODE} · z {zoff:+.4f} · tolerance = collision.core.DESK_SURF_PEN_MM",
            flush=True,
        )
    return SceneSupports(out, tree=tree_dir.name, fixed=None if fx["placed"] or not fx["objs"] else dict(fx, zoff=zoff))


#: The guard's name prefix for a pedestal (the reason reads
#: `<side><support-penetration word><mm>mm(pedestal:<object>)`); it is written into the reject reasons.
PEDESTAL_PREFIX = "pedestal:"
#: Sides of the polygon prism the guard measures a pedestal's cylinder as — circumscribing (the polygon's edges are
#: tangent to the circle), so the measured column is never thinner than the sim's.
PEDESTAL_SIDES = 16


def pedestals(tree_dir, ti: dict) -> list:
    """The pedestals of a desk-scene tree — its desk support statement's discs whose stamp is the pedestal
    (coords.PEDESTAL_COLLIDE's values — a column from the desk top up to the object's floor; whatever it collides with
    in the env, the arm guard measures it).
    One reader: coords.desk_support_discs (stamp on a baked tree · computed on a base tree — the placement then moves it
    with the supports). Aerial trees and desk trees outside the pedestal rule's scope have none (their discs touch
    objects only — the statement is not even read)."""
    if VC.scene_mode(tree_dir) == "aerial" or not VC.pedestal_scene(ti):  # aerial · out of scope = discs only
        return []
    return [d for d in VC.desk_support_discs(tree_dir)["discs"] if d.get("collide") in VC.PEDESTAL_COLLIDE.values()]


def _pedestal_pose_local(d: dict, zoff: float) -> tuple:
    """(stamp-frame pose, [prism vertices]) of one pedestal — the pose in the frame scene_support_poses use (capture
    x·y · z without the scene offset · identity wxyz) so placed_supports moves it like a support object, and its local
    shape = a PEDESTAL_SIDES-gon prism circumscribing the stamped cylinder (radius r · height h, top at the origin)."""
    if d.get("h") is None:
        raise ValueError(f"pedestal {d.get('obj')} has no height h — a stamp from before pedestals (rebake)")
    cx, cy = VC.scene_to_capture(np.array([[float(d["x"]), float(d["y"])]]))[0]
    a = np.linspace(0.0, 2 * np.pi, PEDESTAL_SIDES, endpoint=False)
    rr = float(d["r"]) / np.cos(np.pi / PEDESTAL_SIDES)
    ring = np.stack([rr * np.cos(a), rr * np.sin(a)], 1)
    h = float(d["h"])
    local = np.concatenate([np.c_[ring, np.zeros(len(a))], np.c_[ring, np.full(len(a), -h)]], 0)
    return [float(cx), float(cy), float(d["z"]) - zoff, 1.0, 0.0, 0.0, 0.0], [local]


def _posed(name: str, pose, local: list, zoff: float) -> dict:
    """One support item: its parts (mesh frame) at a tree pose — env fixed-slot pose (coords.support_scene_pose)."""
    from scipy.spatial.transform import Rotation as _R

    xyz, wxyz = VC.support_scene_pose(pose, zoff)  # the env fixed-slot pose (one owner)
    Rs = _R.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_matrix()
    return {
        "obj": name,
        "pose": list(pose),
        "local": local,
        "zoff": zoff,
        "parts": [AS.hull_record(f"{name}#{i}", -1, v @ Rs.T + xyz) for i, v in enumerate(local)],
    }


def placed_supports(supports, dx: float, dy: float, yaw_deg: float, pivot) -> SceneSupports:
    """scene_supports of a **base** tree moved by a candidate placement (capture dx·dy · yaw · pivot) — the supports
    where the bake will put them (coords.place_pose: the bake's rigid_pl transform), so the judge's guard measures what
    the gate measures on the baked tree. `supports` must be scene_supports' own return (checked); a baked tree is read
    with scene_supports alone (its qpos already carries the placement)."""
    sup = checked(supports)
    place = (dx, dy, yaw_deg, pivot)
    gone, cols = _fixed_at(sup, place)
    return SceneSupports(
        [
            _posed(s["obj"], VC.place_pose(s["pose"], *place), s["local"], s["zoff"])
            for s in [s for s in sup if s["obj"] not in gone] + cols
        ],
        tree=sup.tree,
    )


def _fixed_at(sup: SceneSupports, place) -> tuple[set, list]:
    """A base tree's fixed objects re-decided at a candidate placement (static_fixed.at — the formula the bake stamps
    on the baked tree) → (item names that are not there, column items to add): an object the env does not spawn
    there (a non-receiver on the robot's standing spot) loses its body and its pedestal; one standing off the desk box
    gets its column from the floor instead of the base tree's own (desk column · none)."""
    fx = sup.fixed
    if fx is None:
        return set(), []
    from dexengine.placement.scene import static_fixed as SF

    gone, cols = set(), []
    for o, up, base in SF.at(fx, place):
        ped = PEDESTAL_PREFIX + str(o["name"])
        if not up:
            gone |= {str(o["name"]), ped}
        elif base != SF.column_base(o, None, None, False):  # ≠ the base tree's own statement (pedestals())
            gone.add(ped)
            if base is not None:
                z = float(o["row"]["bottom_z_capture"]) + fx["zoff"]
                d = dict(o["row"], z=z, h=z - (VC.FLOOR_Z if base == "floor" else VC.desk_height()))
                pose, local = _pedestal_pose_local(d, fx["zoff"])
                cols.append({"obj": ped, "pose": pose, "local": local, "zoff": fx["zoff"]})
    return gone, cols


def _static(n: int):
    """Identity rotations and zero translations for n poses (a still body)."""
    return np.repeat(np.eye(3)[None], n, 0), np.zeros((n, 3))


def _in_box(p, lo, hi):
    """Points p (…, 3) strictly inside the box lo–hi."""
    return ((p > lo) & (p < hi)).all(-1)


_POSES = 64  # poses per block of the sample products below


def _still_sd(hA: dict, RA, tA, hB: dict) -> np.ndarray:
    """Signed surface distance of a moving hull hA (P poses) to a still hull hB (guard frame) per pose (P,) —
    arm_surface.hull_hull_sd's value wherever the two overlap (sd ≤ 0, the regime every threshold reads), without its
    full sample × plane product: a sample inside the other hull lies inside that hull's box, so only the samples inside
    the other body's box ⊕ NEAR_M meet its planes (the still hull's samples are first cut to the boxes the moving one
    sweeps). Clear poses (sd > 0) keep a floor, not a measurement: the sphere bound when NEAR_M apart, NEAR_M when no
    sample comes within a box, else the smallest plane value of the samples that do."""
    m = AS.NEAR_M
    ca = np.einsum("tij,j->ti", RA, hA["ctr"]) + tA
    sd = np.linalg.norm(ca - hB["ctr"], axis=-1) - hA["rad"] - hB["rad"]
    near = np.nonzero(sd < m)[0]
    if not len(near):
        return sd
    lo_a, hi_a = hA["v"].min(0) - m, hA["v"].max(0) + m  # moving hull's box, its link frame
    lo_b, hi_b = hB["v"].min(0) - m, hB["v"].max(0) + m  # still hull's box, guard frame
    vw = np.einsum("tij,vj->tvi", RA[near], hA["v"]) + tA[near, None, :]
    lo_w, hi_w = vw.min(1), vw.max(1)
    hit = ((lo_w < hi_b) & (hi_w > lo_b)).all(-1)  # the two boxes come within NEAR_M
    sd[near[~hit]] = m
    near = near[hit]
    if not len(near):
        return sd
    s_a, eq_a, eq_b = hA["smp"], hA["eq"], hB["eq"]
    s_b = hB["smp"][_in_box(hB["smp"], lo_w[hit].min(0) - m, hi_w[hit].max(0) + m)]
    val = np.full(len(near), np.inf)
    for i0 in range(0, len(near), _POSES):
        R, t = RA[near[i0 : i0 + _POSES]], tA[near[i0 : i0 + _POSES]]
        pw = np.einsum("tij,pj->tpi", R, s_a) + t[:, None, :]  # moving samples in the guard frame
        k = _in_box(pw, lo_b, hi_b)
        if k.any():
            np.minimum.at(val, i0 + np.nonzero(k)[0], (pw[k] @ eq_b[:, :3].T + eq_b[:, 3]).max(-1))
        if len(s_b):
            pb = np.einsum("tji,tpj->tpi", R, s_b[None, :, :] - t[:, None, :])  # still samples in the link frame
            k = _in_box(pb, lo_a, hi_a)
            if k.any():
                np.minimum.at(val, i0 + np.nonzero(k)[0], (pb[k] @ eq_a[:, :3].T + eq_a[:, 3]).max(-1))
    sd[near] = np.where(np.isfinite(val), val, m)
    return sd


def _min_sd(bodies: list, supports: list) -> list:
    """bodies = [(link, "cap", (a, b, r)) | (link, "hull", (h, R, t))] over P poses → [(minimum signed surface
    distance over the poses, link, object)] per body × support (a hull body through _still_sd — the supports stand
    still in the guard frame)."""
    rows = []
    for link, kind, g in bodies:
        Ri, ti = _static(len(g[0])) if kind == "cap" else (None, None)
        for s in supports:
            sd = min(
                float(
                    (
                        AS.hull_cap_sd(h, Ri, ti, g[0], g[1], g[2]) if kind == "cap" else _still_sd(g[0], g[1], g[2], h)
                    ).min()
                )
                for h in s["parts"]
            )
            rows.append((sd, link, s["obj"]))
    return rows


def arm_support_overlap(ctx: dict, q_all, z_off: float, supports: list, subdiv: int = 8) -> tuple[float, str, str]:
    """Deepest arm-body overlap into a support object over the frames and the retiming chords (m, positive = overlap
    depth; −1e9 with no support or the arm cell OFF) · that link · that object. `supports` = scene_supports' return
    (checked — anything else dies before the empty test)."""
    if not checked(supports) or not arm_measured():
        return -1e9, "", ""
    from dexengine.placement.collision import core as SCL

    q = SCL._chords(np.asarray(q_all, float), subdiv)
    Ts = arm_link_transforms(ctx, q)
    base = ctx["base"] + np.array([0.0, 0.0, z_off])
    bodies = []
    for c in ctx["surf"]["caps"]:
        a, b = capsule_endpoints(c, Ts[:, c["k"]])
        bodies.append((c["link"], "cap", (a + base, b + base, float(c["r"]))))
    for h in ctx["surf"]["hulls"]:
        Tk = Ts[:, h["k"]]
        bodies.append((h["link"], "hull", (h, Tk[:, :3, :3], Tk[:, :3, 3] + base)))
    sd, link, obj = min(_min_sd(bodies, supports))
    return -sd, link, obj


def hand_support_mm(ctx: dict, q0, fingers0, z_off: float, supports: list) -> tuple[float, str, str]:
    """Hand ↔ support report: the deepest hand-body (palm + every finger group of the torso guard's hand tree) overlap
    into a support object at one pose (mm, ≥ 0) · that body · that object. q0 (1, nj) · fingers0 (1, n). Measured only
    when SUPPORT_COLLIDE["hand"] is ON; never a reason (HAND_SUPPORT_MODE)."""
    if HAND_SUPPORT_MODE != "report":
        raise ValueError(f"HAND_SUPPORT_MODE={HAND_SUPPORT_MODE!r} — the only known mode is report")
    if not checked(supports) or _presets().expected_support(1, "hand") != "ON":
        return 0.0, "", ""
    from dexengine.placement.collision import hand_surface as HS

    ht = ctx["torso"]["hand"]
    q0, f0 = np.asarray(q0, float).reshape(1, -1), np.asarray(fingers0, float).reshape(1, -1)
    nj = q0.shape[1]
    W = arm_link_transforms(ctx, q0)[:, nj - 1][:, None] @ HS.finger_frames(ht, f0)
    base = ctx["base"] + np.array([0.0, 0.0, z_off])
    bodies = [
        (
            g["root"] if gi else "palm",
            "hull",
            (
                AS.hull_record(str(g["root"]), -1, np.asarray(g["verts"], float)),
                W[:, gi, :3, :3],
                W[:, gi, :3, 3] + base,
            ),
        )
        for gi, g in enumerate(ht["groups"])
        if g["verts"] is not None
    ]
    sd, link, obj = min(_min_sd(bodies, supports))
    return max(0.0, -sd * 1000.0), link, obj
