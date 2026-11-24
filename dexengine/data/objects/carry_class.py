"""How a carried extra rides its carrier (S-carried, measure and label): held by the carrier's geometry (`inserted`:
a pencil in a sharpener, a tube in a clamp, a cap on its tube) or resting freely on it (`resting`: bread on a plate, a
spoon lying in a bowl). A label is a review statement: no scene builder reads it.

Measured over the carry interval (`carry` of a scene_rules.carry_pass row: carrier, start, end), in the carrier frame at
the interval's middle frame, on the shapes the simulation collides (cooked convex parts; an articulated body: its root
link's parts), each body represented by dense samples of its parts' hull surfaces (DENSE_M):
  overlap_m     how deep rider and carrier interpenetrate at the recorded pose (both ways)
  lift_m        the rider raised along up out of that overlap (steps of CLEAR_STEP_M, at most overlap + LIFT_EXTRA_M)
                before the lateral probes: a rider sunk into its carrier would otherwise probe at the carrier's surface
                level and hit the seams between its parts
  (a) clearance_m   per horizontal direction ±x · ±y (world axes, in the carrier frame), from the lifted pose: how far
                the rider translates before a point outside the other body enters it by CLEAR_EPS_M through a wall —
                a face that is not a support face (normal more than SUPPORT_DEG from up or down); capped at CLEAR_MAX_M
  (b) support   rider–carrier contact points at the recorded pose (within CONTACT_M, or inside) on a carrier face whose
                normal is within SUPPORT_DEG of up (or a rider face within SUPPORT_DEG of down); supported when the
                rider's centroid, seen from above, lies within SUPPORT_GAP_M of their convex hull
  (c) tilt_deg  the largest tilt of the carrier's up axis from its first-frame direction over the frames of the
                interval where the rider keeps its relative pose (extra_slots_geom CARRY_REL_M · CARRY_REL_DEG)
  (d) inside_frac   share of rider samples inside the carrier's outer convex hull but outside its parts (a cavity)
      walls     rider–carrier contact points (within CONTACT_M, or inside) through a wall (not a support face)
Labels (first that applies):
  inserted   boxed — clearance ≤ CLEAR_M both ways along x or along y — or tilt > TILT_DEG with the rider still
  resting    supported and not boxed
  inserted   not supported but in wall contact: it rides without support from below, so the carrier holds it from the
             side (a cap on a tube end, a tube in a clamp — the cooked shapes may not enclose it)
  ambiguous  no support and no contact
Before the geometry: rider and carrier that are parts of one OakInk2 instance (task_info articulation_group) whose part
kind merges (oakink2 part_kinds: hinge · slide · fixed …) are one articulated object that D1-g joins — label
`articulation` (not a carry case; the tree still holds two bodies until D1-g runs on it). A free part of the same
instance (a removable lid or cap) keeps the geometric label and carries `same_group`.
The smallest clearance the reading gives is 2 × CLEAR_STEP_M (a wall touching at the lifted pose). The thresholds are
the named constants below (review values).
"""

from __future__ import annotations

import numpy as np

from dexengine.data.objects import convex_depth as CD
from dexengine.data.objects import extra_slots_geom as XG

CLEAR_M = 0.003  # lateral clearance at or below which a direction is blocked (m)
CLEAR_MAX_M = 0.03  # clearance search range (m); farther = free
CLEAR_STEP_M = 0.001  # clearance and lift search step (m)
CLEAR_EPS_M = 0.001  # entry depth that counts as contact (m)
LIFT_EXTRA_M = 0.002  # lift search beyond the recorded overlap (m)
DENSE_M = 0.002  # hull surface sample spacing (m)
CONTACT_M = 0.003  # a rider point this close to the carrier (or inside) is in contact (m)
SUPPORT_DEG = 30.0  # a support face's normal is within this of up (°)
SUPPORT_GAP_M = 0.01  # centroid (from above) to the support contacts' hull (m)
TILT_DEG = 45.0  # carrier tilt with the rider still above which the rider is held, not resting (°)
LABELS = ("inserted", "resting", "ambiguous", "articulation")
DIRS = ("+x", "-x", "+y", "-y")


def parts(ds: str, o: dict, row: dict | None = None) -> list | None:
    """Cooked convex parts of an object in its pose frame: a rigid body's parts, an articulated body's root link's.
    The measurement is convex-only: a body whose slot row stands as triangles (`collider_shape.row_shape`, a pinned
    still body) raises — a carrier moves, so it never does, and a reader never measures a shape physics does not use."""
    from dexengine.data.objects import collider_shape as CS  # noqa: PLC0415 — owner of the collider shape
    from dexengine.data.objects import hull as _oh  # noqa: PLC0415 — owner of the sim collision shapes

    if row is not None and CS.row_shape(ds, row) == CS.TRIANGLES:
        raise ValueError(f"carry_class: {o.get('name')} stands as triangles (pinned {row.get('kind')}) — convex only")

    try:
        links = _oh.collision_parts(ds, str(o["obj_id"]), bool(o.get("articulation"))).links
    except _oh.HullMissing:
        return None
    first = next(iter(links.values()), None)
    return list(first) if first else None


def body(pieces) -> dict:
    """A convex_depth shape whose points are dense samples of every part's hull surface (DENSE_M) plus the shape's own
    vertices, edge midpoints and face centres."""
    from scipy.spatial import ConvexHull, QhullError  # noqa: PLC0415

    from dexengine.placement.scene import ref_hand_geom as RH  # noqa: PLC0415 — the surface sampler

    sh = CD.shape(pieces)
    dense = [sh["points"]]
    for v in pieces:
        v = np.asarray(v, float)
        try:
            h = ConvexHull(v)
        except QhullError:
            h = ConvexHull(v, qhull_options="QJ")
        dense.append(RH.mesh_samples(h.points, h.simplices, DENSE_M))
    return dict(sh, points=np.concatenate(dense))


def _xf(M: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ M[:3, :3].T + M[:3, 3]


def _overlap(M: np.ndarray, rider: dict, carrier: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-point interpenetration (m) of the rider at relative pose M (rider frame → carrier frame), both ways: (rider
    points in the carrier's parts, carrier points in the rider's parts)."""
    return CD.depths(_xf(M, rider["points"]), carrier), CD.depths(_xf(np.linalg.inv(M), carrier["points"]), rider)


def _moved(M: np.ndarray, step: np.ndarray) -> np.ndarray:
    out = M.copy()
    out[:3, 3] += step
    return out


def lift(M: np.ndarray, up: np.ndarray, rider: dict, carrier: dict, overlap: float) -> tuple[np.ndarray, float]:
    """The rider raised along `up` (carrier frame) until the two bodies interpenetrate no deeper than CLEAR_EPS_M, at
    most overlap + LIFT_EXTRA_M: (pose, lift)."""
    top = overlap + LIFT_EXTRA_M
    for s in np.arange(0.0, top + 1e-9, CLEAR_STEP_M):
        Ms = _moved(M, s * up)
        if max(d.max(initial=0.0) for d in _overlap(Ms, rider, carrier)) <= CLEAR_EPS_M:
            return Ms, round(float(s), 4)
    return _moved(M, top * up), round(float(top), 4)


def _wall_entries(M: np.ndarray, up: np.ndarray, rider: dict, carrier: dict, free: tuple) -> float:
    """Deepest entry (m) through a wall at pose M, of points outside the other body at the probe start (`free`)."""
    cos = np.cos(np.radians(SUPPORT_DEG))
    s_r, n_r = CD.signed(_xf(M, rider["points"]), carrier)
    s_c, n_c = CD.signed(_xf(np.linalg.inv(M), carrier["points"]), rider)
    wall_r = np.abs(n_r @ up) < cos
    wall_c = np.abs((n_c @ M[:3, :3].T) @ up) < cos  # rider face normals in the carrier frame
    d_r = np.where(free[0] & wall_r & (s_r < 0), -s_r, 0.0)
    d_c = np.where(free[1] & wall_c & (s_c < 0), -s_c, 0.0)
    return max(d_r.max(initial=0.0), d_c.max(initial=0.0))


def clearance(M: np.ndarray, horiz: dict, up: np.ndarray, rider: dict, carrier: dict) -> dict:
    """(a): per direction, the translation (m) at which a point outside the other body at M enters it by CLEAR_EPS_M
    through a wall (CLEAR_MAX_M + step: free)."""
    free = tuple(d == 0.0 for d in _overlap(M, rider, carrier))
    out = {}
    for k, u in horiz.items():
        out[k] = round(CLEAR_MAX_M + CLEAR_STEP_M, 4)
        for s in np.arange(CLEAR_STEP_M, CLEAR_MAX_M + 1e-9, CLEAR_STEP_M):
            if _wall_entries(_moved(M, s * u), up, rider, carrier, free) > CLEAR_EPS_M:
                out[k] = round(float(s), 4)
                break
    return out


def _hull_gap(P2: np.ndarray, c2: np.ndarray) -> float:
    """Distance (m) from point c2 to the convex hull of 2-D points P2 (0 inside)."""
    from scipy.spatial import ConvexHull, QhullError  # noqa: PLC0415

    if len(P2) >= 3:
        try:
            h = ConvexHull(P2)
            return max(0.0, float((h.equations[:, :2] @ c2 + h.equations[:, 2]).max()))
        except QhullError:
            pass
    a, b = P2[np.argmin(P2[:, 0] + P2[:, 1])], P2[np.argmax(P2[:, 0] + P2[:, 1])]
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip((c2 - a) @ ab / (ab @ ab), 0.0, 1.0))
    return float(np.linalg.norm(a + t * ab - c2))


def support(M: np.ndarray, up: np.ndarray, rider: dict, carrier: dict, cen: np.ndarray) -> dict:
    """(b): support contacts at pose M and whether the rider centroid `cen` (carrier frame) stands over them."""
    cos = np.cos(np.radians(SUPPORT_DEG))
    Pr, Pc = _xf(M, rider["points"]), carrier["points"]
    s_r, n_r = CD.signed(Pr, carrier, CONTACT_M)
    s_c, n_c = CD.signed(_xf(np.linalg.inv(M), Pc), rider, CONTACT_M)
    up_r, up_c = n_r @ up, (n_c @ M[:3, :3].T) @ up  # carrier / rider face normals · up (carrier frame)
    pts = np.concatenate([Pr[(s_r <= CONTACT_M) & (up_r >= cos)], Pc[(s_c <= CONTACT_M) & (up_c <= -cos)]])
    walls = int(((s_r <= CONTACT_M) & (np.abs(up_r) < cos)).sum() + ((s_c <= CONTACT_M) & (np.abs(up_c) < cos)).sum())
    if not len(pts):
        return {"supported": False, "contacts": 0, "gap_m": None, "walls": walls}
    e1 = np.cross(up, [1.0, 0.0, 0.0] if abs(up[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    E = np.stack([e1, np.cross(up, e1)])
    gap = _hull_gap(pts @ E.T, E @ cen)
    return {"supported": bool(gap <= SUPPORT_GAP_M), "contacts": int(len(pts)), "gap_m": round(gap, 4), "walls": walls}


def tilt(Tc: np.ndarray, Tr: np.ndarray) -> float:
    """(c): largest carrier tilt (°) from its first-frame up axis over the frames the rider keeps its relative pose."""
    rel = np.linalg.inv(Tc) @ Tr
    m = len(rel) // 2
    dp = np.linalg.norm(rel[:, :3, 3] - rel[m, :3, 3], axis=1)
    cosr = (np.einsum("kij,ij->k", rel[:, :3, :3], rel[m, :3, :3]) - 1.0) / 2.0
    still = (dp <= XG.CARRY_REL_M) & (np.degrees(np.arccos(np.clip(cosr, -1.0, 1.0))) <= XG.CARRY_REL_DEG)
    up0 = Tc[0, :3, :3].T @ np.array([0.0, 0.0, 1.0])
    ups = Tc[still, :3, :3] @ up0
    return float(np.degrees(np.arccos(np.clip(ups[:, 2], -1.0, 1.0))).max()) if still.any() else 0.0


def inside_frac(M: np.ndarray, rider: dict, carrier: dict) -> float:
    """(d): share of rider points inside the carrier's outer hull and outside every carrier part."""
    P = _xf(M, rider["points"])
    outer = CD.hulls([np.concatenate([h[0] for h in carrier["hulls"]])])[0]
    inside = np.all(P @ outer[1].T + outer[2] < 0, axis=1)
    if not inside.any():
        return 0.0
    return round(float((CD.depths(P[inside], carrier) == 0.0).sum()) / len(P), 3)


def label(m: dict) -> tuple[str, str]:
    """(label, reason) of a measurement (module doc)."""
    c = m["clearance_m"]
    boxed = [a for a in ("x", "y") if c[f"+{a}"] <= CLEAR_M and c[f"-{a}"] <= CLEAR_M]
    if boxed:
        return LABELS[0], f"clearance ≤ {CLEAR_M * 1e3:.0f} mm both ways along {'/'.join(boxed)}"
    if m["tilt_deg"] > TILT_DEG:
        return LABELS[0], f"carrier tilts {m['tilt_deg']:.0f}° > {TILT_DEG:.0f}° with the rider still"
    if m["support"]["supported"]:
        return LABELS[1], "supported from below, not boxed"
    if m["support"]["walls"]:
        return LABELS[0], "carried without support from below, in wall contact with the carrier (held from the side)"
    return LABELS[2], "not supported and not in contact: nothing in the cooked shapes explains why it rides"


def measure_bodies(pr: list, pc: list, Tc: np.ndarray, Tr: np.ndarray) -> dict:
    """(a)–(d) and the label from the rider's and the carrier's parts and their poses over the carry interval."""
    rider, carrier = body(pr), body(pc)
    mid = len(Tc) // 2
    M = np.linalg.inv(Tc[mid]) @ Tr[mid]
    Rt = Tc[mid, :3, :3].T
    up = Rt @ np.array([0.0, 0.0, 1.0])
    horiz = dict(
        zip(DIRS, (Rt @ e for e in np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], float)), strict=True)
    )
    cen = _xf(M, np.mean([h[0].mean(0) for h in rider["hulls"]], axis=0)[None])[0]
    overlap = max(float(d.max(initial=0.0)) for d in _overlap(M, rider, carrier))
    Ml, lifted = lift(M, up, rider, carrier, overlap)
    m = {
        "overlap_m": round(overlap, 4),
        "lift_m": lifted,
        "clearance_m": clearance(Ml, horiz, up, rider, carrier),
        "support": support(M, up, rider, carrier, cen),
        "tilt_deg": round(tilt(Tc, Tr), 1),
        "inside_frac": inside_frac(M, rider, carrier),
    }
    m["label"], m["reason"] = label(m)
    return m


def same_instance(o: dict, c: dict) -> dict | None:
    """{group, kind, merges} when rider `o` and carrier `c` are parts of one OakInk2 instance (articulation_group), else
    None. kind = the rider's part kind (oakink2 part_kinds; a part missing from the table raises there)."""
    go, gc = o.get("articulation_group") or {}, c.get("articulation_group") or {}
    if not go.get("group") or go.get("group") != gc.get("group"):
        return None
    from dexengine.data.oakink2 import part_kinds as PK  # noqa: PLC0415 — owner of the per-part joint kinds
    from dexengine.data.oakink2 import parts as OP  # noqa: PLC0415

    rid = OP.rid_of(str(o["obj_id"]))
    return {"group": go["group"], "kind": PK.kind_of(rid)[0], "merges": PK.merges(rid)}


def measure(ti: dict, q: np.ndarray, rider_name: str, carry: dict, rows: dict | None = None) -> dict:
    """(a)–(d) and the label of one carried row (`carry`: carrier, start, end in tree frames; `rows` = the clip's
    extra_slots rows by name, read for the collider shape)."""
    objs = {str(o["name"]): o for o in ti.get("objects") or []}
    ds, o, c = str(ti["dataset_name"]), objs[rider_name], objs[carry["carrier"]]
    same = same_instance(o, c)
    if same and same["merges"]:
        why = f"one articulated object (part kind {same['kind']}): D1-g joins rider and carrier — not a carry case"
        return {"label": LABELS[3], "reason": why, "same_group": same}
    rows = rows or {}
    pr, pc = parts(ds, o, rows.get(rider_name)), parts(ds, c, rows.get(carry["carrier"]))
    if pr is None or pc is None:
        return {
            "label": LABELS[2],
            "reason": "no cooked parts (rider or carrier)",
            **({"same_group": same} if same else {}),
        }
    a, b = int(carry["start"]), int(carry["end"]) + 1
    Tc, Tr = XG.pose_mats(q, int(c["qpos_start"]))[a:b], XG.pose_mats(q, int(o["qpos_start"]))[a:b]
    return {**measure_bodies(pr, pc, Tc, Tr), **({"same_group": same} if same else {})}


def rests_on(ds: str, rider: dict, carrier: dict, q: np.ndarray, f: int = 0) -> bool | None:
    """(b) at frame f: does object `rider` rest on object `carrier` — its centroid over their support contacts, on their
    cooked parts (`parts`, `support`)? None when either has no cooked parts. Read by S-dyn-pairs: a still that rides a
    main object loses its collision with it, so a main object resting on that still would lose its support."""
    pr, pc = parts(ds, rider), parts(ds, carrier)
    if pr is None or pc is None:
        return None
    rb, cb = body(pr), body(pc)
    Tr, Tc = XG.pose_mats(q, int(rider["qpos_start"]))[f], XG.pose_mats(q, int(carrier["qpos_start"]))[f]
    M = np.linalg.inv(Tc) @ Tr
    up = Tc[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    cen = _xf(M, np.mean([h[0].mean(0) for h in rb["hulls"]], axis=0)[None])[0]
    return bool(support(M, up, rb, cb, cen)["supported"])
