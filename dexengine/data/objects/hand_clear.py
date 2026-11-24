"""Rule S-hand-clear: how deep a reference hand sits inside a still extra, measured and kept as recorded.

Targets: the still rows (extra_slots STATIC_SPAWN) of every kind whose robot channel is ON in training or in the
delivered scene (`target_kinds`, read from collision_presets.extra_collide; with full physics in the delivered scene
that is every still kind, envelope included). Where the recorded reference hand sits inside one, nothing is moved: the
robot-static collision keeps the robot outside in training and the RL judge decides adoption. What the reading does to
the clip is the stage's (placement/repair/hand_clear_stage.ACTION: kept as recorded, reason word KEPT_WORD).

Ruler (per pair, on the shape physics uses): the convex hull of each reference-hand mesh geom of the tree scene.xml
(placement.scene.ref_hand_geom.RefHands — the hulls the robot USD collides) against the still extra's visual
triangles, the shape a pinned still stands as (collider_shape.pair_depth of the hull and the triangles, both ways;
`TriModel`), max over pairs. A still row that reads another shape raises (collider_shape.require_still_triangles).
Every frame of the tree is read, per hand side. A still extra without its shape cannot be measured: it raises (no
silent pass).
Verdict of a depth (`judge`): `clear` (at most scene_requirement.TRUE_SHAPE_OVERLAP_M) · `need` (deeper: the recorded
hand enters the still extra; measured only, the reference is kept as recorded).
"""

from __future__ import annotations

import numpy as np

from dexengine.data.objects import scene_requirement as SR

RULE_ID = "S-hand-clear"
KEPT_WORD = "hand_clear_kept_as_recorded"  # reason word of a side whose hand enters a still extra (kept as recorded)
VERDICTS = ("clear", "need")
CLEAR_TOL_M = 0.001  # a pair at or above -1 mm counts as free (capture contact error) — retarget/f0/repair_sep reads it
#: The 26 grid directions (unit vectors) — retarget/f0/repair_sep's move search reads them.
DIRS = np.array(
    [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)], float
)
DIRS /= np.linalg.norm(DIRS, axis=1, keepdims=True)
FRAME_BATCH = 64  # tree frames per batched depth call (a speed setting: the values do not depend on it)
PRUNE_EPS_M = 1e-9  # margin of the floor pruning (rounding of the surface distance)
POINT_CHUNK = 32768  # points per signed-distance call (a memory setting: the values do not depend on it)
HULL_CACHE_MAX = 512  # hulls kept by `_hull` (a memory setting: a hull's parts depend only on its vertices)
_HULLS: dict = {}


def _hull(CS, v: np.ndarray):
    """collider_shape.Parts of one hand hull, built once per process for its vertex values (the same reference-hand
    meshes recur for every still row and hand side): keyed by the vertices' shape and bytes, so equal keys are the
    same input and the shared Parts is the one a fresh build gives."""
    v = np.asarray(v, float)
    key = (v.shape, v.tobytes())
    got = _HULLS.get(key)
    if got is None:
        if len(_HULLS) >= HULL_CACHE_MAX:
            _HULLS.clear()
        got = _HULLS[key] = CS.Parts([v])
    return got


class TriModel:
    """Hand hulls `hands` (geom frame) against the still extra's visual triangles `tri` (collider_shape.Triangles,
    object frame), read by collider_shape.pair_depth: `place` a pose, then `depth` or `depth_poses`."""

    def __init__(self, hands: list, tri):
        from dexengine.data.objects import collider_shape as CS  # noqa: PLC0415

        self.cs, self.tri = CS, tri
        self.hulls = [_hull(CS, v) for v in hands]
        self.nh = len(hands)

    def place(self, hand_R, hand_t, piece_R, piece_t) -> None:
        inv = np.eye(4)
        inv[:3, :3] = np.asarray(piece_R, float).T
        inv[:3, 3] = -inv[:3, :3] @ np.asarray(piece_t, float)
        self.M = []
        for i in range(self.nh):
            G = np.eye(4)
            G[:3, :3], G[:3, 3] = hand_R[i], hand_t[i]
            self.M.append(inv @ G)

    def depth(self, hands=None) -> tuple[float, int]:
        """(largest penetration (m) over the hand hulls, the deepest hand index or -1)."""
        best, arg = 0.0, -1
        for i in range(self.nh) if hands is None else hands:
            h, M = self.hulls[i], self.M[i]
            if np.linalg.norm(M[:3, :3] @ h.center + M[:3, 3] - self.tri.center) > h.radius + self.tri.radius:
                continue
            d = self.cs.pair_depth(self.tri, h, M)
            if d > best:
                best, arg = d, i
        return best, arg

    def depth_poses(self, hand_R, hand_t, piece_R, piece_t, floor=0.0, hands=None) -> tuple[np.ndarray, np.ndarray]:
        """`place` + `depth(hands)` for each of n placements (hand_R (n,h,3,3) · hand_t (n,h,3) · piece_R (n,3,3) ·
        piece_t (n,3)), the same values above `floor`:
        every (placement, hull) pair that passes the sphere test reads collider_shape.pair_depth's two terms; a hull
        point nearer than `floor` to the body's surface samples cannot be deeper than `floor` (|signed| <= that
        distance), so only the other points are measured, all pairs' in one `Triangles.depths` call. A value <= `floor`
        is a lower bound (the caller compares with `floor` or more)."""
        pairs, pts = [], []
        for p in range(len(piece_t)):
            self.place(hand_R[p], hand_t[p], piece_R[p], piece_t[p])
            for i in range(self.nh) if hands is None else hands:
                h, M = self.hulls[i], self.M[i]
                if np.linalg.norm(M[:3, :3] @ h.center + M[:3, 3] - self.tri.center) > h.radius + self.tri.radius:
                    continue
                near = self.tri.near(self.cs._xf(M, h.center[None])[0], h.radius)
                into = h.depth(self.cs._xf(np.linalg.inv(M), near)) if len(near) else 0.0
                pairs.append((p, i, into))
                pts.append(self.cs._xf(M, h.points))
        best, arg = np.zeros(len(piece_t)), np.full(len(piece_t), -1)
        if not pairs:
            return best, arg
        P = np.concatenate(pts)
        d = np.zeros(len(P))
        keep = np.flatnonzero(np.all((self.tri.lo < P) & (self.tri.hi > P), axis=1))
        if len(keep) and floor > 0:  # bounded query: a point with no sample within floor reads inf (kept)
            nn, _ = self.tri.tree.query(P[keep], k=1, distance_upper_bound=floor)
            keep = keep[nn >= floor - PRUNE_EPS_M]
        for c0 in range(0, len(keep), POINT_CHUNK):  # bounded temporaries (the measure is per point)
            k = keep[c0 : c0 + POINT_CHUNK]
            d[k] = self.tri.depths(P[k])  # every point is in the box: max(-signed, 0)
        e = 0
        for (p, i, into), x in zip(pairs, pts, strict=True):
            v = max(float(d[e : e + len(x)].max()), into)
            e += len(x)
            if v > best[p]:
                best[p], arg[p] = v, i
        return best, arg


def judge(depth_m: float) -> str:
    """Verdict of one deepest recorded overlap (module doc)."""
    return "clear" if depth_m <= SR.TRUE_SHAPE_OVERLAP_M else "need"


def _hand_meshes(ref) -> tuple[list, list]:
    """(hull vertices per reference-hand mesh geom, side per geom) of RefHands `ref` (its geom order)."""
    from scipy.spatial import ConvexHull  # noqa: PLC0415

    verts = [ConvexHull(ref._mesh(int(ref.m.geom_dataid[g]))[0]).points for g in ref.geoms]
    return verts, list(ref.gside)


def _model(ds: str, o: dict, row: dict, verts: list):
    """The pair model of one still row: its triangles (module doc)."""
    from dexengine.data.objects import collider_shape as CS  # noqa: PLC0415 — owner of the collider shape

    CS.require_still_triangles(ds, row)
    return lambda hands: TriModel([verts[i] for i in hands], CS.triangles(ds, str(o["obj_id"])))


def row_verdict(ref, ds: str, o: dict, T: np.ndarray, row: dict) -> dict:
    """{verdict, depth_m, frame, side, sides} of one still extra `o` (its row `row`, pose per frame T (n,4,4)) against
    the reference hands `ref`: per hand side the deepest reading over every frame (`sides` {side: depth_m}), the row's
    verdict, depth and frame from the deepest side (`side` "" when clear)."""
    verts, gside = _hand_meshes(ref)
    make = _model(ds, o, row, verts)
    sides, frames = {}, {}
    for s in sorted(set(gside)):
        on = [i for i, x in enumerate(gside) if x == s]
        pm, gR, gx = make(on), ref.gR[:, on], ref.gx[:, on]
        best, frame = 0.0, -1
        for k0 in range(0, len(T), FRAME_BATCH):
            sl = slice(k0, k0 + FRAME_BATCH)
            dks, _args = pm.depth_poses(gR[sl], gx[sl], T[sl, :3, :3], T[sl, :3, 3], floor=best)
            j = int(np.argmax(dks))
            if dks[j] > best:
                best, frame = float(dks[j]), k0 + j
        sides[s], frames[s] = round(best, 4), frame
    deep = max(sides, key=lambda s: sides[s], default="")
    depth = sides.get(deep, 0.0)
    v = judge(depth)
    return {
        "verdict": v,
        "depth_m": depth,
        "frame": frames.get(deep, -1),
        "side": deep if v == "need" else "",
        "sides": sides,
    }


def target_kinds(ds: str) -> tuple:
    """The still kinds this rule judges for dataset `ds`: those whose robot channel is ON in training or in the
    delivered scene (collision_presets.extra_collide through the registry window — the one owner of the robot cells)."""
    from dexengine.core import paths as P  # noqa: PLC0415
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415

    cp = P.load_registry("collision_presets")
    still = (XS.KIND_TOUCHED, XS.KIND_RECEIVER, XS.KIND_TOUCHED_SOLID, XS.KIND_ENVELOPE)
    return tuple(k for k in still if "ON" in (cp.extra_collide(k, full=f, ds=ds)["hand"] for f in (False, True)))


def clip_verdicts(tree_dir, ti: dict, q: np.ndarray, rows: list[dict]) -> dict:
    """{row name: row_verdict} for the still rows (STATIC_SPAWN) of the target kinds of one clip; {} when none or
    when the rule is off for the dataset."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415
    from dexengine.data.objects import extra_slots_geom as XG  # noqa: PLC0415
    from dexengine.data.objects import scene_rules as SC  # noqa: PLC0415
    from dexengine.placement.scene import ref_hand_geom as RH  # noqa: PLC0415 — needs mujoco

    ds = str(ti["dataset_name"])
    kinds = target_kinds(ds)
    still = [r for r in rows if r.get("cls") == XS.STATIC_SPAWN and r.get("kind") in kinds]
    if not still or not SC.on(RULE_ID, ds):
        return {}
    objs, ref = {str(o["name"]): o for o in ti.get("objects") or []}, RH.RefHands(tree_dir, ti, q)
    return {
        r["name"]: row_verdict(ref, ds, objs[r["name"]], XG.pose_mats(q, int(objs[r["name"]]["qpos_start"])), r)
        for r in still
    }
