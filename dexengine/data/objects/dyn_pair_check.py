"""Rule S-dyn-pairs, the visual check: how deep two posed bodies pass into each other on their VISUAL meshes, read
on the recorded reference (drop (a), `dyn_pairs.dyn_pairs`) and on a delivered training output (drop (b),
`output_check`).

Why the visual mesh: a both-dynamic pair that overlaps at frame 0 has its collision off in training (collision_presets
DYN_PAIR_COLLIDE), so physics no longer keeps the two apart; the visual meshes are the surfaces the recording was
captured against and the delivered data shows.
Ruler (`ruler`): a `Visual` — the object's visual mesh as collider_shape.Triangles per rigid body, posed at a qpos
row (`links`): a rigid object is one body, collider_shape.triangles (the visual mesh of its training USD); an
articulated object is its base link and its moving link (object_surface.link_mesh: the link's parts with their fixed
attachments, the meshes its USD is built from), the moving link posed by the joint FK at the row's joint value
(data/artic/fk.artic_moving_vertices; the column from qpos_layout.artic_joint_cols). Read both ways
(collider_shape.pair_depth's reading) for every link pair, a bounding-sphere test first, at every frame
(touch_gate.FRAME_STRIDE). A body that cannot be read raises: every pair is measured (no hold).
Screen (`screened`): the question is only "deeper than the cap?". A point reads at most its distance to the nearest
surface sample of the other body (the samples lie on the faces the ruler searches, the nearest one's face among them),
so only points farther than the cap from every sample, inside the other body's box, can read deeper; only those are
read. The reading is therefore pair_depth's wherever it is above the cap, and at most the cap elsewhere (`exact`): the
verdict never differs (tests/data/test_dyn_pairs.py compares it with pair_depth). Measured on recorded pairs: on
closed meshes it leaves 1-3 % of the in-box points; on big open meshes it leaves almost all of them, and a contact
frame there costs about 10 s (a 144k-face plate).
Order (`depth_over`): the verdict needs one frame above the cap, so frames are read in the order given (the statement
gives them deepest first on the collision shapes) and the reading stops at the first one above it; that reading and
frame are returned, not the clip's maximum. A clip with no frame above the cap is read to the end.
Resolution: a vertex or a face of either mesh reads its exact depth; a deepest point inside a face of the deeper body
reads at most the sample spacing low.
Verdict (`verdict`): deeper than scene_requirement.TRUE_SHAPE_OVERLAP_M drops the trajectory (reason DROP_WORD).
Marker `[DYN_PAIRS] output …`, one line per checked output.
"""

from __future__ import annotations

import numpy as np

from dexengine.data.objects import collider_shape as CS
from dexengine.data.objects import scene_requirement as SR
from dexengine.data.objects import touch_gate as TG

MARK = "[DYN_PAIRS]"
DROP, KEEP = "drop", "keep"
DROP_WORD = "dyn_pair_visual_overlap"  # exclusion reason word of a trajectory dropped by this check (a) or (b)


class Visual:
    """An object's visual body (module doc: Ruler): `shapes` = collider_shape.Triangles per rigid body (link frame);
    `links(qrow)` = [(ruler, 4x4 link pose in the object frame)] — the identity for a rigid object, the base at the
    identity and the moving link at `fk_pose` of the row's joint value for an articulated one."""

    def __init__(self, shapes: list, art: dict | None = None, col: int | None = None):
        self.shapes, self.art, self.col = shapes, art, col

    def links(self, qrow) -> list:
        if self.art is None:
            return [(self.shapes[0], np.eye(4))]
        return [(self.shapes[0], np.eye(4)), (self.shapes[1], fk_pose(self.art, float(qrow[self.col])))]


def fk_pose(art: dict, theta: float) -> np.ndarray:
    """4x4 pose of an articulation's moving link (child frame) in the base frame at joint value theta: the joint FK's
    image of the child frame's origin and axes (data/artic/fk.artic_moving_vertices, the FK owner; exact for its rigid
    map)."""
    from dexengine.data.artic.fk import artic_moving_vertices  # noqa: PLC0415

    V = artic_moving_vertices(art, np.vstack([np.zeros(3), np.eye(3)]), theta)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = (V[1:] - V[0]).T, V[0]
    return T


_LINKS: dict = {}


def _link_triangles(ds: str, o: dict) -> list:
    """[base, moving] collider_shape.Triangles of an articulated object's links (object_surface.link_mesh, cached)."""
    from dexengine.core import paths as P  # noqa: PLC0415
    from dexengine.data.object_surface import link_mesh  # noqa: PLC0415

    parts = o["articulation"]["parts"]
    key = (ds, str(o["mesh_dir"]), repr(sorted((k, repr(v)) for k, v in parts.items())))
    if key not in _LINKS:
        src = P.main_root() / P.data_layout.object_path(ds, o["mesh_dir"])
        ms = [link_mesh(src, parts, link, process=False) for link in ("base", "moving")]
        _LINKS[key] = [CS.Triangles(np.asarray(m.vertices, float), np.asarray(m.faces)) for m in ms]
    return _LINKS[key]


def ruler(ds: str, o: dict, tree_dir=None) -> Visual:
    """The visual body of task_info object `o` (module doc: Ruler); an articulated one reads its joint column from the
    tree's qpos layout (`tree_dir` required). A body that cannot be read raises."""
    if not o.get("articulation"):
        return Visual([CS.triangles(ds, str(o["obj_id"]))])
    if tree_dir is None:
        raise ValueError(
            f"{MARK} articulated {o.get('name')!r}: the tree (qpos layout) is required for its joint column"
        )
    from dexengine.core.paths import qpos_layout as QL  # noqa: PLC0415 — owner of the joint columns

    art = o["articulation"]
    col = QL.read(str(tree_dir), nq=None).artic_joint_cols()[str(art["joint"])]
    return Visual(_link_triangles(ds, o), art, int(col))


def _xf(M: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ M[:3, :3].T + M[:3, 3]


def screened(ra, rb, M_ba: np.ndarray, cap: float = SR.TRUE_SHAPE_OVERLAP_M) -> float:
    """collider_shape.pair_depth of rulers ra, rb (M_ba = b's frame in a's frame) where it is above `cap`, at most cap
    elsewhere (module doc: Screen)."""
    best = 0.0
    for r, P in ((ra, _xf(M_ba, rb.points)), (rb, _xf(np.linalg.inv(M_ba), ra.points))):
        P = P[np.all((r.lo < P) & (r.hi > P), axis=1)]
        if len(P):
            d, _i = r.tree.query(P, k=1, distance_upper_bound=cap)
            if np.isinf(d).any():
                best = max(best, r.depth(P[np.isinf(d)]))
    return best


def exact(depth_m: float, cap: float = SR.TRUE_SHAPE_OVERLAP_M) -> bool:
    """Is a `screened` reading the ruler's own value (above the cap), not a bound?"""
    return depth_m > cap


def depth_over(va: Visual, vb: Visual, q: np.ndarray, Ta: np.ndarray, Tb: np.ndarray, frames=None) -> tuple:
    """(reading m, frame) of visual bodies va, vb at object poses Ta, Tb ((T,4,4) world; q = the qpos rows their links
    read) over `frames` in their order (default every FRAME_STRIDE-th frame): the first `screened` link-pair reading
    above the cap, else the largest (at most the cap; frame -1 when nothing reads)."""
    frames = range(0, len(Ta), TG.FRAME_STRIDE) if frames is None else frames
    best, at = 0.0, -1
    for f in frames:
        for ra, La in va.links(q[f]):
            Ma = Ta[f] @ La
            for rb, Lb in vb.links(q[f]):
                Mb = Tb[f] @ Lb
                ca, cb = Ma[:3, :3] @ ra.center + Ma[:3, 3], Mb[:3, :3] @ rb.center + Mb[:3, 3]
                if np.linalg.norm(ca - cb) > ra.radius + rb.radius:
                    continue
                d = screened(ra, rb, np.linalg.inv(Ma) @ Mb)
                if exact(d):
                    return d, int(f)
                if d > best:
                    best, at = d, int(f)
    return best, at


def verdict(depth_m: float) -> str:
    """DROP above the true-shape overlap, else KEEP."""
    return DROP if depth_m > SR.TRUE_SHAPE_OVERLAP_M else KEEP


def output_check(ds: str, objs: dict, pairs: list, poses: dict, joints: dict | None = None) -> dict:
    """Drop (b): the visual reading of each pair (a, b) on a delivered output — `objs` {name: task_info object entry
    (obj_id; mesh_dir and articulation for an articulated one)}, `poses` {name: (T,4,4) world poses of the output},
    `joints` {name: (T,) joint values of the output} for every articulated object. {pairs: [{a, b, depth_m, frame,
    verdict}], drop: (DROP_WORD, description) | None}. A pair without its poses or an articulated object without its
    joint values fails."""
    out = []
    for a, b in pairs:
        if a not in poses or b not in poses:
            raise KeyError(f"{MARK} output: no poses for {a if a not in poses else b} (pair {a}×{b})")
        q, vis = [], []
        for k, n in enumerate((a, b)):
            o = objs[n]
            if o.get("articulation"):
                if n not in (joints or {}):
                    raise KeyError(f"{MARK} output: articulated {n} has no joint values (pair {a}×{b})")
                q.append(np.asarray(joints[n], float).reshape(-1))
                vis.append(Visual(_link_triangles(ds, o), o["articulation"], k))
            else:
                q.append(np.zeros(len(poses[n])))
                vis.append(Visual([CS.triangles(ds, str(o["obj_id"]))]))
        Ta, Tb = np.asarray(poses[a], float), np.asarray(poses[b], float)
        d, f = depth_over(vis[0], vis[1], np.column_stack(q), Ta, Tb)
        out.append({"a": a, "b": b, "depth_m": round(d, 4), "frame": f, "verdict": verdict(d)})
    deep = [f"{p['a']}×{p['b']} {reading(p['depth_m'])} (frame {p['frame']})" for p in out if p["verdict"] == DROP]
    return {"pairs": out, "drop": (DROP_WORD, "output visual overlap " + " · ".join(deep)) if deep else None}


def reading(depth_m: float) -> str:
    """'6.2 mm' for an exact reading, '≤ 5.0 mm' for a bound (`exact`)."""
    return f"{depth_m * 1e3:.1f} mm" if exact(depth_m) else f"≤ {SR.TRUE_SHAPE_OVERLAP_M * 1e3:.1f} mm"
