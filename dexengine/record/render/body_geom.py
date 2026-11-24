"""Geometry helpers of the render body layer (record.render.body): the desk box from the scene config, signed distance
to a box, drawn-mesh vertices of a prim (mesh_points) and the highest point under a footprint (top_under). Part of
the render fingerprint (record.cache.RENDER_CODE_FILES)."""

from __future__ import annotations


def desk_box(desk: dict):
    """Desk top box (env frame): Rig.desk_local · thickness dexengine.core.desk.DESK_TOP_THICK."""
    import numpy as np

    from dexengine.core import desk as DP

    return (
        np.array([desk["near_x"], desk["cy"] - desk["half_y"], desk["top_z"] - DP.DESK_TOP_THICK]),
        np.array([desk["far_x"], desk["cy"] + desk["half_y"], desk["top_z"]]),
    )


def box_sdist(P, box):
    """Signed distance [N] of points [N, 3] to an axis-aligned box (inside = negative, to the nearest face)."""
    import numpy as np

    d = np.maximum(box[0] - P, P - box[1])
    return np.where((d <= 0).all(axis=1), d.max(axis=1), np.linalg.norm(np.maximum(d, 0.0), axis=1))


def mesh_points(prim, frame=None, faces: bool = False, purposes=None):
    """Every Mesh vertex under prim (instance proxies included) in the frame prim's space (prim when no frame) [N, 3].
    With faces: (vertices, triangle indices [T, 3]; polygons fanned). With purposes (e.g. ("default", "render")):
    those purposes only."""
    import numpy as np

    from pxr import Usd, UsdGeom

    t = Usd.TimeCode.Default()
    inv = np.linalg.inv(np.array(UsdGeom.Xformable(frame or prim).ComputeLocalToWorldTransform(t)).T)
    out, tri, n0 = [], [], 0
    for m in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        keep = m.IsA(UsdGeom.Mesh) and (purposes is None or UsdGeom.Imageable(m).ComputePurpose() in purposes)
        pts = UsdGeom.Mesh(m).GetPointsAttr().Get(t) if keep else None
        if pts:
            M = inv @ np.array(UsdGeom.Xformable(m).ComputeLocalToWorldTransform(t)).T
            out.append(np.array(pts, float) @ M[:3, :3].T + M[:3, 3])
            if faces:
                cnt = UsdGeom.Mesh(m).GetFaceVertexCountsAttr().Get(t) or []
                idx = np.asarray(UsdGeom.Mesh(m).GetFaceVertexIndicesAttr().Get(t) or [], np.int64)
                k = 0
                for c in cnt:
                    tri += [(n0 + idx[k], n0 + idx[k + j], n0 + idx[k + j + 1]) for j in range(1, c - 1)]
                    k += c
            n0 += len(pts)
    P = np.concatenate(out) if out else np.zeros((0, 3))
    return (P, np.array(tri, np.int64).reshape(-1, 3)) if faces else P


def top_under(P, tris):
    """Per point [N, 3], the highest z where the vertical line through its xy meets the triangles [T, 3, 3] (-inf if
    none): the top height of a lump lying on the floor (base). Only triangles over the points' xy box are checked."""
    import numpy as np

    top = np.full(len(P), -np.inf)
    if not len(P) or not len(tris):
        return top
    lo, hi = P[:, :2].min(0), P[:, :2].max(0)
    keep = (tris[:, :, :2].max(1) >= lo).all(1) & (tris[:, :, :2].min(1) <= hi).all(1)
    T = tris[keep]
    for i in range(0, len(T), 512):
        a, b, c = T[i : i + 512, 0], T[i : i + 512, 1], T[i : i + 512, 2]
        v0, v1 = (b - a)[:, :2], (c - a)[:, :2]
        den = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
        ok = np.abs(den) > 1e-12
        a, b, c, v0, v1, den = a[ok], b[ok], c[ok], v0[ok], v1[ok], den[ok]
        w = P[:, None, :2] - a[None, :, :2]  # [N, T, 2]
        u = (w[..., 0] * v1[:, 1] - w[..., 1] * v1[:, 0]) / den
        v = (v0[:, 0] * w[..., 1] - v0[:, 1] * w[..., 0]) / den
        inside = (u >= 0) & (v >= 0) & (u + v <= 1)
        z = a[:, 2] + u * (b - a)[:, 2] + v * (c - a)[:, 2]
        top = np.maximum(top, np.where(inside, z, -np.inf).max(1))
    return top
