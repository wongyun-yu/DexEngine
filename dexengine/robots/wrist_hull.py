"""Arm collider hull geometry — the convex hull of a link's URDF ``<visual>`` meshes, reduced to the PhysX vertex cap.

Why the visual mesh: it is the real surface of the link. The one generator of the arm colliders
(dexengine/placement/collision/sim_collider.py) calls ``link_hull`` for every revised link; the USD generator writes
those vertices and the judge measures with them, so the sim shape and the judge shape are the same thing. The rule is
applied to whatever ``<visual>`` elements the URDF link has (origin, scale, any number of meshes) — no mesh file name is
written here.

Why the reduction: PhysX cooks a convex mesh with a vertex cap (64 by default — what the hand hulls of these USDs are
cooked with). Handing it more points lets PhysX choose its own reduction and the cooked shape is then unknown. Here
the hull is reduced deterministically to a **subset of its own vertices** (greedy farthest-point insertion), so the
cooked shape is exactly the hull written into the USD, it never pokes outside the visual mesh, and the worst inward
error is measured and recorded.

Pure numpy/scipy/trimesh (no pxr), so tests and measurements import it without a USD stack.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from dexengine.core import paths as P
from dexengine.core.errors import InputMissingError
from dexengine.placement.collision import geom_cache as GC

_ROOTS: dict = {}  # geom_cache.file_identity(urdf) → parsed URDF root (read here only, never changed)
_SERVER_KEEP = ("_ROOTS",)


def rpy_matrix(rpy) -> np.ndarray:
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def visual_points(urdf: Path, link: str) -> tuple[np.ndarray, list[dict]]:
    """All vertices of ``link``'s ``<visual>`` meshes in the link frame, plus one record per mesh (file, sha256)."""
    v, _f, recs = visual_mesh(urdf, link)
    return v, recs


def visual_mesh(urdf: Path, link: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """(vertices, triangles, records) of ``link``'s ``<visual>`` meshes in the link frame — the one URDF visual reader.
    A mesh file is read through geom_cache.mesh_arrays (trimesh, cached by the file bytes).

    Dies when the link is absent or has no mesh visual — a collider cannot be derived from nothing.
    """
    k = GC.file_identity(urdf)
    if k not in _ROOTS:
        _ROOTS[k] = ET.parse(str(urdf)).getroot()
    root = _ROOTS[k]
    el = next((ln for ln in root.findall("link") if ln.get("name") == link), None)
    if el is None:
        raise InputMissingError(f"[wrist_hull] link {link} is not in the URDF: {urdf}")
    pts, tris, recs, off = [], [], [], 0
    for vis in el.findall("visual"):
        mesh = vis.find("geometry/mesh")
        if mesh is None:
            continue
        o = vis.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
        R = rpy_matrix((o.get("rpy", "0 0 0") if o is not None else "0 0 0").split())
        sc = mesh.get("scale")
        scale = np.array([float(v) for v in sc.split()]) if sc else np.ones(3)
        fn = P.urdf_mesh(urdf, mesh.get("filename"))
        if not fn.is_file():
            raise InputMissingError(f"[wrist_hull] {link}: visual mesh file missing: {fn}")
        mv, mf, sha = GC.mesh_arrays(fn)
        v = mv * scale
        pts.append(v @ R.T + xyz)
        tris.append(mf + off)
        off += len(v)
        recs.append({"file": str(fn), "sha256": sha, "n_vertices": len(v)})
    if not pts:
        raise InputMissingError(f"[wrist_hull] {link} has no mesh <visual>: {urdf}")
    return np.concatenate(pts), np.concatenate(tris), recs


def _outside(hull: ConvexHull, pts: np.ndarray) -> np.ndarray:
    """Distance (m) of each point outside the hull (0 inside) — max over the facet planes."""
    d = pts @ hull.equations[:, :3].T + hull.equations[:, 3]
    return np.maximum(d.max(axis=1), 0.0)


def reduce_hull(points: np.ndarray, cap: int) -> tuple[np.ndarray, dict]:
    """``_reduce_hull`` through the geometry disk cache (geom_cache: key = the points' bytes · cap · this file's code ·
    numpy/scipy versions) — a pure function of its input, so a cached value is the value."""
    pts = np.asarray(points)
    return GC.cached(
        "hull", GC.code_rev((__file__,), ("numpy", "scipy")), (pts, f"cap={int(cap)}"), lambda: _reduce_hull(pts, cap)
    )


def _reduce_hull(points: np.ndarray, cap: int) -> tuple[np.ndarray, dict]:
    """Subset of the full hull's vertices with at most ``cap`` vertices whose hull best covers the full hull.

    Start = the 14 extreme vertices along the axes and the cube diagonals; then repeatedly add the full-hull vertex
    farthest outside the current hull. Stops at the cap (the step that would exceed it is undone) or when nothing is
    outside. Returns (vertices, stats) — stats carry the worst inward error in mm and the volume ratio.
    """
    full = ConvexHull(points)
    V = points[full.vertices]
    dirs = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        + [[sx, sy, sz] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)],
        float,
    )
    sel = list(dict.fromkeys(int(np.argmax(V @ d)) for d in dirs))
    hull = ConvexHull(V[sel])
    while True:
        out = _outside(hull, V)
        k = int(np.argmax(out))
        if out[k] <= 1e-9:
            break
        trial = ConvexHull(V[sel + [k]])
        if len(trial.vertices) > cap:
            break
        sel.append(k)
        hull = trial
    keep = V[sel][hull.vertices]
    hull = ConvexHull(keep)
    stats = {
        "full_hull_vertices": int(len(V)),
        "vertices": int(len(keep)),
        "max_inward_mm": round(float(_outside(hull, V).max()) * 1e3, 3),
        "volume_ratio": round(float(hull.volume / full.volume), 4),
        "volume_cm3_full": round(float(full.volume) * 1e6, 1),
        "volume_cm3": round(float(hull.volume) * 1e6, 1),
    }
    return keep, stats


def link_hull(urdf: Path, link: str, cap: int) -> tuple[np.ndarray, dict, list[dict]]:
    """The sim collider of one revised link: the hull of its ``<visual>`` meshes reduced to ``cap`` vertices.

    Called by sim_collider.bodies only (the USD generator and the collision sets read its result).
    Returns (vertices, stats, meshes).
    """
    pts, meshes = visual_points(Path(urdf), link)
    verts, stats = reduce_hull(pts, cap)
    return verts, stats, meshes


def hull_triangles(vertices: np.ndarray) -> np.ndarray:
    """Outward-oriented triangles (indices into ``vertices``) of the hull of ``vertices`` (all of them are extreme)."""
    hull = ConvexHull(vertices)
    if len(hull.vertices) != len(vertices):
        raise ValueError(
            f"{len(hull.vertices)} hull vertices out of {len(vertices)} — every reduced vertex must be an extreme point"
        )
    tris = hull.simplices.copy()
    c = vertices.mean(axis=0)
    for i, t in enumerate(tris):
        a, b, d = vertices[t]
        if np.dot(np.cross(b - a, d - a), a - c) < 0:
            tris[i] = t[[0, 2, 1]]
    return tris


def capsule_outside_mm(points: np.ndarray, p0: np.ndarray, p1: np.ndarray, radius: float) -> float:
    """How far (mm) the points reach outside a capsule (axis p0–p1, link frame); 0 = all inside. Recorded per replaced
    link: how the hull differs from the vendor capsule it replaces."""
    ax = np.asarray(p1, float) - np.asarray(p0, float)
    t = np.clip((points - p0) @ ax / max(float(ax @ ax), 1e-18), 0.0, 1.0)
    d = np.linalg.norm(points - (np.asarray(p0, float) + t[:, None] * ax), axis=1)
    return round(float(max(d.max() - radius, 0.0)) * 1e3, 3)
