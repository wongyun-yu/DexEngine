"""Surface samples and inside tests for the receiving distance of the extra-object rule (`extra_slots_geom.receivers`):
an object arrives at a still extra when it ends within RECV_M + slack of its **surface** (vertex-to-vertex distance
overstates the gap on broad faces — an object put on a wide keyboard: 30 mm between vertices, 10 mm to the surface) or
with its centre inside the extra's **convex hull** (a container) — the measurement the scene oracle
(`readers.raw.scene_classify._receivers`: sampled surfaces, the centroid in the object's frame) makes, so production
(looser by the slack) still contains the oracle. The hull does not change with the object's pose (an axis-aligned box
does: a box round a rotated object holds corners the object does not).
"""

from __future__ import annotations

import os

import numpy as np

#: Surface spacing (m) of the receiving distance: faces whose longest edge exceeds it are split into a barycentric grid
#: of ceil(edge / RECV_SURF_M) steps (no random numbers — the same points on every machine and version), at most
#: RECV_SURF_KMAX steps per face.
RECV_SURF_M = 0.004
RECV_SURF_KMAX = 256
_FACES: dict = {}


def obj_faces(path) -> np.ndarray:
    """Triangles (M, 3) of an OBJ file — its 'f' lines (vertex indices; polygons fanned into triangles), read once per
    path. A mesh without faces dies (its surface cannot be sampled)."""
    key = str(path)
    if key not in _FACES:
        fs = []
        with open(path, encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                if ln.startswith("f "):
                    ix = [int(t.split("/")[0]) for t in ln.split()[1:]]
                    if min(ix) < 1:
                        raise ValueError(f"{path}: relative or zero face index in {ln.strip()!r} — not read")
                    fs += [(ix[0], ix[i], ix[i + 1]) for i in range(1, len(ix) - 1)]
        if not fs:
            raise ValueError(f"mesh has no faces: {path} — the receiving distance cannot be measured on its surface")
        _FACES[key] = np.asarray(fs, np.int64) - 1  # OBJ indices start at 1
    return _FACES[key]


def surface_points(V: np.ndarray, F: np.ndarray, h: float = RECV_SURF_M) -> np.ndarray:
    """Surface samples of vertices V and triangles F: the vertices plus, for every face whose longest edge exceeds h, a
    barycentric grid of ceil(edge / h) steps (boundary included)."""
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e = np.maximum.reduce([np.linalg.norm(B - A, axis=1), np.linalg.norm(C - B, axis=1), np.linalg.norm(A - C, axis=1)])
    k = np.minimum(np.ceil(e / h).astype(int), RECV_SURF_KMAX)
    out = [V]
    for kk in np.unique(k[k > 1]):
        sel = k == kk
        w = np.array([(i, j) for i in range(kk + 1) for j in range(kk + 1 - i)], float) / kk
        a, b, c = A[sel, None], B[sel, None], C[sel, None]
        out.append(
            (a * (1 - w[:, 0] - w[:, 1])[None, :, None] + b * w[None, :, 0, None] + c * w[None, :, 1, None]).reshape(
                -1, 3
            )
        )
    return np.concatenate(out, 0)


def hull_planes(P: np.ndarray) -> np.ndarray | None:
    """Facet planes (k, 4) of the convex hull of points P — outward normal and offset, a point x is inside when every
    plane gives n·x + d < 0. None when P spans less than a volume (rank < 3: a flat or straight object has no inside);
    any other hull failure dies."""
    from scipy.spatial import ConvexHull, QhullError  # noqa: PLC0415

    P = np.asarray(P, np.float64)
    try:
        return ConvexHull(P).equations
    except QhullError:
        if np.linalg.matrix_rank(P - P.mean(0)) < 3:
            return None
        raise


def inside_test(P: np.ndarray):
    """c → whether point c lies strictly inside the convex hull of points P (same frame). The axis-aligned box of P
    holds the hull, so it screens first and the hull is built only when a point passes it (once per test)."""
    lo, hi, memo = P.min(0), P.max(0), {}

    def test(c: np.ndarray) -> bool:
        if not np.all((c > lo) & (c < hi)):
            return False
        if "H" not in memo:
            memo["H"] = hull_planes(P)
        H = memo["H"]
        return H is not None and bool(np.all(H[:, :3] @ c + H[:, 3] < 0))

    return test


#: Largest vertex residual (m) of the rigid fit: coords' placed vertices are a rigid copy of the mesh (measured 0 over
#: 548 fits of six datasets); anything else (a scaled object) would put the surface samples in the wrong place.
RIGID_FIT_TOL_M = 1e-5


def _rigid(Vl: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) with W ≈ Vl @ R.T + t from the same vertices in the body and world frames (the pose convention stays with
    coords' placers; it is not restated here). A residual above RIGID_FIT_TOL_M dies."""
    a, b = Vl.mean(0), W.mean(0)
    U, _s, Vt = np.linalg.svd((Vl - a).T @ (W - b))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = b - a @ R.T
    res = float(np.abs(Vl @ R.T + t - W).max())
    if res > RIGID_FIT_TOL_M:
        raise ValueError(f"placed vertices are not a rigid copy of the mesh (residual {res:.2e} m) — no surface pose")
    return R, t


def near_placer(tree_dir, ti: dict, q: np.ndarray, placed):
    """(object, frame) → world surface samples, the points the receiving distance reads; `near.inside(object, frame)`
    → a test 'is this world point inside the object' (`inside_test`: the convex hull of the mesh vertices in the
    object's own frame; articulated objects: of their placed vertices at that frame). The pose comes from `placed`
    (coords' vertices) as a rigid fit. Articulated objects keep their placed vertices (their part faces are outside
    coords' ownership). Rule tests swap it with the synthetic geometry."""
    from dexengine.placement.coords import core as C  # noqa: PLC0415

    cache: dict = {}

    def _local(o: dict):
        if o["name"] not in cache:
            path = os.path.join(C._object_mesh_dir(tree_dir, ti, o), "visual.obj")
            Vl = C._mesh_vertices_file(path)
            cache[o["name"]] = (Vl, surface_points(Vl, obj_faces(path)), inside_test(Vl))
        return cache[o["name"]]

    def near(o: dict, f: int) -> np.ndarray:
        if o.get("articulation"):
            return placed(o, f)
        Vl, S, _in = _local(o)
        R, t = _rigid(Vl, placed(o, f))
        return S @ R.T + t

    def inside(o: dict, f: int):
        if o.get("articulation"):
            return inside_test(placed(o, f))
        Vl, _S, own = _local(o)
        R, t = _rigid(Vl, placed(o, f))
        return lambda c: own((c - t) @ R)

    near.inside = inside
    return near
