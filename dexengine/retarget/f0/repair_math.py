"""f0 repair, numeric helpers without f0 state: rotation conversions, the hull point sets of a link (decimation,
the shrunk reverse hull, containment) and the hand-root wrist map."""

from __future__ import annotations

import numpy as np


def _rpy_to_R(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return (
        np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    )


def _quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _R_to_quat_wxyz(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
    q = np.empty(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def _aa_to_R(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _slerp(q0, q1, t):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(np.clip(d, -1, 1))
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


def _decimate_vertices(v, max_hull_verts):
    """At most `max_hull_verts` vertices by deterministic voxel vertex-clustering (coarser grids until it fits, then a
    uniform stride), so the kept points still cover the whole surface including concavities."""
    if len(v) > max_hull_verts:
        lo = v.min(0)
        span = np.maximum(v.max(0) - v.min(0), 1e-9)
        idx = np.arange(len(v))
        for grid in (64, 48, 36, 28, 22, 17, 13, 10, 8, 6, 4, 3, 2):
            key = np.floor((v - lo) / span * grid).astype(np.int64)
            _, cand = np.unique(key, axis=0, return_index=True)
            idx = cand
            if len(cand) <= max_hull_verts:
                break
        if len(idx) > max_hull_verts:  # over even on the coarsest grid → uniform stride
            idx = idx[np.linspace(0, len(idx) - 1, max_hull_verts).astype(int)]
        v = v[np.sort(idx)]
    return v


def _reverse_hull(pts, pen_margin):
    """Delaunay of the link points shrunk by `pen_margin` toward their centroid (None for a flat hull)."""
    from scipy.spatial import Delaunay, QhullError

    # SHRUNKEN link hull for the reverse test: contact-depth <= pen_margin is a
    # legitimate grasp touch, so pull every hull vertex pen_margin toward the
    # centroid — membership then means "deeper than pen_margin".
    c = pts.mean(0)
    d = pts - c
    nrm = np.linalg.norm(d, axis=1, keepdims=True).clip(1e-9)
    shrunk = c + d * np.maximum(0.0, 1.0 - pen_margin / nrm)
    try:
        dl = Delaunay(shrunk)  # built ONCE (n_pen runs thousands of times)
    except QhullError:  # a flat (coplanar) hull has no interior to be inside — the declared case
        dl = None
    return dl


def hulls_contain(links_world, query_world):
    """query points inside ANY hand-link hull. Uses the link-LOCAL Delaunay built at
    init and maps the query into link frame — no per-call hull construction."""
    n = 0
    for _, _, Rw, tw, dl in links_world:
        if dl is None:
            continue
        local = (query_world - tw) @ Rw
        n += int(np.count_nonzero(dl.find_simplex(local) >= 0))
    return n


# qpos[0:6] map
#: Max residual (m) between the wrist-cache positions and the hand-root translation qpos mapped through the root's
#: axes. For a matching cache the residual is floating-point noise (measured over all XHand, Tesollo and Wuji datasets,
#: 97,374 sides: max 7.9e-8, none >= 1e-4). Above it, the cache did not come from that qpos.
WRIST_FIT_MAX_RES = 1e-4


def fit_wrist_qpos_map(qpos6_all, cache_pos, A, where: str = ""):
    """The map  cache_pos = A @ qpos[0:3] + b  of a hand root and its wrist cache, validated. A (3x3) = the root's
    translation axes from the scene (repair_root.root_axes — the hand-root convention owner), exact for stacked slide
    joints whatever the clip does; b and the residual come from the rows. A used to be fit from the rows as well
    (least squares with b): a clip whose hand root does not move in three independent directions leaves that fit
    undetermined — the minimum-norm A came out singular or plainly wrong while the residual still passed, and the
    writers' inverse put the root millimetres to 1e15 m away (clips whose wrists stay still within 1e-4 mm). Rotation
    is not fit here (hinge composition is nonlinear): the writers set the rotation columns through
    repair_root.turn_rows. Returns (A, b, max_res).

    A residual >= WRIST_FIT_MAX_RES raises (ValueError; the f0 engine counts it as a failure of that one clip) instead
    of leaving the pad's hand-root translation at stale values, which would ship a tree whose hand root and wrist cache
    disagree as 'repaired'. The usual cause is a stale wrist cache, i.e. cache and qpos from different generations:
    rerun stage 6 (its argv: data/stages.py --emit-run 6)."""
    A = np.asarray(A, np.float64)
    X = np.asarray(qpos6_all, np.float64)[:, :3]
    Y = np.asarray(cache_pos, np.float64)
    b = (Y - X @ A.T).mean(axis=0)
    res = float(np.abs(X @ A.T + b - Y).max())
    if not res < WRIST_FIT_MAX_RES:
        raise ValueError(
            f"[F0_WRIST_FIT] ✗ {where}: wrist cache ↔ hand-root qpos through the root's axes, residual {res:.2e} m ≥ "
            f"{WRIST_FIT_MAX_RES:g} — the cache did not come from this qpos (stale cache; rerun stage 6)"
        )
    return A, b, res
