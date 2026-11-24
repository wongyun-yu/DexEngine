"""Penetration depth against the cooked convex parts of an object (`hull.collision_parts`): the reading of
`collider_shape.Parts`, the convex side of every true-shape pair (collider_shape.pair_depth), and directly:
  S-prox         a moving object inside a visual-only static during the render rollout (record.scene.overlap)

A point behind every face plane of a convex part is inside that part; its depth is the distance to the part's nearest
face plane (the exit distance). No surface normals are read, so mesh winding cannot flip a reading, and there is no
sample-spacing floor. A body is represented by the points of its parts' hulls: vertices, edge midpoints and face
centres (`shape`).
"""

from __future__ import annotations

import numpy as np


def hulls(pieces) -> list:
    """Convex part vertices -> [(vertices, face normals, face constants)]; flat parts are jiggled slightly (QJ) to make
    a hull."""
    from scipy.spatial import ConvexHull, QhullError  # noqa: PLC0415

    out = []
    for v in pieces:
        v = np.asarray(v, float)
        try:
            h = ConvexHull(v)
        except QhullError:
            h = ConvexHull(v, qhull_options="QJ")
        out.append((v[h.vertices], h.equations[:, :3], h.equations[:, 3]))
    return out


def _points(v: np.ndarray) -> np.ndarray:
    """Vertices, edge midpoints and face centres of one convex part's hull."""
    from scipy.spatial import ConvexHull, QhullError  # noqa: PLC0415

    try:
        h = ConvexHull(v)
    except QhullError:
        h = ConvexHull(v, qhull_options="QJ")
    tri = h.points[h.simplices]
    mids = np.concatenate([(tri[:, 0] + tri[:, 1]) / 2, (tri[:, 1] + tri[:, 2]) / 2, (tri[:, 0] + tri[:, 2]) / 2])
    return np.unique(np.concatenate([h.points[h.vertices], mids, tri.mean(1)]).round(7), axis=0)


def shape(pieces) -> dict:
    """{hulls (`hulls`), lo, hi (per part boxes), points (N,3)} of a body's convex parts (body frame)."""
    vs = [np.asarray(v, float) for v in pieces]
    return {
        "hulls": hulls(vs),
        "lo": [v.min(0) for v in vs],
        "hi": [v.max(0) for v in vs],
        "points": np.concatenate([_points(v) for v in vs]),
    }


def depths(P: np.ndarray, sh: dict) -> np.ndarray:
    """Per point of P (N,3, the body frame of `sh`): its depth inside `sh`'s convex parts (m; the deepest part; 0 when
    outside every part)."""
    out = np.zeros(len(P))
    for (_v, n, d), lo, hi in zip(sh["hulls"], sh["lo"], sh["hi"], strict=True):
        idx = np.nonzero(np.all((lo < P) & (hi > P), axis=1))[0]
        if not len(idx):
            continue
        s = P[idx] @ n.T + d
        inside = np.all(s < 0, axis=1)
        if inside.any():
            k = idx[inside]
            out[k] = np.maximum(out[k], -s[inside].max(axis=1))
    return out


def signed(P: np.ndarray, sh: dict, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Per point of P (body frame of `sh`): (s, n) over the parts whose box (grown by `margin`) holds it — s = the
    largest face-plane value of the nearest part (< 0 inside: minus the exit distance; ≥ 0 outside: a lower bound of the
    distance to it), n = that face's outward normal. A point near no part: s = inf, n = 0."""
    s_out, n_out = np.full(len(P), np.inf), np.zeros((len(P), 3))
    for (_v, n, d), lo, hi in zip(sh["hulls"], sh["lo"], sh["hi"], strict=True):
        idx = np.nonzero(np.all((lo - margin < P) & (hi + margin > P), axis=1))[0]
        if not len(idx):
            continue
        v = P[idx] @ n.T + d
        j = np.argmax(v, axis=1)
        sv = v[np.arange(len(idx)), j]
        upd = sv < s_out[idx]
        s_out[idx[upd]], n_out[idx[upd]] = sv[upd], n[j[upd]]
    return s_out, n_out


def depth(P: np.ndarray, sh: dict) -> float:
    """Deepest point of P (N,3, the body frame of `sh`) inside one of `sh`'s convex parts (m; 0 when none is inside)."""
    return float(depths(P, sh).max()) if len(P) else 0.0
