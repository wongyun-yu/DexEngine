"""Point-triangle geometry of the triangle ruler (collider_shape.Triangles): the closest point on a triangle, the
solid angle a triangle set subtends, the generalised winding number of a mesh (`Winding`), the free-ball regions
that share one winding sum (`regions`) and the exact distance to a triangle set. Part of the ruler's identity
(collider_shape.RULER_MODULES, read by sdf_grid.ruler_id)."""

from __future__ import annotations

import numpy as np

LINK_K = 8  # regions: free balls are tested for overlap with this many nearest query points (a speed setting)


def closest_point(P: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """The closest point of each P[i] on triangle (A[i], B[i], C[i]) (Ericson 5.1.5)."""
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = P - B
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = P - C
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    v, w = vb / den, vc / den
    out = A + ab * v[..., None] + ac * w[..., None]  # interior
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ab = np.clip(d1 / (d1 - d3), 0.0, 1.0)
        t_ac = np.clip(d2 / (d2 - d6), 0.0, 1.0)
        t_bc = np.clip((d4 - d3) / ((d4 - d3) + (d5 - d6)), 0.0, 1.0)
    e_ab, e_ac = (vc <= 0) & (d1 >= 0) & (d3 <= 0), (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    e_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    v_c, v_b, v_a = (d6 >= 0) & (d5 <= d6), (d3 >= 0) & (d4 <= d3), (d1 <= 0) & (d2 <= 0)
    out = np.where(e_bc[..., None], B + (C - B) * np.nan_to_num(t_bc)[..., None], out)
    out = np.where(e_ac[..., None], A + ac * np.nan_to_num(t_ac)[..., None], out)
    out = np.where(v_c[..., None], C, out)
    out = np.where(e_ab[..., None], A + ab * np.nan_to_num(t_ab)[..., None], out)
    out = np.where(v_b[..., None], B, out)
    return np.where(v_a[..., None], A, out)


def solid(T: np.ndarray, P: np.ndarray, block: int = 128) -> np.ndarray:
    """Σ over triangles T (m,3,3) of the solid angle each subtends at P[i], over 4π (Van Oosterom & Strackee 1983)."""
    out = np.zeros(len(P))
    if not len(T) or not len(P):
        return out
    A, B, C = T[:, 0], T[:, 1], T[:, 2]
    for s in range(0, len(P), block):
        q = P[s : s + block, None, :]
        a, b, c = A[None] - q, B[None] - q, C[None] - q
        la, lb, lc = (np.sqrt((x * x).sum(-1)) for x in (a, b, c))
        det = (a * np.cross(b, c)).sum(-1)
        den = la * lb * lc + (a * b).sum(-1) * lc + (b * c).sum(-1) * la + (c * a).sum(-1) * lb
        out[s : s + block] = np.arctan2(det, den).sum(-1) / (2 * np.pi)
    return out


def connected(i: np.ndarray, j: np.ndarray, n: int) -> np.ndarray:
    """Connected-part label of each of n nodes joined by the edges (i, j)."""
    from scipy.sparse import coo_matrix  # noqa: PLC0415
    from scipy.sparse.csgraph import connected_components  # noqa: PLC0415

    return connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)), directed=False)[1]


def regions(P: np.ndarray, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(label of each point, index of one point per label): points joined where their free balls overlap
    (|p − q| < rho_p + rho_q), tested between each point and its LINK_K nearest points."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    n, k = len(P), min(LINK_K, len(P))
    dd, jj = (x.reshape(n, k) for x in cKDTree(P).query(P, k=k))
    link = dd < rho[:, None] + rho[jj]
    lab = connected(np.broadcast_to(np.arange(n)[:, None], (n, k))[link], jj[link], n)
    return lab, np.unique(lab, return_index=True)[1]


def distance(P: np.ndarray, T: np.ndarray, block: int = 1 << 20) -> np.ndarray:
    """Distance of each point to the nearest of triangles T (m,3,3), exact (`closest_point` on every pair)."""
    out, step = np.empty(len(P)), max(1, block // max(len(T), 1))
    for s in range(0, len(P), step):
        q = P[s : s + step, None]
        D = q - closest_point(q, T[None, :, 0], T[None, :, 1], T[None, :, 2])
        out[s : s + step] = np.linalg.norm(D, axis=-1).min(axis=1)
    return out


class Winding:
    """Generalised winding number of a triangle set at query points, exact and hierarchical (Jacobson et al. 2013,
    section 3.3): a box tree over the faces; a cluster whose box does not hold the point contributes minus its cap
    (a fan over the cluster's boundary edges — cluster ∪ cap is closed and lies in the box, so the two sum to 0
    there), a cluster that holds it is opened, a leaf is summed face by face. Vertices are welded by position for the
    boundary edges only (a mesh with split vertices is one surface), the sums use the triangles as given. `closure`
    closes the whole surface: over each connected part of its boundary, a fan from the part's centroid (None when the
    surface has no boundary), so surface ∪ closure has none."""

    LEAF = 48  # faces per leaf (a speed setting: the result does not depend on it)

    def __init__(self, V: np.ndarray, F: np.ndarray):
        self.T = V[F]
        _u, weld = np.unique(V, axis=0, return_inverse=True)
        self.Fw, self.Vw = weld.reshape(-1)[F], _u
        self.nodes: list = []
        self._build(np.arange(len(F)), self.T.mean(axis=1))
        self.closure = self._closure()

    def _boundary(self, fi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Directed boundary edges (i → j, welded ids, with multiplicity) of faces fi: directions that do not cancel."""
        F = self.Fw[fi]
        e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        n = len(self.Vw) + 1
        keys = np.concatenate([e[:, 0] * n + e[:, 1], e[:, 1] * n + e[:, 0]])
        u, inv = np.unique(keys, return_inverse=True)
        net = np.bincount(inv, weights=np.repeat([1.0, -1.0], len(e)), minlength=len(u)).round().astype(int)
        bd = np.repeat(u[net > 0], net[net > 0])
        return bd // n, bd % n

    def _cap(self, fi: np.ndarray):
        i, j = self._boundary(fi)
        keep = (i != i[0]) & (j != i[0]) if len(i) else np.zeros(0, bool)  # a fan from i[0]; its own edges are empty
        return self.Vw[np.stack([np.full(int(keep.sum()), i[0]), j[keep], i[keep]], axis=1)] if keep.any() else None

    def _closure(self):
        i, j = self._boundary(np.arange(len(self.Fw)))
        if not len(i):
            return None
        part = np.unique(connected(i, j, len(self.Vw))[i], return_inverse=True)[1]
        apex = np.stack([np.bincount(part, weights=self.Vw[i, a]) for a in range(3)], 1) / np.bincount(part)[:, None]
        return np.stack([apex[part], self.Vw[j], self.Vw[i]], axis=1)

    def _build(self, fi: np.ndarray, cen: np.ndarray) -> int:
        P = self.T[fi].reshape(-1, 3)
        k = len(self.nodes)
        self.nodes.append(None)
        if len(fi) <= self.LEAF:
            self.nodes[k] = (P.min(0), P.max(0), self._cap(fi), self.T[fi], -1, -1)
            return k
        lo, hi = P.min(0), P.max(0)
        order = fi[np.argsort(cen[fi, int(np.argmax(hi - lo))], kind="stable")]
        m = len(order) // 2
        left, right = self._build(order[:m], cen), self._build(order[m:], cen)
        self.nodes[k] = (lo, hi, self._cap(fi), None, left, right)
        return k

    def __call__(self, P: np.ndarray) -> np.ndarray:
        w = np.zeros(len(P))
        stack = [(0, np.arange(len(P)))]
        while stack:
            k, idx = stack.pop()
            lo, hi, cap, tris, left, right = self.nodes[k]
            inb = np.all((P[idx] >= lo - 1e-6) & (P[idx] <= hi + 1e-6), axis=1)
            out, idx = idx[~inb], idx[inb]
            if len(out) and cap is not None:
                w[out] -= solid(cap, P[out])
            if not len(idx):
                continue
            if tris is not None:
                w[idx] += solid(tris, P[idx])
            else:
                stack += [(left, idx), (right, idx)]
        return w
