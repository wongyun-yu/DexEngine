"""The pinned stills the f0 object repair measures (rule repair_obj, THE BODIES): a still extra whose training preset
keeps the hand colliding stands at its frame-0 pose for the whole clip, in the shape physics collides it with
(collider_shape.row_shape — the visual triangles for the still kinds, S-true-shape).

WHY. A recorded hand that starts inside a pinned still is pushed out by PhysX at the first step (a triangle A/B: 31 of
86 start rows took more than one step, one never cleared). Measured on the delivered wuji2 set at frame 0 (f0 strict
hand, triangle ruler): 150 of the 158 census sides deeper than 1 mm. The f0 repair already owns "the hand at frame 0,
out of what it collides with" (fingers cheap, wrist dear, desk kept) and its spawn-hold window; a still joins it as one
more body, over the same window. Frames after the window are not repaired: the still keeps the hand out in training
(robot ↔ still collision on), and S-hand-clear measures them.

TWO READINGS OF ONE SHAPE.
  measure   (gate, acceptance, report) the exact true-shape depth of the strict hand's collision pieces against the
            still: collider_shape.pair_depth (both ways) per piece, the piece's convex hull in its link frame, max over
            pieces. A body on the convex ruler (a dataset with S-true-shape off) is measured as cooked convex pieces by
            the convex path instead (repair_bodies: class still, the SAT).
  solve     a local signed-distance grid of the triangles at the still's pinned pose (`Grid`): GRID_STEP_M nodes over
            the box the hand's collision hull vertices cover during the window (data poses), widened by GRID_MARGIN_M
            and cut to the still's box, each node the exact signed distance (Triangles.signed). The projection reads
            it by trilinear interpolation: per hand piece, its deepest hull vertex inside (depth above the target)
            leaves along the grid gradient, and its deepest still surface sample inside the piece leaves along the
            piece's face normal. The grid only steers the solve; a result is accepted by the exact measure alone.
Thresholds: repair_obj.THRESHOLDS_MM["still"] (the slot's values: a pinned body does not move either).
Marker `[F0_STILL]`, one line per side with stills.
"""

from __future__ import annotations

import time

import numpy as np

from dexengine.data.objects import collider_shape as CS
from dexengine.retarget.f0 import repair_obj as RO

MARK = "[F0_STILL]"
GRID_STEP_M = 0.002  # node spacing of the solve grid (m)
GRID_MARGIN_M = 0.010  # the grid box: the hand's window box widened by this, cut to the still's box (m)
POINT_CHUNK = 32768  # nodes per signed-distance call (a memory setting: the values do not depend on it)
GRID_BRICK = 8  # the grid fills its nodes in bricks of GRID_BRICK^3 when a read first reaches one (`Grid.at`)
_CORNERS = np.array(list(np.ndindex(2, 2, 2)))  # a cell's 8 corner offsets (node index units)
#: the wall term's gate (`rows`): per hand piece, its hull planes that face these 26 directions (axes, edges, corners)
#: most — a polytope holding the piece. A still sample deeper than the target in the piece is deeper than the target in
#: that polytope too, so only those samples meet the full plane test (a piece holds ~300-4,500 hull planes): the same
#: rows, at a fraction of the plane evaluations.
GATE_DIRS = np.array([d for d in np.ndindex(3, 3, 3) if d != (1, 1, 1)], float) - 1.0
GATE_SLACK_M = 1e-9  # the gate's rounding allowance: its plane values (matmul) and the full test's differ by ulps (m)
#: `deepest`'s screening allowance (m): matmul plane values of link-frame points (a 3-term dot product plus an offset,
#: magnitudes within metres) differ from plane_values by ~1e-16 m; a point within this of the screen's least max is
#: read exactly
PLANE_TIE_M = 1e-12
#: `deepest` reads a batch of at most this many point-plane values exactly at once (measured: screening pays from 8k on;
#: both ways give the same pick)
SCREEN_MIN_READS = 8192
_PIECES: dict = {}


class StillBody:
    """One pinned still: name, kind, ruler (collider_shape.Triangles), world pose M (4x4, its frame-0 pose) and the
    solve grid of each side (`grids`, built by `build_grid`)."""

    def __init__(self, name: str, kind: str, ruler, M: np.ndarray):
        self.name, self.kind, self.ruler = name, kind, ruler
        self.M = np.asarray(M, float)
        self.Minv = np.linalg.inv(self.M)
        self.digest = ruler.digest
        self.grids: dict = {}  # side -> Grid | None (`build_grid`)

    def build_grid(self, side: str, hull_world: list) -> None:
        """The side's solve grid over the box of `hull_world` (world points of the hand's hull vertices, every window
        frame) in the still's frame, widened by GRID_MARGIN_M and cut to the still's box; None when they never reach
        it (no still row then: the exact measure still accepts or refuses)."""
        P = np.vstack(hull_world) @ self.Minv[:3, :3].T + self.Minv[:3, 3]
        lo = np.maximum(P.min(0) - GRID_MARGIN_M, self.ruler.lo - GRID_STEP_M)
        hi = np.minimum(P.max(0) + GRID_MARGIN_M, self.ruler.hi + GRID_STEP_M)
        self.grids[side] = Grid(self.ruler, lo, hi) if np.all(hi > lo) else None


class Grid:
    """Signed distance (m, < 0 inside) of a triangle ruler at GRID_STEP_M nodes over [lo, hi] (its frame); `at(P)` =
    (trilinear value, its gradient) — a point outside the grid reads +inf (outside) and a zero gradient. A node's value
    is computed the first time a read reaches its brick (GRID_BRICK^3 nodes): a solve reads the few bricks the hand
    passes through, and a node holds the value it would hold had the whole grid been filled at once (Triangles.signed
    reads each point on its own: the values do not depend on which points share a call)."""

    def __init__(self, ruler, lo: np.ndarray, hi: np.ndarray):
        t0 = time.time()
        self.ruler = ruler
        self.lo = np.asarray(lo, float)
        self.n = np.maximum(2, np.ceil((np.asarray(hi, float) - self.lo) / GRID_STEP_M).astype(int) + 1)
        self.ax = [self.lo[i] + GRID_STEP_M * np.arange(self.n[i]) for i in range(3)]
        self.S = np.empty(tuple(self.n))
        self.filled = np.zeros(tuple(-(-self.n // GRID_BRICK)), bool)
        self.nodes, self.build_s = int(np.prod(self.n)), time.time() - t0

    def _fill(self, i: np.ndarray) -> None:
        """Fill the bricks holding the cells at node indices i (each cell's 8 corner nodes); a read whose bricks are all
        filled (most reads of a solve) only looks them up."""
        b = np.concatenate([(i + c) // GRID_BRICK for c in _CORNERS])
        b = b[~self.filled[b[:, 0], b[:, 1], b[:, 2]]]
        if not len(b):
            return
        b = np.unique(b, axis=0)
        boxes, pts = [], []
        for bx in b:
            sl = tuple(slice(k * GRID_BRICK, min((k + 1) * GRID_BRICK, n)) for k, n in zip(bx, self.n, strict=True))
            boxes.append(sl)
            pts.append(np.stack(np.meshgrid(*(a[s] for a, s in zip(self.ax, sl, strict=True)), indexing="ij"), -1))
        P = np.concatenate([p.reshape(-1, 3) for p in pts])
        S = np.empty(len(P))
        for c0 in range(0, len(P), POINT_CHUNK):
            S[c0 : c0 + POINT_CHUNK] = self.ruler.signed(P[c0 : c0 + POINT_CHUNK])
        c0 = 0
        for sl, p in zip(boxes, pts, strict=True):
            m = p.shape[0] * p.shape[1] * p.shape[2]
            self.S[sl] = S[c0 : c0 + m].reshape(p.shape[:3])
            c0 += m
        self.filled[b[:, 0], b[:, 1], b[:, 2]] = True

    def at(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = (np.asarray(P, float) - self.lo) / GRID_STEP_M
        ok = np.all((u >= 0) & (u <= self.n - 1), axis=1)
        val, grad = np.full(len(P), np.inf), np.zeros((len(P), 3))
        if not ok.any():
            return val, grad
        u = u[ok]
        i = np.minimum(np.floor(u).astype(int), self.n - 2)
        self._fill(i)
        f = u - i
        c = {(a, b, d): self.S[i[:, 0] + a, i[:, 1] + b, i[:, 2] + d] for a in (0, 1) for b in (0, 1) for d in (0, 1)}
        wx, wy, wz = (np.stack([1 - f[:, k], f[:, k]], 1) for k in range(3))
        dx, dy, dz = (np.array([-1.0, 1.0]) for _k in range(3))
        v = np.zeros(len(u))
        g = np.zeros((len(u), 3))
        for (a, b, d), s in c.items():
            v += wx[:, a] * wy[:, b] * wz[:, d] * s
            g[:, 0] += dx[a] * wy[:, b] * wz[:, d] * s
            g[:, 1] += wx[:, a] * dy[b] * wz[:, d] * s
            g[:, 2] += wx[:, a] * wy[:, b] * dz[d] * s
        val[ok], grad[ok] = v, g / GRID_STEP_M
        return val, grad


def hand_pieces(strict) -> list:
    """[(link, collider_shape.Parts of the piece's hull vertices in its link frame, hull vertices, hull planes)] of the
    strict hand (once per model)."""
    key = id(strict)
    if key not in _PIECES:
        from scipy.spatial import ConvexHull  # noqa: PLC0415

        out = []
        for ln, ps in strict.parts.items():
            for p in ps:
                p = np.asarray(p, float)
                try:
                    h = ConvexHull(p)
                except Exception:  # noqa: BLE001 — qhull refuses flat sets: no volume, no overlap
                    continue
                out.append((ln, CS.Parts([p[h.vertices]]), p[h.vertices], h.equations))
        _PIECES[key] = (strict, out)
    return _PIECES[key][1]


def depth(still: StillBody, strict, LT: dict) -> tuple[float, str]:
    """(exact depth m, deepest link) of the strict hand (link world transforms LT {link: (R, t)}) against the still:
    collider_shape.pair_depth per piece, a bounding-sphere test first."""
    best, bl = 0.0, ""
    for ln, prt, _hv, _eq in hand_pieces(strict):
        if ln not in LT:
            continue
        R, t = LT[ln]
        G = np.eye(4)
        G[:3, :3], G[:3, 3] = R, t
        Mb = still.Minv @ G
        if np.linalg.norm(Mb[:3, :3] @ prt.center + Mb[:3, 3] - still.ruler.center) > prt.radius + still.ruler.radius:
            continue
        d = CS.pair_depth(still.ruler, prt, Mb)
        if d > best:
            best, bl = d, ln
    return best, bl


def window_hull(strict, poses) -> list:
    """World points of the strict hand's hull vertices at each (wrist position, wrist R, independent fingers) pose."""
    out = []
    for wp, wR, indep in poses:
        LT = {ln: (R, t) for ln, _p, R, t, _d in strict.link_points_world(wp, wR, strict.fingers12(indep))}
        out += [hv @ LT[ln][0].T + LT[ln][1] for ln, _prt, hv, _eq in hand_pieces(strict) if ln in LT]
    return out


def plane_values(x: np.ndarray, e: np.ndarray) -> np.ndarray:
    """[n, m] values x·normal + offset of points x [n, 3] at planes e [m, 4] (<= 0 on the inner side), each element
    from its own point and plane only, so a point reads the same values whichever points share the call. `x @ e.T`
    does not: BLAS picks its kernel by the batch shape, a row's rounding follows the row count, and a filter in front
    of it (the wall gate) moved residuals by ulps that the solve grew into other repairs (7 of 358 paired trees)."""
    return x[:, :1] * e[:, 0] + x[:, 1:2] * e[:, 1] + x[:, 2:3] * e[:, 2] + e[:, 3]


def _plane_values_fast(x: np.ndarray, e: np.ndarray) -> np.ndarray:
    """plane_values by matmul: the same numbers up to rounding that follows the batch shape (`deepest`'s screen)."""
    return x @ e[:, :3].T + e[:, 3]


def deepest(x: np.ndarray, e: np.ndarray) -> tuple[int, np.ndarray]:
    """(index of the deepest point of x [n, 3] inside planes e — the least max plane value, the first on ties —, its
    plane values [m]), exactly as plane_values(x, e) over the whole batch gives them: matmul screens the batch, and only
    the points within PLANE_TIE_M of its least max are read by plane_values (the exact deepest point and every point
    tied with it are among them: matmul and plane_values differ by far less than PLANE_TIE_M / 2). A batch of at most
    SCREEN_MIN_READS values is read by plane_values at once (there the screen costs more than it saves)."""
    if len(x) * len(e) <= SCREEN_MIN_READS:
        v = plane_values(x, e)
        k = int(np.argmin(v.max(1)))
        return k, v[k]
    m = _plane_values_fast(x, e).max(1)
    c = np.flatnonzero(m <= m.min() + PLANE_TIE_M)
    v = plane_values(x[c], e)
    k = int(np.argmin(v.max(1)))
    return int(c[k]), v[k]


def gate_planes(e: np.ndarray) -> np.ndarray:
    """The rows of `e` (a piece's hull planes) that face GATE_DIRS most, in their order in `e` (`rows`' gate)."""
    return e[np.unique(np.argmax(e[:, :3] @ GATE_DIRS.T, axis=0))]


def rows(still: StillBody, g, pieces: list, gates: list, T: list, target_m: float, W=None) -> list:
    """The projection's residual rows of one still at link transforms T (Projection.link_T): per hand piece (link index,
    hull vertices (link frame), planes (link frame)) the deepest hull vertex inside the grid's surface by more than
    target_m (leaves along the gradient) and the deepest still surface sample inside the piece by more than target_m
    (a wall thinner than the piece, an open sheet: leaves along the piece's face normal) — the two readings of
    collider_shape.pair_depth. gates[i] = gate_planes of pieces[i]; W[i] = pieces[i]'s hull vertices in the world
    (Projection.world; None: computed here). Row = (residual, link index, link-local point, world direction that
    increases the residual)."""
    tri = still.ruler
    if g is None:
        return []
    Rs, ts = still.M[:3, :3], still.M[:3, 3]
    locs, keep = [], []
    for i, (li, v, _e) in enumerate(pieces):
        R, t = T[li]
        loc = ((v @ R.T + t if W is None else W[i]) - ts) @ Rs  # hull vertices in the still frame
        if np.all(loc.max(0) >= tri.lo) and np.all(loc.min(0) <= tri.hi):  # else neither term reads anything
            locs.append(loc)
            keep.append(i)
    if not keep:
        return []
    cut = np.cumsum([0] + [len(x) for x in locs])
    s, grad = g.at(np.concatenate(locs))
    out = []
    for n, i in enumerate(keep):
        li, v, e = pieces[i]
        sl = slice(cut[n], cut[n + 1])
        k = int(np.argmin(s[sl]))
        if -s[sl][k] > target_m:
            d = grad[sl][k] / max(np.linalg.norm(grad[sl][k]), 1e-12)
            out.append((-s[sl][k] - target_m, li, v[k], -(Rs @ d)))
        R, t = T[li]
        lo, hi = locs[n].min(0), locs[n].max(0)
        near = tri.near((lo + hi) / 2, float(np.linalg.norm(hi - lo)) / 2)
        if not len(near):
            continue
        nl = (near @ Rs.T + ts - t) @ R  # the samples in the link frame
        nl = nl[np.all((nl >= v.min(0)) & (nl <= v.max(0)), axis=1)]  # in the piece's box
        if not len(nl):
            continue
        ge = gates[i]
        nl = nl[(nl @ ge[:, :3].T + ge[:, 3]).max(1) < GATE_SLACK_M - target_m]  # deeper than target in the gate's
        if not len(nl):  # polytope (a sample left out is shallower than target in the piece: it sets no row)
            continue
        j, vj = deepest(nl, e)  # ≤ 0 inside every plane; the sample and its values do not follow the gate's cut
        a = int(np.argmax(vj))
        if -vj[a] > target_m:  # the deepest sample leaves through its exit plane (the least deep face)
            out.append((-vj[a] - target_m, li, nl[j], e[a, :3] @ R.T))
    return out


def with_stills(sat, stills: list, strict, pose) -> RO.Sat:
    """`sat` (repair_obj.Sat of the convex bodies) with one row per still (class still, `depth`) at the hand pose
    (wrist position, wrist R, finger joints); `sat` itself without stills."""
    if not stills:
        return sat
    LT = {ln: (R, t) for ln, _p, R, t, _d in strict.link_points_world(*pose)}
    rows = [(b.name, "still", *depth(b, strict, LT)) for b in stills]
    return RO.Sat([("main", "main", sat[0], sat[1]), *sat.others, *rows])


def side_grids(seq: str, side: str, strict, poses: list, stills: list, rep: dict) -> None:
    """The solve grid of each still over the box the side's hand covers at `poses` (the window's data poses: wrist
    position, wrist R, independent fingers); its sizes and build time go to the report and one MARK line (nothing
    without stills)."""
    if not stills:
        return
    t = time.time()
    hull = window_hull(strict, poses)
    for b in stills:
        if side not in b.grids:
            b.build_grid(side, hull)
    nodes = {b.name: (b.grids[side].nodes if b.grids[side] is not None else 0) for b in stills}
    rep["obj_still_grid_nodes"], rep["obj_still_grid_s"] = nodes, round(time.time() - t, 3)
    print(f"{MARK} {seq} [{side}] grids {nodes} step {GRID_STEP_M} built {time.time() - t:.2f}s", flush=True)
