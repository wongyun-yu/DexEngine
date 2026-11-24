"""Rule S-true-shape: which shape each spawned body collides with (`shape_of`, the one owner) and the ruler every
reader measures that shape with (`body`).

Why: a pinned still body stood as its cooked convex parts, and convex parts fill a rack's holes, a bowl's cavity and
a lamp's socket. An object recorded resting inside such a body started deep in its parts (receiver drop rows: median
21.8 mm on the parts against 3.65 mm on the visual mesh, 447 rows). PhysX collides a kinematic body's triangle mesh
(mesh approximation none) as authored, so a pinned still body stands as its visual mesh, the surface the recording was
captured against.

Shape (`shape_of`): TRIANGLES for a pinned body whose kind is in TRIANGLE_KINDS, on a dataset where the rule is on.
PARTS (the cooked convex parts, `hull.collision_parts`) for every other body: moving kinematic slots, dynamic main
objects (a dynamic PhysX body cannot carry a triangle mesh) and the datasets that opt out (`Profile.scene_rules_off`).

Ruler (`body`): the shape physics uses.
  TRIANGLES  the visual mesh of the training USD (`visual_mesh`: the one mesh under the rigid body's VISUALS child,
             which is the prim the env authors as the collider), with surface samples at most SAMPLE_M
             apart (every vertex, face centre and a barycentric grid on the larger faces)
  PARTS      `hull.collision_parts` (convex_depth shapes)
Depth of points into a body:
  PARTS      exit distance inside the deepest convex part (`convex_depth.depths`)
  TRIANGLES  distance to the nearest face, for a point inside the mesh; 0 outside it and outside the mesh's box. The
             nearest face is searched among the faces of the K_NEAR nearest samples, exact point-triangle distance.
             Inside = the generalised winding number of the triangles at the point is above 1/2 in magnitude
             (Jacobson et al. 2013): the solid angle the surface subtends, summed with orientation; it reads open
             meshes too, and the magnitude reads an inside-out mesh as one wound outward (a hollow shell's cavity
             stays at 0). One rule on every mesh, exact (`Triangles.inside`): surface ∪ closure (a fan over each part
             of the boundary) has an integer winding number, constant off it, so points whose balls free of both
             overlap share it and one sum per region serves them (`triangle_geom.regions`). No nearest-feature
             normal decides inside: one face's normal fails at edges and vertices (a 35 mm error on a
             container rim), the angle-weighted pseudo-normal where shells cross (a point inside one of two crossing
             shells read outside).
A point past a thin wall reads at most half the wall. A pair is therefore read both ways (`pair_depth`); the triangle
samples inside a convex partner read its exit distance, at most SAMPLE_M low (the sample spacing).
Threshold: `scene_requirement.TRUE_SHAPE_OVERLAP_M`. Marker `[TRI_SHAPE]`, once per triangle body read.
"""

from __future__ import annotations

import hashlib

import numpy as np

from dexengine.data.objects import convex_depth as CD
from dexengine.data.objects import triangle_geom as TGM

RULE_ID = "S-true-shape"
TRIANGLES, PARTS = "triangles", "parts"
SHAPES = (TRIANGLES, PARTS)
#: Role of a support object (extra_slots.SPAWNED_ROLES): a support is a kinematic body pinned at its frame-0 pose.
SUPPORT = "support"
#: Kinds whose pinned bodies stand as triangles: the still extras (extra_slots kinds) and the supports.
TRIANGLE_KINDS = ("receiver", "touched", "touched_solid", "envelope", SUPPORT)
VISUALS = "visuals"  # the rigid body's child holding its visual mesh (robots/objects/usd_build.py)
SAMPLE_M = 0.002  # largest spacing of the triangle surface samples (m)
K_NEAR = 8  # nearest samples whose faces are searched for a point's nearest face
CLOSURE_MAX = 512  # inside: a closure with more faces sums every point (a speed setting: the sign does not change)
REACH_EPS_M = 1e-9  # `_reach`: margin for the rounding of distances (m; rounding is ~1e-16 m at metre scale)
#: `depths`: the nearest-sample search stops at this radius (a speed setting: the values do not depend on it). A point
#: farther from every sample reads the radius as its free-ball bound; an unbounded search from a point deep inside a
#: large hollow mesh visits most of the samples (they all lie near the same distance).
DEPTH_QUERY_M = 0.01
SIGN_RULE = "winding"  # the inside rule, named in the [TRI_SHAPE] line (module doc)
MARK = "[TRI_SHAPE]"
#: The modules the ruler is made of, in the order its identity hashes them (sdf_grid.ruler_id): this module and
#: every dexengine module it imports at module level (the triangle geometry, the convex-part depth).
RULER_MODULES = ("collider_shape", "triangle_geom", "convex_depth")
OVERLAP_DROP_WORD = "true_shape_overlap"  # exclusion reason word of a clip dropped by `overlap_drop`
_CACHE: dict = {}


def on(ds: str) -> bool:
    """Is S-true-shape on for dataset `ds` (scene rules registry, default on)?"""
    from dexengine.data.objects import scene_rules as SC  # noqa: PLC0415 — scene_rules imports the rule modules

    return SC.on(RULE_ID, ds)


def shape_of(ds: str, kind: str | None, pinned: bool) -> str:
    """The collider shape of a spawned body: TRIANGLES for a pinned body of a TRIANGLE_KINDS kind on a dataset with
    the rule on, PARTS otherwise (module doc)."""
    return TRIANGLES if pinned and kind in TRIANGLE_KINDS and on(ds) else PARTS


def row_shape(ds: str, row: dict) -> str:
    """`shape_of` of an extra_slots row (its kind and its `static` statement: pinned at frame 0)."""
    return shape_of(ds, row.get("kind"), bool(row.get("static")))


def require_still_triangles(ds: str, row: dict) -> None:
    """The one check of the still readers (hand_clear's ruler): a still target row stands as its
    triangles. A row that reads any other shape (a dataset with S-true-shape off, a kind outside TRIANGLE_KINDS) raises:
    those readers measure triangles only."""
    shape = row_shape(ds, row)
    if shape != TRIANGLES:
        raise ValueError(
            f"{MARK} {ds}/{row.get('name')} (kind {row.get('kind')!r}, static {bool(row.get('static'))}): a still row"
            f" reads {shape!r}; the still readers measure {TRIANGLES!r} only (S-true-shape on: {on(ds)})"
        )


def statement_fp(ds: str, slots: list) -> str:
    """Fingerprint of one clip's collider statement: (name, kind, static, obj_id, `row_shape`) of each slot row of
    extra_slots.slot_rows, sorted, with RULE_ID. The env (train, judge and render build the slots the same way) prints
    it in `[TRI_COLL] statement` and stage 2f writes it in its ledger, so a scene and the measure it was judged with
    can be compared clip by clip."""
    items = sorted(
        [str(r["name"]), str(r.get("kind")), bool(r.get("static")), str(r.get("obj_id")), row_shape(ds, r)]
        for r in slots
    )
    return "ts-" + hashlib.sha256(repr((RULE_ID, items)).encode()).hexdigest()[:12]


class Parts:
    """Cooked convex parts (body frame): `points` (hull vertices, edge midpoints, face centres), `depths(P)`."""

    kind = PARTS

    def __init__(self, pieces: list, digest: str = ""):
        self.pieces = [np.asarray(v, float) for v in pieces]
        self.sh = CD.shape(self.pieces)
        self.points = self.sh["points"]
        self.digest = digest
        self.center = self.points.mean(axis=0)
        self.radius = float(np.linalg.norm(self.points - self.center, axis=1).max())

    def near(self, c: np.ndarray, r: float) -> np.ndarray:
        """The points within r of c (all of them: a convex body has few)."""
        return self.points

    def depths(self, P: np.ndarray) -> np.ndarray:
        return CD.depths(np.asarray(P, float), self.sh)

    def depth(self, P: np.ndarray) -> float:
        return float(self.depths(P).max()) if len(P) else 0.0


def _samples(V: np.ndarray, F: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray, float]:
    """(surface samples (S,3), the face of each, cover): every vertex, every face centre, and a barycentric grid with
    spacing <= step on every face with an edge longer than step. Every surface point lies within `cover` of a sample:
    the grid cuts a face into triangles with edges at most s, whose points lie within s/√3 of a corner."""
    T = V[F]
    edge = np.linalg.norm(T[:, [1, 2, 0]] - T, axis=2).max(axis=1)
    n = np.maximum(1, np.ceil(edge / step).astype(int))
    first = np.full(len(V), -1)
    first[F[:, 0]], first[F[:, 1]], first[F[:, 2]] = np.arange(len(F)), np.arange(len(F)), np.arange(len(F))
    used = first >= 0
    pts, face = [V[used], T.mean(axis=1)], [first[used], np.arange(len(F))]
    for k in np.unique(n[n > 1]):
        idx = np.flatnonzero(n == k)
        B = np.array([(i / k, j / k, 1.0 - (i + j) / k) for i in range(k + 1) for j in range(k + 1 - i)])
        pts.append(np.einsum("bk,mkd->mbd", B, T[idx]).reshape(-1, 3))
        face.append(np.repeat(idx, len(B)))
    cover = float((edge / n).max()) / np.sqrt(3.0) if len(F) else 0.0
    return np.concatenate(pts), np.concatenate(face), cover


class Triangles:
    """A triangle mesh (body frame): `points` (surface samples, SAMPLE_M), `depths(P)` (module doc)."""

    kind = TRIANGLES

    def __init__(self, V: np.ndarray, F: np.ndarray, step: float = SAMPLE_M):
        from scipy.spatial import cKDTree  # noqa: PLC0415

        self.V, self.F = np.asarray(V, float), np.asarray(F, np.int64).reshape(-1, 3)
        T = self.V[self.F]
        nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        ln = np.linalg.norm(nrm, axis=1)
        self.normals = nrm / np.where(ln > 0, ln, 1.0)[:, None]
        self.points, self.sample_face, self.cover = _samples(self.V, self.F, step)
        if not (np.isfinite(self.cover) and self.cover >= 0):  # depths / within rely on it (`_reach`)
            raise ValueError(f"{MARK} sample cover {self.cover!r}: want a finite value >= 0 (`_samples`)")
        self.tree = cKDTree(self.points)
        self.lo, self.hi = self.V.min(axis=0), self.V.max(axis=0)
        self.center = (self.lo + self.hi) / 2
        self.radius = float(np.linalg.norm(self.hi - self.lo) / 2)
        h = hashlib.sha256(self.V.tobytes() + self.F.tobytes())
        self.digest = h.hexdigest()[:16]
        self._wn = None
        self.sign_rule = SIGN_RULE

    def near(self, c: np.ndarray, r: float) -> np.ndarray:
        """The surface samples within r of c."""
        return self.points[self.tree.query_ball_point(np.asarray(c, float), r)]

    def _winding(self) -> TGM.Winding:
        if self._wn is None:
            self._wn = TGM.Winding(self.V, self.F)
        return self._wn

    def winding(self, P: np.ndarray) -> np.ndarray:
        """Generalised winding number at each point, summed (≈ 1 inside, ≈ 0 outside; `triangle_geom.Winding`, built
        once)."""
        return self._winding()(np.asarray(P, float))

    def _nearest(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(distance to the nearest face among the faces of the K_NEAR nearest samples, to the nearest sample); a face
        several of a point's samples lie on is measured once (it gives that point one distance)."""
        k = min(K_NEAR, len(self.points))
        dk, idx = (x.reshape(len(P), k) for x in self.tree.query(P, k=k))
        fs = np.sort(self.sample_face[idx], axis=1)
        r, c = np.nonzero(np.c_[np.ones((len(P), 1), bool), fs[:, 1:] != fs[:, :-1]])
        T = self.V[self.F[fs[r, c]]]
        D = np.linalg.norm(P[r] - TGM.closest_point(P[r], T[:, 0], T[:, 1], T[:, 2]), axis=-1)
        return (np.minimum.reduceat(D, np.flatnonzero(np.r_[True, r[1:] != r[:-1]])) if len(D) else D), dk[:, 0]

    def inside(self, P: np.ndarray, d1: np.ndarray) -> np.ndarray:
        """|winding number| > 1/2 at each point (d1: the distance to the nearest sample, or a lower bound of it, read
        in the box), exact (module doc): summed outside the box, near the surface or closure, and on a closure over
        CLOSURE_MAX faces."""
        W, w = self._winding(), np.zeros(len(P))
        C = W.closure if W.closure is not None else np.zeros((0, 3, 3))
        f = np.flatnonzero(np.all((self.lo - 1e-9 <= P) & (self.hi + 1e-9 >= P), axis=1) & (len(C) <= CLOSURE_MAX))
        rho = np.minimum(d1[f] - self.cover, TGM.distance(P[f], C) if len(C) else np.inf) * (1.0 - 1e-9) - 1e-12
        f, rho = f[rho > 0], rho[rho > 0]
        sums = np.ones(len(P), bool)
        sums[f] = False
        w[sums] = W(P[sums])
        if len(f):
            wc = TGM.solid(C, P[f])
            lab, rep = TGM.regions(P[f], rho)
            w[f] = np.round(W(P[f[rep]]) + wc[rep])[lab] - wc
        return np.abs(w) > 0.5

    def signed(self, P: np.ndarray) -> np.ndarray:
        """Signed distance of each point to its nearest face (< 0 inside: |winding number| above 1/2, `inside`;
        module doc), the face searched among the faces of its K_NEAR nearest samples."""
        P = np.asarray(P, float)
        d, d1 = self._nearest(P)
        return np.where(self.inside(P, d1), -d, d)

    def within(self, P: np.ndarray, r: float) -> np.ndarray:
        """signed(P) <= r at each point (r >= 0), exactly, the sign read only where the distance does not decide: a
        point farther than r from the box is within only if inside, a point whose distance is at most r is within."""
        if not r >= 0:
            raise ValueError(f"{MARK} within: r = {r!r}, want r >= 0")
        P = np.asarray(P, float)
        out, d1 = np.zeros(len(P), bool), np.full(len(P), np.nan)
        near = np.flatnonzero(np.all((self.lo - r - 1e-9 <= P) & (self.hi + r + 1e-9 >= P), axis=1))
        d1[near] = self.tree.query(P[near], k=1)[0]
        out[near[d1[near] <= r - REACH_EPS_M]] = True  # nearest face <= d1 <= r (`_reach`)
        band = near[(d1[near] > r - REACH_EPS_M) & (d1[near] <= self._reach(r))]  # beyond it, the face is > r away
        out[band[self._nearest(P[band])[0] <= r]] = True
        rest = np.flatnonzero(~out)
        out[rest] = self.inside(P[rest], d1[rest])
        return out

    def _reach(self, r: float) -> float:
        """A nearest-sample distance beyond which the nearest face is farther than r: every surface point lies within
        `cover` of a sample (`_samples`: the proven bound of its face grid, taken as the max over the faces), so the
        surface is at least d1 - cover away. REACH_EPS_M covers the rounding of d1, cover and the face distance;
        the nearest face is also at most d1 away (the nearest sample lies on one of the searched faces)."""
        return r + self.cover * (1.0 + 1e-6) + REACH_EPS_M

    def depths(self, P: np.ndarray) -> np.ndarray:
        """max(-signed, 0) in the box (0 outside it), the nearest-face search only where it sets the value: inside,
        or within `_reach(0)` of a sample (a point on the surface reads 0 with signed's sign); any other point is
        outside, farther than 0 from the surface, and reads +0.0."""
        P = np.asarray(P, float)
        out = np.zeros(len(P))
        inb = np.flatnonzero(np.all((self.lo < P) & (self.hi > P), axis=1))
        if len(inb):
            # The search stops at u (inf beyond it); `inside` reads min(d1, u), a lower bound of the nearest-sample
            # distance, so its free balls are smaller and its sums the same (exact), and `need` (d1 <= _reach(0) <= u)
            # is unchanged.
            u = max(DEPTH_QUERY_M, 2.0 * self._reach(0.0))
            Q, d1 = P[inb], self.tree.query(P[inb], k=1, distance_upper_bound=u)[0]
            ins = self.inside(Q, np.minimum(d1, u))
            need = np.flatnonzero(ins | (d1 <= self._reach(0.0)))
            d = self._nearest(Q[need])[0]
            out[inb[need]] = np.maximum(-np.where(ins[need], -d, d), 0.0)
        return out

    def depth(self, P: np.ndarray) -> float:
        return float(self.depths(P).max()) if len(P) else 0.0


def visual_mesh(body):
    """The visual mesh prim of a rigid body prim: the one Mesh under its VISUALS child (usd_build writes
    `<body>/visuals/mesh`; read the same before and after the env authors it as the collider). Not exactly one, or not
    all triangles, raises."""
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    vis = body.GetChild(VISUALS)
    got = [p for p in Usd.PrimRange(vis) if p.IsA(UsdGeom.Mesh)] if vis.IsValid() else []
    if len(got) != 1:
        raise ValueError(
            f"{MARK} {body.GetPath()}: {len(got)} meshes under {VISUALS} (want 1): {[str(p.GetPath()) for p in got]}"
        )
    counts = np.asarray(UsdGeom.Mesh(got[0]).GetFaceVertexCountsAttr().Get() or [])
    if not len(counts) or (counts != 3).any():
        raise ValueError(f"{MARK} {got[0].GetPath()}: not a triangle mesh")
    return got[0]


def triangles(ds: str, obj_id: str) -> Triangles:
    """The visual mesh of the rigid training USD a slot spawns (`hull.usd_file`), body frame; cached."""
    key = (ds, str(obj_id))
    if key in _CACHE:
        return _CACHE[key]
    from pxr import Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

    from dexengine.data.objects import hull as OH  # noqa: PLC0415 — owner of the training USD path

    f = OH.usd_file(ds, str(obj_id), False)
    if not f.is_file():
        raise OH.UsdMissing(f)
    st = Usd.Stage.Open(str(f))
    bodies = [p for p in st.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != 1:
        raise OH.HullMissing(f"{f}: {len(bodies)} rigid bodies (a still slot spawns one)")
    m = visual_mesh(bodies[0])
    xc = UsdGeom.XformCache()
    M = np.array(xc.GetLocalToWorldTransform(m) * xc.GetLocalToWorldTransform(bodies[0]).GetInverse())
    mesh = UsdGeom.Mesh(m)
    V = np.asarray(mesh.GetPointsAttr().Get(), float) @ M[:3, :3] + M[3, :3]  # Gf row-vector convention
    tri = Triangles(V, np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), np.int64))
    print(
        f"{MARK} {ds}/{obj_id} faces {len(tri.F)} samples {len(tri.points)} digest {tri.digest} "
        f"sign {tri.sign_rule} (collider_shape.triangles: {m.GetPath()})",
        flush=True,
    )
    _CACHE[key] = tri
    return tri


def parts(ds: str, obj_id: str) -> Parts:
    """The cooked convex parts of the rigid training USD (`hull.collision_parts`, every link), body frame; cached.
    HullMissing raises."""
    key = ("parts", ds, str(obj_id))
    if key not in _CACHE:
        from dexengine.data.objects import hull as OH  # noqa: PLC0415

        cp = OH.collision_parts(ds, str(obj_id), False)
        _CACHE[key] = Parts([P for ps in cp.links.values() for P in ps], cp.digest)
    return _CACHE[key]


def body(ds: str, obj_id: str, shape: str) -> Triangles | Parts:
    """The ruler of one spawned body in the shape `shape_of` stated (module doc)."""
    if shape not in SHAPES:
        raise ValueError(f"{MARK} unknown collider shape {shape!r} (known {SHAPES})")
    return triangles(ds, obj_id) if shape == TRIANGLES else parts(ds, obj_id)


def row_body(ds: str, o: dict, row: dict) -> Triangles | Parts:
    """`body` of the object `o` of an extra_slots row, in the row's shape (`row_shape`)."""
    return body(ds, str(o["obj_id"]), row_shape(ds, row))


def _xf(M: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ M[:3, :3].T + M[:3, 3]


def _into_b(a, b, M_ba: np.ndarray) -> float:
    near = a.near(_xf(M_ba, b.center[None])[0], b.radius)
    return b.depth(_xf(np.linalg.inv(M_ba), near)) if len(near) else 0.0


def pair_depth(a, b, M_ba: np.ndarray) -> float:
    """The deeper reading of two bodies (rulers `a`, `b`; M_ba = pose of b's frame in a's frame, 4x4): b's points in
    a, and a's points within b's bounding sphere in b."""
    M_ba = np.asarray(M_ba, float)
    return max(a.depth(_xf(M_ba, b.points)), _into_b(a, b, M_ba))


def pair_depth_within(a, b, M_ba: np.ndarray, tol: float) -> float | None:
    """`pair_depth` when it is at most tol, else None (it exceeds tol, by an amount not read): a's points in b are
    read first, b's points in a only when those stay within tol."""
    M_ba = np.asarray(M_ba, float)
    into = _into_b(a, b, M_ba)
    d = max(a.depth(_xf(M_ba, b.points)), into) if into <= tol else np.inf
    return d if d <= tol else None


def overlap_drop(rows: list) -> tuple[str, str] | None:
    """(OVERLAP_DROP_WORD, description) when a still row states a true-shape overlap above
    scene_requirement.TRUE_SHAPE_OVERLAP_M with a moving body (row field `overlap_drop` = {partner, depth_m}, written
    by S-touch-gate and S-envelope), else None: the clip's exclusion verdict (rule: a real overlap above the value
    is dropped, never kept with the collision off)."""
    got = [
        f"{r['name']}×{r['overlap_drop']['partner']} {r['overlap_drop']['depth_m'] * 1000:.1f} mm"
        for r in rows
        if r.get("overlap_drop")
    ]
    return (OVERLAP_DROP_WORD, "true-shape overlap " + " · ".join(got)) if got else None
