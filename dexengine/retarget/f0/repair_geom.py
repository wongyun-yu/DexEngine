"""f0 repair, geometry layer: rotation helpers, the hand model (FK and collision points), the object geometry, hull
containment, the strict-check model and the wrist qpos fit."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from dexengine.core.paths import urdf_mesh
from dexengine.placement.coords.core import artic_moving_vertices
from dexengine.retarget.f0.repair_cfg import HC, finger_limits


# geometry
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


class HandModel:
    """Per-link convex-hull vertices of one hand, FK'd relative to the hand base link."""

    def __init__(self, side: str, max_hull_verts: int = 400, pen_margin: float = 0.0):
        import trimesh

        self.side = side
        self.pen_margin = pen_margin
        self.hc = HC
        self.urdf_path = HC.urdf(side)
        from dexengine.robots import urdf_load  # noqa: PLC0415

        self.model = urdf_load.build_model(str(self.urdf_path))
        self.data = self.model.createData()
        self.base_frame = self.model.getFrameId(HC.base_frame(side))
        # joint name -> configuration index (1-DOF revolute joints -> idx_q scalar)
        self.jidx = {}
        self.jnames = [HC.joint_name(side, f) for f in HC.order]
        for jn in self.jnames:
            jid = self.model.getJointId(jn)
            self.jidx[jn] = self.model.joints[jid].idx_q
        # collision hulls per hand link, parsed straight from the URDF text (origin+mesh)
        txt = self.urdf_path.read_text()
        self.links = []  # (frame_id, local_pts (N,3))
        # Every link on that side is hand (the arm links carry no `<side>_` prefix).
        # SolidWorks-family exports break the attributes across lines (<link\n  name="...">).
        _link_re = r'<link\s+name="(%s_[^"]+)"\s*>(.*?)</link>' % side  # noqa: UP031
        for m in re.finditer(_link_re, txt, re.S):
            lname, body = m.group(1), m.group(2)
            pts_all = []
            for c in re.finditer(r"<collision>(.*?)</collision>", body, re.S):
                cb = c.group(1)
                fm = re.search(r'filename="([^"]+)"', cb)
                if not fm:
                    continue
                oxyz = re.search(r'<origin[^>]*xyz="([^"]+)"', cb)
                orpy = re.search(r'<origin[^>]*rpy="([^"]+)"', cb)
                xyz = np.fromstring(oxyz.group(1), sep=" ") if oxyz else np.zeros(3)
                rpy = np.fromstring(orpy.group(1), sep=" ") if orpy else np.zeros(3)
                sm = re.search(r'scale="([^"]+)"', cb)
                scale = np.fromstring(sm.group(1), sep=" ") if sm else np.ones(3)
                mesh = trimesh.load(urdf_mesh(self.urdf_path, fm.group(1)), force="mesh")
                # FULL COLLISION SURFACE, not the convex hull. The hull's vertices
                # are only the extreme corners, so a concave region — the palm cup, the inside of
                # a curled finger — contributes no test points at all, and an object resting in
                # that cup reads as "no overlap"; random-subsampling those hull vertices (60 per link:
                # 1 620 points against a 37 554-vertex hand, 4 %) makes it worse — audited: all 28
                # "repaired, n_pen=0" sides still overlapped under a strict test (51-4 830 points).
                #
                # Decimation is deterministic voxel vertex-clustering, not random sampling, so the
                # kept points still cover the whole surface (including concavities) and the result
                # is reproducible. Verification later re-tests with EVERY vertex.
                v = np.asarray(mesh.vertices, float) * scale
                v = _decimate_vertices(v, max_hull_verts)
                pts_all.append(v @ _rpy_to_R(rpy).T + xyz)
            if not pts_all:
                continue
            fid = self.model.getFrameId(lname)
            pts = np.vstack(pts_all)
            dl = _reverse_hull(pts, self.pen_margin)
            self.links.append((lname, fid, pts, dl))
        if not self.links:
            raise RuntimeError(f"no {side} hand-link collision meshes found in the URDF")
        # JOINT LIMITS = the reference finger range (owner wrist_shift.finger_ranges): the vendor range clipped by
        # the retargeting IK's hyperextension rule, so this repair never re-opens a pose the retargeting cut away.
        # A joint without a vendor limit stays unbounded.
        ref = finger_limits()[1]
        self.limits = {fj: ref.get(jn, (-np.inf, np.inf)) for fj, jn in zip(HC.order, self.jnames, strict=True)}

    def fingers12(self, indep6):
        return HC.expand(indep6)

    def link_points_world(self, wrist_pos, wrist_R, fingers12):
        """All hull points in trajectory-frame world; also per-link (name, pts)."""
        import pinocchio as pin

        q = pin.neutral(self.model)
        for jn, val in zip(self.jnames, fingers12, strict=True):
            q[self.jidx[jn]] = val
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        Tb = self.data.oMf[self.base_frame]
        Tb_inv = Tb.inverse()
        out = []
        for lname, fid, pts, dl in self.links:
            rel = Tb_inv * self.data.oMf[fid]
            Rw = wrist_R @ rel.rotation
            tw = wrist_R @ rel.translation + wrist_pos
            out.append((lname, pts @ Rw.T + tw, Rw, tw, dl))
        return out


class ObjectGeom:
    """CONCAVE penetration test. The convex hull is only a cheap PRE-FILTER: fingers
    wrapping a tool handle sit inside the hull WITHOUT touching the mesh (TACO is a
    tool-grasping dataset — frame 0 usually holds the tool), and one clip
    showed it: it trained to 100 % yet had hull-n_pen 434. PhysX likewise
    collides a convex DECOMPOSITION (concave-capable), so the truth test is signed
    distance to the actual surface: nearest dense surface sample + its face normal;
    a point is penetrating only if it is INSIDE and deeper than pen_margin."""

    def __init__(
        self,
        mesh_dir: Path,
        n_samples: int = 20000,
        sp_dir: Path | None = None,
        articulation: dict | None = None,
        theta: float = 0.0,
    ):
        """mesh_dir/visual.obj is the geometry. surface_points.npz comes from sp_dir (mesh_dir if
        absent) — in the base layer the mesh (assets/objects) and the derived surface points (the
        meshes layer) live in different directories (the TACO legacy data/taco/meshes/<id> is one
        place).
        articulation (2c, D1 articulated object): given `objects[].articulation` (parts.base/moving · axis · pos ·
        type), the geometry = the base part ∪
        the moving part put through joint FK (θ=theta, the joint value at frame 0) — not the closed
        visual.obj (342 of arctic's 800 start open). The surface points are resampled from that
        composed mesh (surface_points.npz is for the closed pose, so it is not used)."""
        import trimesh
        from scipy.spatial import Delaunay, cKDTree

        self.articulated = bool(articulation)
        if articulation:
            parts = articulation.get("parts") or {}
            # link = part ∪ fixed attachments (artifact_paths.artic_link_files)
            from dexengine.data.object_surface import link_mesh  # noqa: PLC0415

            mb = link_mesh(mesh_dir, parts, "base", process=True)
            mm = link_mesh(mesh_dir, parts, "moving", process=True)
            ax = np.asarray(articulation["axis"], float)
            ax = ax / max(np.linalg.norm(ax), 1e-12)
            V = artic_moving_vertices(
                articulation, np.asarray(mm.vertices, float), float(theta)
            )  # coords owns the axis/anchor/child_pose convention
            mm = trimesh.Trimesh(vertices=V, faces=np.asarray(mm.faces), process=False)
            mesh = trimesh.util.concatenate([mb, mm])
            sp_dir = None  # surface points = composed-mesh samples, not the closed-pose sidecar
            mesh_dir = mesh_dir / "__artic_no_sidecar__"  # send the fallback to a surface_points.npz that cannot exist
            print(
                f"[ARTIC] f0 {Path(str(mesh_dir)).parent.name}: articulated geometry = {parts['base']} ∪ "
                f"FK({parts['moving']}, θ0={float(theta):+.4f} {articulation.get('type', 'hinge')} "
                f"axis={ax.round(3).tolist()}) · surface points resampled",
                flush=True,
            )
        else:
            mesh = trimesh.load(mesh_dir / "visual.obj", force="mesh")
        trimesh.repair.fix_normals(mesh)  # no swallow: a mesh it cannot orient dies here (734/734 assets pass)
        self.watertight = bool(mesh.is_watertight)
        self.mesh = mesh  # keep the original for the exact test (ray casting)
        hull = mesh.convex_hull
        self.delaunay_local = Delaunay(np.asarray(hull.vertices))
        s_pts, s_fid = trimesh.sample.sample_surface(mesh, n_samples, seed=0)
        self.kdt = cKDTree(np.asarray(s_pts))
        self._s_pts = np.asarray(s_pts)
        self._s_nrm = np.asarray(mesh.face_normals)[np.asarray(s_fid)]
        self._lo = self._s_pts.min(0)  # AABB pre-filter for the clearance test
        self._hi = self._s_pts.max(0)
        # 26-DOP (discrete oriented polytope): the convex hull's support value along the 26
        # directions (±x, ±y, ±z, edge and corner diagonals of the cube). The hull is contained in
        # it, so a point farther than `clearance` from the DOP is farther than `clearance` from
        # every surface sample AND outside the hull — it can be neither `deep` nor `close`, so
        # dropping it before the KD query changes nothing. The AABB is 6 of these
        # 26 directions; the extra 20 cut the KD query set by about half on a tool-shaped object
        # (measured on TACO/dg5f_s: 115.6 → 61.6 points per evaluation, kdt 227 → 95 us, filter 7 us).
        _d = np.array(
            [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)],
            dtype=float,
        )
        self._dop_D = _d / np.linalg.norm(_d, axis=1, keepdims=True)
        self._dop_hi = (np.asarray(self.delaunay_local.points) @ self._dop_D.T).max(0)
        sp_path = (sp_dir or mesh_dir) / "surface_points.npz"
        if sp_path.exists():
            sp = np.load(sp_path)
            key = "points" if "points" in sp.files else sp.files[0]
            pts = np.asarray(sp[key], dtype=np.float64).reshape(-1, 3)
        else:
            pts = self._s_pts  # some objects (e.g. 146) have no sidecar — fall back to mesh samples
        if len(pts) > 400:
            pts = pts[np.random.RandomState(0).choice(len(pts), 400, replace=False)]
        self.local_surface = pts

    def penetration_stats(self, R, pos, world_pts, pen_margin, clearance=0.0):
        """(count, depth_sum[m]). depth_sum is CONTINUOUS — the count alone is a step
        function and Nelder-Mead stalls on its plateaus (a side failed with count-only;
        depth gives the simplex a slope to descend).

        `clearance` > 0 additionally requires OUTSIDE points to stay that far from the
        surface. The optimiser needs it because the tested points are a subset of the hand
        surface: driving the sampled points to exactly zero overlap leaves the unsampled
        geometry between them still touching. Solve with a gap, verify with every vertex.
        """
        local = (world_pts - pos) @ R  # world -> object local
        if clearance > 0.0:
            # The clearance test has to count 'outside but close' points too, so the hull
            # pre-filter cannot be used. Instead, filter with the object AABB inflated by
            # clearance — a point outside that box is self-evidently farther from the surface than
            # clearance, so no accuracy is lost. Without it the KD query runs over the whole hand
            # point cloud (about 2 × 10⁴), 38 ms per evaluation, which Nelder-Mead cannot afford.
            # The AABB test is a per-axis boolean AND: building an (N,3) boolean array with `(local >= a) & (local <=
            # b)` and reducing it with `.all(1)` spends most of the filter's time on the last-axis reduction at this
            # size (N≈9,000; measured 227 us -> 50 us). `local` is still computed in full, so `q = local[sel]` below is
            # bit-identical, and the index set is the conjunction of the same six comparisons, so it is identical too.
            _c0 = local[:, 0]
            sel = np.flatnonzero((_c0 >= self._lo[0] - clearance) & (_c0 <= self._hi[0] + clearance))
            if sel.size:
                _c = local[sel]
                sel = sel[
                    (_c[:, 1] >= self._lo[1] - clearance)
                    & (_c[:, 1] <= self._hi[1] + clearance)
                    & (_c[:, 2] >= self._lo[2] - clearance)
                    & (_c[:, 2] <= self._hi[2] + clearance)
                ]
            if sel.size == 0:
                return 0, 0.0
            q = local[sel]
            # Second filter: 26-DOP. `_DOP_EPS` is a 1 nm floating-point margin only; points on the boundary are kept
            # (the filter may only err towards keeping points).
            _keep = np.flatnonzero((self._dop_D @ q.T - self._dop_hi[:, None]).max(0) <= clearance + _DOP_EPS)
            if _keep.size == 0:
                return 0, 0.0
            if _keep.size != sel.size:
                sel = sel[_keep]
                q = local[sel]
            dist, idx = self.kdt.query(q)
            inside = (self.delaunay_local.find_simplex(q) >= 0) & (
                np.einsum("ij,ij->i", q - self._s_pts[idx], self._s_nrm[idx]) < 0
            )
            deep = inside & (dist > pen_margin)
            close = (~inside) & (dist < clearance)
            n = int(np.count_nonzero(deep) + np.count_nonzero(close))
            depth = float(dist[deep].sum() + (clearance - dist[close]).sum())
            return n, depth
        cand = self.delaunay_local.find_simplex(local) >= 0  # hull pre-filter
        if not np.any(cand):
            return 0, 0.0
        q = local[cand]
        dist, idx = self.kdt.query(q)
        inside = np.einsum("ij,ij->i", q - self._s_pts[idx], self._s_nrm[idx]) < 0
        pen = inside & (dist > pen_margin)
        return int(np.count_nonzero(pen)), float(dist[pen].sum())

    def contains_exact(self, R, pos, world_pts):
        """The number of points inside the object — the EXACT test, by ray casting.

        The approximation in `penetration_stats` (the sign of the nearest normal over 20 000
        surface samples) has its spatial resolution tied to the sample spacing. Measured
        0.96~3.03 mm, so penetration shallower than that is in principle indistinguishable — and
        demanding a threshold of 0 mm anyway leaves the optimiser spinning after a 0 buried in
        noise. Every object mesh was confirmed watertight (measured), so the final test at least is
        done without the approximation, with `mesh.contains`. 160 ms for 8 000 points / 4.7 s for
        22 × 10⁴ — unusable in the inner loop, good enough for verification. AABB narrows the
        candidates first (outside the box is self-evidently outside).
        """
        local = (world_pts - pos) @ R
        sel = np.flatnonzero(((local >= self._lo) & (local <= self._hi)).all(1))
        if sel.size == 0:
            return 0
        # PREFILTER + SMALL CHUNKS. trimesh's pure-python ray engine allocates intermediates
        # proportional to (rays x candidate faces): at 86 k faces even a 20 k-ray chunk peaks
        # in the GBs and got the process SIGKILLed (DBK 154, three times, 76 GB system free —
        # a transient spike). Two exact-safe reductions: (1) the convex hull contains the mesh,
        # so points outside the hull are outside the mesh — drop them first; (2) 1 000-ray
        # chunks bound the per-call peak to ~tens of MB.
        q = local[sel]
        q = q[self.delaunay_local.find_simplex(q) >= 0]
        n = 0
        for i in range(0, len(q), 1000):
            n += int(np.count_nonzero(self.mesh.contains(q[i : i + 1000])))
        return n

    def surface_world(self, R, pos):
        return self.local_surface @ R.T + pos


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


_DOP_EPS = 1e-9  # m — floating-point margin of the 26-DOP filter (1 nm), geometrically negligible


_HAND_CACHE: dict = {}


_OBJ_CACHE: dict = {}


_STRICT_CACHE: dict = {}


def _strict_model(side: str):
    """Hand model using every collision-mesh vertex — final verification only (no decimation, no subsampling)."""
    m = _STRICT_CACHE.get((HC.name, side))
    if m is None:
        m = HandModel(side, max_hull_verts=10**9, pen_margin=0.0)
        _STRICT_CACHE[(HC.name, side)] = m
    return m


def _clr_mm(x: float) -> str:
    return f"{x * 1000:.1f}mm"


# qpos[0:6] map
#: Max residual (m) of the affine fit between wrist-cache positions and hand-root translation qpos. For a matching
#: cache the residual is floating-point noise (measured over all XHand, Tesollo and Wuji datasets, 97,374 sides: max
#: 7.9e-8, none >= 1e-4). Above it, the cache did not come from that qpos.
WRIST_FIT_MAX_RES = 1e-4


def fit_wrist_qpos_map(qpos6_all, cache_pos, where: str = ""):
    """Fit  cache_pos = A @ qpos[0:3] + b  (exact for stacked slide joints) and validate.
    Rotation is NOT fit (hinge composition is nonlinear); written frames keep original
    rotation qpos and rely on the wrist cache for truth. Returns (A, b, max_res).

    A residual >= WRIST_FIT_MAX_RES raises (ValueError; the f0 engine counts it as a failure of that one clip) instead
    of leaving the pad's hand-root translation at stale values, which would ship a tree whose hand root and wrist cache
    disagree as 'repaired'. The usual cause is a stale wrist cache, i.e. cache and qpos from different generations:
    rerun stage 6 (hand_pipeline --emit-run 6)."""
    X = qpos6_all[:, :3]
    Y = cache_pos
    Xh = np.hstack([X, np.ones((len(X), 1))])
    sol, *_ = np.linalg.lstsq(Xh, Y, rcond=None)
    A, b = sol[:3].T, sol[3]
    res = float(np.abs(Xh @ sol - Y).max())
    if not res < WRIST_FIT_MAX_RES:
        raise ValueError(
            f"[F0_WRIST_FIT] ✗ {where}: affine fit residual wrist cache ↔ hand-root qpos {res:.2e} m ≥ "
            f"{WRIST_FIT_MAX_RES:g} — the cache did not come from this qpos (stale cache; rerun stage 6)"
        )
    return A, b, res
