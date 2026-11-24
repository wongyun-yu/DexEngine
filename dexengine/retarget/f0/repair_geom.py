"""f0 repair, geometry layer: the hand model (FK and collision points), the object geometry, the strict-check model and
the f0 key's input lists. The numeric helpers (rotations, link hulls, the wrist map) live in repair_math and are
re-exported here (the same objects)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from dexengine.core.paths import urdf_mesh
from dexengine.placement.coords.core import artic_moving_vertices
from dexengine.retarget.f0.repair_cfg import HC, finger_limits
from dexengine.retarget.f0.repair_math import (  # noqa: F401 — moved definitions, re-exported (the same objects)
    WRIST_FIT_MAX_RES,
    _aa_to_R,
    _decimate_vertices,
    _quat_wxyz_to_R,
    _R_to_quat_wxyz,
    _reverse_hull,
    _rpy_to_R,
    _slerp,
    fit_wrist_qpos_map,
    hulls_contain,
)

# ── what the f0 geometry reads (one owner for the loaders here and for the f0 cache key, shadow_key) ────────────────
# The key hashes exactly these, so a file or a URDF element f0 never reads (a stray file beside surface_points.npz, a
# material definition, an arm mesh) cannot move it, and one it does read always does. Lines are strings read from
# the files (no float computed on this host), so the key is the same on every machine.
OBJ_MESH, OBJ_SP = "mesh:visual.obj", "sp:surface_points.npz"


def object_input_files(mesh_dir, sp_dir=None, articulation=None) -> dict:
    """name -> path of every file ObjectGeom opens for one object. Rigid: mesh_dir/visual.obj and the surface points
    (sp_dir, else mesh_dir)/surface_points.npz, which is optional (the loader falls back to mesh samples; the key
    records its absence). Articulated: the link meshes of artifact_paths.artic_link_files (owner of that list; the
    composed mesh is sampled, so no sidecar)."""
    mesh_dir = Path(mesh_dir)
    if articulation:
        from dexengine.core import artifact_paths  # noqa: PLC0415

        parts = articulation.get("parts") or {}
        return {
            f"mesh:{rel}": mesh_dir / rel
            for link in artifact_paths.ARTIC_LINKS
            for rel, _pose in artifact_paths.artic_link_files(parts, link)
        }
    return {OBJ_MESH: mesh_dir / "visual.obj", OBJ_SP: (Path(sp_dir) if sp_dir else mesh_dir) / "surface_points.npz"}


def hand_collision_blocks(urdf_text: str, side: str) -> list:
    """[(link, [(mesh filename, origin xyz, origin rpy, scale) ...])] of the `<side>_` links' <collision> blocks with a
    mesh, in file order, attribute values as written ('' when absent) — what HandModel reads from the URDF text."""
    out = []
    # Every link on that side is hand (the arm links carry no `<side>_` prefix).
    # SolidWorks-family exports break the attributes across lines (<link\n  name="...">).
    _link_re = r'<link\s+name="(%s_[^"]+)"\s*>(.*?)</link>' % side  # noqa: UP031
    for m in re.finditer(_link_re, urdf_text, re.S):
        blocks = []
        for c in re.finditer(r"<collision>(.*?)</collision>", m.group(2), re.S):
            cb = c.group(1)
            fm = re.search(r'filename="([^"]+)"', cb)
            if not fm:
                continue
            oxyz = re.search(r'<origin[^>]*xyz="([^"]+)"', cb)
            orpy = re.search(r'<origin[^>]*rpy="([^"]+)"', cb)
            sm = re.search(r'scale="([^"]+)"', cb)
            blocks.append((fm.group(1), *(x.group(1) if x else "" for x in (oxyz, orpy, sm))))
        out.append((m.group(1), blocks))
    return out


def hand_chain_joints(urdf_text: str, base: str, links: list) -> list:
    """The URDF <joint> elements between the hand base link and each collision link (the transforms HandModel's
    relative FK goes through), as (name, type, parent, child, origin xyz, origin rpy, axis) strings, sorted by name."""
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    root = ET.fromstring(urdf_text)
    by_child = {j.find("child").get("link"): j for j in root.iter("joint") if j.find("child") is not None}

    def path(link):
        out = []
        while link in by_child:
            j = by_child[link]
            out.append(j)
            link = j.find("parent").get("link")
        return out

    if base not in {ln.get("name") for ln in root.iter("link")}:
        raise ValueError(f"[F0_KEY] hand base frame {base!r} is not a URDF link — the chain cannot be read")
    pb = {j.get("name"): j for j in path(base)}
    rel = {}
    for ln in links:  # base -> link = the joints on one root path and not the other
        pl = {j.get("name"): j for j in path(ln)}
        for n in set(pl) ^ set(pb):
            rel[n] = pl.get(n, pb.get(n))

    def _a(el, tag, attr):
        e = el.find(tag)
        return "" if e is None else (e.get(attr) or "")

    return [
        (
            n,
            j.get("type") or "",
            _a(j, "parent", "link"),
            _a(j, "child", "link"),
            _a(j, "origin", "xyz"),
            _a(j, "origin", "rpy"),
            _a(j, "axis", "xyz"),
        )
        for n, j in sorted(rel.items())
    ]


def hand_input_lines(side: str) -> tuple:
    """(key lines, {mesh filename: path}) of what HandModel(side) reads: the hand identity (name, base frame, joint
    names), a non-zero palm offset (HandModel.palm_inv; a hand without keeps its lines), each collision block (link,
    mesh, origin, scale), the joints of the base -> link chains, and the reference finger ranges it clamps to. Owner
    of the f0 key's hand part (shadow_key)."""
    import json  # noqa: PLC0415

    u = Path(HC.urdf(side))
    txt = u.read_text()
    jnames = [HC.joint_name(side, f) for f in HC.order]
    blocks = hand_collision_blocks(txt, side)
    links = [ln for ln, b in blocks if b]
    lines = [f"hand|{HC.name}|base={HC.base_frame(side)}|joints={','.join(jnames)}"]
    from dexengine.placement import arm_ik as AIP  # noqa: PLC0415 — owner of the palm offset (HandModel.palm_inv)

    off = [float(x) for x in AIP.palm_frame_offset(HC.name)]
    if any(off):  # a hand with a palm offset; a hand without keeps its key
        lines.append(f"palm_off|{json.dumps(off)}")
    files = {}
    for ln, b in blocks:
        for fname, oxyz, orpy, sm in b:
            lines.append(f"coll|{ln}|{fname}|{oxyz}|{orpy}|{sm}")
            files[fname] = urdf_mesh(u, fname)
    lines += ["joint|" + "|".join(t) for t in hand_chain_joints(txt, HC.base_frame(side), links)]
    ref = finger_limits()[1]
    lines.append("range|" + json.dumps({jn: list(ref[jn]) for jn in jnames if jn in ref}, sort_keys=True))
    return lines, files


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
        # The wrist cache holds the rig palm. The training robot's URDF palm stands at rig palm + R · palm offset (arm
        # IK, judge and bake build their target so; owner placement.arm_ik.palm_frame_offset), so every palm-relative
        # transform of this model starts from that palm (`palm_inv`). Measured on SharpA (0.0295 m): without it the
        # collision pieces sat 29.5 mm off the tree scene.xml collision geoms (rigid fit residual 0.0 mm either way).
        import pinocchio as pin  # noqa: PLC0415 — the f0 interpreter's dependency

        from dexengine.placement import arm_ik as AIP  # noqa: PLC0415 — owner of the palm offset

        self.palm_off = np.asarray(AIP.palm_frame_offset(HC.name), float)
        self.palm_shift = pin.SE3(np.eye(3), self.palm_off)
        # joint name -> configuration index (1-DOF revolute joints -> idx_q scalar)
        self.jidx = {}
        self.jnames = [HC.joint_name(side, f) for f in HC.order]
        for jn in self.jnames:
            jid = self.model.getJointId(jn)
            self.jidx[jn] = self.model.joints[jid].idx_q
        # collision hulls per hand link, parsed straight from the URDF text (origin+mesh) by hand_collision_blocks
        self.links = []  # (frame_id, local_pts (N,3))
        # link -> its collision meshes' vertex sets (link frame), one array per <collision>: the convex pieces the sim
        # collides (one PhysX convex per mesh), read by the object-overlap measure (repair_obj.sat_depth)
        self.parts = {}
        for lname, blocks in hand_collision_blocks(self.urdf_path.read_text(), side):
            pts_all = []
            for fname, oxyz, orpy, sm in blocks:
                xyz = np.fromstring(oxyz, sep=" ") if oxyz else np.zeros(3)
                rpy = np.fromstring(orpy, sep=" ") if orpy else np.zeros(3)
                scale = np.fromstring(sm, sep=" ") if sm else np.ones(3)
                mesh = trimesh.load(urdf_mesh(self.urdf_path, fname), force="mesh")
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
            self.parts[lname] = pts_all
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

    def palm_inv(self):
        """The inverse placement of the palm as the training robot stands it, after FK of `data`: the URDF palm frame
        moved by the palm offset (`palm_shift`, in palm-local coordinates). Every palm-relative transform reads it."""
        return self.palm_shift * self.data.oMf[self.base_frame].inverse()

    def link_points_world(self, wrist_pos, wrist_R, fingers12):
        """All hull points in trajectory-frame world; also per-link (name, pts)."""
        import pinocchio as pin

        q = pin.neutral(self.model)
        for jn, val in zip(self.jnames, fingers12, strict=True):
            q[self.jidx[jn]] = val
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        Tb_inv = self.palm_inv()
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
        files = object_input_files(mesh_dir, sp_dir, articulation)  # what this loader opens (the f0 key's list too)
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
            print(
                f"[ARTIC] f0 {Path(str(mesh_dir)).name}: articulated geometry = {parts['base']} ∪ "
                f"FK({parts['moving']}, θ0={float(theta):+.4f} {articulation.get('type', 'hinge')} "
                f"axis={ax.round(3).tolist()}) · surface points resampled",
                flush=True,
            )
        else:
            mesh = trimesh.load(files[OBJ_MESH], force="mesh")
        trimesh.repair.fix_normals(mesh)  # no swallow: a mesh it cannot orient dies here (734/734 assets pass)
        self.watertight = bool(mesh.is_watertight)
        self.mesh = mesh  # the original: the exact test's triangles (`_tri`, built on first use)
        self._tri_ruler = None
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
        sp_path = files.get(OBJ_SP)  # none when articulated: composed-mesh samples, not the closed-pose sidecar
        if sp_path is not None and sp_path.exists():
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

    @property
    def _tri(self):
        """The object mesh as the true-shape ruler (collider_shape.Triangles), built on first use."""
        if self._tri_ruler is None:
            from dexengine.data.objects import collider_shape as CS  # noqa: PLC0415 — the one inside test

            self._tri_ruler = CS.Triangles(np.asarray(self.mesh.vertices, float), np.asarray(self.mesh.faces))
        return self._tri_ruler

    def contains_exact(self, R, pos, world_pts):
        """The number of points inside the object — the EXACT test, by the ruler owner.

        The approximation in `penetration_stats` (the sign of the nearest normal over 20 000
        surface samples) has its spatial resolution tied to the sample spacing. Measured
        0.96~3.03 mm, so penetration shallower than that is in principle indistinguishable — and
        demanding a threshold of 0 mm anyway leaves the optimiser spinning after a 0 buried in
        noise. The final test is therefore done without the approximation, on the mesh's own
        triangles: inside = collider_shape.Triangles.signed < 0 (|winding number| > 1/2, by the
        pseudo-normal on a closed welded mesh, summed on an open one), the one inside test of the
        repo — it does not assume a watertight mesh (trimesh `mesh.contains`, ray casting, did).
        AABB narrows the candidates first (outside the box is self-evidently outside), then the
        convex hull (it contains the mesh, so points outside it are outside the mesh).
        """
        local = (world_pts - pos) @ R
        sel = np.flatnonzero(((local >= self._lo) & (local <= self._hi)).all(1))
        if sel.size == 0:
            return 0
        q = local[sel]
        q = q[self.delaunay_local.find_simplex(q) >= 0]
        step = _EXACT_CHUNK  # bounds the per-call memory; each point's answer does not depend on it
        return sum(int(np.count_nonzero(self._tri.signed(q[i : i + step]) < 0)) for i in range(0, len(q), step))

    def surface_world(self, R, pos):
        return self.local_surface @ R.T + pos


_DOP_EPS = 1e-9  # m — floating-point margin of the 26-DOP filter (1 nm), geometrically negligible
_EXACT_CHUNK = 32768  # points per inside call of contains_exact (a memory setting: the count does not depend on it)


_HAND_CACHE: dict = {}


_OBJ_CACHE: dict = {}

#: fields of `objects[].articulation` the articulated geometry is built from (ObjectGeom ·
#: coords.artic_moving_vertices · artifact_paths.artic_link_files); a record without one of them dies instead of
#: falling back to a default
ARTIC_GEOM_FIELDS = ("type", "axis", "pos", "parts")


def object_cache_key(mesh_dir, sp_dir, articulation, theta) -> tuple:
    """The one key of a main object's check geometry: the f0 object cache (_OBJ_CACHE) and the GPU evaluator's obj_key
    (repair_solve passes the scene's key) both read it. The geometry is a function of the key alone, so a clip gets the
    same geometry whichever clips a process solved before it.

    Rigid: (mesh_dir, sp_dir). Articulated: + ("artic", the frame-0 joint value as read, the articulation record as
    sorted JSON). Measured with the key before (mesh, surface points, θ0 rounded to 1e-4 rad), the clip solved
    second in a process got the first one's geometry:
      - another joint record at the same angle — 13 OakInk2 laptop clips share raw_o02_0053_00002 with different fitted
        joints; close_laptop_lid 6b5c334699d5_s002 left n_pen 483 after 39c30d86600d_s005, 0 alone;
      - the same record at another angle within the rounding — pull_out_drawer 2e0d01782266_s000 right n_pen 461 alone,
        459 · 460 after drawer clips whose θ0 differs by < 6e-5 rad.
    An articulated object whose record lacks a geometry field (ARTIC_GEOM_FIELDS, parts.base · parts.moving) dies."""
    if articulation is None:
        return (mesh_dir, sp_dir)
    miss = [f for f in ARTIC_GEOM_FIELDS if not (isinstance(articulation, dict) and articulation.get(f))]
    parts = articulation.get("parts") if isinstance(articulation, dict) else None
    miss += [f"parts.{k}" for k in ("base", "moving") if not (isinstance(parts, dict) and parts.get(k))]
    if miss:
        raise SystemExit(
            f"[F0_OBJ_KEY] ✗ {Path(str(mesh_dir)).name}: articulated object without {miss} in its articulation record "
            "— its geometry cannot be built (owner repair_geom.object_cache_key)"
        )
    rec = json.dumps(articulation, sort_keys=True, separators=(",", ":"), default=str)
    return (mesh_dir, sp_dir, "artic", float(theta), rec)


_STRICT_CACHE: dict = {}


def _strict_model(side: str):
    """Hand model using every collision-mesh vertex — final verification only (no decimation, no subsampling)."""
    m = _STRICT_CACHE.get((HC.name, side))
    if m is None:
        m = HandModel(side, max_hull_verts=10**9, pen_margin=0.0)
        _STRICT_CACHE[(HC.name, side)] = m
    return m
