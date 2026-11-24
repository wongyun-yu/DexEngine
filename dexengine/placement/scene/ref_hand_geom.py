#!/usr/bin/env python3
"""Reference hand geometry — the clip-wide minimum distance between the tree's reference hand link meshes and a point
set (a measuring tool: no rule, no threshold). Read by static_desk.float_rule.

Hands     the **reference hands** = the tree's scene.xml (MJCF — the model body_positions.npz was made with) driven by
          the tree qpos (0/trajectory_kinematic.npz) per frame (mj_kinematics): the hand link mesh geoms. A side = the
          body tree whose joints lie in a task_info qpos_layout kind=hand span (no hand or body name tables). Visual and
          collision copies of the same body · mesh · place count once.
Object    the same model's mesh geoms of that object body (helper group 3 excluded), placed at a given 7-pose (capture
          x·y·z · wxyz).
Distance  nearest pair between two surface samples (vertices + area-proportional face samples → one point per SAMPLE_M
          voxel) — resolution ≈ SAMPLE_M. Branch and bound: per (frame, geom) the lower bound = geom centre ↔ nearest
          object point − geom radius; a bound at or above the current minimum is not measured.
Needs mujoco. A missing scene.xml or hand geometry raises.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: Hand · object surface sample spacing (m) — the distance resolution.
SAMPLE_M = 0.002


class RefHandsError(RuntimeError):
    """The reference hand geometry cannot be built — never passed over silently."""


def mesh_samples(V: np.ndarray, F: np.ndarray, step: float) -> np.ndarray:
    """Mesh surface samples — vertices + area-proportional face samples (fixed seed) → one point per `step` voxel."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.floor(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1) / (step * step)).astype(int)
    idx = np.repeat(np.arange(len(F)), n)
    rng = np.random.default_rng(0)
    s, r = np.sqrt(rng.random(len(idx))), rng.random(len(idx))
    P = np.concatenate([V, (1 - s)[:, None] * a[idx] + (s * (1 - r))[:, None] * b[idx] + (s * r)[:, None] * c[idx]])
    _, keep = np.unique(np.floor(P / step).astype(np.int64), axis=0, return_index=True)
    return P[np.sort(keep)]


class RefHands:
    """The reference hands — both hands' link mesh geoms of the tree scene.xml (MJCF) under FK of the tree qpos
    (`q`; None = the tree's kinematic qpos file) (module "Hands"). Missing → raises."""

    def __init__(self, tree_dir, ti: dict, q=None):
        import mujoco  # noqa: PLC0415 — Isaac python · retarget venv

        tree_dir = Path(tree_dir)
        self.mj, self.m = mujoco, mujoco.MjModel.from_xml_path(str(tree_dir / "scene.xml"))
        m, d = self.m, mujoco.MjData(self.m)
        q = np.load(tree_dir / "0" / "trajectory_kinematic.npz")["qpos"] if q is None else np.asarray(q)
        if q.shape[1] != m.nq:
            raise RefHandsError(f"{tree_dir.name}: qpos width {q.shape[1]} ≠ scene.xml nq {m.nq}")
        span = {
            e["side"]: (int(e["start"]), int(e["start"]) + int(e["width"]))
            for e in ti.get("qpos_layout") or []
            if e.get("kind") == "hand"
        }
        if not span:
            raise RefHandsError(f"{tree_dir.name}: qpos_layout has no hand span")
        root = {}
        for bi in range(m.nbody):
            r = bi
            while r and int(m.body_parentid[r]) != 0:
                r = int(m.body_parentid[r])
            root[bi] = r
        side_of = {
            root[int(m.jnt_bodyid[j])]: s
            for j in range(m.njnt)
            for s, (a0, a1) in span.items()
            if a0 <= int(m.jnt_qposadr[j]) < a1
        }
        self.side = {bi: side_of.get(r, "") for bi, r in root.items()}
        self.body = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bi) for bi in range(m.nbody)]
        geoms, seen = [], set()
        for g in range(m.ngeom):
            bi = int(m.geom_bodyid[g])
            k = (bi, int(m.geom_dataid[g]), *np.round(m.geom_pos[g], 6), *np.round(m.geom_quat[g], 6))
            if self.side[bi] and int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_MESH and k not in seen:
                seen.add(k)
                geoms.append(g)
        if lost := sorted(set(span) - {self.side[int(m.geom_bodyid[g])] for g in geoms}):
            raise RefHandsError(f"{tree_dir.name}: scene.xml has no mesh geoms of hand {lost}")
        self.geoms = geoms  # the hand mesh geoms, in the order of every per-geom array below
        self.pts = [mesh_samples(*self._mesh(int(m.geom_dataid[g])), SAMPLE_M) for g in geoms]
        self.rad = np.array([np.linalg.norm(p, axis=1).max() for p in self.pts])
        self.gside = [self.side[int(m.geom_bodyid[g])] for g in geoms]
        self.gx = np.zeros((len(q), len(geoms), 3))
        self.gR = np.zeros((len(q), len(geoms), 3, 3))
        for k in range(len(q)):
            d.qpos[:] = q[k]
            mujoco.mj_kinematics(m, d)
            self.gx[k] = d.geom_xpos[geoms]
            self.gR[k] = d.geom_xmat[geoms].reshape(-1, 3, 3)

    def _mesh(self, mid: int):
        m = self.m
        va, fa = int(m.mesh_vertadr[mid]), int(m.mesh_faceadr[mid])
        return (
            np.asarray(m.mesh_vert[va : va + int(m.mesh_vertnum[mid])], float),
            np.asarray(m.mesh_face[fa : fa + int(m.mesh_facenum[mid])], int),
        )

    def object_points(self, name: str, pose) -> np.ndarray:
        """Samples of that object body's mesh geoms (helper group 3 excluded) — world points at the 7-pose (capture
        x·y·z · wxyz)."""
        from scipy.spatial.transform import Rotation as R  # noqa: PLC0415

        m = self.m
        gs = [
            g
            for g in range(m.ngeom)
            if self.body[int(m.geom_bodyid[g])] == name
            and int(m.geom_type[g]) == self.mj.mjtGeom.mjGEOM_MESH
            and int(m.geom_group[g]) != 3
        ]
        if not gs:
            raise RefHandsError(f"{name}: scene.xml has no mesh geom")
        out = []
        for g in gs:
            q = m.geom_quat[g]
            out.append(
                R.from_quat([q[1], q[2], q[3], q[0]]).apply(mesh_samples(*self._mesh(int(m.geom_dataid[g])), SAMPLE_M))
                + m.geom_pos[g]
            )
        p = np.asarray(pose, float)
        return R.from_quat([p[4], p[5], p[6], p[3]]).apply(np.concatenate(out)) + p[:3]

    def min_dist(self, P: np.ndarray) -> tuple[float, int, str]:
        """Clip-wide minimum distance (m) between both hands and points P · its frame · side. Branch and bound — a
        (frame, geom) whose lower bound (centre distance − radius) is at or above the current minimum is not measured
        (the result equals the exact minimum)."""
        from scipy.spatial import cKDTree  # noqa: PLC0415

        kd = cKDTree(P)
        T, G = self.gx.shape[:2]
        lb = kd.query(self.gx.reshape(-1, 3))[0].reshape(T, G) - self.rad[None, :]
        best = (float("inf"), -1, "")
        for flat in np.argsort(lb, axis=None):
            k, j = divmod(int(flat), G)
            if lb[k, j] >= best[0] or best[0] <= SAMPLE_M:
                break
            dd = float(kd.query(self.pts[j] @ self.gR[k, j].T + self.gx[k, j])[0].min())
            if dd < best[0]:
                best = (dd, k, self.gside[j])
        return best
