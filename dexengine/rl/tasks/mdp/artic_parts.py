"""Part geometry of articulated objects: two single owners.

1. `joint_kind_span(env, object_key)`: per env, the kind (slide or not) and travel (the width of the physical joint
   limits: hinge rad, slide m) of the **spawned** joint. Asked of the asset the scene loaded, never guessed from names
   or paths: kind = the schema of the joint prim under that env's object prim (UsdPhysics Prismatic | Revolute; the env
   number is read from the `env_<i>` path, not from sort order), travel = the `joint_pos_limits` width (the bake wrote
   the spec's dexengine.data.artic.ranges lo and hi). The PhysX shared meta type (the first env's) is not used: it is
   wrong when hinge and slide spawns mix.
2. `SurfaceParts`: loads the part ids, joint axis, anchor, merge angle and kind of the surface-point files
   (`<asset surface_dir>/surface_points.npz` — the usd_artic layer, generator dexengine/data/artic/surface.py) as
   per-clip tables, and places the moving part's points and normals at the current θ (`pose`):
     hinge  p' = p + (R(axis, θ−θ0) − I)(p − a)      slide  p' = p + axis·(θ−θ0)       (normals rotate only)
   Written as displacements, so θ = θ0 is bit-identical with the input. The file's axis and anchor are in the same
   **base frame** as the point cloud (the generator applies child_pose to the spec's child-frame axis and anchor; the FK
   convention's owner is placement coords.artic_moving_vertices). An articulated object without part_id (a rigid point
   cloud) or with a point count other than the observed cloud (a DEXENGINE_SURF_POINTS subset) fails.

Depends on torch and numpy (pxr and isaaclab only inside joint_kind_span; the rest is tested without Isaac).
"""

from __future__ import annotations

import os
import re

import numpy as np

_SURFACE_FILE = "surface_points.npz"
_KINDS = ("hinge", "slide")
_BASE, _MOVING = (
    0,
    1,
)  # part_id codes: the values the generator (surface.py build_one) writes (base first, moving after)


def _is_slide(root_prim, where: str) -> bool:
    """Kind (slide or not) of the one joint prim under `root_prim`: the only place that reads a joint kind from USD.
    A joint count other than 1 fails."""
    from pxr import Usd, UsdPhysics

    joints = [
        q for q in Usd.PrimRange(root_prim) if q.IsA(UsdPhysics.RevoluteJoint) or q.IsA(UsdPhysics.PrismaticJoint)
    ]
    if len(joints) != 1:
        raise RuntimeError(
            f"[ARTIC][kind] {where}: {len(joints)} joints — must be 1 (objects.usd_bake --articulated contract)"
        )
    return bool(joints[0].IsA(UsdPhysics.PrismaticJoint))


def require_one_kind(usd_paths: list[str], label: str) -> str:
    """Whether the USDs of one object slot (MultiAssetSpawner) share one joint kind. Mixed hinge and slide leave PhysX
    without a shared meta type and Isaac Lab dies at the physics view stage with an obscure error (mixed-spawn smoke
    measured); this fails earlier with the reason. Returns that kind (hinge | slide). A mixed bundle must be split by
    joint kind (a job of the run bundle axis)."""
    from pxr import Usd

    kinds: dict = {}
    for u in usd_paths:
        st = Usd.Stage.Open(u)  # the prims live only while the stage is held
        root = st.GetDefaultPrim() if st else None
        if not root:
            raise RuntimeError(f"[ARTIC][kind] {label}: could not open {u} or it has no default prim")
        kinds.setdefault(_KINDS[_is_slide(root, u)], []).append(u)
    if len(kinds) != 1:
        cnt = ", ".join(f"{k}: {len(v)}" for k, v in kinds.items())
        raise ValueError(
            f"[ARTIC][kind] {label}: mixed joint kinds ({cnt}) — hinge and slide cannot share one physics "
            "view (no PhysX shared meta type). Split the bundle by joint kind"
        )
    return next(iter(kinds))


def joint_kind_span(env, object_key: str):
    """(is_slide (N,) bool, span (N,) float): per env, the kind and limit width of the spawned joint.
    A joint count other than 1, a width that is not positive and finite, or a missing env fails."""
    import torch

    from isaaclab.sim.utils import find_matching_prims

    obj = env.scene[object_key]
    n = int(env.num_envs)
    kinds: list = [None] * n
    for prim in find_matching_prims(obj.cfg.prim_path):
        m = re.search(r"/env_(\d+)/", str(prim.GetPath()) + "/")
        if m is None or int(m.group(1)) >= n:
            raise RuntimeError(
                f"[ARTIC][kind] {object_key}: object prim without a readable env number {prim.GetPath()}"
            )
        kinds[int(m.group(1))] = _is_slide(prim, f"{object_key} {prim.GetPath()}")
    missing = [i for i, k in enumerate(kinds) if k is None]
    if missing:
        raise RuntimeError(
            f"[ARTIC][kind] {object_key}: no joint prim found in {len(missing)}/{n} envs (e.g. {missing[:5]})"
        )
    lim = obj.data.joint_pos_limits[:, 0, :]
    span = (lim[:, 1] - lim[:, 0]).float()
    bad = ~(
        (span > 0) & torch.isfinite(span)
    )  # a joint without limits (inf width) makes the error 0: judge and reward disarmed
    if bool(bad.any()):
        raise RuntimeError(
            f"[ARTIC][kind] {object_key}: {int(bad.sum())} envs with a joint limit width that is not "
            "positive and finite — travel normalisation is impossible (check the bake spec lo and hi)"
        )
    return torch.tensor(kinds, dtype=torch.bool, device=span.device), span


class SurfaceParts:
    """Per-clip articulated surface part table: moving (C,P) · axis (C,3) · anchor (C,3) · theta0 (C,) · slide (C,)."""

    def __init__(self, moving, axis, anchor, theta0, slide):
        self.moving, self.axis, self.anchor, self.theta0, self.slide = moving, axis, anchor, theta0, slide

    @classmethod
    def load(cls, mesh_dirs: list[str], n_points: list[int], device) -> SurfaceParts:
        """Load the part tables from the surface-point file of each of `mesh_dirs` (loader clip order).
        n_points = the length of the point cloud the observation uses (per clip)."""
        import torch

        if len(mesh_dirs) != len(n_points):
            raise ValueError(f"[ARTIC][surface] {len(mesh_dirs)} mesh folders ≠ {len(n_points)} point clouds")
        c, p_max = len(mesh_dirs), max(n_points)
        moving = torch.zeros(c, p_max, dtype=torch.bool)
        axis, anchor = torch.zeros(c, 3), torch.zeros(c, 3)
        theta0, slide = torch.zeros(c), torch.zeros(c, dtype=torch.bool)
        for i, (d, n_pt) in enumerate(zip(mesh_dirs, n_points, strict=True)):
            f = os.path.join(d, _SURFACE_FILE)
            z = np.load(f)
            if "part_id" not in z.files:
                raise ValueError(
                    f"[ARTIC][surface] {f}: no part_id — a rigid point cloud for an articulated object "
                    "(the moving part drops out of the observation). It must come from dexengine/data/artic/surface.py"
                )
            pid = np.asarray(z["part_id"])
            if pid.shape[0] != n_pt:
                raise ValueError(
                    f"[ARTIC][surface] {f}: part_id has {pid.shape[0]} points ≠ observed cloud {n_pt} points — "
                    "a subset (DEXENGINE_SURF_POINTS) or a file that does not pair"
                )
            vals = sorted(int(v) for v in np.unique(pid))
            if vals != [_BASE, _MOVING]:
                raise ValueError(
                    f"[ARTIC][surface] {f}: part_id values {vals} — both base {_BASE} and moving {_MOVING} "
                    "are needed (an unknown value or a one-part cloud leaves no part to rotate)"
                )
            kind = str(z["joint_type"])
            if kind not in _KINDS:
                raise ValueError(f"[ARTIC][surface] {f}: joint kind {kind!r} — only {_KINDS} are known")
            ax = np.asarray(z["joint_axis"], np.float64)
            moving[i, :n_pt] = torch.from_numpy(pid == _MOVING)
            axis[i] = torch.from_numpy(ax / np.linalg.norm(ax))
            anchor[i] = torch.from_numpy(np.asarray(z["joint_anchor"], np.float64))
            theta0[i] = float(z["merge_theta"])
            slide[i] = kind == "slide"
        return cls(*(t.to(device) for t in (moving, axis, anchor, theta0, slide)))

    def rows(self, clip, n: int):
        """Table row per env: the clip id with multiple objects, otherwise 0 (one object)."""
        import torch

        if clip is not None:
            return clip
        if self.moving.shape[0] != 1:
            raise ValueError(f"[ARTIC][surface] the clip table has {self.moving.shape[0]} rows but no clip ids")
        return torch.zeros(n, dtype=torch.long, device=self.moving.device)

    def check_kinds(self, env_is_slide, clip) -> None:
        """Compare the surface-point file's joint kind (statement) with the spawned joint (measure, joint_kind_span) per
        env; a mismatch fails."""
        rows = self.rows(clip, int(env_is_slide.shape[0]))
        bad = self.slide[rows] != env_is_slide
        if bool(bad.any()):
            raise RuntimeError(
                f"[ARTIC][surface] surface-point file joint kind ≠ spawned joint "
                f"{int(bad.sum())}/{bad.numel()} envs — the surface points and the USD come from different specs"
            )

    def pose(self, clip, theta, pts, nrm):
        """Place the moving part's points and normals at θ. theta (N,), pts/nrm (N,P,3) (expanded views allowed) -> two
        new (N,P,3)."""
        import torch

        n = int(theta.shape[0])
        r = self.rows(clip, n)
        mov = self.moving[r][:, : pts.shape[1]].unsqueeze(-1)
        k = self.axis[r].unsqueeze(1)
        d = (theta.float() - self.theta0[r]).view(n, 1, 1)
        sl = self.slide[r].view(n, 1, 1)
        c, s = torch.cos(d), torch.sin(d)

        def _disp(v):  # (R − I) v: exactly 0 when θ = θ0
            kxv = torch.cross(k.expand_as(v), v, dim=-1)
            return (c - 1.0) * v + s * kxv + (1.0 - c) * k * (v * k).sum(-1, keepdim=True)

        dp = torch.where(sl, k * d, _disp(pts - self.anchor[r].unsqueeze(1)))
        dn = torch.where(sl, torch.zeros_like(nrm), _disp(nrm))
        return pts + torch.where(mov, dp, torch.zeros_like(dp)), nrm + torch.where(mov, dn, torch.zeros_like(dn))
