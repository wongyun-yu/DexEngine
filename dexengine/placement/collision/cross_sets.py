#!/usr/bin/env python3
"""Signed distances of the canon collision sets (①~④) — the case check, the bake (cross-repair), the gates and the audit
all use **the same function**.

The sets:
  ① crossed hand·arm ↔ hand·arm (classes hand-hand / hand-arm / arm-hand / arm-arm)
  ② hand ↔ the opposite side's object (Rhand-Lobj / Lhand-Robj)
  ③ object ↔ object
  ④ arm·hand ↔ torso (pairs adjacent within 3 generations of URDF parent-child ancestry are excluded)
Shape = arm and torso links: the colliders the sim spawns (sim_collider — vendor capsules · revision hulls, the vertices
  the robot USD generator writes) · hand, base and head links: the convex hull of the URDF <visual> mesh (geom
  "visual", the default; a visual mesh is not a sim collider). rby1 DAE is converted to STL by trimesh (pycollada) and
  cached. A URDF shape that cannot be built (missing mesh file · not capsule|mesh) dies (link_shapes). Objects
  (obj_geom, separate from geom): "sim" (default, every judge) = the PhysX-cooked hulls of the USD collision parts the
  sim spawns (the bake's dump — objects.hull.body_parts, scene.xml body ↔ USD link by shape) · "visual" = the tree
  scene.xml meshes (the render only). Distance = MuJoCo mj_geomDistance (convex GJK/EPA signed distance, negative =
  penetration depth). Frames: robot link world = pinocchio FK (merged URDF) + base_xyz (arm-IK npz). tree (combo tree
  qpos) → world = Rz(−90°)·p + (0,0,DESK_HEIGHT) (same as arm_ik). Hand links are placed by arm FK, but cross-repair
  previews a wrist target displacement as a "rigid translation of the hand links" (wrist_shift).

Use: L = HandLayout(tree_dir); m = CrossSetModel(L, tree_dir, assets_root); f, o = L.hand_blocks(qpos[t]);
m.set_pose(...); mins, hand_mins, rows = m.distances(0.03)
  mins[set] = the minimum signed distance (m) over **all pairs** of that set · hand_mins[set] = the minimum over **only
    the pairs with no arm link** (= the subset cross-repair can actually move; see `is_hand_pair`) · rows = [(set,
    class, linkA, linkB, distance, fromto6)].
HandLayout lives in hand_layout, the shapes and frame helpers in link_shapes; both are re-exported here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dexengine.placement.collision import det_math as D
from dexengine.placement.collision.cross_sets_build import CrossSetBuild
from dexengine.placement.collision.det_fk import DetFK
from dexengine.placement.collision.hand_layout import HandLayout, _hand_spec_for  # noqa: F401 — re-exported
from dexengine.placement.collision.link_shapes import (  # noqa: F401 — re-exported
    MESH_CACHE,
    RZ_NEG90,
    RZ_POS90,
    _adjacent,
    _mat_from_quat_wxyz,
    _urdf_once,
    artic_part_pose,
    desk_height,
    group_of,
    load_urdf_geometry,
    mat2quat_wxyz_batch,
    quat_mul_wxyz,
    xml_links,
)

_PF_SAID = False
_QRZ_NEG90 = mat2quat_wxyz_batch(RZ_NEG90[None])[0]  # Rz(−90°) as a quaternion — the pure (no BLAS) conversion
SETS = ("1_cross", "2_hand_obj", "3_obj_obj", "4_torso")
OBJ_GEOMS = ("sim", "visual")  # object shapes: PhysX-cooked hulls of the USD parts | scene.xml (render)


def is_hand_pair(cls: str) -> bool:
    """Is this a pair cross-repair can actually open up = a pair with **no arm link at either end** (hand-hand ·
    hand↔opposite object · hand↔torso).

    The degrees of freedom of this step are only the 6D wrist target and finger abduction, so it moves hand links only.
    A pair with an arm in it (arm-arm/arm-hand/arm-torso) is decided by the arm joints (IK decides it, there is no
    direct arm-joint displacement), and this step moves no arm joint.
    Putting such a pair into the objective does not resolve it even with
    the wrist pushed to its cap (measured: a 57mm hand~arm_5 hull), and it ruins the hand-hand penetrations that could
    have been fixed. So pairs with an arm in them are **reported only** (`arm_only_max_depth_mm`)."""
    return "arm" not in cls


class CrossSetModel(CrossSetBuild):
    """The canon collision-set evaluator for one clip (tree). The URDF is built 1 time per process, the objects once
    per tree."""

    def __init__(
        self,
        layout: HandLayout,
        tree_dir: Path | str | None = None,
        assets_root: Path | str | None = None,
        geom: str = "visual",
        desk_h: float | None = None,
        robust_zero: bool = True,
        obj_geom: str = "sim",
    ):
        import mujoco
        import pinocchio as pin

        if obj_geom not in OBJ_GEOMS:
            raise ValueError(f"obj_geom={obj_geom!r} — only {OBJ_GEOMS} are known")
        self.mj, self.pin, self.obj_geom = mujoco, pin, obj_geom
        # mj_geomDistance's GJK degeneracy returns **exactly 0.0** for some mesh-mesh poses (0.96~1.58% of hand-pair
        # rows; a fake "contact" that can hide a real penetration). robust_zero=True works around it by re-querying
        # those pairs with the geom1 body moved 0.1 mm in x (resolves 75~92%; the rest stay 0.0; see zero_requeried).
        self.robust_zero = bool(robust_zero)
        self.zero_requeried = [0, 0]  # (0 pairs requeried, of them now not 0)
        self.layout = layout
        self.urdf = Path(layout.urdf)
        from dexengine.core.arms import ARM_REGISTRY
        from dexengine.robots import urdf_load

        self.arm_spec = ARM_REGISTRY[layout.arm]  # the link groups and the EE name (ArmSpec)
        self.model = _urdf_once("pin", self.urdf, "", lambda: urdf_load.build_model(str(self.urdf)))
        self.data = self.model.createData()
        # The FK of every pose this class computes (link frames · EE · arm Jacobian) is DetFK — pinocchio parses the
        # URDF, but its FK takes glibc sincos, which differs between hosts (det_fk); the ruler amplifies that ulp.
        self.fk = _urdf_once("detfk", self.urdf, "", lambda: DetFK(self.model))
        for side in ("right", "left"):  # EE frame·finger joints must exist in this URDF
            ee = self.arm_spec.ee_body_name(side)
            if not self.model.existFrame(ee):
                raise ValueError(f"{self.urdf.name}: no EE frame {ee} (ArmSpec {layout.arm}) — arm_dq/hand_frame_pos")
            jn_all = {self.model.names[j] for j in range(1, self.model.njoints)}
            missing = [n for n in layout.finger_names[side] if n not in jn_all]
            if missing:
                raise ValueError(
                    f"{self.urdf.name}: spec {layout.hand} finger joints missing from the URDF: {missing[:4]}…"
                )
        self.jq = {self.model.names[j]: self.model.joints[j].idx_q for j in range(1, self.model.njoints)}
        self.jv = {self.model.names[j]: self.model.joints[j].idx_v for j in range(1, self.model.njoints)}
        self.links, self.parent = _urdf_once("shapes", self.urdf, geom, lambda: load_urdf_geometry(self.urdf, geom))
        self.frame_ids = {ln: self.model.getFrameId(ln) for ln in self.links}
        if desk_h is None:  # tree → its z offset (owner coords.scene_z_offset) · no tree (judge HULL) → the constant
            from dexengine.placement.coords import core as _coords

            desk_h = _coords.scene_z_offset(tree_dir) if tree_dir is not None else desk_height()
        self.desk_h = float(desk_h)
        self.assets = Path(assets_root) if assets_root else None
        self.n_obj: dict[str, int] = {}
        self.parts: list[dict] = []  # articulated parts — set_pose = parent pose ∘ joint FK
        self.part_parent: dict[str, str] = {}
        self._build(Path(tree_dir) if tree_dir else None)

    # ── Loading a pose ───────────────────────────────────────────────────────
    def build_q(self, arm_q: dict, finger_q: dict):
        """The full pinocchio joint configuration q — arm_q[side]=(names, values[7]) · finger_q[side]=nF (HandSpec
        action order). Values with a leading batch axis ((B,7) · (B,nF)) give q (B, nq)."""
        # The same values at the same indices as the scalar loop (q[jq[jn]] = vals[k]) — only the name lookups and
        # neutral are done once (bit-equal).
        if not hasattr(self, "_bq_cache"):
            self._bq_cache = {}
            self._q_neutral = np.asarray(self.pin.neutral(self.model), float)
        lead = np.shape(finger_q["right"])[:-1]  # () for one pose · (B,) for a batch of B poses
        q = np.broadcast_to(self._q_neutral, lead + self._q_neutral.shape).copy()
        for side in ("right", "left"):
            names, vals = arm_q[side]
            key = (side, tuple(names))
            idx = self._bq_cache.get(key)
            if idx is None:
                idx = self._bq_cache[key] = np.array([self.jq[jn] for jn in names], int)
            q[..., idx] = np.asarray(vals)[..., : len(idx)]
            fk = ("f", side)
            fidx = self._bq_cache.get(fk)
            if fidx is None:
                fidx = self._bq_cache[fk] = np.array([self.jq[jn] for jn in self.layout.finger_names[side]], int)
            if np.shape(finger_q[side])[-1] != len(fidx):
                raise ValueError(f"finger_q[{side}] length {np.shape(finger_q[side])[-1]} ≠ nF {len(fidx)}")
            q[..., fidx] = np.asarray(finger_q[side])
        return q

    def _ee_fid(self, side: str) -> int:
        c = self.__dict__.setdefault("_ee_fid_cache", {})
        if side not in c:
            c[side] = self.model.getFrameId(self.arm_spec.ee_body_name(side))
        return c[side]

    def hand_frame_pos(self, q, side: str, base_xyz) -> np.ndarray:
        """The world position of that side's EE (ArmSpec.ee_body_name) — q (nq,) or (B, nq) → (3,) or (B, 3).
        cross-repair uses it to measure "how far the wrist actually moved"."""
        return self.fk.frame_pos(q, self._ee_fid(side)) + np.asarray(base_xyz, float)

    def arm_dq(self, q, side: str, arm_names, dp_world, lam: float = 0.03) -> np.ndarray:
        """The increment Δq of **that arm's 7 joints** that moves the hand (EE) by dp_world in the world — 1
        damped-least-squares step.

        After moving the wrist target, cross-repair **does not re-solve** the arm IK: a full DP re-solve makes the
        candidate search run again and leaves the valley the case check had passed (measured: 5 of 8 clips checked hit
        the tracking-error p99 gate even with no repair), which kills the bake outright. Instead it does only a **local
        correction around** the fixed solution — the displacement is ≤20mm, so 1~2 steps converge.
        Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ dp with det_math arithmetic (J from DetFK) · q (nq,) or (B, nq) · dp (3,) or (B, 3)."""
        J = self.fk.frame_jac_lin(q, self._ee_fid(side), [self.model.getJointId(n) for n in arm_names])  # (...,3,k)
        A = np.zeros(J.shape[:-2] + (3, 3))
        for i in range(3):
            for j in range(3):
                s = J[..., i, 0] * J[..., j, 0]
                for c in range(1, J.shape[-1]):
                    s = s + J[..., i, c] * J[..., j, c]
                A[..., i, j] = s + (lam * lam if i == j else 0.0)
        x = D.solve3(A, np.asarray(dp_world, float))
        return (J[..., 0, :] * x[..., 0, None] + J[..., 1, :] * x[..., 1, None]) + J[..., 2, :] * x[..., 2, None]

    def mocap_pose(
        self, arm_q: dict, finger_q: dict, obj_tree: dict, base_xyz, wrist_shift: dict | None = None, prev=None
    ):
        """The mocap (pos (nmocap,3), quat (nmocap,4)) set_pose writes to MjData — **the single owner of the pose load
        formula** (with mocap_from_links): set_pose (one pose, CPU consumers) and xr_gpu.PoseBatch.geoms (batches)
        both call it. prev = the values of the mocap bodies this call does not write (default: the current MjData
        values — set_pose's meaning: a side without an object block keeps its previous value). Link frames come from
        DetFK, quaternions from mat2quat_wxyz_batch (pure arithmetic) in one call."""
        R, p = self.fk.frames(self.build_q(arm_q, finger_q), self._lm_fid)
        return self.mocap_from_links(R, p, obj_tree, base_xyz, wrist_shift, prev)

    def mocap_from_links(self, R, p, obj_tree: dict, base_xyz, wrist_shift: dict | None = None, prev=None):
        """The rest of mocap_pose's formula after the link FK — R (nl,3,3) · p (nl,3) = the DetFK frames of _lm_fid
        (the batch path computes them for B poses in one call, then calls this per pose). det_math arithmetic: no BLAS
        and no libm, so the bits are the same on every host."""
        pos = np.array(self.d.mocap_pos if prev is None else prev[0], float)
        quat = np.array(self.d.mocap_quat if prev is None else prev[1], float)
        P = np.asarray(p, float) + np.asarray(base_xyz, float)
        if wrist_shift is not None:
            if wrist_shift.get("right") is not None:
                P[self._lm_rh] = P[self._lm_rh] + wrist_shift["right"]
            if wrist_shift.get("left") is not None:
                P[self._lm_lh] = P[self._lm_lh] + wrist_shift["left"]
        pos[self._lm_mi] = P
        quat[self._lm_mi] = mat2quat_wxyz_batch(R)
        desk = np.array([0.0, 0.0, self.desk_h])
        qrz = _QRZ_NEG90
        theta_of: dict[str, float] = {}  # parent body → joint value (8th from hand_blocks)
        for s in ("right", "left"):
            mi = self.obj_mocap.get(s)
            blk = obj_tree.get(s) if obj_tree else None
            if mi is None or blk is None:
                continue
            pos[mi] = D.matvec3(RZ_NEG90, np.asarray(blk[:3], float)) + desk
            qn = np.asarray(blk[3:7], float)
            qn = qn / max(float(D.norm(qn)), 1e-9)
            quat[mi] = quat_mul_wxyz(qrz, qn)
            if len(blk) > 7:
                theta_of[self.layout.obj_body[s]] = float(blk[7])
        for part in self.parts:  # part = parent pose ∘ body offset ∘ joint(θ)
            if part["parent"] not in theta_of:
                raise ValueError(
                    f"part {part['name']}: no joint value (the obj_tree block has 7 columns) — pass the 8 "
                    "columns of hand_blocks(qpos_row)"
                )
            mp = self.body_mocap[part["parent"]]
            p_par = np.asarray(pos[mp])
            R_par = _mat_from_quat_wxyz(np.asarray(quat[mp]))
            p_c, R_c = artic_part_pose(p_par, R_par, part, theta_of[part["parent"]], det=True)
            mc = self.body_mocap[part["name"]]
            pos[mc] = p_c
            quat[mc] = mat2quat_wxyz_batch(np.asarray(R_c)[None])[0]
        return pos, quat

    def set_pose(self, arm_q: dict, finger_q: dict, obj_tree: dict, base_xyz, wrist_shift: dict | None = None):
        """arm_q[side]=(names, values[7]) · finger_q[side]=nF (HandSpec order) · obj_tree[side]=qpos 7 (tree frame, None
        when absent)
        · wrist_shift[side] = a rigid world translation (m) of that whole hand's links — an approximate preview that
          leaves the arm alone (for diagnostics). The canon cross-repair actually moves the arm joints with `arm_dq`, so
          it does not use wrist_shift.
        The formula's owner is mocap_pose — this writes MjData and runs mj_kinematics only."""
        pos, quat = self.mocap_pose(arm_q, finger_q, obj_tree, base_xyz, wrist_shift)
        self.d.mocap_pos[:] = pos
        self.d.mocap_quat[:] = quat
        self.mj.mj_kinematics(self.m, self.d)

    # ── Distances ────────────────────────────────────────────────────────────
    def _requery_zero(self, gi: int, gj: int, distmax: float) -> float:
        """Re-query a geom pair that came out exactly 0.0, after perturbing the geom1 body by 0.1mm in x. If
        the re-query is 0 too, return 0 as-is."""
        mj = self.mj
        mi = self.m.body_mocapid[self.m.geom_bodyid[gi]]
        self.zero_requeried[0] += 1
        if mi < 0:
            return 0.0
        p0 = self.d.mocap_pos[mi].copy()
        self.d.mocap_pos[mi] = p0 + np.array([1e-4, 0.0, 0.0])
        mj.mj_kinematics(self.m, self.d)
        d2 = mj.mj_geomDistance(self.m, self.d, gi, gj, distmax, self._fromto)
        self.d.mocap_pos[mi] = p0
        mj.mj_kinematics(self.m, self.d)
        if d2 != 0.0:
            self.zero_requeried[1] += 1
            return float(d2)
        # The fromto of the perturbed state is left behind in _fromto — when we keep the 0, re-measure at the original
        # pose to restore fromto
        mj.mj_geomDistance(self.m, self.d, gi, gj, distmax, self._fromto)
        return 0.0

    def distances(self, distmax: float = 0.03, sets=SETS, want_pairs: bool = True):
        """The per-set minimum signed distance of the current pose.

        Returns (mins, hand_mins, rows)
          mins[set]      = the minimum (m) over **all pairs** of that set. The value the audit and the gates look at.
          hand_mins[set] = the minimum (m) over **only the pairs without an arm link** in that set — cross-repair's
            objective.
          rows           = the pairs with distance < distmax, [(set, class, linkA, linkB, distance, fromto6)] (only when
            want_pairs).
        If no contact at all is within distmax, both minima are distmax."""
        mj = self.mj
        out = dict.fromkeys(sets, distmax)
        hout = dict.fromkeys(sets, distmax)
        rows = []
        # Bounding-sphere prefilter: a pair with |p_i − p_j| > r_i + r_j + distmax is not queried (bit-equal output) —
        # geom_rbound = radius about the geom's local origin · geom_xpos = that origin in world.
        if getattr(self, "_pf", None) is None:
            import numpy as _np

            gi_a = _np.array([q[0] for q in self.pairs])
            gj_a = _np.array([q[1] for q in self.pairs])
            rb = _np.asarray(self.m.geom_rbound, float)
            self._pf = (gi_a, gj_a, rb[gi_a] + rb[gj_a])
        import numpy as _np

        _gi, _gj, _rs = self._pf
        _xp = _np.asarray(self.d.geom_xpos, float)
        _d2 = _np.sum((_xp[_gi] - _xp[_gj]) ** 2, axis=1)
        _keep = _d2 <= (_rs + distmax) ** 2
        global _PF_SAID
        if not _PF_SAID:
            _PF_SAID = True
            print(
                f"[XR_PREFILTER] pairs={len(self.pairs)} kept={int(_keep.sum())} "
                "(bounding-sphere prefilter, owner cross_sets)",
                flush=True,
            )
        for _k, (gi, gj, sname, cls) in enumerate(self.pairs):
            if sname not in out or not _keep[_k]:
                continue
            dist = mj.mj_geomDistance(self.m, self.d, gi, gj, distmax, self._fromto)
            if dist == 0.0 and self.robust_zero:
                dist = self._requery_zero(gi, gj, distmax)
            if dist >= distmax:
                continue
            if dist < out[sname]:
                out[sname] = dist
            if dist < hout[sname] and is_hand_pair(cls):
                hout[sname] = dist
            if want_pairs:
                rows.append((sname, cls, self.ginfo[gi][1], self.ginfo[gj][1], float(dist), self._fromto.copy()))
        return out, hout, rows


def load_ik(path: Path):
    """An arm-IK npz → (arm joint names, joint positions (T, 7), base_xyz)."""
    w = np.load(path, allow_pickle=True)
    return list(w["arm_joint_names"]), np.asarray(w["arm_joint_positions_all"], float), np.asarray(w["base_xyz"], float)
