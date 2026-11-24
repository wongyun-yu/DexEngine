"""Object state observation (ObjectStateV2, 87 dims + the articulation tail) and the idle-side zero observation
(split from observations.v2, which keeps the hand observation and the rotation helpers)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, ObservationTermCfg
from isaaclab.utils.math import axis_angle_from_quat, matrix_from_quat, quat_apply, quat_conjugate, quat_mul

from dexengine.rl.tasks.mdp import physx_props
from dexengine.rl.tasks.mdp.naming import resolve_link7_to_wrist_offset
from dexengine.rl.tasks.mdp.observations.v2 import REF_FPS, q_rot_vecs, rot6d_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


#: Articulation tail of ObjectStateV2 (rule D1-k). Default layout: an articulated object appends
#: (θ, ref−θ) = ARTIC_TAIL_DIMS and a rigid one appends nothing (87 vs 89). Fixed layout (params
#: fixed_layout=1): EVERY object appends (θ, ref−θ, valid) = FIXED_TAIL_DIMS, a rigid object as 0, 0, 0 —
#: one width for rigid and articulated clips. The token layer (rl/agents/models._object_extra_dims)
#: checks the live width against these two counts.
ARTIC_TAIL_DIMS = 2


FIXED_TAIL_DIMS = 3


#: ObjectStateV2 body without a tail: kin 15 + ref 18 + surface 35 + priv 19 (5 tips).
OBJECT_STATE_DIMS = 87


def idle_object_obs(env, width: int = OBJECT_STATE_DIMS) -> torch.Tensor:
    """D16 idle side: the object observation of a side that has no object — zeros of that layout's width
    (OBJECT_STATE_DIMS, + FIXED_TAIL_DIMS under the A-5 fixed layout). It replaces the side's ObjectStateV2 term under
    the SAME term name, so the policy-obs order, the two-tower split and the token slices stay what they are."""
    return torch.zeros((env.num_envs, int(width)), device=env.device)


class ObjectStateV2(ManagerTermBase):
    """87-dim object state (kinematics + ref tracking + surface geometry + privileged).

    Everything in the VIRTUAL-WRIST frame of ``side``'s hand. Merges V1's
    the removed V1 terms (ObjectPrivileged 40 + ObjectTarget 169 + surface 15) without the
    duplicates/dead dims, with the ref-ang-vel and mass fixes.

    Layout:
      object kinematics (15):
        obj_pos (3, wrist frame, wrist→obj), obj_rot6d (6, wrist-relative),
        obj_vel (3), obj_ang_vel (3)
      reference tracking (18):
        delta_obj_pos (3), delta_obj_rot6d (6, cur→ref), delta_obj_vel (3),
        ref_obj_ang_vel (3, finite-diff of ref quats), delta_obj_ang_vel (3)
      contact geometry, per tip (35):
        surface_vec (5×3, tip→nearest surface point), surface_normal (5×3),
        signed_distance (5)
      privileged (19):
        tip_force (5×3, contact-sensor net force, wrist frame),
        friction_static/dynamic + restitution (3), mass (1, DR-true)
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        from dexengine.rl.tasks.mdp.naming import resolve_palm_body_id, resolve_tip_body_names
        from dexengine.rl.tasks.mdp.surface import load_object_surface_pointcloud

        self._side = cfg.params.get("side", "right")
        self._object_key = cfg.params.get("object_key", "object_right")
        mesh_dir = cfg.params.get("mesh_dir", None)
        if mesh_dir is None:
            raise ValueError(
                f"[ObjectStateV2:{self._side}] params['mesh_dir'] required (directory containing surface_points.npz)"
            )
        self._traj_mgr = env.trajectory_manager

        hand = env.scene[f"{self._side}_hand"]
        self._palm_body_id = resolve_palm_body_id(hand, self._side)
        palm_name = hand.data.body_names[self._palm_body_id]
        from dexengine.rl.tasks.mdp.naming import is_arm_ee_body as _is_ee

        if not _is_ee(palm_name):
            raise RuntimeError(
                f"[ObjectStateV2:{self._side}] full-robot mode required (palm body is "
                f"'{palm_name}', expected the active arm's EE link)."
            )
        from dexengine.rl.tasks.mdp.naming import resolve_ee_to_wrist_rot as _rot_pair  # side-aware pair

        self._link7_to_wrist_rot = torch.tensor(list(_rot_pair(hand, self._side)), device=env.device)
        self._link7_to_wrist_trans = torch.tensor(
            list(resolve_link7_to_wrist_offset(hand, self._side)), device=env.device
        )

        self._fingertip_body_ids(cfg, env, hand, load_object_surface_pointcloud, mesh_dir, resolve_tip_body_names)
        if self._fixed_layout:
            print(
                f"[ObjectStateV2:{self._side}] [OBJ_OBS_FIXED] A-5 fixed layout — 87 + {FIXED_TAIL_DIMS} "
                f"(θ · ref−θ · valid bit={int(self._artic_dims)}; rigid objects 0·0·0)",
                flush=True,
            )
        _tail = FIXED_TAIL_DIMS if self._fixed_layout else (ARTIC_TAIL_DIMS if self._artic_dims else 0)
        print(
            f"[ObjectStateV2:{self._side}] {87 + _tail}-dim "
            f"(kin 15 + ref 18 + surface 35 + priv 19{f' + artic {_tail}' if _tail else ''}); "
            f"object_key='{self._object_key}', N_surface_pts={self._surface_pts_local.shape[0]}, "
            f"frame=wrist, rot=6D, mass=DR-true"
        )

    def _fingertip_body_ids(self, cfg, env, hand, load_object_surface_pointcloud, mesh_dir, resolve_tip_body_names):
        """Fingertip body IDs (articulation)."""
        tip_names = resolve_tip_body_names(hand, self._side)[:5]
        self._tip_body_ids = []
        for name in tip_names:
            try:
                self._tip_body_ids.append(hand.data.body_names.index(name))
            except ValueError:  # noqa: PERF203 — each tip body is looked up alone; a missing one is printed
                print(f"[ObjectStateV2:{self._side}] WARNING: tip body '{name}' not found")
                self._tip_body_ids.append(0)
        self._tip_body_ids = torch.tensor(self._tip_body_ids, device=env.device, dtype=torch.long)
        self._tip_names = tip_names
        self._contact_tip_ids = None  # lazy (sensor body list not final at init)

        # Pre-sampled surface point cloud + normals (object-local). --multi_object:
        # mesh_dir is a comma-list in LOADER clip order — load every cloud, pad to the
        # max point count (points at +1e6 so a pad is never the nearest; unit-x normals),
        # gather per env by clip id at call time.
        _dirs = [d for d in str(mesh_dir).split(",") if d]
        if not _dirs:
            raise ValueError(f"[ObjectStateV2:{self._side}] params['mesh_dir'] is empty")
        self._surface_pts_multi = None
        self._surface_normals_multi = None

        _clouds = [load_object_surface_pointcloud(d, env.device) for d in _dirs]

        if len(_clouds) > 1:
            _maxp = max(p.shape[0] for p, _ in _clouds)
            _C = len(_clouds)
            _pts = torch.full((_C, _maxp, 3), 1.0e6, device=env.device)
            _nrm = torch.zeros((_C, _maxp, 3), device=env.device)
            _nrm[..., 0] = 1.0
            for _i, (_p, _n) in enumerate(_clouds):
                _pts[_i, : _p.shape[0]] = _p
                _nrm[_i, : _n.shape[0]] = _n
            self._surface_pts_multi, self._surface_normals_multi = _pts, _nrm
            self._surface_pts_local, self._surface_normals_local = _clouds[0]
        else:
            self._surface_pts_local, self._surface_normals_local = _clouds[0]

        # The nearest surface point comes from one torch.cdist (the old matmul-expansion switch is gone).

        # Whether the object is articulated is asked of **the asset the scene loaded**: with joints the joint count is
        # not 0. Guessing from names or paths would silently drop the 2 columns the day a name changes.
        _o = env.scene[self._object_key]
        self._artic_dims = int(getattr(_o, "num_joints", 0) or 0) > 0
        self._fixed_layout = int(cfg.params.get("fixed_layout", 0) or 0) == 1  # A-5 (default 0 = 87/89)
        # Articulated: place the moving part of the surface points at the current θ (owner artic_parts; rigid objects
        # have no such table and keep the path).
        self._artic_parts = None
        if self._artic_dims:
            print(
                f"[ObjectStateV2:{self._side}] articulated object — observation 87 → 89 (current angle + difference "
                f"to the reference). "
                f"num_joints={int(_o.num_joints)} (rule D1-k)",
                flush=True,
            )
            from dexengine.rl.tasks.mdp.artic_parts import SurfaceParts, joint_kind_span

            self._artic_parts = SurfaceParts.load(_dirs, [int(p.shape[0]) for p, _ in _clouds], env.device)
            _clip0 = self._traj_mgr.clip_ids if self._surface_pts_multi is not None else None
            self._artic_parts.check_kinds(joint_kind_span(env, self._object_key)[0], _clip0)
            print(
                f"[ARTIC][surface] {self._side}: moving-part surface points rotated by θ — clips {len(_dirs)} · "
                f"moving points "
                f"{int(self._artic_parts.moving.sum())} · slide clips {int(self._artic_parts.slide.sum())} "
                "(surface-point file kind = spawned joint kind checked)",
                flush=True,
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        side: str = "right",
        object_key: str = "object_right",
        mesh_dir: str | None = None,
        fixed_layout: int = 0,
    ) -> torch.Tensor:
        N = env.num_envs
        device = env.device
        hand = env.scene[f"{self._side}_hand"]
        obj = env.scene[self._object_key]
        origins = env.scene.env_origins

        out = self._virtual_wrist_pose(env, N, device, hand, obj, origins)

        return out

    def _virtual_wrist_pose(self, env, N, device, hand, obj, origins):
        """Virtual wrist pose (same math as HandObservationV2)."""
        link7_pos_w = hand.data.body_link_pos_w[:, self._palm_body_id, :]
        link7_quat_w = hand.data.body_link_quat_w[:, self._palm_body_id, :]
        r_offset = self._link7_to_wrist_rot.unsqueeze(0).expand(N, -1).contiguous()
        vw_quat = quat_mul(link7_quat_w, r_offset)
        link7_rot = matrix_from_quat(link7_quat_w)
        t_col = self._link7_to_wrist_trans.view(1, 3, 1).expand(N, -1, -1)
        vw_pos = link7_pos_w + torch.bmm(link7_rot, t_col).squeeze(-1)
        vw_conj = quat_conjugate(vw_quat)

        return self._object_kinematics_wrist_frame(env, N, device, hand, obj, origins, vw_conj, vw_pos)

    def _object_kinematics_wrist_frame(self, env, N, device, hand, obj, origins, vw_conj, vw_pos):
        """Object kinematics (15), wrist frame."""
        obj_pos = obj.data.root_pos_w - origins
        obj_quat = obj.data.root_quat_w
        obj_pos_wf = quat_apply(vw_conj, obj_pos - (vw_pos - origins))
        obj_rot6d_wf = rot6d_from_quat(quat_mul(vw_conj, obj_quat))
        obj_vel_wf = quat_apply(vw_conj, obj.data.root_lin_vel_w)
        obj_ang_vel_wf = quat_apply(vw_conj, obj.data.root_ang_vel_w)

        # ---- reference tracking (18), wrist frame ----
        ref_state = self._traj_mgr.get_reference_state(env.episode_length_buf)
        ref_state_next = self._traj_mgr.get_reference_state(env.episode_length_buf + 1)
        ref_obj_pos = ref_state[f"{self._side}_obj_pos"]
        ref_obj_quat = ref_state[f"{self._side}_obj_quat"]
        ref_obj_quat_next = ref_state_next[f"{self._side}_obj_quat"]
        ref_obj_vel = (ref_state_next[f"{self._side}_obj_pos"] - ref_obj_pos) * REF_FPS
        (
            delta_obj_ang_vel_wf,
            delta_obj_pos_wf,
            delta_obj_rot6d,
            delta_obj_vel_wf,
            mass_dev,
            mat_props_dev,
            ref_obj_ang_vel_wf,
            signed_distance,
            surface_nrm_wf,
            surface_vec_wf,
            tip_forces_wf,
        ) = self._reference_angular_velocity(
            env,
            N,
            device,
            hand,
            obj,
            obj_pos,
            obj_quat,
            origins,
            ref_obj_pos,
            ref_obj_quat,
            ref_obj_quat_next,
            ref_obj_vel,
            vw_conj,
        )

        out = torch.cat(
            [
                obj_pos_wf,
                obj_rot6d_wf,
                obj_vel_wf,
                obj_ang_vel_wf,  # 15
                delta_obj_pos_wf,
                delta_obj_rot6d,
                delta_obj_vel_wf,  # 12
                ref_obj_ang_vel_wf,
                delta_obj_ang_vel_wf,  # 6
                surface_vec_wf,
                surface_nrm_wf,
                signed_distance,  # 35
                tip_forces_wf,
                mat_props_dev,
                mass_dev,  # 19
            ],
            dim=-1,
        )  # (N, 87)

        return self._articulated_objects_rule_d1(N, obj, out, ref_state)

    def _articulated_objects_rule_d1(self, N, obj, out, ref_state):
        """Articulated objects (rule D1-k): the policy must see **how far the lid is open**. Two columns are appended,
        the current and the reference angle (87 → 89). Objects without joints get none, so rigid clip observations
        stay **byte-identical**. The asset in the scene decides (no guessing from names). Fixed layout
        (fixed_layout=1): every object carries the same 3-dim tail (θ, ref−θ, valid bit) — a rigid object as 0, 0, 0 —
        so rigid and articulated clips share one observation width (87+3).
        """
        if self._fixed_layout and not self._artic_dims:
            out = torch.cat([out, torch.zeros((N, FIXED_TAIL_DIMS), device=out.device, dtype=out.dtype)], dim=-1)
        if self._artic_dims:
            _q = obj.data.joint_pos[:, :1]  # (N,1) current angle
            # The reference joint angle is owned by the trajectory manager's reference state (the same frame as
            # ref_state above), from the same call as the object pose reference (rule D1-c). A missing key means a
            # loader without articulated references: fail loudly.
            if f"{self._side}_obj_arti" not in ref_state:
                raise KeyError(
                    f"[ObjectStateV2:{self._side}] articulated object but the reference state has no "
                    f"'{self._side}_obj_arti' — a tree/loader without the joint columns (rule D1)"
                )
            _rq = ref_state[f"{self._side}_obj_arti"].reshape(_q.shape[0], 1).to(_q.device)
            out = torch.cat([out, _q, _rq - _q], dim=-1)  # (N, 89)
            if self._fixed_layout:  # A-5: + valid bit 1 → (N, 90)
                out = torch.cat([out, torch.ones((N, 1), device=out.device, dtype=out.dtype)], dim=-1)
        return out

    def _reference_angular_velocity(
        self,
        env,
        N,
        device,
        hand,
        obj,
        obj_pos,
        obj_quat,
        origins,
        ref_obj_pos,
        ref_obj_quat,
        ref_obj_quat_next,
        ref_obj_vel,
        vw_conj,
    ):
        """Reference angular velocity via quat finite difference (V1 shipped zeros here): R_{t+1} = ΔR_world · R_t  ⇒
        ω_world ≈ axis_angle(q_{t+1} ⊗ q_t*) · fps.
        """
        dq_world = quat_mul(ref_obj_quat_next, quat_conjugate(ref_obj_quat))
        ref_obj_ang_vel = axis_angle_from_quat(dq_world) * REF_FPS

        delta_obj_pos_wf = quat_apply(vw_conj, ref_obj_pos - obj_pos)
        delta_obj_rot6d = rot6d_from_quat(quat_mul(quat_conjugate(obj_quat), ref_obj_quat))
        delta_obj_vel_wf = quat_apply(vw_conj, ref_obj_vel - obj.data.root_lin_vel_w)
        ref_obj_ang_vel_wf = quat_apply(vw_conj, ref_obj_ang_vel)
        delta_obj_ang_vel_wf = quat_apply(vw_conj, ref_obj_ang_vel - obj.data.root_ang_vel_w)

        mat_props_dev, signed_distance, surface_nrm_wf, surface_vec_wf, tip_forces_wf = (
            self._contact_geometry_tip_nearest(env, N, device, hand, obj, obj_pos, obj_quat, origins, vw_conj)
        )
        masses = physx_props.read(env, self._object_key, "mass")  # CPU copy
        mass_dev = masses.reshape(N, -1).sum(dim=1, keepdim=True).to(device)  # (N,1)
        return (
            delta_obj_ang_vel_wf,
            delta_obj_pos_wf,
            delta_obj_rot6d,
            delta_obj_vel_wf,
            mass_dev,
            mat_props_dev,
            ref_obj_ang_vel_wf,
            signed_distance,
            surface_nrm_wf,
            surface_vec_wf,
            tip_forces_wf,
        )

    def _contact_geometry_tip_nearest(self, env, N, device, hand, obj, obj_pos, obj_quat, origins, vw_conj):
        """Contact geometry (35): per-tip nearest surface point, wrist frame."""
        tip_pos = hand.data.body_link_pos_w[:, self._tip_body_ids, :] - origins.unsqueeze(1)  # (N,5,3)
        if self._surface_pts_multi is not None:  # --multi_object: per-env mesh clouds
            _clip = self._traj_mgr.clip_ids
            pts_local = self._surface_pts_multi[_clip]
            nrm_local = self._surface_normals_multi[_clip]
        else:
            pts_local = self._surface_pts_local.unsqueeze(0).expand(N, -1, -1)
            nrm_local = self._surface_normals_local.unsqueeze(0).expand(N, -1, -1)
        if self._artic_parts is not None:
            pts_local, nrm_local = self._artic_parts.pose(
                self._traj_mgr.clip_ids if self._surface_pts_multi is not None else None,
                obj.data.joint_pos[:, 0],
                pts_local,
                nrm_local,
            )
        pts_rel = q_rot_vecs(obj_quat, pts_local)  # (N,P,3) object-centred
        nrm_world = q_rot_vecs(obj_quat, nrm_local)
        nrm_world = nrm_world / nrm_world.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        env_idx = torch.arange(N, device=device).unsqueeze(1).expand(-1, tip_pos.shape[1])
        pts_world = pts_rel + obj_pos.unsqueeze(1)  # (N,P,3)
        pts_rel = None  # same lifetime as the old path (same peak memory)
        dist_mat = torch.cdist(tip_pos, pts_world, p=2)  # (N,5,P)
        nearest_idx = dist_mat.argmin(dim=-1)  # (N,5)
        nearest_pt = pts_world[env_idx, nearest_idx]  # (N,5,3)
        nearest_nrm = nrm_world[env_idx, nearest_idx]  # (N,5,3)
        surface_vec = nearest_pt - tip_pos  # tip → surface
        signed_distance = -(surface_vec * nearest_nrm).sum(dim=-1)  # (N,5) +outside
        surface_vec_wf = q_rot_vecs(vw_conj, surface_vec).reshape(N, -1)  # 15
        surface_nrm_wf = q_rot_vecs(vw_conj, nearest_nrm).reshape(N, -1)  # 15

        # ---- privileged (19): tip forces (wrist frame) + material + DR-true mass ----
        contact_sensor = env.scene[f"{self._side}_hand_contact"]
        if self._contact_tip_ids is None:
            sensor_names = contact_sensor.body_names
            self._contact_tip_ids = torch.tensor(
                [sensor_names.index(n) if n in sensor_names else 0 for n in self._tip_names],
                device=device,
                dtype=torch.long,
            )
        tip_forces = contact_sensor.data.net_forces_w[:, self._contact_tip_ids, :]  # (N,5,3)
        tip_forces_wf = q_rot_vecs(vw_conj, tip_forces).reshape(N, -1)  # 15
        # Known limitation: these forces are not clamped. Fingertip contact forces up to 10,214 N were measured (EVAL
        # FINGERPRINT per-dim max): contact solver penetration spikes. Such an outlier can inflate the running
        # observation scaler's variance for that dimension, so normal contact forces (a few N) normalise to nearly 0.

        # Friction and mass (privileged observations) are read through the physx_props window instead of PhysX CPU
        # queries. The values are bit-identical with PhysX (except material key alias cells, which hold the written
        # value
        # until the next oracle comparison; difference < 1e-6, see the physx_props header): every writer (reset
        # randomisation, friction ramp) goes through that window, and the oracle compares with PhysX periodically.
        mat_props_dev = physx_props.read_device(env, self._object_key, "material", device)[:, 0, :]  # (N,3)
        return mat_props_dev, signed_distance, surface_nrm_wf, surface_vec_wf, tip_forces_wf
