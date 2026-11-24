# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ManipTrans-style observation terms for bimanual training.

Implements a class-based observation term that computes concatenated
proprioception, privileged, and target observation streams for each hand.

Observation streams per hand:
  Proprioception (49): 12 q + 12 cos(q) + 12 sin(q) + 13 palm_state
  Privileged    (12): 12 dq
  Target        (23 + N_bodies*9): wrist deltas + body position/velocity tracking

With 13 matched bodies per hand: 49 + 12 + 23 + 117 = 201 dims per hand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, ObservationTermCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_conjugate, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# Import shared naming resolution utilities (dataset vs robot + placeholder left hand)
from dexengine.rl.tasks.mdp.naming import (
    resolve_independent_joints,
    resolve_link7_to_wrist_offset,
    resolve_palm_body_id,
)


def _intrinsic_euler_xyz_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Extract intrinsic XYZ Euler angles from wxyz quaternion.

    Decomposes R = Rx(alpha) @ Ry(beta) @ Rz(gamma), matching the MuJoCo
    virtual wrist chain: rot_x -> rot_y -> rot_z (each in parent body frame).

    Args:
        quat_wxyz: (N, 4) quaternion in wxyz convention.

    Returns:
        (N, 3) tensor of [alpha, beta, gamma] = [rot_x, rot_y, rot_z] in radians.
    """
    R = matrix_from_quat(quat_wxyz)  # (N, 3, 3)
    beta = torch.asin(R[:, 0, 2].clamp(-1.0, 1.0))
    alpha = torch.atan2(-R[:, 1, 2], R[:, 2, 2])
    gamma = torch.atan2(-R[:, 0, 1], R[:, 0, 0])
    return torch.stack([alpha, beta, gamma], dim=-1)


class HandObservation(ManagerTermBase):
    """ManipTrans-style observation for one hand.

    Computes proprioception + privileged + target observations concatenated
    into a single tensor per hand.

    Output dimensions per hand:
      Proprioception: 12 (q) + 12 (cos_q) + 12 (sin_q) + 13 (palm_state) = 49
      Privileged: 12 (dq) = 12
      Target: 3 (delta_wrist_pos) + 3 (ref_wrist_vel) + 3 (delta_wrist_vel)
              + 4 (ref_wrist_quat) + 4 (delta_wrist_quat)
              + 3 (ref_wrist_ang_vel) + 3 (delta_wrist_ang_vel)
              + N_bodies*3 (delta_body_pos) + N_bodies*3 (ref_body_vel)
              + N_bodies*3 (delta_body_vel)
      Total target = 23 + N_bodies*9

      With 13 matched bodies: 49 + 12 + 23 + 117 = 201 per hand.

    The ``hand_name`` parameter selects which hand ("right_hand" or "left_hand").
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Determine hand side from params
        self._hand_name = cfg.params["hand_name"]
        self._prefix = "right_" if "right" in self._hand_name else "left_"
        self._side = "right" if self._prefix == "right_" else "left"

        # Cache trajectory manager reference
        self._traj_mgr = env.trajectory_manager

        # Get articulation
        asset = env.scene[self._hand_name]

        palm_body_name = self._find_independent_dof_ids(cfg, env, asset)
        print(f"[HandObservation:{self._hand_name}] Independent dim (6 wrist + fingers): {self._n_indep}")
        print(f"[HandObservation:{self._hand_name}] Palm body ID: {self._palm_body_id} ({palm_body_name})")
        print(f"[HandObservation:{self._hand_name}] Full-robot mode: {self._is_fullrobot}")
        print(f"[HandObservation:{self._hand_name}] Matched bodies: {self._n_bodies}")
        print(f"[HandObservation:{self._hand_name}] Body orientations: {self._has_body_orientations}")

    def _find_independent_dof_ids(self, cfg, env, asset):
        """Find independent DOF joint indices in PhysX order Auto-detects dataset vs robot vs SharpA naming +
        placeholder left hand.
        """
        independent_joint_names = resolve_independent_joints(asset, self._side)
        indep_ids = [asset.data.joint_names.index(n) for n in independent_joint_names]
        self._indep_joint_ids = torch.tensor(indep_ids, device=env.device)

        # Full independent dimension = 6 virtual-wrist + n finger joints.
        # The resolver returns either the full list (floating hand, includes
        # {side}_pos_x...) or the finger-only list (full robot). 12 for RH56F1,
        # 28 for SharpA (22 fingers).
        _has_virtual_wrist = independent_joint_names[0].endswith("pos_x")
        if _has_virtual_wrist:
            self._n_indep = len(independent_joint_names)
        else:
            self._n_indep = 6 + len(independent_joint_names)

        # Find palm body index in Isaac Lab (end of wrist chain = palm body)
        self._palm_body_id = resolve_palm_body_id(asset, self._side)

        # Get matched body indices from trajectory manager
        if self._side == "right":
            body_ids_isaac = self._traj_mgr.right_body_ids_isaac
        else:
            body_ids_isaac = self._traj_mgr.left_body_ids_isaac
        self._body_ids_tensor = torch.tensor(body_ids_isaac, device=env.device)

        self._n_bodies = self._traj_mgr.n_body_positions
        self._has_body_orientations = self._traj_mgr.has_body_orientations

        # Detect full-robot mode: palm body is link7 instead of wrist_rot_z.
        # In this mode we compute a virtual wrist pose from link7 + fixed offset
        # so that observations match the floating-hand training distribution.
        palm_body_name = asset.data.body_names[self._palm_body_id]
        # Full-robot mode means the palm **is the arm's EE link** (merged by the importer). A hard-coded EE link name
        # only fits one arm (RB-Y1's is link_right_arm_6) and made obs V2 refuse at once, so the arm registry is asked.
        from dexengine.rl.tasks.mdp.naming import is_arm_ee_body as _is_ee

        self._is_fullrobot = _is_ee(palm_body_name)

        if self._is_fullrobot:
            # Fixed offset from link7 to virtual wrist body (wrist_rot_z / C_MC).
            # SIDE-AWARE mounting rotation: right arm Rz(-90°), mirrored left arm
            # Rz(+90°) (measured against the bimanual URDF). The prior
            # single Rz(-90°) gave the LEFT hand a 180°-wrong virtual-wrist orientation
            # in the obs. MUST match the action term's _r_offset_fwd_wxyz.
            from dexengine.rl.tasks.mdp.naming import resolve_ee_to_wrist_rot as _rot_pair  # side-aware pair

            self._link7_to_wrist_rot = torch.tensor(list(_rot_pair(asset, self._side)), device=env.device)
            # Translation: explicit cfg param wins; otherwise resolved from the
            # hand naming (RH56F1 0.0555, SharpA 0.0655). MUST match the action
            # term's link7_to_wrist_offset.
            _t_off = cfg.params.get("link7_to_wrist_offset", None)
            if _t_off is None:
                _t_off = resolve_link7_to_wrist_offset(asset, self._side)
            self._link7_to_wrist_trans = torch.tensor(list(_t_off), device=env.device)
            # Floating hand root pose (used for q_12 decomposition).
            # In training, the floating hand root is at env_origins + (0,0,desk_height)
            # with orientation Rz(-90°). We need this to compute virtual wrist joint
            # values in the same frame as training.
            traj_mgr = env.trajectory_manager
            if hasattr(traj_mgr, "root_quat_rotated"):
                self._floating_root_quat = traj_mgr.root_quat_rotated.clone()
            elif hasattr(traj_mgr, "_base_root_quat"):
                self._floating_root_quat = traj_mgr._base_root_quat.clone()
            else:
                self._floating_root_quat = torch.tensor([0.7071068, 0.0, 0.0, -0.7071068], device=env.device)
            # Desk height offset: floating hand root is at env_origins + (0, 0, desk_height)
            self._floating_root_z_offset = getattr(traj_mgr, "_desk_height", 0.0)

        print(f"[HandObservation:{self._hand_name}] Independent joint IDs: {self._indep_joint_ids.tolist()}")
        return palm_body_name

    def __call__(
        self,
        env: ManagerBasedEnv,
        hand_name: str = "right_hand",
        link7_to_wrist_offset: tuple[float, float, float] | None = None,
    ) -> torch.Tensor:
        """Compute ManipTrans-style observations for one hand.

        Args:
            env: The environment instance.
            hand_name: Name of the hand in the scene (passed via params, but we
                use the cached ``self._hand_name`` set during __init__).

        Returns:
            Observation tensor of shape (num_envs, obs_dim).
        """
        asset = env.scene[self._hand_name]
        N = env.num_envs
        device = env.device

        # ================================================================
        # Full-robot: compute virtual wrist state from link7 + offset
        # ================================================================
        # In full-robot mode, the palm body is link7 (arm EE), not wrist_rot_z.
        # We compute the virtual wrist body pose/velocity so all downstream
        # observations match the floating-hand training distribution.
        if self._is_fullrobot:
            link7_pos_w = asset.data.body_link_pos_w[:, self._palm_body_id, :]
            link7_quat_w = asset.data.body_link_quat_w[:, self._palm_body_id, :]
            link7_lin_vel = asset.data.body_link_lin_vel_w[:, self._palm_body_id, :]
            link7_ang_vel = asset.data.body_link_ang_vel_w[:, self._palm_body_id, :]

            _vw_ang_vel, _vw_lin_vel, _vw_pos_w, _vw_quat_w = self._wrist_quat_from_link7(
                N, link7_ang_vel, link7_lin_vel, link7_pos_w, link7_quat_w
            )
        else:
            # read only under `if self._is_fullrobot` below; bound so the steps can be helpers
            _vw_ang_vel = _vw_lin_vel = _vw_pos_w = _vw_quat_w = None

        # ================================================================
        # Proprioception (49)
        # ================================================================
        joint_pos = asset.data.joint_pos  # (N, 18) or (N, 40+) for full robot
        q_raw = joint_pos[:, self._indep_joint_ids]  # (N, n_fingers or n_indep)
        # Pad to n_indep dims if finger-only (full robot, no virtual wrist joints)
        if q_raw.shape[-1] < self._n_indep:
            if self._is_fullrobot:
                # Compute virtual wrist joint values from actual robot state.
                # The floating hand's virtual joints are: pos_x/y/z (slide) + rot_x/y/z (hinge)
                # relative to the floating hand's root body.
                # In training, the root is at env_origins + (0, 0, desk_height) with Rz(-90°).
                root_q = self._floating_root_quat.unsqueeze(0).expand(N, -1)
                # Floating hand root position: env_origins + desk height Z offset
                floating_root_pos = env.scene.env_origins.clone()
                floating_root_pos[:, 2] += self._floating_root_z_offset
                # Position in root local frame
                wrist_rel_w = _vw_pos_w - floating_root_pos
                root_conj = quat_conjugate(root_q)
                pos_local = quat_apply(root_conj, wrist_rel_w)  # (N, 3)
                # Orientation in root local frame: q_local = conj(q_root) * q_wrist
                quat_local = quat_mul(root_conj, _vw_quat_w)  # (N, 4)
                rot_xyz = _intrinsic_euler_xyz_from_quat(quat_local)  # (N, 3)
                virtual_wrist_q = torch.cat([pos_local, rot_xyz], dim=-1)  # (N, 6)
                q_12 = torch.cat([virtual_wrist_q, q_raw], dim=-1)  # (N, n_indep)
            else:
                wrist_qpos = self._non_fullrobot_finger_fallback(env, N, device)
                q_12 = torch.cat([wrist_qpos, q_raw], dim=-1)  # (N, n_indep)
        else:
            q_12 = q_raw
        return self._palm_and_target_obs(env, N, _vw_ang_vel, _vw_lin_vel, _vw_pos_w, _vw_quat_w, asset, device, q_12)

    def _palm_and_target_obs(self, env, N, _vw_ang_vel, _vw_lin_vel, _vw_pos_w, _vw_quat_w, asset, device, q_12):
        """Palm state, proprioception (49), privileged joint velocities (12) and target stream (23 + n_bodies * 9)."""
        cos_q = torch.cos(q_12)  # (N, 12)
        sin_q = torch.sin(q_12)  # (N, 12)

        # Palm state: [pos_zeroed(3), quat(4), lin_vel(3), ang_vel(3)] = 13
        # Position is zeroed per ManipTrans convention (relative info in target stream)
        palm_pos = torch.zeros(N, 3, device=device)
        if self._is_fullrobot:
            palm_quat = _vw_quat_w
            palm_lin_vel = _vw_lin_vel
            palm_ang_vel = _vw_ang_vel
        else:
            palm_quat = asset.data.body_link_quat_w[:, self._palm_body_id, :]  # (N, 4) wxyz
            palm_lin_vel = asset.data.body_link_lin_vel_w[:, self._palm_body_id, :]  # (N, 3)
            palm_ang_vel = asset.data.body_link_ang_vel_w[:, self._palm_body_id, :]  # (N, 3)
        palm_state = torch.cat([palm_pos, palm_quat, palm_lin_vel, palm_ang_vel], dim=-1)  # (N, 13)

        proprioception = torch.cat([q_12, cos_q, sin_q, palm_state], dim=-1)  # (N, 49)

        # ================================================================
        # Privileged (12)
        # ================================================================
        joint_vel = asset.data.joint_vel  # (N, 18) or (N, 40+)
        dq_raw = joint_vel[:, self._indep_joint_ids]  # (N, n_fingers or n_indep)
        if dq_raw.shape[-1] < self._n_indep:
            if self._is_fullrobot:
                # Use offset-corrected virtual wrist body velocity
                wrist_vel_equiv = torch.cat([_vw_lin_vel, _vw_ang_vel], dim=-1)  # (N, 6)
            else:
                palm_lin_vel_for_joints = asset.data.body_link_lin_vel_w[:, self._palm_body_id, :]
                palm_ang_vel_for_joints = asset.data.body_link_ang_vel_w[:, self._palm_body_id, :]
                wrist_vel_equiv = torch.cat([palm_lin_vel_for_joints, palm_ang_vel_for_joints], dim=-1)
            dq_12 = torch.cat([wrist_vel_equiv, dq_raw], dim=-1)
        else:
            dq_12 = dq_raw

        # ================================================================
        # Target (23 + n_bodies * 9)
        # ================================================================
        ref_body = self._traj_mgr.get_reference_body_pos(env.episode_length_buf)

        # Current palm state in env-local frame
        if self._is_fullrobot:
            cur_palm_pos = _vw_pos_w - env.scene.env_origins  # (N, 3)
        else:
            cur_palm_pos = asset.data.body_link_pos_w[:, self._palm_body_id, :] - env.scene.env_origins  # (N, 3)
        cur_palm_quat = palm_quat  # (N, 4) wxyz
        cur_palm_lin_vel = palm_lin_vel  # (N, 3)
        cur_palm_ang_vel = palm_ang_vel  # (N, 3)

        return self._reference_palm_pose(
            env,
            N,
            asset,
            cur_palm_ang_vel,
            cur_palm_lin_vel,
            cur_palm_pos,
            cur_palm_quat,
            device,
            dq_12,
            proprioception,
            ref_body,
        )

    def _non_fullrobot_finger_fallback(self, env, N, device):
        """Non-fullrobot finger-only fallback: use trajectory reference."""
        # the lookup stores this step's reference wrist pose in _last_{rh,lh}_wrist_qpos, read below
        self._traj_mgr.get_reference_state(env.episode_length_buf)
        if self._side == "right":
            wrist_qpos = getattr(self._traj_mgr, "_last_rh_wrist_qpos", None)
        else:
            wrist_qpos = getattr(self._traj_mgr, "_last_lh_wrist_qpos", None)
        if wrist_qpos is None:
            wrist_qpos = torch.zeros(N, 6, device=device)
        return wrist_qpos

    def _wrist_quat_from_link7(self, N, link7_ang_vel, link7_lin_vel, link7_pos_w, link7_quat_w):
        """wrist_quat = link7_quat * R_OFFSET (Rz(-90°))."""
        r_offset = self._link7_to_wrist_rot.unsqueeze(0).expand(N, -1)
        _vw_quat_w = quat_mul(link7_quat_w, r_offset)

        # wrist_pos = link7_pos + R(link7_quat) @ T_OFFSET
        link7_rot = matrix_from_quat(link7_quat_w)  # (N, 3, 3)
        t_col = self._link7_to_wrist_trans.view(1, 3, 1).expand(N, -1, -1)
        offset_w = torch.bmm(link7_rot, t_col).squeeze(-1)  # (N, 3)
        _vw_pos_w = link7_pos_w + offset_w

        # wrist_lin_vel = link7_vel + omega × offset_world
        _vw_lin_vel = link7_lin_vel + torch.cross(link7_ang_vel, offset_w, dim=-1)
        _vw_ang_vel = link7_ang_vel  # same for rigid attachment
        return _vw_ang_vel, _vw_lin_vel, _vw_pos_w, _vw_quat_w

    def _reference_palm_pose(
        self,
        env,
        N,
        asset,
        cur_palm_ang_vel,
        cur_palm_lin_vel,
        cur_palm_pos,
        cur_palm_quat,
        device,
        dq_12,
        proprioception,
        ref_body,
    ):
        """Reference palm = first body in matched set (hand_base -> wrist_rot_z)."""
        ref_palm_pos = ref_body[f"{self._side}_body_pos"][:, 0, :]  # (N, 3)
        ref_palm_vel = ref_body[f"{self._side}_body_vel"][:, 0, :]  # (N, 3)

        # --- Wrist position deltas ---
        delta_wrist_pos = ref_palm_pos - cur_palm_pos  # (N, 3)
        ref_wrist_vel = ref_palm_vel  # (N, 3)
        delta_wrist_vel = ref_wrist_vel - cur_palm_lin_vel  # (N, 3)

        # --- Wrist orientation deltas ---
        if self._has_body_orientations:
            # Real reference quaternion and angular velocity from precomputed FK
            ref_wrist_quat = ref_body[f"{self._side}_body_quat"][:, 0, :]  # (N, 4) wxyz
            # Delta quaternion: rotation from current to reference (body frame)
            # delta = conj(cur) * ref
            delta_wrist_quat = quat_mul(quat_conjugate(cur_palm_quat), ref_wrist_quat)  # (N, 4)
            # Ensure shortest path (w >= 0)
            sign = torch.sign(delta_wrist_quat[:, 0:1])
            sign[sign == 0] = 1.0
            delta_wrist_quat = delta_wrist_quat * sign

            ref_wrist_ang_vel = ref_body[f"{self._side}_body_ang_vel"][:, 0, :]  # (N, 3)
            delta_wrist_ang_vel = ref_wrist_ang_vel - cur_palm_ang_vel  # (N, 3)
        else:
            # Fallback: placeholder (backward compat with old body_positions.npz)
            ref_wrist_quat = cur_palm_quat
            delta_wrist_quat = torch.zeros(N, 4, device=device)
            delta_wrist_quat[:, 0] = 1.0  # identity quaternion wxyz
            ref_wrist_ang_vel = torch.zeros(N, 3, device=device)
            delta_wrist_ang_vel = ref_wrist_ang_vel - cur_palm_ang_vel

        # --- Body position tracking ---
        cur_body_pos = asset.data.body_link_pos_w[:, self._body_ids_tensor, :]  # (N, n_bodies, 3)
        cur_body_pos_local = cur_body_pos - env.scene.env_origins.unsqueeze(1)  # (N, n_bodies, 3)

        ref_body_pos = ref_body[f"{self._side}_body_pos"]  # (N, n_bodies, 3) env-local
        delta_body_pos = ref_body_pos - cur_body_pos_local  # (N, n_bodies, 3)

        # Body velocities
        cur_body_vel = asset.data.body_link_lin_vel_w[:, self._body_ids_tensor, :]  # (N, n_bodies, 3)
        ref_body_vel = ref_body[f"{self._side}_body_vel"]  # (N, n_bodies, 3)
        delta_body_vel = ref_body_vel - cur_body_vel  # (N, n_bodies, 3)

        # Concatenate target stream
        target = torch.cat(
            [
                delta_wrist_pos,  # 3
                ref_wrist_vel,  # 3
                delta_wrist_vel,  # 3
                ref_wrist_quat,  # 4
                delta_wrist_quat,  # 4
                ref_wrist_ang_vel,  # 3
                delta_wrist_ang_vel,  # 3
                delta_body_pos.reshape(N, -1),  # n_bodies * 3
                ref_body_vel.reshape(N, -1),  # n_bodies * 3
                delta_body_vel.reshape(N, -1),  # n_bodies * 3
            ],
            dim=-1,
        )  # (N, 23 + n_bodies * 9)

        # ================================================================
        # Concatenate all streams
        # ================================================================
        result = torch.cat([proprioception, dq_12, target], dim=-1)
        return result


class ReferenceBaseAction(ManagerTermBase):
    """Observation term: the REFERENCE base action (clean reference-base + residual single-stage).

    Replacement for the legacy ``LastBaseAction`` (which ran a frozen base model and fed
    its raw output into the residual policy observation). This term instead returns the 12-dim
    base action the residual policy is actually correcting — wrist-pose delta = zeros + reference
    finger qpos in the normalized [-1, 1] action space — computed directly from the reference
    trajectory via the action term's ``compute_reference_base_action()``. NO base checkpoint
    and NO base-model forward pass are involved, and the observation base == the control base by
    construction (same method). Same output dim (action_dim) as LastBaseAction, so the policy obs
    dimension and normalizer layout are unchanged.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._action_key = cfg.params.get("action_key", "right_hand_joint_pos")

    def __call__(self, env: ManagerBasedEnv, action_key: str = "right_hand_joint_pos") -> torch.Tensor:
        action_term = env.action_manager._terms[self._action_key]
        if not hasattr(action_term, "compute_reference_base_action"):
            raise RuntimeError(
                f"[ReferenceBaseAction] action term '{self._action_key}' has no "
                "compute_reference_base_action(); ReferenceBaseAction requires the reference-base "
                "residual action term (single-stage)."
            )
        _out = action_term.compute_reference_base_action()  # (N, action_dim)
        return _out
