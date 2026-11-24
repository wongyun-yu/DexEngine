"""Action processing of the arm-IK action: wrist deltas and finger targets to IK commands, and reset."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    pass


class ArmIKProcessMixin:
    """Methods of :class:`ArmIKFingersPositionAction` (action processing of the arm-ik action: wrist deltas and finger
    targets to ik commands, and reset)."""

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process 12-dim actions: 6 wrist deltas + 6 finger positions.

        Per-hand action layout:
          action[0:3]  = wrist position delta (scaled by wrist_pos_scale)
          action[3:6]  = wrist rotation delta (scaled by wrist_rot_scale)
          action[6..11] = finger positions in [-1, 1]

        Pipeline: temporal_ensemble -> EMA -> compute_base -> add_delta -> ik_target_ema.

        Two wrist target modes (configurable via use_trajectory_reference):
          - current-state (default): base = actual link7 state each step
          - trajectory-reference: base = trajectory wrist pose for current frame

        Args:
            actions: (num_envs, 12) wrist deltas + finger targets.
        """
        # -- Temporal ensembling (before EMA) --
        self._raw_actions[:] = actions
        if self._use_temporal_ensembling:
            actions = self._temporal_ensemble(actions)

        # -- EMA smoothing (full 12-dim) --
        if self._ema < 1.0:
            self._processed_actions[:] = self._ema * actions + (1.0 - self._ema) * self._prev_actions
        else:
            self._processed_actions[:] = actions
        self._prev_actions[:] = self._processed_actions

        root_q, target_quat_local, wrist_pos_w = self._compute_wrist_target()
        wrist_quat_w = math_utils.quat_mul(root_q, target_quat_local)

        # -- Convert wrist -> link7 target (batched GPU) --
        link7_pos_w, link7_quat_w = self._wrist_to_link7_batch(wrist_pos_w, wrist_quat_w)

        # -- Transform link7 world pose to robot root frame --
        root_pos_w = self._asset.data.root_pos_w
        root_quat_w = self._asset.data.root_quat_w
        link7_pos_b, link7_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, link7_pos_w, link7_quat_w
        )

        # -- Set IK command (absolute pose in root frame) --
        self._ik_commands[:, :3] = link7_pos_b
        self._ik_commands[:, 3:7] = link7_quat_b

        # -- Feed current EE pose + command to controller --
        ee_pos_b, ee_quat_b = self._compute_ee_pose_in_root()
        self._ik_controller.set_command(self._ik_commands, ee_pos_b, ee_quat_b)

    def _compute_wrist_target(self):
        """Compute wrist target via proper quaternion composition."""
        wrist_pos_delta = self._processed_actions[:, 0:3] * self._wrist_pos_scale
        wrist_rot_delta = self._processed_actions[:, 3:6] * self._wrist_rot_scale

        # Convert rotation delta to quaternion (intrinsic XYZ euler → quat)
        delta_quat = self._euler_xyz_to_quat(wrist_rot_delta)  # (N, 4) wxyz

        if self._use_traj_ref:
            # Trajectory-reference mode: target = trajectory_wrist_pose ⊕ delta
            # The trajectory provides the ground-truth wrist pose for the current frame.
            # The policy delta acts as a correction on top of this stable reference.
            frame_ids = self._env.episode_length_buf
            wrist_pos_w = self._traj_mgr.get_wrist_pos_w(frame_ids) + self._env.scene.env_origins
            wrist_quat_w = self._traj_mgr.get_wrist_quat_w(frame_ids)

            N = wrist_pos_w.shape[0]
            floating_root_pos = self._env.scene.env_origins.clone()
            floating_root_pos[:, 2] += self._floating_root_z_offset
            root_q = self._floating_root_quat.unsqueeze(0).expand(N, -1)
            root_conj = math_utils.quat_conjugate(root_q)

            pos_local = math_utils.quat_apply(root_conj, wrist_pos_w - floating_root_pos)
            quat_local = math_utils.quat_mul(root_conj, wrist_quat_w)

            # Position: linear addition (correct for translation)
            target_pos_local = pos_local + wrist_pos_delta
            # Rotation: proper quaternion composition (body-frame delta)
            target_quat_local = math_utils.quat_mul(quat_local, delta_quat)
        else:
            # Current-state mode (baseline): target = current_virtual_wrist ⊕ delta
            link7_pos_w = self._asset.data.body_pos_w[:, self._ee_body_idx]
            link7_quat_w = self._asset.data.body_quat_w[:, self._ee_body_idx]

            N = link7_pos_w.shape[0]
            r_offset = self._r_offset_fwd_wxyz.unsqueeze(0).expand(N, -1)
            vw_quat_w = math_utils.quat_mul(link7_quat_w, r_offset)

            link7_rot = math_utils.matrix_from_quat(link7_quat_w)
            t_col = self._link7_to_wrist_trans.view(1, 3, 1).expand(N, -1, -1)
            offset_w = torch.bmm(link7_rot, t_col).squeeze(-1)
            vw_pos_w = link7_pos_w + offset_w

            floating_root_pos = self._env.scene.env_origins.clone()
            floating_root_pos[:, 2] += self._floating_root_z_offset
            root_q = self._floating_root_quat.unsqueeze(0).expand(N, -1)
            root_conj = math_utils.quat_conjugate(root_q)

            pos_local = math_utils.quat_apply(root_conj, vw_pos_w - floating_root_pos)
            quat_local = math_utils.quat_mul(root_conj, vw_quat_w)

            # Position: linear addition (correct for translation)
            target_pos_local = pos_local + wrist_pos_delta
            # Rotation: proper quaternion composition (body-frame delta)
            target_quat_local = math_utils.quat_mul(quat_local, delta_quat)

        # -- Optional EMA smoothing on the IK target --
        if self._ik_target_ema < 1.0:
            ema = self._ik_target_ema
            target_pos_local = ema * target_pos_local + (1.0 - ema) * self._prev_wrist_pos_local
            # NLERP for quaternion (shortest-path correction + normalize)
            prev_q = self._prev_wrist_quat_local
            dot = (target_quat_local * prev_q).sum(dim=-1, keepdim=True)
            prev_q = torch.where(dot < 0, -prev_q, prev_q)
            target_quat_local = ema * target_quat_local + (1.0 - ema) * prev_q
            target_quat_local = target_quat_local / target_quat_local.norm(dim=-1, keepdim=True)
        self._prev_wrist_pos_local[:] = target_pos_local
        self._prev_wrist_quat_local[:] = target_quat_local

        # Convert target local pose to wrist world pose
        wrist_pos_w = floating_root_pos + math_utils.quat_apply(root_q, target_pos_local)
        return root_q, target_quat_local, wrist_pos_w

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset action buffers for given environments."""
        if env_ids is None:
            self._raw_actions[:] = 0.0
            self._processed_actions[:] = 0.0
            self._prev_actions[:] = 0.0
            # Initialize arm targets from current actual joint positions
            self._prev_arm_targets[:] = self._asset.data.joint_pos[:, self._arm_joint_ids]
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._prev_actions[env_ids] = 0.0
            self._prev_arm_targets[env_ids] = self._asset.data.joint_pos[env_ids][:, self._arm_joint_ids]

        # Clear temporal ensembling history for reset envs
        if self._use_temporal_ensembling:
            if env_ids is None:
                self._action_history[:] = 0.0
                self._history_count[:] = 0
            else:
                self._action_history[:, env_ids] = 0.0
                self._history_count[env_ids] = 0

        # Initialize wrist target from actual link7 state (only on reset)
        # This grounds the persistent target to the actual arm configuration.
        if env_ids is None:
            ids = slice(None)
        else:
            ids = env_ids
        link7_pos_w = self._asset.data.body_pos_w[ids, self._ee_body_idx]
        link7_quat_w = self._asset.data.body_quat_w[ids, self._ee_body_idx]
        N = link7_pos_w.shape[0]
        r_offset = self._r_offset_fwd_wxyz.unsqueeze(0).expand(N, -1)
        vw_quat_w = math_utils.quat_mul(link7_quat_w, r_offset)
        link7_rot = math_utils.matrix_from_quat(link7_quat_w)
        t_col = self._link7_to_wrist_trans.view(1, 3, 1).expand(N, -1, -1)
        offset_w = torch.bmm(link7_rot, t_col).squeeze(-1)
        vw_pos_w = link7_pos_w + offset_w
        floating_root_pos = self._env.scene.env_origins[ids].clone()
        floating_root_pos[:, 2] += self._floating_root_z_offset
        root_q = self._floating_root_quat.unsqueeze(0).expand(N, -1)
        root_conj = math_utils.quat_conjugate(root_q)
        pos_local = math_utils.quat_apply(root_conj, vw_pos_w - floating_root_pos)
        quat_local = math_utils.quat_mul(root_conj, vw_quat_w)
        self._prev_wrist_pos_local[ids] = pos_local
        self._prev_wrist_quat_local[ids] = quat_local
