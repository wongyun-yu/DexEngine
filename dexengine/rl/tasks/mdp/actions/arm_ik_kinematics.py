"""Arm kinematics of the arm-IK action: end-effector pose, Jacobian, wrist-to-link transform and the arm/finger target
writes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    pass


class ArmKinematicsMixin:
    """Methods of :class:`ArmIKFingersPositionAction` (arm kinematics of the arm-ik action: end-effector pose, jacobian,
    wrist-to-link transform and the arm/finger target writes)."""

    def _temporal_ensemble(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply EgoVLA-style temporal ensembling on actions.

        Exponential-weighted average over action history with separate
        weights for wrist (dims 0 to 5) and finger (dims 6 to 11) components.

        Args:
            actions: (num_envs, 12) raw actions from policy.
        Returns:
            Smoothed actions (num_envs, 12).
        """
        H = self._smooth_history_len
        N = actions.shape[0]

        # Shift history: move all entries one slot older
        self._action_history[1:] = self._action_history[:-1].clone()
        # Insert new action at position 0 (most recent)
        self._action_history[0] = actions

        # Track how many valid history entries each env has
        self._history_count.clamp_(max=H - 1)
        self._history_count += 1

        # Compute weighted average per env
        # weights shape: (H, 12), history shape: (H, N, 12)
        # For each env, only use entries up to history_count
        A = self._smooth_weights.shape[1]
        weights = self._smooth_weights.unsqueeze(1).expand(H, N, A)  # (H, N, A)

        # Mask out entries beyond each env's history count
        idx = torch.arange(H, device=self.device).unsqueeze(1)  # (H, 1)
        valid_mask = idx < self._history_count.unsqueeze(0)  # (H, N)
        valid_mask = valid_mask.unsqueeze(2).expand(H, N, A)  # (H, N, A)

        masked_weights = weights * valid_mask.float()
        weight_sums = masked_weights.sum(dim=0)  # (N, A)
        weight_sums = weight_sums.clamp(min=1e-8)

        smoothed = (self._action_history * masked_weights).sum(dim=0) / weight_sums  # (N, A)
        return smoothed

    def apply_actions(self) -> None:
        """Compute arm IK joint targets and write arm + finger targets."""
        arm_joint_targets = self._apply_arm()
        self._apply_fingers(arm_joint_targets)

    def _apply_arm(self) -> torch.Tensor:
        """Arm part of ``apply_actions`` — IK → arm joint position targets (returned for the recorder).
        The floating-hand subclass overrides this with the wrist wrench (no arm joints)."""
        # -- Arm: DifferentialIK --
        ee_pos_b, ee_quat_b = self._compute_ee_pose_in_root()
        arm_joint_pos = self._asset.data.joint_pos[:, self._arm_joint_ids]
        jacobian = self._compute_arm_jacobian()
        arm_joint_targets = self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, arm_joint_pos)
        # RIGHT-PARK: discard policy/IK arm targets, hold the default pose.
        if self._park:
            # PARK POSE: hold default_joint_pos, matching mdp/events reset
            # park and the left arm's park in right-only training. The scene cfg default
            # is now symmetric (both arms 0 -> straight-down hang) after removing the
            # right-only elbow-up j4=90 override (was cosmetic desk-touch in L renders).
            arm_joint_targets = self._asset.data.default_joint_pos[:, self._arm_joint_ids]
        self._asset.set_joint_position_target(arm_joint_targets, joint_ids=self._arm_joint_ids)
        return arm_joint_targets

    def _apply_fingers(self, arm_joint_targets: torch.Tensor) -> None:
        """Finger part of ``apply_actions`` (+ head hold + target recorder)."""
        # -- Fingers: ManipTrans-style absolute scaling --
        # scale(a, lo, hi) = 0.5 * (a + 1) * (hi - lo) + lo
        finger_actions = self._processed_actions[:, 6:]  # finger portion of the action
        # ⚠ Reader's note: from this absolute mapping ([-1,1] -> the joint's full range) alone, do not conclude that
        # "the fingers have absolute targets with no reference". On the campaign (residual) path ResidualArmIKAction,
        # in REFERENCE-BASE mode, computes **base = the reference finger qpos normalised to [-1,1], plus
        # residual × FINGER_RESIDUAL_SCALE (default 0.3)** and overwrites `inner._processed_actions[:, 6:]` with it
        # (residual_ik_process.compute_reference_base_action / process_actions). So this formula is the decoder that
        # maps that sum back to joint space; the "reference-centred delta" already lives in the outer term. Only on
        # its own (the non-residual floating-hand path) is this a true absolute mapping.
        finger_targets = 0.5 * (finger_actions + 1.0) * self._finger_range + self._finger_lower
        # RIGHT-PARK: hold default finger pose (coupled joints below follow automatically).
        if self._park:
            finger_targets = self._asset.data.default_joint_pos[:, self._finger_joint_ids]
        self._asset.set_joint_position_target(finger_targets, joint_ids=self._finger_joint_ids)

        # -- Coupled joints: explicit coupling (full robot may lack PhysxMimicJointAPI) --
        if self._has_coupling and len(self._coupled_joint_ids) == self._num_coupled_joints:
            src = finger_targets[:, self._coupling_source_idx]  # (N, 6) source joint positions
            coupled_targets = src * self._coupling_ratios  # (N, 6) coupled positions
            self._asset.set_joint_position_target(coupled_targets, joint_ids=self._coupled_joint_ids)

        # -- Head pitch: maintain look-down target (prevents PD drift to 0) --
        # The joint id and the constant target are fixed for the run, so they are built on the first call; the write
        # copies the target into the joint target buffer (same values every call).
        if not hasattr(self, "_head_hold"):
            head_joint_names = [n for n in self._asset.joint_names if "head_pitch" in n]
            self._head_hold = (
                (
                    [self._asset.joint_names.index(head_joint_names[0])],
                    torch.full((self.num_envs, 1), 0.8, device=self.device),
                )
                if head_joint_names
                else None
            )
        if self._head_hold is not None:
            self._asset.set_joint_position_target(self._head_hold[1], joint_ids=self._head_hold[0])

    def _euler_xyz_to_quat(self, euler: torch.Tensor) -> torch.Tensor:
        """Convert intrinsic XYZ euler angles (N, 3) to quaternion (N, 4) wxyz."""
        from isaaclab.utils.math import matrix_from_euler, quat_from_matrix

        rot_mat = matrix_from_euler(euler, "XYZ")  # (N, 3, 3)
        return quat_from_matrix(rot_mat)  # (N, 4) wxyz

    def _compute_ee_pose_in_root(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get current link7 pose in robot root frame."""
        ee_pos_w = self._asset.data.body_pos_w[:, self._ee_body_idx]
        ee_quat_w = self._asset.data.body_quat_w[:, self._ee_body_idx]
        root_pos_w = self._asset.data.root_pos_w
        root_quat_w = self._asset.data.root_quat_w
        return math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

    def _compute_arm_jacobian(self) -> torch.Tensor:
        """Get geometric Jacobian for arm joints at link7 body in root frame.

        Returns:
            Jacobian tensor of shape (num_envs, 6, 7).
        """
        # World-frame Jacobian from PhysX: (num_envs, 6, num_arm_joints)
        jacobian_w = self._asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._arm_joint_ids]
        # Rotate to root frame (same approach as DifferentialInverseKinematicsAction)
        base_rot = self._asset.data.root_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, jacobian_w[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_w[:, 3:, :])
        return jacobian_b

    def _wrist_to_link7_batch(
        self, wrist_pos_w: torch.Tensor, wrist_quat_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert wrist world pose to link7 world target (batched GPU).

        This is the GPU-batched equivalent of the verified scipy-based
        ``wrist_pose_to_link7_target`` from step6:

            link7_rot = wrist_rot * R_OFFSET.inv()
            link7_pos = wrist_pos - link7_rot @ T_OFFSET

        where R_OFFSET = Rz(-90 deg) is the fixed link7->wrist rotation, and
        T_OFFSET = (0, 0, 0.0555) is the translation in link7 frame.

        Args:
            wrist_pos_w: (num_envs, 3) wrist position in world frame.
            wrist_quat_w: (num_envs, 4) wrist orientation wxyz in world frame.

        Returns:
            link7_pos_w: (num_envs, 3) link7 position in world frame.
            link7_quat_w: (num_envs, 4) link7 orientation wxyz in world frame.
        """
        N = wrist_pos_w.shape[0]

        # link7_quat = wrist_quat * R_OFFSET_inv  (quaternion multiply, wxyz convention)
        r_offset_inv_batch = self._r_offset_inv_wxyz.unsqueeze(0).expand(N, -1)
        link7_quat_w = math_utils.quat_mul(wrist_quat_w, r_offset_inv_batch)

        # link7_pos = wrist_pos - R(link7_quat) @ T_OFFSET
        link7_rot_mat = math_utils.matrix_from_quat(link7_quat_w)  # (N, 3, 3)
        t_offset_col = self._t_offset.view(1, 3, 1).expand(N, -1, -1)  # (N, 3, 1)
        offset_world = torch.bmm(link7_rot_mat, t_offset_col).squeeze(-1)  # (N, 3)
        link7_pos_w = wrist_pos_w - offset_world

        return link7_pos_w, link7_quat_w
