"""Reference wrist poses: loading, pose corrections, interpolation and the dataset-to-robot wrist transform."""

from __future__ import annotations

import os

import numpy as np
import torch


class WristPoseMixin:
    """Methods of :class:`TrajectoryManager` (reference wrist poses: loading, pose corrections, interpolation and the
    dataset-to-robot wrist transform)."""

    def _load_pose_corrections(self, data_dir: str) -> dict[int, torch.Tensor]:
        """Load initial pose corrections from penetration optimization.

        Corrections are 18-dim dataset-order qpos adjustments for frame 0.
        Applied to sequences that had finger-object penetration at start.

        Returns:
            Dict mapping clip_index -> 18-dim correction tensor in dataset order.
            Empty dict if no corrections file exists.
        """
        import json

        # Coupling constants for expanding 12 independent DOFs to 18
        FINGER_JOINT_INDICES = [6, 7, 10, 12, 14, 16]
        COUPLING = {
            8: (7, 1.2953),
            9: (8, 0.8962),
            11: (10, 1.1545),
            13: (12, 1.1545),
            15: (14, 1.1545),
            17: (16, 1.1545),
        }

        corrections_path = os.path.join(data_dir, "initial_pose_corrections.json")
        if not os.path.exists(corrections_path):
            return {}

        with open(corrections_path) as f:
            raw_corrections = json.load(f)

        # Map sequence names to clip indices
        seq_names = self._loader.sequence_names
        result = {}
        n_loaded = 0
        for seq_name, dq12_list in raw_corrections.items():
            if seq_name in seq_names:
                clip_idx = seq_names.index(seq_name)
                dq12 = np.array(dq12_list)

                # Expand 12-dim to 18-dim dataset qpos correction
                dq18 = np.zeros(18)
                dq18[:6] = dq12[:6]  # Wrist
                for i, qi in enumerate(FINGER_JOINT_INDICES):
                    dq18[qi] = dq12[6 + i]
                # Apply coupling
                for follower, (driver, ratio) in COUPLING.items():
                    dq18[follower] = ratio * dq18[driver]

                result[clip_idx] = torch.tensor(dq18, dtype=torch.float32, device=self.device)
                n_loaded += 1

        if n_loaded > 0:
            print(f"[TrajectoryManager] Loaded {n_loaded} initial pose corrections")

        # Build a (num_clips, 18) tensor for fast vectorized lookup
        # Clips without corrections get zeros (no change)
        num_clips = self._loader.num_clips
        correction_table = torch.zeros(num_clips, 18, dtype=torch.float32, device=self.device)
        for clip_idx, dq18 in result.items():
            correction_table[clip_idx] = dq18
        return correction_table

    def _apply_pose_corrections(
        self,
        rh_qpos_dataset: torch.Tensor,
        frame_ids: torch.Tensor,
        clip_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply initial pose corrections to frame-0 sequences (vectorized).

        Returns input unchanged if pose_corrections is disabled.
        """
        if not self._use_pose_corrections:
            return rh_qpos_dataset
        # Mask: only apply at frame 0
        frame0_mask = (frame_ids == 0).unsqueeze(-1)  # (N, 1)
        corrections = self._pose_corrections[clip_ids]  # (N, 18)
        return rh_qpos_dataset + corrections * frame0_mask

    def _load_wrist_poses(self, wrist_poses_dir_right: str, traj_offset_z: float):
        """Load pre-dumped wrist world poses for the arm IK.

        Loads per-sequence .npz files, applies Rz(-90°) + table height transform,
        pads to max_length, and stacks into GPU tensors matching clip ordering.

        Args:
            wrist_poses_dir_right: Directory with {sequence_name}.npz files.
            traj_offset_z: Table height offset (e.g. 0.76 for desk surface).

        Returns:
            (wrist_poses, wrist_quats): tensors (num_clips, max_length, {3,4 wxyz}).
        """
        from scipy.spatial.transform import Rotation

        seq_names = self._loader.sequence_names
        num_clips = self._loader.num_clips
        max_length = self._loader.max_length
        rz = Rotation.from_euler("z", -90, degrees=True)
        traj_offset = np.array([0.0, 0.0, traj_offset_z])

        # Pre-allocate padded arrays
        all_pos = np.zeros((num_clips, max_length, 3), dtype=np.float32)
        all_quat = np.zeros((num_clips, max_length, 4), dtype=np.float32)
        # Default identity quaternion for padding
        all_quat[:, :, 0] = 1.0  # w=1

        n_loaded = 0
        n_missing = 0
        for clip_idx, seq_name in enumerate(seq_names):
            npz_path = os.path.join(wrist_poses_dir_right, f"{seq_name}.npz")
            if not os.path.exists(npz_path):
                n_missing += 1
                continue

            data = np.load(npz_path)
            raw_pos = data["positions"]  # (N, 3) replay convention
            raw_quat = data["quaternions"]  # (N, 4) wxyz

            # Transform: Rz(-90°) + table height offset
            pos = rz.apply(raw_pos) + traj_offset
            # per-clip z shift (the loader's dz: desk support, aerial z_off; on the same plane as the reference wrist
            # and objects)
            pos[:, 2] += float(self._loader.clip_dz_of(clip_idx))

            # Transform quaternions — vectorized equivalent of the old per-frame loop:
            # wxyz->xyzw, left-multiply by rz, back to wxyz. scipy Rotation applies the
            # (T,4) batch with the same ops as the loop (verified numerically identical).
            T = len(raw_pos)
            r_raw = Rotation.from_quat(raw_quat[:, [1, 2, 3, 0]])  # wxyz -> xyzw, batched
            q_xyzw = (rz * r_raw).as_quat()  # (T, 4) xyzw
            quat = q_xyzw[:, [3, 0, 1, 2]].astype(np.float32)  # xyzw -> wxyz

            # Pad to max_length
            clip_len = min(T, max_length)
            all_pos[clip_idx, :clip_len] = pos[:clip_len]
            all_quat[clip_idx, :clip_len] = quat[:clip_len]
            # Pad with last frame
            if clip_len < max_length:
                all_pos[clip_idx, clip_len:] = pos[clip_len - 1]
                all_quat[clip_idx, clip_len:] = quat[clip_len - 1]

            n_loaded += 1

        if n_missing > 0:
            # No silent corruption: a missing clip would train against a zero-field wrist reference ("following the
            # reference but no reward"). The arm-IK cache already fails hard (events); this removes the asymmetry of a
            # warning for the wrist only.
            msg = (
                f"[TrajectoryManager] wrist pose missing {n_missing}/{num_clips} "
                f"in {wrist_poses_dir_right} — no silent zero-field run"
            )
            raise ValueError(msg)

        wrist_poses = torch.tensor(all_pos, device=self.device)  # (num_clips, max_length, 3)
        wrist_quats = torch.tensor(all_quat, device=self.device)  # (num_clips, max_length, 4) wxyz
        print(f"[TrajectoryManager] Loaded wrist poses for {n_loaded}/{num_clips} clips from {wrist_poses_dir_right}")
        return wrist_poses, wrist_quats

    def get_wrist_pos_w(self, frame_ids: torch.Tensor | None = None, side: str = "right") -> torch.Tensor:
        """Get reference wrist world positions for all envs.

        Coordinate frame: WORLD frame (env-shared), after the Rz(-90°) +
        traj_offset_z transform applied by ``_load_wrist_poses`` to the
        raw NPZ (verified per frame).

        Args:
            frame_ids: Frame indices (num_envs,). If None, uses start_frames.
                When ``slowdown_factor != 1.0``, ``frame_ids`` are policy-step
                indices; the effective trajectory frame is ``frame_ids /
                slowdown_factor`` with linear interpolation between floor/ceil.
            side: "right" (default) or "left" — selects which hand's wrist table.
                Bimanual left arm IK passes side='left'.

        Returns:
            Wrist positions (num_envs, 3) in world frame.
        """
        poses = self._left_wrist_poses if side == "left" else self._wrist_poses
        if poses is None:
            raise RuntimeError(
                f"Wrist poses not loaded for side='{side}'. Set "
                f"wrist_poses_dir_right{'_left' if side == 'left' else ''} in TrajectoryManagerCfg "
                f"(left auto-derives the sibling '<wrist_poses_dir_right>_left')."
            )
        if frame_ids is None:
            absolute_frames = self.start_frames.float()
        elif self.per_side_clock:
            absolute_frames = self.side_abs_frames_float(side, frame_ids)
        else:
            absolute_frames = (frame_ids.float() / self._slowdown_factor) + self.start_frames.float()
        if self._slowdown_factor == 1.0:
            return poses[self.clip_ids, absolute_frames.long()]
        return self._interp_pos(poses, absolute_frames)

    def get_wrist_quat_w(self, frame_ids: torch.Tensor | None = None, side: str = "right") -> torch.Tensor:
        """Get reference wrist world quaternions for all envs.

        Coordinate frame: WORLD frame, wxyz, after the Rz(-90°) transform.

        Args:
            frame_ids: Frame indices (num_envs,). If None, uses start_frames.
                Slow-down: see ``get_wrist_pos_w``. Quaternions use Slerp
                between floor/ceil frames.
            side: "right" (default) or "left" — selects which hand's wrist table.

        Returns:
            Wrist quaternions (num_envs, 4) wxyz in world frame.
        """
        quats = self._left_wrist_quats if side == "left" else self._wrist_quats
        if quats is None:
            raise RuntimeError(
                f"Wrist quats not loaded for side='{side}'. Set "
                f"wrist_poses_dir_right{'_left' if side == 'left' else ''} in TrajectoryManagerCfg."
            )
        if frame_ids is None:
            absolute_frames = self.start_frames.float()
        elif self.per_side_clock:
            absolute_frames = self.side_abs_frames_float(side, frame_ids)
        else:
            absolute_frames = (frame_ids.float() / self._slowdown_factor) + self.start_frames.float()
        if self._slowdown_factor == 1.0:
            return quats[self.clip_ids, absolute_frames.long()]
        return self._interp_quat(quats, absolute_frames)

    def _interp_pos(self, table: torch.Tensor, abs_frames: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between floor/ceil frames.

        Args:
            table: (num_clips, max_length, D) — per-clip per-frame values.
            abs_frames: (num_envs,) float absolute frame indices.
        Returns:
            (num_envs, D) interpolated values.
        """
        max_idx = table.shape[1] - 1
        f_lo = abs_frames.floor().clamp(0, max_idx).long()
        f_hi = (f_lo + 1).clamp(0, max_idx)
        alpha = (abs_frames - f_lo.float()).clamp(0.0, 1.0).unsqueeze(-1)  # (N, 1)
        v_lo = table[self.clip_ids, f_lo]
        v_hi = table[self.clip_ids, f_hi]
        return (1.0 - alpha) * v_lo + alpha * v_hi

    def _interp_quat(self, table: torch.Tensor, abs_frames: torch.Tensor) -> torch.Tensor:
        """Slerp between floor/ceil quaternions (wxyz).

        Uses geometric slerp via half-angle formula. Quaternions in same
        hemisphere by sign-flip on negative dot product before interpolation.
        """
        max_idx = table.shape[1] - 1
        f_lo = abs_frames.floor().clamp(0, max_idx).long()
        f_hi = (f_lo + 1).clamp(0, max_idx)
        alpha = (abs_frames - f_lo.float()).clamp(0.0, 1.0)  # (N,)
        q_lo = table[self.clip_ids, f_lo]  # (N, 4)
        q_hi = table[self.clip_ids, f_hi]
        # Ensure shortest-path slerp
        dot = (q_lo * q_hi).sum(dim=-1, keepdim=True)
        q_hi = torch.where(dot < 0, -q_hi, q_hi)
        dot = dot.abs().clamp(-1.0, 1.0)
        # When dot is very close to 1, fall back to linear interp + normalize
        omega = torch.acos(dot)
        sin_om = torch.sin(omega)
        # Safe slerp; for small omega use linear approx
        near_one = sin_om.abs() < 1e-5
        s0 = torch.where(near_one, 1.0 - alpha.unsqueeze(-1), torch.sin((1.0 - alpha.unsqueeze(-1)) * omega) / sin_om)
        s1 = torch.where(near_one, alpha.unsqueeze(-1), torch.sin(alpha.unsqueeze(-1) * omega) / sin_om)
        out = s0 * q_lo + s1 * q_hi
        return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def _transform_wrist_dataset_to_robot(
        self,
        qpos_dataset: torch.Tensor,
        frame_ids: torch.Tensor | None = None,
        clip_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Transform wrist rotation joints from the dataset convention to the URDF-hand convention.

        The dataset MJCF hand has a baked body orientation (Rx(90°)*Rz(180°) relative
        to the URDF hand) that is included in the dataset replay world poses
        but not in the raw joint values. This correction must be applied so that the
        URDF floating hand (with Rz(-90°) root) produces the same world-space
        palm orientation as the step5 full-arm IK reference.

        Corrected transform (verified numerically: 0.00005° max error across all frames):
          R = Rx(90°) * Rz(180°) * Rx(-rx) * Rz(-ry) * Ry(-rz)
          (rot_x', rot_y', rot_z') = R.as_euler('XYZ')

        Position joints (0:3) and finger joints (6 to 17) transfer directly.

        Args:
            qpos_dataset: (N, 18) joint values in dataset order
                [0:3] = pos_x, pos_y, pos_z (unchanged)
                [3:6] = rot_x, rot_y, rot_z (transformed)
                [6, 18) = finger joints (unchanged)
            frame_ids: Unused (kept for API compatibility).
            clip_ids: Unused (kept for API compatibility).

        Returns:
            (N, 18) transformed joint values for the URDF hand
        """
        qpos = qpos_dataset.clone()
        rx = qpos_dataset[:, 3]  # (N,)
        ry = qpos_dataset[:, 4]
        rz = qpos_dataset[:, 5]

        # Build rotation matrices on GPU (no CPU transfer).
        # MuJoCo FK: R_dataset = Rx(-rx) * Rz(-ry) * Ry(-rz)
        # Correction: R_correction = Rx(90°) * Rz(180°) (constant, pre-computed)
        # Combined: R = R_correction * R_dataset

        # Pre-compute trig values
        c_rx, s_rx = torch.cos(-rx), torch.sin(-rx)
        c_ry, s_ry = torch.cos(-ry), torch.sin(-ry)
        c_rz, s_rz = torch.cos(-rz), torch.sin(-rz)

        # Rx(-rx): rotation about X
        # [[1, 0, 0], [0, c, -s], [0, s, c]]
        # Rz(-ry): rotation about Z
        # [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        # Ry(-rz): rotation about Y
        # [[c, 0, s], [0, 1, 0], [-s, 0, c]]

        # R_dataset = Rx * Rz * Ry (matrix multiply, all batched)
        N = rx.shape[0]
        zero = torch.zeros(N, device=rx.device, dtype=rx.dtype)
        one = torch.ones(N, device=rx.device, dtype=rx.dtype)

        euler_x, euler_y, euler_z = self._rotation_x(N, c_rx, c_ry, c_rz, one, rx, s_rx, s_ry, s_rz, zero)

        qpos[:, 3] = euler_x
        qpos[:, 4] = euler_y
        qpos[:, 5] = euler_z
        return qpos

    def _rotation_x(self, N, c_rx, c_ry, c_rz, one, rx, s_rx, s_ry, s_rz, zero):
        """Build Rx(-rx)."""
        Rx = torch.stack(
            [
                one,
                zero,
                zero,
                zero,
                c_rx,
                -s_rx,
                zero,
                s_rx,
                c_rx,
            ],
            dim=-1,
        ).reshape(N, 3, 3)

        # Build Rz(-ry)
        Rz = torch.stack(
            [
                c_ry,
                -s_ry,
                zero,
                s_ry,
                c_ry,
                zero,
                zero,
                zero,
                one,
            ],
            dim=-1,
        ).reshape(N, 3, 3)

        # Build Ry(-rz)
        Ry = torch.stack(
            [
                c_rz,
                zero,
                s_rz,
                zero,
                one,
                zero,
                -s_rz,
                zero,
                c_rz,
            ],
            dim=-1,
        ).reshape(N, 3, 3)

        R_dataset = Rx @ Rz @ Ry  # (N, 3, 3)

        # R_correction = Rx(90°) * Rz(180°) — constant, compute once
        if not hasattr(self, "_R_correction"):
            import math

            c90, s90 = math.cos(math.pi / 2), math.sin(math.pi / 2)
            Rx90 = torch.tensor([[1, 0, 0], [0, c90, -s90], [0, s90, c90]], device=rx.device, dtype=rx.dtype)
            c180, s180 = math.cos(math.pi), math.sin(math.pi)
            Rz180 = torch.tensor([[c180, -s180, 0], [s180, c180, 0], [0, 0, 1]], device=rx.device, dtype=rx.dtype)
            self._R_correction = Rx90 @ Rz180  # (3, 3)

        R_corrected = self._R_correction.unsqueeze(0) @ R_dataset  # (N, 3, 3)

        # Extract intrinsic XYZ Euler angles from rotation matrix
        # R = Rx(a) * Ry(b) * Rz(c)
        # R[0,2] = sin(b), R[2,2] = cos(a)*cos(b), R[1,2] = -sin(a)*cos(b)
        # R[0,0] = cos(b)*cos(c), R[0,1] = -cos(b)*sin(c)
        euler_y = torch.asin(R_corrected[:, 0, 2].clamp(-1, 1))
        euler_x = torch.atan2(-R_corrected[:, 1, 2], R_corrected[:, 2, 2])
        euler_z = torch.atan2(-R_corrected[:, 0, 1], R_corrected[:, 0, 0])
        return euler_x, euler_y, euler_z

    def _rotate_pos_batch(self, pos: torch.Tensor) -> torch.Tensor:
        """Apply Rz(-90°) + desk_height to a batch of positions.

        Args:
            pos: (N, 3) positions in +Y forward frame.

        Returns:
            (N, 3) positions in +X forward frame with desk height added.
        """
        # Rz(-90°): (x,y,z) → (y,-x,z)
        rotated = pos @ self._rz_mat.T  # (N, 3)
        return rotated + self._z_offset

    def _rotate_quat_batch(self, quat: torch.Tensor) -> torch.Tensor:
        """Apply Rz(-90°) to a batch of quaternions (wxyz).

        Args:
            quat: (N, 4) quaternions in wxyz format, +Y forward frame.

        Returns:
            (N, 4) quaternions in wxyz format, +X forward frame.
        """
        from isaaclab.utils.math import quat_mul

        rz_expanded = self._rz_quat.unsqueeze(0).expand(quat.shape[0], -1)
        return quat_mul(rz_expanded, quat)
