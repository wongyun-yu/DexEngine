"""Loading side of the trajectory manager's body reference: per-clip body positions, quaternions and angular
velocities read from body_positions.npz, padded, checked and moved to the device (runs once at init)."""

from __future__ import annotations

import os

import numpy as np
import torch

from dexengine.rl.tasks.mdp.trajectory.manager_cfg import _DATASET_TO_ISAAC_BODY_MAP, _resolve_fk_finger_indices


class BodyLoadMixin:
    """Loads the dataset body arrays into the trajectory manager (the per-step lookup is BodyPositionMixin)."""

    def _load_body_positions(
        self,
        data_dir: str,
        isaac_right_body_names: list[str],
        isaac_left_body_names: list[str],
    ) -> None:
        """Load pre-computed body positions from MuJoCo FK and build body name mappings.

        Body positions are stored at ``{data_dir}/{sequence_name}/0/body_positions.npz``.
        Each NPZ contains per-frame body positions, velocities, and fingertip positions
        computed via MuJoCo forward kinematics in the retargeting pipeline.

        This method:
        1. Loads body positions for all clips
        2. Builds a permutation mapping MuJoCo FK body names to Isaac Lab body indices
        3. Pre-pads to max_length and stacks into GPU tensors
        4. Pre-applies USD-baked rotation so references match PhysX body_link_pos_w

        Args:
            data_dir: Root dataset directory.
            isaac_right_body_names: Body names from Isaac Lab right hand articulation.
            isaac_left_body_names: Body names from Isaac Lab left hand articulation.
        """
        seq_names = self._loader.sequence_names
        max_len = self._loader.max_length
        num_clips = self._loader.num_clips

        (
            first_npz_path,
            has_quat,
            left_body_pos_all,
            left_body_vel_all,
            left_tip_pos_all,
            n_dataset_bodies,
            right_body_pos_all,
            right_body_vel_all,
            right_tip_pos_all,
        ) = self._load_first_clip_layout(
            data_dir, isaac_right_body_names, isaac_left_body_names, max_len, num_clips, seq_names
        )
        if not has_quat:
            # Re-check by loading
            check_npz = np.load(first_npz_path)
            has_quat = "right_hand_body_quat" in check_npz.files
            check_npz.close()

        self._load_body_arrays(
            data_dir,
            has_quat,
            left_body_pos_all,
            left_body_vel_all,
            left_tip_pos_all,
            max_len,
            n_dataset_bodies,
            num_clips,
            right_body_pos_all,
            right_body_vel_all,
            right_tip_pos_all,
            seq_names,
        )

    def _load_first_clip_layout(
        self, data_dir, isaac_right_body_names, isaac_left_body_names, max_len, num_clips, seq_names
    ):
        """Load first clip to get body names and dimensions."""
        first_npz_path = os.path.join(data_dir, seq_names[0], "0", "body_positions.npz")
        first_npz = np.load(first_npz_path, allow_pickle=True)
        n_dataset_bodies, n_tips = self._dataset_body_counts(isaac_right_body_names, isaac_left_body_names, first_npz)

        # Pre-allocate numpy arrays for all clips
        right_body_pos_all = np.zeros((num_clips, max_len, n_dataset_bodies, 3), dtype=np.float32)
        left_body_pos_all = np.zeros((num_clips, max_len, n_dataset_bodies, 3), dtype=np.float32)
        right_body_vel_all = np.zeros((num_clips, max_len, n_dataset_bodies, 3), dtype=np.float32)
        left_body_vel_all = np.zeros((num_clips, max_len, n_dataset_bodies, 3), dtype=np.float32)
        right_tip_pos_all = np.zeros((num_clips, max_len, n_tips, 3), dtype=np.float32)
        left_tip_pos_all = np.zeros((num_clips, max_len, n_tips, 3), dtype=np.float32)

        # Check if first NPZ has quaternion data (backward compat with older precompute)
        has_quat = "right_hand_body_quat" in first_npz.files if hasattr(first_npz, "files") else False
        return (
            first_npz_path,
            has_quat,
            left_body_pos_all,
            left_body_vel_all,
            left_tip_pos_all,
            n_dataset_bodies,
            right_body_pos_all,
            right_body_vel_all,
            right_tip_pos_all,
        )

    def _dataset_body_counts(self, isaac_right_body_names, isaac_left_body_names, first_npz):
        """Dataset body and fingertip counts of the first clip."""
        dataset_right_body_names = first_npz["right_hand_body_names"].tolist()
        dataset_left_body_names = first_npz["left_hand_body_names"].tolist()
        dataset_right_tip_names = first_npz["right_fingertip_names"].tolist()
        n_dataset_bodies = len(dataset_right_body_names)
        n_tips = len(dataset_right_tip_names)
        first_npz.close()

        print(f"[TrajectoryManager] MuJoCo FK body names (right, {n_dataset_bodies}): {dataset_right_body_names}")
        print(f"[TrajectoryManager] MuJoCo FK tip names (right, {n_tips}): {dataset_right_tip_names}")
        print(
            f"[TrajectoryManager] Isaac Lab body names (right, {len(isaac_right_body_names)}): {isaac_right_body_names}"
        )

        # Build body name mapping: MuJoCo FK -> Isaac Lab body indices (Rule 4: runtime)
        # MuJoCo FK body names don't include wrist chain bodies (pos_x through rot_y).
        # We map only bodies that exist in both systems.
        self._right_body_ids_isaac = self._build_body_mapping(
            dataset_right_body_names,
            isaac_right_body_names,
            "right",
            self._fk_body_aliases,
            self._fk_palm_isaac_right,
        )
        try:
            self._left_body_ids_isaac = self._build_body_mapping(
                dataset_left_body_names,
                isaac_left_body_names,
                "left",
                self._fk_body_aliases,
                self._fk_palm_isaac_left,
            )
        except ValueError:
            # Left hand might be a placeholder (same USD as right hand)
            # Use right hand mapping as fallback
            self._left_body_ids_isaac = self._right_body_ids_isaac
            print("[TrajectoryManager] Left hand body mapping failed — using right hand IDs as fallback")
        self._n_matched_bodies = len(self._right_body_ids_isaac)
        self._n_tips = n_tips

        # FK body-name exposure + name-resolved finger indices for non-standard
        # layouts. For the legacy 13-body RH56F1 layout this stays None so all
        # downstream consumers keep their hardcoded index tables (drift
        # invariant). SharpA (23 bodies) resolves tips/level1/level2 by name.
        self.fk_body_names_right = list(dataset_right_body_names)
        self.fk_finger_indices = None
        if n_dataset_bodies != 13:
            self.fk_finger_indices = _resolve_fk_finger_indices(dataset_right_body_names, self._fk_tip_body_names)
            print(
                f"[TrajectoryManager] FK finger indices (name-resolved, "
                f"{n_dataset_bodies} bodies): {self.fk_finger_indices}"
            )

        print(f"[TrajectoryManager] Matched {self._n_matched_bodies} bodies per hand")
        print(f"[TrajectoryManager] Right body Isaac IDs: {self._right_body_ids_isaac}")
        print(f"[TrajectoryManager] Left body Isaac IDs: {self._left_body_ids_isaac}")
        return n_dataset_bodies, n_tips

    def _load_body_arrays(
        self,
        data_dir,
        has_quat,
        left_body_pos_all,
        left_body_vel_all,
        left_tip_pos_all,
        max_len,
        n_dataset_bodies,
        num_clips,
        right_body_pos_all,
        right_body_vel_all,
        right_tip_pos_all,
        seq_names,
    ):
        """Allocate the quaternion / angular-velocity arrays, load every clip, pad and move to the device (statements
        unchanged)."""
        left_body_ang_vel_all, left_body_quat_all, right_body_ang_vel_all, right_body_quat_all = (
            self._alloc_quat_arrays(has_quat, max_len, n_dataset_bodies, num_clips)
        )

        # Load body positions for all clips
        for i, seq_name in enumerate(seq_names):
            if (i + 1) % 100 == 0 or (i + 1) == num_clips:
                print(f"  Loading body positions: {i + 1}/{num_clips}")

            npz_path = os.path.join(data_dir, seq_name, "0", "body_positions.npz")
            npz = np.load(npz_path)
            T = npz["right_hand_body_pos"].shape[0]

            self._check_equal_side_lengths(
                left_body_pos_all,
                left_body_vel_all,
                left_tip_pos_all,
                max_len,
                right_body_pos_all,
                right_body_vel_all,
                right_tip_pos_all,
                T,
                i,
                npz,
                seq_name,
            )

            # Load quaternion data if available
            if has_quat:
                for arr_all, key in [
                    (right_body_quat_all, "right_hand_body_quat"),
                    (left_body_quat_all, "left_hand_body_quat"),
                    (right_body_ang_vel_all, "right_hand_body_ang_vel"),
                    (left_body_ang_vel_all, "left_hand_body_ang_vel"),
                ]:
                    data = npz[key]
                    arr_all[i, :T] = data
                    if max_len > T:
                        arr_all[i, T:] = data[-1]

            npz.close()

        self._to_gpu_tensors(
            has_quat,
            left_body_pos_all,
            left_body_vel_all,
            left_tip_pos_all,
            num_clips,
            right_body_pos_all,
            right_body_vel_all,
            right_tip_pos_all,
        )
        if has_quat:
            self._right_body_quat = torch.tensor(right_body_quat_all, device=self.device)
            self._left_body_quat = torch.tensor(left_body_quat_all, device=self.device)
            self._right_body_ang_vel = torch.tensor(right_body_ang_vel_all, device=self.device)
            self._left_body_ang_vel = torch.tensor(left_body_ang_vel_all, device=self.device)

        self._root_rotation_identity_check(has_quat)

    def _alloc_quat_arrays(self, has_quat, max_len, n_dataset_bodies, num_clips):
        """Identity-padded quaternion and angular-velocity arrays when the NPZ files carry them; otherwise all None."""
        if has_quat:
            right_body_quat_all = np.zeros((num_clips, max_len, n_dataset_bodies, 4), dtype=np.float32)
            left_body_quat_all = np.zeros((num_clips, max_len, n_dataset_bodies, 4), dtype=np.float32)
            right_body_ang_vel_all = np.zeros((num_clips, max_len, n_dataset_bodies, 3), dtype=np.float32)
            left_body_ang_vel_all = np.zeros((num_clips, max_len, n_dataset_bodies, 3), dtype=np.float32)
            # Initialize quaternions to identity (w=1) for padding
            right_body_quat_all[..., 0] = 1.0
            left_body_quat_all[..., 0] = 1.0
            print("[TrajectoryManager] Body quaternions and angular velocities available")
        else:
            print(
                "[TrajectoryManager] WARNING: No body quaternions in NPZ files. "
                "Re-run dexengine/placement/fk/body_positions.py to enable wrist orientation tracking."
            )
            # read only under `if has_quat` below; bound so the steps can be helpers
            right_body_quat_all = left_body_quat_all = right_body_ang_vel_all = left_body_ang_vel_all = None
        return left_body_ang_vel_all, left_body_quat_all, right_body_ang_vel_all, right_body_quat_all

    def _to_gpu_tensors(
        self,
        has_quat,
        left_body_pos_all,
        left_body_vel_all,
        left_tip_pos_all,
        num_clips,
        right_body_pos_all,
        right_body_vel_all,
        right_tip_pos_all,
    ):
        """Convert to GPU tensors."""
        _n_dz = int(sum(1 for _c in range(num_clips) if float(self._loader.clip_dz_of(_c)) != 0.0))
        if _n_dz:
            print(
                f"[DESK_Z] body_positions·wrist_poses per-clip z shift applied: clips {_n_dz}/{num_clips} (the "
                f"loader's dz — body, fingertip and wrist references)",
                flush=True,
            )
        self._right_body_pos = torch.tensor(right_body_pos_all, device=self.device)
        self._left_body_pos = torch.tensor(left_body_pos_all, device=self.device)
        self._right_body_vel = torch.tensor(right_body_vel_all, device=self.device)
        self._left_body_vel = torch.tensor(left_body_vel_all, device=self.device)
        self._right_tip_pos = torch.tensor(right_tip_pos_all, device=self.device)
        self._left_tip_pos = torch.tensor(left_tip_pos_all, device=self.device)

        self._has_body_orientations = has_quat

    def _root_rotation_identity_check(self, has_quat):
        """Rule 13: With root=y_180_x_m90 in mdp/events, the double rotation (root_quat * converter_baked_rotation) =
        identity, so PhysX body_link_pos_w directly equals MuJoCo FK positions. No pre-rotation needed.
        """
        tensors_for_size = [
            self._right_body_pos,
            self._left_body_pos,
            self._right_body_vel,
            self._left_body_vel,
            self._right_tip_pos,
            self._left_tip_pos,
        ]
        if has_quat:
            tensors_for_size.extend(
                [
                    self._right_body_quat,
                    self._left_body_quat,
                    self._right_body_ang_vel,
                    self._left_body_ang_vel,
                ]
            )
        tensor_gb = sum(t.nelement() * 4 for t in tensors_for_size) / (1024**3)
        print(f"[TrajectoryManager] Body position tensors: {tensor_gb:.3f} GB")

    def _check_equal_side_lengths(
        self,
        left_body_pos_all,
        left_body_vel_all,
        left_tip_pos_all,
        max_len,
        right_body_pos_all,
        right_body_vel_all,
        right_tip_pos_all,
        T,
        i,
        npz,
        seq_name,
    ):
        """Bimanual / left-passive REQUIRE equal-length right+left trajectories: the episode horizon
        (is_trajectory_done) + frame_progress_pct use the RIGHT clip_lengths for BOTH hands, and the padding loop
        below indexes left arrays with the RIGHT length T. Fail loud if a future clip ever pairs unequal-length hands
        (would silently truncate/misalign the left trajectory).
        """
        if "left_hand_body_pos" in npz.files:
            T_left = npz["left_hand_body_pos"].shape[0]
            assert T_left == T, (
                f"[TrajectoryManager] clip '{seq_name}': left trajectory length ({T_left}) "
                f"!= right ({T}). Bimanual/left-passive require equal-length right+left clips "
                f"(the shared RIGHT clip_lengths drives the episode horizon, frame_progress, "
                f"and the padding below). Re-pair or pad the left clip to match the right."
            )

        # Load and pad to max_length (same pattern as trajectory loader)
        for arr_all, key in [
            (right_body_pos_all, "right_hand_body_pos"),
            (left_body_pos_all, "left_hand_body_pos"),
            (right_body_vel_all, "right_hand_body_vel"),
            (left_body_vel_all, "left_hand_body_vel"),
            (right_tip_pos_all, "right_fingertip_pos"),
            (left_tip_pos_all, "left_fingertip_pos"),
        ]:
            data = npz[key]
            if key.endswith("_pos"):
                _dz_i = float(self._loader.clip_dz_of(i))
                if _dz_i != 0.0:  # per-clip z shift: body and fingertip references on the same plane (the loader's dz)
                    data = data.copy()
                    data[..., 2] += _dz_i
            arr_all[i, :T] = data
            if max_len > T:
                arr_all[i, T:] = data[-1]  # repeat last frame

    @staticmethod
    def _build_body_mapping(
        dataset_body_names: list[str],
        isaac_body_names: list[str],
        side: str,
        aliases: tuple[tuple[str, str], ...] = (),
        palm_isaac_body: str = "",
    ) -> list[int]:
        """Build mapping from MuJoCo FK body order to Isaac Lab body indices.

        For each MuJoCo FK body name, find the corresponding Isaac Lab body index.
        ``{side}_hand_base`` maps to Isaac Lab ``{side}_wrist_rot_z``.
        All finger body names are identical in both systems.

        Args:
            dataset_body_names: Body names from MuJoCo FK NPZ (13 per hand).
            isaac_body_names: Body names from Isaac Lab articulation.
            side: "right" or "left".

        Returns:
            List of Isaac Lab body indices, one per MuJoCo FK body, in dataset FK order.
        """
        isaac_ids = []
        # FK body 0 = the palm (an export convention every hand shares). The palm hangs off the arm EE through a fixed
        # joint and the importer merges it into the EE, so its name disappears: link it **by position (0)**, not by
        # name. The EE name changes with the arm anyway (rby1: link_right_arm_6), so a name table could not follow.
        if palm_isaac_body and dataset_body_names and palm_isaac_body in isaac_body_names:
            aliases = tuple(aliases) + ((dataset_body_names[0], palm_isaac_body),)
        # The spec's aliases come first: they link bodies the importer merged away (the palm, ...).
        _alias = dict(aliases)
        for dataset_name in dataset_body_names:
            # Check maps FIRST (handles right_hand_base -> right_wrist_rot_z), then direct name
            # Try: spec alias → dataset map → direct name
            if _alias.get(dataset_name) in isaac_body_names:
                isaac_idx = isaac_body_names.index(_alias[dataset_name])
            elif _DATASET_TO_ISAAC_BODY_MAP.get(dataset_name) in isaac_body_names:
                isaac_idx = isaac_body_names.index(_DATASET_TO_ISAAC_BODY_MAP[dataset_name])
            elif dataset_name in isaac_body_names:
                # Direct name match (for finger bodies that have identical names)
                isaac_idx = isaac_body_names.index(dataset_name)
            else:
                # Placeholder left hand fallback: try right_ prefix
                fallback = dataset_name.replace("left_", "right_", 1)
                if fallback in isaac_body_names:
                    isaac_idx = isaac_body_names.index(fallback)
                else:
                    raise ValueError(
                        f"MuJoCo FK body '{dataset_name}' not found in Isaac Lab body names: {isaac_body_names}"
                    )
            isaac_ids.append(isaac_idx)
        return isaac_ids
