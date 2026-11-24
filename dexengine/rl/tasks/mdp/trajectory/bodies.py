"""Reference body and fingertip positions from the precomputed forward kinematics: the per-step lookup (the loading
side is BodyLoadMixin in bodies_load.py)."""

from __future__ import annotations

import torch


class BodyPositionMixin:
    """Methods of :class:`TrajectoryManager` (reference body and fingertip positions from the precomputed forward
    kinematics)."""

    def get_reference_body_pos(self, frame_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Get reference body positions/velocities/orientations for all envs.

        Results are cached per step — multiple calls with the same frame_ids
        return the cached result without recomputing.

        Positions are pre-rotated to match the PhysX body_link_pos_w frame
        (rotation baked into joint axes by MJCF→USD converter).

        Args:
            frame_ids: Frame indices of shape (num_envs,). Typically env.episode_length_buf.
                RSI start_frames offset is added internally.

        Returns:
            Dictionary with keys:
            - right_body_pos: (num_envs, n_bodies, 3) in PhysX env-local frame
            - left_body_pos: (num_envs, n_bodies, 3)
            - right_body_vel: (num_envs, n_bodies, 3) rotated to PhysX frame
            - left_body_vel: (num_envs, n_bodies, 3)
            - right_tip_pos: (num_envs, n_tips, 3) in PhysX env-local frame
            - left_tip_pos: (num_envs, n_tips, 3)
            - right_body_quat: (num_envs, n_bodies, 4) wxyz (if available)
            - left_body_quat: (num_envs, n_bodies, 4) wxyz (if available)
            - right_body_ang_vel: (num_envs, n_bodies, 3) (if available)
            - left_body_ang_vel: (num_envs, n_bodies, 3) (if available)
        """
        return self._cached_body_pos(frame_ids)

    def _cached_body_pos(self, frame_ids):
        """Per-step cache: avoids redundant indexing when called by multiple reward terms A changed sum empties the
        whole cache (as before); an equal sum with different real inputs is not a hit.
        """
        sig = self._ref_input_sig(frame_ids)
        env_step, e_body = self._cache_probe(frame_ids, sig, self._cache_body_sig)
        self._check_cache_valid(env_step)
        if self._cache_body_pos is not None:
            if e_body:
                return self._cache_body_pos
            self._ref_cache_collision("get_reference_body_pos", env_step)
            self._cache_body_pos = self._cache_state = self._cache_state_p1 = None
            self._cache_state_key = self._cache_state_p1_key = -1

        # Slowdown-aware: floor(frame_ids / slowdown) + start, clamped per-clip.
        # PER-SIDE CLOCK (reset_scheme=decoupled): right_* tables read the right
        # clock's frames, left_* the left clock's. Disabled -> both identical (legacy).
        # (sig already holds that frame; it is not recomputed by the same function)
        clamped_r = sig[1]
        clamped_l = sig[2] if self.per_side_clock else sig[1]
        clip_ids = self.clip_ids

        result = {
            "right_body_pos": self._right_body_pos[clip_ids, clamped_r],
            "left_body_pos": self._left_body_pos[clip_ids, clamped_l],
            "right_body_vel": self._right_body_vel[clip_ids, clamped_r],
            "left_body_vel": self._left_body_vel[clip_ids, clamped_l],
            "right_tip_pos": self._right_tip_pos[clip_ids, clamped_r],
            "left_tip_pos": self._left_tip_pos[clip_ids, clamped_l],
        }

        if self._has_body_orientations:
            result["right_body_quat"] = self._right_body_quat[clip_ids, clamped_r]
            result["left_body_quat"] = self._left_body_quat[clip_ids, clamped_l]
            result["right_body_ang_vel"] = self._right_body_ang_vel[clip_ids, clamped_r]
            result["left_body_ang_vel"] = self._left_body_ang_vel[clip_ids, clamped_l]

        # Apply coordinate rotation if enabled
        if self._rotate:
            for key in ["right_body_pos", "left_body_pos", "right_tip_pos", "left_tip_pos"]:
                # (N, n_bodies, 3) → rotate each body position
                orig = result[key]
                shape = orig.shape  # (N, B, 3)
                flat = orig.reshape(-1, 3)
                rotated = flat @ self._rz_mat.T + self._z_offset
                result[key] = rotated.reshape(shape)
            for key in ["right_body_vel", "left_body_vel"]:
                # Velocities: rotate direction only (no Z offset)
                orig = result[key]
                shape = orig.shape
                flat = orig.reshape(-1, 3)
                result[key] = (flat @ self._rz_mat.T).reshape(shape)
            if self._has_body_orientations:
                for key in ["right_body_quat", "left_body_quat"]:
                    orig = result[key]  # (N, B, 4) wxyz
                    shape = orig.shape
                    flat = orig.reshape(-1, 4)
                    result[key] = self._rotate_quat_batch(flat).reshape(shape)
                for key in ["right_body_ang_vel", "left_body_ang_vel"]:
                    orig = result[key]
                    shape = orig.shape
                    flat = orig.reshape(-1, 3)
                    result[key] = (flat @ self._rz_mat.T).reshape(shape)

        self._cache_body_pos = result
        self._cache_body_sig = sig
        return result

    @property
    def n_body_positions(self) -> int:
        """Number of matched body positions per hand."""
        return self._n_matched_bodies

    @property
    def has_body_orientations(self) -> bool:
        """Whether body quaternions and angular velocities are available."""
        return self._has_body_orientations

    @property
    def right_body_ids_isaac(self) -> list[int]:
        """Isaac Lab body indices for right hand (matching stored body position order)."""
        return self._right_body_ids_isaac

    @property
    def left_body_ids_isaac(self) -> list[int]:
        """Isaac Lab body indices for left hand (matching stored body position order)."""
        return self._left_body_ids_isaac
