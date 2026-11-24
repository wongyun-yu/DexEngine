"""Episode clock of the clips: reference state initialization, per-side clocks and clip completion."""

from __future__ import annotations

import torch


class ClipClockMixin:
    """Methods of :class:`TrajectoryManager` (episode clock of the clips: reference state initialization, per-side
    clocks and clip completion)."""

    def _build_permutation(
        self, physx_joint_names: list[str], dataset_order: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build dataset->PhysX permutation tensor.

        Returns:
            (perm, physx_hand_ids) where:
            - perm[i] = dataset_index for the i-th hand joint in PhysX order
            - physx_hand_ids[i] = PhysX joint index for the i-th hand joint

        Gracefully skips joints not in dataset_order (e.g., arm/head joints
        when the full robot articulation is passed instead of hand-only).
        Usage: hand_data_physx_order = dataset_traj[:, perm]
        """
        dataset_set = set(dataset_order)
        perm = []
        physx_hand_ids = []
        for physx_idx, physx_name in enumerate(physx_joint_names):
            if physx_name in dataset_set:
                dataset_idx = dataset_order.index(physx_name)
                perm.append(dataset_idx)
                physx_hand_ids.append(physx_idx)
        if len(perm) != len(dataset_order):
            missing = [n for n in dataset_order if n not in physx_joint_names]
            raise ValueError(
                f"Permutation mismatch: matched {len(perm)}/{len(dataset_order)} joints. Missing from PhysX: {missing}"
            )
        return (
            torch.tensor(perm, dtype=torch.long, device=self.device),
            torch.tensor(physx_hand_ids, dtype=torch.long, device=self.device),
        )

    def _dataset_indep_finger_idx(self) -> list[int]:
        """Indices of the independent finger joints in dataset qpos order.

        A hand whose spec gives the finger order derives it from there: the joints minus the dependent (mimic) ones
        are the DOFs the policy commands. No branching on the hand name. Only the legacy path without a spec uses
        RH56F1's [6, 7, 10, 12, 14, 16] (a fact of that hand's dataset layout, so it belongs here).
        """
        names = getattr(self, "_dataset_finger_names", ())
        if names:
            coupled = set(getattr(self, "_dataset_coupled_names", ()))
            return [6 + i for i, n in enumerate(names) if n not in coupled]
        return [6, 7, 10, 12, 14, 16]

    def reset(self, env_ids: torch.Tensor) -> None:
        """Sample new clips and RSI start frames for reset environments."""
        n = len(env_ids)
        if not self._static_clips:  # multi-object: mesh per env is fixed at spawn
            self.clip_ids[env_ids] = self._loader.sample_clips(n)
            self.clip_lengths[env_ids] = self._loader.get_clip_lengths(self.clip_ids[env_ids])

        # RSI: with probability rsi_fraction, start from a frame sampled in the band
        # [rsi_lo_frac, rsi_hi_frac] * (clip_len-1). Default band (0,1) = uniform over clip.
        if self.rsi_fraction > 0.0:
            use_rsi = torch.rand(n, device=self.device) < self.rsi_fraction
            max_start = (self.clip_lengths[env_ids] - 1).clamp(min=0).float()
            lo = max_start * float(self.rsi_lo_frac)
            hi = max_start * float(self.rsi_hi_frac)
            _u = torch.rand(n, device=self.device)
            band_frames = (lo + _u * (hi - lo).clamp(min=0.0)).round().long()
            band_frames = torch.minimum(band_frames, max_start.long())
            self.start_frames[env_ids] = torch.where(
                use_rsi, band_frames, torch.zeros(n, dtype=torch.long, device=self.device)
            )
        else:
            self.start_frames[env_ids] = 0
        # PER-SIDE CLOCK: a full env reset re-syncs both side clocks to the fresh
        # env-level start (side segments are born at elen 0).
        if self.per_side_clock:
            self._side_start[env_ids] = self.start_frames[env_ids].unsqueeze(-1).repeat(1, 2)
            self._side_birth[env_ids] = 0

    def _abs_frames(self, frame_ids: torch.Tensor) -> torch.Tensor:
        """Map policy-step indices to absolute trajectory-frame indices, slowdown-aware.

        ``effective = floor(frame_ids / slowdown_factor) + start_frames``, clamped per-env
        to ``[0, clip_lengths - 1]`` (avoids reading padding past each clip's end). This is
        the SAME timeline used by ``get_wrist_*`` (interp) and ``_ref_for_frame`` (floor) so
        control, rewards, terminations, contact, obs, and render all reference one timeline.
        ``slowdown_factor == 1.0`` reduces to ``frame_ids + start_frames`` (legacy behavior),
        plus the per-clip clamp. ``frame_ids`` are POLICY-STEP indices (e.g.
        ``episode_length_buf``); callers needing a specific absolute frame must convert to
        policy-step units (multiply the desired frame offset by ``slowdown_factor``).
        """
        if self._slowdown_factor == 1.0:
            eff = frame_ids + self.start_frames
        else:
            eff = (frame_ids.float() / self._slowdown_factor).floor().long() + self.start_frames
        last = (self.clip_lengths - 1).clamp(min=0)
        return torch.minimum(eff, last).clamp_(min=0)

    # ------------------------------------------------------------------
    # PER-SIDE CLOCK API (reset_scheme=decoupled). All methods no-op /
    # legacy-equivalent unless enable_per_side_clock() was called.
    # ------------------------------------------------------------------
    def enable_per_side_clock(self) -> None:
        """Arm the per-side clock (called once by the env under reset_scheme=decoupled)."""
        N = self.start_frames.shape[0]
        self._side_start = self.start_frames.unsqueeze(-1).repeat(1, 2).clone()
        self._side_birth = torch.zeros(N, 2, dtype=torch.long, device=self.device)
        self.per_side_clock = True
        print(
            "[TrajectoryManager] PER-SIDE CLOCK enabled: right/left reference frames "
            "advance on independent segment clocks (side_resample on side resets)"
        )

    def _side_idx(self, side: str) -> int:
        return 1 if side == "left" else 0

    def side_abs_frames(self, side: str, frame_ids: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        """(N,) long absolute frames for ``side`` under the per-side clock; falls back
        to the shared ``_abs_frames`` mapping when the clock is disabled."""
        if not self.per_side_clock:
            return self._abs_frames(frame_ids)
        i = self._side_idx(side)
        age = (frame_ids - self._side_birth[:, i]).clamp(min=0)
        if self._slowdown_factor == 1.0:
            eff = age + self._side_start[:, i]
        else:
            eff = (age.float() / self._slowdown_factor).floor().long() + self._side_start[:, i]
        if clamp:
            eff = torch.minimum(eff, (self.clip_lengths - 1).clamp(min=0))
            eff = eff.clamp(min=0)
        return eff

    def side_abs_frames_float(self, side: str, frame_ids: torch.Tensor) -> torch.Tensor:
        """(N,) float absolute frames for ``side`` (wrist-getter interpolation path).
        Clamped to [0, clip_len-1] — same overrun window as side_abs_frames (the
        slowdown==1 getter path indexes with .long() without an internal clamp)."""
        i = self._side_idx(side)
        age = (frame_ids - self._side_birth[:, i]).clamp(min=0)
        eff = (age.float() / self._slowdown_factor) + self._side_start[:, i].float()
        return torch.minimum(eff, (self.clip_lengths - 1).clamp(min=0).float())

    def side_clip_done(self, frame_ids: torch.Tensor) -> torch.Tensor:
        """(N,2) bool: side's UNCLAMPED absolute frame reached its clip end."""
        out = torch.zeros(frame_ids.shape[0], 2, dtype=torch.bool, device=self.device)
        if not self.per_side_clock:
            return out
        for i in range(2):
            age = (frame_ids - self._side_birth[:, i]).clamp(min=0)
            eff = (age.float() / self._slowdown_factor).floor().long() + self._side_start[:, i]
            out[:, i] = eff >= (self.clip_lengths - 1).clamp(min=0)
        return out

    def side_resample(self, side: str, env_ids: torch.Tensor, elen: torch.Tensor) -> None:
        """Resample a NEW RSI start for ``side`` on ``env_ids`` (side reset). Mirrors the
        env-reset band sampling in ``reset()`` exactly (rsi_fraction coin + band draw)."""
        i = self._side_idx(side)
        n = len(env_ids)
        if self.rsi_fraction > 0.0:
            use_rsi = torch.rand(n, device=self.device) < self.rsi_fraction
            max_start = (self.clip_lengths[env_ids] - 1).clamp(min=0).float()
            lo = max_start * float(self.rsi_lo_frac)
            hi = max_start * float(self.rsi_hi_frac)
            _u = torch.rand(n, device=self.device)
            band_frames = (lo + _u * (hi - lo).clamp(min=0.0)).round().long()
            band_frames = torch.minimum(band_frames, max_start.long())
            new_start = torch.where(use_rsi, band_frames, torch.zeros(n, dtype=torch.long, device=self.device))
        else:
            new_start = torch.zeros(n, dtype=torch.long, device=self.device)
        self._side_start[env_ids, i] = new_start
        self._side_birth[env_ids, i] = elen[env_ids]

    # ------------------------------------------------------------------
    # Trajectory completion / episode boundaries.
    # ------------------------------------------------------------------
    def is_trajectory_done(self, frame_ids: torch.Tensor) -> torch.Tensor:
        """Check if trajectory is complete for each env.

        Args:
            frame_ids: Current policy-step indices (num_envs,). When
                ``slowdown_factor != 1.0`` the policy-step counter advances
                ``slowdown_factor`` times more steps per clip than the raw
                frame count, so we compare against ``clip_lengths *
                slowdown_factor``. RSI offset added internally.

        Returns:
            Boolean tensor (num_envs,) -- True if trajectory is done.
        """
        # Done when the effective trajectory frame reaches the clip end. Uses the SAME
        # mapping as _abs_frames (floor(step/slowdown)+start) so RSI start_frames is handled
        # consistently. For start_frames=0 this equals step >= clip_lengths*slowdown (the
        # previous behavior); the old form under-scaled the start_frames term under RSI.
        eff_frame = (frame_ids.float() / self._slowdown_factor).floor() + self.start_frames.float()
        return eff_frame >= self.clip_lengths.float()
