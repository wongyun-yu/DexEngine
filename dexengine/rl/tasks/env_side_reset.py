"""Per-side reset of the bimanual env: per-side rewards, snapping a failed side back onto its reference and parking."""

from __future__ import annotations

import torch


class PerSideResetMixin:
    """Methods of :class:`ArmResidualTrainEnv` (per-side reset of the bimanual env: per-side rewards, snapping a failed
    side back onto its reference and parking)."""

    # ------------------------------------------------------------------
    # PER-SIDE ADVANTAGE (factored credit) — flag-gated reward split.
    # cfg.per_side_reward is set ONLY by rl/train/runner.py under
    # --per_side_advantage 1 --bimanual 1; default (attribute absent) keeps
    # step() byte-identical. Hook choice: RewardManager.compute() already
    # stores every term's step value in _step_reward (= value/dt), so the
    # split is a cached per-term side-fraction vector times that matrix —
    # no term recomputation, no RewardManager edits.
    # ------------------------------------------------------------------
    def _per_side_reward_fracs(self):
        """(right_frac, left_frac) vectors over the reward terms, cached.

        Classification per term: params['side'] ('right'/'left') is authoritative;
        fallback = name suffix _right/_left; anything not clearly side-tagged
        (global/diag terms, side='both') splits 50/50 — so right+left == total
        by construction. Term list is fixed after cfg build; cache keyed on it.
        """
        rm = self.reward_manager
        names = list(rm._term_names)
        cached = getattr(self, "_psr_cache", None)
        if cached is not None and cached[0] == names:
            return cached[1], cached[2]
        right, left, shared = [], [], []
        fr = torch.zeros(len(names), device=self.device)
        for i, nm in enumerate(names):
            params = dict(getattr(rm._term_cfgs[i], "params", None) or {})
            side = str(params.get("side", "") or "")
            if side == "right":
                fr[i] = 1.0
                right.append(nm)
            elif side == "left":
                fr[i] = 0.0
                left.append(nm)
            elif nm.endswith("_right"):
                fr[i] = 1.0
                right.append(nm)
            elif nm.endswith("_left"):
                fr[i] = 0.0
                left.append(nm)
            else:
                fr[i] = 0.5
                shared.append(nm)  # global term: 50/50 keeps the sum exact
        fl = 1.0 - fr
        print(
            f"[PER-SIDE REWARD] {len(names)} terms -> right({len(right)})={right} | "
            f"left({len(left)})={left} | 50/50({len(shared)})={shared}"
        )
        self._psr_cache = (names, fr, fl)
        return fr, fl

    def compute_per_side_rewards(self):
        """(reward_right, reward_left), each (num_envs,), for the LAST computed step.

        _step_reward stores value/dt, so x step_dt recovers each term's actual
        contribution (term_raw x weight x dt); the side-fraction matmul then
        partitions the exact total the RewardManager summed.
        """
        fr, fl = self._per_side_reward_fracs()
        per_term = self.reward_manager._step_reward * self.step_dt  # (N, T)
        return per_term @ fr, per_term @ fl

    # ------------------------------------------------------------------
    # RESET SCHEME=decoupled: per-side snap helpers.
    # ------------------------------------------------------------------
    def _pss_snap_side(self, side: str, env_ids: torch.Tensor):
        """Snap one side's arm+fingers+object onto its reference at the CURRENT
        absolute clip frame (mirrors the frame-indexed paths of
        reset_arm_to_trajectory / reset_hands_to_trajectory / reset_object_to_trajectory,
        restricted to one side and mid-episode frames)."""
        tm = self.trajectory_manager
        if getattr(tm, "per_side_clock", False):
            # decoupled: side clock was just resampled -> maps to the side's NEW start.
            frames = tm.side_abs_frames(side, self.episode_length_buf, clamp=True)[env_ids]
        else:
            slow = float(getattr(tm, "_slowdown_factor", 1.0))
            frames = (self.episode_length_buf[env_ids].float() / slow).long() + tm.start_frames[env_ids]
            frames = torch.minimum(frames, (tm.clip_lengths[env_ids] - 1).clamp(min=0))
        ref = tm.get_reference_state_for_envs(env_ids, frames)

        term = self._reset_fingers_to_reference(side, env_ids, frames, ref, tm)
        if term is not None:
            raw = getattr(term, "_raw_actions", None)
            if raw is None:
                raw = getattr(term, "raw_actions", None)
            if raw is not None:
                raw[env_ids] = 0.0

        # (5) zero the side's contact-mismatch counter (re-grants the full grace window).
        cnt = getattr(self, f"_contact_mismatch_counter_{side}", None)
        if cnt is not None:
            cnt[env_ids] = 0

        # (6) stamp the snap step — the side's gates stay disarmed for _pss_grace steps.
        self._pss_last_snap[env_ids, 0 if side == "right" else 1] = int(self.common_step_counter)

    def _reset_fingers_to_reference(self, side, env_ids, frames, ref, tm):
        """(1) fingers <- reference qpos (full-robot mode: per-hand joint ids)."""
        hand = self.scene[f"{side}_hand"]
        qpos = ref[f"{side}_hand_qpos"]
        if getattr(tm, "_fingers_only", False):
            jids = (tm._right_hand_ids if side == "right" else tm._left_hand_ids).tolist()
            hand.write_joint_state_to_sim(qpos, torch.zeros_like(qpos), joint_ids=jids, env_ids=env_ids)
            hand.set_joint_position_target(qpos, joint_ids=jids, env_ids=env_ids)
        else:
            hand.write_joint_state_to_sim(qpos, torch.zeros_like(qpos), env_ids=env_ids)
            hand.set_joint_position_target(qpos, env_ids=env_ids)

        # (2) arm <- frame-indexed IK cache (loaded lazily by reset_arm_to_trajectory on
        # the global episode-0 reset, which always precedes any snap). Cache absent ->
        # PD-only convergence: the residual base action IS the reference, and the raw
        # residual is zeroed below, so the PD target already pulls the arm home.
        robot = self.scene["robot"]
        # the cache the RESET reads (events.reset_cache_all: the A2 cache under DEXENGINE_ARM_IK_RESET_SUFFIX, else
        # canonical)
        from dexengine.rl.tasks.mdp.events.reset_cache import reset_cache_all as _reset_cache_all

        cache_all, cache_len = _reset_cache_all(self, side)
        if side == "right":
            arm_jids = getattr(self, "_arm_ik_joint_ids_right", None)
            arm_robot = robot
        else:
            arm_jids = getattr(self, "_left_arm_jids_t", None)
            from dexengine.rl.tasks.scene import split_artic as _split_artic_ps

            _split = _split_artic_ps.enabled()
            arm_robot = self.scene["left_hand"] if (_split and "left_hand" in self.scene.keys()) else robot  # noqa: SIM118  — InteractiveScene has no __contains__
        if cache_all is not None and arm_jids is not None:
            clip_ids = tm.clip_ids[env_ids]
            f_arm = frames.clamp(min=0)
            if cache_len is not None:
                f_arm = torch.minimum(f_arm, cache_len[clip_ids] - 1)
            arm_jpos = cache_all[clip_ids, f_arm]  # (N, 7)
            arm_robot.write_joint_state_to_sim(
                arm_jpos, torch.zeros_like(arm_jpos), joint_ids=arm_jids, env_ids=env_ids
            )
            arm_robot.set_joint_position_target(arm_jpos, joint_ids=arm_jids, env_ids=env_ids)
        elif side not in self._pss_warned_arm:
            self._pss_warned_arm.add(side)
            print(
                f"[SIDESNAP] WARNING: no all-frame arm-IK cache for side={side} — "
                f"arm converges via PD toward the reference base instead of teleporting",
                flush=True,
            )

        # (3) object <- reference pose at the frame, zero velocity. Shared-object scene: left -> object_right (T3).
        # D16 idle side: that side has no object — nothing to re-seat (the resolver answers None).
        from dexengine.rl.tasks.scene.collision_bodies import scene_object_key as _sok

        _okey = _sok(
            self.scene.keys(),
            side,
            bool(getattr(self.cfg, "shared_object", False)),
            str(getattr(self.cfg, "idle_side", "") or ""),
        )
        if _okey is not None:
            obj = self.scene[_okey]
            _pp, _pq, _, _ = self.object_pin_pose(
                env_ids, frames, side, obj, ref
            )  # one owner of the fixed pose (object_pin_pose)
            root_state = torch.zeros(len(env_ids), 13, device=self.device)
            root_state[:, :3] = _pp + self.scene.env_origins[env_ids]
            root_state[:, 3:7] = _pq
            obj.write_root_state_to_sim(root_state, env_ids)

        # (4) zero the side's raw residual action (feeds last-action obs + action_rate).
        term = self.action_manager._terms.get(f"{side}_hand_joint_pos")
        return term

    def _pss_apply_pending(self):
        """Top-of-step: execute the snaps queued by the previous step's gate fails."""
        if getattr(self, "_park_on", False):
            fresh = self.episode_length_buf == 0
            for side in ("right", "left"):
                self._parked[side] &= ~fresh  # a new episode un-parks
                newly = self._pss_pending[side] & (self.episode_length_buf > 0)
                self._parked[side] |= newly
                self._pss_counts[side] += int(newly.sum())
                self._pss_pending[side] = torch.zeros_like(newly)
                ids = (self._parked[side] & (self.episode_length_buf > 0)).nonzero(as_tuple=False).squeeze(-1)
                if ids.numel():
                    self._pss_snap_side(side, ids)  # kinematic follow (current frame)
            c = int(self.common_step_counter)
            if c - self._pss_last_log >= int(getattr(self.cfg, "termdiag_steps", 800)):
                print(
                    f"[PARK] step {c}: cumulative parks right={self._pss_counts['right']} "
                    f"left={self._pss_counts['left']} | now parked r={int(self._parked['right'].sum())} "
                    f"l={int(self._parked['left'].sum())}",
                    flush=True,
                )
                self._pss_last_log = c
            return
        for side in ("right", "left"):
            # Apply-time freshness filter: pending was queued at the END
            # of the previous step; if the env was FULLY reset since (frozen-eval
            # boundary calls wrapped.reset() after restore), elen is 0 and the queued
            # snap is STALE — applying it would teleport freshly-placed envs onto
            # random side segments right after every frozen eval.
            pend = self._pss_pending[side] & (self.episode_length_buf > 0)
            if pend.any():
                ids = pend.nonzero(as_tuple=False).squeeze(-1)
                if self._pss_mode == "decoupled":
                    # Independent clock: draw a fresh RSI segment for this side FIRST,
                    # then the snap teleports onto the new segment's start frame.
                    self.trajectory_manager.side_resample(side, ids, self.episode_length_buf)
                self._pss_snap_side(side, ids)
                self._pss_counts[side] += int(ids.numel())
                self._pss_pending[side] = torch.zeros_like(pend)
        c = int(self.common_step_counter)
        if c - self._pss_last_log >= int(getattr(self.cfg, "termdiag_steps", 800)):
            import os as _os_r

            print(
                f"[SIDESNAP][r{_os_r.environ.get('LOCAL_RANK', '0')}] step {c}: cumulative snaps "
                f"right={self._pss_counts['right']} left={self._pss_counts['left']}",
                flush=True,
            )
            self._pss_last_log = c

    def _pss_after_step(self):
        """Post-step: publish per-side dones, queue snaps (skipping envs that fully
        reset this step — their elen is 0 and the RSI reset already re-placed them)."""
        comp = (
            self.trajectory_manager.side_clip_done(self.episode_length_buf) if self._pss_mode == "decoupled" else None
        )
        for si, side in enumerate(("right", "left")):
            fail = getattr(self, f"_side_fail_{side}", None)
            if fail is None:
                fail = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            if comp is not None:
                # decoupled: a side finishing its clip segment also ends that side's
                # GAE segment and cycles it onto a fresh RSI draw (uni auto-reset).
                comp_only = comp[:, si] & ~fail
                n_f, n_c = int(fail.sum()), int(comp_only.sum())
                if n_f or n_c:
                    done_any = fail | comp_only
                    age = (self.episode_length_buf - self.trajectory_manager._side_birth[:, si])[done_any].clamp(min=0)
                    for st in (self._pss_win, self._pss_cum):
                        st["fail"][si] += n_f
                        st["comp"][si] += n_c
                        st["seglen"][si] += int(age.sum())
            self.extras[f"sidefail_{side}"] = fail.clone()  # pure failure (completion excluded)
            if comp is not None:
                fail = fail | comp[:, si]
            self.extras[f"done_{side}"] = fail
            self._pss_pending[side] = fail & (self.episode_length_buf > 0)
            setattr(self, f"_side_fail_{side}", torch.zeros_like(fail))
        # Rolling W&B emit (every 32 steps): segment completion rate + mean length.
        if comp is not None:
            c = int(self.common_step_counter)
            if c - self._pss_win_last >= 32:
                log = self.extras.setdefault("log", {})
                for si, side in enumerate(("right", "left")):
                    tot = self._pss_win["fail"][si] + self._pss_win["comp"][si]
                    if tot > 0:
                        log[f"SideStat/{side}_seg_completion"] = self._pss_win["comp"][si] / tot
                        log[f"SideStat/{side}_seg_len_mean"] = self._pss_win["seglen"][si] / tot
                        log[f"SideStat/{side}_seg_fails_per_env"] = self._pss_win["fail"][si] / self.num_envs
                self._pss_win = {"fail": [0, 0], "comp": [0, 0], "seglen": [0, 0]}
                self._pss_win_last = c

    def pss_side_stats_snapshot(self, reset: bool = True):
        """Cumulative per-side segment stats since the last snapshot (frozen-eval hook).
        Returns {'fail': [r,l], 'comp': [r,l], 'seglen': [r,l]}."""
        out = {k: list(v) for k, v in self._pss_cum.items()}
        if reset:
            self._pss_cum = {"fail": [0, 0], "comp": [0, 0], "seglen": [0, 0]}
        return out
