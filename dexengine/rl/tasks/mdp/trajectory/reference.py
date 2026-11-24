"""Reference state at absolute frames, the precomputed reference table and the cache checks."""

from __future__ import annotations

import torch

from dexengine.rl.tasks.mdp.trajectory.manager_cfg import _REF_STASH, REF_TABLE_ATOL


class ReferenceStateMixin:
    """Methods of :class:`TrajectoryManager` (reference state at absolute frames, the precomputed reference table and
    the cache checks)."""

    def object_pose_window(
        self, side: str, frame_ids: torch.Tensor, window: int, ahead: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference OBJECT poses over the frame window [F-window, F+ahead] for the
        lag-tolerant tracking gate. Column k holds frame F-k for k>=0
        (k=0 = the instantaneous frame) and F+|k| for the leading columns, clamped
        to >= 0 (pre-RSI clip frames are valid reference history — this is what
        gives a not-yet-departed object its takeoff grace) and to clip_lengths-1
        (leading frames never read padding). ``ahead`` mirrors the
        wrist lookahead: an early-arriving object (phase-lead overcorrection) may
        LEAD the reference by up to k frames — the gate must not punish that.
        ahead=0 (default) = byte-identical trailing-only window.
        Per-side-clock aware via ``side_abs_frames`` (falls back to the shared clock).
        Returns (pos (N, window+ahead+1, 3), quat (N, window+ahead+1, 4)) in the same
        env-local frame as ``get_reference_state`` (same loader gather + rotate path).
        """
        F = self.side_abs_frames(side, frame_ids, clamp=True)  # (N,)
        ks = torch.arange(-int(ahead), window + 1, device=F.device, dtype=F.dtype)
        win = (F.unsqueeze(1) - ks.unsqueeze(0)).clamp(min=0)  # (N, K)
        if ahead > 0:
            _last = (self.clip_lengths - 1).clamp(min=0).unsqueeze(1)
            win = torch.minimum(win, _last)
        N, K = win.shape
        cids = self.clip_ids.unsqueeze(1).expand(N, K).reshape(-1)
        # Rule D1: get_frame returns an 8-tuple (… extra, artic). When the joint columns were added this caller missed
        # the sweep and the lag_window termination path (canary) died with "too many values to unpack".
        (_, _, r_pos, r_quat, l_pos, l_quat, _, _) = self._loader.get_frame(cids, win.reshape(-1))
        pos, quat = (r_pos, r_quat) if side != "left" else (l_pos, l_quat)
        if self._rotate:
            pos = self._rotate_pos_batch(pos)
            quat = self._rotate_quat_batch(quat)
        return pos.view(N, K, 3), quat.view(N, K, 4)

    def _ref_state_at_abs(self, absolute_frames: torch.Tensor) -> dict[str, torch.Tensor]:
        """Reference-state body at explicit ABSOLUTE frames (mechanically extracted from
        get_reference_state for the per-side clock; legacy path calls it with
        ``_abs_frames(frame_ids)`` — behavior byte-identical)."""
        if self._ref_table is not None:
            return self._ref_table_read(self.clip_ids, absolute_frames, "abs")
        (rh_qpos_dataset, lh_qpos_dataset, r_obj_pos, r_obj_quat, l_obj_pos, l_obj_quat, _extra, _artic) = (
            self._loader.get_frame(self.clip_ids, absolute_frames)
        )

        # Apply initial pose corrections for frame-0 sequences
        rh_qpos_dataset = self._apply_pose_corrections(rh_qpos_dataset, absolute_frames, self.clip_ids)

        # URDF hand: analytical dataset->URDF-hand wrist rotation transform
        if self._urdf_hand:
            rh_qpos_dataset = self._transform_wrist_dataset_to_robot(rh_qpos_dataset)
            lh_qpos_dataset = self._transform_wrist_dataset_to_robot(lh_qpos_dataset)

        # Apply dataset->PhysX permutation
        # For finger-only mode (full robot), select finger portion (indices 6 to 17) first.
        # Also store wrist qpos (indices 0:6 in dataset order) for observation reconstruction.
        if self._fingers_only:
            rh_qpos_physx = rh_qpos_dataset[:, 6:][:, self._right_perm]
            if self._left_ref_zero:
                lh_qpos_physx = torch.zeros(
                    lh_qpos_dataset.shape[0],
                    len(self._left_perm),
                    device=lh_qpos_dataset.device,
                    dtype=lh_qpos_dataset.dtype,
                )
            else:
                lh_qpos_physx = lh_qpos_dataset[:, 6:][:, self._left_perm]
            self._last_rh_wrist_qpos = rh_qpos_dataset[:, :6]  # (N, 6) wrist in dataset order
            self._last_lh_wrist_qpos = lh_qpos_dataset[:, :6]
            # Store the independent finger joints in dataset canonical order.
            # RH56F1 (18-joint layout): [6, 7, 10, 12, 14, 16]
            # = [thumb_yaw, thumb_pitch, index_prox, middle_prox, ring_prox, pinky_prox]
            # SharpA (28-joint layout): all 22 finger joints are independent (6..27).
            # Same order as TrajectoryData.finger_joint_pos and the action term's finger cfg.
            _indep = self._dataset_indep_finger_idx()
            self._last_rh_finger_dataset = rh_qpos_dataset[:, _indep]  # (N, n_indep_fingers)
            self._last_lh_finger_dataset = lh_qpos_dataset[:, _indep]
        else:
            rh_qpos_physx = rh_qpos_dataset[:, self._right_perm]
            lh_qpos_physx = lh_qpos_dataset[:, self._left_perm]

        # Apply coordinate rotation if enabled (+Y forward → +X forward)
        if self._rotate:
            r_obj_pos = self._rotate_pos_batch(r_obj_pos)
            l_obj_pos = self._rotate_pos_batch(l_obj_pos)
            r_obj_quat = self._rotate_quat_batch(r_obj_quat)
            l_obj_quat = self._rotate_quat_batch(l_obj_quat)

        return {
            "right_hand_qpos": rh_qpos_physx,
            "left_hand_qpos": lh_qpos_physx,
            "right_obj_pos": r_obj_pos,
            "right_obj_quat": r_obj_quat,
            "left_obj_pos": l_obj_pos,
            "left_obj_quat": l_obj_quat,
            # Rule D1: articulated hinge/slide angle; 0 for a non-articulated clip (the scene uses 0).
            "right_obj_arti": _artic[:, 0:1],
            "left_obj_arti": _artic[:, 1:2],
        }

    def _ref_state_per_side_clock(self, frame_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Merged reference state: right_* fields at the RIGHT clock's frames, left_* at
        the LEFT clock's. Two loader reads; the rh wrist/finger stashes from the right
        pass are restored after the left pass (each side's obs-reconstruction stash must
        come from its own clock)."""
        res_r = self._ref_state_at_abs(self.side_abs_frames("right", frame_ids, clamp=True))
        s_rw = self._last_rh_wrist_qpos if self._fingers_only else None
        s_rf = self._last_rh_finger_dataset if self._fingers_only else None
        res_l = self._ref_state_at_abs(self.side_abs_frames("left", frame_ids, clamp=True))
        if self._fingers_only:
            self._last_rh_wrist_qpos = s_rw
            self._last_rh_finger_dataset = s_rf
        return {
            "right_hand_qpos": res_r["right_hand_qpos"],
            "left_hand_qpos": res_l["left_hand_qpos"],
            "right_obj_pos": res_r["right_obj_pos"],
            "right_obj_quat": res_r["right_obj_quat"],
            "left_obj_pos": res_l["left_obj_pos"],
            "left_obj_quat": res_l["left_obj_quat"],
        }

    def get_reference_state(self, frame_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Get reference state at given frame indices for ALL envs.

        Args:
            frame_ids: POLICY-STEP indices of shape (num_envs,). Typically env.episode_length_buf.
                RSI start_frames offset AND slowdown_factor are applied internally
                (effective trajectory frame = floor(frame_ids / slowdown) + start_frames).

        Returns:
            Dictionary with keys:
            - right_hand_qpos: (num_envs, 18) in PhysX joint order
            - left_hand_qpos: (num_envs, 18) in PhysX joint order
            - right_obj_pos: (num_envs, 3)
            - right_obj_quat: (num_envs, 4) wxyz
            - left_obj_pos: (num_envs, 3)
            - left_obj_quat: (num_envs, 4) wxyz
        """
        if self._ref_table is not None:  # table path: one gather, so no cache (its key would be a host sync .item())
            if self.per_side_clock:
                return self._ref_state_per_side_clock(frame_ids)
            return self._ref_state_at_abs(self._abs_frames(frame_ids))
        # Per-step cache: rewards call with both frame_ids and frame_ids+1.
        # Key = frame_ids.sum(), but **a hit needs the same real inputs** (clip and per-side absolute frames): the sum
        # alone would return a stale reference after a reset, side_resample or in-place mapping change with an equal
        # sum. Still one host sync per call (_cache_probe).
        sig = self._ref_input_sig(frame_ids)
        fsum, e0, e1 = self._cache_probe(frame_ids, sig, self._cache_state_sig, self._cache_state_p1_sig)
        if self._cache_state is not None and self._cache_state_key == fsum:
            if e0:
                return self._cache_state
            self._ref_cache_collision("get_reference_state", fsum)
            self._cache_state, self._cache_state_key, self._cache_state_sig = None, -1, None
        if self._cache_state_p1 is not None and self._cache_state_p1_key == fsum:
            if e1:
                return self._cache_state_p1
            self._ref_cache_collision("get_reference_state(+1)", fsum)
            self._cache_state_p1, self._cache_state_p1_key, self._cache_state_p1_sig = None, -1, None

        if self.per_side_clock:
            result = self._ref_state_per_side_clock(frame_ids)
        else:
            result = self._ref_state_at_abs(self._abs_frames(frame_ids))
        # Store in first available cache slot
        if self._cache_state is None:
            self._cache_state = result
            self._cache_state_key = fsum
            self._cache_state_sig = sig
        elif self._cache_state_p1 is None:
            self._cache_state_p1 = result
            self._cache_state_p1_key = fsum
            self._cache_state_p1_sig = sig
        return result

    def get_reference_state_for_envs(self, env_ids: torch.Tensor, frame_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Get reference state for specific environments only.

        Args:
            env_ids: Environment indices of shape (N,).
            frame_ids: Frame indices of shape (N,).

        Returns:
            Same dict structure as get_reference_state but shaped (N, ...).
        """
        clip_ids_sub = self.clip_ids[env_ids]
        if self._ref_table is not None:
            return self._ref_table_read(clip_ids_sub, frame_ids, "envs")
        (rh_qpos_dataset, lh_qpos_dataset, r_obj_pos, r_obj_quat, l_obj_pos, l_obj_quat, _extra, _artic) = (
            self._loader.get_frame(clip_ids_sub, frame_ids)
        )

        # Apply initial pose corrections for frame-0 sequences
        rh_qpos_dataset = self._apply_pose_corrections(rh_qpos_dataset, frame_ids, clip_ids_sub)

        # URDF hand: analytical dataset->URDF-hand wrist rotation transform
        if self._urdf_hand:
            rh_qpos_dataset = self._transform_wrist_dataset_to_robot(rh_qpos_dataset)
            lh_qpos_dataset = self._transform_wrist_dataset_to_robot(lh_qpos_dataset)

        if self._fingers_only:
            rh_qpos_physx = rh_qpos_dataset[:, 6:][:, self._right_perm]
            if self._left_ref_zero:
                lh_qpos_physx = torch.zeros(
                    lh_qpos_dataset.shape[0],
                    len(self._left_perm),
                    device=lh_qpos_dataset.device,
                    dtype=lh_qpos_dataset.dtype,
                )
            else:
                lh_qpos_physx = lh_qpos_dataset[:, 6:][:, self._left_perm]
            self._last_rh_wrist_qpos = rh_qpos_dataset[:, :6]
            self._last_lh_wrist_qpos = lh_qpos_dataset[:, :6]
            self._last_rh_finger_dataset = rh_qpos_dataset[:, 6:12]
            self._last_lh_finger_dataset = lh_qpos_dataset[:, 6:12]
        else:
            rh_qpos_physx = rh_qpos_dataset[:, self._right_perm]
            lh_qpos_physx = lh_qpos_dataset[:, self._left_perm]

        # Apply coordinate rotation if enabled
        if self._rotate:
            r_obj_pos = self._rotate_pos_batch(r_obj_pos)
            l_obj_pos = self._rotate_pos_batch(l_obj_pos)
            r_obj_quat = self._rotate_quat_batch(r_obj_quat)
            l_obj_quat = self._rotate_quat_batch(l_obj_quat)

        return {
            "right_hand_qpos": rh_qpos_physx,
            "left_hand_qpos": lh_qpos_physx,
            "right_obj_pos": r_obj_pos,
            "right_obj_quat": r_obj_quat,
            "left_obj_pos": l_obj_pos,
            "left_obj_quat": l_obj_quat,
            # Rule D1: articulated hinge/slide angle; 0 for a non-articulated clip (the scene uses 0).
            "right_obj_arti": _artic[:, 0:1],
            "left_obj_arti": _artic[:, 1:2],
        }

    def extra_slot_poses(self, env_ids: torch.Tensor, frame_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference poses of the extra object slots: (pos (N, K, 3) env-local · quat (N, K, 4) wxyz). The **same**
        transform as the main objects (the loader's per-clip dz -> Rz(−90°) + desk height, `_rotate_*_batch`), so the
        scene coordinates share the main objects' plane. Empty slots keep the loader's z=−100 (the writer turns them
        into the hidden pose). No slots (K=0) fails: there is no reason to call it."""
        K = int(self._loader.n_extra_slots)
        if K == 0:
            raise RuntimeError("[EXTRA_SLOTS] extra_slot_poses called with 0 slots")
        _, _, _, _, _, _, extra, _ = self._loader.get_frame(self.clip_ids[env_ids], frame_ids)
        n = extra.shape[0]
        pos, quat = extra[..., 0:3].reshape(n * K, 3), extra[..., 3:7].reshape(n * K, 4)
        if self._rotate:
            pos, quat = self._rotate_pos_batch(pos), self._rotate_quat_batch(quat)
        return pos.reshape(n, K, 3), quat.reshape(n, K, 4)

    def _ref_table_read(self, clip_ids: torch.Tensor, frames: torch.Tensor, variant: str) -> dict[str, torch.Tensor]:
        out, stash = self._ref_table_gather(self._ref_table, clip_ids, frames, variant)
        for k, v in stash.items():
            setattr(self, k, v)
        return out

    @staticmethod
    def _ref_table_gather(tbl: dict, clip_ids: torch.Tensor, frames: torch.Tensor, variant: str):
        T = tbl["T"]
        idx = clip_ids * T + frames.clamp(0, T - 1)
        out = {k: v.index_select(0, idx) for k, v in tbl["state"].items()}
        stash = {k: v.index_select(0, idx) for k, v in tbl["stash_" + variant].items()}
        return out, stash

    def _ref_rows_eager(self, clips: torch.Tensor, frames: torch.Tensor, chunk: int, variant: str):
        """Compute (clips, frames) rows with the old path (`_ref_state_at_abs` / `get_reference_state_for_envs` without
        a
        table) chunk rows at a time and join them. A short last piece repeats its end pair, so every call has chunk rows
        (the run's kernel)."""
        n = clips.numel()
        state: dict[str, list] = {}
        stash: dict[str, list] = {}
        for a in range(0, n, chunk):
            c, f = clips[a : a + chunk], frames[a : a + chunk]
            m = c.numel()
            if m < chunk:
                c = torch.cat([c, c[-1:].expand(chunk - m)])
                f = torch.cat([f, f[-1:].expand(chunk - m)])
            self.clip_ids = c
            self.clip_lengths = self._loader.get_clip_lengths(c)
            if variant == "abs":
                st = self._ref_state_at_abs(f)
            else:
                st = self.get_reference_state_for_envs(torch.arange(chunk, device=c.device), f)
            for k, v in st.items():
                state.setdefault(k, []).append(v[:m])
            if self._fingers_only:
                for k in _REF_STASH:
                    stash.setdefault(k, []).append(getattr(self, k)[:m])
        return ({k: torch.cat(v) for k, v in state.items()}, {k: torch.cat(v) for k, v in stash.items()})

    def _ref_table_oracle(
        self, tbl: dict, n_clips: int, chunk: int, variant: str, seed: int, edges: bool
    ) -> tuple[float, int]:
        """Re-run the old path over every (clip, frame) in a shuffled order and chunk-row pieces and compare with the
        table gather. Different keys, shapes, dtypes or NaN positions, or a max absolute difference above
        REF_TABLE_ATOL, fail. No global RNG is used."""
        T = tbl["T"]
        clips = torch.arange(n_clips).repeat_interleave(T)
        frames = torch.arange(T).repeat(n_clips)
        if edges:  # the loader clamp edges (negative, past the end) give the same values too
            clips = torch.cat([clips, torch.arange(n_clips).repeat_interleave(2)])
            frames = torch.cat([frames, torch.tensor([-1, T + 3]).repeat(n_clips)])
        perm = torch.randperm(clips.numel(), generator=torch.Generator(device="cpu").manual_seed(seed))
        dev = self.clip_ids.device
        clips, frames = clips[perm].to(dev), frames[perm].to(dev)
        ref = self._ref_rows_eager(clips, frames, chunk, variant)
        got = self._ref_table_gather(tbl, clips, frames, variant)
        worst = 0.0
        for part, a_d, b_d in (("state", ref[0], got[0]), ("stash", ref[1], got[1])):
            if a_d.keys() != b_d.keys():
                raise RuntimeError(f"[REF_TABLE] {variant} {part} key mismatch old {sorted(a_d)} · table {sorted(b_d)}")
            for k, a in a_d.items():
                b = b_d[k]
                if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a.isnan(), b.isnan()):
                    raise RuntimeError(
                        f"[REF_TABLE] {variant} {k}: old {tuple(a.shape)} {a.dtype} · "
                        f"table {tuple(b.shape)} {b.dtype} (or NaN positions) mismatch"
                    )
                if a.numel():
                    worst = max(worst, float((a.double() - b.double()).abs().nan_to_num(0.0).max()))
        if worst > REF_TABLE_ATOL:
            raise RuntimeError(
                f"[REF_TABLE] {variant} comparison max |Δ| {worst:.3g} > {REF_TABLE_ATOL:g} "
                f"(pieces of {chunk} rows) — the table is not used"
            )
        return worst, int(clips.numel())

    def _build_ref_table(self) -> None:
        import time

        t0 = time.perf_counter()
        if self._use_pose_corrections:
            raise RuntimeError(
                "[REF_TABLE] pose correction is on — the table index knows only the loader rule (clamp) without "
                "corrections"
            )
        n_clips, T = int(self._loader.num_clips), int(self._loader.max_length)
        if T < 2:
            raise RuntimeError(
                f"[REF_TABLE] max_length={T} — frame 0 and the clamp edge overlap, the table index is ambiguous"
            )
        chunk = min(max(int(self.num_envs), 256), 8192)  # the run's row count (= num_envs) per piece
        chunk_b = (
            509 if chunk != 509 else 511
        )  # another row count: measures the kernel difference of reset-subset calls
        keep = {k: getattr(self, k) for k in ("clip_ids", "clip_lengths")}
        had = {k: self.__dict__[k] for k in _REF_STASH if k in self.__dict__}
        try:
            dev = self.clip_ids.device
            clips = torch.arange(n_clips, device=dev).repeat_interleave(T)
            frames = torch.arange(T, device=dev).repeat(n_clips)
            state, stash_abs = self._ref_rows_eager(clips, frames, chunk, "abs")
            _, stash_envs = self._ref_rows_eager(clips, frames, chunk, "envs")
            tbl = {"T": T, "state": state, "stash_abs": stash_abs, "stash_envs": stash_envs}
            err_a, n_a = self._ref_table_oracle(tbl, n_clips, chunk, "abs", seed=1, edges=False)
            err_b, n_b = self._ref_table_oracle(tbl, n_clips, chunk_b, "envs", seed=2, edges=True)
        finally:
            for k, v in keep.items():
                setattr(self, k, v)
            for k in _REF_STASH:
                self.__dict__.pop(k, None)
            self.__dict__.update(had)
        self._ref_table = tbl
        nbytes = sum(v.numel() * v.element_size() for d in (state, stash_abs, stash_envs) for v in d.values())
        print(
            f"[REF_TABLE] ON clips={n_clips} frames={T} rows={n_clips * T} · keys state {len(state)} + side stash "
            f"{len(stash_abs)}×2 · {nbytes / 2**20:.1f} MiB · pieces {chunk}/{chunk_b} · compare abs max|Δ|={err_a:.3g}"
            f"({n_a} rows) · envs max|Δ|={err_b:.3g}({n_b} rows, clamp edges included) ≤ atol {REF_TABLE_ATOL:g} · "
            f"{time.perf_counter() - t0:.1f} s",
            flush=True,
        )

    def _ref_input_sig(self, frame_ids: torch.Tensor) -> torch.Tensor:
        """Real input of a reference lookup = (clip, per-side absolute frame [after clamp]).

        The reference value is a function of this input only (the same premise as the table oracle)."""
        if self.per_side_clock:
            fr = torch.stack(
                (
                    self.side_abs_frames("right", frame_ids, clamp=True),
                    self.side_abs_frames("left", frame_ids, clamp=True),
                )
            )
        else:
            fr = self._abs_frames(frame_ids).unsqueeze(0)
        return torch.cat((self.clip_ids.reshape(1, -1).to(fr.dtype), fr), 0)

    @staticmethod
    def _cache_probe(frame_ids: torch.Tensor, sig: torch.Tensor, *slot_sigs) -> list:
        """Reads [sum, per-slot input match (0/1) …] with **one host sync** (tolist)."""
        parts = [frame_ids.sum().reshape(1).to(torch.long)]
        for ss in slot_sigs:
            if ss is None or ss.shape != sig.shape:
                parts.append(torch.zeros(1, dtype=torch.long, device=frame_ids.device))
            else:
                parts.append((ss == sig).all().reshape(1).to(torch.long))
        return torch.cat(parts).tolist()

    def _ref_cache_collision(self, where: str, fsum: int) -> None:
        """A call with an equal sum key but other inputs, where the old cache would have returned a stale reference.

        Recomputes, and leaves a marker on the 1st, 2nd, 4th, 8th… occurrence (so the log shows boot-once or
        all-run)."""
        self._ref_cache_collisions += 1
        n = self._ref_cache_collisions
        if n & (n - 1) == 0:
            print(
                f"[REF_CACHE] sum-key collision #{n} — {where} sum={fsum} but the clip/absolute frames differ from the "
                "cache: the old sum-only key would have returned a stale reference → recomputed",
                flush=True,
            )

    def _check_cache_valid(self, env_step: int) -> bool:
        """Check if cache is still valid for the current simulation step."""
        if env_step != self._cache_step:
            self._cache_step = env_step
            self._cache_body_pos = None
            self._cache_state = None
            self._cache_state_key = -1
            self._cache_state_p1 = None
            self._cache_state_p1_key = -1
            return False
        return True
