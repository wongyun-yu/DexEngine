"""Residual side of the residual arm action: the reference base action, residual scaling, caps and smoothing (the
IK target and finger command side is residual_ik_target)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    pass


class ResidualProcessMixin:
    """Methods of :class:`ResidualArmIKAction` (composition of the residual action: reference base action, residual
    scaling and the wrist/finger targets)."""

    @staticmethod
    def _quat_from_rotvec(v: torch.Tensor) -> torch.Tensor:
        ang = torch.norm(v, dim=-1)
        axis = torch.zeros_like(v)
        axis[:, 2] = 1.0
        axis = torch.where(ang.unsqueeze(-1) > 1e-6, v / ang.clamp(min=1e-6).unsqueeze(-1), axis)
        return math_utils.quat_from_angle_axis(ang, axis)

    def _abs_box_build(self, root_conj1: torch.Tensor, z_off: float) -> None:
        """DEXENGINE_ACTION_ABS box for this side over every loaded clip's valid reference frames (floating-root frame),
        then a round trip encode → decode of all those frames (dies if it does not reproduce them)."""
        import hashlib
        import math

        tm = self._traj_mgr
        left = self._side_rb == "left"
        poses = tm._left_wrist_poses if left else tm._wrist_poses  # (C, T, 3) env-shared world
        quats = tm._left_wrist_quats if left else tm._wrist_quats  # (C, T, 4) wxyz
        C, T = poses.shape[0], poses.shape[1]
        lens = torch.as_tensor(tm._loader.clip_lengths).to(poses.device)[:C]
        valid = torch.arange(T, device=poses.device).unsqueeze(0) < lens.clamp(max=T).unsqueeze(1)
        p = poses[valid] - torch.tensor([0.0, 0.0, z_off], device=poses.device, dtype=poses.dtype)
        rc = root_conj1.to(poses.device).unsqueeze(0).expand(p.shape[0], -1)
        p_loc = math_utils.quat_apply(rc, p)
        q_loc = math_utils.quat_mul(rc, quats[valid])
        lo, hi = p_loc.min(0).values, p_loc.max(0).values
        s = torch.sign((q_loc * q_loc[0:1]).sum(-1, keepdim=True))
        s[s == 0] = 1.0
        qa = (q_loc * s).sum(0)
        qa = qa / qa.norm()
        box = {
            "anchor": ((lo + hi) / 2).to(self.device),
            "half": ((hi - lo) / 2 + self.ABS_MARGIN_M).to(self.device),
            "quat": qa.to(self.device),
        }
        a_pos = (p_loc.to(self.device) - box["anchor"]) / box["half"]
        qa_n = box["quat"].unsqueeze(0).expand(q_loc.shape[0], -1)
        a_rot = (
            math_utils.axis_angle_from_quat(math_utils.quat_mul(math_utils.quat_conjugate(qa_n), q_loc.to(self.device)))
            / math.pi
        )
        a = torch.cat([a_pos, a_rot], dim=-1)
        dp = (box["anchor"] + a_pos.clamp(-1, 1) * box["half"]) - p_loc.to(self.device)
        dq = math_utils.quat_mul(
            math_utils.quat_conjugate(q_loc.to(self.device)),
            math_utils.quat_mul(qa_n, self._quat_from_rotvec(a_rot.clamp(-1, 1) * math.pi)),
        )
        e_pos, e_rot = (
            float(dp.norm(dim=-1).max()),
            float(torch.norm(math_utils.axis_angle_from_quat(dq), dim=-1).max()),
        )
        h = hashlib.sha256(
            torch.cat([box["anchor"], box["half"], box["quat"]]).cpu().double().numpy().round(6).tobytes()
        ).hexdigest()[:12]

        def fmt(x):
            return "[" + ",".join(f"{float(v):.4f}" for v in x) + "]"

        print(
            f"[ACTION_ABS] on side={self._side_rb} clips={C} frames={int(valid.sum())} anchor_m={fmt(box['anchor'])} "
            f"half_m={fmt(box['half'])} anchor_quat={fmt(box['quat'])} max|a|={float(a.abs().max()):.3f} "
            f"roundtrip pos_err_m={e_pos:.2e} rot_err_rad={e_rot:.2e} box={h}",
            flush=True,
        )
        if e_pos > 1e-4 or e_rot > 1e-3 or float(a.abs().max()) > 1.0 + 1e-5:
            raise RuntimeError(
                f"[ACTION_ABS] FAIL round trip side={self._side_rb} pos {e_pos:.2e} rot {e_rot:.2e} "
                f"max|a| {float(a.abs().max()):.3f}"
            )
        self._abs_box = box

    def _sched_ramp(self, start, end):
        """Six residual-axis ramp: once end is set, interpolates start -> end linearly with the same progress as the
        gate
        curriculum (trajectory_manager.schedule_progress: 0 -> 1 over the first half of training, fixed at 1.0 in
        evaluation). end=None (default) = start as is (byte-identical). Coupling to schedule_progress is the point:
        evaluation automatically becomes the 'final training condition'."""
        if end is None or start is None:
            return start
        p = float(getattr(getattr(self._env, "trajectory_manager", None), "schedule_progress", 1.0) or 1.0)
        return start + (end - start) * p

    def compute_reference_base_action(self) -> torch.Tensor:
        """Reference base action (N, action_dim) for the single-stage reference-base pipeline.

        This is the base the residual policy corrects: wrist-pose delta = zeros (the base wrist
        IS the pure reference pose via use_trajectory_reference) and finger base = reference
        finger qpos mapped to the normalized [-1, 1] action space, aligned to
        ``inner._finger_joint_ids``. Computed directly from the reference trajectory — NO
        base model is involved. Shared as the SINGLE source of truth by
        ``process_actions`` (control) and the ``ReferenceBaseAction`` observation term, so the
        policy's base-action feature == the base it actually corrects.

        Finger-order note (smoke-verified): ``ref["right_hand_qpos"]`` is (N,18) in
        PhysX order aligned to ``traj_mgr._right_hand_ids``; we index it at the positions of
        ``inner._finger_joint_ids`` within ``_right_hand_ids``, reproducing the post-reset joints.
        """
        inner_rb = self._inner
        N, A = self._base_action_buffer.shape[0], self._base_action_buffer.shape[1]
        base = torch.zeros(N, A, device=self.device)  # wrist delta [0:6] stays zero (arm block [0:nw] under ARM_JOINT)
        nw = self._nw
        ref = self._traj_mgr.get_reference_state(self._env.episode_length_buf)
        # Bimanual Option 2: read THIS hand's reference finger qpos + hand-id list.
        # Default side='right' -> ref['right_hand_qpos'] / traj_mgr._right_hand_ids
        # (byte-identical); left term -> ref['left_hand_qpos'] / _left_hand_ids.
        rh_qpos = ref[f"{self._side_rb}_hand_qpos"]  # (N, 18) PhysX order
        if self._refbase_finger_in_hand is None:
            rh_ids = getattr(self._traj_mgr, f"_{self._side_rb}_hand_ids")
            rh_ids_list = rh_ids.tolist() if torch.is_tensor(rh_ids) else list(rh_ids)
            self._refbase_finger_in_hand = torch.tensor(
                [rh_ids_list.index(int(fid)) for fid in inner_rb._finger_joint_ids],
                device=rh_qpos.device,
                dtype=torch.long,
            )
        q_ref = rh_qpos[:, self._refbase_finger_in_hand]  # (N, 6) aligned to _finger_joint_ids
        base[:, nw:] = torch.clamp(
            2.0 * (q_ref - inner_rb._finger_lower) / inner_rb._finger_range - 1.0,
            -1.0,
            1.0,
        )
        return base

    def process_actions(self, residual_actions: torch.Tensor) -> None:
        """Process 12-dim residual actions with proper rotation composition.

        Base and residual are processed SEPARATELY through scaling and
        euler-to-quaternion conversion, then combined on the current wrist
        pose via proper rotation composition. This avoids the euler addition
        error that occurs when adding raw actions before conversion.

        If the LastBaseAction obs term already ran the base model this step
        (setting _base_computed_this_step=True), the base computation is skipped
        here and the same sample stored in _base_action_buffer is reused.
        """
        self._raw_actions[:] = residual_actions
        if self._eff_anneal is not None:
            self._effort_anneal_step()

        (
            N,
            _pos_cap,
            _rot_cap_deg,
            arm_res,
            base,
            base_pos_delta,
            base_rot_quat,
            cur_pos_local,
            cur_quat_local,
            floating_root_pos,
            inner,
            res_pos_delta,
            res_rot_quat,
            residual_used,
            root_q,
            rs_finger,
            rs_wrist,
            wrist_pos_w,
            wrist_quat_w,
        ) = self._smooth_residual(residual_actions)

        # ============================================================
        # Combine: current ⊕ base ⊕ residual
        # ============================================================
        # Position: linear addition (exact)
        target_pos_local = cur_pos_local + base_pos_delta + res_pos_delta
        # Rotation: proper quaternion composition (body-frame)
        target_quat_local = math_utils.quat_mul(math_utils.quat_mul(cur_quat_local, base_rot_quat), res_rot_quat)
        abs_finger = None
        return self._process_ik_target_and_command(
            N,
            _pos_cap,
            _rot_cap_deg,
            abs_finger,
            arm_res,
            base,
            base_pos_delta,
            base_rot_quat,
            cur_pos_local,
            cur_quat_local,
            floating_root_pos,
            inner,
            residual_used,
            root_q,
            rs_finger,
            rs_wrist,
            target_pos_local,
            target_quat_local,
            wrist_pos_w,
            wrist_quat_w,
        )

    def _smooth_residual(self, residual_actions):
        """Smooth only the learned residual before adding it to the frozen base. Raw actions stay unsmoothed for
        logging/rewards, while the physical target receives the filtered correction. The factor is
        cfg.action_ema.
        """
        if self.cfg.action_ema < 1.0:
            ema = float(self.cfg.action_ema)
            residual_used = ema * residual_actions + (1.0 - ema) * self._smoothed_residual_actions
            self._smoothed_residual_actions[:] = residual_used
        else:
            residual_used = residual_actions
            self._smoothed_residual_actions[:] = residual_actions

        N, arm_res, base, inner, residual_used, rs_finger, rs_wrist, wrist_pos_w, wrist_quat_w = (
            self._zero_residual_for_pd_base(residual_used)
        )

        # World → floating-root-local frame
        floating_root_pos = self._env.scene.env_origins.clone()
        floating_root_pos[:, 2] += inner._floating_root_z_offset
        root_q = inner._floating_root_quat.unsqueeze(0).expand(N, -1)
        root_conj = math_utils.quat_conjugate(root_q)

        cur_pos_local = math_utils.quat_apply(root_conj, wrist_pos_w - floating_root_pos)
        cur_quat_local = math_utils.quat_mul(root_conj, wrist_quat_w)

        # ============================================================
        # Process base action (same pipeline as the inner arm-IK term)
        # ============================================================
        base_pos_delta = base[:, 0:3] * inner._wrist_pos_scale
        base_rot_euler = base[:, 3:6] * inner._wrist_rot_scale
        base_rot_quat = inner._euler_xyz_to_quat(base_rot_euler)

        _pos_cap, _rot_cap_deg, res_pos_delta, res_rot_quat = self._cap_residual(inner, residual_used, rs_wrist)
        return (
            N,
            _pos_cap,
            _rot_cap_deg,
            arm_res,
            base,
            base_pos_delta,
            base_rot_quat,
            cur_pos_local,
            cur_quat_local,
            floating_root_pos,
            inner,
            res_pos_delta,
            res_rot_quat,
            residual_used,
            root_q,
            rs_finger,
            rs_wrist,
            wrist_pos_w,
            wrist_quat_w,
        )

    def _cap_residual(self, inner, residual_used, rs_wrist):
        """Ramp-scheduled caps on the wrist residual (position and rotation)."""
        _pos_cap = self._sched_ramp(self.cfg.wrist_pos_residual_max_m, self.cfg.wrist_pos_residual_max_m_end)
        if _pos_cap is not None:
            res_pos_delta = torch.clamp(residual_used[:, 0:3], -1.0, 1.0) * _pos_cap
        else:
            res_pos_delta = residual_used[:, 0:3] * inner._wrist_pos_scale * rs_wrist
        # 3-DoF wrist rotation residual as a ROTATION VECTOR (axis-angle / Rodrigues
        # exp-map), NOT Euler-XYZ. residual[3:6]*scale is the so(3) tangent vector:
        # angle = ||vec|| (rad), axis = vec/||vec||. Isotropic, no gimbal/order
        # dependence; also makes the wrist residual_l2 penalize the true rotation-angle
        # magnitude consistently with the eef_rot GEODESIC angle. (wrist_rs=0 -> vec=0
        # -> identity, so the wrist-locked path equals the Euler form bit for bit.)
        # If a hard cap is set (wrist_rot_residual_max_deg), clamp(raw,-1,1)*rad(cap).
        # Six residual axes: wrist rotation cap ramp (same interpolation)
        _rot_cap_deg = self._sched_ramp(self.cfg.wrist_rot_residual_max_deg, self.cfg.wrist_rot_residual_max_deg_end)
        if _rot_cap_deg is not None:
            res_rot_vec = torch.clamp(residual_used[:, 3:6], -1.0, 1.0) * (_rot_cap_deg * 0.017453292519943295)
        else:
            res_rot_vec = residual_used[:, 3:6] * inner._wrist_rot_scale * rs_wrist
        _res_angle = torch.norm(res_rot_vec, dim=-1)
        # Safe unit axis: quat_from_angle_axis() normalizes the axis internally, so a
        # ZERO vector (wrist_rs=0 -> res_rot_vec==0, every step of exps 1-8) would yield
        # normalize(0)=NaN. Use a default +z axis where angle~0 (the quaternion is the
        # identity there regardless of axis), so the wrist-locked path is NaN-free and
        # byte-identical (identity rotation).
        _default_axis = torch.zeros_like(res_rot_vec)
        _default_axis[:, 2] = 1.0
        _res_axis = torch.where(
            _res_angle.unsqueeze(-1) > 1e-6,
            res_rot_vec / _res_angle.clamp(min=1e-6).unsqueeze(-1),
            _default_axis,
        )
        res_rot_quat = math_utils.quat_from_angle_axis(_res_angle, _res_axis)
        return _pos_cap, _rot_cap_deg, res_pos_delta, res_rot_quat

    def _zero_residual_for_pd_base(self, residual_used):
        """PD reference base: force the learned residual to 0 (render/audit only)."""
        if self._zero_residual:
            residual_used = torch.zeros_like(residual_used)
            self._smoothed_residual_actions[:] = 0.0

        # DEXENGINE_ARM_JOINT_RESIDUAL: the arm block sets the arm joint target here; the wrist path below then runs
        # with a
        # zero wrist residual on the legacy (6 wrist | fingers) layout, so its IK command is the reference wrist and is
        # overwritten in apply_actions by the arm target.
        arm_res = None
        if self._arm_joint_res:
            nw = self._nw
            arm_res = torch.clamp(residual_used[:, 0:nw], -1.0, 1.0)
            q_ref_arm = self._arm_ref_q(self._wrist_frame_ids())
            tgt = q_ref_arm + arm_res * self._arm_res_cap
            if self._parked is not None:  # a parked side holds its reference
                tgt = torch.where(self._parked.unsqueeze(-1), q_ref_arm, tgt)
            first = self._arm_q_target is None
            self._arm_q_target = tgt
            if first:
                print(
                    f"[ARM_JOINT_RESIDUAL] first step side={self._side_rb} mean|ref "
                    f"q|={float(q_ref_arm.abs().mean()):.3f} "
                    f"mean|residual|={float(arm_res.abs().mean()):.3f} (cap units)",
                    flush=True,
                )
            self._arm_step += 1
            residual_used = torch.cat([torch.zeros_like(residual_used[:, 0:6]), residual_used[:, nw:]], dim=-1)

        # Base buffer stays zeros in the single-stage pipeline (the reference-base
        # override below supplies the actual base).
        self._base_action_buffer[:] = 0.0
        base = self._base_action_buffer

        # -- Reference-base: base action = kinematic reference (single-stage pipeline) --
        # base = compute_reference_base_action(): wrist-pose delta zeros (wrist target = pure
        # reference pose via use_trajectory_reference) + reference finger qpos (normalized).
        # The residual composition + [-1,1] clamp downstream are unchanged. This is the SINGLE
        # source of truth shared with the ReferenceBaseAction observation term, so the policy's
        # base-action feature == the base it actually corrects. No base model is involved
        # (the base buffer above is already zeros; this overwrites it
        # with the reference either way, so the result is identical with or without a base ckpt).
        if self._reference_base:
            base = self.compute_reference_base_action()
        if self._arm_joint_res:  # legacy layout for the wrist/finger code below (wrist base = 0)
            base = torch.cat([torch.zeros_like(base[:, 0:6]), base[:, self._nw :]], dim=-1)

        # Per-component residual scales (fall back to global residual_scale if None)
        rs_default = self.cfg.residual_scale
        rs_wrist = self.cfg.wrist_residual_scale if self.cfg.wrist_residual_scale is not None else rs_default
        rs_finger = self.cfg.finger_residual_scale if self.cfg.finger_residual_scale is not None else rs_default
        # Six residual axes: finger scale ramp (start -> end when set, schedule_progress interpolation)
        rs_finger = self._sched_ramp(rs_finger, self.cfg.finger_residual_scale_end)
        inner = self._inner

        # ============================================================
        # Get current wrist pose in floating-root-local frame
        # ============================================================
        if self._use_traj_ref:
            # Trajectory-reference: base pose from trajectory manager (+ wrist lookahead; 0 for parked envs)
            frame_ids = self._wrist_frame_ids()
            # side-aware: the LEFT residual term reads the LEFT wrist reference table.
            wrist_pos_w = self._traj_mgr.get_wrist_pos_w(frame_ids, side=self._side_rb) + self._env.scene.env_origins
            wrist_quat_w = self._traj_mgr.get_wrist_quat_w(frame_ids, side=self._side_rb)
            N = wrist_pos_w.shape[0]
        else:
            # Current-state: read actual link7 → virtual wrist (a parked env holds its current pose)
            self.parked_seen = self._parked
            wrist_pos_w, wrist_quat_w = self._virtual_wrist_w()
            N = wrist_pos_w.shape[0]
        return N, arm_res, base, inner, residual_used, rs_finger, rs_wrist, wrist_pos_w, wrist_quat_w
