"""Arm feed-forward of the residual action: gravity compensation and the reference-velocity lead, with the arm jerk
accumulator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from dexengine.rl.swallow import note_once

if TYPE_CHECKING:
    pass


class ArmFeedforwardMixin:
    """Methods of :class:`ResidualArmIKAction` (arm feed-forward of the residual action: gravity compensation and the
    reference-velocity lead, with the arm jerk accumulator)."""

    def _lead(self, k: int):
        """Wrist lead k (policy steps): 0 for parked envs (parked target = frame 0). Without parking, k as is (int)."""
        return k if self._parked is None else (~self._parked).long() * k

    def _accum_arm_jerk(self) -> None:
        """Accumulate ACHIEVED arm-joint jerk (rad/s^3) into env._arm_jerk_acc. Diagnostic only.

        Why: the visible tremble in the rollout videos has to be judged on
        ABSOLUTE jerk, not the achieved/reference ratio. The ratio is already known to be <= 0.46
        on every clip, i.e. the controller low-passes rather than adds tremble, so a ratio target
        cannot distinguish "smooth" from "inherits a jerky reference". Only the absolute number can.

        Reads measured joint_pos (not the target), so it reflects what the arm actually did.
        Third difference over the last 4 steps, worst joint per env, then mean/max over envs.
        rl.judge snapshot metrics drain and reset this every logging iteration.

        Cost: one (N,7) gather + two subtractions per step. Never raises - a diagnostic must not
        be able to kill a run.
        """
        try:
            ids = getattr(self._env, f"_arm_ik_joint_ids_{getattr(self, '_side_rb', 'right')}", None) or getattr(
                self._env, "_arm_ik_joint_ids_right", None
            )
            if not ids:
                return
            q = self._inner._asset.data.joint_pos[:, ids]
            buf = getattr(self._env, "_arm_jerk_buf", None)
            if buf is None or len(buf) == 0 or buf[-1].shape != q.shape:
                self._env._arm_jerk_buf = [q.clone()]
                return
            buf.append(q.clone())
            if len(buf) < 4:
                return
            del buf[:-4]
            dt = float(self._env.step_dt)
            # third difference: q[t] - 3q[t-1] + 3q[t-2] - q[t-3]
            d3 = (buf[3] - 3.0 * buf[2] + 3.0 * buf[1] - buf[0]).abs().amax(dim=-1) / (dt**3)
            acc = getattr(self._env, "_arm_jerk_acc", None)
            if acc is None:
                acc = self._env._arm_jerk_acc = {"sum": 0.0, "n": 0, "max": 0.0}
            acc["sum"] += float(d3.mean())
            acc["n"] += 1
            acc["max"] = max(acc["max"], float(d3.max()))
        except (AttributeError, IndexError, KeyError, RuntimeError, TypeError) as _exc:
            note_once("ARM_JERK measure", _exc)

    def _apply_arm_feedforward(self) -> None:
        """Add gravity-comp effort (all arm joints) + reference-velocity target (this term's arm) on top
        of the inner arm-IK position PD. Env-var gated diagnostic (DEXENGINE_ARM_FF=1)."""
        robot = self._inner._asset
        self._resolve_ff_joint_ids(robot)
        # (a) gravity compensation effort on all arm joints. Accumulated into `eff_ff` so the
        # torque-mode velocity feedforward (below) can be added in ONE effort write — two separate
        # set_joint_effort_target calls would have the second overwrite the first.
        eff_ff = None
        try:
            gc = robot.root_physx_view.get_gravity_compensation_forces()[:, self._ff_arm_all_ids]
            eff_ff = self._ff_grav_sign * gc
        except Exception:  # noqa: BLE001 — the PhysX query is optional; a failure turns the feedforward term off
            eff_ff = None
        # (b) velocity feedforward on THIS TERM'S arm from the all-frame IK cache finite-difference.
        # The cache is published under the canonical side-suffixed name AND (for the active side)
        # the legacy `_arm_ik_cache_all_right`; resolve canonical-first so a LEFT-uni run engages
        # the FF instead of silently no-oping on a right-only attr name.
        # In split mode each side's term owns its side's articulation, and the left cache and ids are
        # published under the left names, found on that same articulation (mdp/events): both arms take
        # the lead, and the legacy right slot (the other articulation) is never borrowed.
        from dexengine.rl.tasks.scene import split_artic as _split_artic_ffb

        _sd_ff_side = getattr(self, "_side_rb", "right")
        cache_all = getattr(self._env, f"_arm_ik_cache_all_{_sd_ff_side}", None)
        cache_len_t = getattr(self._env, f"_arm_ik_cache_all_len_{_sd_ff_side}", None)
        _cache_src = f"_arm_ik_cache_all_{_sd_ff_side}"
        if cache_all is None and not _split_artic_ffb.enabled():
            cache_all = getattr(self._env, "_arm_ik_cache_all_right", None)
            cache_len_t = getattr(self._env, "_arm_ik_cache_all_len_right", None)
            _cache_src = "_arm_ik_cache_all_right (legacy active-side)"
        # NEVER silently no-op: if the FF was requested but cannot engage, say so once and loudly.
        if not self._ff_warned and (cache_all is None or not self._ff_vel_ids):
            self._ff_warned = True
            _ff_reason = "no all-frame IK cache" if cache_all is None else "no arm joint ids"
            print(
                f"[ResidualArmIKAction] WARNING DEXENGINE_ARM_FF=1 but the VELOCITY feedforward "
                f"CANNOT engage for side='{_sd_ff_side}' -> position-PD only "
                f"(lag = (Kd/Kp)*qdot remains). reason: "
                f"{_ff_reason}. "
                f"Gravity compensation is still applied.",
                flush=True,
            )
        self._apply_arm_feedforward_effort(_cache_src, _sd_ff_side, cache_all, cache_len_t, eff_ff, robot)

    def _apply_arm_feedforward_effort(self, _cache_src, _sd_ff_side, cache_all, cache_len_t, eff_ff, robot):
        """Reference velocity (central difference), the feedforward effort and its write (statements unchanged)."""
        if cache_all is not None and self._ff_vel_ids:
            tm = self._traj_mgr
            elen = self._env.episode_length_buf
            if self._wrist_lookahead > 0:
                # WRIST LOOKAHEAD: the velocity FF must describe the SAME shifted
                # trajectory point as the position target (ref velocity at t+k).
                _sd_ff = float(getattr(tm, "_slowdown_factor", 1.0) or 1.0)
                elen = elen + int(round(self._wrist_lookahead * _sd_ff))
            f0 = tm._abs_frames(elen)  # clamped absolute frame (slowdown-aware)
            clip_ids = tm.clip_ids
            clen = cache_len_t[clip_ids]
            slowdown = float(getattr(tm, "_slowdown_factor", 1.0) or 1.0)
            dt_frame = max(slowdown, 1.0) * float(self._env.step_dt)
            _m, qd_ref, tau_ff = self._reference_derivatives(cache_all, eff_ff, robot, clen, clip_ids, dt_frame, f0)
            if _m in ("torque", "both", "ct"):
                if self._ff_damping is None:
                    _damp_all = robot.root_physx_view.get_dof_dampings().to(qd_ref.device)
                    self._ff_damping = _damp_all[:, self._ff_vel_ids]
                    self._ff_vel_pos_in_all = [self._ff_arm_all_ids.index(j) for j in self._ff_vel_ids]
                tau_ff = self._ff_damping * qd_ref
                if eff_ff is not None:
                    eff_ff[:, self._ff_vel_pos_in_all] = eff_ff[:, self._ff_vel_pos_in_all] + tau_ff
            if not self._ff_logged:
                self._ff_logged = True
                _rb = robot.data.joint_vel_target[0, self._ff_vel_ids].abs().mean().item()
                print(
                    f"[ResidualArmIKAction] ARM_FF ENGAGED side='{_sd_ff_side}' mode={self._ff_mode} "
                    f"cache={_cache_src} shape={tuple(cache_all.shape)} ids={self._ff_ids_src} "
                    f"n_ids={len(self._ff_vel_ids)} dt_frame={dt_frame:.4f}s vel_scale={self._ff_vel_scale} "
                    f"|qd_ref| mean={qd_ref.abs().mean().item():.4f} max={qd_ref.abs().max().item():.4f} rad/s "
                    f"| vel_target READBACK |.|mean={_rb:.4f} rad/s"
                    + (
                        f" | tau_ff |.|mean={tau_ff.abs().mean().item():.3f} Nm "
                        f"(Kd={self._ff_damping.mean().item():.1f})"
                        if tau_ff is not None
                        else ""
                    )
                    + (
                        f" | lead Kd/Kp mean={self._ff_kdkp.mean().item():.4f}s -> |dq| "
                        f"mean={(self._ff_kdkp * qd_ref).abs().mean().item():.4f} rad"
                        if self._ff_kdkp is not None
                        else ""
                    ),
                    flush=True,
                )
        # single effort write: gravity compensation (+ torque-mode velocity feedforward)
        if eff_ff is not None:
            robot.set_joint_effort_target(eff_ff, joint_ids=self._ff_arm_all_ids)

    def _reference_derivatives(self, cache_all, eff_ff, robot, clen, clip_ids, dt_frame, f0):
        """REFERENCE DERIVATIVE: a 1-frame difference of the arm-IK cache is DOMINATED BY RETARGET NOISE — measured on
        141: single-frame joint jumps up to 0.231 rad (13.2 deg) give |qd_ref| max 11.56 rad/s (mean 0.46), and
        feeding that into any feedforward injects exactly the tremble we must avoid (lead mode jerk 3.8x). A CENTRAL
        difference over +-n frames removes it (max 11.56 -> 3.59 rad/s, spike rate 1.07% -> 0.22%) while preserving
        the real motion. n = self._ff_smooth (0 = the legacy 1-frame forward difference).
        """
        _n = self._ff_smooth
        if _n > 0:
            f_hi = torch.minimum(f0 + _n, clen - 1)
            f_lo = torch.clamp(f0 - _n, min=0)
            span = (f_hi - f_lo).clamp(min=1).to(cache_all.dtype) * dt_frame
            qd_ref = (cache_all[clip_ids, f_hi] - cache_all[clip_ids, f_lo]) / span.unsqueeze(-1)
            # second central difference for the inertial feedforward term
            qdd_ref = (cache_all[clip_ids, f_hi] - 2.0 * cache_all[clip_ids, f0] + cache_all[clip_ids, f_lo]) / (
                (_n * dt_frame) ** 2
            )
        else:
            f1 = torch.minimum(f0 + 1, clen - 1)
            qd_ref = (cache_all[clip_ids, f1] - cache_all[clip_ids, f0]) / dt_frame
            qdd_ref = torch.zeros_like(qd_ref)
        qd_ref = qd_ref * self._ff_vel_scale
        qd_ref, qdd_ref = self._park_still(qd_ref, qdd_ref)  # parked env: the reference does not move (no lead)
        _m = self._feedforward_channel(robot, qd_ref, qdd_ref)
        if _m in ("idff", "ct"):
            # INVERSE-DYNAMICS (computed-torque) FEEDFORWARD: tau_ff = M(q)*qdd_ref + G(q),
            # delivered through the effort channel while the PD gains stay UNCHANGED, so the
            # arm generates the reference motion itself and the PD only corrects error ->
            # tracking improves without raising stiffness (i.e. without the tremble that
            # Kp400/Kp900 showed: jerk 3.9x/6.9x with residual ON).
            # G(q) is already in eff_ff (gravity comp). The mass-matrix API is probed ONCE and
            # reported; if absent this degrades to gravity-only and says so loudly.
            if not self._ff_idff_probed:
                self._ff_idff_probed = True
                _v = robot.root_physx_view
                for _nm in ("get_generalized_mass_matrices", "get_mass_matrices"):
                    if hasattr(_v, _nm):
                        self._ff_massfn = getattr(_v, _nm)
                        break
                print(
                    f"[ResidualArmIKAction] ARM_FF idff probe: mass-matrix API="
                    f"{self._ff_massfn.__name__ if self._ff_massfn else 'NONE (gravity-only)'}"
                    f" | coriolis API={'yes' if hasattr(_v, 'get_coriolis_and_centrifugal_forces') else 'no'}",
                    flush=True,
                )
            if self._ff_massfn is not None and eff_ff is not None:
                try:
                    _M = self._ff_massfn().to(qdd_ref.device)  # (N, nd, nd) or (N, nd*nd)
                    if _M.dim() == 2:
                        _nd = int(round(_M.shape[-1] ** 0.5))
                        _M = _M.view(-1, _nd, _nd)
                    _ids = torch.as_tensor(self._ff_vel_ids, dtype=torch.long, device=_M.device)
                    _Msub = _M[:, _ids][:, :, _ids]  # (N, 7, 7) arm block
                    _tau_in = torch.einsum("nij,nj->ni", _Msub, qdd_ref)
                    if self._ff_vel_pos_in_all is None:
                        self._ff_vel_pos_in_all = [self._ff_arm_all_ids.index(j) for j in self._ff_vel_ids]
                    eff_ff[:, self._ff_vel_pos_in_all] = eff_ff[:, self._ff_vel_pos_in_all] + _tau_in
                except Exception as _e:  # noqa: BLE001 — the PhysX query is optional; a failure turns the feedforward term off
                    if not self._ff_warned:
                        self._ff_warned = True
                        print(
                            f"[ResidualArmIKAction] WARNING idff inertial term FAILED ({_e}) "
                            f"-> gravity-only feedforward",
                            flush=True,
                        )
        tau_ff = None
        return _m, qd_ref, tau_ff

    def _feedforward_channel(self, robot, qd_ref, qdd_ref):
        """CHANNEL: the position-PD lag is e_ss = (Kd*qdot + tau_load)/Kp. Two ways to cancel the Kd*qdot part: 'vel'
        -> joint VELOCITY target = qd_ref, so the PhysX drive's damping term reads Kd*(qd_ref - qdot) ~ 0. Measured
        (031, FF on vs off): lag ratio 1.00 vs 1.03 => this channel had NO effect in this configuration. 'torque' ->
        add tau_ff = Kd_joint * qd_ref through the EFFORT channel (the same one gravity compensation uses), which
        cancels the damping brake explicitly. DEXENGINE_ARM_FF_MODE=vel|torque|both (default vel = previous
        behaviour). MODES: vel=velocity target | lead=position lead | torque=Kd*qd_ref effort idff=M*qdd_ref+G |
        ct=FULL computed torque (G + M*qdd_ref + Kd*qd_ref). ct exists because idff alone barely moved the error
        (-1%): the dominant lag term is the drive's own -Kd*qdot damping, which inverse dynamics does not touch.
        Classical computed torque has the damping act on the VELOCITY ERROR, so the +Kd*qd_ref term is required as
        well. All derivatives are central-differenced (see above) so this does not re-inject the retarget spikes that
        made the unsmoothed torque mode destructive.
        """
        _m = self._ff_mode
        if _m in ("vel", "both"):
            robot.set_joint_velocity_target(qd_ref, joint_ids=self._ff_vel_ids)
        if _m in ("lead", "lead2"):
            # POSITION-CHANNEL LEAD — the one channel PhysX definitely honours.
            # The implicit drive force is Kp*(q_t - q) - Kd*qdot, so tracking a moving
            # reference costs a standing error e_ss = (Kd/Kp)*qdot (measured: regression
            # b = 0.286 at Kp100, 0.156 at Kp200 — exactly Kd/Kp). Adding that offset to the
            # POSITION target makes the drive read Kp*(q_ref - q) + Kd*(qd_ref - qdot), i.e.
            # algebraically the velocity feedforward, but delivered through the position
            # channel instead of set_joint_velocity_target (measured inert) or the effort
            # channel (measured destructive: 100% -> 0%).
            # Why not just more lookahead: the lag is velocity-PROPORTIONAL, so a fixed k
            # frames over-leads where the reference is slow and under-leads where it is fast
            # (k=10 was catastrophic, k=5 best, in the earlier sweep). This lead is exact at
            # every speed.
            if self._ff_kdkp is None:
                _st = robot.root_physx_view.get_dof_stiffnesses().to(qd_ref.device)[:, self._ff_vel_ids]
                _dp = robot.root_physx_view.get_dof_dampings().to(qd_ref.device)[:, self._ff_vel_ids]
                self._ff_kdkp = _dp / _st.clamp(min=1e-6)
            q_now = robot.data.joint_pos_target[:, self._ff_vel_ids]
            _dq = self._ff_kdkp * qd_ref
            if _m == "lead2":
                # SECOND-ORDER (inertial) lead. After the velocity lead the residual
                # base-following error is still 2.5x the reference noise floor (0.0374 vs
                # 0.0151 rad on 031), i.e. real dynamics, not noise: accelerating the arm
                # needs torque M*qdd, which a position PD can only produce by holding an
                # error of (M/Kp)*qdd. Add that offset explicitly, using the REAL diagonal
                # inertia from PhysX, still through the position channel (the effort channel
                # failed three times: torque/idff/ct).
                if self._ff_minv is None:
                    _v2 = robot.root_physx_view
                    _fn = getattr(_v2, "get_generalized_mass_matrices", None)
                    _st2 = _v2.get_dof_stiffnesses().to(qd_ref.device)[:, self._ff_vel_ids]
                    if _fn is not None:
                        _M2 = _fn().to(qd_ref.device)
                        if _M2.dim() == 2:
                            _nd2 = int(round(_M2.shape[-1] ** 0.5))
                            _M2 = _M2.view(-1, _nd2, _nd2)
                        _idx2 = torch.as_tensor(self._ff_vel_ids, dtype=torch.long, device=_M2.device)
                        _Md = _M2[:, _idx2, _idx2]  # diagonal inertia of the arm joints
                        self._ff_minv = _Md / _st2.clamp(min=1e-6)
                        print(
                            f"[ResidualArmIKAction] lead2 inertial term: M_diag/Kp mean="
                            f"{self._ff_minv.mean().item():.5f} s^2 (M_diag mean={_Md.mean().item():.4f})",
                            flush=True,
                        )
                    else:
                        self._ff_minv = torch.zeros_like(_st2)
                        print(
                            "[ResidualArmIKAction] WARNING lead2: no mass-matrix API -> velocity lead only",
                            flush=True,
                        )
                _dq = _dq + self._ff_minv * qdd_ref * self._ff_acc_scale
            robot.set_joint_position_target(q_now + _dq, joint_ids=self._ff_vel_ids)
        return _m

    def _resolve_ff_joint_ids(self, robot):
        """Resolve (once) the arm joint ids the feedforward drives and the velocity-feedforward subset."""
        if self._ff_arm_all_ids is None:
            # GATED split-articulation eval: each residual action term's asset holds only
            # ONE arm (right term -> `robot` right view; left term -> `robot_left`), so a
            # two-sided regex would fail the strict find_joints match. Query only the
            # arm joints present in THIS term's articulation. Unset -> both regexes match
            # the shared articulation (byte-identical to the original two-sided query).
            # The arm joint regexes come from the active ArmSpec (an arm literal once killed find_joints on rby1).
            # sides=("left","right") keeps the two-side list byte-identical.
            from dexengine.core.arms import ARM_REGISTRY as _ARM_REG_FF
            from dexengine.rl.tasks.mdp.naming import _active_arm_name as _arm_name_ff
            from dexengine.rl.tasks.scene import split_artic as _split_artic_ff

            _spec_ff = _ARM_REG_FF[_arm_name_ff()]
            if _split_artic_ff.enabled():
                _side = getattr(self, "_side_rb", "right")
                _ff_regex = [_spec_ff.joint_expr(_side)]
            else:
                _ff_regex = [_spec_ff.joint_expr(s) for s in _spec_ff.sides]
            all_ids, _ = robot.find_joints(_ff_regex)
            self._ff_arm_all_ids = all_ids
            # Velocity-FF joint ids, ordered to match the IK-cache columns. mdp/events writes BOTH
            # the canonical side-suffixed name and the legacy `_arm_ik_joint_ids_right` (which has
            # always held the ACTIVE side's ids, whichever side that is — the source of the naming
            # confusion this fix removes). Prefer canonical, accept legacy, and report which.
            _sd_ids = getattr(self, "_side_rb", "right")
            self._ff_vel_ids = getattr(self._env, f"_arm_ik_joint_ids_{_sd_ids}", None)
            self._ff_ids_src = f"_arm_ik_joint_ids_{_sd_ids}"
            if not self._ff_vel_ids and not _split_artic_ff.enabled():
                self._ff_vel_ids = getattr(self._env, "_arm_ik_joint_ids_right", None)
                self._ff_ids_src = "_arm_ik_joint_ids_right (legacy active-side)"
