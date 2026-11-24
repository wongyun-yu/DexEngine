"""Task termination conditions for object manipulation training.

Extends the hand-tracking terminations with object tracking checks:
- Object position error exceeds threshold (with tightening)
- Object rotation error exceeds threshold (with tightening)
- Object velocity explosion
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

from dexengine.core.desk import DESK_TOP_SURFACE_Z
from dexengine.rl.tasks import env_flags

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_GRACE_PERIOD = 8  # ManipTrans task grace period (shorter than hand-tracking's 20)


class ObjectTrackingExceeded(ManagerTermBase):
    """Terminate if object tracking error exceeds threshold.

    Position and rotation thresholds anneal linearly from a loose start value
    to a tight end value over training, driven by the curriculum's
    schedule_progress in [0,1] (0 = start/loose, 1 = end/final/tight):

        effective = start + (end - start) * schedule_progress
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._traj_mgr = env.trajectory_manager
        self._side = cfg.params.get("side", "right")
        # Which scene rigid object this termination reads.
        # Default 'object_right'; the left
        # termination passes object_key='object_left'.
        self._object_key = cfg.params.get("object_key", "object_right")
        # adaptive-term v3: object-local TIP keypoint (farthest FPS kp, [x,y,z]) + desk height for
        # the eval-capture tip-drag statistic. None (default) = tip capture off (byte-identical).
        self._tip_kp_local = cfg.params.get("tip_kp_local", None)
        self._tip_desk_z = float(cfg.params.get("tip_desk_z", DESK_TOP_SURFACE_Z))
        self._tip_kp_t = None  # lazy device tensor

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        pos_start: float = 0.04,
        pos_end: float = 0.02,
        rot_start_deg: float = 60.0,
        rot_end_deg: float = 30.0,
        side: str = "right",
        grace_period: int = _GRACE_PERIOD,
        object_key: str = "object_right",
        tip_kp_local=None,  # consumed via self._tip_kp_local; declared for param-validation
        tip_desk_z: float = DESK_TOP_SURFACE_Z,
        snap_only: bool = False,
        lag_window: int = 0,
        lag_window_ahead: int = 0,
        judge_off: bool = False,
    ) -> torch.Tensor:
        grace_mask = env.episode_length_buf >= grace_period
        if judge_off:
            # Judge measurement arm (owner rl.judge.terms.OBJTRACK_VAR; fresh-process judge only, wired by
            # env_setup.objtrack.apply_judge_objtrack_term): grace is treated as not over, so done, the cause masks and
            # the PARK stash are all false (only the termination is withheld; error computation and capture stay).
            # Default False = byte-identical.
            grace_mask = torch.zeros_like(grace_mask)

        obj = env.scene[self._object_key]
        origins = env.scene.env_origins

        done = self._current_object_state(
            env, pos_start, pos_end, rot_start_deg, rot_end_deg, lag_window, lag_window_ahead, grace_mask, obj, origins
        )
        # RESET SCHEME=decoupled: SNAP-ONLY mode — this gate no longer
        # terminates the episode. The fail mask is OR-accumulated into the per-side
        # buffer that ArmResidualTrainEnv consumes (side snap at the top of the next
        # step + done_right/done_left extras for the per-side GAE). Default False =
        # byte-identical.
        if snap_only:
            # Post-snap grace: gates disarmed for _pss_grace steps after this side's
            # last snap (airborne-teleport catch window; anti snap-treadmill).
            _last = getattr(env, "_pss_last_snap", None)
            if _last is not None:
                _g = int(getattr(env, "_pss_grace", 0))
                _i = 1 if self._side == "left" else 0
                done = done & ((int(env.common_step_counter) - _last[:, _i]) > _g)
            _attr = f"_side_fail_{self._side}"
            buf = getattr(env, _attr, None)
            if buf is None or buf.shape != done.shape:
                buf = torch.zeros_like(done)
            # RESET_SCHEME=park: a side already parked (following kinematically) ignores violations; a violation on this
            # side while the other side is already parked returns a **termination** (both sides failed = end of the
            # episode). Any other violation books a park (stash).
            _parked = getattr(env, "_parked", None)
            if _parked is not None and getattr(env, "_park_on", False):
                _other = "left" if self._side == "right" else "right"
                done = done & ~_parked[self._side]
                _kill = done & _parked[_other]
                setattr(env, _attr, buf | (done & ~_kill))
                return _kill
            setattr(env, _attr, buf | done)
            return torch.zeros_like(done)
        return done

    def _current_object_state(
        self,
        env,
        pos_start,
        pos_end,
        rot_start_deg,
        rot_end_deg,
        lag_window,
        lag_window_ahead,
        grace_mask,
        obj,
        origins,
    ):
        """Current object state."""
        obj_pos = obj.data.root_pos_w - origins
        obj_quat = obj.data.root_quat_w

        # Reference state
        ref_state = self._traj_mgr.get_reference_state(env.episode_length_buf)
        ref_pos = ref_state[f"{self._side}_obj_pos"]
        ref_quat = ref_state[f"{self._side}_obj_quat"]

        return self._linear_threshold_schedule_start(
            env,
            pos_start,
            pos_end,
            rot_start_deg,
            rot_end_deg,
            lag_window,
            lag_window_ahead,
            grace_mask,
            obj_pos,
            obj_quat,
            ref_pos,
            ref_quat,
        )

    def _linear_threshold_schedule_start(
        self,
        env,
        pos_start,
        pos_end,
        rot_start_deg,
        rot_end_deg,
        lag_window,
        lag_window_ahead,
        grace_mask,
        obj_pos,
        obj_quat,
        ref_pos,
        ref_quat,
    ):
        """Linear threshold schedule: start (loose) -> end (tight) over training."""
        progress = self._traj_mgr.schedule_progress

        # Position check
        pos_err = torch.norm(ref_pos - obj_pos, dim=-1)
        effective_pos_thresh = pos_start + (pos_end - pos_start) * progress
        # Rotation check
        from isaaclab.utils.math import quat_conjugate, quat_mul

        quat_err = quat_mul(quat_conjugate(obj_quat), ref_quat)
        angle_err = 2.0 * torch.acos(torch.clamp(torch.abs(quat_err[:, 0]), max=1.0))
        rot_start_rad = rot_start_deg * 3.14159265 / 180.0
        rot_end_rad = rot_end_deg * 3.14159265 / 180.0
        effective_rot_thresh = rot_start_rad + (rot_end_rad - rot_start_rad) * progress

        pos_exceeded = pos_err > effective_pos_thresh
        rot_exceeded = angle_err > effective_rot_thresh
        # Stash the per-component fire masks (pos vs rot, after grace) so the training loop can log
        # object_tracking termination cause SEPARATELY (Termination/object_tracking_pos_percent vs
        # ..._rot_percent). The combined term itself is unchanged: done = pos_fired | rot_fired.
        env._objtrack_pos_fired = pos_exceeded & grace_mask
        env._objtrack_rot_fired = rot_exceeded & grace_mask

        # LAG-WINDOW gate: survive if ANY reference frame in the
        # trailing window [F-W, F] pose-matches the CURRENT object pose — pos AND rot at the
        # SAME window frame (no mixing pos from one frame with rot from another). Tolerates up
        # to W frames of following lag (W=25 = 0.5 s @ 50 Hz); the flip side is a never-moved
        # object also survives ~W frames longer (window still contains not-yet-departed
        # reference poses). lag_window=0 (default) -> byte-identical instantaneous gate.
        # pos_err/angle_err above stay instantaneous (k=0) for the capture/debug paths; the
        # fired masks become min-over-window exceedance (cause diagnostics — under the window
        # they may not sum to the combined count, since done = "no window frame fully matched").
        # lag_window_ahead: with the wrist-lookahead base (ref(t+k)) the
        # object may ARRIVE EARLY by up to k frames when the phase-lead overcorrects
        # (actual lag < k); the forward extension [F+1, F+ahead] keeps the gate from
        # punishing early arrival. 0 (default) = byte-identical trailing-only window.
        _lag_w = int(lag_window)
        _lag_a = int(lag_window_ahead)
        _lag_matched = None
        if _lag_w > 0:
            win_pos, win_quat = self._traj_mgr.object_pose_window(
                self._side, env.episode_length_buf, _lag_w, ahead=_lag_a
            )
            _d_pos = torch.norm(win_pos - obj_pos.unsqueeze(1), dim=-1)  # (N, W+1)
            _dot = torch.abs((win_quat * obj_quat.unsqueeze(1)).sum(dim=-1))
            _d_ang = 2.0 * torch.acos(torch.clamp(_dot, max=1.0))  # (N, W+1)
            _lag_matched = ((_d_pos <= effective_pos_thresh) & (_d_ang <= effective_rot_thresh)).any(dim=1)
            pos_exceeded = _d_pos.min(dim=1).values > effective_pos_thresh
            rot_exceeded = _d_ang.min(dim=1).values > effective_rot_thresh
            env._objtrack_pos_fired = pos_exceeded & grace_mask
            env._objtrack_rot_fired = rot_exceeded & grace_mask

        self._capture_first_episode_peaks(env, angle_err, obj_pos, obj_quat, pos_err, ref_pos, ref_quat)
        self._objtrack_debug_dump(env, angle_err, grace_mask, obj_pos, pos_err, pos_exceeded, ref_pos, rot_exceeded)
        if _lag_matched is not None:
            done = (~_lag_matched) & grace_mask
        else:
            done = (pos_exceeded | rot_exceeded) & grace_mask
        return done

    def _capture_first_episode_peaks(self, env, angle_err, obj_pos, obj_quat, pos_err, ref_pos, ref_quat):
        """Eval capture of the per-env tracking errors (statements unchanged)."""
        # adaptive-term (v2->v3): per-env FIRST-episode peak-error (+v3 tip-drag) capture for the
        # ObjTermAdaptive curriculum. Each frozen-eval probe counts only an env's FIRST episode
        # (early finishers auto-reset but are then ignored). GATED by env._objtrack_eval_capture
        # (False in training -> byte-identical no-op). ALL state lives on `env`, so
        # frozen_eval_snapshot can reset (set _objtrack_cap_prev_el=None) and read
        # (env._objtrack_first_peaks_* / _objtrack_first_drag) without locating this instance.
        if getattr(env, "_objtrack_eval_capture", False):
            el = env.episode_length_buf
            ang_deg = angle_err * (180.0 / 3.14159265)
            prev_el = getattr(env, "_objtrack_cap_prev_el", None)
            if prev_el is None or prev_el.shape != el.shape:
                N = el.shape[0]
                dev = el.device
                env._objtrack_cap_peak_pos = torch.zeros(N, device=dev)
                env._objtrack_cap_peak_rot = torch.zeros(N, device=dev)
                env._objtrack_cap_first_done = torch.zeros(N, dtype=torch.bool, device=dev)
                env._objtrack_first_peaks_pos = torch.zeros(N, device=dev)
                env._objtrack_first_peaks_rot = torch.zeros(N, device=dev)
                # v3 tip-drag counters (knife): ref-tip-airborne steps vs sim-tip-still-on-desk steps
                env._objtrack_cap_air = torch.zeros(N, device=dev)
                env._objtrack_cap_drag = torch.zeros(N, device=dev)
                env._objtrack_first_drag = torch.full((N,), float("nan"), device=dev)
                env._objtrack_cap_prev_el = el.clone()
                prev_el = env._objtrack_cap_prev_el
            first_done = env._objtrack_cap_first_done
            reset_mask = (el < prev_el) & (~first_done)
            if reset_mask.any():
                env._objtrack_first_peaks_pos[reset_mask] = env._objtrack_cap_peak_pos[reset_mask]
                env._objtrack_first_peaks_rot[reset_mask] = env._objtrack_cap_peak_rot[reset_mask]
                env._objtrack_first_drag[reset_mask] = env._objtrack_cap_drag[reset_mask] / env._objtrack_cap_air[
                    reset_mask
                ].clamp(min=1.0)
                first_done[reset_mask] = True
            upd = ~first_done
            env._objtrack_cap_peak_pos[upd] = torch.maximum(env._objtrack_cap_peak_pos[upd], pos_err[upd])
            env._objtrack_cap_peak_rot[upd] = torch.maximum(env._objtrack_cap_peak_rot[upd], ang_deg[upd])
            # v3: tip z above desk from the SAME obj/ref poses (kp = farthest FPS keypoint, wired by
            # train_cfg_overrides via term params). Skipped when tip_kp_local is None.
            if self._tip_kp_local is not None:
                if self._tip_kp_t is None:
                    self._tip_kp_t = torch.tensor(
                        [float(x) for x in self._tip_kp_local], dtype=torch.float32, device=el.device
                    )
                from isaaclab.utils.math import quat_apply

                _kpe = self._tip_kp_t.unsqueeze(0).expand(el.shape[0], 3)
                _tip_sim_z = (obj_pos + quat_apply(obj_quat, _kpe))[:, 2]
                _tip_ref_z = (ref_pos + quat_apply(ref_quat, _kpe))[:, 2]
                _air = (_tip_ref_z > self._tip_desk_z + 0.05) & upd
                env._objtrack_cap_air[_air] += 1.0
                env._objtrack_cap_drag[_air & (_tip_sim_z < self._tip_desk_z + 0.02)] += 1.0
            prev_el.copy_(el)

    def _objtrack_debug_dump(self, env, angle_err, grace_mask, obj_pos, pos_err, pos_exceeded, ref_pos, rot_exceeded):
        """DEXENGINE_OBJTRACK_DEBUG=1 dump of the tracking errors (statements unchanged)."""
        # DEXENGINE_OBJTRACK_DEBUG=1: per-step env-0 trace of the EXACT pos/rot error vs thresholds and
        # which component crosses (eval diagnostic; no behavior change). The last printed step before
        # DONE=1 is the termination frame; its pos_exc/rot_exc says whether pos or rot fired.
        if env_flags.get("DEXENGINE_OBJTRACK_DEBUG") == "1":
            _dbg_n = 4  # env 0..3 (env index = clip index in multi-object runs)
            # once: which asset really stands under the object prim of env0..N (the USD truth)
            if not getattr(env, f"_dbg_prim_dumped_{self._side}", False):
                setattr(env, f"_dbg_prim_dumped_{self._side}", True)
                try:
                    _stage = env.scene.stage
                    _ename = "object_left" if self._side == "left" else "object_right"
                    _tmpl = env.scene[_ename].cfg.prim_path if _ename in env.scene.keys() else None  # noqa: SIM118  — InteractiveScene has no __contains__
                    for _e in range(min(_dbg_n, env.num_envs)):
                        _pp = str(_tmpl).replace("env_.*", f"env_{_e}") if _tmpl else f"/World/envs/env_{_e}/Object"
                        _prim = _stage.GetPrimAtPath(_pp)
                        _kids = [c.GetName() for c in _prim.GetChildren()][:4] if _prim else "(none)"
                        _refs = []
                        if _prim:
                            from pxr import Usd as _U

                            for _c in _U.PrimRange(_prim):
                                _pl = _c.GetPrimStack()
                                for _sp in _pl:
                                    _idp = _sp.layer.identifier
                                    if "objects_taco" in _idp:
                                        _tok = _idp.split("objects_taco/")[-1].split("/")[0]
                                        if _tok not in _refs:
                                            _refs.append(_tok)
                                if len(_refs) >= 2:
                                    break
                        print(
                            f"[PRIM_DBG] side={self._side} env={_e} path={_pp} kids={_kids} obj_usd={_refs}", flush=True
                        )
                except Exception as _pe:  # noqa: BLE001 — debug print only
                    print(f"[PRIM_DBG] side={self._side} failed: {_pe}", flush=True)
            # env 0..3 + side label + clip assignment, to tell apart the (env, side) pattern of a bimanual multi-clip
            # t=0 contamination. Diagnostic only: no behaviour change.
            _tm = getattr(env, "_trajectory_manager", None) or getattr(env, "trajectory_manager", None)
            _ci = getattr(_tm, "clip_ids", None) if _tm is not None else None
            for _i in range(min(_dbg_n, pos_err.shape[0])):
                _done = bool(((pos_exceeded | rot_exceeded) & grace_mask)[_i].item())
                _c = int(_ci[_i].item()) if _ci is not None else -1
                _op = obj_pos[_i] - env.scene.env_origins[_i]
                _rp = ref_pos[_i] - env.scene.env_origins[_i]
                print(
                    f"[OBJTRACK_DBG] side={self._side} env={_i} clip={_c} "
                    f"step={int(env.episode_length_buf[_i].item())} "
                    f"pos_err={pos_err[_i].item() * 100:.2f}cm "
                    f"rot_err={angle_err[_i].item() * 57.29578:.1f}deg "
                    f"obj=({_op[0]:.3f},{_op[1]:.3f},{_op[2]:.3f}) "
                    f"ref=({_rp[0]:.3f},{_rp[1]:.3f},{_rp[2]:.3f}) DONE={int(_done)}",
                    flush=True,
                )


class ObjectTrackingSplitDiag(ManagerTermBase):
    """Diagnostic (NOT a termination): recomputes the ObjectTrackingExceeded pos/rot
    components so pos- vs rot-driven termination can be separated in logs / W&B.

    Register as a zero-ish-weight REWARD term, one per ``component``:
      "pos_err" -> object position error (m)
      "rot_err" -> object rotation error (deg)
      "pos_exc" -> 1.0 if pos error exceeds the (annealed) threshold else 0.0  (× grace)
      "rot_exc" -> 1.0 if rot error exceeds the (annealed) threshold else 0.0  (× grace)

    Uses the SAME ``get_reference_state(episode_length_buf)``, quaternion error, annealed
    thresholds and grace as ``ObjectTrackingExceeded`` — wire it with the SAME (post-scale)
    pos_start/pos_end/rot_start_deg/rot_end_deg params so the split matches the live
    termination exactly. Recomputes (no 1-step stash lag).
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._traj_mgr = env.trajectory_manager
        self._side = cfg.params.get("side", "right")
        # Bimanual Option 2: which scene rigid object this diagnostic reads.
        # Default 'object' keeps the right-only path byte-identical.
        self._object_key = cfg.params.get("object_key", "object_right")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        component: str = "rot_err",
        pos_start: float = 0.04,
        pos_end: float = 0.02,
        rot_start_deg: float = 60.0,
        rot_end_deg: float = 30.0,
        side: str = "right",
        grace_period: int = _GRACE_PERIOD,
        object_key: str = "object_right",
    ) -> torch.Tensor:
        from isaaclab.utils.math import quat_conjugate, quat_mul

        obj = env.scene[self._object_key]
        origins = env.scene.env_origins
        obj_pos = obj.data.root_pos_w - origins
        obj_quat = obj.data.root_quat_w
        ref_state = self._traj_mgr.get_reference_state(env.episode_length_buf)
        ref_pos = ref_state[f"{self._side}_obj_pos"]
        ref_quat = ref_state[f"{self._side}_obj_quat"]
        progress = self._traj_mgr.schedule_progress

        pos_err = torch.norm(ref_pos - obj_pos, dim=-1)  # (N,) m
        quat_err = quat_mul(quat_conjugate(obj_quat), ref_quat)
        angle_err = 2.0 * torch.acos(torch.clamp(torch.abs(quat_err[:, 0]), max=1.0))  # (N,) rad
        eff_pos = pos_start + (pos_end - pos_start) * progress
        rot_start_rad = rot_start_deg * 3.14159265 / 180.0
        rot_end_rad = rot_end_deg * 3.14159265 / 180.0
        eff_rot = rot_start_rad + (rot_end_rad - rot_start_rad) * progress
        grace = (env.episode_length_buf >= grace_period).float()

        if component == "pos_err":
            return pos_err
        if component == "rot_err":
            return angle_err * 180.0 / 3.14159265  # deg
        if component == "pos_exc":
            return (pos_err > eff_pos).float() * grace
        if component == "rot_exc":
            return (angle_err > eff_rot).float() * grace
        return torch.zeros(env.num_envs, device=env.device)
