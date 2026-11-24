"""Custom termination functions for bimanual training.

Termination logic:
- Fingertip tracking: 5 fingertip distance thresholds (linearly scheduled)
- Velocity sanity: wrist, joints_vel, dof_vel checks
- Grace period: all tracking terminations gated on episode_length >= 20
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


from dexengine.rl.tasks.mdp.naming import resolve_palm_body_id


def trajectory_done(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate when trajectory clip is complete.

    Checks if the current frame (episode_length_buf) exceeds the clip length
    for each environment. This supports variable-length clips.
    """
    done = env.trajectory_manager.is_trajectory_done(env.episode_length_buf)
    return done


# MuJoCo FK body index mapping (per hand, 13 bodies) — same as in rewards.py
_THUMB_TIP_IDX = 4  # thumb_distal
_INDEX_TIP_IDX = 6  # index_intermediate
_MIDDLE_TIP_IDX = 8  # middle_intermediate
_RING_TIP_IDX = 10  # ring_intermediate
_PINKY_TIP_IDX = 12  # pinky_intermediate

_FINGERTIP_INDICES = {
    "thumb": _THUMB_TIP_IDX,
    "index": _INDEX_TIP_IDX,
    "middle": _MIDDLE_TIP_IDX,
    "ring": _RING_TIP_IDX,
    "pinky": _PINKY_TIP_IDX,
}

# ManipTrans grace period: termination only fires after this many steps
_GRACE_PERIOD = 20


class FingertipTrackingExceeded(ManagerTermBase):
    """Terminate if any fingertip tracking error exceeds its threshold.

    Per-finger thresholds anneal linearly from a loose start value to a tight
    end value over training, driven by the curriculum's schedule_progress in
    [0,1] (0 = start/loose, 1 = end/final/tight):

        effective_threshold = start + (end - start) * schedule_progress

    A 20-step grace period is applied (the same grace period as ManipTrans).
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._traj_mgr = env.trajectory_manager

        # Cache body IDs for both hands (same pattern as rewards.py)
        self._rh_body_ids = torch.tensor(self._traj_mgr.right_body_ids_isaac, device=env.device)
        self._lh_body_ids = torch.tensor(self._traj_mgr.left_body_ids_isaac, device=env.device)

        # Fingertip indices into the FK body list. The legacy 13-body RH56F1
        # layout keeps the module constants; non-standard layouts (SharpA,
        # 23 bodies) use the trajectory manager's name-resolved indices.
        _fk = getattr(self._traj_mgr, "fk_finger_indices", None)
        self._tip_indices = dict(_fk["tips"]) if _fk else dict(_FINGERTIP_INDICES)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        thresholds_start: dict[str, float],
        thresholds_end: dict[str, float],
        side: str = "both",
    ) -> torch.Tensor:
        """Check if any fingertip exceeds its distance threshold.

        Args:
            env: The environment instance.
            thresholds_start: Per-finger start (loose) distance thresholds in
                meters. Keys: "thumb", "index", "middle", "ring", "pinky".
            thresholds_end: Per-finger end (final/tight) distance thresholds in
                meters. Same keys as thresholds_start.
            side: Which hand(s) to check ("right", "left", "both").

        Effective threshold per finger = start + (end - start) * schedule_progress.

        Returns:
            Boolean tensor (num_envs,) — True if terminated.
        """
        # Grace period: don't terminate before 20 steps
        grace_mask = env.episode_length_buf >= _GRACE_PERIOD

        ref_body = self._traj_mgr.get_reference_body_pos(env.episode_length_buf)
        origins = env.scene.env_origins
        progress = self._traj_mgr.schedule_progress

        rh = env.scene["right_hand"]
        lh = env.scene["left_hand"]

        terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        # Track per-finger distances for diagnostics
        finger_dists: dict[str, torch.Tensor] = {}
        finger_thresholds: dict[str, float] = {}

        for finger, body_idx in self._tip_indices.items():
            # Linear schedule: start (loose) -> end (tight) over training
            threshold = thresholds_start[finger] + (thresholds_end[finger] - thresholds_start[finger]) * progress
            finger_thresholds[finger] = threshold

            if side in ("right", "both"):
                rh_cur = rh.data.body_link_pos_w[:, self._rh_body_ids[body_idx], :] - origins
                rh_ref = ref_body["right_body_pos"][:, body_idx, :]
                rh_dist = torch.norm(rh_ref - rh_cur, dim=-1)
                terminated = terminated | (rh_dist > threshold)
                finger_dists[f"rh_{finger}"] = rh_dist

            if side in ("left", "both"):
                lh_cur = lh.data.body_link_pos_w[:, self._lh_body_ids[body_idx], :] - origins
                lh_ref = ref_body["left_body_pos"][:, body_idx, :]
                lh_dist = torch.norm(lh_ref - lh_cur, dim=-1)
                terminated = terminated | (lh_dist > threshold)
                finger_dists[f"lh_{finger}"] = lh_dist

        result = terminated & grace_mask

        return result


class ArticulatedObjectSanity(ManagerTermBase):
    """Terminate envs whose ARTICULATED object has blown up: root speed > ``lin_vel_max`` (m/s), any joint speed
    > ``joint_vel_max`` (rad/s, the joint velocity limit is 60), or a non-finite root/joint state.

    A reset whose hand pose overlaps the object kicks it (the same kick hits rigid objects); on an articulated object
    the free hinge then spins at the velocity limit and the state goes non-finite — non-finite bounds make the GPU
    broadphase pair every env with every other (aggregate-pair capacity overflow), which slows and corrupts the whole
    scene, not just the kicked env. Reference object speeds stay under ~2 m/s and reference hinge speeds under
    ~45 rad/s, so neither threshold cuts a motion the reference asks for.

    Spawn-hold window: a frame-0 finger inside the moving part kicks the hinge to the velocity limit in the first step,
    which ended every judge env of such a clip at step 1. While SPAWN_HOLD rewrites the root AND the joints to the
    reference every policy step (``pin_objects_to_reference(joints=True)``), the speed measured after such a step is the
    in-step depenetration kick, not a state the episode keeps — the speed rule is not counted for a step that began
    pinned (window owner ``spawn_hold.step_began_pinned`` / ``spawn_hold_steps`` — the same reader the pin uses). A
    non-finite state is counted always (it corrupts the scene whatever the window). A judge-
    parked env (``set_parked`` — owner rl.judge.park, same sink as the θ judge) is counted as before: PARK pins
    the root only, so the window rule does not apply to it. Hold 0 (unset) = the old rule, value for value.
    Marker ``[ARTIC][sanity-window]``.

    Absurd state (owner ``tasks.guards``): a root/joint speed
    above ``ABSURD_VEL`` or a root farther than ``ABSURD_DIST_M`` from the env origin is counted always, like a
    non-finite state — a finite 1e14 state inside the window passed both old rules and its NaN reward poisoned
    the update. The step's (non-finite, absurd) masks are left on the env for ``absurd_state.sanitize_step``
    (rule B). No absurd env = the old result, value for value."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        from dexengine.rl.tasks import guards as absurd_state
        from dexengine.rl.tasks.scene.spawn_hold import spawn_hold_steps

        self._hold = spawn_hold_steps()
        self._parked = None
        self._abs = absurd_state
        print(
            f"[ARTIC][sanity-window] {cfg.params.get('object_key', 'object_right')}: pinned window {self._hold} steps "
            f"— "
            f"steps started in that window skip the speed rule (non-finite states and judge-parked envs always count)",
            flush=True,
        )

    def set_parked(self, mask: torch.Tensor | None) -> None:
        """Judge park mask (num_envs,) bool | None; owner rl.judge.park (the same mask as the θ judge term)."""
        self._parked = mask

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_key: str = "object_right",
        lin_vel_max: float = 10.0,
        joint_vel_max: float = 55.0,
    ) -> torch.Tensor:
        from dexengine.rl.tasks.scene.spawn_hold import step_began_pinned

        obj = env.scene[object_key]
        rs = obj.data.root_state_w
        jp, jv = obj.data.joint_pos, obj.data.joint_vel
        bad = ~(torch.isfinite(rs).all(dim=-1) & torch.isfinite(jp).all(dim=-1) & torch.isfinite(jv).all(dim=-1))
        fast = (obj.data.root_lin_vel_w.norm(dim=-1) > lin_vel_max) | (jv.abs() > joint_vel_max).any(dim=-1)
        fast = torch.nan_to_num(fast.float(), nan=1.0).bool()
        ab = self._abs.absurd(obj, env.scene.env_origins)
        self._abs.note(env, object_key, bad, ab)
        if self._hold > 0:
            exempt = step_began_pinned(env.episode_length_buf, self._hold)
            if self._parked is not None:
                exempt = exempt & ~self._parked.to(device=exempt.device, dtype=torch.bool)
            fast = fast & ~exempt
        return bad | ab | fast


def fh_nan_reset(env):
    """Termination term of DEXENGINE_FH_NAN_RESET=1 (the env cfg attaches it only with that flag): ends, at this step,
    the envs whose reward was non-finite on the previous step (marked by absurd_state._fh_nan_mark); the mark is read
    and cleared."""
    pend = getattr(env, "_fh_nan_pending", None)
    if pend is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    out = pend.clone()
    pend.zero_()
    return out


def _fh_vel_ids(env, art, side: str):
    """Body and joint indices velocity_sanity looks at on path C (DEXENGINE_FH_WRIST=joint): without the virtual links,
    virtual joints and hand root (the same meaning as the default path's 'every body but the root, every joint'). The
    default path returns (None, None), the old formula unchanged (byte-identical)."""
    c = getattr(env, "_velsan_vj", None)
    if c is None:
        from dexengine.rl.tasks import hooks

        c = env._velsan_vj = {"on": hooks.floating_active() and hooks.floating_hand().joint_drive()}
    if not c["on"]:
        return None, None
    if side not in c:
        from dexengine.core.arms import active_hand_name
        from dexengine.rl.tasks import hooks

        _fh = hooks.floating_hand()
        vb, vjn = set(_fh.vj_link_names(side)), set(_fh.vj_joint_names(side))
        root = _fh.root_body(active_hand_name(), side)
        b = [i for i, n in enumerate(art.body_names) if n not in vb and n != root]
        j = [i for i, n in enumerate(art.joint_names) if n not in vjn]
        if len(art.body_names) - len(b) != len(vb) + 1 or len(art.joint_names) - len(j) != len(vjn):
            raise RuntimeError(
                f"[FH_WRIST] velocity_sanity {side}: the virtual link / joint subtraction does not add up "
                f"(bodies {len(art.body_names)}→{len(b)} · joints {len(art.joint_names)}→{len(j)})"
            )
        c[side] = (torch.tensor(b, device=env.device), torch.tensor(j, device=env.device))
    return c[side]


def velocity_sanity(
    env: ManagerBasedRLEnv,
    lin_vel_threshold: float = 100.0,
    ang_vel_threshold: float = 200.0,
    joints_vel_threshold: float = 100.0,
    dof_vel_threshold: float = 200.0,
    side: str = "both",
) -> torch.Tensor:
    """Terminate if velocity exceeds sanity thresholds.

    The same sanity thresholds as ManipTrans:
    - Wrist linear velocity > 100 m/s
    - Wrist angular velocity > 200 rad/s
    - Mean joint body velocity norm > 100 m/s
    - Mean absolute DOF velocity > 200 rad/s

    No grace period (explosions should be caught immediately).

    Args:
        env: The environment instance.
        lin_vel_threshold: Maximum wrist linear velocity magnitude (m/s).
        ang_vel_threshold: Maximum wrist angular velocity magnitude (rad/s).
        joints_vel_threshold: Maximum mean joint body velocity norm (m/s).
        dof_vel_threshold: Maximum mean absolute DOF velocity (rad/s).
        side: Which hand(s) to check ("right", "left", "both").

    Returns:
        Boolean tensor (num_envs,) — True if terminated.
    """
    rh = env.scene["right_hand"]
    lh = env.scene["left_hand"]

    return _palm_body_ids(
        env, lin_vel_threshold, ang_vel_threshold, joints_vel_threshold, dof_vel_threshold, side, lh, rh
    )


def _palm_body_ids(env, lin_vel_threshold, ang_vel_threshold, joints_vel_threshold, dof_vel_threshold, side, lh, rh):
    """Palm body id cache Why: in a per-term profile this one termination term took **28% of the term time** (14.42
    ms/step), with no reason for a blow-up sanity check to cost that. The cause was calling `resolve_palm_body_id`
    **every step**: it imports ARM_REGISTRY and scans the body_names string list with .index(). The body list is
    **fixed** after the scene build, so resolving once is enough. It also resolved both sides although the cfg says
    side="right", a full waste on a right-only run. **The result does not change**: the same termination comes out
    faster.
    """
    _c = getattr(env, "_velsan_palm_ids", None)
    if _c is None:
        _c = env._velsan_palm_ids = {}
    if side in ("right", "both"):
        if "right" not in _c:
            _c["right"] = resolve_palm_body_id(rh, "right")
        rh_palm_id = _c["right"]
    if side in ("left", "both"):
        if "left" not in _c:
            _c["left"] = resolve_palm_body_id(lh, "left")
        lh_palm_id = _c["left"]

    terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Track per-check values for diagnostics
    _diag: dict[str, torch.Tensor] = {}
    _diag_thresh: dict[str, float] = {}

    if side in ("right", "both"):
        # Wrist velocity
        rh_lin_vel = torch.norm(rh.data.body_link_lin_vel_w[:, rh_palm_id, :], dim=-1)
        rh_ang_vel = torch.norm(rh.data.body_link_ang_vel_w[:, rh_palm_id, :], dim=-1)
        terminated = terminated | (rh_lin_vel > lin_vel_threshold) | (rh_ang_vel > ang_vel_threshold)
        _diag["rh_lin_vel"] = rh_lin_vel
        _diag_thresh["rh_lin_vel"] = lin_vel_threshold
        _diag["rh_ang_vel"] = rh_ang_vel
        _diag_thresh["rh_ang_vel"] = ang_vel_threshold

        terminated = _velocity_exceeded(
            env, joints_vel_threshold, dof_vel_threshold, _diag, _diag_thresh, rh, terminated
        )

    if side in ("left", "both"):
        lh_lin_vel = torch.norm(lh.data.body_link_lin_vel_w[:, lh_palm_id, :], dim=-1)
        lh_ang_vel = torch.norm(lh.data.body_link_ang_vel_w[:, lh_palm_id, :], dim=-1)
        terminated = terminated | (lh_lin_vel > lin_vel_threshold) | (lh_ang_vel > ang_vel_threshold)
        _diag["lh_lin_vel"] = lh_lin_vel
        _diag_thresh["lh_lin_vel"] = lin_vel_threshold
        _diag["lh_ang_vel"] = lh_ang_vel
        _diag_thresh["lh_ang_vel"] = ang_vel_threshold

        _vb, _vd = _fh_vel_ids(env, lh, "left")
        lh_body_vel_norms = torch.norm(
            lh.data.body_link_lin_vel_w[:, 1:, :] if _vb is None else lh.data.body_link_lin_vel_w[:, _vb, :], dim=-1
        )
        lh_mean_body_vel = lh_body_vel_norms.mean(dim=-1)
        terminated = terminated | (lh_mean_body_vel > joints_vel_threshold)
        _diag["lh_joints_vel"] = lh_mean_body_vel
        _diag_thresh["lh_joints_vel"] = joints_vel_threshold

        lh_dof_vel_mean = (lh.data.joint_vel if _vd is None else lh.data.joint_vel[:, _vd]).abs().mean(dim=-1)
        terminated = terminated | (lh_dof_vel_mean > dof_vel_threshold)
        _diag["lh_dof_vel"] = lh_dof_vel_mean
        _diag_thresh["lh_dof_vel"] = dof_vel_threshold

    return _log_velocity_failures(env, _diag, _diag_thresh, terminated)


def _velocity_exceeded(env, joints_vel_threshold, dof_vel_threshold, _diag, _diag_thresh, rh, terminated):
    """Joint body velocities: the mean linear speed over the bodies, against the threshold (100 m/s by default)."""
    _vb, _vd = _fh_vel_ids(
        env, rh, "right"
    )  # indices on path C only; the default path is (None, None), the old formula
    rh_body_vel_norms = torch.norm(
        rh.data.body_link_lin_vel_w[:, 1:, :] if _vb is None else rh.data.body_link_lin_vel_w[:, _vb, :], dim=-1
    )  # skip root
    rh_mean_body_vel = rh_body_vel_norms.mean(dim=-1)
    terminated = terminated | (rh_mean_body_vel > joints_vel_threshold)
    _diag["rh_joints_vel"] = rh_mean_body_vel
    _diag_thresh["rh_joints_vel"] = joints_vel_threshold

    # DOF velocities: the mean absolute joint speed, against the threshold (200 rad/s by default)
    rh_dof_vel_mean = (rh.data.joint_vel if _vd is None else rh.data.joint_vel[:, _vd]).abs().mean(dim=-1)
    terminated = terminated | (rh_dof_vel_mean > dof_vel_threshold)
    _diag["rh_dof_vel"] = rh_dof_vel_mean
    _diag_thresh["rh_dof_vel"] = dof_vel_threshold
    return terminated


def _log_velocity_failures(env, _diag, _diag_thresh, terminated):
    """Diagnostics: log which velocity check(s) failed ⚠ The condition order decides the cost. The old version was if
    terminated.any() and env.num_envs <= 64: and Python **evaluates the left side first**, so `terminated.any()`
    forced a blocking GPU->CPU sync every step, while the right side is **always false** at NE=4096 and the sync was
    100% useless. Swapping the order removes it by short-circuiting. At training scale (NE>64) the diagnostics never
    printed, so **the behaviour is the same**.
    """
    if env.num_envs <= 64 and terminated.any():
        term_ids = terminated.nonzero(as_tuple=False).view(-1)
        for eid in term_ids:
            ei = eid.item()
            frame = env.episode_length_buf[ei].item()
            violations = []
            for check_name, val_tensor in _diag.items():
                v = val_tensor[ei].item()
                t = _diag_thresh[check_name]
                if v > t:
                    violations.append(f"{check_name}={v:.1f}>{t:.1f}")
            if violations:
                print(f"[VelocitySanity] env={ei} frame={frame} {' '.join(violations)}")

    return terminated
