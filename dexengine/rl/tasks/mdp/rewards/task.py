"""Object manipulation reward terms.

Extends the hand-tracking rewards with object tracking and contact rewards.
All new terms use the same exp(-k * error) kernel as the hand-tracking set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
from dexengine.rl.tasks.mdp.rewards.task_artic import ObjArtiThetaJudge, _arti_theta_ref, _require_joint  # noqa: F401
from dexengine.rl.tasks.mdp.rewards.task_base import _TaskRewardBase


def residual_norm_penalty(
    env, deadzone: float = 0.5, side: str = "right", dims: tuple | None = None, per_dim_deadzone: float | None = None
) -> torch.Tensor:
    """Soft L2 penalty (with a dead-zone) on the residual action magnitude.

    penalty = -relu(||residual|| - deadzone)**2     (a NEGATIVE reward)

    The residual is the raw policy action for ONE hand: the per-hand residual
    action term's ``raw_actions`` (12-d for the RH56F1 arm-residual term).
    Reading that hand's own slice — NOT the concatenated multi-hand
    ``action_manager.action`` — keeps each hand's validated scale-to-l2 stability
    point intact under Bimanual Option 2 (one residual_l2 term per hand). For the
    single-hand right-only path the term's raw_actions IS the full action, so the
    behavior is byte-identical to the legacy ``action_manager.action`` read.
    Inside the dead-zone (||residual|| <= deadzone) there is NO penalty, so the
    policy may freely deviate from the reference when it genuinely helps
    manipulation. Beyond it the cost grows quadratically, *softly* discouraging
    the unbounded actor-gain drift that drives policy collapse (residual ‖mu‖
    inflates until the IK target / finger targets leave the reference tube). This
    is NOT a hard clip — it adds a restoring gradient toward the base, not a wall.

    Exp-1 stability term. Wired with weight 0.0 by default; set via
    --residual_l2_weight (coefficient) and --residual_l2_deadzone.

    ``dims``: optional (lo, hi) slice of the 12-d residual to penalize. None
    (default) = the full vector (byte-identical legacy behavior). (0, 6) =
    wrist dims, (6, 12) = finger dims — used to anchor the wrist and the
    fingers with INDEPENDENT weights/deadzones (the wrist residual is more
    divergence-prone, so it can carry a stiffer anchor than the fingers).

    ``per_dim_deadzone``: PER-DIM hinge mode (wrist-hinge). When
    set, the norm-deadzone form above is REPLACED by

        penalty = -sum_i relu(|r_i| - per_dim_deadzone)**2

    i.e. each dim is tax-free up to ``per_dim_deadzone`` of its own range and
    quadratically taxed beyond — with the hard-cap action mapping
    (clamp(raw,±1)·cap) a value of 0.7 frees 70% of each dim's cap while
    keeping full anti-divergence pressure near saturation (|r_i| -> 1).
    At per_dim_deadzone=0 this equals the norm form with deadzone=0
    (-sum r_i^2), so a 0 -> target ramp starts as a strict full-L2 guide.
    None (default) = norm mode, byte-identical.
    """
    term_name = f"{side}_hand_joint_pos"
    action_term = env.action_manager._terms.get(term_name)
    if action_term is not None:
        raw = getattr(action_term, "raw_actions", None)
        if raw is None:
            raw = getattr(action_term, "_raw_actions", None)
        action = raw if raw is not None else env.action_manager.action
    else:
        # Fallback (no per-hand term resolved): legacy full-action read.
        action = env.action_manager.action  # (N, A) raw residual action
    if action_term is not None and hasattr(
        action_term, "legacy_dims"
    ):  # DEXENGINE_ARM_JOINT_RESIDUAL layout (identity off)
        dims = action_term.legacy_dims(dims)
    if dims is not None:
        action = action[:, dims[0] : dims[1]]
    if per_dim_deadzone is not None:
        excess_pd = (action.abs() - float(per_dim_deadzone)).clamp(min=0.0)
        return -(excess_pd * excess_pd).sum(dim=-1)
    norm = torch.norm(action, dim=-1)
    excess = (norm - deadzone).clamp(min=0.0)
    return -(excess * excess)


def residual_action_rate(env, side: str = "right", dims: tuple = (0, 6)):
    """Anti-trembling action-RATE penalty: -sum_i (a_i(t) - a_i(t-1))^2.

    Penalizes the STEP-TO-STEP CHANGE of the raw residual action (not its
    amplitude — that's residual_l2/the wrist hinge, which have a zero-gradient
    free zone where high-frequency jitter lives untaxed). Default dims (0, 6)
    = wrist only, so finger closure speed is untouched (grasp-quality guard).
    Reads the flat ActionManager action/prev_action buffers at this side's
    term offset; prev_action resets to 0 on episode reset (same first-step
    behavior as Isaac Lab's stock action_rate_l2). Wired at weight 0 by
    default via --wrist_action_rate_weight.
    """
    am = env.action_manager
    t_side = am._terms.get(f"{side}_hand_joint_pos")
    if t_side is not None and hasattr(t_side, "legacy_dims"):  # DEXENGINE_ARM_JOINT_RESIDUAL layout (identity off)
        dims = t_side.legacy_dims(dims)
    if t_side is not None and getattr(
        t_side, "action_abs", False
    ):  # DEXENGINE_ACTION_ABS: rate of the implied residual
        d = t_side.implied_cur[:, dims[0] : dims[1]] - t_side.implied_prev[:, dims[0] : dims[1]]
        return -(d * d).sum(dim=-1)
    off = 0
    for name, term in am._terms.items():
        if name == f"{side}_hand_joint_pos":
            break
        off += term.action_dim
    a = am.action[:, off + dims[0] : off + dims[1]]
    p = am.prev_action[:, off + dims[0] : off + dims[1]]
    d = a - p
    return -(d * d).sum(dim=-1)


class RewardObjPos(_TaskRewardBase):
    """Object position tracking: exp(-k * ||ref_pos - cur_pos||).

    The dominant task reward (k=80; the weight comes from the reward cfg and the recipe).
    Ensures the policy prioritizes keeping the object on trajectory.
    """

    def __call__(
        self, env: ManagerBasedEnv, k: float = 80.0, side: str = "right", object_key: str = "object_right"
    ) -> torch.Tensor:
        obj = env.scene[self._object_key]
        origins = env.scene.env_origins
        obj_pos = obj.data.root_pos_w - origins

        ref_state = self._traj_mgr.get_reference_state(env.episode_length_buf)
        ref_pos = ref_state[f"{self._side}_obj_pos"]

        dist = torch.norm(ref_pos - obj_pos, dim=-1)
        return torch.exp(-k * dist)


class RewardObjRot(_TaskRewardBase):
    """Object rotation tracking: exp(-k * |angle_error|).

    Uses quaternion angle difference between reference and current orientation.
    """

    def __call__(
        self, env: ManagerBasedEnv, k: float = 3.0, side: str = "right", object_key: str = "object_right"
    ) -> torch.Tensor:
        from isaaclab.utils.math import quat_conjugate, quat_mul

        obj = env.scene[self._object_key]
        obj_quat = obj.data.root_quat_w  # (N, 4) wxyz

        ref_state = self._traj_mgr.get_reference_state(env.episode_length_buf)
        ref_quat = ref_state[f"{self._side}_obj_quat"]  # (N, 4) wxyz

        # Quaternion error -> angle
        quat_err = quat_mul(quat_conjugate(obj_quat), ref_quat)
        # Angle from quaternion: 2 * acos(|w|)
        angle = 2.0 * torch.acos(torch.clamp(torch.abs(quat_err[:, 0]), max=1.0))

        return torch.exp(-k * angle)


class RewardObjVel(_TaskRewardBase):
    """Object velocity tracking: exp(-k * mean(|ref_vel - cur_vel|))."""

    def __call__(
        self, env: ManagerBasedEnv, k: float = 1.0, side: str = "right", object_key: str = "object_right"
    ) -> torch.Tensor:
        obj = env.scene[self._object_key]
        obj_vel = obj.data.root_lin_vel_w

        # Reference velocity via finite differences
        ref_state_t = self._traj_mgr.get_reference_state(env.episode_length_buf)
        ref_state_t1 = self._traj_mgr.get_reference_state(env.episode_length_buf + 1)
        ref_vel = (ref_state_t1[f"{self._side}_obj_pos"] - ref_state_t[f"{self._side}_obj_pos"]) * 50.0

        err = torch.abs(ref_vel - obj_vel).mean(dim=-1)
        return torch.exp(-k * err)


class RewardObjAngVel(_TaskRewardBase):
    """Object angular velocity tracking: exp(-k * mean(|ref_angvel - cur_angvel|)).

    ManipTrans-matched: tracks reference angular velocity.
    Reference computed via finite difference on object quaternion trajectory:
      ang_vel ≈ 2 * (q[t+1] * conj(q[t])).xyz * fps
    """

    def __call__(
        self, env: ManagerBasedEnv, k: float = 1.0, side: str = "right", object_key: str = "object_right"
    ) -> torch.Tensor:
        from isaaclab.utils.math import quat_conjugate, quat_mul

        obj = env.scene[self._object_key]
        obj_ang_vel = obj.data.root_ang_vel_w  # (N, 3)

        # Compute reference angular velocity from object quaternion trajectory
        frame_ids = env.episode_length_buf
        ref_t = self._traj_mgr.get_reference_state(frame_ids)
        ref_t1 = self._traj_mgr.get_reference_state(frame_ids + 1)
        q_t = ref_t[f"{self._side}_obj_quat"]  # (N, 4) wxyz
        q_t1 = ref_t1[f"{self._side}_obj_quat"]  # (N, 4) wxyz
        fps = getattr(self._traj_mgr, "_fps", 50.0)

        # Angular velocity from quaternion finite difference:
        # dq = q_t1 * conj(q_t), ang_vel ≈ 2 * dq.xyz * fps (small angle approx)
        dq = quat_mul(q_t1, quat_conjugate(q_t))  # (N, 4) wxyz
        # Ensure shortest path (w >= 0)
        sign = torch.sign(dq[:, 0:1])
        sign[sign == 0] = 1.0
        dq = dq * sign
        ref_ang_vel = 2.0 * dq[:, 1:4] * fps  # (N, 3)

        err = torch.abs(ref_ang_vel - obj_ang_vel).mean(dim=-1)
        return torch.exp(-k * err)


class RewardObjArtic(_TaskRewardBase):
    """Articulated **joint-value tracking** (rule D1-c): exp(-k · err), err = the joint-kind unit error
    (rl.judge.artic.err_units: hinge rad, slide as a fraction of travel; travel = the spawned joint's limit width,
    artic_parts.joint_kind_span).

    For an articulated object (ArticulationCfg, base_link + moving_link + 1 joint) "opening/closing is the task" (rule
    D1-c), so on top of base pose tracking (RewardObjPos/Rot) the joint value follows the reference's joint column
    (`ref_state[f"{side}_obj_arti"]`, owned by the trajectory manager). k and weight are owned by
    env_setup.object_rewards.add_obj_artic_terms alone (built only for articulated clips; rigid clips have no term, so
    they stay byte-identical). The hinge env error is bit-identical with the error before units were introduced.
    This term on a non-articulated object is a defect and fails loudly at construction. Marker `[ARTIC][reward]`.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        from dexengine.rl.judge.artic import err_units
        from dexengine.rl.tasks.mdp.artic_parts import joint_kind_span

        self._err_units = err_units  # unit owner (the same function as the θ judge)
        n_j = _require_joint(env, self._object_key, "reward")
        self._is_slide, self._span = joint_kind_span(env, self._object_key)
        self._n_calls = 0
        self._log_every = 800  # steps (32 rollouts = 25 iterations)
        self._fk_checked = False
        print(
            f"[ARTIC][reward] {self._side}: RewardObjArtic built — object_key={self._object_key} num_joints={n_j} "
            f"weight={float(cfg.weight):.3f} k={float(cfg.params['k']):.3f}/kind unit · slide env "
            f"{int(self._is_slide.sum())}/{self._is_slide.numel()} · reference key {self._side}_obj_arti · "
            f"log period {self._log_every} steps",
            flush=True,
        )

    def _fk_check(self, obj) -> None:
        """Sign oracle (once): print the simulated base_link->moving_link relative rotation (axis, angle) next to the
        joint value. USD expresses the joint as a local rotation with the axis turned to Z, while the merged surface
        points and the judge use the MuJoCo convention R(axis, θ), so this checks that the signs agree: they do when
        the relative rotation axis points along the tree's `articulation.axis` (arctic: (0,0,−1)) and the angle is θ."""
        from isaaclab.utils.math import quat_conjugate, quat_mul

        names = list(getattr(obj, "body_names", []) or [])
        if "base_link" not in names or "moving_link" not in names:
            print(
                f"[ARTIC][fk-check] {self._side}: body_names={names} — no base_link/moving_link, oracle skipped",
                flush=True,
            )
            return
        ib, im = names.index("base_link"), names.index("moving_link")
        qb = obj.data.body_link_quat_w[:, ib]  # (N,4) wxyz
        qm = obj.data.body_link_quat_w[:, im]
        pb = obj.data.body_link_pos_w[:, ib]
        pm = obj.data.body_link_pos_w[:, im]
        q_rel = quat_mul(quat_conjugate(qb), qm)  # moving rotation in the base frame
        sgn = torch.sign(q_rel[:, :1])
        sgn[sgn == 0] = 1.0
        q_rel = q_rel * sgn
        ang = 2.0 * torch.acos(torch.clamp(q_rel[:, 0], max=1.0))
        ax = q_rel[:, 1:4] / torch.clamp(q_rel[:, 1:4].norm(dim=-1, keepdim=True), min=1e-9)
        th = obj.data.joint_pos[:, 0]
        d_rel = pm - pb
        i = int(torch.argmax(th.abs()))
        print(
            f"[ARTIC][fk-check] {self._side}: env{i} joint_pos={th[i].item():+.4f} rad · "
            f"relative rotation angle {ang[i].item():+.4f} rad axis {[round(v, 3) for v in ax[i].tolist()]} · "
            f"relative position {[round(v, 4) for v in d_rel[i].tolist()]} m "
            f"(axis along the tree articulation.axis + angle≈|θ| means it agrees with the R(axis,θ) convention)",
            flush=True,
        )

    def __call__(
        self, env: ManagerBasedEnv, k: float, side: str = "right", object_key: str = "object_right"
    ) -> torch.Tensor:
        obj = env.scene[self._object_key]
        theta, ref_theta = _arti_theta_ref(env, self._traj_mgr, obj, self._side)
        err = self._err_units(theta, ref_theta, self._is_slide, self._span)  # (N,) kind units
        if not hasattr(env, "_obj_arti_err"):
            env._obj_arti_err = {}
        env._obj_arti_err[self._side] = err.detach()
        if not self._fk_checked:
            self._fk_checked = True
            self._fk_check(obj)
        if self._n_calls % max(self._log_every, 1) == 0:
            # Physics sanity (diagnosis of a PhysX aggregate-pair blow-up): joint speed, root speed, link distance.
            _jv = obj.data.joint_vel[:, 0].abs()
            _rv = obj.data.root_lin_vel_w.norm(dim=-1)
            _names = list(getattr(obj, "body_names", []) or [])
            _dl = float("nan")
            if "base_link" in _names and "moving_link" in _names:
                _pb = obj.data.body_link_pos_w[:, _names.index("base_link")]
                _pm = obj.data.body_link_pos_w[:, _names.index("moving_link")]
                _dl = (_pm - _pb).norm(dim=-1).max().item()
            # Whether kicks (|v_root|>5 m/s) come from envs started at frame 0 or from RSI (mid-frame) starts: separates
            # causes.
            _sf = getattr(self._traj_mgr, "start_frames", None)
            _kick = _rv > 5.0
            _kick_txt = ""
            if _sf is not None:
                _f0 = _sf == 0
                _iv, _iw = int(torch.argmax(_rv)), int(torch.argmax(_jv))
                _kick_txt = (
                    f" · kicks f0-start {int((_kick & _f0).sum())}/{int(_f0.sum())} RSI-start "
                    f"{int((_kick & ~_f0).sum())}/{int((~_f0).sum())} · max|v| env{_iv}"
                    f"(start {int(_sf[_iv])}, ep {int(env.episode_length_buf[_iv])}) max|ω| env{_iw}"
                    f"(start {int(_sf[_iw])}, ep {int(env.episode_length_buf[_iw])}, "
                    f"θ {theta[_iw].item():.3f})"
                )
            print(
                f"[ARTIC][reward] {self._side}: step {self._n_calls} error (kind units) mean {err.mean().item():.4f} "
                f"p90 {torch.quantile(err.float(), 0.9).item():.4f} · ref mean {ref_theta.mean().item():.4f} · "
                f"sim mean {theta.mean().item():.4f} · reward mean {torch.exp(-k * err).mean().item():.4f} · "
                f"θ[{theta.min().item():.3f},{theta.max().item():.3f}] |ω|max {_jv.max().item():.2f} "
                f"|v_root|max {_rv.max().item():.2f} (>5 m/s: {int(_kick.sum())}) |base−moving|max {_dl:.4f} m"
                + _kick_txt,
                flush=True,
            )
        self._n_calls += 1
        return torch.exp(-k * err)
