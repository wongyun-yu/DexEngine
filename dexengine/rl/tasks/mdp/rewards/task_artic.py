"""Articulated-object joint-angle reward and the joint-angle judge term."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _arti_theta_ref(env, traj_mgr, obj, side: str):
    """(current θ (N,), reference θ (N,)) of articulated side `side`, read by both the reward and the θ judge
    termination (one-joint convention; reference column owner: the trajectory manager's `{side}_obj_arti`)."""
    theta = obj.data.joint_pos[:, 0]
    ref_state = traj_mgr.get_reference_state(env.episode_length_buf)
    key = f"{side}_obj_arti"
    if key not in ref_state:
        raise KeyError(
            f"[ARTIC] {side}: the reference state has no '{key}' — a tree/loader without the joint columns (rule D1)"
        )
    return theta, ref_state[key].reshape(theta.shape[0]).to(theta.device)


def _require_joint(env, object_key: str, tag: str) -> int:
    """An articulated term on a rigid object is a defect: fail at construction (an add_obj_artic_terms gate defect)."""
    n_j = int(getattr(env.scene[object_key], "num_joints", 0) or 0)
    if n_j < 1:
        raise ValueError(
            f"[ARTIC][{tag}] object_key={object_key!r} has no joints (num_joints={n_j}) — "
            "a term meant for articulated clips only was attached to a rigid object (an add_obj_artic_terms gate "
            "defect)"
        )
    return n_j


class ObjArtiThetaJudge(ManagerTermBase):
    """Articulated θ judge termination: a training termination term and the fresh-process judge's termination term (same
    env, same term). The predicate and threshold are owned by rl.judge.artic (failure when the kind-unit error stays out
    of the threshold for hold consecutive steps), the term name by rl.judge.terms, and the registration by
    env_setup.object_rewards.add_obj_artic_terms (per articulated side, time_out=False, so an early judge termination
    is
    a failure). The run counter returns to 0 in reset envs. Judge-parked (PARK) envs do not count: the park mask owner
    rl.judge.park passes it with `set_parked` (None = no park, training). Marker `[ARTIC][theta-judge]`."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        from dexengine.rl.judge import artic as artic_judge
        from dexengine.rl.tasks.mdp.artic_parts import joint_kind_span

        self._side = cfg.params["side"]
        self._object_key = cfg.params["object_key"]
        self._traj_mgr = env.trajectory_manager
        _require_joint(env, self._object_key, "theta-judge")
        self._is_slide, self._span = joint_kind_span(env, self._object_key)
        self._counter = artic_judge.ThetaOutCounter(env.num_envs, env.device, float(env.step_dt))
        self._err_units = artic_judge.err_units
        self._parked = None
        print(
            f"[ARTIC][theta-judge] {self._side}: {self._object_key} · slide env {int(self._is_slide.sum())}/"
            f"{self._is_slide.numel()} · hold {self._counter.hold} steps (step_dt {float(env.step_dt):g} s) · "
            f"{artic_judge.describe()}",
            flush=True,
        )

    def reset(self, env_ids=None) -> None:
        self._counter.reset(env_ids)

    def set_parked(self, mask: torch.Tensor | None) -> None:
        """Judge park mask (num_envs,) bool | None; owner rl.judge.park (the same mask as the action term)."""
        self._parked = None if mask is None else mask.to(device=self._is_slide.device, dtype=torch.bool)

    def __call__(self, env: ManagerBasedEnv, side: str, object_key: str) -> torch.Tensor:
        theta, ref = _arti_theta_ref(env, self._traj_mgr, env.scene[self._object_key], self._side)
        return self._counter.update(
            self._err_units(theta, ref, self._is_slide, self._span), self._is_slide, self._parked
        )
