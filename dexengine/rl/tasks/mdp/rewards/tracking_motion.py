"""Reference-tracking rewards on velocities, finger pose, power and action rate."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
from dexengine.rl.tasks.mdp.naming import resolve_wrist_joints
from dexengine.rl.tasks.mdp.rewards.tracking_base import _PALM_IDX, _BimanualRewardBase


class RewardEefVel(_BimanualRewardBase):
    """Wrist linear velocity tracking: exp(-k * ||ref_vel - cur_vel||).

    Reference velocity comes from pre-computed MuJoCo FK body velocities
    (palm body, index 0).
    """

    def __call__(self, env: ManagerBasedEnv, k: float = 1.0, side: str = "both") -> torch.Tensor:
        ref_body = self._traj_mgr.get_reference_body_pos(env.episode_length_buf)

        rh = env.scene["right_hand"]
        lh = env.scene["left_hand"]

        # Right hand
        rh_cur = rh.data.body_link_lin_vel_w[:, self._rh_palm_body_id, :]
        rh_ref = ref_body["right_body_vel"][:, _PALM_IDX, :]
        rh_dist = torch.norm(rh_ref - rh_cur, dim=-1)

        # Left hand
        lh_cur = lh.data.body_link_lin_vel_w[:, self._lh_palm_body_id, :]
        lh_ref = ref_body["left_body_vel"][:, _PALM_IDX, :]
        lh_dist = torch.norm(lh_ref - lh_cur, dim=-1)

        return self._combine(torch.exp(-k * rh_dist), torch.exp(-k * lh_dist))


class RewardJointsVel(_BimanualRewardBase):
    """Joint velocity tracking: exp(-k * mean(|ref_dq - cur_dq|)).

    Reference velocity is computed on-the-fly via finite difference:
    ``(qpos[t+1] - qpos[t]) * fps``.

    Both reference qpos and current joint_vel are in PhysX joint order.
    The 12 independent joint IDs index into that order.
    """

    def __call__(self, env: ManagerBasedEnv, k: float = 1.0, side: str = "both") -> torch.Tensor:
        frame_ids = env.episode_length_buf

        # Reference state at t and t+1
        ref_t = self._traj_mgr.get_reference_state(frame_ids)
        ref_t1 = self._traj_mgr.get_reference_state(frame_ids + 1)

        rh = env.scene["right_hand"]
        lh = env.scene["left_hand"]

        # Right hand: independent joint velocities
        rh_ref_dq = (
            ref_t1["right_hand_qpos"][:, self._rh_traj_indep_ids] - ref_t["right_hand_qpos"][:, self._rh_traj_indep_ids]
        ) * self._fps
        rh_cur_dq = rh.data.joint_vel[:, self._rh_indep_ids]
        # Truncate to match dims (ref may have 12/18, current may have 6/12)
        min_dim = min(rh_ref_dq.shape[-1], rh_cur_dq.shape[-1])
        rh_err = torch.abs(rh_ref_dq[:, :min_dim] - rh_cur_dq[:, :min_dim]).mean(dim=-1)

        # Left hand
        lh_ref_dq = (
            ref_t1["left_hand_qpos"][:, self._lh_traj_indep_ids] - ref_t["left_hand_qpos"][:, self._lh_traj_indep_ids]
        ) * self._fps
        lh_cur_dq = lh.data.joint_vel[:, self._lh_indep_ids]
        min_dim_l = min(lh_ref_dq.shape[-1], lh_cur_dq.shape[-1])
        lh_err = torch.abs(lh_ref_dq[:, :min_dim_l] - lh_cur_dq[:, :min_dim_l]).mean(dim=-1)

        return self._combine(torch.exp(-k * rh_err), torch.exp(-k * lh_err))


class RewardFingerPose(_BimanualRewardBase):
    """Finger joint-angle tracking: exp(-k * mean(|q_ref - q_cur|)) over the independent
    finger joints, vs the retarget reference pose.

    The fingertip-POSITION terms leave a 1-2 DOF null space per finger on high-DoF hands
    (SharpA: 4-5 DOF/finger), which the policy fills with contorted/over-bent configurations
    that hit the tip position with the wrong pose. This term pins the finger pose to the
    faithful retarget. Both reference qpos and current joint_pos are in PhysX order; the
    independent joint IDs index into that order (same convention as RewardJointsVel).
    """

    def __call__(self, env: ManagerBasedEnv, k: float = 4.0, side: str = "both") -> torch.Tensor:
        ref = self._traj_mgr.get_reference_state(env.episode_length_buf)
        rh = env.scene["right_hand"]
        lh = env.scene["left_hand"]

        def _pose(asset, indep_ids, traj_ids, ref_key):
            if indep_ids is None or len(indep_ids) == 0:
                return torch.ones(env.num_envs, device=env.device)  # no indep joints -> inert (1.0)
            ref_q = ref[ref_key][:, traj_ids]
            cur_q = asset.data.joint_pos[:, indep_ids]
            m = min(ref_q.shape[-1], cur_q.shape[-1])
            err = torch.abs(ref_q[:, :m] - cur_q[:, :m]).mean(dim=-1)
            return torch.exp(-k * err)

        rh_r = _pose(rh, self._rh_indep_ids, self._rh_traj_indep_ids, "right_hand_qpos")
        lh_r = _pose(lh, self._lh_indep_ids, self._lh_traj_indep_ids, "left_hand_qpos")
        return self._combine(rh_r, lh_r)


class RewardPower(_BimanualRewardBase):
    """Total power penalty: exp(-k * sum(|vel * torque|)).

    Penalizes excessive actuation effort across all joints.
    """

    def __call__(self, env: ManagerBasedEnv, k: float = 10.0, side: str = "both") -> torch.Tensor:
        rh = env.scene["right_hand"]
        lh = env.scene["left_hand"]

        rh_power = torch.sum(torch.abs(rh.data.joint_vel * rh.data.applied_torque), dim=-1)
        lh_power = torch.sum(torch.abs(lh.data.joint_vel * lh.data.applied_torque), dim=-1)

        return self._combine(torch.exp(-k * rh_power), torch.exp(-k * lh_power))


class RewardWristPower(_BimanualRewardBase):
    """Wrist power penalty: exp(-k * sum(|wrist_vel * wrist_torque|)).

    Only includes 6 wrist joints (pos_x/y/z + rot_x/y/z) per hand.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        rh = env.scene["right_hand"]
        lh = env.scene["left_hand"]
        self._rh_wrist_ids = self._find_joint_ids(rh, resolve_wrist_joints(rh, "right"))
        self._lh_wrist_ids = self._find_joint_ids(lh, resolve_wrist_joints(lh, "left"))

    def __call__(self, env: ManagerBasedEnv, k: float = 2.0, side: str = "both") -> torch.Tensor:
        rh = env.scene["right_hand"]
        lh = env.scene["left_hand"]
        N = env.num_envs

        if self._rh_wrist_ids is not None:
            rh_power = torch.sum(
                torch.abs(rh.data.joint_vel[:, self._rh_wrist_ids] * rh.data.applied_torque[:, self._rh_wrist_ids]),
                dim=-1,
            )
        else:
            rh_power = torch.zeros(N, device=env.device)

        if self._lh_wrist_ids is not None:
            lh_power = torch.sum(
                torch.abs(lh.data.joint_vel[:, self._lh_wrist_ids] * lh.data.applied_torque[:, self._lh_wrist_ids]),
                dim=-1,
            )
        else:
            lh_power = torch.zeros(N, device=env.device)

        return self._combine(torch.exp(-k * rh_power), torch.exp(-k * lh_power))


class RewardActionRate(_BimanualRewardBase):
    """Action rate penalty: exp(-k * ||action_t - action_{t-1}||).

    Penalizes the L1 norm of action changes between consecutive timesteps.
    This encourages smooth policy outputs without adding physical lag
    (unlike EMA filters or increased damping).

    Applies to all 12 action dims (6 wrist force/torque + 6 finger residuals).
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Store previous actions per hand
        self._prev_rh_actions = None
        self._prev_lh_actions = None

    def __call__(self, env: ManagerBasedEnv, k: float = 1.0, side: str = "both") -> torch.Tensor:
        if self._side in ("right", "both"):
            rh_term = env.action_manager._terms["right_hand_joint_pos"]
            rh_actions = rh_term.raw_actions  # (N, 12)
            if self._prev_rh_actions is None:
                self._prev_rh_actions = rh_actions.clone()
            rh_delta = torch.sum(torch.abs(rh_actions - self._prev_rh_actions), dim=-1)
            self._prev_rh_actions = rh_actions.clone()
            rh_reward = torch.exp(-k * rh_delta)

        if self._side in ("left", "both"):
            lh_term = env.action_manager._terms["left_hand_joint_pos"]
            lh_actions = lh_term.raw_actions  # (N, 12)
            if self._prev_lh_actions is None:
                self._prev_lh_actions = lh_actions.clone()
            lh_delta = torch.sum(torch.abs(lh_actions - self._prev_lh_actions), dim=-1)
            self._prev_lh_actions = lh_actions.clone()
            lh_reward = torch.exp(-k * lh_delta)

        if self._side == "right":
            return rh_reward
        elif self._side == "left":
            return lh_reward
        else:
            return self._combine(rh_reward, lh_reward)
