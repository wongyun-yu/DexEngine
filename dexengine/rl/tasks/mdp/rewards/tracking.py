# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ManipTrans-style TRACKING (imitation) reward terms for hand imitation.

NAMING NOTE: this module was historically the "Stage 1" reward set (used to train
a standalone Stage-1 imitation policy). The current canonical recipe is
SINGLE-STAGE (reference-base + residual): these tracking terms run in the SAME
single training stage as the object/contact terms (rewards/task.py,
dexengine/rl/tasks/mdp/rewards/contact.py) — there is no separate imitation pretraining stage.

14 reward terms using exponential kernel: r = exp(-k * distance).
Each term returns (num_envs,) tensor. Bimanual rewards average right + left hands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


from dexengine.rl.tasks.mdp.rewards.tracking_base import (
    _PALM_IDX,
    _BimanualRewardBase,
)

# ---------------------------------------------------------------------------
# 1. Wrist position tracking
# ---------------------------------------------------------------------------


class RewardEefPos(_BimanualRewardBase):
    """Wrist position tracking reward: exp(-k * ||ref_pos - cur_pos||).

    Supports soft hinge via `hinge_threshold_m` (Phase A2 wrist hinge, default 0.0 = disabled).
    When > 0, errors within the threshold give full reward (1.0); exponential decay applies
    only to the excess beyond the threshold. Mirrors the fingertip hinge mechanism.
    """

    def __call__(
        self,
        env: ManagerBasedEnv,
        k: float = 40.0,
        side: str = "both",
        hinge_threshold_m: float = 0.0,
    ) -> torch.Tensor:
        ref_body = self._traj_mgr.get_reference_body_pos(env.episode_length_buf)
        origins = env.scene.env_origins

        rh = env.scene["right_hand"]
        lh = env.scene["left_hand"]

        # Right hand (frame-consistent: link7 -> virtual wrist for the full robot)
        rh_cur, _ = self._virtual_wrist_pose(
            rh, self._rh_palm_body_id, self._rh_is_link7, origins, t_offset=self._rh_wrist_t_offset
        )
        rh_ref = ref_body["right_body_pos"][:, _PALM_IDX, :]
        rh_dist = torch.norm(rh_ref - rh_cur, dim=-1)

        # Left hand
        lh_cur, _ = self._virtual_wrist_pose(
            lh, self._lh_palm_body_id, self._lh_is_link7, origins, t_offset=self._lh_wrist_t_offset
        )
        lh_ref = ref_body["left_body_pos"][:, _PALM_IDX, :]
        lh_dist = torch.norm(lh_ref - lh_cur, dim=-1)

        if hinge_threshold_m > 0.0:
            rh_excess = torch.clamp(rh_dist - hinge_threshold_m, min=0.0)
            lh_excess = torch.clamp(lh_dist - hinge_threshold_m, min=0.0)
            return self._combine(torch.exp(-k * rh_excess), torch.exp(-k * lh_excess))

        return self._combine(torch.exp(-k * rh_dist), torch.exp(-k * lh_dist))


# ---------------------------------------------------------------------------
# 2. Wrist rotation tracking (via joint angles)
# ---------------------------------------------------------------------------


class RewardEefRot(_BimanualRewardBase):
    """Wrist rotation tracking.

    Full robot (IK wrist): quaternion **geodesic** error between the simulated
    virtual-wrist orientation (link7 -> virtual wrist via the fixed transform) and
    the reference palm orientation (``get_reference_body_pos`` ``*_body_quat``):

        angle = 2*acos(|<q_ref, q_cur>|);  reward = exp(-k * angle)   [k in 1/rad]

    This replaces the previous virtual-wrist-joint formulation, which silently
    returned a constant 1.0 for the full robot (no virtual wrist joints) and gave
    zero gradient. Floating hand (virtual wrist joints present): keep the original
    joint-angle formulation.
    """

    def __call__(
        self,
        env: ManagerBasedEnv,
        k: float = 3.0,
        side: str = "both",
        # ⚠ DEAD PARAMETER: nothing ever sets this. env_setup's
        # --enable_hinge block wires the 5 fingertips and eef_pos only, so the
        # rotation hinge has never been active. Left in place so a rotation hinge
        # can be enabled by wiring it there — not by editing this default.
        hinge_threshold_rad: float = 0.0,
    ) -> torch.Tensor:
        from isaaclab.utils.math import quat_conjugate, quat_mul

        N = env.num_envs
        rh = env.scene["right_hand"]
        lh = env.scene["left_hand"]

        # Full-robot path: frame-consistent quaternion geodesic.
        if self._rh_is_link7 or self._lh_is_link7:
            origins = env.scene.env_origins
            ref_body = self._traj_mgr.get_reference_body_pos(env.episode_length_buf)

            def _geodesic(asset, palm_id, is_link7, ref_key, r_offset):
                if not is_link7 or ref_key not in ref_body:
                    return torch.ones(N, device=env.device)
                _, cur_q = self._virtual_wrist_pose(asset, palm_id, True, origins, r_offset=r_offset)
                ref_q = ref_body[ref_key][:, _PALM_IDX, :]
                qe = quat_mul(quat_conjugate(ref_q), cur_q)
                ang = 2.0 * torch.acos(torch.clamp(torch.abs(qe[:, 0]), max=1.0))
                if hinge_threshold_rad > 0.0:
                    ang = torch.clamp(ang - hinge_threshold_rad, min=0.0)
                return torch.exp(-k * ang)

            rh_reward = _geodesic(
                rh, self._rh_palm_body_id, self._rh_is_link7, "right_body_quat", self._rh_wrist_r_offset
            )
            lh_reward = _geodesic(
                lh, self._lh_palm_body_id, self._lh_is_link7, "left_body_quat", self._lh_wrist_r_offset
            )
            return self._combine(rh_reward, lh_reward)

        # Floating-hand path: original virtual-wrist-joint angle error.
        ref_state = self._traj_mgr.get_reference_state(env.episode_length_buf)
        if self._rh_wrist_rot_ids is not None:
            rh_ref_rot = ref_state["right_hand_qpos"][:, self._rh_wrist_rot_ids]
            rh_cur_rot = rh.data.joint_pos[:, self._rh_wrist_rot_ids]
            rh_err = torch.sum(torch.abs(rh_ref_rot - rh_cur_rot), dim=-1)
            rh_reward = torch.exp(-k * rh_err)
        else:
            rh_reward = torch.ones(N, device=env.device)

        if self._lh_wrist_rot_ids is not None:
            lh_ref_rot = ref_state["left_hand_qpos"][:, self._lh_wrist_rot_ids]
            lh_cur_rot = lh.data.joint_pos[:, self._lh_wrist_rot_ids]
            lh_err = torch.sum(torch.abs(lh_ref_rot - lh_cur_rot), dim=-1)
            lh_reward = torch.exp(-k * lh_err)
        else:
            lh_reward = torch.ones(N, device=env.device)

        return self._combine(rh_reward, lh_reward)


# ---------------------------------------------------------------------------
# 3-7. Fingertip position tracking
# ---------------------------------------------------------------------------


class _FingertipReward(_BimanualRewardBase):
    """Fingertip position tracking of the class's FINGER: exp(-k * ||ref - cur||); ``k`` defaults to K_DEFAULT.

    Supports soft hinge via ``hinge_threshold_m`` (default 0.0 = disabled). When > 0, errors within the threshold
    give full reward (1.0); exponential decay applies only to the excess beyond the threshold.
    """

    FINGER: str = ""
    K_DEFAULT: float = 0.0

    def __call__(
        self,
        env: ManagerBasedEnv,
        k: float | None = None,
        side: str = "both",
        hinge_threshold_m: float = 0.0,
        contact_gated_hinge: bool = False,
        hinge_threshold_in_contact: float = 0.04,
        contact_force_threshold: float = 5.0,
    ) -> torch.Tensor:
        if k is None:
            k = self.K_DEFAULT
        return self._fingertip_reward(
            env,
            self.FINGER,
            k,
            hinge_threshold_m,
            contact_gated_hinge=contact_gated_hinge,
            hinge_threshold_in_contact=hinge_threshold_in_contact,
            contact_force_threshold=contact_force_threshold,
        )


class RewardThumbTip(_FingertipReward):
    """Thumb tip position tracking: exp(-k * ||ref - cur||).

    Supports soft hinge via ``hinge_threshold_m`` (default 0.0 = disabled).
    When > 0, errors within the threshold give full reward (1.0); exponential
    decay applies only to the excess beyond the threshold.
    """

    FINGER = "thumb"
    K_DEFAULT = 100.0


class RewardIndexTip(_FingertipReward):
    """Index fingertip position tracking: exp(-k * ||ref - cur||).

    Supports soft hinge via ``hinge_threshold_m`` (default 0.0 = disabled).
    """

    FINGER = "index"
    K_DEFAULT = 90.0


class RewardMiddleTip(_FingertipReward):
    """Middle fingertip position tracking: exp(-k * ||ref - cur||)."""

    FINGER = "middle"
    K_DEFAULT = 80.0


class RewardRingTip(_FingertipReward):
    """Ring fingertip position tracking: exp(-k * ||ref - cur||)."""

    FINGER = "ring"
    K_DEFAULT = 60.0


class RewardPinkyTip(_FingertipReward):
    """Pinky fingertip position tracking: exp(-k * ||ref - cur||)."""

    FINGER = "pinky"
    K_DEFAULT = 60.0


# ---------------------------------------------------------------------------
# 8. Proximal joint (level 1) position tracking
# ---------------------------------------------------------------------------


class RewardLevel1(_BimanualRewardBase):
    """Proximal joint position tracking (average over level_1 bodies).

    exp(-k * mean(||ref - cur|| for each level_1 body))
    """

    def __call__(self, env: ManagerBasedEnv, k: float = 50.0, side: str = "both") -> torch.Tensor:
        return self._body_group_reward(env, self._fk_level1, k)


# ---------------------------------------------------------------------------
# 9. Intermediate joint (level 2) position tracking
# ---------------------------------------------------------------------------


class RewardLevel2(_BimanualRewardBase):
    """Intermediate joint position tracking (average over level_2 bodies).

    exp(-k * mean(||ref - cur|| for each level_2 body))
    """

    def __call__(self, env: ManagerBasedEnv, k: float = 40.0, side: str = "both") -> torch.Tensor:
        return self._body_group_reward(env, self._fk_level2, k)


# ---------------------------------------------------------------------------
# 10. Wrist linear velocity tracking
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 11. Wrist angular velocity tracking
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 12b. Finger joint-angle (pose) tracking [high-DoF anti-contortion]
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 13. Total power penalty
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 14. Wrist power penalty
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 15. Wrist force effort penalty (for force/torque control mode)
# ---------------------------------------------------------------------------
