# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Arm IK + finger position (RL-driven) action term.

The arm is driven by DifferentialIK toward a wrist target. Two modes:

  Default (current-state):  target = current_link7_state + policy_delta
  Trajectory-reference:     target = trajectory_wrist_ref[frame] + policy_delta

Current-state reads actual physics each step (backward-compatible baseline).
Trajectory-reference uses the ground-truth wrist pose from the trajectory
manager, eliminating the state-chasing feedback loop.

The RL policy outputs 12-dim actions:
6 wrist deltas + 6 finger position targets in [-1, 1].

Finger position targets are scaled to joint limits via ManipTrans-style
absolute mapping:

    target = 0.5 * (action + 1) * (upper - lower) + lower

Finger coupling (6 independent -> 12 physical joints) is handled by
PhysxMimicJointAPI in the USD, so we only set the 6 independent joints.

The wrist->link7 offset conversion is done in batched PyTorch on GPU,
equivalent to the verified scipy-based ``wrist_pose_to_link7_target``.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    pass
from dexengine.rl.tasks.mdp.actions.arm_ik_term import (  # noqa: F401
    ArmIKFingersPositionAction,
    check_ema_weight,
    logger,
)

# ------------------------------------------------------------------
# Operations
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


@configclass
class ArmIKFingersPositionActionCfg(ActionTermCfg):
    """Configuration for :class:`ArmIKFingersPositionAction`.

    Combines DifferentialIK for the 7-DOF arm with ManipTrans-style position
    targets for 6 independent finger DOFs. Both wrist and fingers are policy-
    controlled, matching WristPositionFingersPositionAction.

    The RL action space is 12-dimensional (per hand):
      action[0:3]  = wrist position delta (scaled by wrist_pos_scale)
      action[3:6]  = wrist rotation delta (scaled by wrist_rot_scale)
      action[6..11] = finger positions in [-1, 1]

    The wrist target = base + scaled_delta, then converted to link7 target
    for DifferentialIK arm control. Two base modes:
      - current-state (default): base = actual link7 state each step
      - trajectory-reference: base = trajectory wrist pose for current frame

    Optional temporal ensembling (EgoVLA-style) smooths raw actions before
    EMA, with separate weights for wrist and finger components.
    """

    class_type: type[ActionTerm] = ArmIKFingersPositionAction

    asset_name: str = "robot"
    """Name of the robot articulation in the scene."""

    arm_joint_names: list[str] = MISSING
    """Regex/glob patterns matching the 7 arm joints.

    Example: ``["right_arm_[0-6]"]``
    """

    ee_body_name: str = MISSING
    """Name of the end-effector body for IK (link7, not the wrist).

    Example: ``"link_right_arm_6"``
    """

    ik_lambda_val: float = 0.02

    finger_rom_floor: float | None = None
    """When not None, floor the lower bound of the finger flexion joints' action mapping at this value (rad) (a human
    ROM approximation, an experiment blocking the wuji −27° hyperextension). The residual REFERENCE-BASE normalisation
    uses the same bound, so a reference outside the ROM clamps the base too. Default None = byte-identical."""

    finger_rom_floor_exclude: str | None = None
    """Fullmatch regex of the joints the ROM floor skips; no default. With the floor on, the action contract fills it
    from HandSpec.abduction_joint_regex; empty fails at action construction."""
    """Damping coefficient for DLS inverse kinematics.

    Matches the verified value from step6 (``lambda_val=0.02``).
    Higher values increase stability at the cost of tracking accuracy.
    """

    finger_joint_names: list[str] = MISSING
    """Explicit list of the independent finger joint names (defines the
    finger action dimension; action_dim = 6 + len(finger_joint_names)).

    These must be the actuated (independent) joints only. The coupled
    joints are driven automatically by PhysxMimicJointAPI in the USD
    (or by the explicit coupling below for full-robot USDs).

    RH56F1 example (6)::

        [
            "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
            "right_hand_index_1_joint", "right_hand_middle_1_joint",
            "right_hand_ring_1_joint", "right_hand_little_1_joint",
        ]

    SharpA Wave: the 22 vendor-order joints (see naming.SHARPA_FINGER_JOINT_SUFFIXES).
    """

    coupled_joint_names: list[str] | None = None
    """Coupled (mimic) joint names for explicit coupling on full-robot USDs.

    - None (default): legacy RH56F1 auto-list ({side}_hand_{thumb_3,thumb_4,
      index_2,middle_2,ring_2,little_2}_joint) — current behavior.
    - [] : no coupling (fully-actuated hand, e.g. SharpA Wave).
    """

    coupling_ratios: list[float] | None = None
    """Coupling ratios, one per coupled joint (chained ratios pre-multiplied).
    None (default) = legacy RH56F1 ratios."""

    coupling_source_idx: list[int] | None = None
    """Index into finger_joint_names of the driving joint for each coupled joint.
    None (default) = legacy RH56F1 mapping [1, 1, 2, 3, 4, 5]."""

    link7_to_wrist_offset: tuple[float, float, float] | None = None
    """link7 -> virtual-wrist translation in link7 frame (rotation is always
    Rz(-90deg)). None (default) = RH56F1 (0, 0, 0.0555). SharpA Wave uses
    (0, 0, 0.0655) (link7 -> flange; see naming.LINK7_TO_WRIST_OFFSET_SHARPA)."""

    wrist_pos_scale: float = 0.1
    """Scaling factor for wrist position delta (meters per unit action).

    Must match training's WristPositionFingersPositionActionCfg.wrist_pos_scale.
    """

    wrist_rot_scale: float = 0.5
    """Scaling factor for wrist rotation delta (radians per unit action).

    Must match training's WristPositionFingersPositionActionCfg.wrist_rot_scale.
    """

    action_ema: float = 1.0
    """Exponential moving average factor for action smoothing.

    ``processed = ema * new + (1 - ema) * prev``

    - 1.0 = no smoothing (pass-through, matches hand-tracking training).
    """

    ik_target_ema: float = 1.0
    """EMA smoothing on the IK target (wrist qpos).

    ``target = ema * new_target + (1 - ema) * prev_target``

    - 1.0 = no smoothing (pass-through).
    - 0.3-0.5 = moderate smoothing, reduces trembling during stationary phases.

    NOTE: This EMA is applied to the FINAL IK target, after adding the policy delta.
    For the residual path (base + residual), this smooths the combined output.
    Future options: smooth only base, only residual, or only the final target.
    """

    use_trajectory_reference: bool = False
    """If True, compute wrist target as trajectory_reference + delta instead of
    current_state + delta.  This eliminates the state-chasing feedback loop that
    causes trembling, at the cost of requiring a trajectory manager.

    - False (default): target = current_link7_state + delta (baseline, backward-compat)
    - True: target = trajectory_wrist_reference[current_frame] + delta (stable)
    """

    wrist_smooth_weight: float = -1.0
    """Temporal ensembling weight for wrist action dims (0-5).
    Exponential decay: weight[i] = exp(-smooth_weight * i) for history frame i.
    Lower = heavier smoothing. EgoVLA uses 0.2 for EE.
    -1.0 = disabled (no temporal ensembling)."""

    finger_smooth_weight: float = -1.0
    """Temporal ensembling weight for finger action dims (6 to 11).
    EgoVLA uses 0.8 for hands (lighter smoothing than arms).
    -1.0 = disabled (no temporal ensembling)."""

    smooth_history_len: int = 8
    """Number of past actions to keep for temporal ensembling.
    Only used when wrist_smooth_weight >= 0 or finger_smooth_weight >= 0."""
