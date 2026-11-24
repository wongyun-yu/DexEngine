# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Taco training: floating hands with position-controlled wrist.

Phase 1c/d/e integrated config: desk scene + full MDP terms + position wrist.
Uses standalone RH56F1 floating hands (same as Force env) but with:
- Position-controlled wrist actuators (Kp=200, Kd=30), ManipTrans-matched finger physics (Kp=200, Kd=100; the real hand
is not PD-controlled — the old "real HW Kp=10/Kd=0.2" here was the asset-file placeholder)
- WristPositionFingersPositionAction (trajectory-relative delta wrist + PD fingers)
- Full ManipTrans-style observations, rewards, terminations, curriculum, events
- desk (visual: Willow, desk_visual.desk_cfg) with collision cuboids, floor, dome light
- Trajectory data rotated to +X forward at desk_height=0.76m

Scene hierarchy:
  /World/Floor          — 20x20m visual floor (no material of its own)
  /World/GroundPlane    — physics ground plane
  /World/light          — dome light
  {ENV}/Desk            — visual desk (Willow)
  {ENV}/DeskCollisionTop — tabletop collision cuboid
  {ENV}/DeskLeg{0-3}    — 4 leg collision cuboids
  {ENV}/RightHand       — floating RH56F1 right hand (position wrist)
  {ENV}/LeftHand        — floating RH56F1 left hand (position wrist)
  {ENV}/Object          — underground placeholder (no object in hand-tracking training)
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from dexengine.rl.tasks.mdp import terms as mdp

# Common constants live in cfg/constants.py.

##
# Scene definition
##


# This module hosts the task MDP cfg classes (observations / rewards /
# terminations / events / curriculum) consumed by build_task_mdp_cfg_classes.

##
# MDP settings
##


@configclass
class RightHandObservationsCfg:
    """Hand-tracking (arm-IK) base observations. The residual task env swaps
    these for the V2 layout at launch (apply_obs_v2) — see the task obs cfg
    for the current layout."""

    @configclass
    class PolicyCfg(ObsGroup):
        right_hand = ObsTerm(
            func=mdp.HandObservation,
            params={"hand_name": "right_hand"},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RightHandRewardsCfg:
    """Full 14-term ManipTrans reward for right hand with position-controlled wrist.

    Uses RewardWristPower since the wrist is position-controlled
    wrist is driven by PD actuators, not external force commands.
    """

    # -- Position tracking --
    # Wrist rewards made frame-consistent (link7 -> virtual wrist) in mdp.RewardEefPos/
    # RewardEefRot. eef_pos weight 0.1->0.6 (was ~6-9x weaker than a
    # fingertip term while measuring a spurious 5.8cm offset). eef_rot k 1.0->3.0:
    # now a real quaternion-geodesic reward (was a constant 1.0 / inert for the IK wrist).
    eef_pos = RewTerm(func=mdp.RewardEefPos, weight=0.6, params={"k": 40.0, "side": "right"})
    eef_rot = RewTerm(func=mdp.RewardEefRot, weight=0.6, params={"k": 3.0, "side": "right"})
    # -- Exp-1 stability: soft L2 penalty (dead-zone) on residual action magnitude.
    #    weight 0.0 here = inert by default; set via --residual_l2_weight (coefficient)
    #    and --residual_l2_deadzone, applied in rl/train/runner.py train_cfg_overrides. Softly
    #    discourages actor-gain drift (residual ‖mu‖ inflation → policy collapse)
    #    without a hard clip.
    residual_l2 = RewTerm(func=mdp.residual_norm_penalty, weight=0.0, params={"deadzone": 0.5, "side": "right"})
    # -- Fingertip tracking -- (soft hinge controlled by --fingertip_hinge_cm, default 2cm ON;
    #    set 0 for OFF. Applied in rl.train.runner train_cfg_overrides as the single source of truth
    #    so the hinge/non-hinge A/B is reproducible without editing this file.)
    thumb_tip = RewTerm(func=mdp.RewardThumbTip, weight=0.9, params={"k": 100.0, "side": "right"})
    index_tip = RewTerm(func=mdp.RewardIndexTip, weight=0.8, params={"k": 90.0, "side": "right"})
    middle_tip = RewTerm(func=mdp.RewardMiddleTip, weight=0.75, params={"k": 80.0, "side": "right"})
    ring_tip = RewTerm(func=mdp.RewardRingTip, weight=0.6, params={"k": 60.0, "side": "right"})
    pinky_tip = RewTerm(func=mdp.RewardPinkyTip, weight=0.6, params={"k": 60.0, "side": "right"})
    # -- Body group tracking --
    level_1 = RewTerm(func=mdp.RewardLevel1, weight=0.5, params={"k": 50.0, "side": "right"})
    level_2 = RewTerm(func=mdp.RewardLevel2, weight=0.3, params={"k": 40.0, "side": "right"})
    # -- Finger joint-angle (pose) tracking [high-DoF anti-contortion] -- exp(-k*mean|q-q_ref|)
    #    over the independent finger joints vs the retarget pose. The fingertip-POSITION terms
    #    above leave a 1-2 DOF/finger null space on SharpA (4-5 DOF/finger) which the policy fills
    #    with over-bent poses (thumb IP +56°); this pins the pose to the faithful retarget.
    #    weight 0 = OFF by default (RH56F1 coupled, no redundancy); set via --finger_pose_weight.
    finger_pose = RewTerm(func=mdp.RewardFingerPose, weight=0.0, params={"k": 4.0, "side": "right"})
    # -- Velocity tracking --
    eef_vel = RewTerm(func=mdp.RewardEefVel, weight=0.1, params={"k": 1.0, "side": "right"})
    # eef_ang_vel removed: ref (Euler rates) vs cur (PhysX ang_vel) frame mismatch. Weight=0.05, minimal impact.
    joints_vel = RewTerm(func=mdp.RewardJointsVel, weight=0.1, params={"k": 1.0, "side": "right"})
    # -- Power penalties (reduced: they over-damped motion / favoured static
    #    poses; wrist_power also fought wrist tracking). power 0.5->0.2, wrist_power 0.5->0.15. --
    power = RewTerm(func=mdp.RewardPower, weight=0.2, params={"k": 10.0, "side": "right"})
    wrist_power = RewTerm(func=mdp.RewardWristPower, weight=0.15, params={"k": 2.0, "side": "right"})
    # NOTE: action_rate is NOT in this base class. The old comment claiming it
    # mirrored an OakInk addition was wrong — OakInk's base
    # RightHandRewardsCfg never had action_rate at the hand-tracking level
    # (only at the task level via the task reward subclass with weight=0.15).
    # Adding a 0.05 action_rate here silently drifted TACO hand-tracking ArmIK
    # away from OakInk's reward signature. Removed for OakInk parity.


@configclass
class RightHandTerminationsCfg:
    """Full ManipTrans terminations for right hand."""

    trajectory_done = DoneTerm(func=mdp.trajectory_done, time_out=True)

    fingertip_exceeded = DoneTerm(
        func=mdp.FingertipTrackingExceeded,
        time_out=False,
        params={
            "thresholds_start": {"thumb": 0.08, "index": 0.06, "middle": 0.06, "ring": 0.06, "pinky": 0.06},
            "thresholds_end": {"thumb": 0.04, "index": 0.04, "middle": 0.04, "ring": 0.04, "pinky": 0.04},
            "side": "right",
        },
    )

    velocity_sanity = DoneTerm(
        func=mdp.velocity_sanity,
        time_out=False,
        params={
            "lin_vel_threshold": 100.0,
            "ang_vel_threshold": 200.0,
            "joints_vel_threshold": 100.0,
            "dof_vel_threshold": 200.0,
            "side": "right",
        },
    )


@configclass
class RightHandEventCfg:
    """Reset events for right hand — trajectory-aware hand reset."""

    reset_hands = EventTerm(
        func=mdp.reset_hands_to_trajectory,
        mode="reset",
        params={},
    )

    reset_object = EventTerm(
        func=mdp.reset_object_to_trajectory,
        mode="reset",
        params={"side": "right"},
    )

    # Initial pose randomization: disabled by default (fraction=0).
    # Enabled via --randomize CLI flag which sets fraction > 0 at runtime.
    # Only finger joints are perturbed (opening-only). Wrist stays fixed.
    randomize_pose = EventTerm(
        func=mdp.randomize_initial_pose,
        mode="reset",
        params={
            "randomize_fraction": 0.0,  # 0 = disabled; 0.5 = 50% of episodes
            "finger_open_deg": 0.0,  # 0 = disabled; 15.0 = up to 15° opening per joint
            "wrist_z_max_cm": 0.0,  # 0 = disabled; 1.5 = up to 1.5cm upward lift
        },
    )

    # Per-episode physics DR (Gate 0 parity; mirrors OakInk hand-tracking).
    # All 4 events removed at runtime when --enable_physics_dr=none. Ranges set
    # here are placeholders for the launcher's DR ranges (--enable_physics_dr).
    physics_dr_mass = EventTerm(
        func=mdp.randomize_object_mass_at_reset,
        mode="reset",
        params={
            "object_name": "object_right",
            "mass_scale_range": (0.85, 1.15),
        },
    )
    physics_dr_friction = EventTerm(
        func=mdp.randomize_object_friction_at_reset,
        mode="reset",
        params={
            "object_name": "object_right",
            "static_friction_scale_range": (0.75, 1.25),
            "dynamic_friction_scale_range": (0.75, 1.25),
        },
    )
    physics_dr_restitution = EventTerm(
        func=mdp.randomize_object_restitution_at_reset,
        mode="reset",
        params={
            "object_name": "object_right",
            "restitution_scale_range": (0.8, 1.2),
        },
    )
    physics_dr_com = EventTerm(
        func=mdp.randomize_object_com_at_reset,
        mode="reset",
        params={
            "object_name": "object_right",
            "com_offset_range_m": (-0.01, 0.01),
        },
    )


@configclass
class TrackingCurriculumCfg:
    """Curriculum with reduced RSI for position-controlled wrist.

    target_fraction=0.5 ensures 50% of episodes start from frame 0.
    This gives the policy dense training on full-trajectory tracking,
    improving deterministic eval performance.
    """

    rsi_schedule = CurrTerm(
        func=mdp.RsiSchedule,
        params={
            "ramp_episodes": 10000,
            "target_fraction": 0.5,
        },
    )

    threshold_tightening = CurrTerm(
        func=mdp.LinearThresholdSchedule,
        params={"anneal_steps": 1},  # patched at launch by rl/train/runner.py to int(0.5 * total_timesteps)
    )


##
# Environment configuration
##
