# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dataset-agnostic task (object-manipulation) MDP cfg factory.

Builds the task MDP cfg classes (obs/rewards/terminations/events) from a small
parameter set so a new dataset reuses the same construction logic.

This module exposes:

  - ``RightHandTaskObservationsCfg``
  - ``RightHandTaskRewardsCfg`` (depends on a parent rewards cfg)
  - ``RightHandTaskTerminationsCfg`` (depends on a parent terminations cfg)
  - ``RightHandTaskEventCfg`` (depends on a parent event cfg)
  - ``TaskCurriculumCfg`` (depends on a parent curriculum cfg)

Top-level helpers:

  - ``build_task_mdp_cfg_classes(mdp, parents...)`` — builds the
    six dataset-agnostic MDP cfg classes above from a parent set
    (Observations/Rewards/Force-Rewards/Terminations/Events/Curriculum
    are produced as subclasses of the parents the dataset provides).

Dataset packages call it at import time and re-export the returned
classes under their dataset-specific public names.

Each @configclass body lists its fields in the order Isaac Lab reads them (the reward sum order, the event
application order); multi-field term values come from small helpers, so a class body is one line per field.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from dexengine.core.physics import GRAVITY_MPS2
from dexengine.rl.tasks import env_flags
from dexengine.rl.tasks.mdp.observations.object_state_v2 import ObjectStateV2 as _ObjStateV2
from dexengine.rl.tasks.mdp.observations.v2 import HandObservationV2 as _HandObsV2

# Build-time ramp length of the gravity_ramp and friction_decay events: 20 % of the default horizon (2000 iterations
# x 32 rollouts). Training resolves the effective ramp at runtime from --gravity_ramp_fraction.
_FORCE_RAMP_FRACTION = 0.20
_RAMP_STEPS_BUILD_DEFAULT = int(2000 * 32 * _FORCE_RAMP_FRACTION)  # 12800; overridden at runtime


def _arm_joint_obs(env, action_key: str):
    """DEXENGINE_ARM_JOINT_RESIDUAL policy obs (2·arm joints): that action term's arm joints minus the frame's reference
    arm joints, then the arm joint velocities (ResidualArmIKAction.arm_joint_obs)."""
    return env.action_manager._terms[action_key].arm_joint_obs()


def build_task_mdp_cfg_classes(
    *,
    mdp,
    rewards_cfg_parent_cls: type,
    terminations_cfg_parent_cls: type,
    event_cfg_parent_cls: type,
    curriculum_cfg_parent_cls: type,
    bimanual: bool = False,
    left_uni: bool = False,
) -> dict[str, type]:
    """Build the task MDP cfg classes (dataset-agnostic factory).

    Returns a dict mapping the canonical names
    (``RightHandTaskObservationsCfg`` etc.) to freshly-built configclasses
    that subclass the dataset's parent rewards/terminations/events/curriculum.

    Bimanual Option 2 (``bimanual=True``): appends LEFT-hand observation terms
    (after the 4 right terms, before the final ``actions``), LEFT object-tracking
    + hand-tracking rewards (full mirror at equal weight, side='left',
    object_key='object_left'), a per-hand split of the action_rate / residual_l2
    regularizers, and a LEFT object-tracking termination. All gated so the
    right-only path is byte-identical when ``bimanual=False`` (default).

    LEFT-UNI (``left_uni=True``): the OBSERVATION group is the exact
    LEFT mirror of the right-only 5-term set (left_hand / last_base_action with
    action_key='left_hand_joint_pos' / object_privileged / object_target with
    side='left', object_key='object' / actions) — SAME unsuffixed term names so
    the obs layout, W&B keys and the runtime obs_v2 swap stay identical to the
    right-uni build. Rewards/terminations keep their side='right' cfg params
    here — the runtime ``apply_left_uni`` sweep (env_setup.sides, invoked
    from rl.train.runner / rl.eval) flips
    every params['side'] to 'left' in ONE place (single-sweep policy: runtime-
    registered terms like contact-pilot/keypoint/residual_l2_wrist must be
    flipped too, so the flip lives at the END of the cfg-override functions).
    Mutually exclusive with ``bimanual``.
    """
    if left_uni and bimanual:
        raise ValueError("build_task_mdp_cfg_classes: left_uni=True is mutually exclusive with bimanual=True.")

    RightHandTaskObservationsCfg = _build_task_observations_cfg(mdp, bimanual, left_uni)

    RightHandTaskRewardsCfg = _build_task_rewards_cfg(mdp, rewards_cfg_parent_cls, bimanual)

    RightHandTaskTerminationsCfg = _build_task_terminations_cfg(mdp, terminations_cfg_parent_cls, bimanual)

    RightHandTaskEventCfg = _build_task_event_cfg(mdp, event_cfg_parent_cls)

    TaskCurriculumCfg = _build_task_curriculum_cfg(mdp, curriculum_cfg_parent_cls)

    return {
        "RightHandTaskObservationsCfg": RightHandTaskObservationsCfg,
        "RightHandTaskRewardsCfg": RightHandTaskRewardsCfg,
        "RightHandTaskTerminationsCfg": RightHandTaskTerminationsCfg,
        "RightHandTaskEventCfg": RightHandTaskEventCfg,
        "TaskCurriculumCfg": TaskCurriculumCfg,
    }


def _build_task_observations_cfg(mdp, bimanual, left_uni):
    """RightHandTaskObservationsCfg: the policy observation group of the right, left or both hands."""

    # ---------------------------------------------------------------------
    # Observations
    # ---------------------------------------------------------------------
    @configclass
    class RightHandTaskObservationsCfg:
        """Task observations — V2-native (per hand: hand state + reference-base
        action + 87 object_state [kin 15 + ref 18 + surface 35 + priv 19] + last
        action; the hand terms follow the hand's widths). The V1 449-dim layout (HandObservation 201 +
        LastBaseAction + ObjectPrivileged + ObjectTarget/BPS) is gone;
        mesh_dir params are wired at launch (env_setup.build).
        """

        @configclass
        class PolicyCfg(ObsGroup):
            # LEFT-UNI: the right terms are REPLACED by LEFT mirrors under the SAME
            # slot names (arch/W&B/perhand-routing parity). mesh_dir params are
            # wired at launch from the active-side CLI flags.
            # CKPT-LAYOUT CONTRACT: term ORDER is frozen to what the trained
            # checkpoints/normalizers saw — hand, last_base_action, (left mirrors,)
            # actions, object_state(, object_state_left). object_state comes AFTER
            # actions because the historical V1->V2 runtime swap APPENDED it; the
            # RunningStandardScaler stats and token slicing are order-sensitive.
            if left_uni:
                left_hand = ObsTerm(func=_HandObsV2, params={"hand_name": "left_hand"})
                last_base_action = ObsTerm(
                    func=mdp.ReferenceBaseAction,
                    params={"action_key": "left_hand_joint_pos"},
                )
            else:
                right_hand = ObsTerm(func=_HandObsV2, params={"hand_name": "right_hand"})
                last_base_action = ObsTerm(
                    func=mdp.ReferenceBaseAction,
                    params={"action_key": "right_hand_joint_pos"},
                )
                if bimanual:
                    left_hand = ObsTerm(func=_HandObsV2, params={"hand_name": "left_hand"})
                    left_last_base_action = ObsTerm(
                        func=mdp.ReferenceBaseAction,
                        params={"action_key": "left_hand_joint_pos"},
                    )
            actions = ObsTerm(func=mdp.last_action)
            if left_uni:
                object_state = ObsTerm(
                    func=_ObjStateV2,
                    params={"side": "left", "object_key": "object_left", "mesh_dir": ""},
                )
            else:
                object_state = ObsTerm(
                    func=_ObjStateV2,
                    params={"side": "right", "object_key": "object_right", "mesh_dir": ""},
                )
                if bimanual:
                    object_state_left = ObsTerm(
                        func=_ObjStateV2,
                        params={"side": "left", "object_key": "object_left", "mesh_dir": ""},
                    )
            # DEXENGINE_ARM_JOINT_RESIDUAL (RL ablation j · default OFF = no term, layout unchanged): the arm-joint
            # action
            # observes its own arm joints (minus the frame's reference arm joints) and their velocities. LAST, so every
            # term above keeps its offset; the name routes it to its hand (perhand_obs_slices /
            # _token_slices_bimanual).
            if env_flags.get("DEXENGINE_ARM_JOINT_RESIDUAL") == "1":
                arm_joint = ObsTerm(
                    func=_arm_joint_obs,
                    params={"action_key": "left_hand_joint_pos" if left_uni else "right_hand_joint_pos"},
                )
                if bimanual:
                    arm_joint_left = ObsTerm(func=_arm_joint_obs, params={"action_key": "left_hand_joint_pos"})

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        policy: PolicyCfg = PolicyCfg()

    return RightHandTaskObservationsCfg


def _build_task_rewards_cfg(mdp, rewards_cfg_parent_cls, bimanual):
    """RightHandTaskRewardsCfg: the parent hand-tracking rewards plus the object-tracking rewards; bimanual adds the
    left mirror of every term at equal weight."""

    @configclass
    class RightHandTaskRewardsCfg(rewards_cfg_parent_cls):
        """Task rewards: all hand-tracking rewards + object tracking.

        Uses RewardWristPower since the robot uses a position-controlled wrist, not force/torque commands.
        action_rate at 0.10 keeps fast finger motion; obj_pos at 2.5 keeps the object-position reward from
        dominating fingertip tracking (sum ~3.65) and pushing fingertips off the reference before contact; obj_rot
        at 2.0 because rotation tracking limits once the object is moving.
        """

        action_rate = RewTerm(func=mdp.RewardActionRate, weight=0.10, params={"k": 1.0, "side": "right"})
        obj_pos = RewTerm(func=mdp.RewardObjPos, weight=2.5, params={"k": 80.0, "side": "right"})
        obj_rot = RewTerm(func=mdp.RewardObjRot, weight=2.0, params={"k": 3.0, "side": "right"})
        obj_vel = RewTerm(func=mdp.RewardObjVel, weight=0.1, params={"k": 1.0, "side": "right"})
        obj_ang_vel = RewTerm(func=mdp.RewardObjAngVel, weight=0.1, params={"k": 1.0, "side": "right"})

        # Bimanual: the left mirror at equal weight of every right hand-tracking and object-tracking term
        # (side='left', object_key='object_left'; no left farthest-keypoint term). action_rate_left and
        # residual_l2_left are the per-hand regularizers, each reading that hand's 12-dim action slice. The left
        # closure / contact_match / pregrasp / force_dir / grasp terms are wired at launch (rl/train, rl/eval).
        if bimanual:
            action_rate_left = RewTerm(func=mdp.RewardActionRate, weight=0.10, params={"k": 1.0, "side": "left"})
            residual_l2_left = RewTerm(
                func=mdp.residual_norm_penalty, weight=0.0, params={"deadzone": 0.5, "side": "left"}
            )
            obj_pos_left = RewTerm(
                func=mdp.RewardObjPos, weight=2.5, params={"k": 80.0, "side": "left", "object_key": "object_left"}
            )
            obj_rot_left = RewTerm(
                func=mdp.RewardObjRot, weight=2.0, params={"k": 3.0, "side": "left", "object_key": "object_left"}
            )
            obj_vel_left = RewTerm(
                func=mdp.RewardObjVel, weight=0.1, params={"k": 1.0, "side": "left", "object_key": "object_left"}
            )
            obj_ang_vel_left = RewTerm(
                func=mdp.RewardObjAngVel, weight=0.1, params={"k": 1.0, "side": "left", "object_key": "object_left"}
            )
            eef_pos_left = RewTerm(func=mdp.RewardEefPos, weight=0.6, params={"k": 40.0, "side": "left"})
            eef_rot_left = RewTerm(func=mdp.RewardEefRot, weight=0.6, params={"k": 3.0, "side": "left"})
            thumb_tip_left = RewTerm(func=mdp.RewardThumbTip, weight=0.9, params={"k": 100.0, "side": "left"})
            index_tip_left = RewTerm(func=mdp.RewardIndexTip, weight=0.8, params={"k": 90.0, "side": "left"})
            middle_tip_left = RewTerm(func=mdp.RewardMiddleTip, weight=0.75, params={"k": 80.0, "side": "left"})
            ring_tip_left = RewTerm(func=mdp.RewardRingTip, weight=0.6, params={"k": 60.0, "side": "left"})
            pinky_tip_left = RewTerm(func=mdp.RewardPinkyTip, weight=0.6, params={"k": 60.0, "side": "left"})
            level_1_left = RewTerm(func=mdp.RewardLevel1, weight=0.5, params={"k": 50.0, "side": "left"})
            level_2_left = RewTerm(func=mdp.RewardLevel2, weight=0.3, params={"k": 40.0, "side": "left"})
            finger_pose_left = RewTerm(func=mdp.RewardFingerPose, weight=0.0, params={"k": 4.0, "side": "left"})
            eef_vel_left = RewTerm(func=mdp.RewardEefVel, weight=0.1, params={"k": 1.0, "side": "left"})
            joints_vel_left = RewTerm(func=mdp.RewardJointsVel, weight=0.1, params={"k": 1.0, "side": "left"})
            power_left = RewTerm(func=mdp.RewardPower, weight=0.2, params={"k": 10.0, "side": "left"})
            wrist_power_left = RewTerm(func=mdp.RewardWristPower, weight=0.15, params={"k": 2.0, "side": "left"})

    return RightHandTaskRewardsCfg


def _build_task_terminations_cfg(mdp, terminations_cfg_parent_cls, bimanual):
    """RightHandTaskTerminationsCfg: the parent hand-tracking terminations plus object tracking."""

    # ---------------------------------------------------------------------
    # Terminations
    # ---------------------------------------------------------------------
    @configclass
    class RightHandTaskTerminationsCfg(terminations_cfg_parent_cls):
        """Task terminations: hand-tracking terminations + object tracking."""

        # Object-tracking termination schedule (kept CONSISTENT with the
        # AUTHORITATIVE copy in cfg/residual_scene.py — that one OVERRIDES this for
        # the built TACO arm-residual cfg).
        # Effective threshold = start + (end - start) * schedule_progress, where
        # schedule_progress is driven 0->1 over the first ~50% of TRAINING by the
        # threshold_tightening curriculum (see curriculum cfg). So:
        #   TRAIN: translation and rotation anneal start -> end (cfg defaults here; launch.plan overrides them).
        #   EVAL/RENDER: schedule_progress is pinned to 1.0 (curriculum removed) ->
        #               fixed at the *_end values.
        # NOTE: pass --freeze_curriculum ONLY for eval/render. Freezing during TRAINING
        # pins progress=1.0 from iter 0 (the old 2 cm/30 deg fixed-tight bug that prevented
        # the policy from ever experiencing the lift phase) — do NOT freeze in training.
        object_tracking = DoneTerm(
            func=mdp.ObjectTrackingExceeded,
            time_out=False,
            params={
                "pos_start": 0.05,
                "pos_end": 0.03,
                "rot_start_deg": 120.0,
                "rot_end_deg": 90.0,
                "side": "right",
            },
        )

        # Bimanual Option 2: LEFT object-tracking termination — SAME thresholds
        # as the right, side='left', object_key='object_left'. Gated (default
        # OFF → never created, right baseline byte-identical). NOTE: the
        # arm-residual cfg OVERRIDES object_tracking via
        # ArmResidualTerminationsCfg (cfg/envs.py) which adds its
        # own object_tracking_left with the arm grace_period=20.
        if bimanual:
            object_tracking_left = DoneTerm(
                func=mdp.ObjectTrackingExceeded,
                time_out=False,
                params={
                    "pos_start": 0.05,
                    "pos_end": 0.03,
                    "rot_start_deg": 120.0,
                    "rot_end_deg": 90.0,
                    "side": "left",
                    "object_key": "object_left",
                },
            )

        # object_velocity termination DISABLED: unnecessary for this
        # task. Its thresholds (100 m/s, 200 rad/s) are sim-explosion guards, NOT the cause of
        # premature termination (that is object_tracking's rotation threshold). None = the
        # TerminationManager skips it cleanly in train/eval/render.
        object_velocity = None

    return RightHandTaskTerminationsCfg


def _ramp_event(func, **params):
    """An interval event over the gravity/friction ramp (ramp_steps first in its params)."""
    return EventTerm(
        func=func,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={"ramp_steps": _RAMP_STEPS_BUILD_DEFAULT, **params},
    )


def _reset_dr_event(func, **params):
    """A per-reset physics randomization of object_right (object_name first in its params)."""
    return EventTerm(func=func, mode="reset", params={"object_name": "object_right", **params})


def _build_task_event_cfg(mdp, event_cfg_parent_cls):
    """RightHandTaskEventCfg: reset pose, gravity and friction ramps, and the per-reset physics randomization."""

    @configclass
    class RightHandTaskEventCfg(event_cfg_parent_cls):
        """Task events: reset + domain randomization.

        Object gravity ramp: object has disable_gravity=True, receives ramped
        external force (0 → mg). Scene gravity stays at 9.81 so the hand
        always experiences full gravity (consistent with hand-tracking training).
        The per-reset physics randomization (physics_dr_*) is turned off at launch by
        train_cfg_overrides() when --enable_physics_dr=none.
        """

        randomize_pose = EventTerm(
            func=mdp.randomize_initial_pose,
            mode="reset",
            params={
                "randomize_fraction": 0.0,
                "finger_open_deg": 0.0,
                "wrist_z_max_cm": 0.0,
            },
        )
        gravity_ramp = _ramp_event(mdp.ramp_object_gravity, gravity_magnitude=GRAVITY_MPS2)
        friction_decay = _ramp_event(
            mdp.randomize_object_friction, nominal_friction=2.0, initial_multiplier=3.0, final_multiplier=1.0
        )
        physics_dr_mass = _reset_dr_event(mdp.randomize_object_mass_at_reset, mass_scale_range=(0.7, 1.3))
        physics_dr_friction = _reset_dr_event(
            mdp.randomize_object_friction_at_reset,
            static_friction_scale_range=(0.5, 1.5),
            dynamic_friction_scale_range=(0.5, 1.5),
        )
        physics_dr_restitution = _reset_dr_event(
            mdp.randomize_object_restitution_at_reset, restitution_scale_range=(0.5, 1.5)
        )
        physics_dr_com = _reset_dr_event(mdp.randomize_object_com_at_reset, com_offset_range_m=(-0.02, 0.02))

    return RightHandTaskEventCfg


def _build_task_curriculum_cfg(mdp, curriculum_cfg_parent_cls):
    """TaskCurriculumCfg: the parent curriculum plus the threshold-tightening schedule."""

    # ---------------------------------------------------------------------
    # Curriculum
    # ---------------------------------------------------------------------
    @configclass
    class TaskCurriculumCfg(curriculum_cfg_parent_cls):
        """Task curriculum: faster threshold tightening (40x faster than hand-only)."""

        threshold_tightening = CurrTerm(
            func=mdp.LinearThresholdSchedule,
            params={"anneal_steps": 1},  # patched at launch by rl/train/runner.py to int(0.5 * total_timesteps)
        )

    return TaskCurriculumCfg
