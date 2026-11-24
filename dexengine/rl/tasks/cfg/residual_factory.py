# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dataset-agnostic task arm-in-loop (reference-base + residual) cfg factory.

Builds the dataset-specific cfg classes from a small parameter set (class names,
paths, parent classes, object mass, whether per-reset physics_dr events are
included). Object mass uses the single-source DEFAULT_OBJECT_MASS_KG (0.2 kg).

This module exposes ``build_arm_residual_cfg_classes(...)`` which constructs
the dataset-specific cfg classes from a parameter set. Returned classes:

    ArmResidualTerminationsCfg       (parent: caller's task terminations)
    ArmResidualSceneCfg         (parent: caller's eval scene)
    ArmResidualActionsCfg
    ArmResidualEventCfg         (parent: caller's ArmTrainEventCfg)
    <DatasetTrainArmResidualEnvCfg>  (parent: the dataset arm env cfg)
    <DatasetArmResidualBaseEnvCfg>
    <DatasetRHArmResidualEnvCfg>

Dataset packages call this factory at import time and re-export the
returned classes under their dataset-specific public names so that the gym
entry points and any external importers continue to resolve.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from dexengine.rl.tasks.cfg.residual_env import _residual_env_cfg
from dexengine.rl.tasks.cfg.residual_scene import _residual_scene_cfg, _residual_terminations_cfg
from dexengine.rl.tasks.cfg.residual_terms import _residual_actions_cfg, _residual_events


def build_arm_residual_cfg_classes(
    *,
    mdp,
    eval_scene_cfg_cls: type,
    train_arm_env_cfg_cls: type,
    task_observations_cfg_cls: type,
    task_rewards_cfg_cls: type,
    task_terminations_cfg_cls: type,
    task_curriculum_cfg_cls: type,
    arm_train_event_cfg_cls: type,
    max_depenetration_velocity: float,
    ground_friction: float,
    residual_scale_default: float,
    wrist_residual_scale_default: float,
    finger_residual_scale_default: float,
    action_ema_default: float,
    train_arm_residual_env_cfg_name: str,
    arm_residual_base_env_cfg_name: str,
    rh_arm_residual_env_cfg_name: str,
    object_mass_props=None,
    include_physics_dr_events: bool = True,
    hand_spec,
    arm_spec,
    bimanual: bool = False,
    left_uni: bool = False,
) -> dict[str, type]:
    """Build the dataset-specific task arm-in-loop cfg class set (returned by class name).

    ``mdp`` is the dataset's mdp package (tracking terminations, ResidualArmIKActionCfg, pose/gravity/friction
    events and, with ``include_physics_dr_events``, the mass/friction/restitution/com reset DR events, which
    ``ArmResidualEventCfg`` then carries as ``physics_dr_*`` fields). ``eval_scene_cfg_cls`` gives the full-robot
    scene base, ``train_arm_env_cfg_cls`` and ``arm_train_event_cfg_cls`` come from the ``cfg/arm_ik.py`` factory,
    the ``task_*_cfg_cls`` from ``cfg/task.build_task_mdp_cfg_classes``; the floats are ``cfg/constants.py`` values
    and the ``*_name`` strings the public class names. ``object_mass_props`` None means
    ``make_object_mass_props()`` (DEFAULT_OBJECT_MASS_KG for all objects). ``hand_spec`` / ``arm_spec`` are required
    registered specs: the residual action terms use the hand's finger joints, coupling and mount offset (left =
    ``mirror_right_to_left``), the contact sensors its body regex and fingertips, and the arm's joint/EE names and
    mount lookup; the robot articulation itself comes from the inherited ``__post_init__``. ``None`` raises.
    """
    return _left_presence_gates(
        mdp,
        eval_scene_cfg_cls,
        train_arm_env_cfg_cls,
        task_observations_cfg_cls,
        task_rewards_cfg_cls,
        task_terminations_cfg_cls,
        task_curriculum_cfg_cls,
        arm_train_event_cfg_cls,
        max_depenetration_velocity,
        ground_friction,
        residual_scale_default,
        wrist_residual_scale_default,
        finger_residual_scale_default,
        action_ema_default,
        train_arm_residual_env_cfg_name,
        arm_residual_base_env_cfg_name,
        rh_arm_residual_env_cfg_name,
        object_mass_props,
        include_physics_dr_events,
        hand_spec,
        arm_spec,
        bimanual,
        left_uni,
    )


def _left_presence_gates(
    mdp,
    eval_scene_cfg_cls,
    train_arm_env_cfg_cls,
    task_observations_cfg_cls,
    task_rewards_cfg_cls,
    task_terminations_cfg_cls,
    task_curriculum_cfg_cls,
    arm_train_event_cfg_cls,
    max_depenetration_velocity,
    ground_friction,
    residual_scale_default,
    wrist_residual_scale_default,
    finger_residual_scale_default,
    action_ema_default,
    train_arm_residual_env_cfg_name,
    arm_residual_base_env_cfg_name,
    rh_arm_residual_env_cfg_name,
    object_mass_props,
    include_physics_dr_events,
    hand_spec,
    arm_spec,
    bimanual,
    left_uni,
):
    """`left_present` gates the LEFT PHYSICAL PRESENCE (scene left object spawn, left object reset/physics-DR events,
    cross-hand selective self-collision + GPU buffer up-size). `bimanual` ALONE still gates the LEFT POLICY (left
    termination, left action term, left reward, left obs). So a because no left action term is registered) WITHOUT
    any left policy signal.  LEFT-UNI: `left_uni` builds the exact MIRROR of the right-only env — the LEFT hand is
    the SINGLE 12-dim policy, the LEFT object spawns under the single 'object' scene key, and the right arm/hand
    parks at default PD (event-driven, no right action term). left_present stays False → single object, self-
    collision OFF, default GPU buffers = the SAME physics regime as the right-uni build (byte-parity goal). Mutually
    exclusive with bimanual (a two-object build keyed on 'object_left').
    """
    ArmResidualTerminationsCfg, left_present, object_mass_props = _residual_terminations_cfg(
        bimanual=bimanual,
        left_uni=left_uni,
        mdp=mdp,
        object_mass_props=object_mass_props,
        task_terminations_cfg_cls=task_terminations_cfg_cls,
    )

    return _assemble_residual_cfg_set(
        mdp,
        eval_scene_cfg_cls,
        train_arm_env_cfg_cls,
        task_observations_cfg_cls,
        task_rewards_cfg_cls,
        task_curriculum_cfg_cls,
        arm_train_event_cfg_cls,
        max_depenetration_velocity,
        ground_friction,
        residual_scale_default,
        wrist_residual_scale_default,
        finger_residual_scale_default,
        action_ema_default,
        train_arm_residual_env_cfg_name,
        arm_residual_base_env_cfg_name,
        rh_arm_residual_env_cfg_name,
        object_mass_props,
        include_physics_dr_events,
        hand_spec,
        arm_spec,
        bimanual,
        left_uni,
        ArmResidualTerminationsCfg,
        left_present,
    )


def _assemble_residual_cfg_set(
    mdp,
    eval_scene_cfg_cls,
    train_arm_env_cfg_cls,
    task_observations_cfg_cls,
    task_rewards_cfg_cls,
    task_curriculum_cfg_cls,
    arm_train_event_cfg_cls,
    max_depenetration_velocity,
    ground_friction,
    residual_scale_default,
    wrist_residual_scale_default,
    finger_residual_scale_default,
    action_ema_default,
    train_arm_residual_env_cfg_name,
    arm_residual_base_env_cfg_name,
    rh_arm_residual_env_cfg_name,
    object_mass_props,
    include_physics_dr_events,
    hand_spec,
    arm_spec,
    bimanual,
    left_uni,
    ArmResidualTerminationsCfg,
    left_present,
):
    """Assemble the residual cfg classes by name."""
    (
        ArmResidualActionsCfg,
        ArmResidualBaseEnvCfg,
        ArmResidualEventCfg,
        ArmResidualSceneCfg,
        RHArmResidualEnvCfg,
        TrainArmResidualEnvCfg,
    ) = _build_scene_and_cfg_classes(
        mdp,
        eval_scene_cfg_cls,
        train_arm_env_cfg_cls,
        task_observations_cfg_cls,
        task_rewards_cfg_cls,
        task_curriculum_cfg_cls,
        arm_train_event_cfg_cls,
        max_depenetration_velocity,
        ground_friction,
        residual_scale_default,
        wrist_residual_scale_default,
        finger_residual_scale_default,
        action_ema_default,
        train_arm_residual_env_cfg_name,
        arm_residual_base_env_cfg_name,
        rh_arm_residual_env_cfg_name,
        object_mass_props,
        include_physics_dr_events,
        hand_spec,
        arm_spec,
        bimanual,
        left_uni,
        ArmResidualTerminationsCfg,
        left_present,
    )

    return {
        "ArmResidualTerminationsCfg": ArmResidualTerminationsCfg,
        "ArmResidualSceneCfg": ArmResidualSceneCfg,
        "ArmResidualActionsCfg": ArmResidualActionsCfg,
        "ArmResidualEventCfg": ArmResidualEventCfg,
        train_arm_residual_env_cfg_name: TrainArmResidualEnvCfg,
        arm_residual_base_env_cfg_name: ArmResidualBaseEnvCfg,
        rh_arm_residual_env_cfg_name: RHArmResidualEnvCfg,
    }


def _build_scene_and_cfg_classes(
    mdp,
    eval_scene_cfg_cls,
    train_arm_env_cfg_cls,
    task_observations_cfg_cls,
    task_rewards_cfg_cls,
    task_curriculum_cfg_cls,
    arm_train_event_cfg_cls,
    max_depenetration_velocity,
    ground_friction,
    residual_scale_default,
    wrist_residual_scale_default,
    finger_residual_scale_default,
    action_ema_default,
    train_arm_residual_env_cfg_name,
    arm_residual_base_env_cfg_name,
    rh_arm_residual_env_cfg_name,
    object_mass_props,
    include_physics_dr_events,
    hand_spec,
    arm_spec,
    bimanual,
    left_uni,
    ArmResidualTerminationsCfg,
    left_present,
):
    """Scene cfg and the residual cfg classes for the active arm side."""
    ArmResidualSceneCfg, _active_arm, _to_left = _residual_scene_cfg(
        arm_spec=arm_spec,
        eval_scene_cfg_cls=eval_scene_cfg_cls,
        ground_friction=ground_friction,
        hand_spec=hand_spec,
        left_present=left_present,
        left_uni=left_uni,
        max_depenetration_velocity=max_depenetration_velocity,
        object_mass_props=object_mass_props,
    )

    ArmResidualActionsCfg, ArmResidualBaseEnvCfg, ArmResidualEventCfg, RHArmResidualEnvCfg, TrainArmResidualEnvCfg = (
        _build_residual_cfg_classes(
            mdp,
            train_arm_env_cfg_cls,
            task_observations_cfg_cls,
            task_rewards_cfg_cls,
            task_curriculum_cfg_cls,
            arm_train_event_cfg_cls,
            max_depenetration_velocity,
            residual_scale_default,
            wrist_residual_scale_default,
            finger_residual_scale_default,
            action_ema_default,
            train_arm_residual_env_cfg_name,
            arm_residual_base_env_cfg_name,
            rh_arm_residual_env_cfg_name,
            include_physics_dr_events,
            hand_spec,
            bimanual,
            left_uni,
            ArmResidualSceneCfg,
            ArmResidualTerminationsCfg,
            _active_arm,
            _to_left,
            left_present,
        )
    )
    return (
        ArmResidualActionsCfg,
        ArmResidualBaseEnvCfg,
        ArmResidualEventCfg,
        ArmResidualSceneCfg,
        RHArmResidualEnvCfg,
        TrainArmResidualEnvCfg,
    )


def _build_residual_cfg_classes(
    mdp,
    train_arm_env_cfg_cls,
    task_observations_cfg_cls,
    task_rewards_cfg_cls,
    task_curriculum_cfg_cls,
    arm_train_event_cfg_cls,
    max_depenetration_velocity,
    residual_scale_default,
    wrist_residual_scale_default,
    finger_residual_scale_default,
    action_ema_default,
    train_arm_residual_env_cfg_name,
    arm_residual_base_env_cfg_name,
    rh_arm_residual_env_cfg_name,
    include_physics_dr_events,
    hand_spec,
    bimanual,
    left_uni,
    ArmResidualSceneCfg,
    ArmResidualTerminationsCfg,
    _active_arm,
    _to_left,
    left_present,
):
    """Residual actions Base kwargs that match both datasets and BOTH hands exactly. Both hands come from their
    HandSpec (finger joints, link7 offset); the left one is the mirrored spec with side="left". Every
    shared scalar (scales, smoothing, ik lambda, checkpoint placeholder) is identical across hands, so one side-
    parameterised builder reproduces both dicts byte-for-byte (right omits `side` -> cfg default; left passes
    side="left"). The inner ArmIKFingersPositionAction still derives side from finger_joint_names[0].
    """
    (ArmResidualActionsCfg,) = _residual_actions_cfg(
        _active_arm=_active_arm,
        _to_left=_to_left,
        action_ema_default=action_ema_default,
        bimanual=bimanual,
        finger_residual_scale_default=finger_residual_scale_default,
        hand_spec=hand_spec,
        left_uni=left_uni,
        mdp=mdp,
        residual_scale_default=residual_scale_default,
        wrist_residual_scale_default=wrist_residual_scale_default,
    )

    return _build_event_and_env_cfgs(
        mdp,
        train_arm_env_cfg_cls,
        task_observations_cfg_cls,
        task_rewards_cfg_cls,
        task_curriculum_cfg_cls,
        arm_train_event_cfg_cls,
        max_depenetration_velocity,
        train_arm_residual_env_cfg_name,
        arm_residual_base_env_cfg_name,
        rh_arm_residual_env_cfg_name,
        include_physics_dr_events,
        left_uni,
        ArmResidualSceneCfg,
        ArmResidualTerminationsCfg,
        left_present,
        ArmResidualActionsCfg,
    )


def _build_event_and_env_cfgs(
    mdp,
    train_arm_env_cfg_cls,
    task_observations_cfg_cls,
    task_rewards_cfg_cls,
    task_curriculum_cfg_cls,
    arm_train_event_cfg_cls,
    max_depenetration_velocity,
    train_arm_residual_env_cfg_name,
    arm_residual_base_env_cfg_name,
    rh_arm_residual_env_cfg_name,
    include_physics_dr_events,
    left_uni,
    ArmResidualSceneCfg,
    ArmResidualTerminationsCfg,
    left_present,
    ArmResidualActionsCfg,
):
    """Event and env cfg classes of the residual set."""
    (ArmResidualEventCfg,) = _residual_events(
        arm_train_event_cfg_cls=arm_train_event_cfg_cls,
        include_physics_dr_events=include_physics_dr_events,
        left_present=left_present,
        left_uni=left_uni,
        mdp=mdp,
    )

    # ---------------------------------------------------------------------
    # Env cfgs
    # ---------------------------------------------------------------------
    (TrainArmResidualEnvCfg,) = _residual_env_cfg(
        ArmResidualActionsCfg=ArmResidualActionsCfg,
        ArmResidualEventCfg=ArmResidualEventCfg,
        ArmResidualSceneCfg=ArmResidualSceneCfg,
        ArmResidualTerminationsCfg=ArmResidualTerminationsCfg,
        left_present=left_present,
        max_depenetration_velocity=max_depenetration_velocity,
        task_curriculum_cfg_cls=task_curriculum_cfg_cls,
        task_observations_cfg_cls=task_observations_cfg_cls,
        task_rewards_cfg_cls=task_rewards_cfg_cls,
        train_arm_env_cfg_cls=train_arm_env_cfg_cls,
    )

    TrainArmResidualEnvCfg.__name__ = train_arm_residual_env_cfg_name
    TrainArmResidualEnvCfg.__qualname__ = train_arm_residual_env_cfg_name

    ArmResidualBaseEnvCfg = _build_arm_residual_base_env_cfg(ArmResidualActionsCfg, TrainArmResidualEnvCfg)

    ArmResidualBaseEnvCfg.__name__ = arm_residual_base_env_cfg_name
    ArmResidualBaseEnvCfg.__qualname__ = arm_residual_base_env_cfg_name

    RHArmResidualEnvCfg = _build_rh_arm_residual_env_cfg(ArmResidualBaseEnvCfg)

    RHArmResidualEnvCfg.__name__ = rh_arm_residual_env_cfg_name
    RHArmResidualEnvCfg.__qualname__ = rh_arm_residual_env_cfg_name
    return (
        ArmResidualActionsCfg,
        ArmResidualBaseEnvCfg,
        ArmResidualEventCfg,
        RHArmResidualEnvCfg,
        TrainArmResidualEnvCfg,
    )


def _build_arm_residual_base_env_cfg(ArmResidualActionsCfg, TrainArmResidualEnvCfg):
    """ArmResidualBaseEnvCfg: the residual training env with the residual actions."""

    @configclass
    class ArmResidualBaseEnvCfg(TrainArmResidualEnvCfg):
        """Arm residual env config: matches the hand-tracking arm-IK action settings.

        Uses dt=1/100, decimation=2 (inherited from the base cfg) for stability
        at 4096 envs. Arm Kp/Kd come from the arm spec (ArmSpec.arm_drive) via the
        parent {Dataset}TrainArmEnvCfg.__post_init__ (no local override).
        """

        actions: ArmResidualActionsCfg = ArmResidualActionsCfg()

        def __post_init__(self):
            super().__post_init__()
            self.trajectory.ik_error_threshold = 0.04

    return ArmResidualBaseEnvCfg


def _build_rh_arm_residual_env_cfg(ArmResidualBaseEnvCfg):
    """RHArmResidualEnvCfg: the right-hand residual training env."""

    @configclass
    class RHArmResidualEnvCfg(ArmResidualBaseEnvCfg):
        """Right-hand arm residual training.

        Registered as DexEngine-ArmResidual-RightUni-v0 (tasks/registry.py).
        """

    return RHArmResidualEnvCfg
