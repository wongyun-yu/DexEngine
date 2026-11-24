# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared arm-in-loop hand-tracking (ArmIK) config factory.

Dataset-agnostic ArmIK env-cfg factory: ``build_arm_ik_cfg_classes`` builds the
dataset-specific cfg classes from a small parameter set (class names, paths,
parent classes), so a new dataset reuses the same construction logic.

The returned classes are::

    ArmCurriculumCfg            -- RSI disabled (subclass of caller's curriculum)
    ArmTaskCurriculumCfg      -- ArmCurriculumCfg + threshold tightening
    ArmIKActionsCfg       -- ArmIK action term (pure, no dataset deps)
    ArmTrainEventCfg            -- reset events + reset_arm (subclass of caller's events)
    TrainArmEnvCfg              -- the base ManagerBasedRLEnvCfg

Dataset packages call this factory at import time and re-export the
returned classes under their dataset-specific public names so that the
arm-residual cfgs (which reference
``ArmTrainEventCfg``, ``ArmTaskCurriculumCfg``, ``ArmIKActionsCfg``)
continues to resolve the same identities.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from dexengine.core.mounts import mount_for as _mount_for
from dexengine.rl.tasks import hooks
from dexengine.rl.tasks.cfg.arm_ik_env import _build_train_arm_env_cfg


def build_arm_ik_cfg_classes(
    *,
    mdp,
    trajectory_manager_cfg_cls: type,
    scene_cfg_cls: type,
    observations_cfg_cls: type,
    rewards_cfg_cls: type,
    terminations_cfg_cls: type,
    event_cfg_parent_cls: type,
    curriculum_cfg_parent_cls: type,
    data_dir: str,
    arm_ik_poses_dir_right: str,
    wrist_poses_dir_right: str | None = None,
    desk_height: float,
    rigid_linear_damping: float,
    rigid_angular_damping: float,
    max_depenetration_velocity: float,
    finger_armature: float,
    mimic_kp: float,
    mimic_kd: float,
    mimic_armature: float,
    train_arm_env_cfg_name: str,
    hand_spec,
    arm_spec,
) -> dict[str, type]:
    """Build the dataset-specific arm-IK cfg class set.

    ``mdp`` is the dataset's mdp package (RsiSchedule, LinearThresholdSchedule, ArmIKFingersPositionActionCfg,
    reset_arm_to_trajectory); ``*_cls`` are its cfg classes (cfg/scene.py, cfg/task.py); ``data_dir`` and
    ``arm_ik_poses_dir_right`` are its paths; the floats are the physics/control constants of ``cfg/constants.py``
    (finger PD gains are the hand spec's own, ``HandSpec.finger_kp/kd``). ``hand_spec`` / ``arm_spec`` are required
    registered specs: the action term and the reset event take the arm's joint/EE names from the spec, the mount
    offset is looked up per (arm, hand), and ``__post_init__`` replaces ``scene.robot`` with
    ``scene/robot.make_combo_articulation_cfg`` (USD via ``combo_usd_path``, both hands mirrored from the hand spec,
    actuator groups and gains from the specs); ``None`` raises.

    Returns the classes by name: ``ArmCurriculumCfg``, ``ArmTaskCurriculumCfg``, ``ArmIKActionsCfg``,
    ``ArmTrainEventCfg`` and the base env cfg class named ``train_arm_env_cfg_name``.
    """
    ArmCurriculumCfg, ArmTaskCurriculumCfg, _active_arm, _traj_hand_kwargs = _resolve_arm_and_hand(
        mdp, curriculum_cfg_parent_cls, hand_spec, arm_spec
    )

    return _hand_action_term_kwargs(
        mdp,
        trajectory_manager_cfg_cls,
        scene_cfg_cls,
        observations_cfg_cls,
        rewards_cfg_cls,
        terminations_cfg_cls,
        event_cfg_parent_cls,
        data_dir,
        arm_ik_poses_dir_right,
        wrist_poses_dir_right,
        desk_height,
        rigid_linear_damping,
        rigid_angular_damping,
        max_depenetration_velocity,
        finger_armature,
        mimic_kp,
        mimic_kd,
        mimic_armature,
        train_arm_env_cfg_name,
        hand_spec,
        arm_spec,
        ArmCurriculumCfg,
        ArmTaskCurriculumCfg,
        _active_arm,
        _traj_hand_kwargs,
    )


def _hand_action_term_kwargs(
    mdp,
    trajectory_manager_cfg_cls,
    scene_cfg_cls,
    observations_cfg_cls,
    rewards_cfg_cls,
    terminations_cfg_cls,
    event_cfg_parent_cls,
    data_dir,
    arm_ik_poses_dir_right,
    wrist_poses_dir_right,
    desk_height,
    rigid_linear_damping,
    rigid_angular_damping,
    max_depenetration_velocity,
    finger_armature,
    mimic_kp,
    mimic_kd,
    mimic_armature,
    train_arm_env_cfg_name,
    hand_spec,
    arm_spec,
    ArmCurriculumCfg,
    ArmTaskCurriculumCfg,
    _active_arm,
    _traj_hand_kwargs,
):
    """Hand-dependent action-term kwargs, from the hand spec."""
    ArmIKActionsCfg = _build_arm_ik_actions(mdp, hand_spec, _active_arm)

    return _build_arm_cfg_set(
        mdp,
        trajectory_manager_cfg_cls,
        scene_cfg_cls,
        observations_cfg_cls,
        rewards_cfg_cls,
        terminations_cfg_cls,
        event_cfg_parent_cls,
        data_dir,
        arm_ik_poses_dir_right,
        wrist_poses_dir_right,
        desk_height,
        rigid_linear_damping,
        rigid_angular_damping,
        max_depenetration_velocity,
        finger_armature,
        mimic_kp,
        mimic_kd,
        mimic_armature,
        train_arm_env_cfg_name,
        hand_spec,
        arm_spec,
        ArmCurriculumCfg,
        ArmTaskCurriculumCfg,
        _active_arm,
        _traj_hand_kwargs,
        ArmIKActionsCfg,
    )


def _build_arm_cfg_set(
    mdp,
    trajectory_manager_cfg_cls,
    scene_cfg_cls,
    observations_cfg_cls,
    rewards_cfg_cls,
    terminations_cfg_cls,
    event_cfg_parent_cls,
    data_dir,
    arm_ik_poses_dir_right,
    wrist_poses_dir_right,
    desk_height,
    rigid_linear_damping,
    rigid_angular_damping,
    max_depenetration_velocity,
    finger_armature,
    mimic_kp,
    mimic_kd,
    mimic_armature,
    train_arm_env_cfg_name,
    hand_spec,
    arm_spec,
    ArmCurriculumCfg,
    ArmTaskCurriculumCfg,
    _active_arm,
    _traj_hand_kwargs,
    ArmIKActionsCfg,
):
    """Event, env and curriculum classes; returns the class set by name."""
    ArmTrainEventCfg = _build_arm_train_event_cfg(mdp, event_cfg_parent_cls, _active_arm)

    TrainArmEnvCfg = _build_train_arm_env_cfg(
        trajectory_manager_cfg_cls,
        scene_cfg_cls,
        observations_cfg_cls,
        rewards_cfg_cls,
        terminations_cfg_cls,
        data_dir,
        arm_ik_poses_dir_right,
        wrist_poses_dir_right,
        desk_height,
        rigid_linear_damping,
        rigid_angular_damping,
        max_depenetration_velocity,
        finger_armature,
        mimic_kp,
        mimic_kd,
        mimic_armature,
        hand_spec,
        arm_spec,
        ArmCurriculumCfg,
        ArmIKActionsCfg,
        ArmTrainEventCfg,
        _active_arm,
        _traj_hand_kwargs,
    )

    # Rename TrainArmEnvCfg to the dataset-specific name for nicer reprs.
    TrainArmEnvCfg.__name__ = train_arm_env_cfg_name
    TrainArmEnvCfg.__qualname__ = train_arm_env_cfg_name

    return {
        "ArmCurriculumCfg": ArmCurriculumCfg,
        "ArmTaskCurriculumCfg": ArmTaskCurriculumCfg,
        "ArmIKActionsCfg": ArmIKActionsCfg,
        "ArmTrainEventCfg": ArmTrainEventCfg,
        train_arm_env_cfg_name: TrainArmEnvCfg,
    }


def _build_arm_ik_actions(mdp, hand_spec, _active_arm):
    """The arm-IK action cfg class from the hand and arm specs."""
    _arm_ik_action_kwargs = {
        "finger_joint_names": list(hand_spec.finger_joint_names_right),
        "coupled_joint_names": list(hand_spec.coupled_joint_names_right),
        "coupling_ratios": list(hand_spec.coupling_ratios) or None,
        "coupling_source_idx": list(hand_spec.coupling_source_idx) or None,
        # Mount geometry is a property of the (arm, hand) PAIR, so it comes
        # from MOUNT_REGISTRY rather than off the hand spec. A hand with no
        # measured adapter raises here instead of silently mounting at zero.
        "link7_to_wrist_offset": tuple(_mount_for(hand_spec.name, _active_arm.name).ee_to_wrist_offset_for("right")),
    }

    ArmIKActionsCfg = _build_arm_ik_actions_cfg(mdp, _active_arm, _arm_ik_action_kwargs)
    return ArmIKActionsCfg


def _resolve_arm_and_hand(mdp, curriculum_cfg_parent_cls, hand_spec, arm_spec):
    """Arm and hand are identity inputs: no silent default."""
    if arm_spec is None or hand_spec is None:
        raise ValueError(
            f"build_arm_ik_cfg_classes: arm_spec={arm_spec!r} hand_spec={hand_spec!r} — both must be registered "
            "specs (no default; the legacy pair literal path is gone)."
        )

    ArmCurriculumCfg = _build_arm_curriculum_cfg(mdp, curriculum_cfg_parent_cls)

    @configclass
    class ArmTaskCurriculumCfg(ArmCurriculumCfg):
        """Arm-involved task curriculum: RSI disabled, threshold tightening."""

        threshold_tightening = CurrTerm(
            func=mdp.LinearThresholdSchedule,
            params={"anneal_steps": 1},  # patched at launch by rl/train/runner.py to int(0.5 * total_timesteps)
        )

    # ---------------------------------------------------------------------
    # Action: arm-IK (pure, no dataset deps).
    # ---------------------------------------------------------------------
    # Active arm: the spec swaps naming, limits and actuator-group identity in one move.
    _active_arm = arm_spec

    # Per-hand facts for the trajectory manager (from the hand spec).
    def _traj_hand_kwargs(spec):
        # The palm->EE merge link is a fact of the **arm** only (rby1's EE is link_right_arm_6).
        _palm = {
            "fk_palm_isaac_body_right": _active_arm.ee_body_name("right"),
            "fk_palm_isaac_body_left": _active_arm.ee_body_name("left"),
        }
        # The finger order of the dataset qpos = the file order of that hand's URDF. For a hand without coupled
        # (mimic) joints finger_joint_names_right is the whole order. For a coupled hand the mixed order of independent
        # and dependent joints is unknown, so it is refused instead of guessed: a wrong order would shift every finger
        # column during training.
        if spec.coupled_joint_names_right:
            raise NotImplementedError(
                f"HandSpec({spec.name!r}) has {len(spec.coupled_joint_names_right)} coupled joints, but the "
                f"spec carries no **mixed order** of independent/dependent joints for the dataset qpos. Add a "
                "field with every finger joint name in that hand's URDF file order to the spec and pass it here."
            )
        # The palm (hand base) hangs off the arm EE through a **fixed joint**, so the Isaac importer merges it into
        # the EE link and its name disappears. The FK body list still has it, so it is linked by an alias.
        _palm_r = spec.palm_fk_body_right
        _aliases = (
            tuple(
                (_palm_r if side == "right" else _palm_r.replace("right_", "left_", 1), _active_arm.ee_body_name(side))
                for side in ("right", "left")
            )
            if _palm_r
            else ()
        )
        return dict(
            hand_name=spec.name,
            hand_qpos_dim=spec.hand_qpos_dim,
            dataset_finger_joint_names=tuple(spec.finger_joint_names_right),
            dataset_coupled_finger_names=(),
            fk_body_aliases=_aliases,
            fk_tip_body_names=tuple(spec.tip_body_names_right),
            **_palm,
        )

    return ArmCurriculumCfg, ArmTaskCurriculumCfg, _active_arm, _traj_hand_kwargs


def _build_arm_curriculum_cfg(mdp, curriculum_cfg_parent_cls):
    """ArmCurriculumCfg: the parent curriculum with the RSI schedule, off by default."""

    # ---------------------------------------------------------------------
    # Curriculum: RSI disabled for the arm-in-loop pipeline.
    # ---------------------------------------------------------------------
    @configclass
    class ArmCurriculumCfg(curriculum_cfg_parent_cls):
        """Arm-involved curriculum: RSI OFF by default.

        Default target_fraction=0.0 -> always start from frame 0 (the historical
        arm-pipeline behavior). When the all-frame arm-IK precompute exists
        (arm_joint_positions_all; the arm IK stage, all frames), a
        mid-trajectory RSI reset places the arm on the reference at the sampled
        start_frame (events.reset_arm_to_trajectory uses the all-frame cache), so
        RSI is now consistent. The --rsi_* flags (rl/train/runner.py) patch these params to
        enable RSI / a reverse-curriculum band for the RSI experiments.
        """

        rsi_schedule = CurrTerm(
            func=mdp.RsiSchedule,
            params={
                "ramp_episodes": 1,
                "target_fraction": 0.0,  # OFF by default; patched by the --rsi_* flags (rl/train/runner.py)
            },
        )

    return ArmCurriculumCfg


def _build_arm_ik_actions_cfg(mdp, _active_arm, _arm_ik_action_kwargs):
    """ArmIKActionsCfg: the IK-driven arm and the PD fingers of the active arm."""

    @configclass
    class ArmIKActionsCfg:
        """Arm IK action config: reduced action scales, no smoothing.

        wrist_pos_scale=0.02 (±10cm range), wrist_rot_scale=0.1 (±0.5rad).
        Temporal ensembling and EMA disabled — policy outputs raw actions.
        """

        right_hand_joint_pos = mdp.ArmIKFingersPositionActionCfg(
            asset_name="robot",
            arm_joint_names=_active_arm.arm_joint_exprs("right"),  # floating hand = [] (no arm joints)
            ee_body_name=_active_arm.ee_body_name("right"),
            ik_lambda_val=0.05,
            wrist_pos_scale=0.02,
            wrist_rot_scale=0.1,
            use_trajectory_reference=False,
            wrist_smooth_weight=-1.0,
            finger_smooth_weight=-1.0,
            **_arm_ik_action_kwargs,
        )

    return ArmIKActionsCfg


def _build_arm_train_event_cfg(mdp, event_cfg_parent_cls, _active_arm):
    """ArmTrainEventCfg: the dataset reset events plus the arm reset."""

    # ---------------------------------------------------------------------
    # Events: dataset reset events + reset_arm.
    # ---------------------------------------------------------------------
    @configclass
    class ArmTrainEventCfg(event_cfg_parent_cls):
        """Reset events for arm training — includes arm IK reset + hand trajectory reset."""

        # Arm IK reset: solve IK to place wrist at trajectory frame-0 pose.
        # The arm spec's names go through cfg params — explicit dataflow, no
        # hidden module state inside mdp/events.
        # Floating hand: in place of the arm IK reset, a 'root pose = reference wrist' reset
        # (floating_hand.reset_floating_hands_to_trajectory).
        reset_arm = (
            EventTerm(
                func=hooks.floating_hand().reset_floating_hands_to_trajectory,
                mode="reset",
            )
            if _active_arm.is_floating
            else EventTerm(
                func=mdp.reset_arm_to_trajectory,
                mode="reset",
                params={
                    "arm_joint_names": [_active_arm.joint_expr("right")],
                    "ee_body_name": _active_arm.ee_body_name("right"),
                    "left_arm_joint_expr": _active_arm.joint_expr("left"),
                },
            )
        )

    return ArmTrainEventCfg
