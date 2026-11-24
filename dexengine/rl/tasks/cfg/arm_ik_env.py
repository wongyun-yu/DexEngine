# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The arm-in-loop training env cfg class (TrainArmEnvCfg) and its __post_init__ settings.

The class body lists its fields in the order Isaac Lab registers the managers; the trajectory field and the
__post_init__ steps are built by the helpers below.
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from dexengine.rl.tasks import hooks


def _build_train_arm_env_cfg(
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
):
    """TrainArmEnvCfg: the arm-in-loop training env (full robot, IK-driven wrist, PD fingers)."""

    @configclass
    class TrainArmEnvCfg(ManagerBasedRLEnvCfg):
        """Arm-in-loop hand-tracking training configuration (base class).

        Full robot (the registered arm+hand combo) with IK-driven wrist + PD fingers; the same observations, rewards,
        terminations and curriculum as floating-hand training. Subclasses supply the ``actions`` cfg.
        """

        scene: scene_cfg_cls = scene_cfg_cls(num_envs=4096, env_spacing=2.5)  # type: ignore[valid-type]
        observations: observations_cfg_cls = observations_cfg_cls()  # type: ignore[valid-type]
        actions: ArmIKActionsCfg = ArmIKActionsCfg()
        rewards: rewards_cfg_cls = rewards_cfg_cls()  # type: ignore[valid-type]
        terminations: terminations_cfg_cls = terminations_cfg_cls()  # type: ignore[valid-type]
        events: ArmTrainEventCfg = ArmTrainEventCfg()
        curriculum: ArmCurriculumCfg = ArmCurriculumCfg()
        trajectory: trajectory_manager_cfg_cls = _trajectory_cfg(  # type: ignore[valid-type]
            trajectory_manager_cfg_cls,
            data_dir,
            desk_height,
            arm_ik_poses_dir_right,
            wrist_poses_dir_right,
            _traj_hand_kwargs(hand_spec),
        )

        def __post_init__(self):
            """Episode, physics and robot settings from the specs, then the head camera placement (create_env in
            rl/train/runner.py strips the camera for large training runs, once num_envs is final)."""
            _cam = _post_init_sim_settings(
                self,
                arm_spec,
                finger_armature,
                hand_spec,
                max_depenetration_velocity,
                mimic_armature,
                mimic_kd,
                mimic_kp,
                rigid_angular_damping,
                rigid_linear_damping,
            )
            _place_head_camera(self, _cam, _active_arm)
            _print_combo(self, _active_arm, hand_spec)

    return TrainArmEnvCfg


def _trajectory_cfg(
    trajectory_manager_cfg_cls, data_dir, desk_height, arm_ik_poses_dir_right, wrist_poses_dir_right, hand_kwargs
):
    """The trajectory manager cfg: the floating-hand training data plus the wrist poses for the arm IK.

    traj_offset_z = desk height keeps the wrist poses in the world frame (+z = desk surface): the IK action term
    reads get_wrist_pos_w() with floating_root_z_offset = desk height, so an offset of 0 would put the target below
    the floor with use_trajectory_reference=True. The eval env cfg uses the same convention.
    """
    return trajectory_manager_cfg_cls(
        data_dir=data_dir,
        sequence_filter="rh_main",
        rotate_to_x_forward=True,
        desk_height=desk_height,
        traj_offset_z=desk_height,
        urdf_hand=True,
        arm_ik_poses_dir_right=arm_ik_poses_dir_right,
        wrist_poses_dir_right=wrist_poses_dir_right,
        **hand_kwargs,
    )


def _print_combo(self, active_arm, hand_spec):
    """The robot combo this env spawns (arm, hand, USD)."""
    print(
        f"[TrainArm __post_init__] combo articulation: "
        f"arm={active_arm.name} "
        f"hand={hand_spec.name} "
        f"usd={self.scene.robot.spawn.usd_path}"
    )


def _place_head_camera(self, cam, active_arm):
    """The scene's head camera sits at an arm-neutral placeholder (Robot/HeadCamera). An arm whose spec carries its
    ZED prim path gets the camera there (it moves with the hardware); otherwise the camera is removed (spawning under
    a missing prim fails the scene with "Unable to find source prim path")."""
    if cam is None:
        return
    rel = active_arm.head_camera_prim_relpath
    if rel:
        leaf = cam.prim_path.rsplit("/", 1)[1]  # HeadCamera
        cam.prim_path = "{ENV_REGEX_NS}/Robot/" + f"{rel}/{leaf}"
        print(f"[TrainArm __post_init__] head_camera -> {cam.prim_path}")
    else:
        self.scene.head_camera = None
        print(f"[TrainArm __post_init__] head_camera removed (arm={active_arm.name} has no ZED prim path)")


def _post_init_sim_settings(
    self,
    arm_spec,
    finger_armature,
    hand_spec,
    max_depenetration_velocity,
    mimic_armature,
    mimic_kd,
    mimic_kp,
    rigid_angular_damping,
    rigid_linear_damping,
):
    """Episode settings, then the whole robot cfg from the specs; returns the scene's head camera cfg (or None).

    The robot cfg (USD via combo_usd_path, actuator groups, gains, rigid damping; scene/robot.py) keeps the prim
    path and base pose of the scene's robot placeholder, so train and eval scenes keep their own placement. The
    PhysX GPU buffer caps are written in one place, the residual cfg's __post_init__ (every gym-registered env
    subclasses it).
    """
    _post_init_episode_settings(self)

    from dexengine.rl.tasks.scene.robot import make_combo_articulation_cfg

    _prev_robot = self.scene.robot
    self.scene.robot = make_combo_articulation_cfg(
        arm_spec,
        hand_spec,
        prim_path=_prev_robot.prim_path,
        init_pos=tuple(_prev_robot.init_state.pos),
        init_rot=tuple(_prev_robot.init_state.rot),
        finger_armature=finger_armature,
        mimic_kp=mimic_kp,
        mimic_kd=mimic_kd,
        mimic_armature=mimic_armature,
        rigid_linear_damping=rigid_linear_damping,
        rigid_angular_damping=rigid_angular_damping,
        max_depenetration_velocity=max_depenetration_velocity,
    )
    _cam = getattr(self.scene, "head_camera", None)
    return _cam


def _post_init_episode_settings(self):
    """Decimation, episode length and NaN-reset settings.

    The policy runs at 50 Hz, the rate of the reference (0.02 s per frame): decimation 2 x dt 1/100 s, physics at
    100 Hz. Physics at 50 Hz (decimation 1) would halve the PhysX work but is a fidelity change for contact-rich
    grasps, so it needs a quality check before adoption.
    """
    self.decimation = 2
    self.episode_length_s = 30.0
    self.sim.dt = 0.02 / self.decimation  # always keeps the 0.02 s policy period
    self.sim.render_interval = self.decimation
    # DEXENGINE_FH_NAN_RESET=1 (default off, floating hand only): the termination term that cuts envs with a
    # non-finite reward on the next step
    if hooks.floating_nan_reset():
        from isaaclab.managers import TerminationTermCfg as _DoneTerm

        from dexengine.rl.tasks.mdp.terminations.hand import fh_nan_reset as _fh_nan_reset

        self.terminations.fh_nan_reset = _DoneTerm(func=_fh_nan_reset, time_out=False)
        print(
            f"{hooks.NAN_RESET_ON} envs with a non-finite reward (total or per side): that step's reward "
            "0, terminated next step (termination term "
            "fh_nan_reset)",
            flush=True,
        )
