"""The three registered env cfgs (right-uni, left-uni, bimanual), built from the cfg factories.

Chain: scene (`cfg.scene`) -> hand tracking (`cfg.hand_tracking`) -> arm IK (`cfg.arm_ik`) -> task terms
(`cfg.task`) -> reference base + residual (`cfg.residual`). The robot is the registered arm + hand combo
selected by `DEXENGINE_ARM` / `DEXENGINE_HAND` (resolved once, at import). The policy outputs a wrist-pose
residual (6) plus finger residuals; the wrist target becomes arm joint targets through differential IK.

Variants:
  - right-uni: one policy, right arm + hand, right object.
  - bimanual: separate right and left residual action terms trained jointly (summed reward); the task cfgs are
    built with the left observation / reward / termination terms appended.
  - left-uni: the exact mirror of right-uni: one left policy (`left_hand_joint_pos`), the left object under the
    single `object` key, self-collision off. Its rewards/terminations reuse the right-side classes; their
    `side` params are flipped at runtime by `env_setup.sides.apply_left_uni`.
"""

from __future__ import annotations

from dexengine.rl.tasks import env_flags
from dexengine.rl.tasks.cfg.arm_ik import build_arm_ik_cfg_classes
from dexengine.rl.tasks.cfg.constants import (
    ACTION_EMA,
    DESK_HEIGHT,
    FINGER_ARMATURE,
    FINGER_RESIDUAL_SCALE,
    GROUND_FRICTION,
    MAX_DEPENETRATION_VELOCITY,
    MIMIC_ARMATURE,
    MIMIC_KD,
    MIMIC_KP,
    RESIDUAL_SCALE,
    RIGID_ANGULAR_DAMPING,
    RIGID_LINEAR_DAMPING,
    WRIST_RESIDUAL_SCALE,
)
from dexengine.rl.tasks.cfg.hand_tracking import (
    RightHandEventCfg,
    RightHandObservationsCfg,
    RightHandRewardsCfg,
    RightHandTerminationsCfg,
    TrackingCurriculumCfg,
)
from dexengine.rl.tasks.cfg.residual_factory import build_arm_residual_cfg_classes
from dexengine.rl.tasks.cfg.scene import TacoEvalSceneCfg
from dexengine.rl.tasks.cfg.task import build_task_mdp_cfg_classes
from dexengine.rl.tasks.mdp import terms as mdp
from dexengine.rl.tasks.mdp.terms import TrajectoryManagerCfg
from dexengine.robots.selection import combo_data_paths, resolve_robot_selection

# ---------------------------------------------------------------- robot selection
# Both env vars are required and must name registered entries; resolve_robot_selection() raises otherwise.
_ARM_SPEC, _HAND_SPEC = resolve_robot_selection()
# Always printed: the run log must show which robot the cfgs were built for.
print(
    f"[robot_selection] DEXENGINE_ARM={env_flags.get('DEXENGINE_ARM')!r} "
    f"DEXENGINE_HAND={env_flags.get('DEXENGINE_HAND')!r} -> "
    f"arm_spec={getattr(_ARM_SPEC, 'name', None)} hand_spec={getattr(_HAND_SPEC, 'name', None)}",
    flush=True,
)
# One rule maps the combo to its three data dirs (trajectories / wrist are keyed by hand, arm IK by the pair).
# The left dirs are derived at runtime by the `_right` -> `_left` suffix swap (events.reset._load_arm_ik_cache).
_COMBO_PATHS = combo_data_paths(_ARM_SPEC, _HAND_SPEC)
_DATA_DIR = _COMBO_PATHS["data_dir"]
_ARM_IK_DIR_R = _COMBO_PATHS["arm_ik_poses_dir_right"]
_WRIST_DIR_R = _COMBO_PATHS["wrist_poses_dir_right"]

# ---------------------------------------------------------------- arm IK (hand tracking) cfg
_arm_ik_classes = build_arm_ik_cfg_classes(
    hand_spec=_HAND_SPEC,
    arm_spec=_ARM_SPEC,
    mdp=mdp,
    trajectory_manager_cfg_cls=TrajectoryManagerCfg,
    scene_cfg_cls=TacoEvalSceneCfg,
    observations_cfg_cls=RightHandObservationsCfg,
    rewards_cfg_cls=RightHandRewardsCfg,
    terminations_cfg_cls=RightHandTerminationsCfg,
    event_cfg_parent_cls=RightHandEventCfg,
    curriculum_cfg_parent_cls=TrackingCurriculumCfg,
    data_dir=_DATA_DIR,
    arm_ik_poses_dir_right=_ARM_IK_DIR_R,
    wrist_poses_dir_right=_WRIST_DIR_R,
    desk_height=DESK_HEIGHT,
    rigid_linear_damping=RIGID_LINEAR_DAMPING,
    rigid_angular_damping=RIGID_ANGULAR_DAMPING,
    max_depenetration_velocity=MAX_DEPENETRATION_VELOCITY,
    finger_armature=FINGER_ARMATURE,
    mimic_kp=MIMIC_KP,
    mimic_kd=MIMIC_KD,
    mimic_armature=MIMIC_ARMATURE,
    train_arm_env_cfg_name="TacoTrainArmEnvCfg",
)
ArmCurriculumCfg = _arm_ik_classes["ArmCurriculumCfg"]
ArmTaskCurriculumCfg = _arm_ik_classes["ArmTaskCurriculumCfg"]
ArmTrainEventCfg = _arm_ik_classes["ArmTrainEventCfg"]
TacoTrainArmEnvCfg = _arm_ik_classes["TacoTrainArmEnvCfg"]

# ---------------------------------------------------------------- task MDP cfgs (observations, rewards, ...)
_TASK_PARENTS = {
    "mdp": mdp,
    "rewards_cfg_parent_cls": RightHandRewardsCfg,
    "terminations_cfg_parent_cls": RightHandTerminationsCfg,
    "event_cfg_parent_cls": RightHandEventCfg,
    "curriculum_cfg_parent_cls": TrackingCurriculumCfg,
}
_shared_classes = build_task_mdp_cfg_classes(**_TASK_PARENTS)
RightHandTaskObservationsCfg = _shared_classes["RightHandTaskObservationsCfg"]
RightHandTaskRewardsCfg = _shared_classes["RightHandTaskRewardsCfg"]
RightHandTaskTerminationsCfg = _shared_classes["RightHandTaskTerminationsCfg"]
RightHandTaskEventCfg = _shared_classes["RightHandTaskEventCfg"]
TaskCurriculumCfg = _shared_classes["TaskCurriculumCfg"]

_bimanual_shared_classes = build_task_mdp_cfg_classes(**_TASK_PARENTS, bimanual=True)
BimanualTaskObservationsCfg = _bimanual_shared_classes["RightHandTaskObservationsCfg"]
BimanualTaskRewardsCfg = _bimanual_shared_classes["RightHandTaskRewardsCfg"]
BimanualTaskTerminationsCfg = _bimanual_shared_classes["RightHandTaskTerminationsCfg"]

# Left-uni: only the observations differ (left mirror under the unsuffixed names).
_leftuni_shared_classes = build_task_mdp_cfg_classes(**_TASK_PARENTS, left_uni=True)
LeftUniTaskObservationsCfg = _leftuni_shared_classes["RightHandTaskObservationsCfg"]

# ---------------------------------------------------------------- reference base + residual cfgs
# Shared factory arguments. The object mass is the factory default (DEFAULT_OBJECT_MASS_KG). The four per-reset
# physics randomization events always exist; they are filled only when a launch passes --enable_physics_dr and are
# removed for evaluation.
_RESIDUAL_COMMON = {
    "mdp": mdp,
    "hand_spec": _HAND_SPEC,
    "arm_spec": _ARM_SPEC,
    "eval_scene_cfg_cls": TacoEvalSceneCfg,
    "train_arm_env_cfg_cls": TacoTrainArmEnvCfg,
    "task_curriculum_cfg_cls": ArmTaskCurriculumCfg,
    "arm_train_event_cfg_cls": ArmTrainEventCfg,
    "max_depenetration_velocity": MAX_DEPENETRATION_VELOCITY,
    "ground_friction": GROUND_FRICTION,
    "residual_scale_default": RESIDUAL_SCALE,
    "wrist_residual_scale_default": WRIST_RESIDUAL_SCALE,
    "finger_residual_scale_default": FINGER_RESIDUAL_SCALE,
    "action_ema_default": ACTION_EMA,
    "include_physics_dr_events": True,
}

_arm_residual_classes = build_arm_residual_cfg_classes(
    **_RESIDUAL_COMMON,
    task_observations_cfg_cls=RightHandTaskObservationsCfg,
    task_rewards_cfg_cls=RightHandTaskRewardsCfg,
    task_terminations_cfg_cls=RightHandTaskTerminationsCfg,
    train_arm_residual_env_cfg_name="TacoTrainArmResidualEnvCfg",
    arm_residual_base_env_cfg_name="TacoArmResidualBaseEnvCfg",
    rh_arm_residual_env_cfg_name="TacoRHArmResidualEnvCfg",
)
ArmResidualTerminationsCfg = _arm_residual_classes["ArmResidualTerminationsCfg"]
ArmResidualSceneCfg = _arm_residual_classes["ArmResidualSceneCfg"]
ArmResidualActionsCfg = _arm_residual_classes["ArmResidualActionsCfg"]
ArmResidualEventCfg = _arm_residual_classes["ArmResidualEventCfg"]
TacoTrainArmResidualEnvCfg = _arm_residual_classes["TacoTrainArmResidualEnvCfg"]
TacoArmResidualBaseEnvCfg = _arm_residual_classes["TacoArmResidualBaseEnvCfg"]
TacoRHArmResidualEnvCfg = _arm_residual_classes["TacoRHArmResidualEnvCfg"]

TacoBimanualArmResidualEnvCfg = build_arm_residual_cfg_classes(
    **_RESIDUAL_COMMON,
    task_observations_cfg_cls=BimanualTaskObservationsCfg,
    task_rewards_cfg_cls=BimanualTaskRewardsCfg,
    task_terminations_cfg_cls=BimanualTaskTerminationsCfg,
    train_arm_residual_env_cfg_name="TacoTrainBimanualArmResidualEnvCfg",
    arm_residual_base_env_cfg_name="TacoBimanualArmResidualBaseEnvCfg",
    rh_arm_residual_env_cfg_name="TacoBimanualArmResidualEnvCfg",
    bimanual=True,
)["TacoBimanualArmResidualEnvCfg"]

TacoLeftUniArmResidualEnvCfg = build_arm_residual_cfg_classes(
    **_RESIDUAL_COMMON,
    task_observations_cfg_cls=LeftUniTaskObservationsCfg,
    task_rewards_cfg_cls=RightHandTaskRewardsCfg,
    task_terminations_cfg_cls=RightHandTaskTerminationsCfg,
    train_arm_residual_env_cfg_name="TacoTrainLeftUniArmResidualEnvCfg",
    arm_residual_base_env_cfg_name="TacoLeftUniArmResidualBaseEnvCfg",
    rh_arm_residual_env_cfg_name="TacoLeftUniArmResidualEnvCfg",
    left_uni=True,
)["TacoLeftUniArmResidualEnvCfg"]
