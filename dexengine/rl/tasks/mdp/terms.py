# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The MDP namespace the env cfgs read as ``mdp.<name>``.

Isaac Lab's stock terms come first; the task's own terms (actions, observations, rewards, terminations, events,
curriculum, reference trajectory) are imported after them, so a task name wins over a stock name.
"""

from isaaclab.envs.mdp import *  # noqa: F403

from dexengine.rl.tasks.mdp.actions.arm_ik import (  # noqa: F401
    ArmIKFingersPositionAction,
    ArmIKFingersPositionActionCfg,
)
from dexengine.rl.tasks.mdp.actions.residual_ik import ResidualArmIKAction, ResidualArmIKActionCfg  # noqa: F401
from dexengine.rl.tasks.mdp.curriculums import LinearThresholdSchedule, RsiSchedule  # noqa: F401
from dexengine.rl.tasks.mdp.events.randomization import (  # noqa: F401
    ramp_object_gravity,
    randomize_object_com_at_reset,
    randomize_object_friction,
    randomize_object_friction_at_reset,
    randomize_object_mass_at_reset,
    randomize_object_restitution_at_reset,
)
from dexengine.rl.tasks.mdp.events.reset import (  # noqa: F401
    randomize_initial_pose,
    reset_extra_objects,
    reset_hands_to_trajectory,
    reset_object_to_trajectory,
    reset_support_discs,
    reset_support_objects,
)
from dexengine.rl.tasks.mdp.events.reset_arm import reset_arm_to_trajectory  # noqa: F401
from dexengine.rl.tasks.mdp.observations.hand import HandObservation, ReferenceBaseAction  # noqa: F401
from dexengine.rl.tasks.mdp.rewards.task import (  # noqa: F401
    RewardObjAngVel,
    RewardObjArtic,
    RewardObjPos,
    RewardObjRot,
    RewardObjVel,
    residual_norm_penalty,
)
from dexengine.rl.tasks.mdp.rewards.task_keypoint import RewardObjKeypoint, RewardObjKeypointDiag  # noqa: F401
from dexengine.rl.tasks.mdp.rewards.tracking import (  # noqa: F401
    RewardEefPos,
    RewardEefRot,
    RewardIndexTip,
    RewardLevel1,
    RewardLevel2,
    RewardMiddleTip,
    RewardPinkyTip,
    RewardRingTip,
    RewardThumbTip,
)
from dexengine.rl.tasks.mdp.rewards.tracking_motion import (  # noqa: F401
    RewardActionRate,
    RewardEefVel,
    RewardFingerPose,
    RewardJointsVel,
    RewardPower,
    RewardWristPower,
)
from dexengine.rl.tasks.mdp.terminations.hand import (  # noqa: F401
    FingertipTrackingExceeded,
    trajectory_done,
    velocity_sanity,
)
from dexengine.rl.tasks.mdp.terminations.task import (  # noqa: F401
    ObjectTrackingExceeded,
    ObjectTrackingSplitDiag,
)
from dexengine.rl.tasks.mdp.trajectory.manager import TrajectoryManager  # noqa: F401
from dexengine.rl.tasks.mdp.trajectory.manager_cfg import TrajectoryManagerCfg  # noqa: F401
