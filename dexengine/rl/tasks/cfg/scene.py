# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene base for the arm-in-loop cfg chain: desk, lights, head camera, robot placeholder.

The robot itself is NOT defined here: ``TrainArmEnvCfg.__post_init__`` replaces
``scene.robot`` wholesale with the registered (arm, hand) combo articulation
(``scene/robot.py``); this scene only carries its prim path and
default spawn pose (the previous default was
the removed legacy bimanual robot).

Scene hierarchy:
  /World/Floor          -- 20x20m walnut plank visual floor (every render binds the look's floor over it)
  /World/GroundPlane    -- physics ground plane
  /World/light          -- dome light
  {ENV}/Desk            -- visual desk: the Willow table from the look mirror (desk_visual.desk_cfg)
  {ENV}/DeskCollisionTop -- tabletop collision cuboid
  {ENV}/DeskLeg{0-3}    -- 4 leg collision cuboids
  {ENV}/Robot            -- full-robot combo articulation (arms + head + hands; from the specs)
  {ENV}/Object           -- placeholder cuboid (hidden; hand-tracking builds)

Camera:
  Head-mounted 768x768 camera ``HeadCamera``, re-parented under the active arm's ZED
  prim (``ArmSpec.head_camera_prim_relpath``) by ``TrainArmEnvCfg.__post_init__``.
  Rx(-90) offset, ROS convention, focal_length=8.0, horizontal_aperture=20.955
"""

from __future__ import annotations

from dexengine.rl.tasks import env_flags

# Parse DEXENGINE_KEY_LIGHT here (a temporary in a configclass body would be taken for a scene entity)
_KEY_LIGHT_VALS = [float(x) for x in env_flags.get("DEXENGINE_KEY_LIGHT").split(",") if x.strip()]

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from dexengine.core.desk import (
    DESK_LEG_CENTER_Z,
    DESK_LEG_DX,
    DESK_LEG_DY,
    DESK_LEG_HALF,
    DESK_TOP_CENTER_Z,
    DESK_TOP_HALF_X,
    DESK_TOP_HALF_Y,
    DESK_TOP_THICK,
)
from dexengine.rl.tasks.cfg.constants import (
    CAMERA_CLIPPING_RANGE,
    CAMERA_FOCAL_LENGTH,
    CAMERA_FOCUS_DISTANCE,
    CAMERA_HEIGHT,
    CAMERA_HORIZONTAL_APERTURE,
    CAMERA_OFFSET_POS,
    CAMERA_OFFSET_ROT,
    CAMERA_UPDATE_PERIOD,
    CAMERA_WIDTH,
    DESK_OFFSET_X,
    DOME_LIGHT_COLOR,
    DOME_LIGHT_INTENSITY,
    ROBOT_DEFAULT_POS,
)

# Reuse observation, reward, termination, curriculum, event classes from training config
from dexengine.rl.tasks.cfg.hand_tracking import FLOOR_MDL
from dexengine.rl.tasks.scene.desk_visual import desk_cfg as _desk_cfg

##
# Event configuration for eval-arm with IK teleport reset
##


# The eval-arm env cfg chain (EvalArmEventCfg and friends) is gone with the Eval-Arm registration;
# only TacoEvalSceneCfg below is live
# (scene base for the arm-in-loop cfg chain).


##
# Scene definition
##


@configclass
class TacoEvalSceneCfg(InteractiveSceneCfg):
    """Scene for full-arm evaluation with IK-driven wrist + desk.

    The robot is a placeholder the env cfg replaces with the registered combo
    (see ``robot`` below). Same desk scene as training for consistent
    physical interaction.
    """

    # Visual floor -- 20x20m walnut planks. Training and judging draw nothing, and the one canonical look
    # (record.looks.look_d.apply) binds its floor material over it before any frame is drawn, so its own look never
    # shows; v0 keeps it, as the v0 data was generated with it.
    floor = AssetBaseCfg(
        prim_path="/World/Floor",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.005)),
        spawn=sim_utils.MeshCuboidCfg(
            size=(20.0, 20.0, 0.01),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=FLOOR_MDL,
                project_uvw=True,
                texture_scale=(0.5, 0.5),
            ),
        ),
    )

    # Physics ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Every desk dimension is a dexengine.core.desk constant (these are the tabletop top; task scenes refit it from the
    # run's top statement, dexengine.core.desk.apply_desk_top). The hand-tracking scene keeps these values (its desk is
    # visual only, so it stands far away).
    # Visual desk, tabletop at z=0.76m: the Willow USD of the look mirror (desk_visual.desk_cfg — one desk for every
    # scene).
    desk = _desk_cfg(DESK_OFFSET_X)

    # Desk collision: tabletop cuboid (rotated to match visual).
    # NOTE: 0.745 below is the box CENTER; with z-thickness 0.03 the top SURFACE sits at
    # 0.745 + 0.015 = 0.76 m, matching traj_offset_z (the wrist/object reference z-offset).
    desk_collision_top = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DeskCollisionTop",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(DESK_OFFSET_X, 0.0, DESK_TOP_CENTER_Z),
            rot=(0.707, 0.0, 0.0, 0.707),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(2 * DESK_TOP_HALF_Y, 2 * DESK_TOP_HALF_X, DESK_TOP_THICK),
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # Desk collision: 4 individual legs
    desk_leg_0 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DeskLeg0",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-DESK_LEG_DX + DESK_OFFSET_X, DESK_LEG_DY, DESK_LEG_CENTER_Z)),
        spawn=sim_utils.CuboidCfg(
            size=(2 * DESK_LEG_HALF, 2 * DESK_LEG_HALF, 2 * DESK_LEG_CENTER_Z),
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    desk_leg_1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DeskLeg1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-DESK_LEG_DX + DESK_OFFSET_X, -DESK_LEG_DY, DESK_LEG_CENTER_Z)),
        spawn=sim_utils.CuboidCfg(
            size=(2 * DESK_LEG_HALF, 2 * DESK_LEG_HALF, 2 * DESK_LEG_CENTER_Z),
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    desk_leg_2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DeskLeg2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(DESK_LEG_DX + DESK_OFFSET_X, DESK_LEG_DY, DESK_LEG_CENTER_Z)),
        spawn=sim_utils.CuboidCfg(
            size=(2 * DESK_LEG_HALF, 2 * DESK_LEG_HALF, 2 * DESK_LEG_CENTER_Z),
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    desk_leg_3 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DeskLeg3",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(DESK_LEG_DX + DESK_OFFSET_X, -DESK_LEG_DY, DESK_LEG_CENTER_Z)),
        spawn=sim_utils.CuboidCfg(
            size=(2 * DESK_LEG_HALF, 2 * DESK_LEG_HALF, 2 * DESK_LEG_CENTER_Z),
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # Robot — a **placeholder**. TrainArmEnvCfg.__post_init__ replaces it wholesale with the registered (arm, hand)
    # spec's robot (scene/robot.make_combo_articulation_cfg), taking from this placeholder only the prim path, the
    # spawn position (an arm's base_pos wins) and the rotation. It has no spawn and no actuators, so a scene that
    # was never replaced fails cfg validation (MISSING actuators) before spawning.
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_DEFAULT_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Right object -- hidden placeholder for hand-tracking builds (tiny, underground,
    # kinematic). object_left is set dynamically (bimanual / --side left spawn).
    object_right: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ObjectRight",
        spawn=sim_utils.CuboidCfg(
            size=(0.001, 0.001, 0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, -10.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Lights — a grey dome without texture (colour and intensity: cfg/constants.py); the canonical look
    # (record.looks.look_d.apply) turns it off and lights the room with its ceiling panels.
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=DOME_LIGHT_COLOR,
            intensity=DOME_LIGHT_INTENSITY,
            texture_file=None,
        ),
    )

    # DEXENGINE_KEY_LIGHT="intensity[,r,g,b]": a directional key light (DistantLight 3000 + a dark dome give the
    # 'natural grey' shading). Unset = no prim (as before). Every assignment in a configclass body is parsed as a
    # scene entity (a _kl temporary too!), so the parsing lives in the module-level _KEY_LIGHT_VALS.
    if _KEY_LIGHT_VALS:
        key_light = AssetBaseCfg(
            prim_path="/World/keyLight",
            spawn=sim_utils.DistantLightCfg(
                color=(tuple(_KEY_LIGHT_VALS[1:4]) if len(_KEY_LIGHT_VALS) >= 4 else (0.75, 0.75, 0.75)),
                intensity=_KEY_LIGHT_VALS[0],
            ),
        )

    # Head-mounted camera — config from cfg/constants.py.
    # The prim path is arm-neutral: TrainArmEnvCfg.__post_init__ keeps the leaf (HeadCamera) and
    # re-parents it under the active arm's ZED prim (ArmSpec.head_camera_prim_relpath), or drops
    # the camera when the arm has none (the default used to be the removed legacy
    # arm's hardware path).
    head_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/HeadCamera",
        update_period=CAMERA_UPDATE_PERIOD,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAMERA_FOCAL_LENGTH,
            focus_distance=CAMERA_FOCUS_DISTANCE,
            horizontal_aperture=CAMERA_HORIZONTAL_APERTURE,
            clipping_range=CAMERA_CLIPPING_RANGE,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=CAMERA_OFFSET_POS,
            rot=CAMERA_OFFSET_ROT,
            convention="ros",
        ),
    )


##
# MDP settings
##
