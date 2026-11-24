"""Scene and termination cfg classes of the residual env.

Each @configclass body lists its fields in the order Isaac Lab reads them (scene spawn order, termination term
order); multi-line field values come from the helpers below, so a class body is one line per field.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from dexengine.core.hands import mirror_right_to_left as _mirror_right_to_left
from dexengine.rl.tasks.cfg.residual import _OBJ_FRICTION, make_object_mass_props
from dexengine.rl.tasks.scene import split_artic as _split_artic

_ROBOT_NS = "{ENV_REGEX_NS}/Robot/"
_RIGHT_OBJECT_BODY = "{ENV_REGEX_NS}/ObjectRight/base_link"
_LEFT_OBJECT_BODY = "{ENV_REGEX_NS}/ObjectLeft/base_link"


def _fingertip_exceeded(mdp, side):
    """Fingertip tracking-distance termination of one hand; thresholds anneal from start to end (metres)."""
    return DoneTerm(
        func=mdp.FingertipTrackingExceeded,
        time_out=False,
        params={
            "thresholds_start": {"thumb": 0.08, "index": 0.06, "middle": 0.06, "ring": 0.06, "pinky": 0.06},
            "thresholds_end": {"thumb": 0.04, "index": 0.04, "middle": 0.04, "ring": 0.04, "pinky": 0.04},
            "side": side,
        },
    )


def _object_tracking(mdp, side, **object_key):
    """Object-tracking termination of one hand. The threshold is start + (end - start) * curriculum progress: training
    anneals start -> end (the values here are defaults launch.plan overrides), eval
    (--freeze_curriculum, progress 1) uses the end values."""
    return DoneTerm(
        func=mdp.ObjectTrackingExceeded,
        time_out=False,
        params={
            "pos_start": 0.05,
            "pos_end": 0.03,
            "rot_start_deg": 120.0,
            "rot_end_deg": 90.0,
            "side": side,
            **object_key,
            "grace_period": 20,
        },
    )


def _residual_terminations_cfg(*, bimanual, left_uni, mdp, object_mass_props, task_terminations_cfg_cls):
    """Terminations of the residual env: the task terminations plus fingertip and object tracking gates."""
    if left_uni and bimanual:
        raise ValueError(
            "build_arm_residual_cfg_classes: left_uni=True is mutually exclusive with "
            "bimanual (left_uni is a single-object mirror of the right-uni build)."
        )
    left_present = bool(bimanual)

    # None -> DEFAULT_OBJECT_MASS_KG for every object; a cfg may pass its own MassPropertiesCfg.
    if object_mass_props is None:
        object_mass_props = make_object_mass_props()

    @configclass
    class ArmResidualTerminationsCfg(task_terminations_cfg_cls):
        """Arm-involved task terminations, looser than the floating hand's.

        The arm adds IK lag, PD lag and less precise initial poses, so against the hand-only thresholds (start and
        end): fingertip 1.5x, object position 2.5x, object rotation 2x, object grace period 8 -> 20 steps. The thumb
        fingertip thresholds equal the hand-tracking env's, so the thumb gate is the same in every build. This
        object_tracking overrides the one of the dataset terminations class.
        """

        fingertip_exceeded = _fingertip_exceeded(mdp, "right")
        object_tracking = _object_tracking(mdp, "right")
        if bimanual:
            object_tracking_left = _object_tracking(mdp, "left", object_key="object_left")
            fingertip_exceeded_left = _fingertip_exceeded(mdp, "left")

    return ArmResidualTerminationsCfg, left_present, object_mass_props


def _objfilter_sensor(body, object_prim, ns=_ROBOT_NS):
    """A single-body contact sensor filtered against one object prim.

    One body x N envs against one object x N envs fills force_matrix_w; a multi-body sensor with a single filter
    leaves it zero (PhysX filter cardinality). Used for the five per-fingertip sensors of each side.
    """
    return ContactSensorCfg(
        prim_path=ns + body,
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=[object_prim],
    )


def _hand_contact(ns, body_regex):
    """Multi-body hand contact sensor (net force per body, for the rewards and the net+dist gate). It carries no
    object filter (see _objfilter_sensor); object-filtered force comes from the per-fingertip sensors."""
    return ContactSensorCfg(
        prim_path=ns + body_regex,
        update_period=0.0,
        history_length=0,
    )


def _ground(ground_friction):
    """The ground plane with one friction value for the static and dynamic coefficients."""
    return AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=ground_friction,
                dynamic_friction=ground_friction,
            ),
        ),
    )


def _object(prim_path, diffuse_color, max_depenetration_velocity, object_mass_props, **spawn_extra):
    """A 3 cm placeholder cuboid rigid object without gravity (the gravity ramp event applies it).

    The initial pose is a pre-reset placeholder: reset_object_to_trajectory moves the object to the reference pose at
    every reset; 0.768 m is the reset height of a resting object on the 0.76 m desk.
    """
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.03),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=diffuse_color),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                max_linear_velocity=50.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=max_depenetration_velocity,
            ),
            mass_props=object_mass_props,
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
                torsional_patch_radius=0.005,
                min_torsional_patch_radius=0.001,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=_OBJ_FRICTION,
                dynamic_friction=_OBJ_FRICTION,
            ),
            **spawn_extra,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.768),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def _residual_scene_cfg(
    *,
    arm_spec,
    eval_scene_cfg_cls,
    ground_friction,
    hand_spec,
    left_present,
    left_uni,
    max_depenetration_velocity,
    object_mass_props,
):
    """Scene of the residual env: robot, objects, contact sensors and supports."""
    if arm_spec is None or hand_spec is None:
        raise ValueError(
            f"build_arm_residual_cfg_classes: arm_spec={arm_spec!r} hand_spec={hand_spec!r} — both must be "
            "registered specs (no default; the legacy pair literal path is gone)."
        )

    # Active arm. The robot articulation itself is swapped by the base class (cfg/arm_ik TrainArmEnvCfg.__post_init__
    # -> scene/robot), which this env cfg super()-calls; here only the arm-named wiring uses the spec.
    _active_arm = arm_spec

    # Hand-dependent contact-sensor targets, from the hand spec.
    _contact_body_regex = hand_spec.contact_body_regex_right
    _tip_bodies = tuple(hand_spec.tip_body_names_right)

    # Left contact-sensor targets: the right's mirror under the right_ -> left_ substitution
    # (core.hands.mirror_right_to_left, shared with scene/robot so the two sides cannot drift).
    _to_left = _mirror_right_to_left

    _left_contact_body_regex = _to_left(hand_spec.contact_body_regex_right)
    _left_tip_bodies = tuple(_to_left(b) for b in hand_spec.tip_body_names_right)

    # Split articulation: the left links spawn under `/RobotL` (DEXENGINE_BI_SPLIT_ARTIC=1); unset keeps
    # "{ENV_REGEX_NS}/Robot/".
    _LEFT_NS = _split_artic.left_robot_ns() + "/"

    ArmResidualSceneCfg = _build_arm_residual_scene_cfg(
        eval_scene_cfg_cls,
        ground_friction,
        left_present,
        left_uni,
        max_depenetration_velocity,
        object_mass_props,
        _LEFT_NS,
        _contact_body_regex,
        _left_contact_body_regex,
        _left_tip_bodies,
        _tip_bodies,
    )

    return ArmResidualSceneCfg, _active_arm, _to_left


def _build_arm_residual_scene_cfg(
    eval_scene_cfg_cls,
    ground_friction,
    left_present,
    left_uni,
    max_depenetration_velocity,
    object_mass_props,
    _LEFT_NS,
    _contact_body_regex,
    _left_contact_body_regex,
    _left_tip_bodies,
    _tip_bodies,
):
    """ArmResidualSceneCfg: the eval scene with the arm's ground, physical objects and contact sensors."""

    @configclass
    class ArmResidualSceneCfg(eval_scene_cfg_cls):
        """Arm scene: full robot + physical object + contact sensors.

        The sensor field names are a contract (contact_base.OBJ_SENSOR_NAMES; a name/filter mismatch silently zeroes
        force_matrix_w), each keeps its `: ContactSensorCfg` annotation so it stays a scene field, and the fingertip
        order is thumb, index, middle, ring, little. A left-only scene (left_uni) replaces the right sensor group by
        the left one and removes the right object, so a term still bound to object_right fails loudly.
        """

        # No desk legs (fewer collision pairs).
        desk_leg_0 = None
        desk_leg_1 = None
        desk_leg_2 = None
        desk_leg_3 = None

        ground = _ground(ground_friction)
        object_right: RigidObjectCfg = _object(
            "{ENV_REGEX_NS}/ObjectRight", (0.8, 0.2, 0.2), max_depenetration_velocity, object_mass_props
        )
        if not left_uni:
            right_hand_contact: ContactSensorCfg = _hand_contact(_ROBOT_NS, _contact_body_regex)
            right_thumb_obj_s: ContactSensorCfg = _objfilter_sensor(_tip_bodies[0], _RIGHT_OBJECT_BODY)
            right_index_obj_s: ContactSensorCfg = _objfilter_sensor(_tip_bodies[1], _RIGHT_OBJECT_BODY)
            right_middle_obj_s: ContactSensorCfg = _objfilter_sensor(_tip_bodies[2], _RIGHT_OBJECT_BODY)
            right_ring_obj_s: ContactSensorCfg = _objfilter_sensor(_tip_bodies[3], _RIGHT_OBJECT_BODY)
            right_little_obj_s: ContactSensorCfg = _objfilter_sensor(_tip_bodies[4], _RIGHT_OBJECT_BODY)
        else:
            object_right: RigidObjectCfg | None = None
            left_hand_contact: ContactSensorCfg = _hand_contact(_LEFT_NS, _left_contact_body_regex)
            left_thumb_obj_s: ContactSensorCfg = _objfilter_sensor(_left_tip_bodies[0], _LEFT_OBJECT_BODY, _LEFT_NS)
            left_index_obj_s: ContactSensorCfg = _objfilter_sensor(_left_tip_bodies[1], _LEFT_OBJECT_BODY, _LEFT_NS)
            left_middle_obj_s: ContactSensorCfg = _objfilter_sensor(_left_tip_bodies[2], _LEFT_OBJECT_BODY, _LEFT_NS)
            left_ring_obj_s: ContactSensorCfg = _objfilter_sensor(_left_tip_bodies[3], _LEFT_OBJECT_BODY, _LEFT_NS)
            left_little_obj_s: ContactSensorCfg = _objfilter_sensor(_left_tip_bodies[4], _LEFT_OBJECT_BODY, _LEFT_NS)

        # Bimanual: the second physical object and the left sensor group. The left object activates contact sensors
        # so the left per-fingertip sensors report object-filtered force.
        if left_present:
            object_left: RigidObjectCfg = _object(
                "{ENV_REGEX_NS}/ObjectLeft",
                (0.2, 0.2, 0.8),
                max_depenetration_velocity,
                object_mass_props,
                activate_contact_sensors=True,
            )
            left_hand_contact: ContactSensorCfg = _hand_contact(_LEFT_NS, _left_contact_body_regex)
            left_thumb_obj_s: ContactSensorCfg = _objfilter_sensor(_left_tip_bodies[0], _LEFT_OBJECT_BODY, _LEFT_NS)
            left_index_obj_s: ContactSensorCfg = _objfilter_sensor(_left_tip_bodies[1], _LEFT_OBJECT_BODY, _LEFT_NS)
            left_middle_obj_s: ContactSensorCfg = _objfilter_sensor(_left_tip_bodies[2], _LEFT_OBJECT_BODY, _LEFT_NS)
            left_ring_obj_s: ContactSensorCfg = _objfilter_sensor(_left_tip_bodies[3], _LEFT_OBJECT_BODY, _LEFT_NS)
            left_little_obj_s: ContactSensorCfg = _objfilter_sensor(_left_tip_bodies[4], _LEFT_OBJECT_BODY, _LEFT_NS)

    return ArmResidualSceneCfg
