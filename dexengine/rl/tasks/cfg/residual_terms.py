"""Action and event cfg classes of the residual env."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from dexengine.core.mounts import mount_for as _mount_for
from dexengine.core.physics import GRAVITY_MPS2
from dexengine.rl.tasks.cfg.residual import RAMP_STEPS_BUILD_DEFAULT
from dexengine.rl.tasks.mdp.gravity_wrench import MODE as _PREPHYSICS_MODE
from dexengine.rl.tasks.scene import split_artic as _split_artic


def _residual_actions_cfg(
    *,
    _active_arm,
    _to_left,
    action_ema_default,
    bimanual,
    finger_residual_scale_default,
    hand_spec,
    left_uni,
    mdp,
    residual_scale_default,
    wrist_residual_scale_default,
):
    """Action terms of the residual env (one residual arm-IK term per active side)."""

    return _build_residual_actions_cfg(
        _active_arm,
        _to_left,
        action_ema_default,
        bimanual,
        finger_residual_scale_default,
        hand_spec,
        left_uni,
        mdp,
        residual_scale_default,
        wrist_residual_scale_default,
    )


def _build_residual_actions_cfg(
    _active_arm,
    _to_left,
    action_ema_default,
    bimanual,
    finger_residual_scale_default,
    hand_spec,
    left_uni,
    mdp,
    residual_scale_default,
    wrist_residual_scale_default,
):
    """The residual action term kwargs per side and the actions cfg class."""

    def _make_residual_action_kwargs(side, hand_spec):
        if side == "right":
            if hand_spec is None:
                _hand_kwargs = {
                    "finger_joint_names": [
                        "right_hand_thumb_1_joint",
                        "right_hand_thumb_2_joint",
                        "right_hand_index_1_joint",
                        "right_hand_middle_1_joint",
                        "right_hand_ring_1_joint",
                        "right_hand_little_1_joint",
                    ],
                }
            else:
                _hand_kwargs = {
                    "finger_joint_names": list(hand_spec.finger_joint_names_right),
                    "coupled_joint_names": list(hand_spec.coupled_joint_names_right),
                    "coupling_ratios": list(hand_spec.coupling_ratios) or None,
                    "coupling_source_idx": list(hand_spec.coupling_source_idx) or None,
                    # (arm, hand) pair property — see MOUNT_REGISTRY.
                    "link7_to_wrist_offset": tuple(
                        _mount_for(hand_spec.name, _active_arm.name).ee_to_wrist_offset_for("right")
                    ),
                }
            _arm_joint_names = _active_arm.arm_joint_exprs("right")  # floating hand = [] (no arm joints)
            _ee_body_name = _active_arm.ee_body_name("right")
            _side_kwargs = {}  # right uses the action cfg's default side
        else:  # left — RH56F1 literals when no spec; the spec's mirror otherwise.
            _arm_joint_names, _ee_body_name, _hand_kwargs, _side_kwargs = _hand_action_kwargs(
                hand_spec, _active_arm, _to_left
            )
        # Split articulation: when DEXENGINE_BI_SPLIT_ARTIC=1 the LEFT arm+hand
        # are a SEPARATE articulation (`robot_left`). The left action must operate on
        # THAT articulation, so it targets the `left_hand` alias (re-pointed to
        # robot_left in tasks/env_base.py). Unset -> `left_hand` aliases to `robot`,
        # so both asset_names resolve identically (byte-identical). Right always "robot".
        _asset_name = "robot"
        if side == "left" and _split_artic.enabled():
            _asset_name = "left_hand"
        return dict(
            asset_name=_asset_name,
            arm_joint_names=_arm_joint_names,
            ee_body_name=_ee_body_name,
            **_hand_kwargs,
            ik_lambda_val=0.05,
            wrist_pos_scale=0.02,
            wrist_rot_scale=0.1,
            residual_scale=residual_scale_default,
            wrist_residual_scale=wrist_residual_scale_default,
            finger_residual_scale=finger_residual_scale_default,
            action_ema=action_ema_default,
            use_trajectory_reference=False,
            wrist_smooth_weight=-1.0,
            finger_smooth_weight=-1.0,
            **_side_kwargs,
        )

    _action_kwargs = _make_residual_action_kwargs("right", hand_spec)

    return _build_actions_cfg_class(bimanual, hand_spec, left_uni, mdp, _action_kwargs, _make_residual_action_kwargs)


def _build_actions_cfg_class(bimanual, hand_spec, left_uni, mdp, _action_kwargs, _make_residual_action_kwargs):
    """Build the actions cfg by constructing the action term at factory time. Constructed once and reused by both env
    subclasses (matches the original code, where each env cfg instantiated its own ArmResidualActionsCfg()).
    """
    _residual_action_term = mdp.ResidualArmIKActionCfg(**_action_kwargs)

    # Bimanual Option 2: SECOND residual action term for the LEFT hand. Mirror
    # of the right term with left arm joints / left link7 / left finger joints
    # and side='left'. Built only when bimanual=True. RH56F1 left coupling is
    # auto-derived by the inner ArmIKFingersPositionAction (coupled_joint_names
    # left as None -> '{side}_hand_*' auto-list, where side is read from the
    # left-prefixed finger_joint_names[0]).
    _left_residual_action_term = None
    if bimanual:
        # Left always uses RH56F1 literals (never a HandSpec); the helper's
        # left branch ignores hand_spec and appends side="left".
        _left_action_kwargs = _make_residual_action_kwargs("left", hand_spec)
        _left_residual_action_term = mdp.ResidualArmIKActionCfg(**_left_action_kwargs)

    # LEFT-UNI: the SINGLE 12-dim action term is the LEFT residual term. Same
    # kwargs builder as the bimanual left term, plus object_key='object' so the
    # release-latch / phase-C / scripted-grip paths read the SINGLE 'object' key
    # (side='left' alone would resolve to the absent 'object_left'; see
    # mdp/actions/residual_ik.py cfg.object_key).
    _leftuni_residual_action_term = None
    if left_uni:
        _leftuni_action_kwargs = _make_residual_action_kwargs("left", hand_spec)
        _leftuni_residual_action_term = mdp.ResidualArmIKActionCfg(object_key="object", **_leftuni_action_kwargs)

    @configclass
    class ArmResidualActionsCfg:
        """Residual action config: matches the hand-tracking arm-IK action settings.

        Scales and smoothing match the hand-tracking arm IK (no temporal ensembling).
        Uses current-state mode (not trajectory reference) — the residual
        process_actions handles base+residual composition directly.

        Bimanual Option 2: right_hand_joint_pos is declared FIRST and
        left_hand_joint_pos SECOND so the concatenated action manager layout is
        [right | left], matching the obs ordering. The left term is added
        only when bimanual=True.

        LEFT-UNI: left_hand_joint_pos is the ONLY term (12-dim action manager,
        mirror of the right-uni layout). The term name MUST stay
        'left_hand_joint_pos' — mdp/events reset_arm_to_trajectory keys its
        left-arm RSI reset off this exact action-term name.
        """

        if left_uni:
            left_hand_joint_pos = _leftuni_residual_action_term
        else:
            right_hand_joint_pos = _residual_action_term
            if bimanual:
                left_hand_joint_pos = _left_residual_action_term

    return (ArmResidualActionsCfg,)


def _hand_action_kwargs(hand_spec, _active_arm, _to_left):
    """(This branch once read "always RH56F1 literals, never a HandSpec": true while SharpA was right-only, wrong for a
    LEFT bundle where the LEFT hand is the learner.).
    """
    if hand_spec is None:
        _hand_kwargs = {
            "finger_joint_names": [
                "left_hand_thumb_1_joint",
                "left_hand_thumb_2_joint",
                "left_hand_index_1_joint",
                "left_hand_middle_1_joint",
                "left_hand_ring_1_joint",
                "left_hand_little_1_joint",
            ],
        }
    else:
        _hand_kwargs = {
            "finger_joint_names": [_to_left(n) for n in hand_spec.finger_joint_names_right],
            "coupled_joint_names": [_to_left(n) for n in hand_spec.coupled_joint_names_right],
            "coupling_ratios": list(hand_spec.coupling_ratios) or None,
            "coupling_source_idx": list(hand_spec.coupling_source_idx) or None,
            # mount offset by side (MountSpec.ee_to_wrist_offset_for); the ROTATION side is
            # derived inside the action term from finger_joint_names[0].
            "link7_to_wrist_offset": tuple(_mount_for(hand_spec.name, _active_arm.name).ee_to_wrist_offset_for("left")),
        }
    _arm_joint_names = _active_arm.arm_joint_exprs("left")  # floating hand = []
    _ee_body_name = _active_arm.ee_body_name("left")
    _side_kwargs = {"side": "left"}
    return _arm_joint_names, _ee_body_name, _hand_kwargs, _side_kwargs


def _residual_events(*, arm_train_event_cfg_cls, include_physics_dr_events, left_present, left_uni, mdp):
    """Event terms of the residual env: pose randomization, gravity and friction ramps, physics randomization."""

    def _ev_randomize_pose():
        return EventTerm(
            func=mdp.randomize_initial_pose,
            mode="reset",
            params={
                "randomize_fraction": 0.0,
                "finger_open_deg": 0.0,
                "wrist_z_max_cm": 0.0,
            },
        )

    (
        _ev_friction_decay,
        _ev_gravity_ramp,
        _ev_physics_dr_friction,
        _ev_physics_dr_mass,
        _ev_physics_dr_restitution,
        _ev_reset_object_left,
    ) = _make_ramp_and_dr_events(mdp)

    def _ev_physics_dr_com(object_name):
        return EventTerm(
            func=mdp.randomize_object_com_at_reset,
            mode="reset",
            params={"object_name": object_name, "com_offset_range_m": (-0.02, 0.02)},
        )

    return _build_event_cfg_class(
        arm_train_event_cfg_cls,
        include_physics_dr_events,
        left_present,
        left_uni,
        _ev_friction_decay,
        _ev_gravity_ramp,
        _ev_physics_dr_com,
        _ev_physics_dr_friction,
        _ev_physics_dr_mass,
        _ev_physics_dr_restitution,
        _ev_randomize_pose,
        _ev_reset_object_left,
    )


def _build_event_cfg_class(
    arm_train_event_cfg_cls,
    include_physics_dr_events,
    left_present,
    left_uni,
    _ev_friction_decay,
    _ev_gravity_ramp,
    _ev_physics_dr_com,
    _ev_physics_dr_friction,
    _ev_physics_dr_mass,
    _ev_physics_dr_restitution,
    _ev_randomize_pose,
    _ev_reset_object_left,
):
    """Active-object scene key: the side this uni build manipulates (left_uni -> the LEFT object is the single/active
    object; otherwise the right).
    """
    _active_obj = "object_left" if left_uni else "object_right"

    if include_physics_dr_events:

        @configclass
        class ArmResidualEventCfg(arm_train_event_cfg_cls):
            """Arm events: arm reset + object domain randomization."""

            randomize_pose = _ev_randomize_pose()
            gravity_ramp = _ev_gravity_ramp(_active_obj)
            friction_decay = _ev_friction_decay(_active_obj)

            # Paper 1 B1a per-reset physics DR (P0.2+); "train" range by default,
            # removed at runtime via train_cfg_overrides() when --enable_physics_dr=none.
            physics_dr_mass = _ev_physics_dr_mass(_active_obj)
            physics_dr_friction = _ev_physics_dr_friction(_active_obj)
            physics_dr_restitution = _ev_physics_dr_restitution(_active_obj)
            physics_dr_com = _ev_physics_dr_com(_active_obj)

            # LEFT-UNI: the active object IS object_left — reuse the left reset.
            if left_uni:
                reset_object = _ev_reset_object_left()

            # Bimanual Option 2: LEFT object reset + LEFT object DR (gated; default
            # OFF → never created, right baseline byte-identical). Mirror of the
            # right events re-pointed to 'object_left'.
            if left_present:
                reset_object_left = _ev_reset_object_left()
                gravity_ramp_left = _ev_gravity_ramp("object_left")
                friction_decay_left = _ev_friction_decay("object_left")
                physics_dr_mass_left = _ev_physics_dr_mass("object_left")
                physics_dr_friction_left = _ev_physics_dr_friction("object_left")
                physics_dr_restitution_left = _ev_physics_dr_restitution("object_left")
                physics_dr_com_left = _ev_physics_dr_com("object_left")
    else:

        @configclass
        class ArmResidualEventCfg(arm_train_event_cfg_cls):
            """Arm events: arm reset + object domain randomization."""

            randomize_pose = _ev_randomize_pose()
            gravity_ramp = _ev_gravity_ramp(_active_obj)
            friction_decay = _ev_friction_decay(_active_obj)

            # LEFT-UNI: the active object IS object_left — reuse the left reset.
            if left_uni:
                reset_object = _ev_reset_object_left()

            # Bimanual Option 2: LEFT object reset + LEFT gravity/friction
            # (no physics_dr in this branch, mirroring the right). Gated.
            if left_present:
                reset_object_left = _ev_reset_object_left()
                gravity_ramp_left = _ev_gravity_ramp("object_left")
                friction_decay_left = _ev_friction_decay("object_left")

    return (ArmResidualEventCfg,)


def _make_ramp_and_dr_events(mdp):
    """NOTE: the ramp_steps / start_fraction values in the two ramp builders below are."""

    # NOTE: the ramp_steps / start_fraction values in the two ramp builders below are
    # cfg-build skeletons — the gravity_ramp + friction_decay schedules are ALWAYS
    # overridden at runtime in dexengine/rl/train/runner.py: training sets
    # ramp_steps = int(--gravity_ramp_fraction × total_timesteps) (default 0.10) with
    # start_fraction=0.0; eval pins ramp_steps=0 / start_fraction=1.0 (full gravity, no
    # curriculum). They never take effect on any registered train/eval path. The skeleton
    # uses RAMP_STEPS_BUILD_DEFAULT (= GRAVITY_RAMP_FRACTION_DEFAULT × the canonical
    # horizon) so it is fraction-derived rather than a bare magic number.
    def _ev_gravity_ramp(object_name=None):
        p = {"ramp_steps": RAMP_STEPS_BUILD_DEFAULT, "gravity_magnitude": GRAVITY_MPS2, "start_fraction": 0.5}
        if object_name is not None:
            p = {"object_name": object_name, **p}
        return EventTerm(
            func=mdp.ramp_object_gravity,
            mode=_PREPHYSICS_MODE,  # right before physics (after the pose writes); timing owner mdp/gravity_wrench
            params=p,
        )

    def _ev_friction_decay(object_name=None):
        p = {
            "ramp_steps": RAMP_STEPS_BUILD_DEFAULT,
            "nominal_friction": 2.0,
            "initial_multiplier": 3.0,
            "final_multiplier": 1.0,
        }
        if object_name is not None:
            p = {"object_name": object_name, **p}
        return EventTerm(
            func=mdp.randomize_object_friction,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params=p,
        )

    def _ev_reset_object_left():
        return EventTerm(
            func=mdp.reset_object_to_trajectory,
            mode="reset",
            params={"object_cfg": SceneEntityCfg("object_left"), "side": "left"},
        )

    # Paper 1 B1a per-reset physics DR (P0.2+). Active by default with "train" DR
    # range; removed at runtime via train_cfg_overrides() when --enable_physics_dr=none.
    def _ev_physics_dr_mass(object_name):
        return EventTerm(
            func=mdp.randomize_object_mass_at_reset,
            mode="reset",
            params={"object_name": object_name, "mass_scale_range": (0.7, 1.3)},
        )

    def _ev_physics_dr_friction(object_name):
        return EventTerm(
            func=mdp.randomize_object_friction_at_reset,
            mode="reset",
            params={
                "object_name": object_name,
                "static_friction_scale_range": (0.5, 1.5),
                "dynamic_friction_scale_range": (0.5, 1.5),
            },
        )

    def _ev_physics_dr_restitution(object_name):
        return EventTerm(
            func=mdp.randomize_object_restitution_at_reset,
            mode="reset",
            params={"object_name": object_name, "restitution_scale_range": (0.5, 1.5)},
        )

    return (
        _ev_friction_decay,
        _ev_gravity_ramp,
        _ev_physics_dr_friction,
        _ev_physics_dr_mass,
        _ev_physics_dr_restitution,
        _ev_reset_object_left,
    )
