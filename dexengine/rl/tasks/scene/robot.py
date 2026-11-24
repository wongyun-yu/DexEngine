"""ArticulationCfg builder for every registered (arm, hand) combo — ONE place.

WHY
    Every pair gets its articulation cfg built HERE from the registered specs, so
    adding a robot is: register ArmSpec/HandSpec, fill MOUNT_REGISTRY, bake the
    combo USD — and never edit an env factory again. (Before, the legacy
    pair kept a hand-written asset cfg and literal actuator groups in the env
    factories; the pair was deleted and later its arm left the repository —
    this builder is the only path.)

    This also retires two single-arm assumptions the old hand_spec-only branch
    carried (measured, not hypothetical):
      * a hand-keyed robot USD path — a USD containing the ARM is a PAIR
        artifact. Combos resolve through ``robots.selection.combo_usd_path``.
      * left hand hardwired to RH56F1 literals — the generated combo USDs are
        UNIFORM (same hand both sides), so the left side now mirrors the spec
        via ``core.hands.mirror_right_to_left``.

GAIN RULE
    ``None`` in a spec gain field means "use the shared training default given
    by the caller"; a value is a hardware-reasoned override. effort/velocity
    caps always come from the spec (they are the retiming contract). Dict caps
    are keyed by RIGHT joint names in the spec; the bimanual groups here mirror
    the keys so Isaac's per-joint resolution covers both sides.
"""

from __future__ import annotations

import math

from dexengine.core.arms import ArmSpec
from dexengine.core.hands import (
    HandSpec,
    finger_effort_applied,
    finger_effort_scale_from_env,
    finger_velocity_applied,
)
from dexengine.core.hands import mirror_right_to_left as _to_left
from dexengine.robots.selection import USD_CACHE_DIR, assert_generation_coherent, combo_usd_path


def _joint_pos_map(hand: HandSpec, extra: dict[str, float] | None = None) -> dict[str, float]:
    """Initial joint value map: exceptions as explicit keys, the rest as a catch-all without those keys.

    Isaac's ``resolve_matching_names_values`` raises an **error** when one joint matches two patterns ("Multiple
    matches for X: '.*' and 'X'"); there is no priority rule. So a lookahead pattern that leaves the exceptions out is
    used instead of ``.*``.
    """
    overrides: dict[str, float] = {}
    for name, v in hand.joint_pos_defaults.items():
        overrides[name] = v
        overrides[_to_left(name)] = v
    if extra:
        overrides.update(extra)  # e.g. the head pitch: must go in before excl is computed, or it double-matches
    if not overrides:
        return {".*": 0.0}
    excl = "|".join(f"{n}" for n in sorted(overrides))
    return {f"(?!(?:{excl})$).*": 0.0, **overrides}


def _mirror_keys(value):
    """Right-keyed per-joint dict -> both-sides dict; scalars pass through."""
    if isinstance(value, dict):
        out = dict(value)
        out.update({_to_left(k): v for k, v in value.items()})
        return out
    return value


def finger_actuator_key(hand_name: str) -> str:
    """The finger actuator group's name in a combo robot cfg — the ONE spelling the builder below uses and the
    runtime ``--finger_*`` overrides look up (a second spelling is how those overrides silently matched nothing)."""
    if not hand_name:
        raise ValueError("finger_actuator_key: a hand name is required (no default)")
    return f"{hand_name}_fingers"


def make_combo_articulation_cfg(
    arm_spec: ArmSpec | None,
    hand_spec: HandSpec | None,
    *,
    prim_path: str,
    init_pos: tuple[float, float, float],
    init_rot: tuple[float, float, float, float],
    finger_armature: float,
    mimic_kp: float,
    mimic_kd: float,
    mimic_armature: float,
    rigid_linear_damping: float,
    rigid_angular_damping: float,
    max_depenetration_velocity: float,
):
    """Full-robot ArticulationCfg for a combo, from the specs alone.

    Both specs are required (a ``None`` on either axis raises).

    The spawn physics properties are the ones the original hand-written asset cfg
    carried (solver 8/0 iterations, contact sensors on, no self collision) plus the rigid-body damping and the
    depenetration velocity cap the caller passes from ``cfg.constants`` (the one owner — objects read the same
    constant, marker ``[SPAWN_PHYS]``) — baked here because this cfg IS ``scene.robot``.
    """
    # Local imports: constructing this module must not eagerly pull Isaac Sim
    # (same rule as the factories — CI imports the specs without a sim).
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets.articulation import ArticulationCfg

    # The legacy pair is gone, and so is the silent None -> legacy arm/hand default. Both axes must be real specs now.
    if arm_spec is None or hand_spec is None:
        raise ValueError(
            "make_combo_articulation_cfg: arm_spec/hand_spec is None — the legacy pair and the silent default that "
            "filled its place are gone. Pass a registered (arm, hand)."
        )
    arm, hand = arm_spec, hand_spec

    return _check_usd_generation(
        prim_path,
        init_pos,
        init_rot,
        finger_armature,
        mimic_kp,
        mimic_kd,
        mimic_armature,
        rigid_linear_damping,
        rigid_angular_damping,
        max_depenetration_velocity,
        ArticulationCfg,
        ImplicitActuatorCfg,
        arm,
        hand,
        sim_utils,
    )


def _check_usd_generation(
    prim_path,
    init_pos,
    init_rot,
    finger_armature,
    mimic_kp,
    mimic_kd,
    mimic_armature,
    rigid_linear_damping,
    rigid_angular_damping,
    max_depenetration_velocity,
    ArticulationCfg,
    ImplicitActuatorCfg,
    arm,
    hand,
    sim_utils,
):
    """The generation check happens **right here**, the one place where the USD is decided: is the body to spawn one set
    with the runtime base, and was that USD really baked from that variant URDF? A mismatch dies at once: not running
    beats running silently 12–17 cm off.
    """
    assert_generation_coherent(arm.name, hand.name, USD_CACHE_DIR)
    usd_path = combo_usd_path(arm.name, hand.name, USD_CACHE_DIR)

    actuators, usd_path = _combo_actuators_and_articulation(
        finger_armature,
        mimic_kp,
        mimic_kd,
        mimic_armature,
        ImplicitActuatorCfg,
        arm,
        hand,
        usd_path,
    )

    # Contact friction cannot be set here: the UsdFileCfg family (RigidObjectSpawnerCfg) has no physics_material field
    # at
    # all (measured: passing one dies with TypeError). The friction prescription goes on the object spawn
    # (DEXENGINE_OBJECT_FRICTION); contact friction combines both surfaces, so the purpose is the same.
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # True only for the floating hand (the whole hand weightless); normal arms False
                disable_gravity=arm.is_floating,
                max_depenetration_velocity=max_depenetration_velocity,
                linear_damping=rigid_linear_damping,
                angular_damping=rigid_angular_damping,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            activate_contact_sensors=True,
        ),
        init_state=_initial_state(ArticulationCfg, arm, hand, init_pos, init_rot),
        actuators=actuators,
        soft_joint_pos_limit_factor=1.0,
    )


def _initial_state(ArticulationCfg, arm, hand, init_pos, init_rot):
    """Spawn pose and joint defaults of the combo robot.

    An arm that carries its own spawn position wins (RB-Y1 stands on the floor); otherwise the scene value the caller
    passes (cfg.constants.ROBOT_DEFAULT_POS). Every joint defaults to 0, except joints whose limits exclude 0 (Isaac
    refuses to build the env then): Isaac's resolve_matching_names_values refuses double matches ('.*' and 'X'), so the
    exception keys are taken out of the catch-all with a negative lookahead. The arm spec's head_drive.pitch_deg (rby1
    30° down = canonical) pitches the head mount in every training and render env; the camera sits on the head link, so
    mount and camera pitch together. The pitch joint (head_joint_names[-1]) is not driven by the policy; a PD target
    pins it at every reset.
    """
    return ArticulationCfg.InitialStateCfg(
        pos=tuple(arm.base_pos) if arm.base_pos is not None else tuple(init_pos),
        rot=tuple(init_rot),
        joint_pos=_joint_pos_map(
            hand,
            extra=(
                {arm.head_joint_names[-1]: math.radians(arm.head_drive.pitch_deg)} if arm.head_joint_names else None
            ),
        ),
        joint_vel={".*": 0.0},
    )


def _combo_actuators_and_articulation(
    finger_armature,
    mimic_kp,
    mimic_kd,
    mimic_armature,
    ImplicitActuatorCfg,
    arm,
    hand,
    usd_path,
):
    """Actuators (floating hand or arm) and the articulation cfg. The arm and head groups take the arm spec's default
    drives (ArmSpec.arm_drive · head_drive); launch overrides apply later (env_setup.robot_gains)."""
    if arm.is_floating:
        actuators, usd_path = _floating_hand_joints(ImplicitActuatorCfg, hand, usd_path)
    else:
        actuators = {
            f"{arm.name}_arm": ImplicitActuatorCfg(
                joint_names_expr=arm.joint_names_expr(),
                velocity_limit_sim=arm.limit_map("velocity"),
                effort_limit_sim=arm.limit_map("effort"),
                stiffness=arm.arm_drive.kp,
                damping=arm.arm_drive.kd,
            ),
        }
    if arm.head_joint_names:
        actuators[f"{arm.name}_head"] = ImplicitActuatorCfg(
            joint_names_expr=list(arm.head_joint_names),
            effort_limit_sim=arm.head_drive.effort_limit,
            velocity_limit_sim=arm.head_drive.velocity_limit,
            stiffness=arm.head_drive.kp,
            damping=arm.head_drive.kd,
        )

    finger_joints = list(hand.finger_joint_names_right) + [_to_left(n) for n in hand.finger_joint_names_right]
    return _finger_effort_cap(
        finger_armature,
        mimic_kp,
        mimic_kd,
        mimic_armature,
        ImplicitActuatorCfg,
        hand,
        usd_path,
        actuators,
        finger_joints,
    )


def _floating_hand_joints(ImplicitActuatorCfg, hand, usd_path):
    """Floating hand: no arm joints, no arm actuators. The wrist is moved by external wrenches from the action term
    (experiments/ablations/floating_hand/action.py) and the whole hand is weightless (disable_gravity below; the same as
    the FlashCHORD whole-body gravcomp). Home of the values and rules = experiments/ablations/floating_hand/body.py.
    """
    from dexengine.rl.tasks import hooks

    _fh = hooks.floating_hand()
    _fh.require_supported(hand.name)
    # Hand identity marker: the floating ablation job checks it against the bundle hand
    print(f"[NA_HAND] {hand.name} · floating hand articulation", flush=True)
    actuators = {}
    if _fh.joint_drive():
        # Path C (DEXENGINE_FH_WRIST=joint, default off): the virtual-joint asset + PhysX implicit drive of the six
        # virtual joints (gains = WRIST_GAINS)
        usd_path = str(_fh.mode_canon_path(usd_path))
        _vg = _fh.vj_actuator_gains()
        actuators["floating_wrist_vj"] = ImplicitActuatorCfg(
            joint_names_expr=_fh.vj_joint_names("right") + _fh.vj_joint_names("left"),
            stiffness=_vg["stiffness"],
            damping=_vg["damping"],
            effort_limit_sim=_vg["effort_limit_sim"],
            armature=_fh.vj_armature(),
        )
        print(f"{_fh.MARK} {_fh.vj_armature_mark()}", flush=True)
        print(
            f"{_fh.MARK} [FH_WRIST] joint — wrist = six virtual joints (3 prismatic -> revolute Y·X·Z), PhysX "
            f"implicit drive · "
            f"prismatic kp {_fh.WRIST_GAINS.lin_kp:g} kd {_fh.WRIST_GAINS.lin_kd:g} · revolute kp "
            f"{_fh.WRIST_GAINS.rot_kp:g} "
            f"kd {_fh.WRIST_GAINS.rot_kd:g} · merged place {usd_path}",
            flush=True,
        )
    print(
        f"{_fh.MARK} body: 0 arm actuators · whole hand disable_gravity=True · merged USD place {usd_path} "
        "(the split wiring swaps in the per-side free-root USDs, split_artic.split_usd_paths)",
        flush=True,
    )
    return actuators, usd_path


def _finger_effort_cap(
    finger_armature, mimic_kp, mimic_kd, mimic_armature, ImplicitActuatorCfg, hand, usd_path, actuators, finger_joints
):
    """Applied cap = core.hands.finger_effort_applied(hand) = URDF effort × finger_effort_peak_scale (hardware maximum).
    DEXENGINE_FINGER_EFFORT_SCALE (default 1.0 = that applied cap as is) multiplies on top; read through
    finger_effort_scale_from_env. **Below 1 = a conservative cap under the hardware maximum, the canonical per-hand
    value**: owner core.hands.HAND_RL_RECIPES (one table; hand_recipe.py show). With the recipe switch on (default)
    scripts/stage3_rl.sh and the launcher load it from the recipe and create_env checks it against this applied value.
    **Above 1
    = beyond the hardware maximum, diagnostics only** (like the "too weak" check of an earlier hand port). Never used
    in canonical runs.
    """
    _eff_scale = finger_effort_scale_from_env()
    _eff = _mirror_keys(finger_effort_applied(hand))
    if _eff_scale != 1.0:
        _eff = {k: v * _eff_scale for k, v in _eff.items()}
        if _eff_scale > 1.0:
            print(
                f"[combo_articulation] ⚠ diagnostic switch: finger effort ×{_eff_scale} "
                f"(beyond the hardware maximum; never in canonical runs)",
                flush=True,
            )
        else:
            print(
                f"[combo_articulation] finger effort ×{_eff_scale} — conservative cap under the hardware maximum "
                f"(owner of the canonical per-hand value = core.hands.HAND_RL_RECIPES)",
                flush=True,
            )

    def _span(v):
        vals = list(v.values()) if isinstance(v, dict) else [v]
        return f"{min(vals):g}~{max(vals):g}" if len(set(vals)) > 1 else f"{vals[0]:g}"

    _kp = _mirror_keys(hand.finger_kp)
    _kd = _mirror_keys(hand.finger_kd)
    _peak = float(hand.finger_effort_peak_scale)
    _vel = _mirror_keys(finger_velocity_applied(hand))
    _vpeak = float(hand.finger_velocity_peak_scale)
    print(
        f"[FINGER_GAINS] hand={hand.name} effort={_span(_eff)} N·m kp={_span(_kp)} "
        f"kd={_span(_kd)} peak×{_peak:g}"
        + (f"(URDF rating -> nominal fingertip {hand.vendor_fingertip_force_n:g} N)" if _peak != 1.0 else "")
        + f" vel={_span(_vel)} rad/s"
        + (
            f" vel×{_vpeak:g}(URDF -> vendor open/close {hand.vendor_close_repetition_hz:g} Hz)"
            if _vpeak != 1.0
            else ""
        )
        + (
            f" (effort ×{_eff_scale} {'diagnostic' if _eff_scale > 1.0 else 'conservative cap'})"
            if _eff_scale != 1.0
            else ""
        ),
        flush=True,
    )
    actuators[finger_actuator_key(hand.name)] = ImplicitActuatorCfg(
        joint_names_expr=finger_joints,
        # Finger gains are the HAND's (HandSpec requires them — no shared finger gain to fall back to).
        stiffness=_mirror_keys(hand.finger_kp),
        damping=_mirror_keys(hand.finger_kd),
        armature=(finger_armature if hand.finger_armature is None else hand.finger_armature),
        effort_limit_sim=_eff,
        velocity_limit_sim=_vel,
    )
    if hand.coupled_joint_names_right:
        coupled_joints = list(hand.coupled_joint_names_right) + [_to_left(n) for n in hand.coupled_joint_names_right]
        actuators[f"{hand.name}_mimic"] = ImplicitActuatorCfg(
            joint_names_expr=coupled_joints,
            stiffness=mimic_kp,
            damping=mimic_kd,
            armature=mimic_armature,
        )
    return actuators, usd_path
