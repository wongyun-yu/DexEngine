"""**Single source** of the action-contract parameters.

The action contract = the parameters that change how the policy output becomes joint targets. If training and
evaluation differ, **the same checkpoint drives a different robot**: the policy learns on plant A and is scored on
plant B.

Why this module
---------------
The list used to be copied by hand into **four places** (train right, train left, eval right, eval left), and only
9 of its 18 keys had all four filled (action_ema, inner_action_ema, wrist_pos_scale, finger_smooth_weight and
wrist_smooth_weight were missing on both eval sides; the *_end ramp keys and finger_rom_floor on the eval left).
One of those (the left arm's ik_target_ema and wrist_ref_lookahead) had already bitten; the rest were harmless only
because no launcher passed them yet.

Rule: **one value, one place.** All four consumers call this function, and a new contract parameter is one line here.

Byte-identical: every guard (`is not None`, `> 0`) was moved as it was. Keys no launcher passes are left untouched,
so existing runs behave the same.
"""

from dexengine.core.errors import InputMissingError

# 1:1 mapping (args_cli.<key> -> cfg.<key>), assigned when `is not None`.
ACTION_CONTRACT_SIMPLE = (
    "residual_scale",
    "wrist_residual_scale",
    "finger_residual_scale",
    "wrist_pos_residual_max_m",
    "wrist_rot_residual_max_deg",
    # end values of the six residual-axis ramps
    "finger_residual_scale_end",
    "wrist_pos_residual_max_m_end",
    "wrist_rot_residual_max_deg_end",
    "finger_rom_floor",
    "wrist_pos_scale",
)

# Keys that need special handling (see the function body below). Read by the audit script.
ACTION_CONTRACT_SPECIAL = (
    "wrist_ref_lookahead_frames",  # int, default 0, guard > 0
    "ik_target_ema",  # float cast
    "action_ema",  # compound: outer/inner branch + temporal ensemble off
    "finger_effort_anneal_start",  # effort curriculum: start progress + end = the HandSpec hardware cap (function body)
)

ACTION_CONTRACT_KEYS = ACTION_CONTRACT_SIMPLE + ACTION_CONTRACT_SPECIAL


def _hand_specs_module():
    """The hand registry (stdlib-only module)."""
    import dexengine.core.hands as _m

    return _m


def _abduction_regex(hand):
    """HandSpec[hand].abduction_joint_regex, read from the spec file directly (no package import)."""
    _m = _hand_specs_module()
    _spec = _m.HAND_REGISTRY.get(hand) if hand else None
    if _spec is None:
        raise InputMissingError(
            f"[action_contract] hand {hand!r} is not in HAND_REGISTRY — cannot set the ROM-floor exclusion set"
        )
    return getattr(_spec, "abduction_joint_regex", "")


def _hw_finger_effort(hand):
    """Hardware finger cap of hand ``hand`` = core.hands.finger_effort_applied(spec) (a scalar or a joint-regex dict).
    The HandSpec owns the value; this only reads it, from the spec file directly (as _abduction_regex does)."""
    _abduction_regex(hand)  # an unknown hand fails there
    _m = _hand_specs_module()
    return _m.finger_effort_applied(_m.HAND_REGISTRY[hand])


def _apply_effort_anneal(cfg, args_cli, *, label, build_cap_applied, echo, applied) -> None:
    """Effort-curriculum contract key (the one special key of apply_action_contract): puts the start progress, the end
    = the HandSpec hardware cap, and the start cap = the declared build cap plus whether this path loads that cap into
    PhysX, on the action term cfg."""
    # Effort curriculum: start cap = --finger_effort (the build cap, required), end = the HandSpec hardware cap.
    # Evaluation (progress fixed at 1.0) always has the hardware cap, so training, eval and render share the end
    # condition. An ambiguous combination fails.
    _fas = getattr(args_cli, "finger_effort_anneal_start", None)
    if _fas is None or not hasattr(cfg, "finger_effort_anneal_start"):
        return
    _fas = float(_fas)
    if not 0.0 <= _fas < 1.0:
        raise SystemExit(f"[{label}] --finger_effort_anneal_start {_fas} — must be in [0, 1)")
    if getattr(args_cli, "finger_effort", None) is None:
        raise SystemExit(f"[{label}] --finger_effort_anneal_start needs the start cap --finger_effort")
    if _hand_specs_module().finger_effort_scale_from_env() != 1.0:
        raise SystemExit(
            f"[{label}] --finger_effort_anneal_start and DEXENGINE_FINGER_EFFORT_SCALE≠1 give the end cap "
            "two meanings; do not combine them"
        )
    _hand = getattr(args_cli, "hand", None)
    _hw = _hw_finger_effort(_hand)
    _build = float(args_cli.finger_effort)
    _hw_max = max(float(v) for v in (_hw.values() if isinstance(_hw, dict) else [_hw]))
    if not _build > _hw_max:  # a curriculum with nothing to lower is a setup error: fail here whatever the path
        raise InputMissingError(
            f"[{label}] --finger_effort {_build} is not above the hardware cap {_hw} "
            f"(HandSpec[{_hand}]) — the effort curriculum has nothing to lower"
        )
    cfg.finger_effort_anneal_start = _fas
    cfg.finger_effort_hw = _hw
    # The start cap is the declared value, not read back from PhysX: on an eval path that does not load the build cap
    # (rl.eval's default) the initial PhysX cap is the hardware cap, and reading it back failed the boot.
    cfg.finger_effort_build = _build
    cfg.finger_effort_build_applied = bool(build_cap_applied)
    echo(
        f"[{label}] finger_effort_anneal_start = {_fas} (cap {args_cli.finger_effort} → hardware {_hw} "
        f"@progress {_fas}→1.0, HandSpec[{_hand}] · build cap "
        f"{'loaded' if build_cap_applied else 'not loaded (PhysX = hardware cap)'})"
    )
    applied.append(("finger_effort_anneal_start", _fas))


def apply_action_contract(cfg, args_cli, *, label, build_cap_applied, echo=print):
    """Apply the action-contract values to one action term cfg.

    Args:
        cfg:      the action term cfg (right or left arm)
        args_cli: parsed arguments
        label:    log prefix (e.g. "train_cfg_overrides", "eval_cfg_overrides (LEFT)")
        build_cap_applied: whether this path's builder loads ``--finger_effort`` (the build cap) into the finger
                  actuators — a value the caller **decides once** (training = True, eval = DEXENGINE_EVAL_TRAIN_PARITY,
                  the same value as create_env's recipe_build_cap). No default: what the effort curriculum term
                  compares the initial PhysX cap with depends on it.
        echo:     print function
    Returns:
        the (key, value) pairs actually assigned
    """
    applied = []

    for k in ACTION_CONTRACT_SIMPLE:
        v = getattr(args_cli, k, None)
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
            # The wrist residual scale **is ignored once a cap is set**: the cap owns the wrist authority (the comment
            # the original training block carried, moved as is).
            note = ""
            if k == "wrist_residual_scale" and getattr(args_cli, "wrist_pos_residual_max_m", None) is not None:
                note = " (ignored: the --wrist_pos_residual_max_m cap owns the wrist authority)"
            echo(f"[{label}] {k} = {v}{note}")
            applied.append((k, v))

    return _rom_floor_exclude_set(cfg, args_cli, label, build_cap_applied, echo, applied)


def _rom_floor_exclude_set(cfg, args_cli, label, build_cap_applied, echo, applied):
    """Exclude set of the ROM floor: when on, filled with **that hand's abduction joints** from the HandSpec (the old
    action cfg default was wuji joint names and matched nothing on other hands). No declaration fails (it never
    silently floors every joint).
    """
    if getattr(args_cli, "finger_rom_floor", None) is not None and hasattr(cfg, "finger_rom_floor_exclude"):
        _hand = getattr(args_cli, "hand", None)
        _rx = _abduction_regex(_hand)
        if not _rx:
            raise SystemExit(
                f"[{label}] --finger_rom_floor but hand {_hand!r} has an empty "
                "HandSpec.abduction_joint_regex — "
                "declare its abduction joints in the spec"
            )
        cfg.finger_rom_floor_exclude = _rx
        echo(f"[{label}] finger_rom_floor_exclude = {_rx} (HandSpec[{_hand}].abduction_joint_regex)")
        applied.append(("finger_rom_floor_exclude", _rx))

    # WRIST LOOKAHEAD: moves the wrist reference from ref(t) to ref(t+k). A checkpoint trained on ref(t+k) and
    # evaluated on ref(t) fails systematically (measured: frozen 83.3 -> standalone 0.0 on one run before this wiring).
    _wla = int(getattr(args_cli, "wrist_ref_lookahead_frames", 0) or 0)
    if _wla > 0 and hasattr(cfg, "wrist_ref_lookahead_frames"):
        cfg.wrist_ref_lookahead_frames = _wla
        echo(
            f"[{label}] wrist_ref_lookahead_frames = {_wla} "
            f"(wrist reference → ref(t+{_wla}); fingers and observations stay on ref(t))"
        )
        applied.append(("wrist_ref_lookahead_frames", _wla))

    # IK-TARGET EMA: a low-pass on the IK joint targets. The policy steps at 50 Hz, so every action is a step change of
    # the target, and on a stiff arm that step becomes a torque spike (a trembling motion). 1.0 = off.
    _ite = getattr(args_cli, "ik_target_ema", None)
    if _ite is not None and hasattr(cfg, "ik_target_ema"):
        cfg.ik_target_ema = float(_ite)
        echo(f"[{label}] ik_target_ema = {_ite} (IK target low-pass; 1.0 = off)")
        applied.append(("ik_target_ema", float(_ite)))

    # ACTION EMA (compound): ResidualArmIKAction smooths the **combined residual** with the outer cfg.action_ema; its
    # inner_action_ema is asserted == 1.0 (inner EMA bypassed). The old floating-hand action smoothed the inner one
    # instead. It is set on whichever side this action exposes.
    _ae = getattr(args_cli, "action_ema", None)
    if _ae is not None:
        if hasattr(cfg, "action_ema"):
            cfg.action_ema = _ae
            echo(f"[{label}] action_ema (outer, ResidualArmIK) = {_ae}")
            applied.append(("action_ema", _ae))
        else:
            cfg.inner_action_ema = _ae
            applied.append(("inner_action_ema", _ae))
            if hasattr(cfg, "wrist_smooth_weight"):
                cfg.wrist_smooth_weight = -1.0
                cfg.finger_smooth_weight = -1.0
                applied += [("wrist_smooth_weight", -1.0), ("finger_smooth_weight", -1.0)]
            echo(f"[{label}] inner_action_ema={_ae}, temporal ensemble off")

    _apply_effort_anneal(cfg, args_cli, label=label, build_cap_applied=build_cap_applied, echo=echo, applied=applied)
    return applied
