"""Robot overrides of the env cfg: arm gains, robot height offset and hand friction."""

from __future__ import annotations

import torch

from dexengine.rl.tasks import env_flags, switches


def apply_arm_gain_overrides(cfg, args_cli, ctx: str) -> None:
    """Apply --arm_kp/--arm_kd (scalar) and --arm_kp_per_joint/--arm_kd_per_joint
    (one comma-separated value per arm joint, in index order, BOTH arms via a side-wildcard regex) to the
    arm actuator group ("<arm>_arm").

    SHARED by train / eval so the paths can never drift
    (if train applied --arm_kp and eval silently did not, a frozen
    replay and a standalone eval of one checkpoint would disagree). Per-joint takes precedence over scalar.
    Isaac Lab ActuatorBaseCfg accepts dict[str(regex), float] gains
    (_parse_joint_parameter).
    """
    if not (hasattr(cfg, "scene") and hasattr(cfg.scene, "robot")):
        return
    # Floating hand: no arm joint PD. An arm-gain input fails; without one the 'applied 0' marker prints
    # (rule = floating_hand, train and eval).
    from dexengine.rl.tasks import hooks as _hooks_g

    if _hooks_g.floating_active():
        _fh_g = _hooks_g.floating_hand()
        _ents = (cfg.scene.robot, getattr(cfg.scene, "robot_left", None))
        _keys = [k for _e in _ents for k in getattr(_e, "actuators", {})]
        print(
            _fh_g.require_no_arm_gains(args_cli.arm_gains_given, switches.legacy_environ(), ctx, actuator_keys=_keys),
            flush=True,
        )
        return
    # The actuator group name differs per arm ("<arm>_arm"; rby1 has rby1_arm). A hard-coded name would raise
    # KeyError on another arm.
    _acts = cfg.scene.robot.actuators
    _arm_keys = [k for k in _acts if k.endswith("_arm")]
    if len(_arm_keys) != 1:
        raise KeyError(
            f"cannot pin down one arm actuator group: {sorted(_acts)} "
            "— if the '<arm>_arm' convention broke, the gain override would silently miss."
        )
    _act = _acts[_arm_keys[0]]
    if args_cli.arm_kp is not None:
        _act.stiffness = args_cli.arm_kp
        print(f"[{ctx}] arm_kp = {args_cli.arm_kp}")
    if args_cli.arm_kd is not None:
        _act.damping = args_cli.arm_kd
        print(f"[{ctx}] arm_kd = {args_cli.arm_kd}")
    _apply_left_arm_gains(cfg, args_cli, ctx, _act)


def _apply_left_arm_gains(cfg, args_cli, ctx, _act):
    """Split-articulation eval: the left arm is a separate robot_left articulation with its own gains."""
    # GATED split-articulation eval: the LEFT arm is a SEPARATE `robot_left` articulation
    # with its own arm actuator group. Apply the SAME scalar arm_kp/arm_kd
    # so the left arm PD matches the shared-articulation baseline (where both arms get the
    # CLI gains). Per-joint/left-scale env-var paths above only touch `robot` (right).
    # The left group name differs per arm: the legacy pair uses "<arm>_arm_left", a split combo the same "<arm>_arm"
    # as the right (a separate cfg, so no clash). Found by the suffix convention, not a literal (a literal once made
    # the left arm silently miss the CLI gains on the rby1 split).
    _robot_left = getattr(cfg.scene, "robot_left", None)
    _acts_l = getattr(_robot_left, "actuators", {}) if _robot_left is not None else {}
    _arm_keys_l = [k for k in _acts_l if k.endswith(("_arm", "_arm_left"))]
    if len(_arm_keys_l) > 1:
        raise KeyError(f"cannot pin down one left arm actuator group: {sorted(_acts_l)}")
    if _arm_keys_l:
        _act_l = _acts_l[_arm_keys_l[0]]
        if args_cli.arm_kp is not None:
            _act_l.stiffness = args_cli.arm_kp
        if args_cli.arm_kd is not None:
            _act_l.damping = args_cli.arm_kd
        print(f"[{ctx}] (split-artic) LEFT arm_kp/kd applied to robot_left.{_arm_keys_l[0]}")
    _pj_kp = getattr(args_cli, "arm_kp_per_joint", None)
    _pj_kd = getattr(args_cli, "arm_kd_per_joint", None)
    if _pj_kp or _pj_kd:
        # Per-joint keys come from the active arm spec: joint index i on both sides (side wildcard), the same
        # semantics as the old literal regex of the removed legacy arm, which matched no other arm's joints.
        from dexengine.robots import selection as _rs_pj

        _arm_pj, _ = _rs_pj.resolve_robot_selection()
        _idx = range(_arm_pj.index_base, _arm_pj.index_base + _arm_pj.n_joints)
        _keys = [_arm_pj.joint_name_fmt.format(prefix=_arm_pj.joint_prefix, side=".*", index=i) for i in _idx]
    if _pj_kp:
        _v = [float(x) for x in str(_pj_kp).split(",")]
        assert len(_v) == len(_keys), f"--arm_kp_per_joint needs {len(_keys)} values (one per arm joint), got {len(_v)}"
        _act.stiffness = dict(zip(_keys, _v, strict=True))
        print(f"[{ctx}] arm_kp PER-JOINT = {_v} ({_keys[0]}..{_keys[-1]}, both arms; overrides scalar)")
    if _pj_kd:
        _v = [float(x) for x in str(_pj_kd).split(",")]
        assert len(_v) == len(_keys), f"--arm_kd_per_joint needs {len(_keys)} values (one per arm joint), got {len(_v)}"
        _act.damping = dict(zip(_keys, _v, strict=True))
        print(f"[{ctx}] arm_kd PER-JOINT = {_v} ({_keys[0]}..{_keys[-1]}, both arms; overrides scalar)")
    # LEFT-arm gain scale (env-gated diagnostic): DEXENGINE_LEFT_ARM_KP_SCALE=k sets the
    # LEFT arm stiffness to k x the resolved scalar (right unchanged). Tests whether SOFTENING the
    # left arm (less PD reaction torque into the shared articulation) reduces the +25 Nm right-shoulder
    # coupling and recovers fragile bi clips -> coupling is left-torque-mediated (removable).
    _las = env_flags.get("DEXENGINE_LEFT_ARM_KP_SCALE")
    # Only a scale k != 1.0 touches the actuator cfg. The joint regexes come from the arm spec (a legacy literal
    # matched no rby1 joint and Isaac Lab refused to boot: "Not all regular expressions are matched!"), and
    # launch.plan always sets k = 1.0: rewriting the actuator cfg from a scalar to a dict for that no-op is what
    # triggered it. So k == 1.0 leaves the cfg alone.
    if _las and abs(float(_las) - 1.0) > 1e-9:
        _k = float(_las)
        _base_kp = args_cli.arm_kp if args_cli.arm_kp is not None else 200.0
        _base_kd = args_cli.arm_kd if args_cli.arm_kd is not None else 30.0
        from dexengine.robots import selection as _rs_la

        _arm_la, _ = _rs_la.resolve_robot_selection()  # never None: an unregistered selection raises there
        _er, _el = _arm_la.joint_expr("right"), _arm_la.joint_expr("left")
        _act.stiffness = {_er: _base_kp, _el: _base_kp * _k}
        _act.damping = {_er: _base_kd, _el: _base_kd * _k}
        print(
            f"[{ctx}] LEFT ARM KP SCALE x{_k}: left kp={_base_kp * _k} (right kp={_base_kp}); "
            f"joint_expr right={_er!r} left={_el!r}; coupling-source diagnostic"
        )
    elif _las:
        print(
            f"[{ctx}] LEFT ARM KP SCALE={_las} == 1.0 -> no-op, the actuator cfg is left alone "
            f"(rewriting it as a dict would not match the combo joint names and fail the boot)"
        )


def apply_robot_z_off(cfg, ctx_label):
    """DEXENGINE_ROBOT_Z_OFF=<m> raises/lowers the ROBOT mount
    height (root xy fixed). Applied to cfg.scene.robot.init_state.pos
    BEFORE env build; pairs with the arm IK cache built at that z offset for the RSI reset.
    Per-seq env var (like DEXENGINE_DESK_ROBOT_NOCOLL). Unset/0 -> byte-identical."""
    _z = float(env_flags.get("DEXENGINE_ROBOT_Z_OFF") or 0)
    if not _z:
        return
    from dexengine.rl.tasks.scene import split_artic

    _r = getattr(cfg.scene, "robot", None)
    if _r is None or getattr(_r, "init_state", None) is None:
        print(f"[{ctx_label}] DEXENGINE_ROBOT_Z_OFF set but no scene.robot.init_state — skipped")
        return
    # every robot articulation of the scene (with the split, robot_left is a second fixed base): one height for both
    for key in split_artic.ROBOT_KEYS:
        art = getattr(cfg.scene, key, None)
        if art is None:
            continue
        p = art.init_state.pos
        art.init_state.pos = (p[0], p[1], p[2] + _z)
        print(f"[{ctx_label}] ROBOT Z OFFSET {_z:+.3f} m: {key} init pos {tuple(p)} -> {art.init_state.pos}")


# ======================================================================
def disable_robot_gravity(cfg, ctx_label) -> list[str]:
    """Gravity off on every robot articulation of the scene cfg (split_artic.ROBOT_KEYS: with the split, robot_left
    too) — the DEXENGINE_DISABLE_ROBOT_GRAVITY audit (ideal gravity compensation). Returns the entities changed."""
    from dexengine.rl.tasks.scene import split_artic

    done = []
    for key in split_artic.ROBOT_KEYS:
        art = getattr(cfg.scene, key, None)
        if art is not None:
            art.spawn.rigid_props.disable_gravity = True
            done.append(key)
    print(f"[{ctx_label}] DEXENGINE_DISABLE_ROBOT_GRAVITY=1: robot gravity OFF (ideal gravity comp) on {done}.")
    return done


def apply_arm_gain_override(raw_env):
    """Apply DEXENGINE_ARM_KP / DEXENGINE_ARM_KD to the live arm PhysX drive gains (RENDER/AUDIT ONLY,
    env-var gated). The eval cfg carries the arm spec's arm_drive; this overrides the live drive so a
    gain-sweep render/audit reflects the chosen gains. ImplicitActuator -> PhysX does the PD; the
    write persists (reset is a no-op). Training/eval configs are untouched. No env var => no-op.
    Returns (kp, kd) effective on the arm, or (None, None)."""
    kp_env = env_flags.get("DEXENGINE_ARM_KP")
    kd_env = env_flags.get("DEXENGINE_ARM_KD")
    from dexengine.rl.tasks import hooks as _hooks_o

    if _hooks_o.floating_active():
        _fh_o = (
            _hooks_o.floating_hand()
        )  # floating hand: no arm joints; an arm-gain env input fails, none = 'applied 0'
        print(_fh_o.require_no_arm_gains((), switches.legacy_environ(), "arm_gain_override"), flush=True)
        return None, None
    robot = raw_env.scene["robot"] if "robot" in raw_env.scene.keys() else None  # noqa: SIM118  — InteractiveScene has no __contains__
    # (the helper is arm_joint_expr below: the active arm spec's joint regex.)
    if robot is None:
        _t = raw_env.action_manager._terms.get("right_hand_joint_pos")
        robot = getattr(getattr(_t, "_inner", None), "_asset", None)
    if robot is None:
        print("[arm_gain_override] WARN: could not resolve arm articulation; skipped")
        return None, None
    # GATED split-articulation eval: `robot` holds ONLY the right arm (the left arm is a
    # separate `robot_left` articulation), so a two-sided regex would fail the strict
    # find_joints match. Query each side against its own articulation and apply per-view.
    # Joint names differ per arm; the ArmSpec is canonical (a hard-coded name would make eval/render die in
    # find_joints after training had passed, which looks like a 0 completion rate).
    _ax = arm_joint_expr()
    from dexengine.rl.tasks.scene import split_artic as _split_artic

    _split = _split_artic.enabled()
    if _split:
        _targets = [(robot, [_ax("right")])]
        _left = raw_env.scene["robot_left"] if "robot_left" in raw_env.scene.keys() else None  # noqa: SIM118  — InteractiveScene has no __contains__
        if _left is not None:
            _targets.append((_left, [_ax("left")]))
    else:
        _targets = [(robot, [_ax("left"), _ax("right")])]

    kp0 = kd0 = kp1 = kd1 = None
    _n = 0
    for _rob, _regex in _targets:
        arm_ids, _ = _rob.find_joints(_regex)
        aidt = torch.tensor(arm_ids, device=_rob.device, dtype=torch.long)
        _kp0 = float(_rob.data.joint_stiffness[0, aidt].mean().item())
        _kd0 = float(_rob.data.joint_damping[0, aidt].mean().item())
        if kp0 is None:
            kp0, kd0 = _kp0, _kd0
        if kp_env:
            _rob.write_joint_stiffness_to_sim(float(kp_env), joint_ids=arm_ids)
        if kd_env:
            _rob.write_joint_damping_to_sim(float(kd_env), joint_ids=arm_ids)
        kp1 = float(_rob.data.joint_stiffness[0, aidt].mean().item())
        kd1 = float(_rob.data.joint_damping[0, aidt].mean().item())
        _n += len(arm_ids)
    if not (kp_env or kd_env):
        return kp0, kd0
    print(f"[arm_gain_override] {_n} arm joints: Kp {kp0:.0f}->{kp1:.0f}, Kd {kd0:.1f}->{kd1:.1f}")
    return kp1, kd1


def arm_joint_expr():
    """side -> joint regex of the active arm spec. The ArmSpec is the only owner.

    Hard-coded joint names here would make eval/render die in find_joints after training had passed (it looks like
    a 0 completion rate). Called only once the env is up, so the package import is safe. Without an active selection
    resolve_robot_selection fails (no legacy-arm fallback)."""
    from dexengine.robots.selection import resolve_robot_selection

    _spec, _ = resolve_robot_selection()
    return lambda side: _spec.joint_expr(side)


def apply_hand_friction(raw_env, friction):
    """Set the collision static + dynamic friction of every robot articulation at runtime (train + eval).

    The robot ArticulationCfg sets no physics_material, so the shapes keep the USD-baked value (not matched to the
    object's 2.0). This writes one fixed friction to every collision shape of every robot articulation of the scene
    (split_artic.robot_articulations: ``robot`` and, with the split, ``robot_left``) through the PhysX view (persists
    across resets; the object-only friction DR does not touch it), reads it back and prints the shapes written against
    the shapes there are — fewer dies. No-op if friction is None. Mirrors randomize_object_friction."""
    if friction is None:
        return
    from dexengine.rl.tasks.scene import split_artic

    arts = split_artic.robot_articulations(raw_env)
    if not arts:
        t = raw_env.action_manager._terms.get("right_hand_joint_pos")
        asset = getattr(getattr(t, "_inner", None), "_asset", None)
        arts = [("right_hand_joint_pos asset", asset)] if asset is not None else []
    if not arts:
        print("[hand_friction] WARN: could not resolve robot articulation; skipped")
        return
    f, n_env, parts, written = float(friction), raw_env.num_envs, [], 0
    for name, art in arts:
        view = art.root_physx_view
        mat = view.get_material_properties().clone()  # (n_env, n_shapes, 3) CPU
        mat[:, :, 0] = f  # static
        mat[:, :, 1] = f  # dynamic
        view.set_material_properties(mat, torch.arange(n_env, device="cpu"))
        back = view.get_material_properties()
        ok = int(((back[:, :, 0] == f) & (back[:, :, 1] == f)).all(dim=0).sum())  # shapes set in every env
        parts.append(f"{name} {ok}/{view.max_shapes}")
        written += ok
    # the shapes there are: every robot entity the scene holds, counted apart from the list written to
    try:
        keys = list(raw_env.scene.keys())  # noqa: SIM118 — InteractiveScene has no __contains__
    except Exception:  # noqa: BLE001 — no interactive scene: the resolved asset is all there is
        keys = []
    held = [k for k in split_artic.ROBOT_KEYS if k in keys]
    total = (
        sum(int(raw_env.scene[k].root_physx_view.max_shapes) for k in held)
        if held
        else sum(int(a.root_physx_view.max_shapes) for _, a in arts)
    )
    print(
        f"[hand_friction] robot collision friction set to {friction} (static+dynamic) on {written} of {total} shapes "
        f"({' · '.join(parts)}) x {n_env} envs."
    )
    if written != total:
        raise RuntimeError(f"[hand_friction] {written} of {total} robot shapes read back friction {friction}: {parts}")
