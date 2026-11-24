"""Scene and action edits of ``train_cfg_overrides``: objects, residual action terms, PD gains, object tracking."""

from dexengine.rl.env_setup.object_mass import apply_object_mass
from dexengine.rl.env_setup.objtrack import (
    apply_fingertip_term_disable,
    apply_gt_following,
    apply_judge_objtrack_term,
    apply_obj_track_overrides,
    auto_passive_sides,
    wire_objtrack_split_diags,
)
from dexengine.rl.env_setup.robot_gains import apply_arm_gain_overrides, apply_robot_z_off
from dexengine.rl.tasks.mdp.actions.contract import apply_action_contract


def apply_scene_overrides(
    *, _ACTIVE_OBJ_KEY, _active_side, _launch_ctx, _residual_act_name, args_cli, cfg, rollouts, total_timesteps
):
    """Objects, residual action terms, PD gains, object-tracking terms, object mass and the
    gravity/friction ramps (and the pure-tracking diagnostic)."""
    _object_scene_overrides(_ACTIVE_OBJ_KEY, _active_side, args_cli, cfg)
    _residual_action_overrides(_residual_act_name, args_cli, cfg)
    # CLI overrides for PD gains
    apply_arm_gain_overrides(cfg, args_cli, "train_cfg_overrides")

    # Exp-4 (H2): loosen object_tracking termination so episodes survive past the
    # manipulation onset (the un-grasped object diverges from the moving reference and
    # would otherwise terminate right at contact onset, before a grasp can be learned).
    apply_obj_track_overrides(cfg, args_cli, "train_cfg_overrides")
    # Judge object-tracking termination on/off (dexengine.rl.judge.terms; fresh-process judge only; the standard "on"
    # leaves the cfg unchanged; marker [JUDGE_TERM] objtrack=).
    apply_judge_objtrack_term(cfg, _launch_ctx, "train_cfg_overrides")
    wire_objtrack_split_diags(cfg, args_cli, "train_cfg_overrides")
    apply_fingertip_term_disable(cfg, args_cli, "train_cfg_overrides")

    # Shorter ramp window: finish gravity/friction in the first
    # --gravity_ramp_fraction (default 0.10 = 10%) of training,
    # with start_fraction=0.0 (ramps begin at step 0, not halfway through).
    # This prevents a success cliff observed when a longer
    # ramp (20%, start_fraction=0.5) completed exactly as the policy peaked:
    # the policy learned to exploit low-gravity dynamics, then full gravity
    # arrived and the value function was stale -> destructive PPO updates.
    # Object mass (single source): 0.2 kg default for object + object_left,
    # overridable per side by --object_mass / --object_mass_left. physics_dr_mass then
    # randomizes around this nominal during training.
    apply_object_mass(cfg, args_cli, "train_cfg_overrides")
    apply_robot_z_off(cfg, "train_cfg_overrides")

    _ramp_and_pure_tracking_overrides(_ACTIVE_OBJ_KEY, args_cli, cfg, rollouts, total_timesteps)


def _object_scene_overrides(_ACTIVE_OBJ_KEY, _active_side, args_cli, cfg):
    """Object USDs (right / left / shared), support discs and the start-held table."""
    # Object USD is REQUIRED. No cuboid fallback — fail explicitly so we never
    # silently train/eval on the unintended cuboid placeholder.
    if not args_cli.object_usd:
        raise ValueError(
            "an active object USD is required — refusing to fall back to the cuboid "
            "placeholder. Pass --object_usd_right <path> (or --object_usd_left with --side left)."
        )
    # Directly override the scene object with real USD (bypasses __post_init__).
    from dexengine.rl.env_setup.objects import build_object_entity_cfg as _boec
    from dexengine.rl.env_setup.objects import build_start_held as _build_start_held
    from dexengine.rl.env_setup.supports import build_scene_supports as _bss

    # build_object_entity_cfg decides rigid vs articulated objects; no RigidObjectCfg is built here.
    setattr(cfg.scene, _ACTIVE_OBJ_KEY, _boec(args_cli, "train_cfg_overrides", _active_side, "object_usd"))
    print(f"[train_cfg_overrides] {_ACTIVE_OBJ_KEY} USD: {args_cli.object_usd}")
    print("[train_cfg_overrides]   friction=2.0, disable_gravity=True (ramped via event)")
    # Bimanual: LEFT object USD override + LEFT BPS mesh. Mirror
    # of the right blocks above, re-pointed to ObjectLeft / object_target_left.
    # Gated by --bimanual (off by default).
    if getattr(args_cli, "bimanual", 0) and int(getattr(args_cli, "shared_object", 0) or 0):
        # SHARED OBJECT: both hands hold ONE object -> no second prim. The cfg class still
        # carries the placeholder cuboid `object_left` (left_present=True), so drop it HERE (before
        # apply_object_mass logs it); _apply_shared_object (the LAST statement below) re-points every
        # left-side term/sensor to object_right and re-verifies.
        cfg.scene.object_left = None
        print(
            f"[train_cfg_overrides] [shared_object] LEFT object prim NOT built — one prim ObjectRight "
            f"(USD {args_cli.object_usd}); left terms re-pointed at the end of overrides",
            flush=True,
        )
    elif getattr(args_cli, "bimanual", 0):
        if not args_cli.object_usd_left:
            raise ValueError("--bimanual 1 requires --object_usd_left <path> (the LEFT object USD).")
        # The LEFT spawn goes through the SAME builder as the right,
        # so --multi_object (per-clip LEFT objects) works on both sides. The
        # builder abspaths and existence-checks every comma token itself.
        usd_path_left = args_cli.object_usd_left
        cfg.scene.object_left = _boec(args_cli, "train_cfg_overrides[left]", "left", "object_usd_left")
        print(f"[train_cfg_overrides] (left_present) LEFT object USD: {usd_path_left}")
    _bss(cfg, args_cli, "train_cfg_overrides")  # support discs (mid-air scenes, desk supports), same as the eval cfg
    _build_start_held(cfg, args_cli, "train_cfg_overrides")  # start-held objects of the probe build (probe only)


def _residual_action_overrides(_residual_act_name, args_cli, cfg):
    """The residual action terms: right, then left (bimanual)."""
    # Set base checkpoint path in the residual action config.
    # LEFT-UNI: the single residual term is 'left_hand_joint_pos' — resolve the
    # name via _residual_act_name() so this WHOLE nested block (base ckpt,
    # scale/EMA/cap CLI, reference_base, ReferenceBaseAction obs swap) applies
    # to the left term. A hardcoded right-name hasattr gate would silently skip
    # it all (risk #1 of the L-uni verify checklist).
    _act_nm = _residual_act_name()
    if hasattr(cfg.actions, _act_nm):
        act = getattr(cfg.actions, _act_nm)
        _apply_inner_action_overrides(args_cli, cfg, act)
    else:
        print(f"[train_cfg_overrides] WARNING: no {_act_nm} in actions")

    # Bimanual: apply the SAME per-term knobs to the LEFT residual
    # action term (left_hand_joint_pos): residual
    # scales, EMA, reference_base. Gated by --bimanual (off by default).
    # The left term shares the same CLI scales as the right (per-hand
    # regularizers preserve each hand's stability point; the action SCALES
    # are intentionally identical to the right).
    if args_cli.bimanual and hasattr(cfg.actions, "left_hand_joint_pos"):
        actL = cfg.actions.left_hand_joint_pos
        # Same action-contract key list as the right side.
        apply_action_contract(actL, args_cli, label="train_cfg_overrides (LEFT)", build_cap_applied=True)
        if getattr(args_cli, "left_gt_following", 0):
            # LEFT = pure reference execution: zero ALL left residual authority. The left
            # tower still samples actions but they cannot move the hand; the left residual-l2
            # anchor then drives its outputs (and std) toward 0 = exploration noise removed.
            actL.finger_residual_scale = 0.0
            if hasattr(actL, "wrist_residual_scale"):
                actL.wrist_residual_scale = 0.0
            if hasattr(actL, "wrist_pos_residual_max_m"):
                actL.wrist_pos_residual_max_m = 0.0
            if hasattr(actL, "wrist_rot_residual_max_deg"):
                actL.wrist_rot_residual_max_deg = 0.0
            print(
                "[train_cfg_overrides] (left_present) LEFT GT-FOLLOWING: all left residual scales/caps = 0 "
                "(pure reference execution; left tower inert)"
            )
        if getattr(args_cli, "auto_passive_hand", 0):
            _passive = auto_passive_sides(args_cli)
            if "left" in _passive:
                apply_gt_following(actL, "(auto-passive left)", "train_cfg_overrides")
            if "right" in _passive:
                apply_gt_following(act, "(auto-passive right)", "train_cfg_overrides")
        if args_cli.use_trajectory_ref:
            actL.use_trajectory_reference = True
        if args_cli.reference_base and hasattr(actL, "reference_base"):
            actL.reference_base = True
            actL.use_trajectory_reference = True
            print("[train_cfg_overrides] (left_present) LEFT reference_base = True, use_trajectory_reference = True")
        # CLEAN single-stage: swap the LEFT obs 'left_last_base_action' func
        # to ReferenceBaseAction (mirror of the right swap).
        if args_cli.reference_base:
            from dexengine.rl.tasks.mdp.terms import ReferenceBaseAction as _RBA_L

            _polL = getattr(cfg.observations, "policy", None)
            if _polL is not None and hasattr(_polL, "left_last_base_action"):
                _polL.left_last_base_action.func = _RBA_L
                print(
                    "[train_cfg_overrides] (bimanual) CLEAN single-stage: obs "
                    "'left_last_base_action' func -> ReferenceBaseAction"
                )
            else:
                print("[train_cfg_overrides] (bimanual) WARNING: no policy.left_last_base_action to swap")


def _apply_inner_action_overrides(args_cli, cfg, act):
    """CLI overrides for the inner action term (must match training) The action contract (one key list shared by train
    and eval, right and left) applies the CLI overrides of the inner action term.
    """
    apply_action_contract(act, args_cli, label="train_cfg_overrides", build_cap_applied=True)
    # Diagnostic: force trajectory-reference mode
    if args_cli.use_trajectory_ref:
        act.use_trajectory_reference = True
        print("[train_cfg_overrides] use_trajectory_reference = True (diagnostic override)")
    # reference-base: base action = kinematic reference (wrist+fingers)
    # (kinematic reference base); forces use_trajectory_reference on.
    if args_cli.reference_base and hasattr(act, "reference_base"):
        act.reference_base = True
        act.use_trajectory_reference = True
        print("[train_cfg_overrides] reference_base = True, use_trajectory_reference = True")
    # Single-stage (reference-base + residual): swap the legacy
    # the removed frozen-base obs for ReferenceBaseAction
    # (the reference base action; gated by reference_base).
    if args_cli.reference_base:
        from dexengine.rl.tasks.mdp.terms import ReferenceBaseAction as _RBA

        _pol = getattr(cfg.observations, "policy", None)
        if _pol is not None and hasattr(_pol, "last_base_action"):
            _pol.last_base_action.func = _RBA
            print(
                "[train_cfg_overrides] CLEAN single-stage: obs 'last_base_action' func "
                "-> ReferenceBaseAction (reference base, no base model/checkpoint)"
            )
        else:
            print("[train_cfg_overrides] WARNING: no policy.last_base_action obs term to swap")


def _ramp_and_pure_tracking_overrides(_ACTIVE_OBJ_KEY, args_cli, cfg, rollouts, total_timesteps):
    """Gravity / friction ramps, the bimanual ramp mirror and pure tracking."""
    # gravity_ramp + friction_decay ramp duration = fraction-of-total-timesteps
    # (default 0.10 via --gravity_ramp_fraction). Resolved here because
    # total_timesteps (= max_iterations * rollouts, the common_step_counter clock)
    # is only known at train time; the cfg-build builder default is overridden.
    _grf = float(getattr(args_cli, "gravity_ramp_fraction", 0.10))
    ramp_steps = int(total_timesteps * _grf)
    # CLI override for gravity ramp end iteration (absolute iter count).
    # Takes precedence over --gravity_ramp_fraction.
    if args_cli.gravity_ramp_end_iter is not None:
        ramp_steps = int(args_cli.gravity_ramp_end_iter) * rollouts
    if hasattr(cfg.events, "gravity_ramp"):
        cfg.events.gravity_ramp.params["ramp_steps"] = ramp_steps
        cfg.events.gravity_ramp.params["start_fraction"] = 0.0
        if args_cli.gravity_ramp_end_iter is not None:
            print(
                f"[train_cfg_overrides] gravity_ramp.ramp_steps = {ramp_steps} "
                f"(reaches 100% at iter {args_cli.gravity_ramp_end_iter}), start_fraction=0.0"
            )
        else:
            print(
                f"[train_cfg_overrides] gravity_ramp.ramp_steps = {ramp_steps} "
                f"({int(_grf * 100)}% of {total_timesteps}), start_fraction=0.0"
            )
        if args_cli.disable_gravity_ramp:
            cfg.events.gravity_ramp.params["ramp_steps"] = 0
            cfg.events.gravity_ramp.params["start_fraction"] = 1.0
            print("[train_cfg_overrides] DISABLE GRAVITY RAMP: start_fraction=1.0, ramp_steps=0")
    if hasattr(cfg.events, "friction_decay"):
        cfg.events.friction_decay.params["ramp_steps"] = ramp_steps
        print(
            f"[train_cfg_overrides] friction_decay.ramp_steps = {ramp_steps} ({int(_grf * 100)}% of {total_timesteps})"
        )
        if args_cli.disable_friction_decay:
            final_mult = cfg.events.friction_decay.params.get("final_multiplier", 1.0)
            cfg.events.friction_decay.params["initial_multiplier"] = final_mult
            cfg.events.friction_decay.params["ramp_steps"] = 0
            print(f"[train_cfg_overrides] DISABLE FRICTION DECAY: initial=final={final_mult}, ramp_steps=0")

    _mirror_left_ramps(_ACTIVE_OBJ_KEY, args_cli, cfg, ramp_steps)


def _mirror_left_ramps(_ACTIVE_OBJ_KEY, args_cli, cfg, ramp_steps):
    """Bimanual: mirror the gravity_ramp / friction_decay schedule onto the LEFT object's events so both objects share
    the same early curriculum. Gated by --bimanual.
    """
    if getattr(args_cli, "bimanual", 0):
        if hasattr(cfg.events, "gravity_ramp_left"):
            cfg.events.gravity_ramp_left.params["ramp_steps"] = ramp_steps
            cfg.events.gravity_ramp_left.params["start_fraction"] = 0.0
            if args_cli.disable_gravity_ramp:
                cfg.events.gravity_ramp_left.params["ramp_steps"] = 0
                cfg.events.gravity_ramp_left.params["start_fraction"] = 1.0
            print(f"[train_cfg_overrides] (bimanual) gravity_ramp_left.ramp_steps = {ramp_steps}")
        if hasattr(cfg.events, "friction_decay_left"):
            cfg.events.friction_decay_left.params["ramp_steps"] = ramp_steps
            if args_cli.disable_friction_decay:
                _fml = cfg.events.friction_decay_left.params.get("final_multiplier", 1.0)
                cfg.events.friction_decay_left.params["initial_multiplier"] = _fml
                cfg.events.friction_decay_left.params["ramp_steps"] = 0
            print(f"[train_cfg_overrides] (bimanual) friction_decay_left.ramp_steps = {ramp_steps}")

    # Diagnostic: pure tracking mode (zero out object-related rewards)
    if args_cli.pure_tracking:
        _apply_pure_tracking_rewards(cfg)
        if hasattr(cfg, "scene") and getattr(cfg.scene, _ACTIVE_OBJ_KEY, None) is not None:
            # Move object spawn 10m below ground — guaranteed no collision
            getattr(cfg.scene, _ACTIVE_OBJ_KEY).init_state.pos = (0.0, 0.0, -10.0)
            print("[train_cfg_overrides] PURE TRACKING: object spawn moved to (0,0,-10)")

        print(
            "[train_cfg_overrides] PURE TRACKING mode enabled: only wrist+finger tracking rewards active, object "
            "physically removed"
        )


def _apply_pure_tracking_rewards(cfg):
    """Pure tracking: the object reward weights go to 0 and the object terminations and events are removed."""
    if hasattr(cfg, "rewards"):
        # Object rewards: obj_pos, obj_rot, obj_vel, obj_ang_vel, fingertip_force
        for rw_name in ["obj_pos", "obj_rot", "obj_vel", "obj_ang_vel", "fingertip_force"]:
            rw = getattr(cfg.rewards, rw_name, None)
            if rw is not None:
                rw.weight = 0.0
                print(f"[train_cfg_overrides] PURE TRACKING: zeroed reward {rw_name}")
    if hasattr(cfg, "terminations"):
        # Object terminations: object_tracking, object_velocity
        # Fully disable (set to None) — relaxing thresholds is not enough
        # because some terminations may fire on other criteria.
        for term_name in ["object_tracking", "object_velocity"]:
            term = getattr(cfg.terminations, term_name, None)
            if term is not None:
                setattr(cfg.terminations, term_name, None)
                print(f"[train_cfg_overrides] PURE TRACKING: disabled termination {term_name}")

    # Physically remove the object from interfering with the hand:
    # 1. Disable reset_object event (object stays at initial spawn, not at trajectory pose)
    # 2. Disable gravity_ramp event (no force applied → object stays put)
    # 3. Disable friction_decay event (defensive)
    # 4. Move object spawn far below ground (zero collision risk with hand)
    # The object remains in the scene so the object_state obs stays valid,
    # but with residual_scale=0 the network sees garbage and outputs are zeroed anyway.
    if hasattr(cfg, "events"):
        for ev_name in ["reset_object", "gravity_ramp", "friction_decay"]:
            ev = getattr(cfg.events, ev_name, None)
            if ev is not None:
                # Set mode to a sentinel that the manager will not recognize
                # → easiest is to clear it from the cfg via setattr None
                setattr(cfg.events, ev_name, None)
                print(f"[train_cfg_overrides] PURE TRACKING: disabled event {ev_name}")
