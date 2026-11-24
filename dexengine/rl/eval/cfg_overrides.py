"""Env-cfg overrides of ``--task eval`` (the ``extra_cfg_fn`` that ``create_env`` applies before building the env).

They mirror the training overrides so the evaluated or replayed scene equals the trained one (action contract, arm
gains, object assets, mass, observations, reward terms) and then freeze everything the training run ramps or
randomises: no threshold curriculum, gravity and friction pinned at their final values, no physics domain
randomisation unless an ``--enable_physics_dr`` mode is asked for, reset-pose randomisation neutralised.

Isaac-side (imports ``dexengine.rl.env_setup.build``): import after the AppLauncher, as ``eval.run`` does.
"""

from dexengine.rl.env_setup.build import _apply_shared_object
from dexengine.rl.env_setup.object_mass import apply_object_mass
from dexengine.rl.env_setup.object_rewards import add_obj_keypoint_terms
from dexengine.rl.env_setup.objtrack import (
    apply_fingertip_term_disable,
    apply_gt_following,
    apply_obj_track_overrides,
    auto_passive_sides,
    verify_objtrack_schedule,
    wire_objtrack_split_diags,
)
from dexengine.rl.env_setup.robot_gains import apply_arm_gain_overrides, apply_robot_z_off, disable_robot_gravity
from dexengine.rl.env_setup.sides import apply_idle_side, apply_left_uni, apply_obs_v2
from dexengine.rl.tasks import switches
from dexengine.rl.tasks.mdp.actions.contract import apply_action_contract
from dexengine.rl.train.overrides_physics import PHYSICS_DR_RANGES


def make_eval_cfg_overrides(args_cli, _ACTIVE_OBJ_KEY, _active_side, _residual_act_name, _eval_train_parity):
    """Return the eval ``extra_cfg_fn(cfg)``.

    args_cli: the trainer CLI namespace · _ACTIVE_OBJ_KEY: scene key of the active object · _active_side: side of
    the policy hand · _residual_act_name: callable giving the residual action term name · _eval_train_parity:
    whether the training-only finger gains and reset-pose parameters are mirrored (DEXENGINE_EVAL_TRAIN_PARITY set to
    1).
    """

    def _eval_cfg_overrides_bound(cfg):
        return _eval_cfg_overrides(cfg, args_cli, _ACTIVE_OBJ_KEY, _active_side, _residual_act_name, _eval_train_parity)

    return _eval_cfg_overrides_bound


def _eval_cfg_overrides(cfg, args_cli, _ACTIVE_OBJ_KEY, _active_side, _residual_act_name, _eval_train_parity):
    # Remove the threshold-tightening curriculum entirely at eval; a post-env-creation hook sets
    # trajectory_manager.schedule_progress (final/tight thresholds by default).
    if hasattr(cfg, "curriculum") and hasattr(cfg.curriculum, "threshold_tightening"):
        cfg.curriculum.threshold_tightening = None
        print("[eval_cfg_overrides] threshold_tightening curriculum REMOVED")

    # Object-tracking thresholds as at the end of training.
    apply_obj_track_overrides(cfg, args_cli, "eval_cfg_overrides")
    # Eval pins schedule_progress (threshold_tightening removed above + post-env hook),
    # so the effective thresholds are the fixed *_end values. Verify loudly.
    verify_objtrack_schedule(cfg, "eval_cfg_overrides", frozen=True)
    wire_objtrack_split_diags(cfg, args_cli, "eval_cfg_overrides")
    apply_fingertip_term_disable(cfg, args_cli, "eval_cfg_overrides")

    _eval_action_contract(cfg, args_cli, _residual_act_name, _eval_train_parity)
    _eval_scene_and_ramps(cfg, args_cli, _ACTIVE_OBJ_KEY, _active_side)

    _eval_physics_dr(cfg, args_cli)

    _eval_obs_rewards_and_parity(cfg, args_cli, _eval_train_parity)

    # Keypoint reward/diag terms (eval mirror of train_cfg_overrides).
    # optimize=True  (--obj_keypoint_reward 1): adds MAIN terms so the standalone
    #   capture's obj_pos×rot×keypoint 3-way product can include them.
    # optimize=False (keypointremoval): adds DIAG-only terms (weight 1e-9) so the
    #   keypoint error metric is recorded for ablation comparison.
    # extend_ramp=False: eval freezes the curriculum; schedule extension is a no-op.
    add_obj_keypoint_terms(
        cfg,
        args_cli,
        optimize=(int(getattr(args_cli, "obj_keypoint_reward", 0)) == 1),
        extend_ramp=False,
    )
    # Articulated clips: joint-value tracking reward, same owner as training (add_obj_artic_terms), no ramp.
    from dexengine.rl.env_setup.object_rewards import add_obj_artic_terms as _aoat

    _aoat(cfg, args_cli, "eval_cfg_overrides", extend_ramp=False)

    # Left-hand side sweep — must stay the LAST statements (as in training): every runtime term above,
    # including the keypoint block, is registered before the sweep flips the side right -> left.
    apply_left_uni(cfg, args_cli, "eval_cfg_overrides")
    # Shared-object sweep (training/eval parity; no-op unless --shared_object).
    _apply_shared_object(cfg, args_cli, "eval_cfg_overrides")
    # Idle-side sweep (training/eval parity; no-op unless an idle side is set).
    apply_idle_side(cfg, args_cli, "eval_cfg_overrides")


def _eval_action_contract(cfg, args_cli, _residual_act_name, _eval_train_parity):
    """The residual action term at eval: base checkpoint, scales and the reference mode."""
    # The residual action term name ('right_hand_joint_pos', or 'left_hand_joint_pos' for a left-hand build), so
    # this block (base checkpoint, scales, reference_base, observation swap) applies at eval exactly as in training.
    _act_nm_e = _residual_act_name()
    if hasattr(cfg, "actions") and hasattr(cfg.actions, _act_nm_e):
        action_cfg = getattr(cfg.actions, _act_nm_e)
        # The action contract has one owner (apply_action_contract): training and eval, right and left, all run
        # the same list of keys.
        apply_action_contract(action_cfg, args_cli, label="eval_cfg_overrides", build_cap_applied=_eval_train_parity)
        if args_cli.use_trajectory_ref:
            action_cfg.use_trajectory_reference = True
            print("[eval_cfg_overrides] use_trajectory_reference = True")
        # reference-base: base = kinematic reference (wrist+fingers); needed for eval/render parity.
        if getattr(args_cli, "reference_base", 0) and hasattr(action_cfg, "reference_base"):
            action_cfg.reference_base = True
            action_cfg.use_trajectory_reference = True
            print("[eval_cfg_overrides] reference_base = True, use_trajectory_reference = True")

        # Bimanual: apply the same per-term settings to the LEFT residual action term (scales, reference_base)
        # and swap the LEFT observation. Only with --bimanual.
        if args_cli.bimanual and hasattr(cfg.actions, "left_hand_joint_pos"):
            action_cfg_L = cfg.actions.left_hand_joint_pos
            _eval_left_action_contract(args_cli, _eval_train_parity, action_cfg, action_cfg_L)
            if getattr(args_cli, "reference_base", 0):
                from dexengine.rl.tasks.mdp.terms import ReferenceBaseAction as _RBA_LE

                _polLE = getattr(cfg.observations, "policy", None)
                if _polLE is not None and hasattr(_polLE, "left_last_base_action"):
                    _polLE.left_last_base_action.func = _RBA_LE
                    print(
                        "[eval_cfg_overrides] (bimanual) CLEAN single-stage: obs "
                        "'left_last_base_action' func -> ReferenceBaseAction"
                    )


def _eval_left_action_contract(args_cli, _eval_train_parity, action_cfg, action_cfg_L):
    """The left arm runs the same action contract as the right (and the same GT-following overrides)."""
    # The left arm runs the same contract list as the right.
    apply_action_contract(
        action_cfg_L, args_cli, label="eval_cfg_overrides (LEFT)", build_cap_applied=_eval_train_parity
    )
    if getattr(args_cli, "left_gt_following", 0):
        # LEFT GT-following, as in the training overrides.
        action_cfg_L.finger_residual_scale = 0.0
        if hasattr(action_cfg_L, "wrist_residual_scale"):
            action_cfg_L.wrist_residual_scale = 0.0
        if hasattr(action_cfg_L, "wrist_pos_residual_max_m"):
            action_cfg_L.wrist_pos_residual_max_m = 0.0
        if hasattr(action_cfg_L, "wrist_rot_residual_max_deg"):
            action_cfg_L.wrist_rot_residual_max_deg = 0.0
        print("[eval_cfg_overrides] (left_present) LEFT GT-FOLLOWING: all left residual scales/caps = 0")
    if getattr(args_cli, "auto_passive_hand", 0):
        _passive = auto_passive_sides(args_cli)
        if "left" in _passive:
            apply_gt_following(action_cfg_L, "(auto-passive left)", "eval_cfg_overrides")
        if "right" in _passive:
            apply_gt_following(action_cfg, "(auto-passive right)", "eval_cfg_overrides")
    if args_cli.use_trajectory_ref:
        action_cfg_L.use_trajectory_reference = True
    if getattr(args_cli, "reference_base", 0) and hasattr(action_cfg_L, "reference_base"):
        action_cfg_L.reference_base = True
        action_cfg_L.use_trajectory_reference = True
        print("[eval_cfg_overrides] (left_present) LEFT reference_base = True, use_trajectory_reference = True")


def _eval_scene_and_ramps(cfg, args_cli, _ACTIVE_OBJ_KEY, _active_side):
    """Robot gravity, arm gains, object USDs (right / left), support discs, the ramps at their final values, object
    mass and robot z offset."""
    # Robot gravity off (DEXENGINE_DISABLE_ROBOT_GRAVITY=1, render/audit only) = ideal gravity compensation:
    # tells whether a PD base-following error is gravity droop. The object keeps its ramped force (separate
    # asset, its own disable_gravity is already True).
    if switches.text("DEXENGINE_DISABLE_ROBOT_GRAVITY") == "1":
        disable_robot_gravity(cfg, "eval_cfg_overrides")

    _eval_arm_pd_gains(cfg, args_cli, _ACTIVE_OBJ_KEY, _active_side)

    # Object mass: 0.2 kg default for object + object_left, overridable per side by
    # --object_mass / --object_mass_left.
    apply_object_mass(cfg, args_cli, "eval_cfg_overrides")
    apply_robot_z_off(cfg, "eval_cfg_overrides")


def _eval_arm_pd_gains(cfg, args_cli, _ACTIVE_OBJ_KEY, _active_side):
    """Arm PD gains as in training (--arm_kp/--arm_kd on the arm actuator)."""
    # Arm PD gains as in training (--arm_kp/--arm_kd on the arm actuator). Without them the eval arm runs at the
    # cfg default gains (softer, more damped): the wrist sags under the object load, the object never lifts and
    # every env terminates early, while the in-training evaluation (stiff training env) succeeds.
    apply_arm_gain_overrides(cfg, args_cli, "eval_cfg_overrides")

    # Override placeholder cuboid with real object USD
    if args_cli.object_usd:
        # One builder for rigid and articulated objects, shared with training (build_object_entity_cfg): a
        # RigidObjectCfg built here would break articulated clips (object_artic.usd) at boot or weld the lid.
        from dexengine.rl.env_setup.objects import build_object_entity_cfg as _boec

        setattr(cfg.scene, _ACTIVE_OBJ_KEY, _boec(args_cli, "eval_cfg_overrides", _active_side, "object_usd"))
        print(f"[eval_cfg_overrides] {_ACTIVE_OBJ_KEY} USD: {args_cli.object_usd}")

    # Bimanual: LEFT object USD (only with --bimanual).
    if getattr(args_cli, "bimanual", 0) and int(getattr(args_cli, "shared_object", 0) or 0):
        # Shared object: as in training, the placeholder left prim is dropped now and the left terms are
        # re-pointed in _apply_shared_object (last statement).
        cfg.scene.object_left = None
        print(
            f"[eval_cfg_overrides] [shared_object] LEFT object prim NOT built — one prim ObjectRight "
            f"(USD {args_cli.object_usd}); left terms re-pointed at the end of overrides",
            flush=True,
        )
    elif getattr(args_cli, "bimanual", 0):
        if not args_cli.object_usd_left:
            raise ValueError("--bimanual 1 requires --object_usd_left <path> (the LEFT object USD).")
        # The LEFT object uses the same builder as training and the right side, so per-clip LEFT objects
        # (--multi_object) spawn identically at eval.
        from dexengine.rl.env_setup.objects import build_object_entity_cfg as _boec_L

        cfg.scene.object_left = _boec_L(args_cli, "eval_cfg_overrides[left]", "left", "object_usd_left")
        print(f"[eval_cfg_overrides] (left_present) LEFT object USD: {args_cli.object_usd_left}")
    # Support discs (mid-air scenes, desk supports) — the same builder as the training cfg; without it the
    # replay/eval scene would differ from the recorded one.
    from dexengine.rl.env_setup.supports import build_scene_supports as _bss

    _bss(cfg, args_cli, "eval_cfg_overrides")

    # Eval: remove the gravity/friction CURRICULUM (no ramping).
    # Object has disable_gravity=True in its USD; the gravity_ramp event applies
    # the constant gravity force. We keep the event but fix it at full magnitude
    # from step 0 (ramp_steps=0, start_fraction=1.0) so there is NO ramp schedule.
    # Friction_decay similarly fixed to its final value (initial_multiplier=1.0).
    if hasattr(cfg, "events"):
        grav = getattr(cfg.events, "gravity_ramp", None)
        if grav is not None and hasattr(grav, "params"):
            grav.params["ramp_steps"] = 0
            grav.params["start_fraction"] = 1.0
            print("[eval_cfg_overrides] gravity_ramp pinned at 100% (no curriculum)")
        fric = getattr(cfg.events, "friction_decay", None)
        if fric is not None and hasattr(fric, "params"):
            fric.params["ramp_steps"] = 0
            fric.params["initial_multiplier"] = fric.params.get("final_multiplier", 1.0)
            print("[eval_cfg_overrides] friction_decay pinned at final value (no curriculum)")
        # Bimanual: pin the LEFT object gravity/friction the same way.
        if args_cli.bimanual:
            grav_l = getattr(cfg.events, "gravity_ramp_left", None)
            if grav_l is not None and hasattr(grav_l, "params"):
                grav_l.params["ramp_steps"] = 0
                grav_l.params["start_fraction"] = 1.0
                print("[eval_cfg_overrides] (bimanual) gravity_ramp_left pinned at 100%")
            fric_l = getattr(cfg.events, "friction_decay_left", None)
            if fric_l is not None and hasattr(fric_l, "params"):
                fric_l.params["ramp_steps"] = 0
                fric_l.params["initial_multiplier"] = fric_l.params.get("final_multiplier", 1.0)
                print("[eval_cfg_overrides] (bimanual) friction_decay_left pinned at final value")


def _eval_physics_dr(cfg, args_cli):
    """Physics DR at eval (--enable_physics_dr): the DR event ranges, or none."""
    # Physics domain randomisation at eval: --enable_physics_dr train | train_narrow | near_ood | far_ood
    # sets the DR event ranges below; none (default) removes the DR events.
    if getattr(args_cli, "enable_physics_dr", "none") != "none":
        dr_mode = args_cli.enable_physics_dr
        if dr_mode not in PHYSICS_DR_RANGES:
            print(f"[eval_cfg_overrides] WARNING: unknown DR mode '{dr_mode}', skipping")
        elif hasattr(cfg, "events"):
            dr_ranges = PHYSICS_DR_RANGES[dr_mode]
            for ev_name, param_key in [
                ("physics_dr_mass", "mass_scale_range"),
                ("physics_dr_friction", None),
                ("physics_dr_restitution", "restitution_scale_range"),
                ("physics_dr_com", "com_offset_range_m"),
            ]:
                ev = getattr(cfg.events, ev_name, None)
                if ev is None:
                    continue
                if ev_name == "physics_dr_friction":
                    ev.params["static_friction_scale_range"] = dr_ranges["static_friction_scale_range"]
                    ev.params["dynamic_friction_scale_range"] = dr_ranges["dynamic_friction_scale_range"]
                elif param_key is not None:
                    ev.params[param_key] = dr_ranges[param_key]
            print(
                f"[eval_cfg_overrides] PHYSICS DR enabled (mode='{dr_mode}'): "
                f"mass={dr_ranges['mass_scale_range']} "
                f"friction_s={dr_ranges['static_friction_scale_range']}"
            )
    else:
        # No DR mode requested: strip the per-reset physics_dr_* events so eval is deterministic and DR-free.
        # The events exist by default (include_physics_dr_events=True); without this they would fire at eval
        # with their cfg-default ranges.
        if hasattr(cfg, "events"):
            _removed_eval_dr = []
            _dr_ev_names = ["physics_dr_mass", "physics_dr_friction", "physics_dr_restitution", "physics_dr_com"]
            # Bimanual: also strip the LEFT object's per-reset DR.
            if args_cli.bimanual:
                _dr_ev_names += [
                    "physics_dr_mass_left",
                    "physics_dr_friction_left",
                    "physics_dr_restitution_left",
                    "physics_dr_com_left",
                ]
            for ev_name in _dr_ev_names:
                if getattr(cfg.events, ev_name, None) is not None:
                    setattr(cfg.events, ev_name, None)
                    _removed_eval_dr.append(ev_name)
            if _removed_eval_dr:
                print(f"[eval_cfg_overrides] PHYSICS DR removed for eval (train-only): {_removed_eval_dr}")


def _eval_obs_rewards_and_parity(cfg, args_cli, _eval_train_parity):
    """Observation layout, RewardPower strip, the train-parity mirror and the reset-pose randomisation off."""
    _eval_obs_v2_and_power(cfg, args_cli)

    # DEXENGINE_EVAL_TRAIN_PARITY=1 (diagnostic): also apply the two physics/spawn settings that only the
    # training overrides carry — the finger actuator gains --finger_effort/velocity/kd/kp (without this mirror a
    # run that passes them is scored on a different hand plant than it trained on) and the randomize_pose
    # parameters. Off by default because it changes the evaluation condition.
    if _eval_train_parity:
        _eval_finger_gains(cfg, args_cli)
        if (
            (getattr(args_cli, "randomize_finger", 0) or getattr(args_cli, "randomize_wrist", 0))
            and hasattr(cfg, "events")
            and hasattr(cfg.events, "randomize_pose")
        ):
            _rp = cfg.events.randomize_pose.params
            _rp["randomize_fraction"] = args_cli.randomize_fraction
            _rp["finger_open_deg"] = args_cli.finger_open_deg if args_cli.randomize_finger else 0.0
            _rp["wrist_z_max_cm"] = args_cli.wrist_z_max_cm if args_cli.randomize_wrist else 0.0
            print(f"[eval_cfg_overrides] TRAIN_PARITY randomize_pose: {_rp}")

    # Neutralise reset-pose randomisation: the in-training frozen evaluation neutralises reset randomisation
    # and physics DR right before measuring, so eval runs under the same no-randomisation condition (the
    # physics DR events were already removed above).
    if hasattr(cfg, "events") and hasattr(cfg.events, "randomize_pose"):
        _rp0 = cfg.events.randomize_pose.params
        for _k, _v in (("randomize_fraction", 0.0), ("finger_open_deg", 0.0), ("wrist_z_max_cm", 0.0)):
            if _k in _rp0:
                _rp0[_k] = _v
        print("[eval_cfg_overrides] reset randomisation neutralised: randomize_pose -> 0 (same condition as frozen)")


def _eval_finger_gains(cfg, args_cli):
    """Finger actuator overrides (--finger_effort/velocity/kd/kp) on every articulation, as in training."""
    _fg = {
        "effort_limit_sim": getattr(args_cli, "finger_effort", None),
        "velocity_limit_sim": getattr(args_cli, "finger_velocity", None),
        "damping": getattr(args_cli, "finger_kd", None),
        "stiffness": getattr(args_cli, "finger_kp", None),
    }
    if any(v is not None for v in _fg.values()) and hasattr(cfg, "scene"):
        _fg_applied, _fg_seen = False, {}
        for _nm in dir(cfg.scene):
            if _nm.startswith("_"):
                continue
            _acts = getattr(getattr(cfg.scene, _nm, None), "actuators", None)
            if not (isinstance(_acts, dict) and _acts):
                continue
            _fg_seen[_nm] = list(_acts.keys())
            from dexengine.rl.tasks.scene.robot import finger_actuator_key as _fak

            _akey = _fak(args_cli.hand)  # the builder's one spelling of the finger group
            if _akey not in _acts:
                continue
            _fa = _acts[_akey]
            for _attr, _val in _fg.items():
                if _val is not None:
                    setattr(_fa, _attr, float(_val))
            _fg_applied = True
            print(
                f"[eval_cfg_overrides] TRAIN_PARITY FINGER GAINS on {_nm}.{_akey}: "
                f"effort={_fa.effort_limit_sim} vel={_fa.velocity_limit_sim} "
                f"Kd={_fa.damping} Kp={_fa.stiffness}"
            )
        if not _fg_applied:  # a finger gain was given but no finger actuator consumes it: fail
            raise RuntimeError(
                f"[eval_cfg_overrides] TRAIN_PARITY: --finger_* was given but there is no finger "
                f"actuator. Articulations and actuators: {_fg_seen}"
            )


def _eval_obs_v2_and_power(cfg, args_cli):
    """obs_v2 observation wiring as in training, and the optional RewardPower strip."""
    # obs_v2 observations: wire the CLI mesh paths unconditionally, so the eval observation config equals the
    # training one (checkpoint and normaliser compatibility).
    if hasattr(cfg, "observations"):
        apply_obs_v2(cfg, args_cli, "eval_cfg_overrides")

    # Strip the RewardPower term (it saturates and carries no signal on the full robot), as in training.
    if (
        bool(getattr(args_cli, "strip_power_reward", 0))
        and hasattr(cfg, "rewards")
        and getattr(cfg.rewards, "power", None) is not None
    ):
        cfg.rewards.power = None
        print("[eval_cfg_overrides] STRIP power reward (audit: saturated-dead on full robot)")
