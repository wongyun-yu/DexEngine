"""Training-only edits of the env cfg (``train_cfg_overrides``, the ``extra_cfg_fn`` of ``create_env``).

The edits run in five groups, in this order: scene and actions (overrides_scene), schedules and curricula
(overrides_schedule), physics and randomization (overrides_physics), reward terms (overrides_rewards), and the runtime
schemes below (VOC, collision curriculum, per-side advantage, reset scheme, termination diagnostics, side sweeps).
"""

from dexengine.rl.env_setup.build import _apply_shared_object
from dexengine.rl.env_setup.sides import apply_idle_side, apply_left_uni
from dexengine.rl.train import overrides_physics, overrides_rewards, overrides_scene, overrides_schedule


def apply_runtime_schemes(*, args_cli, cfg, rollouts, total_timesteps):
    """Env-side runtime schemes: VOC, collision curriculum, per-side advantage, reset scheme,
    termination diagnostics, then the side sweeps (they must stay last)."""
    # VOC (DexMachina-style virtual object controller, arXiv 2505.24853):
    # TRAIN-ONLY object-assist PD wrench with performance-gated decay. Only sets
    # scalar cfg fields; ArmResidualTrainEnv.load_managers constructs the VOC when
    # cfg.voc_enable is True. --voc 0 (default) leaves cfg untouched.
    if int(getattr(args_cli, "voc", 0)) == 1:
        try:
            from dexengine.rl.tasks.cfg.residual import DEFAULT_OBJECT_MASS_KG as _voc_mass
        except ImportError:
            _voc_mass = 0.2
        cfg.voc_enable = True
        cfg.voc_kp_pos = float(args_cli.voc_kp_pos)
        cfg.voc_kp_rot = float(args_cli.voc_kp_rot)
        cfg.voc_decay = float(args_cli.voc_decay)
        # policy ITERATIONS -> common_step_counter units (1 iter = `rollouts` steps),
        # same counter pattern as threshold_tightening/rsi_schedule.
        cfg.voc_check_steps = max(int(args_cli.voc_check_iters) * rollouts, 1)
        cfg.voc_err_pos = float(args_cli.voc_err_pos)
        cfg.voc_err_rot_deg = float(args_cli.voc_err_rot_deg)
        cfg.voc_kd_rot_scale = float(args_cli.voc_kd_rot_scale)
        cfg.voc_mass = float(_voc_mass)
        # v2 (defaults = v1 behavior): assist force cap + moving/contact decay gate.
        cfg.voc_force_cap_mg = float(getattr(args_cli, "voc_force_cap_mg", 0.0))
        cfg.voc_gate_mode = str(getattr(args_cli, "voc_gate_mode", "ema"))
        cfg.voc_gate_contact_n = float(getattr(args_cli, "voc_gate_contact_n", 0.3))
        # v2.1 (default 0 = v2 behavior): forced final flush; needs the same
        # total-step clock the ramps use (common_step_counter vs MI*rollouts).
        cfg.voc_flush_frac = float(getattr(args_cli, "voc_flush_frac", 0.0))
        cfg.voc_total_steps = int(total_timesteps)
        # v3 (default 'global' = v2.1 behavior): per-env uniform gain randomization.
        cfg.voc_schedule = str(getattr(args_cli, "voc_schedule", "global"))
        print(
            f"[train_cfg_overrides] VOC ENABLED: kp_pos={cfg.voc_kp_pos} N/m "
            f"kp_rot={cfg.voc_kp_rot} Nm/rad decay={cfg.voc_decay} "
            f"check_every={cfg.voc_check_steps} steps ({args_cli.voc_check_iters} iters "
            f"x {rollouts} rollouts) gate=({cfg.voc_err_pos}m, {cfg.voc_err_rot_deg}deg) "
            f"kd_rot_scale={cfg.voc_kd_rot_scale} mass={cfg.voc_mass}kg "
            f"force_cap_mg={cfg.voc_force_cap_mg} gate_mode={cfg.voc_gate_mode} "
            f"gate_contact_n={cfg.voc_gate_contact_n}N "
            f"flush_frac={cfg.voc_flush_frac} schedule={cfg.voc_schedule} "
            f"(total={total_timesteps} steps)"
        )

    _apply_collision_curriculum(args_cli, cfg, rollouts, total_timesteps)


def _apply_collision_curriculum(args_cli, cfg, rollouts, total_timesteps):
    """COLLISION CURRICULUM (bimanual TRAIN only): cross-pairs (RHxLH, RHxobjL, LHxobjR) start FILTERED OFF; one-shot
    runtime re-enable at --collision_curriculum_on_frac of training (env_0 template FilteredPairs edit, no sim re-
    init). Sets scalar cfg fields only; ArmResidualTrainEnv + BimanualSelfCollisionScene consume them. Default 0 =
    cfg untouched. Eval/render never call this override -> full collision. FULL-ISOLATION extension
    (--collision_curriculum_objpair 1): ALSO filter object_right<->object_left at start; re-enabled at the SAME flip.
    Extends the base curriculum only — fail-fast if requested without it.
    """
    _cc_objpair = int(getattr(args_cli, "collision_curriculum_objpair", 0)) == 1
    if _cc_objpair and int(getattr(args_cli, "collision_curriculum", 0)) != 1:
        raise ValueError(
            "--collision_curriculum_objpair 1 requires --collision_curriculum 1 (it extends the base curriculum)"
        )
    if int(getattr(args_cli, "collision_curriculum", 0)) == 1:
        if not int(getattr(args_cli, "bimanual", 0)):
            print(
                "[train_cfg_overrides] WARNING: --collision_curriculum 1 requires "
                "--bimanual 1 — IGNORED (no cross-pairs exist in uni)"
            )
        else:
            # "start,end" = a ramp (the share of envs with the pairs enabled goes 0->100% over that progress span);
            # a single value = a one-shot switch. At 100% the physics equals the strict (full-collision) setting.
            _cc_raw = str(getattr(args_cli, "collision_curriculum_on_frac", "1.0")).strip()
            _cc_ramp = None
            if "," in _cc_raw:
                try:
                    _a_cc, _b_cc = (float(x) for x in _cc_raw.split(","))
                except ValueError as _e_cc:
                    raise ValueError(
                        f"--collision_curriculum_on_frac {_cc_raw!r} cannot be parsed — "
                        f'expected one float or "start,end": {_e_cc}'
                    ) from _e_cc
                if not (0.0 <= _a_cc < _b_cc <= 1.0):
                    raise ValueError(f"--collision_curriculum_on_frac {_cc_raw!r}: must be 0<=start<end<=1")
                _cc_ramp = (_a_cc, _b_cc)
                _cc_frac = _b_cc
            else:
                _cc_frac = float(_cc_raw)
            cfg.collision_curriculum = True
            cfg.collision_curriculum_objpair = _cc_objpair
            cfg.collision_curriculum_flip_step = max(int(_cc_frac * total_timesteps), 0)
            cfg.collision_curriculum_total_steps = int(total_timesteps)
            if _cc_ramp is not None:
                cfg.collision_curriculum_ramp = _cc_ramp
            _cc_cats = "RHxLH, RHxobjL, LHxobjR" + (", objRxobjL" if _cc_objpair else "")
            print(
                f"[train_cfg_overrides] COLLISION CURRICULUM: cross-pairs ({_cc_cats}) "
                + (
                    f"start OFF -> RAMP {_cc_ramp[0]:.2f}->{_cc_ramp[1]:.2f} progress window "
                    f"(share of envs turned on 0->100%, at the end = the same physics as strict) "
                    f"of {total_timesteps}"
                    if _cc_ramp is not None
                    else f"start OFF -> one-shot ENABLE at step "
                    f"{cfg.collision_curriculum_flip_step} "
                    f"({int(_cc_frac * 100)}% of {total_timesteps})"
                )
                + (" [FULL ISOLATION: +object pair]" if _cc_objpair else "")
            )

    _apply_side_advantage(args_cli, cfg, rollouts)
    # SHARED OBJECT sweep — same "last statement" contract as the left-uni sweep above (no-op by default).
    _apply_shared_object(cfg, args_cli, "train_cfg_overrides")
    # IDLE SIDE sweep — same contract; mutually exclusive with the shared object (no-op by default).
    apply_idle_side(cfg, args_cli, "train_cfg_overrides")


def _apply_side_advantage(args_cli, cfg, rollouts):
    """PER-SIDE ADVANTAGE (factored credit): env-side gate. Makes ArmResidualTrainEnv.step() publish
    reward_right/reward_left into the step extras (RewardManager._step_reward split by each term's 'side' param;
    unclear/global terms 50/50 — total reward UNCHANGED). The learner side (PerSidePPO + two-head critic) is wired
    at learner build. Default 0 -> attribute never set.
    """
    if int(getattr(args_cli, "per_side_advantage", 0)) == 1:
        if not int(getattr(args_cli, "bimanual", 0)):
            raise ValueError(
                "--per_side_advantage 1 requires --bimanual 1 (per-side reward "
                "streams need the side-tagged bimanual reward set)."
            )
        cfg.per_side_reward = True
        print(
            "[train_cfg_overrides] PER-SIDE ADVANTAGE: env exposes per-step "
            "reward_right/reward_left extras (total reward unchanged)"
        )

    # RESET SCHEME: 'decoupled' = per-side snap. The four SIDE
    # termination gates become SNAP-ONLY (they stash per-side fail masks on the
    # env and return zeros — episode no longer ends); ArmResidualTrainEnv snaps
    # the failed side back onto its reference at the current clip frame at the
    # top of the next step and publishes done_right/done_left extras for the
    # per-side GAE. Episode-level resets happen on time_out (and any remaining
    # non-side termination) only. Default 'or' -> block skipped.
    _rs = str(getattr(args_cli, "reset_scheme", "or"))
    if _rs in ("snap", "decoupled", "park"):
        if int(getattr(args_cli, "bimanual", 0)) != 1 or int(getattr(args_cli, "per_side_advantage", 0)) != 1:
            raise ValueError(
                f"--reset_scheme {_rs} requires --bimanual 1 AND "
                "--per_side_advantage 1 (side snaps need per-side GAE dones — "
                "without them the value function bootstraps across teleports)."
            )
        cfg.per_side_reset = _rs
        _ss_terms = []
        for _ss_nm in ("object_tracking", "object_tracking_left", "contact_hand_right", "contact_hand_left"):
            _ss_t = getattr(cfg.terminations, _ss_nm, None)
            if _ss_t is not None:
                _ss_t.params["snap_only"] = True
                _ss_terms.append(_ss_nm)
        if not _ss_terms:
            raise ValueError(f"--reset_scheme {_rs}: no side termination terms found to convert")
        print(
            f"[train_cfg_overrides] RESET SCHEME={_rs}: snap-only side gates={_ss_terms}; "
            f"episode ends on time_out only; per-side dones -> extras done_right/done_left"
            + (
                " ; INDEPENDENT side clip clocks (side fails/completions resample that side's RSI segment)"
                if _rs == "decoupled"
                else ""
            )
        )

    # TERMDIAG (bimanual TRAIN, print-only, no flag): periodic per-term termination
    # counts ([TERMDIAG] object_tracking=.. velocity_sanity=.. ...) so runs show WHICH
    # term kills episodes. First print after 1 iter (wiring proof), then every 25 iters.
    if int(getattr(args_cli, "bimanual", 0)) == 1:
        cfg.termdiag_enable = True
        cfg.termdiag_steps = max(25 * rollouts, 1)
        print(
            f"[train_cfg_overrides] TERMDIAG: per-term termination counts every "
            f"{cfg.termdiag_steps} steps (25 iters x {rollouts} rollouts; "
            f"first at {rollouts})"
        )

    # LEFT-UNI side sweep — MUST stay the LAST statement of this function so
    # every runtime-registered term (contact-pilot from create_env, keypoint,
    # residual_l2_wrist, obs_v2 above) is already in cfg when the sweep flips
    # side right->left. Do NOT register side-carrying terms after this line.
    apply_left_uni(cfg, args_cli, "train_cfg_overrides")


def train_cfg_overrides(
    cfg, *, _ACTIVE_OBJ_KEY, _active_side, _launch_ctx, _residual_act_name, args_cli, rollouts, total_timesteps
):
    """Apply training config overrides.

    NOTE: __post_init__ already ran at CfgClass() construction time, BEFORE this
    callback. So setting cfg.object_usd_path here has NO EFFECT on the scene —
    __post_init__'s `if self.object_usd_path:` check already evaluated to False.
    We must directly mutate the cfg.scene object entity here instead.
    """
    overrides_scene.apply_scene_overrides(
        _ACTIVE_OBJ_KEY=_ACTIVE_OBJ_KEY,
        _active_side=_active_side,
        _launch_ctx=_launch_ctx,
        _residual_act_name=_residual_act_name,
        args_cli=args_cli,
        cfg=cfg,
        rollouts=rollouts,
        total_timesteps=total_timesteps,
    )
    overrides_schedule.apply_schedule_overrides(args_cli=args_cli, cfg=cfg, total_timesteps=total_timesteps)
    overrides_physics.apply_physics_overrides(args_cli=args_cli, cfg=cfg)
    overrides_rewards.apply_reward_overrides(
        _residual_act_name=_residual_act_name, args_cli=args_cli, cfg=cfg, total_timesteps=total_timesteps
    )
    apply_runtime_schemes(args_cli=args_cli, cfg=cfg, rollouts=rollouts, total_timesteps=total_timesteps)
