"""Reward overrides of the wrist and of human-pose imitation (steps of overrides_rewards.apply_reward_overrides): the
wrist L2 hinge, fingertip force weight, wrist / eef hinges, the human-pose reward drop / scale and the V2
observation mesh paths."""

from dexengine.rl.env_setup.sides import apply_obs_v2


def wrist_hinge_and_force_overrides(args_cli, cfg, total_timesteps, _res_l2_w, _wrist_l2):
    """Wrist L2 hinge, fingertip force weight and the wrist / eef hinges."""
    enable_wrist_hinge = _apply_wrist_free_zone(args_cli, cfg, total_timesteps, _res_l2_w, _wrist_l2)
    if enable_wrist_hinge and hasattr(cfg, "rewards"):
        eef_pos = getattr(cfg.rewards, "eef_pos", None)
        if eef_pos is not None:
            wrist_thresh = getattr(args_cli, "wrist_hinge_threshold_m", None)
            if wrist_thresh is None:
                wrist_thresh = 0.04
            eef_pos.params["hinge_threshold_m"] = wrist_thresh
            print(
                f"[train_cfg_overrides] WRIST HINGE enabled: eef_pos threshold={wrist_thresh} m (eef_rot is no-op in "
                f"full-robot mode)"
            )

    # Independent eef_pos / eef_rot hinge thresholds (--eef_pos_hinge_cm, --eef_rot_hinge_deg).
    # Gated on the flags being non-None; default None = no-op.
    _eef_pos_hinge_cm = getattr(args_cli, "eef_pos_hinge_cm", None)
    if _eef_pos_hinge_cm is not None and hasattr(cfg, "rewards"):
        _eef_pos_rw = getattr(cfg.rewards, "eef_pos", None)
        if _eef_pos_rw is not None:
            _eef_pos_hinge_m = float(_eef_pos_hinge_cm) / 100.0
            _eef_pos_rw.params["hinge_threshold_m"] = _eef_pos_hinge_m
            print(
                f"[train_cfg_overrides] C4 EEF_POS HINGE: hinge_threshold_m = {_eef_pos_hinge_m:.4f} m "
                f"({_eef_pos_hinge_cm} cm); zero gradient within threshold"
            )
    _eef_rot_hinge_deg = getattr(args_cli, "eef_rot_hinge_deg", None)
    if _eef_rot_hinge_deg is not None and hasattr(cfg, "rewards"):
        import math as _math

        _eef_rot_rw = getattr(cfg.rewards, "eef_rot", None)
        if _eef_rot_rw is not None:
            _eef_rot_hinge_rad = float(_eef_rot_hinge_deg) * _math.pi / 180.0
            _eef_rot_rw.params["hinge_threshold_rad"] = _eef_rot_hinge_rad
            print(
                f"[train_cfg_overrides] C4 EEF_ROT HINGE: hinge_threshold_rad = {_eef_rot_hinge_rad:.4f} rad "
                f"({_eef_rot_hinge_deg} deg); zero gradient within threshold (full-robot geodesic path only)"
            )


def _apply_wrist_free_zone(args_cli, cfg, total_timesteps, _res_l2_w, _wrist_l2):
    """WRIST RESIDUAL-L2 FREE ZONE: per-dim tax-free band on the wrist residual-L2 anchors, ramped in by the
    WristResidualFreeZoneRamp curriculum (0 for the guide phase, then linear to the target). cfg params carry the
    FINAL target so eval/render (--freeze_curriculum removes the curriculum term) keep the trained-final band.
    Default 0 -> block skipped.  ⚠ The flag is named --wrist_l2_hinge but this is **not a tracking hinge**: it bounds
    how large the residual ACTION may grow before the L2 magnitude penalty charges, and says nothing about tracking
    error. The wrist TRACKING hinge (eef_pos.hinge_threshold_m, --enable_hinge) is a separate path that is off unless
    asked for. See WristResidualFreeZoneRamp in dexengine/rl/tasks/mdp/curriculums.py.
    """
    if int(getattr(args_cli, "wrist_l2_hinge", 0)) == 1:
        if _wrist_l2 is None or _res_l2_w <= 0.0:
            raise ValueError(
                "--wrist_l2_hinge 1 requires the wrist split (--wrist_residual_l2_weight and --residual_l2_weight > 0)."
            )
        _hg_free = float(getattr(args_cli, "wrist_l2_hinge_free_frac", 0.7))
        _hg_terms = []
        for _hg_nm in ("residual_l2_wrist", "residual_l2_wrist_left"):
            _hg_t = getattr(cfg.rewards, _hg_nm, None)
            if _hg_t is not None:
                _hg_t.params["per_dim_deadzone"] = _hg_free
                _hg_terms.append(_hg_nm)
        _hg_t0 = int(float(getattr(args_cli, "wrist_l2_hinge_start_frac", 0.10)) * total_timesteps)
        _hg_t1 = max(int(float(getattr(args_cli, "wrist_l2_hinge_end_frac", 0.25)) * total_timesteps), _hg_t0 + 1)
        if hasattr(cfg, "curriculum") and cfg.curriculum is not None:
            from isaaclab.managers import CurriculumTermCfg as _CurrTermWH

            from dexengine.rl.tasks.mdp.curriculums import WristResidualFreeZoneRamp as _WHRamp

            # The term name "wrist_hinge_ramp" is a W&B / TensorBoard metric name; keep it so curves stay
            # comparable across runs.
            cfg.curriculum.wrist_hinge_ramp = _CurrTermWH(
                func=_WHRamp, params={"terms": _hg_terms, "target": _hg_free, "start_step": _hg_t0, "end_step": _hg_t1}
            )
        else:
            print(
                "[train_cfg_overrides] WARNING: cfg.curriculum missing — wrist residual "
                "free zone held at final target from step 0 (no guide phase)"
            )
        print(
            f"[train_cfg_overrides] WRIST RESIDUAL-L2 FREE ZONE: 0 -> {_hg_free} of cap "
            f"over steps [{_hg_t0}, {_hg_t1}] of {total_timesteps}; terms={_hg_terms} "
            f"(NOT a tracking hinge — wrist tracking hinge is OFF)"
        )

    # --fingertip_force_weight 0 disables RewardFingertipForce (it has no pre-contact gradient); the dense soft
    # contact reward replaces it (via --enable_contact_match_reward, which
    # removes it at runtime). None = leave cfg default (weight 1.0).
    _ff_w = getattr(args_cli, "fingertip_force_weight", None)
    if _ff_w is not None and hasattr(cfg, "rewards"):
        _ff = getattr(cfg.rewards, "fingertip_force", None)
        if _ff is not None:
            _ff.weight = float(_ff_w)
            print(
                f"[train_cfg_overrides] fingertip_force weight -> {float(_ff_w)} (deprecated force-only contact reward)"
            )

    # Wrist hinge on eef_pos (eef_rot is a no-op in full-robot mode)
    enable_wrist_hinge = bool(getattr(args_cli, "enable_wrist_hinge", 0))
    return enable_wrist_hinge


def human_pose_and_obs_overrides(args_cli, cfg):
    """Human-pose reward drop / scale and the V2 observation mesh paths."""
    # Limit-case reward: drop human-pose imitation entirely
    disable_human_pose = bool(getattr(args_cli, "disable_human_pose_rewards", 0))
    _HUMAN_POSE_REWARD_NAMES = [
        "eef_pos",
        "eef_rot",
        "thumb_tip",
        "index_tip",
        "middle_tip",
        "ring_tip",
        "pinky_tip",
        "level_1",
        "level_2",
    ]
    if disable_human_pose and hasattr(cfg, "rewards"):
        for rw_name in _HUMAN_POSE_REWARD_NAMES:
            rw = getattr(cfg.rewards, rw_name, None)
            if rw is not None:
                rw.weight = 0.0
        print("[train_cfg_overrides] LIMIT-CASE REWARD: human-pose rewards zeroed.")

    # Scale the human-pose imitation reward weights
    hp_scale = float(getattr(args_cli, "human_pose_weight_scale", 1.0))
    if hp_scale != 1.0 and hasattr(cfg, "rewards") and not disable_human_pose:
        scaled = []
        for rw_name in _HUMAN_POSE_REWARD_NAMES:
            rw = getattr(cfg.rewards, rw_name, None)
            if rw is not None:
                rw.weight = rw.weight * hp_scale
                scaled.append(f"{rw_name}={rw.weight:.3f}")
        print(f"[train_cfg_overrides] HUMAN POSE SCALE={hp_scale}: {', '.join(scaled)}")

    # V2-native obs: wire the CLI mesh paths (unconditional — the cfg declares the V2 layout). Runs LAST.
    if hasattr(cfg, "observations"):
        apply_obs_v2(cfg, args_cli, "train_cfg_overrides")

    # Reward audit fix: strip the dead RewardPower term (exp(-10*P)
    # saturates to 0 on the full robot — zero gradient, 0.000 in every logged return).
    if (
        bool(getattr(args_cli, "strip_power_reward", 0))
        and hasattr(cfg, "rewards")
        and getattr(cfg.rewards, "power", None) is not None
    ):
        cfg.rewards.power = None
        print("[train_cfg_overrides] STRIP power reward (audit: saturated-dead on full robot)")
