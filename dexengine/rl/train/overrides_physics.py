"""Physics and randomization edits of ``train_cfg_overrides``: per-reset DR, object velocity kicks, arm disturbance."""

# Per-episode physics DR
# DR ranges (training ranges of the randomization events).
PHYSICS_DR_RANGES = {
    "train": {
        "mass_scale_range": (0.7, 1.3),
        "static_friction_scale_range": (0.5, 1.5),
        "dynamic_friction_scale_range": (0.5, 1.5),
        "restitution_scale_range": (0.5, 1.5),
        "com_offset_range_m": (-0.02, 0.02),
    },
    # Narrow DR: the full train range without a curriculum collapsed training; this milder range applies
    # from iter 0.
    "train_narrow": {
        "mass_scale_range": (0.85, 1.15),  # ±15%
        "static_friction_scale_range": (0.75, 1.25),  # ±25%
        "dynamic_friction_scale_range": (0.75, 1.25),  # ±25%
        "restitution_scale_range": (0.8, 1.2),  # ±20%
        "com_offset_range_m": (-0.01, 0.01),  # ±1cm
    },
    "near_ood": {
        "mass_scale_range": (1.4, 1.6),
        "static_friction_scale_range": (0.3, 0.45),
        "dynamic_friction_scale_range": (0.3, 0.45),
        "restitution_scale_range": (1.6, 1.8),
        "com_offset_range_m": (-0.03, 0.03),
    },
    "far_ood": {
        "mass_scale_range": (1.8, 2.2),
        "static_friction_scale_range": (0.2, 0.3),
        "dynamic_friction_scale_range": (0.2, 0.3),
        "restitution_scale_range": (1.9, 2.0),
        "com_offset_range_m": (-0.05, 0.05),
    },
}


def apply_physics_overrides(*, args_cli, cfg):
    """Per-reset physics DR, object velocity kicks, arm disturbance wrenches and the additional
    contact component."""

    physics_dr_mode = args_cli.enable_physics_dr
    _b1a_ev_names = ("physics_dr_mass", "physics_dr_friction", "physics_dr_restitution", "physics_dr_com")
    _pv, _pw = _sync_left_object_dr(args_cli, cfg, _b1a_ev_names, physics_dr_mode)
    if (_pv > 0.0 or _pw > 0.0) and hasattr(cfg, "events"):
        import isaaclab.envs.mdp as _il_mdp
        from isaaclab.managers import EventTermCfg as _EventTerm
        from isaaclab.managers import SceneEntityCfg as _SEC

        cfg.events.perturb_object = _EventTerm(
            func=_il_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 2.0),
            params={
                "asset_cfg": _SEC("object"),
                "velocity_range": {
                    "x": (-_pv, _pv),
                    "y": (-_pv, _pv),
                    "z": (-_pv, _pv),
                    "roll": (-_pw, _pw),
                    "pitch": (-_pw, _pw),
                    "yaw": (-_pw, _pw),
                },
            },
        )
        print(
            f"[train_cfg_overrides] OBJECT PERTURB enabled: |v|<={_pv} m/s "
            f"|w|<={_pw} rad/s, interval 1-2 s (basin-widening robustify)"
        )

    # Contact additional components — grasp_stability only.
    enable_additional = bool(args_cli.enable_additional_components)
    if enable_additional and hasattr(cfg, "rewards"):
        gs_rw = getattr(cfg.rewards, "grasp_stability", None)
        if gs_rw is not None:
            gs_rw.weight = 0.2
            print("[train_cfg_overrides] ADDITIONAL COMPONENTS: grasp_stability weight=0.2")
        else:
            print("[train_cfg_overrides] WARNING: grasp_stability reward not found in RewardsCfg")
    else:
        print("[train_cfg_overrides] ADDITIONAL COMPONENTS disabled (--enable_additional_components=0)")


def _sync_left_object_dr(args_cli, cfg, _b1a_ev_names, physics_dr_mode):
    """Bimanual: also inject/strip the LEFT object's per-reset DR events so both objects share the same DR mode. Gated
    by --bimanual.
    """
    if getattr(args_cli, "bimanual", 0):
        _b1a_ev_names = _b1a_ev_names + (
            "physics_dr_mass_left",
            "physics_dr_friction_left",
            "physics_dr_restitution_left",
            "physics_dr_com_left",
        )
    if physics_dr_mode != "none" and hasattr(cfg, "events"):
        dr_ranges = PHYSICS_DR_RANGES[physics_dr_mode]
        _dr_inject = [
            ("physics_dr_mass", "mass_scale_range"),
            ("physics_dr_friction", None),  # friction has two separate params
            ("physics_dr_restitution", "restitution_scale_range"),
            ("physics_dr_com", "com_offset_range_m"),
        ]
        if getattr(args_cli, "bimanual", 0):
            _dr_inject += [
                ("physics_dr_mass_left", "mass_scale_range"),
                ("physics_dr_friction_left", None),
                ("physics_dr_restitution_left", "restitution_scale_range"),
                ("physics_dr_com_left", "com_offset_range_m"),
            ]
        # Inject ranges into the per-reset DR events (already active by default).
        for ev_name, param_key in _dr_inject:
            ev = getattr(cfg.events, ev_name, None)
            if ev is None:
                print(f"[train_cfg_overrides] WARNING: event {ev_name} not found in EventCfg")
                continue
            if ev_name == "physics_dr_friction":
                ev.params["static_friction_scale_range"] = dr_ranges["static_friction_scale_range"]
                ev.params["dynamic_friction_scale_range"] = dr_ranges["dynamic_friction_scale_range"]
            elif param_key is not None:
                ev.params[param_key] = dr_ranges[param_key]
        print(
            f"[train_cfg_overrides] PHYSICS DR enabled (mode='{physics_dr_mode}'): "
            f"mass={dr_ranges['mass_scale_range']} "
            f"friction_s={dr_ranges['static_friction_scale_range']} "
            f"friction_d={dr_ranges['dynamic_friction_scale_range']} "
            f"restitution={dr_ranges['restitution_scale_range']} "
            f"com_m={dr_ranges['com_offset_range_m']}"
        )
        # Removing friction_decay collapses PPO training (measured), so the friction_decay curriculum
        # (3x→1x nominal over the ramp window = --gravity_ramp_fraction × total
        # timesteps, default 10%) is load-bearing. We now PRESERVE
        # friction_decay alongside per-reset physics_dr_friction. The per-reset
        # event scales the current (decayed) friction value, so both
        # mechanisms contribute.
        print(
            "[train_cfg_overrides] PHYSICS DR: friction_decay PRESERVED (curriculum coexists with per-reset DR scaling)"
        )
    else:
        # Remove all 4 DR events for the regression baseline (P0.1)
        removed = []
        if hasattr(cfg, "events"):
            for ev_name in _b1a_ev_names:
                if getattr(cfg.events, ev_name, None) is not None:
                    setattr(cfg.events, ev_name, None)
                    removed.append(ev_name)
        print(f"[train_cfg_overrides] PHYSICS DR disabled (--enable_physics_dr=none); removed events: {removed}")

    # Basin-widening object perturbation: additive
    # random velocity kicks on the RIGHT object during training rollouts.
    # Native isaaclab mdp.push_by_setting_velocity (vel_w += uniform sample),
    # interval mode 1-2 s per env. TRAIN-ONLY: _eval_cfg_overrides never adds
    # this, so standalone evals stay nominal.
    _pv = float(getattr(args_cli, "perturb_obj_vel", 0.0) or 0.0)
    _pw = float(getattr(args_cli, "perturb_obj_angvel", 0.0) or 0.0)
    return _pv, _pw
