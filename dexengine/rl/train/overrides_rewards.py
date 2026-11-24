"""Reward-term edits of ``train_cfg_overrides``: residual L2 split, finger penalty, hinges, weight scales, obs
wiring."""

from dexengine.core.errors import InputMissingError
from dexengine.rl.train.finger_penalty import (
    _finger_penalty_side,
    _finger_penalty_span,
    _finger_penalty_user,
    _report_finger_penalty,
)
from dexengine.rl.train.overrides_wrist_pose import human_pose_and_obs_overrides, wrist_hinge_and_force_overrides


def apply_reward_overrides(*, _residual_act_name, args_cli, cfg, total_timesteps):
    """Fingertip hinges and weights, finger actuator gains, the residual-L2 / action-rate penalties,
    wrist hinges, human-pose weights, V2 observations and the power-term strip."""

    _fingertip_hinge_overrides(args_cli, cfg)

    _finger_gain_overrides(args_cli, cfg)

    _fplan, _fspan, _fspan_src, _res_l2_w, _wrist_l2 = _residual_l2_overrides(_residual_act_name, args_cli, cfg)

    _action_rate_overrides(args_cli, cfg, _fplan, _fspan)
    _report_finger_penalty(cfg, args_cli, _fspan, _fspan_src, _fplan, _residual_act_name=_residual_act_name)

    wrist_hinge_and_force_overrides(args_cli, cfg, total_timesteps, _res_l2_w, _wrist_l2)

    human_pose_and_obs_overrides(args_cli, cfg)


def _fingertip_hinge_overrides(args_cli, cfg):
    """Fingertip tracking hinges and the fingertip weight scale."""
    # Soft hinge on fingertip tracking rewards (per-finger) (H2)
    enable_finger_hinge = bool(getattr(args_cli, "enable_finger_hinge", 0))
    asymmetric = bool(getattr(args_cli, "enable_finger_hinge_asymmetric", 0))
    contact_gated = bool(getattr(args_cli, "enable_contact_gated_hinge", 0))
    _FINGERTIPS = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")
    # Bimanual: every hinge-threshold loop below iterates _FINGERTIPS; append the LEFT tip
    # names so the same fingertip dead-zone (R1 --fingertip_hinge_cm, --enable_finger_hinge,
    # etc.) is applied to the left tips too. Else right tips get a 2cm hinge while left tips
    # decay with pure exp(-k*err) — a silent reward asymmetry. getattr(...,None) skips them
    # in single-hand (no _left terms).
    if bool(getattr(args_cli, "bimanual", 0)):
        _FINGERTIPS = _FINGERTIPS + tuple(f"{t}_left" for t in _FINGERTIPS)
    if enable_finger_hinge and hasattr(cfg, "rewards"):
        if asymmetric:
            # H6 per-finger asymmetric: thumb/index/middle 3cm (conflict-heavy), ring/pinky 2cm
            per_finger = {
                "thumb_tip": 0.03,
                "index_tip": 0.03,
                "middle_tip": 0.03,
                "ring_tip": 0.02,
                "pinky_tip": 0.02,
            }
            for rw_name, thresh in per_finger.items():
                rw = getattr(cfg.rewards, rw_name, None)
                if rw is not None:
                    rw.params["hinge_threshold_m"] = thresh
            print("[train_cfg_overrides] FINGER HINGE ASYMMETRIC: thumb/index/middle=3cm, ring/pinky=2cm")
        elif contact_gated:
            # H10 reward-form variant: state-dependent hinge.
            # In-contact fingers get wider tolerance (4cm), free fingers stay 2cm.
            for rw_name in _FINGERTIPS:
                rw = getattr(cfg.rewards, rw_name, None)
                if rw is not None:
                    rw.params["hinge_threshold_m"] = 0.02
                    rw.params["contact_gated_hinge"] = True
                    rw.params["hinge_threshold_in_contact"] = 0.04
                    rw.params["contact_force_threshold"] = 5.0
            print("[train_cfg_overrides] FINGER HINGE CONTACT-GATED: in-contact=4cm, free=2cm")
        else:
            pose_thresh = getattr(args_cli, "finger_hinge_threshold_m", None)
            if pose_thresh is None:
                pose_thresh = 0.02
            for rw_name in _FINGERTIPS:
                rw = getattr(cfg.rewards, rw_name, None)
                if rw is not None:
                    rw.params["hinge_threshold_m"] = pose_thresh
            print(f"[train_cfg_overrides] FINGER HINGE enabled: fingertip rewards use threshold={pose_thresh} m")

    _apply_fingertip_hinge_toggle(args_cli, cfg, _FINGERTIPS, enable_finger_hinge)


def _apply_fingertip_hinge_toggle(args_cli, cfg, _FINGERTIPS, enable_finger_hinge):
    """R1 fingertip-hinge toggle (--fingertip_hinge_cm, default 2.0 cm = ON; 0 = OFF). Single source of truth for the R1
    hinge/non-hinge A/B. Applied UNLESS a legacy hinge flag already set the fingertip hinge (those take precedence
    above).
    """
    _contact_pilot_hinge = bool(getattr(args_cli, "enable_hinge", 0)) or bool(getattr(args_cli, "hinge_schedule", 0))
    if (not enable_finger_hinge) and (not _contact_pilot_hinge) and hasattr(cfg, "rewards"):
        _ftip_hinge_m = float(getattr(args_cli, "fingertip_hinge_cm", 2.0)) / 100.0
        _hinge_set = []
        for rw_name in _FINGERTIPS:
            rw = getattr(cfg.rewards, rw_name, None)
            if rw is not None:
                rw.params["hinge_threshold_m"] = _ftip_hinge_m
                _hinge_set.append(rw_name)
        _state = "OFF" if _ftip_hinge_m <= 0.0 else f"ON {_ftip_hinge_m * 100:.1f}cm"
        print(
            f"[train_cfg_overrides] FINGERTIP HINGE (R1 toggle): {_state} on {len(_hinge_set)} tips "
            f"{_hinge_set} (threshold={_ftip_hinge_m:.4f} m)"
        )

    # Down-weight the 5 fingertip-tracking rewards (imitation prior) so the
    # contact/object terms dominate during the contact phase. ×1.0 = no change.
    _ft_scale = float(getattr(args_cli, "fingertip_weight_scale", 1.0))
    if _ft_scale != 1.0 and hasattr(cfg, "rewards"):
        _scaled = {}
        for rw_name in _FINGERTIPS:
            rw = getattr(cfg.rewards, rw_name, None)
            if rw is not None:
                rw.weight = float(rw.weight) * _ft_scale
                _scaled[rw_name] = round(rw.weight, 4)
        print(f"[train_cfg_overrides] FINGERTIP WEIGHT SCALE: ×{_ft_scale} -> {_scaled}")


def _finger_gain_overrides(args_cli, cfg):
    """Finger actuator gains (effort, velocity, Kd, Kp)."""
    # RH56F1 finger actuation gains (responsiveness ablation). Applied to the
    # independent-finger actuator ('fingers') of every scene articulation that has one
    # (right_hand/left_hand). Scans all scene articulations and WARNS loudly if none
    # is found, so a flag mismatch can never silently no-op. The passive 'fingers_mimic'
    # actuator (mimic coupling, Kp/Kd=0) is intentionally NOT touched.
    _fg_eff = getattr(args_cli, "finger_effort", None)
    _fg_vel = getattr(args_cli, "finger_velocity", None)
    _fg_kd = getattr(args_cli, "finger_kd", None)
    _fg_kp = getattr(args_cli, "finger_kp", None)
    if any(v is not None for v in (_fg_eff, _fg_vel, _fg_kd, _fg_kp)) and hasattr(cfg, "scene"):
        _fg_applied = False
        _fg_seen = {}
        for _nm in dir(cfg.scene):
            if _nm.startswith("_"):
                continue
            _obj = getattr(cfg.scene, _nm, None)
            _acts = getattr(_obj, "actuators", None)
            if not (isinstance(_acts, dict) and _acts):
                continue
            _fg_seen[_nm] = list(_acts.keys())
            # Independent-finger actuator of the combo robot — its name has one owner
            # (dexengine.rl.tasks.scene.robot.finger_actuator_key). NOT the passive '*_mimic'.
            from dexengine.rl.tasks.scene.robot import finger_actuator_key as _fak

            _akey = _fak(args_cli.hand)
            if _akey not in _acts:
                continue
            _fa = _acts[_akey]
            _old = f"effort={_fa.effort_limit_sim} vel={_fa.velocity_limit_sim} Kd={_fa.damping} Kp={_fa.stiffness}"
            if _fg_eff is not None:
                _fa.effort_limit_sim = float(_fg_eff)
            if _fg_vel is not None:
                _fa.velocity_limit_sim = float(_fg_vel)
            if _fg_kd is not None:
                _fa.damping = float(_fg_kd)
            if _fg_kp is not None:
                _fa.stiffness = float(_fg_kp)
            _fg_applied = True
            print(
                f"[train_cfg_overrides] FINGER GAINS on {_nm}.{_akey}: "
                f"OLD[{_old}] -> NEW[effort={_fa.effort_limit_sim} "
                f"vel={_fa.velocity_limit_sim} Kd={_fa.damping} Kp={_fa.stiffness}]"
            )
        if not _fg_applied:  # a meaningful flag with no consumer is a failure, not a warning
            raise RuntimeError(
                f"[train_cfg_overrides] --finger_* was given but there is no finger actuator — "
                f"refusing to run without it. Articulations/actuators: {_fg_seen}"
            )


def _residual_l2_overrides(_residual_act_name, args_cli, cfg):
    """Finger penalty span and the residual-L2 penalty (combined or wrist / finger split)."""
    return _apply_residual_l2_terms(_residual_act_name, args_cli, cfg)


def _apply_residual_l2_terms(_residual_act_name, args_cli, cfg):
    """Finger penalties and the residual-L2 reward terms."""
    _fplan, _fspan, _fspan_src, _res_l2_w, _wrist_l2 = _apply_finger_penalties(args_cli)
    if _res_l2_w > 0.0 and hasattr(cfg, "rewards"):
        _res_l2 = getattr(cfg.rewards, "residual_l2", None)
        if _wrist_l2 is None:
            _combined_residual_l2(args_cli, cfg, _res_l2, _res_l2_w)
        else:
            _finger_dz, _wrist_dz = _split_residual_l2_wrist(args_cli)
            _wrist_l2_w = float(_wrist_l2)
            if _res_l2 is not None:
                # Repurpose the existing residual_l2 term as the FINGER (penalty plan) term — it reads the run's
                # single residual term (uni) / the right term (bimanual) = _residual_act_name().
                _act_r = _residual_act_name()
                _fdims, _fsc, _fdzs = _finger_penalty_side(cfg, args_cli, _fspan, _act_r, _fplan)
                _res_l2.weight = _res_l2_w * _fsc
                _res_l2.params["dims"] = _fdims
                _res_l2.params["deadzone"] = _finger_dz * _fdzs
                _finger_penalty_user(_fplan, _act_r, "residual_l2", _res_l2_w, _res_l2, _finger_dz)
                # Reuse the RewTerm class + func ref from the existing term to avoid imports.
                _RewTermCls = type(_res_l2)
                _func = _res_l2.func
                _wrist_term = _RewTermCls(
                    func=_func, weight=_wrist_l2_w, params={"deadzone": _wrist_dz, "side": "right", "dims": (0, 6)}
                )
                cfg.rewards.residual_l2_wrist = _wrist_term
                print(
                    f"[train_cfg_overrides] RESIDUAL-L2 SPLIT: wrist(0:6) w={_wrist_l2_w} dz={_wrist_dz} "
                    f"| finger({_fdims[0]}:{_fdims[1]}) w={_res_l2.weight:.6g} "
                    f"dz={_res_l2.params['deadzone']:.6g}"
                )
            else:  # a meaningful weight with no consumer — not a WARNING
                raise InputMissingError(
                    "[train_cfg_overrides] --residual_l2_weight > 0 (split) but "
                    "cfg.rewards.residual_l2 is missing — refusing to run without the finger penalty"
                )
            # Bimanual: split the LEFT anchor the same way.
            if bool(getattr(args_cli, "bimanual", 0)):
                _l2_lscale = float(getattr(args_cli, "residual_l2_left_scale", 1.0))
                _res_l2_l = getattr(cfg.rewards, "residual_l2_left", None)
                if _res_l2_l is not None:
                    _fdims_l, _fsc_l, _fdzs_l = _finger_penalty_side(
                        cfg, args_cli, _fspan, "left_hand_joint_pos", _fplan
                    )
                    _res_l2_l.weight = _res_l2_w * _l2_lscale * _fsc_l
                    _res_l2_l.params["dims"] = _fdims_l
                    _res_l2_l.params["deadzone"] = _finger_dz * _fdzs_l
                    _finger_penalty_user(
                        _fplan, "left_hand_joint_pos", "residual_l2_left", _res_l2_w * _l2_lscale, _res_l2_l, _finger_dz
                    )
                    _RewTermClsL = type(_res_l2_l)
                    _funcL = _res_l2_l.func
                    _wrist_term_l = _RewTermClsL(
                        func=_funcL,
                        weight=_wrist_l2_w * _l2_lscale,
                        params={"deadzone": _wrist_dz, "side": "left", "dims": (0, 6)},
                    )
                    cfg.rewards.residual_l2_wrist_left = _wrist_term_l
                    print(
                        f"[train_cfg_overrides] (bimanual) RESIDUAL-L2 SPLIT LEFT: wrist(0:6) w={_wrist_term_l.weight} "
                        f"dz={_wrist_dz} | finger({_fdims_l[0]}:{_fdims_l[1]}) w={_res_l2_l.weight:.6g} "
                        f"dz={_res_l2_l.params['deadzone']:.6g} "
                        f"(left scale x{_l2_lscale})"
                    )
                else:  # a meaningful weight with no consumer — not a WARNING
                    raise InputMissingError(
                        "[train_cfg_overrides] (bimanual) --residual_l2_weight > 0 (split) but "
                        "cfg.rewards.residual_l2_left is missing — refusing to run without the left "
                        "finger penalty"
                    )
    return _fplan, _fspan, _fspan_src, _res_l2_w, _wrist_l2


def _split_residual_l2_wrist(args_cli):
    """SPLIT residual_l2: wrist(0:6) + finger (penalty plan — dims · weight x scale)."""
    _base_dz = float(getattr(args_cli, "residual_l2_deadzone", 0.5))
    _wrist_dz = (
        float(args_cli.wrist_residual_l2_deadzone)
        if getattr(args_cli, "wrist_residual_l2_deadzone", None) is not None
        else _base_dz
    )
    _finger_dz = (
        float(args_cli.finger_residual_l2_deadzone)
        if getattr(args_cli, "finger_residual_l2_deadzone", None) is not None
        else _base_dz
    )
    return _finger_dz, _wrist_dz


def _apply_finger_penalties(args_cli):
    """Finger penalties: the two terms (split residual-L2 finger term · finger action-rate) read, per side, that side's
    finger dims with weight x w_scale (and the residual-L2 norm deadzone x dz_scale) — owner
    core.hands.finger_penalty_plan (canon derived_norm = every finger, weight x REF_N/n, deadzone x sqrt(n/REF_N) ·
    legacy = the old (6, 12) literal x1, explicit only). Marker line per side after both blocks.
    """
    _fspan, _fspan_src = _finger_penalty_span(args_cli)
    _fplan = {}

    # Exp-1 stability: soft residual-L2 penalty (--residual_l2_weight > 0 activates).
    _res_l2_w = float(getattr(args_cli, "residual_l2_weight", 0.0))
    # Optional SPLIT of residual_l2 into wrist(0:6) + finger (span/weight = the finger-penalty plan).
    # None (default) => combined 12-d residual_l2.
    _wrist_l2 = getattr(args_cli, "wrist_residual_l2_weight", None)
    return _fplan, _fspan, _fspan_src, _res_l2_w, _wrist_l2


def _combined_residual_l2(args_cli, cfg, _res_l2, _res_l2_w):
    """Combined 12-d residual_l2."""
    if _res_l2 is not None:
        _res_l2.weight = _res_l2_w
        _res_l2.params["deadzone"] = float(getattr(args_cli, "residual_l2_deadzone", 0.5))
        print(
            f"[train_cfg_overrides] RESIDUAL-L2: weight={_res_l2_w} "
            f"deadzone={_res_l2.params['deadzone']} (soft anchor vs actor-gain drift)"
        )
    else:
        print("[train_cfg_overrides] WARNING: --residual_l2_weight set but cfg.rewards.residual_l2 missing")
    # Bimanual: mirror the SAME override onto the LEFT residual-L2 anchor.
    # Without this, residual_l2_left stays at its hardcoded weight=0.0 (the left
    # tower has no actor-gain anchor and logs 0.000 = a weight-0 RewardManager skip).
    if bool(getattr(args_cli, "bimanual", 0)):
        _l2_lscale = float(getattr(args_cli, "residual_l2_left_scale", 1.0))
        _res_l2_l = getattr(cfg.rewards, "residual_l2_left", None)
        if _res_l2_l is not None:
            _res_l2_l.weight = _res_l2_w * _l2_lscale
            _res_l2_l.params["deadzone"] = float(getattr(args_cli, "residual_l2_deadzone", 0.5))
            print(
                f"[train_cfg_overrides] (bimanual) RESIDUAL-L2-LEFT mirrored: "
                f"weight={_res_l2_l.weight} (x{_l2_lscale}) deadzone={_res_l2_l.params['deadzone']}"
            )
        else:
            print("[train_cfg_overrides] (bimanual) WARNING: residual_l2_left missing — left anchor NOT set")


def _action_rate_overrides(args_cli, cfg, _fplan, _fspan):
    """Wrist and finger action-rate penalties."""
    # WRIST ACTION-RATE: anti-trembling penalty on the step-to-step
    # CHANGE of the wrist residual (dims 0:6). Fingers excluded (grasp-quality
    # guard). Default 0 -> block skipped.
    _war_w = float(getattr(args_cli, "wrist_action_rate_weight", 0.0) or 0.0)
    if _war_w > 0.0 and hasattr(cfg, "rewards"):
        _war_tmpl = getattr(cfg.rewards, "residual_l2", None)
        if _war_tmpl is None:
            print(
                "[train_cfg_overrides] WARNING: --wrist_action_rate_weight set but "
                "cfg.rewards.residual_l2 missing (needed as RewTerm template) — NOT wired"
            )
        else:
            from dexengine.rl.tasks.mdp.rewards.task import residual_action_rate as _war_fn

            _WarCls = type(_war_tmpl)
            _war_sides = (
                ["right", "left"]
                if bool(getattr(args_cli, "bimanual", 0))
                else [str(getattr(args_cli, "side", "right") or "right")]
            )
            for _ws in _war_sides:
                setattr(
                    cfg.rewards,
                    f"wrist_action_rate_{_ws}",
                    _WarCls(func=_war_fn, weight=_war_w, params={"side": _ws, "dims": (0, 6)}),
                )
                print(
                    f"[train_cfg_overrides] WRIST ACTION-RATE ({_ws}): weight={_war_w} "
                    f"dims=(0,6) — -(a_t - a_(t-1))^2 anti-trembling term"
                )

    # FINGER ACTION-RATE: the same rate penalty on the FINGER residual
    # (dims = the penalty plan per side; legacy = dims 6 to 11). The wrist-only default was a deliberate
    # grasp-quality guard, but
    # measurement on hard clips shows the finger targets CHATTER — |d_target| p95 is
    # about 4x higher on a clip that never succeeds than on one that does, with finger
    # tracking error 3.2x the pure-PD prediction (blocked/thrashing) and finger residual
    # authority at scale 1.5. This flag tests whether smoothing finger commands lets a
    # grasp actually form. Default 0 -> block skipped.
    _far_w = float(getattr(args_cli, "finger_action_rate_weight", 0.0) or 0.0)
    if _far_w > 0.0 and hasattr(cfg, "rewards"):
        _far_tmpl = getattr(cfg.rewards, "residual_l2", None)
        if _far_tmpl is None:  # a meaningful weight with no consumer — not a WARNING ("NOT wired")
            raise InputMissingError(
                "[train_cfg_overrides] --finger_action_rate_weight > 0 but the RewTerm template "
                "cfg.rewards.residual_l2 is missing — refusing to run without the finger action-rate "
                "penalty"
            )
        from dexengine.rl.tasks.mdp.rewards.task import residual_action_rate as _far_fn

        _FarCls = type(_far_tmpl)
        _far_sides = (
            ["right", "left"]
            if bool(getattr(args_cli, "bimanual", 0))
            else [str(getattr(args_cli, "side", "right") or "right")]
        )
        for _fs in _far_sides:
            _act_f = f"{_fs}_hand_joint_pos"  # the term residual_action_rate reads for this side
            _fp_f = _finger_penalty_side(cfg, args_cli, _fspan, _act_f, _fplan)  # rate term: no deadzone
            _far_term = _FarCls(func=_far_fn, weight=_far_w * _fp_f.w_scale, params={"side": _fs, "dims": _fp_f.dims})
            setattr(cfg.rewards, f"finger_action_rate_{_fs}", _far_term)
            _finger_penalty_user(_fplan, _act_f, f"finger_action_rate_{_fs}", _far_w, _far_term)
            print(
                f"[train_cfg_overrides] FINGER ACTION-RATE ({_fs}): weight={_far_term.weight:.6g} "
                f"dims=({_fp_f.dims[0]},{_fp_f.dims[1]}) — anti-chatter on finger commands"
            )
