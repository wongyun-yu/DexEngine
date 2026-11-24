"""Left-hand contact-match terms and diagnostics of the contact pilot (bimanual runs), the mirror of the right block in
env_setup.contact_match."""

from __future__ import annotations

import os


def wire_contact_match_left(_match_on, args_cli, cfg):
    """Left-hand contact-match terms and diagnostics (bimanual): the mirror of the right block."""
    # -----------------------------------------------------------------
    # (4-left) Bimanual Option 2: LEFT contact-pilot reward mirror. Full
    # mirror of the right (4) block — same reward formulations + diagnostics,
    # side='left', object_key='object_left', mesh_dir=--object_mesh_dir_left,
    # term names suffixed '_left'. The LEFT contact annotation is loaded via
    # the ContactMatchBase _contact_source_eff suffix ('<source>_left'). Gated
    # by --bimanual (default OFF → right baseline byte-identical). Requires
    # --enable_contact_match_reward 1 (the master gate `_match_on` above) AND
    # --object_mesh_dir_left.
    # -----------------------------------------------------------------
    if _match_on and bool(getattr(args_cli, "bimanual", 0)):
        _mesh_dir_l = getattr(args_cli, "object_mesh_dir_left", None)
        if _mesh_dir_l is None:
            print(
                "[contact_pilot] (bimanual) WARNING: --object_mesh_dir_left not set; "
                "LEFT contact rewards CANNOT be wired and were skipped."
            )
        else:
            from isaaclab.managers import RewardTermCfg as _RewTermL

            from dexengine.rl.tasks.mdp.rewards.contact import RewardContactPenalty as _RCPenaltyL
            from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagContactMatchRate as _DiagRateL
            from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagContactPrecision as _DiagPrecL
            from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagContactRecall as _DiagRecallL
            from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagGraspTargetDist as _DiagGTDistL
            from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagIndexContact as _DiagIndexL
            from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagRefContactFrac as _DiagRefFracL
            from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagRobotContactFrac as _DiagRobFracL
            from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagThumbContact as _DiagThumbL
            from dexengine.rl.tasks.mdp.rewards.contact_grasp import RewardApproachSoft as _RCApproachL
            from dexengine.rl.tasks.mdp.rewards.contact_grasp import RewardGraspTargetSoft as _RCGraspL
            from dexengine.rl.tasks.mdp.rewards.contact_match import RewardContactMatch as _RCMatchL

            _source = str(getattr(args_cli, "contact_source", "mano"))
            _w = float(getattr(args_cli, "contact_reward_weight", 0.5))
            _F = float(getattr(args_cli, "contact_force_threshold", 0.5))
            _d = float(getattr(args_cli, "contact_distance_threshold", 0.01))
            _mode = str(getattr(args_cli, "contact_reward_mode", "binary"))
            _k = float(getattr(args_cli, "contact_softness_k", 50.0))
            _seq_dir = os.path.abspath(getattr(cfg.trajectory, "data_dir", args_cli.data_dir))
            _register_contact_match_left(
                args_cli,
                cfg,
                _DiagGTDistL,
                _DiagIndexL,
                _DiagPrecL,
                _DiagRateL,
                _DiagRecallL,
                _DiagRefFracL,
                _DiagRobFracL,
                _DiagThumbL,
                _F,
                _RCApproachL,
                _RCGraspL,
                _RCMatchL,
                _RCPenaltyL,
                _RewTermL,
                _d,
                _k,
                _mesh_dir_l,
                _mode,
                _seq_dir,
                _source,
                _w,
            )


def _register_contact_match_left(
    args_cli,
    cfg,
    _DiagGTDistL,
    _DiagIndexL,
    _DiagPrecL,
    _DiagRateL,
    _DiagRecallL,
    _DiagRefFracL,
    _DiagRobFracL,
    _DiagThumbL,
    _F,
    _RCApproachL,
    _RCGraspL,
    _RCMatchL,
    _RCPenaltyL,
    _RewTermL,
    _d,
    _k,
    _mesh_dir_l,
    _mode,
    _seq_dir,
    _source,
    _w,
):
    """Register the left contact-match reward, its shaping/closure terms and diagnostics (object_left)."""
    # side='left' + object_key='object_left'; ContactMatchBase appends
    # '_left' to contact_source internally to load contact_<src>_left.npz.
    _paramsL = {
        "side": "left",
        "contact_source": _source,
        "seq_dir": _seq_dir,
        "mesh_dir": os.path.abspath(_mesh_dir_l),
        "F_threshold": _F,
        "d_threshold": _d,
        "object_key": "object_left",
    }
    if _mode == "soft":
        _reward_clsL = _RCGraspL
    else:
        _reward_clsL = {"binary": _RCMatchL, "penalty": _RCPenaltyL}[_mode]
    _reward_paramsL = dict(_paramsL)
    if _mode == "soft":
        _reward_paramsL["softness_k"] = _k
    cfg.rewards.contact_match_left = _RewTermL(func=_reward_clsL, weight=_w, params=_reward_paramsL)
    print(
        f"[contact_pilot] (left_present) LEFT contact reward '{_mode}' "
        f"({_reward_clsL.__name__}) added: weight={_w} mesh_dir={_mesh_dir_l}"
    )
    if bool(getattr(args_cli, "enable_approach_reward", 0)):
        _aw = float(getattr(args_cli, "contact_approach_weight", 1.5))
        _ak = float(getattr(args_cli, "contact_approach_k", 12.0))
        _apL = dict(_paramsL)
        _apL["softness_k"] = _ak
        cfg.rewards.contact_approach_left = _RewTermL(func=_RCApproachL, weight=_aw, params=_apL)
        print(f"[contact_pilot] (left_present) LEFT approach reward added: weight={_aw} k={_ak:.1f}")
    _wire_shaping_terms_left(args_cli, cfg, _RewTermL, _paramsL)
    _pfL = int(bool(getattr(args_cli, "closure_perfinger", 0)))
    _marginsL = [int(x) for x in str(getattr(args_cli, "closure_margins", "2,5,5,5,5")).split(",")]
    _wire_closure_terms_left(args_cli, cfg, _RewTermL, _marginsL, _paramsL, _pfL)
    _diag_mapL = {
        "diag_contact_match_rate_left": _DiagRateL,
        "diag_contact_precision_left": _DiagPrecL,
        "diag_contact_recall_left": _DiagRecallL,
        "diag_ref_contact_frac_left": _DiagRefFracL,
        "diag_robot_contact_frac_left": _DiagRobFracL,
        "diag_thumb_contact_left": _DiagThumbL,
        "diag_index_contact_left": _DiagIndexL,
        "diag_grasp_target_dist_left": _DiagGTDistL,
    }
    for _name, _cls in _diag_mapL.items():
        setattr(cfg.rewards, _name, _RewTermL(func=_cls, weight=1e-9, params=dict(_paramsL)))
    print("[contact_pilot] (left_present) LEFT contact diagnostics added (×1e-9)")


def _wire_shaping_terms_left(args_cli, cfg, _RewTermL, _paramsL):
    """Force-direction, pre-grasp, grasp-geometry and aperture terms of the left hand (statements unchanged)."""
    if bool(getattr(args_cli, "enable_force_dir_reward", 0)):
        from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagForceActive as _DForceActL
        from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagForcePos as _DForcePosL
        from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagForceProj as _DForceProjL
        from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagForceZ as _DForceZL
        from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagTIBoth as _DTIBothL
        from dexengine.rl.tasks.mdp.rewards.contact import RewardForceDirection as _RCForceL

        _fw = float(getattr(args_cli, "force_reward_weight", 2.5))
        _fpL = dict(_paramsL)
        _fpL["force_cap"] = float(getattr(args_cli, "force_cap", 30.0))
        _fpL["force_k"] = int(getattr(args_cli, "force_k", 3))
        _fpL["force_motion_eps"] = float(getattr(args_cli, "force_motion_eps", 0.003))
        _fpL["force_window_pre"] = int(getattr(args_cli, "force_window_pre", 3))
        _fpL["force_window_post"] = int(getattr(args_cli, "force_window_post", 50))
        cfg.rewards.force_dir_left = _RewTermL(func=_RCForceL, weight=_fw, params=dict(_fpL))
        for _fn, _fc in {
            "diag_force_proj_left": _DForceProjL,
            "diag_force_z_left": _DForceZL,
            "diag_force_pos_left": _DForcePosL,
            "diag_force_active_left": _DForceActL,
            "diag_ti_both_left": _DTIBothL,
        }.items():
            setattr(cfg.rewards, _fn, _RewTermL(func=_fc, weight=1e-9, params=dict(_fpL)))
        print(f"[contact_pilot] (left_present) LEFT force-direction reward added: weight={_fw}")
    if bool(getattr(args_cli, "enable_pregrasp_reward", 0)):
        from dexengine.rl.tasks.mdp.rewards.contact_grasp import RewardDiagPalmDist as _DPalmL
        from dexengine.rl.tasks.mdp.rewards.contact_grasp import RewardPreGraspPalm as _RCPreL

        _pw = float(getattr(args_cli, "pregrasp_weight", 2.0))
        _ppL = dict(_paramsL)
        _ppL["pregrasp_k"] = float(getattr(args_cli, "pregrasp_k", 8.0))
        _ppL["pregrasp_dist"] = float(getattr(args_cli, "pregrasp_dist", 0.12))
        cfg.rewards.pregrasp_palm_left = _RewTermL(func=_RCPreL, weight=_pw, params=dict(_ppL))
        cfg.rewards.diag_palm_dist_left = _RewTermL(func=_DPalmL, weight=1e-09, params=dict(_paramsL))
        print(f"[contact_pilot] (left_present) LEFT pre-grasp palm reward added: weight={_pw}")
    if bool(getattr(args_cli, "enable_grasp_geom_reward", 0)):
        from dexengine.rl.tasks.mdp.rewards.contact_grasp_shape import RewardDiagNFingersContact as _DNFL
        from dexengine.rl.tasks.mdp.rewards.contact_grasp_shape import RewardDiagOppo as _DOppoL
        from dexengine.rl.tasks.mdp.rewards.contact_grasp_shape import RewardGraspOpposition as _RCOppoL
        from dexengine.rl.tasks.mdp.rewards.contact_grasp_shape import RewardGraspSurface as _RCSurfL
        from dexengine.rl.tasks.mdp.rewards.contact_grasp_shape import RewardPreGraspOrient as _RCOrientL

        _sw = float(getattr(args_cli, "grasp_surface_weight", 3.0))
        _spL = dict(_paramsL)
        _spL["surface_k"] = float(getattr(args_cli, "grasp_surface_k", 40.0))
        _spL["nonthumb_scale"] = float(getattr(args_cli, "grasp_nonthumb_scale", 1.5))
        cfg.rewards.grasp_surface_left = _RewTermL(func=_RCSurfL, weight=_sw, params=dict(_spL))
        _ow = float(getattr(args_cli, "pregrasp_orient_weight", 2.0))
        if _ow > 0:
            cfg.rewards.pregrasp_orient_left = _RewTermL(func=_RCOrientL, weight=_ow, params=dict(_paramsL))
        if bool(getattr(args_cli, "enable_opposition", 0)):
            _pw2 = float(getattr(args_cli, "grasp_oppo_weight", 2.0))
            _opL = dict(_paramsL)
            _opL["oppo_near_m"] = float(getattr(args_cli, "grasp_oppo_near_m", 0.03))
            cfg.rewards.grasp_oppo_left = _RewTermL(func=_RCOppoL, weight=_pw2, params=dict(_opL))
        for _dn, _dc in {"diag_n_fingers_left": _DNFL, "diag_oppo_left": _DOppoL}.items():
            setattr(cfg.rewards, _dn, _RewTermL(func=_dc, weight=1e-9, params=dict(_paramsL)))
        print("[contact_pilot] (left_present) LEFT grasp-geom rewards added")
    if bool(getattr(args_cli, "enable_aperture_reward", 0)):
        from dexengine.rl.tasks.mdp.rewards.contact_grasp_shape import RewardDiagApertureDist as _DApL
        from dexengine.rl.tasks.mdp.rewards.contact_grasp_shape import RewardGraspAperture as _RCApL

        _apw = float(getattr(args_cli, "aperture_weight", 4.0))
        _aptgt = str(getattr(args_cli, "aperture_target", "mano"))
        _appL = dict(_paramsL)
        _appL["aperture_k"] = float(getattr(args_cli, "aperture_k", 10.0))
        _appL["aperture_target"] = _aptgt
        cfg.rewards.grasp_aperture_left = _RewTermL(func=_RCApL, weight=_apw, params=dict(_appL))
        _dapL = dict(_paramsL)
        _dapL["aperture_target"] = _aptgt
        cfg.rewards.diag_aperture_dist_left = _RewTermL(func=_DApL, weight=1e-09, params=dict(_dapL))
        print(f"[contact_pilot] (left_present) LEFT pinch-aperture reward added: weight={_apw}")


def _wire_closure_terms_left(args_cli, cfg, _RewTermL, _marginsL, _paramsL, _pfL):
    """Closure-force and closure-potential terms of the left hand (statements unchanged)."""
    if bool(getattr(args_cli, "enable_closure_force", 0)):
        from dexengine.rl.tasks.mdp.rewards.contact_closure import RewardClosureForce as _RCClosFL

        _cfw = float(getattr(args_cli, "closure_force_weight", 2.0))
        _cfpL = dict(_paramsL)
        _cfpL["cap"] = float(getattr(args_cli, "closure_force_cap", 5.0))
        _cfpL["nonthumb_scale"] = float(getattr(args_cli, "closure_force_nonthumb_scale", 1.5))
        _cfpL["perfinger"] = _pfL
        _cfpL["margins"] = _marginsL
        _cfpL["contact_f_min"] = 0.05
        _cfpL["force_immediate"] = int(bool(getattr(args_cli, "closure_force_immediate", 0)))
        cfg.rewards.closure_force_left = _RewTermL(func=_RCClosFL, weight=_cfw, params=dict(_cfpL))
        print(f"[contact_pilot] (left_present) LEFT closure-force reward added: weight={_cfw}")
        from dexengine.rl.tasks.mdp.rewards.contact_closure import RewardDiagClosureFmax as _DClosFmaxL
        from dexengine.rl.tasks.mdp.rewards.contact_closure import RewardDiagClosureFN as _DClosFNL
        from dexengine.rl.tasks.mdp.rewards.contact_closure import RewardDiagClosureSat as _DClosSatL

        for _dn, _dc in {
            "diag_closure_fN_left": _DClosFNL,
            "diag_closure_sat_left": _DClosSatL,
            "diag_closure_fmax_left": _DClosFmaxL,
        }.items():
            setattr(
                cfg.rewards,
                _dn,
                _RewTermL(func=_dc, weight=1e-9, params={**dict(_paramsL), "cap": _cfpL["cap"]}),
            )
    if bool(getattr(args_cli, "enable_closure_potential", 0)):
        from dexengine.rl.tasks.mdp.rewards.contact_closure import RewardClosurePotential as _RCClosPL

        _cpw = float(getattr(args_cli, "closure_potential_weight", 2.0))
        _cppL = dict(_paramsL)
        _cppL["clamp_cm"] = float(getattr(args_cli, "closure_potential_clamp_cm", 1.0))
        _cppL["perfinger"] = _pfL
        _cppL["margins"] = _marginsL
        _cppL["nonthumb_scale"] = float(getattr(args_cli, "closure_force_nonthumb_scale", 1.5))
        _cppL["pot_dist_mode"] = str(getattr(args_cli, "closure_potential_dist", "gt_target"))
        cfg.rewards.closure_potential_left = _RewTermL(func=_RCClosPL, weight=_cpw, params=dict(_cppL))
        print(f"[contact_pilot] (left_present) LEFT closure-potential reward added: weight={_cpw}")
