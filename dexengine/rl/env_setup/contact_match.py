"""Contact-match reward wiring of the contact pilot (right hand, then the left hand in bimanual runs)."""

from __future__ import annotations

import os

from dexengine.rl.env_setup.contact_match_left import wire_contact_match_left
from dexengine.rl.env_setup.objects import abspath_multi


def wire_contact_match(*, _match_on, args_cli, cfg):
    """Contact-match reward terms and diagnostics of the right hand, then of the left hand in bimanual runs."""
    _wire_contact_match_right(_match_on, args_cli, cfg)

    wire_contact_match_left(_match_on, args_cli, cfg)


def _wire_contact_match_right(_match_on, args_cli, cfg):
    """Right-hand contact-match reward, its diagnostics and the optional force-direction, pre-grasp, grasp-geometry,
    aperture, closure and slip terms."""
    if _match_on:
        _mesh_dir = getattr(args_cli, "object_mesh_dir", None)
        if _mesh_dir is None:
            print(
                "[contact_pilot] WARNING: --object_mesh_dir not set; "
                "RewardContactMatch + diagnostics CANNOT be wired and were skipped."
            )
        else:
            from isaaclab.managers import RewardTermCfg as _RewTerm

            from dexengine.rl.tasks.mdp.rewards.contact import RewardContactPenalty as _RCPenalty
            from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagContactMatchRate as _DiagRate
            from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagContactPrecision as _DiagPrec
            from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagContactRecall as _DiagRecall
            from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagGraspTargetDist as _DiagGTDist
            from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagIndexContact as _DiagIndex
            from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagRefContactFrac as _DiagRefFrac
            from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagRobotContactFrac as _DiagRobFrac
            from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagThumbContact as _DiagThumb
            from dexengine.rl.tasks.mdp.rewards.contact_grasp import RewardApproachSoft as _RCApproach
            from dexengine.rl.tasks.mdp.rewards.contact_grasp import RewardGraspTargetSoft as _RCGrasp
            from dexengine.rl.tasks.mdp.rewards.contact_match import RewardContactMatch as _RCMatch

            _grasp_target_on = bool(getattr(args_cli, "enable_grasp_target", 0))
            _source = str(getattr(args_cli, "contact_source", "mano"))
            _w = float(getattr(args_cli, "contact_reward_weight", 0.5))
            _F = float(getattr(args_cli, "contact_force_threshold", 0.5))
            _d = float(getattr(args_cli, "contact_distance_threshold", 0.01))
            _mode = str(getattr(args_cli, "contact_reward_mode", "binary"))
            _k = float(getattr(args_cli, "contact_softness_k", 50.0))
            # seq_dir = trajectory ROOT dir (containing <seq>/0/contact_<source>.npz).
            # cfg.trajectory.data_dir is set in create_env before this fn runs.
            _seq_dir = os.path.abspath(getattr(cfg.trajectory, "data_dir", args_cli.data_dir))
            # Shared params (diagnostics use exactly these — no softness_k).
            _params = {
                "side": "right",
                "contact_source": _source,
                "seq_dir": _seq_dir,
                "mesh_dir": abspath_multi(_mesh_dir),
                "F_threshold": _F,
                "d_threshold": _d,
            }
            _DIAG_W = _register_contact_match_right(
                args_cli,
                cfg,
                _F,
                _RCApproach,
                _RCGrasp,
                _RCMatch,
                _RCPenalty,
                _RewTerm,
                _d,
                _grasp_target_on,
                _k,
                _mode,
                _params,
                _seq_dir,
                _source,
                _w,
            )
            _diag_map = {
                "diag_contact_match_rate": _DiagRate,
                "diag_contact_precision": _DiagPrec,
                "diag_contact_recall": _DiagRecall,
                "diag_ref_contact_frac": _DiagRefFrac,
                "diag_robot_contact_frac": _DiagRobFrac,
                "diag_thumb_contact": _DiagThumb,
                "diag_index_contact": _DiagIndex,
                "diag_grasp_target_dist": _DiagGTDist,
            }
            for _name, _cls in _diag_map.items():
                setattr(cfg.rewards, _name, _RewTerm(func=_cls, weight=_DIAG_W, params=dict(_params)))
                print(
                    f"[contact_pilot] diagnostic (weight={_DIAG_W:g} ≈0; "
                    f"Episode_Reward scaled by {_DIAG_W:g}) added: {_name}"
                )


def _register_contact_match_right(
    args_cli,
    cfg,
    _F,
    _RCApproach,
    _RCGrasp,
    _RCMatch,
    _RCPenalty,
    _RewTerm,
    _d,
    _grasp_target_on,
    _k,
    _mode,
    _params,
    _seq_dir,
    _source,
    _w,
):
    """Register the right contact-match reward in the --contact_reward_mode formulation and its follow-up terms."""
    # Reward formulation selected by --contact_reward_mode. The registered term
    # name stays "contact_match_right" for all modes so the curriculum patch +
    # monitors keep finding it.
    #   binary  -> RewardContactMatch    (object-filtered force gate)
    #   soft    -> RewardGraspTargetSoft (witness_obj GT-target distance) ALWAYS.
    #              The legacy nearest-surface RewardContactSoft is gone
    #              (it rewarded approaching any surface and was
    #              reward-hackable). --enable_grasp_target is redundant (soft is
    #              always GT-target); kept only as an informational flag.
    #   penalty -> RewardContactPenalty  (object-filtered force gate)
    if _mode == "soft":
        _reward_cls = _RCGrasp
        if not _grasp_target_on:
            print(
                "[contact_pilot] NOTE: --contact_reward_mode soft always uses the "
                "witness_obj GT-target (RewardGraspTargetSoft); the nearest-surface "
                "RewardContactSoft was removed (dead code)."
            )
    else:
        _reward_cls = {"binary": _RCMatch, "penalty": _RCPenalty}[_mode]
    _reward_params = dict(_params)
    if _mode == "soft":
        _reward_params["softness_k"] = _k
    cfg.rewards.contact_match_right = _RewTerm(func=_reward_cls, weight=_w, params=_reward_params)
    print(
        f"[contact_pilot] contact reward '{_mode}' ({_reward_cls.__name__}, "
        f"right) added: weight={_w} source={_source} F={_F}N "
        f"d={_d * 100:.1f}cm"
        + (f" k={_k:.1f}" if _mode == "soft" else "")
        + (" [GT-target]" if (_mode == "soft" and _grasp_target_on) else " [nearest-surface]")
        + f" seq_dir={_seq_dir}"
    )
    return _wire_approach_and_closure_right(args_cli, cfg, _RCApproach, _RewTerm, _params)


def _wire_approach_and_closure_right(args_cli, cfg, _RCApproach, _RewTerm, _params):
    """Dense GT-target approach reward, the shaping terms and the closure/slip terms; returns the diagnostic weight."""
    # Exp6B/7B: dense GT-target approach reward (not ref-gated), will-contact mask.
    if bool(getattr(args_cli, "enable_approach_reward", 0)):
        _aw = float(getattr(args_cli, "contact_approach_weight", 1.5))
        _ak = float(getattr(args_cli, "contact_approach_k", 12.0))
        _ap = dict(_params)
        _ap["softness_k"] = _ak
        cfg.rewards.contact_approach_right = _RewTerm(func=_RCApproach, weight=_aw, params=_ap)
        print(
            f"[contact_pilot] DENSE approach reward (RewardApproachSoft, GT-target) "
            f"added: weight={_aw} k={_ak:.1f} (will-contact fingers, all frames)"
        )
    _wire_shaping_terms_right(args_cli, cfg, _RewTerm, _params)
    # CLOSURE-BOOTSTRAP rewards: force-onset / closing-potential.
    # Wired here (shared create_env path) so train AND eval/render get the
    # identical term when the flag is passed — same mechanism as grasp_geom.
    # A1 per-finger closure (default OFF). margins string "t,i,m,r,p" → per-finger release margin.
    _pf = int(bool(getattr(args_cli, "closure_perfinger", 0)))
    _margins = [int(x) for x in str(getattr(args_cli, "closure_margins", "2,5,5,5,5")).split(",")]
    _wire_closure_and_slip_right(args_cli, cfg, _RewTerm, _margins, _params, _pf)
    # Diagnostic aux metrics → logged via Info/Episode_Reward/<name>.
    # SKRL's native env-extras logging picks up env.extras["log"], which
    # RewardManager.reset() populates as Episode_Reward/<term> for every
    # registered term.
    #
    # Note: Isaac Lab's RewardManager.compute()
    # SKIPS any term whose weight == 0.0 (reward_manager.py:145-147) — the
    # term func is never called and Episode_Reward/<name> stays 0. To make
    # the diagnostics actually execute + log WITHOUT measurably changing
    # the total reward, we register them with a NEGLIGIBLE non-zero weight
    # (_DIAG_W). The contribution to the per-step total reward is then
    # _DIAG_W * value(≤1) ≈ 5e-9 per diag term — numerically zero.
    # The logged Episode_Reward value is correspondingly scaled by
    # _DIAG_W, so multiply back by 1/_DIAG_W (=1e8) when reading the
    # metric off W&B/TensorBoard. (A literal weight=0.0 would log nothing.)
    _DIAG_W = 1e-9
    return _DIAG_W


def _wire_shaping_terms_right(args_cli, cfg, _RewTerm, _params):
    """Force-direction, pre-grasp, grasp-geometry and aperture terms of the right hand (statements unchanged)."""
    # Force-direction reward (hand-on-object force aligned with reference motion).
    if bool(getattr(args_cli, "enable_force_dir_reward", 0)):
        from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagForceActive as _DForceAct
        from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagForcePos as _DForcePos
        from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagForceProj as _DForceProj
        from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagForceZ as _DForceZ
        from dexengine.rl.tasks.mdp.rewards.contact import RewardDiagTIBoth as _DTIBoth
        from dexengine.rl.tasks.mdp.rewards.contact import RewardForceDirection as _RCForce

        _fw = float(getattr(args_cli, "force_reward_weight", 2.5))
        _fp = dict(_params)
        _fp["force_cap"] = float(getattr(args_cli, "force_cap", 30.0))
        _fp["force_k"] = int(getattr(args_cli, "force_k", 3))
        _fp["force_motion_eps"] = float(getattr(args_cli, "force_motion_eps", 0.003))
        _fp["force_window_pre"] = int(getattr(args_cli, "force_window_pre", 3))
        _fp["force_window_post"] = int(getattr(args_cli, "force_window_post", 50))
        cfg.rewards.force_dir_right = _RewTerm(func=_RCForce, weight=_fw, params=dict(_fp))
        print(
            f"[contact_pilot] FORCE-DIRECTION reward added: weight={_fw} "
            f"F_cap={_fp['force_cap']}N k={_fp['force_k']} "
            f"window=[fc-{_fp['force_window_pre']},fc+{_fp['force_window_post']}] "
            f"motion_eps={_fp['force_motion_eps']} (thumb+index soft gate)"
        )
        for _fn, _fc in {
            "diag_force_proj": _DForceProj,
            "diag_force_z": _DForceZ,
            "diag_force_pos": _DForcePos,
            "diag_force_active": _DForceAct,
            "diag_ti_both": _DTIBoth,
        }.items():
            setattr(cfg.rewards, _fn, _RewTerm(func=_fc, weight=1e-9, params=dict(_fp)))
        print("[contact_pilot] force diagnostics added: diag_force_{proj,z,pos,active}, diag_ti_both (×1e-9)")
    # the off side leaves a marker too, paired with the on side's "added" line (the launch probe checks
    # FORCE_DIR)
    else:
        print("[contact_pilot] FORCE-DIRECTION reward OFF (--enable_force_dir_reward 0)")
    _wire_pregrasp_terms_right(args_cli, cfg, _RewTerm, _params)


def _wire_pregrasp_terms_right(args_cli, cfg, _RewTerm, _params):
    """Pre-grasp palm reward (pull the palm to a graspable pose) and the terms that build on it."""
    # PRE-GRASP PALM reward: pull the palm to a graspable pose.
    if bool(getattr(args_cli, "enable_pregrasp_reward", 0)):
        from dexengine.rl.tasks.mdp.rewards.contact_grasp import RewardDiagPalmDist as _DPalm
        from dexengine.rl.tasks.mdp.rewards.contact_grasp import RewardPreGraspPalm as _RCPre

        _pw = float(getattr(args_cli, "pregrasp_weight", 2.0))
        _pp = dict(_params)
        _pp["pregrasp_k"] = float(getattr(args_cli, "pregrasp_k", 8.0))
        _pp["pregrasp_dist"] = float(getattr(args_cli, "pregrasp_dist", 0.12))
        cfg.rewards.pregrasp_palm_right = _RewTerm(func=_RCPre, weight=_pw, params=dict(_pp))
        # diag_palm_dist __call__ takes only the base params (no pregrasp_k/dist).
        cfg.rewards.diag_palm_dist = _RewTerm(func=_DPalm, weight=1e-09, params=dict(_params))
        print(
            f"[contact_pilot] PRE-GRASP PALM reward added: weight={_pw} "
            f"k={_pp['pregrasp_k']} d_grasp={_pp['pregrasp_dist']}m (+ diag_palm_dist ×1e-9)"
        )
    # Wrap-ready pose + object-relative grasp geometry.
    if bool(getattr(args_cli, "enable_grasp_geom_reward", 0)):
        from dexengine.rl.tasks.mdp.rewards.contact_grasp_shape import RewardDiagNFingersContact as _DNF
        from dexengine.rl.tasks.mdp.rewards.contact_grasp_shape import RewardDiagOppo as _DOppo
        from dexengine.rl.tasks.mdp.rewards.contact_grasp_shape import RewardGraspOpposition as _RCOppo
        from dexengine.rl.tasks.mdp.rewards.contact_grasp_shape import RewardGraspSurface as _RCSurf
        from dexengine.rl.tasks.mdp.rewards.contact_grasp_shape import RewardPreGraspOrient as _RCOrient

        _sw = float(getattr(args_cli, "grasp_surface_weight", 3.0))
        _sp = dict(_params)
        _sp["surface_k"] = float(getattr(args_cli, "grasp_surface_k", 40.0))
        _sp["nonthumb_scale"] = float(getattr(args_cli, "grasp_nonthumb_scale", 1.5))
        cfg.rewards.grasp_surface_right = _RewTerm(func=_RCSurf, weight=_sw, params=dict(_sp))
        _ow = float(getattr(args_cli, "pregrasp_orient_weight", 2.0))
        if _ow > 0:  # orient is dropped (it rewarded the above-poke) by setting weight 0
            cfg.rewards.pregrasp_orient_right = _RewTerm(func=_RCOrient, weight=_ow, params=dict(_params))
        print(
            f"[contact_pilot] GRASP-GEOM: surface w={_sw} "
            f"(k={_sp['surface_k']}, nonthumb={_sp['nonthumb_scale']}) + orient w={_ow}"
            + ("" if _ow > 0 else " (orient OFF)")
        )
        if bool(getattr(args_cli, "enable_opposition", 0)):
            _pw2 = float(getattr(args_cli, "grasp_oppo_weight", 2.0))
            _op = dict(_params)
            _op["oppo_near_m"] = float(getattr(args_cli, "grasp_oppo_near_m", 0.03))
            cfg.rewards.grasp_oppo_right = _RewTerm(func=_RCOppo, weight=_pw2, params=dict(_op))
            print(f"[contact_pilot] OPPOSITION reward added: w={_pw2} near={_op['oppo_near_m']}m")
        for _dn, _dc in {"diag_n_fingers": _DNF, "diag_oppo": _DOppo}.items():
            setattr(cfg.rewards, _dn, _RewTerm(func=_dc, weight=1e-9, params=dict(_params)))
        print("[contact_pilot] grasp-geometry diagnostics added: diag_n_fingers, diag_oppo (×1e-9)")
    # PINCH-APERTURE pose reward: guide the wrist so the object sits between thumb & fingers.
    if bool(getattr(args_cli, "enable_aperture_reward", 0)):
        from dexengine.rl.tasks.mdp.rewards.contact_grasp_shape import RewardDiagApertureDist as _DAp
        from dexengine.rl.tasks.mdp.rewards.contact_grasp_shape import RewardGraspAperture as _RCAp

        _apw = float(getattr(args_cli, "aperture_weight", 4.0))
        _aptgt = str(getattr(args_cli, "aperture_target", "mano"))
        _app = dict(_params)
        _app["aperture_k"] = float(getattr(args_cli, "aperture_k", 10.0))
        _app["aperture_target"] = _aptgt
        cfg.rewards.grasp_aperture_right = _RewTerm(func=_RCAp, weight=_apw, params=dict(_app))
        _dap = dict(_params)
        _dap["aperture_target"] = _aptgt
        cfg.rewards.diag_aperture_dist = _RewTerm(func=_DAp, weight=1e-09, params=dict(_dap))
        print(
            f"[contact_pilot] PINCH-APERTURE reward added: weight={_apw} "
            f"k={_app['aperture_k']} target={_aptgt} (+ diag_aperture_dist ×1e-9)"
        )


def _wire_closure_and_slip_right(args_cli, cfg, _RewTerm, _margins, _params, _pf):
    """Closure-force, closure-potential and slip terms of the right hand (statements unchanged)."""
    if bool(getattr(args_cli, "enable_closure_force", 0)):
        from dexengine.rl.tasks.mdp.rewards.contact_closure import RewardClosureForce as _RCClosF

        _cfw = float(getattr(args_cli, "closure_force_weight", 2.0))
        _cfp = dict(_params)
        _cfp["cap"] = float(getattr(args_cli, "closure_force_cap", 5.0))
        _cfp["nonthumb_scale"] = float(getattr(args_cli, "closure_force_nonthumb_scale", 1.5))
        _cfp["perfinger"] = _pf
        _cfp["margins"] = _margins
        _cfp["contact_f_min"] = 0.05
        _cfp["force_immediate"] = int(bool(getattr(args_cli, "closure_force_immediate", 0)))
        cfg.rewards.closure_force_right = _RewTerm(func=_RCClosF, weight=_cfw, params=dict(_cfp))
        print(
            f"[contact_pilot] CLOSURE-FORCE reward added: weight={_cfw} "
            f"cap={_cfp['cap']}N nonthumb={_cfp['nonthumb_scale']} "
            f"perfinger={_pf} margins={_margins if _pf else 'n/a'} "
            f"force_immediate={_cfp['force_immediate'] if _pf else 'n/a'}"
        )
        # Closure-force diagnostics (×1e-9, diag_* family — stripped unless
        # --enable_reward_diagnostics 1): grip force level + cap saturation.
        from dexengine.rl.tasks.mdp.rewards.contact_closure import RewardDiagClosureFmax as _DClosFmax
        from dexengine.rl.tasks.mdp.rewards.contact_closure import RewardDiagClosureFN as _DClosFN
        from dexengine.rl.tasks.mdp.rewards.contact_closure import RewardDiagClosureSat as _DClosSat

        for _dn, _dc in {
            "diag_closure_fN": _DClosFN,
            "diag_closure_sat": _DClosSat,
            "diag_closure_fmax": _DClosFmax,
        }.items():
            setattr(cfg.rewards, _dn, _RewTerm(func=_dc, weight=1e-9, params={**dict(_params), "cap": _cfp["cap"]}))
    if bool(getattr(args_cli, "enable_closure_potential", 0)):
        from dexengine.rl.tasks.mdp.rewards.contact_closure import RewardClosurePotential as _RCClosP

        _cpw = float(getattr(args_cli, "closure_potential_weight", 2.0))
        _cpp = dict(_params)
        _cpp["clamp_cm"] = float(getattr(args_cli, "closure_potential_clamp_cm", 1.0))
        _cpp["perfinger"] = _pf
        _cpp["margins"] = _margins
        _cpp["nonthumb_scale"] = float(getattr(args_cli, "closure_force_nonthumb_scale", 1.5))
        _cpp["pot_dist_mode"] = str(getattr(args_cli, "closure_potential_dist", "gt_target"))
        cfg.rewards.closure_potential_right = _RewTerm(func=_RCClosP, weight=_cpw, params=dict(_cpp))
        _pmode = _cpp["pot_dist_mode"] if _pf else "nearest-surface, hand-level gate"
        print(
            f"[contact_pilot] CLOSURE-POTENTIAL reward added: weight={_cpw} "
            f"clamp=±{_cpp['clamp_cm']}cm perfinger={_pf} pot_dist_mode={_pmode} "
            f"({'per-finger window' if _pf else 'legacy hand-level gate'})"
        )
    # P1 SLIP penalty: pose-agnostic grasp-stability (RewardSlipPenalty).
    # Wired in the shared create_env path -> train AND eval get the identical term.
    if bool(getattr(args_cli, "enable_slip_penalty", 0)):
        from dexengine.rl.tasks.mdp.rewards.contact_grasp import RewardSlipPenalty as _RSlip

        _slw = float(getattr(args_cli, "slip_penalty_weight", 2.0))
        _slp = dict(_params)
        _slp["slip_cap"] = float(getattr(args_cli, "slip_cap", 0.5))
        _slp["slip_f_gate"] = float(getattr(args_cli, "slip_f_gate", 0.05))
        _slp["slip_min_fingers"] = int(getattr(args_cli, "slip_min_fingers", 2))
        cfg.rewards.slip_penalty_right = _RewTerm(func=_RSlip, weight=_slw, params=dict(_slp))
        print(
            f"[contact_pilot] P1 SLIP-PENALTY added: weight={_slw} "
            f"cap={_slp['slip_cap']}m/s f_gate={_slp['slip_f_gate']}N "
            f"min_fingers={_slp['slip_min_fingers']}"
        )
