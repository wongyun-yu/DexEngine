"""Schedule and curriculum edits of ``train_cfg_overrides``: ramps, RSI, keypoint/articulation terms, threshold
tightening."""

import os

import numpy as np

from dexengine.core.desk import DESK_TOP_SURFACE_Z
from dexengine.rl.env_setup.object_rewards import add_obj_keypoint_terms
from dexengine.rl.env_setup.objtrack import verify_objtrack_schedule


def apply_schedule_overrides(*, args_cli, cfg, total_timesteps):
    """Reward-weight scales and the step-based curricula (object-follow, adaptive termination,
    level-1/2 ramp, keypoints, pose randomization, threshold tightening, RSI, hinge schedule)."""

    _weight_scale_and_follow_overrides(args_cli, cfg, total_timesteps)

    _obj_term_adaptive_overrides(args_cli, cfg)

    _level12_ramp_and_object_terms(args_cli, cfg, total_timesteps)

    _pose_rand_threshold_rsi_overrides(args_cli, cfg, total_timesteps)


def _weight_scale_and_follow_overrides(args_cli, cfg, total_timesteps):
    """eef weight scale, object-follow scale and the object-follow schedule."""
    # Scale the eef_pos/eef_rot wrist-tracking weights to LOOSEN the wrist-pin so the
    # policy can deviate to a graspable pose (trades reference-tracking). 1.0 = off.
    _eef_scale = float(getattr(args_cli, "eef_weight_scale", 1.0) or 1.0)
    if _eef_scale != 1.0 and hasattr(cfg, "rewards"):
        for _nm in ("eef_pos", "eef_rot"):
            _rw = getattr(cfg.rewards, _nm, None)
            if _rw is not None:
                _rw.weight = float(_rw.weight) * _eef_scale
                print(f"[train_cfg_overrides] {_nm} weight x{_eef_scale} -> {_rw.weight} (wrist-pin loosen)")

    # exp/obj-reward-boost: scale the OBJECT-FOLLOWING reward weights (obj_pos, obj_rot, obj_vel,
    # obj_ang_vel — all reference-tracking exp(-k*|ref-cur|) terms) so object-following dominates more
    # vs closure_force. Touches ONLY these four terms. 1.0 = off.
    _objf_scale = float(getattr(args_cli, "obj_follow_scale", 1.0) or 1.0)
    if _objf_scale != 1.0 and hasattr(cfg, "rewards"):
        # Bimanual: scale the LEFT object-following terms too, else the right object-follow
        # weight is boosted while the left stays at base (silent curriculum asymmetry).
        _objf_names = ["obj_pos", "obj_rot", "obj_vel", "obj_ang_vel"]
        if bool(getattr(args_cli, "bimanual", 0)):
            _objf_names += ["obj_pos_left", "obj_rot_left", "obj_vel_left", "obj_ang_vel_left"]
        for _nm in _objf_names:
            _rw = getattr(cfg.rewards, _nm, None)
            if _rw is not None:
                _old = float(_rw.weight)
                _rw.weight = _old * _objf_scale
                print(f"[train_cfg_overrides] OBJ-FOLLOW SCALE x{_objf_scale}: {_nm} weight {_old} -> {_rw.weight}")

    # exp/obj-reward-ramp: SCHEDULE the object-following weight scale 1.0 -> target over the first 75% of
    # training (ObjFollowSchedule curriculum), then hold. Same 4 terms as the fixed --obj_follow_scale.
    _objf_target = float(getattr(args_cli, "obj_follow_schedule_target", 0.0) or 0.0)
    if _objf_target > 1.0 and hasattr(cfg, "curriculum") and cfg.curriculum is not None:
        from isaaclab.managers import CurriculumTermCfg as _CurrTerm

        from dexengine.rl.tasks.mdp.curriculums import ObjFollowSchedule as _ObjFollowSched

        _ramp = max(int(0.75 * total_timesteps), 1)
        _kp_obj_terms = ["obj_pos", "obj_rot", "obj_vel", "obj_ang_vel"]
        # Bimanual: ramp the LEFT object-following terms on the same schedule (else right
        # ramps to target while left stays at base — silent curriculum asymmetry).
        if bool(getattr(args_cli, "bimanual", 0)):
            _kp_obj_terms += ["obj_pos_left", "obj_rot_left", "obj_vel_left", "obj_ang_vel_left"]
        # keypoint terms appended below if --obj_keypoint_reward is set
        cfg.curriculum.obj_follow_schedule = _CurrTerm(
            func=_ObjFollowSched, params={"obj_terms": _kp_obj_terms, "target_scale": _objf_target, "ramp_steps": _ramp}
        )
        print(
            f"[train_cfg_overrides] OBJ-FOLLOW SCHEDULE: ramp 1.0 -> {_objf_target} over {_ramp} steps "
            f"(75% of total {total_timesteps}); terms={'/'.join(_kp_obj_terms)}"
        )


def _obj_term_adaptive_overrides(args_cli, cfg):
    """Adaptive object-tracking termination (ObjTermAdaptive)."""
    # adaptive-term v3: performance-gated INDEPENDENT pos/rot termination tightening.
    # pos_start/rot_start stay at the 100% baseline (POS_CONST/ROT_CONST); ObjTermAdaptive drives
    # pos_end/rot_end_deg down in physical steps. threshold_tightening anneal_steps=1 => TRAINING
    # schedule_progress=1 => training gate = pos_end. Frozen eval probes at schedule_progress =
    # --eval_threshold_progress (MUST be 0) => eval gate = pos_start = 100% (clean decision metric).
    if (
        int(getattr(args_cli, "obj_term_adaptive", 0)) == 1
        and hasattr(cfg, "curriculum")
        and cfg.curriculum is not None
    ):
        _ot = getattr(getattr(cfg, "terminations", None), "object_tracking", None)
        if _ot is None:
            print("[train_cfg_overrides] OBJ-TERM ADAPTIVE: object_tracking termination MISSING — skipped")
        else:
            _bpos = float(_ot.params.get("pos_start", 0.10))
            _brot = float(_ot.params.get("rot_start_deg", 180.0))
            _ot.params["pos_end"] = _bpos  # training gate starts at 100%
            _ot.params["rot_end_deg"] = _brot
            if abs(float(args_cli.eval_threshold_progress)) > 1e-9:
                print(
                    f"[train_cfg_overrides] WARNING: --obj_term_adaptive needs --eval_threshold_progress 0 "
                    f"for a FIXED-100% decision probe; got {args_cli.eval_threshold_progress} (confounded)."
                )
            if hasattr(cfg.curriculum, "threshold_tightening") and cfg.curriculum.threshold_tightening is not None:
                cfg.curriculum.threshold_tightening.params["anneal_steps"] = 1
                print(
                    "[train_cfg_overrides] OBJ-TERM ADAPTIVE: threshold_tightening.anneal_steps=1 "
                    "(training schedule_progress=1 -> gate=pos_end/rot_end)"
                )
            # v3 tip-drag capture wiring: farthest FPS keypoint (= knife tip) in object-local coords.
            _ta_kp = None
            try:
                _ta_dir = os.path.abspath(args_cli.object_mesh_dir)  # required; no mesh fallback
                _ta_kps = np.load(os.path.join(_ta_dir, "keypoints_fps2.npy")).astype("float64")
                _ta_sp = np.load(os.path.join(_ta_dir, "surface_points.npz"))["points"].astype("float64")
                _ta_far = int(np.argmax(np.linalg.norm(_ta_kps - _ta_sp.mean(0), axis=1)))
                _ta_kp = [float(x) for x in _ta_kps[_ta_far]]
                _ot.params["tip_kp_local"] = _ta_kp
                _ot.params["tip_desk_z"] = DESK_TOP_SURFACE_Z
            except Exception as _ta_exc:  # noqa: BLE001 — the failure is printed; the schedule runs without this override
                print(
                    f"[train_cfg_overrides] OBJ-TERM ADAPTIVE: tip keypoint unavailable ({_ta_exc}) "
                    f"— tip-drag veto will HOLD (nan) unless --obj_term_adaptive_tip_veto < 0"
                )
            from isaaclab.managers import CurriculumTermCfg as _CurrTermTA

            from dexengine.rl.tasks.mdp.curriculums_objterm import ObjTermAdaptive as _ObjTermAdaptive

            _ta_params = {
                "base_pos": _bpos,
                "base_rot": _brot,
                "pos_step_cm": float(args_cli.obj_term_adaptive_pos_step_cm),
                "rot_step_deg": float(args_cli.obj_term_adaptive_rot_step_deg),
                "pos_floor_cm": float(args_cli.obj_term_adaptive_pos_floor_cm),
                "rot_floor_deg": float(args_cli.obj_term_adaptive_rot_floor_deg),
                "n": int(args_cli.obj_term_adaptive_n),
                "cooldown": 1,
                "dC": 0.01,
                "dP": 2.0,
                "floor_complete": 0.55,
                "q": 0.80,
                "margin_pos_cm": float(args_cli.obj_term_adaptive_margin_pos_cm),
                "margin_rot_deg": float(args_cli.obj_term_adaptive_margin_rot_deg),
                "tolC": 0.05,
                "tolP": 10.0,
                "tip_veto": float(args_cli.obj_term_adaptive_tip_veto),
                "backoff": int(args_cli.obj_term_adaptive_backoff),
                "unconditional": int(args_cli.obj_term_adaptive_unconditional),
            }
            cfg.curriculum.obj_term_adaptive = _CurrTermTA(func=_ObjTermAdaptive, params=_ta_params)
            print(
                f"[train_cfg_overrides] OBJ-TERM ADAPTIVE v3: pos {_bpos * 100:.0f}cm "
                f"-{_ta_params['pos_step_cm']}cm/step floor {_ta_params['pos_floor_cm']}cm | "
                f"rot {_brot:.0f}deg -{_ta_params['rot_step_deg']}deg/step floor "
                f"{_ta_params['rot_floor_deg']}deg | n={_ta_params['n']} "
                f"tip_veto={_ta_params['tip_veto']} backoff={_ta_params['backoff']} "
                f"tip_kp={'OK' if _ta_kp is not None else 'MISSING'}"
            )


def _level12_ramp_and_object_terms(args_cli, cfg, total_timesteps):
    """Level-1/2 ramp-down and the object keypoint / articulation terms."""
    # C5: ramp finger-pose guidance (level_1 [+ level_2]) down to 0 over --ramp_level12_frac of training.
    #   --ramp_level12 1  -> contact_pilot KEEPS level_2 too, so BOTH level_1+level_2 ramp.
    #   --ramp_level1  1  -> level_2 stays REMOVED (default); only level_1 ramps 0.5 -> 0.
    # The term loop below only adds terms PRESENT in cfg.rewards, so --ramp_level1 naturally ramps
    # just level_1 (level_2 is None). default (neither) = no-op (level_1 static, level_2 removed).
    _l12_any = int(getattr(args_cli, "ramp_level12", 0)) == 1 or int(getattr(args_cli, "ramp_level1", 0)) == 1
    if _l12_any and hasattr(cfg, "curriculum") and cfg.curriculum is not None:
        from isaaclab.managers import CurriculumTermCfg as _CurrTerm12

        from dexengine.rl.tasks.mdp.curriculums import Level12RampDown as _L12RD

        _l12_frac = float(getattr(args_cli, "ramp_level12_frac", 0.75))
        _l12_ramp = max(int(_l12_frac * total_timesteps), 1)
        _l12_names = ("level_1", "level_2")
        if bool(getattr(args_cli, "bimanual", 0)):
            _l12_names += ("level_1_left", "level_2_left")
        _l12_terms = [nm for nm in _l12_names if getattr(cfg.rewards, nm, None) is not None]
        if _l12_terms:
            cfg.curriculum.level12_ramp_down = _CurrTerm12(
                func=_L12RD, params={"level_terms": _l12_terms, "ramp_steps": _l12_ramp}
            )
            print(
                f"[train_cfg_overrides] C5 LEVEL12 RAMP-DOWN: base -> 0 over {_l12_ramp} steps "
                f"({_l12_frac * 100:.0f}% of total {total_timesteps}); terms={'/'.join(_l12_terms)}"
            )
        else:
            print(
                "[train_cfg_overrides] C5 ramp: no level_1/level_2 terms found in cfg.rewards; "
                "Level12RampDown not registered"
            )

    # exp/obj-keypoint-reward: FPS object-keypoint position rewards (RewardObjKeypoint).
    #   mode='both'     -> both K=2 keypoints, weight=--obj_keypoint_weight each (default 1.25 = W/2).
    #   mode='farthest' -> ONLY the keypoint FARTHEST from the object centroid (hardest extremity to
    #                      track); weight=--obj_keypoint_weight (use 2.5 for center:keypoint = 2.5:2.5).
    # The obj_pos base weight (2.5) is not changed here.
    # Always registers diag terms (weight 1e-9, logging only) even when obj_keypoint_reward=0
    # (keypointremoval ablations), so the metric is comparable across all experiments.
    add_obj_keypoint_terms(
        cfg,
        args_cli,
        optimize=(int(getattr(args_cli, "obj_keypoint_reward", 0)) == 1),
        extend_ramp=True,
    )
    # Articulated clips: joint-value tracking reward obj_arti[_left] (add_obj_artic_terms; no-op for rigid
    # clips).
    from dexengine.rl.env_setup.object_rewards import add_obj_artic_terms as _aoat

    _aoat(cfg, args_cli, "train_cfg_overrides", extend_ramp=True)


def _pose_rand_threshold_rsi_overrides(args_cli, cfg, total_timesteps):
    """Initial-pose randomization, threshold tightening, the objtrack schedule check, RSI and the last schedule."""
    # Initial pose randomization (--randomize_finger / --randomize_wrist)
    if (args_cli.randomize_finger or args_cli.randomize_wrist) and hasattr(cfg.events, "randomize_pose"):
        cfg.events.randomize_pose.params["randomize_fraction"] = args_cli.randomize_fraction
        cfg.events.randomize_pose.params["finger_open_deg"] = (
            args_cli.finger_open_deg if args_cli.randomize_finger else 0.0
        )
        cfg.events.randomize_pose.params["wrist_z_max_cm"] = (
            args_cli.wrist_z_max_cm if args_cli.randomize_wrist else 0.0
        )
        print(
            f"[train_cfg_overrides] Pose randomization ENABLED: "
            f"fraction={args_cli.randomize_fraction}, "
            f"finger={'ON ' + str(args_cli.finger_open_deg) + '°' if args_cli.randomize_finger else 'OFF'}, "
            f"wrist_z={'ON ' + str(args_cli.wrist_z_max_cm) + 'cm' if args_cli.randomize_wrist else 'OFF'}"
        )

    # Linear threshold schedule: anneal schedule_progress 0->1
    # over the first 50% of training, then hold at 1 (final/tight). The
    # residual policy starts near-zero and needs time to learn object
    # manipulation under relaxed thresholds before they tighten.
    anneal_scale = args_cli.tighten_steps_scale if args_cli.tighten_steps_scale is not None else 0.50
    anneal_steps = max(int(total_timesteps * anneal_scale), 1)
    if hasattr(cfg.curriculum, "threshold_tightening"):
        if args_cli.freeze_curriculum:
            # Remove threshold_tightening term by clearing it from the curriculum cfg.
            cfg.curriculum.threshold_tightening = None
            print("[train_cfg_overrides] FREEZE CURRICULUM: threshold_tightening disabled (progress stays at 1.0)")
        else:
            cfg.curriculum.threshold_tightening.params["anneal_steps"] = anneal_steps
            _stairs = int(getattr(args_cli, "tighten_stairs", 0) or 0)
            if _stairs > 1:
                cfg.curriculum.threshold_tightening.params["stairs"] = _stairs
                print(
                    f"[train_cfg_overrides] threshold_tightening STAIRCASE: {_stairs} stairs "
                    f"(gates fixed inside a stair, tightened only at its boundary)"
                )
            print(
                f"[train_cfg_overrides] threshold_tightening.anneal_steps = {anneal_steps} "
                f"({int(anneal_scale * 100)}% of {total_timesteps})"
            )
            if anneal_scale > 1.0:
                # With a scale > 1 the training thresholds anneal only to progress 1/scale while evals force 1.0.
                # Allowed (for reproducing old runs) but loud.
                print(
                    f"[train_cfg_overrides] *** WARNING: tighten_steps_scale "
                    f"{anneal_scale} > 1.0 — end-of-training schedule_progress will be "
                    f"only {1.0 / anneal_scale:.2f}; training will NEVER reach the "
                    f"eval/final thresholds. Intended schedule = 0.5 (anneal over the "
                    f"first half of training). ***"
                )
    _verify_object_tracking_schedule(args_cli, cfg, total_timesteps)


def _verify_object_tracking_schedule(args_cli, cfg, total_timesteps):
    """Verify the effective object_tracking schedule loudly (catches silent freeze/mismatch)."""
    verify_objtrack_schedule(cfg, "train_cfg_overrides", frozen=bool(args_cli.freeze_curriculum))

    # RSI experiment: patch the rsi_schedule curriculum from --rsi_* flags.
    if bool(getattr(args_cli, "rsi_enable", 0)) and hasattr(cfg.curriculum, "rsi_schedule"):
        p = cfg.curriculum.rsi_schedule.params
        p["target_fraction"] = float(args_cli.rsi_target_fraction)
        p["ramp_episodes"] = int(args_cli.rsi_ramp_episodes)
        p["lo_frac"] = float(args_cli.rsi_lo_frac)
        p["hi_frac"] = float(args_cli.rsi_hi_frac)
        if args_cli.rsi_fraction_start is not None:
            p["fraction_start"] = float(args_cli.rsi_fraction_start)
        if args_cli.rsi_lo_frac_end is not None:
            p["lo_frac_end"] = float(args_cli.rsi_lo_frac_end)
        if args_cli.rsi_hi_frac_end is not None:
            p["hi_frac_end"] = float(args_cli.rsi_hi_frac_end)
        if args_cli.rsi_band_ramp_episodes is not None:
            p["band_ramp_episodes"] = int(args_cli.rsi_band_ramp_episodes)
        if getattr(args_cli, "rsi_ramp_frac", None) is not None:
            _rs = max(int(args_cli.rsi_ramp_frac * total_timesteps), 1)
            p["ramp_steps"] = _rs
            p["band_ramp_steps"] = _rs
            print(
                f"[train_cfg_overrides] RSI step-based anneal: ramp_steps={_rs} "
                f"({int(args_cli.rsi_ramp_frac * 100)}% of {total_timesteps} timesteps)"
            )
        print(
            f"[train_cfg_overrides] RSI ENABLED: target_fraction={p['target_fraction']} "
            f"ramp_episodes={p['ramp_episodes']} band=[{p['lo_frac']:.3f},{p['hi_frac']:.3f}]"
            + (
                f"->[{p.get('lo_frac_end')},{p.get('hi_frac_end')}]"
                if "lo_frac_end" in p or "hi_frac_end" in p
                else " (constant)"
            )
            + (f" fraction_start={p['fraction_start']}" if "fraction_start" in p else "")
        )
        print(
            "[train_cfg_overrides] RSI: requires all-frame arm-IK precompute; "
            "reset_arm_to_trajectory will warn if absent (falls back to frame-0)."
        )

    # Contact-pilot hinge schedule: recompute ramp_steps precisely now that
    # rollouts (→ total_timesteps in common_step_counter units) is known.
    # apply_contact_pilot_overrides() registered the term with a coarse
    # placeholder; here we set the correct 40%-of-total_timesteps window.
    if (
        getattr(args_cli, "hinge_schedule", 0)
        and hasattr(cfg, "curriculum")
        and cfg.curriculum is not None
        and hasattr(cfg.curriculum, "contact_pilot_hinge_schedule")
        and cfg.curriculum.contact_pilot_hinge_schedule is not None
    ):
        _hinge_ramp_steps = max(int(total_timesteps * 0.40), 1)
        cfg.curriculum.contact_pilot_hinge_schedule.params["ramp_steps"] = _hinge_ramp_steps
        print(
            f"[train_cfg_overrides] contact_pilot_hinge_schedule.ramp_steps = "
            f"{_hinge_ramp_steps} (40% of {total_timesteps} common-steps)"
        )
