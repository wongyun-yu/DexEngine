"""Contact-pilot overrides of the env cfg: hinge thresholds, level removal, contact-match rewards and terminations."""

from __future__ import annotations

import os

from dexengine.rl.env_setup.contact_match import wire_contact_match
from dexengine.rl.env_setup.objects import abspath_multi
from dexengine.rl.env_setup.objtrack import CONTACT_PILOT_TIP_TERMS


def _wire_contact_termination_and_hinge(*, _hinge_sched_on, _tip_target_m, args_cli, cfg):
    """Contact-mismatch terminations and the hinge threshold schedule."""
    if bool(getattr(args_cli, "enable_contact_termination", 0)):
        _mesh_dir = getattr(args_cli, "object_mesh_dir", None)
        if _mesh_dir is None:
            print("[contact_pilot] WARNING: --object_mesh_dir not set; contact termination skipped.")
        else:
            from isaaclab.managers import TerminationTermCfg as _DoneTerm

            from dexengine.rl.tasks.mdp.rewards.contact_match import HandContactMismatchSustained as _HCMS

            _src = str(getattr(args_cli, "contact_source", "mano"))
            _seq_dir = os.path.abspath(getattr(cfg.trajectory, "data_dir", args_cli.data_dir))
            _fmin = float(getattr(args_cli, "contact_term_force_threshold", 0.05))
            _grace = int(getattr(args_cli, "contact_term_grace_steps", 50))
            cfg.terminations.contact_hand_right = _DoneTerm(
                func=_HCMS,
                time_out=False,
                params={
                    "side": "right",
                    "contact_source": _src,
                    "seq_dir": _seq_dir,
                    "mesh_dir": abspath_multi(_mesh_dir),
                    "force_threshold": _fmin,
                    "grace_steps": _grace,
                    "object_key": "object_right",
                },
            )
            print(f"[contact_pilot] contact termination added: f_min={_fmin}N grace={_grace} source={_src}")
            # Bimanual Option 2: LEFT contact-mismatch termination mirror.
            if bool(getattr(args_cli, "bimanual", 0)):
                _mesh_dir_l = getattr(args_cli, "object_mesh_dir_left", None)
                if _mesh_dir_l is None:
                    print(
                        "[contact_pilot] (bimanual) WARNING: --object_mesh_dir_left not set; "
                        "LEFT contact termination skipped."
                    )
                else:
                    cfg.terminations.contact_hand_left = _DoneTerm(
                        func=_HCMS,
                        time_out=False,
                        params={
                            "side": "left",
                            "contact_source": _src,
                            "seq_dir": _seq_dir,
                            "mesh_dir": abspath_multi(_mesh_dir_l),
                            "force_threshold": _fmin,
                            "grace_steps": _grace,
                            "object_key": "object_left",
                        },
                    )
                    print("[contact_pilot] (left_present) LEFT contact termination added")

    _wire_hinge_schedule(_hinge_sched_on, _tip_target_m, args_cli, cfg)


def _wire_hinge_schedule(_hinge_sched_on, _tip_target_m, args_cli, cfg):
    """Hinge schedule curriculum (linear ramp, --hinge_schedule)."""
    # (5) Hinge schedule curriculum (linear ramp, --hinge_schedule).
    if _hinge_sched_on:
        if not hasattr(cfg, "curriculum") or cfg.curriculum is None:
            print("[contact_pilot] WARNING: cfg.curriculum missing; --hinge_schedule skipped.")
        else:
            from isaaclab.managers import CurriculumTermCfg as _CurrTerm

            from dexengine.rl.tasks.mdp.rewards.contact_closure import HingeThresholdSchedule as _HingeSched

            # The term ramps over the first ~40% of training, measured in
            # env.common_step_counter units (= total_timesteps = max_iterations *
            # rollouts). `rollouts` is NOT known here, so register with a coarse
            # placeholder; the train fn's train_cfg_overrides() patches
            # ramp_steps to the precise 40%-of-total_timesteps window (it has
            # `total_timesteps` in scope). See that block for the exact value.
            _max_iter = int(getattr(args_cli, "max_iterations", 5000))
            _ramp_steps = max(int(_max_iter * 0.40), 1)  # placeholder, patched in train fn
            cfg.curriculum.contact_pilot_hinge_schedule = _CurrTerm(
                func=_HingeSched,
                params={
                    "tip_terms": list(CONTACT_PILOT_TIP_TERMS),
                    "target_m": _tip_target_m,
                    "ramp_steps": _ramp_steps,
                },
            )
            print(
                f"[contact_pilot] hinge_schedule curriculum added: "
                f"ramp 0 → {_tip_target_m:.4f} m over {_ramp_steps} steps "
                f"(40% of max_iterations={_max_iter})"
            )


def apply_contact_pilot_overrides(cfg, args_cli, ctx: str) -> None:
    """Contact-pilot reward wiring (per-finger contact-MATCH reward,
    hinge thresholds, level removal). ALL gated by default-OFF CLI flags, so
    with every new flag at default this function is a no-op and behavior is
    IDENTICAL to before.

    Prints one line per action so a smoke test can confirm via stdout.
    """
    # Master gate: do nothing unless a contact-pilot flag is actually set.
    _hinge_on = bool(getattr(args_cli, "enable_hinge", 0))
    _hinge_sched_on = bool(getattr(args_cli, "hinge_schedule", 0))
    _match_on = bool(getattr(args_cli, "enable_contact_match_reward", 0))
    _term_on = bool(getattr(args_cli, "enable_contact_termination", 0))
    # The force-direction reward is wired only inside the contact-match block: asking for it with matching off would
    # drop it silently, so an input without a consumer fails.
    if bool(getattr(args_cli, "enable_force_dir_reward", 0)) and not _match_on:
        raise RuntimeError(
            "[contact_pilot] --enable_force_dir_reward 1 with --enable_contact_match_reward 0 — the "
            "force-direction reward is wired only inside the contact-match block. Turn both on, or turn "
            "it off with FORCE_DIR=0 (recipe table scripts/_rl_recipe.sh)"
        )
    if not (_hinge_on or _hinge_sched_on or _match_on or _term_on):
        return

    print("[contact_pilot] === contact-pilot overrides active ===")

    _tip_target_m = float(getattr(args_cli, "hinge_fingertip_cm", 1.5)) / 100.0
    _wrist_target_m = float(getattr(args_cli, "hinge_wrist_cm", 2.0)) / 100.0

    _apply_contact_pilot_terms(cfg, args_cli, _hinge_on, _hinge_sched_on, _match_on, _tip_target_m, _wrist_target_m)


def _apply_contact_pilot_terms(cfg, args_cli, _hinge_on, _hinge_sched_on, _match_on, _tip_target_m, _wrist_target_m):
    """Tracking hinges, level-1/2 ramps, legacy grasp-reward removal, contact-match reward, contact termination."""
    # (1) TRACKING hinge thresholds on fingertip + wrist rewards (static, --enable_hinge).
    #     "Error within the threshold scores full reward; exp decay applies only beyond."
    #
    #     ⚠ Do not confuse with --wrist_l2_hinge (rl.train.runner), which is a free zone on
    #     the residual ACTION magnitude, not on tracking error. That one is on in the
    #     recipe; this one is not: the recipe does not set --enable_hinge, so the
    #     wrist tracking hinge stays 0.0 in a recipe run.
    #
    #     ⚠ POSITION ONLY. RewardEefRot accepts `hinge_threshold_rad` but nothing below sets
    #     it, so even --enable_hinge 1 leaves rotation un-hinged. Dead parameter — wire it
    #     here if a rotation hinge is ever wanted.
    if _hinge_on:
        for tip in CONTACT_PILOT_TIP_TERMS:
            if hasattr(cfg.rewards, tip):
                getattr(cfg.rewards, tip).params["hinge_threshold_m"] = _tip_target_m
                print(f"[contact_pilot] hinge: {tip}.hinge_threshold_m = {_tip_target_m:.4f} m")
        if hasattr(cfg.rewards, "eef_pos"):
            cfg.rewards.eef_pos.params["hinge_threshold_m"] = _wrist_target_m
            print(f"[contact_pilot] hinge: eef_pos.hinge_threshold_m = {_wrist_target_m:.4f} m")

    # (2) Level removal. level_2 ALWAYS removed (unless --ramp_level12 1 re-enables it);
    # level_1 removed unless --keep_level1.
    # REMOVE the term (set to None) instead of zeroing its weight, so it is NOT
    # registered with the RewardManager and does NOT clutter the W&B/TB logs.
    _ramp_level12 = bool(getattr(args_cli, "ramp_level12", 0))
    _bim = bool(getattr(args_cli, "bimanual", 0))
    if _ramp_level12:
        # C5: level_2 is KEPT so Level12RampDown can ramp it down; do not remove.
        print("[contact_pilot] C5 --ramp_level12 1: level_2 KEPT (will be ramped to 0 by Level12RampDown curriculum)")
    else:
        if getattr(cfg.rewards, "level_2", None) is not None:
            cfg.rewards.level_2 = None
            print("[contact_pilot] removed level_2 reward (not logged)")
        # Bimanual: mirror the level removal to the LEFT, else level_2_left (and level_1_left)
        # stay registered while the right loses them -> left gets extra body-tracking reward
        # under the summed objective (silent asymmetry).
        if _bim and getattr(cfg.rewards, "level_2_left", None) is not None:
            cfg.rewards.level_2_left = None
            print("[contact_pilot] (bimanual) removed level_2_left reward (symmetric)")
    if getattr(args_cli, "keep_level1", 0):
        print("[contact_pilot] level_1 reward KEPT (--keep_level1 1)")
    elif _ramp_level12:
        # C5: level_1 is KEPT so Level12RampDown can ramp it down; do not remove.
        print("[contact_pilot] C5 --ramp_level12 1: level_1 KEPT (will be ramped to 0 by Level12RampDown curriculum)")
    else:
        if getattr(cfg.rewards, "level_1", None) is not None:
            cfg.rewards.level_1 = None
            print("[contact_pilot] removed level_1 reward (not logged, --keep_level1 0)")
        if _bim and getattr(cfg.rewards, "level_1_left", None) is not None:
            cfg.rewards.level_1_left = None
            print("[contact_pilot] (bimanual) removed level_1_left reward (symmetric)")

    # (3) Contact replaces the legacy grasp rewards — REMOVE them (not logged).
    for legacy in ("fingertip_force", "grasp_stability"):
        if getattr(cfg.rewards, legacy, None) is not None:
            setattr(cfg.rewards, legacy, None)
            print(f"[contact_pilot] removed legacy reward: {legacy} (not logged)")

    # (4) Per-finger contact-MATCH reward + weight-0 diagnostic terms.
    wire_contact_match(_match_on=_match_on, args_cli=args_cli, cfg=cfg)

    # (4b) Hand-level contact-mismatch TERMINATION (NEW per-finger annotation +
    # object-filtered contact force). Gated by --enable_contact_termination.
    _wire_contact_termination_and_hinge(
        _hinge_sched_on=_hinge_sched_on, _tip_target_m=_tip_target_m, args_cli=args_cli, cfg=cfg
    )
