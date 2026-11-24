"""Freeze and restore of the training-time state around a frozen eval (curricula, randomization, DR, ramps, clock)."""

import os

import torch

from dexengine.rl.judge.snapshot_metrics import _effective_fingertip_thresholds, _fmt_fingertip_thresholds


def freeze_training_state(*, args_cli, raw_env, traj_mgr):
    """Save the training-time curriculum, randomization, DR and ramp state and pin eval conditions."""
    (
        saved_class_cfgs,
        saved_progress,
        saved_randomize_params,
        saved_rsi,
        saved_snap_only,
        saved_term_cfgs,
        saved_term_names,
    ) = _save_curriculum_state(args_cli, raw_env, traj_mgr)
    for term_cfg_ev in raw_env.event_manager._mode_term_cfgs.get("reset", []):
        if hasattr(term_cfg_ev, "params") and "randomize_fraction" in term_cfg_ev.params:
            saved_randomize_params = {
                "randomize_fraction": term_cfg_ev.params["randomize_fraction"],
                "finger_open_deg": term_cfg_ev.params.get("finger_open_deg", 0.0),
                "wrist_z_max_cm": term_cfg_ev.params.get("wrist_z_max_cm", 0.0),
            }
            term_cfg_ev.params["randomize_fraction"] = 0.0
            term_cfg_ev.params["finger_open_deg"] = 0.0
            term_cfg_ev.params["wrist_z_max_cm"] = 0.0

    return _disable_reset_physics_dr(
        raw_env,
        saved_class_cfgs,
        saved_progress,
        saved_randomize_params,
        saved_rsi,
        saved_snap_only,
        saved_term_cfgs,
        saved_term_names,
    )


def _disable_reset_physics_dr(
    raw_env,
    saved_class_cfgs,
    saved_progress,
    saved_randomize_params,
    saved_rsi,
    saved_snap_only,
    saved_term_cfgs,
    saved_term_names,
):
    """2c. Disable per-reset physics DR during eval (train-only): neutralize the mass/friction/restitution/COM scale
    ranges to identity so the frozen eval is DR-free. Restored in step 6b. (No-op when DR events are absent/inert.)
    2d. Pin the object gravity ramp to 100% during eval (the frozen eval's conditions must be IDENTICAL to the
    standalone eval, which reports "gravity_ramp pinned at 100% (no curriculum)"). Without this the frozen eval
    measures whatever ramp progress the training happens to be at. With the default ramp (10% of training) it is
    already 1.0 past iter ~40, so this is a no-op for normal runs — it matters for short runs / custom
    --gravity_ramp_*.
    """
    saved_grav_ramp = []
    for term_cfg_ev in _grav_ramp_term_cfgs(raw_env.event_manager):
        if hasattr(term_cfg_ev, "params") and "gravity_magnitude" in term_cfg_ev.params:
            saved_grav_ramp.append((term_cfg_ev, term_cfg_ev.params.get("ramp_steps")))
            term_cfg_ev.params["ramp_steps"] = 1
    if saved_grav_ramp:
        print("    [FROZEN EVAL] gravity ramp pinned to 100% (matches standalone eval)")

    saved_dr_ranges = []
    _DR_IDENTITY = {
        "mass_scale_range": (1.0, 1.0),
        "static_friction_scale_range": (1.0, 1.0),
        "dynamic_friction_scale_range": (1.0, 1.0),
        "restitution_scale_range": (1.0, 1.0),
        "com_offset_range_m": (0.0, 0.0),
    }
    for term_cfg_ev in raw_env.event_manager._mode_term_cfgs.get("reset", []):
        if not hasattr(term_cfg_ev, "params"):
            continue
        _saved_dr = {}
        for _pk, _ident in _DR_IDENTITY.items():
            if _pk in term_cfg_ev.params:
                _saved_dr[_pk] = term_cfg_ev.params[_pk]
                term_cfg_ev.params[_pk] = _ident
        if _saved_dr:
            saved_dr_ranges.append((term_cfg_ev, _saved_dr))

    # 3. Pin gravity/friction at full magnitude (no ramp curriculum).
    # gravity_ramp: ramp_steps=0, start_fraction=1.0 -> full gravity from step 0
    # friction_decay: ramp_steps=0, initial=final -> constant friction
    # Save the curriculum clock: the eval's env.step calls advance it (see frozen_eval.frozen_eval_snapshot).
    _saved_clock = int(getattr(raw_env, "common_step_counter", 0))

    saved_ramp_state = {}
    for term_cfg_ev in _grav_ramp_term_cfgs(raw_env.event_manager):
        if not (hasattr(term_cfg_ev, "params") and "ramp_steps" in term_cfg_ev.params):
            continue
        func_name = str(term_cfg_ev.func)
        saved = {"ramp_steps": term_cfg_ev.params["ramp_steps"]}
        term_cfg_ev.params["ramp_steps"] = 0
        if "start_fraction" in term_cfg_ev.params:
            saved["start_fraction"] = term_cfg_ev.params["start_fraction"]
            term_cfg_ev.params["start_fraction"] = 1.0
        if "initial_multiplier" in term_cfg_ev.params and "final_multiplier" in term_cfg_ev.params:
            saved["initial_multiplier"] = term_cfg_ev.params["initial_multiplier"]
            term_cfg_ev.params["initial_multiplier"] = term_cfg_ev.params["final_multiplier"]
        saved_ramp_state[func_name] = saved
    return (
        _saved_clock,
        saved_class_cfgs,
        saved_dr_ranges,
        saved_grav_ramp,
        saved_progress,
        saved_ramp_state,
        saved_randomize_params,
        saved_rsi,
        saved_snap_only,
        saved_term_cfgs,
        saved_term_names,
    )


def _save_curriculum_state(args_cli, raw_env, traj_mgr):
    """1. Save curriculum state."""
    saved_progress = traj_mgr.schedule_progress
    saved_rsi = traj_mgr.rsi_fraction
    saved_term_names = list(raw_env.curriculum_manager._term_names)
    saved_term_cfgs = list(raw_env.curriculum_manager._term_cfgs)
    saved_class_cfgs = list(raw_env.curriculum_manager._class_term_cfgs)

    # 2. Freeze: disable curriculum, set final conditions
    raw_env.curriculum_manager._term_names.clear()
    raw_env.curriculum_manager._term_cfgs.clear()
    raw_env.curriculum_manager._class_term_cfgs.clear()
    # No threshold curriculum: pin to the forced eval threshold progress
    # (--eval_threshold_progress, default 1.0 = final/tight thresholds).
    _eval_prog = float(args_cli.eval_threshold_progress)
    traj_mgr.schedule_progress = _eval_prog
    traj_mgr.rsi_fraction = 0.0
    print(
        f"    [FROZEN EVAL] schedule_progress forced to {_eval_prog:.4f}"
        f"{' (final/tight)' if _eval_prog == 1.0 else ' (--eval_threshold_progress)'}"
    )
    print(f"    [FROZEN EVAL] {_fmt_fingertip_thresholds(_effective_fingertip_thresholds(raw_env, _eval_prog))}")
    # 2a-bis: print the EFFECTIVE OBJECT gate actually in force.
    # Why: the two lines above report schedule_progress and the FINGERTIP thresholds
    # only. When the fingertip term is absent they print "no effective thresholds",
    # which reads as "the gate is off/unknown" and invites the wrong conclusion that
    # the frozen eval runs looser than the standalone eval. The object gate is what
    # decides completion here, so state it in cm/deg explicitly.
    _tm = raw_env.termination_manager
    _live = dict(zip(getattr(_tm, "_term_names", []), getattr(_tm, "_term_cfgs", []), strict=False))
    for _tn in ("object_tracking", "object_tracking_left"):
        _tc = _live.get(_tn) or getattr(getattr(_tm, "cfg", None), _tn, None)
        if _tc is None or not hasattr(_tc, "params"):
            continue
        _p = _tc.params
        _ps, _pe = _p.get("pos_start"), _p.get("pos_end")
        _rs, _re = _p.get("rot_start_deg"), _p.get("rot_end_deg")
        if _ps is None or _pe is None:
            continue
        _pf = _ps + (_pe - _ps) * _eval_prog
        _rf = (_rs + (_re - _rs) * _eval_prog) if (_rs is not None and _re is not None) else float("nan")
        print(
            f"    [FROZEN EVAL] OBJECT gate {_tn}: {100.0 * _pf:.1f}cm / {_rf:.0f}deg "
            f"(progress={_eval_prog:.2f}, start={100.0 * _ps:.1f}cm end={100.0 * _pe:.1f}cm, "
            f"lag_window={_p.get('lag_window', 0)}, grace={_p.get('grace_period', 0)}) "
            f"— compare with the standalone eval's POS_CONST/ROT_CONST"
        )

    # 2a-ter: RESET_SCHEME=decoupled makes the side gates SNAP-ONLY — a tripped gate does not set terminated, it only
    # snaps that arm back onto the reference. Episodes would then end only by trajectory_done (time_out=True), the
    # shared aggregator's `is_early = terminated & ~truncated` would always be False and every env would count as
    # completed. The eval must measure what the standalone eval measures, so snap_only is off during the frozen eval
    # (the gates terminate normally); step 6c turns it back on.
    saved_snap_only = []
    _tm_so = raw_env.termination_manager
    _live_so = dict(zip(getattr(_tm_so, "_term_names", []), getattr(_tm_so, "_term_cfgs", []), strict=False))
    for _sn in ("object_tracking", "object_tracking_left", "contact_hand_right", "contact_hand_left"):
        _tc_so = _live_so.get(_sn)
        if _tc_so is not None and hasattr(_tc_so, "params") and _tc_so.params.get("snap_only", False):
            saved_snap_only.append(_tc_so)
            _tc_so.params["snap_only"] = False
    if saved_snap_only:
        print(
            f"    [FROZEN EVAL] SNAP-ONLY off "
            f"{[t.params.get('side', '?') for t in saved_snap_only]} "
            f"({len(saved_snap_only)} gates) — this eval uses the termination rule of the standalone eval",
            flush=True,
        )

    # 2b. Disable randomization during eval
    saved_randomize_params = {}
    return (
        saved_class_cfgs,
        saved_progress,
        saved_randomize_params,
        saved_rsi,
        saved_snap_only,
        saved_term_cfgs,
        saved_term_names,
    )


def restore_training_state(
    *,
    _mem_restart_check,
    _saved_clock,
    args_cli,
    iteration,
    raw_env,
    saved_class_cfgs,
    saved_dr_ranges,
    saved_grav_ramp,
    saved_progress,
    saved_ramp_state,
    saved_randomize_params,
    saved_rsi,
    saved_snap_only,
    saved_term_cfgs,
    saved_term_names,
    traj_mgr,
    wrapped,
):
    """Undo ``freeze_training_state``, reset the env for training and run the planned-restart check."""
    _rc_pend, _rc_rank = _restore_curriculum_and_ramps(
        _mem_restart_check,
        _saved_clock,
        args_cli,
        iteration,
        raw_env,
        saved_class_cfgs,
        saved_dr_ranges,
        saved_grav_ramp,
        saved_progress,
        saved_ramp_state,
        saved_randomize_params,
        saved_rsi,
        saved_snap_only,
        saved_term_cfgs,
        saved_term_names,
        traj_mgr,
        wrapped,
    )
    if getattr(raw_env, "_pss_on", False):
        _rc_pend = (
            f" pend_r={int(raw_env._pss_pending['right'].sum())} pend_l={int(raw_env._pss_pending['left'].sum())}"
        )
    print(
        f"    [FROZEN EVAL] RESTORE CHECK rank={_rc_rank}: "
        f"rsi={traj_mgr.rsi_fraction:.3f} prog={traj_mgr.schedule_progress:.3f} "
        f"curr_terms={len(raw_env.curriculum_manager._term_names)}{_rc_pend}",
        flush=True,
    )


def _restore_curriculum_and_ramps(
    _mem_restart_check,
    _saved_clock,
    args_cli,
    iteration,
    raw_env,
    saved_class_cfgs,
    saved_dr_ranges,
    saved_grav_ramp,
    saved_progress,
    saved_ramp_state,
    saved_randomize_params,
    saved_rsi,
    saved_snap_only,
    saved_term_cfgs,
    saved_term_names,
    traj_mgr,
    wrapped,
):
    """6. Restore curriculum, gravity/friction ramp, and reset for training 6a2. Restore the curriculum clock (drop the
    steps the eval rollout added).
    """
    _drift = int(getattr(raw_env, "common_step_counter", 0)) - _saved_clock
    raw_env.common_step_counter = _saved_clock
    if iteration <= (args_cli.eval_interval * 2):  # printed for the first two evals only
        print(
            f"    [EVAL_CLOCK_FIX] the eval advanced the clock by {_drift} steps — restored to {_saved_clock}",
            flush=True,
        )

    # 6b. Restore per-reset physics DR ranges (neutralized in step 2c)
    for _gr_term, _gr_steps in saved_grav_ramp:
        if _gr_steps is not None:
            _gr_term.params["ramp_steps"] = _gr_steps
    for _dr_term, _dr_saved in saved_dr_ranges:
        _dr_term.params.update(_dr_saved)
    # 6c. snap_only back on — training runs decoupled again.
    for _tc_so in saved_snap_only:
        _tc_so.params["snap_only"] = True

    raw_env.curriculum_manager._term_names.extend(saved_term_names)
    raw_env.curriculum_manager._term_cfgs.extend(saved_term_cfgs)
    raw_env.curriculum_manager._class_term_cfgs.extend(saved_class_cfgs)
    traj_mgr.schedule_progress = saved_progress
    traj_mgr.rsi_fraction = saved_rsi

    for term_cfg_ev in _grav_ramp_term_cfgs(raw_env.event_manager):
        if not (hasattr(term_cfg_ev, "params") and "ramp_steps" in term_cfg_ev.params):
            continue
        func_name = str(term_cfg_ev.func)
        if func_name in saved_ramp_state:
            for k, v in saved_ramp_state[func_name].items():
                term_cfg_ev.params[k] = v

    return _restore_randomization_params(_mem_restart_check, iteration, raw_env, saved_randomize_params, wrapped)


def _restore_randomization_params(_mem_restart_check, iteration, raw_env, saved_randomize_params, wrapped):
    """Restore randomization params."""
    if saved_randomize_params:
        for term_cfg_ev in raw_env.event_manager._mode_term_cfgs.get("reset", []):
            if hasattr(term_cfg_ev, "params") and "randomize_fraction" in term_cfg_ev.params:
                term_cfg_ev.params.update(saved_randomize_params)

    # Exit reset (DEXENGINE_EVAL_EXIT_RESET, off by default). Off: skrl's `_reset_once` was already consumed by the
    # last eval batch, so the reset() below is a no-op and the end-of-eval state continues as the training rollout.
    # On: a real full reset (RSI).
    from dexengine.rl.judge.launch import exit_reset, exit_reset_on

    if exit_reset_on():
        exit_reset(wrapped, raw_env, replay=os.environ.get("DEXENGINE_FROZEN_REPLAY", "0") == "1")
    else:
        wrapped.reset()

    # Release fragmented GPU memory pages back to the allocator.
    # Without this, repeated frozen_eval_snapshot calls cause PhysX
    # CUDA allocator fragmentation and eventual OOM (CUDA error 2).
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Planned restart on low GPU headroom (DEXENGINE_MEM_RESTART_FREE_GB=<GB>, 0 = off): PhysX solver buffers only
    # grow, so before the card cannot hold the next doubling the run stops at this iteration's checkpoint and the
    # launcher restarts it with a fresh heap (the curriculum clock is kept through --resume_iteration). Headroom =
    # free after empty_cache − (training peak reserved − reserved now). On trigger it prints [MEM_RESTART] and sets
    # _stop; patched_post unwinds with _Exp1EarlyStop.
    _mem_restart_check(iteration)

    # RESTORE CHECK: verify the
    # training-state restore actually landed on THIS rank + surface any stale
    # per-side snap queue crossing the eval boundary.
    _rc_rank = os.environ.get("LOCAL_RANK", "0")
    _rc_pend = ""
    return _rc_pend, _rc_rank


def _grav_ramp_term_cfgs(event_manager):
    """Event-term cfgs of the modes that hold ramp terms (``dexengine.rl.tasks.mdp.gravity_wrench.ramp_term_cfgs``).

    Imported lazily so loading this module does not import the tasks package (env registration)."""
    from dexengine.rl.tasks.mdp.gravity_wrench import ramp_term_cfgs

    return ramp_term_cfgs(event_manager)
