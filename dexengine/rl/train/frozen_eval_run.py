"""Frozen eval batch: eval size, physics fingerprint line, the balanced eval run, adaptive-term probe, per-side
rates."""

import os

import torch

from dexengine.rl.judge.arm_penetration import collector_for as _arm_pen_for
from dexengine.rl.judge.rollouts import run_eval_balanced
from dexengine.rl.tasks import switches as _switches


def prepare_eval_batch(*, agent, args_cli, cfg, evals_per_clip, iteration, raw_env, traj_mgr, wrapped):
    """Policy function, sample size, W&B logger, warmup, the physics fingerprint line and the
    per-eval capture switches."""
    # 4. Build policy function and run balanced eval
    policy_model = agent.policy
    preprocessor = agent._state_preprocessor

    def policy_fn(obs_tensor):
        processed = preprocessor(obs_tensor)
        mu, _, _ = policy_model.compute({"states": processed}, role="policy")
        return mu

    num_clips = traj_mgr.num_clips
    clip_names = traj_mgr._loader.sequence_names

    return _eval_batch_counts(
        args_cli, cfg, evals_per_clip, iteration, raw_env, wrapped, clip_names, num_clips, policy_fn
    )


def _eval_batch_counts(args_cli, cfg, evals_per_clip, iteration, raw_env, wrapped, clip_names, num_clips, policy_fn):
    """The frozen eval counts ALL training envs: evals_per_clip is raised so that real_total >= num_envs => one full
    warm batch of all envs is counted (a few episodes per clip would be a small, optimistic sample). Cheap: a batch
    steps until its counted envs finish; at high num_envs they run in parallel anyway. Honors a larger explicit
    --eval_evals_per_clip if set.
    """
    _fe_n_envs = wrapped.num_envs
    evals_per_clip = max(evals_per_clip, -(-_fe_n_envs // num_clips))  # ceil(n_envs/num_clips)

    wandb_log = None
    if args_cli.wandb:
        import wandb

        def wandb_log(d, s):
            return wandb.log(d, step=s)

    # Warmup batches: with DEXENGINE_FROZEN_STRICT=1 (the judge default, set by dexengine.rl.judge.launch) the frozen
    # eval uses args_cli.eval_warmup_batches like the standalone eval (argparse default 1); otherwise 0 (no warmup).
    _frozen_strict = os.environ.get("DEXENGINE_FROZEN_STRICT", "0") == "1"
    _warmup_batches = int(getattr(args_cli, "eval_warmup_batches", 1)) if _frozen_strict else 0

    # Physics fingerprint line, read next to the env's [EVAL_PHYS] / [COLL_PRESET] lines (load_managers). Without the
    # collision curriculum no cross-pair filter is authored, so cross_pairs is ON (same as the standalone eval); with
    # the collision curriculum the pairs stay filtered (OFF) during the eval. contact_term / artic_theta are measured
    # from the termination manager's registered terms, not from arguments. The judge sidecar parses this line.
    _phys_coll_curr_on = bool(getattr(cfg, "collision_curriculum", False))
    if not _phys_coll_curr_on:
        _phys_cross_pairs = "ON"
    else:
        _phys_cross_pairs = "OFF"
    _phys_preset = _switches.text("DEXENGINE_COLL_PRESET") or "legacy"
    _phys_ct_names = [
        n for n in getattr(raw_env.termination_manager, "_term_names", []) or [] if str(n).startswith("contact_")
    ]
    _phys_contact_term = 1 if _phys_ct_names else 0
    from dexengine.rl.judge.terms import ARTIC_THETA_TERM as _att  # measured: theta terms of the termination manager

    _phys_artic = sum(
        1 for n in getattr(raw_env.termination_manager, "_term_names", []) or [] if str(n).startswith(_att)
    )

    print(
        f"\n    [FROZEN EVAL iter {iteration}] "
        f"{evals_per_clip} evals/clip x {num_clips} clips "
        f"physics={_phys_preset} eps_per_clip={evals_per_clip} "
        f"contact_term={_phys_contact_term} cross_pairs={_phys_cross_pairs} artic_theta={_phys_artic} "
        f"warmup_batches={_warmup_batches}"
    )
    # Adaptive-RSI distribution report (prints nothing when adaptive RSI is off): the only place that shows whether
    # the start-frame distribution is actually biased.
    try:
        _rsi_rep = raw_env.trajectory_manager.rsi_adapt_report()
        if _rsi_rep != "rsi_adaptive=off":
            print(f"    [FROZEN EVAL] {_rsi_rep}", flush=True)
    except AttributeError as _exc:
        print(f"    [FROZEN EVAL] rsi_adapt_report unavailable: {_exc!r}")

    # adaptive-term: enable per-episode FIRST-episode peak-error (+tip-drag) capture in
    # ObjectTrackingExceeded BEFORE the eval. State lives on raw_env; resetting
    # _objtrack_cap_prev_el to None re-inits a clean capture on the term's next call.
    _adaptive_on = int(getattr(args_cli, "obj_term_adaptive", 0)) == 1
    if _adaptive_on:
        raw_env._objtrack_eval_capture = True
        raw_env._objtrack_cap_prev_el = None

    # decoupled: discard the training-window side counters so the per-side
    # segment metrics below cover ONLY this frozen-eval rollout.
    if getattr(raw_env, "_pss_mode", "") == "decoupled":
        raw_env.pss_side_stats_snapshot(reset=True)
    return _adaptive_on, _warmup_batches, clip_names, evals_per_clip, num_clips, policy_fn, wandb_log


def run_eval_batches(
    *,
    _adaptive_on,
    _warmup_batches,
    args_cli,
    clip_names,
    evals_per_clip,
    iteration,
    num_clips,
    policy_fn,
    raw_env,
    rollouts,
    traj_mgr,
    traj_recorder,
    wandb_log,
    wrapped,
):
    """Run the balanced eval, then collect the adaptive-termination probe and the per-side segment rates."""
    if traj_recorder is not None:  # the trajectory recorder also skips the judge's warmup batches
        traj_recorder.warmup_batches = _warmup_batches
    aggregate, _ = run_eval_balanced(
        wrapped_env=wrapped,
        policy_fn=policy_fn,
        traj_mgr=traj_mgr,
        num_clips=num_clips,
        evals_per_clip=evals_per_clip,
        clip_names=clip_names,
        wandb_log_fn=wandb_log,
        wandb_step=iteration * rollouts,
        warmup_batches=_warmup_batches,
        log_prefix="train_15",
        arm_pen=_arm_pen_for(raw_env, clip_names, num_clips),  # arm-penetration side value, not the judge value
        traj_recorder=traj_recorder,  # trajectory recording (read-only, default None)
    )

    _stash_adaptive_term_probe(
        _adaptive_on, args_cli, evals_per_clip, iteration, raw_env, rollouts, traj_mgr, aggregate
    )

    return _side_segment_metrics(iteration, raw_env, rollouts, wandb_log, aggregate)


def _side_segment_metrics(iteration, raw_env, rollouts, wandb_log, aggregate):
    """PER-SIDE segment metrics (reset_scheme=decoupled): side segment completion = comp/(comp+fail) over THIS eval
    window (env counters were snapshot-reset just before the rollout). Logged to W&B and printed; under decoupled
    these are the PRIMARY metrics (episodes end on time_out only, so the classic completion_rate saturates).
    """
    _side_rates = None
    if getattr(raw_env, "_pss_mode", "") == "decoupled":
        _st = raw_env.pss_side_stats_snapshot(reset=True)
        _side_rates = []
        _side_log = {}
        for _si, _sd in enumerate(("right", "left")):
            _tot = _st["fail"][_si] + _st["comp"][_si]
            _rate = (_st["comp"][_si] / _tot) if _tot > 0 else float("nan")
            _slen = (_st["seglen"][_si] / _tot) if _tot > 0 else float("nan")
            _side_rates.append(_rate)
            # _tot == 0 is the normal case (the frozen eval turns snap-only off, so no segments form): no NaN goes to
            # W&B; seg_count=0 says "no data". The NaN stays in _side_rates for the best_agent isnan guard.
            if _tot > 0:
                _side_log[f"train_15/side_{_sd}_seg_completion"] = _rate
                _side_log[f"train_15/side_{_sd}_seg_len_mean"] = _slen
            _side_log[f"train_15/side_{_sd}_seg_count"] = _tot
            if _tot:
                print(
                    f"    [FROZEN EVAL] SIDE {_sd}: seg_completion={100 * _rate:.1f}% "
                    f"(comp={_st['comp'][_si]} fail={_st['fail'][_si]} "
                    f"mean_len={_slen:.0f})",
                    flush=True,
                )
            else:
                # Segments come from the snap-only path (a tripped gate snaps the arm back and ends a segment). The
                # frozen eval turns snap-only off, so no segments form and the Completion rate above is the measure.
                # (During training decoupled stays on and the SIDESNAP counters keep printing.)
                print(
                    f"    [FROZEN EVAL] SIDE {_sd}: no segments — expected"
                    f" (snap-only is off during eval → the gate terminates normally. "
                    f"The judge uses the Completion rate above)",
                    flush=True,
                )
        if wandb_log is not None:
            wandb_log(_side_log, iteration * rollouts)
    return _side_rates, aggregate


def _stash_adaptive_term_probe(
    _adaptive_on, args_cli, evals_per_clip, iteration, raw_env, rollouts, traj_mgr, aggregate
):
    """adaptive-term: q80 first-episode peak errors + drag_q50 over this probe -> stash the 5-tuple for ObjTermAdaptive.
    Missing capture => q80 stays nan => vetoes HOLD (loud warn).
    """
    if _adaptive_on:
        raw_env._objtrack_eval_capture = False
        _q80_pos_cm = float("nan")
        _q80_rot_deg = float("nan")
        _drag_q50 = float("nan")
        _cap_pos = getattr(raw_env, "_objtrack_cap_peak_pos", None)
        if _cap_pos is None:
            print(
                "    [ADAPTIVE TERM] WARNING: no peak-error capture this eval (term never ran "
                "with capture on?) — q80=nan, vetoes HOLD. BUG if this persists."
            )
        else:
            _total_eps = int(evals_per_clip * traj_mgr.num_clips)
            _N = _cap_pos.shape[0]
            _n_use = max(1, min(_total_eps, _N))
            _fd = raw_env._objtrack_cap_first_done
            _peaks_pos = torch.where(_fd[:_n_use], raw_env._objtrack_first_peaks_pos[:_n_use], _cap_pos[:_n_use])
            _peaks_rot = torch.where(
                _fd[:_n_use], raw_env._objtrack_first_peaks_rot[:_n_use], raw_env._objtrack_cap_peak_rot[:_n_use]
            )
            _q80_pos_cm = float(torch.quantile(_peaks_pos * 100.0, 0.80).item())
            _q80_rot_deg = float(torch.quantile(_peaks_rot, 0.80).item())
            # v3 drag: finished first-episodes have _objtrack_first_drag; unfinished fall back to
            # the live counters (air may be 0 early -> nan-guarded via clamp).
            _fdrag = raw_env._objtrack_first_drag[:_n_use]
            _live = raw_env._objtrack_cap_drag[:_n_use] / raw_env._objtrack_cap_air[:_n_use].clamp(min=1.0)
            _dragv = torch.where(_fd[:_n_use], _fdrag, _live)
            _dragv = _dragv[~torch.isnan(_dragv)]
            if _dragv.numel() > 0:
                _drag_q50 = float(_dragv.median().item())
            print(
                f"    [ADAPTIVE TERM] probe first-episodes over {_n_use} envs "
                f"({int(_fd[:_n_use].sum().item())} finished): q80 pos={_q80_pos_cm:.2f}cm "
                f"rot={_q80_rot_deg:.1f}deg drag_q50={_drag_q50:.2f}"
            )
        if not hasattr(raw_env, "_adaptive_eval_log"):
            raw_env._adaptive_eval_log = []
        raw_env._adaptive_eval_log.append(
            (
                float(aggregate.get("completion_rate", 0.0) or 0.0),
                float(aggregate.get("mean_episode_length", 0.0) or 0.0),
                _q80_pos_cm,
                _q80_rot_deg,
                _drag_q50,
            )
        )
        # W&B: log the curriculum's threshold/reason dict (stashed every call; 1-eval lag ok)
        if args_cli.wandb:
            _wdict = getattr(raw_env, "_adaptive_term_wandb", None)
            if _wdict:
                try:
                    import wandb as _wandb_ta

                    # Skip probe keys that are NaN (empty probe window): no NaN goes to W&B.
                    _probe_kv = {
                        k: v
                        for k, v in {
                            "adaptive_term/probe_drag_q50": _drag_q50,
                            "adaptive_term/probe_q80_pos_cm": _q80_pos_cm,
                            "adaptive_term/probe_q80_rot_deg": _q80_rot_deg,
                        }.items()
                        if v == v
                    }
                    _wandb_ta.log(dict(_wdict, **_probe_kv), step=iteration * rollouts)
                except Exception as _exc:  # noqa: BLE001 — wandb raise types are undocumented
                    print(f"    [FROZEN EVAL] wandb probe log failed: {_exc!r}")
