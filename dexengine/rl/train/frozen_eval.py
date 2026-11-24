"""Frozen eval: the deterministic eval run inside the training process (periodic eval and fresh-process judge).

``frozen_eval_snapshot`` freezes every training-time schedule (frozen_eval_state), runs the balanced eval
(frozen_eval_run), updates the best checkpoints (frozen_eval_select) and restores the training state. The fresh-process
judge (``DEXENGINE_FROZEN_REPLAY=1``) and the render path call the same function, so judge and render see the eval
conditions of training.
"""

import os

from dexengine.core.paths import run_out as _run_out  # a relative output folder is relative to the run tree
from dexengine.rl.train import frozen_eval_run, frozen_eval_select, frozen_eval_state

#: Returned by a split-out step that did not return early (its caller then goes on).
_CONTINUE = object()


def frozen_eval_snapshot(
    iteration,
    evals_per_clip=4,
    traj_recorder=None,
    *,
    _dist,
    _dist_rank,
    _mem_restart_check,
    _voc_flush_on,
    agent,
    args_cli,
    best_frozen_completion,
    best_frozen_return,
    cfg,
    create_env,
    exp_dir,
    frozen_eval_snapshot,
    raw_env,
    rollouts,
    traj_mgr,
    wrapped,
):
    """Run a frozen eval mid-training using shared balanced evaluation.

    Uses the same run_eval_balanced() as standalone eval for identical
    semantics and reporting.

    The curriculum clock (``common_step_counter``) is saved and restored as well: the eval rollout calls env.step(),
    which advances that counter, and every step-based curriculum (gate tightening, gravity/friction ramps, RSI band,
    obj-follow) reads it. Without the restore, evaluating more often would tighten the curricula sooner, so the
    eval interval would change the training schedule.

    Steps:
    1. Save curriculum state (schedule_progress, rsi_fraction, curriculum terms)
    2. Freeze: disable curriculum, set eval conditions
    3. Override gravity/friction ramps to ramp_steps=0
    4. Run balanced evaluation via shared module
    5. Update best_agent.pt if frozen-eval completion improves
       (overrides SKRL's reward-based best-modules heuristic)
    6. Restore everything and resume training
    """
    return _freeze_and_run(
        iteration,
        evals_per_clip,
        traj_recorder,
        _dist,
        _dist_rank,
        _mem_restart_check,
        _voc_flush_on,
        agent,
        args_cli,
        best_frozen_completion,
        best_frozen_return,
        cfg,
        create_env,
        exp_dir,
        frozen_eval_snapshot,
        raw_env,
        rollouts,
        traj_mgr,
        wrapped,
    )


def _freeze_and_run(
    iteration,
    evals_per_clip,
    traj_recorder,
    _dist,
    _dist_rank,
    _mem_restart_check,
    _voc_flush_on,
    agent,
    args_cli,
    best_frozen_completion,
    best_frozen_return,
    cfg,
    create_env,
    exp_dir,
    frozen_eval_snapshot,
    raw_env,
    rollouts,
    traj_mgr,
    wrapped,
):
    """Freeze the training state and run the frozen eval."""
    (
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
    ) = frozen_eval_state.freeze_training_state(args_cli=args_cli, raw_env=raw_env, traj_mgr=traj_mgr)
    return _run_frozen_eval(
        iteration,
        evals_per_clip,
        traj_recorder,
        _dist,
        _dist_rank,
        _mem_restart_check,
        _voc_flush_on,
        agent,
        args_cli,
        best_frozen_completion,
        best_frozen_return,
        cfg,
        create_env,
        exp_dir,
        frozen_eval_snapshot,
        raw_env,
        rollouts,
        traj_mgr,
        wrapped,
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


def _run_frozen_eval(
    iteration,
    evals_per_clip,
    traj_recorder,
    _dist,
    _dist_rank,
    _mem_restart_check,
    _voc_flush_on,
    agent,
    args_cli,
    best_frozen_completion,
    best_frozen_return,
    cfg,
    create_env,
    exp_dir,
    frozen_eval_snapshot,
    raw_env,
    rollouts,
    traj_mgr,
    wrapped,
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
):
    """Run the frozen eval (or its render) on the frozen state, then restore training."""
    return _eval_or_render(
        iteration,
        evals_per_clip,
        traj_recorder,
        _dist,
        _dist_rank,
        _mem_restart_check,
        _voc_flush_on,
        agent,
        args_cli,
        best_frozen_completion,
        best_frozen_return,
        cfg,
        create_env,
        exp_dir,
        frozen_eval_snapshot,
        raw_env,
        rollouts,
        traj_mgr,
        wrapped,
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


def _eval_or_render(
    iteration,
    evals_per_clip,
    traj_recorder,
    _dist,
    _dist_rank,
    _mem_restart_check,
    _voc_flush_on,
    agent,
    args_cli,
    best_frozen_completion,
    best_frozen_return,
    cfg,
    create_env,
    exp_dir,
    frozen_eval_snapshot,
    raw_env,
    rollouts,
    traj_mgr,
    wrapped,
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
):
    """Render mode, or the frozen eval batches."""
    _ret_ = _prepare_and_render(
        iteration,
        evals_per_clip,
        traj_recorder,
        _dist,
        _dist_rank,
        _voc_flush_on,
        agent,
        args_cli,
        best_frozen_completion,
        best_frozen_return,
        cfg,
        create_env,
        exp_dir,
        frozen_eval_snapshot,
        raw_env,
        rollouts,
        traj_mgr,
        wrapped,
        saved_snap_only,
    )
    if _ret_ is not _CONTINUE:
        return _ret_
    frozen_eval_state.restore_training_state(
        _mem_restart_check=_mem_restart_check,
        _saved_clock=_saved_clock,
        args_cli=args_cli,
        iteration=iteration,
        raw_env=raw_env,
        saved_class_cfgs=saved_class_cfgs,
        saved_dr_ranges=saved_dr_ranges,
        saved_grav_ramp=saved_grav_ramp,
        saved_progress=saved_progress,
        saved_ramp_state=saved_ramp_state,
        saved_randomize_params=saved_randomize_params,
        saved_rsi=saved_rsi,
        saved_snap_only=saved_snap_only,
        saved_term_cfgs=saved_term_cfgs,
        saved_term_names=saved_term_names,
        traj_mgr=traj_mgr,
        wrapped=wrapped,
    )
    return None


def _prepare_and_render(
    iteration,
    evals_per_clip,
    traj_recorder,
    _dist,
    _dist_rank,
    _voc_flush_on,
    agent,
    args_cli,
    best_frozen_completion,
    best_frozen_return,
    cfg,
    create_env,
    exp_dir,
    frozen_eval_snapshot,
    raw_env,
    rollouts,
    traj_mgr,
    wrapped,
    saved_snap_only,
):
    """Prepare the eval; in render mode, render and return early."""
    (_adaptive_on, _warmup_batches, clip_names, evals_per_clip, num_clips, policy_fn, wandb_log) = (
        frozen_eval_run.prepare_eval_batch(
            agent=agent,
            args_cli=args_cli,
            cfg=cfg,
            evals_per_clip=evals_per_clip,
            iteration=iteration,
            raw_env=raw_env,
            traj_mgr=traj_mgr,
            wrapped=wrapped,
        )
    )

    _ret = _render_frozen_eval(args_cli, create_env, raw_env, traj_mgr, wrapped, clip_names, num_clips, policy_fn)
    if _ret is not _CONTINUE:
        return _ret
    _side_rates, aggregate = frozen_eval_run.run_eval_batches(
        _adaptive_on=_adaptive_on,
        _warmup_batches=_warmup_batches,
        args_cli=args_cli,
        clip_names=clip_names,
        evals_per_clip=evals_per_clip,
        iteration=iteration,
        num_clips=num_clips,
        policy_fn=policy_fn,
        raw_env=raw_env,
        rollouts=rollouts,
        traj_mgr=traj_mgr,
        traj_recorder=traj_recorder,
        wandb_log=wandb_log,
        wrapped=wrapped,
    )
    frozen_eval_select.update_best_checkpoints(
        _dist=_dist,
        _dist_rank=_dist_rank,
        _side_rates=_side_rates,
        _voc_flush_on=_voc_flush_on,
        agent=agent,
        aggregate=aggregate,
        args_cli=args_cli,
        best_frozen_completion=best_frozen_completion,
        best_frozen_return=best_frozen_return,
        evals_per_clip=evals_per_clip,
        exp_dir=exp_dir,
        frozen_eval_snapshot=frozen_eval_snapshot,
        iteration=iteration,
        num_clips=num_clips,
        raw_env=raw_env,
        saved_snap_only=saved_snap_only,
    )
    return _CONTINUE


def _render_frozen_eval(args_cli, create_env, raw_env, traj_mgr, wrapped, clip_names, num_clips, policy_fn):
    """Render = frozen eval + cameras. At this point the frozen settings (curricula off, progress=1.0, rsi=0, identity
    DR, fixed ramps, cross-pair handling) are applied, so the rendered episodes run under the judge's conditions.
    Only with DEXENGINE_FROZEN_REPLAY=1 (both variables must be set): that path ends the process right after, so
    skipping the restore below is safe. Never on during training.
    """
    if os.environ.get("DEXENGINE_FROZEN_RENDER", "0") == "1" and os.environ.get("DEXENGINE_FROZEN_REPLAY", "0") == "1":
        _fr_ck = os.path.abspath(getattr(args_cli, "resume_from", None) or getattr(args_cli, "resume", None) or "")
        # Replay render (off by default): writes recorded states without physics and draws them with tiled cameras,
        # branching off at this same frozen point (dexengine.rl.render.replay).
        if os.environ.get("DEXENGINE_TRAJ_RENDER", ""):
            from dexengine.rl.render import replay as _trr

            _trr.run(wrapped, policy_fn, traj_mgr, raw_env, clip_names, args_cli=args_cli, checkpoint_path=_fr_ck)
            print("    [FROZEN RENDER] replay render DONE (restore skipped — FROZEN_REPLAY exits next)", flush=True)
            return None
        from dexengine.rl.render.clip_video import _run_per_clip_video_render as _fr_render

        print(
            f"    [FROZEN RENDER] rendering under the frozen eval conditions "
            f"(num_envs={wrapped.num_envs} clips={num_clips} ckpt={os.path.basename(_fr_ck)})",
            flush=True,
        )
        _fr_render(
            wrapped,
            policy_fn,
            traj_mgr,
            num_clips,
            clip_names,
            raw_env,
            checkpoint_path=_fr_ck,
            out_dir=str(_run_out(args_cli.videos_out_dir)),
            max_attempts=args_cli.videos_max_attempts,
            max_episode_steps=args_cli.videos_max_episode_steps,
            fps=args_cli.videos_fps,
            warmup_attempts=args_cli.videos_warmup_attempts,
            select_best_success=bool(args_cli.videos_select_best_success),
            rollouts_per_clip=args_cli.videos_rollouts_per_clip,
            rank_metric=args_cli.videos_rank_metric,
            args_cli=args_cli,
            create_env=create_env,
        )
        print("    [FROZEN RENDER] DONE (restore skipped — FROZEN_REPLAY exits next)", flush=True)
        return None
    return _CONTINUE
