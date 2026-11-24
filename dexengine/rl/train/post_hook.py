"""Post-interaction hook of the PPO learner and the state it shares with the frozen eval.

``patched_post`` runs after every skrl ``post_interaction``: the non-finite guard, the ``[CKPT_SAVED]`` marker, and
every
``--eval_interval`` iterations the episode/termination summary, snapshot metrics, the frozen eval and the
planned-restart
check (``_mem_restart_check``). ``_Exp1EarlyStop`` unwinds ``runner.run()`` when the frozen eval asks to stop.
"""

import os

import torch

from dexengine.rl.judge.snapshot_metrics import snapshot_metrics
from dexengine.rl.train.nonfinite_guard import check_after_update


def prepare_post_state(*, agent, exp_dir, runner, args_cli):
    """State shared by the post-interaction hook and the frozen eval: the checkpoint-path rule and the
    best-checkpoint trackers."""
    original_post = runner.agent.post_interaction

    last_eval_iter = [0]

    def _ckpt_path_at(total_timesteps):
        """Path of the checkpoint skrl actually wrote at this step (``timestep + 1`` of post_interaction), or None.

        The rule follows the skrl base learner: post_interaction counts ``timestep + 1`` and calls write_checkpoint only
        when ``timestep % checkpoint_interval == 0``; write_checkpoint names the file ``agent_<timestep>.pt``. So no
        file exists when MI is not a multiple of the checkpoint interval, or on a distributed rank != 0 (interval 0).
        The rule is followed by an ``os.path.exists`` check, so a mismatch yields None instead of a wrong name.
        ``patched_post`` calls ``original_post`` first, so a checkpoint of this step is already on disk here.
        """
        _ag = runner.agent
        _iv = int(getattr(_ag, "checkpoint_interval", 0) or 0)
        if _iv <= 0 or total_timesteps % _iv != 0:
            return None
        if getattr(_ag, "checkpoint_store_separately", False):
            return None  # per-module checkpoint files: there is no single file name
        _p = os.path.join(getattr(_ag, "experiment_dir", exp_dir), "checkpoints", f"agent_{total_timesteps}.pt")
        return _p if os.path.exists(_p) else None

    # Best-checkpoint tracker (frozen eval completion rate, NOT training reward).
    # SKRL's default best_agent.pt is saved based on the rolling-100 mean of training
    # episode cumulative rewards (a shaped sum), which does not match what we actually
    # care about (frozen-eval completion rate). We track the highest frozen-eval
    # completion rate ourselves and overwrite best_agent.pt whenever it improves.
    best_frozen_completion = [-1.0]
    # Rolling training return recorded at the last best_agent.pt save.
    # Tie-break: equal completion + higher current rolling return -> re-save,
    # so best_agent.pt does not freeze at the first 100%-completion eval.
    best_frozen_return = [-float("inf")]
    # VOC v2.1 flush hygiene: pre-flush frozen evals run WITH assist active —
    # their completion must NOT set best_agent.pt. Our tracker below is gated on
    # voc.flush_complete; ALSO lock skrl's rolling-reward fallback now so no
    # assist-active phase can write best_agent.pt through it either.
    _voc_flush_on = int(getattr(args_cli, "voc", 0)) == 1 and float(getattr(args_cli, "voc_flush_frac", 0.0)) > 0.0
    if _voc_flush_on:
        agent.checkpoint_best_modules["reward"] = float("inf")
        agent.checkpoint_best_modules["saved"] = True
        agent.checkpoint_best_modules["modules"] = {}
        print(
            "[VOC] flush enabled — best_agent.pt restricted to POST-flush frozen "
            "evals (skrl reward-based fallback locked)."
        )
    return (_ckpt_path_at, _voc_flush_on, best_frozen_completion, best_frozen_return, last_eval_iter, original_post)


def _mem_restart_check(iteration, *, args_cli, frozen_eval_snapshot):
    """Planned-restart check (``DEXENGINE_MEM_RESTART_FREE_GB``).

    PhysX solver buffers only grow; when the GPU headroom expected after training resumes drops below the threshold,
    this asks the run to stop at this iteration (``frozen_eval_snapshot._stop``) so the launcher restarts it from this
    iteration's checkpoint with a fresh heap. Called after each frozen eval, or at the same interval when the
    in-training eval is off.
    """
    _mr_thr = float(os.environ.get("DEXENGINE_MEM_RESTART_FREE_GB", "0") or 0.0)
    # Skip the iteration the run resumed at (iteration == --resume_iteration): restarting there would repeat the
    # same iteration forever.
    _mr_resume_it = int(getattr(args_cli, "resume_iteration", 0) or 0)
    if (
        _mr_thr > 0
        and torch.cuda.is_available()
        and os.environ.get("LOCAL_RANK", "0") == "0"
        and int(iteration) > _mr_resume_it
    ):
        _mr_free, _mr_total = torch.cuda.mem_get_info()
        _mr_headroom = (_mr_free - (torch.cuda.max_memory_reserved() - torch.cuda.memory_reserved())) / 2**30
        print(
            f"    [MEM_CHECK] iter {iteration}: free={_mr_free / 2**30:.2f}G torch_peak_reserved="
            f"{torch.cuda.max_memory_reserved() / 2**30:.2f}G → headroom after training resumes ≈"
            f"{_mr_headroom:.2f}G (threshold {_mr_thr:.1f}G)",
            flush=True,
        )
        if _mr_headroom < _mr_thr:
            _es = frozen_eval_snapshot
            _es._mem_restart = int(iteration)
            _es._stop = True
            print(
                f"[MEM_RESTART] iter {iteration}: expected headroom after resume {_mr_headroom:.2f}G < "
                f"{_mr_thr:.1f}G — planned restart from this iteration's ckpt (PhysX heap "
                f"{_physx_gpu_stats()[:60]}…)",
                flush=True,
            )


def patched_post(
    timestep,
    timesteps,
    *,
    _aux_loss_log,
    _aux_loss_scale,
    _aux_target_magnitude_log,
    _ckpt_path_at,
    _fast_diag_flush,
    _mem_restart_check,
    args_cli,
    diag,
    frozen_eval_snapshot,
    history,
    last_eval_iter,
    original_post,
    raw_env,
    rollouts,
    runner,
    side,
    traj_mgr,
):
    """skrl ``post_interaction`` replacement (see the module docstring)."""
    result = original_post(timestep, timesteps)
    iteration = timestep // rollouts
    return _check_update_finite(
        timestep,
        _aux_loss_log,
        _aux_loss_scale,
        _aux_target_magnitude_log,
        _ckpt_path_at,
        _fast_diag_flush,
        _mem_restart_check,
        args_cli,
        diag,
        frozen_eval_snapshot,
        history,
        last_eval_iter,
        raw_env,
        rollouts,
        runner,
        side,
        traj_mgr,
        iteration,
        result,
    )


def _check_update_finite(
    timestep,
    _aux_loss_log,
    _aux_loss_scale,
    _aux_target_magnitude_log,
    _ckpt_path_at,
    _fast_diag_flush,
    _mem_restart_check,
    args_cli,
    diag,
    frozen_eval_snapshot,
    history,
    last_eval_iter,
    raw_env,
    rollouts,
    runner,
    side,
    traj_mgr,
    iteration,
    result,
):
    """After a call that ran the PPO update: finiteness of weights and preprocessors (nonfinite_guard). A non-finite
    update moves this call's checkpoint aside and raises — before the [CKPT_SAVED] marker below.
    """
    check_after_update(runner.agent, timestep, rollouts, _ckpt_path_at)
    # Checkpoint marker: the launcher's auto-resume picks "the last checkpoint saved before the first error line"
    # from these lines. Independent of the in-training eval. The file is checked only when the save rule fires.
    _ck_iv = int(getattr(runner.agent, "checkpoint_interval", 0) or 0)
    if _ck_iv > 0 and (timestep + 1) % _ck_iv == 0:
        _ck_saved = _ckpt_path_at(timestep + 1)
        if _ck_saved:
            # The leading newline matters: skrl's tqdm bar writes to stdout without one, so the marker would
            # otherwise follow a bar fragment on the same line (parsers match it anywhere in a line).
            print(f"\n[CKPT_SAVED] {os.path.basename(_ck_saved)} iter={(timestep + 1) // rollouts}", flush=True)
    # No periodic torch.cuda.empty_cache() here (it destabilized training); the frozen eval empties the cache.
    if iteration > 0 and iteration % args_cli.eval_interval == 0 and iteration != last_eval_iter[0]:
        last_eval_iter[0] = iteration
        _fast_diag_flush()  # collect the step diagnostics before the episode / termination / snapshot output
        print(f"\n  Iteration {iteration:5d}  (timestep {timestep})")
        _print_chunk_stats(args_cli, diag, rollouts, iteration)
        _print_snapshot_metrics(
            _aux_loss_log,
            _aux_loss_scale,
            _aux_target_magnitude_log,
            _mem_restart_check,
            args_cli,
            diag,
            frozen_eval_snapshot,
            history,
            raw_env,
            rollouts,
            runner,
            side,
            traj_mgr,
            iteration,
        )
    return result


def _print_chunk_stats(args_cli, diag, rollouts, iteration):
    """Print the episode statistics of this chunk."""
    if diag["chunk_episodes"] > 0:
        chunk_eps = diag["chunk_episodes"]
        chunk_early_pct = diag["chunk_early"] / chunk_eps * 100
        print(f"    Episodes: {chunk_eps}  Early: {diag['chunk_early']} ({chunk_early_pct:.1f}%)")
        # Per-term termination breakdown
        term_keys = sorted([k for k in diag if k.startswith("chunk_term_")])
        if term_keys:
            print("    Termination breakdown:")
            for k in term_keys:
                tname = k.replace("chunk_term_", "")
                count = diag[k]
                pct = count / max(1, diag["chunk_early"]) * 100
                print(f"      {tname:25s}: {count:8d} ({pct:5.1f}%)")
            # W&B: per-term termination percentages (share of early terminations
            # this chunk), so dashboards show which term dominates over iterations
            # (Termination/object_tracking_percent, .../fingertip_exceeded_percent,
            # .../object_velocity_percent, ...). NOTE: object_tracking is a single
            # combined pos+rot term (mdp.ObjectTrackingExceeded), not split into
            # position/rotation — splitting would require changing that term.
            if args_cli.wandb:
                import wandb as _wandb

                _term_pct = {
                    f"Termination/{k.replace('chunk_term_', '')}_percent": diag[k] / max(1, diag["chunk_early"]) * 100
                    for k in term_keys
                }
                _term_pct["Termination/early_percent"] = chunk_early_pct
                _wandb.log(_term_pct, step=iteration * rollouts)


def _print_snapshot_metrics(
    _aux_loss_log,
    _aux_loss_scale,
    _aux_target_magnitude_log,
    _mem_restart_check,
    args_cli,
    diag,
    frozen_eval_snapshot,
    history,
    raw_env,
    rollouts,
    runner,
    side,
    traj_mgr,
    iteration,
):
    """Print snapshot metrics if available."""
    try:
        record = snapshot_metrics(
            raw_env,
            traj_mgr,
            side,
            iteration,
            rollouts,
            diag,
            agent_ref=runner.agent,
            wandb_enabled=args_cli.wandb,
        )
        history.append(record)
    except Exception as e:  # noqa: BLE001 — a failed report is printed; training goes on
        print(f"    [WARN] Snapshot failed: {e}")

    # Auxiliary force loss: stdout + W&B
    if _aux_loss_scale > 0.0:
        print(
            f"    aux_force_loss = {_aux_loss_log[0]:.4f}  target_mag = {_aux_target_magnitude_log[0]:.4f}  "
            f"(scale={_aux_loss_scale})"
        )
        if args_cli.wandb:
            import wandb as _wb

            _wb.log(
                {
                    "Loss/aux_force_loss": _aux_loss_log[0],
                    "Loss/aux_force_target_magnitude": _aux_target_magnitude_log[0],
                },
                step=iteration * rollouts,
            )

    # Frozen deterministic eval: pause curriculum, run deterministic rollout, resume
    if not args_cli.no_frozen_eval:
        _fe_epc = int(args_cli.eval_evals_per_clip)
        frozen_eval_snapshot(iteration, evals_per_clip=_fe_epc)
        # Exp-1 early stop: flag set inside frozen_eval_snapshot AFTER env
        # restore; raise here to unwind cleanly out of runner.run().
        if getattr(frozen_eval_snapshot, "_stop", False):
            raise _Exp1EarlyStop()
    else:
        # In-training eval off (the canonical setting): the planned-restart check still runs at this interval.
        _mem_restart_check(iteration)
        if getattr(frozen_eval_snapshot, "_stop", False):
            raise _Exp1EarlyStop()

    diag["chunk_episodes"] = 0
    diag["chunk_early"] = 0
    diag["chunk_timeouts"] = 0
    diag["chunk_ep_len_sum"] = 0.0
    diag["chunk_f0_episodes"] = 0
    diag["chunk_f0_early"] = 0
    diag["chunk_rsi_episodes"] = 0
    diag["chunk_rsi_early"] = 0
    # Reset per-term counters
    for k in list(diag.keys()):
        if k.startswith("chunk_term_"):
            diag[k] = 0


def _physx_gpu_stats() -> str:
    """One line of PhysX scene GPU heap / constraint statistics (``physx_stats.physx_stats_line``), or the error."""
    try:
        from dexengine.rl.train.physx_stats import physx_stats_line

        return physx_stats_line()
    except Exception as e:  # noqa: BLE001 — a failed report is printed; training goes on
        return f"physx: {type(e).__name__}: {e}"


class _Exp1EarlyStop(Exception):
    """Raised from the frozen-eval hook to stop training early when the
    episode return has collapsed (Exp-1 early stopping). Caught around runner.run()."""
