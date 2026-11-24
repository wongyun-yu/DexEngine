"""Clip-balanced evaluation with batch-based reporting (the judge loop).

One implementation serves the fresh-process judge, standalone evaluation and the in-training evaluation.

Key design:
- Each batch assigns clips to envs as uniformly as possible
- A batch completes when every env has finished its first episode
- Auto-reset envs that finish early continue running but are ignored
- Reporting is per-batch, not per-step

Layout: result records `rollouts_types` · one batch `rollouts_batch` · console reports `rollouts_report`; this module
holds the assignment and the loop over batches and re-exports the names other code imports from it.
"""

from __future__ import annotations

import sys

from dexengine.rl.judge.rollouts_batch import _run_single_batch
from dexengine.rl.judge.rollouts_run import _run_batches_and_log
from dexengine.rl.judge.rollouts_types import BatchResult, ClipResult

__all__ = ["BatchResult", "ClipResult", "_run_single_batch", "run_eval_balanced"]


def run_eval_balanced(
    wrapped_env,
    policy_fn,
    traj_mgr,
    num_clips: int,
    evals_per_clip: int,
    clip_names: list[str] | None = None,
    per_clip_recorder=None,
    wandb_log_fn=None,
    wandb_step: int | None = None,
    warmup_batches: int = 0,
    log_prefix: str = "eval",
    arm_pen=None,
    traj_recorder=None,
) -> tuple[dict, list[BatchResult]]:
    """Run clip-balanced evaluation with batch-based reporting.

    This is the shared implementation used by both standalone eval and
    frozen in-training eval.

    Args:
        wrapped_env: SKRL-wrapped Isaac Lab environment.
        policy_fn: Callable ``(obs_tensor) -> actions_tensor`` for deterministic
            action computation.  Caller is responsible for torch.no_grad().
        traj_mgr: The environment's TrajectoryManager instance.
        num_clips: Number of clips in the dataset.
        evals_per_clip: Number of evaluation episodes per clip.
        clip_names: Optional list of clip names (length ``num_clips``).
        per_clip_recorder: Optional PerClipRecorder for capturing one trajectory
            per clip. Records from ALL envs, saves first successful per clip.
        wandb_log_fn: Optional callable ``(metrics_dict, step)`` for wandb logging.
        wandb_step: Iteration number for wandb logging.
        warmup_batches: Number of leading batches to run but discard from
            aggregates (eliminates cold-start bias). Default 0 (no warmup).
        arm_pen: Arm-penetration collector (arm_penetration.collector_for), an auxiliary value that never changes
            the judge value. None = not measured.
        traj_recorder: Trajectory recorder (dexengine/rl/eval/traj_record.TrajRecorder; it only reads).
            None = no recording and no extra tensor op.

    Returns:
        Tuple of (aggregate_dict, list_of_BatchResult).
        aggregate_dict has keys: num_episodes, completion_rate,
        early_termination_rate, mean_episode_length, per_clip (list of dicts).
        batch_results contains only the non-warmup batches.
    """
    raw_env = wrapped_env.env.unwrapped
    device = raw_env.device
    n_envs = wrapped_env.num_envs

    real_total_episodes = num_clips * evals_per_clip
    warmup_total = warmup_batches * n_envs
    total_episodes = real_total_episodes + warmup_total

    return _apply_static_clip_assignment(
        wrapped_env,
        policy_fn,
        traj_mgr,
        num_clips,
        evals_per_clip,
        clip_names,
        per_clip_recorder,
        wandb_log_fn,
        wandb_step,
        warmup_batches,
        log_prefix,
        arm_pen,
        traj_recorder,
        device,
        n_envs,
        raw_env,
        real_total_episodes,
        total_episodes,
        warmup_total,
    )


def _apply_static_clip_assignment(
    wrapped_env,
    policy_fn,
    traj_mgr,
    num_clips,
    evals_per_clip,
    clip_names,
    per_clip_recorder,
    wandb_log_fn,
    wandb_step,
    warmup_batches,
    log_prefix,
    arm_pen,
    traj_recorder,
    device,
    n_envs,
    raw_env,
    real_total_episodes,
    total_episodes,
    warmup_total,
):
    """Multi-object scenes (static_clip_assignment): (1) The batches overwrite traj_mgr.clip_ids / clip_lengths /
    start_frames. They are snapshotted here and restored after the loop, so the clip assignment of the evaluation
    never leaks into the training that follows (a stale tail batch would leave the envs beyond batch_size on clip 0:
    reference = clip 0 against the spawned mesh of another clip). (2) In static mode each env's spawned meshes belong
    to a fixed clip, so the assignment must equal the env's fixed clip. The assignment is then the fixed env -> clip
    map repeated per batch (every batch full size, no tail), with B = ceil(target / min envs per clip) real batches
    to fill the quota, and the per-clip totals are reported as counted. A global interleave would rotate the labels
    by (NE·b) % K from the second batch on whenever K does not divide NE.
    """
    _static = bool(getattr(getattr(traj_mgr, "cfg", None), "static_clip_assignment", False))
    _snap_ids = traj_mgr.clip_ids.clone()
    _snap_len = traj_mgr.clip_lengths.clone()
    _snap_sf = traj_mgr.start_frames.clone()
    aggregate, batch_results = _run_balanced_batches(
        wrapped_env,
        policy_fn,
        traj_mgr,
        num_clips,
        evals_per_clip,
        clip_names,
        per_clip_recorder,
        wandb_log_fn,
        wandb_step,
        warmup_batches,
        log_prefix,
        arm_pen,
        traj_recorder,
        _snap_ids,
        _snap_len,
        _snap_sf,
        _static,
        device,
        n_envs,
        real_total_episodes,
        total_episodes,
        warmup_total,
    )

    return _with_extra_slots_report(raw_env, (aggregate, batch_results))


def _run_balanced_batches(
    wrapped_env,
    policy_fn,
    traj_mgr,
    num_clips,
    evals_per_clip,
    clip_names,
    per_clip_recorder,
    wandb_log_fn,
    wandb_step,
    warmup_batches,
    log_prefix,
    arm_pen,
    traj_recorder,
    _snap_ids,
    _snap_len,
    _snap_sf,
    _static,
    device,
    n_envs,
    real_total_episodes,
    total_episodes,
    warmup_total,
):
    """Warm-up and balanced clip assignment over the eval batches, with the batch-count markers."""
    return _run_batches_and_log(
        wrapped_env,
        policy_fn,
        traj_mgr,
        num_clips,
        evals_per_clip,
        clip_names,
        per_clip_recorder,
        wandb_log_fn,
        wandb_step,
        warmup_batches,
        log_prefix,
        arm_pen,
        traj_recorder,
        _snap_ids,
        _snap_len,
        _snap_sf,
        _static,
        device,
        n_envs,
        real_total_episodes,
        total_episodes,
        warmup_total,
    )


def _with_extra_slots_report(raw_env, result):
    """Print the final extra-slot note (dexengine.rl.tasks.scene.extra_slots.final_report) if that module is loaded,
    then return `result` unchanged. An env without slots does nothing; env registration loads the module (not loaded =
    no slots). A printed reference line only; it never changes a judge value."""
    xs_mod = sys.modules.get("dexengine.rl.tasks.scene.extra_slots")
    if xs_mod is not None:
        xs_mod.final_report(raw_env)
    return result
