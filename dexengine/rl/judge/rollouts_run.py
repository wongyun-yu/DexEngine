"""Run the balanced eval batches: the clip-to-env assignment, the per-clip result records, the aggregation and the eval
log.
"""

from __future__ import annotations

import torch

from dexengine.rl.judge.arm_penetration import OFF as PEN_OFF
from dexengine.rl.judge.park_e_oracle import collector_for
from dexengine.rl.judge.rollouts_batch import _run_single_batch
from dexengine.rl.judge.rollouts_report import _print_aggregate_report, _print_batch_report
from dexengine.rl.judge.rollouts_types import ClipResult

#: batch counts of the [C41] collection line; scripts/rl/judge_sidecar.sh reads `real batches N (+warmup W)` from it
C41_BATCHES = "real batches {n} (+warmup {w})"


def build_balanced_assignment(
    total_episodes: int,
    num_clips: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a deterministic balanced clip assignment using round-robin interleaving.

    Produces an interleaved sequence ``[0, 1, 2, ..., C-1, 0, 1, 2, ..., C-1, ...]``
    so that any contiguous slice (batch) contains a diverse mix of clips.

    If ``total_episodes`` is not divisible by ``num_clips``, the first
    ``total_episodes % num_clips`` clips get one extra slot, but the interleaving
    ensures they are spread across batches rather than clustered.

    Returns:
        Tensor of shape ``(total_episodes,)`` with clip IDs in [0, num_clips).
    """
    base_count = total_episodes // num_clips
    remainder = total_episodes % num_clips

    # Build interleaved: repeat [0,1,...,C-1] base_count times,
    # then append [0,1,...,remainder-1] for the extras
    cycle = list(range(num_clips))
    assignments = cycle * base_count + cycle[:remainder]

    return torch.tensor(assignments[:total_episodes], dtype=torch.long, device=device)


def _run_batches_and_log(
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
    """Run the balanced eval batches, aggregate them and log the result."""
    global_assignment, total_episodes = _plan_balanced_assignment(
        num_clips,
        evals_per_clip,
        warmup_batches,
        _snap_ids,
        _static,
        device,
        n_envs,
        real_total_episodes,
        warmup_total,
        total_episodes,
    )

    _ap, _pe, batch_results, clip_results, total_completed, total_early, total_eps, total_step_sum = (
        _make_clip_accumulators(
            wrapped_env,
            policy_fn,
            traj_mgr,
            num_clips,
            clip_names,
            per_clip_recorder,
            warmup_batches,
            arm_pen,
            traj_recorder,
            _snap_ids,
            _snap_len,
            _snap_sf,
            _static,
            n_envs,
            total_episodes,
            global_assignment,
        )
    )

    aggregate = _aggregate_results(batch_results, clip_results, total_completed, total_early, total_eps, total_step_sum)

    _print_aggregate_report(aggregate, num_clips)
    _pe.report()
    _ap.report()

    return _log_eval_to_wandb(wandb_log_fn, wandb_step, log_prefix, aggregate, batch_results, clip_results, total_eps)


def _aggregate_results(batch_results, clip_results, total_completed, total_early, total_eps, total_step_sum):
    """Aggregate the batch and clip results into the eval summary."""
    aggregate = {
        "num_batches": len(batch_results),
        "num_episodes": total_eps,
        "completion_rate": total_completed / max(total_eps, 1),
        "early_termination_rate": total_early / max(total_eps, 1),
        "mean_episode_length": total_step_sum / max(total_eps, 1),
        "per_clip": [
            {
                "clip_id": cr.clip_id,
                "clip_name": cr.clip_name,
                "completed": cr.completed,
                "early_terminated": cr.early_terminated,
                "total": cr.total,
                "completion_rate": cr.completion_rate,
                "excl_a_batches": cr.excl_a_batches,
            }
            for cr in clip_results.values()
        ],
    }
    return aggregate


def _plan_balanced_assignment(
    num_clips,
    evals_per_clip,
    warmup_batches,
    _snap_ids,
    _static,
    device,
    n_envs,
    real_total_episodes,
    warmup_total,
    total_episodes,
):
    """Clip-to-env assignment and episode total of this eval (static scenes keep their assignment)."""
    if _static:
        _fixed_counts = torch.bincount(_snap_ids, minlength=num_clips)
        _min_pb = int(_fixed_counts.min().item())
        if _min_pb <= 0:
            raise RuntimeError(f"[C41] static mode but some clip has 0 envs — NE={n_envs}, K={num_clips}")
        _n_real = -(-evals_per_clip // _min_pb)
        total_episodes = (warmup_batches + _n_real) * n_envs
        global_assignment = _snap_ids.repeat(warmup_batches + _n_real)
        print(
            f"    [C41][static] fixed-env clip collection: {C41_BATCHES.format(n=_n_real, w=warmup_batches)}, "
            f"{int(_fixed_counts.min())}~{int(_fixed_counts.max())} eps/batch per clip, target ≥{evals_per_clip}",
            flush=True,
        )
    # Warmup gets cyclic clip assignment; real eval gets balanced assignment.
    elif warmup_total > 0:
        warmup_assignment = torch.arange(warmup_total, device=device) % num_clips
        real_assignment = build_balanced_assignment(real_total_episodes, num_clips, device)
        global_assignment = torch.cat([warmup_assignment, real_assignment])
    else:
        global_assignment = build_balanced_assignment(real_total_episodes, num_clips, device)
    if not _static:
        # Batch-count marker of the same shape as the static one, so the judge sidecar's row check (real batches =
        # expected, [EVAL_PARK] lines = warm-up + real) also holds for single-object one-hand bundles
        # (replicate_physics,
        # not static). Batches = all warm-up batches at full size, then the real assignment NE at a time.
        n_real = -(-real_total_episodes // n_envs)
        print(
            f"    [C41][balanced] balanced-assignment collection: {C41_BATCHES.format(n=n_real, w=warmup_batches)}, "
            f"{evals_per_clip} eps per clip × {num_clips} clips · {n_envs} envs",
            flush=True,
        )
    return global_assignment, total_episodes


def _log_eval_to_wandb(wandb_log_fn, wandb_step, log_prefix, aggregate, batch_results, clip_results, total_eps):
    """Wandb: log_prefix selects the section; the legacy "eval/" · "eval_clip/" keys are ALWAYS logged too, with
    identical values, so older dashboards keep working.
    """
    if wandb_log_fn is not None and wandb_step is not None:
        log_dict = {
            # New clean keys (train_15/, heldout_5/, etc.)
            f"{log_prefix}/completion_rate": aggregate["completion_rate"] * 100,
            f"{log_prefix}/early_rate": aggregate["early_termination_rate"] * 100,
            f"{log_prefix}/episodes": total_eps,
            f"{log_prefix}/mean_ep_length": aggregate["mean_episode_length"],
            # Legacy keys (unconditional — with log_prefix "eval" the same value is written twice).
            "eval/completion_rate": aggregate["completion_rate"] * 100,
            "eval/early_rate": aggregate["early_termination_rate"] * 100,
            "eval/episodes": total_eps,
            "eval/mean_ep_length": aggregate["mean_episode_length"],
        }
        for cr in clip_results.values():
            log_dict[f"{log_prefix}/per_clip/{cr.clip_name}"] = cr.completion_rate * 100
            log_dict[f"eval_clip/{cr.clip_name}"] = cr.completion_rate * 100  # legacy
        wandb_log_fn(log_dict, wandb_step)
    return aggregate, batch_results


def _make_clip_accumulators(
    wrapped_env,
    policy_fn,
    traj_mgr,
    num_clips,
    clip_names,
    per_clip_recorder,
    warmup_batches,
    arm_pen,
    traj_recorder,
    _snap_ids,
    _snap_len,
    _snap_sf,
    _static,
    n_envs,
    total_episodes,
    global_assignment,
):
    """Per-clip accumulators (only for real eval portion).

    The E-clip oracle alone also collects the warm-up (a report).
    """
    _pe = collector_for(clip_names, num_clips, static=_static)
    _ap = arm_pen or PEN_OFF  # arm penetration (auxiliary, never changes the judge value); None = not measured
    batch_results, clip_results = _init_clip_results(
        wrapped_env,
        policy_fn,
        traj_mgr,
        num_clips,
        clip_names,
        per_clip_recorder,
        warmup_batches,
        traj_recorder,
        n_envs,
        total_episodes,
        global_assignment,
        _ap,
        _pe,
    )

    # Restore the clip assignment so the evaluation's assignment does not leak into training (see (1) above).
    traj_mgr.clip_ids[:] = _snap_ids
    traj_mgr.clip_lengths[:] = _snap_len
    traj_mgr.start_frames[:] = _snap_sf

    # Aggregate (only non-warmup batches / clip_results)
    total_completed = sum(cr.completed for cr in clip_results.values())
    total_early = sum(cr.early_terminated for cr in clip_results.values())
    total_eps = total_completed + total_early
    total_step_sum = sum(b.step_sum for b in batch_results)
    return _ap, _pe, batch_results, clip_results, total_completed, total_early, total_eps, total_step_sum


def _init_clip_results(
    wrapped_env,
    policy_fn,
    traj_mgr,
    num_clips,
    clip_names,
    per_clip_recorder,
    warmup_batches,
    traj_recorder,
    n_envs,
    total_episodes,
    global_assignment,
    _ap,
    _pe,
):
    """Empty per-clip and per-batch result records."""
    clip_results = {}
    for i in range(num_clips):
        name = clip_names[i] if clip_names else f"clip_{i}"
        clip_results[i] = ClipResult(clip_id=i, clip_name=name)

    batch_results = []
    batch_idx = 0
    offset = 0

    while offset < total_episodes:
        batch_size = min(n_envs, total_episodes - offset)

        # Slice from global assignment (guarantees exact per-clip balance)
        clip_assignment = global_assignment[offset : offset + batch_size]
        offset += batch_size

        _ap.batch(clip_assignment, batch_idx < warmup_batches)
        batch = _run_single_batch(
            wrapped_env=wrapped_env,
            policy_fn=policy_fn,
            traj_mgr=traj_mgr,
            clip_assignment=clip_assignment,
            batch_idx=batch_idx,
            batch_size=batch_size,
            per_clip_recorder=per_clip_recorder,
            arm_pen=_ap,
            clip_names=clip_names,
            traj_recorder=traj_recorder,
        )

        is_warmup = batch_idx < warmup_batches
        _pe.add(is_warmup, batch.clip_counts, batch.park_e)

        if is_warmup:
            # Print warmup report but do NOT accumulate into clip_results or batch_results
            _print_batch_report(batch, clip_results, num_clips, clip_names, warmup_prefix="[WARMUP — discarded] ")
        else:
            for cid, (comp, early) in batch.clip_counts.items():
                clip_results[cid].completed += comp
                clip_results[cid].early_terminated += early
            for cid in {
                c for c, _s in batch.park_excl_a
            }:  # gate-A exclusion only affects the PARK gates, not the score
                clip_results[cid].excl_a_batches += 1

            batch_results.append(batch)
            _print_batch_report(batch, clip_results, num_clips, clip_names)

        batch_idx += 1
    return batch_results, clip_results
