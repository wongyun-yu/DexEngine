"""Per-iteration snapshot record: the W&B log, the keypoint metrics, the decoupled completion rates and the chunk
counters.
"""

from __future__ import annotations

from dexengine.rl.env_setup.sides import FINGERTIP_BODY_INDICES
from dexengine.rl.swallow import note_once


def log_snapshot_to_wandb(
    raw_env,
    iteration,
    rollouts,
    diag,
    agent_ref,
    wandb_enabled,
    chunk_eps,
    early_pct,
    f0_complete_pct,
    finger_std,
    fingertip_errs,
    jerk_max,
    jerk_mean,
    mean_body_errs,
    mean_ep_len,
    mean_frame_progress,
    rm,
    rsi_complete_pct,
    sched_progress,
    timeout_pct,
    wrist_errs,
    wrist_resid_std,
):
    """W&B log of the snapshot (termination/<term>_percent is logged separately in the training loop)."""
    record = _log_iteration_to_wandb(
        raw_env,
        iteration,
        rollouts,
        diag,
        agent_ref,
        wandb_enabled,
        chunk_eps,
        early_pct,
        f0_complete_pct,
        finger_std,
        fingertip_errs,
        jerk_max,
        jerk_mean,
        mean_body_errs,
        mean_ep_len,
        mean_frame_progress,
        rm,
        rsi_complete_pct,
        sched_progress,
        timeout_pct,
        wrist_errs,
        wrist_resid_std,
    )
    # Float counters specifically
    diag["chunk_ep_len_sum"] = 0.0
    diag["chunk_progress_sum"] = 0.0
    return record


def _log_iteration_to_wandb(
    raw_env,
    iteration,
    rollouts,
    diag,
    agent_ref,
    wandb_enabled,
    chunk_eps,
    early_pct,
    f0_complete_pct,
    finger_std,
    fingertip_errs,
    jerk_max,
    jerk_mean,
    mean_body_errs,
    mean_ep_len,
    mean_frame_progress,
    rm,
    rsi_complete_pct,
    sched_progress,
    timeout_pct,
    wrist_errs,
    wrist_resid_std,
):
    """W&B logging — explicit per-iteration log. We init wandb manually (no sync_tensorboard), so SKRL's tensorboard
    writer is NOT mirrored to wandb. That means anything we want in wandb has to be logged here directly.  We log
    (episode-sum is the PRIMARY reward diagnostic — never per-step mean): - episodes/complete_pct,
    episodes/frame_progress_pct - curriculum/schedule_progress - RewardEpisodeReturn/<term> (+/total): per-term
    episode return Σ (unbiased) - Diag/<name>: true-scale contact diagnostics (robot_contact_frac, thumb/index
    contact, grasp_target_dist, precision/recall, ref_contact_frac). The diag_* reward terms are registered at weight
    1e-9 (so RewardManager executes them without changing the reward); we rescale by 1e9 here so the dashboard is
    interpretable and NOT misleading, and exclude them from RewardEpisodeReturn. -
    Reward/total_episode_mean/max/min/count: SKRL _track_rewards deque (last 100 eps) Termination/<term>_percent is
    logged separately in the training loop.
    """
    return _iteration_record(
        raw_env,
        iteration,
        rollouts,
        diag,
        agent_ref,
        wandb_enabled,
        chunk_eps,
        early_pct,
        f0_complete_pct,
        finger_std,
        fingertip_errs,
        jerk_max,
        jerk_mean,
        mean_body_errs,
        mean_ep_len,
        mean_frame_progress,
        rm,
        rsi_complete_pct,
        sched_progress,
        timeout_pct,
        wrist_errs,
        wrist_resid_std,
    )


def _iteration_record(
    raw_env,
    iteration,
    rollouts,
    diag,
    agent_ref,
    wandb_enabled,
    chunk_eps,
    early_pct,
    f0_complete_pct,
    finger_std,
    fingertip_errs,
    jerk_max,
    jerk_mean,
    mean_body_errs,
    mean_ep_len,
    mean_frame_progress,
    rm,
    rsi_complete_pct,
    sched_progress,
    timeout_pct,
    wrist_errs,
    wrist_resid_std,
):
    """The per-iteration record (logged to W&B when enabled)."""
    if wandb_enabled:
        import wandb

        _kp_err, _kp_rew, wandb_log = _log_decoupled_completion_rates(
            raw_env,
            agent_ref,
            f0_complete_pct,
            jerk_max,
            jerk_mean,
            mean_frame_progress,
            rm,
            rsi_complete_pct,
            sched_progress,
            timeout_pct,
        )
        _print_keypoint_metrics(_kp_err, _kp_rew, wandb_log)
        # Use timestep as the wandb step so it aligns with SKRL's internal
        # writer (which logs at timesteps == iteration * rollouts).
        wandb.log(wandb_log, step=iteration * rollouts)

    # Build record for history
    record = {
        "iteration": iteration,
        "chunk_episodes": chunk_eps,
        "early_pct": early_pct,
        "mean_ep_len": mean_ep_len,
        "wrist_avg": wrist_errs.mean().item(),
        "body_avg": mean_body_errs.mean().item(),
        "fingertip_avg": {name: fingertip_errs[name].mean().item() for name in FINGERTIP_BODY_INDICES},
        "wrist_resid_std": wrist_resid_std.mean().item() if wrist_resid_std is not None else None,
        "finger_std": finger_std.mean().item() if finger_std is not None else None,
        "sched_progress": sched_progress,
    }

    _reset_chunk_counters(diag)
    return record


def _print_keypoint_metrics(_kp_err, _kp_rew, wandb_log):
    """Print the object-keypoint reward and error lines."""
    if _kp_rew or _kp_err:
        for _kp_side, _sfx in (("right", ""), ("left", "_left")):
            _r0 = float(_kp_rew[(_kp_side, 0)].mean().item()) if (_kp_side, 0) in _kp_rew else None
            _r1 = float(_kp_rew[(_kp_side, 1)].mean().item()) if (_kp_side, 1) in _kp_rew else None
            _e0 = float(_kp_err[(_kp_side, 0)].mean().item() * 100.0) if (_kp_side, 0) in _kp_err else None  # cm
            _e1 = float(_kp_err[(_kp_side, 1)].mean().item() * 100.0) if (_kp_side, 1) in _kp_err else None  # cm
            if _r0 is not None:
                wandb_log[f"reward/object_keypoint_1{_sfx}"] = _r0
            if _r1 is not None:
                wandb_log[f"reward/object_keypoint_2{_sfx}"] = _r1
            if _r0 is not None and _r1 is not None:
                wandb_log[f"reward/object_keypoint_mean{_sfx}"] = (_r0 + _r1) * 0.5
                wandb_log[f"reward/object_keypoint_total{_sfx}"] = _r0 + _r1
            if _e0 is not None:
                wandb_log[f"metrics/object_keypoint_1_error_cm{_sfx}"] = _e0
            if _e1 is not None:
                wandb_log[f"metrics/object_keypoint_2_error_cm{_sfx}"] = _e1
            if _e0 is not None and _e1 is not None:
                wandb_log[f"metrics/object_keypoint_mean_error_cm{_sfx}"] = (_e0 + _e1) * 0.5


def _reset_chunk_counters(diag):
    """Reset chunk counters."""
    for key in [
        "chunk_episodes",
        "chunk_early",
        "chunk_timeouts",
        "chunk_ep_len_sum",
        "chunk_progress_sum",
        "chunk_f0_episodes",
        "chunk_f0_early",
        "chunk_rsi_episodes",
        "chunk_rsi_early",
    ]:
        diag[key] = 0


def _log_decoupled_completion_rates(
    raw_env,
    agent_ref,
    f0_complete_pct,
    jerk_max,
    jerk_mean,
    mean_frame_progress,
    rm,
    rsi_complete_pct,
    sched_progress,
    timeout_pct,
):
    """Under the decoupled mode the three completion rates are structurally 100 % (see above). The keys stay where they
    are; the console label (⚠structural) says why they read 100 %.
    """
    _cp = "episodes/"
    wandb_log = {
        f"{_cp}complete_pct": timeout_pct,
        # Cold (frame-0-only) completion — the RSI-free, eval-matching progress signal.
        # complete_pct above is inflated by RSI mid-trajectory starts (near-clip-end
        # episodes trivially reach trajectory_done); cold_complete_pct tracks the frame-0
        # episodes only, so it matches the balanced/frozen eval. Watch THIS for real
        # multi-clip progress. rsi_complete_pct is the inflated complement, logged so the
        # gap (RSI artifact) is visible on the dashboard.
        # Both are already computed above (zero extra counting); this only adds 2 scalars.
        f"{_cp}cold_complete_pct": f0_complete_pct,
        # ABSOLUTE achieved arm jerk (rad/s^3), the tremble criterion.
        **({"jerk/arm_achieved_mean": jerk_mean, "jerk/arm_achieved_max": jerk_max} if jerk_mean == jerk_mean else {}),
        f"{_cp}rsi_complete_pct": rsi_complete_pct,
        f"{_cp}frame_progress_pct": mean_frame_progress * 100,
        "curriculum/schedule_progress": sched_progress,
    }
    # PRIMARY per-term reward components: episode return Σ (unbiased), from
    # env.extras log × max_episode_length_s. This is what to watch for progress.
    _mels = float(getattr(raw_env, "max_episode_length_s", 0.0) or 0.0)
    _elog = getattr(raw_env, "extras", {}).get("log", {}) if hasattr(raw_env, "extras") else {}
    if _mels > 0 and _elog:
        _tot = 0.0
        _DIAG_RECOVER = 1e9  # diag_* terms are logged at weight 1e-9; rescale to true value
        for name in rm.active_terms:
            v = _elog.get(f"Episode_Reward/{name}")
            if v is None:
                continue
            vv = (v.item() if hasattr(v, "item") else float(v)) * _mels
            if name.startswith("diag_"):
                # Clean, true-scale diagnostic namespace (NOT a reward; excluded from
                # the reward total). Recover the 1e-9 logging weight.
                wandb_log[f"Diag/{name[len('diag_') :]}"] = vv * _DIAG_RECOVER
            else:
                wandb_log[f"RewardEpisodeReturn/{name}"] = vv
                _tot += vv
        wandb_log["RewardEpisodeReturn/total"] = _tot
    # Cumulative per-episode reward — pull from SKRL's internal deque.
    # _track_rewards is a deque of (up to) the last 100 finished episodes'
    # cumulative sums. Empty until at least one episode finishes.
    if agent_ref is not None:
        try:
            tr = list(getattr(agent_ref, "_track_rewards", []))
            if tr:
                import numpy as _np

                arr = _np.asarray(tr, dtype=_np.float32)
                wandb_log["Reward/total_episode_mean"] = float(arr.mean())
                wandb_log["Reward/total_episode_max"] = float(arr.max())
                wandb_log["Reward/total_episode_min"] = float(arr.min())
                wandb_log["Reward/total_episode_count"] = int(arr.size)
        except (AttributeError, TypeError, ValueError) as _exc:
            note_once("SNAPSHOT track rewards", _exc)
    # exp/obj-keypoint-reward: custom per-keypoint reward + error keys.
    # The diag_object_keypoint_* terms are auto-logged above via the diag_* branch
    # (as Diag/object_keypoint_*_err in cm, recovered ×1e9). Additionally log the
    # reward means and error means under reward/* and metrics/* namespaces from the
    # values stashed by RewardObjKeypoint on the env object during the last step.
    # Stash is keyed by (side, kp_index). Emit RIGHT with bare names and LEFT with a
    # '_left' suffix (symmetric W&B convention; no left+right averaging).
    _kp_rew = getattr(raw_env, "_obj_kp_reward", {})
    _kp_err = getattr(raw_env, "_obj_kp_err", {})
    return _kp_err, _kp_rew, wandb_log
