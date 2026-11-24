"""Training-progress snapshot printed (and logged to W&B) at a given iteration.

snapshot_metrics prints the tracking errors, termination statistics, curriculum state, policy std, per-term episode
returns and hand-coupling check of the current rollout, logs them to W&B and returns a history record. All state comes
in as parameters; the shared body-index constants live in dexengine.rl.env_setup.build.
"""

from __future__ import annotations

import torch

from dexengine.rl.env_setup.sides import FINGERTIP_BODY_INDICES, PALM_BODY_INDEX
from dexengine.rl.judge.snapshot_report import print_policy_std_and_rewards


def _effective_fingertip_thresholds(raw_env, progress):
    """Effective per-finger fingertip-termination thresholds (meters) at a
    given schedule progress, read from the LIVE termination cfg (never hardcoded:
    the thresholds differ between tasks).

    Returns ``{finger: meters}`` or None if the ``fingertip_exceeded`` term is
    absent (e.g. eval-only task variants without that termination).
    """
    try:
        cfg = raw_env.termination_manager.get_term_cfg("fingertip_exceeded")
        start = cfg.params["thresholds_start"]
        end = cfg.params["thresholds_end"]
        p = float(progress)
        return {f: start[f] + (end[f] - start[f]) * p for f in start}
    except (TypeError, ValueError, KeyError):
        return None


def _fmt_fingertip_thresholds(eff):
    """One-line formatter for _effective_fingertip_thresholds output."""
    if eff is None:
        return "fingertip_exceeded term absent — no effective thresholds"
    return "effective fingertip thresholds [m]: " + ", ".join(f"{k}={v:.3f}" for k, v in eff.items())


# ======================================================================
# Diagnostic snapshot (shared by train and eval)
# ======================================================================


def snapshot_metrics(raw_env, traj_mgr, side, iteration, rollouts, diag, agent_ref=None, wandb_enabled=False):
    """Print body tracking errors, coupling, rewards at a given iteration.

    Args:
        raw_env: Unwrapped environment.
        traj_mgr: Trajectory manager reference.
        side: "right" or "left".
        iteration: Current PPO iteration.
        rollouts: Steps per rollout.
        diag: Mutable diagnostics dict with chunk counters.
        agent_ref: the skrl PPO learner (for policy std extraction).
        wandb_enabled: Whether to log to W&B.
    """
    hand_asset = raw_env.scene[f"{side}_hand"]
    device = raw_env.device
    frame_ids = raw_env.episode_length_buf

    return _reference_body_positions(
        raw_env, traj_mgr, side, iteration, rollouts, diag, agent_ref, wandb_enabled, device, frame_ids, hand_asset
    )


def _reference_body_positions(
    raw_env, traj_mgr, side, iteration, rollouts, diag, agent_ref, wandb_enabled, device, frame_ids, hand_asset
):
    """Reference body positions."""
    ref_body = traj_mgr.get_reference_body_pos(frame_ids)
    ref_pos = ref_body[f"{side}_body_pos"]  # (N, 13, 3)

    (
        _dec,
        active_mask,
        chunk_eps,
        early_pct,
        f0_complete_pct,
        f0_eps,
        fingertip_errs,
        jerk_max,
        jerk_mean,
        mean_body_errs,
        mean_ep_len,
        mean_frame_progress,
        n_active,
        rsi_complete_pct,
        rsi_eps,
        timeout_pct,
        wrist_errs,
    ) = _current_body_positions(
        raw_env, traj_mgr, side, iteration, rollouts, diag, device, frame_ids, hand_asset, ref_pos
    )

    sched_progress, step_count = _trajectory_complete_wandb_episodes(
        raw_env, traj_mgr, _dec, f0_complete_pct, f0_eps, rsi_complete_pct, rsi_eps
    )
    record = print_policy_std_and_rewards(
        raw_env,
        side,
        iteration,
        rollouts,
        diag,
        agent_ref,
        wandb_enabled,
        active_mask,
        chunk_eps,
        early_pct,
        f0_complete_pct,
        fingertip_errs,
        hand_asset,
        jerk_max,
        jerk_mean,
        mean_body_errs,
        mean_ep_len,
        mean_frame_progress,
        n_active,
        rsi_complete_pct,
        sched_progress,
        step_count,
        timeout_pct,
        wrist_errs,
    )

    return record


def _trajectory_complete_wandb_episodes(raw_env, traj_mgr, _dec, f0_complete_pct, f0_eps, rsi_complete_pct, rsi_eps):
    """ "Trajectory complete" / wandb episodes/complete_pct above pools RSI starts and is therefore INFLATED (mid-clip
    resets finish trivially). Only the Frame-0 number is comparable to the evaluations (they run rsi_fraction=0); the
    label says so inline. Under the decoupled mode both numbers are structurally 100 % (episodes end only by
    trajectory_done), unrelated to the policy, and the "eval-comparable" label would be false, so that case prints
    its own label.
    """
    if _dec:
        print(
            f"    Frame-0: {f0_eps} eps, {f0_complete_pct:.1f}% complete  "
            f"⚠ decoupled — **structural 100%**, not performance (the judge reads the frozen eval's "
            f"Completion rate; wandb: train_15/completion_rate)"
            f"  |  RSI: {rsi_eps} eps, {rsi_complete_pct:.1f}%"
        )
    else:
        print(
            f"    Frame-0: {f0_eps} eps, {f0_complete_pct:.1f}% complete  <-- eval-comparable"
            f"  |  RSI: {rsi_eps} eps, {rsi_complete_pct:.1f}% complete (inflated)"
        )

    # Curriculum state
    sched_progress = traj_mgr.schedule_progress
    rsi_frac = traj_mgr.rsi_fraction
    step_count = raw_env.common_step_counter
    # effective thresholds come from the LIVE termination cfg
    _eff_thr = _effective_fingertip_thresholds(raw_env, sched_progress)
    print("\n    Curriculum:")
    print(f"      schedule_progress:  {sched_progress:.4f}  ({_fmt_fingertip_thresholds(_eff_thr)})")
    print(f"      rsi_fraction:       {rsi_frac:.4f}")
    return sched_progress, step_count


def _current_body_positions(raw_env, traj_mgr, side, iteration, rollouts, diag, device, frame_ids, hand_asset, ref_pos):
    """Current body positions (world -> env-local)."""
    body_ids_isaac = traj_mgr.right_body_ids_isaac if side == "right" else traj_mgr.left_body_ids_isaac
    body_ids_t = torch.tensor(body_ids_isaac, device=device, dtype=torch.long)
    cur_pos = hand_asset.data.body_link_pos_w[:, body_ids_t, :] - raw_env.scene.env_origins.unsqueeze(1)

    # Per-body L2 errors: (N, 13)
    errors = torch.norm(ref_pos - cur_pos, dim=-1)

    # Filter out grace-period environments (episode_length_buf < 20) to match
    # termination logic. Without this, freshly-reset envs inflate Max values.
    _GRACE = 20
    active_mask = frame_ids >= _GRACE
    n_active = active_mask.sum().item()
    if n_active > 0:
        errors_filtered = errors[active_mask]
    else:
        errors_filtered = errors  # fallback: use all if none past grace

    wrist_errs = errors_filtered[:, PALM_BODY_INDEX]
    fingertip_errs = {name: errors_filtered[:, idx] for name, idx in FINGERTIP_BODY_INDICES.items()}
    mean_body_errs = errors_filtered.mean(dim=-1)

    # Chunk termination stats
    chunk_eps = diag["chunk_episodes"]
    chunk_early = diag["chunk_early"]
    chunk_timeouts = diag["chunk_timeouts"]
    early_pct = chunk_early / max(1, chunk_eps) * 100
    timeout_pct = chunk_timeouts / max(1, chunk_eps) * 100
    mean_ep_len = diag["chunk_ep_len_sum"] / max(1, chunk_eps)
    # Mean fraction of trajectory completed across all terminated episodes
    mean_frame_progress = diag.get("chunk_progress_sum", 0.0) / max(1, chunk_eps)

    return _under_decoupled_mode_quantity(
        raw_env,
        iteration,
        rollouts,
        diag,
        active_mask,
        chunk_early,
        chunk_eps,
        chunk_timeouts,
        early_pct,
        fingertip_errs,
        mean_body_errs,
        mean_ep_len,
        mean_frame_progress,
        n_active,
        timeout_pct,
        wrist_errs,
    )


def _under_decoupled_mode_quantity(
    raw_env,
    iteration,
    rollouts,
    diag,
    active_mask,
    chunk_early,
    chunk_eps,
    chunk_timeouts,
    early_pct,
    fingertip_errs,
    mean_body_errs,
    mean_ep_len,
    mean_frame_progress,
    n_active,
    timeout_pct,
    wrist_errs,
):
    """Under the decoupled mode every quantity derived from terminations saturates: the side gate only snaps and never
    terminates, so episodes end by trajectory_done alone (early = 0, timeouts = all, ep_len = T, frame_progress = 100
    %). Every saturated quantity is marked.
    """
    _dec = getattr(raw_env, "_pss_mode", "") == "decoupled"
    _mk = "  ⚠structural" if _dec else ""

    print(f"\n  {'=' * 70}")
    print(f"  Iteration {iteration:6d}  (timestep {iteration * rollouts})")
    print(f"  {'=' * 70}")
    print(f"    Episodes this chunk:  {chunk_eps}")
    print(f"    Early termination:    {chunk_early} ({early_pct:.1f}%){_mk}")
    print(f"    Trajectory complete:  {chunk_timeouts} ({timeout_pct:.1f}%){_mk}")
    print(f"    Mean episode length:  {mean_ep_len:.1f} steps{_mk}")
    print(f"    Mean frame progress:  {mean_frame_progress * 100:.1f}% of clip{_mk}")
    if _dec:
        print(
            "    ⚠ decoupled: the 4 values above derive from terminations and **saturate structurally**"
            " (early=0 · complete=100% · len=T · progress=100%). Not performance."
            " The judge is the frozen eval's Completion rate → wandb train_15/completion_rate."
        )

    # Frame-0 vs RSI breakdown
    f0_eps = diag.get("chunk_f0_episodes", 0)
    f0_early = diag.get("chunk_f0_early", 0)
    rsi_eps = diag.get("chunk_rsi_episodes", 0)
    rsi_early = diag.get("chunk_rsi_early", 0)
    f0_complete_pct = (f0_eps - f0_early) / max(1, f0_eps) * 100
    rsi_complete_pct = (rsi_eps - rsi_early) / max(1, rsi_eps) * 100
    # ACHIEVED arm jerk (rad/s^3), drained from the per-step accumulator of the residual arm IK action. Judged on the
    # ABSOLUTE value: the achieved/reference ratio cannot tell a smooth run from one faithfully reproducing a jerky
    # reference. Compare against the clip's reference jerk to see how much of it is inherited.
    _jk = getattr(raw_env, "_arm_jerk_acc", None)
    jerk_mean = jerk_max = float("nan")
    if _jk and _jk.get("n"):
        jerk_mean = _jk["sum"] / _jk["n"]
        jerk_max = _jk["max"]
        print(f"    Achieved arm jerk:    mean {jerk_mean:.0f}  max {jerk_max:.0f} rad/s^3")
        raw_env._arm_jerk_acc = {"sum": 0.0, "n": 0, "max": 0.0}  # drain so each window is fresh
    return (
        _dec,
        active_mask,
        chunk_eps,
        early_pct,
        f0_complete_pct,
        f0_eps,
        fingertip_errs,
        jerk_max,
        jerk_mean,
        mean_body_errs,
        mean_ep_len,
        mean_frame_progress,
        n_active,
        rsi_complete_pct,
        rsi_eps,
        timeout_pct,
        wrist_errs,
    )
