"""Snapshot report of snapshot_metrics: policy std, the per-term reward breakdown, the coupling check and the W&B log
(split from judge.snapshot_metrics, which keeps the tracking / termination tables and the entry)."""

from __future__ import annotations

import torch

from dexengine.rl.judge.snapshot_wandb import log_snapshot_to_wandb
from dexengine.rl.swallow import note_once


def detect_coupling_pairs(joint_names):
    """The hand-coupling pairs and their naming label (one table today)."""
    return COUPLING_PAIRS_DATASET, "dataset (rig MJCF)"


def get_policy_std(agent_ref):
    """Extract current policy std from the skrl PPO learner."""
    try:
        for name, param in agent_ref.policy.named_parameters():
            if "log_std" in name:
                return param.detach(), torch.exp(param.detach())
    except (AttributeError, RuntimeError) as _exc:
        note_once("SNAPSHOT policy std", _exc)
    return None, None


def print_policy_std_and_rewards(
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
):
    """Policy std and the per-term reward breakdown (episode returns)."""
    print(f"      common_step_counter: {step_count}")

    return _policy_std(
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
        timeout_pct,
        wrist_errs,
    )


def _policy_std(
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
    timeout_pct,
    wrist_errs,
):
    """Current policy std from the skrl PPO learner (wrist and finger parts)."""
    log_std, std = get_policy_std(agent_ref) if agent_ref else (None, None)
    finger_std, wrist_resid_std = _dims_wrist_pose_residual(std)

    rm = _term_reward_breakdown_episode(raw_env)

    return _body_tracking_table(
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
        timeout_pct,
        wrist_errs,
        finger_std,
        rm,
        wrist_resid_std,
    )


def _dims_wrist_pose_residual(std):
    """Dims 0-5 are the WRIST POSE RESIDUAL (pos 3 + rot 3), not a force: the pose residual goes through DLS IK into arm
    joint position TARGETS. Nothing in the live action path applies force.
    """
    wrist_resid_std = finger_std = None
    if std is not None:
        wrist_resid_std = std[:6]
        finger_std = std[6:]
        print("\n    Policy std (lower = more converged, target < 0.1):")
        print(
            f"      Wrist pose residual (0-5):  mean={wrist_resid_std.mean():.4f}  "
            f"range=[{wrist_resid_std.min():.4f}, {wrist_resid_std.max():.4f}]"
        )
        print(
            f"      Finger (dims 6-11): mean={finger_std.mean():.4f}  "
            f"range=[{finger_std.min():.4f}, {finger_std.max():.4f}]"
        )
        print(f"      All dims:           mean={std.mean():.4f}  [{', '.join(f'{v:.3f}' for v in std.tolist())}]")
    return finger_std, wrist_resid_std


def _term_reward_breakdown_episode(raw_env):
    """Per-term reward breakdown: the episode RETURN (Σ) = env.extras["log"]["Episode_Reward/<term>"] (×
    max_episode_length_s to undo RewardManager's /max_ep_len_s normalization), the per-term reward summed over the
    episode and averaged over the most recently completed episodes. It FALLS when episodes shorten, so it is not
    fooled by early termination (a per-step mean is: every term is near-max in the static opening frames a truncated
    episode samples).
    """
    rm = raw_env.reward_manager
    term_names = rm.active_terms

    _max_ep_len_s = float(getattr(raw_env, "max_episode_length_s", 0.0) or 0.0)
    _ep_log = getattr(raw_env, "extras", {}).get("log", {}) if hasattr(raw_env, "extras") else {}
    ep_ret = {}
    for name in term_names:
        v = _ep_log.get(f"Episode_Reward/{name}")
        if v is not None and _max_ep_len_s > 0:
            ep_ret[name] = (v.item() if hasattr(v, "item") else float(v)) * _max_ep_len_s
    if ep_ret:
        ret_strs = [f"{n}={ep_ret[n]:.3f}" for n in term_names if n in ep_ret]
        total_ret = sum(ep_ret.values())
        rhalf = (len(ret_strs) + 1) // 2
        print(f"    Rewards (episode return Σ) total={total_ret:.3f}:  " + "  ".join(ret_strs[:rhalf]))
        if len(ret_strs) > rhalf:
            print("              " + "  ".join(ret_strs[rhalf:]))
    else:
        print("    Rewards (episode return Σ): n/a (no episode completed in extras log yet)")
    return rm


def _body_tracking_table(
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
    timeout_pct,
    wrist_errs,
    finger_std,
    rm,
    wrist_resid_std,
):
    """Body tracking table."""
    _print_body_tracking_rows(fingertip_errs, wrist_errs)
    print(
        f"    {'All bodies':12s}  {mean_body_errs.mean():.4f}  {mean_body_errs.min():.4f}  "
        f"{mean_body_errs.max():.4f}  {mean_body_errs.median():.4f}"
    )

    # Coupling constraint verification (filtered to active envs only)
    joint_pos = hand_asset.data.joint_pos[active_mask] if n_active > 0 else hand_asset.data.joint_pos
    _print_coupling_check(side, hand_asset, joint_pos)

    record = log_snapshot_to_wandb(
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
    return record


def _print_body_tracking_rows(fingertip_errs, wrist_errs):
    """Print the per-body tracking error table."""
    print()
    print(f"    {'Body':12s}  {'Avg':>8s}  {'Min':>8s}  {'Max':>8s}  {'Median':>8s}")
    print(f"    {'-' * 12}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
    print(
        f"    {'Wrist':12s}  {wrist_errs.mean():.4f}  {wrist_errs.min():.4f}  "
        f"{wrist_errs.max():.4f}  {wrist_errs.median():.4f}"
    )
    for name in ["thumb", "index", "middle", "ring", "pinky"]:
        e = fingertip_errs[name]
        print(f"    {name:12s}  {e.mean():.4f}  {e.min():.4f}  {e.max():.4f}  {e.median():.4f}")


def _print_coupling_check(side, hand_asset, joint_pos):
    """Print the coupled-joint check of the hand."""
    joint_names = hand_asset.joint_names
    coupling_pairs, coupling_label = detect_coupling_pairs(joint_names)
    print(f"\n    Coupling (PhysxMimicJointAPI, {coupling_label} naming):")
    max_coupling_err = 0.0
    for f_suffix, d_suffix, ratio in coupling_pairs:
        f_name = f"{side}_{f_suffix}"
        d_name = f"{side}_{d_suffix}"
        if f_name not in joint_names or d_name not in joint_names:
            # Mixed-hand robot: this side has no such coupling (e.g. fully-actuated
            # SharpA right — detect_coupling_pairs matched the RH56F1 left).
            continue
        fi = joint_names.index(f_name)
        di = joint_names.index(d_name)
        err = torch.abs(joint_pos[:, fi] - ratio * joint_pos[:, di])
        max_err = err.max().item()
        mean_err = err.mean().item()
        max_coupling_err = max(max_coupling_err, max_err)
        status = "OK" if max_err < 0.01 else "VIOLATION"
        print(
            f"      [{status}] {f_suffix:30s} = {ratio:.4f} x {d_suffix:30s}  "
            f"err: max={max_err:.6f} mean={mean_err:.6f} rad"
        )
    coupling_ok = max_coupling_err < 0.01
    print(f"    Coupling {'PASS' if coupling_ok else 'FAIL'}: max error = {max_coupling_err:.6f} rad")


# Joint-coupling table of the hand (follower joint, driver joint, ratio): follower = ratio × driver.
COUPLING_PAIRS_DATASET = [
    ("thumb_intermediate_joint", "thumb_proximal_pitch_joint", 1.2953),
    ("thumb_distal_joint", "thumb_intermediate_joint", 0.8962),
    ("index_intermediate_joint", "index_proximal_joint", 1.1545),
    ("middle_intermediate_joint", "middle_proximal_joint", 1.1545),
    ("ring_intermediate_joint", "ring_proximal_joint", 1.1545),
    ("pinky_intermediate_joint", "pinky_proximal_joint", 1.1545),
]
