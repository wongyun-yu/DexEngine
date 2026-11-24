"""One policy rollout of the per-clip render, pinned to the current clip, and the reward bookkeeping that ranks it.

Ranking reads the reward manager's per-term episode sums. The cumulative value is snapshotted before every step; on the
terminating step the manager's reset has already zeroed the sums, so the final value is recovered from
``info["log"]["Episode_Reward/<term>"]``, which stores cum / max_episode_length_s.
"""

from __future__ import annotations

from dexengine.rl.render.clip_audit import _audit_report
from dexengine.rl.render.clip_steps import _FT_TERMS, _rollout_steps


def _resolve_rewards(raw_env, select_best_success, rank_metric):
    """Reward terms available for ranking and the rank metric after the hand-only fallback.

    Raises when ``select_best_success`` needs a term the env does not have.
    Returns (reward_mgr, have_obj_pos, have_obj_rot, have_fingertips, rank_metric, max_ep_len_s).
    """
    reward_mgr = getattr(raw_env, "reward_manager", None)
    have_obj_pos = reward_mgr is not None and "obj_pos" in getattr(reward_mgr, "_episode_sums", {})
    have_obj_rot = reward_mgr is not None and "obj_rot" in getattr(reward_mgr, "_episode_sums", {})
    have_fingertips = reward_mgr is not None and all(t in getattr(reward_mgr, "_episode_sums", {}) for t in _FT_TERMS)
    if select_best_success:
        if rank_metric in ("obj_pos_mean", "obj_pos_cum", "obj_pos_obj_rot_mean"):
            if not have_obj_pos:
                if have_fingertips:
                    print(
                        f"[Render] rank_metric='{rank_metric}' but obj_pos reward not found "
                        f"(hand-only env). Auto-fallback to 'fingertip_quality_mean'."
                    )
                    rank_metric = "fingertip_quality_mean"
                else:
                    raise RuntimeError(
                        f"[Render] --videos_select_best_success=1 with rank_metric='{rank_metric}' "
                        f"needs an 'obj_pos' reward term OR all 5 fingertip terms "
                        f"(hand-only fallback). Neither found — check env cfg."
                    )
        if rank_metric == "obj_pos_obj_rot_mean" and not have_obj_rot:
            raise RuntimeError("[Render] rank_metric=obj_pos_obj_rot_mean needs 'obj_rot' reward term, not found.")
        if rank_metric == "fingertip_quality_mean" and not have_fingertips:
            raise RuntimeError(
                "[Render] rank_metric=fingertip_quality_mean needs thumb_tip, index_tip, "
                "middle_tip, ring_tip, pinky_tip reward terms, not all found."
            )
    max_ep_len_s = float(getattr(raw_env, "max_episode_length_s", 0.0) or 0.0)
    return reward_mgr, have_obj_pos, have_obj_rot, have_fingertips, rank_metric, max_ep_len_s


def _make_reward_readers(reward_mgr, max_ep_len_s):
    """(read_cum, extract_logged_cum): the live episode sum of a term for env 0, and the un-normalized episode sum
    logged on a terminating step (None when unavailable)."""

    def _read_cum(term_name):
        if reward_mgr is None:
            return None
        sums = getattr(reward_mgr, "_episode_sums", None)
        if sums is None or term_name not in sums:
            return None
        return sums[term_name][0].item()

    def _extract_logged_cum(info, term_name):
        """Pull Episode_Reward/<term> from step info dict if present. Returns
        cumulative reward (un-normalized), or None if not available."""
        if info is None or max_ep_len_s <= 0:
            return None
        log = info.get("log") if isinstance(info, dict) else None
        if log is None:
            return None
        key = f"Episode_Reward/{term_name}"
        val = log.get(key)
        if val is None:
            return None
        if hasattr(val, "item"):
            v = val.item()
        else:
            v = float(val)
        return v * max_ep_len_s  # invert reward_manager's /max_episode_length_s

    return _read_cum, _extract_logged_cum


def _make_run_one_rollout(
    wrapped, raw_env, traj_mgr, policy_fn, max_episode_steps, cams, overlay, audit, rewards, _front_only
):
    """Bind one pinned rollout to its env, cameras, fingertip overlay, contact audit and reward readers."""

    def _run_one_rollout_bound():
        return _run_one_rollout(
            wrapped, raw_env, traj_mgr, policy_fn, max_episode_steps, cams, overlay, audit, rewards, _front_only
        )

    return _run_one_rollout_bound


def _run_one_rollout(
    wrapped, raw_env, traj_mgr, policy_fn, max_episode_steps, cams, overlay, audit, rewards, _front_only
):
    """Run a single episode on the pinned clip (traj_mgr.reset is pinned by the caller). Return (is_failure,
    ego_frames, front_frames, metrics) where metrics is a dict with keys
    'obj_pos_cum', 'obj_pos_mean', 'obj_rot_cum', 'obj_rot_mean',
    'fingertip_quality_cum', 'fingertip_quality_mean' (hand-only fallback),
    'n_steps' (zeros when reward terms unavailable)."""
    ego_cam, front_cam, side_cam, sidef_cam = cams
    _ft_marker, _ft_local, _ft_root_q, _ft_root_pos, _ft_slowdown = overlay
    _AUDIT, _aud_rows, _ar_rich, _AR_NAMES, _ar_log = audit
    have_obj_pos, have_obj_rot, have_fingertips, _read_cum, _extract_logged_cum = rewards
    wrapped._reset_once = True
    obs, _ = wrapped.reset()
    ego_fs, front_fs, is_failure, metrics, side_fs, sidef_fs = _collect_side_frames(
        wrapped,
        raw_env,
        traj_mgr,
        policy_fn,
        max_episode_steps,
        _front_only,
        _AR_NAMES,
        _AUDIT,
        _ar_log,
        _ar_rich,
        _aud_rows,
        _extract_logged_cum,
        _ft_local,
        _ft_marker,
        _ft_root_pos,
        _ft_root_q,
        _ft_slowdown,
        _read_cum,
        ego_cam,
        front_cam,
        have_fingertips,
        have_obj_pos,
        have_obj_rot,
        obs,
        side_cam,
        sidef_cam,
    )
    metrics["_side_fs"] = side_fs
    metrics["_sidef_fs"] = sidef_fs
    return is_failure, ego_fs, front_fs, metrics


def _collect_side_frames(
    wrapped,
    raw_env,
    traj_mgr,
    policy_fn,
    max_episode_steps,
    _front_only,
    _AR_NAMES,
    _AUDIT,
    _ar_log,
    _ar_rich,
    _aud_rows,
    _extract_logged_cum,
    _ft_local,
    _ft_marker,
    _ft_root_pos,
    _ft_root_q,
    _ft_slowdown,
    _read_cum,
    ego_cam,
    front_cam,
    have_fingertips,
    have_obj_pos,
    have_obj_rot,
    obs,
    side_cam,
    sidef_cam,
):
    """side / side_front frames travel in metrics (_side_fs, _sidef_fs) so the 4-tuple return stays as it is."""
    ego_fs, front_fs = [], []
    side_fs, sidef_fs = [], []
    if _AUDIT:
        _aud_rows.clear()
    final_ft_cum, final_pos_cum, final_rot_cum, term_b, trunc_b = _run_episode_from_zero(
        wrapped,
        raw_env,
        traj_mgr,
        policy_fn,
        max_episode_steps,
        _front_only,
        _AUDIT,
        _ar_log,
        _extract_logged_cum,
        _ft_local,
        _ft_marker,
        _ft_root_pos,
        _ft_root_q,
        _ft_slowdown,
        _read_cum,
        ego_cam,
        front_cam,
        have_fingertips,
        have_obj_pos,
        have_obj_rot,
        obs,
        side_cam,
        sidef_cam,
        ego_fs,
        front_fs,
        side_fs,
        sidef_fs,
    )
    if _AUDIT and _aud_rows:
        _audit_report(_aud_rows, _ar_rich, _AR_NAMES, term_b, trunc_b)
    is_failure = term_b and not trunc_b
    # The step count comes from the view actually captured: under RENDER_FRONT_ONLY the ego list is empty and
    # n_steps = 1 would turn the *_mean metrics into raw sums.
    _n_frames = len(front_fs if _front_only else ego_fs)
    metrics = _rollout_metrics(_n_frames, final_ft_cum, final_pos_cum, final_rot_cum)
    return ego_fs, front_fs, is_failure, metrics, side_fs, sidef_fs


def _run_episode_from_zero(
    wrapped,
    raw_env,
    traj_mgr,
    policy_fn,
    max_episode_steps,
    _front_only,
    _AUDIT,
    _ar_log,
    _extract_logged_cum,
    _ft_local,
    _ft_marker,
    _ft_root_pos,
    _ft_root_q,
    _ft_slowdown,
    _read_cum,
    ego_cam,
    front_cam,
    have_fingertips,
    have_obj_pos,
    have_obj_rot,
    obs,
    side_cam,
    sidef_cam,
    ego_fs,
    front_fs,
    side_fs,
    sidef_fs,
):
    """Zeroed cumulative terms, then the episode rollout; returns the final sums and the end flags."""
    term_b = trunc_b = False
    # Episode sums are snapshotted before each step (covering steps 0..k-1). On the terminating step the env
    # auto-resets and zeroes them, so the final sum is read from info["log"].
    pre_step_pos = 0.0
    pre_step_rot = 0.0
    final_pos_cum = 0.0
    final_rot_cum = 0.0
    pre_step_ft = dict.fromkeys(_FT_TERMS, 0.0)
    final_ft_cum = dict.fromkeys(_FT_TERMS, 0.0)
    info = None
    final_pos_cum, final_rot_cum, term_b, trunc_b = _rollout_steps(
        wrapped,
        raw_env,
        traj_mgr,
        policy_fn,
        max_episode_steps,
        _front_only,
        _AUDIT,
        _ar_log,
        _extract_logged_cum,
        _ft_local,
        _ft_marker,
        _ft_root_pos,
        _ft_root_q,
        _ft_slowdown,
        _read_cum,
        ego_cam,
        ego_fs,
        final_ft_cum,
        final_pos_cum,
        final_rot_cum,
        front_cam,
        front_fs,
        have_fingertips,
        have_obj_pos,
        have_obj_rot,
        info,
        obs,
        pre_step_ft,
        pre_step_pos,
        pre_step_rot,
        side_cam,
        side_fs,
        sidef_cam,
        sidef_fs,
        term_b,
        trunc_b,
    )
    return final_ft_cum, final_pos_cum, final_rot_cum, term_b, trunc_b


def _rollout_metrics(_n_frames, final_ft_cum, final_pos_cum, final_rot_cum):
    """Per-rollout metrics from the cumulative tracking terms."""
    n_steps = max(_n_frames, 1)
    ft_cum_total = sum(final_ft_cum[t] for t in _FT_TERMS)
    metrics = {
        "obj_pos_cum": final_pos_cum,
        "obj_pos_mean": final_pos_cum / n_steps,
        "obj_rot_cum": final_rot_cum,
        "obj_rot_mean": final_rot_cum / n_steps,
        "fingertip_quality_cum": ft_cum_total,
        "fingertip_quality_mean": ft_cum_total / n_steps,
        "n_steps": _n_frames,
    }
    return metrics


def _make_score_metrics(rank_metric):
    """Score of a rollout's metrics under ``rank_metric`` (an unknown name scores by obj_pos_mean)."""

    def _score_metrics(m):
        if rank_metric == "obj_pos_mean":
            return m["obj_pos_mean"]
        if rank_metric == "obj_pos_cum":
            return m["obj_pos_cum"]
        if rank_metric == "obj_pos_obj_rot_mean":
            return m["obj_pos_mean"] + m["obj_rot_mean"]
        if rank_metric == "fingertip_quality_mean":
            return m["fingertip_quality_mean"]
        if rank_metric == "fingertip_quality_cum":
            return m["fingertip_quality_cum"]
        return m["obj_pos_mean"]

    return _score_metrics
