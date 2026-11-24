"""Per-clip video loop: which clips are rendered (--videos_only_clips, the manifest, skip counts), the render of each
and the manifest check.
"""

from __future__ import annotations

import os
import sys

from dexengine.rl.render.clip_kinematic import kinematic_replay_clip
from dexengine.rl.render.clip_modes import _best_success_clip, _legacy_clip, _write_manifest
from dexengine.rl.render.clip_rollout import _make_run_one_rollout, _make_score_metrics

_DEBUG_MARKERS = (
    "_marker_wrist_current_frame",
    "_marker_wrist_target_frame",
    "_marker_obj_current_frame",
    "_marker_obj_target_frame",
    "_marker_wrist_current_frame_left",
    "_marker_wrist_target_frame_left",
    "_marker_obj_current_frame_left",
    "_marker_obj_target_frame_left",
)


def _apply_render_markers(raw_env, args_cli) -> None:
    """--render_markers 0 hides the eval env's RGB axis markers in every path (policy and kinematic). They are
    created lazily on the env's first step, so this runs after each clip's settle step; a hidden marker's visualize()
    returns early, so it stays hidden."""
    if bool(int(getattr(args_cli, "render_markers", 1))):
        return
    hidden = [m for m in _DEBUG_MARKERS if getattr(raw_env, m, None) is not None]
    for m in hidden:
        getattr(raw_env, m).set_visibility(False)
    print(f"[RENDER_MARKERS] hidden {len(hidden)}")


def _render_static(raw_env, args_cli, clip: str) -> None:
    """Swap in the clip's visual-only static extras at a clip boundary (rules, poses and the [RENDER_STATIC] line are
    owned by dexengine.record.scene.static_extras)."""
    from dexengine.record.scene import static_extras as _rs

    _rs.apply(raw_env, _rs.tree_of(getattr(args_cli, "sequence_dir", None), clip), clip)


def _render_clip_videos(
    wrapped,
    policy_fn,
    traj_mgr,
    num_clips,
    clip_names,
    raw_env,
    checkpoint_path,
    out_dir,
    max_attempts,
    max_episode_steps,
    fps,
    warmup_attempts,
    select_best_success,
    rollouts_per_clip,
    rank_metric,
    args_cli,
    _extract_logged_cum,
    _front_only,
    _read_cum,
    _settle_physx,
    audit,
    cams,
    device,
    front_cam,
    have_fingertips,
    have_obj_pos,
    have_obj_rot,
    original_reset,
    overlay,
    pinned_clip,
):
    """Per-clip eval rates, the clip filter and the per-clip video loop."""
    _run_one_rollout = _make_run_one_rollout(
        wrapped,
        raw_env,
        traj_mgr,
        policy_fn,
        max_episode_steps,
        cams,
        overlay,
        audit,
        (have_obj_pos, have_obj_rot, have_fingertips, _read_cum, _extract_logged_cum),
        _front_only,
    )
    _render_and_check_manifest(
        wrapped,
        traj_mgr,
        num_clips,
        clip_names,
        raw_env,
        checkpoint_path,
        out_dir,
        max_attempts,
        max_episode_steps,
        fps,
        warmup_attempts,
        select_best_success,
        rollouts_per_clip,
        rank_metric,
        args_cli,
        _settle_physx,
        cams,
        device,
        front_cam,
        original_reset,
        overlay,
        pinned_clip,
        _run_one_rollout,
    )


def _render_and_check_manifest(
    wrapped,
    traj_mgr,
    num_clips,
    clip_names,
    raw_env,
    checkpoint_path,
    out_dir,
    max_attempts,
    max_episode_steps,
    fps,
    warmup_attempts,
    select_best_success,
    rollouts_per_clip,
    rank_metric,
    args_cli,
    _settle_physx,
    cams,
    device,
    front_cam,
    original_reset,
    overlay,
    pinned_clip,
    _run_one_rollout,
):
    """Render the selected clips and check that the manifest holds every one."""
    _score_metrics = _make_score_metrics(rank_metric)

    manifest = {}
    rollouts_n = int(rollouts_per_clip) if (select_best_success and rollouts_per_clip) else max_attempts
    _perclip_rates = _clip_eval_rates_json(args_cli)
    _want, mpath = _select_video_clips(
        traj_mgr,
        num_clips,
        clip_names,
        raw_env,
        checkpoint_path,
        out_dir,
        max_attempts,
        max_episode_steps,
        fps,
        warmup_attempts,
        select_best_success,
        rank_metric,
        args_cli,
        _settle_physx,
        cams,
        device,
        front_cam,
        original_reset,
        overlay,
        pinned_clip,
        _perclip_rates,
        _run_one_rollout,
        _score_metrics,
        manifest,
        rollouts_n,
    )
    sys.stdout.flush()
    wrapped.close()
    print(f"  [Render] DONE manifest {len(manifest)}/{_want} clips · env closed · {mpath}", flush=True)
    if len(manifest) != _want:
        raise SystemExit(f"[Render] manifest {len(manifest)}/{_want} — some clips are missing (count check)")


def _clip_eval_rates_json(args_cli):
    """Per-clip eval rates (JSON {short_seq_id: rate in 0..1}) from the eval pass set the rollout budget of the best-
    success mode: a clip with rate > 0 must yield a success video, a 0 % clip stops after one batch.
    """
    _perclip_rates = None
    _pc_path = getattr(args_cli, "videos_perclip_eval", None)
    if _pc_path and os.path.isfile(_pc_path):
        import json as _pc_json

        with open(_pc_path) as _pc_f:
            _perclip_rates = {str(k): float(v) for k, v in _pc_json.load(_pc_f).items()}
        print(f"[Render] per-clip eval rates: {len(_perclip_rates)} clips from {_pc_path}")
    return _perclip_rates


def _select_video_clips(
    traj_mgr,
    num_clips,
    clip_names,
    raw_env,
    checkpoint_path,
    out_dir,
    max_attempts,
    max_episode_steps,
    fps,
    warmup_attempts,
    select_best_success,
    rank_metric,
    args_cli,
    _settle_physx,
    cams,
    device,
    front_cam,
    original_reset,
    overlay,
    pinned_clip,
    _perclip_rates,
    _run_one_rollout,
    _score_metrics,
    manifest,
    rollouts_n,
):
    """Videos_only_clips: render only the clips whose name contains one of the tokens."""
    _n_skip, mpath = _filter_video_clips(
        traj_mgr,
        num_clips,
        clip_names,
        raw_env,
        checkpoint_path,
        out_dir,
        max_attempts,
        max_episode_steps,
        fps,
        warmup_attempts,
        select_best_success,
        rank_metric,
        args_cli,
        _settle_physx,
        cams,
        device,
        front_cam,
        original_reset,
        overlay,
        pinned_clip,
        _perclip_rates,
        _run_one_rollout,
        _score_metrics,
        manifest,
        rollouts_n,
    )
    # Close the env before returning. Returning with it open makes the caller's simulation_app.close() spin forever:
    # omni.replicator stop -> timeline STOP -> Isaac Lab SimulationContext's STOP subscription keeps rendering.
    # env.close() (sim.clear_instance) removes that subscription; the caller's second close is ignored by Isaac Lab
    # (_is_closed), and the caller sets the process exit code.
    _want = num_clips - _n_skip
    return _want, mpath


def _filter_video_clips(
    traj_mgr,
    num_clips,
    clip_names,
    raw_env,
    checkpoint_path,
    out_dir,
    max_attempts,
    max_episode_steps,
    fps,
    warmup_attempts,
    select_best_success,
    rank_metric,
    args_cli,
    _settle_physx,
    cams,
    device,
    front_cam,
    original_reset,
    overlay,
    pinned_clip,
    _perclip_rates,
    _run_one_rollout,
    _score_metrics,
    manifest,
    rollouts_n,
):
    """Apply --videos_only_clips and the existing manifest."""
    _only = [t.strip() for t in str(getattr(args_cli, "videos_only_clips", "") or "").split(",") if t.strip()]
    if _only:
        print(f"[Render] only_clips={_only}")
    _n_skip = 0  # skipped clips; the manifest must cover all the others
    try:
        _n_skip = _count_skipped_clips(
            traj_mgr,
            num_clips,
            clip_names,
            raw_env,
            out_dir,
            max_attempts,
            fps,
            warmup_attempts,
            select_best_success,
            rank_metric,
            args_cli,
            _settle_physx,
            cams,
            device,
            front_cam,
            overlay,
            pinned_clip,
            _perclip_rates,
            _run_one_rollout,
            _score_metrics,
            manifest,
            rollouts_n,
            _only,
            _n_skip,
        )
    finally:
        traj_mgr.reset = original_reset

    mpath = _write_manifest(
        out_dir,
        checkpoint_path,
        warmup_attempts,
        rollouts_n,
        select_best_success,
        rank_metric,
        max_episode_steps,
        fps,
        manifest,
    )
    return _n_skip, mpath


def _count_skipped_clips(
    traj_mgr,
    num_clips,
    clip_names,
    raw_env,
    out_dir,
    max_attempts,
    fps,
    warmup_attempts,
    select_best_success,
    rank_metric,
    args_cli,
    _settle_physx,
    cams,
    device,
    front_cam,
    overlay,
    pinned_clip,
    _perclip_rates,
    _run_one_rollout,
    _score_metrics,
    manifest,
    rollouts_n,
    _only,
    _n_skip,
):
    """Count the clips this render skips."""
    for cid in range(num_clips):
        cname = clip_names[cid] if cid < len(clip_names) else f"clip_{cid}"
        if _only and not any(t in cname for t in _only):
            print(f"  [Render] clip {cid:02d} '{cname}': skipped (--videos_only_clips)")
            _n_skip += 1
            continue
        tag = cname.replace("scene_03__", "").replace("__", "_")[:40]
        pinned_clip[0] = cid

        _settle_physx()
        _apply_render_markers(raw_env, args_cli)
        # Static extras that the training / judge scene does not hold are shown as visual-only shapes, per clip.
        _render_static(raw_env, args_cli, cname)

        if getattr(args_cli, "kinematic_replay", False):
            kinematic_replay_clip(
                raw_env, args_cli, traj_mgr, device, out_dir, fps, cid, tag, cname, cams, overlay, manifest
            )
            continue

        for _ in range(warmup_attempts):  # warmup: outcomes discarded
            _ = _run_one_rollout()

        if select_best_success:
            _best_success_clip(
                out_dir,
                fps,
                cid,
                tag,
                cname,
                args_cli,
                rollouts_n,
                _perclip_rates,
                _run_one_rollout,
                _score_metrics,
                rank_metric,
                warmup_attempts,
                front_cam,
                manifest,
            )
            continue

        _legacy_clip(
            out_dir, fps, cid, tag, cname, max_attempts, _run_one_rollout, warmup_attempts, front_cam, manifest
        )
    return _n_skip
