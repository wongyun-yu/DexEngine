"""Per-clip policy video render.

For every clip of the session the policy is rolled out from the scene cameras (ego head_camera, front, side,
side_front) and one of three modes saves the videos: best-ranked success, first success + first failure (legacy), or a
kinematic replay of the reference (``--kinematic_replay``). A ``manifest.txt`` lists every rendered clip.

Callers: ``dexengine.rl.eval.run`` and the frozen-render branch of ``dexengine.rl.train.runner``. The record session
replaces ``_run_per_clip_video_render`` on this module through a post-import hook, so the function stays defined here.
Recording and replaying stored trajectories is not here (``dexengine.rl.eval.traj_record`` records,
``dexengine.rl.render.replay`` replays).
"""

from __future__ import annotations

import os

import imageio
import torch

# Video encoding quality. imageio's ``quality`` is ffmpeg ``-qscale:v`` (default 5, low bitrate); quality=9 gave
# 82-94 Mbps at 1920x1920@50fps. x264 CRF keeps the picture at a small fraction of that size (measured on a real render:
# CRF 18 -> 6.8 % of the size, PSNR-Y 47.1 dB, visually lossless), so CRF (default and preset owned by
# dexengine.record.video_sink) is the default. CRF and qscale are mutually exclusive in ffmpeg, so ``quality`` is
# dropped whenever CRF is set. DEXENGINE_VIDEO_CRF="" falls back to imageio's default qscale. Every mp4 write of the
# render goes through imageio.mimsave, so the wrapper is installed once on the module.
_VIDEO_QUALITY = os.environ.get("DEXENGINE_VIDEO_QUALITY", "").strip()
from dexengine.record.video_sink import crf as _vs_crf
from dexengine.record.video_sink import preset as _vs_preset

_VIDEO_CRF = _vs_crf()
_VIDEO_PRESET = _vs_preset()
if _VIDEO_QUALITY or _VIDEO_CRF:
    _mimsave_orig = imageio.mimsave

    def _mimsave_hq(uri, ims, **kwargs):
        if str(uri).lower().endswith(".mp4"):
            if _VIDEO_CRF:
                kwargs.pop("quality", None)  # qscale and CRF are mutually exclusive
                kwargs.setdefault("codec", "libx264")
                kwargs.setdefault("pixelformat", "yuv420p")
                op = list(kwargs.get("output_params") or [])
                if "-crf" not in op:
                    op += ["-crf", str(_VIDEO_CRF), "-preset", _VIDEO_PRESET]
                kwargs["output_params"] = op
            elif _VIDEO_QUALITY:
                kwargs.setdefault("quality", float(_VIDEO_QUALITY))
        return _mimsave_orig(uri, ims, **kwargs)

    imageio.mimsave = _mimsave_hq
    if _VIDEO_CRF:
        print(
            f"[render] DEXENGINE_VIDEO_CRF={_VIDEO_CRF} preset={_VIDEO_PRESET} (x264 CRF; quality/qscale are ignored)"
        )
    else:
        print(f"[render] DEXENGINE_VIDEO_QUALITY={_VIDEO_QUALITY} (imageio 0~10, default 5)")

from dexengine.rl.render.clip_audit import _contact_audit
from dexengine.rl.render.clip_overlay import _mano_fingertip_overlay
from dexengine.rl.render.clip_rollout import (
    _make_reward_readers,
    _resolve_rewards,
)
from dexengine.rl.render.clip_video_loop import _render_clip_videos


def _run_per_clip_video_render(
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
    warmup_attempts=5,
    select_best_success=False,
    rollouts_per_clip=None,
    rank_metric="obj_pos_mean",
    args_cli=None,
    create_env=None,
):
    """Render per-clip videos of the policy from the scene cameras (ego head_camera, front, side, side_front).

    Uses the same policy_fn / traj_mgr as the standard eval; frames are read from ``cam.data.output["rgb"]`` after
    every env step (requires --enable_cameras). The outcome criterion matches the balanced eval:
    failure = terminated AND NOT truncated.

    Per clip, the first ``warmup_attempts`` rollouts are discarded (sim settling), then
      - ``select_best_success``: ``rollouts_per_clip`` rollouts; the best-ranked success is saved as
        ``{cid:02d}_{tag}_{view}_best.mp4``, or the best-ranked failure as ``..._failure_best.mp4`` if none succeeded;
      - otherwise (legacy): the first success and the first failure among ``max_attempts`` rollouts are saved as
        ``{cid:02d}_{tag}_{ego|front}_{success|failure}.mp4`` (no failure video at 100 % success);
      - ``--kinematic_replay``: the reference is replayed kinematically instead (``..._{view}_kinematic.mp4``).
    ``manifest.txt`` gets the per-clip warmup and outcome counts. The env is closed at the end, and SystemExit is
    raised when the manifest does not cover every selected clip.

    Side effect: ``traj_mgr.reset`` is replaced to pin every reset to the current clip and restored on exit.
    ``create_env`` is unused; it stays for the callers' signature.
    """
    _check_single_env_and_settle(
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
    )


def _check_single_env_and_settle(
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
):
    """The video render needs one env; settle PhysX before the first clip."""
    n_envs = wrapped.num_envs
    if n_envs != 1:
        raise RuntimeError(
            f"--render_per_clip_videos requires --num_envs 1 (got {n_envs}). "
            "head_camera is only retained at num_envs=1 in create_env."
        )
    _front_only, ego_cam, front_cam, original_reset, overlay, side_cam, sidef_cam = _check_multi_object_spawn(
        traj_mgr, num_clips, raw_env, out_dir, max_attempts, warmup_attempts, args_cli
    )
    device, pinned_clip = _render_device_and_pin(traj_mgr, raw_env)

    _settle_physx_before_render(
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
        _front_only,
        device,
        ego_cam,
        front_cam,
        n_envs,
        original_reset,
        overlay,
        pinned_clip,
        side_cam,
        sidef_cam,
    )


def _render_device_and_pin(traj_mgr, raw_env):
    """Device and pinned-clip state of the render."""
    pinned_clip = [0]
    device = raw_env.device

    def _pinned_reset(env_ids):
        traj_mgr.clip_ids[env_ids] = pinned_clip[0]
        traj_mgr.clip_lengths[env_ids] = traj_mgr._loader.get_clip_lengths(traj_mgr.clip_ids[env_ids])
        traj_mgr.start_frames[env_ids] = 0

    traj_mgr.reset = _pinned_reset
    return device, pinned_clip


def _settle_physx_before_render(
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
    _front_only,
    device,
    ego_cam,
    front_cam,
    n_envs,
    original_reset,
    overlay,
    pinned_clip,
    side_cam,
    sidef_cam,
):
    """PhysX settling (reset, one zero-action step, reset) as in the balanced eval: without it the first rollouts of"""

    # PhysX settling (reset, one zero-action step, reset) as in the balanced eval: without it the first rollouts of a
    # cold num_envs=1 start fail spuriously on some clips.
    def _settle_physx():
        wrapped._reset_once = True
        wrapped.reset()
        zero_act = torch.zeros(n_envs, wrapped.action_space.shape[0], device=device)
        wrapped.step(zero_act)
        wrapped._reset_once = True
        wrapped.reset()

    audit = _contact_audit(raw_env, args_cli)
    reward_mgr, have_obj_pos, have_obj_rot, have_fingertips, rank_metric, max_ep_len_s = _resolve_rewards(
        raw_env, select_best_success, rank_metric
    )
    _read_cum, _extract_logged_cum = _make_reward_readers(reward_mgr, max_ep_len_s)
    cams = (ego_cam, front_cam, side_cam, sidef_cam)
    _render_clip_videos(
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
    )


def _check_multi_object_spawn(traj_mgr, num_clips, raw_env, out_dir, max_attempts, warmup_attempts, args_cli):
    """Multi_object spawns asset[i % n] in env i (MultiAssetSpawnerCfg(random_choice=False)): with one env every clip
    would be rendered with the first clip's object, so refuse instead of writing wrong videos.
    """
    if getattr(args_cli, "multi_object", 0) and num_clips > 1:
        raise RuntimeError(
            f"[Render] --multi_object with {num_clips} clips in ONE env: every clip would be rendered with "
            f"asset[0] (env_i -> asset[i % n]). Render one clip per session (the launcher's RENDER_ONLY=replay "
            f"does this; VIDEOS_ONLY_CLIPS / 1-clip SEQ_DIR for policy renders)."
        )

    # RENDER_FRONT_ONLY=1 skips the ego (head) camera. Every captured view is buffered for the whole rollout, so
    # dropping one halves the peak memory of long sweeps.
    _front_only = os.environ.get("RENDER_FRONT_ONLY", "0") == "1"

    ego_cam = None
    if not _front_only:
        try:
            ego_cam = raw_env.scene["head_camera"]
        except KeyError:
            raise RuntimeError(
                "head_camera not found in scene. Pass --enable_cameras and ensure "
                "the env cfg keeps head_camera at num_envs=1."
            ) from None
    side_cam = None
    try:
        side_cam = raw_env.scene["side_camera"]
        print(f"  [Render] side_camera output keys: {list(side_cam.data.output.keys())}")
    except (AttributeError, KeyError):  # InteractiveScene.__getitem__: KeyError when the camera is absent
        print("  [Render] side_camera not in scene; front/ego only")
    sidef_cam = None
    try:
        sidef_cam = raw_env.scene["side_front_camera"]
    except KeyError:  # InteractiveScene.__getitem__: entity not in the scene
        print("  [Render] side_front_camera not in scene")
    front_cam = None
    try:
        front_cam = raw_env.scene["front_camera"]
        print(f"  [Render] front_camera output keys: {list(front_cam.data.output.keys())}")
    except KeyError:
        print("  [Render] WARNING: front_camera not in scene; ego-only render")
    # Canonical background and lighting (DEXENGINE_AUG_*), applied once on the finished scene. scene_look raises when
    # a requested knob is not applied; the error is not caught here.
    import omni.usd as _ou_aug

    from dexengine.rl.render import scene_look as _m_aug

    _m_aug.apply_from_env(_ou_aug.get_context().get_stage())
    if _front_only:
        if front_cam is None:
            raise RuntimeError("RENDER_FRONT_ONLY=1 but front_camera is not in the scene — nothing would be rendered.")
        print("  [Render] FRONT-ONLY: ego (head_camera) capture disabled")
    else:
        print(f"  [Render] ego head_camera output keys: {list(ego_cam.data.output.keys())}")
    print(f"  [Render] warmup_attempts={warmup_attempts}, eval_attempts={max_attempts}")

    os.makedirs(out_dir, exist_ok=True)

    overlay = _mano_fingertip_overlay(raw_env, args_cli, traj_mgr)

    # Pin every reset to the current clip the way the balanced eval does: set the trajectory manager fields
    # (clip_ids, clip_lengths, start_frames) directly instead of calling the original reset, which would sample a
    # random clip and consume RNG.
    original_reset = traj_mgr.reset
    return _front_only, ego_cam, front_cam, original_reset, overlay, side_cam, sidef_cam
