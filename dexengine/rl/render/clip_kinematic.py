"""Kinematic reference replay of one clip for the per-clip video render (DEXENGINE_RENDER_KINEMATIC): the reference
reset each step and the frames of every camera (split from render.clip_modes)."""

from __future__ import annotations

import math
import os

import imageio
import torch

from dexengine.core.desk import DESK_TOP_SURFACE_Z
from dexengine.rl.render.clip_steps import _grab_frame
from dexengine.rl.swallow import note_once


def kinematic_replay_clip(raw_env, args_cli, traj_mgr, device, out_dir, fps, cid, tag, cname, cams, overlay, manifest):
    """Replay the reference of clip ``cid`` kinematically and save one video per view."""
    ego_cam, front_cam, side_cam, sidef_cam = cams
    _ft_marker, _ft_local, _ft_root_q, _ft_root_pos, _ft_slowdown = overlay
    from dexengine.rl.tasks.mdp.terms import reset_arm_to_trajectory as _kr_arm
    from dexengine.rl.tasks.mdp.terms import reset_hands_to_trajectory as _kr_hands
    from dexengine.rl.tasks.mdp.terms import reset_object_to_trajectory as _kr_obj

    _kr_eids = torch.tensor([0], device=device)
    _kr_T = int(traj_mgr.clip_lengths[0].item())
    # The slowdown factor holds each reference frame for `slowdown` video frames, so the replay is frame-aligned
    # with a policy rollout at the same slowdown.
    _kr_slow = float(getattr(traj_mgr, "_slowdown_factor", 1.0) or 1.0)
    _kr_nsteps = int(math.ceil(_kr_T * _kr_slow))
    # RENDER_STILL_FRAMES="0,mid[,last|,N]": render only these reference frames and save them as PNGs instead of
    # mp4s (camera-angle checks).
    _kr_frames = None
    _sf_env = os.environ.get("RENDER_STILL_FRAMES", "")
    if _sf_env:
        _kr_frames = sorted(
            {
                _kr_T // 2 if s.strip() == "mid" else _kr_T - 1 if s.strip() == "last" else min(int(s), _kr_T - 1)
                for s in _sf_env.split(",")
                if s.strip()
            }
        )
        _kr_nsteps = len(_kr_frames)
    # Axis markers were already hidden after the settle step (_apply_render_markers, kinematic path); the fingertip
    # overlay follows the same flag.
    _kr_markers = bool(int(getattr(args_cli, "render_markers", 1)))
    ego_fs, front_fs, side_fs, sidef_fs = [], [], [], []
    _kinematic_replay_steps(
        raw_env,
        traj_mgr,
        out_dir,
        cid,
        tag,
        _ft_local,
        _ft_marker,
        _ft_root_pos,
        _ft_root_q,
        _kr_T,
        _kr_arm,
        _kr_eids,
        _kr_frames,
        _kr_hands,
        _kr_markers,
        _kr_nsteps,
        _kr_obj,
        _kr_slow,
        ego_cam,
        ego_fs,
        front_cam,
        front_fs,
        side_cam,
        side_fs,
        sidef_cam,
        sidef_fs,
    )
    if ego_fs and _kr_frames is None:
        imageio.mimsave(os.path.join(out_dir, f"{cid:02d}_{tag}_ego_kinematic.mp4"), ego_fs, fps=fps)
    if front_fs and _kr_frames is None:
        imageio.mimsave(os.path.join(out_dir, f"{cid:02d}_{tag}_front_kinematic.mp4"), front_fs, fps=fps)
    if side_fs and _kr_frames is None:
        imageio.mimsave(os.path.join(out_dir, f"{cid:02d}_{tag}_side_kinematic.mp4"), side_fs, fps=fps)
    if sidef_fs and _kr_frames is None:
        imageio.mimsave(os.path.join(out_dir, f"{cid:02d}_{tag}_side_front_kinematic.mp4"), sidef_fs, fps=fps)
    print(
        f"  [Render] clip {cid:02d} '{tag}': KINEMATIC replay saved "
        f"({_kr_T} ref frames x slowdown {_kr_slow} = {_kr_nsteps} video frames, "
        f"front {len(front_fs)}, ego {len(ego_fs)}, side {len(side_fs)})"
    )
    manifest[cid] = {"name": cname, "mode": "kinematic_replay", "frames": _kr_T, "has_front": front_cam is not None}


def _kinematic_replay_steps(
    raw_env,
    traj_mgr,
    out_dir,
    cid,
    tag,
    _ft_local,
    _ft_marker,
    _ft_root_pos,
    _ft_root_q,
    _kr_T,
    _kr_arm,
    _kr_eids,
    _kr_frames,
    _kr_hands,
    _kr_markers,
    _kr_nsteps,
    _kr_obj,
    _kr_slow,
    ego_cam,
    ego_fs,
    front_cam,
    front_fs,
    side_cam,
    side_fs,
    sidef_cam,
    sidef_fs,
):
    """Step the kinematic reference replay of one clip and collect the frames."""
    for _k in range(_kr_nsteps):
        _f = _kr_frames[_k] if _kr_frames is not None else min(int(_k / _kr_slow), _kr_T - 1)
        traj_mgr.start_frames[_kr_eids] = _f
        _kr_hands(raw_env, _kr_eids)
        _kr_arm(raw_env, _kr_eids)
        _bimanual_scene_whose_right(raw_env, traj_mgr, _kr_eids, _kr_obj)
        # MANO reference fingertips at reference frame _f.
        if _ft_marker is not None and _kr_markers:
            import isaaclab.utils.math as _mu_ft

            _ef = max(0, min(_f, _ft_local.shape[0] - 1))
            _ftw = _ft_root_pos[0] + _mu_ft.quat_apply(
                _ft_root_q.unsqueeze(0).expand(_ft_local.shape[1], -1), _ft_local[_ef]
            )
            _ft_marker.visualize(translations=_ftw)
        _render_replay_frame(
            raw_env,
            out_dir,
            cid,
            tag,
            _kr_frames,
            ego_cam,
            ego_fs,
            front_cam,
            front_fs,
            side_cam,
            side_fs,
            sidef_cam,
            sidef_fs,
            _f,
        )


def _bimanual_scene_whose_right(raw_env, traj_mgr, _kr_eids, _kr_obj):
    """A bimanual scene whose right hand is parked has no object_right."""
    if "object_right" in raw_env.scene.keys():  # noqa: SIM118  — InteractiveScene has no __contains__
        _kr_obj(raw_env, _kr_eids)
    # reset_object_to_trajectory defaults to object_right / side="right"; without this call the left object is left
    # to physics while the hands follow the reference.
    if "object_left" in raw_env.scene.keys():  # noqa: SIM118  — InteractiveScene has no __contains__
        try:
            from isaaclab.managers import SceneEntityCfg as _SEC_kr

            _kr_obj(raw_env, _kr_eids, object_cfg=_SEC_kr("object_left"), side="left")
        except Exception as _e_kr:  # noqa: BLE001 — a failed teleport is printed; the replay goes on
            print(f"  [KR] left object teleport failed: {_e_kr}")
    # The wrist arrow follows the reference wrist (start_frames = _f): robot.data.body_pos_w is stale after a
    # teleport without a physics step.
    _wm = getattr(raw_env, "_marker_wrist_current_frame", None)
    if _wm is not None:
        try:
            _wp = traj_mgr.get_wrist_pos_w(None)[0:1] + raw_env.scene.env_origins[0:1]
            _wq = traj_mgr.get_wrist_quat_w(None)[0:1]
            if float(getattr(traj_mgr.cfg, "traj_offset_z", 0.0) or 0.0) < 0.1:
                _wp = _wp.clone()
                _wp[..., 2] += float(getattr(traj_mgr, "_desk_height", DESK_TOP_SURFACE_Z))
            _wm.visualize(_wp, _wq)
            _wt = getattr(raw_env, "_marker_wrist_target_frame", None)
            if _wt is not None:
                _wt.visualize(_wp, _wq)
        except (AttributeError, RuntimeError, TypeError, ValueError) as _exc:
            note_once("CLIP_KINEMATIC wrist marker", _exc)


def _render_replay_frame(
    raw_env,
    out_dir,
    cid,
    tag,
    _kr_frames,
    ego_cam,
    ego_fs,
    front_cam,
    front_fs,
    side_cam,
    side_fs,
    sidef_cam,
    sidef_fs,
    _f,
):
    """Render this replay step and collect the camera frames."""
    raw_env.sim.render()
    if _kr_frames is not None:
        # Still frames: render the same pose N more times so the RTX temporal accumulation (denoiser) converges
        # before the capture. Consecutive video frames accumulate on their own.
        for _ in range(int(os.environ.get("RENDER_STILL_WARMUP", "16"))):
            raw_env.sim.render()
    _refresh_cameras(ego_cam, front_cam, side_cam, sidef_cam)
    ef = _grab_frame(ego_cam)
    if ef is not None:
        ego_fs.append(ef)
        if _kr_frames is not None:
            imageio.imwrite(os.path.join(out_dir, f"{cid:02d}_{tag}_ego_f{_f:04d}.png"), ef)
    if front_cam is not None:
        ff = _grab_frame(front_cam)
        if ff is not None:
            front_fs.append(ff)
            if _kr_frames is not None:
                imageio.imwrite(os.path.join(out_dir, f"{cid:02d}_{tag}_front_f{_f:04d}.png"), ff)
    if side_cam is not None:
        sf = _grab_frame(side_cam)
        if sf is not None:
            side_fs.append(sf)
            if _kr_frames is not None:
                imageio.imwrite(os.path.join(out_dir, f"{cid:02d}_{tag}_side_f{_f:04d}.png"), sf)
    if sidef_cam is not None:
        sff = _grab_frame(sidef_cam)
        if sff is not None:
            sidef_fs.append(sff)
            if _kr_frames is not None:
                imageio.imwrite(os.path.join(out_dir, f"{cid:02d}_{tag}_side_front_f{_f:04d}.png"), sff)


def _refresh_cameras(ego_cam, front_cam, side_cam, sidef_cam):
    """Force a render refresh of every camera in use."""
    for _c in (ego_cam, front_cam, side_cam, sidef_cam):
        if _c is not None:
            try:
                _c.update(0.0, force_recompute=True)
            except TypeError:
                _c.update(0.0)
