"""POST-KIT: the per-clip video cameras that `create_env` adds for --render_per_clip_videos.

Four views share one resolution (DEXENGINE_CAMERA_RES, default 1920, the same override as the head camera, so the
side-by-side composer gets square frames of equal size): ego (the env cfg's head camera, rendered every step),
front, side and side_front. DEXENGINE_RENDER_VIEWS=<comma list> keeps only the named views; a dropped camera is not
created at all, which saves its GPU buffer and the per-step GPU→CPU frame copy.
Camera offsets use the OpenGL convention (the camera looks along its -Z, +Y up); quaternions are (w, x, y, z).
"""

from __future__ import annotations

import os


def add_video_cameras(cfg, num_envs: int, _trr) -> None:
    """Render the head camera every step and add the front / side / side_front cameras (``_trr`` = render.replay)."""
    # Head camera every sim step: the default period 0.1 s (10 fps) repeats each frame 5x at 50 Hz.
    if hasattr(cfg.scene, "head_camera") and cfg.scene.head_camera is not None:
        cfg.scene.head_camera.update_period = 0.0
        print("[create_env] head_camera.update_period overridden to 0.0 (per-step) for video render")
    _CameraCfg, _fcres, _lookat_quat, _sim_utils = _add_front_and_side_cameras(cfg)

    print("[create_env] side_camera added")
    # Side-front camera: a 3/4 view from front-right, above.
    cfg.scene.side_front_camera = _CameraCfg(
        prim_path="{ENV_REGEX_NS}/SideFrontCamera",
        update_period=0.0,
        height=_fcres,
        width=_fcres,
        data_types=["rgb"],
        spawn=_sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 6.0),
        ),
        offset=_CameraCfg.OffsetCfg(
            pos=(0.85, -0.85, 1.75),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="opengl",
        ),
    )
    cfg.scene.side_front_camera.offset.rot = _lookat_quat(0.85, -0.85, 1.75, -0.10, 0.0, 0.95)
    print("[create_env] side_front_camera added")

    # View selection (unset = all four cameras).
    _rv = os.environ.get("DEXENGINE_RENDER_VIEWS", "").strip()
    if _rv:
        _want = {v.strip() for v in _rv.split(",") if v.strip()}
        _known = {"ego", "front", "side", "side_front"}
        if not _want <= _known:
            raise SystemExit(
                f"[create_env] DEXENGINE_RENDER_VIEWS has unknown views {sorted(_want - _known)} "
                f"— choose from {sorted(_known)}"
            )
        if not _want:
            raise SystemExit("[create_env] DEXENGINE_RENDER_VIEWS is empty — name at least one view")
        _drop = []
        for _v, _attr in _trr.VIEW_CAMERAS:  # view ↔ camera table (owner: render.replay)
            if _v not in _want and getattr(cfg.scene, _attr, None) is not None:
                setattr(cfg.scene, _attr, None)
                _drop.append(_attr)
        print(
            f"[create_env] DEXENGINE_RENDER_VIEWS={_rv} → using {sorted(_want)} · dropped {_drop} "
            f"(saves {len(_drop)} camera(s) and {len(_drop)} frame copies per step)"
        )
    if _trr.enabled():
        _trr.configure_scene(cfg, num_envs)


def _add_front_and_side_cameras(cfg):
    """Front and side cameras; returns the camera cfg class, resolution and look-at helper the other views use."""
    # Front camera: third person from +X, close and high at (0.35, 0, 1.45), looking at hand height (0, 0, 1.0)
    # (pitch -52°); focal 12 -> FOV 82°. Robot base at ROBOT_DEFAULT_POS=(-0.5, 0.25, 0.35), desk top z=0.76.
    import isaaclab.sim as _sim_utils
    from isaaclab.sensors import CameraCfg as _CameraCfg

    from dexengine.rl.tasks import switches as _switches  # stdlib only

    _fcres = int(_switches.text("DEXENGINE_CAMERA_RES"))
    cfg.scene.front_camera = _CameraCfg(
        prim_path="{ENV_REGEX_NS}/FrontCamera",
        update_period=0.0,
        height=_fcres,
        width=_fcres,
        data_types=["rgb"],
        spawn=_sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
        ),
        offset=_CameraCfg.OffsetCfg(
            pos=(0.35, 0.0, 1.45),
            rot=(0.6688, 0.2295, 0.2295, 0.6688),  # look-at (0, 0, 1.0), pitch -52.1°
            convention="opengl",
        ),
    )
    print(f"[create_env] front_camera added ({_fcres}x{_fcres} FOV 82°, close+high, looking at hand height from +X)")
    # Side camera: a pure side view from the robot's right (-Y, where most of the activity is), level, looking +Y
    # at the scene centre.
    cfg.scene.side_camera = _CameraCfg(
        prim_path="{ENV_REGEX_NS}/SideCamera",
        update_period=0.0,
        height=_fcres,
        width=_fcres,
        data_types=["rgb"],
        spawn=_sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
        ),
        offset=_CameraCfg.OffsetCfg(
            pos=(-0.10, -1.35, 0.85),
            rot=(0.7071068, 0.7071068, 0.0, 0.0),
            convention="opengl",
        ),
    )

    def _lookat_quat(px, py, pz, tx, ty, tz):
        """(w, x, y, z) of an OpenGL camera at (px, py, pz) looking at (tx, ty, tz), +Z world up."""
        import numpy as _np

        f = _np.array([tx - px, ty - py, tz - pz], float)
        f /= _np.linalg.norm(f)
        up = _np.array([0.0, 0.0, 1.0])
        if abs(float(f @ up)) > 0.985:  # near-vertical view: use +X as up
            up = _np.array([1.0, 0.0, 0.0])
        X = _np.cross(f, up)
        X /= _np.linalg.norm(X)
        Z = -f
        Y = _np.cross(Z, X)
        R = _np.stack([X, Y, Z], 1)
        w = float(_np.sqrt(max(1e-12, 1 + _np.trace(R))) / 2)
        return (
            w,
            float((R[2, 1] - R[1, 2]) / (4 * w)),
            float((R[0, 2] - R[2, 0]) / (4 * w)),
            float((R[1, 0] - R[0, 1]) / (4 * w)),
        )

    return _CameraCfg, _fcres, _lookat_quat, _sim_utils
