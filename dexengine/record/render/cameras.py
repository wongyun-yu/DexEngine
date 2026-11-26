"""Camera setup of the public-data record: **record only, off by default** (it started from a reference render for
user checks).

The canonical render defaults (head camera, front/side views; owners shared_config_arm and dexengine/rl/cli.py) are not
touched. The canonical path never imports this module. Switched on by dexengine/record/session/entry.py (env
DEXENGINE_COLLECT_CAMERAS=1). Robot USD files are not written: the head ZED Mini housing is put into the render USD by
the render body builder (dexengine/robots/usd/head_camera.py).

Setup (owner of the specs, pitch rule, exo and depth values = this file):
  1. ego stereo = ZED Mini · HD720 (1280x720 per eye) · 30 fps · intrinsics from the real calibration (ZEDM_CALIB:
     rectified K, zero distortion and baseline of the physical RB-Y1 head ZED Mini). The public nominal values
     (ZEDM_SPEC, HD1080 fx 1478) stay for comparison only.
     Mount = the calibrated ego_zed_m_camera chain of the physical robot (link_head_2 -> ego_zed_m_camera -> left and
     right optical), tilt (mount rpy) as calibrated: the same pose as the real robot. Right eye = the chain's right
     optical (baseline = |R - L| checked against ZEDM_CALIB).
     Principal point: the Omniverse pinhole draws no aperture offset (principal point = image centre), so the image is
     rendered larger (render_*) and cropped so the calibrated principal point lands in place (OpenCV pixel-centre
     convention) (crop_ego · residual < 0.5 px).
     Owner of the mount values and the housing mesh = head_camera.py (the render USD place and both eyes come from the
     same values).
  2. automatic head pitch = head_pitch_for_desk(): from camera height, desk front edge and depth, the desk fills the
     vertical field of view.
  3. exo front = the canonical front camera moved back along its optical axis by Δ (dolly): exo_dolly() gives the
     smallest Δ that keeps the upper-body keypoints inside the margin.
  4. ego depth = left-eye distance_to_image_plane -> 16-bit mm raw + a viewing colour map (depth_to_png16,
     depth_colormap).
"""

from __future__ import annotations

import math
import os

from dexengine.core.errors import InputMissingError
from dexengine.robots.usd import head_camera as HC  # mount owner (render body builder); no pxr in its header

# numpy is imported inside functions only: session modules read this module before AppLauncher
# ([PRE_APP] guard: Kit loads its own numpy first)
ENV_SWITCH = "DEXENGINE_COLLECT_CAMERAS"  # default off: add_collect_cameras changes the scene only for "1"
#: exo (outer) camera render and store switch, **single owner** (E5 kinematic: head stereo only). Unset, empty or 1 =
#: on (canonical, byte-identical) · 0 = off (no exo render output; videos egoL and egoR only, and so the store keys) ·
#: anything else dies. Readers: render.episode.VIDEOS · render.clip_session.OFF_KEYS · render.rig.capture ·
#: store.store.VIDEO_KEYS · record.episodes.plan (off puts videos into the plan body, so the render cache fingerprint
#: differs). Marker [COLLECT_EXO].
EXO_ENV = "DEXENGINE_COLLECT_EXO"
ALL_VIDEOS = ("egoL", "egoR", "exo")


def exo_on() -> bool:
    v = os.environ.get(EXO_ENV, "")
    if v not in ("", "0", "1"):
        raise ValueError(f"{EXO_ENV}={v!r}: unset, empty, 1 (on) or 0 (off) only")
    return v != "0"


def videos() -> tuple[str, ...]:
    """Video names rendered and stored (render.episode output suffixes); exo off gives the two head eyes only."""
    return ALL_VIDEOS if exo_on() else tuple(v for v in ALL_VIDEOS if v != "exo")


def enabled() -> bool:
    return os.environ.get(ENV_SWITCH, "0") == "1"


# ── 1. ZED Mini public specs: value and source side by side (the source is the value's evidence) ──────────────
_SL_STORE = "https://www.stereolabs.com/store/products/zed-mini"
_SL_FOCAL = (
    "https://support.stereolabs.com/hc/en-us/articles/360007395634-What-is-the-camera-focal-length-and-field-of-view"
)
_CAL = "head ZED Mini of the physical robot, simple-easyhec calibration"  # citation of the calibration source
_SL_MODES = "https://docs.stereolabs.com/docs/video/camera-controls"
# Single owner of the record head camera's intrinsics, resolution and baseline (HD720 + the real calibration). Values =
# that calibration as it is.
ZEDM_CALIB = {
    "mode": ("HD720", "head camera = HD720 + real calibration (the old HD1080 nominal ZEDM_SPEC is for comparison)"),
    "width_px": (1280, f"{_SL_MODES} — HD720 1280x720 per eye"),
    "height_px": (720, f"{_SL_MODES} — HD720"),
    "fps": (30, f"{_SL_STORE} — HD720 60/30/15 fps, 30 chosen"),
    "fx_px": (732.0785522460938, f"{_CAL} — re-measured, rectified K"),
    "fy_px": (732.0785522460938, f"{_CAL}"),
    "cx_px": (633.6272583007812, f"{_CAL} — OpenCV/ROS convention (pixel centre = integer coordinate)"),
    "cy_px": (356.186279296875, f"{_CAL}"),
    "distortion": ((0.0, 0.0, 0.0, 0.0, 0.0), f"{_CAL} — rectified CameraInfo, zero distortion"),
    "baseline_m": (0.0628986191, f"{_CAL} — right optical offset of the same solution"),
}
# Stereolabs public nominal values, for comparison (the record camera uses ZEDM_CALIB; only lens focal length and
# depth range are read here)
ZEDM_SPEC = {
    "mode": ("HD1080", f"{_SL_STORE} — '1080p at 30 or 15 FPS' (comparison only, unused)"),
    "width_px": (1920, "https://docs.stereolabs.com/docs/video/camera-controls — HD1080 = 3840x1080 side-by-side"),
    "height_px": (1080, "same table — HD1080 1920x1080 per eye"),
    "fps": (30, f"{_SL_STORE} — 1080p 30/15 fps, 30 chosen"),
    "fx_px": (1478.0, f"{_SL_FOCAL} — ZED Mini table HD1080 focal 1478 px (nominal; each unit differs)"),
    "fov_deg_HV": ((66.0, 40.0), f"{_SL_FOCAL} — ZED Mini HD1080 H 66° · V 40°"),
    "fx_hd720_px": (736.0, f"{_SL_FOCAL} — HD720 736 px (comparison)"),
    "baseline_mm": (63.0, f"{_SL_STORE} — 'Baseline 63 mm (2.4'')'"),
    "focal_length_mm": (2.8, f"{_SL_STORE} — 'Lens 2.8mm ƒ/2.0'"),
    "fov_max_deg_HVD": ((102.0, 57.0, 118.0), f"{_SL_STORE} — '102° (H) x 57° (V) x 118° (D) max.' (uncalibrated max)"),
    "depth_range_m": ((0.1, 15.0), f"{_SL_STORE} — 'Depth Range 0.1m to 15m'"),
}

# Owner of the mount, the ZED Mini housing and its mesh = head_camera.py of the render body builder (the very place in
# the render USD); read only here. Both eyes = that mount + the calibrated left optical chain · right eye = left eye +
# baseline (ZEDM_SPEC below).
ZEDM_MOUNT = HC.ZEDM_MOUNT


BASELINE_TOL_M = 1e-9
#: principal point of the Omniverse pinhole (OpenCV pixel-centre convention) = (N - 1)/2 + OMNI_PP_SHIFT_PX. 0 = the
#: geometric image centre (a wrong value skews the projection to one side by as much; no run measures it).
OMNI_PP_SHIFT_PX = 0.0


def zedm_rig() -> dict:
    """Poses of the ZED Mini housing, left eye and right eye (the chain's right optical) relative to link_head_2; owner
    head_camera.zedm_rig. A chain baseline |R - L| different from the ZEDM_CALIB baseline dies."""
    rig = HC.zedm_rig()
    b = ZEDM_CALIB["baseline_m"][0]
    if abs(rig["baseline_m"] - b) > BASELINE_TOL_M:
        raise SystemExit(f"[COLLECT_CAM] calibrated chain baseline {rig['baseline_m']:.10f} != ZEDM_CALIB {b:.10f}")
    return rig


def render_span(n: int, c: float) -> tuple[int, int, float]:
    """Image n pixels, target principal point c (OpenCV) -> (render pixels N, crop start k, residual px): the cropped
    principal point (N - 1)/2 + OMNI_PP_SHIFT_PX - k closest to c; of the two smallest valid N (0 <= k, k + n <= N)
    the one with the smaller residual."""
    cands = []
    for N in range(n, n + 256):
        k = round((N - 1) / 2 + OMNI_PP_SHIFT_PX - c)
        if k >= 0 and k + n <= N:
            cands.append((abs((N - 1) / 2 + OMNI_PP_SHIFT_PX - k - c), N, k))
            if len(cands) == 2:
                break
    if not cands:
        raise InputMissingError(f"[COLLECT_CAM] no render size puts principal point {c} inside image {n}")
    e, N, k = min(cands)
    return N, k, e


def zed_intrinsics() -> dict:
    """Pinhole of one HD720 eye (ZEDM_CALIB) + Isaac render size and crop window. fx != fy dies (Omniverse has square
    pixels)."""
    c = {k: v[0] for k, v in ZEDM_CALIB.items()}
    w, h, fx, fy, cx, cy = c["width_px"], c["height_px"], c["fx_px"], c["fy_px"], c["cx_px"], c["cy_px"]
    if fx != fy or any(c["distortion"]):
        raise SystemExit(
            f"[COLLECT_CAM] fx {fx} != fy {fy} or distortion {c['distortion']}: the Omniverse pinhole cannot draw it"
        )
    W, x0, ex = render_span(w, cx)
    H, y0, ey = render_span(h, cy)
    f_mm = ZEDM_SPEC["focal_length_mm"][0]
    return {
        "width": w,
        "height": h,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "distortion": list(c["distortion"]),
        "render_width": W,
        "render_height": H,
        "crop_x0": x0,
        "crop_y0": y0,
        "cx_rendered": (W - 1) / 2 + OMNI_PP_SHIFT_PX - x0,
        "cy_rendered": (H - 1) / 2 + OMNI_PP_SHIFT_PX - y0,
        "principal_point_residual_px": (ex, ey),
        "hfov_deg": math.degrees(math.atan(cx / fx) + math.atan((w - 1 - cx) / fx)),
        "vfov_deg": math.degrees(math.atan(cy / fy) + math.atan((h - 1 - cy) / fy)),
        # Isaac pinhole (USD): aperture = render pixels x (focal / fx) -> the rendered fx is exactly the
        #   calibration
        "focal_length": f_mm,
        "horizontal_aperture": W * f_mm / fx,
        "vertical_aperture": H * f_mm / fy,
        "baseline_m": c["baseline_m"],
        "fps": c["fps"],
        "mode": c["mode"],
    }


def crop_ego(img, k: dict | None = None):
    """Rendered ego image (H_r x W_r x …) -> the h x w window at the calibrated principal point (the crop window of
    zed_intrinsics)."""
    k = zed_intrinsics() if k is None else k
    if tuple(img.shape[:2]) != (k["render_height"], k["render_width"]):
        raise SystemExit(
            f"[COLLECT_CAM] ego image {img.shape[:2]} != render size {(k['render_height'], k['render_width'])}"
        )
    return img[k["crop_y0"] : k["crop_y0"] + k["height"], k["crop_x0"] : k["crop_x0"] + k["width"]]


# ── 2. automatic head pitch ───────────────────────────────────────────────────────────────────────────────
def head_pitch_for_desk(
    cam_x: float, cam_z: float, desk_near_x: float, desk_far_x: float, desk_top_z: float, vfov_deg: float
) -> dict:
    """Downward pitch (deg) that makes the desk fill the vertical field of view. The robot looks along +x (desk front
    edge = robot side = desk_near_x).

    α_near = atan(h / (x_near - x_cam)) · α_far = atan(h / (x_far - x_cam)) (h = camera - top height, down is +).
    When span = α_near - α_far is at most the vertical field V the whole desk is in the image -> pitch = the middle
    (α_near + α_far) / 2. When span > V the desk is larger than the image -> the front edge goes to the image bottom
    (pitch = α_near - V/2: the near side where the hands work first).
    In one formula: pitch = max((α_near + α_far)/2, α_near - V/2).
    A front edge behind the camera (x_near <= x_cam) gives an angle beyond 90°; atan2 keeps it continuous.
    """
    h = cam_z - desk_top_z
    if h <= 0:
        raise ValueError(f"camera below the desk top (h={h:.3f} m): the pitch rule does not apply")
    a_near = math.degrees(math.atan2(h, desk_near_x - cam_x))
    a_far = math.degrees(math.atan2(h, desk_far_x - cam_x))
    span = a_near - a_far
    pitch = max(0.5 * (a_near + a_far), a_near - 0.5 * vfov_deg)
    return {
        "pitch_deg": pitch,
        "alpha_near_deg": a_near,
        "alpha_far_deg": a_far,
        "span_deg": span,
        "vfov_deg": vfov_deg,
        "h_m": h,
        "d_near_m": desk_near_x - cam_x,
        "d_far_m": desk_far_x - cam_x,
        "rule": "center" if span <= vfov_deg else "near-edge-at-bottom",
    }


# ── 3. exo front dolly ────────────────────────────────────────────────────────────────────────────────
EXO_MARGIN_FRAC = 0.02  # image edge margin (2 % of a side): 'tight'
EXO_PAD_HEAD_M = 0.10  # head link origin -> housing and head top margin (sphere radius)
EXO_PAD_BODY_M = 0.08  # torso and shoulder link origin -> surface margin
# upper body = static torso, head, shoulders
EXO_UPPER_BODY_KEYS = ("torso", "head", "camera", "arm_0", "arm_1", "arm_2")


def exo_dolly(points_cam, pads, fx: float, fy: float, w: int, h: int, margin_frac: float = EXO_MARGIN_FRAC) -> float:
    """Smallest Δ (m) along the optical axis backwards that keeps every point (OpenGL camera coordinates: x right, y up,
    -z forward) inside the margin.

    Moving along the optical axis keeps x and y and changes the depth d -> d + Δ -> |u - cx| = fx(|x| + r)/(d + Δ)
    <= w/2 - m ⇔ Δ >= fx(|x| + r)/(w/2 - m) - d (the same vertically). Max over the points; below 0 -> 0 (all
    visible)."""
    import numpy as np

    mx, my = margin_frac * w, margin_frac * h
    x, y, d = points_cam[:, 0], points_cam[:, 1], -points_cam[:, 2]
    need_x = fx * (np.abs(x) + pads) / (w / 2.0 - mx) - d
    need_y = fy * (np.abs(y) + pads) / (h / 2.0 - my) - d
    return float(max(0.0, np.max(np.maximum(need_x, need_y))))


# ── 4. depth store ────────────────────────────────────────────────────────────────────────────────────
DEPTH_VIZ_RANGE_M = (0.1, 2.0)  # viewing colour-map range (ZED Mini minimum 0.1 m to past the desk); raw png unaffected


def depth_to_png16(depth_m):
    """Metres -> uint16 mm raw. inf, NaN and <= 0 = 0 (invalid). Above 65.535 m saturates at 65535."""
    import numpy as np

    d = np.asarray(depth_m, dtype=np.float64).squeeze()
    bad = ~np.isfinite(d) | (d <= 0)
    mm = np.clip(np.round(d * 1000.0), 0, 65535)
    mm[bad] = 0
    return mm.astype(np.uint16)


def depth_colormap(depth_m, rng=DEPTH_VIZ_RANGE_M):
    """Viewing RGB (uint8): turbo, invalid = black."""
    import numpy as np
    from matplotlib import colormaps

    d = np.asarray(depth_m, dtype=np.float64).squeeze()
    bad = ~np.isfinite(d) | (d <= 0)
    t = np.clip((np.where(bad, rng[1], d) - rng[0]) / (rng[1] - rng[0]), 0.0, 1.0)
    rgb = colormaps["turbo"](t, bytes=True)[..., :3]  # = (cmap(t)[..., :3] * 255).astype(uint8): lut scaled first
    rgb[bad] = 0
    return rgb


# ── add the cameras to the scene cfg (off by default) ───────────────────────────────────────────────────

#: Front camera of a drawing scene: third person from +X, close and high at (0.35, 0, 1.45), looking at hand height
#: (0, 0, 1.0) (pitch -52.1°); focal 12 -> FOV 82°. Robot base at ROBOT_DEFAULT_POS=(-0.5, 0.25, 0.35), desk top z=0.76.
#: The exo camera is a copy of it (add_collect_cameras) and pose_check moves it per view.
FRONT_CAMERA = {
    "prim": "{ENV_REGEX_NS}/FrontCamera",
    "focal_length": 12.0,
    "focus_distance": 400.0,
    "horizontal_aperture": 20.955,
    "clipping_range": (0.05, 5.0),
    "pos": (0.35, 0.0, 1.45),
    "rot": (0.6688, 0.2295, 0.2295, 0.6688),  # look-at (0, 0, 1.0), OpenGL convention (w, x, y, z)
}


def add_scene_cameras(cfg) -> str:
    """Cameras of a scene that draws (trainer flag --record_session or --pose_render_out, one env): the env cfg's head
    camera rendered every sim step, plus FRONT_CAMERA at the DEXENGINE_CAMERA_RES square resolution. Training and
    judging scenes have neither (dexengine.rl.app.training_scene). -> marker line."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    from dexengine.rl.tasks import switches  # stdlib only

    head = getattr(cfg.scene, "head_camera", None)
    if head is not None:  # the default period 0.1 s (10 fps) would repeat each frame 5x at 50 Hz
        head.update_period = 0.0
    res, f = int(switches.text("DEXENGINE_CAMERA_RES")), FRONT_CAMERA
    cfg.scene.front_camera = CameraCfg(
        prim_path=f["prim"],
        update_period=0.0,
        height=res,
        width=res,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=f["focal_length"],
            focus_distance=f["focus_distance"],
            horizontal_aperture=f["horizontal_aperture"],
            clipping_range=f["clipping_range"],
        ),
        offset=CameraCfg.OffsetCfg(pos=f["pos"], rot=f["rot"], convention="opengl"),
    )
    return (
        f"[SCENE_CAM] head_camera every step ({'present' if head is not None else 'absent'}) · front_camera "
        f"{res}x{res} focal {f['focal_length']} mm at {f['pos']}"
    )


def add_collect_cameras(cfg) -> str:
    """Add collect_ego_left/right (ZED Mini HD720 calibrated K, calibrated mount, drawn at render size and cropped) and
    collect_exo_front (a copy of the canonical front camera; pose set at run time by render.rig.Rig.calibrate) to the
    scene cfg. The ZED Mini housing mesh is already in the render USD (render body builder).

    The canonical camera cfgs (head_camera, front_camera) are only read."""
    import copy

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    if not enabled():
        return "[COLLECT_CAM] off (DEXENGINE_COLLECT_CAMERAS!=1): scene unchanged"
    head = getattr(cfg.scene, "head_camera", None)
    front = getattr(cfg.scene, "front_camera", None)
    if head is None or front is None:
        raise InputMissingError(
            "[COLLECT_CAM] canonical head_camera and front_camera are not in the scene: start with "
            "--record_session --enable_cameras --num_envs 1 (add_scene_cameras)"
        )
    k = zed_intrinsics()
    rig = zedm_rig()
    segs = head.prim_path.split("/")
    if rig["parent"] not in segs:
        raise InputMissingError(f"[COLLECT_CAM] head camera path {head.prim_path} has no {rig['parent']}")
    link = "/".join(segs[: segs.index(rig["parent"]) + 1])
    for name, leaf, pos, rot, types in (
        ("collect_ego_left", "ZedMiniLeft", rig["left_pos"], rig["left_quat"], ["rgb", "distance_to_image_plane"]),
        ("collect_ego_right", "ZedMiniRight", rig["right_pos"], rig["right_quat"], ["rgb"]),
    ):
        setattr(
            cfg.scene,
            name,
            CameraCfg(
                prim_path=f"{link}/{leaf}",
                update_period=0.0,
                height=k["render_height"],
                width=k["render_width"],
                data_types=types,
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=k["focal_length"],
                    horizontal_aperture=k["horizontal_aperture"],
                    vertical_aperture=k["vertical_aperture"],
                    focus_distance=head.spawn.focus_distance,
                    clipping_range=(head.spawn.clipping_range[0], ZEDM_SPEC["depth_range_m"][0][1]),
                ),
                offset=CameraCfg.OffsetCfg(pos=tuple(pos), rot=tuple(rot), convention="ros"),
            ),
        )

    exo = copy.deepcopy(front)
    exo.prim_path = "{ENV_REGEX_NS}/CollectExoFront"
    cfg.scene.collect_exo_front = exo
    return (
        f"[COLLECT_CAM] added to the scene: ego L/R = {link}/ZedMiniLeft·ZedMiniRight {k['mode']} "
        f"{k['width']}x{k['height']}@{k['fps']} fx=fy={k['fx']:.3f} c=({k['cx']:.3f},{k['cy']:.3f}) · render "
        f"{k['render_width']}x{k['render_height']} · crop ({k['crop_x0']},{k['crop_y0']}) · "
        f"principal point residual "
        f"({k['principal_point_residual_px'][0]:.3f},{k['principal_point_residual_px'][1]:.3f}) px "
        f"(HFOV {k['hfov_deg']:.1f}° · VFOV {k['vfov_deg']:.1f}°) "
        f"B={k['baseline_m'] * 1000:.4f} mm L={tuple(round(v, 4) for v in rig['left_pos'])} "
        f"R={tuple(round(v, 4) for v in rig['right_pos'])} · ZED Mini housing = render USD (head_camera.py) "
        f"· exo = front copy"
    )
