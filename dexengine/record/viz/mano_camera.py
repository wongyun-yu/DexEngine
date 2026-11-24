"""Scene focus, camera fit and the camera sidecar of the MANO mesh video (viz.mano_render). numpy + trimesh only (no
Open3D), so the camera decisions can be checked without a renderer."""

from __future__ import annotations

import json
import math
import os

import numpy as np
import trimesh

from dexengine.core.desk import DESK_TOP_SURFACE_Z as DESK_HEIGHT


def transform_mesh_verts(verts_rest: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to (N,3) vertices."""
    homog = np.concatenate([verts_rest, np.ones((verts_rest.shape[0], 1))], axis=1)
    return (T @ homog.T).T[:, :3]


def load_obj(path):
    if not path:
        return None
    m = trimesh.load(path, force="mesh", process=False)
    return np.asarray(m.vertices), np.asarray(m.faces, dtype=np.int32)


def scene_focus(data: dict, r_obj, l_obj, cam_fit_scope: str) -> tuple[np.ndarray, list]:
    """Per-frame focus (centroid of both hands and both objects) and the point sample of the whole trajectory for
    --cam_fit (hands only with cam_fit_scope 'hands') -> (focus_per_frame [T, 3], fit_pts)."""
    # Per-frame focus = centroid of BOTH hands + BOTH objects (a right-side-only focus makes the
    # "follow" camera track the right side and crop the LEFT hand out of frame).
    # scene_{lo,hi} accumulate the whole-trajectory bbox for --cam_fit auto zoom-out.
    focus_per_frame = np.zeros((data["n_frames"], 3))
    scene_lo = np.full(3, np.inf)
    scene_hi = np.full(3, -np.inf)
    fit_pts = []  # point sample (whole trajectory) for the exact --cam_fit
    for t in range(data["n_frames"]):
        groups = [data["right_verts"][t], data["left_verts"][t]]
        if r_obj is not None:
            groups.append(transform_mesh_verts(r_obj[0], data["right_obj_T"][t]))
        if l_obj is not None:
            groups.append(transform_mesh_verts(l_obj[0], data["left_obj_T"][t]))
        focus_per_frame[t] = np.mean([g.mean(0) for g in groups], axis=0)
        # --cam_fit_scope hands: zoom from the hands only
        fit_groups = groups[:2] if cam_fit_scope == "hands" else groups
        for g in fit_groups:
            scene_lo = np.minimum(scene_lo, g.min(0))
            scene_hi = np.maximum(scene_hi, g.max(0))
            # <= 64 points per frame for the exact fit (below)
            fit_pts.append(g[:: max(1, len(g) // 64)])
    return focus_per_frame, fit_pts


def view_dir(cam_dir, cam_dist):
    """Direction-locked eye: place eye at target − dist·dir each frame so the viewing AXIS is exactly cam_dir at a
    controlled zoom (cam_dist). -> unit direction, or None when cam_dir or cam_dist is missing."""
    dir_unit = None
    if cam_dir is not None and cam_dist is not None:
        n = float(np.linalg.norm(cam_dir))
        if n < 1e-9:
            raise ValueError("cam_dir must be non-zero")
        dir_unit = np.asarray(cam_dir, dtype=np.float64) / n
    return dir_unit


def fit_distance(fit_pts, ref, dir_unit, cam_up, cam_fov, width, height, cam_fit, cam_dist, cam_fit_scope) -> float:
    """--cam_fit: zoom OUT (never in) so the whole-trajectory scene points fit the frame -> the eye distance.

    exact fit: the minimum distance at which every point p (relative to the target ref) fits the frame in the real
    camera basis, in closed form (a bounding sphere radius overestimates up to √3 and makes hands small):
      x=p·right, y=p·up, z=p·fwd, eye=target−d·fwd -> depth z+d, condition |x|·margin <= tan_h·(z+d),
      |y|·margin <= tan_v·(z+d) -> d >= |x|·margin/tan_h − z, d >= |y|·margin/tan_v − z. The max over all points
      = the needed distance; cam_dist is a floor, so small scenes keep their framing."""
    tan_v = math.tan(math.radians(cam_fov) * 0.5)
    tan_h = tan_v * width / height
    up_w = np.asarray(cam_up, dtype=np.float64) / np.linalg.norm(cam_up)
    right = np.cross(dir_unit, up_w)
    right /= np.linalg.norm(right)
    up_c = np.cross(right, dir_unit)
    P = np.concatenate(fit_pts, 0) - ref
    x, y, z = P @ right, P @ up_c, P @ dir_unit
    need = float(max((np.abs(x) * cam_fit / tan_h - z).max(), (np.abs(y) * cam_fit / tan_v - z).max()))
    if need > cam_dist:
        print(
            f"  [render_mano_mesh] cam_fit: dist {cam_dist:.3f} -> {need:.3f} "
            f"(scope={cam_fit_scope}, sample points {len(P)}, margin x{cam_fit})"
        )
        cam_dist = need
    return cam_dist


def camera_record(
    data: dict, dir_unit, cam_dist, cam_target_mode, cam_target_fixed, cam_eye, cam_fov, cam_up, width, height
) -> dict:
    """The camera sidecar (`<out>.cam.json`): the **owner record** of the camera that shot this video. Other renders
    from the same view (kinematic retarget, viz.retarget_render) read only this file; the camera is not resolved twice.
    follow mode has a different target per frame and cannot become a fixed camera, so it writes target=null and
    consumers refuse it."""
    return {
        "frame": "world",
        "dir": (dir_unit.tolist() if dir_unit is not None else None),
        "dist": (float(cam_dist) if dir_unit is not None else None),
        "target": (None if cam_target_mode == "follow" else np.asarray(cam_target_fixed, dtype=np.float64).tolist()),
        "target_mode": cam_target_mode,
        "eye": (None if dir_unit is not None else np.asarray(cam_eye, dtype=np.float64).tolist()),
        "fov_deg": float(cam_fov),
        "up": np.asarray(cam_up, dtype=np.float64).tolist(),
        "width": int(width),
        "height": int(height),
        "n_frames": int(data["n_frames"]),
        "fps": data["fps"],
        "placement_yaw_deg": float(data.get("placement_yaw_deg", 0.0)),
        "placement_xy_shift": data.get("placement_xy_shift", [0.0, 0.0]),
        "placement_pivot": data.get("placement_pivot", [0.0, 0.0]),
        "tree_dir": data.get("tree_dir"),
        "sequence": data.get("sequence"),
        "desk_height": DESK_HEIGHT,
        "source": "dexengine/record/viz/mano_render.py",
    }


def write_camera(cam_rec: dict, cam_json_out: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(cam_json_out)) or ".", exist_ok=True)
    with open(cam_json_out, "w", encoding="utf-8") as _f:
        json.dump(cam_rec, _f, indent=1)
    print(f"  [render_mano_mesh] camera sidecar -> {cam_json_out}", flush=True)


def print_camera(dir_unit, cam_dist, cam_target_mode, cam_target_fixed, cam_eye, cam_fov, cam_up) -> None:
    """The camera line of the render log (direction-locked or fixed eye)."""
    if dir_unit is not None:
        _tgt = (
            "follow"
            if cam_target_mode == "follow"
            else tuple(np.asarray(cam_target_fixed, dtype=np.float64).round(4).tolist())
        )
        print(
            f"  [render_mano_mesh] camera: dir={tuple(dir_unit.round(4).tolist())} dist={cam_dist:.4f} "
            f"(eye=target−dist·dir), target={cam_target_mode}={_tgt}, fov={cam_fov:.1f}°, "
            f"up={tuple(np.asarray(cam_up).round(3).tolist())}"
        )
    else:
        print(
            f"  [render_mano_mesh] camera: eye={tuple(cam_eye.round(3))}, target={cam_target_mode}, fov={cam_fov:.1f}°"
        )
