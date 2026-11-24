"""Full MANO MESH renderer (not markers).

Replays the original MANO articulated hand SURFACE (778-vertex mesh per hand)
+ object mesh, frame-by-frame, via Open3D offscreen rendering, for TACO
sequences (TACO only; other dataset adapters are out of scope).

Dataset-specific loaders return the same in-memory format:
    {
      "frames": [
        {
          "right_verts": (778,3) | None,   # MANO right hand vertices (world frame)
          "left_verts":  (778,3) | None,
          "right_obj_T": (4,4)    | None,  # right object 4x4 transform
          "left_obj_T":  (4,4)    | None,
        }, ...
      ],
      "right_obj_mesh_path": str | None,
      "left_obj_mesh_path":  str | None,
      "fps": int,
    }

Run via the retarget venv (`$RETARGET_PY` = paths.py --what retarget-py; has open3d + smplx + mujoco + imageio):
    $RETARGET_PY \
        -m dexengine.record.viz.mano_render \
        --dataset taco --sequence "brush_brush_bowl__20231005_190" \
        --out results/taco/mano_reference/brush_brush_bowl__20231005_190/mano_mesh.mp4
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import imageio
import numpy as np
import open3d as o3d

# isort: split
# names and suffixes are owned by dexengine/core/paths.py (no literals)

from dexengine.core.desk import DESK_TOP_SURFACE_Z as DESK_HEIGHT
from dexengine.core.errors import InputMissingError
from dexengine.record.viz.mano_camera import (
    camera_record,
    fit_distance,
    load_obj,
    print_camera,
    scene_focus,
    transform_mesh_verts,
    view_dir,
    write_camera,
)
from dexengine.record.viz.mano_model import (
    ISAAC_HEAD_EYE,
    RETARGET_CAM_FOVY,
    get_mano,
    raw_dir_to_world,
    retarget_cam_dir_raw,
)
from dexengine.record.viz.mano_taco import load_taco

COLOR_RIGHT_HAND = (0.85, 0.55, 0.45)  # peachy skin
COLOR_LEFT_HAND = (0.55, 0.85, 0.55)  # green-tinted skin
COLOR_RIGHT_OBJ = (0.40, 0.55, 0.90)  # blue (right object)
COLOR_LEFT_OBJ = (0.90, 0.70, 0.30)  # amber (left object)
# dexengine/record/viz/retarget_render.py reads this value too (single owner)
DESK_COLOR = (0.55, 0.45, 0.35)  # desk top (brown)


def make_o3d_mesh(verts: np.ndarray, faces: np.ndarray, color: tuple) -> o3d.geometry.TriangleMesh:
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    m.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    m.compute_vertex_normals()
    m.paint_uniform_color(color)
    return m


def _scene_parts(data, cam_fit_scope, width, height):
    """Hand faces, object meshes, focus and fit points, the renderer, the desk plane and the lit material."""
    rh_faces = get_mano(True, batch_size=1).faces.astype(np.int32)
    lh_faces = get_mano(False, batch_size=1).faces.astype(np.int32)
    r_obj = load_obj(data["right_obj_mesh_path"])
    l_obj = load_obj(data["left_obj_mesh_path"])
    focus_per_frame, fit_pts = scene_focus(data, r_obj, l_obj, cam_fit_scope)
    renderer = _renderer(width, height)
    desk_verts, desk_faces = _desk_plane()
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    return rh_faces, lh_faces, r_obj, l_obj, focus_per_frame, fit_pts, renderer, desk_verts, desk_faces, mat


def _view_of(cam_target_mode, cam_target_fixed, cam_dist, dir_unit, cam_eye, cam_fov, cam_up):
    """The resolved camera view handed to the frame renderer."""
    view = {
        "mode": cam_target_mode,
        "fixed": cam_target_fixed,
        "dist": cam_dist,
        "dir": dir_unit,
        "eye": cam_eye,
        "fov": cam_fov,
        "up": cam_up,
    }
    return view


def render_sequence(
    data: dict,
    out_path: str,
    width: int = 768,
    height: int = 1920,
    cam_eye: np.ndarray | None = None,
    cam_target_mode: str = "follow",  # "follow" | "workspace" | "fixed"
    cam_target_fixed: np.ndarray | None = None,
    cam_fov: float = 55.0,
    cam_up: np.ndarray | None = None,
    cam_dir: np.ndarray | None = None,
    cam_dist: float | None = None,
    cam_fit: float = 0.0,
    cam_fit_scope: str = "all",
    cam_json_out: str | None = None,
    cam_only: bool = False,
) -> None:
    """Render MANO+object scene with a configurable camera.

    cam_eye: world-coords (3,) for the camera eye. None → ISAAC_HEAD_EYE.
    cam_target_mode:
      - "follow":   per-frame MANO scene centroid (default, follows the action)
      - "workspace": fixed = mean MANO scene centroid across trajectory
      - "fixed":    use cam_target_fixed (must be provided)
    cam_fov: vertical FOV in degrees.
    cam_up:  world up vector. None → (0, 0, 1) for Z-up.
    cam_dir / cam_dist: if BOTH given, the eye is placed at
      `target − cam_dist · normalize(cam_dir)` each frame instead of using
      `cam_eye`. This locks the viewing DIRECTION (cam_dir = eye→target axis)
      exactly while controlling zoom via distance, centred on the resolved
      target (follow=per-frame, workspace/fixed=static). Used to reproduce a
      robot camera's exact optical axis at an arbitrary zoom.
    """
    if cam_eye is None:
        cam_eye = ISAAC_HEAD_EYE
    if cam_up is None:
        cam_up = np.array([0.0, 0.0, 1.0])
    if cam_target_mode == "fixed" and cam_target_fixed is None:
        raise ValueError("cam_target_mode='fixed' requires cam_target_fixed=(x,y,z)")
    rh_faces, lh_faces, r_obj, l_obj, focus_per_frame, fit_pts, renderer, desk_verts, desk_faces, mat = _scene_parts(
        data, cam_fit_scope, width, height
    )

    # Resolve target mode
    if cam_target_mode == "workspace":
        cam_target_fixed = focus_per_frame.mean(0)
    dir_unit = view_dir(cam_dir, cam_dist)
    # --cam_fit: zoom OUT (never in) so the whole-trajectory scene bbox fits the frame.
    # cam_dist becomes max(cam_dist, radius·margin / tan(half-fov)); the preset distance
    # is a floor, so small scenes keep their framing. Default 0.0 = off (byte-identical).
    if cam_fit > 0:
        if dir_unit is None:
            raise ValueError("--cam_fit requires --cam_dir and --cam_dist (direction-locked camera)")
        ref = (
            focus_per_frame.mean(0)
            if (cam_target_mode == "follow" or cam_target_fixed is None)
            else np.asarray(cam_target_fixed, dtype=np.float64)
        )
        cam_dist = fit_distance(
            fit_pts, ref, dir_unit, cam_up, cam_fov, width, height, cam_fit, cam_dist, cam_fit_scope
        )

    if cam_json_out is None:
        cam_json_out = out_path + ".cam.json"
    cam_rec = camera_record(
        data, dir_unit, cam_dist, cam_target_mode, cam_target_fixed, cam_eye, cam_fov, cam_up, width, height
    )
    write_camera(cam_rec, cam_json_out)
    if cam_only:
        print(f"  [render_mano_mesh] --cam_only: frame render skipped (n_frames={data['n_frames']})", flush=True)
        return

    view = _view_of(cam_target_mode, cam_target_fixed, cam_dist, dir_unit, cam_eye, cam_fov, cam_up)
    images = _render_frames(
        renderer, data, mat, (desk_verts, desk_faces), (rh_faces, lh_faces), (r_obj, l_obj), focus_per_frame, view
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.mimsave(out_path, images, fps=data["fps"], macro_block_size=1)
    print(f"  [render_mano_mesh] {data['n_frames']} frames @ {data['fps']} fps -> {out_path}")
    print_camera(dir_unit, cam_dist, cam_target_mode, cam_target_fixed, cam_eye, cam_fov, cam_up)


def _renderer(width: int, height: int):
    """Open3D offscreen renderer with the light-grey background and a sun light when the build supports it."""
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([0.85, 0.88, 0.92, 1.0])
    try:
        renderer.scene.scene.set_sun_light([-0.3, -0.5, -1.0], [1.0, 1.0, 1.0], 90000)
        renderer.scene.scene.enable_sun_light(True)
    except (AttributeError, RuntimeError) as e:  # an Open3D build without the sun-light API: default lighting
        print(f"  [render_mano_mesh] sun light not set ({type(e).__name__}: {e}); default lighting", flush=True)
    return renderer


def _desk_plane() -> tuple[np.ndarray, np.ndarray]:
    """Desk plane at the Isaac desk height (core.desk), broad enough to fill the view -> (verts, faces)."""
    desk_size = 1.5
    desk_verts = np.array(
        [
            [-desk_size, -desk_size, DESK_HEIGHT],
            [desk_size, -desk_size, DESK_HEIGHT],
            [desk_size, desk_size, DESK_HEIGHT],
            [-desk_size, desk_size, DESK_HEIGHT],
        ]
    )
    desk_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return desk_verts, desk_faces


def _render_frames(renderer, data: dict, mat, desk, faces, objs, focus_per_frame, view: dict) -> list:
    """One image per frame: desk, both hands, both objects; camera per view (follow target per frame, or fixed;
    direction-locked eye = target − dist·dir, else the fixed eye)."""
    (desk_verts, desk_faces), (rh_faces, lh_faces), (r_obj, l_obj) = desk, faces, objs
    cam_target_mode, cam_target_fixed, cam_dist = view["mode"], view["fixed"], view["dist"]
    dir_unit, cam_eye, cam_fov, cam_up = view["dir"], view["eye"], view["fov"], view["up"]
    images = []
    for t in range(data["n_frames"]):
        renderer.scene.clear_geometry()
        renderer.scene.add_geometry("desk", make_o3d_mesh(desk_verts, desk_faces, DESK_COLOR), mat)
        renderer.scene.add_geometry("rh", make_o3d_mesh(data["right_verts"][t], rh_faces, COLOR_RIGHT_HAND), mat)
        renderer.scene.add_geometry("lh", make_o3d_mesh(data["left_verts"][t], lh_faces, COLOR_LEFT_HAND), mat)
        if r_obj is not None:
            rv = transform_mesh_verts(r_obj[0], data["right_obj_T"][t])
            renderer.scene.add_geometry("rob", make_o3d_mesh(rv, r_obj[1], COLOR_RIGHT_OBJ), mat)
        if l_obj is not None:
            lv = transform_mesh_verts(l_obj[0], data["left_obj_T"][t])
            renderer.scene.add_geometry("lob", make_o3d_mesh(lv, l_obj[1], COLOR_LEFT_OBJ), mat)

        target = focus_per_frame[t] if cam_target_mode == "follow" else cam_target_fixed
        eye_t = (target - cam_dist * dir_unit) if dir_unit is not None else cam_eye
        renderer.setup_camera(cam_fov, target.tolist(), eye_t.tolist(), cam_up.tolist())
        img = renderer.render_to_image()
        images.append(np.asarray(img))
    return images


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

DATASET_LOADERS = {
    "taco": load_taco,
}


def _parse_xyz(arg: str) -> np.ndarray:
    """Parse 'x,y,z' string into a numpy (3,) array."""
    parts = [float(p.strip()) for p in arg.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected 'x,y,z' got '{arg}'")
    return np.array(parts, dtype=np.float64)


def _add_input_args(parser):
    """Dataset, sequence, output and frame-size options."""
    parser.add_argument("--dataset", required=True, choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--tree_dir",
        default=None,
        help="taco only: render with the objects, placement and time axis of this tree (e.g. a combo "
        "v2 tree): an SBS reference video whose absolute positions match the robot video.",
    )
    parser.add_argument(
        "--legacy_slot_diag",
        action="store_true",
        default=False,
        help="[diagnosis only] draw hand <-> object slots the old fixed way (right = slot 0). For "
        "before/after comparisons, default OFF. Never used by training, evaluation or annotation.",
    )
    parser.add_argument(
        "--z_anchor",
        default="legacy",
        choices=["legacy", "contact"],
        help="[retired] z is always the single constant (taco_frames.HAND_SHIFT_Z); "
        "this flag is accepted for compatibility only and ignored. "
        "Default legacy = existing outputs byte-identical.",
    )
    parser.add_argument("--width", type=int, default=1920, help="output mp4 width (px)")
    parser.add_argument("--height", type=int, default=1920, help="output mp4 height (px)")


def _add_camera_pose_args(parser):
    """Camera pose options (eye, target, fov, frame, preset, up, direction)."""
    parser.add_argument(
        "--cam_eye",
        type=_parse_xyz,
        default=None,
        help="Camera eye position 'x,y,z' in Isaac world coords. "
        f"Default: Isaac robot head pose {tuple(ISAAC_HEAD_EYE)}.",
    )
    parser.add_argument(
        "--cam_target",
        type=str,
        default="follow",
        help="Camera target. 'follow' = per-frame MANO scene centroid "
        "(default, follows the action). 'x,y,z' = fixed world point. "
        "'workspace' = average MANO scene centroid (fixed).",
    )
    parser.add_argument(
        "--cam_fov",
        type=float,
        default=None,
        help="Vertical FOV degrees. Default 55 (narrower than Isaac's 106° "
        "wide-angle for a clearer view of the action); 45 (MuJoCo fovy) with --cam_preset "
        "retarget.",
    )
    parser.add_argument(
        "--cam_frame",
        default="world",
        choices=["world", "raw"],
        help="Frame of --cam_dir. world = this render's world (Isaac) frame (default). raw = the "
        "TACO raw (MANO layer) frame: the clip's placement yaw and Rz(−90°) are applied exactly as to "
        "the hands to turn it into a world direction (marked in the log).",
    )
    parser.add_argument(
        "--cam_preset",
        default=None,
        choices=["retarget"],
        help="retarget = the same view direction and fovy as the retargeting video "
        "(visualization_ik.mp4, camera 'front'). An error together with --cam_dir/--cam_frame (no "
        "silent override). Zoom with --cam_dist/--cam_fit.",
    )
    parser.add_argument(
        "--cam_up", type=_parse_xyz, default=None, help="Camera up vector 'x,y,z'. Default (0,0,1) for Z-up world."
    )
    parser.add_argument(
        "--cam_dir",
        type=_parse_xyz,
        default=None,
        help="Viewing direction 'dx,dy,dz' (eye→target axis), normalized. "
        "If given with --cam_dist, the eye is placed at "
        "target − dist·dir each frame, locking the optical AXIS exactly "
        "while zoom is set by --cam_dist (overrides --cam_eye). Use to "
        "reproduce a robot camera's exact angle at any zoom.",
    )


def _add_camera_fit_args(parser):
    """Camera distance, fit and sidecar options."""
    parser.add_argument(
        "--cam_dist",
        type=float,
        default=None,
        help="Distance (m) from target to eye along --cam_dir. Smaller = closer/zoomed in.",
    )
    parser.add_argument(
        "--cam_fit_scope",
        default="all",
        choices=["all", "hands"],
        help="What --cam_fit fits. all = both hands + both objects (default). hands = both hands only, "
        "for when large objects (pans, pots) make the hands small (objects may then leave the "
        "frame at times). For MANO <-> retarget comparisons.",
    )
    parser.add_argument(
        "--cam_only",
        action="store_true",
        default=False,
        help="Write only the camera sidecar (`<out>.cam.json`) without rendering frames (FK and the "
        "fit still run). Used to hand the same view to dexengine/record/viz/retarget_render.py.",
    )
    parser.add_argument("--cam_json_out", default=None, help="camera sidecar path (default `<out>.cam.json`).")
    parser.add_argument(
        "--cam_fit",
        type=float,
        default=0.0,
        help="Zoom-OUT margin: raise --cam_dist (never lower it) until the whole-"
        "trajectory bbox of BOTH hands + BOTH objects fits the frame, times "
        "this margin (e.g. 1.15). Needs --cam_dir/--cam_dist. 0 = off. "
        "Pair with --cam_target workspace for a fully static camera.",
    )


def _camera_from_args(args, data):
    """Camera eye, up, direction and fov from the options (preset, raw-frame conversion)."""
    cam_eye = args.cam_eye if args.cam_eye is not None else ISAAC_HEAD_EYE
    cam_up = args.cam_up if args.cam_up is not None else np.array([0.0, 0.0, 1.0])

    # --- view direction resolution (owned here alone) ---
    cam_dir = args.cam_dir
    cam_frame = args.cam_frame
    cam_fov = args.cam_fov
    if args.cam_preset == "retarget":
        if args.cam_dir is not None or args.cam_frame != "world":
            raise InputMissingError(
                "[render_mano] --cam_preset retarget cannot be used with --cam_dir/--cam_frame (the preset sets both)."
            )
        cam_dir = retarget_cam_dir_raw()
        cam_frame = "raw"
        if cam_fov is None:
            cam_fov = RETARGET_CAM_FOVY
        print(f"  [render_mano] cam_preset=retarget: raw dir={tuple(cam_dir.round(4))} fovy={cam_fov}", flush=True)
    if cam_fov is None:
        cam_fov = 55.0
    if cam_dir is not None and args.cam_dist is None:
        raise SystemExit("[render_mano] --cam_dir needs --cam_dist (direction-fixed camera).")
    if cam_frame == "raw":
        if cam_dir is None:
            raise SystemExit("[render_mano] --cam_frame raw needs --cam_dir (raw frame).")
        if "placement_yaw_deg" not in data:
            raise SystemExit(
                "[render_mano] --cam_frame raw is supported only by loaders that know the placement yaw (taco)."
            )
        yaw = float(data["placement_yaw_deg"])
        cam_dir = raw_dir_to_world(cam_dir, yaw)
        az = math.degrees(math.atan2(cam_dir[1], cam_dir[0]))
        el = math.degrees(math.asin(cam_dir[2]))
        print(
            f"  [render_mano] cam_frame=raw: placement yaw={yaw:+.1f}° → world dir={tuple(cam_dir.round(4))} "
            f"(az {az:.1f}°, el {el:.1f}°)",
            flush=True,
        )
    return cam_eye, cam_up, cam_dir, cam_fov


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_input_args(parser)

    # --- Camera configuration ---
    _add_camera_pose_args(parser)
    _add_camera_fit_args(parser)

    args = parser.parse_args()

    if args.dataset == "taco":
        data = load_taco(
            args.sequence, tree_dir=args.tree_dir, z_anchor=args.z_anchor, legacy_slot=bool(args.legacy_slot_diag)
        )
    else:
        data = DATASET_LOADERS[args.dataset](args.sequence)
    print(
        f"  [load_{args.dataset}] seq='{args.sequence}' frames={data['n_frames']} fps={data['fps']} "
        f"r_obj={'OK' if data['right_obj_mesh_path'] else '—'} "
        f"l_obj={'OK' if data['left_obj_mesh_path'] else '—'}"
    )

    cam_eye, cam_up, cam_dir, cam_fov = _camera_from_args(args, data)

    # Resolve target: follow / workspace / explicit xyz
    if args.cam_target == "follow":
        cam_target_mode = "follow"
        cam_target_fixed = None
    elif args.cam_target == "workspace":
        # Will be computed inside render_sequence as the trajectory centroid
        cam_target_mode = "workspace"
        cam_target_fixed = None
    else:
        cam_target_mode = "fixed"
        cam_target_fixed = _parse_xyz(args.cam_target)

    render_sequence(
        data,
        args.out,
        width=args.width,
        height=args.height,
        cam_eye=cam_eye,
        cam_target_mode=cam_target_mode,
        cam_target_fixed=cam_target_fixed,
        cam_fov=cam_fov,
        cam_up=cam_up,
        cam_dir=cam_dir,
        cam_dist=args.cam_dist,
        cam_fit=args.cam_fit,
        cam_fit_scope=args.cam_fit_scope,
        cam_json_out=args.cam_json_out,
        cam_only=bool(args.cam_only),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
