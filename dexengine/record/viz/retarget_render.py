#!/usr/bin/env python3
"""MuJoCo render of the kinematic retarget (tree `0/trajectory_kinematic.npz`) with **the same camera as the MANO
video**.

The camera sidecar `<mano>.cam.json` left by the MANO video (dexengine/record/viz/mano_render.py) is the only input; the
tree path is read from it too. The scene.xml (robot hand = that tree's combo, e.g. rby1_sharpa_wave) and trajectory of
**the same tree** are used, so placement and time axis equal the MANO video. Moving the world (Isaac) frame camera into
the scene frame uses only the inverse of dexengine.record.viz.mano_render's chain (z −DESK_HEIGHT -> Rz(+90°)): the
scene qpos of a placed tree is **already in placed coordinates** (measured: object xy = Rz(yaw)(p−pivot)+pivot+shift,
error 0.0 cm) and the MANO gets the same placement, so the yaw is not applied again.

Colours: the robot hand keeps **the asset's own colours** (vendor URDF->MJCF rgba, light-green fingertip pads
included). Objects get the MANO colours (dexengine.record.viz.mano_render constants, side assignment = the same
`side_objects.resolve`), the desk top dexengine.record.viz.mano_render.DESK_COLOR. Reference markers (sites) and
collision shapes (group 3) are hidden.

Usage (retarget venv):
  dexengine/record/viz/mano_render.py … --tree_dir <tree> --out X_mano.mp4      # X_mano.mp4 + X_mano.mp4.cam.json
  python -m dexengine.record.viz.retarget_render --cam_json X_mano.mp4.cam.json --out X_retarget.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from dexengine.core.errors import InputMissingError

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio  # noqa: E402
import mujoco  # noqa: E402

from dexengine.core.desk import DESK_TOP_SURFACE_Z  # noqa: E402  (desk height when the sidecar has none)
from dexengine.record.viz import mano_model as MM  # noqa: E402  (RZ_MINUS_90)
from dexengine.record.viz import mano_render as RM  # noqa: E402  (colours)

RZ_PLUS_90 = MM.RZ_MINUS_90.T  # inverse of Rz(−90°)


def _load_side_objects():
    from dexengine.data import side_objects as mod  # noqa: PLC0415

    return mod


def world_cam_to_scene(cam: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sidecar (world frame) -> scene (tree) frame (eye, right, up_c). The tree scene qpos already includes the
    placement, so the yaw is not applied again (module doc)."""
    if cam.get("frame") != "world":
        raise SystemExit(f"[render_retarget] sidecar frame={cam.get('frame')!r}: only 'world' is supported")
    if cam.get("dir") is None or cam.get("target") is None:
        raise InputMissingError(
            "[render_retarget] the sidecar has no dir/target (follow camera?): only MANO videos "
            "made with a fixed camera (--cam_target workspace|x,y,z) can be carried over"
        )
    d_w = np.asarray(cam["dir"], dtype=np.float64)
    d_w /= np.linalg.norm(d_w)
    t_w = np.asarray(cam["target"], dtype=np.float64)
    dist = float(cam["dist"])
    up_w = np.asarray(cam["up"], dtype=np.float64)
    up_w /= np.linalg.norm(up_w)
    eye_w = t_w - dist * d_w
    z = np.array([0.0, 0.0, float(cam.get("desk_height", DESK_TOP_SURFACE_Z))])
    eye = RZ_PLUS_90 @ (eye_w - z)
    d = RZ_PLUS_90 @ d_w
    up = RZ_PLUS_90 @ up_w
    right = np.cross(d, up)
    right /= np.linalg.norm(right)
    up_c = np.cross(right, d)
    return eye, right, up_c


def _scene_model(scene_xml, eye, right, up_c, fovy, W, H):
    """Scene model: skybox in the desk colour, a lowered headlight, one ceiling light, the mano_view camera; compiled
    with an offscreen buffer of at least W x H."""
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    desk = list(RM.DESK_COLOR)
    for tex in spec.textures:
        if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            tex.rgb1[:] = desk
            tex.rgb2[:] = desk
    # light: the retarget scene's headlight (specular 0.9) and two ceiling lights burn the desk top orange; to match the
    # matte tone of the MANO (Open3D defaultLit) the headlight is lowered, one ceiling light kept, mostly diffuse. The
    # colour constants stay (single owner dexengine.record.viz.mano_render).
    spec.visual.headlight.ambient[:] = [0.55, 0.55, 0.55]
    spec.visual.headlight.diffuse[:] = [0.55, 0.55, 0.55]
    spec.visual.headlight.specular[:] = [0.05, 0.05, 0.05]
    _lights = list(spec.worldbody.lights)
    for i, lt in enumerate(_lights):
        lt.diffuse[:] = [0.75, 0.75, 0.75] if i == 0 else [0.0, 0.0, 0.0]
        lt.specular[:] = [0.1, 0.1, 0.1] if i == 0 else [0.0, 0.0, 0.0]
        lt.castshadow = i == 0
    spec.worldbody.add_camera(
        name="mano_view",
        pos=eye.tolist(),
        xyaxes=[*right.tolist(), *up_c.tolist()],
        fovy=fovy,
        mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED,
    )
    m = spec.compile()
    d = mujoco.MjData(m)
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "mano_view")
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W)
    m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    return desk, m, d, cid


def _color_geoms(tree, m, desk):
    """Object and hand geom colours by side slot, floor = desk colour; a scene without hands or objects dies."""
    so = _load_side_objects().resolve(tree)
    slot_color = {int(so["right"][1]): RM.COLOR_RIGHT_OBJ, int(so["left"][1]): RM.COLOR_LEFT_OBJ}
    body_slot = {"right_object": 0, "left_object": 1}
    bname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(m.nbody)]
    n_hand = n_obj = 0
    for g in range(m.ngeom):
        b = bname[m.geom_bodyid[g]]
        if b in body_slot and int(m.geom_group[g]) in (0, 1):
            m.geom_rgba[g, :3] = slot_color[body_slot[b]]
            m.geom_rgba[g, 3] = 1.0
            n_obj += 1
        elif b.startswith(("right_", "left_")) and int(m.geom_group[g]) in (0, 1):
            n_hand += 1
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if fid < 0:
        raise InputMissingError("[render_retarget] the scene has no geom 'floor': nothing to paint as the desk top")
    m.geom_matid[fid] = -1
    m.geom_rgba[fid] = [*desk, 1.0]
    if n_hand == 0 or n_obj == 0:
        raise SystemExit(
            f"[render_retarget] visual geometry hand {n_hand} · object {n_obj}: the body name rule "
            "(right_/left_/…_object) or the group convention is broken"
        )
    return n_hand, n_obj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--cam_json",
        required=True,
        help="camera sidecar (<mano>.cam.json) written by dexengine.record.viz.mano_render; also gives the tree path",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--traj", default="trajectory_kinematic.npz", help="trajectory file under the tree's 0/ (default kinematic)"
    )
    a = ap.parse_args()

    cam = json.loads(Path(a.cam_json).read_text(encoding="utf-8"))
    if not cam.get("tree_dir"):
        raise InputMissingError(
            "[render_retarget] the sidecar has no tree_dir: remake it with the current dexengine.record.viz.mano_render"
        )
    tree = Path(cam["tree_dir"])
    scene_xml = tree / "scene.xml"
    traj = tree / "0" / a.traj
    for p in (scene_xml, traj):
        if not p.exists():
            raise InputMissingError(f"[render_retarget] missing: {p}")
    W, H = int(cam["width"]), int(cam["height"])
    fovy = float(cam["fov_deg"])
    eye, right, up_c = world_cam_to_scene(cam)

    # scene: skybox -> desk colour, fixed camera added
    desk, m, d, cid = _scene_model(scene_xml, eye, right, up_c, fovy, W, H)

    # colours: the robot hand keeps the asset colours (not painted). Only the objects' visual geometry (group 0 · 1)
    # gets the MANO colours; side assignment by the same resolver.
    n_hand, n_obj = _color_geoms(tree, m, desk)

    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0  # hide reference markers
    opt.geomgroup[:] = 0  # visual meshes (0 · 1) only, collision (3) hidden
    opt.geomgroup[0] = 1
    opt.geomgroup[1] = 1
    npz = np.load(traj)
    q = npz["qpos"]
    fps = float(npz["frequency"]) if "frequency" in npz.files else float(cam["fps"])
    if q.shape[1] != m.nq:
        raise SystemExit(
            f"[render_retarget] qpos width {q.shape[1]} != model nq {m.nq}: trajectory and scene are of "
            "different generations"
        )
    if q.shape[0] != int(cam["n_frames"]):
        print(
            f"  [render_retarget] ⚠ frames {q.shape[0]} != MANO {cam['n_frames']}: different time axes (drawn as "
            "they are)",
            flush=True,
        )

    renderer = mujoco.Renderer(m, height=H, width=W)
    frames = []
    for t in range(q.shape[0]):
        d.qpos[:] = q[t]
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera=cid, scene_option=opt)
        frames.append(renderer.render().copy())
    renderer.close()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    imageio.mimsave(a.out, frames, fps=fps, macro_block_size=1)
    fwd = -np.cross(right, up_c)
    print(f"  [render_retarget] {len(frames)} frames @ {fps:g} fps {W}x{H} -> {a.out}", flush=True)
    print(
        f"  [render_retarget] camera(scene frame): eye={tuple(eye.round(4).tolist())} "
        f"fwd={tuple(fwd.round(4).tolist())} "
        f"fovy={fovy:.1f} (from {a.cam_json}) · tree={tree} · hand visual geoms {n_hand} (asset colours) · object "
        f"visual geoms {n_obj} (MANO colours)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
