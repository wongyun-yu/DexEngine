"""Still render of the reference frame-0 pose on the real training env, without a policy (``--pose_render_out``).

Before a new (arm, hand) combo is trained, a person checks by eye that the robot stands at its place in front of the
desk, that the arm reaches the reference frame-0 pose, and that the hand meets the human (MANO) fingertips. No
checkpoint is needed: the env's reset events already send the arm to its arm-IK frame-0 solution, the hand to the
reference qpos and the object to its reference pose.

Reused, not re-derived:
  * the env's reset event terms (the same ones RSI and the kinematic replay use; the PhysX order of the hand qpos is
    handled there);
  * TrajectoryManager.root_quat_rotated / _desk_height, the transform from MANO coordinates to world;
  * scene["front_camera"] as the render camera, moved per view (created by create_env for --pose_render_out:
    record.render.cameras.add_scene_cameras, --num_envs 1);
  * the one canonical look (record.looks.look_d.apply — room, lights, materials; it sets the frame-guard token) and
    the frame guard (record.looks.look_guard.require) before a pixel is read or written.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from dexengine.core.errors import InputMissingError

# Camera directions from the robot (unnormalized). The robot looks along +x towards the desk (base x < 0, desk top
# x ≈ 0.03), so "front" looks at the robot from the desk side.
VIEWS = {
    "iso": (0.78, -0.90, 0.42),
    "front": (1.00, 0.00, 0.06),
    "side": (0.00, -1.00, 0.06),
    "back": (-1.00, 0.00, 0.06),
    "top": (0.02, -0.10, 1.00),
}


#: renders of the same pose before a still is read (RTX accumulation and MDL loading; the record rig's still warm-up)
STILL_RENDERS = 16


def _save(rgb: np.ndarray, path: str) -> None:
    """Write an RGB uint8 image (PIL, else imageio) — only after the canonical look (frame guard)."""
    from dexengine.record.looks import look_guard as LG

    LG.require(f"rl.render.pose_check._save({os.path.basename(path)})")
    try:
        from PIL import Image

        Image.fromarray(rgb).save(path)
    except ImportError:
        import imageio.v3 as iio

        iio.imwrite(path, rgb)


def _pin_clip(raw_env, sequence: str) -> tuple[int, str]:
    """Pin env 0 to ``sequence`` and return (clip_id, sequence name).

    An empty ``sequence`` keeps the assigned clip. Names come from the loader, because args_cli.sequence is empty in
    --sequence_dir mode.
    """
    tm = raw_env.trajectory_manager
    names = list(tm._loader.sequence_names)
    if sequence:
        if sequence not in names:
            raise InputMissingError(f"pose_render: sequence {sequence!r} is not among the {len(names)} loaded")
        tm.clip_ids[0] = names.index(sequence)
    cid = int(tm.clip_ids[0].item())
    return cid, names[cid]


def _ref_fingertips_world(raw_env, sequence: str, frame: int = 0):
    """Retargeting-FK fingertips (5 per active side) at ``frame`` in world coordinates: ((world [5n, 3], sides), "") or
    (None, reason).

    ``frame`` must match the robot's reset frame, otherwise the tip error measures a frame mismatch instead of a pose
    mismatch.
    """
    tm = raw_env.trajectory_manager
    path = os.path.join(tm._loader.data_dir, sequence, "0", "body_positions.npz")
    if not os.path.isfile(path):
        return None, f"no body_positions.npz ({path})"
    if getattr(tm, "root_quat_rotated", None) is None:
        return None, "trajectory_manager.root_quat_rotated is None (no-rotation mode)"

    import isaaclab.utils.math as math_utils

    from dexengine.rl.tasks.env_base import active_viz_sides

    sides = active_viz_sides(raw_env.scene)
    npz = np.load(path, allow_pickle=True)
    missing = [f"{s}_fingertip_pos" for s in sides if f"{s}_fingertip_pos" not in npz]
    if missing:
        return None, f"missing keys: {', '.join(missing)}"
    _fi = max(0, min(int(frame), int(npz[f"{sides[0]}_fingertip_pos"].shape[0]) - 1))
    local = np.concatenate([npz[f"{s}_fingertip_pos"][_fi, :5, :] for s in sides], axis=0)
    local_t = torch.tensor(local, dtype=torch.float32, device=raw_env.device)

    # Same transform as the per-clip render's fingertip overlay: env origin + desk height, rotated by root_quat_rotated
    root_pos = raw_env.scene.env_origins[0].clone()
    root_pos[2] += float(tm._desk_height)
    q = tm.root_quat_rotated
    world = root_pos + math_utils.quat_apply(q.unsqueeze(0).expand(local_t.shape[0], -1), local_t)
    return (world, sides), ""


def _report_tip_error(raw_env, tips_w, sides) -> None:
    """Print the per-finger error between the robot fingertip sites and the reference fingertips.

    Robot tips come from the same function the reward and contact terms use (tip_sites_w): pad sites, not body
    origins, so a hand whose body origin sits 2.6 cm from the pad (SharpA) is measured correctly.
    """
    from dexengine.rl.tasks.mdp.rewards.contact_base import (
        resolve_tip_body_ids,
        resolve_tip_offsets_tensor,
        tip_sites_w,
    )

    for k, side in enumerate(sides):
        key = f"{side}_hand" if f"{side}_hand" in raw_env.scene.keys() else "robot"  # noqa: SIM118  — InteractiveScene has no __contains__
        hand = raw_env.scene[key]
        ids = resolve_tip_body_ids(hand, side)
        offs = resolve_tip_offsets_tensor(hand, side, raw_env.device)
        rob = tip_sites_w(hand, ids, offs)[0]  # (5,3) pad sites
        # Body origins are measured too: if the site-offset error is the larger one, retargeting matched the body
        # origins (not the sites) to the MANO tips, which is an offset-convention issue of the spec, not the hand.
        rob0 = hand.data.body_pos_w[0][ids]  # (5,3) body origins
        # RH56F1 tips are the *_2 / thumb_4 body origins (PIP / IP link origins) while the reference points are the
        # true fingertips, so a perfect pose would read ~3.5 cm. This diagnostic adds the URDF *_tip frame offsets
        # (offline FK values, hand-local, per side); the training reward's tip sites are unchanged.
        _RH_TIP_OFF = {
            "left": [
                [+0.01668, +0.02531, +0.00676],
                [+0.00492, +0.03963, -0.00600],
                [+0.00596, +0.04251, -0.00610],
                [+0.00492, +0.03963, -0.00600],
                [+0.00512, +0.03254, -0.00635],
            ],
            "right": [
                [-0.01740, +0.02520, +0.00680],
                [+0.00492, +0.03963, +0.00600],
                [+0.00596, +0.04251, +0.00610],
                [+0.00492, +0.03963, +0.00600],
                [+0.00512, +0.03254, +0.00599],
            ],
        }
        _nchk = [hand.data.body_names[i] for i in (ids.tolist() if hasattr(ids, "tolist") else ids)]
        if any("_hand_thumb_4" in n or "_hand_index_2" in n for n in _nchk):
            import isaaclab.utils.math as _mu

            _offt = torch.tensor(_RH_TIP_OFF[side], dtype=rob0.dtype, device=rob0.device)
            rob = rob0 + _mu.quat_apply(hand.data.body_quat_w[0][ids], _offt)
            print(f"  [pose_render] {side} RH56F1 tip offsets applied (URDF *_tip)", flush=True)
        ref = tips_w[5 * k : 5 * (k + 1)]  # (5,3)
        names = [hand.data.body_names[i] for i in (ids.tolist() if hasattr(ids, "tolist") else ids)]
        paired = torch.norm(rob - ref, dim=-1)  # error per finger pair
        nearest = torch.cdist(ref, rob).min(dim=1).values  # error to the nearest robot tip
        per = " ".join(
            f"{n.split('_')[-2] if n.count('_') > 1 else n}={100 * float(d):.1f}"
            for n, d in zip(names, paired, strict=False)
        )
        print(
            f"  [pose_render] {side} tip error (cm, per pair): {per} | "
            f"mean {100 * float(paired.mean()):.1f} max {100 * float(paired.max()):.1f}",
            flush=True,
        )
        _report_pair_vs_nearest(nearest, paired, ref, rob, rob0, side)


def _report_pair_vs_nearest(nearest, paired, ref, rob, rob0, side):
    """per pair ≈ nearest means the finger correspondence is right (otherwise the finger order differs)."""
    print(
        f"  [pose_render] {side} nearest-tip mean {100 * float(nearest.mean()):.1f}cm "
        f"— minus per pair {100 * float(paired.mean() - nearest.mean()):+.1f}cm",
        flush=True,
    )
    # Split the error: if all five tips are off by the same vector, the wrist as a whole missed (arm IK or
    # mount), not the hand shape; the residual after removing the mean vector is the finger-shape difference.
    d0 = torch.norm(rob0 - ref, dim=-1)
    print(
        f"  [pose_render] {side} body-origin mean {100 * float(d0.mean()):.1f}cm "
        f"(site {100 * float(paired.mean()):.1f}cm)",
        flush=True,
    )
    dv = rob - ref  # (5,3)
    mean_v = dv.mean(dim=0)
    resid = torch.norm(dv - mean_v, dim=-1)
    print(
        f"  [pose_render] {side} split: rigid offset "
        f"({100 * float(mean_v[0]):+.1f}, {100 * float(mean_v[1]):+.1f}, "
        f"{100 * float(mean_v[2]):+.1f})cm |{100 * float(torch.norm(mean_v)):.1f}cm| "
        f"+ shape residual mean {100 * float(resid.mean()):.1f}cm",
        flush=True,
    )


def _still(raw_env, vname: str, center, dist: float, out_dir: str, tag: str) -> str:
    """One view: the front camera at `dist` from `center` along VIEWS[vname], STILL_RENDERS renders, then the frame
    read (frame guard) and the PNG. -> file name."""
    from dexengine.record.looks import look_guard as LG

    d3 = VIEWS.get(vname)
    if d3 is None:
        raise SystemExit(f"pose_render: unknown view {vname!r} (known: {', '.join(sorted(VIEWS))})")
    cam = raw_env.scene["front_camera"]
    vd = torch.tensor(d3, device=center.device, dtype=center.dtype)
    eye = center + vd / torch.norm(vd) * dist
    cam.set_world_poses_from_view(eyes=eye.unsqueeze(0), targets=center.unsqueeze(0))
    for _ in range(STILL_RENDERS):
        raw_env.sim.render()
        cam.update(0.0, force_recompute=True)
    LG.require("rl.render.pose_check frame read")
    rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    name = f"{tag}_{vname}.png"
    _save(rgb, os.path.join(out_dir, name))
    return name


def run_pose_render(raw_env, args_cli, out_dir: str, tag: str) -> None:
    """Reset env 0 to the reference pose at --pose_render_frame, draw the reference fingertips as green spheres, print
    the tip-error diagnostics and save one PNG per view in --pose_render_views."""
    os.makedirs(out_dir, exist_ok=True)
    device = raw_env.device
    eids = torch.tensor([0], device=device)
    tm = raw_env.trajectory_manager
    frame = int(getattr(args_cli, "pose_render_frame", 0) or 0)

    cid, seq = _pin_clip(raw_env, str(getattr(args_cli, "pose_render_seq", "") or ""))
    tm.start_frames[eids] = frame
    # The env's registered reset event terms run instead of the reset functions: their default arguments name one
    # arm's links and would be silently wrong for other combos, while the event terms carry the combo's arm, hand,
    # object keys and EE link (the same terms the balanced eval runs).
    raw_env.event_manager.apply(mode="reset", env_ids=eids, global_env_step_count=0)
    print(
        f"  [pose_render] '{tag}' clip {cid} '{seq}' frame={frame} reset "
        f"(arm, hand, object at the reference; event_manager reset terms)",
        flush=True,
    )

    # Reference fingertips as green spheres, same radius and colour as the render overlay.
    tips, why = _ref_fingertips_world(raw_env, seq, frame)
    if tips is None:
        print(f"  [pose_render] WARNING: fingertip overlay skipped — {why}", flush=True)
    else:
        tips_w, sides = tips
        import isaaclab.sim as _sim
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/mano_fingertips",
                markers={
                    "tip": _sim.SphereCfg(
                        radius=0.009, visual_material=_sim.PreviewSurfaceCfg(diffuse_color=(0.05, 0.95, 0.15))
                    )
                },
            )
        )
        marker.visualize(translations=tips_w)
        print(f"  [pose_render] fingertips {tips_w.shape[0]} points (sides: {', '.join(sides)})", flush=True)

    robot = raw_env.scene["robot"]
    import omni.usd

    from dexengine.record.looks import look_d as LOOK

    LOOK.apply(omni.usd.get_context().get_stage())  # the one canonical look (sets the frame-guard token)
    raw_env.sim.render()

    # Numbers before pictures: a wrong transform puts the green points tens of cm away from the robot.
    if tips is not None:
        _report_tip_error(raw_env, tips_w, sides)

    # Auto framing on the robot's body bounds: robot sizes differ a lot between combos.
    p = robot.data.body_pos_w[0]
    lo, hi = p.min(dim=0).values, p.max(dim=0).values
    center = (lo + hi) * 0.5
    radius = float(torch.norm(hi - lo)) * 0.5
    dist = max(radius * float(getattr(args_cli, "pose_render_zoom", 2.6) or 2.6), 0.8)

    want = [v.strip() for v in str(getattr(args_cli, "pose_render_views", "iso")).split(",") if v.strip()]
    saved = [_still(raw_env, vname, center, dist, out_dir, tag) for vname in want]

    z_min = float(robot.data.body_pos_w[0, :, 2].min())
    print(f"  [pose_render] {tag}: lowest body z={z_min:.3f}m — {len(saved)} saved: {', '.join(saved)}", flush=True)
