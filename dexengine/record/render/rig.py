"""Record camera rig (inside Kit): fit and capture the ego ZED Mini eyes, left-eye depth and exo in a render session.

The camera prims are added to the scene cfg by render.cameras.add_collect_cameras (gym.make hook). The ZED Mini housing
mesh is already in the render USD (render body builder, dexengine/robots/usd/head_camera.py; the session does not
change meshes). This file fits those cameras with
  1. automatic head pitch (render.cameras.head_pitch_for_desk; the camera hangs on the head link, so a fixed-point
     iteration) · aerial scenes set the pitch so the clip trajectory's hand and object points are in the left-eye
     image (yaw fixed at 0: render.framing.FRAME_YAW, render.framing.frame; the desk rule is for desk scenes only)
  2. a measured stereo check (baseline, orthogonal optical axes, left eye = robot's left)
  3. the exo dolly (render.cameras.exo_dolly: upper-body keypoints inside the margin)
and reads them per frame with capture(). The stage-2 replay render (render.episode) uses this rig.
  4. the fixed per-clip session state (unused cameras off, fixed-camera check) is render.clip_session; Rig holds only
     the update list (active).
"""

from __future__ import annotations

import math

from dexengine.core.errors import InputMissingError
from dexengine.record.render import cameras as CC

STILL_WARMUP = 16  # same count as the canonical RENDER_STILL_WARMUP default (RTX time accumulation)
CAM_KEYS = ("head_camera", "front_camera", "collect_ego_left", "collect_ego_right", "collect_exo_front")
# eyes drawn at render size, cropped to the calibrated window (CC.crop_ego)
EGO_KEYS = ("collect_ego_left", "collect_ego_right")
#: camera tilt note stored in the render meta (data)
TILT_NOTE = (
    "intended real-robot calibration tilt — the mount rpy of the calibrated head-camera chain as is (same pose "
    "as the real robot)"
)


def collect_scene_look() -> dict:
    """Scene look of the record render = the canonical background and lighting (the dump env's DEXENGINE_AUG_*:
    record.looks.render_look, the same entry scene_look.apply_from_env as the canonical clip render, no copy) plus
    the record render look (looks.look_d.LOOK, look D). A knob applied short or a missing look asset dies; the markers
    [VISUAL_AUG] and [COLLECT_LOOK] are checked again in the session log. Returns the look meta (render_meta
    render_look)."""
    import omni.usd

    from dexengine.record.looks import look_d as collect_look
    from dexengine.rl.render import scene_look

    stage = omni.usd.get_context().get_stage()
    scene_look.apply_from_env(stage)
    return collect_look.apply(stage)


def hide_markers(raw_env) -> None:
    for mn in (
        "_marker_wrist_current_frame",
        "_marker_wrist_target_frame",
        "_marker_obj_current_frame",
        "_marker_obj_target_frame",
        "_marker_wrist_current_frame_left",
        "_marker_wrist_target_frame_left",
        "_marker_obj_current_frame_left",
        "_marker_obj_target_frame_left",
    ):
        m = getattr(raw_env, mn, None)
        if m is not None:
            m.set_visibility(False)


def _fit_pitch(p, d, kz, meas, hist, joint, self, groups, canon_meas):
    """Fixed point of the head joint: the measured optical pitch meets the desk pitch rule (6 tries, else dies); then
    the head framing and pinned heads."""
    for it in range(6):
        rule = CC.head_pitch_for_desk(p[0], p[2], d["near_x"], d["far_x"], d["top_z"], kz["vfov_deg"])
        err = rule["pitch_deg"] - meas
        hist.append(
            {"iter": it, "joint_deg": joint, "measured_optical_pitch_deg": meas, "required_deg": rule["pitch_deg"]}
        )
        if abs(err) < 0.05:
            break
        joint += err
        self.set_head(joint)
        self.render(2)
        p, _q, _f, meas = self.pose("collect_ego_left")
    else:
        raise SystemExit(f"[COLLECT_CAM] pitch fixed point did not converge: {hist}")
    self.frame_head(groups, kz, joint)
    self.set_heads(self.pinned)
    self.render(2)
    p, _q, _f, meas = self.pose("collect_ego_left")
    print(
        f"[COLLECT_CAM] automatic pitch rule={rule['rule']} -> optical axis {rule['pitch_deg']:.2f}° (measured "
        f"{meas:.2f}°) · joint {self.head_joint}={joint:.2f}° (canonical {self.default_deg:.1f}° · canonical ego "
        f"optical axis {canon_meas:.2f}°) · iterations {len(hist)}",
        flush=True,
    )
    return rule


def _check_stereo(self, kz):
    """Measured stereo layout: baseline length, orthogonal optical axes, left eye on +y; the left intrinsics."""
    import numpy as np

    pl, ql, fl, _ = self.pose("collect_ego_left")
    pr, qr, _fr, _ = self.pose("collect_ego_right")
    base = pr - pl
    left_y = float(np.dot(pl - pr, np.array([0.0, 1.0, 0.0])))
    ortho = float(np.dot(base / np.linalg.norm(base), fl))
    if not (abs(np.linalg.norm(base) - kz["baseline_m"]) < 1e-3 and abs(ortho) < 1e-3 and left_y > 0):
        raise SystemExit(
            f"[COLLECT_CAM] measured stereo layout mismatch: |B|={np.linalg.norm(base):.4f} "
            f"ortho={ortho:.4f} left_y-right_y={left_y:.4f}"
        )
    K_l = self.cams["collect_ego_left"].data.intrinsic_matrices[self.e].detach().cpu().numpy().tolist()
    print(
        f"[COLLECT_CAM] measured stereo |B|={np.linalg.norm(base) * 1000:.2f} mm · optical axes orthogonal "
        f"{ortho:+.2e} · left y - right y = {left_y * 1000:+.1f} mm",
        flush=True,
    )
    return pl, ql, fl, pr, qr, K_l


def _exo_dolly(self, pl, pr, d, kz):
    """Exo front camera dolly so the upper-body links and both eyes fit; the view reference for the framing checks."""
    import numpy as np
    import torch

    import isaaclab.utils.math as mu

    fc = self.cams["front_camera"]
    c0 = fc.data.pos_w[self.e].detach().cpu().numpy().astype(float)
    q0 = fc.data.quat_w_opengl[self.e].detach().cpu().numpy().astype(float)
    R0 = mu.matrix_from_quat(torch.tensor(q0).unsqueeze(0))[0].numpy()
    fwd0 = -R0[:, 2]
    Kf = fc.data.intrinsic_matrices[self.e].detach().cpu().numpy()
    W, H = fc.cfg.width, fc.cfg.height
    names, P_up = self._upper_points()
    pads = np.array([CC.EXO_PAD_HEAD_M if ("head" in n or "camera" in n) else CC.EXO_PAD_BODY_M for n in names])
    P = np.vstack([P_up, pl, pr])
    pads = np.concatenate([pads, [CC.EXO_PAD_HEAD_M, CC.EXO_PAD_HEAD_M]])
    delta = CC.exo_dolly((R0.T @ (P - c0).T).T, pads, float(Kf[0, 0]), float(Kf[1, 1]), W, H)
    c1 = c0 - delta * fwd0
    self.cams["collect_exo_front"].set_world_poses(
        torch.tensor(c1, dtype=torch.float32, device=self.dev).unsqueeze(0),
        torch.tensor(q0, dtype=torch.float32, device=self.dev).unsqueeze(0),
        convention="opengl",
        env_ids=self.eids,
    )
    # exo world place (used by readers that skip an off camera's data)
    self.exo_pos_w = np.asarray(c1, float).copy()
    print(f"[COLLECT_CAM] exo dolly Δ={delta:.4f} m · upper-body links {len(names)} + both ZED eyes", flush=True)
    self.view_ref = {
        "upper": names,
        "P": P,
        "desk": {k: d[k] for k in ("near_x", "far_x", "top_z", "cy")},
        "framing": dict(self.framing or {}),
        "kz": kz,
        "pads": pads,
        "delta": delta,
        "c0": c0,
        "R0": R0,
        "Kf": Kf,
        "WH": (W, H),
    }
    return c0, q0, Kf, W, H, names, delta, c1


class Rig:
    def __init__(self, raw_env, args_cli, env_id: int = 0):
        import numpy as np
        import torch

        import omni.usd
        from pxr import Usd, UsdGeom

        from isaaclab.assets import Articulation

        from dexengine.core.arms import active_arm_spec
        from dexengine.rl.env_setup.objects import AERIAL_HIDDEN_Z  # owner of how far aerial scenes sink the desk

        self.env, self.scene, self.dev = raw_env, raw_env.scene, raw_env.device
        if args_cli is None:
            raise InputMissingError("[COLLECT_CAM] Rig has no args_cli: aerial scene or not (scene_aerial) is unknown")
        # work surface = desk top: when an aerial scene sinks the desk by AERIAL_HIDDEN_Z the head pitch still looks at
        # the top before sinking (otherwise the optical axis is at 86°, looking at its own torso)
        self.desk_dz = float(AERIAL_HIDDEN_Z) if int(getattr(args_cli, "scene_aerial", 0) or 0) else 0.0
        self.body = None  # body meta (read by desk)
        self.e = env_id
        self.eids = torch.tensor([env_id], device=self.dev)
        self.cams = {k: self.scene[k] for k in CAM_KEYS}
        self.active = list(CAM_KEYS)  # cameras render updates (after cams_off: only the cameras read)
        self.off = None  # record of cameras turned off (no calibrate after it)
        self.view_ref = None  # calibrate reference (render.clip_session.check_view)
        pj = active_arm_spec().head_joint_names[-1]
        self.robot = self.jidx = None
        for k in self.scene.keys():  # noqa: SIM118  # InteractiveScene is not a dict: iteration needs keys()
            a = self.scene[k]
            if isinstance(a, Articulation) and pj in a.joint_names:
                self.robot, self.jidx = a, a.joint_names.index(pj)
                break
        if self.robot is None:
            raise InputMissingError(f"[COLLECT_CAM] no articulation has head pitch joint {pj}")
        self.head_joint = pj
        self.default_deg = math.degrees(float(self.robot.data.default_joint_pos[env_id, self.jidx]))
        self.auto_deg = None
        # every head joint (measured) · pinned joints {name: rad} (set by calibrate; the others take trajectory values)
        self.head_names = tuple(active_arm_spec().head_joint_names)
        self.head_ids = [self.robot.joint_names.index(n) for n in self.head_names]
        self.pinned = {}
        if len(self.head_names) != 2 or self.head_names[-1] != pj:
            raise SystemExit(f"[COLLECT_CAM] head joints {self.head_names}: must be two (yaw, pitch) (render.framing)")
        self.yaw_joint = self.head_names[0]
        self.canon = {
            n: float(self.robot.data.default_joint_pos[env_id, i])
            for n, i in zip(self.head_names, self.head_ids, strict=True)
        }
        self.framing = None
        stage = omni.usd.get_context().get_stage()

        def usd_T(path):
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise InputMissingError(f"[COLLECT_CAM] USD prim missing: {path}")
            return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=float).T

        # camera pose = PhysX link pose ∘ fixed (link -> camera prim) transform; cam.data.pos_w (USD) does not get
        # joints written without a physics step (measured).
        self.rel = {}
        for key in ("collect_ego_left", "collect_ego_right", "head_camera"):
            cp = self.cams[key].cfg.prim_path.replace("{ENV_REGEX_NS}", f"/World/envs/env_{env_id}")
            cp = cp.replace("env_.*", f"env_{env_id}")
            segs = cp.split("/")
            cand = [s for s in segs if s in self.robot.body_names]
            if not cand:
                raise InputMissingError(f"[COLLECT_CAM] no articulation link on path {cp}")
            lp = "/".join(segs[: segs.index(cand[-1]) + 1])
            self.rel[key] = (self.robot.body_names.index(cand[-1]), np.linalg.inv(usd_T(lp)) @ usd_T(cp))

    def set_head(self, deg: float) -> None:
        self.set_heads({self.head_joint: math.radians(deg)})

    def set_heads(self, rad: dict) -> None:
        """Write head joints {name: rad} (state and target)."""
        import torch

        ids = [self.robot.joint_names.index(n) for n in rad]
        pos = torch.tensor([[float(v) for v in rad.values()]], device=self.dev)
        self.robot.write_joint_state_to_sim(pos, torch.zeros_like(pos), joint_ids=ids, env_ids=self.eids)
        self.robot.set_joint_position_target(pos, joint_ids=ids, env_ids=self.eids)

    def _ego_at(self, q):
        """Left eye (pos, q_ros, fwd) of the head pose (yaw, pitch) rad: the pose_fn of render.framing."""
        self.set_heads({self.yaw_joint: q[0], self.head_joint: q[1]})
        self.render(2)
        p, qr, f, _ = self.pose("collect_ego_left")
        return p, qr, f

    def render(self, n: int = STILL_WARMUP) -> None:
        for _ in range(1 + n):
            self.env.sim.render()
        for k in self.active:
            self.cams[k].update(0.0, force_recompute=True)

    def pose(self, key):
        import numpy as np
        import torch

        import isaaclab.utils.math as mu

        li, rel_T = self.rel[key]
        self.robot.update(1e-6)  # joints written without a physics step -> push the buffer timestamps to reread PhysX
        Tl = np.eye(4)
        Tl[:3, :3] = mu.matrix_from_quat(self.robot.data.body_quat_w[self.e, li].unsqueeze(0))[0].cpu().numpy()
        Tl[:3, 3] = self.robot.data.body_pos_w[self.e, li].detach().cpu().numpy().astype(float)
        Tc = Tl @ rel_T
        Rc = Tc[:3, :3] / np.linalg.norm(Tc[:3, :3], axis=0)  # remove the USD scale
        fwd = -Rc[:, 2]  # USD camera optical axis = -Z
        q_ros = mu.quat_from_matrix(torch.tensor(Rc @ np.diag([1.0, -1.0, -1.0])).unsqueeze(0))[0].numpy()
        return Tc[:3, 3].copy(), q_ros.astype(float), fwd, math.degrees(math.asin(-float(fwd[2])))

    def rgb(self, key):
        import numpy as np

        if key not in self.active:
            raise InputMissingError(f"[COLLECT_CAM] {key} is a camera that is off: it cannot be read")
        fr = self.cams[key].data.output["rgb"][self.e].detach().cpu().numpy()
        if fr.dtype != np.uint8:
            fr = (fr * 255).clip(0, 255).astype(np.uint8)
        return CC.crop_ego(fr[..., :3]) if key in EGO_KEYS else fr[..., :3]

    def depth(self, key="collect_ego_left"):
        return CC.crop_ego(
            self.cams[key].data.output["distance_to_image_plane"][self.e].detach().cpu().numpy().squeeze()
        )

    def desk(self, raw: bool = False) -> dict:
        from dexengine.core import desk as DP
        from dexengine.record.render import desk_stretch as DS

        origin = self.scene.env_origins[self.e].detach().cpu().numpy().astype(float)
        top = getattr(self.env.cfg.scene, "desk_collision_top", None)
        cx = (top.init_state.pos[0] if top is not None else DP.DESK_X_TASK) + origin[0]
        tz = (
            top.init_state.pos[2] + DP.DESK_TOP_THICK / 2 - self.desk_dz if top is not None else DP.DESK_TOP_SURFACE_Z
        ) + origin[2]
        hx, hy = DP.desk_half()
        cy = (top.init_state.pos[1] if top is not None else 0.0) + origin[1]
        d = {
            "origin": origin,
            "near_x": cx - hx,
            "far_x": cx + hx,
            "top_z": tz,
            "half_x": hx,
            "half_y": hy,
            "cy": cy,
            "aerial_desk_dz": self.desk_dz,
        }
        return d if raw else DS.desk_after(d, self.body)

    def desk_local(self, raw: bool = False) -> dict:
        """desk() in env coordinates (supporting surface of the support pillars)."""
        d, o = self.desk(raw), self.desk(raw)["origin"]
        return dict(d, near_x=d["near_x"] - o[0], far_x=d["far_x"] - o[0], cy=d["cy"] - o[1], top_z=d["top_z"] - o[2])

    def frame_head(self, groups: dict | None, kz: dict, joint: float) -> None:
        """Pinned head joints and the framing meta (head_rule). Desk scene = desk fill pitch joint (deg), yaw 0; when
        the hand or object in-frame ratio in that pose is below render.framing.DESK_FILL_MIN_IN it switches to
        trajectory framing · aerial scene = trajectory framing (render.framing.frame: pitch only, yaw canonical 0,
        starting from the desk pitch)."""
        from dexengine.record.render import framing as CF  # numpy: after AppLauncher only ([PRE_APP])

        self.auto_deg = joint
        self.pinned = {self.head_joint: math.radians(joint)}
        if groups is None:
            if self.desk_dz:
                raise InputMissingError("[COLLECT_CAM] aerial scene framing has no trajectory points (groups)")
            return
        self.set_heads(self.pinned)
        self.render(2)
        p, q, _f, _ = self.pose("collect_ego_left")
        fill = CF.stats(groups, p, q, kz)
        scene = "aerial" if self.desk_dz else "desk"
        fr = None
        if self.desk_dz or CF.needs_trajectory(fill):
            fr = CF.frame(self._ego_at, groups, kz, (self.canon[self.yaw_joint], math.radians(joint)))
        gfail = None if self.desk_dz else CF.grasp_unfit(fill, fr)
        if fr is not None and gfail is None:
            self.pinned = {self.yaw_joint: fr["yaw"], self.head_joint: fr["pitch"]}
            self.auto_deg = math.degrees(fr["pitch"])
            self.framing = dict(
                fr,
                scene=scene,
                head_rule="trajectory",
                desk_fill=fill,
                yaw_deg=math.degrees(fr["yaw"]),
                pitch_deg=self.auto_deg,
            )
        else:
            self.framing = {
                "rule": "desk",
                "scene": scene,
                "head_rule": "desk_fill",
                "before": fill,
                "after": fill,
                "yaw_deg": 0.0,
                "pitch_deg": joint,
                "grasp_fail": gfail,
            }
        print(
            f"[COLLECT_FRAME] {scene} head_rule={self.framing['head_rule']} rule={self.framing['rule']} · yaw "
            f"{'free' if self.framing.get('yaw_free') else 'fixed'} · head "
            f"{ {n: round(math.degrees(v), 2) for n, v in self.pinned.items()} } ° · in-frame desk fill {fill} -> "
            f"after {self.framing['after']}",
            flush=True,
        )

    def calibrate(self, groups: dict | None = None) -> dict:
        """In the current scene state (after the caller wrote the first frame): automatic pitch, measured stereo, exo
        dolly. Returns a meta dict. groups = the clip trajectory's points (render.framing.frame: 'all', 'core', 'hand',
        'object' in world coordinates); required for aerial scenes (they set the pitch; yaw canonical 0); for desk
        scenes only the in-frame ratios are measured."""

        if self.off is not None:
            raise SystemExit("[COLLECT_CAM] calibrate after the cameras were turned off: front_camera cannot be read")
        d, kz = self.desk(), CC.zed_intrinsics()
        self.set_head(self.default_deg)
        self.render(2)
        _, _, _, canon_meas = self.pose("head_camera")
        p, _q, _f, meas = self.pose("collect_ego_left")
        joint, hist = self.default_deg, []
        rule = _fit_pitch(p, d, kz, meas, hist, joint, self, groups, canon_meas)
        pl, ql, fl, pr, qr, K_l = _check_stereo(self, kz)
        c0, q0, Kf, W, H, names, delta, c1 = _exo_dolly(self, pl, pr, d, kz)
        o = d["origin"]
        return {
            "zed": kz,
            "zed_mount": {k: v[0] for k, v in CC.ZEDM_MOUNT.items()},
            "zed_mount_tilt": {
                "roll_rad": CC.ZEDM_MOUNT["mount_rpy"][0][0],
                "pitch_rad": CC.ZEDM_MOUNT["mount_rpy"][0][1],
                "yaw_rad": CC.ZEDM_MOUNT["mount_rpy"][0][2],
                "note": TILT_NOTE,
            },
            "zed_rig": CC.zedm_rig(),
            "K_left_isaac_render": K_l,
            "K_left": [[kz["fx"], 0.0, kz["cx"]], [0.0, kz["fy"], kz["cy"]], [0.0, 0.0, 1.0]],
            "K_exo": Kf.tolist(),
            "exo_res": [W, H],
            "ego_left_pose": {"pos": (pl - o).tolist(), "quat_ros_wxyz": ql.tolist(), "fwd": fl.tolist()},
            "ego_right_pose": {"pos": (pr - o).tolist(), "quat_ros_wxyz": qr.tolist()},
            "head_joint": self.head_joint,
            "head_joint_canon_deg": self.default_deg,
            "canon_ego_pitch_measured_deg": canon_meas,
            "head_joint_auto_deg": self.auto_deg,
            "pitch_rule": rule,
            "head_pinned_deg": {n: math.degrees(v) for n, v in self.pinned.items()},
            "framing": self.framing,
            "pitch_iters": hist,
            "desk": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in d.items()},
            "exo": {
                "delta_m": delta,
                "canon_pos": (c0 - o).tolist(),
                "pos": (c1 - o).tolist(),
                "quat_opengl_wxyz": q0.tolist(),
                "margin_frac": CC.EXO_MARGIN_FRAC,
                "upper_links": names,
            },
        }

    def _upper_points(self):
        """Upper-body points of the exo dolly (world): robot articulation links whose name contains an
        EXO_UPPER_BODY_KEYS entry."""
        import numpy as np

        from isaaclab.assets import Articulation

        arts = [
            self.scene[k]
            for k in self.scene.keys()  # noqa: SIM118  # InteractiveScene is not a dict: iteration needs keys()
            if isinstance(self.scene[k], Articulation) and not k.startswith("object")
        ]
        names, pts = [], []
        for a in arts:
            a.update(1e-6)
            ids = [i for i, n in enumerate(a.body_names) if any(k in n for k in CC.EXO_UPPER_BODY_KEYS)]
            names += [a.body_names[i] for i in ids]
            pts.append(a.data.body_pos_w[self.e, ids].detach().cpu().numpy().astype(float))
        if not names:
            raise SystemExit(f"[COLLECT_CAM] upper-body links not found (keys {CC.EXO_UPPER_BODY_KEYS})")
        return names, np.vstack(pts)

    def head_readback(self) -> list[float]:
        """Head joint values as drawn now (rad, head_names order), read back from PhysX (measured, not commanded)."""
        self.robot.update(1e-6)
        q = self.robot.data.joint_pos[self.e]
        return [float(q[i]) for i in self.head_ids]

    def head_meta(self, frames: list) -> dict:
        """The head of render_meta: owner of the stored state's head joints (store.meta.render_head reads it)."""
        return {"joints": list(self.head_names), "pinned_rad": dict(self.pinned), "frame_rad": frames}

    def capture(self, n: int = STILL_WARMUP) -> dict:
        """Draw the current state (head = automatic pitch) n more times, then capture egoL, egoR (RGB uint8), depth_m
        (left eye, float) and exo."""
        self.set_heads(self.pinned)
        self.render(n)
        out = {"egoL": self.rgb("collect_ego_left"), "egoR": self.rgb("collect_ego_right"), "depth_m": self.depth()}
        if CC.exo_on():  # exo off (render.cameras.EXO_ENV=0): that camera's render output is off; not read
            out["exo"] = self.rgb("collect_exo_front")
        return out
