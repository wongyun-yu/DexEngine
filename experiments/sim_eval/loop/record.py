"""Episode recording and render-only outputs: the per-row recorder, the pixel gate, the review videos (head left eye and
exo window) and the reference object poses."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from experiments.sim_eval.loop.common import MARK, _mkdir, _say, box_down
from experiments.sim_eval.loop.control import Targets
from experiments.sim_eval.session_cfg import flag01 as _flag01


class Recorder:
    def _open_pair_views(self, sc, raw_env):
        """One PhysX contact view per ordered pair of the two main objects (rec_obj_pair)."""
        from isaacsim.core.simulation_manager import SimulationManager

        from dexengine.rl.tasks.mdp import hand_object_contact as HOC

        names = [n for n in ("object_right", "object_left") if n in sc.rigid_objects]
        if len(names) != 2:
            raise SystemExit(f"{MARK} rec_obj_pair=1 but the main objects are {names} — two needed")
        if int(raw_env.num_envs) != 1:  # the force matrix is summed over every sensor row — 'env 0' only with one env
            raise SystemExit(f"{MARK} rec_obj_pair=1 needs one env — num_envs {raw_env.num_envs}")
        sv = SimulationManager.get_physics_sim_view()
        for a, b in ((names[0], names[1]), (names[1], names[0])):
            ga, gb = HOC.object_globs(sc.rigid_objects[a]), HOC.object_globs(sc.rigid_objects[b])
            v = sv.create_rigid_contact_view(
                ga if len(ga) > 1 else ga[0],
                filter_patterns=[gb] * len(ga) if len(ga) > 1 else gb,
                max_contact_data_count=HOC.MAX_CONTACT_DATA_PER_PRIM * len(ga) * int(raw_env.num_envs),
            )
            self.pairs.append((f"{a}<-{b}", v))

    def _open_desk_views(self, sc, raw_env):
        """One PhysX object-desk contact view per main object and the desk box (rec_obj_desk)."""
        from isaacsim.core.simulation_manager import SimulationManager

        from dexengine.rl.tasks.mdp import hand_object_contact as HOC
        from dexengine.rl.tasks.scene import object_pin as OP

        names = [n for n in ("object_right", "object_left") if n in sc.rigid_objects]
        if not names:
            raise SystemExit(f"{MARK} rec_obj_desk=1 but no main object (object_right · object_left) in the scene")
        if int(raw_env.num_envs) != 1:
            raise SystemExit(f"{MARK} rec_obj_desk=1 needs one env — num_envs {raw_env.num_envs}")
        desk = [f"{p_}/{OP.DESK_PRIM}" for p_ in sc.env_prim_paths]
        if not sc.stage.GetPrimAtPath(desk[0]).IsValid():
            raise SystemExit(f"{MARK} rec_obj_desk=1 but the desk-top prim {desk[0]} does not exist")
        dg = HOC.env_glob(desk)
        sv = SimulationManager.get_physics_sim_view()
        for a_ in names:
            ga = HOC.object_globs(sc.rigid_objects[a_])
            fp = [[dg]] * len(ga) if len(ga) > 1 else [dg]
            cap = HOC.MAX_CONTACT_DATA_PER_PRIM * len(ga) * int(raw_env.num_envs)
            v = sv.create_rigid_contact_view(
                ga if len(ga) > 1 else ga[0], filter_patterns=fp, max_contact_data_count=cap
            )
            self.desk.append((a_, v))
        self.desk_box = OP.desk_box(raw_env)
        _say(
            f"[E5_REC_OBJ_DESK] on · objects {[n for n, _ in self.desk]} · desk top {dg} · box {self.desk_box}"
            " · column = |F| body sum (env 0) N"
        )

    def __init__(self, raw_env):
        from dexengine.rl.eval import traj_record as TR
        from dexengine.rl.tasks.mdp import hand_object_contact as HOC

        self.TR, self.env = TR, raw_env
        self.ents = TR.scene_entities(raw_env.scene)
        self.hoc = HOC.instance(raw_env)
        self.held = [(s, o) for s, o in raw_env.held_object_assets()]
        # force rows = one per held object — the judge finds them by entity name (episode entities' name). The entity
        # is found by asset identity; not found dies (a side name in place of the entity name made every Lift/Put judge
        # miss the forces).
        sc = raw_env.scene
        tables = {"artic": sc.articulations, "rigid": sc.rigid_objects}
        by_id = {}
        for e in self.ents:
            for n in [e["name"]] + list(e.get("aliases") or []):
                by_id[id(tables[e["kind"]][n])] = e["name"]
        self.held_entities = []
        for s_, o in self.held:
            if id(o) not in by_id:
                raise SystemExit(
                    f"{MARK} held object ({s_}) asset is not a recorded entity — cannot bind the force row to a name"
                )
            self.held_entities.append(by_id[id(o)])
        self.rows, self.targets, self.jpos, self.force = [], [], [], []
        # object–object contact force (Sweep physical contact measure, default off): with both main objects
        # (object_right · object_left), one PhysX contact view per ordered pair (sensor = first object's bodies,
        # filter = second's) —
        # |F| per row (body sum) [N]. On without two main objects dies. Marker [E5_REC_OBJ_PAIR].
        self.pairs, self.pair_force = [], []
        if _flag01("rec_obj_pair"):
            self._open_pair_views(sc, raw_env)
            _say(f"[E5_REC_OBJ_PAIR] on · pairs {[n for n, _ in self.pairs]} · column = |F| body sum (env 0) N")

        # object–desk contact (the v2 judge's support signal, default off):
        # per main object a PhysX contact view (sensor = the object's bodies, filter = the desk-top collider
        # object_pin.DESK_PRIM) — |F| per row (body sum, env 0) [N] + the desk box (env local, object_pin.desk_box).
        # A scene without a desk (desk_none) dies.
        # On without main objects or the desk-top prim dies. Marker [E5_REC_OBJ_DESK].
        self.desk, self.desk_force, self.desk_box = [], [], None
        if _flag01("rec_obj_desk"):
            self._open_desk_views(sc, raw_env)

    def add(self, tg: Targets) -> None:
        self.rows.append(self.TR.state_vector(self.env.scene, self.ents)[0].detach().cpu().numpy())
        self.targets.append(tg.flat()[0].detach().cpu().numpy())
        self.jpos.append(tg.joint_pos_flat()[0].detach().cpu().numpy())
        self.force.append([self.hoc.link_forces(o)[0].norm(dim=-1).detach().cpu().numpy() for _s, o in self.held])
        if self.pairs:
            dt = float(self.env.physics_dt)
            self.pair_force.append(
                [float(v.get_contact_force_matrix(dt).sum(dim=(0, 1)).norm()) for _n, v in self.pairs]
            )
        if self.desk:
            dt = float(self.env.physics_dt)
            self.desk_force.append(
                [float(v.get_contact_force_matrix(dt).sum(dim=(0, 1)).norm()) for _n, v in self.desk]
            )

    def save(self, path: Path, frames: dict, meta: dict) -> None:
        import numpy as np

        egoL, egoR = np.asarray(frames.get("egoL", []), np.uint8), np.asarray(frames.get("egoR", []), np.uint8)
        # ep_no_frames=1 (default off): oracle-replay screening sessions write no egoL/egoR frames into episode.npz —
        # judges and tables read states and forces only (zero-length arrays, shapes kept). The policy input is still
        # captured.
        if _flag01("ep_no_frames"):
            _say(
                f"[E5_EP_NO_FRAMES] {path.parent.name}: egoL {egoL.shape} · egoR {egoR.shape} frames not "
                "written to episode.npz"
            )
            egoL, egoR = egoL[:0], egoR[:0]
        np.savez_compressed(
            path,
            states=np.asarray(self.rows, np.float32),
            targets=np.asarray(self.targets, np.float32),
            joint_pos=np.asarray(self.jpos, np.float32),
            hand_obj_force=np.asarray(self.force, np.float32).reshape(len(self.force), len(self.held), -1),
            egoL=egoL,
            egoR=egoR,
            frame_row=np.asarray(frames.get("row", []), np.int64),
            entities=json.dumps(self.ents),
            meta=json.dumps(meta, ensure_ascii=False, default=float),
            **(
                {
                    "obj_pair_force": np.asarray(self.pair_force, np.float32),
                    "obj_pair_names": json.dumps([n for n, _ in self.pairs]),
                }
                if self.pairs
                else {}
            ),
            **(
                {
                    "obj_desk_force": np.asarray(self.desk_force, np.float32),
                    "obj_desk_names": json.dumps([n for n, _ in self.desk]),
                    "obj_desk_box": np.asarray([] if self.desk_box is None else self.desk_box, np.float32),
                }
                if self.desk
                else {}
            ),
        )


def _gate(S, T, out: Path, job: dict) -> dict:
    """Pixel gate — recorded states as is (no physics), rendered like the data (first image FIRST_WARMUP, then
    FRAME_RENDERS).
    """
    import numpy as np

    from dexengine.record.render import episode as CR

    t0 = time.perf_counter()
    keep_full = set(job.get("gate_full_frames", [0, 30, 60, 90, 120]))
    L, R, full = [], [], {}
    for i, st in enumerate(T["steps"]):
        T["write"](st)
        cap = S.rig.capture(CR.FIRST_WARMUP if i == 0 else CR.FRAME_RENDERS)
        L.append(box_down(cap["egoL"]))
        R.append(box_down(cap["egoR"]))
        if i in keep_full:
            full[f"egoL_{i}"], full[f"egoR_{i}"] = cap["egoL"], cap["egoR"]
    np.savez_compressed(
        out / "frames_gate.npz", egoL=np.asarray(L), egoR=np.asarray(R), frame_row=np.asarray(T["steps"]), **full
    )
    r = {"mode": "gate", "frames": len(L), "wall_s": round(time.perf_counter() - t0, 2)}
    _say(f"gate frames {len(L)} · {r['wall_s']} s → {out / 'frames_gate.npz'}")
    return r


#: Video for human review — recorded states written as is (no physics), the head's left eye rendered again. Size and
#: encoding use only these constants.
RENDER_VIDEO = {"width": 640, "height": 360, "fps": 30, "crf": 20}


#: exo.mp4 window (to see placement and pose): a box fixed in the env frame (desk fixed, robot x·y fixed, robot height
#: per clip) projected
# through the scene's exo camera (copy of the canonical front + dolly Δ, per scene); the bounding square is cropped to
# EXO_OUT². Box = DP rollout main-object centres (+5 cm margin): 97.3 % of episodes stay inside to the end, 99.0 % of
# points (measured on 986 episodes).
EXO_BOX_ENV = {"x": (-0.45, 0.0), "y": (-0.34, 0.26), "z": (0.74, 1.08)}


EXO_OUT = 720


def _exo_window(S, ew: int, eh: int) -> tuple[int, int, int]:
    """EXO_BOX_ENV's 8 corners → projection through this scene's exo camera → bounding square (x0, y0, side), pushed
    inside the image. Camera = the pose Rig.calibrate actually set (view_ref: front pose c0, R0, dolly Δ back along the
    optical axis, intrinsics Kf — exo is a copy of the front cfg). The exo camera's data.pos_w keeps the pre-dolly
    value
    after set_world_poses, so it is not used. A corner behind the camera, no view_ref or an image size ≠ WH dies."""
    import torch

    vr, e = S.rig.view_ref, S.rig.e
    if not vr or (int(vr["WH"][0]), int(vr["WH"][1])) != (ew, eh):
        raise SystemExit(f"{MARK} exo window: no Rig.view_ref or size {None if not vr else vr['WH']} ≠ {(ew, eh)}")
    Rm = torch.as_tensor(vr["R0"], dtype=torch.float64)
    c = (
        torch.as_tensor(vr["c0"], dtype=torch.float64) + float(vr["delta"]) * Rm[:, 2]
    )  # c1 = c0 − Δ·fwd0 · fwd0 = −R0[:, 2]
    K = torch.as_tensor(vr["Kf"], dtype=torch.float64)
    o = S.scene.env_origins[e].detach().cpu().double()
    B = (
        torch.tensor(
            [[x, y, z] for x in EXO_BOX_ENV["x"] for y in EXO_BOX_ENV["y"] for z in EXO_BOX_ENV["z"]],
            dtype=torch.float64,
        )
        + o
    )
    pc = (B - c) @ Rm  # OpenGL camera frame (−z forward · +y up)
    if bool((pc[:, 2] >= -1e-3).any()):
        raise SystemExit(f"{MARK} exo window: a box corner is behind the camera — camera {c.tolist()}")
    u = K[0, 2] + K[0, 0] * pc[:, 0] / -pc[:, 2]
    v = K[1, 2] - K[1, 1] * pc[:, 1] / -pc[:, 2]
    side = int(min(ew, eh, math.ceil(max(float(u.max() - u.min()), float(v.max() - v.min())) / 2) * 2))
    x0 = int(round(float(u.max() + u.min()) / 2 - side / 2))
    y0 = int(round(float(v.max() + v.min()) / 2 - side / 2))
    return min(max(x0, 0), ew - side), min(max(y0, 0), eh - side), side


def _recorded_rows(src, job, mode, P, T):
    """Rows of an earlier run (render-<batch>.<mode>): its states at its frame rows, entity columns checked against
    the trajectory; returns (columns, rows, source)."""
    import numpy as np
    import torch

    prefix, _, smode = src.partition(".")
    ep = (
        Path(job["out"]).parent / f"{prefix}_{Path(job['plan']).parent.name}" / smode / "episode.npz"
    )  # run = <batch>_<job folder>
    if not ep.is_file():
        raise SystemExit(f"{MARK} {mode}: source episode {ep} not found")
    z = np.load(ep)
    ents = {e["name"]: e for e in json.loads(str(z["entities"]))}
    st = torch.as_tensor(z["states"], dtype=P.dtype, device=P.device)
    cols = []
    for e in T["tf"].entities:
        g = ents.get(e["name"])
        if g is None:
            continue  # only in the trajectory (static extra objects) — trajectory rows as is
        if (g["hi"] - g["lo"]) != (e["hi"] - e["lo"]) or list(g.get("joint_names") or []) != list(
            e.get("joint_names") or []
        ):
            raise SystemExit(f"{MARK} {mode}: entity {e['name']} width or joints differ from the trajectory")
        cols.append((e["lo"], e["hi"], g["lo"], g["hi"]))
    miss = sorted(set(ents) - {e["name"] for e in T["tf"].entities})
    if miss:
        raise SystemExit(f"{MARK} {mode}: episode entities {miss} are not in the trajectory")
    rows = [(int(r), st[int(r)]) for r in z["frame_row"]]
    source = str(ep)
    return cols, rows, source


def _open_head_encoder(fps, W, H, crf, vid):
    """ffmpeg encoder of the head left-eye video (RENDER_VIDEO size)."""
    import subprocess

    ff = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "1280x720",
            "-r",
            str(fps),
            "-i",
            "-",
            "-vf",
            f"scale={W}:{H}:flags=area",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(vid),
        ],
        stdin=subprocess.PIPE,
    )
    return ff


def _overlay_recorded(ov, cols, row, S):
    """Write the recorded entity columns over the trajectory row (episode physics frame → render frame)."""
    for lt, ht, le, he in cols:
        row[lt:ht] = ov[le:he]
        if (
            S._e5_swap.dz is not None
            # episode state = physics (stage 1) frame → record render frame (root z − dz; trajectory rows P
            # are already in the render frame)
        ):
            row[lt + 2] -= S._e5_swap.dz


def _open_exo_encoder(ex, S, fps, crf, od):
    """ffmpeg encoder of the exo window, opened at the first exo image size; returns (window, encoder, exo size)."""
    import subprocess

    eh, ew = ex.shape[:2]
    exo_win = _exo_window(S, ew, eh)
    _say(
        f"[E5_EXO] window x0={exo_win[0]} y0={exo_win[1]} side={exo_win[2]} / {ew}x{eh} → "
        f"{EXO_OUT}² · box (env) {EXO_BOX_ENV}"
    )
    fx = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{ew}x{eh}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-vf",
            f"crop={exo_win[2]}:{exo_win[2]}:{exo_win[0]}:{exo_win[1]},scale={EXO_OUT}:{EXO_OUT}:flags=area",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(od / "exo.mp4"),
        ],
        stdin=subprocess.PIPE,
    )
    exo_hw = (eh, ew)
    return exo_win, fx, exo_hw


def _close_encoders(ff, vid, mode, fx, od):
    """Close both encoders; a failed encoder or a missing video dies."""
    ff.stdin.close()
    if ff.wait() != 0 or not vid.is_file():
        raise SystemExit(f"{MARK} {mode}: ffmpeg failed")
    if fx is not None:
        fx.stdin.close()
        if fx.wait() != 0 or not (od / "exo.mp4").is_file():
            raise SystemExit(f"{MARK} {mode}: exo ffmpeg failed")


def _render_states(S, T, out: Path, job: dict, mode: str) -> dict:
    """Mode `render-data` = the data's physics rollout (trajectory rows, written like the pixel gate) ·
    `render-<batch>.<mode>` = the recorded states of an earlier run (<runs>/<batch>_<job folder name>/<mode>/episode.npz
    — for a variant job.clip ≠ the folder) at its frame rows; entities only in the trajectory (static extra objects)
    use the trajectory row of the same time. Different entity names, widths or joint names die. Output
    <out>/<mode>/video.mp4 (head left eye, RENDER_VIDEO) and exo.mp4 (fixed outside camera collect_exo_front — the
    image the same capture already
    drew; window = _exo_window, EXO_BOX_ENV projected bounding square → EXO_OUT²)."""

    import numpy as np

    from dexengine.record.render import episode as CR
    from dexengine.rl.eval import traj_record as TR

    t0 = time.perf_counter()
    src = mode[len("render-") :]
    P = T["P"]  # trajectory rows after body fitting (scene frame) [L, D]
    if src == "data":
        rows = [(int(t), None) for t in T["steps"]]
        source = "data_stage1(trajectory rows)"
    else:
        cols, rows, source = _recorded_rows(src, job, mode, P, T)
    od = _mkdir(out / mode)
    W, H, fps, crf = (RENDER_VIDEO[k] for k in ("width", "height", "fps", "crf"))
    vid = od / "video.mp4"
    ff = _open_head_encoder(fps, W, H, crf, vid)
    n = 0
    # exo encoder — opened at the first image's size (outside camera = copy of the canonical front, size from its cfg)
    fx = None
    for i, (r, ov) in enumerate(rows):
        row = P[min(r, len(P) - 1)].clone()
        if ov is not None:
            _overlay_recorded(ov, cols, row, S)
        TR.write_states(S.scene, T["binds"], row[None], S.eids)
        S.env.sim.physics_sim_view.update_articulations_kinematic()
        cap = S.rig.capture(CR.FIRST_WARMUP if i == 0 else CR.FRAME_RENDERS)
        img = np.ascontiguousarray(cap["egoL"][..., :3], dtype=np.uint8)
        if img.shape != (720, 1280, 3):
            raise SystemExit(f"{MARK} {mode}: image {img.shape} ≠ (720, 1280, 3)")
        ff.stdin.write(img.tobytes())
        ex = cap.get("exo")
        if ex is None:
            raise SystemExit(f"{MARK} {mode}: capture has no exo image (collect_exo_front)")
        ex = np.ascontiguousarray(ex[..., :3], dtype=np.uint8)
        if fx is None:
            exo_win, fx, exo_hw = _open_exo_encoder(ex, S, fps, crf, od)
        elif ex.shape[:2] != exo_hw:
            raise SystemExit(f"{MARK} {mode}: exo image size changed {ex.shape[:2]} ≠ {exo_hw}")
        fx.stdin.write(ex.tobytes())
        n += 1
    _close_encoders(ff, vid, mode, fx, od)
    r = dict(
        mode=mode,
        source=source,
        frames=n,
        video=str(vid),
        wall_s=round(time.perf_counter() - t0, 2),
        **RENDER_VIDEO,
        exo=str(od / "exo.mp4") if fx is not None else None,
        exo_src_hw=list(exo_hw) if fx is not None else None,
        exo_window=list(exo_win) if fx is not None else None,
        exo_box_env=EXO_BOX_ENV,
        exo_out=EXO_OUT,
    )
    (od / "render.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
    _say(f"[E5_RENDER] {mode}: frames {n} · {r['wall_s']} s · source {source} → {vid} · exo {r['exo']}")
    return r


def _reference(traj_mgr, out: Path) -> dict:
    """Reference (kinematic) object poses of every frame — judge input. Env-local frame (the frame of
    traj_record.state_vector, same rotation path as trajectory_manager._ref_state_at_abs). Rows = reference frames (50
    Hz,
    clip_lengths) · right and left object."""
    import numpy as np
    import torch

    L = int(traj_mgr.clip_lengths[0])
    keys = ("right_obj_pos", "right_obj_quat", "left_obj_pos", "left_obj_quat")
    acc = {k: [] for k in keys}
    n = int(traj_mgr.clip_ids.shape[0])
    for f in range(L):
        st = traj_mgr._ref_state_at_abs(torch.full((n,), f, dtype=torch.long, device=traj_mgr.clip_ids.device))
        for k in keys:
            acc[k].append(st[k][0].detach().float().cpu().numpy())
    arr = {k: np.asarray(v, np.float32) for k, v in acc.items()}
    np.savez_compressed(out / "reference.npz", **arr, clip_len=np.int64(L))
    _say(f"reference object poses {L} frames → {out / 'reference.npz'}")
    return {"frames": L}
