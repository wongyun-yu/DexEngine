"""Record stage 2: write one trajectory chosen by stage 1 without physics (kinematic replay) and capture both ZED Mini
eyes, left-eye depth and exo.

Entry = dexengine/record/session/entry.py --stage render (the canonical kinematic replay dump, one-clip session; this
function replaces that clip's render function). Scene = the canonical render (render USD, background, quality; the
same argv/env as canonical RENDER_ONLY=replay) + the record cameras.
Writing = the canonical replay's entity binding and write-back (rl.eval.traj_record.bind, write_states; no copy):
file entities and scene entities are matched by name and joint-name set, a mismatch dies.
Time: the record has one row per policy step (50 Hz); video frame i (30 fps) uses the step closest to time i/30,
round(i/30/step_dt) (nearest physics step). The frame -> step table is kept in the output meta.
Output (<COLLECT_OUT>/): egoL.mp4 · egoR.mp4 · exo.mp4 (H.264, 30 fps) · depth_mm.npy.zst (uint16 [N, H, W], 0 =
invalid; format owner render.writer: lossless zstd, round trip checked) · render_meta.json (camera calibration,
frame -> step, counts, head = the head joints measured per frame (owner of the head joint values of the stored state,
store.meta.render_head), extras_visual = extra objects put in and dropped with reasons (rule owner scene.extras)).
Writing = render.writer (background thread; per-clip renders overlap with the next episode's render at depth 1; the
bytes equal the old inline write).
Markers [COLLECT_RENDER] … DONE frames=<n>/<n> … · the line before: [COLLECT_RENDER] time … (seconds per step; output
only, no effect on output bytes) · the line before that: [COLLECT_RENDER] background write (…) … (proof that the
write ran in the background, wait and overlap seconds).
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from dexengine.core.errors import InputMissingError
from dexengine.record.looks import look_d as LOOK
from dexengine.record.render import body as BODY
from dexengine.record.render import cameras as CC
from dexengine.record.render import episode_frames as EFR  # frame loop and render_meta of one episode
from dexengine.record.render import idle as IDLE
from dexengine.record.render import markers as PM  # P&P place markers (default off, DENSE_PLACE_MARKER)
from dexengine.record.render import rig as RIG
from dexengine.record.render import track as TRK  # owner of the track points (track_points) and their cache
from dexengine.record.render import visibility as HV  # owner of the hand-to-head image marking (default on)
from dexengine.record.render import visibility as VIS
from dexengine.record.render import writer as CW  # owner of the mp4 and depth write layer
from dexengine.record.scene import extras_apply as EXV  # per-trajectory extra objects (rule owner: scene.extras)
from dexengine.record.scene import supports as SUP
from dexengine.record.session import runner as SR

FPS = CC.ZEDM_CALIB["fps"][0]
FIRST_WARMUP = 32  # first frame: the same count as the canonical replay render WARMUP_RENDERS (RTX time convergence)
FRAME_RENDERS = 0  # extra renders per frame: 0 = one render per frame (100 -> 61 ms per frame)
VIDEOS = CC.videos()  # owner of the exo switch: render.cameras (default on = ("egoL", "egoR", "exo"))
DEPTH_FILE = CW.DEPTH_FILE
#: heavy render outputs (byte copies in the dataset): the stage-2 cache is reused only when all exist (record.bundle)
#: · the names the bundle cleanup (record.clean) removes
MEDIA = (*(f"{v}.mp4" for v in VIDEOS), DEPTH_FILE)


def media_files(d: Path) -> tuple[str, ...]:
    """Heavy output names of that render folder: MEDIA, plus the head-view file when the render had hand-to-head
    marking on (render_meta head_view). The stage-2 cache check (reuse only when complete) and the bundle cleanup
    (record.clean) use the same list."""
    rm = Path(d) / "render_meta.json"
    on = rm.is_file() and "head_view" in json.loads(rm.read_text())
    return (*MEDIA, HV.FILE) if on else MEDIA


def frame_steps(n_states: int, step_dt: float, fps: int = FPS) -> list[int]:
    """Video frame i -> record step round(i / fps / step_dt), up to the last step (only frames not past that time)."""
    dur = (n_states - 1) * step_dt
    n = int(math.floor(dur * fps + 1e-9)) + 1
    return [min(n_states - 1, int(round(i / fps / step_dt))) for i in range(n)]


def ego_view_meta(cal: dict, view: dict | None, grasp: dict) -> dict:
    """Episode meta ego_view: in-frame ratios of this trajectory's fixed head pose (first trajectory = the after of the
    calibrate framing · later trajectories = check_view's fill) -> render.framing.ego_view (grasp visibility, 'object
    mostly hidden') + the grasp window."""
    from dexengine.record.render import framing as CF

    fill = (view or {}).get("fill") if view is not None else ((cal or {}).get("framing") or {}).get("after")
    if fill is None:
        raise InputMissingError("[COLLECT_RENDER] ego_view: no in-frame ratios (no framing or camera check record)")
    return dict(CF.ego_view(fill, (view or {}).get("ladder")), grasp=grasp)


def _since_start() -> float:
    """Seconds since this process started (/proc/uptime - /proc/self/stat starttime): the 'boot -> render function'
    time."""
    with open("/proc/self/stat") as f:
        ticks = int(f.read().rsplit(")", 1)[1].split()[19])
    with open("/proc/uptime") as f:
        up = float(f.read().split()[0])
    return up - ticks / os.sysconf("SC_CLK_TCK")


def _tree_dir(args_cli, clip: str) -> Path:
    """Tree folder of a one-clip session: --sequence_dir (tree root, record.tree_ref) / <clip tree name>."""
    from dexengine.core import paths as P

    seq = getattr(args_cli, "sequence_dir", None) if args_cli is not None else None
    tree = Path(seq or "") / (P.clip_name(clip) + P.TREE_SFX)
    if not seq or not (tree / "task_info.json").is_file():
        raise InputMissingError(f"[COLLECT_RENDER] tree {tree} has no task_info.json (sequence_dir {seq!r})")
    return tree


TM_KEYS = (
    "calibrate",
    "settle",
    "capture",
    "body",
    "idle",
    "extras",
    "supports",
    "track",
    "visible",
    "headview",
    "mp4",
    "depth",
    "write_wait",
    "save",
)


class Session:
    """Fixed state of a render session: canonical background and record look (once) · fixed clip reset (clip_ids = the
    current trajectory k) · Rig (from the first trajectory) · first calibrate · unused cameras off
    (render.clip_session.cams_off) · body visual-layer meta (BODY.apply reuse) · scene prim baseline.
    Shared by render_selected (one trajectory) and render.clip_phases.render_clip (several)."""

    def __init__(self, wrapped, traj_mgr, raw_env, args_cli, checkpoint_path):
        import torch

        self.wrapped, self.env, self.args_cli, self.ckpt = wrapped, raw_env, args_cli, checkpoint_path
        self.scene, self.dev = raw_env.scene, raw_env.device
        self.eids = torch.tensor([0], device=self.dev)
        self.look = RIG.collect_scene_look()
        self.k, self.rig, self.cal, self.off, self.body0, self.prims0 = 0, None, None, None, None, None
        self.idle = {}  # trajectory name -> idle-hand measurement (read only; reused on reload)

        def _pinned_reset(env_ids):
            traj_mgr.clip_ids[env_ids] = self.k
            traj_mgr.clip_lengths[env_ids] = traj_mgr._loader.get_clip_lengths(traj_mgr.clip_ids[env_ids])
            traj_mgr.start_frames[env_ids] = 0

        traj_mgr.reset = _pinned_reset


def _bind(S: Session, k: int, clip: str, path: str, tm: dict, hv: bool) -> dict:
    """First half of load_traj and probe_view: reset (clip_ids = k) · entity binding · body · track points · grasp
    moment. Returns the trajectory context."""
    import numpy as np
    import torch

    from dexengine.record.render import slots as SLT  # binding entry (fills empty slots that exist only in the scene)
    from dexengine.rl.eval import traj_file as TF
    from dexengine.rl.eval import traj_record as TR

    clk, t0 = time.perf_counter, time.perf_counter()
    tf = TF.load(path)
    st = tf.selected(clip)
    if st is None:
        raise InputMissingError(f"[COLLECT_RENDER] {clip}: no chosen trajectory (0 successes): not rendered")
    raw_env, scene, eids = S.env, S.scene, S.eids
    S.k = k
    S.wrapped._reset_once = True
    S.wrapped.reset()
    RIG.hide_markers(raw_env)
    binds, st = SLT.bind(scene, raw_env.cfg, k, clip, tf, st, S.dev)
    T = {"clip": clip, "traj": path, "tf": tf, "st": st, "P": torch.from_numpy(np.ascontiguousarray(st)).to(S.dev)}

    def write(t: int) -> None:
        TR.write_states(scene, binds, T["P"][t : t + 1], eids)
        raw_env.sim.physics_sim_view.update_articulations_kinematic()

    def write_any(s) -> None:
        TR.write_states(scene, binds, s, eids)
        raw_env.sim.physics_sim_view.update_articulations_kinematic()

    step_dt = float(tf.meta["step_dt"])
    steps = frame_steps(st.shape[0], step_dt)
    print(
        f"[COLLECT_RENDER] {clip} · trajectory {st.shape[0]} steps x {step_dt * 1000:.0f} ms -> video {len(steps)} "
        f"frames @ {FPS} fps · entities {len(binds)} ({','.join(b.name for b in binds)}) · file {path}",
        flush=True,
    )
    if S.rig is None:
        S.rig = RIG.Rig(raw_env, S.args_cli)
    rig = S.rig
    T["prep"] = clk() - t0
    t = clk()
    dk = rig.desk_local(raw=True)
    st_r, body, boxes = BODY.apply(raw_env, binds, st, bool(rig.desk_dz), write_any, dk, reuse=S.body0)
    if S.body0 is None:
        S.body0 = body
    rig.body = body
    tm["body"] += clk() - t
    T["P"] = torch.from_numpy(np.ascontiguousarray(st_r)).to(S.dev)
    hv_sides = {} if hv else None
    t = clk()
    groups, per = TRK.tracked(tf, binds, write, st.shape[0], hv_sides, st_r, clip)  # framing and in-frame points
    from dexengine.record.render import framing as CF  # numpy: after AppLauncher only ([PRE_APP])

    grasp = CF.add_grasp(groups, per["object"], per["hand"], VIS.object_corners(binds, st, per["object"]), step_dt)
    tm["track"] += clk() - t
    T.update(
        binds=binds,
        write=write,
        steps=steps,
        step_dt=step_dt,
        body=body,
        boxes=boxes,
        groups=groups,
        per=per,
        hv_sides=hv_sides,
        grasp=grasp,
    )
    return T


def load_traj(S: Session, k: int, clip: str, path: str, tm: dict, camera_only: bool = False) -> dict:
    """Load one trajectory and set the scene up to the first frame: reset (clip_ids = k) · entity binding · body ·
    track points (_bind) · idle hands · supports · extra objects · cameras (first trajectory = calibrate + unused
    cameras off · later trajectories = check_view: fixed session camera). Returns the trajectory context.
    camera_only (clip_phases.orig_camera: the original loaded only to fix the camera; its context is discarded and
    never drawn) = the idle hand is not measured: IDLE.skipped prints the marker line saying so, nothing is cached (a
    later drawing load of the same trajectory measures it), and the context has no 'idle' (render_episode refuses
    it)."""
    from dexengine.record.render import clip_session as RCL

    clk = time.perf_counter
    T = _bind(S, k, clip, path, tm, hv=True)
    raw_env, rig, write, steps, body = S.env, S.rig, T["write"], T["steps"], T["body"]
    t = clk()
    if camera_only:
        IDLE.skipped(clip, "loaded only to fix the camera; this load is not drawn")
    elif clip not in S.idle:
        S.idle[clip] = IDLE.measure(
            raw_env, T["binds"], T["tf"], rig.desk_local(), [steps[0], steps[len(steps) // 2]], write, T["boxes"]
        )
    idle_kw = {} if camera_only else {"idle": S.idle[clip]}
    tm["idle"] += clk() - t
    write(0)
    if S.prims0 is None:
        S.prims0 = RCL.prims()  # scene prim baseline: the first trajectory before supports and extras
    t = clk()
    supports = SUP.show(raw_env, rig.desk_local(), body.get("world_shift_m", 0.0))
    tm["supports"] += clk() - t
    marker = PM.show(raw_env, T["binds"], T["P"][0], _tree_dir(S.args_cli, clip), S.k, body.get("world_shift_m", 0.0))
    t = clk()
    if S.cal is None:  # before the extras, so the occlusion gate reads the render camera
        S.cal = rig.calibrate(T["groups"])
        S.off = RCL.cams_off(rig)
        view = None
    else:
        view = RCL.check_view(rig, T["groups"])
    tm["calibrate"] += clk() - t
    from dexengine.record.render import framing as CF

    exo = CF.exo_cam(S.cal)[0]  # fixed for the session (world)

    def cam_fn() -> dict:  # render camera world positions: head left eye (per step) · exo
        return {"ego_left": rig.pose("collect_ego_left")[0], "exo": exo}

    t = clk()
    extras = EXV.apply(
        raw_env,
        T["binds"],
        write,
        steps,
        _tree_dir(S.args_cli, clip),
        clip,
        body.get("world_shift_m", 0.0),
        cam_fn,
        T["per"],
        FPS,
    )
    tm["extras"] += clk() - t
    from dexengine.rl.tasks.scene.all_obj_collision import assert_all_obj_ready, place_all_obj_bodies

    place_all_obj_bodies(raw_env, extras, body.get("world_shift_m", 0.0))  # off = does nothing
    # physics pose vs visual prims (marker 'pose check frame 0'; beyond tol dies; off = as is)
    assert_all_obj_ready(raw_env)
    write(0)
    T.update(**idle_kw, look=dict(S.look, supports=supports), supports=supports, extras=extras, view=view)
    if marker is not None:
        T["marker"] = marker
    return T


def probe_view(S: Session, k: int, clip: str, path: str) -> dict:
    """Measure only a candidate trajectory's ego visibility (trajectory choice of the render session,
    record.episodes.choose): after _bind (reset, load, body, track points), check_view in the fixed head pose (the
    original's angle; camera check included) -> ego_view_meta. No supports, extras, idle hands or render. Called before
    the original's calibrate it dies."""
    from dexengine.record.render import clip_session as RCL

    if S.cal is None:
        raise InputMissingError(
            f"[COLLECT_RENDER] {clip}: candidates cannot be measured before the original's calibrate"
        )
    tm = dict.fromkeys(TM_KEYS, 0.0)
    t = time.perf_counter()
    T = _bind(S, k, clip, path, tm, hv=False)
    T["write"](0)
    ev = ego_view_meta(S.cal, RCL.check_view(S.rig, T["groups"]), T["grasp"])
    return dict(ev, probe_s=round(time.perf_counter() - t, 2))


def render_episode(
    S: Session,
    T: dict,
    out: Path,
    tm: dict,
    extra: dict | None,
    boot_s: float,
    t0: float,
    prev: Pending | None = None,
    gate=None,
) -> Pending:
    """One episode: convergence and brightness gate (LOOK.rev().settle_gate) -> frames (first FIRST_WARMUP, later
    FRAME_RENDERS; mp4 and depth by the background writer render.writer) -> finish the previous episode
    (prev.finalize, overlap depth 1) -> seal its own write. render_meta, the time line and the DONE line come from the
    returned Pending.finalize (after the write join).
    extra = fields to add (render_clip's look, episode_index)."""

    if "idle" not in T:  # load_traj camera_only: the idle hand was not measured
        raise RuntimeError(f"{IDLE.MARK} {T['clip']} was loaded camera-only (idle hand not measured): it is not drawn")
    clk, rig, steps, per, hv_sides, write, cal = (
        time.perf_counter,
        S.rig,
        T["steps"],
        T["per"],
        T["hv_sides"],
        T["write"],
        S.cal,
    )

    look_settle, extra, tap = LOOK.rev().settle_gate(rig, T, extra, gate, FPS, tm)
    out.mkdir(parents=True, exist_ok=True)
    CW.check_device(S.dev)
    W = CW.EpisodeWriter(out, len(steps), VIDEOS, FPS, DEPTH_FILE)
    heads, tally = [], VIS.Tally(VIS.object_margins(T["binds"]))
    hv = HV.Recorder({s: v["links"] for s, v in hv_sides.items()}) if hv_sides is not None else None
    first_cap = 0.0
    first_cap = EFR.capture_frames(steps, clk, write, rig, heads, tm, tap, W, tally, per, cal, hv, hv_sides, first_cap)
    c = clk()
    vis_meta = tally.meta(len(steps))
    tm["visible"] += clk() - c
    meta = EFR.episode_meta(T, look_settle, steps, cal, rig, heads, vis_meta, S, extra)
    if hv is not None:
        c = clk()
        hv.oracle([s["side"] for s in T["tf"].meta["sides"]], meta["visibility"]["hand"]["in_frame"])
        meta["head_view"] = hv.write(out, steps)
        tm["headview"] += clk() - c
    # freeze the current values: dicts changed by the scene or the look before the finish (after the next episode)
    # must not leak into the meta (same serialisation; the bytes equal an immediate write)
    meta = json.loads(json.dumps(meta, ensure_ascii=False, default=float))
    if prev is not None:
        # the previous episode's write overlapped this render: join now (<= 1 pending)
        prev.finalize("overlap depth 1")
    # close the mp4s; depth compression continues in the background (overlaps the next)
    W.seal()
    return Pending(
        W,
        meta,
        out,
        tm,
        {
            "prep": T["prep"],
            "boot_s": boot_s,
            "t0": t0,
            "first_cap": first_cap,
            "n": len(steps),
            "settle_renders": look_settle["renders"],
        },
    )


class Pending:
    """One sealed episode: finalize(mode) = write join (SystemExit on failure or stall, render.writer.wait) ->
    render_meta, write line, time line, DONE line, summary. Finishes once (a second call returns the same summary)."""

    def __init__(self, W, meta: dict, out: Path, tm: dict, ctx: dict):
        self.W, self.meta, self.out, self.tm, self.ctx, self.r = W, meta, out, tm, ctx, None

    def finalize(self, mode: str) -> dict:
        if self.r is None:
            self.r = _finalize(self, mode)
        return self.r


def _finalize(p: Pending, mode: str) -> dict:
    clk, out, tm, meta, x = time.perf_counter, p.out, p.tm, p.meta, p.ctx
    wres = p.W.wait()
    w, dinfo, n = wres["work"], wres["depth"], x["n"]
    tm["mp4"] += w["mp4"] + w["close"]
    tm["depth"] += w["depth"] + w["zstd"]
    tm["write_wait"] += wres["wait_frames_s"] + wres["wait_end_s"]
    meta.update(
        counts=dict.fromkeys(VIDEOS, wres["frames"]), depth_frames=dinfo["frames"], depth_file=dict(dinfo, **CW.CODEC)
    )
    c = clk()
    (out / "render_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=float))
    tm["save"] += clk() - c
    print(CW.line(wres, out, mode), flush=True)
    rest = (tm["capture"] - x["first_cap"]) / max(1, n - 1) * 1000
    print(
        f"[COLLECT_RENDER] time boot -> render function {x['boot_s']:.1f} s · setup {x['prep']:.1f} · "
        f"calibrate {tm['calibrate']:.1f} · settle {tm['settle']:.1f} · frames {n} {tm['capture']:.1f} "
        f"(step {rest:.1f} ms · "
        f"first frame {x['first_cap']:.2f} s) · body {tm['body']:.1f} · idle {tm['idle']:.1f} · "
        f"extras {tm['extras']:.1f} · supports {tm['supports']:.1f} · track points {tm['track']:.1f} · "
        f"occlusion {tm['visible']:.1f} · "
        f"head view {tm['headview']:.1f} · mp4 x3 {tm['mp4']:.1f} (background) · depth compression "
        f"{tm['depth']:.1f} (background) · "
        f"write wait {tm['write_wait']:.1f} · store {tm['save']:.2f} · render function total {clk() - x['t0']:.1f} s",
        flush=True,
    )
    ok = wres["frames"] == n and dinfo["frames"] == n and all((out / m).is_file() for m in media_files(out))
    print(
        f"[COLLECT_RENDER] DONE frames={wres['frames']}/{n} · depth {dinfo['frames']} "
        f"({dinfo['bytes'] / 1e6:.1f} MB · raw {dinfo['raw_bytes'] / 1e6:.0f} MB · round trip equal) · {out}",
        flush=True,
    )
    return {
        "ok": ok,
        "frames": wres["frames"],
        "n": n,
        "out": str(out),
        "settle_renders": x["settle_renders"],
        "times": {k: round(v, 2) for k, v in tm.items()},
        "step_ms": round(rest, 1),
        "first_s": round(x["first_cap"], 2),
        "write": {k: wres[k] for k in ("busy_s", "wait_frames_s", "wait_end_s", "overlap_s", "qmax")},
    }


def render_selected(
    wrapped, policy_fn, traj_mgr, num_clips, clip_names, raw_env, checkpoint_path, out_dir, *_a, args_cli=None, **_kw
):
    """One trajectory per session (COLLECT_TRAJ), look D; output = MEDIA + render_meta.json in <COLLECT_OUT>/."""
    boot_s, t0 = _since_start(), time.perf_counter()
    tm = dict.fromkeys(TM_KEYS, 0.0)
    if num_clips != 1:
        raise SystemExit(f"[COLLECT_RENDER] {num_clips} clips: one-clip sessions only")
    S = Session(wrapped, traj_mgr, raw_env, args_cli, checkpoint_path)
    T = load_traj(S, 0, clip_names[0], os.environ["COLLECT_TRAJ"], tm)
    p = render_episode(S, T, Path(os.environ["COLLECT_OUT"]), tm, None, boot_s, t0)
    r = p.finalize("no overlap: one-trajectory session")  # DONE line; write join before SR.finish
    SR.finish(r["ok"], wrapped)
