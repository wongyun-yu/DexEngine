"""Kinematic replay render: the selected trajectories of a recorded trajectory file (dexengine.rl.eval.traj_record) are
written into the scene without physics and rendered by a pool of tiled cameras.

Entry: the frozen-render branch of dexengine.rl.train.runner (DEXENGINE_FROZEN_RENDER and DEXENGINE_FROZEN_REPLAY).
When DEXENGINE_TRAJ_RENDER holds a trajectory .npz path the render comes here instead of
dexengine.rl.render.clip_video._run_per_clip_video_render, on the same frozen scene cfg as the judge. create_env calls
configure_scene() for the scene layout (cameras, env spacing, floor); this module owns those values.

The scene holds every clip of the session (env e <-> clip: traj_mgr.clip_ids[e] under static clip assignment, else
e % K). Cameras are not per env: a slot pool (dexengine.record.replay_cameras) moves across the clips and renders them,
one mp4 per clip and view.
Printed lines: [REPLAY_SCENE] · [REPLAY_POOL] · [REPLAY_RENDER] … · [REPLAY_MAP] … · [VIDEO_SINK] … ·
[REPLAY_RENDER] DONE …. Counts are checked (videos = selected clips × views, ffprobe frames = trajectory length) and a
mismatch raises.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time

import numpy as np

from dexengine.core import capacity
from dexengine.core.paths import run_out as _run_out  # a relative output folder is relative to the run tree

ENV_VAR = "DEXENGINE_TRAJ_RENDER"
ENV_SPACING_M = 7.5  # room (6 m) + margin: a neighbouring env never enters the room
# Extra renders of the first frame at every round start (session start included), until the RTX temporal state
# (DLSS) of the moved slots converges. Measured tile-to-tile PSNR floor: 0 -> 29.6, 8 -> 35.3, 16 -> 36.4,
# 32 -> 37.7, 64 -> 37.9 dB.
WARMUP_RENDERS = 32
# The replay keeps the scene cfg's RTX settings (the balanced preset of the physics render). Measured on the
# slot-move ghosting: auto exposure is already off, and turning off the DL denoiser, FXAA or the performance
# preset only moves the noise floor; the round warmup above removes it. [REPLAY_RTX] records the values the app
# actually uses.
RTX_READ = (
    "/rtx/post/aa/op",
    "/rtx/post/dlss/execMode",
    "/rtx-transient/dldenoiser/enabled",
    "/rtx/directLighting/sampledLighting/enabled",
    "/rtx/post/histogram/enabled",
    "/rtx/post/tonemap/op",
)
# view name <-> scene camera attribute (create_env's DEXENGINE_RENDER_VIEWS selection reads this table too)
VIEW_CAMERAS = (
    ("ego", "head_camera"),
    ("front", "front_camera"),
    ("side", "side_camera"),
    ("side_front", "side_front_camera"),
)


def enabled() -> bool:
    """True when DEXENGINE_TRAJ_RENDER is set (replay render instead of the per-clip policy render)."""
    return bool(os.environ.get(ENV_VAR, "").strip())


def _rtx_readback() -> str:
    """The [REPLAY_RTX] line: RTX settings the app actually uses."""
    import carb

    st = carb.settings.get_settings()
    return "[REPLAY_RTX] scene cfg as is · " + " ".join(f"{k}={st.get(k)}" for k in RTX_READ)


def configure_scene(cfg, num_envs: int) -> None:
    """Called by create_env in replay mode: camera specs go to the slot pool (per-env camera sensors are removed),
    env spacing and floor size are set for a grid of num_envs envs. Prints [REPLAY_SCENE]."""
    from dexengine.record import replay_cameras as POOL

    got = POOL.take_specs(cfg, VIEW_CAMERAS, remove=True)
    cfg.scene.env_spacing = ENV_SPACING_M
    side = math.ceil(math.sqrt(max(1, int(num_envs))))
    floor_txt = "no floor"
    fl = getattr(cfg.scene, "floor", None)
    if fl is not None and hasattr(getattr(fl, "spawn", None), "size"):
        s = 2.0 * (side + 1) * ENV_SPACING_M
        old = tuple(fl.spawn.size)
        if s > old[0]:
            fl.spawn.size = (s, s, old[2])
        floor_txt = f"floor {old[0]:.0f}→{fl.spawn.size[0]:.0f} m"
    cam_txt = "camera pool (no per-env cameras)"
    print(
        f"[REPLAY_SCENE] mode=render · views {' '.join(got)} · {cam_txt} · env {num_envs} · grid {side} cols · "
        f"env_spacing {ENV_SPACING_M} m · {floor_txt}",
        flush=True,
    )


MARKER_ROOT = "/Visuals"  # root of the Isaac Lab VisualizationMarkers prims (debug axes, targets)


def _hide_markers(raw_env) -> int:
    """Draw no markers (a rule, not a name list): markers are updated by physics steps, so a replay would leave them at
    stale poses. (1) The env's debug markers are created lazily on the first step; setting the env's own
    "attempted, not ready" state keeps them from being created. (2) Existing marker prims (children of MARKER_ROOT)
    are hidden one by one, because a child's MakeVisible would undo a hidden root. Returns the number hidden."""
    import omni.usd
    from pxr import UsdGeom

    if not hasattr(raw_env, "_debug_markers_attempted"):
        raw_env._debug_markers_attempted, raw_env._debug_markers_ready = True, False
    root = omni.usd.get_context().get_stage().GetPrimAtPath(MARKER_ROOT)
    kids = list(root.GetChildren()) if root and root.IsValid() else []
    for k in kids:
        UsdGeom.Imageable(k).MakeInvisible()
    return len(kids)


def _scene_look() -> None:
    """Canonical background and lighting, with the room built inside every env prim."""
    import omni.usd

    from dexengine.rl.render import scene_look

    scene_look.apply_from_env(omni.usd.get_context().get_stage(), per_env_room=True)


def _render(raw_env, cams) -> None:
    """Render once and refresh every (view, camera)."""
    raw_env.sim.render()
    for _v, c in cams:
        c.update(0.0, force_recompute=True)


class _HostRing:
    """Ring of pinned host buffers for GPU [n, H, W, 3] frames. Depth = video-sink queue + 3, so a frame still queued or
    being written is never overwritten (push blocks on a full queue, so the frame of `depth` steps ago has left
    through the pipe). Several times faster than copies to pageable memory."""

    def __init__(self, shape):
        import torch

        from dexengine.record.video_sink import QUEUE

        self.bufs = [torch.empty(shape, dtype=torch.uint8, pin_memory=True) for _ in range(QUEUE + 3)]
        self.k = 0

    def put(self, x) -> np.ndarray:
        b = self.bufs[self.k % len(self.bufs)]
        self.k += 1
        b.copy_(x)
        return b.numpy()


def _gpu_mem_mib() -> int:
    """GPU memory of this process in MiB from nvidia-smi (RTX allocations included, which torch does not see); -1 if
    unknown."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        for ln in r.stdout.splitlines():
            p, m = [x.strip() for x in ln.split(",")]
            if int(p) == os.getpid():
                return int(m)
    except (OSError, ValueError, subprocess.SubprocessError) as _exc:
        print(f"  [Render] nvidia-smi read failed: {_exc!r}")
    return -1


def _env_clips(traj_mgr, clip_names: list[str], n_env: int) -> list[str]:
    """env -> clip name: the env's fixed clip under static (multi-object) assignment (traj_mgr.clip_ids, the same
    assignment as the spawned meshes), else e % K."""
    static = bool(getattr(getattr(traj_mgr, "cfg", None), "static_clip_assignment", False))
    k = len(clip_names)
    ids = traj_mgr.clip_ids.detach().cpu().tolist() if static else [e % k for e in range(n_env)]
    return [clip_names[int(i)] for i in ids]


def _banner(raw_env, views: list[str], res: tuple[int, int], out_dir: str, fps: int) -> None:
    """Start line [REPLAY_RENDER] (render setup, task package, code tree) and the [REPLAY_RTX] line."""
    from dexengine.core.run_identity import tree_identity
    from dexengine.rl import tasks as _pkg  # the task package in use

    tr = tree_identity()
    print(
        f"[REPLAY_RENDER] mode=render N={int(raw_env.num_envs)} res={res[0]}x{res[1]} views={','.join(views)} "
        f"fps={fps} out={out_dir} · pkg={os.path.dirname(_pkg.__file__)} · "
        f"tree={tr['tree']}@{tr['commit'][:12]}({tr['branch']}{', dirty' if tr['dirty'] else ''})",
        flush=True,
    )
    print(_rtx_readback(), flush=True)


def run(wrapped, policy_fn, traj_mgr, raw_env, clip_names, *, args_cli, checkpoint_path: str = "") -> None:
    """Replay-render the trajectory file named by DEXENGINE_TRAJ_RENDER into args_cli.videos_out_dir."""
    from dexengine.record import replay_cameras as POOL

    src = os.environ.get(ENV_VAR, "").strip()
    out_dir = str(_run_out(args_cli.videos_out_dir))
    os.makedirs(out_dir, exist_ok=True)
    fps = int(args_cli.videos_fps)
    sp = POOL.SPECS
    if len({(s.width, s.height) for s in sp}) != 1:
        raise RuntimeError(f"[REPLAY_RENDER] views differ in resolution {[(s.view, s.width, s.height) for s in sp]}")
    _banner(raw_env, [s.view for s in sp], (sp[0].width, sp[0].height), out_dir, fps)
    nm = _hide_markers(raw_env)
    _scene_look()
    print(
        f"[REPLAY_RENDER] markers hidden — env debug markers not created · {MARKER_ROOT} children {nm} hidden "
        "· room and floor per env (scene_look per_env_room)",
        flush=True,
    )
    if not os.path.isfile(src):
        raise FileNotFoundError(f"[REPLAY_RENDER] {ENV_VAR}={src} — the trajectory file is missing")
    _render_file(src, traj_mgr, raw_env, list(clip_names), out_dir, fps)


def _session(tf, env_clip: list[str]):
    """Session clips -> ([(env, clip, selected trajectory)], clips without a selection, file selections outside the
    session, number of session clips)."""
    act, seen, none_sel = [], set(), []
    for e, c in enumerate(env_clip):
        if c in seen:
            continue
        seen.add(c)
        st = tf.selected(c)  # KeyError when the clip is not in the file's roster
        if st is None:
            none_sel.append(c)
        else:
            act.append((e, c, st))
    if not act:
        raise RuntimeError(
            "[REPLAY_RENDER] 0 selected trajectories to draw — check that the session roster and the file match"
        )
    file_sel = {tf.clip_names[i] for i in range(len(tf.clip_names)) if tf.sel[i] >= 0}
    return act, none_sel, sorted(file_sel - {c for _e, c, _s in act}), len(seen)


def _timeline(sched, lens, m: int):
    """[T, m] clip index per slot (-1 = empty slot) and the frame inside that clip."""
    T = max(t0 + lens[j] for slot in sched for j, t0 in slot)
    clip_at = np.full((T, m), -1, np.int64)
    local_at = np.zeros((T, m), np.int64)
    for s, slot in enumerate(sched):
        for j, t0 in slot:
            clip_at[t0 : t0 + lens[j], s] = j
            local_at[t0 : t0 + lens[j], s] = np.arange(lens[j])
    return clip_at, local_at


def _render_file(path, traj_mgr, raw_env, clip_names, out_dir, fps) -> None:
    """Render every selected trajectory of the file: the pool's slots are scheduled over the clips, states are
    written per tile step, and frames stream to one VideoSink per clip and view."""
    import torch

    from dexengine.record import replay_cameras as POOL
    from dexengine.record.video_sink import VideoSink
    from dexengine.rl.env_setup.objects import AERIAL_HIDDEN_Z  # parking position of empty slots
    from dexengine.rl.eval import traj_file as TF
    from dexengine.rl.eval import traj_record as TR

    act, clip_at, counts, h, lens, m, made, n_env, none_sel, outside, t_c, t_r, thr, views, w, wall = (
        _load_and_place_clips(
            path, traj_mgr, raw_env, clip_names, out_dir, fps, AERIAL_HIDDEN_Z, POOL, TF, TR, VideoSink, torch
        )
    )
    _verify_and_report(
        {
            "file": path,
            "n_env": n_env,
            "none_selected": none_sel,
            "outside_session": outside,
            "views": views,
            "res": (w, h),
            "slots": m,
            "tile_frames": int(clip_at.shape[0]),
            "threads_per_stream": thr,
            "wall": wall,
            "t_r": t_r,
            "t_c": t_c,
        },
        act,
        lens,
        made,
        counts,
        out_dir,
        fps,
    )


def _load_and_place_clips(
    path, traj_mgr, raw_env, clip_names, out_dir, fps, AERIAL_HIDDEN_Z, POOL, TF, TR, VideoSink, torch
):
    """Load the trajectory file and place its clips on the env slots."""
    dev, scene = raw_env.device, raw_env.scene
    tf = TF.load(path)
    n_env = int(raw_env.num_envs)
    act, none_sel, outside, n_seen = _session(tf, _env_clips(traj_mgr, clip_names, n_env))
    binds = TR.bind(scene, tf.entities, dev, trajs=[st for _e, _c, st in act], hidden_z=AERIAL_HIDDEN_Z)
    print(
        f"[REPLAY_MAP] env→clip {n_env} env · all {n_seen} session clips are in the file roster ✓ · "
        f"with a selected trajectory {len(act)} · "
        f"no selection (0 successes — not rendered · failure_best kept "
        f"{sum(tf.failure_best(c) is not None for c in none_sel)}) "
        f"{len(none_sel)} · file selections outside this session {len(outside)} · "
        f"entities {len(binds)}({','.join(b.name for b in binds)}) names and joints checked ✓ · file {path}",
        flush=True,
    )
    lens = [st.shape[0] for _e, _c, st in act]
    w, h = POOL.SPECS[0].width, POOL.SPECS[0].height
    views = [s.view for s in POOL.SPECS]
    m = POOL.n_slots(len(views), w, h, len(act))
    sched = POOL.schedule(lens, m)
    clip_at, local_at = _timeline(sched, lens, m)
    pool = POOL.CameraPool(raw_env, m)
    P = torch.empty((len(act), max(lens), tf.states.shape[1]), dtype=torch.float32, device=dev)
    for i, (_e, _c, st) in enumerate(act):
        x = torch.from_numpy(np.ascontiguousarray(st)).to(dev)
        P[i, : x.shape[0]] = x
        P[i, x.shape[0] :] = x[-1]
    env_of = torch.tensor([e for e, _c, _s in act], dtype=torch.long, device=dev)
    thr = max(1, capacity.cpu_limit()[0] // (m * len(views)))  # encoder threads: the usable CPUs (capacity) shared
    rings = {v: _HostRing((m, h, w, 3)) for v in views}
    sinks, made, counts = {}, [], {}
    t_r = t_c = 0.0
    torch.cuda.synchronize(dev)
    t0 = time.time()
    for f in range(clip_at.shape[0]):
        a = time.time()
        live = np.nonzero(clip_at[f] >= 0)[0]
        js, loc = clip_at[f, live], local_at[f, live]
        sl = torch.as_tensor(live, device=dev)
        envs = env_of[torch.as_tensor(js, device=dev)]
        TR.write_states(scene, binds, P[torch.as_tensor(js, device=dev), torch.as_tensor(loc, device=dev)], envs)
        new = loc == 0
        pool.place(sl[torch.as_tensor(new, device=dev)], envs[torch.as_tensor(new, device=dev)], "static")
        pool.place(sl, envs, "dynamic")
        for _ in range(WARMUP_RENDERS if new.any() else 1):  # round start: every slot holds a new clip
            pool.render()
        frames = {v: cam.data.output["rgb"][..., :3] for v, cam in pool.cams}
        torch.cuda.synchronize(dev)
        b = time.time()
        t_r += b - a
        for k, _s in enumerate(live):
            j = int(js[k])
            if new[k]:
                for v in views:
                    p = os.path.join(out_dir, f"{act[j][1]}__{v}.mp4")
                    sinks[(j, v)] = VideoSink(p, w, h, fps, threads=thr)
                    made.append((j, v, p))
        for v, x in frames.items():
            arr = rings[v].put(x)  # all slots (empty tiles too); only live slots are pushed
            for k, s in enumerate(live):
                sinks[(int(js[k]), v)].push(arr[s])
        for k, _s in enumerate(live):
            if loc[k] == lens[int(js[k])] - 1:
                for v in views:
                    sinks[(int(js[k]), v)].finish()  # no wait: the tail encodes while the other streams run
        t_c += time.time() - b
    from concurrent.futures import ThreadPoolExecutor

    with capacity.sized("dexengine.rl.render.replay:close", len(sinks)) as cap, ThreadPoolExecutor(cap.workers) as ex:
        counts = dict(zip(list(sinks), ex.map(lambda k: sinks[k].close(), list(sinks)), strict=True))
    wall = time.time() - t0
    return act, clip_at, counts, h, lens, m, made, n_env, none_sel, outside, t_c, t_r, thr, views, w, wall


def _verify_and_report(info: dict, act, lens, made, counts, out_dir, fps) -> None:
    """Count check (video set = selected clips × views, ffprobe frames = trajectory length), then
    replay_manifest.tsv, replay_summary.json and the DONE line. Raises on a mismatch."""
    from concurrent.futures import ThreadPoolExecutor

    from dexengine.record.video_sink import probe_frames

    wall, t_r, t_c = info["wall"], info["t_r"], info["t_c"]
    (w, h), views, m, T = info["res"], info["views"], info["slots"], info["tile_frames"]
    with capacity.sized("dexengine.rl.render.replay:probe", len(made)) as cap, ThreadPoolExecutor(cap.workers) as ex:
        probed = list(ex.map(lambda r: probe_frames(r[2]), made))
    want = {(j, v) for j in range(len(act)) for v in views}
    got = {(j, v) for j, v, _p in made}
    bad = [
        (act[j][1], v, n, lens[j])
        for (j, v, _p), n in zip(made, probed, strict=True)
        if n != lens[j] or counts[(j, v)] != lens[j]
    ]
    bad += [(act[j][1], v, "video missing", lens[j]) for j, v in sorted(want - got)]
    rows = ["env\tclip\tview\tframes\tprobed\tpath"] + [
        f"{act[j][0]}\t{act[j][1]}\t{v}\t{counts[(j, v)]}\t{n}\t{p}" for (j, v, p), n in zip(made, probed, strict=True)
    ]
    with open(os.path.join(out_dir, "replay_manifest.tsv"), "w") as fh:
        fh.write("\n".join(rows) + "\n")
    vid_s = sum(lens) / float(fps)
    summ = dict(
        info,
        clips=len(act),
        frames=sum(lens),
        wall_s=round(wall, 2),
        render_s=round(t_r, 2),
        copy_encode_s=round(t_c, 2),
        tile_use=round(sum(lens) / float(m * T), 3),
        mp_per_s_effective=round(sum(lens) * len(views) * w * h / wall / 1e6, 1),
        mp_per_s_tile=round(m * len(views) * w * h * T / max(t_r, 1e-9) / 1e6, 1),
        realtime_factor=round(vid_s / wall, 3),
        video_s=round(vid_s, 1),
        gpu_mem_mib=_gpu_mem_mib(),
        bad=bad,
    )
    for k in ("wall", "t_r", "t_c"):
        summ.pop(k)
    with open(os.path.join(out_dir, "replay_summary.json"), "w") as fh:
        json.dump(summ, fh, indent=1, ensure_ascii=False)
    ok = "✓" if not bad else "✗"
    print(
        f"[REPLAY_RENDER] DONE videos {len(got) - len(bad)} = selected clips {len(act)} × views {len(views)} "
        f"{ok} · ffprobe frames = trajectory length {len(made) - len(bad)}/{len(want)} · states {sum(lens)} · "
        f"slots {m} · tile steps {T} "
        f"(use {summ['tile_use']}) · wall {wall:.1f}s(render+read {t_r:.1f} · copy+encoder {t_c:.1f}) · "
        f"effective {summ['mp_per_s_effective']} MP/s · tile {summ['mp_per_s_tile']} MP/s · "
        f"realtime factor {summ['realtime_factor']} · GPU {summ['gpu_mem_mib']} MiB",
        flush=True,
    )
    if bad:
        raise RuntimeError(f"[REPLAY_RENDER] video or frame count mismatch {bad[:5]}")
