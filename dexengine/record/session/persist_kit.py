"""Kit side of the Kit-reuse worker (session.entry --stage render_clips; the canonical path of render_clip; caller
session.persist.run).

render_clips: the Kit app starts once; for every queue job the canonical entry (the rl.cli module) runs again from the
start with that render_clip dump (runpy, name != __main__) and main() is called. The render function
(render.clip_session) is the one of render_clip; at the end it closes the env and returns (finish_render) -> a new USD
stage -> the next job. Same-worker condition = same app key (app_key).
The header uses the standard library only (no numpy before AppLauncher); Kit modules are imported inside the functions.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from dexengine.core.errors import InputMissingError
from dexengine.record import hooks as record_hooks
from dexengine.record.session import kit as KX
from dexengine.record.session import persist_guard as PG
from dexengine.record.session.persist import PERSIST_MARK
from dexengine.record.session.runner import _STATE, MARK, PostImport, finish, load_dump, require, set_arg, tree_line

PERSIST_DONE = "[RENDER_CLIPS] DONE"  # end line of a render_clips session (session.launch reads it as done_pat)
# module name when the canonical entry runs again (!= __main__: skips its main block)
PERSIST_RUN_NAME = "__collect_persist__"
#: fields that differ per clip (left out of the app key): dump argv values and env keys. Anything else that differs
#: is not put into the same session.
PER_CLIP_ARGS = ("--videos_out_dir",)
PER_CLIP_ENV = ("DEXENGINE_SEQ_LIST", "COLLECT_OUT", "COLLECT_PLAN")
#: render_clips session state: when on, the render function end (finish_render) closes only the env, not the app.
_PERSIST: dict = {"on": False, "i": 0, "n": 0, "clip": None, "t0": None, "rtx": None}


def render_clip_launch(dump: str, seq_dir: str, plan_path: str) -> tuple[list[str], dict]:
    """(rl.cli argv, env) of the render_clip stage: the dump plus the entry overrides (the same work in the same order
    as main's render_clip branch). Shared by the render_clips session (per clip) and the planner (render.clip, which
    groups by app key)."""
    from dexengine.record.render import cameras as CC
    from dexengine.record.render import clip_session as RCL

    argv, env = load_dump(dump)
    argv = KX.apply(argv)
    require(argv, "--kinematic_replay", "--record_session", "--enable_cameras")
    argv = KX.apply_render_threads(argv)
    argv = KX.apply_driver_check(argv)  # RTX driver check off (same order as main's render_clip branch)
    set_arg(argv, "--sequence_dir", seq_dir)
    plan = RCL.load_plan(plan_path)
    if os.path.realpath(plan["seq_dir"]) != os.path.realpath(seq_dir):
        raise SystemExit(f"{MARK} --seq-dir {seq_dir} != plan seq_dir {plan['seq_dir']}")
    set_arg(argv, "--videos_out_dir", os.path.join(plan["out"], "_videos"))
    env = dict(env, **{CC.ENV_SWITCH: "1"})
    env.update(COLLECT_OUT=plan["out"], COLLECT_PLAN=os.path.abspath(plan_path), DEXENGINE_SEQ_LIST=plan["seq_list"])
    return argv, env


def app_key(argv: list[str], env: dict) -> str:
    """App key: the first 16 hex of the sha256 of argv and env without the per-clip fields (PER_CLIP_ARGS values,
    PER_CLIP_ENV keys). Only equal keys draw on in one process (proof that the Kit arguments and the env the canonical
    entry reads are the same). Each dump has its own switches file, so the key uses the switch values, not the path."""
    a = [("<clip>" if k and argv[k - 1] in PER_CLIP_ARGS else v) for k, v in enumerate(argv)]
    from dexengine.rl.tasks import switches as SW  # noqa: PLC0415 — stdlib only

    e = {k: v for k, v in sorted(SW.dump_env_view(env).items()) if k not in PER_CLIP_ENV}
    return hashlib.sha256(json.dumps([a, e], ensure_ascii=False).encode()).hexdigest()[:16]


def finish_render(ok: bool, wrapped) -> None:
    """End of the render function: normally (render_clip) session.runner.finish as it is (close the env, then the app
    exits). For render_clips only the env is closed and the call returns (next clip; render_clips closes the app once
    at the end). First the OpenBLAS pool is checked again (KX.check_render_blas; late-loaded libraries included)."""
    KX.check_render_blas()
    if not _PERSIST["on"]:
        finish(ok, wrapped)
        return
    _STATE["done"] = ok
    wrapped.close()
    print(
        f"{MARK} env closed (STOP unsubscribed) — {PERSIST_MARK} clip {_PERSIST['i'] + 1}/{_PERSIST['n']} "
        f"{_PERSIST['clip']} · on to the next clip · ok={ok}",
        flush=True,
    )
    if not ok:
        raise SystemExit(f"{MARK} output count check failed")


def boot_s() -> float:
    """Seconds 'boot -> render function': normally from process start (render.episode._since_start); render_clips
    counts from that clip's entry start (from clip 2 on there is no app start; same render-meta field name)."""
    if _PERSIST["on"]:
        return time.perf_counter() - _PERSIST["t0"]
    from dexengine.record.render import episode as CR

    return CR._since_start()


def _flat(d, pre: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        p = f"{pre}/{k}"
        if isinstance(v, dict):
            out.update(_flat(v, p))
        else:
            out[p] = list(v) if isinstance(v, tuple) else v
    return out


def rtx_restore() -> str:
    """carb /rtx settings: take the values at the moment clip 1 enters the render function; from clip 2 on restore
    them at the same moment and read them back (a mismatch dies). So app settings changed by an earlier clip (look D:
    look_d TONEMAP, QUALITY, WHITE_K) do not leak into the next clip's env build and settle. -> marker line."""
    import carb

    s = carb.settings.get_settings()
    now = _flat(s.get("/rtx"), "/rtx")
    if _PERSIST["rtx"] is None:
        _PERSIST["rtx"] = now
        return f"{PERSIST_MARK} /rtx baseline {len(now)} keys (clip 1, render function entry)"
    ref = _PERSIST["rtx"]
    diff = {k: v for k, v in ref.items() if now.get(k) != v}
    for k, v in diff.items():
        s.set(k, v)
    back = _flat(s.get("/rtx"), "/rtx")
    bad = {k: (v, back.get(k)) for k, v in diff.items() if back.get(k) != v}
    if bad:
        raise SystemExit(f"{PERSIST_MARK} /rtx restore read-back mismatch {len(bad)} {list(bad.items())[:4]}")
    new = sorted(set(now) - set(ref))
    return (
        f"{PERSIST_MARK} /rtx restored {len(diff)} keys {sorted(diff)[:6]} · read-back equal · new keys after clip "
        f"1 {len(new)} {new[:4]}"
    )


def _reuse_app(launcher) -> None:
    """From clip 2 on: the canonical entry's AppLauncher(args) gets the first app back (a stand-in; the app key proves
    the Kit arguments are the same). The pre-Kit numpy gate (preapp.assert_clean) means nothing once Kit is up, so only
    a marker."""
    import isaaclab.app as IA

    from dexengine.rl import preapp as PG

    orig = IA.AppLauncher if not getattr(IA.AppLauncher, "_collect_reuse", False) else IA.AppLauncher.__mro__[1]

    class _Reuse(orig):
        _collect_reuse = True

        def __new__(cls, *_a, **_k):  # noqa: V103 — dunder, called by Python
            print(f"{PERSIST_MARK} AppLauncher reused: clip 1's Kit app as it is (same app key)", flush=True)
            return launcher

    IA.AppLauncher = _Reuse
    PG.assert_clean = lambda where="AppLauncher": print(
        f"{PERSIST_MARK} pre-{where} numpy gate skipped: Kit is already up (clip {_PERSIST['i'] + 1})", flush=True
    )


def _next_stage(app) -> str:
    """Between clips: stop the timeline -> a new USD stage (drops every prim, material, light, camera and augmentation
    of the previous clip) -> collect the previous clip's Python objects (gc, empty the torch cache). The marker carries
    the torch GPU memory (allocated, reserved): growing with every clip is a leak."""
    import gc

    import torch

    import omni.timeline
    import omni.usd

    from isaaclab.sim.utils.stage import create_new_stage

    from dexengine.record.looks import look_guard as LG

    print(LG.clear("Kit-reuse worker: next clip's stage"), flush=True)  # the token belongs to the stage being dropped
    omni.timeline.get_timeline_interface().stop()
    app.update()
    old = sum(1 for _ in omni.usd.get_context().get_stage().TraverseAll())
    create_new_stage()
    app.update()
    new = sum(1 for _ in omni.usd.get_context().get_stage().TraverseAll())
    gc.collect()
    torch.cuda.empty_cache()
    mib = (torch.cuda.memory_allocated() >> 20, torch.cuda.memory_reserved() >> 20)
    from pxr import Sdf

    # layers of old stages that only grow are a leak candidate (with host RSS)
    layers = len(Sdf.Layer.GetLoadedLayers())
    return (
        f"{PERSIST_MARK} new stage: old stage prims {old} dropped · new stage prims {new} · torch GPU allocated "
        f"{mib[0]} · reserved {mib[1]} MiB · host RSS {_rss_mib()} MiB (after gc) · open USD layers {layers}"
    )


#: process-level render state reset between clips: (module, name, initial value). On a new stage the previous clip's
#: prims are gone, so the initial value is right (scene.extras._GEN = generation of the extra-object prim names, a
#: counter that avoids reusing paths on one stage).
PER_CLIP_RESET = (("dexengine.record.scene.extras", "_GEN", [0]),)


def _state_reset() -> str:
    """From clip 2 on, before the entry: PER_CLIP_RESET back to the initial values (loaded modules only). -> marker."""
    done = []
    for mod, name, v0 in PER_CLIP_RESET:
        m = sys.modules.get(mod)
        if m is not None:
            done.append(f"{mod}.{name} {getattr(m, name)} → {v0}")
            getattr(m, name)[:] = v0
    return f"{PERSIST_MARK} process render state reset {done or '(no loaded module)'}"


#: env reset between clips covers our pipeline namespaces (keys the canonical entry and the recorder set to per-clip
#: values). Others (keys Kit and the launcher set at app start) belong to the app: left alone, named in the marker.
ENV_RESET_PREFIX = ("DEXENGINE_", "COLLECT_", "SEQ_", "RENDER_", "FROZEN_")


def _env_reset(env0: dict) -> str:
    """From clip 2 on, before the entry: undo the keys of our namespaces that appeared or changed during clip 1
    relative to the env right before clip 1's entry (env0), so a per-clip value the canonical entry left in the env
    ('set if missing') does not leak into the next clip. -> marker line."""
    now = dict(os.environ)
    ours = [k for k in set(now) | set(env0) if k.startswith(ENV_RESET_PREFIX) and now.get(k) != env0.get(k)]
    for k in ours:
        if k in env0:
            os.environ[k] = env0[k]
        else:
            del os.environ[k]
    other = sorted(k for k in set(now) - set(env0) if not k.startswith(ENV_RESET_PREFIX))
    return (
        f"{PERSIST_MARK} env reset {len(ours)} {sorted(ours)[:6]} · other keys set by the app {len(other)} "
        f"{other[:6]} (kept)"
    )


#: memory growth watch: after every job the host RSS is compared with its value after the first job; above this factor
#: the worker takes no further job and closes the app (the caller starts a new worker for the next job). Measured:
#: clip 1 -> 2 -> 4 at 8.6 -> 9.1 -> 9.2 GB (small growth).
RSS_GROW_MAX = 1.5


def _rss_mib() -> int:
    with open("/proc/self/status") as f:
        return next((int(ln.split()[1]) >> 10 for ln in f if ln.startswith("VmRSS:")), -1)


def resume(jpath: Path, summ: dict, todo: list[dict]) -> tuple[list[dict], set[int]]:
    """Resume inside a worker retry: from the previous attempt's summary (jpath) of the same plan (content sha, same
    chosen episodes), episodes that ended ok (they enter the summary only after the write join = the end marker) are
    returned as slots to skip, and their summary, trajectory and prim-oracle rows are carried over.
    -> (todo unchanged, slots to skip). Marker [RENDER_RESUME]."""
    if not _PERSIST["on"]:
        return todo, set()
    summ["plan_sha"] = hashlib.sha256(Path(summ["plan"]).read_bytes()).hexdigest()[:16]
    try:
        prev = json.loads(jpath.read_text()) if jpath.is_file() else {}
    except ValueError:
        prev = {}
    if not prev or prev.get("plan_sha") != summ["plan_sha"] or prev.get("episodes_plan") != summ.get("episodes_plan"):
        return todo, set()
    done = {
        (e["index"], e["traj"], e["look"]): e
        for e in prev.get("episodes", [])
        # with output only
        if e.get("ok") and (jpath.parent / e["traj"] / e["look"] / "render_meta.json").is_file()
    }
    skip = {i for i, e in enumerate(todo) if (i, e["traj"], e["look"]) in done}
    if skip:
        summ["episodes"] += [done[(i, todo[i]["traj"], todo[i]["look"])] for i in sorted(skip)]
        summ["trajs"] += prev.get("trajs", [])
        summ["prim_oracle"] += prev.get("prim_oracle", [])
        summ["resumed"] = len(skip)
    print(resume_line(len(skip), len(todo)), flush=True)
    return todo, skip


RESUME_MARK = "[RENDER_RESUME]"
RESUME_NONE_LEFT = " left 0"  # end of the resume line when nothing is left (resumed_all reads it)


def resumed_all(text: str) -> bool:
    """The session skipped every episode because an earlier attempt finished them (0 left, nothing drawn). Then the
    per-episode markers (look settle, occlusion, head view) are rightly absent (session.launch.classify)."""
    return any(ln.startswith(RESUME_MARK) and ln.rstrip().endswith(RESUME_NONE_LEFT) for ln in text.splitlines())


def resume_line(skipped: int, todo: int) -> str:
    """The [RENDER_RESUME] marker line (resumed_all reads its end)."""
    return f"{RESUME_MARK} episodes finished by an earlier attempt {skipped}/{todo} skipped · left {todo - skipped}"


SERVE_IDLE_S = 120  # the worker closes the app and ends when the queue is empty this long (caller restarts one)
SERVE_POLL_S = 0.1


def _q_jobs(queue: Path, owner: int):
    """Queue job feed: <n>.job (json {clip, dump, seq_dir, plan}) in number order, only those without a start marker
    <n>.start. Ends (None) on a STOP file, the death of the calling process (owner) or no job for SERVE_IDLE_S."""
    t_idle, idle = time.time(), SERVE_IDLE_S
    while True:
        if (queue / "STOP").exists() or (owner and not os.path.exists(f"/proc/{owner}")):
            return
        todo = sorted(
            (p for p in queue.glob("*.job") if not p.with_suffix(".start").exists()), key=lambda p: int(p.stem)
        )
        if todo:
            yield todo[0]
            t_idle = time.time()
            continue
        if time.time() - t_idle > idle:
            return
        time.sleep(SERVE_POLL_S)


def _log_size() -> int:
    sys.stdout.flush()
    sys.stderr.flush()
    return os.fstat(sys.stdout.fileno()).st_size


def _mark(path: Path, d: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(d))
    os.replace(tmp, path)


def job_start(jf: Path) -> None:
    """Begin a job's log span (<n>.start holds the log size now), then print the persist_job hook lines for that job.
    A line printed here falls inside the job's span, so a span check (session.persist) sees it."""
    _mark(jf.with_suffix(".start"), {"off": _log_size(), "pid": os.getpid()})
    for name in record_hooks.names("persist_job"):
        line = record_hooks.get("persist_job", name)(f"worker job {jf.stem}")
        if line:
            print(line, flush=True)


def _enter_job(i: int, j: dict, jf: Path, argv, env, tree: str, app, launcher, env0) -> dict | None:
    """Make the process ready for job i: from the second job on, a new stage and the first job's env and module state;
    then this job's env and argv, the start marker and the per-job hooks. Returns the env snapshot taken for job 0
    (the state later jobs reset to), else None."""
    from dexengine.record.session import runner as CRN

    if i:
        print(_next_stage(app), flush=True)
    _PERSIST.update(i=i, n=i + 1, clip=j["clip"], t0=time.perf_counter())
    if i:
        print(_env_reset(env0), flush=True)
        print(_state_reset(), flush=True)
    os.environ.update(env)
    os.environ["COLLECT_TREE"] = tree
    snap = dict(os.environ) if i == 0 else None
    sys.argv = list(argv)
    print(f"{PERSIST_MARK} clip {i + 1}/- start {j['clip']} · job {jf.stem} · plan {j['plan']}", flush=True)
    job_start(jf)
    if i:
        for ln in CRN._FIRED.values():  # process hooks fired in the first job stay; print their lines again
            print(ln, flush=True)  # per job span (so a span is classified like a plain session)
        _reuse_app(launcher)
    return snap


def _end_job(i: int, j: dict, jf: Path, rss0: int) -> tuple[int, int]:
    """Check the render function finished, print the job's time and memory, write its .done marker. Returns (this
    job's RSS, the first job's RSS)."""
    if not _STATE["done"]:
        raise InputMissingError(f"{PERSIST_MARK} job {i + 1} {j['clip']}: the render function did not finish (no DONE)")
    _STATE["done"] = False
    dt = time.perf_counter() - _PERSIST["t0"]
    print(f"{PERSIST_MARK} clip {i + 1}/- end {j['clip']} · {dt:.1f} s", flush=True)
    rss = _rss_mib()
    rss0 = rss0 or rss
    print(
        f"{PERSIST_MARK} memory job {i + 1} end RSS {rss} MiB · first job {rss0} MiB · gate x{RSS_GROW_MAX}",
        flush=True,
    )
    _mark(
        jf.with_suffix(".done"),
        {
            "off": json.loads(jf.with_suffix(".start").read_text())["off"],
            "end": _log_size(),
            "s": round(dt, 1),
            "rss_mib": rss,
        },
    )
    return rss, rss0


def render_clips(queue: str, owner: int, hooks: dict) -> None:
    """The render_clips worker (module doc): one Kit app; for every queue job (_q_jobs): reset the env to its value at
    the first job's start and add that job's dump env -> argv -> run the canonical entry module again (_reuse_app from
    the second job on) -> main() -> the render function closes the env and returns -> a new stage. Job markers:
    <n>.start (log size at that moment) · <n>.done (log span, seconds). When the app key differs from the first job's,
    the job is not taken and the app closes (the caller starts a new worker with the new key: one Kit, serial).
    An exception gives a traceback and a hard exit (rc != 0; the caller redoes only that job). End: close marker -> app
    exit -> os._exit(0). Markers: [COLLECT_PERSIST] on · job i start · end · new stage · AppLauncher reused · /rtx
    restored · end line PERSIST_DONE clips=<n>."""
    import runpy
    import traceback

    q = Path(queue)
    tree = tree_line()
    _PERSIST.update(on=True, n=0)
    print(f"{MARK} {tree} · hook targets {sorted(hooks)}", flush=True)
    print(
        f"{PERSIST_MARK} on: worker · queue {q} · caller {owner} · ends after {SERVE_IDLE_S} s idle "
        "(one Kit, canonical entry again per job, new stage)",
        flush=True,
    )
    PG.watch_owner(owner)
    sys.meta_path.insert(0, PostImport(hooks))
    launcher = app = env0 = key0 = None
    i = rss0 = 0
    try:
        for jf in _q_jobs(q, owner):
            j = json.loads(jf.read_text())
            argv, env = render_clip_launch(j["dump"], j["seq_dir"], j["plan"])
            key = app_key(argv, env)
            # close without a start marker; the caller restarts with the new key
            if key0 is not None and key != key0:
                print(
                    f"{PERSIST_MARK} key changed: close the app and restart (job {jf.stem} · app key {key} != {key0})",
                    flush=True,
                )
                break
            snap = _enter_job(i, j, jf, argv, env, tree, app, launcher, env0)
            if i == 0:
                env0, key0 = snap, key
            g = runpy.run_module("dexengine.rl.cli", run_name=PERSIST_RUN_NAME, alter_sys=True)
            if i == 0:
                launcher, app = g["app_launcher"], g["simulation_app"]
            elif g["app_launcher"] is not launcher:
                raise SystemExit(
                    f"{PERSIST_MARK} job {i + 1}: the canonical entry built a new AppLauncher (reuse stand-in failed)"
                )
            g["main"]()
            rss, rss0 = _end_job(i, j, jf, rss0)
            i += 1
            if rss > rss0 * RSS_GROW_MAX:
                print(
                    f"{PERSIST_MARK} memory gate: RSS {rss} > {rss0} x {RSS_GROW_MAX} · this worker closes here "
                    "(the next job gets a new worker)",
                    flush=True,
                )
                break
        print(f"{PERSIST_DONE} clips={i}", flush=True)
    # the same hard exit as the canonical entry's end block (Kit must not hold the process)
    except BaseException as e:  # noqa: BLE001 — every failure (KeyboardInterrupt too) takes the hard exit
        # arm the hard exit first: if the output or flush below hangs or raises, os._exit is never reached (test T2).
        # Normally os._exit ends first and the watcher ends on a false kill -0.
        PG.arm_kill(PG.DIE_GRACE_S, "failure path")
        traceback.print_exc()
        code = e.code if isinstance(e, SystemExit) and isinstance(e.code, int) and e.code else 1
        print(f"{PERSIST_MARK} ✗ ended in job {_PERSIST['i'] + 1} {_PERSIST['clip']}: rc {code}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    if app is not None:
        from dexengine.rl.isaac_cli import mark_closing  # noqa: PLC0415 — loaded by the canonical entry (stdlib only)

        mark_closing(f"record render {PERSIST_DONE} · job {i}")
        # close the last stage (empties the replicator annotator cache) while Kit lives
        print(_next_stage(app), flush=True)
        PG.close_bounded(app)
    os._exit(0)
