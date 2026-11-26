"""Isaac record session entry: start the canonical launch's dump (argv + DEXENGINE_* env) **as it is** and add
only the record hooks.

The canonical code (dexengine.rl.cli, the train and render paths) knows nothing about recording (no flag: record code
is kept apart from inference code). The canonical entry makes the dump: ``DUMP_ONLY=1 … bash scripts/stage3_rl.sh
<alias>`` -> the launcher (dexengine/rl/launch/process.py) writes the trainer's argv and env instead of starting it
(entry rules, per-hand recipe, collision preset and USD variant exactly as canonical). This file reads that dump and
runs the canonical entry in the same process with runpy, hooking each target module right after its first import
(post-import hooks: no numpy or torch before AppLauncher, the [PRE_APP] guard).

Stages (--stage):
  record  stage 1: the trajectory recorder of the frozen entry used by judging (FROZEN_REPLAY_CKPT,
          DEXENGINE_TRAJ_RECORD) is replaced by the record recorder (dexengine/record/rollout/recorder.py: adds the raw
          policy output, absolute joint targets and contact-force columns).
  render  stage 2: the trainer's render seam (dexengine.rl.session_render.render) of the kinematic replay dump (the
          launcher's dump-only RENDER_ONLY=replay) is attached to the record replay render
          (dexengine/record/render/episode.py: writes the chosen trajectory without physics and captures the two ZED
          Mini eyes, depth and exo).
  render_clip  stage 2 per clip: with the same dump, one session draws the clip's trajectories (original + placement
          variants) x look episode plan (--plan, planned by record.episodes) in turn
          (dexengine/record/render/clip_session.py). Dump argv --sequence_dir <- --seq-dir (must equal the plan's
          seq_dir) · env DEXENGINE_SEQ_LIST <- the plan's seq_list (list of all trajectories).
  render_clips  several render_clips in **one process** (worker, --queue, caller session.persist.run; the canonical
          path of render_clip): the Kit app starts once; for each clip the canonical entry runs again from the start
          with that clip's dump (runpy, name != __main__) and main() is called. The render function is the one of
          render_clip; at the end it closes the env and returns -> a new USD stage -> the next clip. Same-session
          condition = same app key (app_key). Between-clip handling and markers: the head of render_clips().
  Plugin stages (record.hooks kind session_stage) add their own --stage names.
Marker: ``[COLLECT_SESSION] hook: …`` (proof that the hook ran; without it the run is invalid).
The run core (dump, hooks, run, finish, record hook) is session/runner.py; the stage-1 cache fingerprint measures that
file only, so render stages added here leave the physics records valid.

Shutdown (root cause, measured stack): if the render function returns without closing the env, the canonical entry's
simulation_app.close() stops the omni.replicator orchestrator -> timeline STOP event -> IsaacLab SimulationContext's
STOP subscription (_app_control_on_stop_handle_fn: 'render() until playing') spins forever (GPU at 80 %, hang). The
canonical frozen judge branch calls wrapped.close() before returning (env.close -> sim.clear_instance removes that
subscription) and exits normally; the record render function does the same with the same public API in finish()
(no os._exit workaround; the process ends through the entry's close -> os._exit).
"""

from __future__ import annotations

import argparse
import os
import sys

from dexengine.record.session import kit as KX  # Kit extension settings (stdlib only, before AppLauncher)
from dexengine.record.session import persist_kit as PK  # Kit side of the Kit-reuse worker (render_clips; stdlib header)
from dexengine.record.session.runner import (
    _STATE,
    HOOK,
    MARK,
    PostImport,
    load_dump,
    prove_hooks,
    record,
    require,
    run_entry,
    set_arg,
)


def _hook_gym(mod):
    from dexengine.record.render import cameras as CC

    _make = mod.make

    def _make_with_cameras(gym_id, *args, cfg=None, **kw):
        if cfg is not None:
            print(CC.add_collect_cameras(cfg), flush=True)
        return _make(gym_id, *args, cfg=cfg, **kw)

    mod.make = _make_with_cameras
    line = f"{HOOK} gymnasium.make → add_collect_cameras, then the original make"
    print(line, flush=True)
    return line


def _render_hooks(fn, name: str) -> dict:
    """The two post-import hooks of the render stages (gymnasium.make cameras and Kit settings check; render function
    swap)."""

    def _render_proved(*args, **kw):
        prove_hooks(["gymnasium"])  # Kit settings applied · gymnasium.make camera hook (before the render)
        if PK._PERSIST["on"]:
            print(PK.rtx_restore(), flush=True)
        return fn(*args, **kw)

    def _hook_render(mod):
        mod.render = _render_proved
        line = f"{HOOK} {mod.HOOK_TARGET} → {name}"
        print(line, flush=True)
        return line

    def _hook_gym_kit(mod):
        dc = KX.check_driver_check()  # empty when no off request (the marker line stays as it was)
        return f"{KX.check_applied()} · {KX.check_render_threads()} · {dc + ' · ' if dc else ''}{_hook_gym(mod)}"

    return {"gymnasium": _hook_gym_kit, "dexengine.rl.session_render": _hook_render}


def _persist_main(queue: str, owner: int, RCL) -> None:
    PK.render_clips(queue, owner, _render_hooks(RCL.render_clip, "collect_render_clip.render_clip"))
    raise SystemExit(f"{PK.PERSIST_MARK} render_clips returned (normally it ends with os._exit)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    from dexengine.record import hooks

    hooks.load_plugins()
    ap.add_argument(
        "--stage",
        required=True,
        choices=("record", "render", "render_clip", "render_clips", *hooks.names("session_stage")),
    )
    ap.add_argument(
        "--dump",
        default="",
        help="canonical dump log (DUMP_ONLY: argv lines + ENV:: lines); required outside render_clips",
    )
    ap.add_argument("--seq-dir", default="", help="render: the dump's SEQ_DIR (tree root, record.tree_ref)")
    ap.add_argument("--out", default="", help="render: output folder")
    ap.add_argument("--traj", default="", help="render: chosen stage-1 trajectory file (.npz)")
    ap.add_argument("--plan", default="", help="render_clip: episode plan plan.json (v1)")
    ap.add_argument("--queue", default="", help="render_clips: worker queue folder (<n>.job; session.persist.run)")
    ap.add_argument("--owner-pid", type=int, default=0, help="render_clips: calling process (the worker ends with it)")
    ap.add_argument("--seq-list", default="", help="plugin stages: session clip list (DEXENGINE_SEQ_LIST)")
    a = ap.parse_args()
    if (a.stage == "render_clips") == bool(a.dump) or (a.stage == "render_clips") != bool(a.queue):
        ap.error("--dump goes with stages other than render_clips, --queue with render_clips only (one is required)")
    if a.stage == "render_clips":
        from dexengine.record.render import clip_phases as RCL

        _persist_main(a.queue, a.owner_pid, RCL)
        return
    if a.stage == "render_clip":
        from dexengine.record.render import clip_phases as RCL

        argv, env = PK.render_clip_launch(a.dump, a.seq_dir, a.plan)
        run_entry(argv, env, _render_hooks(RCL.render_clip, "collect_render_clip.render_clip"))
        raise SystemExit(
            f"{MARK} the canonical entry returned: render.clip_phases.render_clip was not called or "
            f"left early (DONE {_STATE['done']})"
        )
    argv, env = load_dump(a.dump)
    argv = KX.apply(argv)  # Kit registry off · shared extension folder (marker [KIT_EXT])
    if a.stage == "record":
        sys.meta_path.insert(0, PostImport({"gymnasium": KX.check_applied}))
        record(argv, env)
        return
    from dexengine.record.render import cameras as CC

    require(argv, "--kinematic_replay", "--record_session", "--enable_cameras")
    # render-stage worker cap (marker [KIT_THREADS]; stage-1 record untouched)
    argv = KX.apply_render_threads(argv)
    argv = KX.apply_driver_check(argv)  # RTX driver check off (env; default off = argv unchanged)
    set_arg(argv, "--sequence_dir", a.seq_dir)
    set_arg(argv, "--videos_out_dir", os.path.join(a.out, "_videos"))
    env = dict(env, **{CC.ENV_SWITCH: "1", "COLLECT_OUT": a.out})
    plug = hooks.get("session_stage", a.stage)
    if plug is not None:  # plugin stage: may edit argv and env in place; returns the render hook
        fn, name = plug.prepare(a, argv, env)
    else:
        from dexengine.record.render import episode as RP

        env["COLLECT_TRAJ"] = os.path.abspath(a.traj)
        fn, name = RP.render_selected, "collect_render.render_selected"
    run_entry(argv, env, _render_hooks(fn, name))
    # normally never reached (the entry ends with os._exit after the app exit); reaching it means no render call
    raise SystemExit(
        f"{MARK} the canonical entry returned: {name} was not called or left early (DONE {_STATE['done']})"
    )


if __name__ == "__main__":
    main()
