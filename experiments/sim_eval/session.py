"""Isaac entry of one closed-loop evaluation session.

It relaunches the canonical frozen render dump of the scene (written by `prep`, `RENDER_ONLY=<checkpoint>`) and
attaches `closed_loop.run` to the trainer's render seam (dexengine.rl.session_render) through the record session's two
hooks (camera/Kit settings check at gymnasium.make, render-function attach). Neither the record code nor the training
code is changed.

Use: `E5_JOB=<job_run.json> isaaclab.sh -p experiments/sim_eval/session.py` (run_scene does this).
job_run.json = the scene's job.json + {out, modes, cap_mult, episode_arrays, policy/policies, eval}.
Markers: `[COLLECT_SESSION] hook: …` (the hook fired — without it the run is invalid) · `[E5_EVAL] DONE …`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dexengine.record.render import cameras as CC
from dexengine.record.session import entry as SS  # _render_hooks (the record session's hooks)
from dexengine.record.session import kit as KX
from dexengine.record.session.runner import _STATE, MARK, load_dump, require, run_entry, set_arg
from dexengine.rl.tasks import switches
from experiments.sim_eval.session_cfg import value as cfg


def main() -> None:
    job = json.loads(Path(os.environ["E5_JOB"]).read_text())
    plan = json.loads(Path(job["plan"]).read_text())
    argv, env = load_dump(job["dump"])
    argv = KX.apply(argv)
    require(argv, "--record_session", "--enable_cameras")
    argv = KX.apply_render_threads(argv)
    argv = KX.apply_driver_check(argv)
    if os.path.realpath(plan["seq_dir"]) != os.path.realpath(job["seq_dir"]):
        raise SystemExit(f"[E5_EVAL] plan seq_dir {plan['seq_dir']} ≠ job {job['seq_dir']}")
    set_arg(argv, "--sequence_dir", job["seq_dir"])
    videos = Path(job["out"]) / "_videos"
    videos.mkdir(parents=True, exist_ok=True)
    set_arg(argv, "--videos_out_dir", str(videos))
    env = dict(env, **{CC.ENV_SWITCH: "1"})
    if job.get("switches"):  # RL switches of this evaluation over the dump's (run_scene: profile + eval cfg)
        env[switches.SWITCHES_ENV] = switches.derive(
            env[switches.SWITCHES_ENV], job["switches"], Path(job["out"]) / "switches.json"
        )
    env.update(
        COLLECT_OUT=job["out"],
        COLLECT_PLAN=os.path.abspath(job["plan"]),
        DEXENGINE_SEQ_LIST=plan["seq_list"],
        E5_JOB=os.environ["E5_JOB"],
    )
    # place_marker (P&P): the policy sees the same place marker as the record render. The render session reads it from
    # the plan only (markers.style — the plan's place_marker field), so a copy of this scene's plan with that field
    # goes to the session output and COLLECT_PLAN points there; the original plan is untouched. An unknown style dies.
    # Proof that the marker was placed: the `[PLACE_MARKER]` line of the session log.
    pm = cfg("place_marker", "")
    if pm:
        from dexengine.record.render import markers as PM

        PM._valid(pm, "place_marker")
        pp = Path(job["out"]) / "_plan_place_marker.json"
        pp.parent.mkdir(parents=True, exist_ok=True)
        pp.write_text(json.dumps(dict(plan, **{PM.PLAN_KEY: pm}), indent=1))
        env["COLLECT_PLAN"] = str(pp)
        print(f"[E5_PLACE_MARKER] {pm} · plan copy {pp} (original {job['plan']})", flush=True)
    if env.get("DEXENGINE_FROZEN_RENDER") != "1" or env.get("DEXENGINE_FROZEN_REPLAY") != "1":
        raise SystemExit(
            "[E5_EVAL] the dump is not the frozen render branch (DEXENGINE_FROZEN_REPLAY=1 ∧ "
            "DEXENGINE_FROZEN_RENDER=1) — the physics would differ from the judge and stage 1"
        )
    from experiments.sim_eval import closed_loop as E5

    run_entry(argv, env, SS._render_hooks(E5.run, "closed_loop.run"))
    raise SystemExit(
        f"{MARK} the session app returned — closed_loop.run was not called or left early (DONE {_STATE['done']})"
    )


if __name__ == "__main__":
    main()
