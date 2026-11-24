"""The trainer processes of each launch mode (training, smoke, judge, physics render, kinematic replay).

Each function returns the `Call`s a mode starts, with their per-process environment; `dexengine.rl.launch.dispatch`
starts them (or dumps them in a dry run). Training also has follow-up processes (automatic resume after a GPU
memory failure, the post-training video) that `run` builds on demand from the same pieces.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from dexengine.rl.launch import plan as PL
from dexengine.rl.launch import records
from dexengine.rl.launch.argv import Call
from dexengine.rl.launch.plan import Plan
from dexengine.rl.launch.recipe import RecipeError
from dexengine.rl.tasks import switches


def render_look_env(plan: Plan, physx_lite: bool) -> dict:
    """Trainer environment of a rendering process: the render look (owner dexengine.record.looks.render_look), plus
    the light PhysX profile for bimanual renders."""
    from dexengine.record.looks import render_look as RL

    try:
        look = RL.env(plan.req.root, {**os.environ, **plan.env})
    except ValueError as e:
        raise RecipeError(str(e)) from None
    if physx_lite:
        look = {"DEXENGINE_PHYSX_LITE": plan.env.get("DEXENGINE_PHYSX_LITE") or "1", **look}
    print(
        f"[launch] render look: room={Path(look.get('DEXENGINE_AUG_ROOM', 'none')).name} "
        f"desk={Path(look.get('DEXENGINE_AUG_DESK_MDL', 'none')).name} "
        f"key={look.get('DEXENGINE_KEY_LIGHT', 'default')}",
        flush=True,
    )
    return look


def train_call(plan: Plan, *, exp: str | None = None, extra_tail: str = "") -> Call:
    """The training process (also used by automatic resume with `exp` = the resume run and `extra_tail` =
    `--resume_from … --resume_iteration …`)."""
    req, v = plan.req, plan.values
    if req.path == "bi":
        flags = f"{PL.arm_gains(v)} {PL.bi_train_flags(plan)} --no-frozen-eval {PL.policy_flags(plan)} {plan.extra}"
        env, group = PL.bi_env(plan), req.wandb_group or "bi_b1"
    else:
        flags = f"{PL.uni_train_flags(plan)} --no-frozen-eval {PL.policy_flags(plan)} {plan.extra} "
        env, group = {}, req.wandb_group or "uni"
    return Call(
        exp=exp or req.run,
        extra_flags=f"{flags} {extra_tail}".rstrip() if extra_tail else flags,
        entry="train",
        ne=v["NE"],
        mi=v["MI"],
        wandb_group=group,
        bimanual=req.path == "bi",
        seq_dir=req.seq_dir,
        env=env,
        gpu=req.gpu,
    )


def smoke_call(plan: Plan) -> Call:
    """64-env 1-iteration boot check (no checkpoint, no W&B)."""
    req, v = plan.req, plan.values
    if req.path == "bi":
        flags = f"{PL.arm_gains(v)} {PL.bi_train_flags(plan)} --no-frozen-eval {PL.policy_flags(plan)} {plan.extra}"
        exp, env = req.out, PL.bi_env(plan)
    else:
        flags = f"{PL.uni_train_flags(plan)} --no-frozen-eval {PL.policy_flags(plan)} {plan.extra} "
        exp, env = req.run, {}
    return Call(
        exp=exp,
        extra_flags=flags,
        entry="train",
        smoke=True,
        bimanual=req.path == "bi",
        seq_dir=req.seq_dir,
        env={**env, "DEXENGINE_REFBASE_SMOKE": "1"},
        gpu=req.gpu,
    )


#: Env count of a physics render (the post-training video, the one-hand FROZEN_RENDER_OUT render): the render draws
#: env 0 clip after clip (clip_video._run_per_clip_video_render), so one env — never the recipe NE (a 4096-env render
#: scene with replicate_physics hits the rendering kit's scene-build regression: 94 s → 1081 s, state_rl_accel).
RENDER_NE = "1"


def render_ne(plan: Plan, entry: str) -> str:
    """NE of a frozen-replay process that renders (judge_entry), owned by its entry — the recipe NE is never inherited:
    render = RENDER_NE (a FROZEN_REPLAY_NE other than that dies) · replay = the caller's FROZEN_REPLAY_NE, which must
    be given (one env per recorded clip of the trajectory file — cell_session states the bundle's clip count)."""
    given = plan.req.judge_ne
    if entry == "render":
        if given and given != RENDER_NE:
            raise RecipeError(
                f"FROZEN_REPLAY_NE={given}: a physics render runs {RENDER_NE} env (clips in turn on env 0) — "
                "leave FROZEN_REPLAY_NE out or set it to that"
            )
        return RENDER_NE
    if not given:
        raise RecipeError(
            "a trajectory replay (DEXENGINE_TRAJ_RENDER) needs FROZEN_REPLAY_NE = the recorded clip count (one env "
            "per clip) — the recipe NE is not a render's env count"
        )
    return given


def judge_entry(plan: Plan) -> str:
    """Entry of a frozen-replay process (dexengine.rl.app): the judge itself (plain kit, never renders), or a render
    under the judge's frozen conditions — FROZEN_RENDER_OUT = physics render of the checkpoint's rollouts (render) ·
    DEXENGINE_TRAJ_RENDER = kinematic replay of a recorded trajectory file (replay). Log and video names stay the
    judge's (`<run>_frozenreplay`, the given video folder)."""
    if plan.req.frozen_render_out:
        return "render"
    return "replay" if plan.env.get("DEXENGINE_TRAJ_RENDER") else "judge"


def judge_call(plan: Plan) -> Call:
    """Fresh-process judge of one checkpoint: warm-up + PARK batches inside the trainer (`dexengine.rl.judge`) — or,
    with a render request, a render process at the same frozen point (judge_entry)."""
    req, v = plan.req, plan.values
    resume = PL.judge_resume(plan)
    entry = judge_entry(plan)
    env = {"DEXENGINE_FROZEN_REPLAY": "1", "DEXENGINE_FROZEN_REPLAY_ITER": req.judge_iter or "0"}
    if req.judge_epc:
        env["DEXENGINE_FROZEN_REPLAY_EVALS_PER_CLIP"] = req.judge_epc
    render, look = "", {}
    if entry != "judge":
        look = render_look_env(plan, physx_lite=req.path == "bi")
        env.update(look, DEXENGINE_FROZEN_RENDER="1")
        render = (
            f"--render_per_clip_videos --enable_cameras --videos_out_dir {req.traj_render_out or req.frozen_render_out}"
        )
    if req.path == "bi":
        flags = f"{PL.arm_gains(v)} {PL.bi_train_flags(plan)} {PL.policy_flags(plan)} {plan.extra} {resume}" + (
            f" {render}" if render else ""
        )
        env, gpu = {**PL.bi_env(plan), **env}, req.gpu
    else:
        flags = (
            f"{PL.uni_train_flags(plan)} {PL.policy_flags(plan)} {plan.extra}"
            + (f" {render}" if render else "")
            + f" {resume} --enable_contact_termination {PL.judge_contact_term()}"
        )
        gpu = req.gpu.split(",")[0]
    return Call(
        exp=req.out,
        extra_flags=flags,
        entry=entry,
        ne=(req.judge_ne or v["NE"]) if entry == "judge" else render_ne(plan, entry),
        mi="1",
        wandb=False,
        bimanual=req.path == "bi",
        seq_dir=req.seq_dir,
        env=env,
        gpu=gpu,
        context_switches=tuple(k for k in look if k in switches.FIELD_OF),
    )


def render_out(req, name: str) -> str:
    """Default video folder `name` of a render or replay launch: the run tree's RUN_RENDER_OUT, in the argv form
    (records.arg_path — relative only when the run tree is the trainer's cwd)."""
    from dexengine.core import paths as P

    return records.arg_path(req, P.run_out(Path(P.data_layout.RUN_RENDER_OUT) / name, req.run_tree))


def ckpt_dir(plan: Plan) -> Path:
    """Checkpoint folder a render reads: <render_rundir>/checkpoints, else the run's own."""
    req = plan.req
    base = Path(req.render_rundir) if req.render_rundir else records.run_dir(req.run, req.run_tree)
    return base / "checkpoints"


def render_call(plan: Plan, ckpt: str, tag: str, out_dir: str, look: dict) -> Call:
    """Bimanual physics render of one checkpoint (best of 8 rollouts per clip)."""
    req, v = plan.req, plan.values
    sel = PL.render_select(req.markers, req.videos_only_clips)
    flags = (
        f"{PL.arm_gains(v)} {PL.bi_train_flags(plan)}  {sel} {PL.policy_flags(plan)} "
        f"{plan.extra} --resume_from {ckpt} --render_per_clip_videos --enable_cameras "
        f"--videos_warmup_attempts 0 --videos_out_dir {out_dir}"
    )
    env = {**PL.bi_env(plan), "DEXENGINE_FROZEN_REPLAY": "1", "DEXENGINE_FROZEN_RENDER": "1", **look}
    return Call(
        exp=f"{req.run}_vid{tag}",
        extra_flags=flags,
        entry="render",
        ne="1",
        mi="1",
        wandb_group=req.wandb_group,
        bimanual=True,
        seq_dir=req.seq_dir,
        env=env,
        gpu=req.gpu.split(",")[0],
        process_env={"WANDB_MODE": "disabled"},
        context_switches=tuple(k for k in look if k in switches.FIELD_OF),
    )


def render_calls(plan: Plan, look: dict) -> list[Call]:
    """One render per requested checkpoint that exists (missing ones are reported and skipped)."""
    req, d = plan.req, ckpt_dir(plan)
    if not d.is_dir():
        raise RecipeError(f"checkpoint folder missing: {d}")
    out = []
    for ck in req.render_ckpts:
        p = d / ck
        if not p.is_file():
            print(f"[launch] missing checkpoint {p} — skipped", flush=True)
            continue
        tag = f"_{ck.removesuffix('.pt')}_B"
        out.append(render_call(plan, str(p), tag, req.render_out_dir or render_out(req, f"{req.run}{tag}"), look))
    return out


def replay_out_dir(plan: Plan) -> str:
    return plan.req.render_out_dir or render_out(plan.req, f"{plan.req.run}_replay4view")


def replay_call(plan: Plan, view_dir: str, clip: str, look: dict) -> Call:
    """Kinematic reference replay (no policy) of one clip, four views."""
    req, v = plan.req, plan.values
    if req.path == "bi":
        flags = (
            f"{PL.arm_gains(v)} {PL.policy_flags(plan)} {PL.PRESET_TRAIN_FLAGS} --render_markers {req.markers} "
            f"{plan.extra}"
        )
        env = {**PL.bi_env(plan, gate=False), "DEXENGINE_ALLOW_BI_EVAL_PATH": "1"}
    else:
        flags = f"{PL.uni_eval_flags(plan)} {PL.policy_flags(plan)} --render_markers {req.markers} {plan.extra}"
        env = {}
    return Call(
        exp=req.run,
        extra_flags=flags,
        entry="replay",
        task="eval",
        bimanual=req.path == "bi",
        seq_dir=view_dir,
        render_out=replay_out_dir(plan),
        render_tag=f"_{clip}",
        env={**env, **look},
        gpu=req.gpu.split(",")[0],
    )


def replay_clips(plan: Plan) -> list[str]:
    """Clip folders a replay renders: the clip list (DEXENGINE_SEQ_LIST) or every entry of SEQ_DIR."""
    seq_dir, lst = plan.req.seq_dir, plan.env.get("DEXENGINE_SEQ_LIST", "")
    if lst:
        r = subprocess.run(
            ["python3", "-m", "dexengine.record.replay_roster", "--seq-dir", seq_dir, "--seq-list", lst],
            cwd=plan.req.root,
            env={**os.environ, **plan.env},
            capture_output=True,
            text=True,
        )
        if r.returncode:
            raise RecipeError(
                f"[REPLAY_LIST] clip list failed: DEXENGINE_SEQ_LIST={lst} SEQ_DIR={seq_dir}: {r.stderr.strip()[-300:]}"
            )
        clips = [c for c in r.stdout.splitlines() if c]
        print(f"[launch] [REPLAY_LIST] {len(clips)} clips from {lst}", flush=True)
        return clips
    clips = sorted(e.name for e in Path(seq_dir).iterdir() if e.is_dir())
    print(f"[launch] [REPLAY_LIST] no clip list — every folder of SEQ_DIR ({len(clips)})", flush=True)
    return clips
