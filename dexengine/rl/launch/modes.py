"""The trainer processes of each launch mode (training, smoke, judge, and the dump-only render and replay modes).

Each function returns the `Call`s a mode starts, with their per-process environment; `dexengine.rl.launch.dispatch`
starts them (or dumps them in a dry run). Training also has a follow-up process (automatic resume after a GPU memory
failure) that `run` builds on demand from the same pieces.

The render and replay modes (and the one-hand judge's FROZEN_RENDER_OUT) only write dumps (DRYRUN): a
record session (scripts/stage4_record.py; experiments.sim_eval) replays a dump and attaches its render function to
the trainer's seam (--record_session, dexengine.rl.session_render). The RL launcher starts no drawing process.
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


def scene_env(plan: Plan, physx_lite: bool) -> dict:
    """Trainer environment of a drawing process's dump: the build-time scene inputs (owner
    dexengine.record.looks.scene_env; the look itself is applied on the stage by dexengine.record.looks.look_d), plus
    the light PhysX profile for bimanual scenes. A PRESET (the old render-look preset) dies: there is one look."""
    from dexengine.record.looks import scene_env as SE

    environ = {**os.environ, **plan.env}
    if environ.get("PRESET"):
        raise RecipeError(f"PRESET={environ['PRESET']!r} removed — one look (dexengine.record.looks.look_d)")
    look = SE.env(environ)
    if physx_lite:
        look = {"DEXENGINE_PHYSX_LITE": plan.env.get("DEXENGINE_PHYSX_LITE") or "1", **look}
    print(
        f"[launch] record scene env: key={look.get(SE.KEY_LIGHT, environ.get(SE.KEY_LIGHT))} "
        f"focal={look.get(SE.FOCAL, environ.get(SE.FOCAL))} physx_lite={look.get('DEXENGINE_PHYSX_LITE', '-')}",
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


#: Env count of a frozen-render process (the bimanual render dump, the one-hand FROZEN_RENDER_OUT dump): the record
#: session draws env 0 clip after clip, so one env — never the recipe NE (a 4096-env render scene with
#: replicate_physics hits the rendering kit's scene-build regression: 94 s → 1081 s, state_rl_accel).
RENDER_NE = "1"


def render_ne(plan: Plan) -> str:
    """NE of a frozen-render process — the recipe NE is never inherited: RENDER_NE (a FROZEN_REPLAY_NE other than that
    dies)."""
    given = plan.req.judge_ne
    if given and given != RENDER_NE:
        raise RecipeError(
            f"FROZEN_REPLAY_NE={given}: a frozen render runs {RENDER_NE} env (clips in turn on env 0) — "
            "leave FROZEN_REPLAY_NE out or set it to that"
        )
    return RENDER_NE


def judge_entry(plan: Plan) -> str:
    """Entry of a frozen-replay process (dexengine.rl.app): the judge itself (plain kit, never renders), or — with
    FROZEN_RENDER_OUT (dump only) — the judge's frozen point handed to a record session (render). Log names stay the
    judge's (`<run>_frozenreplay`)."""
    return "render" if plan.req.frozen_render_out else "judge"


def judge_call(plan: Plan) -> Call:
    """Fresh-process judge of one checkpoint: warm-up + PARK batches inside the trainer (`dexengine.rl.judge`) — or,
    with FROZEN_RENDER_OUT, the dump of a record-session process at the same frozen point (judge_entry)."""
    req, v = plan.req, plan.values
    resume = PL.judge_resume(plan)
    entry = judge_entry(plan)
    env = {"DEXENGINE_FROZEN_REPLAY": "1", "DEXENGINE_FROZEN_REPLAY_ITER": req.judge_iter or "0"}
    if req.judge_epc:
        env["DEXENGINE_FROZEN_REPLAY_EVALS_PER_CLIP"] = req.judge_epc
    render, look = "", {}
    if entry != "judge":
        look = scene_env(plan, physx_lite=req.path == "bi")
        env.update(look, DEXENGINE_FROZEN_RENDER="1")
        render = f"--record_session --enable_cameras --videos_out_dir {req.frozen_render_out}"
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
        ne=(req.judge_ne or v["NE"]) if entry == "judge" else render_ne(plan),
        mi="1",
        wandb=False,
        bimanual=req.path == "bi",
        seq_dir=req.seq_dir,
        env=env,
        gpu=gpu,
        context_switches=tuple(k for k in look if k in switches.FIELD_OF),
    )


def render_out(req, name: str) -> str:
    """Default output folder `name` of a render or replay dump: the run tree's RUN_RENDER_OUT, in the argv form
    (records.arg_path — relative only when the run tree is the trainer's cwd)."""
    from dexengine.core import paths as P

    return records.arg_path(req, P.run_out(Path(P.data_layout.RUN_RENDER_OUT) / name, req.run_tree))


def ckpt_dir(plan: Plan) -> Path:
    """Checkpoint folder a render reads: <render_rundir>/checkpoints, else the run's own."""
    req = plan.req
    base = Path(req.render_rundir) if req.render_rundir else records.run_dir(req.run, req.run_tree)
    return base / "checkpoints"


def render_call(plan: Plan, ckpt: str, tag: str, out_dir: str, look: dict) -> Call:
    """Bimanual frozen-render dump of one checkpoint (a record session draws it; contact termination off as in the
    judge)."""
    req, v = plan.req, plan.values
    flags = (
        f"{PL.arm_gains(v)} {PL.bi_train_flags(plan)}  --enable_contact_termination 0 {PL.policy_flags(plan)} "
        f"{plan.extra} --resume_from {ckpt} --record_session --enable_cameras --videos_out_dir {out_dir}"
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
    """One render dump per requested checkpoint that exists (missing ones are reported and skipped)."""
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
    """Kinematic replay dump (no policy) of one clip: the scene the record render draws (record stage 2)."""
    req, v = plan.req, plan.values
    if req.path == "bi":
        flags = f"{PL.arm_gains(v)} {PL.policy_flags(plan)} {PL.PRESET_TRAIN_FLAGS} {plan.extra}"
        env = {**PL.bi_env(plan, gate=False), "DEXENGINE_ALLOW_BI_EVAL_PATH": "1"}
    else:
        flags = f"{PL.uni_eval_flags(plan)} {PL.policy_flags(plan)} {plan.extra}"
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
    """Clip folders a replay dump covers: the clip list (DEXENGINE_SEQ_LIST) or every entry of SEQ_DIR."""
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
