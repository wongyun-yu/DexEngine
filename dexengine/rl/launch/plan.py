"""From a launch request to the trainer processes it runs: recipe values, environment and per-process flags.

`prepare` applies the entry rules once (robot flags, per-hand recipe, bimanual collision preset and data checks);
`calls` lists the trainer processes of the request's mode. Flag strings are assembled in the historical order (see
`dexengine.rl.launch.argv`): branch flags first, then the caller's extra flags, so the caller can still override a
branch flag explicitly and the recorded `EXTRA=` line keeps its meaning.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from dexengine.rl.judge import terms as judge_terms
from dexengine.rl.launch import hand_recipe, preflight, recipe, records
from dexengine.rl.launch.argv import split_flags
from dexengine.rl.launch.recipe import RecipeError
from dexengine.rl.launch.request import Request
from dexengine.rl.tasks.scene import collision_presets

#: Collision presets a bimanual launch accepts -> the inter-articulation rows they turn off
#: (r2 = right hand/arm vs left hand/arm, r3l = left arm/hand vs torso). Owner of the rows: collision_presets.
PRESET_SPLIT_OFF = {"canon_r2off": "r2", "canon_r3off": "r3l", "canon_r2r3off": "r2,r3l"}
#: Trainer environment every canonical preset requires; a caller value that differs is refused (exit 7).
PRESET_ENV = {
    "DEXENGINE_BI_SPLIT_ARTIC": "1",
    "DEXENGINE_BI_SELFCOLL": "0",
    "DEXENGINE_DESK_ROBOT_NOCOLL": "0",
    "DEXENGINE_ARM_OBJ_NOCOLL": "0",
    "DEXENGINE_EVAL_CROSS_OFF": "0",
}
#: Flags a preset owns (it passes them itself; a caller copy in EXTRA would win silently).
PRESET_FLAGS = (
    "--collision_curriculum",
    "--collision_curriculum_objpair",
    "--collision_curriculum_on_frac",
    "--enable_contact_termination",
)
PRESET_TRAIN_FLAGS = "--collision_curriculum 0 --enable_contact_termination 0"
#: Rollouts per PPO iteration (checkpoint agent_<iteration × rollouts>.pt).
ROLLOUTS_DEFAULT = "32"


@dataclasses.dataclass
class Plan:
    """A prepared launch: the request, its recipe values, the environment its trainer processes share and the
    caller flags after the entry rules."""

    req: Request
    values: dict
    env: dict
    extra: str
    preset: str = ""
    combo_variant: str = ""


def _entry_extra(req: Request, env: dict) -> str:
    """Caller flags after the entry rules: robot flags, per-hand recipe (exports go into `env`)."""
    extra = req.extra if req.extra is not None else f"--arm {req.arm} --hand {req.hand}"
    try:
        extra, exports, marker = hand_recipe.apply_launch(req.hand, extra, env, ctx="launch")
    except hand_recipe.RecipeError as e:
        raise RecipeError(f"[HAND_RECIPE] {e}") from e
    print(marker, flush=True)
    env.update(exports)
    toks = split_flags(extra)
    for flag, want in (("--arm", req.arm), ("--hand", req.hand)):
        got = [toks[i + 1] for i, t in enumerate(toks[:-1]) if t == flag]
        if got and got[-1] != want:
            raise RecipeError(
                f"EXTRA {flag} {got[-1]} differs from the launch robot {want}: the data resolver and "
                "the trainer would see different robots"
            )
    return extra


def prepare(req: Request) -> Plan:
    """Resolve the recipe and the shared trainer environment of a request; run the bimanual data checks."""
    env = dict(req.env)
    for k in ("DEXENGINE_SNAPSHOT_SRC", "DEXENGINE_SNAP_DIR", "DEXENGINE_RL_SWITCHES"):  # each process gets its own
        env.pop(k, None)
    had = {k: env[k] for k in (*PRESET_ENV, "DEXENGINE_BI_SPLIT_OFF") if k in env}
    values = recipe.resolve(req.path, req.arm, req.hand, {**env, **req.overrides})
    env.update(recipe.trainer_env(values))
    env.update(
        DEXENGINE_ARM=req.arm,
        DEXENGINE_HAND=req.hand,
        DEXENGINE_DATASET=req.dataset,
        DEXENGINE_EMBODIMENT=req.embodiment,
        DEXENGINE_ROBOT_Z_OFF="0",
    )
    if env.get("DEXENGINE_INTRAIN_EVAL") != "0":
        raise RecipeError(
            f"DEXENGINE_INTRAIN_EVAL={env.get('DEXENGINE_INTRAIN_EVAL')!r}: only 0 (the fresh-process "
            "judge scores; evaluation inside training was removed)"
        )
    if env.get("DEXENGINE_FINAL_EVAL", "0") != "0":
        raise RecipeError("DEXENGINE_FINAL_EVAL needs evaluation inside training (removed)")
    extra = _entry_extra(req, env)
    plan = Plan(req=req, values=values, env=env, extra=extra)
    if req.path == "bi":
        _prepare_bimanual(plan, had)
    print(
        f"[launch] {req.path} {req.mode} RUN={req.run} out={req.out} GPU={req.gpu} NE={values['NE']} "
        f"MI={values['MI']} KL={values['KL_THRESH']} combo={req.arm}/{req.hand} "
        f"USD={values['DEXENGINE_COMBO_USD_VARIANT']}",
        flush=True,
    )
    return plan


def _prepare_bimanual(plan: Plan, had: dict) -> None:
    req, v, env = plan.req, plan.values, plan.env
    preset = v["COLL_PRESET"]
    if preset not in PRESET_SPLIT_OFF:
        raise RecipeError(f"COLL_PRESET must be one of {sorted(PRESET_SPLIT_OFF)} (got {preset!r})")
    want = {**PRESET_ENV, "DEXENGINE_BI_SPLIT_OFF": PRESET_SPLIT_OFF[preset]}
    for k, got in had.items():
        if got != want[k]:
            print(
                f"[launch] COLL_PRESET={preset} refused: the caller set {k}={got}, the preset needs {want[k]}",
                flush=True,
            )
            raise SystemExit(7)
    env.update(want, DEXENGINE_COLL_PRESET=preset)
    if any(t in PRESET_FLAGS for t in split_flags(plan.extra)):
        print(
            f"[launch] COLL_PRESET={preset} refused: EXTRA already carries a preset flag {PRESET_FLAGS} "
            f"(the preset passes them; EXTRA=[{plan.extra}])",
            flush=True,
        )
        raise SystemExit(7)
    print(f"[COLL_PRESET] {preset}: {collision_presets.rows(preset)} contact_term=0", flush=True)
    plan.preset = preset
    plan.extra = f"--collision_curriculum 0 {plan.extra}"
    if " --obj_track_grace " not in f" {plan.extra} ":
        plan.extra += f" --obj_track_grace {v['OBJ_TRACK_GRACE']}"
    if v["RESET_SCHEME"] not in ("or", "park"):
        raise RecipeError(f"RESET_SCHEME must be or|park (got {v['RESET_SCHEME']!r})")
    if v["RESET_SCHEME"] == "park" and v["PER_SIDE_ADV"] != "1":
        raise RecipeError("RESET_SCHEME=park needs PER_SIDE_ADV=1")
    plan.combo_variant = preflight.check_bimanual(
        req.seq_dir, req.arm, req.hand, req.dataset, req.embodiment, env, req.root
    )
    env.setdefault("DEXENGINE_COMBO_VARIANT", plan.combo_variant)


# -- flag strings (one owner per segment) --------------------------------------------------------------------------


def arm_gains(v: dict) -> str:
    """Arm PD gain flags (empty for an arm without joint PD)."""
    return f"--arm_kp {v['ARM_KP']} --arm_kd {v['ARM_KD']}" if v["ARM_JOINT_PD"] == "1" else ""


def hinge(v: dict) -> str:
    return (
        f"--wrist_l2_hinge {v['WRIST_HINGE']} --wrist_l2_hinge_free_frac {v['HINGE_FREE']} "
        f"--wrist_l2_hinge_start_frac {v['HINGE_T0']} --wrist_l2_hinge_end_frac {v['HINGE_T1']}"
    )


def wrist_caps(v: dict) -> str:
    return f"--wrist_pos_residual_max_m {v['WRIST_POS_MAX']} --wrist_rot_residual_max_deg {v['WRIST_ROT_MAX']}"


def bi_train_flags(plan: Plan) -> str:
    """Bimanual training flags shared by training, judge and physics render (gates, reset, credit, wrist, parity)."""
    v = plan.values
    posm = f"{float(v['POSC']) / 100:g}"
    return (
        f"--obj_pos_thresh_start 0.10 --obj_rot_thresh_start 180 --tighten_steps_scale {v['TIGHTEN']} "
        f"--obj_pos_thresh_const_left {posm} --obj_rot_thresh_const_left {v['ROTC']} "
        f"--reset_scheme {v['RESET_SCHEME']} --per_side_advantage {v['PER_SIDE_ADV']} {wrist_caps(v)} "
        f"{hinge(v)} --strip_power_reward {v['STRIP_POWER']} --force_window_post {v['FDIR_POST']} "
        f"--obj_lag_window {v['LAGW']} --obj_lag_window_ahead {v['LAGAHEAD']} "
        f"--wrist_ref_lookahead_frames {v['LOOKAHEAD']} --ik_target_ema {v['IK_EMA']} "
        f"--wrist_action_rate_weight {v['ARW']} --finger_action_rate_weight {v['FARW']} {PRESET_TRAIN_FLAGS}"
    )


def policy_flags(plan: Plan) -> str:
    v = plan.values
    if plan.req.path == "bi":
        return f"--policy_arch token --policy_size {v['POLICY_SIZE'] or 'tiny'} --mini_batches {v['MB']}"
    return "--policy_arch token --policy_size tiny" + (f" --mini_batches {v['MB']}" if v.get("MB") else "")


def uni_side_flag(plan: Plan) -> str:
    return "--side left" if plan.req.side == "left" else ""


def uni_common(v: dict) -> str:
    gains = arm_gains(v)
    return (
        f"--strip_power_reward {v['STRIP_POWER']} --force_window_post {v['FDIR_POST']} "
        f"--obj_lag_window {v['LAGW']} {gains + ' ' if gains else ''}--wrist_action_rate_weight {v['ARW']} "
        f"--finger_action_rate_weight {v['FARW']} --ik_target_ema {v['IK_EMA']} "
        f"--wrist_ref_lookahead_frames {v['LOOKAHEAD']} --obj_lag_window_ahead {v['LAGAHEAD']}"
    )


def uni_train_flags(plan: Plan) -> str:
    v = plan.values
    return (
        f"{uni_side_flag(plan)} --obj_pos_thresh_start 0.10 --obj_rot_thresh_start 180 "
        f"--tighten_steps_scale {v['TIGHTEN']} {wrist_caps(v)} {hinge(v)} {uni_common(v)}"
    )


def uni_eval_flags(plan: Plan) -> str:
    return (
        f"{uni_side_flag(plan)} --wrist_pos_residual_max_m 0.07 --wrist_rot_residual_max_deg 120 "
        f"{uni_common(plan.values)}"
    )


def bi_env(plan: Plan, gate: bool = True) -> dict:
    """Per-process environment of bimanual trainer processes (left-arm gain scale, warm-up judge, left gate)."""
    e = {"DEXENGINE_LEFT_ARM_KP_SCALE": "1.0", "DEXENGINE_FROZEN_STRICT": "1"}
    if gate:
        e["DEXENGINE_LEFT_GATE_RAMP"] = plan.values["LGATE_RAMP"]
    return e


def final_ckpt(plan: Plan) -> Path:
    """The run's final checkpoint (absolute, in the launch's run tree): agent_<MI × rollouts>.pt."""
    roll = int(plan.env.get("DEXENGINE_ROLLOUTS") or ROLLOUTS_DEFAULT)
    ckpts = records.run_dir(plan.req.run, plan.req.run_tree) / "checkpoints"
    return ckpts / f"agent_{int(plan.values['MI']) * roll}.pt"


def judge_resume(plan: Plan) -> str:
    """`--resume_from <ckpt>` of a judge, or "" for the policy-free PARK probe (ckpt "none")."""
    if plan.req.ckpt != "none":
        return f"--resume_from {plan.req.ckpt}"
    if plan.env.get("DEXENGINE_PARK_PROBE", "0") == "0":
        raise RecipeError("FROZEN_REPLAY_CKPT=none needs DEXENGINE_PARK_PROBE=<cut step> (a policy-free probe)")
    return ""


def judge_contact_term() -> str:
    """Contact-termination value of the one-hand judge (owner dexengine.rl.judge.terms)."""
    value, _src = judge_terms.contact_term()
    return str(value)
