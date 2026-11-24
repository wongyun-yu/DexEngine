#!/usr/bin/env python3
"""Trainer entry of the residual-PPO RL stage: one Isaac Lab process, started by `dexengine.rl.launch` as
`isaaclab.sh -p dexengine/rl/cli.py <argv>`.

Modes (--task):
  train -- PPO training of the residual policy (one hand via --side {right,left}, or --bimanual 1)
  eval  -- standalone eval / dual two-policy eval / render / kinematic replay

Order of work:
  1. pre-Kit: parse the argv (`dexengine.rl.args`) and resolve robot, data and objects
     (`dexengine.rl.cli_entry`); nothing here may import numpy or torch (`dexengine.rl.preapp`).
     With DEXENGINE_PREFLIGHT_ONLY=1 the scene-input checks of the env build run next and the process ends
     (`dexengine.rl.cli_entry.preflight`; the launcher's dry run).
  2. AppLauncher starts Kit.
  3. post-Kit: register the gym ids, build the env (`create_env`) and hand a RunContext to the runner
     (`dexengine.rl.train.runner.run_train` / `dexengine.rl.eval.run.run_eval`, imported lazily).
"""

# AppLauncher preamble: nothing before AppLauncher may import Isaac Lab / Omniverse, numpy or torch.
import os
import sys

# Started as a script (`isaaclab.sh -p dexengine/rl/cli.py`), Python puts this file's folder first on sys.path. Its
# sibling modules and packages (args, tasks, train, eval, ...) would then also load as top-level names and could
# shadow other modules of the same name. The package is imported from the repository root (PYTHONPATH, set by the
# launcher) instead.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _HERE]

from dexengine.rl import preapp as _preapp  # stdlib only; records who imports numpy/torch before Kit

_preapp.watch()

from isaaclab.app import AppLauncher

from dexengine.rl.args import build_parser
from dexengine.rl.isaac_cli import (  # stdlib only
    SIM_DEVICE_MARK,
    add_app_launcher_args,
    exit_nonzero_on_uncaught,
    sim_device,
)

parser = build_parser()
add_app_launcher_args(parser)
args_cli = parser.parse_args()

from dexengine.rl.tasks import switches as _switches  # stdlib only

# The switches of this process come from the file the launcher wrote (DEXENGINE_RL_SWITCHES); no file, no run.
print(_switches.require_file(), flush=True)

from dexengine.rl.cli_entry import inputs as _inputs  # stdlib only

_inputs.check_side(parser, args_cli)

from dexengine.rl.cli_entry import robot as _robot  # stdlib + dependency-free registries
from dexengine.rl.cli_entry import seqdir as _seqdir

_robot.resolve_robot(parser, args_cli)

# Tree guard: the imported `dexengine` package must come from this script's checkout. Otherwise (e.g. a worktree
# run importing the package of another checkout) every source change of this tree is silently ignored.
# find_spec does not execute anything, so the check is safe before Kit.
import importlib.util as _ilu_guard

_pkg_spec = _ilu_guard.find_spec("dexengine")
_want_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _pkg_spec is None or _pkg_spec.origin is None:
    parser.error("the dexengine package cannot be found (check PYTHONPATH / the installation).")
# origin = <root>/dexengine/__init__.py -> one level up is <root>
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(_pkg_spec.origin), ".."))
if os.path.realpath(_pkg_root) != os.path.realpath(_want_root):
    parser.error(
        f"tree mismatch: the script is in {_want_root} but the dexengine package comes from {_pkg_root}. "
        f"This tree's source changes would be ignored. Fix: "
        f"run with PYTHONPATH={_want_root}."
    )

_inputs.check_left_uni(parser, args_cli)
_inputs.resolve_eval_threshold(parser, args_cli)
_inputs.resolve_dataset(parser, args_cli)
_seqdir.resolve_data(parser, args_cli)
_active_side, _ACTIVE_OBJ_KEY = _inputs.resolve_active_object(parser, args_cli)
_inputs.check_task(parser, args_cli)

# The entry decides the kit (dexengine.rl.app — owner of the rule and its refusals): train · judge on the plain
# `headless.kit` (no RTX renderer; head_camera leaves the cfg before the env is built, app.training_scene), render ·
# replay on the rendering kit (`headless.rendering.kit`, which enable_cameras=True selects).
from dexengine.rl import app as _TK  # stdlib only

try:
    _ENTRY = _TK.entry_of(os.environ)
    _no_render = _TK.plain_kit(
        os.environ,
        task=args_cli.task,
        video=bool(args_cli.video),
        per_clip_videos=bool(getattr(args_cli, "render_per_clip_videos", False)),
        enable_cameras=bool(getattr(args_cli, "enable_cameras", False)),
    )
except ValueError as _e_tk:
    parser.error(str(_e_tk))
if _no_render:
    print(
        f"{_TK.MARK} — entry={_ENTRY} · enable_cameras not forced · head_camera is removed before the env is built",
        flush=True,
    )
else:
    args_cli.enable_cameras = True
    print(f"{_TK.RENDER_MARK} — entry={_ENTRY} · enable_cameras forced", flush=True)
print(_TK.context_line(_ENTRY, _no_render, args_cli.num_envs), flush=True)

# Kit loads its own numpy first: nothing above may have imported numpy/torch (dies naming the import chain).
_preapp.assert_clean()

from dexengine.rl.cli_entry import preflight as _preflight  # stdlib only

# Preflight (DEXENGINE_PREFLIGHT_ONLY=1, set by the launcher's dry run): the scene-input checks of the env build run
# here and the process ends without starting Kit. They import numpy, which the preapp rule above allows only because
# this process never starts Kit.
if _preflight.requested():
    print(_preflight.run(args_cli), flush=True)
    sys.exit(0)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
exit_nonzero_on_uncaught()  # a module-level exception below also ends with rc 1 (Kit would end with rc 0)
# Training without video (num_envs > 1) turns PhysX→Fabric sync off. Fabric is the render copy: training reads
# state from the PhysX tensor views, and the kit already skips the per-step Fabric update
# (fabricUpdateTransformations=false). With it on, start-up initializes the Fabric hierarchy of every prim on the
# GPU, which dies with the cubric assertion `&out != &b` for XHand (~2,400 prims per env) at 4096 envs.
_render_run = (
    args_cli.video
    or getattr(args_cli, "render_per_clip_videos", False)
    or os.environ.get("DEXENGINE_FROZEN_RENDER", "0") == "1"
)
if args_cli.task == "train" and (args_cli.num_envs is None or args_cli.num_envs > 1) and not _render_run:
    import carb as _carb_fx

    _carb_fx.settings.get_settings().set_bool("/physics/fabricEnabled", False)
    print("[FABRIC] training (no video) — /physics/fabricEnabled=false", flush=True)
_preapp.stop()

# Multi-GPU (--distributed, torchrun): AppLauncher has bound this process to cuda:LOCAL_RANK and skrl's
# `config.torch` initializes the NCCL process group from the torchrun env at its first import (in the runner).
# Here the seed is offset per rank (seed + global_rank; rank 0 keeps the base seed) so every rank simulates
# different episodes; create_env pins cfg.sim.device to this rank's GPU.
if getattr(args_cli, "distributed", False):
    if args_cli.seed is not None:
        args_cli.seed = args_cli.seed + app_launcher.global_rank
    print(
        f"[distributed] rank={app_launcher.global_rank} "
        f"local_rank={app_launcher.local_rank} device=cuda:{app_launcher.local_rank} "
        f"world_size={os.environ.get('WORLD_SIZE', '?')} "
        f"effective seed={args_cli.seed} (base+rank)"
    )

# Post-launch imports (Isaac Lab / Omniverse / torch / gym).
import gymnasium as gym  # noqa: I001  — registry (registers the gym ids) loads before build, launcher order
import yaml

from dexengine.rl.tasks import registry as _registry  # importing it registers the gym ids
from dexengine.rl.env_setup.build import resolve_env_cfg
from dexengine.rl.tasks.scene.physx_buffers import collision_stack  # stdlib only
from dexengine.rl.env_setup.objtrack import (
    apply_phase1_contact_aware_overrides,
    gate_reward_diagnostics,
    apply_finger_imit_release,
)
from dexengine.rl.env_setup.contact_pilot import apply_contact_pilot_overrides

AGENT_YAML_PATH = _registry.AGENT_YAML_PATH

# Both tasks start from the right-uni gym id; create_env remaps it for --bimanual 1 and --side left.
GYM_ID_MAP = {
    "train": _registry.RIGHT_UNI,
    "eval": _registry.RIGHT_UNI,
}
_BIMANUAL_GYM_ID = _registry.BIMANUAL
_BIMANUAL_REMAP_GYM_IDS = (_registry.RIGHT_UNI,)
# Left-uni: the exact left mirror of the right-uni build (left_hand_joint_pos only, the left object in the single
# active-object slot, self-collision off).
_LEFTUNI_GYM_ID = _registry.LEFT_UNI
_LEFTUNI_REMAP_GYM_IDS = (_registry.RIGHT_UNI,)


def _resolve_bimanual_gym_id(gym_id: str) -> str:
    """The bimanual gym id when --bimanual 1 is set and ``gym_id`` is the right-uni id; else ``gym_id``."""
    if getattr(args_cli, "bimanual", 0) and gym_id in _BIMANUAL_REMAP_GYM_IDS:
        print(f"[bimanual] --bimanual 1: remapping gym id {gym_id} -> {_BIMANUAL_GYM_ID}")
        return _BIMANUAL_GYM_ID
    return gym_id


def _resolve_leftuni_gym_id(gym_id: str) -> str:
    """The left-uni gym id when --side left is set and ``gym_id`` is the right-uni id; else ``gym_id``."""
    if getattr(args_cli, "left_uni", 0) and gym_id in _LEFTUNI_REMAP_GYM_IDS:
        print(f"[side=left] remapping gym id {gym_id} -> {_LEFTUNI_GYM_ID}")
        return _LEFTUNI_GYM_ID
    return gym_id


def _residual_act_name() -> str:
    """Name of the single residual action term of the run.

    The left-uni build registers only 'left_hand_joint_pos', so every gate on the residual term name must resolve
    through this helper (a hard-coded 'right_hand_joint_pos' hasattr gate would silently skip its block)."""
    return "left_hand_joint_pos" if getattr(args_cli, "left_uni", 0) else "right_hand_joint_pos"


def create_env(
    gym_id: str,
    num_envs: int,
    data_dir: str,
    sequence: str | None = None,
    seed: int | None = None,
    extra_cfg_fn=None,
    *,
    recipe_path: str,
    recipe_build_cap: bool,
):
    """Create a gym environment with common overrides applied.

    Args:
        gym_id: Registered gymnasium environment ID.
        num_envs: Number of parallel environments.
        data_dir: Trajectory data directory (absolute path).
        sequence: Optional sequence filter.
        seed: Optional random seed.
        extra_cfg_fn: Optional callable(cfg) for task-specific overrides.
        recipe_path: "train" (train runner) | "eval" (eval runner) — the hand-recipe check's path. No default:
            each caller states it (a guessed path would accept the other path's finger cap).
        recipe_build_cap: whether this caller's cfg override writes --finger_effort into the finger actuator
            (train: always · eval: only under DEXENGINE_EVAL_TRAIN_PARITY=1). No default.

    Returns:
        (gym.Env, cfg) tuple.
    """
    return _construct_env(gym_id, num_envs, data_dir, sequence, seed, extra_cfg_fn, recipe_path, recipe_build_cap)


def _construct_env(gym_id, num_envs, data_dir, sequence, seed, extra_cfg_fn, recipe_path, recipe_build_cap):
    """Resolve the gym id and env cfg, apply the run overrides and build the env (the single construction point)."""
    # The single env-construction point: every caller and every gym-id-keyed branch below sees the remapped id.
    gym_id = _resolve_bimanual_gym_id(gym_id)
    gym_id = _resolve_leftuni_gym_id(gym_id)
    CfgClass = resolve_env_cfg(gym_id)
    cfg = CfgClass()
    cfg.scene.num_envs = num_envs
    # The PhysX GPU collision stack follows the env count (physx_buffers.collision_stack, the one rule; checked on the
    # stage after load by [PHYSX_BUFFERS]).
    cfg.sim.physx.gpu_collision_stack_size = collision_stack(num_envs)
    # The env cfg's sim device (buffers, PhysX pipeline) follows AppLauncher's --device; multi-GPU pins it to this
    # rank's device (as Isaac Lab's train scripts do); otherwise the cfg default stays (isaac_cli.sim_device).
    _dist = bool(getattr(args_cli, "distributed", False))
    cfg.sim.device, _src = sim_device(
        args_cli.device, parser.get_default("device"), cfg.sim.device, _dist, app_launcher.local_rank if _dist else 0
    )
    print(f"{SIM_DEVICE_MARK} {cfg.sim.device} (from {_src})", flush=True)
    cfg.trajectory.data_dir = os.path.abspath(data_dir)
    if sequence is not None:
        cfg.trajectory.sequence_filter = sequence
    if getattr(args_cli, "multi_object", 0):
        # Multi-object: env->clip is fixed (the per-env mesh is spawn-time) and the
        # PhysX parser must not assume replicated envs (multi-asset spawner requirement).
        cfg.trajectory.static_clip_assignment = True
        cfg.scene.replicate_physics = False
        print("[create_env] multi_object: static clip assignment; replicate_physics=False")
    if seed is not None:
        cfg.seed = seed

    # Head camera: kept for single-env runs, stripped for multi-env runs (decided here because the cfg
    # __post_init__ cannot see the final num_envs).
    from dexengine.rl.render import replay as _trr  # replay-render scene setup (light module; Isaac imports inside)

    _is_arm_loop = gym_id in _registry.GYM_IDS
    if _no_render:  # [TRAIN_KIT] headless — the training scene (no CameraCfg)
        _TK.training_scene(cfg, _ENTRY)
    if _is_arm_loop and hasattr(cfg.scene, "head_camera") and cfg.scene.head_camera is not None:
        # At num_envs > 1 the GPU descriptor pool (RTX ParameterBlocks + render products) runs out:
        # "Unable to allocate descriptor sets" / "Failed to allocate ParameterBlock resources".
        if num_envs > 1 and not _trr.enabled():
            cfg.scene.head_camera = None
            print(f"[create_env] head_camera stripped (num_envs={num_envs} > 1)")
        else:
            print("[create_env] head_camera retained (num_envs=1)")

    env = _add_clip_video_cameras(
        gym_id, num_envs, extra_cfg_fn, recipe_path, recipe_build_cap, _is_arm_loop, _trr, cfg
    )
    if args_cli.video:
        from dexengine.core.paths import run_out as _run_out  # a relative folder is run-tree relative

        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(_run_out(args_cli.video_dir)),
            step_trigger=lambda step: step == 0,
            video_length=5000,
            disable_logger=True,
        )
    return env, cfg


def _add_clip_video_cameras(gym_id, num_envs, extra_cfg_fn, recipe_path, recipe_build_cap, _is_arm_loop, _trr, cfg):
    """Per-clip video cameras (--render_per_clip_videos) and the replay render settings."""
    # Per-clip video cameras (--render_per_clip_videos). The replay render (render.replay, off by default) has no
    # physics and draws N envs at once with tiled cameras, so it lifts the num_envs == 1 requirement.
    if getattr(args_cli, "render_per_clip_videos", False) and (num_envs == 1 or _trr.enabled()) and _is_arm_loop:
        from dexengine.rl.cli_entry import cameras as _cameras

        _cameras.add_video_cameras(cfg, num_envs, _trr)
    elif _trr.enabled():
        raise RuntimeError(
            f"[create_env] {_trr.ENV_VAR} is set but the render camera block was not entered — "
            f"it needs --render_per_clip_videos "
            f"--enable_cameras and an arm-loop gym id"
        )

    # Contact-aware lag compensation Phase 1 (runtime trajmgr config + obs/reward/termination overrides) and the
    # contact-pilot wiring, before the task-specific extra_cfg_fn so that it can still override them. Flag-gated.
    apply_phase1_contact_aware_overrides(cfg, args_cli, "create_env")
    apply_contact_pilot_overrides(cfg, args_cli, "create_env")

    if extra_cfg_fn is not None:
        extra_cfg_fn(cfg)

    # Finger-pose (joint-angle) tracking, an anti-contortion term for high-DoF hands. With no or zero
    # --finger_pose_weight the term is removed (None: not registered, not logged). Applied to the right
    # `finger_pose` and the bimanual left `finger_pose_left` alike.
    _rewards_cfg = getattr(cfg, "rewards", None)
    if _rewards_cfg is not None:
        _fp_w = getattr(args_cli, "finger_pose_weight", None)
        _fp_k = getattr(args_cli, "finger_pose_k", None)
        for _fp_attr in ("finger_pose", "finger_pose_left"):
            if not hasattr(_rewards_cfg, _fp_attr) or getattr(_rewards_cfg, _fp_attr) is None:
                continue
            if _fp_w is None or float(_fp_w) == 0.0:
                setattr(_rewards_cfg, _fp_attr, None)  # not registered
            else:
                _fp_term = getattr(_rewards_cfg, _fp_attr)
                _fp_term.weight = float(_fp_w)
                if _fp_k is not None:
                    _fp_term.params["k"] = float(_fp_k)
                print(f"[finger_pose] {_fp_attr} reward: weight={_fp_term.weight} k={_fp_term.params.get('k')}")

    # Contact-conditioned release of finger imitation (off = attribute never set). After every reward add/remove
    # above, so its term list is the one the RewardManager builds.
    apply_finger_imit_release(cfg, args_cli, "create_env")

    # Reward diagnostics (off by default): every diag_* term is in cfg.rewards by now (contact/force diags from the
    # contact pilot, keypoint diags from the caller); the weight-0 logging-only ones are stripped unless
    # --enable_reward_diagnostics 1.
    gate_reward_diagnostics(cfg, args_cli)

    # Hand RL recipe check (DEXENGINE_HAND_RECIPE; default on, an explicit 0 prints "off" and checks nothing):
    # after every consumer has written its value into cfg (finger actuator · force-curriculum action terms ·
    # imitation release) and before gym.make, the declared argv/env and the applied cfg must both equal the
    # hand's RL recipe, else SystemExit (dexengine.rl.launch.hand_recipe).
    from dexengine.rl.launch.hand_recipe import verify_env_cfg as _verify_hand_recipe
    from dexengine.rl.tasks.scene import split_artic as _split_artic_hr
    from dexengine.rl.tasks.scene.robot import finger_actuator_key as _fak_hr

    _verify_hand_recipe(
        cfg,
        args_cli,
        finger_actuator_key=_fak_hr(args_cli.hand),
        ctx="create_env",
        path=recipe_path,
        build_cap=recipe_build_cap,
        split_bi=_split_artic_hr.bi_enabled(),
    )

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(gym_id, cfg=cfg, render_mode=render_mode)
    return env


def load_agent_yaml():
    """Load the SKRL PPO YAML configuration (single recipe)."""
    path = AGENT_YAML_PATH
    print(f"[load_agent_yaml] Loading: {os.path.basename(path)}")
    with open(path) as f:
        return yaml.safe_load(f)


def detect_side(gym_id: str) -> str:
    """Detect hand side from gym ID."""
    if "LH" in gym_id:
        return "left"
    return "right"


from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunContext:
    """Entry-module surface handed to the runners (train.runner.run_train / eval.run.run_eval). Everything else
    they need comes from their own imports."""

    args_cli: Any
    active_side: str  # "right" | "left" (the uni build side)
    active_obj_key: str  # f"object_{active_side}" scene entity
    gym_id_map: dict
    create_env: Callable
    load_agent_yaml: Callable
    detect_side: Callable
    residual_act_name: Callable


def main():
    """Build the RunContext and dispatch --task to its runner."""
    # Stall evidence: all thread stacks on the watcher's signal (installed after AppLauncher, so after Kit's handlers).
    from dexengine.core import debug as stall_stack

    stall_stack.arm()
    ctx = RunContext(
        args_cli=args_cli,
        active_side=_active_side,
        active_obj_key=_ACTIVE_OBJ_KEY,
        gym_id_map=GYM_ID_MAP,
        create_env=create_env,
        load_agent_yaml=load_agent_yaml,
        detect_side=detect_side,
        residual_act_name=_residual_act_name,
    )
    task = args_cli.task
    if task == "train":
        from dexengine.rl.train.runner import run_train  # lazy: post-AppLauncher module

        run_train(ctx)
    elif task == "eval":
        from dexengine.rl.eval.run import run_eval  # lazy: post-AppLauncher module

        run_eval(ctx)
    else:
        raise ValueError(
            f"Unknown task '{task}'. main() doesn't know how to dispatch it. "
            "If you added the task name to the parser choices and to GYM_ID_MAP, "
            "you also need to add an elif branch here so the matching run_*() "
            "function is called. Otherwise main() returns 0 and "
            "simulation_app.close() deadlocks on an unstarted env (134 threads "
            "futex_do_wait)."
        )
    return 0


if __name__ == "__main__":
    # On an exception Kit keeps the process alive (and its GPU memory held), which breaks the next run in ways
    # that look like code faults. Print the traceback and hard-exit.
    try:
        ret = main()
    except BaseException:  # noqa: BLE001 — any exit path must print the traceback and hard-exit, or Kit holds the GPU
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
    os._exit(ret or 0)
