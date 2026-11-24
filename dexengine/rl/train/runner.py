"""PPO training entry: reference-base + residual policy, trained with skrl PPO.

Imported lazily by ``dexengine.rl.cli`` after ``AppLauncher`` started Kit, so the Isaac/skrl imports at module level are
safe. ``run_train(ctx)`` binds the entry module's ``RunContext`` (args and env factories) and runs the stages in order;
each stage lives in its own module of this package:

  startup        launch checks, W&B, rollout length and curriculum horizon
  overrides*     ``train_cfg_overrides`` — the training-only edits of the env cfg
  env_build      env, boot smoke, experiment folder, tee log
  agent_build    policy/value models, PPO config, PPO learner
  trainer_build  resume weights, sequential trainer, iteration-preserving resume
  aux_loss       auxiliary fingertip-force loss
  step_diag      per-step episode/termination counters
  frozen_eval*   frozen eval (curriculum and randomization frozen) — also the fresh-process judge
  frozen_replay  the fresh-process judge entry (``DEXENGINE_FROZEN_REPLAY=1``)
  post_hook      post-interaction hook: non-finite guard, ``[CKPT_SAVED]``, periodic eval, planned restart
  summary        training run, final summary, held-out manifest

The closures that need the live training objects (``train_cfg_overrides``, ``frozen_eval_snapshot``,
``_mem_restart_check``, ``patched_post``) stay thin wrappers inside ``run_train``: they pass the current objects to the
module function at call time, and state attributes (``frozen_eval_snapshot._stop`` and friends) live on the wrapper
object every caller holds.
"""

# First-import order of the skrl / Isaac Lab stack, kept from the original entry module: the modules below load here,
# in this order, before the train modules. Nothing in this module reads them (hence the bare imports and F401).
import os  # noqa: I001  — import order above is the contract, not alphabetical
import sys  # noqa: F401
import datetime  # noqa: F401

import numpy  # noqa: F401
import torch
import torch.nn  # noqa: F401

import skrl
import isaaclab_rl.skrl  # noqa: F401
import skrl.agents.torch.ppo
import skrl.memories.torch
import skrl.resources.preprocessors.torch
import skrl.trainers.torch  # noqa: F401

import dexengine.rl.judge.arm_penetration  # noqa: F401
import dexengine.rl.judge.rollouts  # noqa: F401
import dexengine.rl.tasks.mdp.actions.contract  # noqa: F401
import dexengine.rl.agents.models  # noqa: F401
import dexengine.rl.train.nonfinite_guard  # noqa: F401
import dexengine.rl.train.rollouts  # noqa: F401
import dexengine.rl.judge.snapshot_metrics  # noqa: F401
import dexengine.rl.env_setup.build  # noqa: F401

from dexengine.rl.train import (
    agent_build,
    aux_loss,
    env_build,
    frozen_eval,
    frozen_replay,
    overrides,
    post_hook,
    startup,
    step_diag,
    summary,
    trainer_build,
)
from dexengine.rl.run_context import bind_run_context
from dexengine.rl.train.trainer_build import _resolve_curriculum_step_offset

#: Returned by a split-out step that did not return early (its caller then goes on).
_CONTINUE = object()

# Entry-module context (RunContext), bound by run_train -> bind_run_context.
args_cli = None
create_env = None
GYM_ID_MAP = None
_ACTIVE_OBJ_KEY = None
_active_side = None
load_agent_yaml = None
_residual_act_name = None


def run_train(ctx):
    """PPO training: reference-base + residual policy (single stage). ``ctx`` is the entry module's RunContext."""
    bind_run_context(globals(), ctx)
    (
        _OT_VAR,
        _dist,
        _dist_rank,
        _hr_ledger,
        _launch_ctx,
        _ot_recorder_class,
        extra_slots_ledger_record,
        gym_id,
        num_envs,
        rollouts,
        total_timesteps,
    ) = startup.launch_setup(GYM_ID_MAP=GYM_ID_MAP, args_cli=args_cli, load_agent_yaml=load_agent_yaml)

    def train_cfg_overrides(cfg):
        return overrides.train_cfg_overrides(
            cfg,
            _ACTIVE_OBJ_KEY=_ACTIVE_OBJ_KEY,
            _active_side=_active_side,
            _launch_ctx=_launch_ctx,
            _residual_act_name=_residual_act_name,
            args_cli=args_cli,
            rollouts=rollouts,
            total_timesteps=total_timesteps,
        )

    return _build_and_train(
        _OT_VAR,
        _dist,
        _dist_rank,
        _hr_ledger,
        _ot_recorder_class,
        extra_slots_ledger_record,
        gym_id,
        num_envs,
        rollouts,
        train_cfg_overrides,
    )


def _build_and_train(
    _OT_VAR,
    _dist,
    _dist_rank,
    _hr_ledger,
    _ot_recorder_class,
    extra_slots_ledger_record,
    gym_id,
    num_envs,
    rollouts,
    train_cfg_overrides,
):
    """Build the env, learner and trainer, then judge or train."""
    (
        _aux_loss_log,
        _aux_loss_scale,
        _aux_target_magnitude_log,
        _ckpt_path_at,
        _fast_diag_flush,
        _mem_restart_check,
        diag,
        exp_dir,
        frozen_eval_snapshot,
        history,
        last_eval_iter,
        original_post,
        raw_env,
        runner,
        side,
        timesteps,
        traj_mgr,
        wrapped,
    ) = _build_training(
        _dist, _dist_rank, _hr_ledger, extra_slots_ledger_record, gym_id, num_envs, rollouts, train_cfg_overrides
    )

    return _judge_or_train(
        _OT_VAR,
        _aux_loss_log,
        _aux_loss_scale,
        _aux_target_magnitude_log,
        _ckpt_path_at,
        _dist_rank,
        _fast_diag_flush,
        _mem_restart_check,
        _ot_recorder_class,
        diag,
        exp_dir,
        frozen_eval_snapshot,
        history,
        last_eval_iter,
        original_post,
        raw_env,
        rollouts,
        runner,
        side,
        timesteps,
        traj_mgr,
        wrapped,
    )


def _build_training(
    _dist, _dist_rank, _hr_ledger, extra_slots_ledger_record, gym_id, num_envs, rollouts, train_cfg_overrides
):
    """Env, PPO learner and trainer."""
    cfg, env, exp_dir, exp_name, raw_env, wrapped = env_build.build_env(
        _dist_rank=_dist_rank,
        _hr_ledger=_hr_ledger,
        extra_slots_ledger_record=extra_slots_ledger_record,
        gym_id=gym_id,
        num_envs=num_envs,
        train_cfg_overrides=train_cfg_overrides,
        _active_side=_active_side,
        _residual_act_name=_residual_act_name,
        args_cli=args_cli,
        create_env=create_env,
    )
    agent, timesteps = agent_build.build_agent(
        _dist=_dist,
        exp_dir=exp_dir,
        exp_name=exp_name,
        raw_env=raw_env,
        rollouts=rollouts,
        wrapped=wrapped,
        args_cli=args_cli,
        load_agent_yaml=load_agent_yaml,
    )
    return _build_trainer_and_hooks(
        _dist, _dist_rank, num_envs, rollouts, agent, cfg, env, exp_dir, raw_env, timesteps, wrapped
    )


def _build_trainer_and_hooks(
    _dist, _dist_rank, num_envs, rollouts, agent, cfg, env, exp_dir, raw_env, timesteps, wrapped
):
    """Trainer, auxiliary loss and step diagnostics."""
    runner = trainer_build.build_trainer(
        agent=agent, env=env, rollouts=rollouts, timesteps=timesteps, wrapped=wrapped, args_cli=args_cli
    )
    _aux_loss_log, _aux_loss_scale, _aux_target_magnitude_log = aux_loss.install_aux_force_loss(
        agent=agent, args_cli=args_cli
    )
    _fast_diag_flush, diag, history, side, traj_mgr = step_diag.install_step_diag(
        num_envs=num_envs, raw_env=raw_env, runner=runner
    )
    (_ckpt_path_at, _voc_flush_on, best_frozen_completion, best_frozen_return, last_eval_iter, original_post) = (
        post_hook.prepare_post_state(agent=agent, exp_dir=exp_dir, runner=runner, args_cli=args_cli)
    )

    def _mem_restart_check(iteration):
        return post_hook._mem_restart_check(iteration, args_cli=args_cli, frozen_eval_snapshot=frozen_eval_snapshot)

    # Frozen eval: pauses training, runs the deterministic eval, resumes. Its state attributes (_stop, _mem_restart,
    # _last_completion, ...) live on this wrapper object.
    def frozen_eval_snapshot(iteration, evals_per_clip=4, traj_recorder=None):
        return frozen_eval.frozen_eval_snapshot(
            iteration,
            evals_per_clip,
            traj_recorder,
            _dist=_dist,
            _dist_rank=_dist_rank,
            _mem_restart_check=_mem_restart_check,
            _voc_flush_on=_voc_flush_on,
            agent=agent,
            args_cli=args_cli,
            best_frozen_completion=best_frozen_completion,
            best_frozen_return=best_frozen_return,
            cfg=cfg,
            create_env=create_env,
            exp_dir=exp_dir,
            frozen_eval_snapshot=frozen_eval_snapshot,
            raw_env=raw_env,
            rollouts=rollouts,
            traj_mgr=traj_mgr,
            wrapped=wrapped,
        )

    return (
        _aux_loss_log,
        _aux_loss_scale,
        _aux_target_magnitude_log,
        _ckpt_path_at,
        _fast_diag_flush,
        _mem_restart_check,
        diag,
        exp_dir,
        frozen_eval_snapshot,
        history,
        last_eval_iter,
        original_post,
        raw_env,
        runner,
        side,
        timesteps,
        traj_mgr,
        wrapped,
    )


def _judge_or_train(
    _OT_VAR,
    _aux_loss_log,
    _aux_loss_scale,
    _aux_target_magnitude_log,
    _ckpt_path_at,
    _dist_rank,
    _fast_diag_flush,
    _mem_restart_check,
    _ot_recorder_class,
    diag,
    exp_dir,
    frozen_eval_snapshot,
    history,
    last_eval_iter,
    original_post,
    raw_env,
    rollouts,
    runner,
    side,
    timesteps,
    traj_mgr,
    wrapped,
):
    """Fresh-process judge (DEXENGINE_FROZEN_REPLAY=1, set by the dexengine.rl.launch judge and render modes): call
    frozen_eval_snapshot itself on the checkpoint loaded by --resume_from, then exit without runner.run(), so the
    weights never change.
    """
    _ret = _judge_or_patch_post(
        _OT_VAR,
        _aux_loss_log,
        _aux_loss_scale,
        _aux_target_magnitude_log,
        _ckpt_path_at,
        _fast_diag_flush,
        _mem_restart_check,
        _ot_recorder_class,
        diag,
        frozen_eval_snapshot,
        history,
        last_eval_iter,
        original_post,
        raw_env,
        rollouts,
        runner,
        side,
        timesteps,
        traj_mgr,
        wrapped,
    )
    if _ret is not _CONTINUE:
        return _ret
    summary.run_and_report(
        _dist_rank=_dist_rank,
        _fast_diag_flush=_fast_diag_flush,
        diag=diag,
        exp_dir=exp_dir,
        frozen_eval_snapshot=frozen_eval_snapshot,
        runner=runner,
        wrapped=wrapped,
        args_cli=args_cli,
    )
    return None


def _judge_or_patch_post(
    _OT_VAR,
    _aux_loss_log,
    _aux_loss_scale,
    _aux_target_magnitude_log,
    _ckpt_path_at,
    _fast_diag_flush,
    _mem_restart_check,
    _ot_recorder_class,
    diag,
    frozen_eval_snapshot,
    history,
    last_eval_iter,
    original_post,
    raw_env,
    rollouts,
    runner,
    side,
    timesteps,
    traj_mgr,
    wrapped,
):
    """Fresh-process judge, or install the post-interaction hook."""
    if os.environ.get("DEXENGINE_FROZEN_REPLAY", "0") == "1":
        frozen_replay.frozen_replay(
            _OT_VAR=_OT_VAR,
            _ot_recorder_class=_ot_recorder_class,
            frozen_eval_snapshot=frozen_eval_snapshot,
            raw_env=raw_env,
            wrapped=wrapped,
            args_cli=args_cli,
        )
        return None

    def patched_post(timestep, timesteps):
        return post_hook.patched_post(
            timestep,
            timesteps,
            _aux_loss_log=_aux_loss_log,
            _aux_loss_scale=_aux_loss_scale,
            _aux_target_magnitude_log=_aux_target_magnitude_log,
            _ckpt_path_at=_ckpt_path_at,
            _fast_diag_flush=_fast_diag_flush,
            _mem_restart_check=_mem_restart_check,
            args_cli=args_cli,
            diag=diag,
            frozen_eval_snapshot=frozen_eval_snapshot,
            history=history,
            last_eval_iter=last_eval_iter,
            original_post=original_post,
            raw_env=raw_env,
            rollouts=rollouts,
            runner=runner,
            side=side,
            traj_mgr=traj_mgr,
        )

    runner.agent.post_interaction = patched_post

    _schedule_clock_offset_curriculum(raw_env, rollouts)
    return _CONTINUE


def _schedule_clock_offset_curriculum(raw_env, rollouts):
    """Schedule-clock offset (--curriculum_step_offset): every step-based schedule reads env.common_step_counter
    (threshold_tightening, ObjFollowSchedule, Level12RampDown, the RsiSchedule band, gravity/friction DR ramps, the
    collision curriculum). A new process counts from 0, so a continued training would rewind every curriculum
    (Level12RampDown would even re-enable rewards it had turned off). Advancing the one clock keeps all consumers
    consistent.
    """
    _cso = _resolve_curriculum_step_offset(args_cli, rollouts)
    if _cso:
        raw_env.common_step_counter = int(_cso)
        print(
            f"[CURRICULUM CONTINUE] common_step_counter starts at {int(_cso)} "
            f"(= {int(_cso) // rollouts} iter). Step-based schedules continue: "
            f"threshold/objfollow/level12/rsi-band/gravity/friction/collision."
        )
