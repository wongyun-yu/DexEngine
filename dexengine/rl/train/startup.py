"""Trainer start-up: launch checks, banner, distributed rank, W&B and the PPO rollout length."""

import os
from datetime import datetime

from dexengine.rl.train.finger_penalty import _finger_penalty_path
from dexengine.rl.train.rollouts import ppo_rollouts


def launch_setup(*, GYM_ID_MAP, args_cli, load_agent_yaml):
    """Launch checks (checkpoint ledgers, judge launch context, trajectory-record consumers), the banner,
    the distributed rank, W&B init and the PPO rollout length / curriculum horizon."""
    task = args_cli.task
    gym_id = GYM_ID_MAP[task]
    num_envs = args_cli.num_envs if args_cli.num_envs is not None else 4096
    _OT_VAR, _dist, _dist_rank, _hr_ledger, _launch_ctx, _ot_recorder_class, extra_slots_ledger_record = (
        _check_hand_recipe_ledger(args_cli, gym_id, num_envs)
    )

    # rollouts = env steps consumed by one PPO update (NE=4096 -> 131,072 samples per update). One owner
    # (rollouts.ppo_rollouts: the yaml rollouts key, or DEXENGINE_ROLLOUTS when set): the PPO update size, the
    # curriculum denominator and the iteration arithmetic all read this value. Total env steps = MI x rollouts, so
    # scaling MI by the same factor keeps the curriculum schedule (rollouts 16 + MI 3200 = rollouts 32 + MI 1600).
    rollouts, _ro_src = ppo_rollouts(load_agent_yaml().get("agent", {}))
    print(
        f"[ROLLOUTS] ppo={rollouts} source={_ro_src} — samples per update {rollouts}xNE, total env steps = "
        f"MI x {rollouts} (shared by PPO, the curriculum and the iteration arithmetic)",
        flush=True,
    )
    # The curriculum denominator: every %-based schedule reads total_timesteps. Without
    # --curriculum_horizon_iterations it follows MI, so extending MI on resume re-times (rewinds) the schedules.
    _ch = getattr(args_cli, "curriculum_horizon_iterations", None)
    if _ch is None:
        total_timesteps = args_cli.max_iterations * rollouts
    else:
        if not 1 <= int(_ch) <= int(args_cli.max_iterations):
            raise SystemExit(
                f"--curriculum_horizon_iterations {_ch} must be between 1 and --max_iterations"
                f"({args_cli.max_iterations})."
            )
        total_timesteps = int(_ch) * rollouts
        print(
            f"[CURRIC_HORIZON] curriculum denominator = {int(_ch)} it x rollouts {rollouts} = {total_timesteps} "
            f"steps (decoupled from the training length MI {args_cli.max_iterations})",
            flush=True,
        )
    return (
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
    )


def _check_hand_recipe_ledger(args_cli, gym_id, num_envs):
    """Hand-recipe checkpoint ledger: a relaunch that re-loads a checkpoint (resume · frozen replay) must match the
    recipe condition the checkpoint was trained with (dexengine.rl.launch.run_dir) — dies before the env is built
    otherwise.
    """
    from dexengine.rl.launch import run_dir as _hr_ledger

    # resume = this launch keeps TRAINING on the loaded weights (not a frozen replay/render) — only then does the
    # ledger also match the finger-penalty span (train-only reward).
    _hr_ledger.enforce(
        args_cli.resume_from or args_cli.resume,
        args_cli,
        ctx="train",
        resume=_finger_penalty_path(args_cli) == "resume",
    )
    _OT_VAR, _launch_ctx, _ot_recorder_class, extra_slots_ledger_record = _check_extra_slot_ledger(args_cli)
    print(f"  Task: {gym_id}")
    print(f"  Num envs: {num_envs}")
    print(f"  Max iterations: {args_cli.max_iterations}")
    print(f"  Eval interval: {args_cli.eval_interval}")
    if args_cli.object_usd:
        print(f"  Object USD: {args_cli.object_usd}")
    print(f"{'=' * 70}\n")

    # Multi-GPU (torch.distributed) rank-aware gating.
    # skrl's config.torch auto-initialized the NCCL process group at import
    # (torchrun env vars). The CORE distributed machinery is upstream/skrl:
    #   - initial-weight broadcast: PPO.__init__ (policy/value.broadcast_parameters)
    #   - gradient all-reduce: PPO._update / PerSidePPO._update (reduce_parameters)
    #   - skrl's base init() zeroes write/checkpoint intervals on non-zero ranks
    # Here we gate only the CUSTOM side channels of this runner: manual W&B
    # init, the tee-log path, best_agent.pt / best_return_agent.pt writes and
    # the heldout manifest (all rank-0 only).
    from skrl import config as _skrl_config

    _dist = bool(_skrl_config.torch.is_distributed)
    _dist_rank = int(_skrl_config.torch.rank) if _dist else 0
    _dist_world = int(_skrl_config.torch.world_size) if _dist else 1
    if _dist:
        print(
            f"[run_train] DISTRIBUTED rank {_dist_rank}/{_dist_world} "
            f"device={_skrl_config.torch.device} num_envs={num_envs}/GPU "
            f"(global {num_envs * _dist_world}) seed={args_cli.seed}"
        )
        if _dist_rank != 0 and args_cli.wandb:
            # Single switch: gates wandb.init below AND every downstream
            # wandb.log (snapshot_metrics, frozen eval, termination breakdown).
            args_cli.wandb = False
            print(f"[run_train] rank {_dist_rank}: W&B disabled (rank-0 only)")

    # W&B init — MUST happen BEFORE the skrl learner is created so
    # that skrl's base init() (in skrl/agents/torch/base.py) does NOT call wandb.init
    # itself with sync_tensorboard=True, which would mirror TensorBoard scalars (at
    # raw timestep counter) into wandb and corrupt the step axis.
    if args_cli.wandb:
        import wandb

        wandb.init(
            project=args_cli.wandb_project,
            group=args_cli.wandb_group,
            tags=args_cli.wandb_tags.split(",") if args_cli.wandb_tags else None,
            name=args_cli.wandb_run_name
            or args_cli.exp_name
            or f"dexengine_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "task": gym_id,
                "num_envs": num_envs,
                "max_iterations": args_cli.max_iterations,
                "eval_interval": args_cli.eval_interval,
                "sequence": args_cli.sequence,
                "data_dir": args_cli.data_dir,
            },
        )
    return _OT_VAR, _dist, _dist_rank, _hr_ledger, _launch_ctx, _ot_recorder_class, extra_slots_ledger_record


def _check_extra_slot_ledger(args_cli):
    """Extra-slot ledger: a launch that loads a checkpoint must use the extra-slot values it was trained with."""
    from dexengine.rl.env_setup.extra_slots_ledger import (
        extra_slots_ledger_check,
        extra_slots_ledger_record,
        ledger_checkpoint,
    )

    extra_slots_ledger_check(ledger_checkpoint(args_cli), "train")
    # Judge launch check (always): retired names, PARK on by default for the fresh-process judge, the in-training
    # eval marker, the PARK / exit-reset combinations (rules in dexengine.rl.judge.launch). It runs before the env is
    # built because it also writes the PARK and warmup defaults into the environment that env start-up reads.
    from dexengine.rl.judge.launch import check_judge_terms, check_launch, launch_ctx

    _launch_ctx = check_launch(
        launch_ctx(
            replay=os.environ.get("DEXENGINE_FROZEN_REPLAY", "0") == "1",
            eval_on=not args_cli.no_frozen_eval,
            oop=False,
            render=os.environ.get("DEXENGINE_FROZEN_RENDER", "0") == "1",
        )
    )
    # Judge termination rule (contact termination, dexengine.rl.judge.terms): a fresh-process judge proves the
    # argument value ([JUDGE_TERM] marker).
    check_judge_terms(_launch_ctx, getattr(args_cli, "enable_contact_termination", 0))
    # The trajectory record (off by default) is accepted only on the path that consumes it.
    _tr_bad = []
    if os.environ.get("DEXENGINE_TRAJ_RECORD", ""):
        if os.environ.get("DEXENGINE_FROZEN_REPLAY", "0") != "1":
            _tr_bad.append(
                "the only consumer of DEXENGINE_TRAJ_RECORD is the one-shot DEXENGINE_FROZEN_REPLAY=1 judge branch"
            )
        if os.environ.get("DEXENGINE_FROZEN_RENDER", "0") == "1":
            _tr_bad.append(
                "DEXENGINE_TRAJ_RECORD and the render branch (DEXENGINE_FROZEN_RENDER=1) do not go together "
                "— the render runs no judge batches"
            )
    # Object-pose trace (an experiment input, off by default): its only consumer is the trajectory recorder
    # above. The rules live in the plugin registered under the hook ``objtrace`` (dexengine.rl.tasks.hooks), which
    # the launch names in DEXENGINE_PLUGINS; it is looked up only when the input is set.
    _OT_VAR = "DEXENGINE_E123_OBJTRACE"
    _ot_recorder_class = None
    if os.environ.get(_OT_VAR, ""):
        from dexengine.rl.tasks import hooks as _hooks

        _hooks.load_plugins()
        _ot = _hooks.get(_hooks.OBJTRACE)
        if _ot is None:
            raise SystemExit(
                f"[TRAJ_REC] {_OT_VAR} is set but no plugin registered the hook {_hooks.OBJTRACE!r} — "
                f"name the object-pose trace plugin in {_hooks.PLUGINS_ENV}"
            )
        if _ot.ENV_VAR != _OT_VAR:
            raise SystemExit(f"[TRAJ_REC] objtrace input name {_ot.ENV_VAR} != {_OT_VAR}")
        _tr_bad += _ot.launch_problems(
            os.environ[_OT_VAR],
            os.environ.get("DEXENGINE_TRAJ_RECORD", ""),
            replay=os.environ.get("DEXENGINE_FROZEN_REPLAY", "0") == "1",
            worker=False,
            render=os.environ.get("DEXENGINE_FROZEN_RENDER", "0") == "1",
        )
        _ot_recorder_class = _ot.recorder_class
    if _tr_bad:
        raise SystemExit("[TRAJ_REC] start-up refused: " + " · ".join(_tr_bad))

    # dexengine.rl.launch.training parses the "  Max iterations:" and "  Eval interval:" lines of this banner.
    print(f"\n{'=' * 70}")
    print("DexEngine Training (reference-base + residual)")
    return _OT_VAR, _launch_ctx, _ot_recorder_class, extra_slots_ledger_record
