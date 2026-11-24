"""Training env construction, boot smoke, experiment folder and tee log."""

import os
from datetime import datetime

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from dexengine.core import paths as P
from dexengine.core import stdio
from dexengine.rl.env_setup.build import _TeeLogger
from dexengine.rl.env_setup.robot_gains import apply_hand_friction
from dexengine.rl.train.finger_penalty import _require_finger_penalty_marks
from dexengine.rl.train.smoke import _refbase_smoke


def build_env(
    *,
    _dist_rank,
    _hr_ledger,
    extra_slots_ledger_record,
    gym_id,
    num_envs,
    train_cfg_overrides,
    _active_side,
    _residual_act_name,
    args_cli,
    create_env,
):
    """Build the training env (``train_cfg_overrides`` edits its cfg), run the optional boot smoke and
    RSI re-reset, create the experiment folder and start the tee log."""
    env, cfg = create_env(
        gym_id,
        num_envs,
        args_cli.data_dir,
        args_cli.sequence,
        args_cli.seed,
        extra_cfg_fn=train_cfg_overrides,
        recipe_path="train",
        recipe_build_cap=True,
    )  # train_cfg_overrides writes --finger_effort
    return _check_finger_penalty_markers(
        _dist_rank, _hr_ledger, extra_slots_ledger_record, _active_side, _residual_act_name, args_cli, cfg, env
    )


def _check_finger_penalty_markers(
    _dist_rank, _hr_ledger, extra_slots_ledger_record, _active_side, _residual_act_name, args_cli, cfg, env
):
    """Every expected side must have printed its finger-penalty marker, else training / resume / judge stop here."""
    _require_finger_penalty_marks(args_cli, _residual_act_name=_residual_act_name)
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
    raw_env = env.unwrapped

    # Hand friction override (--hand_friction): set robot collision friction (persists across
    # resets; object-only friction DR does not touch it). No-op if not set.
    apply_hand_friction(raw_env, getattr(args_cli, "hand_friction", None))

    # Verify-before-train smoke (env-var-gated). Runs after train_cfg_overrides (so
    # reference_base/use_trajectory_reference are applied), reports, then hard-exits.
    if os.environ.get("DEXENGINE_REFBASE_SMOKE") == "1":
        _refbase_smoke(raw_env, args_cli, _residual_act_name=_residual_act_name, _active_side=_active_side)

    # RSI experiment: the env's initial reset (inside create_env) ran before the
    # curriculum set rsi_fraction, so it used frame 0. Initialize the RSI band/fraction
    # from the schedule cfg now and re-reset, so training starts in-band from step 0
    # (the curriculum keeps updating these each reset). Guarded -> no effect when RSI off
    # or when the cfg has no rsi_schedule (e.g. the force path).
    if (
        bool(getattr(args_cli, "rsi_enable", 0))
        and hasattr(raw_env, "trajectory_manager")
        and hasattr(cfg, "curriculum")
        and hasattr(cfg.curriculum, "rsi_schedule")
    ):
        _p = cfg.curriculum.rsi_schedule.params
        _tm = raw_env.trajectory_manager
        if "fraction_start" in _p:
            _tm.rsi_fraction = float(_p["fraction_start"])
        elif int(_p.get("ramp_episodes", 1)) <= 1:
            _tm.rsi_fraction = float(_p.get("target_fraction", 1.0))
        else:
            _tm.rsi_fraction = 0.0
        _tm.rsi_lo_frac = float(_p.get("lo_frac", 0.0))
        _tm.rsi_hi_frac = float(_p.get("hi_frac", 1.0))
        print(
            f"[run_train] RSI init before first reset: rsi_fraction={_tm.rsi_fraction:.3f} "
            f"band_frac=[{_tm.rsi_lo_frac:.3f},{_tm.rsi_hi_frac:.3f}] -> forcing re-reset"
        )
        wrapped.reset()

    # Experiment directory
    exp_name = args_cli.exp_name or f"dexengine_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = str(P.work_area("work_ckpt") / exp_name)  # the run tree's checkpoint area (RUN_CKPT_ROOT), not the cwd
    os.makedirs(exp_dir, exist_ok=True)
    if os.environ.get("DEXENGINE_FROZEN_REPLAY", "0") != "1":  # a replay did not train under this condition
        _hr_ledger.record(exp_dir, args_cli)
        extra_slots_ledger_record(exp_dir, getattr(env.unwrapped, "_coll_author_info", {}).get("extra"))

    # Tee logger (per-rank file in distributed runs — a shared file would interleave)
    log_path = os.path.join(exp_dir, "training_log.txt" if _dist_rank == 0 else f"training_log_rank{_dist_rank}.txt")
    stdio.process_stream("stdout", _TeeLogger(log_path))  # the run log tee, process level

    # The launcher parses these lines too (see startup.launch_setup).
    print("Training: reference-base + residual policy")
    print(f"  Max iterations: {args_cli.max_iterations}")
    print(f"  Eval interval: {args_cli.eval_interval}")
    print(f"  Experiment: {exp_dir}")
    return cfg, env, exp_dir, exp_name, raw_env, wrapped
