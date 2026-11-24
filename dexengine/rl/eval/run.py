"""``--task eval``: kinematic replay render, contact diagnostics, and the one-hand evaluation of a checkpoint.

Imported lazily by ``dexengine.rl.cli`` after the AppLauncher, so module-level Isaac/skrl imports are safe.
``run_eval`` receives the entry module's RunContext (args, create_env, gym-id map, active object and side, small
helpers) and binds it to the module globals below.

Order of ``run_eval``: the bimanual guard (bimanual runs are judged and rendered by the frozen-replay path of training)
· env built with the eval overrides (``cfg_overrides``) · pose render (``--pose_render_out``, no policy) · env-flag
diagnostics · PPO learner and checkpoint load · per-clip video render (``--render_per_clip_videos``; with
``--kinematic_replay`` the policy is never called) · otherwise the balanced per-clip evaluation.
"""

import os
import sys

import torch
from skrl.utils.runner.torch import Runner

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from dexengine.core.paths import run_out as _run_out  # a relative output folder is relative to the run tree
from dexengine.rl.agents.models import _build_dual_policy_fn, _build_policy_models, assert_checkpoint_obs_dim
from dexengine.rl.env_setup.refbase_audit import refbase_pd_audit
from dexengine.rl.env_setup.robot_gains import apply_arm_gain_override, apply_hand_friction
from dexengine.rl.eval.cfg_overrides import make_eval_cfg_overrides
from dexengine.rl.eval.contact_diag import contact_audit
from dexengine.rl.judge.launch import check_launch
from dexengine.rl.judge.rollouts import run_eval_balanced
from dexengine.rl.judge.snapshot_metrics import _effective_fingertip_thresholds, _fmt_fingertip_thresholds
from dexengine.rl.render.clip_video import _run_per_clip_video_render
from dexengine.rl.run_context import bind_run_context

# Entry-module context (RunContext), bound by bind_run_context at the start of run_eval.
args_cli = None
create_env = None
GYM_ID_MAP = None
_ACTIVE_OBJ_KEY = None
_active_side = None
load_agent_yaml = None
_residual_act_name = None


def run_eval(ctx):
    """Evaluate, replay or render a trained checkpoint under the eval overrides (see the module docstring)."""
    bind_run_context(globals(), ctx)
    task = args_cli.task
    gym_id = GYM_ID_MAP[task]
    num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    # Launch context "evalpy": not a judge (reference evaluation, replay, render); PARK off by default. The
    # combination rules belong to dexengine.rl.judge.launch.
    check_launch("evalpy")
    # --pose_render_out uses no policy, so the checkpoint may be None.
    checkpoint_path = os.path.abspath(args_cli.checkpoint) if args_cli.checkpoint else None
    # Hand-recipe checkpoint ledger: the checkpoint's trained recipe condition must match this eval's — dies before
    # the env is built otherwise (the recipe switch defaults on).
    from dexengine.rl.launch import run_dir as _hr_ledger

    _hr_ledger.enforce(checkpoint_path, args_cli, ctx="eval", resume=False)  # eval never trains on it
    from dexengine.rl.env_setup.extra_slots_ledger import extra_slots_ledger_check, ledger_checkpoint

    extra_slots_ledger_check(ledger_checkpoint(args_cli), "eval")  # extra-slot ledger: the value at training time

    # Bimanual runs do not use this path: judging and rendering both go through the frozen-replay path of training
    # (frozen_eval_snapshot via --resume_from with DEXENGINE_FROZEN_REPLAY=1 [+DEXENGINE_FROZEN_RENDER=1]). This path
    # differs from the frozen replay and does not reproduce on bimanual checkpoints (same checkpoint, clips and
    # envs give different values; the cause is not known). One-hand runs use it only as a reference evaluation.
    # The kinematic replay render of a bimanual scene opens it explicitly (DEXENGINE_ALLOW_BI_EVAL_PATH=1).
    if getattr(args_cli, "bimanual", 0) and os.environ.get("DEXENGINE_ALLOW_BI_EVAL_PATH", "0") != "1":
        raise SystemExit(
            "[eval] bimanual runs do not use this eval/render path.\n"
            "  Judging and rendering both go through the frozen path:\n"
            "    judge : FROZEN_REPLAY_CKPT=<ckpt> SIDE=both ... bash scripts/stage3_rl.sh <name>  (chain: "
            "scripts/rl/judge_sidecar.sh)\n"
            "    render: RENDER_ONLY=<ckpt.pt> SIDE=both ... bash scripts/stage3_rl.sh <name>\n"
            "  This path gives other values than the frozen one and does not reproduce under "
            "the same condition.\n"
            "  For a diagnostic, open it explicitly with DEXENGINE_ALLOW_BI_EVAL_PATH=1 — never use its values for "
            "judging."
        )

    _print_eval_header(checkpoint_path, gym_id, num_envs, task)

    # DEXENGINE_EVAL_TRAIN_PARITY, read once: the finger-gain mirror in the eval overrides and the hand-recipe check
    # of create_env must agree on whether --finger_effort reaches the finger actuator on this eval.
    _eval_train_parity = os.environ.get("DEXENGINE_EVAL_TRAIN_PARITY") == "1"

    _eval_cfg_overrides = make_eval_cfg_overrides(
        args_cli, _ACTIVE_OBJ_KEY, _active_side, _residual_act_name, _eval_train_parity
    )

    return _run_in_env(_eval_cfg_overrides, _eval_train_parity, checkpoint_path, gym_id, num_envs)


def _run_in_env(_eval_cfg_overrides, _eval_train_parity, checkpoint_path, gym_id, num_envs):
    """Create the env, apply the eval overrides, run the balanced eval (or render) and close it."""
    # Create environment
    env, cfg = create_env(
        gym_id,
        num_envs,
        args_cli.data_dir,
        args_cli.sequence,
        args_cli.seed,
        extra_cfg_fn=_eval_cfg_overrides,
        recipe_path="eval",
        recipe_build_cap=_eval_train_parity,
    )
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
    raw_env = env.unwrapped

    # Pose render (--pose_render_out): renders the reference pose after reset and exits — no policy and no
    # checkpoint (reset alone places the reference pose). It is the visual check of a new (arm, hand) pair before
    # training. Placed after create_env so it sees exactly the eval overrides (arm gains, IK cache, thresholds).
    if getattr(args_cli, "pose_render_out", ""):
        from dexengine.rl.render.pose_check import run_pose_render

        run_pose_render(raw_env, args_cli, args_cli.pose_render_out, f"{args_cli.arm}_{args_cli.hand}")
        env.close()
        return

    # Arm-gain override (env-var gated, render/audit only) — applied here so it covers both the PD render loop and
    # the reference-base PD audit; without the env var the gains are unchanged.
    apply_arm_gain_override(raw_env)
    clip_names, num_clips, policy_fn, traj_mgr = _apply_hand_friction(checkpoint_path, raw_env, wrapped)

    # --render_per_clip_videos: per-clip video render (the kinematic replay render goes here), then return.
    if args_cli.render_per_clip_videos:
        _run_per_clip_video_render(
            wrapped,
            policy_fn,
            traj_mgr,
            num_clips,
            clip_names,
            raw_env,
            checkpoint_path=checkpoint_path,
            out_dir=str(_run_out(args_cli.videos_out_dir)),
            max_attempts=args_cli.videos_max_attempts,
            max_episode_steps=args_cli.videos_max_episode_steps,
            fps=args_cli.videos_fps,
            warmup_attempts=args_cli.videos_warmup_attempts,
            select_best_success=bool(args_cli.videos_select_best_success),
            rollouts_per_clip=args_cli.videos_rollouts_per_clip,
            rank_metric=args_cli.videos_rank_metric,
            args_cli=args_cli,
            create_env=create_env,
        )
        return  # skip the balanced evaluation

    aggregate, _ = run_eval_balanced(
        wrapped_env=wrapped,
        policy_fn=policy_fn,
        traj_mgr=traj_mgr,
        num_clips=num_clips,
        evals_per_clip=args_cli.eval_evals_per_clip,
        clip_names=clip_names,
        warmup_batches=args_cli.eval_warmup_batches,
    )

    wrapped.close()


def _apply_hand_friction(checkpoint_path, raw_env, wrapped):
    """Hand friction override (--hand_friction): set the robot collision friction."""
    # Hand friction override (--hand_friction): set robot collision friction (default = USD-baked).
    apply_hand_friction(raw_env, getattr(args_cli, "hand_friction", None))

    # Reference-base PD audit (DEXENGINE_REFBASE_PD_AUDIT=1, needs --reference_base 1): steps the env with zero
    # residual under the reference-base PD targets, measures wrist / finger-qpos / fingertip tracking error and
    # object force, then exits.
    if os.environ.get("DEXENGINE_REFBASE_PD_AUDIT") == "1":
        try:
            refbase_pd_audit(raw_env, args_cli)
        except Exception:  # noqa: BLE001 — a diagnostic: print any failure; the finally still closes and exits
            import traceback as _tb

            _tb.print_exc()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)  # hard exit: simulation_app.close() deadlocks on a started env

    # Load the PPO config and create runner
    agent_cfg = load_agent_yaml()
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    _initial_log_std, _max_log_std, _min_log_std, agent = _build_eval_agent(
        agent_cfg, checkpoint_path, raw_env, wrapped
    )

    runner = _eval_runner_and_pins(agent, agent_cfg, raw_env, wrapped)

    # Policy function for the rollouts: the deterministic mean, as in the in-training frozen evaluation, so this
    # eval reports the same number (sampling would add the policy's sigma noise to the score).
    agent = runner.agent
    preprocessor = agent._state_preprocessor
    policy_model = agent.policy

    def policy_fn(obs_tensor):
        with torch.no_grad():
            normed = preprocessor(obs_tensor)
            mu, _, _ = policy_model.compute({"states": normed}, role="policy")
            return mu

    print("  [Eval] policy_fn: deterministic mu (matches frozen_eval_snapshot semantics)")

    # --dual_eval 1 (render only, bimanual scene): replace policy_fn with independent right and left one-hand
    # policies (obs 414 -> 207/207, action 12|12 -> 24). Everything downstream consumes policy_fn only.
    if getattr(args_cli, "dual_eval", 0):
        policy_fn = _build_dual_policy_fn(
            raw_env,
            raw_env.device,
            (_min_log_std, _max_log_std, _initial_log_std),
            size=(getattr(args_cli, "policy_size", "tiny") or "tiny"),
            r_ckpt=os.path.abspath(args_cli.dual_ckpt_right),
            l_ckpt=os.path.abspath(args_cli.dual_ckpt_left),
        )

    # Per-finger contact audit (DEXENGINE_CONTACT_AUDIT=1): prints and exits. With --render_per_clip_videos the
    # render-path audit runs instead (it logs the same rollout it saves to video).
    if os.environ.get("DEXENGINE_CONTACT_AUDIT") == "1" and not args_cli.render_per_clip_videos:
        contact_audit(raw_env, wrapped, policy_fn, args_cli, checkpoint_path, _ACTIVE_OBJ_KEY)

    # Episode-based eval via run_eval_balanced (the same loop as the frozen evaluation). --eval_steps does not
    # control it; --eval_evals_per_clip and --eval_warmup_batches do.
    traj_mgr = raw_env.trajectory_manager
    num_clips = traj_mgr.num_clips
    clip_names = traj_mgr._loader.sequence_names
    print(
        f"  [Eval] Running {args_cli.eval_evals_per_clip} eps/clip x {num_clips} clips "
        f"with {args_cli.eval_warmup_batches} warmup batch(es) discarded"
    )
    return clip_names, num_clips, policy_fn, traj_mgr


def _print_eval_header(checkpoint_path, gym_id, num_envs, task):
    """The eval banner and the finger penalty span in force."""
    task_label = "arm residual (deterministic)" if task == "eval" else task
    print(f"\n{'=' * 70}")
    print(f"DexEngine Eval ({task_label})")
    print(f"  Task: {gym_id}")
    print(f"  Checkpoint: {checkpoint_path or '(none — no-policy mode)'}")
    print(f"  Num envs: {num_envs}")
    print(f"  Eval steps: {args_cli.eval_steps}")
    print(f"{'=' * 70}\n")

    # --finger_penalty_span: the two finger penalties it spans are training-only reward terms; eval scores success,
    # not those rewards, so nothing here consumes it. It still reaches this path on purpose (the one-hand eval stage
    # forwards the training flags), and the eval path always prints its finger-penalty line (span + source) because
    # the log audit requires it on every path.
    from dexengine.core.hands import resolve_finger_penalty_span as _resolve_fspan

    _fspan_e, _fspan_src_e = _resolve_fspan(getattr(args_cli, "finger_penalty_span", None))
    from dexengine.rl.train.finger_penalty import FINGER_PENALTY_PATH as _FINGER_PENALTY_PATH

    print(
        f"{_FINGER_PENALTY_PATH}eval span={_fspan_e}({_fspan_src_e}) — the eval path has no consuming term (two "
        "training-only reward terms · unrelated to the success judge)",
        flush=True,
    )


def _build_eval_agent(agent_cfg, checkpoint_path, raw_env, wrapped):
    """Policy / value models with the YAML log_std bounds, the PPO learner with running preprocessors, the checkpoint
    load and its diagnostics."""
    # The policy/value models are built by the same factory as training, then the checkpoint is loaded.
    #
    # state_preprocessor and value_preprocessor MUST be set
    # BEFORE instantiating PPO. Otherwise, PPO defaults them to None →
    # _empty_preprocessor (identity) → PPO.load() has no
    # checkpoint_modules["state_preprocessor"] entry to populate →
    # the trained RunningStandardScaler mean/variance are silently dropped
    # and the policy sees RAW observations instead of the normalized ones
    # it was trained on → residual outputs are garbage → any non-zero
    # residual_scale destroys the grasp and eval success collapses.
    from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
    from skrl.resources.preprocessors.torch import RunningStandardScaler

    # Read log_std bounds from YAML so eval policy matches training construction.
    _policy_yaml = agent_cfg.get("models", {}).get("policy", {})
    _min_log_std = float(_policy_yaml.get("min_log_std", -20.0))
    _max_log_std = float(_policy_yaml.get("max_log_std", -0.5))
    _initial_log_std = float(_policy_yaml.get("initial_log_std", -1.0))
    print(f"  [EVAL MODEL] log_std bounds: min={_min_log_std} max={_max_log_std} init={_initial_log_std}")
    # Same aux flags as training, so checkpoints with the optional aux submodules (aux_encoder for decoupled,
    # aux_force_head_aware for action-aware) load.
    _eval_aux_decouple = bool(getattr(args_cli, "aux_decouple_encoder", 0))
    _eval_aux_action_aware = bool(getattr(args_cli, "aux_action_aware", 0))
    if _eval_aux_decouple:
        print("  [EVAL MODEL] aux_decouple_encoder=1 — building parallel aux_encoder for checkpoint compat")
    if _eval_aux_action_aware:
        print("  [EVAL MODEL] aux_action_aware=1 — allocating aux_force_head_aware for checkpoint compat")
    return _build_and_load_agent(
        checkpoint_path,
        raw_env,
        wrapped,
        PPO,
        PPO_DEFAULT_CONFIG,
        RunningStandardScaler,
        _initial_log_std,
        _max_log_std,
        _min_log_std,
    )


def _build_and_load_agent(
    checkpoint_path,
    raw_env,
    wrapped,
    PPO,
    PPO_DEFAULT_CONFIG,
    RunningStandardScaler,
    _initial_log_std,
    _max_log_std,
    _min_log_std,
):
    """Policy/value models and the PPO learner, the checkpoint loaded into them."""
    # Per-side advantage checkpoints carry a 2-head value and a 2-dim value preprocessor; build them the same way
    # so PPO.load() matches the shapes.
    _eval_per_side = int(getattr(args_cli, "per_side_advantage", 0)) == 1 and bool(getattr(args_cli, "bimanual", 0))
    if _eval_per_side:
        print("  [EVAL MODEL] per_side_advantage=1 — two-head critic + 2-dim value scaler for ckpt compat")
    policy, value, _eval_arch = _build_policy_models(
        wrapped,
        raw_env,
        raw_env.device,
        args_cli,
        (_min_log_std, _max_log_std, _initial_log_std),
        per_side_value=_eval_per_side,
    )
    print(f"  [EVAL MODEL] policy_arch = {_eval_arch}")
    policy.to(raw_env.device)
    value.to(raw_env.device)

    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg["experiment"] = dict(ppo_cfg.get("experiment", {}))
    ppo_cfg["experiment"]["write_interval"] = 0
    ppo_cfg["experiment"]["checkpoint_interval"] = 0
    agent = _make_and_load_ppo(
        checkpoint_path, raw_env, wrapped, PPO, RunningStandardScaler, _eval_per_side, policy, ppo_cfg, value
    )
    return _initial_log_std, _max_log_std, _min_log_std, agent


def _make_and_load_ppo(
    checkpoint_path, raw_env, wrapped, PPO, RunningStandardScaler, _eval_per_side, policy, ppo_cfg, value
):
    """PPO learner with the training preprocessors, the checkpoint loaded into it (per-side checkpoints included)."""
    # Same preprocessor setup as training, so PPO.load() restores the RunningStandardScaler running_mean /
    # running_variance from the checkpoint.
    ppo_cfg["state_preprocessor"] = RunningStandardScaler
    ppo_cfg["state_preprocessor_kwargs"] = {"size": wrapped.observation_space, "device": raw_env.device}
    ppo_cfg["value_preprocessor"] = RunningStandardScaler
    ppo_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": raw_env.device}
    if _eval_per_side:
        ppo_cfg["value_preprocessor_kwargs"] = {"size": 2, "device": raw_env.device}

    agent = PPO(
        models={"policy": policy, "value": value},
        observation_space=wrapped.observation_space,
        action_space=wrapped.action_space,
        device=raw_env.device,
        cfg=ppo_cfg,
    )
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    if getattr(args_cli, "dual_eval", 0):
        # Two-policy mode: the single bimanual PPO learner is a throwaway (its weights are never used); the right and
        # left one-hand policies are loaded by _build_dual_policy_fn below. There is no merged 24-action checkpoint.
        print("  [DUAL] --dual_eval 1 -> skipping single-agent load (two uni policies loaded separately)")
    elif getattr(args_cli, "kinematic_replay", False):
        # The kinematic replay teleports the reference frame by frame and never calls the policy, so no
        # checkpoint is loaded (a checkpoint of another observation width would otherwise block the render).
        print("  [KR] --kinematic_replay -> policy not used, checkpoint load skipped")
    else:
        assert_checkpoint_obs_dim(checkpoint_path, int(wrapped.observation_space.shape[0]), "eval")
        agent.load(checkpoint_path)
    agent.set_running_mode("eval")
    # Diagnostic: print preprocessor stats so we can verify the ckpt loaded them
    try:
        sp = agent._state_preprocessor
        if hasattr(sp, "running_mean"):
            print(
                f"[INFO] state_preprocessor loaded: "
                f"mean_norm={sp.running_mean.float().norm().item():.4f} "
                f"var_norm={sp.running_variance.float().norm().item():.4f}"
            )
    except (AttributeError, IndexError, RuntimeError, TypeError) as _e:
        print(f"[WARN] could not inspect state_preprocessor: {_e}")
    # Diagnostic: print trained log_std so we can see how noisy eval actions are
    try:
        ls = policy.log_std_parameter.detach().cpu().tolist()
        print(f"[INFO] trained log_std: wrist={ls[:6]} finger={ls[6:]}")
    except (AttributeError, IndexError, RuntimeError, TypeError) as _e:
        print(f"[WARN] could not inspect log_std: {_e}")
    return agent


def _eval_runner_and_pins(agent, agent_cfg, raw_env, wrapped):
    """Minimal skrl runner around the PPO learner; eval progress pins on the trajectory manager; curriculum off."""
    # Create a minimal runner for compatibility with the eval loop
    runner = Runner(wrapped, agent_cfg)
    runner._agent = agent  # our PPO learner replaces the runner's (internal attr)
    # No threshold curriculum at eval. Evaluate at the forced threshold progress
    # (--eval_threshold_progress; default 1.0 = final/tight under the
    # linear-schedule convention).
    if hasattr(raw_env, "trajectory_manager"):
        _eval_prog = float(args_cli.eval_threshold_progress)
        raw_env.trajectory_manager.schedule_progress = _eval_prog
        # Default: eval from frame 0. With --rsi_enable, eval/render from the RSI band
        # (curriculum is disabled at eval, so set the band directly from the flags) — used
        # to VISUALIZE what an RSI policy learned (grasp-hold from a mid-trajectory reset).
        if bool(getattr(args_cli, "rsi_enable", 0)):
            raw_env.trajectory_manager.rsi_fraction = float(getattr(args_cli, "rsi_target_fraction", 1.0))
            raw_env.trajectory_manager.rsi_lo_frac = float(getattr(args_cli, "rsi_lo_frac", 0.0))
            raw_env.trajectory_manager.rsi_hi_frac = float(getattr(args_cli, "rsi_hi_frac", 1.0))
            print(
                f"  [Eval] schedule_progress={_eval_prog:.4f}, RSI ENABLED: "
                f"rsi_fraction={raw_env.trajectory_manager.rsi_fraction:.3f} "
                f"band_frac=[{raw_env.trajectory_manager.rsi_lo_frac:.3f},"
                f"{raw_env.trajectory_manager.rsi_hi_frac:.3f}]"
            )
        else:
            raw_env.trajectory_manager.rsi_fraction = 0.0  # eval from frame 0
            print(
                f"  [Eval] schedule_progress={_eval_prog:.4f}"
                f"{' (final/tight)' if _eval_prog == 1.0 else ' (--eval_threshold_progress)'}, "
                f"rsi_fraction=0.0"
            )
        print(f"  [Eval] {_fmt_fingertip_thresholds(_effective_fingertip_thresholds(raw_env, _eval_prog))}")

    # Disable curriculum (prevents ramping during eval)
    if hasattr(raw_env, "curriculum_manager"):
        raw_env.curriculum_manager._term_names = []
        raw_env.curriculum_manager._term_cfgs = {}
        raw_env.curriculum_manager._class_term_cfgs = {}
        print("  [Eval] Curriculum manager disabled.")
    return runner
