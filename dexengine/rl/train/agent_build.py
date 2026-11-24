"""Residual policy/value models, the PPO config built from the skrl yaml and CLI overrides, and the PPO learner."""

import os

import skrl
import torch
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler

from dexengine.rl.agents.models import _build_policy_models


def build_agent(*, _dist, exp_dir, exp_name, raw_env, rollouts, wrapped, args_cli, load_agent_yaml):
    """Build the residual policy/value models, the PPO config and the PPO learner.

    The base action is always the kinematic reference (compute_reference_base_action); only the residual is learned.
    """
    # Load the skrl PPO config for hyperparameters
    agent_cfg = load_agent_yaml()
    agent_params = agent_cfg.get("agent", {})
    _per_side_adv, device, memory, models, ppo_cfg, timesteps = _ppo_rollouts(
        exp_dir, exp_name, raw_env, rollouts, wrapped, args_cli, agent_cfg, agent_params
    )

    # Create the PPO learner (PER-SIDE ADVANTAGE mode swaps in the factored-credit
    # subclass; the flag-off branch below is the unchanged existing path).
    if _per_side_adv:
        from dexengine.rl.agents.per_side_ppo import PerSidePPO

        _act_dim = int(wrapped.action_space.shape[0])
        ppo_cfg["per_side_dims"] = (_act_dim // 2, _act_dim - _act_dim // 2)
        agent = PerSidePPO(
            models=models,
            memory=memory,
            cfg=ppo_cfg,
            observation_space=wrapped.observation_space,
            action_space=wrapped.action_space,
            device=device,
        )
        print(
            f"  [AGENT] PerSidePPO (per-side advantage): mu split "
            f"[0:{_act_dim // 2}]=right, [{_act_dim // 2}:{_act_dim}]=left; "
            f"per-side GAE + two-head critic + factored PPO-clip loss"
        )
    else:
        # DISTRIBUTED + stock PPO + KL early-stop: upstream skrl 1.4.3 breaks the
        # minibatch loop on the LOCAL (un-synced) KL — ranks can take different
        # branch counts, desyncing the reduce_parameters() all-reduce sequence
        # (silent cross-minibatch gradient mixing, eventual NCCL hang).
        # Fix: KLSafePPO = skrl's _update (restructured) with the early-stop
        # KL all-reduced across ranks (same fix PerSidePPO carries). Selected ONLY
        # when distributed — the single-GPU path keeps upstream PPO.
        _stock_ppo_cls = PPO
        if _dist and float(ppo_cfg.get("kl_threshold") or 0.0) > 0.0:
            from dexengine.rl.train.klsafe_ppo import KLSafePPO as _stock_ppo_cls

            print(
                "  [AGENT] KLSafePPO (distributed stock path): KL early-stop "
                "rank-synced via all-reduce (upstream skrl compares local KL)"
            )
        agent = _stock_ppo_cls(
            models=models,
            memory=memory,
            cfg=ppo_cfg,
            observation_space=wrapped.observation_space,
            action_space=wrapped.action_space,
            device=device,
        )

    # Exp-1 stability: apply weight decay to the PPO optimizer (actor + critic).
    # Directly opposes the actor weight-norm inflation that drives residual-actor collapse
    # (a soft L2 pull on every weight). Off by default (--actor_weight_decay 0).
    _wd = float(getattr(args_cli, "actor_weight_decay", 0.0))
    if _wd > 0.0 and getattr(agent, "optimizer", None) is not None:
        for _g in agent.optimizer.param_groups:
            _g["weight_decay"] = _wd
        print(
            f"[run_train] EXP1 actor weight_decay={_wd} set on PPO optimizer "
            f"({len(agent.optimizer.param_groups)} param group(s))"
        )
    return agent, timesteps


def _ppo_rollouts(exp_dir, exp_name, raw_env, rollouts, wrapped, args_cli, agent_cfg, agent_params):
    """rollouts comes from rollouts.ppo_rollouts (the one owner) — the yaml is not re-read here."""
    timesteps = args_cli.max_iterations * rollouts
    device = raw_env.device

    # Build the custom policy and value models (the recipe picks the architecture)
    # These override the YAML model config — the YAML is only used for PPO hyperparameters.
    # Read log_std bounds from the YAML policy section so train and eval stay in sync.
    _policy_yaml = agent_cfg.get("models", {}).get("policy", {})
    _min_log_std = float(_policy_yaml.get("min_log_std", -20.0))
    _max_log_std = float(_policy_yaml.get("max_log_std", -0.5))
    _initial_log_std = float(_policy_yaml.get("initial_log_std", -1.0))
    if args_cli.min_log_std is not None:
        _min_log_std = args_cli.min_log_std
        print(f"[train] min_log_std override = {_min_log_std}")
    if args_cli.max_log_std is not None:
        _max_log_std = args_cli.max_log_std
        print(f"[train] max_log_std override = {_max_log_std}")
    if args_cli.initial_log_std is not None:
        _initial_log_std = args_cli.initial_log_std
        print(f"[train] initial_log_std override = {_initial_log_std}")
    print(f"  [MODEL] log_std bounds: min={_min_log_std} max={_max_log_std} init={_initial_log_std}")
    _aux_decouple = bool(getattr(args_cli, "aux_decouple_encoder", 0))
    if _aux_decouple:
        print("  [MODEL] aux_decouple_encoder=1 — building parallel aux_encoder, residual_mlp input = 256+256=512")
    _aux_action_aware = bool(getattr(args_cli, "aux_action_aware", 0))
    if _aux_action_aware:
        print("  [MODEL] aux_action_aware=1 — allocating aux_force_head_aware (Linear 256+12=268 → 256 → 15)")
    # PER-SIDE ADVANTAGE: two-head critic + PerSidePPO learner (guarded; default 0
    # keeps the whole existing path — factory, preprocessors, learner — unchanged).
    _per_side_adv = int(getattr(args_cli, "per_side_advantage", 0)) == 1 and bool(getattr(args_cli, "bimanual", 0))
    policy, value, _arch = _build_policy_models(
        wrapped,
        raw_env,
        device,
        args_cli,
        (_min_log_std, _max_log_std, _initial_log_std),
        per_side_value=_per_side_adv,
    )
    print(f"  [MODEL] policy_arch = {_arch}")
    policy.to(device)
    value.to(device)
    models = {"policy": policy, "value": value}

    policy_params = sum(p.numel() for p in policy.parameters())
    value_params = sum(p.numel() for p in value.parameters())
    print(f"  [MODEL] {type(policy).__name__}: {policy_params:,} params")
    print(f"  [MODEL] value model:  {value_params:,} params")

    # Verify near-zero initial residual output
    with torch.no_grad():
        test_in = torch.zeros(1, wrapped.observation_space.shape[0], device=device)
        test_mu, test_log_std, _ = policy.compute({"states": test_in})
        print(
            f"  [INIT] Policy output on zeros: mean={test_mu.mean():.6f}, "
            f"std={test_mu.std():.6f}, max_abs={test_mu.abs().max():.6f}"
        )
        print(f"  [INIT] log_std={test_log_std[0, 0]:.4f} (sigma={test_log_std[0, 0].exp():.4f})")

    ppo_cfg = _build_ppo_cfg(
        exp_dir, exp_name, rollouts, wrapped, args_cli, _per_side_adv, agent_cfg, agent_params, device
    )

    # NOTE: We explicitly do NOT set ppo_cfg["experiment"]["wandb"] = True because
    # wandb has already been initialized manually at the top of run_train.
    # Letting SKRL call wandb.init with sync_tensorboard=True corrupts the step axis
    # (TensorBoard scalars are mirrored at raw timestep counter, fighting our
    # explicit wandb.log(step=iteration*rollouts) calls).

    # Memory
    memory = RandomMemory(memory_size=rollouts, num_envs=wrapped.num_envs, device=device)
    return _per_side_adv, device, memory, models, ppo_cfg, timesteps


def _build_ppo_cfg(exp_dir, exp_name, rollouts, wrapped, args_cli, _per_side_adv, agent_cfg, agent_params, device):
    """PPO config: the YAML PPO block, the CLI overrides, the LR scheduler, the running preprocessors and the
    experiment / checkpoint intervals."""
    ppo_cfg = _ppo_config_from_yaml(
        exp_dir, exp_name, rollouts, wrapped, args_cli, _per_side_adv, agent_cfg, agent_params, device
    )
    # Align SKRL's internal wandb writes with our snapshot_metrics interval so
    # all logged metrics share the same x-axis (timesteps).
    ppo_cfg["experiment"]["write_interval"] = args_cli.eval_interval * rollouts
    # Checkpoint interval (iterations): DEXENGINE_CKPT_INTERVAL when set, else the eval interval — so evaluating less
    # often does not also checkpoint less often.
    _ck_iv = os.environ.get("DEXENGINE_CKPT_INTERVAL", "").strip()
    _ck_iter = int(_ck_iv) if _ck_iv else int(args_cli.eval_interval)
    ppo_cfg["experiment"]["checkpoint_interval"] = _ck_iter * rollouts
    if _ck_iv:
        print(
            f"[train] DEXENGINE_CKPT_INTERVAL={_ck_iter} iter "
            f"(= {_ck_iter * rollouts} timesteps) — eval_interval"
            f"={args_cli.eval_interval} is separate"
        )
    return ppo_cfg


def _ppo_config_from_yaml(
    exp_dir, exp_name, rollouts, wrapped, args_cli, _per_side_adv, agent_cfg, agent_params, device
):
    """Build PPO config from YAML hyperparameters."""
    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg["rollouts"] = rollouts
    ppo_cfg["learning_epochs"] = agent_params.get("learning_epochs", 5)
    ppo_cfg["mini_batches"] = agent_params.get("mini_batches", 4)
    if getattr(args_cli, "mini_batches", None) is not None:
        ppo_cfg["mini_batches"] = int(args_cli.mini_batches)
        print(f"[ppo] mini_batches override = {args_cli.mini_batches}")
    ppo_cfg["discount_factor"] = agent_params.get("discount_factor", 0.99)
    ppo_cfg["lambda"] = agent_params.get("lambda", 0.95)
    ppo_cfg["learning_rate"] = agent_params.get("learning_rate", 5e-4)
    ppo_cfg["grad_norm_clip"] = agent_params.get("grad_norm_clip", 1.0)
    ppo_cfg["ratio_clip"] = agent_params.get("ratio_clip", 0.2)
    ppo_cfg["value_clip"] = agent_params.get("value_clip", 0.2)
    ppo_cfg["clip_predicted_values"] = agent_params.get("clip_predicted_values", True)
    ppo_cfg["entropy_loss_scale"] = agent_params.get("entropy_loss_scale", 0.005)
    ppo_cfg["value_loss_scale"] = agent_params.get("value_loss_scale", 1.0)
    ppo_cfg["kl_threshold"] = agent_params.get("kl_threshold", 0.0)
    ppo_cfg["rewards_shaper_scale"] = agent_params.get("rewards_shaper_scale", 1.0)
    ppo_cfg["time_limit_bootstrap"] = agent_params.get("time_limit_bootstrap", True)
    ppo_cfg["random_timesteps"] = agent_params.get("random_timesteps", 0)
    ppo_cfg["learning_starts"] = agent_params.get("learning_starts", 0)

    # CLI overrides (parallel-experiment support)
    if args_cli.value_loss_scale is not None:
        ppo_cfg["value_loss_scale"] = args_cli.value_loss_scale
        print(f"[train] value_loss_scale override = {args_cli.value_loss_scale}")
    if args_cli.entropy_loss_scale is not None:
        ppo_cfg["entropy_loss_scale"] = args_cli.entropy_loss_scale
        print(f"[train] entropy_loss_scale override = {args_cli.entropy_loss_scale}")
    if args_cli.learning_rate is not None:
        ppo_cfg["learning_rate"] = args_cli.learning_rate
        print(f"[train] learning_rate override = {args_cli.learning_rate}")
    if args_cli.kl_threshold is not None:
        ppo_cfg["kl_threshold"] = args_cli.kl_threshold
        print(f"[train] kl_threshold override = {args_cli.kl_threshold}")
    if args_cli.ratio_clip is not None:
        ppo_cfg["ratio_clip"] = args_cli.ratio_clip
        print(f"[train] ratio_clip override = {args_cli.ratio_clip}")
    if args_cli.grad_norm_clip is not None:
        ppo_cfg["grad_norm_clip"] = args_cli.grad_norm_clip
        print(f"[train] grad_norm_clip override = {args_cli.grad_norm_clip}")

    # Learning rate scheduler: DISABLED for residual training.
    # ManipTrans uses a short warmup then static LR; we keep it fully static.
    # KLAdaptiveLR was dropping LR too aggressively when the residual policy
    # took large early updates, starving the residual of training signal.
    # (Set learning_rate_scheduler in the yaml to re-enable if desired.)
    lr_scheduler = agent_params.get("learning_rate_scheduler", None)
    if lr_scheduler == "KLAdaptiveLR":
        ppo_cfg["learning_rate_scheduler"] = skrl.resources.schedulers.torch.KLAdaptiveLR
        ppo_cfg["learning_rate_scheduler_kwargs"] = agent_params.get(
            "learning_rate_scheduler_kwargs", {"kl_threshold": 0.008}
        )

    # State preprocessor (RunningStandardScaler on 443-dim obs)
    ppo_cfg["state_preprocessor"] = RunningStandardScaler
    ppo_cfg["state_preprocessor_kwargs"] = {"size": wrapped.observation_space, "device": device}
    ppo_cfg["value_preprocessor"] = RunningStandardScaler
    ppo_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    if _per_side_adv:
        # PER-SIDE ADVANTAGE: the two-head critic emits (N, 2) = [V_r, V_l];
        # the value scaler must be 2-dim (new-mode checkpoint format).
        ppo_cfg["value_preprocessor_kwargs"] = {"size": 2, "device": device}
        print("[ppo] PER-SIDE ADVANTAGE: value_preprocessor size = 2 ([V_r, V_l])")

    # Experiment config
    ppo_cfg["experiment"] = agent_cfg.get("agent", {}).get("experiment", {})
    ppo_cfg["experiment"]["directory"] = os.path.dirname(exp_dir)
    ppo_cfg["experiment"]["experiment_name"] = exp_name
    return ppo_cfg
