"""PER-SIDE ADVANTAGE PPO (factored credit) for bimanual two-tower policies.

Subclass of skrl PPO used ONLY under --per_side_advantage 1 --bimanual 1 (the trainer builds it; every other run
uses stock PPO).

Requirements this PPO learner assumes:
  1. env extras expose per-step ``reward_right`` / ``reward_left`` tensors
     (ArmResidualTrainEnv with cfg.per_side_reward=True) with r_R + r_L == total.
  2. the value model is TWO-HEAD: returns (N, 2) = [V_right, V_left]
     (TokenBimanualValueTwoHead / SmallBimanualValueTwoHead).
  3. the policy is a diagonal Gaussian whose mu layout is [right | left]
     (TokenPerHandResidualPolicy / SmallPerHandResidualPolicy) — per-dim
     log-probs are separable, so per-side log-probs are exact partial sums.

Overridden skrl internals:
  - ``init``: extra memory tensors (rewards_right/left, values_two, returns_two,
    advantages_right/left, log_prob_r/l) and the per-side sampled-tensor list.
  - ``act``: additionally stores per-side OLD log-probs (partial sums of the
    per-dim Gaussian log-probs over each tower's action dims).
  - ``record_transition``: reads reward_right/left from infos, evaluates the
    two-head critic once, applies per-side time-limit bootstrap, stores all
    tensors in a single add_samples call.
  - ``_update``: per-side GAE (same gamma/lambda), per-side PPO-clip losses
    summed, value loss = MSE(V_r, ret_r) + MSE(V_l, ret_l), KL (threshold
    early-stop AND any adaptive-LR scheduler) = sum of the per-side KLs.
    DISTRIBUTED (multigpu): the KL is all-reduced to the global mean BEFORE the
    early-stop break so every rank takes the same branch (stock skrl 1.4.3
    breaks on the local KL — desyncs the gradient all-reduce count). The
    reduce_parameters() gradient all-reduce + KLAdaptiveLR all-reduce mirror
    stock PPO. On a single GPU the all-reduce branches are not taken.

Checkpoint format: module names unchanged (policy/value/optimizer/
state_preprocessor/value_preprocessor) but value + value_preprocessor are
2-dim — a format of its own. Runs without --per_side_advantage never construct this class, so their
checkpoint format is the stock one.
"""

# Portions follow skrl (MIT; Copyright (c) 2021 Toni-SM). The licence text is in THIRD_PARTY.md.

import itertools
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl import config
from skrl.agents.torch import Agent
from skrl.agents.torch.ppo import PPO
from skrl.resources.schedulers.torch import KLAdaptiveLR

from dexengine.rl.agents.loss_sums import LossSums


class PerSidePPO(PPO):
    """PPO with per-side reward streams, per-side GAE, and factored policy loss."""

    def __init__(self, models, memory=None, observation_space=None, action_space=None, device=None, cfg=None):
        cfg = dict(cfg or {})
        # (right_dim, left_dim) action split; mu layout is [right | left].
        self._per_side_dims = tuple(cfg.pop("per_side_dims", (12, 12)))
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=cfg,
        )
        self._current_log_prob_r = None
        self._current_log_prob_l = None

    def init(self, trainer_cfg=None) -> None:
        """Create the per-side memory tensors on top of the stock PPO set."""
        super().init(trainer_cfg=trainer_cfg)
        if self.memory is not None:
            self.memory.create_tensor(name="rewards_right", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="rewards_left", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values_two", size=2, dtype=torch.float32)
            self.memory.create_tensor(name="returns_two", size=2, dtype=torch.float32)
            self.memory.create_tensor(name="advantages_right", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages_left", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="log_prob_r", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="log_prob_l", size=1, dtype=torch.float32)
            # reset_scheme=decoupled: per-side done streams. Under the per-side snap the env publishes
            # done_right/done_left extras (a side's gate failure no longer ends the episode); each side's GAE must
            # terminate at ITS OWN done so the value function never bootstraps across the side teleport. Absent
            # extras -> zeros -> the per-side dones reduce to the env-level dones.
            self.memory.create_tensor(name="dones_right", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="dones_left", size=1, dtype=torch.bool)
            # Per-side minibatch fields (replaces the stock 1-stream list).
            self._tensors_names = [
                "states",
                "actions",
                "log_prob_r",
                "log_prob_l",
                "values_two",
                "returns_two",
                "advantages_right",
                "advantages_left",
            ]

    def _split_log_prob(self, actions: torch.Tensor) -> tuple:
        """Per-side log-probs of `actions` under the CURRENT policy distribution.

        Diagonal Gaussian -> per-dim log-probs are independent; each side's
        log-prob is the partial sum over its tower's action dims.
        """
        d_r = self._per_side_dims[0]
        lp = self.policy.distribution(role="policy").log_prob(actions)  # (N, act_dim)
        return lp[..., :d_r].sum(-1, keepdim=True), lp[..., d_r:].sum(-1, keepdim=True)

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        actions, log_prob, outputs = super().act(states, timestep, timesteps)
        # random_timesteps path has no distribution; unused in this repo (always 0).
        if timestep >= self._random_timesteps:
            with torch.no_grad():
                self._current_log_prob_r, self._current_log_prob_l = self._split_log_prob(actions)
        return actions, log_prob, outputs

    def record_transition(
        self, states, actions, rewards, next_states, terminated, truncated, infos: Any, timestep: int, timesteps: int
    ) -> None:
        # The skrl base class handles cumulative-reward tracking / timestep bookkeeping.
        Agent.record_transition(
            self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )
        if self.memory is None:
            return
        self._current_next_states = next_states

        try:
            r_r = infos["reward_right"].detach().to(self.device, dtype=torch.float32).view(-1, 1)
            r_l = infos["reward_left"].detach().to(self.device, dtype=torch.float32).view(-1, 1)
        except (KeyError, TypeError) as e:
            raise RuntimeError(
                "[PerSidePPO] env extras missing reward_right/reward_left — the env must be "
                "built with cfg.per_side_reward=True (--per_side_advantage 1)"
            ) from e

        # Per-side dones (reset_scheme=decoupled). Missing extras -> zeros (env-level dones only).
        _d_r = infos.get("done_right") if hasattr(infos, "get") else None
        _d_l = infos.get("done_left") if hasattr(infos, "get") else None
        d_r = (
            _d_r.detach().to(self.device, dtype=torch.bool).view(-1, 1)
            if _d_r is not None
            else torch.zeros_like(terminated, dtype=torch.bool)
        )
        d_l = (
            _d_l.detach().to(self.device, dtype=torch.bool).view(-1, 1)
            if _d_l is not None
            else torch.zeros_like(terminated, dtype=torch.bool)
        )

        if self._rewards_shaper is not None:
            rewards = self._rewards_shaper(rewards, timestep, timesteps)
            r_r = self._rewards_shaper(r_r, timestep, timesteps)
            r_l = self._rewards_shaper(r_l, timestep, timesteps)

        # Two-head critic: one forward gives both side values (raw scale after inverse).
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            values_two, _, _ = self.value.act({"states": self._state_preprocessor(states)}, role="value")  # (N, 2)
            values_two = self._value_preprocessor(values_two, inverse=True)
        values_total = values_two.sum(-1, keepdim=True)

        # Time-limit bootstrap per stream (total kept consistent = sum of sides).
        if self._time_limit_bootstrap:
            rewards = rewards + self._discount_factor * values_total * truncated
            r_r = r_r + self._discount_factor * values_two[:, 0:1] * truncated
            r_l = r_l + self._discount_factor * values_two[:, 1:2] * truncated

        _samples = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "terminated": terminated,
            "truncated": truncated,
            "log_prob": self._current_log_prob,
            "values": values_total,
            "rewards_right": r_r,
            "rewards_left": r_l,
            "values_two": values_two,
            "log_prob_r": self._current_log_prob_r,
            "log_prob_l": self._current_log_prob_l,
            "dones_right": d_r,
            "dones_left": d_l,
        }
        self.memory.add_samples(**_samples)
        for memory in self.secondary_memories:
            memory.add_samples(**_samples)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Per-side GAE + factored PPO update (mirrors stock PPO._update structure)."""

        sampled_batches, sums = self._advantages_and_minibatches()

        for epoch in range(self._learning_epochs):
            kl_divergences = []

            for (
                sampled_states,
                sampled_actions,
                sampled_log_prob_r,
                sampled_log_prob_l,
                sampled_values_two,
                sampled_returns_two,
                sampled_advantages_r,
                sampled_advantages_l,
            ) in sampled_batches:
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    sampled_states = self._state_preprocessor(sampled_states, train=not epoch)

                    # Refresh the policy distribution on the sampled states, then take
                    # per-side partial log-prob sums of the taken actions.
                    _, _, _ = self.policy.act(
                        {"states": sampled_states, "taken_actions": sampled_actions}, role="policy"
                    )
                    next_log_prob_r, next_log_prob_l = self._split_log_prob(sampled_actions)

                    # KL per side; the total (sum) drives early-stop + adaptive LR.
                    with torch.no_grad():
                        rr = next_log_prob_r - sampled_log_prob_r
                        rl = next_log_prob_l - sampled_log_prob_l
                        _kl_r = ((torch.exp(rr) - 1) - rr).mean()
                        _kl_l = ((torch.exp(rl) - 1) - rl).mean()
                        kl_divergence = _kl_r + _kl_l
                        # DISTRIBUTED: the early-stop break below MUST take the same
                        # branch on every rank — a per-rank break desyncs the
                        # reduce_parameters() all-reduce count (cross-minibatch
                        # gradient mixing, eventual NCCL hang; stock skrl 1.4.3 has
                        # this bug). All-reduce the KL to the global mean so the
                        # decision (and the logged value) is rank-identical.
                        if config.torch.is_distributed:
                            torch.distributed.all_reduce(kl_divergence, op=torch.distributed.ReduceOp.SUM)
                            kl_divergence /= config.torch.world_size
                        kl_divergences.append(kl_divergence)

                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        break

                    entropy_loss, policy_loss, policy_loss_l, policy_loss_r, value_loss = self._per_side_losses(
                        next_log_prob_l,
                        next_log_prob_r,
                        sampled_advantages_l,
                        sampled_advantages_r,
                        sampled_log_prob_l,
                        sampled_log_prob_r,
                        sampled_returns_two,
                        sampled_states,
                        sampled_values_two,
                    )

                self._optimizer_step(entropy_loss, policy_loss, value_loss)

                sums.add("policy", policy_loss)
                sums.add("policy_r", policy_loss_r)
                sums.add("policy_l", policy_loss_l)
                sums.add("value", value_loss)
                if self._entropy_loss_scale:
                    sums.add("entropy", entropy_loss)

            self._step_learning_rate(kl_divergences)

        self._record_transition_data(sums)

    def _per_side_losses(
        self,
        next_log_prob_l,
        next_log_prob_r,
        sampled_advantages_l,
        sampled_advantages_r,
        sampled_log_prob_l,
        sampled_log_prob_r,
        sampled_returns_two,
        sampled_states,
        sampled_values_two,
    ):
        """Entropy, factored per-side clipped-surrogate policy and two-head value losses of one mini-batch."""
        if self._entropy_loss_scale:
            entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
        else:
            entropy_loss = 0

        # Factored policy loss: each tower's ratio graded ONLY by its
        # own side's advantage (PPO-clip per side, summed).
        ratio_r = torch.exp(next_log_prob_r - sampled_log_prob_r)
        surrogate_r = sampled_advantages_r * ratio_r
        surrogate_r_clipped = sampled_advantages_r * torch.clip(ratio_r, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip)
        policy_loss_r = -torch.min(surrogate_r, surrogate_r_clipped).mean()

        ratio_l = torch.exp(next_log_prob_l - sampled_log_prob_l)
        surrogate_l = sampled_advantages_l * ratio_l
        surrogate_l_clipped = sampled_advantages_l * torch.clip(ratio_l, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip)
        policy_loss_l = -torch.min(surrogate_l, surrogate_l_clipped).mean()

        policy_loss = policy_loss_r + policy_loss_l

        # Two-head value loss: MSE per side, summed (optional clip per side).
        predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")
        if self._clip_predicted_values:
            predicted_values = sampled_values_two + torch.clip(
                predicted_values - sampled_values_two, min=-self._value_clip, max=self._value_clip
            )
        value_loss = self._value_loss_scale * (
            F.mse_loss(sampled_returns_two[:, 0:1], predicted_values[:, 0:1])
            + F.mse_loss(sampled_returns_two[:, 1:2], predicted_values[:, 1:2])
        )
        return entropy_loss, policy_loss, policy_loss_l, policy_loss_r, value_loss

    def _optimizer_step(self, entropy_loss, policy_loss, value_loss):
        """Backward pass, gradient reduce/clip and optimizer step on the summed loss (identical to stock PPO)."""
        # optimization step (identical to stock PPO)
        self.optimizer.zero_grad()
        self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

        if config.torch.is_distributed:
            self.policy.reduce_parameters()
            if self.policy is not self.value:
                self.value.reduce_parameters()

        if self._grad_norm_clip > 0:
            self.scaler.unscale_(self.optimizer)
            if self.policy is self.value:
                nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
            else:
                nn.utils.clip_grad_norm_(
                    itertools.chain(self.policy.parameters(), self.value.parameters()), self._grad_norm_clip
                )

        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _step_learning_rate(self, kl_divergences):
        """Scheduler step after an epoch; the KL-adaptive scheduler reads the mean (rank-averaged) total KL."""
        # adaptive LR: total KL (sum of per-side KLs)
        if self._learning_rate_scheduler:
            if isinstance(self.scheduler, KLAdaptiveLR):
                kl = torch.tensor(kl_divergences, device=self.device).mean()
                if config.torch.is_distributed:
                    torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                    kl /= config.torch.world_size
                self.scheduler.step(kl.item())
            else:
                self.scheduler.step()

    def _advantages_and_minibatches(self):
        """Per-side GAE returns and advantages, then the sampled mini-batches and loss sums."""

        def compute_gae_side(rewards, dones, values, last_values, discount_factor, lambda_coefficient):
            """Standard GAE over one side's (T, N, 1) streams; returns/normalized adv."""
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return returns, advantages

        sampled_batches, sums = self._bootstrap_final_values(compute_gae_side)
        return sampled_batches, sums

    def _record_transition_data(self, sums):
        """record data (stock keys + per-side diagnostics)."""
        mean = sums.means(self._learning_epochs * self._mini_batches)
        self.track_data("Loss / Policy loss", mean["policy"])
        self.track_data("Loss / Value loss", mean["value"])
        self.track_data("PerSide / Policy loss right", mean["policy_r"])
        self.track_data("PerSide / Policy loss left", mean["policy_l"])
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", mean["entropy"])

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

    def _bootstrap_final_values(self, compute_gae_side):
        """Bootstrap values for the final transition (two heads, raw scale)."""
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            last_values_two, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float())}, role="value"
            )
            self.value.train(True)
            last_values_two = self._value_preprocessor(last_values_two, inverse=True)  # (N, 2)

        values_two = self.memory.get_tensor_by_name("values_two")  # (T, N, 2) raw scale
        dones = self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated")
        # Per-side dones: env-level dones OR the side's own snap-done (zeros unless
        # reset_scheme=decoupled -> reduces to `dones`).
        dones_r = dones | self.memory.get_tensor_by_name("dones_right")
        dones_l = dones | self.memory.get_tensor_by_name("dones_left")
        returns_r, advantages_r = compute_gae_side(
            rewards=self.memory.get_tensor_by_name("rewards_right"),
            dones=dones_r,
            values=values_two[..., 0:1],
            last_values=last_values_two[..., 0:1],
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )
        returns_l, advantages_l = compute_gae_side(
            rewards=self.memory.get_tensor_by_name("rewards_left"),
            dones=dones_l,
            values=values_two[..., 1:2],
            last_values=last_values_two[..., 1:2],
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )
        returns_two = torch.cat([returns_r, returns_l], dim=-1)  # (T, N, 2)

        # 2-dim value preprocessor: values then returns update the running stats
        # (same double-train order as stock PPO).
        self.memory.set_tensor_by_name("values_two", self._value_preprocessor(values_two, train=True))
        self.memory.set_tensor_by_name("returns_two", self._value_preprocessor(returns_two, train=True))
        self.memory.set_tensor_by_name("advantages_right", advantages_r)
        self.memory.set_tensor_by_name("advantages_left", advantages_l)

        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)

        sums = LossSums(("policy", "policy_r", "policy_l", "value", "entropy"), self.device)
        return sampled_batches, sums
