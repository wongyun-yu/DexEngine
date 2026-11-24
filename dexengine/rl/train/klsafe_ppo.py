"""KL-safe stock PPO for MULTI-GPU runs.

Upstream skrl 1.4.3 PPO breaks the minibatch loop on the LOCAL (per-rank) KL
divergence when ``kl_threshold > 0``. Under torch.distributed the ranks can
therefore take DIFFERENT numbers of optimizer steps, desyncing the
``reduce_parameters()`` all-reduce sequence (silent cross-minibatch gradient
mixing, eventual NCCL hang). PerSidePPO carries the fix on the per-side path;
this class is the same fix for the STOCK path: ``_update`` below is DERIVED
from the installed skrl 1.4.3 ``PPO._update`` (split into helpers) with ONE change — the KL used
by the early-stop break is all-reduced (mean) across ranks first, so every rank
takes the break on the same minibatch. The trainer selects it only for distributed runs with a KL threshold;
single-GPU runs keep the upstream class.
"""

# Portions follow skrl (MIT; Copyright (c) 2021 Toni-SM). The licence text is in THIRD_PARTY.md.

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl import config
from skrl.agents.torch.ppo import PPO
from skrl.resources.schedulers.torch import KLAdaptiveLR

from dexengine.rl.agents.loss_sums import LossSums


class KLSafePPO(PPO):
    """Stock PPO + rank-synchronized KL early-stop (see module docstring)."""

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        self._store_returns_and_advantages()

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)

        sums = LossSums(("policy", "value", "entropy"), self.device)

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for (
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in sampled_batches:
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    sampled_states = self._state_preprocessor(sampled_states, train=not epoch)

                    _, next_log_prob, _ = self.policy.act(
                        {"states": sampled_states, "taken_actions": sampled_actions}, role="policy"
                    )

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence — RANK-SYNCED (the ONE change
                    # vs upstream): all-reduce the KL before the comparison so every
                    # rank takes the break on the same minibatch (same fix as
                    # PerSidePPO; upstream compares the LOCAL per-rank KL).
                    if self._kl_threshold:
                        _kl_sync = kl_divergence.detach().clone()
                        if config.torch.is_distributed:
                            torch.distributed.all_reduce(_kl_sync, op=torch.distributed.ReduceOp.SUM)
                            _kl_sync /= config.torch.world_size
                        if _kl_sync > self._kl_threshold:
                            break

                    entropy_loss, policy_loss, ratio, value_loss = self._ppo_losses(
                        next_log_prob,
                        ratio,
                        sampled_advantages,
                        sampled_log_prob,
                        sampled_returns,
                        sampled_states,
                        sampled_values,
                    )

                self._optimizer_step(entropy_loss, policy_loss, value_loss)

                # update cumulative losses
                sums.add("policy", policy_loss)
                sums.add("value", value_loss)
                if self._entropy_loss_scale:
                    sums.add("entropy", entropy_loss)

            self._step_learning_rate(kl_divergences)

        self._record_data(sums)

    def _ppo_losses(
        self,
        next_log_prob,
        ratio,
        sampled_advantages,
        sampled_log_prob,
        sampled_returns,
        sampled_states,
        sampled_values,
    ):
        """Entropy, clipped-surrogate policy and (optionally clipped) value losses of one mini-batch."""
        # compute entropy loss
        if self._entropy_loss_scale:
            entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
        else:
            entropy_loss = 0

        # compute policy loss
        ratio = torch.exp(next_log_prob - sampled_log_prob)
        surrogate = sampled_advantages * ratio
        surrogate_clipped = sampled_advantages * torch.clip(ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip)

        policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

        # compute value loss
        predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")

        if self._clip_predicted_values:
            predicted_values = sampled_values + torch.clip(
                predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
            )
        value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)
        return entropy_loss, policy_loss, ratio, value_loss

    def _step_learning_rate(self, kl_divergences):
        """Scheduler step after an epoch; the KL-adaptive scheduler reads the mean (rank-averaged) KL."""
        # update learning rate
        if self._learning_rate_scheduler:
            if isinstance(self.scheduler, KLAdaptiveLR):
                kl = torch.tensor(kl_divergences, device=self.device).mean()
                # reduce (collect from all workers/processes) KL in distributed runs
                if config.torch.is_distributed:
                    torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                    kl /= config.torch.world_size
                self.scheduler.step(kl.item())
            else:
                self.scheduler.step()

    def _optimizer_step(self, entropy_loss, policy_loss, value_loss):
        """Backward pass, gradient reduce/clip and optimizer step on the summed loss."""
        # optimization step
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

    def _record_data(self, sums):
        """record data."""
        mean = sums.means(self._learning_epochs * self._mini_batches)
        self.track_data("Loss / Policy loss", mean["policy"])
        self.track_data("Loss / Value loss", mean["value"])
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", mean["entropy"])

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

    def _store_returns_and_advantages(self):
        """Bootstrap the last values, compute GAE returns and advantages, store them in memory."""

        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)

            :param rewards: Rewards obtained by the policy
            :type rewards: torch.Tensor
            :param dones: Signals to indicate that episodes have ended
            :type dones: torch.Tensor
            :param values: Values obtained by the policy
            :type values: torch.Tensor
            :param next_values: Next values obtained by the policy
            :type next_values: torch.Tensor
            :param discount_factor: Discount factor
            :type discount_factor: float
            :param lambda_coefficient: Lambda coefficient
            :type lambda_coefficient: float

            :return: Generalized Advantage Estimator
            :rtype: torch.Tensor
            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            last_values, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float())}, role="value"
            )
            self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)
