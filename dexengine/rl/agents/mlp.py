"""Single-trunk MLP policy/value models (``--policy_arch shared | structured | small``).

Pure torch + skrl mixins (no Isaac import). Class and attribute names are part of the checkpoint format: the
state_dict keys of saved agents must keep loading.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


class ResidualPolicy(GaussianMixin, Model):
    """Residual policy: feature encoder + residual MLP (``--policy_arch shared``, one hand).

    The base action is the kinematic reference (``--reference_base``); this network is the only trained model and
    its output is a residual on that reference.

    Architecture (about 1.5M parameters):
      obs -> encoder (512->512->512->256, SiLU) -> residual MLP (256->256->512->128->64, ELU) -> mu
      log_std: a state-independent parameter, initialised to ``initial_log_std``.
      mu_head: xavier_normal with gain 0.01, so the initial residual is near zero.

    Auxiliary contact-force head ``aux_force_head`` (encoder features 256->256->15): predicts the next-frame
    fingertip forces, shape (batch, 5, 3) = 5 fingertips x xyz. It is always in the state_dict; the trainer's
    ``--aux_force_loss_scale`` decides whether its loss is used.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20.0,
        max_log_std=2.0,
        initial_log_std=-1.0,
        aux_decouple_encoder: bool = False,
        aux_action_aware: bool = False,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        self._build_feature_encoder(initial_log_std, aux_decouple_encoder, aux_action_aware, act_dim, obs_dim)

    def _build_feature_encoder(self, initial_log_std, aux_decouple_encoder, aux_action_aware, act_dim, obs_dim):
        """Feature encoder."""
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
        )

        # Optional decoupled aux encoder: the aux heads read this separate encoder so the aux loss does not
        # backpropagate into the main encoder. Its detached features are concatenated to the main features as
        # extra policy input. Output width 256 = the main encoder's, so the aux heads work either way.
        self._aux_decouple = aux_decouple_encoder
        if self._aux_decouple:
            self.aux_encoder = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.SiLU(),
                nn.Linear(256, 256),
                nn.SiLU(),
                nn.Linear(256, 256),
            )
            residual_in_dim = 256 + 256  # main features + detached aux features
        else:
            self.aux_encoder = None
            residual_in_dim = 256

        # Residual MLP.
        self.residual_mlp = nn.Sequential(
            nn.Linear(residual_in_dim, 256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.mu_head = nn.Linear(64, act_dim)

        # Near-zero initial residual.
        nn.init.xavier_normal_(self.mu_head.weight, gain=0.01)
        nn.init.zeros_(self.mu_head.bias)

        # Sigma: a parameter, not state-dependent.
        self.log_std_parameter = nn.Parameter(torch.full((act_dim,), initial_log_std))

        # Auxiliary contact-force head: encoder features (256) -> hidden (256) -> 15 = 5 fingertips x xyz.
        # Always present in the state_dict; the loss scale is set by the trainer.
        self.aux_force_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 15),
        )

        # Action-aware aux head, allocated only when aux_action_aware=True. Allocating it otherwise would consume
        # torch RNG and shift every later random draw (initialisation and domain randomisation).
        self._aux_action_aware = aux_action_aware
        if aux_action_aware:
            self.aux_force_head_aware = nn.Sequential(
                nn.Linear(256 + act_dim, 256),
                nn.ELU(),
                nn.Linear(256, 15),
            )
        else:
            self.aux_force_head_aware = None

    def _aux_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Return features for aux heads. Decoupled mode → aux_encoder; otherwise main encoder."""
        if self._aux_decouple:
            return self.aux_encoder(obs)
        return self.encoder(obs)

    def compute(self, inputs, role=""):
        obs = inputs["states"]
        features = self.encoder(obs)
        if self._aux_decouple:
            # Detached: the policy loss does not update aux_encoder (the aux loss does, separately).
            aux_feats = self.aux_encoder(obs).detach()
            features = torch.cat([features, aux_feats], dim=-1)
        hidden = self.residual_mlp(features)
        mu = self.mu_head(hidden)
        log_std = self.log_std_parameter.expand_as(mu)
        return mu, log_std, {}

    def predict_forces(self, obs: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        """Predict next-frame fingertip contact forces (aux head forward pass).

        In decoupled mode the gradient flows aux_force_head -> aux_encoder only; otherwise into the main encoder.
        With the action-aware head allocated and ``action`` given, the action is concatenated to the encoder
        features before the head; otherwise the state-only head is used.

        Args:
            obs: (batch, obs_dim) preprocessor-scaled observation.
            action: (batch, act_dim) action taken at this timestep (a[t]). Required
                when aux_action_aware=True; ignored otherwise.

        Returns:
            predicted_forces: (batch, 5, 3) predicted xyz contact force per fingertip.
        """
        features = self._aux_features(obs)
        if self._aux_action_aware and action is not None and self.aux_force_head_aware is not None:
            features_aug = torch.cat([features, action], dim=-1)
            raw = self.aux_force_head_aware(features_aug)
        else:
            raw = self.aux_force_head(features)
        return raw.view(raw.shape[0], 5, 3)


class ResidualValue(DeterministicMixin, Model):
    """Value network: same encoder architecture -> scalar output."""

    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        obs_dim = observation_space.shape[0]

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
        )

        self.value_mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role=""):
        obs = inputs["states"]
        features = self.encoder(obs)
        value = self.value_mlp(features)
        return value, {}


class StructuredResidualPolicy(GaussianMixin, Model):
    """Purpose-separated encoder policy over the obs_v2 one-hand layout (``--policy_arch structured``).

    Three branch encoders — proprio+action-context(54), reference-tracking(66),
    object-state(87) — each n_in→128→64 (SiLU), fused 192 → trunk 256→128 (SiLU)
    → mu head (xavier gain=0.01, near-zero initial residual). log_std = fixed param.
    ~0.13M params (≈11× smaller than ResidualPolicy's 1.5M). NO aux heads.
    Slice indices are registered as buffers (saved in ckpt; eval rebuilds identically
    from the live obs layout — mismatch would fail the load, not silently skew).
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        slices,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20.0,
        max_log_std=2.0,
        initial_log_std=-1.0,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )
        act_dim = action_space.shape[0]
        self.register_buffer("_idx_pa", torch.tensor(slices["proprio_action"], dtype=torch.long))
        self.register_buffer("_idx_tr", torch.tensor(slices["tracking"], dtype=torch.long))
        self.register_buffer("_idx_obj", torch.tensor(slices["object"], dtype=torch.long))

        def _enc(n_in):
            return nn.Sequential(
                nn.Linear(n_in, 128),
                nn.SiLU(),
                nn.Linear(128, 64),
                nn.SiLU(),
            )

        self.enc_pa = _enc(len(slices["proprio_action"]))
        self.enc_tr = _enc(len(slices["tracking"]))
        self.enc_obj = _enc(len(slices["object"]))
        self.trunk = nn.Sequential(
            nn.Linear(192, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(128, act_dim)
        nn.init.xavier_normal_(self.mu_head.weight, gain=0.01)
        nn.init.zeros_(self.mu_head.bias)
        self.log_std_parameter = nn.Parameter(torch.full((act_dim,), initial_log_std))

    def compute(self, inputs, role=""):
        obs = inputs["states"]
        f = torch.cat(
            [
                self.enc_pa(obs.index_select(-1, self._idx_pa)),
                self.enc_tr(obs.index_select(-1, self._idx_tr)),
                self.enc_obj(obs.index_select(-1, self._idx_obj)),
            ],
            dim=-1,
        )
        mu = self.mu_head(self.trunk(f))
        return mu, self.log_std_parameter.expand_as(mu), {}


class StructuredResidualValue(DeterministicMixin, Model):
    """Value net matching StructuredResidualPolicy's branch design (head → 1)."""

    def __init__(self, observation_space, action_space, device, slices, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.register_buffer("_idx_pa", torch.tensor(slices["proprio_action"], dtype=torch.long))
        self.register_buffer("_idx_tr", torch.tensor(slices["tracking"], dtype=torch.long))
        self.register_buffer("_idx_obj", torch.tensor(slices["object"], dtype=torch.long))

        def _enc(n_in):
            return nn.Sequential(
                nn.Linear(n_in, 128),
                nn.SiLU(),
                nn.Linear(128, 64),
                nn.SiLU(),
            )

        self.enc_pa = _enc(len(slices["proprio_action"]))
        self.enc_tr = _enc(len(slices["tracking"]))
        self.enc_obj = _enc(len(slices["object"]))
        self.trunk = nn.Sequential(
            nn.Linear(192, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
        )
        self.value_head = nn.Linear(128, 1)

    def compute(self, inputs, role=""):
        obs = inputs["states"]
        f = torch.cat(
            [
                self.enc_pa(obs.index_select(-1, self._idx_pa)),
                self.enc_tr(obs.index_select(-1, self._idx_tr)),
                self.enc_obj(obs.index_select(-1, self._idx_obj)),
            ],
            dim=-1,
        )
        return self.value_head(self.trunk(f)), {}


class SmallResidualPolicy(GaussianMixin, Model):
    """Plain small MLP (≈0.15M params, size-matched to Structured) — separates
    'purpose-separated structure' from 'fewer parameters' in the architecture comparison.
    obs → 256 → 256 → 128 (SiLU) → mu head (xavier gain=0.01). NO aux heads."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20.0,
        max_log_std=2.0,
        initial_log_std=-1.0,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(128, act_dim)
        nn.init.xavier_normal_(self.mu_head.weight, gain=0.01)
        nn.init.zeros_(self.mu_head.bias)
        self.log_std_parameter = nn.Parameter(torch.full((act_dim,), initial_log_std))

    def compute(self, inputs, role=""):
        mu = self.mu_head(self.net(inputs["states"]))
        return mu, self.log_std_parameter.expand_as(mu), {}


class SmallResidualValue(DeterministicMixin, Model):
    """Small value net matching SmallResidualPolicy."""

    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        obs_dim = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
        )
        self.value_head = nn.Linear(128, 1)

    def compute(self, inputs, role=""):
        return self.value_head(self.net(inputs["states"])), {}
