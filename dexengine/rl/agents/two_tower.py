"""Bimanual two-tower MLP policies and centralized critics (``--policy_arch small | shared`` with ``--bimanual 1``).

The right and left observation slices go through separate towers (no shared weights, no cross-hand gradient path);
the output mean is ordered [right, left] to match the action-manager term order. Pure torch + skrl mixins; class and
attribute names are part of the checkpoint format.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


def _small_tower(n_in):
    return nn.Sequential(
        nn.Linear(n_in, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU()
    )


class SmallPerHandResidualPolicy(GaussianMixin, Model):
    """Two independent small MLP towers over right/left obs slices; heads concat."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        right_idx,
        left_idx,
        per_hand_act_dim=12,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20.0,
        max_log_std=2.0,
        initial_log_std=-1.0,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self, clip_actions=clip_actions, clip_log_std=clip_log_std, min_log_std=min_log_std, max_log_std=max_log_std
        )
        self.register_buffer("_ri", torch.tensor(right_idx, dtype=torch.long))
        self.register_buffer("_li", torch.tensor(left_idx, dtype=torch.long))
        self.net_r = _small_tower(len(right_idx))
        self.net_l = _small_tower(len(left_idx))
        self.head_r = nn.Linear(128, per_hand_act_dim)
        self.head_l = nn.Linear(128, per_hand_act_dim)
        for h in (self.head_r, self.head_l):
            nn.init.xavier_normal_(h.weight, gain=0.01)
            nn.init.zeros_(h.bias)
        self.log_std_parameter = nn.Parameter(torch.full((action_space.shape[0],), initial_log_std))

    def compute(self, inputs, role=""):
        obs = inputs["states"]
        mu = torch.cat(
            [
                self.head_r(self.net_r(obs.index_select(-1, self._ri))),
                self.head_l(self.net_l(obs.index_select(-1, self._li))),
            ],
            dim=-1,
        )
        return mu, self.log_std_parameter.expand_as(mu), {}


class SmallBimanualValue(DeterministicMixin, Model):
    """Centralized critic: small MLP over the full bimanual obs -> 1."""

    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.net = _small_tower(observation_space.shape[0])
        self.value_head = nn.Linear(128, 1)

    def compute(self, inputs, role=""):
        return self.value_head(self.net(inputs["states"])), {}


class SmallBimanualValueTwoHead(DeterministicMixin, Model):
    """PER-SIDE ADVANTAGE critic (small arch): shared MLP trunk, 2-dim head [V_r, V_l]."""

    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.net = _small_tower(observation_space.shape[0])
        self.value_head = nn.Linear(128, 2)

    def compute(self, inputs, role=""):
        return self.value_head(self.net(inputs["states"])), {}


class PerHandResidualPolicy(GaussianMixin, Model):
    """Bimanual per-hand (two-tower) residual policy — the bimanual form of ``--policy_arch shared``.

    The right and left observation slices are routed through SEPARATE encoder+MLP+head
    towers that share no trunk weights and no gradient path. Output mu is ordered
    [right(12), left(12)] to match the ActionManager term order
    (right_hand_joint_pos = dims [0, 12), left_hand_joint_pos = dims [12, 24)).

    Per-tower architecture mirrors ResidualPolicy exactly: encoder(in->512->512->512->256,
    SiLU) -> residual MLP(256->256->512->128->64, ELU) -> head(64->per_hand_act_dim), with
    near-zero head init (xavier gain=0.01). log_std is a single (act_dim,) parameter
    (per-dim independent; the coupling removed is the shared trunk and summed-gradient path,
    not the per-dim std).

    Why: a shared trunk with a summed reward couples the two hands' exploration, so one hand's diverging residual
    can drag the whole policy down. Each hand-object pair stays coupled internally; cross-hand coupling is removed.
    No aux-force head (the factory rejects per-hand + aux).
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        right_idx,
        left_idx,
        per_hand_act_dim,
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
        assert per_hand_act_dim * 2 == act_dim, f"per_hand_act_dim*2 ({per_hand_act_dim * 2}) != act_dim ({act_dim})"

        # Obs index routing — registered as buffers so they move with .to(device) and persist.
        self.register_buffer("right_idx", torch.as_tensor(list(right_idx), dtype=torch.long))
        self.register_buffer("left_idx", torch.as_tensor(list(left_idx), dtype=torch.long))

        def _tower(in_dim, out_dim):
            enc = nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.SiLU(),
                nn.Linear(512, 512),
                nn.SiLU(),
                nn.Linear(512, 512),
                nn.SiLU(),
                nn.Linear(512, 256),
            )
            mlp = nn.Sequential(
                nn.Linear(256, 256),
                nn.ELU(),
                nn.Linear(256, 512),
                nn.ELU(),
                nn.Linear(512, 128),
                nn.ELU(),
                nn.Linear(128, 64),
                nn.ELU(),
            )
            head = nn.Linear(64, out_dim)
            nn.init.xavier_normal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
            return enc, mlp, head

        self.r_encoder, self.r_mlp, self.r_head = _tower(len(self.right_idx), per_hand_act_dim)
        self.l_encoder, self.l_mlp, self.l_head = _tower(len(self.left_idx), per_hand_act_dim)

        self.log_std_parameter = nn.Parameter(torch.full((act_dim,), initial_log_std))

    def compute(self, inputs, role=""):
        obs = inputs["states"]
        mu_r = self.r_head(self.r_mlp(self.r_encoder(obs.index_select(-1, self.right_idx))))
        mu_l = self.l_head(self.l_mlp(self.l_encoder(obs.index_select(-1, self.left_idx))))
        mu = torch.cat([mu_r, mu_l], dim=-1)  # [right(12), left(12)]
        log_std = self.log_std_parameter.expand_as(mu)
        return mu, log_std, {}
