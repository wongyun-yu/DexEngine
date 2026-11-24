"""Semantic-token transformer policies (``--policy_arch token``, ``--policy_size tiny … large``).

The trunk is a standard pre-LN transformer (ViT/DiT-block style: MHA + 4x GELU MLP, residual) over semantic tokens
of the observation (``obs_layout``). PPO needs a tractable log-probability, so the action head is Gaussian. Pure
torch + skrl mixins (no Isaac import); class and attribute names are part of the checkpoint format.

Update-pass acceleration: a trunk call that computes gradients on CUDA (the PPO update) runs under bf16 autocast and
through ``torch.compile`` (LayerNorm, GELU and residual fusion; relative output difference about 3e-7). Rollouts and
evaluation run under ``no_grad`` and always take the eager FP32 path, so actions and evaluation results of the same
weights do not depend on it. Heads, distributions and log-probabilities stay FP32. The compiled backward pass needs
networkx 3.x.

The gradient pass computes the attention written out (``_attention_grad``: the MultiheadAttention's own projections,
softmax(q k^T / sqrt(d_head)) v) instead of the module call: with 17-34 tokens and 16-wide heads the fused
scaled-dot-product kernels the module dispatches to (flash attention) took 54 % of the update's GPU time, and the
written-out form compiles to a few dense matmuls (update minibatch 55.6 -> 36.6 ms bimanual, 78.9 -> 37.4 ms one
hand, one 24 GB GPU). Same parameters, same function; rollouts and evaluation keep the module call (bit-identical).
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from dexengine.rl.agents.obs_layout import _token_slices_bimanual

print(
    "[NUMERICS] token trunk under bf16 autocast in the update (gradient) pass · rollout/eval FP32",
    flush=True,
)
print(
    "[NUMERICS] token trunk compiled with torch.compile in the update (gradient) pass · rollout/eval eager", flush=True
)
print(
    "[NUMERICS] gradient-pass attention written out (_attention_grad) · rollout/eval module call",
    flush=True,
)

# Approximate policy sizes: tiny ≈0.12M, small ≈0.9M, base ≈4.8M, large ≈14M parameters.
_TOKEN_SIZES = {
    "tiny": {"d_model": 64, "n_layers": 2, "n_heads": 4},
    # tiny2x / tiny3x widen tiny at the same depth (2 layers): the step from tiny to small is about 7x in
    # parameters and also doubles the depth, which mixes a capacity question with an optimisation one (deep
    # transformers need warmup/LR tuning). Width-only steps keep the optimisation closest to tiny.
    "tiny2x": {"d_model": 96, "n_layers": 2, "n_heads": 4},  # 251,724 = 2.12x
    "tiny3x": {"d_model": 112, "n_layers": 2, "n_heads": 4},  # 336,684 = 2.84x
    "small": {"d_model": 128, "n_layers": 4, "n_heads": 4},
    "base": {"d_model": 256, "n_layers": 6, "n_heads": 8},
    "large": {"d_model": 384, "n_layers": 8, "n_heads": 8},
}


class _TokenBlock(nn.Module):
    """Pre-LN transformer block (ViT/DiT style): LN→MHA→res, LN→MLP(4x, GELU)→res."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        assert self.attn.dropout == 0.0 and self.attn._qkv_same_embed_dim  # the premises of _attention_grad
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        h = self.ln1(x)
        a = _attention_grad(self.attn, h) if torch.is_grad_enabled() else self.attn(h, h, h, need_weights=False)[0]
        x = x + a
        return x + self.mlp(self.ln2(x))


def _attention_grad(attn: nn.MultiheadAttention, h: torch.Tensor) -> torch.Tensor:
    """``attn(h, h, h, need_weights=False)[0]`` of a batch-first, dropout-free MultiheadAttention, written out with
    its own parameters (packed q/k/v projection, softmax(q k^T / sqrt(d_head)) v, output projection)."""
    b, t, d = h.shape
    nh = attn.num_heads
    qkv = F.linear(h, attn.in_proj_weight, attn.in_proj_bias).view(b, t, 3, nh, d // nh).permute(2, 0, 3, 1, 4)
    w = torch.softmax((qkv[0] * (1.0 / math.sqrt(d // nh))) @ qkv[1].transpose(-1, -2), dim=-1)
    return F.linear((w @ qkv[2]).transpose(1, 2).reshape(b, t, d), attn.out_proj.weight, attn.out_proj.bias)


class _TokenTrunk(nn.Module):
    """Tokenizer + transformer trunk shared by TokenResidualPolicy/Value."""

    def __init__(self, slices: dict, d_model: int, n_layers: int, n_heads: int):
        super().__init__()
        self.token_names = list(slices.keys())
        for i, name in enumerate(self.token_names):
            self.register_buffer(f"_idx_{i}", torch.tensor(slices[name], dtype=torch.long))
        self.embed = nn.ModuleList([nn.Linear(len(slices[n]), d_model) for n in self.token_names])
        self.type_embed = nn.Parameter(torch.zeros(len(self.token_names), d_model))
        nn.init.normal_(self.type_embed, std=0.02)
        self.blocks = nn.ModuleList([_TokenBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, obs):  # (B, obs_dim) -> (B, d_model)
        if torch.is_grad_enabled() and obs.is_cuda:
            return _trunk_update(self, obs)
        return self._trunk(obs)

    def _trunk(self, obs):
        toks = [emb(obs.index_select(-1, getattr(self, f"_idx_{i}"))) for i, emb in enumerate(self.embed)]
        x = torch.stack(toks, dim=1) + self.type_embed.unsqueeze(0)  # (B, T, d)
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x).mean(dim=1)  # mean-pool over tokens


def _trunk_update_path(trunk, obs):
    """The gradient-computing trunk call: bf16 autocast, output cast back to FP32."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return trunk._trunk(obs).float()


# Compiled as a module-level function, not an instance attribute, so state_dict and deepcopy are unaffected.
_trunk_update = torch.compile(_trunk_update_path)


class TokenResidualPolicy(GaussianMixin, Model):
    """Semantic-token transformer policy (tiny → large), Gaussian head for PPO."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        slices,
        size="small",
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
        cfgs = _TOKEN_SIZES[size]
        act_dim = action_space.shape[0]
        self.trunk = _TokenTrunk(slices, **cfgs)
        self.mu_head = nn.Linear(cfgs["d_model"], act_dim)
        nn.init.xavier_normal_(self.mu_head.weight, gain=0.01)
        nn.init.zeros_(self.mu_head.bias)
        self.log_std_parameter = nn.Parameter(torch.full((act_dim,), initial_log_std))

    def compute(self, inputs, role=""):
        mu = self.mu_head(self.trunk(inputs["states"]))
        return mu, self.log_std_parameter.expand_as(mu), {}


class TokenResidualValue(DeterministicMixin, Model):
    """Value net with the same token trunk (head → 1)."""

    def __init__(self, observation_space, action_space, device, slices, size="small", clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        cfgs = _TOKEN_SIZES[size]
        self.trunk = _TokenTrunk(slices, **cfgs)
        self.value_head = nn.Linear(cfgs["d_model"], 1)

    def compute(self, inputs, role=""):
        return self.value_head(self.trunk(inputs["states"])), {}


def _build_dual_policy_fn(raw_env, device, log_std_bounds, size="tiny", r_ckpt=None, l_ckpt=None):
    """Render-only two-policy inference for the bimanual scene (``--dual_eval 1``).

    Each hand is driven by its OWN one-hand token policy — the right-hand checkpoint drives the right hand, the
    left-hand checkpoint the left — with no checkpoint merge. The 414-dim bimanual policy observation is split into
    two 207-dim per-hand slices (``perhand_obs_slices``), each normalised with its checkpoint's own
    RunningStandardScaler and run through its TokenResidualPolicy; the two 12-dim means are concatenated into the
    24-dim bimanual action (right | left).

    The one-hand token architecture (17 tokens in ``_token_dict`` order) equals each hand's dict from
    ``_token_slices_bimanual``, so the per-token embedding weights align 1:1. Each token's ``_idx`` buffer is
    rebuilt from the compact (0-based into 207) remap, so indexing is correct for obs[:, {right,left}_idx]
    whatever the term order.

    r_ckpt / l_ckpt: the right- and left-hand policy checkpoints (--dual_ckpt_*).
    """
    return _load_dual_policy_fn(raw_env, device, log_std_bounds, size, r_ckpt, l_ckpt)


def _load_dual_policy_fn(raw_env, device, log_std_bounds, size, r_ckpt, l_ckpt):
    """Load the two side policies with their preprocessors and return the joint policy function."""
    import gymnasium as _gym
    from skrl.resources.preprocessors.torch import RunningStandardScaler

    from dexengine.rl.train.nonfinite_guard import assert_checkpoint_finite

    _min_ls, _max_ls, _init_ls = log_std_bounds
    assert r_ckpt and l_ckpt, "_build_dual_policy_fn requires r_ckpt and l_ckpt"
    r_ck, l_ck = r_ckpt, l_ckpt
    from dexengine.rl.env_setup.sides import perhand_obs_slices  # local: avoids import-order deps

    right_idx, left_idx = perhand_obs_slices(raw_env, 12, verbose=False)
    assert len(right_idx) == 207 and len(left_idx) == 207, (
        f"[DUAL] per-hand obs != 207: right={len(right_idx)} left={len(left_idx)}"
    )
    sr, sl, _sc = _token_slices_bimanual(raw_env)
    posR = {b: i for i, b in enumerate(right_idx)}
    posL = {b: i for i, b in enumerate(left_idx)}
    compactR = {k: [posR[j] for j in v] for k, v in sr.items()}
    compactL = {k: [posL[j] for j in v] for k, v in sl.items()}
    obs207 = _gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(207,))
    act12 = _gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(12,))

    def _mk(slices, ck_path, tag):
        pol = TokenResidualPolicy(
            observation_space=obs207,
            action_space=act12,
            device=device,
            slices=slices,
            size=size,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=_min_ls,
            max_log_std=_max_ls,
            initial_log_std=_init_ls,
        )
        pol.to(device)
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        assert_checkpoint_finite(ck, f"dual {tag}", ck_path)  # same gate as every checkpoint load site
        pol.load_state_dict(ck["policy"], strict=True)
        # Force the compact indices (authoritative for obs[:, {right,left}_idx]); they override whatever _idx the
        # checkpoint stored (one-hand 0-based, possibly in another order).
        for i, name in enumerate(pol.trunk.token_names):
            getattr(pol.trunk, f"_idx_{i}").copy_(torch.as_tensor(slices[name], dtype=torch.long, device=device))
        pol.eval()
        pre = RunningStandardScaler(size=obs207, device=device)
        pre.load_state_dict(ck["state_preprocessor"])
        pre.eval()
        print(f"  [DUAL {tag}] loaded {ck_path} (preproc mean_norm={pre.running_mean.float().norm().item():.3f})")
        return pol, pre

    pol_r, pre_r = _mk(compactR, r_ck, "R")
    pol_l, pre_l = _mk(compactL, l_ck, "L")
    _ri = torch.as_tensor(right_idx, dtype=torch.long, device=device)
    _li = torch.as_tensor(left_idx, dtype=torch.long, device=device)

    def policy_fn(obs_tensor):
        with torch.no_grad():
            o_r = obs_tensor.index_select(-1, _ri)
            o_l = obs_tensor.index_select(-1, _li)
            mu_r, _, _ = pol_r.compute({"states": pre_r(o_r)}, role="policy")
            mu_l, _, _ = pol_l.compute({"states": pre_l(o_l)}, role="policy")
            return torch.cat([mu_r, mu_l], dim=-1)

    print(
        f"  [DUAL] two-policy render ready: obs 414 -> R{len(right_idx)}+L{len(left_idx)}, "
        f"action 12|12 -> 24 (R={os.path.basename(os.path.dirname(os.path.dirname(r_ck)))}, "
        f"L={os.path.basename(os.path.dirname(os.path.dirname(l_ck)))})"
    )
    return policy_fn


# Bimanual two-tower token policies: each hand gets an independent trunk (no shared weights, no cross-hand gradient),
# so a one-hand tower and the matching side of a two-tower policy share the architecture. Output mu = [right, left].
# The value is a centralized critic over the full bimanual observation. All require the obs_v2 layout.


class TokenPerHandResidualPolicy(GaussianMixin, Model):
    """Two independent token trunks (right/left), heads concatenated [right, left]."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        slices_right,
        slices_left,
        size="tiny",
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
        cfgs = _TOKEN_SIZES[size]
        self.trunk_r = _TokenTrunk(slices_right, **cfgs)
        self.trunk_l = _TokenTrunk(slices_left, **cfgs)
        self.head_r = nn.Linear(cfgs["d_model"], per_hand_act_dim)
        self.head_l = nn.Linear(cfgs["d_model"], per_hand_act_dim)
        for h in (self.head_r, self.head_l):
            nn.init.xavier_normal_(h.weight, gain=0.01)
            nn.init.zeros_(h.bias)
        self.log_std_parameter = nn.Parameter(torch.full((action_space.shape[0],), initial_log_std))

    def compute(self, inputs, role=""):
        obs = inputs["states"]
        mu = torch.cat([self.head_r(self.trunk_r(obs)), self.head_l(self.trunk_l(obs))], dim=-1)
        return mu, self.log_std_parameter.expand_as(mu), {}


class TokenBimanualValue(DeterministicMixin, Model):
    """Centralized critic: single token trunk over ALL 34 tokens (both hands) -> 1."""

    def __init__(self, observation_space, action_space, device, slices_combined, size="tiny", clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        cfgs = _TOKEN_SIZES[size]
        self.trunk = _TokenTrunk(slices_combined, **cfgs)
        self.value_head = nn.Linear(cfgs["d_model"], 1)

    def compute(self, inputs, role=""):
        return self.value_head(self.trunk(inputs["states"])), {}


class TokenBimanualValueTwoHead(DeterministicMixin, Model):
    """PER-SIDE ADVANTAGE critic: TokenBimanualValue trunk, 2-dim head [V_right, V_left].

    Same centralized 34-token trunk as TokenBimanualValue; only the head widens to 2
    so per-side GAE gets its own value stream. Total value = output.sum(-1). Used
    ONLY under --per_side_advantage 1 (its own checkpoint format)."""

    def __init__(self, observation_space, action_space, device, slices_combined, size="tiny", clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        cfgs = _TOKEN_SIZES[size]
        self.trunk = _TokenTrunk(slices_combined, **cfgs)
        self.value_head = nn.Linear(cfgs["d_model"], 2)

    def compute(self, inputs, role=""):
        return self.value_head(self.trunk(inputs["states"])), {}


class TokenBimanualValueTwoTrunk(DeterministicMixin, Model):
    """PER-SIDE ADVANTAGE critic with fully separate parameters (DEXENGINE_CRITIC_SPLIT set to 1).

    TwoHead puts both heads on one trunk, so the left value error's gradient also moves the right value estimate.
    Here each of [V_r, V_l] has its own trunk. Both trunks still read all 34 tokens (centralized information); only
    the gradient paths are separated. The checkpoint shape differs from TwoHead, so train and eval must both be built
    with the same DEXENGINE_CRITIC_SPLIT value (the 2-dim value preprocessor is the same).
    """

    def __init__(self, observation_space, action_space, device, slices_combined, size="tiny", clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        cfgs = _TOKEN_SIZES[size]
        self.trunk_r = _TokenTrunk(slices_combined, **cfgs)
        self.trunk_l = _TokenTrunk(slices_combined, **cfgs)
        self.head_r = nn.Linear(cfgs["d_model"], 1)
        self.head_l = nn.Linear(cfgs["d_model"], 1)

    def compute(self, inputs, role=""):
        obs = inputs["states"]
        return torch.cat([self.head_r(self.trunk_r(obs)), self.head_l(self.trunk_l(obs))], dim=-1), {}
