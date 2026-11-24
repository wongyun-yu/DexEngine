"""The (policy, value) factory used by both training and evaluation, so the two build the identical architecture."""

from __future__ import annotations

from dexengine.rl.agents.mlp import (
    ResidualPolicy,
    ResidualValue,
    SmallResidualPolicy,
    SmallResidualValue,
    StructuredResidualPolicy,
    StructuredResidualValue,
)
from dexengine.rl.agents.obs_layout import _structured_obs_slices, _token_obs_slices, _token_slices_bimanual
from dexengine.rl.agents.token_policies import (
    TokenBimanualValue,
    TokenBimanualValueTwoHead,
    TokenBimanualValueTwoTrunk,
    TokenPerHandResidualPolicy,
    TokenResidualPolicy,
    TokenResidualValue,
)
from dexengine.rl.agents.two_tower import (
    PerHandResidualPolicy,
    SmallBimanualValue,
    SmallBimanualValueTwoHead,
    SmallPerHandResidualPolicy,
)


def _build_policy_models(wrapped, raw_env, device, args_cli, log_std_bounds, per_side_value: bool = False):
    """Factory for (policy, value) honoring --policy_arch.

    log_std_bounds = (min_log_std, max_log_std, initial_log_std).
    per_side_value: PER-SIDE ADVANTAGE mode — build the TWO-HEAD critic ([V_r, V_l])
    instead of the scalar one. Only valid for the bimanual token/small archs; train AND eval must pass the same value
    so checkpoint shapes match.
    Returns (policy, value, arch_str). Used by both the train and eval paths so the
    constructed architecture stays identical across train/eval (critical for checkpoint load).
    """
    _aux_action_aware, _aux_decouple, _fastarch_bi, _init_ls, _max_ls, _min_ls, arch, perhand_obs_slices = (
        _import_env_setup_build(args_cli, log_std_bounds, per_side_value)
    )
    if _fastarch_bi:
        arch, policy, value = _build_bimanual_policy_value(
            wrapped,
            raw_env,
            device,
            args_cli,
            per_side_value,
            _aux_action_aware,
            _aux_decouple,
            _init_ls,
            _max_ls,
            _min_ls,
            arch,
            perhand_obs_slices,
        )
        return policy, value, arch
    if arch == "perhand":
        policy = _build_policy_net(
            wrapped,
            raw_env,
            device,
            args_cli,
            _aux_action_aware,
            _aux_decouple,
            _init_ls,
            _max_ls,
            _min_ls,
            perhand_obs_slices,
        )
    elif arch in ("structured", "token"):
        return _build_structured_or_token(
            wrapped, raw_env, device, args_cli, _aux_action_aware, _aux_decouple, _init_ls, _max_ls, _min_ls, arch
        )
    elif arch == "small":
        return _build_small_policy_value(
            wrapped, device, _aux_action_aware, _aux_decouple, _init_ls, _max_ls, _min_ls, arch
        )
    else:
        policy = ResidualPolicy(
            observation_space=wrapped.observation_space,
            action_space=wrapped.action_space,
            device=device,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=_min_ls,
            max_log_std=_max_ls,
            initial_log_std=_init_ls,
            aux_decouple_encoder=_aux_decouple,
            aux_action_aware=_aux_action_aware,
        )
    value = ResidualValue(
        observation_space=wrapped.observation_space,
        action_space=wrapped.action_space,
        device=device,
        clip_actions=False,
    )
    return policy, value, arch


def _build_structured_or_token(
    wrapped, raw_env, device, args_cli, _aux_action_aware, _aux_decouple, _init_ls, _max_ls, _min_ls, arch
):
    """Structured or token one-hand policy and value (obs_v2 one-hand layout; no aux-force heads)."""
    # One-hand structured/token archs: require the obs_v2 one-hand layout; no aux heads.
    if _aux_decouple or _aux_action_aware:
        raise NotImplementedError(f"--policy_arch {arch} does not support aux-force heads")
    if arch == "structured":
        _slices = _structured_obs_slices(raw_env)
        policy = StructuredResidualPolicy(
            observation_space=wrapped.observation_space,
            action_space=wrapped.action_space,
            device=device,
            slices=_slices,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=_min_ls,
            max_log_std=_max_ls,
            initial_log_std=_init_ls,
        )
        value = StructuredResidualValue(
            observation_space=wrapped.observation_space,
            action_space=wrapped.action_space,
            device=device,
            slices=_slices,
            clip_actions=False,
        )
    else:
        _size = getattr(args_cli, "policy_size", "small") or "small"
        _slices = _token_obs_slices(raw_env)
        policy = TokenResidualPolicy(
            observation_space=wrapped.observation_space,
            action_space=wrapped.action_space,
            device=device,
            slices=_slices,
            size=_size,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=_min_ls,
            max_log_std=_max_ls,
            initial_log_std=_init_ls,
        )
        value = TokenResidualValue(
            observation_space=wrapped.observation_space,
            action_space=wrapped.action_space,
            device=device,
            slices=_slices,
            size=_size,
            clip_actions=False,
        )
        arch = f"token_{_size}"
    _np_pol = sum(p.numel() for p in policy.parameters())
    _np_val = sum(p.numel() for p in value.parameters())
    print(f"  [MODEL] {arch}: policy {_np_pol:,} params, value {_np_val:,} params")
    return policy, value, arch


def _build_small_policy_value(wrapped, device, _aux_action_aware, _aux_decouple, _init_ls, _max_ls, _min_ls, arch):
    """The small residual policy and value (no aux-force heads)."""
    if _aux_decouple or _aux_action_aware:
        raise NotImplementedError("--policy_arch small does not support aux-force heads")
    policy = SmallResidualPolicy(
        observation_space=wrapped.observation_space,
        action_space=wrapped.action_space,
        device=device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=_min_ls,
        max_log_std=_max_ls,
        initial_log_std=_init_ls,
    )
    value = SmallResidualValue(
        observation_space=wrapped.observation_space,
        action_space=wrapped.action_space,
        device=device,
        clip_actions=False,
    )
    print(f"  [MODEL] small: policy {sum(p.numel() for p in policy.parameters()):,} params")
    return policy, value, arch


def _build_bimanual_policy_value(
    wrapped,
    raw_env,
    device,
    args_cli,
    per_side_value,
    _aux_action_aware,
    _aux_decouple,
    _init_ls,
    _max_ls,
    _min_ls,
    arch,
    perhand_obs_slices,
):
    """Bimanual policy and value models (the fast two-tower architecture) and their parameter print."""
    arch, policy, value = _build_fastarch_bimanual(
        wrapped,
        raw_env,
        device,
        args_cli,
        per_side_value,
        _aux_action_aware,
        _aux_decouple,
        _init_ls,
        _max_ls,
        _min_ls,
        perhand_obs_slices,
        arch,
    )
    print(
        f"  [MODEL] {arch}: policy {sum(p.numel() for p in policy.parameters()):,} params, "
        f"value {sum(p.numel() for p in value.parameters()):,} params"
    )
    return arch, policy, value


def _build_policy_net(
    wrapped, raw_env, device, args_cli, _aux_action_aware, _aux_decouple, _init_ls, _max_ls, _min_ls, perhand_obs_slices
):
    """The policy network for the one-hand or bimanual action layout."""
    if not bool(getattr(args_cli, "bimanual", 0)):
        raise ValueError("--policy_arch perhand requires --bimanual 1")
    if _aux_decouple or _aux_action_aware:
        raise NotImplementedError("--policy_arch perhand does not support aux-force heads")
    act_dim = wrapped.action_space.shape[0]
    right_idx, left_idx = perhand_obs_slices(raw_env, act_dim // 2)
    policy = PerHandResidualPolicy(
        observation_space=wrapped.observation_space,
        action_space=wrapped.action_space,
        device=device,
        right_idx=right_idx,
        left_idx=left_idx,
        per_hand_act_dim=act_dim // 2,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=_min_ls,
        max_log_std=_max_ls,
        initial_log_std=_init_ls,
    )
    return policy


def _import_env_setup_build(args_cli, log_std_bounds, per_side_value):
    """Lazy: env_setup.build imports Isaac; this module stays importable on CPU for the forward-hash test."""
    from dexengine.rl.env_setup.sides import perhand_obs_slices

    _min_ls, _max_ls, _init_ls = log_std_bounds
    arch = getattr(args_cli, "policy_arch", "shared")
    # Bimanual is always per-hand (two-tower): a shared trunk with a summed gradient couples the two hands'
    # exploration and lets one hand's diverging residual drag the whole policy down. 'shared' (ResidualPolicy) is
    # one-hand only; a bimanual 'shared' run builds the per-hand ResidualPolicy towers (train and eval alike).
    _bimanual = bool(getattr(args_cli, "bimanual", 0))
    # The token/small archs have a native bimanual two-tower path (below); only 'shared' falls back to perhand.
    _fastarch_bi = _bimanual and arch in ("token", "small")
    if _bimanual and not _fastarch_bi:
        arch = "perhand"
    _aux_decouple = bool(getattr(args_cli, "aux_decouple_encoder", 0))
    _aux_action_aware = bool(getattr(args_cli, "aux_action_aware", 0))
    if per_side_value and not _fastarch_bi:
        raise ValueError(
            "per_side_value=True (--per_side_advantage 1) requires the bimanual "
            "token/small two-tower arch (--bimanual 1 --policy_arch token|small)."
        )
    return _aux_action_aware, _aux_decouple, _fastarch_bi, _init_ls, _max_ls, _min_ls, arch, perhand_obs_slices


def _build_fastarch_bimanual(
    wrapped,
    raw_env,
    device,
    args_cli,
    per_side_value,
    _aux_action_aware,
    _aux_decouple,
    _init_ls,
    _max_ls,
    _min_ls,
    perhand_obs_slices,
    arch,
):
    """Bimanual fast-arch (token / small) policy and value, per-side value head when asked."""
    # Bimanual token/small two-tower policy over the obs_v2 bimanual layout.
    if _aux_decouple or _aux_action_aware:
        raise NotImplementedError(f"--policy_arch {arch} (bimanual) does not support aux-force heads")
    act_dim = wrapped.action_space.shape[0]
    if arch == "token":
        _size = getattr(args_cli, "policy_size", "tiny") or "tiny"
        sr, sl, sc = _token_slices_bimanual(raw_env)
        policy = TokenPerHandResidualPolicy(
            observation_space=wrapped.observation_space,
            action_space=wrapped.action_space,
            device=device,
            slices_right=sr,
            slices_left=sl,
            size=_size,
            per_hand_act_dim=act_dim // 2,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=_min_ls,
            max_log_std=_max_ls,
            initial_log_std=_init_ls,
        )
        # PER-SIDE ADVANTAGE: same trunk, 2-dim head [V_r, V_l].
        _val_cls = TokenBimanualValueTwoHead if per_side_value else TokenBimanualValue
        # DEXENGINE_CRITIC_SPLIT=1: two-trunk critic with separate parameters (per-side advantage only).
        import os as _os_cs

        if per_side_value and _os_cs.environ.get("DEXENGINE_CRITIC_SPLIT", "0") == "1":
            _val_cls = TokenBimanualValueTwoTrunk
        value = _val_cls(
            observation_space=wrapped.observation_space,
            action_space=wrapped.action_space,
            device=device,
            slices_combined=sc,
            size=_size,
            clip_actions=False,
        )
        arch = f"token_{_size}_bimanual" + ("_critsplit" if _val_cls is TokenBimanualValueTwoTrunk else "")
    else:  # small
        right_idx, left_idx = perhand_obs_slices(raw_env, act_dim // 2)
        policy = SmallPerHandResidualPolicy(
            observation_space=wrapped.observation_space,
            action_space=wrapped.action_space,
            device=device,
            right_idx=right_idx,
            left_idx=left_idx,
            per_hand_act_dim=act_dim // 2,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=_min_ls,
            max_log_std=_max_ls,
            initial_log_std=_init_ls,
        )
        _val_cls = SmallBimanualValueTwoHead if per_side_value else SmallBimanualValue
        value = _val_cls(
            observation_space=wrapped.observation_space,
            action_space=wrapped.action_space,
            device=device,
            clip_actions=False,
        )
        arch = "small_bimanual"
    if per_side_value:
        print(f"  [MODEL] PER-SIDE ADVANTAGE critic: {type(value).__name__} (2-head [V_r, V_l])")
    return arch, policy, value
