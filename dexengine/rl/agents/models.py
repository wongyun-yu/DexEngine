"""Policy/value models and observation-slice helpers — one import path for the submodules of ``dexengine.rl.agents``.

``mlp`` (one-hand MLP policies) · ``token_policies`` (token-transformer policies, two-tower token policies, the
two-policy render) · ``two_tower`` (bimanual MLP towers) · ``obs_layout`` (observation slices and the checkpoint
width check) · ``factory`` (``_build_policy_models``). Pure torch + skrl, importable without Isaac. Class and
attribute names are part of the checkpoint format.
"""

from __future__ import annotations

from dexengine.rl.agents.factory import _build_policy_models
from dexengine.rl.agents.mlp import (
    ResidualPolicy,
    ResidualValue,
    SmallResidualPolicy,
    SmallResidualValue,
    StructuredResidualPolicy,
    StructuredResidualValue,
)
from dexengine.rl.agents.obs_layout import (
    ARTIC_OBJ_EXTRA,
    FIXED_OBJ_EXTRA,
    _hand_block_geometry,
    _object_extra_dims,
    _structured_obs_slices,
    _term_starts,
    _token_dict,
    _token_obs_slices,
    _token_slices_bimanual,
    assert_checkpoint_obs_dim,
    checkpoint_obs_dim,
    obs_width_remedy,
)
from dexengine.rl.agents.token_policies import (
    _TOKEN_SIZES,
    TokenBimanualValue,
    TokenBimanualValueTwoHead,
    TokenBimanualValueTwoTrunk,
    TokenPerHandResidualPolicy,
    TokenResidualPolicy,
    TokenResidualValue,
    _build_dual_policy_fn,
    _TokenBlock,
    _TokenTrunk,
    _trunk_update,
    _trunk_update_path,
)
from dexengine.rl.agents.two_tower import (
    PerHandResidualPolicy,
    SmallBimanualValue,
    SmallBimanualValueTwoHead,
    SmallPerHandResidualPolicy,
    _small_tower,
)

__all__ = [
    "ResidualPolicy",
    "ResidualValue",
    "StructuredResidualPolicy",
    "StructuredResidualValue",
    "SmallResidualPolicy",
    "SmallResidualValue",
    "_structured_obs_slices",
    "_term_starts",
    "_hand_block_geometry",
    "ARTIC_OBJ_EXTRA",
    "FIXED_OBJ_EXTRA",
    "_object_extra_dims",
    "checkpoint_obs_dim",
    "assert_checkpoint_obs_dim",
    "obs_width_remedy",
    "_token_dict",
    "_token_obs_slices",
    "_token_slices_bimanual",
    "_TOKEN_SIZES",
    "_TokenBlock",
    "_TokenTrunk",
    "_trunk_update_path",
    "_trunk_update",
    "TokenResidualPolicy",
    "TokenResidualValue",
    "_build_dual_policy_fn",
    "TokenPerHandResidualPolicy",
    "TokenBimanualValue",
    "TokenBimanualValueTwoHead",
    "TokenBimanualValueTwoTrunk",
    "_small_tower",
    "SmallPerHandResidualPolicy",
    "SmallBimanualValue",
    "SmallBimanualValueTwoHead",
    "PerHandResidualPolicy",
    "_build_policy_models",
]
