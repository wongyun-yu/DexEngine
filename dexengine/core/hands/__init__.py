"""Hand specifications.

One frozen `HandSpec` per hand collects every hand-dependent constant of the pipeline (joint lists, contact
bodies, actuator limits and gains, dataset qpos width). Plain dataclasses, no Isaac imports.
`HAND_REGISTRY` maps the hand id to its spec; the values live in the hand spec files (configs/robots/hands/<hand>.yaml,
read by dexengine.core.embodiment), `recipes` holds the per-hand RL training recipes read from the same files.
"""

from __future__ import annotations

from dexengine.core.hands.helpers import (  # noqa: F401
    finger_column,
    finger_effort_applied,
    finger_velocity_applied,
    finger_velocity_vector,
    fingertip_site_offsets_for,
    mirror_right_to_left,
    sim_hyperext_allow_deg,
    strip_side,
)
from dexengine.core.hands.recipes import (  # noqa: F401
    CANON_FINGER_PENALTY_SPAN,
    FINGER_PENALTY_REF_N,
    FINGER_PENALTY_SPANS,
    HAND_RL_RECIPES,
    LEGACY_FINGER_PENALTY_DIMS,
    RECIPE_CHANNELS,
    RETIRED_FINGER_PENALTY_SPANS,
    SPAN_GIVEN,
    FingerPenaltyPlan,
    finger_effort_scale_from_env,
    finger_penalty_plan,
    resolve_finger_penalty_span,
    rl_recipe_for,
)
from dexengine.core.hands.registry import (  # noqa: F401
    HAND_REGISTRY,
)
from dexengine.core.hands.spec import HandSpec  # noqa: F401

#: The SharpA spec under its historical name (rl.tasks.mdp.naming reads its finger joint suffixes).
SHARPA_WAVE_RIGHT = HAND_REGISTRY["sharpa_wave"]
