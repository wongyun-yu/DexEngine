"""Per-hand RL training recipes and the finger-penalty plan."""

from __future__ import annotations

import math
from typing import NamedTuple

from dexengine.core.embodiment import default_catalog
from dexengine.core.hands.registry import HAND_REGISTRY
from dexengine.core.hands.spec import WRIST_ACTION_DIMS, HandRLRecipe  # noqa: F401 — HandRLRecipe re-exported

# ===========================================================================
# Per-hand RL training recipes — applied from the hand name alone
# ===========================================================================
# The per-hand **training-method** values live in each hand spec file (configs/robots/hands/<hand>.yaml, key
# rl_recipe — null = no recipe); the loader (dexengine.core.embodiment) reads them into HAND_RL_RECIPES. Hardware
# values (effort/velocity cap scales, gains) are owned by the HandSpec fields and are not repeated here.
# Applied by dexengine/rl/launch/hand_recipe.py (the launcher puts the table values into EXTRA/env), checked at run
# time in create_env, and audited from the logs.
# Switch DEXENGINE_HAND_RECIPE — its default is owned by hand_recipe.SWITCH_DEFAULT (on).
# An explicit 0 = no apply, no check (the caller's command as is).


#: Recipe field → launch channel: (field, "env"|"flag", name, the channel's code default). Order = the order in
#: EXTRA (that of the canonical Tesollo command). ``finger_effort_scale_from_env`` owns reading the env default.
RECIPE_CHANNELS: tuple[tuple[str, str, str, float | None], ...] = (
    ("finger_effort_scale", "env", "DEXENGINE_FINGER_EFFORT_SCALE", 1.0),
    ("finger_effort", "flag", "--finger_effort", None),
    ("finger_effort_anneal_start", "flag", "--finger_effort_anneal_start", None),
    ("imit_release_min", "flag", "--imit_release_min", None),
    ("imit_release_start", "flag", "--imit_release_start", None),
)
_RECIPE_CHANNEL = {f: (kind, name, default) for f, kind, name, default in RECIPE_CHANNELS}
if tuple(_RECIPE_CHANNEL) != tuple(HandRLRecipe.__dataclass_fields__):
    raise RuntimeError(
        f"RECIPE_CHANNELS fields {tuple(_RECIPE_CHANNEL)} ≠ HandRLRecipe fields "
        f"{tuple(HandRLRecipe.__dataclass_fields__)} — one field = one channel"
    )


def finger_effort_scale_from_env(environ=None) -> float:
    """The **only reader** of DEXENGINE_FINGER_EFFORT_SCALE (unset = the RECIPE_CHANNELS default 1.0).
    The finger actuators, the effort-curriculum rule and the recipe check call it."""
    import os

    _kind, name, default = _RECIPE_CHANNEL["finger_effort_scale"]
    raw = (os.environ if environ is None else environ).get(name)
    return float(default) if raw is None else float(raw)


#: Hand name → recipe, from the hand spec files (every hand file states rl_recipe, so every hand is a key).
#: None = no recipe — nothing is applied or checked even with the switch on (the command as is).
HAND_RL_RECIPES: dict[str, HandRLRecipe | None] = dict(default_catalog().recipes)


def rl_recipe_for(hand: str) -> HandRLRecipe | None:
    """Recipe of hand ``hand`` (None = no recipe). An unregistered hand raises."""
    if hand not in HAND_RL_RECIPES:
        raise KeyError(f"rl_recipe_for: hand {hand!r} not in HAND_RL_RECIPES (registered: {sorted(HAND_RL_RECIPES)})")
    return HAND_RL_RECIPES[hand]


# ---------------------------------------------------------------------------
# Action span, weight scale and deadzone scale of the finger residual-L2 and finger action-rate penalties
# ---------------------------------------------------------------------------
#: Legacy span = a literal from the 12-d (RH56F1) action. It covers only the first 6 of 20–22 fingers. Not
#: canonical — used **only when named** with ``--finger_penalty_span legacy`` (to reproduce or resume runs from
#: before the derived span).
LEGACY_FINGER_PENALTY_DIMS = (6, 12)
#: Reference finger count of derived_norm = the count the legacy span covered. Per side weight × this / n,
#: deadzone × √(n / this).
FINGER_PENALTY_REF_N = LEGACY_FINGER_PENALTY_DIMS[1] - LEGACY_FINGER_PENALTY_DIMS[0]
#: Values of --finger_penalty_span. derived_norm = every finger of the side ``(6, 6+n)``, weight × REF_N/n,
#: residual-L2 deadzone × √(n/REF_N) (total size at the legacy level — see finger_penalty_plan). legacy = the
#: legacy span, weight and deadzone × 1 (explicit choice).
FINGER_PENALTY_SPANS = ("derived_norm", "legacy")
#: Retired name → reason. Passing one fails in the argparse choices, or in resolve/plan for callers that skip the
#: parser. Never revive one, so a name never has two meanings. A retired span in a ledger cannot be resumed.
RETIRED_FINGER_PENALTY_SPANS = {
    "derived": "derived span without scaling — weight × 1 on every finger makes the total size n/6 times larger",
    "derived_scaled": "weight × 6/n only, deadzone unchanged — the deadzone sits on an n-dim norm, so the early "
    "residual L2 is ≈2.07× the legacy span. Never used in a real run → replaced by "
    "derived_norm",
}
#: The **only owner** of the canonical default (every finger, total size at the legacy (6,12) level: weight
#: × REF_N/n, deadzone × √(n/REF_N)). argparse defaults to None and the training path resolves it to this value, so
#: the markers can tell explicit from default.
CANON_FINGER_PENALTY_SPAN = "derived_norm"


class FingerPenaltyPlan(NamedTuple):
    """Finger-penalty plan of one side. ``dims`` = action span ``(lo, hi)``, ``w_scale`` = weight scale of both terms
    (residual L2, action rate), ``dz_scale`` = residual-L2 norm deadzone scale (the rate term has no deadzone)."""

    dims: tuple[int, int]
    w_scale: float
    dz_scale: float


def _retired_or_unknown(where: str, span) -> ValueError:
    if str(span) in RETIRED_FINGER_PENALTY_SPANS:
        return ValueError(
            f"{where}: {span!r} is a retired finger-penalty span — "
            f"{RETIRED_FINGER_PENALTY_SPANS[str(span)]}. Current values: {FINGER_PENALTY_SPANS}"
        )
    return ValueError(f"{where}: {span!r} — must be one of {FINGER_PENALTY_SPANS}")


#: source word resolve_finger_penalty_span returns for an explicitly given span (its readers compare it)
SPAN_GIVEN = "explicit"


def resolve_finger_penalty_span(given) -> tuple[str, str]:
    """The **only owner** of resolving ``--finger_penalty_span`` — ``(span, source)``. An explicit value gives
    ``(value, SPAN_GIVEN)``, ``None`` gives ``(CANON_FINGER_PENALTY_SPAN, "canonical default")``;
    the explicit-source word is compared by its readers. The training marker, the eval marker and the checkpoint
    ledger all call this. An unknown or retired value raises (the same rule for callers that skip the argparse
    choices)."""
    if given is None:
        return CANON_FINGER_PENALTY_SPAN, "canonical default"
    if str(given) not in FINGER_PENALTY_SPANS:
        raise _retired_or_unknown("resolve_finger_penalty_span", given)
    return str(given), SPAN_GIVEN


def finger_penalty_plan(hand: str, span: str, n_term: int) -> FingerPenaltyPlan:
    """Plan of the two finger-penalty terms of one side, ``FingerPenaltyPlan(dims, w_scale, dz_scale)`` — the
    **only owner** of the span and the scales.

    ``n_term`` = finger dimensions of that side's residual action term (``len(term.finger_joint_names)``, per side).
    ``derived_norm`` = ``(WRIST_ACTION_DIMS, WRIST_ACTION_DIMS + n)``, weight ``REF_N / n``,
    deadzone ``√(n / REF_N)``. Both scales are needed to keep the legacy total size: the rate term (a plain sum of
    squares) matches with the weight alone under iid, uniform σ; the residual L2
    ``w·(REF_N/n)·relu(‖r_n‖ − dz·√(n/REF_N))² ≡ w·relu(√(REF_N/n)·‖r_n‖ − dz)²`` equals the legacy formula
    applied to "√REF_N × per-dim RMS" (with the deadzone unchanged the norm grows with n and the deadzone almost never
    engages).
    n must equal HandSpec[hand].n_fingers (raises when the action term and the registry disagree).
    ``legacy`` = ``LEGACY_FINGER_PENALTY_DIMS``, 1.0, 1.0 (needs n ≥ REF_N — the legacy span is never silently cut
    on a shorter action).
    An unknown or retired span, an unregistered hand, or an n the rule cannot apply to raises.
    """
    spec = HAND_REGISTRY.get(hand)
    if spec is None:
        raise ValueError(f"finger_penalty_plan: hand {hand!r} is not in HAND_REGISTRY")
    n = int(n_term)
    if span == "legacy":
        if n < FINGER_PENALTY_REF_N:
            raise ValueError(f"finger_penalty_plan: legacy span {LEGACY_FINGER_PENALTY_DIMS} but {n} fingers")
        return FingerPenaltyPlan(LEGACY_FINGER_PENALTY_DIMS, 1.0, 1.0)
    if span == "derived_norm":
        if n <= 0 or n != spec.n_fingers:
            raise ValueError(
                f"finger_penalty_plan: action term has {n} fingers ≠ HandSpec[{hand}].n_fingers "
                f"{spec.n_fingers} — cannot build the derived span"
            )
        return FingerPenaltyPlan(
            (WRIST_ACTION_DIMS, WRIST_ACTION_DIMS + n), FINGER_PENALTY_REF_N / n, math.sqrt(n / FINGER_PENALTY_REF_N)
        )
    raise _retired_or_unknown("finger_penalty_plan", span)
