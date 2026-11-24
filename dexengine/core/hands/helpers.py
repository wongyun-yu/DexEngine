"""Derived per-joint values of a HandSpec and hand link-name helpers."""

from __future__ import annotations

import re

from dexengine.core.hands.spec import HYPEREXT_ROLES, HandSpec

# ===========================================================================
# Left/right mirror rule
# ===========================================================================


def mirror_right_to_left(name: str) -> str:
    """``right_*`` -> ``left_*`` for joint/body names — the ONE mirror rule.

    Every integrated hand (RH56F1, SharpA) and both arms spell
    sides with a leading ``right_``/``left_`` token somewhere in the name, and
    only the FIRST occurrence is the side. Specs store right-side names only;
    left-side actuator groups, contact regexes and action kwargs all derive
    through this function so the two sides cannot drift apart.
    """
    return name.replace("right_", "left_", 1)


def finger_velocity_applied(spec: HandSpec) -> dict[str, float] | float:
    """The joint-speed limit the sim applies: ``finger_velocity`` × ``finger_velocity_peak_scale``, same shape.
    **The one reader** of that scale — the finger actuator, the fingertip-force oracle and the data pipeline (retiming
    stretches the reference to this limit, the gate measures against it, f0 repair sizes its pads by it — all through
    ``finger_velocity_vector``) read the same value, so a reference the data pipeline calls feasible is feasible in
    the sim."""
    s = float(spec.finger_velocity_peak_scale)
    ov = dict(spec.finger_velocity_peak_scale_overrides)
    if isinstance(spec.finger_velocity, dict):
        unknown = set(ov) - set(spec.finger_velocity)
        if unknown:
            raise ValueError(
                f"{spec.name}: finger_velocity_peak_scale_overrides keys {sorted(unknown)} are not in finger_velocity"
            )
        return {k: float(v) * float(ov.get(k, s)) for k, v in spec.finger_velocity.items()}
    if ov:
        raise ValueError(f"{spec.name}: a scalar finger_velocity cannot take per-joint scales")
    return float(spec.finger_velocity) * s


def per_joint(value, names) -> tuple[float, ...]:
    """A scalar or a per-joint regex dict → one float per joint name, in ``names`` order. Each name must **fullmatch
    exactly one key** (the IsaacLab ``resolve_matching_names_values`` semantics the actuator config gets — no first-
    match, no default): zero or several keys raise ValueError."""
    if not isinstance(value, dict):
        return tuple(float(value) for _ in names)
    out = []
    for n in names:
        hit = [float(v) for k, v in value.items() if re.fullmatch(k, n)]
        if len(hit) != 1:
            raise ValueError(f"joint {n}: {len(hit)} regex keys match (must be exactly one) — keys {sorted(value)}")
        out.append(hit[0])
    return tuple(out)


def finger_velocity_vector(spec: HandSpec) -> tuple[float, ...]:
    """The applied finger speed limit (rad/s) per joint of ``finger_joint_names_right`` (the tree's finger-block
    order; left uses the same order) — the vector retiming, the gate and f0 repair budget with."""
    return per_joint(finger_velocity_applied(spec), spec.finger_joint_names_right)


def finger_effort_applied(spec: HandSpec) -> dict[str, float] | float:
    """The actuator cap the sim applies: ``finger_effort`` × ``finger_effort_peak_scale``, same shape (scalar or
    per-joint regex dict). **The one reader** of the scale — the finger actuator (dexengine.rl.tasks.scene.robot) and
    the fingertip-force oracle both call it, so the cap the RL env enforces is the cap the oracle measured."""
    s = float(spec.finger_effort_peak_scale)
    if isinstance(spec.finger_effort, dict):
        return {k: float(v) * s for k, v in spec.finger_effort.items()}
    return float(spec.finger_effort) * s


def fingertip_site_offsets_for(spec: HandSpec, side: str) -> tuple[tuple[float, float, float], ...]:
    """Resolve ``spec.fingertip_site_offsets`` into tip-body-local 3-vectors ``(n_tips, 3)`` for ``side``.

    **The one interpreter** — the Isaac training env's naming helpers and
    `dexengine/robots/hand_rig.py` (the MJCF rig) call it. Rules:

    * An empty table means all zeros = the tip body origin.
    * Two shapes: ``{"thumb": d, "fingers": d}`` or one key per ``tip_body_names_right``.
      Only the second shape may add **left-body-name keys** (``mirror_right_to_left(tip)``);
      with ``side == "left"`` such a key wins over the right key (see the field docstring, "left" part).
    * A value is a scalar (tip-local +x, the SharpA convention) or a 3-vector.
    * Unknown keys, partial tables and bad values **raise** — so a missing finger never silently measures from the
      body origin.
    """
    if side not in ("right", "left"):
        raise ValueError(f"fingertip_site_offsets_for: side={side!r} (right|left)")
    table = spec.fingertip_site_offsets
    tips = tuple(spec.tip_body_names_right)
    if not table:
        return ((0.0, 0.0, 0.0),) * len(tips)
    keys = set(table)
    if keys == {"thumb", "fingers"}:
        picked = [table["thumb" if i == 0 else "fingers"] for i in range(len(tips))]
    else:
        left_keys = {mirror_right_to_left(t) for t in tips}
        unknown = keys - set(tips) - left_keys
        if not keys >= set(tips) or unknown:
            raise ValueError(
                f"dexengine.core.hands.{spec.name}.fingertip_site_offsets has keys {sorted(keys)}. "
                f"Use either {{'thumb', 'fingers'}} or one key per tip body "
                f"({list(tips)}), optionally plus left-body keys ({sorted(left_keys)}). "
                f"Unknown keys {sorted(unknown)}; partial tables are refused because a "
                "missing finger would silently measure from the body origin."
            )
        picked = []
        for t in tips:
            lk = mirror_right_to_left(t)
            picked.append(table[lk] if side == "left" and lk in table else table[t])
    out = []
    for v in picked:
        if isinstance(v, (int, float)):
            out.append((float(v), 0.0, 0.0))  # scalar => tip-local +x
            continue
        vec = tuple(float(x) for x in v)
        if len(vec) != 3:
            raise ValueError(
                f"dexengine.core.hands.{spec.name}.fingertip_site_offsets: expected a scalar or a 3-vector, got {v!r}."
            )
        out.append(vec)
    return tuple(out)


def strip_side(name: str) -> str:
    """Strip a `right_`/`left_` prefix (unchanged if there is none)."""
    for pre in ("right_", "left_"):
        if name.startswith(pre):
            return name[len(pre) :]
    return name


def finger_column(tips, hand: str = "") -> tuple[int, tuple[str, ...]]:
    """From the tip names, **the position of the `_`-separated field that tells the fingers apart** and its values —
    the one owner of the link → finger assignment rule.

    No naming scheme (`finger1_…` / `thumb_…`) is assumed: split the five side-stripped tip names on `_`; the first
    field whose **values all differ** is the finger field. DG-5F-S `link_1_4` → field 1 · XHand
    `hand_thumb_dip_link` → field 1 · SharpA `thumb_DP` → field 0. A link belongs to a finger when its value in that
    field is that finger's token (otherwise it is outside the fingers — palm etc.).
    If no field separates them, this raises instead of passing silently. A second rule elsewhere (e.g. "the part
    before the tip name's last `_`") would group only the dip links on XHand — so every reader uses this one.
    """
    parts = [strip_side(t).split("_") for t in tips]
    for i in range(min(len(p) for p in parts)):
        col = tuple(p[i] for p in parts)
        if len(set(col)) == len(col):
            return i, col
    raise ValueError(
        f"{hand}: the tip names do not tell the fingers apart {tuple(tips)} — no `_` field has five distinct values"
    )


def sim_hyperext_allow_deg(hand: str) -> dict[str, float] | None:
    """The sim body's hyperextension allowance (HandSpec.sim_finger_hyperext_allow_deg, degrees per flexion role) —
    **the one reader** of that field. ``hand`` = hand id or its retargeting spec directory (``paths.hand_id`` maps both
    ways and dies on an unregistered name). ``None`` = not stated (the sim cut stays at neutral). Caller: the cut window
    with ``--sim`` (dexengine.retarget.ik.hyperext_limits), which truncate.py runs for the sim asset."""
    from dexengine.core.hands.registry import HAND_REGISTRY  # noqa: PLC0415 — the registry imports this module
    from dexengine.core.locations.hands import hand_id  # noqa: PLC0415 — locations loads the registry

    allow = HAND_REGISTRY[hand_id(hand)].sim_finger_hyperext_allow_deg
    return None if allow is None else {k: float(allow[k]) for k in HYPEREXT_ROLES}
