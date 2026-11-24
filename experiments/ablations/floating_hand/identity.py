"""Floating hand: the active-process check, the run preconditions, the marker line and the arm-gain rule.

The identity rules (arm name, supported hands, root body, mounts, data world) are dexengine.core.floating's and are
re-exported here. Loads without isaaclab and without torch. The import of dexengine.rl.tasks.scene.split_artic stays
inside require_supported: split_artic reaches this package through the hooks.
"""

from __future__ import annotations

import os

# the identity rules of the floating arm are owned by dexengine.core.floating (the release registry knows the arm) —
# re-exported here, the import surface of this package
from dexengine.core.floating import (  # noqa: F401
    DATA_WORLD_ARM,
    FLOATING_ARM,
    MARK,
    SUPPORTED_HANDS,
    data_world_arm,
    identity_mounts,
    is_floating,
    root_body,
    source_urdf,
)
from experiments.ablations.floating_hand.wrench import DAMP_INERTIA_RULE, MASS_RULE, WRIST_GAINS


def active() -> bool:
    """Is this process the floating hand (``DEXENGINE_ARM`` — dexengine.core.arms dies when unset)?"""
    from dexengine.core.arms import active_arm_name  # noqa: PLC0415

    return is_floating(active_arm_name())


def require_supported(hand_name: str) -> None:
    """Preconditions of the floating-hand path — dies when one does not hold."""
    if hand_name not in SUPPORTED_HANDS:
        raise RuntimeError(
            f"{MARK} hand {hand_name!r} has no floating body (supported: {SUPPORTED_HANDS}). Generate its "
            "per-side URDF "
            "(experiments/ablations/floating_hand/make_floating_urdf.py), bake it, register the mounts, then add it to "
            "SUPPORTED_HANDS."
        )
    from dexengine.rl.tasks.scene.split_artic import (  # noqa: PLC0415 — split_artic imports this module
        BI_ENV,
        bi_enabled,
    )

    if not bi_enabled():
        raise RuntimeError(
            f"{MARK} the floating hand is one free-root articulation per hand — it needs {BI_ENV}=1 (two "
            "articulations); one articulation cannot hold two free roots."
        )
    from dexengine.rl.tasks import switches  # noqa: PLC0415 — stdlib only

    if switches.text("DEXENGINE_RIGHT_PARK") == "1" or os.environ.get("DEXENGINE_LEFT_KINEMATIC", "0") == "1":
        raise RuntimeError(
            f"{MARK} DEXENGINE_RIGHT_PARK / DEXENGINE_LEFT_KINEMATIC belong to the arm-joint path — they "
            "mean nothing for the floating hand (not ignored silently)."
        )


def describe() -> str:
    """One marker line: the body, the wrist wrench law, the gains and the inertia rules."""
    g = WRIST_GAINS
    return (
        f"{MARK} body = one free-root articulation per hand (fix_base False · whole hand without gravity) · "
        "wrist = external wrench (no IK, no arm PD) · "
        f"F = clip(kp·e + implicit damping, ±{g.lin_force_max:g} N) kp {g.lin_kp:g} N/m kd {g.lin_kd:g} · "
        f"τ = clip(kp·rotvec, ±{g.rot_torque_max:g} N·m) kp {g.rot_kp:g} + implicit damping kd "
        f"{g.rot_kd:g} N·m·s/rad · "
        f"rotational damping inertia = {DAMP_INERTIA_RULE} · linear mass = {MASS_RULE} (root body own · lower bound) · "
        "load observer LoadObserver (last step's velocity change → unknown load d in the implicit solve · wrench "
        "at the centre of mass)"
    )


# ── arm gains (arm joint PD — no consumer on the floating hand)
# ─────────────────────────────────────────────────────
#: env inputs that carry arm joint PD gains (env setup diagnostics · left-arm gain randomisation) — an empty value is
#: not an input. DEXENGINE_LEFT_ARM_KP_SCALE = 1.0 is a no-op (the bimanual launcher sets it on every launch) and does
#: not count.
ARM_GAIN_ENVS = ("DEXENGINE_LEFT_ARM_KP_SCALE", "DEXENGINE_LEFT_ARM_KP_DR", "DEXENGINE_ARM_KP", "DEXENGINE_ARM_KD")


def arm_gain_inputs(given_flags, environ) -> list[str]:
    """The arm gain inputs present. ``given_flags`` = the arm gain flags the parser received EXPLICITLY
    (``args_cli.arm_gains_given`` — an explicit default still counts), ``environ`` = the process env."""
    out = list(given_flags)
    for k in ARM_GAIN_ENVS:
        v = str(environ.get(k, "") or "").strip()
        if not v:
            continue
        if k == "DEXENGINE_LEFT_ARM_KP_SCALE":
            try:
                if abs(float(v) - 1.0) <= 1e-9:
                    continue
            except ValueError:
                pass  # not a number → counts as an input → dies below
        out.append(f"{k}={v}")
    return out


def require_no_arm_gains(given_flags, environ, ctx: str, actuator_keys=()) -> str:
    """The floating hand's arm-gain rule (training and evaluation; called by the env setup's arm-gain override).

    The floating hand has no arm joints — the wrist moves by the wrench (``WRIST_GAINS``). An arm gain input therefore
    has no consumer: one present dies instead of being dropped. Otherwise the marker line "applied 0" is returned (the
    caller prints it). ``actuator_keys`` = the articulations' actuator group names — an arm group (``*_arm`` ·
    ``*_arm_left``) means the body is not the floating hand (dexengine.rl.tasks.scene.robot builds no arm actuator for
    it) → dies."""
    arm_groups = sorted(k for k in actuator_keys if k.endswith(("_arm", "_arm_left")))
    if arm_groups:
        raise SystemExit(
            f"{MARK} [{ctx}] floating hand with arm actuator groups {arm_groups} — dexengine.rl.tasks.scene.robot "
            "builds no arm actuator for the floating hand (wrong body)."
        )
    given = arm_gain_inputs(given_flags, environ)
    if given:
        raise SystemExit(
            f"{MARK} [{ctx}] arm gain inputs {given} have no consumer on the floating hand — there is no "
            "arm joint PD and the wrist gains are floating_hand.WRIST_GAINS. The launcher loads no arm gains "
            "for this arm (canonical table ARM_JOINT_PD=0 · paths.py --what arm-joint-pd). Remove the inputs."
        )
    g = WRIST_GAINS
    return (
        f"{MARK} [{ctx}] arm gains: no arm joint PD — arm gain inputs 0 · applied 0 · wrist gains = "
        f"floating_hand.WRIST_GAINS (linear kp {g.lin_kp:g} kd {g.lin_kd:g} · rotational kp {g.rot_kp:g} "
        f"kd {g.rot_kd:g})"
    )
