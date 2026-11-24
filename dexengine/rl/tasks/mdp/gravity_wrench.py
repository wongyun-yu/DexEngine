"""Timing owner of the object gravity-substitute wrench: the pre-physics event mode (MODE), its application and its
measurement.

Objects have disable_gravity=True and `ramp_object_gravity` applies gravity instead as the object's permanent wrench (a
body-frame force). That force must be computed for **the pose physics will start from**. So the term's mode is MODE, not
interval, and the env calls `apply` once per step just before physics (before super().step in `EvalEnv.step`, after
every pose write such as the judge park, SPAWN_HOLD and per-side snaps). The interval event (end of step) missed three
cases: the first step after a gym reset (`RigidObject.reset` clears the wrench to 0 and physics starts without an
interval event, i.e. without gravity), pose writes between the end of a step and the next physics step (a pinned env's
'gravity' tilted by one step of rotation), and the body-frame transform pose (the function side builds it from the
current pose directly).
"""

from __future__ import annotations

MODE = "prephysics"  # event mode name: the term (cfg) and the caller (env) both read this one constant


def apply(env) -> None:
    """Run the pre-physics event terms. An env without the mode (a diagnostic branch with the gravity term removed)
    prints a marker once and returns."""
    on = getattr(env, "_prephysics_on", None)
    if on is None:
        em = env.event_manager
        on = env._prephysics_on = MODE in em.available_modes
        names = em.active_terms.get(MODE, []) if on else []
        print(
            f"[GRAVITY_WRENCH] applied before physics — mode {MODE} · terms {names}"
            if on
            else f"[GRAVITY_WRENCH] no pre-physics term — this env has no mode {MODE} term",
            flush=True,
        )
    if on:
        env.event_manager.apply(mode=MODE)


def ramp_term_cfgs(event_manager) -> list:
    """Cfgs of the event terms with curriculum ramp parameters, in their two modes (interval, MODE).

    The judge (frozen eval) uses this to find them when it pins the gravity and friction ramps at 100% and restores
    them."""
    return [*event_manager._mode_term_cfgs.get("interval", []), *event_manager._mode_term_cfgs.get(MODE, [])]
