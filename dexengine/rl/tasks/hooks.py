"""Extension points for experiment variants (stdlib only).

The release code never imports ``experiments``. A variant with a different structure (the floating-hand body, for
example) lives in a plugin module that registers its objects here when it is imported. The launcher names the
plugins: ``load_plugins()`` imports every module listed in the switch ``plugins`` (``DEXENGINE_PLUGINS``,
comma-separated dotted module names; ``--set plugins=…`` on the launcher, carried in the switches file). It must be
set before the env cfgs are resolved (``dexengine.rl.tasks.registry`` loads the plugins when it registers the gym ids).

Hook names
  ``floating_hand``  the floating-hand body module. Needed when ``DEXENGINE_ARM`` names the floating arm. It provides
                     ``active()``, ``is_floating(arm)``, ``data_world_arm(arm)``, ``nan_reset()``,
                     ``require_no_arm_gains(...)``, ``require_supported(hand)``, ``joint_drive()``, ``root_body(hand,
                     side)``, ``source_urdf(hand)``, the ``vj_*`` virtual-joint helpers, ``WRIST_GAINS``, ``MARK``,
                     ``DATA_WORLD_ARM`` and ``reset_floating_hands_to_trajectory``.
  ``floating_hand.action``  the inner action class that replaces the arm-IK + finger action on the floating body.
  ``objtrace``       the object-pose trace recorder of a judge run (``DEXENGINE_E123_OBJTRACE``): ``ENV_VAR``,
                     ``launch_problems(...)`` and ``recorder_class(cls, path)``; read by ``dexengine.rl.train.startup``.
"""

from __future__ import annotations

import importlib
from typing import Any

from dexengine.rl.tasks import env_flags

PLUGINS_ENV = "DEXENGINE_PLUGINS"
FLOATING_HAND = "floating_hand"
FLOATING_HAND_ACTION = "floating_hand.action"
OBJTRACE = "objtrace"

_REGISTRY: dict[str, Any] = {}
_LOADED: list[str] = []


def register(name: str, obj: Any) -> None:
    """Register ``obj`` under ``name``. Registering a different object under a taken name is an error."""
    prev = _REGISTRY.get(name)
    if prev is not None and prev is not obj:
        raise RuntimeError(f"[HOOKS] {name!r} is already registered by {prev!r}")
    _REGISTRY[name] = obj


def get(name: str, default: Any = None) -> Any:
    """The object registered under ``name`` (``default`` when nothing is registered)."""
    return _REGISTRY.get(name, default)


def load_plugins(modules: str | None = None) -> list[str]:
    """Import the plugin modules (default: ``DEXENGINE_PLUGINS``); each one registers its hooks on import."""
    raw = env_flags.get(PLUGINS_ENV) if modules is None else modules
    for mod in (m.strip() for m in raw.split(",")):
        if mod and mod not in _LOADED:
            importlib.import_module(mod)
            _LOADED.append(mod)
            print(f"[HOOKS] plugin loaded: {mod}", flush=True)
    return list(_LOADED)


def _active_arm() -> str:
    return env_flags.get("DEXENGINE_ARM").strip()


def floating_hand():
    """The floating-hand plugin, or None when no plugin registered it.

    Asking for the floating arm (``DEXENGINE_ARM``) without the plugin is an error.
    """
    fh = _REGISTRY.get(FLOATING_HAND)
    if fh is None:
        from dexengine.core.arms import FLOATING_ARM  # stdlib-only registry

        if _active_arm() == FLOATING_ARM:
            raise RuntimeError(
                f"[HOOKS] DEXENGINE_ARM={_active_arm()!r} needs the floating-hand plugin: "
                f"set {PLUGINS_ENV} to the module that registers {FLOATING_HAND!r}"
            )
    return fh


def floating_active() -> bool:
    """True when the run uses the floating-hand body (the plugin's ``active()``); False without the plugin."""
    fh = floating_hand()
    return bool(fh is not None and fh.active())


#: Start of the line printed when the reward-NaN reset is armed (cfg/arm_ik); a run log that carries it shows the
#: switch took effect.
NAN_RESET_ON = "[FH_NAN_RESET] on —"


def floating_nan_reset() -> bool:
    """The floating-hand reward-NaN reset switch (``DEXENGINE_FH_NAN_RESET``); it has no consumer on an arm body."""
    fh = floating_hand()
    if fh is not None:
        return bool(fh.nan_reset())
    raw = env_flags.get("DEXENGINE_FH_NAN_RESET")
    if raw not in ("", "0"):
        raise RuntimeError(f"[FLOATING_HAND] DEXENGINE_FH_NAN_RESET={raw!r} applies to the floating-hand body only")
    return False


def floating_palm_point(asset, side: str):
    """Palm-point offset of the floating hand (palm body frame, m), or None (``DEXENGINE_FH_PALM_POINT``).

    The floating-hand plugin owns the env read and its check (``palm_point_mode``); the offset math is
    ``mdp.naming.fh_palm_point_offset``. Without the plugin the switch has no consumer, so a set value fails.
    """
    fh = floating_hand()
    if fh is not None:
        from dexengine.rl.tasks.mdp.naming import fh_palm_point_offset  # (Isaac-free module, lazy)

        return fh_palm_point_offset(asset, side)
    raw = env_flags.get("DEXENGINE_FH_PALM_POINT").strip()
    if raw:
        raise RuntimeError(f"[FLOATING_HAND] DEXENGINE_FH_PALM_POINT={raw!r} applies to the floating-hand body only")
    return None
