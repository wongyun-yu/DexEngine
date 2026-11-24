"""Extension points for plugin code outside this package (experiment variants, a deployment's own operations code).

The release never imports experiment code. A plugin is a module that calls ``register`` at import time.
Stage-4 work runs in fresh processes (Isaac sessions, render workers, augment children), so a caller names
its plugin modules in the environment variable ``DEXENGINE_RECORD_PLUGINS`` (comma-separated module names).
Every record entry point calls ``load_plugins()`` before it dispatches; child processes inherit the variable.

Registry kinds (``register(kind, name, obj)``):
  session_stage  an extra ``--stage`` of ``dexengine.record.session.entry`` (a ``SessionStage``)
  bake           a replacement for the augment bake step, chosen by ``augment.place_gen --bake-from``
  aug_session    a factory ``f() -> hook | None`` that an augment session calls once at start
  driver_spec    the job-spec provider of ``dexengine.record.driver --spec`` (name "lane"): a module with
                 ``pilot_spec(args) -> dict``, ``aug_slot(k) -> str``, ``CCOL.spec_sha(row) -> str``, ``die(msg, rc)``
                 and ``RC_INPUT``
  owner_moves    a tuple of ``dexengine.placement.judge.owner_moves.Move`` rows that keep a deployment's stored judge
                 records comparable (read by ``owner_moves.moves``, which loads the plugins itself)
  persist_job    a callable ``f(job_tag) -> str`` that the Kit-reuse worker (``session.persist_kit.render_clips``)
                 calls at each job start, after the job's log span begins; a non-empty return is printed as one line
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

PLUGINS_ENV = "DEXENGINE_RECORD_PLUGINS"
KINDS = ("session_stage", "bake", "aug_session", "driver_spec", "owner_moves", "persist_job")

_REGISTRY: dict[str, dict[str, Any]] = {k: {} for k in KINDS}
_LOADED: list[str] = []


@dataclass(frozen=True)  # noqa: V102 — plugin API (experiments/policies/kin_data/plugin.py)
class SessionStage:
    """An extra session stage provided by a plugin.

    prepare(args, argv, env) -> (fn, name) runs in the session process after the judge dump is loaded; it may edit
    ``env`` in place and returns the render-hook function and its display name.
    log_problems(text) -> list[str] checks the finished session log (empty list = valid).
    """

    name: str
    prepare: Callable[..., tuple[Callable, str]]
    log_problems: Callable[[str], list[str]]
    timeout_s: int
    hook_line: str
    progress: str | None


def register(kind: str, name: str, obj: Any) -> None:  # noqa: V103 — tests only
    """Register ``obj`` under (kind, name). Registering a different object under a taken name is an error."""
    if kind not in _REGISTRY:
        raise ValueError(f"unknown hook kind {kind!r} (known: {KINDS})")
    old = _REGISTRY[kind].get(name)
    if old is not None and old is not obj:
        raise ValueError(f"hook {kind}:{name} is already registered")
    _REGISTRY[kind][name] = obj


def get(kind: str, name: str) -> Any:
    """The object registered under (kind, name), or None."""
    return _REGISTRY[kind].get(name)


def names(kind: str) -> tuple[str, ...]:
    """Registered names of one kind, in registration order."""
    return tuple(_REGISTRY[kind])


def add_plugin(module: str, env: dict | None = None) -> str:  # noqa: V103 — used by experiments/policies/kin_data/collect.py
    """Append ``module`` to DEXENGINE_RECORD_PLUGINS in ``env`` (default os.environ) unless it is already named there;
    return the new value. Plugins named by the caller stay loaded."""
    env = os.environ if env is None else env
    have = [m.strip() for m in env.get(PLUGINS_ENV, "").split(",") if m.strip()]
    if module not in have:
        env[PLUGINS_ENV] = ",".join([*have, module])
    return env[PLUGINS_ENV]


def load_plugins(env: dict | None = None) -> list[str]:
    """Import the plugin modules named in DEXENGINE_RECORD_PLUGINS (once each); return the loaded names."""
    spec = (os.environ if env is None else env).get(PLUGINS_ENV, "")
    for mod in (m.strip() for m in spec.split(",")):
        if mod and mod not in _LOADED:
            importlib.import_module(mod)
            _LOADED.append(mod)
    return list(_LOADED)
