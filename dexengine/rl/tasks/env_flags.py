"""The one reader of switch values and process-boundary variables inside the RL task package (stdlib only at import).

Every read in the package goes through :func:`get`, and `tests/rl_tasks` checks that no other read exists. A name is
either
- a switch of the package (`dexengine.rl.tasks.switches`): `get` returns its value in the old text form, from the
  trainer process's switches file, or from the environment in a process that has no file; or
- a process-boundary variable of `dexengine.core.config.ENV_VARS` (arm, hand, dataset, embodiment, data root, clip
  list, roster, arm IK cache suffix): `get` reads the environment. `core.config` is imported only for these names.

Two places write a process-boundary value inside the package: `env_setup.build`, which restores a checkpoint's
recorded extra-slot switch, and `rl/cli.py`, which sets ARM / HAND from its flags. A read keeps its moment: module-level
reads still happen at import, the others when their code runs.
"""

from __future__ import annotations

import os

from dexengine.rl.tasks import switches

_UNSET = object()


def get(name: str, default=_UNSET, environ=None) -> str | None:
    """The value of switch or process-boundary variable `name` (old text form); `default` is returned when it is not
    set (default: the switch's unset value, "" for a process-boundary variable). Unknown names fail."""
    if name in switches.FIELD_OF:
        return switches.text(name, switches._UNSET if default is _UNSET else default, environ)
    from dexengine.core import config  # names core owns; its ENV_VARS is their one statement

    if name not in config.ENV_VARS:
        raise KeyError(
            f"[ENV_FLAGS] {name} is neither a switch of the RL task package (dexengine.rl.tasks.switches) "
            "nor a process-boundary variable of dexengine.core.config"
        )
    environ = os.environ if environ is None else environ
    return environ.get(name, "" if default is _UNSET else default)
