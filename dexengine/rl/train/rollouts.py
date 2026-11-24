"""PPO rollouts: the single source of "env steps per PPO update".

Every reader of that number sees the same value: the PPO learner (skrl ``rollouts``, the update size), the curriculum
denominator (``total_timesteps = MI x rollouts``, used by every %-based schedule), the trainer's iteration arithmetic
(checkpoint names, non-finite guard, resume offset), and the launcher and judge sidecar, which turn iterations into
checkpoint names from ``DEXENGINE_ROLLOUTS``.

Rule: the skrl yaml's PPO ``rollouts`` key is the value; ``DEXENGINE_ROLLOUTS``, when set (non-empty), replaces it for
every reader. A yaml without the key, or a value that is not a positive integer, raises — there is no default.
Standard library only, so CPU tests can import it.
"""

from __future__ import annotations

import os

from dexengine.core.errors import InputMissingError

ENV = "DEXENGINE_ROLLOUTS"


def ppo_rollouts(agent_params: dict, environ=None) -> tuple[int, str]:
    """(rollouts, source) — source "env" when DEXENGINE_ROLLOUTS is set, else "yaml" (``agent_params`` = the yaml's
    PPO mapping). Dies (SystemExit) on a missing yaml key or a value that is not a positive integer."""
    env = os.environ if environ is None else environ
    if "rollouts" not in agent_params:
        raise InputMissingError("[ROLLOUTS] agent yaml has no agent.rollouts — the update size is undefined")
    src, raw = ("env", env[ENV]) if env.get(ENV, "") != "" else ("yaml", agent_params["rollouts"])
    text = str(raw).strip()
    if not text.isdigit() or int(text) < 1:
        raise SystemExit(f"[ROLLOUTS] {src} value {raw!r} is not a positive integer ({ENV} · yaml agent.rollouts)")
    return int(text), src
