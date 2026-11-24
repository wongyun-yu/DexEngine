"""Stall evidence without ptrace: when a session's python receives SIGNAL, the Python stacks of all threads go to
stderr (faulthandler is a C-level signal handler, so it writes even when the process is stuck holding the GIL).
The session entry (dexengine/rl/cli.py main — after AppLauncher, so after Kit's handlers) calls arm(); the stall
watcher sends SIGNAL just before it reaps the session.
This file is the one owner of the signal name. Marker: [STALL_STACK] on.
"""

from __future__ import annotations

import faulthandler
import signal
import sys

SIGNAL = signal.SIGUSR1
MARK = "[STALL_STACK]"


def arm() -> None:
    faulthandler.register(SIGNAL, file=sys.stderr, all_threads=True, chain=False)
    print(
        f"{MARK} on — {SIGNAL.name} → Python stacks of all threads (faulthandler · the watcher sends it before "
        "reaping)",
        flush=True,
    )
