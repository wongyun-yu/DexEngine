"""Kit-first import order for the training entry (dexengine/rl/cli.py) — stdlib only.

Isaac Sim's Kit brings its own numpy. When the interpreter's venv carries a numpy with a different ABI (e.g. venv
2.x, Kit 1.26) and the venv copy is imported first, every Kit extension that uses numpy fails to load
(`numpy.dtype size changed … Expected 96 … got 88`: isaacsim.core.api, isaacsim.core.cloner, isaaclab_tasks). On a
machine whose two copies agree the same order is harmless, so the fault is invisible there. Rule: nothing the entry
runs before AppLauncher may import numpy or torch (torch pulls numpy).

  watch()          first statement of the entry: a meta-path hook that records, for each watched package, the import
                   chain that loads it first (it finds nothing itself — the normal finders still import)
  assert_clean()   right before AppLauncher: dies naming that chain when a watched package is already in sys.modules,
                   otherwise prints the marker `[PRE_APP] import order ok`
  stop()           after AppLauncher: removes the hook (Kit and the runners import numpy/torch freely from then on)
"""

from __future__ import annotations

import sys
import traceback

WATCHED = ("numpy", "torch")
MARKER = "[PRE_APP]"
_FIRST: dict[str, list[str]] = {}


class _FirstImportWatch:
    """Records who imports a watched package first; never resolves anything."""

    def find_spec(self, name, path=None, target=None):
        root = name.partition(".")[0]
        if root in WATCHED and root not in _FIRST:
            frames = [
                f
                for f in traceback.extract_stack()[:-1]
                if "importlib" not in f.filename and not f.filename.startswith("<frozen")
            ]
            _FIRST[root] = [f"{f.filename}:{f.lineno}" for f in frames[-5:]]
        # finds nothing (implicit None): the normal finders go on and import the package


def watch() -> None:
    """Install the first-import watch (first statement of the entry)."""
    if not any(isinstance(f, _FirstImportWatch) for f in sys.meta_path):
        sys.meta_path.insert(0, _FirstImportWatch())


def stop() -> None:
    """Remove the watch (after AppLauncher)."""
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _FirstImportWatch)]


def loaded() -> list[str]:
    """The watched packages already in sys.modules."""
    return [m for m in WATCHED if m in sys.modules]


def _chain(m: str) -> str:
    if m in _FIRST:
        return " → ".join(_FIRST[m])
    return "(already loaded when the watch started — interpreter startup or a wrapper)"


def assert_clean(where: str = "AppLauncher") -> None:
    """SystemExit naming the import chain if a watched package is loaded; else print the ok marker."""
    bad = loaded()
    if bad:
        chains = "; ".join(f"{m} ← {_chain(m)}" for m in bad)
        raise SystemExit(
            f"{MARKER} {', '.join(bad)} imported before {where}: {chains}. Kit must load its own numpy "
            "first — a venv numpy with another ABI breaks every Kit extension (numpy.dtype size changed). "
            "Import it lazily inside the function that needs it, or move the caller after AppLauncher."
        )
    print(f"{MARKER} import order ok — {'/'.join(WATCHED)} not loaded before {where}", flush=True)
