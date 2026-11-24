"""The one owner of stdout / stderr redirection (tests/core/test_stdio.py fails on any other in dexengine/).

`contextlib.redirect_stdout` swaps the process's `sys.stdout` for the block: every other thread's prints land in the
block's buffer, and blocks that end out of order in two threads leave `sys.stdout` pointing at a dead buffer (measured
in the f0 prejudge: its 16 prep threads, each quieting the f0 key reader, swallowed the prejudge's own lines).

- `capture`: per thread. The first call puts a router in front of `sys.stdout` (and `sys.stderr`); the router writes
  to the current thread's innermost capture target, or to the stream it stands in front of when the thread has none.
  Captures nest. Prints from threads that the captured call starts itself are not captured (they are other threads).
- `process_stream`: process level — make a stream the process's own stdout or stderr (a run log's tee at process
  start, a forked worker's recorder). It is the plain assignment, owned here so the guard can ban the others.
"""

from __future__ import annotations

import contextlib
import io
import sys
import threading

_LOCAL = threading.local()
_LOCK = threading.Lock()
_STREAMS = ("stdout", "stderr")


class _Router:
    """A stream that writes to this thread's capture target (innermost) or to the stream it stands in front of."""

    def __init__(self, real, which: str):
        self._real = real
        self._which = which

    def _target(self):
        stack = getattr(_LOCAL, self._which, None)
        return stack[-1] if stack else self._real

    def write(self, s):
        return self._target().write(s)

    def flush(self):
        t = self._target()
        if hasattr(t, "flush"):
            t.flush()

    def __getattr__(self, name):  # fileno, isatty, encoding, buffer … of the stream it stands in front of
        return getattr(self._real, name)


def _install(which: str) -> None:
    with _LOCK:
        if not isinstance(getattr(sys, which), _Router):
            setattr(sys, which, _Router(getattr(sys, which), which))


def _resolved(t):
    """A router as a target means "where that stream goes now, for this thread" (else it would route to itself)."""
    while isinstance(t, _Router):
        t = t._target()
    return t


@contextlib.contextmanager
def capture(target=None, *, stderr=False):
    """Route this thread's stdout to `target` for the block (None = a fresh buffer; the target is yielded) — and its
    stderr to the same target (`stderr=True`) or to a stream of its own (`stderr=<stream>`). Other threads keep
    writing where they did."""
    tgt = io.StringIO() if target is None else target
    routes = [("stdout", tgt)] + ([("stderr", tgt if stderr is True else stderr)] if stderr is not False else [])
    pushed = []
    try:
        for w, t in routes:
            dest = _resolved(t)
            _install(w)
            stack = getattr(_LOCAL, w, None)
            if stack is None:
                stack = []
                setattr(_LOCAL, w, stack)
            stack.append(dest)
            pushed.append(w)
        yield tgt
    finally:
        for w in pushed:
            getattr(_LOCAL, w).pop()


def process_stream(which: str, stream):
    """Process level: make `stream` the process's stdout or stderr and return the one it replaces (to put back).
    For a single-threaded process start or a forked worker — use `capture` for anything a thread does."""
    if which not in _STREAMS:
        raise ValueError(f"stdio.process_stream: stdout or stderr, not {which!r}")
    with _LOCK:
        old = getattr(sys, which)
        setattr(sys, which, stream)
    return old
