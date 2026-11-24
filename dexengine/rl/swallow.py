"""One-line markers for exceptions a step deliberately survives (narrow the type, log a marker).

`note_once(tag, exc)` prints `[<tag>] skipped: <exc>` the first time a tag is seen in this process, so a per-step site
does not print every step.
"""

from __future__ import annotations

_SEEN: set[str] = set()


def note_once(tag: str, exc: BaseException) -> None:
    """Print one marker line for ``tag`` (first occurrence only)."""
    if tag in _SEEN:
        return
    _SEEN.add(tag)
    print(f"[{tag}] skipped: {type(exc).__name__}: {exc}", flush=True)
