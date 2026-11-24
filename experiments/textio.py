"""Small text-file helpers shared by the experiment scripts (standard library only)."""

from __future__ import annotations

import os


def read_lines(path: str | os.PathLike, **open_kw) -> list[str]:
    """The file's lines exactly as iterating ``open(path, **open_kw)`` yields them (line ends kept), file closed.

    Use it where a file is iterated inside an expression (a comprehension under a condition, a reader built per item)
    and a ``with`` block would not fit.
    """
    with open(path, **open_kw) as fh:
        return list(fh)
