"""core.errors: the input-missing marker that record.driver_spec reads from a log line."""

from __future__ import annotations

import subprocess
import sys

import pytest

from dexengine.core import errors as E


@pytest.mark.parametrize(
    "msg,want",
    [
        ("[TAG] x is missing", f"[TAG] {E.INPUT_MISSING} x is missing"),
        ("plain x is missing", f"{E.INPUT_MISSING} plain x is missing"),
        ("  ✗ no link", f"{E.INPUT_MISSING}   ✗ no link"),
    ],
)
def test_marker_goes_after_a_leading_tag(msg, want):
    assert E.input_missing(msg) == want
    assert E.input_missing(want) == want  # marking twice changes nothing
    assert E.is_input_missing(want) and not E.is_input_missing(msg)


def test_input_missing_error_exits_like_system_exit():
    """rc 1 and the marked message on stderr, as raise SystemExit(msg) does."""
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "from dexengine.core.errors import InputMissingError\nraise InputMissingError('[X] input y is missing')",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1 and r.stderr.strip() == f"[X] {E.INPUT_MISSING} input y is missing"
    try:
        raise E.InputMissingError("[X] y")
    except SystemExit as e:  # existing `except SystemExit` handlers still catch it
        assert e.code == f"[X] {E.INPUT_MISSING} y"


_ERROR_SINKS = {"die", "_die", "print", "SystemExit", "__init__", "Refused", "InputMissingError"}


def _misrouted_marks(src: str, path: str) -> list[str]:
    """input_missing() calls whose text does not flow into an error path. A marked string is a message: it may be
    raised, passed to die / print / an exception, assigned or returned — never handed to a function as data (a
    marked dataset name reaches registry.dataset as '[INPUT_MISSING] <ds>')."""
    import ast

    tree = ast.parse(src)
    parent = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
    bad = []
    for n in ast.walk(tree):
        f = getattr(n, "func", None)
        if not (isinstance(n, ast.Call) and getattr(f, "id", getattr(f, "attr", None)) == "input_missing"):
            continue
        p = parent.get(n)
        while isinstance(p, (ast.IfExp, ast.BinOp, ast.JoinedStr, ast.FormattedValue)):
            p = parent.get(p)
        if isinstance(p, (ast.Raise, ast.Assign, ast.Return)):
            continue
        name = getattr(getattr(p, "func", None), "id", getattr(getattr(p, "func", None), "attr", ""))
        if isinstance(p, ast.Call) and (name in _ERROR_SINKS or name.endswith(("Error", "Exception"))):
            continue
        bad.append(f"{path}:{n.lineno}")
    return bad


def test_input_missing_only_wraps_messages():
    import pathlib

    root = pathlib.Path(E.__file__).resolve().parents[2]
    bad = [b for f in sorted((root / "dexengine").rglob("*.py")) for b in _misrouted_marks(f.read_text(), str(f))]
    assert not bad, bad


def test_misrouted_mark_is_caught():
    assert _misrouted_marks("follow_pieces(input_missing(ds), hand)\n", "x.py") == ["x.py:1"]
    assert _misrouted_marks("die(input_missing(m) if gap else m)\n", "x.py") == []
