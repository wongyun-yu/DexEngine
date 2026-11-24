"""A dexengine module binds each top-level name once (assignment, def or class): a second binding silently replaces the
first for every reader — a cache alias once took the name of the collider shapes' tuple and every shape check failed."""

from __future__ import annotations

import ast
from collections import Counter

import pytest

from dexengine.core.repo import REPO_ROOT


def rebound(src: str) -> list[str]:
    """Top-level names bound more than once (statements directly in the module body; if / try blocks are left alone)."""
    seen: Counter = Counter()
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign):
            seen.update(t.id for t in n.targets if isinstance(t, ast.Name))
        elif isinstance(n, ast.AnnAssign) and n.value is not None and isinstance(n.target, ast.Name):
            seen[n.target.id] += 1
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            seen[n.name] += 1
    return sorted(k for k, v in seen.items() if v > 1)


def test_no_dexengine_module_binds_a_top_level_name_twice():
    bad = {}
    for p in sorted((REPO_ROOT / "dexengine").rglob("*.py")):
        if v := rebound(p.read_text(encoding="utf-8")):
            bad[str(p.relative_to(REPO_ROOT))] = v
    assert not bad, bad


@pytest.mark.parametrize(
    "src",
    [
        "SHAPES = ('a', 'b')\n_C = {}\nSHAPES = _C\n",
        "def f():\n    pass\n\n\nf = 1\n",
        "X: int = 1\nclass X:\n    pass\n",
    ],
)
def test_the_check_catches_a_planted_rebinding(src):
    assert rebound(src)
