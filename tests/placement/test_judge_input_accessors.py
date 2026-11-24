"""The judge input record's lazy module accessors (`judge.input_fields._x()`) still own what their callers use.

The accessors import inside a function, so a function moving to another module (collision.core → collision.idle_arm)
is invisible to import-time checks and only kills the judge at run time — on every clip, at the `case`
field. This test reads every `_x().name` use in the judge package and checks the module `_x()` returns has `name`."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from dexengine.placement.judge import input_fields as IF

JUDGE = Path(IF.__file__).resolve().parent


def _uses():
    """{accessor name: {attribute, …}} for every `_x().attr` in the judge package where `_x` is an input_fields
    function."""
    acc = {
        n
        for n, v in vars(IF).items()
        if n.startswith("_") and callable(v) and getattr(v, "__module__", "") == IF.__name__
    }
    out = {}
    for f in sorted(JUDGE.glob("*.py")):
        for node in ast.walk(ast.parse(f.read_text(), str(f))):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in acc
                and not node.value.args
            ):
                out.setdefault(node.value.func.id, set()).add(node.attr)
    return out


def test_the_scan_sees_the_case_accessor():
    assert "active_idle_for_tree" in _uses().get("_idle_arm", set())


@pytest.mark.parametrize("name, attrs", sorted((k, tuple(sorted(v))) for k, v in _uses().items()))
def test_accessor_module_has_every_used_attribute(name, attrs):
    try:
        mod = getattr(IF, name)()
    except ImportError as e:  # a third-party package this CPU environment lacks (not a moved symbol of ours)
        if (e.name or "").startswith("dexengine"):
            raise
        pytest.skip(f"{name}: {e}")
    assert importlib.import_module(mod.__name__) is mod
    missing = [a for a in attrs if not hasattr(mod, a)]
    assert not missing, f"{name}() → {mod.__name__} has no {missing}"
