"""Debug marker updates run only when a frame can show them (dexengine/rl/tasks/env_base.py `_markers_can_show`).

The predicate is extracted from the source (env_base imports Isaac Lab) and run on a fake env: any of GUI, RTX sensor
or gym render mode keeps the updates; with none of them the update is skipped and `[MARKERS]` is printed once.
The structure rule: `step` calls `_update_debug_markers` only under `if self._markers_can_show()` (fails on the old,
unconditional call)."""

from __future__ import annotations

import ast
import itertools
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "dexengine" / "rl" / "tasks" / "env_base.py"
TREE = ast.parse(SRC.read_text())


def _method(name):
    for n in ast.walk(TREE):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise AssertionError(f"{name} not found in {SRC}")


def _predicate():
    fn = _method("_markers_can_show")
    mod = ast.Module(body=[fn], type_ignores=[])
    ns: dict = {}
    exec(compile(ast.fix_missing_locations(mod), str(SRC), "exec"), ns)
    return ns["_markers_can_show"]


class _Sim:
    def __init__(self, gui, rtx):
        self._gui, self._rtx = gui, rtx

    def has_gui(self):
        return self._gui

    def has_rtx_sensors(self):
        return self._rtx


class _Env:
    def __init__(self, render_mode, gui, rtx):
        self.render_mode, self.sim = render_mode, _Sim(gui, rtx)


@pytest.mark.parametrize("render_mode,gui,rtx", itertools.product((None, "rgb_array"), (False, True), (False, True)))
def test_predicate(render_mode, gui, rtx, capsys):
    env = _Env(render_mode, gui, rtx)
    got = _predicate()(env)
    assert got == (render_mode is not None or gui or rtx)
    out = capsys.readouterr().out
    assert ("[MARKERS]" in out) == (not got)


def test_skip_line_printed_once(capsys):
    env, pred = _Env(None, False, False), _predicate()
    assert not pred(env) and not pred(env)
    assert capsys.readouterr().out.count("[MARKERS]") == 1


def test_step_gates_the_update():
    step = _method("step")
    calls = [
        n
        for n in ast.walk(step)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_update_debug_markers"
    ]
    assert len(calls) == 1, "step must call _update_debug_markers exactly once"
    gated = [
        n
        for n in ast.walk(step)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Call)
        and isinstance(n.test.func, ast.Attribute)
        and n.test.func.attr == "_markers_can_show"
        and any(c is calls[0] for b in n.body for c in ast.walk(b))
    ]
    assert gated, "the marker update in step is not under `if self._markers_can_show()`"
