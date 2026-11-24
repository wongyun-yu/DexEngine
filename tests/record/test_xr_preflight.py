"""The augment bake's machine gate (record.augment.xr_deps): a missing cross-repair dependency and an interpreter that
cannot compile a Warp kernel stop the round before any bake; a 172-character TMPDIR (a node job's scratch) passes; the
round start really calls it before the augment round (structural, with the call planted out)."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import os
import shutil
import subprocess
import sys

import pytest

from dexengine.record import chain as CH
from dexengine.record.augment import xr_deps as XD


def test_a_missing_dependency_stops_the_round(monkeypatch):
    monkeypatch.setenv("PYDP", sys.executable)
    monkeypatch.setattr(XD, "XR_DEPS", ("no_such_module_xr",))
    with pytest.raises(SystemExit, match=r"\[AUG_XR_DEPS\] ✗ cross-repair dependencies no_such_module_xr"):
        XD.xr_preflight("")


@pytest.mark.skipif(importlib.util.find_spec("warp") is None, reason="warp is not installed here")
def test_a_node_jobs_long_tmpdir_passes_the_compile_check(monkeypatch, tmp_path):
    long = str(tmp_path) + "/" + "a" * (171 - len(str(tmp_path)))
    os.makedirs(long)
    monkeypatch.setenv("PYDP", sys.executable)
    monkeypatch.setenv("TMPDIR", long)
    monkeypatch.setattr(XD, "XR_DEPS", ("warp",))
    line = XD.xr_preflight("")
    assert line.startswith("[AUG_XR_DEPS] cross-repair deps warp OK · [XR_TMP] CHECK OK nvrtc compile in /tmp/xr.")
    assert "caller TMPDIR 172 chars" in line


def _python_without_warp() -> str | None:
    py = shutil.which("python3", path="/usr/bin")
    if not py:
        return None
    r = subprocess.run(
        [py, "-c", "import importlib.util as u; print(u.find_spec('warp') is None)"], capture_output=True
    )
    return py if r.stdout.strip() == b"True" else None


@pytest.mark.skipif(_python_without_warp() is None, reason="no interpreter without warp on this machine")
def test_an_interpreter_that_cannot_compile_stops_the_round(monkeypatch):
    monkeypatch.setenv("PYDP", _python_without_warp())
    monkeypatch.setattr(XD, "XR_DEPS", ())
    with pytest.raises(SystemExit, match=r"compile check failed rc 1 .*ModuleNotFoundError"):
        XD.xr_preflight("")


def gate_before_round(src: str) -> bool:
    """In after_store, a call of xr_preflight comes before the augment round's run call (ASR.run)."""
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "after_store")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    gate = [c.lineno for c in calls if c.func.attr == "xr_preflight"]
    run = [c.lineno for c in calls if c.func.attr == "run" and getattr(c.func.value, "id", "") == "ASR"]
    return bool(gate and run) and min(gate) < min(run)


def test_the_round_start_calls_the_gate_first():
    src = inspect.getsource(CH)
    assert gate_before_round(src)
    planted = src.replace("print(XD.xr_preflight(a.gpu), flush=True)", "pass")
    assert planted != src and not gate_before_round(planted)
