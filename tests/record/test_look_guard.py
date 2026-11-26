"""Frame guard (dexengine.record.looks.look_guard): no frame is written unless the canonical look was applied to the
current stage in this process.

Planted violation: a writer opened without the token dies; the token (set only by look_d.apply, here through its thin
setter on a pxr stage) lets it pass; a token of another stage, or a cleared one, dies again. Structure: the setter has
exactly one caller (look_d.apply), every Isaac RGB read (`.data.output["rgb"]`) and every frame writer of the inventory
calls require().
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

from dexengine.record.looks import look_guard as LG

REPO = Path(__file__).resolve().parents[2]
SCAN = ("dexengine", "experiments", "ops", "scripts")
#: (file, function) of every frame producer; each must call look_guard.require (the inventory of the report)
GUARDED = {
    ("dexengine/record/render/rig.py", "rgb"),
    ("dexengine/record/render/writer.py", "__init__"),
    ("dexengine/record/render/writer.py", "add"),
    ("dexengine/record/looks/augvis/render.py", "save"),
    ("dexengine/rl/render/pose_check.py", "_save"),
    ("dexengine/rl/render/pose_check.py", "_still"),
    ("experiments/sim_eval/loop/record.py", "_open_head_encoder"),
    ("experiments/sim_eval/loop/record.py", "_open_exo_encoder"),
}


@pytest.fixture(autouse=True)
def _clean_token():
    LG.clear("test start")
    yield
    LG.clear("test end")


def _stage():
    pytest.importorskip("pxr")
    from pxr import Usd

    return Usd.Stage.CreateInMemory()


def _writer(tmp_path, monkeypatch):
    from dexengine.record.render import writer as W

    monkeypatch.setattr(W.EpisodeWriter, "_put", lambda self, item: None)  # no encoder thread work in the test
    monkeypatch.setattr(W.threading.Thread, "start", lambda self: None)
    return W.EpisodeWriter(tmp_path / "ep", 1, ("egoL",), 30, "depth_mm.npy.zst")


def test_write_without_look_dies(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match=r"\[LOOK_GUARD\].*without the canonical look"):
        _writer(tmp_path, monkeypatch)


def test_write_after_look_passes(tmp_path, monkeypatch):
    stage = _stage()
    assert LG.MARK in LG.mark(stage, "D")
    w = _writer(tmp_path, monkeypatch)
    w.add({"egoL": object(), "depth_m": object()})  # the guard passes; the queue put is stubbed
    assert w.added == 1


def test_cleared_token_dies_again(tmp_path, monkeypatch):
    LG.mark(_stage(), "D")
    LG.clear("next clip")
    with pytest.raises(SystemExit, match="without the canonical look"):
        _writer(tmp_path, monkeypatch)


def test_token_of_another_stage_dies(monkeypatch):
    applied, current = _stage(), _stage()
    LG.mark(applied, "D")
    fake = types.ModuleType("omni.usd")
    fake.get_context = lambda: types.SimpleNamespace(get_stage=lambda: current)
    monkeypatch.setitem(sys.modules, "omni.usd", fake)
    with pytest.raises(SystemExit, match="the look was applied to stage"):
        LG.require("test")
    fake.get_context = lambda: types.SimpleNamespace(get_stage=lambda: applied)
    LG.require("test")  # the same stage passes


def test_pose_still_without_look_dies(tmp_path):
    pytest.importorskip("torch")
    import numpy as np

    from dexengine.rl.render import pose_check as PC

    with pytest.raises(SystemExit, match=r"\[LOOK_GUARD\]"):
        PC._save(np.zeros((2, 2, 3), np.uint8), str(tmp_path / "x.png"))
    assert not (tmp_path / "x.png").exists()


def _py_files():
    for top in SCAN:
        yield from (REPO / top).rglob("*.py")


def _functions(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _calls(fn, attr: str) -> bool:
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attr for n in ast.walk(fn)
    )


def _guard_aliases(tree) -> set[str]:
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module == "dexengine.record.looks":
            out |= {a.asname or a.name for a in n.names if a.name == "look_guard"}
    return out


def test_token_setter_has_one_caller():
    """look_guard.mark is called only by look_d.apply (the one look function)."""
    callers = set()
    for f in _py_files():
        text = f.read_text(errors="replace")
        if "look_guard" not in text or f.name == "look_guard.py":
            continue
        tree = ast.parse(text)
        names = _guard_aliases(tree)
        for fn in _functions(tree):
            for n in ast.walk(fn):
                if (
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "mark"
                    and isinstance(n.func.value, ast.Name)
                    and n.func.value.id in names
                ):
                    callers.add((str(f.relative_to(REPO)), fn.name))
    assert callers == {("dexengine/record/looks/look_d.py", "apply")}


def test_every_frame_producer_calls_the_guard():
    """Each inventory entry calls require(); every function that reads Isaac RGB (`.data.output["rgb"]`) is in it."""
    for rel, name in GUARDED:
        tree = ast.parse((REPO / rel).read_text())
        fns = [fn for fn in _functions(tree) if fn.name == name]
        assert fns and all(_calls(fn, "require") for fn in fns), (rel, name)
    readers = set()
    for f in _py_files():
        text = f.read_text(errors="replace")
        if '"rgb"]' not in text:
            continue
        for fn in _functions(ast.parse(text)):
            src = ast.unparse(fn)
            if "data.output['rgb']" in src and not any(
                "data.output['rgb']" in ast.unparse(inner) for inner in _functions(fn) if inner is not fn
            ):
                readers.add((str(f.relative_to(REPO)), fn.name))
    assert readers <= GUARDED, readers - GUARDED
