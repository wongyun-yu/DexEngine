"""repair_geom's numeric helpers live in repair_math (one owner): repair_geom re-exports each moved name as the same
object, and neither module defines a name the other owns."""

from __future__ import annotations

import ast
import inspect

import pytest

MOVED = ("_rpy_to_R", "_quat_wxyz_to_R", "_R_to_quat_wxyz", "_aa_to_R", "_slerp", "_decimate_vertices", "_reverse_hull",
         "hulls_contain", "WRIST_FIT_MAX_RES", "fit_wrist_qpos_map")  # fmt: skip


def _defined(mod) -> set:
    out = set()
    for n in ast.parse(inspect.getsource(mod)).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Assign):
            out |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    return out


@pytest.fixture
def mods(monkeypatch):
    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    from dexengine.retarget.f0 import repair_geom, repair_math

    return repair_geom, repair_math


def test_moved_names_are_the_same_objects(mods):
    G, M = mods
    for n in MOVED:
        assert getattr(G, n) is getattr(M, n), n


def test_one_owner_per_name(mods):
    G, M = mods
    assert set(MOVED) <= _defined(M)
    assert not set(MOVED) & _defined(G)
    assert not (_defined(M) - set(MOVED))  # repair_math owns nothing else
