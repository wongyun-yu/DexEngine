"""The f0 exact inside test (repair_geom.ObjectGeom.contains_exact) reads the ruler owner (collider_shape.Triangles),
not trimesh's ray casting."""

from __future__ import annotations

import numpy as np
import pytest

from tests.imports import own_module


def _geom(monkeypatch, tmp_path):
    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    trimesh = pytest.importorskip("trimesh")
    RG = own_module("dexengine.retarget.f0.repair_geom")
    trimesh.creation.box(extents=(0.04, 0.06, 0.02)).export(tmp_path / "visual.obj")
    return RG.ObjectGeom(tmp_path, n_samples=2000), trimesh


def test_exact_count_is_the_ruler_inside_test(monkeypatch, tmp_path):
    """A box posed by (R, pos): points 0.5 and 3 mm inside count, 0.5 and 3 mm outside do not; trimesh's mesh.contains
    is never called; the sign is the ruler owner's one rule (collider_shape.SIGN_RULE)."""
    og, trimesh = _geom(monkeypatch, tmp_path)
    monkeypatch.setattr(trimesh.Trimesh, "contains", lambda *a, **k: pytest.fail("mesh.contains called"))
    local = np.array([[0.0, 0.0, 0.0095], [0.0, 0.0, 0.007], [0.0, 0.0, 0.0105], [0.0, 0.0, 0.013], [0.0195, 0, 0]])
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pos = np.array([0.3, -0.2, 0.1])
    assert og.contains_exact(R, pos, local @ R.T + pos) == 3
    from dexengine.data.objects import collider_shape as CS

    assert og._tri.sign_rule == CS.SIGN_RULE and og.contains_exact(np.eye(3), np.zeros(3), local + 1.0) == 0
