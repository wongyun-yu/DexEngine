"""Inside and depth readers outside collider_shape go through its owner (collider_shape.Triangles): the
articulation limit depth (data.artic.limits) and the hocap carried-hand oracle (readers.raw.hocap). Each: a point
inside reads inside at its depth, outside reads 0, and an inside-out mesh reads the same (|winding number| > 1/2)."""

from __future__ import annotations

import numpy as np
import pytest

from dexengine.data.objects import collider_shape as CS


def _box(half=0.02):
    import trimesh  # noqa: PLC0415

    m = trimesh.creation.box(extents=(2 * half,) * 3)
    return np.asarray(m.vertices), np.asarray(m.faces)


def test_limits_depth_reads_the_owner():
    from dexengine.data.artic import limits as AL

    V, F = _box()
    rng = np.random.default_rng(0)
    inside = rng.uniform(-0.019, 0.019, (AL.KTH + 4, 3))
    outside = rng.uniform(0.03, 0.05, (20, 3))
    P = np.concatenate([inside, outside])
    want = float(np.sort(CS.Triangles(V, F).depths(P)[: len(inside)])[-AL.KTH])
    for T in (CS.Triangles(V, F), CS.Triangles(V, F[:, ::-1])):
        assert AL._depth(T, P) == pytest.approx(want, abs=1e-12)
    assert AL._depth(CS.Triangles(V, F), outside) == 0.0  # fewer than KTH inside


def test_hocap_ruler_is_the_owner(tmp_path):
    import trimesh  # noqa: PLC0415

    from dexengine.data.readers.raw import hocap as HO

    V, F = _box()
    f = tmp_path / "obj.ply"
    trimesh.Trimesh(V, F).export(f)
    tri = HO._ruler(str(f))
    assert isinstance(tri, CS.Triangles)
    assert tri.depths(np.array([[0.0, 0.0, 0.015], [0.0, 0.0, 0.03]])) == pytest.approx([0.005, 0.0], abs=1e-9)
    open_f = tmp_path / "open.ply"
    trimesh.Trimesh(V, F[2:]).export(open_f)
    with pytest.raises(ValueError, match="not watertight"):
        HO._ruler(str(open_f))
