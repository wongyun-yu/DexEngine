"""The batched hand-clear ruler (CPU) gives the same values as one placement at a time: TriModel.depth_poses against
place + depth (bit-equal, random poses round a concave triangle body), and row_verdict whatever the frame batch, per
hand side."""

from __future__ import annotations

import numpy as np
import pytest

from dexengine.data.objects import collider_shape as CS
from dexengine.data.objects import hand_clear as HC

trimesh = pytest.importorskip("trimesh")


def _rack():
    """A concave triangle body: a box with a through hole (annulus extruded), open-mesh shaped like the receivers."""
    m = trimesh.creation.annulus(r_min=0.012, r_max=0.05, height=0.04)
    return CS.Triangles(np.asarray(m.vertices), np.asarray(m.faces))


def _hulls(rng, n=6):
    return [rng.normal(scale=0.008, size=(12, 3)) for _ in range(n)]


def _poses(rng, n, h):
    R = np.stack([np.stack([np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(h)]) for _ in range(n)])
    t = rng.uniform(-0.05, 0.05, size=(n, h, 3))
    pR = np.stack([np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(n)])
    pt = rng.uniform(-0.01, 0.01, size=(n, 3))
    return R, t, pR, pt


def test_batched_depths_equal_one_placement_at_a_time():
    rng = np.random.default_rng(7)
    tri, hulls = _rack(), _hulls(rng)
    tm = HC.TriModel(hulls, tri)
    R, t, pR, pt = _poses(rng, 40, len(hulls))
    d, a = tm.depth_poses(R, t, pR, pt)
    ref = []
    for p in range(40):
        tm.place(R[p], t[p], pR[p], pt[p])
        ref.append(tm.depth())
    assert np.array_equal(d, [x for x, _ in ref]) and np.array_equal(a, [y for _, y in ref])
    assert (d > 0).sum() >= 5  # the poses do overlap (the test is not vacuous)
    floor = float(np.median(d[d > 0]))
    df, af = tm.depth_poses(R, t, pR, pt, floor=floor)
    above = d > floor
    assert np.array_equal(df[above], d[above]) and np.array_equal(af[above], a[above])  # exact above the floor
    assert (df[~above] <= floor).all() and (df <= d).all()  # a lower bound at or below it


def test_row_verdict_does_not_depend_on_the_frame_batch(monkeypatch):
    rng = np.random.default_rng(3)
    tri, hulls = _rack(), _hulls(rng, 4)

    class Ref:  # RefHands stand-in: hull geoms, per-frame poses
        gR, gx = _poses(rng, 30, 4)[:2]

    monkeypatch.setattr(HC, "_hand_meshes", lambda ref: (hulls, ["right", "right", "left", "left"]))
    monkeypatch.setattr(
        HC, "_model", lambda ds, o, row, verts: lambda hands: HC.TriModel([verts[i] for i in hands], tri)
    )
    T = np.repeat(np.eye(4)[None], 30, 0)
    out = []
    for b in (1, 7, 64):
        monkeypatch.setattr(HC, "FRAME_BATCH", b)
        out.append(HC.row_verdict(Ref, "taco", {}, T, {}))
    assert out[0] == out[1] == out[2] and out[0]["depth_m"] > 0
    assert set(out[0]["sides"]) == {"right", "left"} and out[0]["depth_m"] == max(out[0]["sides"].values())
    for side, hands in (("right", [0, 1]), ("left", [2, 3])):  # each side reads its own hulls only
        tm = HC.TriModel([hulls[i] for i in hands], tri)
        d, _a = tm.depth_poses(Ref.gR[:, hands], Ref.gx[:, hands], T[:, :3, :3], T[:, :3, 3])
        assert out[0]["sides"][side] == round(float(d.max()), 4)


def test_a_hull_is_built_once_per_vertex_values(monkeypatch):
    """TriModel reads each hand hull's Parts from one per-process cache keyed by the vertex values: equal vertices
    share one Parts, its values are a fresh build's, and a full cache starts over."""
    monkeypatch.setattr(HC, "_HULLS", {})
    rng = np.random.default_rng(11)
    tri, hulls = _rack(), _hulls(rng, 3)
    a, b = HC.TriModel(hulls, tri), HC.TriModel([h.copy() for h in hulls], tri)
    assert all(x is y for x, y in zip(a.hulls, b.hulls, strict=True)) and len(HC._HULLS) == 3
    for x, v in zip(a.hulls, hulls, strict=True):
        y = CS.Parts([np.asarray(v, float)])
        assert np.array_equal(x.points, y.points) and np.array_equal(x.center, y.center) and x.radius == y.radius
    monkeypatch.setattr(HC, "HULL_CACHE_MAX", 3)
    HC.TriModel(_hulls(rng, 1), tri)
    assert len(HC._HULLS) == 1
