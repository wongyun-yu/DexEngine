"""Triangles.depths / within / signed read the nearest-face search only where its value sets the output, and
measure a face several samples share once: their outputs equal the full reading bit for bit (the reference below is
the reading before, every point searched over all K_NEAR candidates), at the boundaries the shortcut uses — d1 == r,
d1 == r + cover, points on, just inside and just outside the surface, and depths that must read +0.0."""

from __future__ import annotations

import numpy as np
import pytest

from dexengine.data.objects import collider_shape as CS
from dexengine.data.objects import triangle_geom as TGM

trimesh = pytest.importorskip("trimesh")


def _nearest_ref(T, P):
    k = min(CS.K_NEAR, len(T.points))
    dk, idx = (x.reshape(len(P), k) for x in T.tree.query(P, k=k))
    Tr = T.V[T.F[T.sample_face[idx]]]
    Q = TGM.closest_point(np.repeat(P[:, None], k, 1), Tr[..., 0, :], Tr[..., 1, :], Tr[..., 2, :])
    return np.linalg.norm(P[:, None] - Q, axis=-1).min(axis=1), dk[:, 0]


def _signed_ref(T, P):
    d, d1 = _nearest_ref(T, P)
    return np.where(T.inside(P, d1), -d, d)


def _depths_ref(T, P):
    out = np.zeros(len(P))
    inb = np.flatnonzero(np.all((T.lo < P) & (T.hi > P), axis=1))
    if len(inb):
        out[inb] = np.maximum(-_signed_ref(T, P[inb]), 0.0)
    return out


def _within_ref(T, P, r):
    out, d1 = np.zeros(len(P), bool), np.full(len(P), np.nan)
    near = np.flatnonzero(np.all((T.lo - r - 1e-9 <= P) & (T.hi + r + 1e-9 >= P), axis=1))
    d, d1[near] = _nearest_ref(T, P[near])
    out[near[d <= r]] = True
    rest = np.flatnonzero(~out)
    out[rest] = T.inside(P[rest], d1[rest])
    return out


def _meshes():
    box = trimesh.creation.box(extents=(0.06, 0.05, 0.04))
    ring = trimesh.creation.annulus(r_min=0.012, r_max=0.05, height=0.04)
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.03)
    cup = trimesh.Trimesh(sphere.vertices, sphere.faces[sphere.triangles_center[:, 2] < 0.012])  # open: no lid
    return [CS.Triangles(np.asarray(m.vertices), np.asarray(m.faces)) for m in (box, ring, sphere, cup)]


def _points(T, rng):
    """Random points round the body, surface points (on faces), and surface points 1e-12 m in and out."""
    P = [rng.uniform(T.lo - 0.01, T.hi + 0.01, (400, 3))]
    f = rng.integers(0, len(T.F), 200)
    w = rng.dirichlet(np.ones(3), 200)
    S = np.einsum("nk,nkd->nd", w, T.V[T.F[f]])
    P += [S, S + 1e-12 * T.normals[f], S - 1e-12 * T.normals[f], T.points[rng.integers(0, len(T.points), 50)]]
    return np.concatenate(P)


@pytest.mark.parametrize("mesh", range(4))
def test_depths_signed_and_nearest_equal_the_full_reading(mesh):
    T, rng = _meshes()[mesh], np.random.default_rng(mesh)
    P = _points(T, rng)
    d, d1 = T._nearest(P)
    rd, rd1 = _nearest_ref(T, P)
    assert d.tobytes() == rd.tobytes() and d1.tobytes() == rd1.tobytes()
    assert T.tree.query(P, k=1)[0].tobytes() == rd1.tobytes()  # the k=1 search gives the same nearest distance
    assert T.signed(P).tobytes() == _signed_ref(T, P).tobytes()
    got, ref = T.depths(P), _depths_ref(T, P)
    assert got.tobytes() == ref.tobytes()  # sign bits included
    far = (ref == 0) & (rd1 > T._reach(0.0))
    assert far.sum() > 50 and not np.signbit(got[far]).any()  # skipped points read +0.0
    assert (ref > 0).sum() > 20  # inside points are measured (the test is not vacuous)


@pytest.mark.parametrize("mesh", range(4))
def test_within_equals_the_full_reading_at_the_band_edges(mesh):
    T, rng = _meshes()[mesh], np.random.default_rng(10 + mesh)
    P = _points(T, rng)
    d1 = T.tree.query(P, k=1)[0]
    i, j, k = np.flatnonzero(d1 > T._reach(0.0) + 1e-4)[:3]  # points far enough out for r = d1 - cover >= 0
    rs = [0.0, 1e-6, 0.0006, 0.003, T.cover, float(d1[i]), float(d1[j]) - T.cover, float(d1[k]) - T._reach(0.0)]
    assert min(rs) >= 0 and len(set(rs)) == len(rs)  # d1 == r, d1 == r + cover, d1 == _reach(r): every edge is read
    for r in rs:
        assert np.array_equal(T.within(P, r), _within_ref(T, P, r)), r


@pytest.mark.parametrize("reach", [lambda r: r - 1.0, lambda r: r + CS.REACH_EPS_M], ids=["no_band", "no_cover"])
def test_a_band_too_narrow_is_caught(monkeypatch, reach):
    """Negative: with the band cut short (none, or without the cover term) a point whose face is within r but whose
    nearest sample is farther reads out."""
    T = _meshes()[0]
    P = _points(T, np.random.default_rng(3))
    r = 1e-6
    ref = _within_ref(T, P, r)
    monkeypatch.setattr(T, "_reach", reach)
    assert not np.array_equal(T.within(P, r), ref)
