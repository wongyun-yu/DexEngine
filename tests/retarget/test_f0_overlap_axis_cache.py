"""repair_obj_solve.Projection._overlap_rows reads a pair's last separating axis first (self._sep): over a sequence
of hand placements the rows are those of a full reading at every step (a fresh projection, empty cache), and the
cache skips pairs. Synthetic boxes; no data."""

from __future__ import annotations

import numpy as np

from tests.retarget.test_f0_palm_offset import _hand


def _box(c, h):
    """(vertices, plane equations n·x + d <= 0 inside) of an axis box centred at c with half sizes h."""
    c, h = np.asarray(c, float), np.asarray(h, float)
    v = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c
    n = np.vstack([np.eye(3), -np.eye(3)])
    d = -(n @ c) - np.r_[h, h]
    return v, np.c_[n, d]


class _Rec(dict):
    """The axis cache, counting full readings (each one stores its pair's axis)."""

    n = 0

    def __setitem__(self, k, v):
        self.n += 1
        super().__setitem__(k, v)


def _proj(hand, obj):
    from dexengine.retarget.f0.repair_obj_solve import Projection

    P = object.__new__(Projection)
    P.pieces = [(0, v, e) for v, e in hand]
    P.obj = [(v, e, v.min(0), v.max(0)) for v, e in obj]
    P.olo = np.array([o[2] for o in P.obj])
    P.ohi = np.array([o[3] for o in P.obj])
    P._sep = _Rec()
    return P


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def test_rows_equal_full_reading_and_skips(monkeypatch):
    _hand(monkeypatch, "sharpa_wave")  # binds the hand configuration the f0 modules read at import
    hand = [_box([0, 0, 0], [0.01, 0.01, 0.02]), _box([0.0, 0.0, 0.03], [0.008, 0.008, 0.01])]
    obj = [_box([0.03, 0, 0], [0.015, 0.02, 0.02]), _box([-0.05, 0, 0.0], [0.01, 0.01, 0.01])]
    r_t = np.array([0.001, 0.002])
    P = _proj(hand, obj)
    full = 0
    rng = np.random.default_rng(0)
    for k in range(60):
        t = np.array([0.0005 * k, 0.0, 0.0]) + rng.normal(0, 2e-4, 3)  # slides into the first object piece
        T = [(_rz(0.01 * k), t)]
        got = P._overlap_rows(T, r_t)
        F = _proj(hand, obj)
        ref = F._overlap_rows(T, r_t)  # empty cache: every pair read in full
        full += F._sep.n
        assert len(got) == len(ref)
        for a, b in zip(got, ref, strict=True):
            assert a[0] == b[0] and a[1] == b[1]
            assert np.array_equal(a[2], b[2]) and np.array_equal(a[3], b[3])
    assert 0 < P._sep.n < full  # some pairs were skipped, some read in full
