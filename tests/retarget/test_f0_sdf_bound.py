"""f0 object projection contact rows read through carried lower bounds (Projection._touch_min) give the rows of a
full evaluation (every vertex against every plane, the oracle here) — the same link, vertex and normal, the residual
within 1e-12 m — over a walk of small and large moves, and evaluate fewer vertices; also when the object rotation is
built from a quaternion that is not unit (plane normals longer than 1: the bound scales by them)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.retarget.test_f0_palm_offset import _hand, _pose


def _proj(h, wp, wR, indep, rng, oR):
    from dexengine.retarget.f0 import repair_obj as RO
    from dexengine.retarget.f0.repair_obj_solve import Projection

    pieces = [wp + rng.normal(scale=0.03, size=3) + rng.uniform(-0.02, 0.02, size=(30, 3)) for _ in range(4)]
    cobj = RO.CookedObject(pieces)
    touch = {ln for ln, _f, _p, _d in h.links}
    return Projection(h, cobj, oR, np.zeros(3), (wp, wR, indep), -10.0, (indep - 1, indep + 1), touch)


def _full_rows(P, T, W):
    """The contact rows of a full evaluation (every vertex of every touching link against every plane) — the oracle
    the carried bound must reproduce."""
    from dexengine.retarget.f0 import repair_obj as RO

    out = []
    touch_t = RO.OBJ_TOUCH_MM / 1000.0
    for li in P.touch_idx:
        R, t = T[li]
        hv = P._link_verts(T, li, W)
        s, kp, vals = P.sdf(hv)
        k = int(s.argmin())
        if s[k] > touch_t:
            out.append((s[k] - touch_t, li, (hv[k] - t) @ R, P.normal(vals, kp, k)))
    for li in sorted({lj for lj, _v, _e in P.pieces}):
        R, t = T[li]
        hv = P._link_verts(T, li, W)
        k = int(hv[:, 2].argmin())
        if hv[k, 2] < P.floor:
            out.append((P.floor - hv[k, 2], li, (hv[k] - t) @ R, np.array([0.0, 0.0, -1.0])))
    return out


@pytest.mark.parametrize("qnorm", [1.0, 1.02])
def test_bound_rows_equal_full_rows(monkeypatch, qnorm):
    h = _hand(monkeypatch, "sharpa_wave")
    wp, wR = _pose(1)
    indep = np.random.default_rng(5).uniform(-0.2, 0.2, len(h.jnames))
    from dexengine.retarget.f0.repair_math import _quat_wxyz_to_R

    q = np.array([0.9, 0.1, -0.3, 0.2])
    oR = _quat_wxyz_to_R(q / np.linalg.norm(q) * qnorm)  # as f0 builds it: a quaternion not unit scales the normals
    ref = _proj(h, wp, wR, indep, np.random.default_rng(7), oR)
    on = _proj(h, wp, wR, indep, np.random.default_rng(7), oR)
    assert on._lb == {}
    seen = [0, 0]
    real = on.sdf

    def counted(x):
        seen[0] += len(x)
        return real(x)

    on.sdf = counted
    rng = np.random.default_rng(11)
    nd = 6 + len(indep)
    dq = np.zeros(nd)
    n_rows = 0
    for step in range(240):
        dq = dq + rng.normal(scale=0.05 if step % 40 == 39 else 0.002, size=nd)
        T = ref.link_T(dq)
        W = ref.world(T)
        a = _full_rows(ref, T, W)
        b = on._contact_desk_rows(T, W)
        seen[1] += sum(len(ref._link_verts(T, li, W)) for li in ref.touch_idx)
        assert len(a) == len(b)
        for (r1, l1, x1, n1), (r2, l2, x2, n2) in zip(a, b, strict=True):
            assert l1 == l2 and np.array_equal(x1, x2) and np.array_equal(n1, n2)
            assert abs(r1 - r2) <= 1e-12
        n_rows += len(a)
    assert n_rows > 100  # the walk keeps contact rows to compare
    assert seen[0] < 0.6 * seen[1]  # the bound skips vertices
