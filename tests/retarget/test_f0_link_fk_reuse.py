"""Projection.link_T keeps the palm-relative link placements of the last finger configuration: a sequence of dq with
wrist-only moves (the Jacobian's wrist columns), finger moves and signed zeros gives the bits of the link_T before the
reuse (its body copied here as the reference: FK on every call), and a wrist-only move runs no FK."""

from __future__ import annotations

import numpy as np

from tests.retarget.test_f0_palm_offset import _hand, _pose


def _link_T_before(P, dq):
    """Projection.link_T as it was before the reuse: FK and the frame updates on every call."""
    import pinocchio as pin

    from dexengine.retarget.f0.repair_obj_solve import _aa_to_R

    s = P.s
    q = pin.neutral(s.model)
    for jn, val in zip(s.jnames, s.fingers12(P.indep0 + dq[6:]), strict=True):
        q[s.jidx[jn]] = val
    pin.forwardKinematics(s.model, s.data, q)
    pin.updateFramePlacements(s.model, s.data)
    Tb_inv = s.palm_inv()
    wR = _aa_to_R(dq[3:6]) @ P.wR0
    wp = P.wpos0 + dq[:3]
    out = []
    for fid in P.fids:
        rel = Tb_inv * s.data.oMf[fid]
        out.append((wR @ rel.rotation, wR @ rel.translation + wp))
    return out


def test_reused_fk_gives_the_fresh_bits(monkeypatch):
    import pinocchio as pin

    h = _hand(monkeypatch, "sharpa_wave")
    from dexengine.retarget.f0 import repair_obj as RO
    from dexengine.retarget.f0.repair_obj_solve import EPS, Projection

    wp, wR = _pose(2)
    rng = np.random.default_rng(3)
    indep = rng.uniform(-0.2, 0.2, len(h.jnames))
    cobj = RO.CookedObject([np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float) + 5.0])
    P = Projection(h, cobj, np.eye(3), np.zeros(3), (wp, wR, indep), -1.0, (indep - 1, indep + 1), set())
    nd = 6 + len(indep)
    base = rng.normal(scale=0.01, size=nd)
    seq = [base] + [base + EPS * np.eye(nd)[k] for k in range(nd)] + [rng.normal(scale=0.01, size=nd), base.copy()]
    z = base.copy()
    z[6] = -0.0
    seq += [z, z.copy(), np.zeros(nd)]
    calls = []
    real = pin.forwardKinematics
    monkeypatch.setattr(pin, "forwardKinematics", lambda *a: calls.append(1) or real(*a))
    for dq in seq:  # one cache through the whole sequence, as the solve uses it
        got = P.link_T(dq)
        ref = _link_T_before(P, dq)
        for (R1, t1), (R2, t2) in zip(got, ref, strict=True):
            assert np.array_equal(R1, R2) and np.array_equal(t1, t2)
            assert R1.tobytes() == R2.tobytes() and t1.tobytes() == t2.tobytes()
    P._fk = (None, None)
    P.link_T(base)
    n0 = len(calls)
    for k in range(6):
        P.link_T(base + EPS * np.eye(nd)[k])
    assert len(calls) == n0  # the wrist columns run no FK
    P.link_T(base + EPS * np.eye(nd)[6])
    assert len(calls) == n0 + 1
