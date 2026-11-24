"""Arm-flip gate and replace conditions (dexengine.placement.arm_flip) — numeric checks without IK or torch (CPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dexengine.placement import arm_flip as AF


def _ramp(deg: float, start_s: float, dur_s: float) -> np.ndarray:
    """Four-window trajectory of 7 joints where joint 4 turns `deg` degrees linearly from start_s over dur_s seconds."""
    n = 4 * AF.window_frames()
    t = np.arange(n) * AF.FRAME_DT  # seconds
    q = np.zeros((n, 7))
    q[:, 4] = np.clip((t - start_s) / dur_s, 0.0, 1.0) * math.radians(deg)
    return q


#: measurements of a candidate that meets every replace condition (thresholds: error 30 mm · stretch 3.0)
OK = {"travel": 1.0, "err_mm": 10.0, "rot_deg": 2.0, "stretch": 1.2, "excess": 2.5}


def test_gate_fast_turn_hits_slow_or_small_does_not():
    assert AF.window_frames() == int(round(AF.WINDOW_S / AF.FRAME_DT))
    assert AF.triggered(_ramp(200.0, 2.0, 1.0)) is True  # 200° in 1 s
    assert AF.triggered(_ramp(200.0, 1.0, 3.0)) is False  # 200° over 3 s
    assert AF.triggered(_ramp(170.0, 2.0, 1.0)) is False  # 170° in 1 s


def test_window_travel_span_is_the_hit_joint_and_its_travel():
    q = _ramp(200.0, 2.0, 1.0)
    tr, jn, fa, fb = AF.window_travel(q)
    assert jn == 4
    assert tr == pytest.approx(math.radians(200.0))
    assert abs(q[fb, 4] - q[fa, 4] - tr) < 1e-9
    assert AF.window_travel(q[:1]) == (0.0, 0, 0, 0)


def test_accept_names_the_first_failed_condition():
    for change, why in [
        ({}, ""),
        ({"travel": math.pi + 0.1}, AF.WHY_FLIP),
        ({"err_mm": 31.0}, AF.WHY_ERR),
        ({"rot_deg": 9.0}, AF.WHY_ROT),
        ({"stretch": 3.1}, AF.WHY_STRETCH),
        ({"excess": 1.5}, AF.WHY_EXCESS),
    ]:
        assert AF.accept({**OK, **change}, 30.0, 3.0) == (why == "", why), change


def test_gate_not_hit_returns_the_same_object():
    q = _ramp(170.0, 2.0, 1.0)
    out = AF.fix(
        None,
        None,
        q,
        err_mm_fn=None,
        err_mm_max=30.0,
        stretch_fn=None,
        stretch_max=3.0,
        polish_iters=8,
        tag="test",
        torch=None,
    )
    assert out[0] is q and out[1:] == (None, None, None)


def test_unresolved_and_feasible_count():
    # no candidate continues within the thresholds → unresolvable
    assert (
        AF.unresolved({"verdict": AF.VERDICT_KEPT, "feasible": AF.feasible_count(0, {AF.WHY_FLIP: 3, AF.WHY_ERR: 2})})
        is True
    )
    # a rotation that follows the hand (within thresholds, arm travel did not shrink) is not dropped
    assert AF.feasible_count(0, {AF.WHY_EXCESS: 1, AF.WHY_FLIP: 5}) == 1
    assert AF.unresolved({"verdict": AF.VERDICT_KEPT, "feasible": AF.feasible_count(0, {AF.WHY_EXCESS: 1})}) is False
    # replaced · gate not hit → not dropped
    assert AF.unresolved({"verdict": AF.VERDICT_REPLACED, "feasible": 0}) is False
    assert AF.unresolved(None) is False


def test_bake_gate():
    bad = {"verdict": AF.VERDICT_KEPT, "feasible": 0, "travel": math.radians(200.0)}
    clean, flipped = _ramp(170.0, 2.0, 1.0), _ramp(200.0, 2.0, 1.0)
    assert AF.bake_gate("dp", "dp", bad, clean) is True
    assert AF.bake_gate("dp", "dp", bad, None) is True
    assert AF.bake_gate("dp", "dp", bad, flipped) is False  # the judge trajectory already flips
    assert AF.bake_gate("dp", "reuse", bad, clean) is False
    assert AF.bake_gate("off", "dp", bad, clean) is False
    assert AF.bake_gate("dp", "dp", None, clean) is False


class _StillIK:
    """Stand-in arm IK for `fix`: every DLS call returns its seeds unchanged with zero error, so each tracked seed is a
    still trajectory (no arm travel over the hit span)."""

    n = 7

    def __init__(self, torch):
        self.torch = torch
        self.lo = -4.0 * torch.ones(self.n, dtype=torch.float64)
        self.hi = 4.0 * torch.ones(self.n, dtype=torch.float64)

    def _dls(self, T, iters, damp, w, Q):
        z = self.torch.zeros(Q.shape[0], dtype=Q.dtype)
        return Q, z, z, None


@pytest.mark.parametrize("err_mm, verdict, dropped", [(0.0, "VERDICT_REPLACED", False), (1e9, "VERDICT_KEPT", True)])
def test_fix_verdict_reaches_unresolved(err_mm, verdict, dropped):
    """Writer `fix` → reader `unresolved`: a still candidate replaces the flip; with every candidate over the error cap
    the flip is kept and the judge drops it."""
    torch = pytest.importorskip("torch")
    q = _ramp(200.0, 2.0, 1.0)
    Tt = torch.eye(4, dtype=torch.float64).expand(len(q), 4, 4).contiguous()
    out = AF.fix(
        _StillIK(torch),
        Tt,
        q,
        err_mm_fn=lambda e, qq: err_mm,
        err_mm_max=30.0,
        stretch_fn=lambda qq: 1.0,
        stretch_max=3.0,
        polish_iters=1,
        tag="test",
        torch=torch,
    )
    info = out[3]
    assert info["verdict"] == getattr(AF, verdict)
    assert (out[0] is q) is dropped
    assert AF.unresolved(info) is dropped
    # with no feasible count left only the verdict word decides: the reader must know the writer's value
    assert AF.unresolved(dict(info, feasible=0)) is dropped
