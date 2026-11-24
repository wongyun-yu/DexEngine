"""S-hand-clear (CPU, MuJoCo): the depth of a hand hull inside a still extra's piece is measured; a recorded overlap at
or below scene_requirement.TRUE_SHAPE_OVERLAP_M is clear, a deeper one reads `need` (measured only: no move is
searched)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from dexengine.data.objects import hand_clear as HC  # noqa: E402
from dexengine.data.objects import scene_requirement as SR  # noqa: E402
from tests.data.gjk_oracle import GjkModel  # noqa: E402


def _cube(half: float) -> np.ndarray:
    return np.array([(x, y, z) for x in (-half, half) for y in (-half, half) for z in (-half, half)], float)


def _sunk(depth: float) -> GjkModel:
    """A 2 cm hand cube sunk `depth` into the top of a 10 cm static cube, placed."""
    pm = GjkModel([_cube(0.01)], [_cube(0.05)])
    pm.place(np.eye(3)[None], np.array([[0.0, 0.0, 0.05 + 0.01 - depth]]), np.eye(3), np.zeros(3))
    return pm


def test_the_ruler_reads_the_overlap():
    assert _sunk(0.015).depth()[0] == pytest.approx(0.015, abs=1e-4)


@pytest.mark.parametrize(("depth", "verdict"), [(0.004, "clear"), (0.015, "need"), (0.03, "need")])
def test_the_verdict_follows_the_true_shape_overlap(depth, verdict):
    assert HC.judge(_sunk(depth).depth()[0]) == verdict


def test_the_threshold_is_read_from_its_owner(monkeypatch):
    """Negative: with a 2 cm threshold the 15 mm case is clear — the rule reads scene_requirement, it holds no copy."""
    depth = _sunk(0.015).depth()[0]
    assert HC.judge(depth) == "need"
    monkeypatch.setattr(SR, "TRUE_SHAPE_OVERLAP_M", 0.02)
    assert HC.judge(depth) == "clear"


def test_no_move_search_is_left():
    """Rule C is measured only: the move search and its cap are gone (re-adding one is a rule change, not a fix)."""
    assert not any(hasattr(HC, n) for n in ("least_move", "row_need", "STEPS_M", "dropped"))
    assert not hasattr(SR, "HAND_MOVE_OUT_MAX_M") and HC.VERDICTS == ("clear", "need")
