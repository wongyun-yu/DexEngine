"""Robot self-penetration of a delivered trajectory (J-self-pen, owner placement.collision.self_pen): the verdict layer
on planted ruler rows, the per-frame measurement on a stand-in ruler, the recorded joint table read by name, the rule
card, and the neutral pose on the real ruler when a data root is given (DEXENGINE_MAIN_ROOT=<root> pytest ...)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from dexengine.placement.collision import self_pen as SP

REAL_DATA_ROOT = os.environ.get("DEXENGINE_MAIN_ROOT", "")
NAMES = [f"j{i}" for i in range(7)]


def _row(cls: str, depth_mm: float, a: str = "a", b: str = "b") -> tuple:
    """One ruler row (set, class, link a, link b, signed distance m, fromto) with `depth_mm` of penetration."""
    s = "4_torso" if "torso" in cls else "1_cross"
    return (s, cls, a, b, -depth_mm / 1000.0, np.zeros(6))


class _Ruler:
    """A stand-in ruler model: frame t returns the rows planted for t (none for the others)."""

    def __init__(self, rows: dict):
        self.rows, self.t = rows, None

    def set_pose(self, arm_q, finger_q, obj_tree, base):
        self.t = int(arm_q["right"][1][0])  # the planted arm value carries the frame number

    def distances(self, distmax, sets):
        assert sets == SP.RULER_SETS and distmax == SP.QUERY_DISTMAX_M
        return {}, {}, self.rows.get(self.t, [])


def _measure(rows: dict, T: int = 6) -> SP.Result:
    q = np.repeat(np.arange(T, dtype=float)[:, None], 7, axis=1)
    arm = dict.fromkeys(SP.SIDES, (NAMES, q))
    fingers = {s: np.zeros((T, 3)) for s in SP.SIDES}
    return SP.measure(_Ruler(rows), arm, fingers, np.zeros(3))


def test_a_clean_trajectory_is_kept():
    res = _measure({2: [_row("hand-hand", 0.0)], 4: [_row("arm-arm", -3.0)]})  # touching and 3 mm apart
    assert (res.depth_m, res.frame, res.frames) == (0.0, -1, 6)
    assert SP.verdict(res) == "keep" and SP.reason(res) is None


def test_a_planted_25mm_arm_torso_drops():
    res = _measure({3: [_row("Rarm-torso", 25.0, "link_right_arm_2", "link_torso_5"), _row("arm-arm", 1.0)]})
    assert SP.verdict(res) == "drop"
    assert (res.cls, res.a, res.b, res.frame) == ("AT", "link_right_arm_2", "link_torso_5", 3)
    assert SP.reason(res) == "self_pen25.00mm(link_right_arm_2|link_torso_5@3)"
    line = "[SELF_PEN] c1_pl_rt max 25.00 pair link_right_arm_2|link_torso_5 frame 3 -> drop"
    assert SP.marker("c1_pl_rt", res) == line


def test_a_planted_25mm_hand_hand_is_kept():
    res = _measure({1: [_row("hand-hand", 25.0, "right_thumb_tip", "left_index_tip")]})
    assert SP.verdict(res) == "keep" and res.depth_m == 0.0


def test_15mm_is_kept():
    res = _measure({5: [_row("arm-arm", 15.0)], 2: [_row("hand-arm", 12.0)]})
    assert SP.verdict(res) == "keep" and res.cls == "AA" and res.frame == 5


def test_counting_hand_hand_flips_the_hand_hand_case(monkeypatch):
    """Negative of the hand-hand exclusion: with HH counted, the hand-hand test above would fail."""
    monkeypatch.setattr(SP, "COUNTED", (*SP.COUNTED, "HH"))
    assert SP.verdict(_measure({1: [_row("hand-hand", 25.0)]})) == "drop"


def test_the_deepest_counted_pair_and_its_first_frame_win():
    res = _measure({1: [_row("hand-arm", 21.0)], 2: [_row("arm-hand", 30.0, "x", "y")], 4: [_row("arm-hand", 30.0)]})
    assert (res.depth_m, res.cls, res.a, res.frame) == (pytest.approx(0.030), "HA", "x", 2)


def test_every_ruler_class_has_a_pair_class_and_an_unknown_one_raises():
    assert set(SP.CLASS_OF.values()) == {"AA", "AT", "HT", "HA", "HH"} and set(SP.COUNTED) < set(SP.CLASS_OF.values())
    with pytest.raises(ValueError, match="no pair class"):
        SP.worst([_row("hand-obj", 30.0)])


def test_joint_arrays_of_different_lengths_raise():
    arm = {"right": (NAMES, np.zeros((5, 7))), "left": (NAMES, np.zeros((4, 7)))}
    with pytest.raises(ValueError, match="different lengths"):
        SP.measure(_Ruler({}), arm, {s: np.zeros((5, 3)) for s in SP.SIDES}, np.zeros(3))


def test_the_rule_card_reads_its_values_from_the_owner():
    from dexengine.core.rules import rule as RR

    card = next(r for r in RR.resolve() if r.id == "J-self-pen")
    assert card.stage == "final merge" and card.resolved["drop line (m)"] == SP.ROBOT_SELF_PEN_MAX_M
    assert card.resolved["counted pair classes"] == SP.COUNTED and card.resolved["ruler sets"] == SP.RULER_SETS


class _Layout:
    finger_names = {"right": ("rf0", "rf1"), "left": ("lf0", "lf1")}
    urdf = "robot.urdf"


#: The stand-in model's movable joints (model order): arms, head, fingers.
MOVABLE = ("r0", "r1", "head_0", "head_1", "rf0", "rf1", "l0", "l1", "lf0", "lf1")


def _ruler_seeing(seen: dict):
    class Ruler(_Ruler):
        jq = dict.fromkeys(MOVABLE, 0)

        def set_pose(self, body_q, finger_q, obj_tree, base):
            seen.setdefault("body", (body_q["right"][0], body_q["right"][1].tolist(), body_q["left"][0]))
            seen.setdefault("fingers", finger_q["left"].tolist())
            self.t = 0

    return Ruler({0: [_row("arm-arm", 25.0)]})


def test_every_movable_joint_is_read_by_name(monkeypatch):
    """Columns are found by name in any order; the head rides with the arms (no neutral fill)."""
    seen = {}
    monkeypatch.setattr(SP, "RobotLayout", lambda arm, hand: _Layout())
    monkeypatch.setattr(SP, "model_for", lambda layout: _ruler_seeing(seen))
    names = list(reversed(MOVABLE))
    Q = np.arange(2 * len(names), dtype=float).reshape(2, -1)
    res = SP.measure_joints("rby1_wuji2", names, Q)
    assert SP.verdict(res) == "drop" and res.frames == 2
    body = ["r0", "r1", "head_0", "head_1", "l0", "l1"]
    assert seen["body"] == (body, [Q[0, names.index(n)] for n in body], [])
    assert seen["fingers"] == [Q[0, names.index(n)] for n in ("lf0", "lf1")]


@pytest.mark.parametrize("edit", ["drop head_1", "add torso_0"])
def test_a_missing_or_unmovable_recorded_joint_raises(monkeypatch, edit):
    monkeypatch.setattr(SP, "RobotLayout", lambda arm, hand: _Layout())
    monkeypatch.setattr(SP, "model_for", lambda layout: _ruler_seeing({}))
    names = [n for n in MOVABLE if n != "head_1"] if edit == "drop head_1" else [*MOVABLE, "torso_0"]
    with pytest.raises(KeyError, match="no neutral fill") as e:
        SP.measure_joints("rby1_wuji2", names, np.zeros((2, len(names))))
    assert ("missing ['head_1']" if edit == "drop head_1" else "not movable ['torso_0']") in str(e.value)


@pytest.mark.skipif(not REAL_DATA_ROOT, reason="needs the data root (DEXENGINE_MAIN_ROOT=<root> pytest ...)")
def test_the_neutral_pose_on_the_real_ruler_is_kept(monkeypatch):
    """Every movable joint at zero (the default pose) has no counted penetration on the canon ruler; the movable
    joints besides the fingers are both arms and the head (the torso is fixed in the URDF)."""
    monkeypatch.setenv("DEXENGINE_MAIN_ROOT", REAL_DATA_ROOT)
    monkeypatch.setenv("RETARGET_PY", sys.executable)
    lay = SP.RobotLayout("rby1", "wuji2")
    body = SP.body_joints(SP.model_for(lay), lay)
    assert sorted(n for n in body if "_arm_" not in n) == ["head_0", "head_1"] and len(body) == 16
    names = [*body, *lay.finger_names["right"], *lay.finger_names["left"]]
    res = SP.measure_joints("rby1_wuji2", names, np.zeros((3, len(names))))
    assert (SP.verdict(res), res.depth_m, res.frames) == ("keep", 0.0, 3)
