"""Articulated-object FK, support thresholds and DexCanvas pose conventions (pure functions)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from dexengine.data.artic import fk
from dexengine.data.readers import dexcanvas_frames as DF
from dexengine.data.support import bands as B


def test_pose_points_identity_and_pose():
    V = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    assert fk.pose_points(V, None) is V
    pose = {"pos": [0.1, 0.0, 0.0], "quat_wxyz": [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]}  # Rz(90 deg)
    np.testing.assert_allclose(fk.pose_points(V, pose), [[0.1, 1.0, 0.0], [-1.9, 0.0, 0.0]], atol=1e-12)


def test_hinge_rotates_about_anchor_and_slide_translates():
    art = {"axis": [0, 0, 2.0], "pos": [1.0, 0.0, 0.0], "type": "hinge"}  # axis is normalised
    V = np.array([[2.0, 0.0, 0.0]])
    np.testing.assert_allclose(fk.artic_moving_vertices(art, V, np.pi / 2), [[1.0, 1.0, 0.0]], atol=1e-12)
    slide = {"axis": [0, 1.0, 0], "type": "slide"}
    np.testing.assert_allclose(fk.artic_moving_vertices(slide, V, 0.25), [[2.0, 0.25, 0.0]])


def test_child_pose_applied_after_joint():
    q = R.from_euler("x", 90, degrees=True).as_quat()[[3, 0, 1, 2]]
    art = {"axis": [0, 0, 1.0], "child_pose": {"pos": [0, 0, 1.0], "quat_wxyz": q.tolist()}}
    V = np.array([[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(fk.artic_moving_vertices(art, V, 0.0), [[0.0, 0.0, 2.0]], atol=1e-12)


def test_desk_height_rule_boundaries():
    """Ported from the old desk_height self-test: rest / middle / plane branches at the 10 and 20 mm boundaries."""
    from dexengine.data.support import desk_rule as DR

    tz = 0.004
    for drop_mm, want, desk in (
        (9.9, DR.DESK_REST, tz),
        (10.0, DR.DESK_REST, tz),
        (10.1, DR.DESK_MID, tz - 0.00505),
        (20.0, DR.DESK_MID, tz - 0.010),
        (20.1, DR.DESK_PLANE, 0.0),
    ):
        got = DR.desk_height(tz, round(tz - drop_mm / 1000, 4), True)
        assert got[1] == want and abs(got[0] - desk) < 6e-5, (drop_mm, got)
    assert DR.desk_height(tz, tz - 0.0201, False) == (tz, DR.DESK_KEPT)  # dataset without a measured plane
    assert DR.desk_height(tz, None, True) == (tz, DR.DESK_REST)  # not measured
    assert DR.desk_height(tz, tz + 0.05, True) == (tz, DR.DESK_REST)  # object higher than the desk
    assert B.SUPPORT_MIN_VERTS >= 1 and isinstance(B.PEDESTAL_RULE, str)


def test_dexcanvas_object_pose7():
    pose6 = np.array([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, np.pi / 2, 0.0, 0.0]])
    p7 = DF.object_pose7(pose6)
    np.testing.assert_allclose(p7[0], [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(np.linalg.norm(p7[:, 3:], axis=1), 1.0)
    assert DF.OBJECT_EULER == "XYZ"
    with pytest.raises(ValueError):
        DF.object_rotation(np.zeros((2, 2)))
