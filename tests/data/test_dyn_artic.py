"""S-dyn-pairs on an articulated main object (CPU): its visual body is its base and its moving link, the moving link
posed by the joint FK (data/artic/fk) at each frame's joint value; a lid that swings 6 mm into the other main object
drops the trajectory, 4 mm keeps it pair_off, and the same lid closed reads nothing. The reference and the delivered
output read the same way. A body that cannot be read raises (no hold)."""

from __future__ import annotations

import numpy as np
import pytest

from dexengine.data.artic.fk import artic_moving_vertices
from dexengine.data.objects import collider_shape as CS
from dexengine.data.objects import dyn_pair_check as DC
from dexengine.data.objects import dyn_pairs as DP
from dexengine.data.objects import extra_slots_geom as XG
from dexengine.data.objects import scene_rules as SC
from dexengine.data.objects import touch_gate as TG
from dexengine.rl.tasks.scene import collision_presets as CP

T = 30
BASE, LID, CUBE = (0.05, 0.05, 0.02), (0.05, 0.05, 0.005), (0.02, 0.02, 0.02)
#: The lid hinges about +y at the base's -x top edge; in its child frame it lies on the base top (theta = 0 closed).
HINGE = {"type": "hinge", "joint": "lid_hinge", "axis": [0.0, 1.0, 0.0], "pos": [-0.05, 0.0, 0.025]}
LID_AT = np.array([0.0, 0.0, BASE[2] + LID[2]])  # the lid's centre in the child frame


def _box(half, at=(0.0, 0.0, 0.0)):
    import trimesh  # noqa: PLC0415

    m = trimesh.creation.box(extents=2 * np.asarray(half, float))
    return CS.Triangles(np.asarray(m.vertices, float) + np.asarray(at, float), np.asarray(m.faces))


def _corners(half):
    return np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * np.asarray(half)


def test_fk_pose_is_the_fk_owners_map():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(20, 3)) * 0.05
    for art, th in ((HINGE, 0.7), ({**HINGE, "type": "slide"}, 0.03), ({**HINGE, "child_pose": _cp()}, -0.4)):
        T4 = DC.fk_pose(art, th)
        assert np.allclose(P @ T4[:3, :3].T + T4[:3, 3], artic_moving_vertices(art, P, th), atol=1e-12)


def _cp():
    return {"pos": [0.01, -0.02, 0.003], "quat_wxyz": [0.9659258, 0.0, 0.2588190, 0.0]}


def _q(theta, pen):
    """Object A (articulated, columns 0..6, joint column 14) at the origin; cube B (7..13) lying on the lid opened to
    theta, `pen` into its top face (the lid's orientation; the lid is open on frames 10..19, closed elsewhere, where
    the cube floats above it); the cube drifts 10 mm along y, so the two do not move together."""
    from scipy.spatial.transform import Rotation as R  # noqa: PLC0415

    L = DC.fk_pose(HINGE, theta)
    c = L[:3, :3] @ (LID_AT + [0.0, 0.0, LID[2] + CUBE[2] - pen]) + L[:3, 3]
    w = R.from_matrix(L[:3, :3]).as_quat()
    q = np.zeros((T, 15))
    q[:, 3] = 1.0
    q[:, 7:10], q[:, 10:14] = c, [w[3], w[0], w[1], w[2]]
    q[:, 8] += np.linspace(0.0, 0.01, T)
    q[10:20, 14] = theta
    return q


def _ti():
    return {
        "dataset_name": "oakink2",
        "objects": [
            {"name": "left_object", "role": "main", "side": "left", "qpos_start": 7, "mesh_dir": "b", "obj_id": "B"},
            {
                "name": "right_object",
                "role": "main",
                "side": "right",
                "qpos_start": 0,
                "mesh_dir": "a",
                "obj_id": "A",
                "articulation": dict(HINGE, parts={"base": "base.obj", "moving": "lid.obj"}),
            },
        ],
    }


THETA = -0.6  # the lid opens (rotation about +y with the hinge at -x lifts its free edge)


@pytest.fixture
def scene(monkeypatch):
    """Visual: base + lid triangles (link frames) and the cube; collision: the base hull only at frame 0 (the cube sits
    on the lid's line, inside the base's collision box by 3 mm at frame 0 so the pair exists)."""
    links = [_box(BASE), _box(LID, LID_AT)]
    monkeypatch.setattr(SC, "on", lambda rid, ds: rid == DP.RULE_ID)
    monkeypatch.setattr(DC, "_link_triangles", lambda ds, o: links)
    monkeypatch.setattr(CS, "triangles", lambda ds, oid: _box(CUBE))
    big = _corners((0.2, 0.2, 0.2))  # collision boxes that overlap at frame 0: the pair is a frame-0 pair
    monkeypatch.setattr(TG, "body_of", lambda ds, o, tree: (TG.Body([CS.Parts([big])]), ""))
    monkeypatch.setattr(DP, "separated", lambda tree, q: set())  # the pair is kept off (F0-1d frees nothing here)

    class QL:
        @staticmethod
        def read(tree, nq=None):
            return type("L", (), {"artic_joint_cols": staticmethod(lambda: {"lid_hinge": 14})})()

    import dexengine.core.paths as P

    monkeypatch.setattr(P, "qpos_layout", QL, raising=False)
    return links


@pytest.mark.parametrize(("pen", "verdict"), [(0.006, DP.DROP), (0.004, CP.DYN_PAIR_OFF)])
def test_the_opened_lid_reads_its_overlap(scene, pen, verdict):
    q = _q(THETA, pen)
    st = DP.dyn_pairs("/x/clip_pl_rt", _ti(), q, rows=[])
    (p,) = st["pairs"]
    assert p["verdict"] == verdict
    if verdict == DP.DROP:
        assert 10 <= p["visual_frame"] < 20 and abs(p["visual_m"] - pen) < 6e-4
        assert st["drop"][0] == DC.DROP_WORD


def test_the_closed_lid_reads_nothing(scene):
    q = _q(THETA, 0.006)
    q[:, 14] = 0.0  # never opened: the cube stays above the closed lid
    (p,) = DP.dyn_pairs("/x/clip_pl_rt", _ti(), q, rows=[])["pairs"]
    assert p["verdict"] == CP.DYN_PAIR_OFF and not DC.exact(p["visual_m"])


@pytest.mark.parametrize(("pen", "verdict"), [(0.006, DC.DROP), (0.004, DC.KEEP)])
def test_the_delivered_output_reads_the_lid_by_its_joint_values(scene, pen, verdict):
    q, objs = _q(THETA, pen), {o["name"]: o for o in _ti()["objects"]}
    poses = {"right_object": XG.pose_mats(q, 0), "left_object": XG.pose_mats(q, 7)}
    res = DC.output_check("oakink2", objs, [("right_object", "left_object")], poses, {"right_object": q[:, 14]})
    assert res["pairs"][0]["verdict"] == verdict and (res["drop"] is not None) == (verdict == DC.DROP)
    with pytest.raises(KeyError, match="no joint values"):
        DC.output_check("oakink2", objs, [("right_object", "left_object")], poses)


def test_an_articulated_ruler_needs_its_tree():
    with pytest.raises(ValueError, match="tree"):
        DC.ruler("oakink2", _ti()["objects"][1])


def test_a_main_object_without_its_collision_body_raises(scene, monkeypatch):
    monkeypatch.setattr(TG, "body_of", lambda ds, o, tree: (None, "no cooked parts"))
    with pytest.raises(ValueError, match="no collision body"):
        DP.dyn_pairs("/x/clip_pl_rt", _ti(), _q(THETA, 0.006), rows=[])
