"""OakInk2 part grouping (rule D1-g) on synthetic parts: fixed merge, hinge fit and kind check, the joint-kinds table,
and the lever rest-pose rule. No dataset file is read: the official part tree is replaced by a small synthetic one and
the meshes are written into the test's temporary folder."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from dexengine.core import artifact_paths
from dexengine.data.oakink2 import artic_fit as AF
from dexengine.data.oakink2 import attach as AT
from dexengine.data.oakink2 import fixed_merge as FM
from dexengine.data.oakink2 import part_kinds as PK
from dexengine.data.oakink2 import parts as OP

T = 120
LAMP = "O02@0045@00003"  # kinds table: body 00001 + deformable head 00002
LAPTOP = "O02@0053@00003"  # kinds table: body 00002 + hinged lid 00001
MICROWAVE = "O02@0039@00003"  # kinds table: body 00001 + hinged door 00002 + lever 00004


def _tree(monkeypatch, tree: dict) -> None:
    """Replace the official part tree (object_part_tree.json · part_desc.json · instance_id.json) with `tree`."""
    rev = {c: p for p, cs in tree.items() for c in cs}
    desc = {r: {"obj_name": f"part {r}"} for r in rev}
    monkeypatch.setattr(OP, "_load", lambda: (tree, rev, desc, set(tree)))


@pytest.fixture
def part_tree(monkeypatch):
    _tree(
        monkeypatch,
        {
            LAMP: ["O02@0045@00001", "O02@0045@00002"],
            LAPTOP: ["O02@0053@00001", "O02@0053@00002"],
            MICROWAVE: ["O02@0039@00001", "O02@0039@00002", "O02@0039@00004"],
        },
    )


def _q7(p, Rs) -> np.ndarray:
    """Tree free-joint block: [x y z, qw qx qy qz] (wxyz)."""
    return np.hstack([p, Rs.as_quat()[:, [3, 0, 1, 2]]])


def _obj(name: str, obj_id: str, k: int) -> dict:
    return {"name": name, "obj_id": obj_id, "qpos_start": 7 * k}


def _base_motion():
    """A base body that moves and turns in the world (so every relative pose has to be taken in its frame)."""
    t = np.linspace(0.0, 1.0, T)
    p = np.stack([0.3 + 0.2 * t, -0.1 * t, 0.8 + 0.05 * np.sin(3 * t)], 1)
    return t, p, R.from_rotvec(np.outer(t, [0.4, -0.9, 0.6]))


def _write_obj(path, V, F) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"v {x} {y} {z}\n" for x, y, z in V) + "".join(f"f {a} {b} {c}\n" for a, b, c in F))


def test_fixed_merge_one_mesh_parts_moved_by_relative_pose(part_tree, tmp_path):
    rng = np.random.default_rng(0)
    _t, pb, Rb = _base_motion()
    R_rel, p_rel = R.from_rotvec([0.7, 0.2, -1.1]), np.array([0.05, -0.12, 0.31])  # non-symmetric relative pose
    Rh = Rb * R_rel * R.from_rotvec(rng.normal(0.0, 1e-4, (T, 3)))
    ph = pb + Rb.apply(p_rel) + rng.normal(0.0, 1e-4, (T, 3))
    q = np.hstack([_q7(pb, Rb), _q7(ph, Rh)])
    body, head = _obj("left_object", "raw_o02_0045_00001", 0), _obj("supp_object_0", "raw_o02_0045_00002", 1)

    dec = AF.part_clusters([body, head], q)
    assert dec["status"] == "fixed_only"
    rec = FM.record(dec["clusters"][0], q)
    assert (rec["root"], rec["body"], list(rec["members"])) == (LAMP, "O02@0045@00001", ["O02@0045@00002"])
    pose = rec["members"]["O02@0045@00002"]["pose"]
    R_pose = R.from_quat(np.r_[pose["q"][1:], pose["q"][0]])  # stored wxyz
    np.testing.assert_allclose(pose["p"], p_rel, atol=5e-4)
    assert np.degrees((R_pose.inv() * R_rel).magnitude()) < 0.05
    assert FM.rigid(rec)
    aid = FM.asset_id(rec)
    assert FM.is_asset_id(aid) and aid.startswith("raw_o02_0045_00003_f")

    objects = tmp_path / "objects"
    Vb, Fb = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]), [(1, 2, 3)]
    Vh, Fh = np.array([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.07]]), [(1, 2, 3), (1, 2, 4)]
    _write_obj(objects / "raw_o02_0045_00001" / "visual.obj", Vb, Fb)
    _write_obj(objects / "raw_o02_0045_00002" / "visual.obj", Vh, Fh)
    sp = {
        **rec,
        "members": {r: {"obj": m["obj"], "pose": m["pose"]} for r, m in rec["members"].items()},
        "ref_clip": "synthetic",
        "band": {"pos_m": AF.POS_TOL_M, "rot_deg": AF.ROT_TOL_DEG},
    }
    vis = FM.build_mesh(objects, aid, sp)

    assert sorted(p.name for p in (objects / aid).iterdir()) == ["FIXED.json", "visual.obj"]
    V, F = FM._obj_vf(vis)
    np.testing.assert_allclose(V, np.vstack([Vb, R_pose.apply(Vh) + pose["p"]]), atol=1e-6)  # OBJ_DIGITS = 6
    assert F == [[0, 1, 2], [3, 4, 5], [3, 4, 6]]  # part faces offset by the body
    fx = json.loads((objects / aid / "FIXED.json").read_text())
    assert (fx["asset"], fx["body"], fx["n_vert"], fx["n_face"]) == (aid, "O02@0045@00001", 7, 3)
    assert fx["members"]["O02@0045@00002"]["pose"] == pose
    moved = {"O02@0045@00002": {**sp["members"]["O02@0045@00002"], "pose": {"p": [0, 0, 0], "q": [1, 0, 0, 0]}}}
    with pytest.raises(ValueError, match="differs"):  # the asset id names one pose: no silent rewrite
        FM.build_mesh(objects, aid, {**sp, "members": moved})


def test_hinge_fit_recovers_axis_pivot_range_and_kind_check(part_tree):
    t, pb, Rb = _base_motion()
    n = np.array([0.3, -0.5, 0.81]) / np.linalg.norm([0.3, -0.5, 0.81])
    c = np.array([0.12, -0.07, 0.05])  # pivot in the base frame, off the origin
    alpha = np.radians(70.0) * (0.5 - 0.5 * np.cos(np.pi * t))  # 0 → 70°
    R0, p0 = R.from_rotvec([0.1, 0.4, -0.2]), np.array([0.2, 0.1, -0.05])
    Ra = R.from_rotvec(np.outer(alpha, n))
    qA = _q7(pb, Rb)
    qB = _q7(pb + Rb.apply(c + Ra.apply(p0 - c)), Rb * Ra * R0)

    fit = AF.fit_pair(qA, qB)
    assert fit["kind"] == "revolute"
    s = np.sign(np.dot(fit["axis"], n))
    assert abs(np.dot(fit["axis"], n)) == pytest.approx(1.0, abs=1e-9)
    d = np.asarray(fit["pivot"]) - c  # the pivot is fixed only along the axis line
    assert np.linalg.norm(d - np.dot(d, n) * n) < 1e-6
    assert fit["angle_range_deg"] == pytest.approx(70.0, abs=1e-6)
    np.testing.assert_allclose(s * AF.joint_values(qA, qB, fit), alpha - alpha[0], atol=1e-9)

    base, lid = _obj("left_object", "raw_o02_0053_00002", 0), _obj("supp_object_0", "raw_o02_0053_00001", 1)
    dec = AF.part_clusters([base, lid], np.hstack([qA, qB]))
    assert (dec["status"], dec["joint_kind"], dec["fit"]["kind"]) == ("two_clusters", "revolute", "revolute")
    still = AF.part_clusters([base, lid], np.hstack([qA, _q7(pb + Rb.apply(p0), Rb * R0)]))
    assert still["status"] == "one_cluster"  # lid did not move: the joint comes from the spec
    slid = _q7(pb + Rb.apply(p0 + np.outer(0.2 * t, [1.0, 0.0, 0.0])), Rb * R0)
    with pytest.raises(AF.PartExcluded) as e:  # a lid that slides is not the table's hinge
        AF.part_clusters([base, lid], np.hstack([qA, slid]))
    assert e.value.code == "kind_mismatch"


def test_the_toothpaste_cap_is_a_hinge_and_merges_with_its_tube(monkeypatch):
    """The flip cap turns on a hinge on the tube (the base trees fit revolute, about 108° in the flip clips, and never
    free). As a free part the cap and the tube stayed two rigid bodies and the cap read as carried by the tube."""
    tube, cap = "O02@0003@00005", "O02@0003@00004"
    _tree(monkeypatch, {"O02@0003@00006": [cap, tube]})
    assert PK.kind_of(cap) == ("revolute", tube) and PK.merges(cap)
    t, pb, Rb = _base_motion()
    alpha = np.radians(108.0) * t
    c, R0, p0 = np.array([0.02, 0.06, -0.04]), R.from_rotvec([0.0, 0.2, 0.0]), np.array([0.0, 0.08, 0.0])
    Ra = R.from_rotvec(np.outer(alpha, [0.93, -0.37, 0.01]))
    qA, qB = _q7(pb, Rb), _q7(pb + Rb.apply(c + Ra.apply(p0 - c)), Rb * Ra * R0)
    objs = [_obj("right_object", "right_o02_0003_00005", 0), _obj("left_object", "left_o02_0003_00004", 1)]
    dec = AF.part_clusters(objs, np.hstack([qA, qB]))
    assert (dec["status"], dec["joint_kind"], dec["fit"]["kind"]) == ("two_clusters", "revolute", "revolute")


def test_unknown_part_kind_fails_loudly(monkeypatch):
    _tree(monkeypatch, {LAMP: ["O02@0045@00001", "O02@0045@00002", "O02@0045@00009"]})
    with pytest.raises(ValueError, match="has no joint kind"):
        PK.kind_of("O02@0045@00009")
    with pytest.raises(ValueError, match="not an official part"):
        PK.kind_of("O02@9999@00001")
    assert "not in the table O02@0045@00009(part O02@0045@00009)" in PK.check()
    _t, pb, Rb = _base_motion()
    q = np.hstack([_q7(pb, Rb)] * 3)
    objs = [_obj("a", "raw_o02_0045_00001", 0), _obj("b", "raw_o02_0045_00002", 1), _obj("c", "raw_o02_0045_00009", 2)]
    with pytest.raises(ValueError, match="has no joint kind"):  # never taken silently as an independent object
        AF.part_clusters(objs, q)
    with pytest.raises(ValueError, match="no verification band"):
        AF.band("screw")
    with pytest.raises(ValueError, match="only"):
        artifact_paths.object_asset_id(
            "raw_o02_0053_00002", {"type": "screw", "parts": {"moving_id": "raw_o02_0053_00001"}}
        )


def _lever_clip(press_deg: float, door_frames: slice):
    """Microwave: body moving in the world · door hinged on the body, opening over `door_frames` · lever on the body,
    pressed by `press_deg` about its own hinge over frames 0-14 (frame 0 is NOT at rest) and 70-79."""
    t, pb, Rb = _base_motion()
    press = np.zeros(T)
    press[:15] = press[70:80] = 1.0
    rest_R, rest_p = R.from_rotvec([0.05, -0.02, 0.3]), np.array([0.05, -0.15, 0.08])
    Rl_rel = rest_R * R.from_rotvec(np.outer(np.radians(press_deg) * press, [0.0, 1.0, 0.0]))
    door = np.zeros(T)
    door[door_frames] = np.linspace(0.0, 1.0, len(range(T)[door_frames]))
    door[door_frames.stop :] = 1.0
    Rd_rel = R.from_rotvec(np.outer(door, [0.0, 0.0, 1.0]))
    q = np.hstack(
        [
            _q7(pb, Rb),
            _q7(pb + Rb.apply(Rd_rel.apply([0.15, 0.0, 0.05])), Rb * Rd_rel),
            _q7(pb + Rb.apply(np.tile(rest_p, (T, 1))), Rb * Rl_rel),
        ]
    )
    return q, press, rest_R, np.tile(rest_p, (T, 1)), Rl_rel


LEVER_PTS = np.array([[0.0, 0.0, 0.0], [0.08, 0.0, 0.0], [0.08, 0.01, 0.005], [0.0, 0.01, 0.0]])  # hinge at the origin


def test_lever_rest_pose_attach_rule(part_tree, monkeypatch):
    q, press, rest_R, prel, Rrel = _lever_clip(10.0, slice(30, 60))
    rest, dev = AT.lever_rest(Rrel)
    assert rest != 0 and press[rest] == 0 and dev[rest] == pytest.approx(0.0, abs=1e-9)
    _p, R_att = AT.attach_pose(prel, Rrel, lever=True, away_deg=AF.ROT_TOL_DEG)
    assert np.degrees((R_att.inv() * rest_R).magnitude()) < 1e-6  # mean over rest frames only
    _p, R_all = AT.attach_pose(prel, Rrel, lever=False, away_deg=AF.ROT_TOL_DEG)
    assert np.degrees((R_all.inv() * rest_R).magnitude()) > 1.0  # a whole-clip mean is pulled by the presses
    tip = 2.0 * np.sin(np.radians(5.0)) * np.hypot(0.08, 0.005)  # farthest vertex from the lever hinge
    lm = AT.lever_motion(prel, Rrel, LEVER_PTS, None, AF.ROT_TOL_DEG)
    assert lm["tip_m"] == pytest.approx(tip, rel=1e-9) and AT.lever_in_band(lm["tip_m"], lm["drive_share"])

    monkeypatch.setattr(AT, "part_points", lambda o: LEVER_PTS)  # the lever mesh, without a dataset file
    objs = [
        _obj("left_object", "raw_o02_0039_00001", 0),
        _obj("supp_object_0", "raw_o02_0039_00002", 1),
        _obj("supp_object_1", "raw_o02_0039_00004", 2),
    ]
    dec = AF.part_clusters(objs, q)  # one press, door opens with the lever at rest
    assert dec["status"] == "two_clusters"
    assert [o["obj_id"] for o in dec["clusters"][0]["fixed"]] == ["raw_o02_0039_00004"]  # lever fixed to the body
    assert dec["levers"] == [{"obj_id": "raw_o02_0039_00004", "tip_m": round(tip, 4), "drive_share": 0.0}]

    q_far, *_ = _lever_clip(40.0, slice(30, 60))  # pulled far: tip beyond LEVER_TIP_M
    assert 2.0 * np.sin(np.radians(20.0)) * np.hypot(0.08, 0.005) > PK.LEVER_TIP_M
    q_push, *_ = _lever_clip(10.0, slice(70, 80))  # door opens only while the lever is turned
    for qx in (q_far, q_push):
        with pytest.raises(AF.PartExcluded) as e:
            AF.part_clusters(objs, qx)
        assert e.value.code == PK.ATTACH_MOVED["lever"]
