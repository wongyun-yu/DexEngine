"""The stage-1 reuse rule: a solve is reused only when the recorded input fingerprint equals the current one."""

from __future__ import annotations

import json

from dexengine.retarget import input_fp as RFP

DS, ROBOT, EMB, TASK = "fxds", "fxhand", "right", "t1"


def _stage(tmp_path):
    dd = tmp_path / "stage"
    mano = dd / "processed" / DS / "mano" / EMB / TASK
    obj = dd / "processed" / DS / "assets" / "objects" / "o1"
    rig = tmp_path / "rig"
    for d in (mano / "0", obj / "convex", rig):
        d.mkdir(parents=True)
    o1 = f"processed/{DS}/assets/objects/o1"
    (mano / "task_info.json").write_text(
        json.dumps(
            {
                "right_object_mesh_dir": o1,
                "left_object_mesh_dir": None,
                "robot": f"processed/{DS}/assets/robots/{ROBOT}",
                "name": "x",
            }
        )
    )
    (mano / "0" / "trajectory_keypoints.npz").write_bytes(b"k0")
    (obj / "convex" / "0.obj").write_text("v 0 0 0\n")
    (rig / "right.xml").write_text("<mujoco/>")
    td = dd / "processed" / DS / ROBOT / EMB / TASK
    td.mkdir(parents=True)
    return dd, mano, obj, rig, td


def _cur(dd, rig, phase="generate-xml"):
    RFP._CACHE.clear()
    return RFP.current(dd, DS, ROBOT, EMB, TASK, phase, rig)


def test_tree_sha_absent_file_and_folder(tmp_path):
    assert RFP.tree_sha(tmp_path / "nope") == "absent"
    (tmp_path / "a").write_text("1")
    assert RFP.tree_sha(tmp_path / "a") == RFP._sha_file(tmp_path / "a")
    d = tmp_path / "d"
    d.mkdir()
    (d / "x").write_text("1")
    first = RFP.tree_sha(d)
    (d / "__pycache__").mkdir()
    (d / "__pycache__" / "y").write_text("2")  # ignored names do not count
    assert RFP.tree_sha(d) == first
    (d / "x").write_text("2")
    assert RFP.tree_sha(d) != first


def test_object_refs_are_existing_paths_without_the_rig(tmp_path):
    dd, mano, *_ = _stage(tmp_path)
    info = json.loads((mano / "task_info.json").read_text())
    assert RFP.object_refs(info, dd, DS) == [f"processed/{DS}/assets/objects/o1"]


def test_verdict_unknown_fresh_stale(tmp_path):
    dd, mano, obj, rig, td = _stage(tmp_path)
    cur = _cur(dd, rig)
    assert RFP.verdict(td, "generate-xml", cur) == "unknown"
    RFP.stamp(td, "generate-xml", cur)
    assert RFP.verdict(td, "generate-xml", _cur(dd, rig)) == "fresh"
    for change in (
        lambda: (mano / "0" / "trajectory_keypoints.npz").write_bytes(b"k1"),
        lambda: (obj / "convex" / "0.obj").write_text("v 1 0 0\n"),
        lambda: (rig / "right.xml").write_text("<mujoco model='b'/>"),
    ):
        change()
        assert RFP.verdict(td, "generate-xml", _cur(dd, rig)) == "stale"
        RFP.stamp(td, "generate-xml", _cur(dd, rig))


def test_forget_clears_the_phase_and_later_ones(tmp_path):
    dd, _mano, _obj, rig, td = _stage(tmp_path)
    RFP.stamp(td, "generate-xml", _cur(dd, rig))
    RFP.stamp(td, "ik", _cur(dd, rig, "ik"))
    RFP.forget(td, "ik")
    assert RFP.recorded(td, "ik") is None and RFP.recorded(td, "generate-xml") is not None
    RFP.forget(td, "generate-xml")
    assert not (td / RFP.RECORD).exists()
