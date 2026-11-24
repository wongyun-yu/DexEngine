"""The TACO lift names the slot files from scene.xml (right_visual = qpos slot 0, the tool; left_visual = slot
1, the target), not from task_info's measured grasp (which hand holds which object). On a left-hand-tool clip the two
differ; naming the files from the grasp paired each mesh with the other object's trajectory.
"""

from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import pytest

from dexengine.core.errors import InputMissingError
from dexengine.data import side_objects
from dexengine.data.readers import taco_lift as L
from dexengine.data.readers import taco_source

CLIP, TRIP, SID = "pour-in-some_bowl_cup__20231006_277", "(pour in some, bowl, cup)", "20231006_277"


def _scene(right: str, left: str) -> str:
    return (
        "<mujoco><asset>\n"
        f'    <mesh name="right_visual" file="objects/{right}/visual.obj"/>\n'
        f'    <mesh name="left_visual" file="objects/{left}/visual.obj"/>\n'
        "</asset></mujoco>\n"
    )


@pytest.fixture
def world(tmp_path):
    """A V1 tree of a left-hand-tool clip (scene slots 180 · 187, grasp R=187 · L=180) and its raw hand poses."""
    tdir = tmp_path / "v1" / CLIP
    (tdir / "0").mkdir(parents=True)
    (tdir / "scene.xml").write_text(_scene("180", "187"))
    info = {
        "source_ref_dt": L.SRC_DT,
        "right_object_mesh_dir": "processed/taco/assets/objects/187",
        "left_object_mesh_dir": "processed/taco/assets/objects/180",
        "side_assign": "grasp[persist R(0.00,0.92) L(0.91,0.00)]",
    }
    (tdir / "task_info.json").write_text(json.dumps(info))
    qpos = np.zeros((10, 50))
    qpos[:, 36:43] = [0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # slot 0: the tool (180)
    qpos[:, 43:50] = [0.0, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0]  # slot 1: the target (187)
    np.savez(tdir / "0" / "trajectory_kinematic.npz", qpos=qpos, frequency=50.0)
    hp = tmp_path / "raw" / "Hand_Poses" / TRIP / SID
    hp.mkdir(parents=True)
    for side in ("right", "left"):
        with open(hp / f"{side}_hand.pkl", "wb") as f:
            pickle.dump({i: {"hand_trans": np.zeros(3, np.float32)} for i in range(6)}, f)
        (hp / f"{side}_hand_shape.pkl").write_bytes(b"shape")
    a = argparse.Namespace(
        trajectories_root=tmp_path / "v1",
        source_kind=taco_source.V1_TREE,
        real_raw=tmp_path / "raw",
        shim_raw_dir=tmp_path / "shim",
    )
    return a, qpos


def test_slot_files_follow_the_scene_not_the_grasp(world, capsys):
    a, qpos = world
    assert L._lift_sequence(a, CLIP, TRIP, SID) == ("180", "187")
    od = a.shim_raw_dir / "Object_Poses" / TRIP / SID
    assert sorted(p.name for p in od.glob("*.grid50.npy")) == ["target_187.grid50.npy", "tool_180.grid50.npy"]
    assert np.array_equal(np.load(od / "tool_180.grid50.npy"), qpos[:, 36:43])  # mesh 180 carries slot 0
    assert np.array_equal(np.load(od / "target_187.grid50.npy"), qpos[:, 43:50])
    sa = json.loads((od / "side_assign.json").read_text())
    assert (sa["right"], sa["right_slot"], sa["left"], sa["left_slot"]) == ("187", 1, "180", 0)  # the grasp stays
    assert f"[TACO_MESH] scene≠task_info clip={CLIP} slots(tool,target)=(180,187)" in capsys.readouterr().out


def test_no_marker_when_the_grasp_is_the_scene_order(world, capsys):
    a, _ = world
    tdir = a.trajectories_root / CLIP
    info = json.loads((tdir / "task_info.json").read_text())
    info["right_object_mesh_dir"], info["left_object_mesh_dir"] = "objects/180", "objects/187"
    (tdir / "task_info.json").write_text(json.dumps(info))
    assert L.slot_mesh_ids(taco_source.read(a.trajectories_root, taco_source.V1_TREE, CLIP), CLIP) == ("180", "187")
    assert "[TACO_MESH]" not in capsys.readouterr().out


def test_a_scene_without_both_visual_meshes_dies(tmp_path):
    (tmp_path / "scene.xml").write_text('<mesh name="right_visual" file="objects/180/visual.obj"/>')
    with pytest.raises(InputMissingError, match="right_visual and left_visual"):
        side_objects.scene_visual_ids(tmp_path)


def test_v1_tree_whose_scene_order_is_not_the_visuals_dies(world):
    a, _ = world
    tdir = a.trajectories_root / CLIP
    xml = _scene("180", "187").replace(
        '<mesh name="right_visual"', '<mesh name="x" file="objects/187/visual.obj"/>\n<mesh name="right_visual"'
    )
    (tdir / "scene.xml").write_text(xml)
    with pytest.raises(SystemExit, match="not \\(right_visual, left_visual\\)"):
        taco_source.read(a.trajectories_root, taco_source.V1_TREE, CLIP)
