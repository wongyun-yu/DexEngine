"""Base-tree writing, generation declarations, HOT3D GT recordings and piece lists (file-level helpers)."""

import json
import os

import pytest

from dexengine.data import align, tree_io
from dexengine.data.readers import hot3d_gt
from dexengine.data.segment import todo


def test_write_json_if_changed_keeps_mtime(tmp_path):
    f = tmp_path / "task_info.json"
    assert tree_io.write_json_if_changed(f, {"a": 1, "b": "ü"})
    assert f.read_bytes() == json.dumps({"a": 1, "b": "ü"}, ensure_ascii=False, indent=1).encode()
    os.utime(f, (1, 1))
    assert not tree_io.write_json_if_changed(f, {"a": 1, "b": "ü"})
    assert f.stat().st_mtime == 1
    assert tree_io.write_json_if_changed(f, {"a": 2})
    assert tree_io.tree_tmp(tmp_path / "clip_t0").name == "clip_t0.tmp"
    assert tree_io.tree_tmp_glob("clip") == "clip*.tmp"


def test_generation_declaration(tmp_path):
    intake = tmp_path / "INTAKE"
    f = align.decl_path(intake)
    assert f == tmp_path / align.GENERATION_FILE
    assert align.read_decl(f) is None
    f.write_text("note=x\n")
    with pytest.raises(ValueError):
        align.read_decl(f)
    f.write_text("generation=g1\nhand_dir_ref_deg=119.91\n")
    assert align.read_decl(f) == {"generation": "g1", "hand_dir_ref_deg": "119.91"}


def _rec(root, name, flag, pose_bytes):
    d = root / name
    d.mkdir()
    (d / "metadata.json").write_text(json.dumps({} if flag is None else {hot3d_gt.GT_KEY: flag}))
    (d / hot3d_gt.POSE_FILE).write_bytes(pose_bytes)


def test_hot3d_ground_truth_recordings(tmp_path):
    _rec(tmp_path, "P0001_a", True, b"{}\n")
    _rec(tmp_path, "P0002_b", False, b"")
    (tmp_path / "other").mkdir()
    assert hot3d_gt.ground_truth_recordings(tmp_path) == ["P0001_a"]
    _rec(tmp_path, "P0003_c", True, b"")  # flag says GT, pose file empty
    with pytest.raises(ValueError, match="inconsistent recordings 1"):
        hot3d_gt.ground_truth_recordings(tmp_path)
    assert hot3d_gt.main([str(tmp_path)]) == 2


def test_piece_list_rows(tmp_path):
    f = tmp_path / "units.tsv"
    f.write_text("hot3d\tbimanual\tc_t0\n\nhot3d\tright\tc_t1\textra\n")
    assert todo.read_units(str(f)) == [("hot3d", "bimanual", "c_t0"), ("hot3d", "right", "c_t1")]
    f.write_text("hot3d\tbimanual\n")
    with pytest.raises(SystemExit):
        todo.read_units(str(f))
