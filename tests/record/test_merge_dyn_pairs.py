"""S-dyn-pairs drop (b) in the final merge (store.merge_dyn_pairs) on synthetic pieces: two boxes stated as a pair-off
pair, the delivered poses overlapping at frame 20 by a planted depth (tests.record.merge_fixture `dyn`). Needs pyarrow,
ffmpeg and zstd.

  a trajectory planted 6 mm leaves with all its looks, the pieces line records it and verify passes; 4 mm keeps every
  episode · a piece with every trajectory over keeps its line · each distinct pose table is read once · a line
  without the statement fails the merge · verify refuses an edited drop record · the store's line statement names the
  scene's pair by its delivered columns, and no tree raises.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
if not (shutil.which("ffmpeg") and shutil.which("zstd")):
    pytest.skip("ffmpeg and zstd are needed for synthetic pieces", allow_module_level=True)

from dexengine.data.objects import collider_shape as CS  # noqa: E402
from dexengine.record.store import merge as CMG  # noqa: E402
from dexengine.record.store import merge_check as MC  # noqa: E402
from dexengine.record.store import merge_dyn_pairs as DPM  # noqa: E402
from tests.record.merge_fixture import DYN_BOXES, Src, dies, make_piece, measure, merge, verify  # noqa: E402

X, Y = "x_take_pl_rt", "y_take_pl_rt"
PV1 = X.replace("_pl_rt", "_pv01_pl_rt")


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    return Src(tmp_path_factory.mktemp("dyn_src"))


def _boxes(monkeypatch):
    """The visual mesh of each fixture box (collider_shape.triangles stand-in)."""
    import trimesh

    def tri(*a):
        m = trimesh.creation.box(extents=np.full(3, 2 * DYN_BOXES[a[1]]))
        return CS.Triangles(np.asarray(m.vertices, float), np.asarray(m.faces))

    monkeypatch.setattr(CS, "triangles", tri)


def _piece(root: Path, src: Src, dyn: dict) -> Path:
    """Episodes 0 X (look D) · 1 X_pv01 · 2 X (look r2) · 3 Y: trajectories X (0, 2) · X_pv01 (1) · Y (3)."""
    return make_piece(root, src, [(X, 3, "task x"), (Y, 1, "task y")], dyn=dyn)


def _final(tmp: Path) -> Path:
    return tmp / "orig" / "data" / "hot3d" / "collect" / "rby1_test"


def test_a_trajectory_over_the_line_leaves_with_all_its_looks(tmp_path, src, capsys, monkeypatch):
    _boxes(monkeypatch)
    fin = _final(tmp_path)
    out = merge(fin, _piece(tmp_path / "p", src, {2: 6.0, 1: 4.0}), "p1")
    eps = MC.jsonl(fin / MC.EPISODES)
    assert [e["clip"] for e in eps] == [PV1, Y] and out["dyn_pair_dropped"] == 1
    rec = MC.jsonl(fin / MC.PIECES)[-1]
    (d,) = rec["dyn_pair_dropped"]
    assert (d["clip"], d["piece_eps"], d["rule"], d["statement"]["frame"]) == (X, [0, 2], DPM.RULE, 20)
    assert abs(d["statement"]["depth_m"] - 0.006) < 2e-4 and verify(fin, frames=False)["ok"]
    log = capsys.readouterr().out
    assert f"[DYN_PAIRS] output {X}: object_right×object_left 6.0 mm frame 20 -> drop" in log
    assert f"[DYN_PAIRS] output {PV1}: object_right×object_left ≤ 5.0 mm" in log and f"{PV1}" in log
    assert "[DYN_PAIRS_SUM] p1 — trajectories 3 · dropped 1 · episodes kept 2 / 4" in log


def test_four_mm_keeps_every_episode(tmp_path, src, monkeypatch):
    _boxes(monkeypatch)
    fin = _final(tmp_path)
    out = merge(fin, _piece(tmp_path / "p", src, {2: 4.0, 3: 4.0}), "p1")
    assert out["dyn_pair_dropped"] == 0 and out["episodes"] == [0, 4]
    assert "dyn_pair_dropped" not in MC.jsonl(fin / MC.PIECES)[-1] and verify(fin, frames=False)["ok"]


def test_a_piece_with_every_trajectory_over_still_gets_its_line(tmp_path, src, monkeypatch):
    """Every trajectory over the line: the pieces line stays (no episodes, the drop records), verify passes, and the
    next piece chains on."""
    _boxes(monkeypatch)
    fin = _final(tmp_path)
    out = merge(fin, _piece(tmp_path / "p", src, dict.fromkeys(range(4), 6.0)), "p1")
    assert out["episodes"] == [0, 0] and out["dyn_pair_dropped"] == 3
    rec = MC.jsonl(fin / MC.PIECES)
    assert [(r["piece_id"], r["episodes"], len(r["dyn_pair_dropped"])) for r in rec] == [("p1", [0, 0], 3)]
    assert MC.jsonl(fin / MC.EPISODES) == [] and verify(fin, frames=False)["ok"]
    merge(fin, make_piece(tmp_path / "q", src, [("z_take_pl_rt", 1, "task z")], dyn={}), "p2")  # same features
    assert [r["episodes"] for r in MC.jsonl(fin / MC.PIECES)] == [[0, 0], [0, 1]] and verify(fin, frames=False)["ok"]


def test_each_pose_table_is_read_once(tmp_path, src, monkeypatch):
    _boxes(monkeypatch)
    calls = []
    real = DPM._table
    monkeypatch.setattr(DPM, "_table", lambda *a: calls.append(1) or real(*a))
    from dexengine.data.objects import dyn_pair_check as DC

    reads = []
    real_oc = DC.output_check
    monkeypatch.setattr(DC, "output_check", lambda *a, **k: reads.append(1) or real_oc(*a, **k))
    got = DPM.measure(CMG.read_piece(_piece(tmp_path / "p", src, {})))
    assert len(calls) == 4 and len(reads) == 1  # four episode files, one distinct pose table
    assert all(res["verdict"] == DC.KEEP for res, _ks in got.values())


def test_a_line_without_the_statement_fails_the_merge(tmp_path, src):
    p = _piece(tmp_path / "p", src, {})
    lines = MC.jsonl(p / MC.EPISODES)
    for e in lines:
        e.pop(DPM.LINE_KEY)
    (p / MC.EPISODES).write_text("".join(json.dumps(e) + "\n" for e in lines))
    msg = dies(measure, p, want="stored before S-dyn-pairs")
    assert msg is not None and "stored before S-dyn-pairs" in msg


def test_verify_refuses_an_edited_drop_record(tmp_path, src, monkeypatch):
    _boxes(monkeypatch)
    fin = _final(tmp_path)
    merge(fin, _piece(tmp_path / "p", src, {0: 6.0, 2: 6.0}), "p1")
    lines = MC.jsonl(fin / MC.PIECES)
    lines[-1]["dyn_pair_dropped"][0]["statement"]["depth_m"] = 0.004
    (fin / MC.PIECES).write_text("".join(json.dumps(r) + "\n" for r in lines))
    assert any("not over the line" in x for x in DPM.line_problems(lines[-1]))


def test_the_line_statement_names_the_scenes_pair_by_its_delivered_columns(tmp_path, monkeypatch):
    """The tree's one pair-off pair (dyn_pairs.scene_pair) becomes object_<side> columns with its objects' meshes; the
    tree is read once (cached); no tree raises."""
    from dexengine.data.objects import dyn_pairs as DP
    from dexengine.record import tree_ref as TR

    t = tmp_path / "c_pl_rt"
    (t / "0").mkdir(parents=True)
    objs = [
        {"name": "supp_object_2", "role": "main", "side": "right", "obj_id": "A", "mesh_dir": "a"},
        {"name": "left_object", "role": "main", "side": "left", "obj_id": "B", "mesh_dir": "b"},
    ]
    (t / "task_info.json").write_text(json.dumps({"dataset_name": "hot3d", "objects": objs}))
    np.savez(t / "0" / "trajectory_kinematic.npz", qpos=np.zeros((3, 14)))
    reads = []
    monkeypatch.setattr(TR, "meta_tree", lambda *a: (t, ""))
    pair = [{"a": "left_object", "b": "supp_object_2", "f0_m": 0.004}]
    monkeypatch.setattr(DP, "off_pairs", lambda *a: reads.append(1) or pair)
    DPM._tree_statement.cache_clear()
    st = DPM.line_statement({"env": {"sequence_dir": "x"}}, "c_pl_rt")
    assert st["pairs"] == [{"a": "object_left", "b": "object_right", "f0_m": 0.004}] and st["dataset"] == "hot3d"
    assert st["objects"]["object_right"] == {
        "name": "supp_object_2",
        "obj_id": "A",
        "mesh_dir": "a",
        "articulation": None,
    }
    assert DPM.line_statement({"env": {"sequence_dir": "x"}}, "c_pl_rt") == st and len(reads) == 1
    monkeypatch.setattr(TR, "meta_tree", lambda *a: (None, "gone"))
    with pytest.raises(ValueError, match="no tree"):
        DPM.line_statement({"env": {}}, "c_pl_rt")
