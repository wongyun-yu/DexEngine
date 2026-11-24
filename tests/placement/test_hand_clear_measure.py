"""Stage 2f (rule C) is measured only (CPU, no geometry): a side whose hand enters a still extra is KEPT_AS_RECORDED
with its max depth and stays in the chain (ACTION); a tree with a need row reaches the next stage byte-identical; a
tree that still holds a 2f repair output is not measured; the stage has no write path into a tree. The ruler
(hand_clear.clip_verdicts) is faked: this file tests the driver's decisions, not the geometry."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dexengine.data.objects import extra_slots as XS
from dexengine.data.objects import hand_clear as HC
from dexengine.placement.repair import hand_clear_stage as HS
from dexengine.placement.repair import wrist_shift as WS

T = 40
CLEAN, DROP, KEPT = HS.STATUSES
TI = {"dataset_name": "oakink2", "objects": [{"name": "o1", "qpos_start": 0}], "active_sides": ["right", "left"]}
STILL = [{"name": "o1"}]
STAGE = Path(HS.__file__)


def _v(right: float, left: float = 0.0) -> dict:
    """Row verdicts of one still row `o1` with these per-side depths (hand_clear.row_verdict's form)."""
    deep = max(right, left)
    v = HC.judge(deep)
    side = ("right" if right >= left else "left") if v == "need" else ""
    return {"o1": {"verdict": v, "depth_m": deep, "frame": 3, "side": side, "sides": {"right": right, "left": left}}}


def test_a_side_whose_hand_enters_a_still_is_kept_as_recorded_with_its_max_depth():
    st = HS.side_status(_v(0.012), "right")
    assert st["status"] == KEPT and st["depth_m"] == 0.012 and "kept as recorded" in st["why"]
    assert HS.side_status(_v(0.012), "left") == {"status": CLEAN, "why": "", "depth_m": 0.0}  # its own depth


def test_a_side_at_or_below_the_true_shape_overlap_is_clean():
    st = HS.side_status(_v(HS.SR.TRUE_SHAPE_OVERLAP_M), "right")
    assert st["status"] == CLEAN and st["depth_m"] == HS.SR.TRUE_SHAPE_OVERLAP_M


def test_the_action_table_keeps_the_measured_sides_and_excludes_clip_drops_only():
    assert HS.ACTION == {CLEAN: "keep", DROP: "exclude", KEPT: "keep"}
    assert set(HS.ACTION) == set(HS.STATUSES) and HS.STATUSES[:2] == ("CLEAN", "DROP")  # stage_row reads [0] · [1]


def test_only_clip_rows_exclude_and_they_keep_their_words():
    rows = [
        {"clip": "a", "side": "right", "status": KEPT, "why": "o1 0.012 m"},
        {"clip": "b", "side": "right", "status": KEPT, "why": "o1 0.012 m"},
        {"clip": "b", "side": HS.FS.ROW_SIDE, "status": DROP, "why": "o1 gap 31 mm"},
        {"clip": "c", "side": HS.OBJECTS, "status": DROP, "why": "overlap"},
        {"clip": "c", "side": HS.FS.ROW_SIDE, "status": DROP, "why": "o1 gap 25 mm"},
        {"clip": "d", "side": HS.OBJECTS, "status": CLEAN, "why": ""},
        {"clip": "d", "side": HS.FS.ROW_SIDE, "status": CLEAN, "why": ""},
    ]
    v = HS.drop_verdicts(rows)
    assert "a" not in v and "d" not in v  # kept as recorded stays in the chain
    assert v["b"] == (HS.FS.DROP_WORD, f"{HS.FS.ROW_SIDE}: o1 gap 31 mm")
    assert v["c"][0] == HS.CS.OVERLAP_DROP_WORD  # the overlap's word comes first


def test_a_hand_side_drop_has_no_reason_word_and_raises():
    with pytest.raises(ValueError, match="excluded with no reason word"):  # hand sides are measured or kept
        HS.drop_verdicts([{"clip": "b", "side": "left", "status": DROP, "why": "x"}])


def test_a_decision_to_drop_the_kept_sides_is_one_line(monkeypatch):
    monkeypatch.setitem(HS.ACTION, KEPT, "exclude")
    v = HS.drop_verdicts([{"clip": "a", "side": "right", "status": KEPT, "why": "w"}])
    assert v == {"a": (HC.KEPT_WORD, "right: w")}


def _tree(tmp_path: Path) -> Path:
    t = tmp_path / "clipA"
    (t / "0").mkdir(parents=True)
    rng = np.random.default_rng(0)
    np.savez(t / HS.KIN, qpos=rng.normal(size=(T, 60)), qvel=rng.normal(size=(T, 59)))
    np.savez(t / "0/trajectory_ikrollout.npz", qpos=rng.normal(size=(T, 60)))
    np.savez(t / "0/body_positions.npz", pos=rng.normal(size=(T, 5, 3)))
    (t / "task_info.json").write_text(json.dumps(TI))
    return t


def _hashes(t: Path) -> dict:
    return {
        str(f.relative_to(t)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(t.rglob("*")) if f.is_file()
    }


@pytest.fixture
def measured(monkeypatch):
    """tree_rows with the ruler faked: one still row `o1` with a 12 mm right-hand need, a resting float."""
    monkeypatch.setattr(XS, "clip_rows", lambda *a, **k: STILL)
    monkeypatch.setattr(XS, "slot_rows_of", lambda *a, **k: ([{"name": "o1", "kind": "receiver", "static": True}], {}))
    monkeypatch.setattr(HC, "clip_verdicts", lambda *a, **k: _v(0.012))
    st = {"o1": {"verdict": "rest", "gap_m": 0.001, "support": "desk", "first_touch": 9}}
    monkeypatch.setattr(HS.FS, "statement", lambda *a, **k: st)


def test_a_tree_with_a_need_row_reaches_the_next_stage_byte_identical(tmp_path, measured):
    """The tree is not touched (every file's bytes, no new file, no backup); the ledger row says the status with its
    reason word and the side's max depth; the clip gets no exclusion verdict."""
    t = _tree(tmp_path)
    before = _hashes(t)
    clip, rows, err = HS.tree_rows(str(t))
    assert err == "" and clip == "clipA" and _hashes(t) == before
    right = next(r for r in rows if r["side"] == "right")
    assert right["status"] == KEPT and right["rows"] == f"o1:need:0.012 max_depth_m:0.012 {HC.KEPT_WORD}"
    left = next(r for r in rows if r["side"] == "left")
    assert left["status"] == CLEAN and left["rows"] == "o1:clear:0.0 max_depth_m:0.0"
    assert HS.drop_verdicts(rows) == {}
    objects = next(r for r in rows if r["side"] == HS.OBJECTS)
    assert objects["rows"].startswith("statement:ts-")  # the clip's collider statement, as the env prints it
    fl = next(r for r in rows if r["side"] == HS.FS.ROW_SIDE)  # the float row of the clip, statement in `rows`
    assert fl["status"] == CLEAN and fl["rows"] == "o1:rest:0.001:desk:9" and fl["version"] == right["version"]


def test_a_tree_holding_a_2f_repair_output_is_not_measured(tmp_path, measured):
    """A tree that still holds a 2f repair output (wrist_shift tag) is an error, never read as the reference; the
    column does not count it current."""
    t = _tree(tmp_path)
    kin = t / HS.KIN
    with np.load(WS.source_of(kin, HS.TAG)) as z:  # plant an old 2f output: backup, shifted qpos, post sha
        q = np.asarray(z["qpos"]) + 0.01
    np.savez(kin, qpos=q)
    WS.mark_post(kin, HS.TAG)
    assert HS.holds_repair(t)
    clip, rows, err = HS.tree_rows(str(t))
    assert rows == [] and "2f repair output" in err
    assert not HS._current((HS.qpos_sha_file(kin), "v"), t, "v")


def _calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {ast.unparse(n.func).rsplit(".", 1)[-1] for n in ast.walk(tree) if isinstance(n, ast.Call)}


def test_the_stage_has_no_write_path_into_a_tree():
    """Negative (structural): a call that moves the hand or writes a tree file in the stage fails this test — re-adding
    a rule C repair is a rule change, not a fix."""
    writes = {"write_tree", "undo_tree", "shift_row", "save_npz", "apply_delta", "source_of", "mark_post", "revert"}
    assert not _calls(STAGE) & writes
    assert not _calls(STAGE) & {"follow_pieces", "ramp_dilate", "savez", "replace", "copy2"}


def test_a_float_that_jumps_drops_the_clip_through_the_stage_row(monkeypatch):
    st = {"o1": {"verdict": "drop", "gap_m": 0.031, "support": "desk", "rest_frame": 4, "first_touch": 5}}
    monkeypatch.setattr(HS.FS, "statement", lambda *a, **k: st)
    row = HS.FS.stage_row("/x/clipF", TI, np.zeros((T, 60)), [], "sha", "v", HS.STATUSES)
    assert row["status"] == DROP and HS.drop_verdicts([{**row}])["clipF"][0] == HS.FS.DROP_WORD
    assert HS.FS.RULE_ID in HS.rule_version("oakink2")  # the float rule's values make the column stale


def test_the_dyn_pairs_verdicts_exclude_the_clip_with_their_words(tmp_path, measured, monkeypatch):
    """S-dyn-pairs in this step: a both-dynamic pair over the visual cap and a together pair without a one-body path
    each make a DROP row of their side; the clip is excluded with the visual word first (one exclusion path)."""
    from dexengine.data.objects import dyn_pair_check as DC
    from dexengine.data.objects import dyn_pairs as DP

    clip, rows, err = HS.tree_rows(str(_tree(tmp_path)))
    dyn = {r["side"]: r for r in rows if r["side"] in (DP.PAIR_SIDE, DP.TOGETHER_SIDE)}
    assert err == "" and {r["status"] for r in dyn.values()} == {CLEAN} and HS.drop_verdicts(rows) == {}
    st = {
        "pairs": [{"a": "left_object", "b": "right_object", "verdict": DP.DROP}],
        "together": [{"a": "right_object", "b": "o1", "verdict": DP.NO_PATH_V}],
        "drop": (DC.DROP_WORD, "left_object×right_object: visual 6.2 mm"),
        "no_path": (DP.NO_PATH_WORD, "right_object×o1: rests on it"),
    }
    monkeypatch.setattr(DP, "dyn_pairs", lambda *a, **k: st)
    clip, rows, err = HS.tree_rows(str(_tree(tmp_path / "b")))
    dyn = {r["side"]: r for r in rows if r["side"] in (DP.PAIR_SIDE, DP.TOGETHER_SIDE)}
    assert dyn[DP.PAIR_SIDE]["status"] == DROP and dyn[DP.PAIR_SIDE]["rows"] == "left_objectxright_object:drop"
    assert dyn[DP.TOGETHER_SIDE]["status"] == DROP and dyn[DP.TOGETHER_SIDE]["why"] == "right_object×o1: rests on it"
    v = HS.drop_verdicts(rows)
    assert v[clip][0] == DC.DROP_WORD and f"{DP.TOGETHER_SIDE}: right_object×o1" in v[clip][1]
    st["drop"], st["pairs"] = None, []
    assert HS.drop_verdicts(HS.tree_rows(str(_tree(tmp_path / "c")))[1])["clipA"][0] == DP.NO_PATH_WORD
    assert DP.RULE_ID in HS.rule_version("hot3d")  # a rule change makes the ledger rows stale


def test_a_dyn_pair_that_cannot_be_read_leaves_the_tree_unmeasured(tmp_path, measured, monkeypatch):
    """A raise in the S-dyn-pairs reading (no collision body, no visual mesh) is the tree's error: no rows, no
    verdict, the run reports it with the others (never a crash of the pool, never a silent clean row)."""
    from dexengine.data.objects import dyn_pairs as DP

    def boom(*a, **k):
        raise ValueError("left_object has no collision body")

    monkeypatch.setattr(DP, "dyn_pairs", boom)
    clip, rows, err = HS.tree_rows(str(_tree(tmp_path)))
    assert rows == [] and err.startswith("S-dyn-pairs ValueError: left_object has no collision body")
