"""Scene manifest sidecar (CPU): the builder's statement of one trajectory, the one writer of meta/scene_manifest.jsonl
and its oracle. The tree's slot rows, object hulls and the trajectory's self-penetration measurement are stand-ins (the
rule passes, the cooked dumps and the ruler are not under test here); every other value comes from its owner."""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest

from dexengine.data.objects import extra_slots as XS
from dexengine.data.objects import hull as HU
from dexengine.placement.collision import self_pen as SP
from dexengine.record.scene import manifest as MF
from dexengine.record.store import scene_manifest as SM
from dexengine.rl.tasks.scene import collision_presets as CP

HAND, DS = "wuji2", "hot3d"
#: The stand-in measurement of every fixture trajectory: 4.2 mm arm-torso at frame 3 (kept).
CLEAN = SP.Result(0.0042, "AT", "link_left_arm_5", "link_torso_5", 3, 4)
T = {"parquet": Path("episode.parquet"), "info": {}}
ROWS = [
    {"name": "extra_object_0", "cls": XS.STATIC_SPAWN, "why": "receiver", "kind": "receiver", "obj_id": "bowl"}
    | {"static": True},
    {"name": "extra_object_1", "cls": XS.SLOT, "why": "moved", "kind": "moving", "obj_id": "spoon", "static": False},
    {"name": "extra_object_2", "cls": XS.VISUAL_ONLY, "why": "still", "kind": None, "obj_id": "mug"},
]
EPISODES = [
    {"episode_index": 0, "clip": "c1", "round": "orig", "look": {"name": "canon"}, "side": "both"},
    {"episode_index": 1, "clip": "c1", "round": "orig", "look": {"name": "rand_1"}, "side": "both"},
    {"episode_index": 2, "clip": "c2", "round": "orig", "look": {"name": "canon"}, "side": "both"},
]


@pytest.fixture
def scene(tmp_path, monkeypatch, data_root):
    """A tree `c1_pl_rt` (task info, kinematic qpos) whose slot rows and hulls are stand-ins."""
    tree = tmp_path / "c1_pl_rt"
    (tree / "0").mkdir(parents=True)
    objs = [{"name": "right_object", "role": "main", "side": "right", "obj_id": "cup"}] + [
        {"name": r["name"], "role": "extra", "obj_id": r["obj_id"], "qpos_start": 7 * (i + 1)}
        for i, r in enumerate(ROWS)
    ]
    (tree / "task_info.json").write_text(json.dumps({"dataset_name": DS, "objects": objs}))
    np.savez(tree / "0" / "trajectory_kinematic.npz", qpos=np.zeros((4, 28)))
    monkeypatch.setattr(XS, "clip_rows", lambda t, ti, q: [dict(r) for r in ROWS])
    monkeypatch.setattr(HU, "usd_file", lambda ds, i, art: data_root / f"data/{ds}/usd/{i}/object.usd")
    monkeypatch.setattr(HU, "collision_parts", lambda ds, i, art: types.SimpleNamespace(links={"root": [0, 1]}))
    monkeypatch.setattr(MF, "self_penetration", lambda t: SP.statement(CLEAN))
    return tree


@pytest.fixture
def final(tmp_path, scene, monkeypatch):
    """A final with three episodes (the third superseded) and its sidecar written by `sync`."""
    f = tmp_path / "collect" / "rby1_wuji2"
    (f / "meta").mkdir(parents=True)
    (f / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in EPISODES))
    (f / "meta/superseded.jsonl").write_text(json.dumps({"episode_index": 2}) + "\n")
    monkeypatch.setattr(SM, "_tree_paths", lambda arm, hand, ds, eps: {e["episode_index"]: (scene, scene) for e in eps})
    monkeypatch.setattr(SM, "trajectories", lambda final, eps: {e["episode_index"]: (T, "table") for e in eps})
    assert _main("sync", f, "--arm", "rby1") == 0
    return f


def _main(cmd, f, *rest):
    return SM.main([cmd, "--final", str(f), "--hand", HAND, "--dataset", DS, *rest])


def _rows(f: Path) -> list:
    return [json.loads(x) for x in (f / "meta" / SM.SIDECAR).read_text().splitlines()]


def _write(f: Path, rows: list) -> None:
    (f / "meta" / SM.SIDECAR).write_text("".join(SM.row_line(r) for r in rows))


def test_the_statement_holds_and_names_its_owners(scene):
    s = MF.build(scene, T)
    assert MF.problems(s, T) == []
    assert s["self_penetration"] == SP.statement(CLEAN) and s["pair_off"] == []
    assert s["self_penetration"]["rule"]["measured"] == "observation.joint_pos.robot"
    assert s["constants"]["dexengine.data.objects.scene_requirement.TRUE_SHAPE_OVERLAP_M"] > 0
    body = {b["name"]: b for b in s["bodies"]}
    assert body["extra_object_0"]["collision"]["collider"] == "triangles"
    assert body["extra_object_1"]["collision"]["collider"] == "parts"
    assert body["extra_object_0"]["collision"]["usd"] == "data/hot3d/usd/bowl/object.usd"  # data-root relative
    robot = [c for c in s["collisions"] if c["b"] == "extra_object_0" and c["a"].startswith("robot")]
    assert robot[0]["training"] == "ON"  # robot <-> still receiver on in training
    rows = {c["row"]: c for c in s["collisions"] if c["row"] in MF.SELF_ROWS}
    assert {(c["training"], c["delivered"]) for c in rows.values()} == {("OFF", "OFF")}
    assert {r["id"] for r in s["scene_rules"]} >= {"S-true-shape", "S-robot-static"}


def test_a_clip_with_no_rule_change_keeps_its_row_bytes(final):
    before = (final / "meta" / SM.SIDECAR).read_bytes()
    assert _main("sync", final, "--arm", "rby1", "--all") == 0
    assert (final / "meta" / SM.SIDECAR).read_bytes() == before
    assert [r["card"]["look"] for r in _rows(final)] == ["canon", "rand_1"]  # live episodes only


def test_the_written_sidecar_passes_its_oracle(final):
    assert SM.check_scene_manifest(final, HAND, DS) == []
    assert _main("check", final) == 0


def test_a_dropped_row_fails(final):
    _write(final, _rows(final)[:1])
    assert any("missing rows 1" in p for p in SM.check_scene_manifest(final, HAND, DS))
    assert _main("check", final) == 1


def test_a_missing_sidecar_and_extra_or_repeated_rows_fail(final):
    rows = _rows(final)
    _write(final, [*rows, rows[0]])
    assert any("repeated cards 1" in p for p in SM.check_scene_manifest(final, HAND, DS))
    _write(final, [*rows, {**rows[0], "card": {**rows[0]["card"], "clip": "c2", "variant": "c2"}}])
    assert any("rows without a live episode" in p for p in SM.check_scene_manifest(final, HAND, DS))
    (final / "meta" / SM.SIDECAR).unlink()
    assert SM.check_scene_manifest(final, HAND, DS) == [f"{final}: no meta/{SM.SIDECAR}"]


def test_a_flipped_slot_collider_fails(final):
    rows = _rows(final)
    body = next(b for b in rows[0]["scene"]["bodies"] if b["name"] == "extra_object_0")
    body["collision"]["collider"] = "parts"
    _write(final, rows)
    bad = SM.check_scene_manifest(final, HAND, DS)
    assert any("extra_object_0: collider 'parts' != dispatcher 'triangles'" in p for p in bad)


@pytest.mark.parametrize("where", ["row", "card", "scene", "body", "collision", "cell"])
def test_an_unknown_key_fails(final, where):
    rows = _rows(final)
    r = rows[1]
    target = {
        "row": r,
        "card": r["card"],
        "scene": r["scene"],
        "body": r["scene"]["bodies"][1],
        "collision": r["scene"]["bodies"][1]["collision"],
        "cell": r["scene"]["collisions"][0],
    }[where]
    target["colour"] = "red"
    _write(final, rows)
    bad = SM.check_scene_manifest(final, HAND, DS)
    assert bad and any("colour" in p or "keys" in p for p in bad)


def test_a_planted_25mm_stated_self_penetration_fails(final):
    """A row stating 25 mm (kept) where the trajectory measures 4.2 mm: the verdict is not that of its max, and the
    measurement again differs."""
    rows = _rows(final)
    rows[0]["scene"]["self_penetration"]["max_mm"] = 25.0
    _write(final, rows)
    bad = SM.check_scene_manifest(final, HAND, DS)
    assert any("verdict keep != the verdict of 25.0 mm" in p for p in bad)
    assert any("measured again" in p for p in bad)


def test_a_planted_25mm_trajectory_fails(final, monkeypatch):
    """The joints changed under an unchanged row (now 25 mm arm-torso): the oracle's measurement again refuses it."""
    monkeypatch.setattr(MF, "self_penetration", lambda t: SP.statement(SP.Result(0.025, "AT", "a", "b", 2, 4)))
    bad = SM.check_scene_manifest(final, HAND, DS)
    assert bad and all("measured again" in p for p in bad)


def test_a_stated_drop_or_another_rule_fails(final):
    """A delivered trajectory over the line fails, and so does a statement of another threshold."""
    rows = _rows(final)
    rows[0]["scene"]["self_penetration"] = SP.statement(SP.Result(0.025, "AT", "a", "b", 2, 4))
    rows[1]["scene"]["self_penetration"]["rule"]["threshold_mm"] = 30.0
    _write(final, rows)
    bad = SM.check_scene_manifest(final, HAND, DS)
    assert any("/canon:" in p and "over the line: delivered" in p for p in bad)
    assert any("/rand_1:" in p and "!= owner measured" in p for p in bad)


def test_pair_off_cells_and_an_attached_still_are_stated_from_their_owners(scene, monkeypatch):
    """A pair-off pair of the scene statement (dyn_pairs.off_pairs) is a cell of the collision owner's value between
    the two main bodies; a still riding a main object (S-dyn-pairs attach, kind carried_attached) states the carrier
    pose source. A cell with another value or another body fails the oracle."""
    from dexengine.data.objects import dyn_pairs as DP

    ti = json.loads((scene / "task_info.json").read_text())
    ti["objects"].append({"name": "left_object", "role": "main", "side": "left", "obj_id": "jar"})
    (scene / "task_info.json").write_text(json.dumps(ti))
    pair = {"a": "left_object", "b": "right_object", "f0_m": 0.0031}
    monkeypatch.setattr(DP, "off_pairs", lambda t, ti, q: [pair])
    att = ROWS[0] | {"kind": XS.KIND_ATTACHED, "carry": {"carrier": "right_object", "carrier_role": "main"}}
    monkeypatch.setattr(XS, "clip_rows", lambda t, ti, q: [dict(att), *(dict(r) for r in ROWS[1:])])
    s = MF.build(scene, T)
    assert MF.problems(s, T) == []
    (c,) = s["pair_off"]
    assert (c["row"], c["a"], c["b"], c["delivered"], c["training"]) == (
        "R7",
        "left_object",
        "right_object",
        "OFF",
        "OFF",
    )
    assert {b["name"]: b for b in s["bodies"]}["extra_object_0"]["pose"] == MF.POSE["carried_attached"]
    c["training"] = "ON"
    assert any("pair_off cell" in x for x in MF.problems(s, T))
    c["training"], c["b"] = "OFF", "extra_object_1"
    assert any("not two main bodies" in x for x in MF.problems(s, T))


@pytest.mark.parametrize(
    "plant",
    [
        lambda s: s["constants"].update({"dexengine.data.objects.scene_requirement.TRUE_SHAPE_OVERLAP_M": 0.006}),
        lambda s: s["ruler"].update({"collider_shape_blob": "0" * 40}),
        lambda s: s["ruler"].update({"statement_fp": "ts-000000000000"}),
        lambda s: next(c for c in s["collisions"] if c["row"] == "R2").update({"training": "ON"}),
        lambda s: s["scene_rules"][0].update({"on": not s["scene_rules"][0]["on"]}),
        lambda s: s.update({"tree": "c9_pl_rt"}),
        lambda s: s.update({"pair_off": [["a", "b"]]}),
        lambda s: s.update({"rule": s["rule"] + "+OLD"}),
        lambda s: s.update({"preset": "canon"}),
        lambda s: next(c for c in s["collisions"] if c["b"] == "extra_object_0").update({"training": "OFF"}),
    ],
    ids=[
        "constant",
        "ruler",
        "statement",
        "preset_row",
        "scene_rule",
        "tree",
        "pair_off",
        "rule",
        "preset",
        "robot_still",
    ],
)
def test_a_statement_other_than_its_owner_fails(final, plant):
    rows = _rows(final)
    plant(rows[0]["scene"])
    _write(final, rows)
    assert SM.check_scene_manifest(final, HAND, DS)


def test_the_sidecar_has_one_writer():
    """Only the writer module names the sidecar file; the builder is called by that writer."""
    root = Path(MF.__file__).resolve().parents[2]
    users = [p for p in root.rglob("*.py") if SM.SIDECAR in p.read_text(encoding="utf-8")]
    assert users == [Path(SM.__file__).resolve()]
    assert "MF.build(" in Path(SM.__file__).read_text(encoding="utf-8")


def test_the_presets_still_state_the_self_rows_off():
    """The decided training value of robot self-collision is the recipe preset's (rows R1 R2 R3 off)."""
    assert {CP.EXPECTED[MF.PRESET][k] for k in MF.SELF_ROWS} == {"OFF"}


def test_the_all_object_switch_on_is_refused(final, scene, monkeypatch):
    """The training column states the tables with the all-object collision switch off; with it on neither the
    builder nor the oracle states or checks a row."""
    monkeypatch.setenv(CP.ALL_OBJ_ENV, "1")
    with pytest.raises(ValueError, match=CP.ALL_OBJ_ENV):
        MF.build(scene, T)
    with pytest.raises(ValueError, match=CP.ALL_OBJ_ENV):
        SM.check_scene_manifest(final, HAND, DS)
