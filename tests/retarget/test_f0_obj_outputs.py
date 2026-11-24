"""f0 object repair outputs: the window writer, the ledger states (REPAIRED · HELD) and the keys that downstream
stages read."""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest

from dexengine.retarget.f0 import phrases as PH
from dexengine.retarget.f0 import repair_obj as RO
from dexengine.retarget.f0 import shadow_tree as ST
from tests.imports import own_module

BUDGET_OK = "[F0_BUDGET] side=right evals=4/10 status=OK"


def test_window_writer_rewrites_rows_in_place(monkeypatch, tmp_path):
    """write_obj_window: rows 0… of qpos and the wrist cache move by each row's change, the rest and the length stay,
    and the marker carries the rows' hash (the downstream key)."""
    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    W = own_module("dexengine.retarget.f0.repair_obj_write")
    from dexengine.placement.judge.f0_key import OBJ_WINDOW_MARKER
    from dexengine.retarget.f0 import repair_root as RT

    # no scene in this tree: its wrist cache is qpos[:3] + a constant, i.e. the root's axes are the identity
    monkeypatch.setattr(RT, "root_axes", lambda tree, off: np.eye(3))

    H, T, rng = W.HC.H, 40, np.random.default_rng(0)
    qpos = rng.normal(0, 0.1, (T, H + 7))
    (tmp_path / "data/c0/0").mkdir(parents=True)
    np.savez(tmp_path / "data/c0/0/trajectory_kinematic.npz", qpos=qpos, qvel=np.zeros_like(qpos))
    (tmp_path / "w/wrist_poses_right").mkdir(parents=True)
    pos = qpos[:, :3] + np.array([0.1, 0.2, 0.3])
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (T, 1))
    np.savez(tmp_path / "w/wrist_poses_right/c0.npz", positions=pos, quaternions=quat)
    rows = np.zeros((8, W.HC.nd))
    rows[:, 0], rows[:, 6:] = 0.01, 0.05
    r = {
        "_internal": (None,) * 6 + (0, None),
        "obj_window": rows,
        "obj_window_n": 5,
        "obj_window_sat_mm": [1.0] * 5,
        "obj_window_carried": [],
    }
    W.write_obj_window("c0", {"right": r}, tmp_path / "data", tmp_path / "w")
    q2 = np.load(tmp_path / "data/c0/0/trajectory_kinematic.npz")["qpos"]
    p2 = np.load(tmp_path / "w/wrist_poses_right/c0.npz")["positions"]
    assert q2.shape == qpos.shape and np.allclose(q2[8:], qpos[8:]) and np.allclose(p2[8:], pos[8:], atol=1e-6)
    assert np.allclose(p2[:8, 0], pos[:8, 0] + 0.01, atol=1e-6) and np.allclose(q2[:8, 0], qpos[:8, 0] + 0.01)
    assert np.allclose(q2[:8, 6:H], qpos[:8, 6:H] + 0.05)
    mk = json.loads((tmp_path / "data/c0" / OBJ_WINDOW_MARKER).read_text())
    assert mk["rows_sha"] == W.obj_rows_sha({"right": rows}) != W.obj_rows_sha({"right": rows * 2})
    inplace = json.loads((tmp_path / "data/c0/F0_REPAIR_INPLACE.json").read_text())
    assert inplace["mode"] == "window" and inplace["P"] == 0


def test_a_window_repair_is_repaired_in_the_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(ST, "applied_scene", lambda: "test")
    a = types.SimpleNamespace(hand="sharpa_wave", dataset="taco", emb="right", src_layer="base")
    d = tmp_path / "shadow"
    d.mkdir()
    (d / "F0_REPAIR_INPLACE.json").write_text(json.dumps({"mode": "window", "P": 0}))
    job = {"clip": "c0", "t0": 0.0, "d": d, "key_now": "k", "stale": "", "sides": ["right"]}
    res = ST.finish(job, 0, BUDGET_OK, a, Path("/stage"), {"right": Path("/w/right")})
    assert res["status"] == "REPAIRED"


@pytest.mark.parametrize("other_repaired", [False, True])
def test_refused_repair_holds_the_clip_with_its_reason(tmp_path, monkeypatch, other_repaired):
    """A refused side holds the whole clip (even when the other side was repaired): no marker (not judged), the ledger
    note names the side and the reason."""
    monkeypatch.setattr(ST, "applied_scene", lambda: "test")
    a = types.SimpleNamespace(hand="sharpa_wave", dataset="taco", emb="bimanual", src_layer="base")
    d = tmp_path / "shadow"
    d.mkdir()
    if other_repaired:
        (d / "F0_REPAIR_INPLACE.json").write_text(json.dumps({"mode": "window", "P": 0}))
    job = {"clip": "c0", "t0": 0.0, "d": d, "key_now": "k", "stale": "", "sides": ["right", "left"]}
    why = RO.refuse({"side": "left"}, "no result keeps every contact link (dropped: left_tip)")["status"]
    assert why.startswith(PH.SKIP_OBJ)
    tag = f"c0 [left] n_pen=3 (desk 0/object 3/reverse 0 · max depth 0.0mm) -> {why} (wrist -cm/-deg)  (check 0.1s)"
    out = "\n".join([BUDGET_OK, BUDGET_OK.replace("right", "left"), tag])
    res = ST.finish(job, 0, out, a, Path("/stage"), {"right": Path("/w/r"), "left": Path("/w/l")})
    assert res["status"] == "HOLD_OBJ" and not (d / ST.MARKER).exists()
    assert res["log"].startswith("left: object repair refused") and "dropped: left_tip" in res["log"]


def test_held_is_counted_and_is_not_a_run_failure(tmp_path):
    from dexengine.retarget.f0 import ledger as L
    from dexengine.retarget.f0 import shadow_status as SS

    assert SS.is_held(SS.HELD) and not SS.is_run_failure(SS.HELD) and not SS.is_judgeable(SS.HELD)
    assert SS.unsettled_reason(SS.HELD, tmp_path / "none") == ""
    rows = [{"status": s} for s in ("REPAIRED", SS.HELD, "FAIL_F0", SS.DATA_DEFECT)]
    log = tmp_path / "f0.log"
    log.write_text(SS.failclass_marker(rows) + "\n")
    assert L.failclass(log, {"mode": "run", "total": 4}) == {
        "data_defect": 1,
        "run_failure": 1,
        "judgeable": 1,
        "held": 1,
    }


def test_split_failures_lists_held_clips_apart(tmp_path, capsys):
    from dexengine.retarget.f0 import ledger as L
    from dexengine.retarget.f0 import shadow_status as SS

    (tmp_path / "f0/hd").mkdir(parents=True)
    rows = [
        ("a", "REPAIRED", "-"),
        ("b", SS.HELD, "left: object repair refused"),
        ("c", "FAIL_F0", "boom"),
        ("d", SS.DATA_DEFECT, "no main object"),
    ]
    (tmp_path / "f0/hd" / SS.LEDGER_NAME).write_text(
        "clip\tstatus\tP\tsec\tstale\tbskip\tevals\tnote\n"
        + "".join(f"{c}\t{s}\t0\t1\t\t0\t\t{n}\n" for c, s, n in rows)
    )
    (tmp_path / "clips.txt").write_text("a\nb\nc\nd\n")
    rf, hd = tmp_path / "run_failed.txt", tmp_path / "held.txt"
    L.split_failures(
        [str(tmp_path / "f0"), str(tmp_path / "clips.txt"), "hd", "--run-failed", str(rf), "--held", str(hd)]
    )
    assert capsys.readouterr().out.split() == ["d"]
    assert [ln.split("\t")[:2] for ln in hd.read_text().splitlines()] == [["b", SS.HELD]]
    assert [ln.split("\t")[0] for ln in rf.read_text().splitlines()] == ["c"]


def test_placement_ledger_records_held_clips(tmp_path):
    from dexengine.placement import ledger as PL
    from dexengine.placement import ledger_states as LS

    assert "HELD" in LS.STATES and "HELD" in LS.REJECTED
    (tmp_path / "f0_held.txt").write_text("b\tHOLD_OBJ\twhy\n")
    st = {"b": "ADOPTED"}
    PL._fold_f0_held(types.SimpleNamespace(judge_root=[str(tmp_path)]), st, lambda c: c)
    assert st == {"b": "HELD"}


def _repair_mod(monkeypatch):
    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    return own_module("dexengine.retarget.f0.repair")


def _fake_gate(monkeypatch, R, opened: dict, tmp_path):
    """check_and_repair_steps stand-in: a generator that returns the gate decision of each side (no solve)."""

    def steps(clip, side, tree_root, wrist_root, desk_margin, do_repair, dataset="taco"):
        """A generator like the real one: it yields nothing and returns the report."""
        assert do_repair is False
        yield from ()
        return {"side": side, "obj_repair": opened[side], "obj_cooked_digest": "c0ffee"}

    monkeypatch.setattr(R, "check_and_repair_steps", steps)
    clean = {"verdict": "clean", "pairs": [], "new": []}  # the frame-0 separation finds no pair (repair_sep)
    monkeypatch.setattr(R.RSEP, "plan", lambda tree, q=None: clean)
    for s in opened:
        (tmp_path / f"wrist_poses_{s}").mkdir(exist_ok=True)
        (tmp_path / f"wrist_poses_{s}" / "c0.npz").write_bytes(b"")


def test_acted_part_is_empty_unless_the_repair_acts(monkeypatch, tmp_path):
    R = _repair_mod(monkeypatch)
    _fake_gate(monkeypatch, R, {"right": False, "left": True}, tmp_path)
    monkeypatch.setenv("DEXENGINE_F0_OBJ_REPAIR", "1")
    part = R.obj_acted_part("c0", tmp_path, tmp_path, "taco", ["right", "left"])
    assert part == f"obj_repair/left|{RO.policy_line()}|c0ffee"
    assert R.obj_acted_part("c0", tmp_path, tmp_path, "taco", ["right"]) == ""
    monkeypatch.setenv("DEXENGINE_F0_OBJ_REPAIR", "0")
    assert R.obj_acted_part("c0", tmp_path, tmp_path, "taco", ["right", "left"]) == ""
    with pytest.raises(RuntimeError):
        R.obj_acted_part("c0", tmp_path, tmp_path, "taco", ["left"], hand="wuji2")


@pytest.mark.parametrize("acts", [False, True])
def test_f0_key_moves_only_for_a_clip_the_repair_acts_on(monkeypatch, tmp_path, acts):
    """The f0 cache key with the switch on and off: an untouched clip keeps its key byte for byte, a repaired clip
    gets a new one (the key's last line; measured the same way on real clips: shadow_key._feed_obj_acted)."""
    R = _repair_mod(monkeypatch)
    from dexengine.retarget.f0 import shadow_key as SK

    _fake_gate(monkeypatch, R, {"right": acts}, tmp_path)
    monkeypatch.setattr(SK, "clip_sides", lambda stage_src, clip: ["right"])
    keys = {"0": [], "1": []}
    for v in ("0", "1"):
        monkeypatch.setenv("DEXENGINE_F0_OBJ_REPAIR", v)
        SK._feed_obj_acted(
            lambda part, line, out=keys[v]: out.append(line),
            "c0",
            tmp_path,
            {"right": tmp_path / "wrist_poses_right"},
            "taco",
        )
    assert keys["0"] == []
    assert (keys["1"] != keys["0"]) is acts


@pytest.mark.parametrize("rows", [None, "r1", "r2"])
def test_judge_base_field_is_keyed_on_the_f0_output(monkeypatch, tmp_path, rows):
    """No rows written (refused · gate closed · switch off) → "" (the key before the rule); rows → their hash, and
    different rows → a different key."""
    from dexengine.core import paths as P
    from dexengine.placement.judge import input_clip as IC
    from dexengine.placement.judge.f0_key import OBJ_WINDOW_MARKER

    monkeypatch.setenv("DEXENGINE_F0_STAGE", str(tmp_path / "f0"))
    tree = tmp_path / "base" / P.hand_spec_dir("sharpa_wave") / "bimanual" / "c0"
    shadow = tmp_path / "f0" / P.hand_spec_dir("sharpa_wave") / "bimanual" / "c0"
    shadow.mkdir(parents=True)
    if rows:
        (shadow / OBJ_WINDOW_MARKER).write_text(json.dumps({"rows_sha": {"r1": "a" * 64, "r2": "b" * 64}[rows]}))
    suffix = IC._f0_obj("taco", "sharpa_wave", tree)
    assert suffix == ("" if rows is None else "+o" + {"r1": "a", "r2": "b"}[rows] * 12)
