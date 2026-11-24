"""Conditions of the judged-cell bundle runner (record.augment.cells_cond) — its
cases (CPU only, synthetic bundles).

  R9 f0 shadow check: same key or a marker older than a keyless dump = done · a different key or a keyless dump older
     than its marker = judged again · no marker = dies · only the clips judged again lose their old cell list · main
     runs the check in judging runs only. Negative: removing the keyless rule or the key comparison is caught.
  R10 manual holds have one entry: the table (judging runs) · the finished-bundle check = only the USER_HOLD entries of
     RUN_DONE held · no table = unchanged · a bundle held entirely dies · main calls the one entry.
  R5 re-selection: a condition margin becomes the RESELECT_IN of every clip of the group · only a condition with a
     margin gets --place-reselect · a clip in the source chunk's record gets place_reselect_mm, others not.
  R6 source chunk record: run.env place_reselect= without the file dies · with it = its absolute place · none = ''.
"""

from __future__ import annotations

import ast
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from dexengine.core.layout.registry import AXIS_HAND
from dexengine.core.paths import hand_spec_dir
from dexengine.placement.judge import aug_cells as JAC
from dexengine.placement.judge import f0_key as FK
from dexengine.record.augment import cells_cond as ACC

REPO = Path(__file__).resolve().parents[2]
# the runner = main (augment.cells) + its steps (augment.cells_steps)
RUNNER_SRCS = [(REPO / f"dexengine/record/augment/{m}.py").read_text() for m in ("cells", "cells_steps")]
RUNNER = "".join(RUNNER_SRCS)


def _region(bundle: Path, clip: str, key: str | None = None) -> Path:
    p = JAC.region_path(bundle, clip)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, cells=np.zeros((1, 3), np.int32), **({} if key is None else {"f0_key": np.array(key)}))
    return p


def _cells(bundle: Path, clip: str) -> None:
    f = JAC.cells_path(bundle, clip)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"clip": clip, "cells": []}))


@pytest.fixture()
def r9(tmp_path):
    b, f0r = tmp_path / "r9", tmp_path / "f0"
    conds = {c: {"f0": str(f0r), "emb": "bimanual"} for c in ("kS", "kD", "lOld", "lNew", "nM")}

    def mark(c: str, ik: str, mt: float | None = None) -> Path:
        mk = FK.marker(f0r, hand_spec_dir(AXIS_HAND), "bimanual", c)
        mk.parent.mkdir(parents=True, exist_ok=True)
        mk.write_text(json.dumps({"input_key": ik, "fp_design": "f0_in_v3", "status": "CLEAN"}))
        if mt is not None:
            os.utime(mk, (mt, mt))
        return mk

    now = time.time()
    _region(b, "kS", FK.key_of(mark("kS", "s")))
    _region(b, "kD", FK.key_of(mark("kD", "old")))
    mark("kD", "new")
    mark("lOld", "x", now)
    os.utime(_region(b, "lOld"), (now - 900, now - 900))
    mark("lNew", "x", now - 900)
    os.utime(_region(b, "lNew"), (now, now))
    _region(b, "nM")
    return b, conds


DONE = ["kS", "kD", "lOld", "lNew"]


def test_f0_again_sorts_reused_dumps(r9):
    b, conds = r9
    assert ACC.f0_again(b, DONE, ["t0"], conds, AXIS_HAND) == (["kS", "lNew"], ["t0", "kD", "lOld"])


def test_f0_again_dies_without_a_marker(r9):
    b, conds = r9
    with pytest.raises(SystemExit, match="no f0 marker"):
        ACC.f0_again(b, ["nM"], [], conds, AXIS_HAND)


def test_f0_again_negative_rules_are_caught(r9, monkeypatch):
    b, conds = r9
    real = FK.stale_why
    monkeypatch.setattr(FK, "stale_why", lambda key, mt, mk: real(key, mt, mk) if key else "")
    assert ACC.f0_again(b, DONE, [], conds, AXIS_HAND) != (["kS", "lNew"], ["kD", "lOld"])
    monkeypatch.setattr(FK, "stale_why", lambda key, mt, mk: "")
    assert ACC.f0_again(b, DONE, [], conds, AXIS_HAND)[1] == []


def test_drop_stale_cells_removes_only_the_clips_judged_again(r9):
    b, _conds = r9
    for c in ("kS", "kD", "lOld"):
        _cells(b, c)
    assert ACC.drop_stale_cells(b, ["kD", "lOld", "t0"]) == 2
    assert not JAC.cells_path(b, "kD").is_file() and JAC.cells_path(b, "kS").is_file()


def test_runner_checks_f0_only_in_judging_runs():
    """main: the f0 check comes after the stage state and only outside --verify-only; the old cell lists go after the
    verify-only return."""
    src = RUNNER_SRCS[0]
    fns = {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    main = fns["main"]
    i_d, i_f, i_v, i_x = (
        main.index(s)
        for s in (
            "d_done, d_todo = d_state(bundle, live",
            "ACC.f0_again(",
            "if a.verify_only:\n",
            "_run_stages(",
        )
    )
    assert i_d < i_f < i_v < i_x and "if not a.verify_only:" in main[i_d:i_f]
    assert "ACC.drop_stale_cells(" in fns["_run_stages"]


def test_user_holds_one_entry(tmp_path):
    b = tmp_path / "r10"
    b.mkdir()
    tab = b / "hold.tsv"
    tab.write_text("clip\treason\ncB\tcanonical placement judged again\n", encoding="utf-8")
    live, held = ACC.holds(str(tab), ["cA", "cB", "cC"], {}, b, False)
    assert (live, held) == (["cA", "cC"], {"cB": "[USER_HOLD] canonical placement judged again"})
    (b / "RUN_DONE.json").write_text(json.dumps({"held": {"cB": held["cB"], "cC": "[GEN_HOLD] x"}}))
    assert ACC.holds("", ["cA", "cB", "cC"], {}, b, True) == (["cA", "cC"], {"cB": held["cB"]})
    assert ACC.holds("", ["cA", "cB"], {}, b, False) == (["cA", "cB"], {})
    assert ACC.holds("", ["cA"], {}, tmp_path / "none", True) == (["cA"], {})
    with pytest.raises(SystemExit, match="every clip"):
        ACC.holds(str(tab), ["cB"], {}, b, False)


@pytest.mark.parametrize("text", ["clip\twhy\ncB\tx\n", "clip\treason\ncB\t \n", "clip\treason\ncB\tx\ncB\ty\n"])
def test_bad_hold_table_dies(text, tmp_path):
    tab = tmp_path / "hold.tsv"
    tab.write_text(text, encoding="utf-8")
    with pytest.raises(SystemExit, match="hold table"):
        ACC.user_holds(str(tab), ["cA", "cB"], {})


def test_runner_uses_the_one_hold_entry():
    line = "ACC.holds(a.hold_clips, list(clips), {}, bundle, a.verify_only)"
    assert RUNNER.count(line) == 1 and "ACC.user_holds(" not in RUNNER


# ── R5 · R6 re-selection margin ───────────────────────────────────────────────────────────────────────────────────
def _runner():
    from dexengine.record.augment import cells as R  # noqa: PLC0415

    return R


def test_write_reselect_and_judge_script(tmp_path, capsys):
    from dexengine.placement.bake import place_check as PC  # noqa: PLC0415
    from dexengine.record.augment import cells_steps as ACS  # noqa: PLC0415

    cond = {
        "arm": "rby1",
        "hand": AXIS_HAND,
        "dataset": "taco",
        "emb": "bimanual",
        "src_layer": "base",
        "dz": "0",
        "pool": "p",
        "active": "both",
    }
    rcond = dict(cond, place_reselect_mm=14.5)
    rf = ACC.write_reselect(rcond, ["cX", "cY"], tmp_path)
    assert rf == tmp_path / PC.RESELECT_IN and PC.read_reselect(rf) == {"cX": 0.0145, "cY": 0.0145}
    assert rf.read_text(encoding="utf-8") == (
        "clip\tshort_mm\tmargin_mm\tverdict\n"
        "cX\t-\t14.50\taug_cells_run(source chunk re-selection record)\n"
        "cY\t-\t14.50\taug_cells_run(source chunk re-selection record)\n"
    )
    assert f"{PC.RESELECT_TAG} condition margin 14.5 mm · clips 2" in capsys.readouterr().out
    sr = ACS.judge_script(rcond, "L", tmp_path, tmp_path / "c.txt", tmp_path, "ch", 60, "8G", rf)
    assert f"--place-reselect {rf}" in sr and ACC.write_reselect(cond, ["cX"], tmp_path) is None
    assert "--place-reselect" not in ACS.judge_script(cond, "L", tmp_path, tmp_path / "c.txt", tmp_path, "ch", 60, "8G")
    rec = tmp_path / "rs_in.tsv"
    rec.write_text("clip\tshort_mm\tmargin_mm\tverdict\ncR\t12.50\t14.50\tv\n", encoding="utf-8")
    ck = {"reselect": str(rec)}
    assert ACC.with_reselect(dict(cond), ck, {"clip": "cR"}) == dict(cond, place_reselect_mm=14.5)
    assert ACC.with_reselect(dict(cond), ck, {"clip": "cZ"}) == cond
    assert ACC.with_reselect(dict(cond), {"reselect": ""}, {"clip": "cR"}) == cond
    assert list(ACC.with_reselect(dict(cond), ck, {"clip": "cR"}))[-1] == "place_reselect_mm"  # key order as before


def test_chunk_record_reselect(tmp_path, monkeypatch):
    R = _runner()
    assert "place_reselect" not in R.UNSUPPORTED_ENV
    monkeypatch.setattr(R.P, "run_path", lambda layer, hand="", **kw: tmp_path / "chunks" / kw["tag"])
    d = tmp_path / "chunks" / "G1"
    d.mkdir(parents=True)
    env = "tag=t\ndataset=hocap emb=bimanual\ncombo=rby1_wuji\njudge_extra=\n"
    (d / "judge.log").write_text("[ACTIVE_SIDES] mode=auto\n", encoding="utf-8")
    (d / "run.env").write_text(env + "place_reselect=rs_in.tsv\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="the margin is not made up"):
        R.chunk_record(AXIS_HAND, "G1")
    (d / "rs_in.tsv").write_text("clip\tshort_mm\tmargin_mm\tverdict\ncR\t12.50\t14.50\tv\n", encoding="utf-8")
    assert R.chunk_record(AXIS_HAND, "G1")["reselect"] == str(d / "rs_in.tsv")
    (d / "run.env").write_text(env, encoding="utf-8")
    assert R.chunk_record(AXIS_HAND, "G1")["reselect"] == ""


def test_runner_passes_the_group_record():
    # Structural check on the runner's calls (layout-independent: the formatter may wrap them).
    calls = [n for src in RUNNER_SRCS for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)]
    text = {ast.unparse(c) for c in calls}
    want = "judge_script(cond, mode, bundle, cf, work, chunk, timeout_s, mem, ACC.write_reselect(cond, clips, work))"
    assert ast.unparse(ast.parse(want, mode="eval").body) in text
    assert any(ast.unparse(c.func) == "ACC.with_reselect" and c.args and isinstance(c.args[0], ast.Dict) for c in calls)
    dicts = [n for src in RUNNER_SRCS for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Dict)]
    assert any(
        isinstance(k, ast.Constant)
        and k.value == "reselect"
        and isinstance(v, ast.Call)
        and ast.unparse(v.func) == "ACC.reselect_of"
        and ast.unparse(v.args[0]) == "d"
        and ast.unparse(v.args[1]).startswith("kv.get('place_reselect'")
        for dn in dicts
        for k, v in zip(dn.keys, dn.values, strict=True)
    )
