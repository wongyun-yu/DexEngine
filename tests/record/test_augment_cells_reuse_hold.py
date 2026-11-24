"""Judged-cell bundle runner (record.augment.cells): the reuse gate by judge commit and the generation hold — CPU
only, synthetic bundles.

  Reuse by commit: a region dump (cells_pick.reuse_why) or a judge chunk row (cells_steps.l_state) is reused only when
     the judge code commit that wrote it is this tree's (head). Same commit means same judge values, so what is reused
     is a subset of what a judge-value comparison reuses: the output is that of a judgement made now, only the cost
     can grow. A dump of another commit is judged again; chunk rows of another commit refuse the reuse.
     Negative: without the commit comparison the dump and the chunk of another commit are reused.
  One owner of the commit rule (cells_pick.other_commit): reuse_why, l_state and the generation hold read it.
  Generation hold (gen_hold.split): the clips judge D will judge are measured — no region dump, or a dump of another
     commit (judged again; without this a mismatched one would kill the judge) · a mismatched clip alone is held
     with its reason, the bundle goes on · run fields measured once per (dataset, scene, arm) · an unmeasurable
     judge input dies (not held) · every clip held dies · the runner passes the split to the one user-hold entry.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from dexengine.placement.judge import aug_cells as JAC
from dexengine.record.augment import cells_pick as ACP
from dexengine.record.augment import cells_steps as CS
from dexengine.record.augment import gen_hold as GH
from dexengine.record.augment import judged as AJ
from dexengine.record.augment import judged_region as JR

REPO = Path(__file__).resolve().parents[2]
HEAD, OTHER = "a" * 40, "b" * 40


# --- reuse by judge commit ----------------------------------------------------------------------------------------


def _on_grid(monkeypatch) -> None:
    monkeypatch.setattr(ACP, "canon_i", lambda R: 0)


def test_dump_of_this_commit_is_reused(monkeypatch):
    _on_grid(monkeypatch)
    assert ACP.reuse_why("region_c.npz", {"judge_commit": HEAD}, "same", [], HEAD) == ""


def test_dump_of_another_commit_is_judged_again(monkeypatch):
    _on_grid(monkeypatch)
    assert ACP.reuse_why("region_c.npz", {"judge_commit": OTHER}, "same", [], HEAD) == ACP.AGAIN


def test_changed_judge_input_is_refused_whatever_the_commit(monkeypatch):
    _on_grid(monkeypatch)
    for commit in (HEAD, OTHER):
        why = ACP.reuse_why("region_c.npz", {"judge_commit": commit}, "stale", ["rules.j"], HEAD)
        assert why.startswith("region_c.npz(judge input stale")


def test_dump_of_this_commit_off_the_grid_is_refused(monkeypatch):
    def off(R):
        raise ValueError("canonical cell not on the grid")

    monkeypatch.setattr(ACP, "canon_i", off)
    assert "dump of the current commit" in ACP.reuse_why("region_c.npz", {"judge_commit": HEAD}, "same", [], HEAD)
    assert ACP.reuse_why("region_c.npz", {"judge_commit": OTHER}, "same", [], HEAD) == ACP.AGAIN


def _chunks(tmp: Path, monkeypatch, commit: str) -> Path:
    bundle = tmp / "bundle"
    cell = [0, 1, 2, 0.5]
    p = JAC.cells_path(bundle, "c1")
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"cells": [cell], "judge_fp": "fp1"}), encoding="utf-8")
    (bundle / "judged").mkdir()
    (bundle / "judged" / "augcells_D__0.jsonl").write_text("", encoding="utf-8")  # read through read_chunks below
    rows = [{"clip": "c1", "k": 0, "cell": cell, "judge_fp": "fp1", "judge_commit": commit}]
    monkeypatch.setattr(AJ, "read_chunks", lambda judged, done=None: rows)
    monkeypatch.setattr(JAC, "head_commit", lambda: HEAD)
    return bundle


def test_chunks_of_this_commit_are_reused(tmp_path, monkeypatch):
    assert CS.l_state(_chunks(tmp_path, monkeypatch, HEAD), ["c1"]) == "done"


def test_chunks_of_another_commit_refuse_the_reuse(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="another judge commit"):
        CS.l_state(_chunks(tmp_path, monkeypatch, OTHER), ["c1"])


def test_region_dump_commit_is_read_from_the_dump_and_head_from_the_judge():
    """One owner each: d_state passes the dump record (JR.read_region → judge_commit) and JAC.head_commit()."""
    src = (REPO / "dexengine/record/augment/cells_steps.py").read_text()
    calls = {ast.unparse(n) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)}
    assert "ACP.reuse_why(p.name, R, verdict, comps, head)" in calls and "JAC.head_commit()" in calls


# --- generation hold ----------------------------------------------------------------------------------------------


def _cond(scene: str = "s1") -> dict:
    return {"dataset": "ds", "scene": scene, "arm": "rby1", "emb": "bimanual", "hand": "h", "active": "both"}


@pytest.fixture
def fakes(monkeypatch):
    """The judge-side measurements replaced: run fields (counted), fingerprint = '<clip>|<scene>', the reason =
    rows marked 'old' differ."""
    seen = []

    def canon_cond(cond, env):
        seen.append((cond["dataset"], cond["scene"], cond["arm"], env["DEXENGINE_SCENE"]))
        return {"rs": "1"}

    class RR:
        @staticmethod
        def hand_dir():
            return "hand_dir"

    monkeypatch.setattr(GH, "canon_cond", canon_cond)
    monkeypatch.setattr(GH.JFP, "_rr", lambda: RR)
    monkeypatch.setattr(GH.JAC, "head_commit", lambda: HEAD)
    monkeypatch.setattr(JR, "read_region", lambda p: {"judge_commit": OTHER if "_old" in Path(p).name else HEAD})
    monkeypatch.setattr(GH.JFP, "fingerprint", lambda ds, emb, hand, c, scene, cond, active: f"{c}|{scene}")
    monkeypatch.setattr(GH.JFP, "err_fields", lambda fp: ["rules"] if fp.startswith("broken") else [])
    monkeypatch.setattr(
        GH.JAC,
        "gen_reason",
        lambda rows, jfp, where: f"{where[3]}: another generation" if rows[0]["gen"] == "old" else None,
    )
    return seen


def _env(cd: dict) -> dict:
    return {"DEXENGINE_SCENE": cd["scene"]}


def test_mismatched_clip_alone_is_held(tmp_path, fakes, capsys):
    clips = ["c1", "c2", "c3", "c4"]
    rows = {"c1": {"gen": "new"}, "c2": {"gen": "old"}, "c3": {"gen": "new"}, "c4": {"gen": "old"}}
    conds = {c: _cond("s2" if c == "c3" else "s1") for c in clips}
    bundle = tmp_path / "bundle"
    dump = JAC.region_path(bundle, "c4")  # a clip with a region dump is not measured here (d_state compares it)
    dump.parent.mkdir(parents=True)
    np.savez(dump, cells=np.zeros((1, 3)))
    live, held = GH.split(bundle, "ds", clips, rows, conds, _env)
    assert live == ["c1", "c3", "c4"] and held == {"c2": "c2: another generation"}
    assert sorted(fakes) == [("ds", "s1", "rby1", "s1"), ("ds", "s2", "rby1", "s2")]  # once per run-field key
    assert capsys.readouterr().out.count("[GEN_HOLD]") == 1


def test_every_clip_held_dies(tmp_path, fakes):
    with pytest.raises(SystemExit, match="every clip of the bundle"):
        GH.split(tmp_path / "b", "ds", ["c1"], {"c1": {"gen": "old"}}, {"c1": _cond()}, _env)


def test_unmeasurable_judge_input_dies_not_held(tmp_path, fakes):
    rows = {"broken1": {"gen": "new"}, "c2": {"gen": "new"}}
    with pytest.raises(SystemExit, match="cannot be measured"):
        GH.split(tmp_path / "b", "ds", ["broken1", "c2"], rows, {c: _cond() for c in rows}, _env)


def test_runner_passes_the_generation_split_to_the_one_hold_entry():
    src = (REPO / "dexengine/record/augment/cells.py").read_text()
    calls = {ast.unparse(n) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)}
    assert "GH.split(bundle, dataset, clips, rows, conds, lambda cd: judge_env(cd, ''))" in calls
    assert "ACC.holds(live, held, bundle, a.verify_only)" in calls


def test_clip_with_a_dump_of_another_commit_is_measured(tmp_path, fakes):
    """Its dump is judged again, so the generation hold measures it: a mismatched one is held instead of killing
    the judge; a dump of this commit is not measured (d_state compares it)."""
    clips = ["c_old", "c_now", "c_none"]
    bundle = tmp_path / "bundle"
    for c in ("c_old", "c_now"):
        p = JAC.region_path(bundle, c)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, cells=np.zeros((1, 3)))
    assert GH.to_measure(bundle, clips) == ["c_old", "c_none"]
    rows = {"c_old": {"gen": "old"}, "c_now": {"gen": "old"}, "c_none": {"gen": "new"}}
    live, held = GH.split(bundle, "ds", clips, rows, {c: _cond() for c in clips}, _env)
    assert held == {"c_old": "c_old: another generation"} and live == ["c_now", "c_none"]


def test_commit_rule_has_one_owner():
    """Negative of the owner: every commit comparison of the augment runner goes through cells_pick.other_commit —
    reuse_why, l_state and gen_hold.to_measure call it, and no other judge_commit-versus-head comparison exists."""
    srcs = {
        m: (REPO / f"dexengine/record/augment/{m}.py").read_text() for m in ("cells_pick", "cells_steps", "gen_hold")
    }
    fns = {}
    for m, src in srcs.items():
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.FunctionDef):
                fns[(m, n.name)] = {ast.unparse(c.func) for c in ast.walk(n) if isinstance(c, ast.Call)}
    assert "other_commit" in fns[("cells_pick", "reuse_why")]
    assert (
        "ACP.other_commit" in fns[("cells_steps", "l_state")] and "ACP.other_commit" in fns[("gen_hold", "to_measure")]
    )
    compares = [
        (m, ast.unparse(n))
        for m, src in srcs.items()
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Compare) and "judge_commit" in ast.unparse(n) and "head" in ast.unparse(n)
    ]
    assert compares == [("cells_pick", "rec['judge_commit'] != head")]
