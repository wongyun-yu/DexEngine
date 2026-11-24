"""repair_sep.scene_rows: the scene rows of a tree (extra_slots.clip_rows) are computed once per process for the same
tree, task_info and qpos rows; each caller gets its own copy; echo writes the markers the rules printed, else they
are dropped; other qpos rows are computed again."""

from __future__ import annotations

import numpy as np


def _fake(calls):
    def clip_rows(tree, ti, q, strict=True, kinds_only=False):
        calls.append((tree, float(q.sum()), strict, kinds_only))
        print(f"[RULE] {tree} rows={len(ti['objects'])}")
        return [{"name": o["name"], "cls": "slot", "vals": [float(q.sum())]} for o in ti["objects"]]

    return clip_rows


def test_one_computation_own_copies_markers(monkeypatch, capsys):
    from dexengine.data.objects import extra_slots as XS
    from dexengine.retarget.f0 import repair_sep as RSEP
    from dexengine.retarget.f0 import repair_sep_items as RSI

    calls = []
    monkeypatch.setattr(XS, "clip_rows", _fake(calls))
    monkeypatch.setattr(RSI, "_ROWS", {})  # the memo lives with scene_rows (repair_sep re-exports the function)
    ti, q = {"objects": [{"name": "a"}, {"name": "b"}]}, np.arange(6.0).reshape(2, 3)
    r1 = RSEP.scene_rows("/t/x", ti, q, echo=False)
    assert capsys.readouterr().out == ""  # dropped, as under stdio.capture
    r2 = RSEP.scene_rows("/t/x", ti, q.copy(), echo=True)
    assert capsys.readouterr().out == "[RULE] /t/x rows=2\n"  # the markers a direct call prints
    assert len(calls) == 1 and calls[0][2] is False and calls[0][3] is True  # f0 reads classes and kinds only
    assert r1 == r2 and r1 is not r2 and r1[0] is not r2[0]
    r1[0]["vals"].append(1.0)
    assert RSEP.scene_rows("/t/x", ti, q, echo=False)[0]["vals"] == [15.0]
    RSEP.scene_rows("/t/x", ti, q + 1.0, echo=False)  # other rows: computed again
    RSEP.scene_rows("/t/x", {"objects": [{"name": "a"}]}, q, echo=False)  # other task_info: computed again
    assert len(calls) == 3
