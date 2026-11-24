"""Judge input record: field encoding, the comparison rules and the run-field helpers (no data needed)."""

from __future__ import annotations

import pytest

from dexengine.placement.judge import input_record as R
from dexengine.placement.judge import input_run as IR

RUN = {"rs": "v", "rules": "j:a,b:b,g:c", "budget": "1024", "collider": "d"}
CLIP = {"setup": "table:0.0000:", "case": "c", "contact": "none", "base": "q", "objcol": "none"}


def test_encode_parse_round_trip():
    s = R.encode(CLIP, RUN)
    assert s.startswith(R.SCHEMA + "|")
    assert R.parse(s) == {**RUN, **CLIP}
    with pytest.raises(ValueError):
        R.encode({**CLIP, "case": "a|b"}, RUN)


def test_diff_budget_is_monotone_and_rules_follow_decided_stages():
    stored = R.encode(CLIP, RUN)
    assert R.diff(stored, R.encode(CLIP, {**RUN, "budget": "512"}), "EXHAUSTED") == ("same", [])
    assert R.diff(stored, R.encode(CLIP, {**RUN, "budget": "2048"}), "EXHAUSTED") == ("stale", ["budget"])
    gate_only = R.encode(CLIP, {**RUN, "rules": "j:a,b:b,g:z"})
    assert R.diff(stored, gate_only, "EXHAUSTED") == ("same", [])  # judge-dropped rows: judge stage only
    assert R.diff(stored, gate_only, "GUARD_FAIL") == ("stale", ["rules.g"])


def test_flag_text_config_refs_and_stage_positions():
    assert IR._norm(True) == "1" and IR._norm("") == "1" and IR._norm("1.0") == "1" and IR._norm(1200.0) == "1200"
    assert IR._flag_of("configs/placement/judge.yaml:exhaust_max") == "--exhaust-max"
    assert IR._flag_of("dexengine/placement/judge/core.py:--jump-max") == "--jump-max"
    assert IR._flag_of("dexengine/placement/judge/core.py:SOME_CONST") == "SOME_CONST"
    assert IR.stage_of("bake(f0)") == ("j", "b")
    assert IR.stage_of("judge(prediction)") == ("j",)
    with pytest.raises(ValueError):
        IR.stage_of("not-a-stage")


def test_owner_move_revert(monkeypatch):
    """The owner-move rule on synthetic rows: revert, and the equivalence of a record hashed with an old owner symbol
    once a plugin registers the row."""
    from dexengine.placement.judge import owner_moves as OM
    from dexengine.record import hooks

    row = OM.Move("c1", "2999-01-01", {"new.py:X": "old.py:X"}, {"new.py:Y": ("old.py:Y0", 0.5)}, ("all", None), "t")
    assert OM.revert({"j": {"thresholds": {"new.py:X": 1.0}}}, row) == {"j": {"thresholds": {"old.py:X": 1.0}}}
    assert OM.revert({"j": {"thresholds": {"new.py:Y": 0.5}}}, row) == {"j": {"thresholds": {}}}
    assert OM.revert({"j": {"thresholds": {"new.py:Y": 99.0}}}, row) is None  # a changed value is not a move
    flags = IR.canon_run_flags("taco", "table")
    body, cur = IR._rules_body("taco", "table", flags), IR._rules_field("taco", "table", flags)
    sym = next(iter(body["j"]["thresholds"]))
    moved = OM.Move("c2", "2999-01-02", {sym: "old/place.py:" + sym.rsplit(":", 1)[-1]}, {}, ("all", None), "synthetic")
    old = OM.revert(body, moved)
    stored = ",".join(f"{st}:{R._h(old[st])}" for st in R.STAGES)
    stages = list(R.STAGES)
    assert stored != cur
    monkeypatch.setitem(hooks._REGISTRY[OM.HOOK_KIND], "synthetic", ())
    assert OM.equivalent_stages("taco", "table", stored, cur, stages) == set()  # no row: stale
    monkeypatch.setitem(hooks._REGISTRY[OM.HOOK_KIND], "synthetic", (moved,))
    assert OM.equivalent_stages("taco", "table", stored, cur, stages) == set(stages)
    assert OM.equivalent_stages("taco", "table", stored.replace("j:", "j:0", 1), cur, ["j"]) == set()
    monkeypatch.setitem(hooks._REGISTRY[OM.HOOK_KIND], "bad", (moved._replace(scope=("nope", None)),))
    with pytest.raises(KeyError):
        OM.moves()
