"""The rule list: every reference resolves in its owning file, and the registry module keeps its public names."""

from __future__ import annotations

import pytest

from dexengine.core import datasets as RR
from tests.imports import own_module


def _refs(r):
    yield r.owner
    for _label, ref in r.thresholds:
        yield ref
    # the scene-rule and reader-call scopes are not file references — scope_of resolves them (tests below and
    # test_rule_reader_scope)
    if r.scope_from and not r.scope_from.startswith(RR.PROFILE_SCOPE):
        if r.scope_from not in (RR.SCENE_SCOPE, RR.READER_CALL_SCOPE):
            yield r.scope_from


def _split(ref: str):
    if ref.startswith("@"):
        return ref.split(":", 1)
    return ref.rsplit(":", 1)


def test_rule_ids_unique():
    ids = [r.id for r in RR.RULES]
    assert len(ids) == len(set(ids))


def test_every_reference_resolves():
    bad = set()
    for r in RR.RULES:
        for ref in _refs(r):
            ok, _ = RR._source_symbol(*_split(ref))
            if not ok:
                bad.add((r.id, ref))
    assert not bad, sorted(bad)
    for r in RR.RULES:
        if r.scope_from == RR.SCENE_SCOPE:
            assert isinstance(RR.scope_of(r), tuple), r.id


def test_self_references_read_the_rules_package():
    ok, val = RR._source_symbol(RR.SELF_PATH, "EXHAUST_RETRY")
    assert ok and val == RR.EXHAUST_RETRY
    assert RR._source_symbol(RR.SELF_PATH, "ruleset_retire_reason") == (True, None)
    assert RR._source_symbol(RR.SELF_PATH, "NO_SUCH_NAME") == (False, None)


def test_resolve_scopes_and_queries():
    rules = {r.id: r for r in RR.resolve()}
    for r in rules.values():
        assert set(r.datasets) <= set(RR.ALL), r.id
        assert set(r.scenes) <= set(RR.SCENES), r.id
        assert set(r.resolved) == {label for label, _ in r.thresholds}, r.id
    rid = next(r.id for r in rules.values() if r.datasets)
    ds = rules[rid].datasets[0]
    assert RR.applies(rid, ds)
    assert RR.rule_datasets(rid) == tuple(rules[rid].datasets)
    with pytest.raises(KeyError):
        RR.applies("NO-SUCH-RULE", "taco")


def test_stage_tokens_are_known_to_the_judge_fingerprint():
    IR = own_module("dexengine.placement.judge.input_record")
    for r in RR.RULES:
        for token in str(r.stage).split("·"):
            IR.stage_of(token)  # raises on a token the fingerprint does not place


def test_process_identity_is_read_when_asked(monkeypatch):
    monkeypatch.setenv("DEXENGINE_HAND", "wuji2")
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    assert RR.ACTIVE_HAND == "wuji2" and RR.ARM == "rby1"
    assert RR.combo() == "rby1_wuji2"
    monkeypatch.delenv("DEXENGINE_HAND")
    with pytest.raises(SystemExit):
        RR.active_hand()


def test_cli_answers(capsys):
    assert RR.main(["--datasets-scope", "taco,grab"]) == 0
    assert capsys.readouterr().out.split() == ["taco", "grab"]
    assert RR.main(["--datasets-scope", "nope"]) == 3
    assert RR.main(["--f0-axis", "taco"]) == 0
    assert capsys.readouterr().out.count("\t") == 2
    assert RR.main(["--f0-axis", "taco", "--f0-hand-dir", "taco"]) == 4
    assert RR.main(["--ledger-key", "taco"]) == 2


SCENE = (
    "S-carried",
    "S-levels",
    "S-prox",
    "S-touch-gate",
    "S-envelope",
    "S-true-shape",
    "S-robot-static",
    "S-hand-clear",
    "S-desk-lite",
    "S-float-settle",
    "S-dyn-pairs",
)


def _opt_out(monkeypatch, ds: str, off: dict):
    import dataclasses

    monkeypatch.setitem(RR.DATASET_PROFILE, ds, dataclasses.replace(RR.DATASET_PROFILE[ds], scene_rules_off=off))


def test_scene_rules_are_on_for_every_dataset_by_default():
    """The scene rules are the default for every dataset; none opts out."""
    rules = {r.id: r for r in RR.resolve()}
    assert {r.id for r in rules.values() if RR.is_scene_rule(r)} == set(SCENE)
    for rid in SCENE:
        assert rules[rid].scope_from == RR.SCENE_SCOPE and rules[rid].datasets == tuple(sorted(RR.ALL))
        assert all(RR.rule_on(rid, d) for d in RR.ALL), rid
        assert all(RR.applies(rid, d) for d in RR.ALL), rid
        with pytest.raises(ValueError):
            RR.rule_datasets(rid)  # callers read () as all datasets; a scene rule may be empty
    assert all(not p.scene_rules_off for p in RR.DATASET_PROFILE.values())


def test_a_dataset_opts_out_only_by_name_with_a_reason(monkeypatch):
    _opt_out(monkeypatch, "hot3d", {"S-carried": "kept as trained"})
    assert not RR.rule_on("S-carried", "hot3d") and RR.rule_on("S-carried", "taco")
    assert RR.rule_on("S-levels", "hot3d")
    assert "hot3d" not in {r.id: r for r in RR.resolve()}["S-carried"].datasets
    _opt_out(monkeypatch, "hot3d", {"S-carryed": "typo"})
    with pytest.raises(ValueError):
        RR.rule_on("S-carried", "hot3d")  # a typo never leaves a dataset silently ON
    _opt_out(monkeypatch, "hot3d", {"S-carried": " "})
    with pytest.raises(ValueError):
        RR.rule_on("S-carried", "hot3d")  # an opt-out without a reason is a silent fallback
    with pytest.raises(KeyError):
        RR.rule_on("S-carried", "no_such_dataset")
    for d in RR.ALL:
        _opt_out(monkeypatch, d, dict.fromkeys(SCENE, "all out"))
    assert not any(RR.rule_on(rid, d) for rid in SCENE for d in RR.ALL)  # empty = none, never 'all'


def test_judge_fingerprint_carries_no_scene_rule_threshold(monkeypatch):
    """The judge input record hashes the thresholds of the rules that cover a dataset at a judge stage — the scene
    rules add none: the record of every (dataset, scene) is the same whether the dataset keeps them or opts out
    (default ON leaves judge records valid)."""
    IRn = own_module("dexengine.placement.judge.input_run")
    monkeypatch.setitem(IRn._CANON, "rules", RR.resolve())
    on = {(d, sc): IRn.stage_thresholds(d, sc) for d in RR.ALL for sc in RR.SCENES}
    for d in RR.ALL:
        _opt_out(monkeypatch, d, dict.fromkeys(SCENE, "out"))
    monkeypatch.setitem(IRn._CANON, "rules", RR.resolve())
    assert on == {(d, sc): IRn.stage_thresholds(d, sc) for d in RR.ALL for sc in RR.SCENES}


def test_covers_reads_the_scope_of_an_unresolved_rule():
    """Before `resolve()` fills `datasets`, a scoped rule is read through `scope_of` — never as 'all datasets'."""
    import dataclasses

    RR.resolve()
    scoped = [r for r in RR.RULES if r.scope_from and not RR.is_scene_rule(r) and set(r.datasets) != set(RR.ALL)]
    assert scoped, "no scoped rule narrower than all datasets"
    for r in scoped:
        raw = dataclasses.replace(r, datasets=())
        assert [d for d in RR.ALL if RR.covers(raw, d)] == [d for d in RR.ALL if d in r.datasets], r.id


def test_exclusion_scope_retired_hand_dropped_unknown_raises():
    """A retired hand id in a stored exclusion line (Wuji 1, since dropped; hot3d lines still name it) is
    dropped from the scope — never widening it — and any other unknown id still raises."""
    import pytest

    from dexengine.core.locations.hands import RETIRED_HAND_IDS
    from dexengine.core.rules import ledger as L

    assert "wuji" in RETIRED_HAND_IDS
    assert L.excluded_scope("c\tr\tdg5f_s,sharpa_wave,wuji") == ("dg5f_s", "sharpa_wave")
    assert L.excluded_scope("c\tr\twuji") == ()  # applies to no current hand
    assert L.excluded_scope("c\tr\t*") is None
    with pytest.raises(SystemExit):
        L.excluded_scope("c\tr\tbogus_hand")
