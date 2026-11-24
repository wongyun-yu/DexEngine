"""The rule list: every reference resolves in its owning file, and the registry module keeps its public names."""

from __future__ import annotations

import pytest

from dexengine.core import datasets as RR
from tests.imports import own_module


def _refs(r):
    yield r.owner
    for _label, ref in r.thresholds:
        yield ref
    # the reader-call scope is not a file reference — scope_of resolves it (test_rule_reader_scope)
    if r.scope_from and not r.scope_from.startswith(RR.PROFILE_SCOPE):
        if r.scope_from != RR.READER_CALL_SCOPE:
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


def test_covers_reads_the_scope_of_an_unresolved_rule():
    """Before `resolve()` fills `datasets`, a scoped rule is read through `scope_of` — never as 'all datasets'."""
    import dataclasses

    RR.resolve()
    scoped = [r for r in RR.RULES if r.scope_from and set(r.datasets) != set(RR.ALL)]
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
