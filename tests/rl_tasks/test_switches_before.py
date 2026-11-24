"""Records written before a switch existed (CPU). The fixture is a switches file an earlier switches module wrote for
a real launch: it lacks every switch added since. It loads with each such switch filled from its `before` declaration
and a marker; a missing switch without a declaration still fails; every switch the fixture lacks must carry a
declaration (a new switch without one fails here)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dexengine.rl.tasks import switches as S

TRUNK = Path(__file__).parent / "data" / "switches_trunk_20261118.json"


def test_a_trunk_record_loads_with_the_declared_values_and_a_marker(capsys):
    sw, given = S.read(TRUNK)
    out = capsys.readouterr().out
    added = sorted(set(S.FIELDS) - set(json.loads(TRUNK.read_text(encoding="utf-8"))["values"]))
    assert added and set(added) <= set(given)
    for k in added:
        assert getattr(sw, k) == S.BEFORE[k]
        assert f"filled {k}={S.BEFORE[k]!r} (declared value before the switch existed)" in out


def test_every_switch_a_trunk_record_lacks_declares_its_value_before():
    """A switch added after the fixture was written without a `before` declaration would make every older record
    unreadable: it fails here first."""
    have = set(json.loads(TRUNK.read_text(encoding="utf-8"))["values"])
    assert sorted(set(S.FIELDS) - have - set(S.BEFORE)) == []


def test_a_missing_switch_without_a_declaration_fails(tmp_path):
    body = json.loads(TRUNK.read_text(encoding="utf-8"))
    undeclared = next(k for k in body["values"] if k not in S.BEFORE)
    del body["values"][undeclared]
    body["given"] = [g for g in body["given"] if g != undeclared]
    p = tmp_path / "switches.json"
    p.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(S.SwitchError, match=f"missing \\['{undeclared}'\\]"):
        S.read(p)
