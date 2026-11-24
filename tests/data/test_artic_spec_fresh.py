"""The articulation spec is generated under the current merged part kinds. A part kind changed after the spec
was generated (the toothpaste cap free → revolute without the spec stage) reaches the trees as PART_UNMERGED and
loses clips; the spec records part_kinds.merged_fingerprint and this gate, and the spec stage's verify mode, fail on a
mismatch."""

from __future__ import annotations

import json

import pytest

from dexengine.core import datasets as RR
from dexengine.data.artic import spec_collect as SCo
from dexengine.data.oakink2 import part_kinds as PK


def test_committed_spec_carries_the_current_part_kinds():
    doc = json.loads(RR.axis_output(RR.ARTIC_SPEC).read_text(encoding="utf-8"))
    assert SCo.part_kinds_stale(doc) == ""


def test_a_kind_change_makes_the_spec_stale(monkeypatch):
    doc = {SCo.FP_KEY: PK.merged_fingerprint()}
    assert SCo.part_kinds_stale(doc) == ""
    rid = next(r for r, (k, _p) in sorted(PK.SPEC.items()) if k == "revolute")
    monkeypatch.setattr(PK, "SPEC", {**PK.SPEC, rid: ("free", PK.SPEC[rid][1])})  # a hinge turned free
    assert "--remeasure" in SCo.part_kinds_stale(doc)
    monkeypatch.undo()
    assert SCo.part_kinds_stale(doc) == ""


def test_a_spec_without_the_record_is_stale():
    assert "missing" in SCo.part_kinds_stale({"objects": {}})


def test_verify_mode_dies_on_a_stale_spec(tmp_path):
    f = tmp_path / "artic_ranges.json"
    f.write_text(json.dumps({"objects": {}, SCo.FP_KEY: "0" * 16}), encoding="utf-8")
    with pytest.raises(SystemExit, match="part kinds fingerprint"):
        SCo.update(write=False, out=f)
