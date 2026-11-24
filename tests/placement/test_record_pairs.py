"""Judge records written before the release layout read as the same rule set through the release hash pair file
(owner_moves.PAIRS_REL, a generated file of stored → current hash pairs).

- The rules fields the released judge records carry (one per dataset × scene group, rule set v5-20261012a) are
  equivalent in all three stages with the release rows only (no plugin).
- A stored record differing only in its rules field reads `same` (diff_ruleset), so the augment judge's generation
  check passes it.
- Negative: one threshold value changed today → the current hash leaves the pairs → that stage is stale and the
  generation check says "another generation". One pair row removed → that stage is stale. A row of one dataset does
  not carry another dataset with the same rules fields (the move's scope is per dataset).
"""

from __future__ import annotations

import re

import pytest

from dexengine.placement.judge import aug_cells as JAC
from dexengine.placement.judge import input_record as JFP
from dexengine.placement.judge import input_run as IR
from dexengine.placement.judge import owner_moves as OM

STAGES = list(JFP.STAGES)
#: rules field of the released judge records → (dataset, scene) it was recorded for (adopted rows).
RECORDED = {
    "j:317031c92982,b:03989dd63f1c,g:6beff5073b61": ("hrdexdb", "table"),
    "j:3e56708f168b,b:10f93b1e9c7b,g:0492a7f4b581": ("hot3d", "table"),
    "j:40e4ccb8878e,b:03989dd63f1c,g:6beff5073b61": ("grab", "table"),
    "j:578ad1976005,b:fc3467ce0e7a,g:6beff5073b61": ("arctic", "aerial"),
    "j:5b06836e2da2,b:03989dd63f1c,g:6beff5073b61": ("taco", "table"),
    "j:7dd68e28fa51,b:eb69107e1b39,g:6beff5073b61": ("hocap", "table"),  # recorded before the desk-top owner move
    "j:d7e2093d5e6e,b:10f93b1e9c7b,g:0492a7f4b581": ("oakink2", "table"),
}
#: one adopted taco record of the released data (hashes only)
STORED = (
    "jfp3|rs=v5-20261012a|rules=j:5b06836e2da2,b:03989dd63f1c,g:6beff5073b61|budget=1024|"
    "collider=16184e975b63f926+capsule-q40-05c2cdbd48201d22+guard-mesh+desk-mesh|setup=table:0.7563:|"
    "case=79639248b378|contact=6e3977e43f49|base=145637e1b06b|objcol=5c3802cd739c"
)
WHERE = ("taco", "bimanual", "sharpa_wave", "c1")


def _release_rows_only(monkeypatch) -> None:
    """The release process: its own rows only (a plugin registered by another test would also explain them)."""
    monkeypatch.setattr(OM, "moves", lambda: OM.MOVES)


def _now(ds: str, scene: str) -> str:
    return IR._rules_field(ds, scene, IR.canon_run_flags(ds, scene))


def _with_rules(fp: str, rules: str) -> str:
    return re.sub(r"rules=[^|]+", "rules=" + rules, fp)


@pytest.mark.parametrize(("stored", "where"), sorted(RECORDED.items()))
def test_recorded_fields_are_equivalent_through_the_pairs(stored, where, monkeypatch):
    _release_rows_only(monkeypatch)
    assert OM.equivalent_stages(*where, stored, _now(*where), STAGES) == set(STAGES)


def test_stored_record_reads_same_and_passes_the_generation_check(monkeypatch):
    _release_rows_only(monkeypatch)
    cur = _with_rules(STORED, _now("taco", "table"))
    assert cur != STORED
    assert JFP.diff_ruleset(STORED, cur, "ADOPTED", WHERE) == ("same", [])
    assert JAC.gen_reason([{"status": "ADOPTED", JFP.COLUMN: STORED}], cur, WHERE) is None


def test_a_changed_threshold_is_stale_and_another_generation(monkeypatch):
    _release_rows_only(monkeypatch)
    orig = IR.stage_thresholds

    def changed(ds, scene):
        th = orig(ds, scene)
        k = next(k for k, v in sorted(th["b"].items()) if isinstance(v, (int, float)) and not isinstance(v, bool))
        th["b"] = {**th["b"], k: th["b"][k] + 1}
        return th

    monkeypatch.setattr(IR, "stage_thresholds", changed)
    now = _now("taco", "table")
    stored = STORED.split("rules=")[1].split("|")[0]
    assert OM.equivalent_stages("taco", "table", stored, now, STAGES) == {"j", "g"}
    cur = _with_rules(STORED, now)
    assert JFP.diff_ruleset(STORED, cur, "ADOPTED", WHERE) == ("stale", ["rules.b"])
    why = JAC.gen_reason([{"status": "ADOPTED", JFP.COLUMN: STORED}], cur, WHERE)
    assert why and "another generation" in why and "rules.b" in why


def test_a_removed_pair_row_makes_its_stage_stale(monkeypatch):
    _release_rows_only(monkeypatch)
    stored, where = "j:5b06836e2da2,b:03989dd63f1c,g:6beff5073b61", ("taco", "table")
    now = _now(*where)
    row = (*where, "j", JFP.rules_stages(stored)["j"], JFP.rules_stages(now)["j"])
    full = OM.pairs()
    assert row in full
    monkeypatch.setattr(OM, "pairs", lambda: full - {row})
    assert OM.equivalent_stages(*where, stored, now, STAGES) == {"b", "g"}


def test_a_row_of_one_dataset_does_not_carry_another(monkeypatch):
    _release_rows_only(monkeypatch)
    stored, now = "j:5b06836e2da2,b:03989dd63f1c,g:6beff5073b61", _now("taco", "table")
    other = next(ds for ds in ("hocap", "dexcanvas") if _now(ds, "table") == now)
    assert OM.equivalent_stages(other, "table", stored, now, STAGES) == set(STAGES)
    taco_only = frozenset(r for r in OM.pairs() if r[0] == "taco")
    monkeypatch.setattr(OM, "pairs", lambda: taco_only)
    assert OM.equivalent_stages("taco", "table", stored, now, STAGES) == set(STAGES)
    assert OM.equivalent_stages(other, "table", stored, now, STAGES) == set()


def test_pair_file_holds_stage_hashes_only(monkeypatch):
    _release_rows_only(monkeypatch)
    rows = OM.pairs()
    assert rows and all(
        s in STAGES and re.fullmatch(r"[0-9a-f]{12}", a) and re.fullmatch(r"[0-9a-f]{12}", b)
        for _d, _c, s, a, b in rows
    )
