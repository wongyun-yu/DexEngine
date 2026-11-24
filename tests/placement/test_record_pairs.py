"""Judge records written before the release layout read as the same rule set through the release hash pair file
(owner_moves.PAIRS_REL, a generated file of stored → current hash pairs).

- The rules fields the released judge records carry (one per dataset × scene group, rule set v5-20261012a) are
  equivalent in the judge and bake stages with the release rows only (no plugin). Their gate
  stage is stale: the gate card J-gate-dynpairs-xr (S-dyn-pairs after cross repair) entered the gate stage's rules
  hash after they were written, so those gate records are judged again (no row declares them equivalent: that would
  hide a real rule difference). Negative: without the card, the gate stage reads as the reference again through
  the gate rows the pair file held before the card — the card is the whole difference.
- A stored record differing only in its rules field reads `stale` at the gate only, and the augment judge's generation
  check names `rules.g`; the same record re-gated under the card reads `same` and passes the check.
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
#: The gate card that entered the gate stage's hash after the reference records were written (rules registry id).
NEW_GATE_CARD = "J-gate-dynpairs-xr"
#: Stages in which the reference records read as today's (the gate stage is stale through NEW_GATE_CARD).
OLD_SAME = {"j", "b"}
#: Gate rows of the pair file before the card: dataset · scene · stage · stored · current hash.
PRE_CARD_GATE = {
    ("taco", "table", "g", "6beff5073b61", "c83f81ed444e"),
    ("hot3d", "table", "g", "0492a7f4b581", "6ef231f85ca0"),
}


def _release_rows_only(monkeypatch) -> None:
    """The release process: its own rows only (a plugin registered by another test would also explain them)."""
    monkeypatch.setattr(OM, "moves", lambda: OM.MOVES)


def _now(ds: str, scene: str) -> str:
    return IR._rules_field(ds, scene, IR.canon_run_flags(ds, scene))


def _with_rules(fp: str, rules: str) -> str:
    return re.sub(r"rules=[^|]+", "rules=" + rules, fp)


def _regated(stored: str, ds: str, scene: str) -> str:
    """A reference rules field re-gated under the card: its gate hash is the one the pair file's gate row of (ds, scene)
    maps to today's gate hash; its judge and bake hashes stay as stored."""
    now_g = JFP.rules_stages(_now(ds, scene))["g"]
    g = sorted(a for d, c, s, a, b in OM.pairs() if (d, c, s, b) == (ds, scene, "g", now_g))
    assert len(g) == 1, g
    st = JFP.rules_stages(stored)
    return ",".join(f"{k}:{g[0] if k == 'g' else st[k]}" for k in STAGES)


@pytest.mark.parametrize(("stored", "where"), sorted(RECORDED.items()))
def test_recorded_fields_are_equivalent_through_the_pairs(stored, where, monkeypatch):
    _release_rows_only(monkeypatch)
    assert OM.equivalent_stages(*where, stored, _now(*where), STAGES) == OLD_SAME


def test_the_gate_stage_is_stale_through_the_new_card_only(monkeypatch):
    _release_rows_only(monkeypatch)
    rules = IR._rr().resolve()
    assert any(r.id == NEW_GATE_CARD for r in rules)
    monkeypatch.setitem(IR._CANON, "rules", [r for r in rules if r.id != NEW_GATE_CARD])
    full = OM.pairs()
    monkeypatch.setattr(OM, "pairs", lambda: full | PRE_CARD_GATE)
    checked = [(st, w) for st, w in sorted(RECORDED.items()) if (*w, "g") in {r[:3] for r in PRE_CARD_GATE}]
    assert len(checked) == len(PRE_CARD_GATE)
    for stored, where in checked:
        assert OM.equivalent_stages(*where, stored, _now(*where), STAGES) == set(STAGES)


def test_stored_record_reads_stale_at_the_gate_only(monkeypatch):
    _release_rows_only(monkeypatch)
    cur = _with_rules(STORED, _now("taco", "table"))
    assert cur != STORED
    assert JFP.diff_ruleset(STORED, cur, "ADOPTED", WHERE) == ("stale", ["rules.g"])
    why = JAC.gen_reason([{"status": "ADOPTED", JFP.COLUMN: STORED}], cur, WHERE)
    assert why and "another generation" in why and "rules.g" in why and "rules.j" not in why and "rules.b" not in why


def test_a_regated_record_reads_same_and_passes_the_generation_check(monkeypatch):
    _release_rows_only(monkeypatch)
    stored = _with_rules(STORED, _regated(STORED.split("rules=")[1].split("|")[0], "taco", "table"))
    cur = _with_rules(STORED, _now("taco", "table"))
    assert cur != stored
    assert JFP.diff_ruleset(stored, cur, "ADOPTED", WHERE) == ("same", [])
    assert JAC.gen_reason([{"status": "ADOPTED", JFP.COLUMN: stored}], cur, WHERE) is None


def test_a_changed_threshold_is_stale_and_another_generation(monkeypatch):
    _release_rows_only(monkeypatch)
    orig = IR.stage_thresholds

    def changed(ds, scene):
        th = orig(ds, scene)
        k = next(k for k, v in sorted(th["b"].items()) if isinstance(v, (int, float)) and not isinstance(v, bool))
        th["b"] = {**th["b"], k: th["b"][k] + 1}
        return th

    monkeypatch.setattr(IR, "stage_thresholds", changed)
    stored = _regated(STORED.split("rules=")[1].split("|")[0], "taco", "table")
    now = _now("taco", "table")
    assert OM.equivalent_stages("taco", "table", stored, now, STAGES) == {"j", "g"}
    cur, rec = _with_rules(STORED, now), _with_rules(STORED, stored)
    assert JFP.diff_ruleset(rec, cur, "ADOPTED", WHERE) == ("stale", ["rules.b"])
    why = JAC.gen_reason([{"status": "ADOPTED", JFP.COLUMN: rec}], cur, WHERE)
    assert why and "another generation" in why and "rules.b" in why


def test_a_removed_pair_row_makes_its_stage_stale(monkeypatch):
    _release_rows_only(monkeypatch)
    where = ("taco", "table")
    stored, now = _regated("j:5b06836e2da2,b:03989dd63f1c,g:6beff5073b61", *where), _now(*where)
    row = (*where, "j", JFP.rules_stages(stored)["j"], JFP.rules_stages(now)["j"])
    full = OM.pairs()
    assert row in full
    monkeypatch.setattr(OM, "pairs", lambda: full - {row})
    assert OM.equivalent_stages(*where, stored, now, STAGES) == {"b", "g"}


def test_a_row_of_one_dataset_does_not_carry_another(monkeypatch):
    _release_rows_only(monkeypatch)
    stored, now = _regated("j:5b06836e2da2,b:03989dd63f1c,g:6beff5073b61", "taco", "table"), _now("taco", "table")
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
