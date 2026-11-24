"""Placement-search ablation: items of a row, the slot assignment and the row summary."""

import math

import pytest

from experiments.ablations.placement import aggregate as A
from experiments.ablations.placement import rows as R


def test_items_cover_the_set_once_and_assignment_is_lpt():
    items = R.items_for("a")
    clips = [c for it in items for c in it["clips"].split(",")]
    want = [r["clip"] for r in R.read_set(R.ROWS["a"][0])]
    assert sorted(clips) == sorted(want) and len(clips) == len(set(clips))
    assert len({it["tag"] for it in items}) == len(items)
    assert all(it["n"] <= R.ROWS["a"][2] for it in items)
    q, load = R.assign([{"cost": c} for c in (5.0, 4.0, 3.0, 3.0, 1.0)], ["g0", "g1"])
    assert [it["cost"] for it in q["g0"]] == [5.0, 3.0] and [it["cost"] for it in q["g1"]] == [4.0, 3.0, 1.0]
    assert load == {"g0": 8.0, "g1": 8.0}
    with pytest.raises(SystemExit):
        R.items_for("a", only={"not_a_clip"})


def test_summarize_counts_technical_ends_as_missing():
    rec = {
        "c1": {"status": "ADOPTED", "stretch": 1.2, "wall": 10.0, "first": True},
        "c2": {"status": "ADOPTED", "stretch": 1.4, "wall": 20.0, "first": False},
        "c3": {"status": "EXHAUSTED", "stretch": None, "wall": 30.0, "first": False},
        "c4": {"status": "BAKE_ERROR", "stretch": None, "wall": 5.0},
    }
    s = A.summarize("a", ["c1", "c2", "c3", "c4", "c5"], rec)
    assert (s["n"], s["have"], s["missing"], s["accepted"]) == (5, 3, ["c4", "c5"], 2)
    assert math.isclose(s["acc_pct"], 40.0) and math.isclose(s["st_mean"], 1.3)
    assert s["wall_med"] == 20.0 and s["wall_med_warm"] == 25.0 and s["n_warm"] == 2
    assert A.p90([3, 1, 2]) == 3 and math.isnan(A.p90([]))
