"""E5 report and scene-rule helpers: pure-function checks (no stored data needed)."""

from __future__ import annotations

import pytest


def _spec(tmp_path, rows):
    p = tmp_path / "spec.tsv"
    p.write_text("task\twho\trun\tlevel\tjson\tsupersedes\n" + "".join("\t".join(r) + "\n" for r in rows))
    return str(p)


def test_verdict_values_and_batch_tags():
    from experiments.sim_eval.report import verdicts as V

    assert V.JV == ("success", "fail", "invalid")  # stored judge values; readers compare against them
    assert V.tag_of("/x/run_a_166650/L0.json") == ""
    assert V.tag_of("/x/run_a_166650_dfin2x_texx/L0.json") == "dfin2x_texx"
    with pytest.raises(ValueError):
        V.tag_of("/x/run_a/L0.json")


def test_batch_overlap_rule(tmp_path):
    from experiments.sim_eval.report import verdicts as V

    a, b = "/x/r_000001_p1/L0.json", "/x/r_000001_p2/L0.json"
    sup = V.supersedes(_spec(tmp_path, [("pour", "ours", "r", "L0", b, "p1")]))
    assert sup == {("r", "L0", "p2"): {"p1"}}
    assert V.overlap("fail", a, "success", b, "r", "L0", sup) == ("new", "p1")  # explicit supersession wins
    assert V.overlap("success", b, "fail", a, "r", "L0", sup) == ("old", "p1")
    assert V.overlap("judge failed", a, "fail", b, "r", "L0", {}) == ("new", None)  # judged beats unjudged
    assert V.overlap("success", a, "unjudgeable", b, "r", "L0", {}) == ("old", None)
    with pytest.raises(ValueError):
        V.overlap("success", a, "fail", b, "r", "L0", {})  # two judged, no supersession
    with pytest.raises(ValueError):
        V.overlap("success", a, "fail", b, "r", "L0", {("r", "L0", "p1"): {"p2"}, ("r", "L0", "p2"): {"p1"}})
