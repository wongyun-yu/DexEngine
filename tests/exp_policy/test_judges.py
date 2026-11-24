"""E5 judges: the rules files and the pure functions the judges are built from."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

JUDGES = Path(__file__).resolve().parents[2] / "experiments" / "sim_eval" / "judges"


def test_rules_files_are_pinned():
    """Each judge record stores the sha256 of the rules file it was judged with, so the rules bytes are pinned."""
    md5 = {p: hashlib.md5((JUDGES / p).read_bytes()).hexdigest() for p in ("rules_v1.yaml", "rules_v2.yaml")}
    assert md5 == {
        "rules_v1.yaml": "71201323e0155b4cd1d8b23124db2076",
        "rules_v2.yaml": "41f7f4ca611ad11e3e3bb8f1edc689a6",
    }


def test_geometry_helpers():
    from experiments.sim_eval.judges import geometry as G
    from experiments.sim_eval.judges import place_ref as PR

    assert np.allclose(G.quat_R(np.array([1.0, 0, 0, 0])), np.eye(3))
    q = np.array([[0.7071, 0, 0, 0.7071], [0.2, 0.3, -0.5, 0.8]])
    assert np.array_equal(G.quat_R(q), PR.quat_R(q))  # the numpy-only copy in place_ref is the same formula
    m = np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=bool)
    assert G.longest(m, hz=1.0) == 3.0
    assert (
        G.first_hold(m, 3.0, hz=1.0) == 6 and G.first_hold(m, 4.0, hz=1.0) is None and G.first_hold(m, 0.0, hz=1.0) == 1
    )
    p = np.array([[0, 0, 0], [0.1, 0, 0]], dtype=float)
    lin, ang = G.speeds(p, np.repeat(np.eye(3)[None], 2, 0), hz=50.0)
    assert np.allclose(lin, [0.0, 5.0]) and np.allclose(ang, 0.0)


def test_rules_validation_rejects_unknown_values():
    from experiments.sim_eval.judges import judge_v2 as SF

    with open(JUDGES / "rules_v2.yaml") as fh:
        rules = yaml.safe_load(fh)
    lv, ht = SF.v2_of(rules)
    assert lv["gt"] == "data" and ht == "both" and SF.travel_of(rules) == "tracked"
    with pytest.raises(SystemExit):
        SF.travel_of({"sweep": {"travel": "fastest"}})
    with pytest.raises(SystemExit):
        SF.v2_of({"common": {}, "sweep": {"travel": "nearest", "hold_tool": "both"}})


def test_home_map_and_summarize(tmp_path):
    from experiments.sim_eval.judges import judge_v1 as J

    assert J.map_home("/r/a/x/job.json", "/r/a=/r/b") == Path("/r/b/x/job.json")
    with pytest.raises(SystemExit):
        J.map_home("/elsewhere/job.json", "/r/a=/r/b")
    recs = [
        {"judgeable": True, "judgeable_why": [], "invalid": None, "success": True},
        {
            "judgeable": False,
            "judgeable_why": ["reference failed: hold not met ['x']"],
            "invalid": None,
            "success": False,
        },
        {"judgeable": True, "judgeable_why": [], "invalid": "blow-up x 9.00 m/s", "success": False},
    ]
    runs = []
    for i, rec in enumerate(recs + [None]):
        r = tmp_path / f"run{i}"
        (r / "m").mkdir(parents=True)
        (r / "job_run.json").write_text(json.dumps({"axis": "Lift"}))
        if rec is not None:
            (r / "m" / "judge.json").write_text(json.dumps(rec, ensure_ascii=False))
        runs.append(r)
    t = J.summarize(runs, "m")["Lift"]
    assert (t["n"], t["no_judge"], t["invalid"], t["judgeable"], t["success"], t["rate"]) == (4, 1, 1, 1, 1, 1.0)
    assert t["dropped"] == {"reference failed": 1}


def _summarize_one(tmp_path, recs):
    from experiments.sim_eval.judges import judge_v1 as J

    runs = []
    for i, rec in enumerate(recs):
        r = tmp_path / f"run{i}"
        (r / "m").mkdir(parents=True)
        (r / "job_run.json").write_text(json.dumps({"axis": "Lift"}))
        (r / "m" / "judge.json").write_text(json.dumps(rec, ensure_ascii=False))
        runs.append(r)
    return J.summarize(runs, "m")["Lift"]
