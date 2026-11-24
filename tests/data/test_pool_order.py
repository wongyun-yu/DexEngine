"""Records built from a pool's results do not depend on the pool size (capacity may pick any): results that arrive in
finish order are put in clip or target order before anything is written. Before the fix, a base ledger with two
failures listed them in the order the workers finished."""

from __future__ import annotations

import json
import random
import sys
import time
from types import SimpleNamespace

from dexengine.data import base_adapter as BA
from dexengine.data.contact import sidecar as SC


def _res(clip: str, err: str = "") -> dict:
    return {"clip": clip, "out": [] if err else [clip], "skipped": [], "err": err, "merged": 0, "art_groups": 0}


def test_the_base_ledger_lists_failures_in_clip_order():
    results = [_res("c1"), _res("c3", "boom 3"), _res("c2", "boom 2"), _res("c4")]
    jobs = [{"clip": r["clip"], "emb": "bimanual", "sup": {}} for r in results]
    for r in results:
        r["supplemented"] = 0
    yaw = (0.0, {}, {})

    def rec(order):
        d = BA._base_record("hot3d", "sharpa_wave", yaw, jobs, {}, 4, [results[i] for i in order])
        d.pop("time")
        return json.dumps(d)

    a, b = rec([0, 1, 2, 3]), rec([3, 2, 1, 0])
    assert a == b
    assert list(json.loads(a)["fails"]) == ["c2", "c3"]


def _fake_tree(job):
    time.sleep(random.random() / 50)  # finish order differs from job order
    return {"clip": job["clip"], "emb": job["emb"], "out": [], "fresh": [], "conflict": [], "err": ""}


def test_the_sidecar_results_come_back_in_clip_order(monkeypatch):
    monkeypatch.setattr(SC, "do_tree", _fake_tree)
    jobs = [{"clip": f"c{i:02d}", "emb": e} for i in range(12) for e in ("right", "bimanual")]
    random.shuffle(jobs)
    for w in (1, 4):
        got = SC._run_jobs(SimpleNamespace(workers=w), jobs)
        assert [(r["clip"], r["emb"]) for r in got] == sorted((j["clip"], j["emb"]) for j in jobs)


def _fake_case(t):
    time.sleep(random.random() / 50)
    return {"tree": t}, False, None


def test_the_case_records_come_back_in_target_order(monkeypatch):
    """taxonomy reads the hand's thresholds at import: imported here under a hand and dropped again afterwards (no
    module built under this hand stays for later tests)."""
    import dexengine.data.cases as cases

    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")
    try:
        from dexengine.data.cases import taxonomy as TX

        monkeypatch.setattr(TX, "_job", _fake_case)
        targets = [f"t{i}" for i in range(16)]
        for w in (1, 4):
            recs, _n, _g = TX._run(targets, w, time.time())
            assert [r["tree"] for r in recs] == targets
    finally:
        sys.modules.pop("dexengine.data.cases.taxonomy", None)
        if hasattr(cases, "taxonomy"):
            delattr(cases, "taxonomy")
