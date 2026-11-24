"""scene_rules.apply(..., kinds_only=True) (f0's scene rows): the envelope pass gives every row the class, kind and
static statement it gives with the overlap measured, and runs no overlap measure; the default (kinds_only=False) still
measures."""

from __future__ import annotations

import copy

import numpy as np


def _setup(monkeypatch):
    from dexengine.data.objects import envelope as EV
    from dexengine.data.objects import extra_slots as XS
    from dexengine.data.objects import scene_rules as SC

    calls = []

    def overlap(tree_dir, ti, q, r, o, movers):
        calls.append(r["name"])
        r["envelope"]["depth_m"] = 0.02
        r["overlap_drop"] = {"partner": "m", "depth_m": 0.02}

    monkeypatch.setattr(EV, "overlap", overlap)
    monkeypatch.setattr(EV, "swept", lambda *a: None)
    monkeypatch.setattr(EV, "gap", lambda placed, sweep: (0.01, "m"))
    monkeypatch.setattr(EV, "no_shape", lambda ds, o: "")
    monkeypatch.setattr(EV.XG, "geom", lambda tree_dir, ti, q: (lambda o, f: None, None))
    monkeypatch.setattr(SC, "on", lambda rule, ds: rule == EV.RULE_ID)
    ti = {
        "dataset_name": "x",
        "objects": [
            {"name": "m", "role": "main", "mesh_dir": "a", "qpos_start": 0},
            {"name": "e0", "role": "extra", "mesh_dir": "b", "qpos_start": 7},
            {"name": "e1", "role": "extra", "mesh_dir": "c", "qpos_start": 14},
        ],
    }
    rows = [{"name": n, "cls": XS.VISUAL_ONLY, "why": "still", "static": False} for n in ("e0", "e1")]
    return SC, ti, rows, calls


def test_classes_same_without_overlap(monkeypatch):
    SC, ti, rows, calls = _setup(monkeypatch)
    q = np.zeros((3, 21))
    on, off = copy.deepcopy(rows), copy.deepcopy(rows)
    SC.apply("t", ti, q, on, lambda o: "")
    assert calls == ["e0", "e1"]  # the default measures
    SC.apply("t", ti, q, off, lambda o: "", kinds_only=True)
    assert calls == ["e0", "e1"]  # kinds_only: no measure
    keep = ("name", "cls", "kind", "static", "why", "still")
    assert [{k: r.get(k) for k in keep} for r in on] == [{k: r.get(k) for k in keep} for r in off]
    assert all("overlap_drop" not in r and "depth_m" not in r["envelope"] for r in off)
