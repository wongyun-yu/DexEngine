"""Rule S-levels on synthetic scenes (CPU): observed supports at other heights carry what rests on them — no disc there
— and the rule is OFF for every existing dataset."""

from __future__ import annotations

import dataclasses

import numpy as np

from dexengine.core import datasets as RR
from tests.imports import own_module

OS = own_module("dexengine.placement.coords.observed_support")
SK = own_module("dexengine.placement.coords.support_kind")


def _box(c, half, n=7):
    g = np.linspace(-1, 1, n)
    a, b = np.meshgrid(g, g)
    faces = [
        np.c_[a.ravel(), b.ravel(), np.full(a.size, s)][:, p]
        for s in (-1, 1)
        for p in ([0, 1, 2], [0, 2, 1], [2, 0, 1])
    ]
    return np.unique(np.concatenate(faces), axis=0) * np.asarray(half, float) + np.asarray(c, float)


def test_observed_supports_are_static_receivers_of_that_main():
    rows = [
        {"name": "box", "cls": "STATIC_SPAWN", "kind": "receiver", "recv_by": "cup"},
        {"name": "plate", "cls": "STATIC_SPAWN", "kind": "touched"},
        {"name": "tray", "cls": "STATIC_SPAWN", "kind": "receiver", "recv_by": "spoon"},
        {"name": "bowl", "cls": "VISUAL_ONLY", "kind": None, "recv_by": "cup"},
    ]
    assert OS.observed_supports(rows, {"cup"}) == {"cup": ["box"]}


def test_disc_on_an_observed_support_is_not_raised():
    # a cup resting on a box at 0.20 m (multi-level: the box top is the measured support) and a mug on nothing observed
    V = {
        "box": _box((0.0, 0.0, 0.10), (0.08, 0.08, 0.10)),
        "cup": _box((0.0, 0.0, 0.25), (0.03, 0.03, 0.05)),
        "mug": _box((0.4, 0.0, 0.25), (0.03, 0.03, 0.05)),
    }
    discs = [{"obj": "cup", "kind": "rest", "frame": 12}, {"obj": "mug", "kind": "rest", "frame": 12}]
    kept, dropped = OS.carried_by_observed(
        discs, lambda d: d["frame"], lambda n, f: V[n], {"cup": ["box"]}, SK.on_support_object
    )
    assert [d["obj"] for d in dropped] == ["cup"] and [d["obj"] for d in kept] == ["mug"]
    # the same cup lifted off the box (gap above the support band) keeps its disc
    V["cup"] = _box((0.0, 0.0, 0.40), (0.03, 0.03, 0.05))
    kept, dropped = OS.carried_by_observed(
        discs[:1], lambda d: d["frame"], lambda n, f: V[n], {"cup": ["box"]}, SK.on_support_object
    )
    assert not dropped and len(kept) == 1


def test_levels_rule_on_unless_opted_out(monkeypatch):
    assert all(OS.on({"dataset_name": d}) for d in RR.ALL)
    assert not OS.on({"dataset_name": "not_registered"})
    p = RR.DATASET_PROFILE["oakink2"]
    off = {OS.LEVELS_RULE: "test: discs as stamped before"}
    monkeypatch.setitem(RR.DATASET_PROFILE, "oakink2", dataclasses.replace(p, scene_rules_off=off))
    assert not OS.on({"dataset_name": "oakink2"}) and OS.on({"dataset_name": "hot3d"})
