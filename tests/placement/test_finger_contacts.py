"""Cross repair contact: per side, per finger, from the canonical sidecars mapped onto the tree (finger_contacts).

The sidecar reader and the base inputs are stubbed; what is tested is the mapping the cross repair relies on: tree
frame t reads base frame t - P (the f0 pad reads base frame 0), each hand keeps its own five fingers, a side the base
does not expect holds nothing, and an expected side without a sidecar or a sidecar of another length dies."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dexengine.placement.coords import contacts as CT


def _stub(monkeypatch, sidecars: dict, expected: tuple, pad: int):
    sf = SimpleNamespace(
        LAYER_BASE="base",
        sides_for=lambda ti, layer: expected,
        TreeInputs=SimpleNamespace(load=lambda *a: SimpleNamespace(ti={})),
    )
    paths = SimpleNamespace(hand_spec_dir=lambda h: "spec", hand_id=lambda h: h, main_root=lambda: "/root")
    monkeypatch.setattr(CT, "_sf_mod", lambda: sf)
    monkeypatch.setattr(CT, "_paths", lambda: paths)
    monkeypatch.setattr(CT, "sidecar_base_tree", lambda tree, ti: "base")
    monkeypatch.setattr(CT, "tree_f0_pad", lambda tree, ti: pad)
    monkeypatch.setattr(CT, "_contact_sidecars", lambda tree, ti: sidecars)


TI = {"hand": "h", "dataset_name": "d"}


def test_each_hand_keeps_its_fingers_and_the_pad_reads_frame_0(monkeypatch):
    r = np.zeros((4, 5), bool)
    r[0, 0] = r[3, 1] = True
    l_ = np.zeros((4, 5), bool)
    l_[2, 4] = True
    _stub(monkeypatch, {"right": {"fingers": r}, "left": {"fingers": l_}}, ("right", "left"), pad=2)
    c = CT.finger_contacts("tree", TI, 6)
    assert c.shape == (6, 10)
    np.testing.assert_array_equal(c[:, :5], r[[0, 0, 0, 1, 2, 3]])
    np.testing.assert_array_equal(c[:, 5:], l_[[0, 0, 0, 1, 2, 3]])


def test_a_side_the_base_does_not_expect_holds_nothing(monkeypatch):
    r = np.ones((3, 5), bool)
    _stub(monkeypatch, {"right": {"fingers": r}}, ("right",), pad=0)
    c = CT.finger_contacts("tree", TI, 3)
    assert c[:, :5].all() and not c[:, 5:].any()


def test_an_expected_side_without_a_sidecar_dies(monkeypatch):
    _stub(monkeypatch, {"right": {"fingers": np.ones((3, 5), bool)}}, ("right", "left"), pad=0)
    with pytest.raises(ValueError, match="expects a contact sidecar"):
        CT.finger_contacts("tree", TI, 3)


def test_a_sidecar_of_another_length_dies(monkeypatch):
    _stub(monkeypatch, {"right": {"fingers": np.ones((5, 5), bool)}}, ("right",), pad=1)
    with pytest.raises(ValueError, match="tree length"):
        CT.finger_contacts("tree", TI, 5)
