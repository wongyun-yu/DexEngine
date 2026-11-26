"""render_only_build.base_trees over hand base layers (module doc 'Membership'): the axis hand's tree wins; a missing
axis-hand base dies unless the caller states the root's hands (--root-hands, e.g. a root with one hand); the statement
is checked against the measured layers; windows that differ across hands die."""

from __future__ import annotations

import json

import pytest

from dexengine.core import paths as P
from dexengine.record.scene import render_only_build as ROB

AX = "axis_h"


def _tree(root, hand, emb, clip, window=(0, 10)):
    d = root / hand / emb / clip
    d.mkdir(parents=True)
    (d / "task_info.json").write_text(json.dumps({"segment_range": list(window)}))
    return d


@pytest.fixture
def layers(tmp_path, monkeypatch):
    monkeypatch.setattr(ROB, "axis_hand", lambda: AX)
    monkeypatch.setattr(P, "HAND_IDS", (AX, "b_hand", "c_hand"))
    monkeypatch.setattr(P, "hand_spec_dir", lambda h: h)
    monkeypatch.setattr(P, "layer_path", lambda layer, ds, hand, emb: tmp_path / hand / emb)
    return tmp_path


def test_the_axis_hand_tree_wins(layers):
    _tree(layers, "b_hand", "right", "clipA")
    want = _tree(layers, AX, "right", "clipA")
    assert ROB.base_trees("ds") == {"clipA": want}


def test_a_missing_axis_hand_dies_unless_stated(layers):
    _tree(layers, "b_hand", "right", "clipA")
    with pytest.raises(ROB.RenderOnlyError, match="axis hand axis_h has no base tree"):
        ROB.base_trees("ds")


def test_a_stated_root_without_the_axis_hand_takes_the_first_hand_by_name(layers, capsys):
    b = _tree(layers, "b_hand", "right", "clipA")
    _tree(layers, "c_hand", "right", "clipA")
    c = _tree(layers, "c_hand", "left", "clipB")
    assert ROB.base_trees("ds", ("b_hand", "c_hand")) == {"clipA": b, "clipB": c}
    assert "(stated) · axis hand axis_h not among them" in capsys.readouterr().out


def test_a_statement_that_misses_a_hand_with_trees_dies(layers):
    _tree(layers, "b_hand", "right", "clipA")
    _tree(layers, "c_hand", "right", "clipB")
    with pytest.raises(ROB.RenderOnlyError, match="hands with a base tree here are"):
        ROB.base_trees("ds", ("b_hand",))


def test_a_stated_axis_hand_without_trees_dies(layers):
    _tree(layers, "b_hand", "right", "clipA")
    with pytest.raises(ROB.RenderOnlyError, match="stated in --root-hands but has no base tree"):
        ROB.base_trees("ds", (AX, "b_hand"))


def test_an_unknown_stated_hand_dies(layers):
    with pytest.raises(ROB.RenderOnlyError, match="are not hands"):
        ROB.base_trees("ds", ("nope",))


def test_a_stated_root_checks_every_window(layers):
    _tree(layers, "b_hand", "right", "clipA", (0, 10))
    _tree(layers, "c_hand", "right", "clipA", (0, 12))
    with pytest.raises(ROB.RenderOnlyError, match="window of a clip missing for the axis hand differs"):
        ROB.base_trees("ds", ("b_hand", "c_hand"))
