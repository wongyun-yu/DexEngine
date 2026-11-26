"""The record entry builds a missing render-only object index (render_only_entry.ensure_index, its module doc): from
the hands in the root, for the whole dataset, once, under a lock; an index already there is kept; a root the rule
cannot state dies; a clip that cannot be measured writes nothing. Both record entries call it before their work, and
the reader's error names the record entry."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from dexengine.core import paths as P
from dexengine.record import bundle_cell as BC
from dexengine.record.scene import render_only as RO
from dexengine.record.scene import render_only_build as ROB
from dexengine.record.scene import render_only_entry as ROE

AX = "axis_h"
REPO = Path(__file__).resolve().parents[2]


def _tree(root, hand, emb, clip, window=(0, 10)):
    d = root / "base" / hand / emb / clip
    d.mkdir(parents=True)
    (d / "task_info.json").write_text(json.dumps({"segment_range": list(window)}))
    return d


class _Seq:
    """Stand-in for the dataset's measuring module (oracle_module): one sequence per clip."""

    @staticmethod
    def seq_key(clip):
        return clip


@pytest.fixture
def root(tmp_path, monkeypatch):
    def layer_path(layer, ds, hand=None, emb=None):
        if layer == RO.LAYER:
            return tmp_path / "render_only" / ds
        return tmp_path / layer / hand / emb

    monkeypatch.setattr(ROB, "axis_hand", lambda: AX)
    monkeypatch.setattr(P, "HAND_IDS", (AX, "b_hand", "c_hand"))
    monkeypatch.setattr(P, "hand_spec_dir", lambda h: h)
    monkeypatch.setattr(P, "layer_path", layer_path)
    monkeypatch.setattr(ROB, "oracle_module", lambda ds: _Seq)
    calls: list = []

    def gen_seq(ds, key, items):
        calls.append(key)
        return [{"clip": c, "objects": [], "base_tree": str(t), "version": RO.VERSION, "dataset": ds} for c, t in items]

    monkeypatch.setattr(ROB, "gen_seq", gen_seq)
    gen_seq.calls = calls  # the sequences the generator was asked for
    return tmp_path


def _index(root, ds="ds"):
    return json.loads((root / "render_only" / ds / RO.INDEX).read_text())


def test_a_missing_index_is_built_from_the_one_hand_of_the_root(root, capsys):
    _tree(root, "b_hand", "right", "clipA")
    _tree(root, "b_hand", "left", "clipB")
    assert ROE.ensure_index("ds") is True
    assert _index(root)["clips"] == {"clipA": 0, "clipB": 0}
    out = capsys.readouterr().out
    assert "index missing" in out and "root hands b_hand, derived from the base layers" in out
    assert "(stated) · axis hand axis_h not among them" in out  # base_trees' own line: the statement reached it
    assert "[RENDER_ONLY] end ds clips 2" in out


def test_a_root_with_the_axis_hand_takes_the_standard_rule(root, capsys):
    want = _tree(root, AX, "right", "clipA")
    _tree(root, "b_hand", "right", "clipA")
    _tree(root, "b_hand", "right", "clipB")
    assert ROE.ensure_index("ds") is True
    assert _index(root)["clips"] == {"clipA": 0, "clipB": 0}
    assert "axis hand axis_h (standard rule)" in capsys.readouterr().out
    assert ROB.base_trees("ds")["clipA"] == want


def test_an_index_already_there_is_kept(root):
    d = root / "render_only" / "ds"
    d.mkdir(parents=True)
    (d / RO.INDEX).write_text("{}")
    _tree(root, "b_hand", "right", "clipA")
    assert ROE.ensure_index("ds") is False
    assert (d / RO.INDEX).read_text() == "{}" and ROB.gen_seq.calls == []


def test_an_index_written_while_waiting_for_the_lock_is_kept(root, monkeypatch):
    import fcntl

    _tree(root, "b_hand", "right", "clipA")
    ip = root / "render_only" / "ds" / RO.INDEX

    def flock(fd, op):  # the other record entry finishes while this one waits
        ip.write_text('{"by": "other"}')

    monkeypatch.setattr(fcntl, "flock", flock)
    assert ROE.ensure_index("ds") is False
    assert json.loads(ip.read_text()) == {"by": "other"} and ROB.gen_seq.calls == []


@pytest.mark.parametrize(
    "hands",
    [(), ("b_hand", "c_hand")],
    ids=["no hand", "two hands without the axis hand"],
)
def test_a_root_the_rule_cannot_state_dies_and_writes_nothing(root, hands):
    for h in hands:
        _tree(root, h, "right", "clipA")
    with pytest.raises(ROB.RenderOnlyError, match="the hands with a base tree here are"):
        ROE.ensure_index("ds")
    assert not (root / "render_only" / "ds").exists() and ROB.gen_seq.calls == []  # no empty layer folder left


def test_a_broken_multi_hand_root_still_dies_in_base_trees(root):
    _tree(root, AX, "right", "clipA")
    _tree(root, "b_hand", "right", "clipB", (0, 10))
    _tree(root, "c_hand", "right", "clipB", (0, 12))
    with pytest.raises(ROB.RenderOnlyError, match="window of a clip missing for the axis hand differs"):
        ROE.ensure_index("ds")
    assert not (root / "render_only" / "ds" / RO.INDEX).exists()


def test_a_clip_that_cannot_be_measured_writes_nothing(root, monkeypatch):
    _tree(root, "b_hand", "right", "clipA")
    _tree(root, "b_hand", "right", "clipB")

    def gen_seq(ds, key, items):
        return [{"clip": c, "error": "NO_RAW:boom"} if c == "clipB" else {"clip": c, "objects": []} for c, _ in items]

    monkeypatch.setattr(ROB, "gen_seq", gen_seq)
    with pytest.raises(ROB.RenderOnlyError, match="1 of 2 clips could not be measured"):
        ROE.ensure_index("ds")
    assert not (root / "render_only" / "ds" / RO.INDEX).exists()


def test_the_cli_without_a_statement_still_dies_on_a_root_without_the_axis_hand(root):
    _tree(root, "b_hand", "right", "clipA")
    with pytest.raises(ROB.RenderOnlyError, match="axis hand axis_h has no base tree"):
        ROB.generate("ds")


def test_the_reader_names_the_record_entry(root):
    with pytest.raises(RO.RenderOnlyError, match="the record entry .*stage4_record.py bundle or drive. builds it"):
        RO.read_clip("ds", "clipA")


def test_a_bundle_cell_states_one_dataset():
    cells = [
        {"campaign": "c", "bundle": "b1", "env": ["DEXENGINE_DATASET=hot3d", "X=1"]},
        {"campaign": "c", "bundle": "b2", "env": ["DEXENGINE_DATASET=arctic"]},
    ]
    assert BC.cells_datasets(cells) == ["arctic", "hot3d"]
    with pytest.raises(SystemExit, match="DEXENGINE_DATASET"):
        BC.cells_datasets([{"campaign": "c", "bundle": "b3", "env": ["X=1"]}])


def _calls_in(path: str, func: str) -> list[str]:
    tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func)
    calls = sorted((n.lineno, n.col_offset, n) for n in ast.walk(fn) if isinstance(n, ast.Call))  # source order
    return [n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "") for *_, n in calls]


def test_both_record_entries_build_the_index_before_their_work():
    """Structural: the driver builds before its claim and workers; the bundle before stage 1 of a run that renders."""
    drv = _calls_in("dexengine/record/driver.py", "main")
    assert "ensure_render_only" in drv and drv.index("ensure_render_only") < drv.index("claim")
    assert "ensure_index" in _calls_in("dexengine/record/driver.py", "ensure_render_only")
    bun = _calls_in("dexengine/record/bundle.py", "main")
    assert "ensure_render_only" in bun and bun.index("ensure_render_only") < bun.index("collect_cell")
    assert "ensure_index" in _calls_in("dexengine/record/bundle.py", "ensure_render_only")
