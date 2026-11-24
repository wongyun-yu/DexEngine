"""The reader-call scope: a rule whose mechanism is one reader function (D12-g, the object GT gap fill) applies to every
dataset whose reader calls that function, so a new reader that fills object gaps is in D12-g's range without a name
being written into the rule."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

from dexengine.core import datasets as RR
from dexengine.core.rules import reader_calls as RC
from dexengine.data import readers as R

REPO = Path(__file__).resolve().parents[2]
OWNER = "dexengine/data/readers/hot3d.py:_fill_gt_gaps"


def _rule(rid: str):
    (r,) = [r for r in RR.RULES if r.id == rid]
    return r


def _reader_scoped():
    return [r for r in RR.RULES if r.scope_from == RR.READER_CALL_SCOPE]


def test_d12g_range_is_the_readers_that_call_the_gap_fill():
    r = _rule("D12-g")
    assert r.scope_from == RR.READER_CALL_SCOPE and r.owner == OWNER
    assert RR.rule_datasets("D12-g") == RC.reader_scope(OWNER) == ("hot3d",)


def test_a_new_reader_that_calls_the_gap_fill_is_in_d12g(tmp_path, monkeypatch):
    """A reader written like an outside dataset's (it calls hot3d's gap fill through the module) joins the range."""
    (tmp_path / "gapfill_reader_probe.py").write_text(
        "import numpy as np\n"
        "from dexengine.data.readers import hot3d\n\n\n"
        "def fill(samples, t_s):\n"
        "    return hot3d._fill_gt_gaps(samples, np.round(t_s * 1e9))\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    assert "dexcanvas" not in RR.rule_datasets("D12-g")
    monkeypatch.setitem(R.READERS, "dexcanvas", R.ReaderSpec(module="gapfill_reader_probe", driver="stage"))
    assert RR.rule_datasets("D12-g") == ("dexcanvas", "hot3d")
    assert "gapfill_reader_probe" not in sys.modules  # read as source, never imported
    assert RR.covers(dataclasses.replace(_rule("D12-g"), datasets=()), "dexcanvas")


def _modules():
    """(module name, source) of every dexengine module; a package __init__ keeps its `.__init__` suffix so relative
    imports resolve against the package."""
    for p in sorted((REPO / "dexengine").rglob("*.py")):
        yield ".".join(p.relative_to(REPO).with_suffix("").parts), p.read_text(encoding="utf-8")


def test_every_module_calling_an_owner_is_a_reader_in_the_rule_range():
    """A helper that calls the owner for a reader would leave that reader's dataset out of the range unseen; it fails
    here instead (call the owner from the reader module itself)."""
    rules = _reader_scoped()
    assert rules, "no rule uses the reader-call scope"
    row_of = {row.module: ds for ds, row in R.READERS.items()}
    bad = []
    for r in rules:
        fn = RC.owner_function(r.owner)[1]
        scope = RR.rule_datasets(r.id)
        for mod, src in _modules():
            if fn in src and RC.calls_owner(src, mod, r.owner) and row_of.get(mod) not in scope:
                bad.append(f"{r.id}: {mod} calls {r.owner} but is not the reader of a dataset in {scope}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize(
    ("module", "src", "want"),
    [
        ("dexengine.data.readers.hot3d", "def _fill_gt_gaps(a, t):\n    pass\n\n\nx = _fill_gt_gaps(1, 2)\n", True),
        ("x.reader", "from dexengine.data.readers import hot3d\nhot3d._fill_gt_gaps(a, t)\n", True),
        ("dexengine.data.readers.cap", "from . import hot3d as H\nH._fill_gt_gaps(a, t)\n", True),
        ("x.reader", "import dexengine.data.readers.hot3d as H\nH._fill_gt_gaps(a, t)\n", True),
        ("x.reader", "from dexengine.data.readers.hot3d import _fill_gt_gaps as fill\nfill(a, t)\n", True),
        ("x.reader", '"""uses hot3d._fill_gt_gaps"""\nfrom dexengine.data.readers import hot3d\n', False),
        ("x.reader", "from dexengine.data.readers import hot3d\nf = hot3d._fill_gt_gaps\n", False),
        ("x.reader", "from dexengine.data.readers import grab as hot3d\nhot3d._fill_gt_gaps(a, t)\n", False),
        ("x.reader", "def _fill_gt_gaps(a, t):\n    pass\n\n\n_fill_gt_gaps(1, 2)\n", False),
    ],
)
def test_calls_owner_forms(module, src, want):
    assert RC.calls_owner(src, module, OWNER) is want


def test_an_owner_no_reader_calls_raises():
    r = dataclasses.replace(_rule("D12-g"), owner="dexengine/data/readers/hot3d.py:no_such_function", datasets=())
    with pytest.raises(ValueError, match="no reader calls it"):
        RR.scope_of(r)
    with pytest.raises(ValueError, match="<file>.py:<function>"):
        RC.owner_function("dexengine/data/readers/hot3d.py")
