"""Record functions on synthetic inputs match their recorded results (golden/cases.json; its stored description texts
carry the English values of the legacy-value table)."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from tests.record import golden_cases as G

HERE = Path(__file__).resolve().parent


def _norm(x):
    return json.loads(json.dumps(x, ensure_ascii=False, sort_keys=True))


GOLDEN = json.loads((HERE / "golden" / "cases.json").read_text())
GOT = G.cases(lambda k: importlib.import_module(G.MODULES[k]))


@pytest.mark.parametrize("case", sorted(GOLDEN))
def test_case_matches_reference(case):
    assert _norm(GOT[case]) == _norm(GOLDEN[case])


def test_same_cases():
    assert set(GOT) == set(GOLDEN)
