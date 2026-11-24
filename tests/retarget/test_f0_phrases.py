"""The frame-0 repair worker's outcome lines (repair.py + f0.phrases) -> shadow_tree.finish's classification."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from dexengine.retarget.f0 import phrases as PH
from dexengine.retarget.f0 import shadow_tree as F
from tests.imports import own_module

BUDGET_OK = "[F0_BUDGET] side=right evals=4/10 status=OK"


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "applied_scene", lambda: "test")
    a = types.SimpleNamespace(hand="sharpa_wave", dataset="taco", emb="right", src_layer="base")

    def _finish(out: str, rc: int = 0):
        d = tmp_path / "shadow"
        d.mkdir(exist_ok=True)
        job = {"clip": "c0", "t0": 0.0, "d": d, "key_now": "k", "stale": "", "sides": ["right"]}
        res = F.finish(job, rc, out, a, Path("/stage"), {"right": Path("/w/right")})
        mark = json.loads((d / F.MARKER).read_text()) if (d / F.MARKER).is_file() else None
        return res, mark

    return _finish


def _repair(monkeypatch):
    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")  # repair_cfg resolves the active hand and arm at import
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    return own_module("dexengine.retarget.f0.repair_cfg")


def test_budget_skip_status_is_repair_fail(run, monkeypatch):
    R = _repair(monkeypatch)
    res, mark = run("\n".join(["[F0_BUDGET] side=right evals=10/10 status=EXHAUSTED", R.budget_skip_status(10, 10)]))
    assert res["status"] == "REPAIR_FAIL" and mark["status"] == "REPAIR_FAIL"


def test_shallow_skip_is_shallow(run):
    res, _ = run("\n".join([BUDGET_OK, f"{PH.SKIP}: max depth 4.0mm ≤ 10mm (shallow penetration — repair skipped)"]))
    assert res["status"] == "SHALLOW"


@pytest.mark.parametrize(
    "line", [PH.inplace_failed_line(PH.pad_too_big(80)), PH.pad_too_big(61), PH.inplace_failed_line("disk full")]
)
def test_refused_repair_is_refused_big(run, line):
    res, mark = run("\n".join([BUDGET_OK, line]))
    assert res["status"] == "REFUSED_BIG" and mark is not None


def test_unrepaired_overlap_is_repair_fail_with_its_reason(run, monkeypatch):
    R = _repair(monkeypatch)
    st = R._no_solution_status({}, 10, 3)
    res, mark = run("\n".join([BUDGET_OK, PH.unrepaired_line(1), f"    c0 [right] {st}"]))
    assert res["status"] == "REPAIR_FAIL" and st in mark["f0_status"]
    res, _ = run(
        "\n".join([BUDGET_OK, PH.unrepaired_line(1), f"    c0 [right] {R._no_solution_status({'verify': 'v'}, 10, 3)}"])
    )
    assert res["status"] == "REPAIR_FAIL"


def test_already_repaired_without_budget_lines_is_a_race(run):
    res, mark = run(PH.already_repaired_line("c0"))
    assert res["status"] == "FAIL_NO_BUDGET_MARK" and "[F0_RACE]" in res["log"] and mark is None


def test_no_token_is_unexpected(run):
    res, _ = run("\n".join([BUDGET_OK, "nothing happened"]))
    assert res["status"] == "FAIL_NOLEDGER"


def test_only_the_skip_statuses_contain_skip():
    """finish checks `SKIP in out`: a new token or worker line with SKIP would turn a refused clip into SHALLOW."""
    lines = [
        PH.OVERLAP_LEFT,
        PH.PAD_TOO_BIG,
        PH.INPLACE_FAILED,
        PH.UNREPAIRED,
        PH.ALREADY_REPAIRED,
        PH.pad_too_big(70),
        PH.already_repaired_line("s"),
        PH.inplace_failed_line("e"),
        PH.unrepaired_line(2),
    ]
    assert not [x for x in lines if PH.SKIP in x]
