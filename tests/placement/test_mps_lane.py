"""The judge proves its MPS lane contract at boot (dexengine/placement/judge/mps_lane.py)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from dexengine.placement.judge import boot as JB
from dexengine.placement.judge import mps_lane as MPS


def test_boot_proves_the_lane_before_any_other_step():
    """A lane launcher starts judges as MPS lanes by default; the in-process proof and the lane
    VRAM cap only exist if BootMixin.__init__ calls assert_lane first."""
    tree = ast.parse(Path(JB.__file__).read_text(encoding="utf-8"))
    init = next(
        f
        for c in ast.walk(tree)
        if isinstance(c, ast.ClassDef) and c.name == "BootMixin"
        for f in c.body
        if isinstance(f, ast.FunctionDef) and f.name == "__init__"
    )
    calls = [ast.unparse(s.value.func) for s in init.body if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)]
    assert calls[0] == "_MPS.assert_lane"


def test_off_lane_prints_markers_and_sets_no_cap(monkeypatch, capsys):
    for k in (
        "DEXENGINE_JUDGE_MPS",
        "CUDA_MPS_PIPE_DIRECTORY",
        "DEXENGINE_JUDGE_LANE_MIB",
        "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT",
    ):
        monkeypatch.delenv(k, raising=False)
    assert MPS.assert_lane(None, "cpu") is None
    out = capsys.readouterr().out
    assert "[JUDGE_MPS] off(unset)" in out and "[VRAM_CAP] lane off" in out


def test_mps_outside_the_contract_dies(monkeypatch):
    monkeypatch.delenv("DEXENGINE_JUDGE_MPS", raising=False)
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/tmp/mps")
    with pytest.raises(SystemExit, match="outside the contract"):
        MPS.assert_lane(None, "cuda")
