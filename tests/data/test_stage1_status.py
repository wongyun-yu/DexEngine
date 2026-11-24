"""Stage-1 progress table (`stages.py --status`): every input-fingerprint cell names a module that owns `freshness`,
and the table prints on an empty data root. Before the fix the step-2 cell still named `dexengine.data.base_adapter`
(the adapter CLI after the rules moved to `dexengine.data.base_layer`), so `--status` died with INPUT_MISSING
on its first dataset."""

from __future__ import annotations

import importlib

import pytest

from dexengine.data.stage1 import status as ST
from dexengine.data.stage1 import table as T


def _input_fp_owners() -> list[str]:
    return [cell[len(T._INPUT_FP) :] for _name, cell, *_ in T.STAGES if cell.startswith(T._INPUT_FP)]


def test_every_input_fingerprint_owner_has_freshness():
    owners = _input_fp_owners()
    assert owners, "no input-fingerprint cell in STAGES"
    missing = [o for o in owners if not callable(getattr(importlib.import_module(o), "freshness", None))]
    assert not missing, f"input-fingerprint cells name modules without freshness: {missing}"


@pytest.mark.parametrize("ds", ["oakink2"])
def test_status_prints_on_an_empty_root(ds, monkeypatch, capsys):
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    # rules.ACTIVE_HAND resolves from the env at each read (PEP 562); patching the attribute instead leaves a fixed
    # value behind (monkeypatch restores the value it read), and a later test reads that hand
    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")
    assert ST.status([ds]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("hand sharpa_wave")
    assert ST.STATUS_ROWS[ds]["2 base layer"].endswith("0/1")  # empty base layer = empty step, not a crash
