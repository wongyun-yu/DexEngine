"""record.driver names its robot with --arm (required, no default): a call without it dies with the reason before any
queue is read, and with it the record combo is <arm>_<hand>. A bundle records the arm its cells were trained on (each
cell's judge env DEXENGINE_ARM — bundle_cell.cells_arm); a cell without it or cells on two arms die."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from dexengine.core import paths as P
from dexengine.record import driver_launch as DL


def _args(tmp: Path, arm):
    return types.SimpleNamespace(
        tree=P.main_root(),
        arm=arm,
        hand="xhand_pro",
        gpus="0",
        exclusive_gpus="",
        only="",
        spec_mode=False,
        queue_root="",
        round="aug",
        bundle_timeout_per_clip=None,
        bundle_timeout_base=60,
        merge="off",
    )


def test_driver_without_arm_dies(tmp_path, monkeypatch):
    monkeypatch.setenv("DEXENGINE_RL_QUEUE_ROOT", str(tmp_path))
    monkeypatch.setenv("DEXENGINE_MAIN_ROOT", str(P.main_root()))
    with pytest.raises(SystemExit, match=r"--arm is required \(no default\)"):
        DL.check_env(_args(tmp_path, None))


def test_driver_arm_names_the_combo(tmp_path, monkeypatch):
    monkeypatch.setenv("DEXENGINE_RL_QUEUE_ROOT", str(tmp_path))
    monkeypatch.setenv("DEXENGINE_MAIN_ROOT", str(P.main_root()))
    a = _args(tmp_path, "rby1")
    monkeypatch.setattr(DL.PC, "round_gate", lambda a: None)  # the combo is set before the round gate
    DL.check_env(a)
    assert a.combo == "rby1_xhand_pro"


def test_parser_has_no_arm_default():
    import sys

    from dexengine.record import driver as DRV

    argv = sys.argv
    try:
        sys.argv = ["driver", "--merge", "off", "--hours", "1"]
        assert DRV.parse_args().arm is None
    finally:
        sys.argv = argv


def _cell(env):
    return {"campaign": "c", "bundle": "b", "env": env}


def test_bundle_arm_is_the_cells_arm():
    from dexengine.record.bundle_cell import cells_arm

    assert cells_arm([_cell(["SEQ_DIR=x", "DEXENGINE_ARM=rby1"]), _cell(["DEXENGINE_ARM=rby1"])]) == "rby1"
    for cells, match in (
        ([_cell(["SEQ_DIR=x"])], r"judge env DEXENGINE_ARM \[\]"),
        ([_cell(["DEXENGINE_ARM=rby1"]), _cell(["DEXENGINE_ARM=other"])], r"arms \['other', 'rby1'\]"),
        ([], r"arms \[\]"),
    ):
        with pytest.raises(SystemExit, match=match):
            cells_arm(cells)
