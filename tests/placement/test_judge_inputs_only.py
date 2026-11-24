"""The placement judge is a function of its inputs: it never reads an earlier adoption record
(precompute/<ds>/robots/<combo>/placement/<tree>.adopted.json — the bake's input, written after a judgement). Before,
the judge re-expressed each adoption about the record's pivot and, when an anchor failed, adopted the record's
placement shifted by the v2 base move ("transplant") — so a judgement depended on what an earlier run left on disk.

The augmentation-cell stage (judge/aug_cells.py) is the one reader: the canonical placement is its declared input.
"""

from __future__ import annotations

import ast
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

from dexengine.placement.judge import anchor  # noqa: E402

JUDGE = Path(anchor.__file__).resolve().parent
RECORD_READER = "aug_cells.py"  # the augmentation stage: the canonical record is its declared input
RECORD_NAMES = {"placement_dir", "PLACEMENT_SFX"}


def _tree(name: str) -> ast.Module:
    return ast.parse((JUDGE / name).read_text(encoding="utf-8"))


def _names(node) -> set:
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, ast.alias):
            out.add(n.asname or n.name.rsplit(".", 1)[-1])
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and ".adopted.json" in n.value:
            out.add(".adopted.json")
    return out


def test_no_judge_module_but_the_aug_stage_names_the_record_folder():
    """Structure: only the augmentation stage resolves the adoption-record folder or its file suffix."""
    users = sorted(p.name for p in JUDGE.glob("*.py") if _names(_tree(p.name)) & (RECORD_NAMES | {".adopted.json"}))
    assert users == [RECORD_READER]


def test_the_judge_has_no_record_reader_or_fallback():
    from dexengine.placement.judge.core import Judge

    gone = ("_adopted_path", "_adopted_pivot", "_anchor_fallback", "_boot_v2_port", "PL_DIR", "TRANS_CAP")
    assert [g for g in gone if hasattr(Judge, g)] == []


def _swap_calls(tree: ast.Module, fn: str) -> list:
    f = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn)
    return [
        c
        for c in ast.walk(f)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "_pivot_swap"
    ]


@pytest.mark.parametrize(
    ("module", "fn"), [("anchor.py", "_anchor_adopt"), ("exhaust.py", "_adopt_ex"), ("exhaust.py", "_serial_adopt")]
)
def test_every_canonical_adoption_swaps_about_the_clip_pivot(module, fn):
    """Structure: each canonical adoption site calls the one pivot-swap function with p2 = the clip pivot (cpiv)."""
    calls = _swap_calls(_tree(module), fn)
    assert len(calls) == 1
    assert [ast.unparse(a) for a in calls[0].args[:2]] == ["cpiv", "cpiv"]


def _swap(p2, cpiv, dcap, yaw):
    J = types.SimpleNamespace()
    return anchor.AnchorMixin._pivot_swap(J, p2, cpiv, dcap, yaw)


def test_pivot_swap_about_the_clip_pivot_is_the_identity():
    cpiv = np.array([0.21, -0.13, 0.0])
    dcap = np.array([-0.4, 0.07, 0.0])
    d, p2 = _swap(cpiv, cpiv, dcap, 37.5)
    assert np.array_equal(d, dcap[:2]) and np.array_equal(p2, cpiv[:2])


def test_pivot_swap_keeps_the_placement_about_another_pivot():
    """About another pivot (the augmentation cells' canonical pivot) the numbers change, the rigid placement does not:
    R(x − p2) + p2 + d == R(x − cpiv) + cpiv + dcap for every point x."""
    cpiv, p2 = np.array([0.21, -0.13, 0.0]), np.array([0.31, -0.07])
    dcap, yaw = np.array([-0.4, 0.07, 0.0]), 37.5
    d, q2 = _swap(p2, cpiv, dcap, yaw)
    th = np.deg2rad(yaw)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    x = np.random.default_rng(0).normal(0, 0.3, (16, 2))
    a = (x - q2) @ R.T + q2 + d
    b = (x - cpiv[:2]) @ R.T + cpiv[:2] + dcap[:2]
    assert np.array_equal(q2, p2) and np.allclose(a, b, atol=1e-12)


def test_an_anchor_without_a_passing_cell_writes_only_its_reject_row():
    import io

    J = types.SimpleNamespace(REJ={"desk": 3, "reach": 5}, fno=io.StringIO())
    anchor.AnchorMixin._anchor_reject(J, 2, "clip_a", 40)
    assert J.fno.getvalue() == "clip_a\tanchor2\tconfirm_failed(checks 40) reach=5 desk=3\n"


def test_the_boot_banner_states_that_records_are_not_read(capsys):
    """Apply marker: every judge run says it does not read adoption records ([JUDGE_RECORDS])."""
    from dexengine.placement.judge import boot

    J = types.SimpleNamespace(
        combo="rby1_x",
        _JSEED=1,
        sides={"right": {"wd": Path("/w/wrist_poses_right/x")}},
        TRAJ_ROOT="t",
        a=types.SimpleNamespace(wrist_root="auto", traj_root="auto"),
    )
    boot.BootMixin._boot_banner(J)
    assert "[JUDGE_RECORDS] rby1_x: earlier adoption records (the placement layer) are not read" in (
        capsys.readouterr().out
    )
