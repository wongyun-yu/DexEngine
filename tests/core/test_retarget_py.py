"""The retargeting interpreter belongs to the code tree: RETARGET_PY first, then <code tree>/.venv; the data root is
never a candidate, and a miss dies naming the places tried."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from dexengine.core.errors import InputMissingError
from dexengine.core.locations import roots

REPO = Path(__file__).resolve().parents[2]


def _exe(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("#!/bin/sh\n")
    p.chmod(0o755)
    return p


@pytest.fixture
def code_tree(tmp_path, monkeypatch):
    """A code tree apart from the data root (the conftest data root is another folder)."""
    tree = tmp_path / "code_tree"
    tree.mkdir()
    monkeypatch.setattr(roots, "WT", tree)
    monkeypatch.delenv(roots.RETARGET_PY_ENV, raising=False)
    return tree


def test_code_tree_venv_not_data_root(code_tree, data_root):
    _exe(data_root / ".venv/bin/python")  # a venv under the data root is never used
    want = _exe(code_tree / ".venv/bin/python")
    assert roots.retarget_py_candidates() == [str(want)]
    assert roots.require_retarget_py() == str(want)


def test_data_root_venv_alone_is_a_miss(code_tree, data_root):
    _exe(data_root / ".venv/bin/python")
    with pytest.raises(InputMissingError) as e:
        roots.require_retarget_py()
    msg = str(e.value)
    assert str(code_tree / ".venv/bin/python") in msg and "RETARGET_PY" in msg
    assert str(data_root / ".venv/bin/python") not in msg


def test_explicit_env_wins(code_tree, tmp_path, monkeypatch):
    _exe(code_tree / ".venv/bin/python")
    other = _exe(tmp_path / "elsewhere/python")
    monkeypatch.setenv(roots.RETARGET_PY_ENV, str(other))
    assert roots.require_retarget_py() == str(other)


def test_explicit_env_missing_dies_without_fallback(code_tree, tmp_path, monkeypatch):
    _exe(code_tree / ".venv/bin/python")  # present, but an explicit value is not replaced by it
    gone = tmp_path / "gone/python"
    monkeypatch.setenv(roots.RETARGET_PY_ENV, str(gone))
    with pytest.raises(InputMissingError) as e:
        roots.require_retarget_py()
    assert f"RETARGET_PY={gone}" in str(e.value)


def test_cli_prints_the_code_tree_venv(data_root):
    """The shell window (`paths --what retarget-py`) answers the same rule: this code tree's venv, not the data
    root's."""
    env = {k: v for k, v in os.environ.items() if k != roots.RETARGET_PY_ENV}
    env.update(PYTHONPATH=str(REPO), DEXENGINE_MAIN_ROOT=str(data_root))
    r = subprocess.run(
        [sys.executable, "-m", "dexengine.core.paths", "--what", "retarget-py"],
        cwd=data_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(REPO / ".venv/bin/python")
