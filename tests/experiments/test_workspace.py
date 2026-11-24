"""The experiment workspace owner: one required input, no default, paths under it."""

from pathlib import Path

import pytest

from experiments import workspace


def test_root_requires_the_variable(monkeypatch):
    monkeypatch.delenv(workspace.ENV, raising=False)
    with pytest.raises(SystemExit, match=workspace.ENV):
        workspace.root()


def test_layout_under_root(monkeypatch, tmp_path):
    monkeypatch.setenv(workspace.ENV, str(tmp_path))
    assert workspace.root() == tmp_path
    assert workspace.code("spider") == tmp_path / "code" / "spider"
    assert workspace.data("chord", "x.urdf") == tmp_path / "data" / "chord" / "x.urdf"
    assert workspace.results("ours_rl_abl") == tmp_path / "results" / "ours_rl_abl"
    assert workspace.logs() == tmp_path / "logs"


def test_repo_is_this_checkout():
    assert (workspace.REPO / "experiments" / "workspace.py").resolve() == Path(workspace.__file__).resolve()
