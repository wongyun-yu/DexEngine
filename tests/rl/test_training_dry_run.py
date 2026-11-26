"""The training mode starts nothing in a dry run of a run already trained (no trainer, no render, no results row),
draws no video at all (training runs without the renderer; videos are a render launch), and refuses DO_VIDEO=1."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from dexengine.core.repo import REPO_ROOT
from dexengine.rl import launch
from dexengine.rl.launch import training
from dexengine.rl.launch.recipe import RecipeError
from tests.rl.test_launch import CASES, parse_case

TRAINING = REPO_ROOT / "dexengine/rl/launch/training.py"
RENDER_NAMES = ("render_call", "render_calls", "render_out", "scene_env")


def _trained(monkeypatch, tmp_path: Path, path: str):
    """A finished run (bimanual: final checkpoint · one hand: TRAIN_DONE) and recorders of every start and row."""
    ck = tmp_path / "ck" / "agent_51200.pt"
    ck.parent.mkdir()
    ck.touch()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "TRAIN_DONE").write_text("RUN=R1\n")
    started, rows = [], []
    monkeypatch.setattr(training.PL, "final_ckpt", lambda _p: ck)
    monkeypatch.setattr(training.records, "run_dir", lambda _run, _tree: run_dir)
    monkeypatch.setattr(training.process, "start", lambda *a, **k: started.append(a) or 0)
    monkeypatch.setattr(training.process, "resolve_call", lambda *a, **k: None)
    monkeypatch.setattr(training.records, "check_coll_preset", lambda *a, **k: None)
    monkeypatch.setattr(training, "_row", lambda _p, status, _s, tail="": rows.append(status))
    req = SimpleNamespace(run="R1", root=tmp_path, run_tree=tmp_path, dry_run=True, gpu="0", path=path, side="right")
    plan = SimpleNamespace(req=req, values={"NE": "4096", "MI": "1600", "MB": "16"}, preset="p", env={})
    return plan, started, rows


@pytest.mark.parametrize("path", ["bi", "uni"])
def test_a_dry_run_of_a_trained_run_starts_and_writes_nothing(monkeypatch, tmp_path, capsys, path):
    plan, started, rows = _trained(monkeypatch, tmp_path, path)
    assert training.train(plan, tmp_path / "ref", 0) == 0
    assert started == [] and rows == []
    assert "already trained" in capsys.readouterr().out


def _render_calls(src: str) -> list[str]:
    return [
        ast.unparse(n.func)
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call) and ast.unparse(n.func).split(".")[-1] in RENDER_NAMES
    ]


def test_the_training_mode_draws_no_video():
    src = TRAINING.read_text(encoding="utf-8")
    assert _render_calls(src) == []
    broken = src.replace(
        "    return 0\n\n\ndef _dry_trained", "    modes.render_call(plan, ck)\n    return 0\n\n\ndef _dry_trained"
    )
    assert _render_calls(broken) == ["modes.render_call"]  # negative: a render launch in the training mode is seen


def test_do_video_is_a_removed_input():
    env, _ = parse_case(next(c for c in CASES if c.stem == "train_bi_sharpa_wave"))
    assert launch.from_env("R", env | {"DO_VIDEO": "0"}).mode == "train"  # the callers' old 'no video' still parses
    with pytest.raises(RecipeError, match="DO_VIDEO is not a launch input"):
        launch.from_env("R", env | {"DO_VIDEO": "1"})
