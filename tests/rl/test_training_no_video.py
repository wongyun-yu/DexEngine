"""The training mode draws no video (training runs without the renderer; every robot video comes from the record
render), and DO_VIDEO is a removed input: 0 (the callers' old 'no video') still parses, 1 dies."""

from __future__ import annotations

import ast

import pytest

from dexengine.core.repo import REPO_ROOT
from dexengine.rl import launch
from dexengine.rl.launch.recipe import RecipeError
from tests.rl.test_launch import CASES, parse_case

TRAINING = REPO_ROOT / "dexengine/rl/launch/training.py"
RENDER_NAMES = ("render_call", "render_calls", "render_out", "scene_env")


def _render_calls(src: str) -> list[str]:
    return [
        ast.unparse(n.func)
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call) and ast.unparse(n.func).split(".")[-1] in RENDER_NAMES
    ]


def test_the_training_mode_draws_no_video():
    src = TRAINING.read_text(encoding="utf-8")
    assert _render_calls(src) == []
    marker = "    return 0\n\n\ndef _auto_resume"
    assert src.count(marker) == 1
    broken = src.replace(marker, "    modes.render_call(plan, ck)\n" + marker)
    assert _render_calls(broken) == ["modes.render_call"]  # negative: a render launch in the training mode is seen


def test_do_video_is_a_removed_input():
    env, _ = parse_case(next(c for c in CASES if c.stem == "train_bi_sharpa_wave"))
    assert launch.from_env("R", env | {"DO_VIDEO": "0"}).mode == "train"
    with pytest.raises(RecipeError, match="DO_VIDEO is not a launch input"):
        launch.from_env("R", env | {"DO_VIDEO": "1"})
