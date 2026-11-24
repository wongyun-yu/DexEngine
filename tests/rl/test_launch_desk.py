"""The visual desk in launches: one desk (Willow from the look mirror) for every scene — a recorded launch carries no
desk variable, and a launch that still names the removed desk choice dies before any process starts."""

from __future__ import annotations

import contextlib
import io

import pytest

from dexengine.core import desk as DK
from dexengine.rl import launch
from dexengine.rl.launch.recipe import RecipeError
from tests.rl.test_launch import CASES, MANIFEST, parse_case, setup_case

SOME = ("train_bi_sharpa_wave", "judge_uni_render", "replay_bi")  # a training, a judge-form render and a replay


def _resolve(case, tmp_path, monkeypatch, extra: dict):
    env, dumps = parse_case(case)
    dumps.pop(MANIFEST)
    run, root, sub = setup_case(case, env, dumps, tmp_path, monkeypatch)
    with contextlib.redirect_stdout(io.StringIO()):
        return list(launch.resolve(launch.from_env(run, {**sub, **extra}, root=root)))


@pytest.mark.parametrize("case", [c for c in CASES if c.stem in SOME], ids=lambda c: c.stem)
def test_recorded_launches_carry_no_desk(case, tmp_path, monkeypatch):
    for p in _resolve(case, tmp_path, monkeypatch, {}):
        assert not set(DK.REMOVED_DESK_ENV) & set(p.env)


@pytest.mark.parametrize("name", DK.REMOVED_DESK_ENV)
def test_the_removed_desk_choice_dies(name, tmp_path, monkeypatch):
    case = next(c for c in CASES if c.stem == "train_bi_sharpa_wave")
    with pytest.raises(RecipeError, match="one desk"):
        _resolve(case, tmp_path, monkeypatch, {name: "box"})
