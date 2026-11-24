"""process.gpu_bind — the trainer's CUDA_VISIBLE_DEVICES is the call's GPU, and a launcher environment that names a
different GPU stops the launch before any process starts. The mismatch test fails if the check is removed; the
resolve_call test fails if the launcher stops going through the check."""

from __future__ import annotations

import pytest

from dexengine.rl.launch import process
from dexengine.rl.launch.recipe import RecipeError


def test_flag_only(capsys):
    assert process.gpu_bind("2", {}) == "2"
    assert "[GPU_BIND] visible=2 gpu=2 source=flag" in capsys.readouterr().out


def test_empty_environment_value_names_no_gpu(capsys):
    assert process.gpu_bind("1", {"CUDA_VISIBLE_DEVICES": ""}) == "1"
    assert "source=flag" in capsys.readouterr().out


def test_same_gpu_in_both(capsys):
    assert process.gpu_bind("3", {"CUDA_VISIBLE_DEVICES": "3"}) == "3"
    assert "source=env+flag" in capsys.readouterr().out


@pytest.mark.parametrize("outer,gpu", [("2", "0"), ("0,1", "0"), ("GPU-abc", "0")])
def test_mismatch_dies(outer, gpu):
    with pytest.raises(RecipeError, match="different GPUs"):
        process.gpu_bind(gpu, {"CUDA_VISIBLE_DEVICES": outer})


def test_resolve_call_goes_through_the_check(monkeypatch):
    calls = []

    def spy(gpu, environ=None):
        calls.append(gpu)
        raise RecipeError("spy")

    monkeypatch.setattr(process, "gpu_bind", spy)
    plan = type("P", (), {"env": {}, "req": type("R", (), {"switches_from": ""})(), "values": {}})()
    call = process.Call(exp="x", extra_flags="", gpu="5", entry="train")
    with pytest.raises(RecipeError, match="spy"):
        process.resolve_call(plan, call, __import__("pathlib").Path("/nonexistent"))
    assert calls == ["5"]
