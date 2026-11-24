"""GR00T: fine-tuning recipe arguments and the server launcher's pure checks (nothing is started)."""

from __future__ import annotations

import json

import pytest

from experiments.policies.gr00t import server as SV
from experiments.policies.gr00t import train as T

ARGS = [
    "--gr00t-repo",
    "/g",
    "--base",
    "/b",
    "--dataset",
    "/d",
    "--out",
    "/o",
    "--gpus",
    "0,1,2,3,4,5,6,7",
    "--steps",
    "80000",
    "--save-steps",
    "20000",
]


def test_recipe_a_arguments():
    a = T.parse(ARGS)
    got = T.finetune_args(a, 8, 80000)
    pairs = dict(zip(got[::2], got[1::2], strict=False))
    assert pairs["--global-batch-size"] == "256" and pairs["--learning-rate"] == "1.5e-4"
    assert pairs["--num-shards-per-epoch"] == "256" and pairs["--episode-sampling-rate"] == "0.1"
    assert "--tune-visual" not in got and got[-9:] == T.JITTER
    assert a.ft == ["--ft.tail_starts=1", "--ft.gr00t_minmax=1"]


def test_steps_and_shard_rules():
    assert T.steps_from_exposure(150, 1_200_000, 256) == 703000
    a = T.parse(ARGS + ["--shards-per-gpu", "4"])
    with pytest.raises(SystemExit):
        T.shards_per_epoch(a, 8)
    with pytest.raises(SystemExit):
        T.finetune_args(T.parse(ARGS + ["--global-batch", "100"]), 8, 1)


def test_server_checkpoint_paths_and_pidfile(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "Gr00tN1d7", "action_horizon": 40}))
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"processor_kwargs": {"modality_configs": {"new_embodiment": {}}}})
    )
    SV.check_ckpt(tmp_path, "new_embodiment")
    with pytest.raises(SystemExit):
        SV.check_ckpt(tmp_path, "gr1")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "Gr00tN1d6"}))
    with pytest.raises(SystemExit):
        SV.check_ckpt(tmp_path, "new_embodiment")
    P = SV.Paths(str(tmp_path))
    e = P.env()
    assert e["HF_HOME"] == str(tmp_path.resolve() / "hf") and e["WANDB_MODE"] == "disabled"
    pf = tmp_path / "x.pid"
    pf.write_text("999999999\n")
    assert SV.alive(pf) == 0
