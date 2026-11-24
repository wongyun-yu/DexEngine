"""DP training: wrapper options and the recipe arguments (no training is run)."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.policies.dp import train as T
from experiments.policies.dp import train_opts as TO

ARGS = [
    "--dataset",
    "/d/dense_dp_pour_ours",
    "--out",
    "/o/run",
    "--batch",
    "64",
    "--steps",
    "166650",
    "--save-freq",
    "16665",
]


def test_take_strips_dp_options_and_keeps_lerobot_ones():
    rest = TO.take(["--batch_size=64", "--dp.columns=1", "--dp.ddp-comm=bf16", "--seed=42"])
    assert rest == ["--batch_size=64", "--seed=42"]
    assert TO.get("columns") == "1" and TO.get("ddp_comm") == "bf16" and TO.get("amp") == "fp32"
    with pytest.raises(SystemExit):
        TO.take(["--dp.skew_test=5"])
    TO._VALUES.update(TO.DEFAULTS)


def test_recipe_arguments_and_trainer():
    a = T.parse(ARGS)
    got = T.lerobot_args(a, Path("/d/dense_dp_pour_ours"))
    assert got[:2] == ["--dataset.repo_id=e5/dense_dp_pour_ours", "--dataset.root=/d/dense_dp_pour_ours"]
    for x in (
        "--policy.crop_shape=[160,288]",
        "--policy.n_obs_steps=2",
        "--policy.horizon=16",
        "--policy.n_action_steps=8",
        "--seed=42",
        "--batch_size=64",
        "--steps=166650",
        "--save_freq=16665",
        "--wandb.enable=false",
        "--output_dir=/o/run/train",
    ):
        assert x in got
    assert not any(x.startswith("--policy.pretrained_backbone_weights") for x in got)
    a = T.parse(ARGS + ["--dp.frame_cache=/c"])
    assert a.dp == ["--dp.frame_cache=/c"]
    assert T.trainer(a, ["0"])[-1] == "experiments.policies.dp.train_ema"
    assert T.trainer(a, ["0", "1"])[-1] == "experiments.policies.dp.train_ddp"
    with pytest.raises(SystemExit):
        T.parse(ARGS + ["--bogus"])
