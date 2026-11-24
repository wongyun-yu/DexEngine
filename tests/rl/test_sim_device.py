"""The env sim device follows AppLauncher's --device (isaac_cli.sim_device): given → that value, distributed → the
rank's GPU, neither → the cfg default unchanged, both → error; and rl/cli.py builds the env cfg through it and prints
the [SIM_DEVICE] marker (structure check, with a negative test that removing the wiring is caught)."""

from __future__ import annotations

import ast

import pytest

from dexengine.core.repo import REPO_ROOT
from dexengine.rl.isaac_cli import SIM_DEVICE_MARK, sim_device

CLI = REPO_ROOT / "dexengine/rl/cli.py"
DEFAULT = "cuda:0"  # AppLauncher's --device parser default


def test_no_flag_keeps_the_cfg_default():
    assert sim_device(DEFAULT, DEFAULT, "cuda:0", False, 0) == ("cuda:0", "default")
    assert sim_device(None, DEFAULT, "cuda:1", False, 0) == ("cuda:1", "default")


def test_the_flag_sets_the_sim_device():
    assert sim_device("cpu", DEFAULT, "cuda:0", False, 0) == ("cpu", "--device")
    assert sim_device("cuda:1", DEFAULT, "cuda:0", False, 0) == ("cuda:1", "--device")


def test_distributed_takes_the_rank_gpu_and_refuses_the_flag():
    assert sim_device(DEFAULT, DEFAULT, "cuda:0", True, 3) == ("cuda:3", "distributed rank 3")
    with pytest.raises(ValueError, match="distributed"):
        sim_device("cpu", DEFAULT, "cuda:0", True, 0)


def _wired(src: str) -> bool:
    """`_construct_env` assigns cfg.sim.device from a sim_device(...) call and prints SIM_DEVICE_MARK."""
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "_construct_env")
    assigned = printed = False
    for n in ast.walk(fn):
        call = isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
        if call and getattr(n.value.func, "id", "") == "sim_device":
            targets = [e for t in n.targets for e in (t.elts if isinstance(t, ast.Tuple) else [t])]
            assigned |= any(ast.unparse(e) == "cfg.sim.device" for e in targets)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "print":
            printed |= "SIM_DEVICE_MARK" in ast.unparse(n)
    return assigned and printed


def _without(src: str, word: str) -> str:
    """cli.py with every top-level statement of `_construct_env` that mentions `word` removed."""
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_construct_env")
    fn.body = [s for s in fn.body if word not in ast.unparse(s)]
    return ast.unparse(tree)


def test_cli_builds_the_env_cfg_through_sim_device():
    src = CLI.read_text(encoding="utf-8")
    assert _wired(src)
    assert not _wired(_without(src, "sim_device("))  # negative: the assignment removed is caught
    assert not _wired(_without(src, "SIM_DEVICE_MARK"))  # negative: the marker removed is caught
    assert SIM_DEVICE_MARK == "[SIM_DEVICE]"  # the literal the off-card preflight greps
