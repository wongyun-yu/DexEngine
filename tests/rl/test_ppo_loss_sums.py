"""PPO update: the loss sums stay on the device and keep the logged bits (dexengine.rl.agents.loss_sums).

`test_means_bit_equal_to_item_sum` compares LossSums with the Python sum of `.item()` it replaced, bit for bit, on
float32 losses of mixed sign and scale (the old sum started from the int 0). `test_no_item_in_minibatch_loop` is the
structure rule: neither PPO `_update` calls `.item()` inside its minibatch loop (it fails on the pre-change code)."""

from __future__ import annotations

import ast
import struct
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dexengine.rl.agents.loss_sums import LossSums  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PPO_FILES = ("dexengine/rl/agents/per_side_ppo.py", "dexengine/rl/train/klsafe_ppo.py")


def _bits(x: float) -> bytes:
    return struct.pack("<d", x)


@pytest.mark.parametrize("seed", range(20))
def test_means_bit_equal_to_item_sum(seed):
    g = torch.Generator().manual_seed(seed)
    names = ("policy", "value", "entropy")
    epochs, mbs = 5, 16
    old = dict.fromkeys(names, 0)
    sums = LossSums(names, "cpu")
    for _ in range(epochs * mbs):
        for n in names:
            scale = 10.0 ** torch.randint(-6, 4, (1,), generator=g).item()
            x = (torch.randn((), generator=g) * scale).to(torch.float32)
            old[n] += x.item()
            sums.add(n, x)
    new = sums.means(epochs * mbs)
    for n in names:
        assert _bits(new[n]) == _bits(old[n] / (epochs * mbs)), n


def test_untouched_name_is_zero_like_the_int_start():
    sums = LossSums(("entropy",), "cpu")
    assert _bits(sums.means(80)["entropy"]) == _bits(0 / 80)


def test_negative_zero_and_nan_follow_the_python_sum():
    sums = LossSums(("a", "b"), "cpu")
    sums.add("a", torch.tensor(-0.0))
    sums.add("b", torch.tensor(float("nan")))
    out = sums.means(1)
    assert _bits(out["a"]) == _bits((0 + -0.0) / 1)
    assert out["b"] != out["b"]


def _minibatch_loops(tree: ast.AST):
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "_update":
            for node in ast.walk(fn):
                if isinstance(node, ast.For) and isinstance(node.iter, ast.Name) and node.iter.id == "sampled_batches":
                    yield node


@pytest.mark.parametrize("rel", PPO_FILES)
def test_no_item_in_minibatch_loop(rel):
    loops = list(_minibatch_loops(ast.parse((ROOT / rel).read_text())))
    assert loops, f"{rel}: no `for … in sampled_batches` loop in _update — the rule lost its anchor"
    calls = [
        n.lineno
        for loop in loops
        for n in ast.walk(loop)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "item"
    ]
    assert not calls, f"{rel}: .item() inside the minibatch loop at lines {calls} (a host sync per minibatch)"
