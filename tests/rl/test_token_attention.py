"""The gradient-pass attention of the token trunk (`token_policies._attention_grad`) is the MultiheadAttention module's
function: same outputs and same parameter gradients (FP32, CPU) for one- and two-hand token counts, and the rollout
path (no_grad) still calls the module."""

from __future__ import annotations

import pytest

from tests.imports import own_module

torch = pytest.importorskip("torch")
TP = own_module("dexengine.rl.agents.token_policies")


def _block(seed: int, d: int = 64, heads: int = 4):
    torch.manual_seed(seed)
    blk = TP._TokenBlock(d, heads)
    for p in blk.parameters():  # non-trivial biases: the module initialises them to 0
        torch.nn.init.normal_(p, std=0.3)
    return blk


@pytest.mark.parametrize("tokens", [17, 34])
def test_written_attention_equals_module(tokens):
    blk = _block(tokens)
    h = torch.randn(64, tokens, 64)
    ref = blk.attn(h, h, h, need_weights=False)[0]
    out = TP._attention_grad(blk.attn, h)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5), float((out - ref).abs().max())
    params = list(blk.attn.parameters())
    g_ref = torch.autograd.grad(ref.square().sum(), params)
    g_out = torch.autograd.grad(out.square().sum(), params)
    for a, b in zip(g_ref, g_out, strict=True):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-5 * float(a.abs().max())), float((a - b).abs().max())


def test_rollout_path_calls_module_and_grad_path_does_not(monkeypatch):
    blk = _block(1)
    x = torch.randn(8, 17, 64)
    calls = []
    orig = TP._attention_grad
    monkeypatch.setattr(TP, "_attention_grad", lambda a, h: calls.append(1) or orig(a, h))
    with torch.no_grad():
        y0 = blk(x)
    assert calls == []
    y1 = blk(x)
    assert calls == [1]
    assert torch.allclose(y0, y1, rtol=1e-5, atol=1e-5)


def test_premises_enforced(monkeypatch):
    """A block whose attention has dropout (a premise of the written-out form) is refused at construction."""
    real = torch.nn.MultiheadAttention
    monkeypatch.setattr(TP.nn, "MultiheadAttention", lambda d, h, **k: real(d, h, dropout=0.1, **k))
    with pytest.raises(AssertionError):
        TP._TokenBlock(64, 4)


def test_wrong_scale_is_caught():
    """Negative control: the comparison above detects a written-out form with the wrong softmax scale."""
    blk = _block(3)
    h = torch.randn(16, 17, 64)
    b, t, d = h.shape
    a = blk.attn
    qkv = torch.nn.functional.linear(h, a.in_proj_weight, a.in_proj_bias).view(b, t, 3, 4, 16).permute(2, 0, 3, 1, 4)
    w = torch.softmax(qkv[0] @ qkv[1].transpose(-1, -2), dim=-1)  # no 1/sqrt(d_head)
    bad = torch.nn.functional.linear((w @ qkv[2]).transpose(1, 2).reshape(b, t, d), a.out_proj.weight, a.out_proj.bias)
    ref = a(h, h, h, need_weights=False)[0]
    assert not torch.allclose(bad, ref, rtol=1e-5, atol=1e-5)
