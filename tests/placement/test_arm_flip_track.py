"""arm_flip._track's two-chain schedule and the per-stream graph cache of ArmIK._dls_fast (CPU, no IK, no GPU).

`_track` runs its forward chain (f0 → T−1) and its backward chain (f0−1 → 0) interleaved — on the GPU the backward chain
on a second stream. These tests pin what makes that bit-exact: every chain makes the same `_dls` calls, with the same
targets and seeds in the same order, as the serial version, and the tracked result is equal to it. The graph-cache test
pins that a graph (its static buffers and its memory pool) is never shared by two streams."""

from __future__ import annotations

import contextlib
import types

import pytest

torch = pytest.importorskip("torch")

from dexengine.placement import arm_flip as AF  # noqa: E402
from tests.imports import own_module


def _track_serial(ik, Tt, Q0, f0, torch):
    """The serial `_track` before the two-chain schedule (oracle, verbatim)."""
    T, M = Tt.shape[0], Q0.shape[0]
    QQ = torch.zeros((T, M, ik.n), dtype=Q0.dtype, device=Q0.device)
    for rng in (range(f0, T), range(f0 - 1, -1, -1)):
        Q = Q0.clone() if rng.step == 1 else QQ[f0].clone()
        for f in rng:
            Q, _, _, _ = ik._dls(Tt[f : f + 1].expand(M, -1, -1).contiguous(), AF.TRACK_ITERS, 1e-2, 0.0, Q)
            QQ[f] = Q
    return QQ.permute(1, 0, 2).contiguous()


class _LogIK:
    """Stand-in IK: a non-linear, frame-dependent update of the seeds, and a log of every call (frame id read back from
    the target, the seeds and the arguments) so each chain's call sequence can be compared."""

    n = 7

    def __init__(self):
        self.log = []

    def _dls(self, T_tgt, iters, lam, null_alpha, q0):
        assert T_tgt.shape[0] == q0.shape[0]
        f = int(T_tgt[0, 3, 0].item())  # frame id written into an unused cell of the target
        self.log.append((f, q0.clone(), int(iters), float(lam), float(null_alpha), tuple(T_tgt.shape)))
        q = torch.sin(1.3 * q0 + T_tgt[:, 0, :1] * 0.01) + 0.5 * torch.cos(q0 * (f + 1))
        z = torch.zeros(q0.shape[0], dtype=q0.dtype)
        return q, z, z, z


def _targets(T, dtype):
    g = torch.Generator().manual_seed(T)
    Tt = torch.rand((T, 4, 4), generator=g, dtype=dtype)
    Tt[:, 3, 0] = torch.arange(T, dtype=dtype)
    return Tt


def _chains(log, f0):
    fwd = [e for e in log if e[0] >= f0]
    bwd = [e for e in log if e[0] < f0]
    return fwd, bwd


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("T, f0, M", [(9, 0, 3), (9, 1, 1), (9, 4, 5), (9, 8, 2), (1, 0, 4), (12, 3, 12)])
def test_track_equals_serial_and_each_chain_makes_the_same_calls(T, f0, M, dtype):
    Tt = _targets(T, dtype)
    Q0 = torch.rand((M, 7), generator=torch.Generator().manual_seed(M + 100 * f0), dtype=dtype)
    ik_a, ik_b = _LogIK(), _LogIK()
    want = _track_serial(ik_a, Tt, Q0, f0, torch)
    got = AF._track(ik_b, Tt, Q0, f0, torch)
    assert got.shape == want.shape == (M, T, 7)
    assert torch.equal(got, want)  # bit-equal, NaN-free inputs
    assert len(ik_a.log) == len(ik_b.log) == T
    for ca, cb in zip(_chains(ik_a.log, f0), _chains(ik_b.log, f0), strict=True):
        assert [e[0] for e in ca] == [e[0] for e in cb]  # same frames in the same order
        for ea, eb in zip(ca, cb, strict=True):
            assert torch.equal(ea[1], eb[1]) and ea[2:] == eb[2:]  # same seeds, iterations, damping, shapes
    assert ik_b.log[0][0] == f0  # frame f0 first: the backward chain starts from its solution


def test_track_leaves_the_seeds_untouched():
    Tt = _targets(6, torch.float64)
    Q0 = torch.rand((3, 7), dtype=torch.float64)
    keep = Q0.clone()
    AF._track(_LogIK(), Tt, Q0, 2, torch)
    assert torch.equal(Q0, keep)


def test_no_side_stream_off_the_gpu():
    assert AF._side_stream(_LogIK(), torch.device("cpu"), torch) is None


# ── ArmIK._dls_fast: one graph (and one pool) per stream ──────────────────────────────────────────────────────────


class _FakeStream:
    _next = [100]

    def __init__(self, device=None):
        self.cuda_stream = _FakeStream._next[0]
        _FakeStream._next[0] += 1

    def wait_stream(self, other):
        pass


class _FakeCuda:
    """Just enough of torch.cuda for `_dls_fast`'s capture/replay path: streams, a capture context, graphs, pools."""

    def __init__(self):
        self.default = _FakeStream()
        self.cur = self.default
        self.pools = []

    def current_stream(self):
        return self.cur

    def Stream(self, device=None):  # noqa: N802 — torch.cuda name
        return _FakeStream(device)

    @contextlib.contextmanager
    def stream(self, s):
        old, self.cur = self.cur, s
        try:
            yield
        finally:
            self.cur = old

    def graph_pool_handle(self):
        self.pools.append(object())
        return self.pools[-1]

    def CUDAGraph(self):  # noqa: N802 — torch.cuda name
        return types.SimpleNamespace(replay=lambda: None)

    @contextlib.contextmanager
    def graph(self, g, pool=None):
        g.pool = pool
        yield


def test_dls_fast_keeps_one_graph_and_pool_per_stream(monkeypatch):
    BI = own_module("dexengine.placement.reach_map.batch_ik")
    fake = _FakeCuda()
    tproxy = types.SimpleNamespace(cuda=fake, cat=torch.cat)
    monkeypatch.setattr(BI, "torch", tproxy)
    monkeypatch.setattr(BI, "DEV", "cuda")
    monkeypatch.setattr(BI.ArmIK, "_GRAPH_POOLS", {})
    ik = BI.ArmIK.__new__(BI.ArmIK)
    ik.n = 7
    ik._dls_consts = lambda lam: (None, None)
    ik._dls_body = lambda T, iters, lam, na, q, W, eye: (q.clone(), T[:, 0, 0].clone(), T[:, 0, 1].clone(), q[:, 0])
    T = torch.rand((5, 4, 4))
    q = torch.rand((5, 7))
    side = fake.Stream()
    for _ in range(ik.GRAPH_MIN_HITS + 1):
        ik._dls_fast(T, 4, 1e-2, 0.0, q)
        with fake.stream(side):
            ik._dls_fast(T, 4, 1e-2, 0.0, q)
    keys = sorted(ik._dls_graphs, key=lambda k: k[-1])
    assert len(keys) == 2  # one graph per stream for the same shape
    assert keys[0][:-1] == keys[1][:-1] == (BI.ArmIK._bucket(5), 4, 1e-2, 0.0)
    assert {k[-1] for k in keys} == {fake.default.cuda_stream, side.cuda_stream}
    pools = {k[-1]: ik._dls_graphs[k]["g"].pool for k in keys}
    assert pools[fake.default.cuda_stream] is not pools[side.cuda_stream]  # graphs that may overlap never share a pool
    assert set(BI.ArmIK._GRAPH_POOLS) == {fake.default.cuda_stream, side.cuda_stream}
    # buffers are per graph: the side stream's static inputs are not the current stream's
    a, b = (ik._dls_graphs[k] for k in keys)
    assert a["T_in"].data_ptr() != b["T_in"].data_ptr() and a["q_in"].data_ptr() != b["q_in"].data_ptr()
    # each stream counted its own eager warm-up calls (the cache decision never depends on the other stream)
    assert all(v == ik.GRAPH_MIN_HITS for v in ik._dls_hits.values()) and len(ik._dls_hits) == 2
