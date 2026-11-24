"""DP policy server: wire encoding, option validation and the constant-state fix on a stand-in policy."""

from __future__ import annotations

import io
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
msgpack = pytest.importorskip("msgpack")

from experiments.policies.dp import serve_model as SM


def _wire():
    # server.py imports zmq at module level; the codec functions are tested without it.
    import sys

    try:
        import zmq  # noqa: F401
    except ImportError:
        sys.modules.setdefault("zmq", types.ModuleType("zmq"))
    from experiments.policies.dp import server as S

    return S


def test_ndarray_roundtrip_matches_npy_bytes():
    S = _wire()
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    packed = msgpack.packb({"x": a, "k": 3}, default=S.enc)
    out = msgpack.unpackb(packed, object_hook=S.dec)
    assert out["k"] == 3 and out["x"].dtype == np.float32 and np.array_equal(out["x"], a)
    b = io.BytesIO()
    np.save(b, a, allow_pickle=False)
    assert S.enc(a) == {"__ndarray_class__": True, "as_npy": b.getvalue()}


def test_sampler_override_rules():
    assert SM.sampler_override("", 0) is None
    assert SM.sampler_override("DDIM", 10) == ("DDIM", 10)
    with pytest.raises(SystemExit):
        SM.sampler_override("DDIM", 0)
    with pytest.raises(SystemExit):
        SM.sampler_override("EULER", 5)


def test_cli_defaults_equal_the_unset_behaviour():
    S = _wire()
    a = S.parse(["--ckpt", "c", "--root", "r"])
    assert (a.host, a.port, a.out, a.sampler, a.infer_steps) == ("0.0.0.0", 5556, "", "", 0)
    assert not (a.const_state_fix or a.oracle_exact or a.oracle_hold_end) and a.const_state_eps == 0.0


def _stand_in(mn, mx, mode="MIN_MAX"):
    buffers = {"buffer_observation_state.min": torch.tensor(mn), "buffer_observation_state.max": torch.tensor(mx)}
    norm = types.SimpleNamespace(named_buffers=lambda: list(buffers.items()), named_parameters=list)
    return types.SimpleNamespace(
        config=types.SimpleNamespace(normalization_mapping={"STATE": f"NormalizationMode.{mode}"}),
        normalize_inputs=norm,
    )


def test_const_state_dims_exact_and_eps():
    mn = np.zeros(56, np.float32)
    mx = np.ones(56, np.float32)
    mx[14] = 0.0  # head_joints[0] constant
    mx[15] = 8.34e-07  # head_joints[1] float-noise width
    ck = "/r/dense_dp_pour_kin_v3/train/checkpoints/166650/pretrained_model"
    dims, mins, mark = SM.const_state_dims(_stand_in(mn, mx), ck, 0.0)
    assert list(dims) == [14] and mins.dtype == np.float32
    assert mark == "[DP_CONST_STATE] dims=head_joints[0] ckpt=dense_dp_pour_kin_v3@166650 eps=0 widths=0"
    dims, _, mark = SM.const_state_dims(_stand_in(mn, mx), ck, 1e-6)
    assert list(dims) == [14, 15] and mark.startswith("[DP_CONST_STATE] dims=head_joints[0],head_joints[1] ")
    with pytest.raises(SystemExit):
        SM.const_state_dims(_stand_in(mn, mx, "MEAN_STD"), ck, 0.0)
    with pytest.raises(SystemExit):
        SM.const_state_dims(_stand_in(mn, mx), ck, 1e-3)
