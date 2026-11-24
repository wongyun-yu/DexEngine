"""Left keypoint terms (rl.env_setup.object_rewards_left), CPU only.

The split moved the `kp_farthest` default into the path helper, where it was dead, so every single-object left side
died with UnboundLocalError at env build; the helper also dropped the absolute multi-object mesh dir it computed.
These tests build the left terms the way the earlier single block did."""

from __future__ import annotations

import os
import types

import numpy as np
import pytest

from dexengine.rl.env_setup import object_rewards_left as L


def _term(func, weight, params):
    return {"func": func, "weight": weight, "params": params}


def _cfg():
    return types.SimpleNamespace(rewards=types.SimpleNamespace())


def _obj_dir(root, name):
    d = root / name
    d.mkdir()
    np.save(d / "keypoints_fps2.npy", np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32))
    np.savez(d / "surface_points.npz", points=np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]]))
    return d


@pytest.mark.parametrize("mode, want_idx", [("farthest", [1]), ("pair", [0, 1])])
def test_single_object_left_terms_build_with_kp_farthest_0(tmp_path, mode, want_idx):
    d = _obj_dir(tmp_path, "obj")
    path, multi, ok, mesh = L._left_keypoint_paths(os, str(d))
    assert (multi, ok, mesh) == (False, True, str(d)) and path == os.path.join(str(d), "keypoints_fps2.npy")
    cfg = _cfg()
    _coords, _diag, idx, names = L._register_left_keypoint_terms(
        cfg, True, 0.0, mode, 1.0, "rok", "rokd", _term, np, os, path, os.path.abspath(mesh), multi
    )
    assert idx == want_idx and names == [f"object_keypoint_{i + 1}_left" for i in want_idx]
    for i in want_idx:
        assert getattr(cfg.rewards, f"object_keypoint_{i + 1}_left")["params"]["kp_farthest"] == 0
        assert getattr(cfg.rewards, f"diag_object_keypoint_{i + 1}_left_err")["params"]["kp_farthest"] == 0


def test_multi_object_left_dir_comes_back_absolute_and_farthest_is_1(tmp_path, monkeypatch):
    _obj_dir(tmp_path, "a")
    _obj_dir(tmp_path, "b")
    monkeypatch.chdir(tmp_path)
    path, multi, ok, mesh = L._left_keypoint_paths(os, "a,b")
    assert (path, multi, ok) == (None, True, True)
    assert mesh == f"{tmp_path / 'a'},{tmp_path / 'b'}"  # absolute, entry by entry
    cfg = _cfg()
    _coords, _diag, idx, _names = L._register_left_keypoint_terms(
        cfg, True, 0.0, "farthest", 1.0, "rok", "rokd", _term, np, os, path, mesh, multi
    )
    t = cfg.rewards.object_keypoint_1_left["params"]
    assert idx == [0] and t["kp_farthest"] == 1 and t["mesh_dir"] == mesh
