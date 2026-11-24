"""repair._write_sequence: what one sequence's checks write — a not-in-place repair writes the suffixed tree; a clean
in-place re-run whose ledger exists writes nothing."""

from __future__ import annotations

import json
import types

import numpy as np
import pytest

from tests.imports import own_module


def _repair(monkeypatch):
    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    return own_module("dexengine.retarget.f0.repair")


def _args(in_place):
    return types.SimpleNamespace(check_only=False, in_place=in_place, n_blend=20, out_suffix="_f0")


def test_a_clean_rerun_with_its_ledger_writes_nothing(monkeypatch, tmp_path, capsys):
    """Negative: an in-place re-run of a clean tree whose F0_REPAIR.json exists leaves the ledger as it was and makes no
    suffixed tree (before, it wrote `<seq>_f0` with a pad)."""
    R = _repair(monkeypatch)
    root = tmp_path / "trees"
    (root / "c0").mkdir(parents=True)
    ledger = root / "c0/F0_REPAIR.json"
    ledger.write_text(json.dumps({"n_blend_effective": 20}))
    monkeypatch.setattr(R, "write_outputs", lambda *a: pytest.fail("write_outputs called on a clean re-run"))
    R._write_sequence(_args(True), ["right"], root, [], {}, "c0", ["right"], tmp_path / "w")
    assert [p.name for p in root.iterdir()] == ["c0"] and [p.name for p in (root / "c0").iterdir()] == [ledger.name]
    assert json.loads(ledger.read_text()) == {"n_blend_effective": 20}
    assert "ledger exists" in capsys.readouterr().out


def test_a_clean_first_run_writes_its_ledger(monkeypatch, tmp_path):
    R = _repair(monkeypatch)
    (tmp_path / "c0").mkdir()
    R._write_sequence(_args(True), ["right"], tmp_path, [], {}, "c0", ["right"], tmp_path)
    assert json.loads((tmp_path / "c0/F0_REPAIR.json").read_text())["design"] == "clean_v6_noop"


def test_a_repair_not_in_place_writes_the_suffixed_tree(monkeypatch, tmp_path):
    """A desk repair without --in_place writes `<seq>_f0`: the repaired pose is prepended as an n_blend pad, the
    original frames follow unchanged."""
    R = _repair(monkeypatch)
    W = own_module("dexengine.retarget.f0.repair_write")
    from dexengine.retarget.f0 import repair_root as RT

    # no scene in this tree: its wrist cache is qpos[:3] + a constant, i.e. the root's axes are the identity
    monkeypatch.setattr(RT, "root_axes", lambda tree, off: np.eye(3))
    HC = R.HC
    T, rng = 30, np.random.default_rng(2)
    qpos = rng.normal(0, 0.05, (T, HC.H + 7))
    d = tmp_path / "data/c0/0"
    d.mkdir(parents=True)
    lay = [{"kind": "hand", "side": "right", "start": 0, "width": HC.H}, {"kind": "object", "start": HC.H, "width": 7}]
    (d.parent / "task_info.json").write_text(json.dumps({"qpos_layout": lay, "hand_qpos_dim": HC.H, "objects": []}))
    np.savez(
        d / "trajectory_kinematic.npz",
        qpos=qpos,
        qvel=np.zeros_like(qpos),
        qpos_rollout=qpos.copy(),
        contact=np.zeros((T, 4)),
        contact_pos=np.zeros((T, 4, 3)),
    )
    (tmp_path / "w/wrist_poses_right").mkdir(parents=True)
    pos = qpos[:, :3] + np.array([0.1, 0.2, 0.3])
    np.savez(tmp_path / "w/wrist_poses_right/c0.npz", positions=pos, quaternions=np.tile([1.0, 0, 0, 0], (T, 1)))
    monkeypatch.setattr(W, "_tree_sides", lambda tree: ["right"])
    fingers0 = qpos[0, 6 : HC.H].copy()
    internal = (np.zeros(HC.nd), pos[0], np.array([1.0, 0, 0, 0]), np.eye(3), fingers0[HC.indep_idx], fingers0, 0, None)
    repaired = {"right": {"status": "REPAIRED", "_internal": internal}}
    R._write_sequence(_args(False), ["right"], tmp_path / "data", [], repaired, "c0", ["right"], tmp_path / "w")
    out = np.load(tmp_path / "data/c0_f0/0/trajectory_kinematic.npz")["qpos"]
    assert out.shape == (T + 20, qpos.shape[1]) and np.array_equal(out[20:], qpos)
    assert (tmp_path / "w/wrist_poses_right/c0_f0.npz").is_file()
