"""f0 writers put the repaired wrist rotation into the qpos (repair_root): the palm FK of the written rows equals the
wrist-cache rows, a side whose solve did not turn the wrist keeps its bytes, and a shadow written before the fix is
stale."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("mujoco")
HAND = "sharpa_wave"
PALM = "right_hand_C_MC"  # HandSpec.palm_fk_body_right of HAND (geom.palm_fk reads it)
T = 30


def _scene(n_fing: int) -> str:
    """A hand root like the generated trees: slides in a turned body, three hinges (−x, −z, −y) in the palm body."""
    fj = "".join(
        f'<body name="right_f{i}" pos="0.05 {0.002 * i:.3f} 0"><joint name="f{i}" type="hinge" axis="0 1 0"/>'
        '<geom type="sphere" size="0.005"/></body>'
        for i in range(n_fing)
    )
    return (
        '<mujoco><compiler angle="radian"/><worldbody>'
        '<body name="right_root" pos="0.1 -0.2 0.3" quat="0.7071068 0 0 0.7071068">'
        '<joint name="px" type="slide" axis="1 0 0"/><joint name="py" type="slide" axis="0 1 0"/>'
        '<joint name="pz" type="slide" axis="0 0 1"/><geom type="sphere" size="0.01"/>'
        f'<body name="{PALM}" quat="0.9238795 0.3826834 0 0">'
        '<joint name="rx" type="hinge" axis="-1 0 0"/><joint name="rz" type="hinge" axis="0 0 -1"/>'
        '<joint name="ry" type="hinge" axis="0 -1 0"/><geom type="sphere" size="0.02"/>'
        f"{fj}</body></body>"
        '<body name="obj" pos="0.3 0 0"><freejoint/><geom type="box" size="0.01 0.01 0.01"/></body>'
        "</worldbody></mujoco>"
    )


def _env(monkeypatch):
    monkeypatch.setenv("DEXENGINE_HAND", HAND)
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    from dexengine.retarget.f0 import repair_obj_write as OW
    from dexengine.retarget.f0 import repair_root as RT
    from dexengine.retarget.f0 import repair_write as W

    return W, OW, RT


def _palm(tree, qpos):
    import mujoco

    from dexengine.data.base_layer import geom as G

    m = mujoco.MjModel.from_xml_path(str(tree / "scene.xml"))
    return G.palm_fk(m, qpos, ["right"], HAND)


def _tree(tmp_path, H):
    """Tree c0 (scene + qpos) and its wrist cache = the palm FK of every row (what the chain's wrist stage writes)."""
    from scipy.spatial.transform import Rotation as Rot

    rng = np.random.default_rng(0)
    tree = tmp_path / "data/c0"
    (tree / "0").mkdir(parents=True)
    (tree / "scene.xml").write_text(_scene(H - 6))
    lay = [{"kind": "hand", "side": "right", "start": 0, "width": H}, {"kind": "object", "start": H, "width": 7}]
    (tree / "task_info.json").write_text(json.dumps({"qpos_layout": lay, "hand_qpos_dim": H, "objects": []}))
    qpos = np.zeros((T, H + 7))
    qpos[:, :3] = rng.normal(0, 0.05, (T, 3))
    qpos[:, 3:6] = 0.3 + 0.01 * np.arange(T)[:, None] * np.array([1.0, -0.5, 0.7])
    qpos[:, 6:H] = rng.uniform(0, 0.3, (T, H - 6))
    qpos[:, H + 3] = 1.0
    np.savez(tree / "0/trajectory_kinematic.npz", qpos=qpos, qvel=np.zeros_like(qpos))
    fk = _palm(tree, qpos)
    pos = np.stack([f[0][0] for f in fk])
    quat = Rot.from_matrix(np.stack([f[0][1] for f in fk])).as_quat()[:, [3, 0, 1, 2]]
    (tmp_path / "w/wrist_poses_right").mkdir(parents=True)
    np.savez(tmp_path / "w/wrist_poses_right/c0.npz", positions=pos, quaternions=quat)
    return tree, qpos, pos, quat


def _side(W, qpos, pos, quat, dq):
    from dexengine.retarget.f0.repair_geom import _quat_wxyz_to_R

    H = W.HC.H
    indep0 = qpos[0, 6:H][W.HC.indep_idx]
    internal = (np.asarray(dq, float), pos[0], quat[0], _quat_wxyz_to_R(quat[0]), indep0, qpos[0, 6:H], 0, None)
    return {"right": {"status": "REPAIRED", "dq": list(map(float, dq)), "_internal": internal}}


def _rot_err(tree, rows, quat_rows):
    from scipy.spatial.transform import Rotation as Rot

    R = Rot.from_matrix(np.stack([f[0][1] for f in _palm(tree, rows)]))
    want = Rot.from_quat(np.asarray(quat_rows)[:, [1, 2, 3, 0]])
    return float((R.inv() * want).magnitude().max())


def _write_in_place(monkeypatch, tmp_path, dq, fixed=True):
    W, _OW, RT = _env(monkeypatch)
    tree, qpos, pos, quat = _tree(tmp_path, W.HC.H)
    monkeypatch.setattr(W, "_tree_sides", lambda t: ["right"])
    if not fixed:  # the writer before the fix: the rotation reached the wrist cache only
        monkeypatch.setattr(RT, "turns", lambda dq: False)
    sides = _side(W, qpos, pos, quat, dq)
    rq = W._R_to_quat_wxyz(W._aa_to_R(np.asarray(dq[3:6], float)) @ sides["right"]["_internal"][3])
    mode, err = W.write_in_place("c0", sides, tmp_path / "data", tmp_path / "w", 20)
    assert err is None and mode == "prepend"
    q2 = np.load(tree / "0/trajectory_kinematic.npz")["qpos"]
    c2 = np.load(tmp_path / "w/wrist_poses_right/c0.npz")["quaternions"]
    return tree, q2, c2, rq


DQ_TURN = [0.004, -0.002, 0.003, 0.03, -0.02, 0.025] + [0.0] * 22


def test_written_frame0_rotation_is_the_repaired_one(monkeypatch, tmp_path):
    """Planted turn: the qpos FK at frame 0 equals rq, and over the pad the FK follows the cache slerp row by row."""
    tree, q2, c2, rq = _write_in_place(monkeypatch, tmp_path, DQ_TURN)
    assert _rot_err(tree, q2[:1], [rq]) < 1e-6
    assert _rot_err(tree, q2[:20], c2[:20]) < 1e-6  # the f0 cache = the FK of the qpos (float32 cache)
    meta = json.loads((tree / "F0_REPAIR_INPLACE.json").read_text())
    assert meta["sides"]["right"]["root_rot"]["rows"] == 20


def test_old_writer_misses_the_rotation(monkeypatch, tmp_path):
    """Negative: without the rotation write the frame-0 FK is off by the solve's turn (|dq[3:6]| = 2.4 deg)."""
    tree, q2, _c2, rq = _write_in_place(monkeypatch, tmp_path, DQ_TURN, fixed=False)
    assert _rot_err(tree, q2[:1], [rq]) == pytest.approx(np.linalg.norm(DQ_TURN[3:6]), rel=1e-3)


def test_side_without_a_turn_keeps_its_bytes(monkeypatch, tmp_path):
    """dq rotation exactly 0: the fixed writer and the old one write the same qpos bytes, and no record."""
    dq = DQ_TURN[:3] + [0.0, 0.0, 0.0] + [0.01] * 22
    _t, q_new, _c, _ = _write_in_place(monkeypatch, tmp_path / "a", dq)
    _t2, q_old, _c2, _ = _write_in_place(monkeypatch, tmp_path / "b", dq, fixed=False)
    assert q_new.tobytes() == q_old.tobytes()
    assert "root_rot" not in json.loads((_t / "F0_REPAIR_INPLACE.json").read_text())["sides"]["right"]


@pytest.mark.parametrize("fixed", [True, False])
def test_window_rows_turn_the_qpos(monkeypatch, tmp_path, fixed):
    """write_obj_window with turning rows: the qpos FK of each row equals the cache row (negative: without the fix the
    rows stay at the old rotation)."""
    W, OW, RT = _env(monkeypatch)
    tree, qpos, pos, quat = _tree(tmp_path, W.HC.H)
    if not fixed:
        monkeypatch.setattr(RT, "turns", lambda dq: False)
    rows = np.zeros((8, W.HC.nd))
    rows[:, 3:6] = [0.02, 0.01, -0.03]
    r = {
        "_internal": (None,) * 6 + (0, None),
        "obj_window": rows,
        "obj_window_n": 5,
        "obj_window_sat_mm": [1.0] * 5,
        "obj_window_carried": [],
    }
    OW.write_obj_window("c0", {"right": r}, tmp_path / "data", tmp_path / "w")
    q2 = np.load(tree / "0/trajectory_kinematic.npz")["qpos"]
    c2 = np.load(tmp_path / "w/wrist_poses_right/c0.npz")["quaternions"]
    err = _rot_err(tree, q2[:8], c2[:8])
    assert (err < 1e-6) if fixed else (err > 0.03)
    assert np.array_equal(q2[8:], qpos[8:])


@pytest.mark.parametrize("record", [False, True])
def test_shadow_without_the_record_is_stale(monkeypatch, tmp_path, record):
    """cache_verdict: a shadow whose repair turned a wrist and has no root_rot record is STALE; with it FRESH."""
    monkeypatch.setenv("DEXENGINE_HAND", HAND)
    from dexengine.placement.judge.f0_key import MARKER
    from dexengine.retarget.f0 import repair_root as RT
    from dexengine.retarget.f0 import shadow_key as SK

    monkeypatch.setattr(SK, "f0_input_key", lambda *a, **k: "k")
    d = tmp_path / "shadow/c0"
    d.mkdir(parents=True)
    (d / MARKER).write_text(json.dumps({"input_key": "k"}))
    side = {"status": "REPAIRED", "dq": DQ_TURN}
    if record:
        side[RT.ROOT_ROT] = {"rows": 20}
    (d / "F0_REPAIR_INPLACE.json").write_text(json.dumps({"mode": "prepend", "sides": {"right": side}}))
    verdict, why, _k, _j = SK.cache_verdict("c0", tmp_path / "stage", {}, tmp_path / "shadow", "taco", "base")
    assert (verdict, why) == (("FRESH", "") if record else ("STALE", RT.WHY_UNWRITTEN))
    side_flat = {"status": "REPAIRED", "dq": DQ_TURN[:3] + [0.0] * 25}
    (d / "F0_REPAIR_INPLACE.json").write_text(json.dumps({"mode": "prepend", "sides": {"right": side_flat}}))
    assert SK.cache_verdict("c0", tmp_path / "stage", {}, tmp_path / "shadow", "taco", "base")[0] == "FRESH"


# --- the wrist map's translation axes come from the scene (repair_root.root_axes), not from the rows ---


def _old_fit(qpos6_all, cache_pos, A=None, where=""):
    """The wrist map before the fix: A and b fit together from the rows by least squares (the negative cases)."""
    X = np.asarray(qpos6_all, float)[:, :3]
    Xh = np.hstack([X, np.ones((len(X), 1))])
    sol, *_ = np.linalg.lstsq(Xh, np.asarray(cache_pos, float), rcond=None)
    return sol[:3].T, sol[3], float(np.abs(Xh @ sol - cache_pos).max())


MOVE = np.array([0.003, -0.002, 0.004])  # a repair's wrist move (m)


def _round_trip(fit, q6, cache, axes) -> float:
    """What a writer does with the map: the cache rows moved by MOVE, read back into qpos translation (Ainv @ (pos −
    b)); returns the largest distance (m) from the qpos rows that put the root there (q + axes⁻¹ MOVE)."""
    A, b, _res = fit(q6, cache.astype(np.float64), axes, where="t")
    back = (np.linalg.inv(A) @ (cache.astype(np.float64) + MOVE - b).T).T
    return float(np.abs(back - (q6[:, :3] + np.linalg.solve(axes, MOVE))).max())


def test_root_axes_are_the_scene_slides(monkeypatch, tmp_path):
    """root_axes reads the root's slide axes from the scene (the test root body is turned 90 deg about z), and on a
    moving tree the old row fit finds the same axes — the two agree where the rows determine them."""
    W, _OW, RT = _env(monkeypatch)
    tree, qpos, pos, _quat = _tree(tmp_path, W.HC.H)
    A = RT.root_axes(tree, 0)
    assert np.allclose(A, [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], atol=1e-12)
    assert np.abs(_old_fit(qpos[:, :6], pos)[0] - A).max() < 1e-6


def _still_rows(seed: int, jitter: float, T: int = 250):
    """Synthetic rows of a hand root that does not move: qpos translation = a fixed point + `jitter` (m) of noise, the
    root's axes a turned frame, the wrist cache = float32(p0 + A @ q) as the wrist stage stores the palm FK. jitter 0
    leaves the row fit singular, 1e-9 m nearly so, 2e-8 m (the float32 rounding of the cache) well conditioned but
    wrong — the three ways still-wrist clips broke the old fit."""
    from scipy.spatial.transform import Rotation as Rot

    rng = np.random.default_rng(seed)
    A = Rot.from_rotvec(rng.normal(0, 1, 3)).as_matrix()
    p0 = rng.uniform([-0.5, -0.6, 0.8], [0.5, 0.6, 1.2])
    q6 = np.zeros((T, 6))
    q6[:, :3] = rng.uniform(-0.3, 0.3, 3) + jitter * rng.normal(0, 1, (T, 3))
    return q6, (p0 + q6[:, :3] @ A.T).astype(np.float32), A


@pytest.mark.parametrize("jitter", [0.0, 1e-9, 2e-8])
def test_still_wrists_map_back(jitter):
    """Still-wrist rows (synthetic, six roots per jitter): through the scene's axes every cache row maps back onto its
    qpos translation within 1e-6 m, also after a repair moves them. Negative: the old row fit passes its own residual
    check, yet it is singular or wrong on each root and puts the moved rows' root beyond the writers' 2e-4 m
    (millimetres to 1e15 m)."""
    from dexengine.retarget.f0.repair_geom import WRIST_FIT_MAX_RES, fit_wrist_qpos_map

    for seed in range(6):
        q6, cache, axes = _still_rows(seed, jitter)
        assert np.ptp(q6[:, :3], axis=0).max() < 1e-6  # the root does not move
        assert _round_trip(fit_wrist_qpos_map, q6, cache, axes) < 1e-6, seed
        assert _old_fit(q6, cache.astype(np.float64))[2] < WRIST_FIT_MAX_RES  # the old check stays silent
        try:
            old = _round_trip(_old_fit, q6, cache, axes)
        except np.linalg.LinAlgError:
            old = np.inf
        assert old > 2e-4, seed


def _still_tree(tmp_path, H):
    """Tree c0 whose hand root never moves (rotation and fingers do) and its wrist cache = the palm FK of every row."""
    tree, qpos, _pos, _quat = _tree(tmp_path, H)
    from scipy.spatial.transform import Rotation as Rot

    qpos[:, :3] = qpos[0, :3]
    np.savez(tree / "0/trajectory_kinematic.npz", qpos=qpos, qvel=np.zeros_like(qpos))
    fk = _palm(tree, qpos)
    pos = np.stack([f[0][0] for f in fk])
    quat = Rot.from_matrix(np.stack([f[0][1] for f in fk])).as_quat()[:, [3, 0, 1, 2]]
    np.savez(tmp_path / "w/wrist_poses_right/c0.npz", positions=pos, quaternions=quat)
    return tree, qpos, pos, quat


@pytest.mark.parametrize("old", [False, True])
def test_a_still_root_is_written_where_the_cache_is(monkeypatch, tmp_path, old):
    """Desk writer (write_in_place, prepend) on a tree whose hand root never moves, a repair that moves the wrist but
    does not turn it: the written pad rows' palm FK sits on the cache rows (within the writers' 2e-4 m). Negative: with
    the old row fit the writer puts the root far away and nothing stops it."""
    W, _OW, RT = _env(monkeypatch)
    tree, qpos, pos, quat = _still_tree(tmp_path, W.HC.H)
    monkeypatch.setattr(W, "_tree_sides", lambda t: ["right"])
    if old:
        monkeypatch.setattr(W, "fit_wrist_qpos_map", _old_fit)
    dq = [0.004, -0.002, 0.003, 0.0, 0.0, 0.0] + [0.0] * 22
    mode, err = W.write_in_place("c0", _side(W, qpos, pos, quat, dq), tmp_path / "data", tmp_path / "w", 20)
    assert err is None and mode == "prepend"
    q2 = np.load(tree / "0/trajectory_kinematic.npz")["qpos"]
    c2 = np.load(tmp_path / "w/wrist_poses_right/c0.npz")["positions"]
    miss = float(np.abs(np.stack([f[0][0] for f in _palm(tree, q2[:20])]) - c2[:20]).max())
    assert (miss > 0.5) if old else (miss < RT.WRITE_TOL_M)


@pytest.mark.parametrize("old", [False, True])
def test_window_rows_on_a_still_root(monkeypatch, tmp_path, old):
    """Object window writer on a tree whose hand root never moves, turning rows: the rows are written and their palm
    FK equals the cache rows. Negative: with the old row fit the root goes far away and the root-rotation check
    refuses the write ([F0_ROOT_ROT] ✗ written rows' palm FK ↔ cache) — what still-wrist clips did on v0.1."""
    W, OW, RT = _env(monkeypatch)
    tree, _qpos, _pos, _quat = _still_tree(tmp_path, W.HC.H)
    if old:
        monkeypatch.setattr(OW, "fit_wrist_qpos_map", _old_fit)
    rows = np.zeros((8, W.HC.nd))
    rows[:, :3] = [0.002, -0.001, 0.003]
    rows[:, 3:6] = [0.02, 0.01, -0.03]
    r = {"_internal": (None,) * 6 + (0, None), "obj_window": rows, "obj_window_n": 5,
         "obj_window_sat_mm": [1.0] * 5, "obj_window_carried": []}  # fmt: skip
    if old:
        with pytest.raises(ValueError, match=r"\[F0_ROOT_ROT\] ✗ .*written rows' palm FK"):
            OW.write_obj_window("c0", {"right": r}, tmp_path / "data", tmp_path / "w")
        return
    OW.write_obj_window("c0", {"right": r}, tmp_path / "data", tmp_path / "w")
    q2 = np.load(tree / "0/trajectory_kinematic.npz")["qpos"]
    c2 = np.load(tmp_path / "w/wrist_poses_right/c0.npz")
    assert _rot_err(tree, q2[:8], c2["quaternions"][:8]) < 1e-6
    assert float(np.abs(np.stack([f[0][0] for f in _palm(tree, q2[:8])]) - c2["positions"][:8]).max()) < RT.WRITE_TOL_M


@pytest.mark.parametrize("old", [False, True])
def test_a_still_root_in_the_suffixed_tree(monkeypatch, tmp_path, old):
    """Desk writer without --in_place (write_outputs → _pad_length → _write_wrist_caches) on a tree whose hand root
    never moves: the `<seq>_f0` pad rows' palm FK sits on the suffixed cache rows (within 2e-4 m), through the real
    scene's axes. Negative: with the old row fit the pad root lands far from the cache and nothing stops it."""
    W, _OW, RT = _env(monkeypatch)
    tree, qpos, pos, quat = _still_tree(tmp_path, W.HC.H)
    T_ = len(qpos)
    np.savez(tree / "0/trajectory_kinematic.npz", qpos=qpos, qvel=np.zeros_like(qpos), qpos_rollout=qpos.copy(),
             contact=np.zeros((T_, 4)), contact_pos=np.zeros((T_, 4, 3)))  # fmt: skip
    monkeypatch.setattr(W, "_tree_sides", lambda t: ["right"])
    if old:
        monkeypatch.setattr(W, "fit_wrist_qpos_map", _old_fit)
    dq = [0.004, -0.002, 0.003, 0.0, 0.0, 0.0] + [0.0] * 22
    dst = W.write_outputs("c0", _side(W, qpos, pos, quat, dq), tmp_path / "data", tmp_path / "w", 20, "_f0")
    q2 = np.load(dst / "0/trajectory_kinematic.npz")["qpos"]
    c2 = np.load(tmp_path / "w/wrist_poses_right/c0_f0.npz")["positions"]
    P = len(q2) - T_
    assert P >= 20 and np.array_equal(q2[P:], qpos)
    miss = float(np.abs(np.stack([f[0][0] for f in _palm(tree, q2[:P])]) - c2[:P]).max())
    assert (miss > 0.5) if old else (miss < RT.WRITE_TOL_M)
