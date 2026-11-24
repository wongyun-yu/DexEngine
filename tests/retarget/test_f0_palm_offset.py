"""The f0 hand stands where the training robot's hand stands: the URDF palm at the rig palm (the wrist cache) + R · the
palm offset (HandModel.palm_inv, owner placement.arm_ik.palm_frame_offset). Every palm-relative transform reads it: the
host FK (link_points_world), the object projection (Projection.link_T) and the GPU hand spec (palm inverse)."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tests.imports import own_module

REAL_DATA_ROOT = os.environ.get("DEXENGINE_MAIN_ROOT", "")
#: a SharpA tree whose scene.xml carries the rig hand's collision geoms (group 3) — measured 29.5 mm off before the fix
SHARPA_TREE = ("hocap", "right", "subject_1-20231025_170231__0_seg03", "right")


def _hand(monkeypatch, hand: str, side: str = "right"):
    """The f0 strict HandModel of `hand` (every HC reader rebound for this test only)."""
    monkeypatch.setenv("DEXENGINE_HAND", hand)
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    C = own_module("dexengine.retarget.f0.repair_cfg")
    for name in C.HC_READERS:
        m = importlib.import_module(name)
        monkeypatch.setattr(m, "HC", m.HC)  # restored after the test
    C.share_hand_cfg(C.HandCfg(hand, "rby1"))
    from dexengine.retarget.f0 import repair_geom as G

    return G.HandModel(side, max_hull_verts=10**9)


def _pose(seed=0):
    from dexengine.retarget.f0.repair_geom import _aa_to_R

    rng = np.random.default_rng(seed)
    return rng.normal(scale=0.2, size=3), _aa_to_R(rng.normal(scale=0.7, size=3))


@pytest.mark.parametrize("hand", ["sharpa_wave", "wuji2"])
def test_every_piece_moves_by_the_palm_offset(monkeypatch, hand):
    """At any wrist pose, every collision point stands at the offset-free place + wrist R · offset: SharpA 29.5 mm
    along the palm z; a hand without an offset (wuji2) keeps every bit."""
    import pinocchio as pin

    from dexengine.placement import arm_ik as AIP

    if hand != "sharpa_wave":  # its meshes resolve under the data root (SharpA's ship with the package assets)
        if not REAL_DATA_ROOT:
            pytest.skip("needs the data root (DEXENGINE_MAIN_ROOT=<root> pytest ...)")
        monkeypatch.setenv("DEXENGINE_MAIN_ROOT", REAL_DATA_ROOT)
    h = _hand(monkeypatch, hand)
    off = np.asarray(AIP.palm_frame_offset(hand), float)
    assert np.array_equal(h.palm_off, off)
    wp, wR = _pose()
    f = h.fingers12(np.zeros(len(h.jnames)))
    got = h.link_points_world(wp, wR, f)
    h.palm_shift = pin.SE3.Identity()  # the model before the fix: the URDF palm on the rig palm
    before = h.link_points_world(wp, wR, f)
    for (ln, P, _R, _t, _d), (_l, Q, _R0, _t0, _d0) in zip(got, before, strict=True):
        if not off.any():
            assert np.array_equal(P, Q), ln
        else:
            assert np.allclose(P - Q, wR @ off, atol=1e-12), ln
    if hand == "sharpa_wave":
        assert np.isclose(np.linalg.norm(off), 0.0295)


def test_projection_and_gpu_spec_read_the_same_palm(monkeypatch):
    """Projection.link_T and the GPU hand spec's palm inverse give the host FK's transforms on SharpA (offset on)."""
    from dexengine.retarget.f0 import repair_obj as RO
    from dexengine.retarget.f0.gpu_specs import GpuHandSpec
    from dexengine.retarget.f0.repair_obj_solve import Projection

    h = _hand(monkeypatch, "sharpa_wave")
    wp, wR = _pose(1)
    indep = np.zeros(len(h.jnames))
    host = {ln: (R, t) for ln, _p, R, t, _d in h.link_points_world(wp, wR, h.fingers12(indep))}
    cobj = RO.CookedObject([np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float) + 5.0])
    P = Projection(h, cobj, np.eye(3), np.zeros(3), (wp, wR, indep), -1.0, (indep - 1, indep + 1), set())
    for ln, (R, t) in zip(P.names, P.link_T(np.zeros(P.wpos0.shape[0] + 3 + len(indep))), strict=True):
        assert np.allclose(R, host[ln][0], atol=1e-12) and np.allclose(t, host[ln][1], atol=1e-12), ln
    g = GpuHandSpec(h)
    bf = h.model.frames[h.base_frame]
    assert np.allclose(g.palm_inv_t, bf.placement.inverse().translation + h.palm_off, atol=1e-15)
    assert np.allclose(g.palm_inv_R, bf.placement.inverse().rotation, atol=1e-15)


def test_the_f0_key_carries_a_nonzero_palm_offset_only(monkeypatch):
    """hand_key moves with a non-zero palm offset; with the offset 0 it is the key before the rule."""
    from dexengine.placement import arm_ik as AIP
    from dexengine.retarget.f0 import shadow_key as SK

    _hand(monkeypatch, "sharpa_wave")
    monkeypatch.setattr(SK, "_GEOM_MEMO", {})
    on = SK.hand_key("right")[0]
    monkeypatch.setattr(SK, "_GEOM_MEMO", {})
    monkeypatch.setattr(AIP, "palm_frame_offset", lambda hand: np.zeros(3))
    zero = SK.hand_key("right")[0]
    assert on != zero


@pytest.mark.skipif(not REAL_DATA_ROOT, reason="needs the data root (DEXENGINE_MAIN_ROOT=<root> pytest ...)")
def test_the_hand_stands_on_the_rig_collision_geoms(monkeypatch):
    """On a SharpA tree at frame 0, the f0 collision pieces coincide with the tree scene.xml's collision geoms (group
    3, the rig the training robot's IK follows): rigid fit residual and translation 0. Negative: without the palm
    offset they sit 29.5 mm off along the palm z."""
    import mujoco
    import pinocchio as pin

    ds, emb, clip, side = SHARPA_TREE
    tree = Path(REAL_DATA_ROOT) / f"data/{ds}/base/sharpa_wave/{emb}/{clip}"
    wrist = Path(REAL_DATA_ROOT) / f"precompute/{ds}/robots/sharpa_wave/wrist_poses_{side}/{clip}.npz"
    if not (tree / "scene.xml").is_file() or not wrist.is_file():
        pytest.skip(f"{tree} not in this data root")
    h = _hand(monkeypatch, "sharpa_wave", side)
    from dexengine.retarget.f0.repair_geom import _quat_wxyz_to_R

    q = np.load(tree / "0/trajectory_kinematic.npz")["qpos"][:1]
    w = np.load(wrist)
    wp, wR = w["positions"][0].astype(float), _quat_wxyz_to_R(w["quaternions"][0].astype(float))
    ti = json.loads((tree / "task_info.json").read_text())
    hs = next(e for e in ti["qpos_layout"] if e.get("kind") == "hand" and e["side"] == side)
    f = q[0, int(hs["start"]) + 6 : int(hs["start"]) + int(hs["width"])]
    m = mujoco.MjModel.from_xml_path(str(tree / "scene.xml"))
    d = mujoco.MjData(m)
    d.qpos[:] = q[0]
    mujoco.mj_kinematics(m, d)

    def rig(body):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) == b and int(m.geom_group[g]) == 3:
                mid = int(m.geom_dataid[g])
                va = int(m.mesh_vertadr[mid])
                V = np.asarray(m.mesh_vert[va : va + int(m.mesh_vertnum[mid])], float)
                return V @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        return None

    def offsets():
        out = []
        for ln, _p, R, t, _d in h.link_points_world(wp, wR, h.fingers12(f)):
            W = rig(ln)
            if W is not None and len(h.parts[ln]) == 1 and len(h.parts[ln][0]) == len(W):
                out.append(W - (np.asarray(h.parts[ln][0]) @ R.T + t))
        return np.vstack(out)

    D = offsets()
    assert len(D) > 1000 and np.abs(D).max() < 1e-6
    h.palm_shift = pin.SE3.Identity()
    D0 = offsets()
    assert np.allclose(np.linalg.norm(D0, axis=1), 0.0295, atol=1e-6)
