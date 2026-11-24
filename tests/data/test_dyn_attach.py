"""S-dyn-pairs ② attach on synthetic scenes (CPU): 'moves together' at the extra-pair 5 mm / 5° band, a still that moves
together with a main object it overlaps at frame 0 rides it for the whole clip (one body by the carry path), a main
object resting on that still, a still that meets the other main object and a support partner have no path and drop,
and a separating pair is not attached."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dexengine.data.objects import carry_class as CC
from dexengine.data.objects import collider_shape as CS
from dexengine.data.objects import dyn_pairs as DP
from dexengine.data.objects import extra_slots as XS
from dexengine.data.objects import extra_slots_geom as XG
from dexengine.data.objects import scene_rules as SC
from dexengine.data.objects import touch_gate as TG
from dexengine.rl.tasks.scene import extra_carry as EC

T = 40
STILL = (0.05, 0.05, 0.03)  # half sizes (m)
MAIN = (0.02, 0.02, 0.02)


def _box_mesh(half):
    import trimesh  # noqa: PLC0415

    m = trimesh.creation.box(extents=2 * np.asarray(half, float))
    return np.asarray(m.vertices, float), np.asarray(m.faces)


def _corners(half):
    return np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * np.asarray(half)


def _col(xyz):
    return np.c_[xyz, np.ones(len(xyz)), np.zeros((len(xyz), 3))]


def _ti(support=False):
    objs = [
        {"name": "right_object", "role": "main", "side": "right", "qpos_start": 0, "mesh_dir": "a", "obj_id": "M"},
        {"name": "left_object", "role": "main", "side": "left", "qpos_start": 7, "mesh_dir": "b", "obj_id": "F"},
        {"name": "extra_object_0", "role": "support" if support else "extra", "qpos_start": 14, "mesh_dir": "c"},
    ]
    objs[2]["obj_id"] = "S"
    return {"dataset_name": "hot3d", "objects": objs, "idle_sides": []}


def _q(main_at, moves=False, far_at=(1.0, 1.0, 0.0), drift=None):
    """The still at the origin; the right main object at `main_at` (moves: leaves from frame 10); the left main far
    away (`far_at`); drift (m): the main object creeps along +y by that much over the clip."""
    xm = np.tile(np.asarray(main_at, float), (T, 1))
    if moves:
        xm[:, 0] += 0.1 * np.clip(np.arange(T) - 10, 0, None) / T
    if drift is not None:
        xm[:, 1] += np.linspace(0.0, drift, T)
    return np.concatenate([_col(xm), _col(np.tile(far_at, (T, 1))), _col(np.zeros((T, 3)))], axis=1)


def _row():
    return {
        "name": "extra_object_0",
        "cls": XS.STATIC_SPAWN,
        "kind": XS.KIND_ENVELOPE,
        "why": "envelope",
        "static": True,
    }


SIDE = (STILL[0] + MAIN[0] - 0.003, 0.0, 0.0)  # beside the still, 3 mm into its +x wall
ON_TOP = (0.0, 0.0, STILL[2] + MAIN[2] - 0.003)  # on the still's top, 3 mm into it


@pytest.fixture
def scene(monkeypatch):
    shapes = {"M": _corners(MAIN), "F": _corners(MAIN), "S": _corners(STILL)}
    tri = CS.Triangles(*_box_mesh(STILL))
    monkeypatch.setattr(SC, "on", lambda rid, ds: rid in (DP.RULE_ID, SC.ROBOT_RULE))
    monkeypatch.setattr(TG, "body_of", lambda ds, o, tree: (TG.Body([CS.Parts([shapes[o["obj_id"]]])]), ""))
    monkeypatch.setattr(TG, "static_body", lambda ds, o, row: (tri, ""))
    monkeypatch.setattr(CS, "body", lambda ds, oid, shape: tri)
    monkeypatch.setattr(CC, "parts", lambda ds, o, row=None: [shapes[o["obj_id"]]])


def _pass(q, ti=None):
    rows = [_row()]
    DP.together_pass("/x/clip_pl_rt", ti or _ti(), q, rows)
    return rows[0]


@pytest.mark.parametrize(("drift", "ok"), [(0.0049, True), (0.0051, False)])
def test_together_is_the_5mm_band_against_frame_0(drift, ok):
    q = _q(SIDE, drift=drift)
    assert (DP.together(q, 14, 0, DP.EXTRA) is not None) == ok


def test_together_reads_rotation_against_frame_0():
    q = _q(SIDE)
    from scipy.spatial.transform import Rotation as R  # noqa: PLC0415

    for deg, ok in ((4.9, True), (5.1, False)):
        qq = q.copy()
        w = R.from_euler("z", np.linspace(0, deg, T)[:, None], degrees=True).as_quat()  # (T, 1): every scipy
        qq[:, 3:7] = np.c_[w[:, 3], w[:, :3]]
        assert (DP.together(qq, 14, 0, DP.EXTRA) is not None) == ok


def test_a_still_drifting_3mm_rides_the_main_object(scene):
    """3 mm of drift is inside the extra-pair band (outside the main-pair band): the still rides."""
    r = _pass(_q(SIDE, drift=0.003))
    assert r["together"]["verdict"] == DP.ATTACHED and r["kind"] == XS.KIND_ATTACHED and r["carry"]["dev_m"] > 0.002


def test_the_main_band_would_not_attach_it(scene, monkeypatch):
    """Negative: read with the main-pair band (2 mm), the same 3 mm drift is not together and the still stays."""
    real = DP.together
    monkeypatch.setattr(DP, "together", lambda q, a, b, pair: real(q, a, b, DP.MAIN))
    r = _pass(_q(SIDE, drift=0.003))
    assert "together" not in r and r["kind"] == XS.KIND_ENVELOPE


def test_still_that_moves_together_rides_the_main_object(scene):
    q = _q(SIDE)
    r = _pass(q)
    assert r["kind"] == XS.KIND_ATTACHED and r["together"]["verdict"] == DP.ATTACHED
    cy = r["carry"]
    assert (cy["carrier"], cy["carrier_role"], cy["start"], cy["end"]) == ("right_object", "main", 0, T - 1)
    # one body: wherever the simulated main object goes, the still stands at the same relative pose (the carry path)
    pm = torch.tensor([[0.3, -0.1, 0.9]], dtype=torch.float64)
    qm = torch.tensor([[np.cos(0.4), 0.0, 0.0, np.sin(0.4)]], dtype=torch.float64)
    rp, rq = torch.tensor([cy["rel"][:3]], dtype=torch.float64), torch.tensor([cy["rel"][3:]], dtype=torch.float64)
    po, qo = EC.compose(pm, qm, rp, rq)
    Tm, To = np.eye(4), np.eye(4)
    Tm[:3, :3] = XG.pose_mats(np.array([[0, 0, 0, *qm[0].tolist()]]), 0)[0, :3, :3]
    Tm[:3, 3] = pm[0].numpy()
    To[:3, :3] = XG.pose_mats(np.array([[0, 0, 0, *qo[0].tolist()]]), 0)[0, :3, :3]
    To[:3, 3] = po[0].numpy()
    rec = np.linalg.inv(XG.pose_mats(q, 0)[0]) @ XG.pose_mats(q, 14)[0]
    dm, dd = XG.pose_dev(np.linalg.inv(Tm) @ To, rec)
    assert dm < 1e-6 and dd < 1e-3


def test_the_attached_still_reaches_the_env_carry_statement(scene):
    """The scene reads an attached still row (static spawn, pinned) like a carried slot: env_setup.extra_carry states
    its carrier and relative pose; the slot manager writes carrier × rel over its static pose (extra_carry.apply)."""
    import types  # noqa: PLC0415

    from dexengine.rl.env_setup import extra_carry as XC  # noqa: PLC0415

    r = _pass(_q(SIDE))
    assert (r["cls"], r["static"], r["kind"]) == (XS.STATIC_SPAWN, True, XS.KIND_ATTACHED)
    cfg = types.SimpleNamespace()
    XC.carry_statement(cfg, [[r, None]], 2)
    (st,) = cfg.extra_slot_carry
    assert st[1] is None and st[0]["carrier_role"] == "main" and st[0]["carrier_side"] == "right"
    assert st[0]["carrier_slot"] is None and st[0]["rel"] == r["carry"]["rel"]


def test_attached_still_drops_nothing_for_its_overlap_with_that_main(scene):
    rows = [{**_row(), "overlap_drop": {"partner": "right_object", "depth_m": 0.008}}]
    DP.together_pass("/x/clip_pl_rt", _ti(), _q(SIDE), rows)
    assert "overlap_drop" not in rows[0] and rows[0]["together"]["overlap_drop"]["depth_m"] == 0.008
    assert CS.overlap_drop(rows) is None
    rows = [{**_row(), "overlap_drop": {"partner": "left_object", "depth_m": 0.008}}]
    DP.together_pass("/x/clip_pl_rt", _ti(), _q(SIDE), rows)
    assert CS.overlap_drop(rows) is not None  # an overlap with another body still drops


def test_main_object_resting_on_the_still_has_no_path(scene):
    r = _pass(_q(ON_TOP))
    assert r["kind"] == XS.KIND_ENVELOPE and "carry" not in r
    assert r["together"]["verdict"] == DP.NO_PATH_V and r["together"]["why"] == DP.NO_PATH["rests"]


def test_still_that_meets_the_other_main_object_has_no_path(scene):
    far = (-(STILL[0] + MAIN[0] - 0.003), 0.0, 0.0)  # the left main object in the still's -x wall
    r = _pass(_q(SIDE, far_at=far))
    assert "carry" not in r and r["together"]["why"] == DP.NO_PATH["other"]


def test_separating_pair_is_not_attached(scene):
    r = _pass(_q(SIDE, moves=True))
    assert "together" not in r and "carry" not in r and r["kind"] == XS.KIND_ENVELOPE


def test_no_path_drops_the_tree_in_the_statement(scene):
    q = _q(ON_TOP)
    st = DP.dyn_pairs("/x/clip_pl_rt", _ti(), q, rows=[_pass(q)])
    (t,) = st["together"]
    assert (t["a"], t["b"], t["verdict"]) == ("right_object", "extra_object_0", DP.NO_PATH_V)
    assert st["no_path"][0] == DP.NO_PATH_WORD


def test_support_partner_that_moves_together_has_no_path(scene):
    q = _q(ON_TOP)
    st = DP.dyn_pairs("/x/clip_pl_rt", _ti(support=True), q, rows=[])
    (t,) = st["together"]
    assert (t["kind"], t["verdict"], t["why"]) == ("support", DP.NO_PATH_V, DP.NO_PATH["support"])
    assert st["no_path"][0] == DP.NO_PATH_WORD
