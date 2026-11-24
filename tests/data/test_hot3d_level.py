"""CPU, synthetic + the committed table: HOT3D per-recording levelling of the device up axis and the receiver rule
(ground-truth heights on surface samples, bottom z on the whole mesh). Negative tests: an unlisted recording and an
Aria recording keep the device rule byte-for-byte; a table row of a non-levelled device dies; the height test reads
the surface samples, not the floor vertices; a gap above RECV_GAP_M and a centre outside the receiver's XY hull have
no receiver. Positive: a container (an object inside a bowl) and a receiver whose centre of mass lies above the
object (no centre-of-mass condition) receive."""

from __future__ import annotations

import numpy as np
import pytest

from dexengine.data.objects import recv_surface as RS
from dexengine.data.readers import source_frames as SF
from dexengine.data.support import hot3d_level as HL
from dexengine.data.support import work_surface as WS
from dexengine.placement.coords import core as C


def test_level_rotation_flattens_the_plane():
    a, b = 0.021, -0.013
    R = SF.level_rotation(a, b)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(R), 1.0)
    xy = np.random.default_rng(0).uniform(-0.5, 0.5, (50, 2))
    P = np.c_[xy, a * xy[:, 0] + b * xy[:, 1] + 0.7]
    z = (P @ R.T)[:, 2]
    assert np.ptp(z) < 1e-9  # the desk plane becomes horizontal


def test_plane_tilt_matches_arctan_after_rounding():
    rows = SF.level_rows(SF.LEVEL_TABLE)
    h = [float(np.hypot(r["a"], r["b"])) for r in rows.values()]
    assert [round(SF.plane_tilt_deg(r["a"], r["b"]), 4) for r in rows.values()] == [
        round(float(np.degrees(np.arctan(x))), 4) for x in h
    ]


def test_level_rotation_identity_without_slope():
    assert np.array_equal(SF.level_rotation(0.0, 0.0), np.eye(3))


def test_unlisted_and_aria_keep_device_rule(monkeypatch):
    monkeypatch.setattr(SF, "_LEVEL", {"Pq": {"headset": "Quest3", "a": 0.02, "b": 0.0}})
    for headset, rec in (("Aria", "Pa"), ("Quest3", "Pnot_listed")):
        up, rot, level = SF.recording_up(headset, rec)
        up0, rot0 = SF.world_up(headset)
        assert level is None and up == up0 and np.array_equal(rot, rot0)
    samples = {"u": np.random.default_rng(1).normal(size=(5, 8))}
    assert SF.to_z_up(samples, SF.recording_up("Aria", "Pa")[1]) is samples  # identity passes through untouched


def test_listed_recording_is_levelled(monkeypatch):
    monkeypatch.setattr(SF, "_LEVEL", {"Pq": {"headset": "Quest3", "a": 0.02, "b": -0.01}})
    up, rot, level = SF.recording_up("Quest3", "Pq")
    assert up == "+y" and level["version"] == SF.LEVEL_VERSION
    assert np.allclose(rot, SF.level_rotation(0.02, -0.01) @ SF.UP_TO_Z["+y"])
    rec = {"R_to_z_up": rot.tolist(), "level": level}
    assert SF.frame_key(rec) != SF.frame_key({"R_to_z_up": SF.UP_TO_Z["+y"].tolist()})


def test_listed_recording_with_other_headset_dies(monkeypatch):
    monkeypatch.setattr(SF, "_LEVEL", {"Pq": {"headset": "Quest3", "a": 0.02, "b": 0.0}})
    with pytest.raises(ValueError):
        SF.recording_up("Aria", "Pq")


def test_table_row_of_a_non_levelled_device_dies(tmp_path):
    p = tmp_path / "t.tsv"
    p.write_text("recording\theadset\ta\tb\nP1\tAria\t0.01\t0.0\n")
    with pytest.raises(ValueError):
        SF.level_rows(p)
    p.write_text("recording\theadset\ta\tb\nP1\tQuest3\t0.01\t0.0\nP1\tQuest3\t0.01\t0.0\n")
    with pytest.raises(ValueError):
        SF.level_rows(p)


def test_committed_table_is_quest3_only_and_small_tilt():
    rows = SF.level_rows(SF.LEVEL_TABLE)
    assert rows and all(r["headset"] in SF.LEVEL_DEVICES for r in rows.values())
    tilt = [SF.plane_tilt_deg(r["a"], r["b"]) for r in rows.values()]
    assert max(tilt) < 6.0  # a desk-plane fit, not an axis swap


def _rests(a, b, n_obj=4, per=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_obj):
        for _ in range(per):
            x, y = rng.uniform(-0.4, 0.4, 2)
            rows.append({"uid": str(u), "floor": a * x + b * y + 0.75, "cx": x, "cy": y, "w": 1.0})
    return rows


def test_fit_plane_recovers_slope_and_rejects_thin_evidence():
    fit = HL.fit_plane(_rests(0.02, -0.01), 0.75)
    assert fit is not None and abs(fit["a"] - 0.02) < 1e-9 and abs(fit["b"] + 0.01) < 1e-9
    assert HL.fit_plane(_rests(0.02, -0.01, n_obj=2), 0.75) is None  # < MIN_OBJECTS
    assert HL.fit_plane(_rests(0.02, -0.01, n_obj=3, per=1), 0.75) is None  # < MIN_RESTS


# ── receivers: ground-truth heights on surface samples, bottom z on the whole mesh ─────────────
_QUADS = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))


def _boxes(*boxes):
    """(vertices, triangles) of a union of axis boxes [(lo, hi), ...] — every face is a closed box surface."""
    V, F = [], []
    for lo, hi in boxes:
        n = sum(len(v) for v in V)
        V.append(np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float))
        F += [(n + a, n + b, n + c) for q in _QUADS for a, b, c in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))]
    return np.concatenate(V), np.asarray(F, int)


def _obj(*boxes):
    """{'surface': surface samples (recv_surface.surface_points), 'floor': whole-mesh vertices} of a union of boxes."""
    V, F = _boxes(*boxes)
    return {"surface": RS.surface_points(V, F), "floor": V}


def _still(xyz, yaw_deg=0.0):
    """(40, 8) samples [t x y z qw qx qy qz] at rest at xyz, turned yaw_deg about +z."""
    h = np.radians(yaw_deg) / 2
    a = np.tile([0.0, 0.0, 0.0, 0.0, np.cos(h), 0.0, 0.0, np.sin(h)], (40, 1))
    a[:, 0] = np.arange(40) * 1e9 / 30
    a[:, 1:4] = xyz
    return a


BOARD = _obj(([-0.2, -0.2, 0.0], [0.2, 0.2, 0.02]))
PHONE = _obj(([-0.03, -0.07, 0.0], [0.03, 0.07, 0.01]))
# a low step with a tall back block: the receiver's box reaches far above the step, and its centre of mass lies above
# an object standing on the step
STEP = _obj(([-0.1, -0.1, 0.0], [0.1, 0.1, 0.01]), ([0.1, -0.1, 0.0], [0.14, 0.1, 0.3]))


def _recv(geo: dict, pose: dict, uid: str = "phone", yaw: dict | None = None) -> list:
    objs = {u: _still(p, (yaw or {}).get(u, 0.0)) for u, p in pose.items()}
    return WS.receivers_of(objs, lambda u: geo[u]["surface"], lambda u: geo[u]["floor"], uid, objs[uid][20, 0], 30.0)


def test_receiver_bottom_reads_whole_mesh():
    lifted = dict(BOARD, surface=BOARD["surface"].copy())
    lifted["surface"][lifted["surface"][:, 2] < 0.01, 2] = 0.0002  # the samples miss the true bottom by 0.2 mm
    rows = _recv({"board": lifted, "phone": PHONE}, {"board": (0, 0, 0), "phone": (0, 0, 0.02)})
    assert rows == [("board", 0.0)]


def test_height_test_reads_the_surface_samples_not_the_floor_vertices():
    corners = {"surface": BOARD["floor"], "floor": BOARD["floor"]}  # no sample of the board under the phone
    assert _recv({"board": corners, "phone": PHONE}, {"board": (0, 0, 0), "phone": (0, 0, 0.02)}) == []


@pytest.mark.parametrize(
    ("recv", "top", "lift", "received"),
    [
        ("board", 0.02, 0.004, True),
        ("board", 0.02, 0.006, False),
        ("board", 0.02, 0.02, False),
        ("step", 0.01, 0.004, True),  # the step's box reaches 30 cm: only the heights under the phone decide
        ("step", 0.01, 0.006, False),
    ],
)
def test_gap_within_recv_gap_only(recv, top, lift, received):
    geo = {"board": BOARD, "step": STEP}
    rows = _recv({recv: geo[recv], "phone": PHONE}, {recv: (0, 0, 0), "phone": (0, 0, top + lift)})
    assert [v for v, _ in rows] == ([recv] if received else [])


@pytest.mark.parametrize(
    ("xy", "yaw", "received"),
    [
        ((0.19, 0.0), 0.0, True),
        ((0.21, 0.0), 0.0, False),  # overhangs the edge with its centre outside the hull
        ((0.10, 0.10), 45.0, True),
        ((0.15, 0.15), 45.0, False),  # board turned 45°: centre inside its XY box, outside its hull
    ],
)
def test_centre_must_lie_inside_the_receiver_hull(xy, yaw, received):
    # flush on the top face (gap 0) in every case, part of the phone over the board
    rows = _recv({"board": BOARD, "phone": PHONE}, {"board": (0, 0, 0), "phone": (*xy, 0.02)}, yaw={"board": yaw})
    assert [v for v, _ in rows] == (["board"] if received else [])


def test_container_receives_the_object_inside():
    bowl = _obj(
        ([-0.1, -0.1, 0.0], [0.1, 0.1, 0.005]),
        ([-0.1, -0.1, 0.005], [-0.095, 0.1, 0.08]),
        ([0.095, -0.1, 0.005], [0.1, 0.1, 0.08]),
        ([-0.095, -0.1, 0.005], [0.095, -0.095, 0.08]),
        ([-0.095, 0.095, 0.005], [0.095, 0.1, 0.08]),
    )
    cube = _obj(([-0.02, -0.02, 0.0], [0.02, 0.02, 0.03]))
    assert _recv({"bowl": bowl, "cube": cube}, {"bowl": (0, 0, 0), "cube": (0, 0, 0.005)}, "cube") == [("bowl", 0.0)]


def test_no_centre_of_mass_condition():
    assert C.support_contact(PHONE["surface"] + (0, 0, 0.01), STEP["surface"])["above"]  # the removed condition
    assert _recv({"step": STEP, "phone": PHONE}, {"step": (0, 0, 0), "phone": (0, 0, 0.01)}) == [("step", 0.0)]
