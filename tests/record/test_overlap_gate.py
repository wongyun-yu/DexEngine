"""Rule S-prox (CPU): the penetration trigger catches a moving object passing through a visual-only static object,
keeps one that only grazes it, reads a concave static's cavity as outside, and is on for every dataset unless it opts
out. Depth = plane distance to exit of the static's cooked convex parts (`data.objects.convex_depth`); the replaced
surface-sample + normal reading is kept here only as the negative reference (`_old_penetration`)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace as NS

import numpy as np
import pytest

from dexengine.core import datasets as RR
from dexengine.record.scene import overlap as OV
from tests.imports import own_module


def _cube(c=(0.0, 0.0, 0.0), half=0.04, n=9):
    """Surface samples of an axis-aligned cube and their outward normals."""
    g = np.linspace(-1, 1, n)
    a, b = np.meshgrid(g, g)
    P, N = [], []
    for ax in range(3):
        for s in (-1.0, 1.0):
            q = np.zeros((a.size, 3))
            q[:, ax] = s
            q[:, [i for i in range(3) if i != ax]] = np.c_[a.ravel(), b.ravel()]
            P.append(q)
            N.append(np.tile(np.eye(3)[ax] * s, (a.size, 1)))
    return np.concatenate(P) * half + np.asarray(c, float), np.concatenate(N)


def _box_parts(c=(0.0, 0.0, 0.0), half=0.04):
    """One convex part: the 8 corners of an axis-aligned box (half = scalar or (hx, hy, hz))."""
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * half + np.asarray(c, float)
    return [v]


def _solids(R=None, c=None, parts=None):
    """A visual-only 'vase' (8 cm cube, one convex part) at the render pose world = R p + c."""
    R = np.eye(3) if R is None else R
    c = np.zeros(3) if c is None else c
    return OV.solids_of(["vase"], {"vase": parts or _box_parts()}, {"vase": (R, c)}, np.zeros(3))


def _old_solid(B, N):
    """The replaced reading (E6's signed distance: nearest surface sample + its outward normal, body box) — negative
    reference only."""
    from scipy.spatial import cKDTree

    return cKDTree(B), B, N, B.min(0), B.max(0)


def _old_penetration(W, solid) -> float:
    tr, B, N, lo, hi = solid
    Q = np.asarray(W, float)
    Q = Q[np.all((lo <= Q) & (hi >= Q), axis=1)]
    if not len(Q):
        return 0.0
    d, i = tr.query(Q)
    inside = np.einsum("ij,ij->i", Q - B[i], N[i]) < 0
    return float(d[inside].max()) if bool(inside.any()) else 0.0


def _scan(path, half=0.04, n=9, pen=None):
    """A mover cube along `path` (one centre per render step) → the gated row. pen = (W → depth) replaces the kernel."""
    cands, solids = [{"name": "vase", "reason": ""}], _solids()
    best = {"vase": (0.0, -1)}
    for t, c in enumerate(path):
        W = _cube(c, half, n)[0]
        if pen is None:
            OV.pen_update(W, solids, best, t)
        else:
            p = pen(W)
            if p > best["vase"][0]:
                best["vase"] = (p, t)
    OV.gate_rows(cands, best)
    return cands[0]


def test_penetration_is_the_plane_distance_to_exit():
    s = _solids()["vase"]
    assert OV.penetration(np.array([[0.0, 0.0, 0.037]]), s) == pytest.approx(0.003, abs=1e-9)  # 3 mm under the top
    assert OV.penetration(np.array([[0.0, 0.0, 0.0]]), s) == pytest.approx(0.04, abs=1e-9)  # centre: no floor
    assert OV.penetration(np.array([[0.0, 0.0, 0.045]]), s) == 0.0  # 5 mm above it
    assert OV.penetration(np.array([[0.2, 0.0, 0.0]]), s) == 0.0  # outside the box


def test_a_point_just_outside_reads_zero():
    s = _solids()["vase"]
    e = 1e-4
    probes = [
        (0.04 + e, 0, 0),
        (0, -0.04 - e, 0.01),
        (0.01, 0.02, 0.04 + e),
        (0.04 + e, 0.04 + e, 0.04 + e),
        (0.04 + e, 0.039, -0.039),
    ]
    assert all(OV.penetration(np.array([p], float), s) == 0.0 for p in probes)
    assert OV.penetration(np.array([[0.04 - e, 0.0, 0.0]]), s) == pytest.approx(e, abs=1e-9)  # just inside


def test_the_render_pose_and_the_deepest_part_are_used():
    a = np.deg2rad(37.0)
    R = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    c = np.array([0.3, -0.2, 0.75])
    s = _solids(R, c)["vase"]
    body = np.array([[0.0, 0.035, 0.0], [0.0, 0.0, 0.039], [0.0, 0.0, 0.05]])
    assert OV.penetration(body[:1] @ R.T + c, s) == pytest.approx(0.005, abs=1e-9)
    assert OV.penetration(body[1:] @ R.T + c, s) == pytest.approx(0.001, abs=1e-9)
    # two parts (an L): the reading is the max over parts
    two = _box_parts((0.0, 0.0, 0.0), 0.04) + _box_parts((0.07, 0.0, 0.0), 0.03)
    s2 = _solids(parts=two)["vase"]
    assert OV.penetration(np.array([[0.07, 0.0, 0.0], [0.035, 0.0, 0.0]]), s2) == pytest.approx(0.03, abs=1e-9)


def test_trigger_catches_a_pass_through():
    row = _scan([(x, 0.0, 0.0) for x in np.linspace(-0.3, 0.3, 13)])  # straight through the vase
    # at the centre step the mover's samples lie on the vase surface (depth 0); one step (5 cm) off, its face sits
    # 1 cm past the vase centre: 3 cm from the far face
    assert row["reason"] == "rollout_overlap" and row["pen_obj_max_m"] == pytest.approx(0.03, abs=1e-9)
    assert row["pen_obj_step"] in (5, 7)


def test_an_engulfing_mover_is_caught_where_its_surface_crosses_the_static():
    # a 16 cm mover through the 8 cm vase: its walls cross the vase (caught); the fully engulfed step reads 0 (one
    # direction: mover samples inside the static — state_b5convex "Rule vs implementation differences")
    row = _scan([(x, 0.0, 0.0) for x in np.linspace(-0.3, 0.3, 13)], half=0.08, n=17)
    assert row["reason"] == "rollout_overlap" and row["pen_obj_max_m"] > OV.PEN_TRIGGER_M
    assert OV.penetration(_cube((0.0, 0.0, 0.0), 0.08, 17)[0], _solids()["vase"]) == 0.0


@pytest.mark.parametrize("off", [0.0, 0.0013, 0.0025, 0.005, 0.0077])
def test_a_3mm_graze_reads_3mm_at_any_sample_offset(off):
    # 3 mm into the vase's side face (mover face at y = 0.037), mover sample grid offset by `off` along x and z
    row = _scan([(x + off, 0.077, off) for x in np.linspace(-0.3, 0.3, 13)])
    assert row["reason"] == "" and row["pen_obj_max_m"] == pytest.approx(0.003, abs=1e-4)


def test_trigger_keeps_a_clear_pass_and_catches_8mm():
    deep = _scan([(x, 0.072, 0.0) for x in np.linspace(-0.3, 0.3, 13)])  # 8 mm into it
    assert deep["reason"] == "rollout_overlap" and deep["pen_obj_max_m"] == pytest.approx(0.008, abs=1e-4)
    clear = _scan([(x, 0.15, 0.0) for x in np.linspace(-0.3, 0.3, 13)])  # 7 cm beside it
    assert clear["reason"] == "" and clear["pen_obj_max_m"] == 0.0 and clear["pen_obj_step"] == -1


def test_surface_normals_play_no_part():
    """The reading takes the convex parts only: flipped or random surface-file normals cannot change it. The old
    reading (normals) changes with them."""
    B, N = _cube()
    rng = np.random.default_rng(0)
    flip = np.where(rng.random(len(N)) < 0.5, -1.0, 1.0)[:, None]
    probes = np.array([[0.0, 0.0, 0.032], [0.0, 0.032, 0.0], [-0.032, 0.0, 0.0], [0.0, 0.0, 0.05]])
    new = [OV.penetration(p[None], _solids()["vase"]) for p in probes]
    assert new == pytest.approx([0.008, 0.008, 0.008, 0.0], abs=1e-9)
    assert set(_solids()["vase"][0]) == {"hulls", "lo", "hi", "points"}  # convex parts, no surface normals
    old = [_old_penetration(p[None], _old_solid(B, N)) for p in probes]
    old_flip = [_old_penetration(p[None], _old_solid(B, N * flip)) for p in probes]
    old_rand = [_old_penetration(p[None], _old_solid(B, rng.normal(size=N.shape))) for p in probes]
    assert old != old_flip or old != old_rand


def test_negative_the_old_normals_reading_fails_the_graze_and_flipped_normals():
    """Negative test: the replaced reading fails the requirements above. (1) a 3 mm graze with the static's samples
    offset by 5 mm (1 cm grid) reads 5.8 mm and fires; (2) with all normals flipped, a deep pass-through reads 0."""
    B, N = _cube()
    Bo = B.copy()
    Bo[np.abs(np.abs(B[:, 1]) - 0.04) < 1e-12, 0] += 0.005  # side faces' grid offset 5 mm along x
    graze = _scan(
        [(x, 0.077, 0.0) for x in np.linspace(-0.3, 0.3, 13)], pen=lambda W: _old_penetration(W, _old_solid(Bo, N))
    )
    assert graze["reason"] == "rollout_overlap" and graze["pen_obj_max_m"] > 0.0055
    through = _scan(
        [(x, 0.0, 0.0) for x in np.linspace(-0.3, 0.3, 13)], pen=lambda W: _old_penetration(W, _old_solid(B, -N))
    )
    assert through["reason"] == "" and through["pen_obj_max_m"] == 0.0
    # the new reading on the same two cases
    graze_new = _scan([(x, 0.077, 0.0) for x in np.linspace(-0.3, 0.3, 13)])
    assert graze_new["pen_obj_max_m"] == pytest.approx(0.003, abs=1e-4)
    assert _scan([(x, 0.0, 0.0) for x in np.linspace(-0.3, 0.3, 13)])["reason"] == "rollout_overlap"


def test_a_concave_static_reads_its_cavity_as_outside():
    """A bowl-like static (floor + four walls, five convex parts): a mover sitting in the cavity, 1 mm off every inner
    face, reads 0 — the shape PhysX collides has no material there. Entering a wall reads its depth."""
    parts = (
        _box_parts((0.0, 0.0, -0.045), (0.05, 0.05, 0.005))
        + _box_parts((0.045, 0.0, 0.0), (0.005, 0.05, 0.04))
        + _box_parts((-0.045, 0.0, 0.0), (0.005, 0.05, 0.04))
        + _box_parts((0.0, 0.045, 0.0), (0.04, 0.005, 0.04))
        + _box_parts((0.0, -0.045, 0.0), (0.04, 0.005, 0.04))
    )
    s = _solids(parts=parts)["vase"]
    inside_cavity = _cube((0.0, 0.0, 0.0), 0.039, 9)[0]
    assert OV.penetration(inside_cavity, s) == 0.0
    assert OV.penetration(np.array([[0.043, 0.0, 0.0]]), s) == pytest.approx(0.003, abs=1e-9)  # 3 mm into a wall


def test_gate_keeps_the_first_reason():
    cands = [{"name": "vase", "reason": "rollout_overlap"}]
    OV.gate_rows(cands, {"vase": (0.0, -1)})
    assert cands[0]["reason"] == "rollout_overlap" and cands[0]["pen_obj_max_m"] == 0.0


def test_trigger_has_its_own_owner_and_the_robot_clearance_is_untouched():
    EX = own_module("dexengine.record.scene.extras")
    assert OV.PEN_TRIGGER_M == 0.005 and EX.ROLL_CLEAR_M == 0.01
    card = next(r for r in RR.RULES if r.id == OV.PROX_RULE)
    assert [ref for _, ref in card.thresholds] == ["dexengine/record/scene/overlap.py:PEN_TRIGGER_M"]


def test_movers_and_scope(monkeypatch):
    assert OV.is_main(NS(name="object_right")) and not OV.is_main(NS(name="extra_obj_3"))
    assert OV.is_slot(NS(name="extra_obj_3")) and not OV.is_slot(NS(name="object_right"))
    assert not OV.is_main(NS(name="robot")) and not OV.is_slot(NS(name="robot"))
    xe = own_module("dexengine.rl.tasks.scene.extra_slots")
    assert OV.SLOT_PREFIX == xe.PREFIX
    assert all(OV.on(d) for d in RR.ALL)
    p = RR.DATASET_PROFILE["hot3d"]
    off = {OV.PROX_RULE: "test: robot gate only"}
    monkeypatch.setitem(RR.DATASET_PROFILE, "hot3d", dataclasses.replace(p, scene_rules_off=off))
    assert not OV.on("hot3d") and OV.on("taco")


def test_scene_frame_is_the_transform_of_scene_points(monkeypatch):
    """The penetration body frame (`extras.scene_frame`) maps body samples exactly where `scene_points` puts them."""
    EX = own_module("dexengine.record.scene.extras")
    SD = own_module("dexengine.placement.scene.static_desk")
    B = _cube((0.01, -0.02, 0.03), 0.05)[0]
    monkeypatch.setattr(SD, "surface", lambda ds, o: B)
    pose = [0.3, -0.1, 0.8, 0.9, 0.1, -0.3, 0.2]
    pose[3:] = list(np.asarray(pose[3:]) / np.linalg.norm(pose[3:]))
    R, c = EX.scene_frame(pose, 0.12)
    assert np.allclose(B @ R.T + c, EX.scene_points("hot3d", {}, pose, 0.12), atol=1e-12)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
