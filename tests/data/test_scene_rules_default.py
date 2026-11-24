"""Scene rules on by default (CPU, synthetic trees): S-touch-gate (touched statics collide with the main objects unless
the recording passes through them), S-envelope (visual-only statics near the recorded sweep stand and collide, slot cap
drops them first), the convex-part reading they share, and the per-dataset opt-out."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from dexengine.core import datasets as RR
from dexengine.data.objects import collider_shape as CS
from dexengine.data.objects import convex_depth as CD
from dexengine.data.objects import envelope as EV
from dexengine.data.objects import extra_slots as XS
from dexengine.data.objects import extra_slots_geom as XG
from dexengine.data.objects import recv_surface as RS
from dexengine.data.objects import scene_requirement as SR
from dexengine.data.objects import scene_rules as SC
from dexengine.data.objects import touch_gate as TG
from tests.imports import own_module


def _box(c, half):
    """The 8 corners of an axis-aligned box (one convex part)."""
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    return v * np.asarray(half, float) + np.asarray(c, float)


def _grid(half, n=7):
    """Dense box-surface vertices (what `placed` returns for a mesh)."""
    g = np.linspace(-1, 1, n)
    a, b = np.meshgrid(g, g)
    f = [
        np.c_[a.ravel(), b.ravel(), np.full(a.size, s)][:, p]
        for s in (-1, 1)
        for p in ([0, 1, 2], [0, 2, 1], [2, 0, 1])
    ]
    return np.unique(np.concatenate(f), axis=0) * np.asarray(half, float)


HALF = {"cup": (0.03, 0.03, 0.05), "plate": (0.08, 0.08, 0.01), "vase": (0.03, 0.03, 0.06), "far": (0.03, 0.03, 0.06)}


def _tree(monkeypatch, plate_z: float, vase_y: float, parts=None):
    """A cup (main, right hand, bottom at z = 0) sweeps along x at y = 0 above a hand-touched plate (top at
    plate_z + 1 cm) and past an untouched vase at y = vase_y; a far untouched object 0.5 m away."""
    T = 41
    q = np.zeros((T, 28))
    q[:, 0] = np.linspace(-0.2, 0.2, T)
    q[:, 2] = 0.05
    poses = {"cup": 0, "plate": 7, "vase": 14, "far": 21}
    q[:, 9], q[:, 15], q[:, 16], q[:, 22] = plate_z, vase_y, 0.06, 0.5
    for s in poses.values():
        q[:, s + 3] = 1.0
    ti = {
        "dataset_name": "hot3d",
        "idle_sides": [],
        "case": {"contact_frac": {"plate": {"left": 0.3}, "vase": {}, "far": {}}, "grasp_s": {}},
        "objects": [{"name": "cup", "obj_id": "cup", "role": "main", "side": "right", "qpos_start": 0, "mesh_dir": "m"}]
        + [
            {"name": n, "obj_id": n, "role": "extra", "qpos_start": poses[n], "mesh_dir": "m"}
            for n in poses
            if n != "cup"
        ],
    }

    def geom(tree_dir, ti_, q_):
        def placed(o, f):
            T_ = XG.pose_mats(q_[f : f + 1], int(o["qpos_start"]))[0]
            return _grid(HALF[o["name"]]) @ T_[:3, :3].T + T_[:3, 3]

        return placed, lambda Vu, Vl: False

    monkeypatch.setattr(XG, "geom", geom)
    monkeypatch.setattr(RS, "near_placer", _grid_near)
    monkeypatch.setattr(XG, "below_work_surface", lambda *a: False)
    monkeypatch.setattr(XS, "_place_drop", lambda *a: "")
    shapes = parts or {n: [_box((0, 0, 0), h)] for n, h in HALF.items()}
    monkeypatch.setattr(TG, "parts_of", lambda ds, o: shapes.get(o["name"]))
    monkeypatch.setattr(CS, "triangles", lambda ds, oid: _tri_box(HALF[oid]))  # a pinned still body's true shape
    TG.MEMO.clear()
    return "/nonexistent/tree", ti, q


def _grid_near(tree_dir, ti_, q_, placed):
    """recv_surface.near_placer for the synthetic boxes: their grid points are the surface samples already (no
    mesh file), inside = the hull of those points (arrives' own default)."""

    def near(o, f):
        return placed(o, f)

    near.inside = lambda o, f: None
    return near


def _tri_box(half):
    """A closed box as triangles (what `collider_shape.triangles` reads from a pinned body's visual mesh)."""
    import trimesh  # noqa: PLC0415

    m = trimesh.creation.box(extents=2 * np.asarray(half, float))
    return CS.Triangles(np.asarray(m.vertices), np.asarray(m.faces))


def _rows(tree, ti, q):
    return {r["name"]: r for r in XS.clip_rows(tree, ti, q)}


def _opt_out(monkeypatch, *rules):
    p = RR.DATASET_PROFILE["hot3d"]
    off = dict.fromkeys(rules, "test: the rows before the rule")
    monkeypatch.setitem(RR.DATASET_PROFILE, "hot3d", dataclasses.replace(p, scene_rules_off=off))


def test_convex_depth_is_the_exit_distance_of_a_part():
    sh = CD.shape([_box((0, 0, 0), 0.04)])
    assert CD.depth(np.array([[0.0, 0.0, 0.037]]), sh) == pytest.approx(0.003, abs=1e-12)
    assert CD.depth(np.array([[0.0, 0.0, 0.041], [0.5, 0.0, 0.0]]), sh) == 0.0
    assert len(sh["points"]) > 8  # vertices, edge midpoints, face centres


def test_a_touched_static_the_recording_clears_collides_with_the_main_object(monkeypatch):
    """Negative on the old rule: the touched plate kept objects OFF whatever the recording did."""
    tree, ti, q = _tree(monkeypatch, plate_z=-0.0105, vase_y=0.5)  # cup bottom 0.5 mm above the plate top
    r = _rows(tree, ti, q)["plate"]
    assert r["cls"] == XS.STATIC_SPAWN and r["kind"] == XS.KIND_TOUCHED_SOLID
    assert r["touch_gate"]["on"] and r["touch_gate"]["pairs"]["cup"]["depth_m"] == 0.0
    _opt_out(monkeypatch, TG.RULE_ID)
    assert _rows(tree, ti, q)["plate"]["kind"] == XS.KIND_TOUCHED  # opted out = the rows before


def test_a_recorded_overlap_deeper_than_the_true_shape_value_drops_the_clip(monkeypatch):
    """Negative on the old rule: a 7 mm pass-through kept the pair OFF and the clip; now the clip is dropped."""
    tree, ti, q = _tree(monkeypatch, plate_z=-0.003, vase_y=0.5)  # plate top 7 mm into the cup's bottom
    rows = _rows(tree, ti, q)
    r = rows["plate"]
    assert r["kind"] == XS.KIND_TOUCHED and r["touch_gate"]["why"] == "true-shape overlap"
    assert r["touch_gate"]["shape"] == CS.TRIANGLES
    assert r["touch_gate"]["pairs"]["cup"]["depth_m"] == pytest.approx(0.007, abs=1e-4)
    assert r["overlap_drop"] == {"partner": "cup", "depth_m": pytest.approx(0.007, abs=1e-4)}
    assert SR.TRUE_SHAPE_OVERLAP_M < 0.007
    assert CS.overlap_drop(list(rows.values()))[0] == CS.OVERLAP_DROP_WORD
    tree, ti, q = _tree(monkeypatch, plate_z=-0.007, vase_y=0.5)  # 3 mm: within the retargeting contact error
    rows = _rows(tree, ti, q)
    assert rows["plate"]["kind"] == XS.KIND_TOUCHED_SOLID and "overlap_drop" not in rows["plate"]
    assert CS.overlap_drop(list(rows.values())) is None


def test_an_unmeasured_pair_stays_off(monkeypatch):
    tree, ti, q = _tree(monkeypatch, plate_z=-0.02, vase_y=0.5, parts={"plate": [_box((0, 0, 0), HALF["plate"])]})
    r = _rows(tree, ti, q)["plate"]
    assert r["kind"] == XS.KIND_TOUCHED and r["touch_gate"]["why"] == "unmeasured"


def test_a_still_extra_near_the_sweep_stands_as_an_envelope_static(monkeypatch):
    """Negative on the old rule: the untouched vase 1 cm beside the cup's path was visual only."""
    tree, ti, q = _tree(monkeypatch, plate_z=-0.03, vase_y=0.07)  # 1 cm gap between cup and vase faces
    rows = _rows(tree, ti, q)
    v = rows["vase"]
    assert v["cls"] == XS.STATIC_SPAWN and v["why"] == EV.WHY and v["kind"] == XS.KIND_ENVELOPE and v["static"]
    assert v["envelope"]["by"] == "cup" and v["envelope"]["gap_m"] == pytest.approx(0.01, abs=1e-3)
    assert rows["far"]["cls"] == XS.VISUAL_ONLY and "envelope" not in rows["far"]
    assert CS.row_shape("hot3d", v) == CS.TRIANGLES  # a pinned envelope body stands as its visual mesh
    _opt_out(monkeypatch, EV.RULE_ID)
    assert _rows(tree, ti, q)["vase"]["cls"] == XS.VISUAL_ONLY


def test_beyond_the_margin_or_dropped_by_the_placement_rule_stays_visual(monkeypatch):
    tree, ti, q = _tree(monkeypatch, plate_z=-0.03, vase_y=0.06 + SR.ENVELOPE_M + 0.005)
    assert _rows(tree, ti, q)["vase"]["cls"] == XS.VISUAL_ONLY
    tree, ti, q = _tree(monkeypatch, plate_z=-0.03, vase_y=0.07)
    monkeypatch.setattr(XS, "_place_drop", lambda tree_dir, ti_, q_, o: "off_desk" if o["name"] == "vase" else "")
    v = _rows(tree, ti, q)["vase"]
    assert v["cls"] == XS.VISUAL_ONLY and v["envelope"]["drop"] == "off_desk"


def test_a_mover_deep_in_an_envelope_static_drops_the_clip(monkeypatch):
    """The cup's side passes 8 mm into the vase (true shape): the vase stands and the clip is dropped; 2 mm is kept."""
    tree, ti, q = _tree(monkeypatch, plate_z=-0.03, vase_y=0.052)
    v = _rows(tree, ti, q)["vase"]
    assert v["kind"] == XS.KIND_ENVELOPE and v["envelope"]["depth_m"] == pytest.approx(0.008, abs=1e-4)
    assert v["overlap_drop"] == {"partner": "cup", "depth_m": pytest.approx(0.008, abs=1e-4)}
    tree, ti, q = _tree(monkeypatch, plate_z=-0.03, vase_y=0.058)
    v = _rows(tree, ti, q)["vase"]
    assert v["kind"] == XS.KIND_ENVELOPE and "overlap_drop" not in v and v["envelope"]["depth_m"] <= 0.0021


def test_slot_cap_drops_envelope_rows_farthest_first():
    rows = [{"name": f"s{i}", "cls": XS.SLOT} for i in range(8)]
    rows += [
        {
            "name": n,
            "cls": XS.STATIC_SPAWN,
            "why": EV.WHY,
            "still": "still",
            "kind": "envelope",
            "static": True,
            "envelope": {"gap_m": g, "cap": False},
        }
        for n, g in (("near", 0.01), ("far", 0.04), ("mid", 0.02))
    ]
    assert EV.cap(rows, 9) == ["far", "mid"]
    by = {r["name"]: r for r in rows}
    assert by["near"]["cls"] == XS.STATIC_SPAWN and by["far"]["cls"] == XS.VISUAL_ONLY and by["far"]["envelope"]["cap"]
    assert by["far"]["why"] == "still" and "kind" not in by["far"]


def test_rule_version_and_kinds_follow_the_opt_out(monkeypatch):
    assert XS.rule_version(["hot3d"]) == "+".join([XS.RULE, *SC.VERSIONS.values()])
    assert SC.kinds_on("hot3d") == XS.SCOPED_KINDS
    _opt_out(monkeypatch, *SC.VERSIONS)
    assert XS.rule_version(["hot3d"]) == XS.RULE and SC.kinds_on("hot3d") == ()
    assert XS.rule_version(["hot3d", "taco"]) == "+".join([XS.RULE, *SC.VERSIONS.values()])


def test_new_kinds_collide_with_objects_and_with_the_robot_under_full_physics(run_dataset):
    cp = own_module("dexengine.rl.tasks.scene.collision_presets")
    run_dataset(robot_static=False)
    for k in (XS.KIND_TOUCHED_SOLID, XS.KIND_ENVELOPE):
        assert cp.extra_collide(k, full=False) == {"hand": "OFF", "arm": "OFF", "objects": "ON"}
        assert cp.extra_collide(k, full=False, ds="taco") == {"hand": "ON", "arm": "ON", "objects": "ON"}
        assert cp.extra_collide(k, full=True) == {"hand": "ON", "arm": "ON", "objects": "ON"} and cp.extra_has_collider(
            k
        )
    assert cp.extra_collide(XS.KIND_TOUCHED, full=True) == {"hand": "ON", "arm": "ON", "objects": "OFF"}
    assert cp.ALL_OBJ_KEEP_OFF_KINDS == ("touched",)
    assert XS.spawn_key([{"cls": XS.STATIC_SPAWN, "kind": XS.KIND_ENVELOPE}]) == "0/1/0"
