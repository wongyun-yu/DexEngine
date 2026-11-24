"""Carry classifier (CPU): synthetic convex parts (a rider between close walls is inserted, a rider on a plate is
resting, clearance ignores the recorded overlap) and six reviewed cases from the trees (golden/carry_cases.npz: the
cooked parts and the carry-interval poses) — bread on a plate, meat on a plate, a spoon lying in a bowl are resting; a
toothpaste cap on its tube, a test tube in a clamp, a lamp cap are inserted."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dexengine.data.objects import carry_class as CC


def _box(c, half):
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    return v * np.asarray(half, float) + np.asarray(c, float)


RIDER = [_box((0, 0, 0), (0.02, 0.02, 0.02))]


def _measure(monkeypatch, carrier_parts, tilt_deg=0.0):
    """Carrier and rider frames at the origin, tilting together about x over the interval (tilt_deg)."""
    T = 11
    q = np.zeros((T, 14))
    ang = np.radians(np.linspace(0.0, tilt_deg, T)) / 2.0
    q[:, 3], q[:, 4] = np.cos(ang), np.sin(ang)  # carrier wxyz about x
    q[:, 10], q[:, 11] = np.cos(ang), np.sin(ang)  # the rider keeps its relative pose
    ti = {
        "dataset_name": "hot3d",
        "objects": [
            {"name": "carrier", "obj_id": "c", "qpos_start": 0},
            {"name": "rider", "obj_id": "r", "qpos_start": 7},
        ],
    }
    monkeypatch.setattr(CC, "parts", lambda ds, o, row=None: carrier_parts if o["name"] == "carrier" else RIDER)
    return CC.measure(ti, q, "rider", {"carrier": "carrier", "start": 0, "end": T - 1})


def test_a_rider_between_close_walls_is_inserted(monkeypatch):
    walls = [_box((0.0, 0.0225, 0.0), (0.03, 0.0005, 0.03)), _box((0.0, -0.0225, 0.0), (0.03, 0.0005, 0.03))]
    walls += [_box((0.0225, 0.0, 0.0), (0.0005, 0.03, 0.03)), _box((-0.0225, 0.0, 0.0), (0.0005, 0.03, 0.03))]
    m = _measure(monkeypatch, walls)
    assert m["label"] == "inserted" and max(m["clearance_m"].values()) <= CC.CLEAR_M
    assert m["inside_frac"] > 0.5  # the rider sits in the walls' cavity


def test_a_rider_on_a_plate_is_resting(monkeypatch):
    m = _measure(monkeypatch, [_box((0.0, 0.0, -0.025), (0.08, 0.08, 0.005))])
    assert m["label"] == "resting" and m["support"]["supported"] and m["support"]["gap_m"] == 0.0
    assert min(m["clearance_m"].values()) > CC.CLEAR_MAX_M


def test_a_recorded_overlap_does_not_block_every_direction(monkeypatch):
    """The plate top 3 mm into the rider's bottom: the old reading (overall depth grows) called every way blocked."""
    m = _measure(monkeypatch, [_box((0.0, 0.0, -0.022), (0.08, 0.08, 0.005))])
    assert m["label"] == "resting" and min(m["clearance_m"].values()) > CC.CLEAR_MAX_M
    assert m["overlap_m"] == pytest.approx(0.003, abs=1e-6)


def test_a_tilting_carrier_with_the_rider_still_holds_it(monkeypatch):
    m = _measure(monkeypatch, [_box((0.0, 0.0, -0.025), (0.08, 0.08, 0.005))], tilt_deg=60.0)
    assert m["tilt_deg"] == pytest.approx(60.0, abs=0.5) and m["label"] == "inserted"


def test_a_rider_held_against_a_post_with_no_floor_is_inserted(monkeypatch):
    """Nothing under it, yet it rides: the carrier holds it from the side."""
    m = _measure(monkeypatch, [_box((0.0205, 0.0, 0.0), (0.0005, 0.03, 0.03))])
    assert m["label"] == "inserted" and not m["support"]["supported"] and m["support"]["walls"] > 0


def test_a_rider_touching_nothing_is_ambiguous(monkeypatch):
    m = _measure(monkeypatch, [_box((0.1, 0.0, 0.0), (0.005, 0.005, 0.005))])
    assert m["label"] == "ambiguous" and m["support"]["walls"] == 0


GOLDEN = Path(__file__).parent / "golden" / "carry_cases.npz"


def _case(z, name):
    pr = [z[k].astype(float) for k in sorted((k for k in z.files if k.startswith(f"{name}/rider/")), key=_idx)]
    pc = [z[k].astype(float) for k in sorted((k for k in z.files if k.startswith(f"{name}/carrier/")), key=_idx)]
    return pr, pc, z[f"{name}/Tc"].astype(float), z[f"{name}/Tr"].astype(float), str(z[f"{name}/want"])


def _idx(k):
    return int(k.rsplit("/", 1)[1])


REVIEWED = {  # geometry only (measure_bodies); the toothpaste cap is routed to `articulation` before it (measure)
    "bread_plate": "resting",
    "meat_plate": "resting",
    "spoon_bowl": "resting",
    "toothpaste_cap": "inserted",
    "tube_clamp": "inserted",
    "lamp_cap": "inserted",
}


@pytest.mark.parametrize("name", sorted(REVIEWED))
def test_reviewed_cases(name):
    """Negative on the first reading: bread, meat and spoon read inserted (sunk riders probing into part seams), the
    toothpaste cap and the clamped tube read ambiguous (no support, sparse points)."""
    with np.load(GOLDEN) as z:
        pr, pc, Tc, Tr, stored = _case(z, name)
    assert stored == REVIEWED[name]
    m = CC.measure_bodies(pr, pc, Tc, Tr)
    assert m["label"] == REVIEWED[name], (name, m)


def _part_tree(monkeypatch, tree: dict) -> None:
    from dexengine.data.oakink2 import parts as OP

    rev = {c: p for p, cs in tree.items() for c in cs}
    monkeypatch.setattr(OP, "_load", lambda: (tree, rev, {r: {"obj_name": r} for r in rev}, set(tree)))


def _ti(cap: str, body: str) -> dict:
    grp = {"group": "O02@0003@00006"}
    return {
        "dataset_name": "oakink2",
        "objects": [
            {"name": "cap", "obj_id": cap, "qpos_start": 0, "articulation_group": grp},
            {"name": "body", "obj_id": body, "qpos_start": 7, "articulation_group": grp},
        ],
    }


def test_a_hinged_part_of_the_same_object_is_not_a_carry_case(monkeypatch):
    """The toothpaste cap rides its tube only because D1-g has not joined them yet: label articulation, no geometry."""
    _part_tree(monkeypatch, {"O02@0003@00006": ["O02@0003@00004", "O02@0003@00005"]})
    monkeypatch.setattr(CC, "parts", lambda *_a: pytest.fail("a joined pair must not be measured"))
    m = CC.measure(_ti("left_o02_0003_00004", "right_o02_0003_00005"), None, "cap", {"carrier": "body"})
    assert m["label"] == "articulation" and m["same_group"]["kind"] == "revolute"


def test_a_free_part_of_the_same_object_keeps_the_geometry_label(monkeypatch):
    _part_tree(monkeypatch, {"O02@0206@00003": ["O02@0206@00001", "O02@0206@00002"]})
    monkeypatch.setattr(CC, "parts", lambda *_a: None)
    m = CC.measure(_ti("raw_o02_0206_00001", "raw_o02_0206_00002"), None, "cap", {"carrier": "body"})
    assert m["label"] == "ambiguous" and m["same_group"] == {"group": "O02@0003@00006", "kind": "free", "merges": False}
