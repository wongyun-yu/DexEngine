"""S-float-settle data statement (data/objects/float_settle): which free main objects rest above their support, the
settle and drop verdicts, and the supports other than the desk (CPU, synthetic trees)."""

from __future__ import annotations

import numpy as np
import pytest

from dexengine.data.objects import collider_shape as CS
from dexengine.data.objects import float_settle as FS
from dexengine.data.objects import scene_requirement as SR
from dexengine.data.objects import scene_rules as SC
from dexengine.data.objects import touch_gate as TG

T = 30


def _box(half) -> TG.Body:
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * np.asarray(half, float)
    return TG.Body([CS.Parts([v])])


def _q(cup_z: float, plate_z: float = 0.0) -> np.ndarray:
    q = np.zeros((T, 14))
    q[:, 2], q[:, 3] = cup_z, 1.0
    q[:, 9], q[:, 10] = plate_z, 1.0
    return q


def _ti(discs=()) -> dict:
    return {
        "dataset_name": "hot3d",
        "support": {"desk_z": 0.0, "desk_mode": "table"},
        "desk_support": {"discs": list(discs)},
        "objects": [
            {"name": "cup", "obj_id": "cup", "role": "main", "side": "right", "qpos_start": 0, "mesh_dir": "m"},
            {"name": "plate", "obj_id": "plate", "role": "extra", "qpos_start": 7, "mesh_dir": "m"},
        ],
    }


def _NO_ROWS() -> list:
    """The clip rows provider of a statement whose scene bodies the `scene` fixture fakes."""
    return []


@pytest.fixture
def scene(monkeypatch):
    """A cup (half height 5 cm) first touched at frame 12; a plate (half height 1 cm) the training scene spawns with
    the objects cell `cell` (None = no plate)."""
    from dexengine.placement.coords import contacts as CT
    from dexengine.placement.coords import core as C

    monkeypatch.setattr(CT, "base_mismatch", lambda tree, ti: state["stale"])
    state = {"touch": 12, "cell": None, "start_held": False, "stale": {}}
    monkeypatch.setattr(FS, "_first_touch", lambda tree, ti, q, name: (state["touch"], state["touch"]))
    monkeypatch.setattr(C, "start_gaps", lambda tree, ti, q, aerial: [{"obj": "cup", "held": state["start_held"]}])
    monkeypatch.setattr(TG, "body_of", lambda ds, o, tree: (_box((0.03, 0.03, 0.05)), ""))

    def bodies(tree, ti, q, clip_rows):
        if state["cell"] is None:
            return {}
        return {"plate": (_box((0.08, 0.08, 0.01)), state["cell"], True, np.tile(_pose(q, 7)[0], (T, 1, 1)))}

    monkeypatch.setattr(FS, "scene_bodies", bodies)
    return state


def _pose(q, st):
    from dexengine.data.objects import extra_slots_geom as XG

    return XG.pose_mats(q, st)


def test_float_within_cap_settles_onto_the_desk(scene):
    st = FS.statement("tree", _ti(), _q(0.05 + 0.012), _NO_ROWS)
    r = st["cup"]
    assert r["verdict"] == "settle" and r["support"] == "desk" and r["rest_frame"] == 11
    assert r["gap_m"] == pytest.approx(0.012) and r["desk_gap_m"] == pytest.approx(0.012)
    assert FS.dropped(st) is None


def _moving(q: np.ndarray, col: int, dz: float, upto: int = 12) -> np.ndarray:
    """q with the body at column `col` raised linearly by dz from frame 0 to frame `upto` (and held after)."""
    q = q.copy()
    q[:, col + 2] += dz * np.minimum(np.arange(T), upto) / upto
    return q


def test_still_float_above_cap_gets_a_support_not_dropped(scene):
    """A gap > FLOAT_SETTLE_MAX_M with the object still before its first touch: support (no lowering, no drop)."""
    st = FS.statement("tree", _ti(), _q(0.05 + SR.FLOAT_SETTLE_MAX_M + 0.001), _NO_ROWS)
    assert st["cup"]["verdict"] == "support" and st["cup"]["moved_m"] == 0.0
    assert FS.dropped(st) is None


def test_still_high_float_held_over_the_start_window_is_held(scene):
    """The start-support rule calls it held (no support at bake): held, not support and not dropped."""
    scene["start_held"] = True
    st = FS.statement("tree", _ti(), _q(0.05 + 0.1), _NO_ROWS)
    assert st["cup"]["verdict"] == "held" and st["cup"]["start_held"] is True and FS.dropped(st) is None


def test_own_disc_supports_outside_its_measured_window(scene):
    """Rule D40: a disc stands from reset to the end — a stamp window that ends before the rest frame still supports.
    Negative of the old reading (window-limited), which called this a float above the cap."""
    disc = _disc(start=0, end=5, h=0.15)
    r = FS.statement("tree", _ti([disc]), _q(0.05 + 0.15), _NO_ROWS)["cup"]
    assert r["verdict"] == "rest" and r["support"] == "disc"


def test_own_disc_elsewhere_is_not_a_support(scene):
    """Negative: the cup's own disc 30 cm away (another rest place) does not carry it here (coords.disc_covers)."""
    r = FS.statement("tree", _ti([_disc(0.3, start=0, end=5, h=0.012)]), _q(0.05 + 0.012), _NO_ROWS)["cup"]
    assert r["verdict"] == "settle" and r["support"] == "desk"


def test_float_moving_with_another_object_is_carried(scene, monkeypatch):
    """Above the cap and moving with the plate (relative pose constant): carried, the carrier in `support`."""
    q = _moving(_moving(_q(0.06 + 0.05, plate_z=0.05), 0, 0.05), 7, 0.05)  # the cup stands on the plate top
    _geom(monkeypatch, {"cup": (0.03, 0.03, 0.05), "plate": (0.03, 0.03, 0.01)})
    st = FS.statement("tree", _ti(), q, _NO_ROWS)
    assert st["cup"]["verdict"] == "carried" and st["cup"]["support"] == "plate"
    assert FS.dropped(st) is None


def test_moving_together_without_touch_is_not_carried(scene, monkeypatch):
    """Negative: the same motion with the plate 19 cm below the cup (no touch at the anchor, carry_of): drop."""
    q = _moving(_moving(_q(0.05 + 0.25, plate_z=0.05), 0, 0.05), 7, 0.05)
    _geom(monkeypatch, {"cup": (0.03, 0.03, 0.05), "plate": (0.03, 0.03, 0.01)})
    assert FS.statement("tree", _ti(), q, _NO_ROWS)["cup"]["verdict"] == "drop"


def _geom(monkeypatch, half: dict):
    """extra_slots_geom.geom's placed vertices for box objects (half extents per name) at their qpos pose."""
    from dexengine.data.objects import extra_slots_geom as XG

    def placed_of(tree, ti, q):
        def placed(o, f):
            h = np.asarray(half[o["name"]], float)
            v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * h
            return v + q[f, int(o["qpos_start"]) : int(o["qpos_start"]) + 3]

        return placed, None

    monkeypatch.setattr(XG, "geom", placed_of)


def test_float_moving_on_its_own_above_cap_is_dropped(scene):
    """Negative: above the cap, moving with no hand near and no carrier — the clip is dropped with the reason."""
    st = FS.statement("tree", _ti(), _moving(_q(0.05 + 0.1), 0, 0.05), _NO_ROWS)
    assert st["cup"]["verdict"] == "drop"
    word, why = FS.dropped(st)
    assert word == FS.DROP_WORD and "no hand near and no carrier" in why and "> cap 20 mm" in why


def test_resting_object_is_not_a_float(scene):
    assert FS.statement("tree", _ti(), _q(0.05 + 0.002), _NO_ROWS)["cup"]["verdict"] == "rest"


def test_object_on_a_collision_on_body_rests_on_it(scene):
    """The case a desk-only reading gets wrong: the cup stands on the plate (12 mm above the desk) — it rests."""
    scene["cell"] = "ON"
    r = FS.statement("tree", _ti(), _q(0.02 + 0.05, plate_z=0.01), _NO_ROWS)["cup"]
    assert r["verdict"] == "rest" and r["support"] == "plate" and r["desk_gap_m"] == pytest.approx(0.02)


def test_float_above_another_body_is_reported_not_settled(scene):
    scene["cell"] = "ON"
    r = FS.statement("tree", _ti(), _q(0.02 + 0.05 + 0.01, plate_z=0.01), _NO_ROWS)["cup"]
    assert r["verdict"] == "over_body" and r["support"] == "plate"


def test_object_on_a_collision_off_body(scene):
    scene["cell"] = "OFF"
    assert FS.statement("tree", _ti(), _q(0.02 + 0.05, plate_z=0.01), _NO_ROWS)["cup"]["verdict"] == "over_off_body"


def _disc(dx: float = 0.0, **kw) -> dict:
    """An own disc row of the cup, centred under it (scene xy of the capture origin) shifted by dx in x, radius 5 cm."""
    from dexengine.placement.coords import core as C

    x, y = C.capture_to_scene(np.zeros((1, 2)))[0]
    return {"obj": "cup", "x": float(x) + dx, "y": float(y), "r": 0.05} | kw


def test_own_disc_is_a_support(scene):
    disc = _disc(start=0, end=T - 1, h=0.012)
    r = FS.statement("tree", _ti([disc]), _q(0.05 + 0.012), _NO_ROWS)["cup"]
    assert r["verdict"] == "rest" and r["support"] == "disc"


def test_touched_from_the_start_has_no_rest_window(scene):
    scene["touch"] = 0
    assert FS.statement("tree", _ti(), _q(0.08), _NO_ROWS)["cup"]["verdict"] == "held"


def test_rule_off_gives_no_statement(scene, monkeypatch):
    monkeypatch.setattr(SC, "on", lambda rid, ds: rid != FS.RULE_ID)
    assert FS.statement("tree", _ti(), _q(0.08), _NO_ROWS) == {}


def test_marker_counts_verdicts(scene):
    st = FS.statement("tree", _ti(), _q(0.05 + 0.012), _NO_ROWS)
    line = FS.marker("t1", st)
    assert line.startswith("[FLOAT_SETTLE] t1 mains 1 ") and "'settle': 1" in line and "largest 12.0 mm" in line


def test_exclusion_row(scene):
    """The exclusion step's float row: DROP with the reason when a main object above the cap moves on its own."""
    keep = FS.stage_row("/t/clip_a", _ti(), _q(0.05 + 0.012), [], "sha", "v", ("KEEP", "DROP"))
    assert keep["status"] == "KEEP" and keep["side"] == FS.ROW_SIDE
    assert keep["rows"].startswith("cup:settle:0.012:desk:12")
    drop = FS.stage_row("/t/clip_a", _ti(), _moving(_q(0.05 + 0.03), 0, 0.03), [], "sha", "v", ("KEEP", "DROP"))
    assert drop["status"] == "DROP" and drop["clip"] == "clip_a" and "> cap 20 mm" in drop["why"]


def test_first_touch_is_mapped_from_base_to_tree_frames(monkeypatch):
    """The contact reads base frames; the statement is in tree frames (f0 pad and retime through coords._base_to_tree).
    Never touched: the clip length (the whole clip is the rest window)."""
    from dexengine.placement.coords import contacts as CT
    from dexengine.placement.coords import core as C

    monkeypatch.setattr(C, "_base_to_tree", lambda tree, ti, T: lambda b: min(T - 1, b + 3))
    monkeypatch.setattr(C, "_base_T", lambda ti: 20)
    monkeypatch.setattr(C, "frame_toucher", lambda tree, ti, q, name, b2t: lambda b: b >= 10 and name == "cup")
    monkeypatch.setattr(CT, "frame_near", lambda tree, ti, name: lambda b: b >= 6 and name == "mug")
    q = np.zeros((25, 7))
    assert FS._first_touch("t", {}, q, "cup") == (13, 10)
    assert FS._first_touch("t", {}, q, "mug") == (9, 6)  # a hand near before its contact flags: the earlier frame
    assert FS._first_touch("t", {}, q, "plate") == (25, None)


def test_frame_near_reads_either_hand_and_refuses_a_foreign_column(monkeypatch):
    """The other hand within contact distance counts; a sidecar column under another obj_id raises."""
    from dexengine.placement.coords import contacts as CT

    near = np.full((6, 2), 0.1)
    near[3, 1] = 0.01  # the left hand reaches the cup at base frame 3
    cols = {"names": ["plate", "cup"], "ids": ["plate", "cup"], "thr": 0.02}
    monkeypatch.setattr(
        CT,
        "_all_sidecars",
        lambda tree, ti: {"right": cols | {"near_m": np.full((6, 2), 0.1)}, "left": cols | {"near_m": near}},
    )
    ti = _ti()
    f = CT.frame_near("t", ti, "cup")
    assert [b for b in range(6) if f(b)] == [3]
    ti["objects"][0]["obj_id"] = "cup_v2"
    with pytest.raises(ValueError, match="no column cup=cup_v2"):
        CT.frame_near("t", ti, "cup")


def test_object_stale_against_its_base_is_listed_for_regeneration(scene):
    """A tree whose main obj_id differs from its base's: stale_base, the exclusion row carries REGEN_WORD."""
    scene["stale"] = {"cup": ("cup_a", "cup_b")}
    st = FS.statement("tree", _ti(), _q(0.08), _NO_ROWS)
    assert st["cup"]["verdict"] == "stale_base" and FS.dropped(st) is None
    assert FS.regenerate(st) == (FS.REGEN_WORD, "cup cup_a in the tree vs cup_b in its base")
    row = FS.stage_row("/t/clip_a", _ti(), _q(0.08), [], "sha", "v", ("KEEP", "DROP"))
    assert row["status"] == "DROP" and "cup_b in its base" in row["why"]


def test_base_mismatch_reads_the_base_tree(tmp_path, monkeypatch):
    """coords.contacts.base_mismatch: same-named objects compared by obj_id; equal or absent ones are not listed."""
    import json

    from dexengine.placement.coords import contacts as CT

    (tmp_path / "base").mkdir()
    objs = [{"name": "cup", "obj_id": "cup_b"}, {"name": "plate", "obj_id": "plate"}]
    (tmp_path / "base" / "task_info.json").write_text(json.dumps({"objects": objs}))
    monkeypatch.setattr(CT, "sidecar_base_tree", lambda tree, ti: tmp_path / "base")
    ti = {"objects": [{"name": "cup", "obj_id": "cup_a"}, {"name": "plate", "obj_id": "plate"}, {"name": "x"}]}
    assert CT.base_mismatch("t", ti) == {"cup": ("cup_a", "cup_b")}
