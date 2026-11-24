"""S-true-shape readers (CPU): every reader of a still body measures the shape physics uses through the one dispatcher
(collider_shape) — a planted static receiver reaches recv_clearance and the hand-clear ruler as triangles, the
convex-only carry measurement refuses it, the judge fingerprint names a triangle support, and step 2f drops a clip on a
true-shape object overlap with its own reason word."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from dexengine.core import datasets as RR
from dexengine.data.objects import collider_shape as CS
from tests.data.gjk_oracle import GjkModel


def _cube(h, c=(0.0, 0.0, 0.0)):
    return np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)], float) + np.asarray(c, float)


def _tri_box(h):
    import trimesh  # noqa: PLC0415

    m = trimesh.creation.box(extents=(2 * h, 2 * h, 2 * h))
    return CS.Triangles(np.asarray(m.vertices), np.asarray(m.faces))


RECEIVER = {"name": "rack", "kind": "receiver", "static": True}


@pytest.fixture
def planted(monkeypatch):
    """CS.triangles / CS.parts return synthetic bodies and record who asked for which shape."""
    asked = []

    def tri(ds, oid):
        asked.append((oid, CS.TRIANGLES))
        return _tri_box(0.05)

    def parts(ds, oid):
        asked.append((oid, CS.PARTS))
        return CS.Parts([_cube(0.05)])

    monkeypatch.setattr(CS, "triangles", tri)
    monkeypatch.setattr(CS, "parts", parts)
    return asked


def test_recv_clearance_measures_a_static_receiver_as_triangles(planted):
    from dexengine.record.augment import recv_clearance as RC

    RC._PIECES.clear()
    assert RC.obj_body("hot3d", {"obj_id": "rack"}, RECEIVER).kind == CS.TRIANGLES
    assert RC.obj_body("hot3d", {"obj_id": "cup"}).kind == CS.PARTS  # a main object moves: convex parts
    assert planted == [("rack", CS.TRIANGLES), ("cup", CS.PARTS)]


def test_carry_class_refuses_a_triangle_body(planted):
    from dexengine.data.objects import carry_class as CC

    with pytest.raises(ValueError, match="triangles"):
        CC.parts("hot3d", {"name": "rack", "obj_id": "rack"}, RECEIVER)
    assert not planted  # refused before any shape is read


def test_hand_clear_reads_triangles_like_the_exact_convex_ruler_on_a_closed_box(planted):
    """On a closed convex box the triangle ruler and GJK agree on the depth; on an open sheet (no inside) a hull
    straddling it still reads deep."""
    pytest.importorskip("mujoco")
    from dexengine.data.objects import hand_clear as HC

    hand = [_cube(0.01)]
    I3 = np.eye(3)[None]
    t = np.array([[0.0, 0.0, 0.045]])  # hand cube 6 mm into the 5 cm box's top
    gjk, tri = GjkModel(hand, [_cube(0.05)]), HC.TriModel(hand, _tri_box(0.05))
    for m in (gjk, tri):
        m.place(I3, t, np.eye(3), np.zeros(3))
    assert tri.depth()[0] == pytest.approx(gjk.depth()[0], abs=CS.SAMPLE_M)
    V = np.array([[-0.05, -0.05, 0], [0.05, -0.05, 0], [0.05, 0.05, 0], [-0.05, 0.05, 0]], float)
    sheet = HC.TriModel(hand, CS.Triangles(V, np.array([[0, 1, 2], [0, 2, 3]])))
    sheet.place(I3, np.zeros((1, 3)), np.eye(3), np.zeros(3))
    assert sheet.depth()[0] >= 0.01 - CS.SAMPLE_M


def test_the_judge_fingerprint_names_a_triangle_support(monkeypatch):
    from dexengine.placement.judge import input_clip as IC

    monkeypatch.setattr(IC._objhull(), "usd_sha256", lambda f: "ab" * 32)
    monkeypatch.setattr(IC._objhull(), "usd_file", lambda ds, oid, art: f"/u/{oid}.usd")
    assert IC._tri_tail("hot3d", "support", "tray") == (CS.TRIANGLES, "ab" * 8)
    assert IC._tri_tail("hot3d", "main", "cup") == ()  # a parts-only row keeps its field
    p = RR.DATASET_PROFILE["hot3d"]
    off = dict(p.scene_rules_off, **{CS.RULE_ID: "test opt-out"})
    monkeypatch.setitem(RR.DATASET_PROFILE, "hot3d", dataclasses.replace(p, scene_rules_off=off))
    assert IC._tri_tail("hot3d", "support", "tray") == ()


def test_step_2f_drops_a_true_shape_object_overlap_with_its_own_word(monkeypatch):
    from dexengine.data.objects import hand_clear as HC
    from dexengine.placement.repair import hand_clear_stage as HS

    def row(side, status, why=""):
        return {"clip": "c", "side": side, "status": status, "why": why}

    v = HS.drop_verdicts([row("right", "CLEAN"), row(HS.OBJECTS, "DROP", "true-shape overlap plate×cup 7.0 mm")])
    assert v == {"c": (CS.OVERLAP_DROP_WORD, "objects: true-shape overlap plate×cup 7.0 mm")}
    assert HS.drop_verdicts([row("right", HS.KEPT, "x"), row(HS.OBJECTS, "CLEAN")]) == {}  # kept as recorded
    monkeypatch.setitem(HS.ACTION, HS.KEPT, "exclude")  # were kept sides excluded: their word, after the overlap
    v = HS.drop_verdicts([row("right", HS.KEPT, "x"), row(HS.OBJECTS, "CLEAN")])
    assert v["c"][0] == HC.KEPT_WORD
    v = HS.drop_verdicts([row("right", HS.KEPT, "x"), row(HS.OBJECTS, "DROP", "y")])
    assert v["c"][0] == CS.OVERLAP_DROP_WORD  # both: the object overlap names the clip
    rows = [{"name": "plate", "overlap_drop": {"partner": "cup", "depth_m": 0.007}}, {"name": "vase"}]
    assert CS.overlap_drop(rows) == (CS.OVERLAP_DROP_WORD, "true-shape overlap plate×cup 7.0 mm")
    assert CS.overlap_drop([{"name": "vase"}]) is None


def test_a_still_row_off_triangles_raises_in_the_still_reader(planted, monkeypatch):
    """The one shared check (collider_shape.require_still_triangles): with S-true-shape off on the dataset a pinned
    receiver reads parts, and hand_clear's ruler raises before reading any shape; with the rule on it reads the
    triangles."""
    from dexengine.data.objects import hand_clear as HC

    o = {"name": "rack", "obj_id": "rack"}
    assert HC._model("hot3d", o, RECEIVER, [_cube(0.01)])([0]) is not None
    monkeypatch.setattr(CS, "on", lambda ds: ds != "hot3d")
    assert CS.row_shape("hot3d", RECEIVER) == CS.PARTS
    with pytest.raises(ValueError, match="still readers measure 'triangles' only"):
        CS.require_still_triangles("hot3d", RECEIVER)
    with pytest.raises(ValueError, match="still readers measure"):
        HC._model("hot3d", o, RECEIVER, [_cube(0.01)])
    assert planted == [("rack", CS.TRIANGLES)]  # the rule-off calls read no shape at all
