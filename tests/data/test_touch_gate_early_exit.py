"""S-touch-gate kinds_only (f0's scene rows, extra_slots.clip_rows kinds_only=True): a pair stops at its first frame
deeper than the true-shape overlap. Same pass, on and kind as the full measure; fewer frames read when a pair fails;
the failing pair's depth is a lower bound (marked depth_at_least); the full measure (default) is untouched and the
memo keeps the two apart."""

from __future__ import annotations

import numpy as np
import pytest

THR = None


class _Body:
    art = None

    def links(self, qrow):
        return [("main-ruler", np.eye(4))]


class _Static:
    kind = "triangles"


def _setup(monkeypatch, depths):
    """A static row s and a main object m whose frame f sits at x = f: the stubbed collider depth of frame f is
    depths[f]; returns (TG, ti, row, mains, read) with read = the frames the collider was asked about."""
    from dexengine.data.objects import collider_shape as CS
    from dexengine.data.objects import scene_requirement as SR
    from dexengine.data.objects import touch_gate as TG

    global THR
    THR = SR.TRUE_SHAPE_OVERLAP_M
    read = []

    def pair_depth(a, b, M):
        f = int(round(M[0, 3]))
        read.append(f)
        return depths[f]

    monkeypatch.setattr(CS, "pair_depth", pair_depth)
    monkeypatch.setattr(TG, "static_body", lambda ds, o, row: (_Static(), ""))
    monkeypatch.setattr(TG, "body_of", lambda ds, o, tree_dir: (_Body(), ""))
    monkeypatch.setattr(TG, "MEMO", {})
    ti = {
        "dataset_name": "x",
        "objects": [
            {"name": "m", "role": "main", "mesh_dir": "a", "obj_id": "2", "qpos_start": 0},
            {"name": "s", "role": "extra", "mesh_dir": "b", "obj_id": "1", "qpos_start": 7},
        ],
    }
    q = np.zeros((len(depths), 14))
    q[:, 0] = np.arange(len(depths))
    q[:, 3] = q[:, 10] = 1.0
    return TG, ti, q, {"name": "s"}, TG.main_objects(ti), read


@pytest.mark.parametrize(
    ("depths", "on"),
    [([0.0, 0.001, 0.004, 0.002, 0.0], True), ([0.0, 0.001, 0.009, 0.02, 0.003], False), ([0.03, 0.0, 0.04], False)],
)
def test_same_verdict_fewer_frames(monkeypatch, depths, on):
    TG, ti, q, row, mains, read = _setup(monkeypatch, depths)
    full = TG.gate_row("t", ti, q, row, mains)
    n_full = len(read)
    fast = TG.gate_row("t", ti, q, row, mains, kinds_only=True)  # a separate memo entry: measured again
    n_fast = len(read) - n_full
    assert full["on"] is fast["on"] is on
    assert full["pairs"]["m"]["pass"] is fast["pairs"]["m"]["pass"] is on
    assert full["pairs"]["m"]["depth_m"] == round(max(depths), 4) and n_full == len(depths)
    first_over = next((f for f, d in enumerate(depths) if d > THR), None)
    if on:
        assert fast == full and n_fast == len(depths)
    else:
        assert n_fast == first_over + 1  # stops at the first frame over the overlap
        assert fast["pairs"]["m"]["depth_m"] == round(depths[first_over], 4) <= full["pairs"]["m"]["depth_m"]
        assert fast["pairs"]["m"]["depth_at_least"] is True and "depth_at_least" not in full["pairs"]["m"]
    # the memo answers each mode with its own value
    again = TG.gate_row("t", ti, q, row, mains)
    assert again == full and len(read) == n_full + n_fast


def test_scene_pass_kind_same(monkeypatch):
    from dexengine.data.objects import extra_slots as XS
    from dexengine.data.objects import scene_rules as SC

    TG, ti, q, row, mains, read = _setup(monkeypatch, [0.0, 0.02, 0.0])
    rows = {m: [{"name": "s", "cls": XS.STATIC_SPAWN, "kind": XS.KIND_TOUCHED}] for m in (False, True)}
    for m, r in rows.items():
        SC.touch_pass("t", ti, q, r, kinds_only=m)
    assert rows[False][0]["kind"] == rows[True][0]["kind"] == XS.KIND_TOUCHED
    assert rows[False][0]["overlap_drop"]["partner"] == rows[True][0]["overlap_drop"]["partner"] == "m"
