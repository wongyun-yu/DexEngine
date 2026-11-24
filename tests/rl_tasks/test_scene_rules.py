"""Pure scene rules: support lift of a pinned object, extra-slot audit rows."""

from __future__ import annotations

import pytest
import torch

from dexengine.rl.tasks.scene import extra_slot_audit as xa
from dexengine.rl.tasks.scene import object_pin as op

# Lid support disc (owner = right): top face z, radius, thickness. The left body pierces it by 72.68 mm.
TOP, R, H = 0.8319, 0.0159, 0.01
PEN_BODY = 0.07268


def _box(hx: float, hz: float) -> torch.Tensor:
    return torch.tensor([[x, y, z] for x in (-hx, hx) for y in (-hx, hx) for z in (-hz, hz)])


def _disc_table(own: bool, on: bool = True) -> dict:
    return {
        "x": torch.zeros(1, 1),
        "y": torch.zeros(1, 1),
        "r": torch.full((1, 1), R),
        "top": torch.full((1, 1), TOP),
        "on": torch.full((1, 1), on),
        "win": torch.ones(1, 1, dtype=torch.bool),
        "own": torch.full((1, 1), own),
        "h": H,
    }


@pytest.fixture
def body_points():
    pts = (_box(0.010, 0.0598) + torch.tensor([0.0, 0.0, TOP - PEN_BODY + 0.0598]))[None]
    return pts, torch.ones(1, 8, dtype=torch.bool)


def test_foreign_disc_penetration_is_measured_separately(body_points):
    pts, valid = body_points
    raw = op.support_lift_parts(pts, valid, None, _disc_table(own=False))
    assert float(raw["disc"][0]) == pytest.approx(PEN_BODY, abs=1e-6)
    assert float(raw[op.FOREIGN][0]) == pytest.approx(PEN_BODY, abs=1e-6)
    d = _disc_table(own=False)
    cut = op.support_lift_parts(pts, valid, None, {**d, "on": d["on"] & d["own"]})
    assert float(cut["disc"][0]) == 0.0 and float(cut[op.FOREIGN][0]) == 0.0
    mine = op.support_lift_parts(pts, valid, None, _disc_table(own=True))
    assert float(mine["disc"][0]) == pytest.approx(PEN_BODY, abs=1e-6) and float(mine[op.FOREIGN][0]) == 0.0
    assert op.FOREIGN not in op.LIFT_SOURCES


def test_disc_table_without_owner_column_fails(body_points):
    pts, valid = body_points
    d = {k: v for k, v in _disc_table(own=True).items() if k != "own"}
    with pytest.raises(KeyError):
        op.support_lift_parts(pts, valid, None, d)


def test_extra_slot_rows_and_expectation():
    slots = ["/e/slot0", "/e/slot1"]
    robot, obj = {"/e/Robot/a", "/e/Robot/b"}, {"/e/obj"}
    filt = {("/e/slot0", "/e/Robot/a"), ("/e/slot0", "/e/Robot/b"), ("/e/slot1", "/e/obj")}
    got = xa.rows(slots, [None, None], robot, obj, filt)
    assert got == {"RE": "empty:OFF(filter 2) empty:ON", "OE": "empty:ON empty:OFF(filter 1)"}
    assert xa.norm("empty:OFF(filter 2) empty:partial(filter 1 / 2)") == "empty:OFF empty:partial"
    assert xa.expected([None]) == {"RE": "empty:ON", "OE": "empty:OFF"}


def test_extra_slot_check_raises_on_mismatch_and_missing_kinds():
    res = [{"extra_slots": 1, "extra": {"RE": "empty:OFF(filter 2)", "OE": "empty:OFF(filter 1)"}}]
    with pytest.raises(RuntimeError, match="no slot kinds"):
        xa.check(res, [0], None)
    with pytest.raises(RuntimeError, match="differs"):
        xa.check(res, [0], [[None]])
    ok = [{"extra_slots": 1, "extra": {"RE": "empty:ON", "OE": "empty:OFF(filter 1)"}}]
    xa.check(ok, [0], [[None]])
    xa.check([{"extra_slots": 0}], [0], None)


def test_stand_in_is_the_smallest_slot_usd(tmp_path):
    from dexengine.rl.tasks.scene import extra_slots as xe

    sizes = {"b.usd": 30, "a.usd": 10, "c.usd": 10, "d.usd": 50}
    p = {n: tmp_path / n for n in sizes}
    for n, k in sizes.items():
        p[n].write_bytes(b"x" * k)
    rows = [[str(p["b.usd"]), str(p["d.usd"])], [str(p["c.usd"])], [str(p["a.usd"]), None]]
    rule = xe.standin_rule()
    assert rule == "lightest"
    got = xe.stand_in(rows, 1, rule)
    assert got == str(p["a.usd"])  # smallest file; a/c tie broken by path
    assert xe.stand_in(list(reversed(rows)), 1, rule) == got  # clip order does not matter


def test_stand_in_without_any_clip_for_the_slot_fails():
    from dexengine.rl.tasks.scene import extra_slots as xe

    with pytest.raises(RuntimeError, match="no clip has slot 2"):
        xe.stand_in([["/x.usd"], ["/y.usd", None]], 2, xe.standin_rule())
    with pytest.raises(RuntimeError, match="unknown stand-in rule"):
        xe.stand_in([["/x.usd"]], 0, "first")


def _split_labels():
    """Part labels the truth-table rows (`collision_rows._effective_rows`) write through
    `both(label_a, ..., label_b, ...)` (read by AST: the module needs pxr)."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "dexengine/rl/tasks/scene/collision_rows.py"
    pairs = set()
    for node in ast.walk(ast.parse(src.read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "both":
            pairs.add((node.args[0].value, node.args[2].value))
    return pairs


def test_measured_split_labels_match_the_stated_preset_values():
    from dexengine.rl.tasks.scene import collision_presets as cp

    stated = {
        tuple(p.split(":", 1)[0] for p in v.split()) for t in cp.EXPECTED.values() for v in t.values() if ":" in v
    }
    measured = _split_labels()
    assert measured, "no both(...) call found in collision_check"
    assert stated <= measured, f"preset values use labels the audit never writes: {stated - measured}"
    for t in cp.EXPECTED.values():
        for row in cp.XR_SET_ROW.values():
            cp.hand_on(t[row])  # every row the ledger reads parses under the shared vocabulary
    for a, b in measured:
        cp.hand_on(f"{a}:ON {b}:OFF")
