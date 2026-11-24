"""Display-only static extras (dexengine.record.scene.static_extras): physics strip, entry wiring, kind rule, prim
paths. Needs pxr only (no Isaac)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pxr = pytest.importorskip("pxr")

from dexengine.record.scene import extras as EXV  # noqa: E402
from dexengine.record.scene import static_extras as RS  # noqa: E402


def _stage_with_physics():
    """An extras root whose body carries a rigid-body API and whose mesh carries collision APIs."""
    from pxr import Usd, UsdGeom, UsdPhysics

    st = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(st, "/World/CollectLook/extras").GetPrim()
    body = UsdGeom.Xform.Define(st, "/World/CollectLook/extras/e0").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    mesh = UsdGeom.Mesh.Define(st, "/World/CollectLook/extras/e0/geometry/mesh").GetPrim()
    UsdPhysics.CollisionAPI.Apply(mesh)
    UsdPhysics.MeshCollisionAPI.Apply(mesh)
    return st, root


def test_owner_strip_passes_and_physics_left_is_caught():
    from pxr import UsdGeom, UsdPhysics

    from dexengine.record.looks import look_d as LOOK

    st, root = _stage_with_physics()
    assert LOOK._strip_physics(root) == (3, 0)
    RS.assert_visual_only(root)
    _, unstripped = _stage_with_physics()
    with pytest.raises(RS.RenderStaticError):
        RS.assert_visual_only(unstripped)
    UsdPhysics.CollisionAPI.Apply(UsdGeom.Mesh.Define(st, "/World/CollectLook/extras/e1/m").GetPrim())
    assert len(RS.physics_left(root)) == 1  # an API added again after the removal is counted


def _calls(fn: ast.FunctionDef) -> tuple[set[str], set[str]]:
    calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)]
    attr = {
        f"{c.func.value.id}.{c.func.attr}"
        for c in calls
        if isinstance(c.func, ast.Attribute) and isinstance(c.func.value, ast.Name)
    }
    return attr, {c.func.id for c in calls if isinstance(c.func, ast.Name)}


def _function(path: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_entry_reads_the_owner_and_checks_physics():
    attr, plain = _calls(_function(RS.__file__, "apply"))
    assert {"EXV.decide", "EXV.spawn", "EXV.summary"} <= attr
    assert "assert_visual_only" in plain


def test_kind_rule_ignores_a_touched_sibling():
    """Being touched by the hand (here: a sibling in the same part group) does not make an extra a part; only an
    articulation does. A copy that drops for a touched sibling shows up as one extra standing nowhere."""
    case = {"part_groups": {"g": ["a", "b"]}, "contact_frac": {"a": {"right": 0.3}}}
    o_b = {"name": "b", "obj_id": "b"}

    def kind_defect(o):
        grp = next((g for g in case["part_groups"].values() if o["name"] in g), None)
        cf = case["contact_frac"]
        sib = grp and any(float(v) > 0 for s in grp if s != o["name"] for v in (cf.get(s) or {}).values())
        return "part" if sib else EXV._kind(o)

    def standing_nowhere(kind):
        return RS.nowhere([{"name": "a", "reason": "in_sim"}, {"name": "b", "reason": kind(o_b)}])

    assert standing_nowhere(EXV._kind) == (0, "-")
    assert standing_nowhere(kind_defect) == (1, "part 1")
    assert EXV._kind({"name": "c", "articulation": {"j": 1}}) == "part"


def test_prim_paths_are_distinct_per_clip():
    names = [EXV.prim_path(g, k) for g in range(3) for k in range(4)]
    assert [EXV.prim_path(0, k) for k in range(2)] == [f"{EXV.ROOT}/e0", f"{EXV.ROOT}/e1"]
    assert len(set(names)) == len(names)
    _, plain = _calls(_function(EXV.__file__, "spawn"))
    assert "prim_path" in plain


def test_put_points_read_render_only_rows_through_the_decision(monkeypatch):
    """A render-only row is not in the tree's objects[]: its object comes from decide's `objs` (negative on the old
    lookup, which raised KeyError for 'render_only_…' and killed every closing video of such a clip)."""
    got = []
    monkeypatch.setattr(EXV, "scene_points", lambda ds, o, pose, z: got.append(o["obj_id"]) or [len(got)])
    put = [{"name": "render_only_o2", "pose_render": [0] * 7}, {"name": "cup", "pose_render": [0] * 7}]
    dec = {"z_off": 0.0, "objs": {"render_only_o2": {"obj_id": "left_o2"}, "cup": {"obj_id": "cup"}}}
    assert RS.put_points("oakink2", dec, put) == {"render_only_o2": [1], "cup": [2]}
    assert got == ["left_o2", "cup"]
    with pytest.raises(KeyError):
        RS.put_points("oakink2", dict(dec, objs={"cup": {"obj_id": "cup"}}), put)
