"""S-true-shape triangle colliders (CPU, an in-memory USD stage): a triangles cell's body collides with its visual mesh
(approximation none, the parts' material values) and its convex parts are off; parts and empty cells stay as they
were; the read-back audit refuses a planted mismatch; the slot statement writes a triangles cell only where a collider
stands. The bodies are built two ways: defined in the stage, and as the env spawns them (the env prim references an
object USD on disk, so the meshes have no spec in the edit layer before authoring: the ab6 smoke failure)."""

from __future__ import annotations

import types

import numpy as np
import pytest

pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from dexengine.data.objects import collider_shape as CS  # noqa: E402
from dexengine.rl.tasks.scene import triangle_colliders as TC  # noqa: E402
from tests.imports import own_module


def _mesh(stage, path, collide: bool):
    m = UsdGeom.Mesh.Define(stage, path)
    v = [Gf.Vec3f(*s) for s in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3) * 0.02]
    m.CreatePointsAttr(v)
    m.CreateFaceVertexCountsAttr([3, 3])
    m.CreateFaceVertexIndicesAttr([0, 1, 2, 1, 2, 3])
    if collide:
        p = m.GetPrim()
        UsdPhysics.CollisionAPI.Apply(p)
        UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
        UsdPhysics.MaterialAPI.Apply(p).CreateStaticFrictionAttr().Set(2.0)
    return m.GetPrim()


def _object(stage, root):
    """The object USD bake layout under `root`: a rigid base_link, its visual mesh and two convex collision meshes."""
    b = UsdGeom.Xform.Define(stage, f"{root}/base_link").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(b)
    _mesh(stage, f"{root}/base_link/visuals/mesh", False)
    _mesh(stage, f"{root}/base_link/collision_0", True)
    _mesh(stage, f"{root}/base_link/collision_1", True)


def _stage(n_env=2, slots=2, usd_dir=None):
    """Envs × slots. `usd_dir` given: each slot prim references an object USD written there (the env's spawn),
    with the spawner's kinematic over on base_link; otherwise the bodies are defined in the stage."""
    stage = Usd.Stage.CreateInMemory()
    if usd_dir is not None:
        src = Usd.Stage.CreateNew(str(usd_dir / "object.usd"))
        src.SetDefaultPrim(UsdGeom.Xform.Define(src, "/Object").GetPrim())
        _object(src, "/Object")
        src.GetRootLayer().Save()
    roots = []
    for i in range(n_env):
        r = f"/World/envs/env_{i}"
        for k in range(slots):
            slot = f"{r}/ExtraObj_{k}"
            if usd_dir is None:
                _object(stage, slot)
                continue
            stage.DefinePrim(slot, "Xform").GetReferences().AddReference(str(usd_dir / "object.usd"))
            UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(f"{slot}/base_link")).CreateKinematicEnabledAttr().Set(True)
        roots.append(r)
    return stage, roots, (lambda r: [f"{r}/ExtraObj_{k}" for k in range(slots)])


def _enabled(stage, slot):
    return [
        str(p.GetPath())
        for p in Usd.PrimRange(stage.GetPrimAtPath(slot))
        if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False
    ]


@pytest.mark.parametrize("spawned", [False, True], ids=["defined", "referenced"])
def test_triangles_cells_collide_with_the_visual_mesh_and_parts_cells_keep_their_parts(spawned, tmp_path):
    stage, roots, paths = _stage(usd_dir=tmp_path if spawned else None)
    vis0 = f"{roots[0]}/ExtraObj_0/base_link/visuals/mesh"
    assert bool(stage.GetRootLayer().GetPrimAtPath(vis0)) is not spawned  # referenced: no spec before authoring
    cells = TC.slot_cells([[CS.TRIANGLES, CS.PARTS]], roots, paths)
    got = TC.author_triangles(stage, cells, "extra slots", len(roots))
    assert got == {"cells": 2, "disabled": 4, "faces_max": 2}
    for r in roots:
        vis = stage.GetPrimAtPath(f"{r}/ExtraObj_0/base_link/visuals/mesh")
        assert _enabled(stage, f"{r}/ExtraObj_0") == [str(vis.GetPath())]
        assert UsdPhysics.MeshCollisionAPI(vis).GetApproximationAttr().Get() == UsdPhysics.Tokens.none
        assert vis.HasAPI(UsdPhysics.MaterialAPI) and vis.GetAttribute("physics:staticFriction").Get() == 2.0
        assert len(_enabled(stage, f"{r}/ExtraObj_1")) == 2  # parts cell untouched
    if spawned:  # the edits live in the stage's edit layer; the object USD all envs reference is unchanged
        src = Usd.Stage.Open(str(tmp_path / "object.usd"))
        assert not src.GetPrimAtPath("/Object/base_link/visuals/mesh").HasAPI(UsdPhysics.CollisionAPI)


def test_no_statement_or_no_triangles_cell_authors_nothing_and_says_so(capsys):
    stage, roots, paths = _stage()
    assert TC.slot_cells(None, roots, paths) == []
    assert TC.author_triangles(stage, [], "supports", len(roots))["cells"] == 0
    cells = TC.slot_cells([[CS.PARTS, None]], roots, paths)
    assert TC.author_triangles(stage, cells, "extra slots", len(roots)) == {"cells": 0, "disabled": 0, "faces_max": 0}
    assert all(len(_enabled(stage, s)) == 2 for s, _ in cells)
    out = capsys.readouterr().out
    assert "[TRI_COLL] supports triangles cells 0 over 2 envs (no statement)" in out
    assert "[TRI_COLL] extra slots triangles cells 0 over 2 envs (no triangles cell in the statement)" in out


class _View:
    def __init__(self, paths, contact):
        self.prim_paths, self._c = paths, np.asarray(contact, dtype=float)

    def get_contact_offsets(self):
        return self._c

    def get_rest_offsets(self):
        return np.zeros_like(self._c)


def _scene(stage, rows: dict, robot_row):
    """A parsed-scene stand-in: one rigid object per slot (rows {slot path: view row}) and one articulation."""
    objs = {p: types.SimpleNamespace(root_physx_view=_View([f"{p}/base_link"], [row])) for p, row in rows.items()}
    robot = types.SimpleNamespace(root_physx_view=_View(["/World/envs/env_0/Robot"], [robot_row]))
    return types.SimpleNamespace(stage=stage, rigid_objects=objs, articulations={"robot": robot})


def test_offsets_line_reads_the_bodies_and_flags_a_spread_triangles_body(capsys):
    stage, roots, paths = _stage(n_env=1)
    TC.author_triangles(stage, TC.slot_cells([[CS.TRIANGLES, CS.PARTS]], roots, paths), "extra slots", 1)
    tri, parts = paths(roots[0])
    ok = _scene(stage, {tri: [0.00196, 0.00196, 0.00239, 0], parts: [0.00196, 0.00196, 0, 0]}, [0.00196, 0.0059, 0])
    assert TC.offsets_line(ok) == {"tri": 1, "spread": pytest.approx(0.00043), "flagged": []}
    out = capsys.readouterr().out
    assert "triangles bodies 1: contact 1.960–2.390 mm" in out and "flagged > 1 mm 0" in out
    assert "other rigid bodies contact 1.960–1.960 mm · robot links contact 1.960–5.900 mm" in out
    far = _scene(stage, {tri: [0.00196, 0.00196, 0.0035, 0], parts: [0.00196, 0.00196, 0, 0]}, [0.00196, 0])
    assert len(TC.offsets_line(far)["flagged"]) == 1  # 1.54 mm apart: flagged


def test_offsets_line_refuses_a_row_that_does_not_hold_the_usd_shapes():
    stage, roots, paths = _stage(n_env=1)
    TC.author_triangles(stage, TC.slot_cells([[CS.TRIANGLES, CS.PARTS]], roots, paths), "extra slots", 1)
    tri, parts = paths(roots[0])
    short = _scene(stage, {tri: [0.00196, 0, 0, 0], parts: [0.00196, 0.00196, 0, 0]}, [0.00196])
    with pytest.raises(RuntimeError, match="does not hold its 3 USD shapes"):
        TC.offsets_line(short)


def test_clip_cells_follow_env_modulo_clips():
    _stage_, roots, paths = _stage(n_env=3)
    cells = TC.slot_cells([[CS.TRIANGLES, None], [None, CS.TRIANGLES]], roots, paths)
    assert [c for _, c in cells] == [CS.TRIANGLES, None, None, CS.TRIANGLES, CS.TRIANGLES, None]


def test_audit_refuses_a_convex_mesh_left_on_and_a_stray_triangle_collider():
    stage, roots, paths = _stage(n_env=1)
    cells = TC.slot_cells([[CS.TRIANGLES, CS.PARTS]], roots, paths)
    TC.author_triangles(stage, cells, "extra slots", 1)
    UsdPhysics.CollisionAPI(
        stage.GetPrimAtPath(f"{roots[0]}/ExtraObj_0/base_link/collision_0")
    ).GetCollisionEnabledAttr().Set(True)
    with pytest.raises(RuntimeError, match="disagree"):
        TC.audit(stage, cells)
    stage, roots, paths = _stage(n_env=1)
    p = stage.GetPrimAtPath(f"{roots[0]}/ExtraObj_1/base_link/collision_0")
    UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Set(UsdPhysics.Tokens.none)
    with pytest.raises(RuntimeError, match="disagree"):
        TC.audit(stage, TC.slot_cells([[CS.PARTS, CS.PARTS]], roots, paths))


def test_statement_cell_only_where_a_collider_stands(run_dataset):
    xs = own_module("dexengine.rl.env_setup.extra_slots")
    cp = own_module("dexengine.rl.tasks.scene.collision_presets")
    rec = {"kind": "receiver", "shape": CS.TRIANGLES}
    touched = {"kind": "touched", "shape": CS.TRIANGLES}  # robot ON (S-robot-static): a collider stands
    moving = {"kind": "moving", "shape": CS.PARTS}
    run_dataset()
    assert xs._collider_cell(cp, touched) == CS.TRIANGLES
    run_dataset(robot_static=False)  # all partners OFF in training: no collider stands
    assert xs._collider_cell(cp, rec) == CS.TRIANGLES
    assert xs._collider_cell(cp, touched) is None
    assert xs._collider_cell(cp, moving) == CS.PARTS
    assert xs._collider_cell(cp, None) is None
