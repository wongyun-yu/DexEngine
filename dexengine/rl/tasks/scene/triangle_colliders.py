"""Triangle colliders (rule S-true-shape; the shape of each body is stated by data.objects.collider_shape): a slot body
whose statement cell is `triangles` collides with its visual mesh, not its convex parts.

Authoring, before PhysX parses the stage (one function for the extra slots and the supports, `author_triangles`):
under each env's slot prim of a triangles cell (env i → clip i % C), the visual mesh (`collider_shape.visual_mesh`)
gets the collision API with mesh approximation none and the convex parts' physics material values (friction,
restitution; read from the first collision mesh), and every convex collision mesh of that body is disabled. Cells
`parts` and empty cells (None) are left as they are; without a statement (no triangles cell) nothing is authored.
All reads come first (Usd); the writes are layer edits (Sdf specs in the stage's edit target) under one
`Sdf.ChangeBlock`. Usd authoring inside a change block reads stale composition: a slot prim reached only through
its USD reference has no spec in the edit layer yet, and the new attribute's type then resolves empty.
Audit (`audit`), read back after authoring: a triangles cell has exactly one enabled collider, its visual mesh, with
approximation none; every other cell has no enabled collider with approximation none. A mismatch fails. Marker
`[TRI_COLL] <what> triangles cells <n> over <envs> envs · convex meshes disabled <m> · faces max <f> · audit OK`,
printed also when there is no triangles cell (`triangles cells 0`).
Offsets: no shape gets authored contact or rest offsets, the same as the convex parts in production (PhysX picks
them by shape extent). After parsing, `offsets_line` reads the effective values back from the PhysX views as the
oracle (marker `[TRI_COLL] offsets …`) and flags a triangles body whose shapes differ by more than OFFSET_FLAG_M.
"""

from __future__ import annotations

from dexengine.data.objects.collider_shape import TRIANGLES

TAG = "[TRI_COLL]"
#: Flag of the offsets oracle: a triangles body whose shapes (its triangle mesh, the convex parts it replaces) differ
#: in contact offset by more than this (measured on 10 oakink2 cells: at most 0.42 mm).
OFFSET_FLAG_M = 0.001
_MAT = ("physics:staticFriction", "physics:dynamicFriction", "physics:restitution")


def _body(stage, slot: str):
    from pxr import Usd, UsdPhysics  # noqa: PLC0415

    return next((q for q in Usd.PrimRange(stage.GetPrimAtPath(slot)) if q.HasAPI(UsdPhysics.RigidBodyAPI)), None)


def _convex(body) -> list:
    """The body's collision meshes other than its visual mesh."""
    from pxr import Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

    return [p for p in Usd.PrimRange(body) if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI)]


def _plan_one(body) -> dict:
    """Reads of one triangles body (before any write): visual mesh, convex meshes, the material values, faces."""
    from pxr import UsdGeom  # noqa: PLC0415

    from dexengine.data.objects import collider_shape as CS  # noqa: PLC0415

    convex = _convex(body)
    vis = CS.visual_mesh(body)
    if not convex:
        raise RuntimeError(f"{TAG} {body.GetPath()}: no convex collision mesh to take the material from")
    mat = []
    for name in _MAT:
        a = convex[0].GetAttribute(name)
        if a and a.HasAuthoredValue():
            mat.append((name, a.GetTypeName(), a.GetVariability(), a.Get()))
    faces = len(UsdGeom.Mesh(vis).GetFaceVertexCountsAttr().Get() or [])
    return {"vis": vis.GetPath(), "convex": [m.GetPath() for m in convex], "mat": mat, "faces": faces}


def _spec(target, path):
    """The prim spec of a scene path in the edit target (an `over` is created when the prim has none there)."""
    from pxr import Sdf  # noqa: PLC0415

    p = target.MapToSpecPath(path)
    return target.GetLayer().GetPrimAtPath(p) or Sdf.CreatePrimInLayer(target.GetLayer(), p)


def _apply_api(spec, names) -> None:
    """Add applied API schemas to a prim spec's apiSchemas list op (as UsdPrim.ApplyAPI does)."""
    from pxr import Sdf  # noqa: PLC0415

    op = spec.GetInfo("apiSchemas") if spec.HasInfo("apiSchemas") else Sdf.TokenListOp()
    if op.isExplicit:
        op.explicitItems = list(op.explicitItems) + [n for n in names if n not in op.explicitItems]
    else:
        op.deletedItems = [n for n in op.deletedItems if n not in names]
        have = set(op.prependedItems) | set(op.appendedItems)
        op.prependedItems = list(op.prependedItems) + [n for n in names if n not in have]
    spec.SetInfo("apiSchemas", op)


def _set(spec, name, type_name, value, variability=None) -> None:
    from pxr import Sdf  # noqa: PLC0415

    a = spec.attributes.get(name)
    if a is None:
        a = Sdf.AttributeSpec(spec, name, type_name, variability or Sdf.VariabilityVarying)
    a.default = value


def _write_one(target, plan: dict) -> None:
    """Visual mesh → triangle collider, convex meshes off (layer edits; module doc)."""
    from pxr import Sdf, UsdPhysics  # noqa: PLC0415

    T, V = UsdPhysics.Tokens, Sdf.ValueTypeNames
    vis = _spec(target, plan["vis"])
    _apply_api(vis, ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysicsMaterialAPI"])
    _set(vis, T.physicsCollisionEnabled, V.Bool, True)
    _set(vis, T.physicsApproximation, V.Token, T.none, Sdf.VariabilityUniform)
    for name, type_name, variability, value in plan["mat"]:
        _set(vis, name, type_name, value, variability)
    for m in plan["convex"]:
        _set(_spec(target, m), T.physicsCollisionEnabled, V.Bool, False)


def author_triangles(stage, cells: list, what: str, envs: int) -> dict:
    """Author the triangles cells (module doc). `cells` = [(slot prim path, cell)] over every authoring env; `what` =
    the slot family for the marker. → {"cells": triangles cells, "disabled": convex meshes off, "faces_max"}."""
    from pxr import Sdf  # noqa: PLC0415

    tri = [s for s, c in cells if c == TRIANGLES]
    if not tri:
        why = "no statement" if not cells else "no triangles cell in the statement"
        print(f"{TAG} {what} triangles cells 0 over {envs} envs ({why}) · nothing authored", flush=True)
        return {"cells": 0, "disabled": 0, "faces_max": 0}
    plans = []
    for s in tri:
        body = _body(stage, s)
        if body is None:
            raise RuntimeError(f"{TAG} {s}: triangles cell without a rigid body")
        plans.append(_plan_one(body))
    target = stage.GetEditTarget()
    with Sdf.ChangeBlock():
        for plan in plans:
            _write_one(target, plan)
    off, faces = sum(len(p["convex"]) for p in plans), max(p["faces"] for p in plans)
    audit(stage, cells)
    print(
        f"{TAG} {what} triangles cells {len(tri)} over {envs} envs · convex meshes disabled {off} · faces max {faces} "
        "· audit OK (collider_shape.shape_of)",
        flush=True,
    )
    return {"cells": len(tri), "disabled": off, "faces_max": faces}


def _enabled(body) -> list:
    from pxr import Usd, UsdPhysics  # noqa: PLC0415

    return [
        p
        for p in Usd.PrimRange(body)
        if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False
    ]


def audit(stage, cells: list) -> None:
    """Read back every cell (module doc); a mismatch raises with the first offending slots."""
    from pxr import UsdPhysics  # noqa: PLC0415

    from dexengine.data.objects import collider_shape as CS  # noqa: PLC0415

    def none_approx(p) -> bool:
        return p.HasAPI(UsdPhysics.MeshCollisionAPI) and (
            UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() == UsdPhysics.Tokens.none
        )

    bad = []
    for s, c in cells:
        body = _body(stage, s)
        if body is None:
            continue
        on = _enabled(body)
        if c == TRIANGLES:
            vis = CS.visual_mesh(body)
            if [p.GetPath() for p in on] != [vis.GetPath()] or not none_approx(vis):
                bad.append((s, c, [str(p.GetPath()) for p in on]))
        elif any(none_approx(p) for p in on):
            bad.append((s, c, [str(p.GetPath()) for p in on if none_approx(p)]))
    if bad:
        raise RuntimeError(f"{TAG} audit: {len(bad)} slots disagree with their cell: {bad[:3]}")


def _array(x):
    import numpy as np  # noqa: PLC0415

    return np.asarray(x.detach().cpu() if hasattr(x, "detach") else x, dtype=float)


def _shapes(body) -> int:
    from pxr import Usd, UsdPhysics  # noqa: PLC0415

    return sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(body))


def _is_triangles(body) -> bool:
    """The body collides with its visual mesh (the audit's triangles condition), read from the stage."""
    from pxr import UsdPhysics  # noqa: PLC0415

    from dexengine.data.objects import collider_shape as CS  # noqa: PLC0415

    on = _enabled(body)
    if len(on) != 1 or not body.GetChild(CS.VISUALS).IsValid():
        return False
    vis = CS.visual_mesh(body)
    return on[0].GetPath() == vis.GetPath() and (
        UsdPhysics.MeshCollisionAPI(vis).GetApproximationAttr().Get() == UsdPhysics.Tokens.none
    )


def _row(c, i: int, n: int, path: str):
    """The first n entries of view row i (its shapes; the view pads shorter rows with zeros at the end)."""
    if n > c.shape[1] or (c[i, :n] <= 0).any() or (c[i, n:] != 0).any():
        raise RuntimeError(f"{TAG} offsets: {path}: view row {c[i].tolist()} does not hold its {n} USD shapes first")
    return c[i, :n]


def offsets_line(scene) -> dict:
    """After PhysX parsed the stage: effective contact and rest offsets (PhysX views) of the triangles bodies (their
    shapes: the triangle mesh and the convex parts it replaces, collision off), of the other rigid objects and of the
    articulations (robot links); flags a triangles body whose shapes' contact offsets spread more than OFFSET_FLAG_M.
    The views have no shape paths, so the in-body spread bounds |triangle − part| for every part it replaces."""
    import numpy as np  # noqa: PLC0415

    stage = scene.stage
    tri, other, flagged, rest = [], [], [], []
    for _name, obj in sorted(scene.rigid_objects.items()):
        v = obj.root_physx_view
        c, r = _array(v.get_contact_offsets()), _array(v.get_rest_offsets())
        for i, path in enumerate(v.prim_paths):
            body = stage.GetPrimAtPath(path)
            n = _shapes(body)
            if not n:
                continue
            row = _row(c, i, n, path)
            rest.append(r[i, :n])
            if _is_triangles(body):
                tri.append(row)
                if row.max() - row.min() > OFFSET_FLAG_M:
                    flagged.append((path, row.min(), row.max()))
            else:
                other.append(row)
    robot = [_array(a.root_physx_view.get_contact_offsets()) for _n, a in sorted(scene.articulations.items())]

    def mm(rows) -> str:
        v = np.concatenate([np.ravel(x) for x in rows]) if rows else np.zeros(0)
        v = v[v > 0]
        return f"{v.min() * 1e3:.3f}–{v.max() * 1e3:.3f} mm" if len(v) else "none"

    spread = max((x.max() - x.min() for x in tri), default=0.0)
    rv = np.concatenate(rest) if rest else np.zeros(1)
    print(
        f"{TAG} offsets (PhysX effective, none authored) · triangles bodies {len(tri)}: contact {mm(tri)} (their "
        f"triangle mesh and the convex parts it replaces) · in-body spread max {spread * 1e3:.3f} mm · flagged "
        f"> {OFFSET_FLAG_M * 1e3:.0f} mm {len(flagged)}{' ' + str(flagged[:3]) if flagged else ''} · other rigid "
        f"bodies contact {mm(other)} · robot links contact {mm(robot)} · rigid-body shapes rest "
        f"{rv.min() * 1e3:.3f}–{rv.max() * 1e3:.3f} mm",
        flush=True,
    )
    return {"tri": len(tri), "spread": float(spread), "flagged": flagged}


def slot_cells(cols, roots: list, paths) -> list:
    """[(slot prim path, cell)] of a clip × slot statement `cols` (env i → clip i % C) over the envs `roots`;
    `paths(root)` = that env's slot prims in slot order. No statement = []."""
    if not cols:
        return []
    out = []
    for i, r in enumerate(roots):
        row = cols[i % len(cols)]
        out += [(s, row[k] if k < len(row) else None) for k, s in enumerate(paths(r))]
    return out
