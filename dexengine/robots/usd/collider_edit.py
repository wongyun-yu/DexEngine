"""USD edits of the collider revision: author the convex-hull colliders on an arm link and on the world-frame bodies,
from the hull geometry (robots.wrist_hull) and the sim-collider table (placement.collision.sim_collider). Called by
dexengine.robots.usd.collider_rev, which bakes, records and verifies the revised body."""

from __future__ import annotations

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

from dexengine.placement.collision import sim_collider as SC
from dexengine.robots import wrist_hull as G


def _proto_of(layer: Sdf.Layer, inst_path: str) -> str:
    """The internal reference target of an instanceable ``collisions`` prim — and it must have no other user."""
    spec = layer.GetPrimAtPath(inst_path)
    if spec is None or not spec.instanceable:
        raise SystemExit(f"[wrist_hull] {inst_path} is not an instance prim — the asset structure is not as expected")
    refs = [it.primPath for it in spec.referenceList.GetAddedOrExplicitItems() if not it.assetPath]
    if len(refs) != 1:
        raise SystemExit(f"[wrist_hull] {inst_path} has {len(refs)} internal references — expected 1")
    tgt, users = str(refs[0]), []

    def visit(path):
        s = layer.GetPrimAtPath(path)
        if s is not None and any(str(it.primPath) == tgt for it in s.referenceList.GetAddedOrExplicitItems()):
            users.append(str(path))

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    if users != [inst_path]:
        raise SystemExit(f"[wrist_hull] prototype {tgt} users {users} — editable only when {inst_path} is the only one")
    return tgt


def _author_hull(stage: Usd.Stage, parent: str, name: str, verts: np.ndarray, tris: np.ndarray) -> None:
    """Same pattern as the hand colliders of these USDs:
    guide Xform → Xform(Collision + MeshCollision convexHull) → Mesh."""
    top = UsdGeom.Xform.Define(stage, f"{parent}/{name}")
    top.CreatePurposeAttr(UsdGeom.Tokens.guide)
    node = UsdGeom.Xform.Define(stage, f"{parent}/{name}/node_hull")
    for x in (top, node):
        x.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
        x.AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))
        x.AddScaleOp().Set(Gf.Vec3f(1, 1, 1))
    UsdPhysics.CollisionAPI.Apply(node.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(node.GetPrim()).CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
    mesh = UsdGeom.Mesh.Define(stage, f"{parent}/{name}/node_hull/mesh")
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in verts]))
    mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(tris)))
    mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray([int(i) for i in tris.reshape(-1)]))
    lo, hi = verts.min(0), verts.max(0)
    mesh.CreateExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)


def edit_link(stage: Usd.Stage, root: str, hull, rev, vendor: list) -> dict:
    """Replace (mode "replace") or extend (mode "add") one link's colliders with its revision hull (sim_collider.Hull);
    ``vendor`` = the link's URDF capsules (sim_collider.vendor_capsules) the removed USD capsule must match."""
    link, mode = hull.link, hull.mode
    layer = stage.GetRootLayer()
    proto = _proto_of(layer, f"{root}/{link}/collisions")
    rec: dict = {"mode": mode, "prototype": proto}
    # the prototype source is a root `over`, so its children are not "defined" — walk with the all-prims predicate
    kids = list(stage.GetPrimAtPath(proto).GetAllChildren())
    if mode == "replace":
        shapes = [
            d for c in kids for d in Usd.PrimRange(c, Usd.PrimAllPrimsPredicate) if d.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if len(shapes) != 1 or not shapes[0].IsA(UsdGeom.Capsule):
            got = [(str(s.GetPath()), s.GetTypeName()) for s in shapes]
            raise SystemExit(f"[wrist_hull] {link}: expected 1 vendor capsule, got {got}")
        cap = UsdGeom.Capsule(shapes[0])
        rec["removed"] = {
            "prim": str(shapes[0].GetPath()).replace(proto, "collisions"),
            "radius": cap.GetRadiusAttr().Get(),
            "height": cap.GetHeightAttr().Get(),
        }
        if (
            len(vendor) != 1
            or abs(vendor[0].r - rec["removed"]["radius"]) > 1e-6
            or abs(vendor[0].L - rec["removed"]["height"]) > 1e-6
        ):
            raise SystemExit(
                f"[wrist_hull] {link}: USD capsule {rec['removed']} ≠ URDF capsule "
                f"{[(c.r, c.L) for c in vendor]} — the source USD was not baked from this URDF"
            )
        for c in kids:
            stage.RemovePrim(c.GetPath())
    elif any(c.GetName() == rev.prim_name for c in kids):
        raise SystemExit(f"[wrist_hull] {link}: {rev.prim_name} already exists — the source is already a revised body")
    verts, stats = hull.verts, dict(hull.stats)
    tris = G.hull_triangles(verts)
    _author_hull(stage, proto, rev.prim_name, verts, tris)
    # Define() turns undefined ancestors into `def` — the prototype source must stay a root `over` (otherwise it
    # becomes a second, defined root prim with a whole hand under it)
    layer.GetPrimAtPath(proto).specifier = Sdf.SpecifierOver
    if mode == "replace":
        stats["outside_removed_capsule_mm"] = G.capsule_outside_mm(verts, *vendor[0].ends(), vendor[0].r)
    rec.update(meshes=hull.meshes, triangles=int(len(tris)), **stats)
    return rec


def _usd_caps(stage: Usd.Stage, col: str) -> list[tuple[Usd.Prim, np.ndarray, float, float]]:
    """(prim, 4×4 in the stage frame, radius, height) of every capsule under ``col`` (instance proxies)."""
    out = []
    for d in Usd.PrimRange(stage.GetPrimAtPath(col), Usd.TraverseInstanceProxies()):
        if d.HasAPI(UsdPhysics.CollisionAPI):
            if not d.IsA(UsdGeom.Capsule) or UsdGeom.Capsule(d).GetAxisAttr().Get() != "Z":
                raise SystemExit(
                    f"[wrist_hull] {d.GetPath()}: a collider that is not a Z-axis capsule — "
                    "only torso vendor capsules may be here"
                )
            T = np.array(UsdGeom.Xformable(d).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            c = UsdGeom.Capsule(d)
            out.append((d, T, float(c.GetRadiusAttr().Get()), float(c.GetHeightAttr().Get())))
    return out


def edit_world(stage: Usd.Stage, root: str, rev, uroot, torso: list) -> dict:
    """The static root body ``world`` (the URDF root and its fixed torso links, merged by the import): its vendor torso
    capsules (``torso_replace``) leave and every torso hull part (sim_collider.torso_bodies, root coordinates = the
    ``world`` frame) is added as ``rev.torso_prim(link, part)``. Each removed capsule is matched to its URDF capsule by
    radius, height, centre and axis."""
    layer = stage.GetRootLayer()
    proto = _proto_of(layer, f"{root}/world/collisions")
    Tw = np.linalg.inv(
        np.array(
            UsdGeom.Xformable(stage.GetPrimAtPath(f"{root}/world")).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        ).T
    )
    want = {}
    for ln in rev.torso_replace:
        T = SC.fixed_frame(uroot, ln)
        (c,) = SC.vendor_capsules(uroot, ln)
        a, b = (T[:3, :3] @ np.asarray(e, float) + T[:3, 3] for e in c.ends())
        want[ln] = (c.r, c.L, (a + b) / 2, (b - a) / np.linalg.norm(b - a))
    removed = {}
    for d, T, r, h in _usd_caps(stage, f"{root}/world/collisions"):
        T = Tw @ T
        hit = [
            ln
            for ln, (wr, wl, wc, wa) in want.items()
            if abs(wr - r) <= 1e-6
            and abs(wl - h) <= 1e-6
            and np.abs(T[:3, 3] - wc).max() <= 1e-4
            and abs(float(T[:3, 2] @ wa)) >= 1 - 1e-6
        ]
        if len(hit) != 1 or hit[0] in removed:
            raise SystemExit(
                f"[wrist_hull] world: USD capsule {d.GetPath()} (r {r} · h {h} · centre {T[:3, 3].round(4)}) "
                f"matches no single URDF torso capsule {hit} — the source USD was not baked from this URDF"
            )
        removed[hit[0]] = {"prim": str(d.GetPath()).replace(f"{root}/world/", ""), "radius": r, "height": h}
    if set(removed) != set(want):
        raise SystemExit(
            f"[wrist_hull] world: torso capsules to remove {sorted(want)} ≠ found in the USD {sorted(removed)}"
        )
    for c in list(stage.GetPrimAtPath(proto).GetAllChildren()):
        stage.RemovePrim(c.GetPath())
    recs = {}
    for b in torso:
        tris = G.hull_triangles(b.verts)
        _author_hull(stage, proto, rev.torso_prim(b.link, b.part), b.verts, tris)
        recs[b.name] = dict(b.stats, meshes=b.meshes, triangles=int(len(tris)))
    layer.GetPrimAtPath(proto).specifier = Sdf.SpecifierOver
    return {"mode": "torso", "prototype": proto, "links": recs, "removed": removed}
