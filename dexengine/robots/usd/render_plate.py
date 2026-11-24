"""The render plate = the training plate repainted — the one rule and its check (pxr only, no Kit).

A render body (``dexengine.core.arms.RobotGeneration.material_of`` — the metal repaint
``dexengine/robots/usd/render_body.py`` makes) differs from the body it repaints only in materials, and its sim-collider
plate (``<combo><render body><rev suffix>``, the one ``dexengine.core.arms.plate_variant`` names — the one plate every
process spawns) must therefore equal the training plate (``dexengine.core.arms.sim_usd_variant``) except for visual
material prims and visual bindings (and the swapped head camera — ``camera_roots``: the training plate's camera visual
by the head camera rule, the ZED Mini root on the render plate — and the render-only mount parts the pair declares,
``dexengine.robots.usd.render_parts.allowed_roots``; any other prim on one plate only fails the check): same
prims, joints, masses, geometry, colliders to 1e-6 m, and every collider prim gets the same physics material (a repaint
never touches friction or restitution). That material is what ``ComputeBoundMaterial("physics")`` resolves — the
physics-purpose binding, else the all-purpose one, inherited and collection bindings included: a superset of what USD
Physics' own parser (``UsdPhysics.LoadUsdPhysicsFromRange``) takes, so the check is stricter than physics, never looser
— and only a material with the physics material API counts (the parser gives a collider none for a visual material).
Readers: the generator (``collider_rev.py`` — after every render bake and in ``--verify-only``) and the dataset merge
gate. The marker is ``[RENDER_USD_PARITY]``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from dexengine.robots.usd import render_parts as RVP

COLLIDER_TOL_M = 1e-6
MARK = "[RENDER_USD_PARITY]"
_PHYSICS_APIS = (UsdPhysics.CollisionAPI, UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI, UsdPhysics.ArticulationRootAPI)


def attrs(p: Usd.Prim) -> dict:
    """Authored attribute values of a prim."""
    return {a.GetName(): a.Get() for a in p.GetAttributes() if a.HasAuthoredValue()}


def _visual_binding(name: str) -> bool:
    """A material binding of a visual purpose (all-purpose, preview or full; direct or collection) — a repaint may
    rebind it. A binding whose purpose is physics (``material:binding:physics`` or its collection form) is not."""
    return name.startswith("material:binding") and "physics" not in name.split(":")


def _rels(p: Usd.Prim) -> dict:
    """Relationship targets, visual material bindings left out (a repaint may rebind those; a joint's bodies and a
    physics-purpose binding may not change — and what any binding gives a collider as physics is
    ``_physics_material``'s check)."""
    return {
        r.GetName(): [str(t) for t in r.GetTargets()] for r in p.GetRelationships() if not _visual_binding(r.GetName())
    }


def _material_scoped(p: Usd.Prim) -> bool:
    """A visual material prim or anything under it (its shaders) — the only prims a repaint may change. A material with
    the physics material API is physics (friction · restitution · density), compared like any other prim. ``parity``
    asks it of the prim on both plates: a material that is physics on either side is not a repaint."""
    while p and not p.IsPseudoRoot():
        if p.IsA(UsdShade.Material):
            return not p.HasAPI(UsdPhysics.MaterialAPI)
        p = p.GetParent()
    return False


def _physics_material(p: Usd.Prim) -> tuple | None:
    """The physics material a (collider) prim gets: the material ``ComputeBoundMaterial("physics")`` resolves, as
    (path · applied schemas · authored attributes) — None when there is none or it lacks the physics material API."""
    m = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial("physics")[0]
    q = m.GetPrim() if m else None
    return (str(q.GetPath()), list(q.GetAppliedSchemas()), attrs(q)) if q and q.HasAPI(UsdPhysics.MaterialAPI) else None


def camera_roots(train: Usd.Stage, arm: str) -> tuple[dict, dict]:
    """The head camera swap, per plate: ({link: training plate's camera visual}, {link: (ZED Mini root,)}). The
    camera visual is measured on the training plate (``head_camera.camera_visual_names`` — a rule, not a name list;
    the generator checks it against the URDF); an arm whose skeleton declares no head camera link has none (both {})."""
    from dexengine.robots.usd import head_camera as HC

    if not arm or HC.camera_link(arm) is None:
        return {}, {}
    link, mine = HC.swap_rule()
    return {link: tuple(HC.camera_visual_names(train))}, {link: (mine,)}


def _render_only_visual(p: Usd.Prim, roots: dict) -> bool:
    """A prim of a swapped head camera visual (``roots`` = one side of ``camera_roots``) or of a declared render part
    (``roots`` = ``dexengine.robots.usd.render_parts.allowed_roots`` of the plate's pair) — the caller passes only prims
    present on one plate (a part: on the render plate only): its path ``…/<link>/visuals/<root>[/…]`` with ``<root>``
    listed for ``<link>``, not a joint, and no physics schema on it or any prim below it."""
    parts = str(p.GetPath()).split("/")
    if "visuals" not in parts:
        return False
    i = parts.index("visuals")
    if i < 1 or i + 1 >= len(parts) or parts[i + 1] not in roots.get(parts[i - 1], ()):
        return False
    return not any(
        q.IsA(UsdPhysics.Joint) or any(q.HasAPI(api) for api in _PHYSICS_APIS)
        for q in Usd.PrimRange(p, Usd.TraverseInstanceProxies())
    )


def parity(train: Path, render: Path) -> dict:
    """Composed stages, the default prim with its instances resolved (what a spawn references — the layer's other root
    prims are never referenced): every prim outside material scopes is identical (type · applied schemas · authored
    attributes · relationships), every prim of every collider gets the same ``_physics_material`` and the colliders'
    mesh points agree; the repaint itself must not be empty; and the render-only mount parts the plate's pair declares
    are all there at their declared poses (``dexengine.robots.usd.render_parts.problems`` — a plate without them is
    stale)."""
    a, b = Usd.Stage.Open(str(train)), Usd.Stage.Open(str(render))
    pa, pb = (
        {str(p.GetPath()): p for p in Usd.PrimRange(s.GetDefaultPrim(), Usd.TraverseInstanceProxies())} for s in (a, b)
    )
    differ = set(pa) ^ set(pb) | {
        p
        for p in set(pa) & set(pb)
        if (
            pa[p].GetTypeName() != pb[p].GetTypeName()
            or attrs(pa[p]) != attrs(pb[p])
            or _rels(pa[p]) != _rels(pb[p])
            or list(pa[p].GetAppliedSchemas()) != list(pb[p].GetAppliedSchemas())
        )
    }
    paint = {p for p in differ if all(_material_scoped(x) for x in (pa.get(p), pb.get(p)) if x)}
    # the render plate's pair from its directory name (a name that is no combo — a synthetic plate — has none): the
    # head camera swap (camera_roots) and the declared mount parts (dexengine.core.mounts — read through
    # render_visual_parts)
    pair = RVP.pair_of(Path(render).parent.name, strict=False)
    cam_a, cam_b = camera_roots(a, pair[0])
    addon = {p for p in set(pa) - set(pb) - paint if _render_only_visual(pa[p], cam_a)} | {
        p for p in set(pb) - set(pa) - paint if _render_only_visual(pb[p], cam_b)
    }
    roots = RVP.allowed_roots(*pair)
    part = {p for p in set(pb) - set(pa) - paint - addon if roots and _render_only_visual(pb[p], roots)}
    # and the declared parts must be there (a render plate baked before the declaration lacks them)
    missing = RVP.problems(b, *pair) if roots else []
    changed = sorted(differ - paint - addon - part)
    cols = [p for p in pa if pa[p].HasAPI(UsdPhysics.CollisionAPI)]
    dv, phys = 0.0, []
    for d in (d for p in cols for d in Usd.PrimRange(pa[p], Usd.TraverseInstanceProxies())):
        e = pb.get(str(d.GetPath()))  # missing on the render plate: already a changed prim
        if e is not None and _physics_material(d) != _physics_material(e):
            phys.append(str(d.GetPath()))
        if not d.IsA(UsdGeom.Mesh):
            continue
        x = np.asarray(d.GetAttribute("points").Get() or [], float)
        y = np.asarray(e.GetAttribute("points").Get() or [], float) if e is not None and e.IsA(UsdGeom.Mesh) else None
        dv = max(dv, float(np.abs(x - y).max()) if y is not None and x.shape == y.shape and x.size else float("inf"))
    meta = [
        k
        for k in ("upAxis", "metersPerUnit", "kilogramsPerUnit", "defaultPrim")
        if a.GetMetadata(k) != b.GetMetadata(k)
    ]
    return {
        "prims": len(pa),
        "colliders": len(cols),
        "collider_max_dv_m": dv,
        "material_prims_changed": len(paint),
        "changed": changed[:5],
        "n_changed": len(changed),
        "camera_swap_prims": len(addon),
        "render_part_prims": len(part),
        "parts_missing": missing[:5],
        "n_parts_missing": len(missing),
        "meta": meta,
        "physics_changed": phys[:5],
        "n_physics_changed": len(phys),
        "ok": bool(cols)
        and bool(paint)
        and not changed
        and not phys
        and not meta
        and dv <= COLLIDER_TOL_M
        and not missing,
    }


def split_parity(train: Path, render: Path) -> dict:
    """The split pairs (``<plate dir>_split/<plate dir>_split_{R,L}.usd`` — split_arm.py): the same layer text
    once the render plate's name is replaced by the training plate's (a split file only references its plate and
    switches prims off)."""
    names, res = (train.parent.name, render.parent.name), {}
    for s in ("R", "L"):
        f = [d.parent.parent / f"{d.parent.name}_split" / f"{d.parent.name}_split_{s}.usd" for d in (train, render)]
        txt = [Sdf.Layer.FindOrOpen(str(x)).ExportToString() if x.is_file() else None for x in f]
        res[s] = None not in txt and txt[1].replace(names[1], names[0]) == txt[0]
    return {**res, "ok": all(res.values())}


def check(train: Path, render: Path, with_split: bool = True) -> bool:
    """``parity`` (+ ``split_parity``) with the marker line — True only when both hold."""
    base = parity(train, render)
    sp = split_parity(train, render) if with_split else {"skipped": True, "ok": True}
    print(
        f"{MARK} {render.parent.name} ↔ {train.parent.name} ok={base['ok'] and sp['ok']} · "
        f"colliders {base['colliders']} max diff {base['collider_max_dv_m']} m · "
        f"other prims differing outside materials {base['n_changed']} {base['changed']} · "
        f"collider prims with other physics material {base['n_physics_changed']} {base['physics_changed']} · "
        f"material prims {base['material_prims_changed']} · head-camera swap prims {base['camera_swap_prims']} · "
        f"mounted part prims {base['render_part_prims']} · "
        f"declared part problems {base['n_parts_missing']} {base['parts_missing']} · "
        f"meta {base['meta']} · split {sp}",
        flush=True,
    )
    return base["ok"] and sp["ok"]
