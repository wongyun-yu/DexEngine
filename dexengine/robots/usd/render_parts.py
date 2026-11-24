"""Render-only mount parts — the visual parts a (arm, hand) mount declares, authored onto the render body only.

Why (the adapter visual mesh goes onto the render USD only; training conditions are unchanged):
the merged URDF bolts the hand palm 30.4 mm below the RB-Y1 flange (the measured mount transform) but leaves out the
30 mm adapter plate between them, so every render shows the gap; the real robot carries that plate. The render body
(the repaint ``render_body.py`` makes — never a training, judging or baking body) gets it here; the training body does
not.

Owner of the values: ``dexengine.core.mounts.MountSpec.render_visual_parts`` (pair-keyed, with its ``sources`` record) —
this module reads them, never restates them. Owner here: the authoring (``add`` — the generator's edit), the positive
check (``problems`` — the generator after the edit), the parity allowance (``allowed_roots`` — what
``render_plate.parity`` lets the render plate carry beyond the training plate) and the pair of a plate directory
(``pair_of`` — ``paths.split_combo``, the one combo parser). Per part and side: an Xform ``<name>`` under the EE link's
visual prototype at ``ee_to_part`` · its child ``visual`` at ``part_to_mesh`` · ``visual/mesh`` = the mesh file's
vertices (metres, scale 1) · ``<name>/Looks/<Name>`` an OmniPBR material. No collider, mass, rigid body or joint API
anywhere in that subtree. A pair with no declared part is left untouched (its render body stays byte-identical).
Marker: ``[RENDER_VISUAL_PART]``.

The same parts in URDF form: ``urdf/merge.build`` writes a render URDF
``<combo>_render.urdf`` beside the merged one — that robot plus, per side, a visual-only fixed link
``urdf_link(side, part)`` under the EE link (no collision, no inertial). The merged ``<combo>.urdf`` stays without them:
the bake, the arm IK, the judge and placement read it, and placement takes URDF ``<visual>`` meshes of ``right_*`` /
``left_*`` links as hand check shapes (``dexengine.placement.collision.cross_sets.load_urdf_geometry`` · ``group_of``) —
a part there would change the selection. Owner here: the link and joint names (``urdf_link`` · ``urdf_joint``), the
declared-mesh gate both writers pass (``checked_mesh``) and the bake's guard (``render_links_in_urdf`` — a URDF that
carries a declared part is a render URDF, never a bake input). The render URDF is no USD input: the render body gets its
parts only from ``add`` above (one path).
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path

from dexengine.core.errors import InputMissingError

MARK = "[RENDER_VISUAL_PART]"
POSE_TOL = 1e-6  # m · quaternion components — authored values are the declaration's
VISUAL_CHILD = "visual"  # part frame → mesh frame (the link/visual/mesh shape of a URDF-imported body)


def _registries():
    """(paths, arm_specs, mount_specs) — loaded in the call (the registries pull in the package's spec modules)."""
    from dexengine.core import paths as P

    return P, P.load_registry("arm_specs"), P.load_registry("mount_specs")


def pair_of(plate_dir_name: str, strict: bool = True) -> tuple[str, str]:
    """(arm, hand) of a robot plate directory ``<arm>_<hand><variant>`` — ``paths.split_combo`` against the registered
    names. Not a combo: dies (``strict`` — the generator authors only onto a named body) or, for the parity check of a
    synthetic test plate (``strict=False``), gives ("", "") — no pair declares parts, so nothing more is allowed there
    (the strict side of that check)."""
    P, AS, MS = _registries()
    arm, hand, _v = P.split_combo(
        plate_dir_name,
        [a for a, s in AS.ARM_REGISTRY.items() if s is not None],
        [h for _a, h in MS.MOUNT_REGISTRY],
        unsplit_ok=not strict,
    )
    return arm, hand


def parts_of(arm: str, hand: str) -> tuple:
    """The declared render parts of (arm, hand) — () for a pair without any (or not a pair)."""
    if not arm:
        return ()
    _P, _AS, MS = _registries()
    spec = MS.MOUNT_REGISTRY.get((arm, hand))
    return tuple(spec.render_visual_parts) if spec is not None else ()


def ee_links(arm: str) -> list[str]:
    """The EE link of every side of the arm — the frame ``ee_to_part`` is stated in (``MountSpec``'s frame)."""
    _P, AS, _MS = _registries()
    spec = AS.ARM_REGISTRY[arm]
    return [spec.ee_body_name(s) for s in spec.sides]


def allowed_roots(arm: str, hand: str) -> dict[str, tuple[str, ...]]:
    """link → visual roots the render plate may carry beyond its training plate (``render_plate.parity``)."""
    names = tuple(p.name for p in parts_of(arm, hand))
    return dict.fromkeys(ee_links(arm), names) if names else {}


#: The first path part a declared mesh may start with besides a data-layer top (``data_layout.data_tops`` — the vendor
#: stages ``data/_stage/<hand>``, read from the data root): ``assets`` = a tracked asset of this code tree.
TRACKED_TOPS = ("assets",)


def mesh_file(mesh: str) -> Path:
    """Where a declared mesh path lives — repository-relative with an explicit first part: a tracked asset
    (``TRACKED_TOPS`` → this code tree) or a data-layer top (``data_layout.data_tops`` → the data root; the placing rule
    is ``data_layout.repo_place``, window ``paths.repo_rel_path``, the one the vendor hand URDFs are read with).
    Any other
    first part (or an absolute / ``..`` path) dies — no guess where an unknown prefix lives."""
    P, _AS, _MS = _registries()
    head = Path(mesh).parts[0] if mesh and not Path(mesh).is_absolute() else ""
    tops = (*TRACKED_TOPS, *sorted(P.data_layout.data_tops()))
    if head not in tops or ".." in Path(mesh).parts:
        raise SystemExit(
            f"{MARK} declared mesh {mesh!r}: first part {head!r} is not a known place — it must start "
            f"with a tracked asset root {TRACKED_TOPS} or a data layer "
            f"{tuple(sorted(P.data_layout.data_tops()))}"
        )
    return P.repo_rel_path(mesh)


def mesh_path(part) -> Path:
    return mesh_file(part.mesh)


def checked_mesh(part) -> Path:
    """The declared mesh file — dies when it is missing or not the declared bytes (the gate both writers pass: the
    render body's ``add`` and the render URDF's ``dexengine.robots.urdf.merge``)."""
    return _checked_file(part.name, part.mesh, part.mesh_sha256)


@functools.lru_cache(maxsize=8)
def _checked_file(name: str, mesh: str, sha256: str) -> Path:
    f = mesh_file(mesh)
    if not f.is_file():
        raise InputMissingError(f"{MARK} {name}: declared mesh missing {f} — no other mesh is used instead")
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    if h != sha256:
        raise SystemExit(
            f"{MARK} {name}: mesh {mesh} sha256 {h[:16]}… ≠ declared {sha256[:16]}… (the pose fits those vertices only)"
        )
    return f


def _mesh(part):
    """(vertices float32, faces int64) of the declared mesh file — dies when it is missing or not the declared bytes."""
    return _mesh_file(part.name, part.mesh, part.mesh_sha256)


@functools.lru_cache(maxsize=8)
def _mesh_file(name: str, mesh: str, sha256: str):
    import trimesh

    m = trimesh.load(str(_checked_file(name, mesh, sha256)), force="mesh")
    return m.vertices.astype("float32"), m.faces.astype("int64")


def urdf_link(side: str, part) -> str:
    """The part's link in the render URDF — ``<side>_<name>_link`` (the adapter link's name)."""
    return f"{side}_{part.name}_link"


def urdf_joint(side: str, part) -> str:
    """The fixed joint EE link → ``urdf_link`` — ``<side>_<name>_joint``."""
    return f"{side}_{part.name}_joint"


def render_links_in_urdf(urdf) -> list[str]:
    """Links of the URDF file that are a declared render part of any registered pair, on any side of its arm — the
    bake's guard (``dexengine.rl.assets.urdf_to_usd``): such a URDF is a render URDF, never a bake input."""
    import xml.etree.ElementTree as ET

    _P, AS, MS = _registries()
    names = {
        urdf_link(side, part)
        for (arm, _h), spec in MS.MOUNT_REGISTRY.items()
        if spec is not None
        for part in spec.render_visual_parts
        for side in AS.ARM_REGISTRY[arm].sides
    }
    return sorted(n for n in (ln.get("name") for ln in ET.parse(str(urdf)).getroot().findall("link")) if n in names)


def single_user_prototype(stage, path: str, mark: str):
    """The prototype prim an instanceable ``path`` (a link's ``visuals``/``collisions``) internally references — and
    that prototype must have no other user (an edit there must not reach another link). Dies otherwise."""
    lay = stage.GetRootLayer()
    spec = lay.GetPrimAtPath(path)
    refs = list(spec.referenceList.GetAddedOrExplicitItems()) if spec else []
    if len(refs) != 1 or refs[0].assetPath:
        raise SystemExit(f"{mark} {path}: internal reference 1 expected, got {refs}")
    target, users = refs[0].primPath, []

    def visit(p):
        if any(r.primPath == target for r in p.referenceList.GetAddedOrExplicitItems()):
            users.append(str(p.path))
        for c in p.nameChildren:
            visit(c)

    for top in lay.rootPrims:
        visit(top)
    if len(users) != 1:
        raise SystemExit(f"{mark} prototype {target} is referenced by {users} — not {path}'s alone")
    return stage.GetPrimAtPath(target)


def _visual_protos(stage, arm: str) -> list:
    """(EE link, its visual prototype prim) per side — the link's ``visuals`` must carry no transform of its own (the
    part pose is stated in the link frame)."""
    import numpy as np

    from pxr import Usd, UsdGeom

    root, out = stage.GetDefaultPrim().GetPath(), []
    for ln in ee_links(arm):
        vis = stage.GetPrimAtPath(root.AppendPath(f"{ln}/visuals"))
        if not vis:
            raise InputMissingError(f"{MARK} {root}/{ln}/visuals is missing")
        T = np.array(UsdGeom.Xformable(vis).GetLocalTransformation(Usd.TimeCode.Default()))
        if np.max(np.abs(T - np.eye(4))) > POSE_TOL:
            raise SystemExit(f"{MARK} {vis.GetPath()} has its own transform — part poses are stated in the link frame")
        out.append((ln, single_user_prototype(stage, str(vis.GetPath()), MARK)))
    return out


def _xform(stage, path, pos, quat_wxyz):
    from pxr import Gf, UsdGeom

    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, pos)))
    x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*map(float, quat_wxyz)))
    x.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))
    return x


def _material_name(part) -> str:
    return "".join(w.capitalize() for w in part.name.split("_"))


def add(stage, arm: str, hand: str) -> list[str]:
    """The generator's edit (render body only): every declared part of (arm, hand) on every side's EE link visual
    prototype. Returns the marker lines ([] when the pair declares none — the stage is not touched)."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    parts = parts_of(arm, hand)
    if not parts:
        return []
    lines = []
    for part in parts:
        v, f = _mesh(part)
        for ln, proto in _visual_protos(stage, arm):
            if proto.GetChild(part.name):
                raise SystemExit(f"{MARK} {proto.GetPath()}/{part.name} already exists — the body is not a fresh copy")
            spec_before = stage.GetRootLayer().GetPrimAtPath(proto.GetPath()).specifier
            top = _xform(stage, proto.GetPath().AppendChild(part.name), part.ee_to_part_pos, part.ee_to_part_quat)
            vis = _xform(stage, top.GetPath().AppendChild(VISUAL_CHILD), part.part_to_mesh_pos, part.part_to_mesh_quat)
            m = UsdGeom.Mesh.Define(stage, vis.GetPath().AppendChild("mesh"))
            m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
            m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(f)))
            m.CreateFaceVertexIndicesAttr(Vt.IntArray(f.reshape(-1).tolist()))
            m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, v.min(0))), Gf.Vec3f(*map(float, v.max(0)))]))
            mat = UsdShade.Material.Define(stage, top.GetPath().AppendPath(f"Looks/{_material_name(part)}"))
            sh = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("Shader"))
            sh.SetSourceAsset("OmniPBR.mdl", "mdl")
            sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
            sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*part.color))
            sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(part.metallic))
            sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(part.roughness))
            outp = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
            for o in (
                mat.CreateSurfaceOutput("mdl"),
                mat.CreateDisplacementOutput("mdl"),
                mat.CreateVolumeOutput("mdl"),
            ):
                o.ConnectToSource(outp)
            UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
            # Define() turns an undefined `over` ancestor into `def` — the prototype source stays what it was
            stage.GetRootLayer().GetPrimAtPath(proto.GetPath()).specifier = spec_before
            lines.append(
                f"{MARK} {hand} {Path(part.mesh).stem} · {ln} → {proto.GetPath()}/{part.name} "
                f"at {tuple(part.ee_to_part_pos)} q={tuple(part.ee_to_part_quat)} · mesh {len(v)}v/{len(f)}f "
                f"· no collider, mass or joint"
            )
    return lines


def problems(stage, arm: str, hand: str) -> list[str]:
    """What keeps a render body/plate from the declaration (empty = it holds): per side, exactly the declared parts
    under the EE link's visual prototype — no other non-original root is checked here (``render_plate.parity`` owns
    'nothing else differs') — each at its declared poses (translate · orient · unit scale), its mesh = the file's
    vertices, and no physics API (collider · rigid body · mass · articulation) or joint anywhere below it."""
    import numpy as np

    from pxr import Usd, UsdGeom, UsdPhysics

    parts, out = parts_of(arm, hand), []
    phys = (UsdPhysics.CollisionAPI, UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI, UsdPhysics.ArticulationRootAPI)

    def pose_bad(prim, pos, quat):
        ops = {op.GetOpName(): op.Get() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
        t, q, s = ops.get("xformOp:translate"), ops.get("xformOp:orient"), ops.get("xformOp:scale")
        qa = None if q is None else np.array([q.GetReal(), *q.GetImaginary()])
        return [
            w
            for w, bad in (
                ("translate", t is None or np.max(np.abs(np.array(t) - np.array(pos))) > POSE_TOL),
                ("orient", qa is None or np.max(np.abs(qa - np.array(quat))) > POSE_TOL),
                ("scale", s is None or np.max(np.abs(np.array(s) - 1.0)) > POSE_TOL),
            )
            if bad
        ]

    for ln, proto in _visual_protos(stage, arm):
        for part in parts:
            top = proto.GetChild(part.name)
            if not top:
                out.append(f"{ln}: {part.name} missing")
                continue
            vis = top.GetChild(VISUAL_CHILD)
            bad = pose_bad(top, part.ee_to_part_pos, part.ee_to_part_quat)
            bad += [
                f"{VISUAL_CHILD} {w}"
                for w in (pose_bad(vis, part.part_to_mesh_pos, part.part_to_mesh_quat) if vis else ["missing"])
            ]
            meshes = [p for p in Usd.PrimRange(top, Usd.PrimAllPrimsPredicate) if p.IsA(UsdGeom.Mesh)]
            v, _f = _mesh(part)
            pts = np.asarray(meshes[0].GetAttribute("points").Get() or [], float) if len(meshes) == 1 else None
            if pts is None or pts.shape != v.shape or np.max(np.abs(pts - v)) > POSE_TOL:
                bad.append(f"mesh ≠ {part.mesh} ({len(meshes)} meshes)")
            if any(
                p.IsA(UsdPhysics.Joint) or any(p.HasAPI(a) for a in phys)
                for p in Usd.PrimRange(top, Usd.PrimAllPrimsPredicate)
            ):
                bad.append("carries a physics API or joint")
            out += [f"{ln}/{part.name}: {b}" for b in bad]
    return out
