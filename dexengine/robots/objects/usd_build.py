"""pxr builders of the object USD bake (``usd_bake.py``): rigid and articulated object USDs with canonical convex
collision parts. Import only after the Isaac app has started."""

from __future__ import annotations

import os

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

from dexengine.robots.objects.bake_owners import _decompose, _taco_paths

#: Coulomb friction effort (static = dynamic) of articulated joints — the single owner of the value (this asset
#: attribute is baked here; damping, torque and velocity limits are owned by the runtime ARTIC_JOINT_* settings, and the
#: runtime actuator friction is None, so this value is used as is). Units: hinge N·m, slide N. Basis: the effort
#: needed to hold the moving part against gravity under the runtime mass model (0.2 kg total, part USD mass ratio) over
#: the axis hand's 69 specs (hinge 62: p50 0.028 · p90 0.098 · max 0.31 N·m / slide 7: p50 0.71 · p90 1.43 · max
#: 1.72 N), p90 rounded up. Without friction a lid the hand lets go falls over under its own weight and inertia
#: (sim θ > ref θ). It stays below fingertip force 2–5 N × lever 0.05–0.15 m (0.1–0.75 N·m), so the hand can open it.
ARTIC_JOINT_FRICTION = {"hinge": 0.1, "slide": 1.5}

#: Instance name of the PhysX per-axis joint API (joint kind → axis token): angular for hinges, linear for slides.
_FRICTION_AXIS = {"hinge": "angular", "slide": "linear"}


def _apply_joint_friction(joint_prim, jtype: str) -> float:
    """Put the static and dynamic friction effort of PhysxJointAxisAPI (axis token) on the joint prim and prove it by
    reading it back. Returns the value set."""
    eff = float(ARTIC_JOINT_FRICTION[jtype])
    inst = _FRICTION_AXIS[jtype]
    if not joint_prim.ApplyAPI("PhysxJointAxisAPI", inst):
        raise RuntimeError(
            f"could not apply PhysxJointAxisAPI:{inst} to {joint_prim.GetPath()} — schema not registered"
        )
    for leaf in ("staticFrictionEffort", "dynamicFrictionEffort"):
        name = f"physxJointAxis:{inst}:{leaf}"
        joint_prim.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(eff)
        back = joint_prim.GetAttribute(name).Get()
        if back is None or abs(float(back) - eff) > 1e-6:
            raise RuntimeError(f"joint friction read-back failed {joint_prim.GetPath()}.{name}={back} ≠ {eff}")
    return eff


def _apply_self_collisions(root_prim, jtype: str) -> bool | None:
    """Write the self-collision value (between parts of the same articulation) on the articulation root and prove it
    by reading it back — value, kinds written and attribute name are owned by artifact_paths (ARTIC_SELF_COLLISIONS*).
    Returns the value written (None for kinds not written — output unchanged)."""
    tp = _taco_paths()
    want = tp.artic_self_collisions(jtype)
    if jtype not in tp.ARTIC_SELF_COLLISIONS_BAKED:
        return None
    if not root_prim.ApplyAPI(tp.ARTIC_SELF_COLLISIONS_API):
        raise RuntimeError(
            f"could not apply {tp.ARTIC_SELF_COLLISIONS_API} to {root_prim.GetPath()} — schema not registered"
        )
    root_prim.CreateAttribute(tp.ARTIC_SELF_COLLISIONS_ATTR, Sdf.ValueTypeNames.Bool).Set(want)
    back = root_prim.GetAttribute(tp.ARTIC_SELF_COLLISIONS_ATTR).Get()
    if back is None or bool(back) != want:
        raise RuntimeError(
            f"self-collision read-back failed {root_prim.GetPath()}.{tp.ARTIC_SELF_COLLISIONS_ATTR}={back} ≠ {want}"
        )
    return want


def _axis_to_local_rot(axis):
    """USD physics joints take the axis only as an X|Y|Z token. An arbitrary axis is expressed by rotating the joint
    frame: the minimal rotation taking Z to the target axis goes into both bodies' local rotations (there is no place to
    write the axis itself; this is exact, not an approximation)."""
    import numpy as _np

    a = _np.asarray(axis, float)
    n = _np.linalg.norm(a)
    if n < 1e-9:
        raise ValueError(f"the joint axis is the zero vector: {axis}")
    a = a / n
    z = _np.array([0.0, 0.0, 1.0])
    v = _np.cross(z, a)
    c = float(_np.dot(z, a))
    if _np.linalg.norm(v) < 1e-9:  # parallel / antiparallel
        q = Gf.Quatf(1.0, 0.0, 0.0, 0.0) if c > 0 else Gf.Quatf(0.0, 1.0, 0.0, 0.0)
        return q
    v = v / _np.linalg.norm(v)
    ang = _np.arccos(max(-1.0, min(1.0, c)))
    sh = _np.sin(ang / 2.0)
    return Gf.Quatf(float(_np.cos(ang / 2.0)), float(v[0] * sh), float(v[1] * sh), float(v[2] * sh))


def _add_collision_parts(stage, body_path: str, convex_dir: str, friction: float) -> int:
    """Add the parts of ``convex_dir`` (in numeric order) as collision prims ``<body>/collision_<i>``. Returns the part
    count."""
    import trimesh

    hull_files = _decompose().part_files(convex_dir)
    if not hull_files:
        raise RuntimeError(f"0 convex parts: {convex_dir}")
    for i, hull_file in enumerate(hull_files):
        hull_mesh = trimesh.load(os.path.join(convex_dir, hull_file), force="mesh", process=False)
        col_mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(f"{body_path}/collision_{i}"))
        col_mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*v) for v in hull_mesh.vertices.tolist()]))
        col_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(hull_mesh.faces)))
        col_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(hull_mesh.faces.flatten().tolist()))
        col_mesh.GetVisibilityAttr().Set("invisible")
        col_prim = col_mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(col_prim)
        UsdPhysics.MeshCollisionAPI.Apply(col_prim)
        UsdPhysics.MeshCollisionAPI(col_prim).GetApproximationAttr().Set("convexHull")
        UsdPhysics.MaterialAPI.Apply(col_prim)
        mat = UsdPhysics.MaterialAPI(col_prim)
        mat.GetStaticFrictionAttr().Set(friction)
        mat.GetDynamicFrictionAttr().Set(friction)
        mat.GetRestitutionAttr().Set(0.0)
    return len(hull_files)


def _part_body(
    stage,
    parent_path: str,
    name: str,
    mesh_path: str,
    convex_dir: str,
    density: float,
    mass_cap: float,
    friction: float,
    fixed=(),
):
    """One link = rigid body + visual mesh + convex collision (``convex_dir`` — checked by the caller with the convex
    check before the stage is opened). ``fixed`` = [(mesh, convex folder, link-frame pose {pos, quat_wxyz})] — parts
    fixed to this link (artifact_paths.artic_link_files) go under ``<link>/fixed_<i>`` transforms as visuals and
    colliders of the same rigid body (mass = sum of part volumes)."""
    import trimesh

    m = trimesh.load(mesh_path, force="mesh", process=False)
    fx = [(trimesh.load(fp, force="mesh", process=False), cvx, pose) for fp, cvx, pose in fixed]
    body_path = f"{parent_path}/{name}"
    xf = UsdGeom.Xform.Define(stage, Sdf.Path(body_path))
    prim = xf.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim)
    mass = min(density * (abs(m.volume) + sum(abs(fm.volume) for fm, _c, _p in fx)), mass_cap)
    UsdPhysics.MassAPI(prim).GetMassAttr().Set(mass)
    UsdPhysics.MassAPI(prim).GetDensityAttr().Set(density)

    def _vis(path, mm):
        vis = UsdGeom.Mesh.Define(stage, Sdf.Path(path))
        vis.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*v) for v in mm.vertices.tolist()]))
        vis.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(mm.faces)))
        vis.GetFaceVertexIndicesAttr().Set(Vt.IntArray(mm.faces.flatten().tolist()))

    _vis(f"{body_path}/visuals/mesh", m)
    n_parts = _add_collision_parts(stage, body_path, convex_dir, friction)
    for i, (fm, cvx, pose) in enumerate(fx):
        fpath = f"{body_path}/fixed_{i}"
        fxf = UsdGeom.Xform.Define(stage, Sdf.Path(fpath))
        w, x, y, z = (float(v) for v in pose["quat_wxyz"])
        fxf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*[float(v) for v in pose["pos"]]))
        fxf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)).GetNormalized())
        _vis(f"{fpath}/visuals/mesh", fm)
        n_parts += _add_collision_parts(stage, fpath, cvx, friction)
    return body_path, mass, n_parts


def _child_pose_frame(stage, moving_path: str, cp: dict | None, q, pos):
    """(rotation and position of joint frame 0) — with a child_pose the frame moves to that pose and moving_link is
    placed there. Without one (arctic — base frame = child frame) the joint frame stays and moving_link gets no
    transform (bytes unchanged)."""
    if not cp:
        return q, pos
    w, x, y, z = (float(v) for v in cp["quat_wxyz"])
    rc = Gf.Quatd(w, Gf.Vec3d(x, y, z)).GetNormalized()
    tc = Gf.Vec3d(*[float(v) for v in cp["pos"]])
    q0 = Gf.Quatf((rc * Gf.Quatd(q)).GetNormalized())
    pos0 = Gf.Vec3f(Gf.Rotation(rc).TransformDir(Gf.Vec3d(pos)) + tc)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(moving_path))
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(tc)
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(rc)
    return q0, pos0


def _define_joint(bpath, jtype, stage, mpath, spec):
    """The one revolute/prismatic joint between the two links, its frames (child_pose convention) and limits."""
    jp = Sdf.Path(f"{bpath}/joint")
    if jtype == "hinge":
        j = UsdPhysics.RevoluteJoint.Define(stage, jp)
    elif jtype == "slide":
        j = UsdPhysics.PrismaticJoint.Define(stage, jp)
    else:  # an unknown joint kind never becomes a silent fixed joint
        raise ValueError(f"unknown joint kind {jtype!r} — only hinge|slide are known")
    j.CreateBody0Rel().SetTargets([Sdf.Path(bpath)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(mpath)])
    j.CreateAxisAttr().Set("Z")  # an arbitrary axis is expressed by the local rotations below
    q = _axis_to_local_rot(spec["axis"])
    pos = Gf.Vec3f(*[float(x) for x in (spec.get("pos") or [0, 0, 0])])
    q0, pos0 = _child_pose_frame(stage, mpath, spec.get("child_pose"), q, pos)
    j.CreateLocalPos0Attr().Set(pos0)
    j.CreateLocalRot0Attr().Set(q0)
    j.CreateLocalPos1Attr().Set(pos)
    j.CreateLocalRot1Attr().Set(q)
    lo, hi = float(spec["lo"]), float(spec["hi"])
    if jtype == "hinge":  # USD revolute joint limits are in degrees (not radians)
        j.CreateLowerLimitAttr().Set(float(np.degrees(lo)))
        j.CreateUpperLimitAttr().Set(float(np.degrees(hi)))
    else:
        j.CreateLowerLimitAttr().Set(lo)
        j.CreateUpperLimitAttr().Set(hi)
    return j, lo, hi


def create_articulated_object_usd(
    mesh_dir: str,
    output_path: str,
    spec: dict,
    density: float = 200.0,
    mass_cap: float = 0.5,
    friction: float = 2.0,
    *,
    part_convex,
):
    """Articulated object USD — base part + moving part + one joint.

    `spec` is one asset entry of `artic_ranges.json`. Part names, axis, position, child_pose and limits are not made
    here — they are owned by the base layer's `task_info.objects[].articulation` and used as dexengine.data.artic.ranges
    collected them into the canonical joint. The limits are the asset's maximum opening over its clips (canonical
    coordinates). child_pose (the child body's pose in the base frame at θ=0 — convention owner
    coords.artic_moving_vertices, identity when absent) goes into joint frame 0 (base side) and the placed moving_link:
    J0 = child_pose ∘ (pos, q) · J1 = (pos, q) → at θ, moving_link = child_pose ∘ FK_joint(θ) (same convention).

    ``part_convex(mesh_path, name)`` returns a part's checked convex folder.
    """
    parts = spec["parts"]
    jtype = spec["type"]
    # link = main part + parts fixed to it (list owner artifact_paths.artic_link_files — reading only the main part
    # would silently drop levers etc.)
    links = {
        k: [(os.path.join(mesh_dir, rel), pose) for rel, pose in _taco_paths().artic_link_files(parts, k)]
        for k in ("base", "moving")
    }
    base_obj, move_obj = links["base"][0][0], links["moving"][0][0]
    for f, _pose in links["base"] + links["moving"]:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"articulated part mesh missing: {f}")
    # Check convex parts before the stage — a failure must not leave a half-written USD at the output
    base_cvx, move_cvx = part_convex(base_obj, "base_link"), part_convex(move_obj, "moving_link")
    fixed = {
        k: [(f, part_convex(f, f"{k}_link/fixed_{i}"), pose) for i, (f, pose) in enumerate(links[k][1:])] for k in links
    }
    stage = Usd.Stage.CreateNew(output_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = "/Object"
    rp = UsdGeom.Xform.Define(stage, Sdf.Path(root)).GetPrim()
    stage.SetDefaultPrim(rp)
    UsdPhysics.ArticulationRootAPI.Apply(rp)
    bpath, bmass, bn = _part_body(
        stage, root, "base_link", base_obj, base_cvx, density, mass_cap, friction, fixed["base"]
    )
    mpath, mmass, mn = _part_body(
        stage, root, "moving_link", move_obj, move_cvx, density, mass_cap, friction, fixed["moving"]
    )

    j, lo, hi = _define_joint(bpath, jtype, stage, mpath, spec)
    fric = _apply_joint_friction(j.GetPrim(), jtype)
    selfcoll = _apply_self_collisions(rp, jtype)
    stage.GetRootLayer().Save()
    print(
        f"[INFO] articulated USD: {output_path}\n"
        f"[INFO]   {jtype} axis={spec['axis']} pos={spec.get('pos')} child_pose={spec.get('child_pose')} "
        f"limits=[{lo:.4f},{hi:.4f}] "
        f"(rotation written in degrees) · base {bn} convex {bmass:.3f}kg · moving {mn} convex {mmass:.3f}kg",
        flush=True,
    )
    print(
        f"[ARTIC_FRICTION] {os.path.basename(os.path.dirname(output_path))} {jtype} "
        f"static=dynamic={fric} {'N·m' if jtype == 'hinge' else 'N'} (PhysxJointAxisAPI:{_FRICTION_AXIS[jtype]})",
        flush=True,
    )
    if selfcoll is not None:  # n = part (link) pairs the value covers · convex = both links' part counts
        print(
            f"[ARTIC_SELFCOLL] {os.path.basename(os.path.dirname(output_path))} kind={jtype} "
            f"{'on' if selfcoll else 'off'} n=1 (base_link↔moving_link · convex {bn}×{mn} · USD "
            f"{_taco_paths().ARTIC_SELF_COLLISIONS_ATTR}={selfcoll})",
            flush=True,
        )
    return output_path


def _apply_material_collision_and_save(
    mesh_path, output_path, root_path, stage, vis_mesh, trimesh, mesh, convex_dir, base_path, friction
):
    # Textures are not put on the training USD — the render cover carries them (dexengine/robots/objects/
    # render_overlay.py, rules in dexengine/robots/textures/rules.py).
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        # Fallback: use vertex colors as display color
        colors = mesh.visual.vertex_colors[:, :3].astype(float) / 255.0
        vis_mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*c) for c in colors.tolist()]))
        print("[INFO] Applied vertex colors")
    else:
        # Default gray
        vis_mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.6, 0.6, 0.6)]))
        print("[INFO] No texture/colors found, using default gray")

    # Collision = canonical convex parts only (checked by the caller via _ensure_convex — no inline CoACD, no
    # visual-hull fallback: an object without canonical parts is a failure, not a silent single hull).
    if not convex_dir:
        raise RuntimeError(f"no convex part folder: {mesh_path} — there is no inline decomposition")
    n_parts = _add_collision_parts(stage, base_path, convex_dir, friction)
    print(f"[INFO] Loaded {n_parts} canonical convex parts from {convex_dir}")

    # Save
    stage.GetRootLayer().Save()
    print(f"[INFO] Saved object USD: {output_path}")


def create_object_usd(
    mesh_path: str,
    output_path: str,
    density: float = 200.0,
    mass_cap: float = 0.5,
    friction: float = 2.0,
    convex_dir: str | None = None,
):
    """Create a rigid object USD from an OBJ mesh file.

    Args:
        mesh_path: Path to the .obj mesh file.
        output_path: Path to save the output .usd file.
        density: Object density in kg/m³.
        mass_cap: Maximum mass in kg.
        friction: Friction coefficient (static and dynamic).
        convex_dir: Directory with pre-computed convex hull OBJs (0.obj, 1.obj, ...). Skips CoACD.
    """
    import trimesh

    # Load mesh
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    print(f"[INFO] Loaded mesh: {mesh_path}")
    print(f"[INFO]   Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    print(f"[INFO]   Bounds: {mesh.bounds}")
    print(f"[INFO]   Volume: {mesh.volume:.6f} m³")

    # Compute mass (capped)
    mass = min(density * abs(mesh.volume), mass_cap)
    print(f"[INFO]   Mass: {mass:.4f} kg (density={density}, cap={mass_cap})")

    # Create USD stage
    stage = Usd.Stage.CreateNew(output_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Create root xform
    root_path = "/Object"
    root_xform = UsdGeom.Xform.Define(stage, Sdf.Path(root_path))
    root_prim = root_xform.GetPrim()
    stage.SetDefaultPrim(root_prim)

    # Create base_link
    base_path = f"{root_path}/base_link"
    base_xform = UsdGeom.Xform.Define(stage, Sdf.Path(base_path))
    base_prim = base_xform.GetPrim()

    # Apply rigid body API
    UsdPhysics.RigidBodyAPI.Apply(base_prim)

    # Apply mass properties
    UsdPhysics.MassAPI.Apply(base_prim)
    mass_api = UsdPhysics.MassAPI(base_prim)
    mass_api.GetMassAttr().Set(mass)
    mass_api.GetDensityAttr().Set(density)

    # Create visual mesh
    vis_path = f"{base_path}/visuals/mesh"
    vis_mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(vis_path))
    vis_mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*v) for v in mesh.vertices.tolist()]))
    vis_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(mesh.faces)))
    vis_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(mesh.faces.flatten().tolist()))

    # Apply visual material (texture or vertex colors)
    _apply_material_collision_and_save(
        mesh_path, output_path, root_path, stage, vis_mesh, trimesh, mesh, convex_dir, base_path, friction
    )
