"""The MANO contact scene, built in code.

Twelve free hand bodies (two wrists, ten fingertips) carry a site and a non-colliding geom each; ten touch sensors
on the fingertip sites come first among the sensors. Two object bodies follow; their mesh geoms collide with the
fingertip spheres only through explicit pairs. Ten frame-position sensors give each fingertip in its object frame.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco

SIDES = ("right", "left")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
WRIST_HALF = [0.01, 0.02, 0.03]
TIP_RADIUS = 0.02
OBJECT_ARMATURE = 1e-4
OBJECT_FRICTIONLOSS = 1e-4
MASS_SPHERE_RADIUS = 0.01
MASS_SPHERE_POS = [1.0, 1.0, 1.0]
ARTICULATED_DEFAULTS = {
    "joint_axis": [0.0, 0.0, -1.0],
    "joint_range": [0.0, 3.14159],
    "bottom_mesh": "bottom.obj",
    "top_mesh": "top.obj",
}

G = mujoco.mjtGeom


def tip_names() -> list[str]:
    """The ten fingertip body names in fingertip order (right thumb … left pinky)."""
    return [f"{s}_{f}_tip" for s in SIDES for f in FINGERS]


def hand_body_names() -> list[str]:
    """The twelve hand bodies in body order: each side's wrist followed by its five tips."""
    return [n for s in SIDES for n in [f"{s}_wrist"] + [f"{s}_{f}_tip" for f in FINGERS]]


@dataclass
class ContactScene:
    model: mujoco.MjModel
    hinge_qadr: dict[str, int]  # side -> hinge qpos address (articulated objects only)
    object_geoms: dict[str, list[str]]  # side -> collision geom names
    left_ref: str  # reference body of the left fingertips' frame-position sensors


def _rigid_parts(convex_dir: str) -> list[tuple[str, str]]:
    """(stem, path) of the `<n>.obj` parts, numerically sorted; `visual.obj` alone when there is none."""
    names = [n for n in os.listdir(convex_dir) if n.endswith(".obj") and n[:-4].isdigit()]
    names.sort(key=lambda n: int(n[:-4]))
    if names:
        return [(n[:-4], os.path.join(convex_dir, n)) for n in names]
    visual = os.path.join(convex_dir, "visual.obj")
    return [("visual", visual)] if os.path.isfile(visual) else []


def _abs(dataset_dir: str, rel: str | None) -> str | None:
    return None if not rel else os.path.abspath(os.path.join(dataset_dir, rel))


def build_contact_scene(dataset_dir: str, task_info: dict | None, sides: tuple[str, ...]) -> ContactScene:
    """Compile the contact scene for one clip (`task_info=None`: no object meshes)."""
    ti = task_info or {}
    spec = mujoco.MjSpec()
    _add_hands(spec)
    art = ti.get("articulated_object") or None
    object_geoms: dict[str, list[str]] = {}
    for side in SIDES:
        convex_dir = _abs(dataset_dir, ti.get(f"{side}_object_convex_dir")) if side in sides else None
        object_geoms[side] = _add_object(spec, side, art, convex_dir)

    tips_in_emb = [t for t in tip_names() if t.split("_")[0] in sides]
    for side in SIDES:
        for og in object_geoms[side]:
            for tip in tips_in_emb:
                spec.add_pair(name=f"{tip}_{og}", geomname1=tip, geomname2=og)

    left_ref = "left_object" if object_geoms["left"] else "right_object"
    for tip in tip_names():
        ref = "right_object" if tip.startswith("right") else left_ref
        spec.add_sensor(
            name=f"pos_{tip}",
            type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
            objtype=mujoco.mjtObj.mjOBJ_XBODY,
            objname=tip,
            reftype=mujoco.mjtObj.mjOBJ_XBODY,
            refname=ref,
        )
    model = spec.compile()
    hinge_qadr = {}
    for side in SIDES:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_object_hinge")
        if j >= 0:
            hinge_qadr[side] = int(model.jnt_qposadr[j])
    return ContactScene(model, hinge_qadr, object_geoms, left_ref)


def _add_hands(spec: mujoco.MjSpec) -> None:
    """Ground, light, the twelve free hand bodies (box wrists, sphere tips) and the fingertip touch sensors."""
    spec.worldbody.add_geom(name="ground", type=G.mjGEOM_PLANE, size=[100.0, 100.0, 0.1], contype=0, conaffinity=0)
    spec.worldbody.add_light(name="light", pos=[0.0, 0.0, 3.0], dir=[0.0, 0.0, -1.0])
    for name in hand_body_names():
        body = spec.worldbody.add_body(name=name)
        body.add_freejoint(name=f"{name}_joint")
        if name.endswith("_wrist"):
            shape = {"type": G.mjGEOM_BOX, "size": WRIST_HALF}
        else:
            shape = {"type": G.mjGEOM_SPHERE, "size": [TIP_RADIUS, 0.0, 0.0]}
        body.add_site(name=name, **shape)
        body.add_geom(name=name, contype=0, conaffinity=0, group=0, **shape)
    for tip in tip_names():
        spec.add_sensor(
            name=f"touch_{tip}", type=mujoco.mjtSensor.mjSENS_TOUCH, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=tip
        )


def _add_object(spec: mujoco.MjSpec, side: str, art: dict | None, convex_dir: str | None) -> list[str]:
    """One side's object body on a free joint (with a hinged top for an articulated object); returns its collision
    geom names. A body without collision geoms gets a small mass sphere."""
    body = spec.worldbody.add_body(name=f"{side}_object")
    joint = body.add_freejoint(name=f"{side}_object_collision_joint")
    joint.armature = OBJECT_ARMATURE
    joint.frictionloss = OBJECT_FRICTIONLOSS
    is_art = bool(art) and art.get("side") in (side, "both")
    top = _add_hinged_top(body, side, art) if is_art else None
    geoms: list[str] = []
    if convex_dir is not None and os.path.isdir(convex_dir):
        if is_art:
            for part, holder, gname in (
                ("bottom_mesh", body, f"{side}_object_bottom"),
                ("top_mesh", top, f"{side}_object_top_geom"),
            ):
                path = os.path.join(convex_dir, art.get(part, ARTICULATED_DEFAULTS[part]))
                if os.path.isfile(path):
                    _add_mesh_geom(spec, holder, gname, path)
                    geoms.append(gname)
        else:
            for stem, path in _rigid_parts(convex_dir):
                gname = f"{side}_object_{stem}"
                _add_mesh_geom(spec, body, gname, path)
                geoms.append(gname)
    if not geoms:
        body.add_geom(
            name=f"{side}_object_mass",
            type=G.mjGEOM_SPHERE,
            size=[MASS_SPHERE_RADIUS, 0.0, 0.0],
            pos=MASS_SPHERE_POS,
            contype=0,
            conaffinity=0,
        )
    if top is not None and not any(g.endswith("_top_geom") for g in geoms):
        top.add_geom(
            name=f"{side}_object_top_mass",
            type=G.mjGEOM_SPHERE,
            size=[MASS_SPHERE_RADIUS, 0.0, 0.0],
            contype=0,
            conaffinity=0,
        )
    return geoms


def _add_hinged_top(body, side: str, art: dict):
    """The top part of an articulated object: a child body on a limited hinge."""
    top = body.add_body(name=f"{side}_object_top")
    top.add_joint(
        name=f"{side}_object_hinge",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[float(x) for x in art.get("joint_axis", ARTICULATED_DEFAULTS["joint_axis"])],
        range=[float(x) for x in art.get("joint_range", ARTICULATED_DEFAULTS["joint_range"])],
        limited=mujoco.mjtLimited.mjLIMITED_TRUE,
        armature=OBJECT_ARMATURE,
        frictionloss=OBJECT_FRICTIONLOSS,
    )
    return top


def _add_mesh_geom(spec: mujoco.MjSpec, holder, gname: str, path: str) -> None:
    """A mesh asset and its collision geom on `holder` (contacts come only from the explicit tip pairs)."""
    spec.add_mesh(name=f"{gname}_mesh", file=path)
    holder.add_geom(
        name=gname,
        type=G.mjGEOM_MESH,
        meshname=f"{gname}_mesh",
        group=0,
        condim=1,
        contype=0,
        conaffinity=0,
    )
