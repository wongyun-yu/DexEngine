"""Scene construction on top of a hand rig (`MjSpec`): world elements, object meshes and bodies, contact pairs.

Every element is added to the spec loaded from the copied rig file, so attributes left unset take the rig's default
class (geom density 800, condim 1, no collision filters). Element names are a contract; downstream
stages look them up by name.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import numpy as np

from dexengine.retarget.scene.objects import ObjectAssets, is_collision_stem
from dexengine.robots.objects import part_mass as PM

FINGERS = ("thumb", "index", "middle", "ring", "pinky")

OBJECT_DENSITY = 1000.0
OBJECT_ARMATURE = 1e-4
OBJECT_FRICTIONLOSS = 1e-4
HINGE_DAMPING = 0.01
GREEN = [0.0, 1.0, 0.0, 1.0]
WHITE = [1.0, 1.0, 1.0, 1.0]
RED = [1.0, 0.0, 0.0, 1.0]
#: colour of an extra object's marker and trace sites (the trace / track / ref sites of primary objects are green)
EXTRA_SITE_RGBA = [0.5, 0.5, 1.0, 1.0]
OBJECT_SITE_HALF = [0.01, 0.02, 0.03]
PAIR_SOLREF = [0.02, 1.0]
HAND_FRICTION = [1.0, 1.0, 0.1, 0.0, 0.0]
OBJECT_OBJECT_FRICTION = [0.01, 0.01, 0.0001, 0.0, 0.0]

G = mujoco.mjtGeom


@dataclass
class ObjectGeoms:
    """Names of the collision geoms of one object body (pair candidates), in creation order."""

    label: str
    collision: list[str]


def _sphere_site(body, name: str, radius: float, group: int, pos=None, rgba=None):
    site = body.add_site(name=name, type=G.mjGEOM_SPHERE, group=group)
    site.size[0] = radius
    if pos is not None:
        site.pos = [float(x) for x in pos]
    if rgba is not None:
        site.rgba = rgba
    return site


def add_world(spec: mujoco.MjSpec) -> None:
    """Shared textures and material, the floor plane and the `front` camera."""
    spec.add_texture(
        name="skybox",
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7],
        rgb2=[0.0, 0.0, 0.0],
        width=512,
        height=3072,
    )
    spec.add_texture(
        name="groundplane",
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=[0.2, 0.3, 0.4],
        rgb2=[0.1, 0.2, 0.3],
        markrgb=[0.8, 0.8, 0.8],
        width=300,
        height=300,
    )
    spec.add_material(name="groundplane", texuniform=True, texrepeat=[5.0, 5.0], reflectance=0.2)
    spec.worldbody.add_geom(
        name="floor", type=G.mjGEOM_PLANE, size=[0.0, 0.0, 0.05], pos=[0.0, 0.0, 0.0], material="groundplane"
    )
    spec.worldbody.add_camera(
        name="front",
        pos=[0.031, 0.941, 0.844],
        xyaxes=[-0.999, 0.033, 0.0, -0.022, -0.667, 0.745],
        mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
    )


class SceneAssets:
    """Registers object meshes / textures / materials and remembers which names exist."""

    def __init__(self, spec: mujoco.MjSpec, assets_dir: str):
        self.spec = spec
        self.assets_dir = assets_dir
        self.meshes: set[str] = set()
        self.materials: set[str] = set()
        #: materials whose texture binding is written by the post-fix of `write.fix_materials`
        self.material_textures: dict[str, str] = {}

    def rel(self, path: str) -> str:
        return os.path.relpath(path, self.assets_dir)

    def mesh(self, name: str, path: str) -> None:
        if name not in self.meshes:
            self.spec.add_mesh(name=name, file=self.rel(path))
            self.meshes.add(name)

    def textured_material(self, tex_name: str, mat_name: str, png: str) -> None:
        self.spec.add_texture(name=tex_name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=self.rel(png))
        self.spec.add_material(name=mat_name, texrepeat=[1.0, 1.0])
        self.materials.add(mat_name)
        self.material_textures[mat_name] = tex_name

    def primary_visual(self, side: str, obj: ObjectAssets) -> None:
        """Visual mesh, texture and articulated part meshes of one primary object."""
        if obj.visual_file is not None:
            self.mesh(f"{side}_visual", obj.visual_file)
        if obj.texture_file is not None:
            self.textured_material(f"{side}_object_tex", f"{side}_object_visual_mat", obj.texture_file)
        art = obj.articulation
        if art is not None:
            if art.bottom_file is not None:
                self.mesh(f"{side}_bottom", art.bottom_file)
            if art.top_file is not None:
                self.mesh(f"{side}_top", art.top_file)

    def primary_parts(self, side: str, obj: ObjectAssets) -> None:
        for stem, path in obj.parts:
            if stem != "visual":
                self.mesh(f"{side}_{stem}", path)

    def extra(self, i: int, obj: ObjectAssets) -> None:
        self.mesh(f"extra_object_{i}_visual", obj.visual_file)
        if obj.texture_file is not None:
            self.textured_material(f"extra_object_{i}_tex", f"extra_object_{i}_mat", obj.texture_file)
        for stem, path in obj.parts:
            if stem != "visual":
                self.mesh(f"extra_{i}_{stem}", path)


def _visual_geom(body, name: str, mesh: str, material: str | None) -> None:
    geom = body.add_geom(name=name, type=G.mjGEOM_MESH, meshname=mesh, density=0.0, group=0)
    if material is not None:
        geom.material = material
    else:
        geom.rgba = WHITE


def _collision_geom(body, name: str, mesh: str) -> None:
    body.add_geom(name=name, type=G.mjGEOM_MESH, meshname=mesh, density=OBJECT_DENSITY, group=3, rgba=GREEN)


def _mesh_mass(part_file: str) -> float:
    """MuJoCo's mass of one part mesh as a collision geom at OBJECT_DENSITY (mesh defaults, compiled alone — the
    mass the scene gave that geom before the split rule)."""
    s = mujoco.MjSpec()
    s.add_mesh(name="m", file=part_file)
    b = s.worldbody.add_body(name="b")
    b.add_freejoint(name="j")
    b.add_geom(name="g", type=G.mjGEOM_MESH, meshname="m", density=OBJECT_DENSITY)
    return float(s.compile().body_mass[1])


def artic_part_masses(name: str, files: dict) -> dict:
    """{part: mass} of an articulated object's parts ({part: mesh file}): the total stays the scene's total (the parts
    at OBJECT_DENSITY on their meshes, `_mesh_mass`) and is split by the part_mass volume rule. A missing part mesh
    dies. Marker [PART_MASS]."""
    gone = [k for k, f in files.items() if f is None]
    if gone:
        raise FileNotFoundError(f"{PM.TAG} {name}: articulated part mesh missing ({gone}) — no volume, no mass")
    vol = {k: PM.part_volume(f) for k, f in files.items()}
    total = sum(_mesh_mass(f) for f in files.values())
    mass = PM.split_by_volume(total, {k: v for k, (v, _) in vol.items()})
    print(
        f"{PM.TAG} {name}: "
        + " · ".join(f"{k} V={vol[k][0] * 1e6:.1f}cm3 ({vol[k][1]}) m={mass[k]:.4f}kg" for k in files)
        + f" · total={total:.4f}kg (scene total kept)",
        flush=True,
    )
    return mass


def _part_collision_geom(body, name: str, mesh: str, mass: float) -> None:
    """Collision geom of one articulated part carrying its share of the object mass (`artic_part_masses`)."""
    geom = body.add_geom(name=name, type=G.mjGEOM_MESH, meshname=mesh, density=OBJECT_DENSITY, group=3, rgba=GREEN)
    geom.mass = mass


def _free_joint(body, name: str) -> None:
    joint = body.add_freejoint(name=name)
    joint.armature = OBJECT_ARMATURE
    joint.frictionloss = OBJECT_FRICTIONLOSS


def add_extra_body(spec: mujoco.MjSpec, i: int, obj: ObjectAssets, assets: SceneAssets) -> ObjectGeoms:
    """Body `extra_object_<i>` with its free joint, part geoms, visual geom and sites."""
    base = f"extra_object_{i}"
    body = spec.worldbody.add_body(name=base)
    _free_joint(body, f"{base}_joint")
    collision = []
    for stem, _ in obj.parts:
        if stem == "visual":
            name, mesh = f"{base}_collision_visual", f"{base}_visual"
        else:
            name, mesh = f"{base}_{stem}", f"extra_{i}_{stem}"
        if is_collision_stem(stem):
            _collision_geom(body, name, mesh)
            collision.append(name)
        else:
            _visual_geom(body, name, mesh, None)
    mat = f"{base}_mat"
    _visual_geom(body, f"{base}_visual_geom", f"{base}_visual", mat if mat in assets.materials else None)
    body.add_site(name=base, type=G.mjGEOM_BOX, size=OBJECT_SITE_HALF, rgba=EXTRA_SITE_RGBA, group=3)
    _sphere_site(body, f"trace_{base}", 0.01, 4, rgba=EXTRA_SITE_RGBA)
    return ObjectGeoms(base, collision)


def add_primary_body(
    spec: mujoco.MjSpec, side: str, obj: ObjectAssets, assets: SceneAssets, contact_pos: np.ndarray
) -> ObjectGeoms:
    """Body `<side>_object` (+ hinged top for an articulated object), its sites, and the side's ref mocap bodies."""
    base = f"{side}_object"
    body = spec.worldbody.add_body(name=base)
    if side == "left":
        body.gravcomp = 0.0
    _free_joint(body, f"{base}_joint")
    collision = []
    art = obj.articulation
    if art is not None:
        pm = artic_part_masses(base, {"bottom": art.bottom_file, "top": art.top_file})
        _part_collision_geom(body, f"{base}_bottom_collision", f"{side}_bottom", pm["bottom"])
        _visual_geom(body, f"{base}_bottom_visual", f"{side}_bottom", None)
        top = body.add_body(name=f"{base}_top")
        top.add_joint(
            name=f"{base}_hinge",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            pos=art.pos,
            axis=art.axis,
            range=art.range,
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
            armature=OBJECT_ARMATURE,
            frictionloss=OBJECT_FRICTIONLOSS,
            damping=HINGE_DAMPING,
        )
        _part_collision_geom(top, f"{base}_top_collision", f"{side}_top", pm["top"])
        _visual_geom(top, f"{base}_top_visual", f"{side}_top", None)
        collision += [f"{base}_bottom_collision", f"{base}_top_collision"]
    else:
        for stem, _ in obj.parts:
            if stem == "visual":
                name, mesh = f"{base}_collision_visual", f"{side}_visual"
            else:
                name, mesh = f"{base}_{stem}", f"{side}_{stem}"
            if is_collision_stem(stem):
                _collision_geom(body, name, mesh)
                collision.append(name)
            else:
                _visual_geom(body, name, mesh, None)
        if f"{side}_visual" in assets.meshes:
            mat = f"{side}_object_visual_mat"
            _visual_geom(body, base + "_visual", f"{side}_visual", mat if mat in assets.materials else None)
    body.add_site(name=base, type=G.mjGEOM_BOX, size=OBJECT_SITE_HALF, rgba=RED, group=3)
    _sphere_site(body, f"trace_{base}", 0.01, 4, rgba=GREEN)
    for k, finger in enumerate(FINGERS):
        _sphere_site(body, f"track_object_{side}_{finger}_tip", 0.01, 4, pos=contact_pos[k], rgba=GREEN)
    for finger in FINGERS:
        for kind in ("object", "hand"):
            name = f"ref_{kind}_{side}_{finger}_tip"
            ref = spec.worldbody.add_body(name=name, mocap=True)
            _sphere_site(ref, name, 0.02, 4, rgba=GREEN)
    return ObjectGeoms(base, collision)


def _pair(spec: mujoco.MjSpec, g1: str, g2: str, friction: list[float], condim: int) -> None:
    spec.add_pair(name=f"{g1}_{g2}", geomname1=g1, geomname2=g2, condim=condim, friction=friction, solref=PAIR_SOLREF)


def hand_geoms(spec: mujoco.MjSpec) -> list[str]:
    """Every geom of the spec whose name starts with `collision_hand_`, in spec order (rig contract)."""
    return [g.name for g in spec.geoms if g.name.startswith("collision_hand_")]


def add_pairs(
    spec: mujoco.MjSpec, extras: list[ObjectGeoms], right: ObjectGeoms | None, left: ObjectGeoms | None, bimanual: bool
) -> None:
    """Explicit contact pairs; the only way any geom of the scene collides."""
    hands = hand_geoms(spec)
    primaries = [o for o in (right, left) if o is not None]
    for obj in extras + primaries:
        for og in obj.collision:
            for hg in hands + ["floor"]:
                condim = 4 if ("thumb" in hg or "index" in hg) else 3
                _pair(spec, hg, og, HAND_FRICTION, condim)
    if right is not None and left is not None:
        for rg in right.collision:
            for lg in left.collision:
                _pair(spec, rg, lg, OBJECT_OBJECT_FRICTION, 3)
    for i, extra in enumerate(extras):
        for eg in extra.collision:
            for obj in primaries:
                for pg in obj.collision:
                    _pair(spec, eg, pg, OBJECT_OBJECT_FRICTION, 3)
        for other in extras[i + 1 :]:
            for eg in extra.collision:
                for og in other.collision:
                    _pair(spec, eg, og, OBJECT_OBJECT_FRICTION, 3)
    _hand_hand_pairs(spec, hands, bimanual)


def _hand_hand_pairs(spec: mujoco.MjSpec, hands: list[str], bimanual: bool) -> None:
    """Palm/fingertip self-collision pairs; an unordered pair is added once."""

    def side_of(name: str) -> str:
        parts = name.split("_")
        return parts[2] if len(parts) > 2 else ""

    seen: set[frozenset[str]] = set()

    def add(g: str, h: str) -> None:
        key = frozenset((g, h))
        if g == h or key in seen:
            return
        seen.add(key)
        _pair(spec, g, h, HAND_FRICTION, 3)

    for g in hands:
        if "0" not in g:
            continue
        side = side_of(g)
        if bimanual:
            for h in hands:
                if side_of(h) != side and ("0" in h or "1" in h):
                    add(g, h)
        for h in hands:
            if h != g and side_of(h) == side and "0" in h:
                add(g, h)
