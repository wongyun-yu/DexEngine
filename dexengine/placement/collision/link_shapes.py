#!/usr/bin/env python3
"""Shapes and frames of the collision sets (cross_sets): the robot link shapes (load_urdf_geometry — arm and torso
links = the sim colliders of sim_collider, other links = URDF <visual> | <collision>), the link groups (group_of), the
scene.xml object links (xml_links · artic_part_pose) and the quaternion / tree ↔ world frame helpers.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np

from dexengine.placement.collision import geom_cache as GC

#: robot shapes (load_urdf_geometry): visual · collision = the checks (arm · torso = sim colliders) | render = a picture
ROBOT_GEOMS = ("visual", "collision", "render")
RZ_NEG90 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # Rz(-90°): tree → world
RZ_POS90 = RZ_NEG90.T  # world → tree
MESH_CACHE = GC.cache_dir()  # the geometry cache directory (owner geom_cache — env HH_MESH_CACHE)


def desk_height() -> float:
    """The desk top height (m) of the scene frame (owner rl.tasks.cfg.constants.DESK_HEIGHT)."""
    from dexengine.rl.tasks.cfg import constants as SCN

    return float(SCN.DESK_HEIGHT)


def _mat_from_quat_wxyz(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _axis_angle_mat(axis: np.ndarray, th: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / max(np.linalg.norm(a), 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


_BODY_ORIENT_UNREAD = ("euler", "axisangle", "xyaxes", "zaxis")


def xml_links(body, det: bool = False) -> list:
    """A main object body (scene.xml) → its links [(link body element, [(welded body element, pos (3,), R (3,3))])],
    the body itself first, then each child body that carries a joint (an articulated part). A child body with no joint
    has no degree of freedom against its parent (MuJoCo) — it is one rigid body with it, so it is a component of that
    parent's link, posed in the link frame (the fixed attachments the converter welds: task_info
    articulation.parts.fixed · fixed_merge). The rule is the scene's structure, not a name. A joint below a part link
    or below a welded body is a second joint level, and an orientation attribute other than quat cannot be read here —
    both die. det=True (the cross repair's model) composes the frames with det_math (the same bits on every host); the
    default keeps the bits stage 1 (retarget f0) reads."""
    from dexengine.placement.collision import det_math as D

    def below(el, p0, R0, top: bool):
        welded, parts = [], []
        for c in el.findall("body"):
            if c.findall("joint") or c.findall("freejoint"):
                if not top:
                    raise ValueError(
                        f"scene.xml body {c.get('name')} — a joint below a part link · a welded part (a "
                        "second joint level is not supported)"
                    )
                parts.append(c)
                continue
            odd = [k for k in _BODY_ORIENT_UNREAD if c.get(k) is not None]
            if odd:
                raise ValueError(
                    f"scene.xml welded body {c.get('name')} orientation attributes {odd} — only pos · quat are read"
                )
            q = np.array([float(x) for x in c.get("quat", "1 0 0 0").split()], float)
            pos = np.array([float(x) for x in c.get("pos", "0 0 0").split()], float)
            if det:
                pc = p0 + D.matvec3(R0, pos)
                Rc = D.matmul3(R0, _mat_from_quat_wxyz(q / max(float(D.norm(q)), 1e-12)))
            else:
                pc = p0 + R0 @ pos
                Rc = R0 @ _mat_from_quat_wxyz(q / max(np.linalg.norm(q), 1e-12))
            welded += [(c, pc, Rc)] + below(c, pc, Rc, False)[0]
        return welded, parts

    welded, parts = below(body, np.zeros(3), np.eye(3), True)
    return [(body, welded)] + [(c, below(c, np.zeros(3), np.eye(3), False)[0]) for c in parts]


def scene_meshes(sx, assets_root) -> tuple:
    """(mesh root, {mesh name: file}) of a parsed scene.xml: an absolute compiler meshdir is the root, a relative one
    uses assets_root (the dataset's assets layer — the stage's --assets-root)."""
    comp = sx.find("compiler")
    md = comp.get("meshdir") if comp is not None else None
    mesh_root = Path(md) if md and os.path.isabs(md) else assets_root
    return mesh_root, {m.get("name"): m.get("file") for m in sx.find("asset").findall("mesh")}


def geom_files(el, mfile: dict, mesh_root) -> list:
    """The mesh files of a body's geoms (relative to mesh_root). A geom without a mesh, a missing root or a missing
    file dies — a body whose shape cannot be built."""
    fs = [mfile.get(g.get("mesh")) for g in el.findall("geom")]
    if not fs or None in fs or mesh_root is None or not all((Path(mesh_root) / f).is_file() for f in fs):
        raise ValueError(f"{el.get('name')}: mesh geoms {fs} · root {mesh_root} — cannot build the shape")
    return fs


def part_spec(child, parent: str, tree_name: str, det: bool = False) -> dict:
    """An articulated part (a scene.xml child body with one hinge|slide joint) → its joint FK spec for
    artic_part_pose: {name, parent, joint, type, axis (unit), anchor, b_pos, b_R}. Several joints or another joint type
    die. det=True (the cross repair's model): the unit vectors are normalised with det_math's norm (the same bits on
    every host); the default keeps the bits stage 1 reads."""
    from dexengine.placement.collision import det_math as D

    cn = child.get("name")
    js = child.findall("joint")
    if len(js) != 1:
        raise ValueError(
            f"{tree_name}: part body {cn} has {len(js)} joints (must be 1 — multi-joint parts are not supported)"
        )
    j = js[0]
    jt = j.get("type", "hinge")
    if jt not in ("hinge", "slide"):
        raise ValueError(f"{tree_name}: part joint {j.get('name')} type={jt} — hinge|slide only")
    bq = np.array([float(x) for x in child.get("quat", "1 0 0 0").split()], float)
    ax = np.array([float(x) for x in j.get("axis", "0 0 1").split()], float)
    nrm = (lambda v: float(D.norm(v))) if det else np.linalg.norm
    return {
        "name": cn,
        "parent": parent,
        "joint": j.get("name"),
        "type": jt,
        "axis": ax / max(nrm(ax), 1e-12),
        "anchor": np.array([float(x) for x in j.get("pos", "0 0 0").split()], float),
        "b_pos": np.array([float(x) for x in child.get("pos", "0 0 0").split()], float),
        "b_R": _mat_from_quat_wxyz(bq / max(nrm(bq), 1e-12)),
    }


def artic_part_pose(p_par: np.ndarray, R_par: np.ndarray, part: dict, theta: float, det: bool = False):
    """The world pose of an articulated part (a child body) — **the single owner of the joint FK**. Exactly the MuJoCo
    child-body convention: child frame = parent ∘ (b_pos, b_R); hinge rotates by θ about the axis through the child
    frame's anchor, slide translates by axis·θ.
    A point x (child frame) → world: p_par + R_par (b_pos + b_R (R_j x + (anchor − R_j anchor)))  → (R_par b_R R_j, …);
    it matches the xpos/xquat of a MuJoCo FK on the same scene.xml.
    det=True evaluates the same formula with det_math (fixed-order element-wise arithmetic, its own sincos — the bits
    are the same on every host): the cross-repair decision path (CrossSetModel.mocap_from_links). The default keeps
    the numpy/BLAS arithmetic that the f0 object repair (retarget/f0/repair_obj) uses, so its results stay
    the same."""
    b_pos, b_R, a = part["b_pos"], part["b_R"], part["anchor"]
    if det:
        from dexengine.placement.collision import det_math as D

        b_pos, b_R, a = (np.asarray(x, float) for x in (b_pos, b_R, a))
        ax = np.asarray(part["axis"], float)
        if part["type"] == "hinge":
            s_, c_ = D.sincos(float(theta))
            Rj = D.rot_axis(ax / max(float(D.norm(ax)), 1e-12), s_, c_)
            R_c = D.matmul3(D.matmul3(R_par, b_R), Rj)
            p_c = p_par + D.matvec3(R_par, b_pos + D.matvec3(b_R, a - D.matvec3(Rj, a)))
        else:  # slide
            R_c = D.matmul3(R_par, b_R)
            p_c = p_par + D.matvec3(R_par, b_pos + D.matvec3(b_R, ax * float(theta)))
        return p_c, R_c
    if part["type"] == "hinge":
        Rj = _axis_angle_mat(part["axis"], theta)
        R_c = R_par @ b_R @ Rj
        p_c = p_par + R_par @ (b_pos + b_R @ (a - Rj @ a))
    else:  # slide
        R_c = R_par @ b_R
        p_c = p_par + R_par @ (b_pos + b_R @ (part["axis"] * theta))
    return p_c, R_c


def mat2quat_wxyz_batch(R: np.ndarray) -> np.ndarray:
    """Rotation matrices → unit quaternions wxyz (Shepperd's four branches); the normalisation is the pairwise sum
    ((q0²+q2²)+(q1²+q3²)) a 1-D norm (ddot) uses, so it is bit-equal to the scalar form it replaced (np.linalg.norm of
    the per-matrix branch — measured on 22,000 random and 2,000 near-degenerate rotations). R (N,3,3) → (N,4)."""
    t = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    c0 = t > 0
    c1 = ~c0 & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    c2 = ~c0 & ~c1 & (R[:, 1, 1] > R[:, 2, 2])
    with np.errstate(all="ignore"):
        s = np.sqrt(t + 1.0) * 2
        q0 = np.stack(
            [0.25 * s, (R[:, 2, 1] - R[:, 1, 2]) / s, (R[:, 0, 2] - R[:, 2, 0]) / s, (R[:, 1, 0] - R[:, 0, 1]) / s], 1
        )
        s = np.sqrt(1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]) * 2
        q1 = np.stack(
            [(R[:, 2, 1] - R[:, 1, 2]) / s, 0.25 * s, (R[:, 0, 1] + R[:, 1, 0]) / s, (R[:, 0, 2] + R[:, 2, 0]) / s], 1
        )
        s = np.sqrt(1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2]) * 2
        q2 = np.stack(
            [(R[:, 0, 2] - R[:, 2, 0]) / s, (R[:, 0, 1] + R[:, 1, 0]) / s, 0.25 * s, (R[:, 1, 2] + R[:, 2, 1]) / s], 1
        )
        s = np.sqrt(1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1]) * 2
        q3 = np.stack(
            [(R[:, 1, 0] - R[:, 0, 1]) / s, (R[:, 0, 2] + R[:, 2, 0]) / s, (R[:, 1, 2] + R[:, 2, 1]) / s, 0.25 * s], 1
        )
    q = np.where(c0[:, None], q0, np.where(c1[:, None], q1, np.where(c2[:, None], q2, q3)))
    n = np.sqrt((q[:, 0] * q[:, 0] + q[:, 2] * q[:, 2]) + (q[:, 1] * q[:, 1] + q[:, 3] * q[:, 3]))
    return q / n[:, None]


def quat_mul_wxyz(a, b):
    """Quaternion product a ⊗ b (wxyz)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


#: Side → (arm group, hand group) of the placement link groups.
SIDE_GROUPS = {"right": ("RA", "RH"), "left": ("LA", "LH")}


def group_of(link: str, arm) -> str | None:
    """Link of a merged body → RA/RH/LA/LH/TORSO, by the rules of ``arm`` (its ArmSpec; None = a body without an arm):
    an arm link (ArmSpec.body_side) is in its side's arm group, except the EE link (where Isaac merges the palm hull),
    which is in the hand group; a hand link — every merged hand link starts with its side (merge.hand_subtree) — is in
    its side's hand group; the torso links (ArmSpec.is_torso_link) are TORSO; anything else None."""
    side = arm.body_side(link) if arm is not None else None
    if side is not None:
        return SIDE_GROUPS[side][link == arm.ee_body_name(side)]
    for s, (_a, h) in SIDE_GROUPS.items():
        if link.startswith(f"{s}_"):
            return h
    if arm is not None and arm.is_torso_link(link):
        return "TORSO"
    return None


def urdf_arm(urdf):
    """The ArmSpec of the arm a merged URDF carries (sim_collider.arm_of), None for a body without an arm."""
    from dexengine.core import arms as reg
    from dexengine.placement.collision import sim_collider as SC

    name = SC.arm_of(urdf)
    return reg.ARM_REGISTRY[name] if name is not None else None


def as_stl(path: Path) -> str:
    """Formats MuJoCo cannot read, such as DAE, are converted to STL by trimesh (cached, atomic rename). The cache key
    is a crc32, which is **stable across processes** (`hash(str)` varies per process with PYTHONHASHSEED)."""
    if path.suffix.lower() in (".stl", ".obj", ".msh"):
        return str(path)
    MESH_CACHE.mkdir(parents=True, exist_ok=True)
    dst = MESH_CACHE / (path.stem + f"_{zlib.crc32(str(path).encode()) & 0xFFFFFFFF:08x}.stl")
    if not dst.is_file():
        import trimesh

        m = trimesh.load(str(path), force="mesh")
        tmp = dst.with_suffix(f".{os.getpid()}.tmp.stl")
        m.export(str(tmp), file_type="stl")
        os.replace(tmp, dst)
    return str(dst)


_URDF_ONCE: dict = {}  # (what, urdf, mtime, size, geom) → a clip-independent URDF value (pin model · robot shapes)
_SERVER_KEEP = ("_URDF_ONCE",)  # kept by the resident bake worker between clips (clip-independent)


def _urdf_once(what: str, urdf: Path, geom: str, build):
    """CrossSetModel's clip-independent half (the pinocchio model · the robot shapes) built once per process per URDF
    file (path · mtime · size) and geometry mode — the class contract 'the URDF once, the objects per tree'. The shapes
    also follow the arm registry (the collider revision a run may swap) and the mesh cache directory, so both are in
    the key. Readers never mutate them (each model makes its own pinocchio data)."""
    from dexengine.core import arms as reg

    st = Path(urdf).stat()
    k = (what, str(urdf), st.st_mtime_ns, st.st_size, geom) + (
        ()
        if what == "pin"
        else (str(MESH_CACHE), repr(sorted(reg.ARM_SIM_COLLIDER_REV.items())), repr(sorted(reg.ARM_REGISTRY.items())))
    )
    if k not in _URDF_ONCE:
        _URDF_ONCE[k] = build()
    return _URDF_ONCE[k]


def load_urdf_geometry(urdf: Path, mode: str = "visual"):
    """(links: link→list of shapes, parent: child→parent). mode ∈ ROBOT_GEOMS. visual · collision (the checks): arm and
    torso links = the colliders the sim spawns (sim_collider); other links the URDF <visual> meshes | <collision>
    bodies. render (a picture only — never a check): every link the URDF <visual> meshes, arm and torso included."""
    from dexengine.core import paths as P
    from dexengine.placement.collision import sim_collider as SC
    from dexengine.robots import wrist_hull as WG

    if mode not in ROBOT_GEOMS:
        raise ValueError(f"mode={mode!r} — only {ROBOT_GEOMS} are known")
    root = ET.parse(str(urdf)).getroot()
    arm = {} if mode == "render" else SC.shape_dicts(urdf, MESH_CACHE)
    spec = urdf_arm(urdf)
    links: dict[str, list[dict]] = {ln: v for ln, v in arm.items() if v}
    for link in root.findall("link"):
        ln = link.get("name")
        if group_of(ln, spec) is None or ln in arm:
            continue
        for el in link.findall("collision" if mode == "collision" else "visual"):
            g = el.find("geometry")
            kinds = [c.tag for c in (g if g is not None else ())]
            if kinds not in (["capsule"], ["mesh"]):
                raise ValueError(f"{urdf.name}: {ln} <{el.tag}> shape {kinds} — exactly one capsule|mesh")
            o, sh = el.find("origin"), g[0]
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            R = WG.rpy_matrix((o.get("rpy", "0 0 0") if o is not None else "0 0 0").split())
            if sh.tag == "capsule":
                links.setdefault(ln, []).append(
                    {"kind": "capsule", "r": float(sh.get("radius")), "L": float(sh.get("length")), "xyz": xyz, "R": R}
                )
                continue
            fn, sc = P.urdf_mesh(urdf, sh.get("filename")), sh.get("scale")
            if not fn.is_file():  # e.g. a data root without the hand stage — dies, never drops a link
                raise FileNotFoundError(f"{urdf.name}: {ln} <{el.tag}> mesh file missing: {fn}")
            links.setdefault(ln, []).append(
                {
                    "kind": "mesh",
                    "file": as_stl(fn),
                    "xyz": xyz,
                    "R": R,
                    "scale": [float(v) for v in sc.split()] if sc else [1, 1, 1],
                }
            )
    parent = {j.find("child").get("link"): j.find("parent").get("link") for j in root.findall("joint")}
    return links, parent


def _adjacent(a: str, b: str, parent: dict) -> bool:
    """a and b are within 3 generations of URDF parent-child ancestry (excluded from set ④)."""

    def anc(x):
        out = []
        cur = parent.get(x)
        while cur is not None:
            out.append(cur)
            cur = parent.get(cur)
        return out

    return b in anc(a)[:3] or a in anc(b)[:3]
