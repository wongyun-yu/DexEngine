"""URDF kinematics and URDF → MJCF compilation for the hand-rig builder (``dexengine/robots/hand_rig.py``).

Only what is needed to place a site: zero-angle frames along the URDF chain, MuJoCo's own URDF compiler (with an
absolute meshdir injected into a temp copy, never into the vendored file) and the site resolver that hangs a site on
the nearest surviving MJCF body and cross-checks the placement against MuJoCo's forward kinematics.
"""

from __future__ import annotations

import pathlib
import re
import xml.etree.ElementTree as ET

import numpy as np

from dexengine.core.errors import InputMissingError


def _rpy_to_mat(rpy):
    """URDF rpy is extrinsic XYZ (fixed-axis), i.e. R = Rz(y) @ Ry(p) @ Rx(r)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _origin_of(joint):
    o = joint.find("origin")
    xyz = (
        np.array([float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()])
        if o is not None and o.get("xyz")
        else np.zeros(3)
    )
    rpy = np.array([float(v) for v in o.get("rpy").split()]) if o is not None and o.get("rpy") else np.zeros(3)
    T = np.eye(4)
    T[:3, :3] = _rpy_to_mat(rpy)
    T[:3, 3] = xyz
    return T


class Urdf:
    def __init__(self, path: pathlib.Path):
        self.root = ET.parse(path).getroot()
        self.links = {lk.get("name"): lk for lk in self.root.findall("link")}
        self.parent_joint = {}
        for j in self.root.findall("joint"):
            self.parent_joint[j.find("child").get("link")] = j
        roots = [n for n in self.links if n not in self.parent_joint]
        assert len(roots) == 1, f"expected one root link, got {roots}"
        self.root_link = roots[0]

    def chain_to_root(self, link: str) -> list[str]:
        out, cur = [link], link
        while cur in self.parent_joint:
            cur = self.parent_joint[cur].find("parent").get("link")
            out.append(cur)
        return out

    def transform(self, link: str, ancestor: str) -> np.ndarray:
        """4x4 of ``link``'s frame expressed in ``ancestor``'s, at zero joint angles."""
        T = np.eye(4)
        cur = link
        while cur != ancestor:
            j = self.parent_joint.get(cur)
            if j is None:
                raise ValueError(f"{ancestor!r} is not an ancestor of {link!r}")
            T = _origin_of(j) @ T
            cur = j.find("parent").get("link")
        return T

    def fk_world(self, link: str) -> np.ndarray:
        return self.transform(link, self.root_link)


#: Mesh formats MuJoCo can load. COLLADA (.dae) is NOT one of them, and ROS packages ship .dae
#: visuals constantly — DG-5F-S has 27 per side.
_MJ_MESH_OK = (".stl", ".obj", ".msh")


def _swap_unsupported_meshes(txt: str, meshdir: pathlib.Path, urdf: pathlib.Path) -> str:
    """Point unsupported mesh refs at a supported twin in the same directory.

    Only the temp copy is touched; the vendored URDF is never modified. A vendor that ships
    `link_1_1.dae` next to `link_1_1.STL` (DG-5F-S does, 27/27 on both sides) converts with no
    asset work at all.

    If a twin is missing this RAISES instead of dropping the mesh — a hand with a silently
    missing collision or visual geom is exactly the kind of wrong-but-running result this
    repository must not produce.
    """

    def repl(m):
        ref = m.group(1)
        stem, _, ext = ref.rpartition(".")
        if f".{ext}".lower() in _MJ_MESH_OK:
            return m.group(0)
        name = pathlib.PurePosixPath(ref).name.rsplit(".", 1)[0]
        for cand_ext in (".STL", ".stl", ".obj", ".OBJ", ".msh"):
            if (meshdir / f"{name}{cand_ext}").is_file():
                return m.group(0).replace(ref, str(pathlib.PurePosixPath(ref).parent / f"{name}{cand_ext}"))
        raise InputMissingError(
            f"[hand_rig] {urdf.name}: mesh {ref!r} cannot be read by MuJoCo and "
            f"{meshdir} has no twin in a supported format ({', '.join(_MJ_MESH_OK)}). "
            f"Convert the mesh under the same name or fix the URDF — it is not dropped silently here."
        )

    return re.sub(r'filename="([^"]+)"', repl, txt)


def compile_mjcf(urdf: pathlib.Path, meshdir: pathlib.Path, tmpdir: pathlib.Path):
    import mujoco

    txt = urdf.read_text()
    block = (
        f'<mujoco><compiler meshdir="{meshdir}" balanceinertia="true" '
        f'discardvisual="false" autolimits="true"/></mujoco>'
    )
    if "<mujoco>" in txt:
        # The vendor's own block usually carries a RELATIVE meshdir, which breaks
        # as soon as the file is read from anywhere else. Replace, do not append.
        txt = re.sub(r"[ \t]*<mujoco>.*?</mujoco>\s*", block, txt, count=1, flags=re.S)
    else:
        txt, n = re.subn(r"(<robot\b[^>]*>)", r"\1" + block, txt, count=1)
        if n != 1:
            raise ValueError(f"no <robot> element in {urdf}")

    txt = _swap_unsupported_meshes(txt, meshdir, urdf)

    tmpdir.mkdir(parents=True, exist_ok=True)
    tmp_urdf = tmpdir / f"{urdf.stem}_mj.urdf"
    tmp_urdf.write_text(txt)  # never modifies the vendored file
    model = mujoco.MjModel.from_xml_path(str(tmp_urdf))
    out_xml = tmpdir / f"{urdf.stem}_compiled.xml"
    mujoco.mj_saveLastXML(str(out_xml), model)
    return model, ET.parse(out_xml).getroot()


def body_world_pos(model, name: str) -> np.ndarray:
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise KeyError(name)
    return np.array(data.xpos[bid])


def resolve_site(urdf: Urdf, model, surviving: set[str], target_link: str, local_offset) -> tuple[str, np.ndarray]:
    """Where to hang a site so it lands on ``target_link``'s ``local_offset``.

    Returns (body that still exists in the MJCF, position in that body's frame).
    Raises if MuJoCo's own FK disagrees with the URDF chain, which is the whole
    point of doing it this way rather than trusting a hand-written table.
    """
    host = next((n for n in urdf.chain_to_root(target_link) if n in surviving), None)
    if host is None:
        raise ValueError(f"no surviving MJCF body on the chain from {target_link!r}")

    T = urdf.transform(target_link, host)
    pos_in_host = T[:3, :3] @ np.asarray(local_offset, float) + T[:3, 3]

    # Cross-check in world coordinates. MuJoCo's compiled body frame and the URDF
    # chain must agree, or the site is being placed against a frame this script
    # does not actually understand.
    Tw = urdf.fk_world(target_link)
    want_world = Tw[:3, :3] @ np.asarray(local_offset, float) + Tw[:3, 3]
    host_w = urdf.fk_world(host)
    got_world = host_w[:3, :3] @ pos_in_host + host_w[:3, 3]
    if not np.allclose(want_world, got_world, atol=1e-9):
        raise AssertionError(
            f"URDF chain composition is inconsistent for {target_link!r} (want {want_world}, got {got_world})"
        )

    # The synthesised root body has no MuJoCo counterpart to compare against; by
    # construction it sits at the compiled world origin, which is what fk_world
    # returns for the URDF root. Every other host is checked for real.
    if host != urdf.root_link:
        mj_host_world = body_world_pos(model, host)
        if not np.allclose(mj_host_world, host_w[:3, 3], atol=1e-6):
            raise AssertionError(
                f"MuJoCo body {host!r} sits at {mj_host_world}, URDF says {host_w[:3, 3]}. "
                "The compiled frame is not the URDF frame -- site offsets would be wrong."
            )
    return host, pos_in_host
