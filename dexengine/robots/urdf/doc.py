"""URDF surgery helpers of the merged-robot generator (``urdf/merge.py``): parse, cut subtrees, rebase mesh paths."""

from __future__ import annotations

import math
import os
import pathlib
import xml.etree.ElementTree as ET

import numpy as np

from dexengine.core.paths import main_root, urdf_mesh  # data root owner · URDF mesh resolver
from dexengine.core.repo import REPO_ROOT as REPO

#: Checkouts whose paths are re-rooted onto this one (this checkout, then the data root's checkout).
_CANON_ROOTS = tuple(dict.fromkeys((REPO, pathlib.Path(os.path.realpath(main_root())))))


def axis_angle_to_rpy(axis, angle):
    """Rotation about ``axis`` by ``angle`` as URDF rpy (extrinsic XYZ)."""
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / n, y / n, z / n
    c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    m = [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]
    pitch = math.atan2(-m[2][0], math.sqrt(m[0][0] ** 2 + m[1][0] ** 2))
    if abs(math.cos(pitch)) < 1e-9:
        return (math.atan2(-m[1][2], m[1][1]), pitch, 0.0)
    return (math.atan2(m[2][1], m[2][2]), pitch, math.atan2(m[1][0], m[0][0]))


def quat_wxyz_to_rpy(q):
    w, x, y, z = q
    # ZYX euler (URDF rpy is extrinsic XYZ == intrinsic ZYX)
    r = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sp = 2 * (w * y - z * x)
    p = np.arcsin(np.clip(sp, -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return float(r), float(p), float(yaw)


# URDF surgery helpers
def _canon(p: pathlib.Path) -> pathlib.Path:
    """Symlink-resolved path **re-rooted onto this checkout**.

    Resolving is right for links *inside* the repo (`Robots/full_robot/meshes -> ../../../meshes` — the canonical
    URDFs were generated through it). But a worktree reads the vendor trees *from the data root* (the canonical
    checkout — paths.urdf_mesh), and a plain `.resolve()` would write a path that climbs out of
    the worktree (`../` x11 instead of x8) — the output would depend on which tree ran it. So: resolve, then if the
    target lies under this checkout or under the canonical one (paths.main_root — the owner of "where the real
    data lives"), express it under this checkout. Anything else stays as resolved."""
    r = pathlib.Path(os.path.realpath(p))
    for root in _CANON_ROOTS:
        if r.is_relative_to(root):
            return REPO / r.relative_to(root)
    return r


class Doc:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.root = ET.parse(path).getroot()
        self.links = {lk.get("name"): lk for lk in self.root.findall("link")}
        self.joints = list(self.root.findall("joint"))
        self.children_of: dict[str, list[str]] = {}
        self.parent_joint: dict[str, ET.Element] = {}
        for j in self.joints:
            p, c = j.find("parent").get("link"), j.find("child").get("link")
            self.children_of.setdefault(p, []).append(c)
            self.parent_joint[c] = j

    def rebase_meshes(self, elems, out_dir: pathlib.Path):
        """Rewrite <mesh filename> to be relative to out_dir (no file copies).

        Each source file comes from the one URDF mesh resolver (`paths.urdf_mesh` — ``package://`` = the URDF's
        folder, then its parent: SharpA keeps its URDF at the package root, DG-5F-S under ``urdf/``, both measured).
        The written path goes through `_canon` (resolve, then re-root onto this checkout): the same relative path
        whichever checkout (canonical or worktree) generates the URDF; readers map it back with the same resolver.
        """
        n = 0
        for el in elems:
            for m in el.iter("mesh"):
                fn = m.get("filename")
                if not fn:
                    continue
                real = urdf_mesh(self.path, fn)  # package:// = URDF folder, then its parent (ROS `<pkg>/urdf/`)
                if not real.is_file():
                    raise SystemExit(f"mesh not found while rebasing: {fn} -> {real}")
                if (
                    fn.startswith("package://")
                    and self.path.resolve().parent not in real.parents
                    and not getattr(self, "_pkg_root_noted", False)
                ):
                    # Always say which layout was used (never pick silently): DG-5F-S = grandparent, SharpA = URDF
                    # folder
                    print(
                        f"[rebase_meshes] package:// root = URDF grandparent {self.path.parent.parent} "
                        f"({self.path.name})"
                    )
                    self._pkg_root_noted = True
                src = _canon(real)  # the written path is relative to this checkout
                # The Isaac URDF importer derives USD prim names from the mesh
                # file STEM, and a stem with characters outside [A-Za-z0-9_]
                # (RB-Y1 ships LINK_12_v1.1.dae etc.) produces an ill-formed
                # SdfPath -> "RuntimeError: Used null prim" and an EMPTY robot
                # while the wrapper still exits 0 (measured). Vendor
                # files stay pristine; offenders get a one-time sanitised copy
                # in a shared dir and the URDF references that.
                import re as _re

                stem = src.stem
                if _re.search(r"[^A-Za-z0-9_]", stem):
                    safe = _re.sub(r"[^A-Za-z0-9_]", "_", stem) + src.suffix
                    sdir = out_dir.parent / "_sanitized_meshes"
                    sdir.mkdir(parents=True, exist_ok=True)
                    dst = sdir / safe
                    if not dst.exists():
                        import shutil as _sh

                        _sh.copy2(real, dst)
                    src = dst
                m.set("filename", os.path.relpath(src, out_dir))
                n += 1
        return n


def copy_subtree(doc: Doc, root_link: str, drop_below: set[str] = frozenset()):
    """Deep-copies of (links, joints) for the subtree at root_link.

    ``drop_below``: link names whose DESCENDANTS (and attaching joints) are cut;
    the named link itself is kept — how the arm skeleton keeps link7 but sheds
    the old hand.
    """
    keep, stack = [], [root_link]
    while stack:
        n = stack.pop()
        keep.append(n)
        if n in drop_below:
            continue
        stack.extend(doc.children_of.get(n, []))
    keep_set = set(keep)
    links = [ET.fromstring(ET.tostring(doc.links[n])) for n in keep]
    joints = [
        ET.fromstring(ET.tostring(j))
        for j in doc.joints
        if j.find("parent").get("link") in keep_set
        and j.find("child").get("link") in keep_set
        and j.find("parent").get("link") not in drop_below
    ]
    return links, joints


def el_joint(name, jtype, parent, child, xyz, rpy):
    j = ET.Element("joint", {"name": name, "type": jtype})
    ET.SubElement(j, "origin", {"xyz": " ".join(f"{v:.10g}" for v in xyz), "rpy": " ".join(f"{v:.10g}" for v in rpy)})
    ET.SubElement(j, "parent", {"link": parent})
    ET.SubElement(j, "child", {"link": child})
    return j


def _read_back(arm: str, hand: str, out_path: pathlib.Path) -> None:
    """Read-back sanity of a written URDF: every mesh resolves (paths.urdf_mesh)."""
    chk = ET.parse(out_path).getroot()
    n_l, n_j = len(chk.findall("link")), len(chk.findall("joint"))
    missing = []
    for me in chk.iter("mesh"):
        fn = me.get("filename")
        if fn and not urdf_mesh(out_path, fn).is_file():
            missing.append(fn)
    print(
        f"  [{arm}+{hand}] links {n_l} joints {n_j} "
        f"unresolved meshes {len(missing)}"
        + (f" e.g. {missing[:2]}" if missing else "")
        + f" -> {os.path.relpath(out_path, REPO)}"
    )
    if missing:
        raise SystemExit("mesh paths unresolved")
