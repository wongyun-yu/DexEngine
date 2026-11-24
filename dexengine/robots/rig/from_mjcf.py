"""A retargeting rig built from the vendor's own MJCF (``HandSpec.vendor_mjcf``) instead of its URDF.

WHY
    Some vendors ship an MJCF next to the URDF. Sharpa Wave's MJCF carries the vendor's calibrated model, and its
    URDF does not carry the same values: the virtual joint links weigh 0.0025 kg in the MJCF and 8e-5 kg in the URDF,
    and the MJCF root body is the hand base 2.95 cm above the URDF root. The kinematic chain is the same
    (``check_same_kinematics``). The retargeting rig of such a hand is the vendor MJCF with the rig contract applied,
    and this module is that conversion. ``hand_rig.py`` calls it for a hand whose spec names ``vendor_mjcf``.

WHAT IT CHANGES (rules, no per-element table)
    1. ``<mujoco model>`` = ``<side>_<hand>_spider``; the rig template skeleton (compiler · option · default · visual)
       and world assets replace the vendor ``<compiler>`` and ``<default>`` classes; every ``class`` attribute goes
       (the template defaults apply — the rig is kinematic).
    2. Meshes: name side-prefixed unless it already starts with ``<side>_``; file ``<side>/<file>``; copied.
    3. Geoms: ``contype``/``conaffinity`` 0 (the rig has no self contacts; the retargeting scene states its contact
       pairs itself); names and mesh references side-prefixed.
    4. Root body: the base quat and gravcomp, then the tracking camera, the palm site (``HandSpec.rig_palm_site``),
       the six base joints and the palm box, before the vendor's own children.
    5. Each fingertip host body (``HandSpec.tip_body_names_right``) gets the three tip sites at
       ``HandSpec.fingertip_site_offsets`` along its +x and a distal capsule from ``rig_capsule_distal_start`` to the
       tip; its parent gets a middle capsule from its origin to the host's origin when ``rig_capsule_middle``.
    6. ``<contact>`` excludes kept; ``<actuator>`` = the six base actuators, then the vendor's.
"""

from __future__ import annotations

import os
import pathlib
import xml.etree.ElementTree as ET

from dexengine.robots.rig import template as TPL

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
_PALM_SIZE = "0.02 0.03 0.04"
_PALM_BOX = "0.02 0.03 0.05"
_CAPSULE_R = "0.008"


def _g(v: float) -> str:
    return f"{v:.6g}"


def _capsule(name: str, fromto: str) -> ET.Element:
    return ET.Element(
        "geom",
        {
            "name": name,
            "type": "capsule",
            "fromto": fromto,
            "size": _CAPSULE_R,
            "contype": "0",
            "conaffinity": "0",
            "group": "3",
            "density": "0",
            "rgba": "0 1 0 1",
        },
    )


def _tip_offsets(spec) -> list[float]:
    off = spec.fingertip_site_offsets
    return [float(off["thumb"])] + [float(off["fingers"])] * 4


def _side_name(name: str, side: str) -> str:
    """Vendor names are mostly `<side>_…` already; the rest get the prefix (they would collide in the bimanual
    scene)."""
    return name if name.startswith(f"{side}_") else f"{side}_{name}"


def _strip_class(root: ET.Element) -> None:
    for el in root.iter():
        el.attrib.pop("class", None)


def _geoms(body: ET.Element, side: str, mesh_rename: dict[str, str]) -> None:
    for geom in body.iter("geom"):
        if "contype" in geom.attrib:
            geom.set("contype", "0")
        if "conaffinity" in geom.attrib:
            geom.set("conaffinity", "0")
        if geom.get("mesh") in mesh_rename:
            geom.set("mesh", mesh_rename[geom.get("mesh")])
        if geom.get("name"):
            geom.set("name", _side_name(geom.get("name"), side))


def _root_head(base: ET.Element, side: str, palm_pos) -> None:
    head = TPL.base_head(side)  # camera, then the six base joints
    palm = ET.Element("site", {"name": f"{side}_palm", "pos": " ".join(_g(x) for x in palm_pos), "type": "box"})
    palm.set("size", _PALM_SIZE)
    box = ET.Element(
        "geom",
        {
            "type": "box",
            "name": f"collision_hand_{side}_palm_0",
            "size": _PALM_BOX,
            "pos": " ".join(_g(x) for x in palm_pos),
            "contype": "0",
            "conaffinity": "0",
            "group": "3",
            "density": "0",
            "rgba": "0 1 0 1",
        },
    )
    for i, el in enumerate([head[0], palm, *head[1:], box]):
        base.insert(i, el)


def _tips(base: ET.Element, spec, side: str) -> list[tuple[str, str, float]]:
    parent_of = {c: p for p in base.iter("body") for c in p if c.tag == "body"}
    bodies = {b.get("name"): b for b in base.iter("body")}
    placed = []
    for finger, tip_right, off in zip(FINGERS, spec.tip_body_names_right, _tip_offsets(spec), strict=True):
        host_name = tip_right.replace("right_", f"{side}_", 1)
        host = bodies.get(host_name)
        if host is None:
            raise ValueError(f"[rig_mjcf] tip host {host_name!r} not in the vendor MJCF")
        pos = f"{_g(off)} 0 0"
        for prefix in ("", "track_hand_", "trace_hand_"):
            ET.SubElement(host, "site", {"name": f"{prefix}{side}_{finger}_tip", "pos": pos})
        start = spec.rig_capsule_distal_start
        host.append(_capsule(f"collision_hand_{side}_{finger}_0", f"{_g(start)} 0 0 {pos}"))
        if spec.rig_capsule_middle:
            hp = host.get("pos")
            if not hp:
                raise ValueError(f"[rig_mjcf] {host_name} has no pos — the middle capsule has no length")
            parent_of[host].append(_capsule(f"collision_hand_{side}_{finger}_1", f"0 0 0 {hp}"))
        placed.append((finger, host_name, off))
    return placed


def build(hand: str, side: str, spec, mjcf_path: pathlib.Path, out_dir: pathlib.Path):
    """(rig root element, [(mesh source, mesh destination)], tip placements) of ``side`` from the vendor MJCF."""
    vendor = ET.parse(mjcf_path).getroot()
    meshdir = mjcf_path.parent / (vendor.find("compiler").get("meshdir") or "")
    _strip_class(vendor)

    root = ET.Element("mujoco", {"model": f"{side}_{hand}_spider"})
    TPL.add_skeleton(root)
    asset = ET.SubElement(root, "asset")
    TPL.add_world_assets(asset, side)
    mesh_rename, copies = {}, []
    for mesh in vendor.find("asset").findall("mesh"):
        name, fname = mesh.get("name"), os.path.basename(mesh.get("file"))
        mesh_rename[name] = _side_name(name, side)
        copies.append((meshdir / fname, out_dir / "assets" / side / fname))
        ET.SubElement(asset, "mesh", {"name": mesh_rename[name], "file": f"{side}/{fname}"})

    worldbody = ET.SubElement(root, "worldbody")
    TPL.add_light(worldbody)
    bodies = vendor.find("worldbody").findall("body")
    if len(bodies) != 1:
        raise ValueError(f"[rig_mjcf] {mjcf_path}: {len(bodies)} root bodies, the rig needs one")
    base = bodies[0]
    base.set("quat", TPL.base_body_quat())
    base.set("gravcomp", "1")
    _geoms(base, side, mesh_rename)
    _root_head(base, side, spec.rig_palm_site)
    placed = _tips(base, spec, side)
    worldbody.append(base)

    root.append(vendor.find("contact"))
    actuator = ET.SubElement(root, "actuator")
    TPL.add_base_actuators(actuator, side)
    for el in vendor.find("actuator"):
        actuator.append(el)
    return root, copies, placed
