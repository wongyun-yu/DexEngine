"""URDF visual material rule — every writer of a merged URDF goes through this one function.

Why: the Isaac URDF importer (UrdfConverter, Isaac Sim 5.1) creates no material for a named visual material
``<material name="N"><color rgba/></material>`` unless N is defined as a top-level ``<material name="N">`` — the
inline colour is dropped and the mesh gets the white default. An unnamed material with an inline colour, and a
reference by name to a top-level definition, both keep their colour (measured on a small URDF). The wuji2 source URDF
uses the first form (named + inline, no top-level definition), so all 40 baked hand meshes came out white.

Rule (applies to any URDF): a visual material that has both a name and an inline colour gets that colour hoisted into a
top-level definition, and the visual keeps only the name reference. The same name with two different colours, or a
colour that differs from the top-level definition, is an error (which colour is right is unknown). Unnamed materials and
materials without a colour are left alone (the importer already handles them).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

MARK = "[URDF_MATERIALS]"


def _rgba(el: ET.Element | None) -> tuple[float, ...] | None:
    c = el.find("color") if el is not None else None
    return tuple(float(x) for x in c.get("rgba").split()) if c is not None and c.get("rgba") else None


def hoist_named_materials(root: ET.Element) -> int:
    """Hoist the named inline-colour visual materials of root (<robot>) into top-level definitions.

    Returns the number of hoisted names (0 = nothing to do)."""
    top = {m.get("name"): m for m in root.findall("material")}
    added: dict[str, tuple[float, ...]] = {}
    for vis in root.iter("visual"):
        m = vis.find("material")
        if m is None or not m.get("name") or m.find("color") is None:
            continue
        name, rgba = m.get("name"), _rgba(m)
        have = _rgba(top[name]) if name in top else added.get(name)
        if have is not None and have != rgba:
            raise SystemExit(
                f"{MARK} ✗ material {name!r} has two colours: {have} ↔ {rgba} — fix the source to say which is right"
            )
        if name not in top and name not in added:
            added[name] = rgba
        m.remove(m.find("color"))
    at = sum(1 for e in root if e.tag == "material")
    for i, (name, rgba) in enumerate(added.items()):
        el = ET.Element("material", {"name": name})
        ET.SubElement(el, "color", {"rgba": " ".join(f"{v:.10g}" for v in rgba)})
        root.insert(at + i, el)
    return len(added)


def visual_colors(root: ET.Element) -> dict[str, list[tuple[float, ...] | None]]:
    """Per link, the actual colour of each visual (top-level definition or resolved inline value; None without one) —
    the stated side of the bake oracle (usd/check_colors.py)."""
    top = {m.get("name"): _rgba(m) for m in root.findall("material")}
    out: dict[str, list] = {}
    for link in root.findall("link"):
        cols = []
        for vis in link.findall("visual"):
            m = vis.find("material")
            cols.append(None if m is None else (_rgba(m) or top.get(m.get("name"))))
        if cols:
            out[link.get("name")] = cols
    return out
