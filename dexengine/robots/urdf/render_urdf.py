"""The render URDF of the merged-robot generator: ``<stem>_render.urdf`` = the merged robot plus the mount's declared
render parts as visual-only fixed links (see ``urdf/merge.py``, RENDER URDF)."""

from __future__ import annotations

import copy
import os
import pathlib
import xml.etree.ElementTree as ET

from dexengine.core.repo import REPO_ROOT as REPO
from dexengine.robots.urdf import materials as UM
from dexengine.robots.urdf.doc import _canon, _read_back, el_joint, quat_wxyz_to_rpy
from dexengine.robots.usd import render_parts as RVP


def write_render_urdf(
    arm: str, hand: str, root: ET.Element, ee: dict, header: str, out_path: pathlib.Path, parts: tuple
) -> None:
    """The render URDF beside ``out_path`` (module doc: RENDER URDF) — ``root`` (the merged robot, left untouched) plus
    the declared render parts of (arm, hand) (``parts`` = its MountSpec.render_visual_parts) under ``ee[side]``; no
    parts → none (a stale one is removed)."""
    out_dir = _canon(out_path.parent)
    r_path = render_urdf_path(out_path)
    if not parts:
        if r_path.exists():
            r_path.unlink()
            print(f"  [RENDER_URDF] {arm}+{hand}: no declared parts — removed the stale {r_path.name}")
        return
    taken = {e.get("name") for e in root if e.tag in ("link", "joint")}
    clash = sorted(
        taken & {f(side, part) for side in ("right", "left") for part in parts for f in (RVP.urdf_link, RVP.urdf_joint)}
    )
    if clash:
        raise SystemExit(
            f"[RENDER_URDF] {arm}+{hand}: part names {clash} already exist in the merged URDF — rename the parts"
        )
    r_root = copy.deepcopy(root)
    for side in ("right", "left"):
        for part in parts:
            r_root.append(render_part_link(side, part, out_dir))
            r_root.append(
                el_joint(
                    RVP.urdf_joint(side, part),
                    "fixed",
                    ee[side],
                    RVP.urdf_link(side, part),
                    part.ee_to_part_pos,
                    quat_wxyz_to_rpy(part.ee_to_part_quat),
                )
            )
    UM.hoist_named_materials(r_root)
    ET.indent(r_root, space="  ")
    r_header = (
        header + f"<!-- RENDER URDF: {out_path.name} + the mount's declared render parts "
        f"(mount_specs MountSpec.render_visual_parts) as visual-only fixed links - no collision, no inertial. "
        f"Not a bake, IK, judge or placement input (those read {out_path.name}); "
        f"the bake refuses it (render_parts.render_links_in_urdf). -->\n"
    )
    r_path.write_text('<?xml version="1.0"?>\n' + r_header + ET.tostring(r_root, encoding="unicode") + "\n")
    _read_back(arm, hand, r_path)
    got = RVP.render_links_in_urdf(r_path)
    want = sorted(RVP.urdf_link(side, part) for side in ("right", "left") for part in parts)
    if got != want or RVP.render_links_in_urdf(out_path):
        raise SystemExit(
            f"[RENDER_URDF] {arm}+{hand}: render part links {got} ≠ declared {want} "
            f"(or {out_path.name} carries part links)"
        )
    for part in parts:
        print(
            f"  [RENDER_URDF] {arm}+{hand} {part.name} · {', '.join(ee[s] for s in ('right', 'left'))} → "
            f"{', '.join(RVP.urdf_link(s, part) for s in ('right', 'left'))} at {tuple(part.ee_to_part_pos)} "
            f"q={tuple(part.ee_to_part_quat)} · visual only (no collision, no inertial) -> "
            f"{os.path.relpath(r_path, REPO)}"
        )


def render_urdf_path(urdf_path: pathlib.Path) -> pathlib.Path:
    """``<stem>.urdf`` → its render URDF ``<stem>_render.urdf`` (same folder — mesh paths stay relative to it)."""
    return urdf_path.with_name(f"{urdf_path.stem}_render.urdf")


def render_part_link(side: str, part, out_dir: pathlib.Path) -> ET.Element:
    """One declared render part as a URDF link: one ``<visual>`` (origin = part_to_mesh · mesh = the declared file past
    its sha256 gate · colour = the declared one), no ``<collision>``, no ``<inertial>``."""
    link = ET.Element("link", {"name": RVP.urdf_link(side, part)})
    vis = ET.SubElement(link, "visual")
    ET.SubElement(
        vis,
        "origin",
        {
            "xyz": " ".join(f"{v:.10g}" for v in part.part_to_mesh_pos),
            "rpy": " ".join(f"{v:.10g}" for v in quat_wxyz_to_rpy(part.part_to_mesh_quat)),
        },
    )
    geo = ET.SubElement(vis, "geometry")
    ET.SubElement(geo, "mesh", {"filename": os.path.relpath(_canon(RVP.checked_mesh(part)), out_dir)})
    mat = ET.SubElement(vis, "material", {"name": part.name})
    ET.SubElement(mat, "color", {"rgba": " ".join(f"{v:.10g}" for v in (*part.color, 1.0))})
    return link
