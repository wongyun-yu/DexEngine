"""Supplementary objects of a base tree (raw objects added to the delivered scene): their files, marks, assets and
the checks that a tree carries the current ones."""

from __future__ import annotations

import hashlib
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from dexengine.data import raw_sample as RS  # noqa: E402 — owner of the supplement pose sampling rule and version
from dexengine.data.base_layer.common import ROOT, _dest, dl  # noqa: F401


def supplement_input(ds: str, tree_name: str) -> str:
    """**Fingerprint of the supplement pose series** a base tree read: (name, content sha256) per file supplements_for
    opens (INTAKE/raw_objects/<tree>__*.npz) · "none" without any. Supplements are inputs outside retargeting, so even
    on
    the same version a new or changed supplement (support supplements from oakink2.raw_objects --supports --export etc.)
    makes the tree stale (tree_stale_reason)."""
    files = _supplement_files(ds, tree_name)
    if not files:
        return "none"
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode() + b"\0" + hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()[:16]


def supplement_sample_reason(ti: dict) -> str:
    """Tree content measurement: the reason when a supplement object's sampling version (objects[].supplement.sample) is
    not the current rule (raw_sample.RAW_SAMPLE). Such a supplement is posed at a different time than the delivered
    objects, so the tree is rebuilt (supplements_for rejects old supplement files, so supplements are rewritten
    first)."""
    for o in ti.get("objects") or []:
        if o.get("supplement"):
            why = RS.stamp_reason(o["supplement"].get("sample"), f"supplement object {o.get('obj_id')}")
            if why:
                return why
    return ""


def supplement_copy_reason(ti: dict) -> str:
    """Tree content measurement: the reason when a supplement object is a copy with the same part id
    (oakink2.parts.rid_of) as a delivered object (the current supplements_for never loads it, so an old tree that did is
    stale). Only OakInk2 has supplements."""
    objs = ti.get("objects") or []
    sup = [o for o in objs if o.get("supplement")]
    if not sup:
        return ""
    from dexengine.data.oakink2 import parts as OP  # noqa: PLC0415 — owner of the part id rule

    have = {OP.rid_of(o["obj_id"]) for o in objs if not o.get("supplement")}
    dup = sorted(o["name"] for o in sup if OP.rid_of(o["obj_id"]) in have)
    return (
        f"supplement copies {dup} with the same part id as the handoff (the current supplement rule does not load them)"
        if dup
        else ""
    )


# ── ⑥ Raw supplement objects (found in per-dataset reference material, absent from the tree) ─
def _supplement_files(ds: str, tree_name: str) -> list[Path]:
    d = _dest("intake", ds) / "raw_objects"
    return sorted(d.glob(f"{tree_name}__*.npz")) if d.is_dir() else []


def _supp_hand_rid(f: Path, tree_name: str) -> tuple[str, str]:
    """Supplement file name `<tree>__<hand>__<rid>.npz` → (hand, rid); read by both supplements_for and
    supplement_dups."""
    from dexengine.data.oakink2 import parts as OP  # noqa: PLC0415 — owner of the tag

    hand, rid = f.stem[len(tree_name) + 2 :].split("__", 1)
    return OP.supp_tag(hand, f.name), rid


def supplement_dups(ds: str, tree_name: str, have: set) -> list[str]:
    """Files the supplement copy rule (oakink2.parts.SUPP_KEY) drops from this tree: supplements whose rid equals a
    delivered part id (`have`). Reads file names only (the same set as supplements_for's dup)."""
    return [f.name for f in _supplement_files(ds, tree_name) if _supp_hand_rid(f, tree_name)[1] in have]


def supplement_mark(ti: dict, dropped: list[str]) -> dict:
    """Marker of a tree whose content the copy rule changed (some files dropped): rules[SUPP_KEY] = version + dropped
    files. {} when nothing was dropped (no field added; a tree the rule did not change has the same content as the old
    tree)."""
    if not dropped:
        return {}
    from dexengine.data.oakink2 import parts as OP  # noqa: PLC0415

    return {"rules": {**(ti.get("rules") or {}), OP.SUPP_KEY: {"version": OP.SUPP_VERSION, "dropped": list(dropped)}}}


def supplement_mark_reason(ti: dict, tree_name: str) -> str:
    """Tree content ↔ marker: the reason when the copy rule drops supplements from this tree but rules[SUPP_KEY] is not
    the current version (trees the rule changed record its version; without the marker the shared two-hand layer cannot
    tell the generation)."""
    from dexengine.data.oakink2 import parts as OP  # noqa: PLC0415 — owner of the part id and the copy rule version

    have = {OP.rid_of(o["obj_id"]) for o in ti.get("objects") or [] if not o.get("supplement")}
    dup = supplement_dups(str(ti.get("dataset_name")), tree_name, have) if have else []
    got = ((ti.get("rules") or {}).get(OP.SUPP_KEY) or {}).get("version")
    if dup and got != OP.SUPP_VERSION:
        return f"tree with {len(dup)} supplement copies dropped: copy rule version {got!r} ≠ {OP.SUPP_VERSION}"
    return ""


def ensure_supplement_asset(ds: str, rid: str, mesh_src: str) -> tuple[str, str]:
    """assets/objects/raw_<rid>/visual.obj — a copy of the raw mesh (no convex decomposition:
    convex_dir = the same directory, and a marker is left)."""
    oid = "raw_" + rid.lower().replace("@", "_")
    dst = _dest("assets_objects", ds) / oid
    if not (dst / "visual.obj").exists():
        dst.mkdir(parents=True, exist_ok=True)
        tmp = dst / f"visual.{os.getpid()}.tmp"
        shutil.copy2(ROOT / mesh_src, tmp)
        os.replace(tmp, dst / "visual.obj")
        (dst / "PROVENANCE.txt").write_text(
            f"Copy of OakInk2 raw {mesh_src} (oakink2_raw_objects.py). No convex decomposition — physics "
            "needs a separate decomposition.\n"
        )
    return oid, dl.object_ref(ds, oid)  # logical name stored in objects[].mesh_dir (owner dexengine.core.layout)


def insert_supplement_body(roots: dict, name: str, oid: str, ds: str, hand: str) -> None:
    """Append one free-joint body at the end of worldbody in scene.xml/scene_eq.xml (qpos gains 7
    columns at the end). The mesh path is relative to the asset root (objects/<oid>/visual.obj)."""
    for root in roots.values():
        asset = root.find("asset")
        wb = root.find("worldbody")
        m = ET.SubElement(asset, "mesh")
        m.set("name", f"{name}_visual")
        m.set("file", f"objects/{oid}/visual.obj")
        b = ET.SubElement(wb, "body")
        b.set("name", name)
        j = ET.SubElement(b, "joint")
        j.set("name", f"{name}_joint")
        j.set("type", "free")
        j.set("armature", "0.0001")
        j.set("frictionloss", "0.0001")
        g = ET.SubElement(b, "geom")
        g.set("name", f"{name}_collision_visual")
        g.set("type", "mesh")
        g.set("group", "3")
        g.set("density", "1000")
        g.set("rgba", "0 1 0 1")
        g.set("mesh", f"{name}_visual")
        g2 = ET.SubElement(b, "geom")
        g2.set("name", f"{name}_visual")
        g2.set("type", "mesh")
        g2.set("density", "0")
        g2.set("rgba", "1 1 1 1")
        g2.set("mesh", f"{name}_visual")
