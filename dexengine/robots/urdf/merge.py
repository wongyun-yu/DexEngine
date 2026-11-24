#!/usr/bin/env python3
"""Generate a bimanual full-robot URDF for any (arm, hand) pair — ONE generator.

Everything pair-specific lives in the embodiment spec files (configs/robots/{arms,hands,mounts}, read by
dexengine.core.embodiment), so the merge is a mechanical composition: arm skeleton + per-side hand subtree + mount
joint. No arm or hand is named in this file.

WHAT IT BUILDS
    world
      └─ (arm skeleton)                 ArmSpec.skeleton: the vendor body cut at root_link, the links below each EE
      │                                  link and drop_links removed, the fixed_joint_prefix joints welded at their
      │                                  posture, extra_links added — see skeleton_build().
      ├─ right arm EE ── mount joint ── right hand subtree
      └─ left  arm EE ── mount joint ── left  hand subtree

    The mount joint origin comes from the (arm, hand) mount (offset + per-side quat, converted to rpy). Hand subtrees
    are cut at HandSpec.urdf_subtree_root from HandSpec.merge_document(side) (the vendor URDF unless merge_source
    names another document).

    RENDER URDF  A pair whose mount declares render parts (MountSpec.render_visual_parts) also gets
    <stem>_render.urdf beside <stem>.urdf (urdf/render_urdf.py): the same robot plus, per side and part, a
    visual-only fixed link under the EE link. <stem>.urdf itself never carries them — the bake, the arm IK, the
    judge and placement read it, and placement takes right_*/left_* URDF visuals as hand check shapes, so a
    part there would change the selection. A pair without parts gets no render URDF (a stale one is removed). Marker
    [RENDER_URDF].

    A merge document that is not on disk stops the build with its path (SharpA right: the earlier merged robot URDF is
    not in the repository; the committed rby1_sharpa_wave URDF/USD are the canonical assets).

Usage: python dexengine/robots/urdf/merge.py rby1+xhand_pro [--variant SFX] [--out-root DIR] [--spec-root DIR]
"""

from __future__ import annotations

import argparse
import os
import pathlib
import xml.etree.ElementTree as ET

# isort: split
from dexengine.core import embodiment as EMB
from dexengine.core.errors import InputMissingError
from dexengine.core.paths import repo_rel_path  # repo-relative place rule (data layer → data root)
from dexengine.core.repo import ROBOT_ASSETS_DIR
from dexengine.robots.urdf import materials as UM  # visual material rule (named inline colour → top-level definition)
from dexengine.robots.urdf.doc import (  # noqa: F401 — re-exported URDF helpers
    Doc,
    _canon,
    _read_back,
    axis_angle_to_rpy,
    copy_subtree,
    el_joint,
    quat_wxyz_to_rpy,
)
from dexengine.robots.urdf.render_urdf import render_part_link, render_urdf_path, write_render_urdf  # noqa: F401

#: Robot and hand description files (tracked assets).
ROBOTS = ROBOT_ASSETS_DIR


def _canon_gen(arm: str):
    """The canonical generation record of `arm` (``arms.ARM_CANON_GEN``): the variant suffix of the output path (the
    posture's generation joint and the base pose come from the same record through the arm spec)."""
    from dexengine.core import arms  # noqa: PLC0415

    return arms.ARM_CANON_GEN.get(arm)


def apply_posture_override(posture: dict, spec: str) -> dict:
    """``posture`` with a trial override applied before freezing it: "torso_1=0.52,torso_2=-1.05" (rad); only the
    listed joints change. A joint the posture does not list is an error."""
    out = dict(posture)
    for _kv in filter(None, spec.split(",")):
        _k, _, _v = _kv.partition("=")
        _k = _k.strip()
        if _k not in out:
            raise SystemExit(f"--posture: unknown joint {_k!r} (known: {', '.join(out)})")
        out[_k] = float(_v)
    return out


def _fix_joints(joints_to_fix, posture: dict) -> None:
    """Weld each listed joint at its posture angle (origin rpy), dropping its axis/limit/dynamics tags."""
    for j in joints_to_fix:
        name = j.get("name", "")
        angle = posture.get(name, 0.0)
        if angle:
            ax = j.find("axis")
            axis = tuple(float(v) for v in ax.get("xyz").split()) if ax is not None else (0, 0, 1)
            o = j.find("origin")
            if o is None:
                o = ET.SubElement(j, "origin")
            o.set("rpy", " ".join(f"{v:.9f}" for v in axis_angle_to_rpy(axis, angle)))
        j.set("type", "fixed")
        for tag in ("axis", "limit", "dynamics", "safety_controller"):
            for el in j.findall(tag):
                j.remove(el)


def _extra_link(x, out_dir):
    """(link, joint) of one ArmSkeleton.extra_links entry — the link text with {mesh} filled by the path rule
    Doc.rebase_meshes applies (relative to the output directory)."""
    mesh = _canon(repo_rel_path(x.mesh))
    if not mesh.is_file():
        raise SystemExit(f"mesh not found while rebasing: {x.mesh} -> {mesh}")
    text = repo_rel_path(x.link_xml).read_text(encoding="utf-8").strip()
    link = ET.fromstring(text.format(mesh=os.path.relpath(mesh, out_dir)))
    return link, el_joint(x.joint, "fixed", x.parent, link.get("name"), x.xyz, x.rpy)


# --------------------------------------------------------------------------
# arm skeleton
# --------------------------------------------------------------------------
def skeleton_build(arm, out_dir, posture: dict | None = None, free: bool = False):
    """(links, joints, ee) of the arm body described by ``arm.skeleton`` (``ArmSpec``): the vendor subtree at
    root_link without the links below each side's EE link (``ArmSpec.ee_body_name``) and without drop_links, the
    extra_links added, the fixed_joint_prefix joints welded at ``posture`` (default the skeleton's), meshes rebased
    onto ``out_dir``, and the root fixed to a ``world`` link. ``free`` keeps the posture joints revolute (a build for
    picking the posture interactively; the chosen values then go into the arm spec file)."""
    sk = arm.skeleton
    if sk is None:
        raise SystemExit(f"arm {arm.name!r} has no skeleton in its spec file — the merge cannot build its body")
    doc = Doc(repo_rel_path(sk.vendor_urdf))
    sides = ("right", "left")
    links, joints = copy_subtree(doc, sk.root_link, drop_below={arm.ee_body_name(s) for s in sides})
    drop = set(sk.drop_links)
    links = [ln for ln in links if ln.get("name") not in drop]
    joints = [j for j in joints if j.find("child").get("link") not in drop]
    fixed = [] if free else [j for j in joints if j.get("name", "").startswith(sk.fixed_joint_prefix)]
    _fix_joints(fixed, sk.posture if posture is None else posture)
    doc.rebase_meshes(links, out_dir)
    # Extra links come after the rebase: their mesh path is already relative to out_dir (a second rebase would
    # resolve it against the vendor URDF's folder, which only coincides for an out_dir at the same depth).
    for x in sk.extra_links:
        lk, jt = _extra_link(x, out_dir)
        links.append(lk)
        joints.append(jt)
    links.insert(0, ET.Element("link", {"name": "world"}))
    joints.insert(0, el_joint(sk.world_joint, "fixed", "world", sk.root_link, (0, 0, 0), (0, 0, 0)))
    return links, joints, {s: arm.ee_body_name(s) for s in sides}


# --------------------------------------------------------------------------
# hand subtrees
# --------------------------------------------------------------------------
def hand_subtree(hs, side: str, out_dir):
    """(links, joints, root_link) for one side's hand (``HandSpec``), vendor data intact: the subtree at
    ``urdf_subtree_root`` of ``merge_document(side)``."""
    rel = hs.merge_document(side)
    if not rel:
        raise InputMissingError(f"hand {hs.name!r}: its spec has no vendor_urdf — state the URDF to merge")
    path = repo_rel_path(rel)
    if not path.is_file():
        # Stop with the path — never substitute another source.
        raise InputMissingError(
            f"[make_full_robot_urdf] {hs.name}/{side}: the merge document {rel} is not on disk ({path}) — see "
            f"configs/robots/hands/{hs.name}.yaml (vendor_urdf · merge_source)"
        )
    doc = Doc(path)
    root = hs.urdf_subtree_root.format(side=side)
    links, joints = copy_subtree(doc, root)
    # A vendor hand without a side prefix clashes with arm link names (measured: DG-5F-S's link_base and
    # link_5_1..5_4 clash with RB-Y1). Hands already prefixed with {side}_ are unchanged by the guard below. The
    # hand-rig builder applies the same rule.
    ren = {}
    for el in links:
        n = el.get("name")
        if n and not n.startswith(f"{side}_"):
            ren[n] = f"{side}_{n}"
            el.set("name", ren[n])
    if ren:
        for j in joints:
            jn = j.get("name")
            if jn and not jn.startswith(f"{side}_"):
                j.set("name", f"{side}_{jn}")
            for tag in ("parent", "child"):
                e = j.find(tag)
                if e is not None and e.get("link") in ren:
                    e.set("link", ren[e.get("link")])
        root = ren.get(root, root)
        print(f"[hand_subtree] {hs.name}/{side}: the vendor has no side prefix — renamed {len(ren)} links to {side}_*")
    doc.rebase_meshes(links, out_dir)
    return links, joints, root


# --------------------------------------------------------------------------
def build(emb, out_path: pathlib.Path, posture: dict | None = None, free: bool = False) -> None:
    """Write the merged URDF of the resolved combo ``emb`` (``embodiment.resolve``) to `out_path` (and its render
    URDF when the mount declares parts)."""
    arm, hs, m = emb.arm, emb.hand, emb.mount
    out_dir = _canon(out_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    a_links, a_joints, ee = skeleton_build(arm, out_dir, posture, free)

    root = ET.Element("robot", {"name": f"{arm.name}_{hs.name}"})
    for lk in a_links:
        root.append(lk)
    for j in a_joints:
        root.append(j)

    seen = {lk.get("name") for lk in a_links}
    for side in ("right", "left"):
        h_links, h_joints, h_root = hand_subtree(hs, side, out_dir)
        clash = [lk.get("name") for lk in h_links if lk.get("name") in seen]
        if clash:
            raise SystemExit(f"link name clash arm↔hand: {clash[:5]}")
        seen |= {lk.get("name") for lk in h_links}
        for lk in h_links:
            root.append(lk)
        for j in h_joints:
            root.append(j)
        q = m.ee_to_wrist_rot_right if side == "right" else m.ee_to_wrist_rot_left
        rpy = quat_wxyz_to_rpy(q)
        root.append(el_joint(f"{side}_mount_joint", "fixed", ee[side], h_root, m.ee_to_wrist_offset_for(side), rpy))

    n_mat = UM.hoist_named_materials(root)
    print(
        f"  {UM.MARK} {arm.name}+{hs.name}: hoisted {n_mat} named inline-colour materials to top-level definitions",
        flush=True,
    )
    ET.indent(root, space="  ")
    header = (
        f"<!-- GENERATED by dexengine/robots/urdf/merge.py -->\n"
        f"<!-- pair=({arm.name},{hs.name}) mount provisional={m.provisional} "
        f"provenance={m.provenance!r} -->\n"
    )
    out_path.write_text('<?xml version="1.0"?>\n' + header + ET.tostring(root, encoding="unicode") + "\n")
    _read_back(arm.name, hs.name, out_path)
    write_render_urdf(arm.name, hs.name, root, ee, header, out_path, m.render_visual_parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="*", help="arm+hand pairs like rby1+sharpa_wave (no default)")
    ap.add_argument("--out-root", type=pathlib.Path, default=ROBOTS / "full_robot_gen")
    ap.add_argument(
        "--spec-root",
        type=pathlib.Path,
        default=None,
        help="embodiment spec root (arms/ hands/ mounts/); default this tree's configs/robots (marker [EMBODIMENT])",
    )
    # The output path carries the asset variant suffix, so a rerun never writes a non-canonical posture onto the
    # unsuffixed path. Default = the arm's canonical variant; to re-bake the old unsuffixed generation pass
    # --variant "" and put that generation's posture into the arm spec file.
    ap.add_argument(
        "--variant",
        default=None,
        help="asset variant suffix. Default = the arm's canonical one (arms.ARM_CANON_GEN). "
        'Pass --variant "" for the old unsuffixed generation.',
    )
    ap.add_argument(
        "--posture",
        default="",
        help='trial override of the skeleton posture before fixing it, e.g. "torso_1=0.52,torso_2=-1.05" '
        "(rad); only the listed joints change",
    )
    ap.add_argument(
        "--posture-free",
        action="store_true",
        help="keep the posture joints revolute (a build for picking the posture interactively)",
    )
    a = ap.parse_args()
    # A relative --out-root would break out_path.relative_to(REPO) below — make it absolute.
    a.out_root = a.out_root.resolve()

    # No default pair list (it would be a copy of the registry): the caller names what to build.
    if not a.pairs:
        raise InputMissingError(
            "[make_full_robot_urdf] name the pairs to build as <arm>+<hand> (no default) — "
            "registered combos: python3 dexengine/core/paths.py --what combo-ids"
        )
    pairs = [tuple(p.split("+", 1)) for p in a.pairs]
    for arm, hand in pairs:
        emb = EMB.resolve(arm, hand, a.spec_root)
        posture = apply_posture_override(emb.arm.skeleton.posture, a.posture) if emb.arm.skeleton else None
        # Only an arm with a generation axis has a variant suffix.
        g = _canon_gen(arm)
        var = a.variant if a.variant is not None else (g.variant if g is not None else "")
        stem = f"{arm}_{hand}{var}"
        out = a.out_root / stem / f"{stem}.urdf"
        if var == "" and g is not None and g.variant:
            print(
                f"[make_full_robot_urdf] ⚠ {arm}_{hand}: building onto the unsuffixed path — the canonical "
                f"variant is '{g.variant}'. With the canonical posture (upright chest) this "
                f"OVERWRITES the old-generation asset on the unsuffixed path. Make sure that is intended.",
                flush=True,
            )
        build(emb, out, posture=posture, free=a.posture_free)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
