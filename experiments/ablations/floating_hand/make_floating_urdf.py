#!/usr/bin/env python3
"""Per-side URDF of the floating hand (DEXENGINE_ARM=floating): the HAND SUBTREE cut out of the canonical combo URDF.

Why this source: the sim body's committed combo URDF (``rby1_sharpa_wave.urdf``; wuji2
``…_truncated``) is cut below the child of ``{side}_mount_joint`` (the hand root) as it is — meshes,
collision, inertia and joint limits are byte-for-byte the same elements as the arm body's hand (the strongest same-hand
guarantee).

Output: ``<combo URDF dir>/<arm>_<hand>/<arm>_<hand>_split_{R,L}.urdf`` — one per side (a URDF has one root, so the two
hands cannot share a file). The ``_split_{R,L}`` names are owned by split_artic.split_urdf_pair (the same tags as the
USD pair split_usd_pair); the USD bake (build_floating_usd.sh) writes the USD of the same stem. Mesh paths are relative
(``../../…``) and the output has the source's depth (full_robot_gen/<folder>/), so they stay valid — the third check
counts them.

Checks (on the output): per side (1) link · joint counts = the source subtree's (2) elements byte-identical (3) every
mesh file exists (4) no link of the other side or of the arm. Any failure → rc 1. Marker ``[FLOATING_URDF]``.

Virtual-joint body (``--wrist-joints`` — the asset of DEXENGINE_FH_WRIST=joint, off by default): the same hand subtree
under a world-fixed anchor → 3 prismatic → 3 revolute (Y→X→Z) → hand root chain (names, axes, limits, masses owned
by body.VJ_*). Output stem ``<arm>_<hand>_vj`` (body.vj_canon_path); the default files are not touched. Instead of check
(4): root = the anchor, and the added joints are the 6 virtual ones.

Note: no output is committed (build outputs: build_floating_usd.sh runs this); this generator writes next to the combo
URDFs (dexengine.core.repo.FULL_ROBOT_URDF_DIR). The source path comes from dexengine.robots.selection.combo_urdf_path,
and the robot selection derives the split pair from its own URDF folder (selection.URDF_GEN_DIR) — the two folders must
be the same for the generated pair to be found.

    python -m experiments.ablations.floating_hand.make_floating_urdf --hand sharpa_wave [--check] [--wrist-joints]
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import sys
import xml.etree.ElementTree as ET

from dexengine.core import arms as A
from dexengine.core.repo import FULL_ROBOT_URDF_DIR as GEN
from dexengine.core.repo import REPO_ROOT as REPO
from dexengine.rl.tasks.scene import split_artic as S
from experiments.ablations.floating_hand import body as F


def source_urdf(hand: str) -> pathlib.Path:
    """The source = the sim body's combo URDF — rule owner body.source_urdf (dies when missing)."""
    try:
        return F.source_urdf(hand)
    except FileNotFoundError as e:
        raise SystemExit(f"[FLOATING_URDF] no source combo URDF: {e}") from e


def out_paths(hand: str, wrist_joints: bool = False) -> dict[str, pathlib.Path]:
    """Per-side output paths — name rule owner split_artic.split_urdf_pair (merged URDF path → _split_{R,L} in the
    same folder). The robot selection finds the files with the same function. ``wrist_joints`` = the virtual-joint body
    (merged path = body.vj_canon_path)."""
    stem = f"{A.FLOATING_ARM}_{hand}"
    canon = GEN / stem / f"{stem}.urdf"
    if wrist_joints:
        canon = F.vj_canon_path(canon)
    right, left = S.split_urdf_pair(canon)
    return {"right": right, "left": left}


def wrist_chain(side: str, hand_root: str) -> tuple[list[ET.Element], list[ET.Element]]:
    """The virtual joint chain (links, joints) — anchor → … → hand root. Values owned by body (VJ_*)."""
    links_n, joints_n = F.vj_link_names(side), F.vj_joint_names(side)
    links = []
    for n in links_n:
        el = ET.Element("link", {"name": n})
        ine = ET.SubElement(el, "inertial")
        ET.SubElement(ine, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(ine, "mass", {"value": f"{F.VJ_LINK_MASS:g}"})
        i = f"{F.VJ_LINK_INERTIA:g}"
        ET.SubElement(ine, "inertia", {"ixx": i, "ixy": "0", "ixz": "0", "iyy": i, "iyz": "0", "izz": i})
        links.append(el)
    g = F.WRIST_GAINS
    joints = []
    for k, (name, (tag, kind, axis)) in enumerate(zip(joints_n, F.VJ_AXES, strict=True)):
        child = links_n[k + 1] if k + 1 < len(links_n) else hand_root
        el = ET.Element("joint", {"name": name, "type": kind})
        ET.SubElement(el, "parent", {"link": links_n[k]})
        ET.SubElement(el, "child", {"link": child})
        ET.SubElement(el, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(el, "axis", {"xyz": " ".join(f"{v:g}" for v in axis)})
        lim = F.VJ_LIN_LIMIT_M if kind == "prismatic" else F.VJ_ROT_MID_LIMIT if tag == "rx" else F.VJ_ROT_OUTER_LIMIT
        eff = g.lin_force_max if kind == "prismatic" else g.rot_torque_max
        ET.SubElement(
            el, "limit", {"lower": f"{-lim:.6f}", "upper": f"{lim:.6f}", "effort": f"{eff:g}", "velocity": "1000"}
        )
        joints.append(el)
    return links, joints


def cut(root_el: ET.Element, side: str):
    """The subtree below the child of ``{side}_mount_joint``: (root name, link elements, joint elements), source
    order.
    """
    joints = root_el.findall("joint")
    mount = [j for j in joints if j.get("name") == f"{side}_mount_joint"]
    if len(mount) != 1:
        raise SystemExit(f"[FLOATING_URDF] {len(mount)} {side}_mount_joint — not a canonical combo URDF")
    root = mount[0].find("child").get("link")
    kids: dict[str, list[ET.Element]] = {}
    for j in joints:
        kids.setdefault(j.find("parent").get("link"), []).append(j)
    keep_l, keep_j, st = {root}, [], [root]
    while st:
        n = st.pop()
        for j in kids.get(n, []):
            keep_j.append(j)
            c = j.find("child").get("link")
            keep_l.add(c)
            st.append(c)
    links = [el for el in root_el.findall("link") if el.get("name") in keep_l]
    jset = {id(j) for j in keep_j}
    joints_out = [j for j in joints if id(j) in jset]
    return root, links, joints_out


def build(hand: str, write: bool, wrist_joints: bool = False) -> int:
    src = source_urdf(hand)
    src_root = ET.parse(src).getroot()
    bad = 0
    for side, out in out_paths(hand, wrist_joints).items():
        root, links, joints = cut(src_root, side)
        want_root = F.root_body(hand, side)  # the rule (mount joint child) — checked against the cut root
        if wrist_joints:  # joint drive: the virtual chain in front of the hand subtree — root = the anchor
            vl, vj = wrist_chain(side, root)
            links, joints = vl + links, vj + joints
            want_root = F.vj_link_names(side)[0]
        robot = ET.Element("robot", {"name": out.stem})
        for el in links + joints:
            robot.append(copy.deepcopy(el))
        ET.indent(robot, space="  ")
        text = (
            '<?xml version="1.0"?>\n'
            "<!-- GENERATED by experiments/ablations/floating_hand/make_floating_urdf.py from "
            f"{src.relative_to(REPO)} -->\n"
            f"<!-- floating hand ({side}): subtree under {side}_mount_joint child '{root}' "
            + (
                "— 6 virtual wrist joints from a world-fixed anchor (DEXENGINE_FH_WRIST=joint), no arm -->\n"
                if wrist_joints
                else "— free root, no arm -->\n"
            )
            + ET.tostring(robot, encoding="unicode")
            + "\n"
        )
        if write:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text)
        # ── checks ──
        chk = ET.fromstring(text.split("\n", 3)[3]) if write is False else ET.parse(out).getroot()
        n_l, n_j = len(chk.findall("link")), len(chk.findall("joint"))

        def _canon(e):  # elements that differ only in indentation (tail/text whitespace) are the same element
            e = copy.deepcopy(e)
            for x in e.iter():
                x.tail = None
                if x.text is not None and not x.text.strip():
                    x.text = None
            return ET.tostring(e)

        src_by = {(e.tag, e.get("name")): _canon(e) for e in links + joints}
        out_by = {(e.tag, e.get("name")): _canon(e) for e in chk if e.tag in ("link", "joint")}
        same = src_by == out_by
        meshes = [m.get("filename") for m in chk.iter("mesh")]
        missing = [m for m in meshes if not (out.parent / m).resolve().is_file()]
        other = "left" if side == "right" else "right"
        stray = [
            e.get("name")
            for e in chk.findall("link")
            if e.get("name").startswith(f"{other}_") or e.get("name").startswith("link_")
        ]
        children = {j.find("child").get("link") for j in chk.findall("joint")}
        roots = {e.get("name") for e in chk.findall("link")} - children
        ok = (
            n_l == len(links)
            and n_j == len(joints)
            and same
            and not missing
            and not stray
            and roots == {want_root}
            and n_l == n_j + 1
        )
        bad += not ok
        print(
            f"[FLOATING_URDF] {side}: {out.relative_to(REPO)} root {sorted(roots)} links {n_l} joints {n_j} "
            f"(joint types {sorted({j.get('type') for j in chk.findall('joint')})}) "
            f"elements identical {same} meshes {len(meshes)} "
            f"missing {len(missing)} foreign links {len(stray)} → {'OK' if ok else 'FAIL'}",
            flush=True,
        )
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", required=True, help="hand name (no default)")
    ap.add_argument("--check", action="store_true", help="check only, write nothing")
    ap.add_argument(
        "--print-stem", action="store_true", help="print the <arm>_<hand> stem only (the bake shell reads it)"
    )
    ap.add_argument(
        "--wrist-joints",
        action="store_true",
        help="the virtual-joint body (asset of DEXENGINE_FH_WRIST=joint · stem <arm>_<hand>_vj)",
    )
    a = ap.parse_args()
    if a.print_stem:
        print(out_paths(a.hand, a.wrist_joints)["right"].parent.name)
        return 0
    return build(a.hand, write=not a.check, wrist_joints=a.wrist_joints)


if __name__ == "__main__":
    sys.exit(main())
