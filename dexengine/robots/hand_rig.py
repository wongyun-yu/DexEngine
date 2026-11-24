#!/usr/bin/env python3
"""Build a PATH A hand rig MJCF from a vendored URDF, driven by HandSpec.

WHY
    The hand tracker retargets MANO onto a robot hand and emits the qpos our references are built from. It needs a
    rig that follows the PATH A contract (6-DoF base, palm and fingertip sites, collision geoms, actuator order,
    side-prefixed meshes). The MuJoCo boilerplate every rig shares is `rig/template.py`. This is one
    converter for every hand; the hand-specific facts come from HandSpec, which is already the single source for
    them.

HOW
    MuJoCo reads URDF natively, so the hard part -- inertia, mesh refs, joint frames -- is not re-implemented here.
    This script:
      1. hands the URDF to MuJoCo (injecting an absolute meshdir, without touching the vendored file) and saves the
         compiled MJCF (``rig/urdf_kin.py``),
      2. applies the PATH A convention on top: 6-DoF base, palm and fingertip sites (``rig/sites.py``), collision
         geoms, actuator order, side-prefixed meshes.

THE PART THAT NEEDS CARE
    MuJoCo MERGES links joined by fixed joints, so the body a HandSpec names as a fingertip usually does not survive
    into the MJCF (Wuji loses all five `*_tip_link` plus the palm). The fingertip SITE therefore cannot just be attached
    to the named body; it has to be placed on the nearest surviving ancestor, at the offset that lands on the same
    physical point. That offset is derived by walking the URDF's fixed-joint chain, and then CHECKED against MuJoCo's
    own forward kinematics -- if the two disagree the script refuses to write.

USAGE
    <retarget venv python> dexengine/robots/hand_rig.py wuji2
    <retarget venv python> dexengine/robots/hand_rig.py wuji2 --dry-run
    Needs mujoco + scipy, i.e. the retarget venv (paths.py --what retarget-py), not the isaaclab env.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from dexengine.core import paths as _P
from dexengine.core.errors import InputMissingError
from dexengine.robots.rig import from_mjcf
from dexengine.robots.rig import sites as S
from dexengine.robots.rig import template as TPL
from dexengine.robots.rig.sites import (  # noqa: F401 — the builder's helpers, importable from here
    RRC_MANO_AXES,
    _fmt,
    _palm_site_offset,
    _palm_site_quat,
    _site_offsets,
    check_tip_sites,
    load_specs,
    pad_gap,
    pad_lateral_max,
    tip_site_ctx,
    vendor_link_name,
    vendor_of,
    vendor_paths,
)
from dexengine.robots.rig.urdf_kin import Urdf, compile_mjcf, resolve_site  # noqa: F401


def rig_out_dir(rig_root: pathlib.Path | None, name: str) -> pathlib.Path:
    """Folder the rig `name` is written to: the canonical rig folder (`paths.hand_rig_dir`), or `<rig_root>/<name>`
    when `--spider-root` names another root."""
    return _P.hand_rig_dir(name) if rig_root is None else rig_root / name


FINGERS = ("thumb", "index", "middle", "ring", "pinky")

_FORCE = False


def _el(tag, **kw):
    return ET.Element(tag, {k: str(v) for k, v in kw.items()})


def _report_placement_and_write(
    palm_link, palm_host, palm_pos, palm_off, spec, placed, dry_run, out_dir, side, hand, copies, root, rig_name
):
    print(
        f"         palm  {palm_link} -> host {palm_host} @ {_fmt(palm_pos)}"
        f"  [PALM_SITE_OFFSET] {_fmt(palm_off)} (spec right-side declaration "
        f"{_fmt(getattr(spec, 'palm_site_offset', (0.0, 0.0, 0.0)))})"
    )
    for finger, tip, host, pos in placed:
        note = "" if host == tip else "  (merged -> offset derived)"
        print(f"         {finger:7s} {tip} -> host {host} @ {_fmt(pos)}{note}")

    if dry_run:
        return None

    # Never clobber an MJCF this script did not produce. Rig folders can hold
    # vendor work that is not regenerable from anything we have.
    out_path = out_dir / f"{side}.xml"
    # A locked rig is never overwritten: when layout.RETARGET_HAND_ASSETS declares the folder's identity
    # (sha256_xml3) and the current rig has it, a rebuilt rig would break the training entry's rig check
    # (tree_rig_check) and force data regeneration — even when it carries this tool's marker (model=…_spider).
    # Rebuild only with --force (then change the registry hash in the same commit).
    _lock = _P.data_layout.RETARGET_HAND_ASSETS.get(rig_name)
    if (
        not _FORCE
        and _lock is not None
        and out_dir.resolve() == _P.hand_rig_dir(rig_name).resolve()
        and all((out_dir / n).is_file() for n in ("right.xml", "left.xml", "bimanual.xml"))
        and _P.data_layout.asset_hash(str(out_dir)) == _lock.sha256_xml3
    ):
        raise FileExistsError(
            f"{out_dir} is a locked rig (layout.RETARGET_HAND_ASSETS[{rig_name!r}] = "
            f"{_lock.sha256_xml3[:12]}…) — overwriting it splits its identity from every output made with it. "
            "Build with another --out-name/--spider-root, or pass --force only to really replace it."
        )
    if not _FORCE and out_path.exists() and f'model="{side}_{hand}_spider"' not in out_path.read_text():
        raise FileExistsError(
            f"{out_path} exists and was NOT produced by this script "
            f'(no model="{side}_{hand}_spider"). Refusing to overwrite. '
            f"Pick a different --out-name, or pass --force if you really mean it."
        )

    (out_dir / "assets" / side).mkdir(parents=True, exist_ok=True)
    for src, dst in copies:
        if not dst.exists():
            shutil.copy2(src, dst)
    ET.indent(root, space="  ")
    out_path.write_text(ET.tostring(root, encoding="unicode") + "\n")
    return out_path


def _rename_meshes(compiled, side, meshdir, out_dir, asset):
    """Side-prefixed mesh assets; returns (old → new mesh name, (source, destination) file copies)."""
    mesh_rename = {}
    copies = []
    for mesh in compiled.find("asset").findall("mesh"):
        name, fname = mesh.get("name"), os.path.basename(mesh.get("file"))
        new = name if name.startswith(side) else f"{side}_{name}"
        mesh_rename[name] = new
        copies.append((meshdir / fname, out_dir / "assets" / side / fname))
        ET.SubElement(asset, "mesh", {"name": new, "file": f"{side}/{fname}"})
    return mesh_rename, copies


def _synth_base_body(compiled, urdf, surviving, mesh_rename, side):
    """The single hand body that adopts the compiled worldbody children (the URDF root never survives as a body); geoms
    get no contacts and side-prefixed names. Returns (base, bodies by name)."""
    # The URDF ROOT link never survives as a body: MuJoCo attaches it to `world`,
    # so its geoms and every finger chain end up directly under <worldbody>
    # (measured: Wuji leaves 2 palm geoms + 5 finger bodies there). the rig needs a
    # single hand body to hang the 6-DoF base off, so synthesise it here and adopt
    # those children. The compiled world frame IS the URDF root frame, so the
    # children keep their positions and URDF-chain offsets stay valid.
    compiled_world = compiled.find("worldbody")
    base = ET.Element("body", {"name": urdf.root_link, "pos": "0 0 0", "quat": TPL.base_body_quat(), "gravcomp": "1"})
    adopted = [c for c in list(compiled_world) if c.tag in ("body", "geom", "site")]
    for c in adopted:
        base.append(ET.fromstring(ET.tostring(c)))
    if not any(c.tag == "body" for c in adopted):
        raise ValueError("compiled MJCF has no body under worldbody")
    surviving.add(urdf.root_link)  # the synthesised body stands in for the root

    for geom in base.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
        if geom.get("mesh") in mesh_rename:
            geom.set("mesh", mesh_rename[geom.get("mesh")])
        gname = geom.get("name")
        if gname and not gname.startswith((side, "collision_hand_")):
            geom.set("name", f"{side}_{gname}")

    bodies = {b.get("name"): b for b in base.iter("body")}
    bodies[base.get("name")] = base
    return base, bodies


def _add_palm_site(urdf, model, surviving, palm_link, palm_off, side, base, bodies, palm_quat):
    """Palm site (the IK welds it to the MANO wrist) and the palm collision box; inserts the tracking camera and base
    joints first. Returns (palm host, palm position)."""
    # palm site: on the surviving host of the spec's palm link, at the spec's palm-local
    # offset. That offset is NOT cosmetic — the IK welds this site to the MANO wrist, so
    # HandSpec.palm_site_offset decides the systematic part of the palm error.
    palm_host, palm_pos = resolve_site(urdf, model, surviving, palm_link, palm_off)
    head = TPL.base_head(side)  # tracking camera, then the six base joints
    for i, el in enumerate(head):
        base.insert(i, el)

    palm_body = bodies[palm_host]
    palm_attrs = {"name": f"{side}_palm", "pos": _fmt(palm_pos), "type": "box", "size": "0.02 0.03 0.04"}
    if palm_quat is not None:
        if palm_host != palm_link:
            raise ValueError(
                f"palm_site_mano_frame applies only when the site sits on {palm_link!r} (now {palm_host!r})"
            )
        palm_attrs["quat"] = _fmt(np.round(palm_quat, 7))
    ET.SubElement(palm_body, "site", palm_attrs)
    ET.SubElement(
        palm_body,
        "geom",
        {
            "type": "box",
            "name": f"collision_hand_{side}_palm_0",
            "size": "0.02 0.03 0.05",
            "pos": _fmt(palm_pos),
            "contype": "0",
            "conaffinity": "0",
            "group": "3",
            "density": "0",
            "rgba": "0 1 0 1",
        },
    )
    return palm_host, palm_pos


def _add_tip_sites(spec_tips, offsets, urdf, model, surviving, bodies, side):
    """Fingertip sites (plain, track_hand_, trace_hand_) and a distal collision capsule per finger. Returns the
    placements."""
    placed = []
    for finger, tip_link, off in zip(FINGERS, spec_tips, offsets, strict=False):
        host, pos = resolve_site(urdf, model, surviving, tip_link, off)
        body = bodies[host]
        for prefix in ("", "track_hand_", "trace_hand_"):
            ET.SubElement(body, "site", {"name": f"{prefix}{side}_{finger}_tip", "pos": _fmt(pos)})
        # Capsule from the host body origin to the pad, so its length is the real
        # distal segment rather than a per-hand constant.
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"collision_hand_{side}_{finger}_0",
                "type": "capsule",
                "fromto": f"0 0 0 {_fmt(pos)}",
                "size": "0.008",
                "contype": "0",
                "conaffinity": "0",
                "group": "3",
                "density": "0",
                "rgba": "0 1 0 1",
            },
        )
        placed.append((finger, tip_link, host, pos))
    return placed


def _prefix_side(base, side, spec_joints, palm_host):
    """Side prefix on bodies and joints of a vendor URDF with plain names (they would collide in the bimanual scene).
    Returns the renamed spec joints and palm host."""
    # side prefix on bodies/joints
    # A vendor URDF that already spells links `{side}_…` needs nothing here and the guard below
    # makes this a no-op (wuji: 26/27 already prefixed). A single-hand URDF with plain names
    # (DG-5F-S: 0/27) would otherwise emit `link_base`/`joint_1_1`, which COLLIDES with the other
    # hand in the bimanual scene and does not match the names HandSpec declares.
    renamed = {}
    for el in list(base.iter("body")) + list(base.iter("joint")):
        n = el.get("name")
        if n and not n.startswith(f"{side}_"):
            renamed[n] = f"{side}_{n}"
            el.set("name", renamed[n])
    if renamed:
        print(f"  [{side}] the vendor URDF has no side prefix — renamed {len(renamed)} bodies/joints to {side}_*")
    spec_joints = [renamed.get(j, j) for j in spec_joints]
    palm_host = renamed.get(palm_host, palm_host)
    return spec_joints, palm_host


def _add_actuators(root, side, base, spec_joints):
    """Six base actuators, then one position actuator per HandSpec joint; the compiled joint order must equal the spec
    order. Returns the compiled finger joints."""
    # actuators: 6 base first (hand-tracker contract), then the spec's joints
    actuator = ET.SubElement(root, "actuator")
    TPL.add_base_actuators(actuator, side)
    mjcf_joints = [j.get("name") for j in base.iter("joint")][6:]
    missing = [j for j in spec_joints if j not in mjcf_joints]
    if missing:
        raise ValueError(f"HandSpec joints absent from the compiled MJCF: {missing}")
    # The order is checked too: the contract is HandSpec joint order = dataset qpos order (after the 6 virtual wrist
    # DoFs). Checking only the subset would pass a wrong order and route finger values silently to other joints.
    if mjcf_joints != spec_joints:
        raise ValueError(
            f"HandSpec finger_joint_names_right order differs from the compiled MJCF joint order — "
            f"spec {spec_joints[:4]}… vs mjcf {mjcf_joints[:4]}… (length {len(spec_joints)} vs {len(mjcf_joints)})"
        )
    for j in spec_joints:
        ET.SubElement(actuator, "position", {"name": f"{j}_position", "joint": j})
    return mjcf_joints


def _build_from_vendor_mjcf(hand, side, spec, vend, rig_root, dry_run) -> pathlib.Path | None:
    """The rig from the vendor's own MJCF (rig/from_mjcf.py); the robot body stays URDF-built."""
    import mujoco

    mjcf_path = _P.repo_rel_path(spec.vendor_mjcf.format(side=side))
    if not mjcf_path.exists():
        raise FileNotFoundError(f"{mjcf_path} — run `python -m dexengine.robots.vendor fetch`")
    out_dir = rig_out_dir(rig_root, vend["out"])
    root, copies, placed = from_mjcf.build(hand, side, spec, mjcf_path, out_dir)
    palm = tuple(spec.rig_palm_site)
    placed = [(f, host, host, (off, 0.0, 0.0)) for f, host, off in placed]
    print(f"  [{side}] rig from the vendor MJCF {mjcf_path.name}: meshes {len(copies)}")
    out = _report_placement_and_write(
        spec.palm_fk_body_right, root.find("worldbody/body").get("name"), palm, palm, spec, placed, dry_run,
        out_dir, side, hand, copies, root, vend["out"],
    )  # fmt: skip
    if out is not None:
        mujoco.MjModel.from_xml_path(str(out))  # the written rig compiles (missing mesh · bad element → raises)
    return out


def build(
    hand: str, side: str, spec, rig_root: pathlib.Path | None, tmpdir: pathlib.Path, dry_run: bool
) -> pathlib.Path | None:
    import mujoco

    vend = vendor_of(hand, load_specs())
    if spec.vendor_mjcf:
        return _build_from_vendor_mjcf(hand, side, spec, vend, rig_root, dry_run)
    urdf_path, meshdir = vendor_paths(vend, side)
    if not urdf_path.exists():
        raise FileNotFoundError(urdf_path)

    urdf = Urdf(urdf_path)
    model, compiled = compile_mjcf(urdf_path, meshdir, tmpdir)
    surviving = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)} - {"world"}

    def _vname_side(n: str, sd: str) -> str:
        return vendor_link_name(spec, n, sd)

    def _vname(n: str) -> str:
        return _vname_side(n, side)

    def _urdf_of_side(sd: str) -> Urdf:
        return urdf if sd == side else Urdf(vendor_paths(vend, sd)[0])

    spec_tips = [_vname(n) for n in spec.tip_body_names_right]
    spec_joints = [_vname(n) for n in spec.finger_joint_names_right]
    palm_link = _vname(spec.palm_fk_body_right)
    offsets = _site_offsets(spec, side)
    # The fingertip check measures both sides together (left/right agreement catches a wrong lateral sign).
    check_tip_sites(spec, {sd: tip_site_ctx(hand, spec, sd) for sd in ("right", "left")})
    palm_off = _palm_site_offset(spec, side, _urdf_of_side, _vname_side)
    palm_quat = _palm_site_quat(spec, side, palm_link, _vname_side)

    # skeleton
    root = ET.Element("mujoco", {"model": f"{side}_{hand}_spider"})
    TPL.add_skeleton(root)

    # meshes: side-prefixed names, files copied under assets/{side}/
    asset = ET.SubElement(root, "asset")
    TPL.add_world_assets(asset, side)

    out_dir = rig_out_dir(rig_root, vend["out"])
    mesh_rename, copies = _rename_meshes(compiled, side, meshdir, out_dir, asset)

    # body tree
    worldbody = ET.SubElement(root, "worldbody")
    TPL.add_light(worldbody)
    base, bodies = _synth_base_body(compiled, urdf, surviving, mesh_rename, side)

    palm_host, palm_pos = _add_palm_site(urdf, model, surviving, palm_link, palm_off, side, base, bodies, palm_quat)

    placed = _add_tip_sites(spec_tips, offsets, urdf, model, surviving, bodies, side)

    worldbody.append(base)

    spec_joints, palm_host = _prefix_side(base, side, spec_joints, palm_host)

    mjcf_joints = _add_actuators(root, side, base, spec_joints)

    print(
        f"  [{side}] bodies {len(surviving)} (URDF links {len(urdf.links)}), "
        f"meshes {len(mesh_rename)}, joints {len(mjcf_joints)}, "
        f"actuators {6 + len(spec_joints)}"
    )
    return _report_placement_and_write(
        palm_link, palm_host, palm_pos, palm_off, spec, placed, dry_run, out_dir, side, hand, copies, root, vend["out"]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _HS = load_specs()
    ap.add_argument("hand", choices=sorted(k for k, v in _HS.HAND_REGISTRY.items() if v))
    ap.add_argument("--sides", default="right,left")
    ap.add_argument(
        "--spider-root",
        type=pathlib.Path,
        default=None,
        help="write the rig to <root>/<name>/ instead of the canonical assets/robots/hands/<name>/mjcf",
    )
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    ap.add_argument("--out-name", help="rig folder name to write (default per hand)")
    ap.add_argument("--force", action="store_true", help="overwrite MJCFs this script did not produce")
    ap.add_argument(
        "--tmpdir",
        type=pathlib.Path,
        default=None,
        help="keep the compiled intermediate URDF/XML here (default: a fresh temporary folder per run, removed after)",
    )
    a = ap.parse_args()

    if a.out_name:
        S._OUT_NAME = a.out_name
    if a.force:
        global _FORCE
        _FORCE = True

    hs = _HS
    # A draft spec is optional (a per-hand table here would raise KeyError for other hands).
    spec = hs.HAND_REGISTRY.get(a.hand) or getattr(hs, f"{a.hand.upper()}_HAND_RIGHT_DRAFT", None)
    if spec is None:
        raise InputMissingError(f"[hand_rig] hand {a.hand!r} has no HandSpec")
    print(
        f"  hand {a.hand}  spec {spec.name}  "
        f"({'registered' if hs.HAND_REGISTRY.get(a.hand) else 'draft, not registered'})"
    )

    written = []
    # A fresh folder per run: a fixed shared path let two builds at once read each other's half-written URDF.
    with tempfile.TemporaryDirectory(prefix="hand_rig_") as fresh:
        tmpdir = a.tmpdir if a.tmpdir is not None else pathlib.Path(fresh)
        for side in a.sides.split(","):
            p = build(a.hand, side, spec, a.spider_root, tmpdir, a.dry_run)
            if p:
                written.append(p)

    if written:
        bim = rig_out_dir(a.spider_root, vendor_of(a.hand, hs)["out"]) / "bimanual.xml"
        bim.write_text(
            '<mujoco model="bimanual">\n    <include file="right.xml" />\n    <include file="left.xml" />\n</mujoco>\n'
        )
        written.append(bim)
        for p in written:
            print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
