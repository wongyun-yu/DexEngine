#!/usr/bin/env python
"""Split the canonical robot USD into TWO separate articulations (right side + body, left arm + hand).

Why: ``FilteredPairsAPI`` authored on link pairs INSIDE one articulation has no effect in PhysX for this asset. The
links are ``Xform + RigidBodyAPI`` and every collider lives under an instanceable parent
(``link_X/collisions/<mesh>/<shape>``), so the colliders are instance proxies, and API schemas cannot be applied to
proxies (``Apply()`` only warns and returns an invalid schema). A de-instanced (flattened) asset runs out of memory at
scale. The design that works uses no filters at all — with two articulations:

* cross pairs (hand/arm ↔ opposite hand/arm) and left arm/hand ↔ torso = ordinary inter-articulation rigid contacts
  (no filter needed, on automatically);
* inside a hand, inside an arm, parent-child = off via each articulation's ``enabled_self_collisions=False``;
* hand ↔ opposite object, object ↔ object = ordinary rigid pairs, on without changes.

What it makes
-------------
From one canonical ``<combo>.usd`` two files are baked with reference + deactivate only. Nothing is flattened: the
instance prototypes are referenced as they are, so the files are a few KB and scene creation stays fast.

* ``<combo>_split_R.usd`` — torso + head + right arm + right hand. Its root is the SAME ``root_joint`` as the
  canonical one (fixed base); only the left links and joints are ``active=false``.
* ``<combo>_split_L.usd`` — left arm + left hand. The right side, torso and head links and their joints and the
  canonical ``root_joint`` are ``active=false``, and two new prims are made:
    - ``link_<side>_mount`` : the left shoulder mount link, a collider-free rigid body with EXACTLY the transform of
      the base link (the canonical articulation root, here ``world``);
    - ``root_joint_L``      : a fixed joint attaching that mount to the world (+ArticulationRootAPI), copying the
      canonical ``root_joint``'s physxArticulation attributes.
  Only ``body0`` of the branch joint (``left_arm_0``) moves from the base link to the mount. The mount has the base's
  transform, so the joint's ``localPos0/localRot0`` are unchanged and none of the 7 left-arm DOF is lost (action and
  observation layouts are unchanged).

Both articulations share the robot frame: writing the SAME root pose to each reproduces the original relative
placement (keeping them in sync at runtime is the scene's job — writing the root z of only one puts the sides at
different heights).

The canonical file is never touched; only new files in a new folder are written.

Usage::

    python dexengine/robots/usd/split_arm.py \\
        --src data/usd_cache/robots/full_robot_gen/rby1_sharpa_wave/rby1_sharpa_wave.usd

    # check an already baked pair only (no writes)
    python dexengine/robots/usd/split_arm.py --src ... --verify-only

Output: ``<combo>_split/{<combo>_split_R.usd, <combo>_split_L.usd}``. Runtime switch: ``DEXENGINE_BI_SPLIT_ARTIC=1``
(default 0).

Limitation (read before judging results)
----------------------------------------
This two-way split assigns the torso to the right side. Right arm/hand ↔ torso therefore stay inside one
articulation and, with self-collision permanently off, are NOT in contact; only left arm/hand ↔ torso become
inter-articulation contacts. Removing this asymmetry against the canonical collision set (arm/hand ↔ torso on, both
sides) needs a three-way split (torso as its own articulation), which does not exist yet.
"""

from __future__ import annotations

import argparse
import os
import pathlib

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from dexengine.robots.usd.split_arm_check import joint_bodies, verify


def survey(stage: Usd.Stage) -> dict:
    """Survey of the canonical stage: links, joints, root, tree."""
    dp = stage.GetDefaultPrim()
    if not dp:
        raise SystemExit("[split] no defaultPrim — is this a robot USD?")
    rng = list(Usd.PrimRange(dp, Usd.TraverseInstanceProxies()))

    links = [p.GetPath().pathString for p in rng if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints, root_joints = [], []
    for p in rng:
        if not p.IsA(UsdPhysics.Joint):
            continue
        b0, b1 = joint_bodies(p)
        rec = {
            "path": p.GetPath().pathString,
            "name": p.GetName(),
            "type": p.GetTypeName(),
            "b0": b0,
            "b1": b1,
            "artic_root": bool(p.HasAPI(UsdPhysics.ArticulationRootAPI)),
        }
        (root_joints if rec["artic_root"] else joints).append(rec)

    if len(root_joints) != 1:
        raise SystemExit(
            f"[split] {len(root_joints)} ArticulationRootAPI joints (expected 1). "
            "Already split, or an unexpected structure."
        )
    rj = root_joints[0]
    base_link = rj["b1"]
    if base_link is None or base_link not in links:
        raise SystemExit(f"[split] could not resolve the root joint's body1 as the base link: {rj}")

    # parent (body0) -> child (body1) graph
    children: dict[str, list[dict]] = {}
    for j in joints:
        if j["b0"] is None or j["b1"] is None:
            raise SystemExit(f"[split] non-root joint with empty body0/body1: {j['path']}")
        children.setdefault(j["b0"], []).append(j)
    return {
        "default_prim": dp,
        "links": links,
        "joints": joints,
        "root_joint": rj,
        "base_link": base_link,
        "children": children,
    }


def subtree(children: dict, root: str) -> list[str]:
    """Every link reachable from `root` through joints (root included)."""
    out, stack = [], [root]
    while stack:
        cur = stack.pop()
        if cur in out:
            continue
        out.append(cur)
        stack.extend(j["b1"] for j in children.get(cur, ()))
    return out


def pick_branch(s: dict, token: str) -> dict:
    """The one branch from the base link whose joint name contains ``token``."""
    branches = s["children"].get(s["base_link"], [])
    hit = [j for j in branches if token in j["name"].lower()]
    if len(hit) != 1:
        names = [j["name"] for j in branches]
        raise SystemExit(
            f"[split] {len(hit)} branches of the base link ({s['base_link'].split('/')[-1]}) have "
            f"'{token}' in their joint name (expected 1). Branches: {names}"
        )
    return hit[0]


# ---------------------------------------------------------------------------
# Bake
# ---------------------------------------------------------------------------
def _new_stage(dst: pathlib.Path, src: pathlib.Path, dp_name: str) -> Usd.Stage:
    """A thin stage that references src's defaultPrim. Not flattened."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    st = Usd.Stage.CreateNew(str(dst))
    root = UsdGeom.Xform.Define(st, f"/{dp_name}")
    rel = os.path.relpath(src.resolve(), dst.parent.resolve())
    root.GetPrim().GetReferences().AddReference(rel, f"/{dp_name}")
    st.SetDefaultPrim(root.GetPrim())
    # Inherit the canonical up axis and units (a reference does not carry stage metadata).
    src_st = Usd.Stage.Open(str(src))
    UsdGeom.SetStageUpAxis(st, UsdGeom.GetStageUpAxis(src_st))
    UsdGeom.SetStageMetersPerUnit(st, UsdGeom.GetStageMetersPerUnit(src_st))
    return st


def _deactivate(st: Usd.Stage, paths) -> int:
    n = 0
    for p in paths:
        prim = st.GetPrimAtPath(p)
        if not prim.IsValid():
            raise SystemExit(f"[split] prim to deactivate is missing in the copy: {p}")
        prim.SetActive(False)
        n += 1
    return n


def _copy_xform(src_prim: Usd.Prim, dst_prim: Usd.Prim) -> None:
    """Copy the xformOp values verbatim (mount = the base's transform)."""
    sx, dx = UsdGeom.Xformable(src_prim), UsdGeom.Xformable(dst_prim)
    dx.ClearXformOpOrder()
    for op in sx.GetOrderedXformOps():
        parts = op.GetOpName().split(":")  # xformOp:translate[:suffix]
        new = dx.AddXformOp(op.GetOpType(), op.GetPrecision(), parts[2] if len(parts) > 2 else "")
        new.Set(op.Get())


def _add_left_mount(dp_name, mount_name, st_l, src_st, s, mount_mass):
    """L stage: the mount link (copied base pose, fixed mass/inertia) and a fixed root joint to the world with the
    source root joint's attributes and Physx schemas."""
    mount_path = f"/{dp_name}/{mount_name}"
    mount = UsdGeom.Xform.Define(st_l, mount_path).GetPrim()
    _copy_xform(src_st.GetPrimAtPath(s["base_link"]), mount)
    UsdPhysics.RigidBodyAPI.Apply(mount)
    mapi = UsdPhysics.MassAPI.Apply(mount)
    mapi.CreateMassAttr().Set(float(mount_mass))
    # A non-degenerate inertia (10 cm radius of gyration); PhysX ignores it on a fixed-base root.
    _i = float(mount_mass) * 1e-2
    mapi.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(_i, _i, _i))
    mapi.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    mapi.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    rj_src = src_st.GetPrimAtPath(s["root_joint"]["path"])
    rj_l_path = f"/{dp_name}/{s['root_joint']['name']}_L"
    rj_l = UsdPhysics.FixedJoint.Define(st_l, rj_l_path)
    rj_l.CreateBody1Rel().SetTargets([Sdf.Path(mount_path)])  # empty body0 = fixed to the world
    for attr in rj_src.GetAttributes():
        n, v = attr.GetName(), attr.Get()
        if v is None or n in ("visibility", "purpose"):
            continue
        # custom/variability as in the source — physxArticulation:* are non-custom in the canonical body.
        new = rj_l.GetPrim().CreateAttribute(n, attr.GetTypeName(), attr.IsCustom(), attr.GetVariability())
        new.Set(v)
    UsdPhysics.ArticulationRootAPI.Apply(rj_l.GetPrim())
    for sch in rj_src.GetAppliedSchemas():  # PhysxArticulationAPI etc.
        if sch.startswith("Physx"):
            rj_l.GetPrim().AddAppliedSchema(sch)
    return mount_path


def bake(src: pathlib.Path, out_dir: pathlib.Path, suffix: str, token: str, mount_mass: float, force: bool) -> dict:
    src_st = Usd.Stage.Open(str(src))
    s = survey(src_st)
    dp_name = s["default_prim"].GetName()

    split_joint = pick_branch(s, token)
    l_links = subtree(s["children"], split_joint["b1"])
    l_link_set = set(l_links)
    l_joints = [j for j in s["joints"] if j["b1"] in l_link_set]
    r_links = [p for p in s["links"] if p not in l_link_set]
    r_joints = [j for j in s["joints"] if j["b1"] not in l_link_set]

    # Integrity: every left joint except the branch joint has both ends on the left.
    for j in l_joints:
        if j is split_joint:
            continue
        if j["b0"] not in l_link_set:
            raise SystemExit(f"[split] left joint points at a body0 outside the left side: {j['path']} -> {j['b0']}")
    for j in r_joints:
        if j["b0"] in l_link_set or j["b1"] in l_link_set:
            raise SystemExit(f"[split] right joint points at a left body: {j['path']}")

    mount_name = f"link_{token}_mount"
    dst_r = out_dir / f"{src.stem}{suffix}_R.usd"
    dst_l = out_dir / f"{src.stem}{suffix}_L.usd"
    for d in (dst_r, dst_l):
        if d.exists():
            if not force:
                raise SystemExit(f"[split] already exists (--force to overwrite): {d}")
            d.unlink()

    # ---- R: torso + head + right arm + right hand (root = the canonical root_joint) ----
    st_r = _new_stage(dst_r, src, dp_name)
    _deactivate(st_r, [j["path"] for j in l_joints])
    _deactivate(st_r, l_links)
    st_r.GetRootLayer().Save()

    # ---- L: left arm + left hand (+ mount link and a new fixed root joint) ----
    st_l = _new_stage(dst_l, src, dp_name)
    _deactivate(st_l, [j["path"] for j in r_joints])
    _deactivate(st_l, r_links)
    _deactivate(st_l, [s["root_joint"]["path"]])

    mount_path = _add_left_mount(dp_name, mount_name, st_l, src_st, s, mount_mass)

    # Move only the branch joint's body0 from the base to the mount (local poses unchanged).
    sj = st_l.GetPrimAtPath(split_joint["path"])
    UsdPhysics.Joint(sj).CreateBody0Rel().SetTargets([Sdf.Path(mount_path)])
    st_l.GetRootLayer().Save()

    return {
        "src": src,
        "dst_r": dst_r,
        "dst_l": dst_l,
        "survey": s,
        "split_joint": split_joint,
        "mount_path": mount_path,
        "mount_mass": mount_mass,
        "l_links": l_links,
        "r_links": r_links,
        "l_joints": l_joints,
        "r_joints": r_joints,
        "dp_name": dp_name,
    }


def default_src() -> str:
    """Canonical combo USD path — the same rule as at runtime (combo = arm_hand + variant suffix).

    At runtime the canonical path is ``scene.robot.spawn.usd_path`` built by ``robots.selection``, and the split assets
    derive from it as ``<stem>_split/<stem>_split_{R,L}.usd`` (``split_artic.split_usd_paths``). Here the same rule is
    assembled from the environment so that Isaac is not started.
    """
    from dexengine.core import arms as _AS  # noqa: PLC0415
    from dexengine.core.paths import main_root  # noqa: PLC0415

    arm = _AS.active_arm_name()
    hand = _AS.active_hand_name()
    variant = os.environ.get("DEXENGINE_COMBO_USD_VARIANT") or _AS.sim_usd_variant(arm, hand)
    combo = f"{arm}_{hand}{variant}"
    # Robot USDs live under the data root (not relative to the working directory).
    return str(main_root() / "data/usd_cache/robots/full_robot_gen" / combo / f"{combo}.usd")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--src",
        default=None,
        help="canonical robot USD (opened read-only). Default: assembled from "
        "DEXENGINE_ARM/DEXENGINE_HAND/DEXENGINE_COMBO_USD_VARIANT",
    )
    ap.add_argument("--suffix", default="_split", help="new variant suffix (default _split)")
    ap.add_argument(
        "--side-token", default="left", help="joint-name token that picks the branch to split off (default left)"
    )
    ap.add_argument(
        "--mount-mass",
        type=float,
        default=1e-3,
        help="left mount link mass in kg (ignored by PhysX on a fixed base; default 0.001)",
    )
    ap.add_argument("--force", action="store_true", help="overwrite existing copies")
    ap.add_argument("--verify-only", action="store_true", help="do not bake; only check existing copies")
    args = ap.parse_args()

    src = pathlib.Path(args.src or default_src()).resolve()
    if not src.is_file():
        if args.verify_only:  # a checkout without assets skips
            print(f"[split] SKIP — no canonical asset: {src}")
            return 0
        raise SystemExit(f"[split] source missing: {src}")
    out_dir = src.parent.parent / f"{src.stem}{args.suffix}"

    if args.verify_only:
        src_st = Usd.Stage.Open(str(src))
        s = survey(src_st)
        sj = pick_branch(s, args.side_token)
        l_links = subtree(s["children"], sj["b1"])
        info = {
            "src": src,
            "dst_r": out_dir / f"{src.stem}{args.suffix}_R.usd",
            "dst_l": out_dir / f"{src.stem}{args.suffix}_L.usd",
            "mount_mass": args.mount_mass,
            "mount_path": f"/{s['default_prim'].GetName()}/link_{args.side_token}_mount",
        }
        for d in (info["dst_r"], info["dst_l"]):
            if not d.is_file():
                print(f"[split] SKIP — no copy to check (bake first): {d}")
                return 0
        print(
            f"[split] source : {src}\n[split] copies : {info['dst_r']}\n                 {info['dst_l']}"
            f"\n[split] left links {len(l_links)} (branch joint {sj['name']})"
        )
        return 0 if verify(info) else 1

    print(f"[split] source : {src}")
    info = bake(src, out_dir, args.suffix, args.side_token, args.mount_mass, args.force)
    print(f"[split] copies : {info['dst_r']}\n                 {info['dst_l']}")
    print(
        f"[split] branch joint : {info['split_joint']['name']} "
        f"(body0 {info['survey']['base_link'].split('/')[-1]} -> {info['mount_path'].split('/')[-1]})"
    )
    print(
        f"[split] R links {len(info['r_links'])} / joints {len(info['r_joints'])}  ·  "
        f"L links {len(info['l_links'])} / joints {len(info['l_joints'])}"
    )
    ok = verify(info)
    print(
        "\n[split] "
        + (
            "identity check passed — the two articulation assets are ready."
            if ok
            else "*** identity check FAILED — do not use these assets ***"
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
