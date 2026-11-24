"""Checks of the split arm USDs (dexengine.robots.usd.split_arm): the joint graph reading shared with the survey, the
prim census, world transforms against the canonical USD and dangling joints — `verify` is the one entry."""

from __future__ import annotations

from pxr import Usd, UsdGeom, UsdPhysics


# ---------------------------------------------------------------------------
# Asset survey — no literals: every link/joint classification is derived from the joint graph.
# ---------------------------------------------------------------------------
def joint_bodies(prim: Usd.Prim) -> tuple[str | None, str | None]:
    """Absolute (body0, body1) paths of a joint prim; None when empty (= world)."""
    j = UsdPhysics.Joint(prim)
    b0 = j.GetBody0Rel().GetTargets() if j else []
    b1 = j.GetBody1Rel().GetTargets() if j else []
    return (b0[0].pathString if b0 else None, b1[0].pathString if b1 else None)


# ---------------------------------------------------------------------------
# Identity check — any difference is fatal
# ---------------------------------------------------------------------------
def census(stage: Usd.Stage) -> dict:
    dp = stage.GetDefaultPrim()
    plain = list(Usd.PrimRange(dp))
    proxy = list(Usd.PrimRange(dp, Usd.TraverseInstanceProxies()))
    mass = 0.0
    for p in proxy:
        a = p.GetAttribute("physics:mass")
        if a and a.IsValid() and a.Get() is not None:
            mass += float(a.Get())
    return {
        "RigidBodyAPI links": sum(1 for p in proxy if p.HasAPI(UsdPhysics.RigidBodyAPI)),
        "CollisionAPI (proxies)": sum(1 for p in proxy if p.HasAPI(UsdPhysics.CollisionAPI)),
        "RevoluteJoint": sum(1 for p in proxy if p.GetTypeName() == "PhysicsRevoluteJoint"),
        "PrismaticJoint": sum(1 for p in proxy if p.GetTypeName() == "PhysicsPrismaticJoint"),
        "FixedJoint": sum(1 for p in proxy if p.GetTypeName() == "PhysicsFixedJoint"),
        "ArticulationRootAPI": sum(1 for p in proxy if p.HasAPI(UsdPhysics.ArticulationRootAPI)),
        "instanceable prims": sum(1 for p in plain if p.IsInstanceable()),
        "total mass (kg)": round(mass, 6),
    }


def _world_xforms(stage: Usd.Stage) -> dict:
    """Link name -> world 4x4 (for the unchanged-kinematics check)."""
    dp = stage.GetDefaultPrim()
    out = {}
    for p in Usd.PrimRange(dp, Usd.TraverseInstanceProxies()):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            out[p.GetName()] = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return out


def _dangling(stage: Usd.Stage) -> list[str]:
    """Active joints that point at an inactive/absent body."""
    bad = []
    dp = stage.GetDefaultPrim()
    for p in Usd.PrimRange(dp, Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdPhysics.Joint):
            continue
        for tgt in joint_bodies(p):
            if tgt is None:
                continue
            t = stage.GetPrimAtPath(tgt)
            if not t.IsValid() or not t.IsActive():
                bad.append(f"{p.GetPath()} -> {tgt}")
    return bad


def verify(info: dict) -> bool:
    src_st = Usd.Stage.Open(str(info["src"]))
    r_st = Usd.Stage.Open(str(info["dst_r"]))
    l_st = Usd.Stage.Open(str(info["dst_l"]))
    c0, cr, cl = census(src_st), census(r_st), census(l_st)

    # The mount link is an intended addition: the expected totals include it.
    expect_delta = {
        "RigidBodyAPI links": 1,  # link_<side>_mount
        "FixedJoint": 1,  # root_joint_L
        "ArticulationRootAPI": 1,  # two articulations
        "total mass (kg)": info["mount_mass"],
    }
    print(f"\n[split] identity check — source vs (R + L)   mount mass {info['mount_mass']} kg")
    print(f"    {'item':24s} {'source':>14s} {'R':>10s} {'L':>10s} {'R+L':>14s}  {'expΔ':>7s}  verdict")
    ok = True
    for k in c0:
        a, b, c = c0[k], cr[k], cl[k]
        tot = round(b + c, 6) if isinstance(a, float) else b + c
        d = expect_delta.get(k, 0)
        want = round(a + d, 6) if isinstance(a, float) else a + d
        good = abs(tot - want) < 1e-6 if isinstance(a, float) else tot == want
        ok &= good
        print(f"    {k:24s} {a:>14} {b:>10} {c:>10} {tot:>14}  {d:>7}  {'OK' if good else '*** MISMATCH ***'}")

    # (1) Unchanged kinematics: every link's world transform equals the source's (mount excluded).
    x0, xr, xl = _world_xforms(src_st), _world_xforms(r_st), _world_xforms(l_st)
    worst, worst_name = 0.0, ""
    for name, m in list(xr.items()) + list(xl.items()):
        if name not in x0:
            continue
        d = max(abs(m[i][j] - x0[name][i][j]) for i in range(4) for j in range(4))
        if d > worst:
            worst, worst_name = d, name
    kin_ok = worst < 1e-9
    ok &= kin_ok
    print(
        f"    max link world-transform deviation {worst:.3e} ({worst_name or 'n/a'})  "
        f"{'OK' if kin_ok else '*** KINEMATICS CHANGED ***'}"
    )

    # (2) The link set is split exactly in two (no duplicates, nothing missing)
    names0 = set(x0)
    names_rl = set(xr) | (set(xl) - {info["mount_path"].split("/")[-1]})
    dup = set(xr) & set(xl)
    split_ok = (names0 == names_rl) and not dup
    ok &= split_ok
    print(
        f"    link split: R={len(xr)} L={len(xl)} duplicate={len(dup)} missing={len(names0 - names_rl)}  "
        f"{'OK' if split_ok else '*** SPLIT ERROR ***'}"
    )

    # (3) No dangling joints (active joints pointing at inactive bodies)
    for tag, st in (("R", r_st), ("L", l_st)):
        bad = _dangling(st)
        ok &= not bad
        print(f"    {tag} dangling joints {len(bad)}  {'OK' if not bad else '*** ' + str(bad[:3]) + ' ***'}")

    # (4) Exactly one articulation root per copy
    for tag, st in (("R", r_st), ("L", l_st)):
        n = census(st)["ArticulationRootAPI"]
        ok &= n == 1
        print(f"    {tag} ArticulationRootAPI={n}  {'OK' if n == 1 else '*** expected 1 ***'}")

    # (5) Not flattened — the files are small and instancing is alive
    for tag, d in (("R", info["dst_r"]), ("L", info["dst_l"])):
        kb = d.stat().st_size / 1024.0
        small = kb < 512
        ok &= small
        print(f"    {tag} file {kb:.1f} KB  {'OK (referenced, not flattened)' if small else '*** looks flattened ***'}")

    # (6) DOF count — the evidence that action/observation layouts are unchanged
    print(
        f"    DOF: source {c0['RevoluteJoint']} = R {cr['RevoluteJoint']} + L {cl['RevoluteJoint']}"
        f"  {'OK' if cr['RevoluteJoint'] + cl['RevoluteJoint'] == c0['RevoluteJoint'] else '*** MISMATCH ***'}"
    )
    return bool(ok)
