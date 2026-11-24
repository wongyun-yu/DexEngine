"""Are the joint limits PhysX uses those of the spawned variant's URDF? — marker ``[USD_VARIANT]``.

The spawned variant comes from ``robots.selection.resolve_usd_variant`` (owner
``dexengine.core.arms.plate_variant(arm, hand)`` — the hand's plate carries its joint-limit variant
``HandSpec.sim_finger_limits`` → ``dexengine.core.arms.HAND_LIMIT_SUFFIX``, e.g. Tesollo's truncated finger limits).
This reads the **live articulations** (``robot`` and the split ``robot_left``; env 0 — limits are per-asset) and
compares every URDF joint they carry with
``<limit lower/upper>`` of the URDF the variant was baked from (``dexengine.core.arms.urdf_variant``). A mismatch kills
the run: a run whose variant name and physics limits disagree is void. The marker also counts how many joints
differ from the vendor URDF (the offline axis, ``canon_variant``) and their range — the truncated plate shows its 20
finger joints at [0°, 90°] (the two mirrored left-thumb joints at [−90°, 0°]), a vendor plate 0.

An unregistered variant (an experiment named explicitly) is reported unchecked — robots.selection already flags it.
The floating hand (``DEXENGINE_ARM=floating`` — no body-generation axis of its own) is checked against the URDF its
per-side plates are cut from (``floating_hand.source_urdf`` = the data-world combo's sim body, its limit suffix
included), the vendor axis being the data-world arm's canon variant.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

from dexengine.core.arms import (
    ARM_REGISTRY,
    ARM_SIM_COLLIDER_REV,
    FLOATING_ARM,
    active_arm_name,
    active_hand_name,
    canon_variant,
    generation_of,
    plate_variant,
    sim_hand_limits,
    urdf_variant,
)
from dexengine.rl.tasks import hooks
from dexengine.robots import selection as _rs

#: radians — the USD stores revolute limits in degrees as float32 (round trip ≤ 1e-6 rad at 180°)
TOL = 1e-4
#: articulations of the robot in the scene (whole robot · the split left arm)
ROBOT_KEYS = ("robot", "robot_left")


def urdf_limits(path) -> dict[str, tuple[float, float]]:
    """{joint: (lower, upper)} of every revolute/prismatic URDF joint with a limit."""
    out = {}
    for j in ET.parse(str(path)).getroot().findall("joint"):
        lim = j.find("limit")
        if j.get("type") in ("revolute", "prismatic") and lim is not None and lim.get("lower") is not None:
            out[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    return out


def _live(scene) -> dict[str, tuple[float, float]]:
    got = {}
    arts = getattr(scene, "articulations", {}) or {}
    for key in ROBOT_KEYS:
        art = arts.get(key)
        if art is None:
            continue
        lim = art.data.joint_pos_limits[0].detach().cpu().tolist()
        for name, (lo, hi) in zip(art.joint_names, lim, strict=False):
            got.setdefault(name, (float(lo), float(hi)))
    return got


def audit_and_print(scene) -> dict:
    """Print ``[USD_VARIANT]`` for env 0 and die on a mismatch. Returns the summary."""
    arm, hand = active_arm_name(), active_hand_name()
    if arm == FLOATING_ARM:
        return _audit_floating(scene, hand)
    variant = _rs.resolve_usd_variant(arm, hand, _rs.USD_CACHE_DIR)
    lim_sfx = sim_hand_limits(
        arm, hand
    )  # this hand's default (HandSpec); the spawn version may differ by an explicit override
    if generation_of(arm, variant) is None:
        print(
            f"[USD_VARIANT] hand={hand} variant={variant or '(no variant)'} limits=? · unregistered variant — joint "
            f"limit check skipped"
            f" (robots.selection reports it)",
            flush=True,
        )
        return {"variant": variant, "limits": None, "checked": 0}
    spawned = urdf_variant(
        arm, variant
    )  # suffix of the URDF the spawn version was baked from: the limits it really carries
    label = spawned[len(canon_variant(arm)) :] if spawned.startswith(canon_variant(arm)) else spawned
    head = f"[USD_VARIANT] hand={hand} variant={variant or '(no variant)'} limits={label or 'vendor'}" + (
        f" (hand default {lim_sfx or 'vendor'} — explicit override)" if label != lim_sfx else ""
    )
    urdf = _rs.combo_urdf_path(arm, hand, spawned)
    vendor = _rs.combo_urdf_path(arm, hand, canon_variant(arm))
    common, moved = _compare(scene, head, urdf, vendor, label)
    return {
        "variant": variant,
        "limits": label,
        "hand_default": lim_sfx,
        "checked": len(common),
        "moved": len(moved),
        "urdf": urdf.name,
    }


def _audit_floating(scene, hand: str) -> dict:
    """Floating hand: the source of the per-side version = floating_hand.source_urdf (the data-world combo's canonical
    sim body, limit suffix included)."""
    _fh = hooks.floating_hand()
    dw = _fh.DATA_WORLD_ARM
    label = sim_hand_limits(dw, hand)
    head = f"[USD_VARIANT] hand={hand} variant=(floating hand · source {dw}) limits={label or 'vendor'}"
    urdf = _fh.source_urdf(hand)
    vendor = _rs.combo_urdf_path(dw, hand, canon_variant(dw))
    common, moved = _compare(scene, head, urdf, vendor, label)
    return {
        "variant": f"floating:{dw}",
        "limits": label,
        "hand_default": label,
        "checked": len(common),
        "moved": len(moved),
        "urdf": urdf.name,
    }


def _compare(scene, head: str, urdf, vendor, label: str) -> tuple[list[str], list[str]]:
    """Print the spawned joint limits <-> urdf and the count of vendor-axis moves, and fail on a mismatch. (measured
    joints, joints that differ from the vendor)."""
    want, vend, got = urdf_limits(urdf), urdf_limits(vendor), _live(scene)
    common = sorted(j for j in want if j in got)
    if not common:
        raise RuntimeError(
            f"{head} · 0 joints measured (articulation {ROBOT_KEYS} · URDF {urdf.name}) — no run without the check"
        )
    bad = [
        f"{n}: sim [{got[n][0]:.5f}, {got[n][1]:.5f}] ≠ URDF [{want[n][0]:.5f}, {want[n][1]:.5f}] rad"
        for n in common
        if max(abs(got[n][0] - want[n][0]), abs(got[n][1] - want[n][1])) > TOL
    ]
    moved = [n for n in common if n in vend and max(abs(want[n][0] - vend[n][0]), abs(want[n][1] - vend[n][1])) > TOL]
    spans: dict[
        str, int
    ] = {}  # sim range (degrees) of the changed joints -> joint count (joints with the opposite bend sign are [−90, 0])
    for n in moved:
        k = f"[{math.degrees(got[n][0]):.1f}, {math.degrees(got[n][1]):.1f}]°"
        spans[k] = spans.get(k, 0) + 1
    rng = (" · ranges " + " ".join(f"{k}×{c}" for k, c in sorted(spans.items()))) if moved else ""
    print(
        f"{head} urdf={urdf.name} · joints {len(common) - len(bad)}/{len(common)} = URDF"
        f" (URDF only {sum(j not in got for j in want)}) · joints differing from the vendor limits {len(moved)}{rng}"
        + (" · check OK" if not bad else f" · mismatches {len(bad)}"),
        flush=True,
    )
    if label and not moved:
        bad.append(
            f"limit variant version {label} but its URDF differs from the vendor URDF ({vendor.name}) in 0 joints — "
            "not a clipped asset"
        )
    if not label and moved:
        bad.append(f"vendor limit version but {len(moved)} joints differ from the vendor URDF ({moved[:4]})")
    if bad:
        raise RuntimeError(
            f"{head} · the spawned joint limits differ from the variant's URDF — this run is invalid:\n  "
            + "\n  ".join(bad)
        )
    return common, moved


# ---------------------------------------------------------------- arm collider revision [ARM_WRIST_COLLIDER]
# Is the arm-collider revision really on the spawned stage? — marker ``[ARM_WRIST_COLLIDER]``.
#
# The expectation comes from the spawned variant (``robots.selection.resolve_usd_variant`` → ``dexengine.core.arms``
# generation table): a variant carrying the arm's collider revision (``ARM_SIM_COLLIDER_REV``) must show, on every
# revised link, the new convex hull and no vendor capsule; a variant without it (the body generation named explicitly,
# e.g. an A/B) must show the vendor capsule on the ``replace`` links and no hull. The static root body ``world`` is
# audited the same way for the torso (``torso_replace`` capsules ↔ the hull parts of every torso link — part prims
# ``rev.torso_prim(link, i)`` counted from 0 without a gap; how many is the decomposition rule's, read off the stage).
# A mismatch kills the run — a run whose asset name and physics disagree is void; a variant that is no registered
# generation (a retired revision) is told so, with the current variant and the command that bakes it. The stage is
# read, not the file: env_0 only (the colliders are instance prototypes, identical in every env), both articulations
# (``Robot`` and the split ``RobotL``; a link is looked up where it is active).


def _link_state(stage, env_root: str, link: str, prim_name: str) -> dict | None:
    """Colliders of one link on the live stage (the articulation where the link is active)."""
    from pxr import Usd, UsdGeom, UsdPhysics

    for leaf in ("Robot", "RobotL"):
        prim = stage.GetPrimAtPath(f"{env_root}/{leaf}/{link}")
        if not (prim.IsValid() and prim.IsActive()):
            continue
        col = stage.GetPrimAtPath(f"{env_root}/{leaf}/{link}/collisions")
        shapes = (
            [d for d in Usd.PrimRange(col, Usd.TraverseInstanceProxies()) if d.HasAPI(UsdPhysics.CollisionAPI)]
            if col.IsValid()
            else []
        )
        node = stage.GetPrimAtPath(f"{env_root}/{leaf}/{link}/collisions/{prim_name}/node_hull")
        mesh = stage.GetPrimAtPath(f"{env_root}/{leaf}/{link}/collisions/{prim_name}/node_hull/mesh")
        approx = (
            UsdPhysics.MeshCollisionAPI(node).GetApproximationAttr().Get()
            if node.IsValid() and node.HasAPI(UsdPhysics.MeshCollisionAPI)
            else None
        )
        hull = node.IsValid() and node.HasAPI(UsdPhysics.CollisionAPI) and approx == "convexHull" and mesh.IsValid()
        return {
            "leaf": leaf,
            "colliders": len(shapes),
            "capsules": sum(1 for d in shapes if d.IsA(UsdGeom.Capsule)),
            "hull_points": len(mesh.GetAttribute("points").Get() or []) if hull else 0,
        }
    return None


def _world_state(stage, env_root: str, rev) -> dict | None:
    """Colliders of the static root body ``world``: capsule count and, per torso link, the point counts of its hull
    parts ``rev.torso_prim(link, 0..)`` up to the first absent one ([] = no hull)."""
    from pxr import Usd, UsdGeom, UsdPhysics

    for leaf in ("Robot", "RobotL"):
        col = stage.GetPrimAtPath(f"{env_root}/{leaf}/world/collisions")
        if not col.IsValid():
            continue
        shapes = [d for d in Usd.PrimRange(col, Usd.TraverseInstanceProxies()) if d.HasAPI(UsdPhysics.CollisionAPI)]
        hulls = {}
        for ln in tuple(rev.torso_replace) + tuple(rev.torso_add):
            hulls[ln] = []
            while True:
                mesh = stage.GetPrimAtPath(f"{col.GetPath()}/{rev.torso_prim(ln, len(hulls[ln]))}/node_hull/mesh")
                if not mesh.IsValid():
                    break
                hulls[ln].append(len(mesh.GetAttribute("points").Get() or []))
        return {
            "leaf": leaf,
            "colliders": len(shapes),
            "hulls": hulls,
            "capsules": sum(1 for d in shapes if d.IsA(UsdGeom.Capsule)),
        }
    return None


def _audit_world(stage, env_root: str, rev, expect: bool) -> tuple[str, list[str]]:
    """Torso on ``world``: the revision variant carries hull parts on every torso link (each within the cap), nothing
    else and no capsule; the body generation carries the ``torso_replace`` capsules and no hull."""
    st = _world_state(stage, env_root, rev)
    if st is None:
        return "no world", ["world: the static root body has no collisions (Robot·RobotL)"]
    n_h = sum(len(v) for v in st["hulls"].values())
    row = (
        f"world@{st['leaf']} torso hulls {n_h} (links {sum(1 for v in st['hulls'].values() if v)}/{len(st['hulls'])}) "
        f"capsules {st['capsules']} colliders {st['colliders']}"
    )
    if expect and not (
        all(v and all(0 < n <= rev.hull_vertex_cap for n in v) for v in st["hulls"].values())
        and st["capsules"] == 0
        and st["colliders"] == n_h
    ):
        return row, [f"world: expected the torso hulls of revision {rev.suffix} but {row}"]
    if not expect and (n_h or st["capsules"] != len(rev.torso_replace)):
        return row, [f"world: expected {len(rev.torso_replace)} old torso capsules but {row}"]
    return row, []


def audit_wrist_collider(stage, env_root: str, arm: str, variant: str) -> tuple[list[str], list[str]]:
    """(per-link summaries, violations) for every arm link of ``arm`` on the live stage: a revised link carries the
    revision hull (and, if ``replace``, no capsule) under the revision variant and the vendor collider otherwise; a link
    the revision leaves alone never carries a revision hull — so a derivative baked under the same name from another
    link list cannot pass."""
    rev = ARM_SIM_COLLIDER_REV.get(arm)
    if rev is None:
        return [f"arm {arm} has no collider revision"], []
    spec = ARM_REGISTRY[arm]
    expect = variant.endswith(rev.suffix)
    rows, bad = [], []
    for side in spec.sides:
        for k in range(spec.index_base, spec.ee_body_index + 1):
            mode = "replace" if k in rev.replace else "add" if k in rev.add else None
            ln = spec.body_name(side, k)
            st = _link_state(stage, env_root, ln, rev.prim_name)
            if st is None:
                bad.append(f"{ln}: no active link (Robot·RobotL)")
                continue
            rows.append(
                f"{ln}@{st['leaf']} hull {st['hull_points']}v capsules {st['capsules']} "
                f"colliders {st['colliders']}{'' if mode else ' (outside the revision)'}"
            )
            if mode is None and st["hull_points"]:
                bad.append(
                    f"{ln}: hull {st['hull_points']}v on a link outside the revision — a version baked from another "
                    f"link list"
                )
            if mode and expect and not (0 < st["hull_points"] <= rev.hull_vertex_cap and st["capsules"] == 0):
                bad.append(
                    f"{ln}: expected revision {rev.suffix} but hull {st['hull_points']}v · capsules {st['capsules']}"
                )
            if mode and not expect and (st["hull_points"] or (mode == "replace" and st["capsules"] != 1)):
                bad.append(
                    f"{ln}: expected the old colliders but hull {st['hull_points']}v · capsules {st['capsules']}"
                )
    if rev.torso_replace or rev.torso_add:
        row, b2 = _audit_world(stage, env_root, rev, expect)
        rows.append(row)
        bad += b2
    return rows, bad


def audit_wrist_collider_and_print(scene) -> dict:
    """Print the marker for env_0 and die on a mismatch. Returns the summary (for the run banner)."""
    arm, hand = active_arm_name(), active_hand_name()
    variant = _rs.resolve_usd_variant(arm, hand, _rs.USD_CACHE_DIR)
    roots = list(getattr(scene, "env_prim_paths", []) or [])
    if not roots:
        raise RuntimeError("[ARM_WRIST_COLLIDER] no env prim — no stage to audit")
    rows, bad = audit_wrist_collider(scene.stage, roots[0], arm, variant)
    rev = ARM_SIM_COLLIDER_REV.get(arm)
    want = "revision" if rev is not None and variant.endswith(rev.suffix) else "old colliders"
    print(
        f"[ARM_WRIST_COLLIDER] variant={variant or '(no variant)'} expected={want}"
        f"{f' rev={rev.suffix}' if rev is not None else ''} · "
        + " · ".join(rows)
        + (" · check OK" if not bad else f" · mismatches {len(bad)}"),
        flush=True,
    )
    if bad and generation_of(arm, variant) is None:
        bad.append(
            f"variant {variant!r} is not a registered generation (a retired revision?) — "
            f"current canonical plate {plate_variant(arm, hand)!r} · "
            f"bake: python dexengine/robots/usd/collider_rev.py --arm {arm} --hand {hand}"
        )
    if bad:
        raise RuntimeError(
            "[ARM_WRIST_COLLIDER] the spawned variant and the stage's arm colliders differ — this run is invalid:\n  "
            + "\n  ".join(bad)
        )
    return {"variant": variant, "expect": want, "links": rows}
