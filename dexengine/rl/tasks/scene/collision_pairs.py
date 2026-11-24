"""Pair exclusions authored once per env: robot <-> desk, disc <-> other object, arm <-> object (robot <-> support
discs and other robot-off bodies are collision groups: `collision_groups`)."""

from __future__ import annotations

from pxr import Usd, UsdPhysics

from dexengine.rl.tasks.scene.collision_bodies import (
    _active_arm_name,
    _active_hand_name,
    _object_cfg_fields,
    _rigid_bodies,
    author_filtered_pairs,
    classify_body,
    object_prim_paths,
    object_rigid_bodies,
    scene_object_key,
)


def author_robot_desk_nocollision(stage, env_root: str) -> None:
    """Non-scrape diverse-5 prescription: disable ROBOT(arm+hand)
    <-> DESK collision so reference wrist poses at/below the tabletop (e.g.
    brush_roller_pan wrist ref z_min = -2.6 cm below desk) stay trackable instead
    of the arm jamming on the desk and the IK compensating with curled fingers.

    Excludes every robot rigid-body link <-> ``{env_root}/DeskCollisionTop`` pair
    via ``FilteredPairsAPI`` (authored once on env_0 BEFORE cloning). KEEPS
    object<->desk (objects must rest on the tabletop) and everything else.
    Env-gated by ``DEXENGINE_DESK_ROBOT_NOCOLL=1``; unset = byte-identical.
    """
    desk = stage.GetPrimAtPath(f"{env_root}/DeskCollisionTop")
    if not desk.IsValid():
        print(f"[DeskRobotNoColl] WARNING: DeskCollisionTop not found under {env_root} — skipping")
        return
    robot_root = stage.GetPrimAtPath(f"{env_root}/Robot")
    body_paths = [p.GetPath().pathString for p in Usd.PrimRange(robot_root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not body_paths:
        raise RuntimeError(f"[DeskRobotNoColl] no robot rigid bodies under {env_root}/Robot")
    desk_path = desk.GetPath().pathString
    n = 0
    for bp in body_paths:
        prim = stage.GetPrimAtPath(bp)
        api = UsdPhysics.FilteredPairsAPI.Apply(prim)
        rel = api.GetFilteredPairsRel() or api.CreateFilteredPairsRel()
        if desk_path not in set(rel.GetTargets()):
            rel.AddTarget(desk_path)
            n += 1
    print(
        f"[DeskRobotNoColl] env_root={env_root}: EXCLUDED robot<->desk for {n} robot links "
        f"(desk={desk_path}). KEPT object<->desk."
    )


def disc_own_object_path(obj_paths: dict[str, str], owner: str, shared_object: bool, idle_side: str = "") -> str:
    """Disc owner side (`coords.disc_owner_side`: right · left · both) -> the path of that env's own object prim. The
    side -> scene key rule is `scene_object_key` alone (a shared object makes the left one object_right too; an idle
    side has no object). 'both' (shared object) without the shared statement, or a side whose object is missing from
    the scene, fails."""
    if owner == "both":
        if not shared_object:
            raise RuntimeError(
                "[DISC_COLL] disc owner = shared object (both) but the scene has no shared-object statement "
                "(cfg.shared_object)"
            )
        owner = "right"
    key = scene_object_key({f"object_{s}" for s in obj_paths}, owner, shared_object, idle_side)
    side = key[len("object_") :] if key else None
    if side not in obj_paths:
        raise RuntimeError(
            f"[DISC_COLL] the object of disc owner side {owner} is not in the scene (key {key!r} · "
            f"scene objects {sorted(obj_paths)}) "
            "— an idle side cannot own a disc"
        )
    return obj_paths[side]


def author_disc_other_object_nocollision(
    stage,
    env_root: str,
    slot_paths: dict[int, str],
    owner_row: list,
    obj_paths: dict[str, str],
    shared_object: bool,
    idle_side: str = "",
) -> dict:
    """Rule D11″ (d), a support disc supports its own object only: every pair of a real disc body of
    this env clip (slot k, owner side `owner_row[k]`) and **another** object body in the scene is excluded with
    `FilteredPairsAPI`. Own object<->disc stays on; empty slots (owner None, hidden z) are not authored.

    Source = the disc (a non-instanced kinematic RigidObject; rels on the RigidObject side reach PhysX), target = the
    other object bodies (every link of an articulation). Rels sourced on dynamic objects blow up PhysX GPU memory, so
    objects are never the source; the GPU cost of kinematic disc sources is measured by a physics test. After authoring
    the rels are re-read to check that every (disc, other object) pair is filtered and (disc, own object) pairs are 0;
    a mismatch fails.
    Returns: {discs: real disc count, own: {k: own object path}, pairs: targets added this time, want: expected
    pairs}."""
    if len(owner_row) < len(slot_paths):
        raise RuntimeError(
            f"[DISC_COLL] {env_root}: disc owner statement has {len(owner_row)} cells < disc slots {len(slot_paths)}"
        )
    bodies = {p: _rigid_bodies(stage, p, "object") for p in obj_paths.values()}
    own_of: dict[int, str] = {}
    want: dict[str, set[str]] = {}
    forbid: dict[str, set[str]] = {}
    n = 0
    for k, dpath in sorted(slot_paths.items()):
        if owner_row[k] in (
            None,
            "static",
        ):  # empty slot · column under a fixed object (no main-object owner; prim collision is off)
            continue
        own = disc_own_object_path(obj_paths, owner_row[k], shared_object, idle_side)
        own_of[k] = own
        others = [b for p, bs in bodies.items() if p != own for b in bs]
        for db in _rigid_bodies(stage, dpath, "disc"):
            if others:
                n += author_filtered_pairs(stage, [db], others)[1]
            want[db] = set(others)
            forbid[db] = set(bodies[own])
    bad = []
    for db, w in want.items():
        rel = UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath(db)).GetFilteredPairsRel()
        got = {t.pathString for t in rel.GetTargets()} if rel else set()
        if w - got or got & forbid[db]:
            bad.append((db, len(w - got), len(got & forbid[db])))
    if bad:
        raise RuntimeError(
            f"[DISC_COLL] {env_root}: read-back after authoring — (disc, other object) missing · (disc, own object) "
            f"filtered "
            f"{bad[:3]}"
        )
    return {"discs": len(own_of), "own": own_of, "pairs": n, "want": sum(len(w) for w in want.values())}


# Arm<->object physics ablation, DEXENGINE_ARM_OBJ_NOCOLL=1: measures the cost of the penetration
# constraint (fingers <-> desk when grasping flat objects, above all) in a physics where "the hand collides only with
# objects, the arm (links before the wrist) with nothing, and the robot not with the desk"; shared by training and eval.
# Targets: the **arm links** of both arms (before the wrist; RA/LA, the `classify_body` class of "every arm link but
# the
# EE", no hand or finger links) × **every object** in the scene (object_right, object_left; own and other alike).
# The same function and authoring path as author_collcurr_cross_pair_exclusions (`author_filtered_pairs`): the
# "articulation link <-> RigidObject" category whose physical effect is verified (the instanced-collider problem of
# link<->link self-collision does not apply, object prims are not instanced colliders), at the same time (POSTINIT,
# end of load_managers), always run independently of the training-only collision_curriculum field, so eval is not
# trapped behind a training-only gate.
# Not turned off (kept): hand<->object (both), object<->object, object<->desk, arm<->torso, and every other filter.


def author_arm_obj_nocollision(scene, env_root: str) -> dict[str, list[str]]:
    """Exclude every arm link (pre-wrist, hand/finger EXCLUDED) <-> every scene
    object (``object_right`` AND ``object_left``, both sides, cross AND own) pair.

    Args:
        scene: the ``InteractiveScene`` being built (object prim paths are read
            off ``scene.cfg``; see ``object_prim_paths``).
        env_root: the env_0 prim path to author on (POSTINIT call site authors
            once on the template — see module comment above for why that suffices
            for this pair category).

    Returns:
        ``{source_prim_path: [added_target_paths]}`` (object body -> arm-link
        targets) — the exact rel edits, in case a future caller wants to undo them
        (no flip exists today; this ablation is authored permanently for the life
        of the env, like ``author_robot_desk_nocollision``).
    """
    stage = scene.stage
    if not stage.GetPrimAtPath(f"{env_root}/Robot").IsValid():
        raise RuntimeError(f"[ArmObjNoColl] robot prim not found under {env_root}")

    # Both articulations: with the split robot the left arm lives under /RobotL (its links are inactive under
    # /Robot, so PrimRange skips them there). Walking /Robot alone silently left the left arm colliding.
    arm_bodies = []
    count = {"RA": 0, "LA": 0}
    for leaf in ("Robot", "RobotL"):
        root = stage.GetPrimAtPath(f"{env_root}/{leaf}")
        if not root.IsValid():
            continue
        for p in Usd.PrimRange(root):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                g = classify_body(p.GetPath().pathString.rsplit("/", 1)[-1])
                if g in ("RA", "LA"):
                    arm_bodies.append(p.GetPath().pathString)
                    count[g] += 1
    if not count["RA"] or not count["LA"]:
        raise RuntimeError(
            f"[ArmObjNoColl] arm links per side under {env_root}/Robot(+RobotL): right {count['RA']} · "
            f"left {count['LA']} — DEXENGINE_ARM_OBJ_NOCOLL needs the pre-wrist arm links (EE/hand excluded) of "
            "BOTH sides; one side missing would leave that arm colliding silently. Active selection: "
            f"arm={_active_arm_name()!r} hand={_active_hand_name()!r}."
        )

    # Object prim paths come from the cfg that spawns them, never a literal table
    # here — same helper author_collcurr_cross_pair_exclusions uses.
    obj_paths = object_prim_paths(scene, env_root)

    objr_bodies = object_rigid_bodies(stage, obj_paths, "right")
    objl_bodies = object_rigid_bodies(stage, obj_paths, "left")
    all_obj_bodies = objr_bodies + objl_bodies
    if not all_obj_bodies:
        raise RuntimeError(
            f"[ArmObjNoColl] no object rigid bodies found under {env_root} (obj "
            f"prims from scene cfg: {obj_paths or '<none declared>'}; cfg fields "
            f"carrying a prim_path: {_object_cfg_fields(scene)})."
        )

    # Source on the OBJECT bodies (mirrors author_collcurr_cross_pair_exclusions'
    # hand<->object convention, e.g. _exclude(objl_bodies, rh_bodies)) — direction
    # is immaterial to PhysX (FilteredPairsAPI is symmetric) but keeps this
    # function's authoring call textually parallel to its sibling.
    # Source = arm links, target = objects (object-sourced rels blow up PhysX GPU memory: +12 GB for 12 pairs/env)
    pairs, n = author_filtered_pairs(stage, arm_bodies, all_obj_bodies)
    print(
        f"[ArmObjNoColl] env_root={env_root}: EXCLUDED arm<->object for "
        f"{len(arm_bodies)} arm links (right {count['RA']} · left {count['LA']}, pre-wrist, hand/finger excluded) x "
        f"{len(all_obj_bodies)} object bodies (objR={len(objr_bodies)}, "
        f"objL={len(objl_bodies)}); pairs added={n}. KEPT ON: hand<->object (both "
        "sides), object<->object, object<->desk, arm<->torso (B22 if enabled)."
    )
    return pairs
