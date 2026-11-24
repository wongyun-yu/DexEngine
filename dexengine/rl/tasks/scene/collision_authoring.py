"""The **single point** of collision filter authoring: before PhysX parses, right after the spawners made every env
prim.

Why. The old authoring point (`ArmTrainEnv.__init__` POSTINIT) came after `sim.reset()`, i.e. after PhysX had parsed
the stage. Adding `FilteredPairsAPI` to **object prims** at that point (cross pairs RHxobjL/LHxobjR/objRxobjL; for
arm<->object the object is the source) made PhysX rebuild the body, and objects then flew off at 1–3 m/s 40–50 cm away
from the hand right after a reset (probe: helmet clip `abl` 47 cases vs `single` 0, the "it bounces before the helmet
is even touched" mystery). Filters on link prims only (desk, split articulation R2/R3) never showed it.

Here a hook at the end of `InteractiveScene._add_entities_from_cfg()` (the only window where every env's prims exist,
also with `replicate_physics=False`, and PhysX has not parsed yet) authors **every filter in every env**. Training,
the frozen judge and render pass through the same function. POSTINIT keeps only the audit (collision_check) and the
preset check.
"""

from __future__ import annotations

import time

from isaaclab.scene import InteractiveScene

from dexengine.rl.tasks import env_flags
from dexengine.rl.tasks.scene.all_obj_collision import author_all_obj_collision
from dexengine.rl.tasks.scene.collision_extra_slots import author_extra_collision

#: Returned by a split-out step that did not return early (its caller then goes on).


class PostSpawnHookScene(InteractiveScene):
    """Callback right after the spawners, before parsing. Callbacks come through the class attribute `hooks` (a symbol
    swap, so no ctor kwargs)."""

    hooks: list = []  # [(callable(scene), label)]
    AUTHORS_ROBOT_OFF_GROUPS = True  # filter_collisions below authors the planned robot-off groups (collision_groups)

    def _add_entities_from_cfg(self):
        super()._add_entities_from_cfg()
        for fn, label in list(type(self).hooks):
            t0 = time.time()
            fn(self)
            print(f"[PRE_PARSE] {label} done ({time.time() - t0:.1f}s, env {len(self.env_prim_paths)})", flush=True)

    def filter_collisions(self, global_prim_paths=None):
        """Isaac Lab's env groups, then the robot-off groups the hooks planned, before PhysX parses the stage."""
        from dexengine.rl.tasks.scene.collision_groups import author_pending

        super().filter_collisions(global_prim_paths)
        t0 = time.time()
        if author_pending(self) is not None:
            print(f"[PRE_PARSE] robot-off collision groups done ({time.time() - t0:.1f}s)", flush=True)


def for_every_env(scene, fn, label: str):
    """Apply fn(env_root) to every authoring env (authoring_roots; with replicated physics only the env_0 template) and
    return env_0's result (one log line). No scope options."""
    from dexengine.rl.tasks.scene.collision_curriculum import authoring_roots

    roots = authoring_roots(scene)
    t0 = time.time()
    first = None
    for i, r in enumerate(roots):
        out = fn(r)
        if i == 0:
            first = out
    print(f"[COLL_AUTHOR] {label}: authored on {lever(scene)} ({time.time() - t0:.1f}s)", flush=True)
    return first


def lever(scene) -> str:
    """Authoring env marker: 'all N envs' or 'replicated-physics env_0 template (-> N envs)'."""
    from dexengine.rl.tasks.scene.collision_curriculum import authoring_roots

    n = len(getattr(scene, "env_prim_paths", []) or [])
    return f"all {n} envs" if len(authoring_roots(scene)) == n else f"replicated-physics env_0 template (-> {n} envs)"


def author_collision_filters(scene, cfg) -> dict:
    """Author every collision filter. Returns a banner summary (cross_off, arm_obj_pairs, desk, split_off, collcurr)."""
    from dexengine.rl.tasks.scene import split_artic
    from dexengine.rl.tasks.scene.collision_curriculum import (
        author_collcurr_cross_pair_exclusions,
        collcurr_author_envs,
        collcurr_verify_envs,
    )
    from dexengine.rl.tasks.scene.collision_filter import split_author_off_rows, split_off_rows
    from dexengine.rl.tasks.scene.collision_pairs import author_arm_obj_nocollision, author_robot_desk_nocollision

    split = split_artic.bi_enabled()
    info = {
        "cross_off": False,
        "arm_obj_pairs": 0,
        "arm_obj": False,
        "desk": False,
        "split_off": set(),
        "collcurr": False,
    }

    rows = _shared_object_roots(
        scene,
        cfg,
        author_arm_obj_nocollision,
        author_collcurr_cross_pair_exclusions,
        author_robot_desk_nocollision,
        collcurr_author_envs,
        collcurr_verify_envs,
        info,
        split,
        split_off_rows,
    )
    if rows:
        if not split:
            raise RuntimeError(
                "DEXENGINE_BI_SPLIT_OFF only means something with the split articulation (DEXENGINE_BI_SPLIT_ARTIC=1) "
                "— "
                "with a single articulation R2 and R3 are already OFF through the self-collision flag."
            )
        split_author_off_rows(scene, rows)
        info["split_off"] = rows

    return info


def _shared_object_roots(
    scene,
    cfg,
    author_arm_obj_nocollision,
    author_collcurr_cross_pair_exclusions,
    author_robot_desk_nocollision,
    collcurr_author_envs,
    collcurr_verify_envs,
    info,
    split,
    split_off_rows,
):
    """Shared object: env_setup.build._apply_shared_object writes shared_object=True on the env cfg and removes
    scene.object_left. Cross-pair authoring (1)(2) reads that statement and skips ho/oo (the left hand's object =
    objR).
    """
    shared = bool(getattr(cfg, "shared_object", False))
    if shared:
        print("[collision_authoring] SHARED OBJECT: no object_left — cross-pair authoring covers hh only", flush=True)

    # 1) training collcurr start-OFF (--collision_curriculum 1). All envs.
    if getattr(cfg, "collision_curriculum", False):
        collcurr_author_envs(
            scene,
            include_objpair=bool(getattr(cfg, "collision_curriculum_objpair", False)),
            objpair_only=split,
            shared_object=shared,
        )
        print(collcurr_verify_envs(scene, n_check=4), flush=True)
        info["collcurr"] = True

    # 2) eval symmetry: DEXENGINE_EVAL_CROSS_OFF=1 (the same set and function as training collcurr, permanently OFF)
    cross_off = env_flags.get("DEXENGINE_EVAL_CROSS_OFF") == "1"
    if cross_off and not getattr(cfg, "collision_curriculum", False):
        if not getattr(cfg, "bimanual_self_collision_filter", False):
            raise RuntimeError(
                "DEXENGINE_EVAL_CROSS_OFF=1 requires a bimanual build "
                "(cfg.bimanual_self_collision_filter) — RH/LH/objR/objL bodies needed."
            )
        if split:
            raise RuntimeError(
                "DEXENGINE_EVAL_CROSS_OFF=1 cannot be combined with the split articulation "
                "(DEXENGINE_BI_SPLIT_ARTIC=1)."
            )
        scene._collcurr_pairs = for_every_env(
            scene,
            lambda r: author_collcurr_cross_pair_exclusions(
                scene, r, include_objpair=True, objpair_only=False, shared_object=shared
            ),
            "EVAL_CROSS_OFF cross-pairs",
        )
        print(
            "[EVAL_CROSS_OFF] cross-pairs (RHxLH, RHxobjL, LHxobjR, objRxobjL) authored PERMANENTLY OFF "
            "(no flip) — same set/function as train collision_curriculum start-OFF state.",
            flush=True,
        )
        info["cross_off"] = True

    # 3) arm<->object
    if env_flags.get("DEXENGINE_ARM_OBJ_NOCOLL") == "1":
        res = for_every_env(scene, lambda r: author_arm_obj_nocollision(scene, r), "ARM_OBJ_NOCOLL")
        info["arm_obj"] = True
        info["arm_obj_pairs"] = sum(len(v) for v in (res or {}).values())

    _robot_desk_authoring_point(scene, cfg, author_robot_desk_nocollision, info)

    # 5) split articulation inter-articulation rows permanently OFF
    rows = split_off_rows()
    return rows


def _robot_desk_authoring_point(scene, cfg, author_robot_desk_nocollision, info):
    """4) robot<->desk (the only authoring point)."""
    if env_flags.get("DEXENGINE_DESK_ROBOT_NOCOLL") == "1":
        for_every_env(scene, lambda r: author_robot_desk_nocollision(scene.stage, r), "DESK_ROBOT_NOCOLL")
        info["desk"] = True

    # 4b) robot<->support disc (rule D40: disc-hand and disc-arm collisions off): canonical physics, no
    # flag
    info["disc"] = author_disc_collision(scene, cfg)
    # 4c) support objects (rule D42: a support object collides with hands, arms and objects): canonical physics,
    # no flag
    info["support"] = author_support_collision(scene, cfg)
    # 4d) extra object slots (DEXENGINE_EXTRA_SLOTS; 0 = off -> with 0 slot prims nothing is authored, marker only)
    info["extra"] = author_extra_collision(scene, cfg)
    # 4e) all-object collision (collision_presets.ALL_OBJ_ENV; default off = does nothing)
    if _ao := author_all_obj_collision(scene, cfg):
        info["all_obj"] = _ao


def author_support_collision(scene, cfg) -> dict:
    """Author support object slot collisions as stated in `collision_presets.SUPPORT_COLLIDE`. ON partners (hand, arm,
    object) get nothing (PhysX default = collide); the only OFF partner this can build is the desk: the (support body,
    `DeskCollisionTop`) pairs are turned off with FilteredPairs. A slot prim count different from the statement
    `cfg.support_object_slots` (writer `env_setup.supports._build_support_slots`), a missing desk prim, or a table that
    asks to turn off a partner other than the desk fails. After authoring the rels are re-read to count that every
    env × slot body pair is filtered. Marker `[SUPPORT_COLL]`, printed also without supports. Returns a banner summary
    {slots, envs, pairs}."""
    from pxr import Usd, UsdPhysics

    from dexengine.rl.tasks.scene.collision_bodies import author_filtered_pairs, support_object_prim_paths
    from dexengine.rl.tasks.scene.collision_curriculum import authoring_roots
    from dexengine.rl.tasks.scene.collision_presets import SUPPORT_COLLIDE

    roots = authoring_roots(scene)
    said = getattr(cfg, "support_object_slots", None)
    slots0 = support_object_prim_paths(scene, roots[0]) if roots else []
    if len(slots0) != (said or 0):
        raise RuntimeError(
            f"[SUPPORT_COLL] support slot prims {len(slots0)} ≠ statement cfg.support_object_slots={said!r}"
        )
    off = sorted(k for k, v in SUPPORT_COLLIDE.items() if v == "OFF")
    if off != ["desk"] or any(v not in ("ON", "OFF") for v in SUPPORT_COLLIDE.values()):
        raise RuntimeError(
            f"[SUPPORT_COLL] table SUPPORT_COLLIDE={SUPPORT_COLLIDE} — the only partner this authoring "
            "can turn off is the desk"
        )
    rule = " · ".join(f"{k} {v}" for k, v in SUPPORT_COLLIDE.items())
    if not slots0:
        print(f"[SUPPORT_COLL] 0 supports — nothing authored ({lever(scene)} · table {rule})", flush=True)
        return {"slots": 0, "envs": len(roots), "pairs": 0}
    stage, per_env = scene.stage, []
    for r in roots:
        desk = stage.GetPrimAtPath(f"{r}/DeskCollisionTop")
        if not desk.IsValid():
            raise RuntimeError(f"[SUPPORT_COLL] {r}: no DeskCollisionTop — support<->desk off cannot be authored")
        bodies = [
            q.GetPath().pathString
            for s in support_object_prim_paths(scene, r)
            for q in Usd.PrimRange(stage.GetPrimAtPath(s))
            if q.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        author_filtered_pairs(stage, bodies, [desk.GetPath().pathString])
        rels = [UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath(b)).GetFilteredPairsRel() for b in bodies]
        got = sum(desk.GetPath() in rel.GetTargets() for rel in rels)
        per_env.append((len(bodies), got))
    bad = [i for i, (n, g) in enumerate(per_env) if n != len(slots0) or g != n]
    if bad:
        raise RuntimeError(
            f"[SUPPORT_COLL] envs {bad[:5]} with support body·desk pairs different from the {len(slots0)} slots: "
            f"{[per_env[i] for i in bad[:3]]}"
        )
    pairs = sum(g for _, g in per_env)
    print(
        f"[SUPPORT_COLL] {rule} · {lever(scene)} · support slots {len(slots0)} · desk filter pairs {pairs} "
        f"(per env {len(slots0)}) · hand·arm·objects = no filter",
        flush=True,
    )
    return {"slots": len(slots0), "envs": len(roots), "pairs": pairs}


def author_disc_collision(scene, cfg) -> dict:
    """Author the robot side of support disc collisions by the stamp rule; training, the frozen judge and render pass
    through this one function.

    The rule's only value is the cfg statement `cfg.support_disc_collide` (writer
    `env_setup.supports._build_disc_slots`,
    which reads each stamped disc's `collide` and checks it against `coords.DISC_COLLIDE`); its meaning (the RD row) is
    owned by `collision_presets.DISC_COLLIDE_RD` alone (the oracle `collision_check.check_preset` reads the same table).
    Disc prims without a statement, a statement outside the table, or a meaning other than the physics this function
    can build (OFF) fail. Without discs nothing is authored (the stage stays byte-identical) and only the marker
    prints. Marker `[DISC_COLL] robot=off …`; a run without the marker is invalid.
    The second row of the same stamp, RO (rule D11″ (d): a disc supports its own object only), is authored here too
    (`_author_disc_objects`). Robot <-> disc off = collision groups (`collision_groups`: planned here, authored once
    Isaac Lab's env groups exist, read back). Returns a banner summary {collide, envs, slots, discs, robot_off, ro}."""
    from dexengine.rl.tasks.scene.collision_bodies import support_disc_prim_paths
    from dexengine.rl.tasks.scene.collision_curriculum import authoring_roots
    from dexengine.rl.tasks.scene.collision_groups import plan
    from dexengine.rl.tasks.scene.collision_presets import DISC_COLLIDE_RD

    roots = authoring_roots(scene)
    collide = getattr(cfg, "support_disc_collide", None)
    slots0 = support_disc_prim_paths(scene, roots[0]) if roots else []
    if not slots0:
        if collide is not None:
            raise RuntimeError(f"[DISC_COLL] disc collision statement collide={collide!r} but no disc prims")
        print(f"[DISC_COLL] robot=off · 0 discs — nothing authored ({lever(scene)})", flush=True)
        return {"collide": None, "envs": len(roots), "slots": 0, "discs": 0, "robot_off": None, "ro": None}
    if collide is None:
        raise RuntimeError(
            f"[DISC_COLL] {len(slots0)} disc slots but no collision statement (cfg.support_disc_collide) "
            "— discs that did not go through _build_disc_slots"
        )
    rd = DISC_COLLIDE_RD.get(collide)
    if rd != "OFF":  # the only physics this can build is robot<->disc off (discs and pedestals); anything else fails
        raise RuntimeError(
            f"[DISC_COLL] the RD rule {rd!r} of disc collide={collide!r} cannot be authored — table "
            f"collision_presets.DISC_COLLIDE_RD={DISC_COLLIDE_RD}"
        )
    rule = f"robot={rd.lower()}"
    per_env: list[int] = []

    def _plan(r):
        discs = support_disc_prim_paths(scene, r)
        plan(scene, r, discs)
        per_env.append(len(discs))

    for_every_env(scene, _plan, "DISC_COLL robot<->disc")
    if len(per_env) != len(roots) or any(n != per_env[0] for n in per_env):
        raise RuntimeError(
            f"[DISC_COLL] {len(per_env)}/{len(roots)} envs planned · disc counts differ per env {per_env[:5]}"
        )
    ro = _author_disc_objects(scene, cfg, roots, collide)
    kinds = getattr(
        cfg, "support_disc_kinds", None
    )  # pedestal scene: collision partners per kind (coords.PEDESTAL_COLLIDE)
    print(
        f"[DISC_COLL] {rule} · collide={collide}(stamp · owner coords.DISC_COLLIDE · PEDESTAL_COLLIDE)"
        f"{f' · pedestal per kind {kinds}' if kinds else ''} · {lever(scene)} · disc slots "
        f"{len(slots0)} · the whole robot (torso and base included) <-> disc = collision groups (collision_groups, "
        f"{sum(per_env)} disc roots over {len(per_env)} envs) — {ro['text']}",
        flush=True,
    )
    return {
        "collide": collide,
        "envs": len(roots),
        "slots": len(slots0),
        "discs": sum(per_env),
        "robot_off": "groups",
        "ro": ro,
    }


def _author_disc_objects(scene, cfg, roots: list, collide: str) -> dict:
    """RO row (rule D11″ (d)): a disc supports **its own main object** only. For every real disc of env
    i's clip (static assignment i % clip count) every pair (disc, other object body) is turned off. The meaning lives in
    `collision_presets.DISC_COLLIDE_RO` alone, the owner side statement in `cfg.support_disc_owner` (writer
    `env_setup.supports._build_disc_slots` <- `coords.disc_owner_side`). With fewer than two objects in the scene (one
    hand, shared, idle side) there is no other object and nothing is authored (the stage stays byte-identical).
    No statement, a meaning outside the table, or several clips without static assignment (with the env clip changing
    at every reset, the disc owner is not fixed) fail. Returns {rule, envs, pairs, want, discs_env0, text}."""
    from dexengine.rl.tasks.scene.collision_bodies import object_prim_paths, support_disc_slot_paths
    from dexengine.rl.tasks.scene.collision_pairs import author_disc_other_object_nocollision
    from dexengine.rl.tasks.scene.collision_presets import DISC_COLLIDE_RO

    objs0 = object_prim_paths(scene, roots[0])
    if len(set(objs0.values())) < 2:
        return _author_disc_support(cfg, roots, objs0)
    rule = DISC_COLLIDE_RO.get(collide)
    if rule != "OWN":
        raise RuntimeError(
            f"[DISC_COLL] the RO rule {rule!r} of disc collide={collide!r} cannot be authored — table "
            f"collision_presets.DISC_COLLIDE_RO={DISC_COLLIDE_RO}"
        )
    owner = getattr(cfg, "support_disc_owner", None)
    if not owner:
        raise RuntimeError(
            "[DISC_COLL] no disc owner statement (cfg.support_disc_owner) — discs that did not go "
            "through _build_disc_slots"
        )
    static = bool(getattr(getattr(cfg, "trajectory", None), "static_clip_assignment", False))
    if len(owner) > 1 and not static:
        raise RuntimeError(
            f"[DISC_COLL] {len(owner)} clips but no static clip assignment"
            " (cfg.trajectory.static_clip_assignment) — with the env clip changing at every reset the "
            "disc's own object cannot be decided"
        )
    shared = bool(getattr(cfg, "shared_object", False))
    idle = str(getattr(cfg, "idle_side", "") or "")
    per = [
        author_disc_other_object_nocollision(
            scene.stage,
            r,
            support_disc_slot_paths(scene, r),
            owner[i % len(owner)],
            object_prim_paths(scene, r),
            shared,
            idle,
        )
        for i, r in enumerate(roots)
    ]
    pairs, want = sum(d["pairs"] for d in per), sum(d["want"] for d in per)
    if pairs != want:
        raise RuntimeError(
            f"[DISC_COLL] RO filter pairs {pairs} ≠ expected {want} (real discs of every env × other object bodies)"
        )
    n_real = sum(d["discs"] for d in per)
    return {
        "rule": rule,
        "envs": len(roots),
        "pairs": pairs,
        "want": want,
        "discs_env0": per[0]["discs"],
        "text": f"objects=own disc<->own object on · disc<->other object off (filter pairs {pairs} = real discs of "
        f"every env {n_real} × "
        f"other object bodies · clips {len(owner)} · static assignment {static})",
    }


def _author_disc_support(cfg, roots, objs0):
    """The disc supports the only object. If the statement names an owner (that side, outside both) that the pinned pose
    (object_pin.support_disc_rows) removes from the support surface, the physics and the pinned pose disagree on the
    support. Shared object = both only; a one-side scene = that side.
    """
    stray = {o for r in (getattr(cfg, "support_disc_owner", None) or []) for o in r} - {
        None,
        "both",
        "static",
        *(objs0 if len(objs0) == 1 else ()),
    }
    if stray:
        raise RuntimeError(
            f"[DISC_COLL] scene objects {sorted(objs0)} (fewer than 2 distinct prims, no disc<->object "
            f"authoring) but the disc owner statement has {sorted(stray)} — the pinned pose drops that "
            "disc while the physics supports it (split support)"
        )
    return {
        "rule": None,
        "envs": len(roots),
        "pairs": 0,
        "want": 0,
        "discs_env0": 0,
        "text": f"objects=— no other object (scene objects {sorted(objs0)}) · disc<->object left on",
    }
