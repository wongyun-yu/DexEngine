"""Collision authoring of the extra object slots (table = collision_presets.EXTRA_KIND_COLLIDE), called from
collision_authoring.author_collision_filters before PhysX parses the stage."""

from __future__ import annotations


def author_extra_collision(scene, cfg) -> dict:
    """Author extra object slot collisions as stated **per slot kind** in `collision_presets.EXTRA_KIND_COLLIDE`
    (kind = the extra_slots row's kind: moving · touched · receiver).
      ON partner        nothing (PhysX default = collide)
      main object OFF   the (slot body, every main object body) pairs are turned off with FilteredPairs
      hand · arm OFF    the whole robot (torso and base included) <-> the slot is turned off with collision groups
                        (`collision_groups`: planned here, authored once Isaac Lab's env groups exist). A table that
                        turns hand and arm off separately fails (no such authoring).
    Slot kinds = `cfg.extra_slot_kinds`[clip][slot] (writer `env_setup.extra_slots.build_extra_slots`; env i -> clip
    i % C; None = an empty place absent from that clip, hidden, so it is turned off against the main objects only, as
    now). A slot prim count different from the statement `cfg.extra_object_slots`, or no kind statement, fails. After
    authoring the rels are re-read to count that every pair is filtered. Without slots nothing is authored (the stage
    stays byte-identical).
    Colliders: whether a collider stands on a slot body is stated by `collision_presets.extra_has_collider` (only kinds
    with an ON partner in their table; empty places none; builder `build_extra_slots`). Here the stage is **measured**
    (is there an enabled collision shape, CollisionAPI with collisionEnabled ≠ False, under the slot prim); any slot
    that
    disagrees with the statement fails (a collider that should not stand grows memory with the env count, a missing
    one lets the hand pass through the extra). Marker `[EXTRA_COLL]`, two lines, only with slots (the second line =
    per kind whether a collider stands, and the measured counts)."""
    from pxr import Usd, UsdPhysics

    from dexengine.rl.tasks.scene.collision_bodies import author_filtered_pairs, object_prim_paths, slot_prim_paths
    from dexengine.rl.tasks.scene.collision_curriculum import authoring_roots
    from dexengine.rl.tasks.scene.collision_presets import extra_collide, extra_collide_text
    from dexengine.rl.tasks.scene.extra_slots import PREFIX

    said = int(getattr(cfg, "extra_object_slots", 0) or 0)
    roots = authoring_roots(scene)
    slots0 = slot_prim_paths(scene, roots[0], PREFIX, "EXTRA_COLL") if roots else []
    if len(slots0) != said:
        raise RuntimeError(f"[EXTRA_COLL] extra slot prims {len(slots0)} ≠ statement cfg.extra_object_slots={said}")
    if not said:
        return {"slots": 0, "envs": len(roots), "pairs": 0, "robot_off": 0}
    n_filter, n_kind, n_robot, per_env, seen, stage, tables = _author_extra_slot_filters(
        scene,
        cfg,
        PREFIX,
        Usd,
        UsdPhysics,
        author_filtered_pairs,
        extra_collide,
        object_prim_paths,
        roots,
        said,
        slot_prim_paths,
    )
    bad = [i for i, (want, got) in enumerate(per_env) if want != got]
    if bad:
        raise RuntimeError(
            f"[EXTRA_COLL] envs {bad[:5]} with missing slot × main object pairs: {[per_env[i] for i in bad[:3]]}"
        )
    pairs = sum(g for _, g in per_env)
    print(
        f"[EXTRA_COLL] collision partners per kind {extra_collide_text()} · extra slots {said} · kinds "
        f"over {len(roots)} authoring envs "
        + " ".join(f"{k}={v}" for k, v in n_kind.items())
        + f" · main object filter pairs {pairs} · robot-off slots {n_filter} (collision groups: the whole robot <-> "
        f"that slot)",
        flush=True,
    )
    return {
        "slots": said,
        "envs": len(roots),
        "pairs": pairs,
        "robot_off": n_robot,
        "robot_filter": n_filter,
        "robot_mech": "groups",
        "kinds": dict(n_kind),
        "colliders": measured_colliders(stage, seen, list(tables)),
    }


def _author_extra_slot_filters(
    scene,
    cfg,
    PREFIX,
    Usd,
    UsdPhysics,
    author_filtered_pairs,
    extra_collide,
    object_prim_paths,
    roots,
    said,
    slot_prim_paths,
):
    """Author the collision filters of the extra-slot objects."""
    from dexengine.rl.tasks.scene.collision_presets import EXTRA_KIND_COLLIDE, slot_robot_off  # noqa: PLC0415

    kinds = getattr(cfg, "extra_slot_kinds", None)
    tables = {k: extra_collide(k) for k in EXTRA_KIND_COLLIDE}
    _check_extra_slot_kinds(cfg, said, kinds, tables)
    stage, per_env, n_kind, n_robot, n_filter = scene.stage, [], dict.fromkeys([*tables, "empty"], 0), 0, 0
    seen: list = []  # (slot prim, kind): input of the measured collider check

    def _bodies(paths):
        return [
            q.GetPath().pathString
            for s in paths
            for q in Usd.PrimRange(stage.GetPrimAtPath(s))
            if q.HasAPI(UsdPhysics.RigidBodyAPI)
        ]

    for i, r in enumerate(roots):
        slot_roots = slot_prim_paths(scene, r, PREFIX, "EXTRA_COLL")  # name sort = slot number (≤ 10)
        row = kinds[i % len(kinds)]
        objs = _bodies(list(object_prim_paths(scene, r).values()))
        if len(_bodies(slot_roots)) != said or not objs:
            raise RuntimeError(
                f"[EXTRA_COLL] {r}: slot bodies {len(_bodies(slot_roots))} (statement {said}) · main object bodies "
                f"{len(objs)} — the pairs cannot be built"
            )
        obj_off, robot_off = [], []
        for sr, kind in zip(slot_roots, row, strict=True):
            if kind is not None and kind not in tables:
                raise RuntimeError(f"[EXTRA_COLL] {r}: slot kind {kind!r} — only the tables {sorted(tables)}")
            n_kind["empty" if kind is None else kind] += 1
            seen.append((sr, kind))
            t = (
                tables[kind] if kind is not None else {"hand": "ON", "arm": "ON", "objects": "OFF"}
            )  # empty place = hidden
            if t["objects"] == "OFF":
                obj_off.append(sr)
            if slot_robot_off(t):
                robot_off.append(sr)
        slots = _bodies(obj_off)
        if slots:
            author_filtered_pairs(stage, slots, objs)
        got = sum(
            len(
                set(UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath(b)).GetFilteredPairsRel().GetTargets())
                & {stage.GetPrimAtPath(o).GetPath() for o in objs}
            )
            for b in slots
        )
        per_env.append((len(slots) * len(objs), got))
        if robot_off:
            n_robot += len(robot_off)
            n_filter += _author_robot_off(scene, r, robot_off)
    return n_filter, n_kind, n_robot, per_env, seen, stage, tables


def _check_extra_slot_kinds(cfg, said, kinds, tables):
    """Refuse mixed or misdeclared extra-slot kinds."""
    if mixed := sorted(k for k, t in tables.items() if t["hand"] != t["arm"]):
        raise RuntimeError(
            f"[EXTRA_COLL] table {[(k, tables[k]) for k in mixed]} — no authoring turns hand and arm off"
            f" separately (both ON or both OFF)"
        )
    if not kinds or any(len(row) != said for row in kinds):
        raise RuntimeError(
            f"[EXTRA_COLL] slot kind statement cfg.extra_slot_kinds={kinds!r} — must have {said} slots"
            f" per clip (writer build_extra_slots)"
        )
    if said > 10:
        raise RuntimeError(
            f"[EXTRA_COLL] {said} slots > 10 — slot prim order (name sort) would split from number order"
        )
    static = bool(getattr(getattr(cfg, "trajectory", None), "static_clip_assignment", False))
    if len(kinds) > 1 and not static:
        raise RuntimeError(
            f"[EXTRA_COLL] {len(kinds)} clips but no static clip assignment (cfg.trajectory."
            f"static_clip_assignment) — with the env clip changing at every reset the slot kinds cannot "
            f"be decided"
        )


def _author_robot_off(scene, env_root: str, slots: list) -> int:
    """Queue the robot-off slots of one env for their collision groups (`collision_groups`, authored and read back
    once Isaac Lab's env groups exist); returns the number of slots."""
    from dexengine.rl.tasks.scene.collision_groups import plan

    try:
        plan(scene, env_root, slots)
    except RuntimeError as e:
        raise RuntimeError(f"[EXTRA_COLL] {env_root}: robot-off slot groups cannot be planned — {e}") from e
    return len(slots)


def measured_colliders(stage, slots: list, kinds: list) -> dict:
    """{kind | 'empty': number of slots with an enabled collision shape}; `slots` = (slot prim path, kind | None).
    Measures the stage (any CollisionAPI prim under the slot prim with collisionEnabled ≠ False, reading into
    instances). Any slot that disagrees with the statement `collision_presets.extra_has_collider(kind)` fails
    (author_extra_collision). Marker line = collision_presets.collider_line: per kind the statement (with | without)
    and the measured count / that kind's slot count (compared by scripts/rl/judge_sidecar.sh between the training and
    the judge log)."""
    from pxr import Usd, UsdPhysics

    from dexengine.rl.tasks.scene.collision_presets import collider_line, extra_has_collider

    n, tot, wrong = dict.fromkeys([*kinds, "empty"], 0), dict.fromkeys([*kinds, "empty"], 0), []
    for path, kind in slots:
        got = any(
            q.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(q).GetCollisionEnabledAttr().Get() is not False
            for q in Usd.PrimRange(stage.GetPrimAtPath(path), Usd.TraverseInstanceProxies())
        )
        n["empty" if kind is None else kind] += got
        tot["empty" if kind is None else kind] += 1
        if got != extra_has_collider(kind):
            wrong.append(f"{path} kind {kind} measured {'with' if got else 'without'}")
    if wrong:
        raise RuntimeError(
            f"[EXTRA_COLL] the colliders of {len(wrong)} slots differ from the statement "
            f"(collision_presets.extra_has_collider: only kinds with an ON partner in their table): "
            + " · ".join(wrong[:4])
            + " — builder build_extra_slots"
        )
    print(collider_line(kinds, n, tot), flush=True)
    return n
