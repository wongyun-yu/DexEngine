"""All-object collision (collision_presets.ALL_OBJ_ENV): static body collision authoring, measured markers and session
poses (render entry, reset event).
Callers: the authoring `collision_authoring.author_collision_filters`, the render entry
`dexengine.record.render.episode`
(place_all_obj_bodies) and the reset event from `env_setup.objects` (reset_all_obj_multi). Off, all of them do nothing
(byte-identical). Per-kind collision authoring of extra slots belongs to
`collision_authoring.author_extra_collision`."""

from __future__ import annotations

from dexengine.core.errors import InputMissingError


def author_all_obj_collision(scene, cfg) -> dict:
    """All-object collision (collision_presets.ALL_OBJ_ENV) authoring and measured marker; off does nothing
    (byte-identical). For every static body (`build_all_obj_bodies`) it measures the collider, turns robot <-> body off
    (the same mechanism as receiving static slots) and measures on the stage that there is no main object <-> (static
    body + slot) filter at all. No builder, no collider or a remaining filter fails.
    Marker `[ALL_OBJ_COLL] on · …`; a session without it is invalid."""
    from dexengine.rl.tasks.scene.collision_presets import (
        ALL_OBJ_ENV,
        ALL_OBJ_KEEP_OFF_KINDS,
        ALL_OBJ_TAG,
        all_obj_collision,
    )

    return _author_all_obj_filters(scene, cfg, ALL_OBJ_ENV, ALL_OBJ_KEEP_OFF_KINDS, ALL_OBJ_TAG, all_obj_collision)


def _author_all_obj_filters(scene, cfg, ALL_OBJ_ENV, ALL_OBJ_KEEP_OFF_KINDS, ALL_OBJ_TAG, all_obj_collision):
    """Author the all-object collision filters of every env."""
    if not all_obj_collision():
        return {}
    from dexengine.rl.tasks.scene.collision_curriculum import authoring_roots

    rows = getattr(cfg, "all_obj_bodies", None)
    if rows is None:
        raise RuntimeError(
            f"{ALL_OBJ_TAG} {ALL_OBJ_ENV}=1 but no cfg.all_obj_bodies — this entry did not go "
            "through build_all_obj_bodies (no consumer)"
        )
    roots = authoring_roots(scene)
    if len(roots) != 1 and rows:
        raise RuntimeError(
            f"{ALL_OBJ_TAG} {len(roots)} envs — one-env sessions only (the static bodies are one clip's pose)"
        )
    multi = getattr(cfg, "all_obj_multi", None)
    if multi:  # several clips · static bodies present (build_all_obj_bodies): env i gets clip i % C's rows
        from dexengine.rl.tasks.scene.collision_bodies import env_index  # noqa: PLC0415

        C, K = len(multi["clips"]), int(multi["K"])
        if len(roots) != len(getattr(scene, "env_prim_paths", []) or []):
            raise RuntimeError(
                f"{ALL_OBJ_TAG} authoring roots {len(roots)} ≠ env count — the slot bodies differ per env"
                " (replicate_physics must be off)"
            )
        res = [
            _author_all_obj_root(
                scene, cfg, r, multi["rows"][env_index(r) % C], n_leaves=K, clip=env_index(r) % C, quiet=True
            )
            for r in roots
        ]
        cfg.all_obj_authored = True
        print(
            f"{ALL_OBJ_TAG} on · env {len(roots)} · clips {C} · slots {K} · static bodies in the physics "
            f"{sum(x['bodies'] for x in res)}(env 0: {[(x['name'], x['obj_id']) for x in multi['rows'][0]]}) · "
            f"stand-in bodies without collider {sum(K - x['bodies'] for x in res)} · slots "
            f"{sum(x['slots'] for x in res)} · "
            f"main object bodies {sum(x['objects'] for x in res)} · collision pairs {sum(x['pairs'] for x in res)} · "
            f"filters left 0 · "
            f"robot <-> static bodies off · exception (main objects off) {list(ALL_OBJ_KEEP_OFF_KINDS)} slots "
            f"{sum(x['keep_off'] for x in res)}",
            flush=True,
        )
        return {
            "bodies": sum(x["bodies"] for x in res),
            "slots": sum(x["slots"] for x in res),
            "objects": sum(x["objects"] for x in res),
            "pairs": sum(x["pairs"] for x in res),
            "envs": len(roots),
        }
    if len(roots) != 1:  # 0 static bodies (build_all_obj_bodies got several clips): slots and filters measured per env
        res = [_author_all_obj_root(scene, cfg, r, []) for r in roots]
        cfg.all_obj_authored = True
        print(
            f"{ALL_OBJ_TAG} on · env {len(roots)} · 0 static bodies · slots {sum(x['slots'] for x in res)} · "
            f"main object bodies {sum(x['objects'] for x in res)} · collision pairs {sum(x['pairs'] for x in res)} · "
            f"filters left 0 · "
            f"exception (main objects off) {list(ALL_OBJ_KEEP_OFF_KINDS)} slots {sum(x['keep_off'] for x in res)}",
            flush=True,
        )
        return {
            "bodies": 0,
            "slots": sum(x["slots"] for x in res),
            "objects": sum(x["objects"] for x in res),
            "pairs": sum(x["pairs"] for x in res),
            "envs": len(roots),
        }
    out = _author_all_obj_root(scene, cfg, roots[0], rows)
    cfg.all_obj_authored = True
    return out


def _author_all_obj_root(
    scene, cfg, r: str, rows: list, n_leaves: int | None = None, clip: int = 0, quiet: bool = False
) -> dict:
    """Authoring and measurement of one env root (the body of author_all_obj_collision; markers and failure rules as
    they are). n_leaves = that env's static body slot count (multi-clip session: the first len(rows) = that clip's rows,
    the rest = stand-in bodies that must have no collider) · clip = the clip number of the slot kind statement
    (extra_slot_kinds) (env i -> i % C)."""
    from pxr import Usd, UsdPhysics

    from dexengine.rl.tasks.scene.collision_bodies import object_prim_paths, slot_prim_paths
    from dexengine.rl.tasks.scene.collision_presets import ALL_OBJ_LEAF, ALL_OBJ_TAG
    from dexengine.rl.tasks.scene.extra_slots import PREFIX

    stage = scene.stage
    leaves = [f"{r}/{ALL_OBJ_LEAF}{k}" for k in range(len(rows))]
    stand = [f"{r}/{ALL_OBJ_LEAF}{k}" for k in range(len(rows), n_leaves or len(rows))]

    def _bodies(paths):
        return [
            q.GetPath().pathString
            for s in paths
            for q in Usd.PrimRange(stage.GetPrimAtPath(s))
            if q.HasAPI(UsdPhysics.RigidBodyAPI)
        ]

    def _has_collider(path):
        return any(
            q.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(q).GetCollisionEnabledAttr().Get() is not False
            for q in Usd.PrimRange(stage.GetPrimAtPath(path), Usd.TraverseInstanceProxies())
        )

    if none := [x["name"] for x, p in zip(rows, leaves, strict=True) if not _has_collider(p)]:
        raise RuntimeError(
            f"{ALL_OBJ_TAG} static bodies {none} have no enabled collider — they did not get into the physics"
        )
    if bad_s := [p for p in stand if _has_collider(p)]:
        raise RuntimeError(
            f"{ALL_OBJ_TAG} stand-in bodies {bad_s} got a collider — they would collide with objects absent from that "
            f"clip"
        )
    if leaves:
        from dexengine.rl.tasks.scene.collision_groups import plan

        plan(scene, r, leaves)
    from dexengine.rl.tasks.scene.collision_presets import ALL_OBJ_KEEP_OFF_KINDS

    slots = slot_prim_paths(scene, r, PREFIX, "ALL_OBJ_COLL")
    kinds_all = getattr(cfg, "extra_slot_kinds", None) or [[]]
    kinds = list(kinds_all[clip % len(kinds_all)])
    real = [s for s, k in zip(slots, kinds, strict=False) if k is not None and k not in ALL_OBJ_KEEP_OFF_KINDS]
    keep_off = [s for s, k in zip(slots, kinds, strict=False) if k in ALL_OBJ_KEEP_OFF_KINDS]
    objs = _bodies(list(object_prim_paths(scene, r).values()))
    others = _bodies(leaves) + _bodies(real)
    return _check_exception_kinds(
        scene,
        r,
        rows,
        quiet,
        ALL_OBJ_KEEP_OFF_KINDS,
        ALL_OBJ_TAG,
        UsdPhysics,
        _bodies,
        _has_collider,
        keep_off,
        kinds,
        leaves,
        objs,
        others,
        real,
        stage,
    )


def _check_exception_kinds(
    scene,
    r,
    rows,
    quiet,
    ALL_OBJ_KEEP_OFF_KINDS,
    ALL_OBJ_TAG,
    UsdPhysics,
    _bodies,
    _has_collider,
    keep_off,
    kinds,
    leaves,
    objs,
    others,
    real,
    stage,
):
    """Exception kinds (main objects off) must really not collide with the main objects: no collider, or every pair with
    the main objects filtered.
    """
    if bad := [s for s in keep_off if _has_collider(s)]:
        raise RuntimeError(
            f"{ALL_OBJ_TAG} exception kind {ALL_OBJ_KEEP_OFF_KINDS} slots {bad} got a collider — "
            "not off with the main objects"
        )

    def _filtered(a):
        rel = UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath(a)).GetFilteredPairsRel()
        return {t.pathString for t in rel.GetTargets()}

    left = [(a, b) for a in objs for b in others if b in _filtered(a) or a in _filtered(b)]
    if left:
        raise RuntimeError(f"{ALL_OBJ_TAG} {len(left)} main object <-> object filter pairs remain: {left[:3]}")
    n_pair = len(objs) * len(others) + len(objs) * (len(objs) - 1) // 2
    if quiet or (not rows and r != scene.env_prim_paths[0]):
        return {
            "bodies": len(rows),
            "slots": len(real),
            "objects": len(objs),
            "pairs": n_pair,
            "keep_off": len(keep_off),
        }
    print(
        f"{ALL_OBJ_TAG} on · static bodies in the physics {len(rows)} "
        f"{[(x['name'], x['obj_id'], x['source']) for x in rows]} · "
        f"slots {len(real)} (kinds {sorted(str(k) for k in kinds if k is not None)}) · main object bodies {len(objs)} "
        f"· "
        f"collision pairs {n_pair} (main objects × static bodies {len(objs) * len(_bodies(leaves))} · "
        f"main objects × slots {len(objs) * len(_bodies(real))} · between main objects "
        f"{len(objs) * (len(objs) - 1) // 2}) · "
        f"filters left 0 · robot <-> static bodies off · exception (main objects off) {list(ALL_OBJ_KEEP_OFF_KINDS)} "
        f"slots {len(keep_off)}",
        flush=True,
    )
    return {"bodies": len(rows), "slots": len(real), "objects": len(objs), "pairs": n_pair, "keep_off": len(keep_off)}


def place_all_obj_bodies(raw_env, extras: dict, shift_z: float) -> None:
    """After the render entry's decisions (`extras_apply.apply`), place the static bodies at the visual prims' pose;
    off does nothing. Bodies of rows decided against (rollout overlap, occlusion gate) go to the hidden pose (so nothing
    collides with an invisible object). A row absent at spawn time that becomes added fails."""
    from dexengine.rl.tasks.scene.collision_presets import ALL_OBJ_KEY, ALL_OBJ_TAG, all_obj_collision

    if not all_obj_collision():
        return
    import torch

    rows = raw_env.cfg.all_obj_bodies
    placed = {p["name"]: p for p in extras["placed"]}
    if extra := sorted(set(placed) - {x["name"] for x in rows}):
        raise RuntimeError(f"{ALL_OBJ_TAG} a row not stood at spawn time became added: {extra}")
    origin = raw_env.scene.env_origins[0]
    said, poses = [], []
    for k, x in enumerate(rows):
        ent = raw_env.scene[f"{ALL_OBJ_KEY}{k}"]
        p = placed.get(x["name"])
        if p is None:
            pose = ent.data.default_root_state[:, :7].clone()
        else:
            pose = torch.tensor([[*p["scene_xyz"], *p["scene_wxyz"]]], dtype=torch.float32, device=origin.device)
            pose[:, :3] += origin
            pose[:, 2] += float(shift_z)
        ent.write_root_pose_to_sim(pose)
        poses.append(pose)
        said.append(f"{x['name']} {'stands' if p is not None else 'hidden'}")
    raw_env.cfg.all_obj_placed = {n: p["prim"] for n, p in placed.items()}
    raw_env.cfg.all_obj_poses = poses
    print(f"{ALL_OBJ_TAG} pose {said or 'no static bodies'} · world shift {float(shift_z):g} m", flush=True)


def reset_all_obj_multi(env, env_ids) -> None:
    """Static body poses of a multi-clip session (a reset event registered by build_all_obj_bodies): env i -> the pose
    of
    added row k of clip i % C (scene frame, extras.decide's scene_xyz · scene_wxyz, + the env origin); slots
    absent from that clip (stand-in bodies) at the hidden pose. The first time, the PhysX pose is compared with the
    statement and a difference above ALL_OBJ_POSE_TOL_M fails. Marker `[ALL_OBJ_COLL] pose`."""
    import torch

    from dexengine.rl.tasks.scene.collision_presets import ALL_OBJ_KEY, ALL_OBJ_POSE_TOL_M, ALL_OBJ_TAG

    m = env.cfg.all_obj_multi
    C, K = len(m["clips"]), int(m["K"])
    ids = list(range(env.num_envs)) if env_ids is None else [int(i) for i in env_ids.tolist()]
    if not ids:
        return
    org = env.scene.env_origins
    idt = torch.tensor(ids, device=org.device, dtype=torch.long)
    hid = [0.0, 0.0, float(m["hidden_z"]), 1.0, 0.0, 0.0, 0.0]
    want = {}
    for k in range(K):
        rows = [m["rows"][i % C] for i in ids]
        pose = torch.tensor(
            [[*r[k]["scene_xyz"], *r[k]["scene_wxyz"]] if k < len(r) else hid for r in rows],
            dtype=torch.float32,
            device=org.device,
        )
        pose[:, :3] += org[idt]
        env.scene[f"{ALL_OBJ_KEY}{k}"].write_root_pose_to_sim(pose, env_ids=idt)
        want[k] = pose
    if env.__dict__.get("_all_obj_multi_said"):
        return
    env.__dict__["_all_obj_multi_said"] = True
    err, n_on = 0.0, 0
    for k, pose in want.items():
        got = env.scene[f"{ALL_OBJ_KEY}{k}"].root_physx_view.get_transforms()[idt]
        err = max(err, float((got[:, :3] - pose[:, :3]).norm(dim=1).max()))
        n_on += sum(k < len(m["rows"][i % C]) for i in ids)
    print(
        f"{ALL_OBJ_TAG} pose env {len(ids)} · slots {K} · standing {n_on} · hidden {len(ids) * K - n_on} · "
        f"PhysX <-> statement max {err * 1000:.2f} mm",
        flush=True,
    )
    if err > ALL_OBJ_POSE_TOL_M:
        raise RuntimeError(f"{ALL_OBJ_TAG} static body pose is {err * 1000:.1f} mm off the statement")


def assert_all_obj_ready(raw_env) -> None:
    """Just before the policy loop: with the switch given but no authoring and pose writing, fail (no consumer)."""
    from dexengine.rl.tasks.scene.collision_presets import ALL_OBJ_ENV, ALL_OBJ_KEY, ALL_OBJ_TAG, all_obj_collision

    if not all_obj_collision():
        return
    if not getattr(raw_env.cfg, "all_obj_authored", False) or getattr(raw_env.cfg, "all_obj_placed", None) is None:
        raise InputMissingError(
            f"{ALL_OBJ_TAG} {ALL_OBJ_ENV}=1 but authoring and pose writing did not run — no consumer"
        )
    for k, pose in enumerate(
        raw_env.cfg.all_obj_poses
    ):  # every episode: start from the same pose even if the previous episode's reset moved it
        raw_env.scene[f"{ALL_OBJ_KEY}{k}"].write_root_pose_to_sim(pose)
    check_all_obj_pose(raw_env, "frame 0")


def check_all_obj_pose(raw_env, when: str) -> None:
    """Measured physics pose of the static bodies <-> world pose of the visual prims; off does nothing. Marker
    `[ALL_OBJ_COLL] pose check <when> …`; an error above `ALL_OBJ_POSE_TOL_M` (a reset moved it back to the hidden pose,
    ...) fails. Bodies of rows not added must be at the hidden pose."""
    from dexengine.rl.tasks.scene.collision_presets import (
        ALL_OBJ_KEY,
        ALL_OBJ_POSE_TOL_M,
        ALL_OBJ_TAG,
        all_obj_collision,
    )

    if not all_obj_collision():
        return
    import numpy as np

    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    said, bad = [], []
    for k, x in enumerate(raw_env.cfg.all_obj_bodies):
        ent = raw_env.scene[f"{ALL_OBJ_KEY}{k}"]
        got = ent.root_physx_view.get_transforms()[0].detach().cpu().numpy()  # xyz · xyzw
        prim = raw_env.cfg.all_obj_placed.get(x["name"])
        if prim is None:
            want = ent.data.default_root_state[0, :3].detach().cpu().numpy()
            rot = None
        else:
            m = UsdGeom.Xformable(stage.GetPrimAtPath(prim)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            want = np.array(m.ExtractTranslation(), dtype=float)
            q = m.ExtractRotationQuat()
            d = abs(float(np.dot([*q.GetImaginary(), q.GetReal()], got[3:7])))
            rot = float(np.degrees(2 * np.arccos(min(1.0, d))))
        e = float(np.linalg.norm(got[:3] - want))
        said.append(f"{x['name']} {e * 1000:.2f} mm" + (" (hidden)" if prim is None else f" {rot:.2f}°"))
        if e > ALL_OBJ_POSE_TOL_M:
            bad.append(x["name"])
    print(f"{ALL_OBJ_TAG} pose check {when}: {said or 'no static bodies'}", flush=True)
    if bad:
        raise SystemExit(
            f"{ALL_OBJ_TAG} the physics pose of static bodies {bad} differs from the visual prims ({when}) — {said}"
        )
