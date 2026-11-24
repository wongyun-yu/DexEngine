"""S-dyn-pairs in the physics: the two main objects of a clip whose statement says pair_off (cfg statement CFG_KEY,
one bool per loader clip, writer env_setup.dyn_pairs) do not collide with each other; every other pair keeps its row.

Mechanism = collision groups, the robot-off form (collision_groups), not FilteredPairs (collision_presets
DYN_PAIR_COLLIDE states the choice and its memory reading). In an env whose clip (env i -> clip i % C, static clip
assignment) is pair_off, the right and the left object root each leave the env group for a group of their own
(`dyn_a<j>`, `dyn_b<j>`, j = the env group's number). Each lists itself and every group the env group collides with
(read from the stage: the env group, the env's robot and robot-off groups, the global group); those list it back; the
two never list each other, so only object <-> object is cut. Timing as the robot-off groups: the pre-parse hook plans
(`plan`), filter_collisions authors after the robot-off groups (`author_pending`), a plan still pending at POSTINIT
fails (`assert_done`). After authoring the stage is read back (`readback`): each object body in its own group alone,
each group colliding with exactly the env group's partners, the pair blocked, every other body of the env in the groups
it had. The truth table reads the cut as R7 (collision_check), and `audit_line` compares the measured cut of every
audited env with its clip's statement (all pairs cut when pair_off, none otherwise; anything else fails). Markers
`[DYN_PAIRS] groups …` and `[DYN_PAIRS] audit …`. Needs only pxr.
"""

from __future__ import annotations

from dexengine.rl.tasks.scene import collision_groups as CG

CFG_KEY = "dyn_pair_off"  # cfg statement: [bool per loader clip] (env_setup.dyn_pairs)
PLAN_ATTR = "_dyn_pair_plan"  # scene attribute: [(env root, right object root, left object root)]
GROUPS = ("dyn_a", "dyn_b")
MARK = "[DYN_PAIRS]"


def plan(scene, cfg) -> int:
    """Queue the object pair of every env whose clip is pair_off; returns the number of envs queued."""
    from dexengine.rl.tasks.scene.collision_bodies import env_index, object_prim_paths  # noqa: PLC0415

    off = list(getattr(cfg, CFG_KEY, None) or [])
    if not any(off):
        return 0
    static = bool(getattr(getattr(cfg, "trajectory", None), "static_clip_assignment", False))
    if len(off) > 1 and not static:
        raise RuntimeError(f"{MARK} {len(off)} clips without a static clip assignment: the env's pair is undecided")
    todo = []
    for root in list(getattr(scene, "env_prim_paths", []) or []):
        if not off[env_index(root) % len(off)]:
            continue
        CG.plan(scene, root, [])  # the group authoring point and per-env groups must exist (fails otherwise)
        paths = object_prim_paths(scene, root)
        if set(paths) != {"right", "left"}:
            raise RuntimeError(f"{MARK} {root}: object roots {sorted(paths)} — a pair-off clip needs right and left")
        todo.append((root, paths["right"], paths["left"]))
    setattr(scene, PLAN_ATTR, todo)
    return len(todo)


def author_pending(scene) -> dict | None:
    """Author the queued pair groups and read them back (None when nothing is queued)."""
    todo = getattr(scene, PLAN_ATTR, None)
    if not todo:
        return None
    info = author(scene.stage, todo)
    setattr(scene, PLAN_ATTR, None)
    return info


def assert_done(scene) -> None:
    """POSTINIT gate: a plan still pending means the pairs still collide."""
    if todo := getattr(scene, PLAN_ATTR, None):
        raise RuntimeError(f"{MARK} {len(todo)} envs still wait for their pair groups: filter_collisions never ran")


def _rows(stage, todo: list) -> list:
    """[(env root, env group, number, [right root, left root], partner groups)]; partners = every group the env group
    collides with (itself included, the pair groups not), read from the stage, so the objects keep exactly the env
    group's partners (the same before and after authoring)."""
    env2g = CG._env_groups(stage)
    ix = CG.GroupIndex(stage)
    if not ix.invert:
        raise RuntimeError(
            f"{MARK} the stage's collision group filter is not inverted: listed groups would not collide"
        )
    out = []
    for root, a, b in todo:
        g = env2g.get(root)
        if g is None:
            raise RuntimeError(f"{MARK} {root} has no env group under {CG.ROOT}")
        partners = sorted(G for G in ix.filt if ix.allow(g, G) and not G.rsplit("/", 1)[-1].startswith(GROUPS))
        out.append((root, g, g.rsplit("group", 1)[-1], [a, b], partners))
    return out


def author(stage, todo: list) -> dict:
    """todo = [(env root, right object root, left object root)] -> two groups per env, read back. {groups, envs}."""
    from pxr import Sdf  # noqa: PLC0415

    rows = _rows(stage, todo)
    before = _others(stage, rows)
    lay = stage.GetRootLayer()
    for _r, _g, j, _objs, _p in rows:
        for nm in GROUPS:
            if stage.GetPrimAtPath(f"{CG.ROOT}/{nm}{j}").IsValid():
                raise RuntimeError(f"{MARK} {CG.ROOT}/{nm}{j} already exists: authored twice")
    with Sdf.ChangeBlock():
        for _r, g, j, objs, partners in rows:
            gs = lay.GetPrimAtPath(g)
            exc = gs.relationships.get(CG.EXC) or Sdf.RelationshipSpec(gs, CG.EXC, False)
            for nm, obj in zip(GROUPS, objs, strict=True):
                path = f"{CG.ROOT}/{nm}{j}"
                CG._group_spec(lay, path, [obj], [path, *partners])
                exc.targetPathList.Append(obj)
                for p in partners:
                    lay.GetPrimAtPath(p).relationships[CG.FILT].targetPathList.Append(path)
    readback(stage, rows, before)
    info = {"groups": 2 * len(rows), "envs": len(rows)}
    print(
        f"{MARK} groups: object <-> object cut in {info['envs']} envs ({info['groups']} groups · each object still "
        "collides with its env, robot and robot-off groups · read-back ok)",
        flush=True,
    )
    return info


def _others(stage, rows: list) -> dict:
    """{body: its groups} of every body of the authored envs outside the two object roots."""
    ix = CG.GroupIndex(stage)
    out = {}
    for root, _g, _j, objs, _p in rows:
        mine = {b for o in objs for b in CG._bodies(stage, o)}
        out.update({b: ix.groups_of(b) for b in CG._bodies(stage, root) if b not in mine})
    return out


def readback(stage, rows: list, before: dict | None = None) -> None:
    """Every authored env: each object body in its own pair group alone, that group colliding with exactly itself and
    the env group's partners (so not with the other object), and, given `before` (`_others` before authoring), every
    other body still in the groups it had; anything else fails."""
    ix = CG.GroupIndex(stage)
    bad = []
    for root, _g, j, objs, partners in rows:
        for nm, obj in zip(GROUPS, objs, strict=True):
            want = f"{CG.ROOT}/{nm}{j}"
            bodies = CG._bodies(stage, obj)
            wrong = [(b, sorted(v)) for b in bodies if (v := ix.groups_of(b)) != {want}]
            if not bodies or wrong:
                bad.append(f"{root} {obj}: bodies {len(bodies)} · not in {want} alone: {wrong[:2]}")
            reach = sorted(G for G in ix.filt if ix.allow(want, G))
            if reach != sorted({want, *partners}):
                bad.append(f"{root} {want} collides with {reach}, not {sorted({want, *partners})}")
        ga, gb = (f"{CG.ROOT}/{nm}{j}" for nm in GROUPS)
        if ix.allow(ga, gb):
            bad.append(f"{root} {ga} <-> {gb} still collide")
    moved = [b for b, v in (before or {}).items() if ix.groups_of(b) != v]
    if moved:
        bad.append(f"other bodies changed groups: {moved[:2]}")
    if bad:
        raise RuntimeError(f"{MARK} read-back: {len(bad)} mismatches: " + " · ".join(bad[:3]))


def cut(stage, right_bodies, left_bodies) -> int:
    """How many (right object body, left object body) pairs the stage's collision groups cut (the R7 reading)."""
    if not (right_bodies and left_bodies):
        return 0
    ix = CG.GroupIndex(stage)
    return sum(ix.blocked(a, b) for a in right_bodies for b in left_bodies)


def audit_line(rows: list, idx: list) -> str:
    """The audited envs' measured group cut (audit row `dyn_cut` = (cut, right × left body pairs)) against their
    statement (`dyn_pair_off`): pair_off needs every pair cut, any other env none. A mismatch fails."""
    bad, cells = [], []
    for i, r in zip(idx, rows, strict=True):
        n, full = r["dyn_cut"]
        want = full if r["dyn_pair_off"] else 0
        cells.append(f"env_{i} {'pair_off' if r['dyn_pair_off'] else '-'} cut {n}/{full}")
        if n != want or (r["dyn_pair_off"] and not full):
            bad.append(cells[-1])
    if bad:
        raise RuntimeError(
            f"{MARK} audit: the object <-> object group cut differs from the statement: " + " · ".join(bad)
        )
    return f"{MARK} audit: " + " · ".join(cells) + " · = statement"
