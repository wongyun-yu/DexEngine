"""Robot-off bodies as collision groups: the one way a body is turned off against the whole robot.

Rule: inside one env, every rigid body of the robot (`Robot` and `RobotL` of the split articulation, torso and base
included) and every robot-off body never collide; each still collides with every other body of its env and with the
global group; nothing collides across envs. Robot-off bodies = the extra slots whose kind table turns hand and arm off
(`collision_presets.slot_robot_off`), the support discs (rule D40) and the all-object scene's static bodies.

Isaac Lab gives each env one group `/World/collisions/group<i>` (cloner `filter_collisions`, inverted filter:
`physics:filteredGroups` lists the groups that DO collide). Per env this module takes the robot roots and the robot-off
roots out of that env group (`excludes`) and puts them into two groups of their own, `robot<i>` and `robot_off<i>`.
Each lists itself, the env group and the global group when there is one; neither lists the other, so only the
robot <-> robot-off pair is cut. Shapes in no group (the ground plane) stay outside the group filter.

Measured (1024 envs, one robot-off slot per env, same scene; PhysX GPU heap · host pinned /dev/zero map): the same rule
written as FilteredPairs (every robot body x every slot body) holds 22.41 GB · 20.6 GB, almost all of it narrowphase and
simulation; as these groups 3.76 GB · 2.45 GB; without the slot 4.3 GB · 3.1 GB. Robot <-> slot contacts are 0 in both
forms (PhysX rigid contact view, 560 steps).

Timing: env groups exist only after Isaac Lab's `filter_collisions` (after the spawners) and the groups must be
written before PhysX parses the stage. The pre-parse hook therefore only plans (`plan`); the hook scene's
`filter_collisions` authors (`author_pending`); a plan still pending at POSTINIT fails (`assert_done`). A scene that
cannot author groups (no `AUTHORS_ROBOT_OFF_GROUPS`, or replicated physics, where Isaac Lab makes no env groups) fails
at `plan`. After authoring the stage is read back (`GroupIndex`): every robot body, robot-off body and remaining env
body sits in exactly its own group, robot <-> robot-off is blocked, and both reach the env group; anything else fails.
Marker `[ROBOT_OFF_GROUPS]`. Needs only pxr.

No second path: a FilteredPairs entry between a robot body and a robot-off body (extra slot or support disc) fails the
POSTINIT audit wherever it came from (`refuse_filtered`, called by the stage readers of both rows: the truth table's RD
row and the extra-slot audit's RE row).
"""

from __future__ import annotations

ROOT = "/World/collisions"
ROBOT_LEAVES = ("Robot", "RobotL")
ROBOT_GROUP = "robot"
OFF_GROUP = "robot_off"
INC = "collection:colliders:includes"
EXC = "collection:colliders:excludes"
FILT = "physics:filteredGroups"
MARK = "[ROBOT_OFF_GROUPS]"
PLAN_ATTR = "_robot_off_plan"  # scene attribute: {env root: [robot-off roots]} waiting for the env groups
INFO_ATTR = "_robot_off_groups"  # scene attribute: the authored summary


def plan(scene, env_root: str, roots: list) -> None:
    """Queue the robot-off roots of one env; a scene that cannot author groups before parsing fails."""
    if not getattr(type(scene), "AUTHORS_ROBOT_OFF_GROUPS", False):
        raise RuntimeError(
            f"{MARK} scene {type(scene).__name__} has no group authoring point (PostSpawnHookScene.filter_collisions); "
            "robot <-> robot-off would stay on"
        )
    if bool(getattr(getattr(scene, "cfg", None), "replicate_physics", True)):
        raise RuntimeError(f"{MARK} replicate_physics=True: Isaac Lab makes no env groups ({ROOT}/group<i>)")
    if not roots:
        return
    todo = getattr(scene, PLAN_ATTR, None)
    if todo is None:
        todo = {}
        setattr(scene, PLAN_ATTR, todo)
    todo.setdefault(env_root, []).extend(r for r in roots if r not in todo.get(env_root, []))


def author_pending(scene) -> dict | None:
    """Author the queued groups and read them back (None when nothing is queued)."""
    todo = getattr(scene, PLAN_ATTR, None)
    if not todo:
        return None
    info = author(scene.stage, sorted(todo.items()))
    setattr(scene, PLAN_ATTR, None)
    setattr(scene, INFO_ATTR, info)
    return info


def assert_done(scene) -> None:
    """POSTINIT gate: a plan still pending means no group was written and robot <-> robot-off is on."""
    todo = getattr(scene, PLAN_ATTR, None)
    if todo:
        raise RuntimeError(
            f"{MARK} {len(todo)} envs still wait for their robot-off groups: filter_collisions never authored them"
        )


def refuse_filtered(env_root: str, kind: str, pairs) -> None:
    """Fail when the stage reader found FilteredPairs between robot bodies and robot-off bodies of `kind` (`pairs` =
    what it measured on that env)."""
    if pairs:
        raise RuntimeError(
            f"{MARK} {env_root}: robot <-> {kind} cut by FilteredPairs ({len(pairs)}, e.g. "
            f"{sorted(map(str, pairs))[:2]}) — robot-off is these collision groups only"
        )


def _env_groups(stage) -> dict:
    """{env root: env group path} for the cloner's groups (`group<n>` with exactly one include)."""
    root = stage.GetPrimAtPath(ROOT)
    if not root.IsValid() or not root.IsDefined():
        raise RuntimeError(f"{MARK} {ROOT} is not defined: called before Isaac Lab's filter_collisions")
    out = {}
    for g in root.GetChildren():
        n = g.GetName()
        if not (n.startswith("group") and n[5:].isdigit()):
            continue
        rel = g.GetRelationship(INC)
        tg = rel.GetTargets() if rel else []
        if len(tg) != 1:
            raise RuntimeError(f"{MARK} env group {g.GetPath()} includes {len(tg)} paths, not the cloner's one")
        out[tg[0].pathString] = g.GetPath().pathString
    return out


def _rows(stage, todo: list) -> list:
    """[(env root, env group, number, robot roots, robot-off roots)] for authoring and read-back."""
    env2g = _env_groups(stage)
    rows = []
    for r, offs in todo:
        g = env2g.get(r)
        if g is None:
            raise RuntimeError(f"{MARK} {r} has no env group under {ROOT}")
        robots = [f"{r}/{leaf}" for leaf in ROBOT_LEAVES if stage.GetPrimAtPath(f"{r}/{leaf}").IsValid()]
        if not robots or not offs:
            raise RuntimeError(f"{MARK} {r}: robot roots {robots} · robot-off roots {offs}; no pair to cut")
        rows.append((r, g, g.rsplit("group", 1)[-1], robots, list(offs)))
    return rows


def _group_spec(lay, path: str, members: list, partners: list):
    """One PhysicsCollisionGroup prim spec with its members and the groups it collides with."""
    from pxr import Sdf, Usd

    ps = Sdf.PrimSpec(lay.GetPrimAtPath(ROOT), path.rsplit("/", 1)[-1], Sdf.SpecifierDef, "PhysicsCollisionGroup")
    ps.SetInfo(Usd.Tokens.apiSchemas, Sdf.TokenListOp.Create({"CollectionAPI:colliders"}))
    er = Sdf.AttributeSpec(ps, "collection:colliders:expansionRule", Sdf.ValueTypeNames.Token, Sdf.VariabilityUniform)
    er.default = "expandPrims"
    inc = Sdf.RelationshipSpec(ps, INC, False)
    for p in members:
        inc.targetPathList.Append(p)
    filt = Sdf.RelationshipSpec(ps, FILT, False)
    for p in partners:
        filt.targetPathList.Append(p)


def author(stage, todo: list) -> dict:
    """todo = [(env root, [robot-off roots])] -> two groups per env, read back.
    Returns {groups, envs, roots, global}."""
    from pxr import Sdf

    rows = _rows(stage, todo)
    glob = f"{ROOT}/global_group"
    has_glob = stage.GetPrimAtPath(glob).IsValid()
    lay = stage.GetRootLayer()
    for _r, _g, j, _robots, _offs in rows:
        for nm in (f"{ROBOT_GROUP}{j}", f"{OFF_GROUP}{j}"):
            if stage.GetPrimAtPath(f"{ROOT}/{nm}").IsValid():
                raise RuntimeError(f"{MARK} {ROOT}/{nm} already exists: authored twice")
    with Sdf.ChangeBlock():
        for _r, g, j, robots, offs in rows:
            gs = lay.GetPrimAtPath(g)
            exc = gs.relationships.get(EXC) or Sdf.RelationshipSpec(gs, EXC, False)
            for name, members in ((f"{ROBOT_GROUP}{j}", robots), (f"{OFF_GROUP}{j}", offs)):
                path = f"{ROOT}/{name}"
                _group_spec(lay, path, members, [path, g, *([glob] if has_glob else [])])
                for p in members:
                    exc.targetPathList.Append(p)
                gs.relationships[FILT].targetPathList.Append(path)
                if has_glob:
                    lay.GetPrimAtPath(glob).relationships[FILT].targetPathList.Append(path)
    readback(stage, rows)
    info = {"groups": 2 * len(rows), "envs": len(rows), "roots": sum(len(x[4]) for x in rows), "global": has_glob}
    print(
        f"{MARK} robot <-> robot-off = collision groups {info['groups']} (envs {info['envs']} x 2 · robot-off roots "
        f"{info['roots']} · out of the env group · only robot <-> robot-off blocked · global group "
        f"{'yes' if has_glob else 'no'} · read-back ok)",
        flush=True,
    )
    return info


def verify(stage, todo: list) -> None:
    """Read back an authored stage (the same check `author` runs)."""
    readback(stage, _rows(stage, todo))


def _bodies(stage, root: str) -> list:
    from pxr import Usd, UsdPhysics

    return [
        q.GetPath().pathString for q in Usd.PrimRange(stage.GetPrimAtPath(root)) if q.HasAPI(UsdPhysics.RigidBodyAPI)
    ]


def readback(stage, rows) -> None:
    """Every authored env: robot, robot-off and remaining bodies each in exactly the expected group; only
    robot <-> robot-off blocked."""
    ix = GroupIndex(stage)
    bad = []
    for r, g, j, robots, offs in rows:
        rg, og = f"{ROOT}/{ROBOT_GROUP}{j}", f"{ROOT}/{OFF_GROUP}{j}"
        rb = [b for x in robots for b in _bodies(stage, x)]
        ob = [b for x in offs for b in _bodies(stage, x)]
        mine = set(rb) | set(ob)
        rest = [b for b in _bodies(stage, r) if b not in mine]
        for what, bodies, want in (("robot", rb, rg), ("robot-off", ob, og), ("env", rest, g)):
            wrong = [(b, sorted(v)) for b in bodies if (v := ix.groups_of(b)) != {want}]
            if not bodies or wrong:
                bad.append(f"{r} {what} bodies {len(bodies)} · not in {want} alone: {wrong[:2]}")
        ok = not ix.allow(rg, og) and ix.allow(rg, g) and ix.allow(og, g) and ix.allow(rg, rg) and ix.allow(og, og)
        if not ok:
            bad.append(
                f"{r} robot<->robot-off {ix.allow(rg, og)} (must be False) · robot<->env {ix.allow(rg, g)} · "
                f"robot-off<->env {ix.allow(og, g)}"
            )
    if bad:
        raise RuntimeError(f"{MARK} read-back: {len(bad)} mismatches: " + " · ".join(bad[:3]))


class GroupIndex:
    """The stage's collision groups: which groups hold a path (collection includes/excludes with expandPrims, the
    deepest rule decides) and whether two groups collide (inverted filter: listed in `filteredGroups` = collide)."""

    def __init__(self, stage):
        from pxr import Usd

        self.rules: dict = {}  # rule prim path -> [(group, included?)]
        self.filt: dict = {}  # group -> its filteredGroups targets
        self.invert = False
        world = stage.GetPrimAtPath("/World")
        tops = [*stage.GetPseudoRoot().GetChildren(), *(world.GetChildren() if world.IsValid() else [])]
        for p in tops:  # the physics scene sits at the top (Isaac Lab /physicsScene)
            if p.GetTypeName() == "PhysicsScene":
                a = p.GetAttribute("physxScene:invertCollisionGroupFilter")
                self.invert = self.invert or bool(a and a.Get())
        root = stage.GetPrimAtPath(ROOT)
        for g in Usd.PrimRange(root) if root.IsValid() else []:
            if g.GetTypeName() != "PhysicsCollisionGroup":
                continue
            gp = g.GetPath().pathString
            for rel, inc in ((INC, True), (EXC, False)):
                r = g.GetRelationship(rel)
                for t in r.GetTargets() if r else []:
                    self.rules.setdefault(t.pathString, []).append((gp, inc))
            r = g.GetRelationship(FILT)
            self.filt[gp] = {t.pathString for t in (r.GetTargets() if r else [])}

    def groups_of(self, path: str) -> set:
        decided: dict = {}
        p = path
        while p and p != "/":
            for g, inc in self.rules.get(p, ()):
                decided.setdefault(g, inc)
            p = p.rsplit("/", 1)[0]
        return {g for g, inc in decided.items() if inc}

    def allow(self, ga: str, gb: str) -> bool:
        listed = gb in self.filt.get(ga, ()) or ga in self.filt.get(gb, ())
        return listed if self.invert else not listed

    def blocked(self, a: str, b: str) -> bool:
        """Whether the group filter cuts two body paths (a side in no group is outside the filter: False)."""
        ga, gb = self.groups_of(a), self.groups_of(b)
        if not ga or not gb:
            return False
        return not any(self.allow(x, y) for x in ga for y in gb)
