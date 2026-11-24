"""Collision truth table: reads **the real stage** and decides per env whether R1–R8 are on.

Why: a code-level check of exactly which collisions are on and off. The old banner
([COLL_PRESET]) **derived the matrix from environment variables**, i.e. it printed the intent. But FilteredPairs
authoring had a hole that reached env_0 only, so intent and reality differed.
This module does not look at the intent. It walks each env's prims directly and counts
  (1) the articulation structure (whether the split articulation /RobotL exists) and
  `physxArticulation:enabledSelfCollisions`
  (2) the (source class, target class) of every `FilteredPairsAPI` rel, and the pairs the collision groups cut
  (`collision_groups.GroupIndex`, robot <-> robot-off bodies)
then applies the PhysX rules (pairs inside one articulation follow the self-collision flag; pairs between
articulations or rigid bodies collide unless filtered) to give the **effective ON/OFF**. env_0, env_1 and the last
env are compared, and differences between envs are reported loudly.

Rows (the collision table as specified):
  R1 hand internal · arm internal · hand<->same arm | R2 right hand·arm <-> left hand·arm | R3 arm·hand <-> torso | R4
  robot <-> desk
  R5 hand <-> own object | R6 hand·arm <-> other object | R7 object <-> object | R8 object <-> desk
  (+) RX arm <-> own object: not in the table, but ARM_OBJ_NOCOLL touches it, so it is reported too.
  (+) RD robot <-> support disc (rule D40: discs support objects only; a scene without discs is '—'). A row
      independent of presets, so `collision_presets.expected_disc` sets the expectation (statements = the cfg's disc
      slot count and the stamp collide); OFF only when **every pair** (robot body, disc body) is cut by the collision
      groups, otherwise 'partial(filter k/N)'; a FilteredPairs robot <-> disc fails
      (`collision_groups.refuse_filtered`).
  (+) RO support disc <-> object (rule D11″ (d): a disc supports its own main object only). Expected pairs = the real
      disc bodies of the env clip (static assignment i % clip count) × other object bodies (owner side statement
      `cfg.support_disc_owner`). Equal is 'OWN', 0 is 'ON', own-object pairs or unexpected pairs filtered are
      'excess', some are 'partial'. Expectation `collision_presets.expected_disc_obj`.
  (+) RS.<partner> support object <-> hand·arm·object·desk (rule D42): one row per partner, expectation
      `collision_presets.expected_support` (0 support slots = '—').
      OFF only when the (support body, partner) pair is filtered for every support body; some is 'partial'.
  (+) RT object <-> torso (OBJ_TORSO_COLLIDE: ON when there are 0 filters).
  (+) RE robot <-> extra slot · OE main object <-> extra slot: filters and collision groups read together and
      compared with the statement (`extra_slot_audit`).
"""

from __future__ import annotations

from collections import Counter

from pxr import Usd, UsdPhysics

from dexengine.rl.tasks.scene import extra_slot_audit as _xa
from dexengine.rl.tasks.scene.collision_bodies import (
    classify_body,
    object_prim_paths,
    support_disc_prim_paths,
    support_object_prim_paths,
)
from dexengine.rl.tasks.scene.collision_groups import refuse_filtered
from dexengine.rl.tasks.scene.collision_presets import (
    OBJ_TORSO_ROW,
    SUPPORT_COLLIDE,
    SUPPORT_ROW,
)

_ROBOT_CLASSES = ("RH", "RA", "LH", "LA", "OTHER")
_RIGHT = ("RH", "RA")
_LEFT = ("LH", "LA")
_SUP_PARTNER = {
    "RH": "hand",
    "LH": "hand",
    "RA": "arm",
    "LA": "arm",
    "OBJ_R": "objects",
    "OBJ_L": "objects",
    "DESK": "desk",
}
ROWS = (
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6",
    "R7",
    "R8",
    "RX",
    "RD",
    "RO",
    *(f"{SUPPORT_ROW}.{k}" for k in SUPPORT_COLLIDE),
    OBJ_TORSO_ROW,
)


def _pair_row(a: str, b: str) -> str | None:
    s = {a, b}
    if "SUP" in s:
        other = (s - {"SUP"}).pop() if len(s) == 2 else None
        return f"{SUPPORT_ROW}.{_SUP_PARTNER[other]}" if other in _SUP_PARTNER else None
    if "DISC" in s:
        other = (s - {"DISC"}).pop() if len(s) == 2 else None
        if other in _ROBOT_CLASSES:
            return "RD"
        return "RO" if other in ("OBJ_R", "OBJ_L") else None
    if a in _ROBOT_CLASSES and b in _ROBOT_CLASSES:
        if "OTHER" in s:
            return "R3" if s != {"OTHER"} else None
        side_a = "R" if a in _RIGHT else "L"
        side_b = "R" if b in _RIGHT else "L"
        return "R1" if side_a == side_b else "R2"
    if "DESK" in s:
        other = (s - {"DESK"}).pop() if len(s) == 2 else None
        if other in _ROBOT_CLASSES:
            return "R4"
        if other in ("OBJ_R", "OBJ_L"):
            return "R8"
        return None
    if s == {"OBJ_R", "OBJ_L"}:
        return "R7"
    if len(s) == 2 and any(x.startswith("OBJ_") for x in s):
        obj = [x for x in s if x.startswith("OBJ_")][0]
        rob = (s - {obj}).pop()
        if rob == "OTHER":
            return OBJ_TORSO_ROW
        if rob not in _ROBOT_CLASSES:
            return None
        same_side = (obj == "OBJ_R" and rob in _RIGHT) or (obj == "OBJ_L" and rob in _LEFT)
        is_hand = rob in ("RH", "LH")
        if same_side:
            return "R5" if is_hand else "RX"
        return "R6h" if is_hand else "R6a"
    return None


def _classify_env(stage, scene, env_root: str) -> dict[str, str]:
    """Prim path -> class. Robot links by name, objects and the desk by path."""
    cls: dict[str, str] = {}
    _classify_robot_links(stage, env_root, cls)
    try:
        for side, path in object_prim_paths(scene, env_root).items():
            cls[path] = "OBJ_R" if side == "right" else "OBJ_L"
            # object bodies may be child prims: children get the same class
            op = stage.GetPrimAtPath(path)
            if op.IsValid():
                for q in Usd.PrimRange(op):
                    cls.setdefault(q.GetPath().pathString, cls[path])
    except Exception as e:  # noqa: BLE001 — a scene without objects (uni, ...)
        print(f"[COLL_AUDIT] object_prim_paths failed ({e}) — the object rows cannot be judged", flush=True)
    desk = stage.GetPrimAtPath(f"{env_root}/DeskCollisionTop")
    if desk.IsValid():
        for q in Usd.PrimRange(desk):
            cls[q.GetPath().pathString] = "DESK"
    for c, paths in (
        ("DISC", support_disc_prim_paths(scene, env_root)),
        ("SUP", support_object_prim_paths(scene, env_root)),
    ):
        for d in paths:
            dp = stage.GetPrimAtPath(d)
            if dp.IsValid():
                for q in Usd.PrimRange(dp):
                    cls[q.GetPath().pathString] = c
    return cls


def _classify_robot_links(stage, env_root, cls):
    """Robot links of both articulations, classified by body name."""
    for leaf in ("Robot", "RobotL"):
        root = stage.GetPrimAtPath(f"{env_root}/{leaf}")
        if not root.IsValid():
            continue
        for p in Usd.PrimRange(root):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                c = classify_body(p.GetName())
                # The left arm tree of the split articulation is all left side (under RobotL forced to L). Left
                # mount/root
                # links classified OTHER by name (link_left_mount, ...) are also on the left arm side, not the torso ->
                # LA.
                if leaf == "RobotL":
                    c = "LH" if c in ("RH", "LH") else "LA"
                cls[p.GetPath().pathString] = c


def _lookup(cls: dict[str, str], path: str) -> str:
    """Exact match, then upwards through the ancestors (when the rel target is a child prim)."""
    p = path
    while p and p != "/":
        if p in cls:
            return cls[p]
        p = p.rsplit("/", 1)[0]
    return "?"


def audit_env(scene, env_root: str, owner_row=None, shared: bool = False, idle: str = "", extra_kinds=None) -> dict:
    """The real stage truth table of one env. `owner_row` = the disc owner statement of the env clip (RO row
    expectation; without it RO is 'unstated') · `extra_kinds` = the extra slot kind statement of the env clip (RE · OE,
    extra_slot_audit)."""
    stage = scene.stage
    cls = _classify_env(stage, scene, env_root)
    split = stage.GetPrimAtPath(f"{env_root}/RobotL").IsValid()
    (
        both,
        disc_bodies,
        disc_slots,
        filt,
        inter,
        intra,
        obj_bodies,
        rd_pairs,
        ro_pairs,
        robot_bodies,
        rs_pairs,
        sc_l,
        sc_r,
        selfcoll,
        sup_bodies,
        unknown,
    ) = _collect_collision_state(scene, env_root, cls, stage)

    return _effective_rows(
        scene,
        env_root,
        owner_row,
        shared,
        idle,
        extra_kinds,
        both,
        cls,
        disc_bodies,
        disc_slots,
        filt,
        inter,
        intra,
        obj_bodies,
        rd_pairs,
        ro_pairs,
        robot_bodies,
        rs_pairs,
        sc_l,
        sc_r,
        selfcoll,
        split,
        stage,
        sup_bodies,
        unknown,
    )


def _collect_collision_state(scene, env_root, cls, stage):
    """Self-collision flags, filtered pairs and body classes of this env."""
    selfcoll = {}
    for leaf in ("Robot", "RobotL"):
        root = stage.GetPrimAtPath(f"{env_root}/{leaf}")
        if not root.IsValid():
            continue
        flag = None
        for p in Usd.PrimRange(root):
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                a = p.GetAttribute("physxArticulation:enabledSelfCollisions")
                flag = bool(a.Get()) if a and a.HasAuthoredValue() else False
                break
        selfcoll[leaf] = flag
    filt = Counter()
    unknown = 0
    # RD coverage: unique (robot body, disc body) pairs; reading OFF from a single filtered pair would miss partial
    # coverage
    robot_bodies = {k for k, v in cls.items() if v in _ROBOT_CLASSES}
    disc_slots = support_disc_prim_paths(scene, env_root)  # statement (the cfg that makes them)
    disc_bodies = set()
    for d in disc_slots:
        dp = stage.GetPrimAtPath(d)
        if dp.IsValid():
            disc_bodies |= {q.GetPath().pathString for q in Usd.PrimRange(dp) if q.HasAPI(UsdPhysics.RigidBodyAPI)}
    rd_pairs = set()
    ro_pairs = set()
    sup_bodies = {k for k, v in cls.items() if v == "SUP" and stage.GetPrimAtPath(k).HasAPI(UsdPhysics.RigidBodyAPI)}
    rs_pairs = Counter()
    obj_bodies = {
        k for k, v in cls.items() if v in ("OBJ_R", "OBJ_L") and stage.GetPrimAtPath(k).HasAPI(UsdPhysics.RigidBodyAPI)
    }
    env_prim = stage.GetPrimAtPath(env_root)
    for p in Usd.PrimRange(env_prim):
        if not p.HasAPI(UsdPhysics.FilteredPairsAPI):
            continue
        rel = UsdPhysics.FilteredPairsAPI(p).GetFilteredPairsRel()
        if not rel:
            continue
        a = _lookup(cls, p.GetPath().pathString)
        for t in rel.GetTargets():
            b, row, unknown = _classify_pair(
                a, cls, disc_bodies, filt, obj_bodies, p, rd_pairs, ro_pairs, robot_bodies, rs_pairs, t, unknown
            )
    refuse_filtered(env_root, "support disc", rd_pairs)
    filt["RD"] += _group_cut_rd(stage, robot_bodies, disc_bodies, rd_pairs)

    sc_r = selfcoll.get("Robot", False)
    sc_l = selfcoll.get("RobotL", sc_r)

    def intra(flag, row):
        if not flag:
            return "OFF(self-collision flag)"
        return f"OFF(filter {filt[row]})" if filt[row] else "ON"

    def inter(row):
        return f"OFF(filter {filt[row]})" if filt[row] else "ON"

    def both(la, a, lb, b):
        return a if a == b else f"{la}:{a} {lb}:{b}"

    return (
        both,
        disc_bodies,
        disc_slots,
        filt,
        inter,
        intra,
        obj_bodies,
        rd_pairs,
        ro_pairs,
        robot_bodies,
        rs_pairs,
        sc_l,
        sc_r,
        selfcoll,
        sup_bodies,
        unknown,
    )


def _group_cut_rd(stage, robot_bodies: set, disc_bodies: set, rd_pairs: set) -> int:
    """Add the (robot body, disc body) pairs the stage's collision groups cut to the RD coverage; returns how many."""
    if not (robot_bodies and disc_bodies):
        return 0
    from dexengine.rl.tasks.scene.collision_groups import GroupIndex

    ix = GroupIndex(stage)
    cut = {frozenset((r, d)) for r in robot_bodies for d in disc_bodies if ix.blocked(r, d)} - rd_pairs
    rd_pairs |= cut
    return len(cut)


def _classify_pair(a, cls, disc_bodies, filt, obj_bodies, p, rd_pairs, ro_pairs, robot_bodies, rs_pairs, t, unknown):
    """Class pair and truth-table row of one filtered pair."""
    b = _lookup(cls, t.pathString)
    row = _pair_row(a, b)
    if row is None:
        unknown += 1
    else:
        filt[row] += 1
        if row == "RD":
            ends = (
                _anc(p.GetPath().pathString, robot_bodies | disc_bodies),
                _anc(t.pathString, robot_bodies | disc_bodies),
            )
            if None not in ends:
                rd_pairs.add(frozenset(ends))
        if row.startswith(SUPPORT_ROW + "."):
            rs_pairs[row] += 1
        if row == "RO":
            ro_pairs.add(
                (
                    _anc(p.GetPath().pathString, disc_bodies) or p.GetPath().pathString,
                    _anc(t.pathString, obj_bodies) or t.pathString,
                )
            )
    return b, row, unknown


def _anc(path: str, bodies: set[str]) -> str | None:
    """The path itself or its nearest ancestor in the body set (when the rel target is a child prim of a body)."""
    p = path
    while p and p != "/":
        if p in bodies:
            return p
        p = p.rsplit("/", 1)[0]
    return None


# collision_presets.py owns the preset expectation tables; they are not restated here.
from dexengine.rl.tasks.scene.collision_preset_check import norm_one
from dexengine.rl.tasks.scene.collision_presets import (
    DISC_OBJ_ROW,
)
from dexengine.rl.tasks.scene.collision_rows import _effective_rows


def audit_and_print(
    scene, n_check: int = 2, disc_owner=None, shared: bool = False, idle: str = "", extra_kinds=None
) -> list[dict]:
    """Audit env_0, env_1 (when present) and the last env and print the table. Differences between envs are marked ★.
    `extra_kinds` = the env cfg's extra slot kind statement `extra_slot_kinds` ([clip][slot]) for the RE · OE comparison
    (extra_slot_audit.check).
    `disc_owner` = the env cfg's disc owner statement `support_disc_owner` ([clip][slot]); env i's clip = i % clip
    count.
    The RO row has a different filter count per env (different disc counts), so ★ compares values without the evidence
    parentheses."""
    roots = list(getattr(scene, "env_prim_paths", []) or [])
    if not roots:
        return []
    idx = sorted({0, min(1, len(roots) - 1), len(roots) - 1})
    res = [
        audit_env(
            scene,
            roots[i],
            disc_owner[i % len(disc_owner)] if disc_owner else None,
            shared,
            idle,
            extra_kinds[i % len(extra_kinds)] if extra_kinds else None,
        )
        for i in idx
    ]
    print(
        f"[COLL_AUDIT] real stage truth table (num_envs={len(roots)}, "
        f"split={res[0]['split']}, selfcoll={res[0]['selfcoll']}, unclassified targets={res[0]['unknown_targets']})",
        flush=True,
    )
    head = "  row   " + "".join(f"{('env_' + str(i)):>26s}" for i in idx)
    print(head, flush=True)
    diff_rows = []
    for row in ROWS:
        vals = [r["effective"][row] for r in res]
        mark = "★" if len({norm_one(v) if row == DISC_OBJ_ROW else v for v in vals}) > 1 else " "
        if mark == "★":
            diff_rows.append(row)
        print(f"  {row:<4s}{mark}  " + "".join(f"{v:>26s}" for v in vals), flush=True)
    if diff_rows:
        print(
            f"[COLL_AUDIT] ★ collision rules differ between envs: {diff_rows} — env_0 authoring did not reach the "
            f"clones. "
            f"Training, judge and render must not be compared in this state.",
            flush=True,
        )
    _xa.check(res, idx, extra_kinds)  # extra slots RE · OE: nothing happens without slots
    return res
