"""Effective ON/OFF of each collision truth-table row, from the measured filter pairs and self-collision state."""

from __future__ import annotations

from pxr import Usd, UsdPhysics

from dexengine.rl.tasks.scene import extra_slot_audit as _xa
from dexengine.rl.tasks.scene.collision_bodies import (
    object_prim_paths,
    support_disc_slot_paths,
    support_object_prim_paths,
)
from dexengine.rl.tasks.scene.collision_pairs import disc_own_object_path
from dexengine.rl.tasks.scene.collision_preset_check import VACUOUS_TORSO
from dexengine.rl.tasks.scene.collision_presets import OBJ_TORSO_ROW, SUPPORT_COLLIDE, SUPPORT_ROW


def _disc_obj_pairs(stage, scene, env_root: str, owner_row, shared: bool, idle: str) -> tuple[set, set, int] | None:
    """RO expectation: (expected (disc body, other object body) pairs, own-object pairs, scene object count). None
    without an owner statement."""
    objs = object_prim_paths(scene, env_root)
    n_obj = len(set(objs.values()))
    if owner_row is None:
        return None if n_obj >= 2 else (set(), set(), n_obj)

    def bodies(root):
        p = stage.GetPrimAtPath(root)
        if not p.IsValid():
            return []
        return [q.GetPath().pathString for q in Usd.PrimRange(p) if q.HasAPI(UsdPhysics.RigidBodyAPI)]

    ob = {p: bodies(p) for p in set(objs.values())}
    want, own = set(), set()
    for k, d in support_disc_slot_paths(scene, env_root).items():
        if k >= len(owner_row) or owner_row[k] in (None, "static"):  # a fixed object's pedestal: no main-object owner
            continue
        mine = disc_own_object_path(objs, owner_row[k], shared, idle)
        for db in bodies(d):
            want |= {(db, b) for p, bs in ob.items() if p != mine for b in bs}
            own |= {(db, b) for b in ob[mine]}
    return want, own, n_obj


def _effective_rows(
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
):
    """Effective ON/OFF per truth-table row from the collected state."""

    def disc():
        if not disc_bodies:
            return "—(no disc)"
        n_all = len(robot_bodies) * len(disc_bodies)
        k = len(rd_pairs)
        if k == 0:
            return "ON"
        return f"OFF(filter {k})" if k >= n_all else f"partial(filter {k}/{n_all})"

    eff = {
        "R1": both("right", intra(sc_r, "R1"), "left", intra(sc_l, "R1")) if split else intra(sc_r, "R1"),
        "R2": inter("R2") if split else intra(sc_r, "R2"),
        "R3": _r3_vacuous(cls)
        or (both("right", intra(sc_r, "R3"), "left", inter("R3")) if split else intra(sc_r, "R3")),
        "R4": inter("R4"),
        "R5": inter("R5"),
        "R6": both("hand", inter("R6h"), "arm", inter("R6a")),
        "R7": inter("R7"),
        "R8": inter("R8"),
        "RX": inter("RX"),
        "RD": disc(),
        "RO": _ro_value(disc_bodies, ro_pairs, _disc_obj_pairs(stage, scene, env_root, owner_row, shared, idle)),
        OBJ_TORSO_ROW: inter(OBJ_TORSO_ROW),
        **{f"{SUPPORT_ROW}.{k}": _rs_value(len(sup_bodies), rs_pairs[f"{SUPPORT_ROW}.{k}"]) for k in SUPPORT_COLLIDE},
    }
    return {
        "env": env_root,
        "split": split,
        "selfcoll": selfcoll,
        "disc_slots": len(disc_slots),
        **_xa.fields(scene, env_root, extra_kinds, robot_bodies, obj_bodies),
        "support_slots": len(support_object_prim_paths(scene, env_root)),
        "objects": len(set(object_prim_paths(scene, env_root).values())),
        "filtered": dict(filt),
        "unknown_targets": unknown,
        "effective": eff,
    }


def _r3_vacuous(cls: dict[str, str]) -> str:
    """R3 (arm·hand <-> torso) targets torso-class (OTHER) bodies. With a body count of 0 (floating hand: only the two
    hand
    articulations) the row has no pairs and gives VACUOUS_TORSO, otherwise '' (the audit measures by filters); 0 filters
    is never read as 'ON' (the same rule as the '—' of RD and RS)."""
    return "" if any(v == "OTHER" for v in cls.values()) else VACUOUS_TORSO


def _rs_value(n_sup: int, k: int) -> str:
    """Effective RS.<partner> value: k filter pairs with the partner among n support bodies (k = n is all when the
    partner is one body, the desk)."""
    if not n_sup:
        return "—(no support)"
    return "ON" if k == 0 else (f"OFF(filter {k})" if k >= n_sup else f"partial(filter {k}/{n_sup})")


def _ro_value(disc_bodies: set, got: set, exp) -> str:
    """Effective RO value: the measured (disc, object) filter pairs `got` against the stated expectation `exp`=(want,
    own, object count)."""
    if not disc_bodies:
        return "—(no disc)"
    if exp is None:
        return f"unstated(filter {len(got)})"
    want, own, n_obj = exp
    if n_obj < 2:
        return f"excess(filter {len(got)})" if got else "—(no other object)"
    extra = got - want
    if extra:
        return f"excess(filter {len(extra)} · own object {len(got & own)})"
    if got == want:
        return f"OWN(filter {len(got)})"
    return "ON" if not got else f"partial(filter {len(got)}/{len(want)})"
