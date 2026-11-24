"""Record stage-1 column list (called by the rollout.recorder recorder per rollout). Field names, meanings and markers
are owned by the rollout.recorder module doc (action, joint_target, force_total, force_object, partner decomposition,
contact_sep).
channels(env, ents) -> (column list, oracle sensors, partner decomposition reader); column = dict(name, kind, width,
names, read) · read() -> [NE, width] float32 (GPU). The action column also carries dim_names (a name per dimension,
action_dim_names) and term_modes (class and reference per term, term_mode).
"""

from __future__ import annotations

from dexengine.record.rollout.recorder import (
    KIND_OBJ_PARTNER,
    KIND_OBJECT,
    KIND_PARTNER,
    KIND_SEP,
    KIND_TOTAL,
    MARK,
    object_force_rows,
)

#: leading dimensions of the wrist IK action term (ArmIKFingersPositionAction and its wrapper ResidualArmIKAction): that
#: term's action_dim statement (action[0:3] = wrist position Δ · [3:6] = wrist rotation Δ intrinsic XYZ Euler · [6:] =
#: fingers in _finger_joint_names order). The term emits no dimension names, so the slot names live here (owner = that
#: term; remove this table once the term emits names) · the width is checked by the oracle.
WRIST_SLOTS = ("wrist_pos_x", "wrist_pos_y", "wrist_pos_z", "wrist_rot_x", "wrist_rot_y", "wrist_rot_z")


def term_dim_names(name: str, term, dim: int) -> list[str]:
    """Dimension names "<term>.<dim>" of one action term; source = the joint order that term emits: WRIST_SLOTS before
    the finger joints _finger_joint_names (find_joints preserve_order: policy order; for a residual term the inner
    term _inner's) · joint terms (_joint_names, IsaacLab JointAction kinds) use the joints as they are. A width
    different from dim or no source dies (no unnamed dimensions left silently)."""
    src = getattr(term, "_inner", None) or term
    fingers, joints = getattr(src, "_finger_joint_names", None), getattr(term, "_joint_names", None)
    if fingers is not None:
        out = [*WRIST_SLOTS, *map(str, fingers)]
    elif joints is not None:
        out = list(map(str, joints))
    else:
        raise RuntimeError(
            f"{MARK} action term {name} ({type(term).__name__}) has no dimension name source "
            "(_finger_joint_names, _joint_names)"
        )
    if len(out) != dim:
        raise RuntimeError(
            f"{MARK} action term {name}: names {len(out)} != dims {dim}: the source is not this term's order"
        )
    return [f"{name}.{d}" for d in out]


def term_mode(term) -> dict:
    """One action term's statement: class name, reference base (_reference_base), trajectory reference
    (_use_traj_ref). The store checks with it that a one-hand idle-arm hold command (residual 0) is a matching record
    (store.form.check_hold_term; no statement = an old record)."""
    return {
        "cls": type(term).__name__,
        "reference_base": bool(getattr(term, "_reference_base", False)),
        "traj_ref": bool(getattr(term, "_use_traj_ref", False)),
    }


def action_dim_names(am) -> list[str]:
    """Dimension names of the whole policy action (action column): term_dim_names chained in term (active_terms) order
    (width = total dimension)."""
    out = [
        x
        for t, d in zip(am.active_terms, am.action_term_dim, strict=True)
        for x in term_dim_names(t, am.get_term(t), int(d))
    ]
    if len(out) != int(am.total_action_dim):
        raise RuntimeError(f"{MARK} action names {len(out)} != total_action_dim {am.total_action_dim}")
    return out


def entity_key(scene, obj) -> str:
    for tbl in (scene.rigid_objects, scene.articulations):
        for k, v in tbl.items():
            if v is obj:
                return k
    raise RuntimeError(f"{MARK} scene key of the object asset not found")


def object_channels(env) -> list[dict]:
    """force_object channel per object: hand link names, sides (two for a shared object), object body glob (for the
    oracle check)."""
    from dexengine.rl.tasks.mdp import hand_object_contact as HOC

    hoc, n_env = HOC.instance(env), int(env.num_envs)
    names = hoc.link_names()
    if not names:
        raise RuntimeError(f"{MARK} no robot body matches the hand link rule: object forces cannot be recorded")
    by_obj: dict = {}
    for side, obj in env.held_object_assets():
        by_obj.setdefault(id(obj), (obj, []))[1].append(side)
    if not by_obj:
        raise RuntimeError(f"{MARK} no held object (held_object_assets): object forces cannot be recorded")
    out = []
    for obj, sides in by_obj.values():
        key = entity_key(env.scene, obj)
        out.append(
            {
                "name": f"{KIND_OBJECT}/{key}",
                "kind": KIND_OBJECT,
                "width": len(names) * 3,
                "names": list(names),
                "history": 1,
                "sides": sides,
                "object_bodies": HOC.object_globs(obj),
                "obj": obj,
                "read": lambda obj=obj: object_force_rows(hoc.link_forces(obj), n_env),
            }
        )
    return out


def extra_slot_sensors(env) -> set[str]:
    """Scene key set of the extra slot contact sensors; owner dexengine.rl.tasks.scene.extra_slots (key = SENSOR_PREFIX
    + slot index, slot count = slots_of). Scene sensors with that prefix that differ from the stated slot count die ·
    one marker line when any are left out."""
    from dexengine.rl.tasks.scene import extra_slots as XE

    want = {f"{XE.SENSOR_PREFIX}{k}" for k in range(XE.slots_of(env))}
    got = {str(k) for k in env.scene.sensors if str(k).startswith(XE.SENSOR_PREFIX)}
    if got != want:
        raise RuntimeError(f"{MARK} extra slot sensors {sorted(got)} != statement K={len(want)} {sorted(want)}")
    if got:
        print(
            f"{MARK} extra slot sensors {len(got)} left out of force_total (not hand sensors · {sorted(got)}): hand "
            f"<-> extra forces are in force_partner/<sensor>.{XE.PREFIX}<k> (partner decomposition on)",
            flush=True,
        )
    return got


def _policy_channels(out, am, ents, scene):
    """The policy action channel and one joint-target channel per robot articulation."""
    from isaaclab.assets import Articulation

    out.append(
        {
            "name": "action",
            "kind": "action",
            "width": int(am.total_action_dim),
            "names": list(am.active_terms),
            "term_dims": [int(d) for d in am.action_term_dim],
            "dim_names": action_dim_names(am),
            "term_modes": [term_mode(am.get_term(t)) for t in am.active_terms],
            "read": lambda: am.action.float(),
        }
    )
    for e in ents:
        k = e["name"]
        a = scene.articulations[k] if e["kind"] == "artic" else None
        if k.startswith("object") or not isinstance(a, Articulation):
            continue
        out.append(
            {
                "name": f"joint_target/{k}",
                "kind": "joint_target",
                "width": len(a.joint_names),
                "names": list(a.joint_names),
                "read": lambda a=a: a.data.joint_pos_target.float(),
            }
        )


def _partner_channels(partner, out, n_env):
    """Partner channels: per link and partner class, per partner object (by class) and its link separations."""
    for d in partner.links:
        for ci, cl in enumerate(d["classes"]):
            out.append(
                {
                    "name": f"{KIND_PARTNER}/{d['key']}.{cl}",
                    "kind": KIND_PARTNER,
                    "width": len(d["names"]) * 3,
                    "names": list(d["names"]),
                    "history": 1,
                    "partner": cl,
                    "side": d["side"],
                    "read": lambda k=d["key"], ci=ci: (
                        partner.cache[("link", k)][:, :, ci, :].reshape(n_env, -1).float()
                    ),
                }
            )
    for o in partner.objs:
        out.append(
            {
                "name": f"{KIND_OBJ_PARTNER}/{o['key']}",
                "kind": KIND_OBJ_PARTNER,
                "width": len(o["classes"]) * 3,
                "names": list(o["classes"]),
                "history": 1,
                "read": lambda k=o["key"]: partner.cache[("obj", k)].reshape(n_env, -1).float(),
            }
        )
        out.append(
            {
                "name": f"{KIND_SEP}/{o['key']}",
                "kind": KIND_SEP,
                "width": len(partner.link_names),
                "names": list(partner.link_names),
                "history": 1,
                "read": lambda k=o["key"]: partner.cache[("sep", k)].reshape(n_env, -1).float(),
            }
        )


def channels(env, ents: list[dict]) -> tuple[list[dict], list]:
    """Record column list: dict(name, kind, width, names, read) · read() -> [NE, width] float32 (GPU) · second = oracle
    sensors [(sensor name, sensor, object channel name, link column)].
    The joint target articulations = the canonical recorder's entities (traj_record.scene_entities; alias keys of the
    same asset are already folded) that are not objects (no second copy of the entity rule)."""
    from isaaclab.sensors import ContactSensor

    scene, out = env.scene, []
    am = env.action_manager
    _policy_channels(out, am, ents, scene)
    dec = int(env.cfg.decimation)
    filtered = []
    extras = extra_slot_sensors(env)
    for k, s in scene.sensors.items():
        if not isinstance(s, ContactSensor):
            continue
        if s.cfg.filter_prim_paths_expr:
            filtered.append((k, s))
            continue
        if k in extras:  # sensor of an extra slot body: not a hand sensor (module doc force_total)
            continue
        h = s.data.net_forces_w_history
        n_hist = min(dec, int(h.shape[1])) if h is not None else 1
        if n_hist != 1:
            raise RuntimeError(
                f"{MARK} sensor {k} history window {n_hist} > 1: the object force (one last physics "
                "step) cannot follow the same time axis"
            )
        nb = int(s.num_bodies)

        def rd(s=s, n_hist=n_hist):
            hh = s.data.net_forces_w_history
            x = hh[:, :n_hist] if hh is not None else s.data.net_forces_w.unsqueeze(1)
            return x.reshape(x.shape[0], -1).float()

        out.append(
            {
                "name": f"{KIND_TOTAL}/{k}",
                "kind": KIND_TOTAL,
                "width": n_hist * nb * 3,
                "names": list(s.body_names),
                "history": n_hist,
                "read": rd,
            }
        )
    if not any(c["kind"] == KIND_TOTAL for c in out):
        raise RuntimeError(f"{MARK} the scene has no unfiltered contact sensor: total forces cannot be recorded")
    objs = object_channels(env)
    miss = sorted({n for c in out if c["kind"] == KIND_TOTAL for n in c["names"]} - set(objs[0]["names"]))
    # links with total force need object force too to subtract 'non-objects' (one-hand scenes have more object-force
    #   links)
    if miss:
        raise RuntimeError(f"{MARK} {len(miss)} total-force links outside the object-force link rule: {miss[:4]}")
    out += objs
    partner = None
    from dexengine.record.rollout import partner as CP
    from dexengine.rl.tasks.mdp import hand_object_contact as HOC

    sens = [
        (c["name"].split("/", 1)[1], scene.sensors[c["name"].split("/", 1)[1]]) for c in out if c["kind"] == KIND_TOTAL
    ]
    objk = [(c["name"].split("/", 1)[1], c["obj"]) for c in objs]
    partner = CP.PartnerReader(env, sens, objk, HOC.instance(env))
    n_env = int(env.num_envs)
    _partner_channels(partner, out, n_env)
    oracle, lines = oracle_pairs(filtered, objs)
    for ln in lines:
        print(ln, flush=True)
    return out, oracle, partner


ORACLE_FILTER = "[ORACLE_FILTER]"


def oracle_pairs(filtered: list, objs: list[dict]) -> tuple[list, list[str]]:
    """Oracle pairs (rule owner): only filtered contact sensors whose filter path set equals an object channel's
    object_bodies set; per body of that sensor (sensor key, sensor, object channel, object link index). Marker lines:
    one per pair (sensor, filter, bodies, object bodies) + a summary (filtered sensors, pairs, body list per object
    channel; printed even with 0 pairs)."""
    oracle, lines = [], []
    for k, s in filtered:
        flt = {str(e).replace(".*", "*") for e in s.cfg.filter_prim_paths_expr}
        for c in objs:
            if flt == set(c["object_bodies"]):
                lines.append(
                    f"{ORACLE_FILTER} sensor {k} · filters {sorted(flt)} · bodies {list(s.body_names)} · "
                    f"object {c['name']} bodies {sorted(c['object_bodies'])}"
                )
                oracle.extend((k, s, c["name"], c["names"].index(b)) for b in s.body_names)
    paired = len({k for k, *_ in oracle})
    lines.append(
        f"{ORACLE_FILTER} filtered sensors {len(filtered)} · paired sensors {paired} · paired bodies "
        f"{len(oracle)} · object channels " + str({c["name"]: sorted(c["object_bodies"]) for c in objs})
    )
    return oracle, lines
