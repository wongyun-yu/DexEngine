"""Per-tree conversion steps of the D1-g artic conversion (articulated pair · fixed merge); driven by
`artic_convert.convert_tree`."""

from __future__ import annotations

import numpy as np

from dexengine.data import rules as RR
from dexengine.data.artic import ranges as AR
from dexengine.data.base_layer import geom as BG  # owner of load_model and recompute_qvel
from dexengine.data.oakink2 import (
    attach as AT,  # measurements of attach pose, lever tip displacement and the FK oracle
)
from dexengine.data.oakink2 import (
    fixed_merge as FM,  # owner of the rigid-joint output (merged mesh asset, pose assignment, oracle)
)
from dexengine.data.oakink2.artic_decide import (
    _adr,
)
from dexengine.data.oakink2.artic_fit import (  # owner of the pairing rule and thresholds
    POS_TOL_M,
    ROT_TOL_DEG,
    block,
    in_band,
)
from dexengine.data.oakink2.artic_xml import (
    _eq,
    _fixed_task_info,
    _rewrite_xml,
    _rewrite_xml_fixed,
    _self_group,
)

HAND = RR.hand_dir()


def _artic_step(clip, st, q, dec, write, canon):
    """Apply one joint decision to the state `st` (scene · scene_eq · trajectory · task_info; convert_tree writes once
    at the end). Part poses = source trajectory `q` · model and addresses = current state. Outside the FK oracle → a
    reason row."""
    ti, z, T = st["ti"], st["z"], len(q)
    objs = ti.get("objects") or []
    a_o, b_o, fixed_objs, fit, art_t, cj = (dec[k] for k in ("a_o", "b_o", "fixed_objs", "fit", "art_t", "cj"))
    qb = block(q, b_o)
    art, theta = cj["art"], cj["theta"]
    base_name, mov_name = a_o["name"], b_o["name"]
    id2obj = {o["obj_id"]: o for o, _l in fixed_objs}
    spec_fixed = art["parts"].get("fixed") or []
    # the asset id holds the fixed ids, so a mismatch is a defect
    if sorted(f["id"] for f in spec_fixed) != sorted(id2obj):
        return {
            "clip": clip,
            "status": "error:fixed_mismatch",
            "why": f"fixed attachments of the statement {sorted(f['id'] for f in spec_fixed)} ≠ tree cluster "
            f"{sorted(id2obj)}",
        }
    fixed_bodies = [(id2obj[f["id"]]["name"], f["link"], f["pose"]) for f in spec_fixed]
    id2obj_n = {o["name"]: o for o, _l in fixed_objs}
    xml_old = st["xml"]
    xml_new = _rewrite_xml(xml_old, base_name, mov_name, art, fixed_bodies)
    if xml_new is None:
        return {"clip": clip, "status": "skip_xml_body_missing"}
    m_old = BG.load_model(xml_old, "oakink2", HAND)
    m_new = BG.load_model(xml_new, "oakink2", HAND)
    if m_new.nq != m_old.nq - 6 - 7 * len(fixed_bodies):
        return {"clip": clip, "status": f"skip_nq {m_old.nq}->{m_new.nq}"}
    A_old, A_new = _adr(m_old), _adr(m_new)  # qpos rearranged from old joint names to new addresses
    kind = art["type"]
    jname = f"{base_name}_{kind}"
    remap = _remapper(A_old, A_new, m_new, jname, theta, T)
    q_new = remap(z["qpos"])
    # FK oracle: world pose of the new model's <base>_top and fixed attachment bodies vs the source part poses (per
    # part).
    # Levers (fixed at rest) are measured by tip displacement; bound = LEVER_TIP_M (same as pairing) + POS_TOL_M.
    lever = {n: AT.part_points(id2obj_n[n]) for n, *_ in fixed_bodies if AT.is_lever(id2obj_n[n])}
    chk = [(f"{base_name}_top", qb)] + [(n, block(q, id2obj_n[n])) for n, *_ in fixed_bodies]
    err = AT.fk_errors(m_new, q_new, chk, lever)
    if err is None:
        return {"clip": clip, "status": "skip_xml_body_missing"}
    dp, dr, _ds = err[f"{base_name}_top"]
    fp = max((err[n][0] for n, *_ in fixed_bodies if n not in lever), default=0.0)
    fr = max((err[n][1] for n, *_ in fixed_bodies if n not in lever), default=0.0)
    ls = max((err[n][2] for n in lever), default=0.0)
    res = _oracle_row(clip, dec, fit, a_o, b_o, art, art_t, cj, T, dp, dr, fp, fr, ls, lever, fixed_bodies, id2obj_n)
    if (
        not in_band(fit["kind"], dp, dr)
        or fp > POS_TOL_M
        or fr > ROT_TOL_DEG
        or not AT.lever_in_band(ls, slack_m=POS_TOL_M)
    ):
        res.update(status="skip_oracle_fail", exclude="fk_oracle")
        return res
    AR.clamp(cj, theta, res)
    q_new = remap(z["qpos"])
    res["status"] = "converted" if write else "would_convert"
    st["xml"], z["qpos"] = xml_new, q_new
    if not write:
        return res
    side = _write_trajectory(z, ti, m_new, q_new, remap, theta, a_o)
    jadr = A_new[jname][0]
    # task_info
    int(ti["hand_qpos_dim"])  # a tree without hand_qpos_dim is refused (KeyError)
    # bodies merged into the articulation or fixed attachments; removed from objects[] and qpos_layout
    gone = {mov_name, *id2obj_n}
    new_objs = _converted_objects(
        objs, gone, A_new, base_name, a_o, b_o, id2obj_n, art, jname, jadr, side, dp, fit, cj, canon
    )
    lay = _converted_layout(ti, gone, new_objs, A_old, A_new, jname, jadr, a_o)
    _store_objects(ti, lay, new_objs, m_new)
    _eq(st, clip, f"{base_name}/{mov_name}/fixed attachment", _rewrite_xml, base_name, mov_name, art, fixed_bodies)
    return res


def _remapper(A_old, A_new, m_new, jname, theta, T):
    """qpos rearrangement from the old model's joint addresses to the new one's; the new joint column takes θ on
    the source timeline (zero elsewhere)."""

    def remap(Q):
        out = np.zeros((len(Q), m_new.nq), Q.dtype)
        for nm, (s_new, w) in A_new.items():
            if nm == jname:
                out[:, s_new] = theta[: len(Q)] if len(Q) == T else 0.0
                continue
            s_old, w_old = A_old[nm]
            out[:, s_new : s_new + w] = Q[:, s_old : s_old + w]
        return out

    return remap


def _oracle_row(clip, dec, fit, a_o, b_o, art, art_t, cj, T, dp, dr, fp, fr, ls, lever, fixed_bodies, id2obj_n) -> dict:
    """Result row of one joint conversion with its FK oracle measurements (fixed attachment and lever fields only when
    the tree has them, so trees without them keep the old record fields)."""
    res = {
        "clip": clip,
        "kind": fit["kind"],
        "base": a_o["obj_id"],
        "moving": b_o["obj_id"],
        "fk_pos_max_m": round(dp, 4),
        "fk_rot_max_deg": round(dr, 2),
        "range": [round(art["range"][0], 4), round(art["range"][1], 4)],
        "T": T,
        "asset": cj["asset"],
        "restated": bool(cj["rx"] is not None and AR.needs_restate(art, art_t, cj["rx"])),
    }
    if fixed_bodies:  # trees without fixed attachments keep the old record fields
        res.update(
            fixed=[id2obj_n[n]["obj_id"] for n, *_ in fixed_bodies],
            fk_fixed_pos_max_m=round(fp, 4),
            fk_fixed_rot_max_deg=round(fr, 2),
            joint_source=fit.get("source", "fit"),
        )
    if lever:
        res.update(levers=dec["levers"], fk_lever_shift_max_m=round(ls, 4))
    return res


def _write_trajectory(z, ti, m_new, q_new, remap, theta, a_o) -> str:
    """Trajectory arrays of the converted model: rollout remapped, qvel recomputed, θ as the hand side's
    qpos_obj_arti sidecar. Returns that side."""
    dt = (
        float(1.0 / z["frequency"]) if "frequency" in z and float(z["frequency"]) > 0 else float(ti.get("ref_dt", 0.02))
    )
    if "qpos_rollout" in z:
        z["qpos_rollout"] = remap(z["qpos_rollout"])
    z["qvel"] = BG.recompute_qvel(m_new, q_new, dt)
    side = a_o.get("side") if a_o.get("side") in ("right", "left") else "right"
    z[f"qpos_obj_arti_{side}"] = np.asarray(theta)[:, None].astype(q_new.dtype)
    return side


def _converted_objects(
    objs, gone, A_new, base_name, a_o, b_o, id2obj_n, art, jname, jadr, side, dp, fit, cj, canon
) -> list[dict]:
    """objects[] without the merged bodies, with new qpos addresses; the base object becomes the articulated
    object."""
    new_objs = []
    for o in objs:
        if o["name"] in gone:
            continue
        o = dict(o)
        jn = f"{o['name']}_joint"
        if jn in A_new:
            o["qpos_start"] = A_new[jn][0]  # slot is renumbered below by address rank once new_objs is final
        if o["name"] == base_name:
            _merge_into_base(o, a_o, b_o, id2obj_n, art, jname, jadr, side, dp, fit, cj, canon)
        new_objs.append(o)
    return new_objs


def _merge_into_base(o, a_o, b_o, id2obj_n, art, jname, jadr, side, dp, fit, cj, canon) -> None:
    """The base object `o` takes the main-object role of the merged parts and the articulation statement."""
    # The merged articulated object takes the role of whichever part (two link representatives, fixed
    # attachments) was the main object (a moving main part makes the whole object that hand's main object; both
    # → both).
    sides = set()
    for part in (a_o, b_o, *id2obj_n.values()):
        if part.get("role") == "main" and part.get("side"):
            sides |= {"right", "left"} if part["side"] == "both" else {part["side"]}
    if sides:
        o["role"], o["side"] = "main", ("both" if len(sides) == 2 else next(iter(sides)))
    _self_group(o)
    # Statement = art (the spec as is when canonicalised, this tree's own fit when canon is None); parts are **relative
    # paths** from mesh_dir (= base part folder; consumer contract coords.artic_moving_vertices) · child_pose =
    # the
    # child body's pose in the base frame (θ=0).
    o["articulation"] = {
        "type": art["type"],
        "joint": jname,
        "axis": art["axis"],
        "pos": art["pos"],
        "range": art["range"],
        "qpos_index": jadr,
        "npz_key": f"qpos_obj_arti_{side}",
        "parts": art["parts"],
        "child_pose": art["child_pose"],
        "max_abs_diff_vs_qpos": round(dp, 4),
        "fit": {
            k: fit.get(k)
            for k in ("rot_resid_deg", "pos_resid_m", "angle_range_deg", "travel_m", "source")
            if k != "source" or k in fit
        },
        "canon": (
            {"asset": cj["asset"], "spec_sha": canon[1], "s": cj["rx"]["s"], "theta_off": cj["rx"]["theta_off"]}
            if canon is not None
            else "per_tree"
        ),
    }


def _converted_layout(ti, gone, new_objs, A_old, A_new, jname, jadr, a_o) -> list[dict]:
    """qpos_layout of the converted model: merged bodies removed, every block at its new address, the new joint
    column added (sorted by address)."""
    lay = []
    for e in ti.get("qpos_layout") or []:
        if e.get("kind") == "object" and e.get("name") in gone:
            continue
        e = dict(e)
        if e.get("kind") == "object":
            e["start"] = A_new[f"{e['name']}_joint"][0]
            ob = next((x for x in new_objs if x["name"] == e["name"]), None)
            if ob is not None:
                e["role"], e["side"] = ob.get("role"), ob.get("side")
        elif e.get("kind") == "hand":
            first = next(nm for nm, (s, _w) in A_old.items() if s == e["start"])
            e["start"] = A_new[first][0]
        lay.append(e)
    lay.append({"name": jname, "start": jadr, "width": 1, "kind": "joint", "side": a_o.get("side")})
    lay.sort(key=lambda e: e["start"])
    return lay


def _store_objects(ti, lay, new_objs, m_new) -> None:
    """Store the converted layout and objects in task_info, each object's slot = its rank among the address-sorted
    object blocks; per hand, the main-object slot follows (dropped when that hand has no main object)."""
    # slot = rank among the address-sorted object blocks (main∪extra; joint columns are not counted), matching the one
    # definition in base_layer.geom.build_objects(). `(qpos_start-h2)//7` arithmetic does not know the joint column gaps
    # this conversion makes and could give a wrong slot.
    _starts_sorted = sorted({int(o["qpos_start"]) for o in new_objs})
    for o in new_objs:
        o["slot"] = _starts_sorted.index(int(o["qpos_start"]))
    ti["qpos_layout"] = lay
    ti["objects"] = new_objs
    ti["nq"] = int(m_new.nq)
    for s in ("right", "left"):
        if ti.get(f"{s}_object_slot") is not None:
            tgt = next((o for o in new_objs if o.get("role") == "main" and o.get("side") in (s, "both")), None)
            if tgt is not None:
                ti[f"{s}_object_slot"] = tgt["slot"]
            else:
                ti.pop(f"{s}_object_slot", None)


def _fixed_step(clip, st, q, dec, write, canon):
    """Apply one rigid-joint decision to the state `st` (same convention as _artic_step): body = the merged mesh asset
    (obj_id = asset id) · parts leave objects[] and qpos and are welded to the body in the MuJoCo scene (pose = asset
    statement) · FK oracle."""
    cl, aid, ti, z = dec["fixed"], dec["asset"], st["ti"], st["z"]
    sp = canon[2][aid]
    body, members = cl["anchor"], cl["fixed"]
    welds = [(o["name"], sp["members"][FM.OP.rid_of(str(o["obj_id"]))]["pose"]) for o in members]
    xml_old = st["xml"]
    xml_new = _rewrite_xml_fixed(xml_old, body["name"], welds)
    if xml_new is None:
        return {"clip": clip, "status": "skip_xml_body_missing"}
    m_old, m_new = BG.load_model(xml_old, "oakink2", HAND), BG.load_model(xml_new, "oakink2", HAND)
    if m_new.nq != m_old.nq - 7 * len(members):
        return {"clip": clip, "status": f"skip_nq {m_old.nq}->{m_new.nq}"}
    A_old, A_new = _adr(m_old), _adr(m_new)

    def remap(Q):
        out = np.zeros((len(Q), m_new.nq), Q.dtype)
        for nm, (s_new, w) in A_new.items():
            s_old = A_old[nm][0]
            out[:, s_new : s_new + w] = Q[:, s_old : s_old + w]
        return out

    q_new = remap(z["qpos"])
    chk = [(body["name"], block(q, body))] + [(n, block(q, o)) for (n, _p), o in zip(welds, members, strict=True)]
    err = AT.fk_errors(m_new, q_new, chk)  # FK of body and welded bodies vs source poses (independent measurement)
    dp, dr, _ds = err.pop(body["name"])
    pin = {
        n: err.pop(n)[:2] for (n, _p), o in zip(welds, members, strict=True) if FM.pinned(sp, o)
    }  # pinned parts: recorded only
    fp, fr = max((e[0] for e in err.values()), default=0.0), max((e[1] for e in err.values()), default=0.0)
    res = {
        "clip": clip,
        "kind": "fixed",
        "base": body["obj_id"],
        "moving": ",".join(o["obj_id"] for o in members),
        "asset": aid,
        "fixed": [o["obj_id"] for o in members],
        "fk_pos_max_m": round(dp, 4),
        "fk_rot_max_deg": round(dr, 2),
        "fk_fixed_pos_max_m": round(fp, 4),
        "fk_fixed_rot_max_deg": round(fr, 2),
        "T": len(q_new),
        "restated": False,
        **FM.pin_shift(pin),
    }
    if max(dp, fp) > POS_TOL_M or max(dr, fr) > ROT_TOL_DEG:
        res.update(status="skip_oracle_fail", exclude="fk_oracle")
        return res
    res["status"] = "converted" if write else "would_convert"
    st["xml"], z["qpos"] = xml_new, q_new
    if not write:
        return res
    freq = float(z["frequency"]) if "frequency" in z else 0.0
    dt = 1.0 / freq if freq > 0 else float(ti.get("ref_dt", 0.02))
    if "qpos_rollout" in z:
        z["qpos_rollout"] = remap(z["qpos_rollout"])
    z["qvel"] = BG.recompute_qvel(m_new, q_new, dt)
    _fixed_task_info(ti, body, members, aid, A_old, A_new, res, canon[1], m_new.nq)
    _eq(st, clip, f"{body['name']}/fixed part", _rewrite_xml_fixed, body["name"], welds)
    return res
