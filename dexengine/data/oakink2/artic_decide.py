"""Decisions of the articulated converter (rule D1-g): the canonical joint of a part kind, the joint measured in a
tree, and per-instance / per-tree decisions (articulate, fix, rest)."""

from __future__ import annotations

import numpy as np

from dexengine.core import artifact_paths as _TP  # noqa: E402  (object_asset_id)
from dexengine.data.artic import (
    convert_version as D1G,  # noqa: E402 — owner of the version and the fate-count check (tally); dependency-free module
)
from dexengine.data.artic import ranges as AR  # noqa: E402 — owner of the canonical joint spec and reexpress
from dexengine.data.oakink2 import (
    attach as AT,  # noqa: E402 — measurements of attach pose, lever tip displacement and the FK oracle
)
from dexengine.data.oakink2 import (
    fixed_merge as FM,  # noqa: E402 — owner of the rigid-joint output (merged mesh asset, pose assignment, oracle)
)
from dexengine.data.oakink2 import multi_instance as MI  # noqa: E402 — owner of scenes with two or more instances
from dexengine.data.oakink2 import (
    spec_lookup as SL,  # noqa: E402 — owner of the spec joint lookup for one-body clusters (identity = part pair regardless of prefix)
)
from dexengine.data.oakink2.artic_fit import (  # noqa: E402 — owner of the pairing rule and thresholds  # noqa: E402
    ROT_TOL_DEG,
    PartExcluded,
    PartMergeError,
    block,
    joint_values,
    part_clusters,
    rel_pose,
)
from dexengine.data.oakink2.artic_xml import _child_pose, _parts_record  # noqa: F401


def canon_joint(objects: dict, ds: str, art_t: dict, al) -> dict:
    """Tree statement `art_t` (the joint this tree's fit built: type · axis · pos · child_pose · parts · range, child
    frame) and joint values α → canonical joint: {art: the spec's canonical statement as is (part paths and ids from the
    spec, role fields from the tree) + range = spec (lo, hi), theta: θ = s·α + θ_off (reexpress, window = spec (lo,
    hi)),
    rx, asset}; outside the D1-g thresholds in canonical coordinates → {free: reason, asset}. When the same part pair
    has
    canonical joints only of **another kind**, this tree does not fit that reference joint → free (a poor fit to the
    reference joint means two separately moving objects; boundary fits split kinds per hand, and since the asset id
    holds
    the kind, without this branch it would fail as a 'new asset'). A pair missing from the spec entirely fails (the spec
    stage registers new assets first). The free reasons and the ValueError text reach the exclusion ledger (through
    spec_lookup.rest_joints)."""
    base_id = art_t["parts"]["base_id"]
    asset = _TP.object_asset_id(base_id, art_t)
    sp = objects.get(f"{ds}/{asset}")
    if sp is None:
        other = [
            a
            for a in (
                _TP.object_asset_id(base_id, {**art_t, "type": k}) for k in _TP._ARTIC_KINDS if k != art_t["type"]
            )
            if f"{ds}/{a}" in objects
        ]  # the kind vocabulary lives in artifact_paths
        if other:
            return {
                "free": f"type differs (tree {art_t['type']} · canonical {objects[f'{ds}/{other[0]}']['type']})",
                "asset": other[0],
            }
        raise ValueError(
            f"new asset {ds}/{asset} — not in the spec: spec stage (dexengine.data.artic.spec_stage --dataset {ds}) "
            "has not seen this tree"
        )
    rx = AR.reexpress(sp, art_t, (sp["lo"], sp["hi"]))
    if not rx["fits"]:
        return {
            "free": f"off canonical (position {rx['pos_err_m'] * 1e3:.2f} mm · rotation {rx['rot_err_deg']:.2f}°)",
            "asset": asset,
        }
    st = AR.statement(sp)
    art = {**st, "parts": {**art_t["parts"], **st["parts"]}, "range": [sp["lo"], sp["hi"]]}
    return {"art": art, "theta": rx["s"] * np.asarray(al, float) + rx["theta_off"], "rx": rx, "asset": asset}


def _tree_joint(q, a_o, b_o, fixed_objs, fit):
    """Tree statement art_t (child-frame axis and anchor · child_pose · parts · range), joint values α and the
    first-frame
    relative pose (p0, R0). `fit` = the two-body fit (oakink2.artic_fit.fit_pair) or a spec joint placed in the base
    frame (spec_lookup.candidates). Fixed attachment pose = pose relative to the link representative body
    (oakink2.attach.attach_pose, the child_pose convention)."""
    qa, qb = block(q, a_o), block(q, b_o)
    al = joint_values(qa, qb, fit)
    p0, R0 = (x[0] for x in rel_pose(qa, qb))  # first-frame relative pose (base body frame)
    n = np.asarray(fit["axis"], float)
    c = np.asarray(fit.get("pivot", [0, 0, 0]), float)
    pad = 1e-3
    kind = "hinge" if fit["kind"] == "revolute" else "slide"
    fixed = [
        (
            o,
            link,
            _child_pose(
                *AT.attach_pose(*rel_pose(qa if link == "base" else qb, block(q, o)), AT.is_lever(o), ROT_TOL_DEG)
            ),
        )
        for o, link in fixed_objs
    ]
    art_t = {
        "type": kind,
        "axis": [round(float(v), 6) for v in (R0.as_matrix().T @ n)],
        "pos": [round(float(v), 6) for v in ((R0.as_matrix().T @ (c - p0)) if kind == "hinge" else np.zeros(3))],
        "child_pose": _child_pose(p0, R0),
        "parts": _parts_record(a_o, b_o, fixed),
        "range": [round(float(al.min()) - pad, 6), round(float(al.max()) + pad, 6)],
    }  # same text as the tree record
    return art_t, al, p0, R0


def _adr(m) -> dict:
    """Joint name → (qpos address, width)."""
    import mujoco

    out = {}
    for j in range(m.njnt):
        jt = m.jnt_type[j]
        w = 7 if jt == mujoco.mjtJoint.mjJNT_FREE else (4 if jt == mujoco.mjtJoint.mjJNT_BALL else 1)
        out[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)] = (int(m.jnt_qposadr[j]), w)
    return out


def decide(objs: list[dict], q, ds: str, canon, lever: frozenset = frozenset()) -> dict:
    """Conversion decision: pairing rule (part_clusters) · two-cluster fit · one-cluster spec first · canonicalisation.
    Without conversion {status, why, …}; with conversion {a_o, b_o, fixed_objs, fit, art_t, cj} (joint) · {fixed, asset,
    err} (rigid joint) · {multi, …} (two or more merged instances, oakink2.multi_instance.plans). Writing and the FK
    oracle are in convert_tree."""
    try:
        pc = part_clusters(objs, q)  # the one pairing rule (same function as the measurer, oakink2.artic_fit)
    except PartMergeError as e:  # scenes without a decision fail with a reason (rc ≠ 0)
        if isinstance(
            e, PartExcluded
        ):  # a spec verdict = explicit exclusion (reason word = the code set where it was raised)
            return {"status": f"excluded:{e.code}", "exclude": e.code, "why": str(e)[:300]}
        return {"status": f"error:merge_{e.code}", "why": str(e)[:300]}
    if pc["status"] == "no_parts":
        return {"status": "skip_no_parts"}
    if pc["status"] == "multi":
        return MI.plans(pc, lambda p: _decide_one(p, q, ds, canon, lever))
    return _decide_one(pc, q, ds, canon, lever)


def _decide_one(pc: dict, q, ds: str, canon, lever: frozenset = frozenset()) -> dict:
    """Conversion decision for one instance decision `pc` (same shape as decide)."""
    cl, st = pc["clusters"], None
    who = {
        "members": [o["obj_id"] for c in cl for o in [c["anchor"], *c["fixed"]]],
        "clusters": [[o["obj_id"] for o in [c["anchor"], *c["fixed"]]] for c in cl],
    }
    if pc["status"] == "two_clusters":
        fit = pc["fit"]
        if fit["kind"] not in ("revolute", "prismatic"):
            return {
                "status": f"skip_kind={fit['kind']}",
                **who,
                "fit": {k: fit.get(k) for k in ("kind", "rot_resid_deg", "pos_resid_m")},
            }
        a_o, b_o = cl[0]["anchor"], cl[1]["anchor"]
        fixed_objs = [(o, "base") for o in cl[0]["fixed"]] + [(o, "moving") for o in cl[1]["fixed"]]
        art_t, al, _p0, _R0 = _tree_joint(q, a_o, b_o, fixed_objs, fit)
        if canon is None:
            cj = {"art": art_t, "theta": al, "rx": None, "asset": None}
        else:
            cj = canon_joint(canon[0], ds, art_t, al)
            if "free" in cj:
                return {
                    "status": "skip_canon_misfit",
                    "exclude": "canon_misfit",
                    "asset": cj["asset"],
                    "why": cj["free"],
                    **who,
                }
    elif pc["status"] == "fixed_only":
        # Rigid joint: one merged mesh asset (the oracle pick of oakink2.fixed_merge).
        if canon is None:
            return {"status": "skip_fixed_statement", **who}
        return FM.decision(canon[2], cl[0], q, who, ds)
    else:
        # one_cluster: the jointed part (spec) did not move in this clip. The joint comes from the same asset's spec
        # joint
        # (body → jointed part, spec kind; θ nearly constant). Without a matching spec, no conversion and a reason
        # (audit skip_union).
        if canon is None:
            return {
                "status": "skip_union",
                **who,
                "why": "all parts one body · the tree statement does not read the spec",
            }
        got, miss = SL.rest_joints(canon[0], ds, pc, q, _tree_joint, canon_joint)
        if not got:
            return _union_skip(pc, q, ds, canon, who, miss)
        a_o, b_o, fixed_objs, fit, art_t, cj = got[0][3]
        if (st := FM.static_rigid(cl[0], q, cj)) and "skip" not in st:  # a single rigid body instead of a joint
            return {**FM.decision(canon[2], cl[0], q, who, ds), "static_artic": st}
        if pin := FM.lever_pin(cl[0], q, st, lever):  # lever pin
            return FM.lever_decision(canon[2], cl[0], q, who, ds, st, pin)
    return {
        "a_o": a_o,
        "b_o": b_o,
        "fixed_objs": fixed_objs,
        "fit": fit,
        "art_t": art_t,
        "cj": cj,
        "levers": pc.get("levers") or [],
        **({"static_artic": st} if st else {}),
    }


def _union_skip(pc: dict, q, ds: str, canon, who: dict, miss: list) -> dict:
    """Reason record of a one-cluster tree whose spec joints all miss the rest pose: a misfitting spec asset, the
    published joint of the same part pair off this tree's rest pose, or no matching joint spec."""
    cl = pc["clusters"]
    pair = any(c[0] is cl[0]["anchor"] and c[1] is pc["joint"] for c in SL.candidates(canon[0], ds, cl[0], q, True))
    why = (
        ("spec asset exists but does not fit — " + " | ".join(miss))
        if miss
        else (
            "the published joint of the same part pair does not pass this tree's rest pose"
            if pair
            else "no matching joint spec"
        )
    )
    return {
        "status": "skip_union",
        **who,
        "why": "all parts as one body · " + why,
        "exclude": "canon_misfit" if miss else ("rest_off_joint" if pair else "no_joint_spec"),
    }


def _instance_decisions(ti: dict, q) -> list[dict]:
    """Instance decisions (oakink2.multi_instance.instances); undecidable = []. Read by per_tree_*."""
    try:
        return MI.instances(part_clusters(ti.get("objects") or [], q))
    except PartMergeError:
        return []


def per_tree_fixed(ti: dict, q, objects: dict | None = None) -> list[dict]:
    """Rigid-joint records of a tree not merged yet (oakink2.fixed_merge.record: body · parts · clip-median relative
    pose ·
    one per instance), read by the spec stage. With the published spec `objects`, static articulated objects too (same
    verdict as conversion)."""
    out = []
    for p in _instance_decisions(ti, q):
        g = (
            SL.rest_joints(objects, ti["dataset_name"], p, q, _tree_joint, canon_joint)[0]
            if objects and p["status"] == "one_cluster"
            else []
        )
        st = FM.static_rigid(p["clusters"][0], q, g[0][3][5]) if g else None
        pin = FM.lever_pin(p["clusters"][0], q, st, D1G.lever_for(ti))  # lever pin
        if p["status"] == "fixed_only" or (st and "skip" not in st) or pin:
            out.append(FM.record(p["clusters"][0], q, tuple(pin)))
    return out


def per_tree_rest(ti: dict, q, objects: dict) -> list[tuple[str, dict]]:
    """Statements of trees whose jointed part stayed still (one_cluster): the published spec joint placed at this tree's
    first frame (oakink2.spec_lookup.rest_records: the asset of this id combination, else one of the same part pair).
    Read by the spec stage."""
    return [
        r
        for p in _instance_decisions(ti, q)
        if p["status"] == "one_cluster"
        for r in SL.rest_records(ti["dataset_name"], q, objects, p, _tree_joint)
    ]


def per_tree_record(ti: dict, q) -> tuple[str, dict] | None:
    """**Tree statement** of a tree not merged yet (convert_tree's fit with canon=None, no write): (base object id,
    articulation) or None (no pair to merge · one body · kind out of scope · undecidable); at most one joint per tree.
    The spec update reads it through this one function."""
    for p in _instance_decisions(ti, q):
        if p["status"] in ("one_cluster", "two_clusters"):
            dec = _decide_one(p, q, ti["dataset_name"], None)
            return None if "status" in dec or "fixed" in dec else (str(dec["a_o"]["obj_id"]), dec["art_t"])
    return None
