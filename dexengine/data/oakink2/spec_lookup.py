#!/usr/bin/env python3
"""OakInk2 joint spec lookup: finds the published joint to place on a one-body cluster (a tree whose jointed part stayed
still).

Identity rule: a joint's identity = the **prefix-independent set of part ids** (kind · body rid · moving rid · fixed
attachment rids; pair_key). Object id prefixes (left_/right_/raw_, stripped by oakink2.parts.rid_of) only name the mesh
variant the tree loaded, not the thing (meshes of the same rid share a frame and differ only in their split). Asset
folders (USD, meshes) keep the prefixed id, since the bake uses that tree's meshes.
So even when one part pair has different asset ids per prefix combination, the joint statement belongs to the pair:
  exact   the published asset of this tree's id combination, if there is one (as before).
  pair    otherwise a published asset with the same pair_key; its joint must pass through this tree's rest pose within
      the
          D1-g thresholds (dexengine.data.artic.ranges._relate: another frame or off-rail does not fit = oracle). Among
          those that fit, the one reproduced by the most trees (n_fit; ties = key order). Window = the pair's published
          window (in this tree's coordinates) ∪ this tree's interval.
When the rule cannot apply: a part id that is not an official part fails (ValueError) · no published asset of the same
pair, or none that fits = no statement (the converter skips that tree with a reason; the gate before the entry fails
it).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dexengine.core import artifact_paths as _TP  # noqa: E402  (asset ids, kind vocabulary)
from dexengine.data.artic import ranges as AR  # noqa: E402 — owner of reexpress and the fit check
from dexengine.data.oakink2 import parts as OP  # noqa: E402 — owner of prefix-stripped part ids (rid_of)
from dexengine.data.oakink2.artic_fit import block  # noqa: E402

TWO_PI = 2.0 * np.pi


def _rid(obj_id) -> str:
    r = OP.rid_of(str(obj_id))
    if not OP.is_part(r):
        raise ValueError(f"[SPEC_LOOKUP] {obj_id} → {r} is not an official part — no identity from the part id set")
    return r


def pair_key(kind: str, base_id, moving_id, fixed_ids=()) -> tuple:
    """A joint's identity: the prefix-independent set of part ids."""
    return kind, _rid(base_id), _rid(moving_id), tuple(sorted(_rid(i) for i in fixed_ids))


def spec_key(sp: dict) -> tuple:
    p = sp["parts"]
    return pair_key(sp["type"], p["base_id"], p["moving_id"], [f["id"] for f in p.get("fixed") or []])


def rel0(qa, qb):
    """First-frame pose (p, R) of body b in body a's frame; the child_pose convention (joint 0 = first frame)."""
    Ra, Rb = (Rot.from_quat(np.asarray(x[0, 3:7])[[1, 2, 3, 0]]) for x in (qa, qb))
    return Ra.inv().apply(qb[0, :3] - qa[0, :3]), Ra.inv() * Rb


def _placed(sp: dict, q, a: dict, b: dict) -> dict:
    """The spec axis and anchor (child frame) placed in the base frame with this tree's first-frame relative pose."""
    p0, R0 = rel0(block(q, a), block(q, b))
    n = R0.apply(np.asarray(sp["axis"], float))
    pivot = p0 + R0.apply(np.asarray(sp.get("pos") or [0, 0, 0], float))
    return {
        "kind": "revolute" if sp["type"] == "hinge" else "prismatic",
        "axis": (n / np.linalg.norm(n)).tolist(),
        "pivot": pivot.tolist(),
        "source": "spec",
    }


def candidates(objects: dict, ds: str, cluster: dict, q, by_pair: bool = False) -> list[tuple]:
    """Spec candidates of a one-body cluster [(base, moving, [(part, link)], spec joint in the base frame, spec key)]:
    ordered part pairs (base, moving), the rest fixed attachments. by_pair=False: the asset of this tree's id
    combination
    (artifact_paths.object_asset_id) · True: assets of **other id combinations** of the same part pair (pair_key). Links
    come
    from the spec's parts.fixed. The fit is judged by canon_joint (reexpress), the converter FK oracle and
    rest_records."""
    members = [cluster["anchor"], *cluster["fixed"]]
    pool = sorted((k, sp) for k, sp in objects.items() if k.startswith(ds + "/")) if by_pair else ()
    out = []
    for a in members:
        for b in members:
            if a is b:
                continue
            rest = [o for o in members if o is not a and o is not b]
            for kind in _TP._ARTIC_KINDS:  # the kind vocabulary lives in artifact_paths
                parts = {"moving_id": b["obj_id"], **({"fixed": [{"id": o["obj_id"]} for o in rest]} if rest else {})}
                key = f"{ds}/{_TP.object_asset_id(a['obj_id'], {'type': kind, 'parts': parts})}"
                if not by_pair:
                    found = [(key, objects[key])] if key in objects else []

                    def link_of(sp):
                        return {f["id"]: f["link"] for f in sp["parts"].get("fixed") or []}
                else:
                    want = pair_key(kind, a["obj_id"], b["obj_id"], [o["obj_id"] for o in rest])
                    found = [(k, sp) for k, sp in pool if k != key and spec_key(sp) == want]
                    by = {_rid(o["obj_id"]): o["obj_id"] for o in rest}

                    def link_of(sp, by=by):
                        return {by[_rid(f["id"])]: f["link"] for f in sp["parts"].get("fixed") or []}

                for k, sp in found:
                    link = link_of(sp)
                    out.append((a, b, [(o, link[o["obj_id"]]) for o in rest], _placed(sp, q, a, b), k))
    return out


def rest_records(ds: str, q, objects: dict, pc: dict, tree_joint) -> list[tuple[str, dict]]:
    """Tree statements [(base object id, statement)] of trees whose jointed part stayed still (part_clusters
    one_cluster),
    read by the spec stage. The exact matches if any (window widened to the rest pose), otherwise one pair (module
    docstring; it becomes the asset of a new id combination). `tree_joint` = the converter's tree statement builder
    (_tree_joint, owner of the statement shape)."""
    cl = pc["clusters"][0]

    def mine(by_pair):
        return [
            c
            for c in candidates(objects, ds, cl, q, by_pair)
            if c[0] is cl["anchor"] and c[1] is pc["joint"] and c[3]["kind"] == pc["joint_kind"]
        ]

    exact = mine(False)
    if exact:
        return [(str(a["obj_id"]), tree_joint(q, a, b, fx, fit)[0]) for a, b, fx, fit, _k in exact]
    best = None
    for a, b, fx, fit, k in mine(True):
        art, sib = tree_joint(q, a, b, fx, fit)[0], objects[k]
        rel = AR._relate(sib, art)  # does the pair's published joint pass through this tree's rest pose
        if rel["fits"] and (best is None or (-sib["n_fit"], k) < best[0]):
            best = ((-sib["n_fit"], k), a, art, rel, sib)
    if best is None:
        return []
    _key, a, art, rel, sib = best
    lo, hi = sorted(
        (float(sib[x]) - rel["off"]) / rel["s"] for x in ("lo", "hi")
    )  # published window → this tree's coordinates
    lo, hi = min(lo, art["range"][0]), max(hi, art["range"][1])
    if art["type"] == "hinge" and hi - lo > TWO_PI:
        return []  # a window beyond 2π is no limit; this pair cannot place it
    art["range"] = [round(lo, 6), round(hi, 6)]
    return [(str(a["obj_id"]), art)]


def rest_joints(objects: dict, ds: str, pc: dict, q, tree_joint, canon_joint) -> tuple[list, list]:
    """Published joint candidates of a one-body cluster decision `pc` (one_cluster): (got = [(position error, rotation
    error, asset, (body, jointed part, fixed attachments, fit, art_t, cj))] ordered by first-frame fit in canonical
    coordinates, miss = reasons for the ones that did not fit). Read by the conversion decision and the spec stage
    (rigid-joint records). tree_joint · canon_joint = converter functions (same shape as rest_records, no cycle).
    The miss reasons reach the exclusion ledger."""
    cl, got, miss = pc["clusters"], [], []
    for a_o, b_o, fixed_objs, fit, _k in candidates(objects, ds, cl[0], q):
        if a_o is not cl[0]["anchor"] or b_o is not pc["joint"] or fit["kind"] != pc["joint_kind"]:
            continue  # assets that are not the spec kind or pair (body → jointed part) are not candidates
        art_t, al, _p0, _R0 = tree_joint(q, a_o, b_o, fixed_objs, fit)
        try:
            cj = canon_joint(objects, ds, art_t, al)
        except ValueError as e:  # the spec window does not hold this tree's joint values; kept as a reason
            miss.append(f"{a_o['obj_id']}>{b_o['obj_id']}: {str(e)[:120]}")
            continue
        if "free" in cj:
            miss.append(f"{cj['asset']}: {cj['free']}")
            continue
        got.append(
            (
                round(cj["rx"]["pos_err_m"], 6),
                round(cj["rx"]["rot_err_deg"], 4),
                cj["asset"],
                (a_o, b_o, fixed_objs, fit, art_t, cj),
            )
        )
    got.sort(key=lambda g: g[:3])  # the asset whose first-frame pose fits best in canonical coordinates
    return got, miss
