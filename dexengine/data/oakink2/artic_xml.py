"""Scene-file and task_info edits of the articulated converter (rule D1-g): body moves, joint insertion, fixed merges,
the part record, and the repair of a tree's part record."""

from __future__ import annotations

import json
import os
import posixpath
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dexengine.core import paths as _P20  # noqa: E402 — owner of the clip-list entry (read_clip_list)
from dexengine.core.paths import (
    main_root as _data_root,  # noqa: E402 — owner of the data root (env DEXENGINE_MAIN_ROOT)
)

JOINT_ATTR = {
    "limited": "true",
    "armature": "0.0001",
    "damping": "0.01",
    "frictionloss": "0.0001",
}  # same values as arctic trees


def _find_body(root, name):
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    return None


def _parent_of(root, elem):
    for p in root.iter():
        if elem in list(p):
            return p
    return None


def _move_children(src, dst):
    """Move the child elements of body `src` to `dst`; free joints and tracking sites (trace_/track_, kept only on the
    base object) are dropped."""
    for el in list(src):
        if el.tag == "joint":  # free joint removed
            continue
        if el.tag == "site" and el.get("name", "").startswith(("trace_", "track_")):
            continue
        dst.append(el)


def _pose_attr(pose):
    return {"pos": " ".join(f"{v:.6f}" for v in pose["pos"]), "quat": " ".join(f"{v:.6f}" for v in pose["quat_wxyz"])}


def _rewrite_xml(xml_text, base_name, mov_name, art, fixed_bodies=()):
    """XML string with the moving part body moved under the base body as an articulated child; child pose and joint are
    written from the one statement `art` (type · axis · pos · child_pose · range, child frame). `fixed_bodies` = [(body
    name, link base|moving, pose in the link frame)]: fixed attachment parts move under that link body as **jointless
    child bodies** (MuJoCo weld). None on failure."""
    root = ET.fromstring(xml_text)
    base = _find_body(root, base_name)
    mov = _find_body(root, mov_name)
    if base is None or mov is None:
        return None
    fx = [(_find_body(root, n), n, link, pose) for n, link, pose in fixed_bodies]
    if any(b is None for b, *_ in fx):
        return None
    _parent_of(root, mov).remove(mov)
    for b, *_ in fx:
        _parent_of(root, b).remove(b)
    child = ET.SubElement(base, "body", {"name": f"{base_name}_top", **_pose_attr(art["child_pose"])})
    jt = {
        "name": f"{base_name}_{art['type']}",
        "type": art["type"],
        "pos": " ".join(f"{v:.6f}" for v in art["pos"]),
        "axis": " ".join(f"{v:.6f}" for v in art["axis"]),
        "range": f"{art['range'][0]:.6f} {art['range'][1]:.6f}",
        **JOINT_ATTR,
    }
    ET.SubElement(child, "joint", jt)
    _move_children(mov, child)
    for b, n, link, pose in fx:
        welded = ET.SubElement(base if link == "base" else child, "body", {"name": n, **_pose_attr(pose)})
        _move_children(b, welded)
    return ET.tostring(root, encoding="unicode")


def _parts_record(a_o, b_o, fixed=()):
    """Part paths relative to mesh_dir (base part folder), read by the same consumers as the arctic convention
    (`bottom.obj`/`top.obj`). Ids go in separate fields.
    `fixed` = [(object, link base|moving, pose in the link frame)]: fixed attachment parts (statement field `fixed` =
    {id · mesh · link · pose}; read by artifact_paths.artic_link_files). Else the field is left out (bytes equal to
    old trees)."""

    # relative path between two logical references (data/<ds>/assets/objects/<id>), not joined to a root (any root
    # cancels out)
    def rel(o):
        return posixpath.relpath(posixpath.join(o["mesh_dir"], "visual.obj"), a_o["mesh_dir"])

    out = {
        "base": "visual.obj",
        "moving": rel(b_o),
        "base_id": a_o["obj_id"],
        "moving_id": b_o["obj_id"],
        "moving_role": [b_o.get("role"), b_o.get("side")],
        "base_role": [a_o.get("role"), a_o.get("side")],
    }
    if fixed:
        fx = sorted(fixed, key=lambda f: str(f[0]["obj_id"]))
        out["fixed"] = [{"id": o["obj_id"], "mesh": rel(o), "link": link, "pose": pose} for o, link, pose in fx]
        # a fact of the tree, not part of the statement (outside dexengine.data.artic.ranges._PART_KEYS)
        out["fixed_roles"] = {o["obj_id"]: [o.get("role"), o.get("side")] for o, _l, _p in fx}
    return out


def _child_pose(p0, R0):
    q = R0.as_quat()[[3, 0, 1, 2]]
    return {"pos": [round(float(v), 6) for v in p0], "quat_wxyz": [round(float(v), 6) for v in q]}


def repair_parts(tree, write=False):
    """For trees already converted (rules.D1g present): articulation.parts from ids to relative paths, child_pose filled
    from the `<base>_top` pos/quat in scene.xml."""
    ti_p = os.path.join(tree, "task_info.json")
    with open(ti_p) as fh:
        ti = json.load(fh)
    if "D1g" not in (ti.get("rules") or {}):
        return {"clip": os.path.basename(tree), "status": "not_converted"}
    tgt = [o for o in ti["objects"] if o.get("articulation")]
    if len(tgt) != 1:
        return {"clip": os.path.basename(tree), "status": f"artic_objects={len(tgt)}"}
    o = tgt[0]
    art = o["articulation"]
    parts = art.get("parts") or {}
    if parts.get("base", "").endswith(".obj") and art.get("child_pose"):
        return {"clip": os.path.basename(tree), "status": "skip_done"}
    mov_id = parts.get("moving_id") or parts.get("moving")
    base_id = parts.get("base_id") or parts.get("base")
    if base_id != o["obj_id"]:
        return {"clip": os.path.basename(tree), "status": f"base_id_mismatch {base_id}!={o['obj_id']}"}
    # logical names (the stored text); grammar owned by data_layout
    b_dir = _P20.data_layout.object_ref(ti["dataset_name"], mov_id)
    b_phys = _data_root() / _P20.data_layout.object_path(ti["dataset_name"], b_dir)  # physical location = data root
    if not (b_phys / "visual.obj").is_file():
        return {"clip": os.path.basename(tree), "status": f"no_mesh {mov_id}"}
    root = ET.fromstring(Path(os.path.join(tree, "scene.xml")).read_text())
    body = _find_body(root, f"{o['name']}_top")
    if body is None:
        return {"clip": os.path.basename(tree), "status": "no_child_body"}
    p0 = np.array([float(v) for v in (body.get("pos") or "0 0 0").split()])
    qw = [float(v) for v in (body.get("quat") or "1 0 0 0").split()]
    R0 = Rot.from_quat(np.asarray(qw)[[1, 2, 3, 0]])
    a_o = {
        "obj_id": base_id,
        "mesh_dir": o["mesh_dir"],
        "role": parts.get("base_role", [None, None])[0],
        "side": parts.get("base_role", [None, None])[1],
    }
    b_o = {
        "obj_id": mov_id,
        "mesh_dir": b_dir,
        "role": parts.get("moving_role", [None, None])[0],
        "side": parts.get("moving_role", [None, None])[1],
    }
    art["parts"] = _parts_record(a_o, b_o)
    art["child_pose"] = _child_pose(p0, R0)
    ti["rules"]["D1g"]["parts_repaired"] = "C311-2026-09-18"
    if write:
        tmp = ti_p + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(ti, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, ti_p)
    return {
        "clip": os.path.basename(tree),
        "status": "repaired",
        "moving": art["parts"]["moving"],
        "child_pos_norm_m": round(float(np.linalg.norm(p0)), 4),
    }


def _eq(st, clip, what, rewrite, *a) -> None:
    """Apply the same rewrite to scene_eq.xml (if present), so the two scenes never diverge."""
    if st["eq"] is not None:
        st["eq"] = rewrite(st["eq"], *a)
        if st["eq"] is None:
            raise ValueError(f"{clip}: scene_eq.xml has no {what} body")


def _rewrite_xml_fixed(xml_text, base_name, welds):
    """Part bodies as jointless child bodies (weld) under the body; shape = the same union as the merged mesh asset.
    None on failure."""
    root = ET.fromstring(xml_text)
    base = _find_body(root, base_name)
    bs = [(_find_body(root, n), n, pose) for n, pose in welds]
    if base is None or any(b is None for b, *_ in bs):
        return None
    for b, n, pose in bs:
        _parent_of(root, b).remove(b)
        welded = ET.SubElement(base, "body", {"name": n, **_pose_attr({"pos": pose["p"], "quat_wxyz": pose["q"]})})
        _move_children(b, welded)
    return ET.tostring(root, encoding="unicode")


def _self_group(o: dict) -> None:
    """A merged part group is now one object: the remaining body's group record (representative, part list) becomes
    itself (called by both the joint and rigid-joint paths). A record pointing at a removed part as representative would
    be read as a ghost unit by the assignment (hand_object_assign.units_of fails)."""
    if o.get("articulation_group"):
        o["articulation_group"] = {**o["articulation_group"], "parts": [o["name"]], "base": o["name"], "role": "base"}


def _fixed_task_info(ti, body, members, aid, A_old, A_new, res, spec_sha, nq) -> None:
    """Rigid-joint task_info: parts removed, body = the merged mesh asset (obj_id · mesh_dir · inherits the main-object
    role)."""
    gone = {o["name"] for o in members}
    new_objs = _fixed_objects(ti, body, members, gone, aid, A_new, res, spec_sha)
    lay = _fixed_layout(ti, gone, new_objs, A_old, A_new)
    starts = sorted({int(o["qpos_start"]) for o in new_objs})
    for o in new_objs:
        o["slot"] = starts.index(int(o["qpos_start"]))
    ti["qpos_layout"], ti["objects"], ti["nq"] = lay, new_objs, int(nq)
    merged = next(o for o in new_objs if o["name"] == body["name"])
    _fixed_hand_targets(ti, new_objs, merged, aid)


def _fixed_objects(ti, body, members, gone, aid, A_new, res, spec_sha) -> list[dict]:
    """objects[] without the merged parts, with new qpos addresses; the body becomes the merged mesh asset and takes the
    main-object role of any merged part."""
    old_dir = posixpath.dirname(str(body.get("mesh_dir") or "data/oakink2/assets/objects/x"))
    sides = {
        x
        for o in (body, *members)
        if o.get("role") == "main" and o.get("side")
        for x in ({"right", "left"} if o["side"] == "both" else {o["side"]})
    }
    new_objs = []
    for o in ti["objects"]:
        if o["name"] in gone:
            continue
        o = dict(o)
        o["qpos_start"] = A_new[f"{o['name']}_joint"][0]
        if o["name"] == body["name"]:
            o["fixed_merge"] = {"asset": aid, "parts": [body["obj_id"], *res["fixed"]], "spec_sha": spec_sha}
            o["obj_id"], o["mesh_dir"] = aid, posixpath.join(old_dir, aid)
            _self_group(o)
            if sides:
                o["role"], o["side"] = "main", ("both" if len(sides) == 2 else next(iter(sides)))
        new_objs.append(o)
    return new_objs


def _fixed_layout(ti, gone, new_objs, A_old, A_new) -> list[dict]:
    """qpos_layout without the merged parts, every block at its new address (sorted by address)."""
    lay = []
    for e in ti.get("qpos_layout") or []:
        if e.get("kind") == "object" and e.get("name") in gone:
            continue
        e = dict(e)
        if e.get("kind") == "object":
            e["start"] = A_new[f"{e['name']}_joint"][0]
            ob = next(x for x in new_objs if x["name"] == e["name"])
            e["role"], e["side"] = ob.get("role"), ob.get("side")
        elif e.get("kind") == "hand":
            first = next(nm for nm, (st, _w) in A_old.items() if st == e["start"])
            e["start"] = A_new[first][0]
        elif e.get("kind") == "joint":
            e["start"] = A_new[e["name"]][0]
        lay.append(e)
    lay.sort(key=lambda e: e["start"])
    return lay


def _fixed_hand_targets(ti, new_objs, merged, aid) -> None:
    """Per hand: the main-object slot follows the new slots; when the merged object is that hand's main object, its
    mesh and convex folders become the asset's."""
    for s in ("right", "left"):
        tgt = next((o for o in new_objs if o.get("role") == "main" and o.get("side") in (s, "both")), None)
        if ti.get(f"{s}_object_slot") is not None:
            if tgt is not None:
                ti[f"{s}_object_slot"] = tgt["slot"]
            else:
                ti.pop(f"{s}_object_slot", None)
        if (
            tgt is merged
        ):  # the merged object is that hand's main object; its per-hand mesh and convex folders become the asset too
            for k in (f"{s}_object_mesh_dir", f"{s}_object_convex_dir"):
                if ti.get(k):
                    ti[k] = posixpath.join(posixpath.dirname(ti[k]), aid)
