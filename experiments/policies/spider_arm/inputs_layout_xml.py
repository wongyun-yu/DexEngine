"""SPIDER input step `layout`, scene.xml side: one clip's scene rewritten into SPIDER's qpos layout (welded extras ->
static bodies, free 'moving' extras -> fixed children, frozen hinges, leftover bodies removed, empty-object bodies),
run by SPIDER's python. The caller is inputs_layout.cmd_layout.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from experiments.policies.spider_arm.inputs_common import die

ATTACH_MAX_M = 0.02  # a free extra becomes a fixed child of the one main object it stays within this of


def _set_pose(el, pos, quat, where):
    for k in ("euler", "axisangle", "xyaxes", "zaxis"):
        if el.get(k) is not None:
            die(f"{where}: body {el.get('name')} gives its orientation as {k} — the layout step writes quat only")
    el.set("pos", " ".join(repr(float(x)) for x in pos))
    el.set("quat", " ".join(repr(float(x)) for x in quat))


def _vec(s, default):
    return [float(x) for x in s.split()] if s is not None else list(default)


def _empty_slot_body(side):
    """SPIDER's own empty-object body (spider/preprocess/generate_xml.py, 'if left object is not present'): free joint
    with armature 1 and frictionloss 1, gravcomp 1, a 0.1 m sphere of density 10 at (0.5, 0.5, 0.5) for mass, the
    object site and its trace site. One change: the sphere has contype = conaffinity = 0 (it is a mass, never a
    contact)."""
    n = f"{side}_object"
    el = ET.Element("body", {"name": n, "pos": "0 0 0", "quat": "1 0 0 0", "gravcomp": "1"})
    ET.SubElement(el, "joint", {"name": f"{n}_joint", "type": "free", "armature": "1", "frictionloss": "1"})
    ET.SubElement(
        el,
        "geom",
        {
            "name": f"{n}_mass",
            "type": "sphere",
            "size": "0.1",
            "pos": "0.5 0.5 0.5",
            "density": "10",
            "group": "3",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        el,
        "site",
        {"name": n, "type": "box", "size": "0.01 0.02 0.03", "pos": "0 0 0", "rgba": "1 0 0 1", "group": "3"},
    )
    ET.SubElement(el, "site", {"name": f"trace_{n}", "pos": "0 0 0", "size": "0.01", "rgba": "0 1 0 1", "group": "4"})
    return el


def body_name(m, b):
    import mujoco

    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)


def _drop_free(body_el, top, n, clip):
    el = body_el[n]
    js = [c for c in el if c.tag == "freejoint" or (c.tag == "joint" and c.get("type") == "free")]
    if len(js) != 1 or n not in top:
        die(f"{clip}: {n} is not a worldbody child with one free joint")
    el.remove(js[0])
    return el


def _make_static(root, body_el, top, static, extras, q, info, clip):
    """Welded extras -> static bodies at their frame-0 pose; their welds are removed."""
    import numpy as np

    for n, a in static:
        el = _drop_free(body_el, top, n, clip)
        _set_pose(el, q[0, a : a + 3], q[0, a + 3 : a + 7], clip)
        info["static"][n] = {
            "kind": extras[n],
            "src_qpos": a,
            "ref_motion_max_m": float(np.linalg.norm(q[:, a : a + 3] - q[0, a : a + 3], axis=1).max()),
        }
    sset = {n for n, _ in static}
    eq = root.find("equality")
    gone = []
    if eq is not None:
        for e in list(eq):
            if e.get("body1") in sset or e.get("body2") in sset:
                if e.tag != "weld":
                    die(f"{clip}: {e.tag} on a welded extra")
                eq.remove(e)
                gone.append(e.get("body1") if e.get("body1") in sset else e.get("body2"))
        if len(eq) == 0:
            root.remove(eq)
    if sorted(gone) != sorted(sset):
        die(f"{clip}: welds removed for {sorted(gone)}, welded extras {sorted(sset)}")


def _attach_moving(wb, body_el, top, attach, slot_src, present, q, info, clip):
    """Free 'moving' extras -> fixed children of the one main object each stays within ATTACH_MAX_M of."""
    import mujoco
    import numpy as np

    for n, a in attach.items():
        dist = {
            s: float(np.linalg.norm(q[:, a : a + 3] - q[:, slot_src[s][0] : slot_src[s][0] + 3], axis=1).max())
            for s in present
        }
        near = [s for s, x in dist.items() if x <= ATTACH_MAX_M]
        if len(near) != 1:
            die(
                f"{clip}: moving extra {n} max distance to the main objects {dist} — no single one within "
                f"{ATTACH_MAX_M} m to attach it to"
            )
        s = near[0]
        am = slot_src[s][0]
        qi = np.zeros(4)
        mujoco.mju_negQuat(qi, q[0, am + 3 : am + 7])
        prel, qrel = np.zeros(3), np.zeros(4)
        mujoco.mju_rotVecQuat(prel, q[0, a : a + 3] - q[0, am : am + 3], qi)
        mujoco.mju_mulQuat(qrel, qi, q[0, a + 3 : a + 7])
        el = _drop_free(body_el, top, n, clip)
        _set_pose(el, prel, qrel, clip)
        wb.remove(el)
        body_el[s].append(el)
        info["attached"][n] = {"to": s, "src_qpos": a, "max_dist_m": dist[s]}


def _freeze_hinges(body_el, hinge, q, info, clip):
    """Articulated objects' hinges (<main>_top) -> fixed at their frame-0 angle."""
    import mujoco
    import numpy as np

    for n, (jn, a, ref0) in hinge.items():
        el = body_el[n]
        jel = [c for c in el if c.tag == "joint" and c.get("name") == jn]
        if len(jel) != 1 or jel[0].get("class") is not None:
            die(f"{clip}: hinge {jn} not found once (or uses a class) in body {n}")
        jel = jel[0]
        jp = np.array(_vec(jel.get("pos"), (0, 0, 0)))
        ax = np.array(_vec(jel.get("axis"), (0, 0, 1)))
        ax = ax / np.linalg.norm(ax)
        ang = float(q[0, a]) - ref0
        ql = np.zeros(4)
        mujoco.mju_axisAngle2Quat(ql, ax, ang)
        bp = np.array(_vec(el.get("pos"), (0, 0, 0)))
        bq = np.array(_vec(el.get("quat"), (1, 0, 0, 0)))
        rj, off, qn = np.zeros(3), np.zeros(3), np.zeros(4)
        mujoco.mju_rotVecQuat(rj, jp, ql)
        mujoco.mju_rotVecQuat(off, jp - rj, bq)
        mujoco.mju_mulQuat(qn, bq, ql)
        el.remove(jel)
        _set_pose(el, bp + off, qn, clip)
        info["hinge_frozen"][n] = {
            "joint": jn,
            "src_qpos": a,
            "angle_rad": ang,
            "ref_range_deg": float(np.degrees(np.ptp(q[:, a]))),
        }


def _remove_ghosts(root, wb, body_el, top, ghosts, m0, info, clip):
    """Jointless <side>_object leftovers that task_info does not list -> removed with their pairs."""
    import mujoco

    parent = {c: pe for pe in wb.iter() for c in pe}
    for g in ghosts:
        el = body_el.get(g)
        if el is None or any(c.tag == "body" for c in el):
            die(f"{clip}: leftover {g} is not a body without child bodies")
        gg = {c.get("name") for c in el if c.tag == "geom"}
        if None in gg:
            die(f"{clip}: leftover {g} has an unnamed geom")
        for sec in ("equality", "sensor", "tendon", "actuator", "exclude"):
            s_el = root.find(sec)
            if s_el is not None and any(v in gg | {g} for e in s_el.iter() for v in e.attrib.values()):
                die(f"{clip}: leftover {g} is referenced in <{sec}>")
        cont = root.find("contact")
        k = 0
        for e in list(cont):
            if e.tag == "pair" and (e.get("geom1") in gg or e.get("geom2") in gg):
                cont.remove(e)
                k += 1
        bid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, g)
        info["ghosts_removed"][g] = {
            "pairs": k,
            "mass": float(m0.body_mass[bid]),
            "parent": body_name(m0, int(m0.body_parentid[bid])),
            "pos": [round(float(x), 4) for x in m0.body_pos[bid]],
            "meshes": sorted({c.get("mesh") for c in el if c.get("mesh")}),
        }
        parent[el].remove(el)
        top.pop(g, None)


def _insert_empty(wb, top, empty):
    """A missing side -> SPIDER's empty-object body, right before left."""
    kids = list(wb)
    for s in empty:
        if s == "right_object":
            at = kids.index(top["left_object"]) if "left_object" in top else len(kids)
        else:
            at = kids.index(top["right_object"]) + 1
        wb.insert(at, _empty_slot_body(s.split("_")[0]))
        kids = list(wb)


def _drop_static_pairs(root, m1, info, clip):
    """Pairs between two static bodies touch no degree of freedom; drop them (they only use contact budget).
    Returns True when the scene changed (the caller writes and loads it again)."""
    import mujoco

    st = [
        i
        for i in range(m1.npair)
        if int(m1.body_weldid[int(m1.geom_bodyid[int(m1.pair_geom1[i])])]) == 0
        and int(m1.body_weldid[int(m1.geom_bodyid[int(m1.pair_geom2[i])])]) == 0
    ]
    if not st:
        return False
    names = {mujoco.mj_id2name(m1, mujoco.mjtObj.mjOBJ_PAIR, i) for i in st}
    gnames = {
        (
            mujoco.mj_id2name(m1, mujoco.mjtObj.mjOBJ_GEOM, int(m1.pair_geom1[i])),
            mujoco.mj_id2name(m1, mujoco.mjtObj.mjOBJ_GEOM, int(m1.pair_geom2[i])),
        )
        for i in st
    }
    cont = root.find("contact")
    k = 0
    for e in list(cont):
        if e.tag == "pair" and (
            (e.get("name") in names and e.get("name") is not None) or (e.get("geom1"), e.get("geom2")) in gnames
        ):
            cont.remove(e)
            k += 1
    if k != len(st):
        die(f"{clip}: {len(st)} static-static pairs, removed {k}")
    info["pairs_static_removed"] = k
    return True


def rewrite_xml_layout(clip, d, p, m0, q, extras, slot_src, present, static, attach, hinge, ghosts, empty, info):
    """Rewrite scene.xml (p) into SPIDER's layout at <d>/_layout_scene.xml; fills info; returns the new MjModel."""
    import mujoco

    tree = ET.parse(p)
    root = tree.getroot()
    wb = root.find("worldbody")
    body_el = {b.get("name"): b for b in wb.iter("body")}
    top = {c.get("name"): c for c in wb if c.tag == "body"}
    _make_static(root, body_el, top, static, extras, q, info, clip)
    _attach_moving(wb, body_el, top, attach, slot_src, present, q, info, clip)
    _freeze_hinges(body_el, hinge, q, info, clip)
    _remove_ghosts(root, wb, body_el, top, ghosts, m0, info, clip)
    _insert_empty(wb, top, empty)
    tmp = f"{d}/_layout_scene.xml"
    tree.write(tmp)
    m1 = mujoco.MjModel.from_xml_path(tmp)
    if _drop_static_pairs(root, m1, info, clip):
        tree.write(tmp)
        m1 = mujoco.MjModel.from_xml_path(tmp)
    return m1
