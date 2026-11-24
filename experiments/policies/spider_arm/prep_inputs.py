"""SPIDER inputs for the E5 'SPIDER + arm' data source: the 60 E5 Wuji 2 clips, five datasets,
bimanual and right-hand scenes. The rules are those of the tab:same64 SPIDER inputs (baselines/common/prep_start_a.py,
then baselines/common/prep_mass_0p2.py), applied per clip instead of to one TACO x SharpA x bimanual folder:

  start  (the retarget python: needs trimesh and this repo)
    (1) floor plane at the clip's desk height        qpos_layout.desk_z_of(task_info, tree)[0]
    (2) hidden support discs                         coords.compute_desk_support(tree, task_info), same disc rule as
                                                     prep_start_a (static cylinder, contact pairs with its owner's
                                                     object geoms only, thickness = AERIAL_DISC_HEIGHT)
    (3) our collision shapes                         dexengine.data.objects.hull.collision_parts(dataset, obj_id,
    articulated) for every
                                                     object in task_info['objects'] (main, extra, support); a rigid
                                                     object has one link, an articulated one its root link first, then
                                                     its other link, mapped to the bodies <name> and <name>_top.
                                                     A body's collision geoms = its group-3 geoms (the scene generator's
                                                     collision group; checked equal to the geoms in a contact <pair>
                                                     where the body has pairs); each is replaced by the cooked hulls and
                                                     every pair of an old geom is re-made per new geom.
    (3b) extra objects, by Isaac's own rule        extra_slots.clip_rows(tree, task_info) gives each role-extra object a
                                                     kind; its partners are collision_presets.EXTRA_KIND_COLLIDE:
                                                     receiver = fixed at its frame-0 pose (MuJoCo weld), collides with
                                                     the main objects only; touched = fixed, collides with nothing;
                                                     moving = Isaac moves it kinematically along the reference, which
                                                     SPIDER's MuJoCo scene cannot: kept free, with the main object's
                                                     hand
                                                     and floor pairs (recorded per clip as 'moving_free'). Kind None =
                                                     not an Isaac slot: the scene file's own pairs stay (TACO extras).
                                                     A role-support object follows collision_presets.SUPPORT_COLLIDE
                                                     (hand, arm, objects ON; desk OFF): fixed at its frame-0 pose, the
                                                     main objects' hand pairs and object pairs, no floor pair.
  mass   (the method's python: code/spider/.venv/bin/python -I, its own mujoco)
    (4) every rigid link of every object = DEFAULT_OBJECT_MASS_KG (the repo's owner, read by name). Isaac gives every
        object that mass through modify_mass_properties, which is @apply_nested, so each link of an articulated object
        gets it. Done as prep_mass_0p2 does it: a uniform density rescale of the link's mass-bearing geoms.
  layout SPIDER's qpos layout (robot, free right object, free left object): see cmd_layout.
  check  per clip: object link masses, floor z, hull counts, every referenced file exists, SPIDER's layout.

usage — from the repo root, with the retarget python for start and SPIDER's python for the rest; -s -E keeps SPIDER's
python isolated from user site packages and PYTHON* variables while the repo root (cwd) stays importable (the
inputs_* modules use the standard library only at import):
  <retarget python> -m experiments.policies.spider_arm.prep_inputs start  <clips.tsv> <out root> --data <data root>
  <spider python> -s -E -m experiments.policies.spider_arm.prep_inputs mass   <clips.tsv> <out root> --data <data root>
  <spider python> -s -E -m experiments.policies.spider_arm.prep_inputs layout <clips.tsv> <out root> --data <data root>
  <spider python> -s -E -m experiments.policies.spider_arm.prep_inputs check  <clips.tsv> <out root> --data <data root>
  (clips.tsv: clip, dataset, embodiment, task; layout = SPIDER's qpos layout, inputs_layout.cmd_layout)
Out layout = SPIDER's:
<out>/processed/<dataset>/wuji2/<embodiment>/<clip>/{scene.xml,scene_eq.xml,task_info.json,0/*}
the source's asset folder mirrored under <out> at the same path relative to the data root (robots/objects = relative
symlinks into it; hulls = real files in its objects_ours/), and each scene's meshdir rewritten to that mirror (the out
clip folder is one level shallower than data/<ds>/base/..., so the source's relative meshdir would point elsewhere).
Each step ends with a count line; fewer clips done than listed = rc 1. No step overwrites an existing clip.
"""

from __future__ import annotations

import ast
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from experiments.baselines.common.mjcf_pairs import remake_pairs
from experiments.policies.spider_arm import inputs_common as IC
from experiments.policies.spider_arm.inputs_common import (
    OV,
    ROBOT,
    SCENES,
    articulated,
    die,
    meshdir_root,
    out_of,
    owner_const,
    read_clips,
    sha,
    task_objects,
    tree_of,
)
from experiments.policies.spider_arm.inputs_layout import cmd_layout
from experiments.policies.spider_arm.inputs_mass import cmd_check, cmd_mass


# ------------------------------------------------------------------------------------------------------------- start
def _set_meshdir(root, xml_in, meshdir):
    comp = root.findall("compiler")
    if len(comp) != 1 or comp[0].get("meshdir") is None:
        die(f"{xml_in}: needs one <compiler meshdir=...>")
    for k in ("texturedir", "assetdir"):  # every asset path of the compiler follows the same mirror
        v = comp[0].get(k)
        if v is not None and v != comp[0].get("meshdir"):
            die(f"{xml_in}: {k}={v!r} differs from meshdir — one mirror cannot serve both")
        if v is not None:
            comp[0].set(k, meshdir)
    comp[0].set("meshdir", meshdir)


def _swap_hulls(root, xml_in, asset, contact, bodies, hulls):
    """Each body's group-3 geoms -> its convex hull parts; returns ({old geom: new geoms}, {body: new geoms})."""
    pair_geoms = {p.get(k) for p in contact.findall("pair") for k in ("geom1", "geom2")}
    newmap, mesh_drop, body_new = {}, set(), {}
    for bname, (hdir, n) in hulls.items():
        if bname not in bodies:
            die(f"{xml_in}: no body {bname}")
        body = bodies[bname]
        olds = [g for g in body.findall("geom") if g.get("group") == "3"]
        paired = [g for g in body.findall("geom") if g.get("name") in pair_geoms]
        if not olds:
            die(f"{xml_in}: body {bname} has no group-3 (collision) geom")
        if paired and paired != olds:
            die(f"{xml_in}: body {bname}: geoms in pairs {[g.get('name') for g in paired]} != group-3 geoms")
        tmpl = dict(olds[0].attrib)
        first = list(body).index(olds[0])
        for g in olds:
            if g.get("mesh"):
                mesh_drop.add(g.get("mesh"))
            body.remove(g)
        new = []
        for k in range(n):
            mname = f"{bname}_h{k}"
            ET.SubElement(asset, "mesh", {"name": mname, "file": f"{hdir}/hull_{k}.obj"})
            a = dict(tmpl)
            a["name"], a["mesh"], a["type"] = f"{bname}_hull{k}", mname, "mesh"
            for drop in ("size", "fromto"):
                a.pop(drop, None)
            body.insert(first + k, ET.Element("geom", a))
            new.append(a["name"])
        for g in olds:
            newmap[g.get("name")] = new
        body_new[bname] = new
    still = {g.get("mesh") for g in root.iter("geom") if g.get("mesh")}
    for m in list(asset.findall("mesh")):
        if m.get("name") in mesh_drop and m.get("name") not in still:
            asset.remove(m)
    return newmap, body_new


def _floor_attrs(pairs, body_new):
    """The floor pair of an object carries its contact parameters; a disc's pairs copy them (prep_start_a rule)."""
    floor_attr = {}
    for q in pairs:
        if "floor" in (q["geom1"], q["geom2"]):
            other = q["geom2"] if q["geom1"] == "floor" else q["geom1"]
            for bname, geoms in body_new.items():
                if other in geoms:
                    floor_attr.setdefault(bname, {k: v for k, v in q.items() if k not in ("geom1", "geom2", "name")})
    return floor_attr


def _add_discs(xml_in, world, contact, discs, body_new, floor_attr):
    """Support discs (invisible, no own contact) with pairs to their owners' hull parts; returns the pair count."""
    nd_pairs = 0
    for i, d in enumerate(discs):
        nm = f"support_disc_{i}"
        ET.SubElement(
            world,
            "geom",
            {
                "name": nm,
                "type": "cylinder",
                "size": f"{d['r']:.5f} {d['h'] / 2:.5f}",
                "pos": f"{d['x']:.6f} {d['y']:.6f} {d['z_capture'] - d['h'] / 2:.6f}",
                "contype": "0",
                "conaffinity": "0",
                "group": "5",
                "rgba": "0 0 0 0",
            },
        )
        sides = ("right", "left") if d["owner"] == "both" else (d["owner"],)
        owners = [b for b in body_new if any(b in (f"{s}_object", f"{s}_object_top") for s in sides)]
        if not owners:
            die(f"{xml_in}: disc {i} owner {d['owner']} has no object body")
        for bname in owners:
            if bname not in floor_attr:
                die(f"{xml_in}: {bname} has no floor pair to copy the disc contact from")
            for g in body_new[bname]:
                q = dict(floor_attr[bname])
                q.update({"geom1": nm, "geom2": g, "name": f"{nm}__{g}"})
                ET.SubElement(contact, "pair", q)
                nd_pairs += 1
    return nd_pairs


def _copy_main_pairs(contact, main_geoms, body_new, mains, seen, bname, floor_too):
    n = 0
    for p in [p for p in contact.findall("pair") if (p.get("geom1") in main_geoms) != (p.get("geom2") in main_geoms)]:
        other = p.get("geom2") if p.get("geom1") in main_geoms else p.get("geom1")
        if other == "floor" and not floor_too:
            continue
        if any(other in v for k, v in body_new.items() if k not in mains):
            continue
        for a in body_new[bname]:
            if (a, other) in seen:
                continue
            seen.add((a, other))
            q = dict(p.attrib)
            q.update({"geom1": a, "geom2": other, "name": f"{a}__{other}"})
            ET.SubElement(contact, "pair", q)
            n += 1
    return n


def _add_extras(xml_in, root, contact, extras, frame0, mains, bodies, body_new, floor_attr, seen):
    """(3b) extras by kind: welded at frame 0 (receiver / touched / support) or free (moving), with their pairs;
    returns (pair count, {body: SPIDER kind})."""
    eq = root.find("equality")
    n_extra_pairs, how = 0, {}
    main_geoms = [g for m in mains for g in body_new.get(m, [])]
    for bname, kind in extras.items():
        if kind in ("receiver", "touched", "support"):
            pos, quat = frame0[bname]
            bodies[bname].set("pos", " ".join(f"{x:.9g}" for x in pos))
            bodies[bname].set("quat", " ".join(f"{x:.9g}" for x in quat))
            if eq is None:
                eq = ET.SubElement(root, "equality")
            ET.SubElement(eq, "weld", {"name": f"fix_{bname}", "body1": bname})
            how[bname] = f"{kind}_welded"
            if kind == "support":
                n_extra_pairs += _copy_main_pairs(contact, main_geoms, body_new, mains, seen, bname, floor_too=False)
            if kind in ("receiver", "support"):
                for m in mains:
                    if m not in floor_attr:
                        die(f"{xml_in}: main {m} has no floor pair to copy the receiver contact from")
                    for a in body_new[bname]:
                        for b in body_new[m]:
                            q = dict(floor_attr[m])
                            q.update({"geom1": a, "geom2": b, "name": f"{a}__{b}"})
                            ET.SubElement(contact, "pair", q)
                            n_extra_pairs += 1
        elif kind == "moving":
            n_extra_pairs += _copy_main_pairs(contact, main_geoms, body_new, mains, seen, bname, floor_too=True)
            how[bname] = "moving_free"
        else:
            die(f"{xml_in}: extra {bname} kind {kind!r} has no SPIDER rule")
    return n_extra_pairs, how


def transform(xml_in, xml_out, dz, discs, hulls, extras, frame0, mains, meshdir):
    """hulls: {body name: (hull dir relative to meshdir, n parts)}; extras: {body: kind}; frame0: {body: (pos, quat)};
    mains: the main object bodies (root links)."""
    tree = ET.parse(xml_in)
    root = tree.getroot()
    asset, world, contact = root.find("asset"), root.find("worldbody"), root.find("contact")
    _set_meshdir(root, xml_in, meshdir)
    floor = [g for g in world.findall("geom") if g.get("name") == "floor"]
    if len(floor) != 1:
        die(f"{xml_in}: {len(floor)} floor planes")
    floor[0].set("pos", f"0 0 {dz:.6f}")
    bodies = {b.get("name"): b for b in world.iter("body")}
    newmap, body_new = _swap_hulls(root, xml_in, asset, contact, bodies, hulls)
    pairs, seen = remake_pairs(contact, newmap)
    floor_attr = _floor_attrs(pairs, body_new)
    nd_pairs = _add_discs(xml_in, world, contact, discs, body_new, floor_attr)
    n_extra_pairs, how = _add_extras(xml_in, root, contact, extras, frame0, mains, bodies, body_new, floor_attr, seen)
    tree.write(xml_out)
    return {"pairs_remade": len(pairs), "disc_pairs": nd_pairs, "extra_pairs": n_extra_pairs, "extras": how}


def _prepare_clip_dir(r, t, d, out):
    """Check the clip (new dir, robot/embodiment match, meshdir under the data root); link its mesh assets."""
    if os.path.lexists(d):
        die(f"{d} exists — a derived clip is written once")
    xml = Path(f"{t}/scene.xml").read_text()
    ti = json.loads(Path(f"{t}/task_info.json").read_text())
    if ti.get("robot_type") != ROBOT or ti.get("embodiment_type") != r["embodiment"]:
        die(f"{r['clip']}: task_info robot/embodiment {ti.get('robot_type')}/{ti.get('embodiment_type')} != list")
    src_A = meshdir_root(t, xml)
    rel = os.path.relpath(src_A, IC.DATA)
    if rel.startswith(".."):
        die(f"{r['clip']}: meshdir {src_A} is outside {IC.DATA}")
    A = os.path.normpath(f"{out}/{rel}")
    meshdir_new = os.path.relpath(A, d) + "/"
    os.makedirs(A, exist_ok=True)
    for sub in ("robots", "objects"):
        link = f"{A}/{sub}"
        if not os.path.lexists(link):
            os.symlink(os.path.relpath(os.path.realpath(f"{src_A}/{sub}"), A), link)
    # task_info *_mesh_dir paths are relative to the dataset root: processed/<ds>/assets must resolve to A
    ti_assets = f"{out}/processed/{r['dataset']}/assets"
    if not os.path.lexists(ti_assets):
        os.makedirs(os.path.dirname(ti_assets), exist_ok=True)
        os.symlink(os.path.relpath(A, os.path.dirname(ti_assets)), ti_assets)
    if os.path.realpath(ti_assets) != os.path.realpath(A):
        die(f"{r['clip']}: {ti_assets} does not resolve to the scene's meshdir {A}")
    return ti, A, meshdir_new


def _desk_discs(C, t, ti, disc_h):
    """Desk height of the clip and its support discs in capture coordinates (height at least the aerial disc height)."""
    dz = float(C._qpos_layout_mod().desk_z_of(ti, t)[0])
    off = C.desk_height() - dz
    sup = C.compute_desk_support(t, ti)
    discs = [
        {
            "x": float(x["x"]),
            "y": float(x["y"]),
            "z_capture": float(x["z"]) - off,
            "r": float(x["r"]),
            "h": max(float(x.get("h", 0.0)), disc_h),
            "owner": C.disc_owner_side(ti, x),
            "obj": x.get("obj"),
            "kind": x.get("kind"),
        }
        for x in sup["discs"]
    ]
    return dz, discs


def _extra_kinds(np, X, r, t, ti):
    """Frame-0 object poses and the Isaac slot kind of every non-main object."""
    kinds = {x["name"]: x.get("kind") for x in X.clip_rows(t, ti)}
    lay = ti["qpos_layout"]
    lay = ast.literal_eval(lay) if isinstance(lay, str) else lay
    q0 = np.load(f"{t}/0/trajectory_kinematic.npz")["qpos"][0]
    frame0 = {
        e["name"]: (q0[e["start"] : e["start"] + 3], q0[e["start"] + 3 : e["start"] + 7])
        for e in lay
        if e.get("width") == 7
    }
    objs = task_objects(ti)
    mains = [o["name"] for o in objs if o.get("role") == "main"]
    extras = {}
    for o in objs:
        if o.get("role") == "main":
            continue
        k = "support" if o.get("role") == "support" else kinds.get(o["name"], "absent")
        if k == "absent":
            die(f"{r['clip']}: {o['name']} (role {o.get('role')}) not classified by extra_slots.clip_rows")
        if k is None:
            continue  # not an Isaac slot: the scene file's own pairs stay
        if k in ("receiver", "touched", "support") and o["name"] not in frame0:
            die(f"{r['clip']}: no frame-0 pose for {o['name']} in qpos_layout")
        extras[o["name"]] = k
    return frame0, objs, mains, extras


def _write_hulls(np, trimesh, H, r, ti, objs, A):
    """Write the cooked convex hulls of every object link under objects_ours; returns body → (dir, count), meta rows."""
    hulls, hull_meta = {}, {}
    for o in objs:
        name, oid, art = o["name"], str(o["obj_id"]), articulated(ti, o["name"])
        cp = H.collision_parts(r["dataset"], oid, art)
        links = list(cp.links.items())
        if len(links) != (2 if art else 1):
            die(f"{r['clip']}: {name} ({oid}) has {len(links)} cooked links, articulated={art}")
        targets = [name] if not art else [name, f"{name}_top"]
        for (ln, parts), bname in zip(links, targets, strict=True):
            hd_rel = f"objects_ours/{r['dataset']}/{oid}/{ln}"
            hd = f"{A}/{hd_rel}"
            os.makedirs(hd, exist_ok=True)
            for k, v in enumerate(parts):
                f = f"{hd}/hull_{k}.obj"
                if not os.path.exists(f):
                    trimesh.convex.convex_hull(np.asarray(v, float)).export(f)
            hulls[bname] = (hd_rel, len(parts))
            hull_meta[bname] = [oid, ln, len(parts)]
    return hull_meta, hulls


def _link_and_transform(d, t, dz, discs, hulls, extras, frame0, mains, meshdir_new):
    """Link the source clip rollout and task_info; write the transformed scenes; returns the per-scene pairs."""
    os.makedirs(f"{d}/0")
    for f in os.listdir(f"{t}/0"):
        os.symlink(os.path.realpath(f"{t}/0/{f}"), f"{d}/0/{f}")
    os.symlink(os.path.realpath(f"{t}/task_info.json"), f"{d}/task_info.json")
    tx = {f: transform(f"{t}/{f}", f"{d}/{f}", dz, discs, hulls, extras, frame0, mains, meshdir_new) for f in SCENES}
    return tx


def cmd_start(clips_tsv, out):
    import numpy as np
    import trimesh

    from dexengine.data.objects import extra_slots as X
    from dexengine.data.objects import hull as H
    from dexengine.placement.coords import core as C

    disc_h = float(owner_const(f"{OV}/dexengine/rl/env_setup/build.py", "AERIAL_DISC_HEIGHT"))
    rows = read_clips(clips_tsv)
    meta, done = {}, 0
    for r in rows:
        t, d = tree_of(r), out_of(out, r)
        ti, A, meshdir_new = _prepare_clip_dir(r, t, d, out)
        dz, discs = _desk_discs(C, t, ti, disc_h)
        frame0, objs, mains, extras = _extra_kinds(np, X, r, t, ti)
        hull_meta, hulls = _write_hulls(np, trimesh, H, r, ti, objs, A)
        tx = _link_and_transform(d, t, dz, discs, hulls, extras, frame0, mains, meshdir_new)
        meta[r["clip"]] = {
            "dataset": r["dataset"],
            "embodiment": r["embodiment"],
            "desk_z": dz,
            "desk_src": C._qpos_layout_mod().desk_z_of(ti, t)[1],
            "discs": discs,
            "hulls": hull_meta,
            "extra_kinds": extras,
            "pairs": tx,
            "src_sha": {f: sha(f"{t}/{f}") for f in SCENES + ("task_info.json",)},
        }
        done += 1
        print(
            f"[E5_SPIDER_IN] start {r['clip']} {r['dataset']}/{r['embodiment']} desk_z={dz:+.4f} discs={len(discs)} "
            f"hulls={ {k: v[2] for k, v in hull_meta.items()} } extras={tx['scene.xml']['extras']}",
            flush=True,
        )
    mf = f"{out}/start_meta.json"
    old = json.loads(Path(mf).read_text()) if os.path.exists(mf) else {}
    old.update(meta)
    with open(mf, "w") as fh:
        json.dump(old, fh, indent=1)
    print(f"[E5_SPIDER_IN] start DONE {done}/{len(rows)} clips -> {out}")
    sys.exit(0 if done == len(rows) else 1)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="SPIDER inputs of the E5 'SPIDER + arm' rows (module docstring)")
    ap.add_argument("cmd", choices=("start", "mass", "layout", "check"))
    ap.add_argument("clips")
    ap.add_argument("out")
    ap.add_argument("--data", required=True, help="data root holding data/<dataset>/base/<robot>/... trees")
    a = ap.parse_args(argv)
    IC.DATA = os.path.abspath(a.data)
    {"start": cmd_start, "mass": cmd_mass, "layout": cmd_layout, "check": cmd_check}[a.cmd](
        os.path.abspath(a.clips), os.path.abspath(a.out)
    )


if __name__ == "__main__":
    main()
