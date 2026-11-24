"""Start state (a) for the SPIDER and EgoEngine inputs: our pipeline's start rules applied to the scene, inputs only (no
SPIDER or EgoEngine code change). Each rule comes from its owner in dexengine:
  1. per-clip desk height   dz = qpos_layout.desk_z_of(task_info, tree)[0]          -> the scene's floor plane at z = dz
  2. hidden support discs   coords.compute_desk_support(tree) (objects only, invisible, reset to end) -> a static
  cylinder
                            per disc, top at the disc's capture z, thickness at least AERIAL_DISC_HEIGHT, contact pairs
                            with
                            its owner object's parts only (coords.disc_owner_side)
  3. our collision meshes   dexengine.data.objects.hull.collision_parts('taco', id), the PhysX-cooked hulls Isaac
  simulates, replace each
                            object's convex parts; every contact pair of an old part is re-made for each new part
  4. frame-0 finger overlap is not touched (those clips stay SPIDER start-check failures)
The tree each clip was copied from (<data root>/data/taco/base/sharpa_wave/bimanual/<clip>) must be byte-identical to
the
input (kinematic trajectory and scene).
usage: prep_start_a.py <clips.tsv> <src dataset root> <out dataset root>   -> one [START_A] line per clip
       (the dexengine Python; the data root is dexengine.core.paths.main_root())
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

from dexengine.core import layout, paths
from dexengine.core.qpos_layout import desk_z_of
from dexengine.data.objects import hull as H
from dexengine.placement.coords import core as C
from experiments import workspace
from experiments.baselines.common.mjcf_pairs import remake_pairs

SUB = "processed/taco/sharpa_wave/bimanual"
# the disc thickness owner imports torch / Isaac Lab, so its literal is read by name
DISC_OWNER = workspace.REPO / "dexengine/rl/env_setup/build.py"


def owner_const(path, name):
    """A module-level literal read by name from its owner file."""
    for n in ast.parse(Path(path).read_text()).body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name for t in n.targets):
            return ast.literal_eval(n.value)
    raise SystemExit(f"[START_A] {name} not found in {path}")


def sha(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def transform(xml_path, out_path, dz, discs, hulls):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    asset, world = root.find("asset"), root.find("worldbody")
    contact = root.find("contact")
    floor = [g for g in world.findall("geom") if g.get("name") == "floor"]
    assert len(floor) == 1, "one floor plane"
    floor[0].set("pos", f"0 0 {dz:.6f}")
    newmap, mesh_drop = {}, set()
    for side, (oid, parts) in hulls.items():
        body = [b for b in world.iter("body") if b.get("name") == f"{side}_object"]
        assert len(body) == 1, side
        olds = [
            g
            for g in body[0].findall("geom")
            if g.get("type") == "mesh"
            and g.get("mesh", "").startswith(f"{side}_")
            and g.get("mesh") != f"{side}_visual"
        ]
        assert olds, f"{side}: no collision parts"
        tmpl = dict(olds[0].attrib)
        new = []
        for g in olds:
            mesh_drop.add(g.get("mesh"))
            body[0].remove(g)
        for k in range(len(parts)):
            ET.SubElement(asset, "mesh", {"name": f"{side}_h{k}", "file": f"objects_ours/{oid}/hull_{k}.obj"})
            a = dict(tmpl)
            a["name"] = f"{side}_object_h{k}"
            a["mesh"] = f"{side}_h{k}"
            body[0].insert(1 + k, ET.Element("geom", a))
            new.append(a["name"])
        for g in olds:
            newmap[g.get("name")] = new
    for m in list(asset.findall("mesh")):
        if m.get("name") in mesh_drop:
            asset.remove(m)
    pairs, _ = remake_pairs(contact, newmap)
    floor_attr = {}
    for q in pairs:
        if q["geom1"] == "floor":
            floor_attr.setdefault(
                q["geom2"].split("_object")[0], {k: v for k, v in q.items() if k not in ("geom1", "geom2", "name")}
            )
    for i, d in enumerate(discs):
        x, y, ztop, r, h = d["x"], d["y"], d["z_capture"], d["r"], d["h"]
        nm = f"support_disc_{i}"
        ET.SubElement(
            world,
            "geom",
            {
                "name": nm,
                "type": "cylinder",
                "size": f"{r:.5f} {h / 2:.5f}",
                "pos": f"{x:.6f} {y:.6f} {ztop - h / 2:.6f}",
                "contype": "0",
                "conaffinity": "0",
                "group": "5",
                "rgba": "0 0 0 0",
            },
        )
        for side in ("right", "left") if d["owner"] == "both" else (d["owner"],):
            for g in newmap.get(next(iter([o for o in newmap if o.startswith(f"{side}_object")]), ""), []):
                q = dict(floor_attr.get(side, {}))
                q.update({"geom1": nm, "geom2": g, "name": f"{nm}__{g}"})
                ET.SubElement(contact, "pair", q)
    tree.write(out_path)


def _parse_args(argv):
    """Command line: the clips TSV, the source root and the output root."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clips", help="TSV with a 'clip' column")
    ap.add_argument("src", help="source dataset root (SPIDER processed layout)")
    ap.add_argument("out", help="output dataset root")
    a = ap.parse_args(argv)
    return a


def _prepare_clip(src, c, trees, out, disc_h, A, meta):
    """One clip: link the tree files, shift to the desk height, add support discs and object hulls, record its meta."""
    s, t, d = f"{src}/{SUB}/{c}", f"{trees}/{c}", f"{out}/{SUB}/{c}"
    for f in ("0/trajectory_kinematic.npz", "scene.xml", "scene_eq.xml", "task_info.json"):
        if sha(f"{s}/{f}") != sha(f"{t}/{f}"):
            raise SystemExit(f"[START_A] {c}: {f} differs from its tree {t} — rules would not describe this input")
    ti = json.loads(Path(f"{t}/task_info.json").read_text())
    dz, desk_src = desk_z_of(ti, t)
    dz = float(dz)
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
            "gap_m": x.get("gap_m"),
            "kind": x.get("kind"),
        }
        for x in sup["discs"]
    ]
    xml = Path(f"{s}/scene.xml").read_text()
    hulls = {}
    for side in ("right", "left"):
        ids = sorted({f.split("/")[1] for f in re.findall(rf'name="{side}_\d+" file="(objects/[^/]+)/convex/', xml)})
        if not ids:
            continue
        assert len(ids) == 1, (c, side, ids)
        cp = H.collision_parts("taco", ids[0], False)
        ((_, parts),) = cp.links.items()
        hd = f"{A}/objects_ours/{ids[0]}"
        os.makedirs(hd, exist_ok=True)
        for k, v in enumerate(parts):
            f = f"{hd}/hull_{k}.obj"
            if not os.path.exists(f):
                trimesh.convex.convex_hull(np.asarray(v, float)).export(f)
        hulls[side] = (ids[0], parts)
    os.makedirs(f"{d}/0", exist_ok=True)
    for f in os.listdir(f"{s}/0"):
        if not os.path.lexists(f"{d}/0/{f}"):
            os.symlink(os.path.realpath(f"{s}/0/{f}"), f"{d}/0/{f}")
    if not os.path.lexists(f"{d}/task_info.json"):
        os.symlink(os.path.realpath(f"{s}/task_info.json"), f"{d}/task_info.json")
    for f in ("scene.xml", "scene_eq.xml"):
        transform(f"{s}/{f}", f"{d}/{f}", dz, discs, hulls)
    meta[c] = {
        "desk_z": dz,
        "desk_src": desk_src,
        "discs": discs,
        "hull_parts": {k: [v[0], len(v[1])] for k, v in hulls.items()},
        "excluded": sup.get("excluded"),
    }
    return dz, discs


def main(argv=None):
    a = _parse_args(argv)
    disc_h = float(owner_const(DISC_OWNER, "AERIAL_DISC_HEIGHT"))
    trees = paths.main_root() / layout.render("base", "taco", hand="sharpa_wave", emb="bimanual")
    with open(a.clips) as fh:
        clips = [r["clip"] for r in csv.DictReader(fh, delimiter="\t")]
    src, out = a.src, a.out
    A = f"{out}/taco_retarget/processed/taco/assets"
    os.makedirs(A, exist_ok=True)
    for sub in ("robots", "objects"):
        ln = f"{A}/{sub}"
        if not os.path.lexists(ln):
            os.symlink(os.path.relpath(os.path.realpath(f"{src}/taco_retarget/processed/taco/assets/{sub}"), A), ln)
    ln = f"{out}/processed/taco/assets"
    os.makedirs(os.path.dirname(ln), exist_ok=True)
    if not os.path.lexists(ln):
        os.symlink("../../taco_retarget/processed/taco/assets", ln)
    meta, nd, done = {}, 0, 0
    for c in clips:
        dz, discs = _prepare_clip(src, c, trees, out, disc_h, A, meta)
        nd += len(discs)
        done += 1
        print(f"[START_A] {c} desk_z={dz:+.4f} discs={len(discs)} hulls={meta[c]['hull_parts']}", flush=True)
    with open(f"{out}/start_a_meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"[START_A] DONE {done}/{len(clips)} clips · discs {nd} · disc h {disc_h} · meta {out}/start_a_meta.json")
    sys.exit(0 if done == len(clips) else 1)


if __name__ == "__main__":
    main()
