"""Scene build half of the canon collision-set evaluator (cross_sets.CrossSetModel): the MuJoCo scene of the robot
shapes, the object shapes and groups, and the evaluated pairs. CrossSetModel inherits it; the methods read the
attributes CrossSetModel.__init__ sets.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from dexengine.placement.collision.link_shapes import (
    _adjacent,
    geom_files,
    group_of,
    mat2quat_wxyz_batch,
    part_spec,
    scene_meshes,
    xml_links,
)


class CrossSetBuild:
    """Mixin: builds the evaluator's scene (``_build``) — object shapes, object groups and the pair list."""

    # ── Building the model ───────────────────────────────────────────────────
    def _build(self, tree_dir: Path | None):
        mj = self.mj
        spec = mj.MjSpec()
        spec.compiler.degree = False
        mi, shell = 0, mj.mjtMeshInertia.mjMESH_INERTIA_SHELL  # shell: a thin USD part has ~0 volume (mocap)

        def add(body, shapes):  # one geom per shape: capsule · mesh file · USD collision part (V)
            nonlocal mi
            for c in shapes:
                kw = {
                    "pos": c.get("xyz", np.zeros(3)).tolist(),
                    "quat": mat2quat_wxyz_batch(np.asarray(c.get("R", np.eye(3)), float)[None])[0].tolist(),
                }
                if c["kind"] == "capsule":
                    g = body.add_geom(type=mj.mjtGeom.mjGEOM_CAPSULE, size=[c["r"], c["L"] / 2.0, 0.0], **kw)
                else:
                    name = f"m{mi}"
                    mi += 1
                    spec.add_mesh(
                        name=name,
                        **(
                            {"uservert": c["V"].ravel().tolist(), "inertia": shell}
                            if "V" in c
                            else {"file": c["file"], "scale": c.get("scale", [1, 1, 1])}
                        ),
                    )
                    g = body.add_geom(type=mj.mjtGeom.mjGEOM_MESH, meshname=name, **kw)
                g.contype, g.conaffinity = 0, 0
            return len(shapes)

        for ln in sorted(self.links):
            add(spec.worldbody.add_body(name=ln, mocap=True), self.links[ln])
        if tree_dir is not None:
            self._add_scene_objects(tree_dir, add, spec)
        # MuJoCo builds the hull graph (support queries by hill climbing — same values, 11.7× faster on near pairs)
        # only for meshes with contype|conaffinity set, so set them right before compiling and **reset them to 0 in the
        # compiled model** (this model never calls mj_collision — the collision bits keep their meaning).
        return self._compile_and_index(spec, mj)

    def _add_scene_objects(self, tree_dir, add, spec):
        """The main object bodies of the tree's scene.xml and their articulated parts (one mocap body each) with
        their collision shapes; a main object missing from scene.xml raises."""
        sx = ET.parse(str(tree_dir / "scene.xml")).getroot()
        mesh_root, mfile = scene_meshes(sx, self.assets)
        for body in sx.find("worldbody").findall("body"):
            nm = body.get("name")
            if nm not in {v for v in self.layout.obj_body.values() if v}:  # main object bodies only
                continue
            links = xml_links(body, det=True)  # a jointless child = a component of its parent's link (welded)
            shapes = self._obj_shapes(links, mfile, mesh_root)
            n = add(spec.worldbody.add_body(name=nm, mocap=True), shapes[nm])
            # Articulated part bodies are obstacles too — one mocap body each, placed as parent pose ∘
            # joint FK (set_pose); the joint column is layout.artic_joint's (owner qpos_layout), unknown dies.
            for child, _welded in links[1:]:
                cn = child.get("name")
                part = part_spec(child, nm, tree_dir.name, det=True)
                jn, jt = part["joint"], part["type"]
                if jn not in self.layout.artic_joint:
                    raise ValueError(
                        f"{tree_dir.name}: scene.xml joint {jn!r} (part {cn}) is not in "
                        "qpos_layout.artic — its column is unknown"
                    )
                part["col"] = self.layout.artic_joint[jn]
                npart = add(spec.worldbody.add_body(name=cn, mocap=True), shapes[cn])
                n += npart
                self.parts.append(part)
                self.part_parent[cn] = nm
                print(
                    f"[ARTIC] xr {tree_dir.name}: part {cn} ← {nm} joint={jn}({jt}) col={part['col']} "
                    f"axis={part['axis'].round(3).tolist()} anchor={part['anchor'].round(4).tolist()} "
                    f"geoms={npart} — in the obstacle set",
                    flush=True,
                )
            self.n_obj[nm] = n
        lost = sorted({b for b in self.layout.obj_body.values() if b} - set(self.n_obj))
        if lost:
            raise ValueError(f"{tree_dir.name}: main object bodies {lost} are not in scene.xml")
        src = "cooked hulls · object_hull.collision_parts" if self.obj_geom == "sim" else "scene.xml meshes"
        print(
            f"[XR_OBJ] geom={self.obj_geom} parts={sum(self.n_obj.values())} "
            f"bodies={len(self.n_obj) + len(self.parts)} ({src})",
            flush=True,
        )

    def _compile_and_index(self, spec, mj):
        """Compile the spec and index geoms, mocap bodies, link frames and the collision pairs."""
        for _g in spec.geoms:
            _g.contype, _g.conaffinity = 1, 1
        self.m = spec.compile()
        self.d = mj.MjData(self.m)
        self.m.geom_contype[:] = 0
        self.m.geom_conaffinity[:] = 0
        _ng = int((self.m.mesh_graphadr >= 0).sum())
        print(
            f"[XR_HULL] graphs={_ng}/{self.m.nmesh} links={len(self.links)} (hull graph, owner cross_sets)", flush=True
        )
        if _ng != self.m.nmesh:
            raise RuntimeError(f"[XR_HULL] the hull graph was built for only {_ng}/{self.m.nmesh} meshes")
        self.ginfo = []
        for i in range(self.m.ngeom):
            bn = self.m.body(self.m.geom_bodyid[i]).name
            self.ginfo.append((self._obj_group(bn) or group_of(bn, self.arm_spec), bn))
        self.body_mocap = {
            self.m.body(i).name: self.m.body_mocapid[i] for i in range(self.m.nbody) if self.m.body_mocapid[i] >= 0
        }
        self.link_mocap = [(ln, self.body_mocap[ln], self.frame_ids[ln]) for ln in self.links if ln in self.body_mocap]
        self.obj_mocap = {
            s: (self.body_mocap.get(self.layout.obj_body[s]) if self.layout.obj_body[s] else None)
            for s in ("right", "left")
        }
        self.pairs = self._pairs()
        self._fromto = np.zeros(6)
        # Precomputed for the pose load (only the name lookups move ahead — bit-equal to the per-element scalar path)
        self._lm_mi = np.array([mi for _, mi, _ in self.link_mocap], int)
        self._lm_fid = [fid for _, _, fid in self.link_mocap]
        _grp = [group_of(ln, self.arm_spec) for ln, _, _ in self.link_mocap]
        self._lm_rh = np.array([g == "RH" for g in _grp], bool)
        self._lm_lh = np.array([g == "LH" for g in _grp], bool)

    def _obj_shapes(self, links, mfile: dict, mesh_root) -> dict:
        """A main object body's links (xml_links, parent first) → {link body: shapes}: visual = the mesh geoms of the
        link body and of the bodies welded to it (posed in the link frame) · sim = the cooked hulls of the USD link
        matched to it (objects.hull.body_parts — a welded body is part of its link's hulls). Unbuildable dies."""

        def geoms(el):
            return geom_files(el, mfile, mesh_root)

        files = {el.get("name"): geoms(el) for el, _w in links}
        welded = {el.get("name"): [(w.get("name"), geoms(w), p, R) for w, p, R in ws] for el, ws in links if ws}
        for b, ws in welded.items():
            print(
                f"[XR_OBJ] weld {b} ← {[w for w, *_ in ws]} — a jointless child body = a component of its link "
                "(xml_links)",
                flush=True,
            )
        if self.obj_geom == "visual":
            return {
                b: [{"kind": "mesh", "file": str(Path(mesh_root) / f)} for f in fs]
                + [
                    {"kind": "mesh", "file": str(Path(mesh_root) / f), "xyz": p, "R": R}
                    for _w, wf, p, R in welded.get(b, ())
                    for f in wf
                ]
                for b, fs in files.items()
            }
        from dexengine.data.objects import hull as OH

        nm = links[0][0].get("name")
        ds, oid, art = next(v for s, v in self.layout.obj_ident.items() if self.layout.obj_body[s] == nm)
        if not ds:
            raise ValueError(f"{nm}: task_info.dataset_name missing — cannot choose the sim shape (USD)")

        def paths(fs):
            return [Path(mesh_root) / f for f in dict.fromkeys(fs)]

        parts = OH.body_parts(
            ds,
            oid,
            art,
            [(b, paths(fs)) for b, fs in files.items()],
            welded={b: [(paths(wf), p, R) for _w, wf, p, R in ws] for b, ws in welded.items()},
        )
        return {b: [{"kind": "part", "V": v} for v in vs] for b, vs in parts.items()}

    def _obj_group(self, body: str) -> str | None:
        """Main object body → OR (the right hand's main object) / OL (the left hand's) / OS (shared: the main object
        of both hands → sets ②③ have no counterpart)."""
        body = self.part_parent.get(body, body)  # a part belongs to its parent's set
        sides = {s for s, n in self.layout.obj_body.items() if n == body}
        if not sides:
            return None
        return "OS" if len(sides) == 2 else ("OR" if "right" in sides else "OL")

    def _pairs(self):
        """[(geom i, geom j, set, class)] of every pair the sets ①–④ measure."""
        by: dict[str, list[int]] = {}
        for i, (g, _) in enumerate(self.ginfo):
            by.setdefault(g, []).append(i)
        out = []
        R = by.get("RA", []) + by.get("RH", [])
        L = by.get("LA", []) + by.get("LH", [])
        out.extend(
            (
                gi,
                gj,
                "1_cross",
                f"{'hand' if self.ginfo[gi][0] == 'RH' else 'arm'}-{'hand' if self.ginfo[gj][0] == 'LH' else 'arm'}",
            )
            for gi in R
            for gj in L
        )
        out.extend((gi, gj, "2_hand_obj", "Rhand-Lobj") for gi in by.get("RH", []) for gj in by.get("OL", []))
        out.extend((gi, gj, "2_hand_obj", "Lhand-Robj") for gi in by.get("LH", []) for gj in by.get("OR", []))
        out.extend((gi, gj, "3_obj_obj", "obj-obj") for gi in by.get("OR", []) for gj in by.get("OL", []))
        for gi in R + L:
            for gj in by.get("TORSO", []):
                if _adjacent(self.ginfo[gi][1], self.ginfo[gj][1], self.parent):
                    continue
                side = "R" if self.ginfo[gi][0] in ("RA", "RH") else "L"
                c = "hand" if self.ginfo[gi][0] in ("RH", "LH") else "arm"
                out.append((gi, gj, "4_torso", f"{side}{c}-torso"))
        return out
