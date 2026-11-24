"""SPIDER input steps run by SPIDER's python: mass (uniform object link mass) and check."""

from __future__ import annotations

import html
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from experiments.policies.spider_arm.inputs_common import (
    OWNER_MASS,
    SCENES,
    articulated,
    die,
    meshdir_root,
    out_of,
    owner_const,
    read_clips,
    task_objects,
    tree_of,
)


# -------------------------------------------------------------------------------------------------------------- mass
def object_link_bodies(m, ti):
    import mujoco

    out = []
    for o in task_objects(ti):
        for n in [o["name"]] + ([f"{o['name']}_top"] if articulated(ti, o["name"]) else []):
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
            if b < 0:
                die(f"no body {n}")
            out.append((n, b))
    return out


def _parse_scene(mujoco, p, m0, ti):
    """Parse one scene XML and check the density rule applies (top default only, every free body is a task object)."""
    tree = ET.parse(p)
    root = tree.getroot()
    if any(x.get("class") for x in root.iter("default")):
        die(f"{p}: named default classes — the density rule reads the top default only")
    g0 = root.find("default").find("geom") if root.find("default") is not None else None
    dd = float(g0.get("density", 1000)) if g0 is not None else 1000.0
    bodies = {b.get("name"): b for b in root.find("worldbody").iter("body")}
    links = object_link_bodies(m0, ti)
    free_bodies = {
        int(m0.jnt_bodyid[j]) for j in range(m0.njnt) if int(m0.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
    }
    if not free_bodies <= {b for _, b in links}:
        die(f"{p}: a free body is not one of task_info's objects")
    return tree, dd, bodies, links


def _rescale_links(links, bodies, p, dd, m0, M):
    """Rescale the density of each object link's mass-bearing geoms so its mass becomes M; returns per-link records."""
    fi = {}
    for n, b in links:
        el = bodies[n]
        if el.find("inertial") is not None:
            die(f"{p}: {n} has an explicit <inertial>")
        heavy, rhos = [], set()
        for g in el.findall("geom"):
            if g.get("mass") is not None:
                die(f"{p}: {n} geom {g.get('name')} has an explicit mass")
            rho = float(g.get("density", dd))
            if rho > 0:
                heavy.append(g)
                rhos.add(rho)
        if len(rhos) != 1:
            die(f"{p}: {n} mass-bearing geoms do not share one density {sorted(rhos)}")
        m_old, rho = float(m0.body_mass[b]), rhos.pop()
        for g in heavy:
            g.set("density", repr(rho * M / m_old))
        fi[n] = {"m_old": m_old, "rho_old": rho, "rho_new": rho * M / m_old, "n_geoms": len(heavy)}
    return fi


def _verify_mass_step(mujoco, np, tree, p, m0, links, M, fi):
    """Rewrite the scene and check that only the object link masses changed and each is M."""
    tree.write(p)
    m1 = mujoco.MjModel.from_xml_path(p)
    for a in (
        "geom_pos",
        "geom_quat",
        "geom_type",
        "geom_size",
        "geom_bodyid",
        "mesh_vert",
        "body_pos",
        "body_quat",
    ):
        if not np.array_equal(getattr(m0, a), getattr(m1, a)):
            die(f"{p}: {a} changed by the mass step")
    obj_ids = {b for _, b in links}
    for b in range(m1.nbody):
        if b not in obj_ids and not (
            m0.body_mass[b] == m1.body_mass[b] and np.array_equal(m0.body_inertia[b], m1.body_inertia[b])
        ):
            die(f"{p}: body {b} (not an object) changed mass/inertia")
    for n, b in links:
        if abs(float(m1.body_mass[b]) - M) > 1e-9:
            die(f"{p}: {n} mass {float(m1.body_mass[b])!r} != {M}")
        fi[n]["m_new"] = float(m1.body_mass[b])


def _write_meta(mujoco, mf, M, meta):
    """Write mass_0p2_meta.json (mass, owner, rule, mujoco version, per-clip records)."""
    payload = {
        "mass_kg": M,
        "owner": "dexengine.rl.tasks.cfg.residual.DEFAULT_OBJECT_MASS_KG",
        "rule": "every rigid link of every task_info object (Isaac modify_mass_properties is @apply_nested); "
        "uniform density rescale of the link's mass-bearing geoms",
        "mujoco": mujoco.__version__,
        "clips": meta,
    }
    with open(mf, "w") as fh:
        json.dump(payload, fh, indent=1)


def cmd_mass(clips_tsv, out):
    import mujoco
    import numpy as np

    M = float(owner_const(OWNER_MASS, "DEFAULT_OBJECT_MASS_KG"))
    rows = read_clips(clips_tsv)
    mf = f"{out}/mass_0p2_meta.json"
    meta = json.loads(Path(mf).read_text())["clips"] if os.path.exists(mf) else {}
    done = 0
    for r in rows:
        d = out_of(out, r)
        if r["clip"] in meta:
            die(f"{r['clip']}: mass already applied (mass_0p2_meta.json) — the step runs once per clip")
        ti = json.loads(Path(f"{d}/task_info.json").read_text())
        info = {}
        for f in SCENES:
            p = f"{d}/{f}"
            try:
                m0 = mujoco.MjModel.from_xml_path(p)
            except ValueError as e:
                # SPIDER loads scene_eq.xml only when num_dyn > 1 (spider/config.py); ours run num_dyn 1 (run_e5.sh
                # checks it). A scene_eq.xml that does not compile in the SOURCE tree either is recorded, not fixed.
                try:
                    mujoco.MjModel.from_xml_path(f"{tree_of(r)}/{f}")
                    src_ok = True
                except ValueError:
                    src_ok = False
                if f != "scene_eq.xml" or src_ok:
                    die(f"{p}: does not compile ({str(e)[:120]}) — the source does: this tool broke it")
                info[f] = {"skipped": f"source {f} does not compile either: {str(e)[:120]}"}
                continue
            tree, dd, bodies, links = _parse_scene(mujoco, p, m0, ti)
            fi = _rescale_links(links, bodies, p, dd, m0, M)
            _verify_mass_step(mujoco, np, tree, p, m0, links, M, fi)
            info[f] = fi
        meta[r["clip"]] = info
        done += 1
        print(
            f"[E5_SPIDER_IN] mass {r['clip']} "
            + " ".join(f"{n}={v['m_old']:.4f}->{v['m_new']:.4f}" for n, v in info["scene.xml"].items())
            + ("" if "skipped" not in info.get("scene_eq.xml", {}) else " scene_eq.xml=SKIPPED(source broken)"),
            flush=True,
        )
    _write_meta(mujoco, mf, M, meta)
    print(f"[E5_SPIDER_IN] mass DONE {done}/{len(rows)} clips · mass_kg={M}")
    sys.exit(0 if done == len(rows) else 1)


def _check_scene(mujoco, r, d, ti, lm, start, M, bad):
    """Check one derived scene (meshes, object masses, SPIDER joint layout, widths, floor z, hulls); appends to bad."""
    for f in ("scene.xml",):
        xml = Path(f"{d}/{f}").read_text()
        A = meshdir_root(d, xml)
        for mfile in re.findall(r'file="([^"]+)"', xml):
            mfile = html.unescape(mfile)  # XML character references (OakInk2 texture names are CJK)
            if not os.path.exists(os.path.join(A, mfile)):
                bad.append(f"{f}: missing {mfile}")
        m = mujoco.MjModel.from_xml_path(f"{d}/{f}")
        for n, b in object_link_bodies(m, ti):
            if abs(float(m.body_mass[b]) - M) > 1e-9:
                bad.append(f"{f}: {n} mass {float(m.body_mass[b]):.6f}")
        import numpy as np

        zr = np.load(f"{d}/0/trajectory_kinematic.npz")
        slots = ["right_object", "left_object"] if r["embodiment"] == "bimanual" else [f"{r['embodiment']}_object"]
        tail = [
            (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[j])), int(m.jnt_type[j]))
            for j in range(m.njnt)
            if int(m.jnt_qposadr[j]) >= m.nu
        ]
        if tail != [(s, int(mujoco.mjtJoint.mjJNT_FREE)) for s in slots]:
            bad.append(f"{f}: not SPIDER's layout: joints after the robot {tail}")
        if zr["qpos"].shape[1] != m.nq or zr["qvel"].shape[1] != m.nv or (lm and lm["nq"] != m.nq):
            bad.append(f"{f}: reference widths {zr['qpos'].shape[1]}/{zr['qvel'].shape[1]} vs nq/nv {m.nq}/{m.nv}")
        fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if abs(float(m.geom_pos[fl][2]) - start[r["clip"]]["desk_z"]) > 1e-6:
            bad.append(f"{f}: floor z {float(m.geom_pos[fl][2])}")
        for bname, (_oid, _ln, n) in start[r["clip"]]["hulls"].items():
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
            got = sum(
                1
                for g in range(m.ngeom)
                if int(m.geom_bodyid[g]) == b
                and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(f"{bname}_hull")
            )
            if got != n:
                bad.append(f"{f}: {bname} hull geoms {got} != {n}")


# ------------------------------------------------------------------------------------------------------------- check
def cmd_check(clips_tsv, out):
    import mujoco

    M = float(json.loads(Path(f"{out}/mass_0p2_meta.json").read_text())["mass_kg"])
    start = json.loads(Path(f"{out}/start_meta.json").read_text())
    rows = read_clips(clips_tsv)
    ok = 0
    for r in rows:
        d = out_of(out, r)
        ti = json.loads(Path(f"{d}/task_info.json").read_text())
        bad = []
        lm = json.loads(Path(f"{out}/layout_meta.json").read_text())["clips"].get(r["clip"])
        if lm is None:
            bad.append("layout step not applied")
        if os.path.exists(f"{d}/scene_eq.xml"):
            bad.append("scene_eq.xml present (the layout step removes it)")
        _check_scene(mujoco, r, d, ti, lm, start, M, bad)
        print(f"[E5_SPIDER_IN] check {r['clip']} {'ok' if not bad else 'BAD ' + '; '.join(bad[:3])}", flush=True)
        ok += not bad
    print(f"[E5_SPIDER_IN] check DONE {ok}/{len(rows)} ok")
    sys.exit(0 if ok == len(rows) else 1)
