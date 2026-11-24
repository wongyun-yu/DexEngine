"""Object mass = our fixed object mass for the SPIDER and EgoEngine inputs: a derived scene set next to the start-state
(a)
set (data/spider_a · data/egoengine_a, built by prep_start_a.py), which is read only.

Rule: every free object body (right_object · left_object) gets total mass M with the inertia of the same shape at
uniform
density, done as a uniform density rescale of the body's mass-bearing geoms (the collision hulls, density 1000 in the
(a)
scenes): rho' = rho x M / m_old. MuJoCo then computes mass, centre of mass and inertia from the same geoms as before,
only
scaled by M / m_old. M is read by name from its owner, dexengine.rl.tasks.cfg.residual.DEFAULT_OBJECT_MASS_KG (that
module
imports Isaac Lab, so its literal is parsed, not imported), and must match the set label m0p2. A body whose mass-bearing
geoms do not share one density, a geom with an explicit mass, an explicit <inertial>, a missing object body or the
generator's stand-in sphere (<side>_object_mass, a scene without that object) stops the tool.

Verification (every clip · scene.xml and scene_eq.xml · in the method's own mujoco):
  1. body_mass = M (|d| < 1e-9) for all 128 object bodies
  2. inertia tensor in the body frame = old x M / m_old (rel 1e-9) and the centre of mass unchanged
  3. everything the scorer reads is unchanged: geom pos / quat / type / size and mesh vertices bit-equal
  4. every body except the two objects keeps mass and inertia bit-equal
The other files of a clip are the (a) set's own entries: 0/* and task_info.json = the same symlink targets, the assets =
relative symlinks into the (a) root. Method outputs (trajectory_mjwp*.npz · config.yaml) are never carried over.

This file runs standalone in the baseline venvs (`<python> -I prep_mass_0p2.py …`): it imports no repository module.
usage: prep_mass_0p2.py <src (a) root> <out root>     -> one [MASS_0P2] line per clip + <out>/mass_0p2_meta.json
       prep_mass_0p2.py --check <root> <clip>         -> '[MASS_0P2] <clip> scene.xml right=… left=… scene_eq.xml …
       target=… ok'
                                                         (the run-time marker on the scenes the method loads; rc 3 if
                                                         any
                                                         object is not M). Without the owner file (a copy outside the
                                                         repository) the set's own meta gives M.
"""

import ast
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SUB = "processed/taco/sharpa_wave/bimanual"
OWNER = os.path.join(REPO, "dexengine/rl/tasks/cfg/residual.py")
OWNER_NAME = "dexengine.rl.tasks.cfg.residual.DEFAULT_OBJECT_MASS_KG"
CLIPS = os.path.join(REPO, "experiments/scoring/clips/E123_CLIPS_v1.tsv")
SIDES = ("right", "left")
SCENES = ("scene.xml", "scene_eq.xml")
LABEL = "m0p2"  # the derived set's name part: must say the owner's value (a changed owner dies in owner_mass)


def owner_mass() -> float:
    for n in ast.parse(Path(OWNER).read_text()).body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "DEFAULT_OBJECT_MASS_KG" for t in n.targets):
            m = float(ast.literal_eval(n.value))
            if f"m{m:g}".replace(".", "p") != LABEL:
                raise SystemExit(f"[MASS_0P2] FAIL owner DEFAULT_OBJECT_MASS_KG = {m} but this set is labelled {LABEL}")
            return m
    raise SystemExit(f"[MASS_0P2] FAIL DEFAULT_OBJECT_MASS_KG not found in {OWNER}")


def clips_v1() -> list:
    rows = [ln.split("\t") for ln in Path(CLIPS).read_text().splitlines() if ln and not ln.startswith("#")]
    if rows[0][:2] != ["idx", "clip"] or len(rows) != 65:
        raise SystemExit("[MASS_0P2] FAIL clips/E123_CLIPS_v1.tsv header/rows")
    return [r[1] for r in rows[1:]]


def obj_body(m, side) -> int:
    import mujoco

    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_object")
    if b < 0:
        raise SystemExit(f"[MASS_0P2] FAIL no body {side}_object")
    if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_object_mass") >= 0:
        raise SystemExit(
            f"[MASS_0P2] FAIL {side}: the generator's stand-in sphere {side}_object_mass (a scene without this object)"
        )
    return b


def inertia_tensor(m, b) -> np.ndarray:
    import mujoco

    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.body_iquat[b])
    R = R.reshape(3, 3)
    return R @ np.diag(m.body_inertia[b]) @ R.T


def default_density(root) -> float:
    d = root.find("default")
    g = d.find("geom") if d is not None else None
    return float(g.get("density", 1000)) if g is not None else 1000.0


def _rescale_object_density(m0, side, root, dd, M, info):
    """Scale the density of one side object geoms so the object weighs M; its old mass and densities into info."""
    b0 = obj_body(m0, side)
    m_old = float(m0.body_mass[b0])
    body = [x for x in root.find("worldbody").iter("body") if x.get("name") == f"{side}_object"]
    assert len(body) == 1, side
    if body[0].find("inertial") is not None:
        raise SystemExit(f"[MASS_0P2] FAIL {side}_object has an explicit <inertial> (mass not from its geoms)")
    if body[0].find("body") is not None:
        raise SystemExit(f"[MASS_0P2] FAIL {side}_object has child bodies (one rigid object expected)")
    heavy, rhos = [], set()
    for g in body[0].findall("geom"):
        if g.get("mass") is not None:
            raise SystemExit(f"[MASS_0P2] FAIL {side}_object geom {g.get('name')} has an explicit mass")
        rho = float(g.get("density", dd))
        if rho > 0:
            heavy.append(g)
            rhos.add(rho)
    if len(rhos) != 1 or not heavy:
        raise SystemExit(f"[MASS_0P2] FAIL {side}_object: mass-bearing geoms do not share one density {sorted(rhos)}")
    rho = rhos.pop()
    new = rho * M / m_old
    for g in heavy:
        g.set("density", repr(new))
    info[side] = {"m_old": m_old, "rho_old": rho, "rho_new": new, "n_geoms": len(heavy)}


def _check_object_masses(m1, m0, out_xml, M, info):
    """Fail unless only the object bodies changed and each now weighs M with its inertia scaled by M / old mass."""
    objs = {obj_body(m1, s): s for s in SIDES}
    for b in range(m1.nbody):
        if b in objs:
            continue
        if not (m0.body_mass[b] == m1.body_mass[b] and np.array_equal(m0.body_inertia[b], m1.body_inertia[b])):
            raise SystemExit(f"[MASS_0P2] FAIL {out_xml}: body {b} (not an object) changed mass/inertia")
    for b, side in objs.items():
        k = M / info[side]["m_old"]
        mn = float(m1.body_mass[b])
        if abs(mn - M) > 1e-9:
            raise SystemExit(f"[MASS_0P2] FAIL {out_xml}: {side}_object mass {mn!r} ≠ {M}")
        I0, I1 = inertia_tensor(m0, b) * k, inertia_tensor(m1, b)
        if not np.allclose(I1, I0, rtol=1e-9, atol=1e-15) or not np.allclose(
            m0.body_ipos[b], m1.body_ipos[b], rtol=0, atol=1e-12
        ):
            raise SystemExit(
                f"[MASS_0P2] FAIL {out_xml}: {side}_object inertia ≠ old × {k:.6g} or its centre of mass moved"
            )
        info[side].update(m_new=mn, inertia_scale=k, diaginertia_new=[float(x) for x in m1.body_inertia[b]])


def transform(src_xml: str, out_xml: str, M: float) -> dict:
    import mujoco

    m0 = mujoco.MjModel.from_xml_path(src_xml)
    tree = ET.parse(src_xml)
    root = tree.getroot()
    dd = default_density(root)
    if any(d.get("class") for d in root.iter("default")):
        raise SystemExit(
            f"[MASS_0P2] FAIL {src_xml}: named default classes — the effective density rule here reads the "
            "top default only"
        )
    info = {}
    for side in SIDES:
        _rescale_object_density(m0, side, root, dd, M, info)
    tree.write(out_xml)
    m1 = mujoco.MjModel.from_xml_path(out_xml)
    # (3) the scorer's input (geoms, meshes) bit-equal; (4) every non-object body bit-equal
    for a in (
        "geom_pos",
        "geom_quat",
        "geom_type",
        "geom_size",
        "geom_bodyid",
        "mesh_vert",
        "mesh_vertadr",
        "mesh_vertnum",
        "body_pos",
        "body_quat",
    ):
        if not np.array_equal(getattr(m0, a), getattr(m1, a)):
            raise SystemExit(f"[MASS_0P2] FAIL {out_xml}: {a} changed")
    _check_object_masses(m1, m0, out_xml, M, info)
    return info


def check(root: str, clip: str) -> None:
    import mujoco

    # a copy outside the repository has no owner file: the set's own meta carries the owner's value (written by main)
    M = (
        owner_mass()
        if os.path.exists(OWNER)
        else float(json.loads(Path(f"{root}/mass_0p2_meta.json").read_text())["mass_kg"])
    )
    parts, bad = [], 0
    for f in SCENES:
        m = mujoco.MjModel.from_xml_path(f"{root}/{SUB}/{clip}/{f}")
        ms = {s: float(m.body_mass[obj_body(m, s)]) for s in SIDES}
        bad += sum(abs(v - M) > 1e-9 for v in ms.values())
        parts.append(f"{f} " + " ".join(f"{s}={v:.6f}" for s, v in ms.items()))
    print(f"[MASS_0P2] {clip} {' '.join(parts)} target={M:.6f} {'ok' if bad == 0 else f'BAD {bad}'}", flush=True)
    sys.exit(0 if bad == 0 else 3)


def main() -> None:
    if sys.argv[1:2] == ["--check"]:
        return check(sys.argv[2], sys.argv[3])
    import mujoco

    src, out = (os.path.abspath(x) for x in sys.argv[1:3])
    M = owner_mass()
    if os.path.lexists(out):
        raise SystemExit(
            f"[MASS_0P2] FAIL {out} exists — a derived set is written once (remove it by its literal path first)"
        )
    os.makedirs(out)
    os.symlink(os.path.relpath(f"{src}/taco_retarget", out), f"{out}/taco_retarget")
    os.makedirs(f"{out}/processed/taco")
    os.symlink("../../taco_retarget/processed/taco/assets", f"{out}/processed/taco/assets")
    meta, done = (
        {
            "mass_kg": M,
            "owner": OWNER_NAME,
            "src": os.path.relpath(src, os.path.dirname(out)),
            "mujoco": mujoco.__version__,
            "rule": "uniform density rescale of each object body's mass-bearing geoms (inertia = same shape · "
            "uniform density)",
            "clips": {},
        },
        0,
    )
    for c in clips_v1():
        s, d = f"{src}/{SUB}/{c}", f"{out}/{SUB}/{c}"
        os.makedirs(f"{d}/0")
        for f in sorted(os.listdir(f"{s}/0")):
            if not os.path.islink(f"{s}/0/{f}"):
                raise SystemExit(f"[MASS_0P2] FAIL {s}/0/{f} is not a symlink — a method output in the source set")
            os.symlink(os.readlink(f"{s}/0/{f}"), f"{d}/0/{f}")
        if not os.path.islink(f"{s}/task_info.json"):
            raise SystemExit(f"[MASS_0P2] FAIL {s}/task_info.json is not a symlink")
        os.symlink(os.readlink(f"{s}/task_info.json"), f"{d}/task_info.json")
        extra = sorted(set(os.listdir(s)) - {"0", "task_info.json", *SCENES})
        if extra:
            raise SystemExit(f"[MASS_0P2] FAIL {s}: unexpected entries {extra}")
        meta["clips"][c] = {f: transform(f"{s}/{f}", f"{d}/{f}", M) for f in SCENES}
        for side in SIDES:  # both scene files describe the same object (same old mass)
            a, b = (meta["clips"][c][f][side]["m_old"] for f in SCENES)
            if a != b:
                raise SystemExit(f"[MASS_0P2] FAIL {c} {side}: scene.xml mass {a} ≠ scene_eq.xml {b}")
        done += 1
        r = meta["clips"][c]["scene.xml"]
        print(
            f"[MASS_0P2] {c} right {r['right']['m_old']:.4f}→{r['right']['m_new']:.4f} left {r['left']['m_old']:.4f}→"
            f"{r['left']['m_new']:.4f} kg",
            flush=True,
        )
    olds = [meta["clips"][c]["scene.xml"][s]["m_old"] for c in meta["clips"] for s in SIDES]
    meta["old_mass_summary"] = {
        "n": len(olds),
        "median": float(np.median(olds)),
        "min": min(olds),
        "max": max(olds),
        "over_1kg": sum(x > 1 for x in olds),
        "within_10pct_of_M": sum(abs(x - M) <= 0.1 * M for x in olds),
    }
    with open(f"{out}/mass_0p2_meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    n = len(clips_v1())
    print(
        f"[MASS_0P2] DONE {done}/{n} clips · {2 * done} objects · {2 * done * len(SCENES)} scene bodies = {M} kg · old "
        f"{meta['old_mass_summary']} · meta {out}/mass_0p2_meta.json",
        flush=True,
    )
    sys.exit(0 if done == n else 1)


if __name__ == "__main__":  # importable: method_cmds.py and dexmachina/taco_to_dexmachina.py read owner_mass() here
    main()
