"""v4 approach hand ↔ scene geometry (CPU): hand ↔ desk box, scene body list, hand ↔ scene body clearance, and the
scene clearance gate of the chosen approach.

Rule values live in rules.py (SCENE_*, COMBO). Readers: plan.py (approach candidate filter, scene clearance gate) and
stitch.py (hand ↔ desk of the recorded approach poses). Geometry = the judge guard's (hand tree, cooked convex pieces,
support list).
"""

from __future__ import annotations

import contextlib
import io
import json
import re
from pathlib import Path

import numpy as np

from experiments.real_kit.v4 import rules as V4
from experiments.real_kit.v4 import timing as TM

#: Words of the scene-check violation texts; plan.py (" entry ") and bite_blocked ("before") match them.
PRE_WORD, ENTRY_WORD = "before", "entry"


def hand_desk_m(ctx: dict, side: str, q: np.ndarray, fqs: list, z_off: float, dxy) -> float:
    """Max 'penetration depth' of the hand (palm and finger shell vertices — hand_surface.hand_tree) into the desk top
    box (m, negative = margin). The collision desk term measures arm links 0–6 only (no fingers); on the idle-arm
    route the fingertips caught the desk front edge. Signed distance of a point p: inside the footprint = p.z − top;
    outside and below the top = horizontal distance to the footprint; outside and above = distance to the edge. fqs =
    finger value
    lists (each (M, n)) — the worst one."""
    from dexengine.placement.collision import hand_surface as HS
    from dexengine.placement.collision import sim_collider as SC
    from dexengine.placement.collision.geom import arm_link_transforms

    ht = HS.hand_tree(ctx["urdf"], SC.arm_link_names(V4.COMBO[0], side)[-1], HS.finger_names(V4.COMBO[1], side))
    Tl = arm_link_transforms(ctx, q)[:, -1]
    base = ctx["base"] + np.array([0.0, 0.0, z_off])
    (x0, x1), (y0, y1) = np.asarray(dxy, float)
    worst = -1e9
    for fq in fqs:
        F = HS.finger_frames(ht, fq)
        for gi, g in enumerate(ht["groups"]):
            if g["verts"] is None:
                continue
            W = Tl @ F[:, gi]
            p = np.einsum("mij,vj->mvi", W[:, :3, :3], g["verts"]) + W[:, None, :3, 3] + base
            dx = np.maximum(np.maximum(x0 - p[..., 0], p[..., 0] - x1), 0.0)
            dy = np.maximum(np.maximum(y0 - p[..., 1], p[..., 1] - y1), 0.0)
            h = np.hypot(dx, dy)
            dz = p[..., 2] - ctx["ztop"]
            sd = np.where(h <= 0.0, dz, np.where(dz < 0.0, h, np.hypot(h, dz)))
            worst = max(worst, float(-sd.min()))
    return worst


def scene_bodies(tree: Path, sup) -> list:
    """Scene bodies the approach clearance is measured against (hand ↔) = the judge guard's support list
    (support_surface.scene_supports — supports, pedestals, static extras) + the main objects (frame-0 pose, cooked
    convex pieces dexengine.data.objects.hull.body_parts, the same placing function support_surface._posed). An
    articulated main object needs
    joint values for its link poses and cannot be measured — dies."""
    from dexengine.data.objects import hull as OH
    from dexengine.placement.collision import support_surface as SSF
    from dexengine.placement.coords import core as VC

    ti = json.loads((tree / "task_info.json").read_text())
    q = np.load(tree / "0/trajectory_kinematic.npz")["qpos"]
    zoff = float(VC.scene_z_offset(tree))
    out = [{"obj": b["obj"], "parts": b["parts"]} for b in SSF.checked(sup)]
    have = {b["obj"] for b in out}
    for o in ti.get("objects") or []:
        nm = str(o["name"])
        if o.get("role") != "main" or nm in have:
            continue
        if o.get("articulation"):
            raise SystemExit(
                f"{V4.SCENE_TAG} {tree.name}: main object {nm} is articulated — no link pose to measure the approach "
                "clearance from"
            )
        s0 = int(o["qpos_start"])
        mesh = VC._object_mesh_dir(tree, ti, o) / "visual.obj"
        with contextlib.redirect_stdout(io.StringIO()):
            parts = [
                np.asarray(v, float)
                for v in OH.body_parts(str(ti["dataset_name"]), str(o["obj_id"]), False, [(nm, [mesh])])[nm]
            ]
        out.append(SSF._posed(nm, [float(v) for v in q[0, s0 : s0 + 7]], parts, zoff))
    # static bodies = rows extras.decide puts in (physics poses, render-only objects included) that are not listed above
    # — pose = the row's scene_xyz, scene_wxyz (the statement physics and render share) · pieces = cooked convex
    # (recv_clearance.obj_hulls)
    from scipy.spatial.transform import Rotation

    from dexengine.placement.collision import arm_surface as AS
    from dexengine.record.augment import recv_clearance as RC
    from dexengine.record.scene import extras as CE

    have = {b["obj"] for b in out}
    by = {str(o["name"]): o for o in ti.get("objects") or []}
    with contextlib.redirect_stdout(io.StringIO()):
        rows = CE.summary(CE.decide(str(tree), ti)["rows"])[0]
    st = []
    for r in rows:
        if r["name"] in have:
            continue
        if (by.get(r["name"]) or {}).get("articulation"):
            raise SystemExit(
                f"{V4.SCENE_TAG} {tree.name}: static body {r['name']} is articulated — no link pose to measure"
            )
        w, x, y, z = [float(v) for v in r["scene_wxyz"]]
        Rr = Rotation.from_quat([x, y, z, w]).as_matrix()
        xyz = np.asarray(r["scene_xyz"], float)
        nm = f"static:{r['name']}"
        H = RC.obj_hulls(str(ti["dataset_name"]), {"obj_id": r["obj_id"], "articulation": None})
        out.append(
            {
                "obj": nm,
                "parts": [
                    AS.hull_record(f"{nm}#{i}", -1, np.asarray(v, float) @ Rr.T + xyz)
                    for i, (v, _n, _d) in enumerate(H)
                ],
            }
        )
        st.append(nm)
    print(f"{V4.SCENE_TAG} {tree.name} scene bodies {len(out)} · static bodies {len(st)} {st}", flush=True)
    return out


def hand_scene_sd(ctx: dict, q: np.ndarray, f: np.ndarray, z_off: float, bodies: list) -> dict:
    """Hand body (the judge guard's hand tree — palm + finger group shells) ↔ scene bodies → {body: minimum signed
    surface distance over the poses (m, negative = overlap)}. q (P, 7), f (P, n) — the geometry and distance function
    (_min_sd) of
    support_surface.hand_support_mm."""
    from dexengine.placement.collision import arm_surface as AS
    from dexengine.placement.collision import hand_surface as HS
    from dexengine.placement.collision import support_surface as SSF
    from dexengine.placement.collision.geom import arm_link_transforms

    ht = ctx["torso"]["hand"]
    q, f = np.asarray(q, float), np.asarray(f, float)
    W = arm_link_transforms(ctx, q)[:, q.shape[1] - 1][:, None] @ HS.finger_frames(ht, f)
    base = ctx["base"] + np.array([0.0, 0.0, z_off])
    hb = [
        (
            g["root"] if gi else "palm",
            "hull",
            (
                AS.hull_record(str(g["root"]), -1, np.asarray(g["verts"], float)),
                W[:, gi, :3, :3],
                W[:, gi, :3, 3] + base,
            ),
        )
        for gi, g in enumerate(ht["groups"])
        if g["verts"] is not None
    ]
    out: dict = {}
    for sd, _link, obj in SSF._min_sd(hb, bodies):
        out[obj] = min(out.get(obj, np.inf), float(sd))
    return out


def scene_check(ctx, side_data, arrays, K_all, z_off, dxy, bodies) -> tuple[list, dict]:
    """Chosen approach (active arm frames, finger schedule as is) ↔ scene (desk, objects, supports) clearance — rule
    rules.SCENE_*. The first frame of the straight entry = the route sample index interpolated to frames with the same
    schedule (timing.resample) → (violations, {hand: k_ent, f0/pre/ent minimum clearance mm per body})."""
    bad, sc = [], {}
    for side, sd in side_data.items():
        qk, fk = arrays[f"q_{side}"], arrays[f"f_{side}"]
        (idx,) = TM.resample(K_all, sd["sig"], np.arange(len(sd["pq"]), dtype=float)[:, None], e=sd["e"])
        k_ent = int(np.argmax(idx[:, 0] >= sd["n_route"] - 1e-9))
        g = {}
        for tag, sl in (("f0", slice(K_all, K_all + 1)), ("pre", slice(0, k_ent)), ("ent", slice(k_ent, K_all + 1))):
            if sl.stop <= sl.start:
                continue
            d = hand_scene_sd(ctx[side], qk[sl], fk[sl], z_off, bodies)
            d["desk"] = -hand_desk_m(ctx[side], side, qk[sl], [fk[sl]], z_off, dxy)
            g[tag] = d
        for ob, g0 in g["f0"].items():
            need = min(V4.SCENE_CLEAR_M, g0 - V4.SCENE_TOL_M)
            if "pre" in g and g["pre"][ob] < V4.SCENE_CLEAR_M:
                bad.append(f"{side} {ob} {PRE_WORD} {g['pre'][ob] * 1000:.1f} mm")
            if g["ent"][ob] < need:
                bad.append(f"{side} {ob} {ENTRY_WORD} {g['ent'][ob] * 1000:.1f} < {need * 1000:.1f} mm")
        slide = desk_slide_m(ctx[side], side, qk, fk, arrays[f"wrist_{side}"], z_off, dxy)
        if (
            slide > V4.SCENE_DESK_SLIDE_M
        ):  # depends on the finger schedule (different per route) → no skip signal (ENTRY_WORD)
            bad.append(f"{side} desk slide {slide * 1000:.1f} > {V4.SCENE_DESK_SLIDE_M * 1000:.0f} mm")
        sc[side] = dict(
            k_ent=k_ent,
            desk_slide_mm=round(slide * 1000, 1),
            **{t: {ob: round(v * 1000, 1) for ob, v in d.items()} for t, d in g.items()},
        )
    return bad, sc


def desk_slide_m(ctx_s: dict, side: str, q: np.ndarray, f: np.ndarray, wrist: np.ndarray, z_off: float, dxy) -> float:
    """Slide inside the desk (m) — the wrist travel of the run of samples, going back from just before frame 0 (the
    last sample), where hand ↔ desk clearance (hand_desk_m) is < 0 (first sample of that run → frame 0); 0 without
    such a run.
    Rule rules.SCENE_DESK_SLIDE_M."""
    i0 = len(q) - 1
    while i0 > 0 and hand_desk_m(ctx_s, side, q[i0 - 1 : i0], [f[i0 - 1 : i0]], z_off, dxy) > 0:
        i0 -= 1
    if i0 == len(q) - 1:
        return 0.0
    p = np.asarray(wrist, float)[i0:, :3, 3]
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def bite_blocked(ctx, side: str, sd: dict, bad: list, z_off: float, dxy, bodies) -> dict:
    """Blocks at route-independent places → {body: best clearance mm}. Among this candidate's 'pre' (route share)
    violating bodies, keep only those that also miss the route threshold at the backoff point (first pose of the
    straight entry pq[n_route] = q_ent[0], where every route of that backoff rule ends). Fingers take the most generous
    of all values the schedule (rules.finger_schedule) can have there (relaxed → f0 segment, SCENE_BITE_FINGER_SAMPLES
    points); threshold = SCENE_CLEAR_M − SCENE_TOL_M. Non-empty: no route of the same backoff rule can keep that body's
    threshold at its end (plan.py moves to the next rule). Violations mid-route are not blocks (empty dict when the
    backoff point keeps
    the threshold)."""
    pre = {m.group(1) for b in bad if (m := re.match(rf"{re.escape(side)} (.+) {PRE_WORD} -?[\d.]+ mm$", b))}
    if not pre:
        return {}
    q = np.asarray(sd["pq"][sd["n_route"]], float)[None]
    fr, f0 = np.asarray(sd["f_relax"], float), np.asarray(sd["f0"], float)
    best: dict = {}
    for g in np.linspace(0.0, 1.0, V4.SCENE_BITE_FINGER_SAMPLES):
        f = (fr + g * (f0 - fr))[None]
        d = hand_scene_sd(ctx[side], q, f, z_off, bodies)
        d["desk"] = -hand_desk_m(ctx[side], side, q, [f], z_off, dxy)
        for ob in pre:
            best[ob] = max(best.get(ob, -np.inf), float(d[ob]))
    return {ob: round(v * 1000, 1) for ob, v in best.items() if v < V4.SCENE_CLEAR_M - V4.SCENE_TOL_M}
