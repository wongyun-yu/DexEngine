"""Robot body height of the record render. Training, judging and stage-1 rollouts realise the per-clip height dz as a
robot root z shift (arm-IK cache base_z_off -> events._envz_apply_root_z), so on the look-D floor the base sinks in
or floats. Render and store put the base on the floor:
  aerial scenes, world_shift: the robot stays at its default height and the world outside the robot (objects, support
    discs, reference trajectory) moves by -dz: dz is subtracted from the root z of every trajectory entity (a uniform
    translation keeps the relative geometry; the render session checks it with PhysX; the stored columns get the same
    shift, store.meta).
  desk scenes, torso: the desk and the robot root (arms, hands, chest) stay as in the rollout; the torso pitch chain is
    solved again so the chest pose stays and the base reaches the floor (fore-aft offset = base x translation). Visual
    layer only: in the canonical body USD the torso joints are fixed, so the torso links are merged into the single
    visual instance (visuals) of the root body -> that instance is hidden and the same meshes are referenced again with
    per-link transforms. Out of reach dies (no clamping); no solution = render.desk_stretch.
Owner of dz = the recorded trajectory's robot root z - the scene's default root z (measured; constant over the
episode). Torso chain, limits, link <-> mesh = the arm's vendor URDF · canonical posture = the arm spec's skeleton
posture (ArmSpec.skeleton of the active arm — the baked body; checked against the USD visual places).
Marker [COLLECT_BODY] … (session.launch requires it in the render session log).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

from dexengine.core.errors import InputMissingError
from dexengine.record.render.body_geom import box_sdist, desk_box, mesh_points

MARK = "[COLLECT_BODY]"
CONST_TOL_M = 1e-6  # root z constant within the episode · equal between sides
REL_TOL_M = 1e-3  # relative geometry check after the world shift (PhysX FK)
VIS_TOL_M = 1e-3  # canonical posture FK vs USD visual place
SCAN = 40001  # torso solution search: grid of the first pitch joint (the second joint sum is analytic)
DX_QUANT_M = 1e-4  # base shift comparison unit; ties go to the solution closer to the canonical posture
# joint limit margin: the baked canonical torso_1 (-0.5240) passes the arm URDF lower limit (-0.523599) by 0.0004 rad
LIMIT_TOL_RAD = 1e-3


def log_problems(text: str) -> list[str]:
    return [] if f"{MARK} " in text else [f"body height marker '{MARK}' missing"]


# ───────────────────────── torso chain (numpy; tested outside Kit too) ─────────────────────
def _rot(axis, a: float):
    import numpy as np

    x, y, z = (float(v) for v in axis)
    c, s, C = math.cos(a), math.sin(a), 1 - math.cos(a)
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def _rpy(r, p, y):
    return _rot((0, 0, 1), y) @ _rot((0, 1, 0), p) @ _rot((1, 0, 0), r)


def chain(urdf: str | None = None, posture: dict | None = None) -> dict:
    """Torso chain of the arm URDF (base -> torso_* joints, in order), limits, visual mesh name per link, canonical
    posture."""
    if urdf is None or posture is None:
        from dexengine.core.arms import active_arm_spec  # noqa: PLC0415 — the arm skeleton owns URDF and posture
        from dexengine.core.paths import repo_rel_path  # noqa: PLC0415

        sk = active_arm_spec().skeleton
        urdf, posture = urdf or str(repo_rel_path(sk.vendor_urdf)), posture or dict(sk.posture)
    root = ET.parse(urdf).getroot()
    joints = {j.find("child").get("link"): j for j in root.iter("joint")}
    seq = []
    by_parent = {j.find("parent").get("link"): j for j in root.iter("joint") if j.get("name") in posture}
    link = "base"
    while link in by_parent:
        j = by_parent[link]
        o, a, lim = j.find("origin"), j.find("axis"), j.find("limit")
        seq.append(
            {
                "name": j.get("name"),
                "parent": link,
                "child": j.find("child").get("link"),
                "xyz": [float(v) for v in o.get("xyz").split()],
                "rpy": [float(v) for v in (o.get("rpy") or "0 0 0").split()],
                "axis": [float(v) for v in a.get("xyz").split()],
                "lo": float(lim.get("lower")),
                "hi": float(lim.get("upper")),
            }
        )
        link = seq[-1]["child"]
    if {s["name"] for s in seq} != set(posture):
        raise SystemExit(
            f"{MARK} torso chain {[s['name'] for s in seq]} != canonical posture {sorted(posture)} ({urdf})"
        )
    mesh = {}
    for ln in root.iter("link"):
        for v in ln.findall("visual"):
            m = v.find("geometry/mesh")
            if m is not None:
                mesh[m.get("filename").rsplit("/", 1)[-1].rsplit(".", 1)[0]] = ln.get("name")
    parent = {c: j.find("parent").get("link") for c, j in joints.items()}
    return {
        "seq": seq,
        "q0": {s["name"]: float(posture[s["name"]]) for s in seq},
        "mesh": mesh,
        "parent": parent,
        "urdf": urdf,
    }


def fk(ch: dict, q: dict) -> dict:
    """Link transforms (4x4) relative to base: base and the chain links."""
    import numpy as np

    T = {"base": np.eye(4)}
    for s in ch["seq"]:
        M = np.eye(4)
        M[:3, :3] = _rpy(*s["rpy"]) @ _rot(s["axis"], q[s["name"]])
        M[:3, 3] = s["xyz"]
        T[s["child"]] = T[s["parent"]] @ M
    return T


def pitch_joints(ch: dict) -> list[str]:
    """Joints that change the height while keeping the chest orientation: chain joints with axis y (three: two free,
    one fixed by the sum)."""
    names = [s["name"] for s in ch["seq"] if abs(s["axis"][1]) > 0.999]
    if len(names) != 3:
        raise SystemExit(f"{MARK} pitch joints {names}: must be three")
    return names


def solve(ch: dict, dz: float, none_ok: bool = False) -> dict | None:
    """Torso solution whose chest pose (child link of the last pitch joint) = canonical + (0, 0, dz) (relative to base)
    with the smallest base x shift. No solution dies (None with none_ok). Returns q (all joints), base_dx (base x
    shift) and the check errors."""
    import numpy as np

    j1, j2, j3 = pitch_joints(ch)
    by = {s["name"]: s for s in ch["seq"]}
    q0 = ch["q0"]
    T0 = fk(ch, q0)
    chest = by[j3]["child"]
    S = q0[j1] + q0[j2] + q0[j3]
    # first pitch joint place (joints above it do not change)
    p1 = T0[by[j1]["parent"]] @ np.r_[by[j1]["xyz"], 1.0]
    o2, o3 = np.array(by[j2]["xyz"]), np.array(by[j3]["xyz"])
    target = T0[chest][:3, 3] + np.array([0.0, 0.0, dz])

    # rotation about y (earlier joint rpy is 0, chain checked)
    def ry(a, v):
        return np.array([v[0] * math.cos(a) + v[2] * math.sin(a), v[1], -v[0] * math.sin(a) + v[2] * math.cos(a)])

    if any(any(abs(r) > 1e-12 for r in by[n]["rpy"]) for n in (j1, j2, j3)):
        raise SystemExit(f"{MARK} pitch joint origin rpy is not 0: outside this solution's premise")
    best = None
    for t1 in np.linspace(by[j1]["lo"] - LIMIT_TOL_RAD, by[j1]["hi"] + LIMIT_TOL_RAD, SCAN):
        a = ry(t1, o2)
        c = target[2] - p1[2] - a[2]  # = −o3x sin φ + o3z cos φ
        R = math.hypot(o3[0], o3[2])
        if abs(c) > R:
            continue
        d = math.atan2(-o3[0], o3[2])
        for phi in (math.acos(c / R) - d, -math.acos(c / R) - d):
            t2, t3 = phi - t1, S - phi
            if not (
                by[j2]["lo"] - LIMIT_TOL_RAD <= t2 <= by[j2]["hi"] + LIMIT_TOL_RAD
                and by[j3]["lo"] - LIMIT_TOL_RAD <= t3 <= by[j3]["hi"] + LIMIT_TOL_RAD
            ):
                continue
            x = p1[0] + a[0] + ry(phi, o3)[0]
            dx = target[0] - x
            key = (round(abs(dx) / DX_QUANT_M), abs(t1 - q0[j1]) + abs(t2 - q0[j2]))
            if best is None or key < best[0]:
                best = (key, dict(q0, **{j1: float(t1), j2: float(t2), j3: float(t3)}), float(dx))
    if best is None:
        if none_ok:
            return None
        lims = [(by[n]["lo"], by[n]["hi"]) for n in (j1, j2, j3)]
        raise InputMissingError(f"{MARK} the torso cannot make dz {dz:+.4f} m (joint limits {lims}): not clamped")
    _k, q, dx = best
    T1 = fk(ch, q)
    lift = np.eye(4)
    lift[:3, 3] = [dx, 0.0, 0.0]
    err_p = float(np.abs((lift @ T1[chest])[:3, 3] - target).max())
    err_r = float(np.abs(T1[chest][:3, :3] - T0[chest][:3, :3]).max())
    if err_p > 1e-6 or err_r > 1e-9:
        raise SystemExit(f"{MARK} torso solution check failed: position {err_p:.2e} · rotation {err_r:.2e}")
    return {"q": q, "base_dx": dx, "chest": chest, "chest_err_m": err_p, "chest_rot_err": err_r}


def deltas(ch: dict, q: dict, base_dx: float, dz: float) -> dict:
    """Visual transform delta per link relative to root (old base) = Trans(base_dx, 0, -dz) · T(q) · T(q0)^-1; links
    off the chain take the delta of their nearest chain ancestor (fixed transforms below it cancel)."""
    import numpy as np

    T0, T1 = fk(ch, ch["q0"]), fk(ch, q)
    sh = np.eye(4)
    sh[:3, 3] = [base_dx, 0.0, -dz]
    out = {ln: sh @ T1[ln] @ np.linalg.inv(T0[ln]) for ln in T0}
    return out


def link_delta(ch: dict, d: dict, link: str):
    seen, ln = [], link
    while ln not in d:
        seen.append(ln)
        if ln not in ch["parent"]:
            raise SystemExit(f"{MARK} link {link} is not below the torso chain ({seen})")
        ln = ch["parent"][ln]
    return d[ln]


# ───────────────────────── render session (Kit) ─────────────────────────
def _root_path(b) -> str:
    return b.asset.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace("env_.*", "env_0")


def plan(raw_env, binds, st, aerial: bool) -> dict:
    """Measure dz and choose the method (desk scenes solve the torso too; out of reach dies here)."""
    import numpy as np

    robots = [b for b in binds if b.kind == "artic" and not b.name.startswith("object")]
    dzs = []
    for b in robots:
        rz = np.asarray(st[:, b.root_cols.start + 2], float)
        d0 = float(b.asset.data.default_root_state[0, 2])
        if float(rz.max() - rz.min()) > CONST_TOL_M:
            raise SystemExit(f"{MARK} {b.name} root z changes within the episode ({rz.min():.6f}…{rz.max():.6f})")
        dzs.append(float(rz[0]) - d0)
    if not dzs or max(dzs) - min(dzs) > CONST_TOL_M:
        raise SystemExit(f"{MARK} robot articulation root z offsets {dzs}: must be one")
    from dexengine.record.render import desk_stretch as DS

    return DS.decide(dzs[0], aerial)


def shifted(st, binds, body: dict):
    """For the world-shift method, a copy with dz subtracted from every binding's root z; otherwise unchanged."""
    from dexengine.record.render import desk_stretch as DS

    if not DS.world_moved(body):
        return st
    out = st.copy()
    for b in binds:
        out[:, b.root_cols.start + 2] -= body["root_z_offset_m"]
    return out


def check_relative(raw_env, binds, st, sh, write_states) -> float:
    """World shift check (measured): write a few steps in the old and the shifted state and compare body and rigid
    positions minus the robot root position (PhysX)."""
    import numpy as np
    import torch

    robot = next(b for b in binds if b.kind == "artic" and not b.name.startswith("object"))
    worst = 0.0
    for t in sorted({0, st.shape[0] // 2, st.shape[0] - 1}):
        rel = []
        for arr in (st, sh):
            write_states(torch.from_numpy(np.ascontiguousarray(arr[t : t + 1])).to(raw_env.device))
            pts = []
            for b in binds:
                b.asset.update(1e-6)
                pts.append(
                    (b.asset.data.body_pos_w[0] if b.kind == "artic" else b.asset.data.root_pos_w[0:1])
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(float)
                )
            P = np.concatenate(pts)
            rel.append(P - robot.asset.data.root_pos_w[0].detach().cpu().numpy().astype(float))
        worst = max(worst, float(np.abs(rel[1] - rel[0]).max()))
    if worst > REL_TOL_M:
        raise SystemExit(
            f"{MARK} relative geometry off by {worst * 1000:.3f} mm after the world shift (tol {REL_TOL_M * 1000} mm)"
        )
    return worst


def _visuals(stage, raw_env, binds, base_stem: str):
    """Visual instance of the root body: the one 'visuals' instance under the robot articulation whose prototype has
    the base mesh."""
    from pxr import Usd

    hits = []
    for b in binds:
        if b.kind != "artic" or b.name.startswith("object"):
            continue
        root = stage.GetPrimAtPath(_root_path(b))
        hits.extend(
            p
            for p in Usd.PrimRange(root)
            if p.GetName() == "visuals" and p.IsInstance() and p.GetPrototype().GetChild(base_stem)
        )
    if len(hits) != 1:
        raise SystemExit(f"{MARK} {len(hits)} visual instances with the base mesh ({base_stem}): must be one")
    return hits[0]


def _source(vis):
    """References of the instance (asset path, prim path); an empty asset path = the layer that wrote it."""
    for spec in vis.GetPrimStack():
        items = list(spec.referenceList.GetAddedOrExplicitItems()) + list(spec.referenceList.prependedItems)
        for it in items:
            asset = it.assetPath or spec.layer.identifier
            return (spec.layer.ComputeAbsolutePath(asset) if it.assetPath else asset), it.primPath
    raise SystemExit(f"{MARK} {vis.GetPath()}: reference not found")


def apply_torso(raw_env, binds, body: dict) -> dict:
    """Desk-scene visual layer: hide the root visual instance and reference the same meshes again with per-link deltas.
    Check: canonical posture FK vs USD visual place (VIS_TOL_M) · deltas from the chest up = identity."""
    import numpy as np

    import omni.usd
    from pxr import Gf, UsdGeom

    ch, sol = body["_chain"], body["_sol"]
    stage = omni.usd.get_context().get_stage()
    base_stem = next(k for k, v in ch["mesh"].items() if v == "base")
    vis = _visuals(stage, raw_env, binds, base_stem)
    asset, src = _source(vis)
    T0 = fk(ch, ch["q0"])
    d = deltas(ch, sol["q"], sol["base_dx"], body["root_z_offset_m"])
    holder = UsdGeom.Xform.Define(stage, vis.GetParent().GetPath().AppendChild("collect_body"))
    moved, vis_err = [], 0.0
    for c in vis.GetPrototype().GetChildren():
        ln = ch["mesh"].get(c.GetName())
        if ln is None:
            raise SystemExit(f"{MARK} the arm URDF does not know the link of visual mesh {c.GetName()}")
        loc = np.array(UsdGeom.Xformable(c).GetLocalTransformation()).T
        if ln in T0:
            vis_err = max(vis_err, float(np.abs(loc[:3, 3] - T0[ln][:3, 3]).max()))
        D = link_delta(ch, d, ln)
        w = UsdGeom.Xform.Define(stage, holder.GetPath().AppendChild(c.GetName()))
        w.AddTransformOp().Set(Gf.Matrix4d(D.T.tolist()))
        ref = stage.DefinePrim(w.GetPath().AppendChild("mesh_ref"))
        ref.GetReferences().AddReference(asset, src.AppendChild(c.GetName()))
        if np.abs(D - np.eye(4)).max() > 1e-9:
            moved.append(c.GetName())
    if vis_err > VIS_TOL_M:
        raise SystemExit(
            f"{MARK} canonical torso posture FK vs USD visual place {vis_err * 1000:.2f} mm: the posture "
            "differs from the baked body"
        )
    chest_d = float(np.abs(d[sol["chest"]] - np.eye(4)).max())
    if chest_d > 1e-6:
        raise SystemExit(f"{MARK} chest delta is not identity {chest_d:.2e}")
    UsdGeom.Imageable(vis).MakeInvisible()
    return {"visuals": str(vis.GetPath()), "moved": moved, "vis_fk_err_m": vis_err, "source": [asset, str(src)]}


def body_boxes(raw_env, binds, body: dict) -> dict:
    """Per visible body visual mesh (children of the root body visual instance: base, LINK_*, NECK_0), env-frame (lo,
    hi, vertices [N, 3]) from the root pose written now (PhysX), the body delta (torso method only) and the mesh
    vertices (USD). The caller writes the state first (base fixed; root is constant over the episode)."""
    import numpy as np

    import omni.usd
    from pxr import UsdGeom

    import isaaclab.utils.math as mu

    stage = omni.usd.get_context().get_stage()
    ch = body.get("_chain") or chain()
    vis = _visuals(stage, raw_env, binds, next(k for k, v in ch["mesh"].items() if v == "base"))
    own = [b for b in binds if b.kind == "artic" and str(vis.GetPath()).startswith(_root_path(b) + "/")]
    if len(own) != 1:
        raise SystemExit(f"{MARK} visual instance {vis.GetPath()}: {len(own)} articulations, must be one")
    a = own[0].asset
    a.update(1e-6)
    R = np.eye(4)
    R[:3, :3] = mu.matrix_from_quat(a.data.root_quat_w[0:1])[0].detach().cpu().numpy().astype(float)
    R[:3, 3] = (a.data.root_pos_w[0] - raw_env.scene.env_origins[0]).detach().cpu().numpy().astype(float)
    d = None
    if body["method"] == "torso":
        d = deltas(ch, body["_sol"]["q"], body["_sol"]["base_dx"], body["root_z_offset_m"])
    V = np.array(UsdGeom.Xformable(vis).GetLocalTransformation()).T
    out = {}
    for c in vis.GetPrototype().GetChildren():
        loc = np.array(UsdGeom.Xformable(c).GetLocalTransformation()).T
        M = R @ (link_delta(ch, d, ch["mesh"][c.GetName()]) if d is not None else V) @ loc
        P, F = mesh_points(c, faces=True)
        if len(P):
            P = P @ M[:3, :3].T + M[:3, 3]
            out[c.GetName()] = (P.min(0), P.max(0), P, F)
    if not out:
        raise SystemExit(f"{MARK} body visual mesh bounds are empty ({vis.GetPath()})")
    return out


def apply(raw_env, binds, st, aerial: bool, write_states, desk: dict | None = None, reuse: dict | None = None):
    """Render session entry -> (shifted trajectory, meta, body visual boxes). One marker line; with desk (env frame,
    desk scene) the desk top to body clearance. reuse = the meta of the session's previous trajectory: the body visual
    layer is built once per session; the same plan reuses it, a different one dies."""
    import numpy as np
    import torch

    from dexengine.record.render import desk_stretch as DS

    body = plan(raw_env, binds, st, aerial)
    sh = shifted(st, binds, body)
    if reuse is not None and (
        body["method"] != reuse["method"] or abs(body["root_z_offset_m"] - reuse["root_z_offset_m"]) > CONST_TOL_M
    ):
        raise SystemExit(
            f"{MARK} body plan differs from the previous trajectory {body['method']} "
            f"{body['root_z_offset_m']:+.5f} != "
            f"{reuse['method']} {reuse['root_z_offset_m']:+.5f}"
        )
    if DS.world_moved(body):
        body["relative_check_max_m"] = check_relative(raw_env, binds, st, sh, write_states)
    if body["method"] == "torso":
        body["visual"] = apply_torso(raw_env, binds, body) if reuse is None else dict(reuse["visual"], reused=True)
    elif body["method"] == "desk_stretch":
        body["visual"] = (
            DS.apply_desk_stretch(raw_env, body, desk) if reuse is None else dict(reuse["visual"], reused=True)
        )
    write_states(torch.from_numpy(np.ascontiguousarray(sh[0:1])).to(raw_env.device))
    boxes = body_boxes(raw_env, binds, body)
    if desk and not aerial:
        dbox = desk_box(DS.desk_after(desk, body))
        gaps = {n: float(box_sdist(bx[2], dbox).min()) for n, bx in boxes.items()}
        n_min = min(gaps, key=gaps.get)
        body["desk_clearance"] = {
            "min_m": round(gaps[n_min], 4),
            "mesh": n_min,
            "rule": "vertex ↔ top-board box signed distance",
            "per_mesh": {n: round(g, 4) for n, g in gaps.items()},
        }
    meta = {k: v for k, v in body.items() if not k.startswith("_")}
    tq = meta.get("torso_rad")
    print(
        f"{MARK} {meta['method']} · root z offset {meta['root_z_offset_m']:+.4f} m"
        + (
            f" · world shift {meta['world_shift_m']:+.4f} m · "
            f"relative check max {meta['relative_check_max_m'] * 1000:.4f} mm"
            if DS.world_moved(meta)
            else ""
        )
        + DS.mark(meta)
        + (
            f" · torso { {k: round(v, 4) for k, v in tq.items()} } rad · base x {meta['base_shift_m'][0]:+.4f} m · "
            f"visual links moved {meta['visual']['moved']} · canonical FK vs USD "
            f"{meta['visual']['vis_fk_err_m'] * 1000:.3f} mm"
            if tq
            else ""
        )
        + (
            f" · desk top to body vertex min clearance {meta['desk_clearance']['min_m']:+.4f} m "
            f"({meta['desk_clearance']['mesh']})"
            if "desk_clearance" in meta
            else ""
        ),
        flush=True,
    )
    return sh, meta, boxes
