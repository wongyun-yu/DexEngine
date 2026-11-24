"""Record render body height: desk_stretch on desk scenes. When the torso solution (render.body.solve) exists, torso as
now; only without one this method:
  world: the robot keeps its default height and the world outside it (objects, supports, reference trajectory, desk
    top collider) moves by -dz like the aerial world_shift (render.body.shifted: uniform translation; PhysX relative
    check check_relative; stored columns store.meta.world_shift).
  desk visual (Willow): only the legs get longer or shorter: vertex heights z relative to the desk prim origin are
    remapped piecewise linearly (z < STRETCH_Z0_M unchanged: floor bars and feet · z > STRETCH_Z1_M +Δ: frame under
    the top and the top · linear between · Δ = -dz). Vertices within ± STRETCH_GAP_M of the two cut planes (the asset
    differs from the measurement) or a new leg length (z1 - z0 + Δ) below STRETCH_MIN_SPAN_M die (no clamping; no
    limit when stretching).
  top collider (DESK_COLLIDER_KEY): cfg place + world shift · the four leg colliders are outside the rule (invisible;
    the render runs no physics) and are not moved.
Owner of the moved top = desk_after (Rig.desk, the top-to-body clearance of render.body.apply) · owner of the
world-shift decision = world_moved (render.body, store.meta). The marker is the [COLLECT_BODY] line (mark) with
method=desk_stretch, dz, Δ and leg length.
"""

from __future__ import annotations

from dexengine.core.errors import InputMissingError
from dexengine.record.render import body as B

# cut planes (heights in m relative to the desk prim origin), measured on Willow.usd /World/Metal (pxr): floor bar
# vertices <= 0.096 · leg rings 0.148-0.155 · 0.250-0.254 · 0.371 · 0.489-0.492 · 0.587-0.595 · frame under the top
# >= 0.646 · Wood >= 0.7299 · Pads <= 0.016 -> the middles of the vertex-free bands (0.096, 0.148) and (0.595, 0.646).
# Vertices within ± STRETCH_GAP_M of a band die.
STRETCH_Z0_M, STRETCH_Z1_M = 0.122, 0.6205
# half width of the vertex-free band around a cut plane (check: the cut crosses straight legs only)
STRETCH_GAP_M = 0.02
STRETCH_MIN_SPAN_M = 0.05  # minimum length of a shortened leg (z1 - z0 + Δ)
WORLD_SHIFT_METHODS = ("world_shift", "desk_stretch")  # methods that move the world outside the robot by -dz
DESK_COLLIDER_KEY = "desk_collision_top"  # the collider desk_stretch moves (scene cfg key): the top only
MARK = B.MARK


def world_moved(body: dict) -> bool:
    """Does this method move the world outside the robot by -dz? The one owner of that decision."""
    return body.get("method") in WORLD_SHIFT_METHODS


def decide(dz: float, aerial: bool, ch: dict | None = None) -> dict:
    """Body height method (numpy; tested outside Kit too): dz 0 -> none · aerial scene -> world_shift · desk scene ->
    torso (torso solution) · desk_stretch only without a torso solution (stretch_plan dies below the minimum leg).
    The desk is the Willow table of every scene (dexengine.rl.tasks.scene.desk_visual)."""
    if abs(dz) <= B.CONST_TOL_M:
        return {"method": "none", "root_z_offset_m": dz}
    if aerial:
        return {"method": "world_shift", "root_z_offset_m": dz, "world_shift_m": -dz}
    ch = ch or B.chain()
    sol = B.solve(ch, dz, none_ok=True)
    if sol is None:
        return {
            "method": "desk_stretch",
            "root_z_offset_m": dz,
            "world_shift_m": -dz,
            "desk_stretch": stretch_plan(dz),
        }
    return {
        "method": "torso",
        "root_z_offset_m": dz,
        "torso_rad": sol["q"],
        "torso_canon_rad": ch["q0"],
        "base_shift_m": [sol["base_dx"], 0.0, -dz],
        "chest": sol["chest"],
        "chest_err_m": sol["chest_err_m"],
        "_chain": ch,
        "_sol": sol,
    }


def stretch_plan(dz: float) -> dict:
    """desk_stretch numbers of the Willow desk: Δ = -dz · leg length between the cut planes canonical -> new. A new
    length below STRETCH_MIN_SPAN_M dies."""
    span0 = STRETCH_Z1_M - STRETCH_Z0_M
    span = span0 - dz
    if span < STRETCH_MIN_SPAN_M:
        raise SystemExit(
            f"{MARK} desk_stretch: dz {dz:+.4f} m makes the leg {span0:.4f} -> {span:.4f} m < minimum "
            f"{STRETCH_MIN_SPAN_M} m: not clamped"
        )
    return {
        "delta_m": -dz,
        "z0_m": STRETCH_Z0_M,
        "z1_m": STRETCH_Z1_M,
        "span_canon_m": round(span0, 6),
        "span_m": round(span, 6),
        "min_span_m": STRETCH_MIN_SPAN_M,
        "gap_m": STRETCH_GAP_M,
    }


def stretch_z(z, sp: dict):
    """Desk-prim-relative heights z (m, array) -> stretched heights: z < z0 unchanged · z > z1 +Δ · linear between
    (monotone, continuous at the cuts). Points within ± gap of a cut die (premise: the cut crosses straight legs only;
    a changed asset is caught here)."""
    import numpy as np

    z = np.asarray(z, float)
    z0, z1, d, g = sp["z0_m"], sp["z1_m"], sp["delta_m"], sp["gap_m"]
    near = int((np.abs(z - z0) < g).sum() + (np.abs(z - z1) < g).sum())
    if near:
        raise SystemExit(
            f"{MARK} desk_stretch: {near} vertices within ± {g} m of the cut planes z0 {z0} · z1 {z1}: "
            "the desk asset differs from the measurement (re-measure STRETCH_Z0_M, STRETCH_Z1_M)"
        )
    s = (z1 - z0 + d) / (z1 - z0)
    return np.where(z < z0, z, np.where(z > z1, z + d, z0 + (z - z0) * s))


def desk_after(desk: dict | None, body: dict | None) -> dict | None:
    """The render scene's desk (env-frame dict, Rig.desk shape): for desk_stretch a copy with Δ (= world shift) added
    to the top top_z, otherwise unchanged. Aerial scenes (aerial_desk_dz) are untouched. The one owner of the moved
    top (Rig.desk, apply's clearance check)."""
    if desk is None or not body or body.get("method") != "desk_stretch":
        return desk
    return dict(desk, top_z=desk["top_z"] + body["world_shift_m"], desk_stretch_m=body["world_shift_m"])


def _scene_prim(raw_env, key: str) -> str:
    cfg = getattr(raw_env.cfg.scene, key, None)
    if cfg is None:
        raise InputMissingError(f"{MARK} desk_stretch: the scene cfg has no {key}: not a desk scene")
    return cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace("env_.*", "env_0")


def stretch_prim(root, sp: dict) -> dict:
    """Remap the heights of every mesh vertex under the desk visual prim (root; pxr, also used by tests outside Kit),
    measured in world m from the root origin, with stretch_z: only the world z change is turned back into each mesh's
    local frame and the points are overwritten (unchanged points keep their bits). Read-back check: height = mapping
    (1e-4 m) · points with z < z0 bit-identical. Returns per-mesh counts, height ranges before/after and the top's
    highest point before/after."""
    import numpy as np

    from pxr import Usd, UsdGeom, Vt

    t = Usd.TimeCode.Default()
    if not root.IsValid():
        raise InputMissingError(f"{MARK} desk_stretch: desk visual prim {root.GetPath()} missing")
    D = np.array(UsdGeom.Xformable(root).ComputeLocalToWorldTransform(t)).T
    if np.abs(D[:3, 2] / np.linalg.norm(D[:3, 2]) - [0, 0, 1]).max() > 1e-6:
        raise SystemExit(f"{MARK} desk_stretch: desk prim up direction is not world +z ({D[:3, 2]})")
    z_base = float(D[2, 3])
    rows = []
    for m in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not m.IsA(UsdGeom.Mesh):
            continue
        if m.IsInstanceProxy():
            raise InputMissingError(
                f"{MARK} desk_stretch: {m.GetPath()} is an instance proxy: its vertices cannot be overwritten"
            )
        pts = UsdGeom.Mesh(m).GetPointsAttr().Get(t)
        if pts:
            M = np.array(UsdGeom.Xformable(m).ComputeLocalToWorldTransform(t)).T
            P = np.array(pts, np.float32)
            rows.append((m, M, P, (P.astype(float) @ M[:3, :3].T + M[:3, 3])[:, 2] - z_base))
    if not rows:
        raise InputMissingError(f"{MARK} desk_stretch: no mesh vertices under {root.GetPath()}")
    allz = np.concatenate([r[3] for r in rows])
    stretch_z(allz, sp)  # band check (all meshes; one near a cut dies)
    per, worst = {}, 0.0
    for m, M, P, z in rows:
        z2 = stretch_z(z, sp)
        up = np.linalg.inv(M[:3, :3]) @ np.array([0.0, 0.0, 1.0])  # world +z 1 m in local form
        P2 = (P.astype(float) + np.outer(z2 - z, up)).astype(np.float32)
        P2[z2 == z] = P[z2 == z]  # unchanged points keep their bits
        UsdGeom.Mesh(m).GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(P2))
        ext = UsdGeom.Mesh(m).GetExtentAttr()
        if ext.HasAuthoredValue():
            ext.Set(Vt.Vec3fArray.FromNumpy(np.stack([P2.min(0), P2.max(0)])))
        R = np.array(UsdGeom.Mesh(m).GetPointsAttr().Get(t), np.float32)
        zr = (R.astype(float) @ M[:3, :3].T + M[:3, 3])[:, 2] - z_base
        worst = max(worst, float(np.abs(zr - z2).max()))
        fixed = z < sp["z0_m"]
        if not np.array_equal(R[fixed], P[fixed]):
            raise SystemExit(f"{MARK} desk_stretch: vertices of {m.GetName()} with z < z0 moved")
        per[m.GetName()] = {
            "n": int(len(P)),
            "fixed": int(fixed.sum()),
            "raised": int((z > sp["z1_m"]).sum()),
            "stretched": int(((z >= sp["z0_m"]) & (z <= sp["z1_m"])).sum()),
            "z_m": [round(float(z.min()), 4), round(float(z.max()), 4)],
            "z_after_m": [round(float(zr.min()), 4), round(float(zr.max()), 4)],
        }
    if worst > 1e-4:
        raise SystemExit(
            f"{MARK} desk_stretch: re-read vertex heights differ from the mapping by {worst * 1000:.3f} mm"
        )
    return {
        "prim": str(root.GetPath()),
        "z_base_m": z_base,
        "meshes": per,
        "reread_max_err_m": worst,
        "top_visual_m": [round(float(allz.max()), 4), round(max(v["z_after_m"][1] for v in per.values()), 4)],
    }


def apply_desk_stretch(raw_env, body: dict, desk: dict | None) -> dict:
    """Desk-scene desk_stretch scene layer (once per session): desk visual mesh (stretch_prim) and top collider
    (DESK_COLLIDER_KEY) to cfg place + world shift. Check (measured vs stated): collider top = desk_after(desk).top_z
    · visual top highest point = the same value
    (B.VIS_TOL_M)."""
    import numpy as np

    import omni.usd
    from pxr import Usd, UsdGeom

    from dexengine.core import desk as DP

    if not desk:
        raise InputMissingError(f"{MARK} desk_stretch: no desk dict")
    t = Usd.TimeCode.Default()
    stage = omni.usd.get_context().get_stage()
    o = raw_env.scene.env_origins[0].detach().cpu().numpy().astype(float)
    plan = body["desk_stretch"]
    if "desk" in plan:
        raise SystemExit(f"{MARK} desk_stretch: plan for desk {plan['desk']!r} — the scene desk is the Willow table")
    root = stage.GetPrimAtPath(_scene_prim(raw_env, "desk"))
    vis = stretch_prim(root, plan)
    cp = stage.GetPrimAtPath(_scene_prim(raw_env, DESK_COLLIDER_KEY))
    ops = (
        [op for op in UsdGeom.Xformable(cp).GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
        if cp.IsValid()
        else []
    )
    if len(ops) != 1:
        raise SystemExit(f"{MARK} desk_stretch: top collider {cp.GetPath()} has {len(ops)} translate ops, must be one")
    cz = float(getattr(raw_env.cfg.scene, DESK_COLLIDER_KEY).init_state.pos[2])
    v = ops[0].Get()
    if abs(float(v[2]) - cz) > B.CONST_TOL_M:
        raise SystemExit(f"{MARK} desk_stretch: top collider z {float(v[2]):.6f} != cfg {cz:.6f}: already moved")
    ops[0].Set(type(v)(v[0], v[1], cz + body["world_shift_m"]))
    cw = np.array(UsdGeom.Xformable(cp).ComputeLocalToWorldTransform(t)).T
    top_col = float(cw[2, 3]) + DP.DESK_TOP_THICK / 2 - o[2]
    top_vis = vis["top_visual_m"][1] + vis["z_base_m"] - o[2]
    want = desk_after(desk, body)["top_z"]
    if abs(top_col - want) > B.VIS_TOL_M or abs(top_vis - want) > B.VIS_TOL_M:
        raise SystemExit(
            f"{MARK} desk_stretch: moved top collider top {top_col:.4f} · visual top {top_vis:.4f} != "
            f"stated {want:.4f} m"
        )
    return dict(
        vis,
        collider={"prim": str(cp.GetPath()), "top_m": round(top_col, 4)},
        legs="not moved (outside the rule · not visible · the render has no physics)",
    )


def mark(meta: dict) -> str:
    """The desk_stretch part of the [COLLECT_BODY] line (other methods: empty): method, dz, Δ, leg length, top."""
    ds = meta.get("desk_stretch")
    if not ds:
        return ""
    v = meta["visual"]
    how = f"cuts z0 {ds['z0_m']} · z1 {ds['z1_m']}"
    return (
        f" · desk stretch method=desk_stretch dz {meta['root_z_offset_m']:+.4f} m · Δ {ds['delta_m']:+.4f} m · "
        f"leg {ds['span_canon_m']:.4f} -> {ds['span_m']:.4f} m ({how}) · "
        f"visual top {v['top_visual_m'][0]:.4f} -> {v['top_visual_m'][1]:.4f} m · collider top "
        f"{v['collider']['top_m']:.4f} m · re-read height error {v['reread_max_err_m'] * 1000:.3f} mm"
    )
