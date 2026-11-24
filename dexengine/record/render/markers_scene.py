"""Place-marker scene construction inside Kit (see render.markers for the switch, plan and cache-tag side)."""

from __future__ import annotations

import math
from pathlib import Path

from dexengine.core.errors import InputMissingError
from dexengine.record.render.markers import LINE_GAP, LINE_W, MARK, REF_DIR, style
from dexengine.record.render.markers_geom import (
    NSEG,
    REF_MD5,
    load_ref,
    object_meshes,
    support_top,
)
from dexengine.record.render.markers_shapes import (
    LIFT,
    PAD_R,
    PLATE_T,
    ROOT,
    X_TAPE_W,
    band_mesh,
    outline_contour,
    pad_plate,
    x_tape,
)

COLOR = (0.0, 0.78, 0.95)  # cyan: a hue unlike the desk (birch, warm brown) and most objects
# OmniPBR emissive_intensity: the colour stays visible in shade (under objects and hands) (trial: 5 · 20 · 80, chose 20)
EMISSIVE = 20.0
BAND_FRAC, BAND_MIN = 0.15, 0.004  # bottom band = object bottom z + max(BAND_MIN, BAND_FRAC x height)
POS_TOL_M = 0.01
# matte tape (no emission, roughness X_ROUGH), linear RGB
X_COLORS = {"red": (0.55, 0.02, 0.02), "black": (0.02, 0.02, 0.02)}
X_ROUGH = 0.85

MDL = "OmniPBR.mdl"  # Kit default MDL (resolved from the search path, not S3)
# for the check: rotation of the reference trajectory transform (Rz -90°); the value used comes from the env function
RZ = ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))


def _x_material(stage, color: str):
    """X marker material: OmniPBR opaque, matte (tape that takes light, no emission)."""
    from pxr import Gf, Sdf, UsdShade

    mp = f"{ROOT}/Looks/tape_{color}"
    mat = UsdShade.Material.Define(stage, mp)
    sh = UsdShade.Shader.Define(stage, f"{mp}/Shader")
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset(MDL, "mdl")
    sh.SetSourceAssetSubIdentifier(MDL.rsplit(".", 1)[0], "mdl")
    sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*X_COLORS[color]))
    sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(X_ROUGH)
    sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
    sh.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(False)
    out = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for o in (mat.CreateSurfaceOutput("mdl"), mat.CreateDisplacementOutput("mdl"), mat.CreateVolumeOutput("mdl")):
        o.ConnectToSource(out)
    return mat


def _material(stage):
    """OmniPBR opaque cyan + self emission (an MDL path like the support material; UsdPreviewSurface came out black in
    this render)."""
    from pxr import Gf, Sdf, UsdShade

    mp = f"{ROOT}/Looks/mark"
    mat = UsdShade.Material.Define(stage, mp)
    sh = UsdShade.Shader.Define(stage, f"{mp}/Shader")
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset(MDL, "mdl")
    sh.SetSourceAssetSubIdentifier(MDL.rsplit(".", 1)[0], "mdl")
    sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*COLOR))
    sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.6)
    sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
    sh.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
    sh.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*COLOR))
    sh.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(float(EMISSIVE))
    out = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for o in (mat.CreateSurfaceOutput("mdl"), mat.CreateDisplacementOutput("mdl"), mat.CreateVolumeOutput("mdl")):
        o.ConnectToSource(out)
    return mat


def _no_physics(stage, path: str) -> int:
    """Check that no physics API (collision, rigid body, mass …) is under the made prims -> number of prims checked."""
    from pxr import Usd

    ps = list(Usd.PrimRange(stage.GetPrimAtPath(path)))
    bad = [str(p.GetPath()) for p in ps if any(s.startswith(("Physics", "Physx")) for s in p.GetAppliedSchemas())]
    if bad:
        raise SystemExit(f"{MARK} physics API on marker prims {bad[:3]}")
    return len(ps)


def _render_only(stage, path: str) -> int:
    """Markers cast no shadows (doNotCastShadows), so pixels outside the marker do not change (pixel difference
    outside the marker ≈ 0). Reflections and indirect light are off globally in look D (looks.look_d.QUALITY)."""
    from pxr import Sdf, Usd, UsdGeom

    n = 0
    for p in Usd.PrimRange(stage.GetPrimAtPath(path)):
        if p.IsA(UsdGeom.Gprim):
            UsdGeom.PrimvarsAPI(p).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
            n += 1
    if not n:
        raise SystemExit(f"{MARK} no drawable prim under {path}")
    return n


def _env_mapper(tm, dz: float, dev, shift_z: float):
    """Tree-frame point p and rotation R -> env frame (loader clip dz, the reference trajectory transform
    rotate_to_x_forward, the session's world shift_z) as (position, unit wxyz quaternion)."""
    import numpy as np
    import torch

    from isaaclab.utils.math import quat_from_matrix

    def to_env(p, R):
        pp = torch.tensor(np.asarray(p, float) + [0.0, 0.0, dz], dtype=torch.float32, device=dev)[None]
        pe = tm._rotate_pos_batch(pp)[0].cpu().numpy().astype(float) + [0.0, 0.0, shift_z]
        q = quat_from_matrix(torch.tensor(np.asarray(R, float), dtype=torch.float32, device=dev)[None])
        qe = tm._rotate_quat_batch(q)[0].cpu().numpy().astype(float)
        return pe, qe / np.linalg.norm(qe)

    return to_env


def _marker_inputs(raw_env, tree_dir: Path, k: int, shift_z: float):
    """One-clip session and loader-clip checks -> (place single source PR, place targets pt, clip dz, to_env)."""
    if raw_env.num_envs != 1:
        raise SystemExit(f"{MARK} {raw_env.num_envs} envs (one-clip sessions only)")
    PR = load_ref()
    pt = PR.place_target(tree_dir)
    if not pt:
        raise InputMissingError(f"{MARK} {tree_dir.name}: 0 main objects, no place")
    tm = raw_env.trajectory_manager
    if not tm._rotate:
        raise SystemExit(f"{MARK} reference trajectory transform has rotate_to_x_forward off: outside this rule")
    names = tm._loader.sequence_names
    if names[k] != tree_dir.name:
        raise SystemExit(f"{MARK} loader clip {k} = {names[k]} != tree {tree_dir.name}")
    dz = tm._loader.clip_dz_of(k)
    dev = raw_env.device
    return PR, pt, dz, _env_mapper(tm, dz, dev, shift_z)


def _match_start(tree_dir: Path, n: str, v: dict, objs, P0, to_env):
    """The one scene object whose first pose is within POS_TOL_M of place target v's p_start (transform check) ->
    (distance, binding name, binding)."""
    import numpy as np

    if v["frame"] != "world":
        raise InputMissingError(
            f"{MARK} {tree_dir.name}/{n}: frame={v['frame']} (frame of receiving object "
            f"{v['receiver']}): no marker for it yet"
        )
    ps, _ = to_env(v["p_start"], np.eye(3))
    d = sorted((float(np.linalg.norm(P0[b.root_cols][:3] - ps)), b.name) for b in objs)
    if not d or d[0][0] > POS_TOL_M or (len(d) > 1 and d[1][0] <= POS_TOL_M):
        raise SystemExit(
            f"{MARK} {tree_dir.name}/{n}: p_start(env) {np.round(ps, 4).tolist()} ↔ "
            f"scene object first poses {d}: "
            f"exactly one must be within {POS_TOL_M} m (transform check)"
        )
    err, bn = d[0]
    b = next(x for x in objs if x.name == bn)
    return err, bn, b


def _end_pose(PR, v: dict, to_env):
    """End pose of place target v in the env frame -> (position, quaternion, rotation matrix from the quaternion);
    the rotation is checked against Rz·R_end."""
    import numpy as np

    pe, qe = to_env(v["p_end"], v["R_end"])
    # end rotation in the env frame (the value used, from qe)
    Re = PR.quat_R(qe)
    rz_err = float(np.abs(Re - np.array(RZ) @ np.asarray(v["R_end"], float)).max())
    if rz_err > 1e-4:
        raise SystemExit(f"{MARK} end rotation env-frame check failed (quaternion transform vs Rz·R) {rz_err:.2e}")
    return pe, qe, Re


def _pad_receiver(raw_env, stage, s, xcol, tree_dir, n, bn, v, pe, wp, allp, zmin, err, rp):
    """Pad style: the support top under the end pose (support_top with want_z). A support that is not one of our
    support discs means the place is a data object (receiver): no plate, a skip row. -> (skip row | None, sup,
    sup_path, sup_z, sup_tops)."""
    from dexengine.rl.tasks.scene.collision_bodies import SUPPORT_DISC_PREFIX

    lo_, hi_ = allp[:, :2].min(0), allp[:, :2].max(0)
    cw_ = wp[:2] + 0.5 * (lo_ + hi_)
    sup, sup_path, sup_z, sup_tops = support_top(raw_env, stage, float(wp[2] + zmin), cw_, (rp,), want_z=True)
    if sup_path is not None and not str(sup_path).startswith(SUPPORT_DISC_PREFIX):
        row = {
            "style": s,
            "obj": n,
            "entity": bn,
            "skip": "receiver",
            "support": sup_path,
            "receiver": v["receiver"],
            "pos": [round(float(x), 4) for x in pe],
            "z_bottom": round(float(pe[2] + zmin), 4),
            "start_err_m": round(err, 5),
            "physics_api": 0,
            "prims": 0,
        }
        print(
            f"{MARK} style=pad color={xcol} obj={n}({bn}) skip=receiver support={sup_path} · "
            f"bottom z {row['z_bottom']} · start check {err * 1000:.1f} mm · no plate (the place is a data "
            f"object) · "
            f"{tree_dir.name}",
            flush=True,
        )
        return row, sup, sup_path, sup_z, sup_tops
    return None, sup, sup_path, sup_z, sup_tops


def _marker_bbox(geo, xy, wp, zmin, z0, z1):
    """World extent of the marker (bbox_marker) into geo."""
    geo["bbox_marker"] = [
        [round(float(x), 4) for x in (*(xy.min(0) + wp[:2]), wp[2] + zmin + z0)],
        # marker extent (world)
        [round(float(x), 4) for x in (*(xy.max(0) + wp[:2]), wp[2] + zmin + z1)],
    ]


def _say_marker(xcol, shape, n, bn, row, err, geo, n_prims, n_g, meshes, tree_dir):
    """The marker line of one place target."""
    cs = f" color={xcol}" if xcol else ""
    print(
        f"{MARK} style={shape}{cs} obj={n}({bn}) pos={row['pos']} quat={row['quat']} · "
        f"bottom z {row['z_bottom']} · "
        f"moved {row['disp_m']} m · start check {err * 1000:.1f} mm · {geo} · "
        f"physics API 0/{n_prims} prims · shadows off {n_g} · "
        f"meshes {len(meshes)} {row['meshes_src'][:1]} · {tree_dir.name}",
        flush=True,
    )


def _author_marker(raw_env, stage, tree_dir, n, v, *, s, shape, xcol, mat, objs, P0, origin, PR, to_env, dz, shift_z):
    """Marker prims of one place target n (style s) -> its marker row (or the pad skip row of a data-object place)."""
    import numpy as np

    from pxr import Gf, UsdGeom, UsdShade

    err, bn, b = _match_start(tree_dir, n, v, objs, P0, to_env)
    pe, qe, Re = _end_pose(PR, v, to_env)
    rp = b.asset.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace("env_.*", "env_0")
    meshes = object_meshes(stage, rp)
    # end-pose object points (env rotation, origin = object root)
    allp = np.concatenate([m[1] for m in meshes]) @ Re.T
    zmin, zmax = float(allp[:, 2].min()), float(allp[:, 2].max())
    wp = pe + origin
    if shape == "pad":
        skip, sup, sup_path, sup_z, sup_tops = _pad_receiver(
            raw_env, stage, s, xcol, tree_dir, n, bn, v, pe, wp, allp, zmin, err, rp
        )
        if skip is not None:
            return skip
    xf = UsdGeom.Xform.Define(stage, f"{ROOT}/{n}")
    # origin = root xy · end bottom z
    UsdGeom.Xformable(xf).AddTranslateOp().Set(Gf.Vec3d(float(wp[0]), float(wp[1]), float(wp[2] + zmin)))
    z0, z1 = LIFT, LIFT + PLATE_T
    band = zmin + max(BAND_MIN, BAND_FRAC * (zmax - zmin))
    foot = allp[allp[:, 2] <= band][:, :2]
    lo, hi = allp[:, :2].min(0), allp[:, :2].max(0)
    c, r = 0.5 * (lo + hi), 0.5 * float(np.max(hi - lo))
    if s in ("outline", "contour"):
        xy, geo = outline_contour(stage, s, n, foot, band, zmin, z0, z1)
    else:
        a = np.linspace(0, 2 * math.pi, NSEG, endpoint=False)
        u = np.stack([np.cos(a), np.sin(a)], -1)
        if shape == "x":
            xy, geo = x_tape(raw_env, stage, n, wp, zmin, c, r, rp, z0, z1)
        elif shape == "pad":
            xy, geo, z0, z1 = pad_plate(raw_env, stage, n, wp, zmin, c, r, u, sup, sup_path, sup_z, sup_tops)
        elif s == "disc":
            cyl = UsdGeom.Cylinder.Define(stage, f"{ROOT}/{n}/disc")
            cyl.CreateRadiusAttr(r)
            cyl.CreateHeightAttr(PLATE_T)
            cyl.CreateAxisAttr("Z")
            UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3d(float(c[0]), float(c[1]), z0 + 0.5 * PLATE_T))
            xy = c + r * u
        else:
            xy = c + (r + LINE_GAP + LINE_W) * u
            band_mesh(stage, f"{ROOT}/{n}/ring", c + (r + LINE_GAP) * u, xy, z0, z1)
        if shape not in ("x", "pad"):
            geo = {"radius_m": round(r, 4), "center_off_m": [round(float(x), 4) for x in c]}
    UsdShade.MaterialBindingAPI.Apply(xf.GetPrim()).Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    n_prims, n_g = _no_physics(stage, f"{ROOT}/{n}"), _render_only(stage, f"{ROOT}/{n}")
    _marker_bbox(geo, xy, wp, zmin, z0, z1)
    row = dict(
        style=s,
        obj=n,
        entity=bn,
        pos=[round(float(x), 4) for x in pe],
        quat=[round(float(x), 4) for x in qe],
        z_bottom=round(float(pe[2] + zmin), 4),
        p_end_tree=[round(float(x), 4) for x in v["p_end"]],
        disp_m=round(v["disp_m"], 4),
        receiver=v["receiver"],
        frame=v["frame"],
        start_err_m=round(err, 5),
        dz=round(dz, 4),
        shift_z=round(shift_z, 4),
        meshes_src=[m[0] for m in meshes][:6],
        physics_api=0,
        prims=n_prims,
        no_shadow_gprims=n_g,
        **geo,
    )
    _say_marker(xcol, shape, n, bn, row, err, geo, n_prims, n_g, meshes, tree_dir)
    return row


def _style_record(s, shape, xcol, shown: list) -> dict:
    """Marker record of the session (render_meta): style, colour and material values, place source, object rows."""
    if shape == "pad":
        return {
            "style": s,
            "shape": "pad",
            "color": xcol,
            "rgb": list(X_COLORS[xcol]),
            "emissive": 0.0,
            "roughness": X_ROUGH,
            "pad_r_m": PAD_R,
            "thick_m": PLATE_T,
            "mdl": MDL,
            "ref": f"{REF_DIR}/e5_place_ref.py@{REF_MD5[:8]}",
            "objects": shown,
        }
    if shape == "x":
        return {
            "style": s,
            "shape": "x",
            "color": xcol,
            "rgb": list(X_COLORS[xcol]),
            "emissive": 0.0,
            "roughness": X_ROUGH,
            "tape_w_m": X_TAPE_W,
            "mdl": MDL,
            "ref": f"{REF_DIR}/e5_place_ref.py@{REF_MD5[:8]}",
            "objects": shown,
        }
    return {
        "style": s,
        "color": list(COLOR),
        "emissive": EMISSIVE,
        "mdl": MDL,
        "ref": f"{REF_DIR}/e5_place_ref.py@{REF_MD5[:8]}",
        "objects": shown,
    }


def show(raw_env, binds, P0, tree_dir: Path, k: int, shift_z: float) -> dict | None:
    """After load_traj wrote the first state; one-clip sessions. P0 = the first state row the render writes ([D],
    world shift applied). k = this trajectory's loader clip index (session clip_ids). None when off (does nothing)."""
    s = style()
    if s is None:
        return None
    import numpy as np

    import omni.usd
    from pxr import UsdGeom

    PR, pt, dz, to_env = _marker_inputs(raw_env, tree_dir, k, shift_z)
    objs = [b for b in binds if b.name.startswith("object")]
    P0 = np.asarray(P0.detach().cpu().numpy() if hasattr(P0, "detach") else P0, float)
    stage = omni.usd.get_context().get_stage()
    origin = raw_env.scene.env_origins[0].detach().cpu().numpy().astype(float)
    UsdGeom.Xform.Define(stage, ROOT)
    shape, xcol = (s.split("-", 1) + [None])[:2]  # x-red -> ("x", "red") · otherwise the shape only
    mat = _x_material(stage, xcol) if shape in ("x", "pad") else _material(stage)
    shown = [
        _author_marker(
            raw_env,
            stage,
            tree_dir,
            n,
            v,
            s=s,
            shape=shape,
            xcol=xcol,
            mat=mat,
            objs=objs,
            P0=P0,
            origin=origin,
            PR=PR,
            to_env=to_env,
            dz=dz,
            shift_z=shift_z,
        )
        for n, v in pt.items()
    ]
    return _style_record(s, shape, xcol, shown)


def clear(stage) -> list[str]:
    """Remove the marker prims (everything under ROOT, materials included); off means no ROOT, nothing done. Returns
    the removed paths."""
    if stage.GetPrimAtPath(ROOT).IsValid():
        stage.RemovePrim(ROOT)
        return [ROOT]
    return []
