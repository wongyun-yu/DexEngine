"""Render scene look: lighting and material overrides applied to the finished stage, switched on by env knobs only.

The scene visuals (dome light, desk and floor materials) come from the env cfg shared by training and evaluation;
changing them there would change training. So the look is applied afterwards, on the built stage, as material bindings
(strongerThanDescendants) and light attributes. With no knob set nothing changes.

Knobs (all optional; values of the render look: dexengine.record.looks.render_look):
  DEXENGINE_AUG_LIGHT="intensity[,r,g,b]"      e.g. "900" or "700,1.0,0.95,0.9" (/World/light)
  DEXENGINE_AUG_ROOM="wall[|half m[|height m[|cx[|cy]]]]"  4-wall room around the robot (see _spawn_room)
  DEXENGINE_AUG_FLOOR_MDL=<mdl url|path|image> MDL (box-projected) or image material on the floor
  DEXENGINE_AUG_DESK_MDL=<mdl url|path|image>  material on the desk top (the "Wood" prim of envs/*/Desk)
The dome light is the env cfg's (colour and intensity, no texture); this module only sets its intensity and colour
(DEXENGINE_AUG_LIGHT). Removed knobs (no producer: solid DEXENGINE_AUG_FLOOR /
DEXENGINE_AUG_DESK, DEXENGINE_AUG_PROPS, DEXENGINE_DOME_YAW, the refused DEXENGINE_AUG_ROBOT(_STEEL)) fall under the
rule below: set = not applied = error.
DEXENGINE_LOG_EGO_PITCH=1 additionally prints the measured ego camera pitch.

apply_from_env is the single entry (per-clip policy render, replay render and the record render all call it). Every
knob that is switched on (requested(): a DEXENGINE_AUG_* or DEXENGINE_DOME_YAW value that is neither empty nor "0")
must be applied, otherwise it raises — a missing target prim or an unknown knob included. It ends with the MARK line
(applied n/n and the knob names); callers do not swallow its errors.
"""

from __future__ import annotations

import os

MARK = "[VISUAL_AUG]"
_EXTRA_KNOBS = ("DEXENGINE_DOME_YAW",)  # outside the DEXENGINE_AUG_ prefix: removed, listed so that setting it dies


def requested(env=None) -> list[str]:
    """Knobs switched on: every DEXENGINE_AUG_* and _EXTRA_KNOBS entry whose value is neither empty nor '0' (a prefix
    rule, not a name table)."""
    env = os.environ if env is None else env
    return sorted(
        k
        for k, v in env.items()
        if (k.startswith("DEXENGINE_AUG_") or k in _EXTRA_KNOBS) and str(v).strip() not in ("", "0")
    )


def _parse(v: str) -> list[float]:
    """Comma list of floats."""
    return [float(x) for x in v.split(",") if x.strip() != ""]


def _bind_solid(stage, prim_path: str, name: str, rgb, rough, metal):
    """Bind a solid UsdPreviewSurface to prim_path and its descendants (strongerThanDescendants); False if no prim."""
    from pxr import Gf, Sdf, UsdShade

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return False
    mpath = Sdf.Path(f"/World/AugMaterials/{name}")
    mat = UsdShade.Material.Define(stage, mpath)
    sh = UsdShade.Shader.Define(stage, mpath.AppendChild("pbr"))
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metal))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    api.Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    return True


def _bind_mdl(stage, prim_path: str, name: str, mdl: str, project: bool = False) -> bool:
    """Bind an MDL material (sub-identifier = file stem) to prim_path and its descendants (strongerThanDescendants).

    project=True turns on the project_uvw (box projection) input, which a prim without UVs (the floor plane) needs:
    rebinding without it renders a solid colour. Modules without that input (vMaterials_2) ignore it. A local MDL
    path must exist before binding (RuntimeError otherwise); returns False if the prim does not exist.
    """
    from pxr import Sdf, UsdShade

    if "://" not in mdl and not os.path.isfile(mdl):  # a local MDL (background mirror) must exist
        raise RuntimeError(
            f"[visual_aug] MDL file not found: {mdl} — for the local mirror run `python "
            f"dexengine/record/looks/look_d.py --mirror` (no S3 fallback)"
        )
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return False
    stem = mdl.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    mpath = Sdf.Path(f"/World/AugMaterials/{name}")
    mat = UsdShade.Material.Define(stage, mpath)
    sh = UsdShade.Shader.Define(stage, mpath.AppendChild("Shader"))
    sh.SetSourceAsset(mdl, "mdl")
    sh.SetSourceAssetSubIdentifier(stem, "mdl")
    if project:
        sh.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
    out = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(out)
    mat.CreateDisplacementOutput("mdl").ConnectToSource(out)
    mat.CreateVolumeOutput("mdl").ConnectToSource(out)
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    api.Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    return True


def _bind_image(stage, prim_path: str, name: str, img: str, rough: float = 0.7) -> bool:
    """Bind an image texture (UsdPreviewSurface + UsdUVTexture, repeat wrap) to prim_path; needs UVs (st)."""
    from pxr import Sdf, UsdShade

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return False
    mpath = Sdf.Path(f"/World/AugMaterials/{name}")
    mat = UsdShade.Material.Define(stage, mpath)
    pbr = UsdShade.Shader.Define(stage, mpath.AppendChild("pbr"))
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
    st = UsdShade.Shader.Define(stage, mpath.AppendChild("st"))
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, mpath.AppendChild("tex"))
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(img)
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        st.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    )
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    api.Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    return True


def _bind_any(stage, prim_path: str, name: str, src: str) -> bool:
    """Bind by source kind: "solid:r,g,b[,rough]" -> solid colour, *.mdl -> MDL, anything else -> image."""
    if src.startswith("solid:"):
        xs = [float(v) for v in src[6:].split(",")]
        return _bind_solid(stage, prim_path, name, tuple(xs[:3]), xs[3] if len(xs) > 3 else 0.7, 0.0)
    if src.lower().endswith(".mdl"):
        return _bind_mdl(stage, prim_path, name, src)
    return _bind_image(stage, prim_path, name, src)


def _uv_quad(stage, path: str, pts, normal_axis: str, tile: float):
    """Single-sided quad mesh with UVs (st). pts = [p0..p3], counter-clockwise = visible side; tile = metres per UV
    tile."""
    from pxr import Gf, Sdf, UsdGeom, Vt

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    mesh.CreateDoubleSidedAttr(False)  # single-sided (dollhouse): invisible from outside
    import math as _m

    e1 = [pts[1][i] - pts[0][i] for i in range(3)]
    e2 = [pts[3][i] - pts[0][i] for i in range(3)]
    lu = _m.sqrt(sum(v * v for v in e1)) / tile
    lv = _m.sqrt(sum(v * v for v in e2)) / tile
    uv = [(0, 0), (lu, 0), (lu, lv), (0, lv)]
    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    pv.Set(Vt.Vec2fArray([Gf.Vec2f(*t) for t in uv]))
    return mesh


def _spawn_room(stage, spec: str, root: str = "/World/AugRoom", quiet: bool = False) -> bool:
    """Room around the robot: four wall quads, no ceiling. All quads carry UVs, so MDL and image materials both apply;
    walls are single-sided facing inwards (dollhouse), so they never hide the scene from a camera outside.

    spec = "wall material[|half side m[|height m[|centre x[|centre y]]]]" (defaults 3, 2.6, -0.7, 0: a 6 x 6 m room).
    The separator is | because a material such as "solid:r,g,b" contains commas. Returns whether every wall bound.
    """
    from pxr import UsdGeom

    parts = [s.strip() for s in spec.split("|")]
    wall_src = parts[0]
    half = float(parts[1]) if len(parts) > 1 else 3.0
    h = float(parts[2]) if len(parts) > 2 else 2.6
    cx = float(parts[3]) if len(parts) > 3 else -0.7
    cy = float(parts[4]) if len(parts) > 4 else 0.0
    UsdGeom.Xform.Define(stage, root)
    x0, x1, y0, y1 = cx - half, cx + half, cy - half, cy + half
    # each wall counter-clockwise, so the side facing the room centre is the visible one
    walls = {
        "wall_ny": [(x0, y0, 0), (x1, y0, 0), (x1, y0, h), (x0, y0, h)],  # -y wall, faces +y
        "wall_py": [(x1, y1, 0), (x0, y1, 0), (x0, y1, h), (x1, y1, h)],  # +y wall
        "wall_nx": [(x0, y1, 0), (x0, y0, 0), (x0, y0, h), (x0, y1, h)],  # -x wall
        "wall_px": [(x1, y0, 0), (x1, y1, 0), (x1, y1, h), (x1, y0, h)],  # +x wall
    }
    ok = bool(wall_src)
    for nm, pts in walls.items():
        _uv_quad(stage, f"{root}/{nm}", pts, "z", tile=1.5)
        if wall_src:
            ok = _bind_any(stage, f"{root}/{nm}", f"{nm}_mat", wall_src) and ok
    # The floor gets no quad: /World/Floor itself is rebound (DEXENGINE_AUG_FLOOR_MDL). That plane has no UVs, so
    # its material must be world-space projected (the .../Materials/Base/... family).
    if not quiet:
        print(
            f"[visual_aug] room {2 * half:.0f}x{2 * half:.0f}x{h}m @({cx},{cy}) ← walls "
            f"{wall_src.rsplit('/', 1)[-1] if wall_src else '(no material)'}",
            flush=True,
        )
    return ok


def _log_ego_pitch(stage) -> None:
    """Print the measured optical axis of the stage's HeadCamera (DEXENGINE_LOG_EGO_PITCH=1; read-only).

    The ego camera's downward pitch is set by a code constant and logged nowhere. The camera's local -Z axis is printed
    in its parent frame and in world, with the derived downward angles; two runs that differ only in the pitch differ
    by exactly the applied pitch, with no convention to infer.
    """
    import math as _m

    from pxr import Gf, Usd, UsdGeom

    cam = next((p for p in Usd.PrimRange(stage.GetPseudoRoot()) if p.GetName() == "HeadCamera"), None)
    if cam is None:
        print("[visual_aug] ego pitch: no HeadCamera prim — not measured", flush=True)
        return
    xc = UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    par = cam.GetParent()
    xp = UsdGeom.Xformable(par).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    loc = xc * xp.GetInverse()
    f_loc = loc.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)).GetNormalized()
    f_wld = xc.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)).GetNormalized()
    env_v = "(unset→code default 30°, head joint)"
    print(
        f"[visual_aug] ego pitch env={env_v} | optical axis local=({f_loc[0]:+.4f},"
        f"{f_loc[1]:+.4f},{f_loc[2]:+.4f}) local down={-_m.degrees(_m.asin(max(-1.0, min(1.0, f_loc[2])))):+.2f}° | "
        f"world down={-_m.degrees(_m.asin(max(-1.0, min(1.0, f_wld[2])))):+.2f}° @ {cam.GetPath()}",
        flush=True,
    )


def apply_from_env(stage, per_env_room: bool = False) -> list[str]:
    """Apply every switched-on knob to the stage and return the applied knob names (sorted).

    per_env_room=True (replay render, several envs drawn at once) builds the room inside every env prim in env-local
    coordinates; False builds one /World/AugRoom. Raises when a switched-on knob was not applied (module docstring).
    """
    from pxr import Gf

    want, done = requested(), set()

    if os.environ.get("DEXENGINE_LOG_EGO_PITCH", "0") == "1":
        try:
            _log_ego_pitch(stage)
        except Exception as _e_ep:  # noqa: BLE001 — a failed measurement must not stop the look
            print(f"[visual_aug] ego pitch measurement failed: {_e_ep}", flush=True)

    lit = os.environ.get("DEXENGINE_AUG_LIGHT", "")
    if lit:
        xs = _parse(lit)
        lp = stage.GetPrimAtPath("/World/light")
        if lp and lp.IsValid():
            lp.GetAttribute("inputs:intensity").Set(float(xs[0]))
            if len(xs) >= 4:
                lp.GetAttribute("inputs:color").Set(Gf.Vec3f(*xs[1:4]))
            print(f"[visual_aug] light ← {xs}", flush=True)
            done.add("DEXENGINE_AUG_LIGHT")

    return _apply_room(stage, per_env_room, done, want)


def _apply_room(stage, per_env_room, done, want):
    """Room before the floor and desk materials."""
    rm = os.environ.get("DEXENGINE_AUG_ROOM", "")
    if rm and not per_env_room:
        if _spawn_room(stage, rm):
            done.add("DEXENGINE_AUG_ROOM")
    elif rm:
        envs = stage.GetPrimAtPath("/World/envs")
        kids = list(envs.GetChildren()) if envs and envs.IsValid() else []
        if not kids:
            raise RuntimeError("[visual_aug] per_env_room but /World/envs has no env prims")
        # every env gets its room, so the list is built in full before `all` reads it
        spawned = [
            _spawn_room(stage, rm, root=f"{env_prim.GetPath()}/AugRoom", quiet=k > 0) for k, env_prim in enumerate(kids)
        ]
        if all(spawned):
            done.add("DEXENGINE_AUG_ROOM")
        print(f"[visual_aug] room per env ×{len(kids)}(env-local)", flush=True)

    _apply_floor_desk_materials(stage, done)

    miss = [k for k in want if k not in done]
    if "DEXENGINE_AUG_FLOOR_MDL" not in done:  # the scene floor has no material of its own (cfg/scene.py)
        raise RuntimeError(
            f"{MARK} a render needs the render look's floor (DEXENGINE_AUG_FLOOR_MDL, record.looks.render_look) — "
            f"applied {sorted(done)}"
        )
    if miss:
        raise RuntimeError(
            f"{MARK} knobs on but not applied {miss} · applied {sorted(done)} — "
            "one of: target prim missing · material binding failed · unknown knob"
        )
    print(f"{MARK} applied {len(done)}/{len(want)} {sorted(done)} · per_env_room={per_env_room}", flush=True)
    return sorted(done)


def _apply_floor_desk_materials(stage, done):
    """Floor / desk materials (.mdl or image)."""
    flm = os.environ.get("DEXENGINE_AUG_FLOOR_MDL", "")
    if flm:
        ok = (
            _bind_mdl(stage, "/World/Floor", "floor_mat", flm, project=True)
            if flm.lower().endswith(".mdl")
            else _bind_any(stage, "/World/Floor", "floor_mat", flm)
        )
        if ok:
            print(f"[visual_aug] floor ← {flm.rsplit('/', 1)[-1]} (project_uvw)", flush=True)
            done.add("DEXENGINE_AUG_FLOOR_MDL")
    # Only the desk top (the prim named "Wood" in the Desk subtree) is rebound; legs (Metal) and pads keep their
    # look. A desk without a Wood prim is not bound at all (no whole-desk fallback), and the knob then counts as not
    # applied.
    dkm = os.environ.get("DEXENGINE_AUG_DESK_MDL", "")
    if dkm:
        from pxr import Usd as _Usd

        n, miss = 0, 0
        root = stage.GetPrimAtPath("/World/envs")
        if root and root.IsValid():
            for env_prim in root.GetChildren():
                desk = stage.GetPrimAtPath(f"{env_prim.GetPath()}/Desk")
                if not desk or not desk.IsValid():
                    continue
                wood = next((p for p in _Usd.PrimRange(desk) if p.GetName() == "Wood"), None)
                if wood is None:
                    miss += 1
                    continue
                if _bind_any(stage, str(wood.GetPath()), f"desk_top_{n}", dkm):
                    n += 1
        if n:
            print(f"[visual_aug] desk top (Wood) only ×{n} ← {dkm.rsplit('/', 1)[-1]}", flush=True)
        if miss:
            print(
                f"[visual_aug] Desk without a Wood prim ×{miss} — desk top not replaced (legs kept) · not applied",
                flush=True,
            )
        elif n:
            done.add("DEXENGINE_AUG_DESK_MDL")
