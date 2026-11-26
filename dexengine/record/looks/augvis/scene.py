"""Record render visual augment (experiments only, not the default path), scene side: asset root, materials, light,
swept volume and distractors.

augvis.render (render order, measurement, outputs) calls these functions. Values (axes, options, rules) are owned by
augvis.catalog.
  materials      OmniPBR (world projection, texture_scale = 1/measured tile m) or the look D MDL (look_d mirror):
                 top, floor, wall
  light          intensity scale and colour temperature of the 9 look D panels + camera white balance (AC.wb_kelvin);
                 structure and exposure unchanged
  swept volume   'lowest z downward' height field of the top grid (GRID_M): robot link origins + joint pair segments
                 (R_LINK) · objects (OBJ_PAD) · supports
  distractors    Poly Haven models (materials rebound to OmniPBR) · top + height + CLEAR_M < height field over the
                 whole footprint · within EDGE_M of the top edge · dropped when a sight line crosses the cylinder ·
                 TRIES place attempts
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from dexengine.core.errors import InputMissingError
from dexengine.record.looks import look_d as LOOK
from dexengine.record.looks import materials as DM
from dexengine.record.looks.augvis import catalog as AC
from dexengine.record.looks.augvis import color as AVC
from dexengine.record.looks.augvis import sources as SR  # owner of the canonical asset generation (ASSET_GEN)

MARK = AC.MARK


def assets_root() -> Path:
    """Asset root (env AUGVIS_ASSETS, no default; manifest.json lists the assets) · canonical generation check by
    content (augvis.sources asset_gen_line — SHA256SUMS · marker line)."""
    r = os.environ.get("AUGVIS_ASSETS", "")
    if not r or not (Path(r) / "manifest.json").is_file():
        raise InputMissingError(
            f"{MARK} AUGVIS_ASSETS({r!r})/manifest.json missing: run augvis.assets first (no default)"
        )
    print(SR.asset_gen_line(r), flush=True)
    return Path(r)


# ───────────────────────────── materials · light ────────────────────
def _omnipbr(
    stage, prim_path: str, name: str, *, diff=None, rough_tex=None, nrm=None, rgb=None, rough=0.5, tint=None, tile_m=1.0
) -> bool:
    from pxr import Gf, Sdf, UsdShade

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return False
    mpath = Sdf.Path(f"/World/AugMaterials/{name}")
    mat = UsdShade.Material.Define(stage, mpath)
    sh = UsdShade.Shader.Define(stage, mpath.AppendChild("Shader"))
    sh.SetSourceAsset("OmniPBR.mdl", "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
    sh.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(True)
    s = 1.0 / float(tile_m)
    sh.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(s, s))

    def tex(nm, path, raw):
        inp = sh.CreateInput(nm, Sdf.ValueTypeNames.Asset)
        inp.Set(Sdf.AssetPath(str(path)))
        inp.GetAttr().SetMetadata("colorSpace", "raw" if raw else "sRGB")

    if diff:
        tex("diffuse_texture", diff, False)
    else:
        sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    if tint:
        sh.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*tint))
    if rough_tex:
        tex("reflectionroughness_texture", rough_tex, True)
        sh.CreateInput("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float).Set(1.0)
    else:
        sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(rough))
    if nrm:
        tex("normalmap_texture", nrm, True)
    out = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(out)
    mat.CreateDisplacementOutput("mdl").ConnectToSource(out)
    mat.CreateVolumeOutput("mdl").ConnectToSource(out)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    return True


def bind_entry(
    stage, prim_path: str, name: str, entry: dict, look_key: str, man: dict, root: Path, tint=(0, 0.0)
) -> bool:
    """Binds one material + tint (AVC.tint_rgb, linear multiply with the fixed tint) to a prim."""
    from dexengine.record.looks import usd_bind as UB

    # top = the desk material (world m projection) · floor and walls = object coordinate projection
    if entry["kind"] == "look_d":
        if look_key == "desk":
            return bool(DM.bind_desk(stage, prim_path, name, LOOK.local(LOOK.MDL[look_key])))
        return UB.bind_mdl(stage, prim_path, name, LOOK.local(LOOK.MDL[look_key]), project=True)
    t2 = AVC.tint_rgb(tint)
    if entry["kind"] == "solid":
        return _omnipbr(
            stage, prim_path, name, rgb=[entry["rgb"][i] * t2[i] for i in range(3)], rough=entry.get("rough", 0.5)
        )
    t = man["textures"][entry["id"]]
    dims = t.get("dims_mm") or [1000, 1000]
    t1 = entry.get("tint", (1.0, 1.0, 1.0))
    f = t["files"]
    return _omnipbr(
        stage,
        prim_path,
        name,
        diff=root / f["diff"],
        rough_tex=(root / f["rough"]) if "rough" in f else None,
        nrm=(root / f["nor_gl"]) if "nor_gl" in f else None,
        rough=0.6,
        tint=tuple(t1[i] * t2[i] for i in range(3)),
        tile_m=float(dims[0]) / 1000.0,
    )


ROOM_SHADOW_PRIMS = (
    "/World/CollectLook/ceiling",
    "/World/AugRoom/wall_nx",
    "/World/AugRoom/wall_px",
    "/World/AugRoom/wall_ny",
    "/World/AugRoom/wall_py",
)


def _set_room_open(stage, open_: bool) -> None:
    """Lets the dome light into the closed room by making the ceiling and the 4 walls cast no shadow (open_=False keeps
    look D). The walls keep bouncing light (invisibleToSecondaryRays would also remove wall reflections and indirect
    light, so it is not used)."""
    from pxr import Sdf, UsdGeom

    for pth in ROOM_SHADOW_PRIMS:
        prim = stage.GetPrimAtPath(pth)
        if not prim.IsValid():
            raise InputMissingError(f"{MARK} room prim {pth} missing")
        for sub in [prim, *prim.GetChildren()]:
            UsdGeom.PrimvarsAPI(sub).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(open_)


def panel_gains(on, desk_pt, room_h: float) -> list[float]:
    """Lit panel pattern -> per-panel scale so the top centre illuminance (small area light approximation E ∝ dz² /
    d⁴) equals look D with all 9 on."""
    w = []
    for x, y in LOOK.PANEL_XY:
        dz = room_h - float(desk_pt[2])
        d2 = (x - desk_pt[0]) ** 2 + (y - desk_pt[1]) ** 2 + dz * dz
        w.append(dz * dz / (d2 * d2))
    k = sum(w) / sum(w[i] for i in on)
    return [k if i in on else 0.0 for i in range(len(w))]


# the canonical dome (look D leaves it at intensity 0); RTX uses one dome only, none is created
DOME_PRIM = LOOK.CANON_LIGHTS[0]


def _dome(stage, dome: str, spec: dict, man: dict, root: Path) -> None:
    """Drives the one canonical dome prim: dome=none keeps look D (intensity 0), otherwise HDRI · white · intensity ·
    yaw."""
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    d = UsdLux.DomeLight(stage.GetPrimAtPath(DOME_PRIM))
    if not d:
        raise InputMissingError(f"{MARK} canonical dome {DOME_PRIM} missing")
    if dome == "none":
        d.GetIntensityAttr().Set(0.0)
        return
    h = man["hdris"][dome]
    d.CreateTextureFileAttr().Set(Sdf.AssetPath(str(root / h["file"])))
    d.CreateTextureFormatAttr().Set(UsdLux.Tokens.latlong)
    d.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    d.GetIntensityAttr().Set(AC.DOME_BASE * float(spec["dome_scale"]) / max(float(h["mean_lum"]), 1e-4))
    xf = UsdGeom.Xformable(d)
    ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
    yaw = float(spec.get("dome_yaw") or 0)
    if "xformOp:rotateZ:augvis" in ops:
        ops["xformOp:rotateZ:augvis"].Set(yaw)
    else:
        xf.AddRotateZOp(opSuffix="augvis").Set(yaw)


def _lights(stage, spec: dict, man: dict, root: Path, desk_pt) -> dict:
    """Panels (pattern · top illuminance correction · intensity · jitter · colour temperature) · dome · WB ·
    saturation."""
    from pxr import UsdLux

    on = AC.PANEL_PATTERNS[spec.get("panels", "all9")]
    jit = spec.get("jitter") or [1.0] * len(LOOK.PANEL_XY)
    gains = panel_gains(on, desk_pt, LOOK._room(stage)["h"])
    base = LOOK.PANEL_W / math.pi * float(spec["light_scale"])
    for i in range(len(LOOK.PANEL_XY)):
        lt = UsdLux.RectLight(stage.GetPrimAtPath(f"/World/CollectLook/panel_{i}"))
        lt.GetIntensityAttr().Set(base * gains[i] * float(jit[i]))
        lt.GetColorTemperatureAttr().Set(float(spec["light_k"]))
    dome = spec.get("dome", "none")
    _dome(stage, dome, spec, man, root)
    _set_room_open(stage, dome != "none")
    LOOK._set_many(
        {LOOK.WHITE_K: LOOK.blackbody_rgb(AC.wb_kelvin(spec["light_k"])), AC.SAT_KEY: float(spec.get("sat", AC.D_SAT))}
    )
    return {"panels": len(on), "gains": [round(g, 3) for g in gains], "dome": dome}


def apply_materials_lights(stage, spec: dict, man: dict, root: Path, aerial: bool, desk_pt) -> dict:
    """Material names are new per variation (redefining a shader at the same path keeps the previous variation's
    inputs)."""
    import re

    tabs = AC.materials(man)
    ok = {}
    z = (0, 0.0)
    tag = re.sub(r"[^A-Za-z0-9_]", "_", spec["name"])
    if not aerial:
        ent = tabs["desk"][spec["desk"]]
        ok["desk"] = all(
            bind_entry(stage, w, f"augvis_{tag}_desk_{i}", ent, "desk", man, root, spec.get("desk_tint", z))
            for i, w in enumerate(DM.desk_woods(stage))
        )
    ok["floor"] = bind_entry(
        stage,
        "/World/Floor",
        f"augvis_{tag}_floor",
        tabs["floor"][spec["floor"]],
        "floor",
        man,
        root,
        spec.get("floor_tint", z),
    )
    went = tabs["wall"][spec["wall"]]
    ok["wall"] = all(
        bind_entry(
            stage, f"/World/AugRoom/{nm}", f"augvis_{tag}_{nm}", went, "wall", man, root, spec.get("wall_tint", z)
        )
        for nm in ("wall_nx", "wall_px", "wall_ny", "wall_py")
    )
    ok["lights"] = _lights(stage, spec, man, root, desk_pt)
    if not all(ok.values()):
        raise SystemExit(f"{MARK} {spec['name']} apply failed {ok}")
    return ok


def paint_objects(stage, roots: list[str], rgb, tag: str, rough: float | None = None) -> dict:
    """Flat colour object look: binds one flat OmniPBR (diffuse_color_constant = rgb · no texture · roughness rough,
    0.5 by default) to every mesh under roots (task object root prims), strongest (strongerThanDescendants) per root.
    rgb = None removes the earlier paint bindings (the next look of the same session).
    Returns {"roots": n, "meshes": m, "material": path|None}; the [OBJ_COLOR] marker is printed by the caller
    (augvis.render apply_spec)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    if rgb is None:
        for r in roots:
            prim = stage.GetPrimAtPath(r)
            if prim and prim.IsValid():
                UsdShade.MaterialBindingAPI(prim).UnbindDirectBinding()
        return {"roots": len(roots), "meshes": 0, "material": None}
    mp = Sdf.Path(f"/World/AugMaterials/augvis_{tag}_objcolor")  # look layer name (augvis_ = prim oracle LOOK_OWNED)
    if not stage.GetPrimAtPath(mp).IsValid():
        m = UsdShade.Material.Define(stage, mp)
        sh = UsdShade.Shader.Define(stage, mp.AppendChild("Shader"))
        sh.SetSourceAsset("OmniPBR.mdl", "mdl")
        sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in rgb]))
        sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(
            0.5 if rough is None else float(rough)
        )
        sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
        m.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    mat = UsdShade.Material(stage.GetPrimAtPath(mp))
    n_mesh = 0
    for r in roots:
        prim = stage.GetPrimAtPath(r)
        if not prim or not prim.IsValid():
            raise InputMissingError(f"[OBJ_COLOR] object root {r} missing: cannot paint")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
        n_mesh += sum(1 for q in Usd.PrimRange(prim) if q.IsA(UsdGeom.Mesh))
    if n_mesh == 0:
        raise InputMissingError(f"[OBJ_COLOR] 0 meshes under object roots {roots}: nothing to paint")
    return {"roots": len(roots), "meshes": n_mesh, "material": str(mp)}


def prim_obj_id(stage, path: str) -> str | None:
    """Object id of a root prim = <id> of the composed layer path …/<id>/object[_render].usd; None unless exactly one
    (for an object to keep the caller dies)."""
    import re

    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return None
    ids = set()
    for s in prim.GetPrimStack():
        m = re.search(r"/([^/]+)/object(?:_render)?\.usd$", s.layer.realPath or s.layer.identifier)
        if m:
            ids.add(m[1])
    return ids.pop() if len(ids) == 1 else None


def kept_state(stage, roots: list[str]) -> dict:
    """Real state of the kept roots: meshes · meshes bound to the paint material (…_objcolor of paint_objects) ·
    meshes bound to materials tint_objects changed."""
    from pxr import Usd, UsdGeom, UsdShade

    touched = [str(a.GetPath()) for a, _ in _TINT_UNDO if a]
    r = {"meshes": 0, "painted": 0, "tinted": 0}
    for root in roots:
        prim = stage.GetPrimAtPath(root)
        if not prim or not prim.IsValid():
            continue
        for q in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if not q.IsA(UsdGeom.Mesh):
                continue
            r["meshes"] += 1
            mat = UsdShade.MaterialBindingAPI(q).ComputeBoundMaterial()[0]
            mp = str(mat.GetPath()) if mat else ""
            r["painted"] += mp.endswith("_objcolor")
            r["tinted"] += bool(mp) and any(t.startswith(mp + "/") for t in touched)
    return r


#: values tint_objects changed (attribute · old value; no old value = removed). Restored before the next look of the
#: same session or before applying the same look again (so multiplies do not pile up).
UNTINTED = "untinted:"  # prefix of apply_mat results that could not be tinted (counted per reason)
_TINT_UNDO: list = []


def _undo_tints():
    """Restore the attributes an earlier tint scaled (the Kit reuse worker gets a new stage per job)."""
    for attr, old in reversed(_TINT_UNDO):
        # attributes of an earlier stage (the Kit reuse worker gets a new stage per job)
        if not attr:
            continue
        if old is None:
            attr.Clear()
        else:
            attr.Set(old)
    _TINT_UNDO.clear()


def _tint_meshes(roots, stage, r, apply_mat):
    """Every visible mesh under the roots: its bound material tinted once (counts per path taken in r)."""
    from pxr import Usd, UsdGeom, UsdShade

    seen: dict[str, str] = {}
    for root in roots:
        prim = stage.GetPrimAtPath(root)
        if not prim or not prim.IsValid():
            raise InputMissingError(f"[OBJ_TINT] object root {root} missing: cannot tint")
        for q in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if not q.IsA(UsdGeom.Mesh):
                continue
            im = UsdGeom.Imageable(q)
            if (
                im.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render)
                or im.ComputeVisibility() == UsdGeom.Tokens.invisible
            ):
                continue  # collision and hidden meshes are not visible
            r["meshes"] += 1
            mat = UsdShade.MaterialBindingAPI(q).ComputeBoundMaterial()[0]
            how = UNTINTED + "no material"
            if mat:
                mp = str(mat.GetPath())
                how = seen[mp] if mp in seen else seen.setdefault(mp, apply_mat(mat))
            if how.startswith(UNTINTED):
                r["untinted"] += 1
                r["why"][how[len(UNTINTED) :]] = r["why"].get(how[len(UNTINTED) :], 0) + 1
            else:
                r[how] += 1


def tint_objects(stage, roots: list[str], tint) -> dict:
    """Original texture look: per-channel tint on the material bound to every visible mesh under roots (each material
    once). Materials and bindings stay: UsdPreviewSurface + UsdUVTexture (the cover bake form) = texture scale ·
    constant diffuseColor = value x tint · OmniPBR = diffuse_tint. Meshes that cannot be tinted (no material, material
    inside an instance, unknown shader) do not die; they are counted and returned (the marker is printed by the
    caller). tint = None = restore only."""
    from pxr import Gf, Sdf, UsdShade

    _undo_tints()
    r = {"roots": len(roots), "meshes": 0, "texture": 0, "constant": 0, "mdl": 0, "untinted": 0, "why": {}}
    if tint is None:
        return r
    t = [float(c) for c in tint]

    def scaled(attr, old, new):
        _TINT_UNDO.append((attr, old))
        attr.Set(new)

    def apply_mat(mat) -> str:
        """One material; returns the path taken (texture | constant | mdl) or UNTINTED + reason."""
        if mat.GetPrim().IsInstanceProxy():
            return UNTINTED + "material inside an instance"
        sh = mat.ComputeSurfaceSource("mdl")[0]
        if not sh or not sh.GetPrim().IsValid():
            return UNTINTED + "no surface shader"
        if sh.GetImplementationSource() == "sourceAsset":
            sub = sh.GetSourceAssetSubIdentifier("mdl")
            if sub != "OmniPBR":
                return f"{UNTINTED}MDL {sub}"
            inp = sh.GetInput("diffuse_tint") or sh.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f)
            old = inp.GetAttr().Get()
            base = old if old is not None else Gf.Vec3f(1.0, 1.0, 1.0)
            scaled(inp.GetAttr(), old, Gf.Vec3f(*[base[i] * t[i] for i in range(3)]))
            return "mdl"
        sid = sh.GetIdAttr().Get()
        if sid != "UsdPreviewSurface":
            return f"{UNTINTED}shader {sid}"
        inp = sh.GetInput("diffuseColor") or sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        src = [a for a in inp.GetValueProducingAttributes() if UsdShade.Output.IsOutput(a)]
        if src:
            tx = UsdShade.Shader(src[0].GetPrim())
            if tx.GetIdAttr().Get() != "UsdUVTexture":
                return f"{UNTINTED}texture shader {tx.GetIdAttr().Get()}"
            sc = tx.GetInput("scale") or tx.CreateInput("scale", Sdf.ValueTypeNames.Float4)
            old = sc.GetAttr().Get()
            base = old if old is not None else Gf.Vec4f(1.0, 1.0, 1.0, 1.0)
            scaled(sc.GetAttr(), old, Gf.Vec4f(base[0] * t[0], base[1] * t[1], base[2] * t[2], base[3]))
            return "texture"
        old = inp.GetAttr().Get()
        base = old if old is not None else Gf.Vec3f(0.18, 0.18, 0.18)  # UsdPreviewSurface default
        scaled(inp.GetAttr(), old, Gf.Vec3f(*[base[i] * t[i] for i in range(3)]))
        return "constant"

    _tint_meshes(roots, stage, r, apply_mat)
    if r["meshes"] == 0:
        raise InputMissingError(f"[OBJ_TINT] 0 visible meshes under object roots {roots}: nothing to tint")
    return r
