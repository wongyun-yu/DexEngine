"""USD helpers of the record look: material bindings and the UV quad (pxr is imported inside the functions).

bind_solid  a solid UsdPreviewSurface on a prim and its descendants
bind_mdl    an MDL material (sub-identifier = file stem) on a prim and its descendants; a local MDL must exist
uv_quad     a single-sided quad mesh with UVs (st), visible from its counter-clockwise side
Every binding is strongerThanDescendants and lives under MATERIALS (/World/AugMaterials/<name>). Callers: look D
(looks.look_d), the desk top (looks.materials), the random looks (looks.augvis.scene) and the support discs
(scene.supports).
"""

from __future__ import annotations

import math
import os

MATERIALS = "/World/AugMaterials"


def bind_solid(stage, prim_path: str, name: str, rgb, rough, metal) -> bool:
    """Bind a solid UsdPreviewSurface to prim_path and its descendants; False if no prim."""
    from pxr import Gf, Sdf, UsdShade

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return False
    mpath = Sdf.Path(f"{MATERIALS}/{name}")
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


def bind_mdl(stage, prim_path: str, name: str, mdl: str, project: bool = False) -> bool:
    """Bind an MDL material (sub-identifier = file stem) to prim_path and its descendants.

    project=True turns on the project_uvw (box projection) input, which a prim without UVs (the floor plane) needs:
    rebinding without it renders a solid colour. Modules without that input ignore it. A local MDL path must exist
    before binding (RuntimeError otherwise); returns False if the prim does not exist.
    """
    from pxr import Sdf, UsdShade

    if "://" not in mdl and not os.path.isfile(mdl):  # a local MDL (look mirror) must exist
        raise RuntimeError(
            f"[COLLECT_LOOK] MDL file not found: {mdl} — for the local mirror run `python -m "
            f"dexengine.record.looks.look_d --mirror` (no S3 fallback)"
        )
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return False
    stem = mdl.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    mpath = Sdf.Path(f"{MATERIALS}/{name}")
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


def uv_quad(stage, path: str, pts, tile: float):
    """Single-sided quad mesh with UVs (st). pts = [p0..p3], counter-clockwise = visible side; tile = metres per UV
    tile."""
    from pxr import Gf, Sdf, UsdGeom, Vt

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    mesh.CreateDoubleSidedAttr(False)  # single-sided (dollhouse): invisible from outside
    e1 = [pts[1][i] - pts[0][i] for i in range(3)]
    e2 = [pts[3][i] - pts[0][i] for i in range(3)]
    lu = math.sqrt(sum(v * v for v in e1)) / tile
    lv = math.sqrt(sum(v * v for v in e2)) / tile
    uv = [(0, 0), (lu, 0), (lu, lv), (0, lv)]
    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    pv.Set(Vt.Vec2fArray([Gf.Vec2f(*t) for t in uv]))
    return mesh
