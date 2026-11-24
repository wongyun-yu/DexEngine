"""GSO model conversion for the visual-augmentation asset root: OBJ + texture → usda (the asset fetch
(augvis.assets.gso_fetch) calls it; re-exported there)."""

from __future__ import annotations

from pathlib import Path


def obj_to_usd(obj: Path, tex: Path, usd: Path) -> list[float]:
    """GSO OBJ (metres, Z up) + texture -> usda (Mesh · faceVarying st · normals · UsdPreviewSurface network).
    Returns the extent size (m)."""
    import numpy as np

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    V, VT, VN, fv, ft, fn, cnt = [], [], [], [], [], [], []
    for ln in obj.read_text().splitlines():
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("vt "):
            VT.append([float(x) for x in ln.split()[1:3]])
        elif ln.startswith("vn "):
            VN.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "):
            toks = [t.split("/") for t in ln.split()[1:]]
            cnt.append(len(toks))
            for t in toks:
                fv.append(int(t[0]) - 1)
                ft.append(int(t[1]) - 1 if len(t) > 1 and t[1] else 0)
                fn.append(int(t[2]) - 1 if len(t) > 2 and t[2] else 0)
    P = np.asarray(V, float)
    st = Usd.Stage.CreateNew(str(usd))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, "/model")
    st.SetDefaultPrim(root.GetPrim())
    m = UsdGeom.Mesh.Define(st, "/model/mesh")
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in P]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray(cnt))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(fv))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    if VN:
        m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*VN[i]) for i in fn]))
        m.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    pv.Set(Vt.Vec2fArray([Gf.Vec2f(*t) for t in VT]))
    pv.SetIndices(Vt.IntArray(ft))
    mat = UsdShade.Material.Define(st, "/model/mat")
    sh = UsdShade.Shader.Define(st, "/model/mat/pbr")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    rd = UsdShade.Shader.Define(st, "/model/mat/st")
    rd.CreateIdAttr("UsdPrimvarReader_float2")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tx = UsdShade.Shader.Define(st, "/model/mat/tex")
    tx.CreateIdAttr("UsdUVTexture")
    tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f"./{tex.name}"))
    tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    st.GetRootLayer().Save()
    return (P.max(0) - P.min(0)).round(4).tolist()
