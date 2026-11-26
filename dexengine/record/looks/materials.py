"""Desk-top material binding and the blackbody colour helper of look D.

Desk top: look D's desk MDL (look_d.MDL['desk'], birch) is bound to the desk top with a world-space projection in
metres (world_or_object=True, texture_scale = 1/TILE_WORLD_M), independent of the asset's units. With the MDL
default (object space) a centimetre-unit desk asset repeats the pattern every 1 cm and only the mipmap average colour
remains. The augmentation textures (augvis.scene) use the same world projection.
Oracle: before binding, the measured length of one pattern tile (world: scene metres per unit / texture_scale;
object: world length of one object unit / texture_scale) must lie in TILE_M; the value is logged and stored in
render_meta (render_look.applied.desk_tile_m).
"""

from __future__ import annotations

from dexengine.core.errors import InputMissingError

MARK = "[COLLECT_DESK]"
TILE_WORLD_M = 1.0  # one pattern tile of the world projection (m)
TILE_M = (0.2, 5.0)  # allowed tile length (m); an object-space projection of a cm asset (0.01 m) fails


def _mesh(stage, prim_path: str):
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise InputMissingError(f"{MARK} no desk top {prim_path}")
    m = root if root.IsA(UsdGeom.Mesh) else next((p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh)), None)
    if m is None:
        raise InputMissingError(f"{MARK} no mesh under desk top {prim_path}")
    return m


def _world(stage, mesh):
    import numpy as np

    from pxr import Usd, UsdGeom

    M = np.asarray(UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), float)
    return M, float(UsdGeom.GetStageMetersPerUnit(stage))


def tile_m(stage, prim_path: str, space: str, scale: float) -> float:
    """Measured length (m) of one pattern tile: space 'world' (world projection) | 'object' (MDL default)."""
    import numpy as np

    M, mpu = _world(stage, _mesh(stage, prim_path))
    if space == "world":
        return mpu / scale
    if space == "object":  # world length of one object unit (mean row norm of the transform)
        return float(np.linalg.norm(M[:3, :3], axis=1).mean()) * mpu / scale
    raise SystemExit(f"{MARK} unknown projection space {space!r}")


def bind_desk(stage, prim_path: str, name: str, mdl: str) -> float:
    """Bind an MDL to the desk top with the world projection; dies when the tile oracle is outside TILE_M.
    Returns the tile length in m."""
    from pxr import Gf, Sdf

    from dexengine.record.looks import usd_bind as UB

    scale = 1.0 / TILE_WORLD_M
    t = tile_m(stage, prim_path, "world", scale)
    if not TILE_M[0] <= t <= TILE_M[1]:
        raise SystemExit(
            f"{MARK} desk top {prim_path} tile {t:.3f} m outside {TILE_M} (would render as the mipmap average colour)"
        )
    if not UB.bind_mdl(stage, prim_path, name, mdl, project=True):
        raise SystemExit(f"{MARK} desk top {prim_path} material binding failed")
    sh = stage.GetPrimAtPath(f"{UB.MATERIALS}/{name}/Shader")
    sh.CreateAttribute("inputs:world_or_object", Sdf.ValueTypeNames.Bool).Set(True)
    sh.CreateAttribute("inputs:texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(scale, scale))
    print(
        f"{MARK} desk top {prim_path} ← {mdl.rsplit('/', 1)[-1]} · world projection · tile {t:.2f} m"
        f" (gate {TILE_M[0]}–{TILE_M[1]} m)",
        flush=True,
    )
    return t


def desk_woods(stage) -> list[str]:
    """Desk-top prims (first 'Wood' under envs/*/Desk); look D and the augmentation share this choice."""
    from pxr import Usd

    out = []
    for env_prim in stage.GetPrimAtPath("/World/envs").GetChildren():
        desk = stage.GetPrimAtPath(f"{env_prim.GetPath()}/Desk")
        wood = next((p for p in Usd.PrimRange(desk) if p.GetName() == "Wood"), None) if desk.IsValid() else None
        if wood is not None:
            out.append(str(wood.GetPath()))
    return out


def blackbody_rgb(kelvin: float) -> list[float]:
    """Blackbody colour temperature -> linear sRGB (max 1): Kim et al. Planckian locus fit -> XYZ (Y=1) -> sRGB D65."""
    t = float(kelvin)
    if t <= 4000:
        x = -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390
    if t <= 2222:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    X, Y, Z = x / y, 1.0, (1 - x - y) / y
    rgb = [
        3.2406 * X - 1.5372 * Y - 0.4986 * Z,
        -0.9689 * X + 1.8758 * Y + 0.0415 * Z,
        0.0557 * X - 0.2040 * Y + 1.0570 * Z,
    ]
    return [round(c / max(rgb), 4) for c in rgb]
