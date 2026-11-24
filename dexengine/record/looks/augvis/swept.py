"""Visual augmentation, swept volume and distractors: the downward-lowest-z height field over the desk grid (GRID_M)
from robot link origins plus joint-pair segments (R_LINK), objects (OBJ_PAD) and supports; distractor models (Poly
Haven, rebound to OmniPBR) must clear the height field by CLEAR_M over their whole footprint, stay EDGE_M inside the
desk edge and not cross a sight line; TRIES placement attempts.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from dexengine.core.errors import InputMissingError
from dexengine.record.looks import look_d as LOOK
from dexengine.record.looks.augvis import catalog as AC

MARK = AC.MARK
R_LINK, OBJ_PAD, CLEAR_M, EDGE_M, GRID_M, SEG_M = 0.07, 0.02, 0.03, 0.04, 0.01, 0.03
TRIES = 300
ADD_BATCH = 256  # spheres per Swept.add batch (memory: batch x window cells per array)
SIGHT_PAD_M, PAIR_GAP_M = 0.01, 0.02  # sight-line cylinder pad, distractor pair gap (find_spot)


def joint_pairs(stage, art) -> list[tuple[int, int]]:
    from pxr import Usd, UsdPhysics

    rootp = art.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace("env_.*", "env_0")
    names = list(art.body_names)
    pairs = set()
    for p in Usd.PrimRange(stage.GetPrimAtPath(rootp)):
        if not p.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(p)
        b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
        if b0 and b1 and b0[0].name in names and b1[0].name in names:
            pairs.add((names.index(b0[0].name), names.index(b1[0].name)))
    return sorted(pairs)


def object_radius(stage, path: str) -> float:
    import numpy as np

    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    r = cache.ComputeUntransformedBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    if r.IsEmpty():
        raise SystemExit(f"{MARK} object {path} has an empty extent")
    lo, hi = np.array(r.GetMin()), np.array(r.GetMax())
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return float(np.linalg.norm(corners, axis=1).max()) + OBJ_PAD


def support_samples(discs: list, desk: dict, org) -> list[tuple]:
    """Swept samples of a support (disc + post) [(points (n, 3), radius)]: one at the disc centre · with a post, a post
    every SEG_M from the ground (top surface | floor) to the disc centre. A disc set directly on the ground has pillar
    None (scene.supports form): no post."""
    import numpy as np

    out = []
    for dsc in discs:
        c = np.asarray(dsc["center"], float) + org
        out.append((c[None], dsc["r_m"]))
        if not (pil := dsc["pillar"]):
            continue
        zs = np.arange(desk["top_z"] if pil["ground"] == "desk" else 0.0, c[2], SEG_M)
        if len(zs):
            out.append((np.stack([np.full_like(zs, c[0]), np.full_like(zs, c[1]), zs], 1), pil["radius_m"]))
    return out


class Swept:
    """Height field of the top grid (lowest swept z downward); stamps sphere samples (centre · radius)."""

    def __init__(self, desk: dict):
        import numpy as np

        self.d = desk
        self.x = np.arange(desk["near_x"], desk["far_x"] + 1e-9, GRID_M)
        self.y = np.arange(desk["cy"] - desk["half_y"], desk["cy"] + desk["half_y"] + 1e-9, GRID_M)
        self.h = np.full((len(self.x), len(self.y)), np.inf)
        self.n = 0

    def add(self, C, R) -> None:
        """Stamp spheres (centres C [N, 3], radius R scalar | [N]) in batches of ADD_BATCH. Per sphere the cells,
        the float operations and their order are those of _add_one (the loop this replaced); h takes the minimum
        over spheres in the input order, so h and n are bit-equal to calling _add_one sphere by sphere."""
        import numpy as np

        C, R = np.asarray(C, float).reshape(-1, 3), np.broadcast_to(np.asarray(R, float), (len(C),))
        top = self.d["top_z"] + AC.DISTRACT_MAX_H + CLEAR_M + 0.05
        keep = (C[:, 2] - R) < top
        C, R = C[keep], R[keep]
        for s in range(0, len(C), ADD_BATCH):
            self._add_batch(C[s : s + ADD_BATCH], R[s : s + ADD_BATCH])

    def _add_batch(self, C, R) -> None:
        import numpy as np

        mx = np.abs(self.x[None, :] - C[:, 0:1]) <= R[:, None]  # [S, nx]: the cells _add_one picks per sphere
        my = np.abs(self.y[None, :] - C[:, 1:2]) <= R[:, None]
        cx, cy = mx.sum(1), my.sum(1)
        ok = (cx > 0) & (cy > 0)
        ix0, iy0 = mx.argmax(1), my.argmax(1)  # first picked cell
        nx, ny = len(self.x), len(self.y)
        # the picked cells of a sphere are one run (|x - c| <= r on a sorted grid); a sphere whose cells are not one
        # run (never seen) goes through _add_one, in its place in the input order
        run = (nx - 1 - mx[:, ::-1].argmax(1) - ix0 + 1 == cx) & (ny - 1 - my[:, ::-1].argmax(1) - iy0 + 1 == cy)
        if not np.all(run[ok]):
            for c, r in zip(C, R, strict=True):
                self._add_one(c, r)
            return
        C, R, cx, cy, ix0, iy0 = C[ok], R[ok], cx[ok], cy[ok], ix0[ok], iy0[ok]
        self.n += len(C)
        if not len(C):
            return
        ax, ay = np.arange(cx.max()), np.arange(cy.max())
        IX, IY = np.minimum(ix0[:, None] + ax, nx - 1), np.minimum(iy0[:, None] + ay, ny - 1)
        dx = self.x[IX] - C[:, 0:1]  # [S, KX] = _add_one's dx per sphere
        dy = self.y[IY] - C[:, 1:2]
        dx2, dy2, rr = (dx * dx)[:, :, None], (dy * dy)[:, None, :], (R * R)[:, None, None]
        inside = (dx2 + dy2 <= rr) & (ax[None, :] < cx[:, None])[:, :, None] & (ay[None, :] < cy[:, None])[:, None, :]
        zlo = C[:, 2, None, None] - np.sqrt(np.clip(rr - dx2 - dy2, 0, None))
        # cells outside a sphere get +inf (min(h, inf) = h); ufunc.at applies the entries in C order = sphere by
        # sphere as the loop did, so even a +0.0 / -0.0 tie keeps the value the loop kept
        flat = self.h.reshape(-1)
        if not np.shares_memory(flat, self.h):
            raise RuntimeError(f"{MARK} swept height field is not contiguous: the batch update would be lost")
        np.minimum.at(flat, (IX[:, :, None] * ny + IY[:, None, :]).ravel(), np.where(inside, zlo, np.inf).ravel())

    def _add_one(self, c, r) -> None:
        """One sphere (the loop body add had before batching; the reference of the batch path)."""
        import numpy as np

        ix = np.nonzero(np.abs(self.x - c[0]) <= r)[0]
        iy = np.nonzero(np.abs(self.y - c[1]) <= r)[0]
        if not len(ix) or not len(iy):
            return
        dx = self.x[ix][:, None] - c[0]
        dy = self.y[iy][None, :] - c[1]
        inside = dx * dx + dy * dy <= r * r
        zlo = c[2] - np.sqrt(np.clip(r * r - dx * dx - dy * dy, 0, None))
        sub = self.h[np.ix_(ix, iy)]
        self.h[np.ix_(ix, iy)] = np.where(inside, np.minimum(sub, zlo), sub)
        self.n += 1

    def free(self, x, y, rho, h) -> bool:
        import numpy as np

        ix = np.nonzero(np.abs(self.x - x) <= rho + CLEAR_M)[0]
        iy = np.nonzero(np.abs(self.y - y) <= rho + CLEAR_M)[0]
        dx = self.x[ix][:, None] - x
        dy = self.y[iy][None, :] - y
        ins = dx * dx + dy * dy <= (rho + CLEAR_M) ** 2
        return bool(np.all(self.h[np.ix_(ix, iy)][ins] > self.d["top_z"] + h + CLEAR_M))

    def free_frac(self, h=0.15) -> float:
        import numpy as np

        return round(float(np.mean(self.h > self.d["top_z"] + h + CLEAR_M)), 4)


def seg_hits_cyl(P0, P1, x, y, r, z0, z1) -> bool:
    """Does segment P0->P1 cross a vertical cylinder (centre x, y · radius r · z0..z1)? (1 cm samples)"""
    import numpy as np

    P0, P1 = np.asarray(P0, float), np.asarray(P1, float)
    n = max(2, int(np.linalg.norm(P1 - P0) / 0.01))
    t = np.linspace(0, 1, n)[:, None]
    Q = P0 + t * (P1 - P0)
    return bool(np.any(((Q[:, 0] - x) ** 2 + (Q[:, 1] - y) ** 2 <= r * r) & (Q[:, 2] >= z0) & (Q[:, 2] <= z1)))


def _tex_of(inp, depth: int = 0):
    """Walks back the shader inputs to the first UsdUVTexture: (resolved file path, output name); None without one."""
    from pxr import UsdShade

    if depth > 6:
        return None
    got = inp.GetConnectedSources()
    for info in got[0] if isinstance(got, tuple) else got:
        sh = UsdShade.Shader(info.source.GetPrim())
        if sh and sh.GetIdAttr().Get() == "UsdUVTexture":
            f = sh.GetInput("file").Get()
            return (f.resolvedPath or f.path) if f is not None else None
        for sub in UsdShade.Shader(info.source.GetPrim()).GetInputs():
            r = _tex_of(sub, depth + 1)
            if r:
                return r
    return None


def _preview_surface_maps(mat) -> dict:
    """Texture maps (diff · rough · metal · nrm) and the constant diffuse colour (diff_rgb) of a material's
    UsdPreviewSurface shaders."""
    from pxr import Usd, UsdShade

    maps = {}
    for sh_prim in Usd.PrimRange(mat.GetPrim()):
        sh = UsdShade.Shader(sh_prim)
        if not sh or sh.GetIdAttr().Get() != "UsdPreviewSurface":
            continue
        for nm, k in (
            ("diffuseColor", "diff"),
            ("roughness", "rough"),
            ("metallic", "metal"),
            ("normal", "nrm"),
        ):
            i = sh.GetInput(nm)
            if i and i.HasConnectedSource():
                t = _tex_of(i)
                if t:
                    maps[k] = t
            elif i and i.Get() is not None and k == "diff":
                maps["diff_rgb"] = tuple(float(c) for c in i.Get())
    return maps


def _omnipbr_material(stage, mp, maps: dict):
    """An OmniPBR material at path mp with the given maps (mesh UV), all three outputs connected."""
    from pxr import Gf, Sdf, UsdShade

    m2 = UsdShade.Material.Define(stage, mp)
    sh = UsdShade.Shader.Define(stage, mp.AppendChild("Shader"))
    sh.SetSourceAsset("OmniPBR.mdl", "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")

    def tex(nm, path, raw):
        inp = sh.CreateInput(nm, Sdf.ValueTypeNames.Asset)
        inp.Set(Sdf.AssetPath(str(path)))
        inp.GetAttr().SetMetadata("colorSpace", "raw" if raw else "sRGB")

    if "diff" in maps:
        tex("diffuse_texture", maps["diff"], False)
    sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*maps.get("diff_rgb", (0.5, 0.5, 0.5)))
    )
    if "rough" in maps:
        tex("reflectionroughness_texture", maps["rough"], True)
        sh.CreateInput("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float).Set(1.0)
    if "metal" in maps:
        tex("metallic_texture", maps["metal"], True)
        sh.CreateInput("metallic_texture_influence", Sdf.ValueTypeNames.Float).Set(1.0)
    if "nrm" in maps:
        tex("normalmap_texture", maps["nrm"], True)
    o = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    m2.CreateSurfaceOutput("mdl").ConnectToSource(o)
    m2.CreateDisplacementOutput("mdl").ConnectToSource(o)
    m2.CreateVolumeOutput("mdl").ConnectToSource(o)
    return m2


def rebind_model_materials(stage, root_path: str, tag: str) -> dict:
    """The UsdPreviewSurface (+ MaterialX output · UsdTransform2d) network of Poly Haven USDs renders as the red error
    material on this RTX (measured): for every material in the model an OmniPBR (mesh UV) with the same textures
    (diffuse · roughness · metallic · normal) is made and bound strongest in place of each mesh's original material.
    Returns {original material: [maps found]}."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    made, out = {}, {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mat, _rel = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not mat:
            continue
        key = str(mat.GetPath())
        if key not in made:
            maps = _preview_surface_maps(mat)
            made[key] = _omnipbr_material(stage, Sdf.Path(f"/World/AugMaterials/ph_{tag}_{len(made)}"), maps)
            out[key] = sorted(k for k in maps if k != "diff_rgb")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(made[key], bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    if not made:
        raise InputMissingError(f"{MARK} {root_path}: model materials not found (no mesh bindings)")
    return out


def spawn_distractor(stage, path: str, usd: str, yaw: float) -> tuple:
    """Reference at the origin · yaw · up axis · units -> (translate op, extent lo, hi)."""
    from pxr import Gf, Usd, UsdGeom

    ref = Usd.Stage.Open(usd, Usd.Stage.LoadNone)
    mpu, up = UsdGeom.GetStageMetersPerUnit(ref), UsdGeom.GetStageUpAxis(ref)
    xf = UsdGeom.Xform.Define(stage, path)
    xf.GetPrim().GetReferences().AddReference(usd)
    t_op = xf.AddTranslateOp()
    xf.AddRotateZOp().Set(float(yaw))
    if up == "Y":
        xf.AddRotateXOp().Set(90.0)
    xf.AddScaleOp().Set(Gf.Vec3f(mpu, mpu, mpu))
    t_op.Set(Gf.Vec3d(0, 0, 0))
    bb = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        .ComputeWorldBound(xf.GetPrim())
        .ComputeAlignedRange()
    )
    return t_op, bb.GetMin(), bb.GetMax(), {"mpu": mpu, "up": up}


def find_spot(rng: random.Random, d: dict, sw, rho: float, h: float, placed: list, lines: list):
    """Distractor place search: inside desk d · outside swept volume sw · clear of placed ones (placed[{xy, rho}]) and
    sight lines (lines[(eye, target)]) -> (x, y) | None. Rule owner (called by the Kit placement place_distractors and
    by geom.place outside Isaac)."""
    for _ in range(TRIES):
        x = rng.uniform(d["near_x"] + EDGE_M + rho, d["far_x"] - EDGE_M - rho)
        y = rng.uniform(d["cy"] - d["half_y"] + EDGE_M + rho, d["cy"] + d["half_y"] - EDGE_M - rho)
        if not sw.free(x, y, rho, h):
            continue
        if any(math.hypot(x - p["xy"][0], y - p["xy"][1]) < rho + p["rho"] + PAIR_GAP_M for p in placed):
            continue
        if any(seg_hits_cyl(e, t, x, y, rho + SIGHT_PAD_M, d["top_z"], d["top_z"] + h + SIGHT_PAD_M) for e, t in lines):
            continue
        return x, y
    return None


def place_distractors(stage, spec: dict, ctx: dict, man: dict, root: Path) -> tuple[list, list]:
    from pxr import Gf, Sdf

    base = "/World/AugVis/distractors"
    if stage.GetPrimAtPath(base).IsValid():
        stage.RemovePrim(Sdf.Path(base))
    if not spec["distractors"]:
        return [], []
    from pxr import UsdGeom

    UsdGeom.Xform.Define(stage, base)
    rng = random.Random(spec["place_seed"])
    d, sw = ctx["desk"], ctx["swept"]
    table = AC.distractors(man)
    placed, dropped = [], []
    for j, key in enumerate(spec["distractors"]):
        m = man["models"][table[key]["id"]]
        yaw = rng.uniform(0, 360)
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in key)  # prim name (gso: · no leading digit)
        path = f"{base}/d{j}_{safe}"
        t_op, lo, hi, info = spawn_distractor(stage, path, str(root / m["usd"]), yaw)
        sx, sy, h = hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]
        rho = 0.5 * math.hypot(sx, sy)
        cxb, cyb = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
        why = AC.size_why([x / 1000.0 for x in m["dims_mm"]])  # the same rule and input as receipt (unrotated extent)
        hit = None if why is not None else find_spot(rng, d, sw, rho, h, placed, ctx["sightlines"])
        if hit is None:
            stage.RemovePrim(Sdf.Path(path))
            # data (render meta look.apply.dropped)
            dropped.append({"key": key, "why": why or f"no spot ({TRIES} tries — swept volume · occlusion · overlap)"})
            continue
        t_op.Set(Gf.Vec3d(hit[0] - cxb, hit[1] - cyb, d["top_z"] - lo[2]))
        maps = rebind_model_materials(stage, path, f"d{j}_{safe}")
        placed.append(
            dict(
                key=key,
                id=table[key]["id"],
                xy=[round(hit[0], 4), round(hit[1], 4)],
                maps=maps,
                yaw_deg=round(yaw, 1),
                size_m=[round(sx, 3), round(sy, 3), round(h, 3)],
                rho=round(rho, 3),
                **info,
            )
        )
    removed, left = LOOK._strip_physics(stage.GetPrimAtPath(base))
    if left:
        raise SystemExit(f"{MARK} {left} distractor physics APIs left")
    return placed, dropped
