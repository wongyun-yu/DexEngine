"""Look D: the one canonical look of every robot frame (record stage-2 render, camera reference render, the E5
closed-loop sessions that reuse the record render session, and the pose still of rl.render.pose_check). Training and
judging draw nothing (dexengine.rl.app) and never apply it.

apply(stage) is the one function that dresses a built stage, before the first frame:
  1. room: four single-sided wall quads with UVs around the robot (ROOM: 6 x 6 m, 2.6 m high, centre (-0.7, 0)) and a
     visual-only ceiling; the ego camera pitch is measured and logged;
  2. camera: Iray tonemapper (op 7) with a fixed physical exposure (ISO 1600, TONEMAP exposureTime, f/2.8) and white
     balance = blackbody colour of the panel temperature;
  3. closed room: the env cfg's dome and distant key light (CANON_LIGHTS, when present) at intensity 0, no ambient
     light;
  4. nine ceiling LED panels (RectLight, 40 W/pi, normalised, 1.2 x 0.6 m, 4000 K, 1.5 m grid);
  5. RTX settings QUALITY (reflections, indirect diffuse and AO off);
  6. materials (floor, walls, ceiling, desk-top MDLs), baseboard and visual-only background props (collision and
     rigid-body APIs removed);
  7. visible support discs with pillars to the floor (scene.supports, material SUPPORT_MDL);
and ends by setting the frame-guard token (looks.look_guard.mark): no frame writer runs without it.
Assets (S3 NVIDIA Isaac 4.5 MDLs and props) are read only from the local mirror MIRROR_REL under the main data root;
a missing file is an error (no S3 fallback). Build / check the mirror:
``python -m dexengine.record.looks.look_d --mirror | --verify`` (Isaac python: pxr follows USD dependencies).
Markers: ``[COLLECT_LOOK] D applied`` and ``[COLLECT_LOOK] settle drift`` (log_problems checks render and reference
session logs; a run without them is invalid). render_meta.json ``render_look`` = meta() plus the applied readback;
``look_settle`` = looks.look_settle.settle() (the lighting settle before the first frame).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from dexengine.core.errors import InputMissingError
from dexengine.record.looks.look_assets import (
    LOOK,
    MARK,
    MDL,
    MIRROR_REL,
    PROPS,
    build_mirror,
    local,
    mirror_root,
    verify_mirror,
)
from dexengine.record.looks.look_settle import SETTLE_EYES_ENV, SETTLE_EYES_MARK, SETTLE_MARK
from dexengine.record.looks.materials import blackbody_rgb  # noqa: F401  (read as look_d.blackbody_rgb)

CODE_FILES = (
    "dexengine/record/looks/look_d.py",
    "dexengine/record/looks/look_assets.py",  # stage-2 cache fingerprint
    "dexengine/record/looks/materials.py",
    "dexengine/record/looks/usd_bind.py",  # material bindings and the room quads
    "dexengine/record/looks/scene_env.py",  # build-time scene inputs of the dumps (key light, focal length)
    "dexengine/record/looks/look_settle.py",  # the lighting settle before the first frame
)
#: Exposure 0.1109 s = metered 0.0784 x sqrt(2) (rev D4). Brightness gate values live in rev.LOOK_REV.
TONEMAP = {
    "/rtx/post/tonemap/op": 7,
    "/rtx/post/tonemap/filmIso": 1600.0,
    "/rtx/post/tonemap/exposureTime": 0.1109,
    "/rtx/post/tonemap/fNumber": 2.8,
    "/rtx/post/tonemap/irayReinhard/burnHighlights": 0.0,
    "/rtx/post/tonemap/irayReinhard/crushBlacks": 0.0,
    "/rtx/post/tonemap/irayReinhard/saturation": 0.7,
}
#: Reflections, indirect diffuse and AO off.
QUALITY = {
    "/rtx/reflections/enabled": False,
    "/rtx/indirectDiffuse/enabled": False,
    "/rtx/ambientOcclusion/enabled": False,
    "/rtx/sceneDb/ambientLightIntensity": 0.0,
}
WHITE_K = "/rtx/post/tonemap/whitepoint"
PANEL_W, PANEL_K, PANEL_SIZE = 40.0, 4000.0, (1.2, 0.6)
PANEL_XY = tuple((x, y) for x in (-2.1, -0.6, 0.9) for y in (-1.5, 0.0, 1.5))
BASEBOARD = ((0.16, 0.16, 0.17), 0.55, 0.0)
CANON_LIGHTS = ("/World/light", "/World/keyLight")
#: The env cfg's dome (always built). The distant key light exists only when the scene env sets DEXENGINE_KEY_LIGHT
#: (looks.scene_env, the record render's dump); apply zeroes whichever of CANON_LIGHTS exist and records them.
DOME_LIGHT = "/World/light"
#: Room around the robot (m): half side, height, centre x, centre y. Four single-sided wall quads with UVs (WALL_TILE_M
#: per UV tile) face inwards (dollhouse: a camera outside sees through them); the floor is the env cfg's /World/Floor.
ROOM = {"half": 3.0, "h": 2.6, "cx": -0.7, "cy": 0.0}
ROOM_ROOT, WALL_TILE_M = "/World/AugRoom", 1.5
WALLS = ("wall_nx", "wall_px", "wall_ny", "wall_py")
SUPPORT_MDL = "desk"  # support disc and pillar material (MDL key): wood


def rev():
    """The look revision module (imported lazily: it imports this module)."""
    from dexengine.record.looks import rev as LREV

    return LREV


def log_problems(text: str, settle: bool = True) -> list[str]:
    """Session log -> one problem per missing look marker (render and reference sessions). settle = episodes were
    drawn."""
    from dexengine.record.looks import look_guard as LG

    marks = (f"{MARK} {LOOK} applied", f"{LG.MARK} canonical look {LOOK} applied") + ((SETTLE_MARK,) if settle else ())
    if settle and os.environ.get(SETTLE_EYES_ENV, "") == "2":
        marks += (SETTLE_EYES_MARK,)
    bad = [f"render look marker '{m}' missing" for m in marks if m not in text]
    return bad + rev().gate_problems(text)


def meta() -> dict:
    """Static look values stored in the episode metadata."""
    return {
        "render_preset": LOOK,
        "tonemap": TONEMAP,
        "whitepoint": blackbody_rgb(PANEL_K),
        "rtx": dict(QUALITY),
        "panels": {
            "n": len(PANEL_XY),
            "watts": PANEL_W,
            "intensity": round(PANEL_W / math.pi, 4),
            "normalize": True,
            "kelvin": PANEL_K,
            "size_m": list(PANEL_SIZE),
            "xy": [list(p) for p in PANEL_XY],
        },
        "canon_lights_zeroed": list(CANON_LIGHTS),
        "materials": {k: Path(v).name for k, v in MDL.items()},
        "props": [Path(p[0]).name for p in PROPS],
        "mirror": MIRROR_REL,
        "light_variation": None,
        "support_material": Path(MDL[SUPPORT_MDL]).name,
        "look_rev": rev().LOOK_REV,
    }


# ---- apply (inside Kit) ----
def _set_many(d: dict) -> None:
    import carb

    s = carb.settings.get_settings()
    for k, v in d.items():
        s.set(k, v)

    def same(a, b):
        if isinstance(b, (list, tuple)):
            return (
                isinstance(a, (list, tuple)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
            )
        if isinstance(b, float) and isinstance(a, (int, float)):
            return abs(a - b) < 1e-5 * max(1.0, abs(b))
        return a == b

    bad = {k: (v, s.get(k)) for k, v in d.items() if not same(s.get(k), v)}
    if bad:
        raise SystemExit(f"{MARK} carb settings readback mismatch {bad}")


def room_walls() -> dict:
    """Wall name -> quad corners (counter-clockwise seen from the room centre = the visible side), from ROOM."""
    x0, x1 = ROOM["cx"] - ROOM["half"], ROOM["cx"] + ROOM["half"]
    y0, y1, h = ROOM["cy"] - ROOM["half"], ROOM["cy"] + ROOM["half"], ROOM["h"]
    return {
        "wall_ny": [(x0, y0, 0), (x1, y0, 0), (x1, y0, h), (x0, y0, h)],  # -y wall, faces +y
        "wall_py": [(x1, y1, 0), (x0, y1, 0), (x0, y1, h), (x1, y1, h)],  # +y wall
        "wall_nx": [(x0, y1, 0), (x0, y0, 0), (x0, y0, h), (x0, y1, h)],  # -x wall
        "wall_px": [(x1, y0, 0), (x1, y1, 0), (x1, y1, h), (x1, y0, h)],  # +x wall
    }


def _spawn_room(stage) -> None:
    """The four wall quads under ROOM_ROOT (their materials are bound by apply)."""
    from pxr import UsdGeom

    from dexengine.record.looks import usd_bind as UB

    UsdGeom.Xform.Define(stage, ROOM_ROOT)
    for nm, pts in room_walls().items():
        UB.uv_quad(stage, f"{ROOM_ROOT}/{nm}", pts, tile=WALL_TILE_M)


def _room(stage) -> dict:
    """Room bounds read back from the wall quads apply built (ROOM_ROOT)."""
    from pxr import UsdGeom

    xs, ys, zs = [], [], []
    for nm in WALLS:
        prim = stage.GetPrimAtPath(f"{ROOM_ROOT}/{nm}")
        if not prim.IsValid():
            raise InputMissingError(f"{MARK} room wall {nm} missing after _spawn_room")
        for p in UsdGeom.Mesh(prim).GetPointsAttr().Get():
            xs.append(p[0])
            ys.append(p[1])
            zs.append(p[2])
    return {"x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys), "h": max(zs)}


def _place_prop(stage, path: str, src: str, room: dict, wall: str, along, yaw: float) -> dict:
    from pxr import Gf, Usd, UsdGeom

    ref = Usd.Stage.Open(src, Usd.Stage.LoadNone)
    mpu, up = UsdGeom.GetStageMetersPerUnit(ref), UsdGeom.GetStageUpAxis(ref)
    xf = UsdGeom.Xform.Define(stage, path)
    prim = xf.GetPrim()
    prim.GetReferences().AddReference(src)
    t_op = xf.AddTranslateOp()
    xf.AddRotateZOp().Set(float(yaw))
    if up == "Y":
        xf.AddRotateXOp().Set(90.0)
    xf.AddScaleOp().Set(Gf.Vec3f(mpu, mpu, mpu))
    t_op.Set(Gf.Vec3d(0, 0, 0))
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(prim).ComputeAlignedRange()
    lo, hi = bb.GetMin(), bb.GetMax()
    if not (hi[2] - lo[2]) > 0.05:
        raise SystemExit(f"{MARK} prop {src} has empty bounds ({lo}–{hi}) — missing mirror dependency?")
    gap = 0.02
    if wall == "nx":
        x, y = room["x0"] + gap - lo[0], along - (lo[1] + hi[1]) / 2
    elif wall == "px":
        x, y = room["x1"] - gap - hi[0], along - (lo[1] + hi[1]) / 2
    elif wall == "ny":
        x, y = along - (lo[0] + hi[0]) / 2, room["y0"] + gap - lo[1]
    elif wall == "py":
        x, y = along - (lo[0] + hi[0]) / 2, room["y1"] - gap - hi[1]
    else:
        x, y = along[0] - (lo[0] + hi[0]) / 2, along[1] - (lo[1] + hi[1]) / 2
    t_op.Set(Gf.Vec3d(x, y, -lo[2]))
    return {"asset": Path(src).name, "size_m": [round(hi[i] - lo[i], 3) for i in range(3)]}


def _strip_physics(root_prim) -> tuple[int, int]:
    from pxr import Usd, UsdPhysics

    apis = (UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI, UsdPhysics.RigidBodyAPI)
    removed = 0
    for sub in Usd.PrimRange(root_prim):
        for api in apis:
            if sub.HasAPI(api):
                sub.RemoveAPI(api)
                removed += 1
    left = sum(1 for sub in Usd.PrimRange(root_prim) for api in apis if sub.HasAPI(api))
    return removed, left


def _ceiling_panels(stage, h):
    """Ceiling panel rect lights (PANEL_XY) just under the room height."""
    from pxr import Gf, UsdGeom, UsdLux

    for i, (x, y) in enumerate(PANEL_XY):
        lt = UsdLux.RectLight.Define(stage, f"/World/CollectLook/panel_{i}")
        lt.CreateWidthAttr(PANEL_SIZE[0])
        lt.CreateHeightAttr(PANEL_SIZE[1])
        lt.CreateIntensityAttr(PANEL_W / math.pi)
        lt.CreateNormalizeAttr(True)
        lt.CreateEnableColorTemperatureAttr(True)
        lt.CreateColorTemperatureAttr(PANEL_K)
        UsdGeom.Xformable(lt).AddTranslateOp().Set(Gf.Vec3d(x, y, h - 0.005))  # a RectLight emits along local -Z


def _baseboards(stage, x0, y0, y1, x1):
    """Baseboards along the four room walls, solid material."""
    from pxr import Gf, UsdGeom

    from dexengine.record.looks import usd_bind as UB

    bh, bt = 0.10, 0.015
    UsdGeom.Xform.Define(stage, "/World/CollectLook/baseboard")
    boards = {
        "nx": ((x0 + bt / 2, (y0 + y1) / 2), (bt, y1 - y0)),
        "px": ((x1 - bt / 2, (y0 + y1) / 2), (bt, y1 - y0)),
        "ny": (((x0 + x1) / 2, y0 + bt / 2), (x1 - x0, bt)),
        "py": (((x0 + x1) / 2, y1 - bt / 2), (x1 - x0, bt)),
    }
    for nm, ((cx, cy), (sx, sy)) in boards.items():
        c = UsdGeom.Cube.Define(stage, f"/World/CollectLook/baseboard/{nm}")
        c.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(c)
        xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, bh / 2))
        xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, bh))
    UB.bind_solid(stage, "/World/CollectLook/baseboard", "look_baseboard", *BASEBOARD)


def _log_ego_pitch(stage) -> None:
    """Print the measured optical axis of the stage's HeadCamera (read-only).

    The ego camera's downward pitch is set by a code constant and logged nowhere else. The camera's local -Z axis is
    printed in its parent frame and in world, with the derived downward angles; two runs that differ only in the pitch
    differ by exactly the applied pitch, with no convention to infer.
    """
    from pxr import Gf, Usd, UsdGeom

    cam = next((p for p in Usd.PrimRange(stage.GetPseudoRoot()) if p.GetName() == "HeadCamera"), None)
    if cam is None:
        print(f"{MARK} ego pitch: no HeadCamera prim — not measured", flush=True)
        return
    xc = UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    xp = UsdGeom.Xformable(cam.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    f_loc = (xc * xp.GetInverse()).TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)).GetNormalized()
    f_wld = xc.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)).GetNormalized()

    def down(z):
        return -math.degrees(math.asin(max(-1.0, min(1.0, z))))

    print(
        f"{MARK} ego pitch (code default 30°, head joint) | optical axis local=({f_loc[0]:+.4f},{f_loc[1]:+.4f},"
        f"{f_loc[2]:+.4f}) local down={down(f_loc[2]):+.2f}° | world down={down(f_wld[2]):+.2f}° @ {cam.GetPath()}",
        flush=True,
    )


def _zero_lights(stage) -> dict:
    """Intensity 0 on the env cfg's lights (CANON_LIGHTS that exist; the dome must). -> {path: previous intensity}."""
    if not stage.GetPrimAtPath(DOME_LIGHT).IsValid():
        raise InputMissingError(f"{MARK} canonical light {DOME_LIGHT} missing")
    prev = {}
    for path in CANON_LIGHTS:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            print(f"{MARK} {path} absent (no DEXENGINE_KEY_LIGHT in this scene) — nothing to zero", flush=True)
            continue
        a = prim.GetAttribute("inputs:intensity")
        prev[path] = float(a.Get())
        a.Set(0.0)
    return prev


def apply(stage) -> dict:
    """The one function that applies the canonical look to a built stage (before the first frame): room, lights,
    camera exposure, materials, props, then the frame-guard token. Returns render_meta's render_look."""
    from pxr import UsdGeom

    from dexengine.record.looks import look_guard as LG
    from dexengine.record.looks import materials as DM
    from dexengine.record.looks import usd_bind as UB

    n_mirror = verify_mirror(check_sha=False)
    try:
        _log_ego_pitch(stage)
    except Exception as e:  # noqa: BLE001 — a failed measurement (a log line) must not stop the look
        print(f"{MARK} ego pitch measurement failed: {e}", flush=True)
    _spawn_room(stage)
    room = _room(stage)
    prev = _zero_lights(stage)
    _set_many({**TONEMAP, **QUALITY, WHITE_K: blackbody_rgb(PANEL_K)})
    x0, x1, y0, y1, h = room["x0"], room["x1"], room["y0"], room["y1"], room["h"]
    UsdGeom.Xform.Define(stage, "/World/CollectLook")
    UB.uv_quad(stage, "/World/CollectLook/ceiling", [(x0, y0, h), (x0, y1, h), (x1, y1, h), (x1, y0, h)], tile=1.2)
    binds = [
        UB.bind_mdl(stage, "/World/CollectLook/ceiling", "look_ceiling", local(MDL["ceiling"]), project=True),
        UB.bind_mdl(stage, "/World/Floor", "look_floor", local(MDL["floor"]), project=True),
    ]
    binds.extend(
        UB.bind_mdl(stage, f"{ROOM_ROOT}/{nm}", f"look_{nm}", local(MDL["wall"]), project=True) for nm in WALLS
    )
    tiles = [DM.bind_desk(stage, w, f"look_desk_{i}", local(MDL["desk"])) for i, w in enumerate(DM.desk_woods(stage))]
    n_desk = len(tiles)  # desk top: world-metre projection (materials.bind_desk)
    if not all(binds) or n_desk == 0:
        raise SystemExit(f"{MARK} material binding failed (ceiling, floor, walls {binds} · desk tops {n_desk})")
    _ceiling_panels(stage, h)
    _baseboards(stage, x0, y0, y1, x1)
    UsdGeom.Xform.Define(stage, "/World/CollectLook/props")
    placed = [
        _place_prop(stage, f"/World/CollectLook/props/p{i}", local(src), room, wall, along, yaw)
        for i, (src, wall, along, yaw) in enumerate(PROPS)
    ]
    rev().metal0_apply(stage, PROPS)
    removed, left = _strip_physics(stage.GetPrimAtPath("/World/CollectLook/props"))
    if left:
        raise SystemExit(f"{MARK} {left} physics APIs left on props")
    import carb

    s = carb.settings.get_settings()
    rb = {k: s.get(k) for k in (*TONEMAP, *QUALITY, WHITE_K)}
    rb = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in rb.items()}
    print(
        f"{MARK} {LOOK} applied · exposure ISO {rb['/rtx/post/tonemap/filmIso']:.0f} · "
        f"{rb['/rtx/post/tonemap/exposureTime']} s · "
        f"f/{rb['/rtx/post/tonemap/fNumber']} · op {rb['/rtx/post/tonemap/op']} · WB {rb[WHITE_K]} · "
        f"panels {len(PANEL_XY)} × {PANEL_W:.0f} W {PANEL_K:.0f} K · dome/key {prev} -> 0 · GI/reflections/AO off · "
        f"materials 4 · desks {n_desk} · props {len(placed)} · physics APIs removed {removed}, left 0 · "
        f"mirror {n_mirror} files",
        flush=True,
    )
    print(LG.mark(stage, LOOK), flush=True)  # the only setter of the frame-guard token (look_guard)
    return dict(
        meta(),
        applied={
            "rtx_readback": rb,
            "canon_lights_prev": prev,
            "room": room,
            "props": placed,
            "physics_api_removed": removed,
            "mirror_files": n_mirror,
            "desk_tile_m": tiles,
        },
    )


def main() -> None:
    import argparse

    from dexengine.core import config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_data_root_arg(ap)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--mirror", action="store_true", help="S3 -> local mirror (following dependencies), SHA256SUMS, manifest.json"
    )
    g.add_argument("--verify", action="store_true", help="check mirror files and sha256")
    g.add_argument("--meta", action="store_true", help="print the look values (JSON)")
    a = ap.parse_args()
    config.apply_data_root(a)
    if a.mirror:
        build_mirror()
    if a.mirror or a.verify:
        print(f"{MARK} mirror check passed · files {verify_mirror()} · {mirror_root()}", flush=True)
    if a.meta:
        print(json.dumps(meta(), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
