"""Look D: the render look of the recording path (stage-2 render and camera reference render) on top of the canonical
background. RL training, judging and result videos do not use it.

Look D = canonical background (render_look V5_08_bright, rl.render.scene_look: room walls, floor, desk) plus
  1. camera: Iray tonemapper (op 7) with a fixed physical exposure (ISO 1600, TONEMAP exposureTime, f/2.8) and white
     balance = blackbody colour of the panel temperature;
  2. closed room: visual-only ceiling, canonical dome and distant key light at intensity 0, no ambient light;
  3. nine ceiling LED panels (RectLight, 40 W/pi, normalised, 1.2 x 0.6 m, 4000 K, 1.5 m grid);
  4. RTX settings QUALITY (reflections, indirect diffuse and AO off);
  5. materials (floor, walls, desk-top MDLs), baseboard and visual-only background props (collision and rigid-body
     APIs removed);
  6. visible support discs with pillars to the floor (scene.supports, material SUPPORT_MDL).
Assets (S3 NVIDIA Isaac 4.5 MDLs and props) are read only from the local mirror MIRROR_REL under the main data root;
a missing file is an error (no S3 fallback). The mirror also carries the canonical background's base-layer MDLs
(materials.base_roots). Build / check the mirror: ``python -m dexengine.record.looks.look_d --mirror | --verify``
(Isaac python: pxr follows USD dependencies).
Markers: ``[COLLECT_LOOK] D applied`` and ``[COLLECT_LOOK] settle drift`` (log_problems checks render and reference
session logs; a run without them is invalid). render_meta.json ``render_look`` = meta() plus the applied readback;
``look_settle`` = settle().
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
from dexengine.record.looks.materials import blackbody_rgb  # noqa: F401  (read as look_d.blackbody_rgb)

CODE_FILES = (
    "dexengine/record/looks/look_d.py",
    "dexengine/record/looks/look_assets.py",  # stage-2 cache fingerprint
    "dexengine/record/looks/materials.py",
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
SUPPORT_MDL = "desk"  # support disc and pillar material (MDL key): wood
SETTLE_TOL, SETTLE_MAX = 0.6, 60
# Settle stage 1 (loading): mean pixel |delta| (8 bit) between consecutive captures (4 renders each) must drop below
# SETTLE_TOL within SETTLE_MAX rounds, otherwise the run dies.
# Plateau (two-eye settle only): converged also when, from round FLAT_N on, the last FLAT_K deltas have a relative
# half-window difference < FLAT_EPS, a linear trend (change over the window / mean) of magnitude < FLAT_EPS, and a
# mean < FLAT_CAP (a large desk in both eyes keeps the noise floor above SETTLE_TOL after loading has finished).
SETTLE_FLAT_N, SETTLE_FLAT_K, SETTLE_FLAT_EPS = 56, 16, 0.05
SETTLE_FLAT_CAP = 2 * SETTLE_TOL
# Settle stage 2 (light-transport drift): channel-mean difference of two captures DRIFT_GAP renders apart, after
# DRIFT_EXTRA renders, until < DRIFT_TOL (8 bit), at most DRIFT_MAX times; hitting the cap is recorded in
# look_settle.drift_cap. DRIFT_MAX 0 = stage 2 off (with reflections, GI and AO off nothing drifts).
DRIFT_EXTRA, DRIFT_GAP, DRIFT_TOL, DRIFT_MAX = 32, 32, 0.12, 0
SETTLE_EYES_ENV = "DEXENGINE_COLLECT_SETTLE_EYES"  # "2" = settle on both eyes (egoL, egoR) when exo is off
SETTLE_EYES_MARK = f"{MARK} settle two eyes"  # applied marker; log_problems requires it when the env is "2"


def settle_cams() -> tuple:
    """Cameras the settle measures: default (empty) = left eye + exo (exo on) or left eye only (exo off); "2" = both
    eyes with exo off. "2" with exo on, or any other value, is an error."""
    from dexengine.record.render import cameras as CAMS

    v = os.environ.get(SETTLE_EYES_ENV, "")
    if v not in ("", "2"):
        raise ValueError(f"{SETTLE_EYES_ENV}={v!r} — empty or 2 (two eyes with exo off) only")
    exo = CAMS.exo_on()
    if v == "2":
        if exo:
            raise ValueError(f"{SETTLE_EYES_ENV}=2 requires exo off; with exo on the settle uses left eye + exo")
        return ("egoL", "egoR")
    return ("egoL", "exo") if exo else ("egoL",)  # exo off: left eye only (exo is not rendered)


SETTLE_CAMS = settle_cams()
SETTLE_MARK = f"{MARK} settle drift"  # stage 2 ran (log_problems requires it)


def rev():
    """The look revision module (imported lazily: it imports this module)."""
    from dexengine.record.looks import rev as LREV

    return LREV


def log_problems(text: str, settle: bool = True) -> list[str]:
    """Session log -> one problem per missing look marker (render and reference sessions). settle = episodes were
    drawn."""
    marks = (f"{MARK} {LOOK} applied",) + ((SETTLE_MARK,) if settle else ())
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


def _room(stage) -> dict:
    """Room bounds read from the canonical room's wall quads (/World/AugRoom, rl.render.scene_look)."""
    from pxr import UsdGeom

    xs, ys, zs = [], [], []
    for nm in ("wall_nx", "wall_px", "wall_ny", "wall_py"):
        prim = stage.GetPrimAtPath(f"/World/AugRoom/{nm}")
        if not prim.IsValid():
            raise InputMissingError(
                f"{MARK} canonical room wall {nm} missing — look D needs the canonical background (DEXENGINE_AUG_ROOM)"
            )
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

    from dexengine.rl.render import scene_look as VA

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
    VA._bind_solid(stage, "/World/CollectLook/baseboard", "look_baseboard", *BASEBOARD)


def apply(stage) -> dict:
    """Apply look D (after the canonical background, before the first frame). Returns render_meta's render_look."""
    from pxr import UsdGeom

    from dexengine.record.looks import materials as DM
    from dexengine.rl.render import scene_look as VA

    n_mirror = verify_mirror(check_sha=False, with_base=False)
    room = _room(stage)
    prev = {}
    for path in CANON_LIGHTS:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise InputMissingError(f"{MARK} canonical light {path} missing")
        a = prim.GetAttribute("inputs:intensity")
        prev[path] = float(a.Get())
        a.Set(0.0)
    _set_many({**TONEMAP, **QUALITY, WHITE_K: blackbody_rgb(PANEL_K)})
    x0, x1, y0, y1, h = room["x0"], room["x1"], room["y0"], room["y1"], room["h"]
    UsdGeom.Xform.Define(stage, "/World/CollectLook")
    VA._uv_quad(
        stage, "/World/CollectLook/ceiling", [(x0, y0, h), (x0, y1, h), (x1, y1, h), (x1, y0, h)], "z", tile=1.2
    )
    binds = [
        VA._bind_mdl(stage, "/World/CollectLook/ceiling", "look_ceiling", local(MDL["ceiling"]), project=True),
        VA._bind_mdl(stage, "/World/Floor", "look_floor", local(MDL["floor"]), project=True),
    ]
    binds.extend(
        VA._bind_mdl(stage, f"/World/AugRoom/{nm}", f"look_{nm}", local(MDL["wall"]), project=True)
        for nm in ("wall_nx", "wall_px", "wall_ny", "wall_py")
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


def settle_flat(trace: list) -> dict | None:
    """Plateau test (SETTLE_FLAT_* comment): {floor, rel, trend} when flat, else None."""
    import numpy as np

    K = SETTLE_FLAT_K
    if len(trace) < K:
        return None
    w = np.asarray(trace[-K:], float)
    m = float(w.mean())
    rel = abs(float(w[K // 2 :].mean()) - float(w[: K // 2].mean())) / m
    trend = float(np.polyfit(np.arange(K), w, 1)[0]) * (K - 1) / m
    if rel < SETTLE_FLAT_EPS and abs(trend) < SETTLE_FLAT_EPS and m < SETTLE_FLAT_CAP:
        return {"floor": round(m, 3), "rel": round(rel, 4), "trend": round(trend, 4)}
    return None


def settle_load(rig, info: dict | None = None) -> int:
    """Settle stage 1 (MDL compile, texture load): until the mean pixel |delta| of consecutive captures (4 renders) is
    below SETTLE_TOL; returns the number of rounds. Two-eye settles also converge on a plateau (settle_flat, from
    round SETTLE_FLAT_N; recorded in info). Dies when it never converges."""
    import numpy as np

    prev, trace = None, []
    flat_on = SETTLE_CAMS == ("egoL", "egoR")
    for r in range(SETTLE_MAX):
        cap = rig.capture(4)
        cur = np.concatenate([cap[c].reshape(-1) for c in SETTLE_CAMS]).astype(np.int16)
        if prev is not None:
            d = float(np.abs(cur - prev).mean())
            trace.append(round(d, 3))
            if d < SETTLE_TOL:
                print(f"{MARK} settle {r + 1} rounds · |Δ| {d:.3f} < {SETTLE_TOL}", flush=True)
                if SETTLE_CAMS == ("egoL", "egoR"):
                    print(f"{SETTLE_EYES_MARK} {SETTLE_CAMS} · {SETTLE_EYES_ENV}=2", flush=True)
                return r + 1
            f = settle_flat(trace) if flat_on and r + 1 >= SETTLE_FLAT_N else None
            if f:
                print(
                    f"{MARK} settle {r + 1} rounds · plateau floor {f['floor']} · rel {f['rel']} · "
                    f"trend {f['trend']} "
                    f"(above {SETTLE_TOL}; flat = loading done)",
                    flush=True,
                )
                print(f"{SETTLE_EYES_MARK} {SETTLE_CAMS} · {SETTLE_EYES_ENV}=2", flush=True)
                if info is not None:
                    info.update(plateau=True, rounds=r + 1, **f)
                return r + 1
        prev = cur
    print(f"{MARK} settle not converged, |Δ| trace (cameras {SETTLE_CAMS} · tol {SETTLE_TOL}): {trace}", flush=True)
    raise SystemExit(f"{MARK} settle not converged in {SETTLE_MAX} rounds — check material and texture loading")


def chan_means(cap) -> dict:
    """Capture -> per-camera channel means (8-bit floats), the unit of the stage-2 drift measurement."""
    import numpy as np

    return {c: cap[c].reshape(-1, 3).astype(np.float64).mean(0) for c in SETTLE_CAMS}


def settle_drift(rig) -> dict:
    """Settle stage 2: light-transport drift (DRIFT_* comment). drift_cap = True at the cap (not an error)."""
    import numpy as np

    hist = []
    for _ in range(DRIFT_MAX):
        rig.render(DRIFT_EXTRA)
        a = chan_means(rig.capture(4))
        rig.render(DRIFT_GAP)
        b = chan_means(rig.capture(4))
        d = max(float(np.abs(b[c] - a[c]).max()) for c in SETTLE_CAMS)
        hist.append(round(d, 3))
        if d < DRIFT_TOL:
            break
    return {"drift": hist, "drift_tol": DRIFT_TOL, "drift_cap": bool(hist) and hist[-1] >= DRIFT_TOL}


def settle_row(look_settle) -> dict:
    """look_settle -> the episode line's render_settle field (an int = stage 1 only -> drift None)."""
    if isinstance(look_settle, int):
        return {"load_rounds": look_settle, "drift": None, "drift_cap": None}
    return {k: look_settle[k] for k in ("load_rounds", "drift", "drift_cap")}


def settle(rig) -> dict:
    """Convergence before the first frame = stage 1 (settle_load) + stage 2 (settle_drift); stored as render_meta
    look_settle. Marker SETTLE_MARK; '[COLLECT_LOOK] settle cap' when stage 2 hits its cap."""
    import time

    t0 = time.time()
    pl: dict = {}
    n = settle_load(rig, pl)
    dr = settle_drift(rig)
    renders = n * 5 + len(dr["drift"]) * (DRIFT_EXTRA + DRIFT_GAP + 12)
    out = dict(load_rounds=n, **dr, drift_rounds=len(dr["drift"]), renders=renders, s=round(time.time() - t0, 1))
    if pl:  # only plateau-converged settles get this field
        out["load_plateau"] = pl
    tail = (
        f"cap {DRIFT_MAX} rounds — recorded (not a failure, look_settle.drift_cap)"
        if dr["drift_cap"]
        else f"converged in {len(dr['drift'])} rounds"
        if DRIFT_MAX
        else "stage 2 off (cap 0)"
    )
    print(f"{SETTLE_MARK} {dr['drift']} (tol {DRIFT_TOL}) · {tail} · renders {renders} · {out['s']} s", flush=True)
    if dr["drift_cap"]:
        print(f"{MARK} settle cap · last difference {dr['drift'][-1]} >= {DRIFT_TOL}", flush=True)
    return out


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
