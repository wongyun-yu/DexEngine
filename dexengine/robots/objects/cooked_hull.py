"""Dump the PhysX-cooked convex hulls of object USDs (bake stage). Source of the judged object shape
(data.objects.hull.collision_parts).

PhysX rebuilds a USD collision prim (``physics:approximation = convexHull``) under its vertex limit when cooking, so the
shape the simulator collides with is the cooked result, not the authored points. The hulls (vertices, faces) cooked
through the omni.physx cooking API (``request_convex_collision_representation``, synchronous) are moved to link
coordinates and written to a file next to the USD (path data_layout.cooked_hull_file, format hull.write_cooked) with
the source USD sha256 and a cooking fingerprint (engine id + per-prim cooking settings). The judge rejects stale dumps
(hull.cooked_problem).

Targets = every object USD of the dataset (usd_places.object_usds: rigid files in the usd layer, articulated files in
the usd_artic layer); the layers decide, not a name list.
  write:   $ISAAC_PY dexengine/robots/objects/cooked_hull.py (--all | --dataset a[,b]) [--ids x,y | --ids_file F]
           [--force] --headless       current dumps are skipped (--force re-cooks); the same targets are verified
           at the end.
  verify:  python3 dexengine/robots/objects/cooked_hull.py (--all | --dataset a[,b]) --verify-only      (no Isaac)
           dump count = USD count, empty set differences (USD without dump, dump without USD), all current (USD
           sha256, format, engine, fingerprint).
Markers: per object ``[COOKED] <ds>/<id> art=… links=… prims=… verts_max=… polys_max=… cooked_out_auth_mm=…``;
  ``[COOKED_DONE] requested=N written=W fresh=F failed=X``;
  ``[COOKED_VERIFY] usd=N dumps=M fresh=K missing=… stale=… orphan=…``.
  rc 0 = no failures and verification passed. rc 2 before writing when the running engine (omni.physx) differs from
  hull.COOK_ENGINE.
Coordinate oracle (fatal): per link, the authored points of this tool must equal object_pin._collision_meshes_body (that
  link), i.e. one prim selection and one coordinate rule. Cooked vertices more than COOK_OUT_MAX_MM outside the
  authored hull mean a wrong coordinate frame (measured maximum 0.96 mm; real defects are centimetres).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from dexengine.core import artifact_paths
from dexengine.core import paths as P
from dexengine.core.errors import InputMissingError
from dexengine.data.objects import hull as OH
from dexengine.data.objects import usd_places as UP

COOK_OUT_MAX_MM = 2.0
TP = artifact_paths


def _cli(isaac: bool) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="dump PhysX-cooked convex hulls of object USDs (bake stage)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="all training datasets (the canon artifact_paths accepts)")
    g.add_argument("--dataset", help="datasets (comma-separated)")
    ap.add_argument("--ids", help="object asset ids (comma-separated; only with a single --dataset)")
    ap.add_argument("--ids_file", help="one asset id per line (only with a single --dataset)")
    ap.add_argument("--force", action="store_true", help="re-cook current dumps too")
    ap.add_argument("--verify-only", action="store_true", help="verify only, without Isaac")
    if isaac:
        from dexengine.rl.isaac_cli import add_app_launcher_args  # noqa: PLC0415 — only in an isaaclab interpreter

        add_app_launcher_args(ap)
    return ap


def _dump(f: Path) -> Path:
    return Path(P.data_layout.cooked_hull_file(str(f)))


def targets(a) -> list:
    """[(ds, asset id, articulated, USD)]: the dataset's rigid and articulated USDs (usd_places.object_usds). A
    requested id without a USD is fatal."""
    dss = list(TP._valid_datasets()) if a.all else [d for d in a.dataset.split(",") if d]
    want = [x for x in (a.ids or "").split(",") if x]
    if a.ids_file:
        want += [ln.strip() for ln in Path(a.ids_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if want and len(dss) != 1:
        raise SystemExit(
            "[COOKED] --ids / --ids_file need a single --dataset (asset ids are only unique within a dataset)"
        )
    out = []
    for ds in dss:
        t = TP.get(ds)
        root = Path(t.usd_root())
        if not root.is_dir():
            raise InputMissingError(f"[COOKED] {ds}: usd layer {root} is missing")
        out += [(ds, oid, art, f) for oid, art, f in UP.object_usds(ds)]
    if want:
        miss = sorted(set(want) - {oid for _, oid, _, _ in out})
        if miss:
            raise InputMissingError(f"[COOKED] {len(miss)} requested ids have no USD: {miss[:10]}")
        out = [r for r in out if r[1] in set(want)]
    return out


def verify(a, rows: list) -> bool:
    """Dump count = USD count, empty set differences, all current; reasons from hull.cooked_problem (judge's rule)."""
    miss, stale = [], []
    for ds, oid, art, f in rows:
        why = OH.cooked_problem(f)
        if why:
            (stale if _dump(f).is_file() else miss).append(f"{ds}/{oid}:{int(art)} {why}")
    want = {_dump(f) for *_, f in rows}
    found = set()
    if not (a.ids or a.ids_file):  # count 'dumps without USD' only when the whole layer was requested
        for ds in {r[0] for r in rows}:
            for art, root in UP.usd_roots(ds):
                found |= set(root.glob(f"*/{_dump(OH.usd_file(ds, 'x', art)).name}"))
    orphan = sorted(found - want)
    print(
        f"[COOKED_VERIFY] usd={len(rows)} dumps={len(want) - len(miss) + len(orphan)} "
        f"fresh={len(rows) - len(miss) - len(stale)} missing={len(miss)} stale={len(stale)} orphan={len(orphan)}",
        flush=True,
    )
    for x in (miss + stale)[:20] + [f"orphan {g}" for g in orphan[:20]]:
        print(f"[COOKED_VERIFY]   ! {x}", flush=True)
    return bool(rows) and not (miss or stale or orphan)


def main_verify() -> int:
    a = _cli(False).parse_args()
    return 0 if verify(a, targets(a)) else 1


# write (Isaac)
def _cook(stage_id: int, prim):
    """Cooked hull of one prim -> (vertices (k,3) in prim coordinates, faces (F,3), polygon count)."""
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    from pxr import PhysicsSchemaTools

    got: dict = {}

    def on_result(result, convexes):
        got["result"], got["convexes"] = result, convexes

    get_physx_cooking_interface().request_convex_collision_representation(
        stage_id=stage_id,
        collision_prim_id=PhysicsSchemaTools.sdfPathToInt(prim.GetPath()),
        run_asynchronously=False,
        on_result=on_result,
    )
    convs = got.get("convexes") or []
    if got.get("result") != PhysxCollisionRepresentationResult.RESULT_VALID or len(convs) != 1:
        raise RuntimeError(f"{prim.GetPath()}: cooking result {got.get('result')}, {len(convs)} convexes (expected 1)")
    c = convs[0]
    idx = list(c.indices)
    tri = [
        (idx[pg.index_base], idx[pg.index_base + k], idx[pg.index_base + k + 1])
        for pg in c.polygons
        for k in range(1, pg.num_vertices - 1)
    ]  # polygon -> triangle fan
    v = np.array([[p.x, p.y, p.z] for p in c.vertices], float)
    return v, np.asarray(tri, np.int32).reshape(-1, 3), len(c.polygons)


def _settings(prim) -> list:
    """Cooking settings (fingerprint input): approximation, PhysX convex hull vertex limit, min thickness (None when
    not authored -> engine default)."""
    from pxr import PhysxSchema, UsdPhysics

    mc = UsdPhysics.MeshCollisionAPI
    ap = mc(prim).GetApproximationAttr().Get() if prim.HasAPI(mc) else None
    hv = mt = None
    if prim.HasAPI(PhysxSchema.PhysxConvexHullCollisionAPI):
        api = PhysxSchema.PhysxConvexHullCollisionAPI(prim)
        hv, mt = api.GetHullVertexLimitAttr().Get(), api.GetMinThicknessAttr().Get()
    return [str(prim.GetPath()), None if ap is None else str(ap), hv, None if mt is None else float(mt)]


def _out_mm(a: np.ndarray, b: np.ndarray) -> float:
    """Largest plane distance (mm) of points a outside the convex hull of b (QJ joggle builds hulls of flat pieces)."""
    from scipy.spatial import ConvexHull

    eq = ConvexHull(b, qhull_options="QJ").equations
    return max(0.0, float((a @ eq[:, :3].T + eq[:, 3]).max(axis=1).max()) * 1e3)


def dump_one(ctx, op, art: bool, usd: Path) -> str:
    """One USD -> dump. Links = hull.cookable_links (the same rule the judge uses to classify dump problems); the
    authored points must equal object_pin._collision_meshes_body."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    if not ctx.open_stage(str(usd)):
        raise RuntimeError(f"open_stage failed: {usd}")
    st, sid, CA = ctx.get_stage(), ctx.get_stage_id(), UsdPhysics.CollisionAPI
    try:
        links = OH.cookable_links(st, art)
        xc = UsdGeom.XformCache(Usd.TimeCode.Default())
        out, settings, vmax, pmax, c_out = {}, [], 0, 0, 0.0
        for ln, lp in links:
            tf = Gf.Transform(xc.GetLocalToWorldTransform(st.GetPrimAtPath(lp)))
            inv = np.asarray(Gf.Matrix4d().SetTransform(tf.GetRotation(), tf.GetTranslation()).GetInverse(), float)
            cooked, faces, auth = [], [], []
            for p in Usd.PrimRange(st.GetPrimAtPath(lp), Usd.TraverseInstanceProxies()):
                pts = np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get() or [], float) if p.IsA(UsdGeom.Mesh) else []
                if not (len(pts) and p.HasAPI(CA) and CA(p).GetCollisionEnabledAttr().Get() is not False):
                    continue
                m = np.asarray(xc.GetLocalToWorldTransform(p), float) @ inv  # [p,1] @ M_mesh→world @ inv(M_link)
                v, tri, npoly = _cook(sid, p)
                auth.append((np.c_[pts, np.ones(len(pts))] @ m)[:, :3])
                cooked.append((np.c_[v, np.ones(len(v))] @ m)[:, :3])
                faces.append(tri)
                settings.append(_settings(p))
                c_out = max(c_out, _out_mm(cooked[-1], auth[-1]))
                vmax, pmax = max(vmax, len(v)), max(pmax, npoly)
            ref = op._collision_meshes_body(st, lp)[0]
            if len(ref) != len(auth) or any(
                r.shape != x.shape or np.abs(r - x).max() > 1e-9 for r, x in zip(ref, auth, strict=True)
            ):
                raise RuntimeError(
                    f"{ln}: authored points ≠ object_pin._collision_meshes_body — two prim-selection or "
                    "coordinate rules"
                )
            out[ln] = (cooked, faces, np.concatenate(op._collision_meshes_body(st, lp, visual=True)[0]))
        if c_out > COOK_OUT_MAX_MM:
            raise RuntimeError(
                f"cooked vertices outside the authored hull by {c_out:.3f} mm > {COOK_OUT_MAX_MM} — "
                "wrong coordinate frame"
            )
        OH.write_cooked(usd, out, OH.COOK_ENGINE, settings)
    finally:
        ctx.close_stage()
    return (
        f"art={int(art)} links={len(links)} prims={len(settings)} verts_max={vmax} polys_max={pmax} "
        f"cooked_out_auth_mm={c_out:.4f}"
    )


_APP = None


def main_write() -> int:
    global _APP
    a = _cli(True).parse_args()
    from isaaclab.app import AppLauncher

    _APP = AppLauncher(a).app
    from dexengine.rl.isaac_cli import exit_nonzero_on_uncaught  # noqa: PLC0415

    exit_nonzero_on_uncaught()
    import omni.kit.app
    import omni.usd

    eng = omni.kit.app.get_app().get_extension_manager().get_enabled_extension_id("omni.physx")
    print(f"[COOKED_RUN] tree={P.REPO_ROOT} engine={eng} pin={OH.COOK_ENGINE}", flush=True)
    if eng != OH.COOK_ENGINE:
        print(
            f"[COOKED] running engine {eng} ≠ hull.COOK_ENGINE — if the simulator engine changed, update the pin "
            "and re-cook everything (the judge rejects old dumps as stale)",
            flush=True,
        )
        return 2
    rows = targets(a)
    op, ctx = OH._owners()[0], omni.usd.get_context()
    written, fresh, failed = 0, 0, 0
    for ds, oid, art, f in rows:
        if not a.force and not OH.cooked_problem(f):
            fresh += 1
            continue
        try:
            print(f"[COOKED] {ds}/{oid} {dump_one(ctx, op, art, f)}", flush=True)
            written += 1
        except Exception as e:  # noqa: BLE001 — one object's failure shows up in the counts
            print(f"[COOKED] {ds}/{oid} art={int(art)} FAILED {type(e).__name__}: {e}", flush=True)
            failed += 1
    print(f"[COOKED_DONE] requested={len(rows)} written={written} fresh={fresh} failed={failed}", flush=True)
    ok = verify(a, rows)  # report the verification counts even after failures
    return 0 if not failed and ok else 1


def _rc(e: SystemExit) -> int:
    if isinstance(e.code, str):
        print(e.code, file=sys.stderr)
        return 1
    return int(e.code or 0)


if __name__ == "__main__":
    if "--verify-only" in sys.argv[1:]:
        try:
            sys.exit(main_verify())
        except SystemExit as e:
            sys.exit(_rc(e))
    try:
        rc = main_write()
    except SystemExit as e:  # --help, argument errors, target errors
        rc = _rc(e)
    except BaseException:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        rc = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # close() ends the process with rc 0, so a failing run is not closed and keeps its rc
    if rc == 0 and _APP is not None:
        _APP.close()
    os._exit(rc)
