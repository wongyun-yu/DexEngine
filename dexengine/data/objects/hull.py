#!/usr/bin/env python3
"""CPU access to object collision shapes: the geometry the simulation actually collides (body frame).

Readers: parts (collision_parts · body_parts) = judge distances, i.e. the object shape of
dexengine.placement.collision.cross_sets (obj_geom="sim" default: xr ②③ · f0_pen ⑤⑥) · D24 articulated root link
(root_link_points) · the object-shape field of the judge input fingerprint (digest). Hull (hull_points) = the same
reading as the PARK pinned pose (`rl/tasks/scene/object_pin._hull_points_body`, the authored collision parts) · D24
rigid bodies · OakInk2 desk recovery (the bottom of still objects).

Parts = **convex hulls cooked by PhysX**: the USD collision parts (convexHull approximation) rebuilt by PhysX under its
vertex limit. This is the shape the simulation collides, and it can be smaller than the authored parts (pair depth
judge−sim p90 0.63, max 3.05 mm). It cannot be read without Isaac, so the bake stage dumps it per object USD (generator
COOKED_GEN, path data_layout.cooked_hull_file). This module owns the dump format (write_cooked) and its reader
(collision_parts). When the dump is missing or stale (source USD sha256, format, cooking engine, fingerprint), the USD
itself is checked (cookable_links, the generator's rule): no rigid body or collision mesh in the USD → HullMissing
(unmeasurable; the caller drops that object); otherwise CookedHullInvalid (a bake-order defect, fatal; it is not a
HullMissing, so it cannot be swallowed as unmeasurable and never falls back to authored parts or visual meshes). Marker
`[OBJ_SHAPE] cooked …`.
  USD file = `artifact_paths.usd_file(mesh_id, articulated)`, the file the training entry point (dexengine/rl/cli.py)
      spawns.
             mesh_id = asset id (asset_id, artifact_paths.object_asset_id; equal to the object id for rigid bodies).
Articulated objects: collision_parts gives the parts of each link (rigid prim) in that link's frame; placing them is the
  caller's job (articulation FK). D24 uses root_link_points (only the root link `base_link`; exact, since the tree pose
  is that link's pose).
A broken correspondence fails (PartMismatch): body count ≠ link count · root link ≠ first body · the link's visual
  points (CORR_Q quantile of the samples in the dump) lie farther than CORR_TOL_M from that body's mesh.
Fixed attached parts (task_info articulation.parts.fixed · fixed_merge; read by artifact_paths.artic_link_files)
  are **one rigid body with their parent link**: the USD carries them inside that link's rigid body and scene.xml as a
  jointless child body (weld). The body = link correspondence is measured on body mesh ∪ welded part meshes (link
  frame), and each part needs visual points that **fit only that part** (a part present on one side only fails; see the
  welded argument of body_parts).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np

from dexengine.core import paths as P

_OP = None
_TP = None
_CACHE: dict = {}


#: Body ↔ USD link correspondence = the CORR_Q quantile of the link's visual points lies within CORR_TOL_M of the body
#: mesh vertices. The bake copies mesh files as is (quantile 0.00 mm); a quantile because the few vertices the bake adds
#: (TACO 007: 7 of 50,790, at most 1.07 mm) belong to the same mesh.
CORR_TOL_M = 1e-3
CORR_Q = 0.99
# link visual-point samples (evenly spaced); enough for the match and keeps the resident judge process cache small
CORR_SAMPLE = 4096
#: Own points of a welded part = visual points within CORR_TOL_M of that part's mesh and closer by this margin than any
#: other component of the link. The bake copies mesh files as is, so own vertices sit at 0.00 mm and other components'
#: vertices are farther than this except where they overlap (distance 0).
WELD_OWN_MARGIN_M = 1e-5
#: Format of the cooked-hull dump (write_cooked ↔ collision_parts); changing it makes every old dump stale (fatal).
COOKED_SCHEMA = "cooked-hull-1"
#: PhysX extension id that cooks the dumps, the one of the Isaac that runs simulation (training, evaluation). The
#: generator fails when the running omni.physx differs; the reader treats a dump from another engine as stale (fatal),
#: since another engine may cook the same USD differently.
COOK_ENGINE = "omni.physx-107.3.26"
COOKED_GEN = "dexengine/robots/objects/cooked_hull.py"  # dump generator (bake stage); named in failure messages


class HullMissing(RuntimeError):
    """The hull cannot be read (no USD, rigid body or collision mesh); the caller records that tree/object as
    unmeasurable."""


class CookedHullInvalid(RuntimeError):
    """The USD has collision meshes but the cooked-hull dump is missing or stale: a bake-order defect, fatal. Not a
    subclass of HullMissing, so callers that accept unmeasurable (D24 no_usd and others) cannot swallow it under the
    wrong label."""


class PartMismatch(ValueError):
    """scene.xml bodies and USD links do not correspond; the judge would measure a shape other than the simulation's, so
    it fails."""


class CollisionParts(NamedTuple):
    """Collision shape of one USD. links: link name → cooked hulls in that link's frame (root link first) ·
    visual: link → visual-point samples (CORR_SAMPLE, float32; correspondence oracle) ·
    digest: first 16 chars of the sha256 of the part contents (link names, order, float64 points); read by the judge
    input fingerprint (objcol)."""

    usd: Path
    links: dict
    visual: dict
    digest: str


def _taco_paths():
    global _TP
    if _TP is None:
        from dexengine.core import artifact_paths as _TP  # noqa: PLC0415  (paths only, no torch/pxr)
    return _TP


def _owners():
    global _OP
    if _OP is None:
        _OP = P.load_registry("object_pin")  # object_pin without the package __init__ (Isaac); the file of this tree
    return _OP, _taco_paths()


def asset_id(o: dict) -> str:
    """task_info objects[] entry → asset id (key of the usd and meshes folders, artifact_paths.object_asset_id(obj_id,
    articulation))."""
    return _taco_paths().object_asset_id(o["obj_id"], o.get("articulation"))


def usd_file(ds: str, mesh_id: str, articulated: bool) -> Path:
    """USD path that training spawns (owner dexengine.core.artifact_paths). The data root is paths.main_root(),
    independent of the working directory."""
    tp = _taco_paths()
    os.environ.setdefault(
        "DEXENGINE_MAIN_ROOT", str(P.main_root())
    )  # root rule of dexengine.core.artifact_paths._abs (data_layout.data_abspath)
    return Path(tp.get(ds).usd_file(str(mesh_id), articulated=bool(articulated)))


def hull_points(ds: str, mesh_id: str, articulated: bool = False) -> np.ndarray:
    """(V,3) float64 collision hull vertices in the body frame; read once per object and cached."""
    key = (ds, str(mesh_id), bool(articulated))
    if key in _CACHE:
        return _CACHE[key]
    op, _ = _owners()
    f = usd_file(ds, mesh_id, articulated)
    if not f.is_file():
        raise HullMissing(f"no USD: {f}")
    from pxr import Usd

    st = Usd.Stage.Open(str(f))
    root = st.GetDefaultPrim()
    if not root or not root.IsValid():
        raise HullMissing(f"{f}: no default prim")
    try:
        pts, _artic = op._hull_points_body(st, str(root.GetPath()))
    except RuntimeError as e:  # object_pin raises RuntimeError when there is no rigid body or collision mesh
        raise HullMissing(f"{f}: {e}") from e
    _CACHE[key] = np.asarray(pts, dtype=np.float64)
    return _CACHE[key]


def usd_sha256(f) -> str:
    """sha256 of the source USD bytes; the key that tells whether a dump is that USD's cooking."""
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def cook_fp(engine: str, settings) -> str:
    """Cooking fingerprint: first 16 chars of the sha256 of the engine id + per-collision-prim cooking settings
    (approximation, vertex limit, min thickness; None when not authored)."""
    return hashlib.sha256(json.dumps([engine, settings], sort_keys=True).encode()).hexdigest()[:16]


def stage_links(st) -> list:
    """Links [(name, prim path)] of a USD stage: RigidBodyAPI prims, root link (object_pin.ARTIC_ROOT_LINK) first, the
    rest in USD order. Name clashes or nesting (a link inside a link) leave the link frame undefined and fail. The rule
    for dump links."""
    from pxr import UsdPhysics

    op, _ = _owners()
    links = [p for p in st.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    links.sort(key=lambda p: p.GetName() != op.ARTIC_ROOT_LINK)
    names, paths = [p.GetName() for p in links], [str(p.GetPath()) for p in links]
    nested = [b for a in paths for b in paths if b.startswith(a + "/")]
    if not links or len(set(names)) != len(names) or nested or (len(links) > 1 and names[0] != op.ARTIC_ROOT_LINK):
        raise HullMissing(
            f"rigid bodies {names}: none, name clash or nesting {nested}, or articulated root ≠ {op.ARTIC_ROOT_LINK}"
        )
    return list(zip(names, paths, strict=True))


def _require_mesh(op, st, lp, visual: bool) -> None:
    """One link's collision (visual=False) or visual mesh exists, else HullMissing."""
    try:
        op._collision_meshes_body(st, lp, visual=visual)
    except RuntimeError as e:  # object_pin raises RuntimeError when there is no rigid body or collision (visual) mesh
        raise HullMissing(str(e)) from e


def cookable_links(st, articulated: bool) -> list:
    """Cookable links [(name, prim path)] of a USD: stage_links, with a link count matching articulated (articulated =
    two or more) and a collision and visual mesh per link (object_pin._collision_meshes_body). Otherwise HullMissing
    (the USD has no hull). One rule for the generator (before writing) and the reader (when the dump is missing or
    stale, to tell a USD problem from a bake-order problem)."""
    op, _ = _owners()
    links = stage_links(st)
    if (len(links) > 1) != bool(articulated):
        raise HullMissing(f"links {[n for n, _ in links]} — do not match articulated={articulated}")
    for _, lp in links:
        for visual in (False, True):
            _require_mesh(op, st, lp, visual)
    return links


def write_cooked(usd, links: dict, engine: str, settings) -> Path:
    """The one dump writer (generator, tests). links = {link: ([cooked vertices (V,3)], [faces (F,3)], visual points
    (N,3))}, root link first, all in that link's frame. Visual points are stored as CORR_SAMPLE samples. Temp file →
    os.replace."""
    usd = Path(usd)
    out = Path(P.data_layout.cooked_hull_file(str(usd)))
    arr = {
        "schema": COOKED_SCHEMA,
        "usd_sha256": usd_sha256(usd),
        "engine": engine,
        "cook_fp": cook_fp(engine, settings),
        "settings": json.dumps(settings, sort_keys=True),
        "links": np.array(list(links)),
    }
    piece_link = []
    for j, (cooked, faces, vis) in enumerate(links.values()):
        v = np.asarray(vis, dtype=np.float64)
        arr[f"vis_{j}"] = v[:: max(1, len(v) // CORR_SAMPLE)].astype(np.float32)
        for c, fc in zip(cooked, faces, strict=True):
            arr[f"v_{len(piece_link)}"] = np.asarray(c, dtype=np.float64)
            arr[f"f_{len(piece_link)}"] = np.asarray(fc, dtype=np.int32)
            piece_link.append(j)
    arr["piece_link"] = np.asarray(piece_link, dtype=np.int64)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arr)
    os.replace(tmp, out)
    return out


def cooked_problem(usd) -> str:
    """Reason the USD's dump is not the current cooking ('' = current); one rule for the reader (collision_parts) and
    the
    check (generator --verify-only)."""
    f = Path(P.data_layout.cooked_hull_file(str(usd)))
    if not f.is_file():
        return f"no dump ({f.name})"
    need = ("schema", "usd_sha256", "engine", "settings", "cook_fp")
    with np.load(f, allow_pickle=False) as z:
        hd = {k: str(z[k]) for k in need if k in z.files}
    if hd.get("schema") != COOKED_SCHEMA:
        return f"format {hd.get('schema')!r} ≠ {COOKED_SCHEMA!r}"
    if len(hd) != len(need):
        return f"header fields missing {sorted(set(need) - set(hd))}"
    if hd["engine"] != COOK_ENGINE:
        return f"cooking engine {hd['engine']!r} ≠ {COOK_ENGINE!r}"
    if hd["cook_fp"] != cook_fp(hd["engine"], json.loads(hd["settings"])):
        return f"cooking fingerprint {hd['cook_fp']} does not match the settings"
    if hd["usd_sha256"] != usd_sha256(usd):
        return f"source USD changed (dump sha256 {hd['usd_sha256'][:12]} ≠ now {usd_sha256(usd)[:12]})"
    return ""


def collision_parts(ds: str, mesh_id: str, articulated: bool) -> CollisionParts:
    """The shape the simulation collides = the cooked-hull dump from the bake: per link (RigidBodyAPI prim) (V_i,3)
    float64
    in that link's frame. Read once per object and cached. Rigid = 1 link · articulated = root link
    (object_pin.ARTIC_ROOT_LINK) first (checks the generator's order). No USD or links not matching articulated =
    HullMissing. Missing or stale dump: HullMissing when the USD fails cookable_links (no collision mesh), otherwise
    CookedHullInvalid (fatal, not unmeasurable)."""
    key = ("parts", ds, str(mesh_id), bool(articulated))
    if key in _CACHE:
        return _CACHE[key]
    op, _ = _owners()
    f = usd_file(ds, mesh_id, articulated)
    if not f.is_file():
        raise HullMissing(f"no USD: {f}")
    why = cooked_problem(f)
    if why:
        from pxr import Usd  # noqa: PLC0415 — the USD is opened only on the failure path

        try:
            cookable_links(Usd.Stage.Open(str(f)), articulated)
        except HullMissing as e:
            raise HullMissing(f"{f}: {e}") from e
        raise CookedHullInvalid(
            f"{f}: cooked hull {why} — the USD has collision meshes: rebuild it with the bake stage {COOKED_GEN}"
        )
    with np.load(P.data_layout.cooked_hull_file(str(f)), allow_pickle=False) as z:
        names, pl = [str(x) for x in z["links"]], z["piece_link"]
        parts = {
            n: tuple(np.ascontiguousarray(z[f"v_{k}"], dtype=np.float64) for k in np.flatnonzero(pl == j))
            for j, n in enumerate(names)
        }
        vis = {n: z[f"vis_{j}"] for j, n in enumerate(names)}
        sha, eng = str(z["usd_sha256"]), str(z["engine"])
    root_ok = not articulated or (bool(names) and names[0] == op.ARTIC_ROOT_LINK)
    if not names or (len(names) > 1) != bool(articulated) or not root_ok or not all(parts.values()):
        raise HullMissing(f"{f}: dump links {names} — do not match articulated={articulated}, or a link has no parts")
    h = hashlib.sha256()
    for nm in names:
        h.update(nm.encode())
        for v in parts[nm]:
            h.update(np.int64(len(v)).tobytes() + v.tobytes())
    print(
        f"[OBJ_SHAPE] cooked {ds}/{mesh_id} art={int(bool(articulated))} links={len(names)} "
        f"parts={sum(len(v) for v in parts.values())} usd={sha[:12]} engine={eng} (objects.hull.collision_parts)",
        flush=True,
    )
    _CACHE[key] = CollisionParts(f, parts, vis, h.hexdigest()[:16])
    return _CACHE[key]


def root_link_points(ds: str, mesh_id: str) -> np.ndarray:
    """(V,3): only the cooked-hull vertices of the articulated USD's root link (object_pin.ARTIC_ROOT_LINK), in that
    link's frame."""
    return np.concatenate(collision_parts(ds, mesh_id, True).links[_owners()[0].ARTIC_ROOT_LINK])


def _mesh_vertices(path: Path) -> np.ndarray:
    """Mesh file vertices (file coordinates as is, trimesh process=False); the correspondence cache calls it once per
    (USD, file)."""
    import trimesh

    return np.asarray(trimesh.load(str(path), force="mesh", process=False).vertices, dtype=np.float64)


def body_parts(ds: str, mesh_id: str, articulated: bool, bodies: list, welded: dict | None = None) -> dict:
    """scene.xml main-object bodies ↔ USD links → {body name: collision parts in that body's frame}.
    bodies = [(body name, [mesh file paths])]: first = parent body (articulated root link), the rest = part bodies.
    welded = {body name: [([mesh file paths], pos (3,), R (3,3))]}: parts welded to that body (jointless children) and
    their pose in the body frame. A welded part is not a body but a component of the body's link (the USD carries it
    inside the link rigid body, so its parts belong to the link).
    Correspondence rule (not by label): body count = link count · first body = root link · for each link exactly one
    body
    whose mesh ∪ welded part meshes lie within CORR_TOL_M (CORR_Q quantile) of the link's visual points (swapping parts
    fails) · a body with welded parts has at least one own point (WELD_OWN_MARGIN_M) per component (body itself, welded
    parts); a part only in scene.xml and not in the USD link fails. Cached per (USD, per-body files, weld poses)."""
    cp = collision_parts(ds, mesh_id, articulated)
    welded = {b: ws for b, ws in (welded or {}).items() if ws}
    wkey = tuple(
        (
            b,
            tuple(
                (tuple(str(x) for x in fs), np.asarray(p, float).tobytes(), np.asarray(R, float).tobytes())
                for fs, p, R in ws
            ),
        )
        for b, ws in sorted(welded.items())
    )
    key = ("corr", str(cp.usd), tuple(tuple(str(x) for x in fs) for _, fs in bodies), *wkey)
    if key not in _CACHE:
        _CACHE[key] = _match(cp, bodies, welded)
    return {b: cp.links[n] for (b, _), n in zip(bodies, _CACHE[key], strict=True)}


def _match(cp: CollisionParts, bodies: list, welded: dict | None = None) -> list:
    """Matching link names in bodies order; rule in the body_parts docstring. A mismatch raises PartMismatch."""
    from scipy.spatial import cKDTree

    names = list(cp.links)
    welded = welded or {}
    stray = sorted(set(welded) - {b for b, _ in bodies})
    if stray:
        raise PartMismatch(
            f"{cp.usd}: bodies {stray} of welded parts are not in the body list {[b for b, _ in bodies]}"
        )
    if len(bodies) != len(names):
        raise PartMismatch(f"{cp.usd}: bodies {[b for b, _ in bodies]} ↔ USD links {names} — counts differ")
    fit, comps = [], []
    for b, files in bodies:
        if not files or not all(fs for fs, _p, _R in welded.get(b, ())):
            raise PartMismatch(
                f"{cp.usd}: body {b} (or its welded parts) has no mesh file — cannot measure the correspondence"
            )
        cs = [np.concatenate([_mesh_vertices(Path(x)) for x in files])]
        cs += [
            np.concatenate([_mesh_vertices(Path(x)) for x in fs]) @ np.asarray(R, float).T + np.asarray(p, float)
            for fs, p, R in welded.get(b, ())
        ]
        kd = cKDTree(np.concatenate(cs))
        fit.append([n for n in names if float(np.quantile(kd.query(cp.visual[n])[0], CORR_Q)) <= CORR_TOL_M])
        comps.append(cs)
    got = [ls[0] if len(ls) == 1 else None for ls in fit]
    if None in got or len(set(got)) != len(got) or got[0] != names[0]:
        raise PartMismatch(
            f"{cp.usd}: bodies {[b for b, _ in bodies]} ↔ link candidates {fit} "
            f"(tolerance {CORR_TOL_M * 1e3:.1f} mm · first body = {names[0]})"
        )
    for (b, _), n, cs in zip(bodies, got, comps, strict=True):
        if len(cs) == 1:
            continue
        d = np.stack([cKDTree(c).query(cp.visual[n])[0] for c in cs])  # (component, visual point)
        own = [
            int(((d[i] <= CORR_TOL_M) & (d[i] + WELD_OWN_MARGIN_M < np.delete(d, i, 0).min(0))).sum())
            for i in range(len(cs))
        ]
        if 0 in own:
            raise PartMismatch(
                f"{cp.usd}: body {b}: own points per component (body · welded parts {len(cs) - 1}) {own} — "
                f"a component has no own point among the visual points of link {n} (a part only in scene.xml, "
                "not in the USD link)"
            )
        print(
            f"[OBJ_SHAPE] weld {cp.usd.parent.name}: body {b} ↔ link {n} · welded parts {len(cs) - 1} · "
            f"own points per component {own} (objects.hull.body_parts)",
            flush=True,
        )
    return got


def bottoms(pts: np.ndarray, R: np.ndarray, t: np.ndarray, axis: int = 2) -> np.ndarray:
    """Lowest hull coordinate per frame: min_v (R·v)[axis] + t[axis]. R (T,3,3) · t (T,3) → (T,)."""
    return (R[:, axis, :] @ pts.T).min(1) + t[:, axis]


def main(argv=None) -> int:
    """Single entry: `--digest <ds> <mesh_id>:<0|1> …` → one JSON line {"<mesh_id>:<0|1>": collision_parts digest}.
    The objcol field of the judge input fingerprint calls this with the retarget python from interpreters without pxr
    (same function, no copy). If any key cannot be read: rc 2 with the reason (no silent empty value)."""
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--digest",
        nargs="+",
        required=True,
        metavar="<ds> <mesh_id>:<0|1>",
        help="one dataset followed by one or more keys",
    )
    a = ap.parse_args(argv).digest
    if len(a) < 2:
        ap.error("--digest <ds> <mesh_id>:<0|1> … — needs at least one key")
    out = {}
    for k in a[1:]:
        mid, _, art = k.rpartition(":")
        if art not in ("0", "1") or not mid:
            print(f"[object_hull] key {k!r} — <mesh_id>:<0|1>", file=sys.stderr)
            return 2
        try:
            out[k] = collision_parts(a[0], mid, art == "1").digest
        except (HullMissing, CookedHullInvalid, PartMismatch, OSError) as e:
            print(f"[object_hull] {a[0]}/{k}: {type(e).__name__}: {e}", file=sys.stderr)
            return 2
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
