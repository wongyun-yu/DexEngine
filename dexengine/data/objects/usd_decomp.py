"""Is an object USD baked from the current convex decomposition? (rule S-decomp-fresh: the fingerprint and freshness
step; the part-count cap and the re-bake follow it.)

Statement: the bake stamps, per collision body (the prim whose `collision_<i>` children it authored from one part
folder), the folder it read and that folder's identity (`decompose.folder_fp`) into the root layer's customLayerData
under `STAMP_KEY` (`write_stamp`, the one writer, called by `robots/objects/usd_build._add_collision_parts`).

Oracle (`check`): the USD's collision bodies are compared by content with the part folders the bake reads for the
asset (`convex_dirs`: rigid `<id>/convex`; articulated the `convex_<stem>` folders of every object id the asset id
names, `artifact_paths.asset_object_ids`, so a paired asset finds its parts under the base and moving ids). A body
matches a folder when its points equal, part by part, the vertices of that folder's parts as the bake reads them
(`decompose.load_part`, float32 as authored). Verdicts (`VERDICTS`, first that applies):
  no_decomposition   no candidate folder exists
  stale_content      a body equals no candidate folder (part count or vertices differ)
  stale_folder       every body matches, but a matched folder is not canonical now (`decompose.stale_reason`)
  stamp_mismatch     a stamp exists and does not name the matched folder with its current identity
  current            every body matches a canonical folder and the stamp agrees
  current_unstamped  every body matches a canonical folder; the USD has no stamp (baked before stamps existed)
At the cap: a candidate folder with as many parts as the decomposition allows (`COACD_PARAMS.max_convex_hull`).

Report only: no spawn, judge or tree-freshness path reads it. CLI marker per dataset
`[USD_DECOMP] <ds> usds N · current a · current_unstamped b · not current c {verdict: n} · at cap k` and a final
`[USD_DECOMP_DONE]` line; rc 1 when a dataset outside the frozen generation has a USD that is not current.
  <retarget venv> -m dexengine.data.objects.usd_decomp (--dataset <ds>[,<ds>…] | --all) [--out rows.jsonl]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from dexengine.core import artifact_paths as AP
from dexengine.retarget import decompose as D

STAMP_KEY = "dexengine:decomposition"
VERDICTS = ("no_decomposition", "stale_content", "stale_folder", "stamp_mismatch", "current", "current_unstamped")
CURRENT = ("current", "current_unstamped")
MARK = "[USD_DECOMP]"
_PART = re.compile(r"collision_(\d+)$")


def rel_dir(convex_dir) -> str:
    """The stamp's folder name: `<object id>/<part folder>` (relative to the dataset's object layer)."""
    p = Path(convex_dir)
    return f"{p.parent.name}/{p.name}"


def write_stamp(stage, body_path: str, convex_dir: str) -> dict:
    """Stamp one collision body's part folder into the stage's root layer (merged with the bodies stamped before).
    Returns the entry {dir, fp, parts}."""
    entry = {"dir": rel_dir(convex_dir), "fp": D.folder_fp(convex_dir), "parts": len(D.part_files(convex_dir))}
    layer = stage.GetRootLayer()
    data = dict(layer.customLayerData)
    bodies = dict(data.get(STAMP_KEY) or {})
    bodies[str(body_path)] = entry
    data[STAMP_KEY] = bodies
    layer.customLayerData = data
    return entry


def read_stamp(stage) -> dict:
    """{body path: {dir, fp, parts}} of a stamped USD, {} without a stamp."""
    raw = stage.GetRootLayer().customLayerData.get(STAMP_KEY) or {}
    return {str(k): {"dir": str(v["dir"]), "fp": str(v["fp"]), "parts": int(v["parts"])} for k, v in raw.items()}


def usd_bodies(stage) -> dict:
    """{collision body path: [points (P,3) float32 per part, in part order]} of the meshes with the collision API. A
    collision mesh not named `collision_<i>` is not a bake layout: it raises."""
    from pxr import UsdGeom, UsdPhysics  # noqa: PLC0415

    got: dict = {}
    for p in stage.Traverse():
        if not (p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI)):
            continue
        m = _PART.match(p.GetName())
        if not m:
            raise ValueError(f"{MARK} collision mesh {p.GetPath()} is not named collision_<i>")
        pts = np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(), dtype=np.float32).reshape(-1, 3)
        got.setdefault(str(p.GetParent().GetPath()), []).append((int(m.group(1)), pts))
    return {k: [a for _, a in sorted(v, key=lambda t: t[0])] for k, v in got.items()}


def convex_dirs(objects_root, asset_id: str, articulated: bool) -> list[Path]:
    """The part folders the bake reads for an asset (existing ones only): rigid `<id>/convex`; articulated every
    `convex_<stem>` folder of the object ids the asset id names."""
    root = Path(objects_root)
    if not articulated:
        d = Path(D.rigid_convex_dir(str(root / asset_id)))
        return [d] if d.is_dir() else []
    out = []
    for oid in AP.asset_object_ids(asset_id):
        out += sorted(p for p in (root / oid).glob("convex_*") if p.is_dir())
    return out


def folder_parts(convex_dir: str) -> tuple:
    """The parts of one folder as the bake authors them: (vertices float32 per part, in part order)."""
    return tuple(
        np.asarray(D.load_part(str(Path(convex_dir) / n)).vertices, dtype=np.float32) for n in D.part_files(convex_dir)
    )


def _equal(parts: list, folder: tuple) -> bool:
    if len(parts) != len(folder):
        return False
    return all(a.shape == b.shape and np.array_equal(a, b) for a, b in zip(parts, folder, strict=True))


def folder_problem(convex_dir: Path) -> str | None:
    """Why a part folder is not the canonical decomposition of its source mesh now (None = canonical)."""
    rec = D.read_record(str(convex_dir)) or {}
    return D.stale_reason(str(convex_dir.parent / str(rec.get("source") or "visual.obj")), str(convex_dir))


def _match(bodies: dict, cands: list[Path]) -> tuple[dict, list[str]]:
    """{body: matching folder} and the descriptions of the bodies that match none."""
    have = {c: folder_parts(str(c)) for c in cands}
    hit, miss = {}, []
    for b, parts in sorted(bodies.items()):
        m = next((c for c, fp in have.items() if _equal(parts, fp)), None)
        if m is None:
            miss.append(f"{b} parts {len(parts)} vs candidates { ({rel_dir(c): len(v) for c, v in have.items()}) }")
        else:
            hit[b] = m
    return hit, miss


def _stamp_problem(stamp: dict, hit: dict) -> str:
    bad = [
        b
        for b, c in hit.items()
        if (stamp.get(b) or {}).get("dir") != rel_dir(c) or (stamp.get(b) or {}).get("fp") != D.folder_fp(str(c))
    ]
    extra = sorted(set(stamp) - set(hit))
    return f"stamp disagrees on {bad} · stamped bodies without collision parts {extra}" if bad or extra else ""


def verdict(stage, cands: list[Path]) -> tuple[str, str]:
    """(verdict, why) of one opened USD against its candidate folders (module doc)."""
    if not cands:
        return "no_decomposition", "no part folder"
    bodies = usd_bodies(stage)
    if not bodies:
        return "stale_content", "the USD has no collision parts"
    hit, miss = _match(bodies, cands)
    if miss:
        return "stale_content", " · ".join(miss)[:300]
    folder = {rel_dir(c): why for c in set(hit.values()) if (why := folder_problem(c))}
    if folder:
        return "stale_folder", json.dumps(folder)[:300]
    stamp = read_stamp(stage)
    if not stamp:
        return "current_unstamped", f"content equal to {sorted({rel_dir(c) for c in hit.values()})}"
    why = _stamp_problem(stamp, hit)
    return ("stamp_mismatch", why) if why else ("current", "content and stamp equal")


def at_cap(cands: list[Path]) -> int:
    """Candidate folders with as many parts as the decomposition allows."""
    cap = int(D.COACD_PARAMS["max_convex_hull"])
    return sum(len(D.part_files(str(c))) >= cap for c in cands)


def check(ds: str, asset_id: str, articulated: bool) -> dict:
    """One row for the USD the training spawns for (ds, asset, articulated)."""
    from pxr import Usd  # noqa: PLC0415

    from dexengine.core.locations.layers import layer_path  # noqa: PLC0415
    from dexengine.data.objects import hull as OH  # noqa: PLC0415

    usd = OH.usd_file(ds, asset_id, articulated)
    cands = convex_dirs(layer_path("assets_objects", ds), asset_id, articulated)
    v, why = verdict(Usd.Stage.Open(str(usd)), cands)
    return {
        "ds": ds,
        "id": asset_id,
        "art": bool(articulated),
        "verdict": v,
        "why": why,
        "folders": [rel_dir(c) for c in cands],
        "at_cap": at_cap(cands),
    }


def dataset_usds(ds: str) -> list[tuple[str, bool]]:
    """(asset id, articulated) of every object USD of the dataset (usd_places.object_usds — rigid usd layer ·
    articulated usd_artic layer). No usd layer → empty."""
    from dexengine.data.objects import usd_places as UP  # noqa: PLC0415

    if not Path(AP.get(ds).usd_root()).is_dir():
        return []
    return [(m, art) for m, art, _f in UP.object_usds(ds)]


def marker(ds: str, rows: list[dict], frozen: bool) -> str:
    n = {v: sum(r["verdict"] == v for r in rows) for v in VERDICTS}
    bad = {v: k for v, k in n.items() if k and v not in CURRENT}
    return (
        f"{MARK} {ds} usds {len(rows)} · current {n['current']} · current_unstamped {n['current_unstamped']} · "
        f"not current {sum(bad.values())} {bad} · at cap {sum(r['at_cap'] > 0 for r in rows)}"
        + (" · frozen generation (not re-baked)" if frozen else "")
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", help="comma-separated datasets")
    g.add_argument("--all", action="store_true", help="every dataset with a usd layer")
    ap.add_argument("--out", default="", help="one JSON line per USD")
    a = ap.parse_args(argv)
    from dexengine.core import layout  # noqa: PLC0415

    dss = sorted(AP._valid_datasets()) if a.all else [s.strip() for s in a.dataset.split(",") if s.strip()]
    frozen, bad, total, lines = set(layout.frozen_datasets()), 0, 0, []
    for ds in dss:
        rows = [check(ds, i, art) for i, art in dataset_usds(ds)]
        lines += [json.dumps(r) for r in rows]
        print(marker(ds, rows, ds in frozen), flush=True)
        total += len(rows)
        bad += 0 if ds in frozen else sum(r["verdict"] not in CURRENT for r in rows)
    if a.out:
        Path(a.out).write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
    print(f"[USD_DECOMP_DONE] datasets {len(dss)} · usds {total} · not current outside the frozen generation {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
