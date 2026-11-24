#!/usr/bin/env python3
"""Object assets of the `mano`-driver datasets (GRAB, ARCTIC, H2O, OakInk2), built from their raw downloads.

    python -m dexengine.data.objects.build <ds> --raw <data/<ds>/raw> --out <data/<ds>/assets/objects>
           (--mano <data/<ds>/mano/bimanual> | --objects a,b | --all) [--workers N] [--visual-only] [--overwrite]

Writes `<out>/<obj>/visual.obj` (metres, object frame) and its convex collision parts `<out>/<obj>/convex/`: the
folders the readers' `task_info.json` points at (`{side}_object_mesh_dir = processed/<ds>/assets/objects/<obj>`).
ARCTIC objects are articulated: `visual.obj` (the whole object at articulation 0) plus the part meshes `bottom.obj`,
`top.obj` (task_info `articulated_object`), and only the parts get convex parts (`convex_bottom/`, `convex_top/`).

Selection, one is required (no default): `--mano` = every object the reader's clips name (`right/left_object_mesh_dir`
of each `task_info.json` in that MANO layer), `--objects` = these names, `--all` = every object the raw release holds.

Source adapter per dataset (ADAPTERS): where the raw mesh lives and the asset name (both owned by the reader module
`dexengine/data/readers/<ds>.py`), the raw length unit, a frame turn of the vertices, and the articulated part files.
No dataset turns its mesh: OakInk2's y-up -> z-up turn is applied to the object pose by the reader, so the mesh stays
in the object frame (the delivered meshes equal the raw vertices exactly).

visual.obj = the raw `v` list and triangle list in raw order (OBJ texture and normal indices are dropped, so the mesh is
never split at UV or normal seams; polygons are fan-triangulated). Its face-corner positions therefore equal the raw
mesh's, which is what the texture raw re-intake aligns on (dexengine/robots/textures/rules.py RAW_SOURCES); no
material or image is copied. Bytes are deterministic: a rerun leaves an equal file untouched and reuses its parts. A
file with other content is replaced only with `--overwrite` (its convex parts are then decomposed again).

Convex parts come from the decomposition owner (dexengine/retarget/decompose/: parameters, one
OpenMP thread, `_decomp.json` record) through `ensure_convex`; this module has no decomposition arguments.

Markers: `[COACD] …` once (not with --visual-only) · per object `[OBJ_BUILD] <ds>/<obj> <file>=<V>v/<F>f(<action>) …
convex/<dir>=<parts>(<action>)` · end `[OBJ_BUILD_DONE] requested=R ok=K failed=F` (rc 1 when K != R).
Runs in the retarget venv (trimesh; coacd and threadpoolctl unless --visual-only).
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dexengine.core import capacity

STEP = "dexengine.data.objects.build"  # capacity.NEEDS key (this module)
VISUAL = "visual.obj"
SIDES = ("right", "left")


@dataclass(frozen=True)
class MeshRef:
    """One raw mesh: a file, or the member `member` of the zip `path`."""

    path: str
    member: str = ""

    @property
    def fmt(self) -> str:
        return os.path.splitext(self.member or self.path)[1].lower().lstrip(".")

    def read(self) -> bytes:
        if self.member:
            with zipfile.ZipFile(self.path) as z:
                return z.read(self.member)
        return Path(self.path).read_bytes()


@dataclass(frozen=True)
class Adapter:
    """One dataset's raw object meshes and how they become assets."""

    refs: dict  # asset name -> {asset file: MeshRef}
    units_per_m: float = 1.0  # raw length units per metre (vertices are divided by it)
    rotation: tuple | None = None  # 3x3 turn of the scaled vertices into the asset frame (None = same frame)
    parts: tuple = field(default=())  # articulated part files: with parts, only they get convex parts


def decompose_module():
    """The decomposition owner (imported on first use: it needs coacd only when decomposing)."""
    return importlib.import_module("dexengine.retarget.decompose")


# ── raw mesh reading and the asset writer ─────────────────────────────────────────────────────────────────
def parse_obj(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """`v` and `f` lines of an OBJ in file order -> ((V, 3) float64, (F, 3) int64 0-based).

    Polygons are fan-triangulated; texture and normal indices are dropped; negative (relative) indices are resolved."""
    verts: list = []
    faces: list = []
    for ln in data.decode("utf-8", errors="replace").splitlines():
        if ln.startswith("v "):
            verts.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "):
            idx = [int(t.split("/")[0]) for t in ln.split()[1:]]
            idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
            faces.extend([idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1))
    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(f) and (f.min() < 0 or f.max() >= len(v)):
        raise ValueError(f"face index out of range ({len(v)} vertices)")
    return v, f


def parse_ply(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """A PLY's vertex and face lists in file order (trimesh without processing)."""
    import trimesh

    m = trimesh.load(io.BytesIO(data), file_type="ply", force="mesh", process=False)
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


PARSERS = {"obj": parse_obj, "ply": parse_ply}


def read_mesh(ref: MeshRef) -> tuple[np.ndarray, np.ndarray]:
    if ref.fmt not in PARSERS:
        raise ValueError(f"{ref}: no parser for .{ref.fmt} (known: {sorted(PARSERS)})")
    v, f = PARSERS[ref.fmt](ref.read())
    if not len(v) or not len(f):
        raise ValueError(f"{ref}: empty mesh ({len(v)} vertices, {len(f)} faces)")
    return v, f


def to_asset_frame(v: np.ndarray, units_per_m: float, rotation) -> np.ndarray:
    """Raw vertices -> metres in the asset frame: divide by the unit, then turn (row vectors: v @ R^T)."""
    out = v / float(units_per_m)
    if rotation is not None:
        out = out @ np.asarray(rotation, dtype=np.float64).T
    return out


def obj_bytes(v: np.ndarray, f: np.ndarray, note: str) -> bytes:
    """The asset OBJ: one comment line, `v` lines (16 decimals), `f` lines (1-based)."""
    lines = [f"# {note}"]
    lines += [f"v {x:.16f} {y:.16f} {z:.16f}" for x, y, z in v.tolist()]
    lines += [f"f {a} {b} {c}" for a, b, c in (f + 1).tolist()]
    return ("\n".join(lines) + "\n").encode("ascii")


def write_file(path: Path, data: bytes, overwrite: bool) -> str:
    """Write `data` unless the same bytes are there ("same"); other bytes are replaced only with `overwrite`."""
    if path.is_file():
        if path.read_bytes() == data:
            return "same"
        if not overwrite:
            raise RuntimeError(
                f"{path} exists with other content — pass --overwrite to replace it (its convex parts "
                f"are then decomposed again)"
            )
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return "written"


# ── one object ────────────────────────────────────────────────────────────────────────────────────────────
def build_object(
    ds: str, name: str, ad: Adapter, out_root: str, overwrite: bool = False, visual_only: bool = False
) -> dict:
    """Write one asset folder -> {"name", "files": {file: [V, F, action]}, "convex": {dir: [parts, action, reason]}}."""
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {"name": name, "files": {}, "convex": {}}
    for fname, ref in sorted(ad.refs[name].items()):
        v, f = read_mesh(ref)
        v = to_asset_frame(v, ad.units_per_m, ad.rotation)
        data = obj_bytes(v, f, f"DexEngine object asset {ds}/{name}/{fname} (metres) - dexengine.data.objects.build")
        res["files"][fname] = [len(v), len(f), write_file(out / fname, data, overwrite)]
    if visual_only:
        return res
    D = decompose_module()
    missing = [p for p in ad.parts if p not in res["files"]]
    if missing:
        raise ValueError(f"{ds}/{name}: part meshes {missing} not among the written files {sorted(res['files'])}")
    if ad.parts:
        targets = [(str(out / p), D.part_convex_dir(str(out / p))) for p in ad.parts]
    else:
        targets = [(str(out / VISUAL), D.rigid_convex_dir(str(out)))]
    for mesh, cdir in targets:
        stale = D.stale_reason(mesh, cdir, D.installed_coacd_version())
        n, action, reason = D.ensure_convex(mesh, cdir, overwrite=overwrite and stale is not None)
        res["convex"][os.path.basename(cdir)] = [n, action, reason]
    return res


def _job(args: tuple) -> tuple[str, dict | None, str]:
    ds, name, ad, out_root, overwrite, visual_only = args
    try:
        return name, build_object(ds, name, ad, out_root, overwrite, visual_only), ""
    except Exception as e:  # noqa: BLE001 — one object's failure is counted in the end marker, never skipped
        return name, None, f"{type(e).__name__}: {e}"


# ── source adapters: raw locations and names are the readers' ────────────────────────────────────────────
def _grab(raw: Path) -> Adapter:
    from dexengine.data.readers import grab as G

    zp = raw / G.MESH_ZIP
    if not zp.is_file():
        raise FileNotFoundError(f"GRAB object meshes {zp} missing")
    with zipfile.ZipFile(zp) as z:
        members = z.namelist()
    refs = {Path(m).stem: {VISUAL: MeshRef(str(zp), m)} for m in members if m == G.mesh_member(Path(m).stem)}
    return Adapter(refs=refs)


def _arctic(raw: Path) -> Adapter:
    from dexengine.data.readers import arctic as A

    root = A.template_dir(raw)
    refs = {}
    for d in sorted(p for p in root.iterdir() if (p / A.WHOLE_MESH).is_file()):
        refs[d.name] = {
            VISUAL: MeshRef(str(d / A.WHOLE_MESH)),
            **{f: MeshRef(str(d / f)) for f in A.PART_MESHES.values()},
        }
    return Adapter(refs=refs, units_per_m=A.MESH_UNITS_PER_M, parts=tuple(A.PART_MESHES.values()))


def _h2o(raw: Path) -> Adapter:
    from dexengine.data.readers import h2o as H

    zp = H.dl_root(raw) / H.MESH_ZIP
    if not zp.is_file():
        raise FileNotFoundError(f"H2O object meshes {zp} missing")
    return Adapter(refs={n: {VISUAL: MeshRef(str(zp), H.mesh_member(n))} for n in sorted(set(H.CLASS_NAMES.values()))})


def _oakink2(raw: Path) -> Adapter:
    from dexengine.data.readers import oakink2 as O

    align = O.hf_root(raw) / O.MESH_SUB
    refs = {}
    for d in sorted(align.iterdir()):
        if (d / O.MESH_FILE).is_file():
            for side in SIDES:
                refs[O.asset_name(side, d.name)] = {VISUAL: MeshRef(str(d / O.MESH_FILE))}
    return Adapter(refs=refs)


#: dataset -> adapter factory (raw root -> Adapter); the datasets whose reader runs under the `mano` driver
ADAPTERS = {"grab": _grab, "arctic": _arctic, "h2o": _h2o, "oakink2": _oakink2}


# ── selection and CLI ─────────────────────────────────────────────────────────────────────────────────────
def mano_names(ds: str, mano_dir: str | Path) -> list[str]:
    """Object names the clips of a MANO layer refer to (`{side}_object_mesh_dir` of each `<clip>/task_info.json`)."""
    names, n = set(), 0
    for p in sorted(Path(mano_dir).glob("*/task_info.json")):
        ti = json.loads(p.read_text(encoding="utf-8"))
        if ti.get("dataset_name") != ds:
            raise ValueError(f"{p}: dataset_name {ti.get('dataset_name')!r} != {ds!r}")
        n += 1
        names.update(os.path.basename(str(v).rstrip("/")) for s in SIDES if (v := ti.get(f"{s}_object_mesh_dir")))
    if not n:
        raise FileNotFoundError(f"no <clip>/task_info.json under {mano_dir}")
    return sorted(names)


def run(
    ds: str,
    ad: Adapter,
    want: list[str],
    out_root: str,
    workers: int | None = None,
    overwrite: bool = False,
    visual_only: bool = False,
) -> int:
    """Build `want` into `out_root` with markers; rc 0 only when every requested object is built."""
    if not visual_only:
        D = decompose_module()
        os.environ["OMP_NUM_THREADS"] = str(D.COACD_OMP_THREADS)  # pool children start single-threaded too
        print(D.marker(), flush=True)
    print(
        f"[OBJ_BUILD] dataset={ds} out={out_root} requested={len(want)} (pool: the [CPU] line) "
        f"convex={'off (--visual-only)' if visual_only else 'on'}",
        flush=True,
    )
    jobs = [(ds, w, ad, out_root, overwrite, visual_only) for w in want]
    with capacity.sized(STEP, len(jobs), workers) as cap:
        pool = (
            ProcessPoolExecutor(cap.workers, initializer=capacity.worker_init, initargs=cap.initargs)
            if cap.workers > 1
            else None  # one worker: in this process
        )
        return _collect(ds, want, jobs, pool)


def _collect(ds: str, want: list, jobs: list, pool) -> int:
    ok, failed = 0, []
    try:
        for name, res, err in pool.map(_job, jobs) if pool else map(_job, jobs):
            if res is None:
                failed.append(name)
                print(f"[OBJ_BUILD] {ds}/{name} FAILED {err}", flush=True)
                continue
            ok += 1
            fs = " ".join(f"{k}={v[0]}v/{v[1]}f({v[2]})" for k, v in res["files"].items())
            cs = " ".join(f"{k}={v[0]}({v[1]}: {v[2]})" for k, v in res["convex"].items())
            print(f"[OBJ_BUILD] {ds}/{name} {fs} {cs}".rstrip(), flush=True)
    finally:
        if pool:
            pool.shutdown()
    print(f"[OBJ_BUILD_DONE] requested={len(want)} ok={ok} failed={len(failed)}", flush=True)
    return 0 if ok == len(want) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset", choices=sorted(ADAPTERS))
    ap.add_argument("--raw", required=True, help="raw dataset root (the data/<ds>/raw layer)")
    ap.add_argument("--out", required=True, help="object layer to write (data/<ds>/assets/objects)")
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--mano", help="MANO layer whose task_info.json files name the objects (data/<ds>/mano/bimanual)")
    sel.add_argument("--objects", help="comma-separated asset names")
    sel.add_argument("--all", action="store_true", help="every object the raw release holds")
    ap.add_argument("--workers", type=int, default=None, help="pool size (default: capacity.plan)")
    ap.add_argument("--visual-only", action="store_true", help="write the meshes only (no convex decomposition)")
    ap.add_argument("--overwrite", action="store_true", help="replace meshes with other content and their parts")
    a = ap.parse_args(argv)
    from dexengine.core import layout

    if a.dataset in layout.frozen_datasets():
        print(f"[OBJ_BUILD] refused: {a.dataset} is read by a frozen generation (layout.frozen_datasets)", flush=True)
        return 2
    ad = ADAPTERS[a.dataset](Path(a.raw))
    if a.mano:
        want = mano_names(a.dataset, a.mano)
    elif a.objects:
        want = sorted({w.strip() for w in a.objects.split(",") if w.strip()})
    else:
        want = sorted(ad.refs)
    unknown = [w for w in want if w not in ad.refs]
    if unknown or not want:
        print(
            f"[OBJ_BUILD] {len(unknown)} names without a raw mesh under {a.raw}: {unknown[:10]} "
            f"(raw objects {len(ad.refs)}, requested {len(want)})",
            flush=True,
        )
        return 2
    return run(a.dataset, ad, want, a.out, a.workers, a.overwrite, a.visual_only)


if __name__ == "__main__":
    sys.exit(main())
