"""Convex decomposition of object meshes (library part; the one-clip CLI is `__main__.py`).

MuJoCo collides every mesh as its convex hull, so an object's `visual.obj` is split into convex parts with CoACD to
keep containers hollow. Every part folder carries a record (`_decomp.json`) of exactly how its parts were made; parts
made under other settings are refused rather than mixed. The record format and the parameter fingerprint are a
contract shared with thousands of existing records.

Top-level imports are standard library only: the object USD bake imports this module in the Isaac interpreter.
CoACD, trimesh and threadpoolctl are imported where they are used.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil

#: CoACD parameters of every part folder (the fingerprint covers exactly these and the thread count).
COACD_PARAMS: dict = {
    "threshold": 0.07,
    "max_convex_hull": 16,
    "preprocess_mode": "auto",
    "preprocess_resolution": 50,
    "resolution": 2000,
    "mcts_nodes": 50,
    "mcts_iterations": 200,
    "mcts_max_depth": 5,
    "pca": False,
    "merge": True,
    "decimate": False,
    "max_ch_vertex": 32,
    "extrude": False,
    "apx_mode": "ch",
    "seed": 1,
}
#: OpenMP threads CoACD runs with (part bytes depend on it).
COACD_OMP_THREADS = 1
RECORD_NAME = "_decomp.json"
RECORD_SCHEMA = 1


def params_fingerprint() -> str:
    """First 16 hex characters of the SHA-256 of the parameter set and thread count."""
    blob = json.dumps({"params": COACD_PARAMS, "omp_threads": COACD_OMP_THREADS}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def file_md5(path: str) -> str:
    """MD5 hex digest of a file's bytes."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def installed_coacd_version() -> str:
    """Version string of the installed CoACD package (written into every record)."""
    return importlib.metadata.version("coacd")


def part_files(output_dir: str) -> list[str]:
    """Names of the `*.obj` part files in `output_dir`, ordered by the integer value of their stem."""
    names = [n for n in os.listdir(output_dir) if n.endswith(".obj")]
    bad = [n for n in names if not n[: -len(".obj")].isdigit()]
    if bad:
        raise ValueError(f"[COACD] {output_dir}: part files must be named <n>.obj, found {sorted(bad)}")
    return sorted(names, key=lambda n: int(n[: -len(".obj")]))


def read_record(output_dir: str) -> dict | None:
    """The parsed `_decomp.json` of a part folder, or None when there is none."""
    path = os.path.join(output_dir, RECORD_NAME)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def stale_reason(input_file: str, output_dir: str, coacd_version: str | None = None) -> str | None:
    """None when the parts in `output_dir` are current for `input_file`; otherwise why they are not."""
    if not os.path.isdir(output_dir):
        return f"no part folder {output_dir}"
    rec = read_record(output_dir)
    if rec is None:
        return f"no {RECORD_NAME}"
    if rec.get("schema") != RECORD_SCHEMA:
        return f"record schema {rec.get('schema')!r} != {RECORD_SCHEMA}"
    if rec.get("params_fp") != params_fingerprint():
        return f"params_fp {rec.get('params_fp')!r} != {params_fingerprint()}"
    if coacd_version is not None and rec.get("coacd") != coacd_version:
        return f"coacd {rec.get('coacd')!r} != installed {coacd_version}"
    if rec.get("source_md5") != file_md5(input_file):
        return "source mesh changed (md5)"
    parts = rec.get("parts") or {}
    present = part_files(output_dir)
    if set(parts) != set(present):
        return f"part files {present} != record {sorted(parts)}"
    for name in present:
        if file_md5(os.path.join(output_dir, name)) != parts[name]:
            return f"part {name} changed (md5)"
    return None


def folder_fp(output_dir: str) -> str:
    """Identity of one part folder's decomposition: the first 16 hex characters of the SHA-256 of its record's
    parameter fingerprint, source mesh md5 and part md5s. The object USD bake stamps it per collision body
    (`dexengine/data/objects/usd_decomp.py`); a folder without a record raises (nothing to identify)."""
    rec = read_record(output_dir)
    if rec is None:
        raise FileNotFoundError(f"[COACD] no {RECORD_NAME} in {output_dir}")
    blob = json.dumps(
        {"params_fp": rec.get("params_fp"), "source_md5": rec.get("source_md5"), "parts": rec.get("parts") or {}},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_part(path: str):
    """One part file as the bake reads it (trimesh, no processing: vertex order kept). The bake authors these vertices
    as a collision prim's points; the USD freshness oracle compares against the same reading."""
    trimesh = importlib.import_module("trimesh")
    return trimesh.load(path, force="mesh", process=False)


def coacd_parts(vertices, faces) -> list:
    """CoACD convex parts `[(V, F), …]` of one mesh, run with a single OpenMP thread (checked, not assumed)."""
    coacd = importlib.import_module("coacd")
    threadpoolctl = importlib.import_module("threadpoolctl")
    coacd.set_log_level("error")
    with threadpoolctl.threadpool_limits(limits=COACD_OMP_THREADS, user_api="openmp"):
        pools = [p for p in threadpoolctl.threadpool_info() if p.get("user_api") == "openmp"]
        if not any("coacd" in str(p.get("filepath", "")).lower() for p in pools):
            raise RuntimeError(
                f"[COACD] no OpenMP pool of the CoACD library is visible: {[p.get('filepath') for p in pools]}"
            )
        wrong = [(p.get("filepath"), p.get("num_threads")) for p in pools if p.get("num_threads") != COACD_OMP_THREADS]
        if wrong:
            raise RuntimeError(f"[COACD] OpenMP pools not limited to {COACD_OMP_THREADS} thread(s): {wrong}")
        parts = coacd.run_coacd(coacd.Mesh(vertices, faces), **COACD_PARAMS)
    if not parts:
        raise ValueError("[COACD] decomposition returned no parts")
    return list(parts)


def _write_record(output_dir: str, input_file: str, n: int) -> None:
    trimesh = importlib.import_module("trimesh")
    names = part_files(output_dir)
    rec = {
        "schema": RECORD_SCHEMA,
        "params_fp": params_fingerprint(),
        "params": COACD_PARAMS,
        "omp_threads": COACD_OMP_THREADS,
        "coacd": installed_coacd_version(),
        "trimesh": trimesh.__version__,
        "code_md5": file_md5(os.path.abspath(__file__)),
        "source": os.path.basename(input_file),
        "source_md5": file_md5(input_file),
        "n_parts": n,
        "parts": {name: file_md5(os.path.join(output_dir, name)) for name in names},
    }
    tmp = os.path.join(output_dir, f"{RECORD_NAME}.tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    os.replace(tmp, os.path.join(output_dir, RECORD_NAME))


@contextlib.contextmanager
def _folder_lock(output_dir: str):
    """Exclusive lock (flock) on the folder that holds `output_dir`: one writer per object folder at a time, across
    processes. Clips that share an object wait here instead of writing the same part folder together."""
    parent = os.path.dirname(os.path.abspath(output_dir))
    os.makedirs(parent, exist_ok=True)
    fd = os.open(parent, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)  # closing the descriptor releases the lock


def _work_dirs(output_dir: str) -> tuple[str, str]:
    """The hidden siblings a write of `output_dir` uses: the folder being written and the replaced folder."""
    parent, name = os.path.split(os.path.abspath(output_dir))
    return os.path.join(parent, f".{name}.tmp{os.getpid()}"), os.path.join(parent, f".{name}.old{os.getpid()}")


def _clear_leftovers(output_dir: str) -> None:
    """Remove the hidden siblings a killed write of `output_dir` left (only under the folder lock: no writer is
    active then)."""
    parent, name = os.path.split(os.path.abspath(output_dir))
    for entry in os.listdir(parent):
        if entry.startswith((f".{name}.tmp", f".{name}.old")):
            shutil.rmtree(os.path.join(parent, entry))


def decompose_mesh(input_file: str, output_dir: str) -> int:
    """Decompose `input_file` into the part folder `output_dir` (`<i>.obj` and the record); returns the count.

    Parts and record are written into a hidden sibling folder that then takes the place of `output_dir` by rename, so
    a reader sees the old complete folder, no folder, or the new complete folder, never a partial one, and a killed
    run leaves only hidden siblings. Called under the folder lock (`ensure_convex`)."""
    trimesh = importlib.import_module("trimesh")
    mesh = trimesh.load(input_file, force="mesh", process=False, skip_materials=True)
    parts = coacd_parts(mesh.vertices, mesh.faces)
    tmp, old = _work_dirs(output_dir)
    os.makedirs(tmp)
    for i, (v, f) in enumerate(parts):
        trimesh.Trimesh(v, f).export(os.path.join(tmp, f"{i}.obj"))
    written = len(part_files(tmp))
    if written != len(parts):
        raise RuntimeError(f"[COACD] wrote {written} part files for {len(parts)} parts in {tmp}")
    _write_record(tmp, input_file, written)
    if os.path.isdir(output_dir):
        os.rename(output_dir, old)
        os.rename(tmp, output_dir)
        shutil.rmtree(old)
    else:
        os.rename(tmp, output_dir)
    return written


def ensure_convex(input_file: str, output_dir: str, overwrite: bool = False) -> tuple[int, str, str]:
    """Current parts are reused; missing parts are made; stale parts are remade only with `overwrite`.

    Returns `(n_parts, "reused" | "decomposed", reason)`. A stale existing folder without `overwrite` raises
    `RuntimeError`: decomposing again could change the part count and silently diverge from scene files that list
    parts by name. That includes a folder without a record (frozen parts carry none); writes through here never leave
    one. Making parts holds the folder lock, and the check is repeated under it, so a writer that waited reuses the
    parts the one before it made.
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"[COACD] mesh missing: {input_file}")
    reason = stale_reason(input_file, output_dir)
    if reason is None and not overwrite:
        return len(part_files(output_dir)), "reused", "fingerprint match"
    with _folder_lock(output_dir):
        _clear_leftovers(output_dir)
        reason = stale_reason(input_file, output_dir)
        if reason is None and not overwrite:
            return len(part_files(output_dir)), "reused", "fingerprint match"
        if reason is not None and os.path.isdir(output_dir) and not overwrite:
            raise RuntimeError(f"[COACD] {output_dir} holds stale parts ({reason}); pass --overwrite to remake them")
        n = decompose_mesh(input_file, output_dir)
    return n, "decomposed", reason or "--overwrite"


def rigid_convex_dir(mesh_dir: str) -> str:
    """Part folder of a rigid object: `<mesh_dir>/convex`."""
    return os.path.join(mesh_dir, "convex")


def part_convex_dir(mesh_file: str) -> str:
    """Part folder of an articulated part mesh: `<folder>/convex_<stem>`."""
    folder, name = os.path.split(mesh_file)
    return os.path.join(folder, f"convex_{os.path.splitext(name)[0]}")


def decompose_object_dir(mesh_dir: str, output_dir: str | None = None, overwrite: bool = False) -> tuple[int, str, str]:
    """`ensure_convex` of an object folder's `visual.obj` into `output_dir` (default `<mesh_dir>/convex`)."""
    return ensure_convex(os.path.join(mesh_dir, "visual.obj"), output_dir or rigid_convex_dir(mesh_dir), overwrite)


def frozen_datasets() -> frozenset:
    """Datasets whose parts are frozen (owner: `dexengine.core.layout.frozen_datasets`, imported lazily)."""
    layout = importlib.import_module("dexengine.core.layout")
    return frozenset(layout.frozen_datasets())


#: A frozen dataset's parts are the shipped input `<dataset>_convex` (dexengine.core.shipped): CoACD cannot rebuild
#: them (with the canonical parameters + decimate=True, 12 of 19 TACO objects come out byte-equal and 7 differ).
FROZEN_PARTS_INPUT = "{dataset}_convex"
_FROZEN_CHECKED: set = set()


def frozen_convex(mesh_dir: str, dataset: str) -> int:
    """Part count of a frozen dataset's object (its parts are used as they are, never made again). The object folder's
    parent is checked once per process as the shipped input FROZEN_PARTS_INPUT (a release root is verified, an
    unverified root passes as `site`); a missing part folder dies naming that input."""
    name = FROZEN_PARTS_INPUT.format(dataset=dataset)
    objects_root = os.path.dirname(os.path.abspath(mesh_dir))
    if (name, objects_root) not in _FROZEN_CHECKED:
        importlib.import_module("dexengine.core.shipped").state(name, objects_root)
        _FROZEN_CHECKED.add((name, objects_root))
    cdir = rigid_convex_dir(mesh_dir)
    n = len(part_files(cdir)) if os.path.isdir(cdir) else 0
    if n == 0:
        raise RuntimeError(
            f"[DECOMP] frozen dataset object has no convex parts: {cdir} — they are the shipped input {name} "
            f"(configs/release/shipped_inputs.tsv), unpacked into {objects_root}"
        )
    return n


def marker() -> str:
    """Run marker line: parameter fingerprint, decimation, thread count and CoACD version."""
    return (
        f"[COACD] params_fp={params_fingerprint()} decimate={COACD_PARAMS['decimate']} "
        f"threads={COACD_OMP_THREADS} coacd={installed_coacd_version()}"
    )
