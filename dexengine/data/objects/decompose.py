#!/usr/bin/env python3
"""Convex decomposition entry: runs the canonical decomposition per object folder (or mesh file).

Runs in the **retarget venv** (coacd · trimesh · threadpoolctl).

The decomposition arguments, single thread and fingerprint record belong to
`dexengine/retarget/decompose/`; this is only an entry that calls those functions (no arguments
here). The object USD bake (`dexengine/robots/objects/usd_bake.py --decompose_missing`) calls this entry too.

Usage:
  PY="$(python3 dexengine/core/paths.py --what retarget-py)"
  "$PY" dexengine/data/objects/decompose.py --dataset <ds> (--ids a,b | --ids-file F | --stale)
        [--out-root DIR] [--workers N] [--check]
  "$PY" dexengine/data/objects/decompose.py --mesh-file <part.obj> --out-dir <dir>          # articulated part (called
      by the bake)
  --stale     every object of the dataset with a `convex/` whose fingerprint differs from canonical (targets by rule,
      not a name list).
  --out-root  write to `<out-root>/<id>/convex` (staging). Default = in place in the dataset's assets_objects layer.
  --check     print the verdict per object without decomposing (rc 1 = any stale).
  --overwrite overwrite an existing stale convex/; without it that object fails (a changed part count would make
              scene.xml diverge; rule owner decompose.ensure_convex). Not needed for staging (an empty --out-root).
  Datasets of a frozen generation (dexengine.core.layout.frozen_datasets: TACO) are refused (rc 2), also when the
  --out-dir of
  --mesh-file lies inside that object layer (frozen_owner).
Markers: start `[COACD] …` (argument fingerprint, threads) · per object `[DECOMP] <ds>/<id> parts=N action=… reason=…` ·
         end `[DECOMP_DONE] requested=R ok=K failed=F` (rc 1 when R ≠ K).
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STEP = "dexengine.data.objects.decompose"  # capacity.NEEDS key (this module)


def decompose_module():
    """The decomposition owner (`dexengine.retarget.decompose`; imported on first use)."""
    return importlib.import_module("dexengine.retarget.decompose")


def data_layout_module():
    """The data layout tables (dexengine.core.layout)."""
    return importlib.import_module("dexengine.core.layout")


def objects_root(dataset: str) -> Path:
    dl = data_layout_module()
    return Path(dl.data_root(str(REPO))) / dl.render("assets_objects", dataset)


def frozen_owner(out_dir: str) -> str:
    """The dataset whose frozen-generation object layer (physical assets_objects path) contains `out_dir` (realpath),
    else ""."""
    p = os.path.realpath(out_dir)
    for ds in sorted(data_layout_module().frozen_datasets()):
        r = os.path.realpath(objects_root(ds))
        if p == r or p.startswith(r + os.sep):
            return ds
    return ""


def _job(args: tuple[str, str, str, bool]) -> tuple[str, int, str, str]:
    tag, mesh_dir, out_dir, overwrite = args
    try:
        n, action, reason = decompose_module().decompose_object_dir(mesh_dir, out_dir, overwrite=overwrite)
        return tag, n, action, reason
    except Exception as e:  # noqa: BLE001 — a single object's failure shows up in the counts (not skipped silently)
        return tag, -1, "failed", f"{type(e).__name__}: {e}"


def _stale_ids(root: Path) -> list[str]:
    D = decompose_module()
    return [
        d.name
        for d in sorted(p for p in root.iterdir() if (p / "visual.obj").is_file() and (p / "convex").is_dir())
        if D.stale_reason(str(d / "visual.obj"), str(d / "convex"), D.installed_coacd_version()) is not None
    ]


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset")
    ap.add_argument("--ids", default="")
    ap.add_argument("--ids-file")
    ap.add_argument("--stale", action="store_true")
    ap.add_argument("--out-root")
    ap.add_argument("--workers", type=int, default=None, help="pool size (default: capacity.plan)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--mesh-file")
    ap.add_argument("--out-dir")
    return ap


def _one_mesh(ap: argparse.ArgumentParser, a: argparse.Namespace, D) -> int:
    """--mesh-file: decompose one mesh into --out-dir (refused inside a frozen dataset's object layer)."""
    if a.dataset or a.ids or a.ids_file or a.stale or not a.out_dir:
        ap.error("--mesh-file goes only with a single --out-dir")
    if frozen := frozen_owner(a.out_dir):
        print(
            f"[DECOMP] refused: --out-dir {a.out_dir} is inside the object layer of frozen-generation "
            f"dataset {frozen}"
            " (dexengine.core.layout.frozen_datasets) — those object assets are not decomposed again",
            flush=True,
        )
        return 2
    n, action, reason = D.ensure_convex(a.mesh_file, a.out_dir, overwrite=a.overwrite)
    print(f"[DECOMP] {a.mesh_file} parts={n} action={action} reason={reason}", flush=True)
    print("[DECOMP_DONE] requested=1 ok=1 failed=0", flush=True)
    return 0


def _targets(ap: argparse.ArgumentParser, a: argparse.Namespace, root: Path) -> list[str]:
    """Sorted unique object ids from --ids, --ids-file and --stale (none at all is an argument error)."""
    want = [t.strip() for t in a.ids.split(",") if t.strip()]
    if a.ids_file:
        want += [ln.strip() for ln in Path(a.ids_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if a.stale:
        want += _stale_ids(root)
    want = sorted(set(want))
    if not want:
        ap.error("no targets — give one of --ids / --ids-file / --stale")
    return want


def _check(a: argparse.Namespace, D, root: Path, want: list[str]) -> int:
    """--check: report each target's convex parts as current or stale (rc 1 when any is stale)."""
    bad = 0
    for w in want:
        out = Path(a.out_root) / w / "convex" if a.out_root else root / w / "convex"
        r = D.stale_reason(str(root / w / "visual.obj"), str(out), D.installed_coacd_version())
        bad += r is not None
        print(
            f"[DECOMP] {a.dataset}/{w} check={'current' if r is None else 'stale'} reason={r or 'fingerprint match'}",
            flush=True,
        )
    print(f"[DECOMP_DONE] requested={len(want)} current={len(want) - bad} stale={bad}", flush=True)
    return 1 if bad else 0


def _decompose_all(a: argparse.Namespace, root: Path, want: list[str]) -> int:
    """Decompose every target in a process pool sized by capacity (rc 0 only when every object got parts)."""
    from dexengine.core import capacity  # noqa: PLC0415

    jobs = []
    for w in want:
        out = str(Path(a.out_root) / w / "convex") if a.out_root else None
        jobs.append((f"{a.dataset}/{w}", str(root / w), out, a.overwrite))
    ok = 0
    failed = []
    with (
        capacity.sized(STEP, len(jobs), a.workers) as cap,
        ProcessPoolExecutor(cap.workers, initializer=capacity.worker_init, initargs=cap.initargs) as ex,
    ):
        for tag, n, action, reason in ex.map(_job, jobs):
            print(f"[DECOMP] {tag} parts={n} action={action} reason={reason}", flush=True)
            if n > 0:
                ok += 1
            else:
                failed.append(tag)
    print(f"[DECOMP_DONE] requested={len(want)} ok={ok} failed={len(failed)}", flush=True)
    return 0 if ok == len(want) and not failed else 1


def main() -> int:
    ap = _parser()
    a = ap.parse_args()
    # Child processes also start single-threaded (the owner coacd_parts rebinds and checks it with threadpoolctl).
    os.environ["OMP_NUM_THREADS"] = str(decompose_module().COACD_OMP_THREADS)
    D = decompose_module()
    print(D.marker(), flush=True)

    if a.mesh_file:
        return _one_mesh(ap, a, D)

    if not a.dataset:
        ap.error("--dataset is required (no default)")
    if a.dataset in data_layout_module().frozen_datasets():
        print(
            f"[DECOMP] refused: {a.dataset} is read by a frozen generation (dexengine.core.layout.frozen_datasets) — "
            f"its object assets are not decomposed again",
            flush=True,
        )
        return 2
    root = objects_root(a.dataset)
    if not root.is_dir():
        print(f"[DECOMP] no object layer: {root}", flush=True)
        return 2
    want = _targets(ap, a, root)
    missing = [w for w in want if not (root / w / "visual.obj").is_file()]
    if missing:
        print(f"[DECOMP] ids without visual.obj {len(missing)}: {missing[:10]}", flush=True)
        return 2
    print(
        f"[DECOMP] dataset={a.dataset} objects_root={root} out_root={a.out_root or '(in place)'} "
        f"requested={len(want)} workers={a.workers}",
        flush=True,
    )

    if a.check:
        return _check(a, D, root, want)
    return _decompose_all(a, root, want)


if __name__ == "__main__":
    sys.exit(main())
