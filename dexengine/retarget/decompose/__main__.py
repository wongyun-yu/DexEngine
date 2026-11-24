"""Convex decomposition CLI for one clip (`python -m dexengine.retarget.decompose`).

Callers pass: --dataset-dir --dataset-name --embodiment-type --task --data-id 0 [--overwrite]. Prints the
`[COACD] …` marker once and, for a frozen dataset, one `[DECOMP] frozen reuse …` line (callers parse it).
Unknown flags fail.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

from dexengine.retarget import decompose as D
from dexengine.retarget import mjutil

SIDES_OF = {"bimanual": ("right", "left"), "right": ("right",), "left": ("left",)}


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m dexengine.retarget.decompose", description=__doc__, allow_abbrev=False)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--embodiment-type", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--data-id", type=int, default=0)
    ap.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return ap


def _process(
    dataset_dir: str, rel_mesh_dir: str, frozen: str | None, overwrite: bool, label: str
) -> tuple[str, int] | None:
    """Decompose (or reuse, for a frozen dataset — `frozen` = its name) one object folder; returns (relative convex
    dir, parts)."""
    mesh_dir = os.path.join(dataset_dir, rel_mesh_dir)
    visual = os.path.join(mesh_dir, "visual.obj")
    if not os.path.isfile(visual):
        warnings.warn(f"[DECOMP] {label}: no visual mesh {visual}; skipped", stacklevel=2)
        return None
    if frozen:
        n = D.frozen_convex(mesh_dir, frozen)
        print(f"[DECOMP] {label}: {n} parts (frozen dataset, used as they are)", file=sys.stderr)
    else:
        n, action, reason = D.decompose_object_dir(mesh_dir, overwrite=overwrite)
        print(f"[COACD] {label}: {action} {n} parts ({reason})", file=sys.stderr)
    return D.rigid_convex_dir(rel_mesh_dir.rstrip("/")), n


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.embodiment_type not in SIDES_OF:
        raise ValueError(f"[DECOMP] unknown embodiment {args.embodiment_type!r} (expected {sorted(SIDES_OF)})")
    mano_dir = mjutil.processed_data_dir(
        args.dataset_dir, args.dataset_name, "mano", args.embodiment_type, args.task, args.data_id
    )
    info_file = os.path.join(os.path.dirname(mano_dir), "task_info.json")
    if not os.path.isfile(info_file):
        raise FileNotFoundError(f"[DECOMP] task_info.json missing: {info_file}")
    with open(info_file) as f:
        info = json.load(f)
    frozen = args.dataset_name in D.frozen_datasets()
    frozen_ds = args.dataset_name if frozen else None
    if frozen and args.overwrite:
        raise ValueError(
            f"[DECOMP] {args.dataset_name} is a frozen dataset: its parts are never made again (--overwrite refused)"
        )
    print(D.marker(), flush=True)
    n_objects = n_parts = 0
    for side in SIDES_OF[args.embodiment_type]:
        rel = info.get(f"{side}_object_mesh_dir")
        if not rel:
            continue
        res = _process(args.dataset_dir, rel, frozen_ds, args.overwrite, f"{args.task} {side}")
        if res is not None:
            info[f"{side}_object_convex_dir"] = res[0]
            n_objects, n_parts = n_objects + 1, n_parts + res[1]
    for i, extra in enumerate(info.get("extra_objects") or []):
        rel = extra.get("mesh_dir")
        if not rel:
            continue
        res = _process(args.dataset_dir, rel, frozen_ds, args.overwrite, f"{args.task} extra {i}")
        if res is not None:
            extra["convex_dir"] = res[0]
            n_objects, n_parts = n_objects + 1, n_parts + res[1]
    tmp = f"{info_file}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(info, f, indent=2)
    os.replace(tmp, info_file)
    if frozen:
        print(
            f"[DECOMP] frozen reuse ds={args.dataset_name} n={n_objects} parts={n_parts} task={args.task}", flush=True
        )
    return 0


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.retarget.decompose")
    sys.exit(main())
