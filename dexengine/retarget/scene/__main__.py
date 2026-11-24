"""Scene builder CLI (`python -m dexengine.retarget.scene`).

Callers pass: --dataset-dir --dataset-name --task --data-id 0 --robot-type --embodiment-type --no-show-viewer
[--use-visual-mesh-as-collision]. Unknown flags fail.
"""

from __future__ import annotations

import argparse
import sys


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m dexengine.retarget.scene", description=__doc__, allow_abbrev=False)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--data-id", type=int, default=0)
    ap.add_argument("--robot-type", required=True)
    ap.add_argument("--embodiment-type", required=True, choices=("bimanual", "right", "left"))
    ap.add_argument("--show-viewer", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--use-visual-mesh-as-collision", action=argparse.BooleanOptionalAction, default=False)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.show_viewer:
        raise NotImplementedError("the scene viewer is not supported; run with --no-show-viewer")
    from dexengine.retarget.scene.compose import build_scene  # noqa: PLC0415  (MuJoCo import after argument checks)

    build_scene(
        args.dataset_dir,
        args.dataset_name,
        args.task,
        args.data_id,
        args.robot_type,
        args.embodiment_type,
        use_visual_mesh_as_collision=args.use_visual_mesh_as_collision,
    )
    return 0


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.retarget.scene")
    sys.exit(main())
