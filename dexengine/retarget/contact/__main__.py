"""Fingertip contact detector CLI (`python -m dexengine.retarget.contact`).

Callers pass: --dataset-dir --dataset-name --embodiment-type --task --data-id 0 --ref-dt 0.02 --no-show-viewer.
`--ref-dt` is accepted and unused headless. Unknown flags fail.
"""

from __future__ import annotations

import argparse
import sys


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m dexengine.retarget.contact", description=__doc__, allow_abbrev=False)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--embodiment-type", required=True, choices=("bimanual", "right", "left"))
    ap.add_argument("--task", required=True)
    ap.add_argument("--data-id", type=int, default=0)
    ap.add_argument("--ref-dt", type=float, default=0.02)
    ap.add_argument("--show-viewer", action=argparse.BooleanOptionalAction, default=False)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.show_viewer:
        raise NotImplementedError("the contact viewer is not supported; run with --no-show-viewer")
    from dexengine.retarget.contact.detect import detect  # noqa: PLC0415

    detect(args.dataset_dir, args.dataset_name, args.embodiment_type, args.task, args.data_id)
    return 0


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.retarget.contact")
    sys.exit(main())
