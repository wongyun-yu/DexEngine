"""Stage 1 of one hand: progress table, per-step runs, rig identity, and new-hand registration (CLI).

The step table and its logic live in ``dexengine.data.stage1`` (callers import those names from there); this module is
the command line only.

    DEXENGINE_ARM=<arm> DEXENGINE_HAND=<hand> python3 dexengine/data/stages.py --status [--datasets hot3d,grab]
    ... --emit-run 2b --datasets oakink2        # argv lists of one step (JSON)
    ... --emit-rig --datasets hot3d              # rig identity of the outputs (JSON)
    ... --invalidate-stage-stamp --datasets hot3d
    ... --register --arm <arm> [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys

from dexengine.core import config
from dexengine.data import rules as RR
from dexengine.data.stage1.plan import emit_rig, emit_run, invalidate_stage_stamps
from dexengine.data.stage1.register import register
from dexengine.data.stage1.status import STATUS_ROWS, status
from dexengine.data.stage1.table import STAGES, driven_steps, split_steps, tree_unit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--status", action="store_true", help="per-step output counts and the first empty step")
    ap.add_argument("--json", action="store_true", help="with --status: also print the table as JSON")
    ap.add_argument("--register", action="store_true", help="run the new-hand registration chain")
    ap.add_argument("--arm", default=None, help="--register: the arm (required, no default)")
    ap.add_argument("--dry-run", action="store_true", help="--register: print the commands only")
    ap.add_argument("--datasets", default="", help="comma list (default: the campaign scope, rules.datasets_scope)")
    ap.add_argument("--emit-run", default="", metavar="STEP", help="print the argv lists of one step as JSON")
    ap.add_argument("--emit-rig", action="store_true", help="print the rig identity of the dataset's outputs as JSON")
    ap.add_argument(
        "--driven-steps", action="store_true", help="print the ids of the steps --emit-run drives, in run order"
    )
    ap.add_argument("--piece-steps", action="store_true", help="print the steps new pieces rerun (table.split_steps)")
    ap.add_argument(
        "--invalidate-stage-stamp",
        action="store_true",
        help="before retargeting writes into the stage: remove stage stamps that differ from the rig/prior",
    )
    ap.add_argument("--f0-force", action="store_true", help="--emit-run 7: append --force to the f0 argv")
    ap.add_argument(
        "--reuse",
        action="store_true",
        help="--emit-run: send the step's per-tree reuse command when it has one (same-identity rerun)",
    )
    config.add_data_root_arg(ap)
    a = ap.parse_args()
    if a.driven_steps or a.piece_steps:  # the driver's step lists, table order (no second copy in the shell)
        print(" ".join(driven_steps() if a.driven_steps else split_steps()["run"]))
        return 0
    config.apply_data_root(a)
    try:
        ds = list(RR.datasets_scope(a.datasets))
    except KeyError as e:
        print(f"[hand_pipeline] {e}", file=sys.stderr)
        return 3
    if a.reuse and not a.emit_run:
        print("[hand_pipeline] --reuse only goes with --emit-run", file=sys.stderr)
        return 3
    if a.invalidate_stage_stamp:
        if len(ds) != 1:
            print("--invalidate-stage-stamp takes exactly one dataset (--datasets)", file=sys.stderr)
            return 3
        print(invalidate_stage_stamps(ds[0]))
        return 0
    if a.emit_run or a.emit_rig:
        if len(ds) != 1:
            print(f"--emit-run/--emit-rig take exactly one dataset (--datasets; got {len(ds)})", file=sys.stderr)
            return 3
        out = emit_run(a.emit_run, ds[0], f0_force=a.f0_force, reuse=a.reuse) if a.emit_run else emit_rig(ds[0])
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return 0
    if a.register:
        if not a.arm:
            ap.error("--register needs --arm <arm> (no default)")
        rc = register(RR.ACTIVE_HAND, a.arm, a.dry_run)
        if rc or a.dry_run:
            return rc
    rc = status(ds)
    if a.json:
        print(
            json.dumps(
                {
                    "hand": RR.ACTIVE_HAND,
                    "combo": RR.combo(),
                    "stages": [g[0] for g in STAGES],
                    "rows": STATUS_ROWS,
                    "tree_unit_stages": [g[0].split()[0] for g in STAGES if tree_unit(g[1], g[3])],
                },
                ensure_ascii=False,
                indent=1,
            )
        )
    return rc


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.data.stages")
    raise SystemExit(main())
