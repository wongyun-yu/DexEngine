#!/usr/bin/env python3
"""Build the training robot USD of merged combos (URDF → USD → flatten → export → contact reporters → colour check).

Usage:
  python scripts/build_robot_usd.py [--combos <dir>[,<dir>…] | --arm <arm> --hand <hand>] [--dry-run]
      [--log-dir DIR] [--data-root DIR]
  --combos    combo asset folders (`<arm>_<hand><generation suffix>[<variant>]`, e.g. rby1_sharpa_wave or
              rby1_dg5f_s_truncated); default every registered combo's canonical folder
              (`python -m dexengine.core.paths --what combo-dirs`)
  --arm/--hand  the canonical folders of that arm and/or hand instead
  --dry-run   print the plan (inputs with sha256, mesh resolution, outputs, the step commands) and check the inputs;
              starts no Isaac process and writes nothing. rc 0 = inputs complete, 2 = a problem (listed)
In:   assets/robots/full_robot_gen/<dir>/<dir>.urdf (python dexengine/robots/urdf/merge.py <arm>+<hand>), its meshes,
      configs/robots/urdf_converter.yaml
Out:  <data root>/data/usd_cache/robots/full_robot_gen/<dir>/<dir>.usd (+ the *_test.usd intermediates and
      bake_record.json, build.BAKE_RECORD); for a canonical folder this is paths.combo_usd_relpath. Logs per step in
      --log-dir (default /tmp/combo_usd).
Done: COMBO_USD_DONE line with complete counts incl. bake records (dexengine.robots.usd.build); rc 0 = done, 1 = a
      count is short, 2 = bad input. Needs the Isaac Lab environment (scripts/_isaac_env.sh finds it; ISAACLAB_DIR to
      override) and a GPU for steps 1 and 4.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this tree's package (an entry script, not a module)

from dexengine.core import config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog=__doc__.split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--combos", default="", help="comma-separated combo asset folders")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--hand", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log-dir", type=Path, default=Path("/tmp/combo_usd"))
    config.add_data_root_arg(ap)
    a = ap.parse_args(argv)
    config.apply_data_root(a)

    from dexengine.core import paths as P  # noqa: PLC0415 — after the data root is set
    from dexengine.robots.usd import build as B  # noqa: PLC0415

    if a.combos and (a.arm or a.hand):
        ap.error("give --combos or --arm/--hand, not both")
    try:
        dirs = [d for d in a.combos.split(",") if d] if a.combos else B.default_dirs(a.arm, a.hand)
        plans = B.plan(dirs, P.main_root())
    except ValueError as e:
        print(f"[robot_usd] {e}", file=sys.stderr)
        return 2
    lines, bad = B.check_inputs(plans)
    print("\n".join(lines), flush=True)
    if a.dry_run:
        for st, d, cmd in B.commands(plans):
            print(f"[ROBOT_USD_CMD] {st:9} {d}: isaaclab.sh -p {' '.join(cmd)}")
    for b in bad:
        print(f"[robot_usd] ✗ {b}", file=sys.stderr)
    if bad:
        return 2
    if a.dry_run:
        print(f"[robot_usd] dry run: {len(plans)} combos, inputs complete — nothing started, nothing written")
        return 0
    return B.run(plans, a.log_dir, P.main_root())


if __name__ == "__main__":
    sys.exit(main())
