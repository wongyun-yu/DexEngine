"""CLI of the raw re-read (driver ``stage``; see ``dexengine.data.read_raw.stage``).

python -m dexengine.data.read_raw <dataset> read [--dry] [--root <generation root> [--recordings <list>]]
python -m dexengine.data.read_raw <dataset> intake <first hand> [--dry] [--root ... [--recordings ...]]
python -m dexengine.data.read_raw <dataset> derive [--dry]   (any driver: the row's INTAKE derivations, read_raw.derive)
"""

from __future__ import annotations

import argparse
import sys

from dexengine.core import config
from dexengine.data.read_raw import derive, stage


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dexengine.data.read_raw", description=__doc__.splitlines()[0])
    ap.add_argument("dataset")
    ap.add_argument("step", choices=("read", "intake", "derive"))
    ap.add_argument("hand", nargs="?", default="", help="intake: the first hand to retarget (no default)")
    ap.add_argument("--dry", action="store_true", help="print the decisions, write nothing (rc 3 = would refuse)")
    ap.add_argument("--root", default="", help="a generation root to run against (scratch subsets only)")
    ap.add_argument("--recordings", default="", help="with --root: list file of the recordings to read")
    config.add_data_root_arg(ap)
    a = ap.parse_args(argv)
    config.apply_data_root(a)
    tag = f"[REREAD] ds={a.dataset}"
    if a.step == "derive":  # before Run: Run takes only rows of the stage driver
        if a.hand or a.root or a.recordings:
            stage.die(tag, "derive takes no hand, --root or --recordings (--data-root names another data root)")
        return derive.main(a.dataset, a.dry, tag, lambda m: stage.die(tag, m))
    if a.step == "read" and a.hand:
        stage.die(tag, f"unknown argument {a.hand} (<dataset> read [--dry] | <dataset> intake <first hand> [--dry])")
    if a.step == "intake" and not a.hand:
        stage.die(tag, "intake needs the first hand (no default)")
    run = stage.Run(a.dataset, a.dry, a.root, a.recordings)
    run.header(a.step)
    return stage.read(run) if a.step == "read" else stage.intake(run, a.hand)


if __name__ == "__main__":
    sys.exit(main())
