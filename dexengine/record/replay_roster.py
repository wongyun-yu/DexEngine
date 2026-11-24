#!/usr/bin/env python3
"""Trees the kinematic replay render (RENDER_ONLY=replay) runs: the launcher (dexengine.rl.launch.modes) decides them
through this entry when it gets DEXENGINE_SEQ_LIST.

No new rule: the sequence folder identity (paths.seq_dir_identity), the combo split (paths.split_combo) and the list
resolution (train_ready_util.resolve_roster: the same call and env inputs as the --sequence_dir branch of the training
entry dexengine/rl/cli.py) are called as they are. The loop runs only the trees this prints; the old loop ran every
folder in SEQ_DIR and silently dropped the list (record stage 2 wrote the dump of all 419 clips of a bundle to draw one
clip).

Usage  python3 dexengine/record/replay_roster.py --seq-dir <SEQ_DIR> --seq-list <list file | all>
env    DEXENGINE_DATASET · DEXENGINE_EMBODIMENT (required, no default) · DEXENGINE_ROSTER (a named source, optional)
Output stdout = one tree name per line · stderr = the resolve_roster marker line.
       A list entry missing from the folder or outside the list source: rc 2.
"""

from __future__ import annotations

import argparse
import os
import sys

from dexengine.core import paths as P
from dexengine.placement import train_ready_util as TRU


def trees(seq_dir: str, seq_list: str) -> tuple[list[str], str]:
    sd = os.path.abspath(seq_dir)
    have = sorted(d for d in os.listdir(sd) if os.path.isfile(os.path.join(sd, d, "0", "trajectory_kinematic.npz")))
    if not have:
        raise ValueError(f"no tree in the sequence folder (<tree>/0/trajectory_kinematic.npz): {sd}")
    ds, emb = os.environ.get("DEXENGINE_DATASET", "").strip(), os.environ.get("DEXENGINE_EMBODIMENT", "").strip()
    if not ds or not emb:
        raise ValueError(
            "DEXENGINE_DATASET and DEXENGINE_EMBODIMENT are required (list = folded ledger axis; no default)"
        )
    combo, is_view = P.seq_dir_identity(sd)
    _arm, hand, var = P.split_combo(combo, P.ARM_IDS, P.HAND_IDS)
    r = TRU.resolve_roster(
        combo=combo,
        hand=hand,
        dataset=ds,
        emb=emb,
        variant=var,
        trees=have,
        is_view=is_view,
        seq_dir=sd,
        seq_list=seq_list,
        source=os.environ.get("DEXENGINE_ROSTER", "").strip(),
        root=P.main_root(),
    )
    return list(r["train"]), r["marker"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seq-dir", required=True)
    ap.add_argument("--seq-list", required=True, help="list file or all (explicit opt-out)")
    a = ap.parse_args(argv)
    try:
        got, marker = trees(a.seq_dir, a.seq_list)
    except ValueError as e:
        print(f"[REPLAY_LIST] {e}", file=sys.stderr)
        return 2
    print(marker, file=sys.stderr)
    sys.stdout.write("".join(t + "\n" for t in got))
    return 0


if __name__ == "__main__":
    sys.exit(main())
