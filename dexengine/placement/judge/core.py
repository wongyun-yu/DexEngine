#!/usr/bin/env python3
"""Placement-search judge (paper Sec. 3.3): find a robot placement under which both arms can follow the clip.

For every clip the judge
  1. screens the placement grid around the clip pivot with the reachability map (a cell is a candidate when at least
     `--vol-frac` of the frames have a pose in the map),
  2. walks candidate cells from the anchors and, per cell, builds a coarse whole-clip arm path (DP over the map's pose
     candidates, the gate), then confirms it with per-frame IK from the coarse path plus random starts (FULL track),
  3. accepts a placement when the stretch-weighted p99 wrist error is at most `--dp-err-mm`, the retimed clip is at
     most `--dp-max-stretch` times longer, no joint jumps more than `--jump-max`, the elbow swivel stays below
     `--swivel-max-deg` and the final collision check passes,
  4. searches the whole volume exhaustively when every anchor path fails (a failure is then a certificate within the
     grid and thresholds).

Outputs (in --out): plans2_<combo>.jsonl (accepted placements), noconfirm_<combo>.tsv (clips whose candidates were all
rejected), the per-clip input record and budget rows; `--end-marks` gets one `<clip>\t<s>` line per finished clip.

Module layout: args (command line) · config (search conditions, configs/placement/judge.yaml) · common (shared
helpers) · boot · gate · confirm · accept · track · clip · search (the Judge mixins) · exhaust (exhaustive search).
Run as `python -m dexengine.placement.judge.core`.
"""

from __future__ import annotations

import sys
import time

from dexengine.placement.judge import clip_time as _CT  # [CLIP_T] per-clip stage seconds
from dexengine.placement.judge import config as _CFG
from dexengine.placement.judge import input_record as _JFP
from dexengine.placement.judge.accept import AcceptMixin
from dexengine.placement.judge.anchor import AnchorMixin
from dexengine.placement.judge.args import _build_parser, _resolve_f0_roots, config_path
from dexengine.placement.judge.boot import BootMixin
from dexengine.placement.judge.boot_pools import PoolsMixin
from dexengine.placement.judge.clip import ClipMixin
from dexengine.placement.judge.clip_inputs import ClipInputsMixin
from dexengine.placement.judge.common import SWIVEL_MAX_DEG  # noqa: F401 — public name
from dexengine.placement.judge.confirm import ConfirmMixin
from dexengine.placement.judge.gate import GateDPMixin
from dexengine.placement.judge.search import SearchMixin
from dexengine.placement.judge.track import TrackMixin
from dexengine.robots import urdf_load


class Judge(
    BootMixin,
    PoolsMixin,
    GateDPMixin,
    ConfirmMixin,
    AcceptMixin,
    TrackMixin,
    ClipMixin,
    ClipInputsMixin,
    SearchMixin,
    AnchorMixin,
):
    """One judge run: boot once (pools, IK, thresholds, output files), then run_clip per clip.

    Methods read these attributes at call time. Re-bound per clip: sides (active subset) · Z_OFF · DXb/DYb ·
    DESK_XY · clip · a.exhaust_max (x2 for one-arm clips); changed in place: AS_CUR · REJ · FULL · _BEST.
    """


def prove_inputs(J) -> int:
    """Measure every clip's input record strictly before the first verdict: a clip whose inputs cannot be measured
    stops the run (no verdict is written for an unproven input)."""
    a = J.a
    for clip in J.clips:
        _JFP.clip_parts(a.dataset, a.emb, a.hand, clip, J._SCENE, a.active_sides, strict=True)
    print(f"[JUDGE_ENTRY] input proof {len(J.clips)} clips", flush=True)
    return len(J.clips)


def main(argv: list[str] | None = None) -> int:
    """Judge every clip of the list; returns the process exit code."""
    argv = sys.argv[1:] if argv is None else argv
    a = _build_parser(_CFG.load(config_path(argv))).parse_args(argv)
    urdf_load.install()
    _F0ROOT, TRAJ_ROOT = _resolve_f0_roots(a)
    J = Judge(a, _F0ROOT, TRAJ_ROOT)
    prove_inputs(J)
    # End marks are written only here, after run_clip returns (every early return of run_clip comes back here), line
    # buffered so a finished clip is on disk even if the next clip dies.
    marks = open(a.end_marks, "a", buffering=1) if a.end_marks else None  # noqa: SIM115 — lives for the loop
    if marks is not None:
        print(f"[JUDGE_END_MARKS] {a.end_marks} n={len(J.clips)}", flush=True)
    for ci, clip in enumerate(J.clips):
        t0, ok0 = time.time(), J.n_ok
        _CT.start()
        J.run_clip(ci, clip)
        dt = time.time() - t0
        if marks is not None:
            marks.write(f"{clip}\t{dt:.1f}\n")
        J.budget_row(clip)
        J.clip_release()
        print(_CT.line(clip, dt, J.n_ok > ok0), flush=True)
    J.finish()
    if marks is not None:
        marks.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
