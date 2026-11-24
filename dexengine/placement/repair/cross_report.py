"""After-state fields of the cross-repair report (written by cross_steps.report_after)."""

from __future__ import annotations

import time

import numpy as np

from dexengine.placement.collision import cross_sets as CC
from dexengine.placement.repair.cross_util import SET_REPAIR, _set_stat


def after_fields(
    report,
    after,
    after_h,
    a,
    residual,
    unresolved,
    need,
    hand_min_after,
    touched,
    hand_min,
    n_guard,
    n_guard8,
    wr,
    fd,
    ad,
    t0,
):
    """The after-state fields of the report: status, penetration after, achieved moves, residual frames and guard
    counts."""
    report.update(
        after={s: _set_stat(after[s]) for s in CC.SETS},
        after_hand={s: _set_stat(after_h[s]) for s in SET_REPAIR},
        applied=not a.dry_run,
        unresolved_frames=residual,
        unresolved_opt_frames=unresolved,
        spawn_repaired=bool(need[0]),
        spawn_after_mm=round(float(hand_min_after[0] * 1000), 2),
        status="FAILED"
        if residual
        else ("REPAIRED" if touched.any() else ("CLEAN" if not (hand_min < 0).any() else "SKIPPED_SHALLOW")),
        stats={
            "frames_touched": int(touched.sum()),
            "frames_guard_scaled": n_guard,
            "alpha_rejected_by_capsule_guard": n_guard8,
            "wrist_mm": {
                k: {
                    "p50": round(float(np.median(wr[k][touched])), 1) if touched.any() else 0.0,
                    "max": round(float(wr[k].max()), 1),
                }
                for k in wr
            },
            "finger_deg": {
                k: {
                    "p50": round(float(np.median(fd[k][touched])), 1) if touched.any() else 0.0,
                    "max": round(float(fd[k].max()), 1),
                }
                for k in fd
            },
            "arm_deg": {
                k: {
                    "p50": round(float(np.median(ad[k][touched])), 2) if touched.any() else 0.0,
                    "max": round(float(ad[k].max()), 2),
                }
                for k in ad
            },
        },
        sec=round(time.time() - t0, 1),
    )
