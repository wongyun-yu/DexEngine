#!/usr/bin/env python3
"""Stretch (time-axis slowdown) computation — the one implementation the judge, the bake and the retiming share.

The function is shared, not only the thresholds: with the same formula, one extra step on one side (e.g. smoothing)
makes the results diverge — a single frame at r=31 spread over 9 neighbours raises the mean stretch from 1.4x to
3.79x, so a clip that passes the judge would fail the retiming cap.

s[i] = the factor by which interval i->i+1 must be lengthened so the joint speed stays within the limit. There is
no smoothing: only the fast frames are slowed by 1/k. Two joint groups — the arms (the caller's q, vendor URDF limits)
and the fingers (`floor`, the per-interval floor built by `finger_floor` from HandSpec's applied limits) — share one
factor per interval (uniform slowdown, the max of the two groups).
"""

from __future__ import annotations

import numpy as np

SAFETY = 1.15  # retiming safety factor (arms and fingers) — the default of the judge, the bake and the retiming
FRAME_DT = 0.02  # tree frame interval (s) — the one owner: every joint budget (limit x interval) and the resample
# grid (retime.warp.DT) read it


def frame_ratio(q, budget):
    """Per-frame speed ratio of a joint trajectory q (T, n): r_t = max_j |q_j[t+1] - q_j[t]| / budget_j, shape (T-1,).

    The one measurement definition of "the reference stays within the limits"; the stretch and the speed gate call
    it."""
    q = np.asarray(q, dtype=np.float64)
    return (np.abs(np.diff(q, axis=0)) / np.asarray(budget)[None, :]).max(axis=1)


def stretch_profile(q, budget, safety: float = SAFETY, floor=None):
    """Per-interval stretch factor s (T-1,) of a joint trajectory q (T, n); ``budget`` = per-joint one-step budget
    (rad).

    ``floor`` (T-1,) = the interval floor another joint group (the fingers) asks for: s = max(s, floor). A length
    mismatch raises — it means the joints and the tree fingers are on different time axes."""
    s = np.ones(1) if len(q) < 2 else np.maximum(1.0, frame_ratio(q, budget) * safety)
    if floor is None:
        return s
    floor = np.asarray(floor, dtype=np.float64)
    if floor.shape != s.shape:
        raise ValueError(
            f"[stretch] floor length {floor.shape} != stretch {s.shape} — the finger columns and the joint "
            "trajectory are on different time axes"
        )
    return np.maximum(s, floor)


def finger_floor(fq: dict, budget_f, safety: float = SAFETY):
    """Interval floor (T-1,) from finger columns {side: (T, nf)}: the max of stretch_profile over every hand block
    (idle side included). An empty dict raises — no floor of 1 is invented."""
    if not fq:
        raise ValueError("[stretch] no finger columns — no floor is invented")
    return np.max(np.stack([stretch_profile(f, budget_f, safety) for f in fq.values()]), axis=0)


def total_stretch(q, budget, safety: float = SAFETY, floor=None):
    """How many times longer the clip becomes (mean stretch) — the value the judge's and the bake's caps measure."""
    return float(np.mean(stretch_profile(q, budget, safety, floor)))


def raw_ratio_max(q, budget):
    """Largest speed ratio (diagnostic — shows the size of a valley jump)."""
    if len(q) < 2:
        return 0.0
    return float(frame_ratio(q, budget).max())


def weighted_p99(err, q, budget, safety: float = SAFETY, qtl: float = 0.99, floor=None):
    """Tracking-error quantile AFTER retiming, predicted by weighting each frame with its stretch factor.

    The retiming lengthens the fast intervals, and those are where the valley jumps and the large errors are, so the
    bad frames get duplicated and the distribution shifts (measured: a 307-frame trajectory with p99 = 27 mm becomes
    448 frames with p99 = 118 mm). Frame i is lengthened s[i] times (s includes the finger floor), so the s-weighted
    quantile is the post-retiming value. This is the "stretch-weighted p99 wrist error" the acceptance threshold
    (3 cm at the 99th percentile) is measured with — its maths is part of that definition.
    """
    err = np.asarray(err, dtype=np.float64)
    prof = stretch_profile(q, budget, safety, floor)
    w = np.concatenate([prof, prof[-1:]])[: len(err)]
    if len(w) < len(err):
        w = np.concatenate([w, np.full(len(err) - len(w), w[-1] if len(w) else 1.0)])
    o = np.argsort(err)
    cw = np.cumsum(w[o])
    if cw[-1] <= 0:
        return float(np.percentile(err, qtl * 100))
    return float(err[o][min(int(np.searchsorted(cw / cw[-1], qtl)), len(err) - 1)])
