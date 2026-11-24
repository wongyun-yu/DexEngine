"""Coarse path of one arm: per-frame candidates from the reach-map valleys and the DP that picks one per frame.

The DP keeps one IK branch over the whole clip: it picks the sequence of covering solutions with the least joint
motion (per-joint step relative to the velocity budget, steps over the threshold penalised), and linear
interpolation between the picks gives a configuration for every frame. The bake stage (`coarse_path.dp`) polishes
and checks the result.
"""

from __future__ import annotations

import numpy as np

#: DP step cost: a per-joint step over `budget_mult` × the velocity budget costs this many times its size.
JUMP_PENALTY = 50.0
#: DP bonus per unit of joint-limit margin (normalised), subtracted from the cost of a candidate.
MARGIN_BONUS = 1e-3


def collect(a, T, K, VQ, VOK, gi, nx, ny, nz, aiN, _pin):
    """Per-frame candidates → (flattened candidates, per-frame (start, count)).

    Candidates = valid valleys (VOK) of the 3×3×3 cells around the target cell gi[f] for the frame's top atoms
    aiN[f], deduplicated by max joint distance > `a.dedup_deg`. With `_pin` (the judge trajectory) only candidates
    within `a.dp_pin_rad` of it are kept and the pinned pose itself is the first candidate."""
    THR_D = np.deg2rad(a.dedup_deg)
    cand_q, cand_of = [], []  # flattened candidates / per frame (start, count)
    for f in range(T):
        qs = [] if _pin is None else [_pin[f].copy()]
        x0, y0, z0 = gi[f]
        for dx in (0, -1, 1):
            for dy in (0, -1, 1):
                for dz in (0, -1, 1):
                    x, y, zc = x0 + dx, y0 + dy, z0 + dz
                    if not (0 <= x < nx and 0 <= y < ny and 0 <= zc < nz):
                        continue
                    for at in aiN[f]:
                        for kk in range(K):
                            if VOK[x, y, zc, at, kk]:
                                q = VQ[x, y, zc, at, kk].astype(np.float64)
                                if _pin is not None and np.abs(q - _pin[f]).max() > a.dp_pin_rad:
                                    continue
                                if all(np.abs(q - q2).max() > THR_D for q2 in qs):
                                    qs.append(q)
        cand_of.append((len(cand_q), len(qs)))
        cand_q.extend(qs)
    return cand_q, cand_of


def dp_path(qp, ok, mgc, cand_of, T, BUDGET, budget_mult, n):
    """The coarse path q_dp (T, n), or None when no frame has a valid candidate.

    qp / ok / mgc = polished candidate joints, their validity and margin (flattened as in `collect`); BUDGET =
    per-joint velocity budget per frame. Cost between neighbouring picked frames = max_j |Δq_j| / BUDGET_j
    (× JUMP_PENALTY above `budget_mult`) − MARGIN_BONUS × margin. Frames without a valid candidate are skipped and
    filled by linear interpolation of the neighbouring picks (the end frames copy the nearest pick)."""
    prev_ids = None
    prev_cost = None
    choice = {}
    for f2 in range(T):
        s0, c0 = cand_of[f2]
        ids = [s0 + jj for jj in range(c0) if ok[s0 + jj]]
        if not ids:
            continue
        qs = qp[ids]
        if prev_ids is None:
            cost = -MARGIN_BONUS * mgc[ids]
            back = [-1] * len(ids)
        else:
            dq = (np.abs(qs[:, None, :] - qp[prev_ids][None, :, :]) / BUDGET).max(-1)
            pen = np.where(dq > budget_mult, dq * JUMP_PENALTY, dq)
            tot = pen + prev_cost[None, :]
            barg = tot.argmin(1)
            cost = tot[np.arange(len(ids)), barg] - MARGIN_BONUS * mgc[ids]
            back = barg.tolist()
        choice[f2] = (ids, back)
        prev_ids, prev_cost = ids, cost
        last_f = f2
    if not choice:
        return None
    # Backtrack from the minimum cost at the final frame
    ids, back = choice[last_f]
    j = int(np.argmin(prev_cost))
    path = {}
    f2 = last_f
    while True:
        ids, back = choice[f2]
        path[f2] = ids[j]
        prevs = [g for g in choice if g < f2]
        if back[j] < 0 or not prevs:
            break
        j = back[j]
        f2 = max(prevs)
    # Missing frames get interpolation of the neighbouring picks
    q_dp = np.zeros((T, n))
    known = sorted(path)
    for f2 in range(T):
        if f2 in path:
            q_dp[f2] = qp[path[f2]]
        else:
            lo_ = max([g for g in known if g < f2], default=None)
            hi_ = min([g for g in known if g > f2], default=None)
            if lo_ is None:
                q_dp[f2] = qp[path[hi_]]
            elif hi_ is None:
                q_dp[f2] = qp[path[lo_]]
            else:
                w = (f2 - lo_) / (hi_ - lo_)
                q_dp[f2] = (1 - w) * qp[path[lo_]] + w * qp[path[hi_]]
    return q_dp
