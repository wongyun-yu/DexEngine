"""Module-level helpers shared by the judge mixins: frame dedup, plan writers, placement keys."""

from __future__ import annotations

import json

import numpy as np
import torch

from dexengine.placement.judge import config as _CFG
from dexengine.placement.reach_map import batch_ik as OPG

DEV, DT = OPG.DEV, OPG.DT

#: Largest elbow swivel change (the angle the elbow turns about the shoulder-wrist axis) inside a 0.5 s window, in
#: degrees — 180°/0.5 s = 360°/s, a speed a human elbow cannot produce. The judge uses --swivel-max-deg (same default);
#: this constant is the default for other readers.
SWIVEL_MAX_DEG = _CFG.load().swivel_max_deg


# Return values of the dp_confirm stages: None = go to the next stage · _NEXT_SIDE = this side is done (the stage
# already wrote met[side]) · (False, {side: reason}) = reject this placement.
_NEXT_SIDE = "next_side"
DD12 = np.deg2rad(12)  # the gate's 12° Chebyshev dedup radius — single owner (stage1 + side batch)


def _dedup_frames_t(_allq, _allclr, _bnd, T2, DD):
    """Per-frame greedy dedup within DD (Chebyshev, rad), all frames at once on DEV: pad to (T2, Nmax, nj); each round
    picks the first surviving index per frame and drops everything within DD of it. abs/sub/max/compare only, so the
    picks equal the per-frame greedy loop, in the same order. T2 is any number of frames — the
    gate feeds it the frames of **all cells of a side at once** (the round loop is ≤ Nmax rounds of
    ~10 launches, so running it per cell would cost 512× per clip); a row is independent of the
    other rows, so batching cells changes nothing in the picks."""
    cnt = np.diff(_bnd)
    Nmax = int(cnt.max()) if T2 else 0
    if Nmax == 0:
        return (np.zeros((0,) + _allq.shape[1:], _allq.dtype), [(0, 0) for _ in range(T2)], np.zeros(0, _allclr.dtype))
    nj = _allq.shape[1]
    rows = np.repeat(np.arange(T2), cnt)
    cols = np.arange(len(_allq)) - np.repeat(_bnd[:-1], cnt)
    Qp = torch.zeros((T2, Nmax, nj), dtype=torch.float64, device=DEV)
    keep = torch.zeros((T2, Nmax), dtype=torch.bool, device=DEV)
    _r = torch.tensor(rows, dtype=torch.long, device=DEV)
    _c = torch.tensor(cols, dtype=torch.long, device=DEV)
    Qp[_r, _c] = torch.tensor(_allq, dtype=torch.float64, device=DEV)
    keep[_r, _c] = True
    ar = torch.arange(Nmax, device=DEV)
    arT = torch.arange(T2, device=DEV)
    picks = []
    while True:
        first = torch.where(keep, ar, Nmax).amin(1)  # (T2,) Nmax = frame exhausted
        alive = first < Nmax
        if not bool(alive.any()):
            break
        fc = first.clamp(max=Nmax - 1)
        qi = Qp[arT, fc]  # (T2, nj)
        keep &= (Qp - qi[:, None, :]).abs().amax(-1) > DD
        picks.append(torch.where(alive, first, -1))
    P = torch.stack(picks, 1).cpu().numpy()  # (T2, K); -1 once exhausted (monotone)
    # frame-major, pick-order gather (== a per-frame extend loop, without the T2 Python steps)
    valid = P >= 0
    kcnt = valid.sum(1)
    gi = (_bnd[:-1, None] + P)[valid]
    starts = np.concatenate([[0], np.cumsum(kcnt)[:-1]])
    off = list(zip(starts.tolist(), kcnt.tolist(), strict=True))
    return _allq[gi], off, _allclr[gi]


def _dedup_side_batch(pres, DD):
    """One `_dedup_frames_t` call for every cell of a side: concatenate the cells' frames (each cell =
    T2 frames) and split the picks back per cell. pres = [(allq, allclr, bnd, T2)].
    Returns [(cand (n,nj), off [(s,c)]*T2, clrs (n,))] in the same order."""
    if not pres:
        return []
    allq = np.concatenate([p[0] for p in pres])
    allclr = np.concatenate([p[1] for p in pres])
    bnd = [np.zeros(1, np.int64)]
    base = 0
    for _q, _c, b, _T in pres:
        bnd.append(b[1:] + base)
        base += int(b[-1])
    bnd = np.concatenate(bnd)
    Ftot = sum(p[3] for p in pres)
    cand, off, clrs = _dedup_frames_t(allq, allclr, bnd, Ftot, DD)
    out = []
    f0 = 0
    for _q, _c, _b, T2 in pres:
        oc = off[f0 : f0 + T2]
        r0 = oc[0][0] if oc else 0
        r1 = (oc[-1][0] + oc[-1][1]) if oc else 0
        out.append((cand[r0:r1], [(s - r0, c) for s, c in oc], clrs[r0:r1]))
        f0 += T2
    return out


#: Marker of the re-search that keeps unheld static objects on the desk (one line per clip, only when it moved them).
ONDESK_TAG = "[STATIC_ON_DESK]"


class _ClipBuf:
    """Holds one clip's output lines (plans · rejects); only the chosen search's lines are replayed, in order and
    unchanged, to the real writer. Used only for clips with unheld static objects (Judge._search_on_desk): when the
    first search keeps them on the desk its lines are replayed as they are (same bytes)."""

    def __init__(self):
        self.lines: list[str] = []

    def write(self, line: str):
        self.lines.append(line)
        return len(line)

    def flush(self):
        return None

    def replay(self, fh) -> None:
        for ln in self.lines:
            fh.write(ln)


class _PlanWriter:
    """The one writer of plan lines (plans2_<combo>.jsonl): adds the effective exhaust_max, the reach map ids the judge
    verified ({side: store.pool_id} — the bake carries them into the tree's task_info provenance) and, with
    --active-sides auto, the active / idle sides at write time (both change per clip)."""

    def __init__(self, fh, a, as_cur, pool_ids: dict):
        self._fh = fh
        self._a = a  # the judge's argparse namespace — a.exhaust_max changes per clip (x2 one-arm)
        self._as_cur = as_cur  # Judge.AS_CUR (changed in place) — this clip's active / idle sides
        self._pool = dict(pool_ids)  # Judge.POOL_IDS — fixed for the run (booted before the outputs)

    def write(self, line: str):
        d = json.loads(line)
        d["exhaust_max"] = int(self._a.exhaust_max)
        d["pool"] = dict(self._pool)
        if self._a.active_sides == "auto":
            d["active_sides"] = list(self._as_cur["active"])
            d["idle_sides"] = list(self._as_cur["idle"])
        return self._fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    def flush(self):
        return self._fh.flush()

    def close(self):
        return self._fh.close()


def _cand_place(it_: dict) -> tuple:
    """A batched candidate's placement (capture dx, dy, yaw_deg, pivot) — the guard's support placement."""
    return (float(it_["dcap"][0]), float(it_["dcap"][1]), float(it_["yaw"]), it_["piv"])


def _place_key(place) -> tuple:
    """A placement's pivot-free form (yaw, tx, ty) rounded to 1e-9: q = Rz(p − c) + c + d = Rz·p + (d + c − Rz·c)."""
    dx, dy, yaw, piv = place
    th = np.deg2rad(float(yaw))
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    c = np.asarray(piv, float)[:2]
    t = np.array([float(dx), float(dy)]) + c - R @ c
    return (round(float(yaw), 9), round(float(t[0]), 9), round(float(t[1]), 9))


def _same_guard_key(k1: tuple, k2: tuple) -> bool:
    """Equality of two Judge._guard_key keys — z and the placement by value, the q arrays and the clip inputs by
    identity."""
    return (
        k1[0] == k2[0]
        and k1[-1] == k2[-1]
        and len(k1[1]) == len(k2[1])
        and all(a[0] == b[0] and a[1] is b[1] for a, b in zip(k1[1], k2[1], strict=True))
        and all(x is y for x, y in zip(k1[2:-1], k2[2:-1], strict=True))
    )
