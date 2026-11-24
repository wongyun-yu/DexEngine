"""Placement-search ablation (paper Sec. E.5): which cell list the placement judge tries, behind
`dexengine.placement.judge.core --place-strategy`.

Default `canon` = the judge's own search (anchors → map-sorted exhaustion), untouched: the judge only calls into this
module when the flag is something else, so the canonical argv and its outputs stay byte-identical.
Frozen API (the judge imports it): STRATEGIES · check_args(a) · search(J, ck).

Every non-canon strategy only decides the **ordered cell list**. The cells are then judged by the canonical
exhaustion machinery (`dexengine.placement.judge.exhaust._Exhaustion` — sparse gate → FULL track → mesh guard →
final desk check → the same plans.jsonl / ikc / reject lines), so the downstream bake → retime → gate reads them
exactly like a canonical run. Anchors and refinement are skipped for all of them.

  fixed     one cell: the capture scene under one fixed transform — grid centre (dx = dy = 0 from the search origin),
            yaw 0, robot height = search_policy.TACO_CANON (the layer nearest it when the dataset has no such layer).
  random    RANDOM_N cells drawn uniformly (no replacement) from every (yaw, x, y, dz) cell that passes the desk
            footprint mask (the scene rule — objects on the desk, robot spot clear; not the reachability map),
            rng = (SEED, crc32(clip)). Same IK budget as one anchor of ours (max 12 checks).
  mapfirst  the first cell of the map-sorted exhaustion list (`_Exhaustion._candidates()[0]` — needs the volume,
            i.e. the IK-pool coverage map). No anchors.
  allcells  every desk-valid cell, without the map: ordered by the natural-pose distance (the same key ours uses
            after coverage), then |dz|. Cut at --place-strategy-cap cells (budget record 'cell' when cut) and at
            --place-strategy-budget-s seconds (0 = the canonical --clip-budget-s).
  randorder the same budget as ours, without screening and without an order rule: every desk-valid cell in one
            uniform random permutation, rng = (SEED, crc32(clip)), judged under the canonical stop rule untouched —
            --exhaust-cands accepted cells, or the first --exhaust-max cells of the list, or --clip-budget-s seconds.
            (`random` above is a fixed 12-cell draw; this one keeps drawing until ours' own budget ends.)
  randtime  the same permutation as randorder with the cell cap lifted: the stop is --exhaust-cands accepted cells,
            the end of the list, or --place-strategy-budget-s seconds (required, > 0). For the time-matched read —
            randorder spends its 1,024 cells in seconds, so its budget is not ours' GPU time.
  adopted   one cell: the cell production adopted for this clip, read from its placement record
            (`--place-strategy-adopted-dir`/<clip>_pl_rt.adopted.json — required, no default) and mapped back to the
            grid. A record that is missing or does not land on a grid cell gives an empty list (reject line + marker
            with the reason) — never a neighbouring cell. For re-running the arm solver alone at a fixed placement
            (`--track-ablate`); after the clip it prints `[TRACK_ABLATE_CLIP] <clip> …` when that flag is on.

Every list strategy also prints one line per accepted cell (the time-to-accept record of the row):
`[PLACE_STRATEGY_HIT] <clip> strategy=… k=<n-th accepted> t=<s since the search began> checked=… pruned=…`.
Marker per clip: `[PLACE_STRATEGY] <clip> strategy=… list=… judged=… checked=… pruned=… adopted=… wall=…s
grid=…`.
"""

from __future__ import annotations

import json
import time
import zlib
from pathlib import Path

import numpy as np

from dexengine.placement.judge.exhaust import _Exhaustion

TAG = "[PLACE_STRATEGY]"
HIT = "[PLACE_STRATEGY_HIT]"
STRATEGIES = ("canon", "fixed", "random", "mapfirst", "allcells", "randorder", "randtime", "adopted")
SEED = 42
RANDOM_N = 12


def check_args(a) -> None:
    """Die on inputs that would be silently ignored."""
    if (a.place_strategy == "adopted") != bool(a.place_strategy_adopted_dir):
        raise SystemExit(
            f"{TAG} ✗ --place-strategy-adopted-dir goes with --place-strategy adopted, and adopted "
            "needs it (the records are an identity input — no default)"
        )
    if a.place_strategy == "adopted" and not Path(a.place_strategy_adopted_dir).is_dir():
        raise SystemExit(f"{TAG} ✗ --place-strategy-adopted-dir {a.place_strategy_adopted_dir}: no such directory")
    if a.place_strategy == "canon":
        if a.place_strategy_cap != 4096 or a.place_strategy_budget_s != 0.0:
            raise SystemExit(
                f"{TAG} ✗ --place-strategy-cap/--place-strategy-budget-s given without --place-strategy "
                "(canon ignores them)"
            )
        return
    if a.place_strategy == "randtime":
        if a.place_strategy_cap != 4096 or a.place_strategy_budget_s <= 0:
            raise SystemExit(
                f"{TAG} ✗ randtime takes --place-strategy-budget-s > 0 and no --place-strategy-cap "
                "(time is its only budget)"
            )
    elif a.place_strategy != "allcells" and (a.place_strategy_cap != 4096 or a.place_strategy_budget_s != 0.0):
        raise SystemExit(
            f"{TAG} ✗ --place-strategy-cap/--place-strategy-budget-s belong to allcells only "
            f"(got strategy {a.place_strategy})"
        )
    if a.place_strategy_cap <= 0 or a.place_strategy_budget_s < 0:
        raise SystemExit(f"{TAG} ✗ cap must be > 0 and budget ≥ 0")
    if a.aug_cells_mode != "off":
        raise SystemExit(f"{TAG} ✗ --place-strategy {a.place_strategy} does not combine with the aug-cells modes")
    print(
        f"{TAG} strategy={a.place_strategy} cap={a.place_strategy_cap} budget_s={a.place_strategy_budget_s} "
        f"random_n={RANDOM_N} seed={SEED} (default canon = byte-identical)",
        flush=True,
    )


class _ListExhaustion(_Exhaustion):
    """The canonical exhaustion over a given cell list."""

    def __init__(self, J, ck, cands, t0=None):
        super().__init__(J, ck)
        self._given = cands
        self._t_search = time.time() if t0 is None else t0

    def _candidates(self):
        return self._given

    def _hit(self, n0):
        if self.n_found > n0:  # one line per accepted cell: when (since the search began) and after how many checks
            print(
                f"{HIT} {self.ck.clip} strategy={self.J.a.place_strategy} k={self.n_found} "
                f"t={time.time() - self._t_search:.1f}s checked={self.n_ex} pruned={self.n_pruned}",
                flush=True,
            )

    def _adopt_ex(self, *args, **kw):
        n0 = self.n_found
        r = super()._adopt_ex(*args, **kw)
        self._hit(n0)
        return r

    def _serial_adopt(self, *args, **kw):
        n0 = self.n_found
        r = super()._serial_adopt(*args, **kw)
        self._hit(n0)
        return r


def _tuple(J, ck, ky, kx, kyy, dzi):
    """Candidate tuple in `_Exhaustion._candidates` shape: (cvs, cv, ky, kx, kyy, dz, margin, nat) — map fields 0."""
    off = np.array([(kx - J.Wx) * J.step, (kyy - J.Wy) * J.step, -dzi * J.step])
    nat = max(float(np.linalg.norm(ck.cent_cache[s][ky] + off - J.W_NAT[s])) for s in ck.cent_cache)
    return (0, 0, int(ky), int(kx), int(kyy), dzi, 0.0, nat)


def _desk(J, ck):
    a = J.a
    m = getattr(ck, "desk_m", None)
    return np.ones((a.n_yaw, J.NBx, J.NBy), bool) if m is None else np.asarray(m, bool)


def _fixed(J, ck):
    from dexengine.placement import search_policy as DZP  # noqa: PLC0415

    canon = float(DZP.TACO_CANON)
    dzi = min(J.DZ, key=lambda d: (abs(float(d) - canon), abs(float(d))))
    ky = int(np.argmin(np.abs(J.yaws)))
    if abs(float(J.yaws[ky])) > 1e-9:
        raise SystemExit(f"{TAG} ✗ yaw grid has no 0° bin (nearest {J.yaws[ky]})")
    return [_tuple(J, ck, ky, J.Wx, J.Wy, dzi)], f"cell=(0,0,yaw0,dz{dzi})"


def _random(J, ck):
    m = _desk(J, ck)
    ky, kx, kyy = np.nonzero(m)
    n_layer = len(ky)
    n = n_layer * len(J.DZ)
    if n == 0:
        return [], "desk-valid 0"
    rng = np.random.default_rng([SEED, zlib.crc32(ck.clip.encode("utf-8"))])
    pick = rng.choice(n, size=min(RANDOM_N, n), replace=False)
    out = []
    for p in pick:
        li, ci = divmod(int(p), n_layer)
        out.append(_tuple(J, ck, ky[ci], kx[ci], kyy[ci], J.DZ[li]))
    return out, f"desk-valid {n}"


def _allcells(J, ck):
    m = _desk(J, ck)
    ky, kx, kyy = np.nonzero(m)
    cands = []
    for dzi in J.DZ:
        off = np.stack([(kx - J.Wx) * J.step, (kyy - J.Wy) * J.step, np.full(len(kx), -dzi * J.step)], 1)
        nat = np.max(
            np.stack([np.linalg.norm(ck.cent_cache[s][ky] + off - J.W_NAT[s], axis=1) for s in ck.cent_cache]), 0
        )
        cands += [(0, 0, int(ky[u]), int(kx[u]), int(kyy[u]), dzi, 0.0, float(nat[u])) for u in range(len(ky))]
    cands.sort(key=lambda t: (t[7], abs(t[5])))
    return cands, f"desk-valid {len(cands)}"


def _randorder(J, ck):
    cands, grid = _allcells(J, ck)  # every desk-valid cell (tuples in the canonical shape)
    if not cands:
        return cands, grid
    cands.sort(key=lambda t: (t[2], t[3], t[4], float(t[5])))  # a fixed base order, then one permutation
    rng = np.random.default_rng([SEED, zlib.crc32(ck.clip.encode("utf-8"))])
    return [cands[int(i)] for i in rng.permutation(len(cands))], grid


def _adopted(J, ck):
    """The one cell of the clip's production placement record, mapped back to (yaw bin, x cell, y cell, dz layer)."""
    f = Path(J.a.place_strategy_adopted_dir) / J._adopted_path(ck.clip).name
    if not f.is_file():
        return [], f"no-record:{f.name}"
    rec = json.loads(f.read_text(encoding="utf-8"))
    yaw = float(rec["yaw"])
    th = np.deg2rad(yaw)
    r2 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    cpiv = np.asarray(ck.cpiv, np.float64)
    # the record's (dx, dy) is about its own pivot (exhaust._Exhaustion._adopt_ex): d = dcap + (I − R)(cpiv − pivot)
    dcap = np.array([rec["dx"], rec["dy"]], np.float64) - (np.eye(2) - r2) @ (cpiv[:2] - np.asarray(rec["pivot"][:2]))
    b = J.rz.apply(np.array([dcap[0] - ck.d0cap[0], dcap[1] - ck.d0cap[1], 0.0]))
    fx, fy = b[0] / J.step + J.Wx, b[1] / J.step + J.Wy
    kx, kyy = int(round(fx)), int(round(fy))
    ky = int(np.argmin(np.abs(J.yaws - yaw)))
    dz = [d for d in J.DZ if abs(float(d) * J.step - float(rec["robot_z_off"])) < 1e-6]
    why = ""
    if max(abs(fx - kx), abs(fy - kyy)) > 0.02 or not (0 <= kx < J.NBx and 0 <= kyy < J.NBy):
        why = f"off-grid-xy:{fx:.3f},{fy:.3f}"
    elif abs(float(J.yaws[ky]) - yaw) > 1e-6:
        why = f"off-grid-yaw:{yaw}"
    elif len(dz) != 1:
        why = f"off-grid-dz:{rec['robot_z_off']}"
    elif np.abs(J._cell_dcap(ck, kx, kyy)[:2] - dcap).max() > 2e-3:
        why = "cell-roundtrip"
    elif not _desk(J, ck)[ky, kx, kyy]:
        why = "desk-mask"
    if why:
        return [], why
    return [_tuple(J, ck, ky, kx, kyy, dz[0])], f"record=(yaw{yaw:g},x{kx},y{kyy},dz{dz[0]})"


def search(J, ck) -> bool:
    """`Judge._search` for a non-canon strategy (masks already filled). Same return contract: True = run_clip ends
    without its closing (no volume / no spot)."""
    a = J.a
    s = a.place_strategy
    t0 = time.time()
    ck.got_any = False
    ck._v2_tried = True
    if s == "mapfirst":
        if not J._clip_volume(ck):  # writes the canonical 'no volume' / 'no spot' reject line
            print(
                f"{TAG} {ck.clip} strategy={s} list=0 judged=0 checked=0 pruned=0 adopted=0 "
                f"wall={time.time() - t0:.1f}s grid=no-volume",
                flush=True,
            )
            return True
        cands, grid = _Exhaustion(J, ck)._candidates()[:1], "map-sorted first"
    else:
        if getattr(ck, "place_none", None):
            J._clip_volume(ck)  # writes the canonical PLACE_NO_SPOT reject line, returns False
            print(
                f"{TAG} {ck.clip} strategy={s} list=0 judged=0 checked=0 pruned=0 adopted=0 "
                f"wall={time.time() - t0:.1f}s grid=no-spot",
                flush=True,
            )
            return True
        shape = (a.n_yaw, J.NBx, J.NBy)
        z = {d: np.zeros(shape, np.int32) for d in J.DZ}
        ck.V = {d: _desk(J, ck) for d in J.DZ}
        ck.COV, ck.COVS, ck.MGV = z, z, {d: np.zeros(shape, np.float32) for d in J.DZ}
        cands, grid = {
            "fixed": _fixed,
            "random": _random,
            "allcells": _allcells,
            "randorder": _randorder,
            "randtime": _randorder,
            "adopted": _adopted,
        }[s](J, ck)
    if not cands:
        # reject line of the judge's noconfirm record (the judge's record word: candidates)
        J.fno.write(f"{ck.clip}\tEXHAUSTED\t{TAG} {s}: candidates 0 ({grid})\n")
        J.n_no += 1
        print(
            f"{TAG} {ck.clip} strategy={s} list=0 judged=0 checked=0 pruned=0 adopted=0 "
            f"wall={time.time() - t0:.1f}s grid={grid}",
            flush=True,
        )
        return False
    keep = (a.exhaust_max, a.clip_budget_s)
    if s == "allcells":
        a.exhaust_max = a.place_strategy_cap
        if a.place_strategy_budget_s > 0:
            a.clip_budget_s = a.place_strategy_budget_s
    elif s == "randtime":  # no cell cap — the whole permutation, cut by time alone
        a.exhaust_max, a.clip_budget_s = len(cands), a.place_strategy_budget_s
    elif s != "randorder":  # randorder = ours' own budget: --exhaust-max and --clip-budget-s as given
        a.exhaust_max = max(a.exhaust_max, len(cands))
    if J._TFM_STAT is not None:
        J._TFM_STAT.update(dict.fromkeys(J._TFM_STAT, 0))
    try:
        ex = _ListExhaustion(J, ck, cands, t0)
        ex.run()
    finally:
        a.exhaust_max, a.clip_budget_s = keep
    if J._TFM_STAT is not None:  # how often the confirm's rescues ran for this clip (--track-ablate)
        print(
            f"[TRACK_ABLATE_CLIP] {ck.clip} off={','.join(sorted(J._ABL))} "
            + " ".join(f"{k}={v}" for k, v in J._TFM_STAT.items()),
            flush=True,
        )
    print(
        f"{TAG} {ck.clip} strategy={s} list={len(cands)} judged={len(ex._cl)} checked={ex.n_ex} "
        f"pruned={ex.n_pruned} adopted={ex.n_found} wall={time.time() - t0:.1f}s grid={grid}",
        flush=True,
    )
    return False
