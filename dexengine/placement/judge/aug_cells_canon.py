"""Augmentation cells, cell geometry and the canonical-cell exemption (base class of judge.aug_cells.AugCells): cell →
(yaw, dcap, d, p2), the canonical cell grid, the placement and desk checks of a cell, and the exemption that keeps
the canonical cell in an empty volume. Owner of the [AUG_CELLS] marker tag.
"""

from __future__ import annotations

import numpy as np

TAG = "[AUG_CELLS]"
#: Same-function tolerance between the region dump (mode D) and the θ · d mode L computes for the same cell, degrees ·
#: metres. The same formula on two machines can differ in the last bit (1 ulp, 2.7e-20 m measured on XHand ext cells),
#: which an exact float64 comparison reads as a different function.
ULP_T, ULP_D = 1e-9, 1e-12


def dump_mismatch(clip: str, cells, geo, by: dict) -> list:
    """Listed cells whose θ · d differ from the region dump `by` ({cell: (θ, dx, dy)}) beyond ULP_T · ULP_D. When none
    does but some differ in the last bits, prints [DUMP_ULP] (the judge rows then carry the dump values)."""
    dif = [
        (c, abs(by[c][0] - g[0]), max(abs(by[c][1] - float(g[2][0])), abs(by[c][2] - float(g[2][1]))))
        for c, g in zip(cells, geo, strict=True)
        if c in by
    ]
    bad = [c for c, dt, dd in dif if dt > ULP_T or dd > ULP_D]
    if not bad and (ulp := [x for x in dif if x[1] or x[2]]):
        print(
            f"[DUMP_ULP] {clip}: cells that differ from the dump only in the last bits {len(ulp)} · max Δθ "
            f"{max(x[1] for x in ulp):.1e} · max Δd {max(x[2] for x in ulp):.1e} m — judge rows carry the dump values",
            flush=True,
        )
    return bad


class AugCellsCanon:
    """Cell geometry (`_geom`), canonical grid and exemption (`_canon_grid` · `_canon_ok_place` · `_canon_exempt`),
    desk check (`_desk_ok`)."""

    def _canon_grid(self, ck, cn: dict) -> tuple:
        """Grid cell nearest to the canonical placement · its distance (m) (through _geom)."""
        J = self.J
        ky = int(np.argmin([abs((float(y) - cn["yaw_deg"] + 180.0) % 360.0 - 180.0) for y in J.yaws]))
        best = None
        for kx in range(J.NBx):
            for kyy in range(J.NBy):
                d = self._geom(ck, (ky, kx, kyy, cn["dz"]))[2]
                e = max(abs(float(d[0]) - cn["dx"]), abs(float(d[1]) - cn["dy"]))
                if best is None or e < best[0]:
                    best = (e, kx, kyy)
        return (ky, best[1], best[2]), best[0]

    def _canon_ok_place(self, ck, place) -> str:
        """Put-down rule of the canonical cell (place_rule.hit role=canon) — the first violated reason or ''."""
        from dexengine.placement.scene import place_rule as PR  # noqa: PLC0415

        for who, oxy in getattr(ck, "pr_must", ()):
            if why := PR.hit(oxy, place, ck.keepout, ck.fixed_box, role="canon"):
                return f"{who}:{why}"
        return ""

    def _canon_exempt(self, ck, cn: dict) -> None:
        """Only the canonical cell is exempt from the put-down rule's off-desk test (the robot keep-out still applies).
        Marker [CANON_EXEMPT]."""
        J = self.J
        if getattr(ck, "pr", None) is None or not getattr(ck, "pr_must", ()):
            return
        (ky, kx, kyy), e = self._canon_grid(ck, cn)
        if e > J.step / 2.0 or ck.desk_m[ky, kx, kyy]:
            return  # off the grid (the canonical-cell check dies) · already kept
        keep = ck.pr_must
        ck.pr_must = ()
        J._clip_desk_mask(ck)
        base = bool(ck.desk_m[ky, kx, kyy])
        ck.pr_must = keep
        J._clip_desk_mask(ck)
        if not base:
            print(
                f"{TAG} [CANON_EXEMPT] {ck.clip} not applied — the mask without the put-down rule already drops "
                "the canonical cell",
                flush=True,
            )
            return
        yaw, _dc, d, p2 = self._geom(ck, (ky, kx, kyy, cn["dz"]))
        why = self._canon_ok_place(ck, (float(d[0]), float(d[1]), yaw, p2))
        if why:
            print(
                f"{TAG} [CANON_EXEMPT] {ck.clip} not applied — the canonical cell also hits {why} (the robot "
                "keep-out is not exempt)",
                flush=True,
            )
            return
        n0 = int(ck.desk_m.sum())
        ck.desk_m[ky, kx, kyy] = True
        print(
            f"{TAG} [CANON_EXEMPT] {ck.clip} canonical cell ({ky},{kx},{kyy}) exempt — only the put-down rule's "
            f"off_desk · robot keepout unchanged · augmentation candidate cells keep the rule (mask cells {n0} → "
            f"{int(ck.desk_m.sum())})",
            flush=True,
        )

    def _desk_ok(self, ck, k: int, d, p2, yaw) -> bool:
        """Final desk check — only the canonical cell (k 0) uses the put-down rule with role=canon."""
        J = self.J
        if k != 0:
            return J.desk_ok_final(ck, d, p2, yaw)
        keep = ck.pr_must if hasattr(ck, "pr_must") else None
        if keep is None:
            return J.desk_ok_final(ck, d, p2, yaw)
        ck.pr_must = ()
        try:
            ok = J.desk_ok_final(ck, d, p2, yaw)
        finally:
            ck.pr_must = keep
        return ok and not self._canon_ok_place(ck, (float(d[0]), float(d[1]), yaw, p2))

    def _geom(self, ck, cell) -> tuple:
        """Cell → (yaw deg, dcap (3,), d (2,), p2 (2,)) — the same functions as anchor candidates · adoption
        (Judge._cell_dcap · _pivot_swap)."""
        J = self.J
        yaw = float(J.yaws[cell[0]])
        dcap = J._cell_dcap(ck, cell[1], cell[2])
        d, p2 = J._pivot_swap(ck.clip, ck.cpiv, dcap, yaw)
        return yaw, dcap, d, p2
