"""Placement search of one clip: desk checks, guards, volume, budget rows, finish (anchors: judge/anchor.py)."""

from __future__ import annotations

import contextlib
import functools
import time

import numpy as np
import torch

from dexengine.placement import ledger_states as _LS  # reject-row reason heads the status ledger folds
from dexengine.placement.collision import core as SCL
from dexengine.placement.collision import idle_arm as IA
from dexengine.placement.collision import support_surface as SSF  # the arm guard's support objects of the clip
from dexengine.placement.coords import core as VC
from dexengine.placement.judge import budget as _JB  # per-clip budget-cut record (file · cell · marker)
from dexengine.placement.judge import clip_time as _CT  # [CLIP_T] per-clip stage seconds
from dexengine.placement.judge import guard_pool as _GP  # guards evaluated ahead in forked workers
from dexengine.placement.judge.common import _cand_place, _place_key, _same_guard_key
from dexengine.placement.reach_map import batch_ik as OPG
from dexengine.placement.scene import object_desk as _OD  # rule D24: main-object hull vs the desk
from dexengine.placement.scene import place_rule as _PR  # where the main and fixed objects may stand
from dexengine.placement.scene import static_desk as _SD  # desk rule of the render-added static objects

DEV, DT = OPG.DEV, OPG.DT


class SearchMixin:
    def desk_ok_final(self, ck, dxy, piv2, yaw_deg) -> bool:
        """Final check just before adoption — moves the object with **the same formula** as the
        bake's `_tf_points` and sees whether it lands inside the desk. Why the mask is not
        enough: the mask checks against `cpiv`, while the final d swaps the pivot for v2's p2
        and adds `(I-R2)@(cpiv-p2)`. When the two pivots differ, the position the mask
        guaranteed and the position actually placed come apart."""
        if ck.objs_dm is None:
            return True
        for oxy in ck.objs_dm:
            w = VC.capture_to_scene(VC.place_objects(oxy, float(dxy[0]), float(dxy[1]), yaw_deg, piv2))
            if w[:, 0].min() < self.DXb[0] or w[:, 0].max() > self.DXb[1]:
                return False
            if w[:, 1].min() < self.DYb[0] or w[:, 1].max() > self.DYb[1]:
                return False
        for sxy in getattr(ck, "statics", ()):  # the render-added static objects (rule owner static_desk)
            w = VC.capture_to_scene(VC.place_objects(sxy, float(dxy[0]), float(dxy[1]), yaw_deg, piv2))
            if _SD.overhang_frac(w, box=self.DESK_XY) > _SD.OVERHANG_MAX_FRAC:
                return False
        place = (float(dxy[0]), float(dxy[1]), yaw_deg, piv2)
        if ck.obj_deep is not None and _OD.hit(ck.obj_deep, place, self.DESK_XY):  # rule D24 (object_desk)
            return False
        for _who, oxy in getattr(ck, "pr_must", ()):  # the place rule: what must stand (place_rule.hit)
            if _PR.hit(oxy, place, ck.keepout, ck.fixed_box):
                return False
        if getattr(ck, "ondesk", False):
            if SearchMixin.ondesk_off(ck, dxy, piv2, yaw_deg):
                return False
        return True

    @staticmethod
    def ondesk_off(ck, dxy, piv2, yaw_deg) -> list:
        """Names of the grazed still extras that the placement (capture dx · dy · yaw · pivot, the bake's formula)
        pushes against the place rule (place_rule.hit — overlap with the robot's spot, area off the desk), so they are
        dropped from the spawn. Read by the first search's adoption check, the second search's final check and the
        plan line's `place_drop`."""
        place = (float(dxy[0]), float(dxy[1]), yaw_deg, piv2)
        return [name for name, oxy in getattr(ck, "nonrecv", ()) if _PR.hit(oxy, place, ck.keepout, ck.fixed_box)]

    @_CT.timed("guard")
    def _pen_cross_guard(self, zoff: float, q_override: dict | None = None, *, place) -> tuple[bool, str]:
        """Re-checks arm-desk penetration, arm · palm · finger ↔ torso overlap (exact surface ruler) and the
        two-arm crossing just before adoption, with **the same function
        and the same thresholds** as the gate (gate.py) — **every plan-adoption site of the
        judge (anchor, refine, exhaustion _adopt_ex, non-batched exhaustion) calls this one
        function and nothing else** (with two copies the thresholds, the order and the reasons
        drift apart).

        Why: on gate-rejected clips the violating frames sit mid-trajectory (75~77% of the way
        through; crossing 0/485 and penetration 0/1,273 frames inside the f0 inserted stretch) —
        the guard belongs at the adoption site, not in f0. The canonical dexengine/placement/collision/core.py is used
        by import only (no reimplementation).

        Reasons are per check (desk penetration · support penetration · torso surface · arm crossing —
        collision.pen_cross_guard), never lumped.

        q_override: passing {"right":q,"left":q} directly checks without touching LASTQ (for
        callers such as _adopt_ex that must avoid polluting LASTQ on failure).
        When unspecified it reads LASTQ (the all-frame solution dp_accept/track_full_multi
        filled in just before adoption — the same array the bake reuses).

        place: the candidate placement (capture dx, dy, yaw_deg, pivot) the bake will apply — the support term
        measures the supports moved by it (support_surface.placed_supports), the pose the gate reads on the baked
        tree. Required: the base tree's own pose is the capture pose, not the scene the arm moves in.

        The body lives in `collision.pen_cross_guard` — so that the second layer after the bake
        (final_guard.py) calls **the same function**. This function is a thin shell that reads
        LASTQ and passes it on.
        """
        src = q_override if q_override is not None else self.LASTQ
        memo_k = self._guard_key(zoff, src, place)
        m = getattr(self, "_GMEMO", None)
        if m is not None and _same_guard_key(m[0], memo_k):
            return m[1]
        # a flush's guards may have been evaluated ahead (guard_prefetch) — the same _guard_eval on the same inputs
        pre = getattr(self, "_GPRE", None)
        hit, res = pre.take(memo_k) if pre is not None else (False, None)
        if not hit:
            res = self._guard_eval(zoff, src, place)
        self._GMEMO = (memo_k, res)
        return res

    def _guard_key(self, zoff: float, src: dict, place) -> tuple:
        """The guard's memo key: one guard per (q arrays, z, placement, clip inputs) — the pick's guard and the
        adoption site's are the same call. It holds the arrays themselves (compared by identity, _same_guard_key, so a
        freed id cannot alias). The placement enters in its pivot-free form (yaw · d + pivot − Rz·pivot): the pick
        (clip pivot) and the adoption (swapped pivot, compensated d) are one rigid motion, one key."""
        return (
            zoff,
            tuple(sorted((k, v) for k, v in src.items() if v is not None)),
            self.AS_CUR["idle"],
            self.FQ,
            self.SUPPORTS,
            self.DESK_XY,
            _place_key(place),
        )

    def _guard_eval(self, zoff: float, src: dict, place) -> tuple[bool, str]:
        """The guard itself (no memo) — inline from _pen_cross_guard, or ahead in a guard_prefetch worker."""
        src = dict(src)
        if self.AS_CUR["idle"]:
            # An idle arm enters the same guard (desk, self-penetration, two-arm crossing) as a
            # static body in its rest pose.
            _Tq = len(next(iter(src.values()))) if src else 0
            for _sd in self.AS_CUR["idle"]:
                src[_sd] = IA.idle_arm_q(self.a.arm, _sd, _Tq)
        return SCL.pen_cross_guard(
            self.PEN_CTX,
            src,
            z_off=zoff,
            desk_xy=self.DESK_XY,
            fingers=self.FQ,
            supports=SSF.placed_supports(self.SUPPORTS, *place),
        )

    @contextlib.contextmanager
    def guard_prefetch(self, jobs):
        """A flush's guards ahead: jobs = [(zoff, q dict, placement)] in the order the caller's loop will guard them
        (its FULL passes). Inside the block _pen_cross_guard takes each prefetched value by key (guard_pool.Prefetch
        — forked workers, output replayed at the take); whatever the loop does not reach (a stop, a desk
        rejection) is dropped at exit. The caller's loop, its order and its decisions are unchanged."""
        pre = _GP.Prefetch(
            [(self._guard_key(z, q, pl), functools.partial(self._guard_eval, z, q, pl)) for z, q, pl in jobs],
            _same_guard_key,
        )
        self._GPRE = pre
        try:
            yield
        finally:
            self._GPRE = None
            pre.close()

    def _clip_volume(self, ck):
        """Volume per dz layer (two-arm coverage); with none, write a 'no_volume' reject row and return False."""
        a = self.a
        clip, F, n95, idx_cache, desk_m = ck.clip, ck.F, ck.n95, ck.idx_cache, ck.desk_m
        if getattr(ck, "place_none", None):  # no cell keeps what must stand off the robot's spot and on the desk top
            self.fno.write(
                f"{clip}\t-\t{_LS.PLACE_NO_SPOT}({ck.place_none} — no placement keeps the main object "
                f"(start · rest spot) and the must-place static objects off the robot's spot and inside the desk top, "
                f"and the main-object hull out of the desk (D24))\n"
            )
            self.n_no += 1
            return False
        # V95 per dz (erosion)
        V, COV, COVS, MGV, _cov_by_dz = self._dz_volumes(a, idx_cache, F, desk_m, n95)
        ck.V, ck.COV, ck.COVS, ck.MGV = V, COV, COVS, MGV
        if not any(V[d].any() for d in self.DZ):
            _cov_txt = (
                "; ".join(
                    f"dz={_dz:+.2f} "
                    + " ".join(f"{_s[0]}{_p:.0f}%" for _s, _p in _cov.items())
                    + f" both{_both_pct:.0f}%"
                    for _dz, _cov, _both_pct, _nu2 in _cov_by_dz
                )
                if _cov_by_dz
                else ""
            )
            _miss_txt = f" · best_candidate {self._BEST['txt']}" if self._BEST.get("txt") else ""
            self.fno.write(f"{clip}\t-\t{_LS.VOLUME_NONE}({_cov_txt}){_miss_txt}\n")
            self.n_no += 1
            return False
        return True

    def _dz_volumes(self, a, idx_cache, F, desk_m, n95):
        """Per dz layer: the two-arm volume (both arms reach ≥ n95 frames), coverage, strict coverage, margin, and the
        best coverage for the reject reason."""
        V = {}
        COV = {}
        COVS = {}  # dz → minimum two-arm covered frame count (for ordering the exhaustive search)
        MGV = {}  # dz → minimum two-arm capsule margin sum (2nd sort key)
        _cov_by_dz = []  # largest coverage per dz layer, for the 'no_volume' reason
        for dzi in self.DZ:
            cnt = {s: np.zeros((a.n_yaw, self.NBx, self.NBy), np.int32) for s in self.sides}
            cnts = {s2: np.zeros_like(cnt[s2]) for s2 in cnt}
            mgs = {s2: np.zeros((a.n_yaw, self.NBx, self.NBy), np.float32) for s2 in cnt}
            for side, S in self.sides.items():
                P0 = S["P"]
                gis, ais = idx_cache[side]
                Op = S["OKp"][dzi]
                Ops = S["OKsp"][dzi]
                Mps = S["MGsp"][dzi]
                for k in range(a.n_yaw):
                    gi, ai = gis[k], ais[k]
                    giz = np.round(gi[:, 2] - dzi).astype(int)  # half-cell dz: nearest layer
                    inz = (giz >= 0) & (giz < P0["nz"])
                    gx = np.clip(gi[:, 0], 0, P0["nx"] - 1)
                    gy = np.clip(gi[:, 1], 0, P0["ny"] - 1)
                    gz = np.clip(giz, 0, P0["nz"] - 1)
                    c = cnt[side][k]
                    cs = cnts[side][k]
                    ms = mgs[side][k]
                    for f in range(F):
                        if inz[f]:
                            _sl = Op[gx[f] : gx[f] + self.NBx, gy[f] : gy[f] + self.NBy, gz[f]]
                            c += _sl[:, :, ai[f]].any(-1)
                            _sls = Ops[gx[f] : gx[f] + self.NBx, gy[f] : gy[f] + self.NBy, gz[f]]
                            cs += _sls[:, :, ai[f]].any(-1)
                            _slm = Mps[gx[f] : gx[f] + self.NBx, gy[f] : gy[f] + self.NBy, gz[f]]
                            ms += _slm[:, :, ai[f]].max(-1).astype(np.float32)
            if desk_m is not None:
                for s in cnt:
                    cnt[s] = np.where(desk_m, cnt[s], 0)
                    cnts[s] = np.where(desk_m, cnts[s], 0)
            # No erosion: peeling one layer off the volume boundary would lose, wholesale, clips
            # whose only solutions are on the boundary. Confirm actually solves them, so robustness
            # is secured there.
            _nu = F
            _both = np.minimum.reduce([cnt[s] for s in self.sides])
            _cov_by_dz.append(
                (
                    dzi,
                    {s: (float(cnt[s].max()) / _nu * 100.0 if _nu else 0.0) for s in self.sides},
                    (float(_both.max()) / _nu * 100.0 if _nu else 0.0),
                    _nu,
                )
            )
            # AND/min over all active sides (with one side, only that side — with both, the same
            # value as np.minimum(R, L))
            V[dzi] = np.logical_and.reduce([cnt[s] >= n95 for s in self.sides])
            COV[dzi] = np.minimum.reduce([cnt[s] for s in self.sides])
            COVS[dzi] = np.minimum.reduce([cnts[s] for s in self.sides])
            MGV[dzi] = np.minimum.reduce([mgs[s] for s in self.sides])  # minimum two-arm margin sum
        return V, COV, COVS, MGV, _cov_by_dz

    cand_place = staticmethod(_cand_place)

    def exh_resume_start(self, clip, ncands: int) -> int:
        """Start cell of the exhaustive search — 0 without a resume table, else the clip's start. A candidate count
        that differs from the table means the input changed, so the search cannot continue — dies."""
        if self._EXH_RESUME is None:
            return 0
        start, cands = self._EXH_RESUME[clip]
        if cands != ncands:
            raise SystemExit(f"[EXH_RESUME] ✗ {clip}: candidates {ncands} ≠ earlier judge {cands} — the input changed")
        print(
            f"[EXH_RESUME] {clip} from cell {start}/{cands} (the {start} cells the earlier judge covered are skipped) "
            f"· cap {self.a.exhaust_max}",
            flush=True,
        )
        return start

    def note_budget(self, clip, kind, detail: str) -> None:
        """Budget cut handed over by the exhaustive search (None · budget.KINDS); budget_row writes the cell, a cut
        prints the marker."""
        self._bud_kind = kind
        if kind is not None:
            print(f"{_JB.TAG} {clip} {_JB.encode(kind, self._BUD_ROUND)} {detail}", flush=True)

    def budget_row(self, clip) -> None:
        """One budget row; main calls it for every judged clip (an early return in run_clip still passes here)."""
        v = _JB.encode(self._bud_kind, self._BUD_ROUND)
        self.fbud.write(f"{clip}\t{v}\n")
        self._bud_n[v] = self._bud_n.get(v, 0) + 1

    def finish(self) -> None:
        """Close the output files and print the closing markers (done · budget rows · GPU batch counters)."""
        self.fpl.close()
        self.fno.close()
        self.fjfp.close()
        self.fbud.close()
        _cut = {k: v for k, v in self._bud_n.items() if k != _JB.NONE}
        print(
            f"{_JB.TAG} rows {sum(self._bud_n.values())}/{len(self.clips)} · budget cut {sum(_cut.values())}"
            f" {dict(sorted(_cut.items()))} · round {self._BUD_ROUND} · wall clock {self.a.clip_budget_s:g}s · "
            f"cells {self._EXH0} → {self.fbud.name}",
            flush=True,
        )
        if self._AUG is not None:
            self._AUG.finish()
        print(
            f"[judge:{self.combo}] done {self.n_ok}/{len(self.clips)} · confirm solves {self.n_chk} · "
            f"{(time.time() - self.t00) / 60:.1f} min",
            flush=True,
        )
        _ncap = sum(getattr(_S2["ik"], "_dls_ncap", 0) for _S2 in self.ALL_SIDES.values())
        _ncall = sum(getattr(_S2["ik"], "_dls_ncall", 0) for _S2 in self.ALL_SIDES.values())
        _neag = sum(getattr(_S2["ik"], "_dls_neager", 0) for _S2 in self.ALL_SIDES.values())
        _nkey = sum(len(getattr(_S2["ik"], "_dls_graphs", {})) for _S2 in self.ALL_SIDES.values())
        # alloc_retries = how often the allocator had to hand its cache back to fit (a cost in time, never in
        # results) · ooms = allocations that failed even after that
        _ms = torch.cuda.memory_stats() if DEV != "cpu" else {}
        _vpk = (
            f" vram_peak_mib={torch.cuda.max_memory_reserved() // 2**20}"
            f" alloc_peak_mib={torch.cuda.max_memory_allocated() // 2**20}"
            f" alloc_retries={_ms.get('num_alloc_retries', 0)} ooms={_ms.get('num_ooms', 0)}"
            if DEV != "cpu"
            else ""
        )
        print(
            f"[GPU_BATCH] done: cudagraph captures={_ncap} replays={_ncall} eager={_neag} buckets={_nkey} "
            f"clip_release={self._n_clip_release}{_vpk}",
            flush=True,
        )
