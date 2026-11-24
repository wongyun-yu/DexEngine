"""Judge anchor search of one clip: ray cells over the volume, batched or serial pick with the guards, the fallback
when no cell passes, the pivot swap and the adoption of the anchor. Part of the Judge class (judge/core.py
composes the mixins); the volume, the guards and the desk checks it calls live in judge/search.py.
"""

from __future__ import annotations

import json

import numpy as np

from dexengine.placement.judge import clip_time as _CT
from dexengine.placement.judge.common import _cand_place


class AnchorMixin:
    """Anchor search (`_anchor_search`) and its steps."""

    @_CT.timed("anchor")
    def _anchor_search(self, ck):
        """Per anchor: candidates on the ray (anchor → centre) x dz, confirmed with early stop; sets ck.got_any."""
        a = self.a
        # Per anchor: candidates along the ray (anchor → centre) × dz (ascending |dz|) → confirm early-stop
        ck.got_any = False
        ck._v2_tried = False  # the v2 port fallback runs 1 time per clip
        for ai_, (ax_c, ay_c) in enumerate(self.anchors):
            cand = self._anchor_ray_cells(ck, ax_c, ay_c)
            if a.batch_cells > 1:
                picked, checks = self._anchor_pick_batched(ck, cand)
            else:
                picked, checks = self._anchor_pick_serial(ck, cand)
            if picked is None:
                self._anchor_fallback(ck, ai_, checks)
                continue
            self._anchor_adopt(ck, ai_, picked)

    def _anchor_ray_cells(self, ck, ax_c, ay_c):
        """Volume cells on one anchor's ray (within 3 cells of a ray point) x dz layers, in search order."""
        V = ck.V
        sb = self.rz.apply(np.array([ax_c, ay_c, 0.0]))
        axy = np.array([sb[0] / self.step + self.Wx, sb[1] / self.step + self.Wy])  # anchor (window coordinates)
        ctr = np.array([self.Wx, self.Wy], float)
        ray = ctr - axy
        L = np.linalg.norm(ray)
        steps = max(1, int(L))
        cand = []
        seenc = set()
        for t in np.linspace(0, 1, steps + 1):
            px = axy + ray * t
            for dzi in self.DZ:
                ky, kx, kyy = np.nonzero(V[dzi])
                if not len(ky):
                    continue
                d2 = (kx - px[0]) ** 2 + (kyy - px[1]) ** 2
                j = int(d2.argmin())
                if d2[j] > 3.0**2:  # more than 3 cells from the ray point means there is none here
                    continue
                key = (int(ky[j]), int(kx[j]), int(kyy[j]), dzi)
                if key in seenc:
                    continue
                seenc.add(key)
                cand.append(key)
        return cand

    def _anchor_pick_batched(self, ck, cand):
        """Batched anchor: the gate runs in place, one by one; the FULL of the passers is one track_full_multi batch.
        Returns (picked | None, checks)."""
        a = self.a
        traj, cpiv = ck.traj, ck.cpiv
        picked = None
        checks = 0
        # Anchor batching: the gate runs in order, on the spot (same materials, same
        # rejections); only the FULL (track) of gate-passing candidates goes through
        # track_full_multi as one batch. Adoption is the first pass in candidate order — the
        # same answer as serial, with serial FULL time (≤12×6s) replaced by 1 batch (~10s).
        _aq = []
        for ky_, kx_, kyy_, dzi in cand:
            if checks >= a.max_checks:
                break
            yaw = float(self.yaws[ky_])
            dcap = self._cell_dcap(ck, kx_, kyy_)
            checks += 1
            self.n_chk += 1
            okg, metg = self.dp_confirm(traj, cpiv, yaw, dcap, dzi * self.step, step_mult=float(a.stride * a.dp_stride))
            if not okg:
                self._tally("sparse", metg)
                continue
            _aq.append(
                {
                    "piv": cpiv,
                    "yaw": yaw,
                    "dcap": dcap.copy(),
                    "dzm": dzi * self.step,
                    "dzi": dzi,
                    "chk": checks,
                    "gateq": {s2: (self.GATEQ[s2].copy() if s2 in self.GATEQ else None) for s2 in self.sides},
                }
            )
        if _aq:
            _res = list(zip(_aq, self.track_full_multi(_aq), strict=True))
            _gj = [(it_["dzm"], q_, _cand_place(it_)) for it_, (ok_, _m, q_, _e) in _res if ok_ and q_ is not None]
            with self.guard_prefetch(_gj):
                for it_, (ok_, met_, q_, _err) in _res:
                    if ok_ and picked is None:
                        # adoption = the first FULL pass that also passes the deciding ruler (mesh guard) — a guard
                        # rejection moves on to the next pass (its IK is already solved), never drops the anchor
                        self.LASTQ.clear()
                        self.LASTQ.update(q_)
                        if self._guard_pick(it_["dzm"], _cand_place(it_)):
                            picked = (it_["yaw"], it_["dcap"], it_["dzi"], 0.0, met_, it_["chk"])
                    elif not ok_:
                        self._tally("confirm", met_)
        return picked, checks

    def _cell_dcap(self, ck, kx_, kyy_) -> np.ndarray:
        """Window cell (kx, kyy) → candidate origin shift dcap (capture, 3); used by anchors and augmentation cells."""
        bxy = np.array([(kx_ - self.Wx) * self.step, (kyy_ - self.Wy) * self.step, 0.0])
        return self.rz.inv().apply(bxy) + ck.d0cap  # origin displacement

    def _pivot_swap(self, clip, cpiv, dcap, yaw) -> tuple:
        """Pivot swap of an adoption — (d (2,), p2 (2,)): the shift about the clip pivot re-expressed about the adopted
        record's pivot (the bake's _tf_points formula)."""
        p2 = self._adopted_pivot(clip, cpiv)
        th = np.deg2rad(yaw)
        R2 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return np.array([dcap[0], dcap[1]]) + (np.eye(2) - R2) @ (cpiv[:2] - p2), p2

    def _guard_pick(self, zoff: float, place) -> bool:
        """Candidate pick on the deciding ruler: _pen_cross_guard on LASTQ (the search filters are the capsule proxy,
        arm_proxy.COLL_PROXY — the guard is the mesh, GUARD_KIND). A rejection is tallied ('guard') and the caller
        tries its next candidate; the result is remembered for _anchor_adopt's adoption-site call (same q, z)."""
        ok, why = self._pen_cross_guard(zoff, place=place)
        if not ok:
            self._tally("mesh", {"-": f"guard {why}"})
            print(f"[GUARD_NEXT] {self.clip} guard rejected {why} → next candidate", flush=True)
        return ok

    def _anchor_pick_serial(self, ck, cand):
        """Serial anchor: dp_accept (gate → FULL) per candidate, stop at the first pass. Returns (picked | None,
        checks)."""
        a = self.a
        traj, cpiv = ck.traj, ck.cpiv
        picked = None
        checks = 0
        for ky_, kx_, kyy_, dzi in cand:
            if checks >= a.max_checks:
                break
            yaw = float(self.yaws[ky_])
            dcap = self._cell_dcap(ck, kx_, kyy_)
            checks += 1
            self.n_chk += 1
            okd, met = self.dp_accept(traj, cpiv, yaw, dcap, dzi * self.step)
            # the first dp_accept pass that the mesh guard also passes
            if okd and self._guard_pick(dzi * self.step, (dcap[0], dcap[1], yaw, cpiv)):
                picked = (yaw, dcap, dzi, 0.0, met, checks)
                break
        return picked, checks

    def _anchor_fallback(self, ck, ai_, checks):
        """Anchor failed: try the ported v2 placement once per clip, else write a 'confirm_failed' reject row."""
        a = self.a
        clip, traj = ck.clip, ck.traj
        # Fallback: the v2 port placement (v2 pivot as-is, capture-corrected)
        # **1 time per clip only** — re-solving the same v2 placement as a single-path FULL on
        # each of the 3 anchor rounds would burn 20~40s per clip. Pass or fail, it is a retry on
        # the same input, so it carries no information.
        adk = self._adopted_path(clip)
        # A record that exists but cannot be read, or a step that raises, dies — the declared outcomes (not
        # accepted · off the desk · guard rejection) are the branches below, never a swallowed exception.
        if adk.exists() and checks < a.max_checks + 2 and not ck._v2_tried:
            ck._v2_tried = True
            av2 = json.loads(adk.read_text())
            piv2 = np.array(av2["pivot"], float)
            dc2 = np.array([av2["dx"] + self.TRANS_CAP[0], av2["dy"] + self.TRANS_CAP[1], 0.0])
            dz2 = float(av2["robot_z_off"])
            okd2, _m2 = self.dp_accept(traj, piv2, float(av2["yaw"]), dc2, dz2)
            sat2 = 0.0 if okd2 else 1.0
            if okd2 and self.desk_ok_final(ck, dc2, piv2, float(av2["yaw"])):
                _gok, _gwhy = self._pen_cross_guard(dz2, place=(dc2[0], dc2[1], float(av2["yaw"]), piv2))
                if _gok:
                    self.fpl.write(
                        json.dumps(
                            {
                                "clip": clip,
                                "anchor": ai_,
                                "dx": round(float(dc2[0]), 4),
                                "dy": round(float(dc2[1]), 4),
                                "yaw": round(float(av2["yaw"]), 2),
                                "pivot": [float(piv2[0]), float(piv2[1]), 0.0],
                                "robot_z_off": round(dz2, 4),
                                "sat": round(sat2, 4),
                                "checks": checks + 1,
                                "transplant": True,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    self.save_ik(clip, ai_)
                    ck.got_any = True
                    return
                self.fno.write(f"{clip}\tanchor{ai_}\tv2_port_guard_rejected {_gwhy}\n")
        _why = " ".join(f"{k}={v}" for k, v in sorted(self.REJ.items(), key=lambda x: -x[1]))
        self.fno.write(f"{clip}\tanchor{ai_}\tconfirm_failed(checks {checks}) {_why}\n")

    def _anchor_adopt(self, ck, ai_, picked):
        """Anchor adoption: pivot swap, then final desk check → guard (_pen_cross_guard) → plan line · IK save."""
        clip, cpiv = ck.clip, ck.cpiv
        yaw, dcap, dzi, sat, ss, checks = picked
        d, p2 = self._pivot_swap(clip, cpiv, dcap, yaw)
        if not self.desk_ok_final(ck, d, p2, yaw):
            self.fno.write(f"{clip}\tanchor{ai_}\tfinal_desk_check_rejected(left the desk after the pivot swap)\n")
            return
        _gok, _gwhy = self._pen_cross_guard(dzi * self.step, place=(d[0], d[1], yaw, p2))
        if not _gok:
            self.fno.write(f"{clip}\tanchor{ai_}\tfinal_guard_rejected {_gwhy}\n")
            return
        self.fpl.write(
            json.dumps(
                {
                    "clip": clip,
                    "anchor": ai_,
                    "dx": round(float(d[0]), 4),
                    "dy": round(float(d[1]), 4),
                    "yaw": round(yaw, 2),
                    "pivot": [float(p2[0]), float(p2[1]), 0.0],
                    "robot_z_off": round(dzi * self.step, 4),
                    "sat": round(sat, 4),
                    "checks": checks,
                    "met": {k: str(v) for k, v in (ss or {}).items()},
                },  # observability
                ensure_ascii=False,
            )
            + "\n"
        )
        self.save_ik(clip, ai_)
        ck.got_any = True
