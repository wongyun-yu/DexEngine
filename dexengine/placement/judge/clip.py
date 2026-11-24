"""One clip: tree, active sides, input record, wrists, and the search driver (_search, _search_on_desk, run_clip).
The pivot, objects, index cache and desk mask are in judge/clip_inputs.py."""

from __future__ import annotations

import json
import time

import numpy as np
from scipy.spatial.transform import Rotation

from dexengine.core import qpos_layout as _QL
from dexengine.core.errors import InputMissingError
from dexengine.placement import ledger_states as _LS  # reject-row reason heads the status ledger folds
from dexengine.placement.collision import hand_surface as HSF  # the clip's finger values for guards and floors
from dexengine.placement.collision import idle_arm as IA
from dexengine.placement.collision import support_surface as SSF  # the arm guard's support objects of the clip
from dexengine.placement.coords import core as VC
from dexengine.placement.judge import aug_cells_canon as _ACC  # owner of the [AUG_CELLS] tag
from dexengine.placement.judge import input_record as _JFP  # per-clip judge input record
from dexengine.placement.judge.common import ONDESK_TAG, _ClipBuf
from dexengine.placement.judge.exhaust import _Exhaustion
from dexengine.placement.judge.search import SearchMixin
from dexengine.placement.scene import place_rule as _PR  # where the main and fixed objects may stand


class ClipMixin:
    def _search(self, ck, masked: bool = False) -> bool:
        """Search of one clip context (ck): desk mask → volume → anchors → exhaustive search. True = run_clip ends
        without the closing count (no volume, or a placement strategy said so). masked = the caller already filled the
        index and the mask (_search_on_desk)."""
        a = self.a
        if not masked:
            self._clip_index_cache(ck)
            self._clip_desk_mask(ck)
        if self._PS is not None:  # --place-strategy plugin (default canon = None)
            return self._PS.search(self, ck)
        if not self._clip_volume(ck):
            return True
        self._anchor_search(ck)
        # Exhaustive search: when every anchor path fails, judge the whole volume in descending coverage order. A
        # failure here means "no continuous solution inside this placement family (volume)" — a certificate within
        # the limits of the grid and the thresholds.
        if a.exhaustive and not ck.got_any:
            _Exhaustion(self, ck).run()
        return False

    def _search_on_desk(self, clip, tree_dir, traj, ck) -> bool:
        """Search of a clip with grazed still extras (fixed objects that are not held). The first search (the plain
        one) writes into buffers. There is one adopted line per anchor and the bake bakes them in ascending anchor
        order, so every adopted line is checked: when none pushes a grazed object to a spot that breaks the place
        rule (place_rule.hit) the lines are replayed unchanged. Otherwise search again from scratch with only the
        cells where the grazed objects also keep both conditions (ck.ondesk — mask and final check); replay the
        second search's lines when it adopts, else the first search's, with each line's dropped grazed objects in
        `place_drop`."""
        real = (self.fpl, self.fno)
        b0 = (_ClipBuf(), _ClipBuf())
        self.fpl, self.fno = b0
        try:
            stop = self._search(ck)
        finally:
            self.fpl, self.fno = real
        offs = {}
        if not stop and ck.got_any:
            for d in map(json.loads, b0[0].lines):
                if d.get("clip") == clip:
                    offs[int(d["anchor"])] = (
                        SearchMixin.ondesk_off(
                            ck, (d["dx"], d["dy"]), np.asarray(d["pivot"][:2], float), float(d["yaw"])
                        ),
                        f"dx={d['dx']:+.3f} dy={d['dy']:+.3f} yaw={d['yaw']:+.1f}",
                    )
        off = sorted({n for names, _w in offs.values() for n in names})
        if not off:
            b0[0].replay(self.fpl)
            b0[1].replay(self.fno)
            return stop
        keep = (self.n_ok, self.n_no, self._bud_kind)
        self.REJ.clear()  # the wrist cache (self.FULL, from _clip_wrists) is shared by both searches
        self._BEST.clear()
        self._bud_kind = None
        self._seed_clip(clip)
        ck1 = self._clip_objects(clip, tree_dir, traj)
        ck1.ondesk = True
        # Cheap geometric pre-filter: reuse the first search's index (same wrists, same origin) and redo only the
        # mask; when no cell keeps the grazed objects, return to the original placement without any IK or collision.
        ck1.d0s, ck1.idx_cache, ck1.cent_cache, ck1.d0cap = ck.d0s, ck.idx_cache, ck.cent_cache, ck.d0cap
        b1 = (_ClipBuf(), _ClipBuf())
        self.fpl, self.fno = b1
        try:
            self._clip_desk_mask(ck1)
            left = int(ck1.desk_m.sum()) if ck1.desk_m is not None else 0
            stop1 = True if not left else self._search(ck1, masked=True)
        finally:
            self.fpl, self.fno = real
        was = " · ".join(f"anchor{a_} {w}" for a_, (names, w) in sorted(offs.items()) if names)
        n_all = ck1.desk_m.size if ck1.desk_m is not None else 0
        print(f"{ONDESK_TAG} {clip} re-search pre-filter: cells left {left}/{n_all}", flush=True)
        if not stop1 and ck1.got_any:  # a second search that stopped early (stop1) has no got_any
            b1[0].replay(self.fpl)
            b1[1].replay(self.fno)
            r1 = next(json.loads(ln) for ln in b1[0].lines if json.loads(ln).get("clip") == clip)
            print(
                f"{ONDESK_TAG} {clip} pushed {','.join(off)} ({was}) → new placement dx={r1['dx']:+.3f} "
                f"dy={r1['dy']:+.3f} yaw={r1['yaw']:+.1f} · off-spot left 0",
                flush=True,
            )
        else:
            self.n_ok, self.n_no, self._bud_kind = keep
            for ln in b0[0].lines:  # keep the original placement; each line lists its dropped objects
                d = json.loads(ln)
                if d.get("clip") == clip and offs.get(int(d["anchor"]), ((), ""))[0]:
                    d["place_drop"] = list(offs[int(d["anchor"])][0])
                    ln = json.dumps(d, ensure_ascii=False) + "\n"
                self.fpl.write(ln)
            b0[1].replay(self.fno)
            ck.pr_drops = {a_: list(names) for a_, (names, _w) in offs.items() if names}
            print(
                f"{ONDESK_TAG} {clip} pushed {','.join(off)} ({was}) → no placement keeps the grazed objects · "
                f"original placement kept · off-spot left {len(off)} (not spawned there — place_rule.drop_why)",
                flush=True,
            )
        return False

    def run_clip(self, ci: int, clip: str) -> None:
        """Judge one clip. An early `return` skips the closing count (n_ok, the 20-clip progress line)."""
        a = self.a
        self.clip = clip
        self._bud_kind = None  # budget cut (note_budget), per clip
        self._BEST.clear()  # this clip's best-candidate met tracker
        # Randomness is pinned **per clip** — this one line is what makes the output invariant to list splits,
        # parallel order and resume (see `_seed_clip`). It is not pinned
        # again inside a clip: the reseed budget (`--track-reseeds`) is the mechanism for "try
        # again with different randomness", so planting the same seed on every call would draw
        # the same sample over and over and kill that mechanism.
        self._seed_clip(clip)
        tree_dir = self._clip_tree(clip)
        self._clip_active_sides(clip, tree_dir)
        # The torso guard places each hand by this clip's finger values on the wrist cache's time axis (an idle side's
        # = the hand's rest pose, hand_surface.rest_finger_q) — hand_surface.guard_finger_q, the same function as the
        # bake, the gate and the final guard.
        self.FQ = HSF.guard_finger_q(tree_dir, a.hand, idle=self.AS_CUR["idle"])
        # The retiming's finger floor reads every hand block the tree has (tree_finger_q — retime.py does the same;
        # zero blocks die in _ff); read once per clip, the floor is built per time-axis spacing in _ff (clip constant).
        self._FFQ = HSF.tree_finger_q(tree_dir, a.hand)
        self._FFC = {}
        self.SUPPORTS = SSF.scene_supports(tree_dir)  # the guard's support term — the same function as bake · gate
        jfp = self._clip_jfp(clip)
        self.REJ.clear()
        self.FULL.clear()
        traj, skip = self._clip_wrists(clip)
        if self._AUG is not None:  # augmentation cells (modes D · L): region dump / list verdict, no search
            if skip:
                raise InputMissingError(
                    f"{_ACC.TAG} ✗ {clip}: no wrist cache ({skip}) — only adopted clips are accepted"
                )
            self._AUG.clip(self._clip_objects(clip, tree_dir, traj), jfp)
            return
        if skip:
            self.fno.write(f"{clip}\t-\t{skip}\n")
            self.n_no += 1
            return
        ck = self._clip_objects(clip, tree_dir, traj)  # this clip's context — filled and read by the stages below
        stop = self._search_on_desk(clip, tree_dir, traj, ck) if ck.nonrecv else self._search(ck)
        if getattr(ck, "pr", None) is not None:  # the place rule's one result line per clip
            _got = bool(getattr(ck, "got_any", False))
            _dn = sorted({n for v in ck.pr_drops.values() for n in v})
            _why = (
                "-"
                if _got
                else _LS.PLACE_NO_SPOT
                if getattr(ck, "place_none", None)
                else "no adoption in the cells left by the place rule (other conditions)"
                if ck.pr_cut
                else "other conditions (the place rule dropped 0 cells)"
            )
            print(
                f"{_PR.TAG} {clip} result {'adopted' if _got else 'dropped'} · cells cut {ck.pr_cut} · cells left "
                f"{ck.pr_left} · grazed objects dropped {len(_dn)}({','.join(_dn) or '-'}) · reason {_why}",
                flush=True,
            )
        if stop:
            return
        self.n_ok += ck.got_any
        if (ci + 1) % 20 == 0:
            self.fpl.flush()
            print(
                f"[judge:{self.combo}] {ci + 1}/{len(self.clips)} ok={self.n_ok} no={self.n_no} "
                f"solves {self.n_chk} ({time.time() - self.t00:.0f}s)",
                flush=True,
            )

    def _clip_tree(self, clip):
        """This clip's tree (object trajectory + task_info): binds the capture → world z (Z_OFF) and the desk box
        (DXb / DYb) to this tree."""
        a = self.a
        # This clip's tree (object trajectory + task_info). The capture → world z offset is **told
        # by the tree** (coords.scene_z_offset — base layer support.desk_z, TACO staging 0). With no
        # tree the offset is unknowable, so it dies here instead of judging with the desk mask off.
        tree_dir = self.TRAJ_ROOT / clip
        if not (tree_dir / "task_info.json").is_file():
            raise RuntimeError(f"[judge] {clip}: no object tree — {tree_dir} (check --traj-root / --dataset / --emb)")
        self.Z_OFF = VC.scene_z_offset(tree_dir)  # aerial: coords resolves to disk/obj0 plane ([DESK_Z] … aerial:*)
        # The scene desk box is **the one desk top of the tree's dataset** (coords.desk_bounds_scene_for).
        # The tree's actual table outline (GRAB) is used for the first-pass exclusion only, not here — so the
        # desk env spawns and the box judge and render see are one and the same. A tree of another dataset than
        # this run's would be judged with this run's pool (measured at this run's desk top) — it dies.
        if (_tds := VC.tree_dataset(tree_dir)) != a.dataset:
            raise SystemExit(
                f"[judge] {clip}: tree dataset {_tds!r} ≠ this judge's --dataset {a.dataset!r} — the "
                f"desk top and the pool clearance are tied to the dataset"
            )
        self.DXb, self.DYb = VC.desk_bounds_scene_for(tree_dir, shrink_robot=a.desk_shrink_robot)
        self.DESK_XY = VC.desk_bounds_scene_for(tree_dir, margin=0.0)
        _fp_sup = json.loads((tree_dir / "task_info.json").read_text()).get("support") or {}
        if _fp_sup.get("desk_footprint_raw"):
            print(
                f"[DESK_XY] {clip}: desk box x=({self.DXb[0]:+.3f},{self.DXb[1]:+.3f}) "
                f"y=({self.DYb[0]:+.3f},{self.DYb[1]:+.3f}) — the tree's own outline is used for the first-pass "
                f"exclusion only",
                flush=True,
            )
        return tree_dir

    def _clip_active_sides(self, clip, tree_dir):
        """One-hand rule and one-hand embodiment: this clip's active / idle sides (rebinds self.sides) and the budget
        (x2 for one arm)."""
        a = self.a
        # The hands actually present in the tree = qpos_layout (owned by the tree). A one-hand emb
        # tree (1 hand) dies at once in both mode — the chain is fixed to two hands, so
        # carrying on quietly ends in "no wrist cache".
        _lay_obj = _QL.read(str(tree_dir), nq=None)
        _lay_hands = tuple(s for s in ("right", "left") if s in _lay_obj.hands)
        _lay_manip = tuple(_lay_obj.manip_sides())  # judge reads wrists where f0 shadowed (hand ∧ main obj)
        if a.active_sides != "auto" and len(_lay_hands) != 2:
            raise RuntimeError(
                f"[ACTIVE_SIDES] {clip}: one-hand tree (hands={_lay_hands}) — needs --active-sides auto "
                f"(one-hand embodiment); 'both' is for two-hand trees only"
            )
        if a.active_sides == "auto":
            _cti = self._CASE_ROOT / clip / "task_info.json"
            if not _cti.is_file():
                raise RuntimeError(f"[ACTIVE_SIDES] {clip}: no base-tree task_info — {_cti}")
            _case = json.loads(_cti.read_text()).get("case")
            try:
                _act, _idle, _rule = IA.active_idle_for_tree(
                    _case, _lay_hands, _lay_manip
                )  # one owner (same fn as the bake) · manip-side intersection
            except ValueError as _e:
                raise RuntimeError(f"[ACTIVE_SIDES] {clip}: {_e}") from None
            if _idle:
                self.sides = {s: self.ALL_SIDES[s] for s in _act}
                self.AS_CUR["active"] = _act
                self.AS_CUR["idle"] = _idle
                a.exhaust_max = self._EXH0 * 2
                print(
                    f"[ACTIVE_SIDES] {clip} active={_act[0]} idle={self.AS_CUR['idle'][0]}(rest) rule={_rule} "
                    f"case={_case.get('case') if isinstance(_case, dict) else None} "
                    f"exhaust_max={a.exhaust_max}(×2)",
                    flush=True,
                )
            else:
                self.sides = dict(self.ALL_SIDES)
                self.AS_CUR["active"], self.AS_CUR["idle"] = ("right", "left"), ()
                a.exhaust_max = self._EXH0
                print(
                    f"[ACTIVE_SIDES] {clip} active=both case={_case.get('case')} case.active={_case['active']}",
                    flush=True,
                )

    def _clip_jfp(self, clip):
        """One judge input record line (input_record): written after the active sides are set and before the wrist
        cache is read (skip path included). The strict proof ran before the first clip (core.prove_inputs), so a
        field that fails here means the input changed since; it is marked and the judge goes on."""
        a = self.a
        _jfp = _JFP.encode(_JFP.clip_parts(a.dataset, a.emb, a.hand, clip, self._SCENE, a.active_sides), self._JFP_COND)
        self.fjfp.write(f"{clip}\t{_jfp}\n")
        _jbad = [k for k, v in (_JFP.parse(_jfp) or {}).items() if v.startswith(_JFP.ERR)]
        if _jbad:
            print(f"[JUDGE_FP] {clip}: fields not computable {_jbad} — read as stale by the comparison", flush=True)
        return _jfp

    def _clip_wrists(self, clip):
        """Wrist cache: the strided trajectory traj and all frames in self.FULL. Returns (traj, skip reason | None)."""
        a = self.a
        traj = {}
        skip = None
        for side, S in self.sides.items():
            f = S["wd"] / f"{clip}.npz"
            if not f.exists():
                skip = f"{side} no wrist cache"
                break
            z = np.load(f)
            Pp, Q = z["positions"], z["quaternions"]
            if Pp.ndim == 3:
                Pp, Q = Pp[0], Q[0]
            rot_all = Rotation.from_quat(np.stack([Q[:, 1], Q[:, 2], Q[:, 3], Q[:, 0]], 1))
            traj[side] = (Pp[:: a.stride], rot_all[:: a.stride])
            self.FULL[side] = (Pp, rot_all)  # confirm stage only: all frames, same as the bake
        return traj, skip
