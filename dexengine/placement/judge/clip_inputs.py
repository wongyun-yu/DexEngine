"""Judge inputs of one clip: the objects and their pivot, the index cache with the search origin shift, and the desk
mask. Part of the Judge class (judge/core.py composes the mixins); run_clip in judge/clip.py calls them.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from dexengine.placement.coords import core as VC
from dexengine.placement.judge import reselect as _RS
from dexengine.placement.scene import object_desk as _OD
from dexengine.placement.scene import place_rule as _PR
from dexengine.placement.scene import static_desk as _SD
from dexengine.placement.scene import static_fixed as _SF


class ClipInputsMixin:
    """Per-clip inputs: `_clip_objects` · `_clip_pivot` · `_clip_index_cache` · `_origin_shift` · `_clip_desk_mask`."""

    def _clip_pivot(self, clip, traj):
        """The clip pivot (z = 0): mean of the strided wrist positions of every side."""
        own = np.concatenate([traj[s][0] for s in traj]).mean(0)
        own[2] = 0.0
        return own

    def _clip_objects(self, clip, tree_dir, traj):
        """Object trajectories: pivot, object xy, volume threshold n95; builds this clip's context ck.
        One judging window = every frame of the wrist cache (_clip_wrists — volume, gate and confirm use the same
        frames). Trimming the tail (after the objects stop) from the volume alone would leave cells the gate and the
        confirm reject in the volume, spending budget — so the threshold counts all frames too."""
        a = self.a
        F = len(traj[next(iter(self.sides))][0])
        cpiv = self._clip_pivot(clip, traj)
        qk_p = tree_dir / "0/trajectory_kinematic.npz"
        objs = None
        if not qk_p.exists():
            # No silence: in a worktree, when the main_root default points at nothing there is no
            # object tree, and running on would switch **the desk mask and the final desk check
            # off** — the volume inflates (2.5x on one clip) and out-of-desk placements
            # pass. So shout loudly.
            print(f"[judge] warning: no object tree → desk mask OFF — {qk_p} (check the main root)", flush=True)
        if qk_p.exists():
            objs = self._desk_mask_objects(qk_p, clip, a)
        # the static objects the render adds (valid_static · static_required − supports) — the desk check holds
        # them too (rule · threshold · geometry · box: static_desk alone). Pose = this tree's f0 (static_add.rows, the
        # render's function) · names = the base record. Marker [DESK_STATIC] even when 0.
        statics, deep = [], None
        if objs is not None and VC.desk_check_on(self._SCENE):
            # rule D24 (object_desk): the main-object hull vertices deeper than its value below the desk top, every
            # frame — the mask and the final check keep them off the desk footprint
            deep = _OD.deep_points(tree_dir, self.Z_OFF, self.DESK)
            print(_OD.line(clip, deep, self.DESK_XY), flush=True)
            _fp = _SD.footprints(tree_dir)
            statics = [o["xy_cap"] for o in _fp["objs"]]
            print(
                f"{_SD.TAG} {clip} objects {len(statics)} · verdict {_fp['verdict']} · add "
                f"{','.join(_fp['add']) or '-'} · moving, left out {','.join(_fp['moving']) or '-'} · overhang ≤ "
                f"{_SD.OVERHANG_MAX_FRAC:.0%} (physical desk top)",
                flush=True,
            )
        elif objs is not None:  # aerial: no desk — no desk mask, no static check (said, not silent)
            print(f"{_SD.TAG} {clip} objects 0 · scene {self._SCENE} — no desk · no check", flush=True)
        # the place rule (place_rule — desk scenes, every dataset): the outline of each main object where it starts and
        # where it is put down, and of each object the env spawns fixed at frame 0, must stay off the robot's standing
        # spot and on the desk top (area). What must stand (mains · receivers · long-touched) drops the placement
        # (mask + final check); a grazed still extra is dropped from the spawn instead (plan line `place_drop`).
        pr, pr_must, nonrecv, keepout, fixed_box = self._placement_rule(objs, tree_dir, a, clip)
        n95 = int(np.ceil(a.vol_frac * F))
        return SimpleNamespace(
            clip=clip,
            tree_dir=tree_dir,
            traj=traj,
            F=F,
            cpiv=cpiv,
            objs=objs,
            n95=n95,
            statics=statics,
            obj_deep=deep,
            pr=pr,
            pr_must=pr_must,
            keepout=keepout,
            fixed_box=fixed_box,
            nonrecv=nonrecv,
            ondesk=False,
            pr_cut=0,
            pr_left=None,
            pr_drops={},
        )

    def _desk_mask_objects(self, qk_p, clip, a):
        """Object xy of the spawned objects on the judge stride (the desk mask); the object tree must share the wrist
        cache's time axis."""
        qk = np.load(qk_p)["qpos"]
        # Generation consistency: the wrist cache and the object tree must sit on **the same
        # time axis**. Giving the f0 shadow to the wrist only and leaving the object on the old
        # base runs quietly, off by P(=20) frames — the desk mask then looks at the wrong
        # stretch. So shout loudly.
        _dT = len(qk) - len(self.FULL[next(iter(self.sides))][0])
        if abs(_dT) > 2:
            raise RuntimeError(
                f"[judge] {clip}: object tree {len(qk)}f vs wrist cache "
                f"{len(self.FULL[next(iter(self.sides))][0])}f — difference {_dT}. --traj-root / --wrist-root "
                f"are of different generations (before / after f0); give one generation"
            )
        # The desk mask holds the objects the env spawns (coords.desk_mask_slots — main objects): a scene
        # prop is not in the sim scene, so it may land anywhere after placement.
        _sl = VC.desk_mask_slots(qk_p.parent.parent, qk.shape[1])
        objs = [qk[:: a.stride, s0 : s0 + 2].astype(np.float64) for s0, _ in _sl]
        _props = len(VC.tree_object_slots(qk_p.parent.parent, qk.shape[1])) - len(_sl)
        if _props:
            print(f"[DESK_MASK] {clip}: mask objects {len(_sl)} · unspawned props {_props} excluded", flush=True)
        return objs

    def _placement_rule(self, objs, tree_dir, a, clip):
        """The placement rule of this clip (scene/place_rule): keep-out, fixed box, must-hold outlines and the
        non-receiving statics."""
        pr, pr_must, nonrecv, keepout, fixed_box = None, [], [], None, None
        if objs is not None and VC.desk_check_on(self._SCENE):
            pr = _PR.clip(tree_dir, arm=a.arm, margin_m=(_pm := _RS.margin(self._RESELECT, clip)))
            if _pm:
                print(_RS.clip_line(clip, _pm, pr), flush=True)
            keepout, fixed_box = pr["keepout"], pr["box"]
            pr_must = [(f"main:{m['name']}@{m['kind']}", m["outline"]) for m in pr["mains"]] + [
                (f"fixed:{o['name']}@f0", o["xy_cap"]) for o in pr["fixed"] if o["must"]
            ]
            # Grazed still extras do not narrow the placement: when the first search's adoption pushes one to a spot
            # that breaks the rule, search once more with only the cells that keep it (_search_on_desk · ck.ondesk).
            nonrecv = [(str(o["name"]), o["xy_cap"]) for o in pr["fixed"] if not o["must"]]
            print(_PR.marker_clip(clip, pr), flush=True)
        elif objs is not None:
            print(f"{_PR.TAG} {clip} scene {self._SCENE} — no desk · no check", flush=True)
        return pr, pr_must, nonrecv, keepout, fixed_box

    def _clip_index_cache(self, ck):
        """Per yaw: pool grid index, orientation atoms and wrist centre (with the search-origin shift)."""
        a = self.a
        traj, cpiv = ck.traj, ck.cpiv
        # frame index cache (per yaw)
        idx_cache = {}
        cent_cache = {}
        # this clip's origin displacement — one owner (_origin_shift); the desk mask reads the same ck.d0s.
        # pb below is in the robot-base frame (scene axes, after rz), so the volume index takes the **scene**
        # displacement d0b = (d0s, 0) = rz·d0cap; d0cap (capture frame) is for the trajectory (candidate dcap).
        ck.d0s, d0cap = self._origin_shift(cpiv)
        d0b = np.array([ck.d0s[0], ck.d0s[1], 0.0])

        for side, S in self.sides.items():
            P0 = S["P"]
            gis, ais = [], []
            cents = []
            for yaw in self.yaws:
                Ry = Rotation.from_euler("z", yaw, degrees=True)
                P0t, R0t = traj[side]
                Pc = Ry.apply(P0t - cpiv) + cpiv
                Rc = Ry * R0t
                pb = self.rz.apply(Pc + Rc.apply(self.palm_off)) + np.array([0, 0, self.Z_OFF]) - S["base"]
                qb = (self.rz * Rc).as_quat()
                # Look at the top N atoms, as confirm does. With only the nearest 1, a frame's fate hangs
                # on a single atom whose orientation error can be as large as 31° (this was the cause of
                # the fake certificate, and only confirm had been fixed).
                _sim = np.abs(qb @ P0["atoms"].T)
                ais.append(np.argsort(-_sim, axis=1)[:, : a.dp_atoms])
                gis.append(
                    np.round((pb + d0b[None, :] - P0["lo"]) / self.step).astype(int)
                )  # origin displacement applied
                cents.append((pb + d0b[None, :]).mean(0))
            idx_cache[side] = (gis, ais)
            cent_cache[side] = np.stack(cents)
        ck.idx_cache, ck.cent_cache, ck.d0cap = idx_cache, cent_cache, d0cap

    def _origin_shift(self, cpiv):
        """This clip's search-origin shift (scene d0s (2,), capture d0cap (3,)). center-scene off (D0_SCENE None) =
        both 0, so the window is around the clip's own pivot; on = target point − this clip's scene pivot. Read by
        the volume index, the desk mask and the candidate shift (anchors · confirm · exhaustive dcap)."""
        d0s = np.zeros(2)
        d0cap = np.zeros(3)
        if self.D0_SCENE is not None:
            _pv_s = np.array([cpiv[1], -cpiv[0]])  # scene pivot = (cy, -cx)
            d0s = self.D0_SCENE - _pv_s
            d0cap = np.array([-d0s[1], d0s[0], 0.0])  # scene → capture rotation = Rz(+90): (-sy, sx)
        return d0s, d0cap

    def _clip_desk_mask(self, ck):
        """Object view for the desk check (frame-0 mode) and the yaw x grid desk mask."""
        a = self.a
        objs, cpiv = ck.objs, ck.cpiv
        # object view for the desk check — frame0 mode = initial layout only (tail trim keeps the full traj)
        objs_dm = [o[:1].copy() for o in objs] if objs is not None and a.desk_obj_frame0 else objs
        if not VC.desk_check_on(self._SCENE):  # aerial: desk mask + final desk check off (no desk) — after the
            objs_dm = None  # frame-0 view, so --desk-obj-frame0 cannot turn it back on
        desk_m = None
        if objs_dm is not None:
            desk_m = np.zeros((a.n_yaw, self.NBx, self.NBy), bool)
            n_off = n_ko = n_pen = 0  # cells the place rule drops (marker [PLACE_RULE]) · the D24 check (object_desk)
            n_before = 0  # cells left by the point mask + the render statics, before the place rule
            # A cell's scene displacement = window offset (sgx, sgy) + the origin shift d0s (center-scene; 0 when off,
            # and x + 0.0 == x, so the mask is bit-identical then). Without d0s the mask would test a window that
            # is not the one searched.
            gx, gy = self.sgx + ck.d0s[0], self.sgy + ck.d0s[1]
            for k, yaw in enumerate(self.yaws):
                Ryw = Rotation.from_euler("z", yaw, degrees=True).as_matrix()[:2, :2]
                mins = np.array([1e9, 1e9])
                maxs = -mins
                for oxy in objs_dm:
                    pp = (Ryw @ (oxy - cpiv[:2]).T).T + cpiv[:2]
                    ww = pp @ self.RZW.T
                    mins = np.minimum(mins, ww.min(0))
                    maxs = np.maximum(maxs, ww.max(0))
                okx = (gx >= self.DXb[0] - mins[0]) & (gx <= self.DXb[1] - maxs[0])
                oky = (gy >= self.DYb[0] - mins[1]) & (gy <= self.DYb[1] - maxs[1])
                desk_m[k] = okx[:, None] & oky[None, :]
                for sxy in getattr(ck, "statics", ()):  # same rule as desk_ok_final (none → bit-identical)
                    ww = ((Ryw @ (sxy - cpiv[:2]).T).T + cpiv[:2]) @ self.RZW.T
                    desk_m[k] &= _SD.grid_overhang_ok(ww, gx, gy, box=self.DESK_XY)
                n_before += int(desk_m[k].sum())
                for _who, oxy in getattr(ck, "pr_must", ()):  # the place rule: what must stand — its outline
                    ww = ((Ryw @ (oxy - cpiv[:2]).T).T + cpiv[:2]) @ self.RZW.T  # off the robot's spot · its
                    ko = _SF.grid_keepout_ok(ww, gx, gy, ck.keepout)  # area on the desk top (the
                    on = _SD.grid_overhang_ok(ww, gx, gy, box=ck.fixed_box)  # final check decides: _PR.hit)
                    n_ko += int((desk_m[k] & ~ko).sum())
                    n_off += int((desk_m[k] & ko & ~on).sum())
                    desk_m[k] &= ko & on
                if ck.obj_deep is not None and len(ck.obj_deep):  # rule D24: no deep vertex over the desk
                    ww = ((Ryw @ (ck.obj_deep - cpiv[:2]).T).T + cpiv[:2]) @ self.RZW.T
                    ok = _OD.grid_off_box(ww, gx, gy, self.DESK_XY)
                    n_pen += int((desk_m[k] & ~ok).sum())
                    desk_m[k] &= ok
                if getattr(ck, "ondesk", False):  # second search: grazed extras must keep both too
                    for _n, oxy in ck.nonrecv:
                        ww = ((Ryw @ (oxy - cpiv[:2]).T).T + cpiv[:2]) @ self.RZW.T
                        desk_m[k] &= _SF.grid_keepout_ok(ww, gx, gy, ck.keepout) & _SD.grid_overhang_ok(
                            ww, gx, gy, box=ck.fixed_box
                        )
            if getattr(ck, "pr", None) is not None and not getattr(ck, "ondesk", False):
                left = int(desk_m.sum())
                ck.pr_cut, ck.pr_left = n_off + n_ko, left
                print(
                    f"{_PR.TAG} {ck.clip} cells {n_before} → {left}/{desk_m.size} · cut by the place rule "
                    f"{n_off + n_ko} ({_PR.OFF_DESK} n={n_off} {_PR.KEEPOUT} n={n_ko}) · checked outlines "
                    f"{len(ck.pr_must)} · cut by D24 {_OD.DROP_WORD} n={n_pen}",
                    flush=True,
                )
                if not left and n_off + n_ko + n_pen:
                    ck.place_none = f"{_PR.OFF_DESK} n={n_off} {_PR.KEEPOUT} n={n_ko} {_OD.DROP_WORD} n={n_pen}"
        ck.objs_dm, ck.desk_m = objs_dm, desk_m
