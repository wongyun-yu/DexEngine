"""Judge boot, IK pools: per side the reach-map pool (positions, joint seeds, clearance), its desk filter box and the
loaded arm IK objects. Part of the Judge class (judge/core.py composes the mixins).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.ndimage import maximum_filter

from dexengine.placement import arm_ik as AIP
from dexengine.placement.collision import core as SCL
from dexengine.placement.coords import core as VC
from dexengine.placement.judge import guard_pool as _GP
from dexengine.placement.reach_map import batch_ik as OPG
from dexengine.placement.reach_map import clearance as CLR
from dexengine.placement.reach_map import store as PV
from dexengine.placement.retime import stretch as VST


class PoolsMixin:
    """Boot step `_boot_pools` and the per-side pool loader (`_load_side`, `_filt_xy`)."""

    def _boot_pools(self):
        """Per side: IK pool, volume maps, valley joint values and ArmIK (_load_side); active-side table and
        penetration contexts."""
        a = self.a
        self.sides = {}
        self.BUDGET = {}
        self.POOL_IDS = {}  # side → the verified map's id (store.pool_id) — written into every plan line
        for side in ("right", "left"):
            self.sides[side] = self._load_side(side)
        # rebind `sides` per clip to the subset of active sides (both = always the full set)
        self.ALL_SIDES = dict(self.sides)
        self.AS_CUR = {
            "active": ("right", "left"),
            "idle": (),
        }  # this clip's active/idle sides — plan + cross guard read it
        self.step = self.sides["right"]["P"]["step"]
        self.PEN_CTX = {s2: SCL.build_arm_ctx(a.arm, a.hand, s2) for s2 in ("right", "left")}
        # The deciding desk ruler: PEN_CTX is the search proxy (arm_proxy.COLL_PROXY), so every
        # pruning on it is a filter and must be looser than the decision — depth threshold + FILT_SLACK_MM and the
        # footprint shrunk by it (_filt_xy), ε owned by collision.proxy_desk_slack_m (the capsule table vs its hulls).
        # The last penetration verdict inside dp_accept (serial step_mult ≤ 1 check · batch _tfm_post_pen) is a
        # decision: it runs on DESK_CTX = the original mesh (GUARD_KIND), like the guard. Marker [DESK_RULER].
        with SCL.AP.mesh_ruler():
            self.DESK_CTX = {s2: SCL.guard_ctx(self.PEN_CTX[s2]) for s2 in self.PEN_CTX}
        if any(c["proxy"] != SCL.AP.GUARD_KIND for c in self.DESK_CTX.values()):
            raise SystemExit(
                f"[DESK_RULER] ✗ the deciding ruler is not {SCL.AP.GUARD_KIND} — "
                f"{ {s2: c['proxy'] for s2, c in self.DESK_CTX.items()} }"
            )
        self.FILT_SLACK_MM = 1000.0 * max(SCL.proxy_desk_slack_m(self.PEN_CTX[s2]) for s2 in self.PEN_CTX)
        print(
            f"[DESK_RULER] decision={SCL.AP.GUARD_KIND} filter={self.PEN_CTX['right']['proxy']} "
            f"filter slack=+{self.FILT_SLACK_MM:.1f}mm (depth threshold + · desk box −)",
            flush=True,
        )
        _GP.procs()  # guard prefetch size — marker [GUARD_POOL]

    def _filt_xy(self):
        """The search filters' desk footprint: the deciding footprint (DESK_XY — the dataset's desk top) shrunk by
        FILT_SLACK_MM on every edge — a proxy body overhanging into the box by ≤ ε above a hull that passes low beside
        the desk must not be pruned (filter ≤ judge)."""
        e = self.FILT_SLACK_MM / 1000.0
        (x0, x1), (y0, y1) = self.DESK_XY
        return (x0 + e, x1 - e), (y0 + e, y1 - e)

    def _verified_pool(self, side, bx, wd_inputs) -> Path:
        """This side's map folder after its identity and input check; records its id in POOL_IDS (plan lines)."""
        a = self.a
        _pdir = Path(a.pool_root) / f"{self.combo}_{side}"
        if not (_pdir / "meta.npz").exists():
            raise RuntimeError(
                f"[judge] no IK pool: {_pdir}\n"
                f"  The pool root comes from `python -m dexengine.core.paths --what pool_root`; a combo's pool is "
                f"built by `bash scripts/build_reach_map.sh <arm> <hand>`.\n"
                f"  To judge another posture, give the pool built for it with `--pool-root`."
            )
        # Identity check (arm · hand · side · base · palm translation · wrist inputs) — owner store.verify (the same
        # function as the coarse path and the pool tools). Marker [POOL_ID] ([POOL] for a locked input).
        PV.verify(
            _pdir,
            arm=a.arm,
            hand=a.hand,
            side=side,
            base_xyz=bx,
            palm_off=self.palm_off,
            wrist_dir=wd_inputs,
            who="judge",
        )
        self.POOL_IDS[side] = PV.pool_id(_pdir)
        return _pdir

    def _load_side(self, side):
        """One side's pool: identity check, inclusion (loose) / order (strict) / margin maps per dz layer, valley
        joint values, validity and clearance, ArmIK."""
        a = self.a
        urdf, ee, jn, bx, wd = AIP.resolve_combo(a.arm, a.hand, side)
        wd_inputs = wd  # the map builder's input set (store module doc 'Inputs'); the shadow is this run's cache only
        if a.wrist_root:  # f0 shadow wrist cache
            wd = Path(a.wrist_root) / f"wrist_poses_{side}"
        model = pin.buildModelFromUrdf(str(urdf))
        self.BUDGET[side] = (
            np.asarray(model.velocityLimit)[[model.joints[model.getJointId(n)].idx_v for n in jn]].astype(np.float64)
            * VST.FRAME_DT
        )
        _pdir = self._verified_pool(side, bx, wd_inputs)
        # includes the clearance sidecar's ruler / arm-shape check ([CLRCAP_ID]) and its desk top ([CLRCAP_TOP])
        P = PV.load_pool(_pdir, a.arm, desk_top=VC.desk_top_for(a.dataset))
        step = P["step"]
        # aerial scene: no desk, so -1e9 keeps the pool clearance threshold (dr = desk_rel − dz) and the confirm
        # clr_thr from binding at all (volume = reachability only).
        desk_rel = (self.DESK - np.asarray(bx)[2]) if VC.desk_check_on(self._SCENE) else -1e9
        OKp = {}
        OKsp = {}
        MGsp = {}  # dz → strict margin map (padded)
        # ── The ruler for desk clearance ─────────────────────
        # The pool's `vclr_arm` is a single **link-origin z**; the judge measures the collider surface. The origin
        # ruler is optimistic and orders the search unlike the judge, so the clearance is read only from the sidecar
        # measured with **the judge's own desk ruler** (`reach_map.clearance`, identity checked in load_pool).
        # It is the desk component only — the one quantity a dz layer shifts exactly (CLR.desk_bar); the torso moves
        # with the robot and is left to the search scalar and the exact guard. The loose (inclusion) map stays looser
        # than the judge: filter ≤ judge.
        _VCA = P.get("VCAC")
        if _VCA is None:
            raise RuntimeError(
                f"[judge] {side}: no clearance sidecar (clrcap_z*.npz) — build it with the judge's own ruler first: "
                f"`python -m dexengine.placement.reach_map.clearance --pool-root {a.pool_root} --arm {a.arm} "
                f"--hand {a.hand}`"
            )
        _lm = a.arm_margin_cap
        if _lm > -SCL.DESK_SURF_PEN_MM / 1000.0:
            raise RuntimeError(
                f"[judge] --arm-margin-cap {_lm} > −DESK_SURF_PEN_MM — the loose map would be tighter "
                f"than the judge and lose solutions (filter ≤ judge)"
            )
        self._dz_maps(desk_rel, step, P, _lm, _VCA, a, OKsp, MGsp, OKp)
        # For B (continuous judging): valley joint values + valid bits (same conditions as stage_dp)
        pdV, K, VQfull, VOKfull, VCLRfull = self._valley_shards(a, side, P)
        FB = FV = None
        gfp = pdV / "grr_field.npz"
        if gfp.exists():
            gf = np.load(gfp)
            FB, FV = gf["boundary"], gf["valid"]
        return {
            "P": P,
            "base": np.asarray(bx),
            "wd": Path(wd),
            "OKp": OKp,
            "OKsp": OKsp,
            "MGsp": MGsp,
            "VQ": VQfull,
            "VOK": VOKfull,
            "VCLR": VCLRfull,
            "desk_rel": desk_rel,
            "K": K,
            "FB": FB,
            "FV": FV,
            "ik": OPG.ArmIK(side, model, urdf, ee, jn),
        }

    def _dz_maps(self, desk_rel, step, P, _lm, _VCA, a, OKsp, MGsp, OKp):
        """Per desk-offset dz: the loose inclusion map (dilated), the strict order map and the strict margin map, padded
        by the window; fills OKp, OKsp, MGsp."""
        for dzi in self.DZ:
            dr = CLR.desk_bar(desk_rel, dzi * step, 0.0)
            # The threshold must be **the same value as confirm** (0.03/20°). A stricter volume
            # threshold discards in advance placements that confirm would pass — and exhaustive
            # search only sweeps inside the volume, so that difference is never examined.
            OK = (
                (P["VPE"] < 0.03)
                & (P["VRE"] < 20.0)
                & (CLR.desk_bar(desk_rel, dzi * step, _lm) < _VCA)
                & (P["VCH"] > dr + a.hand_margin)
            ).any(-1)
            # Strict map: **no dilation, same accuracy criterion as the judge**.
            # The default OK is an optimistic indicator — VPE<30mm, VRE<20°, plus a 3×3×3 dilation —
            # so "100% coverage" does not mean "passes the judge". Membership uses the optimistic
            # map (so no solution is dropped), **the search order uses the strict map** (try first
            # what will actually pass). The strict map's arm clearance uses **the same ruler as the
            # judge's penetration threshold**. The inclusion (loose) map stays at -5cm so no
            # solution is lost; only the ordering (strict) map is aligned to -DESK_SURF_PEN_MM — with the
            # -5cm ruler, dz=-4 cells rank at 100% on top, are tried first and all die of early
            # penetration, while the dz=-2 cells that are 100% by the judge's own criterion (2cm) do
            # not even make it inside the cap.
            _sm = -SCL.DESK_SURF_PEN_MM / 1000.0
            okK = (
                (P["VPE"] < 0.010)
                & (P["VRE"] < 8.0)
                & (CLR.desk_bar(desk_rel, dzi * step, _sm) < _VCA)
                & (P["VCH"] > dr + a.hand_margin)
            )
            OKs = okK.any(-1)
            # 2nd sort key: the **capsule clearance margin** of strict-OK poses (same ruler as
            # the judge) — key 1 (strict coverage) saturates at 100% across thousands of cells
            # and loses all discriminating power (measured: the winner buried at rank 127~766).
            # Cell value = per-atom max-K margin.
            mgK = (
                (np.clip(_VCA.astype(np.float32) - CLR.desk_bar(desk_rel, dzi * step, _sm), 0, None) * okK)
                .max(-1)
                .astype(np.float16)
            )
            OK = maximum_filter(OK.astype(np.uint8), size=(3, 3, 3, 1)) > 0
            Ops = np.zeros((P["nx"] + 2 * self.Wx, P["ny"] + 2 * self.Wy, P["nz"], P["A"]), bool)
            Ops[self.Wx : -self.Wx, self.Wy : -self.Wy] = OKs
            OKsp[dzi] = Ops
            Mps = np.zeros((P["nx"] + 2 * self.Wx, P["ny"] + 2 * self.Wy, P["nz"], P["A"]), np.float16)
            Mps[self.Wx : -self.Wx, self.Wy : -self.Wy] = mgK
            MGsp[dzi] = Mps
            Op = np.zeros((P["nx"] + 2 * self.Wx, P["ny"] + 2 * self.Wy, P["nz"], P["A"]), bool)
            Op[self.Wx : -self.Wx, self.Wy : -self.Wy] = OK
            OKp[dzi] = Op

    def _valley_shards(self, a, side, P):
        """The valley joints, validity and desk clearance of every map cell, read shard by shard."""
        pdV = Path(a.pool_root) / f"{self.combo}_{side}"
        K = np.load(pdV / "shard_z000.npz")["vpe"].shape[-1]
        VQfull = np.empty((P["nx"], P["ny"], P["nz"], P["A"], K, 7), np.float16)
        VOKfull = np.empty((P["nx"], P["ny"], P["nz"], P["A"], K), bool)
        # Clearance is stored **as a value** and dz is applied at lookup time. A bool baked in
        # advance against dz=0 would judge a placement that lowers the robot by 7cm
        # (robot_z_off=−0.07) at dz=0 — that is, **pass placements that go through the desk**.
        VCLRfull = np.empty((P["nx"], P["ny"], P["nz"], P["A"], K), np.float16)
        for kz in range(P["nz"]):
            shv = np.load(pdV / f"shard_z{kz:03d}.npz")
            VQfull[:, :, kz] = shv["vq"].reshape(P["nx"], P["ny"], P["A"], K, 7)
            VOKfull[:, :, kz] = (shv["vpe"].reshape(P["nx"], P["ny"], P["A"], K) < 0.03) & (
                shv["vre"].reshape(P["nx"], P["ny"], P["A"], K) < 20.0
            )
            VCLRfull[:, :, kz] = np.minimum(
                shv["vclr_arm"].reshape(P["nx"], P["ny"], P["A"], K),
                shv["vclr_hand"].reshape(P["nx"], P["ny"], P["A"], K),
            )
        return pdV, K, VQfull, VOKfull, VCLRfull
