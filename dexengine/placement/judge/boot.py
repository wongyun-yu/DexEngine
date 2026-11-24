"""Judge boot: seeds, grid, search policy, GPU batching, outputs and the clip list (IK pools: judge/boot_pools.py)."""

from __future__ import annotations

import random as _pyrandom
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from dexengine.core.arms import ARM_REGISTRY, LIMB_POINTS
from dexengine.core.paths import TREE_SFX, read_clip_list, ref_base_dir
from dexengine.placement import arm_flip as _AF  # arm-flip gate + re-solve (shared with the bake)
from dexengine.placement import arm_ik as AIP
from dexengine.placement import search_policy as _CSP  # per-dataset dz layers and search origin
from dexengine.placement.collision import core as SCL
from dexengine.placement.collision import core_torch as SCT
from dexengine.placement.collision import geom as SCG
from dexengine.placement.coords import core as VC
from dexengine.placement.judge import aug_cells as _AC  # augmentation cells (modes D · L, default off)
from dexengine.placement.judge import aug_cells_canon as _ACC  # owner of the [AUG_CELLS] tag
from dexengine.placement.judge import budget as _JB  # per-clip budget-cut record (file · cell · marker)
from dexengine.placement.judge import clip_time as _CT  # [CLIP_T] per-clip stage seconds
from dexengine.placement.judge import input_record as _JFP  # per-clip judge input record
from dexengine.placement.judge import mps_lane as _MPS  # MPS lane proof and lane VRAM cap
from dexengine.placement.judge import reselect as _RS  # re-selection margin (--place-reselect)
from dexengine.placement.judge import retry as _RTY  # D37 resume table (--exhaust-resume)
from dexengine.placement.reach_map import batch_ik as OPG

DEV, DT = OPG.DEV, OPG.DT
from dexengine.core import log_markers as LM
from dexengine.placement.judge.args import place_strategy, track_ablate_parts
from dexengine.placement.judge.common import _PlanWriter


class BootMixin:
    def __init__(self, a, _F0ROOT, TRAJ_ROOT):
        self.a = a
        self._F0ROOT = _F0ROOT
        self.TRAJ_ROOT = TRAJ_ROOT
        _MPS.assert_lane(torch, DEV)  # before any CUDA work: [JUDGE_MPS] and [VRAM_CAP] lane markers
        self._boot_rng()
        self._boot_grid()
        self._boot_dz_policy()
        self._boot_pools()
        self._boot_gpu_batch()
        self._boot_vram_cap()
        self._boot_nat_pose()
        self._boot_banner()
        self._boot_outputs()
        self._boot_clips()
        self._boot_run_state()
        self._AUG = self._boot_aug_cells()
        self._PS = place_strategy(a)
        self._ABL = track_ablate_parts(a)
        self._TFM_STAT = {"graft_tried": 0, "graft_taken": 0, "ball_tried": 0, "ball_taken": 0} if self._ABL else None

    def _boot_aug_cells(self):
        """Augmentation cells (modes D · L, judge.aug_cells): None when off (marker only). Bundle or chunk given
        while off dies."""
        a = self.a
        if a.aug_cells_mode == "off":
            if a.aug_cells_dir or a.aug_cells_chunk:
                raise SystemExit(
                    f"{_ACC.TAG} ✗ --aug-cells-mode off with --aug-cells-dir/--aug-cells-chunk — an input nobody reads"
                )
            print(f"{_ACC.TAG} mode=off", flush=True)
            return None
        return _AC.AugCells(self, a.aug_cells_mode, a.aug_cells_dir, a.aug_cells_chunk)

    def _boot_rng(self):
        """Deterministic judge randomness: the base seed and the judge's own generator (per clip: _clip_seed,
        _seed_clip)."""
        # ── Making the judge's randomness deterministic ──────────
        # The multistart draws joint seeds with `torch.rand`; on the global default generator (seeded
        # from OS entropy) the same input would not give the same placement, so the stream is seeded.
        #
        # Why one global manual_seed is not enough: clip lists are split across parallel processes, and with one
        # global seed the stream would depend on the order in which clips are consumed. The seed is therefore derived
        # per clip — the stream of clip c = f(base seed, c) — invariant to splits, parallel order and resume.
        #
        # Python's built-in hash() differs from process to process because of PYTHONHASHSEED, so it
        # is not used (it is itself a source of non-determinism). blake2b pins it instead.
        self._JSEED = int(self.a.seed)
        # Generator dedicated to the judge — kept apart from the global default generator, so that a
        # library drawing randomness globally (inside scipy/torch) cannot shift the judge's stream.
        self._JGEN = torch.Generator(device=DEV)

    def _clip_seed(self, clip: str) -> int:
        """(base seed, clip name) → a reproducible 63bit seed. Invariant across processes and machines."""
        import hashlib

        h = hashlib.blake2b(f"{self._JSEED}|{clip}".encode(), digest_size=8)
        return int.from_bytes(h.digest(), "big") & ((1 << 63) - 1)

    def _seed_clip(self, clip: str) -> int:
        """Pin every source of randomness to this clip's seed, right before judging that clip."""
        sd = self._clip_seed(clip)
        self._JGEN.manual_seed(sd)
        torch.manual_seed(sd)  # global torch (CPU) — insurance
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sd)  # global torch (CUDA) — insurance
        np.random.seed(sd & 0xFFFFFFFF)  # numpy global legacy RNG — insurance
        _pyrandom.seed(sd)  # python random — insurance
        return sd

    def _boot_grid(self):
        """Placement grid: frame rotation, 27 neighbours, desk height, palm offset, radius, yaw, origin, dz layers."""
        a = self.a
        self.rz = Rotation.from_euler("z", -90, degrees=True)
        # 27-neighbour offsets — flattens the original 6-deep loop's order (dx3 → dy3 → dz3) as-is
        self.OFFS27 = np.array([(dx, dy, dz) for dx in (0, -1, 1) for dy in (0, -1, 1) for dz in (0, -1, 1)], dtype=int)
        # World desk-top z (a robot-world constant). The pool grid's desk-relative height (desk_rel = DESK − base_z)
        # and the arm-clearance threshold are world quantities, unrelated to the desk plane in capture coordinates
        # (support.desk_z).
        self.DESK = AIP.DESK_HEIGHT
        # Capture (tree) coordinates → world z offset. The clip loop rebinds it per tree through
        # `coords.scene_z_offset(tree)` (base layer: DESK − support.desk_z; TACO staging: DESK). The 5
        # sites computing `pb = rz·(P + R·palm_off) + [0,0,Z_OFF] − base` read this
        # (methods read self.Z_OFF at call time). Called outside a clip it is None and dies loudly — no silent
        # default.
        self.Z_OFF = None
        self.palm_off = AIP.palm_frame_offset(a.hand)
        self.combo = f"{a.arm}_{a.hand}"
        # Placement grid: the desk is a 76cm(x) × 140cm(y) rectangle, so the per-axis radius matches
        # the desk instead of a ±77cm square — 1075 cells instead of 2025 (53%). The region cut away
        # is where the desk mask zeroes things anyway, so no solution is lost.
        self.Wx = a.half_cells_x if a.half_cells_x > 0 else a.half_cells
        self.Wy = a.half_cells_y if a.half_cells_y > 0 else a.half_cells
        self.NBx, self.NBy = 2 * self.Wx + 1, 2 * self.Wy + 1
        # yaw range: 180 measured adopted placements cluster in −95°~+12°, and |yaw|≤90° covers
        # 99.4% — more than half of the 360° is never used. Keeping the count and narrowing only the
        # range gives **resolution 2.5°** instead of 5.0° (the robot stands in front of the desk, so
        # behind it is physically meaningless).
        self.yaws = np.linspace(a.yaw_lo, a.yaw_hi, a.n_yaw, endpoint=False)
        # Search origin (center-scene): resolved per dataset · scene in _boot_dz_policy (the scene mode is known
        # there) from its owner center_scene_policy. None = off = window around the clip's absolute pivot.
        self.D0_SCENE = None
        # Half-cell dz allowed: best error on infeasible frames is 32~77mm ≈ 1~2 dz cells
        # (35mm) — a solution may sit **between** grid points. Only the volume z-shift is rounded to
        # the nearest layer (the inclusion map is a dilated, optimistic indicator and absorbs the
        # rounding error); the judging itself keeps the continuous dzm.
        self.DZ = [float(x) if "." in x else int(x) for x in a.dz_cells.split(",")]

    def _boot_dz_policy(self):
        """dz layers vs the dataset policy (search_policy), scene mode, budget base (x2 for one-arm clips), anchors,
        and the case root of the one-hand rule."""
        a = self.a
        # The layer list is owned by search_policy; the check below compares the argv value with the policy.
        self._SCENE = VC.scene_mode()  # scene-mode owner (env DEXENGINE_SCENE; default table)
        print(f"[SCENE] mode={self._SCENE}", flush=True)
        if not VC.desk_check_on(self._SCENE):  # the desk checks are off for this scene (owner coords.desk_check_on)
            print(
                f"[DESK_CHECK] off scene={self._SCENE} — desk mask and final desk check (desk_ok_final) off "
                f"(independent of --desk-obj-frame0={int(bool(getattr(a, 'desk_obj_frame0', False)))})",
                flush=True,
            )
        _pol = (
            _CSP.AERIAL if self._SCENE == "aerial" else _CSP.DZ_CELLS.get(a.dataset)
        )  # aerial: vs AERIAL, not the table
        print(
            f"[DZ_CELLS] dataset={a.dataset} scene={self._SCENE} cells={a.dz_cells} "
            f"({', '.join(f'{c:+.1f}cm' for c in _CSP.cells_to_cm(a.dz_cells))}) "
            f"policy={_pol} {'match' if _pol == a.dz_cells else 'MISMATCH'}",
            flush=True,
        )
        if _pol != a.dz_cells:  # blocks a silent non-policy layer
            raise SystemExit(f"[DZ_CELLS] dataset={a.dataset} layers {a.dz_cells} ≠ policy {_pol}")
        self._boot_center_scene()
        self._EXH0 = a.exhaust_max  # 1-arm volume ~1.8× → 2× cap per clip ([ACTIVE_SIDES] below resets it each clip)
        self.anchors = [tuple(map(float, s.split(","))) for s in a.anchors.split(";")]
        print(f"[ACTIVE_SIDES] mode={a.active_sides}", flush=True)
        # applied-state marker, printed on and off
        if a.arm_flip:
            print(
                f"[ARM_FLIP_JUDGE] on=1 window={_AF.WINDOW_S:g}s travel={np.rad2deg(_AF.TRAVEL_MAX_RAD):.0f}deg "
                f"excess={_AF.EXCESS_MIN_RAD:g}rad rot_p99={_AF.ROT_P99_MAX_DEG:g}deg err_mm={a.dp_err_mm:g} "
                f"(owner dexengine/placement/arm_flip.py)",
                flush=True,
            )
        else:
            print(
                f"[ARM_FLIP_JUDGE] on=0 — explicitly off (--arm-flip 0) · default = {_AF.JUDGE_GATE_DEFAULT} "
                "(owner dexengine/placement/arm_flip.py) — the judge input record of this run differs",
                flush=True,
            )
        # Owner of `case` for the one-hand rule = the base-layer tree (task_info.json); the f0 shadow is a copy.
        self._CASE_ROOT = ref_base_dir(a.dataset, a.hand, a.emb, a.src_layer) if a.active_sides == "auto" else None

    def _boot_center_scene(self):
        """Search origin (center-scene): write the dataset policy's effective value into a and print the marker.
        From here on a.center_scene is the effective value (normalised text, '' = off); the judge input record
        carries it."""
        a = self.a
        a.center_scene, _src = _CSP.resolve(a.dataset, self._SCENE, a.center_scene)
        print(_CSP.marker(a.dataset, self._SCENE, a.center_scene, _src), flush=True)  # applied-state marker
        # The search origin is a scene-coordinate point (e.g. robot front x shrunken boundary). Rotation stays about
        # Rotation is still about the pivot (centre of hand activity), so only the pivot's landing point moves to
        # the origin — per clip, d0 = target − pivot (_origin_shift).
        self.D0_SCENE = np.array([float(v) for v in a.center_scene.split(",")], float) if a.center_scene else None

    def _boot_gpu_batch(self):
        """GPU batch path: gate 12° dedup and desk penetration as frame x candidate tensors on DEV, and
        _dls → _dls_fast (FK reuse + CUDA graph). Rules, thresholds, order and RNG consumption are those of the numpy
        reference computations. _TC = torch twins of PEN_CTX."""
        self._TC = {s2: SCT.torch_ctx(self.PEN_CTX[s2], DEV) for s2 in self.PEN_CTX}
        with SCL.AP.mesh_ruler():  # _TCD = torch twins of DESK_CTX (the decision of _tfm_post_pen)
            self._TCD = {s2: SCT.torch_ctx(self.DESK_CTX[s2], DEV) for s2 in self.DESK_CTX}
        for _S2 in self.ALL_SIDES.values():
            _S2["ik"]._dls = _S2["ik"]._dls_fast  # every _dls call site (gate · FULL · serial) → graph path (row
            # buckets, functional contract, see batch_ik.ArmIK._dls_fast)
        print(
            f"[GPU_BATCH] on: dev={DEV} gate_dedup=torch pen_each=torch pen_full=torch(float64) "
            f"dls=fk_reuse+cudagraph(min_hits={OPG.ArmIK.GRAPH_MIN_HITS}) xcheck=0",
            flush=True,
        )

    #: Rows of one gate pen_each batch and poses of one FULL post-check batch (transient GPU batches).
    PEN_EACH_CHUNK = 65536
    POST_CHECK_POSES = 200000

    def _boot_vram_cap(self):
        """Release the clip's cached GPU blocks at every clip boundary (clip_release); the _dls graphs share one
        pool per stream (batch_ik.ArmIK._graph_pool). Neither changes a kernel or a shape."""
        self._n_clip_release = 0
        print(f"[VRAM_CAP] release clip_boundary={'on' if DEV != 'cpu' else 'off(cpu)'} graph_pool=shared", flush=True)

    def clip_release(self) -> None:
        """Clip boundary: hand the cache the clip left behind back to the driver (called by main per clip). Live
        tensors and the graph pool stay, so the next clip's output is the same."""
        if DEV == "cpu":
            return
        torch.cuda.empty_cache()
        self._n_clip_release += 1

    def _boot_nat_pose(self):
        """Natural-posture wrist (elbows bent 90°): second sort key of the exhaustive order (--order-nat)."""
        # A geometric prior on the search order: the closer a candidate's wrist-trajectory centre is to the wrist of
        # "both arms bent 90° at the elbow only", the earlier it is tried (centre-outward search). Upper-arm and
        # forearm lengths come from FK at q = 0 between the arm spec's limb joint frames (ArmSpec.limb_joints).
        spec = ARM_REGISTRY[self.a.arm]
        sh, el, wr = (spec.limb_joints[k] - spec.index_base for k in LIMB_POINTS)
        self.W_NAT = {}
        for s2 in self.PEN_CTX:
            _Ts0 = SCG.arm_link_transforms(self.PEN_CTX[s2], np.zeros((1, spec.n_joints)))[0]
            _sh0, _el0, _wr0 = _Ts0[sh, :3, 3], _Ts0[el, :3, 3], _Ts0[wr, :3, 3]
            _up = float(np.linalg.norm(_el0 - _sh0))
            _fo = float(np.linalg.norm(_wr0 - _el0))
            self.W_NAT[s2] = _sh0 + np.array([_fo, 0.0, -_up])

    def _boot_banner(self):
        """Boot markers: base seed, input generation, adoption records not read."""
        a = self.a
        print(
            f"[JUDGE_RECORDS] {self.combo}: earlier adoption records (the placement layer) are not read — "
            "a judgement is a function of its inputs",
            flush=True,
        )
        print(f"[judge:{self.combo}] base seed {self._JSEED} (derived per clip — invariant to list splits)", flush=True)
        print(
            f"[judge:{self.combo}] inputs — wrists {self.sides['right']['wd'].parent} · objects {self.TRAJ_ROOT} "
            f"({'f0-first' if (a.wrist_root or a.traj_root) else 'source layer (no f0)'})",
            flush=True,
        )

    def _boot_outputs(self):
        """Output files: plans (plans2), rejects (noconfirm), judge input record, budget rows — all line-buffered."""
        a = self.a
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        # Line buffered: with a block buffer, a process killed by a timeout loses every unwritten line, and this file
        # is the evidence of completion (stage 2 builds passed.txt from it).
        # held open for the judge run (line-buffered append; search.py closes it at the end)
        self.fpl = open(self.out / f"plans2_{self.combo}.jsonl", "a", buffering=1)  # noqa: SIM115
        self.fpl = _PlanWriter(self.fpl, a, self.AS_CUR, self.POOL_IDS)  # the one writer (see _PlanWriter)
        # folds [CLIP_T] results; held open for the judge run like fpl
        self.fno = _CT.Tap(open(self.out / f"noconfirm_{self.combo}.tsv", "a", buffering=1))  # noqa: SIM115
        # Judge input record (owner judge.input_record): per clip, one line with the inputs this judge saw (setup,
        # case, roles, active sides, contact inputs, base trajectory + ruleset, conditions, budget). The status ledger
        # carries it on; a later check recomputes it on the current base and rules to find clips whose inputs changed.
        # It does not change verdicts. Clips whose inputs cannot be measured were stopped before the first verdict
        # (core.prove_inputs); an `error:` field here means the input changed after that proof.
        self.fjfp = open(self.out / f"{_JFP.FILE_PREFIX}{self.combo}.tsv", "a", buffering=1)  # noqa: SIM115 — same lifetime as fpl
        self._JFP_COND = _JFP.judge_conditions(a, a.dataset, self._SCENE, self._EXH0)
        print(
            LM.judge_fp(
                _JFP.SCHEMA,
                " ".join(f"{k}={self._JFP_COND[k]}" for k in _JFP.RUN_FIELDS)
                + f" scene={self._SCENE} active_sides={a.active_sides} → {self.fjfp.name}",
            ),
            flush=True,
        )
        # Budget record (owner judge.budget): one row per judged clip (clip · cell); the status ledger carries it.
        # It does not change verdicts. Round = which search budget the base budget (before the one-arm x2) is.
        self.fbud = open(self.out / f"{_JB.FILE_PREFIX}{self.combo}.tsv", "a", buffering=1)  # noqa: SIM115 — same lifetime
        self._BUD_ROUND = _JB.round_token(self._EXH0, _JFP._rr().EXHAUST_RETRY)
        self._bud_kind, self._bud_n = None, {}

    def _boot_clips(self):
        """Clip list, source-axis marker, desk box, grid coordinates, clock."""
        a = self.a
        if a.clips:
            self.clips = read_clip_list(a.clips, dataset=a.dataset, label="judge")
        else:
            self.clips = sorted(
                p.stem
                for p in self.sides["right"]["wd"].glob("*.npz")
                if not p.stem.endswith(TREE_SFX) and ".pre_" not in p.stem
            )
        # applied-state marker: where the source axis resolved to
        print(
            f"[JUDGE_SRC] dataset={a.dataset} emb={a.emb} layer={a.src_layer} f0={self._F0ROOT} "
            f"traj_root={self.TRAJ_ROOT} n={len(self.clips)}",
            flush=True,
        )
        self._RESELECT = _RS.load(a.place_reselect, self.clips)  # {clip: margin m}; a stray record clip dies
        # D37 resume table — every judged clip must be in it (no silent path judges a missing clip from cell 0).
        self._EXH_RESUME = None
        if a.exhaust_resume:
            t = _RTY.read_resume(a.exhaust_resume)
            miss = [c for c in self.clips if c not in t]
            if miss or not self.clips:
                raise SystemExit(
                    f"{_RTY.RESUME_TAG} ✗ of {len(self.clips)} judged clips {len(miss)} are not in the table "
                    f"({miss[:2]}) — {a.exhaust_resume}"
                )
            self._EXH_RESUME = t
            st = [t[c][0] for c in self.clips]
            print(
                f"{_RTY.RESUME_TAG} table {a.exhaust_resume} · all {len(self.clips)} judged clips present "
                f"(table {len(t)} rows) · start cell min {min(st)} max {max(st)}",
                flush=True,
            )
        self.RZW = VC.RZW
        # Desk bounds: the desk top of this run's dataset (owner coords.desk_bounds_dataset → dexengine.core.desk) — the
        # object mask's box (edge margin · robot-side shrink) and the arm-vs-desk box (DESK_XY · margin 0). The clip
        # loop re-reads both from each tree and dies when a tree is of another dataset (_load_side).
        self.DXb, self.DYb = VC.desk_bounds_dataset(a.dataset, a.desk_shrink_robot)
        self.DESK_XY = VC.desk_box_for(a.dataset)
        print(VC.desk_size_mark(a.dataset), flush=True)  # applied-state marker
        self.sgx = (np.arange(self.NBx) - self.Wx) * self.step
        self.sgy = (np.arange(self.NBy) - self.Wy) * self.step
        self.t00 = time.time()

    def _boot_run_state(self):
        """Run-wide state: reject tally, confirm / gate paths, adopted joints, best candidate, counters."""
        self.REJ = {}
        self.FULL = {}
        # per-side tracking error p99 (mm) of the last confirm — exclusion radius for branch-and-bound (6)
        self.LASTERR = {}
        self.TMPQ = {}  # joints dp_confirm has just solved (side -> (T,n))
        self.GATEQ = {}  # path the gatekeeper (sparse) assembled (side -> (T2g,n)) — confirm's seed
        # Input frames between two entries of a gatekeeper path (GATEQ and the batched "gateq"): the gate
        # runs on the --stride input thinned again by dp_stride. Read once here, because dp_accept lowers
        # a.dp_stride to 1 around the all-frame confirm that consumes the path.
        self.GATE_STEP = int(self.a.stride * self.a.dp_stride)
        self.LASTQ = {}  # joints that passed confirm (all frames) — stored on adoption. The **all-frame** wrist
        # trajectory, refreshed per clip (used only in the confirm stage)

        # For a clip whose every candidate failed, the noconfirm row keeps the met string of the best candidate (the
        # one that passed the most check stages). Check order: error → stretch → speed ratio → jump → arm flip → desk
        # / torso penetration → swing; a later stage means closer to a pass. The keys are the reason heads of the met
        # strings (data — kept as written). _BEST is cleared per clip. Verdicts do not change.
        self._MISS_RANK = {
            "error": 0,
            "stretch": 1,
            "speed_ratio": 2,
            "jump": 3,
            "arm_flip": 4,
            "desk_pen": 5,
            "torso_pen": 5,
            "swivel": 6,
        }
        self._BEST = {}
        self.n_ok = self.n_no = self.n_chk = 0
        self.clip = None  # the clip being judged (bound by run_clip)
