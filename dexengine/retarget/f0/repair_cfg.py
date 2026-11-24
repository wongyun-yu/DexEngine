"""f0 repair, configuration layer: the per-hand facts (HandCfg and the active HC), the per-side budget, the solve
contract (objective, NM options, solution pick), status words, finger limits and the main-object folders."""

from __future__ import annotations

import functools
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from dexengine.core.paths import ARM_IDS, HAND_IDS, data_layout, layer_path, main_root, split_combo
from dexengine.core.paths import main_root as _data_root
from dexengine.core.repo import ROBOT_ASSETS_DIR as ROBOTS
from dexengine.data import rules as _RR
from dexengine.retarget.f0 import phrases as PH

# The **dataset qpos layout** of a hand with coupled joints. It lives here because it is not the
# robot spec (the registry) but the order in which the data carries the joints. Attaching a new
# coupled hand adds one line here — `<hand_id>: (full right_* joint order, independent joints)` —
# **and** a coupling expansion in HandCfg.expand. Empty: every registered hand is fully
# actuated.
QPOS_LAYOUT: dict[str, tuple[list[str], list[str]]] = {}


_HS_CACHE = {}


_AS_CACHE = {}


def _canon_variant(arm: str) -> str:
    """Canonical asset-variant suffix of that arm — the value has exactly one owner,
    dexengine.core.arms.ARM_CANON_GEN (no string literal is written here)."""
    if "m" not in _AS_CACHE:
        from dexengine.core import arms  # noqa: PLC0415

        _AS_CACHE["m"] = arms
    m = _AS_CACHE["m"]
    fn = getattr(m, "canon_variant", None)
    if fn is not None:
        return fn(arm) or ""
    g = getattr(m, "ARM_CANON_GEN", {}).get(arm)
    return getattr(g, "variant", "") or ""


def _hand_spec(hand: str):
    """Read the hand spec from the registry — the values are not restated here."""
    if not _HS_CACHE:
        from dexengine.core import hands as m  # noqa: PLC0415

        _HS_CACHE.update({k: v for k, v in m.HAND_REGISTRY.items() if v is not None})
    if hand not in _HS_CACHE:
        raise KeyError(f"unregistered hand: {hand!r} (registered: {sorted(_HS_CACHE)})")
    return _HS_CACHE[hand]


# Δq layout: [wrist dx dy dz (m), wrist wx wy wz (axis-angle rad), n_indep fingers (rad)]
# Weights: thumb+index cheapest < other fingers < wrist rot < wrist pos.
W_WRIST = np.array(
    [
        60.0,
        60.0,
        60.0,  # wrist translation: 1 cm costs like 0.6 rad finger
        8.0,
        8.0,
        8.0,
    ]
)  # wrist rotation


LAMBDA = 1000.0


#: The URDF the f0 repair reads for SharpA (the one place that names it — `f0_sharpa_urdf`): the bytes the original
#: f0 read (the merged rby1 + SharpA URDF of that generation — torso_3 pitch -0.305, the head camera mesh) with only
#: each `<mesh filename>` re-pointed to the file of equal bytes in this tree. It is not the canonical body
#: (`full_robot_gen/<arm>_sharpa_wave<variant>`): f0 builds a pinocchio model of the whole URDF and expresses the hand
#: links relative to the palm, which cancels the arm chain in algebra but not in floating point — measured: changing
#: torso_3 alone moves f0 output (grab qpos 3.7e-5 rad · wrist quaternion 1.4e-4), and these bytes give the original
#: arrays bit for bit. Kept apart from `full_robot_gen/` so no folder scan or public rename sees a second SharpA body.
F0_SHARPA_URDF = "f0/{arm}_sharpa_wave_f0.urdf"


def f0_sharpa_urdf(arm: str) -> Path:
    """The SharpA f0 URDF for `arm` (F0_SHARPA_URDF). A missing file, or any `<mesh filename>` in it that does not
    resolve to a file, stops the run — a skipped mesh would silently drop a collision hull."""
    import re  # noqa: PLC0415

    from dexengine.core.paths import urdf_mesh  # noqa: PLC0415

    p = ROBOTS / F0_SHARPA_URDF.format(arm=arm)
    if not p.is_file():
        raise SystemExit(f"[frame0_repair] ✗ SharpA f0 URDF missing: {p} (F0_SHARPA_URDF) — no fallback body")
    missing = [f for f in re.findall(r'filename="([^"]+)"', p.read_text()) if not urdf_mesh(p, f).is_file()]
    if missing:
        raise SystemExit(f"[frame0_repair] ✗ {p.name}: {len(missing)} mesh files missing, first {missing[:3]}")
    return p


def finger_of_joints(urdf: Path, joints: list[str], tips: tuple[str, ...]) -> dict[str, int]:
    """Each joint's finger: the index in `tips` (HandSpec.tip_body_names_right, thumb..pinky) of the one fingertip
    link in the subtree below the joint's child link. A joint missing from the URDF, or reaching no fingertip or more
    than one, dies — the rule holds for any hand, whatever its joints are called."""
    root = ET.parse(urdf).getroot()
    child = {j.get("name"): j.find("child").get("link") for j in root.findall("joint")}
    below: dict[str, list[str]] = {}
    for j in root.findall("joint"):
        below.setdefault(j.find("parent").get("link"), []).append(j.find("child").get("link"))
    out, bad = {}, {}
    for name in joints:
        if name not in child:
            raise SystemExit(f"[f0] joint {name} is not in {urdf}")
        sub, stack = set(), [child[name]]
        while stack:
            link = stack.pop()
            sub.add(link)
            stack.extend(below.get(link, ()))
        hit = [k for k, tip in enumerate(tips) if tip in sub]
        if len(hit) == 1:
            out[name] = hit[0]
        else:
            bad[name] = [tips[k] for k in hit]
    if bad:
        raise SystemExit(f"[f0] joints that reach no fingertip or several {bad} — fingertips {list(tips)} ({urdf})")
    return out


class HandCfg:
    """One set of per-hand facts — the seam that keeps frame0 repair from baking a hand name into code.

    WHY: f0 repairs the trees of hands with different widths — sharpa (width 28 ·
    22 independent) and the like — so every per-hand value (merged URDF, joint names, coupling ratios,
    qpos width) comes from here instead of global constants. Joint
    names and widths are read from HandSpec (core.hands) — the same fact is never kept in two copies.
    (No hand has its own URDF, joint spelling, palm frame or data root here: all come from
    HandSpec.)
    """

    def __init__(self, hand: str, arm: str):
        # arm: used only to pick the combo URDF (the sharpa_wave branch) — never hard-coded into the path.
        # Measured: the **hand subtrees of the two arms' URDFs are exactly the same** — max
        # FK difference relative to the palm 7.8e-16 m, the 54 hand collision meshes md5-identical
        # in content, 0 arm links caught by the link regex. Even so, bypassing the resolver is itself
        # a trap, so it is a seam.
        self.arm = arm
        self.name = hand
        # The hand's facts (DOF, velocity limit, spec directory) **come from the registry** —
        # numbers written inside a per-hand branch leave that branch alone to rot.
        spec = _hand_spec(hand)
        self.H = int(spec.hand_qpos_dim)
        self.spec_dir = spec.retarget_dirname or spec.name
        # Pad finger speed = the sim's applied limit (core.hands.finger_velocity_vector — the value retiming stretches
        # to and the gate measures against), slowest joint, with the retiming's safety factor (stretch.SAFETY).
        from dexengine.core import hands as _hands  # noqa: PLC0415
        from dexengine.placement.retime import stretch as _VST

        vel = _hands.finger_velocity_vector(spec)
        self.f_cap_rad_s = float(min(vel)) / _VST.SAFETY
        if QPOS_LAYOUT.get(hand):
            # Hand with coupled joints: the dataset qpos carries **every joint** (H of them) and
            # only some of them are independently driven. That arrangement is not the robot spec
            # but the **dataset layout**, so it is not in the registry — it is written per hand
            # here in QPOS_LAYOUT.
            order, indep = QPOS_LAYOUT[hand]
            self.order = list(order)  # full right_* joint names
            self.indep_idx = [self.order.index(f) for f in indep]
            self._joint_full = list(order)
            self._palm = spec.urdf_palm_frame_right
        elif getattr(spec, "coupled_joint_names_right", ()):
            # Coupled joints exist but there is no QPOS_LAYOUT entry — treating the hand as fully
            # actuated rolls on silently with the joint count (H) and the order both wrong. An
            # unknown input dies loudly.
            raise SystemExit(
                f"[f0] hand '{hand}' has {len(spec.coupled_joint_names_right)} coupled joints but "
                f"repair.QPOS_LAYOUT has no dataset qpos layout for it — add one line to QPOS_LAYOUT"
            )
        else:
            # Fully actuated (0 coupled): the joint names are the order.
            self._joint_full = list(spec.finger_joint_names_right)  # the full right_* names
            self.order = self._joint_full
            self.indep_idx = list(range(len(self._joint_full)))
            self._palm = spec.urdf_palm_frame_right
        self.n_f = len(self.order)
        self.n_indep = len(self.indep_idx)
        self.nd = 6 + self.n_indep
        self._tips = tuple(spec.tip_body_names_right)

    @functools.cached_property
    def w_dq(self) -> np.ndarray:
        """Step weights: the wrist's (W_WRIST), then 1.0 for a thumb or index joint and 2.0 for any other finger joint.
        A joint's finger comes from the hand's URDF tree (finger_of_joints), never from its name. Read on first use:
        the module-level HC is built at import, before any URDF is needed."""
        names = [self.order[i] for i in self.indep_idx]
        finger = finger_of_joints(self.urdf("right"), names, self._tips)
        return np.concatenate([W_WRIST, np.asarray([1.0 if finger[n] < 2 else 2.0 for n in names])])

    # names / paths
    def joint_name(self, side: str, f: str) -> str:
        return f if side == "right" else f.replace("right_", "left_", 1)

    def urdf(self, side: str) -> Path:
        if self.name == "sharpa_wave":
            # Use the **generated combo URDF**, not the vendor original. Why: the vendor left/right
            # files are scattered in layout and each resolves package:// its own way, whereas in
            # the generated URDF rebase_meshes has normalised the mesh paths against its own
            # directory, it carries both hand subtrees, and it is a verified file.
            # Arm links are not caught by the {side}_ link regex below (measured: in both URDFs,
            # 0 arm links out of 35 matches).
            # The pose override follows the **same env** as resolve_combo — sourcing canon_env.sh
            # reads the canonical chest00 URDF (no resolver bypass).
            _ov = os.environ.get("DEXENGINE_URDF_OVERRIDE", "")
            if _ov:
                if not Path(_ov).exists():
                    raise SystemExit(f"DEXENGINE_URDF_OVERRIDE path missing: {_ov}")
                return Path(_ov)
            return f0_sharpa_urdf(self.arm)
        # Any other hand: do not branch on the hand name — find it by **rule**.
        # The merged URDF is `full_robot_gen/<arm>_<hand><canon variant>/` and the owner of the
        # variant suffix is dexengine.core.arms (the same rule as resolve_combo). If it is missing, die at
        # once — silently falling back to no variant makes judging and baking run on a different
        # body than the training USD.
        _ov = os.environ.get("DEXENGINE_URDF_OVERRIDE", "")
        if _ov:
            if not Path(_ov).exists():
                raise SystemExit(f"DEXENGINE_URDF_OVERRIDE path missing: {_ov}")
            return Path(_ov)
        _c = f"{self.arm}_{self.name}{_canon_variant(self.arm)}"
        _p = ROBOTS / f"full_robot_gen/{_c}/{_c}.urdf"
        if _p.exists():
            return _p
        # When that arm has no merged URDF: on the measurement that **the hand subtree is
        # independent of the arm** (max hand FK difference between the two arms
        # 7.8e-16 m · hand collision meshes md5-identical · 0 arm links caught by the link regex),
        # use the merged URDF of another arm that carries the same hand. But only when there is
        # **exactly one** candidate — at 0, or at 2 or more, a human has to decide which body it
        # is (no silent choice). f0 uses only the hand subtree, so this substitution is safe.
        _cands = sorted(
            d
            for d in (ROBOTS / "full_robot_gen").iterdir()
            if (d / f"{d.name}.urdf").is_file()
            and split_combo(d.name, ARM_IDS, HAND_IDS, unsplit_ok=True)[1] == self.name
        )  # folder scan
        if len(_cands) == 1:
            _alt = _cands[0] / f"{_cands[0].name}.urdf"
            if not getattr(HandCfg, "_alt_said", False):
                print(
                    f"[frame0_repair] {self.arm}+{self.name} has no merged URDF; using "
                    f"{_cands[0].name}, which carries the same hand (the hand subtree does not depend on the arm). "
                    "f0 uses only the hand.",
                    flush=True,
                )
                HandCfg._alt_said = True
            return _alt
        raise SystemExit(
            f"[frame0_repair] no merged URDF: {_p} · same-hand candidates {[c.name for c in _cands]} — "
            f"build it with dexengine/robots/urdf/merge.py {self.arm}+{self.name} or pass --arm"
        )

    def base_frame(self, side: str) -> str:
        return self._palm if side == "right" else self._palm.replace("right_", "left_", 1)

    def expand(self, indep):
        """Independent joints → all finger joints (coupling expanded).

        Every registered hand is fully actuated (independent == all), so this is the identity.
        A coupled hand needs its own expansion here; none is registered, so
        die rather than pass a
        short vector on as if it were the full one."""
        if self.n_indep != self.n_f:
            raise NotImplementedError(
                f"[frame0_repair] hand '{self.name}' has coupled joints (independent {self.n_indep} < all "
                f"{self.n_f}) but "
                "HandCfg.expand has no coupling expansion"
            )
        return np.asarray(indep, float)

    def data_root(self, dataset: str) -> Path:
        return _data_root() / f"data/{dataset}/trajectories_gen/{self.spec_dir}"

    def wrist_root(self, dataset: str) -> Path:
        return _data_root() / f"precompute/{dataset}/robots/{self.spec_dir}"


# Default to the hand we are actually running (DEXENGINE_HAND, owned by data.rules), not to a
# literal — a literal default would silently pick a fixed hand and its layout
# when f0 runs without --hand, and nothing in the output would say so.
# The arm follows the same rule: building HandCfg from a default arm would fall back to the "exactly one same-hand
# candidate" path whenever `<default arm>_<new hand>` has no merged URDF (and die once there are two). The arm comes
# from DEXENGINE_ARM (datasets, no default).
HC = HandCfg(_RR.ACTIVE_HAND, _RR.active_arm())

#: Modules whose code reads the module global HC. main_steps rebinds HC for every run (--hand/--arm); each reader
#: module imported HC by name, so share_hand_cfg rebinds it in all of them (tests/retarget/test_f0_split.py checks
#: that this tuple equals the modules that read HC).
HC_READERS = (
    "dexengine.retarget.f0.repair_cfg",
    "dexengine.retarget.f0.repair_geom",
    "dexengine.retarget.f0.repair_side",
    "dexengine.retarget.f0.repair_solve",
    "dexengine.retarget.f0.repair_write",
)


def share_hand_cfg(hc) -> None:
    """Make `hc` the HC of every reader module (main_steps calls it right after rebinding its own HC)."""
    import importlib  # noqa: PLC0415

    for name in HC_READERS:
        importlib.import_module(name).HC = hc


CONTACT_BAND = 0.005  # hand point within 5 mm of the object surface counts as touching


CONTACT_MIN_PTS = 3  # ... and a LINK counts as touching at 3+ such points (ignores grazes)


# per-side budget
# This constant is the single owner of the per-side budget. The budget counts evaluations of the objective `n_pen`,
# not wall-clock time (`time.time()`).
#
# Why evaluations: with a wall-clock budget, machine speed and load would change the data (sides on slow cores drop
# out of the repair, giving a different output, while ledgers and completion lines still report success).
#
# Evaluation counts are deterministic: the solver loop (Nelder-Mead) has no RNG (surface samples use `seed=0` and
# `RandomState(0)`, decimation is voxel clustering), and workers are pinned to one BLAS thread. Same input -> same
# evaluation count -> same output, independent of machine, load and parallelism.
#
# Calibration: the canonical 50 clips were run with an unlimited budget to measure per-side `n_evals`; the
# value is the maximum over successfully repaired sides times a margin.
SIDE_BUDGET_EVALS = 45_000
# Calibration measurement (canonical 50 clips, dg5f_s × TACO, one unlimited-budget run, 10 workers):
#   every side repaired · n_evals median 16,418 · p95 23,287 · max 29,351
#   (the maximum: a left side, class=wrist)
# 45,000 = max × 1.53 (= p95 × 1.93 = median × 2.74). The generous side was chosen on purpose: too small and repairs
# silently disappear; too large and sides without a solution only take longer (visibly, in the log). The canonical 50
# have no side without a solution.


# solve contract
# The four values below are the single owner of "how one side is solved". The only solver engine is the GPU
# (`gpu_nm.NMPool`, entry `prejudge.py`); it reads these values and formulas and holds no copies. The NM option names
# and meanings are scipy Nelder-Mead's; NMPool reproduces that arithmetic in order (the self-test `gpu.py --selftest-nm`
# compares bits with scipy).
NM_OPTIONS = {"maxiter": 2500, "xatol": 1e-4, "fatol": 1e-4, "adaptive": True}  # scipy Nelder-Mead options


DEPTH_W = 5e4  # 1 mm of total depth ≈ a movement cost of 50 — depth gives the landscape its slope


STRICT_MARGIN_SCALE = 0.7  # hysteresis: optimise with the check clearance × 0.7 (see _n_pen_strict in repair_side_fns)


def objective_value(cost, cnt, depth):
    """Value NM minimises = movement cost + LAMBDA·penetration count + DEPTH_W·penetration depth (scalars or arrays,
    same operation order)."""
    return cost + LAMBDA * cnt + DEPTH_W * depth


def pick_solution(dqs, oks, cost):
    """Results per initial value (clipped dq, feasible) -> accepted solution (best_x, best_c). The single owner of the
    selection rule; called by the GPU engine (`gpu_engine.GpuEngine.result`).
    Rule: among feasible results (n_pen = 0), the one with strictly lower movement cost, in initial-value order."""
    best_x, best_c = None, np.inf
    for dq, ok in zip(dqs, oks, strict=True):
        if ok and cost(dq) < best_c:
            best_c, best_x = cost(dq), dq
    return best_x, best_c


class SolveRequest:
    """One solve request from `_side_steps` = all initial values of (one side, one attempt, one class).

    The only receiver is the GPU engine (`gpu_engine.GpuEngine`): it batch-evaluates the objective through `geom`,
    collects (dq, feasible) per initial value, picks (`pick_solution`) and adds the evaluations used to the budget meter
    via `add_evals`. The final feasibility verdict is `geom["n_pen"]` (this module's host function). The selection rule,
    clip and cost are the functions this request closes over, so there are no copies."""

    __slots__ = ("tag", "mask", "inits", "clip", "cost", "add_evals", "geom")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def budget_skip_status(budget: int, used: int) -> str:
    """Status string of a side abandoned on budget; this function is the single owner of that text. The driver
    (`prejudge`) identifies budget skips by the `SKIP_BUDGET` prefix."""
    return f"{PH.SKIP_BUDGET}: side eval budget {budget} exceeded (used {used} · no solution, not repaired)"


def _no_solution_status(rep: dict, budget: int, used: int) -> str:
    """Status of a side without a solution. A side cut off by the budget is not called a 'hard constraint violation':
    the cause is the budget we set, not the data."""
    if rep.get("budget_exhausted"):
        return budget_skip_status(budget, used)
    if rep.get("verify"):
        return "FAIL: strict check failed (overlap left over all vertices) — " + rep["verify"]
    return f"FAIL: {PH.OVERLAP_LEFT} frame-0 overlap not brought to 0 (hard constraint violated)"


def finger_limits() -> tuple[dict, dict]:
    """(vendor, reference) joint ranges {joint: (lo, hi)} of both hand URDFs f0 reads — reference = the vendor range
    with the retargeting IK's hyperextension cut (wrist_shift.finger_ranges). The f0 cache key reads the difference."""
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    from dexengine.placement.repair import wrist_shift as WS  # noqa: PLC0415 — the reference finger range owner

    vendor = {}
    for side in ("right", "left"):
        for j in ET.parse(str(HC.urdf(side))).getroot().iter("joint"):
            lm = j.find("limit")
            if lm is not None and lm.get("lower") is not None and lm.get("upper") is not None:
                vendor[j.get("name")] = (float(lm.get("lower")), float(lm.get("upper")))
    return vendor, WS.finger_ranges(HC.name, vendor, "bimanual")  # both side URDFs were read


# core
def main_object_dirs(tree: Path, lay, side: str, dataset: str) -> tuple[Path, Path | None]:
    """The **main object** mesh directory (visual.obj) and surface_points directory of this side.

    Base layer (task_info.layout): the one entry in `objects[]` with role=main · side∈{side,both} —
    mesh_dir is a logical name (`data/<ds>/assets/objects/<id>`) that data_layout.object_path turns into the
    physical folder (STORAGE — TACO lives in the retarget processed tree), the surface points are
    `layer_path("meshes", ds)/<obj_id>`. If the column address disagrees with lay.main, raise
    (oracle). Legacy (TACO staging, no objects[]): the basename of `<side>_object_mesh_dir` under
    `layer_path("meshes", ds)` — the same place as the old `data/taco/meshes/<id>`. Since
    `<side>_object_mesh_dir` is a retarget-relative path, it is not used in the base layer.
    """
    with open(tree / "task_info.json") as fh:
        ti = json.load(fh)
    ds_ti = str(ti.get("dataset_name", ""))
    if ds_ti and ds_ti != dataset:
        raise ValueError(
            f"{tree}: task_info.dataset_name={ds_ti!r} ≠ --dataset {dataset!r} "
            "— the tree and the run conditions diverged"
        )
    if lay.source == "task_info.layout":
        cands = [o for o in lay.objects if o.get("role") == "main" and o.get("side") in (side, "both")]
        if len(cands) != 1:
            raise ValueError(f"{tree}: {side} has {len(cands)} main objects (role=main) — check objects[]")
        o = cands[0]
        if int(o["qpos_start"]) != lay.main[side]:
            raise ValueError(f"{tree}: {side} main-object column {o['qpos_start']} ≠ layout main {lay.main[side]}")
        mesh_dir = main_root() / data_layout.object_path(dataset, o["mesh_dir"])  # logical -> physical (data_layout)
        sp_dir = layer_path("meshes", dataset) / str(o["obj_id"])
        return mesh_dir, sp_dir, (o.get("articulation") or None)
    key = f"{side}_object_mesh_dir"
    if not ti.get(key):
        raise KeyError(f"{tree}: legacy tree without {key}")
    return layer_path("meshes", dataset) / Path(str(ti[key]).rstrip("/")).name, None, None
